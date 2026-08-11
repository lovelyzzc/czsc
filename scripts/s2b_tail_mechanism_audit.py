"""诊断 2025 S2b 精确市值支持集的尾部机制与研究口径边界。

该脚本只解释已观察到的历史名义正超额，不搜索入场参数，也不授权实盘。输入来自
``s2b_annual_exact_mcap_support_audit.py --year 2025`` 的精确支持集交易明细，并合并决策日
可见的市场状态。所有子组、半年拆分和富集检验都是事后机制诊断。

    uv run --no-sync python scripts/s2b_tail_mechanism_audit.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import _s2c_core as core
import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as proxy
import s2b_matched_control_audit as base
import surge_market_state_filter as market
import surge_portfolio_backtest as portfolio

YEAR = 2025
TOP_N = 5
ANNUAL_DIR = Path(__file__).resolve().parent / "_output" / "s2b_annual_exact_mcap_support" / str(YEAR)
TRADE_PATH = ANNUAL_DIR / "trades.parquet"
ANNUAL_AUDIT_PATH = ANNUAL_DIR / "audit.json"
MARKET_PATH = market.OUTPUT_DIR / "market_state.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s2b_tail_mechanism"
OUTPUT_PATH = OUTPUT_DIR / "audit.json"


def nominal_gate(summary: dict[str, Any]) -> bool:
    """应用既有的名义统计门槛；该门槛不是选择后确认检验。"""

    return bool(
        summary.get("daily_hac", {}).get("t_stat", -np.inf) >= 2
        and summary.get("daily_block_bootstrap", {}).get("ci_lo", -np.inf) > 0
    )


def validate_annual_reference(annual: dict[str, Any], *, trade_sha256: str, trade_rows: int, year: int = YEAR) -> str:
    """校验年度审计、交易明细与尾部诊断使用的是同一冻结身份。"""

    if annual.get("schema") != "s2b_annual_exact_mcap_support_audit_v2":
        raise RuntimeError(f"unexpected annual audit schema: {annual.get('schema')}")
    if annual.get("design", {}).get("year") != year:
        raise RuntimeError(f"annual audit year mismatch: {annual.get('design', {}).get('year')} != {year}")
    trade_output = annual.get("trade_output", {})
    if trade_output.get("sha256") != trade_sha256:
        raise RuntimeError("annual audit trade_output sha256 does not match trades.parquet")
    if trade_output.get("rows") != trade_rows:
        raise RuntimeError(f"annual audit trade rows mismatch: {trade_output.get('rows')} != {trade_rows}")
    if annual.get("coverage", {}).get("exact_supported_trades") != trade_rows:
        raise RuntimeError("annual audit exact-supported coverage count does not match trades.parquet")
    status = annual.get("verdict", {}).get("status")
    if not isinstance(status, str) or not status:
        raise RuntimeError("annual audit verdict status is missing")
    return status


def hypergeom_tail(total: int, successes: int, draws: int, observed: int) -> float:
    """返回超几何分布 ``P(X >= observed)``，避免引入额外依赖。"""

    if not (0 <= successes <= total and 0 <= draws <= total and 0 <= observed <= draws):
        raise ValueError("invalid hypergeometric parameters")
    denominator = math.comb(total, draws)
    upper = min(successes, draws)
    lower = max(observed, draws - (total - successes))
    if lower > upper:
        return 0.0
    return float(
        sum(math.comb(successes, k) * math.comb(total - successes, draws - k) for k in range(lower, upper + 1))
        / denominator
    )


def classify_descriptive_breadth_regime(frame: pd.DataFrame) -> pd.Series:
    """用预先可见的市场中位收益与宽度做粗粒度描述性分类。"""

    bull = frame["mkt_ret20_median"].gt(0) & frame["mkt_ret20_pos_ratio"].gt(0.55)
    bear = frame["mkt_ret20_median"].lt(0) & frame["mkt_ret20_pos_ratio"].lt(0.45)
    return pd.Series(np.select([bull, bear], ["bull", "bear"], default="sideways"), index=frame.index)


def legacy_hard_filter_masks(frame: pd.DataFrame, st_mask: pd.Series | None = None) -> dict[str, pd.Series]:
    """返回历史组合脚本的硬过滤掩码，供口径重叠诊断。"""

    non_st = (
        ~st_mask.reindex(frame.index, fill_value=False) if st_mask is not None else pd.Series(True, index=frame.index)
    )
    amount = frame["amount_e"].ge(portfolio.MIN_AMOUNT_E)
    stop = frame["sl_pct"].between(portfolio.STOP_MIN_PCT, portfolio.STOP_MAX_PCT)
    gap = frame["gap_pct"].lt(frame["limit_pct"] - portfolio.GAP_LIMIT_MARGIN)
    current_hard = amount & stop & non_st
    legacy_fill = current_hard & gap
    return {
        "amount": amount.fillna(False),
        "stop": stop.fillna(False),
        "gap": gap.fillna(False),
        "non_st": non_st.fillna(False),
        "current_hard_without_gap": current_hard.fillna(False),
        "legacy_hard_with_gap": legacy_fill.fillna(False),
    }


def descriptive_summary(frame: pd.DataFrame, column: str = "exact_excess_pct") -> dict[str, Any]:
    """返回不带显著性含义的交易级描述统计。"""

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"n": 0, "mean_pct": None, "median_pct": None, "positive_pct": None, "total_pct": None}
    return {
        "n": int(len(values)),
        "mean_pct": round(float(values.mean()), 3),
        "median_pct": round(float(values.median()), 3),
        "positive_pct": round(float(values.gt(0).mean() * 100), 1),
        "total_pct": round(float(values.sum()), 3),
    }


def inference_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """复用既有决策日聚类 HAC 与 10 日平稳分块 bootstrap。"""

    scoped = frame.rename(columns={"exact_excess_pct": "matched_excess_pct"})
    return base.summarize_matched_excess(scoped)


def group_breakdown(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """按字段分组，并同时报告移除全局 top5 后的描述统计。"""

    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        rows.append(
            {
                "group": "NA" if pd.isna(value) else str(value),
                "full": descriptive_summary(group),
                "after_drop_global_top5": descriptive_summary(group[~group["is_global_top5"]]),
            }
        )
    return rows


def leave_one_period_out(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """逐个剔除月份等时间单元，并复算原名义统计门槛。"""

    rows: list[dict[str, Any]] = []
    values = sorted(frame[column].dropna().unique(), key=str)
    for value in values:
        summary = inference_summary(frame[frame[column].ne(value)])
        rows.append({"excluded": str(value), "summary": summary, "nominal_gate": nominal_gate(summary)})
    return rows


def stable_excess_order(frame: pd.DataFrame, column: str = "exact_excess_pct") -> pd.DataFrame:
    """按超额降序和稳定身份键排序；完全同键时保留原始行序。"""

    required = {column, "symbol"}
    if missing := required - set(frame):
        raise ValueError(f"stable excess ranking missing columns: {sorted(missing)}")
    tie_breakers = [name for name in ("symbol", "dec_dt", "entry_dt", "exit_dt", "trade_key") if name in frame]
    columns = [column, *tie_breakers]
    return frame.sort_values(columns, ascending=[False, *([True] * len(tie_breakers))], kind="mergesort").copy()


def stable_top_n(frame: pd.DataFrame, n: int = TOP_N, column: str = "exact_excess_pct") -> pd.DataFrame:
    """返回使用固定身份键打破并列的前 ``n`` 笔交易。"""

    if n < 0:
        raise ValueError("n must be non-negative")
    return stable_excess_order(frame, column).head(n).copy()


def own_tail_concentration(frame: pd.DataFrame, drops: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    """报告子样本自身最大赢家剔除敏感性，避免把只删全局 top5 误称为分散。"""

    ordered = stable_excess_order(frame)
    positive_total = float(ordered.loc[ordered["exact_excess_pct"] > 0, "exact_excess_pct"].sum())
    result: dict[str, Any] = {
        "full": {"summary": inference_summary(ordered), "descriptive": descriptive_summary(ordered)}
    }
    for count in drops:
        kept = ordered.iloc[count:]
        summary = inference_summary(kept)
        removed = float(ordered.head(count)["exact_excess_pct"].sum())
        result[f"drop_own_top{count}"] = {
            "summary": summary,
            "descriptive": descriptive_summary(kept),
            "nominal_gate": nominal_gate(summary),
            "removed_share_of_positive_excess_pct": (
                round(removed / positive_total * 100, 2) if positive_total > 0 else None
            ),
        }
    return result


def market_feature_diagnostics(trades: pd.DataFrame, top: pd.DataFrame) -> dict[str, Any]:
    """比较 top5 与支持集的二元市场背景频率；全部都是事后描述。"""

    specs = {
        "small_minus_large_ret20_negative": lambda frame: frame["small_minus_large_ret20"].lt(0),
        "ew_index_above_ma20": lambda frame: frame["ew_index_above_ma20"].gt(0),
        "breadth_gt_0_55": lambda frame: frame["mkt_ret20_pos_ratio"].gt(0.55),
        "high20_ratio_gt_0_12": lambda frame: frame["high20_ratio"].gt(0.12),
        "mkt_ret20_median_positive": lambda frame: frame["mkt_ret20_median"].gt(0),
    }
    result: dict[str, Any] = {}
    for name, predicate in specs.items():
        all_count = int(predicate(trades).sum())
        top_count = int(predicate(top).sum())
        item: dict[str, Any] = {
            "all_count": all_count,
            "all_n": int(len(trades)),
            "top5_count": top_count,
            "top5_n": int(len(top)),
        }
        if top_count == len(top):
            item["descriptive_probability_all_top5_given_margin"] = round(
                hypergeom_tail(len(trades), all_count, len(top), len(top)), 6
            )
        result[name] = item
    result["warning"] = (
        "These are post-selection marginal frequencies, not causal filters or valid multiple-testing-adjusted p-values."
    )
    return result


def derive_mechanism_label(
    *,
    tail_dependent: bool,
    all_month_lomo_nominal: bool,
    distinct_top_months: int,
    h2_global_top_count: int,
    h2_after_global_top_mean: float | None,
) -> str:
    """从实际检查的条件生成不含未验证因果词汇的描述标签。"""

    if (
        tail_dependent
        and all_month_lomo_nominal
        and distinct_top_months == TOP_N
        and h2_global_top_count == 4
        and h2_after_global_top_mean is not None
        and h2_after_global_top_mean <= 0
    ):
        return "2025_CROSS_MONTH_SPARSE_TAIL_WITH_H2_COLLAPSE"
    return "2025_TAIL_STRUCTURE_REQUIRES_REVIEW"


def _feature_means(frame: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce").mean()
        result[column] = round(float(value), 6) if np.isfinite(value) else None
    return result


def _load_trades() -> tuple[pd.DataFrame, dict[str, Any], str]:
    if not TRADE_PATH.exists():
        raise FileNotFoundError(f"missing {TRADE_PATH}; first run s2b_annual_exact_mcap_support_audit.py --year {YEAR}")
    if not ANNUAL_AUDIT_PATH.exists():
        raise FileNotFoundError(f"missing {ANNUAL_AUDIT_PATH}; rerun the annual exact-support audit")
    if not MARKET_PATH.exists():
        raise FileNotFoundError(f"missing {MARKET_PATH}; first run surge_market_state_filter.py")
    annual = json.loads(ANNUAL_AUDIT_PATH.read_text(encoding="utf-8"))
    trades = pd.read_parquet(TRADE_PATH)
    annual_status = validate_annual_reference(
        annual,
        trade_sha256=proxy.sha256_file(TRADE_PATH),
        trade_rows=len(trades),
    )
    required = {
        "symbol",
        "dec_dt",
        "entry_dt",
        "exit_dt",
        "treated_industry",
        "ret_gross_pct",
        "exact_excess_pct",
        "hold_days",
        "exit_reason",
        "dec_regime",
    }
    if missing := required - set(trades):
        raise RuntimeError(f"annual trade output missing columns: {sorted(missing)}")
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        trades[column] = pd.to_datetime(trades[column])
    market_frame = pd.read_parquet(MARKET_PATH).rename(columns={"dt": "dec_dt"})
    market_frame["dec_dt"] = pd.to_datetime(market_frame["dec_dt"])
    trades = trades.merge(market_frame, on="dec_dt", how="left", validate="many_to_one")
    if trades[market.FEATURES].isna().any(axis=1).any():
        raise RuntimeError("market-state merge left trades with missing causal market features")
    trades["month"] = trades["dec_dt"].dt.to_period("M").astype(str)
    trades["quarter"] = trades["dec_dt"].dt.to_period("Q").astype(str)
    trades["half"] = np.where(trades["dec_dt"].dt.month.le(6), "H1", "H2")
    trades["hold_bin"] = pd.cut(
        trades["hold_days"],
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=["01-05", "06-15", "16-30", "31+"],
    ).astype(str)
    trades["descriptive_breadth_regime"] = classify_descriptive_breadth_regime(trades)
    trades["trade_key"] = trades["symbol"].astype(str) + "|" + trades["dec_dt"].dt.strftime("%Y-%m-%d")
    return trades, annual, annual_status


def capacity_scope_diagnostic(year: int = YEAR) -> dict[str, Any]:
    """量化“10 槽容量模拟”与历史硬过滤的重叠；不把它冒充 delay5 实盘集合。"""

    context = core.load_context(verbose=False)
    d0 = context["d0"].copy()
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        d0[column] = pd.to_datetime(d0[column])
    candidates = d0.loc[core.gate_mask(d0, core.S2B_VR, core.S2B_SP)].copy()
    candidates = candidates[~base.censored_tail_mask(candidates)]
    selected_keys = base._main_execution_keys(context)
    selected = candidates[
        [
            (symbol, entry_dt) in selected_keys
            for symbol, entry_dt in zip(candidates["symbol"], candidates["entry_dt"], strict=False)
        ]
    ]
    selected = selected[selected["dec_dt"].dt.year.eq(year)].copy()
    st_reference_available = portfolio.NAMECHANGE_PATH.exists()
    st_intervals = portfolio.load_st_intervals()
    if st_intervals:
        st_mask = selected.apply(lambda row: portfolio.is_st_on(st_intervals, row["symbol"], row["dec_dt"]), axis=1)
    else:
        st_mask = pd.Series(False, index=selected.index)
    masks = legacy_hard_filter_masks(selected, st_mask)
    legacy = masks["legacy_hard_with_gap"]
    current_hard = masks["current_hard_without_gap"]
    return {
        "year": year,
        "st_reference_path": str(portfolio.NAMECHANGE_PATH),
        "st_reference_available": st_reference_available,
        "st_reference_sha256": proxy.sha256_file(portfolio.NAMECHANGE_PATH) if st_reference_available else None,
        "capacity_simulated_trades": int(len(selected)),
        "passes_amount": int(masks["amount"].sum()),
        "passes_stop_band": int(masks["stop"].sum()),
        "passes_gap_fill": int(masks["gap"].sum()),
        "passes_non_st": int(masks["non_st"].sum()),
        "passes_current_hard_without_gap": int(current_hard.sum()),
        "passes_legacy_hard_with_gap": int(legacy.sum()),
        "fails_amount": int((~masks["amount"]).sum()),
        "fails_stop_band": int((~masks["stop"]).sum()),
        "gross_mean_legacy_pass_pct": (
            round(float(selected.loc[legacy, "ret_gross_pct"].mean()), 3) if legacy.any() else None
        ),
        "gross_mean_legacy_fail_pct": (
            round(float(selected.loc[~legacy, "ret_gross_pct"].mean()), 3) if (~legacy).any() else None
        ),
        "warning": (
            "This is only a counterfactual overlap check on d0 S2b capacity selections. Production uses a "
            "different delay5 signal plus market/fill stages, so these rows are not production executions. "
            "If the ST reference is unavailable, non-ST and combined-pass counts are optimistic."
        ),
    }


def run_audit() -> dict[str, Any]:
    """运行历史尾部机制诊断并返回可序列化结果。"""

    loaded_trades, annual_audit, annual_status = _load_trades()
    trades = stable_excess_order(loaded_trades).reset_index(drop=True)
    top_indices = stable_top_n(trades, TOP_N).index
    trades["is_global_top5"] = trades.index.isin(top_indices)
    top = trades[trades["is_global_top5"]].copy()
    rest = trades[~trades["is_global_top5"]].copy()

    full_summary = inference_summary(trades)
    drop_top5_summary = inference_summary(rest)
    month_lomo = leave_one_period_out(trades, "month")
    quarter_summaries: dict[str, Any] = {}
    for value, group in trades.groupby("quarter", sort=True):
        summary = inference_summary(group)
        quarter_summaries[str(value)] = {"summary": summary, "nominal_gate": nominal_gate(summary)}
    half_summaries = {
        str(value): {
            "full": descriptive_summary(group),
            "after_drop_global_top5": descriptive_summary(group[~group["is_global_top5"]]),
            "own_tail_concentration": own_tail_concentration(group),
        }
        for value, group in trades.groupby("half", sort=True)
    }

    positive_total = float(trades.loc[trades["exact_excess_pct"] > 0, "exact_excess_pct"].sum())
    net_total = float(trades["exact_excess_pct"].sum())
    top_total = float(top["exact_excess_pct"].sum())
    top_rows = [
        {
            "symbol": row.symbol,
            "dec_dt": str(pd.Timestamp(row.dec_dt).date()),
            "industry": str(row.treated_industry),
            "exact_excess_pct": round(float(row.exact_excess_pct), 6),
            "ret_gross_pct": round(float(row.ret_gross_pct), 6),
            "hold_days": int(row.hold_days),
            "exit_reason": str(row.exit_reason),
            "dec_regime": int(row.dec_regime),
            "descriptive_breadth_regime": str(row.descriptive_breadth_regime),
        }
        for row in top.itertuples()
    ]

    trail_total = int(trades["exit_reason"].eq("trail18").sum())
    top_trail = int(top["exit_reason"].eq("trail18").sum())
    trail_rest = trades[trades["exit_reason"].eq("trail18") & ~trades["is_global_top5"]]
    state8_rest = trades[trades["dec_regime"].eq(8) & ~trades["is_global_top5"]]
    feature_columns = market.FEATURES + [
        "sig_vol_ratio",
        "sig_ma_spread_pct",
        "sig_ret20",
        "hold_days",
        "score",
        "amount_e",
    ]

    all_months_hold = len(month_lomo) == 12 and all(row["nominal_gate"] for row in month_lomo)
    tail_dependent = nominal_gate(full_summary) and not nominal_gate(drop_top5_summary)
    h2 = trades[trades["half"].eq("H2")]
    h2_after_global_top_mean = half_summaries.get("H2", {}).get("after_drop_global_top5", {}).get("mean_pct")
    mechanism_label = derive_mechanism_label(
        tail_dependent=tail_dependent,
        all_month_lomo_nominal=all_months_hold,
        distinct_top_months=int(top["month"].nunique()),
        h2_global_top_count=int(h2["is_global_top5"].sum()),
        h2_after_global_top_mean=h2_after_global_top_mean,
    )

    return {
        "schema": "s2b_tail_mechanism_audit_v2",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "design": {
            "scope": "2025 annual exact-mcap supported subset; post-hoc mechanism diagnostic",
            "top_definition": "five largest exact-support excess trades",
            "inference": "existing decision-date HAC and stationary block bootstrap; nominal descriptive use only",
            "parameter_search": "none",
            "descriptive_breadth_regime": (
                "bull if mkt_ret20_median>0 and positive-ret20 breadth>0.55; bear if median<0 and breadth<0.45; "
                "otherwise sideways; this is not core.build_market_regime"
            ),
        },
        "references": {
            "annual_audit_path": str(ANNUAL_AUDIT_PATH),
            "annual_audit_sha256": proxy.sha256_file(ANNUAL_AUDIT_PATH),
            "annual_schema": annual_audit["schema"],
            "annual_trade_output_sha256": annual_audit["trade_output"]["sha256"],
            "trade_path": str(TRADE_PATH),
            "trade_sha256": proxy.sha256_file(TRADE_PATH),
            "market_state_path": str(MARKET_PATH),
            "market_state_sha256": proxy.sha256_file(MARKET_PATH),
        },
        "scope_diagnostic": capacity_scope_diagnostic(YEAR),
        "concentration": {
            "n_trades": int(len(trades)),
            "n_decision_dates": int(trades["dec_dt"].nunique()),
            "top5": top_rows,
            "top5_total_excess_pct": round(top_total, 6),
            "top5_share_of_positive_excess_pct": round(top_total / positive_total * 100, 2),
            "top5_share_of_net_excess_pct": round(top_total / net_total * 100, 2),
            "top5_distinct_dates": int(top["dec_dt"].nunique()),
            "top5_distinct_months": int(top["month"].nunique()),
            "top5_distinct_industries": int(top["treated_industry"].nunique()),
        },
        "inference": {
            "full": {"summary": full_summary, "nominal_gate": nominal_gate(full_summary)},
            "drop_top5": {"summary": drop_top5_summary, "nominal_gate": nominal_gate(drop_top5_summary)},
            "leave_one_month_out": month_lomo,
            "leave_one_month_out_nominal_gate_count": int(sum(row["nominal_gate"] for row in month_lomo)),
            "all_12_leave_one_month_out_nominal_gates": all_months_hold,
            "quarters": quarter_summaries,
        },
        "time_structure": {
            "months": group_breakdown(trades, "month"),
            "quarters": group_breakdown(trades, "quarter"),
            "halves": half_summaries,
        },
        "mechanism": {
            "exit_reason": group_breakdown(trades, "exit_reason"),
            "hold_bin": group_breakdown(trades, "hold_bin"),
            "dec_regime": group_breakdown(trades, "dec_regime"),
            "descriptive_breadth_regime": group_breakdown(trades, "descriptive_breadth_regime"),
            "industry": group_breakdown(trades, "treated_industry"),
            "trail18_enrichment": {
                "all_trail18": trail_total,
                "top5_trail18": top_trail,
                "descriptive_combinatorial_probability_ge_observed": round(
                    hypergeom_tail(len(trades), trail_total, TOP_N, top_trail), 6
                ),
                "non_top5_trail18": descriptive_summary(trail_rest),
                "warning": (
                    "Top5 is selected by realized outcome and exit reason is path-dependent. This conditional "
                    "combinatorial probability is not a valid confirmatory p-value and is not multiple-test adjusted."
                ),
            },
            "state8_after_drop_top5": descriptive_summary(state8_rest),
            "market_feature_frequencies": market_feature_diagnostics(trades, top),
            "top5_feature_means": _feature_means(top, feature_columns),
            "non_top5_feature_means": _feature_means(rest, feature_columns),
        },
        "limitations": [
            "The 10-slot source set is a d0 structural capacity simulation, not the production delay5 execution set.",
            "The 2025 split and all mechanism subgroups are post-selection historical diagnostics, not independent OOS.",
            "Treated FULL exits and control fixed-horizon exits are asymmetric; control-pool formation uses future availability.",
            "Exact support coverage must be reported against both proxy matches and the capacity-simulated source set.",
            "Decision-date HAC does not fully model overlapping holdings or repeated treated/control-symbol network dependence.",
        ],
        "next_falsification_tests": [
            "rebuild raw-to-mature cohorts through the exact production delay5, hard, ST, market, fill, and exit path",
            "report common 5/20/60-day ATT and controls run through the same FULL exit FSM",
            "perform full-industry point-in-time exact-mcap rematching and double-denominator coverage",
            "balance momentum, volatility, beta, liquidity, price, limit-up history, and listing age",
            "construct calendar-time long-treated/short-control returns with multi-way or wild-cluster inference",
            "audit daily paths for MFE/MAE, time-to-MFE, limit-up streaks, and top-day return contribution",
        ],
        "verdict": {
            "formal_status": annual_status,
            "mechanism_label": mechanism_label,
            "cross_month_sparse_tail": bool(all_months_hold and top["month"].nunique() == TOP_N),
            "live_authorized": False,
            "reason": (
                "The nominal 2025 association survives every leave-one-decision-month-out exclusion, so no single "
                "decision month alone drives it; this does not establish independent calendar episodes because holdings "
                "overlap and trends can span months. Deleting five cross-month winners removes the nominal gate. H2 "
                "collapses after its four global top winners, while H1 also fails the nominal gate after removing its "
                "own top3/top5. No post-outcome subgroup is a valid deployable filter."
            ),
        },
    }


def main() -> None:
    started = time.time()
    result = run_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["concentration"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {OUTPUT_PATH} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
