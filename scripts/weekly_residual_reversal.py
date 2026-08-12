"""Test a fixed weekly industry/size-neutralized residual-reversal strategy.

The study is frozen in ``weekly_residual_reversal_protocol_2026-08-12.json``.
Only causal signal ranks are built before the development test; 2024+ outcomes
are not opened unless every development gate passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from czsc._native.research import circular_block_bootstrap, newey_west_hac

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "weekly_residual_reversal_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "weekly_residual_reversal"
REPORT_PATH = SCRIPT_DIR / "WEEKLY_RESIDUAL_REVERSAL_2026-08-12.md"
PRIMARY_BPS = 40


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(locator: str) -> Path:
    path = Path(locator).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_and_verify_protocol() -> dict[str, Any]:
    """Load the protocol and fail closed on frozen-input drift."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "weekly_residual_reversal_protocol_v1":
        raise RuntimeError("unexpected weekly reversal protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    for name in ("ranked_surface", "price_panel"):
        record = protocol["inputs"][name]
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    return protocol


def build_blind_surface(protocol: dict[str, Any]) -> pd.DataFrame:
    """Build the causal five-observation return surface without outcome columns."""

    ranked_path = _resolve(protocol["inputs"]["ranked_surface"]["path"])
    panel_path = _resolve(protocol["inputs"]["price_panel"]["path"])
    causal = pl.read_parquet(
        ranked_path,
        columns=["decision_dt", "symbol", "industry_code", "log_free_float_mcap"],
    ).with_columns(pl.col("decision_dt").cast(pl.Datetime("ns")))
    dates = causal.select(pl.col("decision_dt").unique()).to_series().to_list()
    returns = (
        pl.scan_parquet(panel_path)
        .select("symbol", "dt", "close")
        .sort(["symbol", "dt"])
        .with_columns((pl.col("close") / pl.col("close").shift(5).over("symbol") - 1).alias("raw_return_5obs"))
        .filter(pl.col("dt").is_in(dates))
        .select(pl.col("dt").cast(pl.Datetime("ns")).alias("decision_dt"), "symbol", "raw_return_5obs")
        .collect()
    )
    frame = causal.join(returns, on=["decision_dt", "symbol"], how="left").to_pandas()
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"])
    finite = np.isfinite(frame["raw_return_5obs"]) & np.isfinite(frame["log_free_float_mcap"])
    frame = frame[finite & frame["industry_code"].notna()].copy()
    minimum = int(protocol["signal"]["minimum_complete_weekly_universe"])
    counts = frame.groupby("decision_dt", observed=True).size()
    if (counts < minimum).any():
        bad = counts[counts < minimum].head().to_dict()
        raise RuntimeError(f"weekly causal reversal universe below {minimum}: {bad}")
    return frame.sort_values(["decision_dt", "symbol"], kind="mergesort").reset_index(drop=True)


def rank_reversal_signal(surface: pd.DataFrame) -> pd.DataFrame:
    """Winsorize and neutralize each weekly return cross-section deterministically."""

    rows: list[pd.DataFrame] = []
    for decision_dt, week in surface.groupby("decision_dt", sort=True, observed=True):
        week = week.sort_values("symbol", kind="mergesort").copy()
        raw_low, raw_high = week["raw_return_5obs"].quantile([0.01, 0.99])
        mcap_low, mcap_high = week["log_free_float_mcap"].quantile([0.01, 0.99])
        y = week["raw_return_5obs"].clip(raw_low, raw_high).to_numpy(dtype=float)
        mcap = week["log_free_float_mcap"].clip(mcap_low, mcap_high).to_numpy(dtype=float)
        mcap_std = float(np.std(mcap, ddof=0))
        if not np.isfinite(mcap_std) or mcap_std <= 0:
            raise RuntimeError(f"degenerate weekly market-cap scale on {pd.Timestamp(decision_dt).date()}")
        mcap_z = (mcap - float(np.mean(mcap))) / mcap_std
        industries = pd.Categorical(
            week["industry_code"].astype(str), categories=sorted(week["industry_code"].astype(str).unique())
        )
        dummies = pd.get_dummies(industries, drop_first=True, dtype=float).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(week)), mcap_z, dummies])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        week["residual_return_5obs"] = y - design @ beta
        week = week.sort_values(["residual_return_5obs", "symbol"], kind="mergesort")
        week["reversal_rank"] = np.arange(1, len(week) + 1, dtype=np.int64)
        rows.append(week)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["decision_dt", "symbol"]).any():
        raise RuntimeError("blind signal contains duplicate identities")
    return result.sort_values(["decision_dt", "reversal_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def build_memberships(
    ranked_signal: pd.DataFrame,
    target: int = 50,
    retention_rank: int = 75,
    rank_column: str = "reversal_rank",
) -> pd.DataFrame:
    """Build the fixed 50/75 recursive membership path."""

    previous: set[str] = set()
    rows: list[dict[str, Any]] = []
    for decision_dt, week in ranked_signal.groupby("decision_dt", sort=True, observed=True):
        week = week.sort_values([rank_column, "symbol"], kind="mergesort")
        indexed = week.set_index("symbol", drop=False)
        available = set(indexed.index.astype(str))
        retained = {
            symbol
            for symbol in previous
            if symbol in available and int(indexed.at[symbol, rank_column]) <= retention_rank
        }
        fill = [str(symbol) for symbol in week["symbol"] if str(symbol) not in retained][: target - len(retained)]
        current = retained | set(fill)
        if len(current) != target:
            raise RuntimeError(f"reversal membership underfilled on {pd.Timestamp(decision_dt).date()}")
        for symbol in sorted(current, key=lambda item: (int(indexed.at[item, rank_column]), item)):
            rows.append(
                {
                    "decision_dt": pd.Timestamp(decision_dt),
                    "symbol": symbol,
                    rank_column: int(indexed.at[symbol, rank_column]),
                    "residual_return_5obs": float(indexed.at[symbol, "residual_return_5obs"]),
                    "membership_role": "RETAINED" if symbol in retained else "NEW",
                }
            )
        previous = current
    return pd.DataFrame(rows)


def cost_pair(roundtrip_bps: int) -> tuple[float, float]:
    """Return the frozen cost split."""

    if roundtrip_bps == PRIMARY_BPS:
        return 0.0015, 0.0025
    one_way = roundtrip_bps / 20_000
    return one_way, one_way


def build_weekly_outcomes(
    protocol: dict[str, Any],
    memberships: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
    terminal_liquidation: bool,
) -> pd.DataFrame:
    """Join outcomes only after blind ranks/memberships have been fixed."""

    outcome = pd.read_parquet(
        _resolve(protocol["inputs"]["ranked_surface"]["path"]),
        columns=["decision_dt", "symbol", "entry_tradable", "fwd_5d_open_return"],
    )
    outcome["decision_dt"] = pd.to_datetime(outcome["decision_dt"])
    outcome = outcome[outcome["decision_dt"] <= end_date]
    selected = memberships[memberships["decision_dt"] <= end_date].merge(
        outcome, on=["decision_dt", "symbol"], how="left", validate="one_to_one"
    )
    benchmark = (
        outcome.assign(
            valid=lambda frame: frame["entry_tradable"].fillna(False)
            & np.isfinite(pd.to_numeric(frame["fwd_5d_open_return"], errors="coerce"))
        )
        .query("valid")
        .groupby("decision_dt", observed=True)["fwd_5d_open_return"]
        .agg([("benchmark_return_raw", "mean"), ("benchmark_observations", "size")])
        .reset_index()
    )
    previous: set[str] = set()
    rows: list[dict[str, Any]] = []
    for decision_dt, group in selected.groupby("decision_dt", sort=True, observed=True):
        current = set(group["symbol"].astype(str))
        returns = pd.to_numeric(group["fwd_5d_open_return"], errors="coerce")
        observable = group["entry_tradable"].fillna(False) & np.isfinite(returns)
        rows.append(
            {
                "decision_dt": pd.Timestamp(decision_dt),
                "slots": len(current),
                "observable_slots": int(observable.sum()),
                "buy_count": len(current - previous),
                "sell_count": len(previous - current),
                "terminal_sell_count": 0,
                "gross_return": 0.995 * float(returns.where(observable, 0.0).sum()) / 50,
            }
        )
        previous = current
    weekly = pd.DataFrame(rows).merge(benchmark, on="decision_dt", validate="one_to_one")
    if terminal_liquidation and not weekly.empty:
        weekly.loc[weekly.index[-1], "terminal_sell_count"] = len(previous)
    weekly["benchmark_return"] = 0.995 * weekly["benchmark_return_raw"]
    weekly["observable_target_slot_rate"] = weekly["observable_slots"] / 50
    for bps in (20, 40, 60):
        buy_cost, sell_cost = cost_pair(bps)
        cost = 0.995 * (
            weekly["buy_count"] * buy_cost
            + (weekly["sell_count"] + weekly["terminal_sell_count"]) * sell_cost
        ) / 50
        weekly[f"cost_{bps}bps"] = cost
        weekly[f"net_return_{bps}bps"] = weekly["gross_return"] - cost
        weekly[f"active_return_{bps}bps"] = weekly[f"net_return_{bps}bps"] - weekly["benchmark_return"]
    return weekly


def summarize_segment(
    weekly: pd.DataFrame,
    protocol: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bps: int,
    halves: list[list[str]],
) -> dict[str, Any]:
    """Compute the frozen active-return endpoint and path diagnostics."""

    scoped = weekly[weekly["decision_dt"].between(start, end)].copy()
    active = scoped[f"active_return_{bps}bps"].to_numpy(dtype=float)
    candidate = scoped[f"net_return_{bps}bps"].to_numpy(dtype=float)
    hac = {key: float(value) for key, value in newey_west_hac(active.tolist(), lag=4).items()}
    hac["one_sided_positive_p"] = float(norm.sf(hac["t_stat"]))
    bootstrap = {
        key: float(value)
        for key, value in circular_block_bootstrap(
            active.tolist(),
            block_size=int(protocol["statistics"]["circular_block_weeks"]),
            n_draws=int(protocol["statistics"]["bootstrap_draws"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        ).items()
    }
    wealth = np.cumprod(1 + candidate)
    running_max = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdowns = np.r_[1.0, wealth] / running_max - 1
    half_means = []
    for half_start, half_end in halves:
        part = scoped[scoped["decision_dt"].between(pd.Timestamp(half_start), pd.Timestamp(half_end))]
        half_means.append(
            {
                "start": half_start,
                "end": half_end,
                "weeks": int(len(part)),
                "active_mean": float(part[f"active_return_{bps}bps"].mean()),
            }
        )
    return {
        "weeks": int(len(scoped)),
        "active_mean": float(np.mean(active)),
        "candidate_net_mean": float(np.mean(candidate)),
        "benchmark_mean": float(scoped["benchmark_return"].mean()),
        "active_win_rate": float(np.mean(active > 0)),
        "observable_target_slot_rate": float(scoped["observable_target_slot_rate"].mean()),
        "compounded_candidate_return": float(wealth[-1] - 1),
        "compounded_candidate_max_drawdown": float(np.min(drawdowns)),
        "hac": hac,
        "bootstrap": bootstrap,
        "half_means": half_means,
    }


def evaluate_gate(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the same fixed gate to development and validation."""

    stats = protocol["statistics"]
    checks = {
        "minimum_active_effect": summary["active_mean"] >= float(stats["minimum_active_weekly_effect"]),
        "hac_positive": summary["hac"]["one_sided_positive_p"] < float(stats["one_sided_alpha"]),
        "bootstrap_q05_positive": summary["bootstrap"]["q05"] > 0,
        "both_halves_positive": all(row["active_mean"] > 0 for row in summary["half_means"]),
        "candidate_net_mean_positive": summary["candidate_net_mean"] > 0,
        "observable_rate": summary["observable_target_slot_rate"]
        >= float(stats["minimum_observable_target_slot_rate"]),
        "drawdown_cap": abs(summary["compounded_candidate_max_drawdown"])
        <= float(stats["maximum_compounded_drawdown"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _pct(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value * 100:.3f}%"


def write_report(audit: dict[str, Any]) -> None:
    """Write the failure-preserving study report."""

    dev = audit["development"]["summary_40bps"]
    lines = [
        "# 周频行业/规模中性超跌反转：预注册回顾性检验",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 边界：锁定验证仍是历史时间切分；通过也只能进入日频自融资复核与独立前向测试。",
        "",
        "## 固定策略",
        "",
        "每周用过去 5 个股票观察日收益，对申万一级行业和流通市值做横截面中性化；买入残差跌幅最大的",
        "50 只，排名 75 内保留，下一市场日开盘成交，固定 5 日开盘持有。主成本为买 15bp、卖 25bp。",
        "",
        "## 开发期（2022–2023）",
        "",
        f"- 周数：{dev['weeks']}；组合扣费周均：{_pct(dev['candidate_net_mean'])}；全市场基准：{_pct(dev['benchmark_mean'])}。",
        f"- 主端点主动周均：{_pct(dev['active_mean'])}；HAC t={dev['hac']['t_stat']:.3f}；",
        f"  block-bootstrap q05={_pct(dev['bootstrap']['q05'])}。",
        f"- 目标槽可观察率：{_pct(dev['observable_target_slot_rate'])}；复利最大回撤：{_pct(dev['compounded_candidate_max_drawdown'])}。",
        f"- 开发门：**{'PASS' if audit['development']['gate']['passed'] else 'FAIL'}**。",
        "",
    ]
    for name, passed in audit["development"]["gate"]["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    if not audit["validation_opened"]:
        lines.extend(
            [
                "",
                "## 结论",
                "",
                "开发门未全部通过，因此未计算或写出 2024+ 收益与统计。这个精确的连续超跌反转假设被",
                "回顾性否决，不调整回看期、中性化、持仓数、缓冲排名或持有期救援。",
            ]
        )
    else:
        val = audit["validation"]["summary_40bps"]
        lines.extend(
            [
                "",
                "## 锁定验证期（2024+）",
                "",
                f"- 周数：{val['weeks']}；主动周均：{_pct(val['active_mean'])}；HAC t={val['hac']['t_stat']:.3f}；",
                f"  block-bootstrap q05={_pct(val['bootstrap']['q05'])}。",
                f"- 验证门：**{'PASS' if audit['validation']['gate']['passed'] else 'FAIL'}**；",
                f"  60bp 压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
                "",
                "## 结论",
                "",
                audit["conclusion"],
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the frozen study with a hard development-to-validation gate."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blind = rank_reversal_signal(build_blind_surface(protocol))
    blind.to_parquet(OUTPUT_DIR / protocol["outputs"]["blind_ranked_signal"], index=False)
    memberships = build_memberships(blind)
    dev_start = pd.Timestamp(protocol["sample"]["development"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    val_start = pd.Timestamp(protocol["sample"]["validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["validation"][1])
    dev_memberships = memberships[memberships["decision_dt"].between(dev_start, dev_end)]
    dev_memberships.to_parquet(OUTPUT_DIR / protocol["outputs"]["development_memberships"], index=False)
    development_weekly = build_weekly_outcomes(
        protocol, memberships, end_date=dev_end, terminal_liquidation=False
    )
    development_weekly.to_parquet(OUTPUT_DIR / protocol["outputs"]["development_weekly"], index=False)
    dev_summaries = {
        bps: summarize_segment(
            development_weekly,
            protocol,
            dev_start,
            dev_end,
            bps=bps,
            halves=protocol["sample"]["development_halves"],
        )
        for bps in (20, 40, 60)
    }
    dev_gate = evaluate_gate(dev_summaries[PRIMARY_BPS], protocol)
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "blind_signal_rows": int(len(blind)),
        "blind_signal_weeks": int(blind["decision_dt"].nunique()),
        "membership_rows": int(len(memberships)),
        "development": {
            "summary_20bps": dev_summaries[20],
            "summary_40bps": dev_summaries[40],
            "summary_60bps": dev_summaries[60],
            "gate": dev_gate,
        },
        "validation_opened": bool(dev_gate["passed"]),
        "live_trading_authorized": False,
    }
    if not dev_gate["passed"]:
        audit["status"] = "HYPOTHESIS_FALSIFIED_DEVELOPMENT"
        audit["conclusion"] = "开发门失败；锁定验证未打开。"
    else:
        validation_weekly = build_weekly_outcomes(
            protocol, memberships, end_date=val_end, terminal_liquidation=True
        )
        memberships[memberships["decision_dt"].between(val_start, val_end)].to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_memberships"], index=False
        )
        validation_weekly[validation_weekly["decision_dt"] >= val_start].to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_weekly"], index=False
        )
        val_summaries = {
            bps: summarize_segment(
                validation_weekly,
                protocol,
                val_start,
                val_end,
                bps=bps,
                halves=protocol["sample"]["validation_halves"],
            )
            for bps in (20, 40, 60)
        }
        val_gate = evaluate_gate(val_summaries[PRIMARY_BPS], protocol)
        stress_checks = {
            "development_60bps_active_mean_positive": dev_summaries[60]["active_mean"] > 0,
            "development_60bps_bootstrap_q05_positive": dev_summaries[60]["bootstrap"]["q05"] > 0,
            "validation_60bps_active_mean_positive": val_summaries[60]["active_mean"] > 0,
            "validation_60bps_bootstrap_q05_positive": val_summaries[60]["bootstrap"]["q05"] > 0,
        }
        stress_gate = {"passed": all(stress_checks.values()), "checks": stress_checks}
        passed = bool(val_gate["passed"] and stress_gate["passed"])
        audit["validation"] = {
            "summary_20bps": val_summaries[20],
            "summary_40bps": val_summaries[40],
            "summary_60bps": val_summaries[60],
            "gate": val_gate,
        }
        audit["cost_stress"] = stress_gate
        audit["status"] = (
            "RETROSPECTIVE_REVERSAL_CANDIDATE_DAILY_PATH_AND_FORWARD_TEST_REQUIRED"
            if passed
            else "HYPOTHESIS_FALSIFIED_VALIDATION_OR_COST_STRESS"
        )
        audit["conclusion"] = (
            "开发、锁定验证和 60bp 压力门均通过；下一步必须构建日频自融资路径并进入独立前向测试。"
            if passed
            else "锁定验证或 60bp 压力门失败；该精确策略被回顾性否决，不做参数救援。"
        )
    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit)
    print(json.dumps({"status": audit["status"], "validation_opened": audit["validation_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
