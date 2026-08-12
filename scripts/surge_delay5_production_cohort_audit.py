"""重建 production delay5 的 raw→gate→hard→ST→market→fill→mature cohort。

本脚本只复核冻结的生产镜像规则，不搜索参数，也不做槽位选择或收益确认。输入候选表仍由
``surge_candidates_dump.py`` 生成；该表在落盘前已经要求未来入场 bar 并运行 FULL 退出，
因此 ``raw`` 只是当前 dump 可观察到的门控前 delay5 结构集合，不能证明完整的纯因果 raw。

输出：

- ``scripts/_output/surge_delay5_production_cohort/cohort.parquet``：全部历史 raw 行及七阶段累计布尔；
- ``scripts/_output/surge_delay5_production_cohort/audit.json``：输入身份、规则、漏斗和语义边界。

    uv run --no-sync python scripts/surge_delay5_production_cohort_audit.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import s2b_matched_control_audit as matched
import surge_delay5_mirror_check as mirror
import surge_live as live
import surge_portfolio_backtest as portfolio

SCRIPTS_DIR = Path(__file__).resolve().parent
CANDIDATE_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "candidates.parquet"
PANEL_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "panel.parquet"
MARKET_PATH = SCRIPTS_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
ST_PATH = portfolio.NAMECHANGE_PATH
OUTPUT_DIR = SCRIPTS_DIR / "_output" / "surge_delay5_production_cohort"
COHORT_PATH = OUTPUT_DIR / "cohort.parquet"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
MAIN_START = pd.Timestamp("2024-01-01")

UNIQUE_KEY = ["symbol", "sig_dt", "dec_dt"]
STAGE_COLUMNS = [
    "stage_raw",
    "stage_gate",
    "stage_hard",
    "stage_st",
    "stage_market",
    "stage_fill",
    "stage_mature",
]
REQUIRED_CANDIDATE_COLUMNS = {
    "symbol",
    "mode",
    "delay",
    "dec_regime",
    "sig_dt",
    "dec_dt",
    "entry_dt",
    "exit_dt",
    "exit_reason",
    "hold_days",
    "amount_e",
    "sl_pct",
    "gap_pct",
    "limit_pct",
    "sig_vol_ratio",
    "sig_ma_spread_pct",
    "sig_ret20",
}
REQUIRED_MARKET_COLUMNS = {"dt", "high20_ratio", "ew_index_above_ma20"}
REQUIRED_PANEL_COLUMNS = {"symbol", "dt", "open"}


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    """要求审计输入存在；尤其不允许 ST 参考缺失后静默放行。"""

    if not path.is_file():
        raise FileNotFoundError(f"required {label} input is missing: {path}")


def signal_gate_mask(frame: pd.DataFrame) -> pd.Series:
    """当前 Rust/live 默认 anticipate 信号门：vr≤0.8、spread≥3、ret20≥8。"""

    return mirror._signal_gate(frame).astype(bool)


def hard_rule_mask(frame: pd.DataFrame) -> pd.Series:
    """决策日固定硬过滤；ST、市场状态和次日 gap 不在本层。"""

    return (
        frame["amount_e"].ge(portfolio.MIN_AMOUNT_E)
        & frame["sl_pct"].between(portfolio.STOP_MIN_PCT, portfolio.STOP_MAX_PCT)
    ).fillna(False)


def market_rule_mask(frame: pd.DataFrame) -> pd.Series:
    """决策日市场门，严格复现 ``surge_live.market_gate_open`` 的边界。"""

    return (frame["high20_ratio"].gt(live.MARKET_GATE_HIGH20) & frame["ew_index_above_ma20"].gt(0)).fillna(False)


def fill_rule_mask(frame: pd.DataFrame) -> pd.Series:
    """次一可交易 bar 开盘 gap 模拟；等于板限减 0.3pct 也放弃。"""

    return frame["gap_pct"].lt(frame["limit_pct"] - portfolio.GAP_LIMIT_MARGIN).fillna(False)


def attach_state_fill_observability(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """从逐股 panel 验证 state 触发日之后的下一根开盘是否真实可观察。"""

    if missing := REQUIRED_PANEL_COLUMNS - set(panel):
        raise ValueError(f"panel input missing columns: {sorted(missing)}")
    scoped = panel.loc[:, ["symbol", "dt", "open"]].copy()
    scoped["dt"] = pd.to_datetime(scoped["dt"])
    if scoped["dt"].isna().any():
        raise ValueError("panel contains invalid dates")
    duplicates = scoped.duplicated(["symbol", "dt"], keep=False)
    if duplicates.any():
        examples = scoped.loc[duplicates, ["symbol", "dt"]].head(5).to_dict("records")
        raise ValueError(f"panel is not unique by symbol/date: {examples}")

    scoped = scoped.sort_values(["symbol", "dt"], kind="mergesort")
    grouped = scoped.groupby("symbol", sort=False)
    scoped["state_fill_dt"] = grouped["dt"].shift(-1)
    scoped["state_fill_open"] = grouped["open"].shift(-1)
    scoped["_state_trigger_observed"] = True
    lookup = scoped.rename(columns={"dt": "exit_dt"})
    out = frame.merge(
        lookup[["symbol", "exit_dt", "state_fill_dt", "state_fill_open", "_state_trigger_observed"]],
        on=["symbol", "exit_dt"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    state = out["exit_reason"].eq("state")
    if out.loc[state, "_state_trigger_observed"].isna().any():
        examples = out.loc[state & out["_state_trigger_observed"].isna(), ["symbol", "exit_dt"]].head(5)
        records = examples.to_dict("records")
        raise ValueError(f"state exit trigger is absent from panel: {records}")

    next_open = pd.to_numeric(out["state_fill_open"], errors="coerce")
    observable = out["state_fill_dt"].notna() & np.isfinite(next_open)
    out["state_fill_observed"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[state, "state_fill_observed"] = observable.loc[state].to_numpy(dtype=bool)
    return out.drop(columns="_state_trigger_observed")


def mature_rule_mask(frame: pd.DataFrame) -> pd.Series:
    """排除未完成 max-hold，以及缺少真实次根开盘的 state 尾部退出。"""

    if "state_fill_observed" not in frame:
        raise ValueError("cohort is missing state_fill_observed")
    state_fill_ok = ~frame["exit_reason"].eq("state") | frame["state_fill_observed"].fillna(False)
    return (~matched.censored_tail_mask(frame) & state_fill_ok).astype(bool)


def validate_raw(raw: pd.DataFrame) -> None:
    """验证 raw schema、固定模式以及生产 cohort 身份键。"""

    if missing := REQUIRED_CANDIDATE_COLUMNS - set(raw):
        raise ValueError(f"candidate input missing columns: {sorted(missing)}")
    if not raw["mode"].eq("anticipate").all() or not raw["delay"].eq(5).all():
        raise ValueError("raw cohort must contain only mode=anticipate and delay=5")
    if not raw["dec_regime"].isin(live.EXP_UPTREND_FAMILY).all():
        invalid = sorted(raw.loc[~raw["dec_regime"].isin(live.EXP_UPTREND_FAMILY), "dec_regime"].dropna().unique())
        raise ValueError(f"raw cohort contains invalid decision regimes: {invalid}")
    invalid_timing = (
        raw[["sig_dt", "dec_dt", "entry_dt", "exit_dt"]].isna().any(axis=1)
        | ~raw["sig_dt"].lt(raw["dec_dt"])
        | ~raw["dec_dt"].lt(raw["entry_dt"])
        | ~raw["entry_dt"].le(raw["exit_dt"])
    )
    if invalid_timing.any():
        columns = ["symbol", "sig_dt", "dec_dt", "entry_dt", "exit_dt"]
        examples = raw.loc[invalid_timing, columns].head(5).to_dict("records")
        raise ValueError(f"raw cohort requires sig_dt < dec_dt < entry_dt <= exit_dt: {examples}")
    duplicates = raw.duplicated(UNIQUE_KEY, keep=False)
    if duplicates.any():
        examples = raw.loc[duplicates, UNIQUE_KEY].head(5).to_dict("records")
        raise ValueError(f"raw cohort identity is not unique by {UNIQUE_KEY}: {examples}")


def merge_market_state(raw: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """按决策日 many-to-one 合并市场状态，并拒绝任何缺失。"""

    if missing := REQUIRED_MARKET_COLUMNS - set(market):
        raise ValueError(f"market input missing columns: {sorted(missing)}")
    scoped = market.loc[:, sorted(REQUIRED_MARKET_COLUMNS)].rename(columns={"dt": "dec_dt"}).copy()
    scoped["dec_dt"] = pd.to_datetime(scoped["dec_dt"])
    merged = raw.merge(scoped, on="dec_dt", how="left", validate="many_to_one")
    state_columns = ["high20_ratio", "ew_index_above_ma20"]
    if merged[state_columns].isna().any(axis=1).any():
        missing_dates = merged.loc[merged[state_columns].isna().any(axis=1), "dec_dt"].drop_duplicates().head(10)
        raise ValueError(f"market merge left missing decision dates: {[str(x.date()) for x in missing_dates]}")
    return merged


def apply_stage_flags(frame: pd.DataFrame, non_st_mask: pd.Series) -> pd.DataFrame:
    """添加七阶段累计布尔和首个失败阶段。"""

    if len(non_st_mask) != len(frame):
        raise ValueError("non_st_mask length differs from cohort length")
    non_st = pd.Series(non_st_mask.to_numpy(dtype=bool), index=frame.index)
    rules = {
        "stage_gate": signal_gate_mask(frame),
        "stage_hard": hard_rule_mask(frame),
        "stage_st": non_st,
        "stage_market": market_rule_mask(frame),
        "stage_fill": fill_rule_mask(frame),
        "stage_mature": mature_rule_mask(frame),
    }
    out = frame.copy()
    out["stage_raw"] = True
    previous = out["stage_raw"]
    for stage in STAGE_COLUMNS[1:]:
        out[stage] = previous & rules[stage]
        previous = out[stage]

    first_failed = pd.Series(pd.NA, index=out.index, dtype="string")
    for stage in STAGE_COLUMNS[1:]:
        failed_here = first_failed.isna() & ~out[stage]
        first_failed.loc[failed_here] = stage.removeprefix("stage_")
    out["first_failed_stage"] = first_failed
    return out


def build_cohort(
    candidates: pd.DataFrame,
    market: pd.DataFrame,
    panel: pd.DataFrame,
    st_intervals: dict[str, list],
) -> pd.DataFrame:
    """从共享候选 dump 构建完整历史 production-delay5 漏斗。"""

    if missing := REQUIRED_CANDIDATE_COLUMNS - set(candidates):
        raise ValueError(f"candidate input missing columns: {sorted(missing)}")
    raw = candidates.loc[candidates["mode"].eq("anticipate") & candidates["delay"].eq(5)].copy()
    for column in ("sig_dt", "dec_dt", "entry_dt", "exit_dt"):
        raw[column] = pd.to_datetime(raw[column])
    validate_raw(raw)
    merged = attach_state_fill_observability(merge_market_state(raw, market), panel)
    non_st = pd.Series(
        [
            not portfolio.is_st_on(st_intervals, symbol, dec_dt)
            for symbol, dec_dt in zip(merged["symbol"], merged["dec_dt"], strict=False)
        ],
        index=merged.index,
    )
    result = apply_stage_flags(merged, non_st)
    invalid_state_mature = (
        result["exit_reason"].eq("state") & result["stage_mature"] & ~result["state_fill_observed"].fillna(False)
    )
    if invalid_state_mature.any():
        raise AssertionError("mature state exits must have an observable next-bar open")
    return result.sort_values(["dec_dt", "symbol"], kind="mergesort").reset_index(drop=True)


def funnel_counts(frame: pd.DataFrame) -> dict[str, int]:
    """汇总七个累计阶段的行数。"""

    return {stage.removeprefix("stage_"): int(frame[stage].sum()) for stage in STAGE_COLUMNS}


def input_identity(path: Path, frame: pd.DataFrame, date_column: str) -> dict[str, Any]:
    """生成可复核的输入文件身份。"""

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    maximum = dates.max()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "max_date": str(maximum.date()) if pd.notna(maximum) else None,
    }


def run_audit() -> dict[str, Any]:
    """读取固定本地输入、生成 cohort，并返回审计 JSON。"""

    require_file(CANDIDATE_PATH, "candidate")
    require_file(PANEL_PATH, "candidate panel")
    require_file(MARKET_PATH, "market-state")
    require_file(ST_PATH, "historical ST")

    candidates = pd.read_parquet(CANDIDATE_PATH)
    panel = pd.read_parquet(PANEL_PATH, columns=sorted(REQUIRED_PANEL_COLUMNS))
    market = pd.read_parquet(MARKET_PATH)
    st_frame = pd.read_parquet(ST_PATH)
    st_intervals = portfolio.load_st_intervals()
    if not st_intervals:
        raise RuntimeError(f"historical ST input produced no intervals: {ST_PATH}")

    cohort = build_cohort(candidates, market, panel, st_intervals)
    main = cohort[cohort["dec_dt"].ge(MAIN_START)].copy()
    yearly = {str(int(year)): funnel_counts(group) for year, group in main.groupby(main["dec_dt"].dt.year, sort=True)}
    state = cohort["exit_reason"].eq("state")
    observed_state = cohort["state_fill_observed"].fillna(False)
    filled_state = state & cohort["stage_fill"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(COHORT_PATH, index=False)
    audit: dict[str, Any] = {
        "schema": "surge_delay5_production_cohort_audit_v2",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_completeness_proven": False,
        "design": {
            "identity_key": UNIQUE_KEY,
            "stage_semantics": "cumulative membership in raw→gate→hard→ST→market→fill→mature order",
            "main_window": f"dec_dt >= {MAIN_START.date()}",
            "capacity_selection_applied": False,
            "execution_scope": "entry gap proxy plus candidate-level frozen FULL outcome; state exit maturity is panel-verified",
        },
        "rules": {
            "raw": f"anticipate structural onset, decision delay=5, decision regime in {sorted(live.EXP_UPTREND_FAMILY)}",
            "gate": (
                f"signal-day sig_vol_ratio<={mirror.tr.SURGE_GATE_VOL_RATIO}, "
                f"sig_ma_spread_pct>={mirror.tr.SURGE_GATE_MA_SPREAD}, "
                f"sig_ret20>={mirror.tr.SURGE_GATE_RET20}; above_zg unused"
            ),
            "hard": (
                f"decision amount_e>={portfolio.MIN_AMOUNT_E} and "
                f"{portfolio.STOP_MIN_PCT}<=sl_pct<={portfolio.STOP_MAX_PCT}"
            ),
            "st": "non-ST on decision date according to required local namechange intervals",
            "market": f"decision high20_ratio>{live.MARKET_GATE_HIGH20} and ew_index_above_ma20>0",
            "mature": "not(max_hold with hold_days<59); state additionally requires finite next observed panel-bar open",
        },
        "inputs": {
            "candidates": input_identity(CANDIDATE_PATH, candidates, "dec_dt"),
            "panel": input_identity(PANEL_PATH, panel, "dt"),
            "market_state": input_identity(MARKET_PATH, market, "dt"),
            "historical_st": input_identity(ST_PATH, st_frame, "start_date"),
        },
        "outputs": {
            "cohort": {
                "path": str(COHORT_PATH),
                "sha256": sha256_file(COHORT_PATH),
                "rows": int(len(cohort)),
                "max_decision_date": str(cohort["dec_dt"].max().date()) if len(cohort) else None,
            }
        },
        "funnels": {
            "all_history": funnel_counts(cohort),
            "main_2024_plus": funnel_counts(main),
            "by_decision_year_2024_plus": yearly,
        },
        "state_exit_fill_observability": {
            "all_state_rows": int(state.sum()),
            "observable_next_open": int((state & observed_state).sum()),
            "unobservable_next_open": int((state & ~observed_state).sum()),
            "stage_fill_state_rows": int(filled_state.sum()),
            "stage_fill_observable_next_open": int((filled_state & observed_state).sum()),
            "stage_fill_unobservable_next_open": int((filled_state & ~observed_state).sum()),
        },
        "first_failed_stage": {
            str(stage): int(count)
            for stage, count in cohort["first_failed_stage"].fillna("passed_all").value_counts().sort_index().items()
        },
        "full_semantics_warnings": [
            "The dump calls FULL simulation before persisting a row and drops signals without a future entry bar; raw is future-availability conditioned.",
            "FULL state exits record exit_idx/exit_dt on the trigger bar; maturity now requires a finite open on the following observed panel row.",
            "Regimes 9 and 10 both close the whole candidate trade; the documented 9-reduce/10-close position behavior is not represented.",
            "Hard-filter sl_pct may fall back to decision zd, while FULL SL2 uses StateSnapshot.sl_ref without that fallback.",
            "trail18 triggers from a close and is priced at the same close; fill is a local gap simulation, not a broker execution record.",
            "Historical market state is recomputed from the current qfq panel, and missing namechange history is assumed non-ST by is_st_on.",
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    audit = run_audit()
    print(json.dumps(audit["funnels"], ensure_ascii=False, indent=2))
    print(f"[output] {COHORT_PATH}")
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
