"""重建 production delay5 的 raw→gate→hard→ST→market→fill→mature cohort。

本脚本只复核冻结的生产镜像规则，不搜索参数，也不做槽位选择或收益确认。输入候选表由
``surge_candidates_dump.py`` 生成，所有只依赖决策日及之前信息的结构候选都会落盘；本脚本从
独立价格面板重建下一交易日开盘成交与固定 60 个共同交易日成熟度，不读取 FULL 收益字段。

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
import surge_delay5_mirror_check as mirror
import surge_live as live
import surge_portfolio_backtest as portfolio

SCRIPTS_DIR = Path(__file__).resolve().parent
CANDIDATE_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "candidates.parquet"
PANEL_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "panel.parquet"
CANDIDATE_MANIFEST_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "manifest.json"
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
    "dec_close",
    "full_outcome_complete",
    "amount_e",
    "sl_pct",
    "limit_pct",
    "sig_vol_ratio",
    "sig_ma_spread_pct",
    "sig_ret20",
}
REQUIRED_MARKET_COLUMNS = {"dt", "high20_ratio", "ew_index_above_ma20"}
REQUIRED_PANEL_COLUMNS = {"symbol", "dt", "open", "close"}


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
    """严格下一市场交易日开盘 gap 模拟；等于板限减 0.3pct 也放弃。"""

    strict_next = frame["entry_fill_dt"].eq(frame["next_market_dt"])
    return (
        frame["entry_fill_observed"].fillna(False)
        & strict_next
        & frame["gap_pct"].lt(frame["limit_pct"] - portfolio.GAP_LIMIT_MARGIN)
    ).fillna(False)


def attach_entry_fill_observability(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """仅用逐股面板重建决策日之后下一根 bar 的入场成交。"""

    if missing := REQUIRED_PANEL_COLUMNS - set(panel):
        raise ValueError(f"panel input missing columns: {sorted(missing)}")
    scoped = panel.loc[:, ["symbol", "dt", "open", "close"]].copy()
    scoped["dt"] = pd.to_datetime(scoped["dt"])
    if scoped["dt"].isna().any():
        raise ValueError("panel contains invalid dates")
    duplicates = scoped.duplicated(["symbol", "dt"], keep=False)
    if duplicates.any():
        examples = scoped.loc[duplicates, ["symbol", "dt"]].head(5).to_dict("records")
        raise ValueError(f"panel is not unique by symbol/date: {examples}")

    scoped = scoped.sort_values(["symbol", "dt"], kind="mergesort")
    grouped = scoped.groupby("symbol", sort=False)
    scoped["entry_fill_dt"] = grouped["dt"].shift(-1)
    scoped["entry_fill_open"] = grouped["open"].shift(-1)
    scoped = scoped.rename(columns={"dt": "dec_dt", "close": "panel_dec_close"})
    scoped["_decision_observed"] = True
    out = frame.merge(
        scoped[
            [
                "symbol",
                "dec_dt",
                "panel_dec_close",
                "entry_fill_dt",
                "entry_fill_open",
                "_decision_observed",
            ]
        ],
        on=["symbol", "dec_dt"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if out["_decision_observed"].isna().any():
        examples = out.loc[out["_decision_observed"].isna(), ["symbol", "dec_dt"]].head(5).to_dict("records")
        raise ValueError(f"candidate decision row is absent from panel: {examples}")
    close_diff = (pd.to_numeric(out["dec_close"]) - pd.to_numeric(out["panel_dec_close"])).abs()
    if close_diff.gt(1e-8).any():
        raise ValueError("candidate decision close differs from the independently loaded panel")

    next_open = pd.to_numeric(out["entry_fill_open"], errors="coerce")
    observed = out["entry_fill_dt"].notna() & np.isfinite(next_open) & next_open.gt(0)
    out["entry_fill_observed"] = observed.astype(bool)
    out["entry_dt"] = out["entry_fill_dt"]
    out["entry_price"] = next_open
    out["gap_pct"] = (next_open / pd.to_numeric(out["dec_close"]) - 1) * 100
    return out.drop(columns="_decision_observed")


def mature_rule_mask(frame: pd.DataFrame) -> pd.Series:
    """要求严格入场后的固定 60 个共同市场交易日已经可观察。"""

    if "common_60_mature" not in frame:
        raise ValueError("cohort is missing common_60_mature")
    return frame["common_60_mature"].fillna(False).astype(bool)


def validate_raw(raw: pd.DataFrame) -> None:
    """验证 raw schema、固定模式以及生产 cohort 身份键。"""

    if missing := REQUIRED_CANDIDATE_COLUMNS - set(raw):
        raise ValueError(f"candidate input missing columns: {sorted(missing)}")
    if not raw["mode"].eq("anticipate").all() or not raw["delay"].eq(5).all():
        raise ValueError("raw cohort must contain only mode=anticipate and delay=5")
    if not raw["dec_regime"].isin(live.EXP_UPTREND_FAMILY).all():
        invalid = sorted(raw.loc[~raw["dec_regime"].isin(live.EXP_UPTREND_FAMILY), "dec_regime"].dropna().unique())
        raise ValueError(f"raw cohort contains invalid decision regimes: {invalid}")
    invalid_timing = raw[["sig_dt", "dec_dt"]].isna().any(axis=1) | ~raw["sig_dt"].lt(raw["dec_dt"])
    if invalid_timing.any():
        columns = ["symbol", "sig_dt", "dec_dt"]
        examples = raw.loc[invalid_timing, columns].head(5).to_dict("records")
        raise ValueError(f"raw cohort requires sig_dt < dec_dt: {examples}")
    duplicates = raw.duplicated(UNIQUE_KEY, keep=False)
    if duplicates.any():
        examples = raw.loc[duplicates, UNIQUE_KEY].head(5).to_dict("records")
        raise ValueError(f"raw cohort identity is not unique by {UNIQUE_KEY}: {examples}")


def merge_market_state(raw: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """按决策日 many-to-one 合并市场状态，并拒绝任何缺失。"""

    if missing := REQUIRED_MARKET_COLUMNS - set(market):
        raise ValueError(f"market input missing columns: {sorted(missing)}")
    scoped = market.loc[:, sorted(REQUIRED_MARKET_COLUMNS)].copy()
    scoped["dt"] = pd.to_datetime(scoped["dt"])
    scoped = scoped.sort_values("dt", kind="mergesort")
    if scoped["dt"].duplicated().any():
        raise ValueError("market input is not unique by date")
    scoped["next_market_dt"] = scoped["dt"].shift(-1)
    scoped = scoped.rename(columns={"dt": "dec_dt"})
    scoped["dec_dt"] = pd.to_datetime(scoped["dec_dt"])
    merged = raw.merge(scoped, on="dec_dt", how="left", validate="many_to_one")
    state_columns = ["high20_ratio", "ew_index_above_ma20"]
    if merged[state_columns].isna().any(axis=1).any():
        missing_dates = merged.loc[merged[state_columns].isna().any(axis=1), "dec_dt"].drop_duplicates().head(10)
        raise ValueError(f"market merge left missing decision dates: {[str(x.date()) for x in missing_dates]}")
    return merged


def attach_common_maturity(frame: pd.DataFrame, market: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """附加共同市场日历上的固定期限成熟标记，不读取任何个股未来收益。"""

    if horizon <= 0:
        raise ValueError("common maturity horizon must be positive")
    sessions = pd.DatetimeIndex(pd.to_datetime(market["dt"]).dropna().drop_duplicates().sort_values())
    positions = pd.Series(np.arange(len(sessions)), index=sessions)
    entry_pos = pd.to_datetime(frame["entry_fill_dt"]).map(positions)
    target_pos = entry_pos + horizon - 1
    mature = entry_pos.notna() & target_pos.lt(len(sessions))
    target_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if mature.any():
        target_dates.loc[mature] = sessions[target_pos.loc[mature].astype(int)].to_numpy()
    out = frame.copy()
    out["common_h60_dt"] = target_dates
    out["common_60_mature"] = mature.astype(bool)
    return out


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
    for column in ("sig_dt", "dec_dt"):
        raw[column] = pd.to_datetime(raw[column])
    validate_raw(raw)
    merged = merge_market_state(raw, market)
    merged = attach_entry_fill_observability(merged, panel)
    merged = attach_common_maturity(merged, market)
    non_st = pd.Series(
        [
            not portfolio.is_st_on(st_intervals, symbol, dec_dt)
            for symbol, dec_dt in zip(merged["symbol"], merged["dec_dt"], strict=False)
        ],
        index=merged.index,
    )
    result = apply_stage_flags(merged, non_st)
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


def validate_candidate_manifest(manifest: dict[str, Any], candidates: pd.DataFrame, panel: pd.DataFrame) -> None:
    """验证候选生成器已完整处理本地源文件，且输出身份未漂移。"""

    if manifest.get("schema") != "surge_candidates_dump_manifest_v2":
        raise RuntimeError("unexpected surge candidate manifest schema")
    if manifest.get("raw_completeness_proven") is not True:
        raise RuntimeError("candidate manifest does not prove raw completeness")
    sources = manifest.get("source_files", {})
    processing = sources.get("processing", {})
    if (
        sources.get("all_sources_accounted") is not True
        or int(sources.get("count", -1)) != sum(int(value) for value in processing.values())
        or int(processing.get("unreadable", -1)) != 0
        or int(processing.get("load_rejected_unknown", -1)) != 0
        or int(sources.get("panel_source_count", -1)) != int(sources.get("eligible_source_count", -2))
    ):
        raise RuntimeError("candidate manifest source accounting is incomplete")
    outputs = manifest.get("outputs", {})
    expected = {
        "candidates": (CANDIDATE_PATH, candidates),
        "panel": (PANEL_PATH, panel),
    }
    for label, (path, frame) in expected.items():
        identity = outputs.get(label, {})
        if identity.get("sha256") != sha256_file(path) or int(identity.get("rows", -1)) != len(frame):
            raise RuntimeError(f"candidate manifest has stale {label} binding")


def run_audit() -> dict[str, Any]:
    """读取固定本地输入、生成 cohort，并返回审计 JSON。"""

    require_file(CANDIDATE_PATH, "candidate")
    require_file(PANEL_PATH, "candidate panel")
    require_file(CANDIDATE_MANIFEST_PATH, "candidate manifest")
    require_file(MARKET_PATH, "market-state")
    require_file(ST_PATH, "historical ST")

    candidates = pd.read_parquet(CANDIDATE_PATH)
    panel = pd.read_parquet(PANEL_PATH, columns=sorted(REQUIRED_PANEL_COLUMNS))
    market = pd.read_parquet(MARKET_PATH)
    st_frame = pd.read_parquet(ST_PATH)
    manifest = json.loads(CANDIDATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_candidate_manifest(manifest, candidates, panel)
    st_intervals = portfolio.load_st_intervals()
    if not st_intervals:
        raise RuntimeError(f"historical ST input produced no intervals: {ST_PATH}")

    cohort = build_cohort(candidates, market, panel, st_intervals)
    main = cohort[cohort["dec_dt"].ge(MAIN_START)].copy()
    yearly = {str(int(year)): funnel_counts(group) for year, group in main.groupby(main["dec_dt"].dt.year, sort=True)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(COHORT_PATH, index=False)
    audit: dict[str, Any] = {
        "schema": "surge_delay5_production_cohort_audit_v3",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_completeness_proven": True,
        "design": {
            "identity_key": UNIQUE_KEY,
            "stage_semantics": "cumulative membership in raw→gate→hard→ST→market→fill→mature order",
            "main_window": f"dec_dt >= {MAIN_START.date()}",
            "capacity_selection_applied": False,
            "execution_scope": "panel-rebuilt strict next-session entry plus fixed common 60-session maturity",
            "outcome_fields_used_for_cohort": False,
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
            "fill": "finite panel open on the strict next market session and gap_pct < limit_pct - 0.3pct",
            "mature": "60th common market session from entry is present; no FULL outcome field is read",
        },
        "inputs": {
            "candidates": input_identity(CANDIDATE_PATH, candidates, "dec_dt"),
            "candidate_manifest": {
                "path": str(CANDIDATE_MANIFEST_PATH),
                "sha256": sha256_file(CANDIDATE_MANIFEST_PATH),
            },
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
        "entry_fill_observability": {
            "raw_rows": int(len(cohort)),
            "observable_next_symbol_bar": int(cohort["entry_fill_observed"].sum()),
            "strict_next_market_session": int(
                (cohort["entry_fill_observed"] & cohort["entry_fill_dt"].eq(cohort["next_market_dt"])).sum()
            ),
            "full_outcome_complete_rows": int(cohort["full_outcome_complete"].sum()),
        },
        "first_failed_stage": {
            str(stage): int(count)
            for stage, count in cohort["first_failed_stage"].fillna("passed_all").value_counts().sort_index().items()
        },
        "full_semantics_warnings": [
            "FULL outcomes are carried only for legacy diagnostics and never participate in raw, fill or maturity membership.",
            "FULL exit_dt is the actual fill date; exit_signal_dt separately records close/state trigger dates.",
            "Regimes 9 and 10 both close the whole candidate trade; the documented 9-reduce/10-close position behavior is not represented.",
            "Hard-filter sl_pct may fall back to decision zd, while FULL SL2 uses StateSnapshot.sl_ref without that fallback.",
            "Close-triggered trail18 and state exits use the next observed open, but remain local simulations rather than broker records.",
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
