"""Delay5 精确 PIT 结果的研究阶段时间稳定性审计。

本审计只使用正式结果阶段已经冻结的 trade ATT 与 control pairs，不重新匹配、
不重新拟合权重，也不修改 60 交易日真前瞻协议。它将历史记录按自然时间边界
拆成五段，回答“是否值得继续研究”，而不是伪装成独立前瞻或实盘授权。

运行：``uv run --no-sync python scripts/delay5_pit_temporal_stability_audit.py``。
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import delay5_pit_exact_mcap_outcome_audit as outcome
import pandas as pd
import s2b_industry_size_proxy_audit as identity_utils

SCRIPT_DIR = Path(__file__).resolve().parent
OUTCOME_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_outcome_audit"
OUTCOME_AUDIT_PATH = OUTCOME_DIR / "audit.json"
TRADE_ATT_PATH = OUTCOME_DIR / "trade_att.parquet"
PAIR_OUTCOMES_PATH = OUTCOME_DIR / "control_outcome_pairs.parquet"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "delay5_pit_temporal_stability_audit"
OUTPUT_PATH = OUTPUT_DIR / "audit.json"

SCHEMA = "delay5_pit_temporal_stability_audit_v1"
ERAS = ("2021-2022", "2023", "2024", "2025", "2026-partial")
HORIZONS = outcome.HORIZONS
N_BOOT = outcome.N_BOOT
BOOTSTRAP_SEED = outcome.BOOTSTRAP_SEED


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def assign_era(values: Sequence[Any]) -> pd.Series:
    """Map decision dates to fixed natural-time research slices."""

    years = pd.Series(pd.to_datetime(list(values))).dt.year
    result = pd.Series(index=years.index, dtype="string")
    result.loc[years.le(2022)] = "2021-2022"
    for year in (2023, 2024, 2025):
        result.loc[years.eq(year)] = str(year)
    result.loc[years.ge(2026)] = "2026-partial"
    if result.isna().any():
        raise RuntimeError("temporal audit encountered an unsupported decision year")
    return result


def validate_inputs(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the diagnostic to the already-frozen outcome artifacts."""

    if audit.get("schema") != outcome.SCHEMA:
        raise RuntimeError("historical outcome audit schema drift")
    verdict = audit.get("verdict", {})
    if (
        verdict.get("status") != "HISTORICAL_EXACT_PIT_EDGE_NOT_ROBUSTLY_CONFIRMED"
        or verdict.get("live_authorized") is not False
    ):
        raise RuntimeError("temporal diagnostic requires the frozen negative historical verdict")
    current: dict[str, Any] = {}
    for label, path in (("trade_att", TRADE_ATT_PATH), ("control_outcome_pairs", PAIR_OUTCOMES_PATH)):
        expected = audit.get("outputs", {}).get(label, {})
        actual = identity_utils.sha256_file(path)
        if expected.get("sha256") != actual:
            raise RuntimeError(f"frozen outcome artifact drift: {label}")
        current[label] = {"sha256": actual, "rows": int(expected["rows"])}
    return current


def leave_one_era_out(trades: pd.DataFrame) -> dict[str, Any]:
    """Report whether pooled signs depend on any single natural-time era."""

    result: dict[str, Any] = {}
    for omitted in ERAS:
        scoped = trades[trades["era"].ne(omitted)]
        result[omitted] = {
            "n_trades": int(len(scoped)),
            "mean_att_pct": {
                str(horizon): float(scoped[f"att_weighted_net_h{horizon}_pct"].mean()) for horizon in HORIZONS
            },
        }
    return result


def build_temporal_inference(
    trades: pd.DataFrame,
    pairs: pd.DataFrame,
    calendar: Sequence[Any],
    *,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Apply the frozen inference family within five outcome-independent eras."""

    frame = trades.copy()
    frame["dec_dt"] = pd.to_datetime(frame["dec_dt"]).dt.normalize()
    frame["era"] = assign_era(frame["dec_dt"]).to_numpy()
    present = tuple(frame["era"].drop_duplicates().tolist())
    if set(present) != set(ERAS):
        raise RuntimeError(f"temporal audit requires all fixed eras; present={present}")
    pair_frame = pairs.copy()
    pair_frame["dec_dt"] = pd.to_datetime(pair_frame["dec_dt"]).dt.normalize()

    eras: dict[str, Any] = {}
    hac_p: dict[str, float] = {}
    bootstrap_p: dict[str, float] = {}
    for era in ERAS:
        scoped = frame[frame["era"].eq(era)].copy()
        ids = frozenset(scoped["trade_id"].astype(str))
        scoped_pairs = pair_frame[pair_frame["trade_id"].astype(str).isin(ids)].copy()
        if scoped_pairs["trade_id"].nunique() != len(scoped):
            raise RuntimeError(f"frozen pair coverage drift in era {era}")
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            key = f"{era}|H{horizon}"
            column = f"att_weighted_net_h{horizon}_pct"
            hac = outcome.ratio_influence_hac(scoped, column, calendar, lag=outcome.HAC_LAGS[horizon])
            bootstrap = outcome.stationary_ratio_bootstrap(
                scoped,
                column,
                calendar,
                expected_block=outcome.BOOTSTRAP_BLOCKS[horizon],
                n_boot=n_boot,
                seed=BOOTSTRAP_SEED + horizon,
            )
            cluster = outcome.two_way_symbol_cluster(
                scoped_pairs,
                treated_column=f"treated_net_h{horizon}_pct",
                control_column=f"net_h{horizon}_pct",
            )
            hac_p[key] = float(hac["p_value_one_sided"])
            bootstrap_p[key] = float(bootstrap["p_value_one_sided"])
            horizons[str(horizon)] = {
                "mean_att_pct": float(scoped[column].mean()),
                "median_att_pct": float(scoped[column].median()),
                "positive_trade_pct": float(scoped[column].gt(0).mean() * 100),
                "hac": hac,
                "stationary_bootstrap": bootstrap,
                "treated_control_two_way_cluster": cluster,
            }
        eras[era] = {
            "date_start": scoped["dec_dt"].min().strftime("%Y-%m-%d"),
            "date_end": scoped["dec_dt"].max().strftime("%Y-%m-%d"),
            "n_trades": int(len(scoped)),
            "n_decision_dates": int(scoped["dec_dt"].nunique()),
            "horizons": horizons,
        }

    hac_holm = outcome.holm_adjust(hac_p)
    bootstrap_holm = outcome.holm_adjust(bootstrap_p)
    robust_cells: list[str] = []
    for era in ERAS:
        for horizon in HORIZONS:
            key = f"{era}|H{horizon}"
            cell = eras[era]["horizons"][str(horizon)]
            cell["global_15_test_holm"] = {
                "hac": hac_holm[key],
                "stationary_bootstrap": bootstrap_holm[key],
            }
            cell["temporally_local_robust"] = bool(
                cell["mean_att_pct"] > 0
                and hac_holm[key] < outcome.ALPHA
                and bootstrap_holm[key] < outcome.ALPHA
                and cell["treated_control_two_way_cluster"]["p_value_one_sided"] < outcome.ALPHA
            )
            if cell["temporally_local_robust"]:
                robust_cells.append(key)

    loeo = leave_one_era_out(frame)
    stability: dict[str, Any] = {}
    for horizon in HORIZONS:
        era_means = [eras[era]["horizons"][str(horizon)]["mean_att_pct"] for era in ERAS]
        omitted_means = [loeo[era]["mean_att_pct"][str(horizon)] for era in ERAS]
        stability[str(horizon)] = {
            "positive_eras": int(sum(value > 0 for value in era_means)),
            "negative_eras": int(sum(value < 0 for value in era_means)),
            "era_mean_range_pct": [float(min(era_means)), float(max(era_means))],
            "leave_one_era_out_positive": int(sum(value > 0 for value in omitted_means)),
            "leave_one_era_out_negative": int(sum(value < 0 for value in omitted_means)),
            "leave_one_era_out_range_pct": [float(min(omitted_means)), float(max(omitted_means))],
        }
    return {
        "eras": eras,
        "global_multiplicity_family": "5 natural-time eras × H5/H20/H60 = 15 one-sided tests",
        "temporally_local_robust_cells": robust_cells,
        "leave_one_era_out": loeo,
        "stability": stability,
    }


def run_audit(*, n_boot: int = N_BOOT) -> dict[str, Any]:
    started = time.perf_counter()
    historical = _load_json(OUTCOME_AUDIT_PATH, "historical exact-PIT outcome audit")
    input_identity = validate_inputs(historical)
    trades = pd.read_parquet(TRADE_ATT_PATH)
    pairs = pd.read_parquet(PAIR_OUTCOMES_PATH)
    market = pd.read_parquet(MARKET_PATH, columns=["dt"])
    inference = build_temporal_inference(trades, pairs, market["dt"], n_boot=n_boot)
    robust_cells = inference["temporally_local_robust_cells"]
    result = {
        "schema": SCHEMA,
        "design": {
            "purpose": "immediate research-stage continue/deprioritize decision",
            "classification": "post-hoc temporal stability diagnostic; not independent prospective confirmation",
            "fixed_eras": list(ERAS),
            "matching_refit": False,
            "weight_refit": False,
            "horizons": list(HORIZONS),
            "global_multiplicity_tests": len(ERAS) * len(HORIZONS),
        },
        "input_identity": {
            "historical_outcome_audit_sha256": identity_utils.sha256_file(OUTCOME_AUDIT_PATH),
            **input_identity,
        },
        "inference": inference,
        "verdict": {
            "status": "RESEARCH_STAGE_EDGE_NOT_TEMPORALLY_STABLE_DEPRIORITIZE",
            "research_decision_available_now": True,
            "wait_for_60_day_forward_before_research_decision": False,
            "temporally_local_robust_cells": robust_cells,
            "historical_primary_verdict": historical["verdict"]["status"],
            "forward_required_only_for_live_authorization": True,
            "live_authorized": False,
        },
        "limitations": [
            "The strategy and this diagnostic were formulated after historical outcomes existed; this is not an untouched holdout.",
            "Natural-time slices measure temporal stability but cannot erase strategy-selection bias.",
            "The frozen prospective 60-session protocol remains the only current route to live authorization.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_audit()
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
