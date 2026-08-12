"""Delay5 精确 PIT 研究的独立 60 交易日前瞻协议与结果盲进度审计。

前瞻样本只接纳协议冻结日 ``2026-08-12`` 之后、首 60 个已观察官方交易日内的生产
decision。状态审计仅读取漏斗、成交和固定 H60 日程，不读取任何未来收益。
最后一个入组 decision 的已成交交易全部成熟前，结果访问保持锁定。

首次冻结：``uv run --no-sync python scripts/surge_delay5_forward_audit.py --write-protocol``
进度审计：``uv run --no-sync python scripts/surge_delay5_forward_audit.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import delay5_pit_exact_mcap_plan as pit_plan
import pandas as pd
import s2b_industry_size_proxy_audit as identity_utils

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "surge_delay5_forward_protocol_2026-08-12.json"
PRODUCTION_DIR = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort"
PRODUCTION_AUDIT_PATH = PRODUCTION_DIR / "audit.json"
COHORT_PATH = PRODUCTION_DIR / "cohort.parquet"
CANDIDATE_MANIFEST_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "manifest.json"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
BALANCE_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_balance_audit"
BALANCE_AUDIT_PATH = BALANCE_DIR / "audit.json"
BALANCE_PATH = BALANCE_DIR / "balance.json"
OUTCOME_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_outcome_audit"
OUTCOME_AUDIT_PATH = OUTCOME_DIR / "audit.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "surge_delay5_forward_audit"
STATUS_PATH = OUTPUT_DIR / "status.json"

SCHEMA = "surge_delay5_forward_protocol_v1"
STATUS_SCHEMA = "surge_delay5_forward_status_v1"
FROZEN_AT_UTC = "2026-08-12T08:25:35Z"
GENESIS_SOURCE_CUTOFF = pd.Timestamp("2026-08-11")
DECISION_CUTOFF = pd.Timestamp("2026-08-12")
ACCRUAL_SESSIONS = 60
HORIZONS = (5, 20, 60)
HAC_LAGS = {5: 4, 20: 19, 60: 59}
BOOTSTRAP_BLOCKS = {5: 5, 20: 20, 60: 60}
N_BOOT = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
FORWARD_COHORT_COLUMNS = (
    "symbol",
    "sig_dt",
    "dec_dt",
    "entry_dt",
    "entry_fill_dt",
    "common_h60_dt",
    "stage_raw",
    "stage_gate",
    "stage_hard",
    "stage_st",
    "stage_market",
    "stage_fill",
    "stage_mature",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _repo_locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository: {path}") from exc


def _record(path: Path) -> dict[str, Any]:
    return {"locator": _repo_locator(path), "sha256": identity_utils.sha256_file(path)}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_protocol() -> dict[str, Any]:
    """从已完成的历史审计冻结前瞻设计；不读取任何未来收益表。"""

    plan = _load_json(pit_plan.EXPECTED_PLAN_PATH, "tracked PIT plan")
    balance_audit = _load_json(BALANCE_AUDIT_PATH, "balance audit")
    balance = _load_json(BALANCE_PATH, "balance weights")
    historical = _load_json(OUTCOME_AUDIT_PATH, "historical outcome audit")
    production = _load_json(PRODUCTION_AUDIT_PATH, "production cohort audit")
    candidate_manifest = _load_json(CANDIDATE_MANIFEST_PATH, "candidate manifest")
    pit_plan.validate_design_contract(plan)
    if balance_audit.get("verdict", {}).get("outcome_evaluation_permitted") is not True:
        raise RuntimeError("historical balance gate did not pass")
    if historical.get("verdict", {}).get("live_authorized") is not False:
        raise RuntimeError("historical outcome audit authorization drift")
    if historical.get("verdict", {}).get("status") != "HISTORICAL_EXACT_PIT_EDGE_NOT_ROBUSTLY_CONFIRMED":
        raise RuntimeError("forward protocol expects the frozen negative historical verdict")
    if (
        production.get("raw_completeness_proven") is not True
        or candidate_manifest.get("raw_completeness_proven") is not True
    ):
        raise RuntimeError("raw candidate completeness is not proven")
    if production.get("design", {}).get("outcome_fields_used_for_cohort") is not False:
        raise RuntimeError("production cohort is not outcome blind")
    maxima = {
        production.get("inputs", {}).get(label, {}).get("max_date") for label in ("candidates", "panel", "market_state")
    }
    if maxima != {GENESIS_SOURCE_CUTOFF.strftime("%Y-%m-%d")}:
        raise RuntimeError(f"forward genesis source cutoff drift: {sorted(maxima)}")
    entropy = balance.get("entropy_weighting", {})
    required_features = balance.get("required_features", [])
    if not entropy.get("solver_success") or not required_features:
        raise RuntimeError("frozen entropy transport parameters are unavailable")
    return {
        "schema": SCHEMA,
        "frozen_at_utc": FROZEN_AT_UTC,
        "genesis_source_cutoff_inclusive": GENESIS_SOURCE_CUTOFF.strftime("%Y-%m-%d"),
        "decision_cutoff_inclusive": DECISION_CUTOFF.strftime("%Y-%m-%d"),
        "independence_rule": "only dec_dt strictly after the cutoff; no pre-cutoff trade can enter",
        "accrual": {
            "start": "first observed official A-share session strictly after decision_cutoff_inclusive",
            "sessions": ACCRUAL_SESSIONS,
            "enrollment": "all production delay5 decisions in the first 60 post-freeze sessions",
            "close": "after the 60th post-freeze session and strict-next-session fill observability",
        },
        "production_rules": production["rules"],
        "matching": {
            **plan["design"]["matching"]["primary"],
            "forward_transport": (
                "no refit: apply frozen feature scales and all+2024plus entropy coefficients to every new "
                "treated trade's eligible controls, then normalize within trade"
            ),
            "required_features": required_features,
            "feature_scale": entropy["feature_scale"],
            "coefficients": entropy["coefficients"],
            "future_exact_mcap": (
                "archive one complete daily_basic cross-section on each enrolled decision date before outcome access"
            ),
        },
        "outcome_lock": {
            "outcomes_allowed_during_progress": False,
            "unlock_condition": (
                "accrual is closed, next-session fill is observable for every enrolled decision, and every enrolled "
                "filled trade has its fixed 60th common session observed"
            ),
            "horizons_sessions": list(HORIZONS),
            "entry_session_is_day_one": True,
            "endpoint_offsets": {str(horizon): horizon - 1 for horizon in HORIZONS},
            "path_policy": plan["design"]["outcome_design"]["path_policy"],
            "costs": plan["design"]["outcome_design"]["costs"],
        },
        "inference": {
            "primary_estimand": "equal-weight forward treated-trade ATT",
            "hac_lags": {str(key): value for key, value in HAC_LAGS.items()},
            "stationary_bootstrap_expected_blocks": {str(key): value for key, value in BOOTSTRAP_BLOCKS.items()},
            "bootstrap_replications": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "multiple_testing": "Holm-adjusted one-sided H5/H20/H60 family",
            "network_inference": "treated_symbol × control_symbol two-way cluster sandwich",
            "success_gate": (
                "positive ATT and HAC-Holm<0.05 and stationary-bootstrap-Holm<0.05 and two-way-cluster p<0.05"
            ),
            "no_horizon_cherry_picking": True,
        },
        "source_identity": {
            "git_head_at_freeze": "dc35eed1b2e7b296d3fbbc1ebd459bb0f4a60b04",
            "forward_audit_script": _record(Path(__file__)),
            "pit_plan": _record(pit_plan.EXPECTED_PLAN_PATH),
            "balance_audit": _record(BALANCE_AUDIT_PATH),
            "balance_weights": _record(BALANCE_PATH),
            "historical_outcome_audit": _record(OUTCOME_AUDIT_PATH),
            "production_audit_genesis": _record(PRODUCTION_AUDIT_PATH),
            "candidate_manifest_genesis": _record(CANDIDATE_MANIFEST_PATH),
        },
        "data_capture": {
            "required_before_next_session": [
                "content-addressed causal decision surface",
                "complete exact daily_basic cross-section",
                "production membership and fill decision",
            ],
            "late_backfill_policy": "reported separately and excluded from the primary forward sample",
            "external_anchor": False,
        },
        "authorization": {
            "outcome_access_locked": True,
            "live_authorized": False,
        },
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """协议或执行脚本一旦漂移即停止。"""

    required = (
        protocol.get("schema") == SCHEMA,
        protocol.get("frozen_at_utc") == FROZEN_AT_UTC,
        protocol.get("genesis_source_cutoff_inclusive") == GENESIS_SOURCE_CUTOFF.strftime("%Y-%m-%d"),
        protocol.get("decision_cutoff_inclusive") == DECISION_CUTOFF.strftime("%Y-%m-%d"),
        protocol.get("accrual", {}).get("sessions") == ACCRUAL_SESSIONS,
        protocol.get("outcome_lock", {}).get("outcomes_allowed_during_progress") is False,
        protocol.get("inference", {}).get("no_horizon_cherry_picking") is True,
        protocol.get("authorization") == {"outcome_access_locked": True, "live_authorized": False},
        protocol.get("source_identity", {}).get("forward_audit_script", {}).get("sha256")
        == identity_utils.sha256_file(Path(__file__)),
    )
    if not all(required):
        raise RuntimeError("forward protocol is incomplete or drifted")


def build_progress(
    protocol: Mapping[str, Any],
    market_dates: Sequence[Any],
    cohort: pd.DataFrame,
) -> dict[str, Any]:
    """仅根据官方日历和阶段字段判断 accrual/maturity，不接触收益。"""

    validate_protocol(protocol)
    sessions = pd.DatetimeIndex(pd.to_datetime(list(market_dates))).normalize().drop_duplicates().sort_values()
    post_sessions = sessions[sessions > DECISION_CUTOFF]
    accrual = post_sessions[:ACCRUAL_SESSIONS]
    frame = cohort.copy()
    for column in ("sig_dt", "dec_dt", "entry_dt", "entry_fill_dt", "common_h60_dt"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    forward = frame[frame["dec_dt"].gt(DECISION_CUTOFF)].copy()
    enrolled = forward[forward["dec_dt"].isin(accrual)].copy()
    accrual_closed = len(post_sessions) >= ACCRUAL_SESSIONS
    fill_observable = bool(
        accrual_closed
        and len(post_sessions) > ACCRUAL_SESSIONS
        and (enrolled["dec_dt"] < post_sessions[ACCRUAL_SESSIONS]).all()
    )
    filled = enrolled[enrolled["stage_fill"].astype(bool)].copy()
    mature = filled[filled["stage_mature"].astype(bool)].copy()
    all_filled_mature = bool(fill_observable and len(filled) > 0 and len(mature) == len(filled))
    if len(post_sessions) == 0:
        status = "WAITING_FOR_FIRST_POST_FREEZE_SESSION"
    elif not accrual_closed:
        status = "ACCRUING_FORWARD_DECISIONS"
    elif not fill_observable:
        status = "WAITING_FOR_FINAL_NEXT_SESSION_FILL_OBSERVABILITY"
    elif len(filled) == 0:
        status = "FORWARD_WINDOW_CLOSED_WITH_NO_FILLED_TRADES"
    elif not all_filled_mature:
        status = "WAITING_FOR_ENROLLED_H60_MATURITY"
    else:
        status = "READY_FOR_ONE_SHOT_OUTCOME_EVALUATION"
    stage_counts = {
        name: int(enrolled[column].fillna(False).astype(bool).sum())
        for name, column in (
            ("raw", "stage_raw"),
            ("gate", "stage_gate"),
            ("hard", "stage_hard"),
            ("st", "stage_st"),
            ("market", "stage_market"),
            ("fill", "stage_fill"),
            ("mature", "stage_mature"),
        )
    }
    records = [
        [
            str(row.symbol),
            row.sig_dt.strftime("%Y-%m-%d"),
            row.dec_dt.strftime("%Y-%m-%d"),
            bool(row.stage_fill),
            row.entry_fill_dt.strftime("%Y-%m-%d") if pd.notna(row.entry_fill_dt) else None,
            row.common_h60_dt.strftime("%Y-%m-%d") if pd.notna(row.common_h60_dt) else None,
            bool(row.stage_mature),
        ]
        for row in enrolled.sort_values(["dec_dt", "symbol"], kind="mergesort").itertuples(index=False)
    ]
    return {
        "status": status,
        "post_freeze_sessions_observed": int(len(post_sessions)),
        "accrual_sessions_required": ACCRUAL_SESSIONS,
        "accrual_sessions_observed": int(len(accrual)),
        "accrual_start": accrual.min().strftime("%Y-%m-%d") if len(accrual) else None,
        "accrual_end": accrual.max().strftime("%Y-%m-%d") if len(accrual) == ACCRUAL_SESSIONS else None,
        "latest_observed_session": sessions.max().strftime("%Y-%m-%d") if len(sessions) else None,
        "stage_counts": stage_counts,
        "enrolled_identity": {"rows": len(records), "sha256": _canonical_sha256(records)},
        "outcome_evaluation_permitted": bool(all_filled_mature),
        "outcomes_loaded": False,
        "live_authorized": False,
    }


def run_status() -> dict[str, Any]:
    """验证当前原始完整性并写出结果盲进度状态。"""

    protocol = _load_json(PROTOCOL_PATH, "tracked forward protocol")
    validate_protocol(protocol)
    candidate_manifest = _load_json(CANDIDATE_MANIFEST_PATH, "current candidate manifest")
    production = _load_json(PRODUCTION_AUDIT_PATH, "current production audit")
    if candidate_manifest.get("schema") != "surge_candidates_dump_manifest_v2":
        raise RuntimeError("current candidate manifest schema drift")
    if candidate_manifest.get("raw_completeness_proven") is not True:
        raise RuntimeError("current raw candidate completeness is not proven")
    if production.get("schema") != "surge_delay5_production_cohort_audit_v3":
        raise RuntimeError("current production audit schema drift")
    if production.get("raw_completeness_proven") is not True:
        raise RuntimeError("current production raw completeness is not proven")
    if production.get("design", {}).get("outcome_fields_used_for_cohort") is not False:
        raise RuntimeError("current production cohort used outcome fields")
    cohort = pd.read_parquet(COHORT_PATH, columns=list(FORWARD_COHORT_COLUMNS))
    market = pd.read_parquet(MARKET_PATH, columns=["dt"])
    progress = build_progress(protocol, market["dt"], cohort)
    result = {
        "schema": STATUS_SCHEMA,
        "protocol": _record(PROTOCOL_PATH),
        "current_inputs": {
            "candidate_manifest": _record(CANDIDATE_MANIFEST_PATH),
            "production_audit": _record(PRODUCTION_AUDIT_PATH),
            "cohort": {**_record(COHORT_PATH), "rows": int(len(cohort))},
            "market_state": _record(MARKET_PATH),
        },
        "progress": progress,
        "limitations": [
            "No post-freeze session can be evaluated before it exists and is captured.",
            "The local protocol has no external timestamp anchor; source artifacts must be content-addressed each session.",
            "A 60-session accrual window can still be statistically underpowered if few production trades fill.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protocol", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        if PROTOCOL_PATH.exists():
            raise FileExistsError(f"forward protocol already exists and is immutable: {PROTOCOL_PATH}")
        protocol = build_protocol()
        PROTOCOL_PATH.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"schema": protocol["schema"], "path": str(PROTOCOL_PATH)}, ensure_ascii=False, indent=2))
        return
    result = run_status()
    print(json.dumps(result["progress"], ensure_ascii=False, indent=2))
    print(f"[output] {STATUS_PATH}")


if __name__ == "__main__":
    main()
