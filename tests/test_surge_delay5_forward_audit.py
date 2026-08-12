"""Tests for the frozen Delay5 60-session forward monitor."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_delay5_forward_audit as audit  # noqa: E402


def _protocol() -> dict:
    return {
        "schema": audit.SCHEMA,
        "frozen_at_utc": audit.FROZEN_AT_UTC,
        "genesis_source_cutoff_inclusive": audit.GENESIS_SOURCE_CUTOFF.strftime("%Y-%m-%d"),
        "decision_cutoff_inclusive": audit.DECISION_CUTOFF.strftime("%Y-%m-%d"),
        "accrual": {"sessions": audit.ACCRUAL_SESSIONS},
        "outcome_lock": {"outcomes_allowed_during_progress": False},
        "inference": {"no_horizon_cherry_picking": True},
        "authorization": {"outcome_access_locked": True, "live_authorized": False},
        "source_identity": {"forward_audit_script": {"sha256": audit.identity_utils.sha256_file(Path(audit.__file__))}},
    }


def _cohort(decisions: list[pd.Timestamp], *, filled: bool = False, mature: bool = False) -> pd.DataFrame:
    rows = []
    for index, decision in enumerate(decisions):
        rows.append(
            {
                "symbol": f"T{index}",
                "sig_dt": decision - pd.Timedelta(days=5),
                "dec_dt": decision,
                "entry_dt": decision + pd.Timedelta(days=1),
                "entry_fill_dt": decision + pd.Timedelta(days=1) if filled else pd.NaT,
                "common_h60_dt": decision + pd.Timedelta(days=90) if filled else pd.NaT,
                "stage_raw": True,
                "stage_gate": filled,
                "stage_hard": filled,
                "stage_st": filled,
                "stage_market": filled,
                "stage_fill": filled,
                "stage_mature": mature,
            }
        )
    return pd.DataFrame(rows, columns=audit.FORWARD_COHORT_COLUMNS)


def test_no_post_freeze_session_keeps_outcomes_locked() -> None:
    result = audit.build_progress(_protocol(), [audit.DECISION_CUTOFF], _cohort([]))
    assert result["status"] == "WAITING_FOR_FIRST_POST_FREEZE_SESSION"
    assert result["outcome_evaluation_permitted"] is False
    assert result["outcomes_loaded"] is False


def test_first_sixty_sessions_are_the_only_enrollment_window() -> None:
    sessions = pd.bdate_range(audit.DECISION_CUTOFF + pd.Timedelta(days=1), periods=70)
    decisions = [sessions[0], sessions[59], sessions[60]]
    result = audit.build_progress(_protocol(), sessions, _cohort(decisions))

    assert result["accrual_start"] == sessions[0].strftime("%Y-%m-%d")
    assert result["accrual_end"] == sessions[59].strftime("%Y-%m-%d")
    assert result["enrolled_identity"]["rows"] == 2
    assert result["stage_counts"]["raw"] == 2
    assert result["status"] == "FORWARD_WINDOW_CLOSED_WITH_NO_FILLED_TRADES"


def test_accrual_and_h60_maturity_are_separate_locks() -> None:
    sessions = pd.bdate_range(audit.DECISION_CUTOFF + pd.Timedelta(days=1), periods=61)
    decision = sessions[0]
    waiting = audit.build_progress(_protocol(), sessions, _cohort([decision], filled=True, mature=False))
    ready = audit.build_progress(_protocol(), sessions, _cohort([decision], filled=True, mature=True))

    assert waiting["status"] == "WAITING_FOR_ENROLLED_H60_MATURITY"
    assert waiting["outcome_evaluation_permitted"] is False
    assert ready["status"] == "READY_FOR_ONE_SHOT_OUTCOME_EVALUATION"
    assert ready["outcome_evaluation_permitted"] is True


def test_progress_identity_is_order_independent() -> None:
    sessions = pd.bdate_range(audit.DECISION_CUTOFF + pd.Timedelta(days=1), periods=30)
    cohort = _cohort([sessions[2], sessions[5]])
    first = audit.build_progress(_protocol(), sessions, cohort)
    second = audit.build_progress(_protocol(), sessions[::-1], cohort.iloc[::-1])
    assert first["enrolled_identity"] == second["enrolled_identity"]


def test_protocol_drift_closes_progress() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["accrual"]["sessions"] = 59
    try:
        audit.validate_protocol(protocol)
    except RuntimeError as exc:
        assert "drifted" in str(exc)
    else:
        raise AssertionError("protocol drift was not rejected")


def test_progress_projection_contains_no_outcome_columns() -> None:
    forbidden = ("gross", "net", "return", "exit")
    assert not any(any(token in column.lower() for token in forbidden) for column in audit.FORWARD_COHORT_COLUMNS)
