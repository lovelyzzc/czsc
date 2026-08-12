"""Tests for the research-stage temporal stability diagnostic."""

from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_pit_temporal_stability_audit as audit  # noqa: E402


def test_fixed_era_boundaries_are_calendar_based() -> None:
    dates = pd.to_datetime(["2021-08-01", "2022-12-31", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"])
    assert audit.assign_era(dates).tolist() == [
        "2021-2022",
        "2021-2022",
        "2023",
        "2024",
        "2025",
        "2026-partial",
    ]


def test_leave_one_era_out_uses_every_other_trade() -> None:
    rows = []
    for index, era in enumerate(audit.ERAS):
        row = {"era": era}
        for horizon in audit.HORIZONS:
            row[f"att_weighted_net_h{horizon}_pct"] = float(index + horizon)
        rows.append(row)
    result = audit.leave_one_era_out(pd.DataFrame(rows))
    assert result["2021-2022"]["n_trades"] == 4
    assert result["2021-2022"]["mean_att_pct"]["5"] == pytest.approx((6 + 7 + 8 + 9) / 4)


def test_input_identity_fails_closed_on_artifact_drift() -> None:
    historical = audit._load_json(audit.OUTCOME_AUDIT_PATH, "historical")
    audit.validate_inputs(historical)
    drifted = copy.deepcopy(historical)
    drifted["outputs"]["trade_att"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="artifact drift"):
        audit.validate_inputs(drifted)


def test_temporal_script_cannot_rematch_or_authorize_live() -> None:
    source = inspect.getsource(audit)
    forbidden = ("build_exact_matches(", "fit_conditional_entropy_weights(", "exact_mcap_caliper_controls(")
    assert not any(token in source for token in forbidden)
    assert '"matching_refit": False' in source
    assert '"weight_refit": False' in source
    assert '"live_authorized": False' in source
