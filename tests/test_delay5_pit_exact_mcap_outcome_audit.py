"""Tests for the authorized, frozen-pair exact-mcap outcome stage."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_pit_exact_mcap_outcome_audit as audit  # noqa: E402


def _synthetic_outcome_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = pd.DataFrame(
        {
            "trade_id": ["T|2025-01-02"],
            "symbol": ["T"],
            "dec_dt": pd.to_datetime(["2025-01-02"]),
            "entry_dt": pd.to_datetime(["2025-01-03"]),
        }
    )
    for horizon in audit.HORIZONS:
        for basis in ("gross", "net"):
            treated[f"{basis}_h{horizon}_pct"] = 3.0
            treated[f"{basis}_pessimistic_h{horizon}_pct"] = 2.0
            treated[f"{basis}_optimistic_h{horizon}_pct"] = 3.0
    pairs = pd.DataFrame(
        {
            "trade_id": ["T|2025-01-02", "T|2025-01-02"],
            "control_weight": [0.75, 0.25],
            "control_filled": [True, True],
        }
    )
    for horizon in audit.HORIZONS:
        pairs[f"terminal_h{horizon}"] = False
        for basis in ("gross", "net"):
            pairs[f"{basis}_h{horizon}_pct"] = [0.0, 4.0]
            pairs[f"{basis}_pessimistic_h{horizon}_pct"] = [0.0, 4.0]
            pairs[f"{basis}_optimistic_h{horizon}_pct"] = [0.0, 4.0]
    return treated, pairs


def test_outcome_authorization_fails_when_balance_gate_is_not_passed() -> None:
    plan = {"design": {}}
    balance = {"schema": audit.EXPECTED_BALANCE_SCHEMA, "verdict": {"outcome_evaluation_permitted": False}}
    with pytest.raises(RuntimeError):
        audit.validate_outcome_authorization(plan, balance)


def test_frozen_pairs_require_unique_normalized_nonnegative_weights() -> None:
    pairs = pd.DataFrame(
        {
            "trade_id": ["T1", "T1"],
            "treated_symbol": ["T", "T"],
            "control_symbol": ["C1", "C2"],
            "dec_dt": pd.to_datetime(["2025-01-02"] * 2),
            "year": [2025, 2025],
            "rank": [1, 2],
            "support_ok": [True, True],
            "control_weight": [0.75, 0.25],
        }
    )
    audit.validate_frozen_pairs(pairs, 2)
    drifted = pairs.copy()
    drifted.loc[0, "control_weight"] = 0.70
    with pytest.raises(RuntimeError, match="sum to one"):
        audit.validate_frozen_pairs(drifted, 2)


def test_trade_att_uses_frozen_weights_and_reports_equal_weight_sensitivities() -> None:
    treated, pairs = _synthetic_outcome_inputs()
    result = audit.build_trade_att(treated, pairs)

    assert result.loc[0, "control_weighted_net_h20_pct"] == pytest.approx(1.0)
    assert result.loc[0, "att_weighted_net_h20_pct"] == pytest.approx(2.0)
    assert result.loc[0, "att_equal_mean_net_h20_pct"] == pytest.approx(1.0)
    assert result.loc[0, "att_median_net_h20_pct"] == pytest.approx(1.0)
    assert result.loc[0, "att_weighted_net_h20_lower_pct"] == pytest.approx(1.0)


def test_ratio_influence_hac_preserves_trade_weighted_mean_on_complete_calendar() -> None:
    calendar = pd.bdate_range("2025-01-02", periods=12)
    frame = pd.DataFrame(
        {
            "dec_dt": [calendar[0], calendar[0], calendar[5]],
            "att": [1.0, 3.0, 8.0],
        }
    )
    result = audit.ratio_influence_hac(frame, "att", calendar, lag=4)

    assert result["mean_att_pct"] == pytest.approx(4.0)
    assert result["n_trades"] == 3
    assert result["n_decision_dates"] == 2
    assert result["calendar_sessions"] == 6
    assert result["hac_lag"] == 4


def test_stationary_ratio_bootstrap_is_seeded_and_uses_ratio_not_daily_mean() -> None:
    calendar = pd.bdate_range("2025-01-02", periods=30)
    frame = pd.DataFrame(
        {
            "dec_dt": np.repeat(calendar[::3], 2),
            "att": np.tile([1.0, 3.0], len(calendar[::3])),
        }
    )
    first = audit.stationary_ratio_bootstrap(frame, "att", calendar, expected_block=5, n_boot=300, seed=42)
    second = audit.stationary_ratio_bootstrap(frame, "att", calendar, expected_block=5, n_boot=300, seed=42)

    assert first == second
    assert first["mean_att_pct"] == pytest.approx(2.0)
    assert first["expected_block_sessions"] == 5
    assert first["n_boot_valid"] == 300


def test_holm_adjustment_uses_all_three_named_horizons() -> None:
    adjusted = audit.holm_adjust({"5": 0.01, "20": 0.04, "60": 0.03})
    assert adjusted == pytest.approx({"5": 0.03, "20": 0.06, "60": 0.06})


def test_two_way_cluster_estimate_matches_weighted_trade_att() -> None:
    rows = []
    specifications = [
        ("T1", "A", 2.0, [("C1", 1.0), ("C2", 3.0)]),
        ("T2", "B", 5.0, [("C1", 2.0), ("C3", 4.0)]),
        ("T3", "C", 4.0, [("C2", 1.0), ("C3", 2.0)]),
    ]
    for trade_id, treated_symbol, treated_value, controls in specifications:
        for control_symbol, control_value in controls:
            rows.append(
                {
                    "trade_id": trade_id,
                    "treated_symbol": treated_symbol,
                    "control_symbol": control_symbol,
                    "control_weight": 0.5,
                    "treated": treated_value,
                    "control": control_value,
                }
            )
    result = audit.two_way_symbol_cluster(pd.DataFrame(rows), treated_column="treated", control_column="control")

    expected = np.mean([0.0, 2.0, 2.5])
    assert result["mean_att_pct"] == pytest.approx(expected)
    assert result["treated_symbol_clusters"] == 3
    assert result["control_symbol_clusters"] == 3


def test_outcome_script_cannot_rematch_or_mutate_frozen_control_weights() -> None:
    source = inspect.getsource(audit)
    forbidden = ("build_exact_matches(", "fit_conditional_entropy_weights(", "exact_mcap_caliper_controls(")
    assert not any(token in source for token in forbidden)
    assert "FROZEN_PAIRS_PATH" in source
    assert "control_weight" in source


def test_live_authorization_is_hard_false_in_outcome_source() -> None:
    source = inspect.getsource(audit)
    assert '"live_authorized": False' in source
    assert "forward_validation_required" in source
