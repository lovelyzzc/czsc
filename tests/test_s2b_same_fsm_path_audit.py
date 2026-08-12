"""S2b 同 FULL-FSM 与逐日路径审计的纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_same_fsm_path_audit as audit  # noqa: E402


def _state(*, idx: int = 0, next_open: float = 100.0, sl_ref: float = np.nan) -> SimpleNamespace:
    return SimpleNamespace(idx=idx, next_open=next_open, sl_ref=sl_ref)


def _arrays(close_at_exit: float, *, low_at_exit: float = 100.0) -> dict[str, object]:
    return {
        "opens": np.array([99.0, 100.0, 95.0, 96.0]),
        "closes": np.array([99.0, 110.0, close_at_exit, 96.0]),
        "lows": np.array([98.0, 70.0, low_at_exit, 90.0]),
        "dates": pd.date_range("2025-01-01", periods=4, freq="D"),
        "regime_by_idx": {2: 9},
        "max_hold_days": 3,
    }


def test_full_exit_priority_is_stop_then_trail_then_state() -> None:
    stop = audit.simulate_full_exit(_state(sl_ref=90.0), **_arrays(70.0, low_at_exit=80.0))
    assert stop is not None
    assert stop.reason == "sl2"
    assert stop.exit_price == 90.0

    trail = audit.simulate_full_exit(_state(), **_arrays(70.0, low_at_exit=95.0))
    assert trail is not None
    assert trail.reason == "trail18"
    assert trail.exit_price == 70.0

    state = audit.simulate_full_exit(_state(), **_arrays(95.0, low_at_exit=95.0))
    assert state is not None
    assert state.reason == "state"


def test_state_exit_records_signal_day_and_next_open_fill_day() -> None:
    result = audit.simulate_full_exit(_state(), **_arrays(95.0, low_at_exit=95.0))
    assert result is not None
    assert result.reason == "state"
    assert result.exit_signal_idx == 2
    assert result.exit_fill_idx == 3
    assert result.exit_signal_dt == pd.Timestamp("2025-01-03")
    assert result.exit_fill_dt == pd.Timestamp("2025-01-04")
    assert result.exit_price == 96.0
    assert result.hold_signal_bars == 1
    assert result.hold_fill_bars == 2


def test_path_metrics_cover_excursions_limit_streak_and_max_leg() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "dt": dates,
            "open": [99.0, 100.0, 105.0, 115.0, 100.0],
            "high": [100.0, 105.0, 125.0, 115.0, 112.0],
            "low": [98.0, 95.0, 98.0, 90.0, 92.0],
            "close": [99.0, 105.0, 115.0, 100.0, 110.0],
            "pct_chg": [0.0, 9.9, 9.9, -5.0, 5.0],
        }
    )
    result = audit.simulate_full_exit(
        _state(),
        opens=frame["open"].to_numpy(),
        closes=frame["close"].to_numpy(),
        lows=frame["low"].to_numpy(),
        dates=dates,
        regime_by_idx={},
        max_hold_days=4,
    )
    assert result is not None
    assert result.reason == "max_hold"
    metrics = audit.compute_path_metrics(
        frame,
        result,
        decision_idx=0,
        limit_threshold_pct=9.8,
    )
    assert metrics["mfe_pct"] == pytest.approx(25.0)
    assert metrics["mae_pct"] == pytest.approx(-10.0)
    assert metrics["time_to_mfe_bars"] == 1
    assert metrics["time_to_mae_bars"] == 2
    assert metrics["limit_up_days_held"] == 2
    assert metrics["max_limit_up_streak_held"] == 2
    assert metrics["max_single_leg_contribution_pct"] == pytest.approx(10.0)
    assert metrics["max_single_leg_share_of_gross_pct"] == pytest.approx(100.0)
    assert audit.maximum_true_streak([False, True, True, False, True]) == 2


def test_validate_annual_binding_checks_schema_sha_rows_year_and_warning() -> None:
    annual = {
        "schema": audit.ANNUAL_SCHEMA,
        "design": {
            "year": 2025,
            "outcome_warning": (
                "Treated trades use path-dependent FULL exits; future outcome availability is required."
            ),
        },
        "trade_output": {"sha256": "abc", "rows": 117},
        "coverage": {"exact_supported_trades": 117},
        "verdict": {"status": "BOUND"},
    }
    assert audit.validate_annual_binding(annual, trade_sha256="abc", trade_rows=117) == "BOUND"
    with pytest.raises(RuntimeError, match="sha256"):
        audit.validate_annual_binding(annual, trade_sha256="different", trade_rows=117)
    with pytest.raises(RuntimeError, match="rows mismatch"):
        audit.validate_annual_binding(annual, trade_sha256="abc", trade_rows=116)
    missing_warning = {**annual, "design": {"year": 2025, "outcome_warning": "path-dependent FULL exits"}}
    with pytest.raises(RuntimeError, match="outcome warning"):
        audit.validate_annual_binding(missing_warning, trade_sha256="abc", trade_rows=117)


def test_validate_trade_frame_reconciles_fixed_exit_excess() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "dec_dt": pd.to_datetime(["2025-01-02"]),
            "entry_dt": pd.to_datetime(["2025-01-03"]),
            "exit_dt": pd.to_datetime(["2025-01-10"]),
            "ret_gross_pct": [10.0],
            "hold_days": [5],
            "exit_reason": ["state"],
            "exact_excess_pct": [8.0],
            "exact_control_median_pct": [2.0],
            "exact_caliper_control_symbols": [["a", "b", "c", "d", "e"]],
        }
    )
    audit.validate_trade_frame(frame)
    broken = frame.copy()
    broken["exact_excess_pct"] = 7.0
    with pytest.raises(RuntimeError, match="no longer reconciles"):
        audit.validate_trade_frame(broken)


def test_decision_lookback_is_exactly_20_bars_including_decision_day() -> None:
    dates = pd.date_range("2025-01-01", periods=22, freq="D")
    frame = pd.DataFrame(
        {
            "dt": dates,
            "open": [100.0] * 22,
            "high": [101.0] * 22,
            "low": [99.0] * 22,
            "close": [100.0] * 22,
            "pct_chg": [9.9, 9.9] + [0.0] * 20,
        }
    )
    result = audit.FullExitResult(
        entry_idx=21,
        entry_dt=dates[21],
        entry_price=100.0,
        exit_signal_idx=21,
        exit_signal_dt=dates[21],
        exit_fill_idx=21,
        exit_fill_dt=dates[21],
        exit_price=100.0,
        reason="max_hold",
    )

    metrics = audit.compute_path_metrics(frame, result, decision_idx=20, limit_threshold_pct=9.8)

    assert audit.DECISION_LOOKBACK_BARS == 20
    assert metrics["decision_lookback_20_inclusive_limit_up_days"] == 1


def test_verdict_preserves_only_with_complete_support_and_positive_frozen_top5() -> None:
    comparison = pd.DataFrame(
        {
            "is_top5": [True, True, True, True, True, False],
            "same_fsm_excess_pct": [5.0, 4.0, 3.0, 2.0, 1.0, 5.0],
        }
    )

    verdict = audit.derive_same_fsm_verdict(comparison)

    assert verdict["status"] == "SAME_FSM_SENSITIVITY_PRESERVES_FROZEN_TOP5_POSITIVITY"
    assert verdict["complete_same_fsm_support"] is True
    assert verdict["same_fsm_preserves_frozen_top5_positivity"] is True
    assert verdict["same_fsm_removes_frozen_top5_positivity"] is False
    assert verdict["frozen_top5_share_of_same_fsm_excess_sum_pct"] == pytest.approx(75.0)


def test_verdict_reports_removal_and_fails_closed_on_incomplete_support() -> None:
    comparison = pd.DataFrame(
        {
            "is_top5": [True, True, True, True, True, False],
            "same_fsm_excess_pct": [5.0, 4.0, 3.0, 2.0, -0.1, 1.0],
        }
    )

    removed = audit.derive_same_fsm_verdict(comparison)
    assert removed["status"] == "SAME_FSM_SENSITIVITY_DOES_NOT_PRESERVE_FROZEN_TOP5_POSITIVITY"
    assert removed["same_fsm_preserves_frozen_top5_positivity"] is False
    assert removed["same_fsm_removes_frozen_top5_positivity"] is True

    comparison.loc[5, "same_fsm_excess_pct"] = np.nan
    incomplete = audit.derive_same_fsm_verdict(comparison)
    assert incomplete["status"] == "SAME_FSM_SENSITIVITY_INCOMPLETE_SUPPORT"
    assert incomplete["complete_same_fsm_support"] is False
    assert incomplete["same_fsm_preserves_frozen_top5_positivity"] is False
