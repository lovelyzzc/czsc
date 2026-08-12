"""Production delay5 cohort 漏斗的纯函数与边界测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_delay5_production_cohort_audit as audit  # noqa: E402


def _base_frame(rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sig_vol_ratio": [0.8] * rows,
            "sig_ma_spread_pct": [3.0] * rows,
            "sig_ret20": [8.0] * rows,
            "amount_e": [1.0] * rows,
            "sl_pct": [8.0] * rows,
            "high20_ratio": [0.12001] * rows,
            "ew_index_above_ma20": [1.0] * rows,
            "gap_pct": [9.49] * rows,
            "limit_pct": [9.8] * rows,
            "exit_reason": ["state"] * rows,
            "hold_days": [10] * rows,
            "state_fill_observed": [True] * rows,
        }
    )


def test_signal_gate_uses_current_inclusive_boundaries_and_ignores_above_zg() -> None:
    frame = pd.DataFrame(
        {
            "sig_vol_ratio": [0.8, 0.80001, 0.8, 0.8, 0.8],
            "sig_ma_spread_pct": [3.0, 3.0, 2.999, 3.0, 3.0],
            "sig_ret20": [8.0, 8.0, 8.0, 7.999, 8.0],
            "sig_above_zg": [0, 1, 1, 1, 0],
        }
    )

    assert audit.signal_gate_mask(frame).tolist() == [True, False, False, False, True]


def test_hard_market_and_fill_boundaries() -> None:
    hard = pd.DataFrame(
        {
            "amount_e": [1.0, 0.999, 1.0, 1.0, 1.0],
            "sl_pct": [8.0, 8.0, 20.0, 7.999, 20.001],
        }
    )
    market = pd.DataFrame(
        {
            "high20_ratio": [0.12001, 0.12, 0.12001, 0.13],
            "ew_index_above_ma20": [1.0, 1.0, 0.0, -1.0],
        }
    )
    fill = pd.DataFrame({"gap_pct": [9.499, 9.5, 19.499, 19.5], "limit_pct": [9.8, 9.8, 19.8, 19.8]})

    assert audit.hard_rule_mask(hard).tolist() == [True, False, True, False, False]
    assert audit.market_rule_mask(market).tolist() == [True, False, False, False]
    assert audit.fill_rule_mask(fill).tolist() == [True, False, True, False]


def test_mature_requires_max_hold_completion_and_observable_state_fill() -> None:
    frame = pd.DataFrame(
        {
            "exit_reason": ["max_hold", "max_hold", "state", "state", "trail18"],
            "hold_days": [58, 59, 1, 1, 2],
            "state_fill_observed": pd.array([pd.NA, pd.NA, False, True, pd.NA], dtype="boolean"),
        }
    )

    assert audit.mature_rule_mask(frame).tolist() == [False, True, False, True, True]


def test_stage_flags_are_cumulative_and_record_first_failure() -> None:
    frame = pd.concat([_base_frame() for _ in range(7)], ignore_index=True)
    frame.loc[0, "sig_vol_ratio"] = 0.81
    frame.loc[1, "amount_e"] = 0.99
    frame.loc[3, "high20_ratio"] = 0.12
    frame.loc[4, "gap_pct"] = 9.5
    frame.loc[5, ["exit_reason", "hold_days"]] = ["max_hold", 58]
    frame.loc[6, ["exit_reason", "hold_days"]] = ["max_hold", 59]
    non_st = pd.Series([True, True, False, True, True, True, True])

    result = audit.apply_stage_flags(frame, non_st)

    assert result["first_failed_stage"].tolist() == ["gate", "hard", "st", "market", "fill", "mature", pd.NA]
    assert audit.funnel_counts(result) == {
        "raw": 7,
        "gate": 6,
        "hard": 5,
        "st": 4,
        "market": 3,
        "fill": 2,
        "mature": 1,
    }


def test_validate_raw_rejects_duplicate_identity() -> None:
    row = {
        "symbol": "000001.SZ",
        "mode": "anticipate",
        "delay": 5,
        "dec_regime": 5,
        "sig_dt": pd.Timestamp("2026-01-01"),
        "dec_dt": pd.Timestamp("2026-01-08"),
        "entry_dt": pd.Timestamp("2026-01-09"),
        "exit_dt": pd.Timestamp("2026-01-10"),
        "exit_reason": "state",
        "hold_days": 1,
        "amount_e": 1.0,
        "sl_pct": 8.0,
        "gap_pct": 0.0,
        "limit_pct": 9.8,
        "sig_vol_ratio": 0.8,
        "sig_ma_spread_pct": 3.0,
        "sig_ret20": 8.0,
    }
    frame = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="not unique"):
        audit.validate_raw(frame)

    invalid_timing = frame.iloc[[0]].copy()
    invalid_timing["entry_dt"] = invalid_timing["dec_dt"]
    with pytest.raises(ValueError, match="sig_dt < dec_dt < entry_dt <= exit_dt"):
        audit.validate_raw(invalid_timing)
    invalid_signal_date = frame.iloc[[0]].copy()
    invalid_signal_date["sig_dt"] = invalid_signal_date["dec_dt"]
    with pytest.raises(ValueError, match="sig_dt < dec_dt < entry_dt <= exit_dt"):
        audit.validate_raw(invalid_signal_date)

    invalid_exit_date = frame.iloc[[0]].copy()
    invalid_exit_date["exit_dt"] = invalid_exit_date["dec_dt"]
    with pytest.raises(ValueError, match="sig_dt < dec_dt < entry_dt <= exit_dt"):
        audit.validate_raw(invalid_exit_date)

    invalid_regime = frame.iloc[[0]].copy()
    invalid_regime["dec_regime"] = 9
    with pytest.raises(ValueError, match="invalid decision regimes"):
        audit.validate_raw(invalid_regime)


def test_state_fill_observability_uses_next_symbol_row_and_fails_closed_at_tail() -> None:
    dates = pd.bdate_range("2026-08-07", periods=2)
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "exit_dt": [dates[0], dates[0], dates[0]],
            "exit_reason": ["state", "state", "state"],
            "hold_days": [1, 1, 1],
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ", "000003.SZ", "000003.SZ"],
            "dt": [dates[0], dates[1], dates[0], dates[0], dates[1]],
            "open": [10.0, 10.5, 20.0, 30.0, float("nan")],
        }
    )

    result = audit.attach_state_fill_observability(frame, panel)

    assert result["state_fill_observed"].tolist() == [True, False, False]
    assert result.loc[0, "state_fill_dt"] == dates[1]
    assert result.loc[0, "state_fill_open"] == pytest.approx(10.5)
    assert audit.mature_rule_mask(result).tolist() == [True, False, False]


def test_state_fill_observability_requires_trigger_row_and_unique_panel() -> None:
    date = pd.Timestamp("2026-08-10")
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "exit_dt": [date], "exit_reason": ["state"], "hold_days": [1]})
    missing = pd.DataFrame({"symbol": ["000002.SZ"], "dt": [date], "open": [10.0]})
    duplicate = pd.DataFrame({"symbol": ["000001.SZ", "000001.SZ"], "dt": [date, date], "open": [10.0, 10.0]})

    with pytest.raises(ValueError, match="trigger is absent"):
        audit.attach_state_fill_observability(frame, missing)
    with pytest.raises(ValueError, match="not unique"):
        audit.attach_state_fill_observability(frame, duplicate)


def test_market_merge_requires_many_to_one_and_complete_dates() -> None:
    raw = pd.DataFrame({"dec_dt": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    duplicate_market = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "high20_ratio": [0.2, 0.2],
            "ew_index_above_ma20": [1.0, 1.0],
        }
    )
    incomplete_market = duplicate_market.iloc[:1].copy()

    with pytest.raises(pd.errors.MergeError):
        audit.merge_market_state(raw, duplicate_market)
    with pytest.raises(ValueError, match="missing decision dates"):
        audit.merge_market_state(raw, incomplete_market)


def test_required_st_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="historical ST"):
        audit.require_file(tmp_path / "missing-namechange.parquet", "historical ST")
