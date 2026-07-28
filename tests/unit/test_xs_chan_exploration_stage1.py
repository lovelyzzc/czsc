from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage1 as exploration  # noqa: E402


def _buffer_row(
    decision_dt: str,
    symbol: str,
    rank: int,
    *,
    regime: int,
    ma_allowed: bool,
) -> dict[str, object]:
    return {
        "decision_dt": pd.Timestamp(decision_dt),
        "symbol": symbol,
        "factor_rank": rank,
        "factor_score": 1.0 - rank / 10,
        "industry_code": "I1",
        "industry_name": "Industry 1",
        "board": "MAIN",
        "free_float_mcap": 1_000_000_000.0,
        "log_free_float_mcap": np.log(1_000_000_000.0),
        "mom_120_20": 0.10,
        "lowvol_60": -0.02,
        "adv20": 100_000.0,
        "turnover_rate": 1.0,
        "volume_ratio": 1.0,
        "pe_ttm": 20.0,
        "pb": 2.0,
        "distance_to_sma20": 0.01,
        "ma_allowed": ma_allowed,
        "regime": regime,
        "mcap_bucket": 2,
        "entry_tradable": True,
        "entry_gap": 0.0,
        "fwd_5d_open_return": 0.01,
        "fwd_20d_close_return": 0.02,
        "entry_open": 10.0,
        "entry_high": 10.1,
        "entry_low": 9.9,
        "entry_amount": 100_000.0,
        "exit_5d_open": 10.1,
        "adjusted_close": 10.0,
        "delay_0_tradable": True,
        "delay_0_fwd_5d_return": 0.01,
        "delay_0_entry_amount": 100_000.0,
    }


def test_spec_keeps_v21_frozen_and_prohibits_tuning():
    spec = exploration.load_and_validate_spec()

    assert spec["mode"] == "EXPLORATORY_ONLY"
    assert spec["confirmation_chain"] == "NOT_STARTED"
    assert len(spec["planned_experiments"]) == 8
    assert set(spec["no_tuning"].values()) == {False, True}
    assert all(
        spec["no_tuning"][key] is False
        for key in (
            "parameter_search",
            "threshold_optimization",
            "best_variant_selection",
            "failed_experiment_suppression",
        )
    )


def test_decision_schedule_uses_global_sessions_and_fixed_delays():
    calendar = pd.bdate_range("2024-01-01", periods=50)
    spec = {
        "sample": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "primary_holding_sessions": 5,
            "diagnostic_holding_sessions": 20,
            "signal_delays_sessions": [0, 1, 3, 5],
            "minimum_complete_decision_weeks": 2,
        }
    }

    schedule = exploration.build_decision_schedule(spec, calendar)

    assert len(schedule) >= 2
    first = schedule.iloc[0]
    decision_no = calendar.get_loc(first["decision_dt"])
    assert first["decision_dt"] == pd.Timestamp("2024-01-05")
    assert first["entry_dt"] == calendar[decision_no + 1]
    assert first["delay_5_entry_dt"] == calendar[decision_no + 6]
    assert first["delay_5_exit_dt"] == calendar[decision_no + 11]


def test_buffer_applies_gates_to_new_entries_only_and_does_not_refill():
    spec = {
        "selection": {
            "target_size": 2,
            "retention_max_rank": 3,
            "allowed_chan_regimes": [5, 6, 7, 8],
        },
        "sample": {"signal_delays_sessions": [0]},
    }
    ranked = pd.DataFrame(
        [
            _buffer_row("2024-01-05", "A", 1, regime=5, ma_allowed=True),
            _buffer_row("2024-01-05", "B", 2, regime=1, ma_allowed=False),
            _buffer_row("2024-01-05", "C", 3, regime=7, ma_allowed=True),
            _buffer_row("2024-01-05", "D", 4, regime=7, ma_allowed=True),
            _buffer_row("2024-01-12", "B", 1, regime=1, ma_allowed=False),
            _buffer_row("2024-01-12", "D", 2, regime=7, ma_allowed=True),
            _buffer_row("2024-01-12", "A", 3, regime=1, ma_allowed=False),
            _buffer_row("2024-01-12", "C", 4, regime=7, ma_allowed=True),
        ]
    )

    memberships, proposals = exploration.build_buffered_memberships(spec, ranked)
    chan = memberships[memberships["arm"].eq("FC")]

    first_week = chan[chan["decision_dt"].eq(pd.Timestamp("2024-01-05"))]
    assert first_week["symbol"].tolist() == ["A"]
    assert "C" not in set(first_week["symbol"])
    second_week = chan[chan["decision_dt"].eq(pd.Timestamp("2024-01-12"))]
    assert second_week[["symbol", "membership_role"]].to_dict("records") == [
        {"symbol": "A", "membership_role": "retained"}
    ]
    second_proposals = proposals[proposals["arm"].eq("FC") & proposals["decision_dt"].eq(pd.Timestamp("2024-01-12"))]
    assert second_proposals["symbol"].tolist() == ["B"]


def test_risk_attribution_reconstructs_selected_return():
    dates = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    rows: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    for week_no, decision_dt in enumerate(dates):
        for index in range(120):
            size = index / 119
            lowvol = np.sin(index / 9)
            momentum = np.cos(index / 11)
            industry = "I1" if index < 60 else "I2"
            ret = 0.01 + 0.004 * size - 0.003 * lowvol + 0.002 * momentum + 0.001 * (industry == "I2")
            symbol = f"S{index:03d}"
            rows.append(
                {
                    "decision_dt": decision_dt,
                    "symbol": symbol,
                    "entry_tradable": True,
                    "fwd_5d_open_return": ret + week_no * 0.001,
                    "log_free_float_mcap": size,
                    "lowvol_60": lowvol,
                    "mom_120_20": momentum,
                    "industry_code": industry,
                }
            )
            if index < 30:
                members.append({"decision_dt": decision_dt, "symbol": symbol, "arm": "F"})
    spec = {"attribution": {"summary_hac_lag": 1}}

    weekly, summary = exploration.run_risk_attribution(spec, pd.DataFrame(rows), pd.DataFrame(members))

    assert len(weekly) == 2
    assert weekly["reconstruction_error"].abs().max() < 1e-12
    assert set(summary["component"]) == {
        "selected_return",
        "market",
        "size",
        "industry",
        "low_volatility",
        "momentum",
        "unexplained_residual",
    }


def test_state_returns_are_week_weighted_and_turnover_is_present():
    base = {
        "arm": "F",
        "regime": 5,
        "entry_tradable": True,
        "adv20": 100_000.0,
        "entry_amount": 100_000.0,
        "board": "MAIN",
        "entry_open": 10.0,
        "entry_high": 10.1,
        "entry_low": 9.9,
        "entry_gap": 0.0,
        "fwd_20d_max_drawdown": -0.1,
    }
    memberships = pd.DataFrame(
        [
            {**base, "decision_dt": pd.Timestamp("2024-01-05"), "symbol": "A", "fwd_5d_open_return": 1.0},
            {**base, "decision_dt": pd.Timestamp("2024-01-05"), "symbol": "B", "fwd_5d_open_return": 1.0},
            {**base, "decision_dt": pd.Timestamp("2024-01-12"), "symbol": "B", "fwd_5d_open_return": -1.0},
        ]
    )
    spec = {
        "selection": {
            "allowed_chan_regimes": [5, 6, 7, 8],
            "slot_capital_cny": 199_000.0,
            "adv_participation_limit": 0.05,
        }
    }

    diagnostics = exploration.run_state_diagnostics(spec, memberships)
    state5 = diagnostics[diagnostics["state"].eq("5")].iloc[0]

    assert state5["fwd_5d_n_events"] == 3
    assert state5["fwd_5d_n_weeks"] == 2
    assert state5["fwd_5d_mean"] == pytest.approx(0.0)
    assert state5["identity_turnover_jaccard"] == pytest.approx(0.5)


def test_failure_feature_timing_blocks_future_diagnostics_from_hypotheses():
    memberships = pd.DataFrame(
        [
            {
                "arm": "F",
                "decision_dt": pd.Timestamp("2023-01-06") + pd.Timedelta(weeks=index),
                "symbol": f"S{index}",
                "entry_tradable": True,
                "fwd_5d_open_return": -0.01 if index % 2 == 0 else 0.01,
                "signal_a": float(index),
                "signal_b": float(index % 3),
                "entry_gap": float(index) / 100,
                "fwd_20d_max_drawdown": -0.5 if index % 2 == 0 else -0.01,
                "regime": 7,
                "industry_code": "I1",
                "board": "MAIN",
            }
            for index in range(12)
        ]
    )
    spec = {
        "failure_profile": {
            "split_date": "2024-01-01",
            "decision_time_numeric_features": ["signal_a", "signal_b"],
            "execution_time_numeric_features": ["entry_gap"],
            "outcome_diagnostics": ["fwd_20d_max_drawdown"],
            "categorical_features": ["regime", "industry_code", "board"],
        }
    }
    _pool, numeric, _categorical = exploration.run_failure_profile(spec, memberships)
    attribution = pd.DataFrame(
        [
            {
                "component": "unexplained_residual",
                "mean": -0.001,
                "annualized_arithmetic": -0.052,
                "hac_t": -1.0,
            }
        ]
    )
    states = pd.DataFrame([{"state": "7", "fwd_5d_mean": 0.0}])
    controls = pd.DataFrame(
        [
            {
                "experiment": "simple_trend_gate",
                "variant": "chan_states_5_8",
                "mean_weekly_return": 0.001,
            },
            {
                "experiment": "simple_trend_gate",
                "variant": "sma20",
                "mean_weekly_return": 0.002,
            },
        ]
    )

    hypotheses = exploration.derive_next_hypotheses(attribution, states, numeric, controls)

    timing = numeric.set_index("feature")["feature_timing"].to_dict()
    assert timing["signal_a"] == "decision_time"
    assert timing["entry_gap"] == "execution_time"
    assert timing["fwd_20d_max_drawdown"] == "outcome"
    assert "fwd_20d_max_drawdown" not in hypotheses[2]["statement"]
    assert {row["feature"] for row in hypotheses[2]["stage1_evidence"]} == {"signal_a", "signal_b"}


def test_state_randomization_is_deterministic_and_preserves_weekly_quota():
    proposals = pd.DataFrame(
        [
            {
                "decision_dt": pd.Timestamp("2024-01-05") + pd.Timedelta(weeks=week),
                "symbol": f"S{index}",
                "arm": "FC",
                "entry_tradable": True,
                "fwd_5d_open_return": (index - 2) / 100,
                "gate_passed": index < 2,
            }
            for week in range(3)
            for index in range(5)
        ]
    )
    spec = {
        "negative_controls": {
            "random_repetitions": 5,
            "random_seed_root": 42,
        }
    }

    first_summary, first_distribution = exploration.run_state_randomization(spec, proposals)
    second_summary, second_distribution = exploration.run_state_randomization(spec, proposals)

    pd.testing.assert_frame_equal(first_summary, second_summary)
    pd.testing.assert_frame_equal(first_distribution, second_distribution)
    assert set(first_distribution["selected_rows"]) == {6}
