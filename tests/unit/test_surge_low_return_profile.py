import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_low_return_profile as profile  # noqa: E402


def test_rule_mask_uses_only_explicit_decision_time_conditions():
    df = pd.DataFrame(
        {
            "amt_ratio20": [0.9, 1.0],
            "ret5": [-0.1, 0.0],
            "dec_regime": [5, 7],
            "high20_ratio": [0.15, 0.151],
        }
    )

    assert profile.rule_mask(df, "decision_volume_confirmed").tolist() == [False, True]
    assert profile.rule_mask(df, "short_term_positive").tolist() == [False, True]
    assert profile.rule_mask(df, "main_uptrend_only").tolist() == [False, True]
    assert profile.rule_mask(df, "stronger_market_breadth").tolist() == [False, True]


def test_is_not_worse_treats_zero_as_a_real_statistic():
    assert profile.is_not_worse(0.0, 0.0)
    assert not profile.is_not_worse(None, 0.0)
    assert not profile.is_not_worse(-0.1, 0.0)


def test_simulate_slots_keeps_highest_priority_and_one_position_per_symbol():
    entry = pd.Timestamp("2024-01-02")
    rows = [
        {"symbol": f"S{i:02d}", "entry_dt": entry, "exit_dt": pd.Timestamp("2024-01-10"), "priority": 100 - i}
        for i in range(11)
    ]
    rows.append({"symbol": "S00", "entry_dt": entry, "exit_dt": pd.Timestamp("2024-01-10"), "priority": 200})

    selected = profile.simulate_slots(pd.DataFrame(rows))

    assert len(selected) == 10
    assert selected["symbol"].is_unique
    assert "S10" not in selected["symbol"].tolist()
