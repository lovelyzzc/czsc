from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "weekly_residual_reversal.py"
SPEC = importlib.util.spec_from_file_location("weekly_residual_reversal", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_neutralized_reversal_ranks_low_residual_first() -> None:
    surface = pd.DataFrame(
        {
            "decision_dt": pd.Timestamp("2026-01-02"),
            "symbol": ["A", "B", "C", "D"],
            "industry_code": ["I1", "I1", "I2", "I2"],
            "log_free_float_mcap": [10.0, 11.0, 10.0, 11.0],
            "raw_return_5obs": [-0.20, 0.0, -0.01, 0.01],
        }
    )
    ranked = study.rank_reversal_signal(surface)
    assert ranked.iloc[0]["symbol"] == "A"
    assert list(ranked["reversal_rank"]) == [1, 2, 3, 4]


def test_membership_buffer_retains_rank_75() -> None:
    rows = []
    for date, symbols in (
        ("2026-01-02", [("A", 1), ("B", 2)]),
        ("2026-01-09", [("C", 1), ("A", 2), ("B", 3)]),
    ):
        rows.extend(
            {
                "decision_dt": pd.Timestamp(date),
                "symbol": symbol,
                "reversal_rank": rank,
                "residual_return_5obs": float(rank),
            }
            for symbol, rank in symbols
        )
    memberships = study.build_memberships(pd.DataFrame(rows), target=2, retention_rank=2)
    second = memberships[memberships["decision_dt"].eq(pd.Timestamp("2026-01-09"))]
    assert set(second["symbol"]) == {"A", "C"}
    assert second.set_index("symbol").at["A", "membership_role"] == "RETAINED"


def test_gate_fails_when_any_half_reverses() -> None:
    protocol = {
        "statistics": {
            "minimum_active_weekly_effect": 0.0025,
            "one_sided_alpha": 0.05,
            "minimum_observable_target_slot_rate": 0.98,
            "maximum_compounded_drawdown": 0.35,
        }
    }
    summary = {
        "active_mean": 0.003,
        "hac": {"one_sided_positive_p": 0.01},
        "bootstrap": {"q05": 0.001},
        "half_means": [{"active_mean": 0.002}, {"active_mean": -0.0001}],
        "candidate_net_mean": 0.003,
        "observable_target_slot_rate": 1.0,
        "compounded_candidate_max_drawdown": -0.2,
    }
    gate = study.evaluate_gate(summary, protocol)
    assert not gate["passed"]
    assert not gate["checks"]["both_halves_positive"]


def test_cost_pair_keeps_primary_asymmetry() -> None:
    assert study.cost_pair(40) == (0.0015, 0.0025)
    assert np.allclose(study.cost_pair(60), (0.003, 0.003))


def test_protocol_is_single_variant_and_not_live() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["mechanism"]["single_variant_only"] is True
    assert protocol["live_trading_authorized"] is False
