from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "factor_stable_entry.py"
SPEC = importlib.util.spec_from_file_location("factor_stable_entry", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_entry_filter_does_not_force_exit_of_retained_name() -> None:
    blind = pd.DataFrame(
        [
            {"decision_dt": pd.Timestamp("2026-01-02"), "symbol": "A", "factor_rank": 1, "absolute_shock": 0.1, "stable_new_entry": True},
            {"decision_dt": pd.Timestamp("2026-01-02"), "symbol": "B", "factor_rank": 2, "absolute_shock": 0.2, "stable_new_entry": True},
            {"decision_dt": pd.Timestamp("2026-01-09"), "symbol": "C", "factor_rank": 1, "absolute_shock": 0.1, "stable_new_entry": True},
            {"decision_dt": pd.Timestamp("2026-01-09"), "symbol": "A", "factor_rank": 2, "absolute_shock": 9.0, "stable_new_entry": False},
            {"decision_dt": pd.Timestamp("2026-01-09"), "symbol": "B", "factor_rank": 3, "absolute_shock": 0.2, "stable_new_entry": True},
        ]
    )
    memberships = study.build_arm_memberships(blind, target=2, retention_rank=2)
    stable = memberships[
        memberships["arm"].eq("F_STABLE_ENTRY")
        & memberships["decision_dt"].eq(pd.Timestamp("2026-01-09"))
    ]
    assert set(stable["symbol"]) == {"A", "C"}
    assert stable.set_index("symbol").at["A", "membership_role"] == "RETAINED"


def test_candidate_gate_requires_each_subperiod() -> None:
    protocol = {
        "statistics": {
            "minimum_weekly_improvement": 0.001,
            "one_sided_alpha": 0.05,
            "minimum_observable_target_slot_rate": 0.98,
            "maximum_compounded_drawdown": 0.35,
        }
    }
    summary = {
        "paired_mean": 0.002,
        "hac": {"one_sided_positive_p": 0.01},
        "bootstrap": {"q05": 0.001},
        "subperiods": [
            {"paired_mean": 0.001, "stable_net_mean": 0.002},
            {"paired_mean": -0.001, "stable_net_mean": 0.002},
        ],
        "stable_observable_target_slot_rate": 1.0,
        "stable_compounded_max_drawdown": -0.2,
    }
    gate = study.evaluate_candidate(summary, protocol)
    assert not gate["passed"]
    assert not gate["checks"]["all_subperiod_paired_positive"]


def test_protocol_binds_both_tail_failures_and_forbids_live() -> None:
    protocol = study.load_and_verify_protocol()
    assert set(protocol["genesis"]) >= {"reversal_audit", "momentum_audit"}
    assert protocol["arms"]["entry_filter_only"] is True
    assert protocol["live_trading_authorized"] is False
