from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "fsm_factor_exit_overlay.py"
SPEC = importlib.util.spec_from_file_location("fsm_factor_exit_overlay", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_build_target_memberships_retains_rank_75() -> None:
    rows = []
    for date, symbols in (
        ("2026-01-02", [("A", 1), ("B", 2)]),
        ("2026-01-09", [("C", 1), ("A", 2), ("B", 3)]),
    ):
        rows.extend({"decision_dt": pd.Timestamp(date), "symbol": symbol, "factor_rank": rank} for symbol, rank in symbols)
    result = study.build_target_memberships(pd.DataFrame(rows), target=2, retention_rank=2)
    second = result[result["decision_dt"].eq(pd.Timestamp("2026-01-09"))]
    assert set(second["symbol"]) == {"A", "C"}
    assert second.set_index("symbol").at["A", "membership_role"] == "RETAINED"


def test_cost_pair_preserves_frozen_asymmetric_primary_cost() -> None:
    assert study.cost_pair(40) == (0.0015, 0.0025)
    assert study.cost_pair(60) == (0.003, 0.003)


def test_segment_metrics_detects_drawdown() -> None:
    path = pd.DataFrame(
        {
            "dt": pd.date_range("2026-01-01", periods=4, freq="B"),
            "nav": [1.0, 1.1, 0.88, 1.0],
        }
    )
    metrics = study.segment_metrics(path, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-06"))
    assert abs(metrics["maximum_drawdown"] + 0.2) < 1e-12


def test_primary_gate_requires_every_frozen_check() -> None:
    protocol = {
        "statistics": {
            "minimum_development_overlay_exits": 100,
            "minimum_validation_overlay_exits": 100,
            "minimum_relative_drawdown_reduction": 0.1,
            "minimum_sharpe_improvement": 0.1,
            "noninferiority_margin_annual_return": 0.02,
            "one_sided_alpha": 0.05,
        }
    }
    summary = {
        "overlay_state_exit_fills": 100,
        "relative_drawdown_reduction": 0.1,
        "sharpe_improvement": 0.1,
        "annualized_return_difference": -0.02,
        "weekly_active_hac": {"noninferiority_one_sided_p": 0.049},
    }
    assert study.evaluate_primary_gate(summary, protocol, validation=False)["passed"]
    summary["sharpe_improvement"] = 0.099
    assert not study.evaluate_primary_gate(summary, protocol, validation=False)["passed"]


def test_protocol_is_retrospective_and_single_variant() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["live_trading_authorized"] is False
    assert protocol["mechanism"]["single_variant_only"] is True
    assert protocol["execution"]["overlay_signal_states"] == [9, 10]
