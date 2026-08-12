from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "factor_tsmom_overlay.py"
SPEC = importlib.util.spec_from_file_location("factor_tsmom_overlay", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_tsmom_target_uses_only_prior_decision_anchor() -> None:
    memberships = pd.DataFrame(
        {
            "decision_dt": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-16")],
            "symbol": ["A", "A", "A"],
            "factor_rank": [1, 1, 1],
        }
    )
    baseline = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "nav": [1.0, 1.1, 0.9],
        }
    )
    targets, signals = study.build_tsmom_targets(
        memberships, baseline, lookback=1, end_date=pd.Timestamp("2026-01-16")
    )
    assert targets[pd.Timestamp("2026-01-02")] == []
    assert targets[pd.Timestamp("2026-01-09")] == ["A"]
    assert targets[pd.Timestamp("2026-01-16")] == []
    assert list(signals["risk_on"]) == [False, True, False]


def test_gate_rejects_trivial_always_cash_path() -> None:
    protocol = {
        "statistics": {
            "minimum_risk_on_decision_rate": 0.25,
            "maximum_risk_on_decision_rate": 0.85,
            "minimum_relative_drawdown_reduction": 0.15,
            "minimum_sharpe_improvement": 0.15,
            "maximum_annual_return_underperformance": 0.01,
            "one_sided_alpha": 0.05,
        }
    }
    summary = {
        "risk_on_decision_rate": 0.0,
        "candidate": {"annualized_return": 0.01},
        "relative_drawdown_reduction": 1.0,
        "sharpe_improvement": 1.0,
        "annualized_return_difference": 0.0,
        "weekly_active_hac": {"noninferiority_one_sided_p": 0.01},
    }
    gate = study.evaluate_gate(summary, protocol)
    assert not gate["passed"]
    assert not gate["checks"]["risk_on_rate_floor"]


def test_protocol_is_one_unlevered_variant_and_not_live() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["mechanism"]["single_variant_only"] is True
    assert protocol["overlay"]["lookback_decisions"] == 20
    assert protocol["overlay"]["no_leverage"] is True
    assert protocol["live_trading_authorized"] is False
