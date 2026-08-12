from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "factor_drawdown_reentry.py"
SPEC = importlib.util.spec_from_file_location("factor_drawdown_reentry", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_inverse_target_enters_only_after_nonpositive_trend() -> None:
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
    targets, signals = study.core.build_tsmom_targets(
        memberships,
        baseline,
        lookback=1,
        end_date=pd.Timestamp("2026-01-16"),
        risk_on_positive=False,
    )
    assert targets[pd.Timestamp("2026-01-09")] == []
    assert targets[pd.Timestamp("2026-01-16")] == ["A"]
    assert list(signals["risk_on"]) == [False, False, True]


def test_protocol_binds_failed_positive_trend_and_forbids_live() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["genesis"]["development_is_confirmatory"] is False
    assert protocol["overlay"]["risk_on"] == "less than or equal to zero"
    assert protocol["live_trading_authorized"] is False
