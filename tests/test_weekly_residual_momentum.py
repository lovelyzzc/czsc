from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "weekly_residual_momentum.py"
SPEC = importlib.util.spec_from_file_location("weekly_residual_momentum", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_momentum_ranks_positive_residual_first() -> None:
    surface = pd.DataFrame(
        {
            "decision_dt": pd.Timestamp("2026-01-02"),
            "symbol": ["A", "B", "C", "D"],
            "industry_code": ["I1", "I1", "I2", "I2"],
            "log_free_float_mcap": [10.0, 11.0, 10.0, 11.0],
            "raw_return_5obs": [-0.20, 0.0, -0.01, 0.20],
        }
    )
    ranked = study.rank_momentum_signal(surface)
    assert ranked.iloc[0]["residual_return_5obs"] == ranked["residual_return_5obs"].max()
    assert list(ranked["momentum_rank"]) == [1, 2, 3, 4]


def test_momentum_membership_uses_named_rank_column() -> None:
    ranked = pd.DataFrame(
        {
            "decision_dt": [pd.Timestamp("2026-01-02")] * 3,
            "symbol": ["A", "B", "C"],
            "momentum_rank": [2, 1, 3],
            "residual_return_5obs": [0.2, 0.3, 0.1],
        }
    )
    memberships = study.core.build_memberships(ranked, target=2, retention_rank=2, rank_column="momentum_rank")
    assert set(memberships["symbol"]) == {"A", "B"}


def test_protocol_binds_post_development_genesis() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["classification"].startswith("RETROSPECTIVE_POST_DEVELOPMENT")
    assert protocol["genesis"]["source"].startswith("weekly_residual_reversal")
    assert protocol["live_trading_authorized"] is False
