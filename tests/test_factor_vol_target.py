from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "factor_vol_target.py"
SPEC = importlib.util.spec_from_file_location("factor_vol_target", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_sleeve_zero_exposure_stays_cash() -> None:
    baseline = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "open_nav": [1.0, 1.0],
            "nav": [1.0, 1.1],
        }
    )
    signals = pd.DataFrame(
        {"fill_dt": [pd.Timestamp("2026-01-05")], "target_exposure": [0.0]}
    )
    result = study.simulate_sleeve(
        baseline, signals, buy_cost=0.0015, sell_cost=0.0025, terminal_date=pd.Timestamp("2026-01-06")
    )
    assert result.iloc[-1]["nav"] == 1.0


def test_sleeve_full_exposure_tracks_intraday_after_cost() -> None:
    baseline = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "open_nav": [1.0, 1.0],
            "nav": [1.0, 1.1],
        }
    )
    signals = pd.DataFrame(
        {"fill_dt": [pd.Timestamp("2026-01-05")], "target_exposure": [1.0]}
    )
    result = study.simulate_sleeve(
        baseline, signals, buy_cost=0.0, sell_cost=0.0, terminal_date=pd.Timestamp("2026-01-06")
    )
    assert abs(result.iloc[-1]["nav"] - 1.1) < 1e-12


def test_protocol_freezes_standard_unlevered_rule() -> None:
    protocol = study.load_and_verify_protocol()
    assert protocol["overlay"]["lookback_market_sessions"] == 20
    assert protocol["overlay"]["target_annualized_volatility"] == 0.15
    assert protocol["overlay"]["no_leverage"] is True
    assert protocol["live_trading_authorized"] is False
