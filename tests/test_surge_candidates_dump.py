"""主升浪候选 raw 保留与 FULL 成交语义测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_candidates_dump as dump  # noqa: E402


def _simulation_inputs(*, close_at_signal: float = 95.0, sell_regime: bool = True) -> tuple[list, dict, dict]:
    dates = pd.bdate_range("2026-01-01", periods=5)
    states = [SimpleNamespace(idx=i, next_open=100.0 if i == 0 else np.nan, sl_ref=np.nan) for i in range(5)]
    indicators = {
        "n": 5,
        "open": np.array([99.0, 100.0, 101.0, 97.0, 98.0]),
        "close": np.array([99.0, 110.0, close_at_signal, 98.0, 99.0]),
        "low": np.array([98.0, 99.0, 94.0, 96.0, 97.0]),
        "dates": dates,
    }
    regimes = {2: next(iter(dump.SELL_SET))} if sell_regime else {}
    return states, regimes, indicators


def test_state_exit_separates_signal_and_next_open_fill() -> None:
    states, regimes, indicators = _simulation_inputs()

    result = dump._simulate_full(0, states, regimes, indicators)

    assert result is not None
    assert result.reason == "state"
    assert result.exit_signal_idx == 2
    assert result.exit_fill_idx == 3
    assert result.exit_signal_dt == pd.Timestamp("2026-01-05")
    assert result.exit_fill_dt == pd.Timestamp("2026-01-06")
    assert result.exit_price == 97.0
    fields = dump._outcome_fields(result)
    assert fields["exit_dt"] == fields["exit_fill_dt"]
    assert fields["hold_signal_days"] == 1
    assert fields["hold_days"] == 2


def test_close_triggered_trail_uses_next_open() -> None:
    states, regimes, indicators = _simulation_inputs(close_at_signal=80.0, sell_regime=False)

    result = dump._simulate_full(0, states, regimes, indicators)

    assert result is not None
    assert result.reason == "trail18"
    assert result.exit_signal_idx == 2
    assert result.exit_fill_idx == 3
    assert result.exit_price == 97.0


def test_incomplete_future_is_preserved_as_raw_without_outcome() -> None:
    states, regimes, indicators = _simulation_inputs()
    indicators["n"] = 3
    for name in ("open", "close", "low", "dates"):
        indicators[name] = indicators[name][:3]

    result = dump._simulate_full(0, states, regimes, indicators)
    fields = dump._outcome_fields(result)

    assert result is None
    assert fields["full_outcome_complete"] is False
    assert fields["entry_dt"] is None
    assert fields["exit_signal_dt"] is None
    assert fields["exit_fill_dt"] is None

    frame = pd.DataFrame([fields, {**fields, "full_outcome_complete": True}])
    assert len(dump.completed_outcomes(frame)) == 1


def test_incomplete_max_hold_does_not_create_terminal_close_outcome() -> None:
    states, regimes, indicators = _simulation_inputs(close_at_signal=109.0, sell_regime=False)

    result = dump._simulate_full(0, states, regimes, indicators)

    assert result is None


def test_process_persists_raw_rows_when_full_outcome_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 30,
            "dt": dates,
            "open": np.arange(10.0, 40.0),
            "close": np.arange(10.0, 40.0),
            "low": np.arange(9.0, 39.0),
            "amount": [200_000.0] * 30,
        }
    )
    states = [
        SimpleNamespace(
            idx=i,
            regime=int(dump.Regime.UpwardDeparture),
            feats={"vol_ratio": 0.8, "ma_spread_pct": 3.0, "ret20": 8.0},
            dt=dates[i],
            close=float(10 + i),
            next_open=float(11 + i) if i < 29 else np.nan,
            sl_ref=np.nan,
            zd=9.0,
        )
        for i in range(30)
    ]
    indicators = {
        "n": 30,
        "open": frame["open"].to_numpy(),
        "close": frame["close"].to_numpy(),
        "low": frame["low"].to_numpy(),
        "dates": dates,
    }
    monkeypatch.setattr(dump.tr, "load_stock", lambda _path: frame)
    monkeypatch.setattr(dump.tr, "iter_states", lambda _frame, with_features: states)
    monkeypatch.setattr(dump.tr, "compute_indicators", lambda _frame: indicators)
    monkeypatch.setattr(dump.tr, "limit_pct_for", lambda _symbol: 9.8)
    monkeypatch.setattr(dump.tr, "surge_score", lambda _features: 1.0)
    monkeypatch.setattr(dump, "DELAYS", [0])
    monkeypatch.setattr(dump, "_is_candidate", lambda *_args: True)
    monkeypatch.setattr(dump, "_simulate_full", lambda *_args: None)

    result = dump._process("ignored.parquet")

    assert result["status"] == "processed"
    assert len(result["rows"]) == 58  # 29 decision rows × confirm/anticipate
    assert not any(row["full_outcome_complete"] for row in result["rows"])
    assert any(row["dec_dt"] == dates[-1] and row["entry_dt"] is None for row in result["rows"])
