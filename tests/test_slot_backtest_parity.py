"""Parity test: Rust simulate_surge_backtest vs Python inline implementation.

Ensures the Rust implementation produces identical results to the old Python
`simulate_slots` + `_daily_returns` logic in strict mode (GateLevel::Full).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from czsc._native.research import simulate_surge_backtest

BUY_COST = 0.0015
SELL_COST = 0.0025
N_SLOTS = 10


def _build_synthetic_data():
    """Build synthetic candidates + close panel matching both Python and Rust interfaces."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", "2023-03-31")
    symbols = [f"S{i:04d}" for i in range(30)]

    close_data = {}
    for sym in symbols:
        price = 10.0 + np.random.randn() * 2
        prices = [price]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
        close_data[sym] = prices
    close_df = pd.DataFrame(close_data, index=dates)

    candidates = []
    for i in range(50):
        sym = symbols[i % len(symbols)]
        entry_idx = np.random.randint(0, len(dates) - 10)
        hold = np.random.randint(3, 8)
        exit_idx = min(entry_idx + hold, len(dates) - 1)
        entry_dt = dates[entry_idx]
        exit_dt = dates[exit_idx]
        entry_price = close_data[sym][entry_idx]
        exit_price = close_data[sym][exit_idx]
        ret_gross = (exit_price / entry_price - 1) * 100
        candidates.append(
            {
                "symbol": sym,
                "entry_dt": entry_dt,
                "exit_dt": exit_dt,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret_gross_pct": ret_gross,
                "priority": 100.0 - i,
                "hold_days": (exit_dt - entry_dt).days,
                "gate_level": 3,
                "gate_confidence": 1.0,
                "seg": "train",
                "year": 2023,
            }
        )

    cand_df = pd.DataFrame(candidates).sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(
        drop=True
    )
    return cand_df, close_df


def _python_simulate_slots(df: pd.DataFrame, close_df: pd.DataFrame, n_slots: int):
    """Port of the old Python simulate_slots + _daily_returns."""
    taken_rows = []
    open_until: dict[str, pd.Timestamp] = {}
    for entry_dt, day_grp in df.groupby("entry_dt", sort=True):
        open_until = {s: x for s, x in open_until.items() if x >= entry_dt}
        free = n_slots - len(open_until)
        if free <= 0:
            continue
        for _, r in day_grp.iterrows():
            if free <= 0:
                break
            if r["symbol"] in open_until:
                continue
            open_until[r["symbol"]] = r["exit_dt"]
            taken_rows.append(r)
            free -= 1
    trades = pd.DataFrame(taken_rows).reset_index(drop=True)

    acc: dict[pd.Timestamp, float] = {}
    for r in trades.itertuples():
        sym_closes = close_df[r.symbol]
        c = sym_closes.loc[r.entry_dt : r.exit_dt]
        dts, px = c.index, c.to_numpy(dtype=float)
        entry_eff = r.entry_price * (1 + BUY_COST)
        exit_eff = r.exit_price * (1 - SELL_COST)
        for k, dt in enumerate(dts):
            if len(dts) == 1:
                ret = exit_eff / entry_eff - 1
            elif k == 0:
                ret = px[0] / entry_eff - 1
            elif k == len(dts) - 1:
                ret = exit_eff / px[k - 1] - 1
            else:
                ret = px[k] / px[k - 1] - 1
            acc[dt] = acc.get(dt, 0.0) + ret
    daily = pd.Series(acc).sort_index() / n_slots
    return trades, daily


def test_simulate_slots_parity():
    """Rust simulate_surge_backtest should match Python simulate_slots in strict mode."""
    cand_df, close_df = _build_synthetic_data()

    # --- Python ---
    py_trades, py_daily = _python_simulate_slots(cand_df, close_df, N_SLOTS)

    # --- Rust: prepare inputs ---
    rust_candidates = []
    for _, r in cand_df.iterrows():
        rust_candidates.append(
            {
                "symbol": r["symbol"],
                "entry_dt": r["entry_dt"].strftime("%Y-%m-%d"),
                "exit_dt": r["exit_dt"].strftime("%Y-%m-%d"),
                "entry_price": float(r["entry_price"]),
                "exit_price": float(r["exit_price"]),
                "gate_level": 3,
                "gate_confidence": 1.0,
                "priority": float(r["priority"]),
                "ret_gross_pct": float(r["ret_gross_pct"]),
                "hold_days": int(r["hold_days"]),
                "seg": str(r["seg"]),
                "year": int(r["year"]),
            }
        )

    close_triples = []
    for dt in close_df.index:
        for sym in close_df.columns:
            close_triples.append((sym, dt.strftime("%Y-%m-%d"), float(close_df.loc[dt, sym])))

    rust_result = simulate_surge_backtest(
        rust_candidates,
        close_triples,
        n_slots=N_SLOTS,
        fill_mode="strict",
        buy_cost=BUY_COST,
        sell_cost=SELL_COST,
        train_end_year=2023,
    )

    rust_trades = rust_result["trades"]
    rust_daily = rust_result["daily_returns"]

    # --- Compare trades ---
    assert len(rust_trades) == len(py_trades), (
        f"trade count mismatch: Rust={len(rust_trades)}, Python={len(py_trades)}"
    )
    for i, (rt, (_, pt)) in enumerate(zip(rust_trades, py_trades.iterrows())):
        assert rt["symbol"] == pt["symbol"], f"trade {i}: symbol mismatch"
        assert rt["entry_dt"] == pt["entry_dt"].strftime("%Y-%m-%d"), f"trade {i}: entry_dt mismatch"
        assert rt["exit_dt"] == pt["exit_dt"].strftime("%Y-%m-%d"), f"trade {i}: exit_dt mismatch"
        np.testing.assert_allclose(rt["entry_price"], pt["entry_price"], rtol=1e-10, err_msg=f"trade {i}: entry_price")
        np.testing.assert_allclose(rt["exit_price"], pt["exit_price"], rtol=1e-10, err_msg=f"trade {i}: exit_price")

    # --- Compare daily returns ---
    py_daily_dict = {dt.strftime("%Y-%m-%d"): val for dt, val in py_daily.items()}
    rust_daily_dict = {dr[0]: dr[1] for dr in rust_daily}

    py_dates = sorted(py_daily_dict.keys())
    rust_dates = sorted(rust_daily_dict.keys())
    assert py_dates == rust_dates, f"daily return dates mismatch: Python has {len(py_dates)}, Rust has {len(rust_dates)}"

    for dt in py_dates:
        np.testing.assert_allclose(
            rust_daily_dict[dt],
            py_daily_dict[dt],
            atol=1e-12,
            err_msg=f"daily return mismatch on {dt}",
        )


def test_gate_level_functions():
    """Test the tiered gating functions exposed via PyO3.

    FeatureSnapshot cannot be constructed directly in Python (PyO3 limitation);
    verify the functions are callable and return the correct types.
    """
    from czsc._native.trend_regime import py_classify_gate_level, py_gate_confidence, py_surge_onset_with_level

    assert callable(py_classify_gate_level)
    assert callable(py_gate_confidence)
    assert callable(py_surge_onset_with_level)

    result = py_surge_onset_with_level(0, 7, None, [], "confirm", "full")
    assert result is False


if __name__ == "__main__":
    test_simulate_slots_parity()
    test_gate_level_functions()
    print("All parity tests passed!")
