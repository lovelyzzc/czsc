from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_research as xs  # noqa: E402


def _pilot_config() -> xs.PilotConfig:
    return xs.PilotConfig(
        protocol_id="unit-test",
        start_date="2024-01-01",
        common_cutoff="2025-12-31",
        frequency="weekly",
        holdings=2,
        initial_capital_cny=20_000.0,
        min_history_bars=1,
        min_valid_bars_60=1,
        min_adv20_thousand_cny=0.0,
        liquidity_buckets=2,
        winsor_low=0.01,
        winsor_high=0.99,
        random_seed=7,
        allowed_chan_regimes=(5, 6, 7, 8),
        cash_buffer=0.0,
    )


def _zero_cost() -> xs.CostModel:
    return xs.CostModel(
        commission_bps=0.0,
        transfer_bps=0.0,
        min_commission_cny=0.0,
        stamp_before_bps=0.0,
        stamp_after_bps=0.0,
        base_slippage_bps=0.0,
        impact_bps_at_1pct=0.0,
        max_adv_participation=1.0,
    )


def _market(
    dates: list[str],
    data: dict[str, dict[str, list[float] | list[bool]]],
) -> xs.MarketData:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    rows: list[dict] = []
    for symbol, columns in data.items():
        opens = columns["open"]
        n = len(index)
        assert len(opens) == n
        closes = columns.get("close", opens)
        pre_close = columns.get("pre_close", opens)
        amount = columns.get("amount_cny", [1_000_000_000.0] * n)
        is_st = columns.get("is_st", [False] * n)
        for i, dt in enumerate(index):
            rows.append(
                {
                    "symbol": symbol,
                    "dt": dt,
                    "open": opens[i],
                    "close": closes[i],
                    "pre_close": pre_close[i],
                    "amount_cny": amount[i],
                    "is_st": is_st[i],
                }
            )
    return xs.MarketData.from_long_frame(pd.DataFrame(rows), index, list(data))


def _plan(
    decision_dt: str,
    exec_dt: str,
    symbols: tuple[str, ...],
    *,
    gate_allowed: dict[str, bool] | None = None,
    extra_adv: dict[str, float] | None = None,
) -> xs.TargetPlan:
    adv = dict.fromkeys(symbols, 1000000000.0)
    adv.update(extra_adv or {})
    return xs.TargetPlan(
        decision_dt=pd.Timestamp(decision_dt),
        exec_dt=pd.Timestamp(exec_dt),
        symbols=symbols,
        adv_cny=adv,
        gate_allowed=gate_allowed or dict.fromkeys(symbols, True),
    )


def test_weekly_schedule_uses_last_session_of_week_and_next_session_open():
    # 2024-01-05 is deliberately absent: Thursday must become that week's
    # decision session, with execution on the following Monday.
    calendar = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-15",
        ]
    )

    schedule = xs.make_rebalance_schedule(calendar, "2024-01-01", "2024-01-15", "weekly")

    assert schedule.to_dict("records") == [
        {"decision_dt": pd.Timestamp("2024-01-04"), "exec_dt": pd.Timestamp("2024-01-08")},
        {"decision_dt": pd.Timestamp("2024-01-12"), "exec_dt": pd.Timestamp("2024-01-15")},
    ]


def test_symbol_factors_are_prefix_invariant_when_future_bars_are_appended():
    dates = pd.bdate_range("2024-01-02", periods=170)
    base_close = 10.0 * np.exp(np.linspace(0.0, 0.3, len(dates)))
    raw = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates,
            "open": base_close * 0.999,
            "close": base_close,
            "pre_close": np.r_[base_close[0], base_close[:-1]],
            "vol": 1_000_000.0,
            "amount": np.linspace(20_000.0, 25_000.0, len(dates)),
        }
    )
    prefix = raw.iloc[:150].copy()
    future = raw.iloc[150:].copy()
    future["close"] = np.linspace(1.0, 1_000.0, len(future))
    future["open"] = future["close"] * 1.2
    future["amount"] = 1_000_000_000.0

    prefix_features = xs.compute_symbol_features(prefix, "000001.SZ", {}, _pilot_config())
    full_features = xs.compute_symbol_features(pd.concat([prefix, future]), "000001.SZ", {}, _pilot_config())
    columns = ["dt", "mom_120_20", "lowvol_60", "adv20", "valid60", "universe_valid"]

    pd.testing.assert_frame_equal(
        prefix_features[columns].reset_index(drop=True),
        full_features.loc[full_features["dt"].isin(prefix_features["dt"]), columns].reset_index(drop=True),
        check_exact=True,
    )


def test_missing_stock_rows_count_as_inactive_market_sessions():
    calendar = pd.bdate_range("2024-01-02", periods=140)
    missing = {calendar[-10], calendar[-20], calendar[-30], calendar[-40], calendar[-50]}
    observed = calendar[~calendar.isin(missing)]
    close = 10.0 * np.exp(np.linspace(0.0, 0.2, len(observed)))
    raw = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": observed,
            "open": close,
            "close": close,
            "pre_close": np.r_[close[0], close[:-1]],
            "vol": 1_000_000.0,
            "amount": 100_000.0,
        }
    )

    features = xs.compute_symbol_features(raw, "000001.SZ", {}, _pilot_config(), calendar=calendar)
    last = features.iloc[-1]

    assert len(features) == len(calendar)
    assert last["valid60"] == 55
    assert features.loc[features["dt"].isin(missing), "observed_bar"].eq(False).all()
    assert features.loc[features["dt"].isin(missing), "amount"].eq(0).all()


def test_entry_gate_is_applied_after_topn_and_does_not_backfill():
    decision = pd.Timestamp("2024-01-04")
    execution = pd.Timestamp("2024-01-08")
    ranked = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "dt": decision,
            "factor_rank": [1, 2, 3, 4],
            "adv20": [1_000_000.0] * 4,
            "gate_allowed": [True, False, True, True],
        }
    )
    schedule = pd.DataFrame({"decision_dt": [decision], "exec_dt": [execution]})

    plan = xs.build_target_plans(ranked, schedule, holdings=3)[execution]
    accepted = xs.apply_entry_gate(plan.symbols, held_symbols=(), gate_allowed=plan.gate_allowed)

    assert plan.symbols == ("A", "B", "C")
    assert accepted == ("A", "C")
    assert "D" not in accepted
    assert xs.apply_entry_gate(plan.symbols, held_symbols=("B",), gate_allowed=plan.gate_allowed) == ("A", "B", "C")


def test_random_gate_matches_chan_acceptance_count_and_is_deterministic():
    symbols = ("A", "B", "C", "D")
    first = xs.apply_random_target_count(symbols, held_symbols=("B",), target_count=2, seed=17)
    second = xs.apply_random_target_count(symbols, held_symbols=("B",), target_count=2, seed=17)

    assert len(first) == 2
    assert "B" in first
    assert first == second


def test_decision_at_close_executes_at_next_session_open():
    market = _market(
        ["2024-01-02", "2024-01-03"],
        {
            "A": {
                "open": [9.0, 10.0],
                "close": [9.0, 10.0],
                "pre_close": [9.0, 10.0],
            }
        },
    )
    plan = _plan("2024-01-02", "2024-01-03", ("A",))

    result = xs.simulate_portfolio(
        market,
        {plan.exec_dt: plan},
        "F",
        slots=1,
        initial_capital=10_000.0,
        cost=_zero_cost(),
        cash_buffer=0.0,
    )

    buy = result.trades.query("side == 'buy'").iloc[0]
    assert buy["decision_dt"] == pd.Timestamp("2024-01-02")
    assert buy["exec_dt"] == pd.Timestamp("2024-01-03")
    assert buy["open_price"] == pytest.approx(10.0)
    assert not (result.trades["exec_dt"] == pd.Timestamp("2024-01-02")).any()


def test_non_rebalance_price_drift_is_not_reset_to_free_equal_weights():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    market = _market(
        dates,
        {
            "A": {
                "open": [10.0, 10.0, 10.0, 20.0, 20.0],
                "close": [10.0, 10.0, 20.0, 20.0, 20.0],
                "pre_close": [10.0, 10.0, 10.0, 20.0, 20.0],
            },
            "B": {
                "open": [10.0] * 5,
                "close": [10.0] * 5,
                "pre_close": [10.0] * 5,
            },
        },
    )
    first = _plan("2024-01-02", "2024-01-03", ("A", "B"))
    next_rebalance = _plan("2024-01-05", "2024-01-08", ("A", "B"))

    result = xs.simulate_portfolio(
        market,
        {first.exec_dt: first, next_rebalance.exec_dt: next_rebalance},
        "F",
        slots=2,
        initial_capital=20_000.0,
        cost=_zero_cost(),
        cash_buffer=0.0,
    )

    drift_day = result.holdings[result.holdings["dt"] == pd.Timestamp("2024-01-04")].set_index("symbol")
    assert drift_day.loc["A", "shares"] == drift_day.loc["B", "shares"]
    assert drift_day.loc["A", "weight"] == pytest.approx(2 / 3)
    assert drift_day.loc["B", "weight"] == pytest.approx(1 / 3)
    assert result.trades[result.trades["exec_dt"] == pd.Timestamp("2024-01-04")].empty


@pytest.mark.parametrize(
    ("blocked_open", "blocked_amount", "blocked_status"),
    [
        (9.0, 1_000_000_000.0, "limit_down"),
        (np.nan, 0.0, "suspended"),
    ],
)
def test_blocked_sell_is_retried_each_session_until_filled(blocked_open, blocked_amount, blocked_status):
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ]
    market = _market(
        dates,
        {
            "A": {
                "open": [10.0, 10.0, 10.0, blocked_open, 9.2, 9.3, 9.4],
                "close": [10.0, 10.0, 10.0, np.nan if np.isnan(blocked_open) else blocked_open, 9.2, 9.3, 9.4],
                "pre_close": [10.0, 10.0, 10.0, 10.0, 9.0, 9.2, 9.3],
                "amount_cny": [1_000_000_000.0] * 3 + [blocked_amount] + [1_000_000_000.0] * 3,
            }
        },
    )
    entry = _plan("2024-01-02", "2024-01-03", ("A",))
    exit_plan = _plan("2024-01-04", "2024-01-05", (), extra_adv={"A": 1_000_000_000.0})
    terminal = _plan("2024-01-09", "2024-01-10", ())

    result = xs.simulate_portfolio(
        market,
        {entry.exec_dt: entry, exit_plan.exec_dt: exit_plan, terminal.exec_dt: terminal},
        "F",
        slots=1,
        initial_capital=10_000.0,
        cost=_zero_cost(),
        cash_buffer=0.0,
    )

    sells = result.trades.query("side == 'sell'").reset_index(drop=True)
    assert sells.loc[0, "exec_dt"] == pd.Timestamp("2024-01-05")
    assert sells.loc[0, "status"] == blocked_status
    assert sells.loc[0, "filled_shares"] == 0
    assert sells.loc[1, "exec_dt"] == pd.Timestamp("2024-01-08")
    assert sells.loc[1, "status"] == "filled"
    assert sells.loc[1, "decision_dt"] == pd.Timestamp("2024-01-04")
    assert result.equity.set_index("dt").loc[pd.Timestamp("2024-01-08"), "pending_sells"] == 0


def test_blocked_sells_cannot_make_holdings_exceed_slot_limit():
    dates = [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ]
    market = _market(
        dates,
        {
            "A": {
                "open": [10.0, 10.0, 10.0, 9.0, 9.2, 9.3, 9.4],
                "close": [10.0, 10.0, 10.0, 9.0, 9.2, 9.3, 9.4],
                "pre_close": [10.0, 10.0, 10.0, 10.0, 9.0, 9.2, 9.3],
            },
            "B": {
                "open": [1.0] * 7,
                "close": [1.0] * 7,
                "pre_close": [1.0] * 7,
            },
        },
    )
    entry = _plan("2024-01-02", "2024-01-03", ("A",))
    rotate = _plan("2024-01-04", "2024-01-05", ("B",), extra_adv={"A": 1_000_000_000.0})
    terminal = _plan("2024-01-09", "2024-01-10", ("B",))

    result = xs.simulate_portfolio(
        market,
        {entry.exec_dt: entry, rotate.exec_dt: rotate, terminal.exec_dt: terminal},
        "F",
        slots=1,
        initial_capital=10_050.0,
        cost=_zero_cost(),
        cash_buffer=0.005,
    )

    assert result.equity["n_holdings"].max() <= 1
    blocked_day = result.holdings[result.holdings["dt"] == pd.Timestamp("2024-01-05")]
    assert set(blocked_day["symbol"]) == {"A"}


def test_factor_ic_freezes_top20_before_future_open_availability_filter():
    symbols = [f"S{i:02d}" for i in range(1, 22)]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows = []
    for symbol in symbols:
        for dt in dates:
            open_px = 10.0
            if symbol == "S01" and dt == pd.Timestamp("2024-01-05"):
                open_px = np.nan
            if symbol == "S21" and dt == pd.Timestamp("2024-01-05"):
                open_px = 20.0
            rows.append(
                {
                    "symbol": symbol,
                    "dt": dt,
                    "open": open_px,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "is_st": False,
                }
            )
    market = xs.MarketData.from_long_frame(pd.DataFrame(rows), pd.DatetimeIndex(dates), symbols)
    ranked = pd.DataFrame(
        {
            "symbol": symbols,
            "dt": pd.Timestamp("2024-01-02"),
            "factor_rank": np.arange(1, 22),
            "factor_score": np.linspace(1.0, 0.0, 21),
            "mom_120_20": np.linspace(1.0, 0.0, 21),
            "lowvol_60": np.linspace(1.0, 0.0, 21),
        }
    )
    schedule = pd.DataFrame(
        {
            "decision_dt": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "exec_dt": pd.to_datetime(["2024-01-03", "2024-01-05"]),
        }
    )

    diagnostic = xs.factor_ic_table(ranked, schedule, market).iloc[0]

    assert diagnostic["top20_label_coverage"] == pytest.approx(0.95)
    assert diagnostic["top20_missing_future_open"] == 1
    assert diagnostic["top20_equal_weight_return"] == pytest.approx(0.0)


def test_cost_components_and_round_trip_nav_reconcile_exactly():
    cost = xs.CostModel(
        commission_bps=2.5,
        transfer_bps=0.1,
        min_commission_cny=5.0,
        stamp_before_bps=10.0,
        stamp_after_bps=5.0,
        base_slippage_bps=10.0,
        impact_bps_at_1pct=15.0,
        max_adv_participation=0.05,
    )
    buy = xs._execution_terms("buy", 10.0, 100, 100_000.0, pd.Timestamp("2024-01-03"), cost)
    sell = xs._execution_terms("sell", 10.0, 100, 100_000.0, pd.Timestamp("2023-01-03"), cost)
    for terms in (buy, sell):
        assert terms["total_cost"] == pytest.approx(
            terms["commission"] + terms["transfer"] + terms["stamp"] + terms["base_slippage"] + terms["impact"]
        )
    stressed = xs._execution_terms(
        "buy",
        10.0,
        100,
        100_000.0,
        pd.Timestamp("2024-01-03"),
        xs.CostModel(**{**xs.asdict(cost), "multiplier": 2.0}),
    )
    assert stressed["commission"] == pytest.approx(2 * buy["commission"])
    assert stressed["base_slippage"] == pytest.approx(2 * buy["base_slippage"])
    assert xs._max_fill_shares(10_000, 10.0, 100_000.0, cost) == 500

    slippage_only = xs.CostModel(
        commission_bps=0.0,
        transfer_bps=0.0,
        min_commission_cny=0.0,
        stamp_before_bps=0.0,
        stamp_after_bps=0.0,
        base_slippage_bps=10.0,
        impact_bps_at_1pct=0.0,
        max_adv_participation=1.0,
    )
    market = _market(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        {"A": {"open": [10.0] * 4, "close": [10.0] * 4, "pre_close": [10.0] * 4}},
    )
    entry = _plan("2024-01-02", "2024-01-03", ("A",))
    exit_plan = _plan("2024-01-04", "2024-01-05", (), extra_adv={"A": 1_000_000_000.0})
    result = xs.simulate_portfolio(
        market,
        {entry.exec_dt: entry, exit_plan.exec_dt: exit_plan},
        "F",
        slots=1,
        initial_capital=10_050.0,
        cost=slippage_only,
        cash_buffer=0.0,
    )
    filled = result.trades[result.trades["filled_shares"] > 0]
    assert result.equity.iloc[-1]["nav"] == pytest.approx(10_030.0)
    assert result.equity["daily_cost"].sum() == pytest.approx(20.0)
    assert filled["total_cost"].sum() == pytest.approx(20.0)
