from __future__ import annotations

# ruff: noqa: E402, I001

from copy import deepcopy
from dataclasses import asdict
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_execution_v2_1 as execution


REFERENCE_CAPITAL = 10_000_000.0
CAPACITY_CAPITAL = 100_000_000.0
PROTOCOL_SHA256 = hashlib.sha256(
    execution.canonical_json_bytes(
        __import__("json").loads((SCRIPTS_DIR / "xs_chan_protocol_v2_1.json").read_text(encoding="utf-8"))
    )
).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cost(*, multiplier: float = 0.0, adv_cap: float = 0.05, auction_cap: float = 0.10) -> dict:
    return asdict(
        execution.CostModelV21(
            max_adv_participation=adv_cap,
            official_open_auction_turnover_participation_cap=auction_cap,
            cost_multiplier=multiplier,
        )
    )


def _open_row(
    symbol: str,
    *,
    price: float = 10.0,
    status: str = "trading",
    adv: float = 1_000_000_000.0,
    auction: float = 1_000_000_000.0,
    lot_size: int = 100,
) -> dict:
    return {
        "symbol": symbol,
        "open": price if status == "trading" else 0.0,
        "pre_close": price,
        "status": status,
        "limit_up": price * 2.0,
        "limit_down": price * 0.5,
        "adv20_cny_asof_decision": adv,
        "open_auction_turnover_cny": auction,
        "lot_size": lot_size,
        "source_row_sha256": _digest(f"open:{symbol}:{price}:{status}:{adv}:{auction}"),
    }


def _eod_row(symbol: str, *, close: float = 10.0, status: str = "trading") -> dict:
    return {
        "symbol": symbol,
        "close": close if status == "trading" else 0.0,
        "status": status,
        "source_row_sha256": _digest(f"eod:{symbol}:{close}:{status}"),
    }


def _decision(
    decision_session: str,
    execution_session: str,
    symbols: list[str],
    sizing_nav: float,
    *,
    slots: int = 50,
    cash_buffer: float = 0.005,
    new_entries: list[str] | None = None,
    gate_opportunities: int | None = None,
) -> dict:
    new_entries = list(symbols) if new_entries is None else new_entries
    return {
        "decision_session": decision_session,
        "execution_session": execution_session,
        "ordered_symbols": symbols,
        "new_entry_symbols": new_entries,
        "gate_eligible_new_entry_opportunities": (
            len(new_entries) if gate_opportunities is None else gate_opportunities
        ),
        "slots": slots,
        "cash_buffer_fraction": cash_buffer,
        "sizing_nav_cny": sizing_nav,
        "sizing_nav_record_sha256": _digest(f"nav:{decision_session}:{sizing_nav}"),
        "selection_identity_sha256": execution.selection_identity(decision_session, execution_session, symbols),
        "decision_record_sha256": _digest(f"decision:{decision_session}:{execution_session}:{symbols}"),
    }


def _session(
    session: str,
    rows: list[dict],
    *,
    closes: dict[str, tuple[float, str]] | None = None,
    decision: dict | None = None,
    actions: list[dict] | None = None,
    settlement_session: str | None = None,
) -> dict:
    closes = closes or {str(row["symbol"]): (float(row["pre_close"]), str(row["status"])) for row in rows}
    return {
        "session": session,
        "settlement_session": settlement_session or session,
        "open_snapshot": rows,
        "eod_snapshot": [_eod_row(symbol, close=close, status=status) for symbol, (close, status) in closes.items()],
        "corporate_actions": actions or [],
        "decision": decision,
    }


def _input(
    sessions: list[dict],
    *,
    scenario_id: str = "gross",
    initial_capital: float = REFERENCE_CAPITAL,
    cost: dict | None = None,
) -> dict:
    return {
        "schema": execution.EXECUTION_INPUT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "data_snapshot_sha256": _digest("data-snapshot"),
        "trial_id": PROTOCOL_SHA256,
        "arm_id": "FC",
        "scenario_id": scenario_id,
        "initial_capital_cny": initial_capital,
        "annual_cash_yield": 0.0,
        "terminal_policy": "mark_to_market_no_forced_liquidation",
        "cost_model": cost or _cost(),
        "sessions": sessions,
    }


def _anchor_and_buy_sessions(
    *,
    symbols: list[str],
    opens: dict[str, dict] | None = None,
    closes: dict[str, tuple[float, str]] | None = None,
) -> list[dict]:
    opens = opens or {symbol: _open_row(symbol) for symbol in symbols}
    rows = [opens[symbol] for symbol in symbols]
    return [
        _session("2026-01-02", rows, closes=dict.fromkeys(symbols, (10.0, "trading"))),
        _session(
            "2026-01-05",
            rows,
            closes=closes,
            decision=_decision("2026-01-02", "2026-01-05", symbols, REFERENCE_CAPITAL),
        ),
    ]


def _order_events(result: dict, *, session: str | None = None) -> list[dict]:
    events = [event for event in result["events"] if event["event_type"] == "order"]
    return events if session is None else [event for event in events if event["session"] == session]


def test_replay_starts_each_book_empty_and_sizes_new_entries_from_decision_close_nav():
    result = execution.replay_execution(_input(_anchor_and_buy_sessions(symbols=["A"])))

    assert result["initial_state"] == {
        "schema": execution.PORTFOLIO_STATE_SCHEMA,
        "cash_cny": REFERENCE_CAPITAL,
        "positions": {},
        "pending_sells": {},
        "last_close": {},
        "cash_receivables": {},
        "dividend_tax_liabilities": {},
        "other_assets": [],
        "other_liabilities": [],
        "borrowed_cash_cny": 0.0,
    }
    buy = _order_events(result)[0]
    # 10m * 0.995 / 50 / raw open 10, rounded down to the 100-share board lot.
    assert buy["requested_shares"] == 19_900
    assert buy["filled_shares"] == 19_900
    assert result["final_state"]["positions"]["A"]["shares"] == 19_900

    tampered = _input(_anchor_and_buy_sessions(symbols=["A"]))
    tampered["sessions"][1]["decision"]["sizing_nav_cny"] += 1.0
    with pytest.raises(execution.ExecutionReplayError, match="decision-close NAV"):
        execution.replay_execution(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [("slots", 49), ("slots", 51), ("cash_buffer_fraction", 0.0), ("cash_buffer_fraction", 0.01)],
)
def test_frozen_fifty_slot_and_cash_buffer_contract_cannot_be_changed(field: str, value: float):
    value_input = _input(_anchor_and_buy_sessions(symbols=["A"]))
    value_input["sessions"][1]["decision"][field] = value

    with pytest.raises(execution.ExecutionReplayError, match="50|0.005|frozen"):
        execution.replay_execution(value_input)


def test_retained_position_is_not_topped_up_or_trimmed_at_next_rebalance():
    sessions = _anchor_and_buy_sessions(symbols=["A"], closes={"A": (12.0, "trading")})
    # 19,900 shares rose by CNY 2 between the execution open and decision close.
    next_decision_nav = REFERENCE_CAPITAL + 19_900 * 2.0
    sessions.append(
        _session(
            "2026-01-06",
            [_open_row("A", price=8.0)],
            closes={"A": (8.0, "trading")},
            decision=_decision("2026-01-05", "2026-01-06", ["A"], next_decision_nav, new_entries=[]),
        )
    )

    result = execution.replay_execution(_input(sessions))

    assert result["final_state"]["positions"]["A"]["shares"] == 19_900
    assert _order_events(result, session="2026-01-06") == []


def test_open_fill_uses_both_prior_adv_and_official_auction_capacity_caps():
    opens = {
        # ADV cap: 100k * 5% / CNY 10 = 500 shares.
        "ADV": _open_row("ADV", adv=100_000.0, auction=1_000_000.0),
        # Auction cap: 20k * 10% / CNY 10 = 200 shares.
        "AUC": _open_row("AUC", adv=1_000_000.0, auction=20_000.0),
    }
    result = execution.replay_execution(
        _input(
            _anchor_and_buy_sessions(symbols=["ADV", "AUC"], opens=opens),
            cost=_cost(multiplier=2.0),
        )
    )
    fills = {event["symbol"]: event for event in _order_events(result)}

    # The cap is on actual fill notional, not raw-open notional.  Positive
    # slippage means the boundary board lot (500/200 shares) would exceed it.
    assert fills["ADV"]["filled_shares"] == 400
    assert fills["ADV"]["fill_notional_cny"] <= 5_000.0
    assert fills["ADV"]["status"] == "partial"
    assert fills["AUC"]["filled_shares"] == 100
    assert fills["AUC"]["fill_notional_cny"] <= 2_000.0
    assert fills["AUC"]["status"] == "partial"


def test_full_exits_execute_before_buys_and_blocked_sell_retries_on_next_session():
    sessions = _anchor_and_buy_sessions(symbols=["OLD"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("OLD", status="suspended"), _open_row("NEW")],
                closes={"OLD": (10.0, "suspended"), "NEW": (10.0, "trading")},
                decision=_decision("2026-01-05", "2026-01-06", ["NEW"], REFERENCE_CAPITAL),
            ),
            _session(
                "2026-01-07",
                [_open_row("OLD"), _open_row("NEW")],
                closes={"OLD": (10.0, "trading"), "NEW": (10.0, "trading")},
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    day_one = _order_events(result, session="2026-01-06")
    day_two = _order_events(result, session="2026-01-07")

    assert [(event["side"], event["symbol"], event["status"]) for event in day_one] == [
        ("sell", "OLD", "suspended"),
        ("buy", "NEW", "filled"),
    ]
    assert [(event["side"], event["symbol"], event["status"]) for event in day_two] == [("sell", "OLD", "filled")]
    assert day_two[0]["decision_session"] == "2026-01-05"
    assert "OLD" not in result["final_state"]["positions"]


def test_a_blocked_exit_occupies_one_of_the_fifty_slots():
    first_symbols = ["OLD"]
    sessions = _anchor_and_buy_sessions(symbols=first_symbols, closes={"OLD": (1.0, "trading")})
    entrants = [f"N{index:02d}" for index in range(50)]
    rows = [_open_row("OLD", price=1.0, status="suspended"), *[_open_row(symbol) for symbol in entrants]]
    decision_nav = REFERENCE_CAPITAL - 19_900 * 9.0
    sessions.append(
        _session(
            "2026-01-06",
            rows,
            closes={"OLD": (10.0, "suspended"), **dict.fromkeys(entrants, (10.0, "trading"))},
            decision=_decision("2026-01-05", "2026-01-06", entrants, decision_nav),
        )
    )

    result = execution.replay_execution(_input(sessions))

    assert result["daily"][-1]["holding_count"] == 50
    assert any(event["status"] == "maximum_actual_positions" for event in _order_events(result, session="2026-01-06"))


def test_fifo_dividend_tax_uses_natural_calendar_month_and_year_boundaries():
    assert execution._dividend_tax_rate("2024-01-31", "2024-02-29") == pytest.approx(0.20)
    assert execution._dividend_tax_rate("2024-01-31", "2024-03-01") == pytest.approx(0.10)
    assert execution._dividend_tax_rate("2024-02-29", "2025-02-28") == pytest.approx(0.10)
    assert execution._dividend_tax_rate("2024-02-29", "2025-03-01") == pytest.approx(0.0)

    position = execution.PositionV21(
        lots=[
            execution.TaxLotV21("old", "2025-01-01", 100, 1_000.0, 100.0),
            execution.TaxLotV21("new", "2025-02-15", 100, 2_000.0, 200.0),
        ]
    )
    basis, tax = execution._dispose_fifo_lots(position, 150, "2025-03-01")

    assert basis == pytest.approx(2_000.0)
    assert tax == pytest.approx(100.0 * 0.10 + 100.0 * 0.20)
    assert [(lot.lot_id, lot.shares) for lot in position.lots] == [("new", 50)]


def test_cash_dividend_is_receivable_until_payment_and_tax_is_settled_on_fifo_disposal():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    entitlement = {
        "action_id": "div-1",
        "symbol": "A",
        "effective_session": "2026-01-06",
        "action_type": "cash_dividend",
        "source_row_sha256": _digest("div-1-source"),
        "gross_cash_per_share": 1.0,
        "payment_session": "2026-01-07",
    }
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A")],
                closes={"A": (10.0, "trading")},
                actions=[entitlement],
            ),
            _session(
                "2026-01-07",
                [_open_row("A")],
                closes={"A": (10.0, "trading")},
            ),
            _session(
                "2026-01-08",
                [_open_row("A")],
                closes={"A": (10.0, "trading")},
                decision=_decision(
                    "2026-01-07",
                    "2026-01-08",
                    [],
                    REFERENCE_CAPITAL + 19_900 * 0.8,
                    new_entries=[],
                ),
            ),
        ]
    )
    result = execution.replay_execution(_input(sessions))
    day_entitlement = result["daily"][2]
    day_payment = result["daily"][3]
    sell = next(event for event in _order_events(result) if event["side"] == "sell")

    assert day_entitlement["cash_receivable_value_cny"] == pytest.approx(19_900.0)
    assert day_entitlement["dividend_tax_liability_cny"] == pytest.approx(3_980.0)
    assert day_payment["cash_receivable_value_cny"] == 0.0
    assert day_payment["cash_cny"] - result["daily"][2]["cash_cny"] == pytest.approx(19_900.0)
    assert sell["deferred_dividend_tax_settled_cny"] == pytest.approx(3_980.0)
    assert result["final_state"]["dividend_tax_liabilities"] == {}


@pytest.mark.parametrize(
    ("subtype", "ratio", "taxable", "expected_dates", "expected_shares"),
    [
        ("stock_dividend", 1.5, 0.2, ["2026-01-05", "2026-01-06"], [19_900, 9_950]),
        ("capital_reserve_conversion", 1.5, 0.0, ["2026-01-05", "2026-01-06"], [19_900, 9_950]),
        ("split", 2.0, 0.0, ["2026-01-05"], [39_800]),
        ("consolidation", 0.5, 0.0, ["2026-01-05"], [9_950]),
    ],
)
def test_share_change_subtypes_preserve_parent_or_replace_it_exactly_as_registered(
    subtype: str,
    ratio: float,
    taxable: float,
    expected_dates: list[str],
    expected_shares: list[int],
):
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    action = {
        "action_id": f"share-{subtype}",
        "symbol": "A",
        "effective_session": "2026-01-06",
        "action_type": "share_change",
        "source_row_sha256": _digest(f"share-{subtype}"),
        "post_to_pre_ratio": ratio,
        "fractional_cash_price": 10.0,
        "action_subtype": subtype,
        "taxable_dividend_per_pre_action_share": taxable,
        "new_share_registration_session": "2026-01-06",
    }
    sessions.append(
        _session(
            "2026-01-06",
            [_open_row("A")],
            closes={"A": (10.0, "trading")},
            actions=[action],
        )
    )

    result = execution.replay_execution(_input(sessions))
    lots = result["final_state"]["positions"]["A"]["lots"]

    assert [lot["acquisition_settlement_session"] for lot in lots] == expected_dates
    assert [lot["shares"] for lot in lots] == expected_shares
    assert sum(lot["cost_basis_cny"] for lot in lots) == pytest.approx(199_000.0)
    assert sum(lot["deferred_dividend_gross_cny"] for lot in lots) == pytest.approx(19_900 * taxable)


def test_high_cost_replay_never_borrows_cash_and_records_cash_limited_partial_buys():
    symbols = [f"S{index:02d}" for index in range(50)]
    expensive = _cost(multiplier=50.0)
    expensive["commission_bps"] = 200.0
    result = execution.replay_execution(_input(_anchor_and_buy_sessions(symbols=symbols), cost=expensive))

    assert all(day["cash_cny"] >= -1e-9 for day in result["daily"])
    assert result["final_state"]["borrowed_cash_cny"] == 0.0
    assert any(event["status"] in {"partial", "insufficient_cash"} for event in _order_events(result))


def test_gross_one_x_two_x_and_capacity_replays_keep_selection_identity_and_empty_books():
    def scenario_input(scenario_id: str, cost: dict, capital: float) -> dict:
        rows = [_open_row("A"), _open_row("B")]
        prefix_sessions = [
            _session("2026-01-02", rows),
            _session(
                "2026-01-05",
                rows,
                decision=_decision("2026-01-02", "2026-01-05", ["A"], capital),
            ),
        ]
        prefix_input = _input(
            prefix_sessions,
            scenario_id=scenario_id,
            initial_capital=capital,
            cost=cost,
        )
        scenario_nav = execution.replay_execution(prefix_input)["daily"][-1]["nav_cny"]
        prefix_sessions.append(
            _session(
                "2026-01-06",
                rows,
                decision=_decision("2026-01-05", "2026-01-06", ["B"], scenario_nav),
            )
        )
        return _input(
            prefix_sessions,
            scenario_id=scenario_id,
            initial_capital=capital,
            cost=cost,
        )

    scenario_inputs = {
        "gross": scenario_input("gross", _cost(multiplier=0.0), REFERENCE_CAPITAL),
        "1x": scenario_input("1x", _cost(multiplier=1.0), REFERENCE_CAPITAL),
        "2x": scenario_input("2x", _cost(multiplier=2.0), REFERENCE_CAPITAL),
        "capacity": scenario_input("capacity", _cost(multiplier=1.0), CAPACITY_CAPITAL),
    }
    scenarios = execution.replay_cost_scenarios(scenario_inputs)
    identities = {result["selection_identity_sequence_sha256"] for result in scenarios["scenarios"].values()}

    assert len(identities) == 1
    assert all(not result["initial_state"]["positions"] for result in scenarios["scenarios"].values())
    capacity = scenarios["scenarios"]["capacity"]
    assert capacity["selection_identity_sequence_sha256"] == scenarios["selection_identity_sequence_sha256"]
    assert capacity["initial_state"]["cash_cny"] == CAPACITY_CAPITAL

    forged = deepcopy(scenario_inputs)
    forged["2x"]["sessions"][2]["decision"]["new_entry_symbols"] = []
    with pytest.raises(execution.ExecutionReplayError, match="new-entry identities"):
        execution.replay_cost_scenarios(forged)


def test_semantic_verifier_rejects_even_one_cent_of_forged_cash():
    value_input = _input(_anchor_and_buy_sessions(symbols=["A"]))
    result = execution.replay_execution(value_input)
    result["final_state"]["cash_cny"] += 0.01

    with pytest.raises(execution.ExecutionReplayError, match="semantic replay"):
        execution.verify_execution_result(result, value_input)
