from __future__ import annotations

# ruff: noqa: E402, I001

from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
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
    arm_id: str = "FC",
    seed_id: int | None = None,
    scenario_id: str = "gross",
    initial_capital: float = REFERENCE_CAPITAL,
    cost: dict | None = None,
) -> dict:
    return {
        "schema": execution.EXECUTION_INPUT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "data_snapshot_sha256": _digest("data-snapshot"),
        "trial_id": PROTOCOL_SHA256,
        "arm_id": arm_id,
        "seed_id": seed_id,
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


def _fifty_two_cycle_window() -> tuple[list[str], list[dict], list[dict]]:
    dates = [(date(2026, 1, 2) + timedelta(days=index)).isoformat() for index in range(58)]
    rows = [_open_row("A"), _open_row("B")]
    sessions = [_session(dates[0], rows)]
    for index in range(1, 53):
        sessions.append(
            _session(
                dates[index],
                rows,
                decision=_decision(
                    dates[index - 1],
                    dates[index],
                    ["A"],
                    REFERENCE_CAPITAL,
                    new_entries=["A"] if index == 1 else [],
                ),
            )
        )
    cycles = [
        {
            "week_index": index + 1,
            "start_decision_session": dates[index],
            "end_decision_session": dates[index + 1],
        }
        for index in range(52)
    ]
    return dates, sessions, cycles


def test_replay_starts_each_book_empty_and_sizes_new_entries_from_decision_close_nav():
    result = execution.replay_execution(_input(_anchor_and_buy_sessions(symbols=["A"])))

    assert result["initial_state"] == {
        "schema": execution.PORTFOLIO_STATE_SCHEMA,
        "cash_cny": REFERENCE_CAPITAL,
        "positions": {},
        "pending_sells": {},
        "last_close": {},
        "cash_receivables": {},
        "share_entitlements": {},
        "processed_action_ids": [],
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


def test_board_lot_is_frozen_at_one_hundred_shares():
    with pytest.raises(execution.ExecutionReplayError, match="board-lot size at 100"):
        execution.replay_execution(
            _input(_anchor_and_buy_sessions(symbols=["A"], opens={"A": _open_row("A", lot_size=1)}))
        )


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
        "record_session": "2026-01-06",
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
    ("action_type", "amount_field", "expected_tax"),
    [
        ("cash_dividend", "gross_cash_per_share", 19_900 * 0.20),
        ("rights_issue", "official_disposal_proceeds_per_entitled_share", 0.0),
    ],
)
def test_cash_entitlement_is_frozen_at_record_confirmed_at_effective_and_paid_later(
    action_type: str,
    amount_field: str,
    expected_tax: float,
):
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    action = {
        "action_id": f"delayed-{action_type}",
        "symbol": "A",
        "record_session": "2026-01-06",
        "effective_session": "2026-01-07",
        "action_type": action_type,
        "source_row_sha256": _digest(f"delayed-{action_type}"),
        amount_field: 1.0,
        "payment_session": "2026-01-08",
    }
    sessions.extend(
        [
            _session("2026-01-06", [_open_row("A")], actions=[action]),
            _session("2026-01-07", [_open_row("A")]),
            _session("2026-01-08", [_open_row("A")]),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    record_day, effective_day, payment_day = result["daily"][2:5]

    assert record_day["cash_receivable_value_cny"] == 0.0
    assert record_day["dividend_tax_liability_cny"] == 0.0
    assert effective_day["cash_receivable_value_cny"] == pytest.approx(19_900.0)
    assert effective_day["dividend_tax_liability_cny"] == pytest.approx(expected_tax)
    assert payment_day["cash_receivable_value_cny"] == 0.0
    assert payment_day["cash_cny"] - effective_day["cash_cny"] == pytest.approx(19_900.0)
    confirmations = [event for event in result["events"] if event["event_type"] == "cash_receivable_confirmation"]
    assert [(event["session"], event["action_type"]) for event in confirmations] == [("2026-01-07", action_type)]


def test_pending_settlement_root_binds_every_eod_and_session_evidence_row():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A")],
                actions=[
                    {
                        "action_id": "pending-root-dividend",
                        "symbol": "A",
                        "record_session": "2026-01-06",
                        "effective_session": "2026-01-07",
                        "action_type": "cash_dividend",
                        "source_row_sha256": _digest("pending-root-dividend"),
                        "gross_cash_per_share": 1.0,
                        "payment_session": "2026-01-08",
                    }
                ],
            ),
            _session("2026-01-07", [_open_row("A")]),
            _session("2026-01-08", [_open_row("A")]),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    evidence = {row["session"]: row for row in result["session_evidence"]}
    assert [row["pending_settlement_count"] for row in result["daily"][2:5]] == [1, 2, 1]
    for row in result["daily"]:
        bound = evidence[row["session"]]
        assert bound["pending_settlements_root_sha256"] == row["pending_settlements_root_sha256"]
        assert bound["pending_settlement_count"] == row["pending_settlement_count"]

    empty_summary = execution.pending_settlements_summary(execution.empty_portfolio(REFERENCE_CAPITAL))
    assert set(empty_summary) == {
        "schema",
        "cash_receivables",
        "share_entitlements",
        "terminal_considerations",
        "dividend_tax_liabilities",
        "pending_settlement_count",
    }
    assert empty_summary["pending_settlement_count"] == 0
    assert result["daily"][0]["pending_settlements_root_sha256"] == execution.object_sha256(empty_summary)


def test_cash_entitlement_fails_closed_if_effective_session_is_missing_from_replay():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A")],
                actions=[
                    {
                        "action_id": "missed-effective",
                        "symbol": "A",
                        "record_session": "2026-01-06",
                        "effective_session": "2026-01-07",
                        "action_type": "cash_dividend",
                        "source_row_sha256": _digest("missed-effective"),
                        "gross_cash_per_share": 1.0,
                        "payment_session": "2026-01-09",
                    }
                ],
            ),
            _session("2026-01-08", [_open_row("A")]),
        ]
    )

    with pytest.raises(execution.ExecutionReplayError, match="missed its official effective session"):
        execution.replay_execution(_input(sessions))


def test_multiple_same_day_cash_entitlements_accumulate_once_per_recorded_lot():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    actions = [
        {
            "action_id": f"same-day-dividend-{index}",
            "symbol": "A",
            "record_session": "2026-01-06",
            "effective_session": "2026-01-06",
            "action_type": "cash_dividend",
            "source_row_sha256": _digest(f"same-day-dividend-{index}"),
            "gross_cash_per_share": amount,
            "payment_session": "2026-01-07",
        }
        for index, amount in enumerate((1.0, 0.5), start=1)
    ]
    sessions.extend(
        [
            _session("2026-01-06", [_open_row("A")], actions=actions),
            _session("2026-01-07", [_open_row("A")]),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    gross = 19_900 * 1.5
    assert result["daily"][2]["cash_receivable_value_cny"] == pytest.approx(gross)
    assert result["daily"][2]["dividend_tax_liability_cny"] == pytest.approx(gross * 0.20)
    assert result["daily"][3]["cash_receivable_value_cny"] == 0.0


def test_record_date_entitlement_uses_post_trade_settled_eod_lots():
    buy_action = {
        "action_id": "record-date-buy",
        "symbol": "A",
        "record_session": "2026-01-05",
        "effective_session": "2026-01-05",
        "action_type": "cash_dividend",
        "source_row_sha256": _digest("record-date-buy"),
        "gross_cash_per_share": 1.0,
        "payment_session": "2026-01-06",
    }
    buy_sessions = _anchor_and_buy_sessions(symbols=["A"])
    buy_sessions[1]["corporate_actions"] = [buy_action]
    buy_sessions.append(_session("2026-01-06", [_open_row("A")]))

    bought_on_record = execution.replay_execution(_input(buy_sessions))
    assert bought_on_record["daily"][1]["cash_receivable_value_cny"] == pytest.approx(19_900.0)
    buy_event_index = next(
        index
        for index, event in enumerate(bought_on_record["events"])
        if event["event_type"] == "order" and event["side"] == "buy"
    )
    record_event_index = next(
        index
        for index, event in enumerate(bought_on_record["events"])
        if event["event_type"] == "corporate_action" and event["action_id"] == "record-date-buy"
    )
    assert buy_event_index < record_event_index

    sell_sessions = _anchor_and_buy_sessions(symbols=["A"])
    sell_sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A")],
                decision=_decision("2026-01-05", "2026-01-06", [], REFERENCE_CAPITAL, new_entries=[]),
                actions=[
                    {
                        **buy_action,
                        "action_id": "record-date-sell",
                        "record_session": "2026-01-06",
                        "effective_session": "2026-01-06",
                        "payment_session": "2026-01-07",
                    }
                ],
            ),
            _session("2026-01-07", [_open_row("A")]),
        ]
    )

    sold_on_record = execution.replay_execution(_input(sell_sessions))
    assert sold_on_record["daily"][2]["cash_receivable_value_cny"] == 0.0
    assert sold_on_record["final_state"]["cash_receivables"] == {}
    assert not any(
        event["event_type"] == "cash_receivable_confirmation" and event["action_id"] == "record-date-sell"
        for event in sold_on_record["events"]
    )


@pytest.mark.parametrize(
    ("case", "expected_events", "expected_pending_settlement_count"),
    [
        (
            "cash_dividend",
            [
                ("corporate_action", "entitlement_recorded"),
                ("cash_receivable_confirmation", None),
                ("cash_receivable_payment", None),
            ],
            1,
        ),
        (
            "rights_issue",
            [
                ("corporate_action", "entitlement_recorded"),
                ("cash_receivable_confirmation", None),
                ("cash_receivable_payment", None),
            ],
            0,
        ),
        (
            "share_change",
            [("corporate_action", "entitlement_recorded"), ("corporate_action", "registered")],
            0,
        ),
        (
            "terminal",
            [("corporate_action", "consideration_recorded"), ("terminal_consideration_settlement", None)],
            0,
        ),
    ],
)
def test_same_day_record_effective_and_settlement_equalities_are_fully_replayed(
    case: str,
    expected_events: list[tuple[str, str | None]],
    expected_pending_settlement_count: int,
):
    action_id = f"same-day-{case}"
    common = {
        "action_id": action_id,
        "symbol": "A",
        "effective_session": "2026-01-06",
        "source_row_sha256": _digest(action_id),
    }
    if case == "cash_dividend":
        action = {
            **common,
            "record_session": "2026-01-06",
            "action_type": "cash_dividend",
            "gross_cash_per_share": 1.0,
            "payment_session": "2026-01-06",
        }
    elif case == "rights_issue":
        action = {
            **common,
            "record_session": "2026-01-06",
            "action_type": "rights_issue",
            "official_disposal_proceeds_per_entitled_share": 1.0,
            "payment_session": "2026-01-06",
        }
    elif case == "share_change":
        action = {
            **common,
            "record_session": "2026-01-06",
            "action_type": "share_change",
            "post_to_pre_ratio": 1.5,
            "fractional_cash_price": 0.0,
            "action_subtype": "capital_reserve_conversion",
            "taxable_dividend_per_pre_action_share": 0.0,
            "new_share_registration_session": "2026-01-06",
        }
    else:
        action = {
            **common,
            "action_type": "delist_cash",
            "cash_per_share": 8.0,
            "terminal_reason": "official same-day cash consideration",
            "disposal_settlement_session": "2026-01-06",
        }

    action_status = "delisted" if case == "terminal" else "trading"
    next_status = "not_listed" if case == "terminal" else "trading"
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A", status=action_status)],
                closes={"A": (0.0 if case == "terminal" else 10.0, action_status)},
                actions=[action],
            ),
            _session(
                "2026-01-07",
                [_open_row("A", status=next_status)],
                closes={"A": (0.0 if case == "terminal" else 10.0, next_status)},
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    action_events = [event for event in result["events"] if event.get("action_id") == action_id]
    action_eod = result["daily"][-2]
    action_evidence = result["session_evidence"][-2]

    assert [(event["event_type"], event.get("status")) for event in action_events] == expected_events
    assert result["final_state"]["cash_receivables"] == {}
    assert result["final_state"]["share_entitlements"] == {}
    assert result["final_state"]["other_assets"] == []
    assert action_eod["pending_settlement_count"] == expected_pending_settlement_count
    assert action_evidence["pending_settlement_count"] == expected_pending_settlement_count
    assert action_evidence["pending_settlements_root_sha256"] == action_eod["pending_settlements_root_sha256"]
    if case == "share_change":
        assert result["final_state"]["positions"]["A"]["shares"] == 29_850
    elif case == "terminal":
        assert "A" not in result["final_state"]["positions"]
    else:
        assert result["final_state"]["positions"]["A"]["shares"] == 19_900


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
        "record_session": "2026-01-06",
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


def test_share_entitlement_uses_record_date_lots_and_waits_for_official_registration():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    action = {
        "action_id": "share-delayed",
        "symbol": "A",
        "record_session": "2026-01-06",
        "effective_session": "2026-01-07",
        "action_type": "share_change",
        "source_row_sha256": _digest("share-delayed"),
        "post_to_pre_ratio": 1.5,
        "fractional_cash_price": 10.0,
        "action_subtype": "stock_dividend",
        "taxable_dividend_per_pre_action_share": 0.2,
        "new_share_registration_session": "2026-01-07",
    }
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A")],
                closes={"A": (10.0, "trading")},
                actions=[action],
            ),
            _session("2026-01-07", [_open_row("A")], closes={"A": (10.0, "trading")}),
        ]
    )

    result = execution.replay_execution(_input(sessions))

    record_state = result["daily"][2]
    assert record_state["holdings"] == [{"symbol": "A", "shares": 19_900, "close": 10.0, "value_cny": 199_000.0}]
    assert result["final_state"]["positions"]["A"]["shares"] == 29_850
    assert result["final_state"]["share_entitlements"] == {}
    registered = [
        event
        for event in result["events"]
        if event["event_type"] == "corporate_action" and event["status"] == "registered"
    ]
    assert [event["session"] for event in registered] == ["2026-01-07"]


def test_corporate_action_id_cannot_be_replayed_twice():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    action = {
        "action_id": "div-duplicate",
        "symbol": "A",
        "record_session": "2026-01-06",
        "effective_session": "2026-01-06",
        "action_type": "cash_dividend",
        "source_row_sha256": _digest("div-duplicate"),
        "gross_cash_per_share": 1.0,
        "payment_session": "2026-01-09",
    }
    sessions.extend(
        [
            _session("2026-01-06", [_open_row("A")], actions=[action]),
            _session(
                "2026-01-07",
                [_open_row("A")],
                actions=[
                    {
                        **action,
                        "record_session": "2026-01-07",
                        "effective_session": "2026-01-07",
                    }
                ],
            ),
        ]
    )

    with pytest.raises(execution.ExecutionReplayError, match="already processed"):
        execution.replay_execution(_input(sessions))


def test_terminal_cash_consideration_remains_an_asset_until_official_settlement():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    terminal = {
        "action_id": "delist-cash-1",
        "symbol": "A",
        "effective_session": "2026-01-06",
        "action_type": "delist_cash",
        "source_row_sha256": _digest("delist-cash-1"),
        "cash_per_share": 8.0,
        "terminal_reason": "official cash consideration",
        "disposal_settlement_session": "2026-01-07",
    }
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A", status="delisted")],
                closes={"A": (0.0, "delisted")},
                actions=[terminal],
                settlement_session="2026-01-07",
            ),
            _session(
                "2026-01-07",
                [_open_row("A", status="not_listed")],
                closes={"A": (0.0, "not_listed")},
                settlement_session="2026-01-08",
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))

    effective_day = result["daily"][2]
    settlement_day = result["daily"][3]
    consideration = 19_900 * 8.0
    assert effective_day["other_asset_value_cny"] == pytest.approx(consideration)
    assert effective_day["cash_cny"] < settlement_day["cash_cny"]
    assert settlement_day["other_asset_value_cny"] == 0.0
    assert result["final_state"]["other_assets"] == []
    assert any(event["event_type"] == "terminal_consideration_settlement" for event in result["events"])


def test_delist_share_exit_preserves_an_independently_retained_target_position():
    sessions = _anchor_and_buy_sessions(symbols=["A", "B"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A", status="suspended"), _open_row("B")],
                closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
                decision=_decision("2026-01-05", "2026-01-06", ["B"], REFERENCE_CAPITAL, new_entries=[]),
            ),
            _session(
                "2026-01-07",
                [_open_row("A", status="delisted"), _open_row("B")],
                closes={"A": (0.0, "delisted"), "B": (10.0, "trading")},
                actions=[
                    {
                        "action_id": "retained-target-swap",
                        "symbol": "A",
                        "effective_session": "2026-01-07",
                        "action_type": "delist_share",
                        "source_row_sha256": _digest("retained-target-swap"),
                        "target_symbol": "B",
                        "target_share_ratio": 1.0,
                        "fractional_cash_price": 0.0,
                        "terminal_reason": "official merger consideration",
                        "disposal_settlement_session": "2026-01-08",
                    }
                ],
            ),
            _session(
                "2026-01-08",
                [_open_row("A", status="not_listed"), _open_row("B")],
                closes={"A": (0.0, "not_listed"), "B": (10.0, "trading")},
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    target_sell = next(
        event
        for event in _order_events(result, session="2026-01-08")
        if event["symbol"] == "B" and event["side"] == "sell"
    )
    assert target_sell["requested_shares"] == 19_900
    assert target_sell["filled_shares"] == 19_900
    assert result["final_state"]["positions"]["B"]["shares"] == 19_900
    assert result["final_state"]["pending_sells"] == {}


def test_decision_can_bind_an_exit_to_a_delist_share_consideration_before_target_receipt():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A", status="delisted"), _open_row("B")],
                closes={"A": (0.0, "delisted"), "B": (10.0, "trading")},
                actions=[
                    {
                        "action_id": "pending-window-swap",
                        "symbol": "A",
                        "effective_session": "2026-01-06",
                        "action_type": "delist_share",
                        "source_row_sha256": _digest("pending-window-swap"),
                        "target_symbol": "B",
                        "target_share_ratio": 1.0,
                        "fractional_cash_price": 0.0,
                        "terminal_reason": "official merger consideration",
                        "disposal_settlement_session": "2026-01-08",
                    }
                ],
            ),
            _session(
                "2026-01-07",
                [_open_row("A", status="not_listed"), _open_row("B")],
                closes={"A": (0.0, "not_listed"), "B": (10.0, "trading")},
                decision=_decision("2026-01-06", "2026-01-07", [], REFERENCE_CAPITAL, new_entries=[]),
            ),
            _session(
                "2026-01-08",
                [_open_row("A", status="not_listed"), _open_row("B")],
                closes={"A": (0.0, "not_listed"), "B": (10.0, "trading")},
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    sell = next(event for event in _order_events(result, session="2026-01-08") if event["side"] == "sell")
    assert sell["symbol"] == "B"
    assert sell["requested_shares"] == 19_900
    assert sell["decision_session"] == "2026-01-06"
    assert result["final_state"]["positions"] == {}
    assert result["final_state"]["other_assets"] == []


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


def test_execution_seed_is_part_of_arm_identity_and_cost_scenario_closure():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    seeded = _input(sessions, arm_id="R_match", seed_id=20260720)
    result = execution.replay_execution(seeded)
    assert result["arm_id"] == "R_match"
    assert result["seed_id"] == 20260720

    with pytest.raises(execution.ExecutionReplayError, match="seed_id=null"):
        execution.replay_execution(_input(sessions, arm_id="FC", seed_id=20260720))
    with pytest.raises(execution.ExecutionReplayError, match="frozen seed"):
        execution.replay_execution(_input(sessions, arm_id="R_match", seed_id=None))

    scenarios = {
        "gross": _input(sessions, arm_id="R_match", seed_id=20260720, scenario_id="gross"),
        "1x": _input(sessions, arm_id="R_match", seed_id=20260720, scenario_id="1x"),
    }
    forged = deepcopy(scenarios)
    forged["1x"]["seed_id"] = 20260721
    with pytest.raises(execution.ExecutionReplayError, match="seed identity"):
        execution.replay_cost_scenarios(forged)


def test_registered_new_entry_identity_is_not_rederived_from_local_scenario_holdings():
    sessions = _anchor_and_buy_sessions(symbols=["A"])
    sessions.append(
        _session(
            "2026-01-06",
            [_open_row("A")],
            decision=_decision(
                "2026-01-05",
                "2026-01-06",
                ["A"],
                REFERENCE_CAPITAL,
                new_entries=["A"],
            ),
        )
    )
    result = execution.replay_execution(_input(sessions))

    assert result["decision_evidence"][-1]["actual_holdings_before"] == ["A"]
    assert result["decision_evidence"][-1]["new_entry_symbols"] == ["A"]
    assert _order_events(result, session="2026-01-06") == []


def test_repeated_blocked_exit_preserves_oldest_originating_decision():
    sessions = _anchor_and_buy_sessions(symbols=["A", "B"])
    suspended_rows = [_open_row("A", status="suspended"), _open_row("B")]
    sessions.extend(
        [
            _session(
                "2026-01-06",
                suspended_rows,
                closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
                decision=_decision("2026-01-05", "2026-01-06", ["B"], REFERENCE_CAPITAL, new_entries=[]),
            ),
            _session(
                "2026-01-07",
                suspended_rows,
                closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
            ),
            _session(
                "2026-01-08",
                suspended_rows,
                closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
                decision=_decision("2026-01-07", "2026-01-08", ["B"], REFERENCE_CAPITAL, new_entries=[]),
            ),
        ]
    )
    result = execution.replay_execution(_input(sessions))
    assert result["final_state"]["pending_sells"]["A"]["originating_decision_session"] == "2026-01-05"


def test_same_day_share_registration_updates_pending_sell_and_eod_evidence():
    sessions = _anchor_and_buy_sessions(symbols=["A", "B"])
    sessions.extend(
        [
            _session(
                "2026-01-06",
                [_open_row("A", status="suspended"), _open_row("B")],
                closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
                decision=_decision("2026-01-05", "2026-01-06", ["B"], REFERENCE_CAPITAL, new_entries=[]),
            ),
            _session(
                "2026-01-07",
                [_open_row("A", status="delisted"), _open_row("B")],
                closes={"A": (0.0, "delisted"), "B": (10.0, "trading")},
                actions=[
                    {
                        "action_id": "swap-into-retained-B",
                        "symbol": "A",
                        "effective_session": "2026-01-07",
                        "action_type": "delist_share",
                        "source_row_sha256": _digest("swap-into-retained-B"),
                        "target_symbol": "B",
                        "target_share_ratio": 1.0,
                        "fractional_cash_price": 0.0,
                        "terminal_reason": "official merger consideration",
                        "disposal_settlement_session": "2026-01-08",
                    }
                ],
            ),
            _session(
                "2026-01-08",
                [_open_row("A", status="not_listed"), _open_row("B", status="suspended")],
                closes={"A": (0.0, "not_listed"), "B": (10.0, "suspended")},
                actions=[
                    {
                        "action_id": "same-day-B-capital-conversion",
                        "symbol": "B",
                        "record_session": "2026-01-08",
                        "effective_session": "2026-01-08",
                        "action_type": "share_change",
                        "source_row_sha256": _digest("same-day-B-capital-conversion"),
                        "post_to_pre_ratio": 2.0,
                        "fractional_cash_price": 0.0,
                        "action_subtype": "capital_reserve_conversion",
                        "taxable_dividend_per_pre_action_share": 0.0,
                        "new_share_registration_session": "2026-01-08",
                    }
                ],
            ),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    evidence = result["session_evidence"][-1]
    final_pending = result["final_state"]["pending_sells"]

    assert final_pending == {
        "B": {
            "target_shares": 39_800,
            "originating_decision_session": "2026-01-05",
        }
    }
    assert result["final_state"]["positions"]["B"]["shares"] == 79_600
    assert evidence["pending_sell_count_after"] == 1
    assert evidence["eod_pending_sell_count"] == 1
    assert evidence["pending_sells_after_open_sha256"] != evidence["eod_pending_sells_sha256"]
    assert evidence["eod_pending_sells_sha256"] == execution.object_sha256(final_pending)
    assert any(
        event["event_type"] == "corporate_action"
        and event["action_id"] == "same-day-B-capital-conversion"
        and event["status"] == "registered"
        for event in result["events"]
    )


def test_symbol_level_unwind_tracks_delist_share_receipt_and_sells_the_replacement():
    dates, sessions, cycles = _fifty_two_cycle_window()

    suspended_rows = [_open_row("A", status="suspended"), _open_row("B")]
    sessions.append(
        _session(
            dates[53],
            suspended_rows,
            closes={"A": (10.0, "suspended"), "B": (10.0, "trading")},
            decision=_decision(dates[52], dates[53], [], REFERENCE_CAPITAL, new_entries=[]),
        )
    )
    sessions.append(
        _session(
            dates[54],
            [_open_row("A", status="delisted"), _open_row("B")],
            closes={"A": (0.0, "delisted"), "B": (10.0, "trading")},
            actions=[
                {
                    "action_id": "swap-A-to-B",
                    "symbol": "A",
                    "effective_session": dates[54],
                    "action_type": "delist_share",
                    "source_row_sha256": _digest("swap-A-to-B"),
                    "target_symbol": "B",
                    "target_share_ratio": 1.0,
                    "fractional_cash_price": 0.0,
                    "terminal_reason": "official merger consideration",
                    "disposal_settlement_session": dates[55],
                }
            ],
        )
    )
    sessions.append(
        _session(
            dates[55],
            [_open_row("A", status="not_listed"), _open_row("B")],
            closes={"A": (0.0, "not_listed"), "B": (10.0, "trading")},
        )
    )
    result = execution.replay_execution(_input(sessions))

    component = execution.build_weekly_path_component(result, cycles)
    unwind = component["path"]["post_window_unwind"]

    assert unwind["starting_positions"] == [{"symbol": "A", "shares": 19_900}]
    assert unwind["pending_share_conversions"] == []
    assert unwind["completion_session"] == dates[55]
    assert unwind["sessions"][1]["pending_settlement_count"] == 1
    assert unwind["sessions"][2]["pending_settlement_count"] == 0
    assert unwind["sessions"][0]["sell_orders"] == [{"symbol": "A", "requested_shares": 19_900, "filled_shares": 0}]
    assert unwind["sessions"][1]["terminal_dispositions"] == [
        {
            "event_index": unwind["sessions"][1]["terminal_dispositions"][0]["event_index"],
            "action_id": "swap-A-to-B",
            "action_type": "delist_share",
            "symbol": "A",
            "disposed_shares": 19_900,
        }
    ]
    assert unwind["sessions"][2]["terminal_share_receipts"][0]["target_symbol"] == "B"
    assert unwind["sessions"][2]["sell_orders"] == [
        {"symbol": "B", "requested_shares": 19_900, "filled_shares": 19_900}
    ]
    assert unwind["sessions"][2]["remaining_positions"] == []
    assert result["final_state"]["positions"] == {}
    assert result["final_state"]["pending_sells"] == {}


def test_unwind_waits_for_cash_receivable_and_validates_post_completion_heartbeats():
    dates, sessions, cycles = _fifty_two_cycle_window()
    sessions[52]["corporate_actions"] = [
        {
            "action_id": "window-end-dividend",
            "symbol": "A",
            "record_session": dates[52],
            "effective_session": dates[52],
            "action_type": "cash_dividend",
            "source_row_sha256": _digest("window-end-dividend"),
            "gross_cash_per_share": 1.0,
            "payment_session": dates[54],
        }
    ]
    window_result = execution.replay_execution(_input(sessions))
    window_end_nav = window_result["daily"][-1]["nav_cny"]
    rows = [_open_row("A"), _open_row("B")]
    sessions.extend(
        [
            _session(
                dates[53],
                rows,
                decision=_decision(dates[52], dates[53], [], window_end_nav, new_entries=[]),
            ),
            _session(dates[54], rows),
        ]
    )

    result = execution.replay_execution(_input(sessions))
    component = execution.build_weekly_path_component(result, cycles)
    unwind = component["path"]["post_window_unwind"]

    assert unwind["starting_pending_settlement_count"] == 2
    assert unwind["sessions"][0]["remaining_positions"] == []
    assert unwind["sessions"][0]["pending_settlement_count"] == 1
    assert unwind["sessions"][1]["pending_settlement_count"] == 0
    assert unwind["completion_session"] == dates[54]

    with_extra_empty_day = deepcopy(sessions)
    with_extra_empty_day.append(_session(dates[55], rows))
    extra_result = execution.replay_execution(_input(with_extra_empty_day))
    extra_component = execution.build_weekly_path_component(extra_result, cycles)
    extra_unwind = extra_component["path"]["post_window_unwind"]
    assert extra_unwind["completion_session"] == dates[54]
    assert [row["session"] for row in extra_unwind["sessions"]] == [dates[53], dates[54]]
    assert len(extra_unwind["post_completion_heartbeats"]) == 1
    heartbeat = extra_unwind["post_completion_heartbeats"][0]
    assert heartbeat["session"] == dates[55]
    assert heartbeat["state_sha256"] == extra_unwind["completion_state_sha256"]
    assert heartbeat["nav_cny"] == extra_unwind["completion_economic_state"]["nav_cny"]
    assert heartbeat["cash_cny"] == extra_unwind["completion_economic_state"]["cash_cny"]
    assert heartbeat["pending_settlement_count"] == 0
    assert heartbeat["holdings"] == []
    assert heartbeat["economic_state_sha256"] == extra_unwind["completion_economic_state_sha256"]
    assert heartbeat["administrative_events"] == []
    assert heartbeat["session_events_root_sha256"] == execution.object_sha256([])
    assert heartbeat["session_event_count"] == 0

    with_administrative_heartbeat = deepcopy(sessions)
    with_administrative_heartbeat.append(
        _session(
            dates[55],
            rows,
            actions=[
                {
                    "action_id": "empty-book-dividend",
                    "symbol": "A",
                    "record_session": dates[55],
                    "effective_session": dates[55],
                    "action_type": "cash_dividend",
                    "source_row_sha256": _digest("empty-book-dividend"),
                    "gross_cash_per_share": 1.0,
                    "payment_session": dates[56],
                }
            ],
        )
    )
    administrative_result = execution.replay_execution(_input(with_administrative_heartbeat))
    administrative_component = execution.build_weekly_path_component(administrative_result, cycles)
    administrative_unwind = administrative_component["path"]["post_window_unwind"]
    administrative_row = administrative_unwind["post_completion_heartbeats"][0]
    assert administrative_unwind["completion_session"] == dates[54]
    assert administrative_row["state_sha256"] != administrative_unwind["completion_state_sha256"]
    assert administrative_row["economic_state_sha256"] == administrative_unwind["completion_economic_state_sha256"]
    assert administrative_row["session_event_count"] == 1
    assert administrative_row["administrative_events"][0]["event"]["status"] == "no_position"
    assert administrative_row["session_events_root_sha256"] == execution.object_sha256(
        administrative_row["administrative_events"]
    )

    forged_administrative_event = deepcopy(administrative_result)
    forged_administrative_event["events"][-1]["cash_change_cny"] = 0.01
    with pytest.raises(execution.ExecutionReplayError, match="event after unwind completion"):
        execution.build_weekly_path_component(forged_administrative_event, cycles)

    forged_administrative_state = deepcopy(administrative_result)
    forged_administrative_state["daily"][-1]["state_sha256"] = administrative_unwind["completion_state_sha256"]
    forged_administrative_state["session_evidence"][-1]["eod_portfolio_state_sha256"] = administrative_unwind[
        "completion_state_sha256"
    ]
    with pytest.raises(execution.ExecutionReplayError, match="must advance the state hash"):
        execution.build_weekly_path_component(forged_administrative_state, cycles)

    forged_quiet_state = deepcopy(extra_result)
    forged_quiet_state["daily"][-1]["state_sha256"] = _digest("forged-quiet-heartbeat-state")
    forged_quiet_state["session_evidence"][-1]["eod_portfolio_state_sha256"] = _digest("forged-quiet-heartbeat-state")
    with pytest.raises(execution.ExecutionReplayError, match="without an administrative action"):
        execution.build_weekly_path_component(forged_quiet_state, cycles)

    changed_heartbeat = deepcopy(extra_result)
    changed_heartbeat["daily"][-1]["cash_cny"] += 0.01
    changed_heartbeat["daily"][-1]["nav_cny"] += 0.01
    with pytest.raises(execution.ExecutionReplayError, match="economic state changes after unwind completion"):
        execution.build_weekly_path_component(changed_heartbeat, cycles)

    eventful_heartbeat = deepcopy(extra_result)
    eventful_heartbeat["events"].append(
        {
            "event_type": "cash_receivable_payment",
            "session": dates[55],
            "symbol": "A",
            "action_id": "forged-heartbeat-event",
            "cash_change_cny": 0.0,
        }
    )
    with pytest.raises(execution.ExecutionReplayError, match="event after unwind completion"):
        execution.build_weekly_path_component(eventful_heartbeat, cycles)

    orphan_event = deepcopy(result)
    orphan_event["events"].append(
        {
            "event_type": "cash_receivable_payment",
            "session": dates[55],
            "symbol": "A",
            "action_id": "forged-after-completion",
            "cash_change_cny": 1.0,
        }
    )
    with pytest.raises(execution.ExecutionReplayError, match="no matching EOD session"):
        execution.build_weekly_path_component(orphan_event, cycles)


def test_semantic_verifier_rejects_even_one_cent_of_forged_cash():
    value_input = _input(_anchor_and_buy_sessions(symbols=["A"]))
    result = execution.replay_execution(value_input)
    result["final_state"]["cash_cny"] += 0.01

    with pytest.raises(execution.ExecutionReplayError, match="semantic replay"):
        execution.verify_execution_result(result, value_input)
