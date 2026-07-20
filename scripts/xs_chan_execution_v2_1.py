"""Deterministic V2.1 portfolio and execution replay.

This module deliberately separates the information available at the opening
auction from the end-of-day valuation snapshot.  It is a pure replay engine:
all selections, market observations, corporate actions, and cost parameters
must be supplied as content-addressable inputs.  It never fetches data and it
never decides whether an observation is "good enough" on behalf of the data
validator.

The engine is intentionally independent from the frozen V2 implementation.
V2.1 starts every formal arm from an empty book, retries blocked/partial sells
on every subsequent exchange session, executes sells before buys, and carries
suspended positions only at their last independently observed close.
"""

from __future__ import annotations

import calendar as calendar_module
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

EXECUTION_INPUT_SCHEMA = "xs_chan_execution_input_v2_1"
EXECUTION_RESULT_SCHEMA = "xs_chan_execution_result_v2_1"
SCENARIO_RESULT_SCHEMA = "xs_chan_execution_scenarios_v2_1"
PORTFOLIO_STATE_SCHEMA = "xs_chan_portfolio_state_v2_1"

DEFAULT_LOT_SIZE = 100
TRADING_STATUSES = frozenset({"trading"})
NON_TRADING_STATUSES = frozenset({"suspended", "delisted", "not_listed"})
TERMINAL_ACTION_TYPES = frozenset({"delist_cash", "delist_share", "delist_writeoff"})


class ExecutionReplayError(ValueError):
    """Raised when replay inputs are incomplete, inconsistent, or non-causal."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the protocol-wide canonical representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ExecutionReplayError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ExecutionReplayError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionReplayError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ExecutionReplayError(f"{label} must be finite")
    if minimum is not None and number < minimum:
        raise ExecutionReplayError(f"{label} must be >= {minimum}")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutionReplayError(f"{label} must be a positive integer")
    return value


def _session(value: Any, label: str = "session") -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ExecutionReplayError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != str(value):
        raise ExecutionReplayError(f"{label} must use canonical YYYY-MM-DD form")
    return parsed.isoformat()


@dataclass(frozen=True)
class CostModelV21:
    """Frozen transaction and capacity assumptions for one replay scenario."""

    commission_bps: float = 2.5
    transfer_bps: float = 0.1
    min_commission_cny: float = 5.0
    stamp_duty_bps_before_2023_08_28: float = 10.0
    stamp_duty_bps_from_2023_08_28: float = 5.0
    base_slippage_bps: float = 10.0
    impact_bps_at_1pct: float = 15.0
    max_adv_participation: float = 0.05
    official_open_auction_turnover_participation_cap: float = 0.10
    cost_multiplier: float = 1.0

    def validated(self) -> CostModelV21:
        values = asdict(self)
        for key, value in values.items():
            _finite_number(value, f"cost_model.{key}", minimum=0.0)
        if not 0 < self.max_adv_participation <= 1:
            raise ExecutionReplayError("max_adv_participation must be in (0, 1]")
        if not 0 < self.official_open_auction_turnover_participation_cap <= 1:
            raise ExecutionReplayError("official_open_auction_turnover_participation_cap must be in (0, 1]")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CostModelV21:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ExecutionReplayError(
                f"cost_model keys differ: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
            )
        return cls(**{key: float(value[key]) for key in expected}).validated()


@dataclass
class TaxLotV21:
    """FIFO lot carrying both cost basis and dividend-tax entitlements."""

    lot_id: str
    acquisition_settlement_session: str
    shares: int
    cost_basis_cny: float
    deferred_dividend_gross_cny: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lot_id": self.lot_id,
            "acquisition_settlement_session": self.acquisition_settlement_session,
            "shares": int(self.shares),
            "cost_basis_cny": float(self.cost_basis_cny),
            "deferred_dividend_gross_cny": float(self.deferred_dividend_gross_cny),
        }


@dataclass
class PositionV21:
    lots: list[TaxLotV21] = field(default_factory=list)

    @property
    def shares(self) -> int:
        return int(sum(lot.shares for lot in self.lots))

    @property
    def carrying_cost_cny(self) -> float:
        return float(sum(lot.cost_basis_cny for lot in self.lots))

    def as_dict(self) -> dict[str, Any]:
        ordered = sorted(self.lots, key=lambda lot: (lot.acquisition_settlement_session, lot.lot_id))
        return {
            "shares": self.shares,
            "carrying_cost_cny": self.carrying_cost_cny,
            "lots": [lot.as_dict() for lot in ordered],
        }


@dataclass
class CashReceivableV21:
    action_id: str
    symbol: str
    payment_session: str
    gross_cash_cny: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "symbol": self.symbol,
            "payment_session": self.payment_session,
            "gross_cash_cny": float(self.gross_cash_cny),
        }


@dataclass
class PendingSellV21:
    target_shares: int
    originating_decision_session: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_shares": int(self.target_shares),
            "originating_decision_session": self.originating_decision_session,
        }


@dataclass
class PortfolioStateV21:
    """Mutable internal state with an explicitly serializable representation."""

    cash_cny: float
    positions: dict[str, PositionV21] = field(default_factory=dict)
    pending_sells: dict[str, PendingSellV21] = field(default_factory=dict)
    last_close: dict[str, float] = field(default_factory=dict)
    cash_receivables: dict[str, CashReceivableV21] = field(default_factory=dict)
    dividend_tax_liabilities: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PORTFOLIO_STATE_SCHEMA,
            "cash_cny": float(self.cash_cny),
            "positions": {symbol: self.positions[symbol].as_dict() for symbol in sorted(self.positions)},
            "pending_sells": {symbol: self.pending_sells[symbol].as_dict() for symbol in sorted(self.pending_sells)},
            "last_close": {symbol: float(self.last_close[symbol]) for symbol in sorted(self.last_close)},
            "cash_receivables": {
                action_id: self.cash_receivables[action_id].as_dict() for action_id in sorted(self.cash_receivables)
            },
            "dividend_tax_liabilities": {
                lot_id: float(self.dividend_tax_liabilities[lot_id]) for lot_id in sorted(self.dividend_tax_liabilities)
            },
            "other_assets": [],
            "other_liabilities": [],
            "borrowed_cash_cny": 0.0,
        }


def empty_portfolio(initial_capital_cny: float) -> PortfolioStateV21:
    capital = _finite_number(initial_capital_cny, "initial_capital_cny", minimum=0.0)
    if capital <= 0:
        raise ExecutionReplayError("initial_capital_cny must be positive")
    return PortfolioStateV21(cash_cny=capital)


def _normalise_open_rows(rows: Any, session: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ExecutionReplayError(f"open_snapshot for {session} must be a list")
    result: dict[str, dict[str, Any]] = {}
    exact = {
        "symbol",
        "open",
        "pre_close",
        "status",
        "limit_up",
        "limit_down",
        "adv20_cny_asof_decision",
        "open_auction_turnover_cny",
        "lot_size",
        "source_row_sha256",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != exact:
            raise ExecutionReplayError(f"open_snapshot[{index}] for {session} must have exact V2.1 keys")
        symbol = str(raw["symbol"])
        if not symbol or symbol in result:
            raise ExecutionReplayError(f"duplicate/empty open symbol {symbol!r} on {session}")
        status = str(raw["status"])
        if status not in TRADING_STATUSES | NON_TRADING_STATUSES:
            raise ExecutionReplayError(f"unknown opening status {status!r} for {symbol} on {session}")
        open_px = _finite_number(raw["open"], f"{session}.{symbol}.open", minimum=0.0)
        pre_close = _finite_number(raw["pre_close"], f"{session}.{symbol}.pre_close", minimum=0.0)
        limit_up = _finite_number(raw["limit_up"], f"{session}.{symbol}.limit_up", minimum=0.0)
        limit_down = _finite_number(raw["limit_down"], f"{session}.{symbol}.limit_down", minimum=0.0)
        adv = _finite_number(raw["adv20_cny_asof_decision"], f"{session}.{symbol}.adv20_cny_asof_decision", minimum=0.0)
        auction = _finite_number(
            raw["open_auction_turnover_cny"], f"{session}.{symbol}.open_auction_turnover_cny", minimum=0.0
        )
        lot_size = _positive_int(raw["lot_size"], f"{session}.{symbol}.lot_size")
        _require_sha256(raw["source_row_sha256"], f"{session}.{symbol}.source_row_sha256")
        if status == "trading":
            if min(open_px, pre_close, limit_up, limit_down) <= 0:
                raise ExecutionReplayError(f"trading opening row has non-positive price for {symbol} on {session}")
            if not limit_down <= open_px <= limit_up or not limit_down <= pre_close <= limit_up:
                raise ExecutionReplayError(f"opening prices fall outside official limits for {symbol} on {session}")
        elif open_px != 0:
            raise ExecutionReplayError(f"non-trading opening row must encode open=0 for {symbol} on {session}")
        result[symbol] = {
            **dict(raw),
            "symbol": symbol,
            "open": open_px,
            "pre_close": pre_close,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "adv20_cny_asof_decision": adv,
            "open_auction_turnover_cny": auction,
            "lot_size": lot_size,
            "status": status,
        }
    return result


def _normalise_eod_rows(rows: Any, session: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ExecutionReplayError(f"eod_snapshot for {session} must be a list")
    result: dict[str, dict[str, Any]] = {}
    exact = {"symbol", "close", "status", "source_row_sha256"}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != exact:
            raise ExecutionReplayError(f"eod_snapshot[{index}] for {session} must have exact V2.1 keys")
        symbol = str(raw["symbol"])
        if not symbol or symbol in result:
            raise ExecutionReplayError(f"duplicate/empty EOD symbol {symbol!r} on {session}")
        close = _finite_number(raw["close"], f"{session}.{symbol}.close", minimum=0.0)
        status = str(raw["status"])
        if status not in TRADING_STATUSES | NON_TRADING_STATUSES:
            raise ExecutionReplayError(f"unknown EOD status {status!r} for {symbol} on {session}")
        if status == "trading" and close <= 0:
            raise ExecutionReplayError(f"trading EOD row has non-positive close for {symbol} on {session}")
        if status != "trading" and close != 0:
            raise ExecutionReplayError(f"non-trading EOD row must encode close=0 for {symbol} on {session}")
        _require_sha256(raw["source_row_sha256"], f"{session}.{symbol}.source_row_sha256")
        result[symbol] = {**dict(raw), "symbol": symbol, "close": close, "status": status}
    return result


def _normalise_decision(value: Any, execution_session: str) -> dict[str, Any] | None:
    if value is None:
        return None
    exact = {
        "decision_session",
        "execution_session",
        "ordered_symbols",
        "new_entry_symbols",
        "gate_eligible_new_entry_opportunities",
        "slots",
        "cash_buffer_fraction",
        "sizing_nav_cny",
        "sizing_nav_record_sha256",
        "selection_identity_sha256",
        "decision_record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ExecutionReplayError(f"decision for {execution_session} must have exact V2.1 keys")
    decision_session = _session(value["decision_session"], "decision.decision_session")
    if _session(value["execution_session"], "decision.execution_session") != execution_session:
        raise ExecutionReplayError("decision execution_session does not match enclosing session")
    symbols = value["ordered_symbols"]
    if not isinstance(symbols, list) or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ExecutionReplayError("decision.ordered_symbols must be a list of non-empty strings")
    if len(symbols) != len(set(symbols)):
        raise ExecutionReplayError("decision.ordered_symbols must not contain duplicates")
    new_entries = value["new_entry_symbols"]
    if (
        not isinstance(new_entries, list)
        or any(not isinstance(symbol, str) or not symbol for symbol in new_entries)
        or len(new_entries) != len(set(new_entries))
        or any(symbol not in symbols for symbol in new_entries)
    ):
        raise ExecutionReplayError("decision.new_entry_symbols must be a unique ordered-symbol subset")
    opportunities = value["gate_eligible_new_entry_opportunities"]
    if isinstance(opportunities, bool) or not isinstance(opportunities, int) or opportunities < len(new_entries):
        raise ExecutionReplayError("decision gate-eligible opportunities are invalid")
    slots = _positive_int(value["slots"], "decision.slots")
    if slots != 50:
        raise ExecutionReplayError("V2.1 freezes decision slots at 50")
    if len(symbols) > slots:
        raise ExecutionReplayError("decision contains more ordered symbols than slots")
    cash_buffer = _finite_number(value["cash_buffer_fraction"], "decision.cash_buffer_fraction", minimum=0.0)
    if not 0 <= cash_buffer < 1:
        raise ExecutionReplayError("decision.cash_buffer_fraction must be in [0, 1)")
    if cash_buffer != 0.005:
        raise ExecutionReplayError("V2.1 freezes the cash buffer at 0.005")
    _require_sha256(value["selection_identity_sha256"], "decision.selection_identity_sha256")
    sizing_nav = _finite_number(value["sizing_nav_cny"], "decision.sizing_nav_cny", minimum=0.0)
    if sizing_nav <= 0:
        raise ExecutionReplayError("decision.sizing_nav_cny must be positive")
    _require_sha256(value["sizing_nav_record_sha256"], "decision.sizing_nav_record_sha256")
    _require_sha256(value["decision_record_sha256"], "decision.decision_record_sha256")
    expected_identity = object_sha256(
        {
            "decision_session": decision_session,
            "execution_session": execution_session,
            "ordered_symbols": symbols,
        }
    )
    if value["selection_identity_sha256"] != expected_identity:
        raise ExecutionReplayError("decision selection_identity_sha256 does not match the ordered selection")
    return {
        **dict(value),
        "decision_session": decision_session,
        "execution_session": execution_session,
        "ordered_symbols": list(symbols),
        "new_entry_symbols": list(new_entries),
        "gate_eligible_new_entry_opportunities": opportunities,
        "slots": slots,
        "cash_buffer_fraction": cash_buffer,
        "sizing_nav_cny": sizing_nav,
    }


def selection_identity(decision_session: str, execution_session: str, ordered_symbols: Sequence[str]) -> str:
    """Construct the cost-scenario-independent identity for a frozen selection."""

    decision = _session(decision_session, "decision_session")
    execution = _session(execution_session, "execution_session")
    symbols = [str(symbol) for symbol in ordered_symbols]
    if any(not symbol for symbol in symbols) or len(symbols) != len(set(symbols)):
        raise ExecutionReplayError("ordered_symbols must be unique non-empty strings")
    return object_sha256({"decision_session": decision, "execution_session": execution, "ordered_symbols": symbols})


def _normalise_actions(value: Any, session: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ExecutionReplayError(f"corporate_actions for {session} must be a list")
    result: list[dict[str, Any]] = []
    common = {"action_id", "symbol", "effective_session", "action_type", "source_row_sha256"}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or not common.issubset(raw):
            raise ExecutionReplayError(f"corporate_actions[{index}] for {session} is malformed")
        if _session(raw["effective_session"], "corporate_action.effective_session") != session:
            raise ExecutionReplayError("corporate action effective_session does not match enclosing session")
        action_type = str(raw["action_type"])
        expected = set(common)
        if action_type == "share_change":
            expected |= {
                "post_to_pre_ratio",
                "fractional_cash_price",
                "action_subtype",
                "taxable_dividend_per_pre_action_share",
                "new_share_registration_session",
            }
        elif action_type == "cash_dividend":
            expected |= {"gross_cash_per_share", "payment_session"}
        elif action_type == "rights_issue":
            expected |= {"official_disposal_proceeds_per_entitled_share", "payment_session"}
        elif action_type == "delist_cash":
            expected |= {"cash_per_share", "terminal_reason", "disposal_settlement_session"}
        elif action_type == "delist_share":
            expected |= {
                "target_symbol",
                "target_share_ratio",
                "fractional_cash_price",
                "terminal_reason",
                "disposal_settlement_session",
            }
        elif action_type == "delist_writeoff":
            expected |= {"terminal_reason", "no_value_evidence_sha256", "disposal_settlement_session"}
        else:
            raise ExecutionReplayError(f"unsupported corporate action type {action_type!r}")
        if set(raw) != expected:
            raise ExecutionReplayError(f"corporate action {raw.get('action_id')!r} has incorrect keys")
        _require_sha256(raw["source_row_sha256"], "corporate_action.source_row_sha256")
        symbol = str(raw["symbol"])
        action_id = str(raw["action_id"])
        if not symbol or not action_id:
            raise ExecutionReplayError("corporate action symbol/action_id must be non-empty")
        item = dict(raw)
        item["symbol"] = symbol
        item["action_id"] = action_id
        item["action_type"] = action_type
        if action_type == "share_change":
            multiplier = _finite_number(raw["post_to_pre_ratio"], "corporate_action.post_to_pre_ratio", minimum=0.0)
            fractional_cash = _finite_number(
                raw["fractional_cash_price"], "corporate_action.fractional_cash_price", minimum=0.0
            )
            if multiplier <= 0:
                raise ExecutionReplayError("share-change ratio must be positive")
            subtype = str(raw["action_subtype"])
            if subtype not in {"stock_dividend", "capital_reserve_conversion", "split", "consolidation"}:
                raise ExecutionReplayError("share-change action_subtype is not registered")
            taxable = _finite_number(
                raw["taxable_dividend_per_pre_action_share"],
                "corporate_action.taxable_dividend_per_pre_action_share",
                minimum=0.0,
            )
            if subtype == "stock_dividend" and taxable <= 0:
                raise ExecutionReplayError("taxable stock dividend requires a positive official tax basis")
            if subtype != "stock_dividend" and taxable != 0:
                raise ExecutionReplayError("non-taxable share change must have zero dividend tax basis")
            registration_session = _session(
                raw["new_share_registration_session"], "corporate_action.new_share_registration_session"
            )
            if subtype in {"stock_dividend", "capital_reserve_conversion"} and registration_session != session:
                raise ExecutionReplayError("additive share change must be applied on its official registration session")
            item.update(
                post_to_pre_ratio=multiplier,
                fractional_cash_price=fractional_cash,
                action_subtype=subtype,
                taxable_dividend_per_pre_action_share=taxable,
                new_share_registration_session=registration_session,
            )
        elif action_type == "cash_dividend":
            gross = _finite_number(raw["gross_cash_per_share"], "corporate_action.gross_cash_per_share", minimum=0.0)
            payment_session = _session(raw["payment_session"], "corporate_action.payment_session")
            if payment_session <= session:
                raise ExecutionReplayError("cash dividend payment_session must follow its entitlement date")
            item.update(gross_cash_per_share=gross, payment_session=payment_session)
        elif action_type == "rights_issue":
            proceeds = _finite_number(
                raw["official_disposal_proceeds_per_entitled_share"],
                "corporate_action.official_disposal_proceeds_per_entitled_share",
                minimum=0.0,
            )
            payment_session = _session(raw["payment_session"], "corporate_action.payment_session")
            if payment_session <= session:
                raise ExecutionReplayError("rights proceeds payment must follow entitlement")
            item.update(
                official_disposal_proceeds_per_entitled_share=proceeds,
                payment_session=payment_session,
            )
        elif action_type == "delist_cash":
            item["cash_per_share"] = _finite_number(
                raw["cash_per_share"], "corporate_action.cash_per_share", minimum=0.0
            )
            if not str(raw["terminal_reason"]):
                raise ExecutionReplayError("terminal_reason must be non-empty")
            item["disposal_settlement_session"] = _session(
                raw["disposal_settlement_session"], "corporate_action.disposal_settlement_session"
            )
            if item["disposal_settlement_session"] < session:
                raise ExecutionReplayError("terminal disposal settlement precedes effective session")
        elif action_type == "delist_share":
            target_symbol = str(raw["target_symbol"])
            if not target_symbol or target_symbol == symbol:
                raise ExecutionReplayError("delist share target must be a different non-empty security")
            target_ratio = _finite_number(raw["target_share_ratio"], "corporate_action.target_share_ratio", minimum=0.0)
            if target_ratio <= 0:
                raise ExecutionReplayError("delist target share ratio must be positive")
            item.update(
                target_symbol=target_symbol,
                target_share_ratio=target_ratio,
                fractional_cash_price=_finite_number(
                    raw["fractional_cash_price"], "corporate_action.fractional_cash_price", minimum=0.0
                ),
            )
            if not str(raw["terminal_reason"]):
                raise ExecutionReplayError("terminal_reason must be non-empty")
            item["disposal_settlement_session"] = _session(
                raw["disposal_settlement_session"], "corporate_action.disposal_settlement_session"
            )
            if item["disposal_settlement_session"] < session:
                raise ExecutionReplayError("terminal disposal settlement precedes effective session")
        else:
            _require_sha256(raw["no_value_evidence_sha256"], "corporate_action.no_value_evidence_sha256")
            if not str(raw["terminal_reason"]):
                raise ExecutionReplayError("terminal_reason must be non-empty")
            item["disposal_settlement_session"] = _session(
                raw["disposal_settlement_session"], "corporate_action.disposal_settlement_session"
            )
            if item["disposal_settlement_session"] < session:
                raise ExecutionReplayError("terminal disposal settlement precedes effective session")
        result.append(item)
    ids = [item["action_id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ExecutionReplayError(f"duplicate corporate action id on {session}")
    return sorted(result, key=lambda item: (item["symbol"], item["action_id"]))


def _fill_status(row: Mapping[str, Any], side: str) -> str:
    if row["status"] != "trading":
        return str(row["status"])
    open_px = float(row["open"])
    tolerance = max(abs(open_px), 1.0) * 1e-12
    if side == "buy" and open_px >= float(row["limit_up"]) - tolerance:
        return "limit_up_locked"
    if side == "sell" and open_px <= float(row["limit_down"]) + tolerance:
        return "limit_down_locked"
    return "fillable"


def _capacity_shares(row: Mapping[str, Any], cost: CostModelV21) -> int:
    open_px = float(row["open"])
    if open_px <= 0:
        return 0
    adv_cap = float(row["adv20_cny_asof_decision"]) * cost.max_adv_participation
    auction_cap = float(row["open_auction_turnover_cny"]) * cost.official_open_auction_turnover_participation_cap
    notional_cap = min(adv_cap, auction_cap)
    lot = int(row["lot_size"])
    return max(0, math.floor(notional_cap / open_px / lot) * lot)


def _capacity_fill_shares(
    requested: int,
    *,
    side: str,
    row: Mapping[str, Any],
    cost: CostModelV21,
    session: str,
) -> int:
    """Apply the dual cap to final fill notional, including adverse slippage."""

    lot = int(row["lot_size"])
    shares = math.floor(min(requested, _capacity_shares(row, cost)) / lot) * lot
    notional_cap = min(
        float(row["adv20_cny_asof_decision"]) * cost.max_adv_participation,
        float(row["open_auction_turnover_cny"]) * cost.official_open_auction_turnover_participation_cap,
    )
    while shares > 0:
        terms = _execution_terms(
            side=side,
            open_px=float(row["open"]),
            shares=shares,
            row=row,
            cost=cost,
            session=session,
        )
        if terms["fill_notional_cny"] <= notional_cap + max(notional_cap, 1.0) * 1e-12:
            return shares
        shares -= lot
    return 0


def _execution_terms(
    *, side: str, open_px: float, shares: int, row: Mapping[str, Any], cost: CostModelV21, session: str
) -> dict[str, float]:
    if side not in {"buy", "sell"} or shares <= 0:
        raise ExecutionReplayError("execution terms require a valid side and positive shares")
    open_notional = open_px * shares
    adv = float(row["adv20_cny_asof_decision"])
    direction = 1.0 if side == "buy" else -1.0
    fill_price = open_px
    raw_impact_bps = 0.0
    for _ in range(50):
        fill_notional_guess = fill_price * shares
        participation_guess = fill_notional_guess / adv if adv > 0 else math.inf
        raw_impact_bps = cost.impact_bps_at_1pct * math.sqrt(max(participation_guess, 0.0) / 0.01)
        slip_bps = (cost.base_slippage_bps + raw_impact_bps) * cost.cost_multiplier
        next_price = open_px * (1.0 + direction * slip_bps / 10_000.0)
        if side == "buy":
            next_price = min(next_price, float(row["limit_up"]))
        else:
            next_price = max(next_price, float(row["limit_down"]))
        if abs(next_price - fill_price) <= max(abs(next_price), 1.0) * 1e-14:
            fill_price = next_price
            break
        fill_price = next_price
    fill_notional = fill_price * shares
    participation = fill_notional / adv if adv > 0 else math.inf
    commission = (
        0.0
        if cost.cost_multiplier == 0.0
        else max(
            cost.min_commission_cny,
            fill_notional * cost.commission_bps * cost.cost_multiplier / 10_000.0,
        )
    )
    transfer = fill_notional * cost.transfer_bps * cost.cost_multiplier / 10_000.0
    stamp_rate = (
        cost.stamp_duty_bps_from_2023_08_28 if session >= "2023-08-28" else cost.stamp_duty_bps_before_2023_08_28
    )
    stamp = fill_notional * stamp_rate * cost.cost_multiplier / 10_000.0 if side == "sell" else 0.0
    slippage_cost = abs(fill_price - open_px) * shares
    return {
        "open_price": open_px,
        "fill_price": fill_price,
        "open_notional_cny": open_notional,
        "fill_notional_cny": fill_notional,
        "commission_cny": commission,
        "transfer_fee_cny": transfer,
        "stamp_duty_cny": stamp,
        "slippage_cost_cny": slippage_cost,
        "participation_of_prior_adv": participation,
        "cash_debit_credit_cny": (
            -(fill_notional + commission + transfer) if side == "buy" else fill_notional - commission - transfer - stamp
        ),
    }


def _add_calendar_month(value: date) -> date:
    year = value.year + value.month // 12
    month = value.month % 12 + 1
    day = min(value.day, calendar_module.monthrange(year, month)[1])
    return date(year, month, day)


def _add_calendar_year(value: date) -> date:
    day = min(value.day, calendar_module.monthrange(value.year + 1, value.month)[1])
    return date(value.year + 1, value.month, day)


def _dividend_tax_rate(acquisition_settlement_session: str, disposal_settlement_session: str) -> float:
    """Apply the frozen natural-month/year holding-period boundaries."""

    acquired = date.fromisoformat(acquisition_settlement_session)
    disposed = date.fromisoformat(disposal_settlement_session)
    if disposed < acquired:
        raise ExecutionReplayError("lot disposal settlement precedes acquisition settlement")
    if disposed <= _add_calendar_month(acquired):
        return 0.20
    if disposed <= _add_calendar_year(acquired):
        return 0.10
    return 0.0


def _remeasure_dividend_tax(state: PortfolioStateV21, session: str) -> float:
    liabilities: dict[str, float] = {}
    for position in state.positions.values():
        for lot in position.lots:
            if lot.deferred_dividend_gross_cny <= 0:
                continue
            amount = lot.deferred_dividend_gross_cny * _dividend_tax_rate(lot.acquisition_settlement_session, session)
            if amount > 0:
                liabilities[lot.lot_id] = amount
    state.dividend_tax_liabilities = liabilities
    return float(sum(liabilities.values()))


def _settle_due_receivables(state: PortfolioStateV21, session: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    overdue = [
        receivable.action_id for receivable in state.cash_receivables.values() if receivable.payment_session < session
    ]
    if overdue:
        raise ExecutionReplayError(f"cash dividend receivable missed its official payment session: {overdue}")
    for action_id in sorted(state.cash_receivables):
        receivable = state.cash_receivables[action_id]
        if receivable.payment_session != session:
            continue
        state.cash_cny += receivable.gross_cash_cny
        state.cash_receivables.pop(action_id)
        events.append(
            {
                "event_type": "cash_receivable_payment",
                "session": session,
                "symbol": receivable.symbol,
                "action_id": action_id,
                "cash_change_cny": receivable.gross_cash_cny,
            }
        )
    return events


def _dispose_fifo_lots(position: PositionV21, shares: int, disposal_settlement_session: str) -> tuple[float, float]:
    """Remove shares FIFO and return ``(cost_basis, deferred_tax_paid)``."""

    if shares <= 0 or shares > position.shares:
        raise ExecutionReplayError("FIFO disposal share count is invalid")
    remaining = shares
    cost_basis = 0.0
    deferred_tax = 0.0
    ordered = sorted(position.lots, key=lambda lot: (lot.acquisition_settlement_session, lot.lot_id))
    kept: list[TaxLotV21] = []
    for lot in ordered:
        if remaining <= 0:
            kept.append(lot)
            continue
        disposed = min(remaining, lot.shares)
        fraction = disposed / lot.shares
        disposed_basis = lot.cost_basis_cny * fraction
        disposed_dividend = lot.deferred_dividend_gross_cny * fraction
        cost_basis += disposed_basis
        deferred_tax += disposed_dividend * _dividend_tax_rate(
            lot.acquisition_settlement_session, disposal_settlement_session
        )
        lot.shares -= disposed
        lot.cost_basis_cny -= disposed_basis
        lot.deferred_dividend_gross_cny -= disposed_dividend
        remaining -= disposed
        if lot.shares > 0:
            kept.append(lot)
    if remaining != 0:
        raise ExecutionReplayError("FIFO lot accounting did not conserve shares")
    position.lots = kept
    return cost_basis, deferred_tax


def _apply_corporate_actions(
    state: PortfolioStateV21,
    actions: Sequence[Mapping[str, Any]],
    open_rows: Mapping[str, Mapping[str, Any]],
    session: str,
) -> list[dict[str, Any]]:
    events = _settle_due_receivables(state, session)
    for action in actions:
        symbol = str(action["symbol"])
        position = state.positions.get(symbol)
        if position is None:
            events.append(
                {
                    "event_type": "corporate_action",
                    "session": session,
                    "symbol": symbol,
                    "action_id": action["action_id"],
                    "action_type": action["action_type"],
                    "status": "no_position",
                    "cash_change_cny": 0.0,
                    "share_change": 0,
                    "deferred_tax_settled_cny": 0.0,
                }
            )
            continue
        shares_before = position.shares
        cash_change = 0.0
        share_change = 0
        deferred_tax_settled = 0.0
        action_type = str(action["action_type"])
        if action_type == "share_change":
            resulting_lots: list[TaxLotV21] = []
            subtype = str(action["action_subtype"])
            ratio = float(action["post_to_pre_ratio"])
            additive = subtype in {"stock_dividend", "capital_reserve_conversion"}
            if additive and ratio < 1.0:
                raise ExecutionReplayError("additive share-change ratio must be at least one")
            for lot in sorted(position.lots, key=lambda item: (item.acquisition_settlement_session, item.lot_id)):
                pre_shares = lot.shares
                exact_post_shares = pre_shares * ratio
                exact_changed_shares = pre_shares * (ratio - 1.0) if additive else exact_post_shares
                whole_changed_shares = math.floor(exact_changed_shares + 1e-12)
                fractional = exact_changed_shares - whole_changed_shares
                if not additive and whole_changed_shares <= 0:
                    raise ExecutionReplayError(
                        "official share change eliminates an entire held lot without terminal action"
                    )
                per_post_share_basis = lot.cost_basis_cny / exact_post_shares
                removed_basis = fractional * per_post_share_basis
                cash_in_lieu = fractional * float(action["fractional_cash_price"])
                if fractional > 1e-12 and float(action["fractional_cash_price"]) <= 0:
                    raise ExecutionReplayError("fractional share requires official positive cash-in-lieu evidence")
                cash_change += cash_in_lieu
                taxable_dividend = pre_shares * float(action["taxable_dividend_per_pre_action_share"])
                if additive:
                    # Original quantity/date remain; the additional shares are
                    # a distinct newer FIFO lot.  Existing and stock-dividend
                    # entitlements remain attached to the record-date parent.
                    parent_basis = pre_shares * per_post_share_basis
                    resulting_lots.append(
                        TaxLotV21(
                            lot_id=lot.lot_id,
                            acquisition_settlement_session=lot.acquisition_settlement_session,
                            shares=pre_shares,
                            cost_basis_cny=parent_basis,
                            deferred_dividend_gross_cny=lot.deferred_dividend_gross_cny + taxable_dividend,
                        )
                    )
                    if whole_changed_shares > 0:
                        resulting_lots.append(
                            TaxLotV21(
                                lot_id=f"{lot.lot_id}:{action['action_id']}",
                                acquisition_settlement_session=str(action["new_share_registration_session"]),
                                shares=whole_changed_shares,
                                cost_basis_cny=whole_changed_shares * per_post_share_basis,
                                deferred_dividend_gross_cny=0.0,
                            )
                        )
                else:
                    resulting_lots.append(
                        TaxLotV21(
                            lot_id=f"{lot.lot_id}:{action['action_id']}",
                            acquisition_settlement_session=lot.acquisition_settlement_session,
                            shares=whole_changed_shares,
                            cost_basis_cny=lot.cost_basis_cny - removed_basis,
                            deferred_dividend_gross_cny=lot.deferred_dividend_gross_cny,
                        )
                    )
            position.lots = resulting_lots
            share_change = position.shares - shares_before
        elif action_type in {"cash_dividend", "rights_issue"}:
            if str(action["action_id"]) in state.cash_receivables:
                raise ExecutionReplayError("cash entitlement action_id already exists as a receivable")
            gross_per_share = float(
                action[
                    "gross_cash_per_share"
                    if action_type == "cash_dividend"
                    else "official_disposal_proceeds_per_entitled_share"
                ]
            )
            gross_total = 0.0
            for lot in position.lots:
                if lot.acquisition_settlement_session > session:
                    continue
                entitlement = lot.shares * gross_per_share
                if action_type == "cash_dividend":
                    lot.deferred_dividend_gross_cny += entitlement
                gross_total += entitlement
            state.cash_receivables[str(action["action_id"])] = CashReceivableV21(
                action_id=str(action["action_id"]),
                symbol=symbol,
                payment_session=str(action["payment_session"]),
                gross_cash_cny=gross_total,
            )
        elif action_type in TERMINAL_ACTION_TYPES:
            row = open_rows.get(symbol)
            if row is None or row["status"] != "delisted":
                raise ExecutionReplayError(f"terminal action for {symbol} lacks a same-day delisted status")
            source_basis, deferred_tax_settled = _dispose_fifo_lots(
                position, shares_before, str(action["disposal_settlement_session"])
            )
            cash_change -= deferred_tax_settled
            if action_type == "delist_cash":
                cash_change += shares_before * float(action["cash_per_share"])
            elif action_type == "delist_share":
                target_symbol = str(action["target_symbol"])
                target_row = open_rows.get(target_symbol)
                if target_row is None or target_row["status"] in {"delisted", "not_listed"}:
                    raise ExecutionReplayError("delist share target is absent or not an existing security")
                exact_target_shares = shares_before * float(action["target_share_ratio"])
                target_shares = math.floor(exact_target_shares + 1e-12)
                fractional = exact_target_shares - target_shares
                if fractional > 1e-12 and float(action["fractional_cash_price"]) <= 0:
                    raise ExecutionReplayError("delist share fractional consideration lacks official cash evidence")
                cash_change += fractional * float(action["fractional_cash_price"])
                if target_shares > 0:
                    target_position = state.positions.setdefault(target_symbol, PositionV21())
                    target_position.lots.append(
                        TaxLotV21(
                            lot_id=f"delist:{action['action_id']}:{symbol}",
                            acquisition_settlement_session=str(action["disposal_settlement_session"]),
                            shares=target_shares,
                            cost_basis_cny=source_basis,
                        )
                    )
            share_change = -shares_before
            state.positions.pop(symbol)
            state.pending_sells.pop(symbol, None)
            state.last_close.pop(symbol, None)
        else:  # pragma: no cover - normalisation already excludes this branch
            raise ExecutionReplayError(f"unsupported action type {action_type}")
        state.cash_cny += cash_change
        if state.cash_cny < -1e-9:
            raise ExecutionReplayError("corporate action or deferred dividend tax would make cash negative")
        events.append(
            {
                "event_type": "corporate_action",
                "session": session,
                "symbol": symbol,
                "action_id": action["action_id"],
                "action_type": action_type,
                "status": "applied",
                "cash_change_cny": cash_change,
                "share_change": share_change,
                "deferred_tax_settled_cny": deferred_tax_settled,
            }
        )
    return events


def _record_blocked_order(
    events: list[dict[str, Any]],
    *,
    session: str,
    decision_session: str,
    symbol: str,
    side: str,
    requested: int,
    status: str,
) -> None:
    events.append(
        {
            "event_type": "order",
            "session": session,
            "decision_session": decision_session,
            "symbol": symbol,
            "side": side,
            "status": status,
            "requested_shares": requested,
            "filled_shares": 0,
        }
    )


def _sell(
    state: PortfolioStateV21,
    events: list[dict[str, Any]],
    *,
    session: str,
    settlement_session: str,
    symbol: str,
    pending: PendingSellV21,
    row: Mapping[str, Any],
    cost: CostModelV21,
) -> None:
    position = state.positions.get(symbol)
    current = 0 if position is None else position.shares
    requested = current - pending.target_shares
    if requested <= 0:
        state.pending_sells.pop(symbol, None)
        return
    status = _fill_status(row, "sell")
    if status != "fillable":
        _record_blocked_order(
            events,
            session=session,
            decision_session=pending.originating_decision_session,
            symbol=symbol,
            side="sell",
            requested=requested,
            status=status,
        )
        return
    fill_shares = _capacity_fill_shares(requested, side="sell", row=row, cost=cost, session=session)
    if fill_shares <= 0:
        _record_blocked_order(
            events,
            session=session,
            decision_session=pending.originating_decision_session,
            symbol=symbol,
            side="sell",
            requested=requested,
            status="dual_capacity_cap",
        )
        return
    terms = _execution_terms(
        side="sell", open_px=float(row["open"]), shares=fill_shares, row=row, cost=cost, session=session
    )
    assert position is not None
    cost_reduction, deferred_tax = _dispose_fifo_lots(position, fill_shares, settlement_session)
    state.cash_cny += terms["cash_debit_credit_cny"] - deferred_tax
    if state.cash_cny < -1e-9:
        raise ExecutionReplayError("sell settlement and deferred dividend tax would make cash negative")
    if position.shares == 0:
        state.positions.pop(symbol)
        state.last_close.pop(symbol, None)
    if state.positions.get(symbol, PositionV21()).shares <= pending.target_shares:
        state.pending_sells.pop(symbol, None)
    events.append(
        {
            "event_type": "order",
            "session": session,
            "decision_session": pending.originating_decision_session,
            "symbol": symbol,
            "side": "sell",
            "status": "filled" if fill_shares == requested else "partial",
            "requested_shares": requested,
            "filled_shares": fill_shares,
            "fifo_cost_basis_released_cny": cost_reduction,
            "deferred_dividend_tax_settled_cny": deferred_tax,
            "settlement_session": settlement_session,
            **terms,
        }
    )


def _buy(
    state: PortfolioStateV21,
    events: list[dict[str, Any]],
    *,
    session: str,
    settlement_session: str,
    decision_session: str,
    symbol: str,
    requested: int,
    row: Mapping[str, Any],
    cost: CostModelV21,
) -> None:
    if requested <= 0:
        return
    if symbol not in state.positions and len(state.positions) >= 50:
        _record_blocked_order(
            events,
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="buy",
            requested=requested,
            status="maximum_actual_positions",
        )
        return
    status = _fill_status(row, "buy")
    if status != "fillable":
        _record_blocked_order(
            events,
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="buy",
            requested=requested,
            status=status,
        )
        return
    lot = int(row["lot_size"])
    fill_shares = _capacity_fill_shares(requested, side="buy", row=row, cost=cost, session=session)
    if fill_shares <= 0:
        _record_blocked_order(
            events,
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="buy",
            requested=requested,
            status="dual_capacity_cap",
        )
        return
    while fill_shares > 0:
        terms = _execution_terms(
            side="buy", open_px=float(row["open"]), shares=fill_shares, row=row, cost=cost, session=session
        )
        if -terms["cash_debit_credit_cny"] <= state.cash_cny + 1e-9:
            break
        fill_shares -= lot
    if fill_shares <= 0:
        _record_blocked_order(
            events,
            session=session,
            decision_session=decision_session,
            symbol=symbol,
            side="buy",
            requested=requested,
            status="insufficient_cash",
        )
        return
    terms = _execution_terms(
        side="buy", open_px=float(row["open"]), shares=fill_shares, row=row, cost=cost, session=session
    )
    state.cash_cny += terms["cash_debit_credit_cny"]
    position = state.positions.setdefault(symbol, PositionV21())
    lot_id = object_sha256(
        {
            "domain": "xs_chan_v2_1_tax_lot",
            "session": session,
            "decision_session": decision_session,
            "symbol": symbol,
            "ordinal": len(position.lots),
            "shares": fill_shares,
        }
    )
    position.lots.append(
        TaxLotV21(
            lot_id=lot_id,
            acquisition_settlement_session=settlement_session,
            shares=fill_shares,
            cost_basis_cny=-terms["cash_debit_credit_cny"],
        )
    )
    events.append(
        {
            "event_type": "order",
            "session": session,
            "decision_session": decision_session,
            "symbol": symbol,
            "side": "buy",
            "status": "filled" if fill_shares == requested else "partial",
            "requested_shares": requested,
            "filled_shares": fill_shares,
            "settlement_session": settlement_session,
            **terms,
        }
    )


def _mark_eod(
    state: PortfolioStateV21,
    eod_rows: Mapping[str, Mapping[str, Any]],
    *,
    session: str,
    prior_nav: float | None,
    annual_cash_yield: float,
) -> dict[str, Any]:
    daily_cash_rate = (1.0 + annual_cash_yield) ** (1.0 / 242.0) - 1.0
    cash_yield = state.cash_cny * daily_cash_rate
    state.cash_cny += cash_yield
    position_value = 0.0
    holdings: list[dict[str, Any]] = []
    for symbol in sorted(state.positions):
        position = state.positions[symbol]
        row = eod_rows.get(symbol)
        if row is None:
            raise ExecutionReplayError(f"held symbol {symbol} missing from EOD snapshot on {session}")
        if row["status"] == "trading":
            price = float(row["close"])
            state.last_close[symbol] = price
        elif row["status"] == "suspended":
            price = state.last_close.get(symbol, math.nan)
        elif row["status"] == "delisted":
            raise ExecutionReplayError(f"delisted held symbol {symbol} lacks an applied terminal action on {session}")
        else:
            raise ExecutionReplayError(f"held symbol {symbol} has invalid not_listed status on {session}")
        if not math.isfinite(price) or price <= 0:
            raise ExecutionReplayError(f"held symbol {symbol} has no valid EOD mark on {session}")
        value = position.shares * price
        position_value += value
        holdings.append({"symbol": symbol, "shares": position.shares, "close": price, "value_cny": value})
    receivable_value = float(sum(item.gross_cash_cny for item in state.cash_receivables.values()))
    tax_liability = _remeasure_dividend_tax(state, session)
    nav = state.cash_cny + position_value + receivable_value - tax_liability
    if state.cash_cny < -1e-6 or not math.isfinite(nav) or nav <= 0:
        raise ExecutionReplayError(f"portfolio conservation failed on {session}: cash={state.cash_cny}, nav={nav}")
    daily_return = 0.0 if prior_nav is None else nav / prior_nav - 1.0
    return {
        "session": session,
        "nav_cny": nav,
        "daily_return": daily_return,
        "cash_cny": state.cash_cny,
        "cash_yield_cny": cash_yield,
        "position_value_cny": position_value,
        "cash_receivable_value_cny": receivable_value,
        "dividend_tax_liability_cny": tax_liability,
        "gross_exposure": position_value / nav,
        "holding_count": len(state.positions),
        "pending_sell_count": len(state.pending_sells),
        "holdings": holdings,
        "state_sha256": object_sha256(state.as_dict()),
    }


def replay_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one arm/scenario from an empty initial portfolio.

    ``value`` is retained verbatim under ``input`` in the returned artifact so
    that a semantic verifier can independently invoke this function and compare
    the complete canonical output, rather than trusting summary flags.
    """

    exact = {
        "schema",
        "protocol_sha256",
        "data_snapshot_sha256",
        "trial_id",
        "arm_id",
        "scenario_id",
        "initial_capital_cny",
        "annual_cash_yield",
        "terminal_policy",
        "cost_model",
        "sessions",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ExecutionReplayError("execution input must have exact V2.1 top-level keys")
    if value["schema"] != EXECUTION_INPUT_SCHEMA:
        raise ExecutionReplayError(f"execution schema must be {EXECUTION_INPUT_SCHEMA!r}")
    _require_sha256(value["protocol_sha256"], "protocol_sha256")
    _require_sha256(value["data_snapshot_sha256"], "data_snapshot_sha256")
    if not str(value["trial_id"]) or not str(value["arm_id"]) or not str(value["scenario_id"]):
        raise ExecutionReplayError("trial_id, arm_id, and scenario_id must be non-empty")
    annual_cash_yield = _finite_number(value["annual_cash_yield"], "annual_cash_yield", minimum=0.0)
    if annual_cash_yield != 0.0:
        raise ExecutionReplayError("V2.1 freezes cash interest at zero")
    if value["terminal_policy"] != "mark_to_market_no_forced_liquidation":
        raise ExecutionReplayError("V2.1 only permits mark_to_market_no_forced_liquidation")
    cost = CostModelV21.from_mapping(value["cost_model"])
    sessions = value["sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise ExecutionReplayError("sessions must be a non-empty list")

    state = empty_portfolio(value["initial_capital_cny"])
    initial_state = state.as_dict()
    all_events: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    session_evidence: list[dict[str, Any]] = []
    selection_hashes: list[str] = []
    decision_evidence: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    prior_session: str | None = None
    prior_nav: float | None = None
    nav_by_session: dict[str, float] = {}
    for session_index, raw_session in enumerate(sessions):
        expected_keys = {
            "session",
            "settlement_session",
            "open_snapshot",
            "eod_snapshot",
            "corporate_actions",
            "decision",
        }
        if not isinstance(raw_session, Mapping) or set(raw_session) != expected_keys:
            raise ExecutionReplayError(f"sessions[{session_index}] must have exact V2.1 keys")
        session = _session(raw_session["session"], f"sessions[{session_index}].session")
        settlement_session = _session(
            raw_session["settlement_session"], f"sessions[{session_index}].settlement_session"
        )
        if settlement_session < session:
            raise ExecutionReplayError("trade settlement session cannot precede trade session")
        if session in seen_sessions or (prior_session is not None and session <= prior_session):
            raise ExecutionReplayError("sessions must be unique and strictly increasing")
        seen_sessions.add(session)
        prior_session = session
        open_rows = _normalise_open_rows(raw_session["open_snapshot"], session)
        eod_rows = _normalise_eod_rows(raw_session["eod_snapshot"], session)
        if set(open_rows) != set(eod_rows):
            raise ExecutionReplayError(f"opening and EOD symbol domains differ on {session}")
        actions = _normalise_actions(raw_session["corporate_actions"], session)
        decision = _normalise_decision(raw_session["decision"], session)

        state_before_open = state.as_dict()
        event_start = len(all_events)
        all_events.extend(_apply_corporate_actions(state, actions, open_rows, session))

        desired_shares: dict[str, int] = {}
        if decision is not None:
            selection_hashes.append(str(decision["selection_identity_sha256"]))
            decision_evidence.append(
                {
                    "decision_session": decision["decision_session"],
                    "execution_session": session,
                    "selection_identity_sha256": decision["selection_identity_sha256"],
                    "new_entry_symbols": list(decision["new_entry_symbols"]),
                    "gate_eligible_new_entry_opportunities": int(decision["gate_eligible_new_entry_opportunities"]),
                    "decision_artifact_sha256": object_sha256(decision),
                    "portfolio_before_sha256": object_sha256(state_before_open),
                    "pending_sells_before_sha256": object_sha256(state_before_open["pending_sells"]),
                }
            )
            # The sizing NAV is the already-recorded decision-close NAV.  Using
            # the next session's open to resize the book would leak execution
            # information into a supposedly frozen decision.
            decision_nav = nav_by_session.get(str(decision["decision_session"]))
            if decision_nav is None:
                raise ExecutionReplayError("decision sizing NAV has no prior replayed decision-close endpoint")
            tolerance = max(abs(decision_nav), 1.0) * 1e-10
            if abs(decision_nav - float(decision["sizing_nav_cny"])) > tolerance:
                raise ExecutionReplayError("decision sizing_nav_cny differs from replayed decision-close NAV")
            slot_notional = (
                float(decision["sizing_nav_cny"])
                * (1.0 - float(decision["cash_buffer_fraction"]))
                / int(decision["slots"])
            )
            selected = set(decision["ordered_symbols"])
            for symbol in decision["ordered_symbols"]:
                row = open_rows.get(symbol)
                if row is None:
                    raise ExecutionReplayError(f"selected symbol {symbol} missing from opening snapshot on {session}")
                if symbol in state.positions:
                    # V2.1 deliberately preserves drift: retained positions are
                    # neither topped up nor trimmed at weekly rebalances.
                    desired_shares[symbol] = state.positions[symbol].shares
                elif row["status"] == "trading":
                    lot = int(row["lot_size"])
                    desired_shares[symbol] = math.floor(slot_notional / float(row["open"]) / lot) * lot
                else:
                    desired_shares[symbol] = 0
            for symbol in list(state.positions):
                if symbol not in selected:
                    state.pending_sells[symbol] = PendingSellV21(
                        target_shares=0,
                        originating_decision_session=str(decision["decision_session"]),
                    )
                else:
                    state.pending_sells.pop(symbol, None)
            # A previous pending exit stays pending unless the security is once
            # again selected; in that case the latest target supersedes it.
            for symbol in selected:
                if symbol in state.pending_sells and state.positions[symbol].shares <= desired_shares[symbol]:
                    state.pending_sells.pop(symbol, None)

        # Daily retry, including sessions without a new rebalance.  The current
        # session's auction snapshot supplies the independently observed caps.
        for symbol in sorted(
            state.pending_sells,
            key=lambda item: (state.pending_sells[item].originating_decision_session, item),
        ):
            row = open_rows.get(symbol)
            if row is None:
                raise ExecutionReplayError(f"pending sell symbol {symbol} missing from opening snapshot on {session}")
            _sell(
                state,
                all_events,
                session=session,
                settlement_session=settlement_session,
                symbol=symbol,
                pending=state.pending_sells[symbol],
                row=row,
                cost=cost,
            )

        if decision is not None:
            for symbol in decision["ordered_symbols"]:
                requested = desired_shares[symbol] - state.positions.get(symbol, PositionV21()).shares
                _buy(
                    state,
                    all_events,
                    session=session,
                    settlement_session=settlement_session,
                    decision_session=str(decision["decision_session"]),
                    symbol=symbol,
                    requested=requested,
                    row=open_rows[symbol],
                    cost=cost,
                )

        open_state = state.as_dict()
        session_events = all_events[event_start:]
        order_events = [event for event in session_events if event["event_type"] == "order"]
        requested_orders = [
            {
                key: event[key]
                for key in (
                    "decision_session",
                    "symbol",
                    "side",
                    "requested_shares",
                    "status",
                    "filled_shares",
                )
            }
            for event in order_events
        ]
        fills = [
            {
                key: event[key]
                for key in (
                    "symbol",
                    "side",
                    "filled_shares",
                    "open_price",
                    "fill_price",
                    "fill_notional_cny",
                )
            }
            for event in order_events
            if int(event["filled_shares"]) > 0
        ]
        fees = [
            {
                key: event.get(key, 0.0)
                for key in (
                    "symbol",
                    "side",
                    "commission_cny",
                    "transfer_fee_cny",
                    "stamp_duty_cny",
                    "slippage_cost_cny",
                    "deferred_dividend_tax_settled_cny",
                )
            }
            for event in order_events
            if int(event["filled_shares"]) > 0
        ]
        eod = _mark_eod(
            state,
            eod_rows,
            session=session,
            prior_nav=prior_nav,
            annual_cash_yield=annual_cash_yield,
        )
        daily.append(eod)
        session_evidence.append(
            {
                "session": session,
                "settlement_session": settlement_session,
                "open_snapshot_sha256": object_sha256(raw_session["open_snapshot"]),
                "open_prices_sha256": object_sha256(
                    [{"symbol": symbol, "open": open_rows[symbol]["open"]} for symbol in sorted(open_rows)]
                ),
                "limit_state_sha256": object_sha256(
                    [
                        {
                            "symbol": symbol,
                            "status": open_rows[symbol]["status"],
                            "limit_up": open_rows[symbol]["limit_up"],
                            "limit_down": open_rows[symbol]["limit_down"],
                        }
                        for symbol in sorted(open_rows)
                    ]
                ),
                "corporate_actions_sha256": object_sha256(actions),
                "requested_orders_sha256": object_sha256(requested_orders),
                "fills_sha256": object_sha256(fills),
                "fees_sha256": object_sha256(fees),
                "portfolio_before_sha256": object_sha256(state_before_open),
                "portfolio_after_open_sha256": object_sha256(open_state),
                "pending_sells_before_sha256": object_sha256(state_before_open["pending_sells"]),
                "pending_sells_after_open_sha256": object_sha256(open_state["pending_sells"]),
                "pending_sell_count_before": len(state_before_open["pending_sells"]),
                "pending_sell_count_after": len(open_state["pending_sells"]),
                "raw_close_snapshot_sha256": object_sha256(raw_session["eod_snapshot"]),
                "eod_portfolio_state_sha256": eod["state_sha256"],
                "nav_artifact_sha256": object_sha256({key: item for key, item in eod.items() if key != "state_sha256"}),
            }
        )
        prior_nav = float(eod["nav_cny"])
        nav_by_session[session] = prior_nav

    final_state = state.as_dict()
    output = {
        "schema": EXECUTION_RESULT_SCHEMA,
        "protocol_sha256": value["protocol_sha256"],
        "data_snapshot_sha256": value["data_snapshot_sha256"],
        "trial_id": str(value["trial_id"]),
        "arm_id": str(value["arm_id"]),
        "scenario_id": str(value["scenario_id"]),
        "input_sha256": object_sha256(value),
        "initial_state": initial_state,
        "initial_state_sha256": object_sha256(initial_state),
        "selection_identity_sequence": selection_hashes,
        "selection_identity_sequence_sha256": object_sha256(selection_hashes),
        "decision_evidence": decision_evidence,
        "decision_evidence_sha256": object_sha256(decision_evidence),
        "events": all_events,
        "daily": daily,
        "session_evidence": session_evidence,
        "session_evidence_sha256": object_sha256(session_evidence),
        "final_state": final_state,
        "final_state_sha256": object_sha256(final_state),
    }
    output["result_sha256"] = object_sha256(output)
    return output


def verify_execution_result(value: Mapping[str, Any], expected_input: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a result and require canonical byte-for-byte semantic parity."""

    replayed = replay_execution(expected_input)
    if canonical_json_bytes(value) != canonical_json_bytes(replayed):
        raise ExecutionReplayError("execution result differs from deterministic semantic replay")
    return replayed


def replay_cost_scenarios(scenario_inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Replay frozen selections under multiple costs without changing identity.

    Every scenario supplies its own complete input because transaction costs
    change its prior decision-close NAV and therefore its next frozen sizing
    amount.  Only the pre-cost selection/new-entry identities are required to be
    identical.  Every scenario still starts from the registered empty book.
    """

    if not scenario_inputs:
        raise ExecutionReplayError("at least one cost scenario is required")
    results: dict[str, dict[str, Any]] = {}
    expected_identity: str | None = None
    expected_decision_identity: str | None = None
    trial_id: str | None = None
    arm_id: str | None = None
    for scenario_id in sorted(scenario_inputs):
        execution_input = dict(scenario_inputs[scenario_id])
        if execution_input.get("scenario_id") != str(scenario_id):
            raise ExecutionReplayError("scenario key differs from execution input scenario_id")
        result = replay_execution(execution_input)
        identity = str(result["selection_identity_sequence_sha256"])
        decision_identity = object_sha256(
            [
                {
                    "decision_session": row["decision_session"],
                    "selection_identity_sha256": row["selection_identity_sha256"],
                    "new_entry_symbols": row["new_entry_symbols"],
                    "gate_eligible_new_entry_opportunities": row["gate_eligible_new_entry_opportunities"],
                }
                for row in result["decision_evidence"]
            ]
        )
        if expected_identity is None:
            expected_identity = identity
            expected_decision_identity = decision_identity
            trial_id = str(result["trial_id"])
            arm_id = str(result["arm_id"])
        elif identity != expected_identity:
            raise ExecutionReplayError("selection identities differ across cost scenarios")
        elif decision_identity != expected_decision_identity:
            raise ExecutionReplayError("new-entry identities differ across cost scenarios")
        elif str(result["trial_id"]) != trial_id or str(result["arm_id"]) != arm_id:
            raise ExecutionReplayError("trial or arm identity differs across cost scenarios")
        results[str(scenario_id)] = result
    output = {
        "schema": SCENARIO_RESULT_SCHEMA,
        "trial_id": trial_id,
        "arm_id": arm_id,
        "selection_identity_sequence_sha256": expected_identity,
        "decision_identity_sequence_sha256": expected_decision_identity,
        "scenarios": results,
    }
    output["result_sha256"] = object_sha256(output)
    return output


def event_notional_and_costs(result: Mapping[str, Any]) -> dict[str, float]:
    """Summarize fill notional and explicit costs from replayed order events."""

    if result.get("schema") != EXECUTION_RESULT_SCHEMA:
        raise ExecutionReplayError("not a V2.1 execution result")
    filled = [
        event
        for event in result.get("events", [])
        if event.get("event_type") == "order" and int(event.get("filled_shares", 0)) > 0
    ]
    return {
        "fill_notional_cny": float(sum(float(event["fill_notional_cny"]) for event in filled)),
        "commission_cny": float(sum(float(event["commission_cny"]) for event in filled)),
        "transfer_fee_cny": float(sum(float(event["transfer_fee_cny"]) for event in filled)),
        "stamp_duty_cny": float(sum(float(event["stamp_duty_cny"]) for event in filled)),
        "slippage_cost_cny": float(sum(float(event["slippage_cost_cny"]) for event in filled)),
    }


def build_weekly_path_component(result: Mapping[str, Any], cycles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive formal decision-close weekly returns from a replay result.

    The caller supplies the ledger-registered 52 consecutive endpoint pairs.
    No resampling convention or missing-week replacement is inferred here.
    """

    if result.get("schema") != EXECUTION_RESULT_SCHEMA:
        raise ExecutionReplayError("weekly path requires a V2.1 execution result")
    if not isinstance(cycles, list) or len(cycles) != 52:
        raise ExecutionReplayError("weekly path requires exactly 52 registered cycles")
    daily_rows = result.get("daily")
    if not isinstance(daily_rows, list):
        raise ExecutionReplayError("execution result daily rows are missing")
    by_session = {str(row.get("session")): row for row in daily_rows}
    if len(by_session) != len(daily_rows):
        raise ExecutionReplayError("execution result contains duplicate daily sessions")
    decision_rows = {str(row.get("decision_session")): row for row in result.get("decision_evidence", [])}
    events = result.get("events")
    if not isinstance(events, list):
        raise ExecutionReplayError("execution result events are missing")

    week_labels: list[str] = []
    weekly_returns: list[float] = []
    weekly_turnover: list[float] = []
    new_entry_sets: list[list[str]] = []
    opportunities: list[int] = []
    previous_end: str | None = None
    first_start: str | None = None
    final_end: str | None = None
    for index, raw in enumerate(cycles, start=1):
        exact = {"week_index", "start_decision_session", "end_decision_session"}
        if not isinstance(raw, Mapping) or set(raw) != exact or raw["week_index"] != index:
            raise ExecutionReplayError("cycle records must have exact sequential V2.1 fields")
        start = _session(raw["start_decision_session"], "cycle.start_decision_session")
        end = _session(raw["end_decision_session"], "cycle.end_decision_session")
        if start >= end or (previous_end is not None and start != previous_end):
            raise ExecutionReplayError("weekly cycles must form one contiguous registered endpoint chain")
        if start not in by_session or end not in by_session:
            raise ExecutionReplayError("weekly cycle endpoint is absent from replayed EOD valuations")
        decision = decision_rows.get(start)
        if decision is None:
            raise ExecutionReplayError("weekly cycle start lacks its frozen decision evidence")
        start_nav = _finite_number(by_session[start]["nav_cny"], f"{start}.nav_cny", minimum=0.0)
        end_nav = _finite_number(by_session[end]["nav_cny"], f"{end}.nav_cny", minimum=0.0)
        if min(start_nav, end_nav) <= 0:
            raise ExecutionReplayError("weekly NAV endpoint must be positive")
        filled_notional = sum(
            float(event["fill_notional_cny"])
            for event in events
            if event.get("event_type") == "order"
            and int(event.get("filled_shares", 0)) > 0
            and start < str(event.get("session")) <= end
        )
        week_labels.append(end)
        weekly_returns.append(end_nav / start_nav - 1.0)
        weekly_turnover.append(filled_notional / (2.0 * start_nav))
        new_entry_sets.append(list(decision["new_entry_symbols"]))
        opportunities.append(int(decision["gate_eligible_new_entry_opportunities"]))
        previous_end = end
        first_start = start if first_start is None else first_start
        final_end = end

    assert first_start is not None and final_end is not None
    daily_window = [row for row in daily_rows if first_start < str(row["session"]) <= final_end]
    daily_labels = [str(row["session"]) for row in daily_window]
    if daily_labels != sorted(set(daily_labels)):
        raise ExecutionReplayError("daily exposure window is not strictly increasing and unique")
    component = {
        "week_labels": week_labels,
        "daily_labels": daily_labels,
        "path": {
            "weekly_returns": weekly_returns,
            "daily_post_close_gross_exposure": [float(row["gross_exposure"]) for row in daily_window],
            "weekly_one_way_turnover": weekly_turnover,
            "weekly_new_entry_identity_sets": new_entry_sets,
            "weekly_gate_eligible_new_entry_opportunities": opportunities,
        },
    }
    component["component_sha256"] = object_sha256(component)
    return component


__all__ = [
    "CostModelV21",
    "DEFAULT_LOT_SIZE",
    "EXECUTION_INPUT_SCHEMA",
    "EXECUTION_RESULT_SCHEMA",
    "ExecutionReplayError",
    "PORTFOLIO_STATE_SCHEMA",
    "SCENARIO_RESULT_SCHEMA",
    "canonical_json_bytes",
    "build_weekly_path_component",
    "empty_portfolio",
    "event_notional_and_costs",
    "object_sha256",
    "replay_cost_scenarios",
    "replay_execution",
    "selection_identity",
    "verify_execution_result",
]
