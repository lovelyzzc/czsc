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
PENDING_SETTLEMENT_SCHEMA = "xs_chan_pending_settlements_v2_1"

DEFAULT_LOT_SIZE = 100
TRADING_STATUSES = frozenset({"trading"})
NON_TRADING_STATUSES = frozenset({"suspended", "delisted", "not_listed"})
TERMINAL_ACTION_TYPES = frozenset({"delist_cash", "delist_share", "delist_writeoff"})
SINGLE_ARM_IDS = frozenset({"F", "FC", "FMA"})
SEEDED_ARM_IDS = frozenset({"R_match", "FGR", "FMGR"})
FROZEN_RANDOM_SEEDS = frozenset(range(20260720, 20260740))


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
class CashEntitlementLotV21:
    """Record-date lot snapshot for a later cash-entitlement confirmation."""

    parent_lot_id: str
    acquisition_settlement_session: str
    record_shares: int
    record_cost_basis_cny: float
    record_deferred_dividend_gross_cny: float
    gross_cash_cny: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_lot_id": self.parent_lot_id,
            "acquisition_settlement_session": self.acquisition_settlement_session,
            "record_shares": int(self.record_shares),
            "record_cost_basis_cny": float(self.record_cost_basis_cny),
            "record_deferred_dividend_gross_cny": float(self.record_deferred_dividend_gross_cny),
            "gross_cash_cny": float(self.gross_cash_cny),
        }


@dataclass
class CashReceivableV21:
    action_id: str
    symbol: str
    action_type: str
    record_session: str
    effective_session: str
    payment_session: str
    gross_cash_cny: float
    status: str
    lots: list[CashEntitlementLotV21]

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "symbol": self.symbol,
            "action_type": self.action_type,
            "record_session": self.record_session,
            "effective_session": self.effective_session,
            "payment_session": self.payment_session,
            "gross_cash_cny": float(self.gross_cash_cny),
            "status": self.status,
            "lots": [lot.as_dict() for lot in self.lots],
        }


@dataclass
class ShareEntitlementLotV21:
    """Record-date lot snapshot used for a later official share credit."""

    parent_lot_id: str
    acquisition_settlement_session: str
    pre_action_shares: int
    pre_action_cost_basis_cny: float
    pre_action_deferred_dividend_gross_cny: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_lot_id": self.parent_lot_id,
            "acquisition_settlement_session": self.acquisition_settlement_session,
            "pre_action_shares": int(self.pre_action_shares),
            "pre_action_cost_basis_cny": float(self.pre_action_cost_basis_cny),
            "pre_action_deferred_dividend_gross_cny": float(self.pre_action_deferred_dividend_gross_cny),
        }


@dataclass
class ShareEntitlementV21:
    """A record-date entitlement waiting for its official registration date."""

    action_id: str
    symbol: str
    record_session: str
    registration_session: str
    action_subtype: str
    post_to_pre_ratio: float
    fractional_cash_price: float
    taxable_dividend_per_pre_action_share: float
    source_row_sha256: str
    lots: list[ShareEntitlementLotV21]

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "symbol": self.symbol,
            "record_session": self.record_session,
            "registration_session": self.registration_session,
            "action_subtype": self.action_subtype,
            "post_to_pre_ratio": float(self.post_to_pre_ratio),
            "fractional_cash_price": float(self.fractional_cash_price),
            "taxable_dividend_per_pre_action_share": float(self.taxable_dividend_per_pre_action_share),
            "source_row_sha256": self.source_row_sha256,
            "lots": [lot.as_dict() for lot in self.lots],
        }


@dataclass
class TerminalConsiderationV21:
    """Official terminal consideration waiting for its disposal settlement date."""

    action_id: str
    source_symbol: str
    settlement_session: str
    cash_cny: float
    target_symbol: str | None
    target_shares: int
    target_cost_basis_cny: float
    deferred_dividend_tax_cny: float
    pending_sell_originating_decision_session: str | None

    def asset_dict(self) -> dict[str, Any]:
        return {
            "asset_type": "terminal_consideration",
            "action_id": self.action_id,
            "source_symbol": self.source_symbol,
            "settlement_session": self.settlement_session,
            "cash_cny": float(self.cash_cny),
            "target_symbol": self.target_symbol,
            "target_shares": int(self.target_shares),
            "target_cost_basis_cny": float(self.target_cost_basis_cny),
            "pending_sell_originating_decision_session": self.pending_sell_originating_decision_session,
        }

    def liability_dict(self) -> dict[str, Any] | None:
        if self.deferred_dividend_tax_cny <= 0:
            return None
        return {
            "liability_type": "terminal_deferred_dividend_tax",
            "action_id": self.action_id,
            "settlement_session": self.settlement_session,
            "amount_cny": float(self.deferred_dividend_tax_cny),
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
    share_entitlements: dict[str, ShareEntitlementV21] = field(default_factory=dict)
    terminal_considerations: dict[str, TerminalConsiderationV21] = field(default_factory=dict)
    processed_action_ids: set[str] = field(default_factory=set)
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
            "share_entitlements": {
                action_id: self.share_entitlements[action_id].as_dict() for action_id in sorted(self.share_entitlements)
            },
            "processed_action_ids": sorted(self.processed_action_ids),
            "dividend_tax_liabilities": {
                lot_id: float(self.dividend_tax_liabilities[lot_id]) for lot_id in sorted(self.dividend_tax_liabilities)
            },
            "other_assets": [
                self.terminal_considerations[action_id].asset_dict()
                for action_id in sorted(self.terminal_considerations)
            ],
            "other_liabilities": [
                liability
                for action_id in sorted(self.terminal_considerations)
                if (liability := self.terminal_considerations[action_id].liability_dict()) is not None
            ],
            "borrowed_cash_cny": 0.0,
        }


def pending_settlements_summary(state: PortfolioStateV21) -> dict[str, Any]:
    """Return the complete canonical pending-settlement state.

    Ordinary cash and positions are excluded.  The summary binds every
    unsettled asset, share entitlement, terminal consideration, and lot-level
    dividend-tax obligation that can survive after positions reach zero.
    """

    if not isinstance(state, PortfolioStateV21):
        raise ExecutionReplayError("pending settlement summary requires a V2.1 portfolio state")
    cash_receivables = {
        action_id: state.cash_receivables[action_id].as_dict() for action_id in sorted(state.cash_receivables)
    }
    share_entitlements = {
        action_id: state.share_entitlements[action_id].as_dict() for action_id in sorted(state.share_entitlements)
    }
    terminal_considerations = {
        action_id: {
            "action_id": consideration.action_id,
            "source_symbol": consideration.source_symbol,
            "settlement_session": consideration.settlement_session,
            "cash_cny": float(consideration.cash_cny),
            "target_symbol": consideration.target_symbol,
            "target_shares": int(consideration.target_shares),
            "target_cost_basis_cny": float(consideration.target_cost_basis_cny),
            "deferred_dividend_tax_cny": float(consideration.deferred_dividend_tax_cny),
            "pending_sell_originating_decision_session": (consideration.pending_sell_originating_decision_session),
        }
        for action_id, consideration in sorted(state.terminal_considerations.items())
    }
    dividend_tax_liabilities = {
        lot_id: float(state.dividend_tax_liabilities[lot_id]) for lot_id in sorted(state.dividend_tax_liabilities)
    }
    return {
        "schema": PENDING_SETTLEMENT_SCHEMA,
        "cash_receivables": cash_receivables,
        "share_entitlements": share_entitlements,
        "terminal_considerations": terminal_considerations,
        "dividend_tax_liabilities": dividend_tax_liabilities,
        "pending_settlement_count": (
            len(cash_receivables)
            + len(share_entitlements)
            + len(terminal_considerations)
            + len(dividend_tax_liabilities)
        ),
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
        if lot_size != DEFAULT_LOT_SIZE:
            raise ExecutionReplayError(f"V2.1 freezes board-lot size at {DEFAULT_LOT_SIZE} shares")
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
        or new_entries != [symbol for symbol in symbols if symbol in set(new_entries)]
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
        action_type = str(raw["action_type"])
        effective_session = _session(raw["effective_session"], "corporate_action.effective_session")
        expected = set(common)
        if action_type == "share_change":
            expected |= {
                "record_session",
                "post_to_pre_ratio",
                "fractional_cash_price",
                "action_subtype",
                "taxable_dividend_per_pre_action_share",
                "new_share_registration_session",
            }
        elif action_type == "cash_dividend":
            expected |= {"record_session", "gross_cash_per_share", "payment_session"}
        elif action_type == "rights_issue":
            expected |= {"record_session", "official_disposal_proceeds_per_entitled_share", "payment_session"}
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
        item["effective_session"] = effective_session
        if action_type in {"share_change", "cash_dividend", "rights_issue"}:
            record_session = _session(raw["record_session"], "corporate_action.record_session")
            if record_session != session:
                raise ExecutionReplayError("entitlement action record_session must match its enclosing session")
            if effective_session < record_session:
                raise ExecutionReplayError("corporate action effective_session cannot precede its record_session")
            item["record_session"] = record_session
        elif effective_session != session:
            raise ExecutionReplayError("terminal action effective_session must match its enclosing session")
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
            if registration_session != effective_session:
                raise ExecutionReplayError(
                    "share-change registration session must equal its official effective session"
                )
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
            if payment_session < effective_session:
                raise ExecutionReplayError("cash dividend payment_session cannot precede its effective session")
            item.update(gross_cash_per_share=gross, payment_session=payment_session)
        elif action_type == "rights_issue":
            proceeds = _finite_number(
                raw["official_disposal_proceeds_per_entitled_share"],
                "corporate_action.official_disposal_proceeds_per_entitled_share",
                minimum=0.0,
            )
            payment_session = _session(raw["payment_session"], "corporate_action.payment_session")
            if payment_session < effective_session:
                raise ExecutionReplayError("rights proceeds payment cannot precede the effective session")
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


def _confirm_due_receivables(state: PortfolioStateV21, session: str) -> list[dict[str, Any]]:
    """Turn frozen record-date cash entitlements into assets on the official effective date."""

    overdue = [
        receivable.action_id
        for receivable in state.cash_receivables.values()
        if receivable.status == "frozen_entitlement" and receivable.effective_session < session
    ]
    if overdue:
        raise ExecutionReplayError(f"cash entitlement missed its official effective session: {sorted(overdue)}")
    events: list[dict[str, Any]] = []
    for action_id in sorted(state.cash_receivables):
        receivable = state.cash_receivables[action_id]
        if receivable.status != "frozen_entitlement" or receivable.effective_session != session:
            continue
        if receivable.action_type == "cash_dividend" and receivable.lots:
            position = state.positions.get(receivable.symbol)
            if position is None:
                raise ExecutionReplayError(
                    "record-date cash-dividend lot disappeared before the official effective session"
                )
            current_by_id = {lot.lot_id: lot for lot in position.lots}
            for recorded in receivable.lots:
                current = current_by_id.get(recorded.parent_lot_id)
                basis_tolerance = max(abs(recorded.record_cost_basis_cny), 1.0) * 1e-10
                dividend_tolerance = max(abs(recorded.record_deferred_dividend_gross_cny), 1.0) * 1e-10
                if (
                    current is None
                    or current.shares != recorded.record_shares
                    or abs(current.cost_basis_cny - recorded.record_cost_basis_cny) > basis_tolerance
                    or current.deferred_dividend_gross_cny + dividend_tolerance
                    < recorded.record_deferred_dividend_gross_cny
                ):
                    raise ExecutionReplayError(
                        "record-date cash-dividend lot changed before effective session; "
                        "formal replay cannot infer intervening dividend-tax settlement"
                    )
                current.deferred_dividend_gross_cny += recorded.gross_cash_cny
        receivable.status = "receivable"
        events.append(
            {
                "event_type": "cash_receivable_confirmation",
                "session": session,
                "symbol": receivable.symbol,
                "action_id": action_id,
                "action_type": receivable.action_type,
                "gross_cash_cny": receivable.gross_cash_cny,
            }
        )
    return events


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
        if receivable.status != "receivable":
            raise ExecutionReplayError("cash entitlement reached payment before official receivable confirmation")
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


def _settle_due_share_entitlements(state: PortfolioStateV21, session: str) -> list[dict[str, Any]]:
    """Apply record-date share entitlements only on the official registration session."""

    overdue = [
        action_id
        for action_id, entitlement in state.share_entitlements.items()
        if entitlement.registration_session < session
    ]
    if overdue:
        raise ExecutionReplayError(f"share entitlement missed its official registration session: {sorted(overdue)}")
    events: list[dict[str, Any]] = []
    for action_id in sorted(state.share_entitlements):
        entitlement = state.share_entitlements[action_id]
        if entitlement.registration_session != session:
            continue
        position = state.positions.get(entitlement.symbol)
        if position is None:
            raise ExecutionReplayError("record-date share entitlement lost its source position before registration")
        current_by_id = {lot.lot_id: lot for lot in position.lots}
        recorded_ids = {lot.parent_lot_id for lot in entitlement.lots}
        resulting_lots = [lot for lot in position.lots if lot.lot_id not in recorded_ids]
        share_change = 0
        cash_change = 0.0
        additive = entitlement.action_subtype in {"stock_dividend", "capital_reserve_conversion"}
        ratio = entitlement.post_to_pre_ratio
        if additive and ratio < 1.0:
            raise ExecutionReplayError("additive share-change ratio must be at least one")
        for recorded in entitlement.lots:
            current = current_by_id.get(recorded.parent_lot_id)
            tolerance = max(abs(recorded.pre_action_cost_basis_cny), 1.0) * 1e-10
            if (
                current is None
                or current.shares != recorded.pre_action_shares
                or abs(current.cost_basis_cny - recorded.pre_action_cost_basis_cny) > tolerance
                or abs(current.deferred_dividend_gross_cny - recorded.pre_action_deferred_dividend_gross_cny)
                > tolerance
            ):
                raise ExecutionReplayError(
                    "record-date share lot changed before registration; formal replay cannot infer entitlement netting"
                )
            pre_shares = recorded.pre_action_shares
            exact_post_shares = pre_shares * ratio
            exact_changed_shares = pre_shares * (ratio - 1.0) if additive else exact_post_shares
            whole_changed_shares = math.floor(exact_changed_shares + 1e-12)
            fractional = exact_changed_shares - whole_changed_shares
            if not additive and whole_changed_shares <= 0:
                raise ExecutionReplayError(
                    "official share change eliminates an entire held lot without terminal action"
                )
            per_post_share_basis = recorded.pre_action_cost_basis_cny / exact_post_shares
            removed_basis = fractional * per_post_share_basis
            if fractional > 1e-12 and entitlement.fractional_cash_price <= 0:
                raise ExecutionReplayError("fractional share requires official positive cash-in-lieu evidence")
            cash_change += fractional * entitlement.fractional_cash_price
            taxable_dividend = pre_shares * entitlement.taxable_dividend_per_pre_action_share
            if additive:
                resulting_lots.append(
                    TaxLotV21(
                        lot_id=recorded.parent_lot_id,
                        acquisition_settlement_session=recorded.acquisition_settlement_session,
                        shares=pre_shares,
                        cost_basis_cny=pre_shares * per_post_share_basis,
                        deferred_dividend_gross_cny=(
                            recorded.pre_action_deferred_dividend_gross_cny + taxable_dividend
                        ),
                    )
                )
                if whole_changed_shares > 0:
                    resulting_lots.append(
                        TaxLotV21(
                            lot_id=f"{recorded.parent_lot_id}:{action_id}",
                            acquisition_settlement_session=entitlement.registration_session,
                            shares=whole_changed_shares,
                            cost_basis_cny=whole_changed_shares * per_post_share_basis,
                            deferred_dividend_gross_cny=0.0,
                        )
                    )
                share_change += whole_changed_shares
            else:
                resulting_lots.append(
                    TaxLotV21(
                        lot_id=f"{recorded.parent_lot_id}:{action_id}",
                        acquisition_settlement_session=recorded.acquisition_settlement_session,
                        shares=whole_changed_shares,
                        cost_basis_cny=recorded.pre_action_cost_basis_cny - removed_basis,
                        deferred_dividend_gross_cny=recorded.pre_action_deferred_dividend_gross_cny,
                    )
                )
                share_change += whole_changed_shares - pre_shares
        position.lots = resulting_lots
        pending = state.pending_sells.get(entitlement.symbol)
        if pending is not None and pending.target_shares != 0:
            pending.target_shares = math.floor(pending.target_shares * ratio + 1e-12)
        state.cash_cny += cash_change
        if state.cash_cny < -1e-9:
            raise ExecutionReplayError("share-change cash-in-lieu would make cash negative")
        state.share_entitlements.pop(action_id)
        events.append(
            {
                "event_type": "corporate_action",
                "session": session,
                "symbol": entitlement.symbol,
                "action_id": action_id,
                "action_type": "share_change",
                "status": "registered",
                "cash_change_cny": cash_change,
                "share_change": share_change,
                "deferred_tax_settled_cny": 0.0,
            }
        )
    return events


def _settle_due_terminal_considerations(
    state: PortfolioStateV21,
    open_rows: Mapping[str, Mapping[str, Any]],
    session: str,
) -> list[dict[str, Any]]:
    """Settle official cash/share terminal consideration on its registered date."""

    overdue = [
        action_id
        for action_id, consideration in state.terminal_considerations.items()
        if consideration.settlement_session < session
    ]
    if overdue:
        raise ExecutionReplayError(f"terminal consideration missed its official settlement session: {sorted(overdue)}")
    events: list[dict[str, Any]] = []
    for action_id in sorted(state.terminal_considerations):
        consideration = state.terminal_considerations[action_id]
        if consideration.settlement_session != session:
            continue
        if consideration.target_symbol is not None and consideration.target_shares > 0:
            row = open_rows.get(consideration.target_symbol)
            if row is None or row["status"] in {"delisted", "not_listed"}:
                raise ExecutionReplayError("terminal share consideration target is unavailable on settlement")
            target_position = state.positions.setdefault(consideration.target_symbol, PositionV21())
            target_shares_before_receipt = target_position.shares
            target_position.lots.append(
                TaxLotV21(
                    lot_id=f"delist:{action_id}:{consideration.source_symbol}",
                    acquisition_settlement_session=consideration.settlement_session,
                    shares=consideration.target_shares,
                    cost_basis_cny=consideration.target_cost_basis_cny,
                )
            )
            if consideration.pending_sell_originating_decision_session is not None:
                existing = state.pending_sells.get(consideration.target_symbol)
                originating_session = consideration.pending_sell_originating_decision_session
                if existing is not None:
                    originating_session = min(originating_session, existing.originating_decision_session)
                state.pending_sells[consideration.target_symbol] = PendingSellV21(
                    target_shares=(target_shares_before_receipt if existing is None else existing.target_shares),
                    originating_decision_session=originating_session,
                )
        cash_change = consideration.cash_cny - consideration.deferred_dividend_tax_cny
        state.cash_cny += cash_change
        if state.cash_cny < -1e-9:
            raise ExecutionReplayError("terminal settlement and deferred dividend tax would make cash negative")
        state.terminal_considerations.pop(action_id)
        events.append(
            {
                "event_type": "terminal_consideration_settlement",
                "session": session,
                "symbol": consideration.source_symbol,
                "action_id": action_id,
                "cash_change_cny": cash_change,
                "target_symbol": consideration.target_symbol,
                "target_shares": consideration.target_shares,
                "deferred_tax_settled_cny": consideration.deferred_dividend_tax_cny,
            }
        )
    return events


def _apply_corporate_actions(
    state: PortfolioStateV21,
    actions: Sequence[Mapping[str, Any]],
    open_rows: Mapping[str, Mapping[str, Any]],
    session: str,
) -> list[dict[str, Any]]:
    events = _confirm_due_receivables(state, session)
    events.extend(_settle_due_receivables(state, session))
    events.extend(_settle_due_share_entitlements(state, session))
    events.extend(_settle_due_terminal_considerations(state, open_rows, session))
    for action in actions:
        symbol = str(action["symbol"])
        action_id = str(action["action_id"])
        if action_id in state.processed_action_ids:
            raise ExecutionReplayError(f"corporate action_id was already processed: {action_id}")
        state.processed_action_ids.add(action_id)
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
            if action_id in state.share_entitlements:
                raise ExecutionReplayError("share entitlement action_id is already pending")
            eligible_lots = [
                lot
                for lot in sorted(position.lots, key=lambda item: (item.acquisition_settlement_session, item.lot_id))
                if lot.acquisition_settlement_session <= session
            ]
            state.share_entitlements[action_id] = ShareEntitlementV21(
                action_id=action_id,
                symbol=symbol,
                record_session=str(action["record_session"]),
                registration_session=str(action["new_share_registration_session"]),
                action_subtype=str(action["action_subtype"]),
                post_to_pre_ratio=float(action["post_to_pre_ratio"]),
                fractional_cash_price=float(action["fractional_cash_price"]),
                taxable_dividend_per_pre_action_share=float(action["taxable_dividend_per_pre_action_share"]),
                source_row_sha256=str(action["source_row_sha256"]),
                lots=[
                    ShareEntitlementLotV21(
                        parent_lot_id=lot.lot_id,
                        acquisition_settlement_session=lot.acquisition_settlement_session,
                        pre_action_shares=lot.shares,
                        pre_action_cost_basis_cny=lot.cost_basis_cny,
                        pre_action_deferred_dividend_gross_cny=lot.deferred_dividend_gross_cny,
                    )
                    for lot in eligible_lots
                ],
            )
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
            eligible_lots = [
                lot
                for lot in sorted(position.lots, key=lambda item: (item.acquisition_settlement_session, item.lot_id))
                if lot.acquisition_settlement_session <= session
            ]
            entitlement_lots = [
                CashEntitlementLotV21(
                    parent_lot_id=lot.lot_id,
                    acquisition_settlement_session=lot.acquisition_settlement_session,
                    record_shares=lot.shares,
                    record_cost_basis_cny=lot.cost_basis_cny,
                    record_deferred_dividend_gross_cny=lot.deferred_dividend_gross_cny,
                    gross_cash_cny=lot.shares * gross_per_share,
                )
                for lot in eligible_lots
            ]
            gross_total = float(sum(lot.gross_cash_cny for lot in entitlement_lots))
            state.cash_receivables[str(action["action_id"])] = CashReceivableV21(
                action_id=action_id,
                symbol=symbol,
                action_type=action_type,
                record_session=str(action["record_session"]),
                effective_session=str(action["effective_session"]),
                payment_session=str(action["payment_session"]),
                gross_cash_cny=gross_total,
                status="frozen_entitlement",
                lots=entitlement_lots,
            )
        elif action_type in TERMINAL_ACTION_TYPES:
            row = open_rows.get(symbol)
            if row is None or row["status"] != "delisted":
                raise ExecutionReplayError(f"terminal action for {symbol} lacks a same-day delisted status")
            pending_sell = state.pending_sells.get(symbol)
            pending_sell_originating_session = (
                None if pending_sell is None else pending_sell.originating_decision_session
            )
            source_basis, deferred_tax_settled = _dispose_fifo_lots(
                position, shares_before, str(action["disposal_settlement_session"])
            )
            consideration_cash = 0.0
            target_symbol: str | None = None
            target_shares = 0
            target_basis = 0.0
            if action_type == "delist_cash":
                consideration_cash = shares_before * float(action["cash_per_share"])
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
                consideration_cash = fractional * float(action["fractional_cash_price"])
                target_basis = source_basis
            state.terminal_considerations[action_id] = TerminalConsiderationV21(
                action_id=action_id,
                source_symbol=symbol,
                settlement_session=str(action["disposal_settlement_session"]),
                cash_cny=consideration_cash,
                target_symbol=target_symbol,
                target_shares=target_shares,
                target_cost_basis_cny=target_basis,
                deferred_dividend_tax_cny=deferred_tax_settled,
                pending_sell_originating_decision_session=(
                    pending_sell_originating_session if action_type == "delist_share" else None
                ),
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
                "action_id": action_id,
                "action_type": action_type,
                "status": (
                    "entitlement_recorded"
                    if action_type in {"share_change", "cash_dividend", "rights_issue"}
                    else "consideration_recorded"
                    if action_type in TERMINAL_ACTION_TYPES
                    else "applied"
                ),
                "cash_change_cny": cash_change,
                "share_change": share_change,
                "deferred_tax_settled_cny": 0.0 if action_type in TERMINAL_ACTION_TYPES else deferred_tax_settled,
            }
        )
    events.extend(_confirm_due_receivables(state, session))
    events.extend(_settle_due_receivables(state, session))
    events.extend(_settle_due_share_entitlements(state, session))
    events.extend(_settle_due_terminal_considerations(state, open_rows, session))
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
    row: Mapping[str, Any],
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
            "fill_notional_cny": 0.0,
            "commission_cny": 0.0,
            "transfer_fee_cny": 0.0,
            "stamp_duty_cny": 0.0,
            "slippage_cost_cny": 0.0,
            "adv20_cny_asof_decision": float(row["adv20_cny_asof_decision"]),
            "open_auction_turnover_cny": float(row["open_auction_turnover_cny"]),
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
            row=row,
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
            row=row,
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
            "adv20_cny_asof_decision": float(row["adv20_cny_asof_decision"]),
            "open_auction_turnover_cny": float(row["open_auction_turnover_cny"]),
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
            row=row,
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
            row=row,
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
            row=row,
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
            row=row,
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
            "adv20_cny_asof_decision": float(row["adv20_cny_asof_decision"]),
            "open_auction_turnover_cny": float(row["open_auction_turnover_cny"]),
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
    receivable_value = float(
        sum(item.gross_cash_cny for item in state.cash_receivables.values() if item.status == "receivable")
    )
    terminal_asset_value = 0.0
    terminal_liability_value = 0.0
    for consideration in state.terminal_considerations.values():
        terminal_asset_value += consideration.cash_cny
        terminal_liability_value += consideration.deferred_dividend_tax_cny
        if consideration.target_symbol is not None and consideration.target_shares > 0:
            row = eod_rows.get(consideration.target_symbol)
            if row is None:
                raise ExecutionReplayError(
                    f"terminal consideration target {consideration.target_symbol} is missing from EOD snapshot on {session}"
                )
            if row["status"] == "trading":
                target_price = float(row["close"])
                state.last_close[consideration.target_symbol] = target_price
            else:
                target_price = state.last_close.get(consideration.target_symbol, math.nan)
            if not math.isfinite(target_price) or target_price <= 0:
                raise ExecutionReplayError("terminal share consideration has no valid official target mark")
            terminal_asset_value += consideration.target_shares * target_price
    tax_liability = _remeasure_dividend_tax(state, session)
    pending_settlements = pending_settlements_summary(state)
    pending_settlements_root = object_sha256(pending_settlements)
    nav = (
        state.cash_cny
        + position_value
        + receivable_value
        + terminal_asset_value
        - tax_liability
        - terminal_liability_value
    )
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
        "other_asset_value_cny": terminal_asset_value,
        "other_liability_value_cny": terminal_liability_value,
        "dividend_tax_liability_cny": tax_liability,
        "gross_exposure": position_value / nav,
        "holding_count": len(state.positions),
        "pending_sell_count": len(state.pending_sells),
        "pending_sell_symbols": sorted(state.pending_sells),
        "pending_settlements_root_sha256": pending_settlements_root,
        "pending_settlement_count": int(pending_settlements["pending_settlement_count"]),
        "contingent_slot_symbols": sorted(
            {
                consideration.target_symbol
                for consideration in state.terminal_considerations.values()
                if consideration.target_symbol is not None and consideration.target_shares > 0
            }
        ),
        "holdings": holdings,
        "state_sha256": object_sha256(state.as_dict()),
    }


def _pretrade_nav(state: PortfolioStateV21, open_rows: Mapping[str, Mapping[str, Any]], session: str) -> float:
    """Value the post-action/pre-order book at official raw opening prices."""

    position_value = 0.0
    for symbol, position in state.positions.items():
        row = open_rows.get(symbol)
        if row is None:
            raise ExecutionReplayError(f"pretrade position {symbol} is missing from opening snapshot on {session}")
        if row["status"] == "trading":
            price = float(row["open"])
        else:
            price = state.last_close.get(symbol)
            if price is None or price <= 0:
                raise ExecutionReplayError(f"pretrade position {symbol} lacks a last official close on {session}")
        position_value += position.shares * price
    receivables = sum(item.gross_cash_cny for item in state.cash_receivables.values() if item.status == "receivable")
    terminal_assets = 0.0
    terminal_liabilities = 0.0
    for consideration in state.terminal_considerations.values():
        terminal_assets += consideration.cash_cny
        terminal_liabilities += consideration.deferred_dividend_tax_cny
        if consideration.target_symbol is not None and consideration.target_shares > 0:
            row = open_rows.get(consideration.target_symbol)
            if row is None:
                raise ExecutionReplayError(
                    f"terminal consideration target {consideration.target_symbol} is missing from opening snapshot on {session}"
                )
            if row["status"] == "trading":
                target_price = float(row["open"])
            else:
                target_price = state.last_close.get(consideration.target_symbol, float(row["pre_close"]))
            if not math.isfinite(target_price) or target_price <= 0:
                raise ExecutionReplayError("terminal share consideration has no valid official opening mark")
            terminal_assets += consideration.target_shares * target_price
    liabilities = sum(state.dividend_tax_liabilities.values()) + terminal_liabilities
    nav = state.cash_cny + position_value + receivables + terminal_assets - liabilities
    if not math.isfinite(nav) or nav <= 0:
        raise ExecutionReplayError(f"pretrade NAV is invalid on {session}")
    return float(nav)


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
        "seed_id",
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
    arm_id = str(value["arm_id"])
    seed_id = value["seed_id"]
    if arm_id in SINGLE_ARM_IDS:
        if seed_id is not None:
            raise ExecutionReplayError(f"single arm {arm_id} must have seed_id=null")
    elif arm_id in SEEDED_ARM_IDS:
        if isinstance(seed_id, bool) or not isinstance(seed_id, int) or seed_id not in FROZEN_RANDOM_SEEDS:
            raise ExecutionReplayError(f"seeded arm {arm_id} must use one frozen seed 20260720..20260739")
    else:
        raise ExecutionReplayError(f"unknown V2.1 arm_id {arm_id!r}")
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
        opening_actions = [action for action in actions if action["action_type"] in TERMINAL_ACTION_TYPES]
        record_close_actions = [action for action in actions if action["action_type"] not in TERMINAL_ACTION_TYPES]

        state_before_open = state.as_dict()
        event_start = len(all_events)
        # Effective-date confirmations, payments, registrations, terminal
        # settlements and terminal actions happen before the opening auction.
        # Entitlement actions are keyed by official record date, however, and
        # therefore freeze the settled EOD lot book only after this session's
        # fills have been netted.
        all_events.extend(_apply_corporate_actions(state, opening_actions, open_rows, session))
        pretrade_nav = _pretrade_nav(state, open_rows, session)

        desired_shares: dict[str, int] = {}
        if decision is not None:
            selected = set(decision["ordered_symbols"])
            # ``new_entry_symbols`` is frozen from the protocol's 10m/1x
            # reference book and must remain identical across gross/1x/2x and
            # capacity replays.  Local holdings may diverge after different
            # fills and costs, so non-reference scenarios must not rederive or
            # relabel that registered intervention identity.
            blocked_before = {symbol for symbol in state_before_open["pending_sells"] if symbol not in selected}
            blocked_before.update(
                str(asset["target_symbol"])
                for asset in state_before_open["other_assets"]
                if asset.get("asset_type") == "terminal_consideration"
                and asset.get("target_symbol") is not None
                and str(asset["target_symbol"]) not in selected
            )
            if len(decision["ordered_symbols"]) + len(blocked_before) > int(decision["slots"]):
                raise ExecutionReplayError(
                    "blocked exits and selected targets exceed the frozen 50 actual-position slots"
                )
            for consideration in state.terminal_considerations.values():
                if consideration.target_symbol is None or consideration.target_shares <= 0:
                    continue
                if consideration.target_symbol in selected:
                    consideration.pending_sell_originating_decision_session = None
                elif consideration.pending_sell_originating_decision_session is None:
                    consideration.pending_sell_originating_decision_session = str(decision["decision_session"])
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
                    "actual_holdings_before": sorted(state_before_open["positions"]),
                    "actual_shares_before": {
                        symbol: int(state_before_open["positions"][symbol]["shares"])
                        for symbol in sorted(state_before_open["positions"])
                    },
                    "pending_sell_symbols_before": sorted(state_before_open["pending_sells"]),
                    "pretrade_nav_cny": pretrade_nav,
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
                    # Never reset an older blocked exit's origin.  The frozen
                    # order priority is oldest pending full exit first.
                    state.pending_sells.setdefault(
                        symbol,
                        PendingSellV21(
                            target_shares=0,
                            originating_decision_session=str(decision["decision_session"]),
                        ),
                    )
                else:
                    state.pending_sells.pop(symbol, None)
            # A previous pending exit stays pending unless the security is once
            # again selected; in that case the latest target supersedes it.
            for symbol in selected:
                if symbol in state.pending_sells and state.positions[symbol].shares <= desired_shares[symbol]:
                    state.pending_sells.pop(symbol, None)
            # Ledger decisions happen after the prior close and before the
            # execution session opens.  Build that causal state from the
            # prior-close snapshot; same-session corporate actions and fills
            # belong to the later open record.
            post_decision_state = dict(state_before_open)
            post_decision_pending = {symbol: dict(item) for symbol, item in state_before_open["pending_sells"].items()}
            for symbol in state_before_open["positions"]:
                if symbol not in selected:
                    post_decision_pending.setdefault(
                        symbol,
                        {
                            "target_shares": 0,
                            "originating_decision_session": str(decision["decision_session"]),
                        },
                    )
            for symbol in selected:
                post_decision_pending.pop(symbol, None)
            post_decision_state["pending_sells"] = post_decision_pending
            post_decision_assets: list[dict[str, Any]] = []
            for raw_asset in state_before_open["other_assets"]:
                asset = dict(raw_asset)
                target_symbol = asset.get("target_symbol")
                if asset.get("asset_type") == "terminal_consideration" and target_symbol is not None:
                    if str(target_symbol) in selected:
                        asset["pending_sell_originating_decision_session"] = None
                    elif asset.get("pending_sell_originating_decision_session") is None:
                        asset["pending_sell_originating_decision_session"] = str(decision["decision_session"])
                post_decision_assets.append(asset)
            post_decision_state["other_assets"] = post_decision_assets
            decision_evidence[-1].update(
                {
                    "portfolio_after_decision_sha256": object_sha256(post_decision_state),
                    "pending_sells_after_decision_sha256": object_sha256(post_decision_state["pending_sells"]),
                    "pending_sell_symbols_after_decision": sorted(post_decision_state["pending_sells"]),
                }
            )

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
        all_events.extend(_apply_corporate_actions(state, record_close_actions, open_rows, session))
        session_events = all_events[event_start:]
        order_events = [event for event in session_events if event["event_type"] == "order"]
        terminal_share_receipts = [
            {
                "action_id": str(event["action_id"]),
                "source_symbol": str(event["symbol"]),
                "target_symbol": str(event["target_symbol"]),
                "target_shares": int(event["target_shares"]),
            }
            for event in session_events
            if event.get("event_type") == "terminal_consideration_settlement"
            and event.get("target_symbol") is not None
            and int(event.get("target_shares", 0)) > 0
        ]
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
        eod_pending_sells = state.as_dict()["pending_sells"]
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
                "terminal_share_receipts_sha256": object_sha256(terminal_share_receipts),
                "terminal_share_receipt_count": len(terminal_share_receipts),
                "requested_orders_sha256": object_sha256(requested_orders),
                "fills_sha256": object_sha256(fills),
                "fees_sha256": object_sha256(fees),
                "portfolio_before_sha256": object_sha256(state_before_open),
                "portfolio_after_open_sha256": object_sha256(open_state),
                "pending_sells_before_sha256": object_sha256(state_before_open["pending_sells"]),
                "pending_sells_after_open_sha256": object_sha256(open_state["pending_sells"]),
                "pending_sell_count_before": len(state_before_open["pending_sells"]),
                "pending_sell_count_after": len(open_state["pending_sells"]),
                "eod_pending_sells_sha256": object_sha256(eod_pending_sells),
                "eod_pending_sell_count": len(eod_pending_sells),
                "pretrade_nav_cny": pretrade_nav,
                "raw_close_snapshot_sha256": object_sha256(raw_session["eod_snapshot"]),
                "eod_portfolio_state_sha256": eod["state_sha256"],
                "pending_settlements_root_sha256": eod["pending_settlements_root_sha256"],
                "pending_settlement_count": eod["pending_settlement_count"],
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
        "arm_id": arm_id,
        "seed_id": seed_id,
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
    seed_id: int | None = None
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
            seed_id = result["seed_id"]
        elif identity != expected_identity:
            raise ExecutionReplayError("selection identities differ across cost scenarios")
        elif decision_identity != expected_decision_identity:
            raise ExecutionReplayError("new-entry identities differ across cost scenarios")
        elif str(result["trial_id"]) != trial_id or str(result["arm_id"]) != arm_id or result["seed_id"] != seed_id:
            raise ExecutionReplayError("trial, arm, or seed identity differs across cost scenarios")
        results[str(scenario_id)] = result
    output = {
        "schema": SCENARIO_RESULT_SCHEMA,
        "trial_id": trial_id,
        "arm_id": arm_id,
        "seed_id": seed_id,
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


def build_weekly_path_component(
    result: Mapping[str, Any],
    cycles: Sequence[Mapping[str, Any]],
    *,
    state_gate_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
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
    evidence_rows = result.get("session_evidence")
    if not isinstance(evidence_rows, list):
        raise ExecutionReplayError("execution result session evidence is missing")
    evidence_by_session = {str(row.get("session")): row for row in evidence_rows if isinstance(row, Mapping)}
    if len(evidence_by_session) != len(evidence_rows) or set(evidence_by_session) != set(by_session):
        raise ExecutionReplayError("execution daily rows and session evidence have different session domains")
    for session, row in by_session.items():
        state_root = _require_sha256(row.get("state_sha256"), f"daily[{session}].state_sha256")
        pending_root = _require_sha256(
            row.get("pending_settlements_root_sha256"), f"daily[{session}].pending_settlements_root_sha256"
        )
        pending_count = row.get("pending_settlement_count")
        pending_sell_count = row.get("pending_sell_count")
        evidence = evidence_by_session[session]
        if isinstance(pending_count, bool) or not isinstance(pending_count, int) or pending_count < 0:
            raise ExecutionReplayError(f"daily[{session}].pending_settlement_count is invalid")
        if isinstance(pending_sell_count, bool) or not isinstance(pending_sell_count, int) or pending_sell_count < 0:
            raise ExecutionReplayError(f"daily[{session}].pending_sell_count is invalid")
        _require_sha256(
            evidence.get("eod_pending_sells_sha256"), f"session_evidence[{session}].eod_pending_sells_sha256"
        )
        if (
            evidence.get("eod_portfolio_state_sha256") != state_root
            or evidence.get("eod_pending_sell_count") != pending_sell_count
        ):
            raise ExecutionReplayError(f"daily/session EOD state evidence differs on {session}")
        if (
            evidence.get("pending_settlements_root_sha256") != pending_root
            or evidence.get("pending_settlement_count") != pending_count
        ):
            raise ExecutionReplayError(f"daily/session pending-settlement evidence differs on {session}")
    decision_rows = {str(row.get("decision_session")): row for row in result.get("decision_evidence", [])}
    events = result.get("events")
    if not isinstance(events, list):
        raise ExecutionReplayError("execution result events are missing")

    week_labels: list[str] = []
    weekly_returns: list[float] = []
    weekly_turnover: list[float] = []
    weekly_pretrade_nav: list[float] = []
    weekly_explicit_costs: list[dict[str, float]] = []
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
        pretrade_nav = _finite_number(decision.get("pretrade_nav_cny"), f"{start}.pretrade_nav_cny", minimum=0.0)
        if pretrade_nav <= 0:
            raise ExecutionReplayError("weekly pretrade NAV must be positive")
        weekly_turnover.append(filled_notional / (2.0 * pretrade_nav))
        weekly_pretrade_nav.append(pretrade_nav)
        weekly_events = [
            event for event in events if event.get("event_type") == "order" and start < str(event.get("session")) <= end
        ]
        weekly_explicit_costs.append(
            {
                component: float(sum(float(event.get(component, 0.0)) for event in weekly_events))
                for component in (
                    "commission_cny",
                    "transfer_fee_cny",
                    "stamp_duty_cny",
                    "slippage_cost_cny",
                )
            }
        )
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
    confirmation_orders = [
        {
            "session": str(event["session"]),
            "symbol": str(event["symbol"]),
            "side": str(event["side"]),
            "requested_shares": int(event["requested_shares"]),
            "filled_shares": int(event["filled_shares"]),
            "fill_notional_cny": float(event.get("fill_notional_cny", 0.0)),
            "adv20_cny_asof_decision": float(event["adv20_cny_asof_decision"]),
            "open_auction_turnover_cny": float(event["open_auction_turnover_cny"]),
        }
        for event in events
        if event.get("event_type") == "order" and first_start < str(event.get("session")) <= final_end
    ]
    confirmation_orders.sort(key=lambda item: (item["session"], item["symbol"], item["side"]))

    final_row = by_session[final_end]
    starting_positions = [
        {"symbol": str(item["symbol"]), "shares": int(item["shares"])} for item in final_row["holdings"]
    ]
    starting_positions.sort(key=lambda item: item["symbol"])
    positions = {item["symbol"]: item["shares"] for item in starting_positions}
    starting_pending_settlements_root = str(final_row["pending_settlements_root_sha256"])
    starting_pending_settlement_count = int(final_row["pending_settlement_count"])

    indexed_events = list(enumerate(events))
    pending_conversions: dict[str, str] = {}
    for _, event in indexed_events:
        event_session = str(event.get("session", ""))
        if event_session > final_end:
            continue
        if (
            event.get("event_type") == "corporate_action"
            and event.get("action_type") == "delist_share"
            and int(event.get("share_change", 0)) < 0
        ):
            pending_conversions[str(event["action_id"])] = str(event["symbol"])
        elif event.get("event_type") == "terminal_consideration_settlement" and event.get("target_symbol") is not None:
            pending_conversions.pop(str(event["action_id"]), None)
    starting_pending_conversions = [
        {"action_id": action_id, "source_symbol": pending_conversions[action_id]}
        for action_id in sorted(pending_conversions)
    ]

    post_window_events = [
        (event_index, event) for event_index, event in indexed_events if str(event.get("session", "")) > final_end
    ]
    if any(event.get("event_type") == "order" and event.get("side") == "buy" for _, event in post_window_events):
        raise ExecutionReplayError("post-window unwind cannot contain a new buy order")

    unwind_rows = [row for row in daily_rows if str(row["session"]) > final_end]
    unwind_labels = [str(row["session"]) for row in unwind_rows]
    if unwind_labels != sorted(unwind_labels):
        raise ExecutionReplayError("post-window EOD sessions must be strictly increasing")
    unwind_session_labels = {str(row["session"]) for row in unwind_rows}
    orphan_post_window_events = [
        event for _, event in post_window_events if str(event.get("session", "")) not in unwind_session_labels
    ]
    if orphan_post_window_events:
        raise ExecutionReplayError("post-window event has no matching EOD session")

    economic_state_fields = (
        "nav_cny",
        "cash_cny",
        "cash_yield_cny",
        "position_value_cny",
        "cash_receivable_value_cny",
        "other_asset_value_cny",
        "other_liability_value_cny",
        "dividend_tax_liability_cny",
        "gross_exposure",
        "holding_count",
        "pending_sell_count",
        "pending_sell_symbols",
        "pending_settlements_root_sha256",
        "pending_settlement_count",
        "contingent_slot_symbols",
        "holdings",
    )

    def economic_state(row: Mapping[str, Any]) -> dict[str, Any]:
        session = str(row["session"])
        return {
            **{field: row[field] for field in economic_state_fields},
            "pending_sells_sha256": evidence_by_session[session]["eod_pending_sells_sha256"],
        }

    unwind_sessions: list[dict[str, Any]] = []
    post_completion_heartbeats: list[dict[str, Any]] = []
    completion_session: str | None = (
        final_end
        if (
            not positions
            and not pending_conversions
            and starting_pending_settlement_count == 0
            and int(final_row["pending_sell_count"]) == 0
        )
        else None
    )
    completion_economic_state: dict[str, Any] | None = (
        economic_state(final_row) if completion_session is not None else None
    )
    completion_economic_state_sha256: str | None = (
        object_sha256(completion_economic_state) if completion_economic_state is not None else None
    )
    completion_state_sha256: str | None = (
        _require_sha256(final_row["state_sha256"], f"daily[{final_end}].state_sha256")
        if completion_session is not None
        else None
    )
    prior_heartbeat_state_sha256 = completion_state_sha256
    for row in unwind_rows:
        session = str(row["session"])
        session_events = [(index, event) for index, event in post_window_events if str(event.get("session")) == session]
        if completion_session is not None:
            administrative_events: list[dict[str, Any]] = []
            expected_event_fields = {
                "event_type",
                "session",
                "symbol",
                "action_id",
                "action_type",
                "status",
                "cash_change_cny",
                "share_change",
                "deferred_tax_settled_cny",
            }
            for event_index, event in session_events:
                zero_impact_no_position = (
                    set(event) == expected_event_fields
                    and event.get("event_type") == "corporate_action"
                    and event.get("status") == "no_position"
                    and event.get("session") == session
                    and isinstance(event.get("symbol"), str)
                    and bool(event.get("symbol"))
                    and isinstance(event.get("action_id"), str)
                    and bool(event.get("action_id"))
                    and event.get("action_type")
                    in {"share_change", "cash_dividend", "rights_issue", *TERMINAL_ACTION_TYPES}
                    and not isinstance(event.get("cash_change_cny"), bool)
                    and event.get("cash_change_cny") == 0
                    and not isinstance(event.get("share_change"), bool)
                    and event.get("share_change") == 0
                    and not isinstance(event.get("deferred_tax_settled_cny"), bool)
                    and event.get("deferred_tax_settled_cny") == 0
                )
                if not zero_impact_no_position:
                    raise ExecutionReplayError(
                        "post-window replay contains an event after unwind completion that is not a "
                        "zero-impact no-position corporate action"
                    )
                administrative_events.append({"event_index": event_index, "event": dict(event)})
            heartbeat_state = economic_state(row)
            heartbeat_state_sha256 = object_sha256(heartbeat_state)
            if (
                heartbeat_state != completion_economic_state
                or heartbeat_state_sha256 != completion_economic_state_sha256
            ):
                raise ExecutionReplayError("post-window replay economic state changes after unwind completion")
            state_sha256 = _require_sha256(row["state_sha256"], f"daily[{session}].state_sha256")
            if administrative_events:
                if state_sha256 == prior_heartbeat_state_sha256:
                    raise ExecutionReplayError(
                        "post-window administrative actions after unwind completion must advance the state hash"
                    )
            elif state_sha256 != prior_heartbeat_state_sha256:
                raise ExecutionReplayError(
                    "post-window replay state hash changes after unwind completion without an administrative action"
                )
            post_completion_heartbeats.append(
                {
                    "session": session,
                    "state_sha256": state_sha256,
                    **heartbeat_state,
                    "economic_state_sha256": heartbeat_state_sha256,
                    "administrative_events": administrative_events,
                    "session_events_root_sha256": object_sha256(administrative_events),
                    "session_event_count": len(administrative_events),
                }
            )
            prior_heartbeat_state_sha256 = state_sha256
            continue
        terminal_dispositions: list[dict[str, Any]] = []
        terminal_share_receipts: list[dict[str, Any]] = []
        for event_index, event in session_events:
            if (
                event.get("event_type") == "corporate_action"
                and event.get("action_type") in TERMINAL_ACTION_TYPES
                and int(event.get("share_change", 0)) < 0
            ):
                symbol = str(event["symbol"])
                action_id = str(event["action_id"])
                action_type = str(event["action_type"])
                disposed = -int(event["share_change"])
                available = positions.get(symbol, 0)
                if available == 0 or disposed != available:
                    raise ExecutionReplayError("terminal unwind disposition must consume one complete symbol position")
                positions.pop(symbol)
                if action_type == "delist_share":
                    if action_id in pending_conversions:
                        raise ExecutionReplayError("terminal unwind action_id duplicates a pending share conversion")
                    pending_conversions[action_id] = symbol
                terminal_dispositions.append(
                    {
                        "event_index": event_index,
                        "action_id": action_id,
                        "action_type": action_type,
                        "symbol": symbol,
                        "disposed_shares": disposed,
                    }
                )
            elif (
                event.get("event_type") == "terminal_consideration_settlement"
                and event.get("target_symbol") is not None
            ):
                action_id = str(event["action_id"])
                source_symbol = str(event["symbol"])
                target_symbol = str(event["target_symbol"])
                received = int(event["target_shares"])
                if pending_conversions.get(action_id) != source_symbol:
                    raise ExecutionReplayError("terminal share receipt has no matching pending source conversion")
                if received > 0:
                    positions[target_symbol] = positions.get(target_symbol, 0) + received
                pending_conversions.pop(action_id)
                terminal_share_receipts.append(
                    {
                        "event_index": event_index,
                        "action_id": action_id,
                        "source_symbol": source_symbol,
                        "target_symbol": target_symbol,
                        "received_shares": received,
                    }
                )

        session_orders = [
            event for _, event in session_events if event.get("event_type") == "order" and event.get("side") == "sell"
        ]
        sell_orders: list[dict[str, Any]] = []
        seen_sell_symbols: set[str] = set()
        for event in sorted(session_orders, key=lambda item: str(item["symbol"])):
            symbol = str(event["symbol"])
            requested = int(event["requested_shares"])
            filled = int(event["filled_shares"])
            if symbol in seen_sell_symbols:
                raise ExecutionReplayError("post-window unwind has duplicate sell attempts for one symbol/session")
            if positions.get(symbol, 0) != requested or filled > requested:
                raise ExecutionReplayError("post-window unwind must request each symbol's complete current position")
            seen_sell_symbols.add(symbol)
            positions[symbol] -= filled
            if positions[symbol] == 0:
                positions.pop(symbol)
            sell_orders.append({"symbol": symbol, "requested_shares": requested, "filled_shares": filled})

        remaining_positions = [{"symbol": symbol, "shares": positions[symbol]} for symbol in sorted(positions)]
        replayed_remaining = [
            {"symbol": str(item["symbol"]), "shares": int(item["shares"])} for item in row["holdings"]
        ]
        replayed_remaining.sort(key=lambda item: item["symbol"])
        if remaining_positions != replayed_remaining:
            raise ExecutionReplayError("post-window symbol positions do not conserve through actions and fills")

        filled_shares = sum(item["filled_shares"] for item in sell_orders)
        fill_notional = float(sum(float(event.get("fill_notional_cny", 0.0)) for event in session_orders))
        if (filled_shares == 0) != (fill_notional == 0):
            raise ExecutionReplayError("post-window filled shares and fill notional are inconsistent")
        pending_settlements_root = str(row["pending_settlements_root_sha256"])
        pending_settlement_count = int(row["pending_settlement_count"])
        unwind_sessions.append(
            {
                "session": session,
                "sell_orders": sell_orders,
                "terminal_dispositions": terminal_dispositions,
                "terminal_share_receipts": terminal_share_receipts,
                "remaining_positions": remaining_positions,
                "pending_settlements_root_sha256": pending_settlements_root,
                "pending_settlement_count": pending_settlement_count,
                "session_events_root_sha256": object_sha256(
                    [{"event_index": index, "event": dict(event)} for index, event in session_events]
                ),
                "session_event_count": len(session_events),
                "fill_notional_cny": fill_notional,
                **{
                    component: float(sum(float(event.get(component, 0.0)) for event in session_orders))
                    for component in (
                        "commission_cny",
                        "transfer_fee_cny",
                        "stamp_duty_cny",
                        "slippage_cost_cny",
                    )
                },
            }
        )
        if (
            not positions
            and not pending_conversions
            and pending_settlement_count == 0
            and int(row["pending_sell_count"]) == 0
        ):
            completion_session = session
            completion_economic_state = economic_state(row)
            completion_economic_state_sha256 = object_sha256(completion_economic_state)
            completion_state_sha256 = _require_sha256(row["state_sha256"], f"daily[{session}].state_sha256")
            prior_heartbeat_state_sha256 = completion_state_sha256

    if (
        completion_session is None
        or completion_economic_state is None
        or completion_economic_state_sha256 is None
        or completion_state_sha256 is None
    ):
        raise ExecutionReplayError(
            "post-window unwind must clear positions, pending share conversions, and pending settlements"
        )
    component = {
        "week_labels": week_labels,
        "daily_labels": daily_labels,
        "path": {
            "weekly_returns": weekly_returns,
            "daily_post_close_gross_exposure": [float(row["gross_exposure"]) for row in daily_window],
            "weekly_one_way_turnover": weekly_turnover,
            "weekly_new_entry_identity_sets": new_entry_sets,
            "weekly_gate_eligible_new_entry_opportunities": opportunities,
            "weekly_pretrade_nav_cny": weekly_pretrade_nav,
            "weekly_explicit_costs_cny": weekly_explicit_costs,
            "order_attempts": confirmation_orders,
            "state_gate_observations": [dict(item) for item in (state_gate_observations or [])],
            "post_window_unwind": {
                "confirmation_end_session": final_end,
                "completion_session": completion_session,
                "completion_economic_state": completion_economic_state,
                "completion_economic_state_sha256": completion_economic_state_sha256,
                "completion_state_sha256": completion_state_sha256,
                "starting_positions": starting_positions,
                "pending_share_conversions": starting_pending_conversions,
                "starting_pending_settlements_root_sha256": starting_pending_settlements_root,
                "starting_pending_settlement_count": starting_pending_settlement_count,
                "sessions": unwind_sessions,
                "post_completion_heartbeats": post_completion_heartbeats,
            },
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
    "PENDING_SETTLEMENT_SCHEMA",
    "PORTFOLIO_STATE_SCHEMA",
    "SCENARIO_RESULT_SCHEMA",
    "canonical_json_bytes",
    "build_weekly_path_component",
    "empty_portfolio",
    "event_notional_and_costs",
    "object_sha256",
    "pending_settlements_summary",
    "replay_cost_scenarios",
    "replay_execution",
    "selection_identity",
    "verify_execution_result",
]
