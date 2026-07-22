"""Preregistered V2.1 weekly statistics and three-way evidence decisions.

The functions in this module operate on already replayed portfolio paths.  They
do not accept precomputed t statistics, confidence intervals, or pass/fail
booleans.  Every formal comparison is reconstructed from the first 52 frozen
weekly returns, including equal-weight aggregation across every frozen random
seed before time-series inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

STATISTICS_INPUT_SCHEMA = "xs_chan_statistics_input_v2_1"
STATISTICS_RESULT_SCHEMA = "xs_chan_statistics_result_v2_1"
WEEKLY_PATH_SCHEMA = "xs_chan_weekly_path_v2_1"
RISK_MECHANISM_REPORT_SCHEMA = "xs_chan_risk_mechanism_report_v2_1"
EXPECTED_COMPLETE_WEEKS = 52
EXPECTED_RANDOM_SEEDS = tuple(range(20260720, 20260740))
EXPECTED_RISK_FAMILIES = ("F", "R_match", "FC", "FGR", "FMA", "FMGR")
EXPECTED_RISK_SCENARIOS = ("gross", "1x", "2x", "capacity_1x")
EXPECTED_RISK_ARM_IDS = tuple(
    f"{family}_{scenario}" for family in EXPECTED_RISK_FAMILIES for scenario in EXPECTED_RISK_SCENARIOS
)
EXPECTED_SEEDED_FAMILIES = frozenset({"R_match", "FGR", "FMGR"})
EXPLICIT_COST_COMPONENTS = ("commission_cny", "transfer_fee_cny", "stamp_duty_cny", "slippage_cost_cny")
ALLOWED_CHAN_REGIMES = frozenset({5, 6, 7, 8})
TERMINAL_ACTION_TYPES = frozenset({"delist_cash", "delist_share", "delist_writeoff"})
EXPECTED_COMPARISONS = (
    "F_2x_minus_R_match_2x",
    "FC_gross_minus_FGR_gross",
    "FC_2x_minus_FGR_2x",
    "FC_gross_minus_FMA_gross",
    "FC_2x_minus_FMA_2x",
    "FMA_gross_minus_FMGR_gross",
    "FMA_2x_minus_FMGR_2x",
)
EXPECTED_CONTROL_THRESHOLDS = {
    "maximum_top10_positive_weeks_share": 0.50,
    "maximum_mean_absolute_exposure_gap": 0.01,
    "maximum_p95_absolute_exposure_gap": 0.03,
    "maximum_absolute_exposure_gap": 0.05,
    "maximum_mean_absolute_turnover_gap": 0.025,
    "maximum_p95_absolute_turnover_gap": 0.10,
    "maximum_random_annualized_seed_mean_mcse": 0.005,
    "maximum_bootstrap_quantile_mcse_annualized": 0.0025,
    "minimum_gate_eligible_new_entry_opportunities": 260.0,
}
EXPECTED_COMPARISON_CORE = {
    "F_2x_minus_R_match_2x": ("factor_primary", "F_2x", "R_match_2x", "2x", True, 0.03, True),
    "FC_gross_minus_FGR_gross": (
        "chan_identity_primary",
        "FC_gross",
        "FGR_gross",
        "gross",
        True,
        0.02,
        True,
    ),
    "FC_2x_minus_FGR_2x": ("chan_cost_robustness", "FC_2x", "FGR_2x", "2x", True, 0.015, True),
    "FC_gross_minus_FMA_gross": (
        "chan_specificity_primary",
        "FC_gross",
        "FMA_gross",
        "gross",
        True,
        0.01,
        False,
    ),
    "FC_2x_minus_FMA_2x": (
        "chan_specificity_cost_robustness",
        "FC_2x",
        "FMA_2x",
        "2x",
        True,
        0.01,
        False,
    ),
    "FMA_gross_minus_FMGR_gross": (
        "ma_placebo_identity_diagnostic",
        "FMA_gross",
        "FMGR_gross",
        "gross",
        False,
        0.01,
        True,
    ),
    "FMA_2x_minus_FMGR_2x": (
        "ma_placebo_cost_diagnostic",
        "FMA_2x",
        "FMGR_2x",
        "2x",
        False,
        0.01,
        True,
    ),
}
EXPECTED_PASS_RULE = (
    "hac_lower_95_and_bootstrap_q05_strictly_greater_than_sesoi_with_positive_weekly_median_and_top10_share_lte_0_50"
)
EXPECTED_FALSIFIED_RULE = "hac_upper_95_and_bootstrap_q95_strictly_less_than_sesoi"


class StatisticsError(ValueError):
    """Raised when formal statistical input is incomplete or not preregistered."""


class DegenerateInferenceError(StatisticsError):
    """Raised when a HAC/variance calculation cannot support valid inference."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256 = object_sha256(
    {
        "schema": "xs_chan_pending_settlements_v2_1",
        "cash_receivables": {},
        "share_entitlements": {},
        "terminal_considerations": {},
        "dividend_tax_liabilities": {},
        "pending_settlement_count": 0,
    }
)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StatisticsError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StatisticsError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticsError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number):
        raise StatisticsError(f"{label} must be finite numeric")
    return number


def _as_weekly_array(value: Sequence[Any], label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result <= -1.0):
        raise StatisticsError(f"{label} contains a return <= -100%")
    return result


def _as_exposure_array(value: Sequence[Any], label: str, expected_length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_length:
        raise StatisticsError(f"{label} must contain exactly {expected_length} daily observations")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result < -1e-12) or np.any(result > 1.0 + 1e-12):
        raise StatisticsError(f"{label} must be post-close gross exposure in [0, 1]")
    return result


def _as_turnover_array(value: Sequence[Any], label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result < 0):
        raise StatisticsError(f"{label} must contain non-negative one-way turnover")
    return result


def _validate_week_labels(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} labels")
    labels = tuple(str(item) for item in value)
    if any(not item for item in labels) or len(labels) != len(set(labels)) or labels != tuple(sorted(labels)):
        raise StatisticsError(f"{label} must be unique, non-empty, and increasing")
    return labels


def _validate_daily_labels(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain the full daily confirmation window")
    labels = tuple(str(item) for item in value)
    if any(not item for item in labels) or len(labels) != len(set(labels)) or labels != tuple(sorted(labels)):
        raise StatisticsError(f"{label} must be unique, non-empty, and increasing")
    return labels


def _validate_selection_sets(value: Any, label: str) -> tuple[frozenset[str], ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result: list[frozenset[str]] = []
    for week, symbols in enumerate(value):
        if not isinstance(symbols, list) or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise StatisticsError(f"{label}[{week}] must be a list of symbols")
        if len(symbols) != len(set(symbols)):
            raise StatisticsError(f"{label}[{week}] contains duplicate symbols")
        result.append(frozenset(symbols))
    return tuple(result)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StatisticsError(f"{label} must be a non-negative integer")
    return value


def _split_risk_arm_id(arm_id: str) -> tuple[str, str]:
    for scenario in sorted(EXPECTED_RISK_SCENARIOS, key=len, reverse=True):
        suffix = f"_{scenario}"
        if arm_id.endswith(suffix):
            family = arm_id[: -len(suffix)]
            if family in EXPECTED_RISK_FAMILIES:
                return family, scenario
    raise StatisticsError(f"unknown V2.1 risk arm {arm_id!r}")


@dataclass(frozen=True)
class StateObservation:
    week_label: str
    execution_session: str
    symbol: str
    regime: int
    eligible: bool
    allowed: bool
    requested_shares: int
    filled_shares: int
    entry_weight: float | None
    holding_sessions: int | None
    forward_return: float | None


@dataclass(frozen=True)
class UnwindSummary:
    starting_shares: int
    official_sessions_to_complete: int
    fill_notional_cny: float
    costs_cny: dict[str, float]


@dataclass(frozen=True)
class ParsedRawPath:
    returns: np.ndarray
    exposure: np.ndarray
    turnover: np.ndarray
    selections: tuple[frozenset[str], ...]
    gate_opportunities: np.ndarray
    weekly_cost_rates: dict[str, np.ndarray]
    order_count: int
    positive_fill_order_count: int
    requested_shares: int
    filled_shares: int
    capacity_usage: np.ndarray
    state_observations: tuple[StateObservation, ...]
    unwind: UnwindSummary


@dataclass(frozen=True)
class ArmPath:
    arm_id: str
    family_id: str
    scenario_id: str
    week_labels: tuple[str, ...]
    daily_labels: tuple[str, ...]
    returns: np.ndarray
    exposure: np.ndarray
    turnover: np.ndarray
    selections: tuple[frozenset[str], ...]
    gate_opportunities: np.ndarray
    weekly_cost_rates: dict[str, np.ndarray]
    order_count: int
    positive_fill_order_count: int
    requested_shares: int
    filled_shares: int
    capacity_usage: np.ndarray
    state_observations: tuple[StateObservation, ...]
    unwind_summaries: tuple[UnwindSummary, ...]
    seed_returns: np.ndarray | None
    seed_exposure: np.ndarray | None
    seed_turnover: np.ndarray | None
    seed_selections: tuple[tuple[frozenset[str], ...], ...] | None
    seed_gate_opportunities: np.ndarray | None
    seed_ids: tuple[int, ...]

    @property
    def seeded(self) -> bool:
        return self.seed_returns is not None


def _parse_order_attempts(
    value: Any, label: str, daily_labels: tuple[str, ...]
) -> tuple[int, int, int, int, np.ndarray, dict[tuple[str, str], dict]]:
    exact = {
        "session",
        "symbol",
        "side",
        "requested_shares",
        "filled_shares",
        "fill_notional_cny",
        "adv20_cny_asof_decision",
        "open_auction_turnover_cny",
    }
    if not isinstance(value, list):
        raise StatisticsError(f"{label} must be a list")
    daily_set = set(daily_labels)
    normalised: list[dict[str, Any]] = []
    keys: list[tuple[str, str, str]] = []
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != exact:
            raise StatisticsError(f"{item_label} must have exact V2.1 order fields")
        session = str(raw["session"])
        symbol = str(raw["symbol"])
        side = str(raw["side"])
        if session not in daily_set or not symbol or side not in {"buy", "sell"}:
            raise StatisticsError(f"{item_label} has an invalid identity")
        requested = _nonnegative_int(raw["requested_shares"], f"{item_label}.requested_shares")
        filled = _nonnegative_int(raw["filled_shares"], f"{item_label}.filled_shares")
        if requested <= 0 or filled > requested:
            raise StatisticsError(f"{item_label} must request positive shares and cannot overfill")
        notional = _finite(raw["fill_notional_cny"], f"{item_label}.fill_notional_cny")
        adv = _finite(raw["adv20_cny_asof_decision"], f"{item_label}.adv20_cny_asof_decision")
        auction = _finite(raw["open_auction_turnover_cny"], f"{item_label}.open_auction_turnover_cny")
        if min(notional, adv, auction) < 0 or (filled == 0) != (notional == 0):
            raise StatisticsError(f"{item_label} has inconsistent fill notional")
        capacity = min(0.05 * adv, 0.10 * auction)
        if notional > capacity + max(capacity, 1.0) * 1e-10:
            raise StatisticsError(f"{item_label} exceeds the frozen dual opening capacity")
        usage = 0.0 if capacity == 0 else notional / capacity
        keys.append((session, symbol, side))
        normalised.append(
            {
                "session": session,
                "symbol": symbol,
                "side": side,
                "requested_shares": requested,
                "filled_shares": filled,
                "fill_notional_cny": notional,
                "capacity_usage": usage,
            }
        )
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise StatisticsError(f"{label} must be unique and sorted by session, symbol, side")
    buy_index = {(item["session"], item["symbol"]): item for item in normalised if item["side"] == "buy"}
    return (
        len(normalised),
        sum(item["filled_shares"] > 0 for item in normalised),
        sum(item["requested_shares"] for item in normalised),
        sum(item["filled_shares"] for item in normalised),
        np.asarray([item["capacity_usage"] for item in normalised], dtype=float),
        buy_index,
    )


def _parse_state_observations(
    value: Any,
    *,
    label: str,
    family_id: str,
    week_labels: tuple[str, ...],
    daily_labels: tuple[str, ...],
    gate_opportunities: np.ndarray,
    buy_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[StateObservation, ...]:
    exact = {
        "week_label",
        "execution_session",
        "symbol",
        "regime",
        "eligible_new_entry",
        "allowed_new_entry",
        "requested_shares",
        "filled_shares",
        "entry_fill_notional_cny",
        "pretrade_nav_cny",
        "entry_price",
        "exit_or_window_session",
        "exit_or_window_price",
    }
    if not isinstance(value, list):
        raise StatisticsError(f"{label} must be a list")
    if family_id != "FC" and value:
        raise StatisticsError(f"{label} is only valid for the frozen Chan-gated FC family")
    day_index = {session: index for index, session in enumerate(daily_labels)}
    week_set = set(week_labels)
    observations: list[StateObservation] = []
    identities: list[tuple[str, str]] = []
    allowed_by_week = dict.fromkeys(week_labels, 0)
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != exact:
            raise StatisticsError(f"{item_label} must have exact V2.1 state-observation fields")
        week = str(raw["week_label"])
        session = str(raw["execution_session"])
        symbol = str(raw["symbol"])
        if week not in week_set or session not in day_index or not symbol:
            raise StatisticsError(f"{item_label} has an unknown week/session/symbol")
        regime = _nonnegative_int(raw["regime"], f"{item_label}.regime")
        if regime > 10:
            raise StatisticsError(f"{item_label}.regime must be in 0..10")
        if type(raw["eligible_new_entry"]) is not bool or type(raw["allowed_new_entry"]) is not bool:
            raise StatisticsError(f"{item_label} eligibility fields must be boolean")
        eligible = bool(raw["eligible_new_entry"])
        allowed = bool(raw["allowed_new_entry"])
        if allowed != (eligible and regime in ALLOWED_CHAN_REGIMES):
            raise StatisticsError(f"{item_label} changes the frozen Chan state gate")
        requested = _nonnegative_int(raw["requested_shares"], f"{item_label}.requested_shares")
        filled = _nonnegative_int(raw["filled_shares"], f"{item_label}.filled_shares")
        if filled > requested or (requested > 0 and not allowed):
            raise StatisticsError(f"{item_label} has an ineligible request or overfill")
        fill_notional = _finite(raw["entry_fill_notional_cny"], f"{item_label}.entry_fill_notional_cny")
        nav = _finite(raw["pretrade_nav_cny"], f"{item_label}.pretrade_nav_cny")
        if fill_notional < 0 or nav <= 0 or (filled == 0) != (fill_notional == 0):
            raise StatisticsError(f"{item_label} has inconsistent entry notional/NAV")
        order = buy_index.get((session, symbol))
        if requested > 0 and (
            order is None
            or order["requested_shares"] != requested
            or order["filled_shares"] != filled
            or not math.isclose(order["fill_notional_cny"], fill_notional, rel_tol=1e-12, abs_tol=1e-9)
        ):
            raise StatisticsError(f"{item_label} does not bind to its raw buy order")
        entry_price_raw = raw["entry_price"]
        exit_session_raw = raw["exit_or_window_session"]
        exit_price_raw = raw["exit_or_window_price"]
        if filled == 0:
            if any(item is not None for item in (entry_price_raw, exit_session_raw, exit_price_raw)):
                raise StatisticsError(f"{item_label} unfilled observation must not fabricate holding outcomes")
            entry_weight = None
            holding_sessions = None
            forward_return = None
        else:
            entry_price = _finite(entry_price_raw, f"{item_label}.entry_price")
            exit_price = _finite(exit_price_raw, f"{item_label}.exit_or_window_price")
            exit_session = str(exit_session_raw)
            if entry_price <= 0 or exit_price <= 0 or exit_session not in day_index:
                raise StatisticsError(f"{item_label} has invalid entry/exit marks")
            if day_index[exit_session] < day_index[session]:
                raise StatisticsError(f"{item_label} exits before its entry")
            if not math.isclose(entry_price * filled, fill_notional, rel_tol=1e-12, abs_tol=1e-8):
                raise StatisticsError(f"{item_label}.entry_price does not reconstruct fill notional")
            entry_weight = fill_notional / nav
            if entry_weight > 1.0 + 1e-10:
                raise StatisticsError(f"{item_label}.entry weight exceeds NAV")
            holding_sessions = day_index[exit_session] - day_index[session] + 1
            forward_return = exit_price / entry_price - 1.0
        allowed_by_week[week] += int(allowed)
        identities.append((week, symbol))
        observations.append(
            StateObservation(
                week,
                session,
                symbol,
                regime,
                eligible,
                allowed,
                requested,
                filled,
                entry_weight,
                holding_sessions,
                forward_return,
            )
        )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise StatisticsError(f"{label} must be unique and sorted by week_label, symbol")
    state_buy_keys = {(item.execution_session, item.symbol) for item in observations if item.requested_shares > 0}
    if family_id == "FC" and not state_buy_keys.issubset(set(buy_index)):
        raise StatisticsError(f"{label} contains a state-gated request absent from raw FC buy orders")
    if family_id == "FC" and [allowed_by_week[week] for week in week_labels] != gate_opportunities.tolist():
        raise StatisticsError(f"{label} allowed counts do not reconstruct weekly gate opportunities")
    return tuple(observations)


def _parse_symbol_positions(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, list):
        raise StatisticsError(f"{label} must be a list")
    positions: dict[str, int] = {}
    symbols: list[str] = []
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"symbol", "shares"}:
            raise StatisticsError(f"{item_label} must have exact symbol-position fields")
        symbol = raw["symbol"]
        if not isinstance(symbol, str) or not symbol:
            raise StatisticsError(f"{item_label}.symbol must be a non-empty string")
        shares = _nonnegative_int(raw["shares"], f"{item_label}.shares")
        if shares == 0:
            raise StatisticsError(f"{item_label}.shares must be positive")
        symbols.append(symbol)
        positions[symbol] = shares
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise StatisticsError(f"{label} must be unique and sorted by symbol")
    return positions


def _parse_pending_share_conversions(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise StatisticsError(f"{label} must be a list")
    pending: dict[str, str] = {}
    action_ids: list[str] = []
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"action_id", "source_symbol"}:
            raise StatisticsError(f"{item_label} must have exact pending-conversion fields")
        action_id = raw["action_id"]
        source_symbol = raw["source_symbol"]
        if not isinstance(action_id, str) or not action_id:
            raise StatisticsError(f"{item_label}.action_id must be a non-empty string")
        if not isinstance(source_symbol, str) or not source_symbol:
            raise StatisticsError(f"{item_label}.source_symbol must be a non-empty string")
        action_ids.append(action_id)
        pending[action_id] = source_symbol
    if action_ids != sorted(action_ids) or len(action_ids) != len(set(action_ids)):
        raise StatisticsError(f"{label} must be unique and sorted by action_id")
    if len(set(pending.values())) != len(pending):
        raise StatisticsError(f"{label} cannot contain multiple pending conversions for one source symbol")
    return pending


def _parse_completion_economic_state(value: Any, label: str) -> dict[str, Any]:
    exact = {
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
        "pending_sells_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise StatisticsError(f"{label} must have exact completion-economic-state fields")
    state = dict(value)
    for field in (
        "nav_cny",
        "cash_cny",
        "cash_yield_cny",
        "position_value_cny",
        "cash_receivable_value_cny",
        "other_asset_value_cny",
        "other_liability_value_cny",
        "dividend_tax_liability_cny",
        "gross_exposure",
    ):
        state[field] = _finite(state[field], f"{label}.{field}")
    for field in ("holding_count", "pending_sell_count", "pending_settlement_count"):
        state[field] = _nonnegative_int(state[field], f"{label}.{field}")
    holdings = _parse_symbol_positions(state["holdings"], f"{label}.holdings")
    if state["holding_count"] != len(holdings):
        raise StatisticsError(f"{label}.holding_count differs from holdings")
    for field in ("pending_sell_symbols", "contingent_slot_symbols"):
        symbols = state[field]
        if (
            not isinstance(symbols, list)
            or any(not isinstance(item, str) or not item for item in symbols)
            or symbols != sorted(set(symbols))
        ):
            raise StatisticsError(f"{label}.{field} must be sorted unique non-empty symbols")
    if state["pending_sell_count"] != len(state["pending_sell_symbols"]):
        raise StatisticsError(f"{label}.pending_sell_count differs from pending_sell_symbols")
    pending_root = _require_sha256(state["pending_settlements_root_sha256"], f"{label}.pending_settlements_root_sha256")
    if state["pending_settlement_count"] == 0 and pending_root != EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256:
        raise StatisticsError(f"{label} zero pending settlements must use the canonical empty root")
    _require_sha256(state["pending_sells_sha256"], f"{label}.pending_sells_sha256")
    return state


def _parse_unwind(value: Any, label: str, confirmation_end_session: str) -> UnwindSummary:
    top_level_fields = {
        "confirmation_end_session",
        "completion_session",
        "completion_economic_state",
        "completion_economic_state_sha256",
        "completion_state_sha256",
        "starting_positions",
        "pending_share_conversions",
        "starting_pending_settlements_root_sha256",
        "starting_pending_settlement_count",
        "sessions",
        "post_completion_heartbeats",
    }
    if not isinstance(value, Mapping) or set(value) != top_level_fields:
        raise StatisticsError(f"{label} must have exact V2.1 unwind fields")
    if str(value["confirmation_end_session"]) != confirmation_end_session:
        raise StatisticsError(f"{label}.confirmation_end_session differs from the path endpoint")

    positions = _parse_symbol_positions(value["starting_positions"], f"{label}.starting_positions")
    pending = _parse_pending_share_conversions(value["pending_share_conversions"], f"{label}.pending_share_conversions")
    if set(positions).intersection(pending.values()):
        raise StatisticsError(f"{label} cannot hold a source symbol whose terminal share conversion is pending")
    starting_pending_count = _nonnegative_int(
        value["starting_pending_settlement_count"], f"{label}.starting_pending_settlement_count"
    )
    starting_pending_root = _require_sha256(
        value["starting_pending_settlements_root_sha256"],
        f"{label}.starting_pending_settlements_root_sha256",
    )
    if starting_pending_count == 0 and starting_pending_root != EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256:
        raise StatisticsError(f"{label} zero starting pending settlements must use the canonical empty root")
    starting = sum(positions.values())
    sessions = value["sessions"]
    if not isinstance(sessions, list):
        raise StatisticsError(f"{label}.sessions must be a list")
    starts_complete = not positions and not pending and starting_pending_count == 0
    if starts_complete and sessions:
        raise StatisticsError(f"{label} already-complete book must place later rows in post_completion_heartbeats")
    if not starts_complete and not sessions:
        raise StatisticsError(f"{label} must continue through complete real fill, disposition, or settlement")

    session_fields = {
        "session",
        "sell_orders",
        "terminal_dispositions",
        "terminal_share_receipts",
        "remaining_positions",
        "pending_settlements_root_sha256",
        "pending_settlement_count",
        "session_events_root_sha256",
        "session_event_count",
        "fill_notional_cny",
        *EXPLICIT_COST_COMPONENTS,
    }
    sell_fields = {"symbol", "requested_shares", "filled_shares"}
    disposition_fields = {"event_index", "action_id", "action_type", "symbol", "disposed_shares"}
    receipt_fields = {"event_index", "action_id", "source_symbol", "target_symbol", "received_shares"}
    previous_session = confirmation_end_session
    fill_notional = 0.0
    costs = dict.fromkeys(EXPLICIT_COST_COMPONENTS, 0.0)
    seen_disposition_actions = set(pending)
    seen_terminal_event_indices: set[int] = set()
    last_terminal_event_index = -1
    completed = starts_complete
    computed_completion_session = confirmation_end_session if starts_complete else None

    for index, raw in enumerate(sessions):
        item_label = f"{label}.sessions[{index}]"
        if completed:
            raise StatisticsError(
                f"{label}.sessions continues after symbol positions and pending conversions are complete"
            )
        if not isinstance(raw, Mapping) or set(raw) != session_fields:
            raise StatisticsError(f"{item_label} must have exact V2.1 unwind-session fields")
        session = raw["session"]
        if not isinstance(session, str) or not session or session <= previous_session:
            raise StatisticsError(f"{label}.sessions must be strictly after the confirmation window and increasing")

        dispositions = raw["terminal_dispositions"]
        if not isinstance(dispositions, list):
            raise StatisticsError(f"{item_label}.terminal_dispositions must be a list")
        disposition_keys: list[tuple[int, str]] = []
        terminal_events: list[tuple[int, str, Mapping[str, Any], str]] = []
        for disposition_index, disposition_raw in enumerate(dispositions):
            disposition_label = f"{item_label}.terminal_dispositions[{disposition_index}]"
            if not isinstance(disposition_raw, Mapping) or set(disposition_raw) != disposition_fields:
                raise StatisticsError(f"{disposition_label} must have exact terminal-disposition fields")
            event_index = _nonnegative_int(disposition_raw["event_index"], f"{disposition_label}.event_index")
            action_id = disposition_raw["action_id"]
            action_type = disposition_raw["action_type"]
            symbol = disposition_raw["symbol"]
            if not isinstance(action_id, str) or not action_id:
                raise StatisticsError(f"{disposition_label}.action_id must be a non-empty string")
            if action_type not in TERMINAL_ACTION_TYPES:
                raise StatisticsError(f"{disposition_label}.action_type is not a frozen terminal action")
            if not isinstance(symbol, str) or not symbol:
                raise StatisticsError(f"{disposition_label}.symbol must be a non-empty string")
            disposed = _nonnegative_int(disposition_raw["disposed_shares"], f"{disposition_label}.disposed_shares")
            if disposed == 0:
                raise StatisticsError(f"{disposition_label}.disposed_shares must be positive")
            if action_id in seen_disposition_actions:
                raise StatisticsError(f"{disposition_label}.action_id duplicates a terminal disposition")
            disposition_keys.append((event_index, action_id))
            terminal_events.append((event_index, "disposition", disposition_raw, disposition_label))
        if disposition_keys != sorted(disposition_keys) or len({item[1] for item in disposition_keys}) != len(
            disposition_keys
        ):
            raise StatisticsError(
                f"{item_label}.terminal_dispositions must be unique by action_id and sorted by event_index"
            )

        receipts = raw["terminal_share_receipts"]
        if not isinstance(receipts, list):
            raise StatisticsError(f"{item_label}.terminal_share_receipts must be a list")
        receipt_keys: list[tuple[int, str]] = []
        for receipt_index, receipt_raw in enumerate(receipts):
            receipt_label = f"{item_label}.terminal_share_receipts[{receipt_index}]"
            if not isinstance(receipt_raw, Mapping) or set(receipt_raw) != receipt_fields:
                raise StatisticsError(f"{receipt_label} must have exact terminal-share-receipt fields")
            event_index = _nonnegative_int(receipt_raw["event_index"], f"{receipt_label}.event_index")
            action_id = receipt_raw["action_id"]
            source_symbol = receipt_raw["source_symbol"]
            target_symbol = receipt_raw["target_symbol"]
            if not isinstance(action_id, str) or not action_id:
                raise StatisticsError(f"{receipt_label}.action_id must be a non-empty string")
            if not isinstance(source_symbol, str) or not source_symbol:
                raise StatisticsError(f"{receipt_label}.source_symbol must be a non-empty string")
            if not isinstance(target_symbol, str) or not target_symbol or target_symbol == source_symbol:
                raise StatisticsError(f"{receipt_label}.target_symbol must differ from the non-empty source symbol")
            _nonnegative_int(receipt_raw["received_shares"], f"{receipt_label}.received_shares")
            receipt_keys.append((event_index, action_id))
            terminal_events.append((event_index, "receipt", receipt_raw, receipt_label))
        if receipt_keys != sorted(receipt_keys) or len({item[1] for item in receipt_keys}) != len(receipt_keys):
            raise StatisticsError(
                f"{item_label}.terminal_share_receipts must be unique by action_id and sorted by event_index"
            )
        event_indices = [item[0] for item in terminal_events]
        if len(event_indices) != len(set(event_indices)):
            raise StatisticsError(f"{item_label} terminal event_index values must be unique within the session")
        if any(event_index in seen_terminal_event_indices for event_index in event_indices):
            raise StatisticsError(f"{item_label} reuses a terminal event_index from an earlier session")

        for event_index, event_kind, event_raw, event_label in sorted(terminal_events, key=lambda item: item[0]):
            if event_index <= last_terminal_event_index:
                raise StatisticsError(f"{event_label}.event_index is not globally increasing")
            seen_terminal_event_indices.add(event_index)
            last_terminal_event_index = event_index
            action_id = str(event_raw["action_id"])
            if event_kind == "disposition":
                symbol = str(event_raw["symbol"])
                disposed = int(event_raw["disposed_shares"])
                available = positions.get(symbol, 0)
                if available == 0 or disposed != available:
                    raise StatisticsError(f"{event_label} must dispose one complete symbol position")
                del positions[symbol]
                if event_raw["action_type"] == "delist_share":
                    pending[action_id] = symbol
                seen_disposition_actions.add(action_id)
            else:
                source_symbol = str(event_raw["source_symbol"])
                target_symbol = str(event_raw["target_symbol"])
                received = int(event_raw["received_shares"])
                expected_source = pending.get(action_id)
                if expected_source is None:
                    raise StatisticsError(f"{event_label} has no matching pending delist-share conversion")
                if source_symbol != expected_source:
                    raise StatisticsError(f"{event_label}.source_symbol does not match the pending conversion")
                if received > 0:
                    positions[target_symbol] = positions.get(target_symbol, 0) + received
                del pending[action_id]

        sell_orders = raw["sell_orders"]
        if not isinstance(sell_orders, list):
            raise StatisticsError(f"{item_label}.sell_orders must be a list")
        sell_symbols: list[str] = []
        total_filled = 0
        prior_positions = dict(positions)
        for sell_index, sell_raw in enumerate(sell_orders):
            sell_label = f"{item_label}.sell_orders[{sell_index}]"
            if not isinstance(sell_raw, Mapping) or set(sell_raw) != sell_fields:
                raise StatisticsError(f"{sell_label} must have exact unwind sell-order fields")
            symbol = sell_raw["symbol"]
            if not isinstance(symbol, str) or not symbol:
                raise StatisticsError(f"{sell_label}.symbol must be a non-empty string")
            requested = _nonnegative_int(sell_raw["requested_shares"], f"{sell_label}.requested_shares")
            filled = _nonnegative_int(sell_raw["filled_shares"], f"{sell_label}.filled_shares")
            if symbol not in prior_positions:
                raise StatisticsError(f"{sell_label} requests a symbol absent from prior unwind positions")
            if requested != prior_positions[symbol] or filled > requested:
                raise StatisticsError(f"{sell_label} violates the full-position unwind request")
            sell_symbols.append(symbol)
            positions[symbol] -= filled
            if positions[symbol] == 0:
                del positions[symbol]
            total_filled += filled
        if sell_symbols != sorted(sell_symbols) or len(sell_symbols) != len(set(sell_symbols)):
            raise StatisticsError(f"{item_label}.sell_orders must be unique and sorted by symbol")
        if set(sell_symbols) != set(prior_positions):
            raise StatisticsError(f"{item_label}.sell_orders must attempt every remaining symbol position")

        remaining = _parse_symbol_positions(raw["remaining_positions"], f"{item_label}.remaining_positions")
        if remaining != positions:
            raise StatisticsError(f"{item_label}.remaining_positions violates symbol-level unwind conservation")

        pending_settlement_count = _nonnegative_int(
            raw["pending_settlement_count"], f"{item_label}.pending_settlement_count"
        )
        pending_settlements_root = _require_sha256(
            raw["pending_settlements_root_sha256"], f"{item_label}.pending_settlements_root_sha256"
        )
        if pending_settlement_count == 0 and pending_settlements_root != EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256:
            raise StatisticsError(f"{item_label} zero pending settlements must use the canonical empty root")
        _require_sha256(raw["session_events_root_sha256"], f"{item_label}.session_events_root_sha256")
        session_event_count = _nonnegative_int(raw["session_event_count"], f"{item_label}.session_event_count")
        known_event_count = len(sell_orders) + len(dispositions) + len(receipts)
        if session_event_count < known_event_count:
            raise StatisticsError(f"{item_label}.session_event_count omits represented unwind events")

        notional = _finite(raw["fill_notional_cny"], f"{item_label}.fill_notional_cny")
        if notional < 0 or (total_filled == 0) != (notional == 0):
            raise StatisticsError(f"{item_label} has inconsistent unwind fill notional")
        for component in EXPLICIT_COST_COMPONENTS:
            cost = _finite(raw[component], f"{item_label}.{component}")
            if cost < 0 or (total_filled == 0 and cost != 0):
                raise StatisticsError(f"{item_label}.{component} is inconsistent with filled shares")
            costs[component] += cost
        fill_notional += notional
        previous_session = session
        completed = not positions and not pending and pending_settlement_count == 0
        if completed:
            computed_completion_session = session

    if not completed:
        raise StatisticsError(f"{label} must continue through complete real fill, disposition, or settlement")
    if value["completion_session"] != computed_completion_session:
        raise StatisticsError(f"{label}.completion_session differs from the first triple-zero session")

    completion_state = _parse_completion_economic_state(
        value["completion_economic_state"], f"{label}.completion_economic_state"
    )
    completion_state_sha = _require_sha256(
        value["completion_economic_state_sha256"], f"{label}.completion_economic_state_sha256"
    )
    if object_sha256(completion_state) != completion_state_sha:
        raise StatisticsError(f"{label}.completion_economic_state_sha256 differs from its content")
    _require_sha256(value["completion_state_sha256"], f"{label}.completion_state_sha256")
    if (
        completion_state["holding_count"] != 0
        or completion_state["pending_sell_count"] != 0
        or completion_state["pending_settlement_count"] != 0
        or completion_state["holdings"] != []
        or completion_state["pending_sell_symbols"] != []
        or completion_state["position_value_cny"] != 0
        or completion_state["cash_receivable_value_cny"] != 0
        or completion_state["other_asset_value_cny"] != 0
        or completion_state["other_liability_value_cny"] != 0
        or completion_state["dividend_tax_liability_cny"] != 0
        or completion_state["gross_exposure"] != 0
    ):
        raise StatisticsError(f"{label}.completion_economic_state is not economically flat")

    heartbeat_fields = {
        "session",
        "state_sha256",
        *completion_state,
        "economic_state_sha256",
        "administrative_events",
        "session_events_root_sha256",
        "session_event_count",
    }
    administrative_event_fields = {
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
    heartbeats = value["post_completion_heartbeats"]
    if not isinstance(heartbeats, list):
        raise StatisticsError(f"{label}.post_completion_heartbeats must be a list")
    prior_heartbeat_session = str(computed_completion_session)
    prior_state_sha = str(value["completion_state_sha256"])
    for index, raw_heartbeat in enumerate(heartbeats):
        heartbeat_label = f"{label}.post_completion_heartbeats[{index}]"
        if not isinstance(raw_heartbeat, Mapping) or set(raw_heartbeat) != heartbeat_fields:
            raise StatisticsError(f"{heartbeat_label} must have exact heartbeat fields")
        heartbeat_session = raw_heartbeat["session"]
        if (
            not isinstance(heartbeat_session, str)
            or not heartbeat_session
            or heartbeat_session <= prior_heartbeat_session
        ):
            raise StatisticsError(f"{label}.post_completion_heartbeats must be strictly increasing")
        heartbeat_state = _parse_completion_economic_state(
            {field: raw_heartbeat[field] for field in completion_state}, heartbeat_label
        )
        heartbeat_economic_sha = _require_sha256(
            raw_heartbeat["economic_state_sha256"], f"{heartbeat_label}.economic_state_sha256"
        )
        if heartbeat_state != completion_state or heartbeat_economic_sha != completion_state_sha:
            raise StatisticsError(f"{heartbeat_label} changes economic state after completion")
        administrative_events = raw_heartbeat["administrative_events"]
        if not isinstance(administrative_events, list):
            raise StatisticsError(f"{heartbeat_label}.administrative_events must be a list")
        event_indices: list[int] = []
        for event_index, raw_administrative in enumerate(administrative_events):
            event_label = f"{heartbeat_label}.administrative_events[{event_index}]"
            if not isinstance(raw_administrative, Mapping) or set(raw_administrative) != {"event_index", "event"}:
                raise StatisticsError(f"{event_label} must bind one exact administrative event")
            bound_index = _nonnegative_int(raw_administrative["event_index"], f"{event_label}.event_index")
            event = raw_administrative["event"]
            if not isinstance(event, Mapping) or set(event) != administrative_event_fields:
                raise StatisticsError(f"{event_label}.event has inexact no-position fields")
            if (
                event["event_type"] != "corporate_action"
                or event["session"] != heartbeat_session
                or event["status"] != "no_position"
                or event["action_type"] not in {"share_change", "cash_dividend", "rights_issue", *TERMINAL_ACTION_TYPES}
                or event["cash_change_cny"] != 0
                or event["share_change"] != 0
                or event["deferred_tax_settled_cny"] != 0
            ):
                raise StatisticsError(f"{event_label}.event is not a zero-impact no-position action")
            event_indices.append(bound_index)
        if event_indices != sorted(set(event_indices)):
            raise StatisticsError(f"{heartbeat_label}.administrative_events are not unique and ordered")
        if raw_heartbeat["session_event_count"] != len(administrative_events):
            raise StatisticsError(f"{heartbeat_label}.session_event_count differs from administrative events")
        if raw_heartbeat["session_events_root_sha256"] != object_sha256(administrative_events):
            raise StatisticsError(f"{heartbeat_label}.session_events_root_sha256 differs from events")
        state_sha = _require_sha256(raw_heartbeat["state_sha256"], f"{heartbeat_label}.state_sha256")
        if (administrative_events and state_sha == prior_state_sha) or (
            not administrative_events and state_sha != prior_state_sha
        ):
            raise StatisticsError(f"{heartbeat_label}.state_sha256 does not reflect administrative evolution")
        prior_state_sha = state_sha
        prior_heartbeat_session = heartbeat_session

    sessions_to_complete = 0 if starts_complete else len(sessions)
    return UnwindSummary(starting, sessions_to_complete, fill_notional, costs)


def _parse_raw_path(
    path: Any,
    *,
    path_label: str,
    family_id: str,
    scenario_id: str,
    week_labels: tuple[str, ...],
    daily_labels: tuple[str, ...],
) -> ParsedRawPath:
    exact = {
        "weekly_returns",
        "daily_post_close_gross_exposure",
        "weekly_one_way_turnover",
        "weekly_new_entry_identity_sets",
        "weekly_gate_eligible_new_entry_opportunities",
        "weekly_pretrade_nav_cny",
        "weekly_explicit_costs_cny",
        "order_attempts",
        "state_gate_observations",
        "post_window_unwind",
    }
    if not isinstance(path, Mapping) or set(path) != exact:
        raise StatisticsError(f"{path_label} must have exact V2.1 path keys")
    opportunities_raw = path["weekly_gate_eligible_new_entry_opportunities"]
    if (
        not isinstance(opportunities_raw, list)
        or len(opportunities_raw) != EXPECTED_COMPLETE_WEEKS
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in opportunities_raw)
    ):
        raise StatisticsError(f"{path_label}.weekly_gate_eligible_new_entry_opportunities is invalid")
    opportunities = np.asarray(opportunities_raw, dtype=int)
    nav_raw = path["weekly_pretrade_nav_cny"]
    if not isinstance(nav_raw, list) or len(nav_raw) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{path_label}.weekly_pretrade_nav_cny must contain exactly 52 weeks")
    nav = np.asarray(
        [_finite(item, f"{path_label}.weekly_pretrade_nav_cny[{index}]") for index, item in enumerate(nav_raw)]
    )
    if np.any(nav <= 0):
        raise StatisticsError(f"{path_label}.weekly_pretrade_nav_cny must be positive")
    costs_raw = path["weekly_explicit_costs_cny"]
    if not isinstance(costs_raw, list) or len(costs_raw) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{path_label}.weekly_explicit_costs_cny must contain exactly 52 weeks")
    cost_values = {component: [] for component in EXPLICIT_COST_COMPONENTS}
    for index, raw in enumerate(costs_raw):
        if not isinstance(raw, Mapping) or set(raw) != set(EXPLICIT_COST_COMPONENTS):
            raise StatisticsError(f"{path_label}.weekly_explicit_costs_cny[{index}] has invalid fields")
        for component in EXPLICIT_COST_COMPONENTS:
            amount = _finite(raw[component], f"{path_label}.weekly_explicit_costs_cny[{index}].{component}")
            if amount < 0:
                raise StatisticsError(f"{path_label} explicit costs must be non-negative")
            cost_values[component].append(amount)
    if scenario_id == "gross" and any(any(amount != 0 for amount in values) for values in cost_values.values()):
        raise StatisticsError(f"{path_label} gross scenario must have zero explicit trading costs")
    weekly_cost_rates = {component: np.asarray(values, dtype=float) / nav for component, values in cost_values.items()}
    weekly_cost_rates["total"] = sum(weekly_cost_rates.values(), start=np.zeros(EXPECTED_COMPLETE_WEEKS))
    order_count, filled_order_count, requested, filled, capacity_usage, buy_index = _parse_order_attempts(
        path["order_attempts"], f"{path_label}.order_attempts", daily_labels
    )
    state_observations = _parse_state_observations(
        path["state_gate_observations"],
        label=f"{path_label}.state_gate_observations",
        family_id=family_id,
        week_labels=week_labels,
        daily_labels=daily_labels,
        gate_opportunities=opportunities,
        buy_index=buy_index,
    )
    unwind = _parse_unwind(path["post_window_unwind"], f"{path_label}.post_window_unwind", daily_labels[-1])
    return ParsedRawPath(
        _as_weekly_array(path["weekly_returns"], f"{path_label}.weekly_returns"),
        _as_exposure_array(
            path["daily_post_close_gross_exposure"],
            f"{path_label}.daily_post_close_gross_exposure",
            len(daily_labels),
        ),
        _as_turnover_array(path["weekly_one_way_turnover"], f"{path_label}.weekly_one_way_turnover"),
        _validate_selection_sets(
            path["weekly_new_entry_identity_sets"], f"{path_label}.weekly_new_entry_identity_sets"
        ),
        opportunities,
        weekly_cost_rates,
        order_count,
        filled_order_count,
        requested,
        filled,
        capacity_usage,
        state_observations,
        unwind,
    )


def _parse_arm_path(arm_id: str, value: Any) -> ArmPath:
    family_id, scenario_id = _split_risk_arm_id(arm_id)
    if not isinstance(value, Mapping):
        raise StatisticsError(f"arm_paths.{arm_id} must be an object")
    common = {"schema", "arm_id", "week_labels", "daily_labels", "aggregation", "paths"}
    if set(value) != common or value["schema"] != WEEKLY_PATH_SCHEMA or str(value["arm_id"]) != arm_id:
        raise StatisticsError(f"arm_paths.{arm_id} has an invalid V2.1 schema")
    labels = _validate_week_labels(value["week_labels"], f"arm_paths.{arm_id}.week_labels")
    daily_labels = _validate_daily_labels(value["daily_labels"], f"arm_paths.{arm_id}.daily_labels")
    aggregation = str(value["aggregation"])
    paths = value["paths"]
    if not isinstance(paths, Mapping):
        raise StatisticsError(f"arm_paths.{arm_id}.paths must be an object")
    seeded_family = family_id in EXPECTED_SEEDED_FAMILIES
    expected_aggregation = "equal_mean_all_20_frozen_seeds_before_statistics" if seeded_family else "single_path"
    if aggregation != expected_aggregation:
        raise StatisticsError(f"arm {arm_id} must use its frozen family aggregation")
    expected_keys = {str(seed) for seed in EXPECTED_RANDOM_SEEDS} if seeded_family else {"primary"}
    if set(paths) != expected_keys:
        qualifier = "exactly seeds 20260720..20260739" if seeded_family else "only paths.primary"
        raise StatisticsError(f"arm {arm_id} must contain {qualifier}")
    path_keys = [str(seed) for seed in EXPECTED_RANDOM_SEEDS] if seeded_family else ["primary"]
    parsed = [
        _parse_raw_path(
            paths[key],
            path_label=f"arm_paths.{arm_id}.paths.{key}",
            family_id=family_id,
            scenario_id=scenario_id,
            week_labels=labels,
            daily_labels=daily_labels,
        )
        for key in path_keys
    ]
    seed_returns = np.stack([item.returns for item in parsed], axis=0) if seeded_family else None
    seed_exposure = np.stack([item.exposure for item in parsed], axis=0) if seeded_family else None
    seed_turnover = np.stack([item.turnover for item in parsed], axis=0) if seeded_family else None
    seed_selections = tuple(item.selections for item in parsed) if seeded_family else None
    seed_gate_opportunities = np.stack([item.gate_opportunities for item in parsed], axis=0) if seeded_family else None
    first = parsed[0]
    returns = seed_returns.mean(axis=0) if seed_returns is not None else first.returns
    exposure = seed_exposure.mean(axis=0) if seed_exposure is not None else first.exposure
    turnover = seed_turnover.mean(axis=0) if seed_turnover is not None else first.turnover
    selections = (
        tuple(frozenset().union(*(seed[week] for seed in seed_selections)) for week in range(EXPECTED_COMPLETE_WEEKS))
        if seed_selections is not None
        else first.selections
    )
    opportunities = (
        seed_gate_opportunities.mean(axis=0) if seed_gate_opportunities is not None else first.gate_opportunities
    )
    weekly_cost_rates = {
        component: np.stack([item.weekly_cost_rates[component] for item in parsed], axis=0).mean(axis=0)
        for component in (*EXPLICIT_COST_COMPONENTS, "total")
    }
    capacity_arrays = [item.capacity_usage for item in parsed if item.capacity_usage.size]
    return ArmPath(
        arm_id=arm_id,
        family_id=family_id,
        scenario_id=scenario_id,
        week_labels=labels,
        daily_labels=daily_labels,
        returns=returns,
        exposure=exposure,
        turnover=turnover,
        selections=selections,
        gate_opportunities=opportunities,
        weekly_cost_rates=weekly_cost_rates,
        order_count=sum(item.order_count for item in parsed),
        positive_fill_order_count=sum(item.positive_fill_order_count for item in parsed),
        requested_shares=sum(item.requested_shares for item in parsed),
        filled_shares=sum(item.filled_shares for item in parsed),
        capacity_usage=np.concatenate(capacity_arrays) if capacity_arrays else np.asarray([], dtype=float),
        state_observations=tuple(observation for item in parsed for observation in item.state_observations),
        unwind_summaries=tuple(item.unwind for item in parsed),
        seed_returns=seed_returns,
        seed_exposure=seed_exposure,
        seed_turnover=seed_turnover,
        seed_selections=seed_selections,
        seed_gate_opportunities=seed_gate_opportunities,
        seed_ids=EXPECTED_RANDOM_SEEDS if seeded_family else (),
    )


def newey_west_mean_interval(
    values: Sequence[float], *, lag: int = 4, one_sided_confidence: float = 0.95
) -> dict[str, float]:
    """Return a PSD Newey-West interval for a weekly mean.

    Bartlett weights yield a positive-semidefinite estimator.  A materially
    non-positive long-run variance is an invalid inference condition; it is not
    silently converted into a zero t statistic.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) <= lag + 1 or not np.isfinite(x).all():
        raise DegenerateInferenceError("HAC input is too short or non-finite")
    if lag < 0 or lag >= len(x):
        raise DegenerateInferenceError("HAC lag is outside the valid range")
    mean = float(x.mean())
    centered = x - mean
    n = len(x)
    gamma0 = float(centered @ centered / n)
    lrv = gamma0
    for offset in range(1, lag + 1):
        gamma = float(centered[offset:] @ centered[:-offset] / n)
        lrv += 2.0 * (1.0 - offset / (lag + 1.0)) * gamma
    scale = max(gamma0, np.finfo(float).tiny)
    if lrv < -1e-12 * scale:
        raise DegenerateInferenceError("Newey-West long-run variance is materially negative")
    if lrv <= np.finfo(float).eps * max(scale, 1.0):
        raise DegenerateInferenceError("Newey-West long-run variance is non-positive")
    standard_error = math.sqrt(lrv / n)
    if not 0.5 < one_sided_confidence < 1.0:
        raise DegenerateInferenceError("one-sided confidence must be in (0.5, 1)")
    z = NormalDist().inv_cdf(one_sided_confidence)
    return {
        "weekly_mean": mean,
        "long_run_variance": lrv,
        "weekly_standard_error": standard_error,
        "z_score": mean / standard_error,
        "annualized_mean": mean * 52.0,
        "annualized_lower": (mean - z * standard_error) * 52.0,
        "annualized_upper": (mean + z * standard_error) * 52.0,
        "one_sided_confidence": one_sided_confidence,
        "lag": lag,
    }


def _derived_bootstrap_seed(
    *, method_seed: int, identity: str, window_sha256: str, block_length: int, draws: int
) -> int:
    material = {
        "domain": "xs_chan_v2_1_circular_block_bootstrap",
        "method_seed": int(method_seed),
        "identity": identity,
        "window_sha256": window_sha256,
        "block_length": int(block_length),
        "draws": int(draws),
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(material)).digest()[:8], "big", signed=False)


def circular_block_bootstrap_interval(
    values: Sequence[float],
    *,
    identity: str,
    window_sha256: str,
    method_seed: int = 20260720,
    block_length: int = 4,
    draws: int = 20_000,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, float | int]:
    """Circular block bootstrap with domain-separated deterministic RNG."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) != EXPECTED_COMPLETE_WEEKS or not np.isfinite(x).all():
        raise StatisticsError("bootstrap requires exactly 52 finite weekly values")
    if not isinstance(block_length, int) or block_length <= 0 or block_length > len(x):
        raise StatisticsError("bootstrap block_length is invalid")
    if not isinstance(draws, int) or draws < 1_000:
        raise StatisticsError("bootstrap draws must be at least 1000")
    if not 0 < lower_quantile < upper_quantile < 1:
        raise StatisticsError("bootstrap quantiles are invalid")
    _require_sha256(window_sha256, "window_sha256")
    derived_seed = _derived_bootstrap_seed(
        method_seed=method_seed,
        identity=identity,
        window_sha256=window_sha256,
        block_length=block_length,
        draws=draws,
    )
    rng = np.random.default_rng(derived_seed)
    blocks_needed = math.ceil(len(x) / block_length)
    offsets = np.arange(block_length, dtype=int)
    means = np.empty(draws, dtype=float)
    # Chunking keeps the formal 20k draw run fast without a large transient cube.
    chunk = 1_000
    for first in range(0, draws, chunk):
        size = min(chunk, draws - first)
        starts = rng.integers(0, len(x), size=(size, blocks_needed))
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(x)
        samples = x[indices.reshape(size, -1)[:, : len(x)]]
        means[first : first + size] = samples.mean(axis=1) * 52.0
    lower, upper = np.quantile(means, [lower_quantile, upper_quantile], method="linear")
    # Quantile Monte-Carlo error is estimated from fixed contiguous batches.  It
    # is diagnostic and can be bounded by the protocol without a second RNG.
    batch_count = min(20, draws // 250)
    batch_size = draws // batch_count
    trimmed = means[: batch_count * batch_size].reshape(batch_count, batch_size)
    batch_quantiles = np.quantile(trimmed, [lower_quantile, upper_quantile], axis=1, method="linear")
    lower_mcse = float(np.std(batch_quantiles[0], ddof=1) / math.sqrt(batch_count))
    upper_mcse = float(np.std(batch_quantiles[1], ddof=1) / math.sqrt(batch_count))
    return {
        "annualized_lower": float(lower),
        "annualized_upper": float(upper),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "block_length": block_length,
        "draws": draws,
        "derived_seed": derived_seed,
        "lower_quantile_mcse": lower_mcse,
        "upper_quantile_mcse": upper_mcse,
    }


def top_positive_concentration(values: Sequence[float], count: int = 10) -> float:
    positive = np.sort(np.asarray(values, dtype=float)[np.asarray(values, dtype=float) > 0])[::-1]
    if len(positive) == 0:
        return 0.0
    return float(positive[:count].sum() / positive.sum())


def _active_risk_and_power(values: np.ndarray, sesoi: float) -> dict[str, Any]:
    sample_std = float(np.std(values, ddof=1))
    ratio: float | None = float(math.sqrt(52.0) * np.mean(values) / sample_std) if sample_std > 0 else None
    active_nav = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    active_drawdown = active_nav / np.maximum.accumulate(active_nav) - 1.0
    z_sum = 1.6448536269514722 + 0.8416212335729143
    mde_excess = z_sum * sample_std * math.sqrt(52.0)
    effect_grid = [sesoi, sesoi + mde_excess / 2.0, sesoi + mde_excess, sesoi + 2.0 * mde_excess]
    power_curve: list[dict[str, float | None]] = []
    for annualized_effect in effect_grid:
        if sample_std <= 0:
            power = None
        else:
            noncentral = (annualized_effect - sesoi) / (sample_std * math.sqrt(52.0))
            power = NormalDist().cdf(noncentral - 1.6448536269514722)
        power_curve.append({"annualized_effect": annualized_effect, "iid_planning_power": power})
    return {
        "active_information_ratio": ratio,
        "active_path_maximum_drawdown_including_initial_nav": float(np.min(active_drawdown)),
        "observed_weekly_active_sample_std": sample_std,
        "iid_mde_annualized_excess_over_sesoi_at_80pct_power": mde_excess,
        "iid_minimum_effect_for_80pct_power": sesoi + mde_excess,
        "power_curve": power_curve,
        "dependence_warning": "HAC/block dependence can reduce effective power; curve is IID planning only",
    }


def _distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise StatisticsError("risk distribution input must be one-dimensional and finite")
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "sample_std": None,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "maximum": float(np.max(array)),
    }


def _state_diagnostics(path: ArmPath) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for regime in range(11):
        observations = [item for item in path.state_observations if item.regime == regime]
        filled = [item for item in observations if item.filled_shares > 0]
        requested_shares = sum(item.requested_shares for item in observations)
        filled_shares = sum(item.filled_shares for item in observations)
        result[str(regime)] = {
            "observation_count": len(observations),
            "eligible_new_entry_count": sum(item.eligible for item in observations),
            "allowed_new_entry_count": sum(item.allowed for item in observations),
            "fill_count": len(filled),
            "requested_shares": requested_shares,
            "filled_shares": filled_shares,
            "share_weighted_fill_ratio": None if requested_shares == 0 else filled_shares / requested_shares,
            "mean_entry_weight": None
            if not filled
            else float(np.mean([item.entry_weight for item in filled if item.entry_weight is not None])),
            "holding_session_distribution": _distribution(
                [float(item.holding_sessions) for item in filled if item.holding_sessions is not None]
            ),
            "forward_return_distribution": _distribution(
                [item.forward_return for item in filled if item.forward_return is not None]
            ),
        }
    return result


def _unwind_diagnostics(path: ArmPath) -> dict[str, Any]:
    costs = {
        component: _distribution([item.costs_cny[component] for item in path.unwind_summaries])
        for component in EXPLICIT_COST_COMPONENTS
    }
    total_costs = [sum(item.costs_cny.values()) for item in path.unwind_summaries]
    return {
        "path_count": len(path.unwind_summaries),
        "starting_shares_distribution": _distribution([float(item.starting_shares) for item in path.unwind_summaries]),
        "official_sessions_to_complete_distribution": _distribution(
            [float(item.official_sessions_to_complete) for item in path.unwind_summaries]
        ),
        "fill_notional_cny_distribution": _distribution([item.fill_notional_cny for item in path.unwind_summaries]),
        "cost_cny_distributions": {**costs, "total": _distribution(total_costs)},
    }


def _arm_risk_report(path: ArmPath) -> dict[str, Any]:
    explicit_cost_rates = {
        component: {
            "weekly_distribution": _distribution(path.weekly_cost_rates[component]),
            "annualized_arithmetic_rate": float(np.mean(path.weekly_cost_rates[component]) * 52.0),
        }
        for component in (*EXPLICIT_COST_COMPONENTS, "total")
    }
    return {
        "family_id": path.family_id,
        "scenario_id": path.scenario_id,
        "seed_path_count": len(path.seed_ids) if path.seeded else 1,
        "daily_exposure_distribution": _distribution(path.exposure),
        "weekly_turnover_distribution": _distribution(path.turnover),
        "fill_ratio": {
            "attempted_order_count": path.order_count,
            "positive_fill_order_count": path.positive_fill_order_count,
            "order_fill_ratio": None if path.order_count == 0 else path.positive_fill_order_count / path.order_count,
            "requested_shares": path.requested_shares,
            "filled_shares": path.filled_shares,
            "share_weighted_fill_ratio": None
            if path.requested_shares == 0
            else path.filled_shares / path.requested_shares,
        },
        "capacity_usage_distribution": _distribution(path.capacity_usage),
        "weekly_explicit_cost_rate": explicit_cost_rates,
        "state_specific_gate_and_fill": _state_diagnostics(path),
        "post_window_unwind": _unwind_diagnostics(path),
    }


def _cost_attribution(paths: Mapping[str, ArmPath]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for family in EXPECTED_RISK_FAMILIES:
        gross = paths[f"{family}_gross"]
        one_x = paths[f"{family}_1x"]
        two_x = paths[f"{family}_2x"]
        gross_to_one = gross.returns - one_x.returns
        one_to_two = one_x.returns - two_x.returns
        gross_to_two = gross.returns - two_x.returns
        report[family] = {
            "arm_ids": {
                "gross": gross.arm_id,
                "1x": one_x.arm_id,
                "2x": two_x.arm_id,
            },
            "annualized_arithmetic_return": {
                "gross": float(np.mean(gross.returns) * 52.0),
                "1x": float(np.mean(one_x.returns) * 52.0),
                "2x": float(np.mean(two_x.returns) * 52.0),
            },
            "gross_to_1x_weekly_return_drag_distribution": _distribution(gross_to_one),
            "1x_to_2x_weekly_return_drag_distribution": _distribution(one_to_two),
            "gross_to_2x_weekly_return_drag_distribution": _distribution(gross_to_two),
            "annualized_arithmetic_return_drag": {
                "gross_to_1x": float(np.mean(gross_to_one) * 52.0),
                "1x_to_2x": float(np.mean(one_to_two) * 52.0),
                "gross_to_2x": float(np.mean(gross_to_two) * 52.0),
            },
            "annualized_explicit_cost_rate_by_scenario": {
                scenario: {
                    component: float(np.mean(paths[f"{family}_{scenario}"].weekly_cost_rates[component]) * 52.0)
                    for component in (*EXPLICIT_COST_COMPONENTS, "total")
                }
                for scenario in ("gross", "1x", "2x")
            },
        }
    return report


def _risk_and_mechanism_report(paths: Mapping[str, ArmPath]) -> dict[str, Any]:
    report = {
        "schema": RISK_MECHANISM_REPORT_SCHEMA,
        "methodology": {
            "daily_exposure": "distribution_of_execution_replayed_post_close_gross_exposure",
            "weekly_turnover": "sum_absolute_filled_notional_divided_by_two_times_pretrade_nav",
            "fill_ratio": "filled_shares_divided_by_requested_shares_with_zero_request_reported_as_null",
            "capacity_usage": "fill_notional_divided_by_minimum_of_0_05_adv20_and_0_10_open_auction_turnover",
            "cost_attribution": "arithmetic_weekly_return_differences_and_direct_costs_divided_by_same_scenario_pretrade_nav",
            "holding_sessions": "inclusive_official_sessions_from_entry_through_exit_or_confirmation_window_mark",
            "forward_return": "exit_or_confirmation_window_raw_mark_divided_by_entry_fill_price_minus_one",
            "post_window_unwind_days": "official_sessions_after_confirmation_end_until_zero_remaining_shares",
            "risk_metrics_do_not_replace_registered_comparisons": True,
            "net_only_improvement_without_gross_identity_pass": "cost_management_only_not_predictive_alpha",
        },
        "arm_order": list(EXPECTED_RISK_ARM_IDS),
        "arms": {arm_id: _arm_risk_report(paths[arm_id]) for arm_id in EXPECTED_RISK_ARM_IDS},
        "gross_to_1x_to_2x_cost_attribution": _cost_attribution(paths),
    }
    report["report_sha256"] = object_sha256(report)
    return report


def _random_control_diagnostics(path: ArmPath) -> dict[str, float | int | bool]:
    if not path.seeded or path.seed_returns is None:
        return {
            "seed_count": 0,
            "annualized_seed_mean_mcse": 0.0,
            "maximum_weekly_seed_mean_mcse": 0.0,
        }
    weekly_mcse = path.seed_returns.std(axis=0, ddof=1) / math.sqrt(path.seed_returns.shape[0])
    return {
        "seed_count": int(path.seed_returns.shape[0]),
        "annualized_seed_mean_mcse": float(math.sqrt(float(np.sum(np.square(weekly_mcse))))),
        "maximum_weekly_seed_mean_mcse": float(weekly_mcse.max()),
    }


def _intervention_counts(left: ArmPath, right: ArmPath) -> dict[str, float | int]:
    disagreement_weeks = 0
    assignments = 0.0
    if right.seed_selections is None:
        for week in range(EXPECTED_COMPLETE_WEEKS):
            difference = left.selections[week].symmetric_difference(right.selections[week])
            assignments += len(difference) / 2.0
            disagreement_weeks += int(bool(difference))
    else:
        for week in range(EXPECTED_COMPLETE_WEEKS):
            per_seed_difference: list[float] = []
            for seed in right.seed_selections:
                per_seed_difference.append(len(left.selections[week].symmetric_difference(seed[week])) / 2.0)
            assignments += float(np.mean(per_seed_difference))
            disagreement_weeks += int(any(value > 0 for value in per_seed_difference))
    return {
        "identity_disagreement_weeks": disagreement_weeks,
        "differing_symbol_assignments": assignments,
        "gate_eligible_new_entry_opportunities": int(np.sum(left.gate_opportunities)),
    }


def _validate_comparison_registry(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or tuple(value.keys()) != EXPECTED_COMPARISONS:
        raise StatisticsError("comparison_registry must preserve the exact seven preregistered comparison order")
    required = {
        "role",
        "claim",
        "arm",
        "comparator",
        "cost_basis",
        "decision_gating",
        "effect_metric",
        "sesoi_annualized",
        "minimum_identity_disagreement_weeks",
        "minimum_differing_symbol_assignments",
        "seed_mcse_required",
        "pass_rule",
        "falsified_rule",
        "otherwise_status",
    }
    result: dict[str, dict[str, Any]] = {}
    for identity in EXPECTED_COMPARISONS:
        spec = value[identity]
        if not isinstance(spec, Mapping) or set(spec) != required:
            raise StatisticsError(f"comparison_registry.{identity} must have exact V2.1 fields")
        if spec["effect_metric"] != "annualized_mean_weekly_active_return":
            raise StatisticsError(f"comparison_registry.{identity} has an unregistered effect metric")
        if spec["otherwise_status"] != "INCONCLUSIVE":
            raise StatisticsError(f"comparison_registry.{identity} must use INCONCLUSIVE fallback")
        if type(spec["decision_gating"]) is not bool:
            raise StatisticsError(f"comparison_registry.{identity}.decision_gating must be boolean")
        if type(spec["seed_mcse_required"]) is not bool:
            raise StatisticsError(f"comparison_registry.{identity}.seed_mcse_required must be boolean")
        sesoi = _finite(spec["sesoi_annualized"], f"comparison_registry.{identity}.sesoi_annualized")
        if sesoi <= 0:
            raise StatisticsError(f"comparison_registry.{identity}.sesoi_annualized must be positive")
        for key in ("minimum_identity_disagreement_weeks", "minimum_differing_symbol_assignments"):
            if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] <= 0:
                raise StatisticsError(f"comparison_registry.{identity}.{key} must be positive integer")
        if not str(spec["arm"]) or not str(spec["comparator"]):
            raise StatisticsError(f"comparison_registry.{identity} arm/comparator must be non-empty")
        actual_core = (
            str(spec["role"]),
            str(spec["arm"]),
            str(spec["comparator"]),
            str(spec["cost_basis"]),
            bool(spec["decision_gating"]),
            float(spec["sesoi_annualized"]),
            bool(spec["seed_mcse_required"]),
        )
        if actual_core != EXPECTED_COMPARISON_CORE[identity]:
            raise StatisticsError(f"comparison_registry.{identity} differs from the frozen comparison core")
        if spec["pass_rule"] != EXPECTED_PASS_RULE or spec["falsified_rule"] != EXPECTED_FALSIFIED_RULE:
            raise StatisticsError(f"comparison_registry.{identity} changes a frozen decision rule")
        if spec["minimum_identity_disagreement_weeks"] != 26 or spec["minimum_differing_symbol_assignments"] != 260:
            raise StatisticsError(f"comparison_registry.{identity} changes frozen intervention thresholds")
        result[identity] = dict(spec)
    if [identity for identity in EXPECTED_COMPARISONS if result[identity]["decision_gating"]] != list(
        EXPECTED_COMPARISONS[:5]
    ):
        raise StatisticsError("exactly the first five comparisons must be decision-gating")
    return result


def _comparison_status(
    *,
    spec: Mapping[str, Any],
    hac: Mapping[str, float],
    bootstrap: Mapping[str, float | int],
    concentration: float,
    weekly_median: float,
    maximum_top10_share: float,
    interventions: Mapping[str, int],
    minimum_gate_opportunities: int,
) -> tuple[str, list[str], bool]:
    support_failures: list[str] = []
    if interventions["identity_disagreement_weeks"] < int(spec["minimum_identity_disagreement_weeks"]):
        support_failures.append("minimum_identity_disagreement_weeks")
    if interventions["differing_symbol_assignments"] < int(spec["minimum_differing_symbol_assignments"]):
        support_failures.append("minimum_differing_symbol_assignments")
    if interventions["gate_eligible_new_entry_opportunities"] < minimum_gate_opportunities:
        support_failures.append("minimum_gate_eligible_new_entry_opportunities")
    if support_failures:
        return "INCONCLUSIVE", support_failures, False
    sesoi = float(spec["sesoi_annualized"])
    lower = min(float(hac["annualized_lower"]), float(bootstrap["annualized_lower"]))
    upper = max(float(hac["annualized_upper"]), float(bootstrap["annualized_upper"]))
    if lower > sesoi and concentration <= maximum_top10_share and weekly_median > 0:
        return "PASS", [], True
    if upper < sesoi:
        return "FALSIFIED", [], True
    failures = []
    if lower <= sesoi:
        failures.append("lower_bound_not_above_sesoi")
    if concentration > maximum_top10_share:
        failures.append("positive_week_concentration")
    if weekly_median <= 0:
        failures.append("weekly_active_median_not_positive")
    return "INCONCLUSIVE", failures, True


def evaluate_statistics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all seven comparisons and the preregistered overall status."""

    exact = {
        "schema",
        "protocol_sha256",
        "trial_id",
        "window_sha256",
        "comparison_registry",
        "arm_paths",
        "controls",
        "bootstrap",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise StatisticsError("statistics input must have exact V2.1 top-level keys")
    if value["schema"] != STATISTICS_INPUT_SCHEMA:
        raise StatisticsError(f"statistics schema must be {STATISTICS_INPUT_SCHEMA!r}")
    _require_sha256(value["protocol_sha256"], "protocol_sha256")
    _require_sha256(value["window_sha256"], "window_sha256")
    if not str(value["trial_id"]):
        raise StatisticsError("trial_id must be non-empty")
    registry = _validate_comparison_registry(value["comparison_registry"])
    if not isinstance(value["arm_paths"], Mapping):
        raise StatisticsError("arm_paths must be an object")
    needed_arms = {str(spec[key]) for spec in registry.values() for key in ("arm", "comparator")}
    if not needed_arms.issubset(EXPECTED_RISK_ARM_IDS):
        raise StatisticsError("comparison registry references an arm outside the frozen risk domain")
    if set(value["arm_paths"]) != set(EXPECTED_RISK_ARM_IDS):
        raise StatisticsError("arm_paths must equal the exact 24-arm risk and mechanism reporting domain")
    paths = {arm: _parse_arm_path(arm, value["arm_paths"][arm]) for arm in EXPECTED_RISK_ARM_IDS}
    labels = {path.week_labels for path in paths.values()}
    if len(labels) != 1:
        raise StatisticsError("all arm paths must use the identical frozen 52-week labels")
    daily_labels = {path.daily_labels for path in paths.values()}
    if len(daily_labels) != 1:
        raise StatisticsError("all arm paths must use identical daily exposure labels")

    controls_expected = {
        "maximum_top10_positive_weeks_share",
        "maximum_mean_absolute_exposure_gap",
        "maximum_p95_absolute_exposure_gap",
        "maximum_absolute_exposure_gap",
        "maximum_mean_absolute_turnover_gap",
        "maximum_p95_absolute_turnover_gap",
        "maximum_random_annualized_seed_mean_mcse",
        "maximum_bootstrap_quantile_mcse_annualized",
        "minimum_gate_eligible_new_entry_opportunities",
    }
    controls = value["controls"]
    if not isinstance(controls, Mapping) or set(controls) != controls_expected:
        raise StatisticsError("controls must have exact V2.1 fields")
    thresholds = {key: _finite(controls[key], f"controls.{key}") for key in controls_expected}
    if any(number < 0 for number in thresholds.values()):
        raise StatisticsError("control thresholds must be non-negative")
    if thresholds["maximum_top10_positive_weeks_share"] > 1:
        raise StatisticsError("maximum_top10_positive_weeks_share must be <= 1")
    if thresholds != EXPECTED_CONTROL_THRESHOLDS:
        raise StatisticsError("control thresholds differ from the frozen V2.1 protocol")

    bootstrap_expected = {"method_seed", "block_length_weeks", "draws", "lower_quantile", "upper_quantile"}
    bootstrap_config = value["bootstrap"]
    if not isinstance(bootstrap_config, Mapping) or set(bootstrap_config) != bootstrap_expected:
        raise StatisticsError("bootstrap must have exact V2.1 fields")
    method_seed = int(bootstrap_config["method_seed"])
    block_length = int(bootstrap_config["block_length_weeks"])
    draws = int(bootstrap_config["draws"])
    lower_quantile = _finite(bootstrap_config["lower_quantile"], "bootstrap.lower_quantile")
    upper_quantile = _finite(bootstrap_config["upper_quantile"], "bootstrap.upper_quantile")
    if (method_seed, block_length, draws, lower_quantile, upper_quantile) != (20260720, 4, 20_000, 0.05, 0.95):
        raise StatisticsError("bootstrap configuration differs from the frozen V2.1 protocol")

    random_diagnostics = {arm: _random_control_diagnostics(path) for arm, path in paths.items() if path.seeded}
    random_failures: list[str] = []
    for arm, diagnostic in random_diagnostics.items():
        if diagnostic["seed_count"] != len(EXPECTED_RANDOM_SEEDS):
            random_failures.append(f"{arm}.seed_count")
        if diagnostic["annualized_seed_mean_mcse"] > thresholds["maximum_random_annualized_seed_mean_mcse"]:
            random_failures.append(f"{arm}.annualized_seed_mean_mcse")

    comparisons: dict[str, dict[str, Any]] = {}
    global_control_failures = list(random_failures)
    for identity in EXPECTED_COMPARISONS:
        spec = registry[identity]
        left = paths[str(spec["arm"])]
        right = paths[str(spec["comparator"])]
        active = left.returns - right.returns
        exposure_gaps = np.abs(left.exposure - right.exposure)
        exposure_gap = float(np.mean(exposure_gaps))
        exposure_p95 = float(np.quantile(exposure_gaps, 0.95, method="linear"))
        exposure_max = float(np.max(exposure_gaps))
        turnover_gaps = np.abs(left.turnover - right.turnover)
        turnover_gap = float(np.mean(turnover_gaps))
        turnover_p95 = float(np.quantile(turnover_gaps, 0.95, method="linear"))
        interventions = _intervention_counts(left, right)
        comparison_control_failures: list[str] = []
        if bool(spec["decision_gating"]):
            if exposure_gap > thresholds["maximum_mean_absolute_exposure_gap"]:
                comparison_control_failures.append("mean_absolute_exposure_gap")
            if exposure_p95 > thresholds["maximum_p95_absolute_exposure_gap"]:
                comparison_control_failures.append("p95_absolute_exposure_gap")
            if exposure_max > thresholds["maximum_absolute_exposure_gap"]:
                comparison_control_failures.append("maximum_absolute_exposure_gap")
        matched_turnover = identity in {
            "F_2x_minus_R_match_2x",
            "FC_gross_minus_FGR_gross",
            "FC_2x_minus_FGR_2x",
            "FMA_gross_minus_FMGR_gross",
            "FMA_2x_minus_FMGR_2x",
        }
        if matched_turnover:
            if turnover_gap > thresholds["maximum_mean_absolute_turnover_gap"]:
                comparison_control_failures.append("mean_absolute_turnover_gap")
            if turnover_p95 > thresholds["maximum_p95_absolute_turnover_gap"]:
                comparison_control_failures.append("p95_absolute_turnover_gap")
        try:
            hac = newey_west_mean_interval(active, lag=4, one_sided_confidence=0.95)
            bootstrap = circular_block_bootstrap_interval(
                active,
                identity=identity,
                window_sha256=str(value["window_sha256"]),
                method_seed=method_seed,
                block_length=block_length,
                draws=draws,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            if (
                max(float(bootstrap["lower_quantile_mcse"]), float(bootstrap["upper_quantile_mcse"]))
                > thresholds["maximum_bootstrap_quantile_mcse_annualized"]
            ):
                comparison_control_failures.append("bootstrap_quantile_mcse")
            concentration = top_positive_concentration(active)
            weekly_median = float(np.median(active))
            status, status_reasons, intervention_sufficient = _comparison_status(
                spec=spec,
                hac=hac,
                bootstrap=bootstrap,
                concentration=concentration,
                weekly_median=weekly_median,
                maximum_top10_share=thresholds["maximum_top10_positive_weeks_share"],
                interventions=interventions,
                minimum_gate_opportunities=int(thresholds["minimum_gate_eligible_new_entry_opportunities"]),
            )
            inference_error = None
        except DegenerateInferenceError as exc:
            hac = None
            bootstrap = None
            concentration = top_positive_concentration(active)
            status = "INCONCLUSIVE"
            status_reasons = ["degenerate_inference"]
            intervention_sufficient = (
                interventions["identity_disagreement_weeks"] >= int(spec["minimum_identity_disagreement_weeks"])
                and interventions["differing_symbol_assignments"] >= int(spec["minimum_differing_symbol_assignments"])
                and interventions["gate_eligible_new_entry_opportunities"]
                >= int(thresholds["minimum_gate_eligible_new_entry_opportunities"])
            )
            inference_error = str(exc)
        if comparison_control_failures:
            global_control_failures.extend(f"{identity}.{reason}" for reason in comparison_control_failures)
        comparisons[identity] = {
            "role": spec["role"],
            "decision_gating": bool(spec["decision_gating"]),
            "arm": spec["arm"],
            "comparator": spec["comparator"],
            "sesoi_annualized": float(spec["sesoi_annualized"]),
            "annualized_active_arithmetic": float(active.mean() * 52.0),
            "risk_and_power": _active_risk_and_power(active, float(spec["sesoi_annualized"])),
            "weekly_active_median": float(np.median(active)),
            "top10_positive_weeks_share": concentration,
            "hac": hac,
            "block_bootstrap": bootstrap,
            "mean_absolute_post_close_exposure_gap": exposure_gap,
            "p95_absolute_post_close_exposure_gap": exposure_p95,
            "maximum_absolute_post_close_exposure_gap": exposure_max,
            "mean_absolute_one_way_turnover_gap": turnover_gap,
            "p95_absolute_one_way_turnover_gap": turnover_p95,
            "turnover_control_applies": matched_turnover,
            "interventions": interventions,
            "intervention_sufficient": intervention_sufficient,
            "control_failures": comparison_control_failures,
            "status": status,
            "status_reasons": status_reasons,
            "inference_error": inference_error,
        }

    if global_control_failures:
        overall_status = "INVALID_CONTROL"
    else:
        gating = [comparisons[identity] for identity in EXPECTED_COMPARISONS[:5]]
        factor_status = gating[0]["status"]
        timing_statuses = [item["status"] for item in gating[1:]]
        if any(not comparisons[identity]["intervention_sufficient"] for identity in EXPECTED_COMPARISONS[:5]):
            overall_status = "INSUFFICIENT_INTERVENTION"
        elif factor_status == "FALSIFIED":
            overall_status = "FALSIFIED_FACTOR"
        elif factor_status != "PASS":
            overall_status = "INCONCLUSIVE_FACTOR"
        elif "FALSIFIED" in timing_statuses:
            overall_status = "FACTOR_PASSED_TIMING_FALSIFIED"
        elif any(status != "PASS" for status in timing_statuses):
            overall_status = "FACTOR_PASSED_TIMING_INCONCLUSIVE"
        else:
            overall_status = "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY"

    result = {
        "schema": STATISTICS_RESULT_SCHEMA,
        "protocol_sha256": value["protocol_sha256"],
        "trial_id": str(value["trial_id"]),
        "window_sha256": value["window_sha256"],
        "input_sha256": object_sha256(value),
        "complete_weeks": EXPECTED_COMPLETE_WEEKS,
        "comparison_order": list(EXPECTED_COMPARISONS),
        "comparisons": comparisons,
        "risk_and_mechanism_report": _risk_and_mechanism_report(paths),
        "random_control_diagnostics": random_diagnostics,
        "control_failures": sorted(set(global_control_failures)),
        "overall_status": overall_status,
        "alpha_validated": False,
        "live_trading_allowed": False,
    }
    result["result_sha256"] = object_sha256(result)
    return result


def verify_statistics_result(value: Mapping[str, Any], expected_input: Mapping[str, Any]) -> dict[str, Any]:
    replayed = evaluate_statistics(expected_input)
    if canonical_json_bytes(value) != canonical_json_bytes(replayed):
        raise StatisticsError("statistics result differs from deterministic semantic replay")
    return replayed


__all__ = [
    "EXPECTED_COMPARISONS",
    "EXPECTED_COMPLETE_WEEKS",
    "EXPECTED_RANDOM_SEEDS",
    "EXPECTED_RISK_ARM_IDS",
    "EXPECTED_RISK_FAMILIES",
    "EXPECTED_RISK_SCENARIOS",
    "RISK_MECHANISM_REPORT_SCHEMA",
    "STATISTICS_INPUT_SCHEMA",
    "STATISTICS_RESULT_SCHEMA",
    "StatisticsError",
    "DegenerateInferenceError",
    "WEEKLY_PATH_SCHEMA",
    "canonical_json_bytes",
    "circular_block_bootstrap_interval",
    "evaluate_statistics",
    "newey_west_mean_interval",
    "object_sha256",
    "top_positive_concentration",
    "verify_statistics_result",
]
