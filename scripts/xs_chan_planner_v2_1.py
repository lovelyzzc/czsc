"""Causal, fail-closed decision planner for the preregistered XS/Chan V2.1 trial.

The planner is deliberately separate from the frozen V2 research script.  A
formal call consumes :class:`xs_chan_data_v2_1.VerifiedDataSnapshot`; the
``Mapping[str, DataFrame]`` entry point exists for deterministic unit tests and
is always labelled non-authoritative.  No caller supplied ``passed`` or
``ready`` flag is accepted.

For one completed weekly decision the module recomputes, from source rows up to
that decision only:

* raw-close times the *same-day* adjustment factor, dense-session momentum,
  volatility, ADV and SMA20;
* PIT universe, name/ST, industry and free-float-market-cap eligibility;
* the frozen 50/75 buffered factor target and joint industry/mcap/ADV controls;
* Chan, SMA20 and exact retained/new-quota random gates for all 20 seeds; and
* actual-reference-book slot occupancy, including blocked exits.

The output contains every ordered identity only once and binds gross, 1x, 2x
and capacity replays to that same identity.  ``verify_planning_artifact``
recomputes the complete artifact from its recorded inputs and verified source
frames; merely preserving a JSON hash is therefore insufficient.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR / "xs_chan_protocol_v2_1.json"
ARTIFACT_SCHEMA = "xs_chan_planning_artifact_v2_1"
PLANNER_INPUT_SCHEMA = "xs_chan_planner_input_v2_1"
VERIFICATION_SCHEMA = "xs_chan_planning_verification_v2_1"
FRAME_AUTHORITY = "explicit_frame_mapping_non_authoritative"
SNAPSHOT_AUTHORITY = "verified_data_snapshot"
STATE_COLUMNS = ("symbol", "dt", "regime")
SCENARIOS = ("gross", "1x", "2x", "capacity")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FRAMES = (
    "calendar",
    "security_master",
    "namechange",
    "raw_daily",
    "adj_factor",
    "daily_basic",
    "industry_membership",
    "official_daily_status",
    "chan_states",
)


class PlanningError(ValueError):
    """The decision cannot be frozen without violating the V2.1 protocol."""


class ExactControlError(PlanningError):
    """A preregistered random control cannot be matched exactly."""


@dataclass(frozen=True)
class ReferenceBook:
    """Actual 10m/1x reference-book facts known at the decision close.

    Holdings and blocked exits come only from the arm's reference scenario.
    Other scenarios may have different NAV paths, but must replay the identity
    derived from this book.  Each NAV is bound to its independently recorded
    close-valuation record.
    """

    actual_holdings: tuple[str, ...]
    blocked_exit_symbols: tuple[str, ...]
    sizing_nav_cny_by_scenario: Mapping[str, float]
    sizing_nav_record_sha256_by_scenario: Mapping[str, str]

    def __post_init__(self) -> None:
        holdings = tuple(map(str, self.actual_holdings))
        blocked = tuple(map(str, self.blocked_exit_symbols))
        if any(not symbol for symbol in holdings + blocked):
            raise PlanningError("reference-book symbols must be non-empty strings")
        if len(holdings) != len(set(holdings)) or len(blocked) != len(set(blocked)):
            raise PlanningError("reference-book holdings and blocked exits must be unique")
        if not set(blocked).issubset(set(holdings)):
            raise PlanningError("blocked exits must be a subset of actual holdings")
        nav = {str(key): value for key, value in self.sizing_nav_cny_by_scenario.items()}
        nav_hash = {str(key): str(value) for key, value in self.sizing_nav_record_sha256_by_scenario.items()}
        if set(nav) != set(SCENARIOS) or set(nav_hash) != set(SCENARIOS):
            raise PlanningError(f"sizing NAV mappings must exactly cover {SCENARIOS}")
        normalized_nav: dict[str, float] = {}
        for scenario in SCENARIOS:
            value = nav[scenario]
            if isinstance(value, (bool, np.bool_)):
                raise PlanningError("sizing NAV must be numeric")
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise PlanningError("sizing NAV must be finite and positive")
            if not SHA256_RE.fullmatch(nav_hash[scenario]):
                raise PlanningError("sizing NAV record identity must be a lowercase SHA-256")
            normalized_nav[scenario] = number
        object.__setattr__(self, "actual_holdings", holdings)
        object.__setattr__(self, "blocked_exit_symbols", blocked)
        object.__setattr__(self, "sizing_nav_cny_by_scenario", MappingProxyType(normalized_nav))
        object.__setattr__(
            self,
            "sizing_nav_record_sha256_by_scenario",
            MappingProxyType({scenario: nav_hash[scenario] for scenario in SCENARIOS}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_holdings": list(self.actual_holdings),
            "blocked_exit_symbols": list(self.blocked_exit_symbols),
            "sizing_nav_cny_by_scenario": dict(self.sizing_nav_cny_by_scenario),
            "sizing_nav_record_sha256_by_scenario": dict(self.sizing_nav_record_sha256_by_scenario),
        }

    @classmethod
    def from_value(cls, value: ReferenceBook | Mapping[str, Any]) -> ReferenceBook:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise PlanningError("reference book must be a ReferenceBook or exact mapping")
        expected = {
            "actual_holdings",
            "blocked_exit_symbols",
            "sizing_nav_cny_by_scenario",
            "sizing_nav_record_sha256_by_scenario",
        }
        if set(value) != expected:
            raise PlanningError("reference-book mapping has non-exact fields")
        return cls(
            actual_holdings=tuple(value["actual_holdings"]),
            blocked_exit_symbols=tuple(value["blocked_exit_symbols"]),
            sizing_nav_cny_by_scenario=dict(value["sizing_nav_cny_by_scenario"]),
            sizing_nav_record_sha256_by_scenario=dict(value["sizing_nav_record_sha256_by_scenario"]),
        )


@dataclass(frozen=True)
class PlanningVerification:
    """Machine-readable result produced only by semantic recomputation."""

    valid: bool
    artifact_sha256: str | None
    recomputed_sha256: str | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VERIFICATION_SCHEMA,
            "valid": bool(self.valid),
            "artifact_sha256": self.artifact_sha256,
            "recomputed_sha256": self.recomputed_sha256,
            "errors": list(self.errors),
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite values cannot enter canonical JSON")
        return number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_json(value: Any) -> bytes:
    """Return the protocol-wide canonical UTF-8 JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_protocol(path: str | Path = PROTOCOL_PATH) -> tuple[dict[str, Any], str]:
    """Load only a byte-for-byte copy of the authoritative V2.1 protocol."""

    authority = PROTOCOL_PATH.read_bytes()
    supplied = Path(path).read_bytes()
    if supplied != authority:
        raise PlanningError(f"protocol must be byte-for-byte identical to {PROTOCOL_PATH.name}")
    protocol = json.loads(supplied)
    required = {
        "protocol_id",
        "selection",
        "portfolio_accounting",
        "eligibility",
        "factor_model",
        "timing_arms",
        "controls",
        "execution_rules",
    }
    if missing := required - set(protocol):
        raise PlanningError(f"protocol missing planner fields: {sorted(missing)}")
    selection = protocol["selection"]
    controls = protocol["controls"]
    chan = protocol["timing_arms"]["chan"]
    ma = protocol["timing_arms"]["ma_placebo"]
    if selection["target_size"] != 50 or selection["retention_max_rank"] != 75:
        raise PlanningError("authoritative protocol no longer has the registered 50/75 buffer")
    if selection["cash_buffer"] != 0.005:
        raise PlanningError("authoritative cash buffer differs from 0.005")
    if ma["window"] != 20 or ma["condition"] != "adjusted_close_t_gt_sma20_t":
        raise PlanningError("authoritative protocol no longer has the registered SMA20 gate")
    if tuple(chan["state_columns_whitelist"]) != STATE_COLUMNS:
        raise PlanningError("state projection is not exactly symbol/dt/regime")
    if controls["random_seed"] != 20260720 or controls["random_seed_count"] != 20:
        raise PlanningError("authoritative protocol no longer freezes seeds 20260720..20260739")
    if controls["rng_algorithm_version"] != "xs_chan_rng_v2_1_sha256_seedsequence_v1":
        raise PlanningError("unsupported V2.1 RNG algorithm")
    return protocol, _sha256_bytes(supplied)


def frozen_seeds(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    controls = protocol["controls"]
    start = int(controls["random_seed"])
    count = int(controls["random_seed_count"])
    seeds = tuple(range(start, start + count))
    if seeds != tuple(range(20260720, 20260740)):
        raise PlanningError("seed registry differs from 20260720..20260739")
    return seeds


def reference_book_keys(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return the exact set of independent reference books used for identity freeze."""

    if protocol is None:
        protocol, _ = load_protocol()
    seeds = frozen_seeds(protocol)
    return (
        "F",
        "FC",
        "FMA",
        *(f"R_match@{seed}" for seed in seeds),
        *(f"FGR@{seed}" for seed in seeds),
        *(f"FMGR@{seed}" for seed in seeds),
    )


def empty_reference_books(
    *,
    reference_nav_cny: float = 10_000_000.0,
    capacity_nav_cny: float = 100_000_000.0,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> dict[str, ReferenceBook]:
    """Build deterministic genesis fixtures; formal ledgers replace the record digests."""

    protocol, _ = load_protocol(protocol_path)
    result: dict[str, ReferenceBook] = {}
    for key in reference_book_keys(protocol):
        nav = {
            "gross": float(reference_nav_cny),
            "1x": float(reference_nav_cny),
            "2x": float(reference_nav_cny),
            "capacity": float(capacity_nav_cny),
        }
        result[key] = ReferenceBook(
            actual_holdings=(),
            blocked_exit_symbols=(),
            sizing_nav_cny_by_scenario=nav,
            sizing_nav_record_sha256_by_scenario={
                scenario: object_sha256({"genesis": True, "book": key, "scenario": scenario, "nav_cny": nav[scenario]})
                for scenario in SCENARIOS
            },
        )
    return result


def _normalise_reference_books(
    values: Mapping[str, ReferenceBook | Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, ReferenceBook]:
    if not isinstance(values, Mapping):
        raise PlanningError("reference_books must be a mapping")
    expected = set(reference_book_keys(protocol))
    if set(values) != expected:
        raise PlanningError(
            f"reference-book keys differ; missing={sorted(expected - set(values))[:5]}, "
            f"extra={sorted(set(values) - expected)[:5]}"
        )
    result = {key: ReferenceBook.from_value(values[key]) for key in sorted(expected)}
    maximum = int(protocol["execution_rules"]["maximum_actual_positions"])
    for key, book in result.items():
        if len(book.actual_holdings) > maximum:
            raise PlanningError(f"{key} reference book exceeds {maximum} actual positions")
    return result


def _normalise_previous_random(
    value: Mapping[int | str, Sequence[str]], seeds: Sequence[int]
) -> dict[int, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise PlanningError("previous_r_match_targets_by_seed must be a mapping")
    converted: dict[int, tuple[str, ...]] = {}
    for raw_seed, symbols in value.items():
        if isinstance(raw_seed, bool):
            raise PlanningError("random-control seed keys must be integers")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise PlanningError("random-control seed keys must be integers") from exc
        if str(seed) != str(raw_seed) and not isinstance(raw_seed, int):
            raise PlanningError("random-control seed keys must use canonical integers")
        frozen = tuple(map(str, symbols))
        if any(not symbol for symbol in frozen) or len(frozen) != len(set(frozen)):
            raise PlanningError("previous random targets must be unique non-empty symbols")
        converted[seed] = frozen
    if set(converted) != set(seeds):
        raise PlanningError("previous random-target map must exactly cover all 20 frozen seeds")
    return {seed: converted[seed] for seed in seeds}


def _normalise_date(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise PlanningError(f"{label} contains an invalid date")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    if missing := set(columns) - set(frame):
        raise PlanningError(f"{label} missing columns: {sorted(missing)}")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if frame[list(columns)].isna().any(axis=None) or frame.duplicated(list(columns)).any():
        raise PlanningError(f"{label} must have a complete unique key {tuple(columns)}")


def _prepare_calendar(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    _require_columns(calendar, ("trade_date", "is_open"), "calendar")
    if calendar.empty or not calendar["is_open"].eq(True).all():
        raise PlanningError("calendar must contain official open sessions only")
    sessions = pd.DatetimeIndex(_normalise_date(calendar["trade_date"], "calendar.trade_date"))
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise PlanningError("calendar sessions must be unique and increasing")
    return sessions


def _validate_decision_pair(
    calendar: pd.DataFrame, decision_dt: Any, exec_dt: Any | None
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    sessions = _prepare_calendar(calendar)
    decision = pd.Timestamp(decision_dt).normalize()
    if decision not in sessions:
        raise PlanningError("decision_dt must be an official exchange session")
    position = int(sessions.get_loc(decision))
    if position + 1 >= len(sessions):
        raise PlanningError("calendar has not yet published the next execution session")
    registered_exec = pd.Timestamp(sessions[position + 1]).normalize()
    execution = registered_exec if exec_dt is None else pd.Timestamp(exec_dt).normalize()
    if execution != registered_exec:
        raise PlanningError("exec_dt must be the immediately following official session")
    if decision.isocalendar()[:2] == execution.isocalendar()[:2]:
        raise PlanningError("decision must be the final official session of a completed week")
    return decision, execution, sessions


def _active_interval_value(
    frame: pd.DataFrame,
    decision: pd.Timestamp,
    value_column: str,
    label: str,
) -> Any:
    if frame.empty:
        return None
    mask = frame["effective_from"].le(decision) & (frame["effective_to"].isna() | frame["effective_to"].ge(decision))
    active = frame.loc[mask]
    if len(active) > 1:
        raise PlanningError(f"{label} has overlapping PIT intervals")
    return None if active.empty else active.iloc[0][value_column]


def _contains_st(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return "ST" in str(value).upper()


def _prepare_source_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not isinstance(frames, Mapping) or set(frames) != set(REQUIRED_FRAMES):
        missing = sorted(set(REQUIRED_FRAMES) - set(frames)) if isinstance(frames, Mapping) else list(REQUIRED_FRAMES)
        extra = sorted(set(frames) - set(REQUIRED_FRAMES)) if isinstance(frames, Mapping) else []
        raise PlanningError(f"planner frame mapping must be exact; missing={missing}, extra={extra}")
    out = {name: frame.copy(deep=True) for name, frame in frames.items()}
    if any(not isinstance(frame, pd.DataFrame) for frame in out.values()):
        raise PlanningError("every planner frame must be a pandas DataFrame")

    _require_columns(
        out["security_master"],
        ("ts_code", "name", "board", "list_status", "list_date", "delist_date"),
        "security_master",
    )
    _require_columns(out["namechange"], ("ts_code", "name", "effective_from", "effective_to"), "namechange")
    _require_columns(out["raw_daily"], ("ts_code", "trade_date", "close", "vol", "amount"), "raw_daily")
    _require_columns(out["adj_factor"], ("ts_code", "trade_date", "adj_factor"), "adj_factor")
    _require_columns(out["daily_basic"], ("ts_code", "trade_date", "free_share"), "daily_basic")
    _require_columns(
        out["industry_membership"],
        ("ts_code", "industry_code", "classification_version", "effective_from", "effective_to"),
        "industry_membership",
    )
    _require_columns(
        out["official_daily_status"], ("ts_code", "trade_date", "official_status"), "official_daily_status"
    )
    if tuple(out["chan_states"].columns) != STATE_COLUMNS:
        raise PlanningError(f"chan_states must contain only {STATE_COLUMNS} in that order")

    master = out["security_master"]
    master["ts_code"] = master["ts_code"].astype(str)
    master["list_date"] = _normalise_date(master["list_date"], "security_master.list_date")
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce").dt.normalize()
    _require_unique(master, ("ts_code",), "security_master")
    if not master["list_status"].astype(str).isin({"L", "D", "P"}).all():
        raise PlanningError("security_master list_status must be L/D/P")

    for name in ("raw_daily", "adj_factor", "daily_basic", "official_daily_status"):
        frame = out[name]
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = _normalise_date(frame["trade_date"], f"{name}.trade_date")
        _require_unique(frame, ("ts_code", "trade_date"), name)
    for name in ("namechange", "industry_membership"):
        frame = out[name]
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["effective_from"] = _normalise_date(frame["effective_from"], f"{name}.effective_from")
        frame["effective_to"] = pd.to_datetime(frame["effective_to"], errors="coerce").dt.normalize()
    _require_unique(out["namechange"], ("ts_code", "effective_from"), "namechange")
    _require_unique(
        out["industry_membership"],
        ("ts_code", "classification_version", "effective_from"),
        "industry_membership",
    )
    if not out["industry_membership"]["classification_version"].astype(str).eq("SW2021").all():
        raise PlanningError("industry_membership must use SW2021")

    states = out["chan_states"]
    states["symbol"] = states["symbol"].astype(str)
    states["dt"] = _normalise_date(states["dt"], "chan_states.dt")
    regimes = pd.to_numeric(states["regime"], errors="raise")
    numeric = regimes.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise PlanningError("chan regime must be a finite integer")
    states["regime"] = regimes.astype("int64")
    if not states["regime"].between(0, 10).all():
        raise PlanningError("chan regime must be in 0..10")
    _require_unique(states, ("symbol", "dt"), "chan_states")

    raw = out["raw_daily"]
    adj = out["adj_factor"]
    basic = out["daily_basic"]
    for frame, columns, label in (
        (raw, ("close", "vol", "amount"), "raw_daily"),
        (adj, ("adj_factor",), "adj_factor"),
        (basic, ("free_share",), "daily_basic"),
    ):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        values = frame[list(columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise PlanningError(f"{label} numeric inputs must be finite")
    if (raw["close"] <= 0).any() or (raw[["vol", "amount"]] < 0).any(axis=None):
        raise PlanningError("raw close/volume/amount values are invalid")
    if (adj["adj_factor"] <= 0).any() or (basic["free_share"] <= 0).any():
        raise PlanningError("adjustment factor and free shares must be positive")
    if not out["official_daily_status"]["official_status"].isin({"TRADING", "SUSPENDED"}).all():
        raise PlanningError("official status must be TRADING/SUSPENDED")
    return out


def compute_decision_surface(
    frames: Mapping[str, pd.DataFrame],
    decision_dt: Any,
    *,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> pd.DataFrame:
    """Recompute the causal PIT feature/ranking surface at one decision close."""

    protocol, _ = load_protocol(protocol_path)
    source = _prepare_source_frames(frames)
    decision = pd.Timestamp(decision_dt).normalize()
    sessions = _prepare_calendar(source["calendar"])
    if decision not in sessions:
        raise PlanningError("decision_dt is absent from the official calendar")
    prefix_sessions = sessions[sessions <= decision]

    master = source["security_master"]
    raw = source["raw_daily"].loc[source["raw_daily"]["trade_date"].le(decision)].copy()
    adj = source["adj_factor"].loc[source["adj_factor"]["trade_date"].le(decision)].copy()
    basic = source["daily_basic"].loc[source["daily_basic"]["trade_date"].le(decision)].copy()
    statuses = source["official_daily_status"].loc[source["official_daily_status"]["trade_date"].le(decision)].copy()
    names = source["namechange"].loc[source["namechange"]["effective_from"].le(decision)].copy()
    industries = source["industry_membership"].loc[source["industry_membership"]["effective_from"].le(decision)].copy()
    states = source["chan_states"].loc[source["chan_states"]["dt"].le(decision)].copy()

    raw_key = pd.MultiIndex.from_frame(raw[["ts_code", "trade_date"]])
    adj_key = pd.MultiIndex.from_frame(adj[["ts_code", "trade_date"]])
    if len(raw_key.difference(adj_key)) or len(adj_key.difference(raw_key)):
        raise PlanningError("raw_daily and same-day adj_factor keys must match exactly")
    observed = raw.merge(
        adj[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    observed["observed_adjusted_close"] = observed["close"] * observed["adj_factor"]
    observed_groups = {
        symbol: group.set_index("trade_date") for symbol, group in observed.groupby("ts_code", sort=False)
    }
    status_groups = {symbol: group.set_index("trade_date") for symbol, group in statuses.groupby("ts_code", sort=False)}
    basic_at_decision = basic.loc[basic["trade_date"].eq(decision)].set_index("ts_code", drop=False)
    name_groups = dict(iter(names.groupby("ts_code", sort=False)))
    industry_groups = dict(iter(industries.groupby("ts_code", sort=False)))

    rows: list[dict[str, Any]] = []
    eligibility = protocol["eligibility"]
    for security in master.sort_values("ts_code", kind="mergesort").itertuples(index=False):
        symbol = str(security.ts_code)
        list_date = pd.Timestamp(security.list_date)
        delist_date = pd.NaT if pd.isna(security.delist_date) else pd.Timestamp(security.delist_date)
        member = bool(list_date <= decision and (pd.isna(delist_date) or delist_date > decision))
        symbol_sessions = prefix_sessions[prefix_sessions >= list_date]
        if not member or symbol_sessions.empty:
            continue
        if list_date < prefix_sessions[0] and len(symbol_sessions) < int(
            eligibility["min_market_sessions_since_listing"]
        ):
            raise PlanningError(f"calendar prefix cannot establish listing-age eligibility for {symbol}")

        symbol_status = status_groups.get(symbol)
        if symbol_status is None:
            raise PlanningError(f"official status lifecycle is missing for {symbol}")
        aligned_status = symbol_status.reindex(symbol_sessions)
        if aligned_status["official_status"].isna().any():
            raise PlanningError(f"official status lifecycle has a gap for {symbol}")
        symbol_observed = observed_groups.get(symbol)
        if symbol_observed is None:
            symbol_observed = observed.iloc[:0].set_index("trade_date")
        trading = aligned_status["official_status"].eq("TRADING")
        observed_on_clock = symbol_observed.reindex(symbol_sessions)
        raw_present = observed_on_clock["close"].notna()
        if not raw_present.eq(trading).all():
            raise PlanningError(f"raw/official-status coverage differs for {symbol}")

        adjusted = observed_on_clock["observed_adjusted_close"].where(trading).ffill()
        amount = observed_on_clock["amount"].where(trading, 0.0).fillna(0.0)
        volume = observed_on_clock["vol"].where(trading, 0.0).fillna(0.0)
        valid_session = trading & amount.gt(0) & volume.gt(0)
        log_price = np.log(adjusted)
        mom = log_price.shift(20) - log_price.shift(120)
        lowvol = -log_price.diff().rolling(60, min_periods=60).std(ddof=1)
        sma20 = adjusted.rolling(20, min_periods=20).mean()
        adv20 = amount.rolling(20, min_periods=20).mean()
        valid60 = valid_session.astype("int64").rolling(60, min_periods=60).sum()

        decision_raw = observed_on_clock.loc[decision] if decision in observed_on_clock.index else None
        free_share = math.nan
        if symbol in basic_at_decision.index:
            selected_basic = basic_at_decision.loc[symbol]
            if isinstance(selected_basic, pd.DataFrame):
                raise PlanningError(f"daily_basic decision key is duplicated for {symbol}")
            free_share = float(selected_basic["free_share"])
        raw_close = math.nan if decision_raw is None or pd.isna(decision_raw["close"]) else float(decision_raw["close"])
        free_float_mcap = raw_close * free_share * 10_000.0

        symbol_names = name_groups.get(symbol, names.iloc[:0])
        active_name = (
            str(security.name)
            if symbol_names.empty
            else _active_interval_value(symbol_names, decision, "name", f"namechange[{symbol}]")
        )
        is_st = _contains_st(active_name)
        industry = _active_interval_value(
            industry_groups.get(symbol, industries.iloc[:0]),
            decision,
            "industry_code",
            f"industry[{symbol}]",
        )
        board = None if pd.isna(security.board) else str(security.board)
        values = {
            "market_sessions_since_listing": int(len(symbol_sessions)),
            "valid_sessions_60": float(valid60.iloc[-1]),
            "adv20": float(adv20.iloc[-1]),
            "mom_120_20": float(mom.iloc[-1]),
            "lowvol_60": float(lowvol.iloc[-1]),
            "free_float_mcap": float(free_float_mcap),
            "adjusted_close": float(adjusted.iloc[-1]),
            "sma20": float(sma20.iloc[-1]),
        }
        reasons: list[str] = []
        if not member:
            reasons.append("outside_pit_universe")
        if values["market_sessions_since_listing"] < int(eligibility["min_market_sessions_since_listing"]):
            reasons.append("listing_history_lt_min")
        if not math.isfinite(values["valid_sessions_60"]) or values["valid_sessions_60"] < int(
            eligibility["min_valid_sessions_60"]
        ):
            reasons.append("valid_sessions_60_lt_min")
        if not math.isfinite(values["adv20"]) or values["adv20"] < float(eligibility["min_adv20_thousand_cny"]):
            reasons.append("adv20_lt_min")
        if is_st is not False:
            reasons.append("st_or_unknown_name")
        if board is None or board in set(eligibility["exclude_boards"]):
            reasons.append("excluded_or_unknown_board")
        if industry is None or pd.isna(industry) or not str(industry):
            reasons.append("missing_pit_industry")
        if any(not math.isfinite(values[key]) for key in values if key != "market_sessions_since_listing"):
            reasons.append("missing_pit_or_factor")
        if any(
            values[key] <= 0
            for key in ("adv20", "free_float_mcap", "adjusted_close", "sma20")
            if math.isfinite(values[key])
        ):
            reasons.append("nonpositive_pit_or_factor")
        rows.append(
            {
                "symbol": symbol,
                "dt": decision,
                "pit_universe_member": member,
                **values,
                "is_st": is_st,
                "board": board,
                "industry_code": None if industry is None else str(industry),
                "eligible": not reasons,
                "ineligibility_reasons": "|".join(dict.fromkeys(reasons)),
            }
        )
    if not rows:
        raise PlanningError("decision surface is empty")
    surface = pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(drop=True)
    ranked = _rank_cross_section(surface, protocol)

    state_at_decision = states.loc[states["dt"].eq(decision), list(STATE_COLUMNS)].copy()
    ranked = ranked.merge(state_at_decision, on=["symbol", "dt"], how="left", validate="one_to_one")
    allowed = set(map(int, protocol["timing_arms"]["chan"]["allowed_regimes"]))
    ranked["state_missing"] = ranked["regime"].isna()
    ranked["chan_allowed"] = ranked["regime"].isin(allowed)
    ranked["ma_allowed"] = ranked["adjusted_close"] > ranked["sma20"]
    ranked["factor_rank"] = ranked["factor_rank"].astype("int64")
    return ranked.sort_values(["factor_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def _percentile_bucket(values: pd.Series, symbols: pd.Series, buckets: int) -> pd.Series:
    count = min(int(buckets), len(values))
    if count <= 1:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)
    ordered = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="raise").astype(float), "symbol": symbols.astype(str)},
        index=values.index,
    ).sort_values(["value", "symbol"], kind="mergesort")
    ordinal = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    return pd.qcut(ordinal, count, labels=False).astype("int64").reindex(values.index)


def _neutralized_rank(group: pd.DataFrame, factor: str, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(group[factor], errors="raise").astype(float)
    clipped = values.clip(
        values.quantile(low, interpolation="linear"),
        values.quantile(high, interpolation="linear"),
    )
    industry = group["industry_code"].astype(str)
    mcap = pd.to_numeric(group["free_float_mcap"], errors="raise").astype(float)
    if (mcap <= 0).any() or not np.isfinite(mcap.to_numpy()).all():
        raise PlanningError("eligible free-float market cap must be positive and finite")
    log_mcap = np.log(mcap.to_numpy())
    std = float(np.std(log_mcap, ddof=0))
    scaled = np.zeros(len(group)) if std == 0 else (log_mcap - float(np.mean(log_mcap))) / std
    categories = sorted(industry.unique())
    columns = [np.ones(len(group)), scaled]
    columns.extend((industry.to_numpy() == category).astype(float) for category in categories[1:])
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, clipped.to_numpy(), rcond=None)
    residual = clipped.to_numpy() - design @ beta
    return pd.Series(residual, index=group.index).rank(method="average", pct=True)


def _rank_cross_section(surface: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    eligible = surface.loc[surface["eligible"].eq(True)].copy()
    target = int(protocol["selection"]["target_size"])
    if len(eligible) < target:
        raise PlanningError(f"eligible universe {len(eligible)} is smaller than frozen target {target}")
    formal = ["mom_120_20", "lowvol_60", "adv20", "free_float_mcap", "adjusted_close", "sma20"]
    numeric = eligible[formal].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or eligible["industry_code"].isna().any():
        raise PlanningError("eligible factor/PIT fields must be complete and finite")
    low, high = map(float, protocol["factor_model"]["winsor_limits"])
    eligible["mom_120_20_rank"] = _neutralized_rank(eligible, "mom_120_20", low, high)
    eligible["lowvol_60_rank"] = _neutralized_rank(eligible, "lowvol_60", low, high)
    weights = protocol["factor_model"]["weights"]
    eligible["factor_score"] = (
        float(weights["mom_120_20"]) * eligible["mom_120_20_rank"]
        + float(weights["lowvol_60"]) * eligible["lowvol_60_rank"]
    )
    eligible["mcap_bucket"] = _percentile_bucket(eligible["free_float_mcap"], eligible["symbol"], 5)
    eligible["adv_bucket"] = _percentile_bucket(eligible["adv20"], eligible["symbol"], 5)
    eligible = eligible.sort_values(["factor_score", "symbol"], ascending=[False, True], kind="mergesort")
    eligible["factor_rank"] = np.arange(1, len(eligible) + 1, dtype=np.int64)
    if not pd.api.types.is_integer_dtype(eligible["factor_rank"].dtype):
        raise AssertionError("factor_rank lost integer dtype")
    return eligible


def _strict_previous_symbols(value: Sequence[str], label: str) -> tuple[str, ...]:
    symbols = tuple(map(str, value))
    if any(not symbol for symbol in symbols) or len(symbols) != len(set(symbols)):
        raise PlanningError(f"{label} must contain unique non-empty symbols")
    return symbols


def _buffered_factor_target(
    ranked: pd.DataFrame, previous_targets: Sequence[str], protocol: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, str]]:
    ranks = ranked.set_index("symbol")["factor_rank"]
    rank_values = ranks.to_numpy()
    if not pd.api.types.is_integer_dtype(ranks.dtype) or any(
        isinstance(value, (bool, np.bool_)) for value in rank_values
    ):
        raise PlanningError("factor_rank must remain an integer field; coercion is forbidden")
    expected = np.arange(1, len(ranked) + 1, dtype=np.int64)
    if not np.array_equal(np.sort(rank_values.astype(np.int64)), expected):
        raise PlanningError("factor_rank must be the complete 1..N permutation")
    target_size = int(protocol["selection"]["target_size"])
    retention = int(protocol["selection"]["retention_max_rank"])
    previous = set(_strict_previous_symbols(previous_targets, "previous_factor_targets"))
    ordered = ranked.sort_values(["factor_rank", "symbol"], kind="mergesort")
    retained = ordered.loc[ordered["symbol"].isin(previous) & ordered["factor_rank"].le(retention), "symbol"].tolist()
    selected = retained[:target_size]
    selected_set = set(selected)
    for symbol in ordered["symbol"].astype(str):
        if symbol not in selected_set:
            selected.append(symbol)
            selected_set.add(symbol)
        if len(selected) == target_size:
            break
    if len(selected) != target_size:
        raise PlanningError("buffered selection cannot fill all 50 slots")
    retained_set = set(retained[:target_size])
    return tuple(selected), {
        symbol: ("buffer_retain" if symbol in retained_set else "rank_entry") for symbol in selected
    }


@dataclass(frozen=True)
class _SlotResult:
    targets: tuple[str, ...]
    new_entries: tuple[str, ...]
    blocked_placeholders: tuple[str, ...]
    held_targets: tuple[str, ...]
    exits_requested: tuple[str, ...]


def _resolve_reference_slots(targets: Sequence[str], book: ReferenceBook, protocol: Mapping[str, Any]) -> _SlotResult:
    maximum = int(protocol["execution_rules"]["maximum_actual_positions"])
    ordered = tuple(map(str, targets))
    if len(ordered) != len(set(ordered)):
        raise PlanningError("target identities must be unique")
    held = set(book.actual_holdings)
    blocked = set(book.blocked_exit_symbols)
    target_set = set(ordered)
    blocked_outside = tuple(sorted(blocked - target_set))
    held_targets = tuple(symbol for symbol in ordered if symbol in held)
    available_new = maximum - len(blocked_outside) - len(held_targets)
    if available_new < 0:
        raise PlanningError("blocked exits and retained targets exceed the registered slot budget")
    admitted_new = tuple(symbol for symbol in ordered if symbol not in held)[:available_new]
    admitted = set(held_targets) | set(admitted_new)
    effective = tuple(symbol for symbol in ordered if symbol in admitted)
    if len(effective) + len(blocked_outside) > maximum:
        raise AssertionError("reference slot budget exceeded")
    exits = tuple(sorted(held - set(effective)))
    return _SlotResult(effective, admitted_new, blocked_outside, held_targets, exits)


def _rng(
    protocol: Mapping[str, Any], arm: str, seed_index: int, seed: int, decision: pd.Timestamp
) -> np.random.Generator:
    identity = [
        protocol["controls"]["rng_algorithm_version"],
        protocol["protocol_id"],
        arm,
        int(seed_index),
        decision.date().isoformat(),
        int(seed),
    ]
    entropy = np.frombuffer(hashlib.sha256(canonical_json(identity)).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _stratum_by_symbol(ranked: pd.DataFrame) -> dict[str, tuple[str, int, int]]:
    return {
        str(row.symbol): (str(row.industry_code), int(row.mcap_bucket), int(row.adv_bucket))
        for row in ranked.itertuples(index=False)
    }


def _matched_random_factor_target(
    *,
    ranked: pd.DataFrame,
    factor_targets: Sequence[str],
    factor_previous: Sequence[str],
    random_previous: Sequence[str],
    seed: int,
    seed_index: int,
    decision: pd.Timestamp,
    protocol: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    strata = _stratum_by_symbol(ranked)
    factor_previous_set = set(factor_previous)
    random_previous_set = set(random_previous)
    target_counts = Counter(strata[symbol] for symbol in factor_targets)
    retained_counts = Counter(strata[symbol] for symbol in factor_targets if symbol in factor_previous_set)
    pool_by_stratum: dict[tuple[str, int, int], list[str]] = {}
    for symbol in ranked["symbol"].astype(str):
        pool_by_stratum.setdefault(strata[symbol], []).append(symbol)
    generator = _rng(protocol, "R_MATCH", seed_index, seed, decision)
    selected: list[str] = []
    retained_selected: set[str] = set()
    for stratum in sorted(target_counts):
        pool = sorted(pool_by_stratum.get(stratum, []))
        incumbents = [symbol for symbol in pool if symbol in random_previous_set]
        need_retained = int(retained_counts[stratum])
        if len(incumbents) < need_retained:
            raise ExactControlError(
                f"R_match seed {seed} stratum {stratum} has {len(incumbents)} incumbents; needs {need_retained}"
            )
        incumbent_order = generator.permutation(incumbents).tolist() if incumbents else []
        kept = list(map(str, incumbent_order[:need_retained]))
        selected.extend(kept)
        retained_selected.update(kept)
        need_new = int(target_counts[stratum]) - need_retained
        candidates = [symbol for symbol in pool if symbol not in random_previous_set]
        if len(candidates) < need_new:
            raise ExactControlError(
                f"R_match seed {seed} stratum {stratum} has {len(candidates)} new controls; needs {need_new}"
            )
        chosen = generator.choice(candidates, size=need_new, replace=False).tolist() if need_new else []
        selected.extend(map(str, chosen))
    if len(selected) != len(factor_targets) or len(selected) != len(set(selected)):
        raise ExactControlError(f"R_match seed {seed} did not produce an exact unique target")
    selected = sorted(selected, key=lambda symbol: (strata[symbol], symbol))
    if Counter(strata[symbol] for symbol in selected) != target_counts:
        raise ExactControlError(f"R_match seed {seed} stratum counts differ")
    actual_retained = Counter(strata[symbol] for symbol in selected if symbol in random_previous_set)
    if actual_retained != retained_counts:
        raise ExactControlError(f"R_match seed {seed} retention-by-stratum differs")
    return tuple(selected), {
        symbol: ("matched_retain" if symbol in retained_selected else "matched_new") for symbol in selected
    }


@dataclass(frozen=True)
class _GateResult:
    targets: tuple[str, ...]
    new_entries: tuple[str, ...]
    retained: tuple[str, ...]
    candidate_new_count: int
    allowed_new_count: int
    missing_state_count: int
    blocked_placeholders: tuple[str, ...]
    exits_requested: tuple[str, ...]


def _real_gate(
    base_targets: Sequence[str],
    book: ReferenceBook,
    ranked_by_symbol: pd.DataFrame,
    gate: str,
    protocol: Mapping[str, Any],
) -> _GateResult:
    slots = _resolve_reference_slots(base_targets, book, protocol)
    held = set(book.actual_holdings)
    retained = tuple(symbol for symbol in slots.targets if symbol in held)
    candidates = tuple(symbol for symbol in slots.targets if symbol not in held)
    column = "chan_allowed" if gate == "chan" else "ma_allowed"
    accepted_new = tuple(symbol for symbol in candidates if bool(ranked_by_symbol.loc[symbol, column]))
    accepted = set(retained) | set(accepted_new)
    targets = tuple(symbol for symbol in slots.targets if symbol in accepted)
    missing = sum(bool(ranked_by_symbol.loc[symbol, "state_missing"]) for symbol in candidates) if gate == "chan" else 0
    return _GateResult(
        targets=targets,
        new_entries=accepted_new,
        retained=retained,
        candidate_new_count=len(candidates),
        allowed_new_count=len(accepted_new),
        missing_state_count=int(missing),
        blocked_placeholders=slots.blocked_placeholders,
        exits_requested=tuple(sorted(set(book.actual_holdings) - set(targets))),
    )


def _random_quota_gate(
    base_targets: Sequence[str],
    book: ReferenceBook,
    real_gate: _GateResult,
    *,
    arm: str,
    seed: int,
    seed_index: int,
    decision: pd.Timestamp,
    protocol: Mapping[str, Any],
) -> _GateResult:
    slots = _resolve_reference_slots(base_targets, book, protocol)
    held = set(book.actual_holdings)
    retained = tuple(symbol for symbol in slots.targets if symbol in held)
    candidates = tuple(symbol for symbol in slots.targets if symbol not in held)
    if len(retained) != len(real_gate.retained):
        raise ExactControlError(
            f"{arm} seed {seed} retained quota {len(retained)} differs from real gate {len(real_gate.retained)}"
        )
    need_new = len(real_gate.new_entries)
    if need_new > len(candidates):
        raise ExactControlError(f"{arm} seed {seed} has no pool for exact new-entry quota {need_new}")
    generator = _rng(protocol, arm, seed_index, seed, decision)
    chosen = set(map(str, generator.choice(candidates, size=need_new, replace=False))) if need_new else set()
    accepted = set(retained) | chosen
    targets = tuple(symbol for symbol in slots.targets if symbol in accepted)
    new_entries = tuple(symbol for symbol in targets if symbol not in held)
    if len(targets) != len(real_gate.targets) or len(new_entries) != need_new:
        raise ExactControlError(f"{arm} seed {seed} failed exact retained/new quota")
    return _GateResult(
        targets=targets,
        new_entries=new_entries,
        retained=retained,
        candidate_new_count=len(candidates),
        allowed_new_count=need_new,
        missing_state_count=0,
        blocked_placeholders=slots.blocked_placeholders,
        exits_requested=tuple(sorted(set(book.actual_holdings) - set(targets))),
    )


def _selection_identity(decision: pd.Timestamp, execution: pd.Timestamp, symbols: Sequence[str]) -> str:
    return object_sha256(
        {
            "decision_session": decision.date().isoformat(),
            "execution_session": execution.date().isoformat(),
            "ordered_symbols": list(symbols),
        }
    )


def _arm_payload(
    *,
    arm_key: str,
    symbols: Sequence[str],
    new_entries: Sequence[str],
    blocked: Sequence[str],
    exits: Sequence[str],
    gate_opportunities: int,
    gate_candidate_count: int,
    missing_state_count: int,
    quota: Mapping[str, int] | None,
    selection_reason: Mapping[str, str],
    ranked: pd.DataFrame,
    book: ReferenceBook,
    decision: pd.Timestamp,
    execution: pd.Timestamp,
) -> dict[str, Any]:
    ordered = tuple(map(str, symbols))
    new = tuple(map(str, new_entries))
    if any(symbol not in ordered for symbol in new):
        raise PlanningError(f"{arm_key} new entries must be a target subset")
    indexed = ranked.set_index("symbol", drop=False)
    for symbol in ordered:
        rank = indexed.loc[symbol, "factor_rank"]
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
            raise PlanningError("factor rank must remain an integer in every arm artifact")
    identity = _selection_identity(decision, execution, ordered)
    scenario_identity = dict.fromkeys(SCENARIOS, identity)
    return {
        "arm_key": arm_key,
        "ordered_symbols": list(ordered),
        "new_entry_symbols": list(new),
        "blocked_exit_placeholders": list(blocked),
        "exit_request_symbols": list(exits),
        "factor_rank_by_symbol": {symbol: int(indexed.loc[symbol, "factor_rank"]) for symbol in ordered},
        "selection_reason_by_symbol": {symbol: str(selection_reason[symbol]) for symbol in ordered},
        "stratum_by_symbol": {
            symbol: [
                str(indexed.loc[symbol, "industry_code"]),
                int(indexed.loc[symbol, "mcap_bucket"]),
                int(indexed.loc[symbol, "adv_bucket"]),
            ]
            for symbol in ordered
        },
        "decision_adv_cny_by_symbol": {symbol: float(indexed.loc[symbol, "adv20"]) * 1000.0 for symbol in ordered},
        "gate_eligible_new_entry_opportunities": int(gate_opportunities),
        "gate_candidate_new_entry_count": int(gate_candidate_count),
        "missing_chan_state_new_entry_count": int(missing_state_count),
        "exact_gate_quota": None if quota is None else {key: int(quota[key]) for key in ("total", "retained", "new")},
        "selection_identity_sha256": identity,
        "cost_scenario_selection_identity_sha256": scenario_identity,
        "sizing_nav_reference": {
            scenario: {
                "nav_cny": float(book.sizing_nav_cny_by_scenario[scenario]),
                "valuation_record_sha256": book.sizing_nav_record_sha256_by_scenario[scenario],
            }
            for scenario in SCENARIOS
        },
    }


def _clean_scalar(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _frame_sha256(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    row_hashes = []
    for values in frame.itertuples(index=False, name=None):
        row_hashes.append(object_sha256([_clean_scalar(value) for value in values]))
    return object_sha256({"columns": columns, "row_sha256_multiset": sorted(row_hashes)})


def frame_mapping_sha256(frames: Mapping[str, pd.DataFrame]) -> str:
    prepared = _prepare_source_frames(frames)
    return object_sha256({name: _frame_sha256(prepared[name]) for name in sorted(prepared)})


def _surface_payload(ranked: pd.DataFrame) -> list[dict[str, Any]]:
    columns = (
        "symbol",
        "dt",
        "mom_120_20",
        "lowvol_60",
        "adv20",
        "industry_code",
        "free_float_mcap",
        "adjusted_close",
        "sma20",
        "mom_120_20_rank",
        "lowvol_60_rank",
        "factor_score",
        "mcap_bucket",
        "adv_bucket",
        "factor_rank",
        "regime",
        "state_missing",
        "chan_allowed",
        "ma_allowed",
    )
    rows = []
    for row in ranked.loc[:, list(columns)].sort_values("factor_rank", kind="mergesort").to_dict(orient="records"):
        rows.append({key: _clean_scalar(value) for key, value in row.items()})
    return rows


def _execution_decision_matrix(
    arms: Mapping[str, Mapping[str, Any]],
    decision: pd.Timestamp,
    execution: pd.Timestamp,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand 63 frozen arm identities into the registered 252 replay decisions."""

    rows: list[dict[str, Any]] = []
    for arm_key in reference_book_keys(protocol):
        arm = arms[arm_key]
        family, separator, seed_text = arm_key.partition("@")
        seed = int(seed_text) if separator else None
        for scenario in SCENARIOS:
            sizing = arm["sizing_nav_reference"][scenario]
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "arm_key": arm_key,
                    "scenario": scenario,
                    "decision_session": decision.date().isoformat(),
                    "execution_session": execution.date().isoformat(),
                    "ordered_symbols": list(arm["ordered_symbols"]),
                    "new_entry_symbols": list(arm["new_entry_symbols"]),
                    "blocked_exit_placeholders": list(arm["blocked_exit_placeholders"]),
                    "gate_eligible_new_entry_opportunities": int(arm["gate_eligible_new_entry_opportunities"]),
                    "slots": int(protocol["execution_rules"]["maximum_actual_positions"]),
                    "cash_buffer_fraction": float(protocol["selection"]["cash_buffer"]),
                    "selection_identity_sha256": str(arm["selection_identity_sha256"]),
                    "sizing_nav_cny": float(sizing["nav_cny"]),
                    "sizing_nav_record_sha256": str(sizing["valuation_record_sha256"]),
                }
            )
    if len(rows) != 252:
        raise AssertionError(f"V2.1 execution matrix must contain 252 decisions, got {len(rows)}")
    return rows


def _build_artifact(
    *,
    frames: Mapping[str, pd.DataFrame],
    decision_dt: Any,
    exec_dt: Any | None,
    reference_books: Mapping[str, ReferenceBook | Mapping[str, Any]],
    previous_factor_targets: Sequence[str],
    previous_r_match_targets_by_seed: Mapping[int | str, Sequence[str]],
    data_authority: str,
    data_snapshot_sha256: str,
    protocol_path: str | Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    decision, execution, _ = _validate_decision_pair(frames["calendar"], decision_dt, exec_dt)
    seeds = frozen_seeds(protocol)
    books = _normalise_reference_books(reference_books, protocol)
    previous_factor = _strict_previous_symbols(previous_factor_targets, "previous_factor_targets")
    previous_random = _normalise_previous_random(previous_r_match_targets_by_seed, seeds)
    ranked = compute_decision_surface(frames, decision, protocol_path=protocol_path)
    indexed = ranked.set_index("symbol", drop=False)
    factor_full, factor_reason = _buffered_factor_target(ranked, previous_factor, protocol)

    arms: dict[str, dict[str, Any]] = {}
    f_slots = _resolve_reference_slots(factor_full, books["F"], protocol)
    arms["F"] = _arm_payload(
        arm_key="F",
        symbols=f_slots.targets,
        new_entries=f_slots.new_entries,
        blocked=f_slots.blocked_placeholders,
        exits=f_slots.exits_requested,
        gate_opportunities=len(f_slots.new_entries),
        gate_candidate_count=len(f_slots.new_entries),
        missing_state_count=0,
        quota=None,
        selection_reason=factor_reason,
        ranked=ranked,
        book=books["F"],
        decision=decision,
        execution=execution,
    )

    fc = _real_gate(factor_full, books["FC"], indexed, "chan", protocol)
    fma = _real_gate(factor_full, books["FMA"], indexed, "ma", protocol)
    for key, gate in (("FC", fc), ("FMA", fma)):
        quota = {"total": len(gate.targets), "retained": len(gate.retained), "new": len(gate.new_entries)}
        arms[key] = _arm_payload(
            arm_key=key,
            symbols=gate.targets,
            new_entries=gate.new_entries,
            blocked=gate.blocked_placeholders,
            exits=gate.exits_requested,
            gate_opportunities=gate.allowed_new_count,
            gate_candidate_count=gate.candidate_new_count,
            missing_state_count=gate.missing_state_count,
            quota=quota,
            selection_reason=factor_reason,
            ranked=ranked,
            book=books[key],
            decision=decision,
            execution=execution,
        )

    factor_strata = Counter(tuple(value) for value in arms["F"]["stratum_by_symbol"].values())
    factor_retained_strata = Counter(
        tuple(arms["F"]["stratum_by_symbol"][symbol])
        for symbol in arms["F"]["ordered_symbols"]
        if symbol in set(previous_factor)
    )
    for seed_index, seed in enumerate(seeds):
        r_key = f"R_match@{seed}"
        random_full, random_reason = _matched_random_factor_target(
            ranked=ranked,
            factor_targets=f_slots.targets,
            factor_previous=previous_factor,
            random_previous=previous_random[seed],
            seed=seed,
            seed_index=seed_index,
            decision=decision,
            protocol=protocol,
        )
        r_slots = _resolve_reference_slots(random_full, books[r_key], protocol)
        r_strata = Counter(
            (
                str(indexed.loc[symbol, "industry_code"]),
                int(indexed.loc[symbol, "mcap_bucket"]),
                int(indexed.loc[symbol, "adv_bucket"]),
            )
            for symbol in r_slots.targets
        )
        r_retained_strata = Counter(
            (
                str(indexed.loc[symbol, "industry_code"]),
                int(indexed.loc[symbol, "mcap_bucket"]),
                int(indexed.loc[symbol, "adv_bucket"]),
            )
            for symbol in r_slots.targets
            if symbol in set(previous_random[seed])
        )
        if r_strata != factor_strata or r_retained_strata != factor_retained_strata:
            raise ExactControlError(f"{r_key} ceased to be exact after reference-book slot resolution")
        arms[r_key] = _arm_payload(
            arm_key=r_key,
            symbols=r_slots.targets,
            new_entries=r_slots.new_entries,
            blocked=r_slots.blocked_placeholders,
            exits=r_slots.exits_requested,
            gate_opportunities=len(r_slots.new_entries),
            gate_candidate_count=len(r_slots.new_entries),
            missing_state_count=0,
            quota=None,
            selection_reason=random_reason,
            ranked=ranked,
            book=books[r_key],
            decision=decision,
            execution=execution,
        )

        for control_arm, real_key, real_gate in (("FGR", "FC", fc), ("FMGR", "FMA", fma)):
            key = f"{control_arm}@{seed}"
            random_gate = _random_quota_gate(
                factor_full,
                books[key],
                real_gate,
                arm=control_arm,
                seed=seed,
                seed_index=seed_index,
                decision=decision,
                protocol=protocol,
            )
            quota = {
                "total": len(random_gate.targets),
                "retained": len(random_gate.retained),
                "new": len(random_gate.new_entries),
            }
            real_quota = arms[real_key]["exact_gate_quota"]
            if quota != real_quota:
                raise ExactControlError(f"{key} quota differs from {real_key}")
            arms[key] = _arm_payload(
                arm_key=key,
                symbols=random_gate.targets,
                new_entries=random_gate.new_entries,
                blocked=random_gate.blocked_placeholders,
                exits=random_gate.exits_requested,
                gate_opportunities=random_gate.allowed_new_count,
                gate_candidate_count=random_gate.candidate_new_count,
                missing_state_count=0,
                quota=quota,
                selection_reason=factor_reason,
                ranked=ranked,
                book=books[key],
                decision=decision,
                execution=execution,
            )

    expected_arms = set(reference_book_keys(protocol))
    if set(arms) != expected_arms:
        raise AssertionError("planner did not freeze the exact arm/seed registry")
    ordered_arms = {key: arms[key] for key in reference_book_keys(protocol)}
    surface_payload = _surface_payload(ranked)
    planning_inputs = {
        "previous_factor_targets": list(previous_factor),
        "previous_r_match_targets_by_seed": {str(seed): list(previous_random[seed]) for seed in seeds},
        "reference_books": {key: books[key].to_dict() for key in reference_book_keys(protocol)},
    }
    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha,
        "trial_id": protocol_sha,
        "data_authority": data_authority,
        "data_snapshot_sha256": data_snapshot_sha256,
        "decision_dt": decision.date().isoformat(),
        "exec_dt": execution.date().isoformat(),
        "rng_algorithm_version": str(protocol["controls"]["rng_algorithm_version"]),
        "frozen_seeds": list(seeds),
        "eligible_count": int(len(ranked)),
        "ranked_surface_sha256": object_sha256(surface_payload),
        "ranked_surface": surface_payload,
        "planning_inputs": planning_inputs,
        "arms": ordered_arms,
        "execution_decisions": _execution_decision_matrix(ordered_arms, decision, execution, protocol),
    }
    payload["artifact_sha256"] = object_sha256(payload)
    return payload


def build_planning_artifact_from_frames(
    *,
    frames: Mapping[str, pd.DataFrame],
    decision_dt: Any,
    exec_dt: Any | None,
    reference_books: Mapping[str, ReferenceBook | Mapping[str, Any]],
    previous_factor_targets: Sequence[str] = (),
    previous_r_match_targets_by_seed: Mapping[int | str, Sequence[str]] | None = None,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Build a content-hashable, explicitly non-authoritative test artifact."""

    protocol, _ = load_protocol(protocol_path)
    seeds = frozen_seeds(protocol)
    previous_random = previous_r_match_targets_by_seed
    if previous_random is None:
        previous_random = dict.fromkeys(seeds, ())
    digest = frame_mapping_sha256(frames)
    return _build_artifact(
        frames=frames,
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        reference_books=reference_books,
        previous_factor_targets=previous_factor_targets,
        previous_r_match_targets_by_seed=previous_random,
        data_authority=FRAME_AUTHORITY,
        data_snapshot_sha256=digest,
        protocol_path=protocol_path,
    )


def frames_from_verified_snapshot(snapshot: Any) -> dict[str, pd.DataFrame]:
    """Materialize copies from a validated immutable snapshot, never loose files."""

    try:
        import xs_chan_data_v2_1 as data_v2_1
    except ImportError as exc:  # pragma: no cover - only relevant outside the repository scripts path
        raise PlanningError("xs_chan_data_v2_1 is unavailable") from exc
    if not isinstance(snapshot, data_v2_1.VerifiedDataSnapshot):
        raise PlanningError("formal planning requires VerifiedDataSnapshot, not a duck-typed object")
    frames = {name: snapshot.read_parquet(name) for name in REQUIRED_FRAMES if name != "official_daily_status"}
    frames["official_daily_status"] = snapshot.read_evidence_parquet("official_daily_status")
    return _prepare_source_frames(frames)


def build_planning_artifact_from_snapshot(
    *,
    snapshot: Any,
    decision_dt: Any,
    exec_dt: Any | None,
    reference_books: Mapping[str, ReferenceBook | Mapping[str, Any]],
    previous_factor_targets: Sequence[str] = (),
    previous_r_match_targets_by_seed: Mapping[int | str, Sequence[str]] | None = None,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Formal adapter from an immutable, already verified data snapshot."""

    frames = frames_from_verified_snapshot(snapshot)
    protocol, _ = load_protocol(protocol_path)
    seeds = frozen_seeds(protocol)
    previous_random = previous_r_match_targets_by_seed
    if previous_random is None:
        previous_random = dict.fromkeys(seeds, ())
    manifest_sha = str(snapshot.manifest_sha256)
    if not SHA256_RE.fullmatch(manifest_sha):
        raise PlanningError("verified snapshot has a malformed manifest digest")
    return _build_artifact(
        frames=frames,
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        reference_books=reference_books,
        previous_factor_targets=previous_factor_targets,
        previous_r_match_targets_by_seed=previous_random,
        data_authority=SNAPSHOT_AUTHORITY,
        data_snapshot_sha256=manifest_sha,
        protocol_path=protocol_path,
    )


def _strict_artifact_shape(value: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "protocol_id",
        "protocol_sha256",
        "trial_id",
        "data_authority",
        "data_snapshot_sha256",
        "decision_dt",
        "exec_dt",
        "rng_algorithm_version",
        "frozen_seeds",
        "eligible_count",
        "ranked_surface_sha256",
        "ranked_surface",
        "planning_inputs",
        "arms",
        "execution_decisions",
        "artifact_sha256",
    }
    if set(value) != expected or value.get("schema") != ARTIFACT_SCHEMA:
        raise PlanningError("planning artifact top-level shape is not exact")
    for key in ("protocol_sha256", "trial_id", "data_snapshot_sha256", "ranked_surface_sha256", "artifact_sha256"):
        if not SHA256_RE.fullmatch(str(value.get(key))):
            raise PlanningError(f"planning artifact {key} is malformed")
    without_hash = dict(value)
    claimed = str(without_hash.pop("artifact_sha256"))
    if object_sha256(without_hash) != claimed:
        raise PlanningError("planning artifact content hash differs")
    inputs = value.get("planning_inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "previous_factor_targets",
        "previous_r_match_targets_by_seed",
        "reference_books",
    }:
        raise PlanningError("planning_inputs shape is not exact")


def planner_input_from_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the strict, content-addressed replay input bound by an artifact."""

    if not isinstance(artifact, Mapping):
        raise PlanningError("artifact must be a mapping")
    value = copy.deepcopy(dict(artifact))
    _strict_artifact_shape(value)
    planner_input: dict[str, Any] = {
        "schema": PLANNER_INPUT_SCHEMA,
        "protocol_id": value["protocol_id"],
        "protocol_sha256": value["protocol_sha256"],
        "data_authority": value["data_authority"],
        "data_snapshot_sha256": value["data_snapshot_sha256"],
        "decision_dt": value["decision_dt"],
        "exec_dt": value["exec_dt"],
        "planning_inputs": value["planning_inputs"],
    }
    planner_input["planner_input_sha256"] = object_sha256(planner_input)
    return planner_input


def _strict_planner_input(value: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "protocol_id",
        "protocol_sha256",
        "data_authority",
        "data_snapshot_sha256",
        "decision_dt",
        "exec_dt",
        "planning_inputs",
        "planner_input_sha256",
    }
    if set(value) != expected or value.get("schema") != PLANNER_INPUT_SCHEMA:
        raise PlanningError("planner_input shape is not exact")
    supplied = str(value["planner_input_sha256"])
    if not SHA256_RE.fullmatch(supplied):
        raise PlanningError("planner_input_sha256 is malformed")
    payload = dict(value)
    payload.pop("planner_input_sha256")
    if object_sha256(payload) != supplied:
        raise PlanningError("planner_input content hash differs")


def verify_planning_artifact(
    artifact: Mapping[str, Any],
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
    snapshot: Any | None = None,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> PlanningVerification:
    """Recompute all selection semantics and compare exact canonical bytes.

    Exactly one source must be supplied.  The function never accepts a caller
    assertion about validity; ``valid`` can become true only after recomputation.
    """

    errors: list[str] = []
    claimed: str | None = None
    recomputed_sha: str | None = None
    try:
        if not isinstance(artifact, Mapping):
            raise PlanningError("artifact must be a mapping")
        artifact_copy = copy.deepcopy(dict(artifact))
        claimed = str(artifact_copy.get("artifact_sha256")) if "artifact_sha256" in artifact_copy else None
        _strict_artifact_shape(artifact_copy)
        if (frames is None) == (snapshot is None):
            raise PlanningError("verification requires exactly one of frames or snapshot")
        inputs = artifact_copy["planning_inputs"]
        kwargs = {
            "decision_dt": artifact_copy["decision_dt"],
            "exec_dt": artifact_copy["exec_dt"],
            "reference_books": inputs["reference_books"],
            "previous_factor_targets": inputs["previous_factor_targets"],
            "previous_r_match_targets_by_seed": inputs["previous_r_match_targets_by_seed"],
            "protocol_path": protocol_path,
        }
        if snapshot is not None:
            recomputed = build_planning_artifact_from_snapshot(snapshot=snapshot, **kwargs)
        else:
            assert frames is not None
            recomputed = build_planning_artifact_from_frames(frames=frames, **kwargs)
        recomputed_sha = str(recomputed["artifact_sha256"])
        if canonical_json(recomputed) != canonical_json(artifact_copy):
            raise PlanningError("semantic replay differs from the supplied planning artifact")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return PlanningVerification(
        valid=not errors,
        artifact_sha256=claimed if claimed and SHA256_RE.fullmatch(claimed) else None,
        recomputed_sha256=recomputed_sha,
        errors=tuple(errors),
    )


def verify_planner_result(
    planner_result: Mapping[str, Any],
    planner_input: Mapping[str, Any],
    snapshot: Any,
    *,
    protocol_path: str | Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Compatibility entry point for the independent semantic replay verifier.

    Both input and result are strict content-addressed JSON values.  The input
    is checked against the result before the formal snapshot is replayed.  The
    returned value is canonical-JSON-compatible and contains no trusted caller
    booleans.
    """

    errors: list[str] = []
    result: PlanningVerification | None = None
    input_sha: str | None = None
    try:
        if not isinstance(planner_input, Mapping):
            raise PlanningError("planner_input must be a mapping")
        strict_input = copy.deepcopy(dict(planner_input))
        _strict_planner_input(strict_input)
        input_sha = str(strict_input["planner_input_sha256"])
        expected_input = planner_input_from_artifact(planner_result)
        if canonical_json(strict_input) != canonical_json(expected_input):
            raise PlanningError("planner_input does not exactly bind planner_result")
        result = verify_planning_artifact(planner_result, snapshot=snapshot, protocol_path=protocol_path)
        errors.extend(result.errors)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    payload = {
        "schema": VERIFICATION_SCHEMA,
        "valid": not errors,
        "planner_input_sha256": input_sha,
        "artifact_sha256": None if result is None else result.artifact_sha256,
        "recomputed_sha256": None if result is None else result.recomputed_sha256,
        "execution_decision_count": (
            len(planner_result.get("execution_decisions", [])) if isinstance(planner_result, Mapping) else 0
        ),
        "errors": errors,
    }
    payload["verification_sha256"] = object_sha256(payload)
    return payload


__all__ = [
    "ARTIFACT_SCHEMA",
    "ExactControlError",
    "FRAME_AUTHORITY",
    "PLANNER_INPUT_SCHEMA",
    "PlanningError",
    "PlanningVerification",
    "ReferenceBook",
    "SCENARIOS",
    "SNAPSHOT_AUTHORITY",
    "STATE_COLUMNS",
    "build_planning_artifact_from_frames",
    "build_planning_artifact_from_snapshot",
    "canonical_json",
    "compute_decision_surface",
    "empty_reference_books",
    "frame_mapping_sha256",
    "frames_from_verified_snapshot",
    "frozen_seeds",
    "load_protocol",
    "object_sha256",
    "planner_input_from_artifact",
    "reference_book_keys",
    "verify_planner_result",
    "verify_planning_artifact",
]
