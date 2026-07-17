"""Append-only, hash-chained OOS decision and execution ledger.

The ledger is deliberately small and independent from the research engine.  A
freeze record (sequence 000000) locks the protocol, engine dependencies, and
exchange calendar.  Thereafter records must alternate strictly between a
weekly decision and its next-session execution.

Each record is stored as canonical JSON in a file named
``<sequence>_<record-sha256>.json``.  Records are created with ``O_EXCL`` and
linked by the previous record hash.  Verification rejects non-canonical bytes,
hash mismatches, missing sequence numbers, and forks.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 1
ZERO_HASH = "0" * 64
DEFAULT_MIN_COMPLETE_WEEKS = 52
DEFAULT_EXCHANGE_TIMEZONE = "Asia/Shanghai"
DEFAULT_DECISION_TIME_LOCAL = "15:00:00"
DEFAULT_EXECUTION_TIME_LOCAL = "09:30:00"
DEFAULT_EXECUTION_RECORD_DEADLINE_LOCAL = "15:00:00"
DECISION_DATA_SCHEMA = "xs_chan_v2_decision_v1"
EXECUTION_DATA_SCHEMA = "xs_chan_v2_execution_v1"
EXPECTED_SEED_IDS = tuple(range(20260717, 20260737))
EXPECTED_DEPENDENCY_KEYS = frozenset(
    {
        "data_evidence",
        "state_engine",
        "feature_engine",
        "research_engine",
        "execution_engine",
        "statistics_engine",
        "replay_verifier",
    }
)

_DECISION_DATA_KEYS = {
    "schema",
    "asof_snapshot_sha256",
    "asof_cutoff_utc",
    "engine_sha256",
    "plan_sha256",
    "gate_identity_sha256",
    "reference_books_sha256",
    "adv_snapshot_sha256",
    "state_snapshot_sha256",
    "ma_snapshot_sha256",
    "seed_ids",
    "arm_identity_sha256",
    "invariants",
}
_DECISION_HASH_KEYS = _DECISION_DATA_KEYS - {
    "schema",
    "asof_cutoff_utc",
    "seed_ids",
    "arm_identity_sha256",
    "invariants",
}
_DECISION_INVARIANTS = {
    "blocked_slots_applied",
    "target_count_lte_50",
    "identities_frozen_before_cost_replay",
}
_ARM_KEYS = {"F", "FC", "FMA", "R_MATCH", "FGR", "FMGR"}
_SINGLE_IDENTITY_ARMS = {"F", "FC", "FMA"}
_SEEDED_IDENTITY_ARMS = _ARM_KEYS - _SINGLE_IDENTITY_ARMS

_EXECUTION_DATA_KEYS = {
    "schema",
    "engine_sha256",
    "execution_snapshot_sha256",
    "snapshot_cutoff_utc",
    "raw_ohlc_sha256",
    "stk_limit_sha256",
    "corporate_actions_sha256",
    "requested_orders_sha256",
    "fills_sha256",
    "fees_sha256",
    "portfolio_state_sha256",
    "pending_sells_sha256",
    "gate_identity_sha256",
    "scenario_replay_sha256",
    "invariants",
}
_EXECUTION_HASH_KEYS = _EXECUTION_DATA_KEYS - {"schema", "snapshot_cutoff_utc", "invariants"}
_EXECUTION_INVARIANTS = {
    "all_arms_present",
    "no_negative_cash",
    "max_positions_lte_50",
    "same_identity_all_cost_scenarios",
    "exact_seed_set",
    "corporate_actions_applied",
    "terminal_events_resolved",
}

_RECORD_NAME = re.compile(r"^(?P<sequence>\d{6})_(?P<digest>[0-9a-f]{64})\.json$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "schema_version",
    "sequence",
    "record_type",
    "recorded_at_utc",
    "previous_record_hash",
    "payload",
    "record_hash",
}
_FREEZE_PAYLOAD_KEYS = {
    "protocol_sha256",
    "dependency_sha256",
    "calendar_sha256",
    "calendar_sessions",
    "oos_start_date",
    "min_complete_weeks",
    "exchange_timezone",
    "decision_time_local",
    "execution_time_local",
    "execution_record_deadline_local",
}


class LedgerError(RuntimeError):
    """Base class for ledger failures."""


class LedgerConflictError(LedgerError):
    """Raised when an append would overwrite or conflict with existing history."""


class LedgerValidationError(LedgerError):
    """Raised when the existing chain or a proposed transition is invalid."""


class LedgerTimingError(LedgerValidationError):
    """Raised when a record is written on the wrong side of the execution deadline."""


@dataclass(frozen=True)
class LedgerRecord:
    """A verified canonical record and its source bytes."""

    path: Path
    data: dict[str, Any]
    raw: bytes

    @property
    def sequence(self) -> int:
        return int(self.data["sequence"])

    @property
    def record_hash(self) -> str:
        return str(self.data["record_hash"])

    @property
    def record_type(self) -> str:
        return str(self.data["record_type"])


@dataclass(frozen=True)
class _ChainState:
    records: tuple[LedgerRecord, ...]
    freeze: LedgerRecord
    sessions: tuple[date, ...]
    session_index: Mapping[date, int]
    weekly_pairs: tuple[tuple[date, date], ...]
    pending_decision: LedgerRecord | None
    last_decision_dt: date | None
    completed_weeks: int
    min_complete_weeks: int


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"record is not canonical-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise LedgerValidationError(f"{label} must be a lowercase SHA256 digest")
    return value


def _normalise_utc(value: datetime | str | None) -> datetime:
    if value is None:
        parsed = datetime.now(UTC)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LedgerValidationError(f"invalid UTC timestamp: {value!r}") from exc
    else:
        parsed = value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerValidationError("recorded_at_utc must be timezone-aware")
    return parsed.astimezone(UTC)


def _utc_now() -> datetime:
    """Internal clock seam; production callers cannot inject historical time."""
    return datetime.now(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _recorded_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{label} must be a canonical UTC string")
    parsed = _normalise_utc(value)
    if value != _format_utc(parsed):
        raise LedgerValidationError(f"{label} must use canonical UTC Z format with microseconds")
    return parsed


def _normalise_date(value: date | datetime | str, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise LedgerValidationError(f"{label} must be YYYY-MM-DD: {value!r}") from exc


def _normalise_calendar(values: Iterable[date | datetime | str]) -> tuple[date, ...]:
    sessions = tuple(_normalise_date(value, "calendar session") for value in values)
    if len(sessions) < 2:
        raise LedgerValidationError("calendar must contain at least two sessions")
    if sessions != tuple(sorted(set(sessions))):
        raise LedgerValidationError("calendar sessions must be strictly increasing and unique")
    return sessions


def _calendar_hash(sessions: Sequence[date]) -> str:
    return _sha256_bytes(_canonical_bytes([session.isoformat() for session in sessions]))


def _weekly_decision_pairs(sessions: Sequence[date], start: date) -> tuple[tuple[date, date], ...]:
    """Return consecutive weekly decision/next-session pairs from the first full OOS week."""
    weeks: dict[tuple[int, int], list[date]] = {}
    for session in sessions:
        weeks.setdefault(session.isocalendar()[:2], []).append(session)
    grouped = list(weeks.values())
    start_week = start.isocalendar()[:2]
    start_group_index = next(index for index, group in enumerate(grouped) if group[0].isocalendar()[:2] == start_week)
    # A start in the middle of a frozen-calendar week makes that week partial;
    # confirmation begins with the following complete trading week.
    if grouped[start_group_index][0] != start:
        start_group_index += 1
    session_index = {session: index for index, session in enumerate(sessions)}
    pairs: list[tuple[date, date]] = []
    for group in grouped[start_group_index:]:
        decision_dt = group[-1]
        index = session_index[decision_dt]
        if index + 1 >= len(sessions):
            break
        exec_dt = sessions[index + 1]
        if decision_dt.isocalendar()[:2] == exec_dt.isocalendar()[:2]:
            raise LedgerValidationError("frozen calendar week grouping is internally inconsistent")
        pairs.append((decision_dt, exec_dt))
    return tuple(pairs)


def _build_record(
    sequence: int,
    record_type: str,
    recorded_at_utc: datetime,
    previous_record_hash: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "sequence": int(sequence),
        "record_type": str(record_type),
        "recorded_at_utc": _format_utc(recorded_at_utc),
        "previous_record_hash": _require_sha256(previous_record_hash, "previous_record_hash"),
        "payload": dict(payload),
    }
    record_hash = _sha256_bytes(_canonical_bytes(body))
    record = {**body, "record_hash": record_hash}
    return record, _canonical_bytes(record)


def _record_path(root: Path, sequence: int, record_hash: str) -> Path:
    return root / f"{sequence:06d}_{record_hash}.json"


@contextmanager
def _ledger_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".ledger.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _exclusive_write(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise LedgerConflictError(f"record already exists: {path.name}") from exc
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _scan_records(root: Path) -> tuple[LedgerRecord, ...]:
    if not root.exists() or not root.is_dir():
        raise LedgerValidationError(f"ledger directory does not exist: {root}")
    candidates = sorted(path for path in root.iterdir() if path.suffix == ".json")
    grouped: dict[int, list[tuple[Path, str]]] = {}
    for path in candidates:
        match = _RECORD_NAME.fullmatch(path.name)
        if match is None:
            raise LedgerValidationError(f"invalid ledger record filename: {path.name}")
        sequence = int(match.group("sequence"))
        grouped.setdefault(sequence, []).append((path, match.group("digest")))
    if not grouped:
        raise LedgerValidationError(f"ledger has no freeze record: {root}")
    forks = {sequence: paths for sequence, paths in grouped.items() if len(paths) != 1}
    if forks:
        sequence = min(forks)
        names = sorted(path.name for path, _ in forks[sequence])
        raise LedgerValidationError(f"fork detected at sequence {sequence:06d}: {names}")
    sequences = sorted(grouped)
    expected = list(range(sequences[-1] + 1))
    if sequences != expected:
        missing = sorted(set(expected) - set(sequences))
        raise LedgerValidationError(f"missing ledger sequence numbers: {missing}")

    records: list[LedgerRecord] = []
    previous_hash = ZERO_HASH
    previous_recorded_at: datetime | None = None
    for sequence in sequences:
        path, filename_hash = grouped[sequence][0]
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerValidationError(f"invalid JSON in {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise LedgerValidationError(f"record must be a JSON object: {path.name}")
        if set(data) != _RECORD_KEYS:
            raise LedgerValidationError(f"unexpected record fields in {path.name}")
        if raw != _canonical_bytes(data):
            raise LedgerValidationError(f"record bytes are not canonical: {path.name}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise LedgerValidationError(f"unsupported schema version in {path.name}")
        if data.get("sequence") != sequence:
            raise LedgerValidationError(f"sequence mismatch in {path.name}")
        record_hash = _require_sha256(data.get("record_hash", ""), f"record_hash in {path.name}")
        if record_hash != filename_hash:
            raise LedgerValidationError(f"filename hash mismatch in {path.name}")
        body = {key: value for key, value in data.items() if key != "record_hash"}
        if _sha256_bytes(_canonical_bytes(body)) != record_hash:
            raise LedgerValidationError(f"record hash mismatch in {path.name}")
        if data.get("previous_record_hash") != previous_hash:
            raise LedgerValidationError(f"previous hash mismatch in {path.name}")
        recorded_at = _recorded_utc(data.get("recorded_at_utc"), f"recorded_at_utc in {path.name}")
        if previous_recorded_at is not None and recorded_at <= previous_recorded_at:
            raise LedgerValidationError(f"recorded_at_utc is not strictly increasing in {path.name}")
        records.append(LedgerRecord(path=path, data=data, raw=raw))
        previous_hash = record_hash
        previous_recorded_at = recorded_at
    return tuple(records)


def _freeze_context(
    record: LedgerRecord,
) -> tuple[tuple[date, ...], dict[date, int], tuple[tuple[date, date], ...], int]:
    if record.sequence != 0 or record.record_type != "freeze":
        raise LedgerValidationError("sequence 000000 must be the freeze record")
    if record.data["previous_record_hash"] != ZERO_HASH:
        raise LedgerValidationError("freeze record must use the zero previous hash")
    payload = record.data.get("payload")
    if not isinstance(payload, dict):
        raise LedgerValidationError("freeze payload must be an object")
    if set(payload) != _FREEZE_PAYLOAD_KEYS:
        raise LedgerValidationError("freeze payload fields do not match schema version 1")
    _require_sha256(payload.get("protocol_sha256", ""), "protocol_sha256")
    dependencies = payload.get("dependency_sha256")
    if not isinstance(dependencies, dict) or set(dependencies) != EXPECTED_DEPENDENCY_KEYS:
        raise LedgerValidationError("freeze dependency_sha256 must match the exact frozen dependency closure")
    for name, digest in dependencies.items():
        if not isinstance(name, str) or not name:
            raise LedgerValidationError("dependency names must be non-empty strings")
        _require_sha256(digest, f"dependency {name!r}")
    sessions_raw = payload.get("calendar_sessions")
    if not isinstance(sessions_raw, list):
        raise LedgerValidationError("freeze calendar_sessions must be a list")
    sessions = _normalise_calendar(sessions_raw)
    expected_calendar_hash = _calendar_hash(sessions)
    stored_calendar_hash = _require_sha256(payload.get("calendar_sha256", ""), "calendar_sha256")
    if stored_calendar_hash != expected_calendar_hash:
        raise LedgerValidationError("embedded calendar does not match calendar_sha256")
    oos_start = _normalise_date(payload.get("oos_start_date", ""), "oos_start_date")
    if oos_start not in sessions:
        raise LedgerValidationError("oos_start_date must be an exchange session")
    min_complete_weeks = payload.get("min_complete_weeks")
    if min_complete_weeks != DEFAULT_MIN_COMPLETE_WEEKS:
        raise LedgerValidationError(f"min_complete_weeks must be frozen at {DEFAULT_MIN_COMPLETE_WEEKS}")
    weekly_pairs = _weekly_decision_pairs(sessions, oos_start)
    if len(weekly_pairs) < min_complete_weeks:
        raise LedgerValidationError("frozen calendar does not contain 52 complete weekly decision/execution pairs")
    timezone_name = payload.get("exchange_timezone")
    if timezone_name != DEFAULT_EXCHANGE_TIMEZONE:
        raise LedgerValidationError(f"exchange_timezone must be frozen at {DEFAULT_EXCHANGE_TIMEZONE}")
    try:
        ZoneInfo(str(timezone_name))
    except ZoneInfoNotFoundError as exc:
        raise LedgerValidationError(f"unknown exchange timezone: {timezone_name!r}") from exc
    for field, expected in (
        ("decision_time_local", DEFAULT_DECISION_TIME_LOCAL),
        ("execution_time_local", DEFAULT_EXECUTION_TIME_LOCAL),
        ("execution_record_deadline_local", DEFAULT_EXECUTION_RECORD_DEADLINE_LOCAL),
    ):
        try:
            local_time = time.fromisoformat(str(payload.get(field)))
        except ValueError as exc:
            raise LedgerValidationError(f"{field} must be HH:MM[:SS]") from exc
        if local_time.tzinfo is not None:
            raise LedgerValidationError(f"{field} must not contain a timezone offset")
        if expected is not None and payload[field] != expected:
            raise LedgerValidationError(f"{field} must be frozen at {expected}")
    genesis_recorded_at = _recorded_utc(record.data["recorded_at_utc"], "freeze recorded_at_utc")
    _validate_genesis_time(
        genesis_recorded_at,
        oos_start=oos_start,
        first_decision_dt=weekly_pairs[0][0],
        timezone_name=str(timezone_name),
    )
    return sessions, {session: index for index, session in enumerate(sessions)}, weekly_pairs, min_complete_weeks


def _local_session_time(session_dt: date, freeze: LedgerRecord, field: str) -> datetime:
    payload = freeze.data["payload"]
    timezone = ZoneInfo(str(payload["exchange_timezone"]))
    local_time = time.fromisoformat(str(payload[field]))
    return datetime.combine(session_dt, local_time, tzinfo=timezone).astimezone(UTC)


def _execution_deadline(exec_dt: date, freeze: LedgerRecord) -> datetime:
    return _local_session_time(exec_dt, freeze, "execution_time_local")


def _execution_record_deadline(exec_dt: date, freeze: LedgerRecord) -> datetime:
    return _local_session_time(exec_dt, freeze, "execution_record_deadline_local")


def _decision_close(decision_dt: date, freeze: LedgerRecord) -> datetime:
    return _local_session_time(decision_dt, freeze, "decision_time_local")


def _validate_genesis_time(
    recorded_at_utc: datetime,
    *,
    oos_start: date,
    first_decision_dt: date,
    timezone_name: str,
) -> None:
    timezone = ZoneInfo(timezone_name)
    oos_start_boundary = datetime.combine(oos_start, time.min, tzinfo=timezone).astimezone(UTC)
    first_decision_close = datetime.combine(
        first_decision_dt,
        time.fromisoformat(DEFAULT_DECISION_TIME_LOCAL),
        tzinfo=timezone,
    ).astimezone(UTC)
    if recorded_at_utc >= first_decision_close:
        raise LedgerTimingError("freeze must be recorded before the first OOS decision close")
    if recorded_at_utc >= oos_start_boundary:
        raise LedgerTimingError("freeze must be recorded before oos_start_date begins")


def _require_exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise LedgerValidationError(f"{label} keys must all be strings")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LedgerValidationError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return value


def _require_hash_fields(data: Mapping[str, Any], fields: set[str], label: str) -> None:
    for field in sorted(fields):
        _require_sha256(data[field], f"{label}.{field}")


def _require_true_invariants(data: Any, expected: set[str], label: str) -> None:
    invariants = _require_exact_keys(data, expected, label)
    invalid = sorted(key for key, value in invariants.items() if type(value) is not bool or value is not True)
    if invalid:
        raise LedgerValidationError(f"{label} must contain literal true for every invariant: {invalid}")


def _cutoff_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{label} must be a timezone-aware timestamp string")
    return _normalise_utc(value)


def _validate_arm_identities(value: Any) -> None:
    arms = _require_exact_keys(value, _ARM_KEYS, "decision.data.arm_identity_sha256")
    for arm in sorted(_SINGLE_IDENTITY_ARMS):
        _require_sha256(arms[arm], f"decision.data.arm_identity_sha256.{arm}")
    expected_seed_keys = {str(seed) for seed in EXPECTED_SEED_IDS}
    for arm in sorted(_SEEDED_IDENTITY_ARMS):
        identities = _require_exact_keys(
            arms[arm],
            expected_seed_keys,
            f"decision.data.arm_identity_sha256.{arm}",
        )
        for seed, digest in identities.items():
            _require_sha256(digest, f"decision.data.arm_identity_sha256.{arm}.{seed}")


def _validate_decision_data(
    data: Any,
    *,
    decision_dt: date,
    recorded_at_utc: datetime,
    freeze: LedgerRecord,
) -> None:
    decision_data = _require_exact_keys(data, _DECISION_DATA_KEYS, "decision.data")
    if decision_data["schema"] != DECISION_DATA_SCHEMA:
        raise LedgerValidationError(f"decision.data.schema must be {DECISION_DATA_SCHEMA!r}")
    _require_hash_fields(decision_data, _DECISION_HASH_KEYS, "decision.data")
    if decision_data["engine_sha256"] != freeze.data["payload"]["dependency_sha256"]["research_engine"]:
        raise LedgerValidationError("decision engine_sha256 differs from the frozen research engine")
    if type(decision_data["seed_ids"]) is not list or decision_data["seed_ids"] != list(EXPECTED_SEED_IDS):
        raise LedgerValidationError("decision.data.seed_ids must exactly equal 20260717..20260736")
    _validate_arm_identities(decision_data["arm_identity_sha256"])
    _require_true_invariants(decision_data["invariants"], _DECISION_INVARIANTS, "decision.data.invariants")
    cutoff = _cutoff_utc(decision_data["asof_cutoff_utc"], "decision.data.asof_cutoff_utc")
    if cutoff < _decision_close(decision_dt, freeze):
        raise LedgerTimingError("decision asof_cutoff_utc precedes the decision session close")
    if cutoff > recorded_at_utc:
        raise LedgerTimingError("decision asof_cutoff_utc exceeds recorded_at_utc")


def _validate_execution_data(
    data: Any,
    *,
    exec_dt: date,
    recorded_at_utc: datetime,
    state: _ChainState,
) -> None:
    execution_data = _require_exact_keys(data, _EXECUTION_DATA_KEYS, "execution.data")
    if execution_data["schema"] != EXECUTION_DATA_SCHEMA:
        raise LedgerValidationError(f"execution.data.schema must be {EXECUTION_DATA_SCHEMA!r}")
    _require_hash_fields(execution_data, _EXECUTION_HASH_KEYS, "execution.data")
    if execution_data["engine_sha256"] != state.freeze.data["payload"]["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("execution engine_sha256 differs from the frozen execution engine")
    _require_true_invariants(
        execution_data["invariants"],
        _EXECUTION_INVARIANTS,
        "execution.data.invariants",
    )
    pending = state.pending_decision
    if pending is None:
        raise LedgerValidationError("execution has no pending decision")
    expected_gate_hash = pending.data["payload"]["data"]["gate_identity_sha256"]
    if execution_data["gate_identity_sha256"] != expected_gate_hash:
        raise LedgerValidationError("execution gate_identity_sha256 does not match the pending decision")
    cutoff = _cutoff_utc(execution_data["snapshot_cutoff_utc"], "execution.data.snapshot_cutoff_utc")
    if cutoff < _execution_deadline(exec_dt, state.freeze):
        raise LedgerTimingError("execution snapshot_cutoff_utc precedes the execution session")
    if cutoff > recorded_at_utc:
        raise LedgerTimingError("execution snapshot_cutoff_utc exceeds recorded_at_utc")


def _validate_append_timestamp(recorded_at_utc: datetime, state: _ChainState) -> None:
    chain_head_time = _normalise_utc(str(state.records[-1].data["recorded_at_utc"]))
    if recorded_at_utc <= chain_head_time:
        raise LedgerValidationError("recorded_at_utc must be strictly increasing")


def _validate_decision_values(
    *,
    decision_dt: date,
    exec_dt: date,
    recorded_at_utc: datetime,
    state: _ChainState,
) -> None:
    _validate_append_timestamp(recorded_at_utc, state)
    if state.pending_decision is not None:
        raise LedgerValidationError("cannot append a decision while an execution is pending")
    start = _normalise_date(state.freeze.data["payload"]["oos_start_date"], "oos_start_date")
    if decision_dt < start:
        raise LedgerValidationError("decision_dt precedes the frozen OOS start")
    if state.last_decision_dt is not None and decision_dt <= state.last_decision_dt:
        raise LedgerValidationError("decision_dt must be strictly later than all prior decisions")
    if decision_dt not in state.session_index or exec_dt not in state.session_index:
        raise LedgerValidationError("decision_dt and exec_dt must both be exchange sessions")
    index = state.session_index[decision_dt]
    if index + 1 >= len(state.sessions) or state.sessions[index + 1] != exec_dt:
        raise LedgerValidationError("exec_dt must be the next exchange session after decision_dt")
    if decision_dt.isocalendar()[:2] == exec_dt.isocalendar()[:2]:
        raise LedgerValidationError("decision_dt must be the final exchange session of its ISO week")
    if state.completed_weeks >= len(state.weekly_pairs):
        raise LedgerValidationError("frozen calendar has no next complete OOS week")
    expected_pair = state.weekly_pairs[state.completed_weeks]
    if (decision_dt, exec_dt) != expected_pair:
        expected_decision, expected_exec = expected_pair
        raise LedgerValidationError(
            "decision does not match the next complete frozen-calendar week: "
            f"expected {expected_decision.isoformat()} -> {expected_exec.isoformat()}"
        )
    if recorded_at_utc < _decision_close(decision_dt, state.freeze):
        raise LedgerTimingError("decision cannot be recorded before the decision session close")
    if recorded_at_utc >= _execution_deadline(exec_dt, state.freeze):
        raise LedgerTimingError("decision must be recorded before the execution session")


def _validate_execution_values(
    *,
    decision_hash: str,
    decision_dt: date,
    exec_dt: date,
    recorded_at_utc: datetime,
    state: _ChainState,
) -> None:
    _validate_append_timestamp(recorded_at_utc, state)
    pending = state.pending_decision
    if pending is None:
        raise LedgerValidationError("execution has no pending decision")
    expected_payload = pending.data["payload"]
    if decision_hash != pending.record_hash:
        raise LedgerValidationError("execution decision_hash does not reference the pending decision")
    if decision_dt.isoformat() != expected_payload.get("decision_dt"):
        raise LedgerValidationError("execution decision_dt does not match its decision")
    if exec_dt.isoformat() != expected_payload.get("exec_dt"):
        raise LedgerValidationError("execution exec_dt does not match its decision")
    if recorded_at_utc < _execution_deadline(exec_dt, state.freeze):
        raise LedgerTimingError("execution cannot be recorded before the execution session")
    if recorded_at_utc >= _execution_record_deadline(exec_dt, state.freeze):
        raise LedgerTimingError("execution must be recorded before the execution session close")


def _validate_chain(root: Path) -> _ChainState:
    records = _scan_records(root)
    freeze = records[0]
    sessions, session_index, weekly_pairs, min_complete_weeks = _freeze_context(freeze)
    state = _ChainState(
        records=records[:1],
        freeze=freeze,
        sessions=sessions,
        session_index=session_index,
        weekly_pairs=weekly_pairs,
        pending_decision=None,
        last_decision_dt=None,
        completed_weeks=0,
        min_complete_weeks=min_complete_weeks,
    )
    completed_week_keys: set[tuple[int, int]] = set()
    for record in records[1:]:
        payload = record.data.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise LedgerValidationError(f"{record.record_type} payload must contain a data object")
        recorded_at = _recorded_utc(record.data["recorded_at_utc"], "recorded_at_utc")
        if record.record_type == "decision":
            if set(payload) != {"decision_dt", "exec_dt", "data"}:
                raise LedgerValidationError("decision payload fields do not match schema version 1")
            decision_dt = _normalise_date(payload.get("decision_dt", ""), "decision_dt")
            exec_dt = _normalise_date(payload.get("exec_dt", ""), "exec_dt")
            _validate_decision_values(
                decision_dt=decision_dt,
                exec_dt=exec_dt,
                recorded_at_utc=recorded_at,
                state=state,
            )
            _validate_decision_data(
                payload["data"],
                decision_dt=decision_dt,
                recorded_at_utc=recorded_at,
                freeze=freeze,
            )
            state = _ChainState(
                records=records[: record.sequence + 1],
                freeze=freeze,
                sessions=sessions,
                session_index=session_index,
                weekly_pairs=weekly_pairs,
                pending_decision=record,
                last_decision_dt=decision_dt,
                completed_weeks=state.completed_weeks,
                min_complete_weeks=min_complete_weeks,
            )
        elif record.record_type == "execution":
            if set(payload) != {"decision_hash", "decision_dt", "exec_dt", "data"}:
                raise LedgerValidationError("execution payload fields do not match schema version 1")
            decision_dt = _normalise_date(payload.get("decision_dt", ""), "decision_dt")
            exec_dt = _normalise_date(payload.get("exec_dt", ""), "exec_dt")
            decision_hash = _require_sha256(payload.get("decision_hash", ""), "decision_hash")
            _validate_execution_values(
                decision_hash=decision_hash,
                decision_dt=decision_dt,
                exec_dt=exec_dt,
                recorded_at_utc=recorded_at,
                state=state,
            )
            _validate_execution_data(
                payload["data"],
                exec_dt=exec_dt,
                recorded_at_utc=recorded_at,
                state=state,
            )
            week_key = decision_dt.isocalendar()[:2]
            if week_key in completed_week_keys:
                raise LedgerValidationError(f"duplicate completed week: {week_key}")
            completed_week_keys.add(week_key)
            state = _ChainState(
                records=records[: record.sequence + 1],
                freeze=freeze,
                sessions=sessions,
                session_index=session_index,
                weekly_pairs=weekly_pairs,
                pending_decision=None,
                last_decision_dt=state.last_decision_dt,
                completed_weeks=state.completed_weeks + 1,
                min_complete_weeks=min_complete_weeks,
            )
        else:
            raise LedgerValidationError(f"unsupported record type at {record.sequence:06d}: {record.record_type!r}")
    return state


def _existing_logical_record(
    records: Sequence[LedgerRecord],
    record_type: str,
    identity: Mapping[str, Any],
) -> LedgerRecord | None:
    matches = []
    for record in records:
        if record.record_type != record_type:
            continue
        payload = record.data["payload"]
        if all(payload.get(key) == value for key, value in identity.items()):
            matches.append(record)
    if len(matches) > 1:
        raise LedgerValidationError(f"multiple logical {record_type} records found")
    return matches[0] if matches else None


def _idempotent_or_conflict(
    existing: LedgerRecord,
    *,
    record_type: str,
    recorded_at_utc: datetime,
    payload: Mapping[str, Any],
) -> Path:
    _, candidate = _build_record(
        existing.sequence,
        record_type,
        recorded_at_utc,
        str(existing.data["previous_record_hash"]),
        payload,
    )
    if candidate == existing.raw:
        return existing.path
    raise LedgerConflictError(f"logical {record_type} already exists at sequence {existing.sequence:06d}; bytes differ")


def init_ledger(
    root: str | Path,
    *,
    protocol_sha256: str,
    dependency_sha256: Mapping[str, str],
    calendar_sessions: Iterable[date | datetime | str],
    oos_start_date: date | datetime | str,
    min_complete_weeks: int = DEFAULT_MIN_COMPLETE_WEEKS,
    exchange_timezone: str = DEFAULT_EXCHANGE_TIMEZONE,
    execution_time_local: str = DEFAULT_EXECUTION_TIME_LOCAL,
) -> Path:
    """Create sequence 000000 using an exclusive write.

    Repeating the call is idempotent only when the generated canonical bytes,
    including ``recorded_at_utc``, are exactly the same.
    """
    root_path = Path(root)
    sessions = _normalise_calendar(calendar_sessions)
    start = _normalise_date(oos_start_date, "oos_start_date")
    if start not in sessions:
        raise LedgerValidationError("oos_start_date must be an exchange session")
    if min_complete_weeks != DEFAULT_MIN_COMPLETE_WEEKS:
        raise LedgerValidationError(f"min_complete_weeks must be frozen at {DEFAULT_MIN_COMPLETE_WEEKS}")
    if len(_weekly_decision_pairs(sessions, start)) < min_complete_weeks:
        raise LedgerValidationError("frozen calendar does not contain 52 complete weekly decision/execution pairs")
    if exchange_timezone != DEFAULT_EXCHANGE_TIMEZONE:
        raise LedgerValidationError(f"exchange_timezone must be frozen at {DEFAULT_EXCHANGE_TIMEZONE}")
    try:
        ZoneInfo(exchange_timezone)
    except ZoneInfoNotFoundError as exc:
        raise LedgerValidationError(f"unknown exchange timezone: {exchange_timezone!r}") from exc
    try:
        execution_time = time.fromisoformat(execution_time_local)
    except ValueError as exc:
        raise LedgerValidationError("execution_time_local must be HH:MM[:SS]") from exc
    if execution_time.tzinfo is not None:
        raise LedgerValidationError("execution_time_local must not contain a timezone offset")
    if execution_time_local != DEFAULT_EXECUTION_TIME_LOCAL:
        raise LedgerValidationError(f"execution_time_local must be frozen at {DEFAULT_EXECUTION_TIME_LOCAL}")
    if set(dependency_sha256) != EXPECTED_DEPENDENCY_KEYS:
        raise LedgerValidationError("dependency_sha256 must match the exact frozen dependency closure")
    dependencies = {
        str(name): _require_sha256(digest, f"dependency {name!r}") for name, digest in dependency_sha256.items()
    }
    payload = {
        "protocol_sha256": _require_sha256(protocol_sha256, "protocol_sha256"),
        "dependency_sha256": dependencies,
        "calendar_sha256": _calendar_hash(sessions),
        "calendar_sessions": [session.isoformat() for session in sessions],
        "oos_start_date": start.isoformat(),
        "min_complete_weeks": min_complete_weeks,
        "exchange_timezone": exchange_timezone,
        "decision_time_local": DEFAULT_DECISION_TIME_LOCAL,
        "execution_time_local": execution_time_local,
        "execution_record_deadline_local": DEFAULT_EXECUTION_RECORD_DEADLINE_LOCAL,
    }
    recorded_at = _normalise_utc(_utc_now())
    _validate_genesis_time(
        recorded_at,
        oos_start=start,
        first_decision_dt=_weekly_decision_pairs(sessions, start)[0][0],
        timezone_name=exchange_timezone,
    )
    record, raw = _build_record(0, "freeze", recorded_at, ZERO_HASH, payload)
    path = _record_path(root_path, 0, record["record_hash"])
    with _ledger_lock(root_path):
        json_files = [candidate for candidate in root_path.iterdir() if candidate.suffix == ".json"]
        if json_files:
            existing = _validate_chain(root_path).freeze
            if raw == existing.raw:
                return existing.path
            raise LedgerConflictError("ledger is already initialized with different freeze bytes")
        _exclusive_write(path, raw)
    return path


def append_decision(
    root: str | Path,
    *,
    decision_dt: date | datetime | str,
    exec_dt: date | datetime | str,
    data: Mapping[str, Any],
) -> Path:
    """Append one weekly decision before its next-session execution deadline."""
    root_path = Path(root)
    decision = _normalise_date(decision_dt, "decision_dt")
    execution = _normalise_date(exec_dt, "exec_dt")
    if not isinstance(data, Mapping):
        raise LedgerValidationError("decision data must be an object")
    recorded_at = _normalise_utc(_utc_now())
    payload = {"decision_dt": decision.isoformat(), "exec_dt": execution.isoformat(), "data": dict(data)}
    with _ledger_lock(root_path):
        state = _validate_chain(root_path)
        existing = _existing_logical_record(
            state.records,
            "decision",
            {"decision_dt": decision.isoformat(), "exec_dt": execution.isoformat()},
        )
        if existing is not None:
            return _idempotent_or_conflict(
                existing,
                record_type="decision",
                recorded_at_utc=recorded_at,
                payload=payload,
            )
        _validate_decision_values(
            decision_dt=decision,
            exec_dt=execution,
            recorded_at_utc=recorded_at,
            state=state,
        )
        _validate_decision_data(
            data,
            decision_dt=decision,
            recorded_at_utc=recorded_at,
            freeze=state.freeze,
        )
        sequence = len(state.records)
        previous_hash = state.records[-1].record_hash
        record, raw = _build_record(sequence, "decision", recorded_at, previous_hash, payload)
        path = _record_path(root_path, sequence, record["record_hash"])
        _exclusive_write(path, raw)
    return path


def append_execution(
    root: str | Path,
    *,
    decision_hash: str,
    decision_dt: date | datetime | str,
    exec_dt: date | datetime | str,
    data: Mapping[str, Any],
) -> Path:
    """Append an execution that references the currently pending decision."""
    root_path = Path(root)
    reference = _require_sha256(decision_hash, "decision_hash")
    decision = _normalise_date(decision_dt, "decision_dt")
    execution = _normalise_date(exec_dt, "exec_dt")
    if not isinstance(data, Mapping):
        raise LedgerValidationError("execution data must be an object")
    recorded_at = _normalise_utc(_utc_now())
    payload = {
        "decision_hash": reference,
        "decision_dt": decision.isoformat(),
        "exec_dt": execution.isoformat(),
        "data": dict(data),
    }
    with _ledger_lock(root_path):
        state = _validate_chain(root_path)
        existing = _existing_logical_record(state.records, "execution", {"decision_hash": reference})
        if existing is not None:
            return _idempotent_or_conflict(
                existing,
                record_type="execution",
                recorded_at_utc=recorded_at,
                payload=payload,
            )
        _validate_execution_values(
            decision_hash=reference,
            decision_dt=decision,
            exec_dt=execution,
            recorded_at_utc=recorded_at,
            state=state,
        )
        _validate_execution_data(
            data,
            exec_dt=execution,
            recorded_at_utc=recorded_at,
            state=state,
        )
        sequence = len(state.records)
        previous_hash = state.records[-1].record_hash
        record, raw = _build_record(sequence, "execution", recorded_at, previous_hash, payload)
        path = _record_path(root_path, sequence, record["record_hash"])
        _exclusive_write(path, raw)
    return path


def verify_ledger(root: str | Path) -> dict[str, Any]:
    """Verify local chain structure; semantic replay remains a separate mandatory gate."""
    state = _validate_chain(Path(root))
    pending = state.pending_decision
    confirmation_window = None
    if state.completed_weeks >= state.min_complete_weeks:
        record_count = state.min_complete_weeks * 2
        records = state.records[1 : record_count + 1]
        if len(records) != record_count or any(
            record.record_type != ("decision" if index % 2 == 0 else "execution")
            for index, record in enumerate(records)
        ):
            raise LedgerValidationError("first-52-week record window is incomplete or misordered")
        pairs = state.weekly_pairs[: state.min_complete_weeks]
        identity = {
            "schema": "xs_chan_v2_confirmation_window_identity_v1",
            "protocol_sha256": state.freeze.data["payload"]["protocol_sha256"],
            "dependency_sha256": state.freeze.data["payload"]["dependency_sha256"],
            "calendar_sha256": state.freeze.data["payload"]["calendar_sha256"],
            "decision_dates": [decision.isoformat() for decision, _ in pairs],
            "execution_dates": [execution.isoformat() for _, execution in pairs],
            "record_hashes": [record.record_hash for record in records],
            "confirmation_head_sha256": records[-1].record_hash,
        }
        confirmation_window = {
            **identity,
            "identity_sha256": _sha256_bytes(_canonical_bytes(identity)),
        }
    structural_window_complete = state.completed_weeks >= state.min_complete_weeks
    return {
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "record_count": len(state.records),
        "completed_weeks": state.completed_weeks,
        "min_complete_weeks": state.min_complete_weeks,
        "structural_window_complete": structural_window_complete,
        "semantic_replay_verified": False,
        "confirmatory_oos": False,
        "pending_decision_hash": pending.record_hash if pending is not None else None,
        "chain_head_sha256": state.records[-1].record_hash,
        "protocol_sha256": state.freeze.data["payload"]["protocol_sha256"],
        "dependency_sha256": dict(state.freeze.data["payload"]["dependency_sha256"]),
        "calendar_sha256": state.freeze.data["payload"]["calendar_sha256"],
        "confirmation_window": confirmation_window,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LedgerValidationError(f"expected a JSON object: {path}")
    return value


def _load_calendar_file(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("sessions")
    if not isinstance(value, list):
        raise LedgerValidationError("calendar JSON must be a list or an object with a sessions list")
    return [str(item) for item in value]


def _parse_dependencies(values: Sequence[str]) -> dict[str, str]:
    dependencies: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise LedgerValidationError("dependency must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in dependencies:
            raise LedgerValidationError(f"invalid or duplicate dependency name: {name!r}")
        dependencies[name] = sha256_file(Path(raw_path))
    return dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create the immutable 000000 freeze record")
    init_parser.add_argument("root", type=Path)
    init_parser.add_argument("--protocol", type=Path, required=True)
    init_parser.add_argument("--dependency", action="append", default=[], metavar="NAME=PATH")
    init_parser.add_argument("--calendar", type=Path, required=True)
    init_parser.add_argument("--oos-start-date", required=True)
    init_parser.add_argument("--exchange-timezone", default=DEFAULT_EXCHANGE_TIMEZONE)
    init_parser.add_argument("--execution-time-local", default=DEFAULT_EXECUTION_TIME_LOCAL)

    append_parser = subparsers.add_parser("append", help="append a decision or execution record")
    append_subparsers = append_parser.add_subparsers(dest="append_type", required=True)

    decision_parser = append_subparsers.add_parser("decision", help="append a pre-execution decision")
    decision_parser.add_argument("root", type=Path)
    decision_parser.add_argument("--decision-dt", required=True)
    decision_parser.add_argument("--exec-dt", required=True)
    decision_parser.add_argument("--payload", type=Path, required=True)

    execution_parser = append_subparsers.add_parser("execution", help="append execution for a pending decision")
    execution_parser.add_argument("root", type=Path)
    execution_parser.add_argument("--decision-hash", required=True)
    execution_parser.add_argument("--decision-dt", required=True)
    execution_parser.add_argument("--exec-dt", required=True)
    execution_parser.add_argument("--payload", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify", help="verify the chain and print maturity status")
    verify_parser.add_argument("root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        path = init_ledger(
            args.root,
            protocol_sha256=sha256_file(args.protocol),
            dependency_sha256=_parse_dependencies(args.dependency),
            calendar_sessions=_load_calendar_file(args.calendar),
            oos_start_date=args.oos_start_date,
            min_complete_weeks=DEFAULT_MIN_COMPLETE_WEEKS,
            exchange_timezone=args.exchange_timezone,
            execution_time_local=args.execution_time_local,
        )
        output = {"path": str(path), "record_hash": _scan_records(args.root)[0].record_hash}
    elif args.command == "append":
        if args.append_type == "decision":
            path = append_decision(
                args.root,
                decision_dt=args.decision_dt,
                exec_dt=args.exec_dt,
                data=_load_json_object(args.payload),
            )
        else:
            path = append_execution(
                args.root,
                decision_hash=args.decision_hash,
                decision_dt=args.decision_dt,
                exec_dt=args.exec_dt,
                data=_load_json_object(args.payload),
            )
        output = {"path": str(path), "record_hash": json.loads(path.read_bytes())["record_hash"]}
    else:
        output = verify_ledger(args.root)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
