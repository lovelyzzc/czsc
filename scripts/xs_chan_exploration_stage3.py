"""Collect the Stage 3 XS-Chan forward exploration in an append-only ledger.

Stage 3 is deliberately separate from the V2.1 confirmation chain.  It freezes
one post-hoc mechanism hypothesis, records decision-time projections before
their labels exist, and refuses efficacy output until exactly 52 prospective
weeks have completed.

The three June 2026 dates supported by the current cache are rehearsal fixtures
only.  They are stored outside the prospective ledger and can never count.

Typical lifecycle::

    uv run --no-sync python scripts/xs_chan_exploration_stage3.py check
    uv run --no-sync python scripts/xs_chan_exploration_stage3.py init
    uv run --no-sync python scripts/xs_chan_exploration_stage3.py status

Decision and label commands are intentionally fail-closed.  A prospective
decision can only be appended after its official session close and before the
next official session open; a label can only be appended after its exit close.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import xs_chan_exploration_stage1 as stage1

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SPEC_PATH = SCRIPTS_DIR / "xs_chan_exploration_stage3.json"
SOURCE_PATH = Path(__file__).resolve()
ANCHOR_PATH = SCRIPTS_DIR / "xs_chan_exploration_stage3_anchor.json"
OUTPUT_ROOT = SCRIPTS_DIR / "_output" / "xs_chan_exploration_stage3"
V21_PROTOCOL_PATH = SCRIPTS_DIR / "xs_chan_protocol_v2_1.json"
V21_DOCUMENT_PATH = SCRIPTS_DIR / "XS_CHAN_PILOT_PROTOCOL_V2_1.md"
STAGE1_SPEC_PATH = SCRIPTS_DIR / "xs_chan_exploration_stage1.json"
STAGE1_MEMBERSHIP_PATH = SCRIPTS_DIR / "_output/xs_chan_exploration_stage1/buffered_memberships.parquet"
STAGE2_OUTPUT_ROOT = SCRIPTS_DIR / "_output/xs_chan_exploration_stage2"
RAW_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"

LEDGER_SCHEMA = "xs_chan_stage3_exploration_ledger_v1"
RECORD_SCHEMA = "xs_chan_stage3_exploration_record_v1"
ANCHOR_SCHEMA = "xs_chan_stage3_exploration_anchor_v1"
ZERO_HASH = "0" * 64
EXCHANGE_TIMEZONE = ZoneInfo("Asia/Shanghai")
RECORD_NAME = re.compile(r"^(?P<sequence>\d{6})_(?P<digest>[0-9a-f]{64})\.json$")
RECORD_KEYS = {
    "schema",
    "ledger_id",
    "sequence",
    "previous_hash",
    "record_type",
    "logical_event_key",
    "payload_sha256",
    "recorded_at_utc",
    "payload",
    "record_hash",
}
OUTCOME_KEY_FRAGMENTS = ("return", "difference", "direction", "nav", "win_rate", "confidence", "bootstrap")
DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "close",
    "turnover_rate",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "total_mv",
    "circ_mv",
    "free_share",
)
GENESIS_PAYLOAD_KEYS = {
    "schema",
    "study_id",
    "study_identity",
    "spec_physical_sha256",
    "spec_canonical_sha256",
    "collector_source_sha256",
    "protocol_anchor_git_commit",
    "first_prospective_decision_date",
    "rehearsal_dates_never_count",
    "confirmation_chain",
    "live_trading_authorized",
    "claim_boundary",
}
DECISION_PAYLOAD_KEYS = {
    "schema",
    "classification",
    "week_index",
    "decision_dt",
    "entry_dt",
    "exit_dt",
    "previous_decision_record_hash",
    "reference_manifest_sha256",
    "official_calendar_sha256",
    "official_session_prefix_sha256",
    "decision_path_sha256",
    "decision_path_object",
    "bridge_decision_dates",
    "proposals",
    "current_membership",
    "current_membership_sha256",
    "outcome_columns_forbidden",
}
PROPOSAL_KEYS = {
    "symbol",
    "proposal_order",
    "gate_passed",
    "regime",
    "factor_rank",
    "factor_score",
    "industry_code",
    "mcap_bucket",
    "ma_allowed",
}
LABEL_PAYLOAD_KEYS = {
    "schema",
    "decision_record_hash",
    "decision_dt",
    "entry_dt",
    "exit_dt",
    "raw_source_closure_sha256",
    "label_observation_sha256",
    "label_observation_object",
    "events",
    "event_count",
}
LABEL_EVENT_KEYS = {
    "symbol",
    "proposal_order",
    "regime",
    "factor_rank",
    "factor_decile",
    "entry_tradable",
    "exit_tradable",
    "return_5d",
    "missing_reason",
    "matched",
}
FINAL_PAYLOAD_KEYS = {
    "schema",
    "collection_chain_head",
    "evaluation_sha256",
    "evaluation_object",
    "status",
    "claim_boundary",
}


class Stage3Error(RuntimeError):
    """Raised when Stage 3 cannot preserve its frozen semantics."""


class Stage3ValidationError(Stage3Error):
    """Raised when immutable evidence or a ledger chain is invalid."""


class Stage3ConflictError(Stage3Error):
    """Raised when a logical event is retried with different content."""


class Stage3BlindError(Stage3Error):
    """Raised when efficacy output is requested before the 52-week lock."""


@dataclass(frozen=True)
class LedgerRecord:
    """One validated immutable record."""

    path: Path
    data: dict[str, Any]
    raw: bytes


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp with microseconds."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _reject_constant(token: str) -> Any:
    raise Stage3ValidationError(f"invalid JSON constant: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage3ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(payload: Any) -> bytes:
    """Return the strict canonical representation used by hashes and records."""

    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage3ValidationError(f"JSON object keys must be strings, got {type(key).__name__}")
            if key in result:
                raise Stage3ValidationError(f"duplicate JSON object key after normalization: {key}")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            raise Stage3ValidationError("non-finite floats are forbidden; normalize missing data explicitly")
        return number
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise Stage3ValidationError("non-finite floats are forbidden; normalize missing data explicitly")
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Read one strict JSON object, rejecting duplicate keys and non-finite values."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except OSError as exc:
        raise Stage3ValidationError(f"cannot read JSON: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage3ValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Stage3ValidationError(f"{path} must contain a JSON object")
    return payload


def sha256_bytes(raw: bytes) -> str:
    """Return a lowercase SHA256 digest."""

    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(actual_path: Path, expected: Any, label: str) -> None:
    if not actual_path.is_file():
        raise Stage3ValidationError(f"missing frozen {label}: {actual_path}")
    actual = sha256_file(actual_path)
    if actual != expected:
        raise Stage3ValidationError(f"{label} bytes changed: expected {expected}, got {actual}")


def load_and_validate_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    """Validate Stage 3 boundaries and every available frozen parent byte."""

    spec = read_json(path)
    if (
        spec.get("mode") != "FORWARD_EXPLORATION_ONLY"
        or spec.get("study_type") != "LOCAL_PROSPECTIVE_MECHANISM_REPLICATION"
        or spec.get("confirmation_chain") != "NOT_STARTED"
        or spec.get("live_trading_authorized") is not False
    ):
        raise Stage3ValidationError("Stage 3 must remain forward exploration with no confirmation or live authority")

    exclusion = spec.get("historical_exclusion")
    if not isinstance(exclusion, Mapping):
        raise Stage3ValidationError("historical_exclusion is missing")
    if exclusion.get("first_prospective_decision_date") != "2026-07-31":
        raise Stage3ValidationError("the first prospective decision must remain 2026-07-31")
    if exclusion.get("pre_genesis_rehearsal_decision_dates") != ["2026-06-12", "2026-06-18", "2026-06-26"]:
        raise Stage3ValidationError("the three pre-genesis rehearsal dates changed")
    if exclusion.get("rehearsal_classification") != "PRE_GENESIS_RETROSPECTIVE_PIPELINE_FIXTURE_NEVER_COUNTS":
        raise Stage3ValidationError("rehearsal fixtures must remain permanently non-counting")

    accrual = spec.get("accrual")
    if not isinstance(accrual, Mapping) or accrual.get("window_weeks") != 52:
        raise Stage3ValidationError("Stage 3 must use exactly 52 consecutive weeks")
    if accrual.get("efficacy_interim") is not False or accrual.get("extension_after_gate_failure") is not False:
        raise Stage3ValidationError("interim efficacy and sample extension must remain prohibited")
    no_tuning = spec.get("no_tuning")
    if not isinstance(no_tuning, Mapping) or not no_tuning or any(value is not False for value in no_tuning.values()):
        raise Stage3ValidationError("all Stage 3 tuning and suppression permissions must remain false")

    baseline = spec.get("frozen_v2_1_baseline")
    if not isinstance(baseline, Mapping):
        raise Stage3ValidationError("frozen_v2_1_baseline is missing")
    _require_hash(V21_PROTOCOL_PATH, baseline.get("protocol_physical_sha256"), "V2.1 protocol")
    protocol = read_json(V21_PROTOCOL_PATH)
    if sha256_bytes(canonical_json(protocol)) != baseline.get("protocol_canonical_sha256"):
        raise Stage3ValidationError("V2.1 canonical protocol changed")
    if protocol.get("protocol_id") != baseline.get("protocol_id"):
        raise Stage3ValidationError("V2.1 protocol ID changed")
    if protocol.get("protocol_status") != baseline.get("required_protocol_status"):
        raise Stage3ValidationError("V2.1 protocol lifecycle changed")
    _require_hash(V21_DOCUMENT_PATH, baseline.get("protocol_document_sha256"), "V2.1 protocol document")

    parent = spec.get("frozen_stage2_parent")
    if not isinstance(parent, Mapping):
        raise Stage3ValidationError("frozen_stage2_parent is missing")
    _require_hash(
        SCRIPTS_DIR / str(parent["spec_path"]).removeprefix("scripts/"), parent["spec_physical_sha256"], "Stage 2 spec"
    )
    parent_spec = read_json(SCRIPTS_DIR / str(parent["spec_path"]).removeprefix("scripts/"))
    if sha256_bytes(canonical_json(parent_spec)) != parent["spec_canonical_sha256"]:
        raise Stage3ValidationError("Stage 2 canonical spec changed")
    _require_hash(
        SCRIPTS_DIR / str(parent["source_path"]).removeprefix("scripts/"),
        parent["source_sha256"],
        "Stage 2 source",
    )
    _require_hash(
        SCRIPTS_DIR / str(parent["results_path"]).removeprefix("scripts/"),
        parent["results_sha256"],
        "Stage 2 result document",
    )
    manifest_path = STAGE2_OUTPUT_ROOT / f"STAGE2_{parent['study_identity']}" / "data_manifest.json"
    if manifest_path.exists():
        _require_hash(manifest_path, parent["local_manifest_sha256"], "Stage 2 local manifest")

    for item in spec.get("frozen_stage1_path_bootstrap", []):
        if not isinstance(item, Mapping):
            raise Stage3ValidationError("frozen_stage1_path_bootstrap must contain objects")
        frozen_path = REPO_ROOT / str(item["path"])
        _require_hash(frozen_path, item["sha256"], f"Stage 1 input {item.get('name')}")

    hypothesis = spec.get("primary_hypothesis")
    if not isinstance(hypothesis, Mapping) or hypothesis.get("id") != "S3H1_STATE3_HAS_FORWARD_MATCHED_INCREMENT":
        raise Stage3ValidationError("Stage 3 must retain its single state-3 primary question")
    if spec.get("statistics", {}).get("primary_endpoint_count") != 1:
        raise Stage3ValidationError("Stage 3 must retain exactly one primary endpoint")
    return spec


def study_identity(spec: Mapping[str, Any], source_path: Path = SOURCE_PATH) -> str:
    """Bind a ledger namespace to the canonical protocol and exact collector source."""

    material = b"xs-chan-stage3-exploration-identity-v1\0" + canonical_json(spec) + b"\0" + source_path.read_bytes()
    return sha256_bytes(material)


def resolve_ledger_root(
    spec: Mapping[str, Any] | None = None,
    *,
    source_path: Path = SOURCE_PATH,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """Resolve the content-bound ledger directory."""

    frozen = dict(spec) if spec is not None else load_and_validate_spec()
    return output_root / f"LEDGER_{study_identity(frozen, source_path)}"


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise Stage3ValidationError(f"git {' '.join(args)} failed: {detail}") from exc
    return result.stdout.strip()


def _validate_tracked_file_pushed(path: Path, *, deadline: datetime | None = None) -> str:
    """Prove that exact local bytes are committed on the configured upstream."""

    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    _git_output("ls-files", "--error-unmatch", "--", relative)
    if _git_output("status", "--porcelain=v1", "--", relative):
        raise Stage3ValidationError(f"tracked anchor has uncommitted changes: {relative}")
    commit = _git_output("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise Stage3ValidationError(f"tracked anchor has no commit: {relative}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise Stage3ValidationError(f"working anchor differs from committed bytes: {relative}")
    upstream = _git_output("rev-parse", "--verify", "@{upstream}")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, upstream],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Stage3ValidationError(f"anchor commit {commit} is not present on the configured upstream")
    if deadline is not None:
        committed_at = datetime.fromisoformat(_git_output("show", "-s", "--format=%cI", commit))
        if committed_at > deadline:
            raise Stage3ValidationError(
                f"anchor commit time {committed_at.isoformat()} is after deadline {deadline.isoformat()}"
            )
    return commit


def validate_protocol_anchor(spec: Mapping[str, Any]) -> str:
    """Validate exact protocol/source anchor bytes and their pushed Git commit."""

    anchor = read_json(ANCHOR_PATH)
    expected = {
        "schema": ANCHOR_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": study_identity(spec),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "spec_canonical_sha256": sha256_bytes(canonical_json(spec)),
        "source_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(SOURCE_PATH),
        "frozen_at_utc": spec["frozen_at_utc"],
        "first_prospective_decision_date": spec["historical_exclusion"]["first_prospective_decision_date"],
        "parent_git_commit": spec["freeze_evidence"]["parent_git_commit"],
        "external_timestamp_or_signature": False,
        "validity_requirement": spec["freeze_evidence"]["required_before_first_decision"],
    }
    if anchor != expected:
        raise Stage3ValidationError("tracked Stage 3 protocol anchor differs from exact frozen bytes")
    deadline = datetime.fromisoformat(spec["freeze_evidence"]["anchor_deadline"])
    return _validate_tracked_file_pushed(ANCHOR_PATH, deadline=deadline)


def validate_current_head_anchor_pushed(spec: Mapping[str, Any], records: Sequence[LedgerRecord]) -> str:
    """Require the current head commitment to be committed and pushed before appending."""

    if not records:
        raise Stage3ValidationError("cannot validate the head anchor of an empty chain")
    head = records[-1].data
    path = (
        SCRIPTS_DIR
        / "xs_chan_exploration_stage3_ledger_anchors"
        / (f"{head['sequence']:06d}_{head['record_hash']}.json")
    )
    if not path.exists():
        raise Stage3ValidationError(f"current ledger head has not been exported: {path}")
    anchor = read_json(path)
    if (
        anchor.get("study_identity") != study_identity(spec)
        or anchor.get("sequence") != head["sequence"]
        or anchor.get("record_hash") != head["record_hash"]
        or anchor.get("previous_hash") != head["previous_hash"]
    ):
        raise Stage3ValidationError("tracked ledger head anchor differs from the current local head")
    expected_replay_status = {
        "genesis": "PROTOCOL_SOURCE_REPLAYED",
        "decision_freeze": "DECISION_SOURCE_REPLAYED",
        "label_completion": "LABEL_SOURCE_REPLAYED",
        "final_evaluation": "FINAL_EVALUATION_REPLAYED",
    }[head["record_type"]]
    if anchor.get("source_replay", {}).get("status") != expected_replay_status:
        raise Stage3ValidationError("tracked ledger head anchor lacks the required source replay attestation")
    return _validate_tracked_file_pushed(path)


def _parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise Stage3ValidationError(f"invalid UTC timestamp: {value}") from exc
    else:
        raise Stage3ValidationError("recorded_at_utc must be a datetime or RFC3339 string")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise Stage3ValidationError("recorded_at_utc must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def _format_utc(value: str | datetime) -> str:
    return _parse_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


@contextmanager
def _exclusive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".append.lock").open("a+b") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _exclusive_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise Stage3ConflictError(f"immutable file already exists: {path}") from exc
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


def _record_body_hash(body: Mapping[str, Any]) -> str:
    return sha256_bytes(b"xs-chan-stage3-record-v1\0" + canonical_json(body))


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(b"xs-chan-stage3-payload-v1\0" + canonical_json(payload))


def _record_data(record: Mapping[str, Any] | LedgerRecord) -> Mapping[str, Any]:
    return record.data if isinstance(record, LedgerRecord) else record


def scan_records(root: Path) -> tuple[LedgerRecord, ...]:
    """Recompute and validate the complete chain; never trust a mutable HEAD."""

    records_dir = root / "records"
    if not records_dir.exists():
        return ()
    grouped: dict[int, list[tuple[Path, str]]] = {}
    for path in sorted(records_dir.iterdir()):
        if not path.is_file():
            raise Stage3ValidationError(f"unexpected entry in records directory: {path.name}")
        match = RECORD_NAME.fullmatch(path.name)
        if match is None:
            raise Stage3ValidationError(f"invalid record filename: {path.name}")
        grouped.setdefault(int(match.group("sequence")), []).append((path, match.group("digest")))
    if not grouped:
        return ()
    for sequence, candidates in grouped.items():
        if len(candidates) != 1:
            raise Stage3ValidationError(
                f"fork detected at sequence {sequence:06d}: {[path.name for path, _ in candidates]}"
            )
    expected = list(range(max(grouped) + 1))
    if sorted(grouped) != expected:
        raise Stage3ValidationError(f"record sequence gap: expected {expected}, got {sorted(grouped)}")

    validated: list[LedgerRecord] = []
    previous_hash = ZERO_HASH
    previous_time: datetime | None = None
    ledger_id: str | None = None
    logical_keys: set[str] = set()
    for sequence in expected:
        path, filename_hash = grouped[sequence][0]
        raw = path.read_bytes()
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage3ValidationError(f"invalid record JSON: {path.name}") from exc
        if not isinstance(payload, dict) or set(payload) != RECORD_KEYS:
            raise Stage3ValidationError(f"record schema mismatch: {path.name}")
        if raw != canonical_json(payload):
            raise Stage3ValidationError(f"record bytes are not canonical: {path.name}")
        if payload["schema"] != RECORD_SCHEMA or payload["sequence"] != sequence:
            raise Stage3ValidationError(f"record identity mismatch: {path.name}")
        if not isinstance(payload["ledger_id"], str) or len(payload["ledger_id"]) != 64:
            raise Stage3ValidationError(f"invalid ledger_id: {path.name}")
        if ledger_id is None:
            ledger_id = payload["ledger_id"]
        elif payload["ledger_id"] != ledger_id:
            raise Stage3ValidationError(f"ledger_id changed within chain: {path.name}")
        if payload["previous_hash"] != previous_hash:
            raise Stage3ValidationError(f"previous hash mismatch: {path.name}")
        if not isinstance(payload["logical_event_key"], str) or not payload["logical_event_key"]:
            raise Stage3ValidationError(f"invalid logical event key: {path.name}")
        if payload["logical_event_key"] in logical_keys:
            raise Stage3ValidationError(f"duplicate logical event key: {payload['logical_event_key']}")
        if _payload_hash(payload["payload"]) != payload["payload_sha256"]:
            raise Stage3ValidationError(f"payload hash mismatch: {path.name}")
        body = {key: value for key, value in payload.items() if key != "record_hash"}
        if _record_body_hash(body) != payload["record_hash"] or payload["record_hash"] != filename_hash:
            raise Stage3ValidationError(f"record hash mismatch: {path.name}")
        recorded_at = _parse_utc(payload["recorded_at_utc"])
        if previous_time is not None and recorded_at <= previous_time:
            raise Stage3ValidationError(f"record times are not strictly increasing: {path.name}")
        validated.append(LedgerRecord(path=path, data=payload, raw=raw))
        logical_keys.add(payload["logical_event_key"])
        previous_hash = payload["record_hash"]
        previous_time = recorded_at
    return tuple(validated)


def _preserve_failed_attempt(
    root: Path,
    *,
    reason: str,
    record_type: str,
    logical_event_key: str,
    payload: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
    attempted_at_utc: str,
) -> Path:
    evidence = {
        "schema": "xs_chan_stage3_failed_append_v1",
        "attempted_at_utc": attempted_at_utc,
        "reason": reason,
        "record_type": record_type,
        "logical_event_key": logical_event_key,
        "attempted_payload_sha256": _payload_hash(payload),
        "existing_record_hash": None if existing is None else existing["record_hash"],
        "existing_payload_sha256": None if existing is None else existing["payload_sha256"],
    }
    raw = canonical_json(evidence)
    digest = sha256_bytes(b"xs-chan-stage3-failed-append-v1\0" + raw)
    path = root / "failures" / f"{attempted_at_utc.replace(':', '')}_{digest}.json"
    _exclusive_write(path, raw)
    return path


def append_record(
    root: Path,
    record_type: str,
    logical_event_key: str,
    payload: Mapping[str, Any],
    recorded_at_utc: str | datetime | None = None,
    *,
    ledger_id: str | None = None,
) -> LedgerRecord:
    """Append exactly one immutable event, with idempotent exact retries."""

    if not isinstance(record_type, str) or not record_type:
        raise Stage3ValidationError("record_type must be a non-empty string")
    if not isinstance(logical_event_key, str) or not logical_event_key:
        raise Stage3ValidationError("logical_event_key must be a non-empty string")
    safe_payload = _json_safe(payload)
    if not isinstance(safe_payload, dict):
        raise Stage3ValidationError("record payload must be an object")
    attempted_at = _format_utc(recorded_at_utc or utc_now())

    with _exclusive_lock(root):
        records = scan_records(root)
        resolved_ledger_id = ledger_id or (
            records[0].data["ledger_id"] if records else sha256_bytes(str(root).encode())
        )
        if records and resolved_ledger_id != records[0].data["ledger_id"]:
            failure = _preserve_failed_attempt(
                root,
                reason="LEDGER_IDENTITY_MISMATCH",
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=safe_payload,
                existing=records[0].data,
                attempted_at_utc=attempted_at,
            )
            raise Stage3ConflictError(f"requested ledger_id differs from the chain; evidence={failure}")
        for existing_record in records:
            existing = existing_record.data
            if existing["logical_event_key"] != logical_event_key:
                continue
            if existing["record_type"] == record_type and existing["payload_sha256"] == _payload_hash(safe_payload):
                return existing_record
            failure = _preserve_failed_attempt(
                root,
                reason="SAME_LOGICAL_KEY_DIFFERENT_CONTENT",
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=safe_payload,
                existing=existing,
                attempted_at_utc=attempted_at,
            )
            raise Stage3ConflictError(f"logical event already exists with different content; evidence={failure}")
        if records and records[-1].data["record_type"] == "final_evaluation":
            failure = _preserve_failed_attempt(
                root,
                reason="TERMINAL_CHAIN_ALREADY_FINALIZED",
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=safe_payload,
                existing=records[-1].data,
                attempted_at_utc=attempted_at,
            )
            raise Stage3ConflictError(f"cannot append after final evaluation; evidence={failure}")

        sequence = len(records)
        previous_hash = records[-1].data["record_hash"] if records else ZERO_HASH
        event_time = _parse_utc(attempted_at)
        if records and event_time <= _parse_utc(records[-1].data["recorded_at_utc"]):
            failure = _preserve_failed_attempt(
                root,
                reason="NON_MONOTONIC_RECORD_TIME",
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=safe_payload,
                existing=records[-1].data,
                attempted_at_utc=attempted_at,
            )
            raise Stage3ConflictError(f"record time is not after the current head; evidence={failure}")

        body = {
            "schema": RECORD_SCHEMA,
            "ledger_id": resolved_ledger_id,
            "sequence": sequence,
            "previous_hash": previous_hash,
            "record_type": record_type,
            "logical_event_key": logical_event_key,
            "payload_sha256": _payload_hash(safe_payload),
            "recorded_at_utc": _format_utc(event_time),
            "payload": safe_payload,
        }
        record_hash = _record_body_hash(body)
        record = {**body, "record_hash": record_hash}
        raw = canonical_json(record)
        path = root / "records" / f"{sequence:06d}_{record_hash}.json"
        _exclusive_write(path, raw)
        return LedgerRecord(path=path, data=record, raw=raw)


def initialize_ledger(
    spec: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    source_path: Path = SOURCE_PATH,
) -> LedgerRecord:
    """Create or exactly retry the Stage 3 genesis record."""

    frozen = dict(spec) if spec is not None else load_and_validate_spec()
    identity = study_identity(frozen, source_path)
    resolved_root = root or resolve_ledger_root(frozen, source_path=source_path)
    protocol_anchor_commit = validate_protocol_anchor(frozen)
    genesis = {
        "schema": LEDGER_SCHEMA,
        "study_id": frozen["study_id"],
        "study_identity": identity,
        "spec_physical_sha256": sha256_file(SPEC_PATH) if SPEC_PATH.exists() else None,
        "spec_canonical_sha256": sha256_bytes(canonical_json(frozen)),
        "collector_source_sha256": sha256_file(source_path),
        "protocol_anchor_git_commit": protocol_anchor_commit,
        "first_prospective_decision_date": frozen["historical_exclusion"]["first_prospective_decision_date"],
        "rehearsal_dates_never_count": frozen["historical_exclusion"]["pre_genesis_rehearsal_decision_dates"],
        "confirmation_chain": "NOT_STARTED",
        "live_trading_authorized": False,
        "claim_boundary": frozen["claim_boundary"],
    }
    return append_record(
        resolved_root,
        "genesis",
        f"genesis:{frozen['study_id']}",
        genesis,
        ledger_id=identity,
    )


def classify_decision_date(spec: Mapping[str, Any], decision_date: str | pd.Timestamp) -> str:
    """Classify a date without allowing any pre-genesis observation to count."""

    date_text = pd.Timestamp(decision_date).date().isoformat()
    exclusion = spec["historical_exclusion"]
    if date_text in set(exclusion["pre_genesis_rehearsal_decision_dates"]):
        return exclusion["rehearsal_classification"]
    if date_text < exclusion["first_prospective_decision_date"]:
        return "HISTORICAL_EXCLUDED_NEVER_COUNTS"
    return "PROSPECTIVE_COUNTED_CANDIDATE"


def build_forward_schedule(
    spec: Mapping[str, Any],
    official_sessions: Sequence[str | pd.Timestamp],
) -> pd.DataFrame:
    """Build the immutable first-52-week schedule from official sessions only."""

    calendar = pd.DatetimeIndex(pd.to_datetime(list(official_sessions))).normalize().sort_values().unique()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise Stage3ValidationError("official session calendar must be non-empty, unique and increasing")
    first = pd.Timestamp(spec["historical_exclusion"]["first_prospective_decision_date"])
    table = pd.DataFrame({"session": calendar, "session_no": np.arange(len(calendar), dtype=np.int32)})
    table["week"] = table["session"].dt.to_period("W-FRI")
    decisions = (
        table.loc[table["session"].ge(first)]
        .groupby("week", sort=True, observed=True)["session_no"]
        .max()
        .astype(int)
        .tolist()
    )
    rows: list[dict[str, Any]] = []
    for session_no in decisions:
        if session_no + 6 >= len(calendar):
            continue
        decision_dt = calendar[session_no]
        if not rows and decision_dt != first:
            raise Stage3ValidationError(
                f"official calendar does not establish frozen first decision {first.date()}: got {decision_dt.date()}"
            )
        rows.append(
            {
                "week_index": len(rows) + 1,
                "decision_dt": decision_dt,
                "entry_dt": calendar[session_no + 1],
                "exit_dt": calendar[session_no + 6],
            }
        )
        if len(rows) == int(spec["accrual"]["window_weeks"]):
            break
    return pd.DataFrame(rows, columns=["week_index", "decision_dt", "entry_dt", "exit_dt"])


def build_fc_week(
    selection: Mapping[str, Any],
    ranked_week: pd.DataFrame,
    previous_symbols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Advance the frozen FC 50/75 path, applying the gate only to new entries."""

    required = {"symbol", "factor_rank", "factor_score", "regime"}
    if missing := required - set(ranked_week):
        raise Stage3ValidationError(f"ranked week missing FC fields: {sorted(missing)}")
    if ranked_week.empty or ranked_week["symbol"].duplicated().any():
        raise Stage3ValidationError("ranked week must be non-empty and unique by symbol")
    week = ranked_week.sort_values(["factor_rank", "symbol"], kind="mergesort").reset_index(drop=True)
    indexed = week.set_index("symbol", drop=False)
    eligible = set(indexed.index.astype(str))
    target_size = int(selection["target_size"])
    retention_rank = int(selection["retention_max_rank"])
    allowed = set(map(int, selection["allowed_chan_regimes"]))
    previous = set(map(str, previous_symbols))
    retained = {
        symbol for symbol in previous if symbol in eligible and int(indexed.at[symbol, "factor_rank"]) <= retention_rank
    }
    slots = target_size - len(retained)
    if slots < 0:
        raise Stage3ValidationError("retained FC membership exceeds target size")
    candidates = [str(symbol) for symbol in week["symbol"] if str(symbol) not in retained][:slots]
    proposal_rows: list[dict[str, Any]] = []
    accepted: list[str] = []
    for order, symbol in enumerate(candidates, 1):
        row = indexed.loc[symbol]
        regime = None if pd.isna(row["regime"]) else int(row["regime"])
        gate_passed = regime in allowed
        proposal = {
            "symbol": symbol,
            "proposal_order": order,
            "gate_passed": gate_passed,
            "regime": regime,
            "factor_rank": int(row["factor_rank"]),
            "factor_score": float(row["factor_score"]),
        }
        for column in ("industry_code", "mcap_bucket", "ma_allowed"):
            if column in row.index:
                proposal[column] = _json_safe(row[column])
        proposal_rows.append(proposal)
        if gate_passed:
            accepted.append(symbol)
    current = retained | set(accepted)
    membership_rows = [
        {
            "symbol": symbol,
            "membership_role": "retained" if symbol in retained else "new_entry",
            "factor_rank": int(indexed.at[symbol, "factor_rank"]),
            "factor_score": float(indexed.at[symbol, "factor_score"]),
            "regime": None if pd.isna(indexed.at[symbol, "regime"]) else int(indexed.at[symbol, "regime"]),
        }
        for symbol in sorted(current, key=lambda item: (int(indexed.at[item, "factor_rank"]), item))
    ]
    return pd.DataFrame(membership_rows), pd.DataFrame(proposal_rows)


def _store_canonical_object(root: Path, category: str, payload: Any) -> tuple[str, Path]:
    raw = canonical_json(payload)
    digest = sha256_bytes(raw)
    path = root / "objects" / category / f"{digest}.json"
    if path.exists():
        if path.read_bytes() != raw:
            raise Stage3ValidationError(f"content-addressed object bytes conflict: {path}")
    else:
        _exclusive_write(path, raw)
    return digest, path


def _frame_records(frame: pd.DataFrame, sort_by: Sequence[str]) -> list[dict[str, Any]]:
    ordered = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for row in ordered.to_dict("records"):
        normalized = {
            key: None
            if value is None
            or value is pd.NA
            or value is pd.NaT
            or (isinstance(value, (float, np.floating)) and pd.isna(value))
            else value
            for key, value in row.items()
        }
        records.append(_json_safe(normalized))
    return records


def _api_call(call: Any, label: str, retries: int = 4) -> pd.DataFrame:
    import time

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            result = call()
            if not isinstance(result, pd.DataFrame):
                raise Stage3Error(f"{label} returned a non-DataFrame response")
            return result
        except Exception as exc:  # pragma: no cover - network failure modes vary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise Stage3Error(f"{label} failed after {retries} attempts: {last_error}") from last_error


def validate_raw_source_closure(state_manifest_path: Path) -> dict[str, Any]:
    """Prove that every current raw parquet equals one state-manifest source."""

    state_manifest = read_json(state_manifest_path)
    sources = state_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise Stage3ValidationError("state manifest has no raw source inventory")
    expected_names = {str(row.get("name")) for row in sources if isinstance(row, Mapping)}
    actual_names = {path.name for path in RAW_DIR.glob("*.parquet")}
    if expected_names != actual_names:
        raise Stage3ValidationError(
            "raw source closure differs from state manifest: "
            f"missing={sorted(expected_names - actual_names)[:5]} "
            f"extra={sorted(actual_names - expected_names)[:5]}"
        )
    files: list[dict[str, Any]] = []
    for row in sorted(sources, key=lambda item: str(item["name"])):
        path = RAW_DIR / str(row["name"])
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(row["size"]) or digest != row["sha256"]:
            raise Stage3ValidationError(f"raw source differs from state manifest: {path.name}")
        files.append(
            {
                "name": path.name,
                "size": size,
                "sha256": digest,
                "min_dt": row.get("min_dt"),
                "max_dt": row.get("max_dt"),
            }
        )
    return {
        "schema": "xs_chan_stage3_raw_source_closure_v1",
        "state_manifest_sha256": sha256_file(state_manifest_path),
        "file_count": len(files),
        "coverage_max_dt": max(
            str(row["max_dt"]) for row in files if isinstance(row.get("max_dt"), str) and row["max_dt"]
        ),
        "files": files,
    }


def verify_raw_source_closure_current(closure: Mapping[str, Any]) -> None:
    """Verify a frozen closure against the raw files currently on disk."""

    _require_exact_keys(
        closure,
        {"schema", "state_manifest_sha256", "file_count", "coverage_max_dt", "files"},
        "raw source closure",
    )
    files = closure["files"]
    if closure["schema"] != "xs_chan_stage3_raw_source_closure_v1" or closure["file_count"] != len(files):
        raise Stage3ValidationError("raw source closure identity or count mismatch")
    expected_names = {str(row["name"]) for row in files}
    actual_names = {path.name for path in RAW_DIR.glob("*.parquet")}
    if expected_names != actual_names:
        raise Stage3ValidationError("current raw source names differ from the frozen closure")
    for row in files:
        path = RAW_DIR / str(row["name"])
        if path.stat().st_size != int(row["size"]) or sha256_file(path) != row["sha256"]:
            raise Stage3ValidationError(f"current raw source differs from frozen closure: {path.name}")


def fetch_reference_data(
    spec: Mapping[str, Any],
    decision_dates: Sequence[str | pd.Timestamp],
    *,
    root: Path | None = None,
    state_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Fetch one immutable official-calendar/daily-basic reference manifest."""

    import tinyshare as ts

    dates = sorted({pd.Timestamp(value).normalize() for value in decision_dates})
    if not dates:
        raise Stage3ValidationError("at least one decision date is required")
    resolved_root = root or resolve_ledger_root(spec)
    token = stage1.resolve_tinyshare_token()
    ts.set_token(token)
    pro = ts.pro_api()

    calendar_start = "20200101"
    calendar_end = (max(dates) + pd.Timedelta(days=45)).strftime("%Y%m%d")
    calendar = _api_call(
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=calendar_start,
            end_date=calendar_end,
            fields="exchange,cal_date,is_open,pretrade_date",
        ),
        f"trade_cal[SSE/{calendar_start}:{calendar_end}]",
    )
    required_calendar = {"exchange", "cal_date", "is_open", "pretrade_date"}
    if missing := required_calendar - set(calendar):
        raise Stage3ValidationError(f"official calendar missing fields: {sorted(missing)}")
    calendar = calendar[list(required_calendar)].copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="raise").astype(int)
    calendar = calendar.drop_duplicates(["exchange", "cal_date"], keep="last")
    calendar_records = _frame_records(calendar, ["cal_date", "exchange"])
    calendar_sha, calendar_path = _store_canonical_object(resolved_root, "official_calendar", calendar_records)

    daily_objects: dict[str, dict[str, Any]] = {}
    for decision_dt in dates:
        date_text = decision_dt.strftime("%Y%m%d")
        frame = _api_call(
            lambda date_text=date_text: pro.daily_basic(
                trade_date=date_text,
                fields=",".join(DAILY_BASIC_FIELDS),
            ),
            f"daily_basic[{date_text}]",
        )
        if missing := set(DAILY_BASIC_FIELDS) - set(frame):
            raise Stage3ValidationError(f"daily_basic[{date_text}] missing fields: {sorted(missing)}")
        frame = frame[list(DAILY_BASIC_FIELDS)].copy()
        records = _frame_records(frame, ["ts_code", "trade_date"])
        digest, object_path = _store_canonical_object(resolved_root, "daily_basic", records)
        daily_objects[decision_dt.date().isoformat()] = {
            "sha256": digest,
            "object_path": str(object_path.relative_to(resolved_root)),
            "rows": len(records),
        }

    if state_manifest_path is None:
        state_path, resolved_state_manifest = stage1.discover_state_cache()
    else:
        resolved_state_manifest = state_manifest_path.resolve()
        state_path = resolved_state_manifest.parent / "states.parquet"
        if not resolved_state_manifest.is_file() or not state_path.is_file():
            raise Stage3ValidationError("explicit state manifest must sit beside states.parquet")
    stage1_paths = stage1.StudyPaths(
        output_dir=OUTPUT_ROOT,
        reference_dir=stage1.REFERENCE_DIR,
        daily_basic_dir=stage1.DAILY_BASIC_DIR,
        state_path=state_path,
        state_manifest_path=resolved_state_manifest,
        industry_path=stage1.REFERENCE_DIR / "sw2021_l1_membership.parquet",
    )
    raw_closure = validate_raw_source_closure(stage1_paths.state_manifest_path)
    raw_closure_sha, raw_closure_path = _store_canonical_object(
        resolved_root,
        "raw_source_closure",
        raw_closure,
    )
    manifest = {
        "schema": "xs_chan_stage3_reference_manifest_v1",
        "study_id": spec["study_id"],
        "study_identity": study_identity(spec),
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "collector_source_sha256": sha256_file(SOURCE_PATH),
        "created_at_utc": utc_now(),
        "credential_recorded": False,
        "queries": {
            "official_calendar": {
                "exchange": "SSE",
                "start_date": calendar_start,
                "end_date": calendar_end,
                "fields": ["exchange", "cal_date", "is_open", "pretrade_date"],
            },
            "daily_basic_fields": list(DAILY_BASIC_FIELDS),
        },
        "official_calendar": {
            "sha256": calendar_sha,
            "object_path": str(calendar_path.relative_to(resolved_root)),
            "rows": len(calendar_records),
        },
        "daily_basic": daily_objects,
        "local_inputs": {
            "state_projection_path": str(stage1_paths.state_path),
            "state_projection_sha256": sha256_file(stage1_paths.state_path),
            "state_manifest_path": str(stage1_paths.state_manifest_path),
            "state_manifest_sha256": sha256_file(stage1_paths.state_manifest_path),
            "raw_source_closure_sha256": raw_closure_sha,
            "raw_source_closure_object": str(raw_closure_path.relative_to(resolved_root)),
            "raw_source_file_count": raw_closure["file_count"],
            "industry_membership_path": str(stage1_paths.industry_path),
            "industry_membership_sha256": sha256_file(stage1_paths.industry_path),
            "namechange_path": str(stage1.NAMECHANGE_PATH),
            "namechange_sha256": sha256_file(stage1.NAMECHANGE_PATH),
        },
    }
    manifest_sha, manifest_path = _store_canonical_object(resolved_root, "reference_manifest", manifest)
    return {**manifest, "manifest_sha256": manifest_sha}, manifest_path


def load_reference_manifest(root: Path, path: Path) -> tuple[dict[str, Any], pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Load a manifest and verify every content-addressed object it names."""

    manifest = read_json(path)
    if manifest.get("schema") != "xs_chan_stage3_reference_manifest_v1":
        raise Stage3ValidationError("unsupported Stage 3 reference manifest")
    current_spec = read_json(SPEC_PATH)
    if (
        manifest.get("study_identity") != study_identity(current_spec)
        or manifest.get("spec_physical_sha256") != sha256_file(SPEC_PATH)
        or manifest.get("collector_source_sha256") != sha256_file(SOURCE_PATH)
    ):
        raise Stage3ValidationError("reference manifest is not bound to the current Stage 3 collector")
    expected_manifest_hash = path.stem
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_hash):
        raise Stage3ValidationError("reference manifest filename must be its SHA256")
    if sha256_file(path) != expected_manifest_hash:
        raise Stage3ValidationError("reference manifest filename hash mismatch")

    calendar_info = manifest["official_calendar"]
    calendar_path = root / calendar_info["object_path"]
    _require_hash(calendar_path, calendar_info["sha256"], "official calendar object")
    calendar_records = json.loads(calendar_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    calendar = pd.DataFrame(calendar_records)
    official_sessions = pd.DatetimeIndex(
        pd.to_datetime(calendar.loc[pd.to_numeric(calendar["is_open"]).eq(1), "cal_date"], format="%Y%m%d")
    ).sort_values()
    if official_sessions.empty or official_sessions.has_duplicates:
        raise Stage3ValidationError("official calendar object has invalid open sessions")

    daily: dict[str, pd.DataFrame] = {}
    for decision_date, info in manifest["daily_basic"].items():
        object_path = root / info["object_path"]
        _require_hash(object_path, info["sha256"], f"daily_basic[{decision_date}] object")
        records = json.loads(object_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        frame = pd.DataFrame(records)
        if len(frame) != int(info["rows"]):
            raise Stage3ValidationError(f"daily_basic[{decision_date}] row count mismatch")
        daily[decision_date] = frame
    for label, suffix in (
        ("state projection", "state_projection"),
        ("state manifest", "state_manifest"),
        ("industry membership", "industry_membership"),
        ("namechange", "namechange"),
    ):
        _require_hash(
            Path(manifest["local_inputs"][f"{suffix}_path"]),
            manifest["local_inputs"][f"{suffix}_sha256"],
            label,
        )
    closure_path = root / manifest["local_inputs"]["raw_source_closure_object"]
    _require_hash(
        closure_path,
        manifest["local_inputs"]["raw_source_closure_sha256"],
        "raw source closure object",
    )
    return manifest, official_sessions, daily


def _normalise_daily_basic(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := set(DAILY_BASIC_FIELDS) - set(frame):
        raise Stage3ValidationError(f"daily-basic object missing fields: {sorted(missing)}")
    result = frame[list(DAILY_BASIC_FIELDS)].copy()
    result["symbol"] = result.pop("ts_code").astype(str)
    result["dt"] = pd.to_datetime(result.pop("trade_date"), format="%Y%m%d", errors="raise")
    if result.duplicated(["symbol", "dt"]).any():
        raise Stage3ValidationError("daily-basic object has duplicate symbol/date rows")
    for column in set(DAILY_BASIC_FIELDS) - {"ts_code", "trade_date"}:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    result["free_float_mcap"] = result["close"] * result["free_share"] * 10_000.0
    result["log_free_float_mcap"] = np.log(result["free_float_mcap"].where(result["free_float_mcap"] > 0))
    return result.drop(columns=["close"])


def build_decision_projections(
    spec: Mapping[str, Any],
    decision_dates: Sequence[str | pd.Timestamp],
    official_sessions: pd.DatetimeIndex,
    daily_basic_by_date: Mapping[str, pd.DataFrame],
    reference_manifest: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build decision-only ranked surfaces, filtering raw rows at each prefix."""

    dates = sorted({pd.Timestamp(value).normalize() for value in decision_dates})
    if not dates:
        raise Stage3ValidationError("no decision dates requested")
    missing_daily = [date.date().isoformat() for date in dates if date.date().isoformat() not in daily_basic_by_date]
    if missing_daily:
        raise Stage3ValidationError(f"reference manifest lacks daily-basic dates: {missing_daily}")
    if any(date not in official_sessions for date in dates):
        raise Stage3ValidationError("every decision date must be an official SSE session")

    stage1_spec = read_json(STAGE1_SPEC_PATH)
    causal_spec = copy.deepcopy(stage1_spec)
    causal_spec["sample"]["end_date"] = max(dates).date().isoformat()
    causal_spec["sample"]["minimum_complete_decision_weeks"] = 1
    causal_calendar = official_sessions[official_sessions <= max(dates)]
    dense = stage1.build_dense_feature_panel(causal_spec, causal_calendar)
    decision_date_values = [date.date() for date in dates]
    surface = dense.filter(pl.col("dt").is_in(decision_date_values)).select(
        "symbol",
        "dt",
        "first_observed_dt",
        "last_observed_dt",
        "adjusted_close",
        "mom_120_20",
        "lowvol_60",
        "sma20",
        "adv20",
        "valid_sessions_60",
        "market_sessions_since_first_observation",
        "distance_to_sma20",
        "ma_allowed",
    )

    daily = pd.concat(
        [_normalise_daily_basic(daily_basic_by_date[date.date().isoformat()]) for date in dates],
        ignore_index=True,
    )
    surface = surface.join(
        pl.from_pandas(daily).with_columns(pl.col("dt").cast(pl.Date)),
        on=["symbol", "dt"],
        how="left",
    )
    stage1_paths = stage1.StudyPaths(
        output_dir=OUTPUT_ROOT,
        reference_dir=OUTPUT_ROOT,
        daily_basic_dir=OUTPUT_ROOT,
        state_path=Path(reference_manifest["local_inputs"]["state_projection_path"]),
        state_manifest_path=Path(reference_manifest["local_inputs"]["state_manifest_path"]),
        industry_path=Path(reference_manifest["local_inputs"]["industry_membership_path"]),
    )
    surface = stage1._join_asof_interval(  # noqa: SLF001 - frozen research helper
        surface,
        stage1.load_industry_membership(stage1_paths),
        columns=("industry_code", "industry_name"),
        prefix="",
    )
    surface = stage1._join_asof_interval(  # noqa: SLF001 - frozen research helper
        surface,
        stage1.load_namechange(),
        columns=("active_st_name",),
        prefix="",
    ).with_columns(
        pl.col("active_st_name").fill_null(False),
        pl.col("symbol").map_elements(stage1.infer_board, return_dtype=pl.String).alias("board"),
    )
    states = (
        pl.scan_parquet(stage1_paths.state_path)
        .filter(pl.col("dt").cast(pl.Date).is_in(decision_date_values))
        .select(pl.col("symbol").cast(pl.String), pl.col("dt").cast(pl.Date), pl.col("regime").cast(pl.Int8))
        .collect(engine="streaming")
    )
    if states.select(pl.struct(["symbol", "dt"]).is_duplicated().any()).item():
        raise Stage3ValidationError("state projection has duplicate decision keys")
    surface = surface.join(states, on=["symbol", "dt"], how="left")

    selection = stage1_spec["selection"]
    finite = pl.all_horizontal(
        pl.col(column).is_finite()
        for column in ("mom_120_20", "lowvol_60", "sma20", "adv20", "free_float_mcap", "adjusted_close")
    )
    surface = surface.with_columns(
        (
            (pl.col("market_sessions_since_first_observation") >= int(selection["min_market_sessions_since_listing"]))
            & (pl.col("valid_sessions_60") >= int(selection["min_valid_sessions_60"]))
            & (pl.col("adv20") >= float(selection["min_adv20_thousand_cny"]))
            & finite
            & pl.col("industry_code").is_not_null()
            & pl.col("regime").is_not_null()
            & ~pl.col("board").is_in(list(map(str, selection["exclude_boards"])))
            & ~pl.col("active_st_name")
        ).alias("eligible")
    )
    frame = surface.to_pandas().rename(columns={"dt": "decision_dt"})
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"])
    frame["dt"] = frame["decision_dt"]
    ranked = stage1.rank_weekly_surface(stage1_spec, frame)
    forbidden = [column for column in ranked if any(fragment in column.lower() for fragment in OUTCOME_KEY_FRAGMENTS)]
    if forbidden:
        raise Stage3ValidationError(f"decision projection contains forbidden outcome columns: {forbidden}")
    return {
        date.date().isoformat(): week.reset_index(drop=True)
        for date, week in ranked.groupby("decision_dt", sort=True, observed=True)
    }


def load_initial_fc_membership() -> list[str]:
    """Load the exact frozen FC path state at the last Stage 1 decision."""

    frame = pd.read_parquet(
        STAGE1_MEMBERSHIP_PATH,
        columns=["decision_dt", "arm", "symbol", "factor_rank"],
    )
    selected = frame[
        frame["arm"].eq("FC") & pd.to_datetime(frame["decision_dt"]).eq(pd.Timestamp("2026-06-05"))
    ].sort_values(["factor_rank", "symbol"], kind="mergesort")
    if selected.empty or selected["symbol"].duplicated().any():
        raise Stage3ValidationError("frozen 2026-06-05 FC membership is missing or duplicated")
    return selected["symbol"].astype(str).tolist()


def _official_weekly_decisions(
    official_sessions: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    table = pd.DataFrame({"session": official_sessions})
    table["week"] = table["session"].dt.to_period("W-FRI")
    result = (
        table.loc[table["session"].between(pd.Timestamp(start), pd.Timestamp(end))]
        .groupby("week", sort=True, observed=True)["session"]
        .max()
        .tolist()
    )
    return [pd.Timestamp(value) for value in result]


def _decision_object_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "symbol",
        "decision_dt",
        "factor_rank",
        "factor_score",
        "factor_bucket",
        "mcap_bucket",
        "regime",
        "industry_code",
        "industry_name",
        "board",
        "adjusted_close",
        "mom_120_20",
        "lowvol_60",
        "sma20",
        "adv20",
        "free_float_mcap",
        "log_free_float_mcap",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "distance_to_sma20",
        "ma_allowed",
    ]
    present = [column for column in columns if column in frame]
    records = _frame_records(frame[present], ["factor_rank", "symbol"])
    for row in records:
        for column in ("factor_rank", "factor_bucket", "mcap_bucket", "regime"):
            if column in row and row[column] is not None:
                row[column] = int(row[column])
        if "ma_allowed" in row:
            row["ma_allowed"] = bool(row["ma_allowed"])
    return records


def build_fc_path_bundle(
    spec: Mapping[str, Any],
    decision_dates: Sequence[str | pd.Timestamp],
    projections: Mapping[str, pd.DataFrame],
    *,
    initial_symbols: Sequence[str] | None = None,
    initial_decision_date: str = "2026-06-05",
) -> dict[str, Any]:
    """Advance an ordered sequence of FC decisions and bind every projection."""

    selection = read_json(STAGE1_SPEC_PATH)["selection"]
    initial_membership = list(initial_symbols) if initial_symbols is not None else load_initial_fc_membership()
    previous = list(initial_membership)
    weeks: list[dict[str, Any]] = []
    for decision_dt in sorted(pd.Timestamp(value).normalize() for value in decision_dates):
        date_text = decision_dt.date().isoformat()
        if date_text not in projections:
            raise Stage3ValidationError(f"missing decision projection: {date_text}")
        ranked = projections[date_text]
        members, proposals = build_fc_week(selection, ranked, previous)
        projection_rows = _decision_object_rows(ranked)
        weeks.append(
            {
                "decision_dt": date_text,
                "classification": classify_decision_date(spec, date_text),
                "previous_membership": sorted(previous),
                "previous_membership_sha256": sha256_bytes(canonical_json(sorted(previous))),
                "projection_sha256": sha256_bytes(canonical_json(projection_rows)),
                "projection": projection_rows,
                "eligible_rows": len(ranked),
                "memberships": _frame_records(members, ["factor_rank", "symbol"]),
                "proposals": _frame_records(proposals, ["proposal_order", "symbol"]),
            }
        )
        previous = members["symbol"].astype(str).tolist() if not members.empty else []
    return {
        "schema": "xs_chan_stage3_fc_path_bundle_v1",
        "initial_decision_date": initial_decision_date,
        "initial_membership": sorted(initial_membership),
        "weeks": weeks,
        "final_membership": sorted(previous),
    }


def _session_after(official_sessions: pd.DatetimeIndex, value: str | pd.Timestamp, offset: int) -> pd.Timestamp:
    location = official_sessions.get_indexer([pd.Timestamp(value)])
    if len(location) != 1 or location[0] < 0 or location[0] + offset >= len(official_sessions):
        raise Stage3ValidationError(f"official calendar cannot resolve session offset {offset} after {value}")
    return pd.Timestamp(official_sessions[location[0] + offset])


def _decision_window(
    decision_dt: str | pd.Timestamp,
    entry_dt: str | pd.Timestamp,
) -> tuple[datetime, datetime]:
    start = pd.Timestamp(decision_dt).date()
    end = pd.Timestamp(entry_dt).date()
    return (
        datetime.combine(start, datetime.min.time(), tzinfo=EXCHANGE_TIMEZONE).replace(hour=15),
        datetime.combine(end, datetime.min.time(), tzinfo=EXCHANGE_TIMEZONE).replace(hour=9, minute=30),
    )


def _label_not_before(exit_dt: str | pd.Timestamp) -> datetime:
    return datetime.combine(
        pd.Timestamp(exit_dt).date(),
        datetime.min.time(),
        tzinfo=EXCHANGE_TIMEZONE,
    ).replace(hour=15)


def validate_decision_timing(
    decision_dt: str | pd.Timestamp,
    entry_dt: str | pd.Timestamp,
    recorded_at_utc: str | datetime,
) -> None:
    """Require a freeze after decision close and before the next open."""

    recorded = _parse_utc(recorded_at_utc)
    lower, upper = _decision_window(decision_dt, entry_dt)
    if not (lower <= recorded < upper):
        raise Stage3ValidationError(
            f"decision freeze time {recorded.isoformat()} is outside [{lower.isoformat()}, {upper.isoformat()})"
        )


def validate_label_timing(exit_dt: str | pd.Timestamp, recorded_at_utc: str | datetime) -> None:
    """Require labels to be appended only after the registered exit close."""

    recorded = _parse_utc(recorded_at_utc)
    lower = _label_not_before(exit_dt)
    if recorded < lower:
        raise Stage3ValidationError(f"label time {recorded.isoformat()} precedes exit close {lower.isoformat()}")


def _contains_outcome_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(fragment in str(key).lower() for fragment in OUTCOME_KEY_FRAGMENTS) or _contains_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome_key(item) for item in value)
    return False


def append_decision_freeze(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
    decision_date: str | pd.Timestamp,
) -> LedgerRecord:
    """Build and append one causally timed prospective decision."""

    date_text = pd.Timestamp(decision_date).date().isoformat()
    if classify_decision_date(spec, date_text) != "PROSPECTIVE_COUNTED_CANDIDATE":
        raise Stage3ValidationError(f"{date_text} is not eligible for the prospective ledger")
    records = scan_records(root)
    if not records or records[0].data["record_type"] != "genesis":
        raise Stage3ValidationError("prospective ledger must be initialized first")
    validate_record_semantics(spec, records, root=root)
    validate_current_head_anchor_pushed(spec, records)
    manifest, official_sessions, daily = load_reference_manifest(root, reference_manifest_path)
    schedule = build_forward_schedule(spec, official_sessions)
    row = schedule[schedule["decision_dt"].eq(pd.Timestamp(date_text))]
    if len(row) != 1:
        raise Stage3ValidationError(f"{date_text} is not a complete frozen prospective week")
    schedule_row = row.iloc[0]
    now = utc_now()
    validate_decision_timing(date_text, schedule_row["entry_dt"], now)

    prior_decisions = _prospective_decisions(records)
    week_index = int(schedule_row["week_index"])
    if week_index != len(prior_decisions) + 1:
        raise Stage3ValidationError(f"expected prospective week {len(prior_decisions) + 1}, got {week_index}")
    if week_index == 1:
        path_dates = _official_weekly_decisions(official_sessions, "2026-06-06", date_text)
        initial = load_initial_fc_membership()
    else:
        path_dates = [pd.Timestamp(date_text)]
        initial = prior_decisions[-1]["payload"]["current_membership"]
    projections = build_decision_projections(spec, path_dates, official_sessions, daily, manifest)
    path_bundle = build_fc_path_bundle(
        spec,
        path_dates,
        projections,
        initial_symbols=initial,
        initial_decision_date="2026-06-05"
        if not prior_decisions
        else str(prior_decisions[-1]["payload"]["decision_dt"]),
    )
    path_sha, path_object = _store_canonical_object(root, "decision_path", path_bundle)
    current = path_bundle["weeks"][-1]
    exit_text = pd.Timestamp(schedule_row["exit_dt"]).date().isoformat()
    official_prefix = [
        value.date().isoformat() for value in official_sessions if value <= pd.Timestamp(schedule_row["exit_dt"])
    ]
    payload = {
        "schema": "xs_chan_stage3_decision_freeze_v1",
        "classification": "PROSPECTIVE_COUNTED",
        "week_index": week_index,
        "decision_dt": date_text,
        "entry_dt": pd.Timestamp(schedule_row["entry_dt"]).date().isoformat(),
        "exit_dt": exit_text,
        "previous_decision_record_hash": prior_decisions[-1]["record_hash"] if prior_decisions else None,
        "reference_manifest_sha256": reference_manifest_path.stem,
        "official_calendar_sha256": manifest["official_calendar"]["sha256"],
        "official_session_prefix_sha256": sha256_bytes(canonical_json(official_prefix)),
        "decision_path_sha256": path_sha,
        "decision_path_object": str(path_object.relative_to(root)),
        "bridge_decision_dates": [week["decision_dt"] for week in path_bundle["weeks"][:-1]],
        "proposals": current["proposals"],
        "current_membership": [row["symbol"] for row in current["memberships"]],
        "current_membership_sha256": sha256_bytes(
            canonical_json(sorted(row["symbol"] for row in current["memberships"]))
        ),
        "outcome_columns_forbidden": True,
    }
    if _contains_outcome_key({key: value for key, value in payload.items() if key != "outcome_columns_forbidden"}):
        raise Stage3ValidationError("decision payload contains a forbidden outcome key")
    return append_record(
        root,
        "decision_freeze",
        f"decision:{date_text}",
        payload,
        now,
        ledger_id=study_identity(spec),
    )


def read_label_observation_rows(
    proposals: Sequence[Mapping[str, Any]],
    entry_dt: str | pd.Timestamp,
    exit_dt: str | pd.Timestamp,
) -> list[dict[str, Any]]:
    """Read exact entry/exit rows for every proposal, retaining missing rows."""

    symbols = sorted({str(row["symbol"]) for row in proposals})
    dates = [pd.Timestamp(entry_dt).strftime("%Y%m%d"), pd.Timestamp(exit_dt).strftime("%Y%m%d")]
    observations = (
        pl.scan_parquet(str(RAW_DIR / "*.parquet"))
        .select(
            pl.col("ts_code").cast(pl.String).alias("symbol"),
            pl.col("trade_date").cast(pl.String),
            pl.col("open").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
        )
        .filter(pl.col("symbol").is_in(symbols) & pl.col("trade_date").is_in(dates))
        .collect(engine="streaming")
        .to_pandas()
    )
    if observations.duplicated(["symbol", "trade_date"]).any():
        raise Stage3ValidationError("raw label observations are duplicated")
    lookup = observations.set_index(["symbol", "trade_date"])
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        row: dict[str, Any] = {"symbol": symbol}
        for label, date_text in (("entry", dates[0]), ("exit", dates[1])):
            source = lookup.loc[(symbol, date_text)] if (symbol, date_text) in lookup.index else None
            row[label] = {
                "observed": source is not None,
                "open": None if source is None or not math.isfinite(float(source["open"])) else float(source["open"]),
                "vol": None if source is None or not math.isfinite(float(source["vol"])) else float(source["vol"]),
                "amount": None
                if source is None or not math.isfinite(float(source["amount"]))
                else float(source["amount"]),
            }
        rows.append(row)
    return rows


def capture_label_observations(
    proposals: Sequence[Mapping[str, Any]],
    entry_dt: str | pd.Timestamp,
    exit_dt: str | pd.Timestamp,
    *,
    state_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture exact raw observations plus a verified whole-source closure."""

    rows = read_label_observation_rows(proposals, entry_dt, exit_dt)
    if state_manifest_path is None:
        _, resolved_state_manifest = stage1.discover_state_cache()
    else:
        resolved_state_manifest = state_manifest_path.resolve()
    closure = validate_raw_source_closure(resolved_state_manifest)
    if pd.Timestamp(closure["coverage_max_dt"]) < pd.Timestamp(exit_dt):
        raise Stage3ValidationError("raw source closure does not yet cover the frozen label exit")
    snapshot = {
        "schema": "xs_chan_stage3_label_observation_snapshot_v1",
        "entry_dt": pd.Timestamp(entry_dt).date().isoformat(),
        "exit_dt": pd.Timestamp(exit_dt).date().isoformat(),
        "state_manifest_sha256": sha256_file(resolved_state_manifest),
        "observations": rows,
    }
    return snapshot, closure


def derive_label_events(
    proposals: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Deterministically derive labels from one frozen observation snapshot."""

    observations = snapshot.get("observations")
    if not isinstance(observations, list):
        raise Stage3ValidationError("label observation snapshot has no observations")
    by_symbol = {
        str(row["symbol"]): row
        for row in observations
        if isinstance(row, Mapping) and isinstance(row.get("symbol"), str)
    }
    if len(by_symbol) != len(observations):
        raise Stage3ValidationError("label observation symbols must be unique")
    events: list[dict[str, Any]] = []
    edges = [0, 10, 20, 30, 40, 50]
    for proposal in proposals:
        symbol = str(proposal["symbol"])
        observation = by_symbol.get(symbol)
        entry = observation.get("entry") if observation else None
        exit_row = observation.get("exit") if observation else None
        entry_tradable = bool(
            isinstance(entry, Mapping)
            and entry.get("observed") is True
            and entry.get("open") is not None
            and entry.get("amount") is not None
            and float(entry["amount"]) > 0
            and entry.get("vol") is not None
            and float(entry["vol"]) > 0
        )
        exit_tradable = bool(
            isinstance(exit_row, Mapping)
            and exit_row.get("observed") is True
            and exit_row.get("open") is not None
            and exit_row.get("amount") is not None
            and float(exit_row["amount"]) > 0
            and exit_row.get("vol") is not None
            and float(exit_row["vol"]) > 0
        )
        event_return = (
            float(float(exit_row["open"]) / float(entry["open"]) - 1.0) if entry_tradable and exit_tradable else None
        )
        events.append(
            {
                "symbol": symbol,
                "proposal_order": int(proposal["proposal_order"]),
                "regime": proposal.get("regime"),
                "factor_rank": int(proposal["factor_rank"]),
                "factor_decile": _factor_decile(proposal["factor_rank"], edges),
                "entry_tradable": entry_tradable,
                "exit_tradable": exit_tradable,
                "return_5d": event_return,
                "missing_reason": None
                if event_return is not None
                else ("ENTRY_UNTRADABLE_OR_MISSING" if not entry_tradable else "EXIT_UNTRADABLE_OR_MISSING"),
                "matched": None,
            }
        )
    control_deciles = {
        event["factor_decile"]
        for event in events
        if event["regime"] != 3 and event["entry_tradable"] and event["return_5d"] is not None
    }
    for event in events:
        if event["regime"] == 3 and event["entry_tradable"] and event["return_5d"] is not None:
            event["matched"] = event["factor_decile"] in control_deciles
    return events


def build_label_events(
    proposals: Sequence[Mapping[str, Any]],
    entry_dt: str | pd.Timestamp,
    exit_dt: str | pd.Timestamp,
) -> list[dict[str, Any]]:
    """Compatibility wrapper used by retrospective diagnostics."""

    snapshot, _ = capture_label_observations(proposals, entry_dt, exit_dt)
    return derive_label_events(proposals, snapshot)


def append_label_completion(
    spec: Mapping[str, Any],
    root: Path,
    decision_date: str | pd.Timestamp,
    *,
    state_manifest_path: Path | None = None,
) -> LedgerRecord:
    """Append labels that reference, but never rewrite, a frozen decision."""

    date_text = pd.Timestamp(decision_date).date().isoformat()
    records = scan_records(root)
    decisions = [
        record.data
        for record in records
        if record.data["record_type"] == "decision_freeze" and record.data["payload"]["decision_dt"] == date_text
    ]
    if len(decisions) != 1:
        raise Stage3ValidationError(f"expected one frozen decision for {date_text}, found {len(decisions)}")
    decision = decisions[0]
    validate_record_semantics(spec, records, root=root)
    validate_current_head_anchor_pushed(spec, records)
    now = utc_now()
    validate_label_timing(decision["payload"]["exit_dt"], now)
    snapshot, closure = capture_label_observations(
        decision["payload"]["proposals"],
        decision["payload"]["entry_dt"],
        decision["payload"]["exit_dt"],
        state_manifest_path=state_manifest_path,
    )
    closure_sha, closure_path = _store_canonical_object(root, "raw_source_closure", closure)
    snapshot["raw_source_closure_sha256"] = closure_sha
    snapshot["raw_source_closure_object"] = str(closure_path.relative_to(root))
    observation_sha, observation_path = _store_canonical_object(root, "label_observation", snapshot)
    events = derive_label_events(decision["payload"]["proposals"], snapshot)
    payload = {
        "schema": "xs_chan_stage3_label_completion_v1",
        "decision_record_hash": decision["record_hash"],
        "decision_dt": date_text,
        "entry_dt": decision["payload"]["entry_dt"],
        "exit_dt": decision["payload"]["exit_dt"],
        "raw_source_closure_sha256": closure_sha,
        "label_observation_sha256": observation_sha,
        "label_observation_object": str(observation_path.relative_to(root)),
        "events": events,
        "event_count": len(events),
    }
    return append_record(
        root,
        "label_completion",
        f"label:{date_text}",
        payload,
        ledger_id=study_identity(spec),
    )


def rehearse_pipeline(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
) -> dict[str, Any]:
    """Run the three retrospective fixtures outside the prospective ledger."""

    manifest, official_sessions, daily = load_reference_manifest(root, reference_manifest_path)
    dates = [pd.Timestamp(value) for value in spec["historical_exclusion"]["pre_genesis_rehearsal_decision_dates"]]
    projections = build_decision_projections(spec, dates, official_sessions, daily, manifest)
    path_bundle = build_fc_path_bundle(spec, dates, projections)
    for week in path_bundle["weeks"]:
        if week["classification"] != "PRE_GENESIS_RETROSPECTIVE_PIPELINE_FIXTURE_NEVER_COUNTS":
            raise Stage3ValidationError("rehearsal classification invariant failed")
        decision_dt = pd.Timestamp(week["decision_dt"])
        entry_dt = _session_after(official_sessions, decision_dt, 1)
        exit_dt = _session_after(official_sessions, decision_dt, 6)
        week["entry_dt"] = entry_dt.date().isoformat()
        week["exit_dt"] = exit_dt.date().isoformat()
        snapshot, closure = capture_label_observations(
            week["proposals"],
            entry_dt,
            exit_dt,
            state_manifest_path=Path(manifest["local_inputs"]["state_manifest_path"]),
        )
        closure_sha, closure_path = _store_canonical_object(root, "raw_source_closure", closure)
        snapshot["raw_source_closure_sha256"] = closure_sha
        snapshot["raw_source_closure_object"] = str(closure_path.relative_to(root))
        observation_sha, observation_path = _store_canonical_object(root, "label_observation", snapshot)
        week["raw_source_closure_sha256"] = closure_sha
        week["label_observation_sha256"] = observation_sha
        week["label_observation_object"] = str(observation_path.relative_to(root))
        week["label_events"] = derive_label_events(week["proposals"], snapshot)
    bundle = {
        "schema": "xs_chan_stage3_rehearsal_bundle_v1",
        "study_id": spec["study_id"],
        "study_identity": study_identity(spec),
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "collector_source_sha256": sha256_file(SOURCE_PATH),
        "classification": "PRE_GENESIS_RETROSPECTIVE_PIPELINE_FIXTURE_NEVER_COUNTS",
        "reference_manifest_sha256": reference_manifest_path.stem,
        "prospective_week_count": 0,
        "path": path_bundle,
    }
    digest, object_path = _store_canonical_object(root, "rehearsal", bundle)
    treated = [
        event
        for week in path_bundle["weeks"]
        for event in week["label_events"]
        if event["regime"] == 3 and event["entry_tradable"]
    ]
    return {
        "status": "REHEARSAL_VALID_NEVER_COUNTS",
        "rehearsal_bundle_sha256": digest,
        "rehearsal_bundle_path": str(object_path),
        "decision_weeks": len(path_bundle["weeks"]),
        "proposal_events": sum(len(week["proposals"]) for week in path_bundle["weeks"]),
        "state3_events": len(treated),
        "matched_state3_events": sum(event["matched"] is True for event in treated),
        "prospective_week_count": 0,
        "efficacy_summary_emitted": False,
    }


def _require_exact_keys(payload: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        actual = set(payload) if isinstance(payload, Mapping) else set()
        raise Stage3ValidationError(
            f"{label} schema mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return payload


def _load_content_object(
    root: Path,
    category: str,
    digest: Any,
    relative_path: Any,
) -> Any:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise Stage3ValidationError(f"invalid {category} object digest")
    expected_relative = f"objects/{category}/{digest}.json"
    if relative_path != expected_relative:
        raise Stage3ValidationError(f"{category} object path is not content-derived")
    path = root / expected_relative
    _require_hash(path, digest, f"{category} object")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage3ValidationError(f"invalid {category} object JSON") from exc
    if path.read_bytes() != canonical_json(value):
        raise Stage3ValidationError(f"{category} object bytes are not canonical")
    return value


def _validate_decision_path_object(
    spec: Mapping[str, Any],
    root: Path,
    payload: Mapping[str, Any],
    previous_decision: Mapping[str, Any] | None,
) -> None:
    bundle = _load_content_object(
        root,
        "decision_path",
        payload["decision_path_sha256"],
        payload["decision_path_object"],
    )
    _require_exact_keys(
        bundle,
        {"schema", "initial_decision_date", "initial_membership", "weeks", "final_membership"},
        "decision path",
    )
    expected_initial_date = (
        "2026-06-05" if previous_decision is None else str(previous_decision["payload"]["decision_dt"])
    )
    if (
        bundle["schema"] != "xs_chan_stage3_fc_path_bundle_v1"
        or bundle["initial_decision_date"] != expected_initial_date
    ):
        raise Stage3ValidationError("decision path identity changed")
    expected_initial = (
        sorted(load_initial_fc_membership())
        if previous_decision is None
        else sorted(map(str, previous_decision["payload"]["current_membership"]))
    )
    if bundle["initial_membership"] != expected_initial:
        raise Stage3ValidationError("decision path initial membership differs from the frozen predecessor")
    weeks = bundle["weeks"]
    if not isinstance(weeks, list) or not weeks:
        raise Stage3ValidationError("decision path has no weeks")
    selection = read_json(STAGE1_SPEC_PATH)["selection"]
    previous = expected_initial
    projection_allowed = {
        "symbol",
        "decision_dt",
        "factor_rank",
        "factor_score",
        "factor_bucket",
        "mcap_bucket",
        "regime",
        "industry_code",
        "industry_name",
        "board",
        "adjusted_close",
        "mom_120_20",
        "lowvol_60",
        "sma20",
        "adv20",
        "free_float_mcap",
        "log_free_float_mcap",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "distance_to_sma20",
        "ma_allowed",
    }
    projection_required = {"symbol", "factor_rank", "factor_score", "regime"}
    for week in weeks:
        _require_exact_keys(
            week,
            {
                "decision_dt",
                "classification",
                "previous_membership",
                "previous_membership_sha256",
                "projection_sha256",
                "projection",
                "eligible_rows",
                "memberships",
                "proposals",
            },
            "decision path week",
        )
        if week["classification"] != classify_decision_date(spec, week["decision_dt"]):
            raise Stage3ValidationError("decision path week classification is self-reported incorrectly")
        if week["previous_membership"] != sorted(previous):
            raise Stage3ValidationError("decision path previous membership is discontinuous")
        if week["previous_membership_sha256"] != sha256_bytes(canonical_json(sorted(previous))):
            raise Stage3ValidationError("decision path previous membership hash mismatch")
        projection = week["projection"]
        if not isinstance(projection, list) or len(projection) != int(week["eligible_rows"]):
            raise Stage3ValidationError("decision projection row count mismatch")
        for projection_row in projection:
            keys = set(projection_row) if isinstance(projection_row, Mapping) else set()
            if not projection_required <= keys <= projection_allowed:
                raise Stage3ValidationError("decision projection violates its exact decision-time field allowlist")
        if week["projection_sha256"] != sha256_bytes(canonical_json(projection)):
            raise Stage3ValidationError("decision projection hash mismatch")
        ranked = pd.DataFrame(projection)
        members, proposals = build_fc_week(selection, ranked, previous)
        expected_members = _frame_records(members, ["factor_rank", "symbol"])
        expected_proposals = _frame_records(proposals, ["proposal_order", "symbol"])
        if canonical_json(week["memberships"]) != canonical_json(expected_members):
            raise Stage3ValidationError("decision memberships do not replay from projection")
        if canonical_json(week["proposals"]) != canonical_json(expected_proposals):
            raise Stage3ValidationError("decision proposals do not replay from projection")
        previous = [row["symbol"] for row in expected_members]
    if bundle["final_membership"] != sorted(previous):
        raise Stage3ValidationError("decision path final membership mismatch")
    current = weeks[-1]
    if current["decision_dt"] != payload["decision_dt"]:
        raise Stage3ValidationError("decision path does not terminate at the record date")
    if [week["decision_dt"] for week in weeks[:-1]] != payload["bridge_decision_dates"]:
        raise Stage3ValidationError("decision bridge dates differ from the path object")
    if canonical_json(current["proposals"]) != canonical_json(payload["proposals"]):
        raise Stage3ValidationError("decision proposals differ from the path object")
    current_membership = [row["symbol"] for row in current["memberships"]]
    if current_membership != payload["current_membership"]:
        raise Stage3ValidationError("current membership differs from the path object")


def validate_record_semantics(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | LedgerRecord],
    *,
    root: Path | None = None,
) -> None:
    """Replay exact schemas, calendar, objects and labels in addition to chain bytes."""

    rows = [_record_data(record) for record in records]
    if not rows or rows[0]["record_type"] != "genesis":
        raise Stage3ValidationError("record zero must be Stage 3 genesis")
    identity = study_identity(spec)
    if any(row.get("ledger_id") != identity for row in rows):
        raise Stage3ValidationError("ledger identity is not bound to the current Stage 3 spec and source")
    genesis = _require_exact_keys(rows[0]["payload"], GENESIS_PAYLOAD_KEYS, "genesis payload")
    expected_genesis = {
        "schema": LEDGER_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": identity,
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "spec_canonical_sha256": sha256_bytes(canonical_json(spec)),
        "collector_source_sha256": sha256_file(SOURCE_PATH),
        "protocol_anchor_git_commit": genesis["protocol_anchor_git_commit"],
        "first_prospective_decision_date": spec["historical_exclusion"]["first_prospective_decision_date"],
        "rehearsal_dates_never_count": spec["historical_exclusion"]["pre_genesis_rehearsal_decision_dates"],
        "confirmation_chain": "NOT_STARTED",
        "live_trading_authorized": False,
        "claim_boundary": spec["claim_boundary"],
    }
    if genesis != expected_genesis:
        raise Stage3ValidationError("genesis payload differs from exact frozen semantics")
    if root is not None and genesis["protocol_anchor_git_commit"] != validate_protocol_anchor(spec):
        raise Stage3ValidationError("genesis protocol anchor commit is not the current pushed anchor")

    decisions = _prospective_decisions(records)
    if len(decisions) > int(spec["accrual"]["window_weeks"]):
        raise Stage3ValidationError("prospective decision count exceeds the frozen 52-week window")
    decision_hashes: dict[str, Mapping[str, Any]] = {}
    previous_decision: Mapping[str, Any] | None = None
    previous_calendar: pd.DatetimeIndex | None = None
    previous_exit: pd.Timestamp | None = None
    if root is None and decisions:
        raise Stage3ValidationError("decision semantic replay requires the ledger object root")
    for expected_index, decision in enumerate(decisions, 1):
        payload = _require_exact_keys(decision["payload"], DECISION_PAYLOAD_KEYS, "decision payload")
        if payload["schema"] != "xs_chan_stage3_decision_freeze_v1":
            raise Stage3ValidationError("unsupported decision payload schema")
        date_text = str(payload["decision_dt"])
        if (
            payload["classification"] != "PROSPECTIVE_COUNTED"
            or classify_decision_date(spec, date_text) != "PROSPECTIVE_COUNTED_CANDIDATE"
            or payload["week_index"] != expected_index
        ):
            raise Stage3ValidationError("decision date/index/classification is not derived from the protocol")
        if expected_index == 1 and date_text != spec["historical_exclusion"]["first_prospective_decision_date"]:
            raise Stage3ValidationError("the prospective chain does not start at the frozen first date")
        expected_previous_hash = None if previous_decision is None else previous_decision["record_hash"]
        if payload["previous_decision_record_hash"] != expected_previous_hash:
            raise Stage3ValidationError("decision predecessor hash is discontinuous")
        for proposal in payload["proposals"]:
            _require_exact_keys(proposal, PROPOSAL_KEYS, "decision proposal")
        if payload["outcome_columns_forbidden"] is not True:
            raise Stage3ValidationError("decision payload must assert outcome exclusion")
        assert root is not None
        manifest_path = root / "objects/reference_manifest" / f"{payload['reference_manifest_sha256']}.json"
        manifest, official_sessions, _ = load_reference_manifest(root, manifest_path)
        if payload["official_calendar_sha256"] != manifest["official_calendar"]["sha256"]:
            raise Stage3ValidationError("decision official calendar hash differs from its manifest")
        schedule = build_forward_schedule(spec, official_sessions)
        schedule_row = schedule[schedule["week_index"].eq(expected_index)]
        if len(schedule_row) != 1:
            raise Stage3ValidationError("official calendar cannot derive the decision week index")
        schedule_row = schedule_row.iloc[0]
        expected_dates = (
            schedule_row["decision_dt"].date().isoformat(),
            schedule_row["entry_dt"].date().isoformat(),
            schedule_row["exit_dt"].date().isoformat(),
        )
        if (payload["decision_dt"], payload["entry_dt"], payload["exit_dt"]) != expected_dates:
            raise Stage3ValidationError("decision/entry/exit dates differ from official calendar replay")
        official_prefix = [value.date().isoformat() for value in official_sessions if value <= schedule_row["exit_dt"]]
        if payload["official_session_prefix_sha256"] != sha256_bytes(canonical_json(official_prefix)):
            raise Stage3ValidationError("official session prefix hash mismatch")
        if previous_calendar is not None and previous_exit is not None:
            old_prefix = previous_calendar[previous_calendar <= previous_exit]
            new_prefix = official_sessions[official_sessions <= previous_exit]
            if not old_prefix.equals(new_prefix):
                raise Stage3ValidationError("official calendar rewrote an already-bound session prefix")
        _validate_decision_path_object(spec, root, payload, previous_decision)
        expected_membership_hash = sha256_bytes(canonical_json(sorted(map(str, payload["current_membership"]))))
        if payload["current_membership_sha256"] != expected_membership_hash:
            raise Stage3ValidationError("current membership hash mismatch")
        validate_decision_timing(payload["decision_dt"], payload["entry_dt"], decision["recorded_at_utc"])
        decision_hashes[decision["record_hash"]] = decision
        previous_decision = decision
        previous_calendar = official_sessions
        previous_exit = pd.Timestamp(payload["exit_dt"])

    seen_labels: set[str] = set()
    final_evaluations = 0
    for position, row in enumerate(rows):
        if row["record_type"] in {"genesis", "decision_freeze"}:
            continue
        if row["record_type"] == "label_completion":
            payload = _require_exact_keys(row["payload"], LABEL_PAYLOAD_KEYS, "label payload")
            if payload["schema"] != "xs_chan_stage3_label_completion_v1":
                raise Stage3ValidationError("unsupported label payload schema")
            decision_hash = payload["decision_record_hash"]
            if decision_hash not in decision_hashes or decision_hash in seen_labels:
                raise Stage3ValidationError("label does not reference one unique prospective decision")
            decision = decision_hashes[decision_hash]
            if (
                payload["decision_dt"] != decision["payload"]["decision_dt"]
                or payload["entry_dt"] != decision["payload"]["entry_dt"]
                or payload["exit_dt"] != decision["payload"]["exit_dt"]
            ):
                raise Stage3ValidationError("label dates differ from the frozen decision")
            if payload["event_count"] != len(payload["events"]) or len(payload["events"]) != len(
                decision["payload"]["proposals"]
            ):
                raise Stage3ValidationError("label event population differs from frozen proposals")
            for event in payload["events"]:
                _require_exact_keys(event, LABEL_EVENT_KEYS, "label event")
            if root is None:
                raise Stage3ValidationError("label semantic replay requires the ledger object root")
            snapshot = _load_content_object(
                root,
                "label_observation",
                payload["label_observation_sha256"],
                payload["label_observation_object"],
            )
            _require_exact_keys(
                snapshot,
                {
                    "schema",
                    "entry_dt",
                    "exit_dt",
                    "state_manifest_sha256",
                    "raw_source_closure_sha256",
                    "raw_source_closure_object",
                    "observations",
                },
                "label observation snapshot",
            )
            if (
                snapshot["schema"] != "xs_chan_stage3_label_observation_snapshot_v1"
                or snapshot["entry_dt"] != payload["entry_dt"]
                or snapshot["exit_dt"] != payload["exit_dt"]
                or snapshot["raw_source_closure_sha256"] != payload["raw_source_closure_sha256"]
            ):
                raise Stage3ValidationError("label observation snapshot identity mismatch")
            closure = _load_content_object(
                root,
                "raw_source_closure",
                payload["raw_source_closure_sha256"],
                snapshot["raw_source_closure_object"],
            )
            _require_exact_keys(
                closure,
                {"schema", "state_manifest_sha256", "file_count", "coverage_max_dt", "files"},
                "raw source closure",
            )
            if (
                closure.get("schema") != "xs_chan_stage3_raw_source_closure_v1"
                or closure.get("state_manifest_sha256") != snapshot["state_manifest_sha256"]
                or closure.get("file_count") != len(closure.get("files", []))
                or pd.Timestamp(closure.get("coverage_max_dt")) < pd.Timestamp(payload["exit_dt"])
            ):
                raise Stage3ValidationError("label raw source closure identity mismatch")
            replayed_events = derive_label_events(decision["payload"]["proposals"], snapshot)
            if canonical_json(replayed_events) != canonical_json(payload["events"]):
                raise Stage3ValidationError("label events do not replay from frozen observations")
            validate_label_timing(payload["exit_dt"], row["recorded_at_utc"])
            seen_labels.add(decision_hash)
        elif row["record_type"] == "final_evaluation":
            final_evaluations += 1
            payload = _require_exact_keys(row["payload"], FINAL_PAYLOAD_KEYS, "final evaluation payload")
            if final_evaluations > 1 or position != len(rows) - 1:
                raise Stage3ValidationError("final evaluation must be the unique terminal record")
            if len(decisions) != int(spec["accrual"]["window_weeks"]) or len(seen_labels) != len(decisions):
                raise Stage3ValidationError("final evaluation precedes the complete 52-week lock")
            if payload["collection_chain_head"] != rows[position - 1]["record_hash"]:
                raise Stage3ValidationError("final evaluation does not bind the pre-evaluation chain head")
            if root is None:
                raise Stage3ValidationError("final evaluation replay requires the ledger object root")
            evaluation = _load_content_object(
                root,
                "final_evaluation",
                payload["evaluation_sha256"],
                payload["evaluation_object"],
            )
            replayed = _evaluate_locked_records(spec, records[:position])
            if canonical_json(evaluation) != canonical_json(replayed) or payload["status"] != replayed["status"]:
                raise Stage3ValidationError("final evaluation object does not replay from the locked chain")
        else:
            raise Stage3ValidationError(f"unsupported Stage 3 record type: {row['record_type']}")


def _prospective_decisions(records: Sequence[Mapping[str, Any] | LedgerRecord]) -> list[Mapping[str, Any]]:
    decisions = []
    for item in records:
        record = _record_data(item)
        if record.get("record_type") != "decision_freeze":
            continue
        payload = record.get("payload", {})
        if payload.get("classification") == "PROSPECTIVE_COUNTED":
            decisions.append(record)
    return decisions


def _completed_labels(records: Sequence[Mapping[str, Any] | LedgerRecord]) -> dict[str, Mapping[str, Any]]:
    labels: dict[str, Mapping[str, Any]] = {}
    for item in records:
        record = _record_data(item)
        if record.get("record_type") != "label_completion":
            continue
        payload = record.get("payload", {})
        decision_hash = payload.get("decision_record_hash")
        if isinstance(decision_hash, str):
            labels[decision_hash] = record
    return labels


def _blinded_status_report(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | LedgerRecord],
) -> dict[str, Any]:
    """Return the frozen blinded status vocabulary and no efficacy fields."""

    decisions = _prospective_decisions(records)
    labels = _completed_labels(records)
    treated_events = 0
    matched_events = 0
    presence_weeks = 0
    completed_weeks = 0
    incomplete_label_events = 0
    unique_symbols: set[str] = set()
    symbol_counts: dict[str, int] = {}
    for decision in decisions:
        label = labels.get(decision["record_hash"])
        if label is None:
            continue
        completed_weeks += 1
        week_treated = 0
        for event in label["payload"].get("events", []):
            if event.get("return_5d") is None:
                incomplete_label_events += 1
            if event.get("regime") != 3 or event.get("entry_tradable") is not True:
                continue
            treated_events += 1
            week_treated += 1
            symbol = str(event.get("symbol"))
            unique_symbols.add(symbol)
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            if event.get("matched") is True:
                matched_events += 1
        if week_treated:
            presence_weeks += 1
    total_weeks = int(spec["accrual"]["window_weeks"])
    lifecycle = "REGISTERED"
    if decisions:
        lifecycle = "COLLECTING"
    if len(decisions) >= int(spec["accrual"]["blinded_integrity_checkpoint_weeks"]):
        lifecycle = "BLINDED_26W_AUDIT"
    if len(decisions) == total_weeks and completed_weeks == total_weeks:
        lifecycle = "LOCKED_52W"
    maximum_share = max(symbol_counts.values(), default=0) / treated_events if treated_events else 0.0
    status = {
        "schema": "xs_chan_stage3_blinded_status_v1",
        "study_id": spec["study_id"],
        "lifecycle": lifecycle,
        "record_count": len(records),
        "decision_week_count": len(decisions),
        "completed_label_week_count": completed_weeks,
        "state3_event_count": treated_events,
        "state3_presence_week_count": presence_weeks,
        "matched_treated_count": matched_events,
        "matching_coverage": matched_events / treated_events if treated_events else 0.0,
        "incomplete_label_event_count": incomplete_label_events,
        "unique_state3_symbols": len(unique_symbols),
        "maximum_single_symbol_event_share": maximum_share,
        "prospective_weeks_remaining": max(0, total_weeks - len(decisions)),
        "efficacy_output": "BLINDED_UNTIL_EXACT_52_WEEK_LOCK",
    }
    forbidden = [key for key in status if any(fragment in key.lower() for fragment in OUTCOME_KEY_FRAGMENTS)]
    if forbidden:
        raise Stage3ValidationError(f"blinded status accidentally exposes outcome fields: {forbidden}")
    return status


def _factor_decile(factor_rank: Any, edges: Sequence[int]) -> int | None:
    if factor_rank is None or not math.isfinite(float(factor_rank)):
        return None
    rank = int(factor_rank)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True), 1):
        if int(lower) < rank <= int(upper):
            return index
    return None


def _weekly_differences(
    spec: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    edges = list(map(int, spec["primary_hypothesis"]["factor_rank_decile_edges"]))
    differences: list[float] = []
    treated_count = 0
    matched_count = 0
    presence_by_half = [0, 0]
    symbols: list[str] = []
    incomplete_label_events = 0
    for week_no, decision in enumerate(decisions):
        label = labels[decision["record_hash"]]
        strata: dict[int, dict[str, list[float]]] = {}
        for event in label["payload"].get("events", []):
            if event.get("return_5d") is None:
                incomplete_label_events += 1
            if event.get("entry_tradable") is not True or event.get("return_5d") is None:
                continue
            decile = _factor_decile(event.get("factor_rank"), edges)
            if decile is None:
                continue
            side = "treated" if event.get("regime") == 3 else "control"
            strata.setdefault(decile, {"treated": [], "control": []})[side].append(float(event["return_5d"]))
            if side == "treated":
                symbols.append(str(event.get("symbol")))
        week_sum = 0.0
        week_weight = 0
        week_treated = 0
        for values in strata.values():
            n_treated = len(values["treated"])
            if not n_treated:
                continue
            treated_count += n_treated
            week_treated += n_treated
            if not values["control"]:
                continue
            matched_count += n_treated
            week_sum += n_treated * (float(np.mean(values["treated"])) - float(np.mean(values["control"])))
            week_weight += n_treated
        if week_treated:
            presence_by_half[0 if week_no < 26 else 1] += 1
        differences.append(week_sum / week_weight if week_weight else 0.0)
    counts = pd.Series(symbols).value_counts() if symbols else pd.Series(dtype=int)
    diagnostics = {
        "treated_events": treated_count,
        "matched_treated_events": matched_count,
        "matching_coverage": matched_count / treated_count if treated_count else 0.0,
        "state3_presence_weeks": sum(presence_by_half),
        "presence_weeks_first_half": presence_by_half[0],
        "presence_weeks_second_half": presence_by_half[1],
        "unique_state3_symbols": int(counts.size),
        "maximum_single_symbol_event_share": float(counts.max() / counts.sum()) if not counts.empty else 0.0,
        "incomplete_label_events": incomplete_label_events,
    }
    return np.asarray(differences, dtype=float), diagnostics


def _newey_west_se(values: np.ndarray, lag: int) -> float:
    centered = values - values.mean()
    n = len(centered)
    long_run = float(np.dot(centered, centered) / n)
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    return math.sqrt(max(long_run, 0.0) / n)


def _circular_block_means(
    values: np.ndarray,
    *,
    block_weeks: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    n = len(values)
    blocks_needed = math.ceil(n / block_weeks)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    starts = rng.integers(0, n, size=(draws, blocks_needed))
    offsets = np.arange(block_weeks)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    samples = values[indices.reshape(draws, -1)[:, :n]]
    return samples.mean(axis=1)


def _evaluate_locked_records(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | LedgerRecord],
) -> dict[str, Any]:
    """Evaluate once, and only after the exact 52 decision/label lock exists."""

    decisions = _prospective_decisions(records)
    labels = _completed_labels(records)
    window = int(spec["accrual"]["window_weeks"])
    if len(decisions) != window or any(decision["record_hash"] not in labels for decision in decisions):
        raise Stage3BlindError("efficacy evaluation is forbidden before exactly 52 prospective weeks are complete")
    if [row["payload"]["week_index"] for row in decisions] != list(range(1, window + 1)):
        raise Stage3ValidationError("prospective week indices must be the exact contiguous sequence 1..52")

    weekly, diagnostics = _weekly_differences(spec, decisions, labels)
    accrual = spec["accrual"]
    if (
        diagnostics["incomplete_label_events"] > 0
        or diagnostics["state3_presence_weeks"] < int(accrual["minimum_state3_presence_weeks"])
        or diagnostics["treated_events"] < int(accrual["minimum_state3_events"])
        or diagnostics["unique_state3_symbols"] < int(accrual["minimum_unique_state3_symbols"])
        or diagnostics["maximum_single_symbol_event_share"] > float(accrual["maximum_single_symbol_event_share"])
        or diagnostics["presence_weeks_first_half"] < int(accrual["minimum_state3_presence_per_half"])
        or diagnostics["presence_weeks_second_half"] < int(accrual["minimum_state3_presence_per_half"])
    ):
        status = "INSUFFICIENT_DATA"
    elif diagnostics["matching_coverage"] < float(accrual["minimum_matched_treated_coverage"]):
        status = "INSUFFICIENT_MATCHING"
    else:
        statistics = spec["statistics"]
        mean = float(weekly.mean())
        se = _newey_west_se(weekly, int(statistics["hac_lag"]))
        quantile = float(statistics["one_sided_normal_quantile"])
        lower = mean - quantile * se
        upper = mean + quantile * se
        bootstrap = _circular_block_means(
            weekly,
            block_weeks=int(statistics["circular_block_weeks"]),
            draws=int(statistics["bootstrap_draws"]),
            seed=int(statistics["bootstrap_seed"]),
        )
        q05, q95 = np.quantile(bootstrap, [0.05, 0.95])
        sesoi = float(statistics["sesoi_weekly"])
        if lower > sesoi and float(q05) > sesoi and weekly[:26].mean() > 0 and weekly[26:].mean() > 0:
            status = "FORWARD_SUPPORTED_EXPLORATORY"
        elif upper < sesoi and float(q95) < sesoi:
            status = "FALSIFIED_FOR_REGISTERED_SESOI"
        else:
            status = "INCONCLUSIVE"
        diagnostics.update(
            {
                "weekly_mean": mean,
                "hac_se": se,
                "hac_one_sided_lower": lower,
                "hac_one_sided_upper": upper,
                "bootstrap_q05": float(q05),
                "bootstrap_q95": float(q95),
                "first_half_mean": float(weekly[:26].mean()),
                "second_half_mean": float(weekly[26:].mean()),
                "sesoi_weekly": sesoi,
            }
        )
    return {
        "schema": "xs_chan_stage3_final_evaluation_v1",
        "study_id": spec["study_id"],
        "status": status,
        "window_weeks": window,
        "decision_chain_head": decisions[-1]["record_hash"],
        "collection_chain_head": _record_data(records[-1])["record_hash"],
        "diagnostics": diagnostics,
        "claim_boundary": spec["claim_boundary"],
    }


def status_report(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | LedgerRecord],
    *,
    root: Path,
) -> dict[str, Any]:
    """Validate the chain and then return the operationally blinded status."""

    scanned = scan_records(root)
    supplied = [_record_data(record) for record in records]
    if len(scanned) != len(supplied) or any(
        canonical_json(actual.data) != canonical_json(provided)
        for actual, provided in zip(scanned, supplied, strict=True)
    ):
        raise Stage3ValidationError("supplied status records differ from the ledger stored on disk")
    validate_record_semantics(spec, scanned, root=root)
    return _blinded_status_report(spec, scanned)


def evaluate_records(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | LedgerRecord],
    *,
    root: Path,
) -> dict[str, Any]:
    """Validate every object and refuse efficacy output before the exact lock."""

    scanned = scan_records(root)
    supplied = [_record_data(record) for record in records]
    if len(scanned) != len(supplied) or any(
        canonical_json(actual.data) != canonical_json(provided)
        for actual, provided in zip(scanned, supplied, strict=True)
    ):
        raise Stage3ValidationError("supplied evaluation records differ from the ledger stored on disk")
    validate_record_semantics(spec, scanned, root=root)
    return _evaluate_locked_records(spec, scanned)


def finalize_evaluation(
    spec: Mapping[str, Any],
    root: Path,
) -> LedgerRecord:
    """Persist one content-addressed final evaluation and its immutable commit marker."""

    records = scan_records(root)
    validate_record_semantics(spec, records, root=root)
    existing = [record for record in records if record.data["record_type"] == "final_evaluation"]
    if existing:
        if len(existing) != 1:
            raise Stage3ValidationError("more than one final evaluation record exists")
        return existing[0]
    validate_current_head_anchor_pushed(spec, records)
    evaluation = _evaluate_locked_records(spec, records)
    digest, object_path = _store_canonical_object(root, "final_evaluation", evaluation)
    payload = {
        "schema": "xs_chan_stage3_final_evaluation_commit_v1",
        "collection_chain_head": records[-1].data["record_hash"],
        "evaluation_sha256": digest,
        "evaluation_object": str(object_path.relative_to(root)),
        "status": evaluation["status"],
        "claim_boundary": spec["claim_boundary"],
    }
    return append_record(
        root,
        "final_evaluation",
        "final-evaluation:52w",
        payload,
        ledger_id=study_identity(spec),
    )


def build_anchor(
    spec: Mapping[str, Any] | None = None,
    *,
    source_path: Path = SOURCE_PATH,
    path: Path = ANCHOR_PATH,
    parent_git_commit: str | None = None,
) -> dict[str, Any]:
    """Write the small tracked anchor that binds protocol and collector bytes."""

    frozen = dict(spec) if spec is not None else load_and_validate_spec()
    payload = {
        "schema": ANCHOR_SCHEMA,
        "study_id": frozen["study_id"],
        "study_identity": study_identity(frozen, source_path),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "spec_canonical_sha256": sha256_bytes(canonical_json(frozen)),
        "source_path": str(source_path.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(source_path),
        "frozen_at_utc": frozen["frozen_at_utc"],
        "first_prospective_decision_date": frozen["historical_exclusion"]["first_prospective_decision_date"],
        "parent_git_commit": parent_git_commit or frozen["freeze_evidence"]["parent_git_commit"],
        "external_timestamp_or_signature": False,
        "validity_requirement": frozen["freeze_evidence"]["required_before_first_decision"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise Stage3ConflictError(f"anchor already exists with different bytes: {path}")
    if not path.exists():
        _exclusive_write(path, raw)
    return payload


def validate_current_head_source_replay(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[LedgerRecord],
) -> dict[str, Any]:
    """Replay the current head from on-disk sources before exporting its Git anchor."""

    head = records[-1].data
    if head["record_type"] == "genesis":
        return {
            "status": "PROTOCOL_SOURCE_REPLAYED",
            "protocol_anchor_git_commit": head["payload"]["protocol_anchor_git_commit"],
        }
    if head["record_type"] == "decision_freeze":
        payload = head["payload"]
        manifest_path = root / "objects/reference_manifest" / f"{payload['reference_manifest_sha256']}.json"
        manifest, official_sessions, daily = load_reference_manifest(root, manifest_path)
        closure = _load_content_object(
            root,
            "raw_source_closure",
            manifest["local_inputs"]["raw_source_closure_sha256"],
            manifest["local_inputs"]["raw_source_closure_object"],
        )
        verify_raw_source_closure_current(closure)
        bundle = _load_content_object(
            root,
            "decision_path",
            payload["decision_path_sha256"],
            payload["decision_path_object"],
        )
        decision_dates = [week["decision_dt"] for week in bundle["weeks"]]
        projections = build_decision_projections(spec, decision_dates, official_sessions, daily, manifest)
        replayed = build_fc_path_bundle(
            spec,
            decision_dates,
            projections,
            initial_symbols=bundle["initial_membership"],
            initial_decision_date=bundle["initial_decision_date"],
        )
        if canonical_json(replayed) != canonical_json(bundle):
            raise Stage3ValidationError("current decision path does not replay from the frozen reference inputs")
        return {
            "status": "DECISION_SOURCE_REPLAYED",
            "raw_source_closure_sha256": manifest["local_inputs"]["raw_source_closure_sha256"],
            "decision_path_sha256": payload["decision_path_sha256"],
        }
    if head["record_type"] == "label_completion":
        payload = head["payload"]
        decisions = {
            record.data["record_hash"]: record.data
            for record in records
            if record.data["record_type"] == "decision_freeze"
        }
        decision = decisions[payload["decision_record_hash"]]
        snapshot = _load_content_object(
            root,
            "label_observation",
            payload["label_observation_sha256"],
            payload["label_observation_object"],
        )
        closure = _load_content_object(
            root,
            "raw_source_closure",
            payload["raw_source_closure_sha256"],
            snapshot["raw_source_closure_object"],
        )
        verify_raw_source_closure_current(closure)
        observations = read_label_observation_rows(
            decision["payload"]["proposals"],
            payload["entry_dt"],
            payload["exit_dt"],
        )
        if canonical_json(observations) != canonical_json(snapshot["observations"]):
            raise Stage3ValidationError("label observations do not replay from the frozen raw closure")
        return {
            "status": "LABEL_SOURCE_REPLAYED",
            "raw_source_closure_sha256": payload["raw_source_closure_sha256"],
            "label_observation_sha256": payload["label_observation_sha256"],
        }
    if head["record_type"] == "final_evaluation":
        return {
            "status": "FINAL_EVALUATION_REPLAYED",
            "evaluation_sha256": head["payload"]["evaluation_sha256"],
        }
    raise Stage3ValidationError(f"unsupported head type for source replay: {head['record_type']}")


def export_ledger_head_anchor(
    spec: Mapping[str, Any],
    root: Path,
    *,
    output_dir: Path = SCRIPTS_DIR / "xs_chan_exploration_stage3_ledger_anchors",
) -> tuple[dict[str, Any], Path]:
    """Export a unique tracked commitment to the current local ledger head."""

    records = scan_records(root)
    if not records:
        raise Stage3ValidationError("cannot anchor an empty ledger")
    validate_record_semantics(spec, records, root=root)
    head = records[-1].data
    source_replay = validate_current_head_source_replay(spec, root, records)
    payload = {
        "schema": "xs_chan_stage3_tracked_ledger_head_anchor_v1",
        "study_id": spec["study_id"],
        "study_identity": study_identity(spec),
        "ledger_schema": LEDGER_SCHEMA,
        "sequence": head["sequence"],
        "record_type": head["record_type"],
        "logical_event_key": head["logical_event_key"],
        "recorded_at_utc": head["recorded_at_utc"],
        "record_hash": head["record_hash"],
        "previous_hash": head["previous_hash"],
        "spec_physical_sha256": sha256_file(SPEC_PATH),
        "source_sha256": sha256_file(SOURCE_PATH),
        "source_replay": source_replay,
        "local_record_path": str(head["sequence"]).zfill(6) + "_" + head["record_hash"] + ".json",
        "external_timestamp_or_signature": False,
        "required_follow_up": "commit_and_push_this_unique_anchor_without_rewriting_prior_anchors",
    }
    path = output_dir / f"{head['sequence']:06d}_{head['record_hash']}.json"
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise Stage3ConflictError(f"tracked ledger anchor bytes conflict: {path}")
    else:
        _exclusive_write(path, raw)
    return payload, path


def _print_json(payload: Any) -> None:
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="validate protocol and frozen parent bytes")
    subparsers.add_parser("anchor", help="create the tracked protocol/source anchor")
    subparsers.add_parser("init", help="initialize the content-bound append-only ledger")
    fetch = subparsers.add_parser("fetch-reference-data", help="freeze official calendar and decision-date inputs")
    fetch.add_argument(
        "--decision-date",
        action="append",
        dest="decision_dates",
        help="decision date to fetch; repeat for bridge weeks (defaults to the three rehearsal dates)",
    )
    fetch.add_argument(
        "--state-manifest",
        type=Path,
        help="explicit content-addressed state cache manifest (required if more than one cache exists)",
    )
    rehearse = subparsers.add_parser("rehearse", help="run the three non-counting retrospective fixtures")
    rehearse.add_argument("--reference-manifest", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-decision", help="append one causally timed prospective decision")
    freeze.add_argument("decision_date")
    freeze.add_argument("--reference-manifest", type=Path, required=True)
    label = subparsers.add_parser("complete-label", help="append the frozen decision's five-session label")
    label.add_argument("decision_date")
    label.add_argument(
        "--state-manifest",
        type=Path,
        help="explicit state cache manifest whose raw closure covers the exit",
    )
    subparsers.add_parser(
        "export-head-anchor",
        help="export a unique tracked commitment for commit/push after each append",
    )
    subparsers.add_parser("verify", help="recompute and validate the complete ledger chain")
    subparsers.add_parser("status", help="show blinded counts only")
    subparsers.add_parser("evaluate", help="refuse before 52 weeks, then evaluate exactly once")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 3 governance commands."""

    args = _build_parser().parse_args(argv)
    spec = load_and_validate_spec()
    if args.command == "check":
        _print_json(
            {
                "status": "VALID_FORWARD_EXPLORATION_PROTOCOL",
                "study_id": spec["study_id"],
                "study_identity": study_identity(spec),
                "first_prospective_decision_date": spec["historical_exclusion"]["first_prospective_decision_date"],
                "prospective_weeks_recorded": 0,
                "confirmation_chain": "NOT_STARTED",
            }
        )
        return 0
    if args.command == "anchor":
        _print_json(build_anchor(spec))
        return 0

    root = resolve_ledger_root(spec)
    if args.command == "fetch-reference-data":
        decision_dates = (
            args.decision_dates
            if args.decision_dates
            else spec["historical_exclusion"]["pre_genesis_rehearsal_decision_dates"]
        )
        manifest, manifest_path = fetch_reference_data(
            spec,
            decision_dates,
            root=root,
            state_manifest_path=None if args.state_manifest is None else args.state_manifest.resolve(),
        )
        _print_json(
            {
                "status": "REFERENCE_DATA_FROZEN",
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_path": str(manifest_path),
                "decision_dates": sorted(manifest["daily_basic"]),
            }
        )
        return 0
    if args.command == "rehearse":
        _print_json(rehearse_pipeline(spec, root, args.reference_manifest.resolve()))
        return 0
    if args.command == "init":
        record = initialize_ledger(spec, root=root)
        _print_json({"ledger_root": str(root), "genesis_record_hash": record.data["record_hash"]})
        return 0
    if args.command == "freeze-decision":
        record = append_decision_freeze(
            spec,
            root,
            args.reference_manifest.resolve(),
            args.decision_date,
        )
        _print_json({"status": "DECISION_FROZEN", "record_hash": record.data["record_hash"]})
        return 0
    if args.command == "complete-label":
        record = append_label_completion(
            spec,
            root,
            args.decision_date,
            state_manifest_path=None if args.state_manifest is None else args.state_manifest.resolve(),
        )
        _print_json({"status": "LABEL_COMPLETED", "record_hash": record.data["record_hash"]})
        return 0
    records = scan_records(root)
    if not records:
        raise Stage3ValidationError("ledger is not initialized; run init first")
    if args.command == "verify":
        validate_record_semantics(spec, records, root=root)
        _print_json(
            {
                "status": "VALID_CHAIN",
                "ledger_root": str(root),
                "records": len(records),
                "head": records[-1].data["record_hash"],
            }
        )
        return 0
    if args.command == "status":
        _print_json(status_report(spec, records, root=root))
        return 0
    if args.command == "export-head-anchor":
        anchor, anchor_path = export_ledger_head_anchor(spec, root)
        _print_json({"status": "LEDGER_HEAD_ANCHOR_EXPORTED", "path": str(anchor_path), **anchor})
        return 0
    if args.command == "evaluate":
        record = finalize_evaluation(spec, root)
        _print_json(record.data["payload"])
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
