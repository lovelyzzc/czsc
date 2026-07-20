"""Fail-closed, externally anchored forward ledger for the XS-Chan V2.1 trial.

This module is intentionally independent from the frozen V2 ledger.  It is an
append-only evidence container, not a research or replay engine.  Genesis
claims the one primary chain for a protocol in an external trial registry.
Official exchange calendars are then committed in contiguous, signed segments
before those segments become effective.  A confirmation week is represented by
the following causal event sequence::

    decision -> session_open_execution -> session_eod_valuation -> ...
             -> cycle_close

Every covered session has one open heartbeat, even when there are no orders.
Blocked sells are retried only inside the next session's open record; there is
no intraday or closing-auction retry record.

The local verifier proves schemas, event timing and hash-chain structure only.
It deliberately never promotes a chain to ``confirmatory_oos``; semantic replay
is a separate mandatory gate.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import fcntl
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 21
PROTOCOL_ID = "xs_chan_pilot_v2_1_preregistered_20260720"
CANONICAL_PROTOCOL_SHA256 = "8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203"
ZERO_HASH = "0" * 64
MIN_COMPLETE_WEEKS = 52
EXCHANGE_TIMEZONE = "Asia/Shanghai"
DECISION_CLOSE_LOCAL = "15:00:00"
EXECUTION_OPEN_LOCAL = "09:30:00"
OPEN_EXECUTION_DEADLINE_LOCAL = "10:00:00"
SESSION_CLOSE_LOCAL = "15:00:00"

GENESIS_CONFIG_SCHEMA = "xs_chan_v2_1_genesis_config_v1"
GENESIS_PAYLOAD_SCHEMA = "xs_chan_v2_1_genesis_v1"
PORTFOLIO_STATE_SCHEMA = "xs_chan_portfolio_state_v2_1"
CALENDAR_SEGMENT_SCHEMA = "xs_chan_v2_1_calendar_segment_v1"
ANCHOR_RECEIPT_SCHEMA = "xs_chan_external_anchor_receipt_v1"
ANCHOR_ALGORITHM = "rsa-pkcs1v15-sha256"
REGISTRY_CLAIM_SCHEMA = "xs_chan_v2_1_primary_chain_claim_v1"
DECISION_SCHEMA = "xs_chan_v2_1_decision_v1"
OPEN_EXECUTION_SCHEMA = "xs_chan_v2_1_session_open_execution_v1"
EOD_VALUATION_SCHEMA = "xs_chan_v2_1_session_eod_valuation_v1"
WEEKLY_CLOSE_SCHEMA = "xs_chan_v2_1_cycle_close_v1"
TERMINAL_SCHEMA = "xs_chan_v2_1_terminal_v1"
CONFIRMATION_IDENTITY_SCHEMA = "xs_chan_v2_1_confirmation_window_identity_v1"

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
READINESS_GATE_KEYS = frozenset(
    {
        "state_recompute_engineering",
        "feature_prefix_engineering",
        "source_archive_reconciliation_engineering",
        "suspension_reconciliation_engineering",
        "open_auction_reconciliation_engineering",
        "calendar_append_verifier_engineering",
        "execution_replay_engineering",
        "portfolio_accounting_replay_engineering",
        "corporate_action_replay_engineering",
        "statistics_replay_engineering",
        "artifact_semantic_verifier_engineering",
        "primary_chain_registry_engineering",
        "external_timestamp_anchor_engineering",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
_NONCE = re.compile(r"^[A-Za-z0-9._:-]{16,128}$")
_WEEK_ID = re.compile(r"^\d{4}-W\d{2}$")
_RECORD_NAME = re.compile(r"^(?P<sequence>\d{6})_(?P<digest>[0-9a-f]{64})\.json$")
_RECORD_KEYS = {
    "schema_version",
    "sequence",
    "record_type",
    "recorded_at_utc",
    "previous_record_hash",
    "payload",
    "record_hash",
}
_ANCHORED_PAYLOAD_KEYS = {"event", "record_commitment_sha256", "external_anchor"}
_GENESIS_CONFIG_KEYS = {
    "schema",
    "protocol_id",
    "protocol_sha256",
    "data_bundle_sha256",
    "dependency_sha256",
    "readiness_gates",
    "initial_portfolio_state",
    "initial_portfolio_state_sha256",
    "trial_id",
    "trial_registry_namespace",
    "trial_registry_authority_sha256",
    "chain_role",
    "calendar_source_id",
    "oos_start_date",
    "min_complete_weeks",
    "exchange_timezone",
    "decision_close_local",
    "execution_open_local",
    "open_execution_deadline_local",
    "session_close_local",
}
_INITIAL_PORTFOLIO_KEYS = {
    "schema",
    "cash_cny",
    "positions",
    "pending_sells",
    "last_close",
    "cash_receivables",
    "dividend_tax_liabilities",
    "other_assets",
    "other_liabilities",
    "borrowed_cash_cny",
}
_READINESS_EVIDENCE_KEYS = {"status", "evidence_sha256"}
_ANCHOR_IDENTITY_KEYS = {"algorithm", "provider", "key_id", "public_key_sha256"}
_ANCHOR_RECEIPT_KEYS = {
    "schema",
    "algorithm",
    "provider",
    "key_id",
    "subject_sha256",
    "issued_at_utc",
    "nonce",
    "signature_base64",
}
_CALENDAR_SEGMENT_KEYS = {
    "schema",
    "segment_index",
    "coverage_start",
    "coverage_end",
    "sessions",
    "source_id",
    "source_uri",
    "source_sha256",
    "source_retrieved_at_utc",
}
_DECISION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "decision_artifact_sha256",
    "portfolio_before_sha256",
    "pending_sells_before_sha256",
    "calendar_state_sha256",
}
_OPEN_EXECUTION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "decision_record_sha256",
    "previous_eod_record_sha256",
    "portfolio_before_sha256",
    "pending_sells_before_sha256",
    "pending_sell_count_before",
    "open_prices_sha256",
    "opening_auction_turnover_sha256",
    "limit_state_sha256",
    "corporate_actions_sha256",
    "requested_orders_sha256",
    "fills_sha256",
    "fees_sha256",
    "execution_result_sha256",
    "portfolio_after_sha256",
    "pending_sells_after_sha256",
    "pending_sell_count_after",
}
_EOD_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "session_open_execution_sha256",
    "raw_close_snapshot_sha256",
    "corporate_actions_sha256",
    "portfolio_before_eod_sha256",
    "portfolio_state_sha256",
    "pending_sells_sha256",
    "pending_sell_count",
    "nav_artifact_sha256",
}
_WEEKLY_CLOSE_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "execution_engine_sha256",
    "statistics_engine_sha256",
    "eod_record_sha256",
    "portfolio_state_sha256",
    "pending_sells_sha256",
    "pending_sell_count",
    "weekly_returns_sha256",
    "statistics_input_sha256",
}
_TERMINAL_KEYS = {"schema", "outcome", "reason_code", "evidence_sha256", "details_sha256"}
_SHA_FIELDS_BY_SCHEMA = {
    DECISION_SCHEMA: _DECISION_KEYS - {"schema", "snapshot_cutoff_utc"},
    OPEN_EXECUTION_SCHEMA: _OPEN_EXECUTION_KEYS
    - {
        "schema",
        "snapshot_cutoff_utc",
        "pending_sell_count_before",
        "pending_sell_count_after",
    },
    EOD_VALUATION_SCHEMA: _EOD_KEYS - {"schema", "snapshot_cutoff_utc", "pending_sell_count"},
    WEEKLY_CLOSE_SCHEMA: _WEEKLY_CLOSE_KEYS - {"schema", "snapshot_cutoff_utc", "pending_sell_count"},
}
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


class LedgerError(RuntimeError):
    """Base class for V2.1 ledger errors."""


class LedgerConflictError(LedgerError):
    """An immutable logical identity or primary-chain claim already exists."""


class LedgerValidationError(LedgerError):
    """The chain, schema, receipt or proposed transition is invalid."""


class LedgerTimingError(LedgerValidationError):
    """A causal event was attempted outside its registered time window."""


class ReceiptVerificationError(LedgerValidationError):
    """An external timestamp receipt cannot be cryptographically verified."""


@dataclass(frozen=True)
class VerifiedReceipt:
    """Result returned only after a real signature check succeeds."""

    subject_sha256: str
    issued_at_utc: datetime
    provider: str
    key_id: str
    receipt_sha256: str


class RsaPkcs1v15Sha256ReceiptVerifier:
    """Verify externally issued RSA PKCS#1 v1.5 SHA-256 receipts.

    The verifier contains public material only.  It never accepts a caller
    supplied boolean assertion of validity.  A missing verifier leaves public
    verification closed; a malformed or bad signature raises.
    """

    def __init__(
        self,
        *,
        provider: str,
        key_id: str,
        modulus_hex: str,
        public_exponent: int = 65537,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ReceiptVerificationError("anchor provider must be a non-empty string")
        if not isinstance(key_id, str) or not key_id.strip():
            raise ReceiptVerificationError("anchor key_id must be a non-empty string")
        if not isinstance(modulus_hex, str) or not re.fullmatch(r"[1-9a-f][0-9a-f]+", modulus_hex):
            raise ReceiptVerificationError("RSA modulus_hex must be canonical lowercase hexadecimal")
        modulus = int(modulus_hex, 16)
        if modulus.bit_length() < 1024:
            raise ReceiptVerificationError("RSA anchor key must be at least 1024 bits")
        if type(public_exponent) is not int or public_exponent < 3 or public_exponent % 2 == 0:
            raise ReceiptVerificationError("RSA public_exponent must be an odd integer >= 3")
        if public_exponent >= modulus:
            raise ReceiptVerificationError("RSA public_exponent must be smaller than the modulus")
        self._provider = provider
        self._key_id = key_id
        self._modulus_hex = modulus_hex
        self._modulus = modulus
        self._public_exponent = public_exponent

    @property
    def identity(self) -> dict[str, Any]:
        """Return the exact public-key identity frozen by genesis."""
        key = {"modulus_hex": self._modulus_hex, "public_exponent": self._public_exponent}
        return {
            "algorithm": ANCHOR_ALGORITHM,
            "provider": self._provider,
            "key_id": self._key_id,
            "public_key_sha256": _sha256_bytes(_canonical_bytes(key)),
        }

    def verify(self, receipt: Mapping[str, Any], *, expected_subject_sha256: str) -> VerifiedReceipt:
        """Validate receipt schema, identity, subject and RSA signature."""
        value = _exact_mapping(receipt, _ANCHOR_RECEIPT_KEYS, "external anchor receipt")
        if value["schema"] != ANCHOR_RECEIPT_SCHEMA:
            raise ReceiptVerificationError(f"receipt.schema must be {ANCHOR_RECEIPT_SCHEMA!r}")
        if value["algorithm"] != ANCHOR_ALGORITHM:
            raise ReceiptVerificationError(f"receipt.algorithm must be {ANCHOR_ALGORITHM!r}")
        if value["provider"] != self._provider or value["key_id"] != self._key_id:
            raise ReceiptVerificationError("receipt signer differs from the frozen anchor identity")
        expected = _require_sha256(expected_subject_sha256, "expected receipt subject")
        subject = _require_sha256(value["subject_sha256"], "receipt.subject_sha256")
        if subject != expected:
            raise ReceiptVerificationError("receipt subject does not match the committed artifact")
        issued = _canonical_utc(value["issued_at_utc"], "receipt.issued_at_utc")
        if not isinstance(value["nonce"], str) or _NONCE.fullmatch(value["nonce"]) is None:
            raise ReceiptVerificationError("receipt.nonce must be a 16..128 character external nonce")
        signature_text = value["signature_base64"]
        if not isinstance(signature_text, str):
            raise ReceiptVerificationError("receipt.signature_base64 must be a string")
        try:
            signature = base64.b64decode(signature_text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ReceiptVerificationError("receipt.signature_base64 is invalid") from exc
        if base64.b64encode(signature).decode("ascii") != signature_text:
            raise ReceiptVerificationError("receipt.signature_base64 must use canonical padded base64")

        signed_fields = {key: value[key] for key in sorted(_ANCHOR_RECEIPT_KEYS - {"signature_base64"})}
        message = _canonical_bytes(signed_fields)
        modulus_bytes = (self._modulus.bit_length() + 7) // 8
        if len(signature) != modulus_bytes:
            raise ReceiptVerificationError("receipt signature length does not match the frozen RSA key")
        signature_integer = int.from_bytes(signature, "big")
        if signature_integer >= self._modulus:
            raise ReceiptVerificationError("receipt signature integer is outside the RSA modulus")
        encoded = pow(signature_integer, self._public_exponent, self._modulus).to_bytes(modulus_bytes, "big")
        digest_info = _SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(message).digest()
        padding_length = modulus_bytes - len(digest_info) - 3
        if padding_length < 8:
            raise ReceiptVerificationError("RSA key is too short for PKCS#1 v1.5 SHA-256")
        expected_encoding = b"\x00\x01" + b"\xff" * padding_length + b"\x00" + digest_info
        if not hmac.compare_digest(encoded, expected_encoding):
            raise ReceiptVerificationError("external anchor RSA signature is invalid")
        return VerifiedReceipt(
            subject_sha256=subject,
            issued_at_utc=issued,
            provider=self._provider,
            key_id=self._key_id,
            receipt_sha256=_sha256_bytes(_canonical_bytes(value)),
        )


@dataclass(frozen=True)
class LedgerRecord:
    """One canonical, hash-verified ledger record."""

    path: Path
    data: dict[str, Any]
    raw: bytes

    @property
    def sequence(self) -> int:
        return int(self.data["sequence"])

    @property
    def record_type(self) -> str:
        return str(self.data["record_type"])

    @property
    def record_hash(self) -> str:
        return str(self.data["record_hash"])

    @property
    def recorded_at_utc(self) -> datetime:
        return _canonical_utc(self.data["recorded_at_utc"], "recorded_at_utc")


@dataclass(frozen=True)
class WeekSpec:
    """One fully covered, actual exchange-calendar confirmation week."""

    week_id: str
    week_start: date
    decision_dt: date
    execution_dt: date
    close_dt: date
    sessions: tuple[date, ...]
    calendar_state_sha256: str


@dataclass
class _ReplayState:
    records: list[LedgerRecord]
    genesis: LedgerRecord
    config: dict[str, Any]
    anchor_identity: dict[str, Any]
    receipts_verified: bool
    segments: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[date] = field(default_factory=list)
    completed_weeks: list[dict[str, Any]] = field(default_factory=list)
    active_week: WeekSpec | None = None
    phase: str = "IDLE"
    decision_record: LedgerRecord | None = None
    last_open_record: LedgerRecord | None = None
    last_eod_record: LedgerRecord | None = None
    next_session_index: int = 0
    portfolio_sha256: str = ""
    pending_sells_sha256: str = ""
    pending_sell_count: int = 0
    terminal: LedgerRecord | None = None


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"value is not canonical-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_sha256(value: Any) -> str:
    """Match ``xs_chan_execution_v2_1.object_sha256`` (no record newline)."""
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an evidence artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_initial_portfolio() -> dict[str, Any]:
    """Return ``empty_portfolio(10_000_000.0).as_dict()`` exactly."""
    return {
        "schema": PORTFOLIO_STATE_SCHEMA,
        "cash_cny": 10_000_000.0,
        "positions": {},
        "pending_sells": {},
        "last_close": {},
        "cash_receivables": {},
        "dividend_tax_liabilities": {},
        "other_assets": [],
        "other_liabilities": [],
        "borrowed_cash_cny": 0.0,
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LedgerValidationError(f"{label} must be a lowercase SHA256 digest")
    return value


def _exact_mapping(value: Any, keys: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerValidationError(f"{label} must be an object")
    result = dict(value)
    if set(result) != set(keys):
        missing = sorted(set(keys) - set(result))
        extra = sorted(set(result) - set(keys))
        raise LedgerValidationError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return result


def _normalise_date(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise LedgerValidationError(f"{label} must be a date, not a datetime")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise LedgerValidationError(f"{label} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise LedgerValidationError(f"{label} must be YYYY-MM-DD") from exc
    if value != parsed.isoformat():
        raise LedgerValidationError(f"{label} must use canonical YYYY-MM-DD format")
    return parsed


def _normalise_utc(value: datetime | str, label: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LedgerValidationError(f"{label} must be a timezone-aware timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise LedgerValidationError(f"{label} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerValidationError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise LedgerValidationError(f"{label} must be a canonical UTC string")
    parsed = _normalise_utc(value, label)
    if value != _format_utc(parsed):
        raise LedgerValidationError(f"{label} must use canonical UTC Z format with microseconds")
    return parsed


def _utc_now() -> datetime:
    """Clock seam for tests; public functions and CLI never accept historical time."""
    return datetime.now(UTC)


def _strict_time(value: Any, expected: str, label: str) -> str:
    if value != expected:
        raise LedgerValidationError(f"{label} must be frozen at {expected}")
    try:
        parsed = time.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"{label} must be a local time") from exc
    if parsed.tzinfo is not None:
        raise LedgerValidationError(f"{label} must not contain a timezone offset")
    return value


def _local_boundary(day: date, local_time: str, timezone_name: str) -> datetime:
    return datetime.combine(day, time.fromisoformat(local_time), tzinfo=ZoneInfo(timezone_name)).astimezone(UTC)


def _week_id(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _validate_initial_portfolio(value: Any) -> dict[str, Any]:
    portfolio = _exact_mapping(value, _INITIAL_PORTFOLIO_KEYS, "initial_portfolio_state")
    if portfolio != empty_initial_portfolio():
        raise LedgerValidationError(
            "initial_portfolio_state must equal xs_chan_execution_v2_1.empty_portfolio(10_000_000.0).as_dict()"
        )
    return portfolio


def _validate_readiness_gates(value: Any) -> dict[str, Any]:
    gates = _exact_mapping(value, READINESS_GATE_KEYS, "readiness_gates")
    result: dict[str, Any] = {}
    for gate in sorted(READINESS_GATE_KEYS):
        evidence = _exact_mapping(gates[gate], _READINESS_EVIDENCE_KEYS, f"readiness_gates.{gate}")
        if evidence["status"] != "PASSED":
            raise LedgerValidationError(f"readiness_gates.{gate}.status must be 'PASSED'")
        evidence["evidence_sha256"] = _require_sha256(
            evidence["evidence_sha256"], f"readiness_gates.{gate}.evidence_sha256"
        )
        result[gate] = evidence
    return result


def _validate_genesis_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = _exact_mapping(value, _GENESIS_CONFIG_KEYS, "genesis_config")
    if config["schema"] != GENESIS_CONFIG_SCHEMA:
        raise LedgerValidationError(f"genesis_config.schema must be {GENESIS_CONFIG_SCHEMA!r}")
    if config["protocol_id"] != PROTOCOL_ID:
        raise LedgerValidationError(f"protocol_id must be frozen at {PROTOCOL_ID!r}")
    config["protocol_sha256"] = _require_sha256(config["protocol_sha256"], "protocol_sha256")
    if config["protocol_sha256"] != CANONICAL_PROTOCOL_SHA256:
        raise LedgerValidationError(
            f"protocol_sha256 must equal the frozen canonical protocol digest {CANONICAL_PROTOCOL_SHA256}"
        )
    config["data_bundle_sha256"] = _require_sha256(config["data_bundle_sha256"], "data_bundle_sha256")
    dependencies = _exact_mapping(config["dependency_sha256"], EXPECTED_DEPENDENCY_KEYS, "dependency_sha256")
    config["dependency_sha256"] = {
        name: _require_sha256(dependencies[name], f"dependency_sha256.{name}")
        for name in sorted(EXPECTED_DEPENDENCY_KEYS)
    }
    config["readiness_gates"] = _validate_readiness_gates(config["readiness_gates"])
    config["initial_portfolio_state"] = _validate_initial_portfolio(config["initial_portfolio_state"])
    config["initial_portfolio_state_sha256"] = _require_sha256(
        config["initial_portfolio_state_sha256"], "initial_portfolio_state_sha256"
    )
    expected_initial_hash = _object_sha256(config["initial_portfolio_state"])
    if config["initial_portfolio_state_sha256"] != expected_initial_hash:
        raise LedgerValidationError("initial_portfolio_state_sha256 does not match the canonical execution state")
    if config["trial_id"] != config["protocol_sha256"]:
        raise LedgerValidationError("trial_id must exactly equal canonical protocol_sha256")
    if (
        not isinstance(config["trial_registry_namespace"], str)
        or _NAMESPACE.fullmatch(config["trial_registry_namespace"]) is None
    ):
        raise LedgerValidationError("trial_registry_namespace is invalid")
    config["trial_registry_authority_sha256"] = _require_sha256(
        config["trial_registry_authority_sha256"], "trial_registry_authority_sha256"
    )
    if config["chain_role"] != "PRIMARY":
        raise LedgerValidationError("chain_role must be 'PRIMARY'")
    if not isinstance(config["calendar_source_id"], str) or not config["calendar_source_id"].strip():
        raise LedgerValidationError("calendar_source_id must be a non-empty registered source")
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    if start.weekday() != 0:
        raise LedgerValidationError("oos_start_date must be a Monday confirmation-week boundary")
    config["oos_start_date"] = start.isoformat()
    if config["min_complete_weeks"] != MIN_COMPLETE_WEEKS:
        raise LedgerValidationError(f"min_complete_weeks must be frozen at {MIN_COMPLETE_WEEKS}")
    if config["exchange_timezone"] != EXCHANGE_TIMEZONE:
        raise LedgerValidationError(f"exchange_timezone must be frozen at {EXCHANGE_TIMEZONE}")
    try:
        ZoneInfo(config["exchange_timezone"])
    except ZoneInfoNotFoundError as exc:
        raise LedgerValidationError("exchange_timezone is unavailable") from exc
    for field_name, expected in (
        ("decision_close_local", DECISION_CLOSE_LOCAL),
        ("execution_open_local", EXECUTION_OPEN_LOCAL),
        ("open_execution_deadline_local", OPEN_EXECUTION_DEADLINE_LOCAL),
        ("session_close_local", SESSION_CLOSE_LOCAL),
    ):
        config[field_name] = _strict_time(config[field_name], expected, field_name)
    return config


def _primary_chain_id(config: Mapping[str, Any]) -> str:
    identity = {
        "schema": "xs_chan_v2_1_primary_chain_identity_v1",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "data_bundle_sha256": config["data_bundle_sha256"],
        "trial_id": config["trial_id"],
        "trial_registry_namespace": config["trial_registry_namespace"],
        "trial_registry_authority_sha256": config["trial_registry_authority_sha256"],
        "chain_role": "PRIMARY",
    }
    return _sha256_bytes(_canonical_bytes(identity))


def genesis_commitment_sha256(
    genesis_config: Mapping[str, Any],
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> str:
    """Return the artifact digest an external signer must timestamp for genesis."""
    if not isinstance(receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("a concrete RSA receipt verifier is required")
    config = _validate_genesis_config(genesis_config)
    core = {
        "schema": "xs_chan_v2_1_genesis_commitment_v1",
        "config": config,
        "primary_chain_id": _primary_chain_id(config),
        "anchor_verifier_identity": receipt_verifier.identity,
    }
    return _sha256_bytes(_canonical_bytes(core))


def _validate_segment(value: Mapping[str, Any]) -> dict[str, Any]:
    segment = _exact_mapping(value, _CALENDAR_SEGMENT_KEYS, "calendar_segment")
    if segment["schema"] != CALENDAR_SEGMENT_SCHEMA:
        raise LedgerValidationError(f"calendar_segment.schema must be {CALENDAR_SEGMENT_SCHEMA!r}")
    if type(segment["segment_index"]) is not int or segment["segment_index"] < 0:
        raise LedgerValidationError("calendar_segment.segment_index must be a non-negative integer")
    start = _normalise_date(segment["coverage_start"], "calendar_segment.coverage_start")
    end = _normalise_date(segment["coverage_end"], "calendar_segment.coverage_end")
    if end < start:
        raise LedgerValidationError("calendar segment coverage_end precedes coverage_start")
    if type(segment["sessions"]) is not list:
        raise LedgerValidationError("calendar_segment.sessions must be a list")
    sessions = [_normalise_date(item, "calendar session") for item in segment["sessions"]]
    if sessions != sorted(set(sessions)):
        raise LedgerValidationError("calendar segment sessions must be strictly increasing and unique")
    if any(item < start or item > end for item in sessions):
        raise LedgerValidationError("calendar session lies outside its coverage interval")
    if not isinstance(segment["source_id"], str) or not segment["source_id"].strip():
        raise LedgerValidationError("calendar_segment.source_id must be non-empty")
    uri = segment["source_uri"]
    if not isinstance(uri, str):
        raise LedgerValidationError("calendar_segment.source_uri must be an HTTPS URL")
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "https" or not parsed_uri.netloc:
        raise LedgerValidationError("calendar_segment.source_uri must be an HTTPS URL")
    segment["source_sha256"] = _require_sha256(segment["source_sha256"], "calendar_segment.source_sha256")
    retrieved = _canonical_utc(segment["source_retrieved_at_utc"], "calendar_segment.source_retrieved_at_utc")
    segment["coverage_start"] = start.isoformat()
    segment["coverage_end"] = end.isoformat()
    segment["sessions"] = [item.isoformat() for item in sessions]
    segment["source_retrieved_at_utc"] = _format_utc(retrieved)
    return segment


def calendar_segment_commitment_sha256(primary_chain_id: str, segment: Mapping[str, Any]) -> str:
    """Return the signed commitment for one official-calendar segment."""
    chain_id = _require_sha256(primary_chain_id, "primary_chain_id")
    core = {
        "schema": "xs_chan_v2_1_calendar_segment_commitment_v1",
        "primary_chain_id": chain_id,
        "segment": _validate_segment(segment),
    }
    return _sha256_bytes(_canonical_bytes(core))


def _validate_receipt_shape(receipt: Any) -> dict[str, Any]:
    """Validate immutable receipt fields without claiming signature validity."""
    value = _exact_mapping(receipt, _ANCHOR_RECEIPT_KEYS, "external anchor receipt")
    if value["schema"] != ANCHOR_RECEIPT_SCHEMA:
        raise ReceiptVerificationError(f"receipt.schema must be {ANCHOR_RECEIPT_SCHEMA!r}")
    if value["algorithm"] != ANCHOR_ALGORITHM:
        raise ReceiptVerificationError(f"receipt.algorithm must be {ANCHOR_ALGORITHM!r}")
    if not isinstance(value["provider"], str) or not value["provider"].strip():
        raise ReceiptVerificationError("receipt.provider must be non-empty")
    if not isinstance(value["key_id"], str) or not value["key_id"].strip():
        raise ReceiptVerificationError("receipt.key_id must be non-empty")
    _require_sha256(value["subject_sha256"], "receipt.subject_sha256")
    _canonical_utc(value["issued_at_utc"], "receipt.issued_at_utc")
    if not isinstance(value["nonce"], str) or _NONCE.fullmatch(value["nonce"]) is None:
        raise ReceiptVerificationError("receipt.nonce must be a 16..128 character external nonce")
    signature = value["signature_base64"]
    if not isinstance(signature, str):
        raise ReceiptVerificationError("receipt.signature_base64 must be a string")
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReceiptVerificationError("receipt.signature_base64 is invalid") from exc
    if base64.b64encode(decoded).decode("ascii") != signature:
        raise ReceiptVerificationError("receipt.signature_base64 must use canonical padded base64")
    return value


def _verify_receipt_timing(
    verified: VerifiedReceipt,
    *,
    recorded_at_utc: datetime,
    earliest_utc: datetime | None = None,
    latest_utc: datetime | None = None,
) -> None:
    if verified.issued_at_utc > recorded_at_utc:
        raise ReceiptVerificationError("external receipt was issued after the ledger record")
    if earliest_utc is not None and verified.issued_at_utc < earliest_utc:
        raise ReceiptVerificationError("external receipt predates the artifact source retrieval")
    if latest_utc is not None and verified.issued_at_utc >= latest_utc:
        raise ReceiptVerificationError("external receipt was not anchored before the effective boundary")


def _build_record(
    sequence: int,
    record_type: str,
    recorded_at_utc: datetime,
    previous_record_hash: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "record_type": record_type,
        "recorded_at_utc": _format_utc(recorded_at_utc),
        "previous_record_hash": _require_sha256(previous_record_hash, "previous_record_hash"),
        "payload": dict(payload),
    }
    record_hash = _sha256_bytes(_canonical_bytes(body))
    record = {**body, "record_hash": record_hash}
    return record, _canonical_bytes(record)


def _record_commitment_sha256(
    *,
    primary_chain_id: str,
    previous_record_hash: str,
    record_type: str,
    event_payload: Mapping[str, Any],
) -> str:
    core = {
        "schema": "xs_chan_v2_1_anchored_record_commitment_v1",
        "primary_chain_id": _require_sha256(primary_chain_id, "primary_chain_id"),
        "previous_record_hash": _require_sha256(previous_record_hash, "previous_record_hash"),
        "record_type": record_type,
        "event": dict(event_payload),
    }
    return _sha256_bytes(_canonical_bytes(core))


def _record_path(root: Path, sequence: int, record_hash: str) -> Path:
    return root / f"{sequence:06d}_{record_hash}.json"


@contextmanager
def _exclusive_lock(directory: Path, filename: str):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open("a+b") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _exclusive_write(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise LedgerConflictError(f"immutable file already exists: {path}") from exc
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
    grouped: dict[int, list[tuple[Path, str]]] = {}
    for path in sorted(item for item in root.iterdir() if item.suffix == ".json"):
        match = _RECORD_NAME.fullmatch(path.name)
        if match is None:
            raise LedgerValidationError(f"invalid ledger record filename: {path.name}")
        grouped.setdefault(int(match.group("sequence")), []).append((path, match.group("digest")))
    if not grouped:
        raise LedgerValidationError("ledger has no genesis record")
    for sequence, candidates in sorted(grouped.items()):
        if len(candidates) != 1:
            raise LedgerValidationError(
                f"fork detected at sequence {sequence:06d}: {[item[0].name for item in candidates]}"
            )
    sequences = sorted(grouped)
    expected = list(range(sequences[-1] + 1))
    if sequences != expected:
        raise LedgerValidationError(f"missing ledger sequence numbers: {sorted(set(expected) - set(sequences))}")

    records: list[LedgerRecord] = []
    previous_hash = ZERO_HASH
    previous_time: datetime | None = None
    for sequence in sequences:
        path, filename_hash = grouped[sequence][0]
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerValidationError(f"invalid JSON in {path.name}") from exc
        record = _exact_mapping(data, _RECORD_KEYS, f"record {path.name}")
        if raw != _canonical_bytes(record):
            raise LedgerValidationError(f"record bytes are not canonical: {path.name}")
        if record["schema_version"] != SCHEMA_VERSION:
            raise LedgerValidationError(f"unsupported schema_version in {path.name}")
        if record["sequence"] != sequence:
            raise LedgerValidationError(f"record sequence mismatch in {path.name}")
        if not isinstance(record["record_type"], str):
            raise LedgerValidationError(f"record_type must be a string in {path.name}")
        record_hash = _require_sha256(record["record_hash"], f"record_hash in {path.name}")
        if record_hash != filename_hash:
            raise LedgerValidationError(f"filename hash mismatch in {path.name}")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        if _sha256_bytes(_canonical_bytes(body)) != record_hash:
            raise LedgerValidationError(f"record hash mismatch in {path.name}")
        if record["previous_record_hash"] != previous_hash:
            raise LedgerValidationError(f"previous hash mismatch in {path.name}")
        recorded_at = _canonical_utc(record["recorded_at_utc"], f"recorded_at_utc in {path.name}")
        if previous_time is not None and recorded_at <= previous_time:
            raise LedgerValidationError(f"recorded_at_utc is not strictly increasing in {path.name}")
        records.append(LedgerRecord(path=path, data=record, raw=raw))
        previous_hash = record_hash
        previous_time = recorded_at
    return tuple(records)


def _calendar_state_sha256(segments: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "segment_index": segment["segment_index"],
            "commitment_sha256": segment["commitment_sha256"],
            "coverage_start": segment["segment"]["coverage_start"],
            "coverage_end": segment["segment"]["coverage_end"],
        }
        for segment in segments
    ]
    return _sha256_bytes(_canonical_bytes(identity))


def _week_specs(state: _ReplayState) -> list[WeekSpec]:
    if not state.segments:
        return []
    coverage_start = _normalise_date(state.segments[0]["segment"]["coverage_start"], "coverage_start")
    coverage_end = _normalise_date(state.segments[-1]["segment"]["coverage_end"], "coverage_end")
    oos_start = _normalise_date(state.config["oos_start_date"], "oos_start_date")
    sessions = sorted(state.sessions)
    specs: list[WeekSpec] = []
    monday = oos_start
    while monday + timedelta(days=6) <= coverage_end:
        sunday = monday + timedelta(days=6)
        if monday >= coverage_start:
            week_sessions = tuple(item for item in sessions if monday <= item <= sunday)
            if week_sessions:
                prior = [item for item in sessions if item < week_sessions[0]]
                if prior:
                    specs.append(
                        WeekSpec(
                            week_id=_week_id(monday),
                            week_start=monday,
                            decision_dt=prior[-1],
                            execution_dt=week_sessions[0],
                            close_dt=week_sessions[-1],
                            sessions=week_sessions,
                            calendar_state_sha256=_calendar_state_sha256(state.segments),
                        )
                    )
        monday += timedelta(days=7)
    return specs


def _expected_week(state: _ReplayState) -> WeekSpec | None:
    specs = _week_specs(state)
    index = len(state.completed_weeks)
    return specs[index] if index < len(specs) else None


def _event_window(state: _ReplayState) -> tuple[str, datetime, datetime] | None:
    if state.terminal is not None or len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        return None
    timezone = state.config["exchange_timezone"]
    if state.phase == "IDLE":
        week = _expected_week(state)
        if week is None:
            return None
        return (
            "decision",
            _local_boundary(week.decision_dt, state.config["decision_close_local"], timezone),
            _local_boundary(week.execution_dt, state.config["execution_open_local"], timezone),
        )
    if state.phase in {"DECIDED", "AWAIT_SESSION_OPEN"}:
        assert state.active_week is not None
        session = state.active_week.sessions[state.next_session_index]
        return (
            "session_open_execution",
            _local_boundary(session, state.config["execution_open_local"], timezone),
            _local_boundary(
                session,
                state.config["open_execution_deadline_local"],
                timezone,
            ),
        )
    if state.phase == "OPENED":
        assert state.active_week is not None
        session = state.active_week.sessions[state.next_session_index]
        later_sessions = [item for item in state.sessions if item > session]
        if not later_sessions:
            raise LedgerValidationError("calendar must cover the next session before EOD valuation")
        return (
            "session_eod_valuation",
            _local_boundary(session, state.config["session_close_local"], timezone),
            _local_boundary(later_sessions[0], state.config["execution_open_local"], timezone),
        )
    if state.phase == "FINAL_EOD":
        assert state.active_week is not None
        later_sessions = [item for item in state.sessions if item > state.active_week.close_dt]
        if not later_sessions:
            raise LedgerValidationError("calendar must cover the next session before cycle_close")
        return (
            "cycle_close",
            state.last_eod_record.recorded_at_utc
            if state.last_eod_record is not None
            else datetime.min.replace(tzinfo=UTC),
            _local_boundary(later_sessions[0], state.config["execution_open_local"], timezone),
        )
    raise LedgerValidationError(f"unknown replay phase: {state.phase}")


def _deadline_failure(state: _ReplayState, at_utc: datetime) -> str | None:
    window = _event_window(state)
    if window is None:
        return None
    event, _, deadline = window
    if at_utc >= deadline:
        return f"MISSED_{event.upper()}_DEADLINE"
    return None


def _validate_window(state: _ReplayState, record_type: str, recorded_at: datetime) -> None:
    window = _event_window(state)
    if window is None:
        raise LedgerValidationError(f"{record_type} is not currently expected")
    expected, earliest, deadline = window
    if record_type != expected:
        raise LedgerValidationError(f"expected {expected}, not {record_type}")
    if recorded_at < earliest:
        raise LedgerTimingError(f"{record_type} cannot be recorded before its event boundary")
    if recorded_at >= deadline:
        raise LedgerTimingError(f"{record_type} must be recorded before its registered deadline")


def _validate_snapshot(value: Any, label: str, *, earliest: datetime, recorded_at: datetime) -> None:
    snapshot = _canonical_utc(value, label)
    if snapshot < earliest:
        raise LedgerTimingError(f"{label} precedes the causal market boundary")
    if snapshot > recorded_at:
        raise LedgerTimingError(f"{label} exceeds recorded_at_utc")


def _validate_count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise LedgerValidationError(f"{label} must be a non-negative integer")
    return value


def _validate_event_data(value: Any, schema: str, keys: set[str], label: str) -> dict[str, Any]:
    data = _exact_mapping(value, keys, label)
    if data["schema"] != schema:
        raise LedgerValidationError(f"{label}.schema must be {schema!r}")
    for key in _SHA_FIELDS_BY_SCHEMA[schema]:
        data[key] = _require_sha256(data[key], f"{label}.{key}")
    return data


def _apply_calendar_segment(
    state: _ReplayState,
    record: LedgerRecord,
    *,
    external_anchor: Mapping[str, Any],
    verified_receipt: VerifiedReceipt | None,
) -> None:
    payload = _exact_mapping(
        record.data["payload"],
        {"segment_index", "segment", "commitment_sha256"},
        "calendar_extension payload",
    )
    segment = _validate_segment(payload["segment"])
    if payload["segment_index"] != segment["segment_index"]:
        raise LedgerValidationError("calendar segment index fields disagree")
    if segment["segment_index"] != len(state.segments):
        raise LedgerValidationError("calendar segment indexes must be contiguous from zero")
    commitment = calendar_segment_commitment_sha256(_primary_chain_id(state.config), segment)
    if payload["commitment_sha256"] != commitment:
        raise LedgerValidationError("calendar segment commitment_sha256 mismatch")
    start = _normalise_date(segment["coverage_start"], "coverage_start")
    if state.segments:
        previous_end = _normalise_date(state.segments[-1]["segment"]["coverage_end"], "coverage_end")
        if start != previous_end + timedelta(days=1):
            raise LedgerValidationError("calendar coverage segments must be contiguous without overlap or gaps")
    else:
        oos_start = _normalise_date(state.config["oos_start_date"], "oos_start_date")
        if start >= oos_start:
            raise LedgerValidationError("first calendar segment must cover at least one pre-OOS decision date")
    sessions = [_normalise_date(item, "calendar session") for item in segment["sessions"]]
    if not sessions:
        raise LedgerValidationError("calendar extension must contain at least one official session")
    if set(sessions).intersection(state.sessions):
        raise LedgerValidationError("calendar sessions cannot be repeated across segments")
    effective_boundary = _local_boundary(
        sessions[0], state.config["execution_open_local"], state.config["exchange_timezone"]
    )
    if record.recorded_at_utc >= effective_boundary:
        raise LedgerTimingError("calendar extension must be appended before its first new session opens")
    retrieved = _canonical_utc(segment["source_retrieved_at_utc"], "source_retrieved_at_utc")
    if retrieved > record.recorded_at_utc:
        raise LedgerTimingError("calendar source retrieval exceeds the record timestamp")
    if segment["source_id"] != state.config["calendar_source_id"]:
        raise LedgerValidationError("calendar segment source_id differs from genesis")
    receipt = _validate_receipt_shape(external_anchor)
    if receipt["provider"] != state.anchor_identity["provider"] or receipt["key_id"] != state.anchor_identity["key_id"]:
        raise ReceiptVerificationError("calendar receipt signer differs from genesis")
    claimed_issued_at = _canonical_utc(receipt["issued_at_utc"], "receipt.issued_at_utc")
    if claimed_issued_at < retrieved or claimed_issued_at >= effective_boundary:
        raise ReceiptVerificationError("calendar receipt is outside retrieval-to-first-open interval")
    if verified_receipt is not None:
        _verify_receipt_timing(
            verified_receipt,
            recorded_at_utc=record.recorded_at_utc,
            earliest_utc=retrieved,
            latest_utc=effective_boundary,
        )
    state.segments.append(
        {
            "segment_index": segment["segment_index"],
            "segment": segment,
            "commitment_sha256": commitment,
            "record_hash": record.record_hash,
        }
    )
    state.sessions.extend(sessions)
    state.sessions.sort()


def _apply_decision(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "decision", record.recorded_at_utc)
    payload = _exact_mapping(
        record.data["payload"], {"week_id", "decision_dt", "execution_dt", "close_dt", "data"}, "decision payload"
    )
    week = _expected_week(state)
    if week is None:
        raise LedgerValidationError("no fully covered calendar week is available")
    expected_identity = (
        week.week_id,
        week.decision_dt.isoformat(),
        week.execution_dt.isoformat(),
        week.close_dt.isoformat(),
    )
    actual_identity = (payload["week_id"], payload["decision_dt"], payload["execution_dt"], payload["close_dt"])
    if actual_identity != expected_identity:
        raise LedgerValidationError("decision does not match the next actual complete calendar week")
    data = _validate_event_data(payload["data"], DECISION_SCHEMA, _DECISION_KEYS, "decision.data")
    if data["engine_sha256"] != state.config["dependency_sha256"]["research_engine"]:
        raise LedgerValidationError("decision engine differs from the frozen research engine")
    if data["portfolio_before_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("decision portfolio_before_sha256 differs from chain state")
    if data["pending_sells_before_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("decision pending_sells_before_sha256 differs from chain state")
    if data["calendar_state_sha256"] != week.calendar_state_sha256:
        raise LedgerValidationError("decision calendar_state_sha256 differs from anchored calendar state")
    earliest = _local_boundary(
        week.decision_dt, state.config["decision_close_local"], state.config["exchange_timezone"]
    )
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "decision.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.active_week = week
    state.phase = "DECIDED"
    state.decision_record = record
    state.last_open_record = None
    state.last_eod_record = None
    state.next_session_index = 0


def _apply_open_execution(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "session_open_execution", record.recorded_at_utc)
    assert state.active_week is not None and state.decision_record is not None
    session = state.active_week.sessions[state.next_session_index]
    payload = _exact_mapping(
        record.data["payload"],
        {"week_id", "session_dt", "decision_hash", "data"},
        "session_open_execution payload",
    )
    if payload["week_id"] != state.active_week.week_id or payload["session_dt"] != session.isoformat():
        raise LedgerValidationError("session_open_execution session identity mismatch")
    if payload["decision_hash"] != state.decision_record.record_hash:
        raise LedgerValidationError("session_open_execution does not reference the cycle decision")
    data = _validate_event_data(
        payload["data"], OPEN_EXECUTION_SCHEMA, _OPEN_EXECUTION_KEYS, "session_open_execution.data"
    )
    if data["engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("session_open_execution engine differs from the frozen execution engine")
    if data["decision_record_sha256"] != state.decision_record.record_hash:
        raise LedgerValidationError("session_open_execution decision hash fields disagree")
    expected_eod = state.last_eod_record.record_hash if state.last_eod_record is not None else ZERO_HASH
    if data["previous_eod_record_sha256"] != expected_eod:
        raise LedgerValidationError("session_open_execution previous EOD reference mismatch")
    if data["portfolio_before_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("session_open_execution pre-open portfolio differs from chain state")
    before_count = _validate_count(data["pending_sell_count_before"], "pending_sell_count_before")
    if before_count != state.pending_sell_count or data["pending_sells_before_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("session_open_execution pre-open pending sells differ from chain state")
    after_count = _validate_count(data["pending_sell_count_after"], "pending_sell_count_after")
    if state.next_session_index > 0 and after_count > before_count:
        raise LedgerValidationError("pending sell count cannot increase without a new cycle decision")
    earliest = _local_boundary(session, state.config["execution_open_local"], state.config["exchange_timezone"])
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "session_open_execution.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.portfolio_sha256 = data["portfolio_after_sha256"]
    state.pending_sells_sha256 = data["pending_sells_after_sha256"]
    state.pending_sell_count = after_count
    state.last_open_record = record
    state.phase = "OPENED"


def _apply_eod(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "session_eod_valuation", record.recorded_at_utc)
    assert state.active_week is not None and state.last_open_record is not None
    session = state.active_week.sessions[state.next_session_index]
    payload = _exact_mapping(record.data["payload"], {"week_id", "session_dt", "data"}, "session_eod_valuation payload")
    if payload["week_id"] != state.active_week.week_id or payload["session_dt"] != session.isoformat():
        raise LedgerValidationError("session_eod_valuation session identity mismatch")
    data = _validate_event_data(payload["data"], EOD_VALUATION_SCHEMA, _EOD_KEYS, "session_eod_valuation.data")
    if data["engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("session_eod_valuation engine differs from frozen execution engine")
    if data["session_open_execution_sha256"] != state.last_open_record.record_hash:
        raise LedgerValidationError("session_eod_valuation does not reference its same-session open record")
    count = _validate_count(data["pending_sell_count"], "pending_sell_count")
    if count != state.pending_sell_count or data["pending_sells_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("session_eod_valuation pending-sell state differs from chain state")
    if data["portfolio_before_eod_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("session_eod_valuation pre-EOD portfolio differs from chain state")
    earliest = _local_boundary(session, state.config["session_close_local"], state.config["exchange_timezone"])
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "session_eod_valuation.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.portfolio_sha256 = data["portfolio_state_sha256"]
    state.last_eod_record = record
    if state.next_session_index == len(state.active_week.sessions) - 1:
        state.phase = "FINAL_EOD"
    else:
        state.next_session_index += 1
        state.phase = "AWAIT_SESSION_OPEN"


def _apply_weekly_close(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "cycle_close", record.recorded_at_utc)
    assert state.active_week is not None and state.last_eod_record is not None and state.decision_record is not None
    payload = _exact_mapping(record.data["payload"], {"week_id", "close_dt", "data"}, "weekly_close payload")
    if payload["week_id"] != state.active_week.week_id or payload["close_dt"] != state.active_week.close_dt.isoformat():
        raise LedgerValidationError("weekly_close week identity mismatch")
    data = _validate_event_data(payload["data"], WEEKLY_CLOSE_SCHEMA, _WEEKLY_CLOSE_KEYS, "weekly_close.data")
    if data["execution_engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("weekly_close execution engine differs from genesis")
    if data["statistics_engine_sha256"] != state.config["dependency_sha256"]["statistics_engine"]:
        raise LedgerValidationError("weekly_close statistics engine differs from genesis")
    if data["eod_record_sha256"] != state.last_eod_record.record_hash:
        raise LedgerValidationError("weekly_close does not reference the final EOD valuation")
    if data["portfolio_state_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("weekly_close portfolio state differs from chain state")
    if data["pending_sells_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("weekly_close pending-sell digest differs from chain state")
    if _validate_count(data["pending_sell_count"], "pending_sell_count") != state.pending_sell_count:
        raise LedgerValidationError("weekly_close pending-sell count differs from chain state")
    earliest = state.last_eod_record.recorded_at_utc
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "weekly_close.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.completed_weeks.append(
        {
            "week_id": state.active_week.week_id,
            "week_start": state.active_week.week_start.isoformat(),
            "decision_dt": state.active_week.decision_dt.isoformat(),
            "execution_dt": state.active_week.execution_dt.isoformat(),
            "close_dt": state.active_week.close_dt.isoformat(),
            "sessions": [item.isoformat() for item in state.active_week.sessions],
            "calendar_state_sha256": state.active_week.calendar_state_sha256,
            "decision_record_sha256": state.decision_record.record_hash,
            "weekly_close_record_sha256": record.record_hash,
            "weekly_close_sequence": record.sequence,
        }
    )
    state.active_week = None
    state.phase = "IDLE"
    state.decision_record = None
    state.last_open_record = None
    state.last_eod_record = None
    state.next_session_index = 0


def _apply_terminal(state: _ReplayState, record: LedgerRecord, deadline_failure: str | None) -> None:
    data = _exact_mapping(record.data["payload"], _TERMINAL_KEYS, "terminal payload")
    if data["schema"] != TERMINAL_SCHEMA:
        raise LedgerValidationError(f"terminal.schema must be {TERMINAL_SCHEMA!r}")
    if data["outcome"] not in {"ABORTED", "FAILED"}:
        raise LedgerValidationError("terminal.outcome must be ABORTED or FAILED")
    if not isinstance(data["reason_code"], str) or re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", data["reason_code"]) is None:
        raise LedgerValidationError("terminal.reason_code must be a registered uppercase code")
    _require_sha256(data["evidence_sha256"], "terminal.evidence_sha256")
    _require_sha256(data["details_sha256"], "terminal.details_sha256")
    if deadline_failure is not None and (data["outcome"] != "FAILED" or data["reason_code"] != deadline_failure):
        raise LedgerValidationError("an expired chain may only record its exact missed-deadline failure")
    state.terminal = record


def _validate_genesis_record(
    record: LedgerRecord,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None,
) -> _ReplayState:
    if record.sequence != 0 or record.record_type != "genesis" or record.data["previous_record_hash"] != ZERO_HASH:
        raise LedgerValidationError("sequence 000000 must be the zero-linked genesis record")
    payload = _exact_mapping(
        record.data["payload"],
        {
            "schema",
            "config",
            "primary_chain_id",
            "genesis_commitment_sha256",
            "anchor_verifier_identity",
            "external_anchor",
        },
        "genesis payload",
    )
    if payload["schema"] != GENESIS_PAYLOAD_SCHEMA:
        raise LedgerValidationError(f"genesis payload schema must be {GENESIS_PAYLOAD_SCHEMA!r}")
    config = _validate_genesis_config(payload["config"])
    chain_id = _primary_chain_id(config)
    if payload["primary_chain_id"] != chain_id:
        raise LedgerValidationError("genesis primary_chain_id mismatch")
    identity = _exact_mapping(payload["anchor_verifier_identity"], _ANCHOR_IDENTITY_KEYS, "anchor_verifier_identity")
    if identity["algorithm"] != ANCHOR_ALGORITHM:
        raise LedgerValidationError("genesis anchor algorithm is unsupported")
    _require_sha256(identity["public_key_sha256"], "anchor_verifier_identity.public_key_sha256")
    core = {
        "schema": "xs_chan_v2_1_genesis_commitment_v1",
        "config": config,
        "primary_chain_id": chain_id,
        "anchor_verifier_identity": identity,
    }
    commitment = _sha256_bytes(_canonical_bytes(core))
    if payload["genesis_commitment_sha256"] != commitment:
        raise LedgerValidationError("genesis commitment mismatch")
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    start_boundary = _local_boundary(start, "00:00:00", config["exchange_timezone"])
    if record.recorded_at_utc >= start_boundary:
        raise LedgerTimingError("genesis must be recorded before oos_start_date begins")
    receipt = _validate_receipt_shape(payload["external_anchor"])
    if receipt["provider"] != identity["provider"] or receipt["key_id"] != identity["key_id"]:
        raise ReceiptVerificationError("genesis receipt signer differs from its anchor identity")
    if receipt["subject_sha256"] != commitment:
        raise ReceiptVerificationError("genesis receipt subject differs from its commitment")
    claimed_issued_at = _canonical_utc(receipt["issued_at_utc"], "receipt.issued_at_utc")
    if claimed_issued_at > record.recorded_at_utc or claimed_issued_at >= start_boundary:
        raise ReceiptVerificationError("genesis receipt is outside its registered append interval")
    receipts_verified = False
    if receipt_verifier is not None:
        if receipt_verifier.identity != identity:
            raise ReceiptVerificationError("provided verifier differs from the genesis trust root")
        verified = receipt_verifier.verify(receipt, expected_subject_sha256=commitment)
        _verify_receipt_timing(verified, recorded_at_utc=record.recorded_at_utc, latest_utc=start_boundary)
        receipts_verified = True
    return _ReplayState(
        records=[record],
        genesis=record,
        config=config,
        anchor_identity=identity,
        receipts_verified=receipts_verified,
        portfolio_sha256=config["initial_portfolio_state_sha256"],
        pending_sells_sha256=_object_sha256({}),
        pending_sell_count=0,
    )


def _unwrap_anchored_record(
    state: _ReplayState,
    record: LedgerRecord,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None,
) -> tuple[LedgerRecord, dict[str, Any], VerifiedReceipt | None]:
    wrapper = _exact_mapping(record.data["payload"], _ANCHORED_PAYLOAD_KEYS, "anchored record payload")
    if not isinstance(wrapper["event"], Mapping):
        raise LedgerValidationError("anchored record event must be an object")
    event = dict(wrapper["event"])
    commitment = _record_commitment_sha256(
        primary_chain_id=_primary_chain_id(state.config),
        previous_record_hash=record.data["previous_record_hash"],
        record_type=record.record_type,
        event_payload=event,
    )
    if wrapper["record_commitment_sha256"] != commitment:
        raise LedgerValidationError("anchored record commitment mismatch")
    receipt = _validate_receipt_shape(wrapper["external_anchor"])
    if receipt["provider"] != state.anchor_identity["provider"] or receipt["key_id"] != state.anchor_identity["key_id"]:
        raise ReceiptVerificationError("record receipt signer differs from genesis")
    if receipt["subject_sha256"] != commitment:
        raise ReceiptVerificationError("record receipt subject differs from its commitment")
    issued_at = _canonical_utc(receipt["issued_at_utc"], "receipt.issued_at_utc")
    if issued_at < state.records[-1].recorded_at_utc or issued_at > record.recorded_at_utc:
        raise ReceiptVerificationError("record receipt is outside the prior-head to append-time interval")
    causal_types = {"decision", "session_open_execution", "session_eod_valuation", "cycle_close"}
    if record.record_type in causal_types:
        window = _event_window(state)
        if window is None or window[0] != record.record_type:
            raise LedgerValidationError(f"{record.record_type} is not the next required causal record")
        _, earliest, deadline = window
        if issued_at < earliest or issued_at >= deadline:
            raise ReceiptVerificationError("record receipt is outside the registered market-event window")
    verified: VerifiedReceipt | None = None
    if receipt_verifier is not None:
        if receipt_verifier.identity != state.anchor_identity:
            raise ReceiptVerificationError("provided verifier differs from the genesis trust root")
        verified = receipt_verifier.verify(receipt, expected_subject_sha256=commitment)
    event_data = {**record.data, "payload": event}
    return LedgerRecord(path=record.path, data=event_data, raw=record.raw), receipt, verified


def _validate_chain(
    root: Path,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None,
) -> _ReplayState:
    records = _scan_records(root)
    state = _validate_genesis_record(records[0], receipt_verifier=receipt_verifier)
    for record in records[1:]:
        if state.terminal is not None:
            raise LedgerValidationError("no record may follow an ABORTED or FAILED terminal record")
        if len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
            raise LedgerValidationError("no record may follow the frozen 52-week structural window")
        expired = _deadline_failure(state, record.recorded_at_utc)
        if record.record_type != "chain_abort" and expired is not None:
            raise LedgerValidationError(f"chain irreversibly failed before this record: {expired}")
        event_record, receipt, verified_receipt = _unwrap_anchored_record(
            state, record, receipt_verifier=receipt_verifier
        )
        if record.record_type == "calendar_extension":
            _apply_calendar_segment(
                state,
                event_record,
                external_anchor=receipt,
                verified_receipt=verified_receipt,
            )
        elif record.record_type == "decision":
            _apply_decision(state, event_record)
        elif record.record_type == "session_open_execution":
            _apply_open_execution(state, event_record)
        elif record.record_type == "session_eod_valuation":
            _apply_eod(state, event_record)
        elif record.record_type == "cycle_close":
            _apply_weekly_close(state, event_record)
        elif record.record_type == "chain_abort":
            _apply_terminal(state, event_record, expired)
        else:
            raise LedgerValidationError(f"unsupported record_type: {record.record_type!r}")
        state.records.append(record)
    return state


def _root_identity(root: Path) -> str:
    return _sha256_bytes(_canonical_bytes({"resolved_ledger_root": str(root.resolve())}))


def _claim_path(registry_root: Path, config: Mapping[str, Any]) -> Path:
    # One claim per protocol hash, not per trial.  A failed trial therefore
    # cannot be replaced under the same preregistered protocol.
    return registry_root / "primary" / f"{config['protocol_sha256']}.json"


def _build_registry_claim(
    root: Path, genesis: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    body = {
        "schema": REGISTRY_CLAIM_SCHEMA,
        "trial_registry_namespace": config["trial_registry_namespace"],
        "trial_registry_authority_sha256": config["trial_registry_authority_sha256"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "chain_role": "PRIMARY",
        "primary_chain_id": genesis["payload"]["primary_chain_id"],
        "ledger_root_identity_sha256": _root_identity(root),
        "genesis_commitment_sha256": genesis["payload"]["genesis_commitment_sha256"],
        "genesis_record_sha256": genesis["record_hash"],
    }
    claim = {**body, "claim_sha256": _sha256_bytes(_canonical_bytes(body))}
    return claim, _canonical_bytes(claim)


def _validate_registry_claim(registry_root: Path, root: Path, state: _ReplayState) -> None:
    path = _claim_path(registry_root, state.config)
    if not path.is_file():
        raise LedgerValidationError("primary-chain registry claim is missing")
    raw = path.read_bytes()
    try:
        claim = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerValidationError("primary-chain registry claim is invalid JSON") from exc
    expected, expected_raw = _build_registry_claim(root, state.genesis.data, state.config)
    if raw != expected_raw or claim != expected:
        raise LedgerValidationError("primary-chain registry claim differs from genesis or ledger root")


def init_ledger(
    root: str | Path,
    *,
    genesis_config: Mapping[str, Any],
    trial_registry_root: str | Path,
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Create the one externally anchored primary chain for this protocol."""
    if not isinstance(receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("init_ledger requires a concrete RSA receipt verifier")
    root_path = Path(root)
    registry_root = Path(trial_registry_root)
    config = _validate_genesis_config(genesis_config)
    commitment = genesis_commitment_sha256(config, receipt_verifier=receipt_verifier)
    recorded_at = _normalise_utc(_utc_now(), "clock")
    verified = receipt_verifier.verify(external_anchor, expected_subject_sha256=commitment)
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    start_boundary = _local_boundary(start, "00:00:00", config["exchange_timezone"])
    _verify_receipt_timing(verified, recorded_at_utc=recorded_at, latest_utc=start_boundary)
    if recorded_at >= start_boundary:
        raise LedgerTimingError("genesis must be recorded before oos_start_date begins")
    payload = {
        "schema": GENESIS_PAYLOAD_SCHEMA,
        "config": config,
        "primary_chain_id": _primary_chain_id(config),
        "genesis_commitment_sha256": commitment,
        "anchor_verifier_identity": receipt_verifier.identity,
        "external_anchor": dict(external_anchor),
    }
    record, raw = _build_record(0, "genesis", recorded_at, ZERO_HASH, payload)
    path = _record_path(root_path, 0, record["record_hash"])
    claim, claim_raw = _build_registry_claim(root_path, record, config)
    claim_path = _claim_path(registry_root, config)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        _exclusive_lock(registry_root, ".registry.lock"),
        _exclusive_lock(root_path, ".ledger.lock"),
    ):
        json_files = list(root_path.glob("*.json"))
        if claim_path.exists() or json_files:
            if claim_path.is_file() and len(json_files) == 1:
                existing = _validate_chain(root_path, receipt_verifier=receipt_verifier)
                _validate_registry_claim(registry_root, root_path, existing)
                if existing.genesis.raw == raw and claim_path.read_bytes() == claim_raw:
                    return existing.genesis.path
            raise LedgerConflictError("protocol already has a primary chain or ledger root is initialized")
        _exclusive_write(claim_path, claim_raw)
        try:
            _exclusive_write(path, raw)
        except Exception:
            claim_path.unlink(missing_ok=True)
            raise
    return path


def _append_record(
    root: Path,
    *,
    record_type: str,
    payload: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
    apply: Any,
) -> Path:
    if not isinstance(receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("append requires the concrete genesis RSA receipt verifier")
    recorded_at = _normalise_utc(_utc_now(), "clock")
    with _exclusive_lock(root, ".ledger.lock"):
        state = _validate_chain(root, receipt_verifier=receipt_verifier)
        if state.terminal is not None:
            raise LedgerValidationError("chain is terminal and cannot be extended")
        if len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
            raise LedgerValidationError("the frozen 52-week window is complete and cannot be extended")
        expired = _deadline_failure(state, recorded_at)
        if record_type != "chain_abort" and expired is not None:
            raise LedgerValidationError(f"chain is irreversibly failed: {expired}")
        if recorded_at <= state.records[-1].recorded_at_utc:
            raise LedgerValidationError("recorded_at_utc must be strictly increasing")
        commitment = _record_commitment_sha256(
            primary_chain_id=_primary_chain_id(state.config),
            previous_record_hash=state.records[-1].record_hash,
            record_type=record_type,
            event_payload=payload,
        )
        verified_receipt = receipt_verifier.verify(external_anchor, expected_subject_sha256=commitment)
        if (
            verified_receipt.issued_at_utc < state.records[-1].recorded_at_utc
            or verified_receipt.issued_at_utc > recorded_at
        ):
            raise ReceiptVerificationError("record receipt is outside the prior-head to append-time interval")
        if record_type in {"decision", "session_open_execution", "session_eod_valuation", "cycle_close"}:
            window = _event_window(state)
            if window is None or window[0] != record_type:
                raise LedgerValidationError(f"{record_type} is not the next required causal record")
            _, earliest, deadline = window
            if verified_receipt.issued_at_utc < earliest or verified_receipt.issued_at_utc >= deadline:
                raise ReceiptVerificationError("record receipt is outside the registered market-event window")
        wrapper = {
            "event": dict(payload),
            "record_commitment_sha256": commitment,
            "external_anchor": dict(external_anchor),
        }
        record_data, raw = _build_record(
            len(state.records), record_type, recorded_at, state.records[-1].record_hash, wrapper
        )
        stored_record = LedgerRecord(
            path=_record_path(root, len(state.records), record_data["record_hash"]),
            data=record_data,
            raw=raw,
        )
        event_record = LedgerRecord(
            path=stored_record.path,
            data={**record_data, "payload": dict(payload)},
            raw=raw,
        )
        apply(state, event_record, expired)
        _exclusive_write(stored_record.path, raw)
    return stored_record.path


def append_calendar_segment(
    root: str | Path,
    *,
    segment: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Append one contiguous official-calendar segment before it takes effect."""
    segment_value = _validate_segment(segment)

    def apply(state: _ReplayState, record: LedgerRecord, _: str | None) -> None:
        verified = receipt_verifier.verify(
            external_anchor,
            expected_subject_sha256=_require_sha256(
                external_anchor.get("subject_sha256"), "external_anchor.subject_sha256"
            ),
        )
        _apply_calendar_segment(
            state,
            record,
            external_anchor=external_anchor,
            verified_receipt=verified,
        )

    root_path = Path(root)
    state = _validate_chain(root_path, receipt_verifier=receipt_verifier)
    commitment = calendar_segment_commitment_sha256(_primary_chain_id(state.config), segment_value)
    payload = {
        "segment_index": segment_value["segment_index"],
        "segment": segment_value,
        "commitment_sha256": commitment,
    }
    return _append_record(
        root_path,
        record_type="calendar_extension",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=apply,
    )


def prepare_calendar_extension(root: str | Path, *, segment: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare an exact official-calendar extension for external anchoring."""
    root_path = Path(root)
    state = _validate_chain(root_path, receipt_verifier=None)
    segment_value = _validate_segment(segment)
    event = {
        "segment_index": segment_value["segment_index"],
        "segment": segment_value,
        "commitment_sha256": calendar_segment_commitment_sha256(_primary_chain_id(state.config), segment_value),
    }
    return prepare_record_commitment(root_path, record_type="calendar_extension", event_payload=event)


def _event_payload(root: Path, week_id: str, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
    state = _validate_chain(root, receipt_verifier=None)
    week = state.active_week if state.active_week is not None else _expected_week(state)
    if week is None or not isinstance(week_id, str) or _WEEK_ID.fullmatch(week_id) is None or week.week_id != week_id:
        raise LedgerValidationError("week_id does not identify the currently expected week")
    if kind == "decision":
        return {
            "week_id": week.week_id,
            "decision_dt": week.decision_dt.isoformat(),
            "execution_dt": week.execution_dt.isoformat(),
            "close_dt": week.close_dt.isoformat(),
            "data": dict(data),
        }
    if kind == "session_open_execution":
        if state.decision_record is None:
            raise LedgerValidationError("session_open_execution has no active cycle decision")
        session = week.sessions[state.next_session_index]
        return {
            "week_id": week.week_id,
            "session_dt": session.isoformat(),
            "decision_hash": state.decision_record.record_hash,
            "data": dict(data),
        }
    if kind == "session_eod_valuation":
        session = week.sessions[state.next_session_index]
        return {"week_id": week.week_id, "session_dt": session.isoformat(), "data": dict(data)}
    if kind == "cycle_close":
        return {"week_id": week.week_id, "close_dt": week.close_dt.isoformat(), "data": dict(data)}
    raise LedgerValidationError(f"unsupported event kind: {kind}")


def prepare_record_commitment(
    root: str | Path,
    *,
    record_type: str,
    event_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare the exact next-head commitment for an external timestamp signer.

    The returned commitment is valid only while ``previous_record_hash`` remains
    the chain head.  Append re-computes it under the ledger lock, so a racing
    append fails closed instead of reusing a receipt on another history.
    """
    state = _validate_chain(Path(root), receipt_verifier=None)
    if record_type not in {
        "calendar_extension",
        "decision",
        "session_open_execution",
        "session_eod_valuation",
        "cycle_close",
        "chain_abort",
    }:
        raise LedgerValidationError(f"unsupported formal record_type: {record_type!r}")
    previous = state.records[-1].record_hash
    commitment = _record_commitment_sha256(
        primary_chain_id=_primary_chain_id(state.config),
        previous_record_hash=previous,
        record_type=record_type,
        event_payload=event_payload,
    )
    return {
        "record_type": record_type,
        "previous_record_hash": previous,
        "event": dict(event_payload),
        "record_commitment_sha256": commitment,
    }


def prepare_cycle_event(
    root: str | Path,
    *,
    record_type: str,
    week_id: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare a decision/open/EOD/close event for external anchoring."""
    if record_type not in {
        "decision",
        "session_open_execution",
        "session_eod_valuation",
        "cycle_close",
    }:
        raise LedgerValidationError("record_type is not a cycle event")
    event = _event_payload(Path(root), week_id, record_type, data)
    return prepare_record_commitment(root, record_type=record_type, event_payload=event)


def append_decision(
    root: str | Path,
    *,
    week_id: str,
    data: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Append the frozen decision after the prior session close and before next open."""
    root_path = Path(root)
    payload = _event_payload(root_path, week_id, "decision", data)
    return _append_record(
        root_path,
        record_type="decision",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, _: _apply_decision(state, record),
    )


def append_open_execution(
    root: str | Path,
    *,
    week_id: str,
    data: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Append the mandatory open heartbeat for the current covered session."""
    root_path = Path(root)
    payload = _event_payload(root_path, week_id, "session_open_execution", data)
    return _append_record(
        root_path,
        record_type="session_open_execution",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, _: _apply_open_execution(state, record),
    )


def append_eod_valuation(
    root: str | Path,
    *,
    week_id: str,
    data: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Append one mandatory end-of-session valuation for the active week."""
    root_path = Path(root)
    payload = _event_payload(root_path, week_id, "session_eod_valuation", data)
    return _append_record(
        root_path,
        record_type="session_eod_valuation",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, _: _apply_eod(state, record),
    )


def append_weekly_close(
    root: str | Path,
    *,
    week_id: str,
    data: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Close the active actual calendar week after its final EOD valuation."""
    root_path = Path(root)
    payload = _event_payload(root_path, week_id, "cycle_close", data)
    return _append_record(
        root_path,
        record_type="cycle_close",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, _: _apply_weekly_close(state, record),
    )


def append_terminal(
    root: str | Path,
    *,
    outcome: str,
    reason_code: str,
    evidence_sha256: str,
    details_sha256: str,
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Irreversibly terminate the primary chain as ABORTED or FAILED."""
    payload = {
        "schema": TERMINAL_SCHEMA,
        "outcome": outcome,
        "reason_code": reason_code,
        "evidence_sha256": evidence_sha256,
        "details_sha256": details_sha256,
    }
    return _append_record(
        Path(root),
        record_type="chain_abort",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, expired: _apply_terminal(state, record, expired),
    )


def read_ledger_records(root: str | Path) -> tuple[dict[str, Any], ...]:
    """Return freshly decoded hash-verified records for a semantic replay engine."""
    return tuple(json.loads(record.raw) for record in _scan_records(Path(root)))


def _confirmation_window(state: _ReplayState) -> dict[str, Any] | None:
    if len(state.completed_weeks) < MIN_COMPLETE_WEEKS:
        return None
    week_rows = state.completed_weeks[:MIN_COMPLETE_WEEKS]
    head_sequence = int(week_rows[-1]["weekly_close_sequence"])
    records = state.records[: head_sequence + 1]
    identity = {
        "schema": CONFIRMATION_IDENTITY_SCHEMA,
        "primary_chain_id": _primary_chain_id(state.config),
        "protocol_sha256": state.config["protocol_sha256"],
        "data_bundle_sha256": state.config["data_bundle_sha256"],
        "dependency_sha256": state.config["dependency_sha256"],
        "readiness_gates": state.config["readiness_gates"],
        "trial_id": state.config["trial_id"],
        "weeks": [{key: value for key, value in row.items() if key != "weekly_close_sequence"} for row in week_rows],
        "calendar_segment_commitment_sha256": [item["commitment_sha256"] for item in state.segments],
        "record_hashes_through_52nd_close": [record.record_hash for record in records],
        "confirmation_head_sha256": records[-1].record_hash,
    }
    return {**identity, "identity_sha256": _sha256_bytes(_canonical_bytes(identity))}


def verify_ledger(
    root: str | Path,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None = None,
    trial_registry_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify structure and receipts while remaining closed on semantic claims."""
    root_path = Path(root)
    state = _validate_chain(root_path, receipt_verifier=receipt_verifier)
    registry_verified = False
    if trial_registry_root is not None:
        _validate_registry_claim(Path(trial_registry_root), root_path, state)
        registry_verified = True
    receipts_verified = receipt_verifier is not None and state.receipts_verified
    expired = None if state.terminal is not None else _deadline_failure(state, _normalise_utc(_utc_now(), "clock"))
    if state.terminal is not None:
        operational_status = state.terminal.data["payload"]["outcome"]
    elif len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        operational_status = "STRUCTURAL_WINDOW_COMPLETE"
    elif expired is not None:
        operational_status = "FAILED_BY_DEADLINE"
    else:
        operational_status = "ACTIVE"
    structural_window_complete = len(state.completed_weeks) >= MIN_COMPLETE_WEEKS
    return {
        "valid": receipts_verified and registry_verified,
        "structurally_valid": True,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": state.config["protocol_id"],
        "protocol_sha256": state.config["protocol_sha256"],
        "data_bundle_sha256": state.config["data_bundle_sha256"],
        "primary_chain_id": _primary_chain_id(state.config),
        "trial_id": state.config["trial_id"],
        "record_count": len(state.records),
        "calendar_segment_count": len(state.segments),
        "calendar_state_sha256": _calendar_state_sha256(state.segments),
        "completed_weeks": len(state.completed_weeks),
        "min_complete_weeks": MIN_COMPLETE_WEEKS,
        "operational_status": operational_status,
        "deadline_failure": expired,
        "external_receipts_verified": receipts_verified,
        "primary_registry_verified": registry_verified,
        "structural_window_complete": structural_window_complete,
        "semantic_replay_verified": False,
        "confirmatory_oos": False,
        "next_required_event": _event_window(state)[0] if _event_window(state) is not None else None,
        "chain_head_sha256": state.records[-1].record_hash,
        "confirmation_window": _confirmation_window(state),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerValidationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise LedgerValidationError(f"JSON input must be an object: {path}")
    return value


def _load_verifier(path: Path) -> RsaPkcs1v15Sha256ReceiptVerifier:
    value = _exact_mapping(
        _load_json(path),
        {"provider", "key_id", "modulus_hex", "public_exponent"},
        "anchor public key",
    )
    return RsaPkcs1v15Sha256ReceiptVerifier(**value)


def build_parser() -> argparse.ArgumentParser:
    """Build the operational CLI; no command accepts a historical timestamp."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("root", type=Path)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--trial-registry", type=Path, required=True)
    init.add_argument("--external-anchor", type=Path, required=True)
    init.add_argument("--anchor-public-key", type=Path, required=True)

    append = commands.add_parser("append")
    append_types = append.add_subparsers(dest="append_type", required=True)
    for kind in ("calendar", "decision", "open-execution", "eod-valuation", "cycle-close"):
        command = append_types.add_parser(kind)
        command.add_argument("root", type=Path)
        command.add_argument("--payload", type=Path, required=True)
        command.add_argument("--anchor-public-key", type=Path, required=True)
    terminal = append_types.add_parser("terminal")
    terminal.add_argument("root", type=Path)
    terminal.add_argument("--payload", type=Path, required=True)
    terminal.add_argument("--anchor-public-key", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("--trial-registry", type=Path)
    verify.add_argument("--anchor-public-key", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V2.1 ledger CLI without any backdating escape hatch."""
    args = build_parser().parse_args(argv)
    if args.command == "init":
        verifier = _load_verifier(args.anchor_public_key)
        path = init_ledger(
            args.root,
            genesis_config=_load_json(args.config),
            trial_registry_root=args.trial_registry,
            external_anchor=_load_json(args.external_anchor),
            receipt_verifier=verifier,
        )
        output = {"path": str(path), "record_hash": json.loads(path.read_bytes())["record_hash"]}
    elif args.command == "verify":
        verifier = _load_verifier(args.anchor_public_key) if args.anchor_public_key is not None else None
        output = verify_ledger(
            args.root,
            receipt_verifier=verifier,
            trial_registry_root=args.trial_registry,
        )
    else:
        verifier = _load_verifier(args.anchor_public_key)
        payload = _load_json(args.payload)
        if args.append_type == "calendar":
            value = _exact_mapping(payload, {"segment", "external_anchor"}, "calendar CLI payload")
            path = append_calendar_segment(
                args.root,
                segment=value["segment"],
                external_anchor=value["external_anchor"],
                receipt_verifier=verifier,
            )
        elif args.append_type == "terminal":
            value = _exact_mapping(payload, _TERMINAL_KEYS | {"external_anchor"}, "terminal CLI payload")
            value.pop("schema")
            external_anchor = value.pop("external_anchor")
            value["external_anchor"] = external_anchor
            path = append_terminal(args.root, receipt_verifier=verifier, **value)
        else:
            value = _exact_mapping(payload, {"week_id", "data", "external_anchor"}, "event CLI payload")
            function = {
                "decision": append_decision,
                "open-execution": append_open_execution,
                "eod-valuation": append_eod_valuation,
                "cycle-close": append_weekly_close,
            }[args.append_type]
            path = function(
                args.root,
                week_id=value["week_id"],
                data=value["data"],
                external_anchor=value["external_anchor"],
                receipt_verifier=verifier,
            )
        output = {"path": str(path), "record_hash": json.loads(path.read_bytes())["record_hash"]}
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
