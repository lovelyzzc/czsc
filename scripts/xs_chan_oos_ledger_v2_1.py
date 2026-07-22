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
import math
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

SCHEMA_VERSION = 22
PROTOCOL_ID = "xs_chan_pilot_v2_1_preregistered_20260720"
CANONICAL_PROTOCOL_SHA256 = "8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203"
ZERO_HASH = "0" * 64
MIN_COMPLETE_WEEKS = 52
EXCHANGE_TIMEZONE = "Asia/Shanghai"
DECISION_CLOSE_LOCAL = "15:00:00"
EXECUTION_OPEN_LOCAL = "09:30:00"
OPEN_EXECUTION_DEADLINE_LOCAL = "15:00:00"
SESSION_CLOSE_LOCAL = "15:00:00"

GENESIS_CONFIG_SCHEMA = "xs_chan_v2_1_genesis_config_v2"
GENESIS_PAYLOAD_SCHEMA = "xs_chan_v2_1_genesis_v2"
PORTFOLIO_STATE_SCHEMA = "xs_chan_portfolio_state_v2_1"
CALENDAR_SEGMENT_SCHEMA = "xs_chan_v2_1_calendar_segment_v2"
ANCHOR_RECEIPT_SCHEMA = "xs_chan_external_anchor_receipt_v1"
ANCHOR_ALGORITHM = "rsa-pkcs1v15-sha256"
REGISTRY_CLAIM_SCHEMA = "xs_chan_v2_1_primary_chain_claim_v2"
REGISTRY_DOCUMENT_SCHEMA = "xs_chan_v2_1_primary_chain_registry_document_v1"
DECISION_SCHEMA = "xs_chan_v2_1_decision_v2"
UNWIND_DECISION_SCHEMA = "xs_chan_v2_1_post_window_unwind_decision_v1"
OPEN_EXECUTION_SCHEMA = "xs_chan_v2_1_session_open_execution_v2"
EOD_VALUATION_SCHEMA = "xs_chan_v2_1_session_eod_valuation_v2"
UNWIND_EOD_VALUATION_SCHEMA = "xs_chan_v2_1_post_window_unwind_eod_valuation_v1"
WEEKLY_CLOSE_SCHEMA = "xs_chan_v2_1_cycle_close_v2"
TERMINAL_SCHEMA = "xs_chan_v2_1_terminal_v1"
FINAL_EVALUATION_SCHEMA = "xs_chan_v2_1_final_evaluation_v1"
CONFIRMATION_IDENTITY_SCHEMA = "xs_chan_v2_1_confirmation_window_identity_v1"
READINESS_COMMITMENT_SCHEMA = "xs_chan_pre_candidate_readiness_commitment_v2_1"

FORMAL_LIFECYCLE_STATES = frozenset(
    {
        "LOCKED_PENDING_ENGINEERING_AND_FORWARD_DATA",
        "READY_TO_ANCHOR_PRIMARY_GENESIS",
        "PRIMARY_FORWARD_COLLECTION_ACTIVE",
        "PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY",
        "PRIMARY_CHAIN_TERMINATED_INVALID",
        "EVALUATED",
    }
)
UNWIND_WEEK_ID = "POST_WINDOW_UNWIND"
PREDECESSOR_PROTOCOL_SHA256 = "0ec5cb260f2aec78a6dce838880140981fdd1a64b5c9a2dcf64be010ba5c8b62"
RNG_ALGORITHM_VERSION = "xs_chan_rng_v2_1_sha256_seedsequence_v1"
RNG_ROOT_SEED = 20260720
RNG_SEEDS = tuple(range(RNG_ROOT_SEED, RNG_ROOT_SEED + 20))
SCENARIO_IDS = ("gross", "1x", "2x", "capacity_1x")
SINGLE_BOOK_IDS = ("F", "FC", "FMA")
SEEDED_BOOK_FAMILIES = ("R_match", "FGR", "FMGR")
BOOK_IDS = SINGLE_BOOK_IDS + tuple(f"{family}@{seed}" for family in SEEDED_BOOK_FAMILIES for seed in RNG_SEEDS)
INITIAL_CAPITAL_CNY_BY_SCENARIO = {
    "gross": 10_000_000.0,
    "1x": 10_000_000.0,
    "2x": 10_000_000.0,
    "capacity_1x": 100_000_000.0,
}

EXPECTED_DEPENDENCY_KEYS = frozenset(
    {
        "data_evidence",
        "state_engine",
        "feature_engine",
        "research_engine",
        "execution_engine",
        "ledger_engine",
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
    "predecessor_protocol_sha256",
    "data_bundle_sha256",
    "data_contract_identity_sha256",
    "genesis_data_chain_head_sha256",
    "genesis_data_validation_report_sha256",
    "dependency_sha256",
    "readiness_gates",
    "readiness_report_sha256",
    "initial_state_sha256_by_book_scenario",
    "initial_state_root_sha256",
    "rng_identity",
    "formal_start_eligibility",
    "genesis_calendar_segment",
    "genesis_calendar_head_sha256",
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
    "share_entitlements",
    "processed_action_ids",
    "other_assets",
    "other_liabilities",
    "borrowed_cash_cny",
}
_READINESS_EVIDENCE_KEYS = {"status", "evidence_sha256"}
_READINESS_COMMITMENT_KEYS = {
    "schema",
    "protocol_id",
    "protocol_sha256",
    "trial_id",
    "data_contract_identity_sha256",
    "created_at_utc",
    "dependency_sha256",
    "engineering_gates",
    "report_sha256",
}
_RNG_IDENTITY_KEYS = {"algorithm_version", "root_seed", "seed_count", "seeds", "identity_freeze"}
_FORMAL_START_ELIGIBILITY_KEYS = {
    "schema",
    "readiness_report_sha256",
    "readiness_anchor_receipt_sha256",
    "readiness_completed_at_utc",
    "eligibility_cutoff_utc",
    "candidate_decision_session",
    "candidate_execution_week",
    "candidate_eligible_count",
    "minimum_eligible_symbol_count",
    "earlier_completed_weeks",
    "calendar_head_sha256",
    "data_snapshot_sha256",
    "planner_verifier_sha256",
    "evidence_sha256",
}
_EARLIER_ELIGIBILITY_WEEK_KEYS = {"week_id", "decision_session", "eligible_symbol_count", "evidence_sha256"}
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
    "week_labels",
    "source_id",
    "source_uri",
    "source_sha256",
    "source_published_at_utc",
    "source_retrieved_at_utc",
    "previous_calendar_head_sha256",
    "new_calendar_head_sha256",
}
_DECISION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "decision_artifact_sha256",
    "book_state_before_root_sha256",
    "pending_sells_before_root_sha256",
    "calendar_state_sha256",
    "eligible_symbol_count",
    "eligibility_evidence_sha256",
    "data_chain_head_sha256",
    "data_manifest_sha256",
    "previous_data_chain_head_sha256",
    "data_validation_report_sha256",
    "planner_artifact_sha256",
}
_UNWIND_DECISION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "data_manifest_sha256",
    "previous_data_chain_head_sha256",
    "data_validation_report_sha256",
    "data_chain_head_sha256",
    "confirmation_window_identity_sha256",
    "book_state_before_root_sha256",
    "pending_sells_before_root_sha256",
    "aggregate_position_count_before",
    "aggregate_pending_sell_count_before",
    "exit_orders_root_sha256",
    "full_exit_coverage_root_sha256",
    "book_state_after_decision_root_sha256",
    "pending_sells_after_decision_root_sha256",
    "aggregate_pending_sell_count_after_decision",
    "unwind_evidence_root_sha256",
}
_OPEN_EXECUTION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "decision_record_sha256",
    "previous_eod_record_sha256",
    "book_state_before_root_sha256",
    "pending_sells_before_root_sha256",
    "pending_sell_count_before",
    "aggregate_position_count_before",
    "open_prices_sha256",
    "opening_auction_turnover_sha256",
    "limit_state_sha256",
    "corporate_actions_sha256",
    "terminal_share_receipts_root_sha256",
    "terminal_share_receipt_count",
    "requested_orders_root_sha256",
    "fills_root_sha256",
    "fees_root_sha256",
    "execution_result_root_sha256",
    "book_state_after_root_sha256",
    "pending_sells_after_root_sha256",
    "pending_sell_count_after",
    "aggregate_position_count_after",
}
_EOD_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "engine_sha256",
    "session_open_execution_sha256",
    "raw_close_snapshot_sha256",
    "corporate_actions_sha256",
    "book_state_before_eod_root_sha256",
    "book_state_root_sha256",
    "pending_sells_root_sha256",
    "pending_sell_count",
    "aggregate_position_count",
    "nav_root_sha256",
    "exposure_root_sha256",
    "turnover_root_sha256",
}
_UNWIND_EOD_KEYS = _EOD_KEYS | {
    "data_manifest_sha256",
    "previous_data_chain_head_sha256",
    "data_validation_report_sha256",
    "data_chain_head_sha256",
    "aggregate_pending_settlements_root_sha256",
    "aggregate_pending_settlement_count",
}
_WEEKLY_CLOSE_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "execution_engine_sha256",
    "statistics_engine_sha256",
    "eod_record_sha256",
    "book_state_root_sha256",
    "pending_sells_root_sha256",
    "pending_sell_count",
    "aggregate_position_count",
    "weekly_returns_root_sha256",
    "statistics_input_root_sha256",
}
_TERMINAL_KEYS = {"schema", "outcome", "reason_code", "evidence_sha256", "details_sha256"}
_FINAL_EVALUATION_KEYS = {
    "schema",
    "snapshot_cutoff_utc",
    "confirmation_window_identity_sha256",
    "statistics_result_sha256",
    "semantic_replay_evidence_sha256",
    "evaluation_status",
    "post_window_unwind_completed",
    "post_window_unwind_session_count",
    "post_window_unwind_cost_cny",
    "remaining_position_count",
    "remaining_pending_sell_count",
    "remaining_pending_settlement_count",
    "unwind_completion_book_state_root_sha256",
    "unwind_completion_pending_sells_root_sha256",
    "unwind_completion_pending_settlements_root_sha256",
    "unwind_evidence_sha256",
    "final_evaluation_artifact_sha256",
}
FORMAL_EVALUATION_STATUSES = frozenset(
    {
        "INVALID_DATA_OR_ENGINEERING",
        "INVALID_CHAIN",
        "FORWARD_COLLECTION_REQUIRED",
        "INVALID_CONTROL",
        "INSUFFICIENT_INTERVENTION",
        "FALSIFIED_FACTOR",
        "INCONCLUSIVE_FACTOR",
        "FACTOR_PASSED_TIMING_FALSIFIED",
        "FACTOR_PASSED_TIMING_INCONCLUSIVE",
        "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY",
    }
)
_SHA_FIELDS_BY_SCHEMA = {
    DECISION_SCHEMA: _DECISION_KEYS - {"schema", "snapshot_cutoff_utc", "eligible_symbol_count"},
    UNWIND_DECISION_SCHEMA: _UNWIND_DECISION_KEYS
    - {
        "schema",
        "snapshot_cutoff_utc",
        "aggregate_position_count_before",
        "aggregate_pending_sell_count_before",
        "aggregate_pending_sell_count_after_decision",
    },
    OPEN_EXECUTION_SCHEMA: _OPEN_EXECUTION_KEYS
    - {
        "schema",
        "snapshot_cutoff_utc",
        "pending_sell_count_before",
        "pending_sell_count_after",
        "aggregate_position_count_before",
        "aggregate_position_count_after",
        "terminal_share_receipt_count",
    },
    EOD_VALUATION_SCHEMA: _EOD_KEYS
    - {"schema", "snapshot_cutoff_utc", "pending_sell_count", "aggregate_position_count"},
    UNWIND_EOD_VALUATION_SCHEMA: _UNWIND_EOD_KEYS
    - {
        "schema",
        "snapshot_cutoff_utc",
        "pending_sell_count",
        "aggregate_position_count",
        "aggregate_pending_settlement_count",
    },
    WEEKLY_CLOSE_SCHEMA: _WEEKLY_CLOSE_KEYS
    - {"schema", "snapshot_cutoff_utc", "pending_sell_count", "aggregate_position_count"},
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
    registry_receipt_verified: bool = False
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
    position_count: int = 0
    data_chain_head_sha256: str = ""
    unwind_evidence_root_sha256: str | None = None
    unwind_next_session_after: date | None = None
    unwind_session: date | None = None
    unwind_session_count: int = 0
    pending_settlements_sha256: str | None = None
    pending_settlement_count: int | None = None
    final_evaluation: LedgerRecord | None = None
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


def formal_start_eligibility_evidence_sha256(value: Mapping[str, Any]) -> str:
    """Derive the canonical digest of formal-start evidence, excluding itself."""

    body = dict(value)
    body.pop("evidence_sha256", None)
    return _object_sha256(body)


def data_chain_transition_sha256(
    *,
    data_contract_identity_sha256: str,
    previous_data_chain_head_sha256: str,
    decision_session: str | date,
    snapshot_cutoff_utc: str | datetime,
    data_manifest_sha256: str,
    data_validation_report_sha256: str,
) -> str:
    """Derive one append-only, decision-time data-chain head."""

    payload = {
        "schema": "xs_chan_data_chain_transition_v2_1",
        "data_contract_identity_sha256": _require_sha256(
            data_contract_identity_sha256, "data_contract_identity_sha256"
        ),
        "previous_data_chain_head_sha256": _require_sha256(
            previous_data_chain_head_sha256, "previous_data_chain_head_sha256"
        ),
        "decision_session": _normalise_date(decision_session, "decision_session").isoformat(),
        "snapshot_cutoff_utc": _format_utc(_canonical_utc(snapshot_cutoff_utc, "snapshot_cutoff_utc")),
        "data_manifest_sha256": _require_sha256(data_manifest_sha256, "data_manifest_sha256"),
        "data_validation_report_sha256": _require_sha256(
            data_validation_report_sha256, "data_validation_report_sha256"
        ),
    }
    return _object_sha256(payload)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an evidence artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_initial_portfolio(initial_capital_cny: float = 10_000_000.0) -> dict[str, Any]:
    """Return the canonical empty execution state for one scenario."""
    if (
        type(initial_capital_cny) not in {int, float}
        or not math.isfinite(float(initial_capital_cny))
        or not float(initial_capital_cny) > 0
    ):
        raise LedgerValidationError("initial_capital_cny must be a positive finite number")
    return {
        "schema": PORTFOLIO_STATE_SCHEMA,
        "cash_cny": float(initial_capital_cny),
        "positions": {},
        "pending_sells": {},
        "last_close": {},
        "cash_receivables": {},
        "dividend_tax_liabilities": {},
        "share_entitlements": {},
        "processed_action_ids": [],
        "other_assets": [],
        "other_liabilities": [],
        "borrowed_cash_cny": 0.0,
    }


def canonical_initial_state_sha256_by_book_scenario() -> dict[str, dict[str, str]]:
    """Return all 63 book x 4 scenario canonical initial-state digests."""
    scenario_hashes = {
        scenario: _object_sha256(empty_initial_portfolio(capital))
        for scenario, capital in INITIAL_CAPITAL_CNY_BY_SCENARIO.items()
    }
    return {book_id: {scenario: scenario_hashes[scenario] for scenario in SCENARIO_IDS} for book_id in BOOK_IDS}


def canonical_initial_state_root_sha256() -> str:
    """Return the aggregate root consumed by every later all-book state transition."""
    return _object_sha256(canonical_initial_state_sha256_by_book_scenario())


def canonical_initial_pending_sells_root_sha256() -> str:
    """Return the aggregate empty pending-sell root for all book/scenario leaves."""
    empty_pending = _object_sha256({})
    leaves = {book_id: dict.fromkeys(SCENARIO_IDS, empty_pending) for book_id in BOOK_IDS}
    return _object_sha256(leaves)


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


def _validate_initial_state_matrix(value: Any) -> dict[str, dict[str, str]]:
    books = _exact_mapping(value, set(BOOK_IDS), "initial_state_sha256_by_book_scenario")
    expected = canonical_initial_state_sha256_by_book_scenario()
    result: dict[str, dict[str, str]] = {}
    for book_id in BOOK_IDS:
        scenarios = _exact_mapping(
            books[book_id], set(SCENARIO_IDS), f"initial_state_sha256_by_book_scenario.{book_id}"
        )
        result[book_id] = {
            scenario: _require_sha256(scenarios[scenario], f"initial state {book_id}/{scenario}")
            for scenario in SCENARIO_IDS
        }
        if result[book_id] != expected[book_id]:
            raise LedgerValidationError(f"initial state differs from the canonical empty state for {book_id}")
    return result


def _validate_rng_identity(value: Any) -> dict[str, Any]:
    identity = _exact_mapping(value, _RNG_IDENTITY_KEYS, "rng_identity")
    expected = {
        "algorithm_version": RNG_ALGORITHM_VERSION,
        "root_seed": RNG_ROOT_SEED,
        "seed_count": len(RNG_SEEDS),
        "seeds": list(RNG_SEEDS),
        "identity_freeze": "all_arm_and_seed_symbol_identities_frozen_before_any_cost_or_capacity_replay",
    }
    if identity != expected:
        raise LedgerValidationError("rng_identity differs from the preregistered RNG identity")
    return identity


def _validate_formal_start_eligibility(
    value: Any,
    *,
    oos_start: date,
    candidate_decision_session: date,
    readiness_report_sha256: str,
    calendar_head_sha256: str,
    data_snapshot_sha256: str,
    planner_verifier_sha256: str,
    calendar_sessions: Sequence[date],
) -> dict[str, Any]:
    evidence = _exact_mapping(value, _FORMAL_START_ELIGIBILITY_KEYS, "formal_start_eligibility")
    if evidence["schema"] != "xs_chan_v2_1_formal_start_eligibility_v2":
        raise LedgerValidationError("formal_start_eligibility.schema is unsupported")
    cutoff = _canonical_utc(evidence["eligibility_cutoff_utc"], "formal_start_eligibility.eligibility_cutoff_utc")
    if cutoff >= _local_boundary(oos_start, "00:00:00", EXCHANGE_TIMEZONE):
        raise LedgerValidationError("formal-start eligibility evidence must predate oos_start_date")
    if cutoff != _local_boundary(candidate_decision_session, DECISION_CLOSE_LOCAL, EXCHANGE_TIMEZONE):
        raise LedgerValidationError("formal-start eligibility cutoff must equal the candidate decision close")
    readiness_completed = _canonical_utc(
        evidence["readiness_completed_at_utc"],
        "formal_start_eligibility.readiness_completed_at_utc",
    )
    if readiness_completed >= cutoff:
        raise LedgerValidationError("readiness must complete before the formal-start eligibility cutoff")
    readiness_local_date = readiness_completed.astimezone(ZoneInfo(EXCHANGE_TIMEZONE)).date()
    if readiness_local_date < min(calendar_sessions):
        raise LedgerValidationError("genesis calendar does not cover the full post-readiness eligibility domain")
    if evidence["minimum_eligible_symbol_count"] != 50:
        raise LedgerValidationError("formal-start minimum eligible symbol count must be 50")
    if type(evidence["candidate_eligible_count"]) is not int or evidence["candidate_eligible_count"] < 50:
        raise LedgerValidationError("formal-start eligible symbol count must be at least 50")
    if evidence["candidate_decision_session"] != candidate_decision_session.isoformat():
        raise LedgerValidationError("formal-start candidate decision is not the last official pre-OOS session")
    if evidence["candidate_execution_week"] != _week_id(oos_start):
        raise LedgerValidationError("formal-start candidate execution week differs from oos_start_date")
    expected_bindings = {
        "readiness_report_sha256": readiness_report_sha256,
        "calendar_head_sha256": calendar_head_sha256,
        "data_snapshot_sha256": data_snapshot_sha256,
        "planner_verifier_sha256": planner_verifier_sha256,
    }
    for key, expected in expected_bindings.items():
        if _require_sha256(evidence[key], f"formal_start_eligibility.{key}") != expected:
            raise LedgerValidationError(f"formal-start {key} differs from its Genesis binding")
    evidence["readiness_anchor_receipt_sha256"] = _require_sha256(
        evidence["readiness_anchor_receipt_sha256"],
        "formal_start_eligibility.readiness_anchor_receipt_sha256",
    )
    if type(evidence["earlier_completed_weeks"]) is not list:
        raise LedgerValidationError("formal-start earlier_completed_weeks must be a complete ordered list")
    expected_earlier: list[tuple[str, date]] = []
    monday = oos_start - timedelta(weeks=1)
    while monday >= min(calendar_sessions):
        prior_sessions = [item for item in calendar_sessions if item < monday]
        if prior_sessions:
            decision_session = prior_sessions[-1]
            decision_close = _local_boundary(decision_session, DECISION_CLOSE_LOCAL, EXCHANGE_TIMEZONE)
            if readiness_completed < decision_close < cutoff:
                expected_earlier.append((_week_id(monday), decision_session))
        monday -= timedelta(weeks=1)
    expected_earlier.reverse()
    if len(evidence["earlier_completed_weeks"]) != len(expected_earlier):
        raise LedgerValidationError("formal-start evidence omits or adds a readiness-to-candidate completed week")
    earlier: list[dict[str, Any]] = []
    previous_decision: date | None = None
    for index, (item, expected_week) in enumerate(
        zip(evidence["earlier_completed_weeks"], expected_earlier, strict=True)
    ):
        row = _exact_mapping(item, _EARLIER_ELIGIBILITY_WEEK_KEYS, f"earlier_completed_weeks[{index}]")
        decision_session = _normalise_date(row["decision_session"], "earlier eligibility decision_session")
        if decision_session >= candidate_decision_session:
            raise LedgerValidationError("earlier eligibility evidence must strictly precede the candidate")
        if previous_decision is not None and decision_session <= previous_decision:
            raise LedgerValidationError("earlier eligibility weeks must be strictly ordered")
        if (row["week_id"], decision_session) != expected_week:
            raise LedgerValidationError("earlier eligibility row differs from the complete anchored-calendar domain")
        count = _validate_count(row["eligible_symbol_count"], "earlier eligible_symbol_count")
        if count >= 50:
            raise LedgerValidationError(
                "candidate is not the earliest completed week with at least 50 eligible symbols"
            )
        row["evidence_sha256"] = _require_sha256(row["evidence_sha256"], "earlier eligibility evidence")
        row["decision_session"] = decision_session.isoformat()
        earlier.append(row)
        previous_decision = decision_session
    evidence["earlier_completed_weeks"] = earlier
    evidence["readiness_completed_at_utc"] = _format_utc(readiness_completed)
    evidence["evidence_sha256"] = _require_sha256(
        evidence["evidence_sha256"], "formal_start_eligibility.evidence_sha256"
    )
    evidence["eligibility_cutoff_utc"] = _format_utc(cutoff)
    if evidence["evidence_sha256"] != formal_start_eligibility_evidence_sha256(evidence):
        raise LedgerValidationError("formal-start eligibility evidence digest is not canonically derived")
    return evidence


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
    config["predecessor_protocol_sha256"] = _require_sha256(
        config["predecessor_protocol_sha256"], "predecessor_protocol_sha256"
    )
    if config["predecessor_protocol_sha256"] != PREDECESSOR_PROTOCOL_SHA256:
        raise LedgerValidationError("predecessor_protocol_sha256 differs from the frozen V2 predecessor")
    config["data_bundle_sha256"] = _require_sha256(config["data_bundle_sha256"], "data_bundle_sha256")
    config["data_contract_identity_sha256"] = _require_sha256(
        config["data_contract_identity_sha256"], "data_contract_identity_sha256"
    )
    config["genesis_data_chain_head_sha256"] = _require_sha256(
        config["genesis_data_chain_head_sha256"], "genesis_data_chain_head_sha256"
    )
    config["genesis_data_validation_report_sha256"] = _require_sha256(
        config["genesis_data_validation_report_sha256"], "genesis_data_validation_report_sha256"
    )
    dependencies = _exact_mapping(config["dependency_sha256"], EXPECTED_DEPENDENCY_KEYS, "dependency_sha256")
    config["dependency_sha256"] = {
        name: _require_sha256(dependencies[name], f"dependency_sha256.{name}")
        for name in sorted(EXPECTED_DEPENDENCY_KEYS)
    }
    config["readiness_gates"] = _validate_readiness_gates(config["readiness_gates"])
    config["readiness_report_sha256"] = _require_sha256(config["readiness_report_sha256"], "readiness_report_sha256")
    config["initial_state_sha256_by_book_scenario"] = _validate_initial_state_matrix(
        config["initial_state_sha256_by_book_scenario"]
    )
    config["initial_state_root_sha256"] = _require_sha256(
        config["initial_state_root_sha256"], "initial_state_root_sha256"
    )
    if config["initial_state_root_sha256"] != _object_sha256(config["initial_state_sha256_by_book_scenario"]):
        raise LedgerValidationError("initial_state_root_sha256 does not match all 252 book/scenario leaves")
    config["rng_identity"] = _validate_rng_identity(config["rng_identity"])
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
    genesis_segment = _validate_segment(config["genesis_calendar_segment"])
    if genesis_segment["segment_index"] != 0:
        raise LedgerValidationError("genesis calendar segment_index must be zero")
    if genesis_segment["previous_calendar_head_sha256"] != ZERO_HASH:
        raise LedgerValidationError("genesis calendar must start from the zero calendar head")
    if genesis_segment["source_id"] != config["calendar_source_id"]:
        raise LedgerValidationError("genesis calendar source differs from calendar_source_id")
    if config["genesis_calendar_head_sha256"] != genesis_segment["new_calendar_head_sha256"]:
        raise LedgerValidationError("genesis_calendar_head_sha256 differs from the first segment head")
    segment_sessions = [_normalise_date(item, "genesis calendar session") for item in genesis_segment["sessions"]]
    first_week_end = start + timedelta(days=6)
    if not any(item < start for item in segment_sessions):
        raise LedgerValidationError("genesis calendar must include the prior decision session")
    if not any(start <= item <= first_week_end for item in segment_sessions):
        raise LedgerValidationError("genesis calendar must include the first execution week")
    if _normalise_date(genesis_segment["coverage_end"], "coverage_end") < first_week_end:
        raise LedgerValidationError("genesis calendar must cover through the first execution week")
    covered_oos_weeks = {
        label for item, label in zip(segment_sessions, genesis_segment["week_labels"], strict=True) if item >= start
    }
    confirmation_horizon_end = start + timedelta(weeks=MIN_COMPLETE_WEEKS) - timedelta(days=1)
    if (
        len(covered_oos_weeks) >= MIN_COMPLETE_WEEKS
        or _normalise_date(genesis_segment["coverage_end"], "coverage_end") >= confirmation_horizon_end
    ):
        raise LedgerValidationError("genesis may not preload the future 52-week confirmation calendar")
    config["genesis_calendar_segment"] = genesis_segment
    config["genesis_calendar_head_sha256"] = _require_sha256(
        config["genesis_calendar_head_sha256"], "genesis_calendar_head_sha256"
    )
    candidate_decision = max(item for item in segment_sessions if item < start)
    config["formal_start_eligibility"] = _validate_formal_start_eligibility(
        config["formal_start_eligibility"],
        oos_start=start,
        candidate_decision_session=candidate_decision,
        readiness_report_sha256=config["readiness_report_sha256"],
        calendar_head_sha256=config["genesis_calendar_head_sha256"],
        data_snapshot_sha256=config["data_bundle_sha256"],
        planner_verifier_sha256=config["dependency_sha256"]["research_engine"],
        calendar_sessions=segment_sessions,
    )
    expected_genesis_data_head = data_chain_transition_sha256(
        data_contract_identity_sha256=config["data_contract_identity_sha256"],
        previous_data_chain_head_sha256=ZERO_HASH,
        decision_session=config["formal_start_eligibility"]["candidate_decision_session"],
        snapshot_cutoff_utc=config["formal_start_eligibility"]["eligibility_cutoff_utc"],
        data_manifest_sha256=config["data_bundle_sha256"],
        data_validation_report_sha256=config["genesis_data_validation_report_sha256"],
    )
    if config["genesis_data_chain_head_sha256"] != expected_genesis_data_head:
        raise LedgerValidationError("genesis_data_chain_head_sha256 is not derived from the Genesis data evidence")
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
        "schema": "xs_chan_v2_1_primary_chain_identity_v2",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "trial_registry_namespace": config["trial_registry_namespace"],
        "trial_registry_authority_sha256": config["trial_registry_authority_sha256"],
        "chain_role": "PRIMARY",
        "genesis_config_sha256": _sha256_bytes(_canonical_bytes(config)),
    }
    return _sha256_bytes(_canonical_bytes(identity))


def _registry_authority_sha256(verifier: RsaPkcs1v15Sha256ReceiptVerifier) -> str:
    if not isinstance(verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("a concrete registry receipt verifier is required")
    return _object_sha256(verifier.identity)


def _primary_registry_claim_core(
    genesis_config: Mapping[str, Any],
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> dict[str, Any]:
    if not isinstance(receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("a concrete timestamp receipt verifier is required")
    config = _validate_genesis_config(genesis_config)
    authority_sha256 = _registry_authority_sha256(registry_receipt_verifier)
    if config["trial_registry_authority_sha256"] != authority_sha256:
        raise ReceiptVerificationError("registry verifier differs from the authority frozen by Genesis")
    genesis_core = {
        "schema": "xs_chan_v2_1_genesis_core_v1",
        "config": config,
        "primary_chain_id": _primary_chain_id(config),
        "anchor_verifier_identity": receipt_verifier.identity,
    }
    return {
        "schema": REGISTRY_CLAIM_SCHEMA,
        "registry_operation": "ATOMIC_REGISTER_PRIMARY_IF_ABSENT",
        "claim_result": "REGISTERED_AS_FIRST_PRIMARY",
        "uniqueness_scope_sha256": _object_sha256(
            {
                "trial_registry_namespace": config["trial_registry_namespace"],
                "trial_id": config["trial_id"],
                "chain_role": "PRIMARY",
            }
        ),
        "trial_registry_namespace": config["trial_registry_namespace"],
        "trial_registry_authority_sha256": authority_sha256,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "chain_role": "PRIMARY",
        "primary_chain_id": _primary_chain_id(config),
        "genesis_core_sha256": _sha256_bytes(_canonical_bytes(genesis_core)),
        "registry_verifier_identity": registry_receipt_verifier.identity,
    }


def primary_registry_claim_commitment_sha256(
    genesis_config: Mapping[str, Any],
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> str:
    """Return the immutable external primary-registry claim subject."""
    core = _primary_registry_claim_core(
        genesis_config,
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    return _sha256_bytes(_canonical_bytes(core))


def genesis_commitment_sha256(
    genesis_config: Mapping[str, Any],
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
    registry_external_anchor: Mapping[str, Any],
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> str:
    """Return the Genesis timestamp subject after verifying the external registry claim."""
    config = _validate_genesis_config(genesis_config)
    registry_claim = primary_registry_claim_commitment_sha256(
        config,
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    registry_receipt = registry_receipt_verifier.verify(
        registry_external_anchor,
        expected_subject_sha256=registry_claim,
    )
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    start_boundary = _local_boundary(start, "00:00:00", config["exchange_timezone"])
    retrieved = _canonical_utc(
        config["genesis_calendar_segment"]["source_retrieved_at_utc"],
        "genesis calendar retrieved_at",
    )
    eligibility_cutoff = _canonical_utc(
        config["formal_start_eligibility"]["eligibility_cutoff_utc"],
        "formal-start eligibility cutoff",
    )
    if (
        registry_receipt.issued_at_utc < max(retrieved, eligibility_cutoff)
        or registry_receipt.issued_at_utc >= start_boundary
    ):
        raise ReceiptVerificationError("primary registry receipt is outside the Genesis eligibility interval")
    core = {
        "schema": "xs_chan_v2_1_genesis_commitment_v2",
        "config": config,
        "primary_chain_id": _primary_chain_id(config),
        "anchor_verifier_identity": receipt_verifier.identity,
        "registry_verifier_identity": registry_receipt_verifier.identity,
        "primary_registry_claim_commitment_sha256": registry_claim,
        "primary_registry_external_anchor": dict(registry_external_anchor),
        "primary_registry_external_anchor_receipt_sha256": registry_receipt.receipt_sha256,
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
    if type(segment["week_labels"]) is not list or len(segment["week_labels"]) != len(sessions):
        raise LedgerValidationError("calendar segment week_labels must parallel sessions exactly")
    expected_week_labels = [_week_id(item) for item in sessions]
    if segment["week_labels"] != expected_week_labels:
        raise LedgerValidationError("calendar week_labels must be the canonical ISO week for each session")
    if not isinstance(segment["source_id"], str) or not segment["source_id"].strip():
        raise LedgerValidationError("calendar_segment.source_id must be non-empty")
    uri = segment["source_uri"]
    if not isinstance(uri, str):
        raise LedgerValidationError("calendar_segment.source_uri must be an HTTPS URL")
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "https" or not parsed_uri.netloc:
        raise LedgerValidationError("calendar_segment.source_uri must be an HTTPS URL")
    segment["source_sha256"] = _require_sha256(segment["source_sha256"], "calendar_segment.source_sha256")
    published = _canonical_utc(segment["source_published_at_utc"], "calendar_segment.source_published_at_utc")
    retrieved = _canonical_utc(segment["source_retrieved_at_utc"], "calendar_segment.source_retrieved_at_utc")
    if published > retrieved:
        raise LedgerValidationError("calendar source published_at cannot follow retrieved_at")
    segment["previous_calendar_head_sha256"] = _require_sha256(
        segment["previous_calendar_head_sha256"], "calendar_segment.previous_calendar_head_sha256"
    )
    segment["new_calendar_head_sha256"] = _require_sha256(
        segment["new_calendar_head_sha256"], "calendar_segment.new_calendar_head_sha256"
    )
    segment["coverage_start"] = start.isoformat()
    segment["coverage_end"] = end.isoformat()
    segment["sessions"] = [item.isoformat() for item in sessions]
    segment["week_labels"] = expected_week_labels
    segment["source_published_at_utc"] = _format_utc(published)
    segment["source_retrieved_at_utc"] = _format_utc(retrieved)
    expected_head = calendar_segment_head_sha256(segment)
    if segment["new_calendar_head_sha256"] != expected_head:
        raise LedgerValidationError("calendar segment new head does not match its canonical content")
    return segment


def calendar_segment_head_sha256(segment: Mapping[str, Any]) -> str:
    """Compute the rolling official-calendar head without trusting ``new_calendar_head_sha256``."""
    if not isinstance(segment, Mapping):
        raise LedgerValidationError("calendar_segment must be an object")
    value = _exact_mapping(segment, _CALENDAR_SEGMENT_KEYS, "calendar_segment")
    previous = _require_sha256(value["previous_calendar_head_sha256"], "previous_calendar_head_sha256")
    core_segment = {key: value[key] for key in sorted(_CALENDAR_SEGMENT_KEYS - {"new_calendar_head_sha256"})}
    core = {
        "schema": "xs_chan_v2_1_calendar_head_transition_v1",
        "previous_calendar_head_sha256": previous,
        "segment": core_segment,
    }
    return _sha256_bytes(_canonical_bytes(core))


def calendar_segment_commitment_sha256(primary_chain_id: str, segment: Mapping[str, Any]) -> str:
    """Return the signed commitment for one official-calendar segment."""
    chain_id = _require_sha256(primary_chain_id, "primary_chain_id")
    core = {
        "schema": "xs_chan_v2_1_calendar_segment_commitment_v2",
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
    if not segments:
        return ZERO_HASH
    return _require_sha256(
        segments[-1]["segment"]["new_calendar_head_sha256"],
        "latest calendar head",
    )


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


def _next_unwind_session(state: _ReplayState) -> date | None:
    if len(state.completed_weeks) < MIN_COMPLETE_WEEKS:
        return None
    after = state.unwind_next_session_after
    if after is None:
        after = _normalise_date(state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["close_dt"], "close_dt")
    return next((session for session in state.sessions if session > after), None)


def _event_window(state: _ReplayState) -> tuple[str, datetime, datetime] | None:
    if state.terminal is not None or state.final_evaluation is not None:
        return None
    timezone = state.config["exchange_timezone"]
    if len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        if state.phase == "WINDOW_COMPLETE":
            session = _next_unwind_session(state)
            if session is None:
                return None
            close_sequence = state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["weekly_close_sequence"]
            return (
                "decision",
                state.records[close_sequence].recorded_at_utc,
                _local_boundary(session, state.config["execution_open_local"], timezone),
            )
        if state.phase in {"UNWIND_DECIDED", "UNWIND_AWAIT_SESSION_OPEN"}:
            session = _next_unwind_session(state)
            if session is None:
                return None
            return (
                "session_open_execution",
                _local_boundary(session, state.config["execution_open_local"], timezone),
                _local_boundary(session, state.config["open_execution_deadline_local"], timezone),
            )
        if state.phase == "UNWIND_OPENED":
            if state.unwind_session is None:
                raise LedgerValidationError("post-window unwind session is missing")
            later_sessions = [item for item in state.sessions if item > state.unwind_session]
            return (
                "session_eod_valuation",
                _local_boundary(state.unwind_session, state.config["session_close_local"], timezone),
                _local_boundary(later_sessions[0], state.config["execution_open_local"], timezone)
                if later_sessions
                else datetime.max.replace(tzinfo=UTC),
            )
        if state.phase == "UNWIND_COMPLETE_PENDING_REPLAY":
            return None
        raise LedgerValidationError(f"unknown post-window replay phase: {state.phase}")
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
        return (
            "session_eod_valuation",
            _local_boundary(session, state.config["session_close_local"], timezone),
            _local_boundary(later_sessions[0], state.config["execution_open_local"], timezone)
            if later_sessions
            else datetime.max.replace(tzinfo=UTC),
        )
    if state.phase == "FINAL_EOD":
        assert state.active_week is not None
        return (
            "cycle_close",
            state.last_eod_record.recorded_at_utc
            if state.last_eod_record is not None
            else datetime.min.replace(tzinfo=UTC),
            datetime.max.replace(tzinfo=UTC),
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


def _advance_data_chain(
    state: _ReplayState,
    data: Mapping[str, Any],
    *,
    decision_session: str,
    label: str,
) -> None:
    if data["previous_data_chain_head_sha256"] != state.data_chain_head_sha256:
        raise LedgerValidationError(f"{label} previous data-chain head differs from chain state")
    expected = data_chain_transition_sha256(
        data_contract_identity_sha256=state.config["data_contract_identity_sha256"],
        previous_data_chain_head_sha256=state.data_chain_head_sha256,
        decision_session=decision_session,
        snapshot_cutoff_utc=data["snapshot_cutoff_utc"],
        data_manifest_sha256=data["data_manifest_sha256"],
        data_validation_report_sha256=data["data_validation_report_sha256"],
    )
    if data["data_chain_head_sha256"] != expected:
        raise LedgerValidationError(f"{label} data-chain head is not derived from its immutable snapshot transition")
    state.data_chain_head_sha256 = expected


def _apply_calendar_segment(
    state: _ReplayState,
    record: LedgerRecord,
    *,
    external_anchor: Mapping[str, Any],
    verified_receipt: VerifiedReceipt | None,
) -> None:
    if state.phase == "UNWIND_COMPLETE_PENDING_REPLAY":
        raise LedgerValidationError(
            "the irreversible all-book unwind-completion point permits only final evaluation or abort"
        )
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
    current_head = _calendar_state_sha256(state.segments)
    if segment["previous_calendar_head_sha256"] != current_head:
        raise LedgerValidationError("calendar extension previous head differs from the current calendar head")
    commitment = calendar_segment_commitment_sha256(_primary_chain_id(state.config), segment)
    if payload["commitment_sha256"] != commitment:
        raise LedgerValidationError("calendar segment commitment_sha256 mismatch")
    start = _normalise_date(segment["coverage_start"], "coverage_start")
    if not state.segments:
        raise LedgerValidationError("calendar segment zero must be committed inside Genesis")
    previous_end = _normalise_date(state.segments[-1]["segment"]["coverage_end"], "coverage_end")
    if start != previous_end + timedelta(days=1):
        raise LedgerValidationError("calendar coverage segments must be contiguous without overlap or gaps")
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
    published = _canonical_utc(segment["source_published_at_utc"], "source_published_at_utc")
    if retrieved > record.recorded_at_utc:
        raise LedgerTimingError("calendar source retrieval exceeds the record timestamp")
    if segment["source_id"] != state.config["calendar_source_id"]:
        raise LedgerValidationError("calendar segment source_id differs from genesis")
    receipt = _validate_receipt_shape(external_anchor)
    if receipt["provider"] != state.anchor_identity["provider"] or receipt["key_id"] != state.anchor_identity["key_id"]:
        raise ReceiptVerificationError("calendar receipt signer differs from genesis")
    claimed_issued_at = _canonical_utc(receipt["issued_at_utc"], "receipt.issued_at_utc")
    if claimed_issued_at < max(retrieved, published) or claimed_issued_at >= effective_boundary:
        raise ReceiptVerificationError("calendar receipt is outside retrieval-to-first-open interval")
    if verified_receipt is not None:
        _verify_receipt_timing(
            verified_receipt,
            recorded_at_utc=record.recorded_at_utc,
            earliest_utc=max(retrieved, published),
            latest_utc=effective_boundary,
        )
    state.segments.append(
        {
            "segment_index": segment["segment_index"],
            "segment": segment,
            "commitment_sha256": commitment,
            "record_hash": record.record_hash,
            "record_sequence": record.sequence,
        }
    )
    state.sessions.extend(sessions)
    state.sessions.sort()


def _apply_decision(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "decision", record.recorded_at_utc)
    if len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        payload = _exact_mapping(
            record.data["payload"],
            {"week_id", "decision_kind", "decision_dt", "execution_dt", "data"},
            "post-window unwind decision payload",
        )
        session = _next_unwind_session(state)
        if session is None:
            raise LedgerValidationError("post-window unwind decision has no covered execution session")
        expected = (
            UNWIND_WEEK_ID,
            "POST_WINDOW_UNWIND",
            state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["close_dt"],
            session.isoformat(),
        )
        actual = (
            payload["week_id"],
            payload["decision_kind"],
            payload["decision_dt"],
            payload["execution_dt"],
        )
        if actual != expected:
            raise LedgerValidationError("post-window unwind decision identity mismatch")
        data = _validate_event_data(
            payload["data"],
            UNWIND_DECISION_SCHEMA,
            _UNWIND_DECISION_KEYS,
            "post-window unwind decision.data",
        )
        if data["engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
            raise LedgerValidationError("post-window unwind decision uses the wrong execution engine")
        window = _confirmation_window(state)
        if window is None or data["confirmation_window_identity_sha256"] != window["identity_sha256"]:
            raise LedgerValidationError("post-window unwind decision does not bind the confirmation window")
        if data["book_state_before_root_sha256"] != state.portfolio_sha256:
            raise LedgerValidationError("post-window unwind pre-state differs from the all-book root")
        if data["pending_sells_before_root_sha256"] != state.pending_sells_sha256:
            raise LedgerValidationError("post-window unwind pending root differs from chain state")
        if (
            _validate_count(data["aggregate_position_count_before"], "aggregate_position_count_before")
            != state.position_count
        ):
            raise LedgerValidationError("post-window unwind position count differs from chain state")
        if (
            _validate_count(data["aggregate_pending_sell_count_before"], "aggregate_pending_sell_count_before")
            != state.pending_sell_count
        ):
            raise LedgerValidationError("post-window unwind pending count differs from chain state")
        after_pending_count = _validate_count(
            data["aggregate_pending_sell_count_after_decision"],
            "aggregate_pending_sell_count_after_decision",
        )
        if after_pending_count != state.position_count:
            raise LedgerValidationError(
                "post-window unwind decision must place every aggregate position into exit/pending"
            )
        expected_exit_coverage = _object_sha256(
            {
                "book_state_before_root_sha256": state.portfolio_sha256,
                "aggregate_position_count_before": state.position_count,
                "exit_orders_root_sha256": data["exit_orders_root_sha256"],
            }
        )
        if data["full_exit_coverage_root_sha256"] != expected_exit_coverage:
            raise LedgerValidationError("post-window unwind full-exit coverage root is not ledger-derived")
        close_sequence = state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["weekly_close_sequence"]
        _validate_snapshot(
            data["snapshot_cutoff_utc"],
            "post-window unwind decision.data.snapshot_cutoff_utc",
            earliest=state.records[close_sequence].recorded_at_utc,
            recorded_at=record.recorded_at_utc,
        )
        _advance_data_chain(
            state,
            data,
            decision_session=payload["decision_dt"],
            label="post-window unwind decision",
        )
        state.decision_record = record
        state.unwind_evidence_root_sha256 = data["unwind_evidence_root_sha256"]
        state.portfolio_sha256 = data["book_state_after_decision_root_sha256"]
        state.pending_sells_sha256 = data["pending_sells_after_decision_root_sha256"]
        state.pending_sell_count = after_pending_count
        state.phase = "UNWIND_DECIDED"
        return
    payload = _exact_mapping(
        record.data["payload"],
        {"week_id", "decision_kind", "decision_dt", "execution_dt", "close_dt", "data"},
        "decision payload",
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
    actual_identity = (
        payload["week_id"],
        payload["decision_dt"],
        payload["execution_dt"],
        payload["close_dt"],
    )
    if actual_identity != expected_identity:
        raise LedgerValidationError("decision does not match the next actual complete calendar week")
    if payload["decision_kind"] != "CONFIRMATORY_CYCLE":
        raise LedgerValidationError("confirmatory decision_kind must be CONFIRMATORY_CYCLE")
    data = _validate_event_data(payload["data"], DECISION_SCHEMA, _DECISION_KEYS, "decision.data")
    if data["engine_sha256"] != state.config["dependency_sha256"]["research_engine"]:
        raise LedgerValidationError("decision engine differs from the frozen research engine")
    if data["book_state_before_root_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("decision all-book pre-state root differs from chain state")
    if data["pending_sells_before_root_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("decision pending-sell root differs from chain state")
    if data["calendar_state_sha256"] != week.calendar_state_sha256:
        raise LedgerValidationError("decision calendar_state_sha256 differs from anchored calendar state")
    if _validate_count(data["eligible_symbol_count"], "decision eligible_symbol_count") < 50:
        raise LedgerValidationError("eligible_symbol_count below 50 requires chain_abort, not a decision")
    earliest = _local_boundary(
        week.decision_dt, state.config["decision_close_local"], state.config["exchange_timezone"]
    )
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "decision.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    _advance_data_chain(
        state,
        data,
        decision_session=payload["decision_dt"],
        label="confirmatory decision",
    )
    state.active_week = week
    state.phase = "DECIDED"
    state.decision_record = record
    state.last_open_record = None
    state.last_eod_record = None
    state.next_session_index = 0


def _apply_open_execution(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "session_open_execution", record.recorded_at_utc)
    is_unwind = len(state.completed_weeks) >= MIN_COMPLETE_WEEKS
    if is_unwind:
        session = _next_unwind_session(state)
        if session is None:
            raise LedgerValidationError("no anchored calendar session is available for post-window unwind")
        expected_week_id = UNWIND_WEEK_ID
        if state.decision_record is None:
            raise LedgerValidationError("post-window unwind open has no exit decision")
        origin_hash = state.decision_record.record_hash
    else:
        assert state.active_week is not None and state.decision_record is not None
        session = state.active_week.sessions[state.next_session_index]
        expected_week_id = state.active_week.week_id
        origin_hash = state.decision_record.record_hash
    payload = _exact_mapping(
        record.data["payload"],
        {"week_id", "session_dt", "decision_hash", "data"},
        "session_open_execution payload",
    )
    if payload["week_id"] != expected_week_id or payload["session_dt"] != session.isoformat():
        raise LedgerValidationError("session_open_execution session identity mismatch")
    if payload["decision_hash"] != origin_hash:
        raise LedgerValidationError("session_open_execution does not reference its frozen decision/origin")
    data = _validate_event_data(
        payload["data"], OPEN_EXECUTION_SCHEMA, _OPEN_EXECUTION_KEYS, "session_open_execution.data"
    )
    if data["engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("session_open_execution engine differs from the frozen execution engine")
    if data["decision_record_sha256"] != origin_hash:
        raise LedgerValidationError("session_open_execution decision hash fields disagree")
    expected_eod = state.last_eod_record.record_hash if state.last_eod_record is not None else ZERO_HASH
    if data["previous_eod_record_sha256"] != expected_eod:
        raise LedgerValidationError("session_open_execution previous EOD reference mismatch")
    if data["book_state_before_root_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("session_open_execution pre-open portfolio differs from chain state")
    before_count = _validate_count(data["pending_sell_count_before"], "pending_sell_count_before")
    if (
        before_count != state.pending_sell_count
        or data["pending_sells_before_root_sha256"] != state.pending_sells_sha256
    ):
        raise LedgerValidationError("session_open_execution pre-open pending sells differ from chain state")
    before_positions = _validate_count(data["aggregate_position_count_before"], "aggregate_position_count_before")
    if before_positions != state.position_count:
        raise LedgerValidationError("session_open_execution position count differs from chain state")
    after_count = _validate_count(data["pending_sell_count_after"], "pending_sell_count_after")
    after_positions = _validate_count(data["aggregate_position_count_after"], "aggregate_position_count_after")
    share_receipts = _validate_count(data["terminal_share_receipt_count"], "terminal_share_receipt_count")
    if (is_unwind or state.next_session_index > 0) and after_count > before_count + share_receipts:
        raise LedgerValidationError("pending sell count increase exceeds terminal share receipts")
    if is_unwind and after_positions > before_positions + share_receipts:
        raise LedgerValidationError("post-window position increase exceeds terminal share receipts")
    earliest = _local_boundary(session, state.config["execution_open_local"], state.config["exchange_timezone"])
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "session_open_execution.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.portfolio_sha256 = data["book_state_after_root_sha256"]
    state.pending_sells_sha256 = data["pending_sells_after_root_sha256"]
    state.pending_sell_count = after_count
    state.position_count = after_positions
    state.last_open_record = record
    if is_unwind:
        state.unwind_session = session
        state.phase = "UNWIND_OPENED"
    else:
        state.phase = "OPENED"


def _apply_eod(state: _ReplayState, record: LedgerRecord) -> None:
    _validate_window(state, "session_eod_valuation", record.recorded_at_utc)
    is_unwind = len(state.completed_weeks) >= MIN_COMPLETE_WEEKS
    assert state.last_open_record is not None
    if is_unwind:
        if state.unwind_session is None:
            raise LedgerValidationError("post-window unwind EOD has no open session")
        session = state.unwind_session
        expected_week_id = UNWIND_WEEK_ID
    else:
        assert state.active_week is not None
        session = state.active_week.sessions[state.next_session_index]
        expected_week_id = state.active_week.week_id
    payload = _exact_mapping(record.data["payload"], {"week_id", "session_dt", "data"}, "session_eod_valuation payload")
    if payload["week_id"] != expected_week_id or payload["session_dt"] != session.isoformat():
        raise LedgerValidationError("session_eod_valuation session identity mismatch")
    data_schema = UNWIND_EOD_VALUATION_SCHEMA if is_unwind else EOD_VALUATION_SCHEMA
    data_keys = _UNWIND_EOD_KEYS if is_unwind else _EOD_KEYS
    data = _validate_event_data(payload["data"], data_schema, data_keys, "session_eod_valuation.data")
    if data["engine_sha256"] != state.config["dependency_sha256"]["execution_engine"]:
        raise LedgerValidationError("session_eod_valuation engine differs from frozen execution engine")
    if data["session_open_execution_sha256"] != state.last_open_record.record_hash:
        raise LedgerValidationError("session_eod_valuation does not reference its same-session open record")
    count = _validate_count(data["pending_sell_count"], "pending_sell_count")
    positions = _validate_count(data["aggregate_position_count"], "aggregate_position_count")
    if data["book_state_before_eod_root_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("session_eod_valuation pre-EOD portfolio differs from chain state")
    earliest = _local_boundary(session, state.config["session_close_local"], state.config["exchange_timezone"])
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "session_eod_valuation.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    if is_unwind:
        _advance_data_chain(
            state,
            data,
            decision_session=session.isoformat(),
            label="post-window unwind EOD",
        )
        state.pending_settlements_sha256 = data["aggregate_pending_settlements_root_sha256"]
        state.pending_settlement_count = _validate_count(
            data["aggregate_pending_settlement_count"],
            "aggregate_pending_settlement_count",
        )
    state.portfolio_sha256 = data["book_state_root_sha256"]
    state.pending_sells_sha256 = data["pending_sells_root_sha256"]
    state.pending_sell_count = count
    state.position_count = positions
    state.last_eod_record = record
    if is_unwind:
        state.unwind_session_count += 1
        state.unwind_next_session_after = session
        state.unwind_session = None
        state.last_open_record = None
        if state.position_count == 0 and state.pending_sell_count == 0 and state.pending_settlement_count == 0:
            state.phase = "UNWIND_COMPLETE_PENDING_REPLAY"
        else:
            state.phase = "UNWIND_AWAIT_SESSION_OPEN"
    elif state.next_session_index == len(state.active_week.sessions) - 1:
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
    if data["book_state_root_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("weekly_close portfolio state differs from chain state")
    if data["pending_sells_root_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("weekly_close pending-sell digest differs from chain state")
    if _validate_count(data["pending_sell_count"], "pending_sell_count") != state.pending_sell_count:
        raise LedgerValidationError("weekly_close pending-sell count differs from chain state")
    if _validate_count(data["aggregate_position_count"], "aggregate_position_count") != state.position_count:
        raise LedgerValidationError("weekly_close position count differs from chain state")
    earliest = state.last_eod_record.recorded_at_utc
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "weekly_close.data.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    decision_data = state.decision_record.data["payload"]["data"]
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
            "data_manifest_sha256": decision_data["data_manifest_sha256"],
            "data_chain_head_sha256": decision_data["data_chain_head_sha256"],
            "data_validation_report_sha256": decision_data["data_validation_report_sha256"],
            "weekly_close_record_sha256": record.record_hash,
            "weekly_close_sequence": record.sequence,
        }
    )
    completed_window = len(state.completed_weeks) == MIN_COMPLETE_WEEKS
    state.active_week = None
    state.phase = "WINDOW_COMPLETE" if completed_window else "IDLE"
    state.decision_record = None
    state.last_open_record = None
    if not completed_window:
        state.last_eod_record = None
    else:
        state.unwind_next_session_after = _normalise_date(
            state.completed_weeks[-1]["close_dt"], "confirmation close date"
        )
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


def _apply_final_evaluation(state: _ReplayState, record: LedgerRecord) -> None:
    if len(state.completed_weeks) != MIN_COMPLETE_WEEKS:
        raise LedgerValidationError("final_evaluation requires exactly 52 completed confirmatory cycles")
    if state.phase != "UNWIND_COMPLETE_PENDING_REPLAY":
        raise LedgerValidationError("final_evaluation requires the irreversible all-book unwind-completion phase")
    if state.unwind_evidence_root_sha256 is None or state.decision_record is None:
        raise LedgerValidationError("final_evaluation requires the explicit POST_WINDOW_UNWIND decision")
    data = _exact_mapping(record.data["payload"], _FINAL_EVALUATION_KEYS, "final_evaluation payload")
    if data["schema"] != FINAL_EVALUATION_SCHEMA:
        raise LedgerValidationError(f"final_evaluation.schema must be {FINAL_EVALUATION_SCHEMA!r}")
    window = _confirmation_window(state)
    if window is None or data["confirmation_window_identity_sha256"] != window["identity_sha256"]:
        raise LedgerValidationError("final_evaluation does not bind the frozen 52-cycle window")
    for key in (
        "confirmation_window_identity_sha256",
        "statistics_result_sha256",
        "semantic_replay_evidence_sha256",
        "unwind_completion_book_state_root_sha256",
        "unwind_completion_pending_sells_root_sha256",
        "unwind_evidence_sha256",
        "final_evaluation_artifact_sha256",
    ):
        _require_sha256(data[key], f"final_evaluation.{key}")
    if data["evaluation_status"] not in FORMAL_EVALUATION_STATUSES:
        raise LedgerValidationError("final_evaluation.evaluation_status is not a registered formal status")
    if data["post_window_unwind_completed"] is not True:
        raise LedgerValidationError("post-window real unwind must complete before final_evaluation")
    if (
        _validate_count(data["post_window_unwind_session_count"], "post_window_unwind_session_count")
        != state.unwind_session_count
    ):
        raise LedgerValidationError("final_evaluation unwind session count differs from ledger state")
    cost = data["post_window_unwind_cost_cny"]
    if type(cost) not in {int, float} or not math.isfinite(float(cost)) or float(cost) < 0:
        raise LedgerValidationError("post_window_unwind_cost_cny must be a finite non-negative number")
    remaining_positions = _validate_count(data["remaining_position_count"], "remaining_position_count")
    if remaining_positions != state.position_count or remaining_positions != 0:
        raise LedgerValidationError("final_evaluation requires zero remaining positions")
    remaining_pending = _validate_count(data["remaining_pending_sell_count"], "remaining_pending_sell_count")
    if remaining_pending != 0 or remaining_pending != state.pending_sell_count:
        raise LedgerValidationError("final_evaluation requires zero remaining pending sells")
    remaining_settlements = _validate_count(
        data["remaining_pending_settlement_count"], "remaining_pending_settlement_count"
    )
    if remaining_settlements != 0 or remaining_settlements != state.pending_settlement_count:
        raise LedgerValidationError("final_evaluation requires zero remaining pending settlements")
    if data["unwind_completion_book_state_root_sha256"] != state.portfolio_sha256:
        raise LedgerValidationError("final_evaluation all-book completion root differs from ledger state")
    if data["unwind_completion_pending_sells_root_sha256"] != state.pending_sells_sha256:
        raise LedgerValidationError("final_evaluation pending completion root differs from ledger state")
    if data["unwind_completion_pending_settlements_root_sha256"] != state.pending_settlements_sha256:
        raise LedgerValidationError("final_evaluation pending-settlement completion root differs from ledger state")
    expected_unwind_evidence = _object_sha256(
        {
            "decision_unwind_evidence_root_sha256": state.unwind_evidence_root_sha256,
            "completion_book_state_root_sha256": state.portfolio_sha256,
            "completion_pending_sells_root_sha256": state.pending_sells_sha256,
            "remaining_position_count": state.position_count,
            "remaining_pending_sell_count": state.pending_sell_count,
            "remaining_pending_settlement_count": state.pending_settlement_count,
            "completion_pending_settlements_root_sha256": state.pending_settlements_sha256,
            "unwind_session_count": state.unwind_session_count,
            "post_window_unwind_cost_cny": float(cost),
        }
    )
    if data["unwind_evidence_sha256"] != expected_unwind_evidence:
        raise LedgerValidationError("final_evaluation unwind evidence is not derived from ledger execution state")
    if state.last_eod_record is None:
        raise LedgerValidationError("final_evaluation requires a completed post-window unwind EOD")
    earliest = state.last_eod_record.recorded_at_utc
    _validate_snapshot(
        data["snapshot_cutoff_utc"],
        "final_evaluation.snapshot_cutoff_utc",
        earliest=earliest,
        recorded_at=record.recorded_at_utc,
    )
    state.final_evaluation = record
    state.phase = "FINALIZED"


def _validate_genesis_record(
    record: LedgerRecord,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None,
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None = None,
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
            "registry_verifier_identity",
            "primary_registry_claim_commitment_sha256",
            "primary_registry_external_anchor",
            "primary_registry_external_anchor_receipt_sha256",
            "external_anchor",
            "external_anchor_receipt_sha256",
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
    registry_identity = _exact_mapping(
        payload["registry_verifier_identity"], _ANCHOR_IDENTITY_KEYS, "registry_verifier_identity"
    )
    if registry_identity["algorithm"] != ANCHOR_ALGORITHM:
        raise LedgerValidationError("Genesis registry anchor algorithm is unsupported")
    _require_sha256(registry_identity["public_key_sha256"], "registry_verifier_identity.public_key_sha256")
    if _object_sha256(registry_identity) != config["trial_registry_authority_sha256"]:
        raise ReceiptVerificationError("Genesis registry identity differs from the frozen registry authority")
    genesis_core = {
        "schema": "xs_chan_v2_1_genesis_core_v1",
        "config": config,
        "primary_chain_id": chain_id,
        "anchor_verifier_identity": identity,
    }
    registry_claim_core = {
        "schema": REGISTRY_CLAIM_SCHEMA,
        "registry_operation": "ATOMIC_REGISTER_PRIMARY_IF_ABSENT",
        "claim_result": "REGISTERED_AS_FIRST_PRIMARY",
        "uniqueness_scope_sha256": _object_sha256(
            {
                "trial_registry_namespace": config["trial_registry_namespace"],
                "trial_id": config["trial_id"],
                "chain_role": "PRIMARY",
            }
        ),
        "trial_registry_namespace": config["trial_registry_namespace"],
        "trial_registry_authority_sha256": config["trial_registry_authority_sha256"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "chain_role": "PRIMARY",
        "primary_chain_id": chain_id,
        "genesis_core_sha256": _sha256_bytes(_canonical_bytes(genesis_core)),
        "registry_verifier_identity": registry_identity,
    }
    registry_claim_commitment = _sha256_bytes(_canonical_bytes(registry_claim_core))
    if payload["primary_registry_claim_commitment_sha256"] != registry_claim_commitment:
        raise LedgerValidationError("Genesis primary registry claim commitment mismatch")
    registry_receipt = _validate_receipt_shape(payload["primary_registry_external_anchor"])
    if (
        registry_receipt["provider"] != registry_identity["provider"]
        or registry_receipt["key_id"] != registry_identity["key_id"]
    ):
        raise ReceiptVerificationError("primary registry receipt signer differs from its authority identity")
    if registry_receipt["subject_sha256"] != registry_claim_commitment:
        raise ReceiptVerificationError("primary registry receipt subject differs from its immutable claim")
    registry_receipt_sha256 = _sha256_bytes(_canonical_bytes(registry_receipt))
    if payload["primary_registry_external_anchor_receipt_sha256"] != registry_receipt_sha256:
        raise ReceiptVerificationError("primary registry receipt digest mismatch")
    core = {
        "schema": "xs_chan_v2_1_genesis_commitment_v2",
        "config": config,
        "primary_chain_id": chain_id,
        "anchor_verifier_identity": identity,
        "registry_verifier_identity": registry_identity,
        "primary_registry_claim_commitment_sha256": registry_claim_commitment,
        "primary_registry_external_anchor": registry_receipt,
        "primary_registry_external_anchor_receipt_sha256": registry_receipt_sha256,
    }
    commitment = _sha256_bytes(_canonical_bytes(core))
    if payload["genesis_commitment_sha256"] != commitment:
        raise LedgerValidationError("genesis commitment mismatch")
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    start_boundary = _local_boundary(start, "00:00:00", config["exchange_timezone"])
    if record.recorded_at_utc >= start_boundary:
        raise LedgerTimingError("genesis must be recorded before oos_start_date begins")
    receipt = _validate_receipt_shape(payload["external_anchor"])
    receipt_sha256 = _sha256_bytes(_canonical_bytes(receipt))
    if payload["external_anchor_receipt_sha256"] != receipt_sha256:
        raise ReceiptVerificationError("genesis external anchor receipt digest mismatch")
    if receipt["provider"] != identity["provider"] or receipt["key_id"] != identity["key_id"]:
        raise ReceiptVerificationError("genesis receipt signer differs from its anchor identity")
    if receipt["subject_sha256"] != commitment:
        raise ReceiptVerificationError("genesis receipt subject differs from its commitment")
    claimed_issued_at = _canonical_utc(receipt["issued_at_utc"], "receipt.issued_at_utc")
    registry_issued_at = _canonical_utc(registry_receipt["issued_at_utc"], "registry receipt.issued_at_utc")
    segment = config["genesis_calendar_segment"]
    published = _canonical_utc(segment["source_published_at_utc"], "genesis calendar published_at")
    retrieved = _canonical_utc(segment["source_retrieved_at_utc"], "genesis calendar retrieved_at")
    eligibility_cutoff = _canonical_utc(
        config["formal_start_eligibility"]["eligibility_cutoff_utc"],
        "formal-start eligibility cutoff",
    )
    if retrieved > record.recorded_at_utc:
        raise LedgerTimingError("genesis calendar retrieval exceeds the Genesis record timestamp")
    if record.recorded_at_utc < eligibility_cutoff:
        raise LedgerTimingError("Genesis predates the formal-start eligibility cutoff")
    if registry_issued_at < max(published, retrieved, eligibility_cutoff) or registry_issued_at >= start_boundary:
        raise ReceiptVerificationError("primary registry receipt is outside the Genesis eligibility interval")
    if (
        claimed_issued_at < max(published, retrieved, eligibility_cutoff, registry_issued_at)
        or claimed_issued_at > record.recorded_at_utc
        or claimed_issued_at >= start_boundary
    ):
        raise ReceiptVerificationError("genesis receipt is outside its registered append interval")
    receipts_verified = False
    if receipt_verifier is not None:
        if receipt_verifier.identity != identity:
            raise ReceiptVerificationError("provided verifier differs from the genesis trust root")
        verified = receipt_verifier.verify(receipt, expected_subject_sha256=commitment)
        _verify_receipt_timing(
            verified,
            recorded_at_utc=record.recorded_at_utc,
            earliest_utc=max(published, retrieved, eligibility_cutoff, registry_issued_at),
            latest_utc=start_boundary,
        )
        receipts_verified = True
    registry_receipt_verified = False
    if registry_receipt_verifier is not None:
        if registry_receipt_verifier.identity != registry_identity:
            raise ReceiptVerificationError("provided registry verifier differs from the Genesis registry trust root")
        registry_verified = registry_receipt_verifier.verify(
            registry_receipt,
            expected_subject_sha256=registry_claim_commitment,
        )
        _verify_receipt_timing(
            registry_verified,
            recorded_at_utc=record.recorded_at_utc,
            earliest_utc=max(published, retrieved, eligibility_cutoff),
            latest_utc=start_boundary,
        )
        registry_receipt_verified = True
    segment_commitment = calendar_segment_commitment_sha256(chain_id, segment)
    segment_entry = {
        "segment_index": 0,
        "segment": segment,
        "commitment_sha256": segment_commitment,
        "record_hash": record.record_hash,
        "record_sequence": record.sequence,
    }
    return _ReplayState(
        records=[record],
        genesis=record,
        config=config,
        anchor_identity=identity,
        receipts_verified=receipts_verified,
        registry_receipt_verified=registry_receipt_verified,
        segments=[segment_entry],
        sessions=[_normalise_date(item, "genesis calendar session") for item in segment["sessions"]],
        portfolio_sha256=config["initial_state_root_sha256"],
        pending_sells_sha256=canonical_initial_pending_sells_root_sha256(),
        pending_sell_count=0,
        data_chain_head_sha256=config["genesis_data_chain_head_sha256"],
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
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None = None,
) -> _ReplayState:
    records = _scan_records(root)
    state = _validate_genesis_record(
        records[0],
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    for record in records[1:]:
        if state.terminal is not None:
            raise LedgerValidationError("no record may follow an ABORTED or FAILED terminal record")
        if state.final_evaluation is not None:
            raise LedgerValidationError("no record may follow final_evaluation")
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
        elif record.record_type == "final_evaluation":
            _apply_final_evaluation(state, event_record)
        else:
            raise LedgerValidationError(f"unsupported record_type: {record.record_type!r}")
        state.records.append(record)
    return state


def _claim_path(registry_root: Path, config: Mapping[str, Any]) -> Path:
    # One claim per protocol hash, not per trial.  A failed trial therefore
    # cannot be replaced under the same preregistered protocol.
    return registry_root / "primary" / f"{config['protocol_sha256']}.json"


def _build_registry_claim(genesis: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    payload = genesis["payload"]
    document = {
        "schema": REGISTRY_DOCUMENT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "primary_chain_id": payload["primary_chain_id"],
        "primary_registry_claim_commitment_sha256": payload["primary_registry_claim_commitment_sha256"],
        "registry_verifier_identity": payload["registry_verifier_identity"],
        "primary_registry_external_anchor": payload["primary_registry_external_anchor"],
        "primary_registry_external_anchor_receipt_sha256": payload["primary_registry_external_anchor_receipt_sha256"],
    }
    return document, _canonical_bytes(document)


def _validate_registry_claim(
    registry_root: Path,
    root: Path,
    state: _ReplayState,
    *,
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None,
) -> None:
    del root  # The external primary identity is portable and cannot depend on a local path.
    if not isinstance(registry_receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("external registry authority verifier is required")
    path = _claim_path(registry_root, state.config)
    if not path.is_file():
        raise LedgerValidationError("primary-chain registry claim is missing")
    raw = path.read_bytes()
    try:
        claim = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerValidationError("primary-chain registry claim is invalid JSON") from exc
    expected, expected_raw = _build_registry_claim(state.genesis.data, state.config)
    if raw != expected_raw or claim != expected:
        raise LedgerValidationError("local primary-chain registry mirror differs from Genesis")
    identity = state.genesis.data["payload"]["registry_verifier_identity"]
    if registry_receipt_verifier.identity != identity:
        raise ReceiptVerificationError("registry verifier differs from the Genesis authority identity")
    verified = registry_receipt_verifier.verify(
        expected["primary_registry_external_anchor"],
        expected_subject_sha256=expected["primary_registry_claim_commitment_sha256"],
    )
    if verified.receipt_sha256 != expected["primary_registry_external_anchor_receipt_sha256"]:
        raise ReceiptVerificationError("registry receipt digest differs from the local immutable mirror")
    if verified.issued_at_utc > state.genesis.recorded_at_utc:
        raise ReceiptVerificationError("registry receipt was issued after Genesis")


def _verify_pre_candidate_readiness(
    config: Mapping[str, Any],
    *,
    readiness_report: Mapping[str, Any],
    readiness_external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> VerifiedReceipt:
    """Reject an unanchored or post-candidate readiness claim before Genesis."""

    report = _exact_mapping(
        readiness_report,
        _READINESS_COMMITMENT_KEYS,
        "pre-candidate readiness commitment",
    )
    if report["schema"] != READINESS_COMMITMENT_SCHEMA:
        raise LedgerValidationError("pre-candidate readiness commitment has the wrong schema")
    expected = {
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "data_contract_identity_sha256": config["data_contract_identity_sha256"],
        "dependency_sha256": config["dependency_sha256"],
        "engineering_gates": config["readiness_gates"],
    }
    for key, value in expected.items():
        if _canonical_bytes(report[key]) != _canonical_bytes(value):
            raise LedgerValidationError(f"pre-candidate readiness {key} differs from Genesis")
    report_sha = _require_sha256(report["report_sha256"], "readiness report_sha256")
    body = dict(report)
    body.pop("report_sha256")
    if report_sha != _object_sha256(body):
        raise LedgerValidationError("pre-candidate readiness self-digest differs")
    if report_sha != config["readiness_report_sha256"]:
        raise LedgerValidationError("pre-candidate readiness report differs from Genesis")
    eligibility = config["formal_start_eligibility"]
    if report_sha != eligibility["readiness_report_sha256"]:
        raise LedgerValidationError("formal-start evidence names a different readiness report")
    verified = receipt_verifier.verify(
        readiness_external_anchor,
        expected_subject_sha256=report_sha,
    )
    if verified.receipt_sha256 != eligibility["readiness_anchor_receipt_sha256"]:
        raise ReceiptVerificationError("formal-start evidence names a different readiness receipt")
    created = _canonical_utc(report["created_at_utc"], "readiness.created_at_utc")
    completed = _canonical_utc(
        eligibility["readiness_completed_at_utc"],
        "formal-start readiness_completed_at_utc",
    )
    cutoff = _canonical_utc(
        eligibility["eligibility_cutoff_utc"],
        "formal-start eligibility_cutoff_utc",
    )
    if not created <= verified.issued_at_utc == completed < cutoff:
        raise LedgerTimingError("pre-candidate readiness timestamps violate creation <= anchor == completion < cutoff")
    return verified


def init_ledger(
    root: str | Path,
    *,
    genesis_config: Mapping[str, Any],
    readiness_report: Mapping[str, Any],
    readiness_external_anchor: Mapping[str, Any],
    trial_registry_root: str | Path,
    registry_external_anchor: Mapping[str, Any],
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Create the one externally anchored primary chain for this protocol."""
    if not isinstance(receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("init_ledger requires a concrete RSA receipt verifier")
    if not isinstance(registry_receipt_verifier, RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReceiptVerificationError("init_ledger requires an external registry authority verifier")
    root_path = Path(root)
    registry_root = Path(trial_registry_root)
    config = _validate_genesis_config(genesis_config)
    _verify_pre_candidate_readiness(
        config,
        readiness_report=readiness_report,
        readiness_external_anchor=readiness_external_anchor,
        receipt_verifier=receipt_verifier,
    )
    registry_claim_commitment = primary_registry_claim_commitment_sha256(
        config,
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    verified_registry = registry_receipt_verifier.verify(
        registry_external_anchor,
        expected_subject_sha256=registry_claim_commitment,
    )
    commitment = genesis_commitment_sha256(
        config,
        receipt_verifier=receipt_verifier,
        registry_external_anchor=registry_external_anchor,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    recorded_at = _normalise_utc(_utc_now(), "clock")
    verified = receipt_verifier.verify(external_anchor, expected_subject_sha256=commitment)
    start = _normalise_date(config["oos_start_date"], "oos_start_date")
    start_boundary = _local_boundary(start, "00:00:00", config["exchange_timezone"])
    retrieved = _canonical_utc(
        config["genesis_calendar_segment"]["source_retrieved_at_utc"],
        "genesis calendar retrieved_at",
    )
    eligibility_cutoff = _canonical_utc(
        config["formal_start_eligibility"]["eligibility_cutoff_utc"],
        "formal-start eligibility cutoff",
    )
    _verify_receipt_timing(
        verified_registry,
        recorded_at_utc=recorded_at,
        earliest_utc=max(retrieved, eligibility_cutoff),
        latest_utc=start_boundary,
    )
    _verify_receipt_timing(
        verified,
        recorded_at_utc=recorded_at,
        earliest_utc=max(retrieved, eligibility_cutoff, verified_registry.issued_at_utc),
        latest_utc=start_boundary,
    )
    if recorded_at >= start_boundary:
        raise LedgerTimingError("genesis must be recorded before oos_start_date begins")
    payload = {
        "schema": GENESIS_PAYLOAD_SCHEMA,
        "config": config,
        "primary_chain_id": _primary_chain_id(config),
        "genesis_commitment_sha256": commitment,
        "anchor_verifier_identity": receipt_verifier.identity,
        "registry_verifier_identity": registry_receipt_verifier.identity,
        "primary_registry_claim_commitment_sha256": registry_claim_commitment,
        "primary_registry_external_anchor": dict(registry_external_anchor),
        "primary_registry_external_anchor_receipt_sha256": verified_registry.receipt_sha256,
        "external_anchor": dict(external_anchor),
        "external_anchor_receipt_sha256": verified.receipt_sha256,
    }
    record, raw = _build_record(0, "genesis", recorded_at, ZERO_HASH, payload)
    path = _record_path(root_path, 0, record["record_hash"])
    _, claim_raw = _build_registry_claim(record, config)
    claim_path = _claim_path(registry_root, config)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        _exclusive_lock(registry_root, ".registry.lock"),
        _exclusive_lock(root_path, ".ledger.lock"),
    ):
        json_files = list(root_path.glob("*.json"))
        if claim_path.exists() or json_files:
            if claim_path.is_file() and len(json_files) == 1:
                existing = _validate_chain(
                    root_path,
                    receipt_verifier=receipt_verifier,
                    registry_receipt_verifier=registry_receipt_verifier,
                )
                _validate_registry_claim(
                    registry_root,
                    root_path,
                    existing,
                    registry_receipt_verifier=registry_receipt_verifier,
                )
                if existing.genesis.data["payload"] == payload and claim_path.read_bytes() == claim_raw:
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
        if state.final_evaluation is not None:
            raise LedgerValidationError("chain has a final evaluation and cannot be extended")
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
    if segment_value["segment_index"] != len(state.segments):
        raise LedgerValidationError("calendar extension index is not the next contiguous index")
    if segment_value["previous_calendar_head_sha256"] != _calendar_state_sha256(state.segments):
        raise LedgerValidationError("calendar extension previous head differs from the current calendar head")
    previous_end = _normalise_date(state.segments[-1]["segment"]["coverage_end"], "coverage_end")
    if _normalise_date(segment_value["coverage_start"], "coverage_start") != previous_end + timedelta(days=1):
        raise LedgerValidationError("calendar coverage segments must be contiguous without overlap or gaps")
    event = {
        "segment_index": segment_value["segment_index"],
        "segment": segment_value,
        "commitment_sha256": calendar_segment_commitment_sha256(_primary_chain_id(state.config), segment_value),
    }
    return prepare_record_commitment(root_path, record_type="calendar_extension", event_payload=event)


def _event_payload(root: Path, week_id: str, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
    state = _validate_chain(root, receipt_verifier=None)
    if len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        if week_id != UNWIND_WEEK_ID:
            raise LedgerValidationError("post-window records must use the frozen unwind identity")
        origin_hash = state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["weekly_close_record_sha256"]
        if kind == "decision":
            session = _next_unwind_session(state)
            if session is None:
                raise LedgerValidationError("no covered post-window session is available")
            return {
                "week_id": UNWIND_WEEK_ID,
                "decision_kind": "POST_WINDOW_UNWIND",
                "decision_dt": state.completed_weeks[MIN_COMPLETE_WEEKS - 1]["close_dt"],
                "execution_dt": session.isoformat(),
                "data": dict(data),
            }
        if state.decision_record is None or state.phase not in {
            "UNWIND_DECIDED",
            "UNWIND_AWAIT_SESSION_OPEN",
            "UNWIND_OPENED",
        }:
            raise LedgerValidationError("post-window open/EOD requires the frozen unwind decision")
        origin_hash = state.decision_record.record_hash
        if kind == "session_open_execution":
            session = _next_unwind_session(state)
            if session is None:
                raise LedgerValidationError("no covered post-window session is available")
            return {
                "week_id": UNWIND_WEEK_ID,
                "session_dt": session.isoformat(),
                "decision_hash": origin_hash,
                "data": dict(data),
            }
        if kind == "session_eod_valuation":
            if state.unwind_session is None:
                raise LedgerValidationError("post-window EOD requires its same-session unwind open")
            return {
                "week_id": UNWIND_WEEK_ID,
                "session_dt": state.unwind_session.isoformat(),
                "data": dict(data),
            }
        raise LedgerValidationError("a new decision or cycle_close is forbidden after the 52nd cycle")
    week = state.active_week if state.active_week is not None else _expected_week(state)
    if week is None or not isinstance(week_id, str) or _WEEK_ID.fullmatch(week_id) is None or week.week_id != week_id:
        raise LedgerValidationError("week_id does not identify the currently expected week")
    if kind == "decision":
        return {
            "week_id": week.week_id,
            "decision_kind": "CONFIRMATORY_CYCLE",
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
        "final_evaluation",
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


def prepare_final_evaluation(root: str | Path, *, data: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare the one externally anchored final evaluation record."""
    state = _validate_chain(Path(root), receipt_verifier=None)
    if state.final_evaluation is not None or state.terminal is not None:
        raise LedgerValidationError("chain is already terminal")
    event = _exact_mapping(data, _FINAL_EVALUATION_KEYS, "final_evaluation payload")
    return prepare_record_commitment(root, record_type="final_evaluation", event_payload=event)


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


def append_final_evaluation(
    root: str | Path,
    *,
    data: Mapping[str, Any],
    external_anchor: Mapping[str, Any],
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier,
) -> Path:
    """Append the sole evaluation after the fixed window and real unwind complete."""
    payload = _exact_mapping(data, _FINAL_EVALUATION_KEYS, "final_evaluation payload")
    return _append_record(
        Path(root),
        record_type="final_evaluation",
        payload=payload,
        external_anchor=external_anchor,
        receipt_verifier=receipt_verifier,
        apply=lambda state, record, _: _apply_final_evaluation(state, record),
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
        "predecessor_protocol_sha256": state.config["predecessor_protocol_sha256"],
        "data_bundle_sha256": state.config["data_bundle_sha256"],
        "data_contract_identity_sha256": state.config["data_contract_identity_sha256"],
        "genesis_data_chain_head_sha256": state.config["genesis_data_chain_head_sha256"],
        "genesis_data_validation_report_sha256": state.config["genesis_data_validation_report_sha256"],
        "confirmation_data_chain_head_sha256": week_rows[-1]["data_chain_head_sha256"],
        "dependency_sha256": state.config["dependency_sha256"],
        "readiness_report_sha256": state.config["readiness_report_sha256"],
        "readiness_gates": state.config["readiness_gates"],
        "initial_state_root_sha256": state.config["initial_state_root_sha256"],
        "rng_identity": state.config["rng_identity"],
        "formal_start_eligibility": state.config["formal_start_eligibility"],
        "trial_id": state.config["trial_id"],
        "weeks": [{key: value for key, value in row.items() if key != "weekly_close_sequence"} for row in week_rows],
        "calendar_segment_commitment_sha256": [
            item["commitment_sha256"] for item in state.segments if item["record_sequence"] <= head_sequence
        ],
        "record_hashes_through_52nd_close": [record.record_hash for record in records],
        "confirmation_head_sha256": records[-1].record_hash,
    }
    return {**identity, "identity_sha256": _sha256_bytes(_canonical_bytes(identity))}


def verify_ledger(
    root: str | Path,
    *,
    receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None = None,
    trial_registry_root: str | Path | None = None,
    registry_receipt_verifier: RsaPkcs1v15Sha256ReceiptVerifier | None = None,
) -> dict[str, Any]:
    """Verify structure and receipts while remaining closed on semantic claims."""
    root_path = Path(root)
    state = _validate_chain(
        root_path,
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
    )
    registry_verified = False
    if trial_registry_root is not None and registry_receipt_verifier is not None:
        _validate_registry_claim(
            Path(trial_registry_root),
            root_path,
            state,
            registry_receipt_verifier=registry_receipt_verifier,
        )
        registry_verified = True
    receipts_verified = receipt_verifier is not None and state.receipts_verified
    expired = None if state.terminal is not None else _deadline_failure(state, _normalise_utc(_utc_now(), "clock"))
    if state.terminal is not None or expired is not None:
        lifecycle_status = "PRIMARY_CHAIN_TERMINATED_INVALID"
    elif len(state.completed_weeks) >= MIN_COMPLETE_WEEKS:
        lifecycle_status = "PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY"
    else:
        lifecycle_status = "PRIMARY_FORWARD_COLLECTION_ACTIVE"
    if lifecycle_status not in FORMAL_LIFECYCLE_STATES:
        raise LedgerValidationError("derived lifecycle status is outside the formal registry")
    structural_window_complete = len(state.completed_weeks) >= MIN_COMPLETE_WEEKS
    window = _confirmation_window(state)
    event_window = _event_window(state)
    genesis_payload = state.genesis.data["payload"]
    final_payload = dict(state.final_evaluation.data["payload"]) if state.final_evaluation is not None else None
    record_identities: list[dict[str, Any]] = []
    for record in state.records:
        if record.sequence == 0:
            commitment_sha256 = genesis_payload["genesis_commitment_sha256"]
            receipt_sha256 = genesis_payload["external_anchor_receipt_sha256"]
        else:
            wrapper = record.data["payload"]
            commitment_sha256 = wrapper["record_commitment_sha256"]
            receipt_sha256 = _sha256_bytes(_canonical_bytes(wrapper["external_anchor"]))
        record_identities.append(
            {
                "sequence": record.sequence,
                "record_type": record.record_type,
                "recorded_at_utc": record.data["recorded_at_utc"],
                "previous_record_hash": record.data["previous_record_hash"],
                "record_hash": record.record_hash,
                "record_commitment_sha256": commitment_sha256,
                "external_anchor_receipt_sha256": receipt_sha256,
            }
        )
    unwind_summary = {
        "session_count": state.unwind_session_count,
        "last_completed_session": state.unwind_next_session_after.isoformat()
        if state.unwind_session_count and state.unwind_next_session_after is not None
        else None,
        "book_state_root_sha256": state.portfolio_sha256,
        "pending_sells_root_sha256": state.pending_sells_sha256,
        "remaining_pending_sell_count": state.pending_sell_count,
        "pending_settlements_root_sha256": state.pending_settlements_sha256,
        "remaining_pending_settlement_count": state.pending_settlement_count,
        "execution_completed": state.phase in {"UNWIND_COMPLETE_PENDING_REPLAY", "FINALIZED"},
        "completed": bool(final_payload and final_payload["post_window_unwind_completed"]),
        "cost_cny": final_payload["post_window_unwind_cost_cny"] if final_payload else None,
        "evidence_sha256": final_payload["unwind_evidence_sha256"] if final_payload else None,
    }
    return {
        "valid": receipts_verified and registry_verified and lifecycle_status != "PRIMARY_CHAIN_TERMINATED_INVALID",
        "structurally_valid": True,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": state.config["protocol_id"],
        "protocol_sha256": state.config["protocol_sha256"],
        "predecessor_protocol_sha256": state.config["predecessor_protocol_sha256"],
        "data_bundle_sha256": state.config["data_bundle_sha256"],
        "data_contract_identity_sha256": state.config["data_contract_identity_sha256"],
        "genesis_data_chain_head_sha256": state.config["genesis_data_chain_head_sha256"],
        "current_data_chain_head_sha256": state.data_chain_head_sha256,
        "genesis_config": state.config,
        "genesis_config_sha256": _sha256_bytes(_canonical_bytes(state.config)),
        "dependency_sha256": state.config["dependency_sha256"],
        "dependency_closure_sha256": _object_sha256(state.config["dependency_sha256"]),
        "readiness_report_sha256": state.config["readiness_report_sha256"],
        "readiness_gates": state.config["readiness_gates"],
        "readiness_gate_evidence_root_sha256": _object_sha256(state.config["readiness_gates"]),
        "initial_state_sha256_by_book_scenario": state.config["initial_state_sha256_by_book_scenario"],
        "initial_state_root_sha256": state.config["initial_state_root_sha256"],
        "rng_identity": state.config["rng_identity"],
        "formal_start_eligibility": state.config["formal_start_eligibility"],
        "genesis_calendar_head_sha256": state.config["genesis_calendar_head_sha256"],
        "genesis_record_sha256": state.genesis.record_hash,
        "genesis_commitment_sha256": genesis_payload["genesis_commitment_sha256"],
        "genesis_external_anchor_receipt_sha256": genesis_payload["external_anchor_receipt_sha256"],
        "primary_registry_claim_commitment_sha256": genesis_payload["primary_registry_claim_commitment_sha256"],
        "primary_registry_external_anchor_receipt_sha256": genesis_payload[
            "primary_registry_external_anchor_receipt_sha256"
        ],
        "anchor_verifier_identity": genesis_payload["anchor_verifier_identity"],
        "registry_verifier_identity": genesis_payload["registry_verifier_identity"],
        "primary_chain_id": _primary_chain_id(state.config),
        "trial_id": state.config["trial_id"],
        "record_count": len(state.records),
        "record_identities": record_identities,
        "calendar_segment_count": len(state.segments),
        "calendar_segment_identities": [
            {
                "segment_index": item["segment_index"],
                "commitment_sha256": item["commitment_sha256"],
                "previous_calendar_head_sha256": item["segment"]["previous_calendar_head_sha256"],
                "new_calendar_head_sha256": item["segment"]["new_calendar_head_sha256"],
                "source_sha256": item["segment"]["source_sha256"],
                "record_hash": item["record_hash"],
            }
            for item in state.segments
        ],
        "calendar_state_sha256": _calendar_state_sha256(state.segments),
        "calendar_source_semantics_verified": False,
        "completed_weeks": len(state.completed_weeks),
        "min_complete_weeks": MIN_COMPLETE_WEEKS,
        "lifecycle_status": lifecycle_status,
        "declared_lifecycle_status": "EVALUATED" if state.final_evaluation is not None else lifecycle_status,
        "operational_status": lifecycle_status,
        "deadline_failure": expired,
        "external_receipts_verified": receipts_verified,
        "primary_registry_verified": registry_verified,
        "structural_window_complete": structural_window_complete,
        "semantic_replay_verified": False,
        "final_evaluation_recorded": state.final_evaluation is not None,
        "final_evaluation_verified": False,
        "confirmatory_oos": False,
        "next_required_event": event_window[0] if event_window is not None else None,
        "chain_head_sha256": state.records[-1].record_hash,
        "confirmation_head_sha256": window["confirmation_head_sha256"] if window is not None else None,
        "confirmation_window": window,
        "book_state_root_sha256": state.portfolio_sha256,
        "pending_sells_root_sha256": state.pending_sells_sha256,
        "unwind_summary": unwind_summary,
        "final_evaluation": final_payload,
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
    init.add_argument("--readiness-report", type=Path, required=True)
    init.add_argument("--readiness-external-anchor", type=Path, required=True)
    init.add_argument("--trial-registry", type=Path, required=True)
    init.add_argument("--registry-external-anchor", type=Path, required=True)
    init.add_argument("--registry-public-key", type=Path, required=True)
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
    final = append_types.add_parser("final-evaluation")
    final.add_argument("root", type=Path)
    final.add_argument("--payload", type=Path, required=True)
    final.add_argument("--anchor-public-key", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("--trial-registry", type=Path)
    verify.add_argument("--anchor-public-key", type=Path)
    verify.add_argument("--registry-public-key", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V2.1 ledger CLI without any backdating escape hatch."""
    args = build_parser().parse_args(argv)
    if args.command == "init":
        verifier = _load_verifier(args.anchor_public_key)
        registry_verifier = _load_verifier(args.registry_public_key)
        path = init_ledger(
            args.root,
            genesis_config=_load_json(args.config),
            readiness_report=_load_json(args.readiness_report),
            readiness_external_anchor=_load_json(args.readiness_external_anchor),
            trial_registry_root=args.trial_registry,
            registry_external_anchor=_load_json(args.registry_external_anchor),
            registry_receipt_verifier=registry_verifier,
            external_anchor=_load_json(args.external_anchor),
            receipt_verifier=verifier,
        )
        output = {"path": str(path), "record_hash": json.loads(path.read_bytes())["record_hash"]}
    elif args.command == "verify":
        verifier = _load_verifier(args.anchor_public_key) if args.anchor_public_key is not None else None
        registry_verifier = _load_verifier(args.registry_public_key) if args.registry_public_key is not None else None
        output = verify_ledger(
            args.root,
            receipt_verifier=verifier,
            trial_registry_root=args.trial_registry,
            registry_receipt_verifier=registry_verifier,
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
        elif args.append_type == "final-evaluation":
            value = _exact_mapping(payload, {"data", "external_anchor"}, "final-evaluation CLI payload")
            path = append_final_evaluation(
                args.root,
                data=value["data"],
                external_anchor=value["external_anchor"],
                receipt_verifier=verifier,
            )
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
