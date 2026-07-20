"""Fail-closed, content-addressed data contract for the XS/Chan V2.1 study.

This module is intentionally separate from the frozen V2 implementation.  A
formal validation call accepts exactly one content-addressed manifest; it never
accepts loose parquet paths or already-materialised ``DataFrame`` objects.
Every parquet/source object is read once into immutable bytes, checked against
its address, and only then parsed.  The returned :class:`VerifiedDataSnapshot`
continues to use those verified bytes, so validation and replay cannot observe
different file contents (TOCTOU).

The contract closes four provenance gaps that V2 deliberately left open:

* every L/D/P security intersecting the snapshot has an archived positive or
  zero response for raw prices, adjustments, name changes, corporate actions,
  and official trading/suspension status;
* a missing name-change row is usable only when an archived negative response
  exists (the validator itself never fills history with the current name);
* terminal actions are exactly equal to in-window delistings, and share
  consideration targets must exist and be alive on the effective date;
* every state-engine input row is bound to raw OHLC multiplied by the same-day
  adjustment factor and to an official TRADING/SUSPENDED row.

External timestamp/WORM attestation remains a separate V2.1 engineering gate.
This module proves internal content and lineage consistency, not vendor truth.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CONTRACT_ID = "xs_chan_data_snapshot_v2_1"
MANIFEST_VERSION = 3
PROTOCOL_ID = "xs_chan_pilot_v2_1_preregistered_20260720"
FROZEN_PROTOCOL_PATH = Path(__file__).resolve().with_name("xs_chan_protocol_v2_1.json")
SOURCE_RESPONSE_SCHEMA = "xs_chan_source_response_v2_1"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
SOURCE_RESPONSE_MEDIA_TYPE = "application/vnd.xs-chan.source-response+json;version=2.1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

LIST_STATUSES = frozenset({"L", "D", "P"})
OFFICIAL_STATUSES = frozenset({"TRADING", "SUSPENDED"})
CORPORATE_ACTION_TYPES = (
    "cash_dividend",
    "share_change",
    "rights_issue",
    "delist_cash",
    "delist_share",
    "delist_writeoff",
)
TERMINAL_ACTION_TYPES = frozenset({"delist_cash", "delist_share", "delist_writeoff"})
GLOBAL_ENDPOINTS = frozenset({"calendar", "security_master"})
SYMBOL_ENDPOINTS = frozenset(
    {
        "raw_daily",
        "adj_factor",
        "namechange",
        "corporate_actions",
        "corporate_action_evidence",
        "official_daily_status",
        "open_auction",
        "daily_basic",
        "stk_limit",
        "industry_membership",
    }
)
SOURCE_ENDPOINTS = GLOBAL_ENDPOINTS | SYMBOL_ENDPOINTS
SCOPED_RESPONSE_DATE_COLUMNS = {
    "raw_daily": "trade_date",
    "adj_factor": "trade_date",
    "open_auction": "trade_date",
    "daily_basic": "trade_date",
    "stk_limit": "trade_date",
    "official_daily_status": "trade_date",
    "corporate_actions": "effective_date",
    "corporate_action_evidence": "effective_date",
}

SOURCE_ASOF = "source_asof_utc"
INGESTED_AT = "ingested_at_utc"
COMPUTED_AT = "computed_at_utc"


class ContractError(ValueError):
    """Internal exception converted to a closed validation report at the API boundary."""


@dataclass(frozen=True)
class ArtifactSpec:
    """Exact logical schema and key for one snapshot parquet object."""

    columns: tuple[tuple[str, str], ...]
    key: tuple[str, ...]
    nullable: frozenset[str] = frozenset()
    allow_empty: bool = False
    source_endpoint: str | None = None

    @property
    def kinds(self) -> dict[str, str]:
        return dict(self.columns)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)


PROVENANCE_COLUMNS = ((SOURCE_ASOF, "utc_datetime"), (INGESTED_AT, "utc_datetime"))

ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "calendar": ArtifactSpec(
        columns=(
            ("trade_date", "date"),
            ("is_open", "bool"),
            ("prev_trade_date", "date"),
            ("next_trade_date", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("trade_date",),
        nullable=frozenset({"prev_trade_date", "next_trade_date"}),
        source_endpoint="calendar",
    ),
    "security_master": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("name", "string"),
            ("exchange", "string"),
            ("board", "string"),
            ("list_status", "string"),
            ("list_date", "date"),
            ("delist_date", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code",),
        nullable=frozenset({"delist_date"}),
        source_endpoint="security_master",
    ),
    "raw_daily": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("open", "float64"),
            ("high", "float64"),
            ("low", "float64"),
            ("close", "float64"),
            ("pre_close", "float64"),
            ("vol", "float64"),
            ("amount", "float64"),
            ("pct_chg", "float64"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        source_endpoint="raw_daily",
    ),
    "adj_factor": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("adj_factor", "float64"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        source_endpoint="adj_factor",
    ),
    "namechange": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("name", "string"),
            ("effective_from", "date"),
            ("effective_to", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "effective_from"),
        nullable=frozenset({"effective_to"}),
        allow_empty=True,
        source_endpoint="namechange",
    ),
    "corporate_actions": ArtifactSpec(
        columns=(
            ("action_id", "string"),
            ("ts_code", "string"),
            ("record_date", "date"),
            ("effective_date", "date"),
            ("payment_date", "date"),
            ("action_type", "string"),
            ("action_subtype", "string"),
            ("gross_cash_per_share", "float64"),
            ("taxable_dividend_per_pre_action_share", "float64"),
            ("new_share_registration_date", "date"),
            ("post_to_pre_ratio", "float64"),
            ("fractional_cash_price", "float64"),
            ("official_disposal_proceeds_per_entitled_share", "float64"),
            ("disposal_settlement_date", "date"),
            ("target_symbol", "string"),
            ("target_share_ratio", "float64"),
            ("terminal_cash_per_share", "float64"),
            ("terminal_reason", "string"),
            ("no_value_evidence_sha256", "string"),
            *PROVENANCE_COLUMNS,
        ),
        key=("action_id",),
        nullable=frozenset(
            {
                "record_date",
                "payment_date",
                "action_subtype",
                "gross_cash_per_share",
                "taxable_dividend_per_pre_action_share",
                "new_share_registration_date",
                "post_to_pre_ratio",
                "fractional_cash_price",
                "official_disposal_proceeds_per_entitled_share",
                "disposal_settlement_date",
                "target_symbol",
                "target_share_ratio",
                "terminal_cash_per_share",
                "terminal_reason",
                "no_value_evidence_sha256",
            }
        ),
        allow_empty=True,
        source_endpoint="corporate_actions",
    ),
    "official_daily_status": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("official_status", "string"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        allow_empty=True,
        source_endpoint="official_daily_status",
    ),
    "state_input_binding": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("official_status", "string"),
            ("raw_open", "float64"),
            ("raw_high", "float64"),
            ("raw_low", "float64"),
            ("raw_close", "float64"),
            ("adj_factor", "float64"),
            ("adjusted_open", "float64"),
            ("adjusted_high", "float64"),
            ("adjusted_low", "float64"),
            ("adjusted_close", "float64"),
            ("included_in_state_engine", "bool"),
            ("raw_row_sha256", "string"),
            ("adj_factor_row_sha256", "string"),
            ("status_row_sha256", "string"),
            (COMPUTED_AT, "utc_datetime"),
            ("input_row_sha256", "string"),
        ),
        key=("ts_code", "trade_date"),
        nullable=frozenset(
            {
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "adj_factor",
                "adjusted_open",
                "adjusted_high",
                "adjusted_low",
                "adjusted_close",
                "raw_row_sha256",
                "adj_factor_row_sha256",
            }
        ),
        allow_empty=True,
    ),
    "namechange_reconciliation": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("response_kind", "string"),
            ("response_sha256", "string"),
            ("source_authority", "string"),
            ("requested_start", "date"),
            ("requested_end", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code",),
        allow_empty=True,
    ),
    "raw_daily_reconciliation": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("response_kind", "string"),
            ("response_sha256", "string"),
            ("status_response_sha256", "string"),
            ("source_authority", "string"),
            ("requested_start", "date"),
            ("requested_end", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code",),
        allow_empty=True,
    ),
    "open_auction": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("auction_price", "float64"),
            ("auction_volume", "float64"),
            ("auction_amount", "float64"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        allow_empty=True,
        source_endpoint="open_auction",
    ),
    "open_auction_reconciliation": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("response_kind", "string"),
            ("response_sha256", "string"),
            ("source_authority", "string"),
            ("requested_start", "date"),
            ("requested_end", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code",),
        allow_empty=True,
    ),
    "daily_basic": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("free_share", "float64"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        source_endpoint="daily_basic",
    ),
    "stk_limit": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("trade_date", "date"),
            ("up_limit", "float64"),
            ("down_limit", "float64"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "trade_date"),
        source_endpoint="stk_limit",
    ),
    "industry_membership": ArtifactSpec(
        columns=(
            ("ts_code", "string"),
            ("industry_code", "string"),
            ("industry_name", "string"),
            ("classification_version", "string"),
            ("effective_from", "date"),
            ("effective_to", "date"),
            *PROVENANCE_COLUMNS,
        ),
        key=("ts_code", "classification_version", "effective_from"),
        nullable=frozenset({"effective_to"}),
        source_endpoint="industry_membership",
    ),
    "corporate_action_evidence": ArtifactSpec(
        columns=(
            ("action_id", "string"),
            ("ts_code", "string"),
            ("effective_date", "date"),
            ("action_type", "string"),
            ("authority", "string"),
            ("source_document_id", "string"),
            ("source_document_sha256", "string"),
            ("zero_recovery_text_sha256", "string"),
            *PROVENANCE_COLUMNS,
        ),
        key=("action_id",),
        nullable=frozenset({"zero_recovery_text_sha256"}),
        allow_empty=True,
        source_endpoint="corporate_action_evidence",
    ),
    "universe_reconciliation": ArtifactSpec(
        columns=(
            ("list_status", "string"),
            ("expected_count", "int64"),
            ("received_count", "int64"),
            ("expected_symbol_set_sha256", "string"),
            ("received_symbol_set_sha256", "string"),
            ("request_sha256", "string"),
            ("response_sha256", "string"),
            *PROVENANCE_COLUMNS,
        ),
        key=("list_status",),
    ),
    "chan_states": ArtifactSpec(
        columns=(("symbol", "string"), ("dt", "date"), ("regime", "int64")),
        key=("symbol", "dt"),
        allow_empty=True,
    ),
    "state_audit": ArtifactSpec(
        columns=(
            ("symbol", "string"),
            ("checkpoint_dt", "date"),
            ("prefix_rows", "int64"),
            ("expected_rows", "int64"),
            ("actual_rows", "int64"),
            ("expected_sha256", "string"),
            ("actual_sha256", "string"),
            ("symbol_mismatches", "int64"),
            ("dt_mismatches", "int64"),
            ("regime_mismatches", "int64"),
            ("passed", "bool"),
        ),
        key=("symbol", "checkpoint_dt"),
        allow_empty=True,
    ),
}

# These two large tables are protocol-mandated semantic evidence rather than
# members of data_contract.required_artifacts.  They are nevertheless exact,
# content-addressed manifest objects and are mandatory for the formal gate.
EVIDENCE_SPECS = {name: ARTIFACT_SPECS.pop(name) for name in ("official_daily_status", "state_input_binding")}
ALL_SPECS = {**ARTIFACT_SPECS, **EVIDENCE_SPECS}
REQUIRED_ARTIFACTS = tuple(ARTIFACT_SPECS)
REQUIRED_EVIDENCE_OBJECTS = tuple(EVIDENCE_SPECS)
OBJECT_AUTHORITIES = {
    **{
        name: ("official_source_archive" if spec.source_endpoint else "xs_chan_data_v2_1.reconciliation")
        for name, spec in ARTIFACT_SPECS.items()
    },
    "chan_states": "frozen_hybrid_state_engine",
    "state_audit": "independent_prefix_recompute",
    "official_daily_status": "official_source_archive",
    "state_input_binding": "xs_chan_data_v2_1.state_input_binding",
}
FORMAL_GATES = (
    "content_addressed_manifest",
    "artifact_schema",
    "artifact_semantics",
    "source_time_order",
    "source_archive_binding",
    "universe_response_coverage",
    "namechange_response_evidence",
    "terminal_actions_exact",
    "state_input_row_binding",
)

STATE_COMPONENT_HASH_COLUMNS = {
    "raw_daily": ALL_SPECS["raw_daily"].names,
    "adj_factor": ALL_SPECS["adj_factor"].names,
    "official_daily_status": ALL_SPECS["official_daily_status"].names,
}
STATE_INPUT_HASH_COLUMNS = tuple(name for name in ALL_SPECS["state_input_binding"].names if name != "input_row_sha256")


@dataclass(frozen=True)
class ValidationIssue:
    """One serialisable, machine-readable fail-closed finding."""

    code: str
    message: str
    gate: str
    artifact: str | None = None


@dataclass
class DataValidationReport:
    """Formal V2.1 data-gate result.  Defaults are intentionally closed."""

    valid: bool = False
    data_ready: bool = False
    manifest_sha256: str | None = None
    gates: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(FORMAL_GATES, False))
    evidence: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, gate: str, artifact: str | None = None) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, gate=gate, artifact=artifact))

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible report for replay/ledger integration."""

        return {
            "contract_id": CONTRACT_ID,
            "manifest_version": MANIFEST_VERSION,
            "valid": self.valid,
            "data_ready": self.data_ready,
            "manifest_sha256": self.manifest_sha256,
            "gates": {name: bool(self.gates.get(name, False)) for name in FORMAL_GATES},
            "evidence": copy.deepcopy(self.evidence),
            "issues": [asdict(issue) for issue in self.issues],
        }


class VerifiedDataSnapshot:
    """Immutable verified bytes exposed to the semantic replay verifier.

    The class deliberately exposes copies/re-parses instead of mutable internal
    frames.  It has no filesystem read method, so a caller cannot accidentally
    re-open a changed loose file after validation.
    """

    __slots__ = (
        "_artifact_blobs",
        "_evidence_blobs",
        "_manifest",
        "_responses",
        "created_at_utc",
        "manifest_sha256",
    )

    def __init__(
        self,
        *,
        manifest_sha256: str,
        created_at_utc: str,
        manifest: Mapping[str, Any],
        artifact_blobs: Mapping[str, bytes],
        evidence_blobs: Mapping[str, bytes],
        responses: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.manifest_sha256 = manifest_sha256
        self.created_at_utc = created_at_utc
        self._manifest = copy.deepcopy(dict(manifest))
        self._artifact_blobs = {name: bytes(blob) for name, blob in artifact_blobs.items()}
        self._evidence_blobs = {name: bytes(blob) for name, blob in evidence_blobs.items()}
        self._responses = {
            digest: {key: copy.deepcopy(value) for key, value in response.items() if not key.startswith("_")}
            for digest, response in responses.items()
        }

    @property
    def artifact_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifact_blobs))

    @property
    def source_response_sha256(self) -> tuple[str, ...]:
        return tuple(sorted(self._responses))

    @property
    def evidence_object_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._evidence_blobs))

    def manifest_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def read_artifact_bytes(self, name: str) -> bytes:
        if name not in self._artifact_blobs:
            raise KeyError(name)
        return bytes(self._artifact_blobs[name])

    def read_parquet(self, name: str) -> pd.DataFrame:
        if name not in self._artifact_blobs:
            raise KeyError(name)
        return pq.read_table(io.BytesIO(self._artifact_blobs[name])).to_pandas(date_as_object=False)

    def read_evidence_parquet(self, name: str) -> pd.DataFrame:
        if name not in self._evidence_blobs:
            raise KeyError(name)
        return pq.read_table(io.BytesIO(self._evidence_blobs[name])).to_pandas(date_as_object=False)

    def read_source_response(self, digest: str) -> dict[str, Any]:
        if digest not in self._responses:
            raise KeyError(digest)
        return copy.deepcopy(dict(self._responses[digest]))

    def artifact_sha256(self, name: str) -> str:
        return sha256_bytes(self.read_artifact_bytes(name))


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoding used for source objects (with one final LF)."""

    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def symbol_set_sha256(values: Sequence[object] | pd.Series | pd.Index) -> str:
    """Hash a sorted unique non-empty symbol set without representation ambiguity."""

    symbols = sorted({str(value).strip() for value in values if str(value).strip()})
    return sha256_bytes(canonical_json_bytes(symbols))


def object_provenance_sha256(name: str, object_sha256: str, protocol_sha256: str) -> str:
    """Bind one object's bytes, exact schema, authority, contract, and protocol."""

    if name not in ALL_SPECS or not _is_sha256(object_sha256) or not _is_sha256(protocol_sha256):
        raise ContractError("cannot derive provenance for an unknown or non-addressed object")
    spec = ALL_SPECS[name]
    payload = {
        "contract_id": CONTRACT_ID,
        "manifest_version": MANIFEST_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "name": name,
        "object_sha256": object_sha256,
        "authority": OBJECT_AUTHORITIES[name],
        "columns": list(spec.columns),
        "primary_key": list(spec.key),
        "nullable": sorted(spec.nullable),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def frozen_protocol_sha256() -> str:
    """Return the protocol's registered canonical-JSON identity digest."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"authoritative protocol contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        protocol = json.loads(
            _read_regular_once(FROZEN_PROTOCOL_PATH).decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ContractError(f"invalid JSON constant {token}")),
        )
        canonical = json.dumps(
            protocol,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"cannot canonicalize authoritative V2.1 protocol: {exc}") from exc
    return sha256_bytes(canonical)


def protocol_file_sha256() -> str:
    """Return raw protocol file bytes digest for diagnostics, never identity."""

    return sha256_bytes(_read_regular_once(FROZEN_PROTOCOL_PATH))


def _assert_authoritative_protocol_contract() -> None:
    try:
        protocol = json.loads(_read_regular_once(FROZEN_PROTOCOL_PATH).decode("utf-8"))
        contract = protocol["data_contract"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError(f"cannot read authoritative V2.1 data contract: {exc}") from exc
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("authoritative protocol_id differs from the V2.1 data validator")
    if contract.get("manifest_version") != MANIFEST_VERSION:
        raise ContractError("authoritative manifest_version differs from the V2.1 data validator")
    if set(contract.get("required_artifacts", [])) != set(REQUIRED_ARTIFACTS):
        raise ContractError("authoritative required_artifacts differ from the V2.1 data validator")


def _strict_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ContractError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ContractError(f"invalid JSON constant {token}")),
        )
    except (UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"cannot parse strict JSON {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ContractError(f"{label} root must be an object")
    if value != canonical_json_bytes(parsed):
        raise ContractError(f"{label} must use the canonical JSON encoding")
    return parsed


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _read_regular_once(path: Path) -> bytes:
    """Read a non-symlink regular file once and detect mutation during the read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot open immutable object {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"content object is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after:
            raise ContractError(f"content object changed while being read: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ContractError(f"short read for content object: {path}")
        return payload
    finally:
        os.close(descriptor)


def _reject_symlink_chain(path: Path) -> None:
    """Reject a symlink in any existing component of an input path."""

    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ContractError(f"symlinked snapshot path component is forbidden: {cursor}")


def _parse_utc(value: object, *, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value):
        raise ContractError(f"{label} must be an offset-aware UTC timestamp")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ContractError(f"{label} is not a valid timestamp")
    return parsed


def _parse_date(value: object, *, label: str) -> pd.Timestamp:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed) or getattr(parsed, "tzinfo", None) is not None or parsed != parsed.normalize():
        raise ContractError(f"{label} must be a timezone-naive calendar date")
    return parsed


def _normalise_scalar(value: object, kind: str, *, label: str, nullable: bool) -> Any:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        if nullable:
            return None
        raise ContractError(f"{label} may not be null")
    if kind == "string":
        if not isinstance(value, str):
            raise ContractError(f"{label} must be a string")
        return value
    if kind == "date":
        return _parse_date(value, label=label).strftime("%Y-%m-%d")
    if kind == "utc_datetime":
        return _parse_utc(str(value), label=label).isoformat().replace("+00:00", "Z")
    if kind == "float64":
        if isinstance(value, (bool, np.bool_)):
            raise ContractError(f"{label} must be float64")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(f"{label} must be float64") from exc
        if not np.isfinite(number):
            raise ContractError(f"{label} must be finite")
        return number.hex()
    if kind == "int64":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ContractError(f"{label} must be int64")
        return int(value)
    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ContractError(f"{label} must be boolean")
        return bool(value)
    raise ContractError(f"unknown logical kind {kind!r}")


def canonical_row_sha256(artifact: str, row: Mapping[str, Any]) -> str:
    """Hash one complete artifact row with date/UTC/float canonicalisation."""

    if artifact not in ALL_SPECS:
        raise ContractError(f"unknown artifact {artifact!r}")
    spec = ALL_SPECS[artifact]
    if set(row) != set(spec.names):
        raise ContractError(f"{artifact} row fields differ from the exact contract")
    normalised = {
        name: _normalise_scalar(row[name], kind, label=f"{artifact}.{name}", nullable=name in spec.nullable)
        for name, kind in spec.columns
    }
    return sha256_bytes(canonical_json_bytes({"artifact": artifact, "row": normalised}))


def state_input_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash a state-input row excluding its self-referential digest field."""

    spec = ALL_SPECS["state_input_binding"]
    if set(row) != set(STATE_INPUT_HASH_COLUMNS):
        raise ContractError("state-input hash fields differ from the exact contract")
    normalised = {
        name: _normalise_scalar(
            row[name], spec.kinds[name], label=f"state_input_binding.{name}", nullable=name in spec.nullable
        )
        for name in STATE_INPUT_HASH_COLUMNS
    }
    return sha256_bytes(canonical_json_bytes({"artifact": "state_input_binding", "row": normalised}))


def canonical_source_row(artifact: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON row form required inside a standardised source response."""

    spec = ALL_SPECS[artifact]
    if spec.source_endpoint is None:
        raise ContractError(f"{artifact} is derived and has no source-response row form")
    names = tuple(name for name in spec.names if name not in {SOURCE_ASOF, INGESTED_AT})
    if not set(names).issubset(row):
        raise ContractError(f"{artifact} source row lacks required fields")
    result: dict[str, Any] = {}
    for name in names:
        value = row[name]
        canonical = _normalise_scalar(
            value, spec.kinds[name], label=f"{artifact}.{name}", nullable=name in spec.nullable
        )
        if canonical is None:
            result[name] = None
        elif spec.kinds[name] == "float64":
            result[name] = float(value)
        else:
            result[name] = canonical
    return result


def _arrow_kind_matches(data_type: pa.DataType, kind: str) -> bool:
    if kind == "string":
        return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)
    if kind == "date":
        return pa.types.is_date(data_type) or (pa.types.is_timestamp(data_type) and data_type.tz is None)
    if kind == "utc_datetime":
        return pa.types.is_timestamp(data_type) and str(data_type.tz).upper() in {"UTC", "+00:00"}
    if kind == "float64":
        return pa.types.is_float64(data_type)
    if kind == "int64":
        return pa.types.is_int64(data_type)
    if kind == "bool":
        return pa.types.is_boolean(data_type)
    return False


def _parse_artifact(name: str, payload: bytes) -> pd.DataFrame:
    spec = ALL_SPECS[name]
    try:
        table = pq.read_table(io.BytesIO(payload))
    except Exception as exc:
        raise ContractError(f"cannot parse {name} parquet: {type(exc).__name__}: {exc}") from exc
    if tuple(table.column_names) != spec.names:
        raise ContractError(f"{name} columns/order differ from the exact contract")
    for arrow_field, (expected_name, kind) in zip(table.schema, spec.columns, strict=True):
        if arrow_field.name != expected_name or not _arrow_kind_matches(arrow_field.type, kind):
            raise ContractError(f"{name}.{expected_name} has dtype {arrow_field.type}, expected {kind}")
    try:
        frame = table.to_pandas(date_as_object=False)
    except Exception as exc:
        raise ContractError(f"cannot materialise {name}: {type(exc).__name__}: {exc}") from exc
    if frame.empty and not spec.allow_empty:
        raise ContractError(f"{name} may not be empty")
    for column in spec.names:
        if column not in spec.nullable and frame[column].isna().any():
            raise ContractError(f"{name}.{column} contains null values")
    if frame.duplicated(list(spec.key)).any():
        raise ContractError(f"{name} contains duplicate primary keys")
    for column, kind in spec.columns:
        if kind == "date":
            converted = pd.to_datetime(frame[column], errors="coerce")
            if (
                converted.isna().ne(frame[column].isna()).any()
                or (converted.dropna() != converted.dropna().dt.normalize()).any()
            ):
                raise ContractError(f"{name}.{column} contains an invalid logical date")
            frame[column] = converted
        elif kind == "utc_datetime":
            converted = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if converted.isna().ne(frame[column].isna()).any():
                raise ContractError(f"{name}.{column} contains an invalid UTC timestamp")
            frame[column] = converted
    if not frame.empty:
        ordered_keys = frame.sort_values(list(spec.key), kind="stable")[list(spec.key)].reset_index(drop=True)
        if not frame[list(spec.key)].reset_index(drop=True).equals(ordered_keys):
            raise ContractError(f"{name} rows must be sorted by the primary key")
    return frame


def _validate_manifest_shape(
    manifest: Mapping[str, Any],
) -> tuple[pd.Timestamp, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    expected = {
        "contract_id",
        "manifest_version",
        "protocol_id",
        "protocol_sha256",
        "created_at_utc",
        "artifacts",
        "evidence_objects",
        "source_objects",
    }
    if set(manifest) != expected:
        raise ContractError("manifest fields differ from the exact V2.1 contract")
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ContractError("manifest contract identity/version mismatch")
    _assert_authoritative_protocol_contract()
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_sha256") != frozen_protocol_sha256():
        raise ContractError("manifest does not bind the authoritative V2.1 protocol bytes")
    created = _parse_utc(manifest.get("created_at_utc"), label="manifest.created_at_utc")
    if created > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
        raise ContractError("manifest.created_at_utc is in the future")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ContractError("manifest artifact set differs from the exact V2.1 contract")
    for name, entry in artifacts.items():
        if not isinstance(entry, dict) or set(entry) != {
            "sha256",
            "size",
            "media_type",
            "authority",
            "provenance_sha256",
        }:
            raise ContractError(f"manifest artifact entry is malformed: {name}")
        if not _is_sha256(entry["sha256"]) or not isinstance(entry["size"], int) or entry["size"] <= 0:
            raise ContractError(f"manifest artifact address/size is invalid: {name}")
        if entry["media_type"] != PARQUET_MEDIA_TYPE:
            raise ContractError(f"manifest artifact media type is invalid: {name}")
        if entry["authority"] != OBJECT_AUTHORITIES[name] or entry["provenance_sha256"] != object_provenance_sha256(
            name, entry["sha256"], manifest["protocol_sha256"]
        ):
            raise ContractError(f"manifest artifact authority/provenance is invalid: {name}")
    evidence_objects = manifest.get("evidence_objects")
    if not isinstance(evidence_objects, dict) or set(evidence_objects) != set(REQUIRED_EVIDENCE_OBJECTS):
        raise ContractError("manifest evidence-object set differs from the exact V2.1 contract")
    for name, entry in evidence_objects.items():
        if not isinstance(entry, dict) or set(entry) != {
            "sha256",
            "size",
            "media_type",
            "authority",
            "provenance_sha256",
        }:
            raise ContractError(f"manifest evidence-object entry is malformed: {name}")
        if not _is_sha256(entry["sha256"]) or not isinstance(entry["size"], int) or entry["size"] <= 0:
            raise ContractError(f"manifest evidence-object address/size is invalid: {name}")
        if entry["media_type"] != PARQUET_MEDIA_TYPE:
            raise ContractError(f"manifest evidence-object media type is invalid: {name}")
        if entry["authority"] != OBJECT_AUTHORITIES[name] or entry["provenance_sha256"] != object_provenance_sha256(
            name, entry["sha256"], manifest["protocol_sha256"]
        ):
            raise ContractError(f"manifest evidence-object authority/provenance is invalid: {name}")
    source_objects = manifest.get("source_objects")
    if not isinstance(source_objects, list) or not source_objects:
        raise ContractError("manifest source_objects must be a non-empty list")
    digests: list[str] = []
    for entry in source_objects:
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size", "media_type"}:
            raise ContractError("source object entry is malformed")
        digest = entry["sha256"]
        if not _is_sha256(digest) or not isinstance(entry["size"], int) or entry["size"] <= 0:
            raise ContractError("source object address/size is invalid")
        if entry["media_type"] != SOURCE_RESPONSE_MEDIA_TYPE:
            raise ContractError("source object media type is invalid")
        digests.append(digest)
    if digests != sorted(digests) or len(digests) != len(set(digests)):
        raise ContractError("source objects must be digest-sorted and unique")
    return created, artifacts, evidence_objects, source_objects


def _load_content_addressed_objects(
    manifest_path: str | Path,
) -> tuple[str, dict[str, Any], pd.Timestamp, dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    path = Path(manifest_path).expanduser()
    _reject_symlink_chain(path)
    if path.name.count(".") != 1 or path.suffix != ".json" or not _is_sha256(path.stem):
        raise ContractError("formal entry requires manifests/<physical-sha256>.json")
    if path.parent.name != "manifests":
        raise ContractError("formal manifest must be a direct child of a manifests directory")
    root = path.parent.parent
    for directory in (root, path.parent, root / "objects", root / "objects" / "sha256"):
        if directory.is_symlink() or not directory.is_dir():
            raise ContractError(f"snapshot directory is missing or symlinked: {directory}")
    manifest_bytes = _read_regular_once(path)
    manifest_digest = sha256_bytes(manifest_bytes)
    if manifest_digest != path.stem:
        raise ContractError("manifest filename does not equal its physical SHA256")
    manifest = _strict_json_bytes(manifest_bytes, label="manifest")
    created, artifact_entries, evidence_entries, source_entries = _validate_manifest_shape(manifest)

    descriptors: dict[str, tuple[int, str]] = {}
    for entry in list(artifact_entries.values()) + list(evidence_entries.values()) + source_entries:
        previous = descriptors.setdefault(entry["sha256"], (entry["size"], entry["media_type"]))
        if previous != (entry["size"], entry["media_type"]):
            raise ContractError("one digest has conflicting object descriptors")
    blobs: dict[str, bytes] = {}
    for digest, (expected_size, _) in descriptors.items():
        object_path = root / "objects" / "sha256" / digest
        payload = _read_regular_once(object_path)
        if len(payload) != expected_size or sha256_bytes(payload) != digest:
            raise ContractError(f"content object evidence mismatch: {digest}")
        blobs[digest] = payload
    artifact_blobs = {name: blobs[entry["sha256"]] for name, entry in artifact_entries.items()}
    evidence_blobs = {name: blobs[entry["sha256"]] for name, entry in evidence_entries.items()}
    source_blobs = {entry["sha256"]: blobs[entry["sha256"]] for entry in source_entries}
    return manifest_digest, manifest, created, artifact_blobs, evidence_blobs, source_blobs


def _source_row_names(artifact: str) -> tuple[str, ...]:
    return tuple(name for name in ALL_SPECS[artifact].names if name not in {SOURCE_ASOF, INGESTED_AT})


def _parse_source_responses(source_blobs: Mapping[str, bytes], created: pd.Timestamp) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    seen_keys: set[tuple[str, str | None]] = set()
    expected_fields = {
        "schema_version",
        "provider",
        "endpoint",
        "scope",
        "source_asof_utc",
        "requested_at_utc",
        "responded_at_utc",
        "ingested_at_utc",
        "kind",
        "rows",
    }
    for digest, payload in source_blobs.items():
        response = _strict_json_bytes(payload, label=f"source object {digest}")
        if set(response) != expected_fields or response.get("schema_version") != SOURCE_RESPONSE_SCHEMA:
            raise ContractError(f"source object has an invalid envelope: {digest}")
        if not isinstance(response.get("provider"), str) or not response["provider"].strip():
            raise ContractError(f"source object provider is missing: {digest}")
        endpoint = response.get("endpoint")
        if endpoint not in SOURCE_ENDPOINTS:
            raise ContractError(f"source object endpoint is invalid: {digest}")
        scope = response.get("scope")
        if not isinstance(scope, dict) or set(scope) != {"ts_code", "start_date", "end_date"}:
            raise ContractError(f"source object scope is malformed: {digest}")
        symbol = scope.get("ts_code")
        if endpoint in GLOBAL_ENDPOINTS:
            if symbol is not None:
                raise ContractError(f"global endpoint must use null ts_code: {endpoint}")
        elif not isinstance(symbol, str) or not symbol.strip():
            raise ContractError(f"symbol endpoint must use a non-empty ts_code: {endpoint}")
        start = _parse_date(scope.get("start_date"), label=f"{endpoint}.scope.start_date")
        end = _parse_date(scope.get("end_date"), label=f"{endpoint}.scope.end_date")
        if start > end:
            raise ContractError(f"source response scope is reversed: {endpoint}/{symbol}")
        times = [
            _parse_utc(response.get("source_asof_utc"), label=f"{endpoint}.source_asof_utc"),
            _parse_utc(response.get("requested_at_utc"), label=f"{endpoint}.requested_at_utc"),
            _parse_utc(response.get("responded_at_utc"), label=f"{endpoint}.responded_at_utc"),
            _parse_utc(response.get("ingested_at_utc"), label=f"{endpoint}.ingested_at_utc"),
            created,
        ]
        if any(left > right for left, right in zip(times, times[1:], strict=False)):
            raise ContractError(f"source timestamp order is invalid: {endpoint}/{symbol}")
        rows = response.get("rows")
        kind = response.get("kind")
        if not isinstance(rows, list) or kind not in {"POSITIVE", "ZERO"}:
            raise ContractError(f"source response kind/rows are invalid: {endpoint}/{symbol}")
        if (kind == "POSITIVE") != bool(rows):
            raise ContractError(f"source response kind disagrees with rows: {endpoint}/{symbol}")
        expected_row_fields = set(_source_row_names(endpoint))
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected_row_fields:
                raise ContractError(f"source response row fields are invalid: {endpoint}/{symbol}")
            canonical_source_row(endpoint, row)
            if symbol is not None and row.get("ts_code") != symbol:
                raise ContractError(f"source response row escapes its symbol scope: {endpoint}/{symbol}")
            date_column = SCOPED_RESPONSE_DATE_COLUMNS.get(endpoint)
            if date_column is not None:
                row_date = _parse_date(row[date_column], label=f"{endpoint}.{date_column}")
                if row_date < start or row_date > end:
                    raise ContractError(f"source response row escapes its requested date scope: {endpoint}/{symbol}")
        key = (endpoint, symbol)
        if key in seen_keys:
            raise ContractError(f"duplicate source response for {endpoint}/{symbol}")
        seen_keys.add(key)
        response["_start"] = start
        response["_end"] = end
        response["_source_asof"] = times[0]
        response["_ingested_at"] = times[3]
        response["_digest"] = digest
        responses[digest] = response
    return responses


def _add_failure(report: DataValidationReport, code: str, message: str, gate: str, artifact: str | None = None) -> bool:
    report.add(code, message, gate, artifact)
    return False


def _validate_artifact_semantics(report: DataValidationReport, frames: Mapping[str, pd.DataFrame]) -> bool:
    gate = "artifact_semantics"
    good = True
    calendar = frames["calendar"]
    if not calendar["is_open"].all() or not calendar["trade_date"].is_monotonic_increasing:
        good &= _add_failure(report, "calendar_semantics", "calendar must be sorted open sessions", gate, "calendar")
    expected_previous = calendar["trade_date"].shift(1)
    expected_next = calendar["trade_date"].shift(-1)
    previous_matches = (
        calendar["prev_trade_date"].fillna(pd.Timestamp.min).eq(expected_previous.fillna(pd.Timestamp.min))
    )
    next_matches = calendar["next_trade_date"].fillna(pd.Timestamp.max).eq(expected_next.fillna(pd.Timestamp.max))
    if not previous_matches.all() or not next_matches.all():
        good &= _add_failure(
            report, "calendar_links", "calendar prev/next session links are inconsistent", gate, "calendar"
        )

    master = frames["security_master"]
    if not master["list_status"].isin(LIST_STATUSES).all():
        good &= _add_failure(report, "master_status", "security status is outside L/D/P", gate, "security_master")
    date_order = master["delist_date"].isna() | master["delist_date"].gt(master["list_date"])
    status_dates = master["list_status"].eq("D").eq(master["delist_date"].notna())
    if not date_order.all() or not status_dates.all():
        good &= _add_failure(
            report,
            "master_lifecycle",
            "D requires list_date < delist_date; L/P require null delist_date",
            gate,
            "security_master",
        )
    for column in ("ts_code", "name"):
        if master[column].str.strip().eq("").any():
            good &= _add_failure(report, "master_identifier", f"{column} may not be empty", gate, "security_master")
    if not master["board"].isin({"MAIN", "CHINEXT", "STAR", "BSE"}).all():
        good &= _add_failure(
            report, "master_board", "security board is outside the canonical enum", gate, "security_master"
        )

    raw = frames["raw_daily"]
    prices = raw[["open", "high", "low", "close"]]
    if (
        not np.isfinite(prices.to_numpy(dtype=np.float64)).all()
        or (prices <= 0).any(axis=None)
        or not (raw["high"] >= prices.max(axis=1)).all()
        or not (raw["low"] <= prices.min(axis=1)).all()
    ):
        good &= _add_failure(report, "raw_ohlc", "raw OHLC is non-positive/non-finite or malformed", gate, "raw_daily")
    if (
        not np.isfinite(raw[["pre_close", "vol", "amount", "pct_chg"]].to_numpy(dtype=np.float64)).all()
        or not raw["pre_close"].gt(0).all()
        or not raw[["vol", "amount"]].ge(0).all(axis=None)
    ):
        good &= _add_failure(
            report, "raw_market_fields", "pre-close/volume/amount/pct_chg values are invalid", gate, "raw_daily"
        )
    factors = frames["adj_factor"]["adj_factor"]
    if not np.isfinite(factors.to_numpy(dtype=np.float64)).all() or not factors.gt(0).all():
        good &= _add_failure(report, "adj_factor", "adjustment factors must be finite and positive", gate, "adj_factor")

    names = frames["namechange"]
    if not names.empty:
        if (
            names["name"].str.strip().eq("").any()
            or (names["effective_to"].notna() & names["effective_to"].lt(names["effective_from"])).any()
        ):
            good &= _add_failure(
                report, "namechange_interval", "name history contains an invalid interval", gate, "namechange"
            )
        for _, group in names.groupby("ts_code", sort=False):
            ordered = group.sort_values("effective_from")
            if len(ordered) > 1:
                ends = ordered["effective_to"].fillna(pd.Timestamp.max.normalize()).to_numpy()
                starts = ordered["effective_from"].to_numpy()
                if (starts[1:] <= ends[:-1]).any():
                    good &= _add_failure(
                        report, "namechange_overlap", "name-history intervals overlap", gate, "namechange"
                    )
                    break

    statuses = frames["official_daily_status"]
    if not statuses["official_status"].isin(OFFICIAL_STATUSES).all():
        good &= _add_failure(
            report, "official_status", "official status must be TRADING or SUSPENDED", gate, "official_daily_status"
        )

    actions = frames["corporate_actions"]
    if not actions["action_type"].isin(CORPORATE_ACTION_TYPES).all():
        good &= _add_failure(
            report, "corporate_action_type", "corporate action type is not registered", gate, "corporate_actions"
        )
    optional_action_fields = {
        "record_date",
        "payment_date",
        "action_subtype",
        "gross_cash_per_share",
        "taxable_dividend_per_pre_action_share",
        "new_share_registration_date",
        "post_to_pre_ratio",
        "fractional_cash_price",
        "official_disposal_proceeds_per_entitled_share",
        "disposal_settlement_date",
        "target_symbol",
        "target_share_ratio",
        "terminal_cash_per_share",
        "terminal_reason",
        "no_value_evidence_sha256",
    }
    required_by_type = {
        "cash_dividend": {"record_date", "payment_date", "gross_cash_per_share"},
        "share_change": {
            "record_date",
            "action_subtype",
            "taxable_dividend_per_pre_action_share",
            "new_share_registration_date",
            "post_to_pre_ratio",
            "fractional_cash_price",
        },
        "rights_issue": {
            "record_date",
            "payment_date",
            "official_disposal_proceeds_per_entitled_share",
        },
        "delist_cash": {"disposal_settlement_date", "terminal_cash_per_share", "terminal_reason"},
        "delist_share": {
            "disposal_settlement_date",
            "target_symbol",
            "target_share_ratio",
            "fractional_cash_price",
            "terminal_reason",
        },
        "delist_writeoff": {
            "disposal_settlement_date",
            "terminal_reason",
            "no_value_evidence_sha256",
        },
    }
    numeric = [
        "gross_cash_per_share",
        "taxable_dividend_per_pre_action_share",
        "post_to_pre_ratio",
        "fractional_cash_price",
        "official_disposal_proceeds_per_entitled_share",
        "target_share_ratio",
        "terminal_cash_per_share",
    ]
    if not actions.empty:
        if not actions[["action_id", "ts_code"]].apply(lambda series: series.str.strip().ne("")).all(axis=None):
            good &= _add_failure(
                report,
                "corporate_action_identity",
                "action_id and source security must be non-empty",
                gate,
                "corporate_actions",
            )
        finite_or_null = actions[numeric].apply(lambda series: series.isna() | np.isfinite(series))
        if not finite_or_null.all(axis=None):
            good &= _add_failure(
                report,
                "corporate_action_numeric",
                "corporate action numerics must be finite or null",
                gate,
                "corporate_actions",
            )
        nonnegative = numeric.copy()
        nonnegative.remove("post_to_pre_ratio")
        nonnegative.remove("target_share_ratio")
        if any((actions[column].dropna() < 0).any() for column in nonnegative) or any(
            (actions[column].dropna() <= 0).any() for column in ("post_to_pre_ratio", "target_share_ratio")
        ):
            good &= _add_failure(
                report,
                "corporate_action_value",
                "corporate action cash/ratios are outside their domains",
                gate,
                "corporate_actions",
            )
        for action_type, required in required_by_type.items():
            selected = actions["action_type"].eq(action_type)
            invalid_presence = any(
                not actions.loc[selected, column].notna().eq(column in required).all()
                for column in optional_action_fields
            )
            if invalid_presence:
                good &= _add_failure(
                    report,
                    "corporate_action_applicability",
                    f"field nullability is invalid for {action_type}",
                    gate,
                    "corporate_actions",
                )
        share_change = actions["action_type"].eq("share_change")
        registered_subtypes = {"stock_dividend", "capital_reserve_conversion", "split", "consolidation"}
        if not actions.loc[share_change, "action_subtype"].isin(registered_subtypes).all():
            good &= _add_failure(
                report,
                "corporate_action_subtype",
                "share-change action_subtype is not registered",
                gate,
                "corporate_actions",
            )
        stock_dividend = share_change & actions["action_subtype"].eq("stock_dividend")
        other_share_change = share_change & ~stock_dividend
        if (
            not actions.loc[stock_dividend, "taxable_dividend_per_pre_action_share"].gt(0).all()
            or not actions.loc[other_share_change, "taxable_dividend_per_pre_action_share"].eq(0).all()
        ):
            good &= _add_failure(
                report,
                "corporate_action_tax_basis",
                "stock dividend needs positive tax basis; other share changes need explicit zero",
                gate,
                "corporate_actions",
            )
        entitlement = actions["action_type"].isin({"cash_dividend", "share_change", "rights_issue"})
        payment = actions["action_type"].isin({"cash_dividend", "rights_issue"})
        registration = share_change
        terminal = actions["action_type"].isin(TERMINAL_ACTION_TYPES)
        if (
            not actions.loc[entitlement, "record_date"].le(actions.loc[entitlement, "effective_date"]).all()
            or not actions.loc[payment, "payment_date"].gt(actions.loc[payment, "effective_date"]).all()
            or not actions.loc[registration, "new_share_registration_date"]
            .eq(actions.loc[registration, "effective_date"])
            .all()
            or not actions.loc[terminal, "disposal_settlement_date"].ge(actions.loc[terminal, "effective_date"]).all()
        ):
            good &= _add_failure(
                report,
                "corporate_action_dates",
                "record/payment/registration/disposal dates violate the frozen event order",
                gate,
                "corporate_actions",
            )
        terminal_reasons = actions.loc[terminal, "terminal_reason"]
        target_rows = actions["action_type"].eq("delist_share")
        if (
            terminal_reasons.str.strip().eq("").any()
            or actions.loc[target_rows, "target_symbol"].str.strip().eq("").any()
            or not all(
                _is_sha256(value)
                for value in actions.loc[actions["action_type"].eq("delist_writeoff"), "no_value_evidence_sha256"]
            )
        ):
            good &= _add_failure(
                report,
                "corporate_action_terminal_fields",
                "terminal reason/target/zero-recovery evidence is incomplete",
                gate,
                "corporate_actions",
            )

    for artifact in ("namechange_reconciliation", "raw_daily_reconciliation", "open_auction_reconciliation"):
        reconciliation = frames[artifact]
        valid_rows = (
            reconciliation["response_kind"].isin({"POSITIVE", "ZERO"})
            & reconciliation["response_sha256"].map(_is_sha256)
            & reconciliation["source_authority"].str.strip().ne("")
            & reconciliation["requested_start"].le(reconciliation["requested_end"])
        )
        if artifact == "raw_daily_reconciliation":
            valid_rows &= reconciliation["status_response_sha256"].map(_is_sha256)
        if not valid_rows.all():
            good &= _add_failure(
                report,
                "reconciliation_semantics",
                "reconciliation response kind/hash/authority/range is invalid",
                gate,
                artifact,
            )

    auction = frames["open_auction"]
    if not auction.empty:
        auction_values = auction[["auction_price", "auction_volume", "auction_amount"]]
        auction_good = (
            np.isfinite(auction_values.to_numpy(dtype=np.float64)).all()
            and auction["auction_price"].gt(0).all()
            and auction[["auction_volume", "auction_amount"]].ge(0).all(axis=None)
        )
        if not auction_good:
            good &= _add_failure(
                report, "open_auction_values", "open-auction values are outside their domains", gate, "open_auction"
            )
    free_share = frames["daily_basic"]["free_share"]
    if not np.isfinite(free_share.to_numpy(dtype=np.float64)).all() or not free_share.gt(0).all():
        good &= _add_failure(
            report, "daily_basic_values", "free_share must be finite and positive", gate, "daily_basic"
        )
    limits = frames["stk_limit"]
    if (
        not np.isfinite(limits[["up_limit", "down_limit"]].to_numpy(dtype=np.float64)).all()
        or not limits["down_limit"].gt(0).all()
        or not limits["up_limit"].gt(limits["down_limit"]).all()
    ):
        good &= _add_failure(report, "stk_limit_values", "require 0 < down_limit < up_limit", gate, "stk_limit")
    industry = frames["industry_membership"]
    if (
        not industry["classification_version"].eq("SW2021").all()
        or (industry["effective_to"].notna() & industry["effective_to"].lt(industry["effective_from"])).any()
    ):
        good &= _add_failure(
            report,
            "industry_semantics",
            "industry history must use SW2021 and valid effective intervals",
            gate,
            "industry_membership",
        )

    action_evidence = frames["corporate_action_evidence"]
    evidence_good = (
        action_evidence["action_type"].isin(CORPORATE_ACTION_TYPES)
        & action_evidence["authority"].str.strip().ne("")
        & action_evidence["source_document_id"].str.strip().ne("")
        & action_evidence["source_document_sha256"].map(_is_sha256)
    )
    writeoff = action_evidence["action_type"].eq("delist_writeoff")
    evidence_good &= writeoff.eq(action_evidence["zero_recovery_text_sha256"].notna())
    if not all(_is_sha256(value) for value in action_evidence.loc[writeoff, "zero_recovery_text_sha256"]):
        evidence_good.loc[writeoff] = False
    if not evidence_good.all():
        good &= _add_failure(
            report,
            "corporate_action_evidence",
            "action evidence lacks exact authority/document/zero-recovery proof",
            gate,
            "corporate_action_evidence",
        )

    universe = frames["universe_reconciliation"]
    count_columns = ["expected_count", "received_count"]
    universe_good = (
        set(universe["list_status"]) == LIST_STATUSES
        and len(universe) == len(LIST_STATUSES)
        and np.isfinite(universe[count_columns].to_numpy(dtype=np.float64)).all()
        and universe[count_columns].ge(0).all(axis=None)
        and np.equal(universe[count_columns], np.floor(universe[count_columns])).all(axis=None)
        and universe["expected_count"].eq(universe["received_count"]).all()
    )
    for column in (
        "expected_symbol_set_sha256",
        "received_symbol_set_sha256",
        "request_sha256",
        "response_sha256",
    ):
        universe_good = bool(universe_good and universe[column].map(_is_sha256).all())
    universe_good = bool(
        universe_good and universe["expected_symbol_set_sha256"].eq(universe["received_symbol_set_sha256"]).all()
    )
    if not universe_good:
        good &= _add_failure(
            report,
            "universe_reconciliation",
            "L/D/P counts, symbol hashes, or request/response evidence are invalid",
            gate,
            "universe_reconciliation",
        )

    states = frames["chan_states"]
    if not states["regime"].between(0, 10).all() or not np.equal(states["regime"], np.floor(states["regime"])).all():
        good &= _add_failure(report, "state_regime", "state regime must be an integer in [0,10]", gate, "chan_states")
    audit = frames["state_audit"]
    audit_counts = [
        "prefix_rows",
        "expected_rows",
        "actual_rows",
        "symbol_mismatches",
        "dt_mismatches",
        "regime_mismatches",
    ]
    if not audit.empty:
        audit_good = (
            np.isfinite(audit[audit_counts].to_numpy(dtype=np.float64)).all()
            and audit[audit_counts].ge(0).all(axis=None)
            and np.equal(audit[audit_counts], np.floor(audit[audit_counts])).all(axis=None)
            and audit["passed"].all()
            and audit["expected_rows"].eq(audit["actual_rows"]).all()
            and audit["expected_sha256"].eq(audit["actual_sha256"]).all()
            and audit[["symbol_mismatches", "dt_mismatches", "regime_mismatches"]].eq(0).all(axis=None)
            and audit[["expected_sha256", "actual_sha256"]].map(_is_sha256).all(axis=None)
        )
        if not audit_good:
            good &= _add_failure(
                report, "state_audit", "state audit contains invalid counts/hashes/mismatches", gate, "state_audit"
            )
    return bool(good)


def _validate_source_time_order(
    report: DataValidationReport, frames: Mapping[str, pd.DataFrame], created: pd.Timestamp
) -> bool:
    gate = "source_time_order"
    good = True
    for name, spec in ALL_SPECS.items():
        if SOURCE_ASOF not in spec.names:
            continue
        frame = frames[name]
        ordered = frame[SOURCE_ASOF].le(frame[INGESTED_AT]) & frame[INGESTED_AT].le(created)
        if not ordered.all():
            good &= _add_failure(
                report,
                "source_time_order",
                "require source_asof <= ingested_at <= manifest.created_at",
                gate,
                name,
            )
    computed = frames["state_input_binding"][COMPUTED_AT]
    if not computed.le(created).all():
        good &= _add_failure(
            report, "state_compute_time", "state inputs were computed after the manifest", gate, "state_input_binding"
        )
    return bool(good)


def _response_by_key(responses: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str | None], Mapping[str, Any]]:
    return {(value["endpoint"], value["scope"]["ts_code"]): value for value in responses.values()}


def _validate_source_archive_binding(
    report: DataValidationReport,
    frames: Mapping[str, pd.DataFrame],
    responses: Mapping[str, Mapping[str, Any]],
) -> bool:
    gate = "source_archive_binding"
    good = True
    for artifact, spec in ALL_SPECS.items():
        if spec.source_endpoint is None:
            continue
        expected = Counter(canonical_row_sha256(artifact, row) for row in frames[artifact].to_dict(orient="records"))
        observed: Counter[str] = Counter()
        for response in responses.values():
            if response["endpoint"] != spec.source_endpoint:
                continue
            for source_row in response["rows"]:
                full_row = dict(source_row)
                full_row[SOURCE_ASOF] = response["source_asof_utc"]
                full_row[INGESTED_AT] = response["ingested_at_utc"]
                observed[canonical_row_sha256(artifact, full_row)] += 1
        if observed != expected:
            good &= _add_failure(
                report,
                "source_rows_mismatch",
                "archived response rows do not exactly equal the physical parquet rows",
                gate,
                artifact,
            )
    return bool(good)


def _expected_lifecycle_keys(
    calendar: pd.DataFrame, master: pd.DataFrame
) -> tuple[pd.MultiIndex, set[str], pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(calendar["trade_date"])
    start, end = dates.min(), dates.max()
    tuples: list[tuple[str, pd.Timestamp]] = []
    relevant: set[str] = set()
    for row in master.itertuples(index=False):
        intersects = row.list_date <= end and (pd.isna(row.delist_date) or row.delist_date >= start)
        if not intersects:
            continue
        relevant.add(row.ts_code)
        mask = dates >= row.list_date
        if not pd.isna(row.delist_date):
            mask &= dates < row.delist_date
        tuples.extend((row.ts_code, value) for value in dates[mask])
    index = pd.MultiIndex.from_tuples(tuples, names=["ts_code", "trade_date"])
    return index, relevant, start, end


def _frame_key_index(frame: pd.DataFrame, spec_name: str) -> pd.MultiIndex:
    spec = ALL_SPECS[spec_name]
    return pd.MultiIndex.from_frame(frame[list(spec.key)])


def _same_index_members(left: pd.MultiIndex, right: pd.MultiIndex) -> bool:
    """Compare unique key sets without relying on pandas level dtype metadata."""

    return len(left) == len(right) and not len(left.difference(right)) and not len(right.difference(left))


def _validate_universe_coverage(
    report: DataValidationReport,
    frames: Mapping[str, pd.DataFrame],
    responses: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, set[str]]:
    gate = "universe_response_coverage"
    expected_status_keys, relevant, start, end = _expected_lifecycle_keys(frames["calendar"], frames["security_master"])
    response_map = _response_by_key(responses)
    expected_response_keys = {(endpoint, None) for endpoint in GLOBAL_ENDPOINTS}
    expected_response_keys.update((endpoint, symbol) for endpoint in SYMBOL_ENDPOINTS for symbol in relevant)
    good = True
    if set(response_map) != expected_response_keys:
        missing = sorted(expected_response_keys - set(response_map), key=str)
        extra = sorted(set(response_map) - expected_response_keys, key=str)
        good &= _add_failure(
            report,
            "response_coverage",
            f"source response set is not exact; missing={missing[:5]}, extra={extra[:5]}",
            gate,
        )
    for response in responses.values():
        if response["_start"] != start or response["_end"] != end:
            good &= _add_failure(
                report,
                "response_date_scope",
                "every source request must bind the complete calendar range",
                gate,
            )
            break

    status = frames["official_daily_status"]
    status_keys = _frame_key_index(status, "official_daily_status")
    if not _same_index_members(status_keys, expected_status_keys):
        missing_count = len(expected_status_keys.difference(status_keys))
        extra_count = len(status_keys.difference(expected_status_keys))
        good &= _add_failure(
            report,
            "official_status_coverage",
            f"official status lifecycle grid mismatch; missing={missing_count}, extra={extra_count}",
            gate,
            "official_daily_status",
        )
    trading_keys = _frame_key_index(status.loc[status["official_status"].eq("TRADING")], "official_daily_status")
    raw_keys = _frame_key_index(frames["raw_daily"], "raw_daily")
    adj_keys = _frame_key_index(frames["adj_factor"], "adj_factor")
    if not _same_index_members(raw_keys, trading_keys) or not _same_index_members(adj_keys, trading_keys):
        good &= _add_failure(
            report,
            "raw_status_coverage",
            "raw and adjustment keys must exactly equal official TRADING keys",
            gate,
            "raw_daily",
        )

    for artifact in ("daily_basic", "stk_limit", "chan_states"):
        artifact_keys = _frame_key_index(frames[artifact], artifact)
        if not _same_index_members(raw_keys, artifact_keys):
            good &= _add_failure(
                report,
                "raw_dependent_coverage",
                f"{artifact} keys must exactly equal raw_daily keys",
                gate,
                artifact,
            )
    auction_keys = _frame_key_index(frames["open_auction"], "open_auction")
    if len(auction_keys.difference(raw_keys)):
        good &= _add_failure(
            report,
            "open_auction_coverage",
            "open-auction keys must be a subset of official TRADING/raw keys",
            gate,
            "open_auction",
        )

    industry_groups = dict(iter(frames["industry_membership"].groupby("ts_code", sort=False)))
    uncovered_industry = 0
    for row in frames["raw_daily"][["ts_code", "trade_date"]].itertuples(index=False):
        memberships = industry_groups.get(row.ts_code)
        if memberships is None:
            uncovered_industry += 1
            continue
        covers = memberships["effective_from"].le(row.trade_date) & (
            memberships["effective_to"].isna() | memberships["effective_to"].ge(row.trade_date)
        )
        uncovered_industry += int(covers.sum() != 1)
    if uncovered_industry:
        good &= _add_failure(
            report,
            "industry_coverage",
            f"{uncovered_industry} raw rows lack exactly one PIT industry",
            gate,
            "industry_membership",
        )

    def reconciliation_matches(artifact: str, endpoint: str, *, status_endpoint: str | None = None) -> bool:
        reconciliation = frames[artifact].set_index("ts_code", drop=False)
        if set(reconciliation.index) != relevant or len(reconciliation) != len(relevant):
            return False
        for symbol in relevant:
            response = response_map.get((endpoint, symbol))
            if response is None:
                return False
            row = reconciliation.loc[symbol]
            matches = (
                row["response_kind"] == response["kind"]
                and row["response_sha256"] == response["_digest"]
                and row["source_authority"] == response["provider"]
                and row["requested_start"] == start
                and row["requested_end"] == end
                and row[SOURCE_ASOF] == response["_source_asof"]
                and row[INGESTED_AT] == response["_ingested_at"]
            )
            if status_endpoint is not None:
                status_response = response_map.get((status_endpoint, symbol))
                matches = bool(
                    matches
                    and status_response is not None
                    and row["status_response_sha256"] == status_response["_digest"]
                )
            if not matches:
                return False
        return True

    if not reconciliation_matches("raw_daily_reconciliation", "raw_daily", status_endpoint="official_daily_status"):
        good &= _add_failure(
            report,
            "raw_reconciliation_binding",
            "raw reconciliation does not exactly bind archived raw/status responses",
            gate,
            "raw_daily_reconciliation",
        )
    if not reconciliation_matches("open_auction_reconciliation", "open_auction"):
        good &= _add_failure(
            report,
            "open_auction_reconciliation_binding",
            "open-auction reconciliation does not exactly bind archived responses",
            gate,
            "open_auction_reconciliation",
        )

    universe = frames["universe_reconciliation"].set_index("list_status", drop=False)
    for list_status in LIST_STATUSES:
        symbols = frames["security_master"].loc[frames["security_master"]["list_status"].eq(list_status), "ts_code"]
        expected_count = len(set(symbols))
        expected_digest = symbol_set_sha256(symbols)
        if list_status not in universe.index:
            good &= _add_failure(report, "universe_physical_binding", f"missing {list_status} universe partition", gate)
            continue
        row = universe.loc[list_status]
        if (
            row["expected_count"] != expected_count
            or row["received_count"] != expected_count
            or row["expected_symbol_set_sha256"] != expected_digest
            or row["received_symbol_set_sha256"] != expected_digest
        ):
            good &= _add_failure(
                report,
                "universe_physical_binding",
                f"{list_status} universe partition differs from security_master",
                gate,
                "universe_reconciliation",
            )

    audit = frames["state_audit"]
    state_keys = _frame_key_index(frames["chan_states"], "chan_states")
    if not audit.empty:
        checkpoint_keys = pd.MultiIndex.from_frame(
            audit[["symbol", "checkpoint_dt"]].rename(columns={"checkpoint_dt": "dt"})
        )
        if len(checkpoint_keys.difference(state_keys)):
            good &= _add_failure(
                report,
                "state_audit_coverage",
                "state-audit checkpoints must exist in chan_states",
                gate,
                "state_audit",
            )

    raw_symbols = set(frames["raw_daily"]["ts_code"])
    status_groups = dict(iter(status.groupby("ts_code", sort=False)))
    for symbol in relevant:
        response = response_map.get(("raw_daily", symbol))
        if response is None:
            continue
        if response["kind"] == "ZERO":
            statuses = status_groups.get(symbol)
            if symbol in raw_symbols or (
                statuses is not None and not statuses["official_status"].eq("SUSPENDED").all()
            ):
                good &= _add_failure(
                    report,
                    "raw_zero_unexplained",
                    f"zero raw response is not fully explained by official suspension: {symbol}",
                    gate,
                    "raw_daily",
                )
    return bool(good), relevant


def _validate_namechange_evidence(
    report: DataValidationReport,
    frames: Mapping[str, pd.DataFrame],
    responses: Mapping[str, Mapping[str, Any]],
    relevant: set[str],
) -> bool:
    gate = "namechange_response_evidence"
    response_map = _response_by_key(responses)
    physical_symbols = set(frames["namechange"]["ts_code"])
    master_symbols = set(frames["security_master"]["ts_code"])
    reconciliation = frames["namechange_reconciliation"].set_index("ts_code", drop=False)
    start = frames["calendar"]["trade_date"].min()
    end = frames["calendar"]["trade_date"].max()
    good = True
    if not physical_symbols.issubset(master_symbols) or not physical_symbols.issubset(relevant):
        good &= _add_failure(
            report,
            "namechange_symbol",
            "name history references an unknown/out-of-window security",
            gate,
            "namechange",
        )
    for symbol in relevant:
        response = response_map.get(("namechange", symbol))
        if response is None:
            good &= _add_failure(
                report,
                "namechange_missing_response",
                f"name-history response evidence is missing: {symbol}",
                gate,
                "namechange",
            )
            continue
        if symbol not in reconciliation.index:
            good &= _add_failure(
                report,
                "namechange_reconciliation_missing",
                f"name-history reconciliation is missing: {symbol}",
                gate,
                "namechange_reconciliation",
            )
            continue
        row = reconciliation.loc[symbol]
        if not (
            row["response_kind"] == response["kind"]
            and row["response_sha256"] == response["_digest"]
            and row["source_authority"] == response["provider"]
            and row["requested_start"] == start
            and row["requested_end"] == end
            and row[SOURCE_ASOF] == response["_source_asof"]
            and row[INGESTED_AT] == response["_ingested_at"]
        ):
            good &= _add_failure(
                report,
                "namechange_reconciliation_binding",
                f"name-history reconciliation does not bind the archived response: {symbol}",
                gate,
                "namechange_reconciliation",
            )
        has_rows = symbol in physical_symbols
        if has_rows != (response["kind"] == "POSITIVE"):
            good &= _add_failure(
                report,
                "namechange_response_kind",
                f"name-history positive/negative evidence disagrees with rows: {symbol}",
                gate,
                "namechange",
            )
    if set(reconciliation.index) != relevant or len(reconciliation) != len(relevant):
        good &= _add_failure(
            report,
            "namechange_reconciliation_set",
            "name-history reconciliation must contain every and only intersecting L/D/P security",
            gate,
            "namechange_reconciliation",
        )
    return bool(good)


def _validate_terminal_actions(report: DataValidationReport, frames: Mapping[str, pd.DataFrame]) -> bool:
    gate = "terminal_actions_exact"
    master = frames["security_master"]
    actions = frames["corporate_actions"]
    start = frames["calendar"]["trade_date"].min()
    end = frames["calendar"]["trade_date"].max()
    required = master.loc[
        master["list_status"].eq("D") & master["delist_date"].between(start, end, inclusive="both"),
        ["ts_code", "delist_date"],
    ].rename(columns={"delist_date": "effective_date"})
    actual = actions.loc[actions["action_type"].isin(TERMINAL_ACTION_TYPES), ["ts_code", "effective_date"]]
    required_keys = Counter(map(tuple, required.itertuples(index=False, name=None)))
    actual_keys = Counter(map(tuple, actual.itertuples(index=False, name=None)))
    good = True
    if actual_keys != required_keys or any(count != 1 for count in actual_keys.values()):
        good &= _add_failure(
            report,
            "terminal_action_equality",
            "terminal actions must be exactly one-for-one with in-window D/delist_date rows",
            gate,
            "corporate_actions",
        )
    evidence = frames["corporate_action_evidence"]
    action_keys = Counter(
        map(
            tuple,
            actions[["action_id", "ts_code", "effective_date", "action_type"]].itertuples(index=False, name=None),
        )
    )
    evidence_keys = Counter(
        map(
            tuple,
            evidence[["action_id", "ts_code", "effective_date", "action_type"]].itertuples(index=False, name=None),
        )
    )
    if evidence_keys != action_keys or any(count != 1 for count in evidence_keys.values()):
        good &= _add_failure(
            report,
            "corporate_action_evidence_equality",
            "every and only corporate action must have exactly one authority/document evidence row",
            gate,
            "corporate_action_evidence",
        )
    elif not actions.empty:
        evidence_by_id = evidence.set_index("action_id", drop=False)
        writeoffs = actions.loc[actions["action_type"].eq("delist_writeoff")]
        if any(
            evidence_by_id.loc[row.action_id, "zero_recovery_text_sha256"] != row.no_value_evidence_sha256
            for row in writeoffs.itertuples(index=False)
        ):
            good &= _add_failure(
                report,
                "writeoff_evidence_binding",
                "writeoff zero-value digest differs from its authority evidence",
                gate,
                "corporate_action_evidence",
            )
    master_by_symbol = master.set_index("ts_code", drop=False)
    unknown_sources = actions.loc[~actions["ts_code"].isin(master_by_symbol.index)]
    if not unknown_sources.empty:
        good &= _add_failure(
            report, "action_unknown_security", "corporate action source is unknown", gate, "corporate_actions"
        )
    share_actions = actions.loc[actions["action_type"].eq("delist_share")]
    for row in share_actions.itertuples(index=False):
        target = str(row.target_symbol).strip()
        if target == row.ts_code:
            good &= _add_failure(
                report,
                "delist_share_target_lifecycle",
                "delist-share target cannot be the delisting source security",
                gate,
                "corporate_actions",
            )
            continue
        if target not in master_by_symbol.index:
            good &= _add_failure(
                report,
                "delist_share_target",
                f"delist-share target is unknown: {target}",
                gate,
                "corporate_actions",
            )
            continue
        target_row = master_by_symbol.loc[target]
        alive = (
            target_row["list_status"] in {"L", "D"}
            and target_row["list_date"] <= row.effective_date
            and (pd.isna(target_row["delist_date"]) or row.effective_date < target_row["delist_date"])
        )
        if not alive:
            good &= _add_failure(
                report,
                "delist_share_target_lifecycle",
                f"delist-share target is not alive on effective date: {target}",
                gate,
                "corporate_actions",
            )
    return bool(good)


def _row_hashes(frame: pd.DataFrame, artifact: str) -> pd.Series:
    return pd.Series(
        (canonical_row_sha256(artifact, row) for row in frame.to_dict(orient="records")),
        index=pd.MultiIndex.from_frame(frame[list(ALL_SPECS[artifact].key)]),
        dtype="string",
    )


def _validate_state_input_binding(
    report: DataValidationReport, frames: Mapping[str, pd.DataFrame], created: pd.Timestamp
) -> bool:
    gate = "state_input_row_binding"
    state = frames["state_input_binding"].set_index(["ts_code", "trade_date"], drop=False)
    status = frames["official_daily_status"].set_index(["ts_code", "trade_date"], drop=False)
    raw = frames["raw_daily"].set_index(["ts_code", "trade_date"], drop=False)
    adj = frames["adj_factor"].set_index(["ts_code", "trade_date"], drop=False)
    good = True
    if not state.index.equals(status.index):
        return _add_failure(
            report,
            "state_status_keys",
            "state input ledger keys must exactly equal the official status grid",
            gate,
            "state_input_binding",
        )
    if not state["official_status"].eq(status["official_status"]).all():
        good &= _add_failure(
            report,
            "state_status_value",
            "state input official status differs from source status",
            gate,
            "state_input_binding",
        )
    trading = status["official_status"].eq("TRADING")
    if not raw.index.equals(status.index[trading]) or not adj.index.equals(status.index[trading]):
        return _add_failure(
            report,
            "state_source_keys",
            "state TRADING keys must exactly equal raw and adjustment keys",
            gate,
            "state_input_binding",
        )
    if not state["included_in_state_engine"].eq(trading).all():
        good &= _add_failure(
            report,
            "state_inclusion",
            "only official TRADING rows may enter the state engine",
            gate,
            "state_input_binding",
        )

    raw_columns = ["open", "high", "low", "close"]
    state_raw_columns = [f"raw_{name}" for name in raw_columns]
    state_adjusted_columns = [f"adjusted_{name}" for name in raw_columns]
    trading_state = state.loc[trading]
    raw_values = raw[raw_columns].to_numpy(dtype=np.float64)
    factor_values = adj["adj_factor"].to_numpy(dtype=np.float64)
    if not np.array_equal(trading_state[state_raw_columns].to_numpy(dtype=np.float64), raw_values):
        good &= _add_failure(
            report, "state_raw_values", "state raw OHLC differs from raw_daily", gate, "state_input_binding"
        )
    if not np.array_equal(trading_state["adj_factor"].to_numpy(dtype=np.float64), factor_values):
        good &= _add_failure(
            report, "state_adj_values", "state adjustment differs from same-day factor", gate, "state_input_binding"
        )
    expected_adjusted = raw_values * factor_values[:, None]
    if not np.array_equal(trading_state[state_adjusted_columns].to_numpy(dtype=np.float64), expected_adjusted):
        good &= _add_failure(
            report,
            "state_adjusted_values",
            "state OHLC is not exact float64 raw OHLC x same-day adjustment factor",
            gate,
            "state_input_binding",
        )

    suspended = ~trading
    nullable_value_columns = [*state_raw_columns, "adj_factor", *state_adjusted_columns]
    if state.loc[suspended, nullable_value_columns].notna().any(axis=None):
        good &= _add_failure(
            report, "state_suspended_values", "SUSPENDED rows must not carry price inputs", gate, "state_input_binding"
        )
    if state.loc[suspended, ["raw_row_sha256", "adj_factor_row_sha256"]].notna().any(axis=None):
        good &= _add_failure(
            report,
            "state_suspended_hashes",
            "SUSPENDED rows must not claim raw/adjustment hashes",
            gate,
            "state_input_binding",
        )

    expected_raw_hash = _row_hashes(frames["raw_daily"], "raw_daily")
    expected_adj_hash = _row_hashes(frames["adj_factor"], "adj_factor")
    expected_status_hash = _row_hashes(frames["official_daily_status"], "official_daily_status")
    if not trading_state["raw_row_sha256"].astype("string").eq(expected_raw_hash).all():
        good &= _add_failure(report, "state_raw_hash", "raw row hash mismatch", gate, "state_input_binding")
    if not trading_state["adj_factor_row_sha256"].astype("string").eq(expected_adj_hash).all():
        good &= _add_failure(report, "state_adj_hash", "adjustment row hash mismatch", gate, "state_input_binding")
    if not state["status_row_sha256"].astype("string").eq(expected_status_hash).all():
        good &= _add_failure(
            report, "state_status_hash", "official status row hash mismatch", gate, "state_input_binding"
        )

    status_ingested_ns = status[INGESTED_AT].astype("int64").to_numpy()
    raw_ingested_ns = np.full(len(status), np.iinfo(np.int64).min, dtype=np.int64)
    adj_ingested_ns = np.full(len(status), np.iinfo(np.int64).min, dtype=np.int64)
    raw_ingested_ns[trading.to_numpy()] = raw[INGESTED_AT].astype("int64").to_numpy()
    adj_ingested_ns[trading.to_numpy()] = adj[INGESTED_AT].astype("int64").to_numpy()
    latest_ingested_ns = np.maximum.reduce([status_ingested_ns, raw_ingested_ns, adj_ingested_ns])
    computed_ns = state[COMPUTED_AT].astype("int64").to_numpy()
    if not ((computed_ns >= latest_ingested_ns) & (computed_ns <= created.value)).all():
        good &= _add_failure(
            report,
            "state_compute_order",
            "state input must be computed after all bound sources and before the manifest",
            gate,
            "state_input_binding",
        )
    expected_input_hash = pd.Series(
        (
            state_input_row_sha256({name: row[name] for name in STATE_INPUT_HASH_COLUMNS})
            for row in frames["state_input_binding"].to_dict(orient="records")
        ),
        index=state.index,
        dtype="string",
    )
    if not state["input_row_sha256"].astype("string").eq(expected_input_hash).all():
        good &= _add_failure(report, "state_input_hash", "state input row hash mismatch", gate, "state_input_binding")
    for column in ("raw_row_sha256", "adj_factor_row_sha256", "status_row_sha256", "input_row_sha256"):
        values = state[column].dropna()
        if not values.map(_is_sha256).all():
            good &= _add_failure(report, "state_hash_format", f"invalid hash in {column}", gate, "state_input_binding")
    return bool(good)


def _validate_loaded(
    report: DataValidationReport,
    manifest: Mapping[str, Any],
    created: pd.Timestamp,
    artifact_blobs: Mapping[str, bytes],
    evidence_blobs: Mapping[str, bytes],
    source_blobs: Mapping[str, bytes],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    schema_good = True
    object_blobs = {**artifact_blobs, **evidence_blobs}
    for name in (*REQUIRED_ARTIFACTS, *REQUIRED_EVIDENCE_OBJECTS):
        try:
            frames[name] = _parse_artifact(name, object_blobs[name])
        except Exception as exc:
            schema_good = False
            report.add(
                "artifact_schema",
                f"failed closed: {type(exc).__name__}: {exc}",
                "artifact_schema",
                name,
            )
    report.gates["artifact_schema"] = schema_good
    if not schema_good:
        return frames, {}

    guarded_checks = (
        ("artifact_semantics", lambda: _validate_artifact_semantics(report, frames)),
        ("source_time_order", lambda: _validate_source_time_order(report, frames, created)),
    )
    for gate, check in guarded_checks:
        try:
            report.gates[gate] = bool(check())
        except Exception as exc:
            report.add("semantic_exception", f"failed closed: {type(exc).__name__}: {exc}", gate)

    try:
        responses = _parse_source_responses(source_blobs, created)
    except Exception as exc:
        report.gates["source_time_order"] = False
        report.add(
            "source_response_time_or_shape",
            "a source envelope could not prove its ordered provenance timestamps",
            "source_time_order",
        )
        report.add(
            "source_archive",
            f"failed closed: {type(exc).__name__}: {exc}",
            "source_archive_binding",
        )
        return frames, {}
    try:
        report.gates["source_archive_binding"] = _validate_source_archive_binding(report, frames, responses)
    except Exception as exc:
        report.add("source_binding_exception", f"failed closed: {type(exc).__name__}: {exc}", "source_archive_binding")

    relevant: set[str] = set()
    try:
        coverage_good, relevant = _validate_universe_coverage(report, frames, responses)
        report.gates["universe_response_coverage"] = coverage_good
    except Exception as exc:
        report.add("coverage_exception", f"failed closed: {type(exc).__name__}: {exc}", "universe_response_coverage")
    try:
        report.gates["namechange_response_evidence"] = _validate_namechange_evidence(
            report, frames, responses, relevant
        )
    except Exception as exc:
        report.add(
            "namechange_exception",
            f"failed closed: {type(exc).__name__}: {exc}",
            "namechange_response_evidence",
        )
    try:
        report.gates["terminal_actions_exact"] = _validate_terminal_actions(report, frames)
    except Exception as exc:
        report.add("terminal_exception", f"failed closed: {type(exc).__name__}: {exc}", "terminal_actions_exact")
    try:
        report.gates["state_input_row_binding"] = _validate_state_input_binding(report, frames, created)
    except Exception as exc:
        report.add(
            "state_binding_exception",
            f"failed closed: {type(exc).__name__}: {exc}",
            "state_input_row_binding",
        )
    report.evidence.update(
        {
            "created_at_utc": manifest["created_at_utc"],
            "protocol_id": manifest["protocol_id"],
            "protocol_sha256": manifest["protocol_sha256"],
            "artifact_sha256": {name: manifest["artifacts"][name]["sha256"] for name in sorted(manifest["artifacts"])},
            "evidence_object_sha256": {
                name: manifest["evidence_objects"][name]["sha256"] for name in sorted(manifest["evidence_objects"])
            },
            "artifact_rows": {name: int(len(frame)) for name, frame in sorted(frames.items())},
            "source_response_objects": len(responses),
            "relevant_ldp_symbols": len(relevant),
            "official_status_rows": int(len(frames["official_daily_status"])),
            "state_input_rows": int(len(frames["state_input_binding"])),
        }
    )
    return frames, responses


def load_verified_snapshot(
    manifest_path: str | Path,
) -> tuple[DataValidationReport, VerifiedDataSnapshot | None]:
    """Validate one formal manifest and return immutable replay inputs.

    This is the sole formal loading entry point.  Malformed paths, JSON,
    parquet dtypes, and semantic values are converted to issues; none escape as
    exceptions.  ``snapshot`` is non-null only when every formal data gate is
    true.
    """

    report = DataValidationReport()
    try:
        manifest_sha, manifest, created, artifact_blobs, evidence_blobs, source_blobs = _load_content_addressed_objects(
            manifest_path
        )
        report.manifest_sha256 = manifest_sha
        report.gates["content_addressed_manifest"] = True
        _, responses = _validate_loaded(report, manifest, created, artifact_blobs, evidence_blobs, source_blobs)
        report.data_ready = all(report.gates.get(name, False) for name in FORMAL_GATES)
        report.valid = report.data_ready
        if not report.valid:
            return report, None
        snapshot = VerifiedDataSnapshot(
            manifest_sha256=manifest_sha,
            created_at_utc=manifest["created_at_utc"],
            manifest=manifest,
            artifact_blobs=artifact_blobs,
            evidence_blobs=evidence_blobs,
            responses=responses,
        )
        return report, snapshot
    except Exception as exc:
        report.add(
            "formal_entry_closed",
            f"failed closed: {type(exc).__name__}: {exc}",
            "content_addressed_manifest",
        )
        report.valid = False
        report.data_ready = False
        return report, None


def validate_data_manifest(manifest_path: str | Path) -> DataValidationReport:
    """Return only the serialisable formal data-gate report."""

    report, _ = load_verified_snapshot(manifest_path)
    return report


__all__ = [
    "ALL_SPECS",
    "ARTIFACT_SPECS",
    "COMPUTED_AT",
    "CONTRACT_ID",
    "CORPORATE_ACTION_TYPES",
    "DataValidationReport",
    "EVIDENCE_SPECS",
    "FROZEN_PROTOCOL_PATH",
    "FORMAL_GATES",
    "INGESTED_AT",
    "MANIFEST_VERSION",
    "OFFICIAL_STATUSES",
    "OBJECT_AUTHORITIES",
    "PARQUET_MEDIA_TYPE",
    "PROTOCOL_ID",
    "REQUIRED_ARTIFACTS",
    "REQUIRED_EVIDENCE_OBJECTS",
    "SOURCE_ASOF",
    "SOURCE_RESPONSE_MEDIA_TYPE",
    "SOURCE_RESPONSE_SCHEMA",
    "STATE_INPUT_HASH_COLUMNS",
    "VerifiedDataSnapshot",
    "canonical_json_bytes",
    "canonical_row_sha256",
    "canonical_source_row",
    "frozen_protocol_sha256",
    "load_verified_snapshot",
    "object_provenance_sha256",
    "protocol_file_sha256",
    "sha256_bytes",
    "state_input_row_sha256",
    "symbol_set_sha256",
    "validate_data_manifest",
]
