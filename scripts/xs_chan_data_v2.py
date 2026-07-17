"""Content-addressed data contract for the preregistered XS/Chan V2 study.

This module is deliberately independent from :mod:`czsc` and from every data
vendor SDK.  It only reads a local bundle, verifies its evidence, and derives
two gates:

``forward_start_allowed``
    The immutable historical bundle, state audit, and frozen protocol are fit
    to *start* append-only forward collection.

``confirmatory_oos``
    The first gate is true and an optional external
    :mod:`xs_chan_oos_ledger` has reached the protocol's minimum number of
    complete, internally consistent weekly decision/execution cycles.

Neither value is accepted from a manifest or protocol.  Reserved hand-written
readiness fields are rejected.  A missing artifact, unverifiable semantic
claim, unsafe path, or failed cross-table check closes the corresponding gate.

Bundle layout
-------------
The ten required parquet files use their dataset names as filenames.  A
``source_metadata.json`` file records vendor-neutral source/API provenance and
``protocol.json`` is a byte-for-byte copy of the repository's frozen V2
protocol.  The append-only OOS ledger is deliberately external to the bundle
and can be supplied to :func:`validate_bundle`.

The module never opens a network connection and contains no credentials.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api import types as ptypes

CONTRACT_ID = "xs_chan_data_bundle_v2"
MANIFEST_VERSION = 2
HASH_ALGORITHM = "sha256"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_SOURCE_METADATA = "source_metadata.json"
DEFAULT_PROTOCOL = "protocol.json"
DEFAULT_STATE_CACHE_MANIFEST = "state_cache_manifest.json"
FROZEN_PROTOCOL_PATH = Path(__file__).resolve().with_name("xs_chan_protocol_v2.json")

MIN_HISTORY_SESSIONS = 250
MIN_STATE_AUDIT_SYMBOLS = 100
MIN_STATE_AUDIT_CUTOFFS = 20
MIN_CONFIRMATORY_WEEKS = 52
STATE_DOMAIN = tuple(range(11))
CANONICAL_BOARDS = ("MAIN", "CHINEXT", "STAR", "BSE")
CORPORATE_ACTION_TYPES = (
    "cash_dividend",
    "share_change",
    "rights_issue",
    "delist_cash",
    "delist_share",
    "delist_writeoff",
)
RECONCILIATION_PARTITIONS = (
    "security_master:list_status=L",
    "security_master:list_status=D",
    "security_master:list_status=P",
    "raw_daily:bundle_symbol_set",
    "corporate_actions:coverage_universe",
)
STATE_ENGINE_COMPONENTS = (
    "cache_builder",
    "trend_regime",
    "format_standard_kline",
    "native_extension",
)
REQUIRED_ENGINEERING_GATES = (
    "state_recompute_engineering",
    "feature_prefix_engineering",
    "source_archive_reconciliation_engineering",
    "suspension_reconciliation_engineering",
    "calendar_ledger_binding_engineering",
    "execution_replay_engineering",
    "corporate_action_replay_engineering",
    "statistics_replay_engineering",
    "artifact_semantic_verifier_engineering",
    "external_timestamp_anchor_engineering",
)
FORWARD_DEPENDENCY_KEYS = (
    "data_evidence",
    "state_engine",
    "feature_engine",
    "research_engine",
    "execution_engine",
    "statistics_engine",
    "replay_verifier",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESERVED_READINESS_KEYS = frozenset(
    {
        "ready",
        "data_ready",
        "upgrade_allowed",
        "forward_start_allowed",
        "confirmatory_oos",
        "alpha_validated",
    }
)
FORBIDDEN_SOURCE_KEYS = frozenset(
    {"token", "credential", "credentials", "secret", "password", "api_key", "authorization"}
)


class ContractError(ValueError):
    """Raised when a manifest cannot be built safely."""


@dataclass(frozen=True)
class DatasetSpec:
    """Physical and semantic requirements for one parquet artifact."""

    filename: str
    required_columns: Mapping[str, str]
    unique_key: tuple[str, ...]
    nullable: frozenset[str]
    semantics: Mapping[str, Any]
    date_column: str
    exact_columns: bool = False
    allow_empty: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable validation finding."""

    code: str
    message: str
    dataset: str | None = None
    scope: str = "data"


@dataclass
class ValidationReport:
    """Fail-closed result returned by :func:`validate_bundle`."""

    valid: bool = False
    forward_start_allowed: bool = False
    confirmatory_oos: bool = False
    gates: dict[str, bool] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, dataset: str | None = None, scope: str = "data") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, dataset=dataset, scope=scope))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": CONTRACT_ID,
            "valid": self.valid,
            "forward_start_allowed": self.forward_start_allowed,
            "confirmatory_oos": self.confirmatory_oos,
            "gates": dict(sorted(self.gates.items())),
            "evidence": self.evidence,
            "issues": [asdict(issue) for issue in self.issues],
        }


COMMON_PROVENANCE = {
    "source_asof": "utc_datetime",
    "ingested_at": "utc_datetime",
}


def _columns(*parts: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in parts:
        out.update(part)
    return out


DATASET_SPECS: dict[str, DatasetSpec] = {
    "calendar": DatasetSpec(
        filename="calendar.parquet",
        required_columns=_columns(
            {
                "trade_date": "date",
                "is_open": "bool",
                "prev_trade_date": "date",
                "next_trade_date": "date",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("trade_date",),
        nullable=frozenset({"prev_trade_date", "next_trade_date"}),
        date_column="trade_date",
        semantics={
            "row_scope": "exchange_open_sessions_only",
            "timezone": "Asia/Shanghai",
            "availability_model": "source_asof_and_ingested_at",
            "mutation_policy": "append_only_content_addressed",
        },
    ),
    "security_master": DatasetSpec(
        filename="security_master.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "name": "string",
                "exchange": "string",
                "board": "string",
                "list_date": "date",
                "delist_date": "date",
                "list_status": "string",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code",),
        nullable=frozenset({"delist_date"}),
        date_column="list_date",
        semantics={
            "status_scope": ["L", "D", "P"],
            "universe_rule": "list_date_lte_t_and_delist_date_gt_t",
            "snapshot_policy": "append_only_asof_bundle",
            "missing_security_policy": "ineligible",
            "canonical_board_values": list(CANONICAL_BOARDS),
        },
    ),
    "namechange": DatasetSpec(
        filename="namechange.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "name": "string",
                "effective_from": "date",
                "effective_to": "date",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code", "effective_from"),
        nullable=frozenset({"effective_to"}),
        date_column="effective_from",
        semantics={
            "temporal_model": "effective_dated_non_overlapping",
            "st_rule": "name_contains_st_during_effective_interval",
            "availability_model": "source_asof_and_ingested_at",
            "missing_namechange_policy": "no_change_not_unknown_security",
        },
    ),
    "raw_daily": DatasetSpec(
        filename="raw_daily.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "trade_date": "date",
                "open": "numeric",
                "high": "numeric",
                "low": "numeric",
                "close": "numeric",
                "pre_close": "numeric",
                "vol": "numeric",
                "amount": "numeric",
                "pct_chg": "numeric",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code", "trade_date"),
        nullable=frozenset(),
        date_column="trade_date",
        semantics={
            "price_basis": "raw_unadjusted",
            "execution_price_track": "raw_unadjusted_ohlc",
            "volume_unit": "lots",
            "amount_unit": "thousand_CNY",
            "availability": "post_close_t_for_t_plus_1",
            "mutation_policy": "append_only_content_addressed",
        },
    ),
    "adj_factor": DatasetSpec(
        filename="adj_factor.parquet",
        required_columns=_columns(
            {"ts_code": "string", "trade_date": "date", "adj_factor": "numeric"}, COMMON_PROVENANCE
        ),
        unique_key=("ts_code", "trade_date"),
        nullable=frozenset(),
        date_column="trade_date",
        semantics={
            "factor_price_track": "raw_close_times_same_day_adj_factor",
            "execution_use": "forbidden",
            "availability": "post_close_t_for_t_plus_1",
            "mutation_policy": "append_only_content_addressed",
        },
    ),
    "daily_basic": DatasetSpec(
        filename="daily_basic.parquet",
        required_columns=_columns(
            {"ts_code": "string", "trade_date": "date", "free_share": "numeric"}, COMMON_PROVENANCE
        ),
        unique_key=("ts_code", "trade_date"),
        nullable=frozenset(),
        date_column="trade_date",
        semantics={
            "free_share_unit": "10000_shares",
            "free_float_mcap_formula": "raw_close_times_free_share_times_10000",
            "availability": "post_close_t_for_t_plus_1",
            "missing_value_policy": "ineligible",
        },
    ),
    "stk_limit": DatasetSpec(
        filename="stk_limit.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "trade_date": "date",
                "up_limit": "numeric",
                "down_limit": "numeric",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code", "trade_date"),
        nullable=frozenset(),
        date_column="trade_date",
        semantics={
            "price_basis": "raw_unadjusted",
            "publication_timing": "known_before_execution_session",
            "limit_rule_source": "official_daily_limit_prices",
            "missing_value_policy": "ineligible",
        },
    ),
    "industry_membership": DatasetSpec(
        filename="industry_membership.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "industry_code": "string",
                "industry_name": "string",
                "classification_version": "string",
                "effective_from": "date",
                "effective_to": "date",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code", "classification_version", "effective_from"),
        nullable=frozenset({"effective_to"}),
        date_column="effective_from",
        semantics={
            "classification": "SW2021",
            "temporal_model": "historical_in_date_out_date",
            "membership_scope": "historical_not_current_only",
            "missing_value_policy": "ineligible",
        },
    ),
    "corporate_actions": DatasetSpec(
        filename="corporate_actions.parquet",
        required_columns=_columns(
            {
                "ts_code": "string",
                "effective_date": "date",
                "action_type": "string",
                "cash_per_pre_action_share": "numeric",
                "post_to_pre_share_ratio": "numeric",
                "official_disposal_cash_per_pre_action_share": "numeric",
                "consideration_ts_code": "string",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("ts_code", "effective_date", "action_type"),
        nullable=frozenset(
            {
                "cash_per_pre_action_share",
                "post_to_pre_share_ratio",
                "official_disposal_cash_per_pre_action_share",
                "consideration_ts_code",
            }
        ),
        date_column="effective_date",
        allow_empty=True,
        semantics={
            "action_types": list(CORPORATE_ACTION_TYPES),
            "timing": "apply_entitlements_before_effective_date_open_orders",
            "numeric_nullability": "action_type_specific_exact",
            "delist_share_identifier": "nonempty_consideration_ts_code_required_only_for_delist_share",
            "zero_event_policy": "coverage_proven_by_universe_reconciliation",
            "missing_action_policy": "invalidate_period",
        },
    ),
    "universe_reconciliation": DatasetSpec(
        filename="universe_reconciliation.parquet",
        required_columns=_columns(
            {
                "dataset": "string",
                "partition": "string",
                "trade_date": "date",
                "list_status": "string",
                "expected_count": "integer",
                "received_count": "integer",
                "expected_symbol_set_sha256": "string",
                "received_symbol_set_sha256": "string",
                "request_sha256": "string",
                "response_sha256": "string",
                "missing_count": "integer",
            },
            COMMON_PROVENANCE,
        ),
        unique_key=("dataset", "partition"),
        nullable=frozenset({"trade_date", "list_status"}),
        date_column="source_asof",
        exact_columns=True,
        semantics={
            "partition_contract": list(RECONCILIATION_PARTITIONS),
            "required_list_statuses": ["L", "D", "P"],
            "missing_count_required": 0,
            "expected_received_policy": "count_and_symbol_set_sha256_equal",
            "physical_binding": "security_master_raw_daily_and_corporate_action_coverage",
        },
    ),
    "chan_states": DatasetSpec(
        filename="chan_states.parquet",
        required_columns={"symbol": "string", "dt": "date", "regime": "integer"},
        unique_key=("symbol", "dt"),
        nullable=frozenset(),
        date_column="dt",
        exact_columns=True,
        semantics={
            "state_domain": list(STATE_DOMAIN),
            "causality": "prefix_only",
            "coverage": "every_observed_daily_bar_including_warmup",
            "projection": "exact_symbol_dt_regime_whitelist",
            "warmup_policy": "regime_0_until_120_observed_bars",
            "missing_state_policy": "deny_new_entry",
        },
    ),
    "state_audit": DatasetSpec(
        filename="state_audit.parquet",
        required_columns={
            "symbol": "string",
            "checkpoint_dt": "date",
            "prefix_rows": "integer",
            "expected_rows": "integer",
            "actual_rows": "integer",
            "expected_sha256": "string",
            "actual_sha256": "string",
            "symbol_mismatches": "integer",
            "dt_mismatches": "integer",
            "regime_mismatches": "integer",
            "passed": "bool",
        },
        unique_key=("symbol", "checkpoint_dt"),
        nullable=frozenset(),
        date_column="checkpoint_dt",
        exact_columns=True,
        semantics={
            "audit_type": "prefix_recompute_vs_full_projection_exact",
            "required_symbols": MIN_STATE_AUDIT_SYMBOLS,
            "required_cutoffs_per_symbol": MIN_STATE_AUDIT_CUTOFFS,
            "allowed_mismatches": 0,
        },
    ),
}

REQUIRED_DATASETS = tuple(DATASET_SPECS)

SOURCE_REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {
    "calendar": ("exchange", "start_date", "end_date"),
    "security_master": ("list_statuses", "symbol_count", "symbol_set_sha256"),
    "namechange": ("start_date", "end_date"),
    "raw_daily": ("start_date", "end_date", "adjustment", "symbol_count", "symbol_set_sha256"),
    "adj_factor": ("start_date", "end_date"),
    "daily_basic": ("start_date", "end_date", "fields"),
    "stk_limit": ("start_date", "end_date", "fields"),
    "industry_membership": ("start_date", "end_date", "classification", "membership_scope"),
    "corporate_actions": (
        "start_date",
        "end_date",
        "action_types",
        "coverage_scope",
        "queried_symbol_count",
        "queried_symbol_set_sha256",
    ),
    "universe_reconciliation": ("required_partitions",),
    "chan_states": (
        "engine_sha256",
        "state_domain",
        "price_track",
        "state_cache_manifest_sha256",
        "engine_combination_sha256",
        "raw_input_evidence_sha256",
    ),
    "state_audit": ("minimum_symbols", "minimum_cutoffs", "state_cache_manifest_sha256"),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Return SHA256 of the contract's canonical JSON encoding."""

    return sha256_bytes(_canonical_json(value))


def symbol_set_sha256(values: Sequence[object] | pd.Series | pd.Index) -> str:
    """Hash a unique symbol set using sorted canonical JSON."""

    symbols = sorted({str(value).strip() for value in values if str(value).strip()})
    return sha256_json(symbols)


def _state_engine_combination_sha256(state_manifest: Mapping[str, Any]) -> str:
    identity = state_manifest.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("algorithm_hashes"), Mapping):
        raise ContractError("state-cache identity.algorithm_hashes is missing")
    hashes = dict(identity["algorithm_hashes"])
    if set(hashes) != set(STATE_ENGINE_COMPONENTS) or not all(_is_sha256(value) for value in hashes.values()):
        raise ContractError("state-cache algorithm hashes do not match the frozen engine components")
    return sha256_json(hashes)


def _state_raw_input_evidence_sha256(state_manifest: Mapping[str, Any]) -> str:
    identity = state_manifest.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("sources"), list):
        raise ContractError("state-cache identity.sources is missing")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for entry in identity["sources"]:
        if not isinstance(entry, Mapping):
            raise ContractError("state-cache input fingerprint must be an object")
        name = entry.get("name")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise ContractError("state-cache input names must be unique non-empty strings")
        if not isinstance(size, int) or size < 0 or not _is_sha256(digest):
            raise ContractError("state-cache input size/hash is invalid")
        names.add(name)
        normalized.append({"name": name, "size": size, "sha256": digest})
    if not normalized:
        raise ContractError("state-cache input evidence is empty")
    return sha256_json(sorted(normalized, key=lambda item: item["name"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_protocol_sha256() -> str:
    """Return the byte-level digest of the repository's locked protocol."""

    if not FROZEN_PROTOCOL_PATH.is_file():
        raise ContractError(f"frozen protocol is missing: {FROZEN_PROTOCOL_PATH}")
    return sha256_file(FROZEN_PROTOCOL_PATH)


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest after removing its self-referential hash field."""

    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    return sha256_bytes(_canonical_json(payload))


def compute_data_evidence_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash only immutable data/protocol entries, excluding forward evidence."""

    payload = {
        "contract_id": manifest.get("contract_id"),
        "manifest_version": manifest.get("manifest_version"),
        "datasets": manifest.get("datasets"),
        "source_metadata": manifest.get("source_metadata"),
        "protocol": manifest.get("protocol"),
        "state_cache_manifest": manifest.get("state_cache_manifest"),
    }
    return sha256_bytes(_canonical_json(payload))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ContractError(f"invalid JSON constant: {value}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_scalar(value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _resolve_bundle_argument(root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(value) if value is not None else Path(default)
    return path if path.is_absolute() else root / path


def _safe_bundle_path(root: Path, relative: object, *, expect: str = "file") -> Path:
    """Resolve a manifest path without allowing traversal or symlink aliases."""

    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ContractError(f"unsafe bundle path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"unsafe bundle path: {relative!r}")
    if pure.parts and ":" in pure.parts[0]:
        raise ContractError(f"unsafe bundle path: {relative!r}")

    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContractError(f"symlinks are forbidden in bundles: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ContractError(f"bundle path escapes root: {relative}")
    if expect == "file" and not resolved.is_file():
        raise ContractError(f"bundle path is not a regular file: {relative}")
    if expect == "directory" and not resolved.is_dir():
        raise ContractError(f"bundle path is not a directory: {relative}")
    return resolved


def _relative_to_bundle(root: Path, path: Path, *, expect: str = "file") -> str:
    root_resolved = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ContractError(f"artifact must be inside bundle root: {path}")
    relative = resolved.relative_to(root_resolved).as_posix()
    _safe_bundle_path(root, relative, expect=expect)
    return relative


def _parquet_metadata(path: Path, date_column: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = [{"name": item.name, "type": str(item.type)} for item in parquet.schema_arrow]
    dates = pd.read_parquet(path, columns=[date_column])[date_column]
    normalized = pd.to_datetime(dates, errors="coerce")
    return {
        "rows": int(parquet.metadata.num_rows),
        "schema": schema,
        "date_column": date_column,
        "min_date": None if normalized.empty or normalized.isna().all() else normalized.min().strftime("%Y-%m-%d"),
        "max_date": None if normalized.empty or normalized.isna().all() else normalized.max().strftime("%Y-%m-%d"),
    }


def _contains_forbidden_source_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_SOURCE_KEYS:
                return str(key)
            nested = _contains_forbidden_source_key(item)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _contains_forbidden_source_key(item)
            if nested:
                return nested
    return None


def _validate_source_metadata_shape(source_metadata: Mapping[str, Any]) -> None:
    forbidden = _contains_forbidden_source_key(source_metadata)
    if forbidden:
        raise ContractError(f"source metadata contains forbidden credential key: {forbidden}")
    if set(source_metadata) != set(REQUIRED_DATASETS):
        missing = sorted(set(REQUIRED_DATASETS) - set(source_metadata))
        extra = sorted(set(source_metadata) - set(REQUIRED_DATASETS))
        raise ContractError(f"source metadata dataset mismatch; missing={missing}, extra={extra}")
    for name in REQUIRED_DATASETS:
        entry = source_metadata[name]
        if not isinstance(entry, Mapping):
            raise ContractError(f"source metadata for {name} must be an object")
        for key in ("provider", "api", "parameters", "retrieved_at_utc"):
            if key not in entry:
                raise ContractError(f"source metadata for {name} missing {key}")
        if not isinstance(entry["provider"], str) or not entry["provider"].strip():
            raise ContractError(f"source metadata for {name} has empty provider")
        if not isinstance(entry["api"], str) or not entry["api"].strip():
            raise ContractError(f"source metadata for {name} has empty api")
        params = entry["parameters"]
        if not isinstance(params, Mapping):
            raise ContractError(f"source metadata parameters for {name} must be an object")
        missing_params = [key for key in SOURCE_REQUIRED_PARAMETERS[name] if key not in params]
        if missing_params:
            raise ContractError(f"source metadata parameters for {name} missing {missing_params}")
        timestamp = _parse_utc_scalar(entry["retrieved_at_utc"])
        if pd.isna(timestamp):
            raise ContractError(f"source metadata for {name} has invalid retrieved_at_utc")

    raw_adjustment = str(source_metadata["raw_daily"]["parameters"]["adjustment"]).lower()
    if raw_adjustment not in {"none", "raw", "unadjusted"}:
        raise ContractError("raw_daily source must explicitly declare an unadjusted request")
    status_values = source_metadata["security_master"]["parameters"]["list_statuses"]
    if not isinstance(status_values, list):
        raise ContractError("security_master list_statuses must be a JSON list")
    statuses = {str(value) for value in status_values}
    if statuses != {"L", "D", "P"}:
        raise ContractError("security_master source must request exactly L,D,P")
    for name in ("security_master", "raw_daily"):
        params = source_metadata[name]["parameters"]
        if not isinstance(params["symbol_count"], int) or params["symbol_count"] <= 0:
            raise ContractError(f"{name} source symbol_count must be positive")
        if not _is_sha256(params["symbol_set_sha256"]):
            raise ContractError(f"{name} source symbol_set_sha256 is invalid")
    fields = {str(value) for value in source_metadata["daily_basic"]["parameters"]["fields"]}
    if "free_share" not in fields:
        raise ContractError("daily_basic source fields must include free_share")
    limit_fields = {str(value) for value in source_metadata["stk_limit"]["parameters"]["fields"]}
    if not {"up_limit", "down_limit"}.issubset(limit_fields):
        raise ContractError("stk_limit source fields must include up_limit/down_limit")
    industry = source_metadata["industry_membership"]["parameters"]
    if industry["classification"] != "SW2021" or industry["membership_scope"] != "historical":
        raise ContractError("industry source must declare historical SW2021 membership")
    actions = source_metadata["corporate_actions"]["parameters"]
    if actions["action_types"] != list(CORPORATE_ACTION_TYPES):
        raise ContractError("corporate_actions source action types differ from the frozen contract")
    if actions["coverage_scope"] != "all_raw_symbols_for_full_date_range":
        raise ContractError("corporate_actions source must cover every raw symbol over the full range")
    if not isinstance(actions["queried_symbol_count"], int) or actions["queried_symbol_count"] <= 0:
        raise ContractError("corporate_actions queried_symbol_count must be positive")
    if not _is_sha256(actions["queried_symbol_set_sha256"]):
        raise ContractError("corporate_actions queried_symbol_set_sha256 is invalid")
    reconciliation = source_metadata["universe_reconciliation"]["parameters"]
    if reconciliation["required_partitions"] != list(RECONCILIATION_PARTITIONS):
        raise ContractError("universe reconciliation partitions differ from the frozen contract")
    state_params = source_metadata["chan_states"]["parameters"]
    if state_params["state_domain"] != list(STATE_DOMAIN):
        raise ContractError("chan_states source must declare the complete 0-10 domain")
    if state_params["price_track"] != "raw_times_adj_factor":
        raise ContractError("chan_states source must use raw_times_adj_factor")
    for key in (
        "engine_sha256",
        "state_cache_manifest_sha256",
        "engine_combination_sha256",
        "raw_input_evidence_sha256",
    ):
        if not _is_sha256(state_params[key]):
            raise ContractError(f"chan_states source {key} is invalid")
    if state_params["engine_sha256"] != state_params["engine_combination_sha256"]:
        raise ContractError("chan_states engine_sha256 must equal the state-cache engine combination hash")
    audit_params = source_metadata["state_audit"]["parameters"]
    if int(audit_params["minimum_symbols"]) < MIN_STATE_AUDIT_SYMBOLS:
        raise ContractError("state_audit source minimum_symbols is too small")
    if int(audit_params["minimum_cutoffs"]) < MIN_STATE_AUDIT_CUTOFFS:
        raise ContractError("state_audit source minimum_cutoffs is too small")
    if not _is_sha256(audit_params["state_cache_manifest_sha256"]):
        raise ContractError("state_audit source state_cache_manifest_sha256 is invalid")
    if audit_params["state_cache_manifest_sha256"] != state_params["state_cache_manifest_sha256"]:
        raise ContractError("state and audit source metadata bind different state-cache manifests")


def _validate_state_cache_manifest_shape(state_manifest: Mapping[str, Any]) -> tuple[str, str]:
    """Validate immutable state-cache provenance and return its two derived digests."""

    forbidden = _contains_forbidden_source_key(state_manifest)
    if forbidden:
        raise ContractError(f"state-cache manifest contains forbidden credential key: {forbidden}")
    if state_manifest.get("schema_version") != "xs_chan_state_cache_v1":
        raise ContractError("state-cache schema_version is not xs_chan_state_cache_v1")
    identity = state_manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ContractError("state-cache identity is missing")
    identity_digest = sha256_json(identity)
    if (
        state_manifest.get("cache_id") != identity_digest
        or state_manifest.get("content_digest_sha256") != identity_digest
    ):
        raise ContractError("state-cache identity digest does not match cache_id/content_digest_sha256")
    if identity.get("projection_columns") != ["symbol", "dt", "regime"]:
        raise ContractError("state-cache projection columns differ from the protocol whitelist")
    if identity.get("valid_regimes") != list(STATE_DOMAIN):
        raise ContractError("state-cache identity does not retain the complete 0-10 domain")
    config = identity.get("config")
    if not isinstance(config, Mapping):
        raise ContractError("state-cache identity config is missing")
    if (
        config.get("audit_symbols") != MIN_STATE_AUDIT_SYMBOLS
        or config.get("audit_checkpoints") != MIN_STATE_AUDIT_CUTOFFS
    ):
        raise ContractError("state-cache identity does not freeze the required 100x20 audit")

    engine_digest = _state_engine_combination_sha256(state_manifest)
    identity_hashes = identity["algorithm_hashes"]
    engine = state_manifest.get("engine")
    if not isinstance(engine, Mapping):
        raise ContractError("state-cache engine provenance is missing")
    physical_engine_hashes: dict[str, str] = {}
    for component in STATE_ENGINE_COMPONENTS:
        entry = engine.get(component)
        if not isinstance(entry, Mapping) or not _is_sha256(entry.get("sha256")):
            raise ContractError(f"state-cache engine component {component} is invalid")
        physical_engine_hashes[component] = str(entry["sha256"])
    if physical_engine_hashes != dict(identity_hashes):
        raise ContractError("state-cache engine provenance differs from identity.algorithm_hashes")

    raw_digest = _state_raw_input_evidence_sha256(state_manifest)
    identity_sources = {entry["name"]: entry for entry in identity["sources"]}
    source_records = state_manifest.get("sources")
    if not isinstance(source_records, list) or len(source_records) != len(identity_sources):
        raise ContractError("state-cache source records do not cover every identity input")
    seen_names: set[str] = set()
    seen_symbols: set[str] = set()
    for record in source_records:
        if not isinstance(record, Mapping):
            raise ContractError("state-cache source record must be an object")
        name = record.get("name")
        symbol = record.get("symbol")
        if name not in identity_sources or name in seen_names:
            raise ContractError("state-cache source record names differ from identity inputs")
        fingerprint = identity_sources[str(name)]
        if record.get("size") != fingerprint["size"] or record.get("sha256") != fingerprint["sha256"]:
            raise ContractError("state-cache source record fingerprint differs from identity")
        if not isinstance(symbol, str) or not symbol or symbol in seen_symbols:
            raise ContractError("state-cache source symbols must be unique non-empty strings")
        if record.get("status") != "generated" or record.get("source_rows") != record.get("state_rows"):
            raise ContractError("state-cache source row coverage is incomplete")
        if not isinstance(record.get("source_rows"), int) or record["source_rows"] <= 0:
            raise ContractError("state-cache source row count is invalid")
        seen_names.add(str(name))
        seen_symbols.add(symbol)
    return engine_digest, raw_digest


def build_manifest(
    bundle_root: str | Path,
    *,
    output_path: str | Path | None = None,
    source_metadata_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    state_cache_manifest_path: str | Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and write a content-addressed manifest without using a network.

    All supplied artifacts must live inside ``bundle_root`` and may not be
    symlinks.  Source metadata is mandatory so a caller cannot silently label
    adjusted prices as raw or current industry membership as historical.
    """

    root = Path(bundle_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"bundle root is not a directory: {root}")

    source_path = _resolve_bundle_argument(root, source_metadata_path, DEFAULT_SOURCE_METADATA)
    protocol_file = _resolve_bundle_argument(root, protocol_path, DEFAULT_PROTOCOL)
    state_cache_file = _resolve_bundle_argument(
        root,
        state_cache_manifest_path,
        DEFAULT_STATE_CACHE_MANIFEST,
    )
    source_rel = _relative_to_bundle(root, source_path)
    protocol_rel = _relative_to_bundle(root, protocol_file)
    state_cache_rel = _relative_to_bundle(root, state_cache_file)
    expected_protocol_sha256 = frozen_protocol_sha256()
    if sha256_file(protocol_file) != expected_protocol_sha256:
        raise ContractError(
            f"bundle protocol is not a byte-for-byte copy of {FROZEN_PROTOCOL_PATH.name} ({expected_protocol_sha256})"
        )
    source_metadata = _load_json(source_path)
    _validate_source_metadata_shape(source_metadata)
    state_cache_manifest = _load_json(state_cache_file)
    engine_digest, raw_input_digest = _validate_state_cache_manifest_shape(state_cache_manifest)
    state_params = source_metadata["chan_states"]["parameters"]
    state_cache_sha256 = sha256_file(state_cache_file)
    if state_params["state_cache_manifest_sha256"] != state_cache_sha256:
        raise ContractError("chan_states source metadata does not bind the physical state-cache manifest")
    if state_params["engine_combination_sha256"] != engine_digest:
        raise ContractError("chan_states source metadata engine combination hash mismatch")
    if state_params["raw_input_evidence_sha256"] != raw_input_digest:
        raise ContractError("chan_states source metadata raw-input evidence hash mismatch")

    datasets: dict[str, Any] = {}
    for name, spec in DATASET_SPECS.items():
        path = root / spec.filename
        relative = _relative_to_bundle(root, path)
        metadata = _parquet_metadata(path, spec.date_column)
        datasets[name] = {
            "path": relative,
            "format": "parquet",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            **metadata,
            "semantics": copy.deepcopy(dict(spec.semantics)),
            "source": copy.deepcopy(source_metadata[name]),
        }

    protocol = _load_json(protocol_file)
    manifest: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "manifest_version": MANIFEST_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "created_at_utc": created_at_utc or _utc_now_iso(),
        "datasets": datasets,
        "source_metadata": {
            "path": source_rel,
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        },
        "protocol": {
            "path": protocol_rel,
            "sha256": sha256_file(protocol_file),
            "size_bytes": protocol_file.stat().st_size,
            "protocol_id": protocol.get("protocol_id"),
            "frozen_at_utc": protocol.get("frozen_at_utc"),
        },
        "state_cache_manifest": {
            "path": state_cache_rel,
            "sha256": state_cache_sha256,
            "size_bytes": state_cache_file.stat().st_size,
            "cache_id": state_cache_manifest.get("cache_id"),
        },
    }
    manifest["data_evidence_sha256"] = compute_data_evidence_sha256(manifest)

    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    output = _resolve_bundle_argument(root, output_path, DEFAULT_MANIFEST)
    output_parent = output.parent.resolve(strict=True)
    if output_parent != root or output.is_symlink():
        raise ContractError("manifest output must be a non-symlink direct child of bundle root")
    output.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")
    return manifest


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _is_string_series(series: pd.Series) -> bool:
    if ptypes.is_string_dtype(series.dtype):
        return True
    values = series.dropna()
    return bool(values.map(lambda value: isinstance(value, str)).all())


def _is_utc_datetime_series(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return str(dtype.tz).upper() in {"UTC", "UTC+00:00"}
    return False


def _normalize_date(series: pd.Series) -> pd.Series:
    if ptypes.is_datetime64tz_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.normalize()


def _validate_column_kind(series: pd.Series, kind: str) -> bool:
    if kind == "string":
        return _is_string_series(series)
    if kind == "bool":
        return ptypes.is_bool_dtype(series.dtype)
    if kind == "numeric":
        return ptypes.is_numeric_dtype(series.dtype) and not ptypes.is_bool_dtype(series.dtype)
    if kind == "integer":
        return ptypes.is_integer_dtype(series.dtype) and not ptypes.is_bool_dtype(series.dtype)
    if kind == "date":
        return _normalize_date(series).notna().eq(series.notna()).all()
    if kind == "utc_datetime":
        return _is_utc_datetime_series(series)
    raise AssertionError(f"unknown column kind: {kind}")


def _manifest_artifact(
    report: ValidationReport,
    root: Path,
    entry: object,
    *,
    label: str,
    expect: str = "file",
) -> Path | None:
    if not isinstance(entry, Mapping):
        report.add("manifest_entry", f"{label} entry must be an object", scope="manifest")
        return None
    try:
        path = _safe_bundle_path(root, entry.get("path"), expect=expect)
    except ContractError as exc:
        report.add("unsafe_path", str(exc), dataset=label, scope="manifest")
        return None
    return path


def _check_file_evidence(
    report: ValidationReport,
    path: Path,
    entry: Mapping[str, Any],
    *,
    label: str,
) -> bool:
    good = True
    expected_hash = entry.get("sha256")
    if not _is_sha256(expected_hash) or sha256_file(path) != expected_hash:
        report.add("file_hash", f"SHA256 mismatch for {label}", dataset=label, scope="manifest")
        good = False
    if not isinstance(entry.get("size_bytes"), int) or path.stat().st_size != entry.get("size_bytes"):
        report.add("file_size", f"size mismatch for {label}", dataset=label, scope="manifest")
        good = False
    return good


def _validate_manifest_header(report: ValidationReport, manifest: Mapping[str, Any]) -> bool:
    good = True
    allowed_fields = {
        "contract_id",
        "manifest_version",
        "hash_algorithm",
        "created_at_utc",
        "datasets",
        "source_metadata",
        "protocol",
        "state_cache_manifest",
        "data_evidence_sha256",
        "manifest_sha256",
    }
    unexpected = sorted(set(manifest) - allowed_fields)
    missing = sorted(allowed_fields - set(manifest))
    if unexpected or missing:
        report.add(
            "manifest_fields",
            f"manifest fields differ from contract; missing={missing}, unexpected={unexpected}",
            scope="manifest",
        )
        good = False
    for key in RESERVED_READINESS_KEYS:
        if key in manifest:
            report.add("manual_readiness", f"manifest contains forbidden readiness field: {key}", scope="manifest")
            good = False
    expected = {
        "contract_id": CONTRACT_ID,
        "manifest_version": MANIFEST_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            report.add("manifest_header", f"manifest {key} must equal {value!r}", scope="manifest")
            good = False
    recorded_hash = manifest.get("manifest_sha256")
    if not _is_sha256(recorded_hash) or recorded_hash != compute_manifest_sha256(manifest):
        report.add("manifest_hash", "manifest self-hash mismatch", scope="manifest")
        good = False
    if not _is_sha256(manifest.get("data_evidence_sha256")):
        report.add("data_evidence_hash", "missing or invalid data evidence hash", scope="manifest")
        good = False
    elif manifest["data_evidence_sha256"] != compute_data_evidence_sha256(manifest):
        report.add("data_evidence_hash", "data evidence hash mismatch", scope="manifest")
        good = False
    created = _parse_utc_scalar(manifest.get("created_at_utc"))
    if pd.isna(created):
        report.add("created_at", "manifest created_at_utc is invalid", scope="manifest")
        good = False
    elif created > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
        report.add("created_at", "manifest created_at_utc is in the future", scope="manifest")
        good = False
    return good


def _validate_source_entry(
    report: ValidationReport,
    name: str,
    entry: object,
    frame: pd.DataFrame,
) -> bool:
    if not isinstance(entry, Mapping):
        report.add("source_metadata", "dataset source entry must be an object", dataset=name)
        return False
    if set(entry) != {"provider", "api", "parameters", "retrieved_at_utc"}:
        report.add("source_metadata", "dataset source entry fields differ from the exact contract", dataset=name)
        return False
    retrieved = _parse_utc_scalar(entry.get("retrieved_at_utc"))
    if pd.isna(retrieved):
        report.add("source_metadata", "dataset source retrieved_at_utc is invalid", dataset=name)
        return False
    if {"source_asof", "ingested_at"}.issubset(frame.columns):
        max_source = frame["source_asof"].max()
        max_ingested = frame["ingested_at"].max()
        if retrieved < max_source or retrieved > max_ingested + pd.Timedelta(days=1):
            report.add(
                "source_time",
                "retrieved_at_utc must be >= max source_asof and no later than one day after max ingested_at",
                dataset=name,
            )
            return False
    params = entry.get("parameters")
    if not isinstance(params, Mapping):
        report.add("source_metadata", "dataset source parameters must be an object", dataset=name)
        return False
    if "start_date" in params and "end_date" in params and not frame.empty:
        start = pd.to_datetime(str(params["start_date"]), errors="coerce")
        end = pd.to_datetime(str(params["end_date"]), errors="coerce")
        date_column = DATASET_SPECS[name].date_column
        if pd.isna(start) or pd.isna(end) or start > frame[date_column].min() or end < frame[date_column].max():
            report.add("source_range", "declared source query range does not cover physical rows", dataset=name)
            return False
    return True


def _validate_dataset(
    report: ValidationReport,
    root: Path,
    name: str,
    entry: object,
) -> tuple[bool, pd.DataFrame | None]:
    spec = DATASET_SPECS[name]
    if not isinstance(entry, Mapping):
        report.add("dataset_entry", "dataset manifest entry must be an object", dataset=name)
        return False, None
    path = _manifest_artifact(report, root, entry, label=name)
    if path is None:
        return False, None
    good = _check_file_evidence(report, path, entry, label=name)
    if entry.get("format") != "parquet":
        report.add("dataset_format", "only parquet is allowed", dataset=name)
        good = False
    if entry.get("semantics") != dict(spec.semantics):
        report.add("semantic_declaration", "semantic declaration does not match the frozen contract", dataset=name)
        good = False

    try:
        parquet = pq.ParquetFile(path)
        actual_schema = [{"name": item.name, "type": str(item.type)} for item in parquet.schema_arrow]
        actual_rows = int(parquet.metadata.num_rows)
        frame = pd.read_parquet(path)
    except Exception as exc:
        report.add("parquet_read", f"cannot read parquet: {exc}", dataset=name)
        return False, None
    if entry.get("schema") != actual_schema:
        report.add("schema_fingerprint", "manifest schema differs from parquet schema", dataset=name)
        good = False
    if entry.get("rows") != actual_rows or len(frame) != actual_rows:
        report.add("row_count", "manifest/parquet/dataframe row count mismatch", dataset=name)
        good = False
    if actual_rows == 0 and not spec.allow_empty:
        report.add("empty_dataset", "required dataset is empty", dataset=name)
        good = False

    required = set(spec.required_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        report.add("missing_columns", f"missing required columns: {missing}", dataset=name)
        return False, frame
    if spec.exact_columns and tuple(frame.columns) != tuple(spec.required_columns):
        extra = sorted(set(frame.columns) - required)
        report.add(
            "unsafe_columns",
            f"safe projection has wrong order or extra columns: {extra}",
            dataset=name,
        )
        good = False

    for column, kind in spec.required_columns.items():
        if not _validate_column_kind(frame[column], kind):
            report.add("column_type", f"column {column} must have kind {kind}", dataset=name)
            good = False
    for column in required - set(spec.nullable):
        if frame[column].isna().any():
            report.add("null_value", f"column {column} contains null", dataset=name)
            good = False
    if frame[list(spec.unique_key)].isna().any(axis=None):
        report.add("null_key", f"unique key contains null: {spec.unique_key}", dataset=name)
        good = False
    if frame.duplicated(list(spec.unique_key)).any():
        report.add("duplicate_key", f"duplicate unique key: {spec.unique_key}", dataset=name)
        good = False

    for column, kind in spec.required_columns.items():
        if kind == "date":
            frame[column] = _normalize_date(frame[column])
        elif kind == "utc_datetime" and _is_utc_datetime_series(frame[column]):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    if (
        {"source_asof", "ingested_at"}.issubset(frame.columns)
        and _is_utc_datetime_series(frame["source_asof"])
        and _is_utc_datetime_series(frame["ingested_at"])
        and (frame["source_asof"] > frame["ingested_at"]).any()
    ):
        report.add("provenance_order", "source_asof occurs after ingested_at", dataset=name)
        good = False
    source_good = _validate_source_entry(report, name, entry.get("source"), frame)
    good &= source_good

    date_values = frame[spec.date_column]
    actual_min = None if date_values.empty or date_values.isna().all() else date_values.min().strftime("%Y-%m-%d")
    actual_max = None if date_values.empty or date_values.isna().all() else date_values.max().strftime("%Y-%m-%d")
    if (
        entry.get("date_column") != spec.date_column
        or entry.get("min_date") != actual_min
        or entry.get("max_date") != actual_max
    ):
        report.add("date_fingerprint", "manifest date range differs from physical data", dataset=name)
        good = False
    return good, frame


def _intervals_non_overlapping(
    report: ValidationReport,
    frame: pd.DataFrame,
    *,
    dataset: str,
    group_columns: Sequence[str],
    start_column: str,
    end_column: str,
) -> bool:
    good = True
    if ((frame[end_column].notna()) & (frame[end_column] < frame[start_column])).any():
        report.add("invalid_interval", "effective_to precedes effective_from", dataset=dataset)
        good = False
    for _, group in frame.sort_values(list(group_columns) + [start_column]).groupby(list(group_columns), sort=False):
        starts = group[start_column].to_numpy(dtype="datetime64[ns]")
        ends = group[end_column].fillna(pd.Timestamp.max.normalize()).to_numpy(dtype="datetime64[ns]")
        if len(group) > 1 and (starts[1:] <= ends[:-1]).any():
            report.add("overlapping_interval", "effective intervals overlap", dataset=dataset)
            return False
    return good


def _dataset_semantics(report: ValidationReport, frames: Mapping[str, pd.DataFrame]) -> dict[str, bool]:
    gates: dict[str, bool] = {}

    calendar = frames["calendar"].sort_values("trade_date").reset_index(drop=True)
    calendar_good = bool(calendar["is_open"].all())
    if not calendar_good:
        report.add("calendar_scope", "calendar must contain open sessions only", dataset="calendar")
    dates = calendar["trade_date"]
    expected_prev = dates.shift(1)
    expected_next = dates.shift(-1)
    prev_equal = calendar["prev_trade_date"].fillna(pd.Timestamp.min).eq(expected_prev.fillna(pd.Timestamp.min))
    next_equal = calendar["next_trade_date"].fillna(pd.Timestamp.max).eq(expected_next.fillna(pd.Timestamp.max))
    if not prev_equal.all() or not next_equal.all():
        report.add("calendar_links", "prev/next trade-date links are inconsistent", dataset="calendar")
        calendar_good = False
    if len(calendar) < MIN_HISTORY_SESSIONS:
        report.add("calendar_history", f"calendar has fewer than {MIN_HISTORY_SESSIONS} sessions", dataset="calendar")
        calendar_good = False
    gates["calendar_semantics"] = calendar_good

    security = frames["security_master"]
    security_good = True
    if not security["list_status"].isin({"L", "D", "P"}).all():
        report.add("security_status", "list_status outside L,D,P", dataset="security_master")
        security_good = False
    if not security["board"].isin(CANONICAL_BOARDS).all():
        report.add(
            "security_board", "board is outside the canonical MAIN/CHINEXT/STAR/BSE enum", dataset="security_master"
        )
        security_good = False
    if ((security["delist_date"].notna()) & (security["delist_date"] < security["list_date"])).any():
        report.add("security_dates", "delist_date precedes list_date", dataset="security_master")
        security_good = False
    gates["security_master_semantics"] = security_good

    gates["namechange_semantics"] = _intervals_non_overlapping(
        report,
        frames["namechange"],
        dataset="namechange",
        group_columns=["ts_code"],
        start_column="effective_from",
        end_column="effective_to",
    )

    raw = frames["raw_daily"]
    raw_good = True
    positive = ["open", "high", "low", "close", "pre_close"]
    if not np.isfinite(raw[positive].to_numpy(dtype=float)).all() or (raw[positive] <= 0).any(axis=None):
        report.add("raw_price", "raw prices must be finite and positive", dataset="raw_daily")
        raw_good = False
    if not np.isfinite(raw[["vol", "amount", "pct_chg"]].to_numpy(dtype=float)).all():
        report.add("raw_numeric", "raw numeric fields must be finite", dataset="raw_daily")
        raw_good = False
    if (raw[["vol", "amount"]] < 0).any(axis=None):
        report.add("raw_volume", "volume and amount must be non-negative", dataset="raw_daily")
        raw_good = False
    envelope = (raw["high"] >= raw[["open", "close", "low"]].max(axis=1)) & (
        raw["low"] <= raw[["open", "close", "high"]].min(axis=1)
    )
    if not envelope.all():
        report.add("ohlc_envelope", "OHLC envelope is inconsistent", dataset="raw_daily")
        raw_good = False
    gates["raw_daily_semantics"] = raw_good

    factor_good = bool(
        np.isfinite(frames["adj_factor"]["adj_factor"].to_numpy(dtype=float)).all()
        and (frames["adj_factor"]["adj_factor"] > 0).all()
    )
    if not factor_good:
        report.add("adj_factor_value", "adj_factor must be finite and positive", dataset="adj_factor")
    gates["adj_factor_semantics"] = factor_good

    basic_good = bool(
        np.isfinite(frames["daily_basic"]["free_share"].to_numpy(dtype=float)).all()
        and (frames["daily_basic"]["free_share"] > 0).all()
    )
    if not basic_good:
        report.add("free_share_value", "free_share must be finite and positive", dataset="daily_basic")
    gates["daily_basic_semantics"] = basic_good

    limits = frames["stk_limit"]
    limit_good = bool(
        np.isfinite(limits[["up_limit", "down_limit"]].to_numpy(dtype=float)).all()
        and (limits["down_limit"] > 0).all()
        and (limits["up_limit"] > limits["down_limit"]).all()
    )
    if not limit_good:
        report.add("stk_limit_value", "official limits must satisfy 0 < down_limit < up_limit", dataset="stk_limit")
    gates["stk_limit_semantics"] = limit_good

    industry = frames["industry_membership"]
    industry_good = bool((industry["classification_version"] == "SW2021").all())
    if not industry_good:
        report.add("industry_version", "classification_version must be SW2021", dataset="industry_membership")
    industry_good &= _intervals_non_overlapping(
        report,
        industry,
        dataset="industry_membership",
        group_columns=["ts_code", "classification_version"],
        start_column="effective_from",
        end_column="effective_to",
    )
    gates["industry_membership_semantics"] = industry_good

    actions = frames["corporate_actions"]
    actions_good = True
    if not actions["action_type"].isin(CORPORATE_ACTION_TYPES).all():
        report.add(
            "corporate_action_type", "corporate action type is outside the frozen set", dataset="corporate_actions"
        )
        actions_good = False
    numeric_action_columns = [
        "cash_per_pre_action_share",
        "post_to_pre_share_ratio",
        "official_disposal_cash_per_pre_action_share",
    ]
    finite_actions = actions[numeric_action_columns].apply(
        lambda column: column.isna() | np.isfinite(column.astype(float))
    )
    if not finite_actions.all(axis=None):
        report.add(
            "corporate_action_value", "corporate action numerics must be finite or null", dataset="corporate_actions"
        )
        actions_good = False
    cash = actions["cash_per_pre_action_share"]
    ratio = actions["post_to_pre_share_ratio"]
    disposal = actions["official_disposal_cash_per_pre_action_share"]
    consideration = actions["consideration_ts_code"]
    if ((cash.notna()) & (cash < 0)).any() or ((disposal.notna()) & (disposal < 0)).any():
        report.add(
            "corporate_action_cash", "corporate action cash values must be non-negative", dataset="corporate_actions"
        )
        actions_good = False
    if ((ratio.notna()) & (ratio <= 0)).any():
        report.add("corporate_action_ratio", "post_to_pre_share_ratio must be positive", dataset="corporate_actions")
        actions_good = False
    if consideration.notna().any() and consideration.dropna().astype(str).str.strip().eq("").any():
        report.add(
            "corporate_action_consideration",
            "consideration_ts_code must be a non-empty identifier when present",
            dataset="corporate_actions",
        )
        actions_good = False
    applicability = {
        "cash_dividend": (True, False, False, False),
        "share_change": (False, True, False, False),
        "rights_issue": (False, False, True, False),
        "delist_cash": (True, False, False, False),
        "delist_share": (False, True, False, True),
        "delist_writeoff": (False, False, False, False),
    }
    for action_type, required_values in applicability.items():
        rows = actions["action_type"].eq(action_type)
        actual_values = (cash.notna(), ratio.notna(), disposal.notna(), consideration.notna())
        if any(
            not actual[rows].eq(required).all() for actual, required in zip(actual_values, required_values, strict=True)
        ):
            report.add(
                "corporate_action_applicability",
                f"numeric nullability is invalid for {action_type}",
                dataset="corporate_actions",
            )
            actions_good = False
    gates["corporate_actions_semantics"] = actions_good

    reconciliation = frames["universe_reconciliation"]
    reconciliation_good = True
    if set(reconciliation["partition"]) != set(RECONCILIATION_PARTITIONS) or len(reconciliation) != len(
        RECONCILIATION_PARTITIONS
    ):
        report.add(
            "reconciliation_partitions",
            "universe reconciliation must contain each frozen partition exactly once",
            dataset="universe_reconciliation",
        )
        reconciliation_good = False
    expected_dataset = reconciliation["partition"].str.split(":", n=1).str[0]
    if not reconciliation["dataset"].eq(expected_dataset).all():
        report.add("reconciliation_dataset", "partition dataset prefix mismatch", dataset="universe_reconciliation")
        reconciliation_good = False
    count_match = (
        reconciliation["expected_count"].eq(reconciliation["received_count"])
        & reconciliation["expected_count"].gt(0)
        & reconciliation["missing_count"].eq(0)
    )
    hash_match = reconciliation["expected_symbol_set_sha256"].eq(reconciliation["received_symbol_set_sha256"])
    hash_columns = [
        "expected_symbol_set_sha256",
        "received_symbol_set_sha256",
        "request_sha256",
        "response_sha256",
    ]
    valid_hashes = reconciliation[hash_columns].map(_is_sha256).all(axis=1)
    if not (count_match & hash_match & valid_hashes).all():
        report.add(
            "reconciliation_mismatch",
            "reconciliation requires positive equal counts, equal symbol hashes, zero missing and valid evidence hashes",
            dataset="universe_reconciliation",
        )
        reconciliation_good = False
    security_partitions = reconciliation["dataset"].eq("security_master")
    expected_status = reconciliation.loc[security_partitions, "partition"].str.rsplit("=", n=1).str[-1]
    if (
        not reconciliation.loc[security_partitions, "list_status"]
        .reset_index(drop=True)
        .eq(expected_status.reset_index(drop=True))
        .all()
    ):
        report.add(
            "reconciliation_status", "security partitions must explicitly bind L/D/P", dataset="universe_reconciliation"
        )
        reconciliation_good = False
    non_security = ~security_partitions
    if reconciliation.loc[non_security, "list_status"].notna().any() or reconciliation["trade_date"].notna().any():
        report.add(
            "reconciliation_scope",
            "aggregate frozen partitions must use null trade_date and non-security list_status",
            dataset="universe_reconciliation",
        )
        reconciliation_good = False
    if reconciliation["request_sha256"].duplicated().any() or reconciliation["response_sha256"].duplicated().any():
        report.add(
            "reconciliation_evidence",
            "request/response evidence hashes must be partition-unique",
            dataset="universe_reconciliation",
        )
        reconciliation_good = False
    gates["universe_reconciliation_semantics"] = reconciliation_good

    states = frames["chan_states"]
    states_good = True
    if not states["regime"].isin(STATE_DOMAIN).all():
        report.add("state_domain", "regime outside 0-10", dataset="chan_states")
        states_good = False
    ordered_states = states.sort_values(["symbol", "dt"]).copy()
    ordered_states["bar_no"] = ordered_states.groupby("symbol").cumcount()
    warmup = ordered_states["bar_no"] < 120
    if not (ordered_states.loc[warmup, "regime"] == 0).all():
        report.add("state_warmup", "the first 120 state rows per symbol must use regime 0", dataset="chan_states")
        states_good = False
    gates["chan_states_semantics"] = states_good

    audit = frames["state_audit"]
    audit_good = True
    for column in ("expected_sha256", "actual_sha256"):
        if not audit[column].map(_is_sha256).all():
            report.add("audit_hash", f"invalid {column}", dataset="state_audit")
            audit_good = False
    count_columns = [
        "prefix_rows",
        "expected_rows",
        "actual_rows",
        "symbol_mismatches",
        "dt_mismatches",
        "regime_mismatches",
    ]
    if (audit[count_columns] < 0).any(axis=None) or (audit["prefix_rows"] == 0).any():
        report.add(
            "audit_counts", "audit row counts must be positive and mismatch counts non-negative", dataset="state_audit"
        )
        audit_good = False
    matched = (
        audit["passed"]
        & audit["expected_rows"].eq(audit["actual_rows"])
        & audit["expected_sha256"].eq(audit["actual_sha256"])
        & audit[["symbol_mismatches", "dt_mismatches", "regime_mismatches"]].eq(0).all(axis=1)
    )
    if not matched.all():
        report.add("state_mismatch", "state audit contains a mismatch", dataset="state_audit")
        audit_good = False
    counts = audit.groupby("symbol").size()
    exact_shape = (
        len(audit) == MIN_STATE_AUDIT_SYMBOLS * MIN_STATE_AUDIT_CUTOFFS
        and len(counts) == MIN_STATE_AUDIT_SYMBOLS
        and counts.eq(MIN_STATE_AUDIT_CUTOFFS).all()
    )
    if not exact_shape:
        report.add(
            "state_audit_coverage",
            f"state audit must be exactly {MIN_STATE_AUDIT_SYMBOLS} symbols x {MIN_STATE_AUDIT_CUTOFFS} cutoffs",
            dataset="state_audit",
        )
        audit_good = False
    prefix_order_good = True
    for _, group in audit.sort_values(["symbol", "checkpoint_dt"]).groupby("symbol", sort=False):
        if not group["prefix_rows"].is_monotonic_increasing or group["prefix_rows"].duplicated().any():
            prefix_order_good = False
            break
    if not prefix_order_good:
        report.add("audit_order", "audit prefix_rows must increase strictly with checkpoint_dt", dataset="state_audit")
        audit_good = False
    gates["state_audit_semantics"] = audit_good
    return gates


def _multi_index(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "ts_code",
    date_column: str = "trade_date",
) -> pd.MultiIndex:
    normalized = frame[[symbol_column, date_column]].rename(columns={symbol_column: "symbol", date_column: "dt"})
    return pd.MultiIndex.from_frame(normalized)


def _coverage_by_intervals(
    points: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> tuple[bool, int]:
    """Return whether every point has exactly one effective interval."""

    misses = 0
    interval_groups = dict(iter(intervals.groupby("ts_code", sort=False)))
    for symbol, group in points.groupby("ts_code", sort=False):
        candidates = interval_groups.get(symbol)
        if candidates is None:
            misses += len(group)
            continue
        dates = group["trade_date"].to_numpy(dtype="datetime64[ns]")
        coverage = np.zeros(len(group), dtype=np.int16)
        for row in candidates.itertuples(index=False):
            begin = np.datetime64(getattr(row, start))
            finish_value = getattr(row, end)
            finish = np.datetime64("2262-04-11") if pd.isna(finish_value) else np.datetime64(finish_value)
            coverage += ((dates >= begin) & (dates <= finish)).astype(np.int16)
        misses += int((coverage != 1).sum())
    return misses == 0, misses


def _namechange_covers_raw_ranges(
    report: ValidationReport,
    raw: pd.DataFrame,
    security_master: pd.DataFrame,
    namechange: pd.DataFrame,
) -> bool:
    """Require gapless effective history whenever a symbol has any name record."""

    known_symbols = set(security_master["ts_code"])
    unknown = sorted(set(namechange["ts_code"]) - known_symbols)
    if unknown:
        report.add("namechange_security", f"namechange contains unknown symbols: {unknown[:5]}", scope="cross")
        return False
    gaps = 0
    raw_ranges = raw.groupby("ts_code")["trade_date"].agg(["min", "max"])
    for symbol, intervals in namechange.groupby("ts_code", sort=False):
        if symbol not in raw_ranges.index:
            continue
        ordered = intervals.sort_values("effective_from")
        coverage_start = raw_ranges.loc[symbol, "min"]
        coverage_end = raw_ranges.loc[symbol, "max"]
        if ordered.iloc[0]["effective_from"] > coverage_start:
            gaps += 1
        for previous, current in zip(ordered.iloc[:-1].itertuples(), ordered.iloc[1:].itertuples(), strict=True):
            if pd.isna(previous.effective_to) or current.effective_from != previous.effective_to + pd.Timedelta(days=1):
                gaps += 1
        final_end = ordered.iloc[-1]["effective_to"]
        if not pd.isna(final_end) and final_end < coverage_end:
            gaps += 1
    if gaps:
        report.add(
            "namechange_gap",
            f"namechange effective history has {gaps} gaps over raw/listing-valid ranges",
            scope="cross",
        )
    return gaps == 0


def _cross_dataset_gates(
    report: ValidationReport,
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    raw = frames["raw_daily"]
    raw_key = _multi_index(raw)
    for name in ("adj_factor", "daily_basic", "stk_limit"):
        other = _multi_index(frames[name])
        missing = len(raw_key.difference(other))
        extra = len(other.difference(raw_key))
        good = missing == 0 and extra == 0
        gates[f"raw_key_equals_{name}"] = good
        if not good:
            report.add(
                "cross_key_coverage",
                f"raw_daily vs {name}: missing={missing}, extra={extra}",
                dataset=name,
                scope="cross",
            )
    state_key = _multi_index(frames["chan_states"], symbol_column="symbol", date_column="dt")
    missing = len(raw_key.difference(state_key))
    extra = len(state_key.difference(raw_key))
    state_coverage_good = missing == 0 and extra == 0
    gates["raw_key_equals_chan_states"] = state_coverage_good
    if not state_coverage_good:
        report.add(
            "cross_key_coverage",
            f"raw_daily vs chan_states: missing={missing}, extra={extra}",
            dataset="chan_states",
            scope="cross",
        )

    calendar_dates = pd.Index(frames["calendar"]["trade_date"])
    bad_calendar = int((~raw["trade_date"].isin(calendar_dates)).sum())
    gates["raw_dates_in_calendar"] = bad_calendar == 0
    if bad_calendar:
        report.add("calendar_coverage", f"{bad_calendar} raw rows are outside calendar", scope="cross")

    master = frames["security_master"].set_index("ts_code", drop=False)
    missing_master = raw.loc[~raw["ts_code"].isin(master.index)]
    master_bad = len(missing_master)
    scoped = raw[raw["ts_code"].isin(master.index)].copy()
    if not scoped.empty:
        aligned = master.loc[scoped["ts_code"]]
        listed = scoped["trade_date"].to_numpy() >= aligned["list_date"].to_numpy()
        delist = aligned["delist_date"].reset_index(drop=True)
        before_delist = delist.isna().to_numpy() | (
            scoped["trade_date"].reset_index(drop=True).to_numpy() < delist.to_numpy()
        )
        master_bad += int((~(listed & before_delist)).sum())
    gates["security_master_covers_raw"] = master_bad == 0
    if master_bad:
        report.add("security_coverage", f"{master_bad} raw rows violate listing intervals", scope="cross")

    namechange_coverage = _namechange_covers_raw_ranges(
        report,
        raw,
        frames["security_master"],
        frames["namechange"],
    )
    gates["namechange_gapless_when_present"] = namechange_coverage

    industry_good, industry_misses = _coverage_by_intervals(
        raw,
        frames["industry_membership"],
        start="effective_from",
        end="effective_to",
    )
    gates["industry_covers_raw"] = industry_good
    if not industry_good:
        report.add("industry_coverage", f"{industry_misses} raw rows lack exactly one PIT industry", scope="cross")

    actions = frames["corporate_actions"]
    unknown_action_symbols = int((~actions["ts_code"].isin(master.index)).sum())
    gates["corporate_actions_in_security_master"] = unknown_action_symbols == 0
    if unknown_action_symbols:
        report.add(
            "corporate_action_security",
            f"{unknown_action_symbols} corporate actions reference an unknown security",
            scope="cross",
        )
    terminal_required = frames["security_master"].loc[
        frames["security_master"]["delist_date"].notna()
        & frames["security_master"]["delist_date"].le(raw["trade_date"].max()),
        ["ts_code", "delist_date"],
    ]
    terminal_actions = actions.loc[
        actions["action_type"].isin({"delist_cash", "delist_share", "delist_writeoff"}),
        ["ts_code", "effective_date"],
    ]
    required_terminal_keys = pd.MultiIndex.from_frame(
        terminal_required.rename(columns={"delist_date": "effective_date"})
    )
    actual_terminal_keys = pd.MultiIndex.from_frame(terminal_actions)
    terminal_missing = len(required_terminal_keys.difference(actual_terminal_keys))
    terminal_counts = terminal_actions.value_counts(["ts_code", "effective_date"])
    terminal_conflicts = int(terminal_counts.gt(1).sum())
    gates["terminal_actions_cover_effective_delistings"] = terminal_missing == 0 and terminal_conflicts == 0
    if terminal_missing:
        report.add(
            "terminal_action_missing",
            f"{terminal_missing} effective delistings lack a terminal corporate action",
            scope="cross",
        )
    if terminal_conflicts:
        report.add(
            "terminal_action_conflict",
            f"{terminal_conflicts} delisting keys have multiple mutually exclusive terminal actions",
            scope="cross",
        )

    reconciliation = frames["universe_reconciliation"].set_index("partition", drop=False)
    physical_bindings: dict[str, tuple[int, str]] = {}
    for status in ("L", "D", "P"):
        symbols = frames["security_master"].loc[frames["security_master"]["list_status"].eq(status), "ts_code"]
        physical_bindings[f"security_master:list_status={status}"] = (
            int(symbols.nunique()),
            symbol_set_sha256(symbols),
        )
    raw_symbols = raw["ts_code"].drop_duplicates()
    raw_binding = (int(raw_symbols.nunique()), symbol_set_sha256(raw_symbols))
    physical_bindings["raw_daily:bundle_symbol_set"] = raw_binding
    physical_bindings["corporate_actions:coverage_universe"] = raw_binding
    reconciliation_mismatches = 0
    for partition, (count, digest) in physical_bindings.items():
        if partition not in reconciliation.index:
            reconciliation_mismatches += 1
            continue
        row = reconciliation.loc[partition]
        if isinstance(row, pd.DataFrame):
            reconciliation_mismatches += len(row)
            continue
        if row["received_count"] != count or row["received_symbol_set_sha256"] != digest:
            reconciliation_mismatches += 1
    gates["reconciliation_binds_physical_symbols"] = reconciliation_mismatches == 0
    if reconciliation_mismatches:
        report.add(
            "reconciliation_physical_binding",
            f"{reconciliation_mismatches} reconciliation partitions differ from physical symbol sets",
            scope="cross",
        )

    dataset_entries = manifest.get("datasets")
    source_binding_good = isinstance(dataset_entries, Mapping)
    if source_binding_good:
        try:
            security_params = dataset_entries["security_master"]["source"]["parameters"]
            raw_params = dataset_entries["raw_daily"]["source"]["parameters"]
            action_params = dataset_entries["corporate_actions"]["source"]["parameters"]
            security_symbols = frames["security_master"]["ts_code"].drop_duplicates()
            security_binding = (int(security_symbols.nunique()), symbol_set_sha256(security_symbols))
            source_binding_good = (
                (security_params["symbol_count"], security_params["symbol_set_sha256"]) == security_binding
                and (raw_params["symbol_count"], raw_params["symbol_set_sha256"]) == raw_binding
                and (
                    action_params["queried_symbol_count"],
                    action_params["queried_symbol_set_sha256"],
                )
                == raw_binding
            )
            action_start = pd.to_datetime(str(action_params["start_date"]), errors="coerce")
            action_end = pd.to_datetime(str(action_params["end_date"]), errors="coerce")
            source_binding_good = bool(
                source_binding_good
                and not pd.isna(action_start)
                and not pd.isna(action_end)
                and action_start <= raw["trade_date"].min()
                and action_end >= raw["trade_date"].max()
            )
        except (KeyError, TypeError):
            source_binding_good = False
    gates["source_symbol_and_action_coverage_binding"] = bool(source_binding_good)
    if not source_binding_good:
        report.add(
            "source_symbol_binding",
            "source metadata does not bind physical security/raw symbols and full-range corporate-action coverage",
            scope="cross",
        )

    audit = frames["state_audit"]
    states = frames["chan_states"]
    checkpoint_key = pd.MultiIndex.from_frame(
        audit[["symbol", "checkpoint_dt"]].rename(columns={"checkpoint_dt": "dt"})
    )
    state_key = _multi_index(states, symbol_column="symbol", date_column="dt")
    audit_missing = len(checkpoint_key.difference(state_key))
    gates["audit_checkpoints_in_states"] = audit_missing == 0
    if audit_missing:
        report.add("audit_checkpoint", f"{audit_missing} audit checkpoints lack a state row", scope="cross")
    prefix_mismatches = audit_missing
    if audit_missing == 0:
        ordered = states.sort_values(["symbol", "dt"])[["symbol", "dt"]].copy()
        ordered["state_prefix_rows"] = ordered.groupby("symbol").cumcount() + 1
        expected_prefix = ordered.set_index(["symbol", "dt"])["state_prefix_rows"].reindex(checkpoint_key)
        actual_prefix = audit["prefix_rows"].reset_index(drop=True)
        prefix_mismatches = int((expected_prefix.reset_index(drop=True) != actual_prefix).sum())
        prefix_mismatches += int((audit["expected_rows"].reset_index(drop=True) != actual_prefix).sum())
    gates["audit_prefix_rows_match_states"] = prefix_mismatches == 0
    if prefix_mismatches:
        report.add(
            "audit_prefix_rows",
            f"{prefix_mismatches} audit prefix row counts differ from state checkpoints",
            scope="cross",
        )
    return gates


def _validate_manifest_cutoff(
    report: ValidationReport,
    manifest: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> bool:
    """Ensure no physical or declared source evidence post-dates the manifest."""

    created = _parse_utc_scalar(manifest.get("created_at_utc"))
    if pd.isna(created):
        return False
    violations: list[str] = []
    datasets = manifest.get("datasets")
    for name, frame in frames.items():
        for column in ("source_asof", "ingested_at"):
            if column in frame and not frame.empty and frame[column].max() > created:
                violations.append(f"{name}.{column}")
        if isinstance(datasets, Mapping):
            entry = datasets.get(name)
            source = entry.get("source") if isinstance(entry, Mapping) else None
            retrieved = _parse_utc_scalar(source.get("retrieved_at_utc")) if isinstance(source, Mapping) else pd.NaT
            if pd.isna(retrieved) or retrieved > created:
                violations.append(f"{name}.retrieved_at_utc")
    if violations:
        report.add(
            "manifest_time_cutoff",
            f"evidence post-dates manifest.created_at_utc or has invalid time: {sorted(violations)}",
            scope="cross",
        )
        return False
    return True


def _validate_state_cache_provenance(
    report: ValidationReport,
    root: Path,
    manifest: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> bool:
    """Bind state/audit bytes, engine components and raw input fingerprints."""

    entry = manifest.get("state_cache_manifest")
    path = _manifest_artifact(report, root, entry, label="state_cache_manifest")
    if path is None or not isinstance(entry, Mapping):
        return False
    good = _check_file_evidence(report, path, entry, label="state_cache_manifest")
    try:
        state_manifest = _load_json(path)
        engine_digest, raw_input_digest = _validate_state_cache_manifest_shape(state_manifest)
    except ContractError as exc:
        report.add("state_cache_provenance", str(exc), scope="state")
        return False
    if entry.get("cache_id") != state_manifest.get("cache_id"):
        report.add("state_cache_id", "bundle entry cache_id differs from state-cache manifest", scope="state")
        good = False

    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        return False
    try:
        state_params = datasets["chan_states"]["source"]["parameters"]
        audit_params = datasets["state_audit"]["source"]["parameters"]
    except (KeyError, TypeError):
        report.add("state_cache_source_binding", "state/audit source metadata is malformed", scope="state")
        return False
    state_manifest_sha256 = sha256_file(path)
    if (
        state_params.get("state_cache_manifest_sha256") != state_manifest_sha256
        or audit_params.get("state_cache_manifest_sha256") != state_manifest_sha256
        or state_params.get("engine_combination_sha256") != engine_digest
        or state_params.get("engine_sha256") != engine_digest
        or state_params.get("raw_input_evidence_sha256") != raw_input_digest
    ):
        report.add(
            "state_cache_source_binding", "state/audit source metadata does not bind provenance hashes", scope="state"
        )
        good = False

    projection = state_manifest.get("projection")
    audit_details = state_manifest.get("state_audit_details")
    if not isinstance(projection, Mapping) or not isinstance(audit_details, Mapping):
        report.add("state_cache_outputs", "state-cache output metadata is missing", scope="state")
        return False
    if (
        projection.get("sha256") != datasets["chan_states"].get("sha256")
        or projection.get("rows") != len(frames["chan_states"])
        or projection.get("columns") != ["symbol", "dt", "regime"]
        or audit_details.get("sha256") != datasets["state_audit"].get("sha256")
        or audit_details.get("rows") != len(frames["state_audit"])
        or audit_details.get("columns") != list(DATASET_SPECS["state_audit"].required_columns)
    ):
        report.add(
            "state_cache_output_binding", "state-cache outputs differ from bundle parquet bytes/schema", scope="state"
        )
        good = False

    records = state_manifest.get("sources")
    if not isinstance(records, list):
        return False
    record_rows = {str(item["symbol"]): int(item["source_rows"]) for item in records if isinstance(item, Mapping)}
    raw_rows = frames["raw_daily"].groupby("ts_code").size().to_dict()
    state_rows = frames["chan_states"].groupby("symbol").size().to_dict()
    if record_rows != raw_rows or record_rows != state_rows:
        report.add(
            "state_cache_raw_binding",
            "state-cache source symbol/row evidence differs from raw_daily or chan_states",
            scope="state",
        )
        good = False
    return good


def _validate_protocol(
    report: ValidationReport,
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    entry = manifest.get("protocol")
    path = _manifest_artifact(report, root, entry, label="protocol")
    if path is None or not isinstance(entry, Mapping):
        return False, None
    good = _check_file_evidence(report, path, entry, label="protocol")
    try:
        expected_hash = frozen_protocol_sha256()
    except ContractError as exc:
        report.add("protocol_repository", str(exc), scope="protocol")
        return False, None
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash or entry.get("sha256") != expected_hash:
        report.add(
            "protocol_repository_hash",
            "bundle protocol must byte-for-byte match scripts/xs_chan_protocol_v2.json",
            scope="protocol",
        )
        good = False
    try:
        protocol = _load_json(path)
    except ContractError as exc:
        report.add("protocol_json", str(exc), scope="protocol")
        return False, None
    forbidden = RESERVED_READINESS_KEYS.intersection(protocol)
    if forbidden:
        report.add("manual_readiness", f"protocol contains readiness fields: {sorted(forbidden)}", scope="protocol")
        good = False
    if entry.get("protocol_id") != protocol.get("protocol_id"):
        report.add("protocol_id", "manifest protocol_id differs from protocol", scope="protocol")
        good = False
    if entry.get("frozen_at_utc") != protocol.get("frozen_at_utc"):
        report.add("protocol_freeze", "manifest frozen_at differs from protocol", scope="protocol")
        good = False
    frozen = _parse_utc_scalar(protocol.get("frozen_at_utc"))
    if pd.isna(frozen):
        report.add("protocol_freeze", "protocol frozen_at_utc is invalid", scope="protocol")
        good = False
    if protocol.get("confirmation_mode") != "append_only_forward_only":
        report.add("protocol_mode", "confirmation_mode must be append_only_forward_only", scope="protocol")
        good = False
    contract = protocol.get("data_contract")
    if not isinstance(contract, Mapping):
        report.add("protocol_contract", "protocol data_contract is missing", scope="protocol")
        good = False
    else:
        if int(contract.get("manifest_version", -1)) != MANIFEST_VERSION:
            report.add("protocol_contract", "protocol manifest_version mismatch", scope="protocol")
            good = False
        if set(contract.get("required_artifacts", [])) != set(REQUIRED_DATASETS):
            report.add("protocol_contract", "protocol required_artifacts mismatch", scope="protocol")
            good = False
        if contract.get("corporate_action_types") != list(CORPORATE_ACTION_TYPES):
            report.add("protocol_contract", "protocol corporate action types mismatch", scope="protocol")
            good = False
        if contract.get("universe_reconciliation_partitions") != list(RECONCILIATION_PARTITIONS):
            report.add("protocol_contract", "protocol universe reconciliation partitions mismatch", scope="protocol")
            good = False
        if int(contract.get("minimum_state_prefix_symbols", -1)) < MIN_STATE_AUDIT_SYMBOLS:
            report.add("protocol_contract", "protocol state symbol audit threshold too small", scope="protocol")
            good = False
        if int(contract.get("minimum_state_prefix_cutoffs", -1)) < MIN_STATE_AUDIT_CUTOFFS:
            report.add("protocol_contract", "protocol state cutoff audit threshold too small", scope="protocol")
            good = False
        if int(contract.get("required_state_prefix_mismatches", -1)) != 0:
            report.add("protocol_contract", "protocol must require zero state mismatches", scope="protocol")
            good = False
    forward = protocol.get("forward_validation")
    if not isinstance(forward, Mapping):
        report.add("protocol_forward", "protocol forward_validation is missing", scope="protocol")
        good = False
    else:
        if int(forward.get("minimum_complete_weeks", -1)) < MIN_CONFIRMATORY_WEEKS:
            report.add("protocol_forward", "minimum complete weeks is below 52", scope="protocol")
            good = False
        if forward.get("required_engineering_gates") != list(REQUIRED_ENGINEERING_GATES):
            report.add(
                "protocol_forward", "required engineering gates differ from the frozen contract", scope="protocol"
            )
            good = False
        if forward.get("dependency_sha256_keys") != list(FORWARD_DEPENDENCY_KEYS):
            report.add("protocol_forward", "dependency closure differs from the frozen contract", scope="protocol")
            good = False
        if forward.get("local_ledger_scope") != "structural_integrity_only_never_self_sets_confirmatory_oos":
            report.add("protocol_forward", "local ledger scope differs from the frozen contract", scope="protocol")
            good = False
        for key in (
            "append_only_hash_chain",
            "decision_recorded_before_exec_session",
            "execution_recorded_before_exec_session_close",
            "historical_backfill_forbidden",
            "protocol_or_dependency_change_requires_new_chain",
        ):
            if forward.get(key) is not True:
                report.add("protocol_forward", f"protocol must enable {key}", scope="protocol")
                good = False
    return good, protocol


def _validate_external_ledger(
    report: ValidationReport,
    forward_chain_path: str | Path | None,
    manifest: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Delegate chain verification and bind its immutable freeze to this bundle."""

    evidence: dict[str, Any] = {
        "present": False,
        "valid": False,
        "completed_weeks": 0,
        "minimum_complete_weeks": MIN_CONFIRMATORY_WEEKS,
    }
    if forward_chain_path is None:
        return False, evidence
    evidence["present"] = True

    raw_path = Path(forward_chain_path).expanduser()
    if raw_path.is_symlink():
        report.add("forward_path", "external ledger root may not be a symlink", scope="forward")
        return False, evidence
    try:
        path = raw_path.resolve(strict=True)
    except OSError as exc:
        report.add("forward_path", f"cannot resolve external ledger: {exc}", scope="forward")
        return False, evidence
    if not path.is_dir():
        report.add("forward_path", "external ledger path is not a directory", scope="forward")
        return False, evidence
    if any(item.is_symlink() for item in path.iterdir()):
        report.add("forward_path", "external ledger may not contain symlinks", scope="forward")
        return False, evidence

    try:
        import xs_chan_oos_ledger

        verified = xs_chan_oos_ledger.verify_ledger(path)
    except Exception as exc:
        report.add("forward_ledger", f"OOS ledger verification failed: {exc}", scope="forward")
        return False, evidence
    if not isinstance(verified, Mapping) or verified.get("valid") is not True:
        report.add("forward_ledger", "OOS ledger verifier did not return valid evidence", scope="forward")
        return False, evidence

    completed = verified.get("completed_weeks")
    minimum = verified.get("min_complete_weeks")
    dependencies = verified.get("dependency_sha256")
    if not isinstance(completed, int) or completed < 0 or minimum != MIN_CONFIRMATORY_WEEKS:
        report.add("forward_maturity", "OOS ledger maturity fields violate the frozen 52-week rule", scope="forward")
        return False, evidence
    evidence.update(
        {
            "completed_weeks": completed,
            "minimum_complete_weeks": minimum,
            "chain_head_sha256": verified.get("chain_head_sha256"),
            "confirmation_window": verified.get("confirmation_window"),
        }
    )

    expected_protocol = manifest.get("protocol", {}).get("sha256")
    if verified.get("protocol_sha256") != expected_protocol:
        report.add("forward_protocol", "OOS ledger freeze protocol_sha256 differs from the bundle", scope="forward")
        return False, evidence
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != xs_chan_oos_ledger.EXPECTED_DEPENDENCY_KEYS
        or dependencies.get("data_evidence") != manifest.get("data_evidence_sha256")
    ):
        report.add(
            "forward_dependency",
            "OOS ledger freeze dependency_sha256.data_evidence differs from the bundle",
            scope="forward",
        )
        return False, evidence
    structural_complete = completed >= minimum
    if verified.get("structural_window_complete") is not structural_complete:
        report.add(
            "forward_maturity", "OOS ledger structural-window flag is inconsistent with verified weeks", scope="forward"
        )
        return False, evidence
    window = verified.get("confirmation_window")
    if completed >= minimum and (
        not isinstance(window, Mapping)
        or not _is_sha256(window.get("identity_sha256"))
        or not _is_sha256(window.get("confirmation_head_sha256"))
        or len(window.get("record_hashes", [])) != minimum * 2
    ):
        report.add("forward_window", "OOS ledger lacks the exact first-52-week identity", scope="forward")
        return False, evidence
    if verified.get("semantic_replay_verified") is not False or verified.get("confirmatory_oos") is not False:
        report.add(
            "forward_semantics", "local ledger cannot self-assert semantic replay or confirmatory OOS", scope="forward"
        )
        return False, evidence
    evidence["valid"] = True
    evidence["structural_window_complete"] = structural_complete
    evidence["semantic_replay_verified"] = False
    return False, evidence


def validate_bundle(
    bundle_root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    forward_chain_path: str | Path | None = None,
) -> ValidationReport:
    """Validate a local V2 bundle and derive readiness without overrides."""

    report = ValidationReport()
    try:
        root = Path(bundle_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ContractError(f"bundle root is not a directory: {root}")
        manifest_file = _resolve_bundle_argument(root, manifest_path, DEFAULT_MANIFEST)
        manifest_rel = _relative_to_bundle(root, manifest_file)
        if PurePosixPath(manifest_rel).parent != PurePosixPath("."):
            raise ContractError("manifest must be a direct child of bundle root")
        manifest = _load_json(manifest_file)
    except (ContractError, OSError) as exc:
        report.add("manifest_load", str(exc), scope="manifest")
        report.gates = {
            "manifest_integrity": False,
            "protocol_frozen": False,
            "forward_chain_complete": False,
        }
        return report

    manifest_good = _validate_manifest_header(report, manifest)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        report.add("datasets", "manifest datasets must be an object", scope="manifest")
        datasets = {}
        manifest_good = False
    extra = sorted(set(datasets) - set(REQUIRED_DATASETS))
    missing = sorted(set(REQUIRED_DATASETS) - set(datasets))
    if extra or missing:
        report.add("dataset_set", f"required dataset mismatch; missing={missing}, extra={extra}", scope="manifest")
        manifest_good = False

    source_entry = manifest.get("source_metadata")
    source_path = _manifest_artifact(report, root, source_entry, label="source_metadata")
    if source_path is None or not isinstance(source_entry, Mapping):
        source_good = False
    else:
        source_good = _check_file_evidence(report, source_path, source_entry, label="source_metadata")
        try:
            source_payload = _load_json(source_path)
            _validate_source_metadata_shape(source_payload)
            if any(datasets.get(name, {}).get("source") != source_payload.get(name) for name in REQUIRED_DATASETS):
                report.add(
                    "source_binding", "dataset source entries differ from source_metadata.json", scope="manifest"
                )
                source_good = False
        except ContractError as exc:
            report.add("source_metadata", str(exc), scope="manifest")
            source_good = False
    manifest_good &= source_good

    frames: dict[str, pd.DataFrame] = {}
    dataset_gates: dict[str, bool] = {}
    for name in REQUIRED_DATASETS:
        if name not in datasets:
            dataset_gates[f"dataset_{name}"] = False
            continue
        try:
            good, frame = _validate_dataset(report, root, name, datasets[name])
        except (ContractError, KeyError, TypeError, ValueError, OverflowError) as exc:
            report.add("dataset_validation", f"dataset validation failed closed: {exc}", dataset=name)
            good, frame = False, None
        dataset_gates[f"dataset_{name}"] = good
        if frame is not None and set(DATASET_SPECS[name].required_columns).issubset(frame.columns):
            frames[name] = frame

    semantic_gates: dict[str, bool] = {}
    cross_gates: dict[str, bool] = {}
    if set(frames) == set(REQUIRED_DATASETS):
        semantic_gates = _dataset_semantics(report, frames)
        cross_gates = _cross_dataset_gates(report, frames, manifest)
        cross_gates["manifest_time_cutoff"] = _validate_manifest_cutoff(report, manifest, frames)
        cross_gates["state_cache_provenance"] = _validate_state_cache_provenance(
            report,
            root,
            manifest,
            frames,
        )
        cross_gates["state_recompute_engineering"] = False
        report.add(
            "state_recompute_unavailable",
            "independent state-engine replay is unavailable: bundle lacks replayable raw state-cache inputs and "
            "the local native extension cannot be trusted as the frozen engine",
            scope="engineering",
        )
        for gate in REQUIRED_ENGINEERING_GATES[1:]:
            cross_gates[gate] = False
        report.add(
            "formal_replay_stack_unavailable",
            "source archives, suspension-vs-gap reconciliation, calendar/ledger binding, feature-prefix verification, "
            "execution/corporate-action/statistics replay, semantic artifact verification, and external timestamp "
            "anchoring are not implemented",
            scope="engineering",
        )
    else:
        semantic_gates["all_frames_readable"] = False
        cross_gates["cross_dataset_coverage"] = False
        cross_gates["manifest_time_cutoff"] = False
        cross_gates["state_cache_provenance"] = False
        cross_gates["state_recompute_engineering"] = False
        for gate in REQUIRED_ENGINEERING_GATES[1:]:
            cross_gates[gate] = False

    protocol_good, _ = _validate_protocol(report, root, manifest)
    forward_good, forward_evidence = _validate_external_ledger(report, forward_chain_path, manifest)
    report.evidence["forward_chain"] = forward_evidence
    if frames:
        report.evidence["row_counts"] = {name: len(frame) for name, frame in sorted(frames.items())}
        if "raw_daily" in frames:
            report.evidence["raw_daily_range"] = {
                "min": frames["raw_daily"]["trade_date"].min().strftime("%Y-%m-%d"),
                "max": frames["raw_daily"]["trade_date"].max().strftime("%Y-%m-%d"),
                "symbols": int(frames["raw_daily"]["ts_code"].nunique()),
            }
    report.gates = {
        "manifest_integrity": manifest_good,
        **dataset_gates,
        **semantic_gates,
        **cross_gates,
        "protocol_frozen": protocol_good,
        "forward_chain_complete": forward_good,
    }
    start_gate_names = [name for name in report.gates if name != "forward_chain_complete"]
    report.forward_start_allowed = bool(start_gate_names and all(report.gates[name] for name in start_gate_names))
    report.confirmatory_oos = bool(report.forward_start_allowed and forward_good)
    report.valid = bool(report.forward_start_allowed and (forward_chain_path is None or forward_good))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="hash local artifacts and write manifest.json")
    build.add_argument("--bundle-root", type=Path, required=True)
    build.add_argument("--output", type=Path)
    build.add_argument("--source-metadata", type=Path)
    build.add_argument("--protocol", type=Path)
    build.add_argument("--state-cache-manifest", type=Path)

    validate = subparsers.add_parser("validate", help="validate a local bundle")
    validate.add_argument("--bundle-root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument(
        "--forward-chain",
        "--ledger",
        dest="forward_chain",
        type=Path,
        help="external xs_chan_oos_ledger directory; never embedded in the data bundle",
    )
    validate.add_argument(
        "--require",
        choices=("structure", "forward-start", "confirmatory-oos"),
        default="forward-start",
        help="exit-code gate; default is fail-closed forward-start readiness",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-manifest":
        try:
            manifest = build_manifest(
                args.bundle_root,
                output_path=args.output,
                source_metadata_path=args.source_metadata,
                protocol_path=args.protocol,
                state_cache_manifest_path=args.state_cache_manifest,
            )
        except ContractError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"ok": True, "manifest_sha256": manifest["manifest_sha256"]}, ensure_ascii=False))
        return 0

    report = validate_bundle(args.bundle_root, args.manifest, forward_chain_path=args.forward_chain)
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    if args.require == "structure":
        return 0 if report.valid else 1
    if args.require == "forward-start":
        return 0 if report.forward_start_allowed else 1
    return 0 if report.confirmatory_oos else 1


if __name__ == "__main__":
    raise SystemExit(main())
