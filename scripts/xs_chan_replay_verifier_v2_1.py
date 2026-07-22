"""Independent semantic authority boundary for the XS-Chan V2.1 trial.

The verifier never accepts readiness, portfolio metrics, comparison statistics,
or a confirmatory flag as caller-supplied truth.  It opens a fixed-layout
content-addressed object store, validates the authoritative data snapshot and
primary ledger, replays every portfolio/scenario, reconstructs the exact first
52 cycle paths, and recomputes all seven preregistered comparisons.

External data and timestamp/registry services are intentionally not emulated.
Without their real evidence this module returns a closed formal status.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import xs_chan_data_v2_1 as data_v21
import xs_chan_execution_v2_1 as execution_v21
import xs_chan_oos_ledger_v2_1 as ledger_v21
import xs_chan_statistics_v2_1 as statistics_v21

REPLAY_MANIFEST_SCHEMA = "xs_chan_semantic_replay_manifest_v2_1"
REPLAY_MANIFEST_VERSION = 2
FORMAL_EVIDENCE_SCHEMA = "xs_chan_formal_evidence_v2_1"
READINESS_REPORT_SCHEMA = ledger_v21.READINESS_COMMITMENT_SCHEMA
DATA_CHAIN_VERIFICATION_SCHEMA = "xs_chan_data_chain_verification_v2_1"
PROTOCOL_ID = "xs_chan_pilot_v2_1_preregistered_20260720"
JSON_MEDIA_TYPE = "application/vnd.xs-chan.canonical-json;version=2.1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OBJECT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{2,191}$")
EXPECTED_DEPENDENCIES = (
    "data_evidence",
    "state_engine",
    "feature_engine",
    "research_engine",
    "execution_engine",
    "ledger_engine",
    "statistics_engine",
    "replay_verifier",
)
ENGINEERING_GATES = (
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
)
FORMAL_EVALUATION_STATUSES = (
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
)
SCENARIO_IDS = ("gross", "1x", "2x", "capacity_1x")
SINGLE_FAMILIES = ("F", "FC", "FMA")
SEEDED_FAMILIES = ("R_match", "FGR", "FMGR")
CONFIRMATORY_ARM_SOURCE = {
    "F_2x": ("F", None, "2x"),
    "R_match_2x": ("R_match", "seed", "2x"),
    "FC_gross": ("FC", None, "gross"),
    "FGR_gross": ("FGR", "seed", "gross"),
    "FC_2x": ("FC", None, "2x"),
    "FGR_2x": ("FGR", "seed", "2x"),
    "FMA_gross": ("FMA", None, "gross"),
    "FMA_2x": ("FMA", None, "2x"),
    "FMGR_gross": ("FMGR", "seed", "gross"),
    "FMGR_2x": ("FMGR", "seed", "2x"),
}


class ReplayVerificationError(ValueError):
    """Raised when an authority-bound replay artifact cannot be verified."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ReplayVerificationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplayVerificationError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReplayVerificationError(f"{label} contains non-finite JSON token {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayVerificationError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayVerificationError(f"{label} must be a JSON object")
    return value


def _read_regular_once(path: Path, *, expected_size: int | None = None) -> bytes:
    """Open one immutable object without following symlinks and read it once."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayVerificationError(f"cannot securely open {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayVerificationError(f"content object is not a regular file: {path}")
        if expected_size is not None and metadata.st_size != expected_size:
            raise ReplayVerificationError(f"content object size changed or differs: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ReplayVerificationError(f"short read from content object: {path}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ReplayVerificationError(f"content object grew while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_authoritative_protocol(path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path) if path is not None else Path(__file__).resolve().with_name("xs_chan_protocol_v2_1.json")
    protocol = _strict_json(_read_regular_once(protocol_path), "V2.1 protocol")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ReplayVerificationError("authoritative protocol_id differs from V2.1")
    digest = object_sha256(protocol)
    if protocol.get("canonicalization") != (
        "utf8_json_sort_keys_true_separators_comma_colon_ensure_ascii_false_allow_nan_false"
    ):
        raise ReplayVerificationError("protocol canonicalization declaration differs")
    return protocol, digest


@dataclass(frozen=True)
class FrozenReplayBundle:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    objects: dict[str, bytes]

    def read_json(self, name: str) -> dict[str, Any]:
        if name not in self.objects:
            raise ReplayVerificationError(f"bundle object {name!r} is missing")
        return _strict_json(self.objects[name], f"bundle object {name}")


@dataclass(frozen=True)
class VerifiedForwardDataChain:
    """Verified append-only data history resolved from ledger transitions."""

    genesis_snapshot: data_v21.VerifiedDataSnapshot
    final_snapshot: data_v21.VerifiedDataSnapshot
    snapshots_by_sha256: dict[str, data_v21.VerifiedDataSnapshot]
    decision_transitions: dict[str, dict[str, Any]]
    unwind_transition: dict[str, Any]
    unwind_eod_transitions: dict[str, dict[str, Any]]
    verification: dict[str, Any]


def _parse_aware_utc(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayVerificationError(f"{label} must be timezone-aware ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplayVerificationError(f"{label} must be timezone-aware ISO-8601")
    return parsed.astimezone(UTC)


def _require_snapshot_observation(
    snapshot: data_v21.VerifiedDataSnapshot,
    expected_session: Any,
    label: str,
) -> str:
    expected = str(expected_session)
    try:
        canonical = datetime.strptime(expected, "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ReplayVerificationError(f"{label} expected session is not canonical YYYY-MM-DD") from exc
    if expected != canonical or snapshot.observation_through_session != canonical:
        raise ReplayVerificationError(f"{label} observation_through_session differs from its causal ledger session")
    return canonical


def verify_readiness_report(
    value: Mapping[str, Any],
    *,
    expected_protocol_sha256: str,
    expected_data_contract_identity_sha256: str,
    expected_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the exact, self-addressed pre-candidate readiness report.

    A readiness report is evidence, not a boolean override.  Its 13 gate
    identities, static data contract, and dependency closure are later compared
    byte-for-byte with the values committed by ledger genesis through the
    confirmation window.  Candidate-close data is intentionally excluded: its
    snapshot cannot exist until after this report and is separately bound and
    externally anchored by Genesis.
    """

    exact = {
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
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ReplayVerificationError("readiness report fields differ from the exact V2.1 schema")
    report = dict(value)
    if report["schema"] != READINESS_REPORT_SCHEMA or report["protocol_id"] != PROTOCOL_ID:
        raise ReplayVerificationError("readiness report schema/protocol differs from V2.1")
    protocol_sha = _require_sha256(report["protocol_sha256"], "readiness.protocol_sha256")
    if protocol_sha != expected_protocol_sha256 or report["trial_id"] != protocol_sha:
        raise ReplayVerificationError("readiness report has the wrong protocol/trial identity")
    data_contract_sha = _require_sha256(
        report["data_contract_identity_sha256"],
        "readiness.data_contract_identity_sha256",
    )
    if data_contract_sha != expected_data_contract_identity_sha256:
        raise ReplayVerificationError("readiness report has the wrong static data contract identity")
    created = _parse_aware_utc(report["created_at_utc"], "readiness.created_at_utc")
    if created > datetime.now(UTC):
        raise ReplayVerificationError("readiness report creation time is in the future")

    dependencies = report["dependency_sha256"]
    if not isinstance(dependencies, Mapping) or tuple(sorted(dependencies)) != tuple(sorted(EXPECTED_DEPENDENCIES)):
        raise ReplayVerificationError("readiness dependency closure differs from V2.1")
    normalized_dependencies = {
        name: _require_sha256(dependencies[name], f"readiness.dependency_sha256.{name}")
        for name in EXPECTED_DEPENDENCIES
    }
    if normalized_dependencies != dict(expected_dependencies):
        raise ReplayVerificationError("readiness dependency closure differs from the executing verifier")

    gates = report["engineering_gates"]
    if not isinstance(gates, Mapping) or tuple(sorted(gates)) != tuple(sorted(ENGINEERING_GATES)):
        raise ReplayVerificationError("readiness report must contain exactly the 13 V2.1 engineering gates")
    normalized_gates: dict[str, dict[str, Any]] = {}
    for gate in ENGINEERING_GATES:
        item = gates[gate]
        if not isinstance(item, Mapping) or set(item) != {"status", "evidence_sha256"}:
            raise ReplayVerificationError(f"readiness gate {gate} has incorrect fields")
        if item["status"] != "PASSED":
            raise ReplayVerificationError(f"readiness gate {gate} is not a PASSED pre-candidate commitment")
        normalized_gates[gate] = {
            "status": "PASSED",
            "evidence_sha256": _require_sha256(
                item["evidence_sha256"], f"readiness.engineering_gates.{gate}.evidence_sha256"
            ),
        }

    supplied_digest = _require_sha256(report["report_sha256"], "readiness.report_sha256")
    body = dict(report)
    body.pop("report_sha256")
    if object_sha256(body) != supplied_digest:
        raise ReplayVerificationError("readiness report self-digest differs")
    return {
        **report,
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "dependency_sha256": normalized_dependencies,
        "engineering_gates": normalized_gates,
    }


def verify_readiness_anchor(
    receipt: Mapping[str, Any],
    *,
    readiness_report: Mapping[str, Any],
    receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
) -> dict[str, Any]:
    """Cryptographically establish when the pre-candidate commitment existed."""

    if not isinstance(receipt_verifier, ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier):
        raise ReplayVerificationError("a concrete readiness timestamp receipt verifier is required")
    report_sha = _require_sha256(readiness_report.get("report_sha256"), "readiness.report_sha256")
    try:
        verified = receipt_verifier.verify(receipt, expected_subject_sha256=report_sha)
    except Exception as exc:
        raise ReplayVerificationError(f"readiness timestamp receipt is invalid: {exc}") from exc
    created = _parse_aware_utc(readiness_report.get("created_at_utc"), "readiness.created_at_utc")
    if created > verified.issued_at_utc:
        raise ReplayVerificationError("readiness report creation follows its external timestamp")
    if verified.issued_at_utc > datetime.now(UTC):
        raise ReplayVerificationError("readiness timestamp receipt was issued in the future")
    return {
        "report_sha256": report_sha,
        "issued_at_utc": verified.issued_at_utc.isoformat().replace("+00:00", "Z"),
        "provider": verified.provider,
        "key_id": verified.key_id,
        "receipt_sha256": verified.receipt_sha256,
    }


def _require_verified_planner(value: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the 52-cycle planner diagnostic into a strict fail-closed gate."""

    exact = {
        "schema",
        "valid",
        "batch_input_sha256",
        "batch_artifact_sha256",
        "cycle_count",
        "execution_decision_count",
        "cycle_verification_sha256",
        "errors",
        "verification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ReplayVerificationError("planner verification result has an inexact schema")
    verified = dict(value)
    if verified["schema"] != "xs_chan_planning_batch_verification_v2_1":
        raise ReplayVerificationError("planner verification result has the wrong schema")
    supplied = _require_sha256(verified["verification_sha256"], "planner.verification_sha256")
    body = dict(verified)
    body.pop("verification_sha256")
    if object_sha256(body) != supplied:
        raise ReplayVerificationError("planner verification result self-digest differs")
    for field in ("batch_input_sha256", "batch_artifact_sha256"):
        _require_sha256(verified[field], f"planner.{field}")
    if verified["valid"] is not True or verified["errors"] != []:
        raise ReplayVerificationError("planner semantic replay did not return valid=true with no errors")
    cycle_hashes = verified["cycle_verification_sha256"]
    if not isinstance(cycle_hashes, list) or len(cycle_hashes) != 52:
        raise ReplayVerificationError("planner verification must bind exactly 52 cycle verifications")
    for index, digest in enumerate(cycle_hashes):
        _require_sha256(digest, f"planner.cycle_verification_sha256[{index}]")
    if len(set(cycle_hashes)) != 52:
        raise ReplayVerificationError("planner cycle verification identities must be unique")
    if verified["cycle_count"] != 52 or verified["execution_decision_count"] != 52 * 252:
        raise ReplayVerificationError("planner must emit exactly 52 cycles and 13,104 execution decisions")
    return verified


def _require_state_recompute(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require all-symbol projection replay plus the fixed 100×20 prefix audit."""

    exact = {
        "schema",
        "valid",
        "source_symbol_count",
        "recomputed_symbol_count",
        "required_prefix_audit_symbol_count",
        "prefix_audit_symbol_count",
        "cutoffs_per_prefix_audit_symbol",
        "mismatch_count",
        "evidence_sha256",
        "errors",
        "verification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ReplayVerificationError("state recompute verification has an inexact schema")
    verified = dict(value)
    if verified["schema"] != "xs_chan_state_recompute_verification_v2_1":
        raise ReplayVerificationError("state recompute verification has the wrong schema")
    supplied = _require_sha256(verified["verification_sha256"], "state_recompute.verification_sha256")
    body = dict(verified)
    body.pop("verification_sha256")
    expected = data_v21.sha256_bytes(data_v21.canonical_json_bytes(body))
    if supplied != expected:
        raise ReplayVerificationError("state recompute verification self-digest differs")
    _require_sha256(verified["evidence_sha256"], "state_recompute.evidence_sha256")
    if (
        verified["valid"] is not True
        or type(verified["source_symbol_count"]) is not int
        or verified["source_symbol_count"] <= 0
        or verified["recomputed_symbol_count"] != verified["source_symbol_count"]
        or verified["required_prefix_audit_symbol_count"] != 100
        or verified["prefix_audit_symbol_count"] != 100
        or verified["cutoffs_per_prefix_audit_symbol"] != 20
        or verified["mismatch_count"] != 0
        or verified["errors"] != []
    ):
        raise ReplayVerificationError(
            "state recompute did not prove all-symbol projections plus exact 100×20 zero-mismatch audit"
        )
    return verified


def _require_execution_market_binding(value: Mapping[str, Any], *, expected_input_count: int) -> dict[str, Any]:
    """Require exact snapshot-to-execution market provenance for every book."""

    exact = {
        "schema",
        "valid",
        "data_snapshot_sha256",
        "execution_input_count",
        "official_session_count",
        "bound_open_row_count",
        "bound_eod_row_count",
        "bound_corporate_action_count",
        "evidence_sha256",
        "errors",
        "verification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise ReplayVerificationError("execution market binding has an inexact schema")
    verified = dict(value)
    if verified["schema"] != data_v21.EXECUTION_MARKET_BINDING_SCHEMA:
        raise ReplayVerificationError("execution market binding has the wrong schema")
    supplied = _require_sha256(verified["verification_sha256"], "market_binding.verification_sha256")
    body = dict(verified)
    body.pop("verification_sha256")
    if data_v21.sha256_bytes(data_v21.canonical_json_bytes(body)) != supplied:
        raise ReplayVerificationError("execution market binding self-digest differs")
    _require_sha256(verified["data_snapshot_sha256"], "market_binding.data_snapshot_sha256")
    _require_sha256(verified["evidence_sha256"], "market_binding.evidence_sha256")
    if (
        verified["valid"] is not True
        or verified["errors"] != []
        or verified["execution_input_count"] != expected_input_count
        or verified["official_session_count"] <= 0
        or verified["bound_open_row_count"] <= 0
        or verified["bound_eod_row_count"] <= 0
    ):
        raise ReplayVerificationError("execution market inputs were not completely snapshot-bound")
    return verified


def _object_path(root: Path, digest: str) -> Path:
    return root / "objects" / digest[:2] / digest


def load_replay_bundle(manifest_path: str | Path) -> FrozenReplayBundle:
    """Load a fixed-layout content-addressed replay bundle without loose paths."""

    path = Path(manifest_path).resolve()
    raw_manifest = _read_regular_once(path)
    manifest = _strict_json(raw_manifest, "replay manifest")
    exact = {
        "schema",
        "manifest_version",
        "protocol_id",
        "protocol_sha256",
        "trial_id",
        "data_contract_identity_sha256",
        "genesis_data_manifest_sha256",
        "final_data_manifest_sha256",
        "final_data_chain_head_sha256",
        "primary_chain_id",
        "confirmation_window_identity_sha256",
        "created_at_utc",
        "dependency_sha256",
        "objects",
        "execution_pairs",
        "planner_input_object",
        "planner_result_object",
        "statistics_input_object",
        "statistics_result_object",
        "readiness_report_object",
        "readiness_anchor_receipt_object",
    }
    if set(manifest) != exact:
        raise ReplayVerificationError("replay manifest fields differ from the exact V2.1 schema")
    if manifest["schema"] != REPLAY_MANIFEST_SCHEMA or manifest["manifest_version"] != REPLAY_MANIFEST_VERSION:
        raise ReplayVerificationError("replay manifest schema/version differs")
    protocol, protocol_sha = load_authoritative_protocol()
    del protocol
    if manifest["protocol_id"] != PROTOCOL_ID or manifest["protocol_sha256"] != protocol_sha:
        raise ReplayVerificationError("replay manifest does not bind the canonical V2.1 protocol")
    if manifest["trial_id"] != protocol_sha:
        raise ReplayVerificationError("trial_id must equal the canonical protocol SHA-256")
    for field in (
        "data_contract_identity_sha256",
        "genesis_data_manifest_sha256",
        "final_data_manifest_sha256",
        "final_data_chain_head_sha256",
        "primary_chain_id",
        "confirmation_window_identity_sha256",
    ):
        _require_sha256(manifest[field], field)
    if manifest["data_contract_identity_sha256"] != data_v21.data_contract_identity_sha256():
        raise ReplayVerificationError("replay manifest changes the static V2.1 data contract identity")
    created = _parse_aware_utc(manifest["created_at_utc"], "created_at_utc")
    if created > datetime.now(UTC):
        raise ReplayVerificationError("created_at_utc is in the future")
    dependencies = manifest["dependency_sha256"]
    if not isinstance(dependencies, Mapping) or tuple(sorted(dependencies)) != tuple(sorted(EXPECTED_DEPENDENCIES)):
        raise ReplayVerificationError("dependency_sha256 must have the exact V2.1 engine closure")
    for name in EXPECTED_DEPENDENCIES:
        _require_sha256(dependencies[name], f"dependency_sha256.{name}")

    objects = manifest["objects"]
    if not isinstance(objects, Mapping) or not objects:
        raise ReplayVerificationError("replay objects must be a non-empty mapping")
    blobs: dict[str, bytes] = {}
    descriptor_keys = {"sha256", "size_bytes", "media_type"}
    for name, raw_descriptor in objects.items():
        if not isinstance(name, str) or OBJECT_NAME_PATTERN.fullmatch(name) is None:
            raise ReplayVerificationError(f"invalid replay object name {name!r}")
        if not isinstance(raw_descriptor, Mapping) or set(raw_descriptor) != descriptor_keys:
            raise ReplayVerificationError(f"object descriptor {name!r} has incorrect fields")
        digest = _require_sha256(raw_descriptor["sha256"], f"objects.{name}.sha256")
        size = raw_descriptor["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReplayVerificationError(f"objects.{name}.size_bytes is invalid")
        if raw_descriptor["media_type"] != JSON_MEDIA_TYPE:
            raise ReplayVerificationError(f"objects.{name}.media_type is not registered canonical JSON")
        blob = _read_regular_once(_object_path(path.parent, digest), expected_size=size)
        if bytes_sha256(blob) != digest:
            raise ReplayVerificationError(f"object {name!r} content digest differs")
        parsed = _strict_json(blob, f"object {name}")
        if blob != canonical_json_bytes(parsed):
            raise ReplayVerificationError(f"object {name!r} is not canonically encoded")
        blobs[name] = blob

    role_references: dict[str, Any] = {
        "planner_input": manifest["planner_input_object"],
        "planner_result": manifest["planner_result_object"],
        "statistics_input": manifest["statistics_input_object"],
        "statistics_result": manifest["statistics_result_object"],
        "readiness_report": manifest["readiness_report_object"],
        "readiness_anchor_receipt": manifest["readiness_anchor_receipt_object"],
    }
    if any(not isinstance(name, str) for name in role_references.values()):
        raise ReplayVerificationError("replay scalar object references must be strings")
    pairs = manifest["execution_pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ReplayVerificationError("execution_pairs must be a non-empty list")
    pair_keys = {
        "pair_id",
        "family_id",
        "seed_id",
        "scenario_id",
        "statistics_arm_id",
        "input_object",
        "result_object",
    }
    pair_ids: set[str] = set()
    pair_references: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or set(pair) != pair_keys:
            raise ReplayVerificationError(f"execution_pairs[{index}] fields differ")
        pair_id = str(pair["pair_id"])
        if OBJECT_NAME_PATTERN.fullmatch(pair_id) is None or pair_id in pair_ids:
            raise ReplayVerificationError("execution pair_id is invalid or duplicated")
        pair_ids.add(pair_id)
        if pair["family_id"] not in SINGLE_FAMILIES + SEEDED_FAMILIES:
            raise ReplayVerificationError(f"execution pair {pair_id} has unknown family")
        if pair["scenario_id"] not in SCENARIO_IDS:
            raise ReplayVerificationError(f"execution pair {pair_id} has unknown scenario")
        seed = pair["seed_id"]
        if pair["family_id"] in SINGLE_FAMILIES:
            if seed is not None:
                raise ReplayVerificationError(f"single family {pair_id} must have seed_id=null")
        elif seed not in statistics_v21.EXPECTED_RANDOM_SEEDS:
            raise ReplayVerificationError(f"seeded family {pair_id} has a non-frozen seed")
        if pair["statistics_arm_id"] is not None and pair["statistics_arm_id"] not in CONFIRMATORY_ARM_SOURCE:
            raise ReplayVerificationError(f"execution pair {pair_id} has unknown statistics arm")
        for key in ("input_object", "result_object"):
            if not isinstance(pair[key], str):
                raise ReplayVerificationError(f"execution pair {pair_id}.{key} must be a string")
            pair_references.add(pair[key])
            role_references[f"execution_pair:{pair_id}:{key}"] = pair[key]
    reference_names = list(role_references.values())
    if len(reference_names) != len(set(reference_names)):
        aliases: dict[str, list[str]] = {}
        for role, name in role_references.items():
            aliases.setdefault(str(name), []).append(role)
        duplicated = {name: roles for name, roles in aliases.items() if len(roles) > 1}
        raise ReplayVerificationError(f"one replay object cannot serve multiple semantic roles: {duplicated}")
    referenced = set(reference_names)
    if referenced != set(objects):
        raise ReplayVerificationError(
            f"object reference closure differs: unreferenced={sorted(set(objects) - referenced)}, "
            f"missing={sorted(referenced - set(objects))}"
        )
    digest_roles: dict[str, list[str]] = {}
    for role, name in role_references.items():
        digest_roles.setdefault(str(objects[name]["sha256"]), []).append(role)
    duplicated_digests = {digest: roles for digest, roles in digest_roles.items() if len(roles) > 1}
    if duplicated_digests:
        raise ReplayVerificationError(f"identical content cannot be reused across semantic roles: {duplicated_digests}")
    return FrozenReplayBundle(path, bytes_sha256(raw_manifest), dict(manifest), blobs)


def _sha256_file(path: Path) -> str:
    return bytes_sha256(_read_regular_once(path))


def _actual_dependency_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = {
        "data_evidence": root / "xs_chan_data_v2_1.py",
        "state_engine": root / "trend_regime.py",
        "feature_engine": root / "xs_chan_features_v2.py",
        "research_engine": root / "xs_chan_planner_v2_1.py",
        "execution_engine": root / "xs_chan_execution_v2_1.py",
        "ledger_engine": root / "xs_chan_oos_ledger_v2_1.py",
        "statistics_engine": root / "xs_chan_statistics_v2_1.py",
        "replay_verifier": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ReplayVerificationError(f"dependency files are unavailable: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def _expected_pair_keys() -> set[tuple[str, int | None, str]]:
    keys = {(family, None, scenario) for family in SINGLE_FAMILIES for scenario in SCENARIO_IDS}
    keys.update(
        (family, seed, scenario)
        for family in SEEDED_FAMILIES
        for seed in statistics_v21.EXPECTED_RANDOM_SEEDS
        for scenario in SCENARIO_IDS
    )
    return keys


def _expected_statistics_arm(family: str, scenario: str) -> str | None:
    for arm, (registered_family, _seed_mode, registered_scenario) in CONFIRMATORY_ARM_SOURCE.items():
        if family == registered_family and scenario == registered_scenario:
            return arm
    return None


def _protocol_cost_input(protocol: Mapping[str, Any], multiplier: float) -> dict[str, float]:
    source = protocol["cost_model"]
    return {
        "commission_bps": float(source["commission_bps"]),
        "transfer_bps": float(source["transfer_bps"]),
        "min_commission_cny": float(source["min_commission_cny"]),
        "stamp_duty_bps_before_2023_08_28": float(source["stamp_duty_bps_before_2023_08_28"]),
        "stamp_duty_bps_from_2023_08_28": float(source["stamp_duty_bps_from_2023_08_28"]),
        "base_slippage_bps": float(source["base_slippage_bps"]),
        "impact_bps_at_1pct": float(source["impact_bps_at_1pct"]),
        "max_adv_participation": float(source["max_adv_participation"]),
        "official_open_auction_turnover_participation_cap": float(
            source["official_open_auction_turnover_participation_cap"]
        ),
        "cost_multiplier": multiplier,
    }


def _registered_cycles(ledger_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    window = ledger_report.get("confirmation_window")
    if not isinstance(window, Mapping):
        return []
    weeks = window.get("weeks")
    if not isinstance(weeks, list) or len(weeks) < 52:
        return []
    return [
        {
            "week_index": index,
            "start_decision_session": str(week["decision_dt"]),
            "end_decision_session": str(week["close_dt"]),
        }
        for index, week in enumerate(weeks[:52], start=1)
    ]


@dataclass(frozen=True)
class LedgerConfirmationIndex:
    """Hash-addressed ledger records needed by the semantic replay boundary."""

    record_hashes: tuple[str, ...]
    all_record_hashes: tuple[str, ...]
    records_sha256: str
    genesis_recorded_at_utc: str
    expected_daily_sessions: tuple[str, ...]
    covered_sessions: tuple[str, ...]
    unwind_sessions: tuple[str, ...]
    decision_record_by_session: dict[str, str]
    open_record_by_session: dict[str, str]
    eod_record_by_session: dict[str, str]
    close_record_by_session: dict[str, str]
    unwind_decision_record_sha256: str | None
    final_evaluation_record_sha256: str | None
    event_data_by_record_hash: dict[str, dict[str, Any]]
    recorded_at_utc_by_record_hash: dict[str, str]
    anchor_issued_at_utc_by_record_hash: dict[str, str]


def _ledger_event(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    if record.get("record_type") == "genesis":
        if not isinstance(payload, Mapping):
            raise ReplayVerificationError("ledger genesis payload is missing")
        return dict(payload)
    if not isinstance(payload, Mapping) or set(payload) != {"event", "record_commitment_sha256", "external_anchor"}:
        raise ReplayVerificationError("ledger anchored record wrapper has an inexact schema")
    event = payload["event"]
    if not isinstance(event, Mapping):
        raise ReplayVerificationError("ledger anchored record event is missing")
    return dict(event)


def _build_ledger_confirmation_index(
    records: Any,
    window: Mapping[str, Any],
) -> LedgerConfirmationIndex:
    """Read and index the exact record prefix committed by the 52-week identity."""

    if not isinstance(records, (list, tuple)) or not records:
        raise ReplayVerificationError("ledger record reader returned no records")
    if not isinstance(window, Mapping):
        raise ReplayVerificationError("ledger confirmation window is missing")
    window_hashes = window.get("record_hashes_through_52nd_close")
    if not isinstance(window_hashes, list) or not window_hashes:
        raise ReplayVerificationError("confirmation window record-hash prefix is missing")
    normalized_records: list[dict[str, Any]] = []
    all_hashes: list[str] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ReplayVerificationError(f"ledger record {index} is not an object")
        record = dict(raw)
        digest = _require_sha256(record.get("record_hash"), f"ledger.records[{index}].record_hash")
        if record.get("sequence") != index:
            raise ReplayVerificationError("ledger records are not an exact contiguous sequence")
        normalized_records.append(record)
        all_hashes.append(digest)
    try:
        head_index = all_hashes.index(str(window.get("confirmation_head_sha256")))
    except ValueError as exc:
        raise ReplayVerificationError("confirmation head is absent from freshly read ledger records") from exc
    prefix = all_hashes[: head_index + 1]
    if prefix != window_hashes:
        raise ReplayVerificationError("fresh ledger record prefix differs from confirmation-window identity")

    weeks = window.get("weeks")
    if not isinstance(weeks, list) or len(weeks) != 52:
        raise ReplayVerificationError("confirmation record index requires exactly 52 ledger weeks")
    covered_sessions: list[str] = []
    decision_sessions: list[str] = []
    close_sessions: list[str] = []
    previous_close: str | None = None
    for index, week in enumerate(weeks, start=1):
        if not isinstance(week, Mapping):
            raise ReplayVerificationError(f"ledger week {index} is not an object")
        decision = str(week.get("decision_dt"))
        execution = str(week.get("execution_dt"))
        close = str(week.get("close_dt"))
        sessions = week.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            raise ReplayVerificationError(f"ledger week {index} has no covered sessions")
        normalized_sessions = [str(item) for item in sessions]
        if normalized_sessions != sorted(set(normalized_sessions)):
            raise ReplayVerificationError(f"ledger week {index} sessions are not increasing and unique")
        if normalized_sessions[0] != execution or normalized_sessions[-1] != close:
            raise ReplayVerificationError(f"ledger week {index} execution/close endpoints differ from sessions")
        if previous_close is not None and decision != previous_close:
            raise ReplayVerificationError("ledger week decisions do not form one contiguous NAV endpoint chain")
        previous_close = close
        decision_sessions.append(decision)
        close_sessions.append(close)
        covered_sessions.extend(normalized_sessions)
    if covered_sessions != sorted(set(covered_sessions)):
        raise ReplayVerificationError("covered ledger sessions overlap or are not strictly increasing")
    expected_daily_sessions = [decision_sessions[0], *covered_sessions]
    if expected_daily_sessions != sorted(set(expected_daily_sessions)):
        raise ReplayVerificationError("anchor decision plus covered sessions is not one exact increasing timeline")

    decision_records: dict[str, str] = {}
    open_records: dict[str, str] = {}
    eod_records: dict[str, str] = {}
    close_records: dict[str, str] = {}
    event_data: dict[str, dict[str, Any]] = {}
    recorded_at_by_hash = {
        str(record["record_hash"]): str(record.get("recorded_at_utc")) for record in normalized_records
    }
    anchor_issued_at_by_hash: dict[str, str] = {}
    for index, record in enumerate(normalized_records):
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("external_anchor"), Mapping):
            raise ReplayVerificationError(f"ledger record {index} external timestamp receipt is missing")
        issued = _parse_aware_utc(
            payload["external_anchor"].get("issued_at_utc"),
            f"ledger record {index} anchor issued_at_utc",
        )
        anchor_issued_at_by_hash[str(record["record_hash"])] = issued.isoformat().replace("+00:00", "Z")
    for record in normalized_records[: head_index + 1]:
        record_type = str(record.get("record_type"))
        digest = str(record["record_hash"])
        event = _ledger_event(record)
        event_data[digest] = event
        if record_type == "decision":
            key = str(event.get("decision_dt"))
            target = decision_records
        elif record_type == "session_open_execution":
            key = str(event.get("session_dt"))
            target = open_records
        elif record_type == "session_eod_valuation":
            key = str(event.get("session_dt"))
            target = eod_records
        elif record_type == "cycle_close":
            key = str(event.get("close_dt"))
            target = close_records
        elif record_type in {"genesis", "calendar_extension"}:
            continue
        else:
            raise ReplayVerificationError(
                f"terminal or unsupported record {record_type!r} occurs in confirmation prefix"
            )
        if not key or key == "None" or key in target:
            raise ReplayVerificationError(f"duplicate or missing {record_type} session identity")
        target[key] = digest

    if set(decision_records) != set(decision_sessions):
        raise ReplayVerificationError("ledger decision-record domain differs from the 52 registered weeks")
    if set(open_records) != set(covered_sessions) or set(eod_records) != set(covered_sessions):
        raise ReplayVerificationError("ledger open/EOD record domain differs from every covered official session")
    if set(close_records) != set(close_sessions):
        raise ReplayVerificationError("ledger cycle-close record domain differs from the 52 registered weeks")
    for week in weeks:
        decision = str(week["decision_dt"])
        close = str(week["close_dt"])
        if decision_records[decision] != week.get("decision_record_sha256"):
            raise ReplayVerificationError("ledger decision record hash differs from confirmation week identity")
        if close_records[close] != week.get("weekly_close_record_sha256"):
            raise ReplayVerificationError("ledger close record hash differs from confirmation week identity")

    unwind_decision_hash: str | None = None
    final_evaluation_hash: str | None = None
    unwind_open_records: dict[str, str] = {}
    unwind_eod_records: dict[str, str] = {}
    for record in normalized_records[head_index + 1 :]:
        record_type = str(record.get("record_type"))
        digest = str(record["record_hash"])
        event = _ledger_event(record)
        event_data[digest] = event
        if record_type == "calendar_extension":
            continue
        if record_type == "decision":
            if unwind_decision_hash is not None or event.get("decision_kind") != "POST_WINDOW_UNWIND":
                raise ReplayVerificationError("ledger suffix must contain at most one POST_WINDOW_UNWIND decision")
            unwind_decision_hash = digest
        elif record_type == "session_open_execution":
            if event.get("week_id") != ledger_v21.UNWIND_WEEK_ID or unwind_decision_hash is None:
                raise ReplayVerificationError("post-window open is not ordered after the unwind decision")
            session = str(event.get("session_dt"))
            if not session or session in unwind_open_records:
                raise ReplayVerificationError("post-window open session is missing or duplicated")
            unwind_open_records[session] = digest
        elif record_type == "session_eod_valuation":
            if event.get("week_id") != ledger_v21.UNWIND_WEEK_ID or unwind_decision_hash is None:
                raise ReplayVerificationError("post-window EOD is not ordered after the unwind decision")
            session = str(event.get("session_dt"))
            if not session or session in unwind_eod_records:
                raise ReplayVerificationError("post-window EOD session is missing or duplicated")
            unwind_eod_records[session] = digest
        elif record_type == "final_evaluation":
            if final_evaluation_hash is not None:
                raise ReplayVerificationError("ledger contains multiple final evaluations")
            final_evaluation_hash = digest
        else:
            raise ReplayVerificationError(f"unsupported record {record_type!r} occurs after the confirmation prefix")
    if set(unwind_open_records) != set(unwind_eod_records):
        raise ReplayVerificationError("post-window open/EOD session domains differ")
    unwind_sessions = sorted(unwind_open_records)
    if unwind_sessions != list(unwind_open_records) or unwind_sessions != list(unwind_eod_records):
        raise ReplayVerificationError("post-window unwind sessions are not strictly increasing")
    for session in unwind_sessions:
        if session <= covered_sessions[-1]:
            raise ReplayVerificationError("post-window unwind session does not follow the confirmation close")
        open_records[session] = unwind_open_records[session]
        eod_records[session] = unwind_eod_records[session]
    if final_evaluation_hash is not None and (unwind_decision_hash is None or not unwind_sessions):
        raise ReplayVerificationError("final evaluation lacks an explicit, observed unwind")

    genesis_time = _parse_aware_utc(normalized_records[0].get("recorded_at_utc"), "ledger.genesis.recorded_at_utc")
    return LedgerConfirmationIndex(
        record_hashes=tuple(prefix),
        all_record_hashes=tuple(all_hashes),
        records_sha256=object_sha256(prefix),
        genesis_recorded_at_utc=genesis_time.isoformat().replace("+00:00", "Z"),
        expected_daily_sessions=(*expected_daily_sessions, *unwind_sessions),
        covered_sessions=tuple(covered_sessions),
        unwind_sessions=tuple(unwind_sessions),
        decision_record_by_session=decision_records,
        open_record_by_session=open_records,
        eod_record_by_session=eod_records,
        close_record_by_session=close_records,
        unwind_decision_record_sha256=unwind_decision_hash,
        final_evaluation_record_sha256=final_evaluation_hash,
        event_data_by_record_hash=event_data,
        recorded_at_utc_by_record_hash=recorded_at_by_hash,
        anchor_issued_at_utc_by_record_hash=anchor_issued_at_by_hash,
    )


def _data_manifest_path(root_manifest_path: str | Path, digest: str) -> Path:
    """Resolve one sibling content-addressed manifest without a loose path."""

    digest = _require_sha256(digest, "data manifest digest")
    supplied = Path(root_manifest_path).expanduser()
    if supplied.parent.name != "manifests":
        raise ReplayVerificationError("formal data_manifest_path must be inside a manifests directory")
    return supplied.parent / f"{digest}.json"


def _ledger_data_transition(
    *,
    contract_sha256: str,
    previous_head: str,
    decision_session: str,
    data: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    required = {
        "snapshot_cutoff_utc",
        "data_manifest_sha256",
        "data_validation_report_sha256",
        "previous_data_chain_head_sha256",
        "data_chain_head_sha256",
    }
    if not required.issubset(data):
        raise ReplayVerificationError(f"{label} lacks the exact data-chain evidence")
    transition = {
        "decision_session": str(decision_session),
        "snapshot_cutoff_utc": str(data["snapshot_cutoff_utc"]),
        "data_manifest_sha256": _require_sha256(data["data_manifest_sha256"], f"{label}.data_manifest_sha256"),
        "data_validation_report_sha256": _require_sha256(
            data["data_validation_report_sha256"], f"{label}.data_validation_report_sha256"
        ),
        "previous_data_chain_head_sha256": _require_sha256(
            data["previous_data_chain_head_sha256"], f"{label}.previous_data_chain_head_sha256"
        ),
        "data_chain_head_sha256": _require_sha256(data["data_chain_head_sha256"], f"{label}.data_chain_head_sha256"),
    }
    if transition["previous_data_chain_head_sha256"] != previous_head:
        raise ReplayVerificationError(f"{label} does not extend the prior data-chain head")
    expected = data_v21.data_chain_transition_sha256(
        data_contract_identity_sha256=contract_sha256,
        previous_data_chain_head_sha256=previous_head,
        decision_session=transition["decision_session"],
        snapshot_cutoff_utc=transition["snapshot_cutoff_utc"],
        data_manifest_sha256=transition["data_manifest_sha256"],
        data_validation_report_sha256=transition["data_validation_report_sha256"],
    )
    if transition["data_chain_head_sha256"] != expected:
        raise ReplayVerificationError(f"{label} data-chain head is not canonically derived")
    return transition


def verify_forward_data_chain(
    *,
    data_manifest_path: str | Path,
    bundle_manifest: Mapping[str, Any],
    ledger_report: Mapping[str, Any],
    ledger_index: LedgerConfirmationIndex,
) -> VerifiedForwardDataChain:
    """Resolve and verify Genesis, 52 weekly, and unwind data snapshots.

    The immutable contract is fixed at Genesis, while every decision appends a
    content-addressed rolling snapshot transition.  Each snapshot's explicit
    observation horizon must equal the causal ledger session that consumes it.
    """

    contract_sha = data_v21.data_contract_identity_sha256()
    if (
        bundle_manifest.get("data_contract_identity_sha256") != contract_sha
        or ledger_report.get("data_contract_identity_sha256") != contract_sha
    ):
        raise ReplayVerificationError("replay/ledger data contract identity differs from the executing verifier")
    config = ledger_report.get("genesis_config")
    if not isinstance(config, Mapping):
        raise ReplayVerificationError("ledger Genesis config is missing")
    eligibility = config.get("formal_start_eligibility")
    if not isinstance(eligibility, Mapping):
        raise ReplayVerificationError("ledger Genesis formal-start evidence is missing")
    genesis_manifest = _require_sha256(config.get("data_bundle_sha256"), "ledger Genesis data bundle")
    genesis_report_sha = _require_sha256(
        config.get("genesis_data_validation_report_sha256"), "ledger Genesis data validation report"
    )
    genesis_head = _require_sha256(config.get("genesis_data_chain_head_sha256"), "ledger Genesis data head")
    if genesis_manifest != bundle_manifest.get("genesis_data_manifest_sha256"):
        raise ReplayVerificationError("replay manifest changes the Genesis data snapshot")
    expected_genesis_head = data_v21.data_chain_transition_sha256(
        data_contract_identity_sha256=contract_sha,
        previous_data_chain_head_sha256=ledger_v21.ZERO_HASH,
        decision_session=str(eligibility.get("candidate_decision_session")),
        snapshot_cutoff_utc=str(eligibility.get("eligibility_cutoff_utc")),
        data_manifest_sha256=genesis_manifest,
        data_validation_report_sha256=genesis_report_sha,
    )
    if genesis_head != expected_genesis_head:
        raise ReplayVerificationError("ledger Genesis data-chain head is not canonically derived")

    transitions: list[dict[str, Any]] = []
    previous_head = genesis_head
    decision_transitions: dict[str, dict[str, Any]] = {}
    window = ledger_report.get("confirmation_window")
    weeks = window.get("weeks") if isinstance(window, Mapping) else None
    if not isinstance(weeks, list) or len(weeks) != 52:
        raise ReplayVerificationError("forward data chain requires the exact 52-week confirmation window")
    for index, week in enumerate(weeks, start=1):
        decision_session = str(week.get("decision_dt"))
        record_sha = ledger_index.decision_record_by_session.get(decision_session)
        event = ledger_index.event_data_by_record_hash.get(str(record_sha))
        data = event.get("data") if isinstance(event, Mapping) else None
        if not isinstance(data, Mapping):
            raise ReplayVerificationError(f"ledger decision {index} lacks data-chain evidence")
        transition = _ledger_data_transition(
            contract_sha256=contract_sha,
            previous_head=previous_head,
            decision_session=decision_session,
            data=data,
            label=f"ledger decision {index}",
        )
        transition["ledger_record_sha256"] = str(record_sha)
        for field in ("data_manifest_sha256", "data_validation_report_sha256", "data_chain_head_sha256"):
            if week.get(field) != transition[field]:
                raise ReplayVerificationError(f"confirmation week {index} changes {field}")
        transitions.append(transition)
        decision_transitions[decision_session] = transition
        previous_head = transition["data_chain_head_sha256"]
    if window.get("confirmation_data_chain_head_sha256") != previous_head:
        raise ReplayVerificationError("confirmation-window data head differs from its 52nd transition")

    if ledger_index.unwind_decision_record_sha256 is None:
        raise ReplayVerificationError("forward data chain lacks the post-window unwind transition")
    unwind_event = ledger_index.event_data_by_record_hash[ledger_index.unwind_decision_record_sha256]
    unwind_data = unwind_event.get("data")
    if not isinstance(unwind_data, Mapping):
        raise ReplayVerificationError("post-window unwind lacks data-chain evidence")
    unwind_transition = _ledger_data_transition(
        contract_sha256=contract_sha,
        previous_head=previous_head,
        decision_session=str(unwind_event.get("decision_dt")),
        data=unwind_data,
        label="post-window unwind",
    )
    unwind_transition["ledger_record_sha256"] = ledger_index.unwind_decision_record_sha256
    transitions.append(unwind_transition)
    previous_head = unwind_transition["data_chain_head_sha256"]
    unwind_eod_transitions: dict[str, dict[str, Any]] = {}
    for session in ledger_index.unwind_sessions:
        eod_record_sha = ledger_index.eod_record_by_session[session]
        eod_event = ledger_index.event_data_by_record_hash[eod_record_sha]
        eod_data = eod_event.get("data")
        if not isinstance(eod_data, Mapping):
            raise ReplayVerificationError(f"post-window unwind EOD {session} lacks data-chain evidence")
        transition = _ledger_data_transition(
            contract_sha256=contract_sha,
            previous_head=previous_head,
            decision_session=session,
            data=eod_data,
            label=f"post-window unwind EOD {session}",
        )
        transition["ledger_record_sha256"] = eod_record_sha
        transitions.append(transition)
        unwind_eod_transitions[session] = transition
        previous_head = transition["data_chain_head_sha256"]
    final_manifest = transitions[-1]["data_manifest_sha256"]
    final_head = transitions[-1]["data_chain_head_sha256"]
    if (
        bundle_manifest.get("final_data_manifest_sha256") != final_manifest
        or bundle_manifest.get("final_data_chain_head_sha256") != final_head
        or ledger_report.get("current_data_chain_head_sha256") != final_head
    ):
        raise ReplayVerificationError("replay/ledger final data snapshot or head differs from unwind")
    supplied_final = _data_manifest_path(data_manifest_path, final_manifest)
    if Path(data_manifest_path).expanduser() != supplied_final:
        raise ReplayVerificationError("data_manifest_path must name the replay manifest's final physical snapshot")

    reports: dict[str, data_v21.DataValidationReport] = {}
    snapshots: dict[str, data_v21.VerifiedDataSnapshot] = {}

    def load_snapshot(digest: str) -> data_v21.VerifiedDataSnapshot:
        if digest not in snapshots:
            report, snapshot = data_v21.load_verified_snapshot(_data_manifest_path(data_manifest_path, digest))
            if snapshot is None or report.valid is not True or report.manifest_sha256 != digest:
                raise ReplayVerificationError(f"data snapshot {digest} failed strict validation")
            reports[digest] = report
            snapshots[digest] = snapshot
        return snapshots[digest]

    genesis_snapshot = load_snapshot(genesis_manifest)
    _require_snapshot_observation(
        genesis_snapshot,
        eligibility.get("candidate_decision_session"),
        "Genesis data snapshot",
    )
    if data_v21.data_validation_report_sha256(reports[genesis_manifest]) != genesis_report_sha:
        raise ReplayVerificationError("Genesis data validation report digest differs from ledger")
    prior_snapshot = genesis_snapshot
    extension_digests: list[str] = []
    transition_digests: list[str] = []
    prior_cutoff = _parse_aware_utc(str(eligibility.get("eligibility_cutoff_utc")), "Genesis snapshot cutoff")
    genesis_created = _parse_aware_utc(prior_snapshot.created_at_utc, "Genesis snapshot created_at_utc")
    genesis_recorded = _parse_aware_utc(
        ledger_index.recorded_at_utc_by_record_hash[ledger_index.record_hashes[0]],
        "Genesis ledger recorded_at_utc",
    )
    genesis_event = ledger_index.event_data_by_record_hash[ledger_index.record_hashes[0]]
    registry_anchor = genesis_event.get("primary_registry_external_anchor")
    genesis_anchor = genesis_event.get("external_anchor")
    if not isinstance(registry_anchor, Mapping) or not isinstance(genesis_anchor, Mapping):
        raise ReplayVerificationError("Genesis external timestamp receipts are missing")
    registry_issued = _parse_aware_utc(
        registry_anchor.get("issued_at_utc"),
        "Genesis registry receipt issued_at_utc",
    )
    genesis_issued = _parse_aware_utc(
        genesis_anchor.get("issued_at_utc"),
        "Genesis receipt issued_at_utc",
    )
    indexed_genesis_issued = _parse_aware_utc(
        ledger_index.anchor_issued_at_utc_by_record_hash[ledger_index.record_hashes[0]],
        "indexed Genesis receipt issued_at_utc",
    )
    if genesis_issued != indexed_genesis_issued:
        raise ReplayVerificationError("indexed Genesis timestamp differs from its receipt")
    if not prior_cutoff < genesis_created <= registry_issued <= genesis_issued <= genesis_recorded:
        raise ReplayVerificationError(
            "Genesis data/registry/anchor timestamps violate the candidate-close causal order"
        )
    for index, transition in enumerate(transitions, start=1):
        cutoff = _parse_aware_utc(transition["snapshot_cutoff_utc"], f"data transition {index} cutoff")
        if cutoff < prior_cutoff:
            raise ReplayVerificationError("data-chain cutoffs are not monotone")
        current = load_snapshot(transition["data_manifest_sha256"])
        _require_snapshot_observation(
            current,
            transition["decision_session"],
            f"data snapshot {index}",
        )
        created = _parse_aware_utc(current.created_at_utc, f"data snapshot {index} created_at_utc")
        recorded = _parse_aware_utc(
            ledger_index.recorded_at_utc_by_record_hash[transition["ledger_record_sha256"]],
            f"data transition {index} ledger recorded_at_utc",
        )
        anchor_issued = _parse_aware_utc(
            ledger_index.anchor_issued_at_utc_by_record_hash[transition["ledger_record_sha256"]],
            f"data transition {index} anchor issued_at_utc",
        )
        if not cutoff < created <= anchor_issued <= recorded:
            raise ReplayVerificationError(
                f"data snapshot {index} creation is outside its cutoff-to-anchor causal interval"
            )
        report_sha = data_v21.data_validation_report_sha256(reports[current.manifest_sha256])
        if report_sha != transition["data_validation_report_sha256"]:
            raise ReplayVerificationError(f"data transition {index} validation report differs from ledger")
        extension = data_v21.verify_snapshot_extension(prior_snapshot, current)
        if extension.get("valid") is not True or extension.get("errors") != []:
            raise ReplayVerificationError(f"data transition {index} is not an append-only snapshot extension")
        extension_digests.append(
            _require_sha256(extension.get("verification_sha256"), f"data transition {index} extension digest")
        )
        transition_digests.append(object_sha256(transition))
        prior_snapshot = current
        prior_cutoff = cutoff
    verification = {
        "schema": DATA_CHAIN_VERIFICATION_SCHEMA,
        "valid": True,
        "data_contract_identity_sha256": contract_sha,
        "genesis_data_manifest_sha256": genesis_manifest,
        "genesis_data_chain_head_sha256": genesis_head,
        "final_data_manifest_sha256": final_manifest,
        "final_data_chain_head_sha256": final_head,
        "decision_transition_count": 52,
        "unwind_decision_transition_count": 1,
        "unwind_eod_transition_count": len(unwind_eod_transitions),
        "unique_snapshot_count": len(snapshots),
        "transition_sha256": transition_digests,
        "extension_verification_sha256": extension_digests,
        "errors": [],
    }
    verification["verification_sha256"] = object_sha256(verification)
    return VerifiedForwardDataChain(
        genesis_snapshot=genesis_snapshot,
        final_snapshot=prior_snapshot,
        snapshots_by_sha256=snapshots,
        decision_transitions=decision_transitions,
        unwind_transition=unwind_transition,
        unwind_eod_transitions=unwind_eod_transitions,
        verification=verification,
    )


def _require_execution_pair_identity(
    execution_input: Mapping[str, Any],
    execution_result: Mapping[str, Any],
    *,
    family: str,
    seed: int | None,
    scenario: str,
) -> None:
    """Bind manifest metadata to content, including the formerly label-only seed."""

    for label, value in (("input", execution_input), ("result", execution_result)):
        if "seed_id" not in value:
            raise ReplayVerificationError(
                f"execution {label} requires a content-bound seed_id field; update xs_chan_execution_v2_1 schema"
            )
        if value["seed_id"] != seed:
            raise ReplayVerificationError(f"execution {label} seed_id differs from manifest metadata")
        if value.get("arm_id") != family or value.get("scenario_id") != scenario:
            raise ReplayVerificationError(f"execution {label} family/scenario differs from manifest metadata")


def _require_execution_timeline(
    execution_input: Mapping[str, Any],
    replayed: Mapping[str, Any],
    ledger_index: LedgerConfirmationIndex,
) -> None:
    """Require every book to replay the exact anchored daily/decision timeline."""

    sessions = execution_input.get("sessions")
    if not isinstance(sessions, list):
        raise ReplayVerificationError("execution input sessions are missing")
    input_labels = [str(item.get("session")) for item in sessions if isinstance(item, Mapping)]
    expected = list(ledger_index.expected_daily_sessions)
    if len(input_labels) != len(sessions) or input_labels != expected:
        raise ReplayVerificationError("execution input daily sessions differ from the anchored confirmation timeline")

    expected_decisions = dict(ledger_index.decision_record_by_session)
    unwind_decision_session: str | None = None
    if ledger_index.unwind_decision_record_sha256 is not None:
        unwind_event = ledger_index.event_data_by_record_hash[ledger_index.unwind_decision_record_sha256]
        unwind_decision_session = str(unwind_event.get("decision_dt"))
        if not unwind_decision_session or unwind_decision_session in expected_decisions:
            raise ReplayVerificationError(
                "ledger unwind decision session is missing or collides with a weekly decision"
            )
        expected_decisions[unwind_decision_session] = ledger_index.unwind_decision_record_sha256
    seen_decisions: dict[str, str] = {}
    for item in sessions:
        decision = item.get("decision")
        if decision is None:
            continue
        if not isinstance(decision, Mapping):
            raise ReplayVerificationError("execution decision is not an object")
        decision_session = str(decision.get("decision_session"))
        execution_session = str(decision.get("execution_session"))
        if execution_session != str(item.get("session")):
            raise ReplayVerificationError("execution decision session differs from its enclosing daily record")
        record_sha = str(decision.get("decision_record_sha256"))
        if decision_session in seen_decisions:
            raise ReplayVerificationError("execution input duplicates a weekly decision")
        if decision_session == unwind_decision_session and (
            decision.get("ordered_symbols") != []
            or decision.get("new_entry_symbols") != []
            or decision.get("gate_eligible_new_entry_opportunities") != 0
        ):
            raise ReplayVerificationError("post-window execution decision must be a full-exit, zero-entry decision")
        seen_decisions[decision_session] = record_sha
    if seen_decisions != expected_decisions:
        raise ReplayVerificationError("execution decisions do not bind the exact weekly and unwind ledger decisions")

    for field in ("daily", "session_evidence"):
        rows = replayed.get(field)
        if not isinstance(rows, list) or [str(item.get("session")) for item in rows] != expected:
            raise ReplayVerificationError(f"execution result {field} differs from the anchored daily timeline")
    decisions = replayed.get("decision_evidence")
    if not isinstance(decisions, list):
        raise ReplayVerificationError("execution result decision evidence is missing")
    replayed_decision_sessions = [str(item.get("decision_session")) for item in decisions]
    if replayed_decision_sessions != list(expected_decisions):
        raise ReplayVerificationError("execution result does not contain the exact ledger decision sequence")


def _planner_decision_registry(
    planner_batch: Mapping[str, Any],
    ledger_index: LedgerConfirmationIndex,
) -> dict[tuple[str, int | None, str, str], dict[str, Any]]:
    """Bind every batch artifact to its ledger decision and expand 13,104 decisions."""

    items = planner_batch.get("items")
    if not isinstance(items, list) or len(items) != 52:
        raise ReplayVerificationError("planner result must contain exactly 52 cycle artifacts")
    expected_decision_sessions = list(ledger_index.decision_record_by_session)
    registry: dict[tuple[str, int | None, str, str], dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ReplayVerificationError(f"planner cycle {index + 1} is not an object")
        decision_session = str(item.get("decision_dt"))
        if decision_session != expected_decision_sessions[index]:
            raise ReplayVerificationError("planner cycle decisions differ from the ledger confirmation window")
        ledger_record_sha = ledger_index.decision_record_by_session[decision_session]
        ledger_event = ledger_index.event_data_by_record_hash[ledger_record_sha]
        if str(item.get("exec_dt")) != str(ledger_event.get("execution_dt")):
            raise ReplayVerificationError("planner execution session differs from the ledger decision record")
        ledger_data = ledger_event.get("data")
        if not isinstance(ledger_data, Mapping) or ledger_data.get("decision_artifact_sha256") != item.get(
            "artifact_sha256"
        ):
            raise ReplayVerificationError("ledger decision artifact hash differs from the verified planner cycle")
        decisions = item.get("execution_decisions")
        if not isinstance(decisions, list) or len(decisions) != 252:
            raise ReplayVerificationError("each planner cycle must contain exactly 252 execution decisions")
        for row in decisions:
            if not isinstance(row, Mapping):
                raise ReplayVerificationError("planner execution decision is not an object")
            key = (
                str(row.get("family")),
                row.get("seed"),
                str(row.get("scenario")),
                str(row.get("decision_session")),
            )
            if key in registry:
                raise ReplayVerificationError(f"planner execution decision is duplicated: {key}")
            registry[key] = dict(row)
    if len(registry) != 52 * 252:
        raise ReplayVerificationError("planner decision registry does not contain exactly 13,104 unique rows")
    return registry


def _verify_formal_start_eligibility(
    eligibility: Mapping[str, Any],
    planner_batch: Mapping[str, Any],
    genesis_snapshot: data_v21.VerifiedDataSnapshot,
    planner_v21: Any,
    readiness_report: Mapping[str, Any],
    readiness_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the claimed earliest post-readiness eligible week."""

    if not isinstance(eligibility, Mapping) or set(eligibility) != ledger_v21._FORMAL_START_ELIGIBILITY_KEYS:
        raise ReplayVerificationError("formal-start eligibility evidence has an inexact schema")
    claimed_digest = _require_sha256(eligibility.get("evidence_sha256"), "formal_start_eligibility.evidence_sha256")
    evidence_body = dict(eligibility)
    evidence_body.pop("evidence_sha256")
    if object_sha256(evidence_body) != claimed_digest:
        raise ReplayVerificationError("formal-start eligibility evidence digest is not canonically derived")
    if eligibility.get("readiness_report_sha256") != readiness_report.get("report_sha256"):
        raise ReplayVerificationError("formal-start evidence names a different readiness commitment")
    if readiness_anchor.get("report_sha256") != readiness_report.get("report_sha256"):
        raise ReplayVerificationError("readiness timestamp receipt names a different readiness commitment")
    anchor_receipt_sha = _require_sha256(readiness_anchor.get("receipt_sha256"), "readiness anchor receipt")
    if eligibility.get("readiness_anchor_receipt_sha256") != anchor_receipt_sha:
        raise ReplayVerificationError("formal-start evidence names a different readiness timestamp receipt")
    if _parse_aware_utc(
        eligibility.get("readiness_completed_at_utc"),
        "formal_start_eligibility.readiness_completed_at_utc",
    ) != _parse_aware_utc(readiness_anchor.get("issued_at_utc"), "readiness anchor issued_at_utc"):
        raise ReplayVerificationError(
            "formal-start readiness completion differs from the verified external readiness timestamp"
        )
    items = planner_batch.get("items")
    if not isinstance(items, list) or len(items) != 52 or not isinstance(items[0], Mapping):
        raise ReplayVerificationError("formal-start verification requires the exact 52-cycle planner batch")
    candidate = items[0]
    candidate_session = str(eligibility.get("candidate_decision_session"))
    if (
        candidate.get("decision_dt") != candidate_session
        or candidate.get("eligible_count") != eligibility.get("candidate_eligible_count")
        or candidate.get("data_snapshot_sha256") != genesis_snapshot.manifest_sha256
    ):
        raise ReplayVerificationError("formal-start candidate differs from the first verified planner cycle")
    _require_sha256(candidate.get("ranked_surface_sha256"), "formal-start candidate ranked surface")

    frames = planner_v21.frames_from_verified_snapshot(genesis_snapshot)
    earlier_recomputed: list[dict[str, Any]] = []
    for index, raw in enumerate(eligibility.get("earlier_completed_weeks", [])):
        if not isinstance(raw, Mapping) or set(raw) != ledger_v21._EARLIER_ELIGIBILITY_WEEK_KEYS:
            raise ReplayVerificationError(f"formal-start earlier week {index} has an inexact schema")
        decision_session = str(raw["decision_session"])
        ranked = planner_v21.compute_decision_surface(
            frames,
            decision_session,
            require_target_size=False,
        )
        surface_payload = planner_v21._surface_payload(ranked)
        eligible_count = len(ranked)
        surface_sha = planner_v21.object_sha256(surface_payload)
        if eligible_count >= int(eligibility["minimum_eligible_symbol_count"]):
            raise ReplayVerificationError("formal-start candidate is not the earliest qualifying completed week")
        if raw["eligible_symbol_count"] != eligible_count or raw["evidence_sha256"] != surface_sha:
            raise ReplayVerificationError(
                f"formal-start earlier week {index} differs from frozen planner recomputation"
            )
        earlier_recomputed.append(
            {
                "week_id": raw["week_id"],
                "decision_session": decision_session,
                "eligible_symbol_count": eligible_count,
                "ranked_surface_sha256": surface_sha,
            }
        )
    result = {
        "candidate_decision_session": candidate_session,
        "candidate_eligible_count": int(candidate["eligible_count"]),
        "candidate_ranked_surface_sha256": candidate["ranked_surface_sha256"],
        "earlier_completed_weeks": earlier_recomputed,
        "formal_start_evidence_sha256": claimed_digest,
    }
    return {**result, "verification_sha256": object_sha256(result)}


def _verify_planner_state_continuity(
    planner_batch: Mapping[str, Any],
    results: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    ledger_index: LedgerConfirmationIndex,
) -> dict[str, Any]:
    """Bind weekly planner history and every reference book to replayed state."""

    items = planner_batch.get("items")
    if not isinstance(items, list) or len(items) != 52:
        raise ReplayVerificationError("planner continuity requires exactly 52 cycle artifacts")
    daily = {key: {str(row["session"]): row for row in result["daily"]} for key, result in results.items()}
    previous_factor: list[str] = []
    previous_random = {str(seed): [] for seed in statistics_v21.EXPECTED_RANDOM_SEEDS}
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        decision_session = str(item["decision_dt"])
        planning_inputs = item.get("planning_inputs")
        if not isinstance(planning_inputs, Mapping):
            raise ReplayVerificationError("planner continuity lacks planning_inputs")
        if planning_inputs.get("previous_factor_targets") != previous_factor:
            raise ReplayVerificationError(f"planner cycle {index + 1} changes previous factor targets")
        if planning_inputs.get("previous_r_match_targets_by_seed") != previous_random:
            raise ReplayVerificationError(f"planner cycle {index + 1} changes previous random-control targets")
        valuation_record_sha256 = (
            ledger_index.record_hashes[0] if index == 0 else ledger_index.eod_record_by_session[decision_session]
        )
        expected_books: dict[str, Any] = {}
        for book_id in ledger_v21.BOOK_IDS:
            family, separator, seed_text = book_id.partition("@")
            seed = int(seed_text) if separator else None
            reference_key = (family, seed, "1x")
            reference_day = daily[reference_key].get(decision_session)
            if reference_day is None:
                raise ReplayVerificationError("reference-book decision-close state is absent from execution replay")
            expected_books[book_id] = {
                "actual_holdings": [str(row["symbol"]) for row in reference_day["holdings"]],
                "blocked_exit_symbols": list(reference_day["pending_sell_symbols"]),
                "contingent_slot_symbols": list(reference_day["contingent_slot_symbols"]),
                "sizing_nav_cny_by_scenario": {
                    scenario: float(daily[(family, seed, scenario)][decision_session]["nav_cny"])
                    for scenario in SCENARIO_IDS
                },
                "sizing_nav_record_sha256_by_scenario": dict.fromkeys(SCENARIO_IDS, valuation_record_sha256),
            }
        if canonical_json_bytes(planning_inputs.get("reference_books")) != canonical_json_bytes(expected_books):
            raise ReplayVerificationError(f"planner cycle {index + 1} reference books differ from replayed state")
        previous_factor = list(item["arms"]["F"]["ordered_symbols"])
        previous_random = {
            str(seed): list(item["arms"][f"R_match@{seed}"]["ordered_symbols"])
            for seed in statistics_v21.EXPECTED_RANDOM_SEEDS
        }
        evidence.append(
            {
                "decision_session": decision_session,
                "valuation_record_sha256": valuation_record_sha256,
                "reference_books_sha256": object_sha256(expected_books),
                "previous_factor_targets_sha256": object_sha256(planning_inputs["previous_factor_targets"]),
                "previous_random_targets_sha256": object_sha256(planning_inputs["previous_r_match_targets_by_seed"]),
            }
        )
    return {"cycles": evidence, "continuity_sha256": object_sha256(evidence)}


def _bind_execution_decisions_to_planner(
    execution_input: Mapping[str, Any],
    *,
    family: str,
    seed: int | None,
    scenario: str,
    planner_registry: Mapping[tuple[str, int | None, str, str], Mapping[str, Any]],
) -> None:
    """Compare every execution decision with its verified planner batch row."""

    comparable = {
        "decision_session",
        "execution_session",
        "ordered_symbols",
        "new_entry_symbols",
        "gate_eligible_new_entry_opportunities",
        "slots",
        "cash_buffer_fraction",
        "selection_identity_sha256",
        "sizing_nav_cny",
        "sizing_nav_record_sha256",
    }
    seen = 0
    for raw_session in execution_input["sessions"]:
        decision = raw_session.get("decision")
        if decision is None:
            continue
        decision_session = str(decision.get("decision_session"))
        key = (family, seed, scenario, decision_session)
        expected = planner_registry.get(key)
        if expected is None:
            if (
                decision.get("ordered_symbols") == []
                and decision.get("new_entry_symbols") == []
                and decision.get("gate_eligible_new_entry_opportunities") == 0
            ):
                continue
            raise ReplayVerificationError(f"execution decision has no verified planner row: {key}")
        actual_subset = {field: decision.get(field) for field in comparable}
        expected_subset = {field: expected.get(field) for field in comparable}
        if canonical_json_bytes(actual_subset) != canonical_json_bytes(expected_subset):
            raise ReplayVerificationError(f"execution decision differs from verified planner row: {key}")
        seen += 1
    if seen != 52:
        raise ReplayVerificationError("each execution book must bind exactly 52 verified planner decisions")


def _book_id(key: tuple[str, int | None, str]) -> str:
    family, seed, _scenario = key
    return family if seed is None else f"{family}@{seed}"


def _book_scenario_matrix(values: Mapping[tuple[str, int | None, str], Any]) -> dict[str, dict[str, Any]]:
    """Return the exact 63-book × 4-scenario canonical aggregation matrix."""

    matrix: dict[str, dict[str, Any]] = {book_id: {} for book_id in ledger_v21.BOOK_IDS}
    for key, value in values.items():
        book_id = _book_id(key)
        scenario = key[2]
        if book_id not in matrix or scenario not in SCENARIO_IDS or scenario in matrix[book_id]:
            raise ReplayVerificationError(f"invalid or duplicate book/scenario aggregation key {key}")
        matrix[book_id][scenario] = value
    if any(set(row) != set(SCENARIO_IDS) for row in matrix.values()):
        raise ReplayVerificationError("all-book aggregate matrix is not the exact 63×4 domain")
    return matrix


def _book_scenario_root(values: Mapping[tuple[str, int | None, str], Any]) -> str:
    return object_sha256(_book_scenario_matrix(values))


def _state_gate_observations(
    planner_batch: Mapping[str, Any],
    result: Mapping[str, Any],
    cycles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive every FC candidate-state observation from planner and fills."""

    items = planner_batch.get("items")
    if not isinstance(items, list) or len(items) != 52:
        raise ReplayVerificationError("state diagnostics require the verified 52-cycle planner batch")
    cycle_by_decision = {str(item["start_decision_session"]): item for item in cycles}
    decision_rows = {str(row["decision_session"]): row for row in result["decision_evidence"]}
    daily = [row for row in result["daily"] if str(row["session"]) <= str(cycles[-1]["end_decision_session"])]
    daily_by_session = {str(row["session"]): row for row in daily}
    final_session = str(cycles[-1]["end_decision_session"])
    final_holdings = {str(row["symbol"]): row for row in daily_by_session[final_session]["holdings"]}

    buy_orders: dict[tuple[str, str], Mapping[str, Any]] = {}
    filled_sells: dict[str, list[Mapping[str, Any]]] = {}
    for event in result["events"]:
        if event.get("event_type") != "order" or str(event.get("session")) > final_session:
            continue
        if event.get("side") == "buy":
            key = (str(event["decision_session"]), str(event["symbol"]))
            if key in buy_orders:
                raise ReplayVerificationError(f"FC execution contains duplicate buy attempt {key}")
            buy_orders[key] = event
        elif event.get("side") == "sell" and int(event.get("filled_shares", 0)) > 0:
            filled_sells.setdefault(str(event["symbol"]), []).append(event)
    for events in filled_sells.values():
        events.sort(key=lambda event: str(event["session"]))

    observations: list[dict[str, Any]] = []
    for artifact in items:
        decision_session = str(artifact["decision_dt"])
        cycle = cycle_by_decision.get(decision_session)
        decision_evidence = decision_rows.get(decision_session)
        if cycle is None or decision_evidence is None:
            raise ReplayVerificationError("FC state diagnostic lacks a registered decision endpoint")
        arm = artifact.get("arms", {}).get("FC")
        ranked_rows = artifact.get("ranked_surface")
        if not isinstance(arm, Mapping) or not isinstance(ranked_rows, list):
            raise ReplayVerificationError("planner FC arm or ranked surface is missing")
        candidates = arm.get("gate_candidate_new_entry_symbols")
        if not isinstance(candidates, list) or len(candidates) != arm.get("gate_candidate_new_entry_count"):
            raise ReplayVerificationError("planner FC candidate identities/count differ")
        ranked = {str(row["symbol"]): row for row in ranked_rows if isinstance(row, Mapping)}
        allowed_count = 0
        execution_session = str(artifact["exec_dt"])
        week_label = str(cycle["end_decision_session"])
        for symbol in candidates:
            symbol = str(symbol)
            surface = ranked.get(symbol)
            if surface is None:
                raise ReplayVerificationError(f"FC candidate {symbol} is absent from its ranked surface")
            regime = int(surface["regime"])
            allowed = regime in statistics_v21.ALLOWED_CHAN_REGIMES
            if bool(surface["chan_allowed"]) != allowed:
                raise ReplayVerificationError("planner Chan gate differs from the registered regime set")
            allowed_count += int(allowed)
            order = buy_orders.get((decision_session, symbol))
            requested = 0 if order is None else int(order["requested_shares"])
            filled = 0 if order is None else int(order["filled_shares"])
            notional = 0.0 if order is None else float(order.get("fill_notional_cny", 0.0))
            entry_price: float | None = None
            exit_session: str | None = None
            exit_price: float | None = None
            if requested > 0 and not allowed:
                raise ReplayVerificationError("FC execution requested a state-denied candidate")
            if filled > 0:
                entry_price = float(order["fill_price"])
                later_sells = [
                    event for event in filled_sells.get(symbol, []) if str(event["session"]) >= execution_session
                ]
                if later_sells:
                    exit_session = str(later_sells[0]["session"])
                    exit_price = float(later_sells[0]["fill_price"])
                elif symbol in final_holdings:
                    exit_session = final_session
                    exit_price = float(final_holdings[symbol]["close"])
                else:
                    marked = [
                        (session, next((row for row in day["holdings"] if str(row["symbol"]) == symbol), None))
                        for session, day in daily_by_session.items()
                        if execution_session <= session <= final_session
                    ]
                    marked = [(session, row) for session, row in marked if row is not None]
                    if not marked:
                        raise ReplayVerificationError("filled FC candidate has no official exit or window mark")
                    exit_session, mark = marked[-1]
                    exit_price = float(mark["close"])
            observations.append(
                {
                    "week_label": week_label,
                    "execution_session": execution_session,
                    "symbol": symbol,
                    "regime": regime,
                    "eligible_new_entry": True,
                    "allowed_new_entry": allowed,
                    "requested_shares": requested,
                    "filled_shares": filled,
                    "entry_fill_notional_cny": notional,
                    "pretrade_nav_cny": float(decision_evidence["pretrade_nav_cny"]),
                    "entry_price": entry_price,
                    "exit_or_window_session": exit_session,
                    "exit_or_window_price": exit_price,
                }
            )
        if allowed_count != int(arm["gate_eligible_new_entry_opportunities"]):
            raise ReplayVerificationError("FC state observations do not reconstruct the planner gate quota")
    observations.sort(key=lambda item: (item["week_label"], item["symbol"]))
    if len({(item["week_label"], item["symbol"]) for item in observations}) != len(observations):
        raise ReplayVerificationError("FC state observation identities are duplicated")
    return observations


def _verify_common_market_record_bindings(
    inputs: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    results: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    ledger_index: LedgerConfirmationIndex,
) -> dict[str, Any]:
    """Bind common exogenous market evidence to every ledger open/EOD record.

    Portfolio- and fill-specific hashes still require an all-book aggregate-root
    field in the ledger schema.  This helper closes all fields that are already
    representable without inventing that missing aggregate API.
    """

    pair_labels = {key: f"{key[0]}@{key[1] if key[1] is not None else 'primary'}:{key[2]}" for key in results}
    evidence_by_pair = {
        key: {str(item["session"]): item for item in result["session_evidence"]} for key, result in results.items()
    }
    input_by_pair = {
        key: {str(item["session"]): item for item in execution_input["sessions"]}
        for key, execution_input in inputs.items()
    }
    bound: dict[str, Any] = {}
    for session in (*ledger_index.covered_sessions, *ledger_index.unwind_sessions):
        common_fields = {
            field: {str(evidence_by_pair[key][session][field]) for key in results}
            for field in (
                "open_prices_sha256",
                "limit_state_sha256",
                "corporate_actions_sha256",
                "raw_close_snapshot_sha256",
            )
        }
        differing = {field: values for field, values in common_fields.items() if len(values) != 1}
        if differing:
            raise ReplayVerificationError(
                f"execution books use different exogenous market evidence on {session}: {sorted(differing)}"
            )
        auction_digests = set()
        for key in results:
            rows = input_by_pair[key][session]["open_snapshot"]
            auction_digests.add(
                object_sha256(
                    [
                        {
                            "symbol": str(row["symbol"]),
                            "open_auction_turnover_cny": row["open_auction_turnover_cny"],
                        }
                        for row in sorted(rows, key=lambda row: str(row["symbol"]))
                    ]
                )
            )
        if len(auction_digests) != 1:
            raise ReplayVerificationError(f"execution books use different opening-auction evidence on {session}")

        open_hash = ledger_index.open_record_by_session[session]
        eod_hash = ledger_index.eod_record_by_session[session]
        open_data = ledger_index.event_data_by_record_hash[open_hash].get("data")
        eod_data = ledger_index.event_data_by_record_hash[eod_hash].get("data")
        if not isinstance(open_data, Mapping) or not isinstance(eod_data, Mapping):
            raise ReplayVerificationError("ledger open/EOD event data is missing")
        expected_open = {
            "open_prices_sha256": next(iter(common_fields["open_prices_sha256"])),
            "opening_auction_turnover_sha256": next(iter(auction_digests)),
            "limit_state_sha256": next(iter(common_fields["limit_state_sha256"])),
            "corporate_actions_sha256": next(iter(common_fields["corporate_actions_sha256"])),
        }
        expected_eod = {
            "raw_close_snapshot_sha256": next(iter(common_fields["raw_close_snapshot_sha256"])),
            "corporate_actions_sha256": next(iter(common_fields["corporate_actions_sha256"])),
        }
        for field, expected_digest in expected_open.items():
            if open_data.get(field) != expected_digest:
                raise ReplayVerificationError(f"ledger open record {session}.{field} differs from execution replay")
        for field, expected_digest in expected_eod.items():
            if eod_data.get(field) != expected_digest:
                raise ReplayVerificationError(f"ledger EOD record {session}.{field} differs from execution replay")
        bound[session] = {
            "open_record_sha256": open_hash,
            "eod_record_sha256": eod_hash,
            "market_evidence": {**expected_open, **expected_eod},
            "pair_count": len(pair_labels),
        }
    return {"sessions": bound, "binding_sha256": object_sha256(bound)}


def _require_ledger_fields(data: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, value in expected.items():
        if canonical_json_bytes(data.get(field)) != canonical_json_bytes(value):
            raise ReplayVerificationError(f"{label}.{field} differs from the 252-book semantic replay")


def _verify_all_book_ledger_bindings(
    inputs: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    results: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
    planner_batch: Mapping[str, Any],
    cycles: list[dict[str, Any]],
    ledger_index: LedgerConfirmationIndex,
    *,
    data_chain: VerifiedForwardDataChain,
    dependency_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute every ledger aggregate root from all 252 portfolio books."""

    evidence = {
        key: {str(row["session"]): row for row in result["session_evidence"]} for key, result in results.items()
    }
    daily = {key: {str(row["session"]): row for row in result["daily"]} for key, result in results.items()}
    decisions = {
        key: {str(row["decision_session"]): row for row in result["decision_evidence"]}
        for key, result in results.items()
    }
    events = {
        key: {
            session: [event for event in result["events"] if str(event.get("session")) == session]
            for session in ledger_index.expected_daily_sessions
        }
        for key, result in results.items()
    }
    session_order = list(ledger_index.expected_daily_sessions)
    prior_session = {
        session: (None if index == 0 else session_order[index - 1]) for index, session in enumerate(session_order)
    }

    initial_state_root = _book_scenario_root({key: result["initial_state_sha256"] for key, result in results.items()})
    initial_pending_root = _book_scenario_root(
        {key: object_sha256(result["initial_state"]["pending_sells"]) for key, result in results.items()}
    )

    planner_items = {str(item["decision_dt"]): item for item in planner_batch["items"]}
    decision_bindings: dict[str, Any] = {}
    for decision_session, record_hash in ledger_index.decision_record_by_session.items():
        item = planner_items.get(decision_session)
        if item is None:
            raise ReplayVerificationError("ledger decision has no verified planner cycle")
        data = ledger_index.event_data_by_record_hash[record_hash].get("data")
        if not isinstance(data, Mapping):
            raise ReplayVerificationError("ledger decision data is missing")
        transition = data_chain.decision_transitions.get(decision_session)
        if transition is None or transition["data_manifest_sha256"] != item["data_snapshot_sha256"]:
            raise ReplayVerificationError("ledger/planner weekly data snapshot differs")
        state_root = _book_scenario_root({key: daily[key][decision_session]["state_sha256"] for key in results})
        pending_root = _book_scenario_root(
            {key: evidence[key][decision_session]["pending_sells_after_open_sha256"] for key in results}
        )
        expected = {
            "engine_sha256": dependency_sha256["research_engine"],
            "decision_artifact_sha256": item["artifact_sha256"],
            "book_state_before_root_sha256": state_root,
            "pending_sells_before_root_sha256": pending_root,
            "eligible_symbol_count": int(item["eligible_count"]),
            "eligibility_evidence_sha256": item["ranked_surface_sha256"],
            "snapshot_cutoff_utc": transition["snapshot_cutoff_utc"],
            "data_manifest_sha256": transition["data_manifest_sha256"],
            "previous_data_chain_head_sha256": transition["previous_data_chain_head_sha256"],
            "data_validation_report_sha256": transition["data_validation_report_sha256"],
            "data_chain_head_sha256": transition["data_chain_head_sha256"],
            "planner_artifact_sha256": item["artifact_sha256"],
        }
        _require_ledger_fields(data, expected, f"ledger decision {decision_session}")
        decision_bindings[decision_session] = expected

    session_bindings: dict[str, Any] = {}
    first_unwind = ledger_index.unwind_sessions[0] if ledger_index.unwind_sessions else None
    for session in (*ledger_index.covered_sessions, *ledger_index.unwind_sessions):
        open_hash = ledger_index.open_record_by_session[session]
        eod_hash = ledger_index.eod_record_by_session[session]
        open_data = ledger_index.event_data_by_record_hash[open_hash].get("data")
        eod_data = ledger_index.event_data_by_record_hash[eod_hash].get("data")
        if not isinstance(open_data, Mapping) or not isinstance(eod_data, Mapping):
            raise ReplayVerificationError("ledger open/EOD aggregate data is missing")

        if session == first_unwind:
            unwind_decision_session = str(
                ledger_index.event_data_by_record_hash[ledger_index.unwind_decision_record_sha256]["decision_dt"]
            )
            before_state_values = {
                key: decisions[key][unwind_decision_session]["portfolio_after_decision_sha256"] for key in results
            }
            before_pending_values = {
                key: decisions[key][unwind_decision_session]["pending_sells_after_decision_sha256"] for key in results
            }
            pending_before = sum(
                len(decisions[key][unwind_decision_session]["pending_sell_symbols_after_decision"]) for key in results
            )
        else:
            before_state_values = {key: evidence[key][session]["portfolio_before_sha256"] for key in results}
            before_pending_values = {key: evidence[key][session]["pending_sells_before_sha256"] for key in results}
            pending_before = sum(int(evidence[key][session]["pending_sell_count_before"]) for key in results)
        before_position_count = sum(
            0 if prior_session[session] is None else int(daily[key][prior_session[session]]["holding_count"])
            for key in results
        )
        after_position_count = sum(int(daily[key][session]["holding_count"]) for key in results)
        pending_after = sum(int(evidence[key][session]["pending_sell_count_after"]) for key in results)
        requested_root = _book_scenario_root(
            {key: evidence[key][session]["requested_orders_sha256"] for key in results}
        )
        terminal_share_receipts_root = _book_scenario_root(
            {key: evidence[key][session]["terminal_share_receipts_sha256"] for key in results}
        )
        terminal_share_receipt_count = sum(
            int(evidence[key][session]["terminal_share_receipt_count"]) for key in results
        )
        fills_root = _book_scenario_root({key: evidence[key][session]["fills_sha256"] for key in results})
        fees_root = _book_scenario_root({key: evidence[key][session]["fees_sha256"] for key in results})
        after_open_root = _book_scenario_root(
            {key: evidence[key][session]["portfolio_after_open_sha256"] for key in results}
        )
        pending_after_root = _book_scenario_root(
            {key: evidence[key][session]["pending_sells_after_open_sha256"] for key in results}
        )
        execution_result_root = _book_scenario_root(
            {
                key: object_sha256(
                    {
                        "session": session,
                        "requested_orders_sha256": evidence[key][session]["requested_orders_sha256"],
                        "fills_sha256": evidence[key][session]["fills_sha256"],
                        "fees_sha256": evidence[key][session]["fees_sha256"],
                        "portfolio_after_open_sha256": evidence[key][session]["portfolio_after_open_sha256"],
                        "pending_sells_after_open_sha256": evidence[key][session]["pending_sells_after_open_sha256"],
                    }
                )
                for key in results
            }
        )
        expected_open = {
            "engine_sha256": dependency_sha256["execution_engine"],
            "book_state_before_root_sha256": _book_scenario_root(before_state_values),
            "pending_sells_before_root_sha256": _book_scenario_root(before_pending_values),
            "pending_sell_count_before": pending_before,
            "aggregate_position_count_before": before_position_count,
            "terminal_share_receipts_root_sha256": terminal_share_receipts_root,
            "terminal_share_receipt_count": terminal_share_receipt_count,
            "requested_orders_root_sha256": requested_root,
            "fills_root_sha256": fills_root,
            "fees_root_sha256": fees_root,
            "execution_result_root_sha256": execution_result_root,
            "book_state_after_root_sha256": after_open_root,
            "pending_sells_after_root_sha256": pending_after_root,
            "pending_sell_count_after": pending_after,
            "aggregate_position_count_after": after_position_count,
        }
        _require_ledger_fields(open_data, expected_open, f"ledger open {session}")

        turnover_values: dict[tuple[str, int | None, str], float] = {}
        for key in results:
            filled_notional = sum(
                float(event.get("fill_notional_cny", 0.0))
                for event in events[key][session]
                if event.get("event_type") == "order" and int(event.get("filled_shares", 0)) > 0
            )
            turnover_values[key] = filled_notional / (2.0 * float(evidence[key][session]["pretrade_nav_cny"]))
        eod_state_root = _book_scenario_root(
            {key: evidence[key][session]["eod_portfolio_state_sha256"] for key in results}
        )
        eod_pending_root = _book_scenario_root(
            {key: evidence[key][session]["eod_pending_sells_sha256"] for key in results}
        )
        eod_pending_count = sum(int(evidence[key][session]["eod_pending_sell_count"]) for key in results)
        expected_eod = {
            "engine_sha256": dependency_sha256["execution_engine"],
            "book_state_before_eod_root_sha256": after_open_root,
            "book_state_root_sha256": eod_state_root,
            "pending_sells_root_sha256": eod_pending_root,
            "pending_sell_count": eod_pending_count,
            "aggregate_position_count": after_position_count,
            "nav_root_sha256": _book_scenario_root(
                {key: {"nav_cny": float(daily[key][session]["nav_cny"])} for key in results}
            ),
            "exposure_root_sha256": _book_scenario_root(
                {key: {"gross_exposure": float(daily[key][session]["gross_exposure"])} for key in results}
            ),
            "turnover_root_sha256": _book_scenario_root(turnover_values),
        }
        if session in ledger_index.unwind_sessions:
            transition = data_chain.unwind_eod_transitions.get(session)
            if transition is None:
                raise ReplayVerificationError("post-window EOD lacks a verified data transition")
            pending_settlements_root = _book_scenario_root(
                {key: evidence[key][session]["pending_settlements_root_sha256"] for key in results}
            )
            pending_settlement_count = sum(int(evidence[key][session]["pending_settlement_count"]) for key in results)
            expected_eod.update(
                {
                    "snapshot_cutoff_utc": transition["snapshot_cutoff_utc"],
                    "data_manifest_sha256": transition["data_manifest_sha256"],
                    "previous_data_chain_head_sha256": transition["previous_data_chain_head_sha256"],
                    "data_validation_report_sha256": transition["data_validation_report_sha256"],
                    "data_chain_head_sha256": transition["data_chain_head_sha256"],
                    "aggregate_pending_settlements_root_sha256": pending_settlements_root,
                    "aggregate_pending_settlement_count": pending_settlement_count,
                }
            )
        _require_ledger_fields(eod_data, expected_eod, f"ledger EOD {session}")
        session_bindings[session] = {"open": expected_open, "eod": expected_eod}

    weekly_bindings: dict[str, Any] = {}
    for cycle in cycles:
        start = str(cycle["start_decision_session"])
        end = str(cycle["end_decision_session"])
        close_hash = ledger_index.close_record_by_session[end]
        close_data = ledger_index.event_data_by_record_hash[close_hash].get("data")
        if not isinstance(close_data, Mapping):
            raise ReplayVerificationError("ledger weekly close data is missing")
        weekly_values: dict[tuple[str, int | None, str], dict[str, Any]] = {}
        weekly_returns: dict[tuple[str, int | None, str], float] = {}
        for key in results:
            filled_notional = sum(
                float(event.get("fill_notional_cny", 0.0))
                for event in results[key]["events"]
                if event.get("event_type") == "order"
                and int(event.get("filled_shares", 0)) > 0
                and start < str(event.get("session")) <= end
            )
            decision = decisions[key][start]
            weekly_return = float(daily[key][end]["nav_cny"]) / float(daily[key][start]["nav_cny"]) - 1.0
            weekly_returns[key] = weekly_return
            weekly_values[key] = {
                "start_decision_session": start,
                "end_decision_session": end,
                "weekly_return": weekly_return,
                "weekly_one_way_turnover": filled_notional / (2.0 * float(decision["pretrade_nav_cny"])),
                "new_entry_symbols": list(decision["new_entry_symbols"]),
                "gate_eligible_new_entry_opportunities": int(decision["gate_eligible_new_entry_opportunities"]),
            }
        pending_root = _book_scenario_root({key: evidence[key][end]["eod_pending_sells_sha256"] for key in results})
        expected_close = {
            "execution_engine_sha256": dependency_sha256["execution_engine"],
            "statistics_engine_sha256": dependency_sha256["statistics_engine"],
            "book_state_root_sha256": _book_scenario_root({key: daily[key][end]["state_sha256"] for key in results}),
            "pending_sells_root_sha256": pending_root,
            "pending_sell_count": sum(int(evidence[key][end]["eod_pending_sell_count"]) for key in results),
            "aggregate_position_count": sum(int(daily[key][end]["holding_count"]) for key in results),
            "weekly_returns_root_sha256": _book_scenario_root(weekly_returns),
            "statistics_input_root_sha256": _book_scenario_root(weekly_values),
        }
        _require_ledger_fields(close_data, expected_close, f"ledger weekly close {end}")
        weekly_bindings[end] = expected_close

    unwind_binding: dict[str, Any] | None = None
    if ledger_index.unwind_decision_record_sha256 is not None:
        unwind_event = ledger_index.event_data_by_record_hash[ledger_index.unwind_decision_record_sha256]
        unwind_data = unwind_event.get("data")
        if not isinstance(unwind_data, Mapping):
            raise ReplayVerificationError("ledger unwind decision data is missing")
        decision_session = str(unwind_event["decision_dt"])
        before_state_root = _book_scenario_root({key: daily[key][decision_session]["state_sha256"] for key in results})
        before_pending_root = _book_scenario_root(
            {key: evidence[key][decision_session]["pending_sells_after_open_sha256"] for key in results}
        )
        exit_orders = {
            key: [
                {"symbol": symbol, "requested_shares": int(shares), "target_shares": 0}
                for symbol, shares in decisions[key][decision_session]["actual_shares_before"].items()
            ]
            for key in results
        }
        exit_orders_root = _book_scenario_root(exit_orders)
        after_state_root = _book_scenario_root(
            {key: decisions[key][decision_session]["portfolio_after_decision_sha256"] for key in results}
        )
        after_pending_root = _book_scenario_root(
            {key: decisions[key][decision_session]["pending_sells_after_decision_sha256"] for key in results}
        )
        position_count = sum(len(decisions[key][decision_session]["actual_shares_before"]) for key in results)
        pending_before_count = sum(int(daily[key][decision_session]["pending_sell_count"]) for key in results)
        pending_after_count = sum(
            len(decisions[key][decision_session]["pending_sell_symbols_after_decision"]) for key in results
        )
        full_exit_root = object_sha256(
            {
                "book_state_before_root_sha256": before_state_root,
                "aggregate_position_count_before": position_count,
                "exit_orders_root_sha256": exit_orders_root,
            }
        )
        unwind_evidence_root = object_sha256(
            {
                "confirmation_window_identity_sha256": unwind_data["confirmation_window_identity_sha256"],
                "exit_orders_root_sha256": exit_orders_root,
                "full_exit_coverage_root_sha256": full_exit_root,
                "book_state_after_decision_root_sha256": after_state_root,
                "pending_sells_after_decision_root_sha256": after_pending_root,
            }
        )
        expected_unwind = {
            "engine_sha256": dependency_sha256["execution_engine"],
            "snapshot_cutoff_utc": data_chain.unwind_transition["snapshot_cutoff_utc"],
            "data_manifest_sha256": data_chain.unwind_transition["data_manifest_sha256"],
            "previous_data_chain_head_sha256": data_chain.unwind_transition["previous_data_chain_head_sha256"],
            "data_validation_report_sha256": data_chain.unwind_transition["data_validation_report_sha256"],
            "data_chain_head_sha256": data_chain.unwind_transition["data_chain_head_sha256"],
            "book_state_before_root_sha256": before_state_root,
            "pending_sells_before_root_sha256": before_pending_root,
            "aggregate_position_count_before": position_count,
            "aggregate_pending_sell_count_before": pending_before_count,
            "exit_orders_root_sha256": exit_orders_root,
            "full_exit_coverage_root_sha256": full_exit_root,
            "book_state_after_decision_root_sha256": after_state_root,
            "pending_sells_after_decision_root_sha256": after_pending_root,
            "aggregate_pending_sell_count_after_decision": pending_after_count,
            "unwind_evidence_root_sha256": unwind_evidence_root,
        }
        _require_ledger_fields(unwind_data, expected_unwind, "ledger post-window unwind decision")
        unwind_binding = expected_unwind

    if not ledger_index.unwind_sessions:
        raise ReplayVerificationError("all-book ledger binding requires an observed post-window unwind")
    final_unwind_session = ledger_index.unwind_sessions[-1]
    completion_state_root = _book_scenario_root({key: result["final_state_sha256"] for key, result in results.items()})
    completion_pending_root = _book_scenario_root(
        {key: object_sha256(result["final_state"]["pending_sells"]) for key, result in results.items()}
    )
    completion_pending_settlements_root = _book_scenario_root(
        {key: evidence[key][final_unwind_session]["pending_settlements_root_sha256"] for key in results}
    )
    completion = {
        "book_state_root_sha256": completion_state_root,
        "pending_sells_root_sha256": completion_pending_root,
        "pending_settlements_root_sha256": completion_pending_settlements_root,
        "remaining_position_count": sum(len(result["final_state"]["positions"]) for result in results.values()),
        "remaining_pending_sell_count": sum(len(result["final_state"]["pending_sells"]) for result in results.values()),
        "remaining_pending_settlement_count": sum(
            int(evidence[key][final_unwind_session]["pending_settlement_count"]) for key in results
        ),
    }
    body = {
        "initial_state_root_sha256": initial_state_root,
        "initial_pending_sells_root_sha256": initial_pending_root,
        "decisions": decision_bindings,
        "sessions": session_bindings,
        "weekly_closes": weekly_bindings,
        "unwind_decision": unwind_binding,
        "completion": completion,
    }
    return {**body, "binding_sha256": object_sha256(body)}


def _statistics_controls(protocol: Mapping[str, Any]) -> dict[str, float]:
    validity = protocol["control_validity"]
    return {
        "maximum_top10_positive_weeks_share": float(
            protocol["statistics"]["inference"]["top10_positive_weeks_share_max_for_pass"]
        ),
        "maximum_mean_absolute_exposure_gap": float(validity["exposure"]["mean_absolute_gap_max"]),
        "maximum_p95_absolute_exposure_gap": float(validity["exposure"]["p95_absolute_gap_max"]),
        "maximum_absolute_exposure_gap": float(validity["exposure"]["maximum_absolute_gap_max"]),
        "maximum_mean_absolute_turnover_gap": float(validity["turnover"]["mean_absolute_weekly_gap_max"]),
        "maximum_p95_absolute_turnover_gap": float(validity["turnover"]["p95_absolute_weekly_gap_max"]),
        "maximum_random_annualized_seed_mean_mcse": float(
            validity["monte_carlo_standard_error"]["maximum_annualized_seed_mcse"]
        ),
        "maximum_bootstrap_quantile_mcse_annualized": float(
            validity["monte_carlo_standard_error"]["maximum_bootstrap_quantile_mcse_annualized"]
        ),
        "minimum_gate_eligible_new_entry_opportunities": float(
            validity["intervention"]["minimum_gate_eligible_new_entry_opportunities"]
        ),
    }


def _statistics_bootstrap(protocol: Mapping[str, Any]) -> dict[str, Any]:
    inference = protocol["statistics"]["inference"]
    return {
        "method_seed": int(inference["bootstrap_seed"]),
        "block_length_weeks": int(inference["block_length_weeks"]),
        "draws": int(inference["bootstrap_draws"]),
        "lower_quantile": float(inference["bootstrap_lower_quantile"]),
        "upper_quantile": float(inference["bootstrap_upper_quantile"]),
    }


def _closed_gate_report() -> dict[str, dict[str, Any]]:
    return {
        gate: {"status": "FAILED", "evidence_sha256": "0" * 64, "reason": "not_verified"} for gate in ENGINEERING_GATES
    }


def _gate_pass(gates: dict[str, dict[str, Any]], gate: str, evidence: Any) -> None:
    gates[gate] = {"status": "PASSED", "evidence_sha256": object_sha256(evidence), "reason": None}


def _gate_fail(gates: dict[str, dict[str, Any]], gate: str, reason: str, evidence: Any = None) -> None:
    gates[gate] = {
        "status": "FAILED",
        "evidence_sha256": object_sha256({"reason": reason, "evidence": evidence}),
        "reason": reason,
    }


def _base_evidence(*, protocol_sha256: str | None = None, replay_manifest_sha256: str | None = None) -> dict[str, Any]:
    return {
        "schema": FORMAL_EVIDENCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "trial_id": protocol_sha256,
        "replay_manifest_sha256": replay_manifest_sha256,
        "data_contract_identity_sha256": None,
        "genesis_data_manifest_sha256": None,
        "final_data_manifest_sha256": None,
        "final_data_chain_head_sha256": None,
        "data_chain_verification_sha256": None,
        "primary_chain_id": None,
        "registry_authority_sha256": None,
        "confirmation_window_identity_sha256": None,
        "readiness_report_sha256": None,
        "readiness_anchor_receipt_sha256": None,
        "formal_start_eligibility_verification_sha256": None,
        "engineering_gates": _closed_gate_report(),
        "data_report": None,
        "ledger_report": None,
        "planner_result_sha256": None,
        "execution_replay_sha256": None,
        "execution_market_binding_sha256": None,
        "ledger_record_binding_sha256": None,
        "semantic_replay_evidence_sha256": None,
        "final_evaluation_verified": False,
        "recomputed_evaluation_status": None,
        "statistics_result": None,
        "semantic_replay_verified": False,
        "confirmatory_oos": False,
        "status": "INVALID_DATA_OR_ENGINEERING",
        "alpha_validated": False,
        "live_trading_allowed": False,
        "issues": [],
    }


def _finalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence["status"] not in FORMAL_EVALUATION_STATUSES:
        raise ReplayVerificationError("verifier attempted to emit an unregistered formal status")
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    return {**body, "evidence_sha256": object_sha256(body)}


def _post_window_unwind_cost(
    results: Mapping[tuple[str, int | None, str], Mapping[str, Any]], confirmation_end: str
) -> float:
    components = ("commission_cny", "transfer_fee_cny", "stamp_duty_cny", "slippage_cost_cny")
    ordered_keys = sorted(
        results,
        key=lambda key: (key[0], -1 if key[1] is None else int(key[1]), key[2]),
    )
    return float(
        math.fsum(
            float(event.get(component, 0.0))
            for key in ordered_keys
            for event in results[key]["events"]
            if event.get("event_type") == "order" and str(event.get("session")) > confirmation_end
            for component in components
        )
    )


def _semantic_replay_identity(
    *,
    protocol_sha256: str,
    data_contract_identity_sha256: str,
    genesis_data_manifest_sha256: str,
    final_data_manifest_sha256: str,
    final_data_chain_head_sha256: str,
    data_chain_verification_sha256: str,
    primary_chain_id: str,
    window_sha256: str,
    readiness_report_sha256: str,
    readiness_anchor_receipt_sha256: str,
    ledger_index: LedgerConfirmationIndex,
    planner_result_sha256: str,
    execution_replay_sha256: str,
    execution_market_binding_sha256: str,
    ledger_record_binding_sha256: str,
    statistics_result_sha256: str,
    engineering_gates: Mapping[str, Any],
) -> str:
    final_hash = ledger_index.final_evaluation_record_sha256
    causal_records = [digest for digest in ledger_index.all_record_hashes if digest != final_hash]
    return object_sha256(
        {
            "schema": "xs_chan_semantic_replay_identity_v2_1",
            "protocol_sha256": protocol_sha256,
            "data_contract_identity_sha256": data_contract_identity_sha256,
            "genesis_data_manifest_sha256": genesis_data_manifest_sha256,
            "final_data_manifest_sha256": final_data_manifest_sha256,
            "final_data_chain_head_sha256": final_data_chain_head_sha256,
            "data_chain_verification_sha256": data_chain_verification_sha256,
            "primary_chain_id": primary_chain_id,
            "confirmation_window_identity_sha256": window_sha256,
            "readiness_report_sha256": readiness_report_sha256,
            "readiness_anchor_receipt_sha256": readiness_anchor_receipt_sha256,
            "causal_ledger_record_sha256": causal_records,
            "planner_result_sha256": planner_result_sha256,
            "execution_replay_sha256": execution_replay_sha256,
            "execution_market_binding_sha256": execution_market_binding_sha256,
            "ledger_record_binding_sha256": ledger_record_binding_sha256,
            "statistics_result_sha256": statistics_result_sha256,
            "engineering_gate_evidence_sha256": {
                gate: engineering_gates[gate]["evidence_sha256"] for gate in ENGINEERING_GATES
            },
        }
    )


def _verify_final_evaluation(
    payload: Mapping[str, Any],
    *,
    semantic_replay_evidence_sha256: str,
    statistics_result: Mapping[str, Any],
    window_sha256: str,
    ledger_index: LedgerConfirmationIndex,
    ledger_binding: Mapping[str, Any],
    results: Mapping[tuple[str, int | None, str], Mapping[str, Any]],
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != ledger_v21._FINAL_EVALUATION_KEYS:
        raise ReplayVerificationError("ledger final evaluation has an inexact schema")
    if payload.get("schema") != ledger_v21.FINAL_EVALUATION_SCHEMA:
        raise ReplayVerificationError("ledger final evaluation has the wrong schema")
    artifact_sha = _require_sha256(
        payload.get("final_evaluation_artifact_sha256"), "final_evaluation.final_evaluation_artifact_sha256"
    )
    artifact_body = dict(payload)
    artifact_body.pop("final_evaluation_artifact_sha256")
    if object_sha256(artifact_body) != artifact_sha:
        raise ReplayVerificationError("ledger final evaluation artifact self-digest differs")
    completion = ledger_binding["all_books"]["completion"]
    confirmation_end = str(ledger_index.covered_sessions[-1])
    unwind_cost = _post_window_unwind_cost(results, confirmation_end)
    unwind_data = ledger_index.event_data_by_record_hash[ledger_index.unwind_decision_record_sha256]["data"]
    unwind_evidence = object_sha256(
        {
            "decision_unwind_evidence_root_sha256": unwind_data["unwind_evidence_root_sha256"],
            "completion_book_state_root_sha256": completion["book_state_root_sha256"],
            "completion_pending_sells_root_sha256": completion["pending_sells_root_sha256"],
            "remaining_position_count": completion["remaining_position_count"],
            "remaining_pending_sell_count": completion["remaining_pending_sell_count"],
            "remaining_pending_settlement_count": completion["remaining_pending_settlement_count"],
            "completion_pending_settlements_root_sha256": completion["pending_settlements_root_sha256"],
            "unwind_session_count": len(ledger_index.unwind_sessions),
            "post_window_unwind_cost_cny": unwind_cost,
        }
    )
    expected = {
        "confirmation_window_identity_sha256": window_sha256,
        "statistics_result_sha256": statistics_result["result_sha256"],
        "semantic_replay_evidence_sha256": semantic_replay_evidence_sha256,
        "evaluation_status": statistics_result["overall_status"],
        "post_window_unwind_completed": True,
        "post_window_unwind_session_count": len(ledger_index.unwind_sessions),
        "post_window_unwind_cost_cny": unwind_cost,
        "remaining_position_count": completion["remaining_position_count"],
        "remaining_pending_sell_count": completion["remaining_pending_sell_count"],
        "remaining_pending_settlement_count": completion["remaining_pending_settlement_count"],
        "unwind_completion_book_state_root_sha256": completion["book_state_root_sha256"],
        "unwind_completion_pending_sells_root_sha256": completion["pending_sells_root_sha256"],
        "unwind_completion_pending_settlements_root_sha256": completion["pending_settlements_root_sha256"],
        "unwind_evidence_sha256": unwind_evidence,
    }
    if (
        completion["remaining_position_count"] != 0
        or completion["remaining_pending_sell_count"] != 0
        or completion["remaining_pending_settlement_count"] != 0
    ):
        raise ReplayVerificationError("post-window unwind is not complete in all 252 replayed books")
    last_eod_hash = ledger_index.eod_record_by_session[ledger_index.unwind_sessions[-1]]
    final_hash = ledger_index.final_evaluation_record_sha256
    if final_hash is None:
        raise ReplayVerificationError("final evaluation record identity is missing")
    cutoff = _parse_aware_utc(payload.get("snapshot_cutoff_utc"), "final_evaluation.snapshot_cutoff_utc")
    last_eod_recorded_at = _parse_aware_utc(
        ledger_index.recorded_at_utc_by_record_hash[last_eod_hash],
        "last unwind EOD recorded_at_utc",
    )
    final_recorded_at = _parse_aware_utc(
        ledger_index.recorded_at_utc_by_record_hash[final_hash],
        "final evaluation recorded_at_utc",
    )
    if cutoff < last_eod_recorded_at or cutoff > final_recorded_at:
        raise ReplayVerificationError("final evaluation cutoff is outside the last-unwind-EOD to final-record interval")
    _require_ledger_fields(payload, expected, "ledger final evaluation")


def verify_formal_replay(
    replay_manifest_path: str | Path,
    *,
    data_manifest_path: str | Path,
    ledger_root: str | Path,
    trial_registry_root: str | Path | None,
    receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
    registry_receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
    expected_registry_authority_sha256: str | None,
) -> dict[str, Any]:
    """Run the complete authority-bound semantic replay and return evidence.

    A missing real registry or receipt verifier is a formal chain failure, not a
    switch that callers can override with ``True``.
    """

    evidence = _base_evidence()
    try:
        protocol, protocol_sha = load_authoritative_protocol()
        evidence.update(protocol_sha256=protocol_sha, trial_id=protocol_sha)
        bundle = load_replay_bundle(replay_manifest_path)
        evidence["replay_manifest_sha256"] = bundle.manifest_sha256
    except Exception as exc:
        evidence["issues"].append(f"artifact authority boundary failed: {type(exc).__name__}: {exc}")
        return _finalize_evidence(evidence)

    gates = evidence["engineering_gates"]
    _gate_pass(gates, "artifact_semantic_verifier_engineering", bundle.manifest_sha256)

    try:
        actual_dependencies = _actual_dependency_sha256()
        if bundle.manifest["dependency_sha256"] != actual_dependencies:
            raise ReplayVerificationError("bundle dependency closure differs from the executing verifier")
    except Exception as exc:
        evidence["issues"].append(f"dependency closure failed: {type(exc).__name__}: {exc}")
        return _finalize_evidence(evidence)

    try:
        readiness = verify_readiness_report(
            bundle.read_json(bundle.manifest["readiness_report_object"]),
            expected_protocol_sha256=protocol_sha,
            expected_data_contract_identity_sha256=bundle.manifest["data_contract_identity_sha256"],
            expected_dependencies=actual_dependencies,
        )
        readiness_anchor = verify_readiness_anchor(
            bundle.read_json(bundle.manifest["readiness_anchor_receipt_object"]),
            readiness_report=readiness,
            receipt_verifier=receipt_verifier,
        )
        evidence["readiness_report_sha256"] = readiness["report_sha256"]
        evidence["readiness_anchor_receipt_sha256"] = readiness_anchor["receipt_sha256"]
    except Exception as exc:
        evidence["issues"].append(f"readiness authority boundary failed: {type(exc).__name__}: {exc}")
        return _finalize_evidence(evidence)

    try:
        genesis_data_report, genesis_snapshot = data_v21.load_verified_snapshot(
            _data_manifest_path(data_manifest_path, bundle.manifest["genesis_data_manifest_sha256"])
        )
        data_report, snapshot = data_v21.load_verified_snapshot(data_manifest_path)
        final_manifest_path = _data_manifest_path(data_manifest_path, bundle.manifest["final_data_manifest_sha256"])
    except Exception as exc:
        evidence["issues"].append(f"data snapshot authority boundary failed: {type(exc).__name__}: {exc}")
        return _finalize_evidence(evidence)
    evidence["data_report"] = data_report.to_dict()
    evidence["data_contract_identity_sha256"] = bundle.manifest["data_contract_identity_sha256"]
    evidence["genesis_data_manifest_sha256"] = genesis_data_report.manifest_sha256
    evidence["final_data_manifest_sha256"] = data_report.manifest_sha256
    evidence["final_data_chain_head_sha256"] = bundle.manifest["final_data_chain_head_sha256"]
    if (
        genesis_snapshot is None
        or genesis_data_report.manifest_sha256 != bundle.manifest["genesis_data_manifest_sha256"]
        or snapshot is None
        or data_report.manifest_sha256 != bundle.manifest["final_data_manifest_sha256"]
        or Path(data_manifest_path).expanduser() != final_manifest_path
    ):
        evidence["issues"].append("Genesis/final authoritative data snapshots differ from replay manifest")
        return _finalize_evidence(evidence)
    try:
        state_recompute = _require_state_recompute(data_v21.verify_state_recompute(snapshot))
        _gate_pass(gates, "state_recompute_engineering", state_recompute)
    except Exception as exc:
        reason = f"independent state recompute failed: {type(exc).__name__}: {exc}"
        _gate_fail(gates, "state_recompute_engineering", reason)
        evidence["issues"].append(reason)
        return _finalize_evidence(evidence)

    data_gate_mapping = {
        "source_archive_reconciliation_engineering": ("source_archive_binding", "source_archive"),
        "suspension_reconciliation_engineering": ("universe_response_coverage", "suspension"),
        "open_auction_reconciliation_engineering": ("artifact_semantics", "open_auction"),
    }
    for engineering_gate, (data_gate, domain) in data_gate_mapping.items():
        if data_report.gates.get(data_gate) is True:
            _gate_pass(gates, engineering_gate, {"domain": domain, "report": data_report.to_dict()})
        else:
            _gate_fail(gates, engineering_gate, f"data gate {data_gate} failed")

    try:
        ledger_report = ledger_v21.verify_ledger(
            ledger_root,
            receipt_verifier=receipt_verifier,
            trial_registry_root=trial_registry_root,
            registry_receipt_verifier=registry_receipt_verifier,
        )
    except Exception as exc:
        evidence["issues"].append(f"primary ledger failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    evidence["ledger_report"] = ledger_report
    evidence["primary_chain_id"] = ledger_report.get("primary_chain_id")
    try:
        registry_authority_sha256 = _require_sha256(
            expected_registry_authority_sha256, "expected_registry_authority_sha256"
        )
    except Exception as exc:
        evidence["issues"].append(f"registry trust root failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    evidence["registry_authority_sha256"] = registry_authority_sha256
    window = ledger_report.get("confirmation_window")
    evidence["confirmation_window_identity_sha256"] = (
        window.get("identity_sha256") if isinstance(window, Mapping) else None
    )
    if ledger_report.get("primary_registry_verified") is True:
        _gate_pass(gates, "primary_chain_registry_engineering", ledger_report["primary_chain_id"])
    ledger_identity = (
        ledger_report.get("protocol_id") == PROTOCOL_ID
        and ledger_report.get("protocol_sha256") == protocol_sha
        and ledger_report.get("trial_id") == protocol_sha
        and ledger_report.get("data_bundle_sha256") == bundle.manifest["genesis_data_manifest_sha256"]
        and ledger_report.get("data_contract_identity_sha256") == bundle.manifest["data_contract_identity_sha256"]
        and ledger_report.get("genesis_data_chain_head_sha256")
        == ledger_report.get("genesis_config", {}).get("genesis_data_chain_head_sha256")
        and bundle.manifest["primary_chain_id"] == ledger_report.get("primary_chain_id")
        and ledger_report.get("genesis_config", {}).get("trial_registry_authority_sha256") == registry_authority_sha256
    )
    if ledger_report.get("valid") is not True or not ledger_identity:
        evidence["issues"].append("ledger lacks a real verified registry/receipt chain or has wrong identity")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    eligibility = ledger_report.get("formal_start_eligibility")
    try:
        if (
            not isinstance(eligibility, Mapping)
            or canonical_json_bytes(eligibility)
            != canonical_json_bytes(ledger_report.get("genesis_config", {}).get("formal_start_eligibility"))
            or eligibility.get("readiness_report_sha256") != readiness["report_sha256"]
            or eligibility.get("readiness_anchor_receipt_sha256") != readiness_anchor["receipt_sha256"]
            or _parse_aware_utc(
                eligibility.get("readiness_completed_at_utc"),
                "formal-start readiness_completed_at_utc",
            )
            != _parse_aware_utc(
                readiness_anchor["issued_at_utc"],
                "readiness anchor issued_at_utc",
            )
            or canonical_json_bytes(ledger_report.get("readiness_gates"))
            != canonical_json_bytes(readiness["engineering_gates"])
        ):
            raise ReplayVerificationError("ledger Genesis changes the externally anchored readiness commitment")
    except Exception as exc:
        evidence["issues"].append(f"Genesis readiness binding failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    lifecycle_status = ledger_report.get("lifecycle_status")
    if ledger_report.get("deadline_failure") is not None or lifecycle_status == "PRIMARY_CHAIN_TERMINATED_INVALID":
        evidence["issues"].append("primary ledger is terminal or failed by deadline")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    if ledger_report.get("structural_window_complete") is not True:
        if lifecycle_status != "PRIMARY_FORWARD_COLLECTION_ACTIVE":
            evidence["issues"].append(f"primary ledger has unknown incomplete status {lifecycle_status!r}")
            evidence["status"] = "INVALID_CHAIN"
            return _finalize_evidence(evidence)
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)
    if lifecycle_status != "PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY":
        evidence["issues"].append("completed primary ledger has an inconsistent operational status")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    if not isinstance(window, Mapping) or bundle.manifest["confirmation_window_identity_sha256"] != window.get(
        "identity_sha256"
    ):
        evidence["issues"].append("confirmation window identity differs from primary ledger")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)

    try:
        if (
            not isinstance(eligibility, Mapping)
            or canonical_json_bytes(eligibility)
            != canonical_json_bytes(ledger_report.get("genesis_config", {}).get("formal_start_eligibility"))
            or canonical_json_bytes(eligibility) != canonical_json_bytes(window.get("formal_start_eligibility"))
        ):
            raise ReplayVerificationError(
                "ledger report, Genesis, and confirmation window disagree on formal-start evidence"
            )
        if (
            window.get("protocol_sha256") != protocol_sha
            or window.get("trial_id") != protocol_sha
            or window.get("primary_chain_id") != ledger_report["primary_chain_id"]
            or window.get("data_bundle_sha256") != bundle.manifest["genesis_data_manifest_sha256"]
            or window.get("data_contract_identity_sha256") != bundle.manifest["data_contract_identity_sha256"]
            or window.get("genesis_data_chain_head_sha256") != ledger_report.get("genesis_data_chain_head_sha256")
        ):
            raise ReplayVerificationError("confirmation window changes protocol/trial/data/chain identity")
        if window.get("dependency_sha256") != actual_dependencies:
            raise ReplayVerificationError(
                "confirmation-window dependency closure differs; ledger genesis must add ledger_engine"
            )
        if canonical_json_bytes(window.get("readiness_gates")) != canonical_json_bytes(readiness["engineering_gates"]):
            raise ReplayVerificationError("ledger genesis readiness gates differ from the verified readiness report")
        if (
            window.get("readiness_report_sha256") != readiness["report_sha256"]
            or ledger_report.get("readiness_report_sha256") != readiness["report_sha256"]
        ):
            raise ReplayVerificationError("ledger genesis names a different readiness report")
        ledger_records = ledger_v21.read_ledger_records(ledger_root)
        ledger_index = _build_ledger_confirmation_index(ledger_records, window)
        if ledger_report.get("external_receipts_verified") is True:
            causal_receipts = [
                item for item in ledger_report["record_identities"] if item["record_type"] != "final_evaluation"
            ]
            _gate_pass(gates, "external_timestamp_anchor_engineering", causal_receipts)
        if _parse_aware_utc(readiness_anchor["issued_at_utc"], "readiness anchor issued_at_utc") >= _parse_aware_utc(
            ledger_index.genesis_recorded_at_utc, "ledger.genesis.recorded_at_utc"
        ):
            raise ReplayVerificationError("externally anchored readiness did not precede ledger Genesis")
    except Exception as exc:
        evidence["issues"].append(f"ledger authority binding failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)

    cycles = _registered_cycles(ledger_report)
    if len(cycles) != 52:
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)
    if ledger_index.unwind_decision_record_sha256 is None or not ledger_index.unwind_sessions:
        evidence["issues"].append("post-window real unwind has not started or lacks a completed session")
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)
    try:
        data_chain = verify_forward_data_chain(
            data_manifest_path=data_manifest_path,
            bundle_manifest=bundle.manifest,
            ledger_report=ledger_report,
            ledger_index=ledger_index,
        )
        snapshot = data_chain.final_snapshot
        evidence["data_chain_verification_sha256"] = data_chain.verification["verification_sha256"]
        _gate_pass(
            gates,
            "source_archive_reconciliation_engineering",
            {"final_data_report": data_report.to_dict(), "forward_data_chain": data_chain.verification},
        )
    except Exception as exc:
        evidence["issues"].append(f"forward data-chain verification failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)

    # Planner verification is wired through a strict module API.  Importing it
    # only here lets an incomplete checkout fail closed instead of weakening the
    # artifact boundary.
    try:
        import xs_chan_planner_v2_1 as planner_v21

        planner_input = bundle.read_json(bundle.manifest["planner_input_object"])
        planner_result = bundle.read_json(bundle.manifest["planner_result_object"])
        planner_snapshot_digests = set(planner_result.get("data_snapshot_sha256_sequence", []))
        weekly_snapshots = {digest: data_chain.snapshots_by_sha256[digest] for digest in planner_snapshot_digests}
        verified_planner = _require_verified_planner(
            planner_v21.verify_planning_batch_result(
                planner_result,
                planner_input,
                snapshots_by_sha256=weekly_snapshots,
            )
        )
        formal_start_verification = _verify_formal_start_eligibility(
            eligibility,
            planner_result,
            data_chain.genesis_snapshot,
            planner_v21,
            readiness,
            readiness_anchor,
        )
        planner_registry = _planner_decision_registry(planner_result, ledger_index)
        evidence["planner_result_sha256"] = object_sha256(verified_planner)
        evidence["formal_start_eligibility_verification_sha256"] = formal_start_verification["verification_sha256"]
        _gate_pass(
            gates,
            "feature_prefix_engineering",
            {"planner": verified_planner, "formal_start": formal_start_verification},
        )
    except Exception as exc:
        _gate_fail(gates, "feature_prefix_engineering", f"planner replay failed: {type(exc).__name__}: {exc}")
        evidence["issues"].append(gates["feature_prefix_engineering"]["reason"])
        return _finalize_evidence(evidence)

    pairs_by_key: dict[tuple[str, int | None, str], Mapping[str, Any]] = {}
    input_by_key: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    result_by_key: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    try:
        for pair in bundle.manifest["execution_pairs"]:
            key = (str(pair["family_id"]), pair["seed_id"], str(pair["scenario_id"]))
            if key in pairs_by_key:
                raise ReplayVerificationError(f"duplicate execution family/seed/scenario {key}")
            expected_arm = _expected_statistics_arm(key[0], key[2])
            if pair["statistics_arm_id"] != expected_arm:
                raise ReplayVerificationError(f"statistics arm mapping differs for {key}")
            execution_input = bundle.read_json(pair["input_object"])
            execution_result = bundle.read_json(pair["result_object"])
            _require_execution_pair_identity(
                execution_input,
                execution_result,
                family=key[0],
                seed=key[1],
                scenario=key[2],
            )
            if execution_input.get("protocol_sha256") != protocol_sha:
                raise ReplayVerificationError(f"execution input {key} has wrong protocol")
            if execution_input.get("data_snapshot_sha256") != snapshot.manifest_sha256:
                raise ReplayVerificationError(f"execution input {key} has wrong data snapshot")
            if execution_input.get("trial_id") != protocol_sha or execution_input.get("arm_id") != key[0]:
                raise ReplayVerificationError(f"execution input {key} has wrong trial/arm")
            if execution_input.get("scenario_id") != key[2]:
                raise ReplayVerificationError(f"execution input {key} has wrong scenario")
            multiplier = {"gross": 0.0, "1x": 1.0, "2x": 2.0, "capacity_1x": 1.0}[key[2]]
            if execution_input.get("cost_model") != _protocol_cost_input(protocol, multiplier):
                raise ReplayVerificationError(f"execution input {key} changes frozen costs")
            expected_capital = 100_000_000.0 if key[2] == "capacity_1x" else 10_000_000.0
            if float(execution_input.get("initial_capital_cny", -1)) != expected_capital:
                raise ReplayVerificationError(f"execution input {key} changes scenario capital")
            _bind_execution_decisions_to_planner(
                execution_input,
                family=key[0],
                seed=key[1],
                scenario=key[2],
                planner_registry=planner_registry,
            )
            replayed = execution_v21.verify_execution_result(execution_result, execution_input)
            _require_execution_timeline(execution_input, replayed, ledger_index)
            pairs_by_key[key] = pair
            input_by_key[key] = execution_input
            result_by_key[key] = replayed
        if set(pairs_by_key) != _expected_pair_keys():
            raise ReplayVerificationError("execution scenario closure is incomplete or contains extras")
        planner_continuity = _verify_planner_state_continuity(planner_result, result_by_key, ledger_index)
        _gate_pass(
            gates,
            "feature_prefix_engineering",
            {"planner_verification": verified_planner, "state_continuity": planner_continuity},
        )
        market_binding = _require_execution_market_binding(
            data_v21.verify_execution_market_inputs(
                snapshot,
                [input_by_key[key] for key in sorted(input_by_key, key=str)],
            ),
            expected_input_count=len(_expected_pair_keys()),
        )
        if market_binding["data_snapshot_sha256"] != snapshot.manifest_sha256:
            raise ReplayVerificationError("execution market binding names a different verified snapshot")
        for family in SINGLE_FAMILIES:
            execution_v21.replay_cost_scenarios(
                {
                    scenario: bundle.read_json(pairs_by_key[(family, None, scenario)]["input_object"])
                    for scenario in SCENARIO_IDS
                }
            )
        for family in SEEDED_FAMILIES:
            for seed in statistics_v21.EXPECTED_RANDOM_SEEDS:
                execution_v21.replay_cost_scenarios(
                    {
                        scenario: bundle.read_json(pairs_by_key[(family, seed, scenario)]["input_object"])
                        for scenario in SCENARIO_IDS
                    }
                )
        market_ledger_binding = _verify_common_market_record_bindings(input_by_key, result_by_key, ledger_index)
        all_book_ledger_binding = _verify_all_book_ledger_bindings(
            input_by_key,
            result_by_key,
            planner_result,
            cycles,
            ledger_index,
            data_chain=data_chain,
            dependency_sha256=actual_dependencies,
        )
        if all_book_ledger_binding["initial_state_root_sha256"] != ledger_report["initial_state_root_sha256"]:
            raise ReplayVerificationError("execution initial 63×4 state root differs from ledger Genesis")
        if all_book_ledger_binding["completion"]["book_state_root_sha256"] != ledger_report["book_state_root_sha256"]:
            raise ReplayVerificationError("execution completion state root differs from the ledger head")
        if (
            all_book_ledger_binding["completion"]["pending_sells_root_sha256"]
            != ledger_report["pending_sells_root_sha256"]
        ):
            raise ReplayVerificationError("execution completion pending-sell root differs from the ledger head")
        ledger_binding = {
            "market": market_ledger_binding,
            "all_books": all_book_ledger_binding,
        }
        ledger_binding["binding_sha256"] = object_sha256(ledger_binding)
    except Exception as exc:
        reason = f"execution replay failed: {type(exc).__name__}: {exc}"
        _gate_fail(gates, "execution_replay_engineering", reason)
        _gate_fail(gates, "portfolio_accounting_replay_engineering", reason)
        _gate_fail(gates, "corporate_action_replay_engineering", reason)
        evidence["issues"].append(reason)
        return _finalize_evidence(evidence)

    execution_identity = object_sha256(
        {str(key): result_by_key[key]["result_sha256"] for key in sorted(result_by_key, key=str)}
    )
    evidence["execution_replay_sha256"] = execution_identity
    evidence["execution_market_binding_sha256"] = market_binding["verification_sha256"]
    evidence["ledger_record_binding_sha256"] = ledger_binding["binding_sha256"]
    _gate_pass(gates, "execution_replay_engineering", execution_identity)
    _gate_pass(
        gates,
        "portfolio_accounting_replay_engineering",
        {str(key): result_by_key[key]["final_state_sha256"] for key in sorted(result_by_key, key=str)},
    )
    _gate_pass(
        gates,
        "corporate_action_replay_engineering",
        {
            str(key): [
                event
                for event in result_by_key[key]["events"]
                if event["event_type"] in {"corporate_action", "cash_receivable_payment"}
            ]
            for key in sorted(result_by_key, key=str)
        },
    )
    _gate_pass(
        gates,
        "calendar_append_verifier_engineering",
        {
            "ledger_calendar_state_sha256": ledger_report["calendar_state_sha256"],
            "official_session_timeline": list(ledger_index.expected_daily_sessions),
            "execution_market_binding_sha256": market_binding["verification_sha256"],
        },
    )

    try:
        components: dict[tuple[str, int | None, str], dict[str, Any]] = {}
        for key, result in result_by_key.items():
            state_observations = _state_gate_observations(planner_result, result, cycles) if key[0] == "FC" else []
            components[key] = execution_v21.build_weekly_path_component(
                result,
                cycles,
                state_gate_observations=state_observations,
            )
        if not ledger_index.unwind_sessions:
            raise ReplayVerificationError("statistics path construction requires an observed post-window unwind")
        valid_completion_sessions = {ledger_index.covered_sessions[-1], *ledger_index.unwind_sessions}
        completion_session_by_book = {
            key: str(component["path"]["post_window_unwind"]["completion_session"])
            for key, component in components.items()
        }
        invalid_completion_sessions = {
            str(key): session
            for key, session in completion_session_by_book.items()
            if session not in valid_completion_sessions
        }
        if invalid_completion_sessions:
            raise ReplayVerificationError(
                "one or more book unwind completion sessions are outside the causal ledger timeline"
            )
        aggregate_completion_session = max(
            ledger_index.unwind_sessions[0],
            max(completion_session_by_book.values()),
        )
        if aggregate_completion_session != ledger_index.unwind_sessions[-1]:
            raise ReplayVerificationError(
                "ledger unwind timeline does not end at the first aggregate all-book completion session"
            )
        arm_paths: dict[str, Any] = {}
        for family in statistics_v21.EXPECTED_RISK_FAMILIES:
            for scenario in statistics_v21.EXPECTED_RISK_SCENARIOS:
                arm_id = f"{family}_{scenario}"
                seeded_family = family in SEEDED_FAMILIES
                if not seeded_family:
                    component = components[(family, None, scenario)]
                    arm_paths[arm_id] = {
                        "schema": statistics_v21.WEEKLY_PATH_SCHEMA,
                        "arm_id": arm_id,
                        "week_labels": component["week_labels"],
                        "daily_labels": component["daily_labels"],
                        "aggregation": "single_path",
                        "paths": {"primary": component["path"]},
                    }
                    continue
                seeded = {
                    str(seed): components[(family, seed, scenario)] for seed in statistics_v21.EXPECTED_RANDOM_SEEDS
                }
                first = seeded[str(statistics_v21.EXPECTED_RANDOM_SEEDS[0])]
                if any(
                    component["week_labels"] != first["week_labels"]
                    or component["daily_labels"] != first["daily_labels"]
                    for component in seeded.values()
                ):
                    raise ReplayVerificationError(f"seeded path labels differ for {arm_id}")
                arm_paths[arm_id] = {
                    "schema": statistics_v21.WEEKLY_PATH_SCHEMA,
                    "arm_id": arm_id,
                    "week_labels": first["week_labels"],
                    "daily_labels": first["daily_labels"],
                    "aggregation": "equal_mean_all_20_frozen_seeds_before_statistics",
                    "paths": {seed: component["path"] for seed, component in seeded.items()},
                }
        if tuple(arm_paths) != statistics_v21.EXPECTED_RISK_ARM_IDS:
            raise ReplayVerificationError("statistics path construction differs from the exact 24-arm registry")
        expected_statistics_input = {
            "schema": statistics_v21.STATISTICS_INPUT_SCHEMA,
            "protocol_sha256": protocol_sha,
            "trial_id": protocol_sha,
            "window_sha256": window["identity_sha256"],
            "comparison_registry": protocol["statistics"]["comparison_registry"],
            "arm_paths": arm_paths,
            "controls": _statistics_controls(protocol),
            "bootstrap": _statistics_bootstrap(protocol),
        }
        stored_statistics_input = bundle.read_json(bundle.manifest["statistics_input_object"])
        if canonical_json_bytes(stored_statistics_input) != canonical_json_bytes(expected_statistics_input):
            raise ReplayVerificationError("stored statistics input differs from execution-derived paths")
        stored_statistics_result = bundle.read_json(bundle.manifest["statistics_result_object"])
        statistics_result = statistics_v21.verify_statistics_result(stored_statistics_result, expected_statistics_input)
    except Exception as exc:
        reason = f"statistics replay failed: {type(exc).__name__}: {exc}"
        _gate_fail(gates, "statistics_replay_engineering", reason)
        evidence["issues"].append(reason)
        return _finalize_evidence(evidence)
    _gate_pass(gates, "statistics_replay_engineering", statistics_result)
    evidence["statistics_result"] = statistics_result

    all_engineering_passed = all(gates[gate]["status"] == "PASSED" for gate in ENGINEERING_GATES)
    if not all_engineering_passed:
        evidence["issues"].append("one or more verifier-derived engineering gates remain closed")
        evidence["status"] = "INVALID_DATA_OR_ENGINEERING"
        return _finalize_evidence(evidence)

    semantic_identity = _semantic_replay_identity(
        protocol_sha256=protocol_sha,
        data_contract_identity_sha256=data_chain.verification["data_contract_identity_sha256"],
        genesis_data_manifest_sha256=data_chain.verification["genesis_data_manifest_sha256"],
        final_data_manifest_sha256=data_chain.verification["final_data_manifest_sha256"],
        final_data_chain_head_sha256=data_chain.verification["final_data_chain_head_sha256"],
        data_chain_verification_sha256=data_chain.verification["verification_sha256"],
        primary_chain_id=ledger_report["primary_chain_id"],
        window_sha256=window["identity_sha256"],
        readiness_report_sha256=readiness["report_sha256"],
        readiness_anchor_receipt_sha256=readiness_anchor["receipt_sha256"],
        ledger_index=ledger_index,
        planner_result_sha256=evidence["planner_result_sha256"],
        execution_replay_sha256=execution_identity,
        execution_market_binding_sha256=market_binding["verification_sha256"],
        ledger_record_binding_sha256=ledger_binding["binding_sha256"],
        statistics_result_sha256=statistics_result["result_sha256"],
        engineering_gates=gates,
    )
    evidence["semantic_replay_evidence_sha256"] = semantic_identity
    evidence["recomputed_evaluation_status"] = statistics_result["overall_status"]
    evidence["semantic_replay_verified"] = True
    final_evaluation = ledger_report.get("final_evaluation")
    if final_evaluation is None:
        evidence["issues"].append(
            "semantic replay passed; an externally anchored final evaluation record is still required"
        )
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)
    try:
        _verify_final_evaluation(
            final_evaluation,
            semantic_replay_evidence_sha256=semantic_identity,
            statistics_result=statistics_result,
            window_sha256=window["identity_sha256"],
            ledger_index=ledger_index,
            ledger_binding=ledger_binding,
            results=result_by_key,
        )
    except Exception as exc:
        evidence["issues"].append(f"final evaluation verification failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    evidence["final_evaluation_verified"] = True
    evidence["confirmatory_oos"] = True
    evidence["status"] = statistics_result["overall_status"]
    return _finalize_evidence(evidence)


def verify_evidence_digest(value: Mapping[str, Any]) -> None:
    """Check evidence self-digest; this alone never establishes confirmation."""

    if value.get("schema") != FORMAL_EVIDENCE_SCHEMA:
        raise ReplayVerificationError("not a V2.1 formal evidence object")
    digest = _require_sha256(value.get("evidence_sha256"), "evidence_sha256")
    body = dict(value)
    body.pop("evidence_sha256")
    if object_sha256(body) != digest:
        raise ReplayVerificationError("formal evidence self-digest differs")


def evaluate_v2_1(
    replay_manifest_path: str | Path,
    *,
    data_manifest_path: str | Path,
    ledger_root: str | Path,
    trial_registry_root: str | Path | None,
    receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
    registry_receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
    expected_registry_authority_sha256: str | None,
) -> dict[str, Any]:
    """Public V2.1 evaluation entry point; always performs semantic replay."""

    evidence = verify_formal_replay(
        replay_manifest_path,
        data_manifest_path=data_manifest_path,
        ledger_root=ledger_root,
        trial_registry_root=trial_registry_root,
        receipt_verifier=receipt_verifier,
        registry_receipt_verifier=registry_receipt_verifier,
        expected_registry_authority_sha256=expected_registry_authority_sha256,
    )
    return {
        "status": evidence["status"],
        "alpha_validated": False,
        "live_trading_allowed": False,
        "semantic_replay_verified": evidence["semantic_replay_verified"],
        "confirmatory_oos": evidence["confirmatory_oos"],
        "evidence_sha256": evidence["evidence_sha256"],
        "issues": list(evidence["issues"]),
    }


__all__ = [
    "DATA_CHAIN_VERIFICATION_SCHEMA",
    "ENGINEERING_GATES",
    "FORMAL_EVALUATION_STATUSES",
    "FORMAL_EVIDENCE_SCHEMA",
    "FrozenReplayBundle",
    "REPLAY_MANIFEST_SCHEMA",
    "ReplayVerificationError",
    "VerifiedForwardDataChain",
    "canonical_json_bytes",
    "evaluate_v2_1",
    "load_authoritative_protocol",
    "load_replay_bundle",
    "object_sha256",
    "verify_evidence_digest",
    "verify_forward_data_chain",
    "verify_formal_replay",
    "verify_readiness_anchor",
]
