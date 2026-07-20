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
REPLAY_MANIFEST_VERSION = 1
FORMAL_EVIDENCE_SCHEMA = "xs_chan_formal_evidence_v2_1"
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
        "data_manifest_sha256",
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
    for field in ("data_manifest_sha256", "primary_chain_id", "confirmation_window_identity_sha256"):
        _require_sha256(manifest[field], field)
    try:
        created = datetime.fromisoformat(str(manifest["created_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayVerificationError("created_at_utc must be timezone-aware ISO-8601") from exc
    if created.tzinfo is None or created.utcoffset() is None or created.astimezone(UTC) > datetime.now(UTC):
        raise ReplayVerificationError("created_at_utc is naive or in the future")
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

    scalar_references = {
        manifest["planner_input_object"],
        manifest["planner_result_object"],
        manifest["statistics_input_object"],
        manifest["statistics_result_object"],
        manifest["readiness_report_object"],
    }
    if any(not isinstance(name, str) for name in scalar_references):
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
    referenced = scalar_references | pair_references
    if referenced != set(objects):
        raise ReplayVerificationError(
            f"object reference closure differs: unreferenced={sorted(set(objects) - referenced)}, "
            f"missing={sorted(referenced - set(objects))}"
        )
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
        "data_manifest_sha256": None,
        "primary_chain_id": None,
        "confirmation_window_identity_sha256": None,
        "engineering_gates": _closed_gate_report(),
        "data_report": None,
        "ledger_report": None,
        "planner_result_sha256": None,
        "execution_replay_sha256": None,
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


def verify_formal_replay(
    replay_manifest_path: str | Path,
    *,
    data_manifest_path: str | Path,
    ledger_root: str | Path,
    trial_registry_root: str | Path | None,
    receipt_verifier: ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier | None,
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

    data_report, snapshot = data_v21.load_verified_snapshot(data_manifest_path)
    evidence["data_report"] = data_report.to_dict()
    evidence["data_manifest_sha256"] = data_report.manifest_sha256
    if snapshot is None or data_report.manifest_sha256 != bundle.manifest["data_manifest_sha256"]:
        evidence["issues"].append("authoritative data snapshot is invalid or differs from replay manifest")
        return _finalize_evidence(evidence)
    data_gate_mapping = {
        "state_recompute_engineering": ("state_input_row_binding", "state_audit"),
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
        )
    except Exception as exc:
        evidence["issues"].append(f"primary ledger failed: {type(exc).__name__}: {exc}")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    evidence["ledger_report"] = ledger_report
    evidence["primary_chain_id"] = ledger_report.get("primary_chain_id")
    window = ledger_report.get("confirmation_window")
    evidence["confirmation_window_identity_sha256"] = (
        window.get("identity_sha256") if isinstance(window, Mapping) else None
    )
    if ledger_report.get("calendar_segment_count", 0) > 0:
        _gate_pass(gates, "calendar_append_verifier_engineering", ledger_report["calendar_state_sha256"])
    if ledger_report.get("primary_registry_verified") is True:
        _gate_pass(gates, "primary_chain_registry_engineering", ledger_report["primary_chain_id"])
    if ledger_report.get("external_receipts_verified") is True:
        _gate_pass(gates, "external_timestamp_anchor_engineering", ledger_report["chain_head_sha256"])
    if ledger_report.get("valid") is not True or bundle.manifest["primary_chain_id"] != ledger_report.get(
        "primary_chain_id"
    ):
        evidence["issues"].append("ledger lacks a real verified registry/receipt chain or has wrong identity")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)
    if ledger_report.get("structural_window_complete") is not True:
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)
    if not isinstance(window, Mapping) or bundle.manifest["confirmation_window_identity_sha256"] != window.get(
        "identity_sha256"
    ):
        evidence["issues"].append("confirmation window identity differs from primary ledger")
        evidence["status"] = "INVALID_CHAIN"
        return _finalize_evidence(evidence)

    cycles = _registered_cycles(ledger_report)
    if len(cycles) != 52:
        evidence["status"] = "FORWARD_COLLECTION_REQUIRED"
        return _finalize_evidence(evidence)

    # Planner verification is wired through a strict module API.  Importing it
    # only here lets an incomplete checkout fail closed instead of weakening the
    # artifact boundary.
    try:
        import xs_chan_planner_v2_1 as planner_v21

        planner_input = bundle.read_json(bundle.manifest["planner_input_object"])
        planner_result = bundle.read_json(bundle.manifest["planner_result_object"])
        verified_planner = planner_v21.verify_planner_result(planner_result, planner_input, snapshot=snapshot)
        evidence["planner_result_sha256"] = object_sha256(verified_planner)
        _gate_pass(gates, "feature_prefix_engineering", verified_planner)
    except Exception as exc:
        _gate_fail(gates, "feature_prefix_engineering", f"planner replay failed: {type(exc).__name__}: {exc}")
        evidence["issues"].append(gates["feature_prefix_engineering"]["reason"])
        return _finalize_evidence(evidence)

    pairs_by_key: dict[tuple[str, int | None, str], Mapping[str, Any]] = {}
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
            replayed = execution_v21.verify_execution_result(execution_result, execution_input)
            pairs_by_key[key] = pair
            result_by_key[key] = replayed
        if set(pairs_by_key) != _expected_pair_keys():
            raise ReplayVerificationError("execution scenario closure is incomplete or contains extras")
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

    try:
        components = {
            key: execution_v21.build_weekly_path_component(result, cycles) for key, result in result_by_key.items()
        }
        arm_paths: dict[str, Any] = {}
        for arm_id, (family, seed_mode, scenario) in CONFIRMATORY_ARM_SOURCE.items():
            if seed_mode is None:
                component = components[(family, None, scenario)]
                arm_paths[arm_id] = {
                    "schema": statistics_v21.WEEKLY_PATH_SCHEMA,
                    "arm_id": arm_id,
                    "week_labels": component["week_labels"],
                    "daily_labels": component["daily_labels"],
                    "aggregation": "single_path",
                    "paths": {"primary": component["path"]},
                }
            else:
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

    evidence["semantic_replay_verified"] = True
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
) -> dict[str, Any]:
    """Public V2.1 evaluation entry point; always performs semantic replay."""

    evidence = verify_formal_replay(
        replay_manifest_path,
        data_manifest_path=data_manifest_path,
        ledger_root=ledger_root,
        trial_registry_root=trial_registry_root,
        receipt_verifier=receipt_verifier,
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
    "ENGINEERING_GATES",
    "FORMAL_EVALUATION_STATUSES",
    "FORMAL_EVIDENCE_SCHEMA",
    "FrozenReplayBundle",
    "REPLAY_MANIFEST_SCHEMA",
    "ReplayVerificationError",
    "canonical_json_bytes",
    "evaluate_v2_1",
    "load_authoritative_protocol",
    "load_replay_bundle",
    "object_sha256",
    "verify_evidence_digest",
    "verify_formal_replay",
]
