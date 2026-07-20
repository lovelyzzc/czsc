from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import xs_chan_replay_verifier_v2_1 as replay  # noqa: E402

EXPECTED_PROTOCOL_SHA256 = "8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203"


def _write_object(root: Path, value: dict[str, Any]) -> tuple[str, int]:
    raw = replay.canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path = root / "objects" / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return digest, len(raw)


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    logical = {
        "planner/input": {"kind": "planner-input"},
        "planner/result": {"kind": "planner-result"},
        "statistics/input": {"kind": "statistics-input"},
        "statistics/result": {"kind": "statistics-result"},
        "readiness/report": {"kind": "readiness"},
        "execution/F/input": {"kind": "execution-input"},
        "execution/F/result": {"kind": "execution-result"},
    }
    objects: dict[str, Any] = {}
    for name, value in logical.items():
        digest, size = _write_object(tmp_path, value)
        objects[name] = {
            "sha256": digest,
            "size_bytes": size,
            "media_type": replay.JSON_MEDIA_TYPE,
        }
    manifest = {
        "schema": replay.REPLAY_MANIFEST_SCHEMA,
        "manifest_version": replay.REPLAY_MANIFEST_VERSION,
        "protocol_id": replay.PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "trial_id": EXPECTED_PROTOCOL_SHA256,
        "data_manifest_sha256": "1" * 64,
        "primary_chain_id": "2" * 64,
        "confirmation_window_identity_sha256": "3" * 64,
        "created_at_utc": "2026-07-20T00:00:00Z",
        "dependency_sha256": dict.fromkeys(replay.EXPECTED_DEPENDENCIES, "4" * 64),
        "objects": objects,
        "execution_pairs": [
            {
                "pair_id": "F:@2x",
                "family_id": "F",
                "seed_id": None,
                "scenario_id": "2x",
                "statistics_arm_id": "F_2x",
                "input_object": "execution/F/input",
                "result_object": "execution/F/result",
            }
        ],
        "planner_input_object": "planner/input",
        "planner_result_object": "planner/result",
        "statistics_input_object": "statistics/input",
        "statistics_result_object": "statistics/result",
        "readiness_report_object": "readiness/report",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(replay.canonical_json_bytes(manifest))
    return manifest_path, manifest


def test_authoritative_protocol_uses_canonical_not_raw_file_hash() -> None:
    protocol, digest = replay.load_authoritative_protocol()
    assert protocol["protocol_id"] == replay.PROTOCOL_ID
    assert digest == EXPECTED_PROTOCOL_SHA256
    assert digest != hashlib.sha256((SCRIPTS / "xs_chan_protocol_v2_1.json").read_bytes()).hexdigest()


def test_content_addressed_bundle_loads_only_exact_reference_closure(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    frozen = replay.load_replay_bundle(path)
    assert frozen.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen.read_json("planner/input") == {"kind": "planner-input"}

    extra = copy.deepcopy(manifest)
    digest, size = _write_object(tmp_path, {"unreferenced": True})
    extra["objects"]["orphan/object"] = {
        "sha256": digest,
        "size_bytes": size,
        "media_type": replay.JSON_MEDIA_TYPE,
    }
    path.write_bytes(replay.canonical_json_bytes(extra))
    with pytest.raises(replay.ReplayVerificationError, match="reference closure"):
        replay.load_replay_bundle(path)


def test_object_digest_size_and_canonical_encoding_are_all_enforced(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    descriptor = manifest["objects"]["planner/input"]
    object_path = tmp_path / "objects" / descriptor["sha256"][:2] / descriptor["sha256"]
    object_path.write_bytes(object_path.read_bytes() + b" ")
    with pytest.raises(replay.ReplayVerificationError, match="size"):
        replay.load_replay_bundle(path)

    path, manifest = _bundle(tmp_path)
    descriptor = manifest["objects"]["planner/input"]
    object_path = tmp_path / "objects" / descriptor["sha256"][:2] / descriptor["sha256"]
    object_path.unlink()
    object_path.symlink_to(tmp_path / "manifest.json")
    with pytest.raises(replay.ReplayVerificationError, match="securely open"):
        replay.load_replay_bundle(path)


def test_duplicate_json_keys_fail_before_any_semantic_use(tmp_path: Path) -> None:
    path, _ = _bundle(tmp_path)
    path.write_bytes(b'{"schema":"x","schema":"y"}')
    with pytest.raises(replay.ReplayVerificationError, match="duplicate JSON key"):
        replay.load_replay_bundle(path)


def test_evidence_self_digest_is_structural_only_and_forgery_is_rejected() -> None:
    evidence = {
        "schema": replay.FORMAL_EVIDENCE_SCHEMA,
        "status": "INVALID_DATA_OR_ENGINEERING",
        "semantic_replay_verified": False,
    }
    evidence["evidence_sha256"] = replay.object_sha256(evidence)
    replay.verify_evidence_digest(evidence)
    evidence["semantic_replay_verified"] = True
    with pytest.raises(replay.ReplayVerificationError, match="self-digest"):
        replay.verify_evidence_digest(evidence)


def test_public_evaluation_fails_closed_without_real_data_chain_or_receipts(tmp_path: Path) -> None:
    result = replay.evaluate_v2_1(
        tmp_path / "missing-replay.json",
        data_manifest_path=tmp_path / "missing-data.json",
        ledger_root=tmp_path / "missing-ledger",
        trial_registry_root=None,
        receipt_verifier=None,
    )
    assert result["status"] == "INVALID_DATA_OR_ENGINEERING"
    assert result["semantic_replay_verified"] is False
    assert result["confirmatory_oos"] is False
    assert result["alpha_validated"] is False
    assert result["live_trading_allowed"] is False


def test_manifest_cannot_substitute_raw_protocol_hash_for_trial_identity(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    raw_hash = hashlib.sha256((SCRIPTS / "xs_chan_protocol_v2_1.json").read_bytes()).hexdigest()
    manifest["protocol_sha256"] = raw_hash
    manifest["trial_id"] = raw_hash
    path.write_bytes(replay.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayVerificationError, match="canonical V2.1 protocol"):
        replay.load_replay_bundle(path)
