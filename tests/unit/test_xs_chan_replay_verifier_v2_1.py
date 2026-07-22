from __future__ import annotations

import copy
import hashlib
import sys
from datetime import UTC, date, datetime, timedelta
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
        "readiness/anchor": {"kind": "readiness-anchor"},
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
        "data_contract_identity_sha256": replay.data_v21.data_contract_identity_sha256(),
        "genesis_data_manifest_sha256": "1" * 64,
        "final_data_manifest_sha256": "1" * 64,
        "final_data_chain_head_sha256": "5" * 64,
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
        "readiness_anchor_receipt_object": "readiness/anchor",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(replay.canonical_json_bytes(manifest))
    return manifest_path, manifest


def _readiness(*, dependencies: dict[str, str] | None = None) -> dict[str, Any]:
    value = {
        "schema": replay.READINESS_REPORT_SCHEMA,
        "protocol_id": replay.PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "trial_id": EXPECTED_PROTOCOL_SHA256,
        "data_contract_identity_sha256": replay.data_v21.data_contract_identity_sha256(),
        "created_at_utc": "2026-07-19T00:00:00Z",
        "dependency_sha256": dependencies or dict.fromkeys(replay.EXPECTED_DEPENDENCIES, "4" * 64),
        "engineering_gates": {
            gate: {"status": "PASSED", "evidence_sha256": hashlib.sha256(gate.encode()).hexdigest()}
            for gate in replay.ENGINEERING_GATES
        },
    }
    value["report_sha256"] = replay.object_sha256(value)
    return value


def _planner_verification(*, valid: bool = True) -> dict[str, Any]:
    value = {
        "schema": "xs_chan_planning_batch_verification_v2_1",
        "valid": valid,
        "batch_input_sha256": "1" * 64,
        "batch_artifact_sha256": "2" * 64,
        "cycle_count": 52,
        "execution_decision_count": 52 * 252,
        "cycle_verification_sha256": [hashlib.sha256(f"cycle:{index}".encode()).hexdigest() for index in range(52)],
        "errors": [] if valid else ["semantic replay failed"],
    }
    value["verification_sha256"] = replay.object_sha256(value)
    return value


def _state_recompute_verification(*, valid: bool = True) -> dict[str, Any]:
    value = {
        "schema": "xs_chan_state_recompute_verification_v2_1",
        "valid": valid,
        "source_symbol_count": 5_719,
        "recomputed_symbol_count": 5_719,
        "required_prefix_audit_symbol_count": 100,
        "prefix_audit_symbol_count": 100,
        "cutoffs_per_prefix_audit_symbol": 20,
        "mismatch_count": 0 if valid else 1,
        "evidence_sha256": "3" * 64,
        "errors": [] if valid else ["mismatch"],
    }
    value["verification_sha256"] = replay.data_v21.sha256_bytes(replay.data_v21.canonical_json_bytes(value))
    return value


def _ledger_record(sequence: int, record_type: str, event: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(f"record:{sequence}:{record_type}".encode()).hexdigest()
    recorded_at_utc = f"2025-01-{sequence % 28 + 1:02d}T00:00:00Z"
    payload: dict[str, Any]
    if record_type == "genesis":
        payload = {**event, "external_anchor": {"issued_at_utc": recorded_at_utc}}
    else:
        payload = {
            "event": event,
            "record_commitment_sha256": hashlib.sha256(f"commitment:{sequence}".encode()).hexdigest(),
            "external_anchor": {"issued_at_utc": recorded_at_utc},
        }
    return {
        "schema_version": 21,
        "sequence": sequence,
        "record_type": record_type,
        "recorded_at_utc": recorded_at_utc,
        "previous_record_hash": "0" * 64,
        "payload": payload,
        "record_hash": digest,
    }


def _ledger_confirmation_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [_ledger_record(0, "genesis", {"schema": "genesis"})]
    weeks: list[dict[str, Any]] = []
    prior_close = date(2025, 1, 3)
    for week_index in range(52):
        close = prior_close + timedelta(days=7)
        decision_dt = prior_close.isoformat()
        session = close.isoformat()
        decision = _ledger_record(
            len(records),
            "decision",
            {
                "decision_dt": decision_dt,
                "execution_dt": session,
                "data": {"decision_artifact_sha256": hashlib.sha256(f"artifact:{week_index}".encode()).hexdigest()},
            },
        )
        records.append(decision)
        records.append(_ledger_record(len(records), "session_open_execution", {"session_dt": session, "data": {}}))
        records.append(_ledger_record(len(records), "session_eod_valuation", {"session_dt": session, "data": {}}))
        close_record = _ledger_record(len(records), "cycle_close", {"close_dt": session, "data": {}})
        records.append(close_record)
        weeks.append(
            {
                "decision_dt": decision_dt,
                "execution_dt": session,
                "close_dt": session,
                "sessions": [session],
                "decision_record_sha256": decision["record_hash"],
                "weekly_close_record_sha256": close_record["record_hash"],
            }
        )
        prior_close = close
    hashes = [record["record_hash"] for record in records]
    window = {
        "weeks": weeks,
        "record_hashes_through_52nd_close": hashes,
        "confirmation_head_sha256": hashes[-1],
    }
    return records, window


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
        registry_receipt_verifier=None,
        expected_registry_authority_sha256=None,
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


def test_manifest_cannot_change_static_data_contract_identity(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    manifest["data_contract_identity_sha256"] = "9" * 64
    path.write_bytes(replay.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayVerificationError, match="static V2.1 data contract"):
        replay.load_replay_bundle(path)


def test_replay_rederives_each_ledger_data_transition() -> None:
    contract = replay.data_v21.data_contract_identity_sha256()
    previous = "0" * 64
    decision_session = "2026-07-17"
    cutoff = "2026-07-17T07:00:00.000000Z"
    manifest_sha = "1" * 64
    report_sha = "2" * 64
    head = replay.data_v21.data_chain_transition_sha256(
        data_contract_identity_sha256=contract,
        previous_data_chain_head_sha256=previous,
        decision_session=decision_session,
        snapshot_cutoff_utc=cutoff,
        data_manifest_sha256=manifest_sha,
        data_validation_report_sha256=report_sha,
    )
    data = {
        "snapshot_cutoff_utc": cutoff,
        "data_manifest_sha256": manifest_sha,
        "data_validation_report_sha256": report_sha,
        "previous_data_chain_head_sha256": previous,
        "data_chain_head_sha256": head,
    }
    verified = replay._ledger_data_transition(
        contract_sha256=contract,
        previous_head=previous,
        decision_session=decision_session,
        data=data,
        label="test transition",
    )
    assert verified["data_chain_head_sha256"] == head

    forged = copy.deepcopy(data)
    forged["data_chain_head_sha256"] = "f" * 64
    with pytest.raises(replay.ReplayVerificationError, match="not canonically derived"):
        replay._ledger_data_transition(
            contract_sha256=contract,
            previous_head=previous,
            decision_session=decision_session,
            data=forged,
            label="test transition",
        )


def test_snapshot_observation_horizon_must_equal_its_ledger_transition_session() -> None:
    class Snapshot:
        observation_through_session = "2026-07-18"

    with pytest.raises(replay.ReplayVerificationError, match="causal ledger session"):
        replay._require_snapshot_observation(Snapshot(), "2026-07-17", "weekly snapshot")

    Snapshot.observation_through_session = "2026-07-17"
    assert replay._require_snapshot_observation(Snapshot(), "2026-07-17", "weekly snapshot") == "2026-07-17"


def test_forward_data_chain_replays_all_transitions_and_rejects_one_day_observation_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Snapshot:
        def __init__(self, manifest_sha256: str, session: str) -> None:
            self.manifest_sha256 = manifest_sha256
            self.observation_through_session = session
            self.created_at_utc = f"{session}T07:05:00Z"

    contract = replay.data_v21.data_contract_identity_sha256()
    genesis_session = date(2026, 1, 1)
    genesis_manifest = hashlib.sha256(b"forward-snapshot:genesis").hexdigest()
    genesis_report = replay.data_v21.DataValidationReport(
        valid=True,
        data_ready=True,
        manifest_sha256=genesis_manifest,
        gates=dict.fromkeys(replay.data_v21.FORMAL_GATES, True),
    )
    genesis_report_sha = replay.data_v21.data_validation_report_sha256(genesis_report)
    genesis_cutoff = f"{genesis_session.isoformat()}T07:00:00.000000Z"
    genesis_head = replay.data_v21.data_chain_transition_sha256(
        data_contract_identity_sha256=contract,
        previous_data_chain_head_sha256=replay.ledger_v21.ZERO_HASH,
        decision_session=genesis_session.isoformat(),
        snapshot_cutoff_utc=genesis_cutoff,
        data_manifest_sha256=genesis_manifest,
        data_validation_report_sha256=genesis_report_sha,
    )

    snapshots = {genesis_manifest: Snapshot(genesis_manifest, genesis_session.isoformat())}
    reports = {genesis_manifest: genesis_report}
    recorded_at: dict[str, str] = {}
    events: dict[str, dict[str, Any]] = {}
    all_record_hashes: list[str] = []
    genesis_record_hash = hashlib.sha256(b"forward-ledger:genesis").hexdigest()
    all_record_hashes.append(genesis_record_hash)
    recorded_at[genesis_record_hash] = f"{genesis_session.isoformat()}T07:10:00Z"
    events[genesis_record_hash] = {
        "primary_registry_external_anchor": {"issued_at_utc": f"{genesis_session.isoformat()}T07:06:00Z"},
        "external_anchor": {"issued_at_utc": f"{genesis_session.isoformat()}T07:08:00Z"},
    }
    previous_head = genesis_head

    def append_transition(index: int, session: date, record_role: str) -> tuple[str, dict[str, Any]]:
        nonlocal previous_head
        session_text = session.isoformat()
        manifest_sha = hashlib.sha256(f"forward-snapshot:{index}".encode()).hexdigest()
        report = replay.data_v21.DataValidationReport(
            valid=True,
            data_ready=True,
            manifest_sha256=manifest_sha,
            gates=dict.fromkeys(replay.data_v21.FORMAL_GATES, True),
        )
        report_sha = replay.data_v21.data_validation_report_sha256(report)
        cutoff = f"{session_text}T07:00:00.000000Z"
        head = replay.data_v21.data_chain_transition_sha256(
            data_contract_identity_sha256=contract,
            previous_data_chain_head_sha256=previous_head,
            decision_session=session_text,
            snapshot_cutoff_utc=cutoff,
            data_manifest_sha256=manifest_sha,
            data_validation_report_sha256=report_sha,
        )
        data = {
            "snapshot_cutoff_utc": cutoff,
            "data_manifest_sha256": manifest_sha,
            "data_validation_report_sha256": report_sha,
            "previous_data_chain_head_sha256": previous_head,
            "data_chain_head_sha256": head,
        }
        record_hash = hashlib.sha256(f"forward-ledger:{record_role}:{index}".encode()).hexdigest()
        snapshots[manifest_sha] = Snapshot(manifest_sha, session_text)
        reports[manifest_sha] = report
        events[record_hash] = {"decision_dt": session_text, "data": data}
        recorded_at[record_hash] = f"{session_text}T07:10:00Z"
        all_record_hashes.append(record_hash)
        previous_head = head
        return record_hash, data

    weeks: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    for index in range(1, 53):
        session = genesis_session + timedelta(days=index)
        record_hash, data = append_transition(index, session, "decision")
        session_text = session.isoformat()
        decisions[session_text] = record_hash
        weeks.append(
            {
                "decision_dt": session_text,
                "data_manifest_sha256": data["data_manifest_sha256"],
                "data_validation_report_sha256": data["data_validation_report_sha256"],
                "data_chain_head_sha256": data["data_chain_head_sha256"],
            }
        )
    confirmation_head = previous_head

    unwind_session = genesis_session + timedelta(days=53)
    unwind_record, _ = append_transition(53, unwind_session, "unwind-decision")
    unwind_eod_sessions = (genesis_session + timedelta(days=54), genesis_session + timedelta(days=55))
    eod_records: dict[str, str] = {}
    for offset, session in enumerate(unwind_eod_sessions, start=54):
        record_hash, _ = append_transition(offset, session, "unwind-eod")
        eod_records[session.isoformat()] = record_hash

    final_manifest = next(reversed(snapshots))
    final_head = previous_head
    ledger_report = {
        "data_contract_identity_sha256": contract,
        "genesis_config": {
            "data_bundle_sha256": genesis_manifest,
            "genesis_data_validation_report_sha256": genesis_report_sha,
            "genesis_data_chain_head_sha256": genesis_head,
            "formal_start_eligibility": {
                "candidate_decision_session": genesis_session.isoformat(),
                "eligibility_cutoff_utc": genesis_cutoff,
            },
        },
        "confirmation_window": {
            "weeks": weeks,
            "confirmation_data_chain_head_sha256": confirmation_head,
        },
        "current_data_chain_head_sha256": final_head,
    }
    bundle_manifest = {
        "data_contract_identity_sha256": contract,
        "genesis_data_manifest_sha256": genesis_manifest,
        "final_data_manifest_sha256": final_manifest,
        "final_data_chain_head_sha256": final_head,
    }
    ledger_index = replay.LedgerConfirmationIndex(
        record_hashes=(genesis_record_hash,),
        all_record_hashes=tuple(all_record_hashes),
        records_sha256=replay.object_sha256(all_record_hashes),
        genesis_recorded_at_utc=recorded_at[genesis_record_hash],
        expected_daily_sessions=(),
        covered_sessions=(),
        unwind_sessions=tuple(session.isoformat() for session in unwind_eod_sessions),
        decision_record_by_session=decisions,
        open_record_by_session={},
        eod_record_by_session=eod_records,
        close_record_by_session={},
        unwind_decision_record_sha256=unwind_record,
        final_evaluation_record_sha256=None,
        event_data_by_record_hash=events,
        recorded_at_utc_by_record_hash=recorded_at,
        anchor_issued_at_utc_by_record_hash={
            digest: timestamp.replace("T07:10:00Z", "T07:08:00Z") for digest, timestamp in recorded_at.items()
        },
    )

    def load_snapshot(path: str | Path) -> tuple[replay.data_v21.DataValidationReport, Snapshot]:
        digest = Path(path).stem
        return reports[digest], snapshots[digest]

    monkeypatch.setattr(replay.data_v21, "load_verified_snapshot", load_snapshot)
    monkeypatch.setattr(
        replay.data_v21,
        "verify_snapshot_extension",
        lambda previous, current: {
            "valid": True,
            "errors": [],
            "verification_sha256": hashlib.sha256(
                f"{previous.manifest_sha256}:{current.manifest_sha256}".encode()
            ).hexdigest(),
        },
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    final_path = manifests / f"{final_manifest}.json"

    verified = replay.verify_forward_data_chain(
        data_manifest_path=final_path,
        bundle_manifest=bundle_manifest,
        ledger_report=ledger_report,
        ledger_index=ledger_index,
    )

    assert verified.verification["decision_transition_count"] == 52
    assert verified.verification["unwind_decision_transition_count"] == 1
    assert verified.verification["unwind_eod_transition_count"] == 2
    assert verified.verification["unique_snapshot_count"] == 56
    assert verified.final_snapshot.manifest_sha256 == final_manifest

    events[genesis_record_hash]["primary_registry_external_anchor"]["issued_at_utc"] = (
        f"{genesis_session.isoformat()}T07:04:00Z"
    )
    with pytest.raises(replay.ReplayVerificationError, match="candidate-close causal order"):
        replay.verify_forward_data_chain(
            data_manifest_path=final_path,
            bundle_manifest=bundle_manifest,
            ledger_report=ledger_report,
            ledger_index=ledger_index,
        )
    events[genesis_record_hash]["primary_registry_external_anchor"]["issued_at_utc"] = (
        f"{genesis_session.isoformat()}T07:06:00Z"
    )

    leaked_transition = weeks[10]
    leaked_snapshot = snapshots[leaked_transition["data_manifest_sha256"]]
    leaked_snapshot.observation_through_session = (
        date.fromisoformat(leaked_transition["decision_dt"]) + timedelta(days=1)
    ).isoformat()
    with pytest.raises(replay.ReplayVerificationError, match="causal ledger session"):
        replay.verify_forward_data_chain(
            data_manifest_path=final_path,
            bundle_manifest=bundle_manifest,
            ledger_report=ledger_report,
            ledger_index=ledger_index,
        )


def test_object_roles_cannot_alias_one_logical_object(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    manifest["readiness_report_object"] = manifest["planner_input_object"]
    manifest["objects"].pop("readiness/report")
    path.write_bytes(replay.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayVerificationError, match="multiple semantic roles"):
        replay.load_replay_bundle(path)


def test_object_roles_cannot_hide_alias_behind_two_names(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    manifest["objects"]["readiness/report"] = copy.deepcopy(manifest["objects"]["planner/input"])
    path.write_bytes(replay.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayVerificationError, match="Identical content|identical content"):
        replay.load_replay_bundle(path)


def test_seed_pairs_cannot_reuse_one_input_or_result_role(tmp_path: Path) -> None:
    path, manifest = _bundle(tmp_path)
    base = manifest["execution_pairs"][0]
    manifest["execution_pairs"] = [
        {
            **base,
            "pair_id": "Rmatch:20260720:2x",
            "family_id": "R_match",
            "seed_id": 20260720,
            "statistics_arm_id": "R_match_2x",
        },
        {
            **base,
            "pair_id": "Rmatch:20260721:2x",
            "family_id": "R_match",
            "seed_id": 20260721,
            "statistics_arm_id": "R_match_2x",
        },
    ]
    path.write_bytes(replay.canonical_json_bytes(manifest))
    with pytest.raises(replay.ReplayVerificationError, match="multiple semantic roles"):
        replay.load_replay_bundle(path)


def test_readiness_report_is_exact_self_addressed_and_dependency_bound() -> None:
    dependencies = dict.fromkeys(replay.EXPECTED_DEPENDENCIES, "4" * 64)
    report = _readiness(dependencies=dependencies)
    verified = replay.verify_readiness_report(
        report,
        expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        expected_data_contract_identity_sha256=replay.data_v21.data_contract_identity_sha256(),
        expected_dependencies=dependencies,
    )
    assert verified["report_sha256"] == report["report_sha256"]
    assert len(verified["engineering_gates"]) == 13
    assert "ledger_engine" in verified["dependency_sha256"]
    assert "data_manifest_sha256" not in verified

    corrupted = copy.deepcopy(report)
    corrupted["engineering_gates"][replay.ENGINEERING_GATES[0]]["evidence_sha256"] = "9" * 64
    with pytest.raises(replay.ReplayVerificationError, match="self-digest"):
        replay.verify_readiness_report(
            corrupted,
            expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            expected_data_contract_identity_sha256=replay.data_v21.data_contract_identity_sha256(),
            expected_dependencies=dependencies,
        )

    wrong_contract = copy.deepcopy(report)
    wrong_contract["data_contract_identity_sha256"] = "9" * 64
    wrong_contract["report_sha256"] = replay.object_sha256(
        {key: value for key, value in wrong_contract.items() if key != "report_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="static data contract"):
        replay.verify_readiness_report(
            wrong_contract,
            expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            expected_data_contract_identity_sha256=replay.data_v21.data_contract_identity_sha256(),
            expected_dependencies=dependencies,
        )


def test_readiness_report_rejects_manual_failed_gate_and_missing_ledger_dependency() -> None:
    dependencies = dict.fromkeys(replay.EXPECTED_DEPENDENCIES, "4" * 64)
    report = _readiness(dependencies=dependencies)
    report["engineering_gates"][replay.ENGINEERING_GATES[0]]["status"] = "FAILED"
    report["report_sha256"] = replay.object_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="PASSED pre-candidate commitment"):
        replay.verify_readiness_report(
            report,
            expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            expected_data_contract_identity_sha256=replay.data_v21.data_contract_identity_sha256(),
            expected_dependencies=dependencies,
        )

    missing = dict(dependencies)
    missing.pop("ledger_engine")
    report = _readiness(dependencies=missing)
    with pytest.raises(replay.ReplayVerificationError, match="dependency closure"):
        replay.verify_readiness_report(
            report,
            expected_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
            expected_data_contract_identity_sha256=replay.data_v21.data_contract_identity_sha256(),
            expected_dependencies=dependencies,
        )


def test_readiness_anchor_uses_verified_receipt_time_not_self_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = replay.ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier(
        provider="test-anchor",
        key_id="test-key",
        modulus_hex="f" * 256,
        public_exponent=3,
    )
    report = _readiness()
    issued = datetime(2026, 7, 19, 0, 1, tzinfo=UTC)

    def verified_receipt(self, receipt, *, expected_subject_sha256):
        assert self is verifier and receipt == {"signed": True}
        return replay.ledger_v21.VerifiedReceipt(
            subject_sha256=expected_subject_sha256,
            issued_at_utc=issued,
            provider="test-anchor",
            key_id="test-key",
            receipt_sha256="8" * 64,
        )

    monkeypatch.setattr(
        replay.ledger_v21.RsaPkcs1v15Sha256ReceiptVerifier,
        "verify",
        verified_receipt,
    )
    anchor = replay.verify_readiness_anchor(
        {"signed": True},
        readiness_report=report,
        receipt_verifier=verifier,
    )
    assert anchor["issued_at_utc"] == "2026-07-19T00:01:00Z"
    assert anchor["report_sha256"] == report["report_sha256"]

    late_report = {**report, "created_at_utc": "2026-07-19T00:01:01Z"}
    with pytest.raises(replay.ReplayVerificationError, match="creation follows"):
        replay.verify_readiness_anchor(
            {"signed": True},
            readiness_report=late_report,
            receipt_verifier=verifier,
        )
    with pytest.raises(replay.ReplayVerificationError, match="concrete readiness"):
        replay.verify_readiness_anchor(
            {"signed": True},
            readiness_report=report,
            receipt_verifier=None,
        )


def test_planner_diagnostic_must_be_valid_complete_52_cycle_batch() -> None:
    verified = replay._require_verified_planner(_planner_verification())
    assert verified["valid"] is True
    assert verified["cycle_count"] == 52
    assert verified["execution_decision_count"] == 13_104

    with pytest.raises(replay.ReplayVerificationError, match="valid=true"):
        replay._require_verified_planner(_planner_verification(valid=False))

    incomplete = _planner_verification()
    incomplete["cycle_count"] = 51
    incomplete["verification_sha256"] = replay.object_sha256(
        {key: value for key, value in incomplete.items() if key != "verification_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="exactly 52 cycles"):
        replay._require_verified_planner(incomplete)


def test_state_recompute_requires_all_symbols_plus_real_100_by_20_zero_mismatch_audit() -> None:
    verified = replay._require_state_recompute(_state_recompute_verification())
    assert verified["recomputed_symbol_count"] == verified["source_symbol_count"] == 5_719
    assert verified["prefix_audit_symbol_count"] == 100
    assert verified["cutoffs_per_prefix_audit_symbol"] == 20

    with pytest.raises(replay.ReplayVerificationError, match="100×20 zero-mismatch"):
        replay._require_state_recompute(_state_recompute_verification(valid=False))


def test_formal_start_eligibility_is_recomputed_and_candidate_binds_first_cycle() -> None:
    class Snapshot:
        manifest_sha256 = "a" * 64

    class Planner:
        @staticmethod
        def frames_from_verified_snapshot(snapshot):
            assert snapshot.manifest_sha256 == "a" * 64
            return {}

        @staticmethod
        def compute_decision_surface(frames, decision_session, *, require_target_size):
            assert frames == {} and decision_session == "2026-07-10" and require_target_size is False
            return list(range(49))

        @staticmethod
        def _surface_payload(surface):
            return [{"rank": value} for value in surface]

        object_sha256 = staticmethod(replay.object_sha256)

    earlier_surface_sha = replay.object_sha256([{"rank": value} for value in range(49)])
    eligibility = {
        "schema": "xs_chan_v2_1_formal_start_eligibility_v2",
        "readiness_report_sha256": "1" * 64,
        "readiness_anchor_receipt_sha256": "5" * 64,
        "readiness_completed_at_utc": "2026-07-10T06:00:00.000000Z",
        "eligibility_cutoff_utc": "2026-07-17T07:00:00.000000Z",
        "candidate_decision_session": "2026-07-17",
        "candidate_execution_week": "2026-W30",
        "candidate_eligible_count": 50,
        "minimum_eligible_symbol_count": 50,
        "earlier_completed_weeks": [
            {
                "week_id": "2026-W29",
                "decision_session": "2026-07-10",
                "eligible_symbol_count": 49,
                "evidence_sha256": earlier_surface_sha,
            }
        ],
        "calendar_head_sha256": "2" * 64,
        "data_snapshot_sha256": "a" * 64,
        "planner_verifier_sha256": "3" * 64,
    }
    eligibility["evidence_sha256"] = replay.object_sha256(eligibility)
    planner_batch = {
        "items": [
            {
                "decision_dt": "2026-07-17",
                "eligible_count": 50,
                "data_snapshot_sha256": "a" * 64,
                "ranked_surface_sha256": "4" * 64,
            },
            *[{} for _ in range(51)],
        ]
    }
    readiness_report = {
        "report_sha256": eligibility["readiness_report_sha256"],
        "created_at_utc": "2026-07-10T05:59:59.000000Z",
    }
    readiness_anchor = {
        "report_sha256": eligibility["readiness_report_sha256"],
        "issued_at_utc": eligibility["readiness_completed_at_utc"],
        "receipt_sha256": "5" * 64,
    }
    verified = replay._verify_formal_start_eligibility(
        eligibility,
        planner_batch,
        Snapshot(),
        Planner,
        readiness_report,
        readiness_anchor,
    )
    assert verified["candidate_eligible_count"] == 50
    assert verified["earlier_completed_weeks"][0]["eligible_symbol_count"] == 49

    forged = copy.deepcopy(eligibility)
    forged["earlier_completed_weeks"][0]["eligible_symbol_count"] = 48
    forged["evidence_sha256"] = replay.object_sha256(
        {key: value for key, value in forged.items() if key != "evidence_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="planner recomputation"):
        replay._verify_formal_start_eligibility(
            forged,
            planner_batch,
            Snapshot(),
            Planner,
            readiness_report,
            readiness_anchor,
        )

    with pytest.raises(replay.ReplayVerificationError, match="external readiness timestamp"):
        replay._verify_formal_start_eligibility(
            eligibility,
            planner_batch,
            Snapshot(),
            Planner,
            readiness_report,
            {
                **readiness_anchor,
                "issued_at_utc": "2026-07-10T06:00:01.000000Z",
            },
        )


def test_execution_pair_seed_is_bound_inside_input_and_result() -> None:
    execution_input = {"arm_id": "R_match", "seed_id": 20260720, "scenario_id": "2x"}
    execution_result = dict(execution_input)
    replay._require_execution_pair_identity(
        execution_input,
        execution_result,
        family="R_match",
        seed=20260720,
        scenario="2x",
    )
    execution_result["seed_id"] = 20260721
    with pytest.raises(replay.ReplayVerificationError, match="seed_id differs"):
        replay._require_execution_pair_identity(
            execution_input,
            execution_result,
            family="R_match",
            seed=20260720,
            scenario="2x",
        )
    execution_input.pop("seed_id")
    with pytest.raises(replay.ReplayVerificationError, match="requires a content-bound seed_id"):
        replay._require_execution_pair_identity(
            execution_input,
            {"arm_id": "R_match", "seed_id": 20260720, "scenario_id": "2x"},
            family="R_match",
            seed=20260720,
            scenario="2x",
        )


def test_post_window_cost_is_independent_of_manifest_pair_order() -> None:
    rows = [
        (("F", None, "1x"), 1.0e16),
        (("FC", None, "1x"), 1.0),
        (("FMA", None, "1x"), -1.0e16),
    ]

    def result(value: float) -> dict[str, Any]:
        return {
            "events": [
                {
                    "event_type": "order",
                    "session": "2026-01-02",
                    "commission_cny": value,
                    "transfer_fee_cny": 0.0,
                    "stamp_duty_cny": 0.0,
                    "slippage_cost_cny": 0.0,
                }
            ]
        }

    forward = {key: result(value) for key, value in rows}
    reverse = {key: result(value) for key, value in reversed(rows)}
    assert replay._post_window_unwind_cost(forward, "2026-01-01") == 1.0
    assert replay._post_window_unwind_cost(reverse, "2026-01-01") == 1.0


def test_ledger_confirmation_index_reads_exact_record_prefix_and_session_domains() -> None:
    records, window = _ledger_confirmation_fixture()
    index = replay._build_ledger_confirmation_index(records, window)
    assert len(index.decision_record_by_session) == 52
    assert len(index.open_record_by_session) == 52
    assert len(index.eod_record_by_session) == 52
    assert len(index.close_record_by_session) == 52
    assert len(index.expected_daily_sessions) == 53

    corrupted = copy.deepcopy(window)
    corrupted["weeks"][0]["decision_record_sha256"] = "f" * 64
    with pytest.raises(replay.ReplayVerificationError, match="decision record hash"):
        replay._build_ledger_confirmation_index(records, corrupted)


def test_execution_timeline_cannot_omit_a_covered_ledger_session() -> None:
    records, window = _ledger_confirmation_fixture()
    index = replay._build_ledger_confirmation_index(records, window)
    sessions: list[dict[str, Any]] = []
    for session in index.expected_daily_sessions:
        decision = None
        matching_week = next((item for item in window["weeks"] if item["execution_dt"] == session), None)
        if matching_week is not None:
            decision_session = str(matching_week["decision_dt"])
            decision = {
                "decision_session": decision_session,
                "execution_session": matching_week["execution_dt"],
                "decision_record_sha256": index.decision_record_by_session[decision_session],
            }
        sessions.append({"session": session, "decision": decision})
    replayed = {
        "daily": [{"session": item["session"]} for item in sessions],
        "session_evidence": [{"session": item["session"]} for item in sessions],
        "decision_evidence": [{"decision_session": session} for session in index.decision_record_by_session],
    }
    replay._require_execution_timeline({"sessions": sessions}, replayed, index)

    missing = copy.deepcopy(sessions)
    missing.pop(10)
    with pytest.raises(replay.ReplayVerificationError, match="daily sessions differ"):
        replay._require_execution_timeline({"sessions": missing}, replayed, index)


def test_final_evaluation_binds_pending_settlements_and_last_unwind_cutoff() -> None:
    unwind_hash = "a" * 64
    eod_hash = "b" * 64
    final_hash = "c" * 64
    completion = {
        "book_state_root_sha256": "d" * 64,
        "pending_sells_root_sha256": "e" * 64,
        "pending_settlements_root_sha256": "f" * 64,
        "remaining_position_count": 0,
        "remaining_pending_sell_count": 0,
        "remaining_pending_settlement_count": 0,
    }
    index = replay.LedgerConfirmationIndex(
        record_hashes=("1" * 64,),
        all_record_hashes=("1" * 64, unwind_hash, eod_hash, final_hash),
        records_sha256="2" * 64,
        genesis_recorded_at_utc="2026-01-01T00:00:00Z",
        expected_daily_sessions=("2026-01-01", "2026-01-02"),
        covered_sessions=("2026-01-01",),
        unwind_sessions=("2026-01-02",),
        decision_record_by_session={},
        open_record_by_session={},
        eod_record_by_session={"2026-01-02": eod_hash},
        close_record_by_session={},
        unwind_decision_record_sha256=unwind_hash,
        final_evaluation_record_sha256=final_hash,
        event_data_by_record_hash={unwind_hash: {"data": {"unwind_evidence_root_sha256": "3" * 64}}},
        recorded_at_utc_by_record_hash={
            eod_hash: "2026-01-02T08:00:00Z",
            final_hash: "2026-01-02T08:10:00Z",
        },
        anchor_issued_at_utc_by_record_hash={},
    )
    unwind_evidence = replay.object_sha256(
        {
            "decision_unwind_evidence_root_sha256": "3" * 64,
            "completion_book_state_root_sha256": completion["book_state_root_sha256"],
            "completion_pending_sells_root_sha256": completion["pending_sells_root_sha256"],
            "remaining_position_count": 0,
            "remaining_pending_sell_count": 0,
            "remaining_pending_settlement_count": 0,
            "completion_pending_settlements_root_sha256": completion["pending_settlements_root_sha256"],
            "unwind_session_count": 1,
            "post_window_unwind_cost_cny": 0.0,
        }
    )
    payload = {
        "schema": replay.ledger_v21.FINAL_EVALUATION_SCHEMA,
        "snapshot_cutoff_utc": "2026-01-02T08:05:00Z",
        "confirmation_window_identity_sha256": "4" * 64,
        "statistics_result_sha256": "5" * 64,
        "semantic_replay_evidence_sha256": "6" * 64,
        "evaluation_status": "INCONCLUSIVE_FACTOR",
        "post_window_unwind_completed": True,
        "post_window_unwind_session_count": 1,
        "post_window_unwind_cost_cny": 0.0,
        "remaining_position_count": 0,
        "remaining_pending_sell_count": 0,
        "remaining_pending_settlement_count": 0,
        "unwind_completion_book_state_root_sha256": completion["book_state_root_sha256"],
        "unwind_completion_pending_sells_root_sha256": completion["pending_sells_root_sha256"],
        "unwind_completion_pending_settlements_root_sha256": completion["pending_settlements_root_sha256"],
        "unwind_evidence_sha256": unwind_evidence,
    }
    payload["final_evaluation_artifact_sha256"] = replay.object_sha256(payload)
    kwargs = {
        "semantic_replay_evidence_sha256": "6" * 64,
        "statistics_result": {"result_sha256": "5" * 64, "overall_status": "INCONCLUSIVE_FACTOR"},
        "window_sha256": "4" * 64,
        "ledger_index": index,
        "ledger_binding": {"all_books": {"completion": completion}},
        "results": {("F", None, "1x"): {"events": []}},
    }
    replay._verify_final_evaluation(payload, **kwargs)

    forged = copy.deepcopy(payload)
    forged["remaining_pending_settlement_count"] = 1
    forged["final_evaluation_artifact_sha256"] = replay.object_sha256(
        {key: value for key, value in forged.items() if key != "final_evaluation_artifact_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="semantic replay"):
        replay._verify_final_evaluation(forged, **kwargs)

    stale = copy.deepcopy(payload)
    stale["snapshot_cutoff_utc"] = "2026-01-02T07:59:59Z"
    stale["final_evaluation_artifact_sha256"] = replay.object_sha256(
        {key: value for key, value in stale.items() if key != "final_evaluation_artifact_sha256"}
    )
    with pytest.raises(replay.ReplayVerificationError, match="last-unwind-EOD"):
        replay._verify_final_evaluation(stale, **kwargs)
