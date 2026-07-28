from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage3 as stage3  # noqa: E402


def _spec() -> dict[str, Any]:
    return copy.deepcopy(stage3.load_and_validate_spec())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_hash(record: Mapping[str, Any] | stage3.LedgerRecord) -> str:
    if isinstance(record, stage3.LedgerRecord):
        record = record.data
    for key in ("record_hash", "hash", "chain_hash"):
        value = record.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    raise AssertionError(f"record has no SHA256 commit marker: {sorted(record)}")


def _record_files(root: Path) -> list[Path]:
    records_dir = root / "records"
    assert records_dir.is_dir()
    files = sorted(records_dir.glob("*.json"))
    assert files
    return files


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(dict(item))
            else:
                result.append({"symbol": str(item)})
        return result
    raise AssertionError(f"cannot normalize rows from {type(value)!r}")


def _fc_result_rows(result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(result, Mapping):
        membership = next(
            (
                result[key]
                for key in ("memberships", "members", "target_symbols", "next_previous_symbols")
                if key in result
            ),
            None,
        )
        proposals = result.get("proposals")
    elif isinstance(result, tuple) and len(result) >= 2:
        membership, proposals = result[:2]
    else:
        raise AssertionError(f"unexpected build_fc_week result: {type(result)!r}")
    assert membership is not None
    assert proposals is not None
    return _rows(membership), _rows(proposals)


def _decision_dates(schedule: Any) -> list[str]:
    values: list[Any]
    if isinstance(schedule, pd.DataFrame):
        column = "decision_date" if "decision_date" in schedule else "decision_dt"
        values = schedule[column].tolist()
    else:
        values = list(schedule)

    result = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("decision_date", value.get("decision_dt"))
        result.append(pd.Timestamp(value).date().isoformat())
    return result


def _assert_no_blinded_values(value: Any, sentinels: set[float]) -> None:
    forbidden_key_fragments = {
        "return",
        "weekly_difference",
        "direction",
        "nav",
        "win_rate",
        "confidence",
        "bootstrap",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            assert not any(fragment in normalized for fragment in forbidden_key_fragments), key
            _assert_no_blinded_values(child, sentinels)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_no_blinded_values(child, sentinels)
    elif isinstance(value, float):
        assert all(value != pytest.approx(sentinel) for sentinel in sentinels)


def _semantic_genesis(
    spec: Mapping[str, Any],
    *,
    ledger_id: str | None = None,
    record_hash: str = "1" * 64,
) -> dict[str, Any]:
    identity = stage3.study_identity(spec)
    return {
        "ledger_id": ledger_id or identity,
        "record_type": "genesis",
        "record_hash": record_hash,
        "recorded_at_utc": "2026-07-28T06:00:00.000000Z",
        "payload": {
            "schema": stage3.LEDGER_SCHEMA,
            "study_id": spec["study_id"],
            "study_identity": identity,
            "spec_physical_sha256": stage3.sha256_file(stage3.SPEC_PATH),
            "spec_canonical_sha256": stage3.sha256_bytes(stage3.canonical_json(spec)),
            "collector_source_sha256": stage3.sha256_file(stage3.SOURCE_PATH),
            "protocol_anchor_git_commit": "a" * 40,
            "first_prospective_decision_date": spec["historical_exclusion"]["first_prospective_decision_date"],
            "rehearsal_dates_never_count": spec["historical_exclusion"]["pre_genesis_rehearsal_decision_dates"],
            "confirmation_chain": "NOT_STARTED",
            "live_trading_authorized": False,
            "claim_boundary": spec["claim_boundary"],
        },
    }


def _semantic_decision(spec: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DatetimeIndex, dict[str, Any]]:
    sessions = pd.DatetimeIndex(pd.bdate_range("2026-07-27", "2026-08-17"))
    schedule = stage3.build_forward_schedule(spec, sessions)
    row = schedule.iloc[0]
    decision_dt = row["decision_dt"].date().isoformat()
    entry_dt = row["entry_dt"].date().isoformat()
    exit_dt = row["exit_dt"].date().isoformat()
    calendar_sha = "4" * 64
    official_prefix = [value.date().isoformat() for value in sessions if value <= row["exit_dt"]]
    proposal = {
        "symbol": "000001.SZ",
        "proposal_order": 1,
        "gate_passed": False,
        "regime": 3,
        "factor_rank": 1,
        "factor_score": 0.9,
        "industry_code": None,
        "mcap_bucket": None,
        "ma_allowed": True,
    }
    payload = {
        "schema": "xs_chan_stage3_decision_freeze_v1",
        "classification": "PROSPECTIVE_COUNTED",
        "week_index": 1,
        "decision_dt": decision_dt,
        "entry_dt": entry_dt,
        "exit_dt": exit_dt,
        "previous_decision_record_hash": None,
        "reference_manifest_sha256": "3" * 64,
        "official_calendar_sha256": calendar_sha,
        "official_session_prefix_sha256": stage3.sha256_bytes(stage3.canonical_json(official_prefix)),
        "decision_path_sha256": "5" * 64,
        "decision_path_object": f"objects/decision_path/{'5' * 64}.json",
        "bridge_decision_dates": [],
        "proposals": [proposal],
        "current_membership": [],
        "current_membership_sha256": stage3.sha256_bytes(stage3.canonical_json([])),
        "outcome_columns_forbidden": True,
    }
    decision = {
        "ledger_id": stage3.study_identity(spec),
        "record_type": "decision_freeze",
        "record_hash": "2" * 64,
        "recorded_at_utc": "2026-07-31T08:00:00.000000Z",
        "payload": payload,
    }
    manifest = {"official_calendar": {"sha256": calendar_sha}}
    return decision, sessions, manifest


def _stub_semantic_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    sessions: pd.DatetimeIndex,
    manifest: Mapping[str, Any],
    *,
    replay_decision_path: bool = False,
) -> None:
    monkeypatch.setattr(stage3, "validate_protocol_anchor", lambda _spec: "a" * 40)
    monkeypatch.setattr(
        stage3,
        "load_reference_manifest",
        lambda _root, _path: (dict(manifest), sessions, {}),
    )
    if replay_decision_path:
        monkeypatch.setattr(stage3, "_validate_decision_path_object", lambda *_args: None)


def test_stage3_spec_is_forward_only_and_preserves_v2_1_bytes():
    v2_path = ROOT / "scripts" / "xs_chan_protocol_v2_1.json"
    v2_document_path = ROOT / "scripts" / "XS_CHAN_PILOT_PROTOCOL_V2_1.md"
    before = v2_path.read_bytes()
    before_document = v2_document_path.read_bytes()

    spec = stage3.load_and_validate_spec()

    assert spec["mode"] == "FORWARD_EXPLORATION_ONLY"
    assert spec["study_type"] == "LOCAL_PROSPECTIVE_MECHANISM_REPLICATION"
    assert spec["confirmation_chain"] == "NOT_STARTED"
    assert spec["live_trading_authorized"] is False
    assert spec["historical_exclusion"]["first_prospective_decision_date"] == "2026-07-31"
    assert spec["accrual"]["window_weeks"] == 52
    assert spec["statistics"]["primary_endpoint_count"] == 1
    assert spec["blinding"]["evaluation_before_lock"] == "REFUSE"
    assert v2_path.read_bytes() == before
    assert v2_document_path.read_bytes() == before_document

    frozen = spec["frozen_v2_1_baseline"]
    assert hashlib.sha256(before).hexdigest() == frozen["protocol_physical_sha256"]
    assert _canonical_sha256(json.loads(before)) == frozen["protocol_canonical_sha256"]
    assert hashlib.sha256(before_document).hexdigest() == frozen["protocol_document_sha256"]


def test_study_identity_binds_canonical_spec_and_exact_source_bytes(tmp_path: Path):
    spec = _spec()
    source = tmp_path / "study.py"
    source.write_bytes(b"first\n")
    first = stage3.study_identity(spec, source)
    assert first == stage3.study_identity(copy.deepcopy(spec), source)

    source.write_bytes(b"second\n")
    source_changed = stage3.study_identity(spec, source)
    spec_changed = copy.deepcopy(spec)
    spec_changed["study_id"] = "different"
    both_changed = stage3.study_identity(spec_changed, source)

    assert len(first) == 64
    assert len(source_changed) == 64
    assert len(both_changed) == 64
    assert len({first, source_changed, both_changed}) == 3


def test_append_is_idempotent_and_conflict_is_preserved(tmp_path: Path):
    root = tmp_path / "ledger"
    recorded_at = "2026-07-31T08:00:00Z"
    first = stage3.append_record(
        root,
        "decision_freeze",
        "decision_freeze:2026-07-31",
        {"decision_date": "2026-07-31", "proposals": []},
        recorded_at_utc=recorded_at,
    )
    retry = stage3.append_record(
        root,
        "decision_freeze",
        "decision_freeze:2026-07-31",
        {"decision_date": "2026-07-31", "proposals": []},
        recorded_at_utc=recorded_at,
    )

    assert _record_hash(first) == _record_hash(retry)
    assert len(stage3.scan_records(root)) == 1
    assert len(_record_files(root)) == 1

    with pytest.raises(stage3.Stage3Error, match="conflict|different payload|logical"):
        stage3.append_record(
            root,
            "decision_freeze",
            "decision_freeze:2026-07-31",
            {"decision_date": "2026-07-31", "proposals": [{"symbol": "MUTATED"}]},
            recorded_at_utc=recorded_at,
        )

    assert len(stage3.scan_records(root)) == 1
    failures = [path for path in (root / "failures").rglob("*") if path.is_file()]
    assert failures
    assert all(path.stat().st_size > 0 for path in failures)


def test_scan_detects_payload_tampering(tmp_path: Path):
    root = tmp_path / "tamper"
    stage3.append_record(
        root,
        "decision_freeze",
        "decision_freeze:2026-07-31",
        {"decision_date": "2026-07-31", "proposals": []},
        recorded_at_utc="2026-07-31T08:00:00Z",
    )
    record_path = _record_files(root)[0]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["payload"]["decision_date"] = "2099-01-01"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(stage3.Stage3Error, match="hash|tamper|payload"):
        stage3.scan_records(root)


def test_scan_detects_fork_and_sequence_gap(tmp_path: Path):
    fork_root = tmp_path / "fork"
    for sequence, decision_date in enumerate(("2026-07-31", "2026-08-07"), start=1):
        stage3.append_record(
            fork_root,
            "decision_freeze",
            f"decision_freeze:{decision_date}",
            {"decision_date": decision_date, "proposals": []},
            recorded_at_utc=f"{decision_date}T08:00:00Z",
        )
        assert len(stage3.scan_records(fork_root)) == sequence
    records = _record_files(fork_root)
    sequence = records[-1].name.split("_", maxsplit=1)[0]
    shutil.copyfile(records[-1], records[-1].with_name(f"{sequence}_{'f' * 64}.json"))
    with pytest.raises(stage3.Stage3Error, match="fork|duplicate|sequence"):
        stage3.scan_records(fork_root)

    gap_root = tmp_path / "gap"
    for decision_date in ("2026-07-31", "2026-08-07", "2026-08-14"):
        stage3.append_record(
            gap_root,
            "decision_freeze",
            f"decision_freeze:{decision_date}",
            {"decision_date": decision_date, "proposals": []},
            recorded_at_utc=f"{decision_date}T08:00:00Z",
        )
    gap_records = _record_files(gap_root)
    gap_records[1].unlink()
    with pytest.raises(stage3.Stage3Error, match="gap|sequence|previous"):
        stage3.scan_records(gap_root)


def test_build_fc_week_retains_buffer_names_and_never_refills_after_rejection():
    selection = {
        "target_size": 3,
        "retention_max_rank": 5,
        "allowed_chan_regimes": [5, 6, 7, 8],
    }
    ranked = pd.DataFrame(
        [
            {"symbol": "C", "factor_rank": 1, "factor_score": 0.99, "regime": 3},
            {"symbol": "D", "factor_rank": 2, "factor_score": 0.98, "regime": 5},
            {"symbol": "E", "factor_rank": 3, "factor_score": 0.97, "regime": 4},
            {"symbol": "A", "factor_rank": 4, "factor_score": 0.96, "regime": 1},
            {"symbol": "G", "factor_rank": 5, "factor_score": 0.95, "regime": 7},
            {"symbol": "B", "factor_rank": 6, "factor_score": 0.94, "regime": 7},
        ]
    )

    result = stage3.build_fc_week(selection, ranked, {"A", "B"})
    membership_rows, proposal_rows = _fc_result_rows(result)
    member_symbols = {str(row["symbol"]) for row in membership_rows}
    proposal_symbols = [str(row["symbol"]) for row in proposal_rows]
    gate_by_symbol = {str(row["symbol"]): bool(row["gate_passed"]) for row in proposal_rows}

    assert member_symbols == {"A", "D"}
    assert proposal_symbols == ["C", "D"]
    assert gate_by_symbol == {"C": False, "D": True}
    assert "B" not in member_symbols
    assert "G" not in member_symbols
    assert "G" not in proposal_symbols


def test_rehearsal_dates_never_count_and_july_31_is_first_forward_week():
    spec = _spec()
    rehearsal_classification = spec["historical_exclusion"]["rehearsal_classification"]
    for value in ("2026-06-12", "2026-06-18", "2026-06-26"):
        assert stage3.classify_decision_date(spec, date.fromisoformat(value)) == rehearsal_classification

    assert stage3.classify_decision_date(spec, date(2026, 7, 24)) != "PROSPECTIVE_COUNTED_CANDIDATE"
    assert stage3.classify_decision_date(spec, date(2026, 7, 31)) == "PROSPECTIVE_COUNTED_CANDIDATE"

    sessions = list(pd.bdate_range("2026-06-08", "2026-08-17"))
    schedule_dates = _decision_dates(stage3.build_forward_schedule(spec, sessions))

    assert schedule_dates[:2] == ["2026-07-31", "2026-08-07"]
    assert not {"2026-06-12", "2026-06-18", "2026-06-26"} & set(schedule_dates)


def test_calendar_prefix_extension_never_changes_existing_schedule():
    spec = _spec()
    short_sessions = list(pd.bdate_range("2026-07-27", "2026-08-24"))
    extended_sessions = list(pd.bdate_range("2026-07-27", "2026-09-07"))

    short = _decision_dates(stage3.build_forward_schedule(spec, short_sessions))
    extended = _decision_dates(stage3.build_forward_schedule(spec, extended_sessions))

    assert short == ["2026-07-31", "2026-08-07", "2026-08-14"]
    assert extended[: len(short)] == short
    assert extended == short + ["2026-08-21", "2026-08-28"]


def test_before_52_weeks_evaluation_refuses_and_status_cannot_leak_returns(tmp_path: Path):
    spec = _spec()
    root = tmp_path / "blinded"
    sentinels: set[float] = set()
    decision_dates = pd.date_range("2026-07-31", periods=51, freq="W-FRI")
    for index, decision_ts in enumerate(decision_dates):
        decision_date = decision_ts.date()
        decision_iso = decision_date.isoformat()
        treated_symbol = f"T{index:03d}"
        control_symbol = f"C{index:03d}"
        decision = stage3.append_record(
            root,
            "decision_freeze",
            f"decision_freeze:{decision_iso}",
            {
                "decision_date": decision_iso,
                "classification": "PROSPECTIVE_COUNTED",
                "week_index": index + 1,
                "previous_symbols": [],
                "next_previous_symbols": [],
                "proposals": [
                    {"symbol": treated_symbol, "factor_rank": 1, "regime": 3},
                    {"symbol": control_symbol, "factor_rank": 2, "regime": 4},
                ],
            },
            recorded_at_utc=f"{decision_iso}T08:00:00Z",
        )
        sentinel = 0.987654321 + index / 1_000_000
        sentinels.add(sentinel)
        completion_date = decision_date + timedelta(days=7)
        stage3.append_record(
            root,
            "label_completion",
            f"label_completion:{decision_iso}",
            {
                "decision_date": decision_iso,
                "decision_record_hash": _record_hash(decision),
                "events": [
                    {
                        "symbol": treated_symbol,
                        "regime": 3,
                        "factor_rank": 1,
                        "entry_tradable": True,
                        "matched": True,
                        "return_5d": sentinel,
                    },
                    {
                        "symbol": control_symbol,
                        "regime": 4,
                        "factor_rank": 2,
                        "entry_tradable": True,
                        "matched": False,
                        "return_5d": -sentinel,
                    },
                ],
            },
            recorded_at_utc=f"{completion_date.isoformat()}T07:00:00Z",
        )

    records = stage3.scan_records(root)
    status = stage3._blinded_status_report(spec, records)

    assert status["decision_week_count"] == 51
    assert status["completed_label_week_count"] == 51
    assert status.get("blinded", True) is True
    _assert_no_blinded_values(status, sentinels)
    serialized_status = json.dumps(status, sort_keys=True)
    assert all(format(sentinel, ".9f") not in serialized_status for sentinel in sentinels)

    with pytest.raises(stage3.Stage3Error, match="52|before|lock|REFUSE"):
        stage3._evaluate_locked_records(spec, records)


def test_semantic_replay_rejects_old_june_rehearsal_counting_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _spec()
    root = tmp_path / "june-attack"
    genesis = _semantic_genesis(spec)
    july, _, _ = _semantic_decision(spec)
    june = copy.deepcopy(july)
    june["record_hash"] = "6" * 64
    june["payload"]["week_index"] = 2
    june["payload"]["decision_dt"] = "2026-06-12"
    june["payload"]["entry_dt"] = "2026-06-15"
    june["payload"]["exit_dt"] = "2026-06-22"
    monkeypatch.setattr(stage3, "validate_protocol_anchor", lambda _spec: "a" * 40)

    with pytest.raises(stage3.Stage3ValidationError, match="date|index|classification|first"):
        stage3.validate_record_semantics(spec, [genesis, june, july], root=root)


@pytest.mark.parametrize("api_name", ["status_report", "evaluate_records"])
def test_public_outputs_reject_records_not_loaded_from_ledger_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_name: str,
):
    spec = _spec()
    root = tmp_path / "fabricated-chain"
    genesis = _semantic_genesis(spec)
    monkeypatch.setattr(stage3, "validate_protocol_anchor", lambda _spec: "a" * 40)

    api = getattr(stage3, api_name)
    with pytest.raises(stage3.Stage3ValidationError, match="differ from the ledger stored on disk"):
        api(spec, [genesis], root=root)


def test_semantic_replay_rejects_future_outcome_field_in_decision_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _spec()
    root = tmp_path / "future-open"
    genesis = _semantic_genesis(spec)
    decision, _, _ = _semantic_decision(spec)
    decision["payload"]["future_open"] = 123.45
    monkeypatch.setattr(stage3, "validate_protocol_anchor", lambda _spec: "a" * 40)

    with pytest.raises(stage3.Stage3ValidationError, match="decision payload schema"):
        stage3.validate_record_semantics(spec, [genesis, decision], root=root)


def test_semantic_replay_rejects_forged_label_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _spec()
    root = tmp_path / "forged-label"
    genesis = _semantic_genesis(spec)
    decision, sessions, manifest = _semantic_decision(spec)
    _stub_semantic_dependencies(monkeypatch, sessions, manifest, replay_decision_path=True)

    state_manifest_sha = "7" * 64
    closure = {
        "schema": "xs_chan_stage3_raw_source_closure_v1",
        "state_manifest_sha256": state_manifest_sha,
        "file_count": 0,
        "coverage_max_dt": decision["payload"]["exit_dt"],
        "files": [],
    }
    closure_sha, closure_path = stage3._store_canonical_object(root, "raw_source_closure", closure)
    snapshot = {
        "schema": "xs_chan_stage3_label_observation_snapshot_v1",
        "entry_dt": decision["payload"]["entry_dt"],
        "exit_dt": decision["payload"]["exit_dt"],
        "state_manifest_sha256": state_manifest_sha,
        "raw_source_closure_sha256": closure_sha,
        "raw_source_closure_object": str(closure_path.relative_to(root)),
        "observations": [
            {
                "symbol": "000001.SZ",
                "entry": {"observed": True, "open": 100.0, "vol": 1.0, "amount": 1.0},
                "exit": {"observed": True, "open": 110.0, "vol": 1.0, "amount": 1.0},
            }
        ],
    }
    observation_sha, observation_path = stage3._store_canonical_object(root, "label_observation", snapshot)
    events = stage3.derive_label_events(decision["payload"]["proposals"], snapshot)
    events[0]["return_5d"] = 9.0
    label = {
        "ledger_id": stage3.study_identity(spec),
        "record_type": "label_completion",
        "record_hash": "8" * 64,
        "recorded_at_utc": "2026-08-10T08:00:00.000000Z",
        "payload": {
            "schema": "xs_chan_stage3_label_completion_v1",
            "decision_record_hash": decision["record_hash"],
            "decision_dt": decision["payload"]["decision_dt"],
            "entry_dt": decision["payload"]["entry_dt"],
            "exit_dt": decision["payload"]["exit_dt"],
            "raw_source_closure_sha256": closure_sha,
            "label_observation_sha256": observation_sha,
            "label_observation_object": str(observation_path.relative_to(root)),
            "events": events,
            "event_count": len(events),
        },
    }

    with pytest.raises(stage3.Stage3ValidationError, match="events do not replay"):
        stage3.validate_record_semantics(spec, [genesis, decision, label], root=root)


def test_semantic_replay_rejects_arbitrary_ledger_identity():
    spec = _spec()
    genesis = _semantic_genesis(spec, ledger_id="f" * 64)

    with pytest.raises(stage3.Stage3ValidationError, match="ledger identity"):
        stage3.validate_record_semantics(spec, [genesis])


def test_append_after_terminal_record_is_rejected_and_preserved(tmp_path: Path):
    root = tmp_path / "terminal"
    stage3.append_record(
        root,
        "final_evaluation",
        "final-evaluation:52w",
        {"status": "INCONCLUSIVE"},
        recorded_at_utc="2027-08-01T08:00:00Z",
    )

    with pytest.raises(stage3.Stage3ConflictError, match="after final"):
        stage3.append_record(
            root,
            "decision_freeze",
            "decision:2027-08-06",
            {"decision_dt": "2027-08-06"},
            recorded_at_utc="2027-08-06T08:00:00Z",
        )

    assert len(stage3.scan_records(root)) == 1
    failures = list((root / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8"))["reason"] == "TERMINAL_CHAIN_ALREADY_FINALIZED"


def test_semantic_replay_rejects_deleted_decision_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _spec()
    root = tmp_path / "deleted-decision-object"
    genesis = _semantic_genesis(spec)
    decision, sessions, manifest = _semantic_decision(spec)
    digest, object_path = stage3._store_canonical_object(root, "decision_path", {"placeholder": True})
    decision["payload"]["decision_path_sha256"] = digest
    decision["payload"]["decision_path_object"] = str(object_path.relative_to(root))
    object_path.unlink()
    _stub_semantic_dependencies(monkeypatch, sessions, manifest)

    with pytest.raises(stage3.Stage3ValidationError, match="missing frozen decision_path object"):
        stage3.validate_record_semantics(spec, [genesis, decision], root=root)


def test_semantic_replay_rejects_deleted_final_evaluation_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _spec()
    spec["accrual"]["window_weeks"] = 0
    root = tmp_path / "deleted-final-object"
    genesis = _semantic_genesis(spec)
    digest, object_path = stage3._store_canonical_object(root, "final_evaluation", {"placeholder": True})
    object_path.unlink()
    final = {
        "ledger_id": stage3.study_identity(spec),
        "record_type": "final_evaluation",
        "record_hash": "9" * 64,
        "recorded_at_utc": "2026-07-28T07:00:00.000000Z",
        "payload": {
            "schema": "xs_chan_stage3_final_evaluation_commit_v1",
            "collection_chain_head": genesis["record_hash"],
            "evaluation_sha256": digest,
            "evaluation_object": str(object_path.relative_to(root)),
            "status": "INCONCLUSIVE",
            "claim_boundary": spec["claim_boundary"],
        },
    }
    monkeypatch.setattr(stage3, "validate_protocol_anchor", lambda _spec: "a" * 40)

    with pytest.raises(stage3.Stage3ValidationError, match="missing frozen final_evaluation object"):
        stage3.validate_record_semantics(spec, [genesis, final], root=root)
