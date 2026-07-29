from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import xs_chan_exploration_stage3 as stage3  # noqa: E402
import xs_chan_stage3_weekly as weekly  # noqa: E402

FROZEN_COLLECTOR_SHA256 = "4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea"
FROZEN_SPEC_SHA256 = "3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa"


def _spec() -> dict[str, Any]:
    return {"accrual": {"window_weeks": 52}}


def _decision(
    index: int,
    sequence: int,
    decision_date: str,
    entry_date: str,
    exit_date: str,
) -> dict[str, Any]:
    return {
        "ledger_id": "f" * 64,
        "sequence": sequence,
        "record_hash": f"{index:064x}",
        "previous_hash": f"{max(index - 1, 0):064x}",
        "record_type": "decision_freeze",
        "recorded_at_utc": f"{decision_date}T07:00:00.000000Z",
        "logical_event_key": f"decision:{decision_date}",
        "payload": {
            "classification": "PROSPECTIVE_COUNTED",
            "week_index": index,
            "decision_dt": decision_date,
            "entry_dt": entry_date,
            "exit_dt": exit_date,
            "reference_manifest_sha256": "a" * 64,
            "proposals": [],
        },
    }


def _label(
    decision: dict[str, Any],
    sequence: int,
    *,
    recorded_at: str = "2026-08-10T10:00:00.000000Z",
) -> dict[str, Any]:
    return {
        "ledger_id": "f" * 64,
        "sequence": sequence,
        "record_hash": f"{sequence + 100:064x}",
        "previous_hash": f"{sequence + 99:064x}",
        "record_type": "label_completion",
        "recorded_at_utc": recorded_at,
        "logical_event_key": f"label:{decision['payload']['decision_dt']}",
        "payload": {
            "decision_record_hash": decision["record_hash"],
            "decision_dt": decision["payload"]["decision_dt"],
            "entry_dt": decision["payload"]["entry_dt"],
            "exit_dt": decision["payload"]["exit_dt"],
        },
    }


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "week_index": 1,
                "decision_dt": pd.Timestamp("2026-07-31"),
                "entry_dt": pd.Timestamp("2026-08-03"),
                "exit_dt": pd.Timestamp("2026-08-10"),
            },
            {
                "week_index": 2,
                "decision_dt": pd.Timestamp("2026-08-07"),
                "entry_dt": pd.Timestamp("2026-08-10"),
                "exit_dt": pd.Timestamp("2026-08-17"),
            },
            {
                "week_index": 3,
                "decision_dt": pd.Timestamp("2026-08-14"),
                "entry_dt": pd.Timestamp("2026-08-17"),
                "exit_dt": pd.Timestamp("2026-08-24"),
            },
            {
                "week_index": 4,
                "decision_dt": pd.Timestamp("2026-08-21"),
                "entry_dt": pd.Timestamp("2026-08-24"),
                "exit_dt": pd.Timestamp("2026-08-31"),
            },
        ]
    )


def _install_schedule(monkeypatch: pytest.MonkeyPatch, schedule: pd.DataFrame | None = None) -> None:
    monkeypatch.setattr(
        weekly,
        "_schedule_for_decision",
        lambda *_args, **_kwargs: (schedule if schedule is not None else _schedule()),
    )


def test_frozen_collector_and_spec_bytes_are_unchanged() -> None:
    assert stage3.sha256_file(stage3.SOURCE_PATH) == FROZEN_COLLECTOR_SHA256
    # Keep the literal split out of the operator; this test catches accidental edits.
    assert stage3.sha256_file(stage3.SPEC_PATH) == FROZEN_SPEC_SHA256


@pytest.mark.parametrize(
    ("clock", "allowed"),
    [
        (datetime(2026, 8, 10, 7, 0, tzinfo=UTC), False),  # 15:00 Shanghai
        (datetime(2026, 8, 10, 9, 59, 59, tzinfo=UTC), False),
        (datetime(2026, 8, 10, 10, 0, tzinfo=UTC), True),  # 18:00 Shanghai
    ],
)
def test_label_release_is_18_not_exit_close(clock: datetime, allowed: bool) -> None:
    if allowed:
        assert weekly.validate_label_release("2026-08-10", now=clock)
    else:
        with pytest.raises(weekly.WeeklyOperationError, match="18:00"):
            weekly.validate_label_release("2026-08-10", now=clock)


def test_d2_is_required_before_l1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_schedule(monkeypatch)
    d1 = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    d2 = _decision(2, 2, "2026-08-07", "2026-08-10", "2026-08-17")
    target = weekly.derive_oldest_unlabeled_target(
        _spec(),
        tmp_path,
        [d1, d2],
        now=datetime(2026, 8, 10, 10, tzinfo=UTC),
    )
    assert target is not None
    assert [row["record_hash"] for row in target.required_later_decisions] == [
        d2["record_hash"]
    ]


def test_long_holiday_requires_every_decision_through_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    schedule = _schedule()
    schedule.loc[schedule["week_index"].eq(1), "exit_dt"] = pd.Timestamp("2026-08-18")
    _install_schedule(monkeypatch, schedule)
    d1 = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-18")
    d2 = _decision(2, 2, "2026-08-07", "2026-08-10", "2026-08-17")
    d3 = _decision(3, 3, "2026-08-14", "2026-08-17", "2026-08-24")
    target = weekly.derive_oldest_unlabeled_target(
        _spec(),
        tmp_path,
        [d1, d2, d3],
        now=datetime(2026, 8, 18, 10, tzinfo=UTC),
    )
    assert target is not None
    assert [row["payload"]["week_index"] for row in target.required_later_decisions] == [2, 3]


def test_next_decision_close_has_priority_over_overdue_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_schedule(monkeypatch)
    records = [
        _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10"),
        _decision(2, 2, "2026-08-07", "2026-08-10", "2026-08-17"),
    ]
    with pytest.raises(weekly.RequiredDecisionPending) as raised:
        weekly.derive_oldest_unlabeled_target(
            _spec(),
            tmp_path,
            records,
            now=datetime(2026, 8, 14, 7, tzinfo=UTC),
        )
    assert raised.value.week_index == 3


def test_missed_next_decision_window_aborts_before_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_schedule(monkeypatch)
    records = [
        _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10"),
        _decision(2, 2, "2026-08-07", "2026-08-10", "2026-08-17"),
    ]
    with pytest.raises(weekly.StudyAbortedInvalid, match="MISSED_DECISION_WINDOW"):
        weekly.derive_oldest_unlabeled_target(
            _spec(),
            tmp_path,
            records,
            now=datetime(2026, 8, 17, 1, 30, tzinfo=UTC),
        )


def test_oldest_label_prefix_cannot_be_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_schedule(monkeypatch)
    d1 = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    d2 = _decision(2, 2, "2026-08-07", "2026-08-10", "2026-08-17")
    with pytest.raises(weekly.StudyAbortedInvalid, match="oldest-decision prefix"):
        weekly.derive_oldest_unlabeled_target(
            _spec(),
            tmp_path,
            [d1, d2, _label(d2, 3, recorded_at="2026-08-17T10:00:00Z")],
            now=datetime(2026, 8, 18, tzinfo=UTC),
        )


def test_same_day_label_before_required_decision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_schedule(monkeypatch)
    d1 = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    d2 = _decision(2, 3, "2026-08-07", "2026-08-10", "2026-08-17")
    l1 = _label(d1, 2)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        weekly,
        "validate_label_authorized_anchor_pair",
        lambda *_a, **_k: {},
    )
    with pytest.raises(weekly.StudyAbortedInvalid, match="calendar-priority"):
        weekly.validate_label_chain(
            _spec(),
            tmp_path,
            [d1, l1, d2],
            require_pushed=False,
        )


def test_direct_collector_label_is_permanently_unauthorized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "scripts"
    tracked.mkdir()
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tracked)
    d1 = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    with pytest.raises(weekly.StudyAbortedInvalid, match="direct collector label"):
        weekly.validate_label_sidecar_inventory(tmp_path / "ledger", [d1, _label(d1, 2)])


def test_inert_pre_record_authorization_does_not_brick_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    auth = {
        "anticipated_record": {"record_hash": "e" * 64},
        "expected_record_hash": "e" * 64,
    }
    digest = stage3.sha256_bytes(stage3.canonical_json(auth))
    path = root / "objects" / weekly.LABEL_AUTHORIZATION_CATEGORY / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(stage3.canonical_json(auth))
    tracked = tmp_path / "scripts"
    tracked.mkdir()
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tracked)

    weekly.validate_label_sidecar_inventory(root, [])


def test_symlinked_evidence_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(weekly.WeeklyOperationError, match="symlink"):
        weekly._stable_file_bytes(link, label="test evidence")


def test_label_sidecar_is_minimal_and_contains_no_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    record = _label(
        _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10"),
        2,
    )
    auth_path = tmp_path / "ledger" / "objects" / weekly.LABEL_AUTHORIZATION_CATEGORY / f"{'a' * 64}.json"
    monkeypatch.setattr(
        weekly,
        "load_label_append_authorization",
        lambda *_a, **_k: (
            {
                "operator_source_sha256": "b" * 64,
                "git": {"head": "c" * 40},
            },
            "a" * 64,
            auth_path,
        ),
    )
    sidecar, _ = weekly._expected_label_sidecar(
        {"study_id": "test"},
        tmp_path / "ledger",
        record,
        require_current_operator=False,
    )
    text = json.dumps(sidecar, sort_keys=True).lower()
    assert "anticipated_record" not in text
    assert "return_5d" not in text
    assert '"events"' not in text
    assert "authorization\"" not in text


def test_complete_label_without_apply_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        weekly,
        "status_snapshot",
        lambda **_kwargs: {"state": "TEST_READ_ONLY", "formal_ledger_mutated": False},
    )
    monkeypatch.setattr(
        weekly,
        "complete_next_label",
        lambda **_kwargs: pytest.fail("formal completion ran without --apply-ledger"),
    )
    assert weekly.main(["complete-label"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_NO_LEDGER_MUTATION"
    assert payload["snapshot"]["state"] == "TEST_READ_ONLY"


def test_unified_chain_has_no_cross_event_edge_for_genesis_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = {
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {},
    }
    monkeypatch.setattr(weekly, "validate_label_chain", lambda *_a, **_k: [])
    monkeypatch.setattr(
        weekly.first_week,
        "validate_anchor_for_record",
        lambda *_a, **_k: pytest.fail("genesis anchor queried without a D/L edge"),
    )
    assert (
        weekly.validate_stage3_operator_pair_chain(
            _spec(),
            tmp_path,
            [genesis],
            require_pushed=True,
        )
        == []
    )


def test_unified_chain_seeds_causality_from_actual_genesis_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = {
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {"protocol_anchor_git_commit": "1" * 40},
    }
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    monkeypatch.setattr(weekly, "validate_label_chain", lambda *_a, **_k: [])
    monkeypatch.setattr(
        weekly.first_week,
        "validate_anchor_for_record",
        lambda *_a, **_k: {"commit": "2" * 40},
    )
    monkeypatch.setattr(
        weekly,
        "validate_record_authorized_anchor_pair",
        lambda *_a, **_k: {
            "record_hash": decision["record_hash"],
            "record_type": "decision_freeze",
            "authorization_sha256": "a" * 64,
            "authorized_git_head": "3" * 40,
            "commit": "4" * 40,
        },
    )
    edges: list[tuple[str, str]] = []
    monkeypatch.setattr(
        weekly,
        "_require_git_ancestor",
        lambda ancestor, descendant, **_kwargs: edges.append((ancestor, descendant)),
    )
    weekly.validate_stage3_operator_pair_chain(
        _spec(),
        tmp_path,
        [genesis, decision],
        require_pushed=True,
    )
    assert edges == [("2" * 40, "3" * 40)]


def test_unified_chain_scopes_and_ignores_current_decision_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = {
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {},
    }
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    records = [genesis, decision]
    ignored = {"000002_current-decision.json"}
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(weekly, "validate_label_chain", lambda *_a, **_k: [])
    monkeypatch.setattr(
        weekly.first_week,
        "validate_anchor_for_record",
        lambda *_a, **_k: {"commit": "1" * 40},
    )

    def validate_decision_pair(
        *_args: Any,
        inventory_records: object,
        ignored_sidecar_names: object,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        observed.append((inventory_records, ignored_sidecar_names))
        return {"anchor": {}, "commit": "3" * 40}

    monkeypatch.setattr(
        weekly.first_week,
        "validate_authorized_anchor_pair",
        validate_decision_pair,
    )
    monkeypatch.setattr(
        weekly.first_week,
        "load_append_authorization",
        lambda *_a, **_k: ({}, "a" * 64, tmp_path / "auth.json"),
    )
    monkeypatch.setattr(weekly, "_authorization_git", lambda _auth: {"head": "2" * 40})
    monkeypatch.setattr(weekly, "_require_git_ancestor", lambda *_a, **_k: None)
    weekly.validate_stage3_operator_pair_chain(
        _spec(),
        tmp_path,
        records,
        require_pushed=True,
        verify_actual_remote=False,
        ignored_decision_sidecar_names=ignored,
    )
    assert observed == [(records, ignored)]


def test_successor_edge_consumes_validated_prefix_without_remote_or_rescan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = {
        "ledger_id": "f" * 64,
        "sequence": 0,
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {},
    }
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    decision["ledger_id"] = genesis["ledger_id"]
    decision["previous_hash"] = genesis["record_hash"]
    label = _label(decision, 2)
    label["ledger_id"] = genesis["ledger_id"]
    label["previous_hash"] = decision["record_hash"]
    prefix_evidence = [
        {
            "record_hash": decision["record_hash"],
            "record_type": "decision_freeze",
            "commit": "a" * 40,
        }
    ]
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: pytest.fail("supplied prefix evidence was rescanned"),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("successor edge performed actual-remote I/O"),
    )
    monkeypatch.setattr(
        weekly,
        "load_label_append_authorization",
        lambda *_a, **_k: ({}, "b" * 64, tmp_path / "auth.json"),
    )
    monkeypatch.setattr(weekly, "_authorization_git", lambda _auth: {"head": "c" * 40})
    monkeypatch.setattr(
        weekly,
        "_require_git_ancestor",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.WeeklyOperationError("broken successor edge")
        ),
    )

    with pytest.raises(weekly.WeeklyOperationError, match="broken successor edge"):
        weekly.validate_successor_authorized_head(
            _spec(),
            tmp_path,
            [genesis, decision],
            label,
            prefix_pair_evidence=prefix_evidence,
        )


def test_later_failure_does_not_invalidate_historical_failure_inventory(
    tmp_path: Path,
) -> None:
    failures = tmp_path / "failures"
    failures.mkdir()
    historical_path = failures / "0001.json"
    historical_path.write_bytes(b"historical")
    historical = weekly.first_week._file_inventory(tmp_path, "failures")
    (failures / "0002.json").write_bytes(b"later")

    assert weekly._validate_historical_failure_inventory(
        tmp_path,
        {"failure_files": historical},
    ) == historical


def test_historical_failure_tamper_is_rejected(tmp_path: Path) -> None:
    failures = tmp_path / "failures"
    failures.mkdir()
    historical_path = failures / "0001.json"
    historical_path.write_bytes(b"historical")
    historical = weekly.first_week._file_inventory(tmp_path, "failures")
    historical_path.write_bytes(b"tampered")

    with pytest.raises(
        weekly.WeeklyOperationError,
        match="historical preserved failure",
    ):
        weekly._validate_historical_failure_inventory(
            tmp_path,
            {"failure_files": historical},
        )


def test_preflight_receipt_rejects_any_contract_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_bytes(stage3.canonical_json({"state": "bound"}))
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    target = weekly.LabelTarget(1, decision, ())
    evidence = weekly.LabelEvidence(
        raw={"audit_sha256": "a" * 64, "parquet_inventory_sha256": "b" * 64},
        state={"manifest_sha256": stage3.sha256_file(state_path)},
        closure={"coverage_max_dt": "2026-08-10"},
        closure_sha256="c" * 64,
        snapshot={},
        observation_sha256="d" * 64,
        payload={"event_count": 0},
    )
    git = {"head": "e" * 40}
    clock = datetime(2026, 8, 10, 10, tzinfo=UTC)
    ledger = {"head": decision["record_hash"], "record_count": 1}
    spec = {**_spec(), "study_id": "test"}
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (decision,))
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: {"status": "NO_PENDING_DECISION_WINDOW"},
    )
    monkeypatch.setattr(
        weekly.first_week,
        "load_append_authorization",
        lambda *_a, **_k: ({}, "1" * 64, tmp_path / "decision-auth.json"),
    )
    report = {
        "schema": "xs_chan_stage3_weekly_label_preflight_v1",
        "mode": "READ_ONLY_OLDEST_LABEL",
        "generated_at_utc": clock,
        "study_id": "test",
        "study_identity": "f" * 64,
        "spec_physical_sha256": stage3.sha256_file(stage3.SPEC_PATH),
        "collector_source_sha256": stage3.sha256_file(stage3.SOURCE_PATH),
        "operator_source_sha256": stage3.sha256_file(weekly.OPERATOR_SOURCE_PATH),
        "target": target.summary(),
        "global_decision_priority": {"status": "NO_PENDING_DECISION_WINDOW"},
        "time_gate": weekly.validate_label_release(target.exit_date, now=clock),
        "git": git,
        "required_decision_evidence": [
            {
                "record_hash": decision["record_hash"],
                "record_type": "decision_freeze",
                "authorization_sha256": "1" * 64,
            }
        ],
        "existing_label_evidence": [],
        "exit_evidence": evidence.summary(),
        "state_manifest_path": str(state_path),
        "state_manifest_sha256": stage3.sha256_file(state_path),
        "ledger_before": ledger,
        "ledger_after": ledger,
        "expected_head": decision["record_hash"],
        "formal_ledger_mutated": False,
        "event_payload_emitted": False,
        "efficacy_output": "FORBIDDEN",
    }
    weekly._validate_label_preflight_receipt(
        spec,
        tmp_path,
        report,
        target=target,
        evidence=evidence,
        state_manifest_path=state_path,
        expected_head=decision["record_hash"],
        git=git,
        operator_source_sha256=stage3.sha256_file(weekly.OPERATOR_SOURCE_PATH),
    )
    mutated = dict(report)
    mutated["event_payload_emitted"] = True
    with pytest.raises(weekly.WeeklyOperationError, match="does not exactly bind"):
        weekly._validate_label_preflight_receipt(
            spec,
            tmp_path,
            mutated,
            target=target,
            evidence=evidence,
            state_manifest_path=state_path,
            expected_head=decision["record_hash"],
            git=git,
            operator_source_sha256=stage3.sha256_file(weekly.OPERATOR_SOURCE_PATH),
        )


def _install_formal_preflight_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    weekly.LabelTarget,
    weekly.LabelEvidence,
    Path,
    dict[str, Any],
]:
    root = tmp_path / "ledger"
    records_dir = root / "records"
    records_dir.mkdir(parents=True)
    state_path = tmp_path / "state.json"
    state_path.write_bytes(stage3.canonical_json({"state": "bound"}))
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    record_path = records_dir / "000001.json"
    record_raw = stage3.canonical_json(decision)
    record_path.write_bytes(record_raw)
    record = stage3.LedgerRecord(record_path, decision, record_raw)
    target = weekly.LabelTarget(1, decision, ())
    evidence = weekly.LabelEvidence(
        raw={"audit_sha256": "a" * 64, "parquet_inventory_sha256": "b" * 64},
        state={"manifest_sha256": stage3.sha256_file(state_path)},
        closure={"coverage_max_dt": "2026-08-10"},
        closure_sha256="c" * 64,
        snapshot={},
        observation_sha256="d" * 64,
        payload={"event_count": 0},
    )
    git = {"head": "e" * 40}
    pair = {
        "record_hash": decision["record_hash"],
        "record_type": "decision_freeze",
        "anchor": {
            "path": "/tracked/anchor.json",
            "sha256": "2" * 64,
            "record_hash": decision["record_hash"],
            "commit": "3" * 40,
            "deadline": datetime(2026, 8, 2, 17, 30, tzinfo=UTC),
        },
        "authorization_sha256": "1" * 64,
        "authorization_path": "/ledger/objects/authorization.json",
        "commit": "3" * 40,
        "authorized_git_head": "4" * 40,
    }
    record_files = [
        {
            "path": str(record_path.relative_to(root)),
            "size": len(record_raw),
            "sha256": stage3.sha256_file(record_path),
        }
    ]
    ledger = {
        "record_count": 1,
        "head": decision["record_hash"],
        "head_type": "decision_freeze",
        "record_files": record_files,
        "failure_files": [],
        "closure_sha256": stage3.sha256_bytes(
            stage3.canonical_json({"records": record_files, "failures": []})
        ),
    }
    spec = {**_spec(), "study_id": "test"}
    clock = datetime(2026, 8, 10, 10, tzinfo=UTC)
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (record,))
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: {"status": "NO_PENDING_DECISION_WINDOW"},
    )
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: [pair],
    )
    monkeypatch.setattr(
        weekly.first_week,
        "load_append_authorization",
        lambda *_a, **_k: ({}, "1" * 64, tmp_path / "decision-auth.json"),
    )
    report = {
        "schema": "xs_chan_stage3_weekly_label_preflight_v1",
        "mode": "READ_ONLY_OLDEST_LABEL",
        "generated_at_utc": clock,
        "study_id": "test",
        "study_identity": "f" * 64,
        "spec_physical_sha256": stage3.sha256_file(stage3.SPEC_PATH),
        "collector_source_sha256": stage3.sha256_file(stage3.SOURCE_PATH),
        "operator_source_sha256": stage3.sha256_file(weekly.OPERATOR_SOURCE_PATH),
        "target": target.summary(),
        "global_decision_priority": {"status": "NO_PENDING_DECISION_WINDOW"},
        "time_gate": weekly.validate_label_release(target.exit_date, now=clock),
        "git": git,
        "required_decision_evidence": [pair],
        "existing_label_evidence": [],
        "exit_evidence": evidence.summary(),
        "state_manifest_path": str(state_path),
        "state_manifest_sha256": stage3.sha256_file(state_path),
        "ledger_before": ledger,
        "ledger_after": ledger,
        "expected_head": decision["record_hash"],
        "formal_ledger_mutated": False,
        "event_payload_emitted": False,
        "efficacy_output": "FORBIDDEN",
    }
    return spec, root, report, target, evidence, state_path, git


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("delete-commit", None),
        ("commit", "5" * 40),
        ("authorized_git_head", "6" * 40),
        ("anchor", {"path": "/changed-anchor.json"}),
    ],
)
def test_historical_preflight_requires_full_exact_pair_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    spec, root, report, target, evidence, state_path, git = (
        _install_formal_preflight_receipt(monkeypatch, tmp_path)
    )
    mutated = json.loads(
        weekly.canonical_json(weekly.first_week._json_safe(report))
    )
    item = mutated["required_decision_evidence"][0]
    if mutation == "delete-commit":
        del item["commit"]
    else:
        item[mutation] = value

    with pytest.raises(
        weekly.WeeklyOperationError,
        match="exact historical prefix",
    ):
        weekly._validate_label_preflight_receipt(
            spec,
            root,
            mutated,
            target=target,
            evidence=evidence,
            state_manifest_path=state_path,
            expected_head=report["expected_head"],
            git=git,
            operator_source_sha256=report["operator_source_sha256"],
        )


def _install_rebuildable_label_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    root = tmp_path / "ledger"
    root.mkdir()
    state_path = tmp_path / "state.json"
    state_path.write_bytes(stage3.canonical_json({"state": "bound"}))
    state_sha = stage3.sha256_file(state_path)
    closure = {
        "coverage_max_dt": "2026-08-10",
        "state_manifest_sha256": state_sha,
    }
    closure_sha = stage3.sha256_bytes(stage3.canonical_json(closure))
    closure_path = root / "objects" / "raw_source_closure" / f"{closure_sha}.json"
    closure_path.parent.mkdir(parents=True)
    closure_path.write_bytes(stage3.canonical_json(closure))
    snapshot = {
        "entry_dt": "2026-08-03",
        "exit_dt": "2026-08-10",
        "raw_source_closure_sha256": closure_sha,
        "raw_source_closure_object": str(closure_path.relative_to(root)),
    }
    observation_sha = stage3.sha256_bytes(stage3.canonical_json(snapshot))
    observation_path = (
        root / "objects" / "label_observation" / f"{observation_sha}.json"
    )
    observation_path.parent.mkdir(parents=True)
    observation_path.write_bytes(stage3.canonical_json(snapshot))
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    decision["payload"]["proposals"] = []
    payload = {
        "schema": "xs_chan_stage3_label_completion_v1",
        "decision_record_hash": decision["record_hash"],
        "decision_dt": "2026-07-31",
        "entry_dt": "2026-08-03",
        "exit_dt": "2026-08-10",
        "raw_source_closure_sha256": closure_sha,
        "label_observation_sha256": observation_sha,
        "label_observation_object": str(observation_path.relative_to(root)),
        "events": [],
        "event_count": 0,
    }
    label = _label(decision, 2)
    label["payload"] = payload
    target = weekly.LabelTarget(1, decision, ())
    evidence = weekly.LabelEvidence(
        raw={
            "audit_sha256": "a" * 64,
            "parquet_inventory_sha256": "b" * 64,
        },
        state={
            "manifest_sha256": state_sha,
            "raw_parquet_inventory_sha256": "b" * 64,
        },
        closure=closure,
        closure_sha256=closure_sha,
        snapshot=snapshot,
        observation_sha256=observation_sha,
        payload=payload,
    )
    git = _authorized_git("c" * 40)["git"]
    preflight = {
        "git": git,
        "exit_evidence": evidence.summary(),
    }
    preflight_raw = stage3.canonical_json(preflight)
    preflight_sha = stage3.sha256_bytes(preflight_raw)
    preflight_path = (
        root / "objects" / weekly.LABEL_PREFLIGHT_CATEGORY / f"{preflight_sha}.json"
    )
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_bytes(preflight_raw)
    spec = {"study_id": "test", **_spec()}
    operator_sha = stage3.sha256_file(weekly.OPERATOR_SOURCE_PATH)
    decision_record = stage3.LedgerRecord(
        tmp_path / "decision.json",
        decision,
        b"",
    )
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (decision_record,))
    monkeypatch.setattr(stage3, "derive_label_events", lambda *_a, **_k: [])
    monkeypatch.setattr(
        weekly,
        "_schedule_for_decision",
        lambda *_a, **_k: pd.DataFrame(
            [
                {
                    "week_index": 1,
                    "decision_dt": pd.Timestamp("2026-07-31"),
                    "entry_dt": pd.Timestamp("2026-08-03"),
                    "exit_dt": pd.Timestamp("2026-08-10"),
                }
            ]
        ),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "_git_file_sha256_at_commit",
        lambda *_a, **_k: operator_sha,
    )
    monkeypatch.setattr(
        weekly,
        "_validate_label_preflight_receipt",
        lambda *_a, **_k: None,
    )
    authorization = weekly._build_label_authorization(
        spec,
        root,
        label,
        target=target,
        evidence=evidence,
        state_manifest_path=state_path,
        preflight_report=preflight,
        preflight_path=preflight_path,
        git=git,
        operator_source_sha256=operator_sha,
        validate_preflight_receipt=False,
    )
    return spec, root, label, authorization


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("week_index", 99),
        ("authorization_created_at_utc", "2026-08-10T10:00:01.000000Z"),
        ("time_gate", {"checked_at_utc": "changed"}),
    ],
)
def test_historical_label_authorization_is_fully_canonical_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    spec, root, label, authorization = _install_rebuildable_label_authorization(
        monkeypatch,
        tmp_path,
    )
    mutated = dict(authorization)
    mutated[field] = value
    raw = stage3.canonical_json(mutated)
    digest = stage3.sha256_bytes(raw)
    path = root / "objects" / weekly.LABEL_AUTHORIZATION_CATEGORY / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    monkeypatch.setattr(
        weekly,
        "_authorization_objects",
        lambda _root: [(path, mutated)],
    )

    with pytest.raises(
        weekly.WeeklyOperationError,
        match="exact canonical historical authorization",
    ):
        weekly.load_label_append_authorization(
            spec,
            root,
            label,
            require_current_operator=False,
        )


def test_nested_label_authorization_replay_uses_one_validation_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        {"record_hash": f"{index:064x}"}
        for index in range(1, 6)
    ]
    calls: dict[str, int] = {}

    def nested_loader(
        spec: Mapping[str, Any],
        root: Path,
        record: Mapping[str, Any],
        *,
        require_current_operator: bool,
    ) -> tuple[dict[str, Any], str, Path]:
        record_hash = str(record["record_hash"])
        calls[record_hash] = calls.get(record_hash, 0) + 1
        index = records.index(record)
        for prior in records[:index]:
            weekly.load_label_append_authorization(
                spec,
                root,
                prior,
                require_current_operator=require_current_operator,
            )
        return {}, record_hash, tmp_path / f"{record_hash}.json"

    monkeypatch.setattr(
        weekly,
        "_load_label_append_authorization_impl",
        nested_loader,
    )
    weekly.load_label_append_authorization(
        _spec(),
        tmp_path,
        records[-1],
        require_current_operator=False,
    )
    assert calls == {record["record_hash"]: 1 for record in records}


def test_label_raw_closure_cannot_advance_past_exact_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    target = weekly.LabelTarget(1, decision, ())
    state_path = tmp_path / "state.json"
    state_path.write_bytes(stage3.canonical_json({"state": "test"}))
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "validate_raw_ready",
        lambda *_a, **_k: {"parquet_inventory_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_state_ready",
        lambda *_a, **_k: {
            "raw_parquet_inventory_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        stage3,
        "capture_label_observations",
        lambda *_a, **_k: (
            {},
            {
                "coverage_max_dt": "2026-08-11",
                "state_manifest_sha256": "b" * 64,
            },
        ),
    )
    with pytest.raises(weekly.WeeklyOperationError, match="end exactly"):
        weekly.build_label_evidence(
            target,
            state_path,
            data_dir=tmp_path,
        )


def test_posthoc_label_authorization_is_rejected_before_evidence_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "weekly posthoc test"},
        "2026-08-10T09:00:00Z",
    )
    label = stage3.append_record(
        root,
        "label_completion",
        "label:2026-07-31",
        {"decision_dt": "2026-07-31"},
        "2026-08-10T10:00:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    dummy = weekly.LabelEvidence({}, {}, {}, "a" * 64, {}, "b" * 64, {})
    target = weekly.LabelTarget(
        1,
        {
            "record_hash": "c" * 64,
            "payload": {
                "decision_dt": "2026-07-31",
                "entry_dt": "2026-08-03",
                "exit_dt": "2026-08-10",
            },
        },
        (),
    )
    with (
        weekly.first_week.authorization_ledger_lock(root),
        pytest.raises(weekly.WeeklyOperationError, match="before the record exists"),
    ):
        weekly.store_label_append_authorization(
            {"study_id": "unused"},
            root,
            label.data,
            target=target,
            evidence=dummy,
            state_manifest_path=tmp_path / "missing-state.json",
            preflight_report={},
            fresh_git={},
        )


def _install_recovery_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], list[bool]]:
    label = _label(
        _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10"),
        2,
    )
    record = SimpleNamespace(data=label)
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    monkeypatch.setattr(stage3, "load_and_validate_spec", lambda: {"study_id": "test"})
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (record,))
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(weekly.first_week, "operator_lock", lambda _root: nullcontext())
    monkeypatch.setattr(weekly, "cache_lock", lambda _root: nullcontext())
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    requested_current: list[bool] = []

    def load_auth(*_args: Any, require_current_operator: bool, **_kwargs: Any):
        requested_current.append(require_current_operator)
        return (
            {"git": {"head": "a" * 40}},
            "b" * 64,
            root / "objects" / "auth.json",
        )

    monkeypatch.setattr(weekly, "load_label_append_authorization", load_auth)
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_a, **_k: {"actual_remote_verified": False},
    )
    return label, requested_current


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authorized_git(head: str) -> dict[str, Any]:
    return {
        "git": {
            "branch": weekly.first_week.REQUIRED_GIT_BRANCH,
            "head": head,
            "upstream": weekly.first_week.REQUIRED_GIT_UPSTREAM,
            "upstream_head": head,
            "remote_head": head,
            "remote_fetch_url": weekly.first_week.REQUIRED_GIT_REMOTE_URL,
            "remote_push_url": weekly.first_week.REQUIRED_GIT_REMOTE_URL,
            "remote_verified": True,
            "worktree_clean": True,
        }
    }


def _install_exact_committed_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    extra_path: bool = False,
) -> tuple[Path, Path, dict[str, Any], str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "pair-test")
    _git(repo, "config", "user.name", "Stage3 Test")
    _git(repo, "config", "user.email", "stage3@example.invalid")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-m", "authorized parent")
    authorized_head = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(stage3, "REPO_ROOT", repo)
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", repo / "scripts")
    root = repo / "ledger"
    root.mkdir()
    label = _label(
        _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10"),
        2,
    )
    anchor_path = weekly._anchor_path(label)
    sidecar_path = weekly._label_sidecar_path(label)
    anchor_path.parent.mkdir(parents=True)
    sidecar_path.parent.mkdir(parents=True)
    anchor_path.write_text("anchor\n", encoding="utf-8")
    expected_sidecar = {"schema": "minimal-test-sidecar", "bound": label["record_hash"]}
    sidecar_path.write_bytes(
        json.dumps(
            expected_sidecar,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if extra_path:
        (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "exact label pair")
    pair_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        weekly,
        "_expected_label_sidecar",
        lambda *_a, **_k: (expected_sidecar, sidecar_path),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_anchor_for_record",
        lambda *_a, **_k: {"record_hash": label["record_hash"]},
    )
    return root, repo, label, authorized_head, pair_commit


def test_inspect_committed_label_pair_accepts_exact_clean_pushed_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _repo, label, authorized_head, pair_commit = _install_exact_committed_pair(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: {"remote_head": pair_commit},
    )

    inspected = weekly._inspect_committed_label_pair(
        {"study_id": "test"},
        root,
        label,
        _authorized_git(authorized_head),
    )

    assert inspected == {
        "commit": pair_commit,
        "authorized_head": authorized_head,
        "remote_head": pair_commit,
        "remote_contains_pair": True,
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("wrong-parent", "authorized head as parent"),
        ("extra-path", "extra or missing paths"),
        ("dirty", "clean worktree"),
        ("remote-diverged", "neither the authorized parent nor a descendant"),
    ],
)
def test_inspect_committed_label_pair_rejects_invalid_local_or_remote_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    root, repo, label, authorized_head, pair_commit = _install_exact_committed_pair(
        monkeypatch,
        tmp_path,
        extra_path=failure == "extra-path",
    )
    authorization = _authorized_git(
        pair_commit if failure == "wrong-parent" else authorized_head
    )
    remote_head = pair_commit
    if failure == "dirty":
        (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    elif failure == "remote-diverged":
        _git(repo, "switch", "-c", "remote-diverged", authorized_head)
        (repo / "remote-only.txt").write_text("remote\n", encoding="utf-8")
        _git(repo, "add", "remote-only.txt")
        _git(repo, "commit", "-m", "diverged remote")
        remote_head = _git(repo, "rev-parse", "HEAD")
        _git(repo, "switch", "pair-test")
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: {"remote_head": remote_head},
    )

    with pytest.raises(weekly.WeeklyOperationError, match=message):
        weekly._inspect_committed_label_pair(
            {"study_id": "test"},
            root,
            label,
            authorization,
        )


def test_pushed_label_recovery_is_idempotent_before_current_raw_or_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    label, requested_current = _install_recovery_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(
        weekly,
        "_inspect_committed_label_pair",
        lambda *_a, **_k: {
            "commit": "c" * 40,
            "remote_head": "d" * 40,
            "remote_contains_pair": True,
        },
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_raw_ready",
        lambda *_a, **_k: pytest.fail("pushed recovery touched current raw"),
    )
    result = weekly.recover_label_evidence(
        record_hash=label["record_hash"],
        decision_date_assertion=label["payload"]["decision_dt"],
        data_dir=tmp_path / "raw",
    )
    assert result["status"] == "AUTHORIZED_LABEL_EVIDENCE_ALREADY_PUSHED"
    assert requested_current == [False]


def test_pushed_current_pair_cannot_hide_a_broken_successor_edge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    label, _ = _install_recovery_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(
        weekly,
        "_inspect_committed_label_pair",
        lambda *_a, **_k: {
            "commit": "c" * 40,
            "remote_head": "d" * 40,
            "remote_contains_pair": True,
        },
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.WeeklyOperationError("broken successor edge")
        ),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("successor recovery performed extra actual-remote I/O"),
    )

    with pytest.raises(weekly.WeeklyOperationError, match="broken successor edge"):
        weekly.recover_label_evidence(
            record_hash=label["record_hash"],
            decision_date_assertion=None,
            data_dir=tmp_path / "raw",
        )


def test_unpushed_exact_pair_waits_for_push_without_recapturing_raw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    label, requested_current = _install_recovery_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(
        weekly,
        "_inspect_committed_label_pair",
        lambda *_a, **_k: {
            "commit": "c" * 40,
            "remote_head": "a" * 40,
            "remote_contains_pair": False,
        },
    )
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: {"status": "NEXT_DECISION_WINDOW_OPEN_OR_FUTURE"},
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_raw_ready",
        lambda *_a, **_k: pytest.fail("committed pair recovery touched current raw"),
    )
    result = weekly.recover_label_evidence(
        record_hash=label["record_hash"],
        decision_date_assertion=None,
        data_dir=tmp_path / "raw",
    )
    assert result["status"] == "LABEL_EVIDENCE_COMMIT_AWAITING_PUSH"
    assert requested_current == [False]


def test_unpushed_pair_cannot_hide_a_missed_next_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    label, _ = _install_recovery_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(
        weekly,
        "_inspect_committed_label_pair",
        lambda *_a, **_k: {
            "commit": "c" * 40,
            "remote_head": "a" * 40,
            "remote_contains_pair": False,
        },
    )
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.StudyAbortedInvalid("MISSED_DECISION_WINDOW")
        ),
    )
    with pytest.raises(weekly.StudyAbortedInvalid, match="MISSED_DECISION_WINDOW"):
        weekly.recover_label_evidence(
            record_hash=label["record_hash"],
            decision_date_assertion=None,
            data_dir=tmp_path / "raw",
        )


def test_status_aborts_current_label_pending_when_successor_edge_is_broken(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = {
        "ledger_id": "f" * 64,
        "sequence": 0,
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {},
    }
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    label = _label(decision, 2)
    label["ledger_id"] = genesis["ledger_id"]
    label["previous_hash"] = genesis["record_hash"]
    records = (
        stage3.LedgerRecord(tmp_path / "genesis.json", genesis, b""),
        stage3.LedgerRecord(tmp_path / "label.json", label, b""),
    )
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    monkeypatch.setattr(stage3, "load_and_validate_spec", lambda: {"study_id": "test"})
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: genesis["ledger_id"])
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "ledger_closure",
        lambda _root: {"head": label["record_hash"], "record_count": 2},
    )
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        weekly,
        "_label_sidecar_path",
        lambda _record: tmp_path / "missing-sidecar.json",
    )
    monkeypatch.setattr(
        weekly,
        "_anchor_path",
        lambda _record: tmp_path / "missing-anchor.json",
    )
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.WeeklyOperationError("broken successor edge")
        ),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("status performed actual-remote I/O"),
    )

    result = weekly.status_snapshot(
        decision_date_assertion=None,
        state_manifest_path=None,
        data_dir=data,
    )

    assert result["state"] == "ABORTED_INVALID"
    assert result["reason"] == "broken successor edge"


def test_status_ready_hashes_datetime_pair_evidence_without_network_or_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    record = stage3.LedgerRecord(tmp_path / "decision.json", decision, b"")
    target = weekly.LabelTarget(1, decision, ())
    report = {
        "generated_at_utc": datetime(2026, 8, 10, 10, tzinfo=UTC),
        "time_gate": {
            "checked_at_utc": datetime(2026, 8, 10, 10, tzinfo=UTC),
        },
        "required_decision_evidence": [
            {
                "record_hash": decision["record_hash"],
                "anchor": {
                    "deadline": datetime(2026, 8, 2, 17, 30, tzinfo=UTC),
                },
            }
        ],
    }
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        stage3,
        "load_and_validate_spec",
        lambda: {"study_id": "test", **_spec()},
    )
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (record,))
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "ledger_closure",
        lambda _root: {"head": decision["record_hash"], "record_count": 1},
    )
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    network_scopes: list[bool] = []
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, verify_actual_remote, **_k: (
            network_scopes.append(verify_actual_remote) or []
        ),
    )
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: {"status": "NO_PENDING_DECISION_WINDOW"},
    )
    monkeypatch.setattr(
        weekly,
        "derive_oldest_unlabeled_target",
        lambda *_a, **_k: target,
    )
    monkeypatch.setattr(
        weekly,
        "inspect_inventory",
        lambda _data: {"max_dt": "20260810"},
    )
    monkeypatch.setattr(
        weekly,
        "build_label_preflight",
        lambda *_a, verify_remote, **_k: (
            pytest.fail("READY status requested actual remote")
            if verify_remote
            else (report, target, object())
        ),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("READY status performed actual-remote I/O"),
    )
    monkeypatch.setattr(
        stage3,
        "_exclusive_write",
        lambda *_a, **_k: pytest.fail("READY status wrote formal evidence"),
    )

    result = weekly.status_snapshot(
        decision_date_assertion=None,
        state_manifest_path=state_path,
        data_dir=data,
        now=datetime(2026, 8, 10, 10, tzinfo=UTC),
    )

    assert result["state"] == "READY_TO_COMPLETE_ONE_LABEL"
    assert result["preflight_sha256"] == stage3.sha256_bytes(
        stage3.canonical_json(report)
    )
    assert network_scopes == [False]


@pytest.mark.parametrize("recovery_case", ["missing-sidecar", "unpushed-exact-pair"])
def test_status_maps_recoverable_current_decision_to_wait_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recovery_case: str,
) -> None:
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    record = stage3.LedgerRecord(tmp_path / "decision.json", decision, b"")
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    monkeypatch.setattr(
        stage3,
        "load_and_validate_spec",
        lambda: {"study_id": "test", **_spec()},
    )
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (record,))
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "ledger_closure",
        lambda _root: {"head": decision["record_hash"], "record_count": 1},
    )
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.WeeklyOperationError("current decision pair is not pushed")
        ),
    )
    status_calls: list[tuple[str, bool]] = []

    def decision_status(
        *,
        decision_date: str,
        data_dir: Path,
        now: datetime,
    ) -> dict[str, Any]:
        status_calls.append((decision_date, data_dir == data))
        return {
            "state": "DECISION_EVIDENCE_COMMIT_PENDING",
            "recovery_case": recovery_case,
            "formal_ledger_mutated": False,
        }

    monkeypatch.setattr(weekly.first_week, "status_snapshot", decision_status)
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("current-D weekly status performed actual-remote I/O"),
    )
    monkeypatch.setattr(
        stage3,
        "_exclusive_write",
        lambda *_a, **_k: pytest.fail("current-D weekly status wrote evidence"),
    )

    result = weekly.status_snapshot(
        decision_date_assertion=None,
        state_manifest_path=None,
        data_dir=data,
        now=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert result["state"] == "WAIT_DECISION_EVIDENCE"
    assert result["decision_status"]["recovery_case"] == recovery_case
    assert status_calls == [("2026-07-31", True)]


def test_status_aborts_current_decision_with_broken_successor_edge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decision = _decision(1, 1, "2026-07-31", "2026-08-03", "2026-08-10")
    record = stage3.LedgerRecord(tmp_path / "decision.json", decision, b"")
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    monkeypatch.setattr(
        stage3,
        "load_and_validate_spec",
        lambda: {"study_id": "test", **_spec()},
    )
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (record,))
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "ledger_closure",
        lambda _root: {"head": decision["record_hash"], "record_count": 1},
    )
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, **_k: (_ for _ in ()).throw(
            weekly.WeeklyOperationError("broken successor edge")
        ),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "status_snapshot",
        lambda **_k: {
            "state": "INVALID_DECISION_EVIDENCE_CHAIN",
            "formal_ledger_mutated": False,
        },
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("broken-edge status performed actual-remote I/O"),
    )

    result = weekly.status_snapshot(
        decision_date_assertion=None,
        state_manifest_path=None,
        data_dir=data,
        now=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert result["state"] == "ABORTED_INVALID"
    assert "INVALID_DECISION_EVIDENCE_CHAIN" in result["reason"]


@pytest.mark.parametrize(
    "decision_state",
    [
        "WAIT_DECISION_DATE",
        "WAIT_DAILY_RELEASE",
        "NEEDS_RAW_TARGET",
        "MISSED_SAFE_APPLY_WINDOW",
        "MISSED_DECISION_WINDOW",
    ],
)
def test_genesis_status_mirrors_bootstrap_decision_without_network_or_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decision_state: str,
) -> None:
    genesis = {
        "record_type": "genesis",
        "record_hash": "0" * 64,
        "payload": {},
    }
    root = tmp_path / "ledger"
    data = tmp_path / "raw"
    data.mkdir()
    monkeypatch.setattr(stage3, "load_and_validate_spec", lambda: {"study_id": "test", **_spec()})
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "f" * 64)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (genesis,))
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly.first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        weekly.first_week,
        "ledger_closure",
        lambda _root: {"head": "0" * 64, "record_count": 1},
    )
    monkeypatch.setattr(weekly, "_validate_label_order_for_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(weekly, "validate_label_sidecar_inventory", lambda *_a, **_k: None)
    scopes: list[bool] = []
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_a, verify_actual_remote, **_k: scopes.append(verify_actual_remote) or [],
    )
    status_calls: list[tuple[str | None, Path, datetime]] = []

    def decision_status(
        *,
        decision_date: str | None,
        data_dir: Path,
        now: datetime,
    ) -> dict[str, Any]:
        status_calls.append((decision_date, data_dir, now))
        return {
            "state": decision_state,
            "decision_date": "2026-07-31",
            "formal_ledger_mutated": False,
        }

    monkeypatch.setattr(
        weekly.first_week,
        "status_snapshot",
        decision_status,
    )
    monkeypatch.setattr(
        weekly,
        "validate_no_missed_decision_window",
        lambda *_a, **_k: pytest.fail("genesis status used post-D1 priority"),
    )
    monkeypatch.setattr(
        weekly,
        "derive_oldest_unlabeled_target",
        lambda *_a, **_k: pytest.fail("genesis status tried to derive a label"),
    )
    monkeypatch.setattr(
        weekly.first_week,
        "validate_actual_remote_branch",
        lambda: pytest.fail("genesis status performed actual-remote I/O"),
    )
    monkeypatch.setattr(
        stage3,
        "_exclusive_write",
        lambda *_a, **_k: pytest.fail("genesis status wrote formal evidence"),
    )
    clock = datetime(2026, 7, 31, 6, tzinfo=UTC)
    result = weekly.status_snapshot(
        decision_date_assertion="2026-07-31",
        state_manifest_path=None,
        data_dir=data,
        now=clock,
    )
    assert result["state"] == decision_state
    assert result["decision_date"] == "2026-07-31"
    assert result["decision_status"]["state"] == decision_state
    assert result["global_decision_priority"] == {
        "status": "BOOTSTRAP_DECISION_STATUS_MIRRORED",
        "decision_state": decision_state,
        "decision_date": "2026-07-31",
    }
    assert status_calls == [("2026-07-31", data, clock)]
    assert scopes == [False]
