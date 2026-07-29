from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage3 as stage3  # noqa: E402
import xs_chan_stage3_first_week as operator  # noqa: E402
import xs_chan_stage3_weekly as weekly  # noqa: E402


def _spec() -> dict[str, Any]:
    return {
        "accrual": {"window_weeks": 52},
        "historical_exclusion": {
            "first_prospective_decision_date": "2026-07-31",
        },
    }


def _record(
    sequence: int,
    record_type: str,
    record_hash: str,
    previous_hash: str,
    payload: Mapping[str, Any],
) -> stage3.LedgerRecord:
    data = {
        "sequence": sequence,
        "record_type": record_type,
        "record_hash": record_hash,
        "previous_hash": previous_hash,
        "ledger_id": "f" * 64,
        "payload": dict(payload),
    }
    return stage3.LedgerRecord(Path(f"{sequence:06d}_{record_hash}.json"), data, b"")


def _genesis() -> stage3.LedgerRecord:
    return _record(0, "genesis", "0" * 64, stage3.ZERO_HASH, {})


def _decision(
    sequence: int,
    week_index: int,
    previous_hash: str,
    *,
    manifest_sha: str,
) -> stage3.LedgerRecord:
    dates = {
        1: ("2026-07-31", "2026-08-03", "2026-08-10"),
        2: ("2026-08-07", "2026-08-10", "2026-08-17"),
        3: ("2026-08-14", "2026-08-17", "2026-08-24"),
    }
    decision_date, entry_date, exit_date = dates[week_index]
    return _record(
        sequence,
        "decision_freeze",
        str(week_index) * 64,
        previous_hash,
        {
            "classification": "PROSPECTIVE_COUNTED",
            "week_index": week_index,
            "decision_dt": decision_date,
            "entry_dt": entry_date,
            "exit_dt": exit_date,
            "reference_manifest_sha256": manifest_sha,
            "current_membership": [f"{week_index:06d}.SZ", "999999.SZ"],
        },
    )


def _label(
    sequence: int,
    decision: stage3.LedgerRecord,
    previous_hash: str,
) -> stage3.LedgerRecord:
    return _record(
        sequence,
        "label_completion",
        "a" * 64,
        previous_hash,
        {"decision_record_hash": decision.data["record_hash"]},
    )


def _install_context_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = pd.DataFrame(
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
        ]
    )
    daily_by_manifest = {
        "b" * 64: {
            "2026-07-31": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "999999.SZ"],
                    "trade_date": ["2026-07-31", "2026-07-31"],
                }
            )
        },
        "c" * 64: {
            "2026-08-07": pd.DataFrame(
                {
                    "ts_code": ["000002.SZ", "999999.SZ"],
                    "trade_date": ["2026-08-07", "2026-08-07"],
                }
            )
        },
    }

    def load_reference(
        _root: Path,
        path: Path,
    ) -> tuple[dict[str, Any], pd.DatetimeIndex, dict[str, pd.DataFrame]]:
        return (
            {},
            pd.DatetimeIndex(pd.bdate_range("2026-07-01", "2026-09-30")),
            daily_by_manifest.get(path.stem, {}),
        )

    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stage3, "load_reference_manifest", load_reference)
    monkeypatch.setattr(stage3, "build_forward_schedule", lambda *_args, **_kwargs: schedule.copy())
    monkeypatch.setattr(stage3, "load_initial_fc_membership", lambda: ["000000.SZ"])
    monkeypatch.setattr(operator, "REFERENCE_SYMBOL_ABSOLUTE_MINIMUM", 1)


def test_d2_context_is_unique_single_date_and_inherits_d1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)

    context = operator.derive_decision_operation_context(
        _spec(),
        tmp_path,
        decision_date="2026-08-07",
        records=(_genesis(), d1),
    )

    assert context.week_index == 2
    assert context.reference_dates == ("2026-08-07",)
    assert context.initial_decision_date == "2026-07-31"
    assert context.initial_membership == tuple(d1.data["payload"]["current_membership"])
    assert context.previous_decision_record_hash == d1.data["record_hash"]
    assert context.expected_head == d1.data["record_hash"]
    assert context.previous_reference_manifest_sha256 == "b" * 64
    assert context.previous_daily_basic_date == "2026-07-31"
    assert context.previous_daily_basic_symbols == ("000001.SZ", "999999.SZ")


def test_next_context_rejects_skip_and_d53(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    with pytest.raises(operator.FirstWeekOperationError, match="require D2=2026-08-07"):
        operator.derive_decision_operation_context(
            _spec(),
            tmp_path,
            decision_date="2026-08-14",
            records=(_genesis(), d1),
        )

    records = [_genesis()]
    previous_hash = records[0].data["record_hash"]
    for index in range(1, 53):
        record = _record(
            index,
            "decision_freeze",
            f"{index:064x}",
            previous_hash,
            {
                "classification": "PROSPECTIVE_COUNTED",
                "week_index": index,
            },
        )
        records.append(record)
        previous_hash = record.data["record_hash"]
    with pytest.raises(operator.FirstWeekOperationError, match="D53 is forbidden"):
        operator.derive_decision_operation_context(
            _spec(),
            tmp_path,
            records=records,
        )


def test_d3_requires_due_l1_and_binds_its_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    d2 = _decision(2, 2, d1.data["record_hash"], manifest_sha="c" * 64)
    without_label = (_genesis(), d1, d2)
    with pytest.raises(operator.FirstWeekOperationError, match="missing decision hashes"):
        operator.derive_decision_operation_context(
            _spec(),
            tmp_path,
            decision_date="2026-08-14",
            records=without_label,
        )

    l1 = _label(3, d1, d2.data["record_hash"])
    context = operator.derive_decision_operation_context(
        _spec(),
        tmp_path,
        decision_date="2026-08-14",
        records=(*without_label, l1),
    )
    assert context.expected_head == l1.data["record_hash"]
    assert context.due_label_bindings == (
        (d1.data["record_hash"], l1.data["record_hash"]),
    )

    pushed: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        weekly,
        "validate_label_authorized_anchor_pair",
        lambda _spec, _root, record, *, require_pushed: (
            pushed.append((record["record_hash"], require_pushed))
            or {"record_hash": record["record_hash"]}
        ),
    )
    evidence = operator.validate_due_label_evidence(
        _spec(),
        tmp_path,
        (*without_label, l1),
        context,
    )
    assert pushed == [(l1.data["record_hash"], True)]
    assert evidence[0]["authorized_anchor_pair"]["record_hash"] == l1.data["record_hash"]


def test_d3_missing_l1_fails_before_raw_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    d2 = _decision(2, 2, d1.data["record_hash"], manifest_sha="c" * 64)
    records = (_genesis(), d1, d2)
    monkeypatch.setattr(stage3, "load_and_validate_spec", _spec)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: tmp_path / "ledger")
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    sync_called = False

    def forbidden_sync(_args: Any) -> dict[str, Any]:
        nonlocal sync_called
        sync_called = True
        return {}

    monkeypatch.setattr(operator, "run_sync", forbidden_sync)
    with pytest.raises(operator.FirstWeekOperationError, match="missing decision hashes"):
        operator.prepare_decision_data(
            decision_date="2026-08-14",
            data_dir=tmp_path / "raw",
            snapshot_root=tmp_path / "snapshots",
            state_output_root=tmp_path / "states",
            workers=1,
        )
    assert sync_called is False


def test_same_date_label_still_runs_global_audit_before_raw_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    d1.data["payload"]["exit_dt"] = "2026-08-14"
    d2 = _decision(2, 2, d1.data["record_hash"], manifest_sha="c" * 64)
    l1 = _label(3, d1, d2.data["record_hash"])
    records = (_genesis(), d1, d2, l1)
    monkeypatch.setattr(stage3, "load_and_validate_spec", _spec)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: tmp_path / "ledger")
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    monkeypatch.setattr(operator, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(operator, "validate_prepare_time", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(operator, "validate_apply_window", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(operator, "validate_git_ready", lambda **_kwargs: {})
    monkeypatch.setattr(operator, "validate_existing_anchor_chain", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(operator, "validate_prior_decision_evidence", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(operator, "validate_due_label_evidence", lambda *_args, **_kwargs: [])
    audit_called = False

    def reject_global(*_args: Any, **_kwargs: Any) -> None:
        nonlocal audit_called
        audit_called = True
        raise operator.FirstWeekOperationError("global label order is unauthorized")

    monkeypatch.setattr(operator, "validate_global_weekly_label_evidence", reject_global)
    monkeypatch.setattr(
        operator,
        "run_sync",
        lambda _args: pytest.fail("raw sync ran before the global label audit"),
    )
    with pytest.raises(operator.FirstWeekOperationError, match="global label order"):
        operator.prepare_decision_data(
            decision_date="2026-08-14",
            data_dir=tmp_path / "raw",
            snapshot_root=tmp_path / "snapshots",
            state_output_root=tmp_path / "states",
            workers=1,
        )
    assert audit_called is True


@pytest.mark.parametrize(
    "current_symbols",
    [
        {f"{index:06d}.SZ" for index in range(94)},
        {f"{index:06d}.SZ" for index in range(106)},
    ],
)
def test_d2_daily_basic_rejects_large_change_from_d1(
    monkeypatch: pytest.MonkeyPatch,
    current_symbols: set[str],
) -> None:
    monkeypatch.setattr(operator, "REFERENCE_SYMBOL_ABSOLUTE_MINIMUM", 1)
    previous = {f"{index:06d}.SZ" for index in range(100)}
    daily = {
        "2026-08-07": pd.DataFrame(
            {
                "ts_code": sorted(current_symbols),
                "trade_date": ["2026-08-07"] * len(current_symbols),
            }
        )
    }
    with pytest.raises(operator.FirstWeekOperationError, match=r"retained only|changed"):
        operator.validate_reference_symbol_completeness(
            daily,
            raw_summary={},
            decision_date="2026-08-07",
            reference_dates=("2026-08-07",),
            previous_symbols=previous,
        )


def test_d2_reference_is_one_date_and_uses_inherited_membership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / f"{'d' * 64}.json"
    state_path = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    context = operator.DecisionOperationContext(
        week_index=2,
        decision_count=1,
        decision_date="2026-08-07",
        entry_date="2026-08-10",
        exit_date="2026-08-17",
        reference_dates=("2026-08-07",),
        initial_decision_date="2026-07-31",
        initial_membership=("000001.SZ",),
        previous_decision_record_hash="1" * 64,
        expected_head="1" * 64,
        ledger_id="f" * 64,
        bootstrap=False,
        previous_reference_manifest_sha256="b" * 64,
        previous_daily_basic_date="2026-07-31",
        previous_daily_basic_symbols=("000001.SZ",),
    )
    state_summary = {"manifest_sha256": "e" * 64, "projection_sha256": "f" * 64}
    raw_file = {"name": "000001.SZ.parquet", "sha256": "9" * 64, "max_dt": "20260807"}
    raw_summary = {
        "data_dir": str(raw_dir),
        "file_count": 1,
        "target_daily_symbols": operator._symbol_set_evidence({"000001.SZ"}),
    }
    manifest = {
        "local_inputs": {
            "state_manifest_path": str(state_path.resolve()),
            "state_manifest_sha256": state_summary["manifest_sha256"],
            "state_projection_sha256": state_summary["projection_sha256"],
            "raw_source_closure_sha256": "8" * 64,
            "raw_source_closure_object": "objects/raw_source_closure/closure.json",
        }
    }
    daily = {
        "2026-08-07": pd.DataFrame(
            {"ts_code": ["000001.SZ"], "trade_date": ["2026-08-07"]}
        )
    }
    sessions = pd.DatetimeIndex(pd.bdate_range("2026-07-01", "2026-09-30"))
    monkeypatch.setattr(
        stage3,
        "load_reference_manifest",
        lambda *_args, **_kwargs: (manifest, sessions, daily),
    )
    monkeypatch.setattr(
        stage3,
        "_load_content_object",
        lambda *_args, **_kwargs: {
            "coverage_max_dt": "2026-08-07",
            "file_count": 1,
            "files": [raw_file],
        },
    )
    monkeypatch.setattr(stage3, "verify_raw_source_closure_current", lambda _closure: None)
    monkeypatch.setattr(operator, "inspect_inventory", lambda _path: {"records": [raw_file]})
    monkeypatch.setattr(operator, "DEFAULT_MIN_DAILY_ROWS", 1)
    monkeypatch.setattr(operator, "REFERENCE_SYMBOL_ABSOLUTE_MINIMUM", 1)
    monkeypatch.setattr(
        stage3,
        "build_forward_schedule",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "week_index": 2,
                    "decision_dt": pd.Timestamp("2026-08-07"),
                    "entry_dt": pd.Timestamp("2026-08-10"),
                    "exit_dt": pd.Timestamp("2026-08-17"),
                }
            ]
        ),
    )
    monkeypatch.setattr(stage3, "build_decision_projections", lambda *_args, **_kwargs: {})
    captured: dict[str, Any] = {}

    def build_path(
        _spec: Mapping[str, Any],
        decision_dates: Any,
        _projections: Any,
        *,
        initial_symbols: Any,
        initial_decision_date: str,
    ) -> dict[str, Any]:
        captured.update(
            dates=tuple(decision_dates),
            initial=tuple(initial_symbols),
            initial_date=initial_decision_date,
        )
        return {
            "weeks": [
                {
                    "decision_dt": "2026-08-07",
                    "proposals": [],
                    "memberships": [{"symbol": "000001.SZ"}],
                }
            ],
            "final_membership": ["000001.SZ"],
        }

    monkeypatch.setattr(stage3, "build_fc_path_bundle", build_path)
    summary = operator.validate_reference_ready(
        _spec(),
        tmp_path,
        reference_path,
        state_path,
        "2026-08-07",
        state_summary=state_summary,
        raw_summary=raw_summary,
        context=context,
    )
    assert summary["bridge_dates"] == ["2026-08-07"]
    assert summary["week_index"] == 2
    assert captured == {
        "dates": ("2026-08-07",),
        "initial": ("000001.SZ",),
        "initial_date": "2026-07-31",
    }


def _report_context() -> operator.DecisionOperationContext:
    return operator.DecisionOperationContext(
        week_index=2,
        decision_count=1,
        decision_date="2026-08-07",
        entry_date="2026-08-10",
        exit_date="2026-08-17",
        reference_dates=("2026-08-07",),
        initial_decision_date="2026-07-31",
        initial_membership=("000001.SZ",),
        previous_decision_record_hash="1" * 64,
        expected_head="a" * 64,
        ledger_id="f" * 64,
        bootstrap=False,
        previous_reference_manifest_sha256="b" * 64,
        previous_daily_basic_date="2026-07-31",
        previous_daily_basic_symbols=("000001.SZ",),
    )


def _store_canonical_report(root: Path, payload: Mapping[str, Any]) -> Path:
    raw = stage3.canonical_json(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path = root / f"{digest}.json"
    path.write_bytes(raw)
    return path


def test_receipt_and_final_guard_reject_context_substitution(tmp_path: Path) -> None:
    context = _report_context()
    reference_path = tmp_path / f"{'d' * 64}.json"
    state_path = tmp_path / "state.json"
    state_path.write_bytes(b"state")
    base = {
        "schema": "xs_chan_stage3_first_week_preflight_v1",
        "formal_ledger_mutated": False,
        "efficacy_output": "FORBIDDEN",
        "operator_source_sha256": "e" * 64,
        "ledger_before": {"head": context.expected_head},
        "ledger_after": {"head": context.expected_head},
        "decision_context": context.evidence(),
        "reference": {
            "manifest_sha256": reference_path.stem,
            "bridge_dates": list(context.reference_dates),
            "decision_inputs_only": True,
            "prospective_week_count": context.decision_count,
            "bridge_path_sha256": "9" * 64,
            "decision_context": context.evidence(),
        },
        "state": {"manifest_sha256": operator.sha256_file(state_path)},
    }
    receipt = {
        **base,
        "mode": "APPLIED_DATA_ONLY_FULL_BRIDGE",
        "rehearsal": {
            "prospective_week_count": 1,
            "efficacy_summary_emitted": False,
        },
    }
    receipt_path = _store_canonical_report(tmp_path, receipt)
    operator.validate_preflight_receipt(
        receipt_path,
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        expected_head=context.expected_head,
        expected_operator_source_sha256="e" * 64,
        context=context,
    )

    tampered = dict(receipt)
    tampered["decision_context"] = {**context.evidence(), "week_index": 3}
    tampered_path = _store_canonical_report(tmp_path, tampered)
    with pytest.raises(operator.FirstWeekOperationError, match="exact derived decision context"):
        operator.validate_preflight_receipt(
            tampered_path,
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            expected_head=context.expected_head,
            expected_operator_source_sha256="e" * 64,
            context=context,
        )

    final = {**base, "mode": "READ_ONLY_FULL_BRIDGE"}
    operator._validate_final_preflight_report(
        final,
        expected_head=context.expected_head,
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        expected_operator_source_sha256="e" * 64,
        context=context,
    )
    final["reference"] = {
        **final["reference"],
        "decision_context": {**context.evidence(), "entry_date": "2026-08-11"},
    }
    with pytest.raises(operator.FirstWeekOperationError, match="exact derived decision context"):
        operator._validate_final_preflight_report(
            final,
            expected_head=context.expected_head,
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            expected_operator_source_sha256="e" * 64,
            context=context,
        )


def test_prior_decision_evidence_requires_every_pushed_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    d2 = _decision(2, 2, d1.data["record_hash"], manifest_sha="c" * 64)
    seen: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(
        operator,
        "validate_authorized_anchor_pair",
        lambda _spec, _root, record, *, require_pushed, verify_remote: (
            seen.append((record["record_hash"], require_pushed, verify_remote))
            or {"record_hash": record["record_hash"]}
        ),
    )
    operator.validate_prior_decision_evidence(
        _spec(),
        tmp_path,
        (_genesis(), d1, d2),
    )
    assert seen == [
        (d1.data["record_hash"], True, True),
        (d2.data["record_hash"], True, True),
    ]


def test_recovery_uses_payload_entry_and_requires_current_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    genesis = _genesis()
    d1 = _decision(1, 1, genesis.data["record_hash"], manifest_sha="b" * 64)
    d2 = _decision(2, 2, d1.data["record_hash"], manifest_sha="c" * 64)
    records = (genesis, d1, d2)
    reference_path = tmp_path / "ledger" / "objects" / "reference_manifest" / f"{'c' * 64}.json"
    receipt_path = tmp_path / "ledger" / "objects" / operator.PREFLIGHT_CATEGORY / f"{'d' * 64}.json"
    state_path = tmp_path / "state.json"
    authorization_path = tmp_path / "authorization.json"
    anchor_path = tmp_path / "anchor.json"
    sidecar_path = tmp_path / "sidecar.json"
    for path in (reference_path, receipt_path, state_path, authorization_path, anchor_path, sidecar_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    @contextmanager
    def unlocked(*_args: Any, **_kwargs: Any) -> Iterator[None]:
        yield

    monkeypatch.setattr(operator, "operator_lock", unlocked)
    monkeypatch.setattr(operator, "cache_lock", unlocked)
    monkeypatch.setattr(stage3, "_exclusive_lock", unlocked)
    monkeypatch.setattr(operator, "atomic_stage3_writes", unlocked)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(operator, "_validate_recovery_anchor_inventory", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(operator, "_validate_recovery_authorization_inventory", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(operator, "validate_prior_decision_evidence", lambda *_args, **_kwargs: [])
    pair_chain_calls: list[tuple[int, bool, bool, set[str] | None]] = []
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda _spec, _root, prior_records, *, require_pushed, verify_actual_remote,
        ignored_decision_sidecar_names: (
            pair_chain_calls.append(
                (
                    len(prior_records),
                    require_pushed,
                    verify_actual_remote,
                    ignored_decision_sidecar_names,
                )
            )
            or []
        ),
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_args, **_kwargs: {
            "predecessor_pair_commit": "8" * 40,
            "successor_authorized_git_head": "9" * 40,
            "actual_remote_verified": False,
        },
    )
    authorization = {
        "state_manifest_path": str(state_path.resolve()),
        "reference_manifest_object": str(reference_path.relative_to(tmp_path / "ledger")),
        "preflight_report_object": str(receipt_path.relative_to(tmp_path / "ledger")),
        "raw_audit_sha256": "1" * 64,
        "raw_parquet_inventory_sha256": "2" * 64,
        "raw_source_closure_sha256": "3" * 64,
        "state_manifest_sha256": "4" * 64,
        "state_projection_sha256": "5" * 64,
        "reference_manifest_sha256": reference_path.stem,
        "decision_path_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        operator,
        "load_append_authorization",
        lambda *_args, **_kwargs: (authorization, "7" * 64, authorization_path),
    )
    pair_calls = 0

    def pair(*_args: Any, require_pushed: bool, **_kwargs: Any) -> dict[str, Any]:
        nonlocal pair_calls
        pair_calls += 1
        if require_pushed:
            raise operator.FirstWeekOperationError("not pushed yet")
        return {}

    monkeypatch.setattr(operator, "validate_authorized_anchor_pair", pair)
    deadline_dates: list[str] = []
    monkeypatch.setattr(
        operator,
        "validate_recovery_deadline",
        lambda entry_date, **_kwargs: deadline_dates.append(entry_date) or {"entry_date": entry_date},
    )
    monkeypatch.setattr(operator, "validate_recovery_git_ready", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        operator,
        "validate_raw_ready",
        lambda *_args, **_kwargs: {"audit_sha256": "1" * 64, "parquet_inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        operator,
        "validate_state_ready",
        lambda *_args, **_kwargs: {
            "manifest_sha256": "4" * 64,
            "projection_sha256": "5" * 64,
            "raw_parquet_inventory_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        operator,
        "validate_reference_ready",
        lambda *_args, **_kwargs: {
            "manifest_sha256": reference_path.stem,
            "raw_source_closure_sha256": "3" * 64,
            "bridge_path_sha256": "6" * 64,
        },
    )
    d2.data["payload"]["decision_path_sha256"] = "6" * 64
    monkeypatch.setattr(
        operator,
        "export_authorization_sidecar",
        lambda *_args, **_kwargs: ({"authorized": True}, sidecar_path),
    )
    monkeypatch.setattr(
        stage3,
        "export_ledger_head_anchor",
        lambda *_args, **_kwargs: ({"anchored": True}, anchor_path),
    )
    monkeypatch.setattr(operator, "validate_existing_anchor_chain", lambda *_args, **_kwargs: [])

    result = operator.recover_first_decision_anchor(
        spec=_spec(),
        root=tmp_path / "ledger",
        record_hash=d2.data["record_hash"],
        decision_date="2026-08-07",
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=receipt_path,
        expected_head=d1.data["record_hash"],
        data_dir=tmp_path / "raw",
    )
    assert result["status"] == "AUTHORIZED_DECISION_ANCHOR_RECOVERED"
    assert deadline_dates == ["2026-08-10"] * 3
    assert pair_calls == 2
    assert pair_chain_calls == [(2, True, True, None)]

    label = _label(3, d1, d2.data["record_hash"])
    monkeypatch.setattr(stage3, "scan_records", lambda _root: (*records, label))
    with pytest.raises(operator.FirstWeekOperationError, match="ledger changed"):
        operator.recover_first_decision_anchor(
            spec=_spec(),
            root=tmp_path / "ledger",
            record_hash=d2.data["record_hash"],
            decision_date="2026-08-07",
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            preflight_report_path=receipt_path,
            expected_head=d1.data["record_hash"],
            data_dir=tmp_path / "raw",
        )


def test_fresh_calendar_may_extend_exit_but_never_rewrite_prior_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    records = (_genesis(), d1)
    old_sessions = pd.DatetimeIndex(pd.bdate_range("2026-07-01", "2026-09-30"))
    fresh_sessions = old_sessions.drop(pd.Timestamp("2026-08-11"))
    rewritten_sessions = fresh_sessions.drop(pd.Timestamp("2026-08-07"))
    daily = {
        "2026-07-31": pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "999999.SZ"],
                "trade_date": ["2026-07-31", "2026-07-31"],
            }
        )
    }

    def load_reference(
        _root: Path,
        path: Path,
    ) -> tuple[dict[str, Any], pd.DatetimeIndex, dict[str, pd.DataFrame]]:
        sessions = {
            "b" * 64: old_sessions,
            "d" * 64: fresh_sessions,
            "e" * 64: rewritten_sessions,
        }[path.stem]
        return {}, sessions, daily if path.stem == "b" * 64 else {}

    def schedule(
        _spec: Mapping[str, Any],
        sessions: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        fresh = pd.Timestamp("2026-08-11") not in sessions
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
                    "exit_dt": pd.Timestamp("2026-08-18" if fresh else "2026-08-17"),
                },
            ]
        )

    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stage3, "load_reference_manifest", load_reference)
    monkeypatch.setattr(stage3, "build_forward_schedule", schedule)
    monkeypatch.setattr(operator, "REFERENCE_SYMBOL_ABSOLUTE_MINIMUM", 1)
    provisional = operator.derive_decision_operation_context(
        _spec(),
        tmp_path,
        records=records,
    )
    final = operator.derive_decision_operation_context(
        _spec(),
        tmp_path,
        records=records,
        reference_manifest_path=tmp_path / f"{'d' * 64}.json",
    )
    operator.validate_provisional_final_context(provisional, final)
    assert provisional.exit_date == "2026-08-17"
    assert final.exit_date == "2026-08-18"
    assert final.decision_date == provisional.decision_date
    assert final.entry_date == provisional.entry_date

    with pytest.raises(operator.FirstWeekOperationError, match="rewrites the frozen session prefix"):
        operator.derive_decision_operation_context(
            _spec(),
            tmp_path,
            records=records,
            reference_manifest_path=tmp_path / f"{'e' * 64}.json",
        )


def test_v1_context_replay_ignores_future_evidence_method_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _report_context()
    stored = operator.serialize_decision_operation_context_v1(context)
    monkeypatch.setattr(
        operator.DecisionOperationContext,
        "evidence",
        lambda self: {
            **operator.serialize_decision_operation_context_v1(self),
            "future_v2_only": True,
        },
    )
    operator._require_context_evidence(stored, context, label="historical authorization")

    tampered = {**stored, "exit_date": "2026-08-19"}
    with pytest.raises(operator.FirstWeekOperationError, match="exact derived decision context"):
        operator._require_context_evidence(
            tampered,
            context,
            label="historical authorization",
        )


def _install_status_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: tuple[stage3.LedgerRecord, ...],
) -> None:
    monkeypatch.setattr(operator, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(stage3, "load_and_validate_spec", _spec)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: tmp_path / "ledger")
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    monkeypatch.setattr(operator, "validate_authorization_sidecar_inventory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(stage3, "status_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(stage3, "study_identity", lambda _spec: "study")
    monkeypatch.setattr(
        operator,
        "inspect_inventory",
        lambda _path: {
            "file_count": 2,
            "max_dt": "20260731",
            "content_inventory_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        operator,
        "validate_git_ready",
        lambda **_kwargs: {"worktree_clean": True},
    )


def test_status_auto_and_explicit_d2_use_dynamic_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    records = (_genesis(), d1)
    _install_status_dependencies(monkeypatch, tmp_path, records)

    automatic = operator.status_snapshot(
        decision_date=None,
        data_dir=tmp_path / "raw",
        now=pd.Timestamp("2026-08-01T00:00:00Z").to_pydatetime(),
    )
    explicit = operator.status_snapshot(
        decision_date="2026-08-07",
        data_dir=tmp_path / "raw",
        now=pd.Timestamp("2026-08-01T00:00:00Z").to_pydatetime(),
    )
    for snapshot in (automatic, explicit):
        assert snapshot["week_index"] == 2
        assert snapshot["decision_date"] == "2026-08-07"
        assert snapshot["entry_date"] == "2026-08-10"
        assert snapshot["exit_date"] == "2026-08-17"
        assert snapshot["formal_ledger_mutated"] is False


@pytest.mark.parametrize("explicit", [False, True])
def test_preflight_cli_auto_or_explicit_d2_binds_context_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    explicit: bool,
) -> None:
    _install_context_sources(monkeypatch)
    d1 = _decision(1, 1, "0" * 64, manifest_sha="b" * 64)
    records = (_genesis(), d1)
    monkeypatch.setattr(stage3, "load_and_validate_spec", _spec)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: tmp_path / "ledger")
    monkeypatch.setattr(stage3, "scan_records", lambda _root: records)
    captured: list[tuple[str, operator.DecisionOperationContext | None]] = []

    def preflight(
        _spec: Mapping[str, Any],
        _root: Path,
        _reference: Path,
        _state: Path,
        decision_date: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured.append((decision_date, kwargs.get("context")))
        return {"formal_ledger_mutated": False}

    monkeypatch.setattr(operator, "build_preflight_report", preflight)
    argv = []
    if explicit:
        argv.extend(["--decision-date", "2026-08-07"])
    argv.extend(
        [
            "preflight-decision",
            "--reference-manifest",
            str(tmp_path / f"{'d' * 64}.json"),
            "--state-manifest",
            str(tmp_path / "state.json"),
        ]
    )
    assert operator.main(argv) == 0
    assert captured[0][0] == "2026-08-07"
    assert captured[0][1] is not None
    assert captured[0][1].week_index == 2
    assert '"formal_ledger_mutated": false' in capsys.readouterr().out.lower()


def test_decision_pair_strict_remote_rejects_local_only_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "scripts" / "anchors" / "anchor.json"
    sidecar_path = tmp_path / "scripts" / "authorizations" / "sidecar.json"
    pair_commit = "c" * 40
    authorization_parent = "a" * 40
    monkeypatch.setattr(stage3, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(operator, "validate_authorization_sidecar_inventory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        operator,
        "validate_authorization_sidecar",
        lambda *_args, **_kwargs: {
            "path": str(sidecar_path),
            "commit": pair_commit,
        },
    )
    monkeypatch.setattr(
        operator,
        "validate_anchor_for_record",
        lambda *_args, **_kwargs: {
            "path": str(anchor_path),
            "commit": pair_commit,
        },
    )
    monkeypatch.setattr(
        operator,
        "load_append_authorization",
        lambda *_args, **_kwargs: (
            {"git": {"head": authorization_parent}},
            "1" * 64,
            tmp_path / "authorization.json",
        ),
    )
    expected_paths = "\n".join(
        sorted(
            {
                anchor_path.relative_to(tmp_path).as_posix(),
                sidecar_path.relative_to(tmp_path).as_posix(),
            }
        )
    )

    def git_output(_root: Path, *args: str) -> str:
        if args[:3] == ("rev-list", "--parents", "-n"):
            return f"{pair_commit} {authorization_parent}"
        if args and args[0] == "diff-tree":
            return expected_paths
        if args[:2] == ("merge-base", "--is-ancestor"):
            raise operator.FirstWeekOperationError("not an ancestor")
        raise AssertionError(args)

    monkeypatch.setattr(operator, "_git_output", git_output)
    monkeypatch.setattr(
        operator,
        "validate_actual_remote_branch",
        lambda **_kwargs: {"remote_head": authorization_parent},
    )
    monkeypatch.setattr(stage3, "scan_records", lambda _root: ())
    with pytest.raises(operator.FirstWeekOperationError, match="actual remote head"):
        operator.validate_authorized_anchor_pair(
            _spec(),
            tmp_path / "ledger",
            {
                "record_hash": "1" * 64,
                "payload": {"entry_dt": "2026-08-10"},
            },
            require_pushed=True,
            verify_remote=True,
        )


def test_status_after_d52_reports_label_work_instead_of_d53(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [_genesis()]
    previous_hash = records[0].data["record_hash"]
    first = pd.Timestamp("2026-07-31")
    for index in range(1, 53):
        decision_date = first + pd.Timedelta(weeks=index - 1)
        record = _record(
            index,
            "decision_freeze",
            f"{index:064x}",
            previous_hash,
            {
                "classification": "PROSPECTIVE_COUNTED",
                "week_index": index,
                "decision_dt": decision_date.date().isoformat(),
                "entry_dt": (decision_date + pd.Timedelta(days=3)).date().isoformat(),
                "exit_dt": (decision_date + pd.Timedelta(days=10)).date().isoformat(),
                "reference_manifest_sha256": "b" * 64,
                "current_membership": [],
            },
        )
        records.append(record)
        previous_hash = record.data["record_hash"]
    chain = tuple(records)
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    _install_status_dependencies(monkeypatch, tmp_path, chain)
    snapshot = operator.status_snapshot(
        decision_date=None,
        data_dir=tmp_path / "raw",
        now=pd.Timestamp("2027-08-01T00:00:00Z").to_pydatetime(),
    )
    assert snapshot["state"] == "DECISION_WINDOW_COMPLETE"
    assert snapshot["decision_window_complete"] is True
    assert snapshot["outstanding_label_count"] == 52
    assert snapshot["next_weekly_action"] == "COMPLETE_OUTSTANDING_LABELS"
