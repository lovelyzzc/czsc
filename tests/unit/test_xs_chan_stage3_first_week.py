from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _sync_daily_data as sync  # noqa: E402
import xs_chan_exploration_stage3 as stage3  # noqa: E402
import xs_chan_stage3_first_week as first_week  # noqa: E402

FROZEN_COLLECTOR_SHA256 = "4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea"
FROZEN_SPEC_SHA256 = "3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa"
FROZEN_STUDY_IDENTITY = "f0afe61d932fbdf68b5b5ee242b68c7eed75bc932e4f6cf785bd7c9af51fb607"
TEST_DECISION_PATH_SHA256 = "9" * 64


def _spec() -> dict[str, Any]:
    return copy.deepcopy(stage3.load_and_validate_spec())


def _append_seed_record(root: Path) -> stage3.LedgerRecord:
    return stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "temporary first-week operator test ledger"},
        "2026-07-28T06:00:00Z",
    )


def _write_raw_symbol(
    root: Path,
    symbol: str,
    trade_date: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol}.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "pre_close": 10.0,
                "change": 0.0,
                "pct_chg": 0.0,
                "vol": 100.0,
                "amount": 1_000.0,
            }
        ],
        columns=sync.DAILY_COLUMNS,
    ).to_parquet(path, index=False)
    return path


def _install_guard_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verifier: Callable[[Mapping[str, Any]], None],
    anchor_path: Path,
) -> None:
    monkeypatch.setattr(
        stage3,
        "read_json",
        lambda _path: {
            "local_inputs": {
                "raw_source_closure_sha256": "a" * 64,
                "raw_source_closure_object": "objects/raw_source_closure/a.json",
            }
        },
    )
    monkeypatch.setattr(
        stage3,
        "_load_content_object",
        lambda *_args, **_kwargs: {"closure": "frozen"},
    )
    monkeypatch.setattr(stage3, "verify_raw_source_closure_current", verifier)
    monkeypatch.setattr(
        first_week,
        "validate_anchor_for_record",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stage3,
        "export_ledger_head_anchor",
        lambda *_args, **_kwargs: ({"status": "test-anchor"}, anchor_path),
    )


def _install_fake_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_append_decision_freeze(
        _spec: Mapping[str, Any],
        root: Path,
        reference_manifest_path: Path,
        decision_date: str,
    ) -> stage3.LedgerRecord:
        collector_time = stage3.utc_now()
        return stage3.append_record(
            root,
            "decision_freeze",
            f"decision:{decision_date}",
            {
                "decision_dt": decision_date,
                "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
                "decision_path_sha256": TEST_DECISION_PATH_SHA256,
                "reference_manifest_sha256": reference_manifest_path.stem,
            },
            collector_time,
        )

    monkeypatch.setattr(
        stage3,
        "append_decision_freeze",
        fake_append_decision_freeze,
    )


def _reference_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    daily_dates: Sequence[str],
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    spec = _spec()
    reference_path = tmp_path / f"{'b' * 64}.json"
    state_path = tmp_path / "state-source-manifest.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    state_summary = {
        "manifest_sha256": "c" * 64,
        "projection_sha256": "d" * 64,
    }
    raw_summary = {
        "data_dir": str(raw_dir),
        "file_count": 1,
    }
    local_inputs = {
        "state_manifest_path": str(state_path.resolve()),
        "state_manifest_sha256": state_summary["manifest_sha256"],
        "state_projection_sha256": state_summary["projection_sha256"],
        "raw_source_closure_sha256": "e" * 64,
        "raw_source_closure_object": "objects/raw_source_closure/e.json",
    }
    manifest = {"local_inputs": local_inputs}
    daily = {
        date_text: pd.DataFrame(
            {
                "ts_code": [f"{index:06d}.SZ"],
                "trade_date": [date_text],
            }
        )
        for index, date_text in enumerate(daily_dates, 1)
    }
    sessions = pd.DatetimeIndex(pd.bdate_range("2026-06-08", first_week.FIRST_WEEK_EXIT_DATE))
    raw_file = {
        "name": "000001.SZ.parquet",
        "sha256": "f" * 64,
    }
    closure = {
        "coverage_max_dt": first_week.FIRST_WEEK_REFERENCE_DATES[-1],
        "file_count": 1,
        "files": [raw_file],
    }

    monkeypatch.setattr(
        stage3,
        "load_reference_manifest",
        lambda *_args, **_kwargs: (manifest, sessions, daily),
    )
    monkeypatch.setattr(
        stage3,
        "_load_content_object",
        lambda *_args, **_kwargs: closure,
    )
    monkeypatch.setattr(
        stage3,
        "verify_raw_source_closure_current",
        lambda _closure: None,
    )
    monkeypatch.setattr(
        first_week,
        "inspect_inventory",
        lambda _data_dir: {"records": [raw_file]},
    )
    monkeypatch.setattr(
        stage3,
        "_official_weekly_decisions",
        lambda *_args, **_kwargs: [pd.Timestamp(value) for value in first_week.FIRST_WEEK_REFERENCE_DATES],
    )
    monkeypatch.setattr(first_week, "DEFAULT_MIN_DAILY_ROWS", 1)
    monkeypatch.setattr(
        stage3,
        "build_forward_schedule",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "week_index": 1,
                    "decision_dt": pd.Timestamp("2026-07-31"),
                    "entry_dt": pd.Timestamp(first_week.FIRST_WEEK_ENTRY_DATE),
                    "exit_dt": pd.Timestamp(first_week.FIRST_WEEK_EXIT_DATE),
                }
            ]
        ),
    )
    monkeypatch.setattr(
        stage3,
        "build_decision_projections",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stage3,
        "load_initial_fc_membership",
        lambda: [],
    )
    monkeypatch.setattr(
        stage3,
        "build_fc_path_bundle",
        lambda *_args, **_kwargs: {
            "weeks": [
                {
                    "decision_dt": date_text,
                    "proposals": [],
                    "memberships": [],
                }
                for date_text in first_week.FIRST_WEEK_REFERENCE_DATES
            ],
            "final_membership": [],
        },
    )
    return spec, reference_path, state_path, state_summary, raw_summary


def test_prepare_time_opens_exactly_at_daily_release() -> None:
    release = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="preparation is forbidden",
    ):
        first_week.validate_prepare_time(
            "2026-07-31",
            now=release - timedelta(microseconds=1),
        )

    gate = first_week.validate_prepare_time("2026-07-31", now=release)
    assert gate["checked_at_utc"] == release
    assert gate["daily_release_not_before_utc"] == release


def test_operator_rejects_a_raw_directory_the_frozen_collector_will_not_read(
    tmp_path: Path,
) -> None:
    assert first_week.validate_frozen_data_dir(stage3.RAW_DIR) == stage3.RAW_DIR.resolve()
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="frozen raw directory",
    ):
        first_week.validate_frozen_data_dir(tmp_path / "other-raw")


def test_strict_json_and_compact_integer_trade_dates(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"same": 1, "same": 2}', encoding="utf-8")
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="cannot read JSON evidence",
    ):
        first_week._read_json(duplicate)
    assert first_week._trade_date_text(20260731) == "2026-07-31"


def test_raw_validator_accepts_one_exact_v2_audit_and_rejects_object_drift(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "active"
    snapshot_root = tmp_path / "snapshots"
    _write_raw_symbol(data_dir, "000001.SZ", "20260731")
    inventory = sync.inspect_inventory(data_dir)
    sync.atomic_write_json(
        data_dir / "manifest.json",
        {
            "schema": "a_stock_daily_qfq_active_snapshot_v1",
            "safe_end_date": "20260731",
            "content_inventory_sha256": inventory["content_inventory_sha256"],
            "file_count": inventory["file_count"],
        },
    )
    snapshot, snapshot_inventory = sync.archive_raw_snapshot(
        data_dir,
        inventory,
        snapshot_root,
    )
    api_object = tmp_path / "objects" / "calendar.json"
    api_object.parent.mkdir()
    api_object.write_text('{"calendar": "frozen"}\n', encoding="utf-8")
    execution_binding = sync.build_execution_binding(
        argparse.Namespace(
            data_dir=data_dir,
            snapshot_root=snapshot_root,
            end_date="20260731",
            min_daily_rows=sync.DEFAULT_MIN_DAILY_ROWS,
            new_symbol_sleep_seconds=0.0,
            apply=True,
        ),
        data_dir.resolve(),
        snapshot_root.resolve(),
        "20260731",
        require_clean_and_pushed=False,
    )
    audit = {
        "schema": "a_stock_daily_qfq_sync_audit_v2",
        "data_dir": str(data_dir.resolve()),
        "safe_end_date": "20260731",
        "after_inventory": {key: value for key, value in inventory.items() if key != "records"},
        "expected_active_inventory_sha256": inventory["content_inventory_sha256"],
        "execution_binding": execution_binding,
        "trade_dates": ["20260731"],
        "api_response_objects": {
            "official_calendar": {
                "path": str(api_object),
                "sha256": sync.sha256_file(api_object),
            },
            "daily": {
                "20260731": {
                    "path": str(api_object),
                    "sha256": sync.sha256_file(api_object),
                }
            },
            "adj_factor": {
                "20260731": {
                    "path": str(api_object),
                    "sha256": sync.sha256_file(api_object),
                }
            },
        },
        "full_qfq_refresh_symbol_count": 0,
        "full_qfq_refresh_objects": [],
        "new_raw_snapshot": {
            "path": str(snapshot),
            "closure_sha256": snapshot_inventory["snapshot_payload_sha256"],
            "parquet_content_inventory_sha256": inventory["content_inventory_sha256"],
        },
    }
    audit_raw = (
        json.dumps(
            audit,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    audit_dir = data_dir.parent / sync.AUDIT_DIR_NAME
    audit_dir.mkdir(exist_ok=True)
    audit_path = audit_dir / f"{sync.sha256_bytes(audit_raw)}.json"
    audit_path.write_bytes(audit_raw)

    summary = first_week.validate_raw_ready(
        "2026-07-31",
        data_dir=data_dir,
    )
    assert summary["audit_sha256"] == audit_path.stem
    assert summary["parquet_inventory_sha256"] == inventory["content_inventory_sha256"]
    assert summary["frozen_object_count"] == 3

    snapshot_manifest_path = snapshot / sync.SNAPSHOT_MANIFEST_FILE_NAME
    snapshot_manifest_bytes = snapshot_manifest_path.read_bytes()
    snapshot_manifest = json.loads(snapshot_manifest_bytes.decode("utf-8"))
    snapshot_manifest["files"] = []
    sync.atomic_write_json(snapshot_manifest_path, snapshot_manifest)
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="snapshot identity differs",
    ):
        first_week.validate_raw_ready(
            "2026-07-31",
            data_dir=data_dir,
        )
    snapshot_manifest_path.write_bytes(snapshot_manifest_bytes)

    api_object.write_text('{"calendar": "drifted"}\n', encoding="utf-8")
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="frozen raw input object differs",
    ):
        first_week.validate_raw_ready(
            "2026-07-31",
            data_dir=data_dir,
        )


@pytest.mark.parametrize(
    ("now", "allowed"),
    [
        (datetime(2026, 7, 31, 6, 59, 59, 999999, tzinfo=UTC), False),
        (datetime(2026, 7, 31, 7, 0, tzinfo=UTC), True),
        (datetime(2026, 8, 2, 23, 29, 59, 999999, tzinfo=UTC), True),
        (datetime(2026, 8, 2, 23, 30, tzinfo=UTC), False),
    ],
)
def test_apply_window_is_lower_inclusive_and_reserved_upper_exclusive(
    now: datetime,
    allowed: bool,
) -> None:
    if not allowed:
        with pytest.raises(
            first_week.FirstWeekOperationError,
            match="decision apply must start",
        ):
            first_week.validate_apply_window(
                "2026-07-31",
                first_week.FIRST_WEEK_ENTRY_DATE,
                now=now,
            )
        return

    gate = first_week.validate_apply_window(
        "2026-07-31",
        first_week.FIRST_WEEK_ENTRY_DATE,
        now=now,
    )
    assert gate["checked_at_utc"] == now
    assert gate["latest_safe_start_utc"] == datetime(
        2026,
        8,
        2,
        23,
        30,
        tzinfo=UTC,
    )


def test_read_only_preflight_keeps_ledger_byte_equal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    _append_seed_record(root)
    before = first_week.ledger_closure(root)
    monkeypatch.setattr(
        first_week,
        "validate_frozen_data_dir",
        lambda path: path,
    )
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda **_kwargs: {"worktree_clean": True},
    )
    monkeypatch.setattr(
        first_week,
        "validate_existing_anchor_chain",
        lambda *_args, **_kwargs: [{"record_hash": before["head"]}],
    )
    monkeypatch.setattr(
        first_week,
        "validate_raw_ready",
        lambda *_args, **_kwargs: {"parquet_inventory_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        first_week,
        "validate_state_ready",
        lambda *_args, **_kwargs: {"raw_parquet_inventory_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        first_week,
        "validate_reference_ready",
        lambda *_args, **_kwargs: {
            "bridge_dates": list(first_week.FIRST_WEEK_REFERENCE_DATES),
            "prospective_week_count": 0,
        },
    )

    report = first_week.build_preflight_report(
        _spec(),
        root,
        tmp_path / "reference.json",
        tmp_path / "state.json",
        "2026-07-31",
        data_dir=tmp_path / "raw",
        verify_remote=False,
        generated_at=datetime(2026, 7, 31, 10, 0, tzinfo=UTC),
    )

    assert report["mode"] == "READ_ONLY_FULL_BRIDGE"
    assert report["formal_ledger_mutated"] is False
    assert report["ledger_before"] == before
    assert report["ledger_after"] == before
    assert first_week.ledger_closure(root) == before


def test_anchor_chain_rejects_an_orphan_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    anchor_dir = tmp_path / "xs_chan_exploration_stage3_ledger_anchors"
    anchor_dir.mkdir()
    (anchor_dir / f"000001_{'a' * 64}.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(stage3, "scan_records", lambda _root: ())

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="exact ledger chain",
    ):
        first_week.validate_existing_anchor_chain(
            spec,
            tmp_path / "ledger",
            require_pushed=True,
        )


def test_anchor_commit_at_deadline_is_rejected_as_not_strictly_before(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    anchor_dir = tmp_path / "xs_chan_exploration_stage3_ledger_anchors"
    anchor_dir.mkdir()
    record = {
        "sequence": 0,
        "record_type": "genesis",
        "logical_event_key": "genesis:test",
        "recorded_at_utc": "2026-07-28T06:00:00.000000Z",
        "record_hash": "1" * 64,
        "previous_hash": stage3.ZERO_HASH,
        "payload": {"protocol_anchor_git_commit": "2" * 40},
    }
    expected = {
        "schema": "xs_chan_stage3_tracked_ledger_head_anchor_v1",
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "ledger_schema": stage3.LEDGER_SCHEMA,
        "sequence": record["sequence"],
        "record_type": record["record_type"],
        "logical_event_key": record["logical_event_key"],
        "recorded_at_utc": record["recorded_at_utc"],
        "record_hash": record["record_hash"],
        "previous_hash": record["previous_hash"],
        "spec_physical_sha256": stage3.sha256_file(stage3.SPEC_PATH),
        "source_sha256": stage3.sha256_file(stage3.SOURCE_PATH),
        "source_replay": {
            "status": "PROTOCOL_SOURCE_REPLAYED",
            "protocol_anchor_git_commit": "2" * 40,
        },
        "local_record_path": f"000000_{'1' * 64}.json",
        "external_timestamp_or_signature": False,
        "required_follow_up": ("commit_and_push_this_unique_anchor_without_rewriting_prior_anchors"),
    }
    anchor_path = anchor_dir / f"000000_{'1' * 64}.json"
    anchor_path.write_text(
        json.dumps(expected, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(
        stage3,
        "_validate_tracked_file_pushed",
        lambda *_args, **_kwargs: "3" * 40,
    )
    deadline = first_week._decision_close("2026-07-31")
    monkeypatch.setattr(
        first_week,
        "_git_output",
        lambda *_args: deadline.isoformat(),
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="strictly before",
    ):
        first_week.validate_anchor_for_record(
            spec,
            tmp_path / "ledger",
            record,
            require_pushed=True,
        )


def test_prepare_cli_without_apply_flag_never_calls_mutator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        first_week,
        "status_snapshot",
        lambda **_kwargs: {"formal_ledger_mutated": False},
    )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("prepare-decision dry run called a mutation path")

    monkeypatch.setattr(first_week, "prepare_decision_data", fail_if_called)
    monkeypatch.setattr(first_week, "run_sync", fail_if_called)

    result = first_week.main(
        [
            "--data-dir",
            str(tmp_path / "raw"),
            "prepare-decision",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_NO_DATA_MUTATION"
    assert payload["required_flag"] == "--apply-data"


def test_freeze_cli_without_apply_flag_runs_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        stage3,
        "resolve_ledger_root",
        lambda _spec: tmp_path / "ledger",
    )
    monkeypatch.setattr(
        first_week,
        "build_preflight_report",
        lambda *_args, **_kwargs: {"formal_ledger_mutated": False},
    )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("freeze-decision dry run called the formal append path")

    monkeypatch.setattr(first_week, "freeze_first_decision", fail_if_called)
    result = first_week.main(
        [
            "--data-dir",
            str(tmp_path / "raw"),
            "freeze-decision",
            "--reference-manifest",
            str(tmp_path / "reference.json"),
            "--state-manifest",
            str(tmp_path / "state.json"),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_NO_LEDGER_MUTATION"
    assert payload["required_flag"] == "--apply-ledger"


@pytest.mark.parametrize(
    "daily_dates",
    [
        first_week.FIRST_WEEK_REFERENCE_DATES[:-1],
        (*first_week.FIRST_WEEK_REFERENCE_DATES, "2026-08-01"),
    ],
)
def test_reference_rejects_any_non_exact_bridge_date_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    daily_dates: Sequence[str],
) -> None:
    fixture = _reference_fixture(monkeypatch, tmp_path, daily_dates)

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="exactly the registered eight bridge dates",
    ):
        first_week.validate_reference_ready(
            fixture[0],
            tmp_path / "ledger",
            fixture[1],
            fixture[2],
            "2026-07-31",
            state_summary=fixture[3],
            raw_summary=fixture[4],
        )


def test_reference_accepts_the_complete_exact_eight_date_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _reference_fixture(
        monkeypatch,
        tmp_path,
        first_week.FIRST_WEEK_REFERENCE_DATES,
    )

    summary = first_week.validate_reference_ready(
        fixture[0],
        tmp_path / "ledger",
        fixture[1],
        fixture[2],
        "2026-07-31",
        state_summary=fixture[3],
        raw_summary=fixture[4],
    )

    assert summary["bridge_dates"] == list(first_week.FIRST_WEEK_REFERENCE_DATES)
    assert summary["bridge_week_count"] == 8
    assert summary["decision_date"] == "2026-07-31"
    assert summary["entry_date"] == first_week.FIRST_WEEK_ENTRY_DATE
    assert summary["exit_date"] == first_week.FIRST_WEEK_EXIT_DATE
    assert summary["prospective_week_count"] == 0
    assert summary["formal_ledger_mutated"] is False


def test_guarded_append_replaces_collector_clock_with_fresh_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    original_append = stage3.append_record
    original_lock = stage3._exclusive_lock
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "anchor.json",
    )
    _install_fake_collector(monkeypatch)
    times = iter(
        (
            "2026-07-31T07:01:00Z",
            "2026-07-31T07:02:00Z",
        )
    )
    monkeypatch.setattr(stage3, "utc_now", lambda: next(times))

    record, _, _ = first_week.append_first_decision_with_final_guards(
        {},
        root,
        tmp_path / "reference.json",
        "2026-07-31",
        expected_head=head.data["record_hash"],
        expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
    )

    assert record.data["recorded_at_utc"] == "2026-07-31T07:02:00.000000Z"
    assert len(stage3.scan_records(root)) == 2
    assert stage3.append_record is original_append
    assert stage3._exclusive_lock is original_lock


def test_guarded_append_rejects_when_fresh_clock_crosses_entry_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    original_append = stage3.append_record
    original_lock = stage3._exclusive_lock
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "unused-anchor.json",
    )
    _install_fake_collector(monkeypatch)
    times = iter(
        (
            "2026-07-31T07:01:00Z",
            "2026-08-03T01:30:00Z",
        )
    )
    monkeypatch.setattr(stage3, "utc_now", lambda: next(times))

    with pytest.raises(
        stage3.Stage3ValidationError,
        match="outside",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        )

    assert len(stage3.scan_records(root)) == 1
    assert stage3.append_record is original_append
    assert stage3._exclusive_lock is original_lock


def test_guarded_append_rejects_at_reserved_final_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    before = first_week.ledger_closure(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "unused-anchor.json",
    )
    _install_fake_collector(monkeypatch)
    times = iter(
        (
            "2026-07-31T07:01:00Z",
            "2026-08-02T23:30:00Z",
        )
    )
    monkeypatch.setattr(stage3, "utc_now", lambda: next(times))

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="decision apply must start",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        )

    assert first_week.ledger_closure(root) == before


def test_guarded_append_rechecks_and_rejects_raw_drift_inside_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    checks = 0

    def verify_raw(_closure: Mapping[str, Any]) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise stage3.Stage3ValidationError("raw closure drifted")

    _install_guard_dependencies(
        monkeypatch,
        verifier=verify_raw,
        anchor_path=tmp_path / "unused-anchor.json",
    )
    _install_fake_collector(monkeypatch)
    monkeypatch.setattr(
        stage3,
        "utc_now",
        lambda: "2026-07-31T07:01:00Z",
    )

    with pytest.raises(
        stage3.Stage3ValidationError,
        match="raw closure drifted",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        )

    assert checks == 2
    assert len(stage3.scan_records(root)) == 1


def test_guarded_append_binds_the_preflight_decision_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    before = first_week.ledger_closure(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "unused-anchor.json",
    )
    _install_fake_collector(monkeypatch)
    monkeypatch.setattr(
        stage3,
        "utc_now",
        lambda: "2026-07-31T07:01:00Z",
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="preflight path changed",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256="8" * 64,
        )

    assert first_week.ledger_closure(root) == before


def test_guarded_append_rejects_expected_head_mismatch_before_collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    _append_seed_record(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "unused-anchor.json",
    )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("collector ran after the expected-head guard failed")

    monkeypatch.setattr(stage3, "append_decision_freeze", fail_if_called)

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="ledger head changed",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            "2026-07-31",
            expected_head="0" * 64,
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        )

    assert len(stage3.scan_records(root)) == 1


def test_frozen_collector_spec_and_study_identity_are_unchanged() -> None:
    spec = stage3.load_and_validate_spec()

    assert stage3.sha256_file(stage3.SOURCE_PATH) == FROZEN_COLLECTOR_SHA256
    assert stage3.sha256_file(stage3.SPEC_PATH) == FROZEN_SPEC_SHA256
    assert stage3.study_identity(spec) == FROZEN_STUDY_IDENTITY
