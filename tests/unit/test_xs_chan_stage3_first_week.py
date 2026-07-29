from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from contextvars import Context, copy_context
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
TEST_FORMAL_GIT = {
    "branch": first_week.REQUIRED_GIT_BRANCH,
    "head": "a" * 40,
    "upstream": first_week.REQUIRED_GIT_UPSTREAM,
    "upstream_head": "a" * 40,
    "remote_head": "a" * 40,
    "remote_fetch_url": first_week.REQUIRED_GIT_REMOTE_URL,
    "remote_push_url": first_week.REQUIRED_GIT_REMOTE_URL,
    "remote_verified": True,
    "worktree_clean": True,
}


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


def _source_completeness_fixture(
    trade_date: str,
    symbols: set[str],
) -> dict[str, Any]:
    rule = sync.source_symbol_completeness_rule()
    symbol_evidence = {
        "count": len(symbols),
        "sha256": sync.sha256_bytes(sync.canonical_json(sorted(symbols))),
    }
    without_hash = {
        "schema": "a_stock_daily_qfq_source_symbol_completeness_v1",
        "rule": rule,
        "baseline": {
            "trade_date": trade_date,
            "symbols": symbol_evidence,
        },
        "sessions": [
            {
                "trade_date": trade_date,
                "daily_symbols": symbol_evidence,
                "raw_adj_factor_symbols": symbol_evidence,
                "passed": True,
            }
        ],
        "passed": True,
    }
    return {
        **without_hash,
        "report_sha256": sync.sha256_bytes(sync.canonical_json(without_hash)),
    }


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
        first_week,
        "validate_existing_anchor_chain",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        stage3,
        "export_ledger_head_anchor",
        lambda *_args, **_kwargs: ({"status": "test-anchor"}, anchor_path),
    )
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda **_kwargs: {"head": "a" * 40},
    )
    monkeypatch.setattr(
        first_week,
        "store_append_authorization",
        lambda *_args, **_kwargs: (
            {"schema": first_week.AUTHORIZATION_SCHEMA},
            "b" * 64,
            anchor_path.parent / "append-authorization.json",
        ),
    )
    monkeypatch.setattr(
        first_week,
        "export_authorization_sidecar",
        lambda *_args, **_kwargs: (
            {"schema": first_week.AUTHORIZATION_SIDECAR_SCHEMA},
            anchor_path.parent / "authorization-sidecar.json",
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        lambda *_args, **_kwargs: {},
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
            ledger_id=stage3.scan_records(root)[0].data["ledger_id"],
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
                "ts_code": ["000001.SZ"],
                "trade_date": [date_text],
            }
        )
        for date_text in daily_dates
    }
    sessions = pd.DatetimeIndex(pd.bdate_range("2026-06-08", first_week.FIRST_WEEK_EXIT_DATE))
    raw_file = {
        "name": "000001.SZ.parquet",
        "sha256": "f" * 64,
        "max_dt": "20260731",
    }
    raw_summary["target_daily_symbols"] = first_week._symbol_set_evidence({"000001.SZ"})
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
    monkeypatch.setattr(first_week, "REFERENCE_SYMBOL_ABSOLUTE_MINIMUM", 1)
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


def test_git_readiness_binds_the_exact_branch_remote_url_and_actual_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    head = "a" * 40
    outputs = {
        ("symbolic-ref", "--quiet", "--short", "HEAD"): first_week.REQUIRED_GIT_BRANCH,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("rev-parse", "HEAD"): head,
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ): first_week.REQUIRED_GIT_UPSTREAM,
        ("rev-parse", "@{upstream}"): head,
        ("remote", "get-url", "mine"): first_week.REQUIRED_GIT_REMOTE_URL,
        (
            "remote",
            "get-url",
            "--push",
            "mine",
        ): first_week.REQUIRED_GIT_REMOTE_URL,
        (
            "ls-remote",
            "--heads",
            "mine",
            "refs/heads/feat/surge-wave-strategy",
        ): f"{head}\trefs/heads/feat/surge-wave-strategy",
    }
    monkeypatch.setattr(
        first_week,
        "_git_output",
        lambda _root, *args: outputs[args],
    )

    result = first_week.validate_git_ready(
        repo_root=tmp_path,
        verify_remote=True,
    )
    assert result["head"] == result["upstream_head"] == result["remote_head"]
    assert result["remote_verified"] is True

    outputs[("remote", "get-url", "--push", "mine")] = "git@example.invalid:wrong/repo.git"
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="remote URL differs",
    ):
        first_week.validate_git_ready(
            repo_root=tmp_path,
            verify_remote=True,
        )


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
        source_symbol_completeness=_source_completeness_fixture(
            "20260731",
            {"000001.SZ"},
        ),
    )
    audit = {
        "schema": "a_stock_daily_qfq_sync_audit_v2",
        "data_dir": str(data_dir.resolve()),
        "safe_end_date": "20260731",
        "after_inventory": {key: value for key, value in inventory.items() if key != "records"},
        "expected_active_inventory_sha256": inventory["content_inventory_sha256"],
        "execution_binding": execution_binding,
        "source_symbol_completeness": _source_completeness_fixture(
            "20260731",
            {"000001.SZ"},
        ),
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
    spec = _spec()
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
        spec,
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


def test_daily_basic_completeness_rejects_a_legacy_floor_truncation() -> None:
    complete = [f"{index:06d}.SZ" for index in range(5_000)]
    truncated = complete[:1_000]
    daily = {
        date_text: pd.DataFrame(
            {"ts_code": (truncated if date_text == first_week.FIRST_WEEK_REFERENCE_DATES[-1] else complete)}
        )
        for date_text in first_week.FIRST_WEEK_REFERENCE_DATES
    }

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="only 1000 unique symbols",
    ):
        first_week.validate_reference_symbol_completeness(
            daily,
            raw_summary={
                "data_dir": "/unused",
                "target_daily_symbols": {"count": 5_000, "sha256": "a" * 64},
            },
            decision_date="2026-07-31",
        )


def test_daily_basic_completeness_requires_the_complete_raw_target_universe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    daily_symbols = {f"{index:06d}.SZ" for index in range(4_000)}
    raw_symbols = {f"{index:06d}.SZ" for index in range(5_000)}
    daily = {
        date_text: pd.DataFrame({"ts_code": sorted(daily_symbols)})
        for date_text in first_week.FIRST_WEEK_REFERENCE_DATES
    }
    records = [
        {
            "name": f"{symbol}.parquet",
            "max_dt": "20260731",
        }
        for symbol in sorted(raw_symbols)
    ]
    monkeypatch.setattr(
        first_week,
        "inspect_inventory",
        lambda _path: {"records": records},
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="omits symbols present in the exact raw daily universe",
    ):
        first_week.validate_reference_symbol_completeness(
            daily,
            raw_summary={
                "data_dir": str(tmp_path),
                "target_daily_symbols": first_week._symbol_set_evidence(raw_symbols),
            },
            decision_date="2026-07-31",
        )


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

    record, *_ = first_week.append_first_decision_with_final_guards(
        {},
        root,
        tmp_path / "reference.json",
        tmp_path / "state.json",
        tmp_path / "preflight.json",
        "2026-07-31",
        expected_head=head.data["record_hash"],
        expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        final_preflight_report={},
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
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
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
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
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
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
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
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256="8" * 64,
            final_preflight_report={},
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
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head="0" * 64,
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
        )

    assert len(stage3.scan_records(root)) == 1


def test_guarded_append_rejects_an_orphan_sidecar_before_ledger_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    before = first_week.ledger_closure(root)
    tracked_scripts = tmp_path / "tracked-scripts"
    orphan = tracked_scripts / first_week.AUTHORIZATION_SIDECAR_DIR_NAME / "orphan.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tracked_scripts)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "unused-anchor.json",
    )
    _install_fake_collector(monkeypatch)

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="exact authorized decision chain",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
        )

    assert first_week.ledger_closure(root) == before


def test_guarded_append_rejects_an_orphan_anchor_before_ledger_mutation(
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

    def reject_orphan_anchor(
        _spec: Mapping[str, Any],
        checked_root: Path,
        *,
        require_pushed: bool,
    ) -> list[dict[str, Any]]:
        assert require_pushed is True
        assert len(stage3.scan_records(checked_root)) == 1
        raise first_week.FirstWeekOperationError("tracked anchor inventory has an orphan")

    monkeypatch.setattr(
        first_week,
        "validate_existing_anchor_chain",
        reject_orphan_anchor,
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="anchor inventory has an orphan",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
        )

    assert first_week.ledger_closure(root) == before


def test_frozen_collector_spec_and_study_identity_are_unchanged() -> None:
    spec = stage3.load_and_validate_spec()

    assert stage3.sha256_file(stage3.SOURCE_PATH) == FROZEN_COLLECTOR_SHA256
    assert stage3.sha256_file(stage3.SPEC_PATH) == FROZEN_SPEC_SHA256
    assert stage3.study_identity(spec) == FROZEN_STUDY_IDENTITY


def test_append_authorization_is_content_addressed_and_exactly_binds_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    reference_dir = root / "objects" / "reference_manifest"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / f"{'b' * 64}.json"
    reference_path.write_text("{}", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text('{"state": "frozen"}', encoding="utf-8")
    receipt_dir = root / "objects" / first_week.PREFLIGHT_CATEGORY
    receipt_dir.mkdir(parents=True)
    receipt_path = receipt_dir / f"{'c' * 64}.json"
    receipt_path.write_text("{}", encoding="utf-8")
    final_report = {
        "git": TEST_FORMAL_GIT,
        "raw": {
            "audit_sha256": "d" * 64,
            "parquet_inventory_sha256": "e" * 64,
        },
        "reference": {
            "raw_source_closure_sha256": "f" * 64,
            "bridge_path_sha256": TEST_DECISION_PATH_SHA256,
        },
        "state": {
            "projection_sha256": "1" * 64,
        },
    }
    decision_payload = {
        "decision_dt": "2026-07-31",
        "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
        "decision_path_sha256": TEST_DECISION_PATH_SHA256,
        "reference_manifest_sha256": reference_path.stem,
    }
    anticipated = first_week._anticipated_record_data(
        (genesis,),
        record_type="decision_freeze",
        logical_event_key="decision:2026-07-31",
        payload=decision_payload,
        recorded_at_utc="2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    monkeypatch.setattr(
        first_week,
        "validate_preflight_receipt",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        first_week,
        "_validate_final_preflight_report",
        lambda *_args, **_kwargs: None,
    )

    with first_week.authorization_ledger_lock(root):
        authorization, authorization_sha, authorization_path = first_week.store_append_authorization(
            spec,
            root,
            anticipated,
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            preflight_report_path=receipt_path,
            final_preflight_report=final_report,
            fresh_git=TEST_FORMAL_GIT,
        )
    assert len(stage3.scan_records(root)) == 1
    assert authorization_path.stem == authorization_sha == stage3.sha256_file(authorization_path)
    assert authorization["expected_record_hash"] == anticipated["record_hash"]

    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        decision_payload,
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    loaded, loaded_sha, loaded_path = first_week.load_append_authorization(
        spec,
        root,
        decision.data,
    )
    assert loaded == authorization
    assert loaded_sha == authorization_sha
    assert loaded_path == authorization_path

    monkeypatch.setattr(stage3, "SCRIPTS_DIR", tmp_path / "tracked-scripts")
    monkeypatch.setattr(
        first_week,
        "_git_file_sha256_at_commit",
        lambda *_args, **_kwargs: authorization["operator_source_sha256"],
    )
    sidecar, sidecar_path = first_week.export_authorization_sidecar(
        spec,
        root,
        decision.data,
    )
    validated = first_week.validate_authorization_sidecar(
        spec,
        root,
        decision.data,
        require_pushed=False,
    )
    assert sidecar["authorization_sha256"] == authorization_sha
    assert validated["path"] == str(sidecar_path)
    sidecar_path.unlink()
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="sidecar is missing",
    ):
        first_week.validate_authorization_sidecar(
            spec,
            root,
            decision.data,
            require_pushed=False,
        )
    assert not sidecar_path.exists()


def test_posthoc_authorization_for_existing_record_is_rejected_without_writing_intent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
            "decision_path_sha256": TEST_DECISION_PATH_SHA256,
            "reference_manifest_sha256": "b" * 64,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )

    with (
        pytest.raises(
            first_week.FirstWeekOperationError,
            match="before its record or logical event exists",
        ),
        first_week.authorization_ledger_lock(root),
    ):
        first_week.store_append_authorization(
            _spec(),
            root,
            decision.data,
            reference_manifest_path=tmp_path / "missing-reference.json",
            state_manifest_path=tmp_path / "missing-state.json",
            preflight_report_path=tmp_path / "missing-preflight.json",
            final_preflight_report={"git": {}},
            fresh_git={},
        )

    assert not (root / "objects" / first_week.FINAL_PREFLIGHT_CATEGORY).exists()
    assert not (root / "objects" / first_week.APPEND_AUTHORIZATION_CATEGORY).exists()


def test_append_authorization_requires_and_shares_the_physical_ledger_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    anticipated = first_week._anticipated_record_data(
        (genesis,),
        record_type="decision_freeze",
        logical_event_key="decision:2026-07-31",
        payload={
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
        },
        recorded_at_utc="2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="guarded physical ledger lock",
    ):
        first_week.store_append_authorization(
            _spec(),
            root,
            anticipated,
            reference_manifest_path=tmp_path / "missing-reference.json",
            state_manifest_path=tmp_path / "missing-state.json",
            preflight_report_path=tmp_path / "missing-preflight.json",
            final_preflight_report={"git": {}},
            fresh_git={},
        )

    attempting = threading.Event()
    finished = threading.Event()
    stale_context: Context | None = None

    def direct_append() -> None:
        attempting.set()
        stage3.append_record(
            root,
            "decision_freeze",
            "decision:2026-07-31",
            anticipated["payload"],
            anticipated["recorded_at_utc"],
            ledger_id=genesis.data["ledger_id"],
        )
        finished.set()

    worker = threading.Thread(target=direct_append)
    with first_week.authorization_ledger_lock(root):
        stale_context = copy_context()
        worker.start()
        assert attempting.wait(timeout=1)
        assert not finished.wait(timeout=0.1)
        assert len(stage3.scan_records(root)) == 1
    assert finished.wait(timeout=2)
    worker.join(timeout=2)
    assert len(stage3.scan_records(root)) == 2
    assert stale_context is not None
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="guarded physical ledger lock",
    ):
        stale_context.run(
            first_week.store_append_authorization,
            _spec(),
            root,
            anticipated,
            reference_manifest_path=tmp_path / "missing-reference.json",
            state_manifest_path=tmp_path / "missing-state.json",
            preflight_report_path=tmp_path / "missing-preflight.json",
            final_preflight_report={"git": {}},
            fresh_git={},
        )


def test_authorization_lock_lease_rejects_an_inherited_process_identity(
    tmp_path: Path,
) -> None:
    lease = first_week._AuthorizationLockLease(tmp_path.resolve())
    lease.pid += 1
    token = first_week._AUTHORIZED_LEDGER_LOCK_LEASE.set(lease)
    try:
        with pytest.raises(
            first_week.FirstWeekOperationError,
            match="forked process inherited",
        ):
            first_week._active_authorization_lock_lease()
    finally:
        first_week._AUTHORIZED_LEDGER_LOCK_LEASE.reset(token)


def test_authorization_lock_cleans_physical_registry_after_cross_context_exit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    entered_context = Context()
    exited_context = Context()
    lock = first_week.authorization_ledger_lock(root)

    entered_context.run(lock.__enter__)
    with pytest.raises(ValueError, match="different Context"):
        exited_context.run(lock.__exit__, None, None, None)

    assert root.resolve() not in first_week._physical_authorization_lock_registry()
    with first_week.authorization_ledger_lock(root):
        assert first_week._physical_authorization_lock_owned(root)


def test_guarded_append_does_not_expose_its_reentrant_lock_bypass_to_other_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    reference_path = tmp_path / f"{'b' * 64}.json"
    before = first_week.ledger_closure(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "anchor.json",
    )
    _install_fake_collector(monkeypatch)
    monkeypatch.setattr(
        stage3,
        "utc_now",
        lambda: "2026-07-31T07:01:00Z",
    )
    store_entered = threading.Event()
    release_store = threading.Event()
    guarded_finished = threading.Event()
    direct_attempting = threading.Event()
    direct_finished = threading.Event()
    failures: list[BaseException] = []
    copied_lock_context: dict[str, Context] = {}

    def paused_store(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str, Path]:
        copied_lock_context["value"] = copy_context()
        with pytest.raises(
            first_week.FirstWeekOperationError,
            match="another authorization context attempted",
        ):
            Context().run(
                stage3.append_record,
                root,
                "decision_freeze",
                "decision:2026-07-31",
                {
                    "decision_dt": "2026-07-31",
                    "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
                    "decision_path_sha256": TEST_DECISION_PATH_SHA256,
                    "reference_manifest_sha256": reference_path.stem,
                },
                "2026-07-31T07:01:00Z",
                ledger_id=genesis.data["ledger_id"],
            )
        store_entered.set()
        if not release_store.wait(timeout=2):
            raise AssertionError("test did not release the authorization store")
        return (
            {"schema": first_week.AUTHORIZATION_SCHEMA},
            "c" * 64,
            tmp_path / "authorization.json",
        )

    monkeypatch.setattr(first_week, "store_append_authorization", paused_store)

    def guarded_append() -> None:
        try:
            first_week.append_first_decision_with_final_guards(
                {},
                root,
                reference_path,
                tmp_path / "state.json",
                tmp_path / "preflight.json",
                "2026-07-31",
                expected_head=genesis.data["record_hash"],
                expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
                final_preflight_report={},
            )
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)
        finally:
            guarded_finished.set()

    def direct_append() -> None:
        direct_attempting.set()
        try:
            copied_lock_context["value"].run(
                stage3.append_record,
                root,
                "decision_freeze",
                "decision:2026-07-31",
                {
                    "decision_dt": "2026-07-31",
                    "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
                    "decision_path_sha256": TEST_DECISION_PATH_SHA256,
                    "reference_manifest_sha256": reference_path.stem,
                },
                "2026-07-31T07:01:00Z",
                ledger_id=genesis.data["ledger_id"],
            )
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)
        finally:
            direct_finished.set()

    guarded_worker = threading.Thread(target=guarded_append)
    guarded_worker.start()
    assert store_entered.wait(timeout=2)
    direct_worker = threading.Thread(target=direct_append)
    direct_worker.start()
    assert direct_attempting.wait(timeout=1)
    assert not direct_finished.wait(timeout=0.1)
    assert first_week.ledger_closure(root) == before

    release_store.set()
    assert guarded_finished.wait(timeout=2)
    assert direct_finished.wait(timeout=2)
    guarded_worker.join(timeout=2)
    direct_worker.join(timeout=2)
    assert failures == []
    assert len(stage3.scan_records(root)) == 2


def test_direct_decision_without_preappend_authorization_is_never_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
            "decision_path_sha256": TEST_DECISION_PATH_SHA256,
            "reference_manifest_sha256": "b" * 64,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    before = first_week.ledger_closure(root)
    monkeypatch.setattr(
        stage3,
        "validate_record_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        first_week,
        "_validate_recovery_anchor_inventory",
        lambda *_args, **_kwargs: False,
    )

    def fail_if_exported(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("an unauthorized direct decision reached an evidence export")

    monkeypatch.setattr(
        stage3,
        "export_ledger_head_anchor",
        fail_if_exported,
    )
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="no unique append-before-record authorization",
    ):
        first_week.recover_first_decision_anchor(
            spec=_spec(),
            root=root,
            record_hash=decision.data["record_hash"],
            decision_date="2026-07-31",
            reference_manifest_path=tmp_path / "missing-reference.json",
            state_manifest_path=tmp_path / "missing-state.json",
            preflight_report_path=tmp_path / "missing-receipt.json",
            expected_head=genesis.data["record_hash"],
            data_dir=data_dir,
            now=datetime(2026, 7, 31, 10, 1, tzinfo=UTC),
        )
    assert first_week.ledger_closure(root) == before


def test_status_marks_a_direct_decision_unauthorized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
            "decision_path_sha256": TEST_DECISION_PATH_SHA256,
            "reference_manifest_sha256": "b" * 64,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    monkeypatch.setattr(
        first_week,
        "validate_frozen_data_dir",
        lambda path: path,
    )
    monkeypatch.setattr(
        stage3,
        "resolve_ledger_root",
        lambda _spec: root,
    )
    monkeypatch.setattr(
        stage3,
        "validate_record_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stage3,
        "status_report",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        first_week,
        "inspect_inventory",
        lambda _path: {
            "file_count": 1,
            "max_dt": "20260731",
            "content_inventory_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda **_kwargs: TEST_FORMAL_GIT,
    )

    snapshot = first_week.status_snapshot(
        decision_date="2026-07-31",
        data_dir=tmp_path / "raw",
        now=datetime(2026, 7, 31, 10, 2, tzinfo=UTC),
    )
    assert snapshot["state"] == "UNAUTHORIZED_DECISION_PRESENT"
    assert snapshot["formal_ledger_mutated"] is False


def test_guarded_append_persists_authorization_before_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "anchor.json",
    )
    _install_fake_collector(monkeypatch)
    monkeypatch.setattr(
        stage3,
        "utc_now",
        lambda: "2026-07-31T07:01:00Z",
    )
    captured: dict[str, Any] = {}

    def store_before_record(
        _spec: Mapping[str, Any],
        _root: Path,
        anticipated: Mapping[str, Any],
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], str, Path]:
        assert len(stage3.scan_records(root)) == 1
        captured["anticipated"] = dict(anticipated)
        return (
            {"schema": first_week.AUTHORIZATION_SCHEMA},
            "b" * 64,
            tmp_path / "append-authorization.json",
        )

    monkeypatch.setattr(
        first_week,
        "store_append_authorization",
        store_before_record,
    )
    record, *_ = first_week.append_first_decision_with_final_guards(
        {},
        root,
        tmp_path / "reference.json",
        tmp_path / "state.json",
        tmp_path / "preflight.json",
        "2026-07-31",
        expected_head=head.data["record_hash"],
        expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
        final_preflight_report={},
    )
    assert captured["anticipated"]["record_hash"] == record.data["record_hash"]
    assert captured["anticipated"] == record.data


def test_guarded_append_rechecks_actual_remote_before_authorizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    head = _append_seed_record(root)
    before = first_week.ledger_closure(root)
    _install_guard_dependencies(
        monkeypatch,
        verifier=lambda _closure: None,
        anchor_path=tmp_path / "anchor.json",
    )
    _install_fake_collector(monkeypatch)
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda **_kwargs: (_ for _ in ()).throw(first_week.FirstWeekOperationError("actual remote drifted")),
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="actual remote drifted",
    ):
        first_week.append_first_decision_with_final_guards(
            {},
            root,
            tmp_path / "reference.json",
            tmp_path / "state.json",
            tmp_path / "preflight.json",
            "2026-07-31",
            expected_head=head.data["record_hash"],
            expected_decision_path_sha256=TEST_DECISION_PATH_SHA256,
            final_preflight_report={},
        )
    assert first_week.ledger_closure(root) == before


def test_atomic_stage3_write_is_complete_or_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    target = root / "records" / "record.json"
    original_write = stage3._exclusive_write
    requested_modes: list[int] = []
    original_fchmod = first_week.os.fchmod

    def record_fchmod(descriptor: int, mode: int) -> None:
        requested_modes.append(mode)
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(first_week.os, "fchmod", record_fchmod)

    with first_week.atomic_stage3_writes(root):
        stage3._exclusive_write(target, b"complete-record")

    assert target.read_bytes() == b"complete-record"
    assert requested_modes == [0o644]
    assert not (root / first_week.ATOMIC_STAGING_DIR).exists()
    assert stage3._exclusive_write is original_write

    failed_target = root / "records" / "failed.json"
    monkeypatch.setattr(
        first_week.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link failed")),
    )
    with (
        pytest.raises(OSError, match="link failed"),
        first_week.atomic_stage3_writes(root),
    ):
        stage3._exclusive_write(failed_target, b"must-not-be-partial")
    assert not failed_target.exists()
    assert not (root / first_week.ATOMIC_STAGING_DIR).exists()
    assert stage3._exclusive_write is original_write


def test_atomic_stage3_write_rejects_unknown_staging_residue(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    staging = root / first_week.ATOMIC_STAGING_DIR
    staging.mkdir(parents=True)
    (staging / "unknown").write_text("must not be silently removed", encoding="utf-8")

    with (
        pytest.raises(
            first_week.FirstWeekOperationError,
            match="unexpected atomic staging entry",
        ),
        first_week.atomic_stage3_writes(root),
    ):
        pytest.fail("unknown staging residue reached the write phase")


def test_atomic_stage3_writer_delegates_other_threads_to_their_original_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger-a"
    other_target = tmp_path / "ledger-b" / "records" / "record.json"
    finished = threading.Event()
    failures: list[BaseException] = []

    def write_other_ledger() -> None:
        try:
            stage3._exclusive_write(other_target, b"other-ledger")
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)
        finally:
            finished.set()

    with first_week.atomic_stage3_writes(root):
        worker = threading.Thread(target=write_other_ledger)
        worker.start()
        assert finished.wait(timeout=2)
        worker.join(timeout=2)
        assert failures == []
        assert other_target.read_bytes() == b"other-ledger"
        staging = root / first_week.ATOMIC_STAGING_DIR
        assert list(staging.iterdir()) == []


def test_atomic_stage3_write_rejects_a_symlink_staging_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "write_protected"
    protected.write_text("must survive", encoding="utf-8")
    root.mkdir()
    staging = root / first_week.ATOMIC_STAGING_DIR
    try:
        staging.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem does not support test symlinks: {exc}")

    with (
        pytest.raises(
            first_week.FirstWeekOperationError,
            match="cannot be a symlink",
        ),
        first_week.atomic_stage3_writes(root),
    ):
        pytest.fail("symlink staging reached the write phase")
    assert protected.read_text(encoding="utf-8") == "must survive"


def test_atomic_stage3_write_does_not_follow_a_staging_symlink_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "write_protected"
    protected.write_text("must survive", encoding="utf-8")
    moved = root / "moved-staging"
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem does not support test symlinks: {exc}")
    probe.unlink()

    with (
        pytest.raises(
            first_week.FirstWeekOperationError,
            match="directory entry changed",
        ),
        first_week.atomic_stage3_writes(root),
    ):
        staging = root / first_week.ATOMIC_STAGING_DIR
        staging.rename(moved)
        staging.symlink_to(outside, target_is_directory=True)

    assert protected.read_text(encoding="utf-8") == "must survive"


def test_recovery_deadline_extends_past_safe_start_but_not_entry_open() -> None:
    entry_open = first_week._entry_open(first_week.FIRST_WEEK_ENTRY_DATE).astimezone(UTC)
    safe_start = entry_open - first_week.ANCHOR_PUSH_RESERVE

    result = first_week.validate_recovery_deadline(
        first_week.FIRST_WEEK_ENTRY_DATE,
        now=safe_start,
    )

    assert result["checked_at_utc"] == safe_start
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="before entry open",
    ):
        first_week.validate_recovery_deadline(
            first_week.FIRST_WEEK_ENTRY_DATE,
            now=entry_open,
        )


def test_recovery_git_allows_only_exact_evidence_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    status = {
        "value": "?? scripts/sidecar.json\n?? scripts/anchor.json",
    }

    def git_output(_repo_root: Path, *args: str) -> str:
        if args[0] == "symbolic-ref":
            return first_week.REQUIRED_GIT_BRANCH
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[0] == "rev-parse" and args[-1] == "@{upstream}":
            return first_week.REQUIRED_GIT_UPSTREAM if "--abbrev-ref" in args else "a" * 40
        if args[:2] == ("remote", "get-url"):
            return first_week.REQUIRED_GIT_REMOTE_URL
        if args[0] == "ls-remote":
            return f"{'a' * 40}\trefs/heads/feat/surge-wave-strategy"
        if args[0] == "status":
            return status["value"]
        raise AssertionError(args)

    monkeypatch.setattr(first_week, "_git_output", git_output)
    allowed = {"scripts/sidecar.json", "scripts/anchor.json"}

    result = first_week.validate_recovery_git_ready(
        {"git": TEST_FORMAL_GIT},
        allowed_dirty_paths=allowed,
        repo_root=tmp_path,
    )

    assert result["allowed_dirty_paths"] == sorted(allowed)
    status["value"] += "\n M scripts/unrelated.py"
    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="beyond the exact decision evidence",
    ):
        first_week.validate_recovery_git_ready(
            {"git": TEST_FORMAL_GIT},
            allowed_dirty_paths=allowed,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "commit_line",
    [
        f"{'c' * 40} {'b' * 40}",
        f"{'c' * 40} {'a' * 40} {'b' * 40}",
    ],
)
def test_authorized_pair_requires_single_exact_authorized_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit_line: str,
) -> None:
    anchor_path = stage3.REPO_ROOT / "scripts" / "test-anchor.json"
    sidecar_path = stage3.REPO_ROOT / "scripts" / "test-sidecar.json"
    record = {"record_hash": "d" * 64}
    monkeypatch.setattr(
        first_week,
        "validate_authorization_sidecar_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorization_sidecar",
        lambda *_args, **_kwargs: {
            "path": str(sidecar_path),
            "commit": "c" * 40,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_anchor_for_record",
        lambda *_args, **_kwargs: {
            "path": str(anchor_path),
            "commit": "c" * 40,
        },
    )
    monkeypatch.setattr(
        first_week,
        "load_append_authorization",
        lambda *_args, **_kwargs: (
            {"git": {"head": "a" * 40}},
            "e" * 64,
            tmp_path / "authorization.json",
        ),
    )
    monkeypatch.setattr(
        first_week,
        "_git_output",
        lambda _repo_root, *_args: commit_line,
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="exactly the authorized Git head as its parent",
    ):
        first_week.validate_authorized_anchor_pair(
            {},
            tmp_path,
            record,
            require_pushed=True,
        )


@pytest.mark.parametrize("crosses_entry_during_export", [False, True])
def test_recovery_with_exact_preexisting_authorization_revalidates_all_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    crosses_entry_during_export: bool,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    reference_path = root / "objects" / "reference_manifest" / f"{'b' * 64}.json"
    state_path = tmp_path / "state.json"
    receipt_path = root / "objects" / first_week.PREFLIGHT_CATEGORY / f"{'c' * 64}.json"
    reference_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    reference_path.write_text("{}", encoding="utf-8")
    receipt_path.write_text("{}", encoding="utf-8")
    state_path.write_text("{}", encoding="utf-8")
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
            "decision_path_sha256": TEST_DECISION_PATH_SHA256,
            "reference_manifest_sha256": reference_path.stem,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    authorization = {
        "state_manifest_path": str(state_path.resolve()),
        "reference_manifest_object": str(reference_path.relative_to(root)),
        "preflight_report_object": str(receipt_path.relative_to(root)),
        "git": TEST_FORMAL_GIT,
        "raw_audit_sha256": "d" * 64,
        "raw_parquet_inventory_sha256": "e" * 64,
        "raw_source_closure_sha256": "f" * 64,
        "state_manifest_sha256": "1" * 64,
        "state_projection_sha256": "2" * 64,
        "reference_manifest_sha256": reference_path.stem,
        "decision_path_sha256": TEST_DECISION_PATH_SHA256,
    }
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    authorization_path = root / "objects" / first_week.APPEND_AUTHORIZATION_CATEGORY / f"{'3' * 64}.json"
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text("{}", encoding="utf-8")
    sidecar_path = tmp_path / "sidecar.json"
    anchor_path = tmp_path / "anchor.json"
    sidecar_path.write_text("{}", encoding="utf-8")
    anchor_path.write_text("{}", encoding="utf-8")
    deadline_checks = 0

    def recovery_deadline(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
        nonlocal deadline_checks
        deadline_checks += 1
        if crosses_entry_during_export and deadline_checks == 3:
            raise first_week.FirstWeekOperationError("entry open crossed during evidence export")
        return {"checked": True}

    monkeypatch.setattr(first_week, "validate_recovery_deadline", recovery_deadline)
    monkeypatch.setattr(
        stage3,
        "validate_record_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        first_week,
        "_validate_recovery_anchor_inventory",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        first_week,
        "_validate_recovery_authorization_inventory",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        first_week,
        "load_append_authorization",
        lambda *_args, **_kwargs: (
            authorization,
            "3" * 64,
            authorization_path,
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_git_ready",
        lambda *_args, **_kwargs: TEST_FORMAL_GIT,
    )
    monkeypatch.setattr(
        first_week,
        "validate_raw_ready",
        lambda *_args, **_kwargs: {
            "audit_sha256": "d" * 64,
            "parquet_inventory_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_state_ready",
        lambda *_args, **_kwargs: {
            "manifest_sha256": "1" * 64,
            "projection_sha256": "2" * 64,
            "raw_parquet_inventory_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_reference_ready",
        lambda *_args, **_kwargs: {
            "manifest_sha256": reference_path.stem,
            "raw_source_closure_sha256": "f" * 64,
            "bridge_path_sha256": TEST_DECISION_PATH_SHA256,
        },
    )
    monkeypatch.setattr(
        first_week,
        "export_authorization_sidecar",
        lambda *_args, **_kwargs: ({"authorized": True}, sidecar_path),
    )
    monkeypatch.setattr(
        stage3,
        "export_ledger_head_anchor",
        lambda *_args, **_kwargs: ({"anchored": True}, anchor_path),
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        lambda *_args, **kwargs: (
            (_ for _ in ()).throw(first_week.FirstWeekOperationError("pair not pushed"))
            if kwargs.get("require_pushed")
            else {}
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_existing_anchor_chain",
        lambda *_args, **_kwargs: [],
    )

    if crosses_entry_during_export:
        with pytest.raises(
            first_week.FirstWeekOperationError,
            match="entry open crossed during evidence export",
        ):
            first_week.recover_first_decision_anchor(
                spec=_spec(),
                root=root,
                record_hash=decision.data["record_hash"],
                decision_date="2026-07-31",
                reference_manifest_path=reference_path,
                state_manifest_path=state_path,
                preflight_report_path=receipt_path,
                expected_head=genesis.data["record_hash"],
                data_dir=data_dir,
                now=datetime(2026, 7, 31, 10, 2, tzinfo=UTC),
            )
        assert deadline_checks == 3
        return

    result = first_week.recover_first_decision_anchor(
        spec=_spec(),
        root=root,
        record_hash=decision.data["record_hash"],
        decision_date="2026-07-31",
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=receipt_path,
        expected_head=genesis.data["record_hash"],
        data_dir=data_dir,
        now=datetime(2026, 7, 31, 10, 2, tzinfo=UTC),
    )
    assert result["status"] == "AUTHORIZED_DECISION_ANCHOR_RECOVERED"
    assert result["record_hash"] == decision.data["record_hash"]
    assert result["formal_ledger_mutated"] is False
    assert deadline_checks == 3


def test_recovery_of_already_pushed_pair_is_idempotent_after_entry_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    genesis = _append_seed_record(root)
    reference_path = root / "objects" / "reference_manifest" / f"{'b' * 64}.json"
    receipt_path = root / "objects" / first_week.PREFLIGHT_CATEGORY / f"{'c' * 64}.json"
    state_path = tmp_path / "state.json"
    for path in (reference_path, receipt_path, state_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": first_week.FIRST_WEEK_ENTRY_DATE,
            "decision_path_sha256": TEST_DECISION_PATH_SHA256,
            "reference_manifest_sha256": reference_path.stem,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    authorization_path = root / "objects" / first_week.APPEND_AUTHORIZATION_CATEGORY / f"{'3' * 64}.json"
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text("{}", encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    sidecar_path = tmp_path / "sidecar.json"
    anchor_path.write_text('{"anchored":true}', encoding="utf-8")
    sidecar_path.write_text('{"authorized":true}', encoding="utf-8")
    authorization = {
        "state_manifest_path": str(state_path.resolve()),
        "reference_manifest_object": str(reference_path.relative_to(root)),
        "preflight_report_object": str(receipt_path.relative_to(root)),
    }
    monkeypatch.setattr(
        stage3,
        "validate_record_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        first_week,
        "_validate_recovery_anchor_inventory",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        first_week,
        "_validate_recovery_authorization_inventory",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        first_week,
        "load_append_authorization",
        lambda *_args, **_kwargs: (
            authorization,
            "3" * 64,
            authorization_path,
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        lambda *_args, **_kwargs: {
            "anchor": {"path": str(anchor_path)},
            "authorization": {"path": str(sidecar_path)},
            "commit": "4" * 40,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda **_kwargs: TEST_FORMAL_GIT,
    )

    def fail_if_revalidated(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("an already-pushed evidence pair re-entered mutable recovery")

    monkeypatch.setattr(first_week, "validate_recovery_deadline", fail_if_revalidated)
    monkeypatch.setattr(first_week, "validate_raw_ready", fail_if_revalidated)

    result = first_week.recover_first_decision_anchor(
        spec=_spec(),
        root=root,
        record_hash=decision.data["record_hash"],
        decision_date="2026-07-31",
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=receipt_path,
        expected_head=genesis.data["record_hash"],
        data_dir=tmp_path / "raw",
        now=datetime(2026, 8, 10, 10, 0, tzinfo=UTC),
    )

    assert result["status"] == "DECISION_EVIDENCE_ALREADY_COMMITTED"
    assert result["evidence_commit"] == "4" * 40
    assert result["formal_ledger_mutated"] is False
