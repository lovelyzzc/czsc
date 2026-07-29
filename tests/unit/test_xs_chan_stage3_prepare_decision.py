from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage3 as stage3  # noqa: E402
import xs_chan_stage3_first_week as operator  # noqa: E402


@dataclass
class _PrepareHarness:
    root: Path
    data_dir: Path
    snapshot_root: Path
    state_output_root: Path
    state_manifest_path: Path
    reference_path: Path
    context: operator.DecisionOperationContext
    start_clock: datetime
    before: dict[str, Any]
    events: list[str]
    apply_window_clocks: list[datetime | None]
    raw_namespaces: list[Any]
    build_calls: list[tuple[Path, Path, int, operator.state_cache.StateCacheConfig]]
    stored_reports: list[dict[str, Any]]

    def prepare(self) -> dict[str, Any]:
        return operator.prepare_decision_data(
            decision_date=None,
            data_dir=self.data_dir,
            snapshot_root=self.snapshot_root,
            state_output_root=self.state_output_root,
            workers=3,
            now=self.start_clock,
        )

    def validate_receipt(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return operator.validate_preflight_receipt(
            Path(str(result["preflight_report_path"])),
            reference_manifest_path=Path(str(result["reference_manifest_path"])),
            state_manifest_path=Path(str(result["state_manifest_path"])),
            expected_head=str(result["expected_head"]),
            context=self.context,
        )


def _install_prepare_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    raw_statuses: Sequence[str],
) -> _PrepareHarness:
    root = tmp_path / "ledger"
    data_dir = tmp_path / "raw"
    snapshot_root = tmp_path / "snapshots"
    state_output_root = tmp_path / "states"
    expected_state_dir = state_output_root / "CHAN_STATE_CACHE_fixture"
    state_manifest_path = expected_state_dir / "source_manifest.json"
    data_dir.mkdir()
    snapshot_root.mkdir()
    state_output_root.mkdir()

    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:prepare-decision-fixture",
        {"purpose": "physical prepare-decision integration ledger"},
        "2026-07-28T06:00:00Z",
    )
    head = str(genesis.data["record_hash"])
    ledger_id = str(genesis.data["ledger_id"])
    spec = {
        "study_id": "stage3-prepare-decision-fixture",
        "accrual": {"window_weeks": 52},
        "historical_exclusion": {
            "first_prospective_decision_date": "2026-07-31",
        },
    }
    context = operator.DecisionOperationContext(
        week_index=1,
        decision_count=0,
        decision_date="2026-07-31",
        entry_date="2026-08-03",
        exit_date="2026-08-10",
        reference_dates=operator.FIRST_WEEK_REFERENCE_DATES,
        initial_decision_date="2026-06-05",
        initial_membership=("000001.SZ",),
        previous_decision_record_hash=None,
        expected_head=head,
        ledger_id=ledger_id,
        bootstrap=True,
    )
    start_clock = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)
    events: list[str] = []
    apply_window_clocks: list[datetime | None] = []
    raw_namespaces: list[Any] = []
    build_calls: list[tuple[Path, Path, int, operator.state_cache.StateCacheConfig]] = []
    stored_reports: list[dict[str, Any]] = []
    attempt = 0

    def mark(label: str) -> None:
        events.append(f"{attempt}:{label}")

    monkeypatch.setattr(stage3, "load_and_validate_spec", lambda: spec)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)

    def derive_context(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        *,
        decision_date: str | None = None,
        reference_manifest_path: Path | None = None,
        records: Sequence[stage3.LedgerRecord] | None = None,
    ) -> operator.DecisionOperationContext:
        nonlocal attempt
        assert actual_spec is spec
        assert actual_root == root
        assert records is None
        if decision_date is None and reference_manifest_path is None:
            attempt += 1
            mark("derive:auto")
        elif reference_manifest_path is None:
            assert decision_date == context.decision_date
            mark("derive:locked")
        else:
            assert decision_date == context.decision_date
            assert reference_manifest_path.resolve() == reference_path.resolve()
            mark("derive:final-reference")
        return context

    monkeypatch.setattr(operator, "derive_decision_operation_context", derive_context)

    def frozen_data(candidate: Path) -> Path:
        mark("frozen-data")
        assert candidate.resolve() == data_dir.resolve()
        return candidate.resolve()

    monkeypatch.setattr(operator, "validate_frozen_data_dir", frozen_data)

    def prepare_time(
        decision_date: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, str]:
        mark("prepare-time")
        assert decision_date == context.decision_date
        assert now == start_clock
        return {"clock": start_clock.isoformat()}

    monkeypatch.setattr(operator, "validate_prepare_time", prepare_time)

    def apply_window(
        decision_date: str,
        entry_date: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, str]:
        assert (decision_date, entry_date) == (context.decision_date, context.entry_date)
        apply_window_clocks.append(now)
        phase = "start" if now is not None else "finish-fresh-clock"
        mark(f"apply-window:{phase}")
        return {"phase": phase}

    monkeypatch.setattr(operator, "validate_apply_window", apply_window)

    formal_git = {"head": "3" * 40, "remote_verified": True}

    def git_ready(*, verify_remote: bool) -> dict[str, Any]:
        mark("git-ready")
        assert verify_remote is True
        return formal_git

    monkeypatch.setattr(operator, "validate_git_ready", git_ready)

    @contextmanager
    def traced_operator_lock(actual_root: Path) -> Iterator[None]:
        assert actual_root == root
        mark("operator-lock:enter")
        try:
            yield
        finally:
            mark("operator-lock:exit")

    @contextmanager
    def traced_cache_lock(actual_data_dir: Path) -> Iterator[None]:
        assert actual_data_dir == data_dir.resolve()
        mark("cache-lock:enter")
        try:
            yield
        finally:
            mark("cache-lock:exit")

    monkeypatch.setattr(operator, "operator_lock", traced_operator_lock)
    monkeypatch.setattr(operator, "cache_lock", traced_cache_lock)

    def assert_physical_chain(
        actual_records: Sequence[stage3.LedgerRecord],
    ) -> None:
        assert len(actual_records) == 1
        assert actual_records[0].path.is_file()
        assert actual_records[0].data["record_hash"] == head

    def anchor_chain(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        *,
        require_pushed: bool,
    ) -> list[Any]:
        mark("anchor-chain")
        assert actual_spec is spec
        assert actual_root == root
        assert require_pushed is True
        return []

    def prior_evidence(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        actual_records: Sequence[stage3.LedgerRecord],
    ) -> list[Any]:
        mark("prior-evidence")
        assert actual_spec is spec
        assert actual_root == root
        assert_physical_chain(actual_records)
        return []

    def due_evidence(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        actual_records: Sequence[stage3.LedgerRecord],
        actual_context: operator.DecisionOperationContext,
    ) -> list[Any]:
        mark("due-evidence")
        assert actual_spec is spec
        assert actual_root == root
        assert_physical_chain(actual_records)
        assert actual_context is context
        return []

    def global_evidence(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        actual_records: Sequence[stage3.LedgerRecord],
        *,
        verify_actual_remote: bool,
    ) -> list[Any]:
        mark("global-evidence")
        assert actual_spec is spec
        assert actual_root == root
        assert_physical_chain(actual_records)
        assert verify_actual_remote is True
        return []

    monkeypatch.setattr(operator, "validate_existing_anchor_chain", anchor_chain)
    monkeypatch.setattr(operator, "validate_prior_decision_evidence", prior_evidence)
    monkeypatch.setattr(operator, "validate_due_label_evidence", due_evidence)
    monkeypatch.setattr(operator, "validate_global_weekly_label_evidence", global_evidence)

    statuses = tuple(raw_statuses)

    def raw_sync(args: Any) -> dict[str, Any]:
        mark(f"raw:{statuses[len(raw_namespaces)]}")
        raw_namespaces.append(args)
        assert args.data_dir == data_dir.resolve()
        assert args.snapshot_root == snapshot_root.resolve()
        assert args.end_date == "20260731"
        assert args.min_daily_rows == operator.DEFAULT_MIN_DAILY_ROWS
        assert args.new_symbol_sleep_seconds == operator.DEFAULT_NEW_SYMBOL_SLEEP_SECONDS
        assert args.apply is True
        return {
            "status": statuses[len(raw_namespaces) - 1],
            "safe_end_date": "20260731",
        }

    monkeypatch.setattr(operator, "run_sync", raw_sync)

    raw_inventory_sha = "4" * 64
    raw_audit_sha = "5" * 64

    def raw_ready(
        decision_date: str,
        *,
        data_dir: Path,
    ) -> dict[str, Any]:
        mark("raw-ready")
        assert decision_date == context.decision_date
        assert data_dir.resolve() == globals_data_dir
        return {
            "data_dir": str(data_dir),
            "file_count": 1,
            "audit_sha256": raw_audit_sha,
            "parquet_inventory_sha256": raw_inventory_sha,
        }

    globals_data_dir = data_dir.resolve()
    monkeypatch.setattr(operator, "validate_raw_ready", raw_ready)

    def expected_state_cache_dir(
        actual_data_dir: Path,
        actual_output_root: Path,
    ) -> Path:
        mark("state-identity")
        assert actual_data_dir == data_dir.resolve()
        assert actual_output_root == state_output_root.resolve()
        return expected_state_dir

    monkeypatch.setattr(operator, "_expected_state_cache_dir", expected_state_cache_dir)

    def build_state_cache(
        actual_data_dir: Path,
        actual_output_root: Path,
        *,
        workers: int,
        config: operator.state_cache.StateCacheConfig,
    ) -> operator.state_cache.StateCacheBuildResult:
        mark("state-build")
        build_calls.append((actual_data_dir, actual_output_root, workers, config))
        expected_state_dir.mkdir()
        state_manifest_path.write_bytes(b'{"fixture":"built-state"}')
        return operator.state_cache.StateCacheBuildResult(
            cache_id="fixture",
            cache_dir=expected_state_dir,
            projection_path=expected_state_dir / "states.parquet",
            manifest_path=state_manifest_path,
            audit_path=expected_state_dir / "state_audit.json",
            audit_parquet_path=expected_state_dir / "state_audit.parquet",
            audit_passed=True,
        )

    monkeypatch.setattr(operator.state_cache, "build_state_cache", build_state_cache)

    state_projection_sha = "6" * 64

    def state_ready(
        actual_manifest_path: Path,
        decision_date: str,
        *,
        data_dir: Path,
    ) -> dict[str, Any]:
        mark("state-ready")
        assert actual_manifest_path == state_manifest_path
        assert decision_date == context.decision_date
        assert data_dir.resolve() == globals_data_dir
        return {
            "manifest_sha256": operator.sha256_file(actual_manifest_path),
            "projection_sha256": state_projection_sha,
            "raw_parquet_inventory_sha256": raw_inventory_sha,
        }

    monkeypatch.setattr(operator, "validate_state_ready", state_ready)

    reference_raw = b'{"fixture":"reference"}'
    reference_sha = operator.sha256_bytes(reference_raw)
    reference_path = root / "objects" / "reference_manifest" / f"{reference_sha}.json"

    def fetch_reference(
        actual_spec: Mapping[str, Any],
        reference_dates: Sequence[str],
        *,
        root: Path,
        state_manifest_path: Path,
    ) -> tuple[dict[str, Any], Path]:
        mark("reference-fetch")
        assert actual_spec is spec
        assert tuple(reference_dates) == context.reference_dates
        assert root == reference_path.parents[2]
        assert state_manifest_path.resolve() == expected_state_manifest.resolve()
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        if reference_path.exists():
            assert reference_path.read_bytes() == reference_raw
        else:
            reference_path.write_bytes(reference_raw)
        return {"manifest_sha256": reference_sha}, reference_path

    expected_state_manifest = state_manifest_path
    monkeypatch.setattr(stage3, "fetch_reference_data", fetch_reference)

    before = operator.ledger_closure(root)
    assert before["record_count"] == 1
    assert len(before["record_files"]) == 1
    bridge_path_sha = "7" * 64
    operator_source_sha = operator.sha256_file(Path(operator.__file__).resolve())

    def preflight(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        actual_reference_path: Path,
        actual_state_path: Path,
        decision_date: str,
        *,
        data_dir: Path,
        verify_remote: bool,
        generated_at: datetime | None,
        context: operator.DecisionOperationContext,
    ) -> dict[str, Any]:
        mark("preflight")
        assert actual_spec is spec
        assert actual_root == root
        assert actual_reference_path == reference_path
        assert actual_state_path == state_manifest_path
        assert decision_date == context.decision_date
        assert data_dir.resolve() == globals_data_dir
        assert verify_remote is True
        assert generated_at == start_clock
        assert context is expected_context
        closure = operator.ledger_closure(root)
        assert closure == before
        context_evidence = context.evidence()
        return {
            "schema": "xs_chan_stage3_first_week_preflight_v1",
            "mode": "READ_ONLY_FULL_BRIDGE",
            "generated_at_utc": start_clock.isoformat(),
            "study_id": spec["study_id"],
            "study_identity": "8" * 64,
            "decision_date": context.decision_date,
            "decision_context": context_evidence,
            "collector_source_sha256": "9" * 64,
            "operator_source_sha256": operator_source_sha,
            "git": formal_git,
            "ledger_before": closure,
            "ledger_after": closure,
            "formal_ledger_mutated": False,
            "anchors": [],
            "prior_decision_evidence": [],
            "due_label_evidence": [],
            "global_label_evidence": [],
            "raw": {
                "audit_sha256": raw_audit_sha,
                "parquet_inventory_sha256": raw_inventory_sha,
            },
            "state": {
                "manifest_sha256": operator.sha256_file(state_manifest_path),
                "projection_sha256": state_projection_sha,
                "raw_parquet_inventory_sha256": raw_inventory_sha,
            },
            "reference": {
                "manifest_sha256": reference_sha,
                "bridge_dates": list(context.reference_dates),
                "bridge_path_sha256": bridge_path_sha,
                "decision_context": context_evidence,
                "decision_inputs_only": True,
                "prospective_week_count": context.decision_count,
            },
            "efficacy_output": "FORBIDDEN",
            "next_state": "READY_DECISION_WINDOW",
        }

    expected_context = context
    monkeypatch.setattr(operator, "build_preflight_report", preflight)

    def rehearse(
        actual_spec: Mapping[str, Any],
        actual_root: Path,
        actual_reference_path: Path,
    ) -> dict[str, Any]:
        mark("rehearse")
        assert actual_spec is spec
        assert actual_root == root
        assert actual_reference_path == reference_path
        return {
            "status": "REHEARSAL_VALID_NEVER_COUNTS",
            "prospective_week_count": context.decision_count,
            "efficacy_summary_emitted": False,
        }

    monkeypatch.setattr(stage3, "rehearse_pipeline", rehearse)

    real_assert_ledger_unchanged = operator.assert_ledger_unchanged

    def assert_ledger_unchanged(
        expected: Mapping[str, Any],
        actual_root: Path,
    ) -> dict[str, Any]:
        mark("ledger-unchanged")
        return real_assert_ledger_unchanged(expected, actual_root)

    monkeypatch.setattr(operator, "assert_ledger_unchanged", assert_ledger_unchanged)

    real_store_preflight = operator.store_preflight_report

    def store_preflight(
        actual_root: Path,
        report: Mapping[str, Any],
    ) -> tuple[str, Path]:
        status = str(report["raw_sync"]["status"])
        mark(f"store:{status}")
        stored_reports.append(dict(report))
        return real_store_preflight(actual_root, report)

    monkeypatch.setattr(operator, "store_preflight_report", store_preflight)

    return _PrepareHarness(
        root=root,
        data_dir=data_dir,
        snapshot_root=snapshot_root,
        state_output_root=state_output_root,
        state_manifest_path=state_manifest_path,
        reference_path=reference_path,
        context=context,
        start_clock=start_clock,
        before=before,
        events=events,
        apply_window_clocks=apply_window_clocks,
        raw_namespaces=raw_namespaces,
        build_calls=build_calls,
        stored_reports=stored_reports,
    )


def _event_position(events: Sequence[str], value: str) -> int:
    return list(events).index(value)


def _assert_common_result(
    harness: _PrepareHarness,
    result: Mapping[str, Any],
) -> None:
    report_path = Path(str(result["preflight_report_path"]))
    assert result["status"] == "DECISION_DATA_PREPARED"
    assert result["decision_context"] == harness.context.evidence()
    assert result["reference_manifest_path"] == str(harness.reference_path)
    assert result["state_manifest_path"] == str(harness.state_manifest_path)
    assert result["preflight_report_sha256"] == report_path.stem
    assert operator.sha256_file(report_path) == report_path.stem
    assert result["expected_head"] == harness.before["head"]
    assert result["formal_ledger_mutated"] is False
    assert operator.ledger_closure(harness.root) == harness.before


def test_prepare_decision_published_then_current_reuses_state_and_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_prepare_harness(
        monkeypatch,
        tmp_path,
        raw_statuses=("PUBLISHED", "ALREADY_CURRENT"),
    )

    first = harness.prepare()
    assert len(harness.build_calls) == 1
    assert operator.ledger_closure(harness.root) == harness.before
    first_receipt = harness.validate_receipt(first)
    second = harness.prepare()
    assert operator.ledger_closure(harness.root) == harness.before
    second_receipt = harness.validate_receipt(second)

    assert len(harness.build_calls) == 1
    assert len(harness.raw_namespaces) == 2
    assert harness.apply_window_clocks == [
        harness.start_clock,
        None,
        harness.start_clock,
        None,
    ]
    assert harness.events.count("1:state-build") == 1
    assert "2:state-build" not in harness.events
    for attempt, status in enumerate(("PUBLISHED", "ALREADY_CURRENT"), start=1):
        assert harness.events.count(f"{attempt}:derive:auto") == 1
        assert harness.events.count(f"{attempt}:derive:locked") == 1
        assert harness.events.count(f"{attempt}:derive:final-reference") == 1
        assert harness.events.count(f"{attempt}:ledger-unchanged") == 2
        assert (
            _event_position(harness.events, f"{attempt}:raw:{status}")
            < _event_position(harness.events, f"{attempt}:state-ready")
            < _event_position(harness.events, f"{attempt}:reference-fetch")
            < _event_position(harness.events, f"{attempt}:preflight")
            < _event_position(harness.events, f"{attempt}:apply-window:finish-fresh-clock")
            < _event_position(harness.events, f"{attempt}:store:{status}")
        )

    _assert_common_result(harness, first)
    _assert_common_result(harness, second)
    assert first_receipt["raw_sync"]["status"] == "PUBLISHED"
    assert second_receipt["raw_sync"]["status"] == "ALREADY_CURRENT"
    assert first["preflight_report_path"] != second["preflight_report_path"]
    assert len(harness.stored_reports) == 2
    assert operator.ledger_closure(harness.root) == harness.before


def test_prepare_decision_current_raw_rebuilds_state_missing_after_raw_only_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_prepare_harness(
        monkeypatch,
        tmp_path,
        raw_statuses=("ALREADY_CURRENT",),
    )
    assert not harness.state_manifest_path.parent.exists()

    result = harness.prepare()

    assert len(harness.build_calls) == 1
    assert harness.events.count("1:state-build") == 1
    assert (
        _event_position(harness.events, "1:raw:ALREADY_CURRENT")
        < _event_position(harness.events, "1:state-build")
        < _event_position(harness.events, "1:store:ALREADY_CURRENT")
    )
    _assert_common_result(harness, result)
    receipt = harness.validate_receipt(result)
    assert receipt["raw_sync"]["status"] == "ALREADY_CURRENT"
    assert operator.ledger_closure(harness.root) == harness.before
