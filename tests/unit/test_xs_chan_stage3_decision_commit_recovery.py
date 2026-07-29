from __future__ import annotations

import copy
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage3 as stage3  # noqa: E402
import xs_chan_stage3_first_week as first_week  # noqa: E402
import xs_chan_stage3_weekly as weekly  # noqa: E402

AUTHORIZED_HEAD = "a" * 40
EVIDENCE_COMMIT = "b" * 40


def _authorization() -> dict[str, Any]:
    return {
        "git": {
            "branch": first_week.REQUIRED_GIT_BRANCH,
            "head": AUTHORIZED_HEAD,
            "upstream": first_week.REQUIRED_GIT_UPSTREAM,
            "upstream_head": AUTHORIZED_HEAD,
            "remote_head": AUTHORIZED_HEAD,
            "remote_fetch_url": first_week.REQUIRED_GIT_REMOTE_URL,
            "remote_push_url": first_week.REQUIRED_GIT_REMOTE_URL,
            "remote_verified": True,
            "worktree_clean": True,
        },
    }


def _pair_paths(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    anchor = tmp_path / "scripts" / "anchors" / "anchor.json"
    sidecar = tmp_path / "scripts" / "authorizations" / "sidecar.json"
    anchor.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)
    anchor.write_bytes(b'{"anchor":true}\n')
    sidecar.write_bytes(b'{"authorization":true}\n')
    pair = {
        "anchor": {"path": str(anchor)},
        "authorization": {"path": str(sidecar)},
        "commit": None,
        "remote": None,
    }
    return anchor, sidecar, pair


def _actual_remote(remote_head: str) -> dict[str, str]:
    return {
        "branch": first_week.REQUIRED_GIT_BRANCH,
        "upstream": first_week.REQUIRED_GIT_UPSTREAM,
        "remote_fetch_url": first_week.REQUIRED_GIT_REMOTE_URL,
        "remote_push_url": first_week.REQUIRED_GIT_REMOTE_URL,
        "remote_head": remote_head,
    }


@pytest.mark.parametrize("current_sidecar_exists", [False, True])
def test_d2_recovery_prefix_inventory_allows_only_the_current_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    current_sidecar_exists: bool,
) -> None:
    root = tmp_path / "ledger"
    scripts_dir = tmp_path / "scripts"
    sidecar_dir = scripts_dir / first_week.AUTHORIZATION_SIDECAR_DIR_NAME
    sidecar_dir.mkdir(parents=True)
    monkeypatch.setattr(stage3, "SCRIPTS_DIR", scripts_dir)
    d1_hash = "1" * 64
    d2_hash = "2" * 64
    genesis = {
        "sequence": 0,
        "record_type": "genesis",
        "record_hash": "0" * 64,
    }
    d1 = {
        "sequence": 1,
        "record_type": "decision_freeze",
        "record_hash": d1_hash,
    }
    authorization_dir = root / "objects" / first_week.APPEND_AUTHORIZATION_CATEGORY
    authorization_dir.mkdir(parents=True)
    for record_hash in (d1_hash, d2_hash):
        raw = first_week.canonical_json({"expected_record_hash": record_hash})
        digest = first_week.sha256_bytes(raw)
        (authorization_dir / f"{digest}.json").write_bytes(raw)
    d1_name = f"000001_{d1_hash}.json"
    d2_name = f"000002_{d2_hash}.json"
    (sidecar_dir / d1_name).write_text("{}\n", encoding="utf-8")
    if current_sidecar_exists:
        (sidecar_dir / d2_name).write_text("{}\n", encoding="utf-8")

    first_week.validate_authorization_sidecar_inventory(
        root,
        [genesis, d1],
        ignored_sidecar_names={d2_name} if current_sidecar_exists else None,
    )

    if current_sidecar_exists:
        with pytest.raises(first_week.FirstWeekOperationError, match="extra"):
            first_week.validate_authorization_sidecar_inventory(
                root,
                [genesis, d1],
            )


@pytest.mark.parametrize("verify_actual_remote", [False, True])
def test_global_pair_chain_runs_without_labels_and_forwards_remote_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verify_actual_remote: bool,
) -> None:
    records = [SimpleNamespace(data={"record_type": "genesis"})]
    calls: list[dict[str, Any]] = []

    def validator(
        spec: dict[str, Any],
        root: Path,
        actual_records: list[SimpleNamespace],
        *,
        require_pushed: bool,
        verify_actual_remote: bool,
        ignored_decision_sidecar_names: set[str] | None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "spec": spec,
                "root": root,
                "records": actual_records,
                "require_pushed": require_pushed,
                "verify_actual_remote": verify_actual_remote,
                "ignored_decision_sidecar_names": ignored_decision_sidecar_names,
            }
        )
        return [{"validated": True}]

    monkeypatch.setattr(weekly, "validate_stage3_operator_pair_chain", validator)
    spec = {"study_id": "test"}

    result = first_week.validate_global_weekly_label_evidence(
        spec,
        tmp_path,
        records,  # type: ignore[arg-type]
        verify_actual_remote=verify_actual_remote,
    )

    assert result == [{"validated": True}]
    assert calls == [
        {
            "spec": spec,
            "root": tmp_path,
            "records": records,
            "require_pushed": True,
            "verify_actual_remote": verify_actual_remote,
            "ignored_decision_sidecar_names": None,
        }
    ]


@pytest.mark.parametrize("verify_remote", [False, True])
def test_decision_preflight_forwards_its_remote_scope_to_pair_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verify_remote: bool,
) -> None:
    spec = copy.deepcopy(stage3.load_and_validate_spec())
    root = tmp_path / "ledger"
    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "decision preflight remote-scope test"},
        "2026-07-28T06:00:00Z",
    )
    seen: list[bool] = []
    monkeypatch.setattr(first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda *, verify_remote: {"remote_verified": verify_remote},
    )
    monkeypatch.setattr(
        first_week,
        "validate_existing_anchor_chain",
        lambda *_args, **_kwargs: [{"record_hash": genesis.data["record_hash"]}],
    )
    monkeypatch.setattr(
        first_week,
        "validate_prior_decision_evidence",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        first_week,
        "validate_due_label_evidence",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        first_week,
        "validate_global_weekly_label_evidence",
        lambda *_args, verify_actual_remote, **_kwargs: (
            seen.append(verify_actual_remote) or []
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_raw_ready",
        lambda *_args, **_kwargs: {"parquet_inventory_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        first_week,
        "validate_state_ready",
        lambda *_args, **_kwargs: {
            "raw_parquet_inventory_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_reference_ready",
        lambda *_args, **_kwargs: {
            "bridge_dates": list(first_week.FIRST_WEEK_REFERENCE_DATES),
            "prospective_week_count": 0,
        },
    )

    first_week.build_preflight_report(
        spec,
        root,
        tmp_path / "reference.json",
        tmp_path / "state.json",
        "2026-07-31",
        data_dir=tmp_path / "raw",
        verify_remote=verify_remote,
        generated_at=datetime(2026, 7, 31, 10, 0, tzinfo=UTC),
    )

    assert seen == [verify_remote]


@pytest.mark.parametrize(
    ("current_pair_pushed", "expected_state"),
    [
        (False, "DECISION_EVIDENCE_COMMIT_PENDING"),
        (True, "INVALID_DECISION_EVIDENCE_CHAIN"),
    ],
)
@pytest.mark.parametrize("auto_next", [False, True])
def test_status_audits_prefix_and_never_hides_a_broken_current_edge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    current_pair_pushed: bool,
    expected_state: str,
    auto_next: bool,
) -> None:
    root = tmp_path / "ledger"
    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "status local-prefix audit test"},
        "2026-07-28T06:00:00Z",
    )
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": "2026-08-03",
            "decision_path_sha256": "8" * 64,
            "reference_manifest_sha256": "9" * 64,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        first_week,
        "validate_authorization_sidecar_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(stage3, "status_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        first_week,
        "inspect_inventory",
        lambda _path: {
            "file_count": 1,
            "max_dt": "20260731",
            "content_inventory_sha256": "a" * 64,
        },
    )
    git_scopes: list[bool] = []
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda *, verify_remote: (
            git_scopes.append(verify_remote)
            or {
                "head": AUTHORIZED_HEAD,
                "worktree_clean": True,
            }
        ),
    )
    pair_audits: list[tuple[int, bool, set[str] | None]] = []

    def pair_chain(
        _spec: dict[str, Any],
        _root: Path,
        prefix: list[stage3.LedgerRecord],
        *,
        require_pushed: bool,
        verify_actual_remote: bool,
        ignored_decision_sidecar_names: set[str] | None,
    ) -> list[dict[str, Any]]:
        del require_pushed
        pair_audits.append(
            (
                len(prefix),
                verify_actual_remote,
                ignored_decision_sidecar_names,
            )
        )
        if len(prefix) == 2:
            reason = (
                "broken cross-event Git causality"
                if current_pair_pushed
                else "current pair is not committed"
            )
            raise first_week.FirstWeekOperationError(reason)
        return []

    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        pair_chain,
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                first_week.FirstWeekOperationError(
                    "broken cross-event Git causality"
                )
            )
            if current_pair_pushed
            else lambda *_args, **_kwargs: {
                "predecessor_pair_commit": AUTHORIZED_HEAD,
                "successor_authorized_git_head": AUTHORIZED_HEAD,
                "actual_remote_verified": False,
            }
        ),
    )
    monkeypatch.setattr(
        first_week,
        "load_append_authorization",
        lambda *_args, **_kwargs: ({"git": {"head": AUTHORIZED_HEAD}}, "b" * 64, authorization_path),
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        (
            (lambda *_args, **_kwargs: {"commit": EVIDENCE_COMMIT})
            if current_pair_pushed
            else lambda *_args, **_kwargs: (_ for _ in ()).throw(
                first_week.FirstWeekOperationError("current pair is not committed")
            )
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_actual_remote_branch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("status must not query the actual remote")
        ),
    )
    next_context = first_week.DecisionOperationContext(
        week_index=2,
        decision_count=1,
        decision_date="2026-08-07",
        entry_date="2026-08-10",
        exit_date="2026-08-17",
        reference_dates=("2026-08-07",),
        initial_decision_date="2026-07-31",
        initial_membership=(),
        previous_decision_record_hash=decision.data["record_hash"],
        expected_head=decision.data["record_hash"],
        ledger_id=genesis.data["ledger_id"],
        bootstrap=False,
    )
    monkeypatch.setattr(
        first_week,
        "derive_decision_operation_context",
        lambda *_args, **_kwargs: next_context,
    )

    result = first_week.status_snapshot(
        decision_date=None if auto_next else "2026-07-31",
        data_dir=tmp_path / "raw",
        now=datetime(2026, 7, 31, 10, 2, tzinfo=UTC),
    )

    assert result["state"] == expected_state
    assert result["global_label_evidence"]["valid"] is (not current_pair_pushed)
    assert result["global_label_evidence"]["complete_chain_valid"] is False
    assert (
        result["global_label_evidence"]["excluded_current_decision_hash"]
        == decision.data["record_hash"]
    )
    assert git_scopes == [False]
    assert pair_audits == [(2, False, None), (1, False, None)]


@pytest.mark.parametrize(
    ("weekly_label_state", "expected_state"),
    [
        ("LABEL_EVIDENCE_COMMIT_PENDING", "PRIOR_LABEL_EVIDENCE_PENDING"),
        ("ABORTED_INVALID", "INVALID_LABEL_EVIDENCE_CHAIN"),
    ],
)
def test_status_does_not_advance_past_current_label_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    weekly_label_state: str,
    expected_state: str,
) -> None:
    root = tmp_path / "ledger"
    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "status current-label blocker test"},
        "2026-07-28T06:00:00Z",
    )
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": "2026-08-03",
            "decision_path_sha256": "8" * 64,
            "reference_manifest_sha256": "9" * 64,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    stage3.append_record(
        root,
        "label_completion",
        f"label:{decision.data['record_hash']}",
        {
            "decision_record_hash": decision.data["record_hash"],
            "decision_dt": "2026-07-31",
            "entry_dt": "2026-08-03",
            "exit_dt": "2026-08-10",
            "raw_source_closure_sha256": "a" * 64,
            "label_observation_sha256": "b" * 64,
        },
        "2026-08-10T10:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    monkeypatch.setattr(first_week, "validate_frozen_data_dir", lambda path: path)
    monkeypatch.setattr(stage3, "resolve_ledger_root", lambda _spec: root)
    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        first_week,
        "validate_authorization_sidecar_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(stage3, "status_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        first_week,
        "inspect_inventory",
        lambda _path: {
            "file_count": 1,
            "max_dt": "20260810",
            "content_inventory_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        first_week,
        "validate_git_ready",
        lambda *, verify_remote: (
            {"head": AUTHORIZED_HEAD, "worktree_clean": True}
            if verify_remote is False
            else (_ for _ in ()).throw(AssertionError("status queried the remote"))
        ),
    )
    monkeypatch.setattr(
        weekly,
        "validate_stage3_operator_pair_chain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            first_week.FirstWeekOperationError("current label pair is pending")
        ),
    )
    monkeypatch.setattr(
        weekly,
        "status_snapshot",
        lambda **_kwargs: {"state": weekly_label_state},
    )
    monkeypatch.setattr(
        first_week,
        "validate_actual_remote_branch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("status must not query the actual remote")
        ),
    )
    next_context = first_week.DecisionOperationContext(
        week_index=2,
        decision_count=1,
        decision_date="2026-08-07",
        entry_date="2026-08-10",
        exit_date="2026-08-17",
        reference_dates=("2026-08-07",),
        initial_decision_date="2026-07-31",
        initial_membership=(),
        previous_decision_record_hash=decision.data["record_hash"],
        expected_head=stage3.scan_records(root)[-1].data["record_hash"],
        ledger_id=genesis.data["ledger_id"],
        bootstrap=False,
    )
    monkeypatch.setattr(
        first_week,
        "derive_decision_operation_context",
        lambda *_args, **_kwargs: next_context,
    )

    result = first_week.status_snapshot(
        decision_date=None,
        data_dir=tmp_path / "raw",
        now=datetime(2026, 8, 11, 0, 0, tzinfo=UTC),
    )

    assert result["state"] == expected_state
    assert result["global_label_evidence"]["complete_chain_valid"] is False


def test_exact_pair_commit_uses_direct_remote_even_when_tracking_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor, sidecar, pair = _pair_paths(tmp_path)
    remote_ref = "refs/heads/feat/surge-wave-strategy"
    calls: list[tuple[str, ...]] = []

    def git_output(_repo_root: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return EVIDENCE_COMMIT
        if args[0] == "status":
            return ""
        if args[0] == "symbolic-ref":
            return first_week.REQUIRED_GIT_BRANCH
        if args[:3] == ("rev-parse", "--abbrev-ref", "--symbolic-full-name"):
            return first_week.REQUIRED_GIT_UPSTREAM
        if args[:2] == ("remote", "get-url"):
            return first_week.REQUIRED_GIT_REMOTE_URL
        if args[0] == "ls-remote":
            return f"{EVIDENCE_COMMIT}\t{remote_ref}"
        if args[0] == "rev-list":
            return f"{EVIDENCE_COMMIT} {AUTHORIZED_HEAD}"
        if args[0] == "diff-tree":
            return "\n".join(
                (
                    anchor.relative_to(tmp_path).as_posix(),
                    sidecar.relative_to(tmp_path).as_posix(),
                )
            )
        if args[0] == "show":
            return "2026-08-02T00:00:00+00:00"
        raise AssertionError(args)

    monkeypatch.setattr(first_week, "_git_output", git_output)
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        lambda *_args, **_kwargs: pair,
    )
    monkeypatch.setattr(
        first_week,
        "_git_file_bytes_at_commit",
        lambda _root, _commit, relative: (tmp_path / relative).read_bytes(),
    )

    result = first_week.validate_recovery_evidence_commit(
        {},
        tmp_path / "ledger",
        {"payload": {"entry_dt": "2026-08-03"}},
        _authorization(),
        repo_root=tmp_path,
    )

    assert result["commit"] == EVIDENCE_COMMIT
    assert result["remote_contains_commit"] is True
    assert result["git"]["remote_head"] == EVIDENCE_COMMIT
    assert any(args[0] == "ls-remote" for args in calls)
    assert ("rev-parse", "@{upstream}") not in calls


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dirty", "completely clean worktree"),
        ("wrong_parent", "single-parent child"),
        ("merge_parent", "single-parent child"),
        ("extra_path", "exactly the matching anchor"),
        ("wrong_bytes", "bytes differ"),
    ],
)
def test_pair_commit_rejects_non_exact_recovery_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    anchor, sidecar, pair = _pair_paths(tmp_path)

    def git_output(_repo_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return EVIDENCE_COMMIT
        if args[0] == "status":
            return " M scripts/unrelated.py" if case == "dirty" else ""
        if args[0] == "rev-list":
            if case == "wrong_parent":
                return f"{EVIDENCE_COMMIT} {'c' * 40}"
            if case == "merge_parent":
                return f"{EVIDENCE_COMMIT} {AUTHORIZED_HEAD} {'c' * 40}"
            return f"{EVIDENCE_COMMIT} {AUTHORIZED_HEAD}"
        if args[0] == "diff-tree":
            paths = [
                anchor.relative_to(tmp_path).as_posix(),
                sidecar.relative_to(tmp_path).as_posix(),
            ]
            if case == "extra_path":
                paths.append("scripts/unrelated.py")
            return "\n".join(paths)
        if args[0] == "show":
            return "2026-08-02T00:00:00+00:00"
        raise AssertionError(args)

    monkeypatch.setattr(first_week, "_git_output", git_output)
    monkeypatch.setattr(
        first_week,
        "validate_actual_remote_branch",
        lambda **_kwargs: _actual_remote(AUTHORIZED_HEAD),
    )
    monkeypatch.setattr(
        first_week,
        "validate_authorized_anchor_pair",
        lambda *_args, **_kwargs: pair,
    )

    def committed_bytes(_root: Path, _commit: str, relative: str) -> bytes:
        if case == "wrong_bytes" and relative == anchor.relative_to(tmp_path).as_posix():
            return b"wrong\n"
        return (tmp_path / relative).read_bytes()

    monkeypatch.setattr(first_week, "_git_file_bytes_at_commit", committed_bytes)

    with pytest.raises(first_week.FirstWeekOperationError, match=message):
        first_week.validate_recovery_evidence_commit(
            {},
            tmp_path / "ledger",
            {"payload": {"entry_dt": "2026-08-03"}},
            _authorization(),
            repo_root=tmp_path,
        )


def _install_recovery_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    remote_contains_commit: bool,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    Path,
    Path,
    stage3.LedgerRecord,
    stage3.LedgerRecord,
]:
    root = tmp_path / "ledger"
    genesis = stage3.append_record(
        root,
        "genesis",
        "genesis:test",
        {"purpose": "decision evidence commit recovery test"},
        "2026-07-28T06:00:00Z",
    )
    reference_path = root / "objects" / "reference_manifest" / f"{'c' * 64}.json"
    preflight_path = root / "objects" / first_week.PREFLIGHT_CATEGORY / f"{'d' * 64}.json"
    state_path = tmp_path / "state.json"
    for path in (reference_path, preflight_path, state_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    decision = stage3.append_record(
        root,
        "decision_freeze",
        "decision:2026-07-31",
        {
            "decision_dt": "2026-07-31",
            "entry_dt": "2026-08-03",
            "decision_path_sha256": "e" * 64,
            "reference_manifest_sha256": reference_path.stem,
        },
        "2026-07-31T07:01:00Z",
        ledger_id=genesis.data["ledger_id"],
    )
    anchor_path = tmp_path / "anchor.json"
    sidecar_path = tmp_path / "sidecar.json"
    anchor_path.write_text('{"anchor":true}\n', encoding="utf-8")
    sidecar_path.write_text('{"authorization":true}\n', encoding="utf-8")
    authorization_path = (
        root
        / "objects"
        / first_week.APPEND_AUTHORIZATION_CATEGORY
        / f"{'f' * 64}.json"
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text("{}\n", encoding="utf-8")
    authorization = {
        **_authorization(),
        "state_manifest_path": str(state_path.resolve()),
        "reference_manifest_object": str(reference_path.relative_to(root)),
        "preflight_report_object": str(preflight_path.relative_to(root)),
    }

    monkeypatch.setattr(stage3, "validate_record_semantics", lambda *_args, **_kwargs: None)
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
        lambda *_args, **_kwargs: (authorization, "f" * 64, authorization_path),
    )
    monkeypatch.setattr(
        first_week,
        "validate_prior_decision_evidence",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        first_week,
        "validate_global_weekly_label_evidence",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_args, **_kwargs: {
            "predecessor_pair_commit": AUTHORIZED_HEAD,
            "successor_authorized_git_head": AUTHORIZED_HEAD,
            "actual_remote_verified": False,
        },
    )

    def pushed_pair(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("require_pushed"):
            raise first_week.FirstWeekOperationError("local tracking ref is stale")
        raise AssertionError("mutable pair validation must not run in the recovery route test")

    monkeypatch.setattr(first_week, "validate_authorized_anchor_pair", pushed_pair)
    monkeypatch.setattr(
        first_week,
        "_git_output",
        lambda _root, *args: (
            EVIDENCE_COMMIT
            if args == ("rev-parse", "HEAD")
            else (_ for _ in ()).throw(AssertionError(args))
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_evidence_commit",
        lambda *_args, **_kwargs: {
            "commit": EVIDENCE_COMMIT,
            "pair": {
                "anchor": {"path": str(anchor_path)},
                "authorization": {"path": str(sidecar_path)},
            },
            "git": {
                **_actual_remote(EVIDENCE_COMMIT if remote_contains_commit else AUTHORIZED_HEAD),
                "head": EVIDENCE_COMMIT,
                "worktree_clean": True,
                "remote_contains_commit": remote_contains_commit,
            },
            "remote_contains_commit": remote_contains_commit,
        },
    )
    return (
        authorization,
        root,
        reference_path,
        state_path,
        preflight_path,
        genesis,
        decision,
    )


def test_recovery_accepts_remote_pair_after_deadline_with_stale_tracking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _authorization_payload,
        root,
        reference_path,
        state_path,
        preflight_path,
        genesis,
        decision,
    ) = _install_recovery_route(
        monkeypatch,
        tmp_path,
        remote_contains_commit=True,
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an already remote-contained commit must be idempotent after entry")
        ),
    )

    result = first_week.recover_first_decision_anchor(
        spec=copy.deepcopy(stage3.load_and_validate_spec()),
        root=root,
        record_hash=decision.data["record_hash"],
        decision_date="2026-07-31",
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=preflight_path,
        expected_head=genesis.data["record_hash"],
        data_dir=tmp_path / "raw",
        now=datetime(2026, 8, 3, 2, 0, tzinfo=UTC),
    )

    assert result["status"] == "DECISION_EVIDENCE_ALREADY_COMMITTED"
    assert result["evidence_commit"] == EVIDENCE_COMMIT
    assert result["formal_ledger_mutated"] is False


def test_recovery_rejects_a_broken_successor_edge_before_any_commit_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _authorization_payload,
        root,
        reference_path,
        state_path,
        preflight_path,
        genesis,
        decision,
    ) = _install_recovery_route(
        monkeypatch,
        tmp_path,
        remote_contains_commit=True,
    )
    monkeypatch.setattr(
        weekly,
        "validate_successor_authorized_head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            first_week.FirstWeekOperationError("broken successor authorization edge")
        ),
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("broken causality reached a commit/deadline branch")
        ),
    )

    with pytest.raises(
        first_week.FirstWeekOperationError,
        match="broken successor authorization edge",
    ):
        first_week.recover_first_decision_anchor(
            spec=copy.deepcopy(stage3.load_and_validate_spec()),
            root=root,
            record_hash=decision.data["record_hash"],
            decision_date="2026-07-31",
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            preflight_report_path=preflight_path,
            expected_head=genesis.data["record_hash"],
            data_dir=tmp_path / "raw",
            now=datetime(2026, 8, 3, 2, 0, tzinfo=UTC),
        )


def test_recovery_waits_for_push_without_rewriting_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _authorization_payload,
        root,
        reference_path,
        state_path,
        preflight_path,
        genesis,
        decision,
    ) = _install_recovery_route(
        monkeypatch,
        tmp_path,
        remote_contains_commit=False,
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_deadline",
        lambda *_args, **_kwargs: {"checked_at_utc": "before-entry"},
    )

    def must_not_rewrite(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the existing evidence pair must not be rewritten")

    monkeypatch.setattr(first_week, "validate_raw_ready", must_not_rewrite)
    monkeypatch.setattr(first_week, "export_authorization_sidecar", must_not_rewrite)
    monkeypatch.setattr(stage3, "export_ledger_head_anchor", must_not_rewrite)

    result = first_week.recover_first_decision_anchor(
        spec=copy.deepcopy(stage3.load_and_validate_spec()),
        root=root,
        record_hash=decision.data["record_hash"],
        decision_date="2026-07-31",
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=preflight_path,
        expected_head=genesis.data["record_hash"],
        data_dir=tmp_path / "raw",
        now=datetime(2026, 8, 3, 1, 0, tzinfo=UTC),
    )

    assert result["status"] == "EVIDENCE_COMMIT_AWAITING_PUSH"
    assert result["next_state"] == "DECISION_EVIDENCE_PUSH_PENDING"
    assert result["formal_ledger_mutated"] is False


def test_recovery_fails_when_unpushed_pair_misses_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _authorization_payload,
        root,
        reference_path,
        state_path,
        preflight_path,
        genesis,
        decision,
    ) = _install_recovery_route(
        monkeypatch,
        tmp_path,
        remote_contains_commit=False,
    )
    monkeypatch.setattr(
        first_week,
        "validate_recovery_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            first_week.FirstWeekOperationError("before entry open")
        ),
    )

    with pytest.raises(first_week.FirstWeekOperationError, match="missed its push deadline"):
        first_week.recover_first_decision_anchor(
            spec=copy.deepcopy(stage3.load_and_validate_spec()),
            root=root,
            record_hash=decision.data["record_hash"],
            decision_date="2026-07-31",
            reference_manifest_path=reference_path,
            state_manifest_path=state_path,
            preflight_report_path=preflight_path,
            expected_head=genesis.data["record_hash"],
            data_dir=tmp_path / "raw",
            now=datetime(2026, 8, 3, 2, 0, tzinfo=UTC),
        )
