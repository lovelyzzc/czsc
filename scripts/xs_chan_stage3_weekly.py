"""Fail-closed weekly operator for Stage 3 prospective label completion.

The frozen collector deliberately accepts labels in any causal order after
their exit close.  This operational wrapper adds the stricter prospective
contract: only the oldest unlabeled decision may be completed, and every
decision scheduled on or before that label's exit must already be durably
authorized, anchored, committed, and pushed.

Read-only commands never write the formal ledger.  A label mutation requires
both ``--apply-ledger`` and the exact current head.  Raw/state preparation is
separately gated by ``--apply-data``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import xs_chan_exploration_stage3 as stage3
import xs_chan_stage3_first_week as first_week
import xs_chan_state_cache as state_cache
from _sync_daily_data import (
    DEFAULT_DATA_DIR,
    DEFAULT_SNAPSHOT_ROOT,
    MARKET_TIMEZONE,
    cache_lock,
    canonical_json,
    inspect_inventory,
    sha256_bytes,
    sha256_file,
)

LABEL_RELEASE_HOUR = 18
LABEL_PREFLIGHT_CATEGORY = "weekly_label_preflight"
LABEL_AUTHORIZATION_CATEGORY = "weekly_label_append_authorization"
LABEL_AUTHORIZATION_SCHEMA = "xs_chan_stage3_weekly_label_append_authorization_v1"
LABEL_SIDECAR_SCHEMA = "xs_chan_stage3_weekly_label_operator_authorization_v1"
LABEL_SIDECAR_DIR_NAME = "xs_chan_exploration_stage3_weekly_authorizations"
DEFAULT_STATE_OUTPUT_ROOT = first_week.DEFAULT_STATE_OUTPUT_ROOT
OPERATOR_SOURCE_PATH = Path(__file__).resolve()
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LABEL_AUTHORIZATION_KEYS = {
    "schema",
    "study_id",
    "study_identity",
    "spec_physical_sha256",
    "collector_source_sha256",
    "operator_source_path",
    "operator_source_sha256",
    "authorization_created_at_utc",
    "anticipated_record",
    "expected_record_hash",
    "expected_head",
    "record_type",
    "logical_event_key",
    "decision_record_hash",
    "week_index",
    "decision_date",
    "entry_date",
    "exit_date",
    "state_manifest_path",
    "state_manifest_sha256",
    "raw_audit_sha256",
    "raw_parquet_inventory_sha256",
    "raw_source_closure_object",
    "raw_source_closure_sha256",
    "label_observation_object",
    "label_observation_sha256",
    "preflight_report_object",
    "preflight_report_sha256",
    "git",
    "time_gate",
}
_LABEL_VALIDATION_SESSION: ContextVar[dict[str, dict[object, object]] | None] = (
    ContextVar("stage3_weekly_label_validation_session", default=None)
)
LABEL_PREFLIGHT_KEYS = {
    "schema",
    "mode",
    "generated_at_utc",
    "study_id",
    "study_identity",
    "spec_physical_sha256",
    "collector_source_sha256",
    "operator_source_sha256",
    "target",
    "global_decision_priority",
    "time_gate",
    "git",
    "required_decision_evidence",
    "existing_label_evidence",
    "exit_evidence",
    "state_manifest_path",
    "state_manifest_sha256",
    "ledger_before",
    "ledger_after",
    "expected_head",
    "formal_ledger_mutated",
    "event_payload_emitted",
    "efficacy_output",
}


class WeeklyOperationError(first_week.FirstWeekOperationError):
    """A weekly operational invariant failed closed."""


class StudyAbortedInvalid(WeeklyOperationError):
    """A mandatory decision window or authorization invariant was missed."""


class RequiredDecisionPending(WeeklyOperationError):
    """A calendar-priority decision must be frozen before this label."""

    def __init__(self, week_index: int, decision_date: str, entry_date: str) -> None:
        self.week_index = week_index
        self.decision_date = decision_date
        self.entry_date = entry_date
        super().__init__(
            f"decision week {week_index} ({decision_date}) has priority over the label"
        )


@dataclass(frozen=True)
class LabelTarget:
    """The uniquely derived oldest-unlabeled prospective decision."""

    week_index: int
    decision_record: Mapping[str, Any]
    required_later_decisions: tuple[Mapping[str, Any], ...]

    @property
    def decision_date(self) -> str:
        return str(self.decision_record["payload"]["decision_dt"])

    @property
    def entry_date(self) -> str:
        return str(self.decision_record["payload"]["entry_dt"])

    @property
    def exit_date(self) -> str:
        return str(self.decision_record["payload"]["exit_dt"])

    def summary(self) -> dict[str, Any]:
        return {
            "week_index": self.week_index,
            "decision_record_hash": self.decision_record["record_hash"],
            "decision_date": self.decision_date,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "required_later_decisions": [
                {
                    "week_index": int(row["payload"]["week_index"]),
                    "decision_date": str(row["payload"]["decision_dt"]),
                    "record_hash": str(row["record_hash"]),
                }
                for row in self.required_later_decisions
            ],
        }


@dataclass(frozen=True)
class LabelEvidence:
    """Exact exit-date inputs and the deterministic collector payload."""

    raw: Mapping[str, Any]
    state: Mapping[str, Any]
    closure: Mapping[str, Any]
    closure_sha256: str
    snapshot: Mapping[str, Any]
    observation_sha256: str
    payload: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "raw": dict(self.raw),
            "state": dict(self.state),
            "raw_source_closure_sha256": self.closure_sha256,
            "coverage_max_dt": str(self.closure["coverage_max_dt"]),
            "label_observation_sha256": self.observation_sha256,
            "label_payload_sha256": stage3._payload_hash(self.payload),
            "proposal_observation_count": int(self.payload["event_count"]),
        }


def _as_utc(now: datetime | None = None) -> datetime:
    return first_week._as_utc(now)


def _iso_date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _label_release(exit_date: str) -> datetime:
    return datetime.combine(
        pd.Timestamp(exit_date).date(),
        time(LABEL_RELEASE_HOUR, 0),
        tzinfo=MARKET_TIMEZONE,
    )


def validate_label_release(
    exit_date: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Require the post-close publication delay, not merely the 15:00 close."""

    current = _as_utc(now)
    release = _label_release(exit_date).astimezone(UTC)
    if current < release:
        raise WeeklyOperationError(
            f"label completion is forbidden before the exact 18:00 release {release.isoformat()}"
        )
    return {
        "checked_at_utc": current,
        "exit_release_not_before_utc": release,
    }


def _record_rows(
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        record.data if isinstance(record, stage3.LedgerRecord) else record
        for record in records
    ]


def _decision_rows(
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in _record_rows(records)
        if row.get("record_type") == "decision_freeze"
        and row.get("payload", {}).get("classification") == "PROSPECTIVE_COUNTED"
    ]


def _label_rows(
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in _record_rows(records)
        if row.get("record_type") == "label_completion"
    ]


def _schedule_for_decision(
    spec: Mapping[str, Any],
    root: Path,
    decision: Mapping[str, Any],
) -> pd.DataFrame:
    """Replay the schedule only from the decision's bound content object."""

    digest = str(decision["payload"]["reference_manifest_sha256"])
    if not HEX64.fullmatch(digest):
        raise WeeklyOperationError("decision has an invalid reference-manifest digest")
    path = root / "objects" / "reference_manifest" / f"{digest}.json"
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise WeeklyOperationError("decision-bound reference manifest is missing or differs")
    _, official_sessions, _ = stage3.load_reference_manifest(root, path)
    schedule = stage3.build_forward_schedule(spec, official_sessions)
    target = schedule.loc[
        schedule["week_index"].eq(int(decision["payload"]["week_index"]))
    ]
    if len(target) != 1:
        raise WeeklyOperationError("decision-bound official calendar cannot replay its week")
    row = target.iloc[0]
    expected = (
        _iso_date(row["decision_dt"]),
        _iso_date(row["entry_dt"]),
        _iso_date(row["exit_dt"]),
    )
    actual = (
        str(decision["payload"]["decision_dt"]),
        str(decision["payload"]["entry_dt"]),
        str(decision["payload"]["exit_dt"]),
    )
    if expected != actual:
        raise WeeklyOperationError("decision dates differ from its bound official calendar")
    return schedule


def _next_missing_decision(
    spec: Mapping[str, Any],
    root: Path,
    decisions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Derive the next decision only from the newest bound official calendar."""

    if not decisions or len(decisions) >= int(spec["accrual"]["window_weeks"]):
        return None
    schedule = _schedule_for_decision(spec, root, decisions[-1])
    expected_index = len(decisions) + 1
    row = schedule.loc[schedule["week_index"].eq(expected_index)]
    if len(row) != 1:
        raise WeeklyOperationError(
            "newest decision-bound calendar cannot derive the next prospective decision"
        )
    value = row.iloc[0]
    return {
        "week_index": expected_index,
        "decision_dt": _iso_date(value["decision_dt"]),
        "entry_dt": _iso_date(value["entry_dt"]),
        "exit_dt": _iso_date(value["exit_dt"]),
    }


def validate_no_missed_decision_window(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Abort globally once the next unrecorded decision's entry open is reached."""

    decisions = _decision_rows(records)
    pending = _next_missing_decision(spec, root, decisions)
    current = _as_utc(now)
    if pending is None:
        return {
            "status": "NO_PENDING_DECISION_WINDOW",
            "checked_at_utc": current,
            "next_decision": None,
        }
    entry_open = first_week._entry_open(str(pending["entry_dt"])).astimezone(UTC)
    decision_close = first_week._decision_close(str(pending["decision_dt"])).astimezone(UTC)
    if current >= entry_open:
        raise StudyAbortedInvalid(
            "MISSED_DECISION_WINDOW: "
            f"week {pending['week_index']} entry opened at {entry_open.isoformat()}"
        )
    if current >= decision_close:
        raise RequiredDecisionPending(
            int(pending["week_index"]),
            str(pending["decision_dt"]),
            str(pending["entry_dt"]),
        )
    return {
        "status": "NEXT_DECISION_WINDOW_OPEN_OR_FUTURE",
        "checked_at_utc": current,
        "next_decision": dict(pending),
        "decision_close_utc": decision_close,
        "entry_open_utc": entry_open,
    }


def derive_oldest_unlabeled_target(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    *,
    decision_date_assertion: str | None = None,
    now: datetime | None = None,
) -> LabelTarget | None:
    """Derive one target and enforce prefix label order plus calendar priority."""

    rows = _record_rows(records)
    decisions = _decision_rows(records)
    labels = _label_rows(records)
    decision_by_hash = {str(row["record_hash"]): row for row in decisions}
    labeled_hashes: list[str] = []
    for label in labels:
        decision_hash = str(label.get("payload", {}).get("decision_record_hash", ""))
        if decision_hash not in decision_by_hash or decision_hash in labeled_hashes:
            raise StudyAbortedInvalid("label chain does not reference unique prospective decisions")
        labeled_hashes.append(decision_hash)
        validate_label_release(
            str(label["payload"]["exit_dt"]),
            now=stage3._parse_utc(str(label["recorded_at_utc"])),
        )
    expected_prefix = [str(row["record_hash"]) for row in decisions[: len(labels)]]
    if labeled_hashes != expected_prefix:
        raise StudyAbortedInvalid("labels are not the exact oldest-decision prefix")
    if len(labels) == len(decisions):
        return None

    decision_priority = validate_no_missed_decision_window(
        spec,
        root,
        records,
        now=now,
    )
    del decision_priority
    decision = decisions[len(labels)]
    schedule = _schedule_for_decision(spec, root, decision)
    exit_date = pd.Timestamp(decision["payload"]["exit_dt"])
    later_schedule = schedule.loc[
        schedule["week_index"].gt(int(decision["payload"]["week_index"]))
        & schedule["decision_dt"].le(exit_date)
    ]
    by_index = {int(row["payload"]["week_index"]): row for row in decisions}
    required: list[Mapping[str, Any]] = []
    for scheduled in later_schedule.itertuples(index=False):
        later = by_index.get(int(scheduled.week_index))
        if later is None or str(later["payload"]["decision_dt"]) != _iso_date(scheduled.decision_dt):
            entry_date = _iso_date(scheduled.entry_dt)
            entry_open = first_week._entry_open(entry_date).astimezone(UTC)
            if _as_utc(now) >= entry_open:
                raise StudyAbortedInvalid(
                    "MISSED_DECISION_WINDOW: "
                    f"week {int(scheduled.week_index)} entry opened at {entry_open.isoformat()}"
                )
            raise RequiredDecisionPending(
                int(scheduled.week_index),
                _iso_date(scheduled.decision_dt),
                entry_date,
            )
        required.append(later)
    target = LabelTarget(
        week_index=int(decision["payload"]["week_index"]),
        decision_record=decision,
        required_later_decisions=tuple(required),
    )
    if decision_date_assertion is not None:
        asserted = _iso_date(decision_date_assertion)
        if asserted != target.decision_date:
            raise WeeklyOperationError(
                f"CLI decision date is only an equality assertion; expected "
                f"{target.decision_date}, got {asserted}"
            )
    # A label must be later in the immutable chain than every calendar-priority
    # decision.  Existing rows are checked too, so a direct collector bypass is
    # visible on the next read-only audit.
    sequence_by_hash = {str(row["record_hash"]): int(row["sequence"]) for row in rows}
    for later in required:
        if sequence_by_hash[str(later["record_hash"])] <= sequence_by_hash[
            str(decision["record_hash"])
        ]:
            raise StudyAbortedInvalid("later decision sequence is not causally ordered")
    return target


def _tracked_record_file_name(record: Mapping[str, Any]) -> str:
    return f"{int(record['sequence']):06d}_{record['record_hash']}.json"


def _anchor_path(record: Mapping[str, Any]) -> Path:
    return (
        stage3.SCRIPTS_DIR
        / "xs_chan_exploration_stage3_ledger_anchors"
        / _tracked_record_file_name(record)
    )


def _label_sidecar_path(record: Mapping[str, Any]) -> Path:
    return stage3.SCRIPTS_DIR / LABEL_SIDECAR_DIR_NAME / _tracked_record_file_name(record)


def _safe_object_path(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str):
        raise WeeklyOperationError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WeeklyOperationError(f"{label} path is not root-relative")
    resolved_root = root.resolve()
    unresolved = resolved_root / candidate
    try:
        unresolved.relative_to(resolved_root)
    except ValueError as exc:
        raise WeeklyOperationError(f"{label} escapes the ledger root") from exc
    return unresolved


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise WeeklyOperationError(f"{label} is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise WeeklyOperationError(f"{label} contains a symlink component: {current}")


def _stable_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one immutable evidence file without following its final symlink."""

    _reject_symlink_components(path, label=label)
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise WeeklyOperationError(f"{label} is not a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise WeeklyOperationError(f"{label} changed before its no-follow read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.stat(path, follow_symlinks=False)
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(final):
        raise WeeklyOperationError(f"{label} changed during its no-follow read")
    return b"".join(chunks)


def _stable_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = _stable_file_bytes(path, label=label)
    return _canonical_json_object(raw, label=label)


def _canonical_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeeklyOperationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise WeeklyOperationError(f"{label} is not a canonical JSON object")
    return value


def _authorization_record_hash(authorization: Mapping[str, Any]) -> str:
    direct = authorization.get("expected_record_hash")
    anticipated = authorization.get("anticipated_record")
    nested = anticipated.get("record_hash") if isinstance(anticipated, Mapping) else None
    value = direct if isinstance(direct, str) else nested
    return str(value or "")


def _authorization_git(authorization: Mapping[str, Any]) -> Mapping[str, Any]:
    git = authorization.get("git")
    if not isinstance(git, Mapping):
        raise WeeklyOperationError("operator authorization has no Git evidence")
    required = {
        "branch": first_week.REQUIRED_GIT_BRANCH,
        "upstream": first_week.REQUIRED_GIT_UPSTREAM,
        "remote_fetch_url": first_week.REQUIRED_GIT_REMOTE_URL,
        "remote_push_url": first_week.REQUIRED_GIT_REMOTE_URL,
        "remote_verified": True,
        "worktree_clean": True,
    }
    if any(git.get(key) != value for key, value in required.items()):
        raise WeeklyOperationError("operator authorization Git destination is invalid")
    head = str(git.get("head", ""))
    if (
        not re.fullmatch(r"[0-9a-f]{40,64}", head)
        or git.get("upstream_head") != head
        or git.get("remote_head") != head
    ):
        raise WeeklyOperationError("operator authorization did not bind the actual remote head")
    return git


def _authorization_sidecar_directories() -> tuple[Path, ...]:
    names = {
        LABEL_SIDECAR_DIR_NAME,
        first_week.AUTHORIZATION_SIDECAR_DIR_NAME,
    }
    # Generalized decision operators may use a distinct tracked directory.
    for candidate in stage3.SCRIPTS_DIR.glob("xs_chan_exploration_stage3*authorizations"):
        names.add(candidate.name)
    return tuple(stage3.SCRIPTS_DIR / name for name in sorted(names))


def _load_any_authorization_sidecar(
    root: Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any], Path]:
    """Load one exact tracked sidecar for either a decision or weekly label."""

    name = _tracked_record_file_name(record)
    candidates: list[Path] = []
    for directory in _authorization_sidecar_directories():
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise WeeklyOperationError(f"authorization inventory is not a real directory: {directory}")
        path = directory / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise WeeklyOperationError(f"authorization sidecar is not a regular file: {path}")
            candidates.append(path)
    if len(candidates) != 1:
        raise WeeklyOperationError(
            f"record has {len(candidates)} tracked authorization sidecars; expected exactly one"
        )
    path = candidates[0]
    sidecar_raw = _stable_file_bytes(path, label="tracked authorization sidecar")
    try:
        sidecar = json.loads(sidecar_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeeklyOperationError("tracked authorization sidecar is invalid JSON") from exc
    if sidecar_raw != json.dumps(
        sidecar,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n":
        raise WeeklyOperationError("tracked authorization sidecar bytes are not canonical")
    if (
        sidecar.get("study_id") is None
        or sidecar.get("study_identity") != record["ledger_id"]
        or sidecar.get("record_hash") != record["record_hash"]
        or sidecar.get("record_type") != record["record_type"]
        or sidecar.get("sequence") != record["sequence"]
        or sidecar.get("previous_hash") != record["previous_hash"]
    ):
        raise WeeklyOperationError("tracked authorization sidecar identifies another record")
    auth_sha = str(sidecar.get("authorization_sha256", ""))
    auth_path = _safe_object_path(
        root,
        sidecar.get("authorization_object"),
        label="authorization object",
    )
    auth_raw = _stable_file_bytes(auth_path, label="authorization object")
    auth_object = _canonical_json_object(auth_raw, label="authorization object")
    embedded = sidecar.get("authorization")
    if record["record_type"] == "label_completion":
        if sidecar.get("schema") != LABEL_SIDECAR_SCHEMA or embedded is not None:
            raise WeeklyOperationError(
                "tracked label sidecar must be minimal and must not embed outcome authorization"
            )
        authorization: Mapping[str, Any] = auth_object
    else:
        if not isinstance(embedded, Mapping):
            raise WeeklyOperationError("decision sidecar does not embed its authorization")
        authorization = embedded
    if (
        not HEX64.fullmatch(auth_sha)
        or auth_path.stem != auth_sha
        or sha256_bytes(auth_raw) != auth_sha
        or canonical_json(auth_object) != canonical_json(authorization)
        or _authorization_record_hash(authorization) != record["record_hash"]
    ):
        raise WeeklyOperationError("authorization object, sidecar, and record are not identical")
    return dict(sidecar), authorization, path


def _actual_remote_contains_commit(commit: str, remote_head: str) -> None:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, remote_head),
        cwd=stage3.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WeeklyOperationError(
            f"evidence commit {commit} is not an ancestor of actual remote {remote_head}"
        )


def validate_record_authorized_anchor_pair(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
    current_git: Mapping[str, Any] | None = None,
    verify_actual_remote: bool = True,
    decision_inventory_records: Sequence[
        stage3.LedgerRecord | Mapping[str, Any]
    ]
    | None = None,
    ignored_decision_sidecar_names: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a single-parent, exact-two-file authorization/anchor commit."""

    if record["record_type"] == "decision_freeze":
        pair = first_week.validate_authorized_anchor_pair(
            spec,
            root,
            record,
            require_pushed=require_pushed,
            inventory_records=decision_inventory_records,
            ignored_sidecar_names=ignored_decision_sidecar_names,
        )
        authorization, authorization_sha, authorization_path = first_week.load_append_authorization(
            spec,
            root,
            record,
            require_current_operator=False,
        )
        git = _authorization_git(authorization)
        commit = pair["commit"]
        if require_pushed and verify_actual_remote:
            actual_git = (
                dict(current_git)
                if current_git is not None
                else first_week.validate_actual_remote_branch()
            )
            _actual_remote_contains_commit(str(commit), str(actual_git["remote_head"]))
        return {
            "record_hash": record["record_hash"],
            "record_type": record["record_type"],
            "anchor": pair["anchor"],
            "authorization_sha256": authorization_sha,
            "authorization_path": str(authorization_path),
            "commit": commit,
            "authorized_git_head": str(git["head"]),
        }
    if record["record_type"] != "label_completion":
        raise WeeklyOperationError(
            f"unsupported operator-authorization record type: {record['record_type']}"
        )
    # The label-specific loader below fully replays schema, operator, state,
    # closure, observation, and anticipated record.  The generic sidecar read
    # alone is deliberately insufficient.
    load_label_append_authorization(
        spec,
        root,
        record,
        require_current_operator=False,
    )
    expected_sidecar, expected_sidecar_path = _expected_label_sidecar(
        spec,
        root,
        record,
        require_current_operator=False,
    )
    expected_sidecar_raw = json.dumps(
        expected_sidecar,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if _stable_file_bytes(
        expected_sidecar_path,
        label="tracked label authorization sidecar",
    ) != expected_sidecar_raw:
        raise WeeklyOperationError(
            "tracked label authorization sidecar differs from its exact minimal payload"
        )
    sidecar, authorization, sidecar_path = _load_any_authorization_sidecar(root, record)
    if sidecar_path != expected_sidecar_path:
        raise WeeklyOperationError("label authorization sidecar is stored at a non-canonical path")
    git = _authorization_git(authorization)
    anchor = first_week.validate_anchor_for_record(
        spec,
        root,
        record,
        require_pushed=require_pushed,
    )
    sidecar_commit: str | None = None
    if require_pushed:
        try:
            sidecar_commit = stage3._validate_tracked_file_pushed(sidecar_path)
        except stage3.Stage3ValidationError as exc:
            raise WeeklyOperationError(
                f"authorization sidecar is not committed and pushed: {exc}"
            ) from exc
        if sidecar_commit != anchor["commit"]:
            raise WeeklyOperationError("authorization sidecar and head anchor are split across commits")
        commit_line = first_week._git_output(
            stage3.REPO_ROOT,
            "rev-list",
            "--parents",
            "-n",
            "1",
            sidecar_commit,
        ).split()
        if len(commit_line) != 2 or commit_line[1] != git["head"]:
            raise WeeklyOperationError(
                "evidence commit must have exactly the authorized Git head as parent"
            )
        expected_paths = {
            _anchor_path(record).resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
            sidecar_path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
        }
        actual_paths = set(
            filter(
                None,
                first_week._git_output(
                    stage3.REPO_ROOT,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    sidecar_commit,
                ).splitlines(),
            )
        )
        if actual_paths != expected_paths:
            raise WeeklyOperationError(
                "evidence commit must change exactly one anchor and its authorization sidecar"
            )
        if record["record_type"] == "decision_freeze":
            deadline = first_week._entry_open(str(record["payload"]["entry_dt"]))
            committed_at = datetime.fromisoformat(
                first_week._git_output(
                    stage3.REPO_ROOT,
                    "show",
                    "-s",
                    "--format=%cI",
                    sidecar_commit,
                )
            )
            if committed_at >= deadline:
                raise WeeklyOperationError("decision authorization pair was pushed after entry open")
        if verify_actual_remote:
            actual_git = (
                dict(current_git)
                if current_git is not None
                else first_week.validate_actual_remote_branch()
            )
            _actual_remote_contains_commit(sidecar_commit, str(actual_git["remote_head"]))
    return {
        "record_hash": record["record_hash"],
        "record_type": record["record_type"],
        "anchor": anchor,
        "authorization_sidecar_path": str(sidecar_path),
        "authorization_sidecar_sha256": sha256_file(sidecar_path),
        "authorization_sha256": str(sidecar["authorization_sha256"]),
        "authorized_git_head": str(git["head"]),
        "commit": sidecar_commit,
    }


def validate_required_decision_pairs(
    spec: Mapping[str, Any],
    root: Path,
    target: LabelTarget,
    *,
    current_git: Mapping[str, Any] | None = None,
    verify_actual_remote: bool = True,
) -> list[dict[str, Any]]:
    """Require the target and every calendar-priority later decision pair."""

    return [
        validate_record_authorized_anchor_pair(
            spec,
            root,
            decision,
            require_pushed=True,
            current_git=current_git,
            verify_actual_remote=verify_actual_remote,
        )
        for decision in (target.decision_record, *target.required_later_decisions)
    ]


def _require_git_ancestor(ancestor: str, descendant: str, *, label: str) -> None:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=stage3.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WeeklyOperationError(
            f"{label}: {ancestor} is not an ancestor of {descendant}"
        )


def validate_stage3_operator_pair_chain(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    *,
    require_pushed: bool,
    verify_actual_remote: bool = True,
    ignored_label_sidecar_names: set[str] | None = None,
    ignored_decision_sidecar_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate every D/L pair and prove cross-event Git causality."""

    rows = _record_rows(records)
    if not rows or rows[0]["record_type"] != "genesis":
        raise WeeklyOperationError("operator pair chain requires the genesis record")
    validate_label_chain(
        spec,
        root,
        records,
        require_pushed=require_pushed,
        verify_actual_remote=verify_actual_remote,
        ignored_sidecar_names=ignored_label_sidecar_names,
    )
    event_rows = [
        row
        for row in rows[1:]
        if row["record_type"] in {"decision_freeze", "label_completion"}
    ]
    if not event_rows:
        # Genesis-only callers already validate the genesis anchor through the
        # first-week anchor chain.  Cross-event causality begins at the first
        # D/L authorization, so there is no pair edge to prove yet.
        return []
    genesis_anchor = first_week.validate_anchor_for_record(
        spec,
        root,
        rows[0],
        require_pushed=require_pushed,
    )
    prior_pair_commit = genesis_anchor.get("commit")
    if require_pushed and not isinstance(prior_pair_commit, str):
        raise WeeklyOperationError("pushed genesis anchor has no evidence commit")
    evidence: list[dict[str, Any]] = []
    for row in event_rows:
        pair = validate_record_authorized_anchor_pair(
            spec,
            root,
            row,
            require_pushed=require_pushed,
            verify_actual_remote=verify_actual_remote,
            decision_inventory_records=records,
            ignored_decision_sidecar_names=ignored_decision_sidecar_names,
        )
        authorized_head = str(pair["authorized_git_head"])
        if isinstance(prior_pair_commit, str):
            _require_git_ancestor(
                prior_pair_commit,
                authorized_head,
                label=f"cross-event Git causality before {row['logical_event_key']}",
            )
        commit = pair.get("commit")
        if require_pushed and not isinstance(commit, str):
            raise WeeklyOperationError("pushed pair chain has no evidence commit")
        if isinstance(commit, str):
            prior_pair_commit = commit
        evidence.append(pair)
    return evidence


def validate_successor_authorized_head(
    spec: Mapping[str, Any],
    root: Path,
    prefix_records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    successor_record: Mapping[str, Any],
    *,
    prefix_pair_evidence: Sequence[Mapping[str, Any]] | None = None,
    require_pushed_prefix: bool = True,
) -> dict[str, Any]:
    """Prove the preceding D/L pair commit causally precedes a successor auth."""

    prefix_rows = _record_rows(prefix_records)
    if not prefix_rows or prefix_rows[0].get("record_type") != "genesis":
        raise WeeklyOperationError("successor authorization requires a genesis prefix")
    if (
        successor_record.get("record_type")
        not in {"decision_freeze", "label_completion"}
        or successor_record.get("previous_hash") != prefix_rows[-1].get("record_hash")
        or int(successor_record.get("sequence", -1))
        != int(prefix_rows[-1].get("sequence", -1)) + 1
        or successor_record.get("ledger_id") != prefix_rows[0].get("ledger_id")
    ):
        raise WeeklyOperationError(
            "successor authorization is not the exact next record after its prefix"
        )
    evidence = (
        list(prefix_pair_evidence)
        if prefix_pair_evidence is not None
        else validate_stage3_operator_pair_chain(
            spec,
            root,
            prefix_records,
            require_pushed=require_pushed_prefix,
            verify_actual_remote=False,
            ignored_label_sidecar_names=(
                {_tracked_record_file_name(successor_record)}
                if successor_record["record_type"] == "label_completion"
                and _label_sidecar_path(successor_record).is_file()
                else None
            ),
            ignored_decision_sidecar_names=(
                {_tracked_record_file_name(successor_record)}
                if successor_record["record_type"] == "decision_freeze"
                and first_week._authorization_sidecar_path(successor_record).is_file()
                else None
            ),
        )
    )
    event_rows = [
        row
        for row in prefix_rows[1:]
        if row["record_type"] in {"decision_freeze", "label_completion"}
    ]
    if (
        len(evidence) != len(event_rows)
        or any(
            item.get("record_hash") != row["record_hash"]
            or item.get("record_type") != row["record_type"]
            for item, row in zip(evidence, event_rows, strict=True)
        )
    ):
        raise WeeklyOperationError(
            "successor edge received evidence for a different operator-pair prefix"
        )
    if evidence:
        predecessor_commit = evidence[-1].get("commit")
    else:
        genesis_anchor = first_week.validate_anchor_for_record(
            spec,
            root,
            prefix_rows[0],
            require_pushed=require_pushed_prefix,
        )
        predecessor_commit = genesis_anchor.get("commit")
    if not isinstance(predecessor_commit, str):
        raise WeeklyOperationError(
            "successor edge has no committed preceding authorization pair"
        )
    if successor_record["record_type"] == "decision_freeze":
        authorization, authorization_sha, _ = first_week.load_append_authorization(
            spec,
            root,
            successor_record,
            require_current_operator=False,
        )
    else:
        authorization, authorization_sha, _ = load_label_append_authorization(
            spec,
            root,
            successor_record,
            require_current_operator=False,
        )
    authorized_head = str(_authorization_git(authorization)["head"])
    _require_git_ancestor(
        predecessor_commit,
        authorized_head,
        label=(
            "successor authorization does not descend from the preceding "
            f"operator pair before {successor_record['logical_event_key']}"
        ),
    )
    return {
        "predecessor_pair_commit": predecessor_commit,
        "successor_record_hash": successor_record["record_hash"],
        "successor_record_type": successor_record["record_type"],
        "successor_authorization_sha256": authorization_sha,
        "successor_authorized_git_head": authorized_head,
        "actual_remote_verified": False,
    }


def build_label_evidence(
    target: LabelTarget,
    state_manifest_path: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> LabelEvidence:
    """Read exact exit-date raw/state bytes and derive the collector payload."""

    data_dir = first_week.validate_frozen_data_dir(data_dir)
    state_path = state_manifest_path.expanduser().absolute()
    _stable_file_bytes(state_path, label="selected state manifest")
    raw = first_week.validate_raw_ready(target.exit_date, data_dir=data_dir)
    state = first_week.validate_state_ready(
        state_path,
        target.exit_date,
        data_dir=data_dir,
    )
    if state["raw_parquet_inventory_sha256"] != raw["parquet_inventory_sha256"]:
        raise WeeklyOperationError("raw and state evidence bind different exit closures")
    snapshot, closure = stage3.capture_label_observations(
        target.decision_record["payload"]["proposals"],
        target.entry_date,
        target.exit_date,
        state_manifest_path=state_path,
    )
    coverage = _iso_date(str(closure.get("coverage_max_dt", "")))
    if coverage != target.exit_date:
        raise WeeklyOperationError(
            f"label raw closure must end exactly at {target.exit_date}; got {coverage}"
        )
    if closure.get("state_manifest_sha256") != state["manifest_sha256"]:
        raise WeeklyOperationError("label raw closure differs from the selected state manifest")
    stage3.verify_raw_source_closure_current(closure)
    closure_sha = sha256_bytes(canonical_json(closure))
    closure_object = f"objects/raw_source_closure/{closure_sha}.json"
    complete_snapshot = {
        **snapshot,
        "raw_source_closure_sha256": closure_sha,
        "raw_source_closure_object": closure_object,
    }
    observation_sha = sha256_bytes(canonical_json(complete_snapshot))
    events = stage3.derive_label_events(
        target.decision_record["payload"]["proposals"],
        complete_snapshot,
    )
    payload = {
        "schema": "xs_chan_stage3_label_completion_v1",
        "decision_record_hash": target.decision_record["record_hash"],
        "decision_dt": target.decision_date,
        "entry_dt": target.entry_date,
        "exit_dt": target.exit_date,
        "raw_source_closure_sha256": closure_sha,
        "label_observation_sha256": observation_sha,
        "label_observation_object": f"objects/label_observation/{observation_sha}.json",
        "events": events,
        "event_count": len(events),
    }
    return LabelEvidence(
        raw=raw,
        state=state,
        closure=closure,
        closure_sha256=closure_sha,
        snapshot=complete_snapshot,
        observation_sha256=observation_sha,
        payload=payload,
    )


def build_label_preflight(
    spec: Mapping[str, Any],
    root: Path,
    state_manifest_path: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    decision_date_assertion: str | None = None,
    verify_remote: bool,
    now: datetime | None = None,
) -> tuple[dict[str, Any], LabelTarget, LabelEvidence]:
    """Run all label gates without writing objects or ledger records."""

    before = first_week.ledger_closure(root)
    records = stage3.scan_records(root)
    stage3.validate_record_semantics(spec, records, root=root)
    operator_pair_chain = validate_stage3_operator_pair_chain(
        spec,
        root,
        records,
        require_pushed=True,
        verify_actual_remote=verify_remote,
    )
    global_priority = validate_no_missed_decision_window(
        spec,
        root,
        records,
        now=now,
    )
    target = derive_oldest_unlabeled_target(
        spec,
        root,
        records,
        decision_date_assertion=decision_date_assertion,
        now=now,
    )
    if target is None:
        raise WeeklyOperationError("there is no recorded unlabeled prospective decision")
    time_gate = validate_label_release(target.exit_date, now=now)
    git = first_week.validate_git_ready(verify_remote=verify_remote)
    decision_evidence = [
        item for item in operator_pair_chain if item["record_type"] == "decision_freeze"
    ]
    existing_label_evidence = [
        item for item in operator_pair_chain if item["record_type"] == "label_completion"
    ]
    evidence = build_label_evidence(
        target,
        state_manifest_path,
        data_dir=data_dir,
    )
    after = first_week.assert_ledger_unchanged(before, root)
    report = {
        "schema": "xs_chan_stage3_weekly_label_preflight_v1",
        "mode": "READ_ONLY_OLDEST_LABEL",
        "generated_at_utc": _as_utc(now),
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "spec_physical_sha256": sha256_file(stage3.SPEC_PATH),
        "collector_source_sha256": sha256_file(stage3.SOURCE_PATH),
        "operator_source_sha256": sha256_file(OPERATOR_SOURCE_PATH),
        "target": target.summary(),
        "global_decision_priority": global_priority,
        "time_gate": time_gate,
        "git": git,
        "required_decision_evidence": decision_evidence,
        "existing_label_evidence": existing_label_evidence,
        "exit_evidence": evidence.summary(),
        "state_manifest_path": str(state_manifest_path.expanduser().absolute()),
        "state_manifest_sha256": sha256_bytes(
            _stable_file_bytes(
                state_manifest_path.expanduser().absolute(),
                label="selected state manifest",
            )
        ),
        "ledger_before": before,
        "ledger_after": after,
        "expected_head": before["head"],
        "formal_ledger_mutated": False,
        "event_payload_emitted": False,
        "efficacy_output": "FORBIDDEN",
    }
    return report, target, evidence


def prepare_label_data(
    *,
    decision_date_assertion: str | None,
    data_dir: Path,
    snapshot_root: Path,
    state_output_root: Path,
    workers: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Advance raw/state to exactly the oldest label exit, never the ledger."""

    spec = stage3.load_and_validate_spec()
    root = stage3.resolve_ledger_root(spec)
    data_dir = first_week.validate_frozen_data_dir(data_dir)
    records = stage3.scan_records(root)
    stage3.validate_record_semantics(spec, records, root=root)
    validate_no_missed_decision_window(spec, root, records, now=now)
    target = derive_oldest_unlabeled_target(
        spec,
        root,
        records,
        decision_date_assertion=decision_date_assertion,
        now=now,
    )
    if target is None:
        raise WeeklyOperationError("there is no recorded unlabeled prospective decision")
    release_gate = validate_label_release(target.exit_date, now=now)
    first_week.validate_git_ready(verify_remote=True)
    with first_week.operator_lock(root):
        before = first_week.ledger_closure(root)
        locked_records = stage3.scan_records(root)
        locked_target = derive_oldest_unlabeled_target(
            spec,
            root,
            locked_records,
            decision_date_assertion=target.decision_date,
            now=now,
        )
        if locked_target is None or locked_target.summary() != target.summary():
            raise WeeklyOperationError("oldest label target changed before data preparation")
        first_week.validate_existing_anchor_chain(
            spec,
            root,
            require_pushed=True,
        )
        validate_stage3_operator_pair_chain(
            spec,
            root,
            locked_records,
            require_pushed=True,
            verify_actual_remote=True,
        )
        raw_result = first_week.run_sync(
            argparse.Namespace(
                data_dir=data_dir,
                snapshot_root=snapshot_root,
                end_date=pd.Timestamp(target.exit_date).strftime("%Y%m%d"),
                min_daily_rows=first_week.DEFAULT_MIN_DAILY_ROWS,
                new_symbol_sleep_seconds=first_week.DEFAULT_NEW_SYMBOL_SLEEP_SECONDS,
                apply=True,
            )
        )
        expected_end = pd.Timestamp(target.exit_date).strftime("%Y%m%d")
        if (
            raw_result.get("status") not in {"PUBLISHED", "ALREADY_CURRENT"}
            or raw_result.get("safe_end_date") != expected_end
        ):
            raise WeeklyOperationError(
                f"raw sync did not close exactly on {target.exit_date}: {raw_result}"
            )
        with cache_lock(data_dir):
            first_week.validate_raw_ready(target.exit_date, data_dir=data_dir)
            state_manifest = first_week.build_or_reuse_state_cache(
                data_dir,
                state_output_root,
                workers=workers,
            )
            report, _, evidence = build_label_preflight(
                spec,
                root,
                state_manifest,
                data_dir=data_dir,
                decision_date_assertion=target.decision_date,
                verify_remote=True,
                now=now,
            )
        first_week.assert_ledger_unchanged(before, root)
    return {
        "status": "LABEL_DATA_PREPARED_NEVER_APPENDED",
        "target": target.summary(),
        "state_manifest_path": str(state_manifest),
        "state_manifest_sha256": evidence.state["manifest_sha256"],
        "raw_result": raw_result,
        "preflight": report,
        "release_gate": release_gate,
        "formal_ledger_mutated": False,
    }


def _operator_relative_path() -> str:
    try:
        return OPERATOR_SOURCE_PATH.relative_to(stage3.REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise WeeklyOperationError("weekly operator is not below the bound Git repository") from exc


def _validate_operator_git_binding(
    authorization: Mapping[str, Any],
    *,
    require_current_operator: bool,
) -> None:
    git = _authorization_git(authorization)
    relative = str(authorization.get("operator_source_path", ""))
    if relative != _operator_relative_path():
        raise WeeklyOperationError("authorization operator path differs from this weekly operator")
    authorized_sha = str(authorization.get("operator_source_sha256", ""))
    if not HEX64.fullmatch(authorized_sha):
        raise WeeklyOperationError("authorization operator source hash is invalid")
    git_sha = first_week._git_file_sha256_at_commit(
        stage3.REPO_ROOT,
        str(git["head"]),
        relative,
    )
    if git_sha != authorized_sha:
        raise WeeklyOperationError("authorized Git commit contains different operator bytes")
    if require_current_operator and sha256_file(OPERATOR_SOURCE_PATH) != authorized_sha:
        raise WeeklyOperationError("current weekly operator differs from the pre-record authorization")


def _validate_historical_failure_inventory(
    root: Path,
    ledger_before: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay only the failure files bound by a historical ledger receipt."""

    historical = ledger_before.get("failure_files")
    if not isinstance(historical, list):
        raise WeeklyOperationError(
            "label preflight ledger closure has no historical failure inventory"
        )
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in historical:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "size", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or int(item["size"]) < 0
            or not HEX64.fullmatch(str(item.get("sha256", "")))
        ):
            raise WeeklyOperationError(
                "label preflight historical failure inventory is malformed"
            )
        relative = str(item["path"])
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or relative_path.parts[0] != "failures"
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or relative in seen
        ):
            raise WeeklyOperationError(
                "label preflight historical failure path is invalid"
            )
        seen.add(relative)
        path = root / relative_path
        raw = _stable_file_bytes(path, label="historical preserved failure")
        if len(raw) != int(item["size"]) or sha256_bytes(raw) != item["sha256"]:
            raise WeeklyOperationError(
                "historical preserved failure is missing or differs"
            )
        validated.append(
            {
                "path": relative,
                "size": int(item["size"]),
                "sha256": str(item["sha256"]),
            }
        )
    if [item["path"] for item in validated] != sorted(
        item["path"] for item in validated
    ):
        raise WeeklyOperationError(
            "label preflight historical failure inventory is not canonical"
        )
    return validated


def _validate_label_preflight_receipt_impl(
    spec: Mapping[str, Any],
    root: Path,
    report: Mapping[str, Any],
    *,
    target: LabelTarget,
    evidence: LabelEvidence,
    state_manifest_path: Path,
    expected_head: str,
    git: Mapping[str, Any],
    operator_source_sha256: str,
) -> None:
    """Strictly validate every field of the final non-counting label receipt."""

    state_path = state_manifest_path.expanduser().absolute()
    expected_time_gate = validate_label_release(
        target.exit_date,
        now=stage3._parse_utc(str(report.get("generated_at_utc"))),
    )
    before = report.get("ledger_before")
    after = report.get("ledger_after")
    required = report.get("required_decision_evidence")
    existing = report.get("existing_label_evidence")
    global_priority = report.get("global_decision_priority")
    ledger_objects = stage3.scan_records(root)
    ledger_rows = _record_rows(ledger_objects)
    head_positions = [
        index for index, row in enumerate(ledger_rows) if row["record_hash"] == expected_head
    ]
    if len(head_positions) != 1:
        raise WeeklyOperationError("preflight expected head is not in the current immutable chain")
    prefix = ledger_rows[: head_positions[0] + 1]
    expected_decisions = [
        row["record_hash"]
        for row in prefix
        if row["record_type"] == "decision_freeze"
    ]
    expected_labels = [
        row["record_hash"]
        for row in prefix
        if row["record_type"] == "label_completion"
    ]
    replayed_priority = validate_no_missed_decision_window(
        spec,
        root,
        prefix,
        now=stage3._parse_utc(str(report.get("generated_at_utc"))),
    )
    if canonical_json(first_week._json_safe(global_priority)) != canonical_json(
        first_week._json_safe(replayed_priority)
    ):
        raise WeeklyOperationError(
            "label preflight global decision priority does not replay"
        )
    decision_auth_shas: dict[str, str] = {}
    for row in prefix:
        if row["record_type"] != "decision_freeze":
            continue
        _, digest, _ = first_week.load_append_authorization(
            spec,
            root,
            row,
            require_current_operator=False,
        )
        decision_auth_shas[str(row["record_hash"])] = digest
    label_auth_shas = {
        _authorization_record_hash(value): path.stem
        for path, value in _authorization_objects(root)
        if _authorization_record_hash(value) in set(expected_labels)
    }
    if all(isinstance(item, stage3.LedgerRecord) for item in ledger_objects):
        prefix_objects = list(ledger_objects[: head_positions[0] + 1])
        tail_rows = ledger_rows[head_positions[0] + 1 :]
        ignored_label_sidecars = {
            _tracked_record_file_name(row)
            for row in tail_rows
            if row["record_type"] == "label_completion"
            and _label_sidecar_path(row).is_file()
        }
        ignored_decision_sidecars = {
            _tracked_record_file_name(row)
            for row in tail_rows
            if row["record_type"] == "decision_freeze"
            and first_week._authorization_sidecar_path(row).is_file()
        }
        actual_pair_evidence = validate_stage3_operator_pair_chain(
            spec,
            root,
            prefix_objects,
            require_pushed=True,
            verify_actual_remote=False,
            ignored_label_sidecar_names=ignored_label_sidecars or None,
            ignored_decision_sidecar_names=ignored_decision_sidecars or None,
        )
        actual_decision_evidence = [
            item
            for item in actual_pair_evidence
            if item["record_type"] == "decision_freeze"
        ]
        actual_label_evidence = [
            item
            for item in actual_pair_evidence
            if item["record_type"] == "label_completion"
        ]
        if (
            canonical_json(first_week._json_safe(required))
            != canonical_json(first_week._json_safe(actual_decision_evidence))
            or canonical_json(first_week._json_safe(existing))
            != canonical_json(first_week._json_safe(actual_label_evidence))
        ):
            raise WeeklyOperationError(
                "label preflight pair evidence is not the exact historical prefix"
            )
        record_files = [
            {
                "path": str(item.path.relative_to(root)),
                "size": item.path.stat().st_size,
                "sha256": sha256_file(item.path),
            }
            for item in prefix_objects
        ]
        if not isinstance(before, Mapping):
            raise WeeklyOperationError("label preflight ledger closure is malformed")
        failure_files = _validate_historical_failure_inventory(root, before)
        expected_closure = {
            "record_count": len(prefix_objects),
            "head": prefix_objects[-1].data["record_hash"],
            "head_type": prefix_objects[-1].data["record_type"],
            "record_files": record_files,
            "failure_files": failure_files,
            "closure_sha256": sha256_bytes(
                canonical_json(
                    {
                        "records": record_files,
                        "failures": failure_files,
                    }
                )
            ),
        }
        if canonical_json(before) != canonical_json(expected_closure):
            raise WeeklyOperationError(
                "label preflight ledger closure is not the exact expected-head prefix"
            )
    if (
        set(report) != LABEL_PREFLIGHT_KEYS
        or report.get("schema") != "xs_chan_stage3_weekly_label_preflight_v1"
        or report.get("mode") != "READ_ONLY_OLDEST_LABEL"
        or report.get("study_id") != spec["study_id"]
        or report.get("study_identity") != stage3.study_identity(spec)
        or report.get("spec_physical_sha256") != sha256_file(stage3.SPEC_PATH)
        or report.get("collector_source_sha256") != sha256_file(stage3.SOURCE_PATH)
        or report.get("operator_source_sha256") != operator_source_sha256
        or canonical_json(report.get("target")) != canonical_json(target.summary())
        or canonical_json(first_week._json_safe(report.get("time_gate")))
        != canonical_json(first_week._json_safe(expected_time_gate))
        or canonical_json(report.get("git")) != canonical_json(git)
        or canonical_json(report.get("exit_evidence"))
        != canonical_json(evidence.summary())
        or report.get("state_manifest_path") != str(state_path)
        or report.get("state_manifest_sha256")
        != sha256_bytes(_stable_file_bytes(state_path, label="preflight state manifest"))
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or canonical_json(before) != canonical_json(after)
        or before.get("head") != expected_head
        or report.get("expected_head") != expected_head
        or report.get("formal_ledger_mutated") is not False
        or report.get("event_payload_emitted") is not False
        or report.get("efficacy_output") != "FORBIDDEN"
        or not isinstance(required, list)
        or [item.get("record_hash") for item in required if isinstance(item, Mapping)]
        != expected_decisions
        or any(
            not isinstance(item, Mapping)
            or item.get("record_type") != "decision_freeze"
            or not HEX64.fullmatch(str(item.get("authorization_sha256", "")))
            or decision_auth_shas.get(str(item.get("record_hash")))
            != item.get("authorization_sha256")
            for item in required
        )
        or not isinstance(existing, list)
        or [item.get("record_hash") for item in existing if isinstance(item, Mapping)]
        != expected_labels
        or any(
            not isinstance(item, Mapping)
            or item.get("record_type") != "label_completion"
            or not HEX64.fullmatch(str(item.get("authorization_sha256", "")))
            or label_auth_shas.get(str(item.get("record_hash")))
            != item.get("authorization_sha256")
            for item in existing
        )
        or not isinstance(global_priority, Mapping)
        or global_priority.get("status")
        not in {"NEXT_DECISION_WINDOW_OPEN_OR_FUTURE", "NO_PENDING_DECISION_WINDOW"}
        or stage3._contains_outcome_key(report)
    ):
        raise WeeklyOperationError(
            "final label preflight receipt does not exactly bind the safe transition"
        )


def _validate_label_preflight_receipt(
    spec: Mapping[str, Any],
    root: Path,
    report: Mapping[str, Any],
    *,
    target: LabelTarget,
    evidence: LabelEvidence,
    state_manifest_path: Path,
    expected_head: str,
    git: Mapping[str, Any],
    operator_source_sha256: str,
) -> None:
    """Share validated historical label auths across nested prefix replay."""

    session = _LABEL_VALIDATION_SESSION.get()
    if session is not None:
        _validate_label_preflight_receipt_impl(
            spec,
            root,
            report,
            target=target,
            evidence=evidence,
            state_manifest_path=state_manifest_path,
            expected_head=expected_head,
            git=git,
            operator_source_sha256=operator_source_sha256,
        )
        return
    token = _LABEL_VALIDATION_SESSION.set({"authorization": {}})
    try:
        _validate_label_preflight_receipt_impl(
            spec,
            root,
            report,
            target=target,
            evidence=evidence,
            state_manifest_path=state_manifest_path,
            expected_head=expected_head,
            git=git,
            operator_source_sha256=operator_source_sha256,
        )
    finally:
        _LABEL_VALIDATION_SESSION.reset(token)


def _build_label_authorization(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    target: LabelTarget,
    evidence: LabelEvidence,
    state_manifest_path: Path,
    preflight_report: Mapping[str, Any],
    preflight_path: Path,
    git: Mapping[str, Any],
    operator_source_sha256: str | None = None,
    validate_preflight_receipt: bool = True,
) -> dict[str, Any]:
    """Build the complete append-before-record authorization payload."""

    if canonical_json(preflight_report.get("git")) != canonical_json(git):
        raise WeeklyOperationError("preflight and authorization Git evidence differ")
    operator_sha = operator_source_sha256 or sha256_file(OPERATOR_SOURCE_PATH)
    if validate_preflight_receipt:
        _validate_label_preflight_receipt(
            spec,
            root,
            preflight_report,
            target=target,
            evidence=evidence,
            state_manifest_path=state_manifest_path,
            expected_head=str(record["previous_hash"]),
            git=git,
            operator_source_sha256=operator_sha,
        )
    relative = _operator_relative_path()
    if first_week._git_file_sha256_at_commit(
        stage3.REPO_ROOT,
        str(git["head"]),
        relative,
    ) != operator_sha:
        raise WeeklyOperationError("formal Git HEAD does not contain the running weekly operator")
    closure_path = root / f"objects/raw_source_closure/{evidence.closure_sha256}.json"
    observation_path = root / f"objects/label_observation/{evidence.observation_sha256}.json"
    closure_raw = _stable_file_bytes(closure_path, label="collector raw closure")
    observation_raw = _stable_file_bytes(
        observation_path,
        label="collector label observation",
    )
    if (
        sha256_bytes(closure_raw) != evidence.closure_sha256
        or closure_raw != canonical_json(evidence.closure)
        or sha256_bytes(observation_raw) != evidence.observation_sha256
        or observation_raw != canonical_json(evidence.snapshot)
    ):
        raise WeeklyOperationError("collector observation objects differ before authorization")
    state_path = state_manifest_path.expanduser().absolute()
    state_raw = _stable_file_bytes(state_path, label="state manifest")
    return {
        "schema": LABEL_AUTHORIZATION_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "spec_physical_sha256": sha256_file(stage3.SPEC_PATH),
        "collector_source_sha256": sha256_file(stage3.SOURCE_PATH),
        "operator_source_path": relative,
        "operator_source_sha256": operator_sha,
        "authorization_created_at_utc": record["recorded_at_utc"],
        "anticipated_record": dict(record),
        "expected_record_hash": record["record_hash"],
        "expected_head": record["previous_hash"],
        "record_type": record["record_type"],
        "logical_event_key": record["logical_event_key"],
        "decision_record_hash": target.decision_record["record_hash"],
        "week_index": target.week_index,
        "decision_date": target.decision_date,
        "entry_date": target.entry_date,
        "exit_date": target.exit_date,
        "state_manifest_path": str(state_path),
        "state_manifest_sha256": sha256_bytes(state_raw),
        "raw_audit_sha256": evidence.raw["audit_sha256"],
        "raw_parquet_inventory_sha256": evidence.raw["parquet_inventory_sha256"],
        "raw_source_closure_object": str(closure_path.relative_to(root)),
        "raw_source_closure_sha256": evidence.closure_sha256,
        "label_observation_object": str(observation_path.relative_to(root)),
        "label_observation_sha256": evidence.observation_sha256,
        "preflight_report_object": str(preflight_path.relative_to(root)),
        "preflight_report_sha256": sha256_file(preflight_path),
        "git": dict(git),
        "time_gate": first_week._json_safe(
            validate_label_release(
                target.exit_date,
                now=stage3._parse_utc(str(record["recorded_at_utc"])),
            )
        ),
    }


def store_label_append_authorization(
    spec: Mapping[str, Any],
    root: Path,
    anticipated_record: Mapping[str, Any],
    *,
    target: LabelTarget,
    evidence: LabelEvidence,
    state_manifest_path: Path,
    preflight_report: Mapping[str, Any],
    fresh_git: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    """Store an exact intent while holding the same physical ledger lock."""

    resolved_root = root.expanduser().resolve()
    lease = first_week._active_authorization_lock_lease()
    if (
        lease is None
        or lease.root != resolved_root
        or not first_week._physical_authorization_lock_owned(resolved_root)
    ):
        raise WeeklyOperationError(
            "label authorization requires the guarded physical ledger lock"
        )
    records = stage3.scan_records(root)
    current_head = records[-1].data["record_hash"] if records else stage3.ZERO_HASH
    existing_hashes = {record.data["record_hash"] for record in records}
    existing_keys = {record.data["logical_event_key"] for record in records}
    if (
        anticipated_record.get("record_type") != "label_completion"
        or int(anticipated_record.get("sequence", -1)) != len(records)
        or anticipated_record.get("previous_hash") != current_head
        or anticipated_record.get("record_hash") in existing_hashes
        or anticipated_record.get("logical_event_key") in existing_keys
    ):
        raise WeeklyOperationError(
            "label authorization must be written against the exact current head "
            "before the record exists"
        )
    preflight_sha, preflight_path = stage3._store_canonical_object(
        root,
        LABEL_PREFLIGHT_CATEGORY,
        preflight_report,
    )
    if preflight_sha != sha256_file(preflight_path):
        raise WeeklyOperationError("label preflight is not exactly content-addressed")
    authorization = _build_label_authorization(
        spec,
        root,
        anticipated_record,
        target=target,
        evidence=evidence,
        state_manifest_path=state_manifest_path,
        preflight_report=preflight_report,
        preflight_path=preflight_path,
        git=fresh_git,
    )
    digest, path = stage3._store_canonical_object(
        root,
        LABEL_AUTHORIZATION_CATEGORY,
        authorization,
    )
    if digest != sha256_file(path):
        raise WeeklyOperationError("label authorization is not exactly content-addressed")
    return authorization, digest, path


def _authorization_objects(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    directory = root / "objects" / LABEL_AUTHORIZATION_CATEGORY
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise WeeklyOperationError("label authorization object inventory is not a real directory")
    values: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise WeeklyOperationError(f"unexpected label authorization object: {path}")
        raw = _stable_file_bytes(path, label="label authorization object")
        if not HEX64.fullmatch(path.stem) or sha256_bytes(raw) != path.stem:
            raise WeeklyOperationError(f"label authorization filename/hash mismatch: {path}")
        value = _canonical_json_object(raw, label="label authorization object")
        values.append((path, value))
    return values


def _load_label_append_authorization_impl(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_current_operator: bool,
) -> tuple[dict[str, Any], str, Path]:
    """Load and statically replay the unique pre-record label authorization."""

    matches = [
        (path, value)
        for path, value in _authorization_objects(root)
        if _authorization_record_hash(value) == record["record_hash"]
    ]
    if len(matches) != 1:
        raise WeeklyOperationError(
            "label has no unique append-before-record authorization; posthoc authorization is forbidden"
        )
    path, authorization = matches[0]
    if set(authorization) != LABEL_AUTHORIZATION_KEYS:
        raise WeeklyOperationError("label authorization schema keys differ")
    anticipated = authorization.get("anticipated_record")
    if (
        authorization.get("schema") != LABEL_AUTHORIZATION_SCHEMA
        or not isinstance(anticipated, Mapping)
        or canonical_json(anticipated) != canonical_json(record)
        or authorization.get("expected_record_hash") != record["record_hash"]
        or authorization.get("expected_head") != record["previous_hash"]
        or authorization.get("record_type") != "label_completion"
        or authorization.get("logical_event_key") != record["logical_event_key"]
        or authorization.get("study_id") != spec["study_id"]
        or authorization.get("study_identity") != stage3.study_identity(spec)
        or authorization.get("spec_physical_sha256") != sha256_file(stage3.SPEC_PATH)
        or authorization.get("collector_source_sha256") != sha256_file(stage3.SOURCE_PATH)
    ):
        raise WeeklyOperationError("label authorization does not exactly bind the record/study")
    _validate_operator_git_binding(
        authorization,
        require_current_operator=require_current_operator,
    )
    payload = record["payload"]
    if (
        authorization.get("decision_record_hash") != payload["decision_record_hash"]
        or authorization.get("decision_date") != payload["decision_dt"]
        or authorization.get("entry_date") != payload["entry_dt"]
        or authorization.get("exit_date") != payload["exit_dt"]
        or authorization.get("raw_source_closure_sha256")
        != payload["raw_source_closure_sha256"]
        or authorization.get("label_observation_sha256")
        != payload["label_observation_sha256"]
    ):
        raise WeeklyOperationError("label authorization fields differ from the anticipated payload")
    validate_label_release(
        str(payload["exit_dt"]),
        now=stage3._parse_utc(str(record["recorded_at_utc"])),
    )
    state_path = Path(str(authorization.get("state_manifest_path", ""))).expanduser().absolute()
    state_raw = _stable_file_bytes(state_path, label="authorized state manifest")
    if (
        sha256_bytes(state_raw) != authorization.get("state_manifest_sha256")
    ):
        raise WeeklyOperationError("authorized state manifest is missing or differs")
    closure_path = _safe_object_path(
        root,
        authorization.get("raw_source_closure_object"),
        label="authorized raw closure",
    )
    observation_path = _safe_object_path(
        root,
        authorization.get("label_observation_object"),
        label="authorized label observation",
    )
    preflight_path = _safe_object_path(
        root,
        authorization.get("preflight_report_object"),
        label="authorized label preflight",
    )
    closure_raw = _stable_file_bytes(closure_path, label="authorized raw closure")
    observation_raw = _stable_file_bytes(
        observation_path,
        label="authorized label observation",
    )
    preflight_raw = _stable_file_bytes(
        preflight_path,
        label="authorized label preflight",
    )
    if (
        closure_path.stem != authorization["raw_source_closure_sha256"]
        or sha256_bytes(closure_raw) != authorization["raw_source_closure_sha256"]
        or observation_path.stem != authorization["label_observation_sha256"]
        or sha256_bytes(observation_raw) != authorization["label_observation_sha256"]
        or preflight_path.stem != authorization["preflight_report_sha256"]
        or sha256_bytes(preflight_raw) != authorization["preflight_report_sha256"]
    ):
        raise WeeklyOperationError("authorized evidence path/digest binding differs")
    closure = _canonical_json_object(closure_raw, label="authorized raw closure")
    snapshot = _canonical_json_object(observation_raw, label="authorized label observation")
    preflight = _canonical_json_object(preflight_raw, label="authorized label preflight")
    decision_records = [
        candidate.data
        for candidate in stage3.scan_records(root)
        if candidate.data["record_type"] == "decision_freeze"
    ]
    matching_decisions = [
        decision
        for decision in decision_records
        if decision["record_hash"] == payload["decision_record_hash"]
    ]
    if len(matching_decisions) != 1:
        raise WeeklyOperationError("authorized label decision is not unique")
    decision = matching_decisions[0]
    schedule = _schedule_for_decision(spec, root, decision)
    due = schedule.loc[
        schedule["week_index"].gt(int(decision["payload"]["week_index"]))
        & schedule["decision_dt"].le(pd.Timestamp(payload["exit_dt"]))
    ]
    by_index = {int(item["payload"]["week_index"]): item for item in decision_records}
    required_later: list[Mapping[str, Any]] = []
    for scheduled in due.itertuples(index=False):
        later = by_index.get(int(scheduled.week_index))
        if (
            later is None
            or int(later["sequence"]) >= int(record["sequence"])
            or str(later["payload"]["decision_dt"]) != _iso_date(scheduled.decision_dt)
        ):
            raise WeeklyOperationError(
                "authorized label preceded a calendar-priority decision"
            )
        required_later.append(later)
    target = LabelTarget(
        week_index=int(decision["payload"]["week_index"]),
        decision_record=decision,
        required_later_decisions=tuple(required_later),
    )
    exit_evidence = preflight.get("exit_evidence")
    if not isinstance(exit_evidence, Mapping):
        raise WeeklyOperationError("authorized preflight has no exit evidence")
    raw_summary = exit_evidence.get("raw")
    state_summary = exit_evidence.get("state")
    if (
        not isinstance(raw_summary, Mapping)
        or not isinstance(state_summary, Mapping)
        or raw_summary.get("audit_sha256") != authorization["raw_audit_sha256"]
        or raw_summary.get("parquet_inventory_sha256")
        != authorization["raw_parquet_inventory_sha256"]
        or state_summary.get("manifest_sha256")
        != authorization["state_manifest_sha256"]
    ):
        raise WeeklyOperationError(
            "authorized preflight raw/state summary differs from authorization"
        )
    historical_evidence = LabelEvidence(
        raw=raw_summary,
        state=state_summary,
        closure=closure,
        closure_sha256=str(authorization["raw_source_closure_sha256"]),
        snapshot=snapshot,
        observation_sha256=str(authorization["label_observation_sha256"]),
        payload=payload,
    )
    if (
        _iso_date(closure.get("coverage_max_dt")) != payload["exit_dt"]
        or closure.get("state_manifest_sha256") != authorization["state_manifest_sha256"]
        or snapshot.get("raw_source_closure_sha256") != closure_path.stem
        or snapshot.get("raw_source_closure_object")
        != str(closure_path.relative_to(root))
        or snapshot.get("entry_dt") != payload["entry_dt"]
        or snapshot.get("exit_dt") != payload["exit_dt"]
        or canonical_json(
            stage3.derive_label_events(
                next(
                    decision.data["payload"]["proposals"]
                    for decision in stage3.scan_records(root)
                    if decision.data["record_hash"] == payload["decision_record_hash"]
                ),
                snapshot,
            )
        )
        != canonical_json(payload["events"])
    ):
        raise WeeklyOperationError("authorized label objects do not replay the exact payload")
    _validate_label_preflight_receipt(
        spec,
        root,
        preflight,
        target=target,
        evidence=historical_evidence,
        state_manifest_path=state_path,
        expected_head=str(record["previous_hash"]),
        git=_authorization_git(authorization),
        operator_source_sha256=str(authorization["operator_source_sha256"]),
    )
    expected_authorization = _build_label_authorization(
        spec,
        root,
        record,
        target=target,
        evidence=historical_evidence,
        state_manifest_path=state_path,
        preflight_report=preflight,
        preflight_path=preflight_path,
        git=_authorization_git(authorization),
        operator_source_sha256=str(authorization["operator_source_sha256"]),
        validate_preflight_receipt=False,
    )
    if canonical_json(authorization) != canonical_json(expected_authorization):
        raise WeeklyOperationError(
            "label authorization is not the exact canonical historical authorization"
        )
    return authorization, path.stem, path


def load_label_append_authorization(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_current_operator: bool,
) -> tuple[dict[str, Any], str, Path]:
    """Cache fully validated historical auths during nested prefix replay."""

    session = _LABEL_VALIDATION_SESSION.get()
    owns_session = session is None
    token = None
    if session is None:
        session = {"authorization": {}}
        token = _LABEL_VALIDATION_SESSION.set(session)
    cache = session.setdefault("authorization", {})
    key = (
        str(root.expanduser().resolve()),
        str(record.get("record_hash", "")),
        require_current_operator,
    )
    cached = cache.get(key)
    if cached is not None:
        authorization, digest, path = cached
        return dict(authorization), str(digest), Path(path)
    try:
        result = _load_label_append_authorization_impl(
            spec,
            root,
            record,
            require_current_operator=require_current_operator,
        )
        cache[key] = result
        return result
    finally:
        if owns_session and token is not None:
            _LABEL_VALIDATION_SESSION.reset(token)


def validate_label_sidecar_inventory(
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    *,
    allow_missing_current_authorized: bool = False,
    ignored_sidecar_names: set[str] | None = None,
) -> None:
    """Require exact one-to-one label authorization and tracked sidecar inventories."""

    labels = _label_rows(records)
    label_by_hash = {str(row["record_hash"]): row for row in labels}
    authorized: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, value in _authorization_objects(root):
        record_hash = _authorization_record_hash(value)
        if record_hash not in label_by_hash:
            # A process may die after persisting exact intent but before the
            # immutable record write.  Such content-addressed intent is inert:
            # it has no sidecar, cannot authorize a different record hash, and
            # is ignored exactly like the first-week operator's unmatched
            # intents.  Existing records still require one unique match below.
            continue
        if record_hash in authorized:
            raise WeeklyOperationError("label has multiple append-before-record authorizations")
        authorized[record_hash] = (path, value)
    missing_auth = set(label_by_hash) - set(authorized)
    if missing_auth:
        raise StudyAbortedInvalid(
            "direct collector label is permanently unauthorized: "
            f"{sorted(missing_auth)}"
        )
    directory = stage3.SCRIPTS_DIR / LABEL_SIDECAR_DIR_NAME
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise WeeklyOperationError("tracked weekly authorization inventory is not a real directory")
    actual: set[str] = set()
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise WeeklyOperationError(f"unexpected weekly authorization sidecar: {path}")
            actual.add(path.name)
    ignored = set() if ignored_sidecar_names is None else set(ignored_sidecar_names)
    if ignored - actual:
        raise WeeklyOperationError(
            f"requested ignored weekly sidecars do not exist: {sorted(ignored - actual)}"
        )
    actual -= ignored
    expected = {
        _tracked_record_file_name(label_by_hash[record_hash])
        for record_hash in authorized
    }
    allowed = {frozenset(expected)}
    if allow_missing_current_authorized and labels:
        current = labels[-1]
        if current["record_hash"] in authorized:
            allowed.add(frozenset(expected - {_tracked_record_file_name(current)}))
    if frozenset(actual) not in allowed:
        raise WeeklyOperationError(
            "tracked weekly sidecars differ from the exact authorized label chain: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _expected_label_sidecar(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_current_operator: bool,
) -> tuple[dict[str, Any], Path]:
    authorization, auth_sha, auth_path = load_label_append_authorization(
        spec,
        root,
        record,
        require_current_operator=require_current_operator,
    )
    payload = {
        "schema": LABEL_SIDECAR_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "sequence": record["sequence"],
        "record_type": record["record_type"],
        "logical_event_key": record["logical_event_key"],
        "recorded_at_utc": record["recorded_at_utc"],
        "record_hash": record["record_hash"],
        "previous_hash": record["previous_hash"],
        "authorization_object": str(auth_path.relative_to(root)),
        "authorization_sha256": auth_sha,
        "operator_source_sha256": authorization["operator_source_sha256"],
        "authorized_git_head": authorization["git"]["head"],
        "external_timestamp_or_signature": False,
        "required_follow_up": (
            "commit_and_push_with_the_matching_head_anchor_before_the_next_formal_event"
        ),
    }
    return payload, _label_sidecar_path(record)


def export_label_authorization_sidecar(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    payload, path = _expected_label_sidecar(
        spec,
        root,
        record,
        require_current_operator=True,
    )
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if path.exists():
        if _stable_file_bytes(path, label="tracked label sidecar") != raw:
            raise WeeklyOperationError(f"tracked label sidecar bytes conflict: {path}")
    else:
        stage3._exclusive_write(path, raw)
    return payload, path


def validate_label_authorized_anchor_pair(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
    verify_actual_remote: bool = True,
) -> dict[str, Any]:
    """Public strict validator used by the following decision operator."""

    validate_label_sidecar_inventory(
        root,
        stage3.scan_records(root),
    )
    load_label_append_authorization(
        spec,
        root,
        record,
        require_current_operator=False,
    )
    return validate_record_authorized_anchor_pair(
        spec,
        root,
        record,
        require_pushed=require_pushed,
        verify_actual_remote=require_pushed and verify_actual_remote,
    )


def validate_authorized_label_pair(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
    verify_actual_remote: bool = True,
) -> dict[str, Any]:
    """Compatibility/public spelling for strict label-pair validation."""

    return validate_label_authorized_anchor_pair(
        spec,
        root,
        record,
        require_pushed=require_pushed,
        verify_actual_remote=verify_actual_remote,
    )


def validate_label_chain(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    require_pushed: bool,
    verify_actual_remote: bool = True,
    ignored_sidecar_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Audit every label's order, 18:00 gate, authorization, and calendar DAG."""

    decisions = _decision_rows(records)
    labels = _label_rows(records)
    decision_by_hash = {str(row["record_hash"]): row for row in decisions}
    expected = [str(row["record_hash"]) for row in decisions[: len(labels)]]
    actual = [str(row["payload"]["decision_record_hash"]) for row in labels]
    if actual != expected:
        raise StudyAbortedInvalid("labels are not the exact oldest-decision prefix")
    validate_label_sidecar_inventory(
        root,
        records,
        ignored_sidecar_names=ignored_sidecar_names,
    )
    evidence: list[dict[str, Any]] = []
    for label in labels:
        decision = decision_by_hash[str(label["payload"]["decision_record_hash"])]
        validate_label_release(
            str(label["payload"]["exit_dt"]),
            now=stage3._parse_utc(str(label["recorded_at_utc"])),
        )
        schedule = _schedule_for_decision(spec, root, decision)
        due = schedule.loc[
            schedule["week_index"].gt(int(decision["payload"]["week_index"]))
            & schedule["decision_dt"].le(pd.Timestamp(label["payload"]["exit_dt"]))
        ]
        by_index = {int(row["payload"]["week_index"]): row for row in decisions}
        for scheduled in due.itertuples(index=False):
            later = by_index.get(int(scheduled.week_index))
            if (
                later is None
                or str(later["payload"]["decision_dt"]) != _iso_date(scheduled.decision_dt)
                or int(later["sequence"]) >= int(label["sequence"])
            ):
                raise StudyAbortedInvalid(
                    "label was recorded before every calendar-priority decision"
                )
        evidence.append(
            validate_record_authorized_anchor_pair(
                spec,
                root,
                label,
                require_pushed=require_pushed,
                verify_actual_remote=verify_actual_remote,
            )
        )
    return evidence


def validate_weekly_label_chain(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord | Mapping[str, Any]],
    *,
    require_pushed: bool,
    verify_actual_remote: bool = True,
) -> list[dict[str, Any]]:
    """Stable integration entry point for the generalized decision operator."""

    return validate_label_chain(
        spec,
        root,
        records,
        require_pushed=require_pushed,
        verify_actual_remote=verify_actual_remote,
    )


def append_label_with_final_guards(
    spec: Mapping[str, Any],
    root: Path,
    state_manifest_path: Path,
    target: LabelTarget,
    evidence: LabelEvidence,
    preflight_report: Mapping[str, Any],
    *,
    expected_head: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[
    stage3.LedgerRecord,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    Path,
]:
    """Authorize then append one exact label under the physical ledger lock."""

    authorization_path: Path | None = None
    state_manifest_path = state_manifest_path.expanduser().absolute()
    data_dir = first_week.validate_frozen_data_dir(data_dir)
    stage3.verify_raw_source_closure_current(evidence.closure)

    with first_week.authorization_ledger_lock(root), first_week.atomic_stage3_writes(root):
        original_append = stage3.append_record
        original_lock = stage3._exclusive_lock
        owner_lease = first_week._active_authorization_lock_lease()
        if (
            owner_lease is None
            or owner_lease.root != root.expanduser().resolve()
            or not first_week._physical_authorization_lock_owned(root)
        ):
            raise WeeklyOperationError("guarded label append lost its physical lock")

        @contextmanager
        def reuse_current_lock_or_block(lock_root: Path) -> Iterator[None]:
            lease = first_week._active_authorization_lock_lease()
            resolved = lock_root.expanduser().resolve()
            if (
                lease is owner_lease
                and lease.root == resolved
                and first_week._physical_authorization_lock_owned(resolved)
            ):
                yield
                return
            if first_week._physical_authorization_lock_owned(resolved):
                raise WeeklyOperationError(
                    "another authorization context attempted to reuse the label lock"
                )
            with original_lock(lock_root):
                yield

        current = stage3.scan_records(root)
        if not current or current[-1].data["record_hash"] != expected_head:
            raise WeeklyOperationError("ledger head changed before guarded label append")
        stage3.validate_record_semantics(spec, current, root=root)
        validate_stage3_operator_pair_chain(
            spec,
            root,
            current,
            require_pushed=True,
            verify_actual_remote=True,
        )
        first_week.validate_existing_anchor_chain(
            spec,
            root,
            require_pushed=True,
        )
        locked_target = derive_oldest_unlabeled_target(
            spec,
            root,
            current,
            decision_date_assertion=target.decision_date,
        )
        if locked_target is None or locked_target.summary() != target.summary():
            raise WeeklyOperationError("oldest label target changed inside append lock")
        locked_git = first_week.validate_git_ready(verify_remote=True)
        del locked_git

        def guarded_append(
            append_root: Path,
            record_type: str,
            logical_event_key: str,
            payload: Mapping[str, Any],
            recorded_at_utc: str | datetime | None = None,
            *,
            ledger_id: str | None = None,
        ) -> stage3.LedgerRecord:
            lease = first_week._active_authorization_lock_lease()
            resolved = append_root.expanduser().resolve()
            if lease is not owner_lease:
                if first_week._physical_authorization_lock_owned(resolved):
                    raise WeeklyOperationError(
                        "another authorization context attempted the guarded label append"
                    )
                return original_append(
                    append_root,
                    record_type,
                    logical_event_key,
                    payload,
                    recorded_at_utc,
                    ledger_id=ledger_id,
                )
            if (
                lease.root != resolved
                or not first_week._physical_authorization_lock_owned(resolved)
            ):
                raise WeeklyOperationError("guarded label append targeted another ledger")
            del recorded_at_utc
            nonlocal authorization_path
            records = stage3.scan_records(root)
            if (
                append_root.resolve() != root.resolve()
                or record_type != "label_completion"
                or not records
                or records[-1].data["record_hash"] != expected_head
                or logical_event_key != f"label:{target.decision_date}"
                or ledger_id != records[0].data["ledger_id"]
                or canonical_json(payload) != canonical_json(evidence.payload)
            ):
                raise WeeklyOperationError(
                    "collector label target, head, or exact observation payload changed"
                )
            fresh_now = stage3.utc_now()
            validate_label_release(
                target.exit_date,
                now=stage3._parse_utc(stage3._format_utc(fresh_now)),
            )
            fresh_git = first_week.validate_git_ready(verify_remote=True)
            if canonical_json(preflight_report.get("git")) != canonical_json(fresh_git):
                raise WeeklyOperationError("Git changed after initial label preflight")
            fresh_report, fresh_target, fresh_evidence = build_label_preflight(
                spec,
                root,
                state_manifest_path,
                data_dir=data_dir,
                decision_date_assertion=target.decision_date,
                verify_remote=True,
                now=stage3._parse_utc(stage3._format_utc(fresh_now)),
            )
            if (
                fresh_target.summary() != target.summary()
                or canonical_json(fresh_evidence.payload) != canonical_json(evidence.payload)
            ):
                raise WeeklyOperationError("raw/state observations changed inside the append lock")
            anticipated = first_week._anticipated_record_data(
                records,
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=payload,
                recorded_at_utc=fresh_now,
                ledger_id=str(ledger_id),
            )
            stage3.validate_record_semantics(
                spec,
                [*records, anticipated],
                root=root,
            )
            _, _, authorization_path = store_label_append_authorization(
                spec,
                root,
                anticipated,
                target=target,
                evidence=fresh_evidence,
                state_manifest_path=state_manifest_path,
                preflight_report=fresh_report,
                fresh_git=fresh_git,
            )
            appended = original_append(
                append_root,
                record_type,
                logical_event_key,
                payload,
                fresh_now,
                ledger_id=ledger_id,
            )
            if canonical_json(appended.data) != canonical_json(anticipated):
                raise WeeklyOperationError(
                    "immutable label differs from its pre-record authorization"
                )
            return appended

        stage3._exclusive_lock = reuse_current_lock_or_block
        stage3.append_record = guarded_append
        try:
            record = stage3.append_label_completion(
                spec,
                root,
                target.decision_date,
                state_manifest_path=state_manifest_path,
            )
        finally:
            stage3.append_record = original_append
            stage3._exclusive_lock = original_lock

        if record.data["record_hash"] != stage3.scan_records(root)[-1].data["record_hash"]:
            raise WeeklyOperationError("guarded label is not the final ledger head")
        if authorization_path is None:
            raise WeeklyOperationError(
                "collector returned without a durable pre-record label authorization"
            )
        sidecar, sidecar_path = export_label_authorization_sidecar(
            spec,
            root,
            record.data,
        )
        anchor, anchor_path = stage3.export_ledger_head_anchor(spec, root)
        validate_label_sidecar_inventory(root, stage3.scan_records(root))
        validate_label_authorized_anchor_pair(
            spec,
            root,
            record.data,
            require_pushed=False,
        )
        first_week.validate_existing_anchor_chain(
            spec,
            root,
            require_pushed=False,
        )
    return (
        record,
        anchor,
        anchor_path,
        sidecar,
        sidecar_path,
        authorization_path,
    )


def complete_next_label(
    *,
    state_manifest_path: Path,
    expected_head: str,
    decision_date_assertion: str | None,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Complete at most one oldest label, or recover its pre-authorized evidence."""

    spec = stage3.load_and_validate_spec()
    root = stage3.resolve_ledger_root(spec)
    data_dir = first_week.validate_frozen_data_dir(data_dir)
    initial = stage3.scan_records(root)
    if initial and initial[-1].data["record_type"] == "label_completion":
        current_label = initial[-1].data
        if (
            current_label["previous_hash"] == expected_head
            and (
                decision_date_assertion is None
                or _iso_date(decision_date_assertion)
                == current_label["payload"]["decision_dt"]
            )
        ):
            return recover_label_evidence(
                record_hash=current_label["record_hash"],
                decision_date_assertion=decision_date_assertion,
                data_dir=data_dir,
                now=now,
            )
    if not initial or initial[-1].data["record_hash"] != expected_head:
        raise WeeklyOperationError("expected head is not the current pre-label head")
    first_week.validate_git_ready(verify_remote=True)
    with first_week.operator_lock(root), cache_lock(data_dir):
        preflight, target, evidence = build_label_preflight(
            spec,
            root,
            state_manifest_path,
            data_dir=data_dir,
            decision_date_assertion=decision_date_assertion,
            verify_remote=True,
            now=now,
        )
        (
            record,
            anchor,
            anchor_path,
            sidecar,
            sidecar_path,
            authorization_path,
        ) = append_label_with_final_guards(
            spec,
            root,
            state_manifest_path,
            target,
            evidence,
            preflight,
            expected_head=expected_head,
            data_dir=data_dir,
        )
    return {
        "status": "OLDEST_LABEL_COMPLETED_EVIDENCE_EXPORTED",
        "target": target.summary(),
        "record_hash": record.data["record_hash"],
        "recorded_at_utc": record.data["recorded_at_utc"],
        "authorization_path": str(authorization_path),
        "authorization_sha256": Path(authorization_path).stem,
        "authorization_sidecar_path": str(sidecar_path),
        "authorization_sidecar_sha256": sha256_file(sidecar_path),
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor": anchor,
        "formal_ledger_mutated": True,
        "ledger_records_appended": 1,
        "next_state": "LABEL_EVIDENCE_COMMIT_PENDING",
        "required_action": (
            "commit and push exactly the matching label authorization sidecar "
            "and head anchor in one commit before any next formal event"
        ),
    }


def _validate_label_order_for_recovery(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord],
) -> None:
    """Replay order/timing for recovery without requiring the current sidecar."""

    decisions = _decision_rows(records)
    labels = _label_rows(records)
    expected = [str(row["record_hash"]) for row in decisions[: len(labels)]]
    actual = [str(row["payload"]["decision_record_hash"]) for row in labels]
    if actual != expected:
        raise StudyAbortedInvalid("recovery label chain is not the oldest-decision prefix")
    by_hash = {str(row["record_hash"]): row for row in decisions}
    by_index = {int(row["payload"]["week_index"]): row for row in decisions}
    for label in labels:
        validate_label_release(
            str(label["payload"]["exit_dt"]),
            now=stage3._parse_utc(str(label["recorded_at_utc"])),
        )
        decision = by_hash[str(label["payload"]["decision_record_hash"])]
        schedule = _schedule_for_decision(spec, root, decision)
        due = schedule.loc[
            schedule["week_index"].gt(int(decision["payload"]["week_index"]))
            & schedule["decision_dt"].le(pd.Timestamp(label["payload"]["exit_dt"]))
        ]
        for scheduled in due.itertuples(index=False):
            later = by_index.get(int(scheduled.week_index))
            if (
                later is None
                or int(later["sequence"]) >= int(label["sequence"])
                or str(later["payload"]["decision_dt"]) != _iso_date(scheduled.decision_dt)
            ):
                raise StudyAbortedInvalid(
                    "recovery found a label before a calendar-priority decision"
                )


def _inspect_committed_label_pair(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate a clean exact local pair commit without trusting tracking refs."""

    anchor_path = _anchor_path(record)
    sidecar_path = _label_sidecar_path(record)
    if not anchor_path.is_file() or not sidecar_path.is_file():
        return None
    expected_sidecar, _ = _expected_label_sidecar(
        spec,
        root,
        record,
        require_current_operator=False,
    )
    expected_raw = json.dumps(
        expected_sidecar,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if _stable_file_bytes(sidecar_path, label="committed label sidecar") != expected_raw:
        raise WeeklyOperationError("local committed label sidecar bytes differ")
    first_week.validate_anchor_for_record(
        spec,
        root,
        record,
        require_pushed=False,
    )
    relative_paths = [
        path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix()
        for path in (anchor_path, sidecar_path)
    ]
    commits = {
        first_week._git_output(
            stage3.REPO_ROOT,
            "log",
            "-1",
            "--format=%H",
            "--",
            relative,
        )
        for relative in relative_paths
    }
    if "" in commits or len(commits) != 1:
        return None
    commit = commits.pop()
    if first_week._git_output(stage3.REPO_ROOT, "rev-parse", "HEAD") != commit:
        raise WeeklyOperationError("exact pair commit exists but is not the current clean HEAD")
    if first_week._git_output(
        stage3.REPO_ROOT,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise WeeklyOperationError("committed label pair recovery requires a clean worktree")
    for path, relative in zip((anchor_path, sidecar_path), relative_paths, strict=True):
        committed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=stage3.REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != _stable_file_bytes(path, label="committed label evidence"):
            raise WeeklyOperationError("working label evidence differs from its pair commit")
    parents = first_week._git_output(
        stage3.REPO_ROOT,
        "rev-list",
        "--parents",
        "-n",
        "1",
        commit,
    ).split()
    authorized_head = str(_authorization_git(authorization)["head"])
    if len(parents) != 2 or parents[1] != authorized_head:
        raise WeeklyOperationError(
            "local label pair commit must have exactly the authorized head as parent"
        )
    changed = set(
        filter(
            None,
            first_week._git_output(
                stage3.REPO_ROOT,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit,
            ).splitlines(),
        )
    )
    if changed != set(relative_paths):
        raise WeeklyOperationError("local label pair commit contains extra or missing paths")
    remote = first_week.validate_actual_remote_branch()
    remote_head = str(remote["remote_head"])
    remote_contains = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, remote_head),
        cwd=stage3.REPO_ROOT,
        check=False,
        capture_output=True,
    ).returncode == 0
    if not remote_contains and remote_head != authorized_head:
        raise WeeklyOperationError(
            "actual remote is neither the authorized parent nor a descendant of the pair"
        )
    return {
        "commit": commit,
        "authorized_head": authorized_head,
        "remote_head": remote_head,
        "remote_contains_pair": remote_contains,
    }


def recover_label_evidence(
    *,
    record_hash: str,
    decision_date_assertion: str | None,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recover only sidecar/anchor for one pre-authorized current-head label."""

    spec = stage3.load_and_validate_spec()
    root = stage3.resolve_ledger_root(spec)
    data_dir = first_week.validate_frozen_data_dir(data_dir)
    with first_week.operator_lock(root), cache_lock(data_dir):
        records = stage3.scan_records(root)
        stage3.validate_record_semantics(spec, records, root=root)
        if (
            not records
            or records[-1].data["record_hash"] != record_hash
            or records[-1].data["record_type"] != "label_completion"
        ):
            raise WeeklyOperationError(
                "recovery is allowed only for the exact current-head label"
            )
        label = records[-1].data
        if (
            decision_date_assertion is not None
            and _iso_date(decision_date_assertion) != label["payload"]["decision_dt"]
        ):
            raise WeeklyOperationError("recovery CLI date differs from the current-head label")
        _validate_label_order_for_recovery(spec, root, records)
        validate_label_sidecar_inventory(
            root,
            records,
            allow_missing_current_authorized=True,
        )
        authorization, auth_sha, auth_path = load_label_append_authorization(
            spec,
            root,
            label,
            require_current_operator=False,
        )
        # Every earlier D/L pair must already be pushed and causally present.
        prefix_pair_evidence = validate_stage3_operator_pair_chain(
            spec,
            root,
            records[:-1],
            require_pushed=True,
            verify_actual_remote=True,
            ignored_label_sidecar_names=(
                {_tracked_record_file_name(label)}
                if _label_sidecar_path(label).is_file()
                else None
            ),
        )
        committed_pair = _inspect_committed_label_pair(
            spec,
            root,
            label,
            authorization,
        )
        validate_successor_authorized_head(
            spec,
            root,
            records[:-1],
            label,
            prefix_pair_evidence=prefix_pair_evidence,
            require_pushed_prefix=True,
        )
        if committed_pair is not None:
            if committed_pair["remote_contains_pair"]:
                return {
                    "status": "AUTHORIZED_LABEL_EVIDENCE_ALREADY_PUSHED",
                    "record_hash": record_hash,
                    "authorization_path": str(auth_path),
                    "authorization_sha256": auth_sha,
                    "evidence_commit": committed_pair["commit"],
                    "actual_remote_head": committed_pair["remote_head"],
                    "formal_ledger_mutated": False,
                }
            with suppress(RequiredDecisionPending):
                validate_no_missed_decision_window(
                    spec,
                    root,
                    records,
                    now=now,
                )
            return {
                "status": "LABEL_EVIDENCE_COMMIT_AWAITING_PUSH",
                "record_hash": record_hash,
                "authorization_path": str(auth_path),
                "authorization_sha256": auth_sha,
                "evidence_commit": committed_pair["commit"],
                "actual_remote_head": committed_pair["remote_head"],
                "formal_ledger_mutated": False,
                "required_action": "push the already exact local pair commit; do not rewrite it",
            }

        with suppress(RequiredDecisionPending):
            validate_no_missed_decision_window(spec, root, records, now=now)
            # Dirty evidence recovery has priority while the next decision
            # window remains live, but never after a missed entry-open window.
        load_label_append_authorization(
            spec,
            root,
            label,
            require_current_operator=True,
        )
        anchor_path = _anchor_path(label)
        sidecar_path = _label_sidecar_path(label)
        allowed_dirty = {
            anchor_path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
            sidecar_path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
        }
        recovery_git = first_week.validate_recovery_git_ready(
            authorization,
            allowed_dirty_paths=allowed_dirty,
        )
        raw = first_week.validate_raw_ready(
            str(label["payload"]["exit_dt"]),
            data_dir=data_dir,
        )
        state_path = Path(str(authorization["state_manifest_path"])).absolute()
        state = first_week.validate_state_ready(
            state_path,
            str(label["payload"]["exit_dt"]),
            data_dir=data_dir,
        )
        closure_path = _safe_object_path(
            root,
            authorization["raw_source_closure_object"],
            label="authorized recovery closure",
        )
        closure = _canonical_json_object(
            _stable_file_bytes(closure_path, label="authorized recovery closure"),
            label="authorized recovery closure",
        )
        stage3.verify_raw_source_closure_current(closure)
        if (
            raw["audit_sha256"] != authorization["raw_audit_sha256"]
            or raw["parquet_inventory_sha256"]
            != authorization["raw_parquet_inventory_sha256"]
            or state["manifest_sha256"] != authorization["state_manifest_sha256"]
            or state["raw_parquet_inventory_sha256"]
            != authorization["raw_parquet_inventory_sha256"]
        ):
            raise WeeklyOperationError(
                "recovery raw/state differs from pre-record authorization; "
                "newer raw may never be recaptured"
            )

        with first_week.authorization_ledger_lock(root), first_week.atomic_stage3_writes(root):
            locked = stage3.scan_records(root)
            if not locked or locked[-1].data["record_hash"] != record_hash:
                raise WeeklyOperationError("current label head changed during recovery")
            load_label_append_authorization(
                spec,
                root,
                label,
                require_current_operator=True,
            )
            sidecar, sidecar_path = export_label_authorization_sidecar(
                spec,
                root,
                label,
            )
            anchor, anchor_path = stage3.export_ledger_head_anchor(spec, root)
            validate_label_sidecar_inventory(root, locked)
            validate_label_authorized_anchor_pair(
                spec,
                root,
                label,
                require_pushed=False,
            )
        final_git = first_week.validate_recovery_git_ready(
            authorization,
            allowed_dirty_paths=allowed_dirty,
        )
    return {
        "status": "AUTHORIZED_LABEL_EVIDENCE_RECOVERED",
        "record_hash": record_hash,
        "authorization_path": str(auth_path),
        "authorization_sha256": auth_sha,
        "authorization_sidecar_path": str(sidecar_path),
        "authorization_sidecar_sha256": sha256_file(sidecar_path),
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor": anchor,
        "recovery_git": recovery_git,
        "final_recovery_git": final_git,
        "formal_ledger_mutated": False,
        "ledger_records_appended": 0,
        "next_state": "LABEL_EVIDENCE_COMMIT_PENDING",
        "required_action": (
            "commit and push exactly the recovered label sidecar and matching "
            "anchor in one commit before any next formal event"
        ),
    }


def status_snapshot(
    *,
    decision_date_assertion: str | None,
    state_manifest_path: Path | None,
    data_dir: Path = DEFAULT_DATA_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the next local action without network access or writes."""

    spec = stage3.load_and_validate_spec()
    root = stage3.resolve_ledger_root(spec)
    data_dir = first_week.validate_frozen_data_dir(data_dir)
    current = _as_utc(now)
    records = stage3.scan_records(root)
    base = {
        "schema": "xs_chan_stage3_weekly_label_status_v1",
        "checked_at_utc": current,
        "study_identity": stage3.study_identity(spec),
        "ledger": first_week.ledger_closure(root),
        "formal_ledger_mutated": False,
        "remote_validation": "LOCAL_TRACKING_ONLY_NO_NETWORK",
    }
    try:
        stage3.validate_record_semantics(spec, records, root=root)
        _validate_label_order_for_recovery(spec, root, records)
        validate_label_sidecar_inventory(
            root,
            records,
            allow_missing_current_authorized=True,
        )
    except StudyAbortedInvalid as exc:
        return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}
    except (WeeklyOperationError, first_week.FirstWeekOperationError, stage3.Stage3Error) as exc:
        return {**base, "state": "INVALID_LOCAL_EVIDENCE", "reason": str(exc)}

    def validate_recoverable_current_label(
        current_label: Mapping[str, Any],
    ) -> None:
        prefix = records[:-1]
        prefix_pair_evidence = validate_stage3_operator_pair_chain(
            spec,
            root,
            prefix,
            require_pushed=True,
            verify_actual_remote=False,
            ignored_label_sidecar_names=(
                {_tracked_record_file_name(current_label)}
                if _label_sidecar_path(current_label).is_file()
                else None
            ),
        )
        validate_successor_authorized_head(
            spec,
            root,
            prefix,
            current_label,
            prefix_pair_evidence=prefix_pair_evidence,
            require_pushed_prefix=True,
        )

    labels = _label_rows(records)
    if labels:
        current_label = labels[-1]
        sidecar = _label_sidecar_path(current_label)
        anchor = _anchor_path(current_label)
        if not sidecar.is_file() or not anchor.is_file():
            if records[-1].data["record_hash"] != current_label["record_hash"]:
                return {
                    **base,
                    "state": "ABORTED_INVALID",
                    "reason": "a non-current label pair is missing before a later formal event",
                }
            try:
                validate_recoverable_current_label(current_label)
            except (WeeklyOperationError, first_week.FirstWeekOperationError, stage3.Stage3Error) as exc:
                return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}
            return {
                **base,
                "state": "LABEL_EVIDENCE_COMMIT_PENDING",
                "record_hash": current_label["record_hash"],
                "required_action": "recover-label, then commit/push the exact pair",
            }
        try:
            validate_label_authorized_anchor_pair(
                spec,
                root,
                current_label,
                require_pushed=True,
                verify_actual_remote=False,
            )
        except (WeeklyOperationError, first_week.FirstWeekOperationError, stage3.Stage3Error):
            if records[-1].data["record_hash"] != current_label["record_hash"]:
                return {
                    **base,
                    "state": "ABORTED_INVALID",
                    "reason": "a non-current label pair was not pushed before a later event",
                }
            try:
                validate_recoverable_current_label(current_label)
            except (
                WeeklyOperationError,
                first_week.FirstWeekOperationError,
                stage3.Stage3Error,
            ) as exc:
                return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}
            return {
                **base,
                "state": "LABEL_EVIDENCE_COMMIT_PENDING",
                "record_hash": current_label["record_hash"],
                "required_action": "commit/push the exact current label pair",
            }
    try:
        validate_stage3_operator_pair_chain(
            spec,
            root,
            records,
            require_pushed=True,
            verify_actual_remote=False,
        )
    except (WeeklyOperationError, first_week.FirstWeekOperationError, stage3.Stage3Error) as exc:
        current_record = records[-1].data if records else None
        if (
            isinstance(current_record, Mapping)
            and current_record.get("record_type") == "decision_freeze"
        ):
            try:
                decision_status = first_week.status_snapshot(
                    decision_date=str(current_record["payload"]["decision_dt"]),
                    data_dir=data_dir,
                    now=current,
                )
            except (
                WeeklyOperationError,
                first_week.FirstWeekOperationError,
                stage3.Stage3Error,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as status_exc:
                return {
                    **base,
                    "state": "ABORTED_INVALID",
                    "reason": str(status_exc),
                }
            if decision_status.get("state") == "DECISION_EVIDENCE_COMMIT_PENDING":
                return {
                    **base,
                    "state": "WAIT_DECISION_EVIDENCE",
                    "record_hash": current_record["record_hash"],
                    "decision_date": current_record["payload"]["decision_dt"],
                    "decision_status": decision_status,
                    "required_action": (
                        "rerun freeze-decision --apply-ledger with the exact "
                        "authorized inputs/head to recover evidence if needed, "
                        "then commit and push the exact decision pair"
                    ),
                }
            return {
                **base,
                "state": "ABORTED_INVALID",
                "reason": (
                    "current decision evidence is not safely recoverable: "
                    f"{decision_status.get('state', str(exc))}"
                ),
                "decision_status": decision_status,
            }
        return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}

    if not _decision_rows(records):
        try:
            decision_status = first_week.status_snapshot(
                decision_date=decision_date_assertion,
                data_dir=data_dir,
                now=current,
            )
        except (
            WeeklyOperationError,
            first_week.FirstWeekOperationError,
            stage3.Stage3Error,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            return {**base, "state": "INVALID_LOCAL_EVIDENCE", "reason": str(exc)}
        decision_state = decision_status.get("state")
        if not isinstance(decision_state, str) or not decision_state:
            return {
                **base,
                "state": "INVALID_LOCAL_EVIDENCE",
                "reason": "bootstrap decision status did not provide a state",
            }
        return {
            **base,
            "state": decision_state,
            "decision_date": decision_status.get("decision_date"),
            "decision_status": decision_status,
            "global_decision_priority": {
                "status": "BOOTSTRAP_DECISION_STATUS_MIRRORED",
                "decision_state": decision_state,
                "decision_date": decision_status.get("decision_date"),
            },
        }

    try:
        priority = validate_no_missed_decision_window(
            spec,
            root,
            records,
            now=current,
        )
    except RequiredDecisionPending as exc:
        return {
            **base,
            "state": "WAIT_DECISION_PRIORITY",
            "week_index": exc.week_index,
            "decision_date": exc.decision_date,
        }
    except StudyAbortedInvalid as exc:
        return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}
    try:
        target = derive_oldest_unlabeled_target(
            spec,
            root,
            records,
            decision_date_assertion=decision_date_assertion,
            now=current,
        )
    except RequiredDecisionPending as exc:
        return {
            **base,
            "state": "WAIT_DECISION_PRIORITY",
            "week_index": exc.week_index,
            "decision_date": exc.decision_date,
        }
    except StudyAbortedInvalid as exc:
        return {**base, "state": "ABORTED_INVALID", "reason": str(exc)}
    except WeeklyOperationError as exc:
        return {**base, "state": "INVALID_LOCAL_EVIDENCE", "reason": str(exc)}
    if target is None:
        complete = len(_decision_rows(records)) == int(spec["accrual"]["window_weeks"])
        return {
            **base,
            "state": "ALL_LABELS_COMPLETE" if complete else "WAIT_NEXT_DECISION",
            "global_decision_priority": priority,
        }
    release = _label_release(target.exit_date).astimezone(UTC)
    if current < release:
        return {
            **base,
            "state": "WAIT_LABEL_RELEASE",
            "target": target.summary(),
            "release_utc": release,
            "global_decision_priority": priority,
        }
    inventory = inspect_inventory(data_dir)
    maximum = inventory.get("max_dt")
    expected = pd.Timestamp(target.exit_date).strftime("%Y%m%d")
    if maximum is not None and str(maximum) > expected:
        return {
            **base,
            "state": "ABORTED_INVALID",
            "reason": (
                f"active raw advanced past exact label exit: {maximum} > {expected}"
            ),
            "target": target.summary(),
        }
    if str(maximum) != expected:
        return {
            **base,
            "state": "NEEDS_RAW_EXIT",
            "target": target.summary(),
            "raw_max_dt": maximum,
            "required_max_dt": expected,
        }
    if state_manifest_path is None:
        return {
            **base,
            "state": "NEEDS_STATE_EXIT",
            "target": target.summary(),
        }
    try:
        report, _, _ = build_label_preflight(
            spec,
            root,
            state_manifest_path,
            data_dir=data_dir,
            decision_date_assertion=target.decision_date,
            verify_remote=False,
            now=current,
        )
    except (WeeklyOperationError, first_week.FirstWeekOperationError, stage3.Stage3Error) as exc:
        return {
            **base,
            "state": "INVALID_LOCAL_EVIDENCE",
            "reason": str(exc),
            "target": target.summary(),
        }
    return {
        **base,
        "state": "READY_TO_COMPLETE_ONE_LABEL",
        "target": target.summary(),
        "preflight_sha256": sha256_bytes(stage3.canonical_json(report)),
        "expected_head": records[-1].data["record_hash"],
        "required_action": "complete-label --apply-ledger --expected-head <head>",
    }


def _print_json(payload: Any) -> None:
    print(
        json.dumps(
            first_week._json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-date")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--state-manifest", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="read-only next action; no APIs or writes")
    prepare = commands.add_parser(
        "prepare-label",
        help="prepare exact-exit raw/state only with --apply-data",
    )
    prepare.add_argument("--apply-data", action="store_true")
    prepare.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    prepare.add_argument("--state-output-root", type=Path, default=DEFAULT_STATE_OUTPUT_ROOT)
    prepare.add_argument("--workers", type=int, default=min(state_cache.mp.cpu_count(), 8))
    complete = commands.add_parser(
        "complete-label",
        help="append one oldest label only with --apply-ledger",
    )
    complete.add_argument("--apply-ledger", action="store_true")
    complete.add_argument("--expected-head")
    recovery = commands.add_parser(
        "recover-label",
        help="recover only a pre-authorized current-head label",
    )
    recovery.add_argument("--apply-ledger", action="store_true")
    recovery.add_argument("--record-hash")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print_json(
                status_snapshot(
                    decision_date_assertion=args.decision_date,
                    state_manifest_path=args.state_manifest,
                    data_dir=args.data_dir,
                )
            )
            return 0
        if args.command == "prepare-label":
            if not args.apply_data:
                _print_json(
                    {
                        "status": "DRY_RUN_NO_DATA_MUTATION",
                        "snapshot": status_snapshot(
                            decision_date_assertion=args.decision_date,
                            state_manifest_path=args.state_manifest,
                            data_dir=args.data_dir,
                        ),
                        "required_flag": "--apply-data",
                        "formal_ledger_mutated": False,
                    }
                )
                return 0
            _print_json(
                prepare_label_data(
                    decision_date_assertion=args.decision_date,
                    data_dir=args.data_dir,
                    snapshot_root=args.snapshot_root,
                    state_output_root=args.state_output_root,
                    workers=args.workers,
                )
            )
            return 0
        if args.command == "complete-label":
            if not args.apply_ledger:
                _print_json(
                    {
                        "status": "DRY_RUN_NO_LEDGER_MUTATION",
                        "snapshot": status_snapshot(
                            decision_date_assertion=args.decision_date,
                            state_manifest_path=args.state_manifest,
                            data_dir=args.data_dir,
                        ),
                        "required_flag": "--apply-ledger",
                        "formal_ledger_mutated": False,
                    }
                )
                return 0
            if args.state_manifest is None or not args.expected_head:
                raise WeeklyOperationError(
                    "formal label completion requires --state-manifest and --expected-head"
                )
            _print_json(
                complete_next_label(
                    state_manifest_path=args.state_manifest,
                    expected_head=args.expected_head,
                    decision_date_assertion=args.decision_date,
                    data_dir=args.data_dir,
                )
            )
            return 0
        if args.command == "recover-label":
            if not args.apply_ledger:
                _print_json(
                    {
                        "status": "DRY_RUN_NO_LEDGER_MUTATION",
                        "snapshot": status_snapshot(
                            decision_date_assertion=args.decision_date,
                            state_manifest_path=args.state_manifest,
                            data_dir=args.data_dir,
                        ),
                        "required_flag": "--apply-ledger",
                        "formal_ledger_mutated": False,
                    }
                )
                return 0
            if not args.record_hash:
                raise WeeklyOperationError("label recovery requires --record-hash")
            _print_json(
                recover_label_evidence(
                    record_hash=args.record_hash,
                    decision_date_assertion=args.decision_date,
                    data_dir=args.data_dir,
                )
            )
            return 0
        raise WeeklyOperationError(f"unsupported weekly command: {args.command}")
    except StudyAbortedInvalid as exc:
        _print_json(
            {
                "status": "ABORTED_INVALID",
                "reason": str(exc),
                "formal_ledger_mutated": False,
            }
        )
        return 2
    except (
        WeeklyOperationError,
        first_week.FirstWeekOperationError,
        stage3.Stage3Error,
        first_week.DailySyncError,
        state_cache.StateCacheError,
        FileExistsError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
        first_week.pa.ArrowException,
        first_week.pl.exceptions.PolarsError,
    ) as exc:
        print(f"weekly Stage 3 operation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
