"""Fail-closed operator for the first XS-Chan Stage 3 prospective week.

This module is deliberately separate from the frozen Stage 3 collector.  It
does not change the study identity; it adds operational gates around the
registered collector:

* prepare raw/state/reference evidence only after the target daily bar release;
* validate the full eight-date bridge without writing the formal ledger;
* re-check time, raw closure and expected head inside the ledger append lock;
* export the unique head anchor immediately after a successful decision append.

Read-only commands never call market APIs and never write the ledger.  Formal
mutation requires both an explicit apply flag and the expected current head.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xs_chan_exploration_stage3 as stage3
import xs_chan_state_cache as state_cache
from _sync_daily_data import (
    AUDIT_DIR_NAME,
    DAILY_BAR_READY_HOUR,
    DEFAULT_DATA_DIR,
    DEFAULT_MIN_DAILY_ROWS,
    DEFAULT_NEW_SYMBOL_SLEEP_SECONDS,
    DEFAULT_SNAPSHOT_ROOT,
    MARKET_TIMEZONE,
    SNAPSHOT_MANIFEST_FILE_NAME,
    SOURCE_SYMBOL_ABSOLUTE_MINIMUM,
    SOURCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE,
    SOURCE_SYMBOL_MIN_PREVIOUS_SESSION_COVERAGE,
    DailySyncError,
    active_manifest_matches,
    assert_execution_binding_current,
    cache_lock,
    canonical_json,
    inspect_inventory,
    run_sync,
    sha256_bytes,
    sha256_file,
)
from xs_chan_exploration_stage3 import (
    EXCHANGE_TIMEZONE,
    Stage3Error,
    Stage3ValidationError,
)

FIRST_WEEK_REFERENCE_DATES = (
    "2026-06-12",
    "2026-06-18",
    "2026-06-26",
    "2026-07-03",
    "2026-07-10",
    "2026-07-17",
    "2026-07-24",
    "2026-07-31",
)
BRIDGE_START_EXCLUSIVE = "2026-06-06"
FIRST_WEEK_ENTRY_DATE = "2026-08-03"
FIRST_WEEK_EXIT_DATE = "2026-08-10"
ANCHOR_PUSH_RESERVE = timedelta(hours=2)
DEFAULT_STATE_OUTPUT_ROOT = Path.home() / ".ts_data_cache" / "xs_chan_state_cache_stage3"
OPERATOR_LOCK_FILE = ".first_week_operator.lock"
ATOMIC_STAGING_DIR = ".first_week_atomic_staging"
PREFLIGHT_CATEGORY = "first_week_preflight"
FINAL_PREFLIGHT_CATEGORY = "first_week_final_preflight"
APPEND_AUTHORIZATION_CATEGORY = "first_week_append_authorization"
AUTHORIZATION_SCHEMA = "xs_chan_stage3_first_week_append_authorization_v1"
AUTHORIZATION_SIDECAR_SCHEMA = "xs_chan_stage3_first_week_operator_authorization_v1"
AUTHORIZATION_SIDECAR_DIR_NAME = "xs_chan_exploration_stage3_operator_authorizations"
REFERENCE_SYMBOL_ABSOLUTE_MINIMUM = SOURCE_SYMBOL_ABSOLUTE_MINIMUM
REFERENCE_SYMBOL_MIN_PREVIOUS_COVERAGE = SOURCE_SYMBOL_MIN_PREVIOUS_SESSION_COVERAGE
REFERENCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE = SOURCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE
REFERENCE_SYMBOL_MAX_RAW_EXTRA_SHARE = SOURCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE
REQUIRED_GIT_BRANCH = "feat/surge-wave-strategy"
REQUIRED_GIT_UPSTREAM = "mine/feat/surge-wave-strategy"
REQUIRED_GIT_REMOTE_URL = "git@github.com:lovelyzzc/czsc.git"


class _AuthorizationLockLease:
    """A process/thread-bound, revocable proof that the physical lock is held."""

    __slots__ = ("active", "pid", "root", "thread_id")

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pid = os.getpid()
        self.thread_id = threading.get_ident()
        self.active = True


_AUTHORIZED_LEDGER_LOCK_LEASE: ContextVar[_AuthorizationLockLease | None] = ContextVar(
    "xs_chan_stage3_authorized_ledger_lock_lease",
    default=None,
)
_AUTHORIZATION_PHYSICAL_LOCKS = threading.local()
_STAGE3_PATCH_LOCK = threading.RLock()


class FirstWeekOperationError(RuntimeError):
    """An operational invariant failed before a safe next transition."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_json(payload: Any) -> None:
    print(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise FirstWeekOperationError(f"git {' '.join(args)} failed: {detail}") from exc
    return completed.stdout.strip()


def _git_file_sha256_at_commit(
    repo_root: Path,
    commit: str,
    relative_path: str,
) -> str:
    try:
        completed = subprocess.run(
            ("git", "show", f"{commit}:{relative_path}"),
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).decode(errors="replace").strip()
        raise FirstWeekOperationError(
            f"cannot read authorized operator bytes from Git commit {commit}: {detail}"
        ) from exc
    return sha256_bytes(completed.stdout)


def validate_git_ready(
    *,
    repo_root: Path = stage3.REPO_ROOT,
    verify_remote: bool,
) -> dict[str, Any]:
    """Require a clean, attached, pushed branch; optionally query the remote."""

    branch = _git_output(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    dirty = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    if dirty:
        raise FirstWeekOperationError(f"Git worktree must be clean; first dirty entry: {dirty[0]}")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    try:
        upstream_name = _git_output(
            repo_root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_head = _git_output(repo_root, "rev-parse", "@{upstream}")
    except FirstWeekOperationError as exc:
        raise FirstWeekOperationError("formal operation requires a configured upstream branch") from exc
    if upstream_head != head:
        raise FirstWeekOperationError(f"HEAD {head} differs from upstream {upstream_head}")

    if branch != REQUIRED_GIT_BRANCH or upstream_name != REQUIRED_GIT_UPSTREAM:
        raise FirstWeekOperationError(
            "first-week formal operations are bound to "
            f"{REQUIRED_GIT_UPSTREAM}; got branch={branch!r}, upstream={upstream_name!r}"
        )
    remote_name, separator, remote_branch = upstream_name.partition("/")
    if not separator or not remote_name or not remote_branch:
        raise FirstWeekOperationError(f"cannot resolve remote branch from upstream {upstream_name!r}")
    fetch_url = _git_output(repo_root, "remote", "get-url", remote_name)
    push_url = _git_output(repo_root, "remote", "get-url", "--push", remote_name)
    if fetch_url != REQUIRED_GIT_REMOTE_URL or push_url != REQUIRED_GIT_REMOTE_URL:
        raise FirstWeekOperationError(
            "first-week remote URL differs from the frozen operator destination: "
            f"fetch={fetch_url!r}, push={push_url!r}"
        )

    remote_head: str | None = None
    if verify_remote:
        remote_output = _git_output(
            repo_root,
            "ls-remote",
            "--heads",
            remote_name,
            f"refs/heads/{remote_branch}",
        )
        matches = [line.split(maxsplit=1)[0] for line in remote_output.splitlines() if line.strip()]
        if matches != [head]:
            raise FirstWeekOperationError(f"remote branch {upstream_name} does not resolve exactly to HEAD")
        remote_head = matches[0]
    return {
        "branch": branch,
        "head": head,
        "upstream": upstream_name,
        "upstream_head": upstream_head,
        "remote_head": remote_head,
        "remote_fetch_url": fetch_url,
        "remote_push_url": push_url,
        "remote_verified": verify_remote,
        "worktree_clean": True,
    }


def validate_recovery_git_ready(
    authorization: Mapping[str, Any],
    *,
    allowed_dirty_paths: set[str],
    repo_root: Path = stage3.REPO_ROOT,
) -> dict[str, Any]:
    """Allow only the exact uncommitted evidence files after an append crash."""

    branch = _git_output(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    head = _git_output(repo_root, "rev-parse", "HEAD")
    upstream = _git_output(
        repo_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    upstream_head = _git_output(repo_root, "rev-parse", "@{upstream}")
    remote_name, separator, remote_branch = upstream.partition("/")
    if not separator or not remote_name or not remote_branch:
        raise FirstWeekOperationError(f"cannot resolve recovery upstream {upstream!r}")
    fetch_url = _git_output(repo_root, "remote", "get-url", remote_name)
    push_url = _git_output(repo_root, "remote", "get-url", "--push", remote_name)
    remote_output = _git_output(
        repo_root,
        "ls-remote",
        "--heads",
        remote_name,
        f"refs/heads/{remote_branch}",
    )
    remote_matches = [line.split(maxsplit=1)[0] for line in remote_output.splitlines() if line.strip()]
    authorized_git = authorization.get("git")
    if not isinstance(authorized_git, Mapping):
        raise FirstWeekOperationError("append authorization has no Git evidence")
    current_core = {
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "upstream_head": upstream_head,
        "remote_head": remote_matches[0] if len(remote_matches) == 1 else None,
        "remote_fetch_url": fetch_url,
        "remote_push_url": push_url,
        "remote_verified": len(remote_matches) == 1,
    }
    authorized_core = {key: authorized_git.get(key) for key in current_core}
    if current_core != authorized_core or remote_matches != [head] or authorized_git.get("worktree_clean") is not True:
        raise FirstWeekOperationError("recovery Git branch, URLs or actual remote differ from the authorized append")

    dirty_entries = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    dirty_paths: set[str] = set()
    for entry in dirty_entries:
        if len(entry) < 4 or " -> " in entry[3:]:
            raise FirstWeekOperationError(f"cannot safely parse recovery Git status entry: {entry!r}")
        dirty_paths.add(entry[3:])
    if not dirty_paths <= allowed_dirty_paths:
        raise FirstWeekOperationError(
            "recovery worktree contains changes beyond the exact decision evidence: "
            f"{sorted(dirty_paths - allowed_dirty_paths)}"
        )
    return {
        **current_core,
        "worktree_clean": not dirty_entries,
        "allowed_dirty_paths": sorted(dirty_paths),
    }


def _file_inventory(root: Path, subdirectory: str) -> list[dict[str, Any]]:
    directory = root / subdirectory
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def ledger_closure(root: Path) -> dict[str, Any]:
    """Hash both successful records and preserved failure attempts."""

    records = stage3.scan_records(root)
    record_files = _file_inventory(root, "records")
    failure_files = _file_inventory(root, "failures")
    material = {
        "records": record_files,
        "failures": failure_files,
    }
    return {
        "record_count": len(records),
        "head": None if not records else records[-1].data["record_hash"],
        "head_type": None if not records else records[-1].data["record_type"],
        "record_files": record_files,
        "failure_files": failure_files,
        "closure_sha256": sha256_bytes(canonical_json(material)),
    }


def assert_ledger_unchanged(
    before: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    after = ledger_closure(root)
    if canonical_json(before) != canonical_json(after):
        raise FirstWeekOperationError("a non-ledger operation changed records or preserved failures")
    return after


@contextmanager
def operator_lock(root: Path) -> Iterator[None]:
    """Serialize first-week operators before taking the raw or ledger lock."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / OPERATOR_LOCK_FILE
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def authorization_ledger_lock(root: Path) -> Iterator[None]:
    """Hold the physical ledger lock and expose that fact only in this context."""

    resolved = root.expanduser().resolve()
    if _active_authorization_lock_lease() is not None or _physical_authorization_lock_owned(resolved):
        raise FirstWeekOperationError("authorization ledger lock cannot be nested")
    with stage3._exclusive_lock(root):
        physical_locks = _physical_authorization_lock_registry()
        owner = (os.getpid(), threading.get_ident())
        physical_locks[resolved] = owner
        lease = _AuthorizationLockLease(resolved)
        token = _AUTHORIZED_LEDGER_LOCK_LEASE.set(lease)
        try:
            yield
        finally:
            lease.active = False
            try:
                _AUTHORIZED_LEDGER_LOCK_LEASE.reset(token)
            finally:
                if physical_locks.pop(resolved, None) != owner:
                    raise FirstWeekOperationError("authorization physical lock ownership changed unexpectedly")


def _active_authorization_lock_lease() -> _AuthorizationLockLease | None:
    lease = _AUTHORIZED_LEDGER_LOCK_LEASE.get()
    if lease is not None and lease.pid != os.getpid():
        raise FirstWeekOperationError(
            "a forked process inherited an authorization lock lease; Stage 3 writes are forbidden"
        )
    if lease is None or not lease.active or lease.thread_id != threading.get_ident():
        return None
    return lease


def _physical_authorization_lock_registry() -> dict[Path, tuple[int, int]]:
    registry = getattr(_AUTHORIZATION_PHYSICAL_LOCKS, "roots", None)
    if registry is None:
        registry = {}
        _AUTHORIZATION_PHYSICAL_LOCKS.roots = registry
    return registry


def _physical_authorization_lock_owned(root: Path) -> bool:
    owner = _physical_authorization_lock_registry().get(root.expanduser().resolve())
    if owner is None:
        return False
    current = (os.getpid(), threading.get_ident())
    if owner[0] != current[0]:
        raise FirstWeekOperationError(
            "a forked process inherited physical Stage 3 lock ownership; writes are forbidden"
        )
    return owner == current


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_stage3_writes(root: Path) -> Iterator[None]:
    """Serialize temporary Stage 3 monkeypatches within this Python process."""

    with _STAGE3_PATCH_LOCK, _atomic_stage3_writes_locked(root):
        yield


@contextmanager
def _atomic_stage3_writes_locked(root: Path) -> Iterator[None]:
    """Make every frozen collector write complete-or-absent during this operation."""

    root.mkdir(parents=True, exist_ok=True)
    staging = root / ATOMIC_STAGING_DIR
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_descriptor = os.open(root, directory_flags)
    staging_descriptor: int | None = None
    try:
        with suppress(FileExistsError):
            os.mkdir(ATOMIC_STAGING_DIR, mode=0o700, dir_fd=root_descriptor)
        try:
            staging_descriptor = os.open(
                ATOMIC_STAGING_DIR,
                directory_flags,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise FirstWeekOperationError(
                f"atomic staging directory cannot be a symlink or non-directory: {staging}"
            ) from exc
        staging_stat = os.fstat(staging_descriptor)
        if not stat.S_ISDIR(staging_stat.st_mode):
            raise FirstWeekOperationError(f"atomic staging path is not a directory: {staging}")
        os.fsync(staging_descriptor)
        os.fsync(root_descriptor)
    except Exception:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(root_descriptor)
        raise
    assert staging_descriptor is not None

    def clear_known_staging_files() -> None:
        for name in sorted(os.listdir(staging_descriptor)):
            entry_stat = os.stat(
                name,
                dir_fd=staging_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(entry_stat.st_mode) or not name.startswith("write_"):
                raise FirstWeekOperationError(f"unexpected atomic staging entry: {staging / name}")
            os.unlink(name, dir_fd=staging_descriptor)

    try:
        clear_known_staging_files()
        os.fsync(staging_descriptor)
    except Exception:
        os.close(staging_descriptor)
        os.close(root_descriptor)
        raise

    original_write = stage3._exclusive_write
    owner = (os.getpid(), threading.get_ident())

    def atomic_exclusive_write(path: Path, raw: bytes) -> None:
        if (os.getpid(), threading.get_ident()) != owner:
            original_write(path, raw)
            return
        parent_existed = path.parent.is_dir()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            _fsync_directory(path.parent)
            _fsync_directory(path.parent.parent)
        if path.exists():
            raise stage3.Stage3ConflictError(f"immutable file already exists: {path}")
        if path.parent.stat().st_dev != staging_stat.st_dev:
            raise FirstWeekOperationError(f"atomic staging and final path are on different filesystems: {path}")
        temporary_name = f"write_{secrets.token_hex(16)}"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=staging_descriptor,
            )
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as file:
                descriptor = None
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            os.fsync(staging_descriptor)
            try:
                os.link(
                    temporary_name,
                    path,
                    src_dir_fd=staging_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise stage3.Stage3ConflictError(f"immutable file already exists: {path}") from exc
            _fsync_directory(path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=staging_descriptor)
            os.fsync(staging_descriptor)

    stage3._exclusive_write = atomic_exclusive_write
    try:
        yield
    finally:
        stage3._exclusive_write = original_write
        try:
            clear_known_staging_files()
            os.fsync(staging_descriptor)
            current_stat = os.stat(
                ATOMIC_STAGING_DIR,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if (
                current_stat.st_dev != staging_stat.st_dev
                or current_stat.st_ino != staging_stat.st_ino
                or not stat.S_ISDIR(current_stat.st_mode)
            ):
                raise FirstWeekOperationError(f"atomic staging directory entry changed while it was open: {staging}")
            try:
                os.rmdir(ATOMIC_STAGING_DIR, dir_fd=root_descriptor)
            except OSError as exc:
                raise FirstWeekOperationError(f"atomic staging directory could not be removed: {staging}") from exc
            os.fsync(root_descriptor)
        finally:
            try:
                os.close(staging_descriptor)
            finally:
                os.close(root_descriptor)


def _as_utc(now: datetime | None = None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        raise FirstWeekOperationError("operation clock must be timezone-aware")
    return value.astimezone(UTC)


def validate_frozen_data_dir(data_dir: Path) -> Path:
    resolved = data_dir.expanduser().resolve()
    frozen = stage3.RAW_DIR.expanduser().resolve()
    if resolved != frozen:
        raise FirstWeekOperationError(
            f"Stage 3 collector reads only its frozen raw directory; expected {frozen}, got {resolved}"
        )
    return resolved


def _decision_close(decision_date: str) -> datetime:
    return datetime.combine(
        pd.Timestamp(decision_date).date(),
        time(15, 0),
        tzinfo=EXCHANGE_TIMEZONE,
    )


def _entry_open(entry_date: str) -> datetime:
    return datetime.combine(
        pd.Timestamp(entry_date).date(),
        time(9, 30),
        tzinfo=EXCHANGE_TIMEZONE,
    )


def _daily_release(decision_date: str) -> datetime:
    return datetime.combine(
        pd.Timestamp(decision_date).date(),
        time(DAILY_BAR_READY_HOUR, 0),
        tzinfo=MARKET_TIMEZONE,
    )


def validate_prepare_time(
    decision_date: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _as_utc(now)
    release = _daily_release(decision_date).astimezone(UTC)
    if current < release:
        raise FirstWeekOperationError(f"daily data preparation is forbidden before {release.isoformat()}")
    return {
        "checked_at_utc": current,
        "daily_release_not_before_utc": release,
    }


def validate_apply_window(
    decision_date: str,
    entry_date: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _as_utc(now)
    lower = _decision_close(decision_date).astimezone(UTC)
    upper = _entry_open(entry_date).astimezone(UTC)
    latest_safe_start = upper - ANCHOR_PUSH_RESERVE
    if not (lower <= current < latest_safe_start):
        raise FirstWeekOperationError(
            f"decision apply must start inside [{lower.isoformat()}, {latest_safe_start.isoformat()})"
        )
    return {
        "checked_at_utc": current,
        "decision_close_utc": lower,
        "entry_open_utc": upper,
        "latest_safe_start_utc": latest_safe_start,
        "anchor_push_reserve_seconds": int(ANCHOR_PUSH_RESERVE.total_seconds()),
    }


def validate_recovery_deadline(
    entry_date: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Allow a pre-authorized crash recovery until, but never at, entry open."""

    current = _as_utc(now)
    entry_open = _entry_open(entry_date).astimezone(UTC)
    if current >= entry_open:
        raise FirstWeekOperationError(
            f"decision evidence recovery must finish before entry open {entry_open.isoformat()}"
        )
    return {
        "checked_at_utc": current,
        "entry_open_utc": entry_open,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = stage3.read_json(path)
    except Stage3ValidationError as exc:
        raise FirstWeekOperationError(f"cannot read JSON evidence: {path}") from exc
    return payload


def _trade_date_text(value: Any) -> str:
    text = str(value).strip()
    try:
        parsed = (
            pd.to_datetime(text, format="%Y%m%d", errors="raise")
            if re.fullmatch(r"\d{8}", text)
            else pd.Timestamp(value)
        )
    except (TypeError, ValueError) as exc:
        raise FirstWeekOperationError(f"invalid daily-basic trade date: {value!r}") from exc
    return pd.Timestamp(parsed).date().isoformat()


def _symbol_set_evidence(symbols: set[str]) -> dict[str, Any]:
    normalized = sorted(map(str, symbols))
    return {
        "count": len(normalized),
        "sha256": sha256_bytes(canonical_json(normalized)),
    }


def _iter_frozen_response_records(audit: Mapping[str, Any]) -> Iterator[dict[str, str]]:
    api_objects = audit.get("api_response_objects")
    if not isinstance(api_objects, Mapping):
        raise FirstWeekOperationError("raw audit has no API response object registry")
    for value in api_objects.values():
        if isinstance(value, Mapping) and "path" in value and "sha256" in value:
            yield {"path": str(value["path"]), "sha256": str(value["sha256"])}
            continue
        if not isinstance(value, Mapping):
            raise FirstWeekOperationError("raw API response object registry is malformed")
        for record in value.values():
            if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:
                raise FirstWeekOperationError("raw API response object record is malformed")
            yield {"path": str(record["path"]), "sha256": str(record["sha256"])}
    full_qfq = audit.get("full_qfq_refresh_objects")
    if not isinstance(full_qfq, list):
        raise FirstWeekOperationError("raw audit has no full-qfq object registry")
    for record in full_qfq:
        if not isinstance(record, Mapping) or "object_path" not in record or "object_sha256" not in record:
            raise FirstWeekOperationError("full-qfq object record is malformed")
        yield {
            "path": str(record["object_path"]),
            "sha256": str(record["object_sha256"]),
        }


def _verify_snapshot_v2(
    snapshot: Mapping[str, Any],
    *,
    expected_inventory: Mapping[str, Any],
    expected_snapshot_root: Path,
) -> dict[str, Any]:
    path = Path(str(snapshot.get("path", "")))
    if not path.is_dir():
        raise FirstWeekOperationError(f"raw snapshot directory is missing: {path}")
    if path.resolve().parent != expected_snapshot_root.resolve():
        raise FirstWeekOperationError("raw snapshot path is outside the execution-bound snapshot root")
    manifest_path = path / SNAPSHOT_MANIFEST_FILE_NAME
    manifest = _read_json(manifest_path)
    payload_files = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.iterdir())
        if item.is_file() and item.name != SNAPSHOT_MANIFEST_FILE_NAME
    ]
    payload_sha = sha256_bytes(canonical_json(payload_files))
    expected_parquet_closure = str(expected_inventory["content_inventory_sha256"])
    parquet_payload_files = [record for record in payload_files if str(record["name"]).endswith(".parquet")]
    expected_parquet_files = [
        {
            "name": str(record["name"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in expected_inventory["records"]
    ]
    auxiliary_files = [record for record in payload_files if not str(record["name"]).endswith(".parquet")]
    if (
        manifest.get("schema") != "a_stock_daily_qfq_raw_snapshot_v2"
        or path.name != f"RAW_{payload_sha}"
        or manifest.get("closure_sha256") != payload_sha
        or manifest.get("payload_closure_sha256") != payload_sha
        or manifest.get("parquet_content_inventory_sha256") != expected_parquet_closure
        or int(manifest.get("file_count", -1)) != int(expected_inventory["file_count"])
        or manifest.get("min_dt") != expected_inventory["min_dt"]
        or manifest.get("max_dt") != expected_inventory["max_dt"]
        or manifest.get("columns") != expected_inventory["columns"]
        or canonical_json(manifest.get("files")) != canonical_json(expected_inventory["records"])
        or canonical_json(manifest.get("auxiliary_files")) != canonical_json(auxiliary_files)
        or canonical_json(parquet_payload_files) != canonical_json(expected_parquet_files)
        or snapshot.get("closure_sha256") != payload_sha
        or snapshot.get("parquet_content_inventory_sha256") != expected_parquet_closure
    ):
        raise FirstWeekOperationError("raw v2 snapshot identity differs from its bytes")
    return {
        "name": path.name,
        "payload_closure_sha256": payload_sha,
        "parquet_content_inventory_sha256": expected_parquet_closure,
    }


def validate_raw_ready(
    decision_date: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Bind the active raw cache to one fully verified v2 publication audit."""

    resolved_data = data_dir.expanduser().resolve()
    inventory = inspect_inventory(resolved_data)
    target = pd.Timestamp(decision_date).strftime("%Y%m%d")
    if inventory["max_dt"] != target:
        raise FirstWeekOperationError(f"active raw ends at {inventory['max_dt']}; expected exactly {target}")
    if not active_manifest_matches(resolved_data, inventory, target):
        raise FirstWeekOperationError("active raw manifest does not bind the exact target closure")

    audit_dir = resolved_data.parent / AUDIT_DIR_NAME
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(audit_dir.glob("*.json")):
        if path.stem != sha256_file(path):
            raise FirstWeekOperationError(f"raw audit filename hash mismatch: {path}")
        audit = _read_json(path)
        if (
            audit.get("schema") == "a_stock_daily_qfq_sync_audit_v2"
            and audit.get("safe_end_date") == target
            and Path(str(audit.get("data_dir", ""))).resolve() == resolved_data
            and audit.get("after_inventory", {}).get("content_inventory_sha256")
            == inventory["content_inventory_sha256"]
            and audit.get("expected_active_inventory_sha256") == inventory["content_inventory_sha256"]
        ):
            matching.append((path, audit))
    if len(matching) != 1:
        raise FirstWeekOperationError(f"expected one exact v2 raw audit for {target}, found {len(matching)}")
    audit_path, audit = matching[0]
    inventory_summary = {key: value for key, value in inventory.items() if key != "records"}
    if canonical_json(audit.get("after_inventory")) != canonical_json(inventory_summary):
        raise FirstWeekOperationError("raw audit after-inventory does not equal the complete active cache")
    api_objects = audit.get("api_response_objects")
    trade_dates = audit.get("trade_dates")
    full_qfq_objects = audit.get("full_qfq_refresh_objects")
    if (
        not isinstance(api_objects, Mapping)
        or set(api_objects) != {"official_calendar", "daily", "adj_factor"}
        or not isinstance(trade_dates, list)
        or not trade_dates
        or not isinstance(api_objects.get("daily"), Mapping)
        or not isinstance(api_objects.get("adj_factor"), Mapping)
        or set(map(str, api_objects["daily"])) != set(map(str, trade_dates))
        or set(map(str, api_objects["adj_factor"])) != set(map(str, trade_dates))
        or not isinstance(full_qfq_objects, list)
        or int(audit.get("full_qfq_refresh_symbol_count", -1)) != len(full_qfq_objects)
    ):
        raise FirstWeekOperationError("raw audit does not freeze the exact calendar/daily/factor/full-qfq response set")

    execution_binding = audit.get("execution_binding")
    if not isinstance(execution_binding, Mapping):
        raise FirstWeekOperationError("raw audit execution binding is missing")
    try:
        assert_execution_binding_current(
            execution_binding,
            require_source_completeness=True,
        )
    except Exception as exc:
        raise FirstWeekOperationError(f"raw audit execution binding is no longer current: {exc}") from exc
    binding = execution_binding.get("binding")
    if not isinstance(binding, Mapping):
        raise FirstWeekOperationError("raw execution binding payload is malformed")
    parameters = binding.get("parameters")
    if not isinstance(parameters, Mapping):
        raise FirstWeekOperationError("raw execution parameters are malformed")
    input_evidence = binding.get("input_evidence")
    source_completeness = audit.get("source_symbol_completeness")
    target_symbols = {
        Path(str(record["name"])).stem
        for record in inventory["records"]
        if _trade_date_text(record.get("max_dt")) == pd.Timestamp(decision_date).date().isoformat()
    }
    target_symbol_evidence = _symbol_set_evidence(target_symbols)
    if (
        not isinstance(input_evidence, Mapping)
        or not isinstance(source_completeness, Mapping)
        or canonical_json(input_evidence.get("source_symbol_completeness")) != canonical_json(source_completeness)
        or source_completeness.get("passed") is not True
        or source_completeness.get("report_sha256")
        != sha256_bytes(
            canonical_json({key: value for key, value in source_completeness.items() if key != "report_sha256"})
        )
        or not isinstance(source_completeness.get("sessions"), list)
        or not source_completeness["sessions"]
        or source_completeness["sessions"][-1].get("trade_date") != target
        or canonical_json(source_completeness["sessions"][-1].get("daily_symbols"))
        != canonical_json(target_symbol_evidence)
    ):
        raise FirstWeekOperationError("raw audit does not bind a valid complete target-universe report")
    if (
        parameters.get("apply") is not True
        or parameters.get("safe_end_date") != target
        or str(parameters.get("requested_end_date")) != target
        or int(parameters.get("minimum_daily_rows", -1)) != DEFAULT_MIN_DAILY_ROWS
        or Path(str(parameters.get("data_dir", ""))).resolve() != resolved_data
    ):
        raise FirstWeekOperationError("raw execution parameters do not describe this formal active cache")
    snapshot_root = Path(str(parameters.get("snapshot_root", ""))).resolve()

    object_count = 0
    for record in _iter_frozen_response_records(audit):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise FirstWeekOperationError(f"frozen raw input object differs from its audit: {path}")
        object_count += 1
    new_snapshot = _verify_snapshot_v2(
        audit.get("new_raw_snapshot", {}),
        expected_inventory=inventory,
        expected_snapshot_root=snapshot_root,
    )
    active_manifest_path = resolved_data / "manifest.json"
    return {
        "data_dir": str(resolved_data),
        "target_date": target,
        "file_count": inventory["file_count"],
        "min_dt": inventory["min_dt"],
        "max_dt": inventory["max_dt"],
        "parquet_inventory_sha256": inventory["content_inventory_sha256"],
        "active_manifest_sha256": sha256_file(active_manifest_path),
        "audit_sha256": audit_path.stem,
        "audit_path": str(audit_path),
        "execution_binding_sha256": execution_binding["binding_sha256"],
        "source_completeness_sha256": source_completeness["report_sha256"],
        "target_daily_symbols": target_symbol_evidence,
        "target_raw_adj_factor_symbol_count": source_completeness["sessions"][-1]["raw_adj_factor_symbols"]["count"],
        "frozen_object_count": object_count,
        "new_snapshot": new_snapshot,
    }


def validate_state_ready(
    state_manifest_path: Path,
    decision_date: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Validate cache identity, engine, raw binding, projection and 100x20 audit."""

    manifest_path = state_manifest_path.expanduser().resolve()
    cache_dir = manifest_path.parent
    projection_path = cache_dir / "states.parquet"
    audit_path = cache_dir / "state_audit.json"
    audit_detail_path = cache_dir / "state_audit.parquet"
    for path in (
        manifest_path,
        projection_path,
        audit_path,
        audit_detail_path,
    ):
        if not path.is_file():
            raise FirstWeekOperationError(f"state cache artifact is missing: {path}")

    manifest = _read_json(manifest_path)
    cache_id = str(manifest.get("cache_id", ""))
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise FirstWeekOperationError("state cache identity is missing")
    expected_config = {
        "audit_symbols": state_cache.DEFAULT_AUDIT_SYMBOLS,
        "audit_checkpoints": state_cache.DEFAULT_AUDIT_CHECKPOINTS,
        "audit_seed": state_cache.DEFAULT_AUDIT_SEED,
        "warmup_bars": state_cache.WARMUP_BARS,
    }
    if manifest.get("config") != expected_config:
        raise FirstWeekOperationError("formal state cache must use the frozen 100x20 audit configuration")
    current_engine = state_cache._engine_provenance()
    if canonical_json(manifest.get("engine")) != canonical_json(current_engine):
        raise FirstWeekOperationError("state cache engine bytes or runtime differ from the current operator")

    inventory = inspect_inventory(data_dir.expanduser().resolve())
    expected_sources = [
        {
            "name": str(record["name"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in inventory["records"]
    ]
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list):
        raise FirstWeekOperationError("state manifest source inventory is missing")
    actual_sources = [
        {
            "name": str(record.get("name")),
            "size": int(record.get("size", -1)),
            "sha256": str(record.get("sha256")),
        }
        for record in manifest_sources
        if isinstance(record, Mapping)
    ]
    if actual_sources != expected_sources:
        raise FirstWeekOperationError("state cache sources do not equal the complete active raw inventory")
    expected_cache_id, expected_identity = state_cache._cache_identity(
        state_cache.StateCacheConfig(),
        current_engine,
        expected_sources,
    )
    identity_sha = hashlib.sha256(state_cache._canonical_json(identity)).hexdigest()
    if (
        manifest.get("schema_version") != state_cache.SCHEMA_VERSION
        or canonical_json(identity) != canonical_json(expected_identity)
        or manifest.get("content_digest_sha256") != identity_sha
        or cache_id != identity_sha
        or cache_id != expected_cache_id
        or cache_dir.name != f"CHAN_STATE_CACHE_{cache_id}"
    ):
        raise FirstWeekOperationError("state cache identity does not bind its exact engine, config and raw sources")

    projection_meta = manifest.get("projection")
    if not isinstance(projection_meta, Mapping):
        raise FirstWeekOperationError("state projection metadata is missing")
    physical_rows = int(pq.ParquetFile(projection_path).metadata.num_rows)
    physical_schema = pq.read_schema(projection_path)
    expected_projection_schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("dt", pa.timestamp("ns"), nullable=False),
            pa.field("regime", pa.int8(), nullable=False),
        ]
    )
    if (
        projection_meta.get("path") != "states.parquet"
        or projection_meta.get("sha256") != sha256_file(projection_path)
        or int(projection_meta.get("rows", -1)) != physical_rows
        or tuple(projection_meta.get("columns", ())) != state_cache.STATE_COLUMNS
        or physical_schema != expected_projection_schema
    ):
        raise FirstWeekOperationError("state projection bytes, rows or schema differ from the manifest")

    target = pd.Timestamp(decision_date).normalize().to_pydatetime()
    projection_scan = pl.scan_parquet(projection_path)
    projection_summary = (
        projection_scan.select(
            pl.len().alias("rows"),
            pl.struct("symbol", "dt").n_unique().alias("unique_keys"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("dt").max().alias("maximum_dt"),
            pl.col("regime").min().alias("minimum_regime"),
            pl.col("regime").max().alias("maximum_regime"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    expected_symbols = {Path(record["name"]).stem for record in expected_sources}
    projection_symbols = set(
        projection_scan.select(pl.col("symbol").unique()).collect(engine="streaming").get_column("symbol").to_list()
    )
    if (
        int(projection_summary["rows"]) != physical_rows
        or int(projection_summary["unique_keys"]) != physical_rows
        or int(projection_summary["symbols"]) != len(expected_sources)
        or projection_symbols != expected_symbols
        or pd.Timestamp(projection_summary["maximum_dt"]).normalize() != pd.Timestamp(decision_date)
        or int(projection_summary["minimum_regime"]) < min(state_cache.VALID_REGIMES)
        or int(projection_summary["maximum_regime"]) > max(state_cache.VALID_REGIMES)
        or int(projection_meta.get("symbols", -1)) != len(expected_sources)
    ):
        raise FirstWeekOperationError("state projection coverage, uniqueness or value domain is invalid")

    target_summary = (
        projection_scan.filter(pl.col("dt") == pl.lit(target))
        .select(
            pl.len().alias("rows"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("regime").min().alias("minimum_regime"),
            pl.col("regime").max().alias("maximum_regime"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if (
        int(target_summary["rows"]) < DEFAULT_MIN_DAILY_ROWS
        or target_summary["rows"] != target_summary["symbols"]
        or int(target_summary["minimum_regime"]) < min(state_cache.VALID_REGIMES)
        or int(target_summary["maximum_regime"]) > max(state_cache.VALID_REGIMES)
    ):
        raise FirstWeekOperationError("state projection lacks a complete unique target-date cross-section")

    audit_info = manifest.get("state_audit")
    audit_detail_info = manifest.get("state_audit_details")
    if (
        not isinstance(audit_info, Mapping)
        or audit_info.get("path") != "state_audit.json"
        or audit_info.get("sha256") != sha256_file(audit_path)
        or not isinstance(audit_detail_info, Mapping)
        or audit_detail_info.get("path") != "state_audit.parquet"
        or audit_detail_info.get("sha256") != sha256_file(audit_detail_path)
    ):
        raise FirstWeekOperationError("state audit objects differ from the source manifest")
    audit = _read_json(audit_path)
    prefix = audit.get("prefix_full_audit")
    if not isinstance(prefix, Mapping):
        raise FirstWeekOperationError("state prefix/full audit is missing")
    physical_audit_rows = int(pq.ParquetFile(audit_detail_path).metadata.num_rows)
    physical_audit_schema = pq.read_schema(audit_detail_path)
    audit_details = pl.read_parquet(audit_detail_path)
    audit_symbol_counts = audit_details.group_by("symbol").len().get_column("len").to_list()
    zero_mismatch = sum(
        int(audit_details.get_column(column).sum())
        for column in (
            "symbol_mismatches",
            "dt_mismatches",
            "regime_mismatches",
        )
    )
    input_summary = audit.get("input_summary")
    if (
        audit.get("schema_version") != state_cache.SCHEMA_VERSION
        or audit.get("cache_id") != cache_id
        or audit.get("passed") is not True
        or audit.get("data_gate_ready") is not True
        or int(audit.get("mismatch_count", -1)) != 0
        or prefix.get("passed") is not True
        or int(prefix.get("configured_symbols", -1)) != state_cache.DEFAULT_AUDIT_SYMBOLS
        or int(prefix.get("configured_checkpoints_per_symbol", -1)) != state_cache.DEFAULT_AUDIT_CHECKPOINTS
        or int(prefix.get("configured_seed", -1)) != state_cache.DEFAULT_AUDIT_SEED
        or int(prefix.get("audited_symbols", -1)) != state_cache.DEFAULT_AUDIT_SYMBOLS
        or int(prefix.get("expected_comparisons", -1))
        != state_cache.DEFAULT_AUDIT_SYMBOLS * state_cache.DEFAULT_AUDIT_CHECKPOINTS
        or int(prefix.get("comparison_count", -1))
        != state_cache.DEFAULT_AUDIT_SYMBOLS * state_cache.DEFAULT_AUDIT_CHECKPOINTS
        or int(prefix.get("mismatch_count", -1)) != 0
        or prefix.get("errors") != []
        or prefix.get("field_mismatches") != dict.fromkeys(state_cache.STATE_COLUMNS, 0)
        or not isinstance(input_summary, Mapping)
        or int(input_summary.get("source_files", -1)) != len(expected_sources)
        or int(input_summary.get("generated_files", -1)) != len(expected_sources)
        or int(input_summary.get("skipped_files", -1)) != 0
        or int(input_summary.get("validation_errors", -1)) != 0
        or int(audit_detail_info.get("rows", -1))
        != state_cache.DEFAULT_AUDIT_SYMBOLS * state_cache.DEFAULT_AUDIT_CHECKPOINTS
        or physical_audit_rows != state_cache.DEFAULT_AUDIT_SYMBOLS * state_cache.DEFAULT_AUDIT_CHECKPOINTS
        or tuple(audit_detail_info.get("columns", ())) != state_cache.AUDIT_DETAIL_COLUMNS
        or physical_audit_schema != state_cache._audit_detail_schema()
        or audit_detail_info.get("schema")
        != {field.name: str(field.type) for field in state_cache._audit_detail_schema()}
        or canonical_json(audit.get("projection")) != canonical_json(projection_meta)
        or canonical_json(audit.get("audit_details")) != canonical_json(audit_detail_info)
        or not bool(audit_details.get_column("passed").all())
        or zero_mismatch != 0
        or not bool((audit_details.get_column("expected_sha256") == audit_details.get_column("actual_sha256")).all())
        or len(audit_symbol_counts) != state_cache.DEFAULT_AUDIT_SYMBOLS
        or set(map(int, audit_symbol_counts)) != {state_cache.DEFAULT_AUDIT_CHECKPOINTS}
    ):
        raise FirstWeekOperationError("state cache did not pass the exact 100x20 zero-mismatch gate")
    return {
        "cache_id": cache_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "projection_path": str(projection_path),
        "projection_sha256": projection_meta["sha256"],
        "projection_rows": physical_rows,
        "target_date": pd.Timestamp(decision_date).date().isoformat(),
        "target_rows": int(target_summary["rows"]),
        "source_file_count": len(expected_sources),
        "raw_parquet_inventory_sha256": inventory["content_inventory_sha256"],
        "audit_sha256": audit_info["sha256"],
        "audit_comparisons": prefix["comparison_count"],
        "audit_mismatches": 0,
    }


def validate_reference_symbol_completeness(
    daily: Mapping[str, pd.DataFrame],
    *,
    raw_summary: Mapping[str, Any],
    decision_date: str,
) -> dict[str, Any]:
    """Reject a truncated daily-basic bridge before any path is frozen."""

    previous_symbols: set[str] | None = None
    sessions: list[dict[str, Any]] = []
    for date_text in FIRST_WEEK_REFERENCE_DATES:
        frame = daily[date_text]
        symbols = set(frame["ts_code"].astype(str))
        if len(symbols) < REFERENCE_SYMBOL_ABSOLUTE_MINIMUM:
            raise FirstWeekOperationError(
                f"daily_basic[{date_text}] has only {len(symbols)} unique symbols; "
                f"fixed completeness minimum is {REFERENCE_SYMBOL_ABSOLUTE_MINIMUM}"
            )
        session: dict[str, Any] = {
            "trade_date": date_text,
            "symbols": _symbol_set_evidence(symbols),
            "passed": True,
        }
        if previous_symbols is not None:
            retained = previous_symbols & symbols
            required_retained = math.ceil(len(previous_symbols) * REFERENCE_SYMBOL_MIN_PREVIOUS_COVERAGE)
            added = symbols - previous_symbols
            removed = previous_symbols - symbols
            symmetric_count = len(added) + len(removed)
            maximum_symmetric_count = math.floor(len(previous_symbols) * REFERENCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE)
            if len(retained) < required_retained:
                raise FirstWeekOperationError(
                    f"daily_basic[{date_text}] retained only {len(retained)}/{len(previous_symbols)} "
                    "symbols from the previous registered bridge date"
                )
            if symmetric_count > maximum_symmetric_count:
                raise FirstWeekOperationError(
                    f"daily_basic[{date_text}] changed {symmetric_count}/{len(previous_symbols)} "
                    "symbols versus the previous registered bridge date"
                )
            session.update(
                {
                    "previous_symbols": _symbol_set_evidence(previous_symbols),
                    "retained_from_previous": len(retained),
                    "required_retained_from_previous": required_retained,
                    "previous_coverage": len(retained) / len(previous_symbols),
                    "added_since_previous": _symbol_set_evidence(added),
                    "removed_since_previous": _symbol_set_evidence(removed),
                    "symmetric_change_count": symmetric_count,
                    "maximum_symmetric_change_count": maximum_symmetric_count,
                    "symmetric_change_share": symmetric_count / len(previous_symbols),
                }
            )
        sessions.append(session)
        previous_symbols = symbols

    inventory = inspect_inventory(Path(str(raw_summary["data_dir"])))
    target_iso = pd.Timestamp(decision_date).date().isoformat()
    raw_target_symbols = {
        Path(str(record["name"])).stem
        for record in inventory["records"]
        if _trade_date_text(record.get("max_dt")) == target_iso
    }
    if not raw_target_symbols:
        raise FirstWeekOperationError("raw active cache has no symbols on the formal target session")
    raw_target_evidence = _symbol_set_evidence(raw_target_symbols)
    if canonical_json(raw_summary.get("target_daily_symbols")) != canonical_json(raw_target_evidence):
        raise FirstWeekOperationError("raw target symbol evidence changed during reference validation")
    target_symbols = set(daily[target_iso]["ts_code"].astype(str))
    raw_missing_from_daily_basic = raw_target_symbols - target_symbols
    daily_basic_only = target_symbols - raw_target_symbols
    maximum_daily_basic_only = math.floor(len(raw_target_symbols) * REFERENCE_SYMBOL_MAX_RAW_EXTRA_SHARE)
    if raw_missing_from_daily_basic:
        raise FirstWeekOperationError(
            "target daily-basic response omits symbols present in the exact raw daily universe: "
            f"{len(raw_missing_from_daily_basic)} missing"
        )
    if len(daily_basic_only) > maximum_daily_basic_only:
        raise FirstWeekOperationError(
            "target daily-basic response has an abnormal symbol expansion versus raw daily: "
            f"{len(daily_basic_only)}/{len(raw_target_symbols)}"
        )
    target_raw_comparison = {
        "raw_daily_symbols": raw_target_evidence,
        "daily_basic_symbols": _symbol_set_evidence(target_symbols),
        "raw_missing_from_daily_basic": _symbol_set_evidence(raw_missing_from_daily_basic),
        "daily_basic_only": _symbol_set_evidence(daily_basic_only),
        "maximum_daily_basic_only": maximum_daily_basic_only,
        "passed": True,
    }
    rule = {
        "schema": "xs_chan_stage3_daily_basic_symbol_completeness_rule_v1",
        "absolute_minimum_symbols": REFERENCE_SYMBOL_ABSOLUTE_MINIMUM,
        "minimum_previous_bridge_coverage": REFERENCE_SYMBOL_MIN_PREVIOUS_COVERAGE,
        "maximum_previous_bridge_symmetric_change_share": (REFERENCE_SYMBOL_MAX_SYMMETRIC_CHANGE_SHARE),
        "target_raw_daily_subset_required": True,
        "maximum_target_daily_basic_only_share": REFERENCE_SYMBOL_MAX_RAW_EXTRA_SHARE,
    }
    report_without_hash = {
        "schema": "xs_chan_stage3_daily_basic_symbol_completeness_v1",
        "rule": {
            **rule,
            "rule_sha256": sha256_bytes(canonical_json(rule)),
        },
        "sessions": sessions,
        "target_raw_comparison": target_raw_comparison,
        "passed": True,
    }
    return {
        **report_without_hash,
        "report_sha256": sha256_bytes(canonical_json(report_without_hash)),
    }


def validate_reference_ready(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    decision_date: str,
    *,
    state_summary: Mapping[str, Any],
    raw_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact eight-date reference and simulate the full bridge."""

    resolved_reference = reference_manifest_path.expanduser().resolve()
    resolved_state = state_manifest_path.expanduser().resolve()
    manifest, official_sessions, daily = stage3.load_reference_manifest(
        root,
        resolved_reference,
    )
    local_inputs = manifest.get("local_inputs")
    if not isinstance(local_inputs, Mapping):
        raise FirstWeekOperationError("reference local-input binding is missing")
    if (
        Path(str(local_inputs.get("state_manifest_path", ""))).resolve() != resolved_state
        or local_inputs.get("state_manifest_sha256") != state_summary["manifest_sha256"]
        or local_inputs.get("state_projection_sha256") != state_summary["projection_sha256"]
    ):
        raise FirstWeekOperationError("reference manifest does not bind the explicitly selected state cache")

    closure = stage3._load_content_object(
        root,
        "raw_source_closure",
        local_inputs.get("raw_source_closure_sha256"),
        local_inputs.get("raw_source_closure_object"),
    )
    try:
        stage3.verify_raw_source_closure_current(closure)
    except Stage3ValidationError as exc:
        raise FirstWeekOperationError(f"reference raw closure is no longer current: {exc}") from exc
    if (
        pd.Timestamp(str(closure.get("coverage_max_dt"))) < pd.Timestamp(decision_date)
        or int(closure.get("file_count", -1)) != int(raw_summary["file_count"])
        or {str(row["name"]): str(row["sha256"]) for row in closure.get("files", [])}
        != {
            str(row["name"]): str(row["sha256"])
            for row in inspect_inventory(Path(str(raw_summary["data_dir"])))["records"]
        }
    ):
        raise FirstWeekOperationError("reference raw closure does not cover and equal the active target cache")

    bridge_dates = tuple(
        value.date().isoformat()
        for value in stage3._official_weekly_decisions(
            official_sessions,
            BRIDGE_START_EXCLUSIVE,
            decision_date,
        )
    )
    if bridge_dates != FIRST_WEEK_REFERENCE_DATES or tuple(sorted(daily)) != FIRST_WEEK_REFERENCE_DATES:
        raise FirstWeekOperationError("formal reference must contain exactly the registered eight bridge dates")
    for date_text in FIRST_WEEK_REFERENCE_DATES:
        frame = daily[date_text]
        if len(frame) < DEFAULT_MIN_DAILY_ROWS:
            raise FirstWeekOperationError(f"daily_basic[{date_text}] has only {len(frame)} rows")
        if (
            "ts_code" not in frame
            or "trade_date" not in frame
            or frame["ts_code"].astype(str).duplicated().any()
            or {_trade_date_text(value) for value in frame["trade_date"]} != {date_text}
        ):
            raise FirstWeekOperationError(f"daily_basic[{date_text}] has an invalid date or symbol cross-section")
    daily_basic_completeness = validate_reference_symbol_completeness(
        daily,
        raw_summary=raw_summary,
        decision_date=decision_date,
    )

    schedule = stage3.build_forward_schedule(spec, official_sessions)
    target_row = schedule.loc[schedule["decision_dt"].eq(pd.Timestamp(decision_date))]
    if len(target_row) != 1:
        raise FirstWeekOperationError("official calendar does not contain one complete first prospective week")
    schedule_row = target_row.iloc[0]
    entry_date = pd.Timestamp(schedule_row["entry_dt"]).date().isoformat()
    exit_date = pd.Timestamp(schedule_row["exit_dt"]).date().isoformat()
    if int(schedule_row["week_index"]) != 1 or entry_date != FIRST_WEEK_ENTRY_DATE or exit_date != FIRST_WEEK_EXIT_DATE:
        raise FirstWeekOperationError("official first-week entry/exit schedule differs from the registered operation")

    projections = stage3.build_decision_projections(
        spec,
        bridge_dates,
        official_sessions,
        daily,
        manifest,
    )
    path_bundle = stage3.build_fc_path_bundle(
        spec,
        bridge_dates,
        projections,
        initial_symbols=stage3.load_initial_fc_membership(),
        initial_decision_date="2026-06-05",
    )
    weeks = path_bundle.get("weeks")
    if (
        not isinstance(weeks, list)
        or len(weeks) != len(FIRST_WEEK_REFERENCE_DATES)
        or weeks[-1].get("decision_dt") != decision_date
        or stage3._contains_outcome_key(path_bundle)
    ):
        raise FirstWeekOperationError("eight-date bridge simulation is incomplete or contains outcome data")
    return {
        "manifest_path": str(resolved_reference),
        "manifest_sha256": resolved_reference.stem,
        "state_manifest_sha256": state_summary["manifest_sha256"],
        "raw_source_closure_sha256": local_inputs["raw_source_closure_sha256"],
        "daily_basic_completeness": daily_basic_completeness,
        "bridge_dates": list(bridge_dates),
        "bridge_path_sha256": sha256_bytes(canonical_json(path_bundle)),
        "bridge_week_count": len(weeks),
        "proposal_count": sum(len(week.get("proposals", [])) for week in weeks),
        "final_membership_count": len(path_bundle["final_membership"]),
        "decision_date": decision_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "week_index": 1,
        "decision_inputs_only": True,
        "prospective_week_count": 0,
        "formal_ledger_mutated": False,
    }


def _expected_source_replay(
    root: Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    record_type = record["record_type"]
    payload = record["payload"]
    if record_type == "genesis":
        return {
            "status": "PROTOCOL_SOURCE_REPLAYED",
            "protocol_anchor_git_commit": payload["protocol_anchor_git_commit"],
        }
    if record_type == "decision_freeze":
        manifest_path = root / "objects" / "reference_manifest" / f"{payload['reference_manifest_sha256']}.json"
        manifest = stage3.read_json(manifest_path)
        return {
            "status": "DECISION_SOURCE_REPLAYED",
            "raw_source_closure_sha256": manifest["local_inputs"]["raw_source_closure_sha256"],
            "decision_path_sha256": payload["decision_path_sha256"],
        }
    if record_type == "label_completion":
        return {
            "status": "LABEL_SOURCE_REPLAYED",
            "raw_source_closure_sha256": payload["raw_source_closure_sha256"],
            "label_observation_sha256": payload["label_observation_sha256"],
        }
    if record_type == "final_evaluation":
        return {
            "status": "FINAL_EVALUATION_REPLAYED",
            "evaluation_sha256": payload["evaluation_sha256"],
        }
    raise FirstWeekOperationError(f"unsupported ledger record type: {record_type}")


def _anchor_deadline(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
) -> datetime | None:
    if record["record_type"] == "genesis":
        first = spec["historical_exclusion"]["first_prospective_decision_date"]
        return _decision_close(first)
    if record["record_type"] == "decision_freeze":
        return _entry_open(record["payload"]["entry_dt"])
    return None


def validate_anchor_for_record(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
) -> dict[str, Any]:
    """Validate every anchor field, replay digest and applicable deadline."""

    path = (
        stage3.SCRIPTS_DIR
        / "xs_chan_exploration_stage3_ledger_anchors"
        / f"{int(record['sequence']):06d}_{record['record_hash']}.json"
    )
    if not path.is_file():
        raise FirstWeekOperationError(f"ledger anchor is missing: {path}")
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
        "spec_physical_sha256": sha256_file(stage3.SPEC_PATH),
        "source_sha256": sha256_file(stage3.SOURCE_PATH),
        "source_replay": _expected_source_replay(root, record),
        "local_record_path": (f"{int(record['sequence']):06d}_{record['record_hash']}.json"),
        "external_timestamp_or_signature": False,
        "required_follow_up": ("commit_and_push_this_unique_anchor_without_rewriting_prior_anchors"),
    }
    if canonical_json(stage3.read_json(path)) != canonical_json(expected):
        raise FirstWeekOperationError(f"ledger anchor does not equal the complete expected payload: {path}")
    commit: str | None = None
    deadline = _anchor_deadline(spec, record)
    if require_pushed:
        try:
            commit = stage3._validate_tracked_file_pushed(
                path,
                deadline=deadline,
            )
        except Stage3ValidationError as exc:
            raise FirstWeekOperationError(f"ledger anchor is not validly committed and pushed: {exc}") from exc
        if deadline is not None:
            committed_at = datetime.fromisoformat(
                _git_output(
                    stage3.REPO_ROOT,
                    "show",
                    "-s",
                    "--format=%cI",
                    commit,
                )
            )
            if committed_at >= deadline:
                raise FirstWeekOperationError(
                    "ledger anchor commit must be strictly before its deadline: "
                    f"{committed_at.isoformat()} >= {deadline.isoformat()}"
                )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_hash": record["record_hash"],
        "commit": commit,
        "deadline": deadline,
    }


def validate_existing_anchor_chain(
    spec: Mapping[str, Any],
    root: Path,
    *,
    require_pushed: bool,
) -> list[dict[str, Any]]:
    records = stage3.scan_records(root)
    anchor_dir = stage3.SCRIPTS_DIR / "xs_chan_exploration_stage3_ledger_anchors"
    expected_names = {f"{int(record.data['sequence']):06d}_{record.data['record_hash']}.json" for record in records}
    actual_names = {path.name for path in anchor_dir.iterdir()} if anchor_dir.is_dir() else set()
    if actual_names != expected_names:
        raise FirstWeekOperationError(
            "tracked ledger anchor directory differs from the exact ledger chain: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    return [
        validate_anchor_for_record(
            spec,
            root,
            record.data,
            require_pushed=require_pushed,
        )
        for record in records
    ]


def build_preflight_report(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    decision_date: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    verify_remote: bool,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run all read-only first-week gates and prove the ledger stayed byte-equal."""

    target = pd.Timestamp(decision_date).date().isoformat()
    registered = spec["historical_exclusion"]["first_prospective_decision_date"]
    if target != registered:
        raise FirstWeekOperationError(f"first-week operator only accepts registered date {registered}")
    validate_frozen_data_dir(data_dir)
    before = ledger_closure(root)
    if before["record_count"] != 1 or before["head_type"] != "genesis":
        raise FirstWeekOperationError("first decision preflight requires the genesis-only ledger")
    validate_authorization_sidecar_inventory(
        root,
        stage3.scan_records(root),
    )
    git = validate_git_ready(verify_remote=verify_remote)
    anchors = validate_existing_anchor_chain(
        spec,
        root,
        require_pushed=True,
    )
    raw = validate_raw_ready(target, data_dir=data_dir)
    state = validate_state_ready(
        state_manifest_path,
        target,
        data_dir=data_dir,
    )
    if state["raw_parquet_inventory_sha256"] != raw["parquet_inventory_sha256"]:
        raise FirstWeekOperationError("raw audit and state cache bind different parquet closures")
    reference = validate_reference_ready(
        spec,
        root,
        reference_manifest_path,
        state_manifest_path,
        target,
        state_summary=state,
        raw_summary=raw,
    )
    after = assert_ledger_unchanged(before, root)
    operator_path = Path(__file__).resolve()
    return {
        "schema": "xs_chan_stage3_first_week_preflight_v1",
        "mode": "READ_ONLY_FULL_BRIDGE",
        "generated_at_utc": _as_utc(generated_at),
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "decision_date": target,
        "collector_source_sha256": sha256_file(stage3.SOURCE_PATH),
        "operator_source_sha256": sha256_file(operator_path),
        "git": git,
        "ledger_before": before,
        "ledger_after": after,
        "formal_ledger_mutated": False,
        "anchors": anchors,
        "raw": raw,
        "state": state,
        "reference": reference,
        "efficacy_output": "FORBIDDEN",
        "next_state": "READY_DECISION_WINDOW",
    }


def store_preflight_report(
    root: Path,
    report: Mapping[str, Any],
) -> tuple[str, Path]:
    digest, path = stage3._store_canonical_object(
        root,
        PREFLIGHT_CATEGORY,
        report,
    )
    return digest, path


def _expected_state_cache_dir(
    data_dir: Path,
    output_root: Path,
) -> Path:
    config = state_cache.StateCacheConfig()
    sources = sorted(data_dir.resolve().glob("*.parquet"))
    fingerprints = state_cache._source_fingerprints(sources)
    provenance = state_cache._engine_provenance()
    cache_id, _ = state_cache._cache_identity(
        config,
        provenance,
        fingerprints,
    )
    return output_root.resolve() / f"CHAN_STATE_CACHE_{cache_id}"


def build_or_reuse_state_cache(
    data_dir: Path,
    output_root: Path,
    *,
    workers: int,
) -> Path:
    expected_dir = _expected_state_cache_dir(data_dir, output_root)
    manifest_path = expected_dir / "source_manifest.json"
    if expected_dir.exists():
        if not manifest_path.is_file():
            raise FirstWeekOperationError(f"expected state cache is incomplete: {expected_dir}")
        return manifest_path
    result = state_cache.build_state_cache(
        data_dir,
        output_root,
        workers=workers,
        config=state_cache.StateCacheConfig(),
    )
    if not result.audit_passed or result.cache_dir != expected_dir:
        raise FirstWeekOperationError("state builder returned an unexpected or failed cache")
    return result.manifest_path


def prepare_decision_data(
    *,
    decision_date: str,
    data_dir: Path,
    snapshot_root: Path,
    state_output_root: Path,
    workers: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply data-only transitions and produce a content-addressed preflight."""

    spec = stage3.load_and_validate_spec()
    target = pd.Timestamp(decision_date).date().isoformat()
    registered = spec["historical_exclusion"]["first_prospective_decision_date"]
    if target != registered:
        raise FirstWeekOperationError(f"data preparation only accepts registered date {registered}")
    data_dir = validate_frozen_data_dir(data_dir)
    prepare_gate = validate_prepare_time(target, now=now)
    apply_gate = validate_apply_window(
        target,
        FIRST_WEEK_ENTRY_DATE,
        now=now,
    )
    git = validate_git_ready(verify_remote=True)
    root = stage3.resolve_ledger_root(spec)
    with operator_lock(root):
        ledger_before = ledger_closure(root)
        if ledger_before["record_count"] != 1 or ledger_before["head_type"] != "genesis":
            raise FirstWeekOperationError("data preparation requires the genesis-only ledger")
        raw_result = run_sync(
            argparse.Namespace(
                data_dir=data_dir,
                snapshot_root=snapshot_root,
                end_date=pd.Timestamp(target).strftime("%Y%m%d"),
                min_daily_rows=DEFAULT_MIN_DAILY_ROWS,
                new_symbol_sleep_seconds=(DEFAULT_NEW_SYMBOL_SLEEP_SECONDS),
                apply=True,
            )
        )
        if raw_result.get("status") not in {"PUBLISHED", "ALREADY_CURRENT"} or raw_result.get(
            "safe_end_date"
        ) != pd.Timestamp(target).strftime("%Y%m%d"):
            raise FirstWeekOperationError(f"raw sync did not close exactly on {target}: {raw_result}")

        with cache_lock(data_dir):
            raw_summary = validate_raw_ready(
                target,
                data_dir=data_dir,
            )
            state_manifest_path = build_or_reuse_state_cache(
                data_dir,
                state_output_root,
                workers=workers,
            )
            state_summary = validate_state_ready(
                state_manifest_path,
                target,
                data_dir=data_dir,
            )
            manifest, reference_path = stage3.fetch_reference_data(
                spec,
                FIRST_WEEK_REFERENCE_DATES,
                root=root,
                state_manifest_path=state_manifest_path,
            )
            if manifest["manifest_sha256"] != reference_path.stem:
                raise FirstWeekOperationError("reference fetch returned a non-content-derived manifest")
            report = build_preflight_report(
                spec,
                root,
                reference_path,
                state_manifest_path,
                target,
                data_dir=data_dir,
                verify_remote=True,
                generated_at=now,
            )
            rehearsal = stage3.rehearse_pipeline(
                spec,
                root,
                reference_path,
            )
            if (
                rehearsal.get("status") != "REHEARSAL_VALID_NEVER_COUNTS"
                or rehearsal.get("prospective_week_count") != 0
                or rehearsal.get("efficacy_summary_emitted") is not False
            ):
                raise FirstWeekOperationError("final reference rehearsal violated the non-counting gate")
            finish_gate = validate_apply_window(
                target,
                FIRST_WEEK_ENTRY_DATE,
            )
            assert_ledger_unchanged(ledger_before, root)
            report = {
                **report,
                "mode": "APPLIED_DATA_ONLY_FULL_BRIDGE",
                "prepare_time_gate": prepare_gate,
                "decision_apply_gate": apply_gate,
                "preparation_finished_gate": finish_gate,
                "raw_sync": {
                    "status": raw_result["status"],
                    "safe_end_date": raw_result["safe_end_date"],
                    "audit_sha256": raw_summary["audit_sha256"],
                },
                "rehearsal": rehearsal,
                "git": git,
                "formal_ledger_mutated": False,
                "next_state": "READY_DECISION_WINDOW",
            }
            digest, report_path = store_preflight_report(root, report)
            assert_ledger_unchanged(ledger_before, root)
    return {
        "status": "DECISION_DATA_PREPARED",
        "decision_date": target,
        "reference_manifest_path": str(reference_path),
        "reference_manifest_sha256": reference_path.stem,
        "state_manifest_path": str(state_manifest_path),
        "state_manifest_sha256": state_summary["manifest_sha256"],
        "raw_audit_sha256": raw_summary["audit_sha256"],
        "preflight_report_path": str(report_path),
        "preflight_report_sha256": digest,
        "formal_ledger_mutated": False,
        "expected_head": ledger_before["head"],
        "next_state": "READY_DECISION_WINDOW",
    }


def validate_preflight_receipt(
    path: Path,
    *,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    expected_head: str,
    expected_operator_source_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stem != sha256_file(resolved):
        raise FirstWeekOperationError("preflight report path must be its exact physical SHA256")
    report = _read_json(resolved)
    operator_source_sha256 = (
        expected_operator_source_sha256
        if expected_operator_source_sha256 is not None
        else sha256_file(Path(__file__).resolve())
    )
    if (
        report.get("schema") != "xs_chan_stage3_first_week_preflight_v1"
        or report.get("mode") != "APPLIED_DATA_ONLY_FULL_BRIDGE"
        or report.get("formal_ledger_mutated") is not False
        or report.get("efficacy_output") != "FORBIDDEN"
        or report.get("operator_source_sha256") != operator_source_sha256
        or report.get("ledger_before", {}).get("head") != expected_head
        or report.get("ledger_after", {}).get("head") != expected_head
        or report.get("reference", {}).get("manifest_sha256") != reference_manifest_path.resolve().stem
        or report.get("state", {}).get("manifest_sha256") != sha256_file(state_manifest_path.resolve())
        or report.get("rehearsal", {}).get("prospective_week_count") != 0
        or report.get("rehearsal", {}).get("efficacy_summary_emitted") is not False
        or report.get("reference", {}).get("bridge_dates") != list(FIRST_WEEK_REFERENCE_DATES)
        or report.get("reference", {}).get("decision_inputs_only") is not True
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(
                report.get("reference", {}).get(
                    "bridge_path_sha256",
                    "",
                )
            ),
        )
    ):
        raise FirstWeekOperationError("preflight receipt does not bind the requested non-counting preparation")
    return report


def _root_relative_object_path(
    root: Path,
    path: Path,
    *,
    label: str,
) -> str:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FirstWeekOperationError(f"{label} must be stored below the canonical ledger root") from exc
    if not resolved.is_file():
        raise FirstWeekOperationError(f"{label} is missing: {resolved}")
    return relative.as_posix()


def _anticipated_record_data(
    records: Sequence[stage3.LedgerRecord],
    *,
    record_type: str,
    logical_event_key: str,
    payload: Mapping[str, Any],
    recorded_at_utc: str | datetime,
    ledger_id: str,
) -> dict[str, Any]:
    """Build the exact record hash before the immutable record is created."""

    safe_payload = stage3._json_safe(payload)
    if not isinstance(safe_payload, dict):
        raise FirstWeekOperationError("guarded append payload must be a JSON object")
    sequence = len(records)
    previous_hash = records[-1].data["record_hash"] if records else stage3.ZERO_HASH
    body = {
        "schema": stage3.RECORD_SCHEMA,
        "ledger_id": ledger_id,
        "sequence": sequence,
        "previous_hash": previous_hash,
        "record_type": record_type,
        "logical_event_key": logical_event_key,
        "payload_sha256": stage3._payload_hash(safe_payload),
        "recorded_at_utc": stage3._format_utc(recorded_at_utc),
        "payload": safe_payload,
    }
    return {
        **body,
        "record_hash": stage3._record_body_hash(body),
    }


def _validate_final_preflight_report(
    report: Mapping[str, Any],
    *,
    expected_head: str,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    expected_operator_source_sha256: str | None = None,
) -> None:
    operator_source_sha256 = (
        expected_operator_source_sha256
        if expected_operator_source_sha256 is not None
        else sha256_file(Path(__file__).resolve())
    )
    if (
        report.get("schema") != "xs_chan_stage3_first_week_preflight_v1"
        or report.get("mode") != "READ_ONLY_FULL_BRIDGE"
        or report.get("formal_ledger_mutated") is not False
        or report.get("efficacy_output") != "FORBIDDEN"
        or report.get("operator_source_sha256") != operator_source_sha256
        or report.get("ledger_before", {}).get("head") != expected_head
        or report.get("ledger_after", {}).get("head") != expected_head
        or report.get("reference", {}).get("manifest_sha256") != reference_manifest_path.resolve().stem
        or report.get("state", {}).get("manifest_sha256") != sha256_file(state_manifest_path.resolve())
        or report.get("reference", {}).get("bridge_dates") != list(FIRST_WEEK_REFERENCE_DATES)
        or report.get("reference", {}).get("decision_inputs_only") is not True
        or report.get("reference", {}).get("prospective_week_count") != 0
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(report.get("reference", {}).get("bridge_path_sha256", "")),
        )
    ):
        raise FirstWeekOperationError("final pre-append report does not bind the exact guarded decision")


def _expected_append_authorization(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    preflight_report_path: Path,
    final_preflight_report: Mapping[str, Any],
    final_preflight_report_path: Path,
    operator_source_sha256: str | None = None,
) -> dict[str, Any]:
    payload = record["payload"]
    expected_head = str(record["previous_hash"])
    expected_operator_sha = (
        operator_source_sha256 if operator_source_sha256 is not None else sha256_file(Path(__file__).resolve())
    )
    validate_preflight_receipt(
        preflight_report_path,
        reference_manifest_path=reference_manifest_path,
        state_manifest_path=state_manifest_path,
        expected_head=expected_head,
        expected_operator_source_sha256=expected_operator_sha,
    )
    _validate_final_preflight_report(
        final_preflight_report,
        expected_head=expected_head,
        reference_manifest_path=reference_manifest_path,
        state_manifest_path=state_manifest_path,
        expected_operator_source_sha256=expected_operator_sha,
    )
    recorded_at = stage3._parse_utc(str(record["recorded_at_utc"]))
    time_gate = validate_apply_window(
        str(payload["decision_dt"]),
        str(payload["entry_dt"]),
        now=recorded_at,
    )
    git = final_preflight_report.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("branch") != REQUIRED_GIT_BRANCH
        or git.get("upstream") != REQUIRED_GIT_UPSTREAM
        or git.get("remote_fetch_url") != REQUIRED_GIT_REMOTE_URL
        or git.get("remote_push_url") != REQUIRED_GIT_REMOTE_URL
        or git.get("remote_verified") is not True
        or git.get("worktree_clean") is not True
        or git.get("head") != git.get("upstream_head")
        or git.get("head") != git.get("remote_head")
    ):
        raise FirstWeekOperationError("final preflight Git evidence is not the exact actual remote branch")

    reference_object = _root_relative_object_path(
        root,
        reference_manifest_path,
        label="reference manifest",
    )
    preflight_object = _root_relative_object_path(
        root,
        preflight_report_path,
        label="applied-data preflight receipt",
    )
    final_preflight_object = _root_relative_object_path(
        root,
        final_preflight_report_path,
        label="final pre-append report",
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "spec_physical_sha256": sha256_file(stage3.SPEC_PATH),
        "collector_source_sha256": sha256_file(stage3.SOURCE_PATH),
        "operator_source_sha256": expected_operator_sha,
        "authorization_created_at_utc": record["recorded_at_utc"],
        "ledger_id": record["ledger_id"],
        "sequence": int(record["sequence"]),
        "expected_head": expected_head,
        "record_type": record["record_type"],
        "logical_event_key": record["logical_event_key"],
        "recorded_at_utc": record["recorded_at_utc"],
        "payload_sha256": record["payload_sha256"],
        "expected_record_hash": record["record_hash"],
        "decision_date": payload["decision_dt"],
        "entry_date": payload["entry_dt"],
        "reference_manifest_object": reference_object,
        "reference_manifest_sha256": reference_manifest_path.resolve().stem,
        "state_manifest_path": str(state_manifest_path.expanduser().resolve()),
        "state_manifest_sha256": sha256_file(state_manifest_path.expanduser().resolve()),
        "preflight_report_object": preflight_object,
        "preflight_report_sha256": sha256_file(preflight_report_path.expanduser().resolve()),
        "final_preflight_report_object": final_preflight_object,
        "final_preflight_report_sha256": sha256_file(final_preflight_report_path.expanduser().resolve()),
        "raw_audit_sha256": final_preflight_report["raw"]["audit_sha256"],
        "raw_parquet_inventory_sha256": final_preflight_report["raw"]["parquet_inventory_sha256"],
        "raw_source_closure_sha256": final_preflight_report["reference"]["raw_source_closure_sha256"],
        "state_projection_sha256": final_preflight_report["state"]["projection_sha256"],
        "decision_path_sha256": final_preflight_report["reference"]["bridge_path_sha256"],
        "git": dict(git),
        "time_gate": _json_safe(time_gate),
    }


def store_append_authorization(
    spec: Mapping[str, Any],
    root: Path,
    anticipated_record: Mapping[str, Any],
    *,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    preflight_report_path: Path,
    final_preflight_report: Mapping[str, Any],
    fresh_git: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    """Persist the exact authorization before the immutable record write."""

    resolved_root = root.expanduser().resolve()
    lease = _active_authorization_lock_lease()
    if lease is None or lease.root != resolved_root or not _physical_authorization_lock_owned(resolved_root):
        raise FirstWeekOperationError(
            "append authorization may only be stored while holding the guarded physical ledger lock"
        )
    if canonical_json(final_preflight_report.get("git")) != canonical_json(fresh_git):
        raise FirstWeekOperationError("Git changed between final preflight and the locked append")
    records = stage3.scan_records(root)
    current_head = records[-1].data["record_hash"] if records else stage3.ZERO_HASH
    current_ledger_id = records[0].data["ledger_id"] if records else anticipated_record.get("ledger_id")
    existing_hashes = {record.data["record_hash"] for record in records}
    existing_keys = {record.data["logical_event_key"] for record in records}
    if (
        int(anticipated_record.get("sequence", -1)) != len(records)
        or anticipated_record.get("previous_hash") != current_head
        or anticipated_record.get("ledger_id") != current_ledger_id
        or anticipated_record.get("record_hash") in existing_hashes
        or anticipated_record.get("logical_event_key") in existing_keys
    ):
        raise FirstWeekOperationError(
            "append authorization must be created against the exact current ledger head "
            "before its record or logical event exists"
        )
    final_digest, final_path = stage3._store_canonical_object(
        root,
        FINAL_PREFLIGHT_CATEGORY,
        final_preflight_report,
    )
    if final_digest != sha256_file(final_path):
        raise FirstWeekOperationError("final preflight object is not content-addressed by its exact bytes")
    authorization = _expected_append_authorization(
        spec,
        root,
        anticipated_record,
        reference_manifest_path=reference_manifest_path,
        state_manifest_path=state_manifest_path,
        preflight_report_path=preflight_report_path,
        final_preflight_report=final_preflight_report,
        final_preflight_report_path=final_path,
    )
    digest, path = stage3._store_canonical_object(
        root,
        APPEND_AUTHORIZATION_CATEGORY,
        authorization,
    )
    if digest != sha256_file(path):
        raise FirstWeekOperationError("append authorization is not content-addressed by its exact bytes")
    return authorization, digest, path


def load_append_authorization(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_current_operator: bool = True,
) -> tuple[dict[str, Any], str, Path]:
    """Load one pre-existing intent and prove it exactly anticipated record."""

    directory = root / "objects" / APPEND_AUTHORIZATION_CATEGORY
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            if path.stem != sha256_file(path):
                raise FirstWeekOperationError(f"append authorization filename hash mismatch: {path}")
            authorization = _read_json(path)
            if canonical_json(authorization) != path.read_bytes():
                raise FirstWeekOperationError(f"append authorization bytes are not canonical: {path}")
            if authorization.get("expected_record_hash") == record["record_hash"]:
                candidates.append((path, authorization))
    if len(candidates) != 1:
        raise FirstWeekOperationError(
            "decision has no unique append-before-record authorization; "
            "a direct frozen-collector append is permanently unauthorized"
        )
    path, authorization = candidates[0]
    authorized_operator_sha = str(authorization.get("operator_source_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", authorized_operator_sha):
        raise FirstWeekOperationError("append authorization has an invalid operator source hash")
    if require_current_operator:
        if authorized_operator_sha != sha256_file(Path(__file__).resolve()):
            raise FirstWeekOperationError("current first-week operator bytes differ from the pre-record authorization")
    else:
        authorized_git = authorization.get("git")
        if not isinstance(authorized_git, Mapping):
            raise FirstWeekOperationError("append authorization has no Git evidence")
        operator_relative_path = Path(__file__).resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix()
        if (
            _git_file_sha256_at_commit(
                stage3.REPO_ROOT,
                str(authorized_git.get("head", "")),
                operator_relative_path,
            )
            != authorized_operator_sha
        ):
            raise FirstWeekOperationError("authorized Git commit does not contain the recorded operator bytes")
    reference_path = root / str(authorization.get("reference_manifest_object", ""))
    state_path = Path(str(authorization.get("state_manifest_path", "")))
    preflight_path = root / str(authorization.get("preflight_report_object", ""))
    final_preflight_path = root / str(authorization.get("final_preflight_report_object", ""))
    if (
        not final_preflight_path.is_file()
        or final_preflight_path.stem != sha256_file(final_preflight_path)
        or final_preflight_path.stem != authorization.get("final_preflight_report_sha256")
    ):
        raise FirstWeekOperationError("authorized final preflight object is missing or differs")
    final_preflight = _read_json(final_preflight_path)
    if canonical_json(final_preflight) != final_preflight_path.read_bytes():
        raise FirstWeekOperationError("authorized final preflight bytes are not canonical")
    expected = _expected_append_authorization(
        spec,
        root,
        record,
        reference_manifest_path=reference_path,
        state_manifest_path=state_path,
        preflight_report_path=preflight_path,
        final_preflight_report=final_preflight,
        final_preflight_report_path=final_preflight_path,
        operator_source_sha256=authorized_operator_sha,
    )
    if canonical_json(authorization) != canonical_json(expected):
        raise FirstWeekOperationError("append authorization does not exactly bind the decision record and evidence")
    return authorization, path.stem, path


def _authorization_sidecar_path(record: Mapping[str, Any]) -> Path:
    return (
        stage3.SCRIPTS_DIR
        / AUTHORIZATION_SIDECAR_DIR_NAME
        / f"{int(record['sequence']):06d}_{record['record_hash']}.json"
    )


def validate_authorization_sidecar_inventory(
    root: Path,
    records: Sequence[stage3.LedgerRecord],
    *,
    allow_missing_current_authorized: bool = False,
) -> None:
    """Require sidecars to equal records that have a durable authorization."""

    record_by_hash = {
        str(record.data["record_hash"]): record for record in records if record.data["record_type"] == "decision_freeze"
    }
    authorized_record_hashes: set[str] = set()
    authorization_directory = root / "objects" / APPEND_AUTHORIZATION_CATEGORY
    if authorization_directory.exists():
        if authorization_directory.is_symlink() or not authorization_directory.is_dir():
            raise FirstWeekOperationError("append authorization object path is not a real directory")
        for path in sorted(authorization_directory.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise FirstWeekOperationError(f"unexpected append authorization object: {path}")
            if path.stem != sha256_file(path):
                raise FirstWeekOperationError(f"append authorization filename hash mismatch: {path}")
            authorization = _read_json(path)
            if canonical_json(authorization) != path.read_bytes():
                raise FirstWeekOperationError(f"append authorization bytes are not canonical: {path}")
            expected_record_hash = str(authorization.get("expected_record_hash", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", expected_record_hash):
                raise FirstWeekOperationError(f"append authorization has an invalid record hash: {path}")
            if expected_record_hash in record_by_hash:
                authorized_record_hashes.add(expected_record_hash)

    directory = stage3.SCRIPTS_DIR / AUTHORIZATION_SIDECAR_DIR_NAME
    expected = {
        f"{int(record.data['sequence']):06d}_{record.data['record_hash']}.json"
        for record_hash, record in record_by_hash.items()
        if record_hash in authorized_record_hashes
    }
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise FirstWeekOperationError("tracked authorization sidecar path is not a real directory")
    actual: set[str] = set()
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise FirstWeekOperationError(f"unexpected tracked authorization sidecar: {path}")
            actual.add(path.name)
    allowed_inventories = {frozenset(expected)}
    if allow_missing_current_authorized and records:
        current = records[-1].data
        if current["record_hash"] in authorized_record_hashes:
            current_name = f"{int(current['sequence']):06d}_{current['record_hash']}.json"
            allowed_inventories.add(frozenset(expected - {current_name}))
    if frozenset(actual) not in allowed_inventories:
        raise FirstWeekOperationError(
            "tracked authorization sidecar directory differs from the exact authorized decision chain: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _expected_authorization_sidecar(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_current_operator: bool,
) -> tuple[dict[str, Any], Path]:
    """Build the exact tracked proof without writing it."""

    authorization, authorization_sha, authorization_path = load_append_authorization(
        spec,
        root,
        record,
        require_current_operator=require_current_operator,
    )
    payload = {
        "schema": AUTHORIZATION_SIDECAR_SCHEMA,
        "study_id": spec["study_id"],
        "study_identity": stage3.study_identity(spec),
        "sequence": record["sequence"],
        "record_type": record["record_type"],
        "logical_event_key": record["logical_event_key"],
        "recorded_at_utc": record["recorded_at_utc"],
        "record_hash": record["record_hash"],
        "previous_hash": record["previous_hash"],
        "authorization_object": str(authorization_path.relative_to(root)),
        "authorization_sha256": authorization_sha,
        "authorization": authorization,
        "external_timestamp_or_signature": False,
        "required_follow_up": "commit_and_push_with_the_matching_head_anchor_before_entry_open",
    }
    return payload, _authorization_sidecar_path(record)


def export_authorization_sidecar(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Export a Git-tracked proof derived only from the pre-append intent."""

    payload, path = _expected_authorization_sidecar(
        spec,
        root,
        record,
        require_current_operator=True,
    )
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        if path.read_bytes() != raw:
            raise FirstWeekOperationError(f"tracked authorization sidecar bytes conflict: {path}")
    else:
        stage3._exclusive_write(path, raw)
    return payload, path


def validate_authorization_sidecar(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
) -> dict[str, Any]:
    expected, path = _expected_authorization_sidecar(
        spec,
        root,
        record,
        require_current_operator=False,
    )
    if not path.is_file():
        raise FirstWeekOperationError(f"tracked authorization sidecar is missing: {path}")
    expected_raw = (
        json.dumps(
            expected,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if path.read_bytes() != expected_raw:
        raise FirstWeekOperationError("tracked authorization sidecar differs from its exact expected payload")
    commit: str | None = None
    deadline = _entry_open(str(record["payload"]["entry_dt"]))
    if require_pushed:
        try:
            commit = stage3._validate_tracked_file_pushed(
                path,
                deadline=deadline,
            )
        except Stage3ValidationError as exc:
            raise FirstWeekOperationError(
                f"operator authorization sidecar is not validly committed and pushed: {exc}"
            ) from exc
        committed_at = datetime.fromisoformat(
            _git_output(
                stage3.REPO_ROOT,
                "show",
                "-s",
                "--format=%cI",
                commit,
            )
        )
        if committed_at >= deadline:
            raise FirstWeekOperationError(
                "operator authorization commit must be strictly before entry open: "
                f"{committed_at.isoformat()} >= {deadline.isoformat()}"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_hash": record["record_hash"],
        "authorization_sha256": expected["authorization_sha256"],
        "commit": commit,
        "deadline": deadline,
    }


def validate_authorized_anchor_pair(
    spec: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    require_pushed: bool,
) -> dict[str, Any]:
    """Require the decision anchor and operator authorization as one commit."""

    validate_authorization_sidecar_inventory(
        root,
        stage3.scan_records(root),
    )
    authorization = validate_authorization_sidecar(
        spec,
        root,
        record,
        require_pushed=require_pushed,
    )
    anchor = validate_anchor_for_record(
        spec,
        root,
        record,
        require_pushed=require_pushed,
    )
    if require_pushed:
        if authorization["commit"] != anchor["commit"]:
            raise FirstWeekOperationError("decision anchor and operator authorization must share one commit")
        authorization_object, _, _ = load_append_authorization(
            spec,
            root,
            record,
            require_current_operator=False,
        )
        commit_line = _git_output(
            stage3.REPO_ROOT,
            "rev-list",
            "--parents",
            "-n",
            "1",
            str(anchor["commit"]),
        ).split()
        authorized_head = str(authorization_object["git"]["head"])
        if len(commit_line) != 2 or commit_line[1] != authorized_head:
            raise FirstWeekOperationError(
                "decision evidence commit must have exactly the authorized Git head as its parent"
            )
        expected_paths = {
            Path(anchor["path"]).resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
            Path(authorization["path"]).resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
        }
        changed_paths = set(
            filter(
                None,
                _git_output(
                    stage3.REPO_ROOT,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    str(anchor["commit"]),
                ).splitlines(),
            )
        )
        if changed_paths != expected_paths:
            raise FirstWeekOperationError(
                "decision evidence commit must contain exactly the matching anchor and authorization sidecar"
            )
    return {
        "anchor": anchor,
        "authorization": authorization,
        "commit": anchor["commit"],
    }


def append_first_decision_with_final_guards(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    preflight_report_path: Path,
    decision_date: str,
    *,
    expected_head: str,
    expected_decision_path_sha256: str,
    final_preflight_report: Mapping[str, Any],
) -> tuple[
    stage3.LedgerRecord,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    Path,
]:
    """Append under one ledger lock with a fresh final clock and raw check."""

    manifest = stage3.read_json(reference_manifest_path)
    local_inputs = manifest["local_inputs"]
    closure = stage3._load_content_object(
        root,
        "raw_source_closure",
        local_inputs["raw_source_closure_sha256"],
        local_inputs["raw_source_closure_object"],
    )
    stage3.verify_raw_source_closure_current(closure)

    authorization_path: Path | None = None

    with authorization_ledger_lock(root), atomic_stage3_writes(root):
        original_append = stage3.append_record
        original_lock = stage3._exclusive_lock
        owner_lease = _active_authorization_lock_lease()
        if (
            owner_lease is None
            or owner_lease.root != root.expanduser().resolve()
            or not _physical_authorization_lock_owned(root)
        ):
            raise FirstWeekOperationError("guarded append lost its physical authorization lock")

        @contextmanager
        def reuse_current_lock_or_block(lock_root: Path) -> Iterator[None]:
            lease = _active_authorization_lock_lease()
            resolved_lock_root = lock_root.expanduser().resolve()
            if (
                lease is owner_lease
                and lease.root == resolved_lock_root
                and _physical_authorization_lock_owned(resolved_lock_root)
            ):
                yield
                return
            if _physical_authorization_lock_owned(resolved_lock_root):
                raise FirstWeekOperationError(
                    "another authorization context attempted to reuse a physical Stage 3 lock"
                )
            with original_lock(lock_root):
                yield

        records = stage3.scan_records(root)
        if not records or records[-1].data["record_hash"] != expected_head:
            raise FirstWeekOperationError("ledger head changed before the guarded append")
        validate_authorization_sidecar_inventory(root, records)
        validate_existing_anchor_chain(
            spec,
            root,
            require_pushed=True,
        )

        def guarded_append(
            append_root: Path,
            record_type: str,
            logical_event_key: str,
            payload: Mapping[str, Any],
            recorded_at_utc: str | datetime | None = None,
            *,
            ledger_id: str | None = None,
        ) -> stage3.LedgerRecord:
            lease = _active_authorization_lock_lease()
            resolved_append_root = append_root.expanduser().resolve()
            if lease is not owner_lease:
                if _physical_authorization_lock_owned(resolved_append_root):
                    raise FirstWeekOperationError("another authorization context attempted the guarded Stage 3 append")
                return original_append(
                    append_root,
                    record_type,
                    logical_event_key,
                    payload,
                    recorded_at_utc,
                    ledger_id=ledger_id,
                )
            if lease.root != resolved_append_root or not _physical_authorization_lock_owned(resolved_append_root):
                raise FirstWeekOperationError(
                    "guarded append attempted another ledger while the authorization lock was active"
                )
            del recorded_at_utc
            nonlocal authorization_path
            current = stage3.scan_records(root)
            if (
                append_root.resolve() != root.resolve()
                or record_type != "decision_freeze"
                or not current
                or current[-1].data["record_hash"] != expected_head
                or payload.get("decision_path_sha256") != expected_decision_path_sha256
                or payload.get("reference_manifest_sha256") != reference_manifest_path.stem
                or ledger_id != current[0].data["ledger_id"]
            ):
                raise FirstWeekOperationError("guarded decision append target, head or preflight path changed")
            fresh_git = validate_git_ready(verify_remote=True)
            stage3.verify_raw_source_closure_current(closure)
            fresh_now = stage3.utc_now()
            stage3.validate_decision_timing(
                payload["decision_dt"],
                payload["entry_dt"],
                fresh_now,
            )
            validate_apply_window(
                payload["decision_dt"],
                payload["entry_dt"],
                now=pd.Timestamp(fresh_now).to_pydatetime(),
            )
            anticipated = _anticipated_record_data(
                current,
                record_type=record_type,
                logical_event_key=logical_event_key,
                payload=payload,
                recorded_at_utc=fresh_now,
                ledger_id=str(ledger_id),
            )
            _, _, authorization_path = store_append_authorization(
                spec,
                root,
                anticipated,
                reference_manifest_path=reference_manifest_path,
                state_manifest_path=state_manifest_path,
                preflight_report_path=preflight_report_path,
                final_preflight_report=final_preflight_report,
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
                raise FirstWeekOperationError("immutable decision record differs from its pre-append authorization")
            return appended

        stage3._exclusive_lock = reuse_current_lock_or_block
        stage3.append_record = guarded_append
        try:
            record = stage3.append_decision_freeze(
                spec,
                root,
                reference_manifest_path,
                decision_date,
            )
        finally:
            stage3.append_record = original_append
            stage3._exclusive_lock = original_lock

        if record.data["record_hash"] != stage3.scan_records(root)[-1].data["record_hash"]:
            raise FirstWeekOperationError("guarded decision record was not the final ledger head")
        if authorization_path is None:
            raise FirstWeekOperationError("guarded append returned without a durable pre-record authorization")
        authorization_sidecar, authorization_sidecar_path = export_authorization_sidecar(
            spec,
            root,
            record.data,
        )
        anchor, anchor_path = stage3.export_ledger_head_anchor(
            spec,
            root,
        )
        validate_authorized_anchor_pair(
            spec,
            root,
            record.data,
            require_pushed=False,
        )
        validate_existing_anchor_chain(
            spec,
            root,
            require_pushed=False,
        )
    return (
        record,
        anchor,
        anchor_path,
        authorization_sidecar,
        authorization_sidecar_path,
        authorization_path,
    )


def _validate_recovery_anchor_inventory(
    spec: Mapping[str, Any],
    root: Path,
    records: Sequence[stage3.LedgerRecord],
) -> bool:
    """Allow exactly the pushed prefix and an optional current-head anchor."""

    anchor_dir = stage3.SCRIPTS_DIR / "xs_chan_exploration_stage3_ledger_anchors"
    actual = {path.name for path in anchor_dir.iterdir()} if anchor_dir.is_dir() else set()
    prefix = {f"{int(record.data['sequence']):06d}_{record.data['record_hash']}.json" for record in records[:-1]}
    current_name = f"{int(records[-1].data['sequence']):06d}_{records[-1].data['record_hash']}.json"
    if actual not in (prefix, prefix | {current_name}):
        raise FirstWeekOperationError(
            "recovery anchor inventory must be the exact pushed prefix with at most the current head"
        )
    for record in records[:-1]:
        validate_anchor_for_record(
            spec,
            root,
            record.data,
            require_pushed=True,
        )
    return current_name in actual


def _validate_recovery_authorization_inventory(
    records: Sequence[stage3.LedgerRecord],
) -> bool:
    """Allow no sidecar yet or exactly the current decision sidecar."""

    directory = stage3.SCRIPTS_DIR / AUTHORIZATION_SIDECAR_DIR_NAME
    actual = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    expected_prefix = {
        f"{int(record.data['sequence']):06d}_{record.data['record_hash']}.json"
        for record in records[:-1]
        if record.data["record_type"] == "decision_freeze"
    }
    current_name = f"{int(records[-1].data['sequence']):06d}_{records[-1].data['record_hash']}.json"
    if actual not in (expected_prefix, expected_prefix | {current_name}):
        raise FirstWeekOperationError(
            "recovery authorization inventory must be the exact prior decision set with at most the current sidecar"
        )
    return current_name in actual


def recover_first_decision_anchor(
    *,
    spec: Mapping[str, Any],
    root: Path,
    record_hash: str,
    decision_date: str,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    preflight_report_path: Path,
    expected_head: str,
    data_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recover only a record exactly anticipated by a durable locked intent."""

    with operator_lock(root), cache_lock(data_dir), stage3._exclusive_lock(root):
        records = stage3.scan_records(root)
        if (
            len(records) != 2
            or records[-1].data["record_hash"] != record_hash
            or records[-1].data["record_type"] != "decision_freeze"
            or records[-1].data["payload"]["decision_dt"] != decision_date
            or records[-1].data["previous_hash"] != expected_head
        ):
            raise FirstWeekOperationError("ledger changed before guarded decision-anchor recovery")
        stage3.validate_record_semantics(spec, records, root=root)
        anchor_already_exists = _validate_recovery_anchor_inventory(
            spec,
            root,
            records,
        )
        sidecar_already_exists = _validate_recovery_authorization_inventory(records)
        authorization, authorization_sha, authorization_path = load_append_authorization(
            spec,
            root,
            records[-1].data,
            require_current_operator=False,
        )
        if (
            Path(str(authorization["state_manifest_path"])).resolve() != state_manifest_path.expanduser().resolve()
            or (root / str(authorization["reference_manifest_object"])).resolve()
            != reference_manifest_path.expanduser().resolve()
            or (root / str(authorization["preflight_report_object"])).resolve()
            != preflight_report_path.expanduser().resolve()
        ):
            raise FirstWeekOperationError("recovery inputs differ from the pre-record authorization")

        decision = records[-1].data
        try:
            pushed_pair = validate_authorized_anchor_pair(
                spec,
                root,
                decision,
                require_pushed=True,
            )
        except (FirstWeekOperationError, Stage3Error):
            pushed_pair = None
        if pushed_pair is not None:
            current_git = validate_git_ready(verify_remote=True)
            anchor_path = Path(str(pushed_pair["anchor"]["path"]))
            authorization_sidecar_path = Path(str(pushed_pair["authorization"]["path"]))
            return {
                "status": "DECISION_EVIDENCE_ALREADY_COMMITTED",
                "record_hash": record_hash,
                "anchor_path": str(anchor_path),
                "anchor_sha256": sha256_file(anchor_path),
                "anchor": _read_json(anchor_path),
                "authorization_path": str(authorization_path),
                "authorization_sha256": authorization_sha,
                "authorization_sidecar_path": str(authorization_sidecar_path),
                "authorization_sidecar_sha256": sha256_file(authorization_sidecar_path),
                "authorization_sidecar": _read_json(authorization_sidecar_path),
                "evidence_commit": pushed_pair["commit"],
                "current_git": current_git,
                "formal_ledger_mutated": False,
                "next_state": "DECISION_AUTHORIZED_WAIT_EXIT",
                "required_action": "wait for the registered exit label window",
            }

        initial_gate = validate_recovery_deadline(
            FIRST_WEEK_ENTRY_DATE,
            now=now,
        )
        authorization, authorization_sha, authorization_path = load_append_authorization(
            spec,
            root,
            decision,
            require_current_operator=True,
        )
        anchor_path = (
            stage3.SCRIPTS_DIR
            / "xs_chan_exploration_stage3_ledger_anchors"
            / f"{int(decision['sequence']):06d}_{decision['record_hash']}.json"
        )
        authorization_sidecar_path = _authorization_sidecar_path(decision)
        allowed_dirty_paths = {
            anchor_path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
            authorization_sidecar_path.resolve().relative_to(stage3.REPO_ROOT.resolve()).as_posix(),
        }
        recovery_git = validate_recovery_git_ready(
            authorization,
            allowed_dirty_paths=allowed_dirty_paths,
        )
        raw = validate_raw_ready(
            decision_date,
            data_dir=data_dir,
        )
        state = validate_state_ready(
            state_manifest_path,
            decision_date,
            data_dir=data_dir,
        )
        if state["raw_parquet_inventory_sha256"] != raw["parquet_inventory_sha256"]:
            raise FirstWeekOperationError("recovery raw and state evidence bind different closures")
        reference = validate_reference_ready(
            spec,
            root,
            reference_manifest_path,
            state_manifest_path,
            decision_date,
            state_summary=state,
            raw_summary=raw,
        )
        if (
            authorization["raw_audit_sha256"] != raw["audit_sha256"]
            or authorization["raw_parquet_inventory_sha256"] != raw["parquet_inventory_sha256"]
            or authorization["raw_source_closure_sha256"] != reference["raw_source_closure_sha256"]
            or authorization["state_manifest_sha256"] != state["manifest_sha256"]
            or authorization["state_projection_sha256"] != state["projection_sha256"]
            or authorization["reference_manifest_sha256"] != reference["manifest_sha256"]
            or authorization["decision_path_sha256"] != reference["bridge_path_sha256"]
            or decision["payload"]["decision_path_sha256"] != reference["bridge_path_sha256"]
        ):
            raise FirstWeekOperationError("current recovery evidence differs from the pre-record authorization")

        final_recovery_git = validate_recovery_git_ready(
            authorization,
            allowed_dirty_paths=allowed_dirty_paths,
        )
        current_gate = validate_recovery_deadline(FIRST_WEEK_ENTRY_DATE)
        with atomic_stage3_writes(root):
            authorization_sidecar, authorization_sidecar_path = export_authorization_sidecar(
                spec,
                root,
                decision,
            )
            anchor, anchor_path = stage3.export_ledger_head_anchor(
                spec,
                root,
            )
            validate_authorized_anchor_pair(
                spec,
                root,
                decision,
                require_pushed=False,
            )
            validate_existing_anchor_chain(
                spec,
                root,
                require_pushed=False,
            )
        finished_gate = validate_recovery_deadline(FIRST_WEEK_ENTRY_DATE)
    return {
        "status": (
            "AUTHORIZED_DECISION_EVIDENCE_REVALIDATED"
            if anchor_already_exists and sidecar_already_exists
            else "AUTHORIZED_DECISION_ANCHOR_RECOVERED"
        ),
        "record_hash": record_hash,
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor": anchor,
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha,
        "authorization_sidecar_path": str(authorization_sidecar_path),
        "authorization_sidecar_sha256": sha256_file(authorization_sidecar_path),
        "authorization_sidecar": authorization_sidecar,
        "recovery_git": recovery_git,
        "final_recovery_git": final_recovery_git,
        "initial_time_gate": initial_gate,
        "recovery_time_gate": current_gate,
        "recovery_finished_gate": finished_gate,
        "formal_ledger_mutated": False,
        "next_state": "DECISION_EVIDENCE_COMMIT_PENDING",
        "required_action": ("commit and push exactly the matching anchor and operator authorization sidecar"),
    }


def freeze_first_decision(
    *,
    decision_date: str,
    reference_manifest_path: Path,
    state_manifest_path: Path,
    preflight_report_path: Path,
    expected_head: str,
    data_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the only formal first-decision append and export its anchor."""

    spec = stage3.load_and_validate_spec()
    target = pd.Timestamp(decision_date).date().isoformat()
    if target != spec["historical_exclusion"]["first_prospective_decision_date"]:
        raise FirstWeekOperationError("formal first-decision date differs from the registered date")
    data_dir = validate_frozen_data_dir(data_dir)
    root = stage3.resolve_ledger_root(spec)
    initial_records = stage3.scan_records(root)
    if (
        len(initial_records) == 2
        and initial_records[-1].data["record_type"] == "decision_freeze"
        and initial_records[-1].data["payload"]["decision_dt"] == target
        and initial_records[-1].data["previous_hash"] == expected_head
    ):
        return recover_first_decision_anchor(
            spec=spec,
            root=root,
            record_hash=initial_records[-1].data["record_hash"],
            decision_date=target,
            reference_manifest_path=reference_manifest_path,
            state_manifest_path=state_manifest_path,
            preflight_report_path=preflight_report_path,
            expected_head=expected_head,
            data_dir=data_dir,
            now=now,
        )

    time_gate = validate_apply_window(
        target,
        FIRST_WEEK_ENTRY_DATE,
        now=now,
    )
    git = validate_git_ready(verify_remote=True)

    with operator_lock(root), cache_lock(data_dir):
        receipt = validate_preflight_receipt(
            preflight_report_path,
            reference_manifest_path=reference_manifest_path,
            state_manifest_path=state_manifest_path,
            expected_head=expected_head,
        )
        current_report = build_preflight_report(
            spec,
            root,
            reference_manifest_path,
            state_manifest_path,
            target,
            data_dir=data_dir,
            verify_remote=True,
            generated_at=now,
        )
        if current_report["ledger_before"]["head"] != expected_head:
            raise FirstWeekOperationError("preflight expected head differs immediately before append")
        expected_path_sha = current_report["reference"]["bridge_path_sha256"]
        if receipt["reference"]["bridge_path_sha256"] != expected_path_sha:
            raise FirstWeekOperationError("stored receipt and current preflight derive different decision paths")
        (
            record,
            anchor,
            anchor_path,
            authorization_sidecar,
            authorization_sidecar_path,
            authorization_path,
        ) = append_first_decision_with_final_guards(
            spec,
            root,
            reference_manifest_path.resolve(),
            state_manifest_path.resolve(),
            preflight_report_path.resolve(),
            target,
            expected_head=expected_head,
            expected_decision_path_sha256=expected_path_sha,
            final_preflight_report=current_report,
        )
        finished_at = _as_utc()
        entry_open = _entry_open(record.data["payload"]["entry_dt"]).astimezone(UTC)
        if finished_at >= entry_open:
            raise FirstWeekOperationError(
                "decision was validly appended but anchor export crossed the entry-open deadline; "
                f"preserved anchor={anchor_path}"
            )
    return {
        "status": "DECISION_FROZEN_ANCHOR_EXPORTED",
        "decision_date": target,
        "record_hash": record.data["record_hash"],
        "recorded_at_utc": record.data["recorded_at_utc"],
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor": anchor,
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sidecar["authorization_sha256"],
        "authorization_sidecar_path": str(authorization_sidecar_path),
        "authorization_sidecar_sha256": sha256_file(authorization_sidecar_path),
        "authorization_sidecar": authorization_sidecar,
        "git_before_append": authorization_sidecar["authorization"]["git"],
        "initial_git_check": git,
        "time_gate": time_gate,
        "formal_ledger_mutated": True,
        "next_state": "DECISION_EVIDENCE_COMMIT_PENDING",
        "required_action": (
            "commit and push exactly the exported anchor and matching operator authorization "
            "sidecar in one commit before entry open"
        ),
        "anchor_deadline_utc": entry_open,
    }


def status_snapshot(
    *,
    decision_date: str,
    data_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a read-only operational state without network access."""

    data_dir = validate_frozen_data_dir(data_dir)
    spec = stage3.load_and_validate_spec()
    root = stage3.resolve_ledger_root(spec)
    current = _as_utc(now)
    decision_close = _decision_close(decision_date).astimezone(UTC)
    daily_release = _daily_release(decision_date).astimezone(UTC)
    entry_open = _entry_open(FIRST_WEEK_ENTRY_DATE).astimezone(UTC)
    latest_safe_start = entry_open - ANCHOR_PUSH_RESERVE
    records = stage3.scan_records(root)
    stage3.validate_record_semantics(spec, records, root=root)
    validate_authorization_sidecar_inventory(
        root,
        records,
        allow_missing_current_authorized=True,
    )
    blinded = stage3.status_report(spec, records, root=root)
    inventory = inspect_inventory(data_dir.expanduser().resolve())
    target = pd.Timestamp(decision_date).strftime("%Y%m%d")

    git: dict[str, Any]
    try:
        git = {
            "ready": True,
            "readiness_scope": "LOCAL_TRACKING_ONLY_NO_NETWORK",
            **validate_git_ready(verify_remote=False),
        }
    except FirstWeekOperationError as exc:
        git = {"ready": False, "reason": str(exc)}

    target_date = pd.Timestamp(decision_date).date().isoformat()
    decisions = [
        record
        for record in records
        if (record.data["record_type"] == "decision_freeze" and record.data["payload"]["decision_dt"] == target_date)
    ]
    labels = (
        []
        if not decisions
        else [
            record
            for record in records
            if (
                record.data["record_type"] == "label_completion"
                and record.data["payload"]["decision_record_hash"] == decisions[0].data["record_hash"]
            )
        ]
    )

    if len(decisions) > 1 or len(labels) > 1:
        raise FirstWeekOperationError("first-week ledger contains duplicate decision or label events")
    if labels:
        try:
            validate_authorized_anchor_pair(
                spec,
                root,
                decisions[0].data,
                require_pushed=True,
            )
            validate_anchor_for_record(
                spec,
                root,
                labels[0].data,
                require_pushed=True,
            )
            state = "FIRST_WEEK_COMPLETE"
        except (FirstWeekOperationError, Stage3Error):
            try:
                load_append_authorization(
                    spec,
                    root,
                    decisions[0].data,
                    require_current_operator=False,
                )
            except (FirstWeekOperationError, Stage3Error):
                state = "UNAUTHORIZED_DECISION_PRESENT"
            else:
                state = "LABEL_ANCHOR_PENDING"
    elif decisions:
        try:
            load_append_authorization(
                spec,
                root,
                decisions[0].data,
                require_current_operator=False,
            )
        except (FirstWeekOperationError, Stage3Error):
            state = "UNAUTHORIZED_DECISION_PRESENT"
        else:
            try:
                validate_authorized_anchor_pair(
                    spec,
                    root,
                    decisions[0].data,
                    require_pushed=True,
                )
                state = "DECISION_AUTHORIZED_WAIT_EXIT"
            except (FirstWeekOperationError, Stage3Error):
                state = (
                    "INVALID_DECISION_EVIDENCE_DEADLINE"
                    if current >= entry_open
                    else "DECISION_EVIDENCE_COMMIT_PENDING"
                )
    elif current >= entry_open:
        state = "MISSED_DECISION_WINDOW"
    elif current >= latest_safe_start:
        state = "MISSED_SAFE_APPLY_WINDOW"
    elif current < decision_close:
        state = "WAIT_DECISION_DATE"
    elif current < daily_release:
        state = "WAIT_DAILY_RELEASE"
    elif inventory["max_dt"] != target:
        state = "NEEDS_RAW_TARGET"
    else:
        state = "NEEDS_STATE_AND_FINAL_REFERENCE"
    return {
        "schema": "xs_chan_stage3_first_week_status_v1",
        "checked_at_utc": current,
        "market_time": current.astimezone(EXCHANGE_TIMEZONE),
        "study_identity": stage3.study_identity(spec),
        "decision_date": pd.Timestamp(decision_date).date().isoformat(),
        "entry_date": FIRST_WEEK_ENTRY_DATE,
        "exit_date": FIRST_WEEK_EXIT_DATE,
        "time_gate": {
            "decision_close_utc": decision_close,
            "daily_release_utc": daily_release,
            "latest_safe_start_utc": latest_safe_start,
            "entry_open_utc": entry_open,
        },
        "git": git,
        "ledger": ledger_closure(root),
        "blinded_status": blinded,
        "raw": {
            "file_count": inventory["file_count"],
            "max_dt": inventory["max_dt"],
            "target_dt": target,
            "parquet_inventory_sha256": inventory["content_inventory_sha256"],
        },
        "state": state,
        "formal_ledger_mutated": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-date",
        default=FIRST_WEEK_REFERENCE_DATES[-1],
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "status",
        help="read-only local state; never calls market APIs",
    )

    preflight = subparsers.add_parser(
        "preflight-decision",
        help="read-only exact eight-date simulation; never writes the ledger",
    )
    preflight.add_argument(
        "--reference-manifest",
        type=Path,
        required=True,
    )
    preflight.add_argument(
        "--state-manifest",
        type=Path,
        required=True,
    )

    prepare = subparsers.add_parser(
        "prepare-decision",
        help="with --apply-data, update raw/state/reference but never the ledger",
    )
    prepare.add_argument("--apply-data", action="store_true")
    prepare.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
    )
    prepare.add_argument(
        "--state-output-root",
        type=Path,
        default=DEFAULT_STATE_OUTPUT_ROOT,
    )
    prepare.add_argument(
        "--workers",
        type=int,
        default=min(state_cache.mp.cpu_count(), 8),
    )

    freeze = subparsers.add_parser(
        "freeze-decision",
        help="formal append only with --apply-ledger and expected head",
    )
    freeze.add_argument("--apply-ledger", action="store_true")
    freeze.add_argument(
        "--reference-manifest",
        type=Path,
        required=True,
    )
    freeze.add_argument(
        "--state-manifest",
        type=Path,
        required=True,
    )
    freeze.add_argument("--preflight-report", type=Path)
    freeze.add_argument("--expected-head")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print_json(
                status_snapshot(
                    decision_date=args.decision_date,
                    data_dir=args.data_dir,
                )
            )
            return 0
        if args.command == "preflight-decision":
            spec = stage3.load_and_validate_spec()
            root = stage3.resolve_ledger_root(spec)
            report = build_preflight_report(
                spec,
                root,
                args.reference_manifest,
                args.state_manifest,
                args.decision_date,
                data_dir=args.data_dir,
                verify_remote=False,
            )
            _print_json(
                {
                    "status": "PREFLIGHT_PASS_NEVER_COUNTS",
                    "report": report,
                }
            )
            return 0
        if args.command == "prepare-decision":
            if not args.apply_data:
                _print_json(
                    {
                        "status": "DRY_RUN_NO_DATA_MUTATION",
                        "snapshot": status_snapshot(
                            decision_date=args.decision_date,
                            data_dir=args.data_dir,
                        ),
                        "required_flag": "--apply-data",
                    }
                )
                return 0
            _print_json(
                prepare_decision_data(
                    decision_date=args.decision_date,
                    data_dir=args.data_dir.expanduser().resolve(),
                    snapshot_root=(args.snapshot_root.expanduser().resolve()),
                    state_output_root=(args.state_output_root.expanduser().resolve()),
                    workers=args.workers,
                )
            )
            return 0
        if args.command == "freeze-decision":
            if not args.apply_ledger:
                spec = stage3.load_and_validate_spec()
                root = stage3.resolve_ledger_root(spec)
                report = build_preflight_report(
                    spec,
                    root,
                    args.reference_manifest,
                    args.state_manifest,
                    args.decision_date,
                    data_dir=args.data_dir,
                    verify_remote=False,
                )
                _print_json(
                    {
                        "status": "DRY_RUN_NO_LEDGER_MUTATION",
                        "report": report,
                        "required_flag": "--apply-ledger",
                    }
                )
                return 0
            if not args.expected_head or args.preflight_report is None:
                raise FirstWeekOperationError("--apply-ledger requires --expected-head and --preflight-report")
            _print_json(
                freeze_first_decision(
                    decision_date=args.decision_date,
                    reference_manifest_path=(args.reference_manifest.expanduser().resolve()),
                    state_manifest_path=(args.state_manifest.expanduser().resolve()),
                    preflight_report_path=(args.preflight_report.expanduser().resolve()),
                    expected_head=args.expected_head,
                    data_dir=args.data_dir.expanduser().resolve(),
                )
            )
            return 0
    except (
        FirstWeekOperationError,
        Stage3Error,
        DailySyncError,
        state_cache.StateCacheError,
        FileExistsError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        pa.ArrowException,
        pl.exceptions.PolarsError,
    ) as exc:
        print(f"[FAIL-CLOSED] {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
