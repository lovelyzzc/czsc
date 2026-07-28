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
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
PREFLIGHT_CATEGORY = "first_week_preflight"


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

    remote_head: str | None = None
    if verify_remote:
        remote_name, separator, remote_branch = upstream_name.partition("/")
        if not separator or not remote_name or not remote_branch:
            raise FirstWeekOperationError(f"cannot resolve remote branch from upstream {upstream_name!r}")
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
        "worktree_clean": True,
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
        assert_execution_binding_current(execution_binding)
    except Exception as exc:
        raise FirstWeekOperationError(f"raw audit execution binding is no longer current: {exc}") from exc
    binding = execution_binding.get("binding")
    if not isinstance(binding, Mapping):
        raise FirstWeekOperationError("raw execution binding payload is malformed")
    parameters = binding.get("parameters")
    if not isinstance(parameters, Mapping):
        raise FirstWeekOperationError("raw execution parameters are malformed")
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
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stem != sha256_file(resolved):
        raise FirstWeekOperationError("preflight report path must be its exact physical SHA256")
    report = _read_json(resolved)
    if (
        report.get("schema") != "xs_chan_stage3_first_week_preflight_v1"
        or report.get("mode") != "APPLIED_DATA_ONLY_FULL_BRIDGE"
        or report.get("formal_ledger_mutated") is not False
        or report.get("efficacy_output") != "FORBIDDEN"
        or report.get("operator_source_sha256") != sha256_file(Path(__file__).resolve())
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


@contextmanager
def _no_ledger_lock(_root: Path) -> Iterator[None]:
    yield


def append_first_decision_with_final_guards(
    spec: Mapping[str, Any],
    root: Path,
    reference_manifest_path: Path,
    decision_date: str,
    *,
    expected_head: str,
    expected_decision_path_sha256: str,
) -> tuple[stage3.LedgerRecord, dict[str, Any], Path]:
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

    original_append = stage3.append_record
    original_lock = stage3._exclusive_lock
    with original_lock(root):
        records = stage3.scan_records(root)
        if not records or records[-1].data["record_hash"] != expected_head:
            raise FirstWeekOperationError("ledger head changed before the guarded append")
        validate_anchor_for_record(
            spec,
            root,
            records[-1].data,
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
            del recorded_at_utc
            current = stage3.scan_records(root)
            if (
                append_root.resolve() != root.resolve()
                or record_type != "decision_freeze"
                or not current
                or current[-1].data["record_hash"] != expected_head
                or payload.get("decision_path_sha256") != expected_decision_path_sha256
                or payload.get("reference_manifest_sha256") != reference_manifest_path.stem
            ):
                raise FirstWeekOperationError("guarded decision append target, head or preflight path changed")
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
            return original_append(
                append_root,
                record_type,
                logical_event_key,
                payload,
                fresh_now,
                ledger_id=ledger_id,
            )

        stage3._exclusive_lock = _no_ledger_lock
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
        anchor, anchor_path = stage3.export_ledger_head_anchor(
            spec,
            root,
        )
        validate_anchor_for_record(
            spec,
            root,
            record.data,
            require_pushed=False,
        )
    return record, anchor, anchor_path


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
        recovery_deadline = _entry_open(initial_records[-1].data["payload"]["entry_dt"]).astimezone(UTC)
        if _as_utc(now) >= recovery_deadline:
            raise FirstWeekOperationError(
                "decision exists but its anchor cannot be recovered after entry open; the prospective chain is invalid"
            )
        with (
            operator_lock(root),
            cache_lock(data_dir),
            stage3._exclusive_lock(root),
        ):
            current_records = stage3.scan_records(root)
            if (
                len(current_records) != 2
                or current_records[-1].data["record_hash"] != initial_records[-1].data["record_hash"]
            ):
                raise FirstWeekOperationError("ledger changed during decision-anchor recovery")
            if _as_utc() >= recovery_deadline:
                raise FirstWeekOperationError(
                    "decision-anchor recovery crossed entry open while waiting for locks; "
                    "the prospective chain is invalid"
                )
            anchor, anchor_path = stage3.export_ledger_head_anchor(
                spec,
                root,
            )
            validate_anchor_for_record(
                spec,
                root,
                current_records[-1].data,
                require_pushed=False,
            )
        return {
            "status": "DECISION_ALREADY_FROZEN_ANCHOR_EXPORTED",
            "record_hash": current_records[-1].data["record_hash"],
            "anchor_path": str(anchor_path),
            "anchor_sha256": sha256_file(anchor_path),
            "anchor": anchor,
            "formal_ledger_mutated": False,
            "next_state": "DECISION_ANCHOR_PENDING",
        }

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
        record, anchor, anchor_path = append_first_decision_with_final_guards(
            spec,
            root,
            reference_manifest_path.resolve(),
            target,
            expected_head=expected_head,
            expected_decision_path_sha256=expected_path_sha,
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
        "git_before_append": git,
        "time_gate": time_gate,
        "formal_ledger_mutated": True,
        "next_state": "DECISION_ANCHOR_PENDING",
        "required_action": ("commit and push exactly the exported anchor before entry open"),
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
    records = stage3.scan_records(root)
    stage3.validate_record_semantics(spec, records, root=root)
    blinded = stage3.status_report(spec, records, root=root)
    inventory = inspect_inventory(data_dir.expanduser().resolve())
    target = pd.Timestamp(decision_date).strftime("%Y%m%d")

    git: dict[str, Any]
    try:
        git = {
            "ready": True,
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
            validate_anchor_for_record(
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
            state = "LABEL_ANCHOR_PENDING"
    elif decisions:
        try:
            validate_anchor_for_record(
                spec,
                root,
                decisions[0].data,
                require_pushed=True,
            )
            state = "DECISION_ANCHORED_WAIT_EXIT"
        except (FirstWeekOperationError, Stage3Error):
            state = "INVALID_DECISION_ANCHOR_DEADLINE" if current >= entry_open else "DECISION_ANCHOR_PENDING"
    elif current >= entry_open:
        state = "MISSED_DECISION_WINDOW"
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
