"""安全地批量同步全 A 股日线前复权研究缓存。

既有股票使用 ``daily`` 接口按交易日批量拉取，再用 ``pre_close`` 修复新增窗口内
的前复权接缝；旧缓存中不存在的新上市股票则用 ``pro_bar(adj="qfq")`` 拉取完整
历史。所有结果先写入暂存区并验证，先把更新前后 bytes 固化为内容寻址快照，再用
同文件系统内的原子替换更新协议冻结的 active raw 路径。发布中断会自动回滚。

默认仅生成并验证计划；显式传入 ``--apply`` 才发布：

    uv run --no-sync python scripts/_sync_daily_data.py --end-date 20260727
    uv run --no-sync python scripts/_sync_daily_data.py --end-date 20260727 --apply
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _repair_qfq_seams import SEAM_THRESHOLD, find_seams, flatten_seams
from sync_a_stock_daily import (
    DAILY_BAR_READY_HOUR,
    DEFAULT_START_DATE,
    MARKET_TIMEZONE,
    download_one,
    get_expected_end_date,
    pro,
    read_trade_date_bounds,
)

DEFAULT_DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
DEFAULT_SNAPSHOT_ROOT = Path.home() / ".ts_data_cache" / "xs_chan_raw_snapshots"
DEFAULT_END_DATE = datetime.now(MARKET_TIMEZONE).strftime("%Y%m%d")
DEFAULT_MIN_DAILY_ROWS = 1_000
DEFAULT_NEW_SYMBOL_SLEEP_SECONDS = 0.3
FULL_QFQ_MAX_ATTEMPTS = 4
FULL_QFQ_BACKOFF_CAP_SECONDS = 8
DAILY_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
)
NUMERIC_COLUMNS = DAILY_COLUMNS[2:]
RUN_PREFIX = ".a_stock_daily_qfq_sync_"
SNAPSHOT_PREFIX = "RAW_"
ARCHIVE_TEMP_PREFIX = ".tmp_raw_archive_"
AUDIT_DIR_NAME = "a_stock_daily_qfq_sync_audits"
LOCK_FILE_NAME = ".a_stock_daily_qfq_sync.lock"
JOURNAL_FILE_NAME = "journal.json"
SNAPSHOT_MANIFEST_FILE_NAME = "snapshot_manifest.json"
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(__file__).with_name("sync_a_stock_daily.py").resolve(),
    Path(__file__).with_name("_repair_qfq_seams.py").resolve(),
)
DEPENDENCY_DISTRIBUTIONS = ("numpy", "pandas", "pyarrow", "tinyshare")


class DailySyncError(RuntimeError):
    """增量同步无法满足 fail-closed 不变量。"""


def canonical_json(payload: Any) -> bytes:
    """返回稳定 UTF-8 JSON 字节。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """同步目录项；用于保证 rename/replace 在断电后仍可恢复。"""

    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_bytes(path: Path, raw: bytes, *, replace_existing: bool = True) -> None:
    """在目标目录内原子写 bytes，并同步文件与父目录。"""

    parent_existed = path.parent.is_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        fsync_directory(path.parent.parent)
    if path.exists() and not replace_existing:
        if path.read_bytes() != raw:
            raise DailySyncError(f"refusing to overwrite conflicting immutable object: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalize_trade_date(value: object) -> str:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return text
    return pd.Timestamp(value).strftime("%Y%m%d")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON，并同步临时文件与最终目录项。"""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, raw)


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_execution_binding(
    args: argparse.Namespace,
    data_dir: Path,
    snapshot_root: Path,
    safe_end: str,
    *,
    require_clean_and_pushed: bool,
) -> dict[str, Any]:
    """绑定执行源码、依赖、Git 状态和所有会改变同步结果的参数。"""

    repo_root = Path(__file__).resolve().parents[1]
    git_head = _git_output(repo_root, "rev-parse", "HEAD")
    try:
        upstream_head = _git_output(repo_root, "rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as exc:
        if require_clean_and_pushed:
            raise DailySyncError("apply requires a configured upstream branch") from exc
        upstream_head = None
    dirty_entries = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if require_clean_and_pushed and dirty_entries:
        raise DailySyncError("apply requires a clean Git worktree so execution bytes are immutable")
    if require_clean_and_pushed and upstream_head != git_head:
        raise DailySyncError("apply requires HEAD to equal the pushed upstream commit")

    source_files = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        for path in SOURCE_FILES
    ]
    lock_path = repo_root / "uv.lock"
    dependencies = {
        name: importlib.metadata.version(name)
        for name in DEPENDENCY_DISTRIBUTIONS
    }
    binding = {
        "schema": "a_stock_daily_qfq_execution_binding_v1",
        "source_files": source_files,
        "git": {
            "head": git_head,
            "upstream_head": upstream_head,
            "worktree_clean": not dirty_entries,
        },
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": dependencies,
            "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        },
        "parameters": {
            "data_dir": str(data_dir),
            "snapshot_root": str(snapshot_root),
            "requested_end_date": str(args.end_date),
            "safe_end_date": safe_end,
            "apply": bool(args.apply),
            "minimum_daily_rows": int(args.min_daily_rows),
            "new_symbol_sleep_seconds": float(args.new_symbol_sleep_seconds),
            "full_qfq_start_date": DEFAULT_START_DATE,
            "market_timezone": str(MARKET_TIMEZONE),
            "daily_bar_ready_hour": DAILY_BAR_READY_HOUR,
            "qfq_seam_threshold": SEAM_THRESHOLD,
            "retry_policy": {
                "calendar": {"max_attempts": 1},
                "daily": {"max_attempts": 1},
                "adjustment_factor": {"max_attempts": 1},
                "full_qfq": {
                    "max_attempts": FULL_QFQ_MAX_ATTEMPTS,
                    "exponential_backoff_seconds": [1, 2, 4],
                    "backoff_cap_seconds": FULL_QFQ_BACKOFF_CAP_SECONDS,
                },
            },
            "calendar_exchange": "SSE",
            "daily_fields": list(DAILY_COLUMNS),
            "adjustment_factor_fields": ["ts_code", "trade_date", "adj_factor"],
            "full_qfq_query": {"adj": "qfq", "freq": "D", "asset": "E"},
        },
    }
    return {
        "binding": binding,
        "binding_sha256": sha256_bytes(canonical_json(binding)),
    }


def assert_execution_binding_current(execution_binding: Mapping[str, Any]) -> None:
    """发布前后均确认已冻结的执行环境没有漂移。"""

    binding = execution_binding.get("binding")
    expected_sha = execution_binding.get("binding_sha256")
    if not isinstance(binding, Mapping) or expected_sha != sha256_bytes(canonical_json(binding)):
        raise DailySyncError("execution binding is malformed or has been modified")
    expected_sources = {
        str(record["path"]): str(record["sha256"])
        for record in binding.get("source_files", [])
        if isinstance(record, Mapping) and "path" in record and "sha256" in record
    }
    repo_root = Path(__file__).resolve().parents[1]
    current_sources = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for path in SOURCE_FILES
    }
    if current_sources != expected_sources:
        raise DailySyncError("execution source bytes changed during synchronization")
    git_binding = binding.get("git")
    if not isinstance(git_binding, Mapping) or _git_output(repo_root, "rev-parse", "HEAD") != git_binding.get("head"):
        raise DailySyncError("Git HEAD changed during synchronization")
    if binding.get("environment", {}).get("uv_lock_sha256") != (
        sha256_file(repo_root / "uv.lock") if (repo_root / "uv.lock").is_file() else None
    ):
        raise DailySyncError("uv.lock changed during synchronization")


@contextmanager
def cache_lock(data_dir: Path) -> Iterator[None]:
    """同一 active raw cache 只允许一个发布者。"""

    lock_path = data_dir.parent / LOCK_FILE_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parquet_bounds(path: Path) -> tuple[str, str]:
    bounds = read_trade_date_bounds(path)
    if bounds is None:
        try:
            dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
        except Exception as exc:
            raise DailySyncError(f"cannot read trade_date from {path}: {exc}") from exc
        if dates.empty:
            raise DailySyncError(f"empty raw parquet: {path}")
        bounds = (normalize_trade_date(dates.min()), normalize_trade_date(dates.max()))
    return normalize_trade_date(bounds[0]), normalize_trade_date(bounds[1])


def inspect_inventory(data_dir: Path) -> dict[str, Any]:
    """读取全部 parquet，而不是抽样文件，生成元数据级 inventory。"""

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise DailySyncError(f"no parquet files found in {data_dir}")
    records: list[dict[str, Any]] = []
    schemas: set[tuple[str, ...]] = set()
    for path in files:
        try:
            schema = tuple(pq.ParquetFile(path).schema_arrow.names)
        except Exception as exc:
            raise DailySyncError(f"cannot inspect parquet schema for {path}: {exc}") from exc
        schemas.add(schema)
        minimum, maximum = _parquet_bounds(path)
        records.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "min_dt": minimum,
                "max_dt": maximum,
            }
        )
    if len(schemas) != 1:
        raise DailySyncError(f"raw cache has {len(schemas)} physical column variants")
    schema = next(iter(schemas))
    if schema != DAILY_COLUMNS:
        raise DailySyncError(f"raw cache columns differ from the expected qfq schema: {schema}")
    return {
        "file_count": len(records),
        "min_dt": min(row["min_dt"] for row in records),
        "max_dt": max(row["max_dt"] for row in records),
        "columns": list(schema),
        "content_inventory_sha256": sha256_bytes(canonical_json(records)),
        "records": records,
    }


def active_manifest_matches(data_dir: Path, inventory: Mapping[str, Any], safe_end: str) -> bool:
    """仅信任本同步器原子发布、且 closure 与当前 bytes 一致的 active manifest。"""

    path = data_dir / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("schema") == "a_stock_daily_qfq_active_snapshot_v1"
        and manifest.get("safe_end_date") == safe_end
        and manifest.get("content_inventory_sha256") == inventory["content_inventory_sha256"]
        and manifest.get("file_count") == inventory["file_count"]
    )


def _validate_daily_frame(frame: pd.DataFrame, trade_date: str, minimum_rows: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DailySyncError(f"daily[{trade_date}] returned no rows")
    missing = set(DAILY_COLUMNS) - set(frame.columns)
    if missing:
        raise DailySyncError(f"daily[{trade_date}] missing columns: {sorted(missing)}")
    result = frame.loc[:, DAILY_COLUMNS].copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result["trade_date"] = result["trade_date"].map(normalize_trade_date)
    result = result.sort_values("trade_date", kind="mergesort", ignore_index=True)
    if set(result["trade_date"]) != {trade_date}:
        raise DailySyncError(f"daily[{trade_date}] contains another trade_date")
    if result["ts_code"].isna().any() or result["ts_code"].eq("").any():
        raise DailySyncError(f"daily[{trade_date}] contains an empty ts_code")
    if result["ts_code"].duplicated().any():
        raise DailySyncError(f"daily[{trade_date}] contains duplicated ts_code rows")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise DailySyncError(f"daily[{trade_date}] contains non-finite {column}")
    if len(result) < minimum_rows:
        raise DailySyncError(f"daily[{trade_date}] returned only {len(result)} rows; minimum is {minimum_rows}")
    return result.sort_values("ts_code", kind="mergesort", ignore_index=True)


def fetch_trade_calendar(start_date: str, end_date: str) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """冻结官方 SSE 日历，并返回闭区间内所有交易日。"""

    started = time.monotonic()
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    required = {"exchange", "cal_date", "is_open", "pretrade_date"}
    if calendar is None or calendar.empty or not required <= set(calendar):
        raise DailySyncError("official SSE trade calendar is unavailable or malformed")
    normalized = calendar.loc[:, ["exchange", "cal_date", "is_open", "pretrade_date"]].copy()
    normalized["exchange"] = normalized["exchange"].astype(str)
    normalized["cal_date"] = normalized["cal_date"].map(normalize_trade_date)
    normalized["pretrade_date"] = normalized["pretrade_date"].map(normalize_trade_date)
    normalized["is_open"] = pd.to_numeric(normalized["is_open"], errors="raise").astype(int)
    if normalized.duplicated(["exchange", "cal_date"]).any():
        raise DailySyncError("official SSE trade calendar contains duplicate dates")
    normalized = normalized.sort_values(["cal_date", "exchange"], kind="mergesort", ignore_index=True)
    open_calendar = normalized.loc[normalized["is_open"].eq(1)]
    dates = sorted(
        normalize_trade_date(value)
        for value in open_calendar["cal_date"]
        if start_date <= normalize_trade_date(value) <= end_date
    )
    if not dates:
        return (
            [],
            normalized,
            {
                "rows": len(normalized),
                "canonical_csv_sha256": sha256_bytes(normalized.to_csv(index=False, lineterminator="\n").encode()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )
    if dates[-1] != end_date:
        raise DailySyncError(f"safe end date {end_date} is not present in the official SSE calendar response")
    raw = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    evidence = {
        "rows": len(normalized),
        "canonical_csv_sha256": sha256_bytes(raw),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return dates, normalized, evidence


def fetch_daily_frames(trade_dates: Sequence[str], minimum_rows: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """每个交易日一次 API 调用，并记录稳定的响应摘要。"""

    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    fields = ",".join(DAILY_COLUMNS)
    for trade_date in trade_dates:
        started = time.monotonic()
        frame = pro.query("daily", trade_date=trade_date, fields=fields)
        normalized = _validate_daily_frame(frame, trade_date, minimum_rows)
        response_bytes = normalized.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
        evidence.append(
            {
                "trade_date": trade_date,
                "rows": len(normalized),
                "canonical_csv_sha256": sha256_bytes(response_bytes),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        frames.append(normalized)
        print(f"[FETCH] {trade_date}: {len(normalized)} rows")
    if not frames:
        return pd.DataFrame(columns=DAILY_COLUMNS), evidence
    return pd.concat(frames, ignore_index=True), evidence


def fetch_adjustment_factors(
    trade_dates: Sequence[str],
    minimum_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """冻结同步窗口每个交易日的复权因子，任何尺度变化都触发完整 qfq 重抓。"""

    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for trade_date in trade_dates:
        started = time.monotonic()
        frame = pro.query(
            "adj_factor",
            trade_date=trade_date,
            fields="ts_code,trade_date,adj_factor",
        )
        if frame is None or frame.empty:
            raise DailySyncError(f"adj_factor[{trade_date}] returned no rows")
        missing = {"ts_code", "trade_date", "adj_factor"} - set(frame)
        if missing:
            raise DailySyncError(f"adj_factor[{trade_date}] missing columns: {sorted(missing)}")
        normalized = frame.loc[:, ["ts_code", "trade_date", "adj_factor"]].copy()
        normalized["ts_code"] = normalized["ts_code"].astype(str)
        normalized["trade_date"] = normalized["trade_date"].map(normalize_trade_date)
        normalized["adj_factor"] = pd.to_numeric(normalized["adj_factor"], errors="raise")
        if set(normalized["trade_date"]) != {trade_date}:
            raise DailySyncError(f"adj_factor[{trade_date}] contains another trade_date")
        if normalized["ts_code"].duplicated().any():
            raise DailySyncError(f"adj_factor[{trade_date}] contains duplicated ts_code rows")
        if not np.isfinite(normalized["adj_factor"].to_numpy(dtype=float)).all():
            raise DailySyncError(f"adj_factor[{trade_date}] contains non-finite values")
        if len(normalized) < minimum_rows:
            raise DailySyncError(
                f"adj_factor[{trade_date}] returned only {len(normalized)} rows; minimum is {minimum_rows}"
            )
        normalized = normalized.sort_values("ts_code", kind="mergesort", ignore_index=True)
        response_bytes = normalized.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
        evidence.append(
            {
                "trade_date": trade_date,
                "rows": len(normalized),
                "canonical_csv_sha256": sha256_bytes(response_bytes),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        frames.append(normalized)
        print(f"[FACTOR] {trade_date}: {len(normalized)} rows")
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]), evidence
    return pd.concat(frames, ignore_index=True), evidence


def freeze_api_response_objects(
    object_root: Path,
    calendar: pd.DataFrame,
    daily: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict[str, Any]:
    """把本轮 calendar/daily/adj-factor 响应保存为不可覆盖的内容寻址 CSV。"""

    def store(category: str, frame: pd.DataFrame, sort_by: Sequence[str]) -> dict[str, Any]:
        normalized = frame.sort_values(list(sort_by), kind="mergesort", ignore_index=True)
        raw = normalized.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
        digest = sha256_bytes(raw)
        path = object_root / category / f"{digest}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise DailySyncError(f"content-addressed API response conflicts: {path}")
        else:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        return {"path": str(path), "sha256": digest, "rows": len(normalized)}

    return {
        "official_calendar": store("official_calendar", calendar, ["cal_date", "exchange"]),
        "daily": {
            str(trade_date): store("daily", group, ["ts_code", "trade_date"])
            for trade_date, group in daily.groupby("trade_date", sort=True)
        },
        "adj_factor": {
            str(trade_date): store("adj_factor", group, ["ts_code", "trade_date"])
            for trade_date, group in factors.groupby("trade_date", sort=True)
        },
    }


def identify_factor_refresh_symbols(
    factors: pd.DataFrame,
    trade_dates: Sequence[str],
    existing_symbols: set[str],
) -> list[str]:
    """因子改变或窗口内缺任一因子的既有代码必须全量刷新。"""

    expected_dates = set(trade_dates)
    refresh: list[str] = []
    for symbol, group in factors.loc[factors["ts_code"].isin(existing_symbols)].groupby("ts_code", sort=True):
        observed_dates = set(group["trade_date"])
        if observed_dates != expected_dates or group["adj_factor"].nunique(dropna=False) != 1:
            refresh.append(str(symbol))
    factor_symbols = set(factors["ts_code"])
    # 没有任何 factor 记录的代码通常已退市；只有本窗口实际出现 daily 行时才需要在调用方 fail-closed。
    return sorted(set(refresh) & factor_symbols)


def merge_incremental_rows(local: pd.DataFrame, incoming: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """以远端重抓行替换重叠日期，并保留严格递增、唯一的本地历史。"""

    if tuple(local.columns) != DAILY_COLUMNS:
        raise DailySyncError(f"{symbol}: local columns differ from the expected qfq schema")
    if set(incoming.columns) != set(DAILY_COLUMNS):
        raise DailySyncError(f"{symbol}: incoming columns differ from the expected qfq schema")
    old = local.loc[:, DAILY_COLUMNS].copy()
    new = incoming.loc[:, DAILY_COLUMNS].copy()
    old["ts_code"] = old["ts_code"].astype(str)
    new["ts_code"] = new["ts_code"].astype(str)
    old["trade_date"] = old["trade_date"].map(normalize_trade_date)
    new["trade_date"] = new["trade_date"].map(normalize_trade_date)
    if set(old["ts_code"]) != {symbol} or set(new["ts_code"]) != {symbol}:
        raise DailySyncError(f"{symbol}: ts_code does not match its parquet filename")
    if old["trade_date"].duplicated().any() or new["trade_date"].duplicated().any():
        raise DailySyncError(f"{symbol}: duplicated trade_date before merge")
    for column in NUMERIC_COLUMNS:
        old[column] = pd.to_numeric(old[column], errors="raise")
        new[column] = pd.to_numeric(new[column], errors="raise").astype(old[column].dtype)
    combined = (
        pd.concat([old, new], ignore_index=True)
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date", kind="mergesort", ignore_index=True)
    )
    if len(combined) < len(old):
        raise DailySyncError(f"{symbol}: merge removed historical rows")
    if combined["trade_date"].duplicated().any() or not combined["trade_date"].is_monotonic_increasing:
        raise DailySyncError(f"{symbol}: merged trade_date sequence is invalid")
    return combined.loc[:, DAILY_COLUMNS]


def _validate_full_qfq(frame: pd.DataFrame | None, symbol: str, end_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise DailySyncError(f"full qfq download returned no rows for new symbol {symbol}")
    missing = set(DAILY_COLUMNS) - set(frame)
    if missing:
        raise DailySyncError(f"full qfq download for {symbol} missing columns: {sorted(missing)}")
    result = frame.loc[:, DAILY_COLUMNS].copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result["trade_date"] = result["trade_date"].map(normalize_trade_date)
    result = result.sort_values("trade_date", kind="mergesort", ignore_index=True)
    if set(result["ts_code"]) != {symbol}:
        raise DailySyncError(f"full qfq download returned another symbol for {symbol}")
    if result["trade_date"].duplicated().any():
        raise DailySyncError(f"full qfq download has duplicated dates for {symbol}")
    if result["trade_date"].max() > end_date:
        raise DailySyncError(f"full qfq download extends beyond the safe end date for {symbol}")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise")
        finite = np.isfinite(result[column].to_numpy(dtype=float))
        if not finite.all():
            invalid_positions = np.flatnonzero(~finite).tolist()
            earliest_reference_null = column in {"pre_close", "change", "pct_chg"} and invalid_positions == [0]
            if not earliest_reference_null:
                raise DailySyncError(
                    f"full qfq download contains non-finite {column} outside the earliest row for {symbol}"
                )
    return result


def fetch_new_symbol_histories(
    symbols: Sequence[str],
    end_date: str,
    *,
    sleep_seconds: float,
    expected_max_dates: Mapping[str, str],
    object_root: Path | None = None,
    retries: int = FULL_QFQ_MAX_ATTEMPTS,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """为需重刷代码拉取完整 qfq 历史；任何失败都会阻止发布。"""

    histories: dict[str, pd.DataFrame] = {}
    evidence: list[dict[str, Any]] = []
    cache_dir = None if object_root is None else object_root / "full_qfq" / end_date
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for index, symbol in enumerate(sorted(symbols), 1):
        cache_path = None if cache_dir is None else cache_dir / f"{symbol}.parquet"
        cache_manifest_path = None if cache_path is None else cache_path.with_suffix(".json")
        reused = bool(
            cache_path is not None
            and cache_path.is_file()
            and cache_manifest_path is not None
            and cache_manifest_path.is_file()
        )
        if reused:
            try:
                cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DailySyncError(f"cannot read full qfq cache manifest for {symbol}: {exc}") from exc
            expected_cache_fields = {
                "schema": "a_stock_daily_qfq_full_refresh_object_v1",
                "ts_code": symbol,
                "start_date": DEFAULT_START_DATE,
                "end_date": end_date,
                "adjust": "qfq",
            }
            if any(cache_manifest.get(key) != value for key, value in expected_cache_fields.items()):
                raise DailySyncError(f"full qfq cache manifest query identity differs for {symbol}")
            if cache_manifest.get("parquet_sha256") != sha256_file(cache_path):
                raise DailySyncError(f"full qfq cache object hash differs for {symbol}")
        frame: pd.DataFrame | None = pd.read_parquet(cache_path) if reused else None
        if frame is None:
            for attempt in range(retries):
                frame = download_one(symbol, DEFAULT_START_DATE, end_date)
                if frame is not None and not frame.empty:
                    break
                if attempt + 1 < retries:
                    time.sleep(min(2**attempt, FULL_QFQ_BACKOFF_CAP_SECONDS))
        histories[symbol] = _validate_full_qfq(frame, symbol, end_date)
        expected_max = expected_max_dates.get(symbol)
        actual_max = histories[symbol]["trade_date"].max()
        if expected_max is not None and actual_max != expected_max:
            raise DailySyncError(
                f"full qfq history for {symbol} ends at {actual_max}, expected observed daily max {expected_max}"
            )
        if cache_path is not None and cache_manifest_path is not None and not reused:
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            histories[symbol].to_parquet(temporary, index=False)
            os.replace(temporary, cache_path)
            atomic_write_json(
                cache_manifest_path,
                {
                    "schema": "a_stock_daily_qfq_full_refresh_object_v1",
                    "ts_code": symbol,
                    "start_date": DEFAULT_START_DATE,
                    "end_date": end_date,
                    "adjust": "qfq",
                    "rows": len(histories[symbol]),
                    "min_dt": histories[symbol]["trade_date"].min(),
                    "max_dt": actual_max,
                    "parquet_sha256": sha256_file(cache_path),
                },
            )
        physical_sha = (
            sha256_file(cache_path)
            if cache_path is not None
            else sha256_bytes(histories[symbol].to_csv(index=False, lineterminator="\n", float_format="%.12g").encode())
        )
        evidence.append(
            {
                "ts_code": symbol,
                "rows": len(histories[symbol]),
                "min_dt": histories[symbol]["trade_date"].min(),
                "max_dt": actual_max,
                "object_path": None if cache_path is None else str(cache_path),
                "object_sha256": physical_sha,
                "reused": reused,
            }
        )
        source_label = "QFQ-CACHE" if reused else "QFQ"
        print(f"[{source_label}] {index}/{len(symbols)} {symbol}: {len(histories[symbol])} rows")
        if sleep_seconds and not reused and index < len(symbols):
            time.sleep(sleep_seconds)
    return histories, evidence


def scan_seams(paths: Sequence[Path]) -> dict[str, list[str]]:
    affected: dict[str, list[str]] = {}
    for path in paths:
        seams = find_seams(path)
        if seams:
            affected[path.name] = seams
    return affected


def stage_updates(
    data_dir: Path,
    daily: pd.DataFrame,
    full_histories: Mapping[str, pd.DataFrame],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """生成并验证全部待发布 parquet，不触碰 live cache。"""

    staged_dir = run_dir / "staged"
    staged_dir.mkdir()
    daily_by_symbol = {symbol: group.copy() for symbol, group in daily.groupby("ts_code", sort=True)}
    existing_names = {path.stem for path in data_dir.glob("*.parquet")}
    unresolved = sorted(set(daily_by_symbol) - existing_names - set(full_histories))
    if unresolved:
        raise DailySyncError(f"new symbols have no full qfq history: {unresolved}")
    updates: list[dict[str, Any]] = []
    seams_fixed = 0

    for symbol in sorted(set(daily_by_symbol) | set(full_histories)):
        live = data_dir / f"{symbol}.parquet"
        staged = staged_dir / live.name
        had_original = live.exists()
        if symbol in full_histories:
            combined = full_histories[symbol]
            if had_original:
                local = pd.read_parquet(live)
                local_dates = set(local["trade_date"].map(normalize_trade_date))
                refreshed_dates = set(combined["trade_date"].map(normalize_trade_date))
                if not local_dates <= refreshed_dates:
                    missing_dates = sorted(local_dates - refreshed_dates)
                    raise DailySyncError(
                        f"{symbol}: full qfq refresh removed existing trade dates: {missing_dates[:5]}"
                    )
                before_sha = sha256_file(live)
                before_rows = len(local)
                before_min, before_max = _parquet_bounds(live)
            else:
                before_sha = None
                before_rows = 0
                before_min = None
                before_max = None
        elif had_original:
            local = pd.read_parquet(live)
            combined = merge_incremental_rows(local, daily_by_symbol[symbol], symbol)
            before_sha = sha256_file(live)
            before_rows = len(local)
            before_min, before_max = _parquet_bounds(live)
        else:  # pragma: no cover - unresolved guard above makes this unreachable
            raise DailySyncError(f"{symbol}: no live cache or full qfq history")
        combined.to_parquet(staged, index=False)
        fixed = flatten_seams(staged)
        seams_fixed += fixed
        residual = find_seams(staged)
        if residual:
            raise DailySyncError(f"{symbol}: qfq seams remain after staged repair: {residual[:5]}")
        after_min, after_max = _parquet_bounds(staged)
        updates.append(
            {
                "name": live.name,
                "had_original": had_original,
                "before_sha256": before_sha,
                "after_sha256": sha256_file(staged),
                "before_rows": before_rows,
                "after_rows": len(pd.read_parquet(staged, columns=["trade_date"])),
                "rows_added": len(combined) - before_rows,
                "before_min_dt": before_min,
                "before_max_dt": before_max,
                "after_min_dt": after_min,
                "after_max_dt": after_max,
                "qfq_seams_fixed": fixed,
                "update_method": "full_qfq_refresh" if symbol in full_histories else "daily_append_factor_unchanged",
            }
        )
    return updates, seams_fixed


def archive_raw_snapshot(
    data_dir: Path,
    inventory: Mapping[str, Any],
    snapshot_root: Path,
    *,
    staged_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """固化完整 raw snapshot；目录 identity 覆盖 parquet 与辅助文件 bytes。"""

    snapshot_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=ARCHIVE_TEMP_PREFIX, dir=snapshot_root))
    base_names = {str(row["name"]) for row in inventory["records"]}
    try:
        for row in inventory["records"]:
            name = str(row["name"])
            override = None if staged_dir is None else staged_dir / name
            source = override if override is not None and override.is_file() else data_dir / name
            destination = temporary / name
            shutil.copy2(source, destination)
            expected_sha = sha256_file(override) if override is not None and override.is_file() else str(row["sha256"])
            if sha256_file(destination) != expected_sha:
                raise DailySyncError(f"snapshot copy differs from source: {name}")
        if staged_dir is not None:
            for source in sorted(staged_dir.glob("*.parquet")):
                if source.name in base_names:
                    continue
                destination = temporary / source.name
                shutil.copy2(source, destination)
                if sha256_file(destination) != sha256_file(source):
                    raise DailySyncError(f"new snapshot copy differs from staged source: {source.name}")
        auxiliary_files: list[dict[str, Any]] = []
        for source in sorted(data_dir.iterdir()):
            if (
                not source.is_file()
                or source.suffix == ".parquet"
                or source.name == SNAPSHOT_MANIFEST_FILE_NAME
            ):
                continue
            destination = temporary / source.name
            shutil.copy2(source, destination)
            auxiliary_files.append(
                {
                    "name": source.name,
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

        archived_inventory = inspect_inventory(temporary)
        payload_files = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != SNAPSHOT_MANIFEST_FILE_NAME
        ]
        payload_closure_sha = sha256_bytes(canonical_json(payload_files))
        parquet_closure_sha = str(archived_inventory["content_inventory_sha256"])
        target = snapshot_root / f"{SNAPSHOT_PREFIX}{payload_closure_sha}"
        manifest = {
            "schema": "a_stock_daily_qfq_raw_snapshot_v2",
            "closure_sha256": payload_closure_sha,
            "payload_closure_sha256": payload_closure_sha,
            "parquet_content_inventory_sha256": parquet_closure_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_active_raw_path": str(data_dir),
            "file_count": archived_inventory["file_count"],
            "min_dt": archived_inventory["min_dt"],
            "max_dt": archived_inventory["max_dt"],
            "columns": archived_inventory["columns"],
            "files": archived_inventory["records"],
            "auxiliary_files": auxiliary_files,
        }
        atomic_write_json(temporary / SNAPSHOT_MANIFEST_FILE_NAME, manifest)
        archived_inventory["snapshot_payload_sha256"] = payload_closure_sha
        archived_inventory["snapshot_payload_files"] = payload_files
        if target.exists():
            existing = inspect_inventory(target)
            existing_manifest_path = target / SNAPSHOT_MANIFEST_FILE_NAME
            try:
                existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DailySyncError(f"invalid snapshot manifest: {existing_manifest_path}") from exc
            existing_payload_files = [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(target.iterdir())
                if path.is_file() and path.name != SNAPSHOT_MANIFEST_FILE_NAME
            ]
            existing_payload_sha = sha256_bytes(canonical_json(existing_payload_files))
            if (
                existing["content_inventory_sha256"] != parquet_closure_sha
                or existing_payload_sha != payload_closure_sha
                or existing_manifest.get("schema") != "a_stock_daily_qfq_raw_snapshot_v2"
                or existing_manifest.get("payload_closure_sha256") != payload_closure_sha
                or existing_manifest.get("parquet_content_inventory_sha256") != parquet_closure_sha
            ):
                raise DailySyncError(f"content-addressed raw snapshot conflicts: {target}")
            existing["snapshot_payload_sha256"] = existing_payload_sha
            existing["snapshot_payload_files"] = existing_payload_files
            shutil.rmtree(temporary)
            return target, existing
        os.rename(temporary, target)
        fsync_directory(snapshot_root)
        return target, archived_inventory
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def materialize_active_candidate(
    data_dir: Path,
    inventory: Mapping[str, Any],
    run_dir: Path,
    daily: pd.DataFrame,
    safe_end: str,
) -> tuple[Path, dict[str, Any]]:
    """把 base + staged 组装成完整候选目录，并执行全市场 closure 检查。"""

    candidate = run_dir / "candidate_active"
    candidate.mkdir()
    staged_dir = run_dir / "staged"
    base_names = {str(row["name"]) for row in inventory["records"]}
    for row in inventory["records"]:
        name = str(row["name"])
        staged = staged_dir / name
        source = staged if staged.is_file() else data_dir / name
        shutil.copy2(source, candidate / name)
    for staged in sorted(staged_dir.glob("*.parquet")):
        if staged.name not in base_names:
            shutil.copy2(staged, candidate / staged.name)

    candidate_inventory = inspect_inventory(candidate)
    if candidate_inventory["max_dt"] != safe_end:
        raise DailySyncError(
            f"candidate active raw ends at {candidate_inventory['max_dt']}, expected safe end {safe_end}"
        )
    expected_file_count = int(inventory["file_count"]) + len(
        {path.name for path in staged_dir.glob("*.parquet")} - base_names
    )
    if candidate_inventory["file_count"] != expected_file_count:
        raise DailySyncError(
            f"candidate file count {candidate_inventory['file_count']} differs from expected {expected_file_count}"
        )
    residual_seams = scan_seams(sorted(candidate.glob("*.parquet")))
    if residual_seams:
        raise DailySyncError(f"candidate active raw contains qfq seams: {list(residual_seams)[:10]}")

    for symbol, group in daily.groupby("ts_code", sort=True):
        path = candidate / f"{symbol}.parquet"
        if not path.is_file():
            raise DailySyncError(f"candidate is missing daily symbol {symbol}")
        dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"].map(normalize_trade_date)
        expected_dates = set(group["trade_date"])
        if not expected_dates <= set(dates):
            raise DailySyncError(f"candidate {symbol} is missing fetched daily dates")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise DailySyncError(f"candidate {symbol} trade dates are duplicated or unsorted")

    current = inspect_inventory(data_dir)
    if current["content_inventory_sha256"] != inventory["content_inventory_sha256"]:
        raise DailySyncError("active raw changed while the candidate directory was being prepared")
    atomic_write_json(
        candidate / "manifest.json",
        {
            "schema": "a_stock_daily_qfq_active_snapshot_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "safe_end_date": safe_end,
            "base_content_inventory_sha256": inventory["content_inventory_sha256"],
            "content_inventory_sha256": candidate_inventory["content_inventory_sha256"],
            "file_count": candidate_inventory["file_count"],
        },
    )
    return candidate, candidate_inventory


def atomic_exchange_directories(left: Path, right: Path) -> None:
    """Linux renameat2(RENAME_EXCHANGE)：目录级原子交换，不暴露半发布路径。"""

    if left.parent.stat().st_dev != right.parent.stat().st_dev:
        raise DailySyncError("active and candidate directories must be on the same filesystem")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise DailySyncError("libc renameat2 is unavailable; refusing non-atomic active raw publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(left),
        at_fdcwd,
        os.fsencode(right),
        rename_exchange,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), f"{left} <-> {right}")
    fsync_directory(left.parent)
    if right.parent != left.parent:
        fsync_directory(right.parent)


def _inventory_digest_if_directory(path: Path) -> str | None:
    if not path.is_dir():
        return None
    try:
        return str(inspect_inventory(path)["content_inventory_sha256"])
    except DailySyncError:
        return None


def recover_orphaned_runs(data_dir: Path) -> list[str]:
    """按 journal 恢复目录交换；有效 audit 文件是唯一提交标记。"""

    recovered: list[str] = []
    for run_dir in sorted(data_dir.parent.glob(f"{RUN_PREFIX}*")):
        if not run_dir.is_dir():
            continue
        journal_path = run_dir / JOURNAL_FILE_NAME
        if not journal_path.exists():
            if (run_dir / "candidate_active").is_dir():
                raise DailySyncError(
                    f"orphan candidate has no durable publication journal; refusing to delete {run_dir}"
                )
            shutil.rmtree(run_dir)
            recovered.append(f"discarded_unpublished:{run_dir.name}")
            continue
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        audit_path_value = journal.get("audit_path")
        audit_path = Path(audit_path_value) if isinstance(audit_path_value, str) else None
        if audit_path is not None and audit_path.is_file() and audit_path.stem == sha256_file(audit_path):
            shutil.rmtree(run_dir)
            recovered.append(f"cleaned_committed:{run_dir.name}")
            continue
        candidate = run_dir / "candidate_active"
        active_digest = _inventory_digest_if_directory(data_dir)
        candidate_digest = _inventory_digest_if_directory(candidate)
        old_digest = journal.get("old_inventory_sha256")
        new_digest = journal.get("new_inventory_sha256")
        if active_digest == new_digest and candidate_digest == old_digest:
            atomic_exchange_directories(data_dir, candidate)
        elif active_digest != old_digest:
            raise DailySyncError(
                f"cannot automatically recover {run_dir}; active={active_digest}, "
                f"candidate={candidate_digest}, expected_old={old_digest}, expected_new={new_digest}"
            )
        shutil.rmtree(run_dir)
        recovered.append(f"rolled_back:{run_dir.name}")
    return recovered


def discard_orphaned_archive_temporaries(snapshot_root: Path) -> list[str]:
    """删除从未发布且不可能被引用的 archive 临时目录。"""

    discarded: list[str] = []
    for path in sorted(snapshot_root.glob(f"{ARCHIVE_TEMP_PREFIX}*")):
        if path.is_dir():
            shutil.rmtree(path)
            discarded.append(f"discarded_archive_temporary:{path.name}")
    return discarded


def publish_candidate(
    data_dir: Path,
    candidate: Path,
    run_dir: Path,
    audit_payload: dict[str, Any],
) -> Path:
    """一次目录交换发布 candidate；异常时交换回旧 active。"""

    journal_path = run_dir / JOURNAL_FILE_NAME
    journal: dict[str, Any] = {
        "schema": "a_stock_daily_qfq_directory_exchange_journal_v2",
        "phase": "PREPARED",
        "data_dir": str(data_dir),
        "candidate_path": str(candidate),
        "old_inventory_sha256": audit_payload["before_inventory"]["content_inventory_sha256"],
        "new_inventory_sha256": audit_payload["expected_active_inventory_sha256"],
        "audit_path": None,
    }
    execution_binding = audit_payload.get("execution_binding")
    if not isinstance(execution_binding, Mapping):
        raise DailySyncError("audit payload is missing its execution binding")
    assert_execution_binding_current(execution_binding)
    atomic_write_json(journal_path, journal)
    exchanged = False
    try:
        atomic_exchange_directories(data_dir, candidate)
        exchanged = True
        journal["phase"] = "ACTIVE_EXCHANGED"
        atomic_write_json(journal_path, journal)
        after = inspect_inventory(data_dir)
        if after["content_inventory_sha256"] != journal["new_inventory_sha256"]:
            raise DailySyncError("active raw closure differs after atomic directory exchange")
        audit_payload["after_inventory"] = {key: value for key, value in after.items() if key != "records"}
        audit_payload["published_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        audit_raw = (
            json.dumps(
                audit_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        audit_sha = sha256_bytes(audit_raw)
        audit_dir = data_dir.parent / AUDIT_DIR_NAME
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{audit_sha}.json"
        journal["audit_path"] = str(audit_path)
        journal["phase"] = "AUDIT_PREPARED"
        atomic_write_json(journal_path, journal)
        assert_execution_binding_current(execution_binding)
        atomic_write_bytes(audit_path, audit_raw, replace_existing=False)
        return audit_path
    except Exception:
        if exchanged:
            active_digest = _inventory_digest_if_directory(data_dir)
            candidate_digest = _inventory_digest_if_directory(candidate)
            if active_digest == journal["new_inventory_sha256"] and candidate_digest == journal["old_inventory_sha256"]:
                atomic_exchange_directories(data_dir, candidate)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--min-daily-rows", type=int, default=DEFAULT_MIN_DAILY_ROWS)
    parser.add_argument("--new-symbol-sleep-seconds", type=float, default=DEFAULT_NEW_SYMBOL_SLEEP_SECONDS)
    parser.add_argument("--apply", action="store_true", help="通过全部暂存验证后原子发布；默认仅验证计划")
    return parser.parse_args(argv)


def run_sync(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.expanduser().resolve()
    snapshot_root = args.snapshot_root.expanduser().resolve()
    if args.min_daily_rows <= 0 or args.new_symbol_sleep_seconds < 0:
        raise DailySyncError("row threshold must be positive and sleep seconds cannot be negative")
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    with cache_lock(data_dir):
        recovered = recover_orphaned_runs(data_dir)
        recovered.extend(discard_orphaned_archive_temporaries(snapshot_root))
        for action in recovered:
            print(f"[RECOVER] {action}")
        before = inspect_inventory(data_dir)
        safe_end = get_expected_end_date(args.end_date)
        execution_binding = build_execution_binding(
            args,
            data_dir,
            snapshot_root,
            safe_end,
            require_clean_and_pushed=bool(args.apply),
        )
        if safe_end < before["max_dt"]:
            raise DailySyncError(
                f"safe end {safe_end} precedes current cache max {before['max_dt']}; historical rollback is forbidden"
            )
        if safe_end == before["max_dt"] and active_manifest_matches(data_dir, before, safe_end):
            return {
                "status": "ALREADY_CURRENT",
                "safe_end_date": safe_end,
                "cache_max_date": before["max_dt"],
                "active_inventory_sha256": before["content_inventory_sha256"],
                "recovery_actions": recovered,
            }
        trade_dates, calendar, calendar_evidence = fetch_trade_calendar(before["max_dt"], safe_end)
        if not trade_dates:
            return {
                "status": "ALREADY_CURRENT",
                "safe_end_date": safe_end,
                "cache_max_date": before["max_dt"],
                "recovery_actions": recovered,
            }

        initial_seams = scan_seams([data_dir / row["name"] for row in before["records"]])
        if initial_seams:
            raise DailySyncError(
                f"live cache already contains qfq seams; repair before incremental sync: {list(initial_seams)[:10]}"
            )
        daily, daily_evidence = fetch_daily_frames(trade_dates, args.min_daily_rows)
        factors, factor_evidence = fetch_adjustment_factors(trade_dates, args.min_daily_rows)
        existing_symbols = {Path(row["name"]).stem for row in before["records"]}
        new_symbols = sorted(set(daily["ts_code"]) - existing_symbols)
        daily_existing = set(daily["ts_code"]) & existing_symbols
        factors_by_symbol = {
            symbol: set(group["trade_date"])
            for symbol, group in factors.loc[factors["ts_code"].isin(daily_existing)].groupby("ts_code")
        }
        missing_factor_symbols = sorted(
            symbol for symbol in daily_existing if factors_by_symbol.get(symbol, set()) != set(trade_dates)
        )
        factor_refresh_symbols = identify_factor_refresh_symbols(factors, trade_dates, existing_symbols)
        full_refresh_symbols = sorted(set(new_symbols) | set(factor_refresh_symbols) | set(missing_factor_symbols))
        expected_max_dates = {
            str(symbol): str(group["trade_date"].max()) for symbol, group in daily.groupby("ts_code", sort=False)
        }
        full_histories, full_history_evidence = fetch_new_symbol_histories(
            full_refresh_symbols,
            safe_end,
            sleep_seconds=args.new_symbol_sleep_seconds,
            expected_max_dates=expected_max_dates,
            object_root=data_dir.parent / "a_stock_daily_qfq_sync_objects" if args.apply else None,
        )

        run_dir = Path(tempfile.mkdtemp(prefix=RUN_PREFIX, dir=data_dir.parent))
        try:
            updates, seams_fixed = stage_updates(data_dir, daily, full_histories, run_dir)
            candidate, candidate_inventory = materialize_active_candidate(
                data_dir,
                before,
                run_dir,
                daily,
                safe_end,
            )
            audit_payload = {
                "schema": "a_stock_daily_qfq_sync_audit_v2",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_dir": str(data_dir),
                "execution_binding": execution_binding,
                "source": {
                    "bulk_existing": "tinyshare.daily",
                    "adjustment_factor": "tinyshare.adj_factor",
                    "factor_changed_and_new_symbols": "tinyshare.pro_bar(adj=qfq)",
                    "official_calendar": "tinyshare.trade_cal(exchange=SSE)",
                    "credentials_recorded": False,
                },
                "requested_end_date": args.end_date,
                "safe_end_date": safe_end,
                "overlap_start_date": before["max_dt"],
                "trade_dates": trade_dates,
                "official_calendar_response": calendar_evidence,
                "daily_responses": daily_evidence,
                "adjustment_factor_responses": factor_evidence,
                "before_inventory": {key: value for key, value in before.items() if key != "records"},
                "new_symbols": new_symbols,
                "new_symbol_count": len(new_symbols),
                "factor_refresh_symbols": factor_refresh_symbols,
                "factor_refresh_symbol_count": len(factor_refresh_symbols),
                "missing_factor_symbols": missing_factor_symbols,
                "full_qfq_refresh_symbol_count": len(full_refresh_symbols),
                "full_qfq_refresh_objects": full_history_evidence,
                "updated_file_count": len(updates),
                "qfq_seams_fixed": seams_fixed,
                "initial_qfq_seam_count": 0,
                "expected_active_inventory_sha256": candidate_inventory["content_inventory_sha256"],
                "publish_protocol": "immutable_before_after_archives_then_renameat2_directory_exchange",
                "updates": list(updates),
            }
            if not args.apply:
                return {
                    "status": "VALIDATED_DRY_RUN",
                    "safe_end_date": safe_end,
                    "trade_dates": trade_dates,
                    "updated_file_count": len(updates),
                    "new_symbols": new_symbols,
                    "factor_refresh_symbol_count": len(factor_refresh_symbols),
                    "full_qfq_refresh_symbol_count": len(full_refresh_symbols),
                    "qfq_seams_fixed": seams_fixed,
                    "candidate_inventory_sha256": candidate_inventory["content_inventory_sha256"],
                    "recovery_actions": recovered,
                }
            old_snapshot, old_snapshot_inventory = archive_raw_snapshot(
                data_dir,
                before,
                snapshot_root,
            )
            new_snapshot, new_snapshot_inventory = archive_raw_snapshot(
                candidate,
                candidate_inventory,
                snapshot_root,
            )
            response_objects = freeze_api_response_objects(
                data_dir.parent / "a_stock_daily_qfq_sync_objects",
                calendar,
                daily,
                factors,
            )
            if old_snapshot_inventory["content_inventory_sha256"] != before["content_inventory_sha256"]:
                raise DailySyncError("archived old raw closure differs from the active pre-update closure")
            if new_snapshot_inventory["content_inventory_sha256"] != candidate_inventory["content_inventory_sha256"]:
                raise DailySyncError("archived new raw closure differs from the validated candidate closure")
            audit_payload["old_raw_snapshot"] = {
                "path": str(old_snapshot),
                "closure_sha256": old_snapshot_inventory["snapshot_payload_sha256"],
                "parquet_content_inventory_sha256": old_snapshot_inventory["content_inventory_sha256"],
            }
            audit_payload["new_raw_snapshot"] = {
                "path": str(new_snapshot),
                "closure_sha256": new_snapshot_inventory["snapshot_payload_sha256"],
                "parquet_content_inventory_sha256": new_snapshot_inventory["content_inventory_sha256"],
            }
            audit_payload["api_response_objects"] = response_objects
            current_execution_binding = build_execution_binding(
                args,
                data_dir,
                snapshot_root,
                safe_end,
                require_clean_and_pushed=bool(args.apply),
            )
            if current_execution_binding != execution_binding:
                raise DailySyncError("execution binding changed before publication")
            audit_path = publish_candidate(data_dir, candidate, run_dir, audit_payload)
            result = {
                "status": "PUBLISHED",
                "safe_end_date": safe_end,
                "trade_dates": trade_dates,
                "updated_file_count": len(updates),
                "new_symbols": new_symbols,
                "qfq_seams_fixed": seams_fixed,
                "audit_path": str(audit_path),
                "audit_sha256": audit_path.stem,
                "old_raw_snapshot": str(old_snapshot),
                "new_raw_snapshot": str(new_snapshot),
                "active_inventory_sha256": candidate_inventory["content_inventory_sha256"],
                "recovery_actions": recovered,
            }
            shutil.rmtree(run_dir)
            return result
        except Exception:
            if run_dir.exists() and not (run_dir / JOURNAL_FILE_NAME).exists():
                shutil.rmtree(run_dir)
            raise
        finally:
            if not args.apply and run_dir.exists():
                shutil.rmtree(run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_sync(args)
    except (DailySyncError, OSError) as exc:
        print(f"[FAIL-CLOSED] {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
