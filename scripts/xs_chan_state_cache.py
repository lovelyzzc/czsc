"""Build and audit a causal 0-10 Chan-regime projection.

This module deliberately does not use :func:`trend_regime.load_stock`: that
loader applies the legacy whole-file ``MIN_BARS=500`` universe filter.  Every
non-empty source file receives a daily projection.  Sources at or below the
120-bar warmup boundary are all regime 0; longer sources are replayed after
their explicit warmup rows.  The published projection contains exactly
``symbol, dt, regime``.

The local development extension can lag the source package and miss
``resample_bars``.  Importing the package top level would then fail before the
state code is reached, so the same narrow in-process shim used by the existing
surge research scripts is installed before importing ``trend_regime``.

Default production run::

    uv run --no-sync python scripts/xs_chan_state_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import types
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PACKAGE_DIR = REPO_ROOT / "czsc"
DEFAULT_DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
DEFAULT_OUTPUT_ROOT = SCRIPTS_DIR / "_output" / "xs_chan_state_cache"
TREND_REGIME_PATH = SCRIPTS_DIR / "trend_regime.py"
FORMAT_KLINE_PATH = PACKAGE_DIR / "_format_standard_kline.py"

STATE_COLUMNS = ("symbol", "dt", "regime")
AUDIT_DETAIL_COLUMNS = (
    "symbol",
    "checkpoint_dt",
    "prefix_rows",
    "expected_rows",
    "actual_rows",
    "expected_sha256",
    "actual_sha256",
    "symbol_mismatches",
    "dt_mismatches",
    "regime_mismatches",
    "passed",
)
VALID_REGIMES = frozenset(range(11))
WARMUP_BARS = 120
SCHEMA_VERSION = "xs_chan_state_cache_v1"
DEFAULT_AUDIT_SYMBOLS = 100
DEFAULT_AUDIT_CHECKPOINTS = 20
DEFAULT_AUDIT_SEED = 20260717
REQUIRED_SOURCE_COLUMNS = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "pct_chg",
)
NUMERIC_SOURCE_COLUMNS = ("open", "high", "low", "close", "vol", "amount", "pct_chg")
STRICT_NON_NULL_SOURCE_COLUMNS = tuple(column for column in REQUIRED_SOURCE_COLUMNS if column != "pct_chg")


class StateCacheError(RuntimeError):
    """Base class for fail-closed cache errors."""


class StateCacheInputError(StateCacheError):
    """A source file does not satisfy the frozen input contract."""


class StateProjectionError(StateCacheError):
    """Generated state rows do not satisfy the narrow projection contract."""


@dataclass(frozen=True)
class StateCacheConfig:
    """Parameters that affect cache identity or its mandatory audit."""

    audit_symbols: int = DEFAULT_AUDIT_SYMBOLS
    audit_checkpoints: int = DEFAULT_AUDIT_CHECKPOINTS
    audit_seed: int = DEFAULT_AUDIT_SEED
    warmup_bars: int = WARMUP_BARS

    def validate(self) -> None:
        if self.audit_symbols <= 0:
            raise ValueError("audit_symbols must be positive")
        if self.audit_checkpoints <= 0:
            raise ValueError("audit_checkpoints must be positive")
        if self.warmup_bars != WARMUP_BARS:
            raise ValueError(f"warmup_bars is frozen at {WARMUP_BARS}")


@dataclass(frozen=True)
class StateCacheBuildResult:
    """Published cache location and its machine-readable data-gate result."""

    cache_id: str
    cache_dir: Path
    projection_path: Path
    manifest_path: Path
    audit_path: Path
    audit_parquet_path: Path
    audit_passed: bool


@dataclass(frozen=True)
class _BuildTask:
    index: int
    source_path: str
    source_sha256: str
    part_path: str


@dataclass(frozen=True)
class _AuditTask:
    source_path: str
    source_sha256: str
    full_part_path: str
    checkpoints: int


ProjectionGenerator = Callable[[pd.DataFrame], pd.DataFrame]
_TREND_REGIME: Any | None = None


def install_czsc_shim() -> None:
    """Expose only the API imported by ``trend_regime`` without ``czsc.__init__``."""
    current = sys.modules.get("czsc")
    required = ("CZSC", "Freq", "format_standard_kline")
    if current is not None and all(hasattr(current, name) for name in required):
        return

    package = types.ModuleType("czsc")
    package.__package__ = "czsc"
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["czsc"] = package
    try:
        native = importlib.import_module("czsc._native")
        formatter = importlib.import_module("czsc._format_standard_kline")
    except Exception:
        if current is None:
            sys.modules.pop("czsc", None)
        else:
            sys.modules["czsc"] = current
        raise
    package.CZSC = native.CZSC
    package.Freq = native.Freq
    package.format_standard_kline = formatter.format_standard_kline
    package._native = native


def get_trend_regime() -> Any:
    """Load the frozen state implementation through the narrow compatibility shim."""
    global _TREND_REGIME
    if _TREND_REGIME is None:
        previous = sys.modules.get("czsc")
        required = ("CZSC", "Freq", "format_standard_kline")
        shim_required = previous is None or not all(hasattr(previous, name) for name in required)
        install_czsc_shim()
        try:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))
            module = importlib.import_module("trend_regime")
        finally:
            if shim_required:
                if previous is None:
                    sys.modules.pop("czsc", None)
                else:
                    sys.modules["czsc"] = previous
        domain = {int(regime) for regime in module.ALL_REGIMES}
        if domain != VALID_REGIMES:
            raise StateProjectionError(f"trend_regime domain changed: {sorted(domain)}")
        if int(module.WARMUP_BARS) != WARMUP_BARS:
            raise StateProjectionError(
                f"trend_regime.WARMUP_BARS changed: {module.WARMUP_BARS}; expected {WARMUP_BARS}"
            )
        _TREND_REGIME = module
    return _TREND_REGIME


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_source_hash(path: str | Path, expected: str, phase: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise StateCacheInputError(
            f"source mutated after cache identity was frozen during {phase}: {path}; expected {expected}, got {actual}"
        )


def _parse_dates(values: pd.Series, path: Path) -> pd.Series:
    text = values.astype(str).str.strip()
    try:
        if text.str.fullmatch(r"\d{8}").all():
            parsed = pd.to_datetime(text, format="%Y%m%d", errors="raise")
        else:
            parsed = pd.to_datetime(values, errors="raise")
    except Exception as exc:
        raise StateCacheInputError(f"{path}: invalid trade_date values") from exc
    if parsed.isna().any():
        raise StateCacheInputError(f"{path}: null trade_date values")
    return parsed.astype("datetime64[ns]")


def load_source_frame(path: str | Path) -> pd.DataFrame:
    """Read one parquet directly and enforce the state engine's input contract."""
    source = Path(path).resolve()
    if not source.is_file():
        raise StateCacheInputError(f"source is not a file: {source}")
    try:
        available = set(pq.read_schema(source).names)
    except Exception as exc:
        raise StateCacheInputError(f"cannot read parquet schema: {source}") from exc
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - available)
    if missing:
        raise StateCacheInputError(f"{source}: missing required columns {missing}")
    try:
        frame = pd.read_parquet(source, columns=list(REQUIRED_SOURCE_COLUMNS))
    except Exception as exc:
        raise StateCacheInputError(f"cannot read parquet data: {source}") from exc
    if frame.empty:
        raise StateCacheInputError(f"{source}: empty source")
    if frame[list(STRICT_NON_NULL_SOURCE_COLUMNS)].isna().any().any():
        nulls = frame[list(STRICT_NON_NULL_SOURCE_COLUMNS)].isna().sum()
        bad = {name: int(count) for name, count in nulls.items() if count}
        raise StateCacheInputError(f"{source}: null required values {bad}")

    symbols = frame["ts_code"].astype(str).unique()
    if len(symbols) != 1 or not symbols[0]:
        raise StateCacheInputError(f"{source}: expected exactly one non-empty ts_code, got {symbols.tolist()}")
    frame["trade_date"] = _parse_dates(frame["trade_date"], source)
    if frame["trade_date"].duplicated().any():
        duplicates = frame.loc[frame["trade_date"].duplicated(keep=False), "trade_date"]
        sample = sorted(pd.DatetimeIndex(duplicates).strftime("%Y-%m-%d").unique())[:5]
        raise StateCacheInputError(f"{source}: duplicate trade_date values {sample}")

    frame = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    pct_null = frame["pct_chg"].isna()
    if pct_null.any():
        null_rows = np.flatnonzero(pct_null.to_numpy()).tolist()
        if null_rows != [0]:
            raise StateCacheInputError(
                f"{source}: pct_chg may be null only on the earliest observed bar, got rows {null_rows[:5]}"
            )
        # A listing's first observed bar has no prior close.  It is inside the
        # explicit 120-bar regime-0 warmup, so the only causal normalization is
        # the same neutral value used when the optional field is absent.
        frame.loc[0, "pct_chg"] = 0.0

    for column in NUMERIC_SOURCE_COLUMNS:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except Exception as exc:
            raise StateCacheInputError(f"{source}: non-numeric {column}") from exc
        if not np.isfinite(numeric.to_numpy()).all():
            raise StateCacheInputError(f"{source}: non-finite {column}")
        frame[column] = numeric
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise StateCacheInputError(f"{source}: non-positive OHLC value")
    if (frame[["vol", "amount"]] < 0).any().any():
        raise StateCacheInputError(f"{source}: negative volume or amount")

    frame = frame.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    frame["symbol"] = str(symbols[0])
    return frame.sort_values("dt", kind="mergesort").reset_index(drop=True)


def _empty_projection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="object"),
            "dt": pd.Series(dtype="datetime64[ns]"),
            "regime": pd.Series(dtype="int8"),
        }
    )


def validate_projection(frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    """Normalize and validate the strict ``symbol,dt,regime`` physical projection."""
    if tuple(frame.columns) != STATE_COLUMNS:
        raise StateProjectionError(f"projection columns must be exactly {STATE_COLUMNS}, got {tuple(frame.columns)}")
    if frame.empty:
        if not allow_empty:
            raise StateProjectionError("projection is empty")
        return _empty_projection()
    if frame.isna().any().any():
        raise StateProjectionError("projection contains null values")

    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str)
    if result["symbol"].eq("").any():
        raise StateProjectionError("projection contains an empty symbol")
    try:
        result["dt"] = pd.to_datetime(result["dt"], errors="raise").astype("datetime64[ns]")
        numeric = pd.to_numeric(result["regime"], errors="raise")
    except Exception as exc:
        raise StateProjectionError("projection contains an invalid dt or regime") from exc
    if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))).all():
        raise StateProjectionError("regime values must be integers")
    regimes = set(numeric.astype(int).unique())
    illegal = sorted(regimes - VALID_REGIMES)
    if illegal:
        raise StateProjectionError(f"illegal regime values: {illegal}")
    result["regime"] = numeric.astype("int8")
    if result.duplicated(["symbol", "dt"]).any():
        raise StateProjectionError("projection contains duplicate symbol/dt rows")
    return result.sort_values(["symbol", "dt"], kind="mergesort").reset_index(drop=True)


def project_snapshots(symbol: str, snapshots: Sequence[Any]) -> pd.DataFrame:
    """Physically discard every state field except the three-column whitelist."""
    if not snapshots:
        return _empty_projection()
    projection = pd.DataFrame(
        {
            "symbol": [str(symbol)] * len(snapshots),
            "dt": [snapshot.dt for snapshot in snapshots],
            "regime": [snapshot.regime for snapshot in snapshots],
        },
        columns=list(STATE_COLUMNS),
    )
    return validate_projection(projection)


def generate_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Replay one already-validated source frame and retain every state in 0..10."""
    if frame.empty:
        return _empty_projection()
    symbol = str(frame["symbol"].iloc[0])
    warmup_rows = min(len(frame), WARMUP_BARS)
    warmup = pd.DataFrame(
        {
            "symbol": symbol,
            "dt": frame["dt"].iloc[:warmup_rows].to_numpy(),
            "regime": np.zeros(warmup_rows, dtype=np.int8),
        },
        columns=list(STATE_COLUMNS),
    )
    if len(frame) > WARMUP_BARS:
        trend_regime = get_trend_regime()
        snapshots = trend_regime.iter_states(frame, with_features=False, tail=None)
        post_warmup = project_snapshots(symbol, snapshots)
    else:
        post_warmup = _empty_projection()
    projection = validate_projection(pd.concat([warmup, post_warmup], ignore_index=True))
    expected_rows = len(frame)
    if len(projection) != expected_rows:
        raise StateProjectionError(
            f"{frame['symbol'].iloc[0]}: expected {expected_rows} state rows, got {len(projection)}"
        )
    expected_dates = frame["dt"].reset_index(drop=True).astype("datetime64[ns]")
    if not projection["dt"].reset_index(drop=True).equals(expected_dates):
        raise StateProjectionError(f"{symbol}: state dates do not exactly cover every observed source bar")
    return projection


def projection_sha256(frame: pd.DataFrame) -> str:
    projection = validate_projection(frame, allow_empty=True)
    payload = {
        "columns": list(projection.columns),
        "dtypes": [str(dtype) for dtype in projection.dtypes],
        "rows": len(projection),
    }
    digest = hashlib.sha256(_canonical_json(payload))
    digest.update(pd.util.hash_pandas_object(projection, index=False, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def _regime_counts(projection: pd.DataFrame) -> dict[str, int]:
    counts = projection["regime"].value_counts().to_dict() if not projection.empty else {}
    return {str(regime): int(counts.get(regime, 0)) for regime in range(11)}


def _process_source(task: _BuildTask) -> dict[str, Any]:
    frame = load_source_frame(task.source_path)
    _assert_source_hash(task.source_path, task.source_sha256, "state generation")
    symbol = str(frame["symbol"].iloc[0])
    base = {
        "index": task.index,
        "path": task.source_path,
        "sha256": task.source_sha256,
        "symbol": symbol,
        "source_rows": len(frame),
        "min_dt": frame["dt"].iloc[0],
        "max_dt": frame["dt"].iloc[-1],
    }
    projection = generate_projection(frame)
    part_path = Path(task.part_path)
    projection.to_parquet(part_path, index=False)
    return {
        **base,
        "status": "generated",
        "state_rows": len(projection),
        "state_counts": _regime_counts(projection),
        "projection_sha256": projection_sha256(projection),
        "part_path": str(part_path),
    }


def _run_build_tasks(tasks: Sequence[_BuildTask], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        results = [_process_source(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_process_source, tasks, chunksize=1))
    return sorted(results, key=lambda item: int(item["index"]))


def _checkpoint_lengths(row_count: int, checkpoint_count: int) -> list[int]:
    available = row_count - WARMUP_BARS
    if available < checkpoint_count:
        raise StateCacheInputError(
            f"{row_count} rows provide only {available} unique post-warmup checkpoints; need {checkpoint_count}"
        )
    values = np.linspace(WARMUP_BARS + 1, row_count, num=checkpoint_count, dtype=int)
    checkpoints = sorted(set(map(int, values)))
    if len(checkpoints) != checkpoint_count:
        raise AssertionError("checkpoint construction did not produce the configured count")
    return checkpoints


def _field_mismatches(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, int]:
    left = expected.reset_index(drop=True)
    right = actual.reset_index(drop=True)
    common = min(len(left), len(right))
    length_gap = abs(len(left) - len(right))
    mismatches: dict[str, int] = {}
    for column in STATE_COLUMNS:
        if common:
            unequal = ~left.loc[: common - 1, column].eq(right.loc[: common - 1, column])
            count = int(unequal.sum())
        else:
            count = 0
        mismatches[column] = count + length_gap
    return mismatches


def _audit_against_full(
    source: pd.DataFrame,
    full_projection: pd.DataFrame,
    checkpoints: int,
    generator: ProjectionGenerator,
) -> dict[str, Any]:
    full = validate_projection(full_projection)
    symbol = str(source["symbol"].iloc[0])
    rows: list[dict[str, Any]] = []
    total_field_mismatches = Counter(dict.fromkeys(STATE_COLUMNS, 0))
    for prefix_rows in _checkpoint_lengths(len(source), checkpoints):
        prefix = source.iloc[:prefix_rows].copy()
        actual = validate_projection(generator(prefix))
        cutoff = pd.Timestamp(prefix["dt"].iloc[-1])
        expected = validate_projection(full[full["dt"] <= cutoff].copy())
        field_mismatches = _field_mismatches(expected, actual)
        total_field_mismatches.update(field_mismatches)
        rows.append(
            {
                "prefix_rows": prefix_rows,
                "cutoff": cutoff,
                "expected_rows": len(expected),
                "actual_rows": len(actual),
                "expected_sha256": projection_sha256(expected),
                "actual_sha256": projection_sha256(actual),
                "field_mismatches": field_mismatches,
                "passed": not any(field_mismatches.values()),
            }
        )
    return {
        "symbol": symbol,
        "source_rows": len(source),
        "comparisons": rows,
        "comparison_count": len(rows),
        "field_mismatches": dict(total_field_mismatches),
        "mismatch_count": int(sum(total_field_mismatches.values())),
        "passed": not any(total_field_mismatches.values()),
    }


def audit_frame_prefix_invariance(
    source: pd.DataFrame,
    checkpoints: int,
    generator: ProjectionGenerator = generate_projection,
) -> dict[str, Any]:
    """Small/public audit primitive used by tests and focused investigations."""
    if len(source) <= WARMUP_BARS:
        raise StateCacheInputError(f"prefix audit requires more than {WARMUP_BARS} rows")
    full = validate_projection(generator(source))
    return _audit_against_full(source, full, checkpoints, generator)


def _audit_source(task: _AuditTask) -> dict[str, Any]:
    source = load_source_frame(task.source_path)
    _assert_source_hash(task.source_path, task.source_sha256, "prefix audit")
    full = pd.read_parquet(task.full_part_path, columns=list(STATE_COLUMNS))
    result = _audit_against_full(source, full, task.checkpoints, generate_projection)
    result["source_sha256"] = task.source_sha256
    return result


def _run_audit_tasks(tasks: Sequence[_AuditTask], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        results = [_audit_source(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_audit_source, tasks, chunksize=1))
    return sorted(results, key=lambda item: str(item["symbol"]))


def _selection_key(item: dict[str, Any], seed: int) -> str:
    material = f"{seed}:{item['symbol']}:{item['sha256']}".encode()
    return hashlib.sha256(material).hexdigest()


def run_prefix_audit(
    generated: Sequence[dict[str, Any]],
    config: StateCacheConfig,
    workers: int,
) -> dict[str, Any]:
    """Run the configured deterministic 100x20 audit (or an explicit smaller test configuration)."""
    eligible = [
        item
        for item in generated
        if item["status"] == "generated" and int(item["source_rows"]) - config.warmup_bars >= config.audit_checkpoints
    ]
    eligible.sort(key=lambda item: _selection_key(item, config.audit_seed))
    selected = eligible[: config.audit_symbols]
    tasks = [
        _AuditTask(
            source_path=str(item["path"]),
            source_sha256=str(item["sha256"]),
            full_part_path=str(item["part_path"]),
            checkpoints=config.audit_checkpoints,
        )
        for item in selected
    ]
    results = _run_audit_tasks(tasks, workers) if tasks else []
    fields = Counter(dict.fromkeys(STATE_COLUMNS, 0))
    for result in results:
        fields.update(result["field_mismatches"])
    enough_symbols = len(selected) == config.audit_symbols
    expected_comparisons = config.audit_symbols * config.audit_checkpoints
    comparison_count = sum(int(result["comparison_count"]) for result in results)
    passed = (
        enough_symbols
        and comparison_count == expected_comparisons
        and all(bool(result["passed"]) for result in results)
        and not any(fields.values())
    )
    errors = []
    if not enough_symbols:
        errors.append(f"insufficient audit symbols: required {config.audit_symbols}, eligible {len(eligible)}")
    return {
        "method": "deterministic_sha256_sample_prefix_vs_full_exact_projection",
        "configured_symbols": config.audit_symbols,
        "configured_checkpoints_per_symbol": config.audit_checkpoints,
        "configured_seed": config.audit_seed,
        "eligible_symbols": len(eligible),
        "audited_symbols": len(selected),
        "expected_comparisons": expected_comparisons,
        "comparison_count": comparison_count,
        "compared_fields": list(STATE_COLUMNS),
        "field_mismatches": dict(fields),
        "mismatch_count": int(sum(fields.values())),
        "errors": errors,
        "symbol_results": results,
        "passed": passed,
    }


def _audit_detail_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("checkpoint_dt", pa.timestamp("ns"), nullable=False),
            pa.field("prefix_rows", pa.int64(), nullable=False),
            pa.field("expected_rows", pa.int64(), nullable=False),
            pa.field("actual_rows", pa.int64(), nullable=False),
            pa.field("expected_sha256", pa.string(), nullable=False),
            pa.field("actual_sha256", pa.string(), nullable=False),
            pa.field("symbol_mismatches", pa.int64(), nullable=False),
            pa.field("dt_mismatches", pa.int64(), nullable=False),
            pa.field("regime_mismatches", pa.int64(), nullable=False),
            pa.field("passed", pa.bool_(), nullable=False),
        ]
    )


def write_audit_details(prefix_audit: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Publish one strict row per audited symbol/checkpoint and return manifest metadata."""
    rows: list[dict[str, Any]] = []
    for symbol_result in prefix_audit["symbol_results"]:
        symbol = str(symbol_result["symbol"])
        for comparison in symbol_result["comparisons"]:
            mismatches = comparison["field_mismatches"]
            rows.append(
                {
                    "symbol": symbol,
                    "checkpoint_dt": pd.Timestamp(comparison["cutoff"]),
                    "prefix_rows": int(comparison["prefix_rows"]),
                    "expected_rows": int(comparison["expected_rows"]),
                    "actual_rows": int(comparison["actual_rows"]),
                    "expected_sha256": str(comparison["expected_sha256"]),
                    "actual_sha256": str(comparison["actual_sha256"]),
                    "symbol_mismatches": int(mismatches["symbol"]),
                    "dt_mismatches": int(mismatches["dt"]),
                    "regime_mismatches": int(mismatches["regime"]),
                    "passed": bool(comparison["passed"]),
                }
            )
    rows.sort(key=lambda row: (row["symbol"], row["checkpoint_dt"], row["prefix_rows"]))
    if len(rows) != int(prefix_audit["comparison_count"]):
        raise StateProjectionError(
            f"audit detail row count mismatch: expected {prefix_audit['comparison_count']}, got {len(rows)}"
        )
    schema = _audit_detail_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    if tuple(table.column_names) != AUDIT_DETAIL_COLUMNS:
        raise StateProjectionError(f"audit detail columns changed: {tuple(table.column_names)}")
    pq.write_table(table, output_path, compression="zstd", write_statistics=True)
    physical = pq.read_schema(output_path)
    published_rows = int(pq.ParquetFile(output_path).metadata.num_rows)
    if physical != schema or published_rows != len(rows):
        raise StateProjectionError("published audit-detail schema or row count differs from the frozen table")
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "columns": list(AUDIT_DETAIL_COLUMNS),
        "schema": {field.name: str(field.type) for field in schema},
        "rows": published_rows,
    }


def _merge_parts(generated: Sequence[dict[str, Any]], output_path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("dt", pa.timestamp("ns"), nullable=False),
            pa.field("regime", pa.int8(), nullable=False),
        ]
    )
    writer: pq.ParquetWriter | None = None
    try:
        for item in sorted(generated, key=lambda value: int(value["index"])):
            if item["status"] != "generated":
                continue
            table = pq.read_table(item["part_path"], columns=list(STATE_COLUMNS)).cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression="zstd",
                    use_dictionary=["symbol"],
                    write_statistics=True,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None or not output_path.is_file():
        raise StateProjectionError("no source projection was generated")
    physical = pq.read_schema(output_path)
    if tuple(physical.names) != STATE_COLUMNS:
        raise StateProjectionError(f"published projection columns changed: {tuple(physical.names)}")
    expected_rows = sum(int(item["state_rows"]) for item in generated)
    published_rows = int(pq.ParquetFile(output_path).metadata.num_rows)
    if published_rows != expected_rows:
        raise StateProjectionError(f"published row count mismatch: expected {expected_rows}, got {published_rows}")


def _engine_provenance() -> dict[str, Any]:
    trend_regime = get_trend_regime()
    native = sys.modules.get("czsc._native")
    if native is None:
        raise StateProjectionError("czsc._native was not loaded by the compatibility shim")
    native_path = Path(native.__file__).resolve()
    trend_path = Path(trend_regime.__file__).resolve()
    return {
        "cache_builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "trend_regime": {"path": str(trend_path), "sha256": sha256_file(trend_path)},
        "format_standard_kline": {"path": str(FORMAT_KLINE_PATH), "sha256": sha256_file(FORMAT_KLINE_PATH)},
        "native_extension": {"path": str(native_path), "sha256": sha256_file(native_path)},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }


def _source_fingerprints(source_paths: Sequence[Path]) -> list[dict[str, Any]]:
    fingerprints = []
    for index, path in enumerate(source_paths):
        resolved = path.resolve()
        fingerprints.append(
            {
                "index": index,
                "path": str(resolved),
                "name": resolved.name,
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return fingerprints


def _cache_identity(
    config: StateCacheConfig,
    provenance: dict[str, Any],
    sources: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "projection_columns": list(STATE_COLUMNS),
        "valid_regimes": sorted(VALID_REGIMES),
        "algorithm_hashes": {
            name: entry["sha256"] for name, entry in provenance.items() if isinstance(entry, dict) and "sha256" in entry
        },
        "runtime_versions": provenance["versions"],
        "sources": [{"name": item["name"], "size": item["size"], "sha256": item["sha256"]} for item in sources],
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return digest, identity


def build_state_cache(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = 1,
    config: StateCacheConfig | None = None,
    source_files: Sequence[str | Path] | None = None,
) -> StateCacheBuildResult:
    """Build an atomically published, content-addressed and overwrite-protected cache."""
    selected_config = config or StateCacheConfig()
    selected_config.validate()
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if source_files is None:
        sources = sorted(Path(data_dir).resolve().glob("*.parquet"))
    else:
        sources = sorted({Path(path).resolve() for path in source_files}, key=str)
    if not sources:
        raise StateCacheInputError(f"no parquet sources found in {Path(data_dir).resolve()}")
    if any(not source.is_file() for source in sources):
        missing = [str(source) for source in sources if not source.is_file()]
        raise StateCacheInputError(f"source files do not exist: {missing}")

    fingerprints = _source_fingerprints(sources)
    provenance = _engine_provenance()
    digest, identity = _cache_identity(selected_config, provenance, fingerprints)
    cache_id = digest
    cache_dir = root / f"CHAN_STATE_CACHE_{cache_id}"
    if cache_dir.exists():
        raise FileExistsError(f"content-addressed cache already exists and cannot be overwritten: {cache_dir}")

    temporary = Path(tempfile.mkdtemp(prefix=f".tmp_{cache_id}_", dir=root))
    parts_dir = temporary / "parts"
    parts_dir.mkdir()
    try:
        tasks = [
            _BuildTask(
                index=int(item["index"]),
                source_path=str(item["path"]),
                source_sha256=str(item["sha256"]),
                part_path=str(parts_dir / f"{int(item['index']):06d}.parquet"),
            )
            for item in fingerprints
        ]
        generated = _run_build_tasks(tasks, min(workers, len(tasks)))
        symbols = [str(item["symbol"]) for item in generated]
        duplicate_symbols = sorted(symbol for symbol, count in Counter(symbols).items() if count > 1)
        if duplicate_symbols:
            raise StateCacheInputError(f"multiple source files resolve to the same symbol: {duplicate_symbols}")

        prefix_audit = run_prefix_audit(generated, selected_config, min(workers, len(tasks)))
        if not prefix_audit["passed"]:
            raise StateProjectionError(
                "prefix/full causality audit failed; refusing to publish a content-addressed state cache"
            )
        projection_path = temporary / "states.parquet"
        _merge_parts(generated, projection_path)
        audit_parquet_path = temporary / "state_audit.parquet"
        audit_detail_meta = write_audit_details(prefix_audit, audit_parquet_path)
        projection_rows = sum(int(item["state_rows"]) for item in generated)
        combined_counts = Counter({str(regime): 0 for regime in range(11)})
        for item in generated:
            combined_counts.update(item["state_counts"])
        projection_meta = {
            "path": "states.parquet",
            "sha256": sha256_file(projection_path),
            "columns": list(STATE_COLUMNS),
            "physical_schema": str(pq.read_schema(projection_path)),
            "rows": projection_rows,
            "symbols": sum(item["status"] == "generated" for item in generated),
            "regime_counts": {str(regime): int(combined_counts[str(regime)]) for regime in range(11)},
        }
        state_audit = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": cache_id,
            "gate_name": "full_0_to_10_chan_state_cache",
            "projection_contract": {
                "columns": list(STATE_COLUMNS),
                "valid_regimes": sorted(VALID_REGIMES),
                "all_regimes_retained": True,
                "whole_file_min_bars_filter_used": False,
                "generation_rule": "every_non_empty_source",
                "warmup_policy": "first_120_observed_bars_are_explicit_regime_0",
                "first_observed_pct_chg_null_policy": "causally_normalized_to_zero_only_on_earliest_bar",
            },
            "content_addressed": True,
            "overwrite_protected": True,
            "input_summary": {
                "source_files": len(generated),
                "generated_files": sum(item["status"] == "generated" for item in generated),
                "skipped_files": sum(item["status"] != "generated" for item in generated),
                "validation_errors": 0,
            },
            "projection": projection_meta,
            "prefix_full_audit": prefix_audit,
            "audit_details": audit_detail_meta,
            "mismatch_count": prefix_audit["mismatch_count"],
            "passed": bool(prefix_audit["passed"]),
            "data_gate_ready": bool(prefix_audit["passed"]),
        }
        audit_path = temporary / "state_audit.json"
        _write_json(audit_path, state_audit)

        source_records = []
        for fingerprint, result in zip(fingerprints, generated, strict=True):
            record = {**fingerprint, **{key: value for key, value in result.items() if key != "part_path"}}
            source_records.append(record)
        source_manifest = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": cache_id,
            "content_digest_sha256": digest,
            "identity": identity,
            "engine": provenance,
            "config": asdict(selected_config),
            "projection": projection_meta,
            "state_audit": {"path": "state_audit.json", "sha256": sha256_file(audit_path)},
            "state_audit_details": audit_detail_meta,
            "sources": source_records,
        }
        manifest_path = temporary / "source_manifest.json"
        _write_json(manifest_path, source_manifest)
        shutil.rmtree(parts_dir)

        if cache_dir.exists():
            raise FileExistsError(f"cache appeared during build and will not be overwritten: {cache_dir}")
        os.rename(temporary, cache_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return StateCacheBuildResult(
        cache_id=cache_id,
        cache_dir=cache_dir,
        projection_path=cache_dir / "states.parquet",
        manifest_path=cache_dir / "source_manifest.json",
        audit_path=cache_dir / "state_audit.json",
        audit_parquet_path=cache_dir / "state_audit.parquet",
        audit_passed=bool(prefix_audit["passed"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 8))
    parser.add_argument("--audit-symbols", type=int, default=DEFAULT_AUDIT_SYMBOLS)
    parser.add_argument("--audit-checkpoints", type=int, default=DEFAULT_AUDIT_CHECKPOINTS)
    parser.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = StateCacheConfig(
        audit_symbols=args.audit_symbols,
        audit_checkpoints=args.audit_checkpoints,
        audit_seed=args.audit_seed,
    )
    try:
        result = build_state_cache(
            args.data_dir,
            args.output_root,
            workers=args.workers,
            config=config,
        )
    except (StateCacheError, FileExistsError, ValueError) as exc:
        print(f"[FAIL-CLOSED] {exc}", file=sys.stderr)
        return 2
    print(f"[CACHE] {result.cache_dir}")
    print(f"[AUDIT] {'PASS' if result.audit_passed else 'FAIL'}: {result.audit_path}")
    return 0 if result.audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
