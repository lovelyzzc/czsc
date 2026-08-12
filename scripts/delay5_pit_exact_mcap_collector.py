"""Collect and verify point-in-time ``daily_basic`` market capitalisation.

The collector is deliberately separate from the outcome analysis.  It accepts only the
frozen delay5 exact-mcap request manifest v2, downloads one complete Tushare cross-section
per decision date, and stores canonical content-addressed objects.  A final materialised
parquet is published only after every requested key closes.

Examples::

    uv run --with tinyshare==0.1028.0 python scripts/delay5_pit_exact_mcap_collector.py collect
    uv run --no-sync python scripts/delay5_pit_exact_mcap_collector.py verify

The command line reads credentials exclusively from ``TINYSHARE_TOKEN`` or
``TUSHARE_TOKEN``.  Tests and library callers can instead inject a ``pro`` client.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REQUEST_PATH = SCRIPT_DIR / "_output" / "delay5_common_horizon_att" / "exact_mcap_request_manifest.json"
CACHE_ENV = "CZSC_DELAY5_EXACT_MCAP_CACHE"
TOKEN_ENVS = ("TINYSHARE_TOKEN", "TUSHARE_TOKEN")

REQUEST_SCHEMA = "delay5_exact_mcap_request_manifest_v2"
OBJECT_SCHEMA = "delay5_tushare_daily_basic_object_v1"
JOURNAL_SCHEMA = "delay5_pit_exact_mcap_journal_v1"
FINAL_SCHEMA = "delay5_pit_exact_mcap_collection_v1"
VERIFY_SCHEMA = "delay5_pit_exact_mcap_verification_v1"
SOURCE = "tushare.daily_basic"
DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "close",
    "total_share",
    "float_share",
    "total_mv",
    "circ_mv",
)
NUMERIC_FIELDS = DAILY_BASIC_FIELDS[2:]
MIN_CROSS_SECTION_ROWS = 4_000
MAX_CROSS_SECTION_ROWS_EXCLUSIVE = 6_000
DEFAULT_RATE_HZ = 1.0
DEFAULT_MAX_ATTEMPTS = 3
PORTABLE_REQUEST_FIELDS = (
    "schema",
    "canonicalization",
    "cohort",
    "request_identity",
    "request_records",
    "treated_request_identity",
    "treated_request_records",
    "request_dates_identity",
    "request_dates",
    "request_symbols_identity",
    "request_symbols",
    "closure",
)

MATERIALIZED_COLUMNS = (
    "symbol",
    "dec_dt",
    "close_cny",
    "total_share_10k_shares",
    "float_share_10k_shares",
    "total_mv_10k_cny",
    "circ_mv_10k_cny",
    "exact_circ_mv_cny",
    "source",
    "source_object_sha256",
)


class ManifestError(ValueError):
    """The frozen request manifest is malformed or has identity drift."""


class CrossSectionError(ValueError):
    """A provider response violates a per-date hard gate."""


class CollectionError(RuntimeError):
    """The requested collection could not close after bounded retries."""


class VerificationError(RuntimeError):
    """A persisted collection failed integrity verification."""


@dataclass(frozen=True)
class RequestPlan:
    """Validated, immutable projection of the request manifest."""

    request_records: tuple[tuple[str, str], ...]
    treated_request_records: tuple[tuple[str, str], ...]
    request_dates: tuple[str, ...]
    request_symbols: tuple[str, ...]
    request_sha256: str
    treated_sha256: str
    dates_sha256: str
    symbols_sha256: str
    treated_trades: int

    @property
    def requested_by_date(self) -> dict[str, frozenset[str]]:
        grouped: dict[str, set[str]] = {date_text: set() for date_text in self.request_dates}
        for symbol, date_text in self.request_records:
            grouped[date_text].add(symbol)
        return {date_text: frozenset(symbols) for date_text, symbols in grouped.items()}


def default_cache_root() -> Path:
    """Return the portable cache root, honouring the dedicated environment override."""

    configured = os.environ.get(CACHE_ENV)
    return Path(configured).expanduser() if configured else Path.home() / ".ts_data_cache" / "delay5_exact_mcap_v1"


def canonical_json(value: Any) -> bytes:
    """Encode deterministic compact JSON as UTF-8 without a trailing newline."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_records(records: list[Any] | tuple[Any, ...]) -> bytes:
    """Encode a lexicographically sorted top-level record array."""

    return json.dumps(
        sorted(records),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_identity(records: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Return the exact identity format used by the v2 request producer."""

    ordered = sorted(records)
    payload = canonical_records(ordered)
    return {
        "count": len(ordered),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first3": ordered[:3],
        "last3": ordered[-3:] if ordered else [],
    }


def query_identity(plan: RequestPlan) -> dict[str, Any]:
    """Freeze the exact provider calls implied by a validated request."""

    records = [[SOURCE, date_text.replace("-", ""), ",".join(DAILY_BASIC_FIELDS)] for date_text in plan.request_dates]
    return payload_identity(records)


def dependency_versions() -> dict[str, str | None]:
    """Record runtime package provenance without inspecting credentials."""

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in ("pandas", "tinyshare", "tushare"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_pair_records(value: Any, *, name: str) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{name} must be a non-empty list")
    records: list[list[str]] = []
    for index, record in enumerate(value):
        if not isinstance(record, list) or len(record) != 2:
            raise ManifestError(f"{name}[{index}] must be [symbol, YYYY-MM-DD]")
        symbol, date_text = record
        if not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip():
            raise ManifestError(f"{name}[{index}] has invalid symbol")
        try:
            parsed = datetime.strptime(str(date_text), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError as exc:
            raise ManifestError(f"{name}[{index}] has invalid date") from exc
        if parsed != date_text:
            raise ManifestError(f"{name}[{index}] date is not canonical")
        records.append([symbol, date_text])
    if records != sorted(records) or len(records) != len({tuple(record) for record in records}):
        raise ManifestError(f"{name} must be sorted and unique")
    return records


def _require_identity(manifest: dict[str, Any], key: str, records: list[Any]) -> dict[str, Any]:
    actual = manifest.get(key)
    expected = payload_identity(records)
    if actual != expected:
        raise ManifestError(f"{key} does not match canonical records")
    return expected


def load_request_manifest(path: Path = DEFAULT_REQUEST_PATH) -> RequestPlan:
    """Load and fully validate a delay5 exact-mcap request manifest v2."""

    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read request manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != REQUEST_SCHEMA:
        raise ManifestError(f"request manifest schema must be {REQUEST_SCHEMA}")

    request_records = _normalise_pair_records(manifest.get("request_records"), name="request_records")
    treated_records = _normalise_pair_records(manifest.get("treated_request_records"), name="treated_request_records")
    request_identity = _require_identity(manifest, "request_identity", request_records)
    treated_identity = _require_identity(manifest, "treated_request_identity", treated_records)

    dates = manifest.get("request_dates")
    symbols = manifest.get("request_symbols")
    if not isinstance(dates, list) or dates != sorted({date_text for _, date_text in request_records}):
        raise ManifestError("request_dates does not close over request_records")
    if not isinstance(symbols, list) or symbols != sorted({symbol for symbol, _ in request_records}):
        raise ManifestError("request_symbols does not close over request_records")
    dates_identity = _require_identity(manifest, "request_dates_identity", dates)
    symbols_identity = _require_identity(manifest, "request_symbols_identity", symbols)

    request_keys = {tuple(record) for record in request_records}
    treated_keys = {tuple(record) for record in treated_records}
    if not treated_keys.issubset(request_keys):
        raise ManifestError("not every treated key is requested")
    cohort = manifest.get("cohort")
    if not isinstance(cohort, dict):
        raise ManifestError("cohort is absent")
    if cohort.get("treated_trades") != len(treated_records):
        raise ManifestError("cohort treated_trades does not equal treated request closure")
    if cohort.get("decision_dates") != len(dates):
        raise ManifestError("cohort decision_dates does not equal request date closure")
    expected_closure = {
        "request_keys_unique": True,
        "treated_keys_unique": True,
        "all_treated_requested": True,
    }
    if manifest.get("closure") != expected_closure:
        raise ManifestError("request closure gates are not all true")

    return RequestPlan(
        request_records=tuple(tuple(record) for record in request_records),
        treated_request_records=tuple(tuple(record) for record in treated_records),
        request_dates=tuple(dates),
        request_symbols=tuple(symbols),
        request_sha256=request_identity["sha256"],
        treated_sha256=treated_identity["sha256"],
        dates_sha256=dates_identity["sha256"],
        symbols_sha256=symbols_identity["sha256"],
        treated_trades=len(treated_records),
    )


def portable_request_projection(path: Path = DEFAULT_REQUEST_PATH) -> dict[str, Any]:
    """Return the validated request without machine-local cache diagnostics."""

    load_request_manifest(path)
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read request manifest: {exc}") from exc
    missing = set(PORTABLE_REQUEST_FIELDS) - set(manifest)
    if missing:
        raise ManifestError(f"portable request projection missing fields: {sorted(missing)}")
    return {field: manifest[field] for field in PORTABLE_REQUEST_FIELDS}


def _date_text(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _finite_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalise_cross_section(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    requested_symbols: set[str] | frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one complete provider response and return its canonical object."""

    if not isinstance(frame, pd.DataFrame):
        raise CrossSectionError("daily_basic response is not a DataFrame")
    missing_fields = set(DAILY_BASIC_FIELDS) - set(frame.columns)
    if missing_fields:
        raise CrossSectionError(f"daily_basic response missing fields: {sorted(missing_fields)}")
    if frame.empty:
        raise CrossSectionError("daily_basic response is empty")
    if len(frame) < MIN_CROSS_SECTION_ROWS:
        raise CrossSectionError(f"daily_basic response has fewer than {MIN_CROSS_SECTION_ROWS} rows")
    if len(frame) >= MAX_CROSS_SECTION_ROWS_EXCLUSIVE:
        raise CrossSectionError(
            f"daily_basic response has at least {MAX_CROSS_SECTION_ROWS_EXCLUSIVE} rows; possible truncation"
        )

    expected_date = _date_text(trade_date)
    if expected_date != trade_date:
        raise CrossSectionError("requested trade date must use YYYY-MM-DD")
    normalised_dates = frame["trade_date"].map(_date_text)
    if normalised_dates.isna().any() or not normalised_dates.eq(expected_date).all():
        raise CrossSectionError("daily_basic response contains a wrong or invalid trade_date")
    symbols = frame["ts_code"].astype("string")
    if symbols.isna().any() or symbols.str.strip().ne(symbols).any() or symbols.str.len().eq(0).any():
        raise CrossSectionError("daily_basic response contains an invalid ts_code")
    if symbols.duplicated().any():
        raise CrossSectionError("daily_basic response contains duplicate symbols")

    available = set(symbols.astype(str))
    missing_requested = sorted(set(requested_symbols) - available)
    if missing_requested:
        raise CrossSectionError(f"requested symbols are missing: {missing_requested[:5]}")

    projected: list[dict[str, Any]] = []
    null_counts = dict.fromkeys(NUMERIC_FIELDS, 0)
    for source_row in frame.loc[:, DAILY_BASIC_FIELDS].itertuples(index=False, name=None):
        symbol = str(source_row[0])
        row: dict[str, Any] = {"ts_code": symbol, "trade_date": expected_date}
        for field, raw_value in zip(NUMERIC_FIELDS, source_row[2:], strict=True):
            value = _finite_number(raw_value)
            row[field] = value
            if value is None and symbol not in requested_symbols:
                null_counts[field] += 1
        if symbol in requested_symbols and (row["circ_mv"] is None or row["circ_mv"] <= 0):
            raise CrossSectionError(f"requested symbol has invalid circ_mv: {symbol}")
        projected.append(row)
    projected.sort(key=lambda row: row["ts_code"])
    diagnostics = {
        "rows": len(projected),
        "requested_symbols": len(requested_symbols),
        "non_requested_null_counts": {field: max(0, count) for field, count in null_counts.items() if count},
    }
    content = {
        "schema": OBJECT_SCHEMA,
        "source": SOURCE,
        "trade_date": expected_date,
        "fields": list(DAILY_BASIC_FIELDS),
        "rows": projected,
    }
    return content, diagnostics


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json(value))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _portable_path(path_text: str) -> PurePosixPath:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise VerificationError(f"artifact path is not portable: {path_text}")
    return path


def _object_relpath(object_sha: str) -> str:
    return f"objects/daily_basic/{object_sha}.json"


def _journal_path(cache_root: Path, request_sha: str) -> Path:
    return cache_root / "journals" / f"{request_sha}.json"


def _request_snapshot_path(cache_root: Path, request_sha: str) -> Path:
    return cache_root / "requests" / f"{request_sha}.json"


def _manifest_path(cache_root: Path, request_sha: str) -> Path:
    return cache_root / "manifests" / f"exact_mcap_{request_sha}.json"


def _materialized_path(cache_root: Path, request_sha: str) -> Path:
    return cache_root / "materialized" / f"exact_mcap_{request_sha}.parquet"


def _report_path(cache_root: Path, request_sha: str) -> Path:
    return cache_root / "reports" / f"verify_{request_sha}.json"


def _new_journal(plan: RequestPlan) -> dict[str, Any]:
    return {
        "schema": JOURNAL_SCHEMA,
        "request_sha256": plan.request_sha256,
        "dates": {},
    }


def _load_journal(cache_root: Path, plan: RequestPlan) -> dict[str, Any]:
    path = _journal_path(cache_root, plan.request_sha256)
    if not path.exists():
        return _new_journal(plan)
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"journal cannot be read safely: {exc}") from exc
    if journal.get("schema") != JOURNAL_SCHEMA or journal.get("request_sha256") != plan.request_sha256:
        raise CollectionError("journal identity mismatch")
    if not isinstance(journal.get("dates"), dict):
        raise CollectionError("journal date map is malformed")
    return journal


def _read_verified_object(
    cache_root: Path,
    entry: dict[str, Any],
    *,
    trade_date: str,
    requested_symbols: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_sha = entry.get("object_sha256")
    path_text = entry.get("object_path")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or not isinstance(path_text, str):
        raise VerificationError("journal object reference is malformed")
    path = cache_root / _portable_path(path_text)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise VerificationError(f"object hash mismatch for {trade_date}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"object JSON is invalid for {trade_date}") from exc
    if canonical_json(content) != path.read_bytes():
        raise VerificationError(f"object is not canonical for {trade_date}")
    if content.get("schema") != OBJECT_SCHEMA or content.get("source") != SOURCE:
        raise VerificationError(f"object schema/source mismatch for {trade_date}")
    frame = pd.DataFrame(content.get("rows"))
    normalised, diagnostics = normalise_cross_section(
        frame,
        trade_date=trade_date,
        requested_symbols=requested_symbols,
    )
    if normalised != content:
        raise VerificationError(f"object projection drift for {trade_date}")
    return content, diagnostics


def _fetch_one_date(
    pro: Any,
    *,
    trade_date: str,
    requested_symbols: frozenset[str],
    max_attempts: int,
    retry_sleep: Callable[[float], None],
    retry_delay: float,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            frame = pro.daily_basic(
                trade_date=trade_date.replace("-", ""),
                fields=",".join(DAILY_BASIC_FIELDS),
            )
            content, diagnostics = normalise_cross_section(
                frame,
                trade_date=trade_date,
                requested_symbols=requested_symbols,
            )
            return content, diagnostics, attempt
        except Exception as exc:  # provider exceptions and hard-gate failures share the bounded retry policy
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < max_attempts and retry_delay > 0:
                retry_sleep(retry_delay)
    raise CollectionError(f"daily_basic {trade_date} failed after {max_attempts} attempts; last error: {errors[-1]}")


def _materialized_rows_from_object(
    content: dict[str, Any],
    *,
    trade_date: str,
    requested_symbols: frozenset[str],
    object_sha256: str,
) -> list[dict[str, Any]]:
    """Project requested rows from one verified object using the public parquet contract."""

    by_symbol = {row["ts_code"]: row for row in content["rows"]}
    rows: list[dict[str, Any]] = []
    for symbol in sorted(requested_symbols):
        source_row = by_symbol[symbol]
        circ_mv = float(source_row["circ_mv"])
        rows.append(
            {
                "symbol": symbol,
                "dec_dt": pd.Timestamp(trade_date),
                "close_cny": source_row["close"],
                "total_share_10k_shares": source_row["total_share"],
                "float_share_10k_shares": source_row["float_share"],
                "total_mv_10k_cny": source_row["total_mv"],
                "circ_mv_10k_cny": circ_mv,
                "exact_circ_mv_cny": circ_mv * 10_000.0,
                "source": SOURCE,
                "source_object_sha256": object_sha256,
            }
        )
    return rows


def _sorted_materialized_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the deterministic parquet row order shared by publisher and verifier."""

    frame = pd.DataFrame(rows, columns=MATERIALIZED_COLUMNS)
    frame["dec_dt"] = pd.to_datetime(frame["dec_dt"]).dt.normalize()
    return frame.sort_values(["symbol", "dec_dt"], kind="mergesort").reset_index(drop=True)


def _require_materialized_matches_objects(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Fail unless every parquet value is a null-safe exact projection of source objects."""

    if len(actual) != len(expected):
        raise VerificationError("materialized/object row count mismatch")
    actual = _sorted_materialized_frame(actual.to_dict(orient="records"))
    expected = _sorted_materialized_frame(expected.to_dict(orient="records"))
    numeric_columns = MATERIALIZED_COLUMNS[2:8]
    text_columns = ("symbol", "source", "source_object_sha256")

    for column in text_columns:
        actual_null = actual[column].isna()
        expected_null = expected[column].isna()
        if not actual_null.equals(expected_null):
            raise VerificationError(f"materialized/object null mismatch in {column}")
        if not actual.loc[~actual_null, column].astype(str).equals(expected.loc[~expected_null, column].astype(str)):
            raise VerificationError(f"materialized/object value mismatch in {column}")

    if not actual["dec_dt"].equals(expected["dec_dt"]):
        raise VerificationError("materialized/object value mismatch in dec_dt")

    for column in numeric_columns:
        actual_null = actual[column].isna()
        expected_null = expected[column].isna()
        if not actual_null.equals(expected_null):
            raise VerificationError(f"materialized/object null mismatch in {column}")
        actual_values = pd.to_numeric(actual.loc[~actual_null, column], errors="coerce")
        expected_values = pd.to_numeric(expected.loc[~expected_null, column], errors="coerce")
        if actual_values.isna().any() or not actual_values.equals(expected_values):
            raise VerificationError(f"materialized/object value mismatch in {column}")


def _materialize(
    cache_root: Path,
    plan: RequestPlan,
    journal: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    requested = plan.requested_by_date
    rows: list[dict[str, Any]] = []
    object_entries: list[dict[str, Any]] = []
    for trade_date in plan.request_dates:
        entry = journal["dates"].get(trade_date)
        if not isinstance(entry, dict):
            raise CollectionError(f"journal has no closed object for {trade_date}")
        content, diagnostics = _read_verified_object(
            cache_root,
            entry,
            trade_date=trade_date,
            requested_symbols=requested[trade_date],
        )
        rows.extend(
            _materialized_rows_from_object(
                content,
                trade_date=trade_date,
                requested_symbols=requested[trade_date],
                object_sha256=entry["object_sha256"],
            )
        )
        object_entries.append(
            {
                "trade_date": trade_date,
                "object_sha256": entry["object_sha256"],
                "object_path": entry["object_path"],
                "rows": diagnostics["rows"],
                "requested_symbols": diagnostics["requested_symbols"],
                "non_requested_null_counts": diagnostics["non_requested_null_counts"],
            }
        )
    result = _sorted_materialized_frame(rows)
    actual_keys = set(zip(result["symbol"], result["dec_dt"].dt.strftime("%Y-%m-%d"), strict=True))
    if actual_keys != set(plan.request_records) or len(result) != len(plan.request_records):
        raise CollectionError("materialized request-key closure failed")
    return result, object_entries


def collect_exact_mcap(
    *,
    request_path: Path = DEFAULT_REQUEST_PATH,
    cache_root: Path | None = None,
    pro: Any | None = None,
    rate_hz: float = DEFAULT_RATE_HZ,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect all request dates, materialise exact CNY units, and verify the result."""

    plan = load_request_manifest(request_path)
    portable_request = portable_request_projection(request_path)
    portable_request_payload = canonical_json(portable_request)
    portable_request_sha = hashlib.sha256(portable_request_payload).hexdigest()
    request_file_sha = sha256_file(Path(request_path))
    query = query_identity(plan)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    if rate_hz < 0:
        raise ValueError("rate_hz must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if pro is None:
        pro = create_env_client()
    request_snapshot_path = _request_snapshot_path(root, plan.request_sha256)
    _atomic_bytes(request_snapshot_path, portable_request_payload)
    journal = _load_journal(root, plan)
    requested = plan.requested_by_date
    interval = 1.0 / rate_hz if rate_hz else 0.0
    last_api_call_at: float | None = None

    for trade_date in plan.request_dates:
        entry = journal["dates"].get(trade_date)
        if isinstance(entry, dict):
            try:
                _read_verified_object(
                    root,
                    entry,
                    trade_date=trade_date,
                    requested_symbols=requested[trade_date],
                )
                continue
            except VerificationError:
                pass
        if last_api_call_at is not None and interval:
            remaining = interval - (time.monotonic() - last_api_call_at)
            if remaining > 0:
                sleep(remaining)
        content, diagnostics, attempts = _fetch_one_date(
            pro,
            trade_date=trade_date,
            requested_symbols=requested[trade_date],
            max_attempts=max_attempts,
            retry_sleep=sleep,
            retry_delay=interval,
        )
        last_api_call_at = time.monotonic()
        payload = canonical_json(content)
        object_sha = hashlib.sha256(payload).hexdigest()
        object_relpath = _object_relpath(object_sha)
        _atomic_bytes(root / object_relpath, payload)
        journal["dates"][trade_date] = {
            "object_sha256": object_sha,
            "object_path": object_relpath,
            "rows": diagnostics["rows"],
            "attempts_last_fetch": attempts,
        }
        _atomic_json(_journal_path(root, plan.request_sha256), journal)

    materialized, objects = _materialize(root, plan, journal)
    materialized_path = _materialized_path(root, plan.request_sha256)
    _atomic_parquet(materialized_path, materialized)
    materialized_sha = sha256_file(materialized_path)
    final_manifest = {
        "schema": FINAL_SCHEMA,
        "source": SOURCE,
        "provenance": {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_manifest_file_sha256": request_file_sha,
            "request_file_sha_is_local_diagnostic": True,
            "query_canonical_sha256": query["sha256"],
            "query_count": query["count"],
            "runtime_versions": dependency_versions(),
            "credential_recorded": False,
            "historical_exactness": (
                "point-in-time value for each historical trade_date; provider revision-vintage/as-of history "
                "is not available and is not claimed"
            ),
        },
        "request": {
            "sha256": plan.request_sha256,
            "keys": len(plan.request_records),
            "dates": len(plan.request_dates),
            "symbols": len(plan.request_symbols),
            "treated_sha256": plan.treated_sha256,
            "dates_sha256": plan.dates_sha256,
            "symbols_sha256": plan.symbols_sha256,
            "treated_trades": plan.treated_trades,
        },
        "portable_request": {
            "path": request_snapshot_path.relative_to(root).as_posix(),
            "sha256": portable_request_sha,
            "schema": REQUEST_SCHEMA,
        },
        "provider_contract": {
            "endpoint": "daily_basic",
            "fields": list(DAILY_BASIC_FIELDS),
            "one_complete_cross_section_per_date": True,
            "source_mixing": False,
        },
        "units": {
            "close_cny": "CNY/share",
            "total_share_10k_shares": "10,000 shares",
            "float_share_10k_shares": "10,000 shares",
            "total_mv_10k_cny": "10,000 CNY",
            "circ_mv_10k_cny": "10,000 CNY",
            "exact_circ_mv_cny": "CNY; circ_mv_10k_cny * 10,000",
        },
        "materialized": {
            "path": materialized_path.relative_to(root).as_posix(),
            "sha256": materialized_sha,
            "rows": len(materialized),
            "columns": list(MATERIALIZED_COLUMNS),
        },
        "objects": objects,
        "closure": {
            "all_request_keys_materialized": len(materialized) == len(plan.request_records),
            "all_request_dates_verified": len(objects) == len(plan.request_dates),
            "all_treated_requested": set(plan.treated_request_records).issubset(set(plan.request_records)),
        },
    }
    manifest_path = _manifest_path(root, plan.request_sha256)
    _atomic_json(manifest_path, final_manifest)
    verify_collection(request_path=request_path, cache_root=root, write_report=True)
    return final_manifest


def _load_final_manifest(root: Path, plan: RequestPlan) -> dict[str, Any]:
    path = _manifest_path(root, plan.request_sha256)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"final manifest cannot be read: {exc}") from exc
    if manifest.get("schema") != FINAL_SCHEMA or manifest.get("source") != SOURCE:
        raise VerificationError("final manifest schema/source mismatch")
    return manifest


def verify_collection(
    *,
    request_path: Path = DEFAULT_REQUEST_PATH,
    cache_root: Path | None = None,
    write_report: bool = True,
    _manifest_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify request binding, every object, parquet integrity, units, and closure."""

    plan = load_request_manifest(request_path)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    manifest = _manifest_override if _manifest_override is not None else _load_final_manifest(root, plan)
    if manifest.get("schema") != FINAL_SCHEMA or manifest.get("source") != SOURCE:
        raise VerificationError("final manifest schema/source mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise VerificationError("final manifest provenance is absent")
    query = query_identity(plan)
    local_request_sha = provenance.get("request_manifest_file_sha256")
    if not isinstance(local_request_sha, str) or len(local_request_sha) != 64:
        raise VerificationError("local request manifest file hash diagnostic is malformed")
    if provenance.get("request_file_sha_is_local_diagnostic") is not True:
        raise VerificationError("request file hash must be marked as a local diagnostic")
    if provenance.get("query_canonical_sha256") != query["sha256"] or provenance.get("query_count") != query["count"]:
        raise VerificationError("provider query binding mismatch")
    if provenance.get("credential_recorded") is not False:
        raise VerificationError("credential provenance gate failed")
    versions = provenance.get("runtime_versions")
    if not isinstance(versions, dict) or set(versions) != {"python", "pandas", "tinyshare", "tushare"}:
        raise VerificationError("runtime version provenance is malformed")
    if not isinstance(versions["python"], str) or not isinstance(versions["pandas"], str):
        raise VerificationError("required runtime version provenance is absent")
    collected_at = provenance.get("collected_at_utc")
    if not isinstance(collected_at, str):
        raise VerificationError("collection timestamp provenance is invalid")
    try:
        collected_timestamp = pd.Timestamp(collected_at)
    except (TypeError, ValueError) as exc:
        raise VerificationError("collection timestamp provenance is invalid") from exc
    if collected_timestamp.tzinfo is None or collected_timestamp.utcoffset().total_seconds() != 0:
        raise VerificationError("collection timestamp provenance is not UTC")
    exactness = provenance.get("historical_exactness")
    if not isinstance(exactness, str) or "revision-vintage" not in exactness or "not claimed" not in exactness:
        raise VerificationError("historical exactness provenance is malformed")
    request = manifest.get("request", {})
    expected_request = {
        "sha256": plan.request_sha256,
        "keys": len(plan.request_records),
        "dates": len(plan.request_dates),
        "symbols": len(plan.request_symbols),
        "treated_sha256": plan.treated_sha256,
        "dates_sha256": plan.dates_sha256,
        "symbols_sha256": plan.symbols_sha256,
        "treated_trades": plan.treated_trades,
    }
    if request != expected_request:
        raise VerificationError("final manifest request binding mismatch")

    portable_meta = manifest.get("portable_request")
    if not isinstance(portable_meta, dict) or portable_meta.get("schema") != REQUEST_SCHEMA:
        raise VerificationError("portable request metadata is malformed")
    portable_path_text = portable_meta.get("path")
    portable_sha = portable_meta.get("sha256")
    if not isinstance(portable_path_text, str) or not isinstance(portable_sha, str) or len(portable_sha) != 64:
        raise VerificationError("portable request reference is malformed")
    portable_path = root / _portable_path(portable_path_text)
    if not portable_path.is_file() or sha256_file(portable_path) != portable_sha:
        raise VerificationError("portable request snapshot hash mismatch")
    try:
        portable_document = json.loads(portable_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError("portable request snapshot JSON is invalid") from exc
    if canonical_json(portable_document) != portable_path.read_bytes():
        raise VerificationError("portable request snapshot is not canonical")
    if not isinstance(portable_document, dict) or set(portable_document) != set(PORTABLE_REQUEST_FIELDS):
        raise VerificationError("portable request snapshot fields are malformed")
    try:
        portable_plan = load_request_manifest(portable_path)
    except ManifestError as exc:
        raise VerificationError(f"portable request snapshot is invalid: {exc}") from exc
    if portable_plan != plan:
        raise VerificationError("portable request canonical identity mismatch")

    if manifest.get("closure") != {
        "all_request_keys_materialized": True,
        "all_request_dates_verified": True,
        "all_treated_requested": True,
    }:
        raise VerificationError("final manifest closure is not fully true")

    object_records = manifest.get("objects")
    if not isinstance(object_records, list) or len(object_records) != len(plan.request_dates):
        raise VerificationError("final manifest object-date closure mismatch")
    by_date = {entry.get("trade_date"): entry for entry in object_records if isinstance(entry, dict)}
    if set(by_date) != set(plan.request_dates):
        raise VerificationError("final manifest contains missing/duplicate object dates")
    requested = plan.requested_by_date
    object_sha_by_date: dict[str, str] = {}
    expected_materialized_rows: list[dict[str, Any]] = []
    for trade_date in plan.request_dates:
        entry = by_date[trade_date]
        content, _ = _read_verified_object(
            root,
            entry,
            trade_date=trade_date,
            requested_symbols=requested[trade_date],
        )
        object_sha_by_date[trade_date] = entry["object_sha256"]
        expected_materialized_rows.extend(
            _materialized_rows_from_object(
                content,
                trade_date=trade_date,
                requested_symbols=requested[trade_date],
                object_sha256=entry["object_sha256"],
            )
        )

    materialized_meta = manifest.get("materialized", {})
    path_text = materialized_meta.get("path")
    if not isinstance(path_text, str):
        raise VerificationError("materialized path is absent")
    materialized_path = root / _portable_path(path_text)
    if not materialized_path.is_file() or sha256_file(materialized_path) != materialized_meta.get("sha256"):
        raise VerificationError("materialized parquet hash mismatch")
    frame = pd.read_parquet(materialized_path)
    if list(frame.columns) != list(MATERIALIZED_COLUMNS):
        raise VerificationError("materialized parquet schema mismatch")
    if len(frame) != len(plan.request_records) or len(frame) != materialized_meta.get("rows"):
        raise VerificationError("materialized parquet row closure mismatch")
    frame["dec_dt"] = pd.to_datetime(frame["dec_dt"]).dt.normalize()
    keys = list(zip(frame["symbol"], frame["dec_dt"].dt.strftime("%Y-%m-%d"), strict=True))
    if len(keys) != len(set(keys)) or set(keys) != set(plan.request_records):
        raise VerificationError("materialized parquet key closure mismatch")
    circ = pd.to_numeric(frame["circ_mv_10k_cny"], errors="coerce")
    exact = pd.to_numeric(frame["exact_circ_mv_cny"], errors="coerce")
    if circ.isna().any() or circ.le(0).any() or not exact.eq(circ.mul(10_000)).all():
        raise VerificationError("materialized exact-mcap unit conversion mismatch")
    if not frame["source"].eq(SOURCE).all():
        raise VerificationError("materialized source mismatch")
    expected_object_sha = frame["dec_dt"].dt.strftime("%Y-%m-%d").map(object_sha_by_date)
    if not frame["source_object_sha256"].eq(expected_object_sha).all():
        raise VerificationError("materialized source-object binding mismatch")
    _require_materialized_matches_objects(frame, _sorted_materialized_frame(expected_materialized_rows))

    report = {
        "schema": VERIFY_SCHEMA,
        "request_sha256": plan.request_sha256,
        "request_file_sha_is_local_diagnostic": True,
        "checks": {
            "request_binding": True,
            "provenance_binding": True,
            "portable_paths": True,
            "object_hashes_and_hard_gates": True,
            "materialized_hash_schema_and_units": True,
            "materialized_exact_object_projection": True,
            "request_and_treated_closure": True,
        },
        "counts": {
            "keys": len(plan.request_records),
            "dates": len(plan.request_dates),
            "treated_trades": plan.treated_trades,
        },
        "artifacts": {
            "manifest": _manifest_path(root, plan.request_sha256).relative_to(root).as_posix(),
            "materialized": materialized_path.relative_to(root).as_posix(),
            "portable_request": portable_path.relative_to(root).as_posix(),
        },
        "verified": True,
    }
    if write_report:
        _atomic_json(_report_path(root, plan.request_sha256), report)
    return report


def upgrade_portable_manifest(
    *,
    request_path: Path = DEFAULT_REQUEST_PATH,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically upgrade a fully valid pre-portability final manifest without API calls.

    The legacy manifest remains untouched until an in-memory upgraded candidate has passed
    the same object, parquet, projection, provenance, and closure checks as a fresh result.
    Objects and the materialised parquet are never rewritten.
    """

    plan = load_request_manifest(request_path)
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    manifest_path = _manifest_path(root, plan.request_sha256)
    manifest = _load_final_manifest(root, plan)
    provenance = manifest.get("provenance")
    request = manifest.get("request")
    if not isinstance(provenance, dict) or not isinstance(request, dict):
        raise VerificationError("legacy final manifest provenance/request is malformed")

    portability_markers = (
        "portable_request" in manifest,
        "request_file_sha_is_local_diagnostic" in provenance,
        "dates_sha256" in request,
        "symbols_sha256" in request,
    )
    if any(portability_markers):
        if not all(portability_markers):
            raise VerificationError("final manifest contains a partial portability migration")
        report = verify_collection(request_path=request_path, cache_root=root, write_report=True)
        return {
            "schema": "delay5_pit_exact_mcap_portable_migration_v1",
            "upgraded": False,
            "request_sha256": plan.request_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "report_sha256": sha256_file(_report_path(root, plan.request_sha256)),
            "verified": report["verified"],
        }

    legacy_request = {
        "sha256": plan.request_sha256,
        "keys": len(plan.request_records),
        "dates": len(plan.request_dates),
        "symbols": len(plan.request_symbols),
        "treated_sha256": plan.treated_sha256,
        "treated_trades": plan.treated_trades,
    }
    if request != legacy_request:
        raise VerificationError("legacy final manifest request binding mismatch")

    portable_document = portable_request_projection(request_path)
    portable_payload = canonical_json(portable_document)
    portable_sha = hashlib.sha256(portable_payload).hexdigest()
    portable_path = _request_snapshot_path(root, plan.request_sha256)
    _atomic_bytes(portable_path, portable_payload)

    candidate = copy.deepcopy(manifest)
    candidate["provenance"]["request_file_sha_is_local_diagnostic"] = True
    candidate["request"]["dates_sha256"] = plan.dates_sha256
    candidate["request"]["symbols_sha256"] = plan.symbols_sha256
    candidate["portable_request"] = {
        "path": portable_path.relative_to(root).as_posix(),
        "sha256": portable_sha,
        "schema": REQUEST_SCHEMA,
    }

    # Fail closed before replacing the legacy final manifest.
    verify_collection(
        request_path=request_path,
        cache_root=root,
        write_report=False,
        _manifest_override=candidate,
    )
    _atomic_json(manifest_path, candidate)
    report = verify_collection(request_path=request_path, cache_root=root, write_report=True)
    return {
        "schema": "delay5_pit_exact_mcap_portable_migration_v1",
        "upgraded": True,
        "request_sha256": plan.request_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(_report_path(root, plan.request_sha256)),
        "portable_request_sha256": portable_sha,
        "verified": report["verified"],
    }


def _env_token() -> str:
    values = [(name, os.environ.get(name, "").strip()) for name in TOKEN_ENVS]
    present = [(name, value) for name, value in values if value]
    if not present:
        raise CollectionError(f"set exactly one of {' or '.join(TOKEN_ENVS)}")
    if len({value for _, value in present}) > 1:
        raise CollectionError(f"{TOKEN_ENVS[0]} and {TOKEN_ENVS[1]} disagree")
    return present[0][1]


def create_env_client() -> Any:
    """Create a tinyshare client from environment-only credentials."""

    token = _env_token()
    try:
        import tinyshare as ts
    except ImportError as exc:
        raise CollectionError("tinyshare is required for collection") from exc
    ts.set_token(token)
    return ts.pro_api()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect/resume and publish a verified parquet")
    verify = subparsers.add_parser("verify", help="verify an existing collection without network access")
    migrate = subparsers.add_parser("migrate", help="atomically add portable bindings to a valid legacy manifest")
    for command in (collect, verify, migrate):
        command.add_argument("--request", type=Path, default=DEFAULT_REQUEST_PATH)
        command.add_argument("--cache-root", type=Path, default=None)
    collect.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    collect.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect":
        result = collect_exact_mcap(
            request_path=args.request,
            cache_root=args.cache_root,
            rate_hz=args.rate_hz,
            max_attempts=args.max_attempts,
        )
        summary = {"schema": result["schema"], "request": result["request"], "closure": result["closure"]}
    elif args.command == "verify":
        summary = verify_collection(request_path=args.request, cache_root=args.cache_root)
    else:
        summary = upgrade_portable_manifest(request_path=args.request, cache_root=args.cache_root)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
