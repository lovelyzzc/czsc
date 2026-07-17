"""Assemble the frozen V2 weekly feature surface from one validated data bundle.

The formal entry point accepts a bundle directory, never independent feature,
ADV, calendar, or universe files.  It first requires
``xs_chan_data_v2.validate_bundle(...).forward_start_allowed`` and then binds
every consumed artifact, the frozen protocol, and every output to one
content-addressed manifest.

Price features use a dense exchange-session clock.  A suspended session carries
the last adjusted close forward and contributes zero amount; PIT fields such as
free float shares and industry membership are never forward- or backward-filled.
Missing PIT data therefore makes the row ineligible instead of selecting a
replacement security.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xs_chan_data_v2 as data_v2
import xs_chan_research_v2 as research_v2

SCHEMA_VERSION = "xs_chan_features_v2_v1"
OUTPUT_PREFIX = "XS_CHAN_FEATURES_V2_"
INPUT_DATASETS = (
    "calendar",
    "security_master",
    "namechange",
    "raw_daily",
    "adj_factor",
    "daily_basic",
    "industry_membership",
)
SCHEDULE_COLUMNS = ("decision_dt", "exec_dt")
FEATURE_COLUMNS = (
    "symbol",
    "dt",
    "pit_universe_member",
    "market_sessions_since_listing",
    "valid_sessions_60",
    "is_st",
    "board",
    "adv20",
    "mom_120_20",
    "lowvol_60",
    "industry_code",
    "free_float_mcap",
    "adjusted_close",
    "sma20",
    "eligible",
    "ineligibility_reasons",
)
ALL_ADV_COLUMNS = ("symbol", "dt", "adv20")
VALID_LIST_STATUSES = frozenset({"L", "D", "P"})


class FeatureAssemblyError(RuntimeError):
    """The formal bundle cannot produce a causally valid feature surface."""


@dataclass(frozen=True)
class FeatureFrames:
    """Deterministic in-memory outputs before content-addressed publication."""

    schedule: pd.DataFrame
    features: pd.DataFrame
    all_adv: pd.DataFrame
    target_size: int


@dataclass(frozen=True)
class FeatureBuildResult:
    """Published artifact paths and their immutable run identity."""

    run_id: str
    output_dir: Path
    schedule_path: Path
    features_path: Path
    all_adv_path: Path
    manifest_path: Path


def _canonical_json(value: Any) -> bytes:
    def default(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(item).isoformat()
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            value = float(item)
            if not np.isfinite(value):
                raise ValueError("non-finite value cannot enter canonical JSON")
            return value
        raise TypeError(f"cannot encode {type(item)!r}")

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise FeatureAssemblyError(f"{label} missing columns: {missing}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise FeatureAssemblyError(f"{label} contains an invalid date")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def _normalize_calendar(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    _require_columns(calendar, ("trade_date", "is_open"), "calendar")
    if not calendar["is_open"].eq(True).all():
        raise FeatureAssemblyError("calendar must contain open exchange sessions only")
    sessions = pd.DatetimeIndex(_normalize_dates(calendar["trade_date"], "calendar.trade_date"))
    sessions = sessions.sort_values()
    if len(sessions) < 2 or sessions.has_duplicates:
        raise FeatureAssemblyError("calendar sessions must be unique and contain at least two dates")
    return sessions


def build_weekly_schedule(
    calendar: pd.DataFrame,
    last_observed_session: Any,
    first_observed_session: Any | None = None,
) -> pd.DataFrame:
    """Return completed-week close decisions followed by the next exchange session."""

    sessions = _normalize_calendar(calendar)
    cutoff = pd.Timestamp(last_observed_session).normalize()
    start = sessions[0] if first_observed_session is None else pd.Timestamp(first_observed_session).normalize()
    rows = [
        {"decision_dt": pd.Timestamp(current), "exec_dt": pd.Timestamp(following)}
        for current, following in zip(sessions[:-1], sessions[1:], strict=True)
        if current.isocalendar()[:2] != following.isocalendar()[:2] and start <= current <= cutoff
    ]
    if not rows:
        raise FeatureAssemblyError("calendar/raw coverage contains no completed weekly decision")
    schedule = pd.DataFrame(rows, columns=list(SCHEDULE_COLUMNS))
    research_v2.validate_weekly_schedule(schedule, calendar)
    return schedule.reset_index(drop=True)


def _validate_unique(frame: pd.DataFrame, key: Sequence[str], label: str) -> None:
    if frame[list(key)].isna().any(axis=None):
        raise FeatureAssemblyError(f"{label} contains a null key")
    if frame.duplicated(list(key)).any():
        raise FeatureAssemblyError(f"{label} contains duplicate key {tuple(key)}")


def _prepare_inputs(
    calendar: pd.DataFrame,
    security_master: pd.DataFrame,
    namechange: pd.DataFrame,
    raw_daily: pd.DataFrame,
    adj_factor: pd.DataFrame,
    daily_basic: pd.DataFrame,
    industry_membership: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.DatetimeIndex]:
    sessions = _normalize_calendar(calendar)
    _require_columns(
        security_master,
        ("ts_code", "name", "board", "list_date", "delist_date", "list_status"),
        "security_master",
    )
    _require_columns(namechange, ("ts_code", "name", "effective_from", "effective_to"), "namechange")
    _require_columns(raw_daily, ("ts_code", "trade_date", "close", "vol", "amount"), "raw_daily")
    _require_columns(adj_factor, ("ts_code", "trade_date", "adj_factor"), "adj_factor")
    _require_columns(daily_basic, ("ts_code", "trade_date", "free_share"), "daily_basic")
    _require_columns(
        industry_membership,
        ("ts_code", "industry_code", "classification_version", "effective_from", "effective_to"),
        "industry_membership",
    )

    security = security_master.copy()
    security["ts_code"] = security["ts_code"].astype(str)
    security["list_date"] = _normalize_dates(security["list_date"], "security_master.list_date")
    delist = pd.to_datetime(security["delist_date"], errors="coerce")
    security["delist_date"] = delist.dt.normalize().astype("datetime64[ns]")
    _validate_unique(security, ("ts_code",), "security_master")
    if not security["list_status"].astype(str).isin(VALID_LIST_STATUSES).all():
        raise FeatureAssemblyError("security_master list_status must be within L/D/P")

    raw = raw_daily.copy()
    raw["ts_code"] = raw["ts_code"].astype(str)
    raw["trade_date"] = _normalize_dates(raw["trade_date"], "raw_daily.trade_date")
    _validate_unique(raw, ("ts_code", "trade_date"), "raw_daily")
    if raw.empty:
        raise FeatureAssemblyError("raw_daily cannot be empty")
    if not raw["trade_date"].isin(sessions).all():
        raise FeatureAssemblyError("raw_daily contains a date outside the exchange calendar")
    for column in ("close", "vol", "amount"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce").astype(float)
    if not np.isfinite(raw[["close", "vol", "amount"]].to_numpy()).all():
        raise FeatureAssemblyError("raw_daily close/vol/amount must be finite")
    if (raw["close"] <= 0).any() or (raw[["vol", "amount"]] < 0).any(axis=None):
        raise FeatureAssemblyError("raw_daily has invalid close, volume, or amount")

    factor = adj_factor.copy()
    factor["ts_code"] = factor["ts_code"].astype(str)
    factor["trade_date"] = _normalize_dates(factor["trade_date"], "adj_factor.trade_date")
    factor["adj_factor"] = pd.to_numeric(factor["adj_factor"], errors="coerce").astype(float)
    _validate_unique(factor, ("ts_code", "trade_date"), "adj_factor")

    basic = daily_basic.copy()
    basic["ts_code"] = basic["ts_code"].astype(str)
    basic["trade_date"] = _normalize_dates(basic["trade_date"], "daily_basic.trade_date")
    basic["free_share"] = pd.to_numeric(basic["free_share"], errors="coerce").astype(float)
    _validate_unique(basic, ("ts_code", "trade_date"), "daily_basic")

    raw_key = pd.MultiIndex.from_frame(raw[["ts_code", "trade_date"]])
    for label, frame in (("adj_factor", factor), ("daily_basic", basic)):
        key = pd.MultiIndex.from_frame(frame[["ts_code", "trade_date"]])
        if len(raw_key.difference(key)) or len(key.difference(raw_key)):
            raise FeatureAssemblyError(f"raw_daily key must exactly equal {label} key")

    observed = raw.merge(
        factor[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        basic[["ts_code", "trade_date", "free_share"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    if (
        observed[["adj_factor", "free_share"]].isna().any(axis=None)
        or not np.isfinite(observed[["adj_factor", "free_share"]].to_numpy()).all()
        or (observed[["adj_factor", "free_share"]] <= 0).any(axis=None)
    ):
        raise FeatureAssemblyError("adj_factor and free_share must be complete, finite, and positive on raw rows")
    observed["observed_adjusted_close"] = observed["close"] * observed["adj_factor"]

    changes = namechange.copy()
    changes["ts_code"] = changes["ts_code"].astype(str)
    changes["effective_from"] = _normalize_dates(changes["effective_from"], "namechange.effective_from")
    changes["effective_to"] = pd.to_datetime(changes["effective_to"], errors="coerce").dt.normalize()
    _validate_unique(changes, ("ts_code", "effective_from"), "namechange")

    industry = industry_membership.copy()
    industry["ts_code"] = industry["ts_code"].astype(str)
    industry["effective_from"] = _normalize_dates(industry["effective_from"], "industry.effective_from")
    industry["effective_to"] = pd.to_datetime(industry["effective_to"], errors="coerce").dt.normalize()
    _validate_unique(industry, ("ts_code", "classification_version", "effective_from"), "industry_membership")
    if not industry["classification_version"].astype(str).eq("SW2021").all():
        raise FeatureAssemblyError("industry membership must use SW2021")

    covered_dates = sessions[(sessions >= observed["trade_date"].min()) & (sessions <= observed["trade_date"].max())]
    observed_dates = pd.Index(observed["trade_date"].unique())
    absent_market_dates = covered_dates.difference(observed_dates)
    if len(absent_market_dates):
        sample = [pd.Timestamp(value).date().isoformat() for value in absent_market_dates[:5]]
        raise FeatureAssemblyError(f"raw_daily has no market-wide row on open sessions: {sample}")

    return {
        "sessions": sessions,
        "security": security,
        "observed": observed,
        "namechange": changes,
        "industry": industry,
    }


def _interval_values(
    dates: pd.DatetimeIndex,
    intervals: pd.DataFrame,
    value_column: str,
    *,
    default: Any,
) -> pd.Series:
    values = pd.Series([default] * len(dates), index=dates, dtype="object")
    if intervals.empty:
        return values
    ordered = intervals.sort_values("effective_from", kind="mergesort")
    previous_end: pd.Timestamp | None = None
    for row in ordered.itertuples(index=False):
        start = pd.Timestamp(row.effective_from)
        end_value = row.effective_to
        end = pd.Timestamp.max.normalize() if pd.isna(end_value) else pd.Timestamp(end_value)
        if end < start or (previous_end is not None and start <= previous_end):
            raise FeatureAssemblyError("effective intervals overlap or have an invalid end")
        mask = (dates >= start) & (dates <= end)
        values.loc[mask] = getattr(row, value_column)
        previous_end = end
    return values


def _contains_st(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return "ST" in str(value).upper()


def _assemble_symbol(
    security_row: Any,
    sessions: pd.DatetimeIndex,
    decisions: pd.DatetimeIndex,
    symbol_observed: pd.DataFrame,
    symbol_changes: pd.DataFrame,
    symbol_industry: pd.DataFrame,
) -> pd.DataFrame:
    symbol = str(security_row.ts_code)
    dense = pd.DataFrame(index=sessions)
    dense.index.name = "dt"
    symbol_observed = symbol_observed.set_index("trade_date").sort_index()
    dense = dense.join(
        symbol_observed[["close", "vol", "amount", "free_share", "observed_adjusted_close"]],
        how="left",
    )
    observed_session = dense["close"].notna()
    valid_session = observed_session & dense["vol"].gt(0) & dense["amount"].gt(0)
    dense["adjusted_close"] = dense["observed_adjusted_close"].ffill()
    dense["amount_dense"] = dense["amount"].fillna(0.0)

    log_price = np.log(dense["adjusted_close"])
    dense["mom_120_20"] = log_price.shift(20) - log_price.shift(120)
    dense["lowvol_60"] = -log_price.diff().rolling(60, min_periods=60).std(ddof=1)
    dense["sma20"] = dense["adjusted_close"].rolling(20, min_periods=20).mean()
    dense["adv20"] = dense["amount_dense"].rolling(20, min_periods=20).mean()
    dense["valid_sessions_60"] = valid_session.astype(np.int8).rolling(60, min_periods=60).sum()

    list_date = pd.Timestamp(security_row.list_date)
    delist_date = pd.NaT if pd.isna(security_row.delist_date) else pd.Timestamp(security_row.delist_date)
    listing_start = int(sessions.searchsorted(list_date, side="left"))
    session_number = np.arange(len(sessions), dtype=np.int32) - listing_start + 1
    dense["market_sessions_since_listing"] = np.maximum(session_number, 0)
    member = (sessions >= list_date) & (str(security_row.list_status) in VALID_LIST_STATUSES)
    if not pd.isna(delist_date):
        member &= sessions < delist_date
    dense["pit_universe_member"] = member

    # ``security_master.name`` is a current snapshot.  Once effective-dated
    # name history exists it must be authoritative: using the current name in
    # an uncovered historical gap would backfill future ST knowledge.  A symbol
    # with no namechange history follows the contract's explicit "no change"
    # policy and may use its master name throughout.
    default_name = security_row.name if symbol_changes.empty else None
    active_names = _interval_values(
        sessions,
        symbol_changes,
        "name",
        default=default_name,
    )
    dense["is_st"] = active_names.map(_contains_st)

    active_industry = _interval_values(
        sessions,
        symbol_industry,
        "industry_code",
        default=None,
    )
    dense["industry_code"] = active_industry
    dense["board"] = None if pd.isna(security_row.board) else str(security_row.board)
    # ``daily_basic.free_share`` is point-in-time data and is deliberately not
    # carried through a suspension.  Missing PIT data therefore leaves market
    # cap missing and makes that symbol ineligible for the decision date.
    dense["free_float_mcap"] = dense["close"] * dense["free_share"] * 10_000.0
    dense["symbol"] = symbol

    selected = dense.loc[dense.index.isin(decisions)].reset_index()
    return selected[
        [
            "symbol",
            "dt",
            "pit_universe_member",
            "market_sessions_since_listing",
            "valid_sessions_60",
            "is_st",
            "board",
            "adv20",
            "mom_120_20",
            "lowvol_60",
            "industry_code",
            "free_float_mcap",
            "adjusted_close",
            "sma20",
        ]
    ]


def assemble_feature_frames(
    *,
    calendar: pd.DataFrame,
    security_master: pd.DataFrame,
    namechange: pd.DataFrame,
    raw_daily: pd.DataFrame,
    adj_factor: pd.DataFrame,
    daily_basic: pd.DataFrame,
    industry_membership: pd.DataFrame,
    eligibility_spec: research_v2.EligibilitySpec,
    target_size: int,
) -> FeatureFrames:
    """Pure, prefix-invariant assembler used by the formal bundle entry point."""

    if target_size <= 0:
        raise ValueError("target_size must be positive")

    prepared = _prepare_inputs(
        calendar,
        security_master,
        namechange,
        raw_daily,
        adj_factor,
        daily_basic,
        industry_membership,
    )
    sessions = prepared["sessions"]
    security = prepared["security"]
    observed = prepared["observed"]
    changes = prepared["namechange"]
    industry = prepared["industry"]
    assert isinstance(sessions, pd.DatetimeIndex)
    assert isinstance(security, pd.DataFrame)
    assert isinstance(observed, pd.DataFrame)
    assert isinstance(changes, pd.DataFrame)
    assert isinstance(industry, pd.DataFrame)

    schedule = build_weekly_schedule(
        calendar,
        observed["trade_date"].max(),
        observed["trade_date"].min(),
    )
    decisions = pd.DatetimeIndex(schedule["decision_dt"])
    empty_observed = observed.iloc[:0].copy()
    empty_changes = changes.iloc[:0].copy()
    empty_industry = industry.iloc[:0].copy()
    observed_groups = {str(symbol): group.copy() for symbol, group in observed.groupby("ts_code", sort=False)}
    change_groups = {str(symbol): group.copy() for symbol, group in changes.groupby("ts_code", sort=False)}
    industry_groups = {str(symbol): group.copy() for symbol, group in industry.groupby("ts_code", sort=False)}
    symbol_frames = [
        _assemble_symbol(
            row,
            sessions,
            decisions,
            observed_groups.get(str(row.ts_code), empty_observed),
            change_groups.get(str(row.ts_code), empty_changes),
            industry_groups.get(str(row.ts_code), empty_industry),
        )
        for row in security.sort_values("ts_code", kind="mergesort").itertuples(index=False)
    ]
    if not symbol_frames:
        raise FeatureAssemblyError("security_master cannot be empty")
    features = pd.concat(symbol_frames, ignore_index=True)
    # The shared eligibility engine intentionally accepts only complete boolean
    # PIT flags.  Preserve an uncovered name-history interval as auditable
    # ``<NA>`` in the feature surface, while evaluating it conservatively as ST
    # so unknown status can never enter the eligible pool.
    effective_is_st = features["is_st"].astype("boolean")
    eligibility_input = features.copy()
    eligibility_input["is_st"] = effective_is_st.fillna(True).astype(bool)
    features = research_v2.derive_eligibility(eligibility_input, eligibility_spec)
    features["is_st"] = effective_is_st
    features = features.loc[:, list(FEATURE_COLUMNS)].sort_values(["dt", "symbol"], kind="mergesort")
    features = features.reset_index(drop=True)
    decision_dates = pd.DatetimeIndex(schedule["decision_dt"])
    eligible_counts = (
        features.groupby("dt", sort=True, observed=True)["eligible"]
        .sum()
        .reindex(decision_dates, fill_value=0)
        .astype(int)
    )
    sufficient = eligible_counts.ge(target_size)
    if not sufficient.any():
        raise FeatureAssemblyError(f"no weekly decision has an eligible pool of at least target_size={target_size}")
    first_position = int(np.flatnonzero(sufficient.to_numpy())[0])
    later_counts = eligible_counts.iloc[first_position:]
    if (later_counts < target_size).any():
        bad = later_counts[later_counts < target_size]
        sample = {pd.Timestamp(dt).date().isoformat(): int(count) for dt, count in bad.iloc[:5].items()}
        raise FeatureAssemblyError(f"eligible pool fell below target_size={target_size} after formal start: {sample}")
    schedule = schedule.iloc[first_position:].reset_index(drop=True)
    selected_dates = set(pd.DatetimeIndex(schedule["decision_dt"]))
    features = features[features["dt"].isin(selected_dates)].reset_index(drop=True)
    all_adv = features.loc[:, list(ALL_ADV_COLUMNS)].copy()
    if features.duplicated(["symbol", "dt"]).any() or all_adv.duplicated(["symbol", "dt"]).any():
        raise AssertionError("assembled decision rows are not unique")
    result = FeatureFrames(schedule=schedule, features=features, all_adv=all_adv, target_size=target_size)
    _validate_output_alignment(result)
    return result


def _validate_output_alignment(frames: FeatureFrames) -> dict[str, Any]:
    """Require exact decision-date alignment and return frozen coverage evidence."""

    schedule_dates = pd.DatetimeIndex(pd.to_datetime(frames.schedule["decision_dt"]))
    feature_dates = pd.DatetimeIndex(pd.to_datetime(frames.features["dt"].unique())).sort_values()
    adv_dates = pd.DatetimeIndex(pd.to_datetime(frames.all_adv["dt"].unique())).sort_values()
    if schedule_dates.empty or schedule_dates.has_duplicates or not schedule_dates.is_monotonic_increasing:
        raise FeatureAssemblyError("formal schedule must contain unique increasing decision dates")
    if not schedule_dates.equals(feature_dates) or not schedule_dates.equals(adv_dates):
        raise FeatureAssemblyError("schedule, decision features, and all_adv must have exactly the same dates")
    eligible_counts = frames.features.groupby("dt", sort=True, observed=True)["eligible"].sum().reindex(schedule_dates)
    if eligible_counts.isna().any():
        raise FeatureAssemblyError("published features are missing one or more formal decision dates")
    eligible_counts = eligible_counts.astype(int)
    if (eligible_counts < frames.target_size).any():
        raise FeatureAssemblyError("published eligible counts violate the frozen target_size")
    return {
        "first_decision_dt": schedule_dates[0].date().isoformat(),
        "last_decision_dt": schedule_dates[-1].date().isoformat(),
        "decision_count": len(schedule_dates),
        "eligible_count_min": int(eligible_counts.min()),
        "target_size": int(frames.target_size),
    }


def _strict_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeatureAssemblyError(f"cannot read validated manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FeatureAssemblyError("validated manifest root must be an object")
    return value


def _bundle_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FeatureAssemblyError(f"invalid manifest artifact path: {relative!r}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise FeatureAssemblyError(f"unsafe manifest artifact path: {relative!r}")
    return path


def _load_validated_bundle(
    bundle_root: str | Path,
    manifest_path: str | Path | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any], research_v2.V2Config]:
    root = Path(bundle_root).expanduser().resolve(strict=True)
    report = data_v2.validate_bundle(root, manifest_path)
    if not report.forward_start_allowed:
        failed = sorted(
            name for name, passed in report.gates.items() if not passed and name != "forward_chain_complete"
        )
        raise FeatureAssemblyError(f"bundle is not forward-start ready; failed_gates={failed}")

    manifest_file = Path(manifest_path) if manifest_path is not None else root / data_v2.DEFAULT_MANIFEST
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    manifest_file = manifest_file.resolve(strict=True)
    manifest = _strict_manifest(manifest_file)
    recorded_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(recorded_manifest_hash, str) or recorded_manifest_hash != data_v2.compute_manifest_sha256(
        manifest
    ):
        raise FeatureAssemblyError("bundle manifest mutated after validation")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise FeatureAssemblyError("validated manifest datasets entry is missing")

    frames: dict[str, pd.DataFrame] = {}
    input_hashes: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name in INPUT_DATASETS:
        entry = datasets.get(name)
        if not isinstance(entry, Mapping):
            raise FeatureAssemblyError(f"validated manifest is missing dataset {name}")
        path = _bundle_path(root, entry.get("path"))
        expected = str(entry.get("sha256"))
        if _sha256_file(path) != expected:
            raise FeatureAssemblyError(f"dataset mutated after validation: {name}")
        frames[name] = pd.read_parquet(path)
        paths[name] = path
        input_hashes[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": expected,
            "rows": int(entry.get("rows", -1)),
            "size_bytes": int(entry.get("size_bytes", -1)),
        }
    for name, path in paths.items():
        if _sha256_file(path) != input_hashes[name]["sha256"]:
            raise FeatureAssemblyError(f"dataset mutated while assembling features: {name}")

    protocol_entry = manifest.get("protocol")
    if not isinstance(protocol_entry, Mapping):
        raise FeatureAssemblyError("validated manifest protocol entry is missing")
    protocol_path = _bundle_path(root, protocol_entry.get("path"))
    if _sha256_file(protocol_path) != protocol_entry.get("sha256"):
        raise FeatureAssemblyError("protocol mutated after bundle validation")
    config, _ = research_v2.load_protocol(protocol_path)
    if _sha256_file(protocol_path) != protocol_entry.get("sha256"):
        raise FeatureAssemblyError("protocol mutated while assembling features")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "data_evidence_sha256": manifest.get("data_evidence_sha256"),
        "bundle_manifest_sha256": manifest.get("manifest_sha256"),
        "protocol_sha256": protocol_entry.get("sha256"),
        "feature_engine_sha256": _sha256_file(Path(__file__).resolve()),
        "research_dependency_sha256": _sha256_file(Path(research_v2.__file__).resolve()),
        "inputs": input_hashes,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "clock": "dense_exchange_sessions_suspension_carry_adjusted_close_amount_zero",
        "pit_fill_policy": "industry_name_and_free_share_never_filled_missing_means_ineligible",
        "windows": {"momentum": [120, 20], "lowvol": 60, "sma": 20, "adv": 20, "valid": 60},
    }
    return frames, manifest, identity, config


def _parquet_evidence(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": int(parquet.metadata.num_rows),
        "schema": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
    }


def publish_feature_frames(
    frames: FeatureFrames,
    identity: Mapping[str, Any],
    output_root: str | Path,
) -> FeatureBuildResult:
    """Publish one exclusive content-addressed directory; existing identities are immutable."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    coverage = _validate_output_alignment(frames)
    identity_payload = dict(identity)
    if "feature_coverage" in identity_payload:
        raise ValueError("identity key 'feature_coverage' is reserved by the feature publisher")
    identity_payload["feature_coverage"] = coverage
    run_id = _sha256_bytes(_canonical_json(identity_payload))
    final_dir = root / f"{OUTPUT_PREFIX}{run_id}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite content-addressed feature output: {final_dir}")

    temporary = Path(tempfile.mkdtemp(prefix=f".tmp_{run_id}_", dir=root))
    try:
        schedule_path = temporary / "weekly_schedule.parquet"
        features_path = temporary / "decision_features.parquet"
        all_adv_path = temporary / "all_decision_adv.parquet"
        frames.schedule.loc[:, list(SCHEDULE_COLUMNS)].to_parquet(schedule_path, index=False)
        frames.features.loc[:, list(FEATURE_COLUMNS)].to_parquet(features_path, index=False)
        frames.all_adv.loc[:, list(ALL_ADV_COLUMNS)].to_parquet(all_adv_path, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "identity": identity_payload,
            "coverage": coverage,
            "outputs": {
                "schedule": _parquet_evidence(schedule_path),
                "features": _parquet_evidence(features_path),
                "all_adv": _parquet_evidence(all_adv_path),
            },
        }
        manifest["manifest_sha256"] = _sha256_bytes(_canonical_json(manifest))
        manifest_path = temporary / "feature_manifest.json"
        with manifest_path.open("xb") as file:
            file.write(json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode())
            file.write(b"\n")
        if final_dir.exists():
            raise FileExistsError(f"feature output appeared during build: {final_dir}")
        os.rename(temporary, final_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return FeatureBuildResult(
        run_id=run_id,
        output_dir=final_dir,
        schedule_path=final_dir / "weekly_schedule.parquet",
        features_path=final_dir / "decision_features.parquet",
        all_adv_path=final_dir / "all_decision_adv.parquet",
        manifest_path=final_dir / "feature_manifest.json",
    )


def build_feature_artifacts(
    bundle_root: str | Path,
    output_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> FeatureBuildResult:
    """Validate one bundle, assemble all formal inputs together, and publish exclusively."""

    frames, _, identity, config = _load_validated_bundle(bundle_root, manifest_path)
    outputs = assemble_feature_frames(
        calendar=frames["calendar"],
        security_master=frames["security_master"],
        namechange=frames["namechange"],
        raw_daily=frames["raw_daily"],
        adj_factor=frames["adj_factor"],
        daily_basic=frames["daily_basic"],
        industry_membership=frames["industry_membership"],
        eligibility_spec=config.eligibility,
        target_size=config.selection.target_size,
    )
    return publish_feature_frames(outputs, identity, output_root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = build_feature_artifacts(args.bundle_root, args.output_root, manifest_path=args.manifest)
    except (FeatureAssemblyError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"ok": True, "run_id": result.run_id, "output_dir": str(result.output_dir)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
