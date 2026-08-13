"""Run the independent Stage 1 exploratory study for the XS-Chan strategy.

This module deliberately does not import or mutate the V2.1 ledger, execution
engine, planner, verifier, protocol, or forward artifacts.  It consumes the
existing contaminated historical pressure cache and writes a separate
exploratory evidence bundle.

Usage:

    uv run --no-sync python scripts/xs_chan_exploration_stage1.py fetch-reference-data
    uv run --no-sync python scripts/xs_chan_exploration_stage1.py run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from xs_chan_research_v2 import _newey_west_t_weekly, rank_cross_section_v2

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SPEC_PATH = SCRIPTS_DIR / "xs_chan_exploration_stage1.json"
V21_PROTOCOL_PATH = SCRIPTS_DIR / "xs_chan_protocol_v2_1.json"
V21_DOCUMENT_PATH = SCRIPTS_DIR / "XS_CHAN_PILOT_PROTOCOL_V2_1.md"
OUTPUT_DIR = SCRIPTS_DIR / "_output" / "xs_chan_exploration_stage1"
REFERENCE_DIR = OUTPUT_DIR / "reference_data"
DAILY_BASIC_DIR = REFERENCE_DIR / "daily_basic_weekly"
RAW_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"
STATE_CACHE_ROOT = Path.home() / ".ts_data_cache" / "xs_chan_state_cache_v2"

STATE_NAMES = {
    0: "NotTradable",
    1: "Downtrend",
    2: "FirstBuy",
    3: "SecondBuy",
    4: "PivotBuilding",
    5: "UpwardDeparture",
    6: "ThirdBuy",
    7: "MainUptrend",
    8: "Acceleration",
    9: "Divergence",
    10: "Breakdown",
}
STATE_NAMES_ZH = {
    0: "不可交易",
    1: "下跌走势",
    2: "一买观察",
    3: "二买转强",
    4: "中枢构造",
    5: "向上离开中枢",
    6: "三买确认",
    7: "主升延续",
    8: "加速主升",
    9: "背驰衰竭",
    10: "结构破坏",
}


class ExplorationError(RuntimeError):
    """Raised when the exploratory evidence bundle cannot be built safely."""


@dataclass(frozen=True)
class StudyPaths:
    """Resolved input and output paths for a study run."""

    output_dir: Path
    reference_dir: Path
    daily_basic_dir: Path
    state_path: Path
    state_manifest_path: Path
    industry_path: Path


@dataclass
class ExperimentRecord:
    """One append-only-in-memory experiment status record."""

    experiment_id: str
    deliverable: str
    status: str = "PLANNED"
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_seconds: float | None = None
    output_rows: dict[str, int] | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_tail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "deliverable": self.deliverable,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "duration_seconds": self.duration_seconds,
            "output_rows": self.output_rows or {},
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_tail": self.traceback_tail,
        }


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp with microseconds."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(payload: Any) -> bytes:
    """Return the canonical JSON representation used by the V2.1 protocol."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write strict, stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a strict JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: _reject_constant(token))
    if not isinstance(payload, dict):
        raise ExplorationError(f"{path} must contain a JSON object")
    return payload


def _reject_constant(token: str) -> Any:
    raise ExplorationError(f"invalid JSON constant: {token}")


def load_and_validate_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    """Load the study spec and prove that the frozen V2.1 baseline is unchanged."""

    spec = read_json(path)
    if spec.get("mode") != "EXPLORATORY_ONLY" or spec.get("confirmation_chain") != "NOT_STARTED":
        raise ExplorationError("study must remain EXPLORATORY_ONLY with no confirmation chain")
    no_tuning = spec.get("no_tuning")
    if not isinstance(no_tuning, Mapping) or any(
        no_tuning.get(key) is not False
        for key in (
            "parameter_search",
            "threshold_optimization",
            "best_variant_selection",
            "failed_experiment_suppression",
        )
    ):
        raise ExplorationError("the study spec must prohibit tuning and result suppression")
    planned = spec.get("planned_experiments")
    if not isinstance(planned, list) or len(planned) != 8:
        raise ExplorationError("the complete eight-experiment registry must be frozen before running")
    experiment_ids = [row.get("id") for row in planned if isinstance(row, Mapping)]
    if len(experiment_ids) != len(set(experiment_ids)) or any(not value for value in experiment_ids):
        raise ExplorationError("planned experiment IDs must be unique non-empty strings")

    baseline = spec.get("frozen_baseline")
    if not isinstance(baseline, Mapping):
        raise ExplorationError("frozen_baseline is missing")
    if sha256_file(V21_PROTOCOL_PATH) != baseline.get("protocol_physical_sha256"):
        raise ExplorationError("V2.1 protocol physical bytes differ from the frozen exploratory baseline")
    protocol = read_json(V21_PROTOCOL_PATH)
    canonical_digest = hashlib.sha256(canonical_json(protocol)).hexdigest()
    if canonical_digest != baseline.get("protocol_canonical_sha256"):
        raise ExplorationError("V2.1 canonical protocol digest differs from the frozen baseline")
    if protocol.get("protocol_id") != baseline.get("protocol_id"):
        raise ExplorationError("V2.1 protocol ID differs from the frozen baseline")
    if protocol.get("protocol_status") != baseline.get("required_protocol_status"):
        raise ExplorationError("V2.1 protocol status changed; exploration refuses to reinterpret it")
    if sha256_file(V21_DOCUMENT_PATH) != baseline.get("protocol_document_sha256"):
        raise ExplorationError("V2.1 protocol document differs from the frozen baseline")
    return spec


def discover_state_cache() -> tuple[Path, Path]:
    """Resolve the single published V2 state projection and its manifest."""

    manifests = sorted(STATE_CACHE_ROOT.glob("CHAN_STATE_CACHE_*/source_manifest.json"))
    if len(manifests) != 1:
        raise ExplorationError(f"expected one published state cache, found {len(manifests)}")
    manifest_path = manifests[0]
    state_path = manifest_path.parent / "states.parquet"
    if not state_path.exists():
        raise ExplorationError(f"missing state projection: {state_path}")
    return state_path, manifest_path


def resolve_paths(output_dir: Path = OUTPUT_DIR) -> StudyPaths:
    """Resolve all study paths without creating confirmation artifacts."""

    state_path, state_manifest_path = discover_state_cache()
    reference_dir = output_dir / "reference_data"
    return StudyPaths(
        output_dir=output_dir,
        reference_dir=reference_dir,
        daily_basic_dir=reference_dir / "daily_basic_weekly",
        state_path=state_path,
        state_manifest_path=state_manifest_path,
        industry_path=reference_dir / "sw2021_l1_membership.parquet",
    )


def load_market_calendar(end_date: str | pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Build the observed global market-session calendar from all cached bars."""

    if not RAW_DIR.exists():
        raise ExplorationError(f"missing raw pressure cache: {RAW_DIR}")
    scan = pl.scan_parquet(str(RAW_DIR / "*.parquet")).select(
        pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=True).alias("dt")
    )
    if end_date is not None:
        scan = scan.filter(pl.col("dt") <= pd.Timestamp(end_date).date())
    dates = scan.unique().sort("dt").collect(engine="streaming")["dt"].to_list()
    calendar = pd.DatetimeIndex(pd.to_datetime(dates))
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ExplorationError("observed market calendar must be non-empty, unique and increasing")
    return calendar


def build_decision_schedule(spec: Mapping[str, Any], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Create the fixed weekly decision/entry/horizon schedule."""

    sample = spec["sample"]
    start = pd.Timestamp(sample["start_date"])
    end = min(pd.Timestamp(sample["end_date"]), calendar.max())
    table = pd.DataFrame({"session_dt": calendar, "session_no": np.arange(len(calendar), dtype=np.int32)})
    table["week"] = table["session_dt"].dt.to_period("W-FRI")
    decisions = (
        table.loc[table["session_dt"].between(start, end)]
        .groupby("week", sort=True, observed=True)["session_no"]
        .max()
        .astype(int)
        .to_numpy()
    )
    max_offset = 1 + max(
        int(sample["diagnostic_holding_sessions"]),
        max(map(int, sample["signal_delays_sessions"])) + int(sample["primary_holding_sessions"]),
    )
    decisions = decisions[decisions + max_offset < len(calendar)]
    rows: list[dict[str, Any]] = []
    for session_no in decisions:
        row = {
            "decision_dt": calendar[session_no],
            "decision_session_no": int(session_no),
            "entry_dt": calendar[session_no + 1],
            "exit_5d_dt": calendar[session_no + 1 + int(sample["primary_holding_sessions"])],
            "exit_20d_dt": calendar[session_no + 1 + int(sample["diagnostic_holding_sessions"])],
        }
        for delay in map(int, sample["signal_delays_sessions"]):
            row[f"delay_{delay}_entry_dt"] = calendar[session_no + 1 + delay]
            row[f"delay_{delay}_exit_dt"] = calendar[session_no + 1 + delay + int(sample["primary_holding_sessions"])]
        rows.append(row)
    schedule = pd.DataFrame(rows)
    if len(schedule) < int(sample["minimum_complete_decision_weeks"]):
        raise ExplorationError(
            f"only {len(schedule)} complete decision weeks; need {sample['minimum_complete_decision_weeks']}"
        )
    return schedule


def resolve_tinyshare_token() -> str:
    """Resolve a token without ever writing it into exploratory artifacts."""

    token = (os.getenv("TINYSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN") or "").strip()
    if not token:
        raise ExplorationError("TINYSHARE_TOKEN or TUSHARE_TOKEN must be set and non-empty")
    return token


def _api_call_with_retry(call: Callable[[], pd.DataFrame], label: str, retries: int = 4) -> pd.DataFrame:
    """Run a read-only data API call with bounded retry."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            result = call()
            if not isinstance(result, pd.DataFrame):
                raise ExplorationError(f"{label} returned a non-DataFrame response")
            return result
        except Exception as exc:  # pragma: no cover - network failures vary by environment
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise ExplorationError(f"{label} failed after {retries} attempts: {last_error}") from last_error


def fetch_reference_data(
    spec: Mapping[str, Any],
    paths: StudyPaths,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Fetch/cache weekly daily-basic and effective-dated SW2021 L1 membership."""

    import tinyshare as ts

    paths.reference_dir.mkdir(parents=True, exist_ok=True)
    paths.daily_basic_dir.mkdir(parents=True, exist_ok=True)
    calendar = load_market_calendar(spec["sample"]["end_date"])
    schedule = build_decision_schedule(spec, calendar)
    token = resolve_tinyshare_token()
    ts.set_token(token)
    pro = ts.pro_api()

    daily_fields = "ts_code,trade_date,close,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv,free_share"
    fetched_dates = 0
    reused_dates = 0
    row_count = 0
    for index, decision_dt in enumerate(schedule["decision_dt"], 1):
        date_text = pd.Timestamp(decision_dt).strftime("%Y%m%d")
        path = paths.daily_basic_dir / f"{date_text}.parquet"
        if path.exists() and not refresh:
            frame = pd.read_parquet(path)
            reused_dates += 1
        else:
            frame = _api_call_with_retry(
                lambda date_text=date_text: pro.daily_basic(trade_date=date_text, fields=daily_fields),
                f"daily_basic[{date_text}]",
            )
            required = set(daily_fields.split(","))
            if missing := required - set(frame):
                raise ExplorationError(f"daily_basic[{date_text}] missing fields: {sorted(missing)}")
            frame = frame[list(daily_fields.split(","))].copy()
            frame.to_parquet(path, index=False)
            fetched_dates += 1
        row_count += len(frame)
        if index % 25 == 0 or index == len(schedule):
            print(
                f"[reference] daily_basic {index}/{len(schedule)} fetched={fetched_dates} reused={reused_dates}",
                flush=True,
            )

    if not paths.industry_path.exists() or refresh:
        classifications = _api_call_with_retry(
            lambda: pro.index_classify(level="L1", src="SW2021"),
            "index_classify[SW2021/L1]",
        )
        required_class = {"index_code", "industry_name", "industry_code"}
        if missing := required_class - set(classifications):
            raise ExplorationError(f"SW2021 classification missing fields: {sorted(missing)}")
        member_parts: list[pd.DataFrame] = []
        for row in classifications.sort_values("index_code").itertuples(index=False):
            members = _api_call_with_retry(
                lambda code=str(row.index_code): pro.index_member(index_code=code),
                f"index_member[{row.index_code}]",
            )
            required_member = {"con_code", "in_date", "out_date"}
            if missing := required_member - set(members):
                raise ExplorationError(f"index_member[{row.index_code}] missing fields: {sorted(missing)}")
            part = members[list(required_member)].copy()
            part["index_code"] = str(row.index_code)
            part["industry_code"] = str(row.industry_code)
            part["industry_name"] = str(row.industry_name)
            member_parts.append(part)
        industry = pd.concat(member_parts, ignore_index=True)
        industry = industry.rename(
            columns={"con_code": "symbol", "in_date": "effective_from", "out_date": "effective_to"}
        )
        industry = industry[
            ["symbol", "industry_code", "industry_name", "index_code", "effective_from", "effective_to"]
        ]
        industry = industry.drop_duplicates(
            ["symbol", "industry_code", "effective_from", "effective_to"], keep="last"
        ).sort_values(["symbol", "effective_from", "industry_code"], kind="mergesort")
        industry.to_parquet(paths.industry_path, index=False)
    industry = pd.read_parquet(paths.industry_path)

    manifest = {
        "study_id": spec["study_id"],
        "created_at_utc": utc_now(),
        "credential_recorded": False,
        "daily_basic": {
            "decision_dates": int(len(schedule)),
            "rows": int(row_count),
            "fetched_dates": int(fetched_dates),
            "reused_dates": int(reused_dates),
            "fields": daily_fields.split(","),
            "file_count": int(len(list(paths.daily_basic_dir.glob("*.parquet")))),
        },
        "industry_membership": {
            "rows": int(len(industry)),
            "symbols": int(industry["symbol"].nunique()),
            "industries": int(industry["industry_code"].nunique()),
            "path": str(paths.industry_path),
            "sha256": sha256_file(paths.industry_path),
        },
    }
    write_json(paths.reference_dir / "reference_manifest.json", manifest)
    return manifest


def load_daily_basic(paths: StudyPaths, schedule: pd.DataFrame) -> pd.DataFrame:
    """Load the exact weekly daily-basic cache required by the frozen schedule."""

    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for decision_dt in schedule["decision_dt"]:
        date_text = pd.Timestamp(decision_dt).strftime("%Y%m%d")
        path = paths.daily_basic_dir / f"{date_text}.parquet"
        if not path.exists():
            missing.append(date_text)
            continue
        parts.append(pd.read_parquet(path))
    if missing:
        preview = ", ".join(missing[:5])
        raise ExplorationError(
            f"missing {len(missing)} weekly daily-basic files ({preview}); run fetch-reference-data first"
        )
    frame = pd.concat(parts, ignore_index=True)
    required = {
        "ts_code",
        "trade_date",
        "close",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "total_mv",
        "circ_mv",
        "free_share",
    }
    if missing_columns := required - set(frame):
        raise ExplorationError(f"daily-basic cache missing fields: {sorted(missing_columns)}")
    frame = frame[sorted(required)].copy()
    frame["symbol"] = frame.pop("ts_code").astype(str)
    frame["dt"] = pd.to_datetime(frame.pop("trade_date"), format="%Y%m%d", errors="raise")
    if frame.duplicated(["symbol", "dt"]).any():
        raise ExplorationError("weekly daily-basic cache has duplicate symbol/date rows")
    for column in required - {"ts_code", "trade_date"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["free_float_mcap"] = frame["close"] * frame["free_share"] * 10_000.0
    frame["log_free_float_mcap"] = np.log(frame["free_float_mcap"].where(frame["free_float_mcap"] > 0))
    return frame.drop(columns=["close"])


def load_industry_membership(paths: StudyPaths) -> pd.DataFrame:
    """Load and normalize effective-dated SW2021 L1 membership."""

    if not paths.industry_path.exists():
        raise ExplorationError("missing SW2021 membership; run fetch-reference-data first")
    frame = pd.read_parquet(paths.industry_path)
    required = {"symbol", "industry_code", "industry_name", "effective_from", "effective_to"}
    if missing := required - set(frame):
        raise ExplorationError(f"industry membership missing fields: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["industry_code"] = frame["industry_code"].astype(str)
    frame["industry_name"] = frame["industry_name"].astype(str)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], format="%Y%m%d", errors="raise")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], format="%Y%m%d", errors="coerce")
    frame["effective_to_sort"] = frame["effective_to"].fillna(pd.Timestamp.max.normalize())
    frame = frame.sort_values(
        ["symbol", "effective_from", "effective_to_sort", "industry_code"],
        kind="mergesort",
    )
    duplicated = frame.duplicated(["symbol", "effective_from"], keep=False)
    if duplicated.any():
        open_counts = (
            frame.loc[duplicated & frame["effective_to"].isna()]
            .groupby(["symbol", "effective_from"], observed=True)
            .size()
        )
        if (open_counts > 1).any():
            raise ExplorationError("SW2021 membership has multiple open industries from the same effective date")
        # SW2021 transition rows can reuse the original listing date while the
        # predecessor classification ends on 2021-12-10.  The Stage 1 sample
        # begins in 2022, so the interval with the latest end is the only one
        # that can be active in-sample.
        frame = frame.drop_duplicates(["symbol", "effective_from"], keep="last")
    return frame.drop(columns="effective_to_sort").reset_index(drop=True)


def load_namechange() -> pd.DataFrame:
    """Load historical name intervals used only for the exploratory ST exclusion."""

    if not NAMECHANGE_PATH.exists():
        raise ExplorationError(f"missing name-change pressure cache: {NAMECHANGE_PATH}")
    frame = pd.read_parquet(NAMECHANGE_PATH)
    required = {"ts_code", "name", "start_date", "end_date"}
    if missing := required - set(frame):
        raise ExplorationError(f"name-change cache missing fields: {sorted(missing)}")
    frame = frame[list(required)].rename(
        columns={"ts_code": "symbol", "start_date": "effective_from", "end_date": "effective_to"}
    )
    frame["symbol"] = frame["symbol"].astype(str)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], format="%Y%m%d", errors="raise")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], format="%Y%m%d", errors="coerce")
    frame["active_st_name"] = frame["name"].str.upper().str.contains("ST", regex=False) | frame["name"].str.contains(
        "退", regex=False
    )
    frame = frame.sort_values(["symbol", "effective_from"], kind="mergesort")
    frame = frame.drop_duplicates(["symbol", "effective_from"], keep="last")
    return frame.reset_index(drop=True)


def infer_board(symbol: str) -> str:
    """Infer the protocol board category from an A-share symbol."""

    code, _, exchange = symbol.partition(".")
    if exchange == "BJ" or code.startswith(("4", "8", "92")):
        return "BSE"
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301", "302")):
        return "CHINEXT"
    return "MAIN"


def build_dense_feature_panel(
    spec: Mapping[str, Any],
    calendar: pd.DatetimeIndex,
) -> pl.DataFrame:
    """Build exchange-session-clock features from the contaminated qfq pressure cache."""

    sample_end = pd.Timestamp(spec["sample"]["end_date"]).date()
    raw = (
        pl.scan_parquet(str(RAW_DIR / "*.parquet"))
        .select(
            pl.col("ts_code").cast(pl.String).alias("symbol"),
            pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=True).alias("dt"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
            pl.col("pct_chg").cast(pl.Float64),
        )
        .filter(pl.col("dt") <= sample_end)
        .collect(engine="streaming")
        .sort(["symbol", "dt"])
    )
    if raw.is_empty() or raw.select(pl.struct(["symbol", "dt"]).is_duplicated().any()).item():
        raise ExplorationError("qfq pressure cache must be non-empty and unique by symbol/date")
    bounds = raw.group_by("symbol").agg(
        pl.col("dt").min().alias("first_observed_dt"),
        pl.col("dt").max().alias("last_observed_dt"),
    )
    calendar_frame = pl.from_pandas(pd.DataFrame({"dt": calendar[calendar <= pd.Timestamp(sample_end)]})).with_columns(
        pl.col("dt").cast(pl.Date)
    )
    dense = (
        bounds.join(calendar_frame, how="cross")
        .filter(pl.col("dt").ge(pl.col("first_observed_dt")) & pl.col("dt").le(pl.col("last_observed_dt")))
        .join(raw, on=["symbol", "dt"], how="left")
        .sort(["symbol", "dt"])
        .with_columns(
            pl.col("close").is_not_null().alias("observed_bar"),
            (
                pl.col("close").is_not_null()
                & pl.col("amount").fill_null(0.0).gt(0)
                & pl.col("vol").fill_null(0.0).gt(0)
            ).alias("valid_trade"),
            pl.col("close").forward_fill().over("symbol").alias("adjusted_close"),
            pl.col("amount").fill_null(0.0).alias("amount_filled"),
            pl.col("vol").fill_null(0.0).alias("vol_filled"),
            pl.col("dt").cum_count().over("symbol").cast(pl.Int32).alias("market_sessions_since_first_observation"),
        )
        .with_columns(pl.col("adjusted_close").log().alias("log_adjusted_close"))
        .with_columns(
            (
                pl.col("log_adjusted_close").shift(20).over("symbol")
                - pl.col("log_adjusted_close").shift(120).over("symbol")
            ).alias("mom_120_20"),
            pl.col("log_adjusted_close").diff().over("symbol").alias("log_return"),
            pl.col("adjusted_close").rolling_mean(window_size=20, min_samples=20).over("symbol").alias("sma20"),
            pl.col("amount_filled").rolling_mean(window_size=20, min_samples=20).over("symbol").alias("adv20"),
            pl.col("valid_trade")
            .cast(pl.Int16)
            .rolling_sum(window_size=60, min_samples=60)
            .over("symbol")
            .alias("valid_sessions_60"),
        )
        .with_columns(
            (-pl.col("log_return").rolling_std(window_size=60, min_samples=60, ddof=1).over("symbol")).alias(
                "lowvol_60"
            ),
            (pl.col("adjusted_close") / pl.col("sma20") - 1.0).alias("distance_to_sma20"),
            (pl.col("adjusted_close") > pl.col("sma20")).fill_null(False).alias("ma_allowed"),
        )
    )
    if dense["adjusted_close"].null_count():
        raise ExplorationError("dense feature panel could not carry the first observed price")
    return dense


def _join_exact_observation(
    surface: pl.DataFrame,
    dense: pl.DataFrame,
    *,
    date_column: str,
    prefix: str,
) -> pl.DataFrame:
    """Join an exact market-session observation to the decision surface."""

    source = dense.select(
        "symbol",
        pl.col("dt").alias(date_column),
        pl.col("open").alias(f"{prefix}_open"),
        pl.col("high").alias(f"{prefix}_high"),
        pl.col("low").alias(f"{prefix}_low"),
        pl.col("adjusted_close").alias(f"{prefix}_close"),
        pl.col("amount_filled").alias(f"{prefix}_amount"),
        pl.col("observed_bar").alias(f"{prefix}_observed"),
    )
    return surface.join(source, on=["symbol", date_column], how="left")


def _join_asof_interval(
    surface: pl.DataFrame,
    intervals: pd.DataFrame,
    *,
    columns: Sequence[str],
    prefix: str,
) -> pl.DataFrame:
    """Attach the latest interval starting on/before each decision date."""

    right = pl.from_pandas(intervals[["symbol", "effective_from", "effective_to", *columns]]).with_columns(
        pl.col("effective_from").cast(pl.Date),
        pl.col("effective_to").cast(pl.Date),
    )
    left = surface.sort(["symbol", "dt"])
    right = right.sort(["symbol", "effective_from"])
    joined = left.join_asof(
        right,
        left_on="dt",
        right_on="effective_from",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    active = pl.col("effective_from").is_not_null() & (
        pl.col("effective_to").is_null() | pl.col("dt").le(pl.col("effective_to"))
    )
    expressions = [
        pl.when(active).then(pl.col(column)).otherwise(None).alias(f"{prefix}{column}") for column in columns
    ]
    drop_columns = ["effective_from", "effective_to"]
    if prefix:
        drop_columns.extend(columns)
    return joined.with_columns(*expressions).drop(drop_columns)


def build_decision_surface(
    spec: Mapping[str, Any],
    paths: StudyPaths,
    schedule: pd.DataFrame,
    dense: pl.DataFrame,
) -> pd.DataFrame:
    """Assemble the full historical weekly surface, labels and eligibility flags."""

    decision_dates = schedule["decision_dt"].dt.date.to_list()
    schedule_pl = pl.from_pandas(schedule).with_columns(pl.all().exclude("decision_session_no").cast(pl.Date))
    surface = (
        dense.filter(pl.col("dt").is_in(decision_dates))
        .select(
            "symbol",
            "dt",
            "first_observed_dt",
            "last_observed_dt",
            "adjusted_close",
            "mom_120_20",
            "lowvol_60",
            "sma20",
            "adv20",
            "valid_sessions_60",
            "market_sessions_since_first_observation",
            "distance_to_sma20",
            "ma_allowed",
        )
        .join(schedule_pl, left_on="dt", right_on="decision_dt", how="inner")
    )
    surface = _join_exact_observation(surface, dense, date_column="entry_dt", prefix="entry")
    surface = _join_exact_observation(surface, dense, date_column="exit_5d_dt", prefix="exit_5d")
    surface = _join_exact_observation(surface, dense, date_column="exit_20d_dt", prefix="exit_20d")
    for delay in map(int, spec["sample"]["signal_delays_sessions"]):
        surface = _join_exact_observation(
            surface,
            dense,
            date_column=f"delay_{delay}_entry_dt",
            prefix=f"delay_{delay}_entry",
        )
        surface = _join_exact_observation(
            surface,
            dense,
            date_column=f"delay_{delay}_exit_dt",
            prefix=f"delay_{delay}_exit",
        )

    daily_basic = pl.from_pandas(load_daily_basic(paths, schedule)).with_columns(pl.col("dt").cast(pl.Date))
    surface = surface.join(daily_basic, on=["symbol", "dt"], how="left")
    industry = load_industry_membership(paths)
    surface = _join_asof_interval(
        surface,
        industry,
        columns=("industry_code", "industry_name"),
        prefix="",
    )
    names = load_namechange()
    surface = _join_asof_interval(surface, names, columns=("active_st_name",), prefix="")
    surface = surface.with_columns(
        pl.col("active_st_name").fill_null(False),
        pl.col("symbol").map_elements(infer_board, return_dtype=pl.String).alias("board"),
    )

    states = (
        pl.scan_parquet(paths.state_path)
        .filter(pl.col("dt").cast(pl.Date).is_in(decision_dates))
        .select(pl.col("symbol").cast(pl.String), pl.col("dt").cast(pl.Date), pl.col("regime").cast(pl.Int8))
        .collect(engine="streaming")
    )
    if states.select(pl.struct(["symbol", "dt"]).is_duplicated().any()).item():
        raise ExplorationError("state projection has duplicate decision keys")
    surface = surface.join(states, on=["symbol", "dt"], how="left")

    selection = spec["selection"]
    excluded_boards = list(map(str, selection["exclude_boards"]))
    finite_columns = [
        "mom_120_20",
        "lowvol_60",
        "sma20",
        "adv20",
        "free_float_mcap",
        "adjusted_close",
    ]
    finite = pl.all_horizontal(pl.col(column).is_finite() for column in finite_columns)
    surface = surface.with_columns(
        (
            pl.col("entry_open").is_finite()
            & pl.col("entry_observed").fill_null(False)
            & pl.col("entry_amount").fill_null(0.0).gt(0)
        ).alias("entry_tradable"),
        (pl.col("entry_open") / pl.col("adjusted_close") - 1.0).alias("entry_gap"),
        (pl.col("exit_5d_open") / pl.col("entry_open") - 1.0).alias("fwd_5d_open_return"),
        (pl.col("exit_20d_close") / pl.col("entry_open") - 1.0).alias("fwd_20d_close_return"),
    )
    for delay in map(int, spec["sample"]["signal_delays_sessions"]):
        surface = surface.with_columns(
            (
                pl.col(f"delay_{delay}_entry_open").is_finite()
                & pl.col(f"delay_{delay}_entry_observed").fill_null(False)
                & pl.col(f"delay_{delay}_entry_amount").fill_null(0.0).gt(0)
            ).alias(f"delay_{delay}_tradable"),
            (pl.col(f"delay_{delay}_exit_open") / pl.col(f"delay_{delay}_entry_open") - 1.0).alias(
                f"delay_{delay}_fwd_5d_return"
            ),
        )
    surface = surface.with_columns(
        (
            (pl.col("market_sessions_since_first_observation") >= int(selection["min_market_sessions_since_listing"]))
            & (pl.col("valid_sessions_60") >= int(selection["min_valid_sessions_60"]))
            & (pl.col("adv20") >= float(selection["min_adv20_thousand_cny"]))
            & finite
            & pl.col("industry_code").is_not_null()
            & pl.col("regime").is_not_null()
            & ~pl.col("board").is_in(excluded_boards)
            & ~pl.col("active_st_name")
        ).alias("eligible")
    )
    result = surface.to_pandas()
    result["dt"] = pd.to_datetime(result["dt"])
    result = result.rename(columns={"dt": "decision_dt"})
    result["dt"] = result["decision_dt"]
    for column in result.columns:
        if column.endswith("_dt"):
            result[column] = pd.to_datetime(result[column])
    return result


def rank_weekly_surface(spec: Mapping[str, Any], surface: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen factor definition and deterministic cross-sectional ranks."""

    ranked_input = surface.rename(columns={"decision_dt": "dt"}).copy()
    ranked_input = ranked_input.loc[:, ~ranked_input.columns.duplicated()].copy()
    ranked = rank_cross_section_v2(
        ranked_input,
        q_low=float(spec["selection"]["winsor_limits"][0]),
        q_high=float(spec["selection"]["winsor_limits"][1]),
        buckets=5,
    )
    if ranked.empty:
        raise ExplorationError("factor ranking produced no eligible weekly surface")
    ranked = ranked.rename(columns={"dt": "decision_dt"})
    if ranked.duplicated(["decision_dt", "symbol"]).any():
        raise ExplorationError("ranked weekly surface has duplicate decision identities")
    target = int(spec["selection"]["target_size"])
    incomplete = ranked.groupby("decision_dt", observed=True).size().lt(target)
    if incomplete.any():
        bad = [pd.Timestamp(value).date().isoformat() for value in incomplete[incomplete].index[:5]]
        raise ExplorationError(f"eligible universe below target size on {int(incomplete.sum())} weeks: {bad}")
    return ranked.sort_values(["decision_dt", "factor_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def build_buffered_memberships(
    spec: Mapping[str, Any],
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build separate F, FC and FMA paths while applying gates to new entries only."""

    target_size = int(spec["selection"]["target_size"])
    retention_rank = int(spec["selection"]["retention_max_rank"])
    allowed = set(map(int, spec["selection"]["allowed_chan_regimes"]))
    previous: dict[str, set[str]] = {"F": set(), "FC": set(), "FMA": set()}
    membership_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []

    for decision_dt, week in ranked.groupby("decision_dt", sort=True, observed=True):
        week = week.sort_values(["factor_rank", "symbol"], kind="mergesort").copy()
        indexed = week.set_index("symbol", drop=False)
        eligible_symbols = set(indexed.index)
        for arm in ("F", "FC", "FMA"):
            retained = {
                symbol
                for symbol in previous[arm]
                if symbol in eligible_symbols and int(indexed.at[symbol, "factor_rank"]) <= retention_rank
            }
            slots = target_size - len(retained)
            candidates = [symbol for symbol in week["symbol"] if symbol not in retained][:slots]
            accepted: list[str] = []
            for order, symbol in enumerate(candidates, 1):
                row = indexed.loc[symbol]
                if arm == "F":
                    gate_passed = True
                elif arm == "FC":
                    gate_passed = pd.notna(row["regime"]) and int(row["regime"]) in allowed
                else:
                    gate_passed = bool(row["ma_allowed"])
                proposal_rows.append(
                    {
                        "decision_dt": pd.Timestamp(decision_dt),
                        "arm": arm,
                        "symbol": symbol,
                        "proposal_order": int(order),
                        "gate_passed": bool(gate_passed),
                        "regime": None if pd.isna(row["regime"]) else int(row["regime"]),
                        "industry_code": str(row["industry_code"]),
                        "mcap_bucket": int(row["mcap_bucket"]),
                        "factor_rank": int(row["factor_rank"]),
                        "factor_score": float(row["factor_score"]),
                    }
                )
                if gate_passed:
                    accepted.append(symbol)
            target = retained | set(accepted)
            for symbol in sorted(target, key=lambda value: (int(indexed.at[value, "factor_rank"]), value)):
                row = indexed.loc[symbol]
                membership_rows.append(
                    {
                        "decision_dt": pd.Timestamp(decision_dt),
                        "arm": arm,
                        "symbol": symbol,
                        "membership_role": "retained" if symbol in retained else "new_entry",
                        "factor_rank": int(row["factor_rank"]),
                        "factor_score": float(row["factor_score"]),
                        "regime": None if pd.isna(row["regime"]) else int(row["regime"]),
                    }
                )
            previous[arm] = target

    memberships = pd.DataFrame(membership_rows)
    proposals = pd.DataFrame(proposal_rows)
    if memberships.empty or proposals.empty:
        raise ExplorationError("buffered path construction produced no memberships or proposals")
    evidence_columns = [
        "decision_dt",
        "symbol",
        "industry_code",
        "industry_name",
        "board",
        "free_float_mcap",
        "log_free_float_mcap",
        "mom_120_20",
        "lowvol_60",
        "adv20",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "distance_to_sma20",
        "ma_allowed",
        "regime",
        "factor_score",
        "factor_rank",
        "mcap_bucket",
        "entry_tradable",
        "entry_gap",
        "fwd_5d_open_return",
        "fwd_20d_close_return",
        "entry_open",
        "entry_high",
        "entry_low",
        "entry_amount",
        "exit_5d_open",
        "adjusted_close",
    ]
    for delay in map(int, spec["sample"]["signal_delays_sessions"]):
        evidence_columns.extend(
            [
                f"delay_{delay}_tradable",
                f"delay_{delay}_fwd_5d_return",
                f"delay_{delay}_entry_amount",
            ]
        )
    evidence = ranked[evidence_columns].drop_duplicates(["decision_dt", "symbol"])
    memberships = memberships.merge(
        evidence,
        on=["decision_dt", "symbol", "factor_rank", "factor_score", "regime"],
        how="left",
        validate="many_to_one",
    )
    proposals = proposals.merge(
        evidence,
        on=[
            "decision_dt",
            "symbol",
            "industry_code",
            "mcap_bucket",
            "factor_rank",
            "factor_score",
            "regime",
        ],
        how="left",
        validate="many_to_one",
    )
    return memberships, proposals


def add_forward_drawdown(
    rows: pd.DataFrame,
    dense: pl.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int = 20,
) -> pd.DataFrame:
    """Attach exact-calendar 20-session path MDD without inventing delist terminal values."""

    out = rows.copy().reset_index(drop=True)
    if out.empty:
        out["fwd_20d_max_drawdown"] = pd.Series(dtype=float)
        return out
    session_lookup = pd.Series(np.arange(len(calendar), dtype=np.int32), index=calendar)
    decision_no = out["decision_dt"].map(session_lookup)
    if decision_no.isna().any():
        raise ExplorationError("a membership decision date is absent from the market calendar")
    base = pd.DataFrame(
        {
            "row_id": np.arange(len(out), dtype=np.int64),
            "symbol": out["symbol"].astype(str),
            "decision_no": decision_no.astype(int).to_numpy(),
            "entry_open": pd.to_numeric(out["entry_open"], errors="coerce").to_numpy(),
        }
    )
    offsets = pd.DataFrame({"offset": np.arange(1, horizon + 2, dtype=np.int16)})
    requests = base.merge(offsets, how="cross")
    requests["session_no"] = requests["decision_no"] + requests["offset"]
    requests = requests[requests["session_no"] < len(calendar)].copy()
    requests["dt"] = calendar.take(requests["session_no"].to_numpy())
    path_prices = dense.select(
        pl.col("symbol"),
        pl.col("dt"),
        pl.col("adjusted_close").alias("path_close"),
    )
    request_pl = pl.from_pandas(requests[["row_id", "symbol", "offset", "dt"]]).with_columns(pl.col("dt").cast(pl.Date))
    joined = request_pl.join(path_prices, on=["symbol", "dt"], how="left").to_pandas()
    joined = joined.merge(base[["row_id", "entry_open"]], on="row_id", how="left")

    def path_mdd(group: pd.DataFrame) -> float:
        entry = float(group["entry_open"].iloc[0])
        closes = group.sort_values("offset")["path_close"].to_numpy(dtype=float)
        if not math.isfinite(entry) or entry <= 0 or len(closes) < horizon or not np.isfinite(closes[:horizon]).all():
            return math.nan
        path = np.concatenate(([entry], closes[:horizon]))
        return float(np.min(path / np.maximum.accumulate(path) - 1.0))

    drawdowns = joined.groupby("row_id", sort=False, observed=True).apply(path_mdd, include_groups=False)
    out["fwd_20d_max_drawdown"] = pd.Series(np.nan, index=out.index, dtype=float)
    out.loc[drawdowns.index.astype(int), "fwd_20d_max_drawdown"] = drawdowns.to_numpy(dtype=float)
    return out


def weekly_identity_turnover(frame: pd.DataFrame, *, group_column: str | None = None) -> dict[Any, float]:
    """Return average Jaccard identity turnover for one or more state cohorts."""

    groups: list[tuple[Any, pd.DataFrame]] = (
        [("all", frame)] if group_column is None else list(frame.groupby(group_column, sort=True, observed=True))
    )
    result: dict[Any, float] = {}
    for key, scoped in groups:
        previous: set[str] | None = None
        values: list[float] = []
        for _, week in scoped.groupby("decision_dt", sort=True, observed=True):
            current = set(week["symbol"].astype(str))
            if previous is not None and (current or previous):
                values.append(1.0 - len(current & previous) / len(current | previous))
            previous = current
        result[key] = float(np.mean(values)) if values else math.nan
    return result


def finite_series(values: pd.Series | Sequence[float]) -> pd.Series:
    """Normalize a numeric vector to finite observations only."""

    series = pd.Series(values, dtype=float)
    return series[np.isfinite(series.to_numpy(dtype=float))].reset_index(drop=True)


def return_stats(values: pd.Series | Sequence[float], *, hac_lag: int = 4) -> dict[str, float | int | None]:
    """Summarize a weekly/event return vector without converting absence to zero."""

    series = finite_series(values)
    if series.empty:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "win_rate": None,
            "hac_t": None,
        }
    hac_t = _newey_west_t_weekly(series.to_numpy(dtype=float), max_lag=min(hac_lag, len(series) - 1))
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else None,
        "win_rate": float((series > 0).mean()),
        "hac_t": None if not math.isfinite(hac_t) else float(hac_t),
    }


def _standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    std = float(numeric.std(ddof=0))
    return (numeric - float(numeric.mean())) / std if math.isfinite(std) and std > 0 else numeric * 0.0


def run_risk_attribution(
    spec: Mapping[str, Any],
    ranked: pd.DataFrame,
    memberships: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute weekly buffered factor-pool returns with a cross-sectional risk model."""

    selected = memberships[memberships["arm"].eq("F")][["decision_dt", "symbol"]].copy()
    weekly_rows: list[dict[str, Any]] = []
    for decision_dt, universe in ranked.groupby("decision_dt", sort=True, observed=True):
        universe = universe[
            universe["entry_tradable"].fillna(False)
            & np.isfinite(pd.to_numeric(universe["fwd_5d_open_return"], errors="coerce"))
        ].copy()
        pool_symbols = set(selected.loc[selected["decision_dt"].eq(pd.Timestamp(decision_dt)), "symbol"].astype(str))
        pool = universe[universe["symbol"].isin(pool_symbols)].copy()
        if len(universe) < 100 or len(pool) < 20:
            continue

        universe["size_z"] = _standardize(universe["log_free_float_mcap"])
        universe["lowvol_z"] = _standardize(universe["lowvol_60"])
        universe["momentum_z"] = _standardize(universe["mom_120_20"])
        dummies = pd.get_dummies(universe["industry_code"].astype(str), prefix="industry", dtype=float)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        dummies = dummies - dummies.mean(axis=0)
        if len(dummies.columns):
            dummies = dummies.iloc[:, 1:]
        design = pd.concat(
            [
                pd.Series(1.0, index=universe.index, name="intercept"),
                universe[["size_z", "lowvol_z", "momentum_z"]],
                dummies,
            ],
            axis=1,
        )
        y = universe["fwd_5d_open_return"].astype(float)
        beta, *_ = np.linalg.lstsq(design.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
        fitted = design.to_numpy(dtype=float) @ beta
        residual = y.to_numpy(dtype=float) - fitted
        selected_mask = universe["symbol"].isin(pool_symbols).to_numpy()
        if not selected_mask.any():
            continue
        selected_design = design.to_numpy(dtype=float)[selected_mask]
        selected_y = y.to_numpy(dtype=float)[selected_mask]
        contributions = selected_design * beta
        industry_start = 4
        total_ss = float(np.square(y.to_numpy(dtype=float) - float(y.mean())).sum())
        residual_ss = float(np.square(residual).sum())
        row = {
            "decision_dt": pd.Timestamp(decision_dt),
            "n_universe": int(len(universe)),
            "n_selected": int(selected_mask.sum()),
            "selected_return": float(selected_y.mean()),
            "market": float(contributions[:, 0].mean()),
            "size": float(contributions[:, 1].mean()),
            "low_volatility": float(contributions[:, 2].mean()),
            "momentum": float(contributions[:, 3].mean()),
            "industry": float(contributions[:, industry_start:].sum(axis=1).mean())
            if contributions.shape[1] > industry_start
            else 0.0,
            "unexplained_residual": float((selected_y - contributions.sum(axis=1)).mean()),
            "cross_sectional_r2": float(1.0 - residual_ss / total_ss) if total_ss > 0 else math.nan,
        }
        row["reconstruction_error"] = float(
            row["selected_return"]
            - sum(
                row[key] for key in ("market", "size", "industry", "low_volatility", "momentum", "unexplained_residual")
            )
        )
        weekly_rows.append(row)
    weekly = pd.DataFrame(weekly_rows)
    if weekly.empty or float(weekly["reconstruction_error"].abs().max()) > 1e-10:
        raise ExplorationError("risk attribution is empty or does not reconstruct selected returns")

    summary_rows: list[dict[str, Any]] = []
    for component in (
        "selected_return",
        "market",
        "size",
        "industry",
        "low_volatility",
        "momentum",
        "unexplained_residual",
    ):
        stats = return_stats(weekly[component], hac_lag=int(spec["attribution"]["summary_hac_lag"]))
        summary_rows.append(
            {
                "component": component,
                **stats,
                "annualized_arithmetic": None if stats["mean"] is None else float(stats["mean"]) * 52.0,
                "share_of_selected_mean": (
                    math.nan
                    if abs(float(weekly["selected_return"].mean())) < 1e-12
                    else float(weekly[component].mean() / weekly["selected_return"].mean())
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return weekly, summary


def run_state_diagnostics(
    spec: Mapping[str, Any],
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize coverage, returns, drawdown, identity turnover and tradability by state."""

    factor_pool = memberships[memberships["arm"].eq("F")].copy()
    allowed_states = list(map(int, spec["selection"]["allowed_chan_regimes"]))
    turnover = weekly_identity_turnover(factor_pool[factor_pool["regime"].isin(allowed_states)], group_column="regime")
    turnover["allowed_5_8"] = weekly_identity_turnover(factor_pool[factor_pool["regime"].isin(allowed_states)])["all"]
    slot_capital = float(spec["selection"]["slot_capital_cny"])
    participation_limit = float(spec["selection"]["adv_participation_limit"])
    rows: list[dict[str, Any]] = []
    cohorts: list[tuple[str, pd.DataFrame]] = [
        (str(state), factor_pool[factor_pool["regime"].eq(state)].copy()) for state in allowed_states
    ]
    cohorts.append(("allowed_5_8", factor_pool[factor_pool["regime"].isin(allowed_states)].copy()))
    total = len(factor_pool)
    for state_key, cohort in cohorts:
        weekly_returns = _weekly_cohort_returns(cohort[cohort["entry_tradable"].fillna(False)])
        return_summary = return_stats(weekly_returns)
        market_weeks = int(factor_pool["decision_dt"].nunique())
        weekly_coverage = (
            cohort.groupby("decision_dt", observed=True)
            .size()
            .reindex(sorted(factor_pool["decision_dt"].unique()), fill_value=0)
        )
        adv_capacity = slot_capital <= (pd.to_numeric(cohort["adv20"], errors="coerce") * 1_000.0 * participation_limit)
        day_capacity = slot_capital <= (
            pd.to_numeric(cohort["entry_amount"], errors="coerce") * 1_000.0 * participation_limit
        )
        board_limit = np.where(cohort["board"].eq("CHINEXT"), 0.198, 0.098)
        one_price = (
            np.isclose(cohort["entry_open"], cohort["entry_high"], rtol=0, atol=1e-10)
            & np.isclose(cohort["entry_open"], cohort["entry_low"], rtol=0, atol=1e-10)
            & (cohort["entry_gap"] >= board_limit)
        )
        drawdown = finite_series(cohort["fwd_20d_max_drawdown"])
        state_int = int(state_key) if state_key != "allowed_5_8" else None
        turnover_key: str | int = "allowed_5_8" if state_int is None else state_int
        rows.append(
            {
                "state": state_key,
                "state_name": "Allowed5To8" if state_int is None else STATE_NAMES[state_int],
                "state_name_zh": "状态5–8合计" if state_int is None else STATE_NAMES_ZH[state_int],
                "n_pool_rows": int(len(cohort)),
                "pool_coverage": float(len(cohort) / total) if total else math.nan,
                "weeks_with_observation": int(cohort["decision_dt"].nunique()),
                "total_weeks": market_weeks,
                "mean_names_per_week": float(weekly_coverage.mean()),
                "median_names_per_week": float(weekly_coverage.median()),
                "identity_turnover_jaccard": float(turnover.get(turnover_key, math.nan)),
                "entry_tradable_rate": float(cohort["entry_tradable"].fillna(False).mean())
                if len(cohort)
                else math.nan,
                "adv_capacity_proxy_rate": float(adv_capacity.fillna(False).mean()) if len(cohort) else math.nan,
                "next_day_capacity_proxy_rate": float(day_capacity.fillna(False).mean()) if len(cohort) else math.nan,
                "one_price_limit_up_proxy_rate": float(pd.Series(one_price).fillna(False).mean())
                if len(cohort)
                else math.nan,
                "median_adv20_thousand_cny": float(pd.to_numeric(cohort["adv20"], errors="coerce").median())
                if len(cohort)
                else math.nan,
                "fwd_5d_n_events": int(
                    cohort.loc[
                        cohort["entry_tradable"].fillna(False),
                        "fwd_5d_open_return",
                    ]
                    .notna()
                    .sum()
                ),
                "fwd_5d_n_weeks": int(return_summary["n"]),
                "fwd_5d_mean": return_summary["mean"],
                "fwd_5d_median": return_summary["median"],
                "fwd_5d_win_rate": return_summary["win_rate"],
                "fwd_5d_hac_t": return_summary["hac_t"],
                "fwd_20d_mdd_n": int(len(drawdown)),
                "fwd_20d_mdd_mean": float(drawdown.mean()) if len(drawdown) else math.nan,
                "fwd_20d_mdd_p10": float(drawdown.quantile(0.10)) if len(drawdown) else math.nan,
                "fwd_20d_mdd_median": float(drawdown.median()) if len(drawdown) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def run_failure_profile(
    spec: Mapping[str, Any],
    memberships: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare non-positive and profitable factor-pool outcomes without promoting filters."""

    pool = memberships[
        memberships["arm"].eq("F")
        & memberships["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(memberships["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    if pool.empty:
        raise ExplorationError("failure profile has no tradable labeled factor-pool rows")
    pool["failure_primary"] = pool["fwd_5d_open_return"] <= 0
    pool["weekly_return_quintile"] = pool.groupby("decision_dt", observed=True)["fwd_5d_open_return"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            q=min(5, len(values)),
            labels=False,
            duplicates="drop",
        )
    )
    pool["failure_bottom_quintile"] = pool["weekly_return_quintile"].eq(0)
    split_date = pd.Timestamp(spec["failure_profile"]["split_date"])
    pool["segment"] = np.where(pool["decision_dt"] < split_date, "early", "late")

    timed_features = [
        (
            "decision_time",
            list(spec["failure_profile"]["decision_time_numeric_features"]),
        ),
        (
            "execution_time",
            list(spec["failure_profile"]["execution_time_numeric_features"]),
        ),
        ("outcome", list(spec["failure_profile"]["outcome_diagnostics"])),
    ]
    numeric_rows: list[dict[str, Any]] = []
    for segment in ("all", "early", "late"):
        scoped = pool if segment == "all" else pool[pool["segment"].eq(segment)]
        failure = scoped[scoped["failure_primary"]]
        profit = scoped[~scoped["failure_primary"]]
        for feature_timing, features in timed_features:
            for feature in features:
                overall = pd.to_numeric(scoped[feature], errors="coerce")
                loss_values = pd.to_numeric(failure[feature], errors="coerce")
                profit_values = pd.to_numeric(profit[feature], errors="coerce")
                std = float(overall.std(ddof=0))
                difference = float(loss_values.mean() - profit_values.mean())
                numeric_rows.append(
                    {
                        "segment": segment,
                        "feature_timing": feature_timing,
                        "feature": feature,
                        "n_total": int(overall.notna().sum()),
                        "n_failure": int(loss_values.notna().sum()),
                        "n_profit": int(profit_values.notna().sum()),
                        "failure_mean": float(loss_values.mean()),
                        "profit_mean": float(profit_values.mean()),
                        "mean_difference": difference,
                        "standardized_difference": difference / std if math.isfinite(std) and std > 0 else math.nan,
                        "failure_median": float(loss_values.median()),
                        "profit_median": float(profit_values.median()),
                    }
                )

    categorical_rows: list[dict[str, Any]] = []
    overall_failure_rate = float(pool["failure_primary"].mean())
    for feature in spec["failure_profile"]["categorical_features"]:
        grouped = (
            pool.assign(category=pool[feature].astype("string").fillna("<MISSING>"))
            .groupby("category", sort=True, observed=True)
            .agg(n=("failure_primary", "size"), failure_rate=("failure_primary", "mean"))
            .reset_index()
        )
        grouped["feature"] = feature
        grouped["overall_failure_rate"] = overall_failure_rate
        grouped["failure_rate_lift"] = grouped["failure_rate"] - overall_failure_rate
        categorical_rows.extend(grouped.to_dict("records"))
    return pool, pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def _weekly_cohort_returns(
    frame: pd.DataFrame,
    *,
    return_column: str = "fwd_5d_open_return",
) -> pd.Series:
    scoped = frame[np.isfinite(pd.to_numeric(frame[return_column], errors="coerce"))].copy()
    if scoped.empty:
        return pd.Series(dtype=float)
    return scoped.groupby("decision_dt", sort=True, observed=True)[return_column].mean().astype(float)


def _null_tail_probability(actual: float, null_values: Sequence[float]) -> float:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((1 + np.sum(values >= actual)) / (1 + len(values))) if len(values) else math.nan


def run_state_randomization(
    spec: Mapping[str, Any],
    proposals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute gate identity within week while preserving every weekly gate count."""

    scoped = proposals[
        proposals["arm"].eq("FC")
        & proposals["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(proposals["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    if scoped.empty or not scoped["gate_passed"].any():
        raise ExplorationError("state-randomization control has no tradable gated proposals")
    actual_weekly = _weekly_cohort_returns(scoped[scoped["gate_passed"]])
    actual = float(actual_weekly.mean())
    repetitions = int(spec["negative_controls"]["random_repetitions"])
    root = int(spec["negative_controls"]["random_seed_root"])
    distribution_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(root + 10_000 + repetition)
        weekly_values: list[float] = []
        total_selected = 0
        for _decision_dt, week in scoped.groupby("decision_dt", sort=True, observed=True):
            count = int(week["gate_passed"].sum())
            if count <= 0:
                continue
            chosen = rng.choice(week.index.to_numpy(), size=count, replace=False)
            values = week.loc[chosen, "fwd_5d_open_return"].to_numpy(dtype=float)
            weekly_values.append(float(np.mean(values)))
            total_selected += len(values)
        distribution_rows.append(
            {
                "experiment": "state_randomization",
                "repetition": int(repetition),
                "mean_weekly_return": float(np.mean(weekly_values)) if weekly_values else math.nan,
                "weeks": int(len(weekly_values)),
                "selected_rows": int(total_selected),
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    null_values = distribution["mean_weekly_return"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "experiment": "state_randomization",
                "variant": "actual_chan_gate",
                "n_events": int(scoped["gate_passed"].sum()),
                "weeks": int(len(actual_weekly)),
                "mean_weekly_return": actual,
                "median_weekly_return": float(actual_weekly.median()),
                "null_mean": float(np.nanmean(null_values)),
                "actual_minus_null": float(actual - np.nanmean(null_values)),
                "one_sided_null_p": _null_tail_probability(actual, null_values),
                "match_rate": 1.0,
            }
        ]
    )
    return summary, distribution


def run_industry_size_matched_random(
    spec: Mapping[str, Any],
    ranked: pd.DataFrame,
    proposals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample same-week, same-industry and same-size-quintile identities."""

    actual = proposals[
        proposals["arm"].eq("FC")
        & proposals["gate_passed"]
        & proposals["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(proposals["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    universe = ranked[
        ranked["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(ranked["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    if actual.empty or universe.empty:
        raise ExplorationError("industry-size control lacks actual or control observations")
    actual_weekly = _weekly_cohort_returns(actual)
    actual_mean = float(actual_weekly.mean())
    repetitions = int(spec["negative_controls"]["random_repetitions"])
    root = int(spec["negative_controls"]["random_seed_root"])
    actual_identity = set(zip(actual["decision_dt"], actual["symbol"], strict=False))
    grouped_pool = {
        key: group[~group.apply(lambda row: (row["decision_dt"], row["symbol"]) in actual_identity, axis=1)].copy()
        for key, group in universe.groupby(["decision_dt", "industry_code", "mcap_bucket"], observed=True)
    }
    distribution_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(root + 20_000 + repetition)
        picked: list[pd.DataFrame] = []
        requested = 0
        matched = 0
        for key, cohort in actual.groupby(["decision_dt", "industry_code", "mcap_bucket"], sort=True, observed=True):
            count = len(cohort)
            requested += count
            pool = grouped_pool.get(key)
            if pool is None or pool.empty:
                continue
            take = min(count, len(pool))
            chosen = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            picked.append(pool.loc[chosen])
            matched += take
        controls = pd.concat(picked, ignore_index=True) if picked else universe.iloc[:0].copy()
        weekly = _weekly_cohort_returns(controls)
        distribution_rows.append(
            {
                "experiment": "industry_size_matched_random",
                "repetition": int(repetition),
                "mean_weekly_return": float(weekly.mean()) if len(weekly) else math.nan,
                "weeks": int(len(weekly)),
                "requested_rows": int(requested),
                "matched_rows": int(matched),
                "match_rate": float(matched / requested) if requested else math.nan,
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    null_values = distribution["mean_weekly_return"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "experiment": "industry_size_matched_random",
                "variant": "actual_chan_gate",
                "n_events": int(len(actual)),
                "weeks": int(len(actual_weekly)),
                "mean_weekly_return": actual_mean,
                "median_weekly_return": float(actual_weekly.median()),
                "null_mean": float(np.nanmean(null_values)),
                "actual_minus_null": float(actual_mean - np.nanmean(null_values)),
                "one_sided_null_p": _null_tail_probability(actual_mean, null_values),
                "match_rate": float(distribution["match_rate"].mean()),
            }
        ]
    )
    return summary, distribution


def run_signal_delay(
    spec: Mapping[str, Any],
    proposals: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the fixed, preregistered delay set without selecting a winner."""

    actual = proposals[proposals["arm"].eq("FC") & proposals["gate_passed"]].copy()
    if actual.empty:
        raise ExplorationError("signal-delay diagnostic has no Chan-gated proposals")
    rows: list[dict[str, Any]] = []
    for delay in map(int, spec["sample"]["signal_delays_sessions"]):
        tradable_column = f"delay_{delay}_tradable"
        return_column = f"delay_{delay}_fwd_5d_return"
        scoped = actual[
            actual[tradable_column].fillna(False) & np.isfinite(pd.to_numeric(actual[return_column], errors="coerce"))
        ]
        weekly = _weekly_cohort_returns(scoped, return_column=return_column)
        stats = return_stats(weekly)
        rows.append(
            {
                "experiment": "signal_delay",
                "variant": f"delay_{delay}",
                "delay_sessions": delay,
                "n_events": int(len(scoped)),
                "tradable_rate": float(actual[tradable_column].fillna(False).mean()),
                "weeks": int(len(weekly)),
                "mean_weekly_return": stats["mean"],
                "median_weekly_return": stats["median"],
                "hac_t": stats["hac_t"],
            }
        )
    return pd.DataFrame(rows)


def run_simple_trend_gate(proposals: pd.DataFrame) -> pd.DataFrame:
    """Compare the Chan gate with the fixed SMA20 gate on the same proposal identities."""

    scoped = proposals[
        proposals["arm"].eq("FC")
        & proposals["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(proposals["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    if scoped.empty:
        raise ExplorationError("simple-trend control has no tradable proposals")
    variants = {
        "ungated_factor_proposals": pd.Series(True, index=scoped.index),
        "chan_states_5_8": scoped["gate_passed"].astype(bool),
        "sma20": scoped["ma_allowed"].fillna(False).astype(bool),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in variants.items():
        cohort = scoped[mask].copy()
        weekly = _weekly_cohort_returns(cohort)
        stats = return_stats(weekly)
        rows.append(
            {
                "experiment": "simple_trend_gate",
                "variant": name,
                "n_events": int(len(cohort)),
                "coverage": float(mask.mean()),
                "weeks": int(len(weekly)),
                "mean_weekly_return": stats["mean"],
                "median_weekly_return": stats["median"],
                "hac_t": stats["hac_t"],
            }
        )
    result = pd.DataFrame(rows)
    chan_mean = float(result.loc[result["variant"].eq("chan_states_5_8"), "mean_weekly_return"].iloc[0])
    sma_mean = float(result.loc[result["variant"].eq("sma20"), "mean_weekly_return"].iloc[0])
    result["chan_minus_sma20"] = chan_mean - sma_mean
    return result


def derive_next_hypotheses(
    attribution_summary: pd.DataFrame,
    state_diagnostics: pd.DataFrame,
    failure_numeric: pd.DataFrame,
    negative_controls: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create three fixed-form, falsifiable next-stage hypotheses from Stage 1 evidence."""

    residual_row = attribution_summary.loc[attribution_summary["component"].eq("unexplained_residual")].iloc[0]
    state7 = state_diagnostics.loc[state_diagnostics["state"].astype(str).eq("7")].iloc[0]
    sma_rows = negative_controls[
        negative_controls["experiment"].eq("simple_trend_gate")
        & negative_controls["variant"].isin(["chan_states_5_8", "sma20"])
    ]
    chan_mean = float(sma_rows.loc[sma_rows["variant"].eq("chan_states_5_8"), "mean_weekly_return"].iloc[0])
    sma_mean = float(sma_rows.loc[sma_rows["variant"].eq("sma20"), "mean_weekly_return"].iloc[0])
    profiles = failure_numeric[
        failure_numeric["segment"].eq("all") & failure_numeric["feature_timing"].eq("decision_time")
    ].copy()
    profiles["abs_standardized_difference"] = profiles["standardized_difference"].abs()
    top_features = profiles.sort_values(
        ["abs_standardized_difference", "feature"], ascending=[False, True], kind="mergesort"
    ).head(2)
    feature_names = top_features["feature"].astype(str).tolist()
    return [
        {
            "id": "H1_STATE7_HAS_CHAN_SPECIFIC_INCREMENT",
            "statement": (
                "状态7在相同因子候选中不是普通趋势门的替代标签；其固定身份相对SMA20和行业规模匹配随机组"
                "具有至少20bp/周的正增量。"
            ),
            "next_test": "冻结状态7、SMA20和行业规模匹配身份，在未用于本阶段的后续样本上比较周收益差。",
            "pass_rule": "两项差值均值均>=0.002且两个时间半段同号。",
            "falsify_rule": "任一差值<=0，或两个时间半段符号相反。",
            "stage1_evidence": {
                "state7_mean_return": state7["fwd_5d_mean"],
                "chan_gate_mean_weekly_return": chan_mean,
                "sma20_mean_weekly_return": sma_mean,
                "chan_minus_sma20": chan_mean - sma_mean,
            },
        },
        {
            "id": "H2_FACTOR_HAS_RESIDUAL_AFTER_RISK_ATTRIBUTION",
            "statement": "50/75因子池在市场、规模、行业、低波和动量归因后仍有至少20bp/周的剩余收益。",
            "next_test": "冻结本阶段归因模型，在按时间前推的历史留出段或新数据上只计算未解释残差。",
            "pass_rule": "残差均值>=0.002且早晚两个样本段均为正。",
            "falsify_rule": "残差均值<=0，或仅由单一时期贡献。",
            "stage1_evidence": {
                "residual_mean": residual_row["mean"],
                "residual_annualized_arithmetic": residual_row["annualized_arithmetic"],
                "residual_hac_t": residual_row["hac_t"],
            },
        },
        {
            "id": "H3_FAILURE_PROFILE_REPLICATES_OUT_OF_TIME",
            "statement": f"低收益样本的两个最强可观测画像（{', '.join(feature_names)}）能跨时间复现。",
            "next_test": "只用2022–2023确定方向，在2024年以后检查同方向标准化差异与失败率提升。",
            "pass_rule": "两个特征在晚期样本均保持同方向且|标准化差异|>=0.10。",
            "falsify_rule": "任一特征反号或两个特征均低于0.10。",
            "stage1_evidence": top_features[["feature", "standardized_difference", "mean_difference"]].to_dict(
                "records"
            ),
        },
    ]


def directory_fingerprint(paths: Sequence[Path]) -> str:
    """Hash file names, sizes and content hashes for an exploratory input set."""

    entries = [
        {
            "name": path.name,
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    return hashlib.sha256(canonical_json(entries)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _format_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def _format_num(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    """Render a compact Markdown table from dictionaries."""

    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_report(
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    attribution_summary: pd.DataFrame,
    state_diagnostics: pd.DataFrame,
    failure_numeric: pd.DataFrame,
    failure_categorical: pd.DataFrame,
    negative_controls: pd.DataFrame,
    hypotheses: Sequence[Mapping[str, Any]],
    registry: Sequence[ExperimentRecord],
    output_path: Path,
) -> None:
    """Write the human-readable Stage 1 evidence report."""

    attribution_rows = []
    for row in attribution_summary.to_dict("records"):
        attribution_rows.append(
            {
                "component": row["component"],
                "weekly": _format_pct(row["mean"]),
                "annualized": _format_pct(row["annualized_arithmetic"]),
                "HAC t": _format_num(row["hac_t"], 2),
                "share": _format_pct(row["share_of_selected_mean"]),
            }
        )
    state_rows = []
    for row in state_diagnostics.to_dict("records"):
        state_rows.append(
            {
                "state": row["state_name_zh"],
                "coverage": _format_pct(row["pool_coverage"]),
                "n": row["fwd_5d_n_events"],
                "5d mean": _format_pct(row["fwd_5d_mean"]),
                "win": _format_pct(row["fwd_5d_win_rate"]),
                "MDD": _format_pct(row["fwd_20d_mdd_mean"]),
                "turnover": _format_pct(row["identity_turnover_jaccard"]),
                "tradable": _format_pct(row["entry_tradable_rate"]),
            }
        )
    top_numeric = (
        failure_numeric[failure_numeric["segment"].eq("all") & failure_numeric["feature_timing"].eq("decision_time")]
        .assign(abs_diff=lambda frame: frame["standardized_difference"].abs())
        .sort_values(["abs_diff", "feature"], ascending=[False, True], kind="mergesort")
        .head(8)
    )
    failure_rows = [
        {
            "feature": row["feature"],
            "loss": _format_num(row["failure_mean"]),
            "profit": _format_num(row["profit_mean"]),
            "std diff": _format_num(row["standardized_difference"]),
        }
        for row in top_numeric.to_dict("records")
    ]
    outcome_rows = [
        {
            "feature": row["feature"],
            "loss": _format_num(row["failure_mean"]),
            "profit": _format_num(row["profit_mean"]),
            "std diff": _format_num(row["standardized_difference"]),
        }
        for row in failure_numeric[
            failure_numeric["segment"].eq("all") & failure_numeric["feature_timing"].isin(["execution_time", "outcome"])
        ].to_dict("records")
    ]
    control_rows = []
    for row in negative_controls.to_dict("records"):
        control_rows.append(
            {
                "experiment": row.get("experiment"),
                "variant": row.get("variant"),
                "n": row.get("n_events", ""),
                "weekly": _format_pct(row.get("mean_weekly_return")),
                "delta/null": _format_pct(
                    row.get("actual_minus_null")
                    if pd.notna(row.get("actual_minus_null"))
                    else row.get("chan_minus_sma20")
                ),
                "p": _format_num(row.get("one_sided_null_p"), 3),
            }
        )
    registry_rows = [
        {
            "id": record.experiment_id,
            "status": record.status,
            "seconds": _format_num(record.duration_seconds, 1),
            "error": record.error_message or "",
        }
        for record in registry
    ]
    hypothesis_sections = []
    for hypothesis in hypotheses:
        evidence = json.dumps(_json_safe(hypothesis["stage1_evidence"]), ensure_ascii=False, sort_keys=True)
        hypothesis_sections.extend(
            [
                f"### {hypothesis['id']}",
                "",
                str(hypothesis["statement"]),
                "",
                f"- 下一测试：{hypothesis['next_test']}",
                f"- 通过：{hypothesis['pass_rule']}",
                f"- 证伪：{hypothesis['falsify_rule']}",
                f"- Stage 1 证据：`{evidence}`",
                "",
            ]
        )

    lines = [
        f"# 横截面因子 × 缠论探索研究 Stage 1 结果（{spec['study_id']}）",
        "",
        "## 结论边界",
        "",
        "本报告是独立的污染历史探索，不修改 V2.1、不启动确认链，也不授权实盘。归因是事后描述，"
        "负对照用于提出或淘汰机制假设，不是确认性统计。",
        "",
        f"- 排定样本：{manifest['schedule']['scheduled_start']} 至 {manifest['schedule']['scheduled_end']}，"
        f"{manifest['schedule']['scheduled_weeks']} 个完整周",
        f"- 实际分析：{manifest['schedule']['analyzed_start']} 至 {manifest['schedule']['analyzed_end']}，"
        f"{manifest['schedule']['analyzed_weeks']} 周；因固定资格门结构性剔除 "
        f"{manifest['schedule']['dropped_weeks']} 周",
        f"- 决策面：{manifest['surface']['rows']:,} 行，合格行 {manifest['surface']['eligible_rows']:,}",
        f"- 因子池：{manifest['memberships']['factor_pool_rows']:,} 个周度身份",
        f"- V2.1 协议物理 SHA256：`{manifest['frozen_baseline']['protocol_physical_sha256']}`",
        "",
        "## 收益归因",
        "",
        markdown_table(
            attribution_rows,
            (
                ("component", "分量"),
                ("weekly", "周均贡献"),
                ("annualized", "算术年化"),
                ("HAC t", "HAC t"),
                ("share", "占池收益"),
            ),
        ),
        "",
        "各周截面模型使用居中的规模、低波、动量和申万一级行业暴露；六个分量逐周精确重构因子池收益。"
        "“未解释残差”不是 alpha，只表示该描述模型没有解释的部分。",
        "",
        "## 状态 5–8",
        "",
        markdown_table(
            state_rows,
            (
                ("state", "状态"),
                ("coverage", "因子池覆盖"),
                ("n", "收益事件数"),
                ("5d mean", "5日周均收益"),
                ("win", "周胜率"),
                ("MDD", "20日路径MDD"),
                ("turnover", "身份换手"),
                ("tradable", "次日可观测开盘"),
            ),
        ),
        "",
        "成交能力仅是下一 session 是否有开盘行、ADV20 与 5% 参与率代理；没有集合竞价和官方停复牌证据，"
        "不得解释为真实可成交率。",
        "",
        "## 低收益样本画像",
        "",
        f"主失败定义为入池且次日可观测开盘后 5-session 收益不高于 0。最强的八个决策时数值差异如下；"
        f"正号表示亏损样本均值更高，负号表示更低。分类画像另存 {len(failure_categorical):,} 行。",
        "",
        markdown_table(
            failure_rows,
            (
                ("feature", "特征"),
                ("loss", "亏损均值"),
                ("profit", "盈利均值"),
                ("std diff", "标准化差异"),
            ),
        ),
        "",
        "执行时与结果诊断单独列示，禁止冒充决策时预测特征：",
        "",
        markdown_table(
            outcome_rows,
            (
                ("feature", "字段"),
                ("loss", "亏损均值"),
                ("profit", "盈利均值"),
                ("std diff", "标准化差异"),
            ),
        ),
        "",
        "这些差异没有被升级为过滤条件，也没有进行阈值搜索。未来 20-session MDD 只描述亏损路径，不参与假设特征选择。",
        "",
        "## 负对照",
        "",
        markdown_table(
            control_rows,
            (
                ("experiment", "实验"),
                ("variant", "变体"),
                ("n", "事件数"),
                ("weekly", "周均收益"),
                ("delta/null", "相对空值/门差"),
                ("p", "随机尾概率"),
            ),
        ),
        "",
        "随机尾概率只描述固定 100 次随机实验中的位置，不是多重检验校正后的确认性 p 值。",
        "",
        "## 下一轮三个可证伪假设",
        "",
        *hypothesis_sections,
        "## 实验登记",
        "",
        markdown_table(
            registry_rows,
            (("id", "实验"), ("status", "状态"), ("seconds", "秒"), ("error", "错误")),
        ),
        "",
        "## 数据限制",
        "",
        "- qfq 行情会被未来公司行为重写，只能用于历史压力探索。",
        "- 状态缓存来自同一污染行情；其 2,000 次 prefix 重放只证明工程因果一致，不证明数据正式有效。",
        "- 每周 daily_basic 与申万成员关系是运行时 API 缓存，不是 V2.1 要求的不可变原始响应档案。",
        "- 缺少官方停复牌、集合竞价、完整公司行为与退市终值；缺失标签保持缺失，不做归零或成交猜测。",
        "- 收益使用 qfq 开盘代理，不含 V2.1 的逐日账户会计、税费、容量和受阻卖出。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _start_registry(spec: Mapping[str, Any], output_dir: Path) -> dict[str, ExperimentRecord]:
    records = {
        row["id"]: ExperimentRecord(experiment_id=row["id"], deliverable=row["deliverable"])
        for row in spec["planned_experiments"]
    }
    _write_registry(spec, records, output_dir)
    return records


def _write_registry(
    spec: Mapping[str, Any],
    records: Mapping[str, ExperimentRecord],
    output_dir: Path,
) -> None:
    write_json(
        output_dir / "experiment_registry.json",
        {
            "study_id": spec["study_id"],
            "mode": spec["mode"],
            "updated_at_utc": utc_now(),
            "experiments": [records[key].as_dict() for key in records],
        },
    )


def execute_experiment(
    spec: Mapping[str, Any],
    records: dict[str, ExperimentRecord],
    output_dir: Path,
    experiment_id: str,
    function: Callable[[], Any],
    row_counter: Callable[[Any], Mapping[str, int]],
) -> Any:
    """Execute and persist one experiment status, including every failure."""

    record = records[experiment_id]
    record.status = "RUNNING"
    record.started_at_utc = utc_now()
    started = time.monotonic()
    _write_registry(spec, records, output_dir)
    try:
        result = function()
    except Exception as exc:
        record.status = "FAILED"
        record.completed_at_utc = utc_now()
        record.duration_seconds = round(time.monotonic() - started, 6)
        record.error_type = type(exc).__name__
        record.error_message = str(exc)
        record.traceback_tail = "\n".join(traceback.format_exc().splitlines()[-12:])
        _write_registry(spec, records, output_dir)
        raise
    record.status = "COMPLETED"
    record.completed_at_utc = utc_now()
    record.duration_seconds = round(time.monotonic() - started, 6)
    record.output_rows = {str(key): int(value) for key, value in row_counter(result).items()}
    _write_registry(spec, records, output_dir)
    return result


def run_study(spec: Mapping[str, Any], paths: StudyPaths) -> dict[str, Any]:
    """Build, execute and verify the complete Stage 1 exploratory evidence bundle."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    records = _start_registry(spec, paths.output_dir)
    calendar = load_market_calendar(spec["sample"]["end_date"])
    schedule = build_decision_schedule(spec, calendar)
    print(f"[study] calendar={len(calendar)} sessions decisions={len(schedule)}", flush=True)
    dense = build_dense_feature_panel(spec, calendar)
    print(f"[study] dense feature panel={dense.height:,} rows", flush=True)
    surface = build_decision_surface(spec, paths, schedule, dense)
    print(
        f"[study] decision surface={len(surface):,} eligible={int(surface['eligible'].sum()):,}",
        flush=True,
    )
    target_size = int(spec["selection"]["target_size"])
    schedule_coverage = (
        surface.groupby("decision_dt", observed=True)["eligible"]
        .sum()
        .reindex(schedule["decision_dt"], fill_value=0)
        .rename("eligible_count")
        .reset_index()
    )
    schedule_coverage["status"] = np.where(
        schedule_coverage["eligible_count"].ge(target_size),
        "ANALYZED",
        "DROPPED_BELOW_TARGET",
    )
    schedule_coverage.to_parquet(paths.output_dir / "schedule_coverage.parquet", index=False)
    ranked = rank_weekly_surface(spec, surface)
    analyzed_dates = pd.DatetimeIndex(sorted(ranked["decision_dt"].unique()))
    expected_analyzed = pd.DatetimeIndex(
        schedule_coverage.loc[schedule_coverage["status"].eq("ANALYZED"), "decision_dt"]
    )
    if not analyzed_dates.equals(expected_analyzed):
        raise ExplorationError("ranked decision dates differ from schedule eligibility coverage")
    memberships, proposals = build_buffered_memberships(spec, ranked)
    memberships = add_forward_drawdown(memberships, dense, calendar, horizon=20)
    print(
        f"[study] ranked={len(ranked):,} memberships={len(memberships):,} proposals={len(proposals):,}",
        flush=True,
    )

    surface.to_parquet(paths.output_dir / "decision_surface.parquet", index=False)
    ranked.to_parquet(paths.output_dir / "ranked_surface.parquet", index=False)
    memberships.to_parquet(paths.output_dir / "buffered_memberships.parquet", index=False)
    proposals.to_parquet(paths.output_dir / "gate_proposals.parquet", index=False)
    schedule.to_parquet(paths.output_dir / "decision_schedule.parquet", index=False)

    attribution_weekly, attribution_summary = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E01_RISK_ATTRIBUTION",
        lambda: run_risk_attribution(spec, ranked, memberships),
        lambda result: {"weekly": len(result[0]), "summary": len(result[1])},
    )
    attribution_weekly.to_parquet(paths.output_dir / "attribution_weekly.parquet", index=False)
    attribution_summary.to_parquet(paths.output_dir / "attribution_summary.parquet", index=False)

    state_diagnostics = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E02_STATE_5_8_DIAGNOSTICS",
        lambda: run_state_diagnostics(spec, memberships),
        lambda result: {"state_rows": len(result)},
    )
    state_diagnostics.to_parquet(paths.output_dir / "state_diagnostics.parquet", index=False)

    failure_pool, failure_numeric, failure_categorical = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E03_LOW_RETURN_PROFILE",
        lambda: run_failure_profile(spec, memberships),
        lambda result: {
            "sample_rows": len(result[0]),
            "numeric_profile_rows": len(result[1]),
            "categorical_profile_rows": len(result[2]),
        },
    )
    failure_pool.to_parquet(paths.output_dir / "failure_sample.parquet", index=False)
    failure_numeric.to_parquet(paths.output_dir / "failure_numeric_profile.parquet", index=False)
    failure_categorical.to_parquet(paths.output_dir / "failure_categorical_profile.parquet", index=False)

    state_random_summary, state_random_distribution = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E04_STATE_RANDOMIZATION",
        lambda: run_state_randomization(spec, proposals),
        lambda result: {"summary_rows": len(result[0]), "null_rows": len(result[1])},
    )
    state_random_distribution.to_parquet(paths.output_dir / "state_randomization_null.parquet", index=False)

    matched_summary, matched_distribution = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E05_INDUSTRY_SIZE_MATCHED_RANDOM",
        lambda: run_industry_size_matched_random(spec, ranked, proposals),
        lambda result: {"summary_rows": len(result[0]), "null_rows": len(result[1])},
    )
    matched_distribution.to_parquet(paths.output_dir / "industry_size_matched_null.parquet", index=False)

    delay_results = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E06_SIGNAL_DELAY",
        lambda: run_signal_delay(spec, proposals),
        lambda result: {"delay_rows": len(result)},
    )
    trend_results = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E07_SIMPLE_TREND_GATE",
        lambda: run_simple_trend_gate(proposals),
        lambda result: {"gate_rows": len(result)},
    )
    negative_controls = pd.concat(
        [state_random_summary, matched_summary, delay_results, trend_results],
        ignore_index=True,
        sort=False,
    )
    negative_controls.to_parquet(paths.output_dir / "negative_controls.parquet", index=False)

    hypotheses = execute_experiment(
        spec,
        records,
        paths.output_dir,
        "E08_NEXT_HYPOTHESES",
        lambda: derive_next_hypotheses(
            attribution_summary,
            state_diagnostics,
            failure_numeric,
            negative_controls,
        ),
        lambda result: {"hypotheses": len(result)},
    )
    write_json(paths.output_dir / "next_hypotheses.json", hypotheses)

    daily_basic_files = list(paths.daily_basic_dir.glob("*.parquet"))
    manifest = {
        "study_id": spec["study_id"],
        "mode": spec["mode"],
        "completed_at_utc": utc_now(),
        "spec": {
            "path": str(SPEC_PATH),
            "sha256": sha256_file(SPEC_PATH),
        },
        "frozen_baseline": {
            "protocol_path": str(V21_PROTOCOL_PATH),
            "protocol_physical_sha256": sha256_file(V21_PROTOCOL_PATH),
            "protocol_canonical_sha256": hashlib.sha256(canonical_json(read_json(V21_PROTOCOL_PATH))).hexdigest(),
            "protocol_document_sha256": sha256_file(V21_DOCUMENT_PATH),
            "git_commit_at_freeze": spec["frozen_baseline"]["git_commit"],
        },
        "inputs": {
            "raw_qfq_files": int(len(list(RAW_DIR.glob("*.parquet")))),
            "state_path": str(paths.state_path),
            "state_sha256": sha256_file(paths.state_path),
            "state_manifest_sha256": sha256_file(paths.state_manifest_path),
            "daily_basic_files": int(len(daily_basic_files)),
            "daily_basic_fingerprint": directory_fingerprint(daily_basic_files),
            "industry_sha256": sha256_file(paths.industry_path),
            "namechange_sha256": sha256_file(NAMECHANGE_PATH),
        },
        "schedule": {
            "scheduled_start": schedule["decision_dt"].min().date().isoformat(),
            "scheduled_end": schedule["decision_dt"].max().date().isoformat(),
            "scheduled_weeks": int(len(schedule)),
            "analyzed_start": analyzed_dates.min().date().isoformat(),
            "analyzed_end": analyzed_dates.max().date().isoformat(),
            "analyzed_weeks": int(len(analyzed_dates)),
            "dropped_weeks": int(schedule_coverage["status"].ne("ANALYZED").sum()),
            "dropped_week_records": schedule_coverage.loc[schedule_coverage["status"].ne("ANALYZED")]
            .assign(decision_dt=lambda frame: frame["decision_dt"].dt.date.astype(str))
            .to_dict("records"),
            "calendar_sessions": int(len(calendar)),
        },
        "surface": {
            "rows": int(len(surface)),
            "symbols": int(surface["symbol"].nunique()),
            "eligible_rows": int(surface["eligible"].sum()),
            "eligible_symbols": int(ranked["symbol"].nunique()),
            "entry_tradable_rate_eligible": float(ranked["entry_tradable"].fillna(False).mean()),
            "fwd_5d_label_rate_eligible": float(ranked["fwd_5d_open_return"].notna().mean()),
        },
        "memberships": {
            "rows": int(len(memberships)),
            "factor_pool_rows": int(memberships["arm"].eq("F").sum()),
            "chan_pool_rows": int(memberships["arm"].eq("FC").sum()),
            "sma_pool_rows": int(memberships["arm"].eq("FMA").sum()),
            "proposal_rows": int(len(proposals)),
        },
        "experiments": [record.as_dict() for record in records.values()],
        "limitations": [
            "contaminated_qfq_history",
            "contaminated_state_cache",
            "reference_api_responses_not_immutable_archives",
            "no_official_suspension_or_open_auction_evidence",
            "no_complete_corporate_action_or_delist_terminal_evidence",
            "qfq_open_return_proxy_without_accounting_costs",
        ],
    }
    write_json(paths.output_dir / "data_manifest.json", manifest)
    summary = {
        "study_id": spec["study_id"],
        "mode": spec["mode"],
        "attribution": attribution_summary.to_dict("records"),
        "state_diagnostics": state_diagnostics.to_dict("records"),
        "failure_top_numeric": (
            failure_numeric[
                failure_numeric["segment"].eq("all") & failure_numeric["feature_timing"].eq("decision_time")
            ]
            .assign(abs_diff=lambda frame: frame["standardized_difference"].abs())
            .sort_values(["abs_diff", "feature"], ascending=[False, True])
            .head(10)
            .drop(columns="abs_diff")
            .to_dict("records")
        ),
        "failure_execution_and_outcome_diagnostics": failure_numeric[
            failure_numeric["segment"].eq("all") & failure_numeric["feature_timing"].isin(["execution_time", "outcome"])
        ].to_dict("records"),
        "negative_controls": negative_controls.to_dict("records"),
        "hypotheses": hypotheses,
        "limitations": manifest["limitations"],
    }
    write_json(paths.output_dir / "summary.json", _json_safe(summary))
    write_report(
        spec,
        manifest,
        attribution_summary,
        state_diagnostics,
        failure_numeric,
        failure_categorical,
        negative_controls,
        hypotheses,
        list(records.values()),
        paths.output_dir / "report.md",
    )
    load_and_validate_spec()
    if any(record.status != "COMPLETED" for record in records.values()):
        raise ExplorationError("not every planned experiment completed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-reference-data", help="Fetch/cache weekly size and PIT industry inputs")
    fetch.add_argument("--refresh", action="store_true")
    subparsers.add_parser("run", help="Run all eight fixed exploratory experiments")
    subparsers.add_parser("check", help="Validate spec, baseline and cached reference coverage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_and_validate_spec(args.spec)
    paths = resolve_paths(args.output_dir)
    if args.command == "fetch-reference-data":
        manifest = fetch_reference_data(spec, paths, refresh=bool(args.refresh))
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "check":
        calendar = load_market_calendar(spec["sample"]["end_date"])
        schedule = build_decision_schedule(spec, calendar)
        daily = load_daily_basic(paths, schedule)
        industry = load_industry_membership(paths)
        print(
            json.dumps(
                {
                    "study_id": spec["study_id"],
                    "decision_weeks": len(schedule),
                    "daily_basic_rows": len(daily),
                    "industry_rows": len(industry),
                    "baseline_unchanged": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = run_study(spec, paths)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
