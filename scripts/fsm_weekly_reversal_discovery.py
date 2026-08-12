"""Discover and temporally validate a weekly FSM reversal strategy.

The protocol is frozen in ``fsm_weekly_reversal_protocol_2026-08-12.json``.
Matching is outcome-blind and persisted before any H5/H20/H60 return is joined.
This is retrospective research and can never authorize live trading.

Run::

    uv run --no-sync python scripts/fsm_weekly_reversal_discovery.py
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from czsc._native.research import circular_block_bootstrap, newey_west_hac

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "fsm_weekly_reversal_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "fsm_weekly_reversal_discovery"
REPORT_PATH = SCRIPT_DIR / "FSM_WEEKLY_REVERSAL_DISCOVERY_2026-08-12.md"

FEATURES = (
    "log_circ_mv",
    "ret5",
    "ret20",
    "ret60",
    "vol20",
    "vol60",
    "log_price",
    "log_amount",
    "liq20",
    "ma_spread_pct",
    "volume_ratio",
)
HORIZONS = (5, 20, 60)
FORBIDDEN_BLIND_TOKENS = ("return", "outcome", "exit", "h5", "h20", "h60", "future")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(locator: str) -> Path:
    path = Path(locator).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _weekly_inventory_sha(directory: Path) -> str:
    """Reproduce the protocol's sorted ``sha256sum | sha256sum`` identity."""

    lines = []
    for path in sorted(directory.glob("*.parquet"), key=lambda item: item.name):
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def load_and_verify_protocol() -> dict[str, Any]:
    """Load the frozen protocol and fail closed on input drift."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "fsm_weekly_reversal_discovery_protocol_v1":
        raise RuntimeError("unexpected reversal protocol schema")
    inputs = protocol["inputs"]
    for key in ("state_cache", "price_panel", "candidate_manifest", "industry_membership", "historical_names"):
        record = inputs[key]
        path = _resolve(record["path"])
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {key} expected={record['sha256']} actual={actual}")
    weekly = inputs["weekly_daily_basic"]
    directory = _resolve(weekly["directory"])
    files = list(directory.glob("*.parquet"))
    if len(files) != int(weekly["file_count"]):
        raise RuntimeError("weekly daily_basic file-count drift")
    actual_inventory = _weekly_inventory_sha(directory)
    if actual_inventory != weekly["inventory_sha256"]:
        raise RuntimeError("weekly daily_basic inventory drift")
    return protocol


def _active_interval_rows(
    surface: pl.DataFrame,
    intervals: pl.DataFrame,
    *,
    start: str,
    end: str,
) -> pl.DataFrame:
    """Attach the unique interval active on each surface date."""

    joined = surface.join(intervals, on="symbol", how="inner")
    return joined.filter((pl.col(start) <= pl.col("dt")) & (pl.col(end).is_null() | (pl.col("dt") <= pl.col(end))))


def _st_keys(decision_dates: Sequence[pd.Timestamp], namechange_path: Path) -> pl.DataFrame:
    """Materialize historical ST/退 keys only for frozen weekly dates."""

    names = pd.read_parquet(namechange_path)
    names = names[names["name"].astype(str).str.upper().str.contains("ST|退", regex=True)].copy()
    dates = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize()
    rows: list[tuple[str, pd.Timestamp]] = []
    for row in names.itertuples(index=False):
        start = pd.to_datetime(row.start_date)
        end = pd.to_datetime(row.end_date) if pd.notna(row.end_date) else dates.max()
        for date in dates[(dates >= start) & (dates <= end)]:
            rows.append((str(row.ts_code), pd.Timestamp(date)))
    if not rows:
        return pl.DataFrame({"symbol": [], "dt": []}, schema={"symbol": pl.String, "dt": pl.Date})
    return pl.from_pandas(pd.DataFrame(rows, columns=["symbol", "dt"]).drop_duplicates()).with_columns(
        pl.col("dt").cast(pl.Date)
    )


def build_blind_surface(protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, float]]:
    """Build the decision-time-only weekly surface and development scales."""

    inputs = protocol["inputs"]
    panel_path = _resolve(inputs["price_panel"]["path"])
    state_path = _resolve(inputs["state_cache"]["path"])
    daily_dir = _resolve(inputs["weekly_daily_basic"]["directory"])

    panel = pl.read_parquet(panel_path).with_columns(pl.col("dt").cast(pl.Datetime("ns")))
    states = pl.read_parquet(state_path).with_columns(pl.col("dt").cast(pl.Datetime("ns")))
    calendar = pd.DatetimeIndex(panel.select("dt").unique().sort("dt").to_series().to_list()).normalize()
    next_session = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(calendar[:-1], calendar[1:], strict=False)}

    excluded_prefix = pl.any_horizontal(
        [pl.col("symbol").str.starts_with(prefix) for prefix in protocol["sample"]["universe"]["excluded_prefixes"]]
    )
    base = (
        panel.join(states, on=["symbol", "dt"], how="inner")
        .filter(~excluded_prefix)
        .sort(["symbol", "dt"])
        .with_columns(
            (pl.col("regime") != pl.col("regime").shift(1).over("symbol")).fill_null(True).alias("_state_change"),
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("_daily_ret"),
            pl.col("regime").shift(1).over("symbol").alias("prev_regime"),
            pl.col("dt").shift(-1).over("symbol").alias("observed_next_dt"),
            pl.col("open").shift(-1).over("symbol").alias("entry_open"),
        )
        .with_columns(pl.col("_state_change").cum_sum().over("symbol").alias("_run_id"))
        .with_columns(
            pl.when(pl.col("regime") == 1)
            .then(pl.col("regime").cum_count().over(["symbol", "_run_id"]))
            .otherwise(0)
            .alias("_down_run"),
            ((pl.col("close") / pl.col("close").shift(5).over("symbol") - 1) * 100).alias("ret5"),
            ((pl.col("close") / pl.col("close").shift(20).over("symbol") - 1) * 100).alias("ret20"),
            ((pl.col("close") / pl.col("close").shift(60).over("symbol") - 1) * 100).alias("ret60"),
            (pl.col("_daily_ret").rolling_std(20).over("symbol") * 100).alias("vol20"),
            (pl.col("_daily_ret").rolling_std(60).over("symbol") * 100).alias("vol60"),
            pl.col("amount_e").rolling_mean(20).over("symbol").alias("liq20"),
            ((pl.col("close") / pl.col("close").rolling_mean(20).over("symbol") - 1) * 100).alias("ma_spread_pct"),
        )
        .with_columns(pl.col("_down_run").shift(1).over("symbol").fill_null(0).alias("down_days"))
        .with_columns(pl.col("dt").cast(pl.Date))
    )

    daily = (
        pl.read_parquet(str(daily_dir / "*.parquet"))
        .select("ts_code", "trade_date", "circ_mv", "volume_ratio")
        .rename({"ts_code": "symbol", "trade_date": "dt"})
        .with_columns(
            pl.col("dt").str.strptime(pl.Date, "%Y%m%d", strict=True),
            (pl.col("circ_mv").cast(pl.Float64) * 10_000).alias("circ_mv_cny"),
            pl.col("volume_ratio").cast(pl.Float64),
        )
        .select("symbol", "dt", "circ_mv_cny", "volume_ratio")
    )
    surface = base.join(daily, on=["symbol", "dt"], how="inner")

    industry = (
        pl.read_parquet(_resolve(inputs["industry_membership"]["path"]))
        .select("symbol", "industry_code", "effective_from", "effective_to")
        .with_columns(
            pl.col("effective_from").str.strptime(pl.Date, "%Y%m%d", strict=True),
            pl.col("effective_to").str.strptime(pl.Date, "%Y%m%d", strict=False),
        )
    )
    surface = _active_interval_rows(surface, industry, start="effective_from", end="effective_to")
    if surface.select(pl.struct("symbol", "dt").is_duplicated().any()).item():
        raise RuntimeError("industry assignment is not unique by symbol/date")

    decision_dates = surface.select("dt").unique().sort("dt").to_series().to_list()
    st = _st_keys(decision_dates, _resolve(inputs["historical_names"]["path"]))
    surface = surface.join(st.with_columns(pl.lit(True).alias("is_st")), on=["symbol", "dt"], how="left").with_columns(
        pl.col("is_st").fill_null(False),
        pl.col("close").log().alias("log_price"),
        pl.col("amount_e").log().alias("log_amount"),
        pl.col("circ_mv_cny").log().alias("log_circ_mv"),
    )

    next_map = pl.from_pandas(
        pd.DataFrame({"dt": list(next_session), "required_entry_dt": list(next_session.values())})
    ).with_columns(pl.all().cast(pl.Date))
    surface = surface.join(next_map, on="dt", how="left").with_columns(
        (pl.col("entry_open") / pl.col("close") - 1).mul(100).alias("gap_pct"),
        pl.when(pl.any_horizontal([pl.col("symbol").str.starts_with(prefix) for prefix in ("300", "301", "302")]))
        .then(pl.lit(19.8))
        .otherwise(pl.lit(9.8))
        .alias("limit_pct"),
    )
    complete = pl.all_horizontal([pl.col(feature).is_finite() for feature in FEATURES])
    eligible = (
        (pl.col("amount_e") >= float(protocol["sample"]["universe"]["minimum_decision_amount_100m_cny"]))
        & ~pl.col("is_st")
        & (pl.col("circ_mv_cny") > 0)
        & complete
        & (pl.col("observed_next_dt").cast(pl.Date) == pl.col("required_entry_dt"))
        & pl.col("entry_open").is_finite()
        & (pl.col("entry_open") > 0)
        & (pl.col("gap_pct") < pl.col("limit_pct") - 0.3)
    )
    surface = surface.with_columns(
        eligible.alias("eligible"),
        (
            (pl.col("prev_regime") == 1)
            & pl.col("regime").is_in(protocol["signal_family"]["target_regimes"])
            & pl.col("down_days").is_between(5, 20, closed="both")
        ).alias("family_event"),
    ).filter(pl.col("eligible"))

    keep = [
        "symbol",
        "dt",
        "regime",
        "prev_regime",
        "down_days",
        "industry_code",
        "circ_mv_cny",
        "close",
        "entry_open",
        "family_event",
        *FEATURES,
    ]
    frame = surface.select(keep).to_pandas()
    frame["dt"] = pd.to_datetime(frame["dt"])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    frame["segment"] = np.where(frame["dt"].le(dev_end), "development", "validation")
    scales = frame.loc[frame["segment"].eq("development"), list(FEATURES)].std(ddof=0).to_dict()
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise RuntimeError("development feature scales are degenerate")
    return frame, calendar, {key: float(value) for key, value in scales.items()}


def cell_id(target: int, low: int, high: int) -> str:
    return f"R{target}_D{low:02d}_{high:02d}"


def build_blind_matches(
    surface: pd.DataFrame,
    protocol: Mapping[str, Any],
    scales: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixed same-date/industry/regime/size matches without outcomes."""

    controls = surface[~surface["family_event"]].copy()
    grouped = {
        key: group.copy()
        for key, group in controls.groupby(["dt", "regime", "industry_code"], sort=False, observed=True)
    }
    feature_scale = np.array([scales[name] for name in FEATURES], dtype=float)
    attempts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    minimum = int(protocol["outcome_blind_matching"]["minimum_controls"])
    maximum = int(protocol["outcome_blind_matching"]["maximum_controls"])
    ratio = float(protocol["outcome_blind_matching"]["circ_mv_caliper_ratio_inclusive"])

    for target in protocol["signal_family"]["target_regimes"]:
        for low, high in protocol["signal_family"]["downtrend_run_buckets_inclusive"]:
            cid = cell_id(int(target), int(low), int(high))
            treated = surface[
                surface["family_event"]
                & surface["regime"].eq(target)
                & surface["down_days"].between(low, high, inclusive="both")
            ]
            for row in treated.itertuples(index=False):
                trade_id = f"{row.symbol}|{row.dt:%Y-%m-%d}|{cid}"
                pool = grouped.get((row.dt, row.regime, row.industry_code))
                if pool is None:
                    eligible = pool if pool is not None else controls.iloc[:0]
                else:
                    eligible = pool[
                        pool["circ_mv_cny"].between(row.circ_mv_cny / ratio, row.circ_mv_cny * ratio, inclusive="both")
                    ].copy()
                support = len(eligible) >= minimum
                attempts.append(
                    {
                        "trade_id": trade_id,
                        "cell_id": cid,
                        "segment": row.segment,
                        "treated_symbol": row.symbol,
                        "dt": row.dt,
                        "target_regime": int(target),
                        "down_low": int(low),
                        "down_high": int(high),
                        "eligible_controls": int(len(eligible)),
                        "support_ok": bool(support),
                        "reason": "supported" if support else "fewer_than_minimum_controls",
                    }
                )
                if not support:
                    continue
                treated_values = np.array([getattr(row, feature) for feature in FEATURES], dtype=float)
                control_values = eligible.loc[:, FEATURES].to_numpy(dtype=float)
                distance = np.sqrt(np.square((control_values - treated_values) / feature_scale).sum(axis=1))
                selected = (
                    eligible.assign(distance=distance)
                    .sort_values(["distance", "symbol"], kind="mergesort")
                    .head(maximum)
                )
                weight = 1.0 / len(selected)
                for rank, control in enumerate(selected.itertuples(index=False), start=1):
                    record: dict[str, Any] = {
                        "trade_id": trade_id,
                        "cell_id": cid,
                        "segment": row.segment,
                        "treated_symbol": row.symbol,
                        "control_symbol": control.symbol,
                        "dt": row.dt,
                        "target_regime": int(target),
                        "rank": rank,
                        "control_weight": weight,
                        "distance": float(control.distance),
                        "treated_circ_mv_cny": float(row.circ_mv_cny),
                        "control_circ_mv_cny": float(control.circ_mv_cny),
                    }
                    for feature in FEATURES:
                        record[f"treated_{feature}"] = float(getattr(row, feature))
                        record[f"control_{feature}"] = float(getattr(control, feature))
                    pairs.append(record)

    attempt_frame = pd.DataFrame(attempts).sort_values(["dt", "cell_id", "treated_symbol"], kind="mergesort")
    pair_frame = pd.DataFrame(pairs).sort_values(["dt", "cell_id", "treated_symbol", "rank"], kind="mergesort")
    assert_blind(attempt_frame, "match attempts")
    assert_blind(pair_frame, "matched pairs")
    return pair_frame.reset_index(drop=True), attempt_frame.reset_index(drop=True)


def assert_blind(frame: pd.DataFrame, label: str) -> None:
    forbidden = [name for name in frame if any(token in name.lower() for token in FORBIDDEN_BLIND_TOKENS)]
    if forbidden:
        raise RuntimeError(f"{label} contains outcome-like columns: {forbidden}")


def balance_summary(pairs: pd.DataFrame, attempts: pd.DataFrame) -> dict[str, Any]:
    """Compute trade-level SMD and support coverage before outcome access."""

    result: dict[str, Any] = {}
    for (cid, segment), attempt_group in attempts.groupby(["cell_id", "segment"], observed=True):
        scoped = pairs[(pairs["cell_id"] == cid) & (pairs["segment"] == segment)]
        supported = int(attempt_group["support_ok"].sum())
        coverage = supported / len(attempt_group) if len(attempt_group) else 0.0
        stats: dict[str, float] = {}
        if not scoped.empty:
            trade_rows = scoped.drop_duplicates("trade_id", keep="first")
            for feature in FEATURES:
                treated = trade_rows[f"treated_{feature}"].to_numpy(float)
                control = (
                    scoped.assign(_weighted=scoped[f"control_{feature}"] * scoped["control_weight"])
                    .groupby("trade_id", sort=False)["_weighted"]
                    .sum()
                    .reindex(trade_rows["trade_id"])
                    .to_numpy(float)
                )
                pooled = math.sqrt((treated.var(ddof=0) + control.var(ddof=0)) / 2)
                stats[feature] = float((treated.mean() - control.mean()) / pooled) if pooled > 0 else 0.0
        result[f"{cid}|{segment}"] = {
            "n_attempted": int(len(attempt_group)),
            "n_supported": supported,
            "coverage": float(coverage),
            "smd": stats,
            "max_abs_smd": float(max((abs(value) for value in stats.values()), default=float("inf"))),
        }
    return result


def _endpoint_map(calendar: pd.DatetimeIndex, horizon: int) -> dict[pd.Timestamp, pd.Timestamp]:
    return {
        pd.Timestamp(calendar[index]): pd.Timestamp(calendar[index + horizon])
        for index in range(len(calendar) - horizon)
    }


def add_outcomes(
    pairs: pd.DataFrame,
    attempts: pd.DataFrame,
    surface: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    panel_path: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join fixed common-session outcomes after blind pairs are frozen."""

    supported = attempts[attempts["support_ok"]].copy()
    identities = [supported[["treated_symbol", "dt"]].rename(columns={"treated_symbol": "symbol"})]
    identities.append(pairs[["control_symbol", "dt"]].rename(columns={"control_symbol": "symbol"}))
    keys = pd.concat(identities, ignore_index=True).drop_duplicates(["symbol", "dt"])
    base = surface.set_index(["symbol", "dt"])[["entry_open"]]
    keys = keys.join(base, on=["symbol", "dt"], how="left")
    for horizon in HORIZONS:
        endpoint = _endpoint_map(calendar, horizon)
        keys[f"endpoint_{horizon}"] = keys["dt"].map(endpoint)
    requests = pl.from_pandas(keys).with_columns(
        [pl.col(f"endpoint_{horizon}").cast(pl.Datetime("ns")) for horizon in HORIZONS]
    )
    prices = (
        pl.read_parquet(panel_path, columns=["symbol", "dt", "close"])
        .with_columns(pl.col("dt").cast(pl.Datetime("ns")).alias("price_dt"))
        .drop("dt")
        .sort(["symbol", "price_dt"])
    )
    for horizon in HORIZONS:
        requests = (
            requests.sort(["symbol", f"endpoint_{horizon}"])
            .join_asof(
                prices,
                left_on=f"endpoint_{horizon}",
                right_on="price_dt",
                by="symbol",
                strategy="backward",
            )
            .rename({"close": f"endpoint_close_{horizon}", "price_dt": f"observed_close_dt_{horizon}"})
        )
    outcomes = requests.to_pandas()
    buy = float(protocol["execution"]["buy_cost"])
    sell = float(protocol["execution"]["sell_cost"])
    for horizon in HORIZONS:
        outcomes[f"net_h{horizon}"] = (
            outcomes[f"endpoint_close_{horizon}"] / outcomes["entry_open"] * (1 - sell) / (1 + buy) - 1
        )

    treated_outcomes = outcomes.rename(columns={"symbol": "treated_symbol"})
    control_outcomes = outcomes.rename(columns={"symbol": "control_symbol"})
    pair_outcomes = pairs.merge(treated_outcomes, on=["treated_symbol", "dt"], how="left").merge(
        control_outcomes, on=["control_symbol", "dt"], how="left", suffixes=("_treated", "_control")
    )
    trades: list[dict[str, Any]] = []
    for trade_id, group in pair_outcomes.groupby("trade_id", sort=False):
        first = group.iloc[0]
        record = {
            "trade_id": trade_id,
            "cell_id": first["cell_id"],
            "segment": first["segment"],
            "symbol": first["treated_symbol"],
            "dt": first["dt"],
            "n_controls": int(len(group)),
        }
        complete = True
        for horizon in HORIZONS:
            treated_value = float(first[f"net_h{horizon}_treated"])
            control_values = group[f"net_h{horizon}_control"].to_numpy(float)
            weights = group["control_weight"].to_numpy(float)
            if not np.isfinite(treated_value) or not np.isfinite(control_values).all():
                complete = False
                break
            control_value = float(np.average(control_values, weights=weights))
            record[f"treated_net_h{horizon}"] = treated_value
            record[f"control_net_h{horizon}"] = control_value
            record[f"att_net_h{horizon}"] = treated_value - control_value
        if complete:
            trades.append(record)
    return pd.DataFrame(trades), pair_outcomes


def holm_adjust(values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment for a fixed named family."""

    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        result[key] = running
    return result


def infer_cell(
    trades: pd.DataFrame,
    *,
    half_boundary: pd.Timestamp,
    seed: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Infer the primary weekly H20 ATT for one cell and segment."""

    weekly = trades.groupby("dt", sort=True)["att_net_h20"].mean().sort_index()
    lag = int(protocol["statistics"]["newey_west_lag_weeks"])
    early = trades.loc[trades["dt"].lt(half_boundary), "att_net_h20"]
    late = trades.loc[trades["dt"].ge(half_boundary), "att_net_h20"]
    common = {
        "n_trades": int(len(trades)),
        "n_weeks": int(len(weekly)),
        "mean_att": float(trades["att_net_h20"].mean()) if len(trades) else float("nan"),
        "median_att": float(trades["att_net_h20"].median()) if len(trades) else float("nan"),
        "positive_trade_rate": float(trades["att_net_h20"].gt(0).mean()) if len(trades) else float("nan"),
        "half_means": {"early": float(early.mean()), "late": float(late.mean())},
        "year_means": {
            str(year): float(group["att_net_h20"].mean())
            for year, group in trades.groupby(trades["dt"].dt.year, observed=True)
        },
        "diagnostic_mean_att": {
            str(horizon): float(trades[f"att_net_h{horizon}"].mean())
            for horizon in HORIZONS
            if horizon != 20 and len(trades)
        },
        "absolute_treated_net_h20": float(trades["treated_net_h20"].mean()) if len(trades) else float("nan"),
    }
    if len(weekly) <= lag:
        return {
            **common,
            "status": "INSUFFICIENT_WEEKS_FOR_HAC",
            "one_sided_p": 1.0,
            "hac": {"t_stat": float("nan"), "n": int(len(weekly)), "lag": lag},
            "bootstrap": {"q05": float("nan"), "n_draws": 0},
        }
    hac = newey_west_hac(weekly.tolist(), lag=lag)
    bootstrap = circular_block_bootstrap(
        weekly.tolist(),
        block_size=int(protocol["statistics"]["circular_block_weeks"]),
        n_draws=int(protocol["statistics"]["bootstrap_draws"]),
        seed=seed,
    )
    return {
        **common,
        "status": "ESTIMATED",
        "one_sided_p": float(norm.sf(hac["t_stat"])),
        "hac": hac,
        "bootstrap": bootstrap,
    }


def _development_pass(
    inference: Mapping[str, Any],
    balance: Mapping[str, Any],
    adjusted_p: float,
    protocol: Mapping[str, Any],
) -> bool:
    stats = protocol["statistics"]
    return bool(
        balance["coverage"] >= protocol["outcome_blind_matching"]["coverage_threshold"]
        and balance["max_abs_smd"] <= protocol["outcome_blind_matching"]["maximum_absolute_smd"]
        and inference["n_trades"] >= stats["minimum_development_supported_trades"]
        and inference["mean_att"] >= stats["minimum_effect_net_att"]
        and adjusted_p < 0.05
        and inference["bootstrap"]["q05"] > 0
        and min(inference["half_means"].values()) > 0
    )


def _validation_pass(
    inference: Mapping[str, Any],
    balance: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> bool:
    stats = protocol["statistics"]
    return bool(
        balance["coverage"] >= protocol["outcome_blind_matching"]["coverage_threshold"]
        and balance["max_abs_smd"] <= protocol["outcome_blind_matching"]["maximum_absolute_smd"]
        and inference["n_trades"] >= stats["minimum_validation_supported_trades"]
        and inference["mean_att"] >= stats["minimum_effect_net_att"]
        and inference["one_sided_p"] < 0.05
        and inference["bootstrap"]["q05"] > 0
        and min(inference["half_means"].values()) > 0
    )


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def render_report(audit: Mapping[str, Any]) -> str:
    """Render the compact research handoff."""

    selected = audit["selection"].get("selected_cell")
    verdict = audit["verdict"]["status"]
    lines = [
        "# FSM 周度反转策略发现报告（2026-08-12）",
        "",
        f"- 最终判定：`{verdict}`",
        f"- 开发段选中单元：`{selected or 'NONE'}`",
        "- 研究性质：历史发现 + 时间验证，不是独立 OOS",
        "- 实盘授权：`false`",
        "",
        "## 机制与口径",
        "",
        "研究对象是状态 1（下跌走势）持续后，在周度截面新鲜切换到状态 2/3/4 的反转。",
        "每笔交易在决策日收盘识别、下一交易日开盘进入、固定 H20 收盘退出；买卖成本为 15/25bps。",
        "控制组严格同日、同当前状态、同申万一级行业，并使用当日流通市值 2 倍卡尺和 11 项因果特征最近邻。",
        "",
        "## 开发段 9 单元",
        "",
        "| 单元 | 支持交易 | 覆盖 | 最大|SMD| | H20净ATT | HAC t | Holm p | q05 | 两半均值 | 通过 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cid, row in audit["development"].items():
        inf = row["inference"]
        bal = row["balance"]
        half = inf.get("half_means", {})
        lines.append(
            f"| {cid} | {inf['n_trades']} | {bal['coverage']:.1%} | {bal['max_abs_smd']:.3f} | "
            f"{inf['mean_att']:.2%} | {inf['hac']['t_stat']:.2f} | {row['holm_one_sided_p']:.3f} | "
            f"{inf['bootstrap']['q05']:.2%} | {half.get('early', float('nan')):.2%}/{half.get('late', float('nan')):.2%} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## 时间验证", ""])
    validation = audit.get("validation")
    if not validation:
        lines.append("开发段没有单元通过冻结门槛，因此没有读取验证段结果。")
    else:
        inf = validation["inference"]
        bal = validation["balance"]
        lines.extend(
            [
                f"锁定 `{selected}`：支持交易 {inf['n_trades']}，覆盖 {bal['coverage']:.1%}，最大 |SMD| {bal['max_abs_smd']:.3f}。",
                f"H20 净 ATT {inf['mean_att']:.2%}，HAC t={inf['hac']['t_stat']:.2f}，单侧 p={inf['one_sided_p']:.3f}，bootstrap q05={inf['bootstrap']['q05']:.2%}。",
                f"验证两半均值：{inf['half_means']['early']:.2%} / {inf['half_means']['late']:.2%}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 历史样本此前已被其他 CZSC 研究看过；即使通过，也只能建立待前瞻验证候选。",
            "- 周度 daily_basic 提供决策日精确流通市值；匹配表在结果读取前落盘并哈希。",
            "- 退市终值与真实集合竞价不可得，长停牌使用 LOCF；这些限制禁止实盘授权。",
            "- 若失败，不得在同一历史样本改目标状态、持续天数、H20 或阈值救回。",
            "",
            "## 复现",
            "",
            "```bash",
            "uv run --no-sync python scripts/fsm_weekly_reversal_discovery.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    """Execute the complete fail-closed study."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    surface, calendar, scales = build_blind_surface(protocol)
    pairs, attempts = build_blind_matches(surface, protocol, scales)
    pairs_path = OUTPUT_DIR / "matched_pairs_blind.parquet"
    attempts_path = OUTPUT_DIR / "match_attempts.parquet"
    pairs.to_parquet(pairs_path, index=False)
    attempts.to_parquet(attempts_path, index=False)
    balance = balance_summary(pairs, attempts)
    balance_path = OUTPUT_DIR / "balance.json"
    balance_path.write_text(json.dumps(_clean_json(balance), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Development outcomes are accessed for the fixed nine-cell family only after blind artifacts exist.
    development_pairs = pairs[pairs["segment"].eq("development")].copy()
    development_attempts = attempts[attempts["segment"].eq("development")].copy()
    dev_trades, _ = add_outcomes(
        development_pairs,
        development_attempts,
        surface,
        calendar,
        _resolve(protocol["inputs"]["price_panel"]["path"]),
        protocol,
    )
    seed_root = int(protocol["statistics"]["bootstrap_seed_root"])
    development: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for index, cid in enumerate(sorted(attempts["cell_id"].unique())):
        trades = dev_trades[dev_trades["cell_id"].eq(cid)].copy()
        inference = infer_cell(
            trades,
            half_boundary=pd.Timestamp("2023-01-01"),
            seed=seed_root + index,
            protocol=protocol,
        )
        development[cid] = {"inference": inference, "balance": balance[f"{cid}|development"]}
        raw_p[cid] = float(inference["one_sided_p"])
    adjusted = holm_adjust(raw_p)
    passing = []
    for cid, row in development.items():
        row["holm_one_sided_p"] = adjusted[cid]
        row["pass"] = _development_pass(row["inference"], row["balance"], adjusted[cid], protocol)
        if row["pass"]:
            passing.append(cid)
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda cid: (
                -development[cid]["inference"]["bootstrap"]["q05"],
                -development[cid]["inference"]["mean_att"],
                cid,
            ),
        )[0]

    validation = None
    all_trades = dev_trades.copy()
    if selected is not None:
        validation_pairs = pairs[(pairs["segment"].eq("validation")) & (pairs["cell_id"].eq(selected))].copy()
        validation_attempts = attempts[
            (attempts["segment"].eq("validation")) & (attempts["cell_id"].eq(selected))
        ].copy()
        val_trades, _ = add_outcomes(
            validation_pairs,
            validation_attempts,
            surface,
            calendar,
            _resolve(protocol["inputs"]["price_panel"]["path"]),
            protocol,
        )
        inference = infer_cell(
            val_trades,
            half_boundary=pd.Timestamp("2025-01-01"),
            seed=seed_root + 100,
            protocol=protocol,
        )
        validation = {
            "inference": inference,
            "balance": balance[f"{selected}|validation"],
        }
        validation["pass"] = _validation_pass(inference, validation["balance"], protocol)
        all_trades = pd.concat([all_trades, val_trades], ignore_index=True)

    if selected is None:
        status = "NO_DEVELOPMENT_CELL_PASSED"
    elif validation and validation["pass"]:
        status = "RETROSPECTIVE_CANDIDATE_FORWARD_TEST_REQUIRED"
    else:
        status = "HYPOTHESIS_FALSIFIED_RETROSPECTIVE"
    trade_path = OUTPUT_DIR / "trade_att.parquet"
    all_trades.to_parquet(trade_path, index=False)
    audit = {
        "schema": "fsm_weekly_reversal_discovery_audit_v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "input_identity": {
            "surface_rows": int(len(surface)),
            "decision_dates": int(surface["dt"].nunique()),
            "blind_pairs_sha256": sha256_file(pairs_path),
            "match_attempts_sha256": sha256_file(attempts_path),
            "balance_sha256": sha256_file(balance_path),
        },
        "development_scales": scales,
        "development": development,
        "selection": {"passing_cells": passing, "selected_cell": selected},
        "validation": validation,
        "verdict": {
            "status": status,
            "confirmation_chain": "NOT_STARTED",
            "live_trading_authorized": False,
        },
        "outputs": {
            "trade_att": {"path": str(trade_path), "sha256": sha256_file(trade_path), "rows": int(len(all_trades))},
            "blind_pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path), "rows": int(len(pairs))},
            "match_attempts": {
                "path": str(attempts_path),
                "sha256": sha256_file(attempts_path),
                "rows": int(len(attempts)),
            },
        },
        "limitations": [
            "The historical sample was visible to earlier CZSC research and is not independent OOS.",
            "Delisting terminal values and auction microstructure are unavailable.",
            "LOCF handles suspension but can overstate the value of a delisted security.",
            "A passing result only justifies a new frozen prospective protocol.",
        ],
    }
    audit_path = OUTPUT_DIR / "audit.json"
    audit_path.write_text(json.dumps(_clean_json(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(audit), encoding="utf-8")
    return audit


def main() -> None:
    audit = run()
    print(json.dumps(audit["selection"], ensure_ascii=False, indent=2))
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
