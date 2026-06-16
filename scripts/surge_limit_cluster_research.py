"""Limit-up cluster detector research for the surge-regime strategy.

This is the first-stage research for the next direction recorded in
SURGE_REGIME_NEXT_DIRECTIONS_2026-06-12.md: a complementary detector for
limit-up-dense main-up moves that the current FSM can cut into 5 -> 10 -> 5
fragments.

Pre-declared scope:
- No existing surge-regime thresholds are changed.
- The target is the existing price label ``t1_px30`` from surge_factor_dump:
  max close return in the next 40 bars >= 30%.
- The primary universe is broad structure-ready bars:
  regime in {4, 5, 6}, not already inside an FSM surge event, with a non-null
  t1_px30 label.
- Detectors are coarse a-priori limit-up cluster rules, deduped by symbol with
  a 20-bar cooldown. This script is a label/coverage gate only; it does not
  claim tradable alpha.

Pass criterion for "worth mirror-backtesting" (applied on OOS years >= 2024):
1. deduped event count >= 200;
2. t1_px30 rate lift >= 1.5x versus the primary universe;
3. at least 50 t1_px30-positive OOS events are not covered by the current
   delay5 baseline (anticipate + market gate + hard filters).

If no detector passes, the limit-up cluster direction should stop or be
redesigned before any slower portfolio mirror work.

    uv run --no-sync python scripts/surge_limit_cluster_research.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_limit_cluster"
FEATURE_CACHE = OUTPUT_DIR / "limit_features.parquet"
REPORT_PATH = Path(__file__).resolve().parent / "SURGE_REGIME_LIMIT_CLUSTER_RESEARCH_2026-06-16.md"
DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

FIRST_TEST_YEAR = 2024
COOLDOWN_BARS = 20
PASS_MIN_OOS_EVENTS = 200
PASS_MIN_LIFT = 1.5
PASS_MIN_INCREMENTAL_WINNERS = 50

PRIMARY_REGIMES = {4, 5, 6}
MIN_BARS = 500
LIMIT_PCT_MAIN = 9.8
LIMIT_PCT_CHINEXT = 19.8
CHINEXT_PREFIX = ("300", "301", "302")
EXCLUDE_PREFIX = ("688", "920", "83", "43")

SURGE_GATE_VOL_RATIO = 1.2
SURGE_GATE_MA_SPREAD = 3.0
SURGE_GATE_RET20 = 8.0
MIN_AMOUNT_E = 1.0
STOP_MIN_PCT = 8.0
STOP_MAX_PCT = 20.0
GAP_LIMIT_MARGIN = 0.3


def round_float(value: float | int | None, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def limit_pct_for(code: str) -> float:
    return LIMIT_PCT_CHINEXT if str(code).startswith(CHINEXT_PREFIX) else LIMIT_PCT_MAIN


def load_stock(parquet_path: str | Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None
    if len(df) < MIN_BARS:
        return None
    code = str(df["ts_code"].iloc[0])
    if code.startswith(EXCLUDE_PREFIX):
        return None
    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)


def _is_st_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "退" in text


def load_st_intervals() -> dict[str, list[tuple[str, str, bool]]]:
    if not NAMECHANGE_PATH.exists():
        return {}
    nc = pd.read_parquet(NAMECHANGE_PATH)
    nc["end_date"] = nc["end_date"].fillna("99999999")
    out: dict[str, list[tuple[str, str, bool]]] = {}
    for code, group in nc.groupby("ts_code"):
        group = group.sort_values("start_date")
        out[str(code)] = list(
            zip(group["start_date"], group["end_date"], [_is_st_name(x) for x in group["name"]], strict=False)
        )
    return out


def is_st_on(intervals: dict[str, list[tuple[str, str, bool]]], symbol: str, date: pd.Timestamp) -> bool:
    rows = intervals.get(str(symbol))
    if not rows:
        return False
    d = pd.Timestamp(date).strftime("%Y%m%d")
    best = None
    for start, _end, is_st in rows:
        if start <= d:
            best = is_st
    return bool(best) if best is not None else False


def _rolling_count(values: pd.Series, window: int) -> pd.Series:
    return values.astype(float).rolling(window, min_periods=1).sum()


def _max_streak_in_window(is_limit_up: pd.Series, window: int) -> pd.Series:
    streak = []
    cur = 0
    for flag in is_limit_up.astype(bool):
        cur = cur + 1 if flag else 0
        streak.append(cur)
    return pd.Series(streak, index=is_limit_up.index).rolling(window, min_periods=1).max()


def _limit_features_one(parquet_path: str) -> pd.DataFrame | None:
    df = load_stock(parquet_path)
    if df is None:
        return None

    symbol = str(df["symbol"].iloc[0])
    limit_pct = limit_pct_for(symbol)
    close = df["close"].astype(float)
    amount = df["amount"].astype(float) if "amount" in df.columns else pd.Series(np.nan, index=df.index)
    pct_chg = df["pct_chg"].astype(float) if "pct_chg" in df.columns else close.pct_change().mul(100)
    bar_no = pd.Series(np.arange(len(df), dtype=np.int32), index=df.index)

    is_limit_up = pct_chg >= limit_pct - 0.3
    is_near_limit_up = pct_chg >= limit_pct - 1.0
    high20 = close.rolling(20, min_periods=1).max()
    high60 = close.rolling(60, min_periods=1).max()
    amount20 = amount.rolling(20, min_periods=1).mean()

    last_lu_bar = pd.Series(np.where(is_limit_up, bar_no, np.nan), index=df.index).ffill()
    days_since_limit_up = bar_no - last_lu_bar
    days_since_limit_up = days_since_limit_up.where(last_lu_bar.notna(), 999)

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "dt": df["dt"],
            "bar_no": bar_no.astype(np.int32),
            "amount_e": amount / 1e5,
            "limit_pct": limit_pct,
            "limit_up": is_limit_up.astype(np.int8),
            "near_limit_up": is_near_limit_up.astype(np.int8),
            "limit_up_5": _rolling_count(is_limit_up, 5),
            "limit_up_10": _rolling_count(is_limit_up, 10),
            "limit_up_20": _rolling_count(is_limit_up, 20),
            "near_limit_up_20": _rolling_count(is_near_limit_up, 20),
            "max_limit_streak_20": _max_streak_in_window(is_limit_up, 20),
            "days_since_limit_up": days_since_limit_up,
            "drawdown_20": (close / high20 - 1.0) * 100,
            "drawdown_60": (close / high60 - 1.0) * 100,
            "amount_ratio20": amount / amount20,
            "ret10": close.pct_change(10) * 100,
            "ret20_raw": close.pct_change(20) * 100,
        }
    )
    float_cols = [
        "amount_e",
        "limit_pct",
        "limit_up_5",
        "limit_up_10",
        "limit_up_20",
        "near_limit_up_20",
        "max_limit_streak_20",
        "days_since_limit_up",
        "drawdown_20",
        "drawdown_60",
        "amount_ratio20",
        "ret10",
        "ret20_raw",
    ]
    out[float_cols] = out[float_cols].astype("float32")
    return out


def load_limit_features(force: bool = False) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not force:
        print(f"[features] cache hit: {FEATURE_CACHE}")
        return pd.read_parquet(FEATURE_CACHE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    workers = min(mp.cpu_count(), 8)
    print(f"[features] building limit-up features for {len(files)} files with {workers} workers")
    t0 = time.time()
    parts = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_limit_features_one, files, chunksize=50), 1):
            if res is not None:
                parts.append(res)
            if i % 1000 == 0 or i == len(files):
                rows = sum(len(x) for x in parts)
                print(f"  [{i}/{len(files)}] rows={rows} elapsed={time.time() - t0:.0f}s")
    features = pd.concat(parts, ignore_index=True)
    features.to_parquet(FEATURE_CACHE, index=False)
    print(f"[features] wrote {len(features)} rows -> {FEATURE_CACHE} ({time.time() - t0:.0f}s)")
    return features


def load_primary_universe() -> tuple[pd.DataFrame, dict]:
    bars = pd.read_parquet(OUTPUT_DIR.parent / "surge_factors" / "bars.parquet")
    bars = bars[bars["regime"].isin(PRIMARY_REGIMES) & (bars["in_surge"] == 0) & bars["t1_px30"].notna()].copy()

    features = load_limit_features()
    cols = [
        "symbol",
        "dt",
        "bar_no",
        "amount_e",
        "limit_up_5",
        "limit_up_10",
        "limit_up_20",
        "near_limit_up_20",
        "max_limit_streak_20",
        "days_since_limit_up",
        "drawdown_20",
        "drawdown_60",
        "amount_ratio20",
        "ret10",
        "ret20_raw",
    ]
    df = bars.merge(features[cols], on=["symbol", "dt"], how="left")

    market_path = OUTPUT_DIR.parent / "surge_market_state_filter" / "market_state.parquet"
    if market_path.exists():
        market = pd.read_parquet(market_path, columns=["dt", "high20_ratio", "ew_index_above_ma20"])
    else:
        raise FileNotFoundError(f"Missing market-state file: {market_path}")
    df = df.merge(market, on="dt", how="left")
    df["market_gate"] = (df["high20_ratio"] > 0.12) & (df["ew_index_above_ma20"] > 0)

    delay5 = load_delay5_universe()
    delay5_mask = delay5_base_mask(delay5)
    delay5_keys = (
        delay5.loc[delay5_mask, ["symbol", "dec_dt"]]
        .drop_duplicates()
        .rename(columns={"dec_dt": "dt"})
        .assign(delay5_base=1)
    )
    df = df.merge(delay5_keys, on=["symbol", "dt"], how="left")
    df["delay5_base"] = df["delay5_base"].fillna(0).astype(np.int8)

    meta = {
        "primary_rows": int(len(df)),
        "primary_symbols": int(df["symbol"].nunique()),
        "delay5_base_rows": int(len(delay5_keys)),
    }
    return df.reset_index(drop=True), meta


def load_delay5_universe() -> pd.DataFrame:
    cand = pd.read_parquet(OUTPUT_DIR.parent / "surge_candidates" / "candidates.parquet")
    df = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5)].copy()
    market = pd.read_parquet(
        OUTPUT_DIR.parent / "surge_market_state_filter" / "market_state.parquet",
        columns=["dt", "high20_ratio", "ew_index_above_ma20"],
    )
    df = df.merge(market, left_on="dec_dt", right_on="dt", how="left").drop(columns=["dt"])
    st_intervals = load_st_intervals()
    if st_intervals:
        df["is_st"] = df.apply(lambda r: is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
    else:
        df["is_st"] = False
    df["gap_ok"] = (df["gap_pct"] < df["limit_pct"] - GAP_LIMIT_MARGIN).fillna(False)
    return df.reset_index(drop=True)


def delay5_base_mask(df: pd.DataFrame) -> pd.Series:
    mask = (
        (~df["is_st"])
        & df["gap_ok"]
        & (df["sig_vol_ratio"] >= SURGE_GATE_VOL_RATIO)
        & (df["sig_ma_spread_pct"] >= SURGE_GATE_MA_SPREAD)
        & (df["sig_ret20"] >= SURGE_GATE_RET20)
        & (df["sig_above_zg"] == 1)
        & (df["amount_e"] >= MIN_AMOUNT_E)
        & df["sl_pct"].between(STOP_MIN_PCT, STOP_MAX_PCT)
        & (df["high20_ratio"] > 0.12)
        & (df["ew_index_above_ma20"] > 0)
    )
    return mask.fillna(False)


def detector_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    liquid = (df["amount_e"] >= 1.0).fillna(False)
    stop_broad = df["sl_dist"].between(5, 35).fillna(False)
    not_extreme_drawdown = (df["drawdown_20"] >= -25).fillna(False)

    return {
        # Direct dense-board breakout: enough limit-up pressure in the last month
        # while still structure-ready and not too far from the 20-day high.
        "lc_breakout_v1": (
            liquid
            & stop_broad
            & (df["limit_up_20"] >= 2)
            & (df["ret20"] >= 15)
            & (df["ma_spread_pct"] >= 0)
            & (df["drawdown_20"] >= -12)
        ),
        # Pullback after clustered boards: avoids same-day board chasing and
        # asks that the stock is still near the recent high.
        "lc_pullback_v1": (
            liquid
            & stop_broad
            & (df["limit_up_20"] >= 2)
            & df["days_since_limit_up"].between(1, 8)
            & df["drawdown_20"].between(-18, -2)
            & (df["ma_spread_pct"] >= -3)
        ),
        # Very dense short-window boards, with a loose drawdown guard.
        "lc_dense10_v1": (
            liquid
            & stop_broad
            & (df["limit_up_10"] >= 2)
            & (df["ret10"] >= 8)
            & not_extreme_drawdown
        ),
        # One actual board plus several near-board days: captures high-volatility
        # routes that fail the strict two-board count.
        "lc_near_board_v1": (
            liquid
            & stop_broad
            & (df["limit_up_20"] >= 1)
            & (df["near_limit_up_20"] >= 3)
            & (df["ret20"] >= 10)
            & (df["dist_hi250"] >= -25)
        ),
    }


def diagnostic_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Post-hoc diagnostics after v1 missed 000636.

    These rules are intentionally excluded from the pass verdict. They answer
    whether a longer post-board memory can recover the named failure case
    without becoming a broad no-edge filter.
    """
    liquid = (df["amount_e"] >= 1.0).fillna(False)
    return {
        "diag_post_board_base_loose": (
            liquid
            & df["days_since_limit_up"].between(20, 60)
            & df["drawdown_60"].between(-30, -2)
            & df["ma_spread_pct"].between(-8, 4)
            & df["sl_dist"].between(0, 15)
        ),
        "diag_post_board_base_tight": (
            liquid
            & df["days_since_limit_up"].between(20, 60)
            & df["drawdown_60"].between(-20, -2)
            & df["ma_spread_pct"].between(-6, 3)
            & df["sl_dist"].between(0, 10)
            & df["regime"].isin([4, 5])
        ),
        "diag_post_board_reclaim": (
            liquid
            & df["days_since_limit_up"].between(20, 80)
            & df["drawdown_60"].between(-25, -1)
            & (df["ret5"] > 0)
            & df["ma_spread_pct"].between(-6, 5)
            & df["sl_dist"].between(0, 15)
        ),
    }


def dedupe_events(df: pd.DataFrame, mask: pd.Series, cooldown: int = COOLDOWN_BARS) -> pd.DataFrame:
    raw = df[mask.fillna(False)].sort_values(["symbol", "bar_no", "dt"]).copy()
    keep = []
    for _symbol, group in raw.groupby("symbol", sort=False):
        last_bar = -10**9
        for idx, bar_no in zip(group.index, group["bar_no"], strict=False):
            if pd.isna(bar_no):
                continue
            if int(bar_no) - last_bar > cooldown:
                keep.append(idx)
                last_bar = int(bar_no)
    return df.loc[keep].sort_values(["dt", "symbol"]).reset_index(drop=True)


def base_rates(df: pd.DataFrame) -> dict[str, float]:
    return {
        "ALL": float(df["t1_px30"].mean()),
        "IS": float(df.loc[df["year"] < FIRST_TEST_YEAR, "t1_px30"].mean()),
        "OOS": float(df.loc[df["year"] >= FIRST_TEST_YEAR, "t1_px30"].mean()),
    }


def event_stats(events: pd.DataFrame, rates: dict[str, float]) -> dict:
    rows = {}
    groups = {
        "ALL": events,
        "IS": events[events["year"] < FIRST_TEST_YEAR],
        "OOS": events[events["year"] >= FIRST_TEST_YEAR],
    }
    for tag, group in groups.items():
        if len(group) == 0:
            rows[tag] = {"n": 0}
            continue
        px_rate = float(group["t1_px30"].mean())
        rows[tag] = {
            "n": int(len(group)),
            "symbols": int(group["symbol"].nunique()),
            "t1_px30_rate_pct": round_float(px_rate * 100, 1),
            "lift": round_float(px_rate / rates[tag], 2) if rates[tag] > 0 else None,
            "fwd40_mean_pct": round_float(group["fwd40_max"].mean()),
            "fwd40_median_pct": round_float(group["fwd40_max"].median()),
            "fwd20_mean_pct": round_float(group["fwd20"].mean()),
            "fwd20_median_pct": round_float(group["fwd20"].median()),
            "delay5_cover_pct": round_float(group["delay5_base"].mean() * 100, 1),
            "onset_cover_pct": round_float(group["is_anticipate_onset"].mean() * 100, 1),
            "market_gate_pct": round_float(group["market_gate"].mean() * 100, 1),
            "incremental_t1_count": int(((group["t1_px30"] == 1) & (group["delay5_base"] == 0)).sum()),
        }
    return rows


def pass_verdict(stats: dict) -> dict:
    oos = stats.get("OOS", {})
    passed = (
        (oos.get("n") or 0) >= PASS_MIN_OOS_EVENTS
        and (oos.get("lift") or 0) >= PASS_MIN_LIFT
        and (oos.get("incremental_t1_count") or 0) >= PASS_MIN_INCREMENTAL_WINNERS
    )
    failed = []
    if (oos.get("n") or 0) < PASS_MIN_OOS_EVENTS:
        failed.append("oos_n")
    if (oos.get("lift") or 0) < PASS_MIN_LIFT:
        failed.append("lift")
    if (oos.get("incremental_t1_count") or 0) < PASS_MIN_INCREMENTAL_WINNERS:
        failed.append("incremental_t1")
    return {"passed": bool(passed), "failed": failed}


def yearly_stats(events: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in events.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "n": int(len(group)),
                "t1_px30_rate_pct": round_float(group["t1_px30"].mean() * 100, 1),
                "delay5_cover_pct": round_float(group["delay5_base"].mean() * 100, 1),
                "fwd40_mean_pct": round_float(group["fwd40_max"].mean()),
            }
        )
    return rows


def case_000636(events: pd.DataFrame) -> dict:
    hit = events[events["symbol"].eq("000636.SZ")].copy()
    cols = ["dt", "regime", "t1_px30", "fwd40_max", "days_since_limit_up", "drawdown_60", "ma_spread_pct", "sl_dist"]
    return {
        "count": int(len(hit)),
        "latest": [
            {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else round_float(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in hit[cols].tail(8).to_dict("records")
        ],
    }


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(summary: dict) -> None:
    lines = [
        "# Surge Regime limit-up cluster detector research (2026-06-16)",
        "",
        "First-stage label/coverage test for a complementary detector targeting limit-up-dense main-up moves.",
        "No existing strategy threshold is changed; this is not a trading backtest.",
        "",
        "## Pre-declared pass criterion",
        "",
        f"- OOS deduped events >= {PASS_MIN_OOS_EVENTS}",
        f"- OOS t1_px30 lift >= {PASS_MIN_LIFT}x versus the primary universe",
        f"- OOS t1-positive events not covered by current delay5 baseline >= {PASS_MIN_INCREMENTAL_WINNERS}",
        "",
        "## Universe",
        "",
        "Primary universe: regime in {4,5,6}, not already in an FSM surge event, non-null t1_px30.",
        "",
        "```json",
        json.dumps(summary["meta"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "Base t1_px30 rates:",
        "",
        "```json",
        json.dumps(summary["base_rates_pct"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Detector Verdicts",
    ]

    verdict_rows = []
    for name, item in summary["detectors"].items():
        oos = item["stats"].get("OOS", {})
        verdict_rows.append(
            {
                "detector": name,
                "passed": item["verdict"]["passed"],
                "oos_n": oos.get("n"),
                "oos_rate%": oos.get("t1_px30_rate_pct"),
                "lift": oos.get("lift"),
                "incr_t1": oos.get("incremental_t1_count"),
                "delay5_cover%": oos.get("delay5_cover_pct"),
                "fwd40_mean%": oos.get("fwd40_mean_pct"),
                "000636_hits": item.get("case_000636", {}).get("count"),
                "failed": ",".join(item["verdict"]["failed"]),
            }
        )
    lines.extend(
        markdown_table(
            verdict_rows,
            [
                "detector",
                "passed",
                "oos_n",
                "oos_rate%",
                "lift",
                "incr_t1",
                "delay5_cover%",
                "fwd40_mean%",
                "000636_hits",
                "failed",
            ],
        )
    )

    selected = summary["selected"]
    lines.extend(["", "## Conclusion", ""])
    if selected:
        lines.append(f"Selected for mirror-backtest: `{selected}`.")
    else:
        lines.append(
            "No detector passed the pre-declared first-stage gate. Do not move to portfolio mirror-backtest without redesign."
        )

    lines.extend(["", "## Post-Hoc Diagnostic: Longer Post-Board Memory", ""])
    lines.append(
        "After v1 missed 000636.SZ, three long-memory post-board base rules were tested as diagnostics only. "
        "They are not eligible for the pre-declared pass verdict."
    )
    diag_rows = []
    for name, item in summary.get("diagnostics", {}).items():
        oos = item["stats"].get("OOS", {})
        diag_rows.append(
            {
                "diagnostic": name,
                "oos_n": oos.get("n"),
                "oos_rate%": oos.get("t1_px30_rate_pct"),
                "lift": oos.get("lift"),
                "fwd40_mean%": oos.get("fwd40_mean_pct"),
                "000636_hits": item.get("case_000636", {}).get("count"),
            }
        )
    lines.extend(markdown_table(diag_rows, ["diagnostic", "oos_n", "oos_rate%", "lift", "fwd40_mean%", "000636_hits"]))

    lines.extend(["", "## Detector Details"])
    for name, item in summary["detectors"].items():
        lines.extend(["", f"### {name}", ""])
        detail_rows = []
        for seg, row in item["stats"].items():
            detail_rows.append({"segment": seg, **row})
        lines.extend(
            markdown_table(
                detail_rows,
                [
                    "segment",
                    "n",
                    "symbols",
                    "t1_px30_rate_pct",
                    "lift",
                    "fwd40_mean_pct",
                    "fwd40_median_pct",
                    "fwd20_mean_pct",
                    "delay5_cover_pct",
                    "incremental_t1_count",
                ],
            )
        )
        lines.extend(["", "Yearly:"])
        lines.extend(markdown_table(item["yearly"], ["year", "n", "t1_px30_rate_pct", "delay5_cover_pct", "fwd40_mean_pct"]))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[universe] loading primary universe")
    df, meta = load_primary_universe()
    rates = base_rates(df)
    print(
        "[universe] rows={rows} symbols={symbols} base OOS t1={rate:.1f}%".format(
            rows=len(df), symbols=df["symbol"].nunique(), rate=rates["OOS"] * 100
        )
    )

    masks = detector_masks(df)
    detectors = {}
    for name, mask in masks.items():
        events = dedupe_events(df, mask)
        stats = event_stats(events, rates)
        verdict = pass_verdict(stats)

        market_events = events[events["market_gate"]].copy()
        market_stats = event_stats(market_events, rates)

        detectors[name] = {
            "raw_rows": int(mask.fillna(False).sum()),
            "events": int(len(events)),
            "stats": stats,
            "market_gate_stats": market_stats,
            "yearly": yearly_stats(events),
            "verdict": verdict,
            "case_000636": case_000636(events),
        }
        events.to_parquet(OUTPUT_DIR / f"events_{name}.parquet", index=False)
        print(
            f"[{name}] events={len(events)} OOS={stats['OOS'].get('n', 0)} "
            f"rate={stats['OOS'].get('t1_px30_rate_pct')} lift={stats['OOS'].get('lift')} "
            f"pass={verdict['passed']}"
        )

    diagnostics = {}
    for name, mask in diagnostic_masks(df).items():
        events = dedupe_events(df, mask)
        stats = event_stats(events, rates)
        diagnostics[name] = {
            "raw_rows": int(mask.fillna(False).sum()),
            "events": int(len(events)),
            "stats": stats,
            "yearly": yearly_stats(events),
            "case_000636": case_000636(events),
        }
        events.to_parquet(OUTPUT_DIR / f"events_{name}.parquet", index=False)
        print(
            f"[{name}] diagnostic events={len(events)} OOS={stats['OOS'].get('n', 0)} "
            f"rate={stats['OOS'].get('t1_px30_rate_pct')} lift={stats['OOS'].get('lift')} "
            f"000636={diagnostics[name]['case_000636']['count']}"
        )

    passing = [
        (name, item)
        for name, item in detectors.items()
        if item["verdict"]["passed"]
    ]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda kv: (
                kv[1]["stats"]["OOS"].get("lift") or 0,
                kv[1]["stats"]["OOS"].get("incremental_t1_count") or 0,
            ),
        )[0]

    summary = {
        "meta": meta | {"cooldown_bars": COOLDOWN_BARS, "first_test_year": FIRST_TEST_YEAR},
        "base_rates_pct": {k: round_float(v * 100, 1) for k, v in rates.items()},
        "detectors": detectors,
        "diagnostics": diagnostics,
        "selected": selected,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")
    print(f"[report] {REPORT_PATH}")


if __name__ == "__main__":
    main()
