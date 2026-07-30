"""Compute market-level weekly benchmarks aligned to the xs_chan decision calendar.

Produces equal-weight all-A and CSI1000 index weekly returns using the same
5-session open-to-open return window as the xs_chan strategy:

    fwd_5d_open_return = open[session_no + 6] / open[session_no + 1] - 1

The EW-All benchmark answers: "What would an investor earn by holding ALL
tradable stocks equally for the same 5-session window?"

The CSI1000 benchmark provides a market-cap-weighted small-cap reference.

Usage:
    uv run --no-sync python scripts/xs_chan_market_benchmark.py generate
    uv run --no-sync python scripts/xs_chan_market_benchmark.py generate --refresh-index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
INDEX_CACHE_DIR = Path.home() / ".ts_data_cache" / "index_daily"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "xs_chan_market_benchmark"
SOURCE_PATH = Path(__file__).resolve()
MANIFEST_SCHEMA = "xs_chan_market_benchmark_v2"
HOLDING_SESSIONS = 5
MIN_TRADABLE_STOCKS = 200
MIN_OPEN_PRICE = 1.0
MIN_AMOUNT = 0.0

CSI1000_CODE = "000852.SH"
CSI500_CODE = "000905.SH"
INDEX_CODES = [CSI1000_CODE, CSI500_CODE]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_data_closure() -> dict[str, int | str]:
    """Bind every raw parquet by name, size, and physical SHA-256."""

    paths = sorted(DATA_DIR.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"no raw market parquet files found under {DATA_DIR}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        size = path.stat().st_size
        physical_sha256 = _sha256_file(path)
        digest.update(f"{path.name}\0{size}\0{physical_sha256}\n".encode())
        total_bytes += size
    return {
        "algorithm": "sha256_of_sorted_name_nul_size_nul_physical_sha256_lines",
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "digest_sha256": digest.hexdigest(),
    }


def load_market_calendar(end_date: str | None = None) -> pd.DatetimeIndex:
    """Build market-session calendar from all daily parquet files."""

    if pl is not None:
        scan = pl.scan_parquet(str(DATA_DIR / "*.parquet")).select(
            pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=True).alias("dt")
        )
        if end_date is not None:
            scan = scan.filter(pl.col("dt") <= pd.Timestamp(end_date).date())
        dates = scan.unique().sort("dt").collect(engine="streaming")["dt"].to_list()
    else:
        frames = []
        for p in sorted(DATA_DIR.glob("*.parquet"))[:50]:
            df = pd.read_parquet(p, columns=["trade_date"])
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        combined["dt"] = pd.to_datetime(combined["trade_date"], format="%Y%m%d")
        if end_date is not None:
            combined = combined[combined["dt"] <= pd.Timestamp(end_date)]
        dates = sorted(combined["dt"].unique())

    calendar = pd.DatetimeIndex(pd.to_datetime(dates))
    if calendar.empty or not calendar.is_monotonic_increasing:
        raise RuntimeError("market calendar is empty or not monotonic")
    return calendar


def build_decision_dates(
    calendar: pd.DatetimeIndex,
    start_date: str = "20220101",
    end_date: str | None = None,
) -> pd.DatetimeIndex:
    """Derive weekly decision dates using the same W-FRI logic as xs_chan."""

    table = pd.DataFrame({"session_dt": calendar, "session_no": np.arange(len(calendar), dtype=np.int32)})
    table["week"] = table["session_dt"].dt.to_period("W-FRI")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else calendar.max()
    scoped = table[table["session_dt"].between(start, end)]
    decisions = scoped.groupby("week", sort=True, observed=True)["session_no"].max()
    decision_sessions = calendar[decisions.to_numpy()]
    return pd.DatetimeIndex(decision_sessions)


def load_daily_panel(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Load all A-stock daily data into a session-indexed panel.

    Returns DataFrame with columns: symbol, session_no, open, close, amount
    """

    start_ts = time.time()
    session_lookup = pd.Series(np.arange(len(calendar), dtype=np.int32), index=calendar)

    if pl is not None:
        print("[benchmark] Loading daily panel via polars...", flush=True)
        raw = (
            pl.scan_parquet(str(DATA_DIR / "*.parquet"))
            .select(
                pl.col("ts_code").alias("symbol"),
                pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=True).alias("dt"),
                pl.col("open").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("amount").cast(pl.Float64),
            )
            .collect(engine="streaming")
        )
        panel = raw.to_pandas()
    else:
        print("[benchmark] Loading daily panel via pandas...", flush=True)
        frames = []
        for p in sorted(DATA_DIR.glob("*.parquet")):
            df = pd.read_parquet(p, columns=["ts_code", "trade_date", "open", "close", "amount"])
            frames.append(df)
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.rename(columns={"ts_code": "symbol"})
        panel["dt"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")

    panel["dt"] = pd.to_datetime(panel["dt"])
    panel["session_no"] = panel["dt"].map(session_lookup)
    panel = panel.dropna(subset=["session_no"])
    panel["session_no"] = panel["session_no"].astype(np.int32)

    elapsed = time.time() - start_ts
    print(f"[benchmark] Loaded {len(panel):,} rows, {panel['symbol'].nunique()} symbols in {elapsed:.1f}s", flush=True)
    return panel[["symbol", "session_no", "open", "close", "amount"]].copy()


def compute_ew_weekly_returns(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute equal-weight all-A 5-session open-to-open returns per decision week.

    For each decision_dt at session_no=D:
      entry_open = open at session D+1
      exit_open  = open at session D+1+HOLDING_SESSIONS = D+6
      stock_return = exit_open / entry_open - 1
      ew_return = mean(stock_return) across all tradable stocks

    The weekly universe is fixed using entry-session observability only.
    Missing exit opens receive the same zero-return proxy used by the Stage 2
    fixed-slot stress path and are disclosed through ``observable_exit_rate``.
    """

    session_lookup = pd.Series(np.arange(len(calendar), dtype=np.int32), index=calendar)
    decision_sessions = decision_dates.map(lambda d: session_lookup.get(d))

    pivot_open = panel.pivot_table(index="session_no", columns="symbol", values="open", aggfunc="last")
    pivot_amount = panel.pivot_table(index="session_no", columns="symbol", values="amount", aggfunc="last")

    rows = []
    for dec_dt, dec_session in zip(decision_dates, decision_sessions, strict=True):
        if pd.isna(dec_session):
            continue
        entry_session = int(dec_session) + 1
        exit_session = entry_session + HOLDING_SESSIONS

        if entry_session not in pivot_open.index or exit_session not in pivot_open.index:
            continue

        entry_opens = pivot_open.loc[entry_session]
        exit_opens = pivot_open.loc[exit_session]

        tradable = entry_opens.notna() & (entry_opens > MIN_OPEN_PRICE)
        if pivot_amount is not None and entry_session in pivot_amount.index:
            entry_amounts = pivot_amount.loc[entry_session]
            tradable = tradable & entry_amounts.notna() & (entry_amounts > MIN_AMOUNT)

        if tradable.sum() < MIN_TRADABLE_STOCKS:
            continue

        selected_entry = entry_opens[tradable]
        selected_exit = exit_opens.reindex(selected_entry.index)
        observable_exit = selected_exit.notna() & np.isfinite(selected_exit) & (selected_exit > 0)
        returns = pd.Series(0.0, index=selected_entry.index, dtype=float)
        returns.loc[observable_exit] = selected_exit.loc[observable_exit] / selected_entry.loc[observable_exit] - 1.0

        rows.append(
            {
                "decision_dt": dec_dt,
                "ew_all_return": float(returns.mean()),
                "ew_all_median_return": float(returns.median()),
                "ew_all_std": float(returns.std()),
                "tradable_count": int(len(returns)),
                "observable_exit_count": int(observable_exit.sum()),
                "observable_exit_rate": float(observable_exit.mean()),
            }
        )

    result = pd.DataFrame(rows)
    print(f"[benchmark] EW-All: {len(result)} weeks computed", flush=True)
    return result


def _resolve_tushare_token() -> str:
    token = os.getenv("TINYSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if not token:
        token_file = Path.home() / ".tushare_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        raise RuntimeError("No Tushare token found (TINYSHARE_TOKEN/TUSHARE_TOKEN env or ~/.tushare_token)")
    return token


def fetch_index_daily(
    ts_code: str,
    start_date: str = "20210101",
    end_date: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch index daily data via Tushare, with local parquet cache."""

    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = INDEX_CACHE_DIR / f"{ts_code.replace('.', '_')}.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        df["trade_date"] = df["trade_date"].astype(str)
        if end_date:
            df = df[df["trade_date"] <= end_date]
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if not df.empty:
            print(f"[benchmark] Index {ts_code}: loaded {len(df)} rows from cache", flush=True)
            return df.sort_values("trade_date").reset_index(drop=True)

    import tinyshare as ts

    token = _resolve_tushare_token()
    ts.set_token(token)
    pro = ts.pro_api()

    edt = end_date or datetime.now().strftime("%Y%m%d")
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=edt)
    if df is None or df.empty:
        raise RuntimeError(f"Tushare returned no data for index {ts_code}")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(f"[benchmark] Index {ts_code}: fetched {len(df)} rows, cached to {cache_path}", flush=True)
    return df


def compute_index_weekly_returns(
    index_daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    ts_code: str,
) -> pd.DataFrame:
    """Compute index open-to-open returns over the strategy's exact window.

    Entry is the official index open at ``D+1`` and exit is the open at
    ``D+1+HOLDING_SESSIONS``. This matches the strategy proxy instead of
    silently adding the decision-day close-to-next-open interval.
    """

    session_lookup = pd.Series(np.arange(len(calendar), dtype=np.int32), index=calendar)
    index_daily = index_daily.copy()
    index_daily["dt"] = pd.to_datetime(index_daily["trade_date"], format="%Y%m%d")
    index_daily["session_no"] = index_daily["dt"].map(session_lookup)
    index_daily = index_daily.dropna(subset=["session_no"]).sort_values("session_no")
    index_daily["session_no"] = index_daily["session_no"].astype(np.int32)

    if "open" not in index_daily:
        raise RuntimeError(f"index data for {ts_code} does not contain open")
    open_by_session = index_daily.set_index("session_no")["open"]

    col_name = f"{ts_code.split('.')[0].lower()}_return"
    rows = []
    for dec_dt in decision_dates:
        dec_session = session_lookup.get(dec_dt)
        if pd.isna(dec_session):
            continue
        dec_session = int(dec_session)
        entry_session = dec_session + 1
        exit_session = entry_session + HOLDING_SESSIONS

        if entry_session not in open_by_session.index or exit_session not in open_by_session.index:
            continue

        entry_open = open_by_session.loc[entry_session]
        exit_open = open_by_session.loc[exit_session]

        if not (np.isfinite(entry_open) and np.isfinite(exit_open) and entry_open > 0):
            continue

        rows.append(
            {
                "decision_dt": dec_dt,
                col_name: float(exit_open / entry_open - 1.0),
            }
        )

    result = pd.DataFrame(rows)
    print(f"[benchmark] Index {ts_code}: {len(result)} weeks computed", flush=True)
    return result


def generate_benchmarks(
    *,
    start_date: str = "20220101",
    end_date: str | None = None,
    refresh_index: bool = False,
) -> Path:
    """Generate the full market benchmark table."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[benchmark] Building market calendar...", flush=True)
    # ``end_date`` limits decision dates, not input sessions. The strategy
    # still needs D+1 entry and D+6 exit sessions after the last decision.
    calendar = load_market_calendar()
    decision_dates = build_decision_dates(calendar, start_date=start_date, end_date=end_date)
    session_lookup = pd.Series(np.arange(len(calendar), dtype=np.int32), index=calendar)
    decision_sessions = decision_dates.map(lambda dt: session_lookup.get(dt))
    complete_mask = np.asarray(
        [
            pd.notna(session_no) and int(session_no) + 1 + HOLDING_SESSIONS < len(calendar)
            for session_no in decision_sessions
        ],
        dtype=bool,
    )
    incomplete_dates = decision_dates[~complete_mask]
    if end_date is not None and len(incomplete_dates):
        raise RuntimeError(
            "requested decision range lacks D+6 market data for "
            f"{len(incomplete_dates)} dates; last={incomplete_dates[-1].date()}"
        )
    decision_dates = decision_dates[complete_mask]
    print(f"[benchmark] Calendar: {len(calendar)} sessions, {len(decision_dates)} decision weeks", flush=True)

    panel = load_daily_panel(calendar)
    ew_weekly = compute_ew_weekly_returns(panel, calendar, decision_dates)
    del panel
    if len(ew_weekly) != len(decision_dates) or set(ew_weekly["decision_dt"]) != set(decision_dates):
        raise RuntimeError(
            "EW-All benchmark does not cover every complete decision date: "
            f"expected={len(decision_dates)} observed={len(ew_weekly)}"
        )

    index_frames = []
    index_inputs: list[dict[str, object]] = []
    index_data_end = calendar.max().strftime("%Y%m%d")
    for code in INDEX_CODES:
        try:
            index_daily = fetch_index_daily(
                code,
                start_date=start_date,
                end_date=index_data_end,
                refresh=refresh_index,
            )
            idx_weekly = compute_index_weekly_returns(index_daily, calendar, decision_dates, code)
            if len(idx_weekly) != len(decision_dates) or set(idx_weekly["decision_dt"]) != set(decision_dates):
                raise RuntimeError(
                    f"{code} does not cover every requested decision date: "
                    f"expected={len(decision_dates)} observed={len(idx_weekly)}"
                )
            index_frames.append(idx_weekly)
            cache_path = INDEX_CACHE_DIR / f"{code.replace('.', '_')}.parquet"
            index_inputs.append(
                {
                    "code": code,
                    "available": True,
                    "output_column": f"{code.split('.')[0].lower()}_return",
                    "weekly_rows": len(idx_weekly),
                    "cache_path": str(cache_path.resolve()),
                    "cache_sha256": _sha256_file(cache_path),
                }
            )
        except Exception as exc:
            print(f"[benchmark] WARNING: Failed to fetch {code}: {exc}", flush=True)
            index_inputs.append(
                {
                    "code": code,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    result = ew_weekly.copy()
    for idx_df in index_frames:
        result = result.merge(idx_df, on="decision_dt", how="left")

    output_path = OUTPUT_DIR / "market_benchmark_weekly.parquet"
    result.to_parquet(output_path, index=False)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "start_date": start_date,
        "requested_decision_end_date": end_date,
        "decision_start_date": result["decision_dt"].min().date().isoformat(),
        "decision_end_date": result["decision_dt"].max().date().isoformat(),
        "market_data_end_date": calendar.max().date().isoformat(),
        "calendar_sessions": int(len(calendar)),
        "decision_weeks": int(len(decision_dates)),
        "incomplete_decision_dates_excluded": [dt.date().isoformat() for dt in incomplete_dates],
        "output_weeks": int(len(result)),
        "holding_sessions": HOLDING_SESSIONS,
        "ew_all_mean_tradable_count": int(result["tradable_count"].mean()) if "tradable_count" in result else 0,
        "ew_all_mean_observable_exit_rate": (
            float(result["observable_exit_rate"].mean()) if "observable_exit_rate" in result else None
        ),
        "ew_all_min_observable_exit_rate": (
            float(result["observable_exit_rate"].min()) if "observable_exit_rate" in result else None
        ),
        "ew_all_missing_exit_observations": (
            int((result["tradable_count"] - result["observable_exit_count"]).sum())
            if {"tradable_count", "observable_exit_count"} <= set(result.columns)
            else None
        ),
        "columns": list(result.columns),
        "output_sha256": _sha256_file(output_path),
        "generator_path": str(SOURCE_PATH),
        "generator_sha256": _sha256_file(SOURCE_PATH),
        "raw_data_closure": _physical_data_closure(),
        "requested_index_codes": INDEX_CODES,
        "available_index_codes": [entry["code"] for entry in index_inputs if entry["available"]],
        "index_inputs": index_inputs,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    verify_benchmark(OUTPUT_DIR)
    print(f"[benchmark] Output: {output_path} ({len(result)} rows)", flush=True)
    print(f"[benchmark] Manifest: {manifest_path}", flush=True)
    return output_path


def verify_benchmark(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    """Fail closed on stale source, raw inputs, index caches, or output bytes."""

    path = output_dir / "market_benchmark_weekly.parquet"
    manifest_path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Market benchmark not found at {path}. Run: "
            "uv run --no-sync python scripts/xs_chan_market_benchmark.py generate"
        )
    if not manifest_path.is_file():
        raise RuntimeError(f"market benchmark manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            f"unsupported market benchmark manifest schema: {manifest.get('schema')!r}; regenerate benchmark"
        )
    if manifest.get("generator_sha256") != _sha256_file(SOURCE_PATH):
        raise RuntimeError("market benchmark generator changed; regenerate benchmark")
    if manifest.get("output_sha256") != _sha256_file(path):
        raise RuntimeError("market benchmark parquet hash does not match its manifest")
    observed_closure = _physical_data_closure()
    if manifest.get("raw_data_closure") != observed_closure:
        raise RuntimeError("raw A-share input closure changed; regenerate market benchmark")
    for entry in manifest.get("index_inputs", []):
        if not entry.get("available"):
            continue
        cache_path = Path(str(entry.get("cache_path", "")))
        if not cache_path.is_file() or _sha256_file(cache_path) != entry.get("cache_sha256"):
            raise RuntimeError(f"index cache changed or is missing for {entry.get('code')}")

    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError("market benchmark is empty")
    if frame.duplicated("decision_dt").any():
        raise RuntimeError("market benchmark contains duplicate decision dates")
    if list(frame.columns) != manifest.get("columns"):
        raise RuntimeError("market benchmark columns do not match its manifest")
    if len(frame) != int(manifest.get("output_weeks", -1)):
        raise RuntimeError("market benchmark row count does not match its manifest")
    observed_start = pd.Timestamp(frame["decision_dt"].min()).date().isoformat()
    observed_end = pd.Timestamp(frame["decision_dt"].max()).date().isoformat()
    if observed_start != manifest.get("decision_start_date") or observed_end != manifest.get("decision_end_date"):
        raise RuntimeError("market benchmark date range does not match its manifest")
    return manifest


def load_benchmark_weekly(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Load a fully verified pre-computed market benchmark weekly table."""

    verify_benchmark(output_dir)
    path = output_dir / "market_benchmark_weekly.parquet"
    df = pd.read_parquet(path)
    df["decision_dt"] = pd.to_datetime(df["decision_dt"])
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate market benchmark weekly returns")
    gen.add_argument("--start-date", default="20220101")
    gen.add_argument("--end-date", default=None)
    gen.add_argument("--refresh-index", action="store_true", help="Re-fetch index data from Tushare")

    subparsers.add_parser("info", help="Show benchmark info from manifest")
    subparsers.add_parser("verify", help="Verify benchmark source, inputs, and output bytes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        generate_benchmarks(
            start_date=args.start_date,
            end_date=args.end_date,
            refresh_index=args.refresh_index,
        )
    elif args.command == "info":
        manifest_path = OUTPUT_DIR / "manifest.json"
        if not manifest_path.exists():
            print("No benchmark generated yet. Run 'generate' first.")
            return 1
        print(manifest_path.read_text())
    elif args.command == "verify":
        print(json.dumps(verify_benchmark(OUTPUT_DIR), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
