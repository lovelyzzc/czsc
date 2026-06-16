"""Sell-side value research for the surge-regime state machine.

Question:
    Does the FSM's sell side (Divergence=9 / Breakdown=10) have value as a
    holding risk warning beyond the existing surge-entry strategy?

Pre-declared protocol:
- Build all causal FSM state sequences with the same ``trend_regime.iter_states``
  logic used by live/research paths.
- A sell warning is the first bar entering state 9 or 10 after a non-sell state.
- Primary population: previous state is in the uptrend family {5, 6, 7, 8};
  this approximates "I already hold a trend stock; should this state tell me to
  reduce/exit?"
- Decision is at signal close T. Evaluation uses next-open to next-open returns
  over horizons {5, 10, 20}. If the sell warning is useful, those forward returns
  should underperform same-date/same-amount-decile random controls.
- Avoid-excess = same-date/same-amount-decile median return - event return. Positive means selling
  avoided underperformance.

Pass criterion for a standalone risk-warning module:
- Primary population, OOS (year >= 2024), horizon 10:
  avoid-excess mean > 0, median > 0, t >= 2, and IS mean > 0.

Implementation note:
The local source package currently has a ``czsc.__init__`` / native-extension
surface mismatch for normal imports. This script installs a minimal in-process
``czsc`` shim exposing only CZSC/Freq/format_standard_kline before importing
``trend_regime``; it does not modify package files.

    .venv/bin/python scripts/surge_sell_warning_research.py
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PACKAGE_DIR = REPO_ROOT / "czsc"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def install_czsc_shim() -> None:
    """Expose just the API trend_regime imports, without running czsc.__init__."""
    if "czsc" in sys.modules and hasattr(sys.modules["czsc"], "CZSC"):
        return
    pkg = types.ModuleType("czsc")
    pkg.__path__ = [str(PACKAGE_DIR)]
    sys.modules["czsc"] = pkg
    native = importlib.import_module("czsc._native")
    pkg.CZSC = native.CZSC
    pkg.Freq = native.Freq
    fmt = importlib.import_module("czsc._format_standard_kline")
    pkg.format_standard_kline = fmt.format_standard_kline


install_czsc_shim()
import trend_regime as tr  # noqa: E402


OUTPUT_DIR = SCRIPTS_DIR / "_output" / "surge_sell_warning"
EVENT_CACHE = OUTPUT_DIR / "sell_events.parquet"
REPORT_PATH = SCRIPTS_DIR / "SURGE_REGIME_SELL_WARNING_RESEARCH_2026-06-16.md"
PANEL_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "panel.parquet"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

FIRST_TEST_YEAR = 2024
HORIZONS = (5, 10, 20)
PRIMARY_PREV_REGIMES = {5, 6, 7, 8}
CONTROL_MIN_GROUP = 10


def round_float(value: float | int | None, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def t_stat(values: pd.Series) -> float | None:
    v = values.dropna()
    if len(v) < 2:
        return None
    std = v.std()
    if std <= 0 or pd.isna(std):
        return None
    return float(v.mean() / (std / np.sqrt(len(v))))


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


def _process_one(parquet_path: str) -> list[dict[str, Any]] | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df, with_features=False)
    if len(states) < 30:
        return None
    symbol = str(df["symbol"].iloc[0])
    dates = pd.to_datetime(df["dt"]).to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float) if "amount" in df.columns else np.full(len(df), np.nan)
    n = len(df)

    rows: list[dict[str, Any]] = []
    for i in range(1, len(states)):
        cur, prev = states[i], states[i - 1]
        if cur.regime not in tr.SELL_REGIMES or prev.regime in tr.SELL_REGIMES:
            continue
        entry_idx = cur.idx + 1
        if entry_idx >= n or not np.isfinite(opens[entry_idx]):
            continue
        row: dict[str, Any] = {
            "symbol": symbol,
            "dec_dt": cur.dt,
            "entry_dt": pd.Timestamp(dates[entry_idx]),
            "sell_regime": int(cur.regime),
            "prev_regime": int(prev.regime),
            "idx": int(cur.idx),
            "entry_idx": int(entry_idx),
            "close": float(cur.close),
            "entry_open": float(opens[entry_idx]),
            "amount_e": round(float(amount[cur.idx]) / 1e5, 3) if np.isfinite(amount[cur.idx]) else np.nan,
            "seg": "test" if cur.dt.year >= FIRST_TEST_YEAR else "train",
            "year": int(cur.dt.year),
            "primary": int(prev.regime in PRIMARY_PREV_REGIMES),
        }
        for h in HORIZONS:
            exit_idx = entry_idx + h
            if exit_idx < n and np.isfinite(opens[exit_idx]):
                row[f"exit_dt_{h}"] = pd.Timestamp(dates[exit_idx])
                row[f"ret_o2o_{h}_pct"] = round((opens[exit_idx] / opens[entry_idx] - 1.0) * 100, 4)
            else:
                row[f"exit_dt_{h}"] = pd.NaT
                row[f"ret_o2o_{h}_pct"] = np.nan
        rows.append(row)
    return rows or None


def build_sell_events(force: bool = False) -> pd.DataFrame:
    if EVENT_CACHE.exists() and not force:
        print(f"[events] cache hit: {EVENT_CACHE}")
        return pd.read_parquet(EVENT_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    workers = min(mp.cpu_count(), 8)
    print(f"[events] building FSM sell events from {len(files)} files with {workers} workers")
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_one, files, chunksize=20), 1):
            if res:
                rows.extend(res)
            if i % 500 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] events={len(rows)} elapsed={time.time() - t0:.0f}s")
    events = pd.DataFrame(rows)
    st_intervals = load_st_intervals()
    if st_intervals and len(events):
        events["is_st"] = events.apply(lambda r: is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
        events = events[~events["is_st"]].drop(columns=["is_st"]).reset_index(drop=True)
    events.to_parquet(EVENT_CACHE, index=False)
    print(f"[events] wrote {len(events)} rows -> {EVENT_CACHE} ({time.time() - t0:.0f}s)")
    return events


def add_controls(events: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic same-date/same-amount-decile median controls.

    For each stock/date in the panel, the control return is measured from the
    next open after ``dt`` to the next-open horizon, exactly matching the event
    decision semantics. Controls are the median return of all stocks in the
    same decision-date amount decile; groups with fewer than CONTROL_MIN_GROUP
    finite returns are ignored.
    """
    panel = pd.read_parquet(PANEL_PATH, columns=["symbol", "dt", "open", "amount_e"])
    panel = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    panel["amount_decile"] = np.floor(panel.groupby("dt")["amount_e"].rank(pct=True).mul(10).clip(upper=9.999))
    out = events.merge(
        panel[["symbol", "dt", "amount_decile"]].rename(columns={"dt": "dec_dt"}),
        on=["symbol", "dec_dt"],
        how="left",
    )
    for h in HORIZONS:
        entry_open = panel.groupby("symbol")["open"].shift(-1)
        exit_open = panel.groupby("symbol")["open"].shift(-(h + 1))
        panel[f"ctrl_ret_{h}_pct"] = (exit_open / entry_open - 1.0) * 100
        grouped = panel.dropna(subset=["amount_decile", f"ctrl_ret_{h}_pct"]).groupby(["dt", "amount_decile"])[
            f"ctrl_ret_{h}_pct"
        ]
        ctrl = grouped.agg(["median", "count"]).reset_index()
        ctrl.loc[ctrl["count"] < CONTROL_MIN_GROUP, "median"] = np.nan
        ctrl = ctrl.rename(columns={"dt": "dec_dt", "median": f"control_median_{h}_pct"})
        out = out.merge(ctrl[["dec_dt", "amount_decile", f"control_median_{h}_pct"]], on=["dec_dt", "amount_decile"], how="left")
        out[f"avoid_excess_{h}_pct"] = out[f"control_median_{h}_pct"] - out[f"ret_o2o_{h}_pct"]
    return out


def value_stats(df: pd.DataFrame, horizon: int) -> dict[str, Any]:
    avoid = df[f"avoid_excess_{horizon}_pct"].dropna()
    event_ret = df.loc[avoid.index, f"ret_o2o_{horizon}_pct"]
    control = df.loc[avoid.index, f"control_median_{horizon}_pct"]
    if len(avoid) < 30:
        return {"n": int(len(avoid))}
    return {
        "n": int(len(avoid)),
        "event_mean_pct": round_float(event_ret.mean()),
        "event_median_pct": round_float(event_ret.median()),
        "control_median_mean_pct": round_float(control.mean()),
        "avoid_mean_pct": round_float(avoid.mean()),
        "avoid_median_pct": round_float(avoid.median()),
        "avoid_t": round_float(t_stat(avoid)),
        "avoid_positive_pct": round_float((avoid > 0).mean() * 100, 1),
    }


def summarize(events: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "primary_all": events[events["primary"] == 1],
        "primary_breakdown": events[(events["primary"] == 1) & (events["sell_regime"] == int(tr.Regime.Breakdown))],
        "primary_divergence": events[(events["primary"] == 1) & (events["sell_regime"] == int(tr.Regime.Divergence))],
        "all_sell_events": events,
    }
    summary: dict[str, Any] = {"groups": {}, "yearly_primary_h10": []}
    for name, group in groups.items():
        summary["groups"][name] = {}
        for seg_name, seg_df in {
            "ALL": group,
            "IS": group[group["year"] < FIRST_TEST_YEAR],
            "OOS": group[group["year"] >= FIRST_TEST_YEAR],
        }.items():
            summary["groups"][name][seg_name] = {str(h): value_stats(seg_df, h) for h in HORIZONS}

    primary = events[events["primary"] == 1]
    for year, group in primary.groupby("year"):
        row = {"year": int(year)}
        row.update({f"h{h}": value_stats(group, h).get("avoid_mean_pct") for h in HORIZONS})
        row["n_h10"] = value_stats(group, 10).get("n")
        row["h10_t"] = value_stats(group, 10).get("avoid_t")
        summary["yearly_primary_h10"].append(row)

    oos_h10 = summary["groups"]["primary_all"]["OOS"]["10"]
    is_h10 = summary["groups"]["primary_all"]["IS"]["10"]
    passed = (
        (oos_h10.get("avoid_mean_pct") or 0) > 0
        and (oos_h10.get("avoid_median_pct") or 0) > 0
        and (oos_h10.get("avoid_t") or 0) >= 2
        and (is_h10.get("avoid_mean_pct") or 0) > 0
    )
    failed = []
    if (oos_h10.get("avoid_mean_pct") or 0) <= 0:
        failed.append("oos_mean")
    if (oos_h10.get("avoid_median_pct") or 0) <= 0:
        failed.append("oos_median")
    if (oos_h10.get("avoid_t") or 0) < 2:
        failed.append("oos_t")
    if (is_h10.get("avoid_mean_pct") or 0) <= 0:
        failed.append("is_mean")
    summary["verdict"] = {"passed": bool(passed), "failed": failed}
    return summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(summary: dict[str, Any], events: pd.DataFrame) -> None:
    lines = [
        "# Surge Regime sell-side warning research (2026-06-16)",
        "",
        "This report tests whether FSM sell states 9/10 have value as a holding risk warning.",
        "Positive avoid-excess means selling at next open avoided underperformance versus same-date/same-amount-decile median controls.",
        "",
        "## Protocol",
        "",
        "- Sell warning: first transition into Divergence(9) or Breakdown(10) after a non-sell state.",
        "- Primary population: previous state in {5,6,7,8}.",
        "- Returns: next-open to next-open over 5/10/20 bars.",
        "- Pass criterion: primary OOS horizon-10 avoid mean > 0, median > 0, t >= 2, and IS mean > 0.",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(summary["verdict"], ensure_ascii=False, indent=2),
        "```",
        "",
        "Event counts:",
        "",
        "```json",
        json.dumps(
            {
                "events": int(len(events)),
                "primary_events": int((events["primary"] == 1).sum()),
                "oos_primary_events": int(((events["primary"] == 1) & (events["year"] >= FIRST_TEST_YEAR)).sum()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Primary Results",
    ]

    rows = []
    for seg in ["ALL", "IS", "OOS"]:
        for h in HORIZONS:
            stat = summary["groups"]["primary_all"][seg][str(h)]
            rows.append({"segment": seg, "horizon": h, **stat})
    lines.extend(
        markdown_table(
            rows,
            [
                "segment",
                "horizon",
                "n",
                "event_mean_pct",
                "event_median_pct",
                "control_median_mean_pct",
                "avoid_mean_pct",
                "avoid_median_pct",
                "avoid_t",
                "avoid_positive_pct",
            ],
        )
    )

    lines.extend(["", "## Breakdown vs Divergence", ""])
    rows = []
    for group_name in ["primary_breakdown", "primary_divergence"]:
        for seg in ["IS", "OOS"]:
            stat = summary["groups"][group_name][seg]["10"]
            rows.append({"group": group_name, "segment": seg, **stat})
    lines.extend(
        markdown_table(
            rows,
            [
                "group",
                "segment",
                "n",
                "event_mean_pct",
                "event_median_pct",
                "avoid_mean_pct",
                "avoid_median_pct",
                "avoid_t",
                "avoid_positive_pct",
            ],
        )
    )

    lines.extend(["", "## Yearly Primary Horizon-10", ""])
    lines.extend(markdown_table(summary["yearly_primary_h10"], ["year", "n_h10", "h5", "h10", "h20", "h10_t"]))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = build_sell_events()
    print(f"[events] loaded {len(events)} rows; primary={int((events['primary'] == 1).sum())}")
    valued = add_controls(events)
    valued.to_parquet(OUTPUT_DIR / "sell_events_with_controls.parquet", index=False)
    summary = summarize(valued)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary, valued)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")
    print(f"[report] {REPORT_PATH}")


if __name__ == "__main__":
    main()
