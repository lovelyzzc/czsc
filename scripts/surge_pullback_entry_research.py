"""Pullback-entry research for the surge-regime anticipate setup.

This is the next iteration after the market-state walk-forward study. It keeps
the anticipate signal and the selected market-state gate fixed, then compares
immediate entry with delayed pullback entries:

- delay0: original anticipate entry.
- delay5: decision five bars after the upward-departure signal.
- delay7: decision seven bars after the upward-departure signal.
- delay5_first_else7: first viable entry among delay 5 and delay 7 for the same
  original signal, after signal gates, hard filters and market gate.

The dumped candidate rows already enforce "decision day still belongs to the
uptrend family", so this script does not rerun the FSM or change surge_onset.

    uv run --no-sync python scripts/surge_pullback_entry_research.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import trend_regime as tr

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_pullback_entry"


@dataclass(frozen=True)
class EntryVariant:
    name: str
    delays: tuple[int, ...]
    market_gate: str
    dedupe_signal: bool = False


VARIANTS = [
    EntryVariant("original_delay0", (0,), "none"),
    EntryVariant("state_delay0", (0,), "high20_and_index"),
    EntryVariant("state_delay5", (5,), "high20_and_index"),
    EntryVariant("state_delay7", (7,), "high20_and_index"),
    EntryVariant("state_delay5_first_else7", (5, 7), "high20_and_index", dedupe_signal=True),
]
SELECTABLE_VARIANTS = ["state_delay0", "state_delay5", "state_delay7", "state_delay5_first_else7"]


def market_gate_specs() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "none": lambda df: pd.Series(True, index=df.index),
        "high20_and_index": lambda df: (df["high20_ratio"] > 0.12) & (df["ew_index_above_ma20"] > 0),
    }


def _signal_gate(df: pd.DataFrame) -> pd.Series:
    return (
        (df["sig_vol_ratio"] >= tr.SURGE_GATE_VOL_RATIO)
        & (df["sig_ma_spread_pct"] >= tr.SURGE_GATE_MA_SPREAD)
        & (df["sig_above_zg"] == 1)
        & (df["sig_ret20"] >= tr.SURGE_GATE_RET20)
    ).fillna(False)


def _hard_gate(df: pd.DataFrame) -> pd.Series:
    return (
        (df["amount_e"] >= spb.MIN_AMOUNT_E)
        & df["sl_pct"].between(spb.STOP_MIN_PCT, spb.STOP_MAX_PCT)
        & (df["gap_pct"] < df["limit_pct"] - spb.GAP_LIMIT_MARGIN)
    ).fillna(False)


def _apply_st_filter(df: pd.DataFrame, st_intervals: dict) -> pd.DataFrame:
    if not st_intervals or df.empty:
        return df
    st_mask = df.apply(lambda row: spb.is_st_on(st_intervals, row["symbol"], row["dec_dt"]), axis=1)
    return df[~st_mask].copy()


def _prepare_base(cand: pd.DataFrame, market: pd.DataFrame, st_intervals: dict) -> pd.DataFrame:
    df = cand[cand["mode"] == "anticipate"].copy()
    df = df[_signal_gate(df) & _hard_gate(df)]
    df = _apply_st_filter(df, st_intervals)
    df = df.merge(market, left_on="dec_dt", right_on="dt", how="left")
    df["priority"] = [
        tr.priority_score(score, sl_pct, 0, regime)
        for score, sl_pct, regime in zip(df["score"], df["sl_pct"], df["dec_regime"], strict=False)
    ]
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - spb.SELL_COST) / (1 + spb.BUY_COST) - 1) * 100
    return df


def variant_candidates(base_df: pd.DataFrame, variant: EntryVariant) -> pd.DataFrame:
    gates = market_gate_specs()
    df = base_df[base_df["delay"].isin(variant.delays)].copy()
    df = df[gates[variant.market_gate](df)]
    if variant.dedupe_signal:
        df = df.sort_values(
            ["symbol", "sig_dt", "delay", "entry_dt", "priority"], ascending=[True, True, True, True, False]
        )
        df = df.drop_duplicates(["symbol", "sig_dt"], keep="first")
    return df.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)


def _values_stats(values: pd.Series) -> dict:
    values = values.dropna()
    if len(values) < 30:
        return {"n": int(len(values))}
    return {
        "n": int(len(values)),
        "mean_pct": msf.round_float(values.mean()),
        "median_pct": msf.round_float(values.median()),
        "t": msf.round_float(msf.t_stat(values)),
        "positive_pct": msf.round_float((values > 0).mean() * 100, 1),
    }


def _trade_metrics(trades: pd.DataFrame, daily: pd.Series, years: list[int]) -> dict:
    scoped = trades[trades["year"].isin(years)].copy() if len(trades) else pd.DataFrame()
    daily_scoped = daily[daily.index.year.isin(years)] if len(daily) else pd.Series(dtype=float)
    return {
        "years": years,
        "excess": _values_stats(scoped["excess_pct"]) if len(scoped) else {"n": 0},
        "pair": msf.pair_stats_ascii(scoped),
        "curve": msf.curve_stats_ascii(daily_scoped),
    }


def simulate_variant(candidates: pd.DataFrame, sampler: msf.StableControlSampler, slots: int) -> dict:
    trades, daily = spb.simulate_slots(candidates, slots)
    trades = msf.add_excess(trades, sampler) if len(trades) else trades.assign(excess_pct=[])
    years = sorted(int(y) for y in candidates["year"].dropna().unique())
    return {
        "candidates": int(len(candidates)),
        "trades": trades,
        "daily": daily,
        "years": years,
        "year_metrics": {str(year): _trade_metrics(trades, daily, [year]) for year in years},
    }


def _score_train(metrics: dict) -> tuple[float, float]:
    excess = metrics.get("excess") or {}
    t_stat = excess.get("t")
    mean = excess.get("mean_pct")
    if t_stat is None or mean is None or mean <= 0:
        return -np.inf, -np.inf
    return float(t_stat), float(mean)


def _select_variant(records: dict[str, dict], train_years: list[int], min_train_trades: int) -> tuple[str, dict]:
    best_name = "state_delay0"
    best_train = _trade_metrics(records[best_name]["trades"], records[best_name]["daily"], train_years)
    best_score = (-np.inf, -np.inf)
    for name in SELECTABLE_VARIANTS:
        train = _trade_metrics(records[name]["trades"], records[name]["daily"], train_years)
        if (train.get("excess") or {}).get("n", 0) < min_train_trades:
            continue
        score = _score_train(train)
        if score > best_score:
            best_name = name
            best_train = train
            best_score = score
    return best_name, best_train


def walk_forward(records: dict[str, dict], first_test_year: int, min_train_years: int, min_train_trades: int) -> dict:
    years = sorted({int(year) for record in records.values() for year in record["years"]})
    folds = []
    selected_test_trades = []
    selected_test_daily = []
    baseline_test_trades = []
    baseline_test_daily = []

    for test_year in years:
        if test_year < first_test_year:
            continue
        train_years = [year for year in years if year < test_year]
        if len(train_years) < min_train_years:
            continue
        selected_name, train_metrics = _select_variant(records, train_years, min_train_trades)
        selected_record = records[selected_name]
        baseline_record = records["state_delay0"]
        test_metrics = _trade_metrics(selected_record["trades"], selected_record["daily"], [test_year])
        baseline_metrics = _trade_metrics(baseline_record["trades"], baseline_record["daily"], [test_year])

        selected_trades = selected_record["trades"]
        baseline_trades = baseline_record["trades"]
        selected_year_trades = selected_trades[selected_trades["year"] == test_year]
        baseline_year_trades = baseline_trades[baseline_trades["year"] == test_year]
        if len(selected_year_trades):
            selected_test_trades.append(selected_year_trades)
        if len(baseline_year_trades):
            baseline_test_trades.append(baseline_year_trades)
        selected_test_daily.append(selected_record["daily"][selected_record["daily"].index.year == test_year])
        baseline_test_daily.append(baseline_record["daily"][baseline_record["daily"].index.year == test_year])

        folds.append(
            {
                "test_year": test_year,
                "train_years": train_years,
                "selected_variant": selected_name,
                "train": train_metrics,
                "test": test_metrics,
                "state_delay0_test": baseline_metrics,
            }
        )

    selected_trades = pd.concat(selected_test_trades, ignore_index=True) if selected_test_trades else pd.DataFrame()
    baseline_trades = pd.concat(baseline_test_trades, ignore_index=True) if baseline_test_trades else pd.DataFrame()
    selected_daily = pd.concat(selected_test_daily).sort_index() if selected_test_daily else pd.Series(dtype=float)
    baseline_daily = pd.concat(baseline_test_daily).sort_index() if baseline_test_daily else pd.Series(dtype=float)
    fold_years = [fold["test_year"] for fold in folds]
    return {
        "folds": folds,
        "selected_aggregate": _trade_metrics(selected_trades, selected_daily, fold_years),
        "state_delay0_aggregate": _trade_metrics(baseline_trades, baseline_daily, fold_years),
    }


def _row_from_metrics(name: str, label: str, metrics: dict, candidates: int | None = None) -> dict:
    excess = metrics.get("excess") or {}
    pair = metrics.get("pair") or {}
    curve = metrics.get("curve") or {}
    row = {
        "name": name,
        "label": label,
        "years": ",".join(str(y) for y in metrics.get("years", [])),
        "n": excess.get("n", 0),
        "excess": excess.get("mean_pct"),
        "t": excess.get("t"),
        "net": pair.get("net_mean_pct"),
        "annual": curve.get("annual_pct"),
        "mdd": curve.get("max_drawdown_pct"),
    }
    if candidates is not None:
        row["candidates"] = candidates
    return row


def _flatten_variant_rows(records: dict[str, dict], years: list[int]) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        record = records[variant.name]
        metrics = _trade_metrics(record["trades"], record["daily"], years)
        rows.append(_row_from_metrics(variant.name, variant.market_gate, metrics, candidates=record["candidates"]))
    return rows


def _flatten_variant_year_rows(records: dict[str, dict], years: list[int]) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        record = records[variant.name]
        for year in years:
            metrics = _trade_metrics(record["trades"], record["daily"], [year])
            row = _row_from_metrics(variant.name, str(year), metrics)
            row["year"] = year
            rows.append(row)
    return rows


def _flatten_fold_rows(folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        test = fold["test"]
        base = fold["state_delay0_test"]
        test_excess = test.get("excess") or {}
        test_pair = test.get("pair") or {}
        base_excess = base.get("excess") or {}
        base_pair = base.get("pair") or {}
        rows.append(
            {
                "year": fold["test_year"],
                "selected": fold["selected_variant"],
                "train_years": ",".join(str(y) for y in fold["train_years"]),
                "selected_n": test_excess.get("n", 0),
                "selected_excess": test_excess.get("mean_pct"),
                "selected_t": test_excess.get("t"),
                "selected_net": test_pair.get("net_mean_pct"),
                "state_d0_n": base_excess.get("n", 0),
                "state_d0_excess": base_excess.get("mean_pct"),
                "state_d0_t": base_excess.get("t"),
                "state_d0_net": base_pair.get("net_mean_pct"),
            }
        )
    return rows


def write_report(summary: dict, output_dir: Path) -> None:
    years = summary["oos_years"]
    lines = [
        "# Surge pullback-entry research",
        "",
        "Fixed setup: anticipate candidates, original signal gates, existing hard filters, and the prior "
        "`high20_ratio > 0.12 & ew_index_above_ma20` market-state gate for delayed variants.",
        "",
        "## Fixed Variant OOS",
    ]
    lines.extend(
        msf.markdown_table(
            summary["variant_oos_rows"],
            ["name", "label", "candidates", "years", "n", "excess", "t", "net", "annual", "mdd"],
        )
    )
    lines.extend(["", "## Fixed Variant By Year"])
    lines.extend(
        msf.markdown_table(
            summary["variant_year_rows"],
            ["name", "year", "n", "excess", "t", "net", "annual", "mdd"],
        )
    )
    lines.extend(["", "## Walk-Forward Entry Selection"])
    lines.extend(
        msf.markdown_table(
            summary["walk_forward_rows"],
            [
                "year",
                "selected",
                "train_years",
                "selected_n",
                "selected_excess",
                "selected_t",
                "selected_net",
                "state_d0_n",
                "state_d0_excess",
                "state_d0_t",
                "state_d0_net",
            ],
        )
    )
    lines.extend(["", "## Walk-Forward Aggregate"])
    lines.extend(
        msf.markdown_table(
            summary["walk_forward_aggregate_rows"],
            ["name", "label", "years", "n", "excess", "t", "net", "annual", "mdd"],
        )
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"OOS years: {', '.join(str(year) for year in years)}.",
            "The `delay5_first_else7` variant deduplicates by `(symbol, sig_dt)` after all gates and keeps the first viable delayed entry.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=10, help="Slot count for the portfolio mirror.")
    parser.add_argument("--first-test-year", type=int, default=2024, help="First calendar year included in OOS.")
    parser.add_argument("--min-train-years", type=int, default=2)
    parser.add_argument("--min-train-trades", type=int, default=60)
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market_path = msf.OUTPUT_DIR / "market_state.parquet"
    market = pd.read_parquet(market_path) if market_path.exists() else msf.load_market_state()
    cand = pd.read_parquet(msf.CAND_DIR / "candidates.parquet")
    st_intervals = spb.load_st_intervals()
    sampler = msf.StableControlSampler()

    print("[prepare] anticipate base candidates")
    base = _prepare_base(cand, market, st_intervals)
    records = {}
    for variant in VARIANTS:
        print(f"[variant] {variant.name}")
        candidates = variant_candidates(base, variant)
        records[variant.name] = simulate_variant(candidates, sampler, args.slots)

    oos_years = sorted(int(year) for year in base.loc[base["year"] >= args.first_test_year, "year"].dropna().unique())
    walk = walk_forward(records, args.first_test_year, args.min_train_years, args.min_train_trades)
    summary = {
        "slots": args.slots,
        "first_test_year": args.first_test_year,
        "min_train_years": args.min_train_years,
        "min_train_trades": args.min_train_trades,
        "oos_years": oos_years,
        "variants": {
            name: {
                "candidates": record["candidates"],
                "years": record["years"],
                "year_metrics": record["year_metrics"],
            }
            for name, record in records.items()
        },
        "variant_oos_rows": _flatten_variant_rows(records, oos_years),
        "variant_year_rows": _flatten_variant_year_rows(records, oos_years),
        "walk_forward": walk,
        "walk_forward_rows": _flatten_fold_rows(walk["folds"]),
        "walk_forward_aggregate_rows": [
            _row_from_metrics("walk_forward_selected", "selected_by_prior_years", walk["selected_aggregate"]),
            _row_from_metrics("state_delay0", "market_gate_immediate_entry", walk["state_delay0_aggregate"]),
        ],
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary, OUTPUT_DIR)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
