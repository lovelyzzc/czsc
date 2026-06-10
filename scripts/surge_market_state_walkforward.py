"""Walk-forward validation for surge-regime market-state filters.

The previous iteration found several market-state gates that pass the full
2024+ OOS criterion. This script asks the stricter question: could a gate have
been selected using only years that were already known, and would it still work
in the next year?

Selection rule for each mode and test year:

1. Build the existing real-trading mirror for every fixed a-priori gate.
2. Use only trades whose decision year is earlier than the test year.
3. Pick the non-baseline gate with the highest positive train excess t-stat,
   requiring at least --min-train-trades trades; otherwise use baseline.
4. Report that selected gate's next-year performance.

The candidate definition, costs, slot simulator, ST filter and random-control
method are reused from surge_portfolio_backtest.py and
surge_market_state_filter.py.

    uv run --no-sync python scripts/surge_market_state_walkforward.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_market_state_walkforward"


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
    scoped = trades[trades["year"].isin(years)].copy()
    daily_scoped = daily[daily.index.year.isin(years)]
    return {
        "years": years,
        "excess": _values_stats(scoped["excess_pct"]) if len(scoped) else {"n": 0},
        "pair": msf.pair_stats_ascii(scoped),
        "curve": msf.curve_stats_ascii(daily_scoped),
    }


def _score_train(metrics: dict) -> tuple[float, float]:
    excess = metrics.get("excess") or {}
    t_stat = excess.get("t")
    mean = excess.get("mean_pct")
    if t_stat is None or mean is None or mean <= 0:
        return -np.inf, -np.inf
    return float(t_stat), float(mean)


def _simulate_all_gates(base_df: pd.DataFrame, sampler: msf.StableControlSampler, slots: int) -> dict[str, dict]:
    records = {}
    for gate_name, gate_rule, gate_fn in msf.gate_specs():
        filtered = base_df[gate_fn(base_df)].sort_values(["entry_dt", "priority"], ascending=[True, False])
        filtered = filtered.reset_index(drop=True)
        trades, daily = spb.simulate_slots(filtered, slots)
        trades = msf.add_excess(trades, sampler) if len(trades) else trades.assign(excess_pct=[])
        records[gate_name] = {
            "gate": gate_name,
            "rule": gate_rule,
            "candidates": int(len(filtered)),
            "trades": trades,
            "daily": daily,
        }
    return records


def _select_gate(records: dict[str, dict], train_years: list[int], min_train_trades: int) -> tuple[str, dict]:
    baseline = records["baseline"]
    best_gate = "baseline"
    best_train = _trade_metrics(baseline["trades"], baseline["daily"], train_years)
    best_score = (-np.inf, -np.inf)

    for gate_name, record in records.items():
        if gate_name == "baseline":
            continue
        train = _trade_metrics(record["trades"], record["daily"], train_years)
        if (train.get("excess") or {}).get("n", 0) < min_train_trades:
            continue
        score = _score_train(train)
        if score > best_score:
            best_gate = gate_name
            best_train = train
            best_score = score
    return best_gate, best_train


def _walk_forward_mode(
    base_df: pd.DataFrame,
    sampler: msf.StableControlSampler,
    slots: int,
    min_train_years: int,
    min_train_trades: int,
    first_test_year: int,
) -> dict:
    records = _simulate_all_gates(base_df, sampler, slots)
    years = sorted(int(y) for y in base_df["year"].dropna().unique())
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
        selected_gate, train_metrics = _select_gate(records, train_years, min_train_trades)
        selected_record = records[selected_gate]
        baseline_record = records["baseline"]
        test_metrics = _trade_metrics(selected_record["trades"], selected_record["daily"], [test_year])
        baseline_metrics = _trade_metrics(baseline_record["trades"], baseline_record["daily"], [test_year])

        selected_year_trades = selected_record["trades"][selected_record["trades"]["year"] == test_year]
        baseline_year_trades = baseline_record["trades"][baseline_record["trades"]["year"] == test_year]
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
                "selected_gate": selected_gate,
                "train": train_metrics,
                "test": test_metrics,
                "baseline_test": baseline_metrics,
            }
        )

    selected_trades = pd.concat(selected_test_trades, ignore_index=True) if selected_test_trades else pd.DataFrame()
    baseline_trades = pd.concat(baseline_test_trades, ignore_index=True) if baseline_test_trades else pd.DataFrame()
    selected_daily = pd.concat(selected_test_daily).sort_index() if selected_test_daily else pd.Series(dtype=float)
    baseline_daily = pd.concat(baseline_test_daily).sort_index() if baseline_test_daily else pd.Series(dtype=float)
    fold_years = [fold["test_year"] for fold in folds]

    gate_years = {}
    for gate_name, record in records.items():
        gate_years[gate_name] = {
            str(year): _trade_metrics(record["trades"], record["daily"], [year]) for year in fold_years
        }

    return {
        "folds": folds,
        "selected_aggregate": _trade_metrics(selected_trades, selected_daily, fold_years),
        "baseline_aggregate": _trade_metrics(baseline_trades, baseline_daily, fold_years),
        "gate_years": gate_years,
    }


def _flatten_fold_rows(folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        test_excess = fold["test"].get("excess") or {}
        test_pair = fold["test"].get("pair") or {}
        base_excess = fold["baseline_test"].get("excess") or {}
        base_pair = fold["baseline_test"].get("pair") or {}
        rows.append(
            {
                "year": fold["test_year"],
                "selected_gate": fold["selected_gate"],
                "train_years": ",".join(str(y) for y in fold["train_years"]),
                "selected_n": test_excess.get("n", 0),
                "selected_excess": test_excess.get("mean_pct"),
                "selected_t": test_excess.get("t"),
                "selected_net": test_pair.get("net_mean_pct"),
                "baseline_n": base_excess.get("n", 0),
                "baseline_excess": base_excess.get("mean_pct"),
                "baseline_t": base_excess.get("t"),
                "baseline_net": base_pair.get("net_mean_pct"),
            }
        )
    return rows


def _aggregate_row(label: str, metrics: dict) -> dict:
    excess = metrics.get("excess") or {}
    pair = metrics.get("pair") or {}
    curve = metrics.get("curve") or {}
    return {
        "portfolio": label,
        "years": ",".join(str(y) for y in metrics.get("years", [])),
        "n": excess.get("n", 0),
        "excess": excess.get("mean_pct"),
        "t": excess.get("t"),
        "net": pair.get("net_mean_pct"),
        "annual": curve.get("annual_pct"),
        "mdd": curve.get("max_drawdown_pct"),
    }


def _write_report(summary: dict, output_dir: Path) -> None:
    lines = [
        "# Surge market-state walk-forward validation",
        "",
        "Selection uses only years before each test year. The gate universe is fixed by "
        "`surge_market_state_filter.py`; no threshold is learned from the test year.",
        "",
    ]
    for mode, result in summary["modes"].items():
        lines.extend([f"## {mode}", "", "### Fold Results"])
        lines.extend(
            msf.markdown_table(
                _flatten_fold_rows(result["folds"]),
                [
                    "year",
                    "selected_gate",
                    "train_years",
                    "selected_n",
                    "selected_excess",
                    "selected_t",
                    "selected_net",
                    "baseline_n",
                    "baseline_excess",
                    "baseline_t",
                    "baseline_net",
                ],
            )
        )
        lines.extend(["", "### Aggregate"])
        aggregate_rows = [
            _aggregate_row("walk_forward_selected", result["selected_aggregate"]),
            _aggregate_row("baseline_same_years", result["baseline_aggregate"]),
        ]
        lines.extend(
            msf.markdown_table(aggregate_rows, ["portfolio", "years", "n", "excess", "t", "net", "annual", "mdd"])
        )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=10, help="Slot count for the portfolio mirror.")
    parser.add_argument("--first-test-year", type=int, default=2024, help="First calendar year to include as OOS fold.")
    parser.add_argument("--min-train-years", type=int, default=2, help="Minimum prior years before a fold is active.")
    parser.add_argument("--min-train-trades", type=int, default=60, help="Minimum prior trades required for a gate.")
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market_path = msf.OUTPUT_DIR / "market_state.parquet"
    market = pd.read_parquet(market_path) if market_path.exists() else msf.load_market_state()
    cand = pd.read_parquet(msf.CAND_DIR / "candidates.parquet")
    st_intervals = spb.load_st_intervals()
    sampler = msf.StableControlSampler()

    summary = {
        "slots": args.slots,
        "first_test_year": args.first_test_year,
        "min_train_years": args.min_train_years,
        "min_train_trades": args.min_train_trades,
        "modes": {},
    }

    for mode in spb.MODES:
        print(f"[mode] {mode}")
        base_df = spb.gated_candidates(cand, mode, st_intervals)
        base_df = base_df.merge(market, left_on="dec_dt", right_on="dt", how="left")
        summary["modes"][mode] = _walk_forward_mode(
            base_df=base_df,
            sampler=sampler,
            slots=args.slots,
            min_train_years=args.min_train_years,
            min_train_trades=args.min_train_trades,
            first_test_year=args.first_test_year,
        )

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    _write_report(summary, OUTPUT_DIR)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
