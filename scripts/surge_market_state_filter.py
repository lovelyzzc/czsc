"""Market-state filter iteration for the surge-regime strategy.

This script is the next iteration after SURGE_REGIME_AUDIT_2026-06-10.md.
It keeps the existing signal definition fixed and evaluates causal market
state filters on top of the dumped candidates:

1. Candidate-level buckets: group already-gated candidates by market state and
   measure gross excess versus same-date/same-amount-decile random controls.
2. Portfolio-level mirror: apply simple a-priori state gates before the existing
   priority sort and slot simulator, then reuse the same random-control verdict.

No FSM or surge_onset changes are made here, so surge_candidates_dump.py does
not need to be rerun unless upstream candidate logic changes.

    uv run --no-sync python scripts/surge_market_state_filter.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import surge_portfolio_backtest as spb

CAND_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_market_state_filter"
FEATURES = [
    "mkt_ret20_median",
    "mkt_ret20_pos_ratio",
    "high20_ratio",
    "small_minus_large_ret20",
    "ew_index_above_ma20",
]


class StableControlSampler(spb.ControlSampler):
    """Order-invariant K-sample control, keyed by each trade identity."""

    @staticmethod
    def _seed_for(trade) -> int:
        parts = [
            str(trade.symbol),
            pd.Timestamp(trade.dec_dt).strftime("%Y%m%d"),
            pd.Timestamp(trade.entry_dt).strftime("%Y%m%d"),
            pd.Timestamp(trade.exit_dt).strftime("%Y%m%d"),
        ]
        digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def excess_for(self, trade) -> float:
        dec_dt, entry_dt, exit_dt = trade.dec_dt, trade.entry_dt, trade.exit_dt
        if dec_dt not in self.decile.index or entry_dt not in self.dates or exit_dt not in self.dates:
            return np.nan
        row = self.decile.loc[dec_dt]
        my_decile = row.get(trade.symbol)
        if pd.isna(my_decile):
            return np.nan
        pool = row.index[(row == my_decile) & (row.index != trade.symbol)]
        if len(pool) < spb.CONTROL_MIN_VALID:
            return np.nan
        rng = np.random.default_rng(self._seed_for(trade))
        pick = rng.choice(pool, size=min(spb.CONTROL_K, len(pool)), replace=False)
        open_prices = self.open_w.loc[entry_dt, pick].to_numpy(dtype=float)
        close_prices = self.close_w.loc[exit_dt, pick].to_numpy(dtype=float)
        returns = close_prices / open_prices - 1
        returns = returns[np.isfinite(returns)]
        if len(returns) < spb.CONTROL_MIN_VALID:
            return np.nan
        return trade.ret_gross_pct - float(np.median(returns)) * 100


def t_stat(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std()
    return float(values.mean() / (std / np.sqrt(len(values)))) if std > 0 else np.nan


def round_float(value: float, digits: int = 2) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def compute_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Build causal market-state variables known at each decision close."""
    panel = panel.sort_values(["symbol", "dt"]).copy()
    panel["ret1"] = panel.groupby("symbol")["close"].pct_change()
    panel["ret20"] = panel.groupby("symbol")["close"].pct_change(20)
    panel["high20"] = panel.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    panel["is_high20"] = (panel["close"] >= panel["high20"]).astype(float)

    amount_rank = panel.groupby("dt")["amount_e"].rank(pct=True)
    panel["amount_decile"] = np.floor(amount_rank.mul(10).clip(upper=9.999))

    state = panel.groupby("dt").agg(
        mkt_ret20_median=("ret20", "median"),
        mkt_ret20_pos_ratio=("ret20", lambda x: float((x > 0).mean())),
        high20_ratio=("is_high20", "mean"),
        ew_ret1=("ret1", "mean"),
    )
    ew_index = (1 + state["ew_ret1"].fillna(0)).cumprod()
    state["ew_index"] = ew_index
    state["ew_index_ma20"] = ew_index.rolling(20, min_periods=20).mean()
    state["ew_index_above_ma20"] = (state["ew_index"] > state["ew_index_ma20"]).astype(float)

    small = panel[panel["amount_decile"] <= 1].groupby("dt")["ret20"].median()
    large = panel[panel["amount_decile"] >= 8].groupby("dt")["ret20"].median()
    state["small_minus_large_ret20"] = small - large
    return state.reset_index()


def load_market_state() -> pd.DataFrame:
    panel = pd.read_parquet(CAND_DIR / "panel.parquet", columns=["symbol", "dt", "close", "amount_e"])
    return compute_market_state(panel)


def gate_specs() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        ("baseline", "no market-state filter", lambda df: pd.Series(True, index=df.index)),
        (
            "mkt_ret20_median_gt_0",
            "cross-sectional median 20-day return > 0",
            lambda df: df["mkt_ret20_median"] > 0,
        ),
        (
            "mkt_ret20_pos_ratio_gt_055",
            "share of stocks with positive 20-day return > 55%",
            lambda df: df["mkt_ret20_pos_ratio"] > 0.55,
        ),
        (
            "high20_ratio_gt_012",
            "share of stocks making a 20-day high > 12%",
            lambda df: df["high20_ratio"] > 0.12,
        ),
        (
            "ew_index_above_ma20",
            "equal-weight market index above its MA20",
            lambda df: df["ew_index_above_ma20"] > 0,
        ),
        (
            "breadth_and_high20",
            "positive-ret20 share > 55% and 20-day-high share > 12%",
            lambda df: (df["mkt_ret20_pos_ratio"] > 0.55) & (df["high20_ratio"] > 0.12),
        ),
        (
            "breadth_and_index",
            "positive-ret20 share > 55% and equal-weight index above MA20",
            lambda df: (df["mkt_ret20_pos_ratio"] > 0.55) & (df["ew_index_above_ma20"] > 0),
        ),
        (
            "high20_and_index",
            "20-day-high share > 12% and equal-weight index above MA20",
            lambda df: (df["high20_ratio"] > 0.12) & (df["ew_index_above_ma20"] > 0),
        ),
        (
            "ret20_high20_index",
            "median ret20 > 0, high20 share > 12%, and equal-weight index above MA20",
            lambda df: (df["mkt_ret20_median"] > 0) & (df["high20_ratio"] > 0.12) & (df["ew_index_above_ma20"] > 0),
        ),
    ]


def add_excess(df: pd.DataFrame, sampler: StableControlSampler) -> pd.DataFrame:
    excess = [sampler.excess_for(row) for row in df.itertuples()]
    return df.assign(excess_pct=excess)


def excess_stats(df: pd.DataFrame, sampler: StableControlSampler) -> list[dict]:
    df = add_excess(df, sampler)
    groups = [
        ("ALL", df),
        ("IS", df[df["seg"] == "train"]),
        ("OOS", df[df["seg"] == "test"]),
    ]
    groups.extend((str(year), group) for year, group in df.groupby("year"))

    rows = []
    for tag, group in groups:
        values = group["excess_pct"].dropna()
        if len(values) < 30:
            continue
        rows.append(
            {
                "segment": tag,
                "n": int(len(values)),
                "excess_mean_pct": round_float(values.mean()),
                "excess_median_pct": round_float(values.median()),
                "t": round_float(t_stat(values)),
                "positive_excess_pct": round_float((values > 0).mean() * 100, 1),
            }
        )
    return rows


def pair_stats_ascii(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {}
    returns = trades["ret_net_pct"].to_numpy(dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    return {
        "trades": int(len(returns)),
        "win_rate_pct": round_float(len(wins) / len(returns) * 100, 1),
        "net_mean_pct": round_float(np.mean(returns)),
        "net_median_pct": round_float(np.median(returns)),
        "gross_mean_pct": round_float(trades["ret_gross_pct"].mean()),
        "avg_hold_days": round_float(trades["hold_days"].mean(), 1),
        "profit_loss_ratio": round_float(abs(wins.mean() / losses.mean())) if len(wins) and len(losses) else None,
    }


def curve_stats_ascii(daily: pd.Series) -> dict:
    if len(daily) < 20:
        return {}
    equity = (1 + daily).cumprod()
    total = equity.iloc[-1] - 1
    years = len(daily) / 252
    annual = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    std = daily.std()
    sharpe = daily.mean() / std * np.sqrt(252) if std > 0 else np.nan
    max_dd = (1 - equity / equity.cummax()).max()
    return {
        "annual_pct": round_float(annual * 100, 1),
        "sharpe": round_float(sharpe),
        "max_drawdown_pct": round_float(max_dd * 100, 1),
        "calmar": round_float(annual / max_dd) if max_dd > 0 else None,
        "days": int(len(daily)),
    }


def candidate_bucket_report(df: pd.DataFrame, feature: str, segment: str = "test") -> list[dict]:
    scoped = df[df["seg"] == segment].copy()
    scoped = scoped[scoped[feature].notna()]
    if len(scoped) < 30:
        return []
    if scoped[feature].nunique(dropna=True) <= 2:
        scoped["bucket"] = scoped[feature].map(lambda x: str(int(x)))
    else:
        scoped["bucket"] = pd.qcut(scoped[feature], 5, duplicates="drop")
    rows = []
    for bucket, group in scoped.groupby("bucket", observed=True):
        values = group["excess_pct"].dropna()
        if len(values) < 30:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "n": int(len(values)),
                "excess_mean_pct": round_float(values.mean()),
                "t": round_float(t_stat(values)),
                "net_mean_pct": round_float(group["ret_net_pct"].mean()),
            }
        )
    return rows


def simulate_gate(
    base_df: pd.DataFrame,
    gate_name: str,
    gate_rule: str,
    mask: pd.Series,
    sampler: StableControlSampler,
    slots: int,
) -> dict:
    filtered = base_df[mask].sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)
    trades, daily = spb.simulate_slots(filtered, slots)
    excess_rows = excess_stats(trades, sampler) if len(trades) else []
    oos_excess = next((row for row in excess_rows if row["segment"] == "OOS"), {})
    is_excess = next((row for row in excess_rows if row["segment"] == "IS"), {})
    oos_trades = trades[trades["seg"] == "test"] if len(trades) else trades
    is_trades = trades[trades["seg"] == "train"] if len(trades) else trades
    oos_daily = daily[daily.index > spb.TRAIN_END] if len(daily) else daily
    is_daily = daily[daily.index <= spb.TRAIN_END] if len(daily) else daily
    verdict = (
        "pass"
        if oos_excess and (oos_excess.get("excess_mean_pct") or 0) > 0 and abs(oos_excess.get("t") or 0) >= 2
        else "fail"
    )
    return {
        "gate": gate_name,
        "rule": gate_rule,
        "verdict": verdict,
        "candidates": int(len(filtered)),
        "trades": int(len(trades)),
        "oos_trades": int(len(oos_trades)),
        "is_excess": is_excess,
        "oos_excess": oos_excess,
        "is_pair": pair_stats_ascii(is_trades),
        "oos_pair": pair_stats_ascii(oos_trades),
        "is_curve": curve_stats_ascii(is_daily),
        "oos_curve": curve_stats_ascii(oos_daily),
    }


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(summary: dict, output_dir: Path) -> None:
    lines = [
        "# Surge market-state filter iteration",
        "",
        "This report evaluates causal market-state filters on the fixed candidate dump.",
        "Pass criterion: OOS gross excess versus same-date/same-amount-decile controls is positive with |t| >= 2.",
        "",
        "## Portfolio gates",
    ]
    for mode, rows in summary["portfolio_gates"].items():
        lines.extend(["", f"### {mode}"])
        flat_rows = []
        for row in rows:
            oos_excess = row.get("oos_excess") or {}
            oos_pair = row.get("oos_pair") or {}
            oos_curve = row.get("oos_curve") or {}
            flat_rows.append(
                {
                    "gate": row["gate"],
                    "verdict": row["verdict"],
                    "oos_n": row["oos_trades"],
                    "oos_excess": oos_excess.get("excess_mean_pct"),
                    "oos_t": oos_excess.get("t"),
                    "oos_net": oos_pair.get("net_mean_pct"),
                    "oos_ann": oos_curve.get("annual_pct"),
                    "oos_mdd": oos_curve.get("max_drawdown_pct"),
                }
            )
        lines.extend(
            markdown_table(
                flat_rows,
                ["gate", "verdict", "oos_n", "oos_excess", "oos_t", "oos_net", "oos_ann", "oos_mdd"],
            )
        )

    lines.extend(["", "## Candidate buckets"])
    for mode, features in summary["candidate_buckets"].items():
        lines.extend(["", f"### {mode}"])
        for feature, rows in features.items():
            lines.extend(["", f"#### {feature}"])
            lines.extend(markdown_table(rows, ["bucket", "n", "excess_mean_pct", "t", "net_mean_pct"]))

    selected = summary.get("selected_gates", {})
    lines.extend(["", "## Selected historical passes"])
    for mode, rows in selected.items():
        lines.extend(["", f"### {mode}"])
        if not rows:
            lines.append("No gate passed the OOS criterion.")
        else:
            lines.extend(markdown_table(rows, ["gate", "oos_excess", "oos_t", "oos_net", "oos_ann", "oos_mdd"]))

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=10, help="Slot count for the portfolio mirror.")
    parser.add_argument("--skip-candidate-buckets", action="store_true", help="Only run portfolio gate tests.")
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[market] computing causal market-state features")
    market = load_market_state()
    market.to_parquet(OUTPUT_DIR / "market_state.parquet", index=False)

    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    st_intervals = spb.load_st_intervals()
    sampler = StableControlSampler()

    summary: dict = {
        "slots": args.slots,
        "market_yearly": market.assign(year=market["dt"].dt.year)
        .groupby("year")[FEATURES]
        .mean()
        .round(4)
        .reset_index()
        .to_dict("records"),
        "candidate_buckets": {},
        "portfolio_gates": {},
        "selected_gates": {},
    }

    for mode in spb.MODES:
        print(f"[mode] {mode}")
        base_df = spb.gated_candidates(cand, mode, st_intervals)
        base_df = base_df.merge(market, left_on="dec_dt", right_on="dt", how="left")

        if not args.skip_candidate_buckets:
            print("  candidate buckets")
            candidate_df = add_excess(base_df, sampler)
            summary["candidate_buckets"][mode] = {
                feature: candidate_bucket_report(candidate_df, feature) for feature in FEATURES
            }
        else:
            summary["candidate_buckets"][mode] = {}

        print("  portfolio gates")
        gate_rows = []
        for gate_name, rule, fn in gate_specs():
            gate_rows.append(simulate_gate(base_df, gate_name, rule, fn(base_df), sampler, args.slots))
        summary["portfolio_gates"][mode] = gate_rows

        selected = []
        for row in gate_rows:
            if row["gate"] == "baseline" or row["verdict"] != "pass":
                continue
            oos_excess = row.get("oos_excess") or {}
            oos_pair = row.get("oos_pair") or {}
            oos_curve = row.get("oos_curve") or {}
            selected.append(
                {
                    "gate": row["gate"],
                    "oos_excess": oos_excess.get("excess_mean_pct"),
                    "oos_t": oos_excess.get("t"),
                    "oos_net": oos_pair.get("net_mean_pct"),
                    "oos_ann": oos_curve.get("annual_pct"),
                    "oos_mdd": oos_curve.get("max_drawdown_pct"),
                }
            )
        summary["selected_gates"][mode] = selected

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary, OUTPUT_DIR)

    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
