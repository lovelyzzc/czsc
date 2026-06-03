"""Scan A-share daily strategy families around holding period and score style.

The production strategy script is intentionally simple. This optimizer is the
research workbench for comparing holding periods, candidate universes, and
ranking formulas before a profile is promoted into
``a_share_daily_classified_strategy.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import a_share_daily_classified_strategy as base


OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "a_share_daily_strategy_optimization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--fee-rate", type=float, default=0.0007)
    return parser.parse_args()


def add_optimizer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code", sort=False)
    df["ret5"] = g["close"].pct_change(5).astype("float32")
    df["ret10"] = g["close"].pct_change(10).astype("float32")
    df["vol5"] = g["close"].pct_change().groupby(df["ts_code"], sort=False).rolling(5, min_periods=5).std().reset_index(level=0, drop=True).astype("float32")
    df["ma20_gap"] = (df["close"] / df["ma20"] - 1).astype("float32")
    df["ma60_gap"] = (df["close"] / df["ma60"] - 1).astype("float32")
    df["ll20_gap"] = (df["close"] / df["ll20"] - 1).astype("float32")
    return df


def calc_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    log_amt = np.log(df["amt20"].clip(lower=1))
    ma20_gap_abs = df["ma20_gap"].abs()
    return {
        "baseline": 0.45 * df["ret120"] + 0.25 * df["ret60"] - 0.20 * df["ret20"] - 0.15 * df["vol20"] - 0.08 * log_amt,
        "momentum": 0.35 * df["ret120"] + 0.55 * df["ret60"] - 0.10 * df["ret20"] - 0.30 * df["vol20"] - 0.08 * log_amt,
        "lowvol": 0.45 * df["ret120"] + 0.25 * df["ret60"] - 0.20 * df["ret20"] - 0.45 * df["vol20"] - 0.08 * log_amt,
        "short_momentum": 0.20 * df["ret120"] + 0.35 * df["ret60"] + 0.35 * df["ret20"] + 0.15 * df["ret5"] - 0.45 * df["vol20"] - 0.06 * log_amt,
        "pullback": 0.35 * df["ret120"] + 0.30 * df["ret60"] - 0.55 * df["ret20"] - 0.35 * df["ret5"] - 0.25 * df["vol20"] - 0.06 * log_amt,
        "breakout": 0.30 * df["ret120"] + 0.45 * df["ret60"] + 0.20 * df["ret20"] - 0.45 * df["vol20"] - 0.06 * log_amt,
        "quality": 0.35 * df["ret120"] + 0.20 * df["ret60"] - 0.10 * df["ret20"] - 0.60 * df["vol20"] - 0.06 * log_amt - 0.04 * ma20_gap_abs,
    }


def make_family_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    hist = df["ma250"].notna()
    liquid = (df["amt20"] >= 20_000) & (df["close"] >= 2) & (df["open"] > 0)
    market = df["breadth120"] > 0.38
    no_limit = df["pct_chg"] < 9.2
    no_chase = no_limit & (df["ret20"] < 0.25) & (df["close"] <= df["ma20"] * 1.20)
    trend_loose = (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.98) & (df["ma60"] > df["ma120"] * 0.94)
    trend_mid = (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.99) & (df["ma60"] > df["ma120"] * 0.96)
    base_mask = hist & liquid & market & no_chase
    small_amt = df["amt20"].between(20_000, 200_000)
    mid_amt = df["amt20"].between(20_000, 350_000)

    return {
        "small_trend": base_mask & trend_loose & small_amt & (df["ret60"] > -0.05) & (df["ret120"] > -0.10),
        "small_trend_mid": base_mask & trend_mid & small_amt & (df["ret60"] > -0.05) & (df["ret120"] > -0.10),
        "wide_trend": base_mask & trend_loose & mid_amt & (df["ret60"] > -0.05) & (df["ret120"] > -0.10),
        "lowvol_trend": base_mask & trend_loose & mid_amt & (df["ret60"] > 0) & (df["ret120"] > -0.05) & (df["vol20"] < 0.035),
        "breakout_60": base_mask & trend_loose & mid_amt & (df["close"] > df["hh60"] * 0.98) & (df["ret60"] > 0.08),
        "trend_pullback": base_mask & trend_mid & mid_amt & (df["ret120"] > 0.08) & (df["ret60"] > -0.05) & (df["ret20"].between(-0.12, 0.12)) & (df["ma20_gap"].abs() < 0.08),
        "short_reversal": base_mask & trend_loose & mid_amt & (df["ret60"] > 0.02) & (df["ret5"] < -0.02) & (df["ret20"] < 0.12) & (df["ma60_gap"] > 0),
        "near_ma20_momentum": base_mask & trend_loose & mid_amt & (df["ret60"] > 0.05) & (df["ma20_gap"].abs() < 0.05),
    }


def calc_metrics(rets: list[float], turns: list[float]) -> dict[str, float]:
    r = pd.Series(rets, dtype="float64")
    t = pd.Series(turns, dtype="float64")
    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(r) / 252, 1 / 252)
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    std = float(r.std(ddof=0))
    sharpe = float(r.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": float((r > 0).mean()),
        "avg_turnover": float(t.mean()),
        "days": int(len(r)),
    }


def run_case(
    ret_values: np.ndarray,
    trade_dates: list[str],
    ranked_candidates: dict[str, np.ndarray],
    top_n: int,
    rebalance_days: int,
    fee_rate: float,
    cash_if_empty: bool,
) -> dict[str, float]:
    previous_idx = np.array([], dtype=np.int32)
    current_idx = np.array([], dtype=np.int32)
    rets: list[float] = []
    turns: list[float] = []
    counts: list[int] = []

    for pos in range(0, len(trade_dates) - 2):
        signal_date = trade_dates[pos]
        turnover = 0.0
        if pos % rebalance_days == 0:
            candidates = ranked_candidates.get(signal_date, np.array([], dtype=np.int32))[:top_n]
            if len(candidates) or cash_if_empty:
                new_idx = candidates.astype(np.int32, copy=False)
                turnover = calc_equal_weight_turnover(previous_idx, new_idx)
                previous_idx = new_idx
                current_idx = new_idx

        raw_ret = float(ret_values[pos + 1, current_idx].mean()) if len(current_idx) else 0.0
        rets.append(raw_ret - turnover * fee_rate)
        turns.append(turnover)
        counts.append(len(current_idx))

    metrics = calc_metrics(rets, turns)
    metrics["avg_names"] = float(np.mean(counts))
    metrics["candidate_days"] = len(ranked_candidates)
    return metrics


def calc_equal_weight_turnover(previous_idx: np.ndarray, new_idx: np.ndarray) -> float:
    if len(previous_idx) == 0 and len(new_idx) == 0:
        return 0.0
    if len(previous_idx) == 0:
        return 1.0
    if len(new_idx) == 0:
        return 1.0
    overlap = len(np.intersect1d(previous_idx, new_idx, assume_unique=False))
    return (
        overlap * abs(1 / len(new_idx) - 1 / len(previous_idx))
        + (len(new_idx) - overlap) / len(new_idx)
        + (len(previous_idx) - overlap) / len(previous_idx)
    )


def build_ranked_candidates(
    df: pd.DataFrame,
    mask: pd.Series,
    score: pd.Series,
    max_top_n: int,
    symbol_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    selected = df.loc[mask, ["trade_date", "ts_code"]].copy()
    selected["score"] = score[mask].astype(float)
    selected = selected.dropna(subset=["score"])
    if selected.empty:
        return {}

    selected = selected.sort_values(["trade_date", "score"], ascending=[True, False])
    selected = selected.groupby("trade_date", sort=True).head(max_top_n)
    ranked: dict[str, np.ndarray] = {}
    for dt, g in selected.groupby("trade_date", sort=True):
        ranked[str(dt)] = np.array([symbol_to_idx[s] for s in g["ts_code"]], dtype=np.int32)
    return ranked


def write_summary(output_dir: Path, results: pd.DataFrame, manifest: dict) -> None:
    top_calmar = results.sort_values(["calmar", "annual_return"], ascending=False).head(15)
    top_5d = results[results["rebalance_days"].eq(5)].sort_values(["calmar", "annual_return"], ascending=False).head(15)
    top_family = (
        results.sort_values(["family", "calmar", "annual_return"], ascending=[True, False, False])
        .groupby("family", as_index=False)
        .head(3)
    )

    lines = [
        "# A 股日线策略优化扫描",
        "",
        f"- 数据区间：`{manifest.get('start_date')}` 至 `{manifest.get('expected_trade_end_date', manifest.get('end_date'))}`",
        f"- 股票数：`{manifest.get('total_symbols')}`",
        "- 成本：换手成本 `0.0007`",
        "",
        "## Top Calmar",
        "",
        top_calmar.to_markdown(index=False),
        "",
        "## Top 5 日调仓",
        "",
        top_5d.to_markdown(index=False),
        "",
        "## 各策略家族 Top 3",
        "",
        top_family.to_markdown(index=False),
        "",
        "## 结论提示",
        "",
        "- `rebalance_days=5` 用于比较短持仓；若显著落后 15/20 日，说明当前日线因子更偏波段而非短线。",
        "- `cash_if_empty=False` 与正式脚本一致：若调仓日无新候选，则沿用上一期持仓。",
        "- 该扫描仍有生存者偏差，结果只用于 profile 研究，不直接作为实盘承诺。",
    ]
    (output_dir / "optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.load_manifest(args.data_dir)
    end_date = args.end_date or manifest.get("expected_trade_end_date") or manifest.get("end_date")
    cfg = base.StrategyConfig(end_date=end_date)

    print(f"[INFO] loading features from {args.data_dir}", flush=True)
    t0 = time.time()
    df = base.load_features(args.data_dir, end_date=end_date, max_symbols=args.max_symbols)
    df = base.add_market_breadth(df, cfg)
    df = add_optimizer_features(df)
    print(f"[INFO] feature rows={len(df)}, symbols={df['ts_code'].nunique()}, elapsed={time.time() - t0:.1f}s", flush=True)

    trade_dates = sorted(d for d in df["trade_date"].unique() if d >= args.start_date and (not end_date or d <= end_date))
    ret_matrix = df.pivot_table(index="trade_date", columns="ts_code", values="oo_ret", aggfunc="last")
    ret_matrix = ret_matrix.reindex(trade_dates).fillna(0).astype("float32")
    ret_values = ret_matrix.to_numpy(dtype=np.float32, copy=False)
    symbol_to_idx = {symbol: i for i, symbol in enumerate(ret_matrix.columns)}
    scores = calc_scores(df)
    families = make_family_masks(df)

    family_scores: dict[str, list[str]] = {
        "small_trend": ["baseline", "momentum", "lowvol", "quality"],
        "small_trend_mid": ["baseline", "momentum", "lowvol", "quality"],
        "wide_trend": ["baseline", "momentum", "lowvol", "quality"],
        "lowvol_trend": ["lowvol", "quality", "momentum"],
        "breakout_60": ["breakout", "momentum", "lowvol"],
        "trend_pullback": ["pullback", "quality", "momentum"],
        "short_reversal": ["pullback", "short_momentum", "quality"],
        "near_ma20_momentum": ["momentum", "short_momentum", "lowvol"],
    }

    rows: list[dict] = []
    case_count = 0
    max_top_n = 300
    for family, mask in families.items():
        candidate_rows = int(mask.sum())
        if candidate_rows < 30_000:
            continue
        for score_name in family_scores[family]:
            ranked_candidates = build_ranked_candidates(df, mask, scores[score_name], max_top_n=max_top_n, symbol_to_idx=symbol_to_idx)
            if not ranked_candidates:
                continue
            for top_n in [60, 80, 120, 160, 220, 300]:
                for rebalance_days in [5, 10, 15, 20]:
                    for cash_if_empty in [False, True]:
                        metrics = run_case(
                            ret_values=ret_values,
                            trade_dates=trade_dates,
                            ranked_candidates=ranked_candidates,
                            top_n=top_n,
                            rebalance_days=rebalance_days,
                            fee_rate=args.fee_rate,
                            cash_if_empty=cash_if_empty,
                        )
                        rows.append({
                            "family": family,
                            "score": score_name,
                            "top_n": top_n,
                            "rebalance_days": rebalance_days,
                            "cash_if_empty": cash_if_empty,
                            "candidate_rows": candidate_rows,
                            **metrics,
                        })
                        case_count += 1
        print(f"[INFO] family={family}, cases={case_count}, elapsed={time.time() - t0:.1f}s", flush=True)

    results = pd.DataFrame(rows)
    results = results.sort_values(["calmar", "annual_return"], ascending=False).reset_index(drop=True)
    results.to_csv(args.output_dir / "optimization_results.csv", index=False)
    write_summary(args.output_dir, results, manifest)
    payload = {
        "manifest": manifest,
        "cases": len(results),
        "top_calmar": results.head(20).to_dict("records"),
        "top_5d": results[results["rebalance_days"].eq(5)].head(20).to_dict("records"),
    }
    (args.output_dir / "optimization_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default), encoding="utf-8")

    cols = [
        "family", "score", "top_n", "rebalance_days", "cash_if_empty",
        "total_return", "annual_return", "sharpe", "max_drawdown", "calmar",
        "avg_turnover", "avg_names", "candidate_days",
    ]
    print("[INFO] Top Calmar")
    print(results[cols].head(20).to_string(index=False))
    print("[INFO] Top 5d")
    print(results[results["rebalance_days"].eq(5)][cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
