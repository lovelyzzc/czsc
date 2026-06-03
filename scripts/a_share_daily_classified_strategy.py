"""A-share daily classified strategy research.

This script reads the local qfq daily cache built by ``sync_a_stock_daily.py``,
assigns each stock-date to one mutually exclusive planning class, runs a simple
open-to-open portfolio backtest, and writes the latest live planning table.

Run from outside the local czsc source tree when using the PyPI czsc package:

    cd /home/lovelyzzc/the-way
    uv run --no-project --with czsc --with pandas --with pyarrow --with numpy \
        python /home/lovelyzzc/czsc/scripts/a_share_daily_classified_strategy.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "a_share_daily_classified_strategy"

BREAKOUT_CLASS = "A1_BREAKOUT_HOLD"
CORE_CLASS = "A2_SMALL_TREND_CORE"
PULLBACK_CLASS = "A3_TREND_PULLBACK"

ACTION_MAP = {
    "S0_SUSPENDED_OR_STALE": ("avoid", "suspended_pool", "suspended"),
    "D0_INSUFFICIENT_HISTORY": ("avoid", "avoid_pool", "insufficient_history"),
    "D1_ILLIQUID_OR_LOW_PRICE": ("avoid", "avoid_pool", "liquidity"),
    "C1_MARKET_WEAK": ("reduce_or_avoid", "avoid_pool", "weak_market"),
    "C2_LIMIT_UP_BLOCKED": ("avoid_chase", "avoid_pool", "overheated_or_limit"),
    "C3_BEAR_TREND": ("reduce_or_avoid", "avoid_pool", "bear_trend"),
    BREAKOUT_CLASS: ("buy_candidate", "satellite_buy_pool", "breakout"),
    CORE_CLASS: ("buy_candidate", "core_buy_pool", "normal"),
    PULLBACK_CLASS: ("buy_candidate", "satellite_buy_pool", "pullback"),
    "B1_TREND_HOLD": ("hold", "hold_pool", "normal"),
    "B2_REPAIR_WATCH": ("watch", "watch_pool", "repair"),
    "D2_NO_EDGE": ("avoid", "avoid_pool", "no_edge"),
}


@dataclass(frozen=True)
class StrategyConfig:
    start_date: str = "20210101"
    end_date: str | None = None
    top_n: int = 120
    rebalance_days: int = 20
    fee_rate: float = 0.0007
    min_amount_20: float = 20_000.0
    max_small_amount_20: float = 200_000.0
    market_breadth_floor: float = 0.38
    czsc_snapshot_limit: int = 80
    profile: str = "small_trend_mid_momentum_v3"
    score_profile: str = "momentum"
    core_trend_profile: str = "mid"
    market_policy: str = "carry"
    cap_policy: str = "none"


PROFILE_PRESETS = {
    "small_trend_core_v1": {
        "top_n": 160,
        "rebalance_days": 10,
        "score_profile": "baseline",
        "core_trend_profile": "loose",
    },
    "small_trend_momentum_v2": {
        "top_n": 120,
        "rebalance_days": 20,
        "score_profile": "momentum",
        "core_trend_profile": "loose",
    },
    "small_trend_mid_momentum_v3": {
        "top_n": 120,
        "rebalance_days": 20,
        "score_profile": "momentum",
        "core_trend_profile": "mid",
    },
    "small_trend_lowvol_v2": {
        "top_n": 160,
        "rebalance_days": 20,
        "score_profile": "lowvol",
        "core_trend_profile": "loose",
        "market_policy": "carry",
        "cap_policy": "none",
    },
    "small_trend_lowvol_risk_v4": {
        "top_n": 220,
        "rebalance_days": 20,
        "score_profile": "lowvol",
        "core_trend_profile": "mid",
        "market_policy": "c1_80_cash",
        "cap_policy": "growth60",
    },
}

for preset in PROFILE_PRESETS.values():
    preset.setdefault("market_policy", "carry")
    preset.setdefault("cap_policy", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default=StrategyConfig.profile)
    parser.add_argument("--start-date", default=StrategyConfig.start_date)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=None, help="Override profile top_n.")
    parser.add_argument("--rebalance-days", type=int, default=None, help="Override profile rebalance days.")
    parser.add_argument("--fee-rate", type=float, default=StrategyConfig.fee_rate)
    parser.add_argument("--max-symbols", type=int, default=None, help="Smoke-test limit.")
    parser.add_argument("--skip-czsc-snapshot", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, end_date: str | None) -> StrategyConfig:
    preset = PROFILE_PRESETS[args.profile]
    return StrategyConfig(
        start_date=args.start_date,
        end_date=end_date,
        top_n=args.top_n or preset["top_n"],
        rebalance_days=args.rebalance_days or preset["rebalance_days"],
        fee_rate=args.fee_rate,
        profile=args.profile,
        score_profile=preset["score_profile"],
        core_trend_profile=preset["core_trend_profile"],
        market_policy=preset["market_policy"],
        cap_policy=preset["cap_policy"],
    )


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if pd.isna(obj):
        return None
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def read_symbol_features(path: Path, end_date: str | None) -> pd.DataFrame | None:
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "amount"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception as exc:
        print(f"[WARN] read failed: {path.name}: {exc}", file=sys.stderr)
        return None

    if df.empty:
        return None
    if end_date:
        df = df[df["trade_date"].astype(str) <= end_date]
    if df.empty:
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "pct_chg", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    close = df["close"]
    amount = df["amount"]
    ret1 = close.pct_change()

    df["bar_no"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["ma20"] = close.rolling(20, min_periods=20).mean()
    df["ma60"] = close.rolling(60, min_periods=60).mean()
    df["ma120"] = close.rolling(120, min_periods=120).mean()
    df["ma250"] = close.rolling(250, min_periods=250).mean()
    df["hh60"] = close.rolling(60, min_periods=60).max()
    df["ll20"] = close.rolling(20, min_periods=20).min()
    df["ret20"] = close.pct_change(20)
    df["ret60"] = close.pct_change(60)
    df["ret120"] = close.pct_change(120)
    df["vol20"] = ret1.rolling(20, min_periods=20).std()
    df["amt20"] = amount.rolling(20, min_periods=20).mean()
    df["next_open"] = df["open"].shift(-1)
    df["oo_ret"] = safe_divide(df["next_open"], df["open"]) - 1

    float_cols = [
        "open", "high", "low", "close", "pct_chg", "amount", "ma20", "ma60",
        "ma120", "ma250", "hh60", "ll20", "ret20", "ret60", "ret120",
        "vol20", "amt20", "next_open", "oo_ret",
    ]
    df[float_cols] = df[float_cols].astype("float32")
    return df


def load_features(data_dir: Path, end_date: str | None, max_symbols: int | None) -> pd.DataFrame:
    paths = sorted(p for p in data_dir.glob("*.parquet"))
    if max_symbols:
        paths = paths[:max_symbols]
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames: list[pd.DataFrame] = []
    total = len(paths)
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        frame = read_symbol_features(path, end_date=end_date)
        if frame is not None:
            frames.append(frame)
        if i % 500 == 0 or i == total:
            print(f"[INFO] loaded {i}/{total} files, frames={len(frames)}, elapsed={time.time() - t0:.1f}s")

    if not frames:
        raise RuntimeError("No usable symbol data loaded.")

    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df


def add_market_breadth(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    liquid = (
        df["ma120"].notna()
        & (df["amt20"] >= cfg.min_amount_20)
        & (df["close"] >= 2)
        & (df["open"] > 0)
    )
    up120 = liquid & (df["close"] > df["ma120"])
    breadth = up120.groupby(df["trade_date"]).sum() / liquid.groupby(df["trade_date"]).sum().replace(0, np.nan)
    df["breadth120"] = df["trade_date"].map(breadth).astype("float32")
    return df


def calc_score_core(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    ma20_gap = (df["close"] / df["ma20"] - 1).abs()
    log_amt = np.log(df["amt20"].clip(lower=1))

    if cfg.score_profile == "baseline":
        score = (
            0.45 * df["ret120"]
            + 0.25 * df["ret60"]
            - 0.20 * df["ret20"]
            - 0.15 * df["vol20"]
            - 0.08 * log_amt
        )
    elif cfg.score_profile == "momentum":
        score = (
            0.35 * df["ret120"]
            + 0.55 * df["ret60"]
            - 0.10 * df["ret20"]
            - 0.30 * df["vol20"]
            - 0.08 * log_amt
        )
    elif cfg.score_profile == "lowvol":
        score = (
            0.45 * df["ret120"]
            + 0.25 * df["ret60"]
            - 0.20 * df["ret20"]
            - 0.45 * df["vol20"]
            - 0.08 * log_amt
        )
    else:
        raise ValueError(f"unknown score_profile: {cfg.score_profile}")

    return score.astype("float32")


def classify_rows(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = add_market_breadth(df, cfg)

    hist = df["ma250"].notna()
    liquid = (df["amt20"] >= cfg.min_amount_20) & (df["close"] >= 2) & (df["open"] > 0)
    market_ok = df["breadth120"] > cfg.market_breadth_floor
    trend_loose = (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.98) & (df["ma60"] > df["ma120"] * 0.94)
    trend_mid = (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.99) & (df["ma60"] > df["ma120"] * 0.96)
    trend = (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]) & (df["ma60"] > df["ma120"] * 0.96)
    bear = hist & liquid & market_ok & ((df["close"] < df["ma120"]) | (df["ma20"] < df["ma60"] * 0.96))
    blocked = hist & liquid & market_ok & ((df["pct_chg"] >= 9.2) | (df["ret20"] > 0.25) | (df["close"] > df["ma20"] * 1.20))

    no_chase = (df["pct_chg"] < 9.2) & (df["ret20"] < 0.25) & (df["close"] <= df["ma20"] * 1.20)
    breakout_hold = (
        hist & liquid & market_ok & trend_loose & no_chase
        & (df["close"] > df["hh60"] * 0.98)
        & (df["ret60"] > 0.08)
    )
    pullback = (
        hist & liquid & market_ok & trend & no_chase
        & (df["ret120"] > 0.08)
        & (df["ret60"] > -0.05)
        & (df["ret20"] > -0.10)
        & (df["ret20"] < 0.12)
        & ((df["close"] / df["ma20"] - 1).abs() < 0.08)
    )
    core_trend = trend_mid if cfg.core_trend_profile == "mid" else trend_loose
    small_trend = (
        hist & liquid & market_ok & core_trend & no_chase
        & (df["ret60"] > -0.05)
        & (df["ret120"] > -0.10)
        & (df["amt20"] <= cfg.max_small_amount_20)
    )
    hold = (
        hist & liquid & market_ok
        & (df["close"] > df["ma120"])
        & (df["ma20"] > df["ma60"] * 0.96)
        & (df["ret60"] > -0.05)
    )
    repair = (
        hist & liquid & market_ok
        & (df["close"] > df["ma250"])
        & (df["close"] < df["ma60"])
        & (df["ret20"] > -0.12)
    )

    df["setup_class"] = "D2_NO_EDGE"
    df.loc[repair, "setup_class"] = "B2_REPAIR_WATCH"
    df.loc[hold, "setup_class"] = "B1_TREND_HOLD"
    df.loc[pullback, "setup_class"] = PULLBACK_CLASS
    df.loc[breakout_hold, "setup_class"] = BREAKOUT_CLASS
    df.loc[small_trend, "setup_class"] = CORE_CLASS

    df.loc[bear & df["setup_class"].eq("D2_NO_EDGE"), "setup_class"] = "C3_BEAR_TREND"
    df.loc[hist & liquid & ~market_ok, "setup_class"] = "C1_MARKET_WEAK"
    df.loc[blocked, "setup_class"] = "C2_LIMIT_UP_BLOCKED"
    df.loc[hist & ~liquid, "setup_class"] = "D1_ILLIQUID_OR_LOW_PRICE"
    df.loc[~hist, "setup_class"] = "D0_INSUFFICIENT_HISTORY"

    ma20_gap = (df["close"] / df["ma20"] - 1).abs()
    df["score_core"] = calc_score_core(df, cfg)
    df["score_aggressive"] = (
        0.45 * df["ret120"]
        + 0.25 * df["ret60"]
        - 0.20 * df["ret20"]
        - 0.15 * df["vol20"]
        - 0.08 * np.log1p(df["amt20"].clip(lower=0))
    ).astype("float32")

    mapped = df["setup_class"].map(ACTION_MAP)
    df["action"] = mapped.map(lambda x: x[0])
    df["plan_bucket"] = mapped.map(lambda x: x[1])
    df["risk_class"] = mapped.map(lambda x: x[2])

    df["is_core_candidate"] = df["setup_class"].eq(CORE_CLASS)
    df["is_benchmark_member"] = hist & liquid
    stop_gap = np.clip(df["vol20"].fillna(0.04) * 2.0, 0.06, 0.18)
    df["entry_ref"] = df["close"]
    df["stop_ref"] = np.minimum(df["ma60"], df["close"] * (1 - stop_gap)).astype("float32")
    df["invalid_condition"] = "close < ma60 or ma20 < ma60 or breadth120 <= 0.38"
    return df


def make_next_date_map(trade_dates: list[str]) -> dict[str, str]:
    return {trade_dates[i]: trade_dates[i + 1] for i in range(len(trade_dates) - 1)}


def calc_metrics(ret: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    ret = ret.fillna(0).astype(float)
    if ret.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "days": 0,
        }

    equity = (1 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(ret) / 252, 1 / 252)
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1)
    std = float(ret.std(ddof=0))
    sharpe = float(ret.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    win_rate = float((ret > 0).mean())
    avg_turnover = float(turnover.fillna(0).mean()) if turnover is not None and not turnover.empty else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "days": int(len(ret)),
    }


def prepare_candidate_map(df: pd.DataFrame, cfg: StrategyConfig) -> dict[str, list[str]]:
    core = df.loc[
        df["is_core_candidate"] & (df["trade_date"] >= cfg.start_date),
        ["trade_date", "ts_code", "score_core"],
    ].dropna(subset=["score_core"])
    scan_limit = max(cfg.top_n, 600 if cfg.cap_policy != "none" else cfg.top_n)
    candidates: dict[str, list[str]] = {}
    for dt, g in core.groupby("trade_date", sort=True):
        ranked = g.sort_values("score_core", ascending=False).head(scan_limit)
        top = select_with_cap(ranked, cfg)
        candidates[str(dt)] = top["ts_code"].tolist()
    return candidates


def benchmark_returns(df: pd.DataFrame, trade_dates: list[str]) -> pd.Series:
    next_map = make_next_date_map(trade_dates)
    bench = df.loc[df["is_benchmark_member"], ["trade_date", "ts_code"]].copy()
    bench["exec_date"] = bench["trade_date"].map(next_map)
    bench = bench.dropna(subset=["exec_date"])
    ret = df[["trade_date", "ts_code", "oo_ret"]].rename(columns={"trade_date": "exec_date"})
    merged = bench.merge(ret, on=["exec_date", "ts_code"], how="left")
    out = merged.groupby("exec_date")["oo_ret"].mean().fillna(0).astype(float)
    return out


def market_exposure_by_date(df: pd.DataFrame, cfg: StrategyConfig, trade_dates: list[str]) -> pd.Series:
    if cfg.market_policy == "carry":
        return pd.Series(1.0, index=trade_dates, dtype="float32")

    c1_pct = (
        df["setup_class"].eq("C1_MARKET_WEAK").groupby(df["trade_date"]).sum()
        / df.groupby("trade_date").size().replace(0, np.nan)
    )
    c1_pct = c1_pct.reindex(trade_dates).fillna(1.0)

    if cfg.market_policy == "c1_80_cash":
        exposure = (c1_pct <= 0.80).astype("float32")
    elif cfg.market_policy == "c1_step":
        exposure = pd.Series(np.where(c1_pct <= 0.60, 1.0, np.where(c1_pct <= 0.80, 0.5, 0.0)), index=c1_pct.index, dtype="float32")
    else:
        raise ValueError(f"unknown market_policy: {cfg.market_policy}")
    return exposure


def board_code(symbol: str) -> str:
    if symbol.endswith(".BJ") or symbol.startswith(("43", "83", "87", "88", "92")):
        return "bj"
    if symbol.startswith("688"):
        return "star"
    if symbol.startswith(("300", "301")):
        return "chinext"
    return "other"


def select_with_cap(candidates: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    if cfg.cap_policy == "none" or candidates.empty:
        return candidates.head(cfg.top_n)

    if cfg.cap_policy == "growth60":
        growth_limit = max(1, int(cfg.top_n * 0.60))
        bj_limit = None
    elif cfg.cap_policy == "bj30_growth70":
        growth_limit = max(1, int(cfg.top_n * 0.70))
        bj_limit = max(1, int(cfg.top_n * 0.30))
    elif cfg.cap_policy == "bj30":
        growth_limit = None
        bj_limit = max(1, int(cfg.top_n * 0.30))
    else:
        raise ValueError(f"unknown cap_policy: {cfg.cap_policy}")

    rows: list[int] = []
    growth_count = 0
    bj_count = 0
    for idx, row in candidates.iterrows():
        board = board_code(str(row["ts_code"]))
        is_growth = board in {"bj", "star", "chinext"}
        if growth_limit is not None and is_growth and growth_count >= growth_limit:
            continue
        if bj_limit is not None and board == "bj" and bj_count >= bj_limit:
            continue

        rows.append(idx)
        if is_growth:
            growth_count += 1
        if board == "bj":
            bj_count += 1
        if len(rows) >= cfg.top_n:
            break

    return candidates.loc[rows]


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trade_dates = sorted(d for d in df["trade_date"].unique() if (not cfg.end_date or d <= cfg.end_date))
    trade_dates = [d for d in trade_dates if d >= cfg.start_date]
    if len(trade_dates) < cfg.rebalance_days + 3:
        raise RuntimeError("Not enough trade dates for backtest.")

    ret_matrix = df.pivot_table(index="trade_date", columns="ts_code", values="oo_ret", aggfunc="last")
    ret_matrix = ret_matrix.reindex(trade_dates).fillna(0).astype("float32")
    candidate_map = prepare_candidate_map(df, cfg)
    bench_ret = benchmark_returns(df, trade_dates)
    exposure_by_date = market_exposure_by_date(df, cfg, trade_dates)

    rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    current_symbols: list[str] = []

    for pos in range(0, len(trade_dates) - 2):
        signal_date = trade_dates[pos]
        exec_date = trade_dates[pos + 1]
        is_rebalance = pos % cfg.rebalance_days == 0

        if is_rebalance:
            candidates = candidate_map.get(signal_date, [])
            if candidates:
                current_symbols = candidates

        exposure = float(exposure_by_date.get(signal_date, 1.0))
        if current_symbols and exposure > 0:
            weight = exposure / len(current_symbols)
            new_weights = {s: weight for s in current_symbols}
            raw_ret = float(ret_matrix.loc[exec_date, current_symbols].mean()) * exposure
        else:
            new_weights = {}
            raw_ret = 0.0

        turnover = sum(abs(new_weights.get(s, 0.0) - previous_weights.get(s, 0.0)) for s in set(new_weights) | set(previous_weights))
        fee = turnover * cfg.fee_rate
        strategy_ret = raw_ret - fee
        previous_weights = new_weights

        rows.append({
            "trade_date": exec_date,
            "source_signal_date": signal_date,
            "strategy_ret": strategy_ret,
            "strategy_raw_ret": raw_ret,
            "benchmark_ret": float(bench_ret.get(exec_date, 0.0)),
            "turnover": turnover,
            "fee": fee,
            "exposure": exposure if current_symbols else 0.0,
            "holding_count": len(current_symbols) if current_symbols and exposure > 0 else 0,
            "is_rebalance": is_rebalance,
        })
        for symbol, w in new_weights.items():
            weights_rows.append({
                "trade_date": exec_date,
                "source_signal_date": signal_date,
                "ts_code": symbol,
                "weight": w,
            })

    daily = pd.DataFrame(rows)
    daily["strategy_equity"] = (1 + daily["strategy_ret"]).cumprod()
    daily["strategy_raw_equity"] = (1 + daily["strategy_raw_ret"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_ret"]).cumprod()
    weights = pd.DataFrame(weights_rows)

    metrics = {
        "strategy": calc_metrics(daily["strategy_ret"], daily["turnover"]),
        "strategy_raw": calc_metrics(daily["strategy_raw_ret"], daily["turnover"]),
        "benchmark_survivor_liquid": calc_metrics(daily["benchmark_ret"], None),
        "params": asdict(cfg),
        "first_trade_date": daily["trade_date"].iloc[0],
        "last_trade_date": daily["trade_date"].iloc[-1],
    }
    metrics["strategy"]["avg_exposure"] = float(daily["exposure"].mean())
    metrics["strategy_raw"]["avg_exposure"] = float(daily["exposure"].mean())
    return daily, weights, metrics


def latest_classification(df: pd.DataFrame, expected_end_date: str | None) -> pd.DataFrame:
    idx = df.groupby("ts_code", sort=False)["trade_date"].idxmax()
    latest = df.loc[idx].copy().sort_values("ts_code").reset_index(drop=True)
    if expected_end_date:
        stale = latest["trade_date"] < expected_end_date
        latest.loc[stale, "setup_class"] = "S0_SUSPENDED_OR_STALE"
        mapped = latest.loc[stale, "setup_class"].map(ACTION_MAP)
        latest.loc[stale, "action"] = mapped.map(lambda x: x[0])
        latest.loc[stale, "plan_bucket"] = mapped.map(lambda x: x[1])
        latest.loc[stale, "risk_class"] = mapped.map(lambda x: x[2])

    preferred = {
        CORE_CLASS: 0,
        BREAKOUT_CLASS: 1,
        PULLBACK_CLASS: 2,
        "B1_TREND_HOLD": 3,
        "B2_REPAIR_WATCH": 4,
    }
    latest["_rank_class"] = latest["setup_class"].map(preferred).fillna(9)
    latest = latest.sort_values(["_rank_class", "score_core", "score_aggressive"], ascending=[True, False, False])
    return latest.drop(columns=["_rank_class"]).reset_index(drop=True)


def add_czsc_snapshots(plan: pd.DataFrame, data_dir: Path, limit: int) -> tuple[pd.DataFrame, str | None]:
    if plan.empty or limit <= 0:
        return plan, None
    try:
        from czsc import CZSC, format_standard_kline
    except Exception as exc:
        return plan, f"czsc import failed: {exc}"

    records: list[dict[str, Any]] = []
    for symbol in plan["ts_code"].head(limit):
        path = data_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        try:
            raw = pd.read_parquet(path).tail(1000).copy()
            raw = raw.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
            raw["dt"] = pd.to_datetime(raw["dt"])
            bars = format_standard_kline(raw, freq="日线")
            try:
                c = CZSC(bars, max_bi_num=50)
            except TypeError:
                c = CZSC(bars)

            bi_list = getattr(c, "bi_list", []) or []
            fx_list = getattr(c, "fx_list", []) or []
            last_bi = bi_list[-1] if bi_list else None
            records.append({
                "ts_code": symbol,
                "czsc_bi_count": len(bi_list),
                "czsc_fx_count": len(fx_list),
                "last_bi_direction": str(getattr(last_bi, "direction", "")) if last_bi else "",
                "last_bi_sdt": str(getattr(last_bi, "sdt", "")) if last_bi else "",
                "last_bi_edt": str(getattr(last_bi, "edt", "")) if last_bi else "",
                "last_bi_high": getattr(last_bi, "high", np.nan) if last_bi else np.nan,
                "last_bi_low": getattr(last_bi, "low", np.nan) if last_bi else np.nan,
                "last_bi_power_price": getattr(last_bi, "power_price", np.nan) if last_bi else np.nan,
            })
        except Exception as exc:
            records.append({
                "ts_code": symbol,
                "czsc_bi_count": np.nan,
                "czsc_fx_count": np.nan,
                "last_bi_direction": "",
                "last_bi_sdt": "",
                "last_bi_edt": "",
                "last_bi_high": np.nan,
                "last_bi_low": np.nan,
                "last_bi_power_price": np.nan,
                "czsc_error": str(exc)[:200],
            })

    if not records:
        return plan, "no czsc snapshot records"
    snap = pd.DataFrame(records)
    return plan.merge(snap, on="ts_code", how="left"), None


def class_distribution(df: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    hist_dist = df.groupby(["trade_date", "setup_class"]).size().reset_index(name="count")
    latest_dist = latest["setup_class"].value_counts().rename_axis("setup_class").reset_index(name="latest_count")
    latest_dist["trade_date"] = "LATEST"
    return pd.concat([hist_dist, latest_dist[["trade_date", "setup_class", "latest_count"]]], ignore_index=True)


def percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_summary(
    output_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    latest_plan: pd.DataFrame,
    latest_all: pd.DataFrame,
    czsc_error: str | None,
) -> None:
    strategy = metrics["strategy"]
    raw = metrics["strategy_raw"]
    bench = metrics["benchmark_survivor_liquid"]
    lines = [
        "# A 股日线完全分类策略研究",
        "",
        "## 数据口径",
        "",
        f"- 数据目录：`{manifest.get('save_dir', DATA_DIR)}`",
        f"- 缓存更新时间：`{manifest.get('updated_at', 'unknown')}`",
        f"- 数据区间：`{manifest.get('start_date', 'unknown')}` 至 `{manifest.get('expected_trade_end_date', manifest.get('end_date', 'unknown'))}`",
        f"- 股票数：`{manifest.get('total_symbols', 'unknown')}`；失败数：`{manifest.get('fail_count', 'unknown')}`",
        "",
        "## 策略参数",
        "",
        f"- 策略：`{metrics['params']['profile']}`",
        f"- 排序分数：`{metrics['params']['score_profile']}`",
        f"- 趋势过滤：`{metrics['params']['core_trend_profile']}`",
        f"- 市场风险策略：`{metrics['params']['market_policy']}`；板块上限：`{metrics['params']['cap_policy']}`",
        f"- 核心池：`{CORE_CLASS}`，按 `score_core` 取前 `{metrics['params']['top_n']}` 只",
        f"- 调仓：每 `{metrics['params']['rebalance_days']}` 个交易日；成交：信号日收盘后计划，下一交易日开盘执行",
        f"- 成本：单边换手成本 `{metrics['params']['fee_rate']:.4f}`",
        "",
        "## 回测结果",
        "",
        "| 组合 | 总收益 | 年化 | Sharpe | 最大回撤 | Calmar | 日胜率 | 平均换手 | 平均暴露 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        (
            f"| 分类核心策略 | {percent(strategy['total_return'])} | {percent(strategy['annual_return'])} | "
            f"{strategy['sharpe']:.2f} | {percent(strategy['max_drawdown'])} | {strategy['calmar']:.2f} | "
            f"{percent(strategy['win_rate'])} | {strategy['avg_turnover']:.2f} | {percent(strategy.get('avg_exposure', 1.0))} |"
        ),
        (
            f"| 不计成本策略 | {percent(raw['total_return'])} | {percent(raw['annual_return'])} | "
            f"{raw['sharpe']:.2f} | {percent(raw['max_drawdown'])} | {raw['calmar']:.2f} | "
            f"{percent(raw['win_rate'])} | {raw['avg_turnover']:.2f} | {percent(raw.get('avg_exposure', 1.0))} |"
        ),
        (
            f"| 生存者流动性基准 | {percent(bench['total_return'])} | {percent(bench['annual_return'])} | "
            f"{bench['sharpe']:.2f} | {percent(bench['max_drawdown'])} | {bench['calmar']:.2f} | "
            f"{percent(bench['win_rate'])} | - | - |"
        ),
        "",
        "## 最新分类分布",
        "",
    ]
    for setup_class, count in latest_all["setup_class"].value_counts().items():
        lines.append(f"- `{setup_class}`: {count}")

    lines.extend([
        "",
        "## 最新核心计划 Top 20",
        "",
        "| 代码 | 日期 | 收盘 | 分类 | 动作 | score_core | 止损参考 | CZSC 最后一笔 |",
        "|:--|:--|--:|:--|:--|--:|--:|:--|",
    ])
    top = latest_plan[latest_plan["plan_bucket"].isin(["core_buy_pool", "satellite_buy_pool"])].head(20)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['ts_code']} | {row['trade_date']} | {row['close']:.2f} | {row['setup_class']} | "
            f"{row['action']} | {row['score_core']:.4f} | {row['stop_ref']:.2f} | "
            f"{row.get('last_bi_direction', '')} |"
        )

    lines.extend([
        "",
        "## 产物",
        "",
        "- `daily_equity.csv`: 策略与基准日度收益、净值、换手",
        "- `weights.csv`: 每日持仓权重；风险策略降仓时权重合计小于 1",
        "- `latest_classification.csv`: 全市场最新互斥分类",
        "- `latest_plan.csv`: 最新可执行计划池",
        "- `classification_distribution.csv`: 历史与最新分类分布",
        "- `summary.json`: 参数、数据口径、核心指标",
        "",
        "## 审计备注",
        "",
        "- 当前基准与策略都使用本地缓存中的当前上市股票集合，存在生存者偏差；后续应接入退市与历史成分。",
        "- 回测使用前复权日线、开盘到开盘收益、等权组合，没有处理冲击成本、涨跌停无法成交、最小交易单位和停牌复牌细节。",
        f"- 分类是完全互斥的研究协议；`C*` / `D*` 类用于计划规避，核心回测只交易 `{CORE_CLASS}`。",
    ])
    if czsc_error:
        lines.append(f"- CZSC 结构快照未完整生成：`{czsc_error}`")

    output_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.data_dir)
    end_date = args.end_date or manifest.get("expected_trade_end_date") or manifest.get("end_date")
    cfg = build_config(args, end_date=end_date)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] data_dir={args.data_dir}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] cfg={cfg}")

    features = load_features(args.data_dir, end_date=end_date, max_symbols=args.max_symbols)
    classified = classify_rows(features, cfg)
    daily, weights, metrics = run_backtest(classified, cfg)

    expected_end_date = manifest.get("expected_trade_end_date") or end_date
    latest_all = latest_classification(classified, expected_end_date=expected_end_date)
    plan_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "amount",
        "setup_class", "action", "plan_bucket", "risk_class", "score_core", "score_aggressive",
        "entry_ref", "stop_ref", "invalid_condition", "breadth120", "ret20", "ret60", "ret120",
        "ma20", "ma60", "ma120", "ma250", "amt20", "vol20",
    ]
    latest_plan = latest_all.loc[
        latest_all["plan_bucket"].isin(["core_buy_pool", "satellite_buy_pool", "hold_pool", "watch_pool"]),
        plan_cols,
    ].copy()

    czsc_error = None
    if not args.skip_czsc_snapshot:
        latest_plan, czsc_error = add_czsc_snapshots(latest_plan, args.data_dir, cfg.czsc_snapshot_limit)

    daily.to_csv(args.output_dir / "daily_equity.csv", index=False)
    weights.to_csv(args.output_dir / "weights.csv", index=False)
    latest_all[plan_cols].to_csv(args.output_dir / "latest_classification.csv", index=False)
    latest_plan.to_csv(args.output_dir / "latest_plan.csv", index=False)
    class_distribution(classified, latest_all).to_csv(args.output_dir / "classification_distribution.csv", index=False)

    summary = {
        "manifest": manifest,
        "metrics": metrics,
        "latest_counts": latest_all["setup_class"].value_counts().to_dict(),
        "latest_plan_counts": latest_plan["plan_bucket"].value_counts().to_dict(),
        "czsc_snapshot_error": czsc_error,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_summary(args.output_dir, manifest, metrics, latest_plan, latest_all, czsc_error)

    print("[INFO] done")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
