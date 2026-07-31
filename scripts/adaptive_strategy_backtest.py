"""市场环境自适应策略路由验证

基于 FSM 审计数据验证 "牛市追涨 vs 熊市/震荡均值回复" 的策略路由逻辑。

使用等权指数（面板中所有股票的等权平均收盘价）作为市场环境指标，
按 60 日收益率分类为 bull/sideways/bear，然后在不同市场环境下对比：
- 原始 surge_onset 信号的超额收益
- 均值回复信号的超额收益
- 自适应路由（牛市用追涨、熊/震荡用均值回复）的超额收益

输出：
- scripts/_output/adaptive_strategy_results.json
- 控制台打印对比

    uv run --no-sync python scripts/adaptive_strategy_backtest.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
STATES_PATH = OUTPUT_DIR / "fsm_audit" / "states.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"
CANDIDATES_PATH = OUTPUT_DIR / "surge_candidates" / "candidates.parquet"

FWD_WINDOWS = [5, 10, 20]
TRAIN_END = pd.Timestamp("2023-12-31")
LOOKBACK = 60
BULL_THRESHOLD = 10.0
BEAR_THRESHOLD = -10.0

RECOVERY_REGIMES = {2, 3, 4}
DOWNTREND = 1
UPTREND_REGIMES = {5, 6, 7, 8}


def build_market_index(panel: pd.DataFrame) -> pd.DataFrame:
    """构建等权市场指数。"""
    daily = panel.groupby("dt").agg(
        eq_close=("close", "mean"),
        n_stocks=("close", "count"),
    ).reset_index().sort_values("dt")
    return daily


def classify_market(index_df: pd.DataFrame) -> pd.DataFrame:
    """对市场指数分类环境。"""
    closes = index_df["eq_close"].values
    n = len(closes)
    regimes = []
    for i in range(n):
        if i < LOOKBACK:
            regimes.append("sideways")
        else:
            ret_pct = (closes[i] / closes[i - LOOKBACK] - 1) * 100
            if ret_pct >= BULL_THRESHOLD:
                regimes.append("bull")
            elif ret_pct <= BEAR_THRESHOLD:
                regimes.append("bear")
            else:
                regimes.append("sideways")
    index_df = index_df.copy()
    index_df["market_regime"] = regimes
    return index_df


def detect_reversion_signals(states: pd.DataFrame, min_down: int = 5, max_down: int = 20) -> pd.DataFrame:
    """检测均值回复信号。"""
    states_sorted = states.sort_values(["symbol", "dt"]).reset_index(drop=True)
    signals = []
    for sym, grp in states_sorted.groupby("symbol"):
        regimes = grp["regime"].values
        dates = grp["dt"].values
        down_count = 0
        for i in range(1, len(regimes)):
            if regimes[i - 1] == DOWNTREND:
                down_count += 1
            else:
                down_count = 0
            if regimes[i - 1] == DOWNTREND and int(regimes[i]) in RECOVERY_REGIMES:
                if min_down <= down_count <= max_down:
                    signals.append({"symbol": sym, "dt": dates[i], "signal_type": "reversion", "down_days": down_count})
                down_count = 0
    return pd.DataFrame(signals)


def main():
    t0 = time.time()
    print("[加载] 数据 ...")
    states = pd.read_parquet(STATES_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)

    # 计算面板前向收益
    print("[计算] 面板前向收益 ...")
    panel_sorted = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    for w in FWD_WINDOWS:
        panel_sorted[f"fwd_{w}d"] = (
            panel_sorted.groupby("symbol")["close"]
            .transform(lambda s: s.shift(-w) / s - 1) * 100
        )

    # 构建市场指数和环境分类
    print("[构建] 市场环境指数 ...")
    index_df = build_market_index(panel_sorted)
    index_df = classify_market(index_df)
    market_map = dict(zip(index_df["dt"], index_df["market_regime"]))
    env_counts = index_df["market_regime"].value_counts()
    print(f"  市场环境分布: {dict(env_counts)}")

    # 追涨信号（candidates delay=0, mode=confirm/anticipate）
    surge_signals = candidates[candidates["delay"] == 0][["symbol", "entry_dt", "mode"]].copy()
    surge_signals.rename(columns={"entry_dt": "dt"}, inplace=True)
    surge_signals["signal_type"] = "surge"

    # 均值回复信号
    print("[检测] 均值回复信号 ...")
    rev_signals = detect_reversion_signals(states)
    print(f"  均值回复信号: {len(rev_signals)}")

    # 合并市场环境标签
    surge_merged = surge_signals.merge(
        panel_sorted[["symbol", "dt"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
        on=["symbol", "dt"], how="left"
    )
    surge_merged["market_regime"] = surge_merged["dt"].map(market_map)
    surge_merged["seg"] = np.where(surge_merged["dt"] <= TRAIN_END, "train", "test")

    rev_merged = rev_signals.merge(
        panel_sorted[["symbol", "dt"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
        on=["symbol", "dt"], how="left"
    )
    rev_merged["market_regime"] = rev_merged["dt"].map(market_map)
    rev_merged["seg"] = np.where(rev_merged["dt"] <= TRAIN_END, "train", "test")

    # 全市场基准
    baseline = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        baseline[col] = panel_sorted[col].dropna().mean()

    # 分析：按市场环境 × 信号类型
    results = {"baseline": {k: round(v, 4) for k, v in baseline.items()}}

    for sig_type, df in [("surge", surge_merged), ("reversion", rev_merged)]:
        sig_results = {}
        for env in ["bull", "sideways", "bear"]:
            oos_sub = df.loc[(df["market_regime"] == env) & (df["seg"] == "test")]
            has_fwd = oos_sub["fwd_5d"].notna()
            valid = oos_sub.loc[has_fwd]
            env_data = {"n_oos": int(len(valid))}
            for w in FWD_WINDOWS:
                col = f"fwd_{w}d"
                if len(valid) > 0:
                    mean = valid[col].mean()
                    excess = mean - baseline[col]
                    env_data[f"{col}_mean"] = round(mean, 4)
                    env_data[f"{col}_excess"] = round(excess, 4)
                else:
                    env_data[f"{col}_mean"] = None
                    env_data[f"{col}_excess"] = None
            sig_results[env] = env_data
        results[sig_type] = sig_results

    # 自适应路由策略模拟：牛市用 surge，熊市+震荡用 reversion
    adaptive_signals = pd.concat([
        surge_merged[surge_merged["market_regime"] == "bull"],
        rev_merged[rev_merged["market_regime"].isin(["bear", "sideways"])],
    ], ignore_index=True)
    oos_adaptive = adaptive_signals[
        (adaptive_signals["seg"] == "test") & adaptive_signals["fwd_5d"].notna()
    ]

    adaptive_data = {"n_oos": int(len(oos_adaptive))}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        if len(oos_adaptive) > 0:
            mean = oos_adaptive[col].mean()
            excess = mean - baseline[col]
            adaptive_data[f"{col}_mean"] = round(mean, 4)
            adaptive_data[f"{col}_excess"] = round(excess, 4)
        else:
            adaptive_data[f"{col}_mean"] = None
            adaptive_data[f"{col}_excess"] = None
    results["adaptive_route"] = adaptive_data

    # 对比：纯 surge（所有环境）
    oos_surge_all = surge_merged[
        (surge_merged["seg"] == "test") & surge_merged["fwd_5d"].notna()
    ]
    surge_all_data = {"n_oos": int(len(oos_surge_all))}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        if len(oos_surge_all) > 0:
            mean = oos_surge_all[col].mean()
            surge_all_data[f"{col}_mean"] = round(mean, 4)
            surge_all_data[f"{col}_excess"] = round(mean - baseline[col], 4)
    results["surge_all_env"] = surge_all_data

    # 输出
    out_path = OUTPUT_DIR / "adaptive_strategy_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[输出] {out_path}")

    # 打印汇总
    print("\n" + "=" * 90)
    print("市场环境自适应策略验证")
    print("=" * 90)

    print(f"\n全市场基准: fwd_5d={baseline['fwd_5d']:.4f}% | fwd_20d={baseline['fwd_20d']:.4f}%")

    print("\n--- 追涨信号 (surge) 按市场环境 ---")
    for env in ["bull", "sideways", "bear"]:
        d = results["surge"][env]
        print(f"  {env:>8}: n={d['n_oos']:>6} | fwd_5d_excess={d.get('fwd_5d_excess', 'N/A'):>8} | fwd_20d_excess={d.get('fwd_20d_excess', 'N/A'):>8}")

    print("\n--- 均值回复信号 (reversion) 按市场环境 ---")
    for env in ["bull", "sideways", "bear"]:
        d = results["reversion"][env]
        print(f"  {env:>8}: n={d['n_oos']:>6} | fwd_5d_excess={d.get('fwd_5d_excess', 'N/A'):>8} | fwd_20d_excess={d.get('fwd_20d_excess', 'N/A'):>8}")

    print("\n--- 策略对比 ---")
    sa = results["surge_all_env"]
    ar = results["adaptive_route"]
    print(f"  纯追涨 (所有环境): n={sa['n_oos']:>6} | fwd_5d_excess={sa.get('fwd_5d_excess', 'N/A'):>8} | fwd_20d_excess={sa.get('fwd_20d_excess', 'N/A'):>8}")
    print(f"  自适应路由:         n={ar['n_oos']:>6} | fwd_5d_excess={ar.get('fwd_5d_excess', 'N/A'):>8} | fwd_20d_excess={ar.get('fwd_20d_excess', 'N/A'):>8}")

    if sa.get("fwd_5d_excess") and ar.get("fwd_5d_excess"):
        improvement = ar["fwd_5d_excess"] - sa["fwd_5d_excess"]
        print(f"\n  自适应 vs 纯追涨 fwd_5d 超额改善: {improvement:+.4f}%")

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
