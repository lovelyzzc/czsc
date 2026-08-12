"""全量综合回测 —— 6 种策略变体横向对比 + 深度分析

将 FSM 审计优化方向的 4 个维度（门控翻转、均值回复、市场环境自适应、持仓周期）
组合为 6 种策略变体，在同一数据集上进行槽位组合回测并横向对比：

| 策略 | 信号源 | 门控 | 市场环境 | 持仓 |
|------|--------|------|---------|------|
| S0-Baseline | surge confirm+anticipate | vr>=1.2, sp>=3.0, zg=True | 无 | 原始 |
| S1-FlippedGate | surge | vr<=0.8, sp>=3.0, zg=False | 无 | 原始 |
| S2-OptimalGate | surge | vr<=1.2, sp>=10.0, zg=False | 无 | 原始 |
| S3-Reversion | 下跌态出口(R1→R2/R4) | 无门控 | 无 | 20 日固定 |
| S4-Adaptive | 牛市surge + 熊/震荡reversion | 各自门控 | 分环境 | 各自 |
| S5-Combined | 合并 S2+S3 候选池 | 综合优先级 | 分环境 | 按regime |

输出：
- scripts/_output/comprehensive_backtest.json   完整结果
- 控制台打印对比表

    uv run --no-sync python scripts/comprehensive_backtest.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from surge_candidates_dump import completed_outcomes

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
CANDIDATES_PATH = OUTPUT_DIR / "surge_candidates" / "candidates.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"
STATES_PATH = OUTPUT_DIR / "fsm_audit" / "states.parquet"

TRAIN_END = pd.Timestamp("2023-12-31")
N_SLOTS = 10
BUY_COST = 0.0015
SELL_COST = 0.0025
FWD_WINDOWS = [5, 10, 20]
RECOVERY_REGIMES = {2, 3, 4}
DOWNTREND = 1
LOOKBACK = 30
BULL_THRESHOLD = 8.0
BEAR_THRESHOLD = -8.0


def load_data():
    """加载所有数据。"""
    print("[加载] 数据 ...")
    cand = completed_outcomes(pd.read_parquet(CANDIDATES_PATH))
    panel = pd.read_parquet(PANEL_PATH)
    states = pd.read_parquet(STATES_PATH)
    print(f"  候选: {len(cand)} | 面板: {len(panel)} | FSM状态: {len(states)}")
    return cand, panel, states


def compute_panel_fwd(panel: pd.DataFrame) -> pd.DataFrame:
    """为面板添加前向收益。"""
    ps = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    for w in FWD_WINDOWS:
        ps[f"fwd_{w}d"] = ps.groupby("symbol")["close"].transform(lambda s: s.shift(-w) / s - 1) * 100
    return ps


def build_market_regime(panel: pd.DataFrame) -> dict:
    """构建市场环境映射 dt → regime。"""
    daily = panel.groupby("dt").agg(eq_close=("close", "mean")).reset_index().sort_values("dt")
    closes = daily["eq_close"].values
    dates = daily["dt"].values
    regime_map = {}
    for i in range(len(closes)):
        if i < LOOKBACK:
            regime_map[dates[i]] = "sideways"
        else:
            ret = (closes[i] / closes[i - LOOKBACK] - 1) * 100
            if ret >= BULL_THRESHOLD:
                regime_map[dates[i]] = "bull"
            elif ret <= BEAR_THRESHOLD:
                regime_map[dates[i]] = "bear"
            else:
                regime_map[dates[i]] = "sideways"
    return regime_map


def detect_reversion(states: pd.DataFrame, min_down: int = 5, max_down: int = 20) -> pd.DataFrame:
    """检测均值回复信号。"""
    states_s = states.sort_values(["symbol", "dt"]).reset_index(drop=True)
    signals = []
    for sym, grp in states_s.groupby("symbol"):
        regimes = grp["regime"].values
        dates = grp["dt"].values
        dc = 0
        for i in range(1, len(regimes)):
            if regimes[i - 1] == DOWNTREND:
                dc += 1
            else:
                dc = 0
            if regimes[i - 1] == DOWNTREND and int(regimes[i]) in RECOVERY_REGIMES:
                if min_down <= dc <= max_down:
                    signals.append(
                        {
                            "symbol": sym,
                            "dt": dates[i],
                            "to_regime": int(regimes[i]),
                            "down_days": dc,
                        }
                    )
                dc = 0
    return pd.DataFrame(signals)


def gate_filter(cand: pd.DataFrame, vr_th: float, vr_dir: str, sp_th: float, sp_dir: str, use_zg: bool) -> pd.Series:
    """向量化门控过滤。"""
    vr = cand["sig_vol_ratio"]
    sp = cand["sig_ma_spread_pct"]
    zg = cand["sig_above_zg"]

    if vr_dir == "gte":
        vr_ok = vr >= vr_th
    else:
        vr_ok = vr <= vr_th
    vr_ok = vr_ok & vr.notna()

    if sp_dir == "gte":
        sp_ok = sp >= sp_th
    else:
        sp_ok = sp <= sp_th
    sp_ok = sp_ok & sp.notna()

    zg_ok = zg.astype(bool) if use_zg else pd.Series(True, index=cand.index)
    return vr_ok & sp_ok & zg_ok


def simulate_portfolio(
    entries: pd.DataFrame, panel_fwd: pd.DataFrame, n_slots: int = N_SLOTS, hold_days_col: str = None
) -> dict:
    """简化的槽位组合模拟。

    entries 需要列：symbol, dt, priority, [hold_days]
    返回 dict 统计结果。
    """
    merged = entries.merge(
        panel_fwd[["symbol", "dt", "close"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
        on=["symbol", "dt"],
        how="left",
    )
    merged["seg"] = np.where(merged["dt"] <= TRAIN_END, "train", "test")
    merged = merged.dropna(subset=["fwd_5d"])

    if len(merged) == 0:
        return _empty_result()

    merged = merged.sort_values(["dt", "priority"], ascending=[True, False])

    # 贪心槽位分配
    trades = []
    occupied = {}  # symbol → release_dt
    for dt, day_grp in merged.groupby("dt"):
        occupied = {s: rd for s, rd in occupied.items() if rd > dt}
        free = n_slots - len(occupied)
        if free <= 0:
            continue
        for _, row in day_grp.iterrows():
            if free <= 0:
                break
            if row["symbol"] in occupied:
                continue
            hd = int(row.get(hold_days_col, 20)) if hold_days_col and hold_days_col in row.index else 20
            release = dt + pd.Timedelta(days=hd + 2)
            occupied[row["symbol"]] = release
            trades.append(row)
            free -= 1

    if not trades:
        return _empty_result()

    trades_df = pd.DataFrame(trades)

    result = {}
    for seg_label, mask in [
        ("ALL", pd.Series(True, index=trades_df.index)),
        ("IS", trades_df["seg"] == "train"),
        ("OOS", trades_df["seg"] == "test"),
    ]:
        sub = trades_df.loc[mask]
        n = len(sub)
        seg_data = {"n_trades": n}
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            if n > 0:
                mean_ret = sub[col].mean()
                net_ret = mean_ret - (BUY_COST + SELL_COST) * 100
                std = sub[col].std()
                win_rate = (sub[col] > 0).mean() * 100
                seg_data[col] = {
                    "mean": round(mean_ret, 4),
                    "net": round(net_ret, 4),
                    "std": round(std, 4),
                    "win_rate": round(win_rate, 1),
                    "sharpe_est": round(mean_ret / std * np.sqrt(252 / w), 3) if std > 0 else 0,
                }
            else:
                seg_data[col] = {"mean": 0, "net": 0, "std": 0, "win_rate": 0, "sharpe_est": 0}
        # 年度分解
        if seg_label == "OOS" and n > 0:
            yearly = {}
            for yr, yr_grp in sub.groupby(sub["dt"].dt.year):
                yearly[str(yr)] = {
                    "n": len(yr_grp),
                    "fwd_5d_mean": round(yr_grp["fwd_5d"].mean(), 4),
                    "fwd_20d_mean": round(yr_grp["fwd_20d"].mean(), 4),
                    "win_rate_5d": round((yr_grp["fwd_5d"] > 0).mean() * 100, 1),
                }
            seg_data["yearly"] = yearly
        result[seg_label] = seg_data
    return result


def analyze_capacity(entries: pd.DataFrame, panel_fwd: pd.DataFrame, label: str, n_slots: int = N_SLOTS) -> dict:
    """分析策略的槽位容量：利用率、空仓天数、有效年化。"""
    merged = entries.merge(
        panel_fwd[["symbol", "dt", "close"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
        on=["symbol", "dt"],
        how="left",
    ).dropna(subset=["fwd_5d"])
    merged = merged.sort_values(["dt", "priority"], ascending=[True, False])

    oos = merged[merged["dt"] > TRAIN_END]
    if len(oos) == 0:
        return {"label": label, "error": "no OOS data"}

    oos_trade_dates = sorted(panel_fwd.loc[panel_fwd["dt"] > TRAIN_END, "dt"].unique())
    n_total_days = len(oos_trade_dates)

    occupied = {}
    daily_stats = []
    for dt in oos_trade_dates:
        occupied = {s: rd for s, rd in occupied.items() if rd > dt}
        day_entries = oos[oos["dt"] == dt]
        free = n_slots - len(occupied)
        new_fills = 0
        for _, row in day_entries.iterrows():
            if free <= 0:
                break
            if row["symbol"] in occupied:
                continue
            release = dt + pd.Timedelta(days=22)
            occupied[row["symbol"]] = release
            free -= 1
            new_fills += 1
        daily_stats.append(
            {
                "dt": dt,
                "occupied": len(occupied),
                "utilization": len(occupied) / n_slots,
                "new_fills": new_fills,
            }
        )

    ds = pd.DataFrame(daily_stats)
    ds["year"] = pd.to_datetime(ds["dt"]).dt.year
    ds["month"] = pd.to_datetime(ds["dt"]).dt.to_period("M")

    empty_days = int((ds["occupied"] == 0).sum())
    avg_util = round(ds["utilization"].mean() * 100, 1)

    yearly = {}
    for yr, yg in ds.groupby("year"):
        yearly[str(yr)] = {
            "avg_utilization_pct": round(yg["utilization"].mean() * 100, 1),
            "empty_days": int((yg["occupied"] == 0).sum()),
            "total_days": len(yg),
            "avg_occupied": round(yg["occupied"].mean(), 1),
        }

    return {
        "label": label,
        "oos_total_days": n_total_days,
        "empty_days": empty_days,
        "empty_pct": round(empty_days / n_total_days * 100, 1) if n_total_days > 0 else 0,
        "avg_utilization_pct": avg_util,
        "avg_occupied_slots": round(ds["occupied"].mean(), 1),
        "yearly": yearly,
    }


def _empty_result():
    return {seg: {"n_trades": 0} for seg in ["ALL", "IS", "OOS"]}


def main():
    t0 = time.time()
    cand, panel, states = load_data()

    print("[计算] 面板前向收益 ...")
    panel_fwd = compute_panel_fwd(panel)

    # 全市场基准
    baseline = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        baseline[col] = round(panel_fwd[col].dropna().mean(), 4)
    print(f"  全市场基准: fwd_5d={baseline['fwd_5d']}% | fwd_20d={baseline['fwd_20d']}%")

    print("[构建] 市场环境 ...")
    mkt_map = build_market_regime(panel_fwd)
    mkt_counts = pd.Series(mkt_map.values()).value_counts().to_dict()
    print(f"  环境分布: {mkt_counts}")

    # delay=0 候选
    d0 = cand[cand["delay"] == 0].copy()

    print("[检测] 均值回复信号 ...")
    rev_signals = detect_reversion(states, min_down=5, max_down=20)
    rev_signals = rev_signals[rev_signals["to_regime"].isin({2, 4})]  # 只保留 FirstBuy 和 PivotBuilding
    print(f"  均值回复信号(R2+R4): {len(rev_signals)}")

    results = {"baseline": baseline, "strategies": {}}

    # ─── S0: New Default (flipped gate) ──────────────────────────
    print("\n[S0] NewDefault: vr<=0.8, sp>=3.0, zg=False (新默认) ...")
    s0_mask = gate_filter(d0, 0.8, "lte", 3.0, "gte", False)
    s0_entries = d0.loc[s0_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S0_NewDefault"] = simulate_portfolio(s0_entries, panel_fwd)

    # ─── S0L: Legacy Baseline (old defaults) ──────────────────────
    print("[S0L] LegacyBaseline: vr>=1.2, sp>=3.0, zg=True (旧默认) ...")
    s0l_mask = gate_filter(d0, 1.2, "gte", 3.0, "gte", True)
    s0l_entries = d0.loc[s0l_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S0L_Legacy"] = simulate_portfolio(s0l_entries, panel_fwd)

    # ─── S2: Optimal Gate ─────────────────────────────────────────
    print("[S2] OptimalGate: vr<=1.2, sp>=10.0, zg=False ...")
    s2_mask = gate_filter(d0, 1.2, "lte", 10.0, "gte", False)
    s2_entries = d0.loc[s2_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S2_OptimalGate"] = simulate_portfolio(s2_entries, panel_fwd)

    # ─── S2a/S2b/S2c: 精搜索验证 Top-3 ─────────────────────────
    print("[S2a] Refined_vr09_sp12: vr<=0.9, sp>=12.0, zg=False ...")
    s2a_mask = gate_filter(d0, 0.9, "lte", 12.0, "gte", False)
    s2a_entries = d0.loc[s2a_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S2a_vr09_sp12"] = simulate_portfolio(s2a_entries, panel_fwd)

    print("[S2b] Refined_vr08_sp12: vr<=0.8, sp>=12.0, zg=False ...")
    s2b_mask = gate_filter(d0, 0.8, "lte", 12.0, "gte", False)
    s2b_entries = d0.loc[s2b_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S2b_vr08_sp12"] = simulate_portfolio(s2b_entries, panel_fwd)

    print("[S2c] Refined_vr09_sp10: vr<=0.9, sp>=10.0, zg=False ...")
    s2c_mask = gate_filter(d0, 0.9, "lte", 10.0, "gte", False)
    s2c_entries = d0.loc[s2c_mask, ["symbol", "entry_dt", "score"]].rename(
        columns={"entry_dt": "dt", "score": "priority"}
    )
    results["strategies"]["S2c_vr09_sp10"] = simulate_portfolio(s2c_entries, panel_fwd)

    # ─── S3: Pure Reversion ───────────────────────────────────────
    print("[S3] Reversion: R1→R2/R4, hold=20d ...")
    rev_merged = rev_signals.copy()
    rev_merged["priority"] = 50.0 + np.random.default_rng(42).uniform(-5, 5, len(rev_merged))
    s3_entries = rev_merged[["symbol", "dt", "priority"]].copy()
    results["strategies"]["S3_Reversion"] = simulate_portfolio(s3_entries, panel_fwd)

    # ─── S4: Adaptive (bull→surge, bear/sideways→reversion) ──────
    print("[S4] Adaptive: 牛市追涨(S0) + 熊/震荡均值回复(S3) ...")
    s0_entries_tagged = s0_entries.copy()
    s0_entries_tagged["market"] = s0_entries_tagged["dt"].map(mkt_map)
    s3_entries_tagged = s3_entries.copy()
    s3_entries_tagged["market"] = s3_entries_tagged["dt"].map(mkt_map)

    s4_entries = pd.concat(
        [
            s0_entries_tagged[s0_entries_tagged["market"] == "bull"][["symbol", "dt", "priority"]],
            s3_entries_tagged[s3_entries_tagged["market"].isin(["bear", "sideways"])][["symbol", "dt", "priority"]],
        ],
        ignore_index=True,
    )
    results["strategies"]["S4_Adaptive"] = simulate_portfolio(s4_entries, panel_fwd)

    # ─── S4b: Adaptive with S2b bull arm (vr<=0.8, sp>=12) ─────
    print("[S4b] Adaptive(S2b): 牛市追涨(S2b sp>=12) + 熊/震荡均值回复(S3) ...")
    s2b_entries_tagged = s2b_entries.copy()
    s2b_entries_tagged["market"] = s2b_entries_tagged["dt"].map(mkt_map)
    s4b_entries = pd.concat(
        [
            s2b_entries_tagged[s2b_entries_tagged["market"] == "bull"][["symbol", "dt", "priority"]],
            s3_entries_tagged[s3_entries_tagged["market"].isin(["bear", "sideways"])][["symbol", "dt", "priority"]],
        ],
        ignore_index=True,
    )
    results["strategies"]["S4b_Adaptive_S2b"] = simulate_portfolio(s4b_entries, panel_fwd)

    # ─── S4c: Adaptive with S2c bull arm (vr<=0.9, sp>=10) ─────
    print("[S4c] Adaptive(S2c): 牛市追涨(S2c sp>=10) + 熊/震荡均值回复(S3) ...")
    s2c_entries_tagged = s2c_entries.copy()
    s2c_entries_tagged["market"] = s2c_entries_tagged["dt"].map(mkt_map)
    s4c_entries = pd.concat(
        [
            s2c_entries_tagged[s2c_entries_tagged["market"] == "bull"][["symbol", "dt", "priority"]],
            s3_entries_tagged[s3_entries_tagged["market"].isin(["bear", "sideways"])][["symbol", "dt", "priority"]],
        ],
        ignore_index=True,
    )
    results["strategies"]["S4c_Adaptive_S2c"] = simulate_portfolio(s4c_entries, panel_fwd)

    # ─── S7 已废弃（2026-07-31）────────────────────────────────
    # S2c 牛市增量补位在 s2c_incremental_validation.py 里被判死：卫星 44 笔实现 -89.6%，
    # 同时挤掉 49 笔 S2b（实际机会成本 +206.2%），净边际 -313.4%，Sharpe 0.332 vs S2b 1.049。
    # 后续 s8_incremental_validation.py 进一步证明：即使取消牛市过滤、并保证 S2b 交易集合
    # 逐笔不变（强制替换 / 独立预算），S2c 增量在 walk-forward 上仍不显著
    # （ΔSharpe -0.03~-0.10，卫星 pair 级净超额中位数 -1.0%~-1.3%）。
    # 详见 scripts/S8_INCREMENTAL_VALIDATION_2026-07-31.md

    # ─── S5: Combined Pool ───────────────────────────────────────
    print("[S5] Combined: S0+S3 候选合并，综合优先级 ...")
    s0_pool = s0_entries.copy()
    s0_pool["source"] = "surge"
    s0_pool["priority"] = s0_pool["priority"] * 0.8

    s3_pool = s3_entries.copy()
    s3_pool["source"] = "reversion"
    s3_pool["priority"] = s3_pool["priority"] * 1.2

    s5_entries = pd.concat([s0_pool, s3_pool], ignore_index=True)
    results["strategies"]["S5_Combined"] = simulate_portfolio(s5_entries[["symbol", "dt", "priority"]], panel_fwd)

    # ─── 全市场对照组 ─────────────────────────────────────────────
    print("[BM] 全市场随机对照 ...")
    oos_dates = panel_fwd.loc[panel_fwd["dt"] > TRAIN_END, "dt"].unique()
    rng = np.random.default_rng(42)
    random_entries = []
    for dt in rng.choice(oos_dates, size=min(5000, len(oos_dates)), replace=False):
        day_stocks = panel_fwd[panel_fwd["dt"] == dt]
        if len(day_stocks) > 0:
            sampled = day_stocks.sample(n=min(N_SLOTS, len(day_stocks)), random_state=int(dt.astype(int) % 2**31))
            for _, r in sampled.iterrows():
                random_entries.append({"symbol": r["symbol"], "dt": dt, "priority": 50.0})
    random_df = pd.DataFrame(random_entries)
    results["strategies"]["BM_Random"] = simulate_portfolio(random_df, panel_fwd)

    # ─── 容量分析 ──────────────────────────────────────────────────
    print("\n[容量] 分析 S0/S2b/S4b 的槽位利用率 ...")
    cap_s0 = analyze_capacity(s0_entries, panel_fwd, "S0_NewDefault")
    cap_s2b = analyze_capacity(s2b_entries, panel_fwd, "S2b_vr08_sp12")
    cap_s4b = analyze_capacity(s4b_entries, panel_fwd, "S4b_Adaptive_S2b")
    results["capacity_analysis"] = {
        "S0_NewDefault": cap_s0,
        "S2b_vr08_sp12": cap_s2b,
        "S4b_Adaptive_S2b": cap_s4b,
    }
    for cap in [cap_s0, cap_s2b, cap_s4b]:
        print(
            f"  {cap['label']}: 利用率={cap.get('avg_utilization_pct', 0)}%, "
            f"空仓={cap.get('empty_days', 0)}天({cap.get('empty_pct', 0)}%), "
            f"日均持仓={cap.get('avg_occupied_slots', 0)}槽"
        )

    # ─── 输出 ────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "comprehensive_backtest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[输出] {out_path}")

    # ─── 打印对比表 ──────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("全量综合回测结果对比")
    print("=" * 120)

    print(f"\n全市场基准: fwd_5d={baseline['fwd_5d']}% | fwd_20d={baseline['fwd_20d']}%")

    # OOS 对比表
    print(
        f"\n{'策略':<20} {'OOS交易数':>10} {'fwd_5d均值':>12} {'fwd_5d净值':>12} {'胜率_5d':>8} {'Sharpe_5d':>10} {'fwd_20d均值':>12} {'fwd_20d净值':>12} {'胜率_20d':>8}"
    )
    print("-" * 120)

    for name, data in results["strategies"].items():
        oos = data.get("OOS", {})
        n = oos.get("n_trades", 0)
        f5 = oos.get("fwd_5d", {})
        f20 = oos.get("fwd_20d", {})
        print(
            f"{name:<20} {n:>10} "
            f"{f5.get('mean', 0):>12.4f} {f5.get('net', 0):>12.4f} {f5.get('win_rate', 0):>7.1f}% {f5.get('sharpe_est', 0):>10.3f} "
            f"{f20.get('mean', 0):>12.4f} {f20.get('net', 0):>12.4f} {f20.get('win_rate', 0):>7.1f}%"
        )

    # OOS 年度分解
    print("\n" + "=" * 120)
    print("OOS 年度分解（fwd_5d 均值 %）")
    print("=" * 120)
    all_years = set()
    for data in results["strategies"].values():
        yearly = data.get("OOS", {}).get("yearly", {})
        all_years.update(yearly.keys())
    all_years = sorted(all_years)

    header = f"{'策略':<20}" + "".join(f"{y:>10}" for y in all_years)
    print(header)
    print("-" * len(header))

    for name, data in results["strategies"].items():
        yearly = data.get("OOS", {}).get("yearly", {})
        row = f"{name:<20}"
        for y in all_years:
            yd = yearly.get(y, {})
            val = yd.get("fwd_5d_mean", 0)
            row += f"{val:>10.4f}"
        print(row)

    # 超额收益对比
    print("\n" + "=" * 120)
    print("OOS 超额收益对比（vs 全市场基准）")
    print("=" * 120)
    print(f"{'策略':<20} {'fwd_5d超额':>12} {'fwd_10d超额':>12} {'fwd_20d超额':>12} {'vs S0改善(5d)':>14}")
    print("-" * 120)

    s0_oos = results["strategies"].get("S0_NewDefault", {}).get("OOS", {})
    s0_5d = s0_oos.get("fwd_5d", {}).get("mean", 0)

    for name, data in results["strategies"].items():
        oos = data.get("OOS", {})
        excess = {}
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            mean = oos.get(col, {}).get("mean", 0)
            excess[col] = mean - baseline[col]

        improve = oos.get("fwd_5d", {}).get("mean", 0) - s0_5d
        print(
            f"{name:<20} "
            f"{excess['fwd_5d']:>+12.4f} "
            f"{excess['fwd_10d']:>+12.4f} "
            f"{excess['fwd_20d']:>+12.4f} "
            f"{improve:>+14.4f}"
        )

    # IS vs OOS 一致性
    print("\n" + "=" * 120)
    print("IS vs OOS 一致性检查（fwd_5d 均值）")
    print("=" * 120)
    print(f"{'策略':<20} {'IS_fwd_5d':>12} {'OOS_fwd_5d':>12} {'差值':>10} {'方向一致':>10}")
    print("-" * 80)

    for name, data in results["strategies"].items():
        is_d = data.get("IS", {}).get("fwd_5d", {})
        oos_d = data.get("OOS", {}).get("fwd_5d", {})
        is_m = is_d.get("mean", 0)
        oos_m = oos_d.get("mean", 0)
        diff = oos_m - is_m
        consistent = "✓" if (is_m > 0 and oos_m > 0) or (is_m < 0 and oos_m < 0) else "✗"
        print(f"{name:<20} {is_m:>12.4f} {oos_m:>12.4f} {diff:>+10.4f} {consistent:>10}")

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
