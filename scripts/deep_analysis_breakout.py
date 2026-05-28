"""深度分析：对比不同入场策略的历史表现

研究问题：
1. 涨停当日直接追涨 vs 等待确认后介入，哪种胜率更高？
2. 「确认」的最佳定义是什么？（连续 N 日站稳 ZG 上方？回踩不破 ZG？）
3. 哪些特征的突破后续表现更好？
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "_output"
df = pd.read_csv(OUTPUT / "zs_breakout_signals.csv")

print(f"原始样本: {len(df)}")
print("=" * 80)

# ====================================================================
# 第一部分：扩展数据 —— 回到原始 parquet 计算「确认」相关指标
# ====================================================================

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"


def enrich_signal(row):
    """为每个信号补充确认相关指标"""
    symbol = row["symbol"]
    signal_date = str(int(row["date"]))
    pf = DATA_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return None

    stock_df = pd.read_parquet(pf)
    stock_df = stock_df.sort_values("trade_date", ascending=True, ignore_index=True)
    dates = stock_df["trade_date"].values.astype(str)

    idx = np.where(dates == signal_date)[0]
    if len(idx) == 0:
        return None
    idx = idx[0]

    zg = row["zs_zg"]
    result = {}

    # 信号后 1~5 日是否每日收盘都站稳 ZG 上方
    for d in [1, 2, 3, 5]:
        if idx + d < len(stock_df):
            future_closes = stock_df["close"].values[idx + 1 : idx + d + 1]
            future_lows = stock_df["low"].values[idx + 1 : idx + d + 1]
            result[f"close_above_zg_{d}d"] = bool(np.all(future_closes > zg))
            result[f"low_above_zg_{d}d"] = bool(np.all(future_lows > zg))
            # 次日表现
            if d == 1:
                result["next_day_ret"] = round(
                    (float(stock_df["close"].values[idx + 1]) / float(row["close"]) - 1) * 100, 2
                )
                result["next_day_open_ret"] = round(
                    (float(stock_df["open"].values[idx + 1]) / float(row["close"]) - 1) * 100, 2
                )
        else:
            result[f"close_above_zg_{d}d"] = None
            result[f"low_above_zg_{d}d"] = None
            if d == 1:
                result["next_day_ret"] = None
                result["next_day_open_ret"] = None

    # 信号后第 3 日入场（确认后买入），计算后续收益
    if idx + 3 < len(stock_df):
        entry_price = float(stock_df["close"].values[idx + 3])
        result["confirm_entry_price"] = round(entry_price, 2)
        for n in [5, 10, 20]:
            end_idx = idx + 3 + n
            if end_idx < len(stock_df):
                fc = stock_df["close"].values[idx + 4 : end_idx + 1]
                fh = stock_df["high"].values[idx + 4 : end_idx + 1]
                fl = stock_df["low"].values[idx + 4 : end_idx + 1]
                result[f"confirm_ret_{n}d"] = round((float(fc[-1]) / entry_price - 1) * 100, 2)
                result[f"confirm_max_gain_{n}d"] = round((float(fh.max()) / entry_price - 1) * 100, 2)
                result[f"confirm_max_dd_{n}d"] = round((float(fl.min()) / entry_price - 1) * 100, 2)
            else:
                result[f"confirm_ret_{n}d"] = None
                result[f"confirm_max_gain_{n}d"] = None
                result[f"confirm_max_dd_{n}d"] = None
    else:
        result["confirm_entry_price"] = None
        for n in [5, 10, 20]:
            result[f"confirm_ret_{n}d"] = None
            result[f"confirm_max_gain_{n}d"] = None
            result[f"confirm_max_dd_{n}d"] = None

    # 回踩入场：信号后 5 日内最低价作为入场价
    if idx + 5 < len(stock_df):
        pullback_lows = stock_df["low"].values[idx + 1 : idx + 6]
        pullback_entry = float(pullback_lows.min())
        result["pullback_entry_price"] = round(pullback_entry, 2)
        pullback_idx = idx + 1 + int(np.argmin(pullback_lows))
        for n in [5, 10, 20]:
            end_idx = pullback_idx + n
            if end_idx < len(stock_df):
                fc = stock_df["close"].values[pullback_idx + 1 : end_idx + 1]
                fh = stock_df["high"].values[pullback_idx + 1 : end_idx + 1]
                fl = stock_df["low"].values[pullback_idx + 1 : end_idx + 1]
                if len(fc) > 0:
                    result[f"pullback_ret_{n}d"] = round((float(fc[-1]) / pullback_entry - 1) * 100, 2)
                    result[f"pullback_max_gain_{n}d"] = round((float(fh.max()) / pullback_entry - 1) * 100, 2)
                else:
                    result[f"pullback_ret_{n}d"] = None
                    result[f"pullback_max_gain_{n}d"] = None
            else:
                result[f"pullback_ret_{n}d"] = None
                result[f"pullback_max_gain_{n}d"] = None
    else:
        result["pullback_entry_price"] = None
        for n in [5, 10, 20]:
            result[f"pullback_ret_{n}d"] = None
            result[f"pullback_max_gain_{n}d"] = None

    return result


import time

print("正在补充确认指标（需逐只读取 parquet）...")
t0 = time.time()
enriched = []
for i, (_, row) in enumerate(df.iterrows()):
    if (i + 1) % 2000 == 0:
        elapsed = time.time() - t0
        print(f"  进度 {i+1}/{len(df)}  {elapsed:.0f}s")
    extra = enrich_signal(row)
    enriched.append(extra)

df_extra = pd.DataFrame(enriched)
df = pd.concat([df, df_extra], axis=1)
print(f"数据补充完成，耗时 {time.time()-t0:.0f}s\n")

# ====================================================================
# 第二部分：策略对比分析
# ====================================================================


def print_strategy_stats(name, rets, gains=None, label=""):
    valid = rets.dropna()
    if len(valid) < 10:
        print(f"  {name}: 样本不足 ({len(valid)})")
        return
    n = len(valid)
    win = (valid > 0).sum()
    bullish = (valid > 5).sum()
    bearish = (valid < -5).sum()
    neutral = n - bullish - bearish
    print(f"  {name} (n={n:,}){label}")
    print(f"    平均收益: {valid.mean():+.2f}%  中位: {valid.median():+.2f}%  胜率: {win/n*100:.1f}%")
    print(f"    上涨(>5%): {bullish/n*100:.1f}%  震荡(±5%): {neutral/n*100:.1f}%  下跌(<-5%): {bearish/n*100:.1f}%")
    if gains is not None:
        vg = gains.dropna()
        if len(vg) > 0:
            print(f"    平均最大涨幅: {vg.mean():+.2f}%")


print("=" * 80)
print("  策略 A: 涨停当日收盘买入（追涨）")
print("=" * 80)
for n in [5, 10, 20]:
    print_strategy_stats(f"{n}日", df[f"ret_{n}d"], df.get(f"max_gain_{n}d"))

print("\n" + "=" * 80)
print("  策略 B: 涨停后第3日收盘确认买入（等确认）")
print("=" * 80)
for n in [5, 10, 20]:
    print_strategy_stats(f"{n}日", df[f"confirm_ret_{n}d"], df.get(f"confirm_max_gain_{n}d"))

print("\n" + "=" * 80)
print("  策略 C: 涨停后5日内最低点买入（回踩低吸）")
print("=" * 80)
for n in [5, 10, 20]:
    print_strategy_stats(f"{n}日", df[f"pullback_ret_{n}d"], df.get(f"pullback_max_gain_{n}d"))

# ====================================================================
# 第三部分：确认条件的筛选效果
# ====================================================================

print("\n\n" + "=" * 80)
print("  确认条件筛选：哪些突破是「真突破」？")
print("=" * 80)

conditions = {
    "无筛选（全部）": df,
    "次日收阳 (次日收益>0)": df[df["next_day_ret"] > 0] if "next_day_ret" in df.columns else pd.DataFrame(),
    "次日大涨 (次日收益>3%)": df[df["next_day_ret"] > 3] if "next_day_ret" in df.columns else pd.DataFrame(),
    "连续2日收盘>ZG": df[df["close_above_zg_2d"] == True] if "close_above_zg_2d" in df.columns else pd.DataFrame(),
    "连续3日收盘>ZG": df[df["close_above_zg_3d"] == True] if "close_above_zg_3d" in df.columns else pd.DataFrame(),
    "连续3日最低>ZG（强确认）": df[df["low_above_zg_3d"] == True] if "low_above_zg_3d" in df.columns else pd.DataFrame(),
    "连续5日收盘>ZG": df[df["close_above_zg_5d"] == True] if "close_above_zg_5d" in df.columns else pd.DataFrame(),
    "突破幅度>5%": df[df["breakout_pct"] > 5],
    "突破幅度<5%（温和突破）": df[(df["breakout_pct"] > 0) & (df["breakout_pct"] <= 5)],
}

print(f"\n{'条件':<30} {'样本':>6} {'20日胜率':>8} {'20日均收':>8} {'20日中位':>8} {'上涨%':>6} {'下跌%':>6}")
print("-" * 80)

results_for_report = []
for cond_name, sub in conditions.items():
    if len(sub) < 10:
        continue
    v = sub[sub["ret_20d"].notna()]
    if len(v) < 10:
        continue
    n = len(v)
    wr = (v["ret_20d"] > 0).sum() / n * 100
    mr = v["ret_20d"].mean()
    md = v["ret_20d"].median()
    bp = (v["ret_20d"] > 5).sum() / n * 100
    brp = (v["ret_20d"] < -5).sum() / n * 100
    print(f"  {cond_name:<28} {n:>6,} {wr:>7.1f}% {mr:>+7.2f}% {md:>+7.2f}% {bp:>5.1f}% {brp:>5.1f}%")
    results_for_report.append({
        "condition": cond_name, "samples": n,
        "win_rate_20d": round(wr, 1), "mean_ret_20d": round(mr, 2),
        "median_ret_20d": round(md, 2),
        "bullish_20d": round(bp, 1), "bearish_20d": round(brp, 1),
    })

# ====================================================================
# 第四部分：「确认后入场」的最优时机
# ====================================================================

print("\n\n" + "=" * 80)
print("  最优策略：确认后买入 vs 直接追涨（20日口径对比）")
print("=" * 80)

# 对「连续3日收盘>ZG」的子集，对比不同入场方式
if "close_above_zg_3d" in df.columns:
    confirmed = df[df["close_above_zg_3d"] == True].copy()
    if len(confirmed) > 50:
        print(f"\n筛选条件: 涨停后连续3日收盘站稳ZG上方 (n={len(confirmed):,})")
        print(f"（占全部信号的 {len(confirmed)/len(df)*100:.1f}%）\n")

        # 策略 A: 涨停当日买，持20日
        va = confirmed[confirmed["ret_20d"].notna()]["ret_20d"]
        # 策略 B: 第3日确认后买，持20日
        vb = confirmed[confirmed["confirm_ret_20d"].notna()]["confirm_ret_20d"]

        for label, v in [("A. 涨停当日买入", va), ("B. 第3日确认后买入", vb)]:
            if len(v) < 10:
                continue
            n = len(v)
            print(f"  {label} (n={n:,})")
            print(f"    平均: {v.mean():+.2f}%  中位: {v.median():+.2f}%  胜率: {(v>0).sum()/n*100:.1f}%")
            print(f"    上涨>5%: {(v>5).sum()/n*100:.1f}%  下跌<-5%: {(v<-5).sum()/n*100:.1f}%")
            print(f"    P25: {v.quantile(0.25):+.2f}%  P75: {v.quantile(0.75):+.2f}%")

# ====================================================================
# 第五部分：「次日表现」分析 —— 追涨的即时风险
# ====================================================================

print("\n\n" + "=" * 80)
print("  追涨的即时风险：涨停次日表现")
print("=" * 80)

if "next_day_ret" in df.columns:
    ndr = df["next_day_ret"].dropna()
    ndo = df["next_day_open_ret"].dropna()
    print(f"  次日开盘 vs 涨停收盘: 平均 {ndo.mean():+.2f}%  中位 {ndo.median():+.2f}%")
    print(f"  次日收盘 vs 涨停收盘: 平均 {ndr.mean():+.2f}%  中位 {ndr.median():+.2f}%")
    print(f"  次日继续涨: {(ndr>0).sum()/len(ndr)*100:.1f}%")
    print(f"  次日涨停 (>9.5%): {(ndr>9.5).sum()/len(ndr)*100:.1f}%")
    print(f"  次日跌 >3%: {(ndr<-3).sum()/len(ndr)*100:.1f}%")
    print(f"  次日跌 >5%: {(ndr<-5).sum()/len(ndr)*100:.1f}%")

    # 次日分组
    print(f"\n  按次日走势分组后的20日表现:")
    df["next_day_group"] = pd.cut(
        df["next_day_ret"],
        bins=[-100, -5, -2, 0, 3, 9.5, 100],
        labels=["暴跌<-5%", "下跌-5~-2%", "微跌-2~0%", "微涨0~3%", "大涨3~9.5%", "涨停>9.5%"]
    )
    for grp_name, grp in df.groupby("next_day_group", observed=True):
        v20 = grp[grp["ret_20d"].notna()]["ret_20d"]
        if len(v20) < 10:
            continue
        n = len(v20)
        print(f"    {grp_name:<14}  n={n:>5,}  "
              f"20日均收: {v20.mean():+6.2f}%  胜率: {(v20>0).sum()/n*100:5.1f}%  "
              f"上涨: {(v20>5).sum()/n*100:5.1f}%  下跌: {(v20<-5).sum()/n*100:5.1f}%")

# 保存结果
with open(OUTPUT / "deep_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results_for_report, f, ensure_ascii=False, indent=2)

print(f"\n\n[文件] {OUTPUT / 'deep_analysis_results.json'}")
