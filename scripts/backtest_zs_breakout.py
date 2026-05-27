"""全 A 股回测：「涨停突破最近中枢上沿」信号的后续走势统计

扫描逻辑：
1. 对每只股票运行 CZSC 日线分析
2. 逐日检查：当日涨停（涨幅 >= 9.5%）且收盘价突破最近中枢 ZG
3. 记录信号触发后 5/10/20 日的收益、最大涨幅、最大回撤
4. 汇总统计，输出概率分布
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from czsc import CZSC, ZS, Freq, format_standard_kline

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
OUTPUT = Path(__file__).resolve().parent / "_output"
OUTPUT.mkdir(parents=True, exist_ok=True)

LIMIT_UP_PCT = 9.5
HOLD_DAYS = [5, 10, 20]


def extract_zs_list(bi_list):
    """非重叠中枢序列提取"""
    result = []
    i = 0
    while i <= len(bi_list) - 3:
        try:
            zs = ZS(bi_list[i : i + 3])
        except Exception:
            i += 1
            continue
        if not zs.is_valid():
            i += 1
            continue
        j = i + 3
        zg, zd = zs.zg, zs.zd
        while j < len(bi_list):
            bi = bi_list[j]
            bh = max(float(bi.fx_a.fx), float(bi.fx_b.fx))
            bl = min(float(bi.fx_a.fx), float(bi.fx_b.fx))
            if bl < zg and bh > zd:
                j += 1
            else:
                break
        result.append(zs)
        i = j
    return result


def scan_stock(parquet_path: Path) -> list[dict]:
    """扫描单只股票，返回所有信号触发记录"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return []

    if len(df) < 100:
        return []

    df = df.sort_values("trade_date", ascending=True, ignore_index=True)

    # 准备 bars
    dfc = df.copy()
    dfc["dt"] = pd.to_datetime(dfc["trade_date"])
    dfc["symbol"] = dfc["ts_code"]
    dfc["vol"] = (dfc["vol"] * 100).astype(int)
    dfc["amount"] = (dfc["amount"] * 1000).astype(float)
    dfc = dfc[["symbol", "dt", "open", "high", "low", "close", "vol", "amount"]]

    try:
        bars = format_standard_kline(dfc, freq=Freq.D)
    except Exception:
        return []

    if len(bars) < 100:
        return []

    try:
        c = CZSC(bars)
    except Exception:
        return []

    bi_list = list(c.bi_list)
    if len(bi_list) < 6:
        return []

    zs_list = extract_zs_list(bi_list)
    if not zs_list:
        return []

    # 用 DataFrame 做后续收益计算
    closes = df["close"].values
    pct_chgs = df["pct_chg"].values if "pct_chg" in df.columns else np.diff(closes, prepend=closes[0]) / np.maximum(closes, 1e-9) * 100
    dates = df["trade_date"].values

    records = []

    for bar_idx in range(1, len(df)):
        pct = pct_chgs[bar_idx]
        if pct < LIMIT_UP_PCT:
            continue

        close = closes[bar_idx]
        dt = dates[bar_idx]

        # 找到该日期时已经形成的中枢
        bar_dt = pd.to_datetime(dt)
        latest_zs = None
        for zs in zs_list:
            if zs.edt <= bar_dt:
                latest_zs = zs

        if latest_zs is None:
            continue

        # 检查：收盘价突破中枢上沿
        if close <= latest_zs.zg:
            continue

        # 额外条件：涨停当日从中枢区间内或下方拉起
        prev_close = closes[bar_idx - 1]
        if prev_close > latest_zs.zg * 1.05:
            continue

        # 计算后续 N 日收益
        future = {}
        for n in HOLD_DAYS:
            end_idx = bar_idx + n
            if end_idx >= len(df):
                future[f"ret_{n}d"] = None
                future[f"max_gain_{n}d"] = None
                future[f"max_dd_{n}d"] = None
                future[f"back_below_zg_{n}d"] = None
                continue

            future_closes = closes[bar_idx + 1 : end_idx + 1]
            future_lows = df["low"].values[bar_idx + 1 : end_idx + 1]
            future_highs = df["high"].values[bar_idx + 1 : end_idx + 1]

            ret = (future_closes[-1] / close - 1) * 100
            max_gain = (future_highs.max() / close - 1) * 100
            max_dd = (future_lows.min() / close - 1) * 100
            back_below = future_lows.min() < latest_zs.zg

            future[f"ret_{n}d"] = round(ret, 2)
            future[f"max_gain_{n}d"] = round(max_gain, 2)
            future[f"max_dd_{n}d"] = round(max_dd, 2)
            future[f"back_below_zg_{n}d"] = bool(back_below)

        records.append({
            "symbol": str(df.iloc[bar_idx]["ts_code"]),
            "date": str(dt),
            "close": round(float(close), 2),
            "pct_chg": round(float(pct), 2),
            "zs_zg": round(float(latest_zs.zg), 2),
            "zs_zd": round(float(latest_zs.zd), 2),
            "zs_sdt": str(latest_zs.sdt.strftime("%Y-%m-%d")),
            "zs_edt": str(latest_zs.edt.strftime("%Y-%m-%d")),
            "prev_close": round(float(prev_close), 2),
            "breakout_pct": round((float(close) / float(latest_zs.zg) - 1) * 100, 2),
            **future,
        })

    return records


def main():
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[扫描] 共 {len(parquet_files)} 只股票")

    all_records = []
    t0 = time.time()
    errors = 0

    for i, pf in enumerate(parquet_files):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(parquet_files) - i - 1) / speed
            print(f"  进度 {i+1}/{len(parquet_files)}  已找到 {len(all_records)} 个信号  "
                  f"速度 {speed:.0f} 只/s  预计剩余 {eta:.0f}s")
        try:
            recs = scan_stock(pf)
            all_records.extend(recs)
        except Exception:
            errors += 1

    elapsed = time.time() - t0
    print(f"\n[完成] 耗时 {elapsed:.1f}s, 扫描 {len(parquet_files)} 只, "
          f"错误 {errors} 只, 找到 {len(all_records)} 个信号实例")

    if not all_records:
        print("未找到任何信号实例！")
        return

    # 保存原始数据
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT / "zs_breakout_signals.csv", index=False, encoding="utf-8-sig")

    # 统计分析
    print("\n" + "=" * 70)
    print("  「涨停突破最近中枢上沿」信号统计")
    print("=" * 70)
    print(f"  样本总数: {len(df)}")
    print(f"  覆盖股票: {df['symbol'].nunique()} 只")
    print(f"  时间跨度: {df['date'].min()} ~ {df['date'].max()}")

    for n in HOLD_DAYS:
        col_ret = f"ret_{n}d"
        col_gain = f"max_gain_{n}d"
        col_dd = f"max_dd_{n}d"
        col_back = f"back_below_zg_{n}d"

        valid = df[df[col_ret].notna()].copy()
        if valid.empty:
            continue

        total = len(valid)
        # 分类
        bullish = valid[valid[col_ret] > 5]
        bearish = valid[valid[col_ret] < -5]
        neutral = valid[(valid[col_ret] >= -5) & (valid[col_ret] <= 5)]

        back_below = valid[valid[col_back] == True]

        print(f"\n  --- {n} 日后统计 (有效样本 {total}) ---")
        print(f"  平均收益: {valid[col_ret].mean():+.2f}%")
        print(f"  中位收益: {valid[col_ret].median():+.2f}%")
        print(f"  胜率(>0): {(valid[col_ret] > 0).sum() / total * 100:.1f}%")
        print(f"  平均最大涨幅: {valid[col_gain].mean():+.2f}%")
        print(f"  平均最大回撤: {valid[col_dd].mean():.2f}%")
        print(f"  走势分类:")
        print(f"    上涨(>{'+'}5%):  {len(bullish):4d} 次  {len(bullish)/total*100:5.1f}%")
        print(f"    震荡(±5%):   {len(neutral):4d} 次  {len(neutral)/total*100:5.1f}%")
        print(f"    下跌(<-5%):  {len(bearish):4d} 次  {len(bearish)/total*100:5.1f}%")
        print(f"  跌回ZG以下:    {len(back_below):4d} 次  {len(back_below)/total*100:5.1f}%")

    # 输出收益分布
    for n in HOLD_DAYS:
        col = f"ret_{n}d"
        valid = df[df[col].notna()]
        if valid.empty:
            continue
        print(f"\n  {n}日收益分位数:")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            print(f"    P{int(q*100):2d}: {valid[col].quantile(q):+.2f}%")

    # 保存统计结果为 JSON（供 HTML 报告使用）
    stats = {"total": len(df), "stocks": int(df["symbol"].nunique()),
             "date_range": [str(df["date"].min()), str(df["date"].max())]}
    for n in HOLD_DAYS:
        valid = df[df[f"ret_{n}d"].notna()]
        if valid.empty:
            continue
        t = len(valid)
        stats[f"{n}d"] = {
            "samples": t,
            "mean_ret": round(float(valid[f"ret_{n}d"].mean()), 2),
            "median_ret": round(float(valid[f"ret_{n}d"].median()), 2),
            "win_rate": round(float((valid[f"ret_{n}d"] > 0).sum() / t * 100), 1),
            "bullish_pct": round(float((valid[f"ret_{n}d"] > 5).sum() / t * 100), 1),
            "neutral_pct": round(float(((valid[f"ret_{n}d"] >= -5) & (valid[f"ret_{n}d"] <= 5)).sum() / t * 100), 1),
            "bearish_pct": round(float((valid[f"ret_{n}d"] < -5).sum() / t * 100), 1),
            "avg_max_gain": round(float(valid[f"max_gain_{n}d"].mean()), 2),
            "avg_max_dd": round(float(valid[f"max_dd_{n}d"].mean()), 2),
            "back_below_zg_pct": round(float(valid[f"back_below_zg_{n}d"].sum() / t * 100), 1),
            "quantiles": {
                str(int(q * 100)): round(float(valid[f"ret_{n}d"].quantile(q)), 2)
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]
            },
        }

    with open(OUTPUT / "zs_breakout_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[文件] 原始数据: {OUTPUT / 'zs_breakout_signals.csv'}")
    print(f"[文件] 统计结果: {OUTPUT / 'zs_breakout_stats.json'}")


if __name__ == "__main__":
    main()
