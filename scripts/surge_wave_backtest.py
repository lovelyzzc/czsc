"""主升浪追涨策略回测

识别已确认进入主升浪加速阶段的标的，回测追涨效果。

入场条件（7 项特征，等权打分，>= 5 触发）：
  S1 向上笔加速    — 最近 3 向上笔力度递增
  S2 上下力度比    — 向上笔均力度 / 向下笔均力度 > 1.5
  S3 脱离中枢      — 收盘价 > 最近中枢 ZG × 1.2
  S4 低点抬升      — 最近 2 向下笔低点逐笔抬升
  S5 MA 多头扩散   — MA5>MA10>MA20 且 MA5-MA20 散度 > 15%
  S6 DIF 加速      — DIF > 0 且 DIF > 5 日前 DIF
  S7 涨幅确认      — 最近 20 日涨幅 > 15%

退出策略对比：
  A 笔结构止损(SL2)  — 跌破最近向下笔低点
  B 跟踪止损 20%     — 从最高点回撤 20%
  C 趋势反转退出     — 向下笔力度 > 前一向上笔力度（趋势反转）

最大持有 60 天（主升浪持有期更长）。
数据源：~/.ts_data_cache/a_stock_daily_qfq/
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from wbt import generate_backtest_report

from czsc import CZSC, Freq, WeightBacktest, format_standard_kline

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_wave"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
SCORE_THRESHOLD = 5
MAX_HOLD_DAYS = 60
TRAILING_STOP = 0.20


def _extract_zs(bis_list):
    zs_list = []
    i = 0
    while i < len(bis_list) - 2:
        b1, b2, b3 = bis_list[i], bis_list[i + 1], bis_list[i + 2]
        zg = min(b1.high, b2.high, b3.high)
        zd = max(b1.low, b2.low, b3.low)
        if zg > zd:
            zs = {"zg": zg, "zd": zd, "bis": 3}
            j = i + 3
            while j < len(bis_list):
                bj = bis_list[j]
                if bj.high >= zd and bj.low <= zg:
                    zs["bis"] += 1
                    j += 1
                else:
                    break
            zs_list.append(zs)
            i = j
        else:
            i += 1
    return zs_list


def _score_surge(bis_up_to, zs_up_to, close_val, dif_now, dif_5ago,
                 ma5, ma10, ma20, ret20):
    """主升浪 7 项打分"""
    feats = {}

    up_bis = [b for b in bis_up_to if b.direction.value == "向上"]
    dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]

    # S1: 向上笔加速 — 最近 3 向上笔力度递增
    if len(up_bis) >= 3:
        p1, p2, p3 = up_bis[-3].power, up_bis[-2].power, up_bis[-1].power
        feats["S1_笔力加速"] = int(p3 > p2 > p1 and p1 > 0)
    else:
        feats["S1_笔力加速"] = 0

    # S2: 上下力度比 > 1.5
    if len(up_bis) >= 2 and len(dn_bis) >= 2:
        up_avg = np.mean([b.power for b in up_bis[-3:]])
        dn_avg = np.mean([b.power for b in dn_bis[-3:]])
        feats["S2_力度比"] = int(up_avg > dn_avg * 1.5) if dn_avg > 0 else 0
    else:
        feats["S2_力度比"] = 0

    # S3: 脱离中枢 — 收盘价 > 最近中枢 ZG × 1.2
    if zs_up_to:
        feats["S3_脱离中枢"] = int(close_val > zs_up_to[-1]["zg"] * 1.2)
    else:
        feats["S3_脱离中枢"] = 0

    # S4: 低点抬升 — 最近 2 向下笔低点递升
    if len(dn_bis) >= 2:
        feats["S4_低点抬升"] = int(dn_bis[-1].low > dn_bis[-2].low)
    else:
        feats["S4_低点抬升"] = 0

    # S5: MA 多头扩散
    if ma5 > ma10 > ma20:
        spread = (ma5 - ma20) / ma20 * 100
        feats["S5_MA扩散"] = int(spread > 15)
    else:
        feats["S5_MA扩散"] = 0

    # S6: DIF 加速
    feats["S6_DIF加速"] = int(dif_now > 0 and dif_now > dif_5ago)

    # S7: 涨幅确认 — 20 日涨幅 > 15%
    feats["S7_涨幅确认"] = int(ret20 > 15)

    total = sum(feats.values())
    return total, feats


def _process_stock(parquet_path: str) -> dict | None:
    """单只股票：逐日打分 + 三种退出策略"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if len(df) < MIN_BARS:
        return None

    code = df["ts_code"].iloc[0]
    if code.startswith(("688", "920", "83", "43")):
        return None

    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    try:
        bars = format_standard_kline(df, freq=Freq.D)
        c = CZSC(bars)
    except Exception:
        return None

    bis = c.bi_list
    if len(bis) < 10:
        return None

    close = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    vol = df["vol"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
    symbol = code
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    start_idx = max(120, n // 4)

    signals = []
    for idx in range(start_idx, n):
        if abs(pct_chg[idx]) >= 9.8:
            continue

        dt = dates[idx]
        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dt)]
        if len(bis_up_to) < 8:
            continue

        zs_up_to = _extract_zs(bis_up_to)

        dif_5ago = dif[idx - 5] if idx >= 5 else 0
        ret20 = (close[idx] / close[idx - 20] - 1) * 100 if idx >= 20 else 0

        total, _ = _score_surge(
            bis_up_to, zs_up_to, close[idx],
            dif[idx], dif_5ago,
            ma5[idx], ma10[idx], ma20[idx], ret20,
        )

        if total >= SCORE_THRESHOLD:
            dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]
            up_bis_up = [b for b in bis_up_to if b.direction.value == "向上"]

            sl_bi = dn_bis_up[-1].low if dn_bis_up else None
            last_up_power = up_bis_up[-1].power if up_bis_up else None

            signals.append({
                "dt": pd.Timestamp(dt),
                "idx": idx,
                "close": close[idx],
                "sl_bi": sl_bi,
                "last_up_power": last_up_power,
            })

    if not signals:
        return None

    # 去重：间距 >= 15 天（主升浪信号更稀疏）
    filtered = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - filtered[-1]["idx"] >= 15:
            filtered.append(s)
    signals = filtered

    result = {"symbol": symbol, "n_signals": len(signals)}

    # --- 策略 A: SL2 笔结构止损 ---
    holds_a, pairs_a = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds_a.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                            "pos": 1, "price": close[j]})
            if sl_price and low_arr[j] <= sl_price and j > entry_idx:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs_a.append({"symbol": symbol, "ret_pct": round(ret, 2), "exit_reason": exit_reason})

    # --- 策略 B: 跟踪止损 20% ---
    holds_b, pairs_b = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        peak = entry_price
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            if close[j] > peak:
                peak = close[j]
            holds_b.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                            "pos": 1, "price": close[j]})
            if j > entry_idx and (peak - close[j]) / peak >= TRAILING_STOP:
                exit_price = close[j]
                exit_reason = "trailing_stop"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs_b.append({"symbol": symbol, "ret_pct": round(ret, 2), "exit_reason": exit_reason})

    # --- 策略 C: 趋势反转退出 ---
    holds_c, pairs_c = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        ref_up_power = sig["last_up_power"]
        exit_price, exit_reason = None, "max_hold"

        bis_after = [bi for bi in bis if bi.sdt >= pd.Timestamp(dates[entry_idx])]
        exit_idx = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)

        for bi in bis_after:
            bi_end_idx = None
            for k in range(entry_idx, n):
                if pd.Timestamp(dates[k]) >= bi.edt:
                    bi_end_idx = k
                    break
            if bi_end_idx is None:
                bi_end_idx = n - 1

            if (bi.direction.value == "向下" and ref_up_power
                    and bi.power > ref_up_power):
                exit_idx = bi_end_idx
                exit_reason = "trend_reversal"
                break

        for j in range(entry_idx, min(exit_idx + 1, n)):
            holds_c.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                            "pos": 1, "price": close[j]})

        exit_price = close[min(exit_idx, n - 1)]
        ret = (exit_price / entry_price - 1) * 100
        pairs_c.append({"symbol": symbol, "ret_pct": round(ret, 2), "exit_reason": exit_reason})

    result["holds_A"] = holds_a
    result["pairs_A"] = pairs_a
    result["holds_B"] = holds_b
    result["pairs_B"] = pairs_b
    result["holds_C"] = holds_c
    result["pairs_C"] = pairs_c

    return result


def main():
    print("=" * 80)
    print("  主升浪追涨策略 — 全 A 股回测")
    print("=" * 80)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] {len(parquet_files)} 只个股")

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] {n_workers} 进程")

    t0 = time.time()
    file_list = [str(p) for p in parquet_files]

    all_results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_stock, file_list, chunksize=20), 1):
            if res is not None:
                all_results.append(res)
            if i % 500 == 0 or i == len(file_list):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(file_list)}] 有信号 {len(all_results)} | "
                      f"{elapsed:.0f}s | ETA {eta:.0f}s")

    print(f"\n[完成] 耗时 {time.time()-t0:.0f}s | {len(all_results)} 只股票产生信号")

    if not all_results:
        print("[ERROR] 无股票产生信号")
        return

    total_signals = sum(r["n_signals"] for r in all_results)
    print(f"  总信号数: {total_signals}")

    modes = {
        "A_笔结构止损": ("holds_A", "pairs_A"),
        "B_跟踪止损20%": ("holds_B", "pairs_B"),
        "C_趋势反转退出": ("holds_C", "pairs_C"),
    }

    all_stats = []
    for tag, (holds_key, pairs_key) in modes.items():
        print(f"\n{'='*70}")
        print(f"  [{tag}]")
        print(f"{'='*70}")

        all_holds, all_pairs = [], []
        for r in all_results:
            all_holds.extend(r[holds_key])
            all_pairs.extend(r[pairs_key])

        if not all_holds:
            print("  无持仓数据")
            continue

        n_exit = {}
        for p in all_pairs:
            reason = p["exit_reason"]
            n_exit[reason] = n_exit.get(reason, 0) + 1

        dfw = pd.DataFrame(all_holds)
        dfw = dfw.rename(columns={"pos": "weight"})
        if dfw.duplicated(subset=["dt", "symbol"]).any():
            dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
                weight=("weight", "max"), price=("price", "first"),
            )
        dfw = dfw[["dt", "symbol", "weight", "price"]]

        try:
            wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
            stats = wb.stats
        except Exception as e:
            print(f"  WeightBacktest 失败: {e}")
            continue

        stats["tag"] = tag
        stats["退出分布"] = str(n_exit)

        rets = [p["ret_pct"] for p in all_pairs]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]

        stats["交易笔数"] = len(rets)
        stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
        stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
        stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
        stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
        stats["平均收益"] = f"{np.mean(rets):.2f}%"
        stats["收益中位数"] = f"{np.median(rets):.2f}%"

        for k in ["交易笔数", "退出分布", "胜率", "盈亏比",
                   "平均盈利", "平均亏损", "平均收益", "收益中位数",
                   "年化收益", "夏普比率", "最大回撤", "卡玛比率"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            out_html = OUTPUT_DIR / f"{tag}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"主升浪策略 - {tag}",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"    HTML: {out_html.name}")
        except Exception as e:
            print(f"    HTML 报告失败: {e}")

        all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] 所有策略均无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 100)
    print("  主升浪追涨策略 — 三种退出方式对比")
    print("=" * 100)
    display_cols = [c for c in [
        "交易笔数", "胜率", "盈亏比", "平均盈利", "平均亏损",
        "平均收益", "收益中位数",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR}")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
