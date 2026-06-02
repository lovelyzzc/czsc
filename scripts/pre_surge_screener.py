"""主升前 CZSC 特征筛选与回测

基于 000636 主升浪前的缠论结构特征，提炼 7 项可量化的筛选条件，
在全 A 股日线数据上逐日扫描打分，对比三种退出策略的回测效果。

筛选条件（打分制，满分 7，>= 5 触发买入）：
  C1 中枢上移        — 最近 3 个中枢 ZD 递增
  C2 窄幅收敛        — 最近中枢宽度 < 10%
  C3 向上笔力度突增  — 最近向上笔力度 > 前均值 x 1.5
  C4 向下笔力度递减  — 最近两向下笔力度递减且幅度 < 12%
  C5 DIF 零轴附近    — DIF 绝对值 < 相对阈值 或刚上穿零轴
  C6 缩量后放量      — 前期量比 < 0.8，当前量比 > 1.2
  C7 不回前中枢      — 最近下跌笔低点 > 前一中枢 ZG

退出策略对比：
  A 缠论退出    — 向下笔力度 > 前一向下笔 x 1.2
  B 固定持有    — 持有 20 个交易日
  C 跟踪止损    — 从最高点回撤 15%

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

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "pre_surge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
SCORE_THRESHOLD = 5
HOLD_DAYS_B = 20
TRAILING_STOP_C = 0.15


def _extract_zs(bis_list):
    """从笔列表提取非重叠中枢"""
    zs_list = []
    i = 0
    while i < len(bis_list) - 2:
        b1, b2, b3 = bis_list[i], bis_list[i + 1], bis_list[i + 2]
        zg = min(b1.high, b2.high, b3.high)
        zd = max(b1.low, b2.low, b3.low)
        if zg > zd:
            zs = {"sdt": b1.sdt, "edt": b3.edt, "zg": zg, "zd": zd, "bis": 3}
            j = i + 3
            while j < len(bis_list):
                bj = bis_list[j]
                if bj.high >= zd and bj.low <= zg:
                    zs["edt"] = bj.edt
                    zs["bis"] += 1
                    j += 1
                else:
                    break
            zs_list.append(zs)
            i = j
        else:
            i += 1
    return zs_list


def _score_stock(bis, zs_list, dif_val, vol_ratio_prev, vol_ratio_now):
    """对当前状态打分，返回 (总分, 各项得分 dict)"""
    scores = {}

    # C1: 中枢上移 — 最近 3 个中枢 ZD 递增
    if len(zs_list) >= 3:
        scores["C1"] = int(zs_list[-1]["zd"] > zs_list[-2]["zd"] > zs_list[-3]["zd"])
    else:
        scores["C1"] = 0

    # C2: 窄幅收敛 — 最近中枢宽度 < 10%
    if zs_list:
        width = (zs_list[-1]["zg"] - zs_list[-1]["zd"]) / zs_list[-1]["zd"]
        scores["C2"] = int(width < 0.10)
    else:
        scores["C2"] = 0

    # C3: 向上笔力度突增
    up_bis = [b for b in bis if b.direction.value == "向上"]
    if len(up_bis) >= 3:
        recent = up_bis[-1].power
        prev_avg = np.mean([b.power for b in up_bis[-3:-1]])
        scores["C3"] = int(recent > prev_avg * 1.5) if prev_avg > 0 else 0
    else:
        scores["C3"] = 0

    # C4: 向下笔力度递减且幅度小
    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if len(dn_bis) >= 2:
        last_dn = dn_bis[-1]
        prev_dn = dn_bis[-2]
        last_pct = (last_dn.high / last_dn.low - 1)
        scores["C4"] = int(last_dn.power < prev_dn.power and last_pct < 0.12)
    else:
        scores["C4"] = 0

    # C5: DIF 零轴附近（相对于价格的比例）
    if dif_val is not None:
        scores["C5"] = int(abs(dif_val) < 0.5)
    else:
        scores["C5"] = 0

    # C6: 缩量后放量
    scores["C6"] = int(vol_ratio_prev < 0.8 and vol_ratio_now > 1.2)

    # C7: 不回前中枢
    if len(zs_list) >= 2 and len(dn_bis) >= 1:
        scores["C7"] = int(dn_bis[-1].low > zs_list[-2]["zg"])
    else:
        scores["C7"] = 0

    return sum(scores.values()), scores


def _process_stock(parquet_path: str) -> list[dict] | None:
    """单只股票：CZSC 分析 + 逐日打分 + 生成买入信号和三种退出的 holds"""
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
    if len(bis) < 8:
        return None

    zs_list = _extract_zs(bis)
    if len(zs_list) < 3:
        return None

    close = df["close"].values
    vol = df["vol"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values

    bi_end_dates = {bi.edt: bi for bi in bis}
    bi_date_to_idx = {}
    for i, bi in enumerate(bis):
        bi_date_to_idx[bi.edt] = i

    symbol = code
    n = len(df)
    start_idx = max(120, n // 4)

    signals = []

    for idx in range(start_idx, n):
        dt = dates[idx]

        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dt)]
        if len(bis_up_to) < 6:
            continue

        zs_up_to = _extract_zs(bis_up_to)
        if len(zs_up_to) < 3:
            continue

        vr_now = vol_ma5[idx] / vol_ma20[idx] if vol_ma20[idx] > 0 else 1.0
        vr_prev = vol_ma5[idx - 5] / vol_ma20[idx - 5] if idx >= 5 and vol_ma20[idx - 5] > 0 else 1.0
        dif_val = dif[idx]

        total, detail = _score_stock(bis_up_to, zs_up_to, dif_val, vr_prev, vr_now)

        if total >= SCORE_THRESHOLD and abs(pct_chg[idx]) < 9.8:
            signals.append({
                "dt": pd.Timestamp(dt),
                "idx": idx,
                "symbol": symbol,
                "close": close[idx],
                "score": total,
            })

    if not signals:
        return None

    MIN_GAP = 10
    filtered = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - filtered[-1]["idx"] >= MIN_GAP:
            filtered.append(s)
    signals = filtered

    holds_a, holds_b, holds_c = [], [], []

    for sig in signals:
        entry_idx = sig["idx"]
        entry_price = sig["close"]
        entry_dt = sig["dt"]

        # --- Exit B: fixed 20-day hold ---
        exit_idx_b = min(entry_idx + HOLD_DAYS_B, n - 1)
        for j in range(entry_idx, exit_idx_b + 1):
            holds_b.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                            "pos": 1, "price": close[j]})

        # --- Exit C: trailing stop 15% ---
        peak = entry_price
        for j in range(entry_idx, n):
            if close[j] > peak:
                peak = close[j]
            holds_c.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                            "pos": 1, "price": close[j]})
            if (peak - close[j]) / peak >= TRAILING_STOP_C:
                break

        # --- Exit A: BI-based exit ---
        bis_after = [bi for bi in bis if bi.sdt >= entry_dt]
        exit_done = False
        dn_power_ref = None

        dn_before = [b for b in bis if b.direction.value == "向下" and b.edt <= entry_dt]
        if dn_before:
            dn_power_ref = dn_before[-1].power

        for bi in bis_after:
            bi_end_idx = None
            for k in range(entry_idx, n):
                if pd.Timestamp(dates[k]) >= bi.edt:
                    bi_end_idx = k
                    break
            if bi_end_idx is None:
                bi_end_idx = n - 1

            for j in range(entry_idx, min(bi_end_idx + 1, n)):
                if pd.Timestamp(dates[j]) not in {h["dt"] for h in holds_a if h["symbol"] == symbol}:
                    holds_a.append({"dt": pd.Timestamp(dates[j]), "symbol": symbol,
                                    "pos": 1, "price": close[j]})

            if bi.direction.value == "向下" and dn_power_ref is not None:
                if bi.power > dn_power_ref * 1.2:
                    exit_done = True
                    break
                dn_power_ref = bi.power

        if not exit_done:
            max_hold = min(entry_idx + 60, n)
            for j in range(entry_idx, max_hold):
                dt_j = pd.Timestamp(dates[j])
                if dt_j not in {h["dt"] for h in holds_a if h["symbol"] == symbol}:
                    holds_a.append({"dt": dt_j, "symbol": symbol,
                                    "pos": 1, "price": close[j]})

    # 计算逐笔交易盈亏
    def _calc_pairs(holds_list, entry_signals):
        pairs = []
        for sig in entry_signals:
            entry_p = sig["close"]
            entry_d = sig["dt"]
            exits = [h for h in holds_list if h["symbol"] == symbol and h["dt"] > entry_d]
            if exits:
                exit_h = exits[-1]
                exit_p = exit_h["price"]
                ret = (exit_p / entry_p - 1) * 100
                pairs.append({"symbol": symbol, "entry_dt": entry_d,
                              "entry_price": entry_p, "exit_price": exit_p, "ret_pct": ret})
        return pairs

    pairs_a = _calc_pairs(holds_a, signals)
    pairs_b = _calc_pairs(holds_b, signals)
    pairs_c = _calc_pairs(holds_c, signals)

    result = {
        "symbol": symbol,
        "n_signals": len(signals),
        "holds_A": holds_a,
        "holds_B": holds_b,
        "holds_C": holds_c,
        "pairs_A": pairs_a,
        "pairs_B": pairs_b,
        "pairs_C": pairs_c,
    }
    return [result]


def main():
    print("=" * 70)
    print("  主升前 CZSC 特征筛选 — 全 A 股回测")
    print("=" * 70)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] 发现 {len(parquet_files)} 只个股")

    if not parquet_files:
        print("[ERROR] 未找到 parquet 文件")
        return

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] {n_workers} 个进程 (spawn)")

    t0 = time.time()
    file_list = [str(p) for p in parquet_files]

    all_results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_stock, file_list, chunksize=20), 1):
            if res is not None:
                all_results.extend(res)
            if i % 500 == 0 or i == len(file_list):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(file_list)}] 有信号 {len(all_results)} | "
                      f"{elapsed:.0f}s | {speed:.0f} 只/s | ETA {eta:.0f}s")

    print(f"\n[Phase 1 完成] 耗时 {time.time()-t0:.0f}s | {len(all_results)} 只股票产生信号")

    if not all_results:
        print("[ERROR] 无股票产生买入信号，请降低 SCORE_THRESHOLD")
        return

    total_signals = sum(r["n_signals"] for r in all_results)
    print(f"  总买入信号数: {total_signals}")

    modes = {
        "A_缠论退出": ("holds_A", "pairs_A"),
        "B_固定持有20日": ("holds_B", "pairs_B"),
        "C_跟踪止损15%": ("holds_C", "pairs_C"),
    }

    all_stats = []
    for tag, (holds_key, pairs_key) in modes.items():
        print(f"\n{'='*60}")
        print(f"  [{tag}]")
        print(f"{'='*60}")

        all_holds = []
        all_pairs = []
        for r in all_results:
            if r[holds_key]:
                all_holds.extend(r[holds_key])
            if r[pairs_key]:
                all_pairs.extend(r[pairs_key])

        if not all_holds:
            print("  无持仓数据")
            continue

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
        stats["n_stocks"] = len(set(r["symbol"] for r in all_results if r[holds_key]))
        stats["n_signals"] = total_signals

        if all_pairs:
            rets = [p["ret_pct"] for p in all_pairs]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            stats["交易笔数"] = len(rets)
            stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
            stats["平均盈利"] = f"{np.mean(wins):.1f}%" if wins else "0%"
            stats["平均亏损"] = f"{np.mean(losses):.1f}%" if losses else "0%"
            stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
            stats["平均收益"] = f"{np.mean(rets):.1f}%"

        for k in ["年化收益", "夏普比率", "最大回撤", "卡玛比率",
                   "交易笔数", "胜率", "平均盈利", "平均亏损", "盈亏比", "平均收益"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            out_html = OUTPUT_DIR / f"{tag}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"主升前特征筛选 - {tag}",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"    HTML: {out_html.name}")
        except Exception as e:
            print(f"    HTML 报告失败: {e}")

        all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] 所有退出策略均无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 80)
    print("  主升前特征筛选 — 三种退出策略对比")
    print("=" * 80)
    display_cols = [c for c in [
        "n_stocks", "交易笔数", "胜率", "盈亏比", "平均收益",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR}")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
