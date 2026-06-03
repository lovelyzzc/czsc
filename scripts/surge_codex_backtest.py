"""Surge-wave x Codex2.0 策略融合回测

将 quant-codex2.0 的 ATR 动态止损、固定百分比止损、止盈目标等策略
整合到 surge-wave 回测框架中，对比 8 组退出策略组合。

策略矩阵:
  0  baseline   信号日收盘 + SL2
  1  BD         D回调入场  + SL2 + 分级跟踪
  2  ATR        D回调入场  + ATR动态止损
  3  FIXED5     D回调入场  + -5%固定止损
  4  TP10_SL2   D回调入场  + SL2 + 10%止盈
  5  ATR_BD     D回调入场  + ATR动态止损 + 分级跟踪
  6  TP10_BD    D回调入场  + SL2 + 10%止盈 + 分级跟踪
  7  HYBRID     D回调入场  + ATR动态止损 + 10%止盈 + 分级跟踪
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

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_codex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
SCORE_THRESHOLD = 5
MAX_HOLD_DAYS = 60
WAIT_DAYS = 10
SIGNAL_DEDUP_GAP = 15

FIXED_STOP_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
ATR_PERIOD = 20
ATR_FLOOR_FACTOR = 0.85

WYCKOFF_MULTIPLIERS = {
    "Markup": 2.0,
    "Accumulation": 1.5,
    "Distribution": 1.0,
    "Markdown": 0.8,
}

MODES = [
    "baseline", "BD", "ATR", "FIXED5",
    "TP10_SL2", "ATR_BD", "TP10_BD", "HYBRID",
]

MODE_TAGS = {
    "baseline":  "0_基线_SL2",
    "BD":        "1_BD_回调跟踪",
    "ATR":       "2_ATR_动态止损",
    "FIXED5":    "3_固定5%止损",
    "TP10_SL2":  "4_SL2+10%止盈",
    "ATR_BD":    "5_ATR+跟踪",
    "TP10_BD":   "6_SL2+止盈+跟踪",
    "HYBRID":    "7_ATR+止盈+跟踪",
}


# ---------------------------------------------------------------------------
# 信号生成（复用 surge_bd_combo.py）
# ---------------------------------------------------------------------------

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


def _score_surge_raw(bis_up_to, zs_up_to, close_val, dif_now, dif_5ago,
                     ma5, ma10, ma20, ret20):
    feats = [0] * 7
    up_bis = [b for b in bis_up_to if b.direction.value == "向上"]
    dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]

    if len(up_bis) >= 3:
        p1, p2, p3 = up_bis[-3].power, up_bis[-2].power, up_bis[-1].power
        feats[0] = int(p3 > p2 > p1 and p1 > 0)

    if len(up_bis) >= 2 and len(dn_bis) >= 2:
        up_avg = np.mean([b.power for b in up_bis[-3:]])
        dn_avg = np.mean([b.power for b in dn_bis[-3:]])
        feats[1] = int(up_avg > dn_avg * 1.5) if dn_avg > 0 else 0

    if zs_up_to:
        feats[2] = int(close_val > zs_up_to[-1]["zg"] * 1.2)

    if len(dn_bis) >= 2:
        feats[3] = int(dn_bis[-1].low > dn_bis[-2].low)

    if ma5 > ma10 > ma20:
        spread = (ma5 - ma20) / ma20 * 100
        feats[4] = int(spread > 15)
    else:
        feats[4] = 0

    feats[5] = int(dif_now > 0 and dif_now > dif_5ago)
    feats[6] = int(ret20 > 15)
    return feats


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _prepare_stock(parquet_path):
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
    if len(c.bi_list) < 10:
        return None
    return df, c.bi_list, code


def _compute_indicators(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values

    return {
        "close": close,
        "high": high,
        "low": low,
        "pct_chg": df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(n),
        "dates": df["dt"].values,
        "dif": ema12 - ema26,
        "ma5": pd.Series(close).rolling(5).mean().values,
        "ma10": pd.Series(close).rolling(10).mean().values,
        "ma20": pd.Series(close).rolling(20).mean().values,
        "atr": _compute_atr(high, low, close, ATR_PERIOD),
        "wyckoff_mult": _compute_wyckoff_multipliers(close),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Codex2.0 策略: ATR 动态止损 + 简化 Wyckoff 阶段
# ---------------------------------------------------------------------------

def _compute_atr(high, low, close, period=20):
    """ATR（Average True Range），源自 codex2.0 atr_numba"""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_wyckoff_multipliers(close, lookback=50):
    """简化 Wyckoff 阶段判定 → ATR 倍数数组

    源自 codex2.0 identify_wyckoff_phase_enhanced 的核心逻辑，
    用 50 日涨幅替代完整 Wyckoff 事件信号管线。
    """
    n = len(close)
    mult = np.full(n, 1.0)
    for i in range(lookback, n):
        ret50 = (close[i] / close[i - lookback] - 1)
        if ret50 > 0.10:
            mult[i] = WYCKOFF_MULTIPLIERS["Markup"]
        elif ret50 < -0.10:
            mult[i] = WYCKOFF_MULTIPLIERS["Markdown"]
        elif abs(ret50) <= 0.05 and close[i] < np.mean(close[i - lookback:i]):
            mult[i] = WYCKOFF_MULTIPLIERS["Accumulation"]
        else:
            mult[i] = WYCKOFF_MULTIPLIERS["Distribution"]
    return mult


def _atr_stop_price(close_val, atr_val, wyckoff_mult_val):
    """单 bar 的 ATR 动态止损价，源自 codex2.0 calculate_stop_loss_enhanced"""
    if np.isnan(atr_val):
        return close_val * ATR_FLOOR_FACTOR
    stop = close_val - atr_val * wyckoff_mult_val
    floor = close_val * ATR_FLOOR_FACTOR
    return max(stop, floor)


# ---------------------------------------------------------------------------
# 信号扫描
# ---------------------------------------------------------------------------

def _find_signals(bis, ind, start_idx):
    close, dates, dif = ind["close"], ind["dates"], ind["dif"]
    ma5, ma10, ma20 = ind["ma5"], ind["ma10"], ind["ma20"]
    pct_chg, n = ind["pct_chg"], ind["n"]
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
        feats = _score_surge_raw(
            bis_up_to, zs_up_to, close[idx],
            dif[idx], dif_5ago, ma5[idx], ma10[idx], ma20[idx], ret20,
        )
        total = sum(feats)
        if total >= SCORE_THRESHOLD:
            dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]
            signals.append({
                "idx": idx, "close": close[idx], "total": total,
                "sl_bi": dn_bis[-1].low if dn_bis else None,
            })
    return signals


def _dedup(signals, gap=SIGNAL_DEDUP_GAP):
    if not signals:
        return []
    out = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - out[-1]["idx"] >= gap:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# 核心: 多策略 bar-by-bar 模拟
# ---------------------------------------------------------------------------

def _simulate_trade(mode, sig, ind, code):
    """模拟单笔交易，返回 (holds_list, pair_dict) 或 None"""
    close, low_arr, high_arr = ind["close"], ind["low"], ind["high"]
    dates, ma10 = ind["dates"], ind["ma10"]
    atr, wyckoff_mult = ind["atr"], ind["wyckoff_mult"]
    n = ind["n"]

    use_pullback = mode != "baseline"
    use_sl2 = mode in ("baseline", "BD", "TP10_SL2", "TP10_BD")
    use_atr_stop = mode in ("ATR", "ATR_BD", "HYBRID")
    use_fixed_stop = mode == "FIXED5"
    use_take_profit = mode in ("TP10_SL2", "TP10_BD", "HYBRID")
    use_trailing = mode in ("BD", "ATR_BD", "TP10_BD", "HYBRID")

    # -- 入场 --
    if use_pullback:
        sig_idx = sig["idx"]
        entry_idx = None
        for w in range(sig_idx + 1, min(sig_idx + WAIT_DAYS + 1, n)):
            if close[w] <= ma10[w] * 1.02:
                entry_idx = w
                break
        if entry_idx is None:
            return None
        entry_price = close[entry_idx]
    else:
        entry_idx = sig["idx"]
        entry_price = sig["close"]

    sl_bi = sig["sl_bi"]
    fixed_stop = entry_price * (1 - FIXED_STOP_PCT) if use_fixed_stop else None
    peak = entry_price
    holds, exit_price, exit_reason = [], None, "max_hold"

    # -- 逐 bar 退出判定 --
    for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
        if close[j] > peak:
            peak = close[j]

        holds.append({
            "dt": pd.Timestamp(dates[j]),
            "symbol": code,
            "weight": 1,
            "price": close[j],
        })

        if j == entry_idx:
            continue

        # 1) SL2 笔结构止损
        if use_sl2 and sl_bi and low_arr[j] <= sl_bi:
            exit_price = sl_bi
            exit_reason = "sl2"
            break

        # 2) ATR 动态止损（入场时刻的 ATR 止损价，固定不随后续更新）
        if use_atr_stop:
            atr_stop = _atr_stop_price(
                close[entry_idx], atr[entry_idx], wyckoff_mult[entry_idx]
            )
            if low_arr[j] <= atr_stop:
                exit_price = atr_stop
                exit_reason = "atr_stop"
                break

        # 3) 固定百分比止损
        if use_fixed_stop and low_arr[j] <= fixed_stop:
            exit_price = fixed_stop
            exit_reason = "fixed_stop_5%"
            break

        # 4) 止盈目标
        if use_take_profit:
            gain = close[j] / entry_price - 1
            if gain >= TAKE_PROFIT_PCT:
                exit_price = close[j]
                exit_reason = "take_profit_10%"
                break

        # 5) 分级跟踪止损
        if use_trailing:
            gain_pct = peak / entry_price - 1
            dd = (peak - close[j]) / peak if peak > 0 else 0
            if gain_pct >= 0.25 and dd >= 0.10:
                exit_price = close[j]
                exit_reason = "trail_10%"
                break
            if gain_pct >= 0.10 and dd >= 0.15:
                exit_price = close[j]
                exit_reason = "trail_15%"
                break

    if exit_price is None:
        last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
        exit_price = close[last_j]

    ret = (exit_price / entry_price - 1) * 100
    pair = {"ret_pct": round(ret, 2), "exit_reason": exit_reason}
    return holds, pair


def _process(parquet_path: str) -> dict | None:
    """一次加载，同时产生全部策略组合的结果"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    result = {}
    for mode in MODES:
        holds_all, pairs_all = [], []
        for sig in signals:
            trade = _simulate_trade(mode, sig, ind, code)
            if trade is None:
                continue
            h, p = trade
            holds_all.extend(h)
            pairs_all.append(p)
        if holds_all:
            result[mode] = {"holds": holds_all, "pairs": pairs_all}

    return result if result else None


# ---------------------------------------------------------------------------
# 统计 + 报告
# ---------------------------------------------------------------------------

def _run_mode(tag, all_holds, all_pairs):
    if not all_holds:
        print(f"  [{tag}] 无持仓数据")
        return None

    dfw = pd.DataFrame(all_holds)
    if dfw.duplicated(subset=["dt", "symbol"]).any():
        dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
            weight=("weight", "max"), price=("price", "first"),
        )
    dfw = dfw[["dt", "symbol", "weight", "price"]]

    try:
        wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
        stats = wb.stats
    except Exception as e:
        print(f"  [{tag}] WeightBacktest 失败: {e}")
        return None

    rets = [p["ret_pct"] for p in all_pairs]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    n_exit = {}
    for p in all_pairs:
        r = p["exit_reason"]
        n_exit[r] = n_exit.get(r, 0) + 1

    stats["tag"] = tag
    stats["退出分布"] = str(n_exit)
    stats["交易笔数"] = len(rets)
    stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
    stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
    stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
    stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
    stats["平均收益"] = f"{np.mean(rets):.2f}%"
    stats["收益中位数"] = f"{np.median(rets):.2f}%"

    try:
        out_html = OUTPUT_DIR / f"codex_{tag}.html"
        generate_backtest_report(
            df=dfw, output_path=str(out_html),
            title=f"Surge×Codex - {tag}",
            fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
        )
    except Exception:
        pass

    return stats


def main():
    t0 = time.time()
    print("=" * 100)
    print("  Surge-wave × Codex2.0 策略融合回测")
    print("=" * 100)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    file_list = [str(p) for p in parquet_files]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(parquet_files)} 只个股 | {n_workers} 进程")
    print(f"[策略] {len(MODES)} 组退出策略对比\n")

    mode_data = {m: {"holds": [], "pairs": []} for m in MODES}

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, file_list, chunksize=20), 1):
            if res is not None:
                for m in res:
                    mode_data[m]["holds"].extend(res[m]["holds"])
                    mode_data[m]["pairs"].extend(res[m]["pairs"])
            if i % 1000 == 0 or i == len(file_list):
                print(f"  [{i}/{len(file_list)}] | {time.time()-t0:.0f}s")

    print(f"\n[扫描完成] {time.time()-t0:.0f}s\n")

    all_stats = []
    for m in MODES:
        tag = MODE_TAGS[m]
        stats = _run_mode(tag, mode_data[m]["holds"], mode_data[m]["pairs"])
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("[ERROR] 无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n" + "=" * 130)
    print("  Surge-wave × Codex2.0 — 8 组退出策略对比")
    print("=" * 130)
    display_cols = [c for c in [
        "交易笔数", "胜率", "盈亏比",
        "平均盈利", "平均亏损", "平均收益", "收益中位数",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    for s in all_stats:
        print(f"\n  [{s['tag']}] 退出分布: {s['退出分布']}")

    best = max(all_stats, key=lambda s: s.get("卡玛比率", 0))
    print(f"\n{'='*80}")
    print(f"  卡玛最优: {best['tag']}")
    print(f"    年化 {best.get('年化收益', 'N/A')} | "
          f"夏普 {best.get('夏普比率', 'N/A')} | "
          f"卡玛 {best.get('卡玛比率', 'N/A')} | "
          f"回撤 {best.get('最大回撤', 'N/A')}")
    print(f"{'='*80}")

    with open(OUTPUT_DIR / "codex_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")
    print(f"[输出] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
