"""B+D 组合回测：回调入场 + 混合退出

只跑 基线 / B / D / B+D 四组做精准对比，节省时间。
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
    n = len(df)
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    return {
        "close": close, "high": df["high"].values, "low": df["low"].values,
        "pct_chg": df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(n),
        "dates": df["dt"].values, "dif": ema12 - ema26,
        "ma5": pd.Series(close).rolling(5).mean().values,
        "ma10": pd.Series(close).rolling(10).mean().values,
        "ma20": pd.Series(close).rolling(20).mean().values,
        "n": n,
    }


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


def _dedup(signals, gap=15):
    if not signals:
        return []
    out = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - out[-1]["idx"] >= gap:
            out.append(s)
    return out


def _process(parquet_path: str) -> dict | None:
    """一次加载，同时产生 4 组策略的结果"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    ma10 = ind["ma10"]
    n = ind["n"]
    WAIT_DAYS = 10

    result = {}

    for mode in ["baseline", "B", "D", "BD"]:
        holds, pairs = [], []

        for sig in signals:
            # D / BD: 回调入场
            if mode in ("D", "BD"):
                sig_idx = sig["idx"]
                actual_entry_idx = None
                actual_entry_price = None
                for w in range(sig_idx + 1, min(sig_idx + WAIT_DAYS + 1, n)):
                    if close[w] <= ma10[w] * 1.02:
                        actual_entry_idx = w
                        actual_entry_price = close[w]
                        break
                if actual_entry_idx is None:
                    continue
                entry_idx = actual_entry_idx
                entry_price = actual_entry_price
            else:
                entry_idx = sig["idx"]
                entry_price = sig["close"]

            sl_price = sig["sl_bi"]
            peak = entry_price
            exit_price, exit_reason = None, "max_hold"

            for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
                if close[j] > peak:
                    peak = close[j]

                holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                              "weight": 1, "price": close[j]})

                if j == entry_idx:
                    continue

                # SL2 兜底（所有模式都有）
                if sl_price and low_arr[j] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl2" if mode in ("B", "BD") else "stop_loss"
                    break

                # B / BD: 混合跟踪止损
                if mode in ("B", "BD"):
                    gain_pct = (peak / entry_price - 1)
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
            pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

        if holds:
            result[mode] = {"holds": holds, "pairs": pairs}

    return result if result else None


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
        out_html = OUTPUT_DIR / f"opt_{tag}.html"
        generate_backtest_report(
            df=dfw, output_path=str(out_html),
            title=f"主升浪 - {tag}",
            fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
        )
    except Exception:
        pass

    return stats


def main():
    t0 = time.time()
    print("=" * 100)
    print("  主升浪策略 — B+D 组合精准对比")
    print("=" * 100)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    file_list = [str(p) for p in parquet_files]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(parquet_files)} 只个股 | {n_workers} 进程\n")

    # 一次遍历同时产生 4 组结果
    mode_data = {m: {"holds": [], "pairs": []} for m in ["baseline", "B", "D", "BD"]}

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

    tags = {
        "baseline": "0_基线",
        "B": "B_混合退出",
        "D": "D_回调入场",
        "BD": "H_组合BD",
    }

    all_stats = []
    for m, tag in tags.items():
        stats = _run_mode(tag, mode_data[m]["holds"], mode_data[m]["pairs"])
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("[ERROR] 无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n" + "=" * 120)
    print("  主升浪策略 — B+D 组合对比")
    print("=" * 120)
    display_cols = [c for c in [
        "交易笔数", "胜率", "盈亏比",
        "平均盈利", "平均亏损", "平均收益", "收益中位数",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    for s in all_stats:
        print(f"\n  [{s['tag']}] 退出分布: {s['退出分布']}")

    best = max(all_stats, key=lambda s: s.get("卡玛比率", 0))
    print(f"\n  卡玛最优: {best['tag']} "
          f"(年化 {best.get('年化收益', 'N/A')} / "
          f"夏普 {best.get('夏普比率', 'N/A')} / "
          f"卡玛 {best.get('卡玛比率', 'N/A')} / "
          f"回撤 {best.get('最大回撤', 'N/A')})")

    with open(OUTPUT_DIR / "bd_combo_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
