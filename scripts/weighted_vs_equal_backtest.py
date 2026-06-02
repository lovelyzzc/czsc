"""加权打分 vs 等权打分 回测对比

对比两种入场打分方式 + SL2 笔结构止损的实际表现：
  A 等权基准    — C1-C7 各 1 分，总分 >= 5 入场
  B 加权优化    — C1=1.0 C2-C5,C7=0.5 C6=3.0，加权分 >= 4 入场

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

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "weight_cmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
MAX_HOLD_DAYS = 40

EQUAL_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
EQUAL_THRESHOLD = 5.0

OPTIMIZED_WEIGHTS = [1.0, 0.5, 0.5, 0.5, 0.5, 3.0, 0.5]
OPT_THRESHOLDS = [4.0, 4.5, 5.0, 5.5]


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


def _calc_features(bis, zs_list, dif_val, vr_prev, vr_now):
    """返回 C1-C7 的 0/1 列表"""
    feats = [0] * 7

    if len(zs_list) >= 3:
        feats[0] = int(zs_list[-1]["zd"] > zs_list[-2]["zd"] > zs_list[-3]["zd"])

    if zs_list:
        width = (zs_list[-1]["zg"] - zs_list[-1]["zd"]) / zs_list[-1]["zd"]
        feats[1] = int(width < 0.10)

    up_bis = [b for b in bis if b.direction.value == "向上"]
    if len(up_bis) >= 3:
        recent = up_bis[-1].power
        prev_avg = np.mean([b.power for b in up_bis[-3:-1]])
        feats[2] = int(recent > prev_avg * 1.5) if prev_avg > 0 else 0

    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if len(dn_bis) >= 2:
        last_dn = dn_bis[-1]
        prev_dn = dn_bis[-2]
        last_pct = (last_dn.high / last_dn.low - 1)
        feats[3] = int(last_dn.power < prev_dn.power and last_pct < 0.12)

    if dif_val is not None:
        feats[4] = int(abs(dif_val) < 0.5)

    feats[5] = int(vr_prev < 0.8 and vr_now > 1.2)

    if len(zs_list) >= 2 and len(dn_bis) >= 1:
        feats[6] = int(dn_bis[-1].low > zs_list[-2]["zg"])

    return feats


def _process_stock(parquet_path: str) -> dict | None:
    """单只股票：生成两种打分方式的信号 + SL2 止损 holds/pairs"""
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
    low_arr = df["low"].values
    vol = df["vol"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
    symbol = code
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values

    start_idx = max(120, n // 4)

    configs = [("equal", EQUAL_WEIGHTS, EQUAL_THRESHOLD)]
    for t in OPT_THRESHOLDS:
        configs.append((f"opt_{t}", OPTIMIZED_WEIGHTS, t))

    result = {"symbol": symbol}

    for tag, weights, threshold in configs:
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

            feats = _calc_features(bis_up_to, zs_up_to, dif[idx], vr_prev, vr_now)
            score = sum(w * f for w, f in zip(weights, feats))

            if score >= threshold and abs(pct_chg[idx]) < 9.8:
                dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]
                sl_bi = dn_bis_up[-1].low if dn_bis_up else None

                signals.append({
                    "dt": pd.Timestamp(dt),
                    "idx": idx,
                    "close": close[idx],
                    "sl_bi": sl_bi,
                })

        if not signals:
            result[f"n_{tag}"] = 0
            result[f"holds_{tag}"] = []
            result[f"pairs_{tag}"] = []
            continue

        filtered = [signals[0]]
        for s in signals[1:]:
            if s["idx"] - filtered[-1]["idx"] >= 10:
                filtered.append(s)
        signals = filtered

        result[f"n_{tag}"] = len(signals)

        holds = []
        pairs = []
        for sig in signals:
            entry_idx = sig["idx"]
            entry_price = sig["close"]
            sl_price = sig["sl_bi"]

            exit_price = None
            exit_reason = "max_hold"

            for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
                holds.append({
                    "dt": pd.Timestamp(dates[j]),
                    "symbol": symbol,
                    "pos": 1,
                    "price": close[j],
                })
                if sl_price is not None and low_arr[j] <= sl_price and j > entry_idx:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                    break

            if exit_price is None:
                last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
                exit_price = close[last_j]

            ret = (exit_price / entry_price - 1) * 100
            pairs.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": round(exit_price, 2),
                "ret_pct": round(ret, 2),
                "exit_reason": exit_reason,
            })

        result[f"holds_{tag}"] = holds
        result[f"pairs_{tag}"] = pairs

    return result


def main():
    print("=" * 80)
    print("  加权打分 vs 等权打分 — 全 A 股回测对比（SL2 笔结构止损）")
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
                print(f"  [{i}/{len(file_list)}] 完成 {len(all_results)} | "
                      f"{elapsed:.0f}s | ETA {eta:.0f}s")

    print(f"\n[扫描完成] 耗时 {time.time()-t0:.0f}s")

    tags = {"equal": "A_等权(>=5)"}
    for t in OPT_THRESHOLDS:
        tags[f"opt_{t}"] = f"B_加权(>={t})"

    all_stats = []
    for key, label in tags.items():
        print(f"\n{'='*70}")
        print(f"  [{label}]")
        print(f"{'='*70}")

        all_holds = []
        all_pairs = []
        n_stocks = 0
        n_signals = 0
        for r in all_results:
            h = r.get(f"holds_{key}", [])
            p = r.get(f"pairs_{key}", [])
            ns = r.get(f"n_{key}", 0)
            if h:
                all_holds.extend(h)
                all_pairs.extend(p)
                n_stocks += 1
                n_signals += ns

        print(f"  股票数: {n_stocks} | 信号数: {n_signals}")

        if not all_holds:
            print("  无持仓数据")
            continue

        n_sl = sum(1 for p in all_pairs if p["exit_reason"] == "stop_loss")

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

        stats["tag"] = label
        stats["股票数"] = n_stocks
        stats["信号数"] = n_signals

        rets = [p["ret_pct"] for p in all_pairs]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]

        stats["交易笔数"] = len(rets)
        stats["止损触发"] = n_sl
        stats["止损率"] = f"{n_sl/len(rets)*100:.1f}%" if rets else "0%"
        stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
        stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
        stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
        stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
        stats["平均收益"] = f"{np.mean(rets):.2f}%"
        stats["收益中位数"] = f"{np.median(rets):.2f}%"

        for k in ["股票数", "信号数", "交易笔数", "止损触发", "止损率",
                   "胜率", "盈亏比", "平均盈利", "平均亏损", "平均收益", "收益中位数",
                   "年化收益", "夏普比率", "最大回撤", "卡玛比率"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            safe_key = key
            out_html = OUTPUT_DIR / f"{safe_key}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"权重回测 - {label}",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"    HTML: {out_html.name}")
        except Exception as e:
            print(f"    HTML 报告失败: {e}")

        all_stats.append(stats)

    if len(all_stats) < 2:
        print("\n[ERROR] 对比数据不足")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 100)
    print("  加权打分 vs 等权打分 — 对比总览")
    print("=" * 100)
    display_cols = [c for c in [
        "股票数", "信号数", "交易笔数", "止损触发", "止损率",
        "胜率", "盈亏比", "平均盈利", "平均亏损", "平均收益",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR}")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
