"""主升浪策略特征权重优化 + 加权 vs 等权全量回测

一体化流程：
  Phase 1 — 采集样本（S1-S7 特征 + SL2 止损收益）
  Phase 2 — 单因子分析 + 逻辑回归 → 推荐权重
  Phase 3 — 等权 vs 多组权重全量 WeightBacktest 对比（SL2 退出）

数据源：~/.ts_data_cache/a_stock_daily_qfq/
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from wbt import generate_backtest_report

from czsc import CZSC, Freq, WeightBacktest, format_standard_kline

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_wave"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
MAX_HOLD_DAYS = 60

FEATURE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
FEATURE_LABELS = [
    "S1_笔力加速", "S2_力度比", "S3_脱离中枢", "S4_低点抬升",
    "S5_MA扩散", "S6_DIF加速", "S7_涨幅确认",
]


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
    """返回 S1-S7 各项 0/1 列表（共 7 个元素）"""
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


# ── Phase 1: 采集样本 ──

def _collect_samples(parquet_path: str) -> list[dict] | None:
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
    low_arr = df["low"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    start_idx = max(120, n // 4)
    samples = []
    last_sample_idx = -100

    for idx in range(start_idx, n - MAX_HOLD_DAYS):
        if idx - last_sample_idx < 10:
            continue
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
            dif[idx], dif_5ago,
            ma5[idx], ma10[idx], ma20[idx], ret20,
        )
        total = sum(feats)
        if total < 2:
            continue

        # SL2 止损收益
        dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]
        sl_bi = dn_bis_up[-1].low if dn_bis_up else None
        sl2_ret = None
        if sl_bi is not None:
            for j in range(idx + 1, min(idx + MAX_HOLD_DAYS, n)):
                if low_arr[j] <= sl_bi:
                    sl2_ret = (sl_bi / close[idx] - 1) * 100
                    break
            if sl2_ret is None:
                last_j = min(idx + MAX_HOLD_DAYS - 1, n - 1)
                sl2_ret = (close[last_j] / close[idx] - 1) * 100
        else:
            last_j = min(idx + MAX_HOLD_DAYS - 1, n - 1)
            sl2_ret = (close[last_j] / close[idx] - 1) * 100

        samples.append({
            "symbol": code, "dt": str(dt)[:10], "close": close[idx],
            **{fn: feats[i] for i, fn in enumerate(FEATURE_NAMES)},
            "total": total, "sl2_ret": round(sl2_ret, 2),
        })
        last_sample_idx = idx

    return samples if samples else None


# ── Phase 2: 权重分析 ──

def analyze_single_factors(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  方法 1：单因子分析（SL2 止损收益）")
    print("=" * 80)

    ret_col = "sl2_ret"
    results = []
    for feat, label in zip(FEATURE_NAMES, FEATURE_LABELS):
        mask1 = df[feat] == 1
        mask0 = df[feat] == 0
        n1, n0 = mask1.sum(), mask0.sum()
        mean1 = df.loc[mask1, ret_col].mean() if n1 > 0 else 0
        mean0 = df.loc[mask0, ret_col].mean() if n0 > 0 else 0
        wr1 = (df.loc[mask1, ret_col] > 0).mean() * 100 if n1 > 0 else 0
        wr0 = (df.loc[mask0, ret_col] > 0).mean() * 100 if n0 > 0 else 0

        results.append({
            "特征": label, "=1样本": n1, "=0样本": n0,
            "=1频率%": round(n1 / len(df) * 100, 1),
            "=1均收益%": round(mean1, 2), "=0均收益%": round(mean0, 2),
            "收益差%": round(mean1 - mean0, 2),
            "=1胜率%": round(wr1, 1), "=0胜率%": round(wr0, 1),
            "胜率差%": round(wr1 - wr0, 1),
        })

    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))

    deltas = rdf["收益差%"].values
    pos_deltas = np.maximum(deltas, 0)
    if pos_deltas.sum() > 0:
        raw_weights = pos_deltas / pos_deltas.sum() * 7
    else:
        raw_weights = np.ones(7)
    rounded_weights = [round(w * 2) / 2 for w in raw_weights]

    print(f"\n  单因子推荐权重：")
    for label, w, rw in zip(FEATURE_LABELS, raw_weights, rounded_weights):
        print(f"    {label}: {w:.2f} -> {rw}")

    return rounded_weights


def analyze_logistic_regression(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  方法 2：逻辑回归")
    print("=" * 80)

    X = df[FEATURE_NAMES].values
    y = (df["sl2_ret"] > 0).astype(int).values

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv_scores = cross_val_score(lr, X, y, cv=5, scoring="accuracy")
    print(f"  5 折 CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    lr.fit(X, y)
    coefs = lr.coef_[0]

    print(f"\n  回归系数：")
    for label, c in zip(FEATURE_LABELS, coefs):
        print(f"    {label}: {c:.4f}")

    pos_coefs = np.maximum(coefs, 0)
    if pos_coefs.sum() > 0:
        raw_weights = pos_coefs / pos_coefs.sum() * 7
    else:
        raw_weights = np.ones(7)
    rounded_weights = [round(w * 2) / 2 for w in raw_weights]

    print(f"\n  逻辑回归推荐权重：")
    for label, w, rw in zip(FEATURE_LABELS, raw_weights, rounded_weights):
        print(f"    {label}: {w:.2f} -> {rw}")

    return rounded_weights


# ── Phase 3: 全量 WeightBacktest ──

def _bt_one_stock(args) -> list[dict] | None:
    """单只股票：用给定权重和阈值生成 SL2 持仓"""
    parquet_path, weights, threshold = args
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
    low_arr = df["low"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
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

        feats = _score_surge_raw(
            bis_up_to, zs_up_to, close[idx],
            dif[idx], dif_5ago,
            ma5[idx], ma10[idx], ma20[idx], ret20,
        )
        weighted_score = sum(w * f for w, f in zip(weights, feats))

        if weighted_score >= threshold:
            dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]
            sl_bi = dn_bis_up[-1].low if dn_bis_up else None
            signals.append({"idx": idx, "close": close[idx], "sl_bi": sl_bi})

    if not signals:
        return None

    filtered = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - filtered[-1]["idx"] >= 15:
            filtered.append(s)
    signals = filtered

    holds = []
    pairs = []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})
            if sl_price and low_arr[j] <= sl_price and j > entry_idx:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def run_full_backtest(file_list, weights, threshold, tag, n_workers):
    """用给定权重和阈值跑全量 WeightBacktest"""
    args_list = [(f, weights, threshold) for f in file_list]

    all_holds, all_pairs = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_bt_one_stock, args_list, chunksize=20):
            if res is not None:
                all_holds.extend(res["holds"])
                all_pairs.extend(res["pairs"])

    if not all_holds:
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
        print(f"  WeightBacktest 失败: {e}")
        return None

    rets = [p["ret_pct"] for p in all_pairs]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    stats["tag"] = tag
    stats["weights"] = str(weights)
    stats["threshold"] = threshold
    stats["交易笔数"] = len(rets)
    stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
    stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
    stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
    stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
    stats["平均收益"] = f"{np.mean(rets):.2f}%"

    try:
        out_html = OUTPUT_DIR / f"weighted_{tag}.html"
        generate_backtest_report(
            df=dfw, output_path=str(out_html),
            title=f"主升浪加权 - {tag}",
            fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
        )
    except Exception:
        pass

    return stats


def main():
    t_start = time.time()
    print("=" * 100)
    print("  主升浪策略 — 特征权重优化 + 加权回测")
    print("=" * 100)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    file_list = [str(p) for p in parquet_files]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(parquet_files)} 只个股 | {n_workers} 进程\n")

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: 样本采集
    # ═══════════════════════════════════════════════════════════════════
    print("▶ Phase 1: 采集 S1-S7 特征样本 + SL2 收益 ...")
    t0 = time.time()
    all_samples = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_collect_samples, file_list, chunksize=20), 1):
            if res is not None:
                all_samples.extend(res)
            if i % 1000 == 0 or i == len(file_list):
                print(f"  [{i}/{len(file_list)}] 样本 {len(all_samples)} | {time.time()-t0:.0f}s")

    print(f"  采集完成: {len(all_samples)} 样本 | {time.time()-t0:.0f}s")

    if len(all_samples) < 200:
        print("[ERROR] 样本量不足")
        return

    df_samples = pd.DataFrame(all_samples)
    print(f"  正收益 {(df_samples['sl2_ret']>0).sum()} | "
          f"负收益 {(df_samples['sl2_ret']<=0).sum()}")
    print(f"  总分分布:\n{df_samples['total'].value_counts().sort_index().to_string()}")

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: 权重分析
    # ═══════════════════════════════════════════════════════════════════
    w1 = analyze_single_factors(df_samples)
    w2 = analyze_logistic_regression(df_samples)

    final_weights = [round((a + b) / 2 * 2) / 2 for a, b in zip(w1, w2)]
    print("\n" + "=" * 80)
    print("  综合推荐权重（两种方法平均）")
    print("=" * 80)
    for label, w in zip(FEATURE_LABELS, final_weights):
        print(f"    {label}: {w}")

    # 快速样本级回测对比
    print("\n" + "=" * 80)
    print("  样本级快速对比（SL2 收益）")
    print("=" * 80)

    configs_quick = [
        ("等权基准", [1.0] * 7),
        ("单因子权重", w1),
        ("逻辑回归权重", w2),
        ("综合权重", final_weights),
    ]
    for tag, w in configs_quick:
        ws = sum(ww * df_samples[fn] for ww, fn in zip(w, FEATURE_NAMES))
        thr = sum(w) * 5 / 7
        mask = ws >= thr
        qualified = df_samples.loc[mask, "sl2_ret"]
        n_q = len(qualified)
        if n_q == 0:
            print(f"  {tag}: 无合格样本")
            continue
        wr = (qualified > 0).mean() * 100
        avg = qualified.mean()
        print(f"  {tag}: thr={thr:.1f} | 合格 {n_q} | "
              f"胜率 {wr:.1f}% | 均收益 {avg:.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 3: 全量 WeightBacktest — 多阈值搜索
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  ▶ Phase 3: 全量 WeightBacktest（SL2 退出）")
    print("=" * 100)

    equal_weights = [1.0] * 7

    configs_full = [
        ("等权_thr5", equal_weights, 5.0),
        ("综合加权_thr_auto", final_weights, round(sum(final_weights) * 5 / 7, 1)),
    ]

    # 额外测试加权的多个阈值
    w_sum = sum(final_weights)
    for thr in [4.0, 4.5, 5.0, 5.5, 6.0]:
        if thr != round(w_sum * 5 / 7, 1):
            configs_full.append((f"综合加权_thr{thr}", final_weights, thr))

    all_bt_stats = []
    for tag, w, thr in configs_full:
        print(f"\n  [{tag}] weights={w}, threshold={thr}")
        t0 = time.time()
        stats = run_full_backtest(file_list, w, thr, tag, n_workers)
        if stats is None:
            print(f"    -> 无结果")
            continue

        for k in ["交易笔数", "胜率", "盈亏比", "平均盈利", "平均亏损",
                   "平均收益", "年化收益", "夏普比率", "最大回撤", "卡玛比率"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")
        print(f"    耗时: {time.time()-t0:.0f}s")
        all_bt_stats.append(stats)

    if not all_bt_stats:
        print("\n[ERROR] 所有回测均无结果")
        return

    # ═══════════════════════════════════════════════════════════════════
    #  汇总对比
    # ═══════════════════════════════════════════════════════════════════
    cmp = pd.DataFrame(all_bt_stats).set_index("tag")
    print("\n\n" + "=" * 120)
    print("  主升浪策略 — 等权 vs 加权 回测对比汇总")
    print("=" * 120)
    display_cols = [c for c in [
        "threshold", "交易笔数", "胜率", "盈亏比",
        "平均盈利", "平均亏损", "平均收益",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "weight_optimization.json", "w", encoding="utf-8") as f:
        json.dump(all_bt_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  推荐权重: FEATURE_WEIGHTS = {final_weights}  # S1-S7")
    best = max(all_bt_stats, key=lambda s: s.get("卡玛比率", 0))
    print(f"  最佳配置: {best['tag']} (卡玛 {best.get('卡玛比率', 'N/A')})")
    print(f"  最佳阈值: SCORE_THRESHOLD = {best['threshold']}")

    print(f"\n[完成] 总耗时 {time.time()-t_start:.0f}s")
    print(f"[文件] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
