"""特征权重优化器

采集全 A 股历史信号的 C1-C7 特征值 + 未来 20 日收益率，
通过单因子分析和逻辑回归两种方法得出最优权重，
并与等权基准进行回测对比。

数据源：~/.ts_data_cache/a_stock_daily_qfq/
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from czsc import CZSC, Freq, WeightBacktest, format_standard_kline

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FWD_DAYS = 20
SAMPLE_THRESHOLD = 2
FEE_RATE = 0.0002
MAX_HOLD_DAYS = 40

FEATURE_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def _extract_zs(bis_list):
    """从笔列表提取非重叠中枢"""
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


def _score_features(bis, zs_list, dif_val, vol_ratio_prev, vol_ratio_now):
    """返回 C1-C7 各项 0/1 列表"""
    feats = [0] * 7

    # C1: 中枢上移
    if len(zs_list) >= 3:
        feats[0] = int(zs_list[-1]["zd"] > zs_list[-2]["zd"] > zs_list[-3]["zd"])

    # C2: 窄幅收敛
    if zs_list:
        width = (zs_list[-1]["zg"] - zs_list[-1]["zd"]) / zs_list[-1]["zd"]
        feats[1] = int(width < 0.10)

    # C3: 向上笔力度突增
    up_bis = [b for b in bis if b.direction.value == "向上"]
    if len(up_bis) >= 3:
        recent = up_bis[-1].power
        prev_avg = np.mean([b.power for b in up_bis[-3:-1]])
        feats[2] = int(recent > prev_avg * 1.5) if prev_avg > 0 else 0

    # C4: 向下笔力度递减
    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if len(dn_bis) >= 2:
        last_dn = dn_bis[-1]
        prev_dn = dn_bis[-2]
        last_pct = (last_dn.high / last_dn.low - 1)
        feats[3] = int(last_dn.power < prev_dn.power and last_pct < 0.12)

    # C5: DIF 零轴附近
    if dif_val is not None:
        feats[4] = int(abs(dif_val) < 0.5)

    # C6: 缩量后放量
    feats[5] = int(vol_ratio_prev < 0.8 and vol_ratio_now > 1.2)

    # C7: 不回前中枢
    if len(zs_list) >= 2 and len(dn_bis) >= 1:
        feats[6] = int(dn_bis[-1].low > zs_list[-2]["zg"])

    return feats


def _collect_samples(parquet_path: str) -> list[dict] | None:
    """单只股票：逐日采集特征 + 未来收益"""
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
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values

    start_idx = max(120, n // 4)
    samples = []
    last_sample_idx = -100

    for idx in range(start_idx, n - FWD_DAYS):
        if idx - last_sample_idx < 10:
            continue

        if abs(pct_chg[idx]) >= 9.8:
            continue

        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dates[idx])]
        if len(bis_up_to) < 6:
            continue

        zs_up_to = _extract_zs(bis_up_to)
        if len(zs_up_to) < 3:
            continue

        vr_now = vol_ma5[idx] / vol_ma20[idx] if vol_ma20[idx] > 0 else 1.0
        vr_prev = vol_ma5[idx - 5] / vol_ma20[idx - 5] if idx >= 5 and vol_ma20[idx - 5] > 0 else 1.0

        feats = _score_features(bis_up_to, zs_up_to, dif[idx], vr_prev, vr_now)
        total = sum(feats)

        if total < SAMPLE_THRESHOLD:
            continue

        fwd_ret = (close[idx + FWD_DAYS] / close[idx] - 1) * 100

        # 笔结构止损收益（SL2）
        dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]
        sl_bi = dn_bis_up[-1].low if dn_bis_up else None
        sl2_ret = fwd_ret
        if sl_bi is not None:
            low_arr = df["low"].values
            for j in range(idx + 1, min(idx + MAX_HOLD_DAYS, n)):
                if low_arr[j] <= sl_bi:
                    sl2_ret = (sl_bi / close[idx] - 1) * 100
                    break
            else:
                last_j = min(idx + MAX_HOLD_DAYS - 1, n - 1)
                sl2_ret = (close[last_j] / close[idx] - 1) * 100

        samples.append({
            "symbol": code,
            "dt": str(dates[idx])[:10],
            "close": close[idx],
            "C1": feats[0], "C2": feats[1], "C3": feats[2], "C4": feats[3],
            "C5": feats[4], "C6": feats[5], "C7": feats[6],
            "total": total,
            "fwd_ret20": round(fwd_ret, 2),
            "sl2_ret": round(sl2_ret, 2),
        })
        last_sample_idx = idx

    return samples if samples else None


def analyze_single_factors(df: pd.DataFrame):
    """单因子分析：每个 Ci 的条件期望差和胜率差"""
    print("\n" + "=" * 80)
    print("  方法 1：单因子分析")
    print("=" * 80)

    ret_col = "sl2_ret"
    results = []

    for feat in FEATURE_NAMES:
        mask1 = df[feat] == 1
        mask0 = df[feat] == 0

        n1, n0 = mask1.sum(), mask0.sum()
        mean1 = df.loc[mask1, ret_col].mean() if n1 > 0 else 0
        mean0 = df.loc[mask0, ret_col].mean() if n0 > 0 else 0
        wr1 = (df.loc[mask1, ret_col] > 0).mean() * 100 if n1 > 0 else 0
        wr0 = (df.loc[mask0, ret_col] > 0).mean() * 100 if n0 > 0 else 0

        delta_ret = mean1 - mean0
        delta_wr = wr1 - wr0

        results.append({
            "特征": feat,
            "=1样本": n1,
            "=0样本": n0,
            "=1频率%": round(n1 / len(df) * 100, 1),
            "=1均收益%": round(mean1, 2),
            "=0均收益%": round(mean0, 2),
            "收益差%": round(delta_ret, 2),
            "=1胜率%": round(wr1, 1),
            "=0胜率%": round(wr0, 1),
            "胜率差%": round(delta_wr, 1),
        })

    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))

    # 归一化收益差作为权重
    deltas = rdf["收益差%"].values
    pos_deltas = np.maximum(deltas, 0)
    if pos_deltas.sum() > 0:
        raw_weights = pos_deltas / pos_deltas.sum() * 7
    else:
        raw_weights = np.ones(7)
    rounded_weights = [round(w * 2) / 2 for w in raw_weights]

    print(f"\n  单因子推荐权重（归一化收益差）：")
    for feat, w, rw in zip(FEATURE_NAMES, raw_weights, rounded_weights):
        print(f"    {feat}: {w:.2f} -> {rw}")

    return rounded_weights


def analyze_logistic_regression(df: pd.DataFrame):
    """逻辑回归拟合权重 + 5 折交叉验证"""
    print("\n" + "=" * 80)
    print("  方法 2：逻辑回归")
    print("=" * 80)

    X = df[FEATURE_NAMES].values
    y = (df["sl2_ret"] > 0).astype(int).values

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv_scores = cross_val_score(lr, X, y, cv=5, scoring="accuracy")
    print(f"  5 折 CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  各折: {[round(s, 4) for s in cv_scores]}")

    lr.fit(X, y)
    coefs = lr.coef_[0]

    print(f"\n  回归系数（原始）：")
    for feat, c in zip(FEATURE_NAMES, coefs):
        print(f"    {feat}: {c:.4f}")

    # 归一化到总和 7
    pos_coefs = np.maximum(coefs, 0)
    if pos_coefs.sum() > 0:
        raw_weights = pos_coefs / pos_coefs.sum() * 7
    else:
        raw_weights = np.ones(7)
    rounded_weights = [round(w * 2) / 2 for w in raw_weights]

    print(f"\n  逻辑回归推荐权重：")
    for feat, w, rw in zip(FEATURE_NAMES, raw_weights, rounded_weights):
        print(f"    {feat}: {w:.2f} -> {rw}")

    return rounded_weights


def backtest_with_weights(df_samples: pd.DataFrame, weights: list[float], tag: str):
    """用给定权重对样本重新打分并回测"""
    df = df_samples.copy()
    df["weighted_score"] = sum(w * df[feat] for w, feat in zip(weights, FEATURE_NAMES))

    threshold = sum(weights) * 5 / 7
    qualified = df[df["weighted_score"] >= threshold].copy()

    if len(qualified) == 0:
        return {"tag": tag, "threshold": round(threshold, 2), "交易笔数": 0}

    rets = qualified["sl2_ret"].values
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    stats = {
        "tag": tag,
        "threshold": round(threshold, 2),
        "交易笔数": len(rets),
        "胜率%": round(len(wins) / len(rets) * 100, 1),
        "平均收益%": round(np.mean(rets), 2),
        "中位收益%": round(np.median(rets), 2),
        "平均盈利%": round(np.mean(wins), 2) if len(wins) > 0 else 0,
        "平均亏损%": round(np.mean(losses), 2) if len(losses) > 0 else 0,
        "盈亏比": round(abs(np.mean(wins) / np.mean(losses)), 2) if len(losses) > 0 and len(wins) > 0 else 0,
    }
    return stats


def main():
    print("=" * 80)
    print("  特征权重优化器 — C1-C7 权重分析")
    print("=" * 80)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] {len(parquet_files)} 只个股")

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] {n_workers} 进程")

    t0 = time.time()
    file_list = [str(p) for p in parquet_files]

    all_samples = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_collect_samples, file_list, chunksize=20), 1):
            if res is not None:
                all_samples.extend(res)
            if i % 500 == 0 or i == len(file_list):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(file_list)}] 样本 {len(all_samples)} | "
                      f"{elapsed:.0f}s | ETA {eta:.0f}s")

    print(f"\n[采集完成] 耗时 {time.time()-t0:.0f}s | 总样本 {len(all_samples)}")

    if len(all_samples) < 100:
        print("[ERROR] 样本量不足")
        return

    df = pd.DataFrame(all_samples)
    print(f"  样本分布: 正收益 {(df['sl2_ret']>0).sum()} | 负收益 {(df['sl2_ret']<=0).sum()}")
    print(f"  总分分布:\n{df['total'].value_counts().sort_index().to_string()}")

    # ── 方法 1：单因子分析 ──
    w1 = analyze_single_factors(df)

    # ── 方法 2：逻辑回归 ──
    w2 = analyze_logistic_regression(df)

    # ── 综合权重 ──
    final_weights = [round((a + b) / 2 * 2) / 2 for a, b in zip(w1, w2)]
    print("\n" + "=" * 80)
    print("  综合推荐权重（两种方法平均，四舍五入到 0.5）")
    print("=" * 80)
    for feat, w in zip(FEATURE_NAMES, final_weights):
        print(f"    {feat}: {w}")

    # ── 回测对比 ──
    equal_weights = [1.0] * 7
    configs = [
        ("等权基准", equal_weights),
        ("单因子权重", w1),
        ("逻辑回归权重", w2),
        ("综合权重", final_weights),
    ]

    print("\n" + "=" * 80)
    print("  回测对比（使用 SL2 笔结构止损收益）")
    print("=" * 80)

    bt_results = []
    for tag, w in configs:
        stats = backtest_with_weights(df, w, tag)
        bt_results.append(stats)

    bt_df = pd.DataFrame(bt_results).set_index("tag")
    print(bt_df.to_string())

    # ── 打印可直接复制的权重常量 ──
    print("\n" + "=" * 80)
    print("  可复制到 daily_scan.py 的权重配置")
    print("=" * 80)
    print(f"FEATURE_WEIGHTS = {final_weights}  # C1-C7")
    threshold = round(sum(final_weights) * 5 / 7, 1)
    print(f"SCORE_THRESHOLD = {threshold}  # 等效原来的 5/7 比例")

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
