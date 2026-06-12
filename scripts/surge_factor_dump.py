"""因子库扩展 dump：每个「结构就绪」bar 的因果特征 + 前向主升标签

对全 A 股做单次全量流式重放，对每个 regime ∈ {4,5,6,7,8} 的 bar 输出：

- 结构/量价因子（全部 ≤t 因果）：tr.compute_features 全集 + ret5/ret60/距250日高/
  10日振幅/20日量额比/dif标准化/突破幅度(close/zg−1)/止损距离；
- 标签：
  - ``t1_px30``（**主标签，价格口径**）未来 40 根内最大收盘涨幅 ≥30%（前向不足 20 根记 NaN）。
    选价格口径的原因（实证，见 000636 案例）：FSM 事件口径在涨停密集/高波动强趋势中
    被「单日跌回中枢上沿→破坏→次日再突破」反复打断，{7,8} 驻留极短，
    `find_surges` 对该股整段历史找到 0 个事件，而价格口径 40 根 +63.8% 真实存在——
    FSM 标签系统性漏掉最强势的一类主升；
  - ``fwd40_max``   未来 40 根最大收盘涨幅 %（连续强度）；
  - ``t1_surge40``（对照标签，FSM 口径）未来 40 根内启动 find_surges 事件；
  - ``fwd20``       前向 20 根原始收益（市场调整在研究脚本做：减等权指数 fwd20）；
  - ``in_surge``    当前 bar 已处于某 FSM 主升事件区间内；
- ``is_anticipate_onset``：该 bar 是否为带门控的 anticipate 启动（现行入场定义标记，
  供 P2 广义筛选 vs 现行定义对比）。

输出 scripts/_output/surge_factors/bars.parquet（float32 压缩）。
下游：surge_factor_research.py。

    uv run --no-sync python scripts/surge_factor_dump.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_characteristics as sc
import trend_regime as tr

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_factors"
EMIT_REGIMES = {4, 5, 6, 7, 8}
T1_HORIZON = 40  # 未来 N 根内启动主升 → 标签 1
FWD = 20

FEAT_KEYS = sc.FEATURE_KEYS  # up_dn_power_ratio/ma_spread_pct/last_up_angle/vol_ratio/ret20/n_pivots/pivot_width_pct/above_zg


def _process(parquet_path: str) -> pd.DataFrame | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df, with_features=True)
    if len(states) < 60:
        return None
    ind = tr.compute_indicators(df)
    close, amount = ind["close"], df["amount"].to_numpy(dtype=float) if "amount" in df.columns else None
    high, low = ind["high"], ind["low"]
    n = ind["n"]

    surges, _ = sc.find_surges(states, ind)
    starts = sorted(s["start_k"] for s in surges)
    spans = [(s["start_k"], s["end_k"]) for s in surges]

    regimes = [s.regime for s in states]
    rows = []
    for k, snap in enumerate(states):
        if snap.regime not in EMIT_REGIMES or not snap.feats:
            continue
        idx = snap.idx
        # ---- 标签 ----
        t1 = int(any(k < s_k <= k + T1_HORIZON for s_k in starts))
        in_surge = int(any(a <= k <= b for a, b in spans))
        fwd20 = (close[idx + FWD] / close[idx] - 1) * 100 if idx + FWD < n else np.nan
        if idx + FWD < n:  # 前向至少 20 根才打价格标签（防右删失偏置）
            fwd_win = close[idx + 1 : min(idx + 1 + T1_HORIZON, n)]
            fwd40_max = (fwd_win.max() / close[idx] - 1) * 100
            t1_px = int(fwd40_max >= 30.0)
        else:
            fwd40_max, t1_px = np.nan, np.nan
        # ---- 现行入场定义标记 ----
        prior = regimes[max(0, k - tr.SURGE_PRIOR_WINDOW) : k]
        onset = k >= 1 and tr.surge_onset(regimes[k - 1], snap.regime, snap.feats, prior, "anticipate")
        # ---- 扩展因子 ----
        c = close[idx]
        ret5 = (c / close[idx - 5] - 1) * 100 if idx >= 5 else np.nan
        ret60 = (c / close[idx - 60] - 1) * 100 if idx >= 60 else np.nan
        lo250 = max(0, idx - 250)
        dist_hi250 = (c / close[lo250 : idx + 1].max() - 1) * 100
        amp10 = float(np.mean((high[idx - 9 : idx + 1] - low[idx - 9 : idx + 1]) / close[idx - 9 : idx + 1])) * 100 if idx >= 9 else np.nan
        amt_ratio20 = (
            float(amount[idx] / amount[idx - 20 : idx].mean())
            if amount is not None and idx >= 20 and amount[idx - 20 : idx].mean() > 0
            else np.nan
        )
        dif_norm = float(ind["dif"][idx] / c) * 100 if c > 0 else np.nan
        breakout_pct = (c / snap.zg - 1) * 100 if snap.zg == snap.zg and snap.zg > 0 else np.nan
        sl_dist = (c - snap.sl_ref) / c * 100 if snap.sl_ref == snap.sl_ref and snap.sl_ref > 0 else np.nan

        rows.append(
            {
                "symbol": df["symbol"].iloc[0],
                "dt": snap.dt,
                "regime": int(snap.regime),
                "t1_px30": t1_px,
                "fwd40_max": fwd40_max,
                "t1_surge40": t1,
                "in_surge": in_surge,
                "fwd20": fwd20,
                "is_anticipate_onset": int(onset),
                **{key: snap.feats.get(key) for key in FEAT_KEYS},
                "ret5": ret5,
                "ret60": ret60,
                "dist_hi250": dist_hi250,
                "amp10": amp10,
                "amt_ratio20": amt_ratio20,
                "dif_norm": dif_norm,
                "breakout_pct": breakout_pct,
                "sl_dist": sl_dist,
            }
        )
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只 | {n_workers} 进程 | 输出 regime∈{sorted(EMIT_REGIMES)} 的 bar")

    ctx = mp.get_context("spawn")
    parts = []
    total = 0
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, files, chunksize=20), 1):
            if res is not None:
                parts.append(res)
                total += len(res)
            if i % 500 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 行 {total} | {time.time() - t0:.0f}s")

    bars = pd.concat(parts, ignore_index=True)
    for col in bars.columns:
        if bars[col].dtype == np.float64:
            bars[col] = bars[col].astype(np.float32)
    bars["year"] = bars["dt"].dt.year.astype(np.int16)
    bars.to_parquet(OUTPUT_DIR / "bars.parquet", index=False)
    px_rate = bars["t1_px30"].mean()
    fsm_rate = bars.loc[bars["in_surge"] == 0, "t1_surge40"].mean()
    print(
        f"[完成] {len(bars)} 行 → bars.parquet | T1 基率 价格口径 {px_rate * 100:.1f}% / "
        f"FSM口径(非在途) {fsm_rate * 100:.1f}% | {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
