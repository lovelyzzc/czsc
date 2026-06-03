"""主升浪特征研究 —— 主升浪个股通常具备哪些特征、处于什么阶段？

基于 ``trend_regime`` 的因果状态序列，做**描述性**统计（监督式刻画）：

1. **主升浪事件标注**：连续处于 ``{7 主升延续, 8 加速主升}``（容忍少量 5/6/9 过渡）
   且区间涨幅 ≥ 阈值的片段记为一次主升浪。标注用到「事件已发生」这一后验信息
   （因变量），属正常的事后刻画，**不用于交易规则**。
2. **阶段路径**：每次主升浪启动前 ``PRIOR_WINDOW`` 根里走过哪些状态 —— 回答
   「主升前处于什么阶段」（典型路径 4 中枢构造 → 5 离开 → 6 三买 → 7 主升）。
3. **启动特征**：启动当根的因果结构特征分布（笔力度比/角度/MA 散度/DIF/量比/
   中枢数与宽度/ret20）—— 回答「主升浪个股通常具备哪些特征」。并与**对照组**
   （处于 5/6/7 但未引发主升的样本）对比中位数，看哪些特征具判别力。
4. **止损止盈建议**：主升浪区间内最大回撤分布（→ 跟踪止损幅度）、结束状态分布
   （背驰 9 vs 结构破坏 10）、以及「首次进入 9/10 离场 vs 持有到区间峰值」的
   收益捕获对比。

    uv run --no-sync python scripts/surge_characteristics.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trend_regime as tr
from trend_regime import REGIME_CN, Regime

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "trend_regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = tr.DATA_DIR
SURGE_CORE = frozenset({Regime.MainUptrend, Regime.Acceleration})  # {7, 8}
TRANSIENT = frozenset({Regime.UpwardDeparture, Regime.ThirdBuy, Regime.Divergence})  # 容忍的内部过渡
MIN_SURGE_GAIN = 0.30  # 区间涨幅阈值（主升浪定义）
MIN_SURGE_BARS = 10  # 区间最少 bar 数
MERGE_GAP = 3  # 容忍的非核心状态间隔
PRIOR_WINDOW = 40  # 启动前回看窗口
FEATURE_KEYS = [
    "up_dn_power_ratio",
    "ma_spread_pct",
    "last_up_angle",
    "vol_ratio",
    "ret20",
    "n_pivots",
    "pivot_width_pct",
    "above_zg",
]


def find_surges(states, ind):
    """返回该股的主升浪事件列表 + 对照组特征（处于 5/6/7 但非主升前兆的样本）。"""
    close, low = ind["close"], ind["low"]
    regimes = [s.regime for s in states]
    n = len(states)
    surges, surge_prior_ks = [], set()

    k = 0
    while k < n:
        if regimes[k] not in SURGE_CORE:
            k += 1
            continue
        start_k, end_k, gap, j = k, k, 0, k + 1
        while j < n:
            if regimes[j] in SURGE_CORE:
                end_k, gap = j, 0
            elif regimes[j] in TRANSIENT:
                gap += 1
                if gap > MERGE_GAP:
                    break
            else:
                break
            j += 1

        s_idx, e_idx = states[start_k].idx, states[end_k].idx
        onset_close = states[start_k].close
        run_peak, max_dd = onset_close, 0.0
        for t in range(s_idx, e_idx + 1):
            run_peak = max(run_peak, close[t])
            max_dd = max(max_dd, (run_peak - low[t]) / run_peak if run_peak > 0 else 0.0)
        gain = close[s_idx : e_idx + 1].max() / onset_close - 1.0
        dur = end_k - start_k + 1

        if gain >= MIN_SURGE_GAIN and dur >= MIN_SURGE_BARS and states[start_k].feats:
            lo = max(0, start_k - PRIOR_WINDOW)
            prior = {regimes[p] for p in range(lo, start_k)}
            surge_prior_ks.update(range(lo, start_k + 1))
            # 首次进入 9/10 的离场收益 vs 区间峰值收益
            first_sell_gain = np.nan
            for p in range(start_k, min(end_k + 2, n)):
                if regimes[p] in (Regime.Divergence, Regime.Breakdown):
                    first_sell_gain = states[p].close / onset_close - 1.0
                    break
            surges.append(
                {
                    "gain": round(float(gain), 3),
                    "dur": int(dur),
                    "max_dd": round(float(max_dd), 3),
                    "end_state": int(regimes[min(end_k + 1, n - 1)]),
                    "prev_state": int(states[start_k - 1].regime) if start_k > 0 else -1,
                    "had_pivot": int(Regime.PivotBuilding in prior),
                    "had_departure": int(Regime.UpwardDeparture in prior),
                    "had_thirdbuy": int(Regime.ThirdBuy in prior),
                    "first_sell_gain": round(float(first_sell_gain), 3) if not np.isnan(first_sell_gain) else None,
                    **{key: states[start_k].feats.get(key) for key in FEATURE_KEYS},
                }
            )
        k = max(end_k + 1, k + 1)

    # 对照组：处于 5/6/7 但不在任何主升浪前窗/启动点的样本特征
    control = []
    for k2, s in enumerate(states):
        if k2 in surge_prior_ks:
            continue
        if s.regime in (Regime.UpwardDeparture, Regime.ThirdBuy, Regime.MainUptrend) and s.feats:
            control.append({key: s.feats.get(key) for key in FEATURE_KEYS})
    return surges, control


def _process(parquet_path):
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df, with_features=True)
    if len(states) < 60:
        return None
    ind = tr.compute_indicators(df)
    surges, control = find_surges(states, ind)
    if not surges and not control:
        return None
    return {"symbol": df["symbol"].iloc[0], "surges": surges, "control": control}


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _quantile_table(records, keys):
    df = pd.DataFrame(records)
    rows = {}
    for key in keys:
        if key in df.columns:
            s = pd.to_numeric(df[key], errors="coerce").dropna()
            if len(s):
                rows[key] = {
                    "中位": round(s.median(), 2),
                    "P25": round(s.quantile(0.25), 2),
                    "P75": round(s.quantile(0.75), 2),
                }
    return rows


def build_report(all_surges, all_control):
    lines = ["# 主升浪特征研究报告\n"]
    n = len(all_surges)
    lines.append(
        f"- 主升浪样本数：**{n}**（定义：连续处于 7/8 且区间涨幅 ≥ {MIN_SURGE_GAIN:.0%}、≥ {MIN_SURGE_BARS} bar）"
    )
    if n == 0:
        return "\n".join(lines)

    df = pd.DataFrame(all_surges)
    lines.append(f"- 区间涨幅：中位 **{df['gain'].median():.1%}**，P75 {df['gain'].quantile(0.75):.1%}")
    lines.append(f"- 持续天数：中位 **{df['dur'].median():.0f}**，P75 {df['dur'].quantile(0.75):.0f}")

    # 阶段路径
    lines.append(f"\n## 1. 主升前处于什么阶段（启动前 {PRIOR_WINDOW} 根路径）\n")
    lines.append(f"- 启动前曾构造中枢(4)：**{df['had_pivot'].mean():.0%}**")
    lines.append(f"- 启动前曾向上离开中枢(5)：**{df['had_departure'].mean():.0%}**")
    lines.append(f"- 启动前曾出现三买(6)：**{df['had_thirdbuy'].mean():.0%}**")
    full_path = ((df["had_pivot"] == 1) & (df["had_departure"] == 1)).mean()
    lines.append(f"- 同时经过 4 中枢构造 → 5 向上离开：**{full_path:.0%}**（典型主升路径）")
    prev_dist = (
        df["prev_state"]
        .map(lambda r: REGIME_CN.get(Regime(r), str(r)) if r >= 0 else "无")
        .value_counts(normalize=True)
    )
    lines.append("- 启动当根的紧邻前一状态分布：")
    for name, frac in prev_dist.head(5).items():
        lines.append(f"    - {name}: {frac:.0%}")

    # 启动特征 vs 对照
    lines.append("\n## 2. 主升浪个股启动时的特征（中位数：主升组 vs 对照组）\n")
    sq = _quantile_table(all_surges, FEATURE_KEYS)
    cq = _quantile_table(all_control, FEATURE_KEYS)
    lines.append("| 特征 | 主升组中位 | 对照组中位 | 判别方向 |")
    lines.append("|---|---|---|---|")
    label = {
        "up_dn_power_ratio": "上/下笔力度比",
        "ma_spread_pct": "MA5-MA20散度%",
        "last_up_angle": "最近向上笔角度",
        "vol_ratio": "量比(20日)",
        "ret20": "20日涨幅%",
        "n_pivots": "中枢个数",
        "pivot_width_pct": "中枢宽度%",
        "above_zg": "立于中枢上方比例",
    }
    for key in FEATURE_KEYS:
        sv = sq.get(key, {}).get("中位")
        cv = cq.get(key, {}).get("中位")
        arrow = ""
        if sv is not None and cv is not None:
            arrow = "↑ 主升更高" if sv > cv else ("↓ 主升更低" if sv < cv else "≈")
        lines.append(f"| {label.get(key, key)} | {sv} | {cv} | {arrow} |")

    # 止损止盈
    lines.append("\n## 3. 止损止盈策略建议（数据驱动）\n")
    dd = df["max_dd"]
    lines.append(
        f"- **区间内最大回撤**（自滚动高点）：中位 **{dd.median():.1%}**，P75 {dd.quantile(0.75):.1%}，P90 {dd.quantile(0.90):.1%}"
    )
    lines.append(f"  → 跟踪止损幅度宜取约 **{dd.quantile(0.75):.0%}**（覆盖 75% 正常洗盘，过紧会被甩下车）")
    end_dist = df["end_state"].map(lambda r: REGIME_CN.get(Regime(r), str(r))).value_counts(normalize=True)
    lines.append("- **主升浪结束方式**（区间结束后紧邻状态）：")
    for name, frac in end_dist.head(5).items():
        lines.append(f"    - {name}: {frac:.0%}")
    fsg = pd.to_numeric(df["first_sell_gain"], errors="coerce").dropna()
    if len(fsg):
        lines.append(
            f"- **背驰/破坏离场的收益捕获**：首次进入 9/10 离场的中位收益 **{fsg.median():.1%}**，"
            f"对照区间峰值涨幅中位 {df['gain'].median():.1%} "
            f"→ 背驰减仓可锁定约 {fsg.median() / df['gain'].median() * 100:.0f}% 的峰值涨幅"
        )
    lines.append(
        "\n**综合建议**：主升中段（7）用 **笔结构止损 SL2 / 中枢下沿** 托底、让利润奔跑；"
        "加速段（8）切换为 **跟踪止损**（≈上面 P75 回撤）；出现 **背驰(9)** 减仓、"
        "**结构破坏(10)** 清仓。与回测结论一致：结构/波动止损优于一见风吹草动就走的快速状态止损。"
    )
    return "\n".join(lines)


def main():
    t0 = time.time()
    print("=" * 90)
    print("  主升浪特征研究")
    print("=" * 90)
    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只个股 | {n_workers} 进程\n")

    all_surges, all_control = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, files, chunksize=20), 1):
            if res:
                all_surges.extend(res["surges"])
                all_control.extend(res["control"])
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 主升浪累计 {len(all_surges)} | {time.time() - t0:.0f}s")

    print(f"\n[扫描完成] {time.time() - t0:.0f}s | 主升浪 {len(all_surges)} 次 | 对照样本 {len(all_control)}\n")

    report = build_report(all_surges, all_control)
    print(report)
    (OUTPUT_DIR / "surge_characteristics.md").write_text(report, encoding="utf-8")
    with open(OUTPUT_DIR / "surge_characteristics.json", "w", encoding="utf-8") as f:
        json.dump({"surges": all_surges, "n_control": len(all_control)}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[输出] {OUTPUT_DIR / 'surge_characteristics.md'}")


if __name__ == "__main__":
    main()
