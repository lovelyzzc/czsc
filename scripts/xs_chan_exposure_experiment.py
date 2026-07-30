"""XS-Chan 暴露缺口实验 — FC-Fill 变体分析

**纯描述性分析**：不修改 Stage 2/3 任何代码或输出。

实验设计：
读取 Stage 2 冻结的 path_weekly.parquet，构建 FC-Fill 变体：
当 FC 路径在某周欠配（持仓 < target_slots）时，用 F 路径候选补充空位。
比较 FC-Fill vs F 路径的净收益差异，量化暴露缺口的净贡献。

    uv run --no-sync python scripts/xs_chan_exposure_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE2_OUTPUT_DIRS = list((REPO_ROOT / "scripts/_output/xs_chan_exploration_stage2").glob("STAGE2_*"))
XS_CHAN_REPORT_DIR = REPO_ROOT / "scripts/_output/xs_chan_excess_return_report"

OUTPUT_DIR = REPO_ROOT / "scripts/_output/xs_chan_exposure_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SLOTS = 50
WEEKS_PER_YEAR = 52.0
ROUNDTRIP_COST_BPS = 40


def find_stage2_dir() -> Path | None:
    """找到最新的 Stage 2 输出目录。"""
    dirs = sorted(STAGE2_OUTPUT_DIRS, key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        if (d / "path_weekly.parquet").exists():
            return d
    return None


def load_weekly_paths(stage2_dir: Path) -> pd.DataFrame:
    """加载冻结的 path_weekly.parquet。"""
    return pd.read_parquet(stage2_dir / "path_weekly.parquet")


def compute_fc_fill(weekly: pd.DataFrame) -> pd.DataFrame:
    """构建 FC-Fill 变体：FC 欠配时用 F 补充。

    weekly 的结构（来自 Stage 2）：
    - week_label: 周标签
    - arm: 路径名称 (FC / F / FMA / R_match_*)
    - return_net: 周净收益率
    - n_holdings: 当周持仓数
    - exposure: 暴露率 = n_holdings / target_slots
    """
    fc = weekly[weekly["arm"] == "FC"].set_index("week_label").sort_index()
    f = weekly[weekly["arm"] == "F"].set_index("week_label").sort_index()

    common_weeks = fc.index.intersection(f.index)
    fc = fc.loc[common_weeks]
    f = f.loc[common_weeks]

    fc_exposure = fc["n_holdings"] / TARGET_SLOTS
    f_exposure = f["n_holdings"] / TARGET_SLOTS

    # FC-Fill: 当 FC 欠配时，按比例混入 F 收益
    # fill_weight = max(0, target - fc_holdings) / target
    fill_weight = ((TARGET_SLOTS - fc["n_holdings"]) / TARGET_SLOTS).clip(lower=0, upper=1)
    fc_weight = 1.0 - fill_weight

    fc_fill_return = fc_weight * fc["return_net"] + fill_weight * f["return_net"]
    fc_fill_exposure = fc_weight * fc_exposure + fill_weight * f_exposure

    result = pd.DataFrame({
        "week_label": common_weeks,
        "fc_return": fc["return_net"].values,
        "f_return": f["return_net"].values,
        "fc_fill_return": fc_fill_return.values,
        "fc_holdings": fc["n_holdings"].values,
        "f_holdings": f["n_holdings"].values,
        "fill_weight": fill_weight.values,
        "fc_exposure": fc_exposure.values,
        "fc_fill_exposure": fc_fill_exposure.values,
    }).reset_index(drop=True)

    return result


def compute_active_stats(weekly_diffs: np.ndarray, label: str) -> dict:
    """计算活跃收益统计。"""
    n = len(weekly_diffs)
    if n < 3:
        return {"label": label, "n": n, "status": "insufficient_data"}

    mean = float(np.nanmean(weekly_diffs))
    std = float(np.nanstd(weekly_diffs, ddof=1))
    se = std / np.sqrt(n) if std > 0 else np.nan
    t = mean / se if se > 0 else np.nan
    ann_mean = mean * WEEKS_PER_YEAR
    ann_se = se * np.sqrt(WEEKS_PER_YEAR) if not np.isnan(se) else np.nan
    ir = mean / std * np.sqrt(WEEKS_PER_YEAR) if std > 0 else np.nan

    return {
        "label": label,
        "n": n,
        "weekly_mean_bps": round(mean * 10000, 1),
        "weekly_se_bps": round(se * 10000, 1) if not np.isnan(se) else np.nan,
        "t_stat": round(t, 2) if not np.isnan(t) else np.nan,
        "annualized_pct": round(ann_mean * 100, 1),
        "annualized_se_pct": round(ann_se * 100, 1) if not np.isnan(ann_se) else np.nan,
        "IR": round(ir, 2) if not np.isnan(ir) else np.nan,
    }


def main():
    t0 = time.time()

    stage2_dir = find_stage2_dir()
    if stage2_dir is None:
        print("[ERROR] 找不到 Stage 2 输出目录（path_weekly.parquet）")
        return

    print(f"[Stage 2] {stage2_dir}")
    weekly = load_weekly_paths(stage2_dir)
    print(f"[数据] {len(weekly)} 行，路径 {weekly['arm'].unique()}")

    fc_fill = compute_fc_fill(weekly)
    print(f"\n[FC-Fill] {len(fc_fill)} 周")
    print(f"  FC 欠配周数（fill_weight > 0）: {(fc_fill['fill_weight'] > 0).sum()}")
    print(f"  FC 欠配率: {(fc_fill['fill_weight'] > 0).mean():.1%}")
    print(f"  平均 fill_weight: {fc_fill['fill_weight'].mean():.3f}")
    print(f"  FC 平均持仓: {fc_fill['fc_holdings'].mean():.1f} / {TARGET_SLOTS}")

    # 三条路径对比
    pairs = {
        "FC-Fill vs F": fc_fill["fc_fill_return"].values - fc_fill["f_return"].values,
        "FC vs F (原始)": fc_fill["fc_return"].values - fc_fill["f_return"].values,
        "FC-Fill vs FC": fc_fill["fc_fill_return"].values - fc_fill["fc_return"].values,
    }

    print(f"\n{'=' * 80}")
    print("暴露缺口实验结果")
    print(f"{'=' * 80}")

    results = {}
    for pair_name, diffs in pairs.items():
        valid = diffs[np.isfinite(diffs)]
        stats = compute_active_stats(valid, pair_name)
        results[pair_name] = stats
        print(f"\n  {pair_name}:")
        print(f"    周均值:     {stats.get('weekly_mean_bps', 'N/A')} bps")
        print(f"    年化:       {stats.get('annualized_pct', 'N/A')}%")
        print(f"    t-stat:     {stats.get('t_stat', 'N/A')}")
        print(f"    IR:         {stats.get('IR', 'N/A')}")

    # 暴露分析
    exposure_gap = (1.0 - fc_fill["fc_exposure"]).mean()
    exposure_filled = (1.0 - fc_fill["fc_fill_exposure"]).mean()

    print(f"\n{'=' * 80}")
    print("暴露分析")
    print(f"{'=' * 80}")
    print(f"  FC 平均暴露缺口:      {exposure_gap:.1%}")
    print(f"  FC-Fill 残留缺口:     {exposure_filled:.1%}")
    print(f"  缺口修复率:           {(1 - exposure_filled / max(exposure_gap, 1e-9)):.1%}")

    # 权益曲线
    fc_equity = (1 + fc_fill["fc_return"]).cumprod()
    f_equity = (1 + fc_fill["f_return"]).cumprod()
    fc_fill_equity = (1 + fc_fill["fc_fill_return"]).cumprod()

    fc_fill.to_parquet(OUTPUT_DIR / "fc_fill_weekly.parquet", index=False)

    # Summary
    summary = {
        "stage2_dir": str(stage2_dir),
        "n_weeks": int(len(fc_fill)),
        "exposure_gap": round(float(exposure_gap), 4),
        "exposure_filled_residual": round(float(exposure_filled), 4),
        "pair_stats": results,
        "fc_avg_holdings": round(float(fc_fill["fc_holdings"].mean()), 1),
        "f_avg_holdings": round(float(fc_fill["f_holdings"].mean()), 1),
        "underfill_weeks_pct": round(float((fc_fill["fill_weight"] > 0).mean()) * 100, 1),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Plotly equity curve
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(fc_equity))), y=fc_equity, name="FC", mode="lines"))
        fig.add_trace(go.Scatter(x=list(range(len(f_equity))), y=f_equity, name="F", mode="lines"))
        fig.add_trace(go.Scatter(
            x=list(range(len(fc_fill_equity))), y=fc_fill_equity,
            name="FC-Fill", mode="lines", line=dict(dash="dash")
        ))
        fig.update_layout(
            title="XS-Chan 暴露缺口实验: FC vs F vs FC-Fill",
            yaxis_type="log", yaxis_title="累计净值",
            xaxis_title="周序号",
        )
        fig.write_html(OUTPUT_DIR / "equity_comparison.html", include_plotlyjs="cdn")
        print(f"\n[图表] {OUTPUT_DIR / 'equity_comparison.html'}")
    except Exception as e:
        print(f"  equity HTML 失败: {e}")

    print(f"\n[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
