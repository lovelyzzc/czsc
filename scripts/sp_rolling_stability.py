"""sp 阈值滚动稳定性分析

固定 vr<=0.8, zg=False，对 sp=3/5/8/10/12/15 六个阈值做 12 个月滚动窗口超额分析：
- 每个窗口内计算 5d/20d 均值超额
- 统计"正超额窗口比例"和"最大连续负超额月数"

输出: scripts/_output/sp_rolling_stability.json

    uv run --no-sync python scripts/sp_rolling_stability.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
CANDIDATES_PATH = OUTPUT_DIR / "surge_candidates" / "candidates.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"

FWD_WINDOWS = [5, 20]
SP_THRESHOLDS = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
VR_THRESHOLD = 0.8
ROLLING_MONTHS = 12


def load_and_prepare():
    """加载数据并计算前向收益。"""
    cand = pd.read_parquet(CANDIDATES_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    ps = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    for w in FWD_WINDOWS:
        ps[f"fwd_{w}d"] = ps.groupby("symbol")["close"].transform(
            lambda s: s.shift(-w) / s - 1
        ) * 100

    d0 = cand[cand["delay"] == 0].copy()
    d0 = d0.merge(ps[["symbol", "dt"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
                  left_on=["symbol", "entry_dt"], right_on=["symbol", "dt"],
                  how="left", suffixes=("", "_panel"))
    if "dt_panel" in d0.columns:
        d0.drop(columns=["dt_panel"], inplace=True)
    if "dt" not in d0.columns:
        d0.rename(columns={"entry_dt": "dt"}, inplace=True)

    return d0, ps


def gate_mask(d0: pd.DataFrame, sp_th: float) -> pd.Series:
    """固定 vr<=0.8, zg=False 的门控。"""
    vr_ok = (d0["sig_vol_ratio"] <= VR_THRESHOLD) & d0["sig_vol_ratio"].notna()
    sp_ok = (d0["sig_ma_spread_pct"] >= sp_th) & d0["sig_ma_spread_pct"].notna()
    return vr_ok & sp_ok


def max_consecutive_negative(series: pd.Series) -> int:
    """计算最大连续负值（<=0）的个数。"""
    max_run = 0
    current = 0
    for v in series:
        if v <= 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def rolling_analysis(d0: pd.DataFrame, panel_fwd: pd.DataFrame):
    """对每个 sp 阈值做滚动窗口分析。"""
    d0["month"] = pd.to_datetime(d0["dt"]).dt.to_period("M")
    panel_fwd["month"] = pd.to_datetime(panel_fwd["dt"]).dt.to_period("M")

    all_months = sorted(d0["month"].unique())

    baseline_monthly = {}
    for m in all_months:
        m_panel = panel_fwd[panel_fwd["month"] == m]
        baseline_monthly[m] = {
            f"fwd_{w}d": m_panel[f"fwd_{w}d"].dropna().mean() for w in FWD_WINDOWS
        }

    results = {}

    for sp_th in SP_THRESHOLDS:
        mask = gate_mask(d0, sp_th)
        filtered = d0[mask & d0["fwd_5d"].notna()].copy()

        strat_monthly = {}
        for m, mg in filtered.groupby("month"):
            strat_monthly[m] = {
                f"fwd_{w}d": mg[f"fwd_{w}d"].mean() for w in FWD_WINDOWS
            }
            strat_monthly[m]["n"] = len(mg)

        windows = []
        for i in range(len(all_months)):
            end_m = all_months[i]
            start_idx = i - ROLLING_MONTHS + 1
            if start_idx < 0:
                continue
            start_m = all_months[start_idx]
            window_months = all_months[start_idx:i + 1]

            strat_vals = {f"fwd_{w}d": [] for w in FWD_WINDOWS}
            base_vals = {f"fwd_{w}d": [] for w in FWD_WINDOWS}
            n_total = 0

            for m in window_months:
                if m in strat_monthly:
                    n_m = strat_monthly[m]["n"]
                    n_total += n_m
                    for w in FWD_WINDOWS:
                        strat_vals[f"fwd_{w}d"].append(
                            (strat_monthly[m][f"fwd_{w}d"], n_m)
                        )
                if m in baseline_monthly:
                    bm = baseline_monthly[m]
                    for w in FWD_WINDOWS:
                        v = bm[f"fwd_{w}d"]
                        if not np.isnan(v):
                            base_vals[f"fwd_{w}d"].append(v)

            window_result = {
                "end_month": str(end_m),
                "start_month": str(start_m),
                "n_trades": n_total,
            }

            for w in FWD_WINDOWS:
                col = f"fwd_{w}d"
                s_pairs = strat_vals[col]
                if s_pairs:
                    total_n = sum(p[1] for p in s_pairs)
                    weighted_mean = sum(p[0] * p[1] for p in s_pairs) / total_n if total_n > 0 else np.nan
                else:
                    weighted_mean = np.nan

                b_vals = base_vals[col]
                base_mean = np.mean(b_vals) if b_vals else np.nan

                excess = weighted_mean - base_mean if not (np.isnan(weighted_mean) or np.isnan(base_mean)) else np.nan
                window_result[f"{col}_mean"] = round(weighted_mean, 4) if not np.isnan(weighted_mean) else None
                window_result[f"{col}_baseline"] = round(base_mean, 4) if not np.isnan(base_mean) else None
                window_result[f"{col}_excess"] = round(excess, 4) if not np.isnan(excess) else None

            windows.append(window_result)

        excess_5d = pd.Series([w["fwd_5d_excess"] for w in windows if w["fwd_5d_excess"] is not None])
        excess_20d = pd.Series([w["fwd_20d_excess"] for w in windows if w["fwd_20d_excess"] is not None])

        results[f"sp_{sp_th}"] = {
            "sp_threshold": sp_th,
            "vr_threshold": VR_THRESHOLD,
            "total_candidates": int(mask.sum()),
            "with_fwd": int((mask & d0["fwd_5d"].notna()).sum()),
            "summary": {
                "fwd_5d": {
                    "n_windows": len(excess_5d),
                    "positive_windows": int((excess_5d > 0).sum()),
                    "positive_pct": round((excess_5d > 0).mean() * 100, 1) if len(excess_5d) > 0 else 0,
                    "mean_excess": round(excess_5d.mean(), 4) if len(excess_5d) > 0 else None,
                    "median_excess": round(excess_5d.median(), 4) if len(excess_5d) > 0 else None,
                    "max_consecutive_negative": max_consecutive_negative(excess_5d),
                },
                "fwd_20d": {
                    "n_windows": len(excess_20d),
                    "positive_windows": int((excess_20d > 0).sum()),
                    "positive_pct": round((excess_20d > 0).mean() * 100, 1) if len(excess_20d) > 0 else 0,
                    "mean_excess": round(excess_20d.mean(), 4) if len(excess_20d) > 0 else None,
                    "median_excess": round(excess_20d.median(), 4) if len(excess_20d) > 0 else None,
                    "max_consecutive_negative": max_consecutive_negative(excess_20d),
                },
            },
            "windows": windows,
        }

    return results


def main():
    t0 = time.time()
    print("[加载] 数据 ...")
    d0, panel_fwd = load_and_prepare()
    print(f"  delay0 候选: {len(d0)} | 面板: {len(panel_fwd)}")

    print("[分析] 滚动稳定性 ...")
    results = rolling_analysis(d0, panel_fwd)

    out_path = OUTPUT_DIR / "sp_rolling_stability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[输出] {out_path}")

    print("\n" + "=" * 100)
    print("sp 阈值滚动稳定性摘要（12 月滚动窗口）")
    print("=" * 100)
    print(f"{'sp阈值':>8} {'候选数':>8} {'正超额窗口%_5d':>16} {'均值超额_5d':>14} "
          f"{'最大连续负月_5d':>16} {'正超额窗口%_20d':>16} {'均值超额_20d':>14}")
    print("-" * 100)

    for key in sorted(results.keys(), key=lambda k: results[k]["sp_threshold"]):
        r = results[key]
        s5 = r["summary"]["fwd_5d"]
        s20 = r["summary"]["fwd_20d"]
        print(
            f"{r['sp_threshold']:>8.1f} {r['with_fwd']:>8} "
            f"{s5['positive_pct']:>15.1f}% {s5['mean_excess'] or 0:>14.4f} "
            f"{s5['max_consecutive_negative']:>16} "
            f"{s20['positive_pct']:>15.1f}% {s20['mean_excess'] or 0:>14.4f}"
        )

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
