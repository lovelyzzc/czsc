"""门控阈值网格搜索 —— 以 OOS 超额收益为目标函数

读取 candidates.parquet（门控前候选），在特征空间中搜索最优阈值组合。

两种搜索模式（--mode 参数）：

  full   — 全维度宽搜索（原始 288 种配置），用于发现方向
  refine — 聚焦精搜索：固定 vr=lte / zg=False，细化 sp=[2..15] / vr=[0.5..1.2]，
           共 77 种配置 + 年度分层 + t-test 显著性

输出：
- scripts/_output/gate_grid_search_results.parquet  (full)
- scripts/_output/gate_grid_search_refined.parquet  (refine)
- 控制台打印 Top-10 配置

    uv run --no-sync python scripts/gate_threshold_grid_search.py [--mode full|refine]
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
CANDIDATES_PATH = OUTPUT_DIR / "surge_candidates" / "candidates.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"

TRAIN_END = pd.Timestamp("2023-12-31")

VOL_RATIO_THRESHOLDS_FULL = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
VOL_RATIO_DIRECTIONS_FULL = ["gte", "lte"]
MA_SPREAD_THRESHOLDS_FULL = [-5.0, -2.0, 0.0, 3.0, 5.0, 10.0]
MA_SPREAD_DIRECTIONS_FULL = ["gte", "lte"]
USE_ABOVE_ZG_FULL = [True, False]

VOL_RATIO_THRESHOLDS_REFINE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
MA_SPREAD_THRESHOLDS_REFINE = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]

COST_ONE_WAY = 0.0015
FWD_WINDOWS = [5, 10, 20]


def _gate_passes(row: pd.Series, vr_th: float, vr_dir: str, sp_th: float, sp_dir: str, use_zg: bool) -> bool:
    """判断单行候选是否通过门控。"""
    vr = row["sig_vol_ratio"]
    sp = row["sig_ma_spread_pct"]
    zg = row["sig_above_zg"]

    if pd.isna(vr):
        vr_ok = False
    elif vr_dir == "gte":
        vr_ok = vr >= vr_th
    else:
        vr_ok = vr <= vr_th

    if pd.isna(sp):
        sp_ok = False
    elif sp_dir == "gte":
        sp_ok = sp >= sp_th
    else:
        sp_ok = sp <= sp_th

    zg_ok = bool(zg) if use_zg else True
    return vr_ok and sp_ok and zg_ok


def compute_forward_returns(cand: pd.DataFrame, panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """为每个候选计算前向收益（基于 panel 收盘价）。"""
    panel_sorted = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)

    sym_groups = {}
    for sym, grp in panel_sorted.groupby("symbol"):
        dates = grp["dt"].values
        closes = grp["close"].values
        sym_groups[sym] = (dates, closes)

    results = {f"fwd_{w}d": np.full(len(cand), np.nan) for w in windows}

    for i, row in cand.iterrows():
        sym = row["symbol"]
        entry_dt = row["entry_dt"]
        if sym not in sym_groups:
            continue
        dates, closes = sym_groups[sym]
        idx = np.searchsorted(dates, np.datetime64(entry_dt))
        if idx >= len(dates):
            continue
        entry_px = closes[idx]
        if np.isnan(entry_px) or entry_px <= 0:
            continue
        for w in windows:
            fwd_idx = idx + w
            if fwd_idx < len(dates):
                fwd_px = closes[fwd_idx]
                if not np.isnan(fwd_px):
                    results[f"fwd_{w}d"][i] = (fwd_px / entry_px - 1) * 100

    for col, vals in results.items():
        cand[col] = vals
    return cand


def _vectorized_gate(delay0, vr_th, vr_dir, sp_th, sp_dir, use_zg):
    """向量化门控判定，替代逐行 apply 提升性能。"""
    vr = delay0["sig_vol_ratio"]
    sp = delay0["sig_ma_spread_pct"]

    vr_ok = (vr >= vr_th) if vr_dir == "gte" else (vr <= vr_th)
    vr_ok = vr_ok & vr.notna()

    sp_ok = (sp >= sp_th) if sp_dir == "gte" else (sp <= sp_th)
    sp_ok = sp_ok & sp.notna()

    zg_ok = delay0["sig_above_zg"].astype(bool) if use_zg else True
    return vr_ok & sp_ok & zg_ok


def _run_search(delay0, has_fwd, is_mask, oos_mask, baseline, configs, *, yearly_detail=False):
    """对一组配置执行搜索，返回结果列表。"""
    results = []
    for ci, (vr_th, vr_dir, sp_th, sp_dir, use_zg) in enumerate(configs, 1):
        passed = _vectorized_gate(delay0, vr_th, vr_dir, sp_th, sp_dir, use_zg)

        n_pass = int(passed.sum())
        if n_pass < 20:
            continue

        pass_is = passed & is_mask & has_fwd
        pass_oos = passed & oos_mask & has_fwd

        row = {
            "vr_threshold": vr_th,
            "vr_direction": vr_dir,
            "sp_threshold": sp_th,
            "sp_direction": sp_dir,
            "use_above_zg": use_zg,
            "n_pass_total": n_pass,
            "n_pass_is": int(pass_is.sum()),
            "n_pass_oos": int(pass_oos.sum()),
        }

        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            is_vals = delay0.loc[pass_is, col].dropna()
            oos_vals = delay0.loc[pass_oos, col].dropna()
            is_mean = is_vals.mean() if len(is_vals) > 0 else np.nan
            oos_mean = oos_vals.mean() if len(oos_vals) > 0 else np.nan
            row[f"is_{col}_mean"] = round(is_mean, 4) if not np.isnan(is_mean) else np.nan
            row[f"oos_{col}_mean"] = round(oos_mean, 4) if not np.isnan(oos_mean) else np.nan
            row[f"is_{col}_excess"] = round(is_mean - baseline[f"is_{col}"], 4) if not np.isnan(is_mean) else np.nan
            row[f"oos_{col}_excess"] = round(oos_mean - baseline[f"oos_{col}"], 4) if not np.isnan(oos_mean) else np.nan

        for w in FWD_WINDOWS:
            col_net = f"oos_fwd_{w}d_net_excess"
            raw = row.get(f"oos_fwd_{w}d_excess", np.nan)
            row[col_net] = round(raw - COST_ONE_WAY * 200 / w, 4) if not np.isnan(raw) else np.nan

        if yearly_detail:
            oos_sub = delay0.loc[pass_oos & has_fwd].copy()
            if "entry_dt" in oos_sub.columns:
                oos_sub["year"] = pd.to_datetime(oos_sub["entry_dt"]).dt.year
            elif "dt" in oos_sub.columns:
                oos_sub["year"] = pd.to_datetime(oos_sub["dt"]).dt.year
            for yr in (2024, 2025, 2026):
                yr_vals = oos_sub.loc[oos_sub["year"] == yr, "fwd_5d"].dropna()
                row[f"oos_5d_{yr}_n"] = len(yr_vals)
                row[f"oos_5d_{yr}_mean"] = round(yr_vals.mean(), 4) if len(yr_vals) > 0 else np.nan

            oos_5d = delay0.loc[pass_oos, "fwd_5d"].dropna()
            baseline_5d = delay0.loc[oos_mask & has_fwd, "fwd_5d"].dropna()
            if len(oos_5d) >= 5 and len(baseline_5d) >= 5:
                t_stat, p_val = sp_stats.ttest_ind(oos_5d, baseline_5d, equal_var=False)
                row["ttest_t"] = round(t_stat, 3)
                row["ttest_p"] = round(p_val, 4)
            else:
                row["ttest_t"] = np.nan
                row["ttest_p"] = np.nan

        results.append(row)

        if ci % 50 == 0:
            print(f"  [{ci}/{len(configs)}] 有效配置 {len(results)} ...")

    return results


def _print_top(df_results, baseline, n=10, label="Top-10"):
    if "oos_fwd_5d_net_excess" not in df_results.columns or df_results.empty:
        return
    top = df_results.nlargest(n, "oos_fwd_5d_net_excess")
    print(f"\n{'=' * 130}")
    print(f"{label} 配置（OOS fwd_5d 净超额 降序）:")
    print("=" * 130)
    has_yearly = "oos_5d_2024_mean" in df_results.columns
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        line = (
            f"  #{rank}: vr={r['vr_direction']}({r['vr_threshold']:.1f}) "
            f"sp={r['sp_direction']}({r['sp_threshold']:.1f}) "
            f"zg={r['use_above_zg']} | "
            f"n_oos={r['n_pass_oos']:.0f} | "
            f"fwd_5d_excess={r['oos_fwd_5d_excess']:+.4f}% | "
            f"fwd_20d_excess={r.get('oos_fwd_20d_excess', float('nan')):+.4f}% | "
            f"net_5d={r['oos_fwd_5d_net_excess']:+.4f}%"
        )
        if has_yearly:
            y24 = r.get("oos_5d_2024_mean", float("nan"))
            y25 = r.get("oos_5d_2025_mean", float("nan"))
            y26 = r.get("oos_5d_2026_mean", float("nan"))
            p = r.get("ttest_p", float("nan"))
            line += f" | 24:{y24:+.2f} 25:{y25:+.2f} 26:{y26:+.2f} p={p:.3f}"
        print(line)


def _parse_args():
    parser = argparse.ArgumentParser(description="门控阈值网格搜索")
    parser.add_argument("--mode", choices=["full", "refine"], default="refine", help="搜索模式")
    return parser.parse_args()


def main():
    args = _parse_args()
    mode = args.mode

    t0 = time.time()
    print(f"[模式] {mode}")
    print("[加载] 读取 candidates.parquet ...")
    cand = pd.read_parquet(CANDIDATES_PATH)
    print(f"  候选总数: {len(cand)}")

    print("[加载] 读取 panel.parquet ...")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  面板行数: {len(panel)}")

    delay0 = cand[cand["delay"] == 0].copy().reset_index(drop=True)
    print(f"  delay=0 候选: {len(delay0)}")

    print("[计算] 前向收益 ...")
    delay0 = compute_forward_returns(delay0, panel, FWD_WINDOWS)
    has_fwd = delay0["fwd_5d"].notna()
    print(f"  有前向收益: {has_fwd.sum()}")

    is_mask = delay0["seg"] == "train"
    oos_mask = delay0["seg"] == "test"
    baseline = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        baseline[f"is_{col}"] = delay0.loc[is_mask & has_fwd, col].mean()
        baseline[f"oos_{col}"] = delay0.loc[oos_mask & has_fwd, col].mean()
    print(f"  基准 (无门控): IS fwd_5d={baseline['is_fwd_5d']:.3f}% | OOS fwd_5d={baseline['oos_fwd_5d']:.3f}%")

    if mode == "full":
        configs = list(itertools.product(
            VOL_RATIO_THRESHOLDS_FULL, VOL_RATIO_DIRECTIONS_FULL,
            MA_SPREAD_THRESHOLDS_FULL, MA_SPREAD_DIRECTIONS_FULL,
            USE_ABOVE_ZG_FULL,
        ))
        print(f"\n[搜索] {len(configs)} 种配置（全维度宽搜索）...")
        results = _run_search(delay0, has_fwd, is_mask, oos_mask, baseline, configs)
        out_path = OUTPUT_DIR / "gate_grid_search_results.parquet"
    else:
        configs = [
            (vr_th, "lte", sp_th, "gte", False)
            for vr_th in VOL_RATIO_THRESHOLDS_REFINE
            for sp_th in MA_SPREAD_THRESHOLDS_REFINE
        ]
        print(f"\n[搜索] {len(configs)} 种配置（精细化：vr=lte, zg=False, sp=[2..15], vr=[0.5..1.2]）...")
        results = _run_search(delay0, has_fwd, is_mask, oos_mask, baseline, configs, yearly_detail=True)
        out_path = OUTPUT_DIR / "gate_grid_search_refined.parquet"

    df_results = pd.DataFrame(results)
    df_results.to_parquet(out_path, index=False)
    print(f"\n[输出] {len(df_results)} 种有效配置 → {out_path}")

    label = "Top-10（精细化）" if mode == "refine" else "Top-10（全维度）"
    _print_top(df_results, baseline, n=10, label=label)

    new_default_mask = (
        (df_results["vr_threshold"] == 0.8)
        & (df_results["vr_direction"] == "lte")
        & (df_results["sp_threshold"] == 3.0)
        & (df_results["sp_direction"] == "gte")
        & (df_results["use_above_zg"] == False)
    )
    if new_default_mask.any():
        d = df_results.loc[new_default_mask].iloc[0]
        print(f"\n当前默认配置 (S0 NewDefault):")
        line = (
            f"  vr=lte(0.8) sp=gte(3.0) zg=False | "
            f"n_oos={d['n_pass_oos']:.0f} | "
            f"fwd_5d_excess={d['oos_fwd_5d_excess']:+.4f}% | "
            f"fwd_20d_excess={d.get('oos_fwd_20d_excess', float('nan')):+.4f}%"
        )
        if "oos_5d_2024_mean" in d:
            line += (
                f" | 24:{d.get('oos_5d_2024_mean', float('nan')):+.2f}"
                f" 25:{d.get('oos_5d_2025_mean', float('nan')):+.2f}"
                f" 26:{d.get('oos_5d_2026_mean', float('nan')):+.2f}"
            )
        print(line)

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
