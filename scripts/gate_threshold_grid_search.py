"""门控阈值网格搜索 —— 以 OOS 超额收益为目标函数

读取 candidates.parquet（门控前候选），在特征空间中搜索最优阈值组合，
同时支持方向翻转（Gte/Lte）。

搜索维度：
- vol_ratio_threshold ∈ [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
- vol_ratio_direction ∈ {gte, lte}
- ma_spread_threshold ∈ [-5, -2, 0, 3, 5, 10]
- ma_spread_direction ∈ {gte, lte}
- use_above_zg ∈ {True, False}

目标函数：OOS (seg=='test') 上 fwd_5d/fwd_20d 条件均值超额。
对照组：全市场同期均值。

输出：
- scripts/_output/gate_grid_search_results.parquet
- 控制台打印 Top-10 配置

    uv run --no-sync python scripts/gate_threshold_grid_search.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
CANDIDATES_PATH = OUTPUT_DIR / "surge_candidates" / "candidates.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"

VOL_RATIO_THRESHOLDS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
VOL_RATIO_DIRECTIONS = ["gte", "lte"]
MA_SPREAD_THRESHOLDS = [-5.0, -2.0, 0.0, 3.0, 5.0, 10.0]
MA_SPREAD_DIRECTIONS = ["gte", "lte"]
USE_ABOVE_ZG_OPTIONS = [True, False]

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


def main():
    t0 = time.time()
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

    # 全市场基准
    is_mask = delay0["seg"] == "train"
    oos_mask = delay0["seg"] == "test"
    baseline = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        baseline[f"is_{col}"] = delay0.loc[is_mask & has_fwd, col].mean()
        baseline[f"oos_{col}"] = delay0.loc[oos_mask & has_fwd, col].mean()
    print(f"  基准 (无门控): IS fwd_5d={baseline['is_fwd_5d']:.3f}% | OOS fwd_5d={baseline['oos_fwd_5d']:.3f}%")

    configs = list(itertools.product(
        VOL_RATIO_THRESHOLDS,
        VOL_RATIO_DIRECTIONS,
        MA_SPREAD_THRESHOLDS,
        MA_SPREAD_DIRECTIONS,
        USE_ABOVE_ZG_OPTIONS,
    ))
    print(f"\n[搜索] {len(configs)} 种配置 ...")

    results = []
    for ci, (vr_th, vr_dir, sp_th, sp_dir, use_zg) in enumerate(configs, 1):
        passed = delay0.apply(
            lambda r: _gate_passes(r, vr_th, vr_dir, sp_th, sp_dir, use_zg), axis=1
        )

        n_pass = passed.sum()
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
            "n_pass_total": int(n_pass),
            "n_pass_is": int(pass_is.sum()),
            "n_pass_oos": int(pass_oos.sum()),
        }

        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            is_mean = delay0.loc[pass_is, col].mean() if pass_is.sum() > 0 else np.nan
            oos_mean = delay0.loc[pass_oos, col].mean() if pass_oos.sum() > 0 else np.nan
            row[f"is_{col}_mean"] = round(is_mean, 4) if not np.isnan(is_mean) else np.nan
            row[f"oos_{col}_mean"] = round(oos_mean, 4) if not np.isnan(oos_mean) else np.nan
            row[f"is_{col}_excess"] = round(is_mean - baseline[f"is_{col}"], 4) if not np.isnan(is_mean) else np.nan
            row[f"oos_{col}_excess"] = round(oos_mean - baseline[f"oos_{col}"], 4) if not np.isnan(oos_mean) else np.nan

        # 扣除交易成本的净超额（单边 0.15%）
        for w in FWD_WINDOWS:
            col_net = f"oos_fwd_{w}d_net_excess"
            raw = row.get(f"oos_fwd_{w}d_excess", np.nan)
            if not np.isnan(raw):
                row[col_net] = round(raw - COST_ONE_WAY * 200 / w, 4)
            else:
                row[col_net] = np.nan

        results.append(row)

        if ci % 100 == 0:
            print(f"  [{ci}/{len(configs)}] 有效配置 {len(results)} ...")

    df_results = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "gate_grid_search_results.parquet"
    df_results.to_parquet(out_path, index=False)
    print(f"\n[输出] {len(df_results)} 种有效配置 → {out_path}")

    # 排序：OOS fwd_5d 净超额降序
    if "oos_fwd_5d_net_excess" in df_results.columns:
        top = df_results.nlargest(10, "oos_fwd_5d_net_excess")
        print("\n" + "=" * 100)
        print("Top-10 配置（OOS fwd_5d 净超额 降序）:")
        print("=" * 100)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  #{rank}: vr={r['vr_direction']}({r['vr_threshold']:.1f}) "
                f"sp={r['sp_direction']}({r['sp_threshold']:.1f}) "
                f"zg={r['use_above_zg']} | "
                f"n_oos={r['n_pass_oos']:.0f} | "
                f"fwd_5d_excess={r['oos_fwd_5d_excess']:.4f}% | "
                f"fwd_20d_excess={r.get('oos_fwd_20d_excess', float('nan')):.4f}% | "
                f"net_5d={r['oos_fwd_5d_net_excess']:.4f}%"
            )

    # 对比 Top-1 vs 当前默认 (vr>=1.2, sp>=3.0, zg=True)
    default_mask = (
        (df_results["vr_threshold"] == 1.2)
        & (df_results["vr_direction"] == "gte")
        & (df_results["sp_threshold"] == 3.0)
        & (df_results["sp_direction"] == "gte")
        & (df_results["use_above_zg"] == True)
    )
    if default_mask.any():
        d = df_results.loc[default_mask].iloc[0]
        print(f"\n当前默认配置:")
        print(
            f"  vr=gte(1.2) sp=gte(3.0) zg=True | "
            f"n_oos={d['n_pass_oos']:.0f} | "
            f"fwd_5d_excess={d['oos_fwd_5d_excess']:.4f}% | "
            f"fwd_20d_excess={d.get('oos_fwd_20d_excess', float('nan')):.4f}%"
        )

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
