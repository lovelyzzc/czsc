"""均值回复信号回测 —— 买入下跌态出口，含随机对照

基于 FSM 审计数据（states.parquet + panel.parquet），检测 Regime 1(Downtrend) →
{2(FirstBuy), 3(SecondBuy), 4(PivotBuilding)} 的转换，作为均值回复入场信号。

回测指标：
1. 信号条件前向收益 vs 全市场基准
2. 分下跌持续天数的超额分布
3. 随机对照 beta 剥离
4. IS/OOS 分切

输出：
- scripts/_output/reversion_backtest_results.json
- 控制台打印汇总

    uv run --no-sync python scripts/reversion_signal_backtest.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
STATES_PATH = OUTPUT_DIR / "fsm_audit" / "states.parquet"
PANEL_PATH = OUTPUT_DIR / "surge_candidates" / "panel.parquet"

RECOVERY_REGIMES = {2, 3, 4}  # FirstBuy, SecondBuy, PivotBuilding
DOWNTREND = 1
FWD_WINDOWS = [5, 10, 20]
TRAIN_END = pd.Timestamp("2023-12-31")
N_RANDOM_CONTROLS = 50


def _compute_fwd_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """为面板中的每个 (symbol, dt) 计算前向收益。"""
    panel_sorted = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        panel_sorted[col] = (
            panel_sorted.groupby("symbol")["close"]
            .transform(lambda s: s.shift(-w) / s - 1)
            * 100
        )
    return panel_sorted


def detect_reversion_signals(states: pd.DataFrame, min_down: int = 5, max_down: int = 20) -> pd.DataFrame:
    """检测从 Downtrend → Recovery 的转换信号。"""
    states_sorted = states.sort_values(["symbol", "dt"]).reset_index(drop=True)

    signals = []
    for sym, grp in states_sorted.groupby("symbol"):
        regimes = grp["regime"].values
        dates = grp["dt"].values
        n = len(regimes)
        down_count = 0

        for i in range(1, n):
            if regimes[i - 1] == DOWNTREND:
                down_count += 1
            else:
                down_count = 0

            if regimes[i - 1] == DOWNTREND and int(regimes[i]) in RECOVERY_REGIMES:
                if min_down <= down_count <= max_down:
                    signals.append({
                        "symbol": sym,
                        "dt": dates[i],
                        "from_regime": int(regimes[i - 1]),
                        "to_regime": int(regimes[i]),
                        "down_days": down_count,
                    })
                down_count = 0

    return pd.DataFrame(signals)


def main():
    t0 = time.time()
    print("[加载] 读取 states.parquet ...")
    states = pd.read_parquet(STATES_PATH)
    print(f"  行数: {len(states)}, 股票数: {states['symbol'].nunique()}")

    print("[加载] 读取 panel.parquet ...")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  行数: {len(panel)}")

    print("[计算] 面板前向收益 ...")
    panel_fwd = _compute_fwd_returns(panel)

    print("[检测] 均值回复信号 ...")
    signals = detect_reversion_signals(states, min_down=5, max_down=20)
    print(f"  总信号数: {len(signals)}")

    if len(signals) == 0:
        print("[错误] 没有检测到信号，退出")
        return

    signals_merged = signals.merge(
        panel_fwd[["symbol", "dt"] + [f"fwd_{w}d" for w in FWD_WINDOWS]],
        on=["symbol", "dt"],
        how="left",
    )
    has_fwd = signals_merged["fwd_5d"].notna()
    print(f"  有前向收益信号: {has_fwd.sum()}")

    signals_merged["seg"] = np.where(signals_merged["dt"] <= TRAIN_END, "train", "test")

    # 全市场基准
    baseline = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        valid = panel_fwd[col].dropna()
        baseline[col] = valid.mean()
    print(f"  全市场基准: fwd_5d={baseline['fwd_5d']:.4f}% | fwd_20d={baseline['fwd_20d']:.4f}%")

    results = {"baseline": {k: round(v, 4) for k, v in baseline.items()}}

    # 1. 信号整体超额
    for seg_label, seg_mask in [("ALL", pd.Series(True, index=signals_merged.index)),
                                 ("IS", signals_merged["seg"] == "train"),
                                 ("OOS", signals_merged["seg"] == "test")]:
        sub = signals_merged.loc[seg_mask & has_fwd]
        seg_data = {"n_signals": int(len(sub))}
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            mean = sub[col].mean()
            excess = mean - baseline[col]
            median = sub[col].median()
            seg_data[col] = {
                "mean": round(mean, 4),
                "median": round(median, 4),
                "excess": round(excess, 4),
                "std": round(sub[col].std(), 4),
                "t_stat": round(excess / (sub[col].std() / np.sqrt(len(sub))), 3) if len(sub) > 1 else 0,
            }
        results[seg_label] = seg_data

    # 2. 按到达态分组
    by_target = {}
    regime_names = {2: "FirstBuy", 3: "SecondBuy", 4: "PivotBuilding"}
    for r, rname in regime_names.items():
        sub = signals_merged.loc[(signals_merged["to_regime"] == r) & has_fwd]
        seg_oos = sub[sub["seg"] == "test"]
        by_target[rname] = {
            "n_all": int(len(sub)),
            "n_oos": int(len(seg_oos)),
        }
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            by_target[rname][f"oos_{col}_mean"] = round(seg_oos[col].mean(), 4) if len(seg_oos) > 0 else None
            by_target[rname][f"oos_{col}_excess"] = (
                round(seg_oos[col].mean() - baseline[col], 4) if len(seg_oos) > 0 else None
            )
    results["by_target_regime"] = by_target

    # 3. 按下跌持续天数分桶
    bins = [(5, 7), (8, 12), (13, 20)]
    by_duration = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}d"
        sub = signals_merged.loc[
            (signals_merged["down_days"] >= lo)
            & (signals_merged["down_days"] <= hi)
            & has_fwd
        ]
        seg_oos = sub[sub["seg"] == "test"]
        by_duration[label] = {
            "n_all": int(len(sub)),
            "n_oos": int(len(seg_oos)),
        }
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            by_duration[label][f"oos_{col}_mean"] = round(seg_oos[col].mean(), 4) if len(seg_oos) > 0 else None
            by_duration[label][f"oos_{col}_excess"] = (
                round(seg_oos[col].mean() - baseline[col], 4) if len(seg_oos) > 0 else None
            )
    results["by_down_duration"] = by_duration

    # 4. 随机对照 beta 剥离
    print(f"[随机对照] {N_RANDOM_CONTROLS} 次抽样 ...")
    all_dates = panel_fwd["dt"].unique()
    random_means = {f"fwd_{w}d": [] for w in FWD_WINDOWS}

    rng = np.random.default_rng(42)
    oos_signals = signals_merged.loc[(signals_merged["seg"] == "test") & has_fwd]
    n_oos = len(oos_signals)

    if n_oos > 0:
        for trial in range(N_RANDOM_CONTROLS):
            sampled_dates = rng.choice(all_dates, size=n_oos, replace=True)
            random_sample = panel_fwd[panel_fwd["dt"].isin(sampled_dates)]
            if len(random_sample) == 0:
                continue
            actual_sample = random_sample.sample(n=min(n_oos, len(random_sample)), random_state=trial)
            for w in FWD_WINDOWS:
                col = f"fwd_{w}d"
                m = actual_sample[col].mean()
                if not np.isnan(m):
                    random_means[col].append(m)

        random_control = {}
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            rm = random_means[col]
            sig_mean = oos_signals[col].mean()
            rc_mean = np.mean(rm) if rm else np.nan
            rc_std = np.std(rm) if rm else np.nan
            excess_vs_rc = sig_mean - rc_mean if not np.isnan(rc_mean) else np.nan
            z_score = excess_vs_rc / rc_std if rc_std > 0 else np.nan
            random_control[col] = {
                "signal_mean": round(sig_mean, 4),
                "random_control_mean": round(rc_mean, 4) if not np.isnan(rc_mean) else None,
                "excess_vs_control": round(excess_vs_rc, 4) if not np.isnan(excess_vs_rc) else None,
                "z_score": round(z_score, 3) if not np.isnan(z_score) else None,
            }
        results["random_control_oos"] = random_control

    # 输出
    out_path = OUTPUT_DIR / "reversion_backtest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[输出] {out_path}")

    # 打印汇总
    print("\n" + "=" * 80)
    print("均值回复信号回测汇总")
    print("=" * 80)

    for seg in ["ALL", "IS", "OOS"]:
        d = results[seg]
        print(f"\n{seg} (n={d['n_signals']}):")
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            print(f"  {col}: mean={d[col]['mean']:.4f}% | excess={d[col]['excess']:.4f}% | t={d[col]['t_stat']:.2f}")

    print("\n按到达态:")
    for rname, rd in results["by_target_regime"].items():
        print(f"  {rname}: n_oos={rd['n_oos']} | oos_fwd_5d_excess={rd.get('oos_fwd_5d_excess', 'N/A')} | oos_fwd_20d_excess={rd.get('oos_fwd_20d_excess', 'N/A')}")

    print("\n按下跌持续天数:")
    for label, dd in results["by_down_duration"].items():
        print(f"  {label}: n_oos={dd['n_oos']} | oos_fwd_5d_excess={dd.get('oos_fwd_5d_excess', 'N/A')} | oos_fwd_20d_excess={dd.get('oos_fwd_20d_excess', 'N/A')}")

    if "random_control_oos" in results:
        print("\n随机对照 (OOS):")
        for col, rc in results["random_control_oos"].items():
            print(f"  {col}: signal={rc['signal_mean']} | control={rc['random_control_mean']} | excess={rc['excess_vs_control']} | z={rc['z_score']}")

    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
