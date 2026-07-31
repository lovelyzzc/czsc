"""FSM 分层预测力检验

四层解耦检验 FSM 状态描述、门控条件、入场时点、市场环境各自的前瞻预测力。

输出: scripts/_output/fsm_audit/
  - states.parquet           全市场日频 regime 缓存
  - test1_regime_returns.json
  - test2_gate_incremental.json
  - test3_event_study.json
  - test4_market_interaction.json
  - audit_summary.json       汇总诊断
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

RAW_DIR = os.path.expanduser("~/.ts_data_cache/a_stock_daily_qfq")
CANDIDATES_PATH = "scripts/_output/surge_candidates/candidates.parquet"
PANEL_PATH = "scripts/_output/surge_candidates/panel.parquet"
OUTPUT_DIR = Path("scripts/_output/fsm_audit")
STATES_PATH = OUTPUT_DIR / "states.parquet"

FWD_WINDOWS = [1, 5, 10, 20]
REGIME_NAMES = {
    0: "NotTradable",
    1: "Downtrend",
    2: "FirstBuy",
    3: "SecondBuy",
    4: "PivotBuilding",
    5: "UpwardDeparture",
    6: "ThirdBuy",
    7: "MainUptrend",
    8: "Acceleration",
    9: "Divergence",
    10: "Breakdown",
}


# ─── Step 0: generate state cache ──────────────────────────────

def step0_build_state_cache() -> dict:
    """Build full-market daily regime cache via Rust parallel projector."""
    if STATES_PATH.exists():
        import polars as pl

        df = pl.read_parquet(STATES_PATH)
        print(f"[Step 0] State cache already exists: {df.shape[0]:,} rows")
        return {"output_path": str(STATES_PATH), "total_rows": df.shape[0], "cached": True}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from czsc._native.research import build_state_cache

    print(f"[Step 0] Building state cache from {RAW_DIR} ...")
    t0 = time.time()
    result = build_state_cache(RAW_DIR, str(STATES_PATH), warmup_bars=120)
    elapsed = time.time() - t0
    print(f"[Step 0] Done in {elapsed:.1f}s — {result['total_rows']:,} rows, {result['source_count']} stocks")
    return result


# ─── helpers ────────────────────────────────────────────────────

def _load_panel_pd() -> pd.DataFrame:
    return pd.read_parquet(PANEL_PATH)


def _load_states_pd() -> pd.DataFrame:
    return pd.read_parquet(STATES_PATH)


def _load_candidates_pd() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATES_PATH)


def _ttest_vs_pop(sample: np.ndarray, pop_mean: float) -> dict:
    """One-sample t-test: sample mean vs pop_mean."""
    n = len(sample)
    if n < 3:
        return {"n": n, "mean": float(np.nanmean(sample)), "t_stat": np.nan, "p_value": np.nan}
    mean = float(np.nanmean(sample))
    se = float(np.nanstd(sample, ddof=1) / np.sqrt(n))
    t_stat = (mean - pop_mean) / se if se > 0 else np.nan
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan
    return {"n": n, "mean": mean, "se": se, "t_stat": t_stat, "p_value": p_value}


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test between two independent samples."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"n_a": len(a), "n_b": len(b), "t_stat": np.nan, "p_value": np.nan}
    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


# ─── Test 1: regime conditional returns ─────────────────────────

def test1_regime_returns() -> dict:
    """FSM regime conditional forward return distributions."""
    print("[Test 1] Loading state cache + panel ...")
    states = _load_states_pd()
    panel = _load_panel_pd()

    states["dt"] = pd.to_datetime(states["dt"])
    panel["dt"] = pd.to_datetime(panel["dt"])

    merged = panel.merge(states, on=["symbol", "dt"], how="inner")
    merged.sort_values(["symbol", "dt"], inplace=True)

    for w in FWD_WINDOWS:
        merged[f"fwd_{w}d"] = merged.groupby("symbol")["close"].transform(
            lambda s: s.shift(-w) / s - 1
        )

    pop_means = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        pop_means[col] = float(merged[col].mean())

    regime_results = {}
    for regime in range(11):
        sub = merged[merged["regime"] == regime]
        regime_key = f"{regime}_{REGIME_NAMES[regime]}"
        fwd_tests = {}
        for w in FWD_WINDOWS:
            col = f"fwd_{w}d"
            vals = sub[col].dropna().values
            fwd_tests[col] = _ttest_vs_pop(vals, pop_means[col])
        regime_results[regime_key] = {"count": len(sub), "fwd_tests": fwd_tests}

    transition = merged.copy()
    transition["regime_next"] = transition.groupby("symbol")["regime"].shift(-1)
    transition = transition.dropna(subset=["regime_next"])
    transition["regime_next"] = transition["regime_next"].astype(int)

    trans_matrix = {}
    for r_from in range(11):
        sub = transition[transition["regime"] == r_from]
        total = len(sub)
        if total == 0:
            continue
        row = {}
        for r_to in range(11):
            row[str(r_to)] = float((sub["regime_next"] == r_to).sum() / total)
        trans_matrix[str(r_from)] = row

    monotonicity = {}
    for w in FWD_WINDOWS:
        col = f"fwd_{w}d"
        uptrend = merged[merged["regime"].isin([7, 8])][col].dropna().values
        downtrend = merged[merged["regime"].isin([1, 4])][col].dropna().values
        monotonicity[col] = _welch_ttest(uptrend, downtrend)

    result = {
        "pop_means": pop_means,
        "regime_results": regime_results,
        "transition_matrix": trans_matrix,
        "monotonicity_uptrend_vs_downtrend": monotonicity,
    }

    out_path = OUTPUT_DIR / "test1_regime_returns.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Test 1] Saved to {out_path}")
    return result


# ─── Test 2: gate incremental predictive power ──────────────────

def test2_gate_incremental() -> dict:
    """Gate conditions' incremental power after controlling for FSM regime."""
    print("[Test 2] Loading candidates ...")
    cands = _load_candidates_pd()

    sub = cands[(cands["delay"] == 0) & (cands["mode"] == "confirm")].copy()
    uptrend = sub[sub["dec_regime"].isin([7, 8])].copy()

    print(f"[Test 2] Uptrend candidates (regime 7/8, delay=0, confirm): {len(uptrend):,}")

    results = {}

    for feat, n_bins in [("sig_vol_ratio", 5), ("sig_ma_spread_pct", 5)]:
        valid = uptrend[uptrend[feat].notna()].copy()
        if len(valid) < 50:
            results[feat] = {"error": "insufficient data"}
            continue
        valid["quintile"] = pd.qcut(valid[feat], n_bins, labels=False, duplicates="drop")
        bins = {}
        for q in sorted(valid["quintile"].unique()):
            q_data = valid[valid["quintile"] == q]["ret_gross_pct"].dropna().values
            bins[f"Q{int(q)}"] = _ttest_vs_pop(q_data, float(valid["ret_gross_pct"].mean()))
        results[feat] = {"bins": bins, "total_n": len(valid)}

    for feat in ["sig_above_zg"]:
        valid = uptrend[uptrend[feat].notna()].copy()
        if len(valid) < 50:
            results[feat] = {"error": "insufficient data"}
            continue
        pop_mean = float(valid["ret_gross_pct"].mean())
        groups = {}
        for val in sorted(valid[feat].unique()):
            g_data = valid[valid[feat] == val]["ret_gross_pct"].dropna().values
            groups[f"val={val}"] = _ttest_vs_pop(g_data, pop_mean)
        results[feat] = {"groups": groups, "total_n": len(valid)}

    gate_pass = uptrend[
        (uptrend["sig_vol_ratio"] >= 1.2)
        & (uptrend["sig_ma_spread_pct"] >= 3.0)
        & (uptrend["sig_above_zg"] == 1.0)
    ]
    gate_fail = uptrend[
        ~(
            (uptrend["sig_vol_ratio"] >= 1.2)
            & (uptrend["sig_ma_spread_pct"] >= 3.0)
            & (uptrend["sig_above_zg"] == 1.0)
        )
    ]
    results["gate_pass_vs_fail"] = _welch_ttest(
        gate_pass["ret_gross_pct"].dropna().values,
        gate_fail["ret_gross_pct"].dropna().values,
    )

    out_path = OUTPUT_DIR / "test2_gate_incremental.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Test 2] Saved to {out_path}")
    return results


# ─── Test 3: event study (CAR) ──────────────────────────────────

def test3_event_study() -> dict:
    """CAR event study around surge_onset trigger, [-20, +40] window."""
    print("[Test 3] Loading data ...")
    cands = _load_candidates_pd()
    panel = _load_panel_pd()
    panel["dt"] = pd.to_datetime(panel["dt"])
    cands["sig_dt"] = pd.to_datetime(cands["sig_dt"])

    events = cands[(cands["delay"] == 0)].drop_duplicates(subset=["symbol", "sig_dt", "mode"])

    panel.sort_values(["symbol", "dt"], inplace=True)
    panel["daily_ret"] = panel.groupby("symbol")["close"].transform(lambda s: s.pct_change())

    mkt_ret_series = panel.groupby("dt")["daily_ret"].mean()
    panel = panel.merge(mkt_ret_series.rename("mkt_ret"), left_on="dt", right_index=True, how="left")
    panel["ar"] = panel["daily_ret"].fillna(0) - panel["mkt_ret"].fillna(0)

    panel["_row_idx"] = np.arange(len(panel))
    sym_groups = panel.groupby("symbol")
    sym_dt_map: dict[str, dict] = {}
    sym_ar_map: dict[str, np.ndarray] = {}
    for sym, grp in sym_groups:
        dt_arr = grp["dt"].values
        ar_arr = grp["ar"].values
        sym_dt_map[sym] = {d: i for i, d in enumerate(dt_arr)}
        sym_ar_map[sym] = ar_arr

    PRE, POST = 20, 40
    WINDOW_LEN = PRE + POST + 1

    car_by_mode = {}
    for mode in ["confirm", "anticipate"]:
        mode_events = events[events["mode"] == mode]
        syms = mode_events["symbol"].values
        sig_dts = mode_events["sig_dt"].values

        car_matrix = []
        for i in range(len(syms)):
            sym = syms[i]
            sdt = sig_dts[i]
            dt_map = sym_dt_map.get(sym)
            if dt_map is None:
                continue
            anchor = dt_map.get(sdt)
            if anchor is None:
                continue
            ar_arr = sym_ar_map[sym]
            if anchor - PRE < 0 or anchor + POST >= len(ar_arr):
                continue
            window_ar = ar_arr[anchor - PRE : anchor + POST + 1]
            car_matrix.append(np.cumsum(window_ar))

        if not car_matrix:
            car_by_mode[mode] = {"error": "no valid events"}
            continue

        car_arr = np.array(car_matrix)
        mean_car = np.nanmean(car_arr, axis=0)
        se_car = np.nanstd(car_arr, axis=0, ddof=1) / np.sqrt(car_arr.shape[0])

        car_by_mode[mode] = {
            "n_events": car_arr.shape[0],
            "offsets": list(range(-PRE, POST + 1)),
            "mean_car": [float(x) for x in mean_car],
            "se_car": [float(x) for x in se_car],
            "car_at_day0": float(mean_car[PRE]),
            "car_at_day1": float(mean_car[PRE + 1]) if PRE + 1 < len(mean_car) else None,
            "car_at_day5": float(mean_car[PRE + 5]) if PRE + 5 < len(mean_car) else None,
            "car_at_day20": float(mean_car[PRE + 20]) if PRE + 20 < len(mean_car) else None,
            "car_at_day40": float(mean_car[PRE + 40]) if PRE + 40 < len(mean_car) else None,
            "pre_event_drift": float(mean_car[PRE] - mean_car[0]),
            "post_event_drift": float(mean_car[-1] - mean_car[PRE]),
        }

    result = {"car_by_mode": car_by_mode}

    out_path = OUTPUT_DIR / "test3_event_study.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Test 3] Saved to {out_path}")
    return result


# ─── Test 4: market state interaction ────────────────────────────

def test4_market_interaction() -> dict:
    """FSM predictive power conditioned on bull/bear/sideways market regimes."""
    print("[Test 4] Loading data ...")
    states = _load_states_pd()
    panel = _load_panel_pd()
    states["dt"] = pd.to_datetime(states["dt"])
    panel["dt"] = pd.to_datetime(panel["dt"])

    merged = panel.merge(states, on=["symbol", "dt"], how="inner")
    merged.sort_values(["symbol", "dt"], inplace=True)

    merged["fwd_5d"] = merged.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1)

    market_index = panel.groupby("dt")["close"].mean().sort_index()
    market_ret60 = market_index.pct_change(60)
    market_regime_map = {}
    for dt, r60 in market_ret60.items():
        if pd.isna(r60):
            market_regime_map[dt] = "unknown"
        elif r60 > 0.10:
            market_regime_map[dt] = "bull"
        elif r60 < -0.10:
            market_regime_map[dt] = "bear"
        else:
            market_regime_map[dt] = "sideways"

    merged["market_state"] = merged["dt"].map(market_regime_map)
    merged = merged[merged["market_state"].isin(["bull", "bear", "sideways"])]

    results = {}

    for mkt_state in ["bull", "bear", "sideways"]:
        mkt_sub = merged[merged["market_state"] == mkt_state]
        pop_mean = float(mkt_sub["fwd_5d"].mean())

        regime_tests = {}
        for regime in [7, 8]:
            r_sub = mkt_sub[mkt_sub["regime"] == regime]
            vals = r_sub["fwd_5d"].dropna().values
            regime_tests[f"regime_{regime}"] = _ttest_vs_pop(vals, pop_mean)
            regime_tests[f"regime_{regime}"]["pop_mean"] = pop_mean

        results[mkt_state] = {
            "total_n": len(mkt_sub),
            "pop_mean_fwd5d": pop_mean,
            "regime_tests": regime_tests,
        }

    for regime in [7, 8]:
        bull_vals = merged[(merged["regime"] == regime) & (merged["market_state"] == "bull")]["fwd_5d"].dropna().values
        bear_vals = merged[(merged["regime"] == regime) & (merged["market_state"] == "bear")]["fwd_5d"].dropna().values
        results[f"regime_{regime}_bull_vs_bear"] = _welch_ttest(bull_vals, bear_vals)

    is_uptrend = merged["regime"].isin([7, 8])
    for mkt in ["bull", "bear", "sideways"]:
        mkt_mask = merged["market_state"] == mkt
        uptrend_vals = merged[is_uptrend & mkt_mask]["fwd_5d"].dropna().values
        other_vals = merged[~is_uptrend & mkt_mask]["fwd_5d"].dropna().values
        results[f"uptrend_vs_others_in_{mkt}"] = _welch_ttest(uptrend_vals, other_vals)

    out_path = OUTPUT_DIR / "test4_market_interaction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Test 4] Saved to {out_path}")
    return results


# ─── Summary ────────────────────────────────────────────────────

def summarize(t1: dict, t2: dict, t3: dict, t4: dict) -> dict:
    """Generate audit summary with key conclusions."""
    verdicts = []

    sig_regimes = []
    for regime_key, data in t1["regime_results"].items():
        fwd5 = data["fwd_tests"].get("fwd_5d", {})
        if fwd5.get("p_value", 1.0) < 0.05 and fwd5.get("n", 0) > 100:
            sig_regimes.append(regime_key)

    if sig_regimes:
        verdicts.append(f"FSM regimes with significant fwd_5d: {sig_regimes}")
    else:
        verdicts.append("No FSM regime shows significant fwd_5d predictive power")

    mono = t1.get("monotonicity_uptrend_vs_downtrend", {}).get("fwd_5d", {})
    if mono.get("p_value", 1.0) < 0.05:
        direction = "uptrend > downtrend" if mono.get("mean_a", 0) > mono.get("mean_b", 0) else "downtrend > uptrend"
        verdicts.append(f"Monotonicity (regime 7/8 vs 1/4): PASS ({direction}), p={mono['p_value']:.4f}")
    else:
        verdicts.append(f"Monotonicity (regime 7/8 vs 1/4): FAIL, p={mono.get('p_value', 'N/A')}")

    gate = t2.get("gate_pass_vs_fail", {})
    if gate.get("p_value", 1.0) < 0.05:
        verdicts.append(
            f"Gate pass vs fail: significant, mean_pass={gate.get('mean_a', 'N/A'):.2f}%, "
            f"mean_fail={gate.get('mean_b', 'N/A'):.2f}%, p={gate['p_value']:.4f}"
        )
    else:
        verdicts.append(f"Gate conditions: no significant incremental power, p={gate.get('p_value', 'N/A')}")

    for mode in ["confirm", "anticipate"]:
        car_data = t3.get("car_by_mode", {}).get(mode, {})
        pre = car_data.get("pre_event_drift")
        post = car_data.get("post_event_drift")
        if pre is not None and post is not None:
            if pre > 0.01:
                verdicts.append(f"[{mode}] Pre-event CAR drift = {pre*100:.2f}% → signal lags (price-in)")
            if post > 0.01:
                verdicts.append(f"[{mode}] Post-event CAR drift = {post*100:.2f}% → residual alpha")
            elif post <= 0:
                verdicts.append(f"[{mode}] Post-event CAR drift = {post*100:.2f}% → no forward value")

    only_bull = True
    for mkt in ["bear", "sideways"]:
        test = t4.get(f"uptrend_vs_others_in_{mkt}", {})
        if test.get("p_value", 1.0) < 0.05 and test.get("mean_a", 0) > test.get("mean_b", 0):
            only_bull = False
            break
    if only_bull:
        verdicts.append("FSM uptrend outperformance only in bull market → captures beta, not alpha")
    else:
        verdicts.append("FSM uptrend outperforms in non-bull markets → genuine stock-picking ability")

    summary = {"verdicts": verdicts, "timestamp": pd.Timestamp.now().isoformat()}

    out_path = OUTPUT_DIR / "audit_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    for v in verdicts:
        print(f"  • {v}")
    print(f"\nSaved to {out_path}")
    return summary


# ─── main ────────────────────────────────────────────────────────

def _load_or_run_json(path: Path, fn):
    """Load cached JSON result or run function and cache."""
    if path.exists():
        print(f"  [cached] {path.name}")
        with open(path) as f:
            return json.load(f)
    return fn()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    step0_result = step0_build_state_cache()
    print()

    t1 = _load_or_run_json(OUTPUT_DIR / "test1_regime_returns.json", test1_regime_returns)
    print()
    t2 = _load_or_run_json(OUTPUT_DIR / "test2_gate_incremental.json", test2_gate_incremental)
    print()
    t3 = _load_or_run_json(OUTPUT_DIR / "test3_event_study.json", test3_event_study)
    print()
    t4 = _load_or_run_json(OUTPUT_DIR / "test4_market_interaction.json", test4_market_interaction)
    print()

    summarize(t1, t2, t3, t4)


if __name__ == "__main__":
    main()
