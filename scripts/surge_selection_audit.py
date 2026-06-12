"""选股条件价值审计：逐项回答「是否还值得优化」（anticipate + 市场门 + delay5 管线）

对当前实验变体的每个选股条件做 分离力（被剔除人群 vs 保留人群的超额差）与
坡度（a-priori 等距网格上的局部敏感性）检验，输出处置判定：

    锁死保留 / 平台区不值得调 / 值得前推优化（再过 walk-forward 确认）/ 移除候选 / 不可判

预声明判定标准（运行前写定，候选级 OOS 为主、要求 IS 同向；候选级 t 因交易重叠偏高，
仅作筛查，确认靠 walk-forward）：

- 分离力：|保留-剔除| 超额均值差 ≥1.5%（t≥1.5）；
- 坡度：网格最优点比基线 OOS 超额均值 +≥1.5% 且中位数同时改善 且 IS 同向；
- 平台：网格内变动 <1%；
- 反向：剔除人群超额 ≥ 保留人群。

walk-forward 确认（仅触发项）：逐测试年用更早年份选网格值（t 最高且均值>0，
min_train_trades=60，不足退基线），前推聚合 OOS 超额 t≥2 且中位数 ≥ 基线 → 确认。

基线复现校验：全条件=当前取值时，10 槽 OOS 必须复现 n=210 / +7.58% / t≈2.96。

    uv run --no-sync python scripts/surge_selection_audit.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import trend_regime as tr

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_selection_audit"
CACHE_PATH = OUTPUT_DIR / "excess_cache.parquet"

FIRST_TEST_YEAR = 2024
SLOTS = 10
MIN_TRAIN_TRADES = 60

# 基线 = 当前实验变体取值（全部 a-priori，沿用既有）
BASE = {
    "vr": tr.SURGE_GATE_VOL_RATIO,  # 1.2
    "spread": tr.SURGE_GATE_MA_SPREAD,  # 3.0
    "ret20": tr.SURGE_GATE_RET20,  # 8.0
    "above_zg": True,
    "amount": spb.MIN_AMOUNT_E,  # 1.0
    "sl": (8, 20),  # = (spb.STOP_MIN_PCT, spb.STOP_MAX_PCT)，用 int 字面量保证网格标签一致
    "high20": 0.12,
    "index": True,
}

# a-priori 等距网格（禁止跑完后加密找最优）
GRIDS = {
    "C1_vol_ratio": [None, 1.0, 1.2, 1.5, 2.0],
    "C2_ma_spread": [None, 1.5, 3.0, 4.5, 6.0],
    "C3_ret20": [None, 4.0, 8.0, 12.0, 16.0],
    "C4_above_zg": [True, False],
    "C7_amount": [0.5, 1.0, 2.0, 3.0],
    "C8_sl_band": [None, (5, 20), (8, 20), (8, 25), (8, 30), (5, 30)],
    "C9_high20": [None, 0.08, 0.10, 0.12, 0.15, 0.20],
}
SEP_DIFF, SEP_T = 1.5, 1.5  # 分离力阈值
SLOPE_DIFF, PLATEAU_BAND = 1.5, 1.0  # 坡度/平台阈值（候选级超额均值，百分点）


# --------------------------------------------------------------------------- #
# 宇宙构建 + excess 缓存
# --------------------------------------------------------------------------- #
def load_universe() -> pd.DataFrame:
    """anticipate × delay5 全部门控前候选行 + 市场状态 + ST 标记 + 衍生列。"""
    cand = pd.read_parquet(msf.CAND_DIR / "candidates.parquet")
    df = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5)].copy()
    market = pd.read_parquet(msf.OUTPUT_DIR / "market_state.parquet")
    df = df.merge(
        market[["dt", "high20_ratio", "ew_index_above_ma20"]], left_on="dec_dt", right_on="dt", how="left"
    ).drop(columns=["dt"])
    st_intervals = spb.load_st_intervals()
    df["is_st"] = (
        df.apply(lambda r: spb.is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1) if st_intervals else False
    )
    df["gap_ok"] = (df["gap_pct"] < df["limit_pct"] - spb.GAP_LIMIT_MARGIN).fillna(False)
    df["priority"] = [
        tr.priority_score(s, sl, 0, rg) for s, sl, rg in zip(df["score"], df["sl_pct"], df["dec_regime"], strict=False)
    ]
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - spb.SELL_COST) / (1 + spb.BUY_COST) - 1) * 100
    return df.reset_index(drop=True)


def with_excess(df: pd.DataFrame) -> pd.DataFrame:
    """全宇宙单次 excess 计算（StableControlSampler，逐行确定性），缓存复用。"""
    keys = ["symbol", "sig_dt", "dec_dt", "entry_dt", "exit_dt"]
    if CACHE_PATH.exists():
        cache = pd.read_parquet(CACHE_PATH)
        merged = df.merge(cache, on=keys, how="left")
        if merged["excess_pct"].notna().sum() >= len(df) * 0.5:  # 缓存命中（NaN 行本就存在）
            print(f"[excess] 缓存命中 {CACHE_PATH.name}")
            return merged
    sampler = msf.StableControlSampler()
    t0 = time.time()
    out = []
    for i, row in enumerate(df.itertuples(), 1):
        out.append(sampler.excess_for(row))
        if i % 20000 == 0:
            print(f"  [excess] {i}/{len(df)} | {time.time() - t0:.0f}s")
    df = df.assign(excess_pct=out)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df[[*keys, "excess_pct"]].to_parquet(CACHE_PATH, index=False)
    print(f"[excess] 计算完成 {time.time() - t0:.0f}s → 缓存 {CACHE_PATH.name}")
    return df


# --------------------------------------------------------------------------- #
# 条件掩码与统计
# --------------------------------------------------------------------------- #
def cond_mask(df: pd.DataFrame, p: dict) -> pd.Series:
    """给定参数字典 p（同 BASE 结构）返回布尔掩码。None/False 表示该门关闭。"""
    m = ~df["is_st"] & df["gap_ok"]
    if p["vr"] is not None:
        m &= df["sig_vol_ratio"] >= p["vr"]
    if p["spread"] is not None:
        m &= df["sig_ma_spread_pct"] >= p["spread"]
    if p["ret20"] is not None:
        m &= df["sig_ret20"] >= p["ret20"]
    if p["above_zg"]:
        m &= df["sig_above_zg"] == 1
    m &= df["amount_e"] >= p["amount"]
    if p["sl"] is not None:
        m &= df["sl_pct"].between(p["sl"][0], p["sl"][1])
    else:
        m &= df["sl_pct"] > 0
    if p["high20"] is not None:
        m &= df["high20_ratio"] > p["high20"]
    if p["index"]:
        m &= df["ew_index_above_ma20"] > 0
    return m.fillna(False)


def cstats(values: pd.Series) -> dict:
    v = values.dropna()
    if len(v) < 30:
        return {"n": int(len(v))}
    return {
        "n": int(len(v)),
        "mean": msf.round_float(v.mean()),
        "median": msf.round_float(v.median()),
        "t": msf.round_float(msf.t_stat(v)),
    }


def cand_stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    sub = df[mask]
    return {
        "oos": cstats(sub.loc[sub["seg"] == "test", "excess_pct"]),
        "is": cstats(sub.loc[sub["seg"] == "train", "excess_pct"]),
    }


def trade_sim(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = df[mask].sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)
    trades, _ = spb.simulate_slots(sub, SLOTS)
    return trades


def trade_stats(trades: pd.DataFrame, seg: str = "test") -> dict:
    if not len(trades):
        return {"n": 0}
    sub = trades[trades["seg"] == seg]
    out = cstats(sub["excess_pct"])
    if len(sub):
        out["net_mean"] = msf.round_float(sub["ret_net_pct"].mean())
        out["net_median"] = msf.round_float(sub["ret_net_pct"].median())
    return out


def point_eval(df: pd.DataFrame, params: dict, label: str, keep_trades: dict | None = None) -> dict:
    mask = cond_mask(df, params)
    trades = trade_sim(df, mask)
    if keep_trades is not None:
        keep_trades[label] = trades
    cs = cand_stats(df, mask)
    return {
        "point": label,
        "cand_n": int(mask.sum()),
        "cand_oos": cs["oos"],
        "cand_is": cs["is"],
        "trade_oos": trade_stats(trades, "test"),
        "trade_is": trade_stats(trades, "train"),
    }


def flat_rows(points: list[dict]) -> list[dict]:
    rows = []
    for p in points:
        rows.append(
            {
                "point": p["point"],
                "cand_n": p["cand_n"],
                "c_oos_n": p["cand_oos"].get("n"),
                "c_oos_mean": p["cand_oos"].get("mean"),
                "c_oos_med": p["cand_oos"].get("median"),
                "c_is_mean": p["cand_is"].get("mean"),
                "t_oos_n": p["trade_oos"].get("n"),
                "t_oos_mean": p["trade_oos"].get("mean"),
                "t_oos_med": p["trade_oos"].get("median"),
                "t_net_mean": p["trade_oos"].get("net_mean"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 判定引擎（预声明规则）
# --------------------------------------------------------------------------- #
def slope_verdict(points: list[dict], base_label: str) -> dict:
    """网格坡度判定：候选级 OOS 均值/中位数 + IS 同向。"""
    base = next(p for p in points if p["point"] == base_label)
    b_mean = base["cand_oos"].get("mean") or 0.0
    b_med = base["cand_oos"].get("median") or 0.0
    best, best_gain = None, 0.0
    spread_vals = []
    for p in points:
        mean = p["cand_oos"].get("mean")
        if mean is None:
            continue
        spread_vals.append(mean)
        if p["point"] == base_label:
            continue
        gain = mean - b_mean
        med_gain = (p["cand_oos"].get("median") or -99) - b_med
        is_gain = (p["cand_is"].get("mean") or -99) - (base["cand_is"].get("mean") or 0)
        if gain >= SLOPE_DIFF and med_gain > 0 and is_gain > 0 and gain > best_gain:
            best, best_gain = p["point"], gain
    grid_range = (max(spread_vals) - min(spread_vals)) if spread_vals else 0.0
    if best:
        return {"verdict": "值得前推优化", "best_point": best, "gain": msf.round_float(best_gain)}
    if grid_range < PLATEAU_BAND:
        return {"verdict": "平台区不值得调", "grid_range": msf.round_float(grid_range)}
    return {"verdict": "锁死保留（基线即高地）", "grid_range": msf.round_float(grid_range)}


def separation(df: pd.DataFrame, base_mask: pd.Series, relax_mask: pd.Series) -> dict:
    """分离力：基线保留 vs 仅被该条件剔除（放松该条件后多出来的人群）。"""
    kept = df[base_mask & (df["seg"] == "test")]["excess_pct"].dropna()
    removed = df[relax_mask & ~base_mask & (df["seg"] == "test")]["excess_pct"].dropna()
    if len(removed) < 30 or len(kept) < 30:
        return {"removed_n": int(len(removed)), "note": "样本不足"}
    diff = kept.mean() - removed.mean()
    pooled_t = msf.t_stat(removed) if len(removed) else np.nan
    out = {
        "kept_n": int(len(kept)),
        "kept_mean": msf.round_float(kept.mean()),
        "removed_n": int(len(removed)),
        "removed_mean": msf.round_float(removed.mean()),
        "removed_median": msf.round_float(removed.median()),
        "removed_t": msf.round_float(pooled_t),
        "diff": msf.round_float(diff),
    }
    if removed.mean() >= kept.mean():
        out["separation"] = "反向（剔除的人群不差）→ 移除候选"
    elif diff >= SEP_DIFF:
        out["separation"] = "有分离力（条件在挡刀）"
    else:
        out["separation"] = "弱分离"
    return out


# --------------------------------------------------------------------------- #
# Walk-forward 前推确认（触发项）
# --------------------------------------------------------------------------- #
def walk_forward_grid(trades_by_point: dict[str, pd.DataFrame], base_label: str) -> dict:
    years = sorted({int(y) for t in trades_by_point.values() if len(t) for y in t["year"].unique()})
    folds, sel_trades = [], []
    for ty in [y for y in years if y >= FIRST_TEST_YEAR]:
        train_years = [y for y in years if y < ty]
        if len(train_years) < 2:
            continue
        best_label, best_score = base_label, (-np.inf, -np.inf)
        for label, trades in trades_by_point.items():
            tr_seg = trades[trades["year"].isin(train_years)]["excess_pct"].dropna()
            if len(tr_seg) < MIN_TRAIN_TRADES:
                continue
            t_stat, mean = msf.t_stat(tr_seg), tr_seg.mean()
            if mean > 0 and (t_stat, mean) > best_score:
                best_label, best_score = label, (t_stat, mean)
        test = trades_by_point[best_label]
        test = test[test["year"] == ty]
        sel_trades.append(test)
        folds.append({"year": ty, "selected": best_label, "test_n": int(len(test))})
    agg = pd.concat(sel_trades, ignore_index=True) if sel_trades else pd.DataFrame()
    base_trades = trades_by_point[base_label]
    base_oos = base_trades[base_trades["year"] >= FIRST_TEST_YEAR]
    agg_stats = cstats(agg["excess_pct"]) if len(agg) else {"n": 0}
    base_stats = cstats(base_oos["excess_pct"]) if len(base_oos) else {"n": 0}
    confirmed = (
        (agg_stats.get("mean") or 0) > 0
        and (agg_stats.get("t") or 0) >= 2
        and (agg_stats.get("median") or -99) >= (base_stats.get("median") or 0)
    )
    return {"folds": folds, "selected_agg": agg_stats, "baseline_agg": base_stats, "confirmed": bool(confirmed)}


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-walkforward", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[宇宙] anticipate × delay5 门控前候选 + 市场状态 + ST")
    df = load_universe()
    print(f"  {len(df)} 行 | OOS(≥{FIRST_TEST_YEAR}) {int((df['year'] >= FIRST_TEST_YEAR).sum())} 行")
    df = with_excess(df)

    summary: dict = {"base": {k: str(v) for k, v in BASE.items()}, "sections": {}}
    wf_trades: dict[str, dict[str, pd.DataFrame]] = {}

    # —— 基线复现校验 ——
    base_point = point_eval(df, BASE, "baseline")
    print(f"[基线] {base_point['trade_oos']}")
    expect = {"n": 210, "mean": 7.58}
    got = base_point["trade_oos"]
    if got.get("n") != expect["n"] or abs((got.get("mean") or 0) - expect["mean"]) > 0.02:
        print(f"  !! 基线未复现既知数 {expect}，检查脚本/数据后再解读")
    summary["baseline_check"] = {"expect": expect, "got": got}

    # —— A1-A4 / C7-C9：单变量网格 ——
    sweeps = [
        ("C1_vol_ratio", "vr", GRIDS["C1_vol_ratio"], 1.2),
        ("C2_ma_spread", "spread", GRIDS["C2_ma_spread"], 3.0),
        ("C3_ret20", "ret20", GRIDS["C3_ret20"], 8.0),
        ("C4_above_zg", "above_zg", GRIDS["C4_above_zg"], True),
        ("C7_amount", "amount", GRIDS["C7_amount"], 1.0),
        ("C8_sl_band", "sl", GRIDS["C8_sl_band"], (8, 20)),
        ("C9_high20", "high20", GRIDS["C9_high20"], 0.12),
    ]
    for name, key, grid, base_val in sweeps:
        print(f"[{name}] 网格 {len(grid)} 点")
        points, trades_map = [], {}
        for val in grid:
            params = {**BASE, key: val}
            if name == "C9_high20" and val is None:
                params["index"] = False  # 市场门整体关闭点
            label = f"{key}={val}" if not isinstance(val, tuple) else f"{key}={val[0]}-{val[1]}"
            points.append(point_eval(df, params, label, trades_map))
        base_label = f"{key}={base_val}" if not isinstance(base_val, tuple) else f"{key}={base_val[0]}-{base_val[1]}"
        verdict = slope_verdict(points, base_label)
        # 分离力：放松点 = 网格中最宽松取值
        relax_val = None if None in grid or False in grid else min(g for g in grid if g is not None)
        relax_params = {**BASE, key: relax_val}
        if name == "C9_high20":
            relax_params["index"] = False
        sep = separation(df, cond_mask(df, BASE), cond_mask(df, relax_params))
        summary["sections"][name] = {"points": flat_rows(points), "slope": verdict, "separation": sep}
        wf_trades[name] = trades_map
        print(f"  → {verdict} | 分离: {sep.get('separation', sep.get('note'))}")

    # —— A5：决策日状态质量 / 排序价值（候选级，基线人群内）——
    print("[A5] dec_regime / score / priority 分位")
    base_mask = cond_mask(df, BASE)
    sub = df[base_mask & (df["seg"] == "test")].copy()
    regime_rows = [
        {"group": f"regime={rg}", **cstats(g["excess_pct"]), "net_mean": msf.round_float(g["ret_net_pct"].mean())}
        for rg, g in sub.groupby("dec_regime")
    ]
    quint_rows = []
    for col in ["score", "priority"]:
        sub["q"] = pd.qcut(sub[col], 5, duplicates="drop", labels=False)
        for q, g in sub.groupby("q"):
            quint_rows.append({"group": f"{col}_q{int(q) + 1}", **cstats(g["excess_pct"])})
    summary["sections"]["A5_quality"] = {"regime": regime_rows, "quintiles": quint_rows}

    # —— A6：绑定瀑布（每级剔除人群的 OOS 超额）——
    print("[A6] 绑定瀑布")
    stages = [
        (
            "S0_全宇宙(存活到第5日)",
            {
                **BASE,
                "vr": None,
                "spread": None,
                "ret20": None,
                "above_zg": False,
                "amount": 0,
                "sl": None,
                "high20": None,
                "index": False,
            },
        ),
        ("S1_+信号门", {**BASE, "amount": 0, "sl": None, "high20": None, "index": False}),
        ("S2_+成交额≥1亿", {**BASE, "sl": None, "high20": None, "index": False}),
        ("S3_+止损带8-20", {**BASE, "high20": None, "index": False}),
        ("S4_+市场门", BASE),
    ]
    waterfall = []
    prev_mask = None
    for label, params in stages:
        m = cond_mask(df, params)
        row = {
            "stage": label,
            "n": int(m.sum()),
            **{f"oos_{k}": v for k, v in cstats(df[m & (df["seg"] == "test")]["excess_pct"]).items()},
        }
        if prev_mask is not None:
            removed = df[prev_mask & ~m & (df["seg"] == "test")]["excess_pct"].dropna()
            row["removed_n"] = int(len(removed))
            row["removed_mean"] = msf.round_float(removed.mean()) if len(removed) >= 30 else None
        waterfall.append(row)
        prev_mask = m
    summary["sections"]["A6_waterfall"] = waterfall

    # —— Phase B：触发项 walk-forward ——
    if not args.skip_walkforward:
        print("[B] walk-forward 前推确认（触发项）")
        wf_results = {}
        for name, sec in summary["sections"].items():
            if not isinstance(sec, dict) or "slope" not in sec:
                continue
            v = sec["slope"]["verdict"]
            s = (sec.get("separation") or {}).get("separation", "")
            if v == "值得前推优化" or "移除候选" in s:
                base_key = name.split("_")[0]
                key = {
                    "C1": "vr",
                    "C2": "spread",
                    "C3": "ret20",
                    "C4": "above_zg",
                    "C7": "amount",
                    "C8": "sl",
                    "C9": "high20",
                }[base_key]
                bv = BASE[key]
                base_label = f"{key}={bv}" if not isinstance(bv, tuple) else f"{key}={bv[0]}-{bv[1]}"
                wf_results[name] = walk_forward_grid(wf_trades[name], base_label)
                print(f"  {name}: confirmed={wf_results[name]['confirmed']} | {wf_results[name]['selected_agg']}")
        summary["walk_forward"] = wf_results

    # —— 报告 ——
    write_report(summary)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


def write_report(summary: dict) -> None:
    cols = [
        "point",
        "cand_n",
        "c_oos_n",
        "c_oos_mean",
        "c_oos_med",
        "c_is_mean",
        "t_oos_n",
        "t_oos_mean",
        "t_oos_med",
        "t_net_mean",
    ]
    lines = ["# Selection-condition audit (anticipate + market gate + delay5)", ""]
    lines.append(f"Baseline check: {json.dumps(summary['baseline_check'], ensure_ascii=False)}")
    lines.append("")
    lines.append("## Verdicts")
    vrows = []
    for name, sec in summary["sections"].items():
        if isinstance(sec, dict) and "slope" in sec:
            vrows.append(
                {
                    "condition": name,
                    "slope_verdict": sec["slope"]["verdict"],
                    "best_point": sec["slope"].get("best_point", ""),
                    "separation": (sec.get("separation") or {}).get(
                        "separation", (sec.get("separation") or {}).get("note", "")
                    ),
                    "removed_mean": (sec.get("separation") or {}).get("removed_mean", ""),
                }
            )
    lines.extend(msf.markdown_table(vrows, ["condition", "slope_verdict", "best_point", "separation", "removed_mean"]))
    for name, sec in summary["sections"].items():
        lines.extend(["", f"## {name}"])
        if isinstance(sec, dict) and "points" in sec:
            lines.extend(msf.markdown_table(sec["points"], cols))
            lines.append("")
            lines.append(f"slope: `{json.dumps(sec['slope'], ensure_ascii=False)}`")
            lines.append(f"separation: `{json.dumps(sec['separation'], ensure_ascii=False)}`")
        elif name == "A5_quality":
            lines.extend(msf.markdown_table(sec["regime"], ["group", "n", "mean", "median", "t", "net_mean"]))
            lines.append("")
            lines.extend(msf.markdown_table(sec["quintiles"], ["group", "n", "mean", "median", "t"]))
        elif name == "A6_waterfall":
            lines.extend(
                msf.markdown_table(sec, ["stage", "n", "oos_n", "oos_mean", "oos_median", "removed_n", "removed_mean"])
            )
    if summary.get("walk_forward"):
        lines.extend(
            [
                "",
                "## Walk-forward confirmations",
                "",
                "```json",
                json.dumps(summary["walk_forward"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
