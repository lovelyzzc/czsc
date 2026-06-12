"""因子判别力研究 + 交易级对决（Phase C+D）—— 强势主升能否被 ex-ante 筛出？

输入：surge_factor_dump 的 bars.parquet + selection_audit 的 delay5 宇宙/excess 缓存。

人群与目标（预声明）：
- P1（策略内）= delay5 基线人群，y = excess 年内 pct-rank；尾部标签 = t1_px30
  （未来 40 根最大涨幅 ≥30%，价格口径）；
- P2（广义）= 全市场 regime∈{4,5,6} 且不在 FSM 主升途中的 bar，
  目标 = 市场调整前向 20 根收益的月内 rank 与 t1_px30；
- 单因子有效：月度 FM rank-IC，OOS |t|≥2 且与 IS 同号（结论用）；
- 合成因果性：composite 的因子选择与方向**只用 IS（≤2023）**（|IS t|≥2），
  在 OOS 上评估尾部捕获 lift（判定阈 1.5×）与 P2 广义筛选；
- Phase D walk-forward：逐测试年用更早年份重选因子集，composite 排序 vs priority 排序
  的槽位镜像对决；composite 胜出标准 = OOS 超额均值与中位数均 ≥ priority；
- LR 上限：逐年 walk-forward 拟合 L2 逻辑回归报 AUC，仅作可筛性上限参考，不入策略。

    uv run --no-sync python scripts/surge_factor_research.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import surge_selection_audit as ssa

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_factors"
MIN_MONTH_ROWS = 10
IS_T_GATE = 2.0  # composite 因子入选门槛（仅用 IS 月度 IC t）
FIRST_TEST_YEAR = 2024

BAR_FACTORS = [
    "up_dn_power_ratio",
    "last_up_angle",
    "ma_spread_pct",
    "vol_ratio",
    "ret20",
    "n_pivots",
    "pivot_width_pct",
    "ret5",
    "ret60",
    "dist_hi250",
    "amp10",
    "amt_ratio20",
    "dif_norm",
    "breakout_pct",
    "sl_dist",
]
P1_EXTRA = ["rel_ret20", "path_volchg"]  # P1 专属（决策日相对强度 / 信号→决策量能变化）


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_bars() -> pd.DataFrame:
    bars = pd.read_parquet(OUTPUT_DIR / "bars.parquet")
    market = pd.read_parquet(msf.OUTPUT_DIR / "market_state.parquet", columns=["dt", "ew_index", "mkt_ret20_median"])
    market = market.sort_values("dt").reset_index(drop=True)
    market["ew_fwd20"] = (market["ew_index"].shift(-20) / market["ew_index"] - 1) * 100
    return bars.merge(market[["dt", "ew_fwd20", "mkt_ret20_median"]], on="dt", how="left")


def build_p1(bars: pd.DataFrame) -> pd.DataFrame:
    uni = ssa.load_universe()
    uni = ssa.with_excess(uni)
    p1 = uni[ssa.cond_mask(uni, ssa.BASE)].copy()
    dec_cols = ["symbol", "dt", "t1_px30", "fwd40_max", *BAR_FACTORS, "mkt_ret20_median"]
    p1 = p1.merge(bars[dec_cols].rename(columns={"dt": "dec_dt"}), on=["symbol", "dec_dt"], how="left")
    sig_cols = ["symbol", "dt", "vol_ratio"]
    p1 = p1.merge(
        bars[sig_cols].rename(columns={"dt": "sig_dt", "vol_ratio": "vol_ratio_sig"}),
        on=["symbol", "sig_dt"],
        how="left",
    )
    p1["rel_ret20"] = p1["ret20"] - p1["mkt_ret20_median"] * 100
    p1["path_volchg"] = p1["vol_ratio"] / p1["vol_ratio_sig"]
    p1["month"] = p1["dec_dt"].dt.to_period("M")
    p1 = p1[p1["excess_pct"].notna()].copy()
    p1["y"] = p1.groupby("year")["excess_pct"].rank(pct=True)
    return p1


def build_p2(bars: pd.DataFrame) -> pd.DataFrame:
    p2 = bars[bars["regime"].isin([4, 5, 6]) & (bars["in_surge"] == 0) & bars["t1_px30"].notna()].copy()
    p2["adj_fwd20"] = p2["fwd20"] - p2["ew_fwd20"]
    p2["month"] = p2["dt"].dt.to_period("M")
    p2["y"] = p2.groupby("month")["adj_fwd20"].rank(pct=True)
    return p2


# --------------------------------------------------------------------------- #
# FM-IC 机件
# --------------------------------------------------------------------------- #
def monthly_ic(df: pd.DataFrame, factor: str, target: str = "y") -> pd.Series:
    ics = {}
    for m, g in df.groupby("month"):
        sub = g[[factor, target]].dropna()
        if len(sub) < MIN_MONTH_ROWS:
            continue
        ics[m] = sub[factor].corr(sub[target], method="spearman")
    return pd.Series(ics).dropna()


def split_ic(ics: pd.Series) -> tuple[dict, dict]:
    is_ics = ics[ics.index.map(lambda p: p.year < FIRST_TEST_YEAR)]
    oos_ics = ics[ics.index.map(lambda p: p.year >= FIRST_TEST_YEAR)]

    def stats(s):
        if len(s) < 4:
            return {"months": int(len(s))}
        return {
            "months": int(len(s)),
            "ic": msf.round_float(s.mean(), 3),
            "t": msf.round_float(msf.t_stat(s)),
            "pos%": msf.round_float((s > 0).mean() * 100, 1),
        }

    return stats(is_ics), stats(oos_ics)


def factor_table(df: pd.DataFrame, factors: list[str], target: str = "y") -> list[dict]:
    rows = []
    for f in factors:
        if f not in df.columns:
            continue
        s_is, s_oos = split_ic(monthly_ic(df, f, target))
        same = (
            s_is.get("ic") is not None and s_oos.get("ic") is not None and np.sign(s_is["ic"]) == np.sign(s_oos["ic"])
        )
        rows.append(
            {
                "factor": f,
                "is_ic": s_is.get("ic"),
                "is_t": s_is.get("t"),
                "oos_ic": s_oos.get("ic"),
                "oos_t": s_oos.get("t"),
                "oos_pos%": s_oos.get("pos%"),
                "valid": bool(same and abs(s_oos.get("t") or 0) >= 2),
                "is_selected": bool(abs(s_is.get("t") or 0) >= IS_T_GATE),
                "is_sign": int(np.sign(s_is["ic"])) if s_is.get("ic") else 0,
            }
        )
    return rows


def composite_score(df: pd.DataFrame, selected: list[tuple[str, int]]) -> pd.Series:
    """月内 pct-rank 等权合成（方向 = 训练段 IC 符号）。"""
    if not selected:
        return pd.Series(np.nan, index=df.index)
    parts = []
    for f, sign in selected:
        r = df.groupby("month")[f].rank(pct=True)
        parts.append(sign * r)
    return pd.concat(parts, axis=1).mean(axis=1)


def capture_lift(df: pd.DataFrame, score_col: str, label_col: str, top_q: float = 0.8) -> dict:
    sub = df[df[score_col].notna() & df[label_col].notna()].copy()
    if len(sub) < 100:
        return {"n": int(len(sub))}
    sub["fq"] = sub.groupby("month")[score_col].rank(pct=True)
    top = sub[sub["fq"] >= top_q]
    base = sub[label_col].mean()
    out = {
        "n": int(len(sub)),
        "top_n": int(len(top)),
        "base%": msf.round_float(base * 100, 1),
        "top%": msf.round_float(top[label_col].mean() * 100, 1),
        "lift": msf.round_float(top[label_col].mean() / base, 2) if base > 0 else None,
    }
    return out


def double_sort(df: pd.DataFrame, factor: str, control: str = "score") -> dict:
    """控制 score 五分位后，因子上下半区的 y 差（月内分组），验真增量。"""
    sub = df[[factor, control, "y", "month"]].dropna().copy()
    if len(sub) < 300:
        return {"n": int(len(sub))}
    sub["cq"] = sub.groupby("month")[control].rank(pct=True).mul(5).clip(upper=4.999).astype(int)
    sub["fh"] = sub.groupby(["month", "cq"])[factor].rank(pct=True) >= 0.5
    spread = sub[sub["fh"]]["y"].mean() - sub[~sub["fh"]]["y"].mean()
    return {"n": int(len(sub)), "y_spread_ctrl_score": msf.round_float(spread, 4)}


# --------------------------------------------------------------------------- #
# Phase D：composite 排序 vs priority 排序（槽位镜像 walk-forward）
# --------------------------------------------------------------------------- #
def order_duel(p1: pd.DataFrame, factors: list[str]) -> dict:
    ics_by_factor = {f: monthly_ic(p1, f) for f in factors if f in p1.columns}
    years = sorted(p1["year"].unique())
    folds, duel_trades = [], {"composite": [], "priority": []}
    for ty in [y for y in years if y >= FIRST_TEST_YEAR]:
        selected = []
        for f, ics in ics_by_factor.items():
            tm = ics[ics.index.map(lambda p, _ty=ty: p.year < _ty)]
            if len(tm) >= 12 and abs(msf.t_stat(tm) or 0) >= IS_T_GATE:
                selected.append((f, int(np.sign(tm.mean()))))
        comp = composite_score(p1, selected)
        for name, ordering in [("composite", comp), ("priority", p1["priority"])]:
            sub = p1.assign(_ord=ordering).copy()
            sub = sub[sub["_ord"].notna()]
            sub = sub.sort_values(["entry_dt", "_ord"], ascending=[True, False]).reset_index(drop=True)
            sub["priority"] = sub["_ord"]  # simulate_slots 后续统计不依赖 priority 列，仅排序用
            trades, _ = spb.simulate_slots(sub, 10)
            year_trades = trades[trades["year"] == ty]
            duel_trades[name].append(year_trades)
        folds.append({"year": ty, "n_factors": len(selected), "factors": [f for f, _ in selected]})

    out = {"folds": folds}
    for name, parts in duel_trades.items():
        agg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        v = agg["excess_pct"].dropna() if len(agg) else pd.Series(dtype=float)
        out[name] = {
            "n": int(len(v)),
            "excess_mean": msf.round_float(v.mean()) if len(v) else None,
            "excess_median": msf.round_float(v.median()) if len(v) else None,
            "t": msf.round_float(msf.t_stat(v)) if len(v) else None,
            "net_mean": msf.round_float(agg["ret_net_pct"].mean()) if len(agg) else None,
        }
    c, p = out["composite"], out["priority"]
    out["composite_wins"] = bool(
        c.get("excess_mean") is not None
        and p.get("excess_mean") is not None
        and c["excess_mean"] >= p["excess_mean"]
        and (c.get("excess_median") or -99) >= (p.get("excess_median") or 0)
    )
    return out


# --------------------------------------------------------------------------- #
# LR 上限（围栏：逐年 walk-forward 拟合，仅作参考）
# --------------------------------------------------------------------------- #
def lr_upper_bound(p1: pd.DataFrame, factors: list[str], label: str = "t1_px30") -> list[dict]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return [{"note": "sklearn 不可用，跳过"}]
    cols = [f for f in factors if f in p1.columns]
    df = p1[[*cols, label, "year"]].copy()
    df = df[df[label].notna()]
    df[cols] = df[cols].fillna(df[cols].median())
    rows = []
    rng = np.random.default_rng(42)
    for ty in sorted(df.loc[df["year"] >= FIRST_TEST_YEAR, "year"].unique()):
        train = df[df["year"] < ty]
        if len(train) > 300_000:  # P2 大样本时子采样，控制拟合耗时
            train = train.iloc[rng.choice(len(train), 300_000, replace=False)]
        test = df[df["year"] == ty]
        if train[label].nunique() < 2 or test[label].nunique() < 2 or len(train) < 300:
            continue
        scaler = StandardScaler().fit(train[cols])
        model = LogisticRegression(max_iter=2000, C=0.1).fit(scaler.transform(train[cols]), train[label])
        auc = roc_auc_score(test[label], model.predict_proba(scaler.transform(test[cols]))[:, 1])
        rows.append(
            {"test_year": int(ty), "train_n": int(len(train)), "test_n": int(len(test)), "auc": msf.round_float(auc, 3)}
        )
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    print("[载入] bars + P1 + P2")
    bars = load_bars()
    p1 = build_p1(bars)
    p2 = build_p2(bars)
    join_miss = int(p1["ret5"].isna().sum())
    print(f"  bars {len(bars)} | P1 {len(p1)}（dec join 缺失 {join_miss}）| P2 {len(p2)}")

    all_p1_factors = BAR_FACTORS + P1_EXTRA
    print("[P1] 单因子月度 FM-IC（y=年内 rank 超额）")
    p1_rows = factor_table(p1, all_p1_factors)
    for r in p1_rows:
        print(f"  {r['factor']:<20} IS={r['is_ic']}({r['is_t']}) OOS={r['oos_ic']}({r['oos_t']}) valid={r['valid']}")

    print("[P1] 对 t1_px30 标签的 FM-IC")
    p1_t1_rows = factor_table(p1.assign(y=p1["t1_px30"]), all_p1_factors)

    valid_factors = [r["factor"] for r in p1_rows if r["valid"]]
    ds_rows = [{"factor": f, **double_sort(p1, f)} for f in valid_factors]

    # composite（因果：仅 IS 选择）
    is_selected = [(r["factor"], r["is_sign"]) for r in p1_rows if r["is_selected"] and r["is_sign"] != 0]
    print(f"[composite] IS 入选因子 {len(is_selected)}: {[f for f, _ in is_selected]}")
    p1["comp"] = composite_score(p1, is_selected)
    oos_p1 = p1[p1["seg"] == "test"]
    comp_tail = capture_lift(oos_p1, "comp", "t1_px30")
    comp_tail_excess = capture_lift(oos_p1.assign(tail=(oos_p1["y"] >= 0.8).astype(int)), "comp", "tail")
    comp_year = {}
    for y, g in oos_p1.groupby("year"):
        comp_year[str(int(y))] = capture_lift(g, "comp", "t1_px30")

    # P2 广义筛选
    print("[P2] 广义筛选（{4,5,6} 非在途）")
    p2_rows = factor_table(p2, BAR_FACTORS)  # 目标 = 月内 rank 调整 fwd20
    p2_is_selected = []
    for r in p2_rows:
        if r["is_selected"] and r["is_sign"] != 0:
            p2_is_selected.append((r["factor"], r["is_sign"]))
    p2["comp"] = composite_score(p2, p2_is_selected)
    p2_oos = p2[p2["year"] >= FIRST_TEST_YEAR]
    p2_capture = capture_lift(p2_oos, "comp", "t1_px30", top_q=0.9)
    onset_sub = p2_oos[p2_oos["is_anticipate_onset"] == 1]
    p2_onset = {
        "n": int(len(onset_sub)),
        "t1_rate%": msf.round_float(onset_sub["t1_px30"].mean() * 100, 1) if len(onset_sub) else None,
        "base%": msf.round_float(p2_oos["t1_px30"].mean() * 100, 1),
    }

    print("[D] composite vs priority 槽位对决（walk-forward 因子选择）")
    duel = order_duel(p1, all_p1_factors)
    print(f"  composite={duel['composite']} | priority={duel['priority']} | wins={duel['composite_wins']}")

    print("[LR] 可筛性上限（仅参考）")
    lr_rows = lr_upper_bound(p1, all_p1_factors)
    lr_p2_rows = lr_upper_bound(p2.assign(seg="test"), BAR_FACTORS) if len(p2) else []

    summary = {
        "p1_n": len(p1),
        "p2_n": len(p2),
        "p1_factor_ic": p1_rows,
        "p1_factor_ic_t1": p1_t1_rows,
        "double_sort_valid": ds_rows,
        "composite_is_factors": [f for f, _ in is_selected],
        "composite_tail_t1_oos": comp_tail,
        "composite_tail_excess_oos": comp_tail_excess,
        "composite_tail_t1_by_year": comp_year,
        "p2_factor_ic": p2_rows,
        "p2_composite_capture_top10": p2_capture,
        "p2_anticipate_onset_capture": p2_onset,
        "order_duel": duel,
        "lr_upper_bound_p1": lr_rows,
        "lr_upper_bound_p2": lr_p2_rows,
    }
    with (OUTPUT_DIR / "research_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


def write_report(s: dict) -> None:
    ic_cols = ["factor", "is_ic", "is_t", "oos_ic", "oos_t", "oos_pos%", "valid"]
    lines = [
        "# Factor research: can strong surges be screened ex-ante?",
        "",
        f"P1 (delay5 baseline) n={s['p1_n']}; P2 (regime 4/5/6, not in surge) n={s['p2_n']}.",
        "",
        "## P1 factor ICs (target = within-year excess rank)",
    ]
    lines.extend(msf.markdown_table(s["p1_factor_ic"], ic_cols))
    lines.extend(["", "## P1 factor ICs (target = t1_px30 surge label)"])
    lines.extend(msf.markdown_table(s["p1_factor_ic_t1"], ic_cols))
    lines.extend(["", "## Double-sort of valid factors vs score (incremental?)"])
    lines.extend(msf.markdown_table(s["double_sort_valid"], ["factor", "n", "y_spread_ctrl_score"]))
    lines.extend(
        [
            "",
            "## Composite (IS-selected factors, causal)",
            "",
            f"factors: {s['composite_is_factors']}",
            f"OOS t1_px30 capture: `{json.dumps(s['composite_tail_t1_oos'])}`",
            f"OOS excess-tail capture: `{json.dumps(s['composite_tail_excess_oos'])}`",
            f"by year: `{json.dumps(s['composite_tail_t1_by_year'])}`",
        ]
    )
    lines.extend(["", "## P2 broad screening (target = monthly rank of mkt-adj fwd20)"])
    lines.extend(msf.markdown_table(s["p2_factor_ic"], ic_cols))
    lines.extend(
        [
            "",
            f"P2 composite top-decile t1_px30 capture: `{json.dumps(s['p2_composite_capture_top10'])}`",
            f"P2 current anticipate-onset capture: `{json.dumps(s['p2_anticipate_onset_capture'])}`",
            "",
            "## Order duel (slot mirror, walk-forward factor selection)",
            "",
            "```json",
            json.dumps(s["order_duel"], ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## LR upper bound (reference only)",
            "",
            f"P1: `{json.dumps(s['lr_upper_bound_p1'])}`",
            f"P2: `{json.dumps(s['lr_upper_bound_p2'])}`",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
