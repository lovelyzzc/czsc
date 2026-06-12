"""因子层研究 Stage1：存量因子快扫（delay5 基线人群，零 dump）

回答先验问题：candidates.parquet 已有的可观测量（信号门特征/score/止损/成交额/路径收益/
相对强度）对「年内标准化超额」与「尾部赢家标签」有没有判别力？为 Stage2 扩展因子
（多为同族表亲）校准预期。

方法（预声明）：
- 目标 y = excess_pct 的年内 pct-rank（消 2025 主导）；尾部标签 = 年内 excess top20%；
- 月度 Fama-MacFeth rank-IC：按决策月分组算 spearman(因子, y)，对月度 IC 序列做 t；
  IS(≤2023)/OOS(≥2024) 分段，「有效」= OOS |t|≥2 且与 IS 同号；
- 尾部捕获：OOS 内按月取因子 top-20%，对尾部标签的捕获率 / 基率 20% = lift；
- gap 因子为入场晨知信息（决策收盘不可知），单独标注。

    uv run --no-sync python scripts/surge_factor_stage1.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_selection_audit as ssa

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_factors"
MIN_MONTH_ROWS = 10
TAIL_Q = 0.8  # 年内 excess top 20% 为尾部赢家


def build_population() -> pd.DataFrame:
    df = ssa.load_universe()
    df = ssa.with_excess(df)
    df = df[ssa.cond_mask(df, ssa.BASE)].copy()
    # 路径收益：信号日→决策日（panel close）
    panel = pd.read_parquet(msf.CAND_DIR / "panel.parquet", columns=["symbol", "dt", "close"])
    df = df.merge(panel.rename(columns={"dt": "sig_dt", "close": "sig_close"}), on=["symbol", "sig_dt"], how="left")
    df = df.merge(panel.rename(columns={"dt": "dec_dt", "close": "dec_close"}), on=["symbol", "dec_dt"], how="left")
    df["path_ret"] = (df["dec_close"] / df["sig_close"] - 1) * 100
    # 相对强度：信号日 ret20 − 当日市场中位 ret20
    market = pd.read_parquet(msf.OUTPUT_DIR / "market_state.parquet", columns=["dt", "mkt_ret20_median"])
    df = df.merge(market.rename(columns={"dt": "sig_dt", "mkt_ret20_median": "mkt_med_sig"}), on="sig_dt", how="left")
    df["rel_ret20"] = df["sig_ret20"] - df["mkt_med_sig"] * 100
    df["log_amount"] = np.log10(df["amount_e"].clip(lower=0.01))
    df["month"] = df["dec_dt"].dt.to_period("M")
    df = df[df["excess_pct"].notna()].copy()
    df["y"] = df.groupby("year")["excess_pct"].rank(pct=True)
    df["tail"] = (df["y"] >= TAIL_Q).astype(int)
    return df


FACTORS = {
    "sig_vol_ratio": "信号日量比",
    "sig_ma_spread_pct": "信号日MA散度",
    "sig_ret20": "信号日ret20",
    "score": "决策日主升强度score",
    "priority": "决策日priority",
    "sl_pct": "止损幅度",
    "log_amount": "log成交额",
    "path_ret": "信号→决策5日收益",
    "rel_ret20": "相对强度(ret20−市场中位)",
    "gap_pct": "次日开盘gap(入场晨知)",
}


def monthly_ic(df: pd.DataFrame, factor: str) -> pd.Series:
    ics = {}
    for m, g in df.groupby("month"):
        sub = g[[factor, "y"]].dropna()
        if len(sub) < MIN_MONTH_ROWS:
            continue
        ics[m] = sub[factor].corr(sub["y"], method="spearman")
    return pd.Series(ics).dropna()


def ic_stats(ics: pd.Series) -> dict:
    if len(ics) < 4:
        return {"months": int(len(ics))}
    return {
        "months": int(len(ics)),
        "ic_mean": msf.round_float(ics.mean(), 3),
        "ic_t": msf.round_float(msf.t_stat(ics)),
        "ic_pos_pct": msf.round_float((ics > 0).mean() * 100, 1),
    }


def tail_capture(df: pd.DataFrame, factor: str) -> dict:
    """OOS：按月因子 top-20% 子集的尾部标签捕获率 vs 基率。"""
    oos = df[(df["seg"] == "test") & df[factor].notna()].copy()
    if not len(oos):
        return {}
    oos["fq"] = oos.groupby("month")[factor].rank(pct=True)
    top = oos[oos["fq"] >= 0.8]
    if len(top) < 30:
        return {"top_n": int(len(top))}
    base = oos["tail"].mean()
    rate = top["tail"].mean()
    return {
        "top_n": int(len(top)),
        "base_rate": msf.round_float(base * 100, 1),
        "top_rate": msf.round_float(rate * 100, 1),
        "lift": msf.round_float(rate / base, 2) if base > 0 else None,
    }


def quintile_spectrum(df: pd.DataFrame, factor: str) -> list[dict]:
    oos = df[(df["seg"] == "test") & df[factor].notna()].copy()
    if len(oos) < 150:
        return []
    oos["q"] = oos.groupby("month")[factor].rank(pct=True).mul(5).clip(upper=4.999).astype(int)
    rows = []
    for q, g in oos.groupby("q"):
        rows.append(
            {
                "q": int(q) + 1,
                "n": int(len(g)),
                "excess_mean": msf.round_float(g["excess_pct"].mean()),
                "excess_median": msf.round_float(g["excess_pct"].median()),
                "tail_rate": msf.round_float(g["tail"].mean() * 100, 1),
            }
        )
    return rows


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_population()
    n_is, n_oos = int((df["seg"] == "train").sum()), int((df["seg"] == "test").sum())
    print(f"[人群] delay5 基线 {len(df)} 行（excess 有效）| IS {n_is} | OOS {n_oos}")

    rows, detail = [], {}
    for factor, cn in FACTORS.items():
        ics = monthly_ic(df, factor)
        is_ics = ics[ics.index.map(lambda p: p.year <= 2023)]
        oos_ics = ics[ics.index.map(lambda p: p.year >= 2024)]
        s_is, s_oos = ic_stats(is_ics), ic_stats(oos_ics)
        cap = tail_capture(df, factor)
        same_sign = (
            s_is.get("ic_mean") is not None
            and s_oos.get("ic_mean") is not None
            and np.sign(s_is["ic_mean"]) == np.sign(s_oos["ic_mean"])
        )
        valid = bool(same_sign and abs(s_oos.get("ic_t") or 0) >= 2)
        rows.append(
            {
                "factor": factor,
                "cn": cn,
                "is_ic": s_is.get("ic_mean"),
                "is_t": s_is.get("ic_t"),
                "oos_ic": s_oos.get("ic_mean"),
                "oos_t": s_oos.get("ic_t"),
                "oos_pos%": s_oos.get("ic_pos_pct"),
                "tail_lift": cap.get("lift"),
                "valid": valid,
            }
        )
        detail[factor] = {"is": s_is, "oos": s_oos, "tail_capture": cap, "quintiles": quintile_spectrum(df, factor)}
        print(f"  {factor:<22} IS_IC={s_is.get('ic_mean')} OOS_IC={s_oos.get('ic_mean')} "
              f"OOS_t={s_oos.get('ic_t')} lift={cap.get('lift')} valid={valid}")

    summary = {"n": len(df), "n_is": n_is, "n_oos": n_oos, "factors": rows, "detail": detail}
    with (OUTPUT_DIR / "stage1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    lines = [
        "# Factor stage1: existing observables on delay5 baseline population",
        "",
        f"Population: {len(df)} rows (IS {n_is} / OOS {n_oos}); y = within-year pct-rank of excess; "
        "monthly Fama-MacBeth rank-IC; valid = OOS |t|>=2 and same sign as IS.",
        "",
    ]
    lines.extend(msf.markdown_table(rows, ["factor", "cn", "is_ic", "is_t", "oos_ic", "oos_t", "oos_pos%", "tail_lift", "valid"]))
    for factor in FACTORS:
        q = detail[factor]["quintiles"]
        if q:
            lines.extend(["", f"## {factor} OOS quintiles"])
            lines.extend(msf.markdown_table(q, ["q", "n", "excess_mean", "excess_median", "tail_rate"]))
    (OUTPUT_DIR / "stage1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
