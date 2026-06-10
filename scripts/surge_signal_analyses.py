"""主升浪信号层分析：新鲜度衰减 + 门控敏感性（读 candidates.parquet，秒级）

1. 新鲜度衰减：信号后第 d 根决策入场（d∈{0,1,2,3,5,7,10}）的净收益曲线，
   回答「daily_scan 的 SCAN_WINDOW=10 是否太宽」。
2. 门控敏感性：量比/散度/(ret20) 阈值 ×{0.8,1.0,1.2} 网格的 IS/OOS 矩阵，
   验证结论不在阈值刀刃上（稳健性检查，不取最优）。

    uv run --no-sync python scripts/surge_signal_analyses.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
import trend_regime as tr

CAND_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
BUY_COST, SELL_COST = 0.0015, 0.0025
MIN_AMOUNT_E = 1.0
STOP_MIN_PCT, STOP_MAX_PCT = 8.0, 20.0


def _net(df: pd.DataFrame) -> pd.Series:
    return ((1 + df["ret_gross_pct"] / 100) * (1 - SELL_COST) / (1 + BUY_COST) - 1) * 100


def _apply_gates(df: pd.DataFrame, mode: str, vol=None, spread=None, ret20=None) -> pd.DataFrame:
    vol = tr.SURGE_GATE_VOL_RATIO if vol is None else vol
    spread = tr.SURGE_GATE_MA_SPREAD if spread is None else spread
    ret20 = tr.SURGE_GATE_RET20 if ret20 is None else ret20
    g = (df["sig_vol_ratio"] >= vol) & (df["sig_ma_spread_pct"] >= spread) & (df["sig_above_zg"] == 1)
    if mode == "anticipate":
        g &= df["sig_ret20"] >= ret20
    return df[g.fillna(False)]


def _hard(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["amount_e"] >= MIN_AMOUNT_E)
        & df["sl_pct"].between(STOP_MIN_PCT, STOP_MAX_PCT)
        & (df["gap_pct"] < df["limit_pct"] - 0.3)
    ]


def _row(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return {"n": len(df)}
    net = _net(df)
    return {
        "n": len(df),
        "净均值%": round(net.mean(), 2),
        "净中位%": round(float(net.median()), 2),
        "胜率%": round((net > 0).mean() * 100, 1),
    }


def freshness_decay(cand: pd.DataFrame):
    print("=" * 100)
    print("  新鲜度衰减：信号后第 d 根决策入场的净收益（门控 + 硬过滤后）")
    print("=" * 100)
    for mode in ("confirm", "anticipate"):
        df = _hard(_apply_gates(cand[cand["mode"] == mode], mode))
        rows = {}
        for (d, seg), g in df.groupby(["delay", "seg"]):
            rows.setdefault(d, {})[seg] = _row(g)
        tbl = []
        for d in sorted(rows):
            r = {"delay": d}
            for seg in ("train", "test"):
                s = rows[d].get(seg, {})
                r[f"{seg}_n"] = s.get("n", 0)
                r[f"{seg}_净均值%"] = s.get("净均值%")
                r[f"{seg}_胜率%"] = s.get("胜率%")
            tbl.append(r)
        print(f"\n[{mode}]")
        print(pd.DataFrame(tbl).to_string(index=False))


def gate_sensitivity(cand: pd.DataFrame):
    print("\n" + "=" * 100)
    print("  门控敏感性：阈值 ×{0.8, 1.0, 1.2}（delay=0，门控 + 硬过滤后净均值%/胜率%）")
    print("=" * 100)
    mults = [0.8, 1.0, 1.2]
    for mode in ("confirm", "anticipate"):
        base = cand[(cand["mode"] == mode) & (cand["delay"] == 0)]
        dims = [("vol", tr.SURGE_GATE_VOL_RATIO), ("spread", tr.SURGE_GATE_MA_SPREAD)] + (
            [("ret20", tr.SURGE_GATE_RET20)] if mode == "anticipate" else []
        )
        print(f"\n[{mode}] 基准: " + ", ".join(f"{k}={v:g}" for k, v in dims))
        rows = []
        combos = product(mults, repeat=len(dims))
        for ms in combos:
            kw = {k: v * m for (k, v), m in zip(dims, ms, strict=False)}
            df = _hard(_apply_gates(base, mode, **kw))
            r = {dims[i][0]: round(list(kw.values())[i], 2) for i in range(len(dims))}
            for seg in ("train", "test"):
                s = _row(df[df["seg"] == seg])
                r[f"{seg}_n"] = s.get("n", 0)
                r[f"{seg}_净均值%"] = s.get("净均值%")
                r[f"{seg}_胜率%"] = s.get("胜率%")
            rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    freshness_decay(cand)
    gate_sensitivity(cand)


if __name__ == "__main__":
    main()
