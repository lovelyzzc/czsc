"""主升浪策略实盘镜像组合回测 v2 — Rust 后端 + 分层门控

与 surge_portfolio_backtest.py 输出格式兼容，但核心计算全部由
`czsc._native.research.simulate_surge_backtest` / `compute_surge_excess` 完成。

新增功能：
- `fill_mode` 参数化扫描（strict / partial_fill / all_fill / confidence_weighted）
- GateLevel / gate_confidence 字段（来自 czsc._native.trend_regime）

依赖 `surge_candidates_dump.py` 的输出（candidates.parquet / panel.parquet）。

    uv run --no-sync python scripts/surge_portfolio_backtest_v2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from czsc._native.trend_regime import (
    py_classify_gate_level as classify_gate_level,
    py_gate_confidence as gate_confidence,
    py_priority_score as priority_score,
    py_surge_score as surge_score,
)
from czsc._native.research import (
    simulate_surge_backtest_parquet as simulate_surge_backtest,
    compute_surge_excess_parquet as compute_surge_excess,
)

CAND_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_portfolio_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

BUY_COST = 0.0015
SELL_COST = 0.0025
SLOT_COUNTS = [10, 20]
MODES = ["confirm", "anticipate"]
FILL_MODES = ["strict", "partial_fill", "all_fill", "confidence_weighted"]
TRAIN_END = pd.Timestamp("2023-12-31")
TRAIN_END_YEAR = 2023
CONTROL_K = 50
CONTROL_MIN_VALID = 10
RNG_SEED = 42

MIN_AMOUNT_E = 1.0
STOP_MIN_PCT, STOP_MAX_PCT = 8.0, 20.0
GAP_LIMIT_MARGIN = 0.3

# Gate thresholds from Rust (reproduced for Python-side pre-filtering)
SURGE_GATE_VOL_RATIO = 1.2
SURGE_GATE_MA_SPREAD = 3.0
SURGE_GATE_RET20 = 8.0


def _is_st_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "退" in text


def load_st_intervals() -> dict[str, list[tuple[str, str, bool]]]:
    if not NAMECHANGE_PATH.exists():
        print(f"[WARN] {NAMECHANGE_PATH} 不存在，跳过历史 ST 过滤")
        return {}
    nc = pd.read_parquet(NAMECHANGE_PATH)
    nc["end_date"] = nc["end_date"].fillna("99999999")
    out: dict[str, list] = {}
    for code, g in nc.groupby("ts_code"):
        g = g.sort_values("start_date")
        out[code] = list(zip(g["start_date"], g["end_date"], [_is_st_name(x) for x in g["name"]], strict=False))
    return out


def is_st_on(intervals: dict, symbol: str, date: pd.Timestamp) -> bool:
    rows = intervals.get(symbol)
    if not rows:
        return False
    d = date.strftime("%Y%m%d")
    best = None
    for start, _end, is_st in rows:
        if start <= d:
            best = is_st
    return bool(best) if best is not None else False


def _classify_gate_level_row(row: pd.Series, mode: str) -> tuple[int, float]:
    """对一行候选计算 GateLevel(0-3) 和 gate_confidence(0-1)。"""
    vr_ok = row.get("sig_vol_ratio", 0) >= SURGE_GATE_VOL_RATIO
    sp_ok = row.get("sig_ma_spread_pct", 0) >= SURGE_GATE_MA_SPREAD
    zg_ok = bool(row.get("sig_above_zg", 0))
    pass_count = int(vr_ok) + int(sp_ok) + int(zg_ok)

    vr = row.get("sig_vol_ratio", 0.0)
    sp = row.get("sig_ma_spread_pct", 0.0)
    vr_score = min(max(vr / SURGE_GATE_VOL_RATIO, 0), 2) / 2 if not np.isnan(vr) and vr > 0 else 0.0
    sp_score = min(max(sp / SURGE_GATE_MA_SPREAD, 0), 2) / 2 if not np.isnan(sp) and sp > 0 else 0.0
    zg_score = 1.0 if zg_ok else 0.0
    confidence = min(vr_score * 0.4 + sp_score * 0.35 + zg_score * 0.25, 1.0)

    return pass_count, confidence


def gated_candidates(cand: pd.DataFrame, mode: str, st_intervals: dict) -> pd.DataFrame:
    """应用硬过滤但保留所有门控层级（GateLevel 由 Rust 侧过滤）。"""
    df = cand[(cand["mode"] == mode) & (cand["delay"] == 0)].copy()

    # 基础路径条件（surge_onset 中的非门控部分已由 dump 保证），
    # 但 anticipate 需要 ret20 条件
    if mode == "anticipate":
        df = df[df["sig_ret20"] >= SURGE_GATE_RET20]

    # 硬过滤
    df = df[
        (df["amount_e"] >= MIN_AMOUNT_E)
        & df["sl_pct"].between(STOP_MIN_PCT, STOP_MAX_PCT)
        & (df["gap_pct"] < df["limit_pct"] - GAP_LIMIT_MARGIN)
    ].copy()

    if st_intervals:
        st_mask = df.apply(lambda r: is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
        n_st = int(st_mask.sum())
        df = df[~st_mask]
        print(f"  [{mode}] ST/退市历史过滤剔除 {n_st} 笔")

    # 计算 GateLevel + confidence
    gate_info = df.apply(lambda r: _classify_gate_level_row(r, mode), axis=1)
    df["gate_level"] = [g[0] for g in gate_info]
    df["gate_confidence"] = [g[1] for g in gate_info]

    df["priority"] = [
        priority_score(s, sl, 0, rg)
        for s, sl, rg in zip(df["score"], df["sl_pct"], df["dec_regime"], strict=False)
    ]
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - SELL_COST) / (1 + BUY_COST) - 1) * 100
    return df.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)


def _to_candidate_dicts(df: pd.DataFrame) -> list[dict]:
    """将候选 DataFrame 转为字典列表（Rust 输入格式），向量化。"""
    _fmt_dt = lambda s: s.dt.strftime("%Y-%m-%d") if hasattr(s, "dt") else s.astype(str)
    records = []
    syms = df["symbol"].astype(str).values
    entry_dts = _fmt_dt(df["entry_dt"]).values
    exit_dts = _fmt_dt(df["exit_dt"]).values
    entry_px = df["entry_price"].values.astype(float)
    exit_px = df["exit_price"].values.astype(float)
    gate_lvl = df["gate_level"].values.astype(int)
    gate_conf = df["gate_confidence"].values.astype(float)
    priorities = df["priority"].values.astype(float)
    ret_gross = df["ret_gross_pct"].values.astype(float)
    hold = df["hold_days"].values.astype(int) if "hold_days" in df.columns else np.zeros(len(df), dtype=int)
    seg = df["seg"].astype(str).values if "seg" in df.columns else [""] * len(df)
    year = df["year"].values.astype(int) if "year" in df.columns else np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        records.append({
            "symbol": syms[i],
            "entry_dt": entry_dts[i],
            "exit_dt": exit_dts[i],
            "entry_price": entry_px[i],
            "exit_price": exit_px[i],
            "gate_level": gate_lvl[i],
            "gate_confidence": gate_conf[i],
            "priority": priorities[i],
            "ret_gross_pct": ret_gross[i],
            "hold_days": int(hold[i]),
            "seg": seg[i],
            "year": int(year[i]),
        })
    return records


def main():
    t0 = time.time()
    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    print(f"[候选] {len(cand)} 行（含全部 delay）")
    st_intervals = load_st_intervals()

    panel_path = str(CAND_DIR / "panel.parquet")
    panel_info = pd.read_parquet(CAND_DIR / "panel.parquet", columns=["symbol"])
    print(f"[面板] {len(panel_info)} 行 | ST 区间 {len(st_intervals)} 只")
    del panel_info

    summary: dict = {}

    for mode in MODES:
        df = gated_candidates(cand, mode, st_intervals)
        print(f"\n{'=' * 110}\n  {mode} | 过滤后候选 {len(df)} 笔（含全部 GateLevel）\n{'=' * 110}")
        summary[mode] = {"n_candidates": int(len(df))}

        cand_dicts = _to_candidate_dicts(df)

        for fill_mode in FILL_MODES:
            print(f"\n--- {mode} | fill_mode={fill_mode} ---")

            for n_slots in SLOT_COUNTS:
                tag = f"N{n_slots}"
                print(f"\n  [{mode}|{fill_mode}|{tag}] Rust simulate_surge_backtest...")
                t_bt = time.time()
                result = simulate_surge_backtest(
                    cand_dicts, panel_path,
                    n_slots=n_slots, fill_mode=fill_mode,
                    buy_cost=BUY_COST, sell_cost=SELL_COST,
                    train_end_year=TRAIN_END_YEAR,
                )
                elapsed = time.time() - t_bt

                n_trades = len(result["trades"])
                print(f"  成交 {n_trades} 笔 ({elapsed:.1f}s)")

                for seg in result["segments"]:
                    label = seg["label"]
                    c = seg["curve"]
                    p = seg["pair"]
                    print(f"  {label}: 年化{c['annual_return_pct']:.1f}% 夏普{c['sharpe']:.2f} "
                          f"回撤{c['max_drawdown_pct']:.1f}% 卡玛{c['calmar']:.2f} | "
                          f"胜率{p['win_rate_pct']:.1f}% 盈亏比{p['profit_loss_ratio']:.2f} "
                          f"净均值{p['net_mean_pct']:.2f}%")

                key = f"{fill_mode}_{tag}"
                summary.setdefault(mode, {})
                summary[mode][key] = {
                    "n_trades": n_trades,
                    "segments": result["segments"],
                }

                # Save trades
                if result["trades"]:
                    trades_df = pd.DataFrame(result["trades"])
                    trades_df.to_parquet(
                        OUTPUT_DIR / f"trades_{mode}_{fill_mode}_{tag}.parquet", index=False
                    )

                # Beta stripping (only for first slot count, strict mode)
                if n_slots == SLOT_COUNTS[0] and fill_mode == "strict" and result["trades"]:
                    print(f"\n  [beta 剥离] {mode}|{fill_mode}|{tag}...")
                    excess = compute_surge_excess(
                        result["trades"],
                        panel_path,
                        k=CONTROL_K, min_valid=CONTROL_MIN_VALID, seed=RNG_SEED,
                    )
                    for seg in excess["segments"]:
                        print(f"  {seg['label']}: n={seg['n']} 超额均值{seg['excess_mean_pct']:.2f}% "
                              f"t={seg['t_stat']:.2f} 正比{seg['positive_rate_pct']:.1f}%")
                    print(f"  [判定] {excess['verdict']}")
                    summary[mode][f"{fill_mode}_{tag}_excess"] = excess["segments"]
                    summary[mode][f"{fill_mode}_{tag}_verdict"] = excess["verdict"]

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
