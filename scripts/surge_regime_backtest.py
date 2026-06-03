"""主升浪启动 OOS 回测 —— 确认追入 vs 启动埋伏

把 `trend_regime.surge_onset` 的因果「主升浪启动」信号做成入场，对比两种买点：

- **confirm 确认追入**：状态跳变进入 `7 主升延续 / 8 加速主升` + 主升门控（放量/发散/立于中枢上方）
  + 启动前走过 `中枢构造(4) → 向上离开(5)`。
- **anticipate 启动埋伏**：状态跳变进入 `5 向上离开中枢` + 更强门控（加 ret20≥8）+ 走过 `4`。

退出统一（源自 `surge_characteristics` 研究）：
- `FULL`：笔结构止损 SL2 + 浮盈后 ~18% 跟踪止损（区间回撤 P75≈18%）+ 进入 `背驰(9)/破坏(10)`
  次日开盘退出 + 最大持有；
- `SL2`：仅 SL2 + 最大持有（对照）。

防过拟合 / 因果：门控阈值粗粒度 a-priori（取在全样本中位数之下，不按回测精调）；
全程流式 `iter_states` 因果；次日开盘成交；OOS 切 train≤2023 / test≥2024 两段分别评估。
单只票同时只持一仓（顺序扫描）。

    uv run --no-sync python scripts/surge_regime_backtest.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trend_regime as tr
from trend_regime_backtest import (
    FEE_RATE,
    MAX_HOLD_DAYS,
    SELL_SET,
    TRAIN_END,
    YEARLY_DAYS,
    _exit_dist,
    _print_table,
    _split_by_seg,
    _stats_for,
    _to_dfw,
)
from wbt import generate_backtest_report

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = tr.DATA_DIR
TRAIL_STOP = 0.18  # 跟踪止损幅度（研究：主升浪区间内回撤 P75≈18%）

ENTRY_MODES = ["confirm", "anticipate"]
EXIT_MODES = ["FULL", "SL2"]
COMBOS = [f"{e}|{x}" for e in ENTRY_MODES for x in EXIT_MODES]


def _simulate_surge(p_entry, states, regime_by_idx, ind, exit_mode, symbol):
    """从主升浪启动信号 p_entry 模拟一笔交易（次日开盘入场）。"""
    n = ind["n"]
    o, c, lo = ind["open"], ind["close"], ind["low"]

    sig = states[p_entry]
    entry_idx = sig.idx + 1
    if entry_idx >= n or np.isnan(sig.next_open):
        return None, None, None
    entry_price = sig.next_open
    sl_ref = sig.sl_ref
    seg = "train" if pd.Timestamp(sig.dt) <= TRAIN_END else "test"

    use_trail = exit_mode == "FULL"
    use_state = exit_mode == "FULL"

    peak = entry_price
    holds = []
    exit_idx, exit_price, reason = None, None, "max_hold"
    last_j = min(entry_idx + MAX_HOLD_DAYS, n) - 1

    for j in range(entry_idx, last_j + 1):
        peak = max(peak, c[j])
        holds.append(
            {"dt": pd.Timestamp(ind["dates"][j]), "symbol": symbol, "weight": 1, "price": float(c[j]), "seg": seg}
        )
        if j == entry_idx:
            continue
        # 1) SL2 笔结构止损（跳空按开盘）
        if not np.isnan(sl_ref) and lo[j] <= sl_ref:
            exit_idx, exit_price, reason = j, min(o[j], sl_ref), "sl2"
            break
        # 2) 浮盈后 ~18% 跟踪止损
        if use_trail and peak > entry_price and (peak - c[j]) / peak >= TRAIL_STOP:
            exit_idx, exit_price, reason = j, c[j], "trail18"
            break
        # 3) 状态退出：进入背驰/结构破坏 → 次日开盘
        if use_state and regime_by_idx.get(j) in SELL_SET:
            exit_idx, exit_price, reason = j, o[j + 1] if j + 1 < n else c[j], "state"
            break

    if exit_idx is None:
        exit_idx, exit_price = last_j, c[last_j]

    holds = holds[: exit_idx - entry_idx + 1]
    pair = {
        "symbol": symbol,
        "entry_dt": pd.Timestamp(sig.dt),
        "ret_pct": round((exit_price / entry_price - 1) * 100, 3),
        "hold_days": exit_idx - entry_idx,
        "exit_reason": reason,
        "seg": seg,
    }
    return holds, pair, exit_idx


def _walk_surge(states, regime_by_idx, regimes, ind, symbol, entry_mode, exit_mode):
    """顺序扫描，单票同时只持一仓：主升浪启动入场 → 退出后再找下一启动。"""
    holds_all, pairs_all = [], []
    p = 1
    while p < len(states):
        prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
        if not tr.surge_onset(states[p - 1].regime, states[p].regime, states[p].feats, prior, entry_mode):
            p += 1
            continue
        holds, pair, exit_idx = _simulate_surge(p, states, regime_by_idx, ind, exit_mode, symbol)
        if pair is None:
            p += 1
            continue
        holds_all.extend(holds)
        pairs_all.append(pair)
        while p < len(states) and states[p].idx <= exit_idx:
            p += 1
    return holds_all, pairs_all


def _process(parquet_path):
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df, with_features=True)
    if len(states) < 30:
        return None
    ind = tr.compute_indicators(df)
    regimes = [s.regime for s in states]
    regime_by_idx = {s.idx: s.regime for s in states}
    symbol = df["symbol"].iloc[0]

    out = {}
    for entry_mode in ENTRY_MODES:
        for exit_mode in EXIT_MODES:
            holds, pairs = _walk_surge(states, regime_by_idx, regimes, ind, symbol, entry_mode, exit_mode)
            if pairs:
                out[f"{entry_mode}|{exit_mode}"] = {"holds": holds, "pairs": pairs}
    return out or None


def main():
    t0 = time.time()
    print("=" * 100)
    print("  主升浪启动 OOS 回测 — 确认追入 vs 启动埋伏")
    print("=" * 100)
    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(
        f"[数据] {len(files)} 只 | {n_workers} 进程 | 门控 量比≥{tr.SURGE_GATE_VOL_RATIO} 散度≥{tr.SURGE_GATE_MA_SPREAD}%"
    )
    print(f"[组合] {ENTRY_MODES} × {EXIT_MODES} | 跟踪止损 {TRAIL_STOP:.0%} | OOS train≤{TRAIN_END.date()}\n")

    agg = {c: {"train": {"holds": [], "pairs": []}, "test": {"holds": [], "pairs": []}} for c in COMBOS}
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, files, chunksize=20), 1):
            if res:
                for combo, data in res.items():
                    for p, h in _split_by_seg(data["holds"], data["pairs"]):
                        agg[combo][p]["holds"].extend(h["holds"])
                        agg[combo][p]["pairs"].extend(h["pairs"])
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {time.time() - t0:.0f}s")
    print(f"\n[扫描完成] {time.time() - t0:.0f}s\n")

    rows = {"train": [], "test": []}
    for combo in COMBOS:
        for seg in ("train", "test"):
            r = _stats_for(agg[combo][seg]["holds"], agg[combo][seg]["pairs"], combo)
            if r:
                rows[seg].append(r)
    _print_table("IN-SAMPLE（train ≤2023）", rows["train"])
    _print_table("OUT-OF-SAMPLE（test ≥2024）", rows["test"])

    for combo in COMBOS:
        for seg in ("train", "test"):
            pairs = agg[combo][seg]["pairs"]
            if pairs:
                print(f"\n[{combo} | {seg}] 退出分布: {_exit_dist(pairs)}")

    # HTML：两种入场的 FULL 退出
    for entry in ENTRY_MODES:
        for seg in ("train", "test"):
            dfw = _to_dfw(agg[f"{entry}|FULL"][seg]["holds"])
            if dfw is not None:
                try:
                    generate_backtest_report(
                        df=dfw,
                        output_path=str(OUTPUT_DIR / f"{entry}_FULL_{seg}.html"),
                        title=f"主升浪 {entry}|FULL ({seg})",
                        fee_rate=FEE_RATE,
                        weight_type="ts",
                        yearly_days=YEARLY_DAYS,
                    )
                except Exception as e:
                    print(f"  HTML 失败 {entry}/{seg}: {e}")

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
