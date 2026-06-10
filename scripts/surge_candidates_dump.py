"""主升浪候选信号全市场抽取（门控前 + 多延迟模拟）—— 下游分析的共享数据源

对每只股票全量流式重放，记录**门控前**的主升浪候选（仅状态跳变 + 路径条件，
量比/散度/ret20 门控不在此应用，存特征供下游后置过滤 → 门控敏感性扫描零成本）：

- confirm    跳变进入 7/8，且 prior 40 根走过 4 与 5；
- anticipate 跳变进入 5，且 prior 40 根走过 4。

每个候选 × 决策延迟 d∈{0,1,2,3,5,7,10}（信号后第 d 根收盘决策、次日开盘入场，
要求决策日仍处于上行家族 5/6/7/8），独立模拟 FULL 退出（SL2 + 18% 跟踪 +
背驰/破坏次日开盘退出 + 最大持有 60），记录毛收益（成本由下游加）。

与 `surge_regime_backtest.py` 的差异：候选独立模拟、允许同票时间重叠
（组合层会强制单票单仓；pair 级分析接受重叠）。

输出：
- scripts/_output/surge_candidates/candidates.parquet  一行 = (候选, delay)
- scripts/_output/surge_candidates/panel.parquet       全市场 dt×symbol 的 open/close/amount

    uv run --no-sync python scripts/surge_candidates_dump.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trend_regime as tr
from trend_regime import Regime

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELAYS = [0, 1, 2, 3, 5, 7, 10]
MAX_HOLD_DAYS = 60
TRAIL_STOP = 0.18
TRAIN_END = pd.Timestamp("2023-12-31")
SELL_SET = tr.SELL_REGIMES  # {9, 10}
UPTREND_FAMILY = {Regime.UpwardDeparture, Regime.ThirdBuy, Regime.MainUptrend, Regime.Acceleration}


def _is_candidate(prev: int, regime: int, prior: list[int], mode: str) -> bool:
    """门控前候选：仅状态跳变 + 路径条件（surge_onset 去掉特征门控的部分）。"""
    prior_set = set(prior)
    if mode == "confirm":
        entered = prev not in (Regime.MainUptrend, Regime.Acceleration) and regime in (
            Regime.MainUptrend,
            Regime.Acceleration,
        )
        return entered and Regime.PivotBuilding in prior_set and Regime.UpwardDeparture in prior_set
    entered = prev != Regime.UpwardDeparture and regime == Regime.UpwardDeparture
    return entered and Regime.PivotBuilding in prior_set


def _simulate_full(p_dec: int, states: list, regime_by_idx: dict, ind: dict):
    """从决策 bar p_dec 模拟一笔 FULL 退出交易（与 surge_regime_backtest._simulate_surge
    同逻辑，去掉 holds 构建以提速）。返回 (entry_idx, entry_price, exit_idx, exit_price, reason)。"""
    n = ind["n"]
    o, c, lo = ind["open"], ind["close"], ind["low"]
    sig = states[p_dec]
    entry_idx = sig.idx + 1
    if entry_idx >= n or np.isnan(sig.next_open):
        return None
    entry_price = sig.next_open
    sl_ref = sig.sl_ref

    peak = entry_price
    exit_idx, exit_price, reason = None, None, "max_hold"
    last_j = min(entry_idx + MAX_HOLD_DAYS, n) - 1
    for j in range(entry_idx, last_j + 1):
        peak = max(peak, c[j])
        if j == entry_idx:
            continue
        if not np.isnan(sl_ref) and lo[j] <= sl_ref:
            exit_idx, exit_price, reason = j, min(o[j], sl_ref), "sl2"
            break
        if peak > entry_price and (peak - c[j]) / peak >= TRAIL_STOP:
            exit_idx, exit_price, reason = j, c[j], "trail18"
            break
        if regime_by_idx.get(j) in SELL_SET:
            exit_idx, exit_price, reason = j, o[j + 1] if j + 1 < n else c[j], "state"
            break
    if exit_idx is None:
        exit_idx, exit_price = last_j, c[last_j]
    return entry_idx, entry_price, exit_idx, exit_price, reason


def _feat(feats: dict | None, key: str) -> float:
    if not feats:
        return np.nan
    v = feats.get(key)
    return np.nan if v is None else float(v)


def _process(parquet_path: str) -> list[dict] | None:
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
    amount = df["amount"].to_numpy(dtype=float) if "amount" in df.columns else np.full(ind["n"], np.nan)
    limit_pct = tr.limit_pct_for(symbol)

    rows = []
    for mode in ("confirm", "anticipate"):
        for p in range(1, len(states)):
            prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
            if not _is_candidate(states[p - 1].regime, states[p].regime, prior, mode):
                continue
            sig_feats = states[p].feats or {}
            for d in DELAYS:
                p_dec = p + d
                if p_dec >= len(states) or states[p_dec].regime not in UPTREND_FAMILY:
                    continue  # 决策日已不在上行家族 → 实盘不会列出
                sim = _simulate_full(p_dec, states, regime_by_idx, ind)
                if sim is None:
                    continue
                entry_idx, entry_price, exit_idx, exit_price, reason = sim
                dec = states[p_dec]
                dec_close = dec.close
                sl = dec.sl_ref if dec.sl_ref == dec.sl_ref else dec.zd
                sl = sl if (sl == sl and sl > 0) else np.nan
                sl_pct = (dec_close - sl) / dec_close * 100 if sl == sl else np.nan
                dec_amt = amount[dec.idx]
                rows.append(
                    {
                        "symbol": symbol,
                        "mode": mode,
                        "delay": d,
                        "sig_dt": states[p].dt,
                        "dec_dt": dec.dt,
                        "dec_regime": int(dec.regime),
                        "entry_idx": entry_idx,
                        "entry_dt": pd.Timestamp(ind["dates"][entry_idx]),
                        "entry_price": float(entry_price),
                        "exit_idx": exit_idx,
                        "exit_dt": pd.Timestamp(ind["dates"][exit_idx]),
                        "exit_price": float(exit_price),
                        "ret_gross_pct": round((exit_price / entry_price - 1) * 100, 3),
                        "hold_days": exit_idx - entry_idx,
                        "exit_reason": reason,
                        "sl_ref": float(sl) if sl == sl else np.nan,
                        "sl_pct": round(sl_pct, 2) if sl_pct == sl_pct else np.nan,
                        "score": tr.surge_score(dec.feats),
                        "gap_pct": round((entry_price / dec_close - 1) * 100, 2),
                        "amount_e": round(dec_amt / 1e5, 3) if dec_amt == dec_amt else np.nan,
                        "limit_pct": limit_pct,
                        "sig_vol_ratio": _feat(sig_feats, "vol_ratio"),
                        "sig_ma_spread_pct": _feat(sig_feats, "ma_spread_pct"),
                        "sig_ret20": _feat(sig_feats, "ret20"),
                        "sig_above_zg": _feat(sig_feats, "above_zg"),
                    }
                )
    return rows or None


def _panel_one(parquet_path: str) -> pd.DataFrame | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "dt": df["dt"],
            "open": df["open"].astype(float),
            "close": df["close"].astype(float),
            "amount_e": (df["amount"].astype(float) / 1e5) if "amount" in df.columns else np.nan,
        }
    )
    return out


def main():
    t0 = time.time()
    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只 | {n_workers} 进程 | 延迟集 {DELAYS}")

    ctx = mp.get_context("spawn")
    all_rows = []
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, files, chunksize=20), 1):
            if res:
                all_rows.extend(res)
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 候选行 {len(all_rows)} | {time.time() - t0:.0f}s")

    cand = pd.DataFrame(all_rows)
    cand["seg"] = np.where(cand["dec_dt"] <= TRAIN_END, "train", "test")
    cand["year"] = cand["dec_dt"].dt.year
    cand.to_parquet(OUTPUT_DIR / "candidates.parquet", index=False)
    print(f"[候选] {len(cand)} 行 → candidates.parquet")

    print("[面板] 构建 dt×symbol 价格面板 ...")
    panels = []
    with ctx.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_panel_one, files, chunksize=50):
            if res is not None:
                panels.append(res)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(OUTPUT_DIR / "panel.parquet", index=False)
    print(f"[面板] {len(panel)} 行 → panel.parquet")
    print(f"[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
