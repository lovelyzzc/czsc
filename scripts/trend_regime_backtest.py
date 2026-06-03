"""走势类型状态机 —— 右侧买点 + 止损止盈矩阵 + OOS 回测

基于 ``trend_regime.iter_states`` 的因果状态序列，按「状态跳变」驱动买卖：

- **买点（右侧确认为主）**：状态跳变进入买点集合时入场，**次日开盘成交**。
  - 主集合 ``W56 = {5 向上离开中枢, 6 三买确认}``（用户选定）
  - 对照 ``T6 = {6 三买确认}``（更纯、更少）
- **卖点（统一）**：跳变进入 ``{9 背驰衰竭, 10 结构破坏}`` 平仓，叠加 **笔结构止损 SL2**
  （跌破入场时最近向下笔低点）与最大持有期；并对比多组止损止盈退出方式。
- **退出矩阵**（回答「此时止损止盈策略是什么」）：
  ``STATE`` 纯状态退出 / ``STATE_SL2`` 状态+SL2 / ``SL2`` 仅SL2 /
  ``ATR`` ATR动态止损 / ``FIXED`` 固定-8% / ``TRAIL`` SL2+跟踪 / ``TP_TRAIL`` SL2+止盈+跟踪。
  ATR/Wyckoff 逻辑沿用 ``surge_codex_backtest.py``。
- **OOS 防过拟合**：按入场日期切分 train(≤2023-12-31) / test(≥2024-01-01)，两段分别统计；
  全市场汇总，单只票内同时只持一仓（顺序扫描，不重叠）。

去未来化：信号收盘确认、**次日开盘成交**；止损按当日触及价（跳空则按开盘）成交；
涨停日不构成买点状态（已在分类器置为 NotTradable）。

    uv run --no-sync python scripts/trend_regime_backtest.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trend_regime as tr
from trend_regime import Regime
from wbt import generate_backtest_report

from czsc import WeightBacktest

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "trend_regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = tr.DATA_DIR
FEE_RATE = 0.0002
MAX_HOLD_DAYS = 60
YEARLY_DAYS = 252

# 止损止盈参数（少量、固定，理论驱动）
FIXED_STOP_PCT = 0.08
TAKE_PROFIT_PCT = 0.20  # 主升浪给足空间，避免过早止盈
ATR_PERIOD = 20
ATR_FLOOR_FACTOR = 0.85
WYCKOFF_MULT = {"Markup": 2.0, "Accumulation": 1.5, "Distribution": 1.0, "Markdown": 0.8}

# OOS 时间切分
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

# 买点集合
BUY_SETS = {
    "W56_离开+三买": frozenset({Regime.UpwardDeparture, Regime.ThirdBuy}),
    "T6_仅三买": frozenset({Regime.ThirdBuy}),
}
SELL_SET = tr.SELL_REGIMES  # {9, 10}

# 退出模式
MODES = ["STATE", "STATE_SL2", "SL2", "ATR", "FIXED", "TRAIL", "TP_TRAIL"]


# --------------------------------------------------------------------------- #
# ATR / Wyckoff（沿用 surge_codex_backtest.py）
# --------------------------------------------------------------------------- #
def _compute_atr(high, low, close, period=ATR_PERIOD):
    n = len(high)
    tr_arr = np.zeros(n)
    tr_arr[0] = high[0] - low[0]
    for i in range(1, n):
        tr_arr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr_arr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr_arr[i]) / period
    return atr


def _wyckoff_mult(close, lookback=50):
    n = len(close)
    mult = np.full(n, 1.0)
    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1
        if ret > 0.10:
            mult[i] = WYCKOFF_MULT["Markup"]
        elif ret < -0.10:
            mult[i] = WYCKOFF_MULT["Markdown"]
        elif abs(ret) <= 0.05 and close[i] < close[i - lookback : i].mean():
            mult[i] = WYCKOFF_MULT["Accumulation"]
        else:
            mult[i] = WYCKOFF_MULT["Distribution"]
    return mult


def _atr_stop(close_val, atr_val, mult_val):
    if np.isnan(atr_val):
        return close_val * ATR_FLOOR_FACTOR
    return max(close_val - atr_val * mult_val, close_val * ATR_FLOOR_FACTOR)


# --------------------------------------------------------------------------- #
# 单笔交易模拟（次日开盘入场 + 多模式退出）
# --------------------------------------------------------------------------- #
def _simulate(p_entry, states, regime_by_idx, ind, mode, symbol):
    """从状态序列下标 p_entry 的买点信号开始模拟一笔交易。

    返回 (holds_rows, pair, exit_raw_idx)；无法入场返回 (None, None, None)。
    """
    n = ind["n"]
    o, c, lo = ind["open"], ind["close"], ind["low"]
    atr, wmult = ind["atr"], ind["wmult"]

    sig = states[p_entry]
    t = sig.idx
    entry_idx = t + 1  # 次日开盘成交
    if entry_idx >= n or np.isnan(sig.next_open):
        return None, None, None
    entry_price = sig.next_open
    sl_ref = sig.sl_ref  # 入场时最近向下笔低点（SL2 结构止损）
    seg = "train" if pd.Timestamp(sig.dt) <= TRAIN_END else "test"  # 按入场日期定段

    use_state = mode in ("STATE", "STATE_SL2")
    use_sl2 = mode in ("STATE_SL2", "SL2", "TRAIL", "TP_TRAIL")
    use_atr = mode == "ATR"
    use_fixed = mode == "FIXED"
    use_trail = mode in ("TRAIL", "TP_TRAIL")
    use_tp = mode == "TP_TRAIL"

    atr_stop_px = _atr_stop(c[entry_idx], atr[entry_idx], wmult[entry_idx]) if use_atr else None
    fixed_stop_px = entry_price * (1 - FIXED_STOP_PCT) if use_fixed else None

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

        # 1) SL2 结构止损（跌破入场最近向下笔低点）—— 跳空按开盘
        if use_sl2 and not np.isnan(sl_ref) and lo[j] <= sl_ref:
            exit_idx, exit_price, reason = j, min(o[j], sl_ref), "sl2"
            break
        # 2) ATR 动态止损
        if use_atr and lo[j] <= atr_stop_px:
            exit_idx, exit_price, reason = j, min(o[j], atr_stop_px), "atr"
            break
        # 3) 固定百分比止损
        if use_fixed and lo[j] <= fixed_stop_px:
            exit_idx, exit_price, reason = j, min(o[j], fixed_stop_px), "fixed"
            break
        # 4) 止盈（次日开盘出）
        if use_tp and c[j] / entry_price - 1 >= TAKE_PROFIT_PCT:
            exit_idx, exit_price, reason = j, o[j + 1] if j + 1 < n else c[j], "take_profit"
            break
        # 5) 分级跟踪止损
        if use_trail:
            gain, dd = peak / entry_price - 1, (peak - c[j]) / peak if peak > 0 else 0
            if (gain >= 0.25 and dd >= 0.10) or (gain >= 0.10 and dd >= 0.15):
                exit_idx, exit_price, reason = j, c[j], "trail"
                break
        # 6) 状态退出：进入背驰/结构破坏 → 次日开盘出
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
        "buy_regime": int(sig.regime),
    }
    return holds, pair, exit_idx


def _walk(states, regime_by_idx, ind, symbol, buy_set, mode):
    """顺序扫描：单只票内同时只持一仓，买点跳变入场，到期/止损/状态退出后再找下一买点。"""
    holds_all, pairs_all = [], []
    p = 1
    while p < len(states):
        cur, prev = states[p].regime, states[p - 1].regime
        is_buy_transition = cur in buy_set and prev not in buy_set
        if not is_buy_transition:
            p += 1
            continue
        holds, pair, exit_idx = _simulate(p, states, regime_by_idx, ind, mode, symbol)
        if pair is None:
            p += 1
            continue
        holds_all.extend(holds)
        pairs_all.append(pair)
        # 跳到出场 bar 之后再继续扫描（避免重叠持仓）
        while p < len(states) and states[p].idx <= exit_idx:
            p += 1
    return holds_all, pairs_all


def _process(parquet_path):
    """单只票：产出全部 (buy_set × mode) 组合的 holds/pairs。"""
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df)
    if len(states) < 30:
        return None

    ind = tr.compute_indicators(df)
    ind["atr"] = _compute_atr(ind["high"], ind["low"], ind["close"])
    ind["wmult"] = _wyckoff_mult(ind["close"])
    regime_by_idx = {s.idx: s.regime for s in states}
    symbol = df["symbol"].iloc[0]

    out = {}
    for bs_name, bs in BUY_SETS.items():
        for mode in MODES:
            holds, pairs = _walk(states, regime_by_idx, ind, symbol, bs, mode)
            if pairs:
                out[f"{bs_name}|{mode}"] = {"holds": holds, "pairs": pairs}
    return out or None


# --------------------------------------------------------------------------- #
# 聚合 + 统计
# --------------------------------------------------------------------------- #
def _stats_for(holds, pairs, tag):
    """对某 (combo, segment) 的 holds/pairs 计算组合级 + 交易级指标。"""
    if not pairs:
        return None
    rets = np.array([p["ret_pct"] for p in pairs])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    row = {
        "组合": tag,
        "交易数": len(rets),
        "胜率": round(len(wins) / len(rets) * 100, 1),
        "盈亏比": round(abs(wins.mean() / losses.mean()), 2) if len(wins) and len(losses) else np.nan,
        "平均收益%": round(rets.mean(), 2),
        "中位收益%": round(float(np.median(rets)), 2),
        "平均持有": round(np.mean([p["hold_days"] for p in pairs]), 1),
    }
    # 组合级（WeightBacktest，close-to-close 口径）
    if holds:
        dfw = pd.DataFrame(holds)
        if dfw.duplicated(subset=["dt", "symbol"]).any():
            dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(weight=("weight", "max"), price=("price", "first"))
        dfw = dfw[["dt", "symbol", "weight", "price"]]
        try:
            st = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=YEARLY_DAYS).stats
            row["年化%"] = round(st.get("年化收益", np.nan) * 100, 1) if st.get("年化收益") is not None else np.nan
            row["夏普"] = round(st.get("夏普比率", np.nan), 2)
            row["卡玛"] = round(st.get("卡玛比率", np.nan), 2)
            row["最大回撤%"] = round(st.get("最大回撤", np.nan) * 100, 1) if st.get("最大回撤") is not None else np.nan
        except Exception:
            pass
    return row


def _exit_dist(pairs):
    d = {}
    for p in pairs:
        d[p["exit_reason"]] = d.get(p["exit_reason"], 0) + 1
    return d


def main():
    t0 = time.time()
    print("=" * 100)
    print("  走势类型状态机 — 右侧买点 + 止损止盈矩阵 + OOS 回测")
    print("=" * 100)

    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只个股 | {n_workers} 进程")
    print(
        f"[组合] {len(BUY_SETS)} 买点集合 × {len(MODES)} 退出模式 | OOS: train≤{TRAIN_END.date()} / test≥{TEST_START.date()}\n"
    )

    combos = [f"{bs}|{m}" for bs in BUY_SETS for m in MODES]
    agg = {combo: {"train": {"holds": [], "pairs": []}, "test": {"holds": [], "pairs": []}} for combo in combos}

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

    rows_train, rows_test = [], []
    for combo in combos:
        r_tr = _stats_for(agg[combo]["train"]["holds"], agg[combo]["train"]["pairs"], combo)
        r_te = _stats_for(agg[combo]["test"]["holds"], agg[combo]["test"]["pairs"], combo)
        if r_tr:
            rows_train.append(r_tr)
        if r_te:
            rows_test.append(r_te)

    _print_table("IN-SAMPLE（train ≤2023）", rows_train)
    _print_table("OUT-OF-SAMPLE（test ≥2024）", rows_test)

    # 退出分布 + HTML（主组合）
    primary = f"{list(BUY_SETS)[0]}|STATE_SL2"
    for seg in ("train", "test"):
        pairs = agg[primary][seg]["pairs"]
        if pairs:
            print(f"\n[{primary} | {seg}] 退出分布: {_exit_dist(pairs)}")
            dfw = _to_dfw(agg[primary][seg]["holds"])
            if dfw is not None:
                try:
                    generate_backtest_report(
                        df=dfw,
                        output_path=str(OUTPUT_DIR / f"primary_{seg}.html"),
                        title=f"走势状态机 {primary} ({seg})",
                        fee_rate=FEE_RATE,
                        weight_type="ts",
                        yearly_days=YEARLY_DAYS,
                    )
                except Exception as e:
                    print(f"  HTML 失败: {e}")

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({"train": rows_train, "test": rows_test}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


def _split_by_seg(holds, pairs):
    """把一只票某 combo 的 holds/pairs 按交易入场段切成 train/test 两份。"""
    by_seg = {"train": {"holds": [], "pairs": []}, "test": {"holds": [], "pairs": []}}
    # holds 行已按交易入场段打 seg 标签（一笔交易整体归属其入场段，即使持有跨年）。
    for h in holds:
        by_seg[h["seg"]]["holds"].append(h)
    for p in pairs:
        by_seg[p["seg"]]["pairs"].append(p)
    return by_seg.items()


def _to_dfw(holds):
    if not holds:
        return None
    dfw = pd.DataFrame(holds)
    if dfw.duplicated(subset=["dt", "symbol"]).any():
        dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(weight=("weight", "max"), price=("price", "first"))
    return dfw[["dt", "symbol", "weight", "price"]]


def _print_table(title, rows):
    print("\n" + "=" * 120)
    print(f"  {title}")
    print("=" * 120)
    if not rows:
        print("  无数据")
        return
    cols = [
        "组合",
        "交易数",
        "胜率",
        "盈亏比",
        "平均收益%",
        "中位收益%",
        "平均持有",
        "年化%",
        "夏普",
        "卡玛",
        "最大回撤%",
    ]
    df = pd.DataFrame(rows)
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
