"""主升浪策略多角度优化回测

基线 + 5 个独立优化方向 + 最优组合，控制变量逐一对比。

方向 A：市场环境过滤器（大盘多头时才入场）
方向 B：混合退出策略（SL2 兜底 + 浮盈跟踪止损）
方向 C：动态笔止损（持仓期间上移止损位）
方向 D：回调入场（等待回调至 MA10 附近）
方向 E：仓位管理（高分加仓）

数据源：~/.ts_data_cache/a_stock_daily_qfq/
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from wbt import generate_backtest_report

from czsc import CZSC, Freq, WeightBacktest, format_standard_kline

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_wave"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
SCORE_THRESHOLD = 5
MAX_HOLD_DAYS = 60

MARKET_PROXY_CODES = [
    "600519.SH", "601318.SH", "601398.SH", "600036.SH", "000858.SZ",
]


# ═══════════════════════════════════════════════════════════════════════
#  共用工具函数
# ═══════════════════════════════════════════════════════════════════════

def _extract_zs(bis_list):
    zs_list = []
    i = 0
    while i < len(bis_list) - 2:
        b1, b2, b3 = bis_list[i], bis_list[i + 1], bis_list[i + 2]
        zg = min(b1.high, b2.high, b3.high)
        zd = max(b1.low, b2.low, b3.low)
        if zg > zd:
            zs = {"zg": zg, "zd": zd, "bis": 3}
            j = i + 3
            while j < len(bis_list):
                bj = bis_list[j]
                if bj.high >= zd and bj.low <= zg:
                    zs["bis"] += 1
                    j += 1
                else:
                    break
            zs_list.append(zs)
            i = j
        else:
            i += 1
    return zs_list


def _score_surge_raw(bis_up_to, zs_up_to, close_val, dif_now, dif_5ago,
                     ma5, ma10, ma20, ret20):
    """返回 S1-S7 各项 0/1 列表"""
    feats = [0] * 7
    up_bis = [b for b in bis_up_to if b.direction.value == "向上"]
    dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]

    if len(up_bis) >= 3:
        p1, p2, p3 = up_bis[-3].power, up_bis[-2].power, up_bis[-1].power
        feats[0] = int(p3 > p2 > p1 and p1 > 0)

    if len(up_bis) >= 2 and len(dn_bis) >= 2:
        up_avg = np.mean([b.power for b in up_bis[-3:]])
        dn_avg = np.mean([b.power for b in dn_bis[-3:]])
        feats[1] = int(up_avg > dn_avg * 1.5) if dn_avg > 0 else 0

    if zs_up_to:
        feats[2] = int(close_val > zs_up_to[-1]["zg"] * 1.2)

    if len(dn_bis) >= 2:
        feats[3] = int(dn_bis[-1].low > dn_bis[-2].low)

    if ma5 > ma10 > ma20:
        spread = (ma5 - ma20) / ma20 * 100
        feats[4] = int(spread > 15)
    else:
        feats[4] = 0

    feats[5] = int(dif_now > 0 and dif_now > dif_5ago)
    feats[6] = int(ret20 > 15)

    return feats


def _prepare_stock(parquet_path):
    """公用的数据加载 + CZSC 构建，返回 (df, bis, code) 或 None"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if len(df) < MIN_BARS:
        return None

    code = df["ts_code"].iloc[0]
    if code.startswith(("688", "920", "83", "43")):
        return None

    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    try:
        bars = format_standard_kline(df, freq=Freq.D)
        c = CZSC(bars)
    except Exception:
        return None

    if len(c.bi_list) < 10:
        return None

    return df, c.bi_list, code


def _compute_indicators(df):
    """预计算技术指标，返回 dict of arrays"""
    close = df["close"].values
    n = len(df)
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    return {
        "close": close,
        "high": df["high"].values,
        "low": df["low"].values,
        "pct_chg": df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(n),
        "dates": df["dt"].values,
        "dif": ema12 - ema26,
        "ma5": pd.Series(close).rolling(5).mean().values,
        "ma10": pd.Series(close).rolling(10).mean().values,
        "ma20": pd.Series(close).rolling(20).mean().values,
        "n": n,
    }


def _find_signals(bis, ind, start_idx):
    """扫描产生 score>=THRESHOLD 的信号列表"""
    close, dates, dif = ind["close"], ind["dates"], ind["dif"]
    ma5, ma10, ma20 = ind["ma5"], ind["ma10"], ind["ma20"]
    pct_chg, n = ind["pct_chg"], ind["n"]

    signals = []
    for idx in range(start_idx, n):
        if abs(pct_chg[idx]) >= 9.8:
            continue

        dt = dates[idx]
        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dt)]
        if len(bis_up_to) < 8:
            continue

        zs_up_to = _extract_zs(bis_up_to)
        dif_5ago = dif[idx - 5] if idx >= 5 else 0
        ret20 = (close[idx] / close[idx - 20] - 1) * 100 if idx >= 20 else 0

        feats = _score_surge_raw(
            bis_up_to, zs_up_to, close[idx],
            dif[idx], dif_5ago, ma5[idx], ma10[idx], ma20[idx], ret20,
        )
        total = sum(feats)
        if total >= SCORE_THRESHOLD:
            dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]
            up_bis = [b for b in bis_up_to if b.direction.value == "向上"]
            signals.append({
                "idx": idx, "close": close[idx], "total": total,
                "sl_bi": dn_bis[-1].low if dn_bis else None,
                "last_up_power": up_bis[-1].power if up_bis else None,
                "dt_str": str(dt)[:10],
            })
    return signals


def _dedup_signals(signals, gap=15):
    if not signals:
        return []
    filtered = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - filtered[-1]["idx"] >= gap:
            filtered.append(s)
    return filtered


# ═══════════════════════════════════════════════════════════════════════
#  市场环境过滤器 (方向 A)
# ═══════════════════════════════════════════════════════════════════════

def build_market_regime() -> dict:
    """用 5 只大盘股构建逐日市场环境: {date_str: True=多头}"""
    all_series = []
    for code in MARKET_PROXY_CODES:
        p = DATA_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["dt"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("dt").set_index("dt")
        c = df["close"]
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        bull = ((c > ma20) & (ma20 > ma60)).astype(int)
        all_series.append(bull)

    if not all_series:
        return {}

    combined = pd.concat(all_series, axis=1).dropna()
    vote = combined.mean(axis=1)
    regime = (vote >= 0.6).to_dict()  # >= 3/5 stocks bullish
    return {str(k.date()): v for k, v in regime.items()}


# ═══════════════════════════════════════════════════════════════════════
#  各模式的 worker 函数
# ═══════════════════════════════════════════════════════════════════════

def _worker_baseline(parquet_path: str) -> dict | None:
    """基线：等权 thr=5 + SL2 + max_hold=60"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup_signals(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})
            if sl_price and low_arr[j] <= sl_price and j > entry_idx:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_A(args) -> dict | None:
    """方向 A：市场环境过滤"""
    parquet_path, regime = args
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    raw_signals = _find_signals(bis, ind, start_idx)
    # 过滤：只保留大盘多头日的信号
    filtered = [s for s in raw_signals if regime.get(s["dt_str"], False)]
    signals = _dedup_signals(filtered)
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})
            if sl_price and low_arr[j] <= sl_price and j > entry_idx:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_B(parquet_path: str) -> dict | None:
    """方向 B：混合退出（SL2 + 分级跟踪止损）"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup_signals(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        peak = entry_price
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            if close[j] > peak:
                peak = close[j]

            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})

            if j == entry_idx:
                continue

            # SL2 兜底
            if sl_price and low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_reason = "sl2"
                break

            gain_pct = (peak / entry_price - 1)
            drawdown_from_peak = (peak - close[j]) / peak if peak > 0 else 0

            # 浮盈 >= 25%: 收紧到 10% 跟踪止损
            if gain_pct >= 0.25 and drawdown_from_peak >= 0.10:
                exit_price = close[j]
                exit_reason = "trail_10%"
                break

            # 浮盈 >= 10%: 15% 跟踪止损
            if gain_pct >= 0.10 and drawdown_from_peak >= 0.15:
                exit_price = close[j]
                exit_reason = "trail_15%"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_C(parquet_path: str) -> dict | None:
    """方向 C：动态笔止损（持仓期间上移止损位）"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup_signals(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        current_sl = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})

            if j == entry_idx:
                continue

            # 更新止损位：检查是否有更新的向下笔完成
            dt_j = pd.Timestamp(dates[j])
            dn_bis_now = [b for b in bis
                          if b.direction.value == "向下" and b.edt <= dt_j]
            if dn_bis_now:
                new_sl = dn_bis_now[-1].low
                if current_sl is None or new_sl > current_sl:
                    current_sl = new_sl

            if current_sl and low_arr[j] <= current_sl:
                exit_price = current_sl
                exit_reason = "dynamic_sl"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_D(parquet_path: str) -> dict | None:
    """方向 D：回调入场（等待价格回到 MA10 附近）"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    raw_signals = _find_signals(bis, ind, start_idx)
    raw_signals = _dedup_signals(raw_signals)
    if not raw_signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    ma10 = ind["ma10"]
    n = ind["n"]
    WAIT_DAYS = 10

    holds, pairs = [], []
    for sig in raw_signals:
        sig_idx = sig["idx"]
        actual_entry_idx = None
        actual_entry_price = None

        # 在信号日后 WAIT_DAYS 内等待回调到 MA10
        for w in range(sig_idx + 1, min(sig_idx + WAIT_DAYS + 1, n)):
            if close[w] <= ma10[w] * 1.02:
                actual_entry_idx = w
                actual_entry_price = close[w]
                break

        if actual_entry_idx is None:
            continue

        sl_price = sig["sl_bi"]
        exit_price, exit_reason = None, "max_hold"

        for j in range(actual_entry_idx, min(actual_entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})

            if j == actual_entry_idx:
                continue

            if sl_price and low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(actual_entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / actual_entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    if not holds:
        return None
    return {"holds": holds, "pairs": pairs}


def _worker_E(parquet_path: str) -> dict | None:
    """方向 E：仓位管理（5/7=0.5, 6/7=1.0, 7/7=2.0）"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    signals = _dedup_signals(_find_signals(bis, ind, start_idx))
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]

    SCORE_WEIGHT_MAP = {5: 0.5, 6: 1.0, 7: 2.0}

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        pos_weight = SCORE_WEIGHT_MAP.get(sig["total"], 1.0)
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": pos_weight, "price": close[j]})

            if j == entry_idx:
                continue

            if sl_price and low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_combo(args) -> dict | None:
    """组合模式：A(市场过滤) + B(混合退出) + E(仓位管理)"""
    parquet_path, regime = args
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    raw_signals = _find_signals(bis, ind, start_idx)
    filtered = [s for s in raw_signals if regime.get(s["dt_str"], False)]
    signals = _dedup_signals(filtered)
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]
    SCORE_WEIGHT_MAP = {5: 0.5, 6: 1.0, 7: 2.0}

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        sl_price = sig["sl_bi"]
        pos_weight = SCORE_WEIGHT_MAP.get(sig["total"], 1.0)
        peak = entry_price
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            if close[j] > peak:
                peak = close[j]

            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": pos_weight, "price": close[j]})

            if j == entry_idx:
                continue

            if sl_price and low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_reason = "sl2"
                break

            gain_pct = (peak / entry_price - 1)
            dd = (peak - close[j]) / peak if peak > 0 else 0

            if gain_pct >= 0.25 and dd >= 0.10:
                exit_price = close[j]
                exit_reason = "trail_10%"
                break

            if gain_pct >= 0.10 and dd >= 0.15:
                exit_price = close[j]
                exit_reason = "trail_15%"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


def _worker_BD(parquet_path: str) -> dict | None:
    """组合 B+D：回调入场 + 混合退出"""
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    raw_signals = _find_signals(bis, ind, start_idx)
    raw_signals = _dedup_signals(raw_signals)
    if not raw_signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    ma10 = ind["ma10"]
    n = ind["n"]
    WAIT_DAYS = 10

    holds, pairs = [], []
    for sig in raw_signals:
        sig_idx = sig["idx"]
        actual_entry_idx = None
        actual_entry_price = None

        for w in range(sig_idx + 1, min(sig_idx + WAIT_DAYS + 1, n)):
            if close[w] <= ma10[w] * 1.02:
                actual_entry_idx = w
                actual_entry_price = close[w]
                break

        if actual_entry_idx is None:
            continue

        sl_price = sig["sl_bi"]
        peak = actual_entry_price
        exit_price, exit_reason = None, "max_hold"

        for j in range(actual_entry_idx, min(actual_entry_idx + MAX_HOLD_DAYS, n)):
            if close[j] > peak:
                peak = close[j]

            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": 1, "price": close[j]})

            if j == actual_entry_idx:
                continue

            if sl_price and low_arr[j] <= sl_price:
                exit_price = sl_price
                exit_reason = "sl2"
                break

            gain_pct = (peak / actual_entry_price - 1)
            dd = (peak - close[j]) / peak if peak > 0 else 0

            if gain_pct >= 0.25 and dd >= 0.10:
                exit_price = close[j]
                exit_reason = "trail_10%"
                break

            if gain_pct >= 0.10 and dd >= 0.15:
                exit_price = close[j]
                exit_reason = "trail_15%"
                break

        if exit_price is None:
            last_j = min(actual_entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / actual_entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    if not holds:
        return None
    return {"holds": holds, "pairs": pairs}


def _worker_combo2(args) -> dict | None:
    """组合模式 2：A(市场过滤) + C(动态止损) + E(仓位管理)"""
    parquet_path, regime = args
    prepared = _prepare_stock(parquet_path)
    if prepared is None:
        return None
    df, bis, code = prepared
    ind = _compute_indicators(df)
    start_idx = max(120, ind["n"] // 4)

    raw_signals = _find_signals(bis, ind, start_idx)
    filtered = [s for s in raw_signals if regime.get(s["dt_str"], False)]
    signals = _dedup_signals(filtered)
    if not signals:
        return None

    close, low_arr, dates = ind["close"], ind["low"], ind["dates"]
    n = ind["n"]
    SCORE_WEIGHT_MAP = {5: 0.5, 6: 1.0, 7: 2.0}

    holds, pairs = [], []
    for sig in signals:
        entry_idx, entry_price = sig["idx"], sig["close"]
        current_sl = sig["sl_bi"]
        pos_weight = SCORE_WEIGHT_MAP.get(sig["total"], 1.0)
        exit_price, exit_reason = None, "max_hold"

        for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
            holds.append({"dt": pd.Timestamp(dates[j]), "symbol": code,
                          "weight": pos_weight, "price": close[j]})

            if j == entry_idx:
                continue

            dt_j = pd.Timestamp(dates[j])
            dn_bis_now = [b for b in bis
                          if b.direction.value == "向下" and b.edt <= dt_j]
            if dn_bis_now:
                new_sl = dn_bis_now[-1].low
                if current_sl is None or new_sl > current_sl:
                    current_sl = new_sl

            if current_sl and low_arr[j] <= current_sl:
                exit_price = current_sl
                exit_reason = "dynamic_sl"
                break

        if exit_price is None:
            last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
            exit_price = close[last_j]

        ret = (exit_price / entry_price - 1) * 100
        pairs.append({"ret_pct": round(ret, 2), "exit_reason": exit_reason})

    return {"holds": holds, "pairs": pairs}


# ═══════════════════════════════════════════════════════════════════════
#  运行引擎
# ═══════════════════════════════════════════════════════════════════════

def _run_mode(tag, worker_fn, file_list, n_workers, use_regime=False, regime=None):
    """通用回测运行：收集 holds/pairs → WeightBacktest → stats"""
    print(f"\n{'='*80}")
    print(f"  [{tag}]")
    print(f"{'='*80}")

    t0 = time.time()
    all_holds, all_pairs = [], []

    if use_regime:
        args_list = [(f, regime) for f in file_list]
    else:
        args_list = file_list

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        iterator = pool.imap_unordered(worker_fn, args_list, chunksize=20)
        for i, res in enumerate(iterator, 1):
            if res is not None:
                all_holds.extend(res["holds"])
                all_pairs.extend(res["pairs"])
            if i % 1000 == 0:
                print(f"    [{i}/{len(file_list)}] ...")

    elapsed = time.time() - t0

    if not all_holds:
        print(f"    无持仓数据 ({elapsed:.0f}s)")
        return None

    dfw = pd.DataFrame(all_holds)
    if dfw.duplicated(subset=["dt", "symbol"]).any():
        dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
            weight=("weight", "max"), price=("price", "first"),
        )
    dfw = dfw[["dt", "symbol", "weight", "price"]]

    try:
        wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
        stats = wb.stats
    except Exception as e:
        print(f"    WeightBacktest 失败: {e} ({elapsed:.0f}s)")
        return None

    rets = [p["ret_pct"] for p in all_pairs]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    n_exit = {}
    for p in all_pairs:
        r = p["exit_reason"]
        n_exit[r] = n_exit.get(r, 0) + 1

    stats["tag"] = tag
    stats["退出分布"] = str(n_exit)
    stats["交易笔数"] = len(rets)
    stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
    stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
    stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
    stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
    stats["平均收益"] = f"{np.mean(rets):.2f}%"
    stats["收益中位数"] = f"{np.median(rets):.2f}%"

    for k in ["交易笔数", "退出分布", "胜率", "盈亏比",
              "平均盈利", "平均亏损", "平均收益", "收益中位数",
              "年化收益", "夏普比率", "最大回撤", "卡玛比率"]:
        if k in stats:
            print(f"    {k}: {stats[k]}")

    try:
        out_html = OUTPUT_DIR / f"opt_{tag}.html"
        generate_backtest_report(
            df=dfw, output_path=str(out_html),
            title=f"主升浪优化 - {tag}",
            fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
        )
        print(f"    HTML: {out_html.name}")
    except Exception as e:
        print(f"    HTML 失败: {e}")

    print(f"    耗时: {elapsed:.0f}s")
    return stats


def main():
    t_start = time.time()
    print("=" * 100)
    print("  主升浪策略 — 多角度优化回测")
    print("=" * 100)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    file_list = [str(p) for p in parquet_files]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(parquet_files)} 只个股 | {n_workers} 进程")

    # 预构建市场环境
    print("\n[准备] 构建大盘市场环境过滤器...")
    regime = build_market_regime()
    bull_days = sum(1 for v in regime.values() if v)
    total_days = len(regime)
    print(f"  大盘多头日: {bull_days}/{total_days} "
          f"({bull_days/total_days*100:.1f}%)" if total_days > 0 else "  无数据")

    all_stats = []

    # ── 基线 ──
    stats = _run_mode("0_基线", _worker_baseline, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ── 方向 A：市场环境过滤 ──
    stats = _run_mode("A_市场过滤", _worker_A, file_list, n_workers,
                      use_regime=True, regime=regime)
    if stats:
        all_stats.append(stats)

    # ── 方向 B：混合退出 ──
    stats = _run_mode("B_混合退出", _worker_B, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ── 方向 C：动态笔止损 ──
    stats = _run_mode("C_动态止损", _worker_C, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ── 方向 D：回调入场 ──
    stats = _run_mode("D_回调入场", _worker_D, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ── 方向 E：仓位管理 ──
    stats = _run_mode("E_仓位管理", _worker_E, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ── 组合 1: A + B + E ──
    stats = _run_mode("F_组合ABE", _worker_combo, file_list, n_workers,
                      use_regime=True, regime=regime)
    if stats:
        all_stats.append(stats)

    # ── 组合 2: A + C + E ──
    stats = _run_mode("G_组合ACE", _worker_combo2, file_list, n_workers,
                      use_regime=True, regime=regime)
    if stats:
        all_stats.append(stats)

    # ── 组合 3: B + D ──
    stats = _run_mode("H_组合BD", _worker_BD, file_list, n_workers)
    if stats:
        all_stats.append(stats)

    # ═══════════════════════════════════════════════════════════════════
    #  汇总对比
    # ═══════════════════════════════════════════════════════════════════
    if not all_stats:
        print("\n[ERROR] 所有模式均无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 120)
    print("  主升浪策略 — 多角度优化对比汇总")
    print("=" * 120)
    display_cols = [c for c in [
        "交易笔数", "胜率", "盈亏比",
        "平均盈利", "平均亏损", "平均收益", "收益中位数",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "optimization_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    # 找最优
    best = max(all_stats, key=lambda s: s.get("卡玛比率", 0))
    print(f"\n  卡玛最优: {best['tag']} "
          f"(年化 {best.get('年化收益', 'N/A')} / "
          f"夏普 {best.get('夏普比率', 'N/A')} / "
          f"卡玛 {best.get('卡玛比率', 'N/A')} / "
          f"回撤 {best.get('最大回撤', 'N/A')})")

    best_sharpe = max(all_stats, key=lambda s: s.get("夏普比率", 0))
    print(f"  夏普最优: {best_sharpe['tag']} "
          f"(夏普 {best_sharpe.get('夏普比率', 'N/A')})")

    print(f"\n[完成] 总耗时 {time.time()-t_start:.0f}s")
    print(f"[文件] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
