"""Surge regime 实验模式（delay5）的 live 共享逻辑 —— daily_scan 与镜像核对共用的单一真源

提供三件事（全部因果，仅用 ≤t 数据）：

1. ``build_live_panel``：按 ``trend_regime.load_stock`` 同口径读全市场 parquet 尾部，
   构建 (symbol, dt, close, amount_e) 面板（市场状态计算的输入）。
2. ``live_market_state``：直接调用 ``surge_market_state_filter.compute_market_state``
   （不复制公式），返回市场状态时间序列；``market_gate_open`` 给出
   `high20_ratio > 0.12 & ew_index_above_ma20` 的门判定（前推验证选定门）。
3. ``detect_delay5``：在状态序列上检测「anticipate 信号后第 5 个交易日、今日仍处
   上行家族」的实验候选（与 surge_candidates_dump 的 (mode=anticipate, delay=5)
   行选股口径一致；收益模拟/gap 过滤不在此处——gap 是次日开盘执行时规则）。

注意：dump 口径的上行家族是 {5,6,7,8}（不含 9 背驰），与 ``trend_regime.UPTREND_FAMILY``
（含 9）不同，此处必须用前者。
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import trend_regime as tr
from trend_regime import Regime

PANEL_TAIL_BARS = 80  # ret20/high20 需 21 根、等权指数 MA20 需再 20 根 → 80 充裕
EXP_DELAY = 5  # 信号后第 5 个交易日收盘决策、次日开盘入场
MARKET_GATE_HIGH20 = 0.12  # 前推验证选定门（沿用，不调参）
# dump 口径上行家族（不含 9 背驰；tr.UPTREND_FAMILY 含 9，不可混用）
EXP_UPTREND_FAMILY = frozenset(
    {int(Regime.UpwardDeparture), int(Regime.ThirdBuy), int(Regime.MainUptrend), int(Regime.Acceleration)}
)


# --------------------------------------------------------------------------- #
# 市场状态（live）
# --------------------------------------------------------------------------- #
def _panel_tail_one(parquet_path: str) -> pd.DataFrame | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    t = df.tail(PANEL_TAIL_BARS)
    return pd.DataFrame(
        {
            "symbol": t["symbol"],
            "dt": t["dt"],
            "close": t["close"].astype(float),
            "amount_e": (t["amount"].astype(float) / 1e5) if "amount" in t.columns else np.nan,
        }
    )


def build_live_panel(files: list[str] | None = None, processes: int = 8) -> pd.DataFrame:
    """全市场尾部面板（与研究 panel 同源同口径，仅截取尾部以提速）。"""
    if files is None:
        files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    ctx = mp.get_context("spawn")
    parts = []
    with ctx.Pool(min(mp.cpu_count(), processes)) as pool:
        for res in pool.imap_unordered(_panel_tail_one, files, chunksize=50):
            if res is not None:
                parts.append(res)
    return pd.concat(parts, ignore_index=True)


def live_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    """市场状态时间序列（单一真源：msf.compute_market_state）。

    尾部面板下，前 ~40 个交易日因 rolling 暖机无效，调用方只应使用尾部
    （最后 ``PANEL_TAIL_BARS - 41`` 个交易日）的行。
    """
    state = msf.compute_market_state(panel)
    return state.dropna(subset=["mkt_ret20_median", "ew_index_ma20"]).reset_index(drop=True)


def market_gate_open(state_row: pd.Series | dict) -> bool:
    """前推验证选定的市场状态门：high20_ratio > 0.12 且 等权指数在 MA20 上方。"""
    return bool(float(state_row["high20_ratio"]) > MARKET_GATE_HIGH20 and float(state_row["ew_index_above_ma20"]) > 0)


# --------------------------------------------------------------------------- #
# delay5 实验候选检测（与 dump 的 mode=anticipate, delay=5 行同口径）
# --------------------------------------------------------------------------- #
def detect_delay5(states: list[tr.StateSnapshot]) -> dict | None:
    """今日（states[-1]）是否为某 anticipate 信号的第 5 个交易日决策日。

    条件（与 surge_candidates_dump + surge_pullback_entry_research 的选股口径一致）：
    信号 bar p = 今日-5 处发生 anticipate 启动（含信号日门控，由 surge_onset 内置），
    且今日 regime ∈ {5,6,7,8}。返回决策日（今日）视角的候选字段；不含收益模拟。
    """
    last_i = len(states) - 1
    p = last_i - EXP_DELAY
    if p < 1:
        return None
    dec = states[last_i]
    if dec.regime not in EXP_UPTREND_FAMILY:
        return None
    regimes = [s.regime for s in states]
    prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
    sig = states[p]
    if not tr.surge_onset(states[p - 1].regime, sig.regime, sig.feats, prior, "anticipate"):
        return None

    sl = dec.sl_ref if dec.sl_ref == dec.sl_ref else dec.zd  # NaN 退化用中枢下沿
    sl = sl if (sl == sl and sl > 0) else None
    sl_pct = round((dec.close - sl) / dec.close * 100, 2) if sl else None
    score = tr.surge_score(dec.feats)
    feats = dec.feats or {}
    sig_feats = sig.feats or {}
    return {
        "sig_dt": sig.dt,
        "dec_dt": dec.dt,
        "dec_regime": int(dec.regime),
        "close": float(dec.close),
        "sl_ref": float(sl) if sl else None,
        "sl_pct": sl_pct,
        "score": score,
        # 镜像口径：研究端 delayed 入场的 priority 也用 freshness=0
        "priority": tr.priority_score(score, sl_pct, 0, dec.regime),
        "vol_ratio": feats.get("vol_ratio"),
        "ma_spread_pct": feats.get("ma_spread_pct"),
        "ret20": feats.get("ret20"),
        "sig_vol_ratio": sig_feats.get("vol_ratio"),
        "sig_ma_spread_pct": sig_feats.get("ma_spread_pct"),
        "sig_ret20": sig_feats.get("ret20"),
    }


def amount_to_e(amount: float) -> float:
    """成交额（千元）→ 亿元，统一 round 3（与 surge_candidates_dump 的 amount_e 同口径）。

    daily_scan / 镜像核对 / dump 三处必须用同一舍入，否则 1 亿阈值附近
    pass_hard 判定会在镜像验证与真实扫描之间漂移。
    """
    return round(amount / 1e5, 3) if amount == amount and amount > 0 else 0.0


def hard_filter_reasons(sl_pct: float | None, amount_e: float, *, min_amount_e: float = 1.0) -> list[str]:
    """硬过滤（与 daily_scan / 回测镜像一致）：止损带 8-20%、成交额 ≥1 亿。ST 由调用方判。"""
    reasons = []
    if sl_pct is None:
        reasons.append("无有效止损")
    elif sl_pct <= 0:
        reasons.append("止损已穿透")
    elif not 8.0 <= sl_pct <= 20.0:
        reasons.append("止损幅度不在8-20%")
    if amount_e < min_amount_e:
        reasons.append(f"成交额<{min_amount_e:g}亿")
    return reasons


PANEL_PATH = Path(__file__).resolve().parent / "_output" / "surge_regime_picks" / "market_state_live.parquet"


def append_market_state_log(state_tail: pd.DataFrame, path: Path = PANEL_PATH) -> None:
    """把 live 计算的市场状态尾部 append 到前向审计日志（同 dt 覆盖，免疫日后 qfq 漂移争议）。"""
    cols = [c for c in state_tail.columns if c != "ew_ret1"]
    new = state_tail[cols].copy()
    new["scanned_at"] = pd.Timestamp.now().floor("s")
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old[~old["dt"].isin(new["dt"])], new], ignore_index=True).sort_values("dt")
    else:
        merged = new
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
