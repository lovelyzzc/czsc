"""走势类型划分（11 态）—— 因果安全的缠论状态机核心模块

把个股日线走势分类为 0..10 共 11 个走势类型：

    0  NotTradable       不可交易（暖机不足 / 涨跌停 / 停牌）
    1  Downtrend         下跌走势
    2  FirstBuy          一买观察（下跌底背驰）
    3  SecondBuy         二买转强（回调不破前低）
    4  PivotBuilding     中枢构造（≥3 笔有效重叠、区间震荡）
    5  UpwardDeparture   向上离开中枢（收盘突破中枢上沿）
    6  ThirdBuy          三买确认（回踩不回中枢）
    7  MainUptrend       主升延续（多头排列、逐笔抬升）
    8  Acceleration      加速主升（笔力度/角度加速 + MA 扩散）
    9  Divergence        背驰衰竭（价创新高、力度与 MACD 不配合）
    10 Breakdown         结构破坏（跌破笔低点/中枢下沿/趋势反转）

核心计算（FSM / 指标 / 中枢提取 / 特征 / 评分）全部由 Rust 实现
（``czsc._native.iter_regime_states`` 等），Python 端纯透传。

被 ``trend_regime_backtest.py`` / ``surge_characteristics.py`` 复用。

自检（断言因果安全 + 打印状态时间线）：

    uv run --no-sync python scripts/trend_regime.py --self-check 000636.SZ
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from czsc import CZSC, Freq, format_standard_kline
from czsc._native.trend_regime import (
    FeatureSnapshot,
    Regime,
    StateSnapshot,
    iter_regime_states,
    py_priority_score as priority_score,
    py_surge_onset as surge_onset,
    py_surge_score as surge_score,
)

ALL_REGIMES = tuple(Regime.from_int(i) for i in range(11))

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"

# --------------------------------------------------------------------------- #
# 模块级常量
# --------------------------------------------------------------------------- #
MIN_BARS = 500
WARMUP_BARS = 120
MIN_BIS = 6
LIMIT_PCT_MAIN = 9.8
LIMIT_PCT_CHINEXT = 19.8
CHINEXT_PREFIX = ("300", "301", "302")
EXCLUDE_PREFIX = ("688", "920", "83", "43")

SURGE_PRIOR_WINDOW = 40

REGIME_CN = {
    Regime.NotTradable: "不可交易",
    Regime.Downtrend: "下跌走势",
    Regime.FirstBuy: "一买观察",
    Regime.SecondBuy: "二买转强",
    Regime.PivotBuilding: "中枢构造",
    Regime.UpwardDeparture: "向上离开中枢",
    Regime.ThirdBuy: "三买确认",
    Regime.MainUptrend: "主升延续",
    Regime.Acceleration: "加速主升",
    Regime.Divergence: "背驰衰竭",
    Regime.Breakdown: "结构破坏",
}

BUY_REGIMES_MAIN = frozenset({int(Regime.ThirdBuy)})
BUY_REGIMES_WIDE = frozenset({int(Regime.UpwardDeparture), int(Regime.ThirdBuy)})
SELL_REGIMES = frozenset({int(Regime.Divergence), int(Regime.Breakdown)})
UPTREND_FAMILY = frozenset(
    {int(Regime.UpwardDeparture), int(Regime.ThirdBuy), int(Regime.MainUptrend), int(Regime.Acceleration), int(Regime.Divergence)}
)


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def limit_pct_for(code: str) -> float:
    """按板返回涨跌停判定阈值（创业板 20cm，其余 10cm；688/83/43 已被排除）。"""
    return LIMIT_PCT_CHINEXT if str(code).startswith(CHINEXT_PREFIX) else LIMIT_PCT_MAIN


def load_stock(parquet_path: str | Path) -> pd.DataFrame | None:
    """读取单只个股日线 qfq parquet，做基础过滤并标准化列名。"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None
    if len(df) < MIN_BARS:
        return None
    code = df["ts_code"].iloc[0]
    if str(code).startswith(EXCLUDE_PREFIX):
        return None
    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 向后兼容：部分脚本需要按列名访问指标数组
# --------------------------------------------------------------------------- #
def compute_indicators(df: pd.DataFrame) -> dict:
    """预计算因果指标（向后兼容旧 Python 接口，非性能路径上的消费脚本可继续使用）。"""
    import numpy as np

    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy()
    ret20 = np.full(n, 0.0)
    if n > 20:
        ret20[20:] = (close[20:] / close[:-20] - 1.0) * 100.0
    return {
        "close": close,
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "vol": df["vol"].to_numpy(dtype=float) if "vol" in df else np.ones(n),
        "pct_chg": df["pct_chg"].to_numpy(dtype=float) if "pct_chg" in df else np.zeros(n),
        "dates": df["dt"].to_numpy(),
        "dif": ema12 - ema26,
        "ma5": pd.Series(close).rolling(5).mean().to_numpy(),
        "ma10": pd.Series(close).rolling(10).mean().to_numpy(),
        "ma20": pd.Series(close).rolling(20).mean().to_numpy(),
        "ret20": ret20,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# iter_states — 委托给 Rust
# --------------------------------------------------------------------------- #
def iter_states(
    df: pd.DataFrame, freq: Freq = Freq.D, with_features: bool = False, tail: int | None = None
) -> list[StateSnapshot]:
    """流式重放整只个股，返回每个 bar 的因果状态快照。

    底层调用 Rust ``iter_regime_states``，消除逐 bar Python-Rust 边界穿越。
    """
    bars = format_standard_kline(df, freq=freq)
    if len(bars) <= WARMUP_BARS:
        return []
    limit_pct = limit_pct_for(df["symbol"].iloc[0])
    return iter_regime_states(bars, freq, with_features, tail, limit_pct, WARMUP_BARS, MIN_BIS)


# --------------------------------------------------------------------------- #
# 自检：因果安全断言 + 状态时间线
# --------------------------------------------------------------------------- #
def _self_check(symbol: str) -> int:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        print(f"[ERROR] 找不到数据：{path}")
        return 1
    df = load_stock(path)
    if df is None:
        print(f"[ERROR] {symbol} 数据不合格（bar 数 < {MIN_BARS} 或属排除板）")
        return 1

    bars = format_standard_kline(df, freq=Freq.D)
    n = len(bars)
    print(f"[{symbol}] bars={n}  {df['dt'].iloc[0].date()} → {df['dt'].iloc[-1].date()}")

    # —— 因果断言：流式 @t 的 bi_list 必须等于 batch CZSC(bars[:t+1]).bi_list ——
    czsc = CZSC(bars[:WARMUP_BARS])
    sample = list(range(WARMUP_BARS, n, max(1, (n - WARMUP_BARS) // 40)))
    sample_set = set(sample)
    checked = 0
    for idx in range(WARMUP_BARS, n):
        czsc.update(bars[idx])
        if idx in sample_set:
            ref = CZSC(bars[: idx + 1]).bi_list
            cur = czsc.bi_list

            def _key(bl):
                return [(b.sdt, b.edt, round(b.high, 4), round(b.low, 4), b.direction.value) for b in bl]

            assert _key(cur) == _key(ref), f"因果泄漏：流式与全量在 idx={idx} 笔结构不一致"
            checked += 1
    print(f"[因果断言] 通过 ✓ —— {checked} 个采样 bar 上「流式 bi_list == 全量构造 bi_list」")

    # —— 状态时间线（Rust 路径）——
    states = iter_states(df)
    counts = pd.Series([s.regime for s in states]).value_counts().sort_index()
    print("\n[状态分布]")
    for r, cnt in counts.items():
        rname = REGIME_CN.get(Regime.from_int(r), f"?{r}")
        print(f"  {r:>2} {rname:<12} {cnt:>5}  ({cnt / len(states) * 100:4.1f}%)")

    print("\n[状态切换时间线]（仅展示状态发生变化的 bar）")
    prev_r = None
    shown = 0
    for s in states:
        if s.regime != prev_r:
            regime_e = Regime.from_int(s.regime)
            prev_e = Regime.from_int(s.prev_regime)
            arrow = (
                "★买"
                if s.regime in BUY_REGIMES_WIDE
                else ("☆卖" if s.regime in SELL_REGIMES else "  ")
            )
            print(
                f"  {s.dt.date()}  {prev_e!r:>24} → "
                f"{regime_e!r:<24} {arrow}  close={s.close:.2f}"
            )
            prev_r = s.regime
            shown += 1
            if shown >= 60:
                print("  ...（已截断）")
                break
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--self-check":
        return _self_check(argv[1])
    print(__doc__)
    print("用法: python scripts/trend_regime.py --self-check <SYMBOL，如 000636.SZ>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
