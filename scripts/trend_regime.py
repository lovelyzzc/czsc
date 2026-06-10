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

设计原则（对应用户「不得使用未来函数」硬约束）：

1. **流式重放**：用 ``CZSC(bars[:warmup])`` 暖机后逐 bar ``c.update(bar)``，
   在每个 bar 仅用「截至当前已收敛的 ``bi_list``」分类。已验证流式与
   ``CZSC(bars[:t+1])`` 全量构造 **逐字节一致**（见 ``--self-check``），因此
   每个 bar 的笔结构只含 ≤t 信息，不存在「最后一笔用未来 bar 确认」的泄漏。
2. **状态机（FSM）**：走势类型本质是有序演化，分类用 ``prev_regime`` 上下文做
   有限状态转移，而非对当前 bar 的无状态多分类——避免下跌途中误报「结构破坏」。
3. **原生中枢**：用 ``czsc._native.ZS(bis).is_valid()`` 识别中枢，替代各脚本手搓
   的 ``_extract_zs``。
4. **少量、理论驱动的阈值**：阈值集中为模块级常量，便于 OOS 冻结与敏感性扫描，
   不按指标网格搜最优。

被 ``trend_regime_backtest.py`` / ``surge_characteristics.py`` 复用。

自检（断言因果安全 + 打印状态时间线）：

    uv run --no-sync python scripts/trend_regime.py --self-check 000636.SZ
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from czsc import CZSC, Freq, format_standard_kline
from czsc._native import ZS

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"

# --------------------------------------------------------------------------- #
# 模块级常量（少量、理论驱动）
# --------------------------------------------------------------------------- #
MIN_BARS = 500  # 个股最少 bar 数（数据足够长才纳入）
WARMUP_BARS = 120  # 暖机 bar 数：之前一律 NotTradable
MIN_BIS = 6  # 分类所需最少笔数
LIMIT_PCT_MAIN = 9.8  # 主板涨跌停阈值（当日 |pct_chg|≥此值视为无法成交）
LIMIT_PCT_CHINEXT = 19.8  # 创业板 20cm 阈值（2020-08 注册制改革后，数据起点 2021-05 全程适用）
CHINEXT_PREFIX = ("300", "301", "302")  # 创业板代码段
ZS_MIN_BIS = 3  # 构成中枢的最少笔数
EXCLUDE_PREFIX = ("688", "920", "83", "43")  # 科创板/北交所（涨跌停规则不同）

ACCEL_SPREAD = 15.0  # 加速态：MA5-MA20 散度 (%)
ACCEL_RET20 = 15.0  # 加速态：20 日涨幅 (%)

# 主升浪启动门控（粗粒度、a-priori，刻意取在 surge_characteristics 全样本中位数
# 之下：量比 1.31 / 散度 5.09 / ret20 11.9 → 取 1.2 / 3.0 / 8.0；不按回测指标精调，防过拟合）
SURGE_GATE_VOL_RATIO = 1.2  # 放量
SURGE_GATE_MA_SPREAD = 3.0  # 均线发散 (%)
SURGE_GATE_RET20 = 8.0  # 埋伏变体额外要求的 20 日涨幅 (%)
SURGE_PRIOR_WINDOW = 40  # 启动前回看窗口（核验是否走过 中枢构造→离开）


class Regime(IntEnum):
    """11 种走势类型。数值即用户给定的编号 0..10。"""

    NotTradable = 0
    Downtrend = 1
    FirstBuy = 2
    SecondBuy = 3
    PivotBuilding = 4
    UpwardDeparture = 5
    ThirdBuy = 6
    MainUptrend = 7
    Acceleration = 8
    Divergence = 9
    Breakdown = 10


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

# 买点状态（右侧确认为主）/ 卖点状态（统一）
BUY_REGIMES_MAIN = frozenset({Regime.ThirdBuy})
BUY_REGIMES_WIDE = frozenset({Regime.UpwardDeparture, Regime.ThirdBuy})
SELL_REGIMES = frozenset({Regime.Divergence, Regime.Breakdown})
UPTREND_FAMILY = frozenset(
    {Regime.UpwardDeparture, Regime.ThirdBuy, Regime.MainUptrend, Regime.Acceleration, Regime.Divergence}
)


@dataclass(slots=True)
class StateSnapshot:
    """单个 bar 的因果分类快照（全部字段仅含 ≤idx 信息）。"""

    idx: int
    dt: pd.Timestamp
    regime: int
    prev_regime: int
    close: float
    next_open: float  # 次日开盘价（成交价，最后一根为 NaN）
    sl_ref: float  # 笔结构止损参考：最近向下笔低点（无则 NaN）
    zg: float  # 最近中枢上沿（无则 NaN）
    zd: float  # 最近中枢下沿（无则 NaN）
    feats: dict | None = None  # 可选：该 bar 的因果特征（with_features=True 时填充）


# --------------------------------------------------------------------------- #
# 数据加载 / 指标
# --------------------------------------------------------------------------- #
def limit_pct_for(code: str) -> float:
    """按板返回涨跌停判定阈值（创业板 20cm，其余 10cm；688/83/43 已被排除）。"""
    return LIMIT_PCT_CHINEXT if str(code).startswith(CHINEXT_PREFIX) else LIMIT_PCT_MAIN


def load_stock(parquet_path: str | Path) -> pd.DataFrame | None:
    """读取单只个股日线 qfq parquet，做基础过滤并标准化列名。

    返回带 ``symbol/dt`` 列、按日期升序的 DataFrame；不合格返回 None。
    """
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


def compute_indicators(df: pd.DataFrame) -> dict:
    """预计算因果指标（rolling/ewm 仅用过去值，天然因果）。"""
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
# 中枢提取（原生 ZS）
# --------------------------------------------------------------------------- #
def extract_zs_list(bis: list) -> list:
    """用原生 ``ZS(bis).is_valid()`` 把笔列表切成非重叠中枢，按时间升序返回。"""
    zs_list = []
    i = 0
    n = len(bis)
    while i <= n - ZS_MIN_BIS:
        zs = ZS(bis[i : i + ZS_MIN_BIS])
        if not zs.is_valid():
            i += 1
            continue
        k = i + ZS_MIN_BIS
        while k < n:
            grown = ZS(bis[i : k + 1])
            if grown.is_valid():
                zs, k = grown, k + 1
            else:
                break
        zs_list.append(zs)
        i = k
    return zs_list


def _dir_up(bi) -> bool:
    return bi.direction.value == "向上"


def _dif_extreme_over(dates, dif, sdt, edt, want_max: bool) -> float:
    """笔时间区间 [sdt, edt] 内 DIF 的极值（背驰的 MACD 度量）。"""
    lo = int(np.searchsorted(dates, np.datetime64(sdt)))
    hi = int(np.searchsorted(dates, np.datetime64(edt), side="right"))
    if hi <= lo:
        return float(dif[min(lo, len(dif) - 1)])
    seg = dif[lo:hi]
    return float(seg.max() if want_max else seg.min())


# --------------------------------------------------------------------------- #
# 走势类型 FSM
# --------------------------------------------------------------------------- #
def classify_fsm(prev: Regime, bis: list, zs_list: list, ind: dict, idx: int) -> Regime:
    """给定前一状态 + 截至 idx 的因果笔结构，返回当前走势类型。

    纯函数：仅依赖 ``prev`` 与 ≤idx 的数据，便于单测与敏感性扫描。
    """
    up = [b for b in bis if _dir_up(b)]
    dn = [b for b in bis if not _dir_up(b)]
    last = bis[-1]
    zs = zs_list[-1] if zs_list else None

    c = ind["close"][idx]
    ma5, ma10, ma20 = ind["ma5"][idx], ind["ma10"][idx], ind["ma20"][idx]
    dif, dates = ind["dif"], ind["dates"]

    # ---- 条件（全部仅用 ≤idx 数据）---------------------------------------- #
    # 顶背驰：价创新高，但笔力度 + MACD 不配合
    top_div = False
    if len(up) >= 2 and up[-1].high > up[-2].high and up[-1].power < up[-2].power:
        dif_now = _dif_extreme_over(dates, dif, up[-1].sdt, up[-1].edt, want_max=True)
        dif_prev = _dif_extreme_over(dates, dif, up[-2].sdt, up[-2].edt, want_max=True)
        top_div = dif_now < dif_prev

    # 底背驰：价创新低，但笔力度 + MACD 不配合
    bot_div = False
    if len(dn) >= 2 and dn[-1].low < dn[-2].low and dn[-1].power < dn[-2].power:
        dif_now = _dif_extreme_over(dates, dif, dn[-1].sdt, dn[-1].edt, want_max=False)
        dif_prev = _dif_extreme_over(dates, dif, dn[-2].sdt, dn[-2].edt, want_max=False)
        bot_div = dif_now > dif_prev

    # 二买：回调向下笔低点抬升（不破前低）且当前已转上
    second_buy = len(dn) >= 2 and dn[-1].low > dn[-2].low and _dir_up(last)
    # 三买：离开中枢后回踩低点不回中枢上沿
    third_buy = zs is not None and dn and dn[-1].low > zs.zg and _dir_up(last)
    # 有效中枢且价在区间内
    in_pivot = zs is not None and zs.zd <= c <= zs.zg
    # 向上突破中枢上沿
    breakout_up = zs is not None and c > zs.zg
    # 多头排列 + 笔力度占优
    bull_stack = ma5 > ma10 > ma20
    up_dom = bool(up and dn and up[-1].power >= np.mean([b.power for b in dn[-2:]]))
    main_up = bull_stack and up_dom and (zs is None or c > zs.zg)
    # 加速：MA 扩散 + 涨幅陡升，且笔力度或角度较前一向上笔加速
    accel = (
        len(up) >= 2
        and (up[-1].power > up[-2].power or up[-1].angle > up[-2].angle)
        and ma20 > 0
        and (ma5 - ma20) / ma20 * 100 > ACCEL_SPREAD
        and ind["ret20"][idx] > ACCEL_RET20
    )
    # 趋势反转：向下笔力度反超最近向上笔且当前处于向下笔
    trend_rev = bool(dn and up and dn[-1].power > up[-1].power and not _dir_up(last))
    # 震荡/中枢向下解决：收盘跌破中枢下沿（→ 回到下跌，而非结构破坏）
    down_resolve = zs is not None and c < zs.zd
    # 上升结构破坏：跌回中枢上沿（突破失败/支撑失守）或趋势反转
    bd_up = (zs is not None and c < zs.zg) or trend_rev

    # ---- 有限状态转移 ----------------------------------------------------- #
    # 下跌/筑底阶段：1/2/3，可进入 4 中枢
    if prev in (Regime.Downtrend, Regime.FirstBuy, Regime.SecondBuy):
        if breakout_up and third_buy:
            return Regime.ThirdBuy
        if breakout_up:
            return Regime.UpwardDeparture
        if in_pivot:
            return Regime.PivotBuilding
        if bot_div:
            return Regime.FirstBuy
        if second_buy:
            return Regime.SecondBuy
        return Regime.Downtrend

    # 中枢构造：4 —— 向上突破→5；向下解决→下跌；否则继续构造
    if prev == Regime.PivotBuilding:
        if breakout_up:
            return Regime.UpwardDeparture
        if down_resolve:
            return Regime.Downtrend
        return Regime.PivotBuilding

    # 向上离开中枢：5
    if prev == Regime.UpwardDeparture:
        if bd_up:
            return Regime.Breakdown
        if third_buy:
            return Regime.ThirdBuy
        if accel:
            return Regime.Acceleration
        if main_up:
            return Regime.MainUptrend
        return Regime.UpwardDeparture

    # 主升家族：6/7/8 —— 先判破坏/背驰再判加速/延续
    if prev in (Regime.ThirdBuy, Regime.MainUptrend, Regime.Acceleration):
        if bd_up:
            return Regime.Breakdown
        if top_div:
            return Regime.Divergence
        if accel:
            return Regime.Acceleration
        if main_up:
            return Regime.MainUptrend
        return prev if prev != Regime.ThirdBuy else Regime.MainUptrend

    # 背驰衰竭：9
    if prev == Regime.Divergence:
        if bd_up:
            return Regime.Breakdown
        if accel and not top_div:
            return Regime.Acceleration
        if main_up and not top_div:
            return Regime.MainUptrend
        return Regime.Divergence

    # 结构破坏：10 —— 复位到下跌，除非快速重夺中枢
    if prev == Regime.Breakdown:
        if breakout_up and third_buy:
            return Regime.ThirdBuy
        if breakout_up:
            return Regime.UpwardDeparture
        if in_pivot:
            return Regime.PivotBuilding
        return Regime.Downtrend

    return Regime.Downtrend


def compute_features(bis: list, zs_list: list, ind: dict, idx: int) -> dict:
    """该 bar 的因果结构特征（供主升浪特征研究使用，全部仅用 ≤idx 数据）。"""
    up = [b for b in bis if _dir_up(b)]
    dn = [b for b in bis if not _dir_up(b)]
    zs = zs_list[-1] if zs_list else None
    ma5, ma20 = ind["ma5"][idx], ind["ma20"][idx]
    vol = ind["vol"]
    up_pow = float(np.mean([b.power for b in up[-3:]])) if up else 0.0
    dn_pow = float(np.mean([b.power for b in dn[-3:]])) if dn else 0.0
    vol_ratio = float(vol[idx] / vol[idx - 20 : idx].mean()) if idx >= 20 and vol[idx - 20 : idx].mean() > 0 else np.nan
    return {
        "up_dn_power_ratio": round(up_pow / dn_pow, 3) if dn_pow > 0 else np.nan,
        "last_up_angle": round(up[-1].angle, 2) if up else np.nan,
        "ma_spread_pct": round((ma5 - ma20) / ma20 * 100, 2) if ma20 > 0 else np.nan,
        "dif": round(float(ind["dif"][idx]), 4),
        "vol_ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else np.nan,
        "n_pivots": len(zs_list),
        "pivot_width_pct": round((zs.zg - zs.zd) / zs.zd * 100, 2) if zs is not None and zs.zd > 0 else np.nan,
        "ret20": round(float(ind["ret20"][idx]), 2),
        "above_zg": int(zs is not None and ind["close"][idx] > zs.zg),
    }


def _gates_pass(feats: dict | None) -> bool:
    """主升浪启动的特征门控（放量 + 均线发散 + 立于中枢上方）。"""
    if not feats:
        return False
    vr = feats.get("vol_ratio")
    sp = feats.get("ma_spread_pct")
    return (
        vr is not None
        and not (isinstance(vr, float) and np.isnan(vr))
        and vr >= SURGE_GATE_VOL_RATIO
        and sp is not None
        and not (isinstance(sp, float) and np.isnan(sp))
        and sp >= SURGE_GATE_MA_SPREAD
        and int(feats.get("above_zg", 0)) == 1
    )


def surge_onset(prev: int, regime: int, feats: dict | None, prior_regimes, mode: str = "confirm") -> bool:
    """因果「主升浪启动」检测（仅用 ≤t 数据）。

    - ``confirm``  确认追入：状态跳变进入 7/8 且通过门控，且启动前窗口走过 4 与 5。
    - ``anticipate`` 启动埋伏：状态跳变进入 5 且通过更强门控（加 ret20）且窗口走过 4。
    """
    prior = set(prior_regimes)
    if mode == "confirm":
        entered = prev not in (Regime.MainUptrend, Regime.Acceleration) and regime in (
            Regime.MainUptrend,
            Regime.Acceleration,
        )
        path_ok = Regime.PivotBuilding in prior and Regime.UpwardDeparture in prior
        return bool(entered and path_ok and _gates_pass(feats))
    if mode == "anticipate":
        entered = prev != Regime.UpwardDeparture and regime == Regime.UpwardDeparture
        path_ok = Regime.PivotBuilding in prior
        ret_ok = feats is not None and (feats.get("ret20") or 0) >= SURGE_GATE_RET20
        return bool(entered and path_ok and ret_ok and _gates_pass(feats))
    raise ValueError(f"未知 mode: {mode}")


def surge_score(feats: dict | None) -> float:
    """主升浪强度打分 0..100（供 skill 排序，不参与是否触发判定）。"""
    if not feats:
        return 0.0

    def _v(key, default=0.0):
        x = feats.get(key)
        return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

    # 量比、均线散度、20日涨幅、向上笔角度，各自截断后线性合成
    s = (
        min(_v("vol_ratio") / 2.0, 1.0) * 30  # 量比 2.0 封顶
        + min(_v("ma_spread_pct") / 15.0, 1.0) * 30  # 散度 15% 封顶
        + min(max(_v("ret20"), 0) / 30.0, 1.0) * 20  # ret20 30% 封顶
        + min(max(_v("last_up_angle"), 0) / 45.0, 1.0) * 20  # 角度 45° 封顶
    )
    return round(s, 1)


# 当前状态质量分（priority_score 的组成部分；加速>三买>主升>离开）
REGIME_QUALITY = {Regime.Acceleration: 20, Regime.ThirdBuy: 18, Regime.MainUptrend: 16, Regime.UpwardDeparture: 12}
# 止损幅度奖励带：满分区间与选股硬过滤一致（8-20%），过近/过远递减
STOP_FULL_BAND = (8.0, 20.0)
STOP_PARTIAL_BAND = (5.0, 30.0)


def priority_score(score: float, sl_pct: float | None, freshness: int, regime: int, scan_window: int = 10) -> float:
    """选股/回测共用的综合优先级 = 主升强度(35) + 止损可控(25) + 新鲜度(20) + 状态质量(20)。

    单一真源：daily_scan 排序与组合回测的候选排序都走这里，保证「回测的就是实盘排的」。
    """
    p = min(score, 100.0) * 0.35
    if sl_pct is None or (isinstance(sl_pct, float) and np.isnan(sl_pct)) or sl_pct <= 0:
        p -= 30  # 止损穿透/无效
    elif STOP_FULL_BAND[0] <= sl_pct <= STOP_FULL_BAND[1]:
        p += 25
    elif STOP_PARTIAL_BAND[0] <= sl_pct < STOP_FULL_BAND[0] or STOP_FULL_BAND[1] < sl_pct <= STOP_PARTIAL_BAND[1]:
        p += 15
    else:
        p += 5
    p += max(0.0, (scan_window - freshness) / scan_window) * 20
    p += REGIME_QUALITY.get(Regime(regime), 10)
    return round(max(p, 0.0), 1)


def _seed_regime(bis: list, zs_list: list, ind: dict, idx: int) -> Regime:
    """暖机结束时的初始状态：用一次无状态启发式播种，避免 FSM 起步偏差。"""
    zs = zs_list[-1] if zs_list else None
    c = ind["close"][idx]
    ma5, ma10, ma20 = ind["ma5"][idx], ind["ma10"][idx], ind["ma20"][idx]
    if zs is not None and zs.zd <= c <= zs.zg:
        return Regime.PivotBuilding
    if ma5 > ma10 > ma20 and (zs is None or c > zs.zg):
        return Regime.MainUptrend
    return Regime.Downtrend


# --------------------------------------------------------------------------- #
# 流式重放 → 状态序列
# --------------------------------------------------------------------------- #
def iter_states(
    df: pd.DataFrame, freq: Freq = Freq.D, with_features: bool = False, tail: int | None = None
) -> list[StateSnapshot]:
    """流式重放整只个股，返回每个 bar 的因果状态快照。

    - ``with_features=True``：为每个可交易 bar 附 ``feats``（结构特征字典）。
    - ``tail``：仅暖机到 ``n-tail``、流式分类最后 ``tail`` 根（每日选股快路径，
      对「今日」无未来数据仍因果）；``None`` 则全程从 WARMUP 开始（回测用）。
    """
    ind = compute_indicators(df)
    n = ind["n"]
    bars = format_standard_kline(df, freq=freq)
    if n <= WARMUP_BARS:
        return []
    limit_pct = limit_pct_for(df["symbol"].iloc[0])

    start = WARMUP_BARS if tail is None else max(WARMUP_BARS, n - tail)
    czsc = CZSC(bars[:start])  # 暖机（= 全量构造 bars[:start]，已验证因果一致）
    out: list[StateSnapshot] = []
    prev = Regime.NotTradable
    seeded = False

    for idx in range(start, n):
        czsc.update(bars[idx])  # 处理后 bi_list 反映 bars[:idx+1]（已验证因果）
        bis = czsc.bi_list

        not_tradable = len(bis) < MIN_BIS or abs(ind["pct_chg"][idx]) >= limit_pct or ind["vol"][idx] <= 0
        if not_tradable:
            regime = Regime.NotTradable
            zs_list, dn = [], []
        else:
            zs_list = extract_zs_list(bis)
            if not seeded:
                prev = _seed_regime(bis, zs_list, ind, idx)
                seeded = True
            regime = classify_fsm(prev, bis, zs_list, ind, idx)
            dn = [b for b in bis if not _dir_up(b)]

        zs = zs_list[-1] if zs_list else None
        feats = compute_features(bis, zs_list, ind, idx) if with_features and regime != Regime.NotTradable else None
        out.append(
            StateSnapshot(
                idx=idx,
                dt=pd.Timestamp(ind["dates"][idx]),
                regime=int(regime),
                prev_regime=int(prev),
                close=float(ind["close"][idx]),
                next_open=float(ind["open"][idx + 1]) if idx + 1 < n else float("nan"),
                sl_ref=float(dn[-1].low) if dn else float("nan"),
                zg=float(zs.zg) if zs is not None else float("nan"),
                zd=float(zs.zd) if zs is not None else float("nan"),
                feats=feats,
            )
        )
        if regime != Regime.NotTradable:
            prev = regime

    return out


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

    # —— 状态时间线 ——
    states = iter_states(df)
    counts = pd.Series([s.regime for s in states]).value_counts().sort_index()
    print("\n[状态分布]")
    for r, cnt in counts.items():
        print(f"  {r:>2} {REGIME_CN[Regime(r)]:<12} {cnt:>5}  ({cnt / len(states) * 100:4.1f}%)")

    print("\n[状态切换时间线]（仅展示状态发生变化的 bar）")
    prev = None
    shown = 0
    for s in states:
        if s.regime != prev:
            arrow = "★买" if s.regime in BUY_REGIMES_WIDE else ("☆卖" if s.regime in SELL_REGIMES else "  ")
            print(
                f"  {s.dt.date()}  {Regime(s.prev_regime).name:>16} → "
                f"{Regime(s.regime).name:<16} {arrow}  close={s.close:.2f}"
            )
            prev = s.regime
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
