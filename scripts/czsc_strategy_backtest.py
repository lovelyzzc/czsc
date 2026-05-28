"""新策略研究回测（全 A 股日线数据）

抛弃传统缠论一买/二买/三买/背驰，利用 CZSC 框架中 bar/tas/vol/jcc/pressure 等
信号模块构建 6 种新策略，覆盖突破、趋势、反转、均值回归等不同交易流派。

策略列表：
    1. 波动率压缩突破 — 窄幅整理 + 放量突破新高
    2. MACD零轴共振 — DIF零轴附近二次金叉 + 均线多头排列
    3. TD序列反转 — 神奇九转买点 + RSI超卖确认
    4. 量价背离反转 — 量价极值买点 + 看涨吞没形态
    5. 布林KDJ超卖反弹 — 布林下轨极值 + KDJ超卖转多
    6. 双均线趋势回踩 — 双均线强势多头 + 支撑位 + 相对低位

数据源：~/.ts_data_cache/a_stock_daily_qfq/ 下的 Tushare 日线前复权 parquet
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import pandas as pd
from wbt import generate_backtest_report

from czsc import (
    CzscStrategyBase,
    Event,
    Position,
    WeightBacktest,
    format_standard_kline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "strategy_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
FREQ = "日线"

_ZDT = f"{FREQ}_D1_涨跌停V230331_涨停_任意_任意_0"
_BI_DOWN = f"{FREQ}_D1_表里关系V230101_向下_任意_任意_0"

INTERVAL = 3600 * 24 * 5
TIMEOUT = 120
STOP_LOSS = 500


def _exit_bi_down() -> Event:
    return Event.load({"name": "笔向下_平多", "operate": "平多", "signals_all": [_BI_DOWN]})


# ==================== 策略 1: 波动率压缩突破 ====================

_VOL_SQUEEZE_OPEN = [
    f"{FREQ}_窄幅震荡N5_形态V241013_满足_任意_任意_0",
    f"{FREQ}_D1W20_事件V240428_收盘新高_任意_任意_0",
    f"{FREQ}_D1K_量柱V221216_梯量_价升_任意_0",
]


class VolSqueezeStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="波动率压缩突破_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "窄幅放量突破_开多", "operate": "开多",
                "signals_all": _VOL_SQUEEZE_OPEN, "signals_not": [_ZDT],
            })],
            exits=[_exit_bi_down()],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "bar_zfzd_V241013", "freq": FREQ, "n": 5},
            {"name": "bar_break_V240428", "freq": FREQ, "di": 1, "w": 20},
            {"name": "vol_ti_suo_V221216", "freq": FREQ, "di": 1},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 2: MACD 零轴共振 ====================

_MACD_ZERO_OPEN = [
    f"{FREQ}_D1N100MD1J2S0_MACD交叉数量V230625_0轴下第2次金叉以后_任意_任意_0",
    f"{FREQ}_DIF分层W60T20_完全分类V241010_零轴附近_任意_任意_0",
    f"{FREQ}_D1SMA5#10#20_均线系统V230513_多头排列_任意_任意_0",
]
_MACD_EXIT = f"{FREQ}_D1MACD12#26#9#MACD_BS辅助V221028_空头_向下_任意_0"


class MACDZeroStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="MACD零轴共振_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "零轴二次金叉_开多", "operate": "开多",
                "signals_all": _MACD_ZERO_OPEN, "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "MACD死叉_平多", "operate": "平多", "signals_all": [_MACD_EXIT]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "tas_cross_status_V230625", "freq": FREQ, "di": 1,
             "n": 100, "md": 1, "j": 2, "s": 0},
            {"name": "tas_dif_layer_V241010", "freq": FREQ, "w": 60, "t": 20},
            {"name": "tas_ma_system_V230513", "freq": FREQ, "di": 1, "ma_seq": "5#10#20"},
            {"name": "tas_macd_base_V221028", "freq": FREQ, "di": 1,
             "fastperiod": 12, "slowperiod": 26, "signalperiod": 9, "key": "MACD"},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 3: TD 序列反转 ====================

_TD9_BUY = f"{FREQ}_神奇九转N9_BS辅助V240616_买点_任意_任意_0"
_RSI_OVERSOLD = f"{FREQ}_D1T30RSI14_RSI辅助V230227_超卖_任意_任意_0"
_TD9_SELL = f"{FREQ}_神奇九转N9_BS辅助V240616_卖点_任意_任意_0"


class TD9Strategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="TD序列反转_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "TD9超卖_开多", "operate": "开多",
                "signals_all": [_TD9_BUY, _RSI_OVERSOLD], "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "TD卖点_平多", "operate": "平多", "signals_all": [_TD9_SELL]}),
            ],
            interval=INTERVAL, timeout=60, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "bar_td9_V240616", "freq": FREQ, "n": 9},
            {"name": "tas_rsi_base_V230227", "freq": FREQ, "di": 1, "th": 30, "timeperiod": 14},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 4: 量价背离极值反转 ====================

_VOLPRICE_BS1 = f"{FREQ}_D1N20量价_BS1辅助_看多_任意_任意_0"
_ENGULF_BULL = f"{FREQ}_D1_吞没形态_满足_看涨吞没_任意_0"


class VolPriceRevStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="量价背离反转_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "量价极值吞没_开多", "operate": "开多",
                "signals_all": [_VOLPRICE_BS1, _ENGULF_BULL], "signals_not": [_ZDT],
            })],
            exits=[_exit_bi_down()],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "bar_vol_bs1_V230224", "freq": FREQ, "di": 1, "n": 20},
            {"name": "jcc_ten_mo_V221028", "freq": FREQ, "di": 1},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 5: 布林 + KDJ 超卖反弹 ====================

_BOLL_BEAR_EXTREME = f"{FREQ}_D1BOLL20_强弱V221112_空头_超强_任意_0"
_KDJ_BULL = f"{FREQ}_D1T20KDJ9#3#3#K值突破1#3_BS辅助V230401_多头_任意_任意_0"
_BOLL_BULL = f"{FREQ}_D1BOLL20_强弱V221112_多头_任意_任意_0"


class BollKDJStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="布林KDJ超卖反弹_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "布林超卖KDJ转多_开多", "operate": "开多",
                "signals_all": [_BOLL_BEAR_EXTREME, _KDJ_BULL], "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "布林转多_平多", "operate": "平多", "signals_all": [_BOLL_BULL]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "tas_boll_power_V221112", "freq": FREQ, "di": 1, "timeperiod": 20},
            {"name": "tas_kdj_evc_V230401", "freq": FREQ, "di": 1, "th": 20, "key": "K",
             "fastk_period": 9, "slowk_period": 3, "slowd_period": 3,
             "min_count": 1, "max_count": 3},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 6: 双均线趋势回踩 + 支撑位 ====================

_DUAL_MA_BULL = f"{FREQ}_D1T100#SMA#5#20_JX辅助V221203_多头_强势_任意_0"
_SUPPORT = f"{FREQ}_D1W20_支撑压力V240402_支撑位_任意_任意_0"
_REL_LOW = f"{FREQ}_N20_BS辅助V240328_相对低点_任意_任意_0"
_DUAL_MA_BEAR = f"{FREQ}_D1T100#SMA#5#20_JX辅助V221203_空头_任意_任意_0"


class DualMAPullbackStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="双均线趋势回踩_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "均线强势回踩支撑_开多", "operate": "开多",
                "signals_all": [_DUAL_MA_BULL, _SUPPORT, _REL_LOW],
                "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "均线转空_平多", "operate": "平多", "signals_all": [_DUAL_MA_BEAR]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=400, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "tas_double_ma_V221203", "freq": FREQ, "di": 1,
             "th": 100, "ma_type": "SMA", "timeperiod1": 5, "timeperiod2": 20},
            {"name": "pressure_support_V240402", "freq": FREQ, "di": 1, "w": 20},
            {"name": "xl_bar_position_V240328", "freq": FREQ, "n": 20},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略注册 ====================

STRATEGY_TAGS = [
    "1_波动率压缩突破", "2_MACD零轴共振", "3_TD序列反转",
    "4_量价背离反转", "5_布林KDJ超卖反弹", "6_双均线趋势回踩",
]
STRATEGY_CLASSES = [
    VolSqueezeStrategy, MACDZeroStrategy, TD9Strategy,
    VolPriceRevStrategy, BollKDJStrategy, DualMAPullbackStrategy,
]


# ==================== 子进程入口 ====================

HOLDS_DIR = OUTPUT_DIR / "_holds_tmp"


def _process_stock(parquet_path: str) -> dict | None:
    """单只股票全策略回测，holds 写入磁盘 parquet，返回轻量摘要"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if len(df) < MIN_BARS:
        return None

    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    try:
        bars = format_standard_kline(df, freq=FREQ)
    except Exception:
        return None

    symbol = bars[0].symbol
    n_bars = len(bars)
    sdt = bars[n_bars // 4].dt.strftime("%Y-%m-%d")

    result = {"symbol": symbol}
    holds_parts = []
    for tag, cls in zip(STRATEGY_TAGS, STRATEGY_CLASSES):
        try:
            strategy = cls(symbol=symbol)
            res = strategy.backtest(bars, sdt=sdt)
            pairs = res.pairs_df()
            holds = res.holds_df()

            info = {"pairs": len(pairs)}
            if not pairs.empty and "盈亏比例" in pairs.columns:
                profits = pairs["盈亏比例"]
                info["win"] = int((profits > 0).sum())
                info["loss"] = int((profits <= 0).sum())
            else:
                info["win"] = 0
                info["loss"] = 0

            if not holds.empty:
                h = holds[["dt", "symbol", "pos", "price"]].copy()
                h["tag"] = tag
                holds_parts.append(h)

            result[tag] = info
        except Exception:
            continue

    if holds_parts:
        combined = pd.concat(holds_parts, ignore_index=True)
        safe_name = symbol.replace(".", "_")
        combined.to_parquet(HOLDS_DIR / f"{safe_name}.parquet", index=False)

    return result if len(result) > 1 else None


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("  新策略研究回测 — 6 大策略（全 A 股日线数据）")
    print("=" * 70)

    shutil.rmtree(HOLDS_DIR, ignore_errors=True)
    HOLDS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] 发现 {len(parquet_files)} 只个股 ({DATA_DIR})")

    if not parquet_files:
        print("[ERROR] 未找到任何 parquet 文件")
        return

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] 使用 {n_workers} 个进程 (spawn 模式)")

    t_start = time.time()
    file_list = [str(p) for p in parquet_files]

    all_results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_stock, file_list, chunksize=20), 1):
            if res is not None:
                all_results.append(res)
            if i % 500 == 0 or i == len(file_list):
                elapsed = time.time() - t_start
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(file_list)}] 有效 {len(all_results)} | "
                      f"耗时 {elapsed:.0f}s | {speed:.1f} 只/s | ETA {eta:.0f}s")

    total_elapsed = time.time() - t_start
    print(f"\n[回测完成] 总耗时 {total_elapsed:.0f}s | 有效 {len(all_results)}/{len(file_list)} 只")

    # Phase 1: 汇总交易对统计
    pair_stats: dict[str, dict] = {}
    for tag in STRATEGY_TAGS:
        ps = {"win": 0, "loss": 0, "pairs": 0, "stocks": 0}
        for r in all_results:
            if tag in r:
                info = r[tag]
                ps["pairs"] += info["pairs"]
                ps["win"] += info.get("win", 0)
                ps["loss"] += info.get("loss", 0)
                if info["pairs"] > 0:
                    ps["stocks"] += 1
        pair_stats[tag] = ps

    # Phase 2: 读取 holds parquet 文件，按策略跑 WeightBacktest
    print("\n[Phase 2] 读取 holds 文件，运行 WeightBacktest ...")
    holds_files = sorted(HOLDS_DIR.glob("*.parquet"))
    print(f"  holds 文件: {len(holds_files)} 个")

    all_holds = []
    for f in holds_files:
        try:
            all_holds.append(pd.read_parquet(f))
        except Exception:
            continue

    if not all_holds:
        print("[ERROR] 无 holds 数据")
        return

    holds_combined = pd.concat(all_holds, ignore_index=True)
    del all_holds
    print(f"  holds 总行数: {len(holds_combined):,}")

    all_stats = []
    for tag in STRATEGY_TAGS:
        print(f"\n{'='*60}")
        print(f"  [{tag}]")
        print(f"{'='*60}")

        ps = pair_stats[tag]
        tag_holds = holds_combined[holds_combined["tag"] == tag].copy()

        if tag_holds.empty:
            print("  [WARN] 无持仓数据")
            continue

        dfw = tag_holds[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"})
        if dfw.duplicated(subset=["dt", "symbol"]).any():
            dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
                weight=("weight", "mean"), price=("price", "first"),
            )
        dfw = dfw[["dt", "symbol", "weight", "price"]]

        try:
            wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
            stats = wb.stats
        except Exception as e:
            print(f"  [ERROR] WeightBacktest 失败: {e}")
            continue

        stats["tag"] = tag
        stats["stocks_count"] = ps["stocks"]
        stats["pairs_count"] = ps["pairs"]
        stats["elapsed_s"] = round(total_elapsed, 1)

        total = ps["win"] + ps["loss"]
        if total > 0:
            stats["pair_win_rate"] = round(ps["win"] / total * 100, 1)

        print(f"  覆盖: {ps['stocks']} 只股票 | {ps['pairs']} 笔交易")
        for k in ["年化收益", "夏普比率", "最大回撤", "卡玛比率", "pair_win_rate"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            out_html = OUTPUT_DIR / f"{tag}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"新策略研究 - {tag}（全A股日线）",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"  HTML: {out_html.name} ({out_html.stat().st_size/1024:.1f} KB)")
        except Exception as e:
            print(f"  [WARN] HTML 报告生成失败: {e}")

        all_stats.append(stats)

    del holds_combined

    if not all_stats:
        print("\n[ERROR] 所有策略都未产生有效结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 80)
    print("  策略对比汇总（全 A 股日线）")
    print("=" * 80)
    display_cols = [c for c in [
        "stocks_count", "pairs_count", "pair_win_rate",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "comparison_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    shutil.rmtree(HOLDS_DIR, ignore_errors=True)

    print(f"\n[文件] {OUTPUT_DIR / 'comparison_stats.json'}")
    print(f"[完成] 所有策略回测完成")


if __name__ == "__main__":
    main()
