"""趋势起爆到加速阶段策略研究（全 A 股日线数据）

核心理念：趋势生命周期为 底部整理 → 起爆 → 加速 → 巅峰 → 衰竭。
本研究聚焦"起爆到加速"窗口，用 4 种不同维度捕捉同一现象：

策略列表：
    1. MACD转势起爆 — DIF穿零轴 + MACD方向向上 + 价格加速上涨
    2. 三连阳放量起爆 — 三K新高涨依次放量 + 均线多头排列
    3. 笔趋势动量加速 — 笔上升趋势 + MACD强势 + 区间价格加速
    4. 8K结构量能爆发 — 8K上涨结构 + 区间动量强势 + 成交量放大

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


# ==================== 策略 1: MACD 转势起爆 ====================
# DIF 穿越零轴（趋势刚转多）+ MACD 柱子方向向上（动量增强）+ 价格拟合加速上涨

_DIF_NEAR_ZERO = f"{FREQ}_DIF分层W60T20_完全分类V241010_零轴附近_任意_任意_0"
_MACD_DIR_UP = f"{FREQ}_D1K#MACD12#26#9方向_BS辅助V221106_向上_任意_任意_0"
_POLYFIT_ACC_UP = f"{FREQ}_D1W20_分类V240428_加速上涨_任意_任意_0"
_MACD_DIR_DOWN = f"{FREQ}_D1K#MACD12#26#9方向_BS辅助V221106_向下_任意_任意_0"


class MACDIgnitionStrategy(CzscStrategyBase):
    """DIF 零轴转势 + MACD 向上 + 价格加速"""

    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="MACD转势起爆_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "DIF零轴+MACD向上+加速_开多", "operate": "开多",
                "signals_all": [_DIF_NEAR_ZERO, _MACD_DIR_UP, _POLYFIT_ACC_UP],
                "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "MACD转向_平多", "operate": "平多",
                            "signals_all": [_MACD_DIR_DOWN]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "tas_dif_layer_V241010", "freq": FREQ, "w": 60, "t": 20},
            {"name": "tas_macd_direct_V221106", "freq": FREQ, "di": 1,
             "fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            {"name": "bar_polyfit_V240428", "freq": FREQ, "di": 1, "w": 20},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 2: 三连阳放量起爆 ====================
# 三根K线创新高 + 依次放量（最直观的起爆形态）+ 均线多头排列确认方向

_TRIPLE_BULL_VOL = f"{FREQ}_D1三K加速_裸K形态V230506_新高涨_依次放量_任意_0"
_MA_BULL_ALIGN = f"{FREQ}_D1SMA5#10#20_均线系统V230513_多头排列_任意_任意_0"


class TripleBullStrategy(CzscStrategyBase):
    """三K新高涨放量 + 均线多头排列"""

    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="三连阳放量起爆_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "三连阳放量+均线多头_开多", "operate": "开多",
                "signals_all": [_TRIPLE_BULL_VOL, _MA_BULL_ALIGN],
                "signals_not": [_ZDT],
            })],
            exits=[_exit_bi_down()],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "bar_triple_V230506", "freq": FREQ, "di": 1},
            {"name": "tas_ma_system_V230513", "freq": FREQ, "di": 1, "ma_seq": "5#10#20"},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 3: 笔趋势 + 动量加速 ====================
# 笔高低点形成上升趋势 + MACD 强势（不是超强，避免追高）+ 价格加速

_BI_UPTREND = f"{FREQ}_D4N1笔趋势_高低点辅助判断V230913_上升趋势_任意_任意_0"
_MACD_STRONG = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_强势_任意_任意_0"
_PRICE_ACC = f"{FREQ}_D1W10_加速V221110_上涨_任意_任意_0"
_MACD_WEAK = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_弱势_任意_任意_0"


class BiTrendAccelStrategy(CzscStrategyBase):
    """笔上升趋势 + MACD 强势 + 价格加速"""

    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="笔趋势动量加速_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "笔上升+MACD强势+加速_开多", "operate": "开多",
                "signals_all": [_BI_UPTREND, _MACD_STRONG, _PRICE_ACC],
                "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "MACD转弱_平多", "operate": "平多",
                            "signals_all": [_MACD_WEAK]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "cxt_bi_trend_V230913", "freq": FREQ, "di": 4, "n": 1},
            {"name": "tas_macd_power_V221108", "freq": FREQ, "di": 1,
             "fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            {"name": "bar_accelerate_V221110", "freq": FREQ, "di": 1, "window": 10},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略 4: 趋势跟踪 + 绝对动量加速 ====================
# 60 日趋势跟踪确认多头 + 绝对动量达到强势 + 价格拟合加速上涨

_TREND_BULL = f"{FREQ}_D1N60趋势跟踪_BS辅助V240209_多头_任意_任意_0"
_ABS_MOMENTUM = f"{FREQ}_D1N20T1000_绝对动量V230227_强势_任意_任意_0"
_TREND_BEAR = f"{FREQ}_D1N60趋势跟踪_BS辅助V240209_空头_任意_任意_0"


class TrendMomentumAccelStrategy(CzscStrategyBase):
    """60 日趋势多头 + 绝对动量强势 + 价格加速上涨"""

    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="趋势动量加速_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "趋势多头+动量强+加速_开多", "operate": "开多",
                "signals_all": [_TREND_BULL, _ABS_MOMENTUM, _POLYFIT_ACC_UP],
                "signals_not": [_ZDT],
            })],
            exits=[
                _exit_bi_down(),
                Event.load({"name": "趋势转空_平多", "operate": "平多",
                            "signals_all": [_TREND_BEAR]}),
            ],
            interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "bar_trend_V240209", "freq": FREQ, "di": 1, "N": 60},
            {"name": "bar_bpm_V230227", "freq": FREQ, "di": 1, "n": 20, "th": 1000},
            {"name": "bar_polyfit_V240428", "freq": FREQ, "di": 1, "w": 20},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


# ==================== 策略注册 ====================

STRATEGY_TAGS = [
    "1_MACD转势起爆", "2_三连阳放量起爆", "3_笔趋势动量加速", "4_趋势动量加速",
]
STRATEGY_CLASSES = [
    MACDIgnitionStrategy, TripleBullStrategy, BiTrendAccelStrategy, TrendMomentumAccelStrategy,
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
    print("  趋势起爆到加速 — 4 大策略回测（全 A 股日线数据）")
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
                title=f"趋势起爆加速 - {tag}（全A股日线）",
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
    print("  策略对比汇总（趋势起爆到加速 · 全 A 股日线）")
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
