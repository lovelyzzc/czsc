"""缠论 5 大策略统一回测（真实 A 股日线数据）

策略列表：
    1. 一买策略 — 趋势反转（逆势抄底）
    2. 二买策略 — 趋势确认（回踩买入）
    3. 三买策略 — 中枢突破（顺势追涨）
    4. 笔趋势跟踪 — 最简单的趋势策略
    5. 背驰策略 — 多笔形态 + MACD 辅助

数据源：~/.ts_data_cache/a_stock_daily_qfq/ 下的 Tushare 日线前复权 parquet
无分钟级数据，多级别联立策略跳过。
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
_DIETING = f"{FREQ}_D1_涨跌停V230331_跌停_任意_任意_0"
_BI_DOWN = f"{FREQ}_D1_表里关系V230101_向下_任意_任意_0"
_BI_UP = f"{FREQ}_D1_表里关系V230101_向上_任意_任意_0"

INTERVAL = 3600 * 24 * 5
TIMEOUT = 120
STOP_LOSS = 500


def _exit_bi_down() -> Event:
    return Event.load({
        "name": "笔向下_平多",
        "operate": "平多",
        "signals_all": [_BI_DOWN],
    })


# ==================== 策略定义 ====================

def build_first_buy_position(symbol: str) -> Position:
    return Position(
        name="一买策略_多头", symbol=symbol,
        opens=[Event.load({
            "name": "一买_开多", "operate": "开多",
            "signals_all": [f"{FREQ}_D1B_BUY1_一买_任意_任意_0"],
            "signals_not": [_ZDT],
        })],
        exits=[_exit_bi_down()],
        interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
    )


class FirstBuyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_first_buy_position(self.symbol)]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        if "cxt_first_buy_V221126" not in names:
            base.append({"name": "cxt_first_buy_V221126", "freq": FREQ, "di": 1})
        return base


def build_second_buy_position(symbol: str) -> Position:
    return Position(
        name="二买策略_多头", symbol=symbol,
        opens=[Event.load({
            "name": "二买_开多", "operate": "开多",
            "signals_all": [f"{FREQ}_D1#SMA#21_BS2辅助V230320_二买_任意_任意_0"],
            "signals_not": [_ZDT],
        })],
        exits=[_exit_bi_down()],
        interval=INTERVAL, timeout=TIMEOUT, stop_loss=400, t0=False,
    )


class SecondBuyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_second_buy_position(self.symbol)]


def build_third_buy_position(symbol: str) -> Position:
    return Position(
        name="三买策略_多头", symbol=symbol,
        opens=[
            Event.load({
                "name": "纯笔三买_开多", "operate": "开多",
                "signals_all": [f"{FREQ}_D1_三买辅助V230228_三买_任意_任意_0"],
                "signals_not": [_ZDT],
            }),
            Event.load({
                "name": "均线三买_开多", "operate": "开多",
                "signals_all": [f"{FREQ}_D1#SMA#34_BS3辅助V230318_三买_任意_任意_0"],
                "signals_not": [_ZDT],
            }),
        ],
        exits=[_exit_bi_down()],
        interval=INTERVAL, timeout=TIMEOUT, stop_loss=300, t0=False,
    )


class ThirdBuyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_third_buy_position(self.symbol)]


def build_bi_trend_position(symbol: str) -> Position:
    return Position(
        name="笔趋势_非多即空", symbol=symbol,
        opens=[
            Event.load({"name": "笔向上_开多", "operate": "开多",
                         "signals_all": [_BI_UP], "signals_not": [_ZDT]}),
            Event.load({"name": "笔向下_开空", "operate": "开空",
                         "signals_all": [_BI_DOWN], "signals_not": [_DIETING]}),
        ],
        exits=[], interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS,
    )


class BiTrendStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_bi_trend_position(self.symbol)]


def build_divergence_position(symbol: str) -> Position:
    return Position(
        name="背驰策略_多头", symbol=symbol,
        opens=[
            Event.load({
                "name": "aAb底背驰_开多", "operate": "开多",
                "signals_all": [f"{FREQ}_D1五笔_形态V230619_aAb式底背驰_任意_任意_0"],
                "signals_not": [_ZDT],
            }),
            Event.load({
                "name": "类三买_开多", "operate": "开多",
                "signals_all": [f"{FREQ}_D1五笔_形态V230619_类三买_任意_任意_0"],
                "signals_not": [_ZDT],
            }),
        ],
        exits=[_exit_bi_down()],
        interval=INTERVAL, timeout=TIMEOUT, stop_loss=STOP_LOSS, t0=False,
    )


class DivergenceStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_divergence_position(self.symbol)]


STRATEGY_TAGS = ["1_一买策略", "2_二买策略", "3_三买策略", "4_笔趋势跟踪", "5_背驰策略"]
STRATEGY_CLASSES = [FirstBuyStrategy, SecondBuyStrategy, ThirdBuyStrategy, BiTrendStrategy, DivergenceStrategy]


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
    print("  缠论 5 大策略统一回测（真实 A 股日线数据）")
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
                title=f"缠论策略 - {tag}（全A股日线）",
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
