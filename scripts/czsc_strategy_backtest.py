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
import time
from pathlib import Path

import pandas as pd

from czsc import (
    CzscStrategyBase,
    Event,
    Position,
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

def _process_stock(parquet_path: str) -> dict | None:
    """单只股票全策略回测，返回轻量摘要（不返回 holds）"""
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
    for tag, cls in zip(STRATEGY_TAGS, STRATEGY_CLASSES):
        try:
            strategy = cls(symbol=symbol)
            res = strategy.backtest(bars, sdt=sdt)
            pairs = res.pairs_df()
            holds = res.holds_df()

            info = {"pairs": len(pairs), "holds_rows": len(holds)}
            if not pairs.empty and "盈亏比例" in pairs.columns:
                profits = pairs["盈亏比例"]
                info["win"] = int((profits > 0).sum())
                info["loss"] = int((profits <= 0).sum())
                info["total_return"] = float(profits.sum())
                info["mean_return"] = float(profits.mean())
                winning = profits[profits > 0]
                losing = profits[profits <= 0]
                info["avg_win"] = float(winning.mean()) if len(winning) > 0 else 0
                info["avg_loss"] = float(abs(losing.mean())) if len(losing) > 0 else 0

                if len(holds) > 1:
                    holds_sorted = holds.sort_values("dt")
                    cum = (1 + holds_sorted["pos"] * holds_sorted["price"].pct_change().fillna(0)).cumprod()
                    peak = cum.cummax()
                    dd = (cum - peak) / peak
                    info["max_drawdown"] = float(abs(dd.min()))
                    info["final_return"] = float(cum.iloc[-1] - 1)
            else:
                info["win"] = 0
                info["loss"] = 0

            result[tag] = info
        except Exception:
            continue

    return result if len(result) > 1 else None


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("  缠论 5 大策略统一回测（真实 A 股日线数据）")
    print("=" * 70)

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

    # 汇总统计
    all_stats = []
    for tag in STRATEGY_TAGS:
        total_pairs = 0
        total_win = 0
        total_loss = 0
        returns = []
        drawdowns = []
        avg_wins = []
        avg_losses = []
        stocks_with_signal = 0

        for r in all_results:
            if tag not in r:
                continue
            info = r[tag]
            total_pairs += info["pairs"]
            total_win += info.get("win", 0)
            total_loss += info.get("loss", 0)
            if info["pairs"] > 0:
                stocks_with_signal += 1
                if "mean_return" in info:
                    returns.append(info["mean_return"])
                if "max_drawdown" in info:
                    drawdowns.append(info["max_drawdown"])
                if info.get("avg_win", 0) > 0:
                    avg_wins.append(info["avg_win"])
                if info.get("avg_loss", 0) > 0:
                    avg_losses.append(info["avg_loss"])

        stats = {"tag": tag, "stocks_count": stocks_with_signal, "pairs_count": total_pairs}

        total = total_win + total_loss
        if total > 0:
            stats["pair_win_rate"] = round(total_win / total * 100, 1)
            stats["win_count"] = total_win
            stats["loss_count"] = total_loss

        if returns:
            stats["avg_return_pct"] = round(float(pd.Series(returns).mean() * 100), 2)
            stats["median_return_pct"] = round(float(pd.Series(returns).median() * 100), 2)

        if drawdowns:
            stats["avg_max_drawdown"] = round(float(pd.Series(drawdowns).mean()), 4)
            stats["median_max_drawdown"] = round(float(pd.Series(drawdowns).median()), 4)

        if avg_wins and avg_losses:
            mean_win = float(pd.Series(avg_wins).mean())
            mean_loss = float(pd.Series(avg_losses).mean())
            stats["profit_loss_ratio"] = round(mean_win / mean_loss, 2) if mean_loss > 0 else 0

        stats["elapsed_s"] = round(total_elapsed, 1)

        print(f"\n{'='*60}")
        print(f"  [{tag}]")
        print(f"{'='*60}")
        for k, v in stats.items():
            if k != "tag":
                print(f"    {k}: {v}")

        all_stats.append(stats)

    # 汇总表
    if all_stats:
        cmp = pd.DataFrame(all_stats).set_index("tag")
        print("\n\n" + "=" * 80)
        print("  策略对比汇总（全 A 股日线）")
        print("=" * 80)
        display_cols = [c for c in [
            "stocks_count", "pairs_count", "pair_win_rate", "profit_loss_ratio",
            "avg_return_pct", "avg_max_drawdown",
        ] if c in cmp.columns]
        print(cmp[display_cols].to_string())

    with open(OUTPUT_DIR / "comparison_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR / 'comparison_stats.json'}")
    print(f"[完成] 所有策略回测完成")


if __name__ == "__main__":
    main()
