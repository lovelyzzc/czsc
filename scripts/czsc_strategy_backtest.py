"""缠论 6 大策略统一回测对比

策略列表：
    1. 一买策略 — 趋势反转（逆势抄底）
    2. 二买策略 — 趋势确认（回踩买入）
    3. 三买策略 — 中枢突破（顺势追涨）
    4. 笔趋势跟踪 — 最简单的趋势策略
    5. 背驰策略 — 多笔形态 + MACD 辅助
    6. 多级别联立 — 大级别定方向 + 小级别找买点

所有策略使用相同的模拟数据、相同的回测区间、相同的评价标准。
"""

from __future__ import annotations

import json
import time
import traceback
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
from czsc.mock import generate_symbol_kines

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "strategy_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "SYM"
SDT_DATA = "20180101"
EDT_DATA = "20240101"
SDT_BT = "2019-01-01"
FEE_RATE = 0.0002
SEED = 42

F30 = "30分钟"
F60 = "60分钟"
FD = "日线"

_ZDT_30 = f"{F30}_D1_涨跌停V230331_涨停_任意_任意_0"
_ZDT_60 = f"{F60}_D1_涨跌停V230331_涨停_任意_任意_0"
_DIETING_30 = f"{F30}_D1_涨跌停V230331_跌停_任意_任意_0"
_BI_DOWN_30 = f"{F30}_D1_表里关系V230101_向下_任意_任意_0"
_BI_UP_30 = f"{F30}_D1_表里关系V230101_向上_任意_任意_0"
_BI_DOWN_60 = f"{F60}_D1_表里关系V230101_向下_任意_任意_0"
_BI_UP_60 = f"{F60}_D1_表里关系V230101_向上_任意_任意_0"


def _exit_bi_down(freq: str) -> Event:
    return Event.load({
        "name": f"{freq}笔向下_平多",
        "operate": "平多",
        "signals_all": [f"{freq}_D1_表里关系V230101_向下_任意_任意_0"],
    })


def _exit_bi_up(freq: str) -> Event:
    return Event.load({
        "name": f"{freq}笔向上_平空",
        "operate": "平空",
        "signals_all": [f"{freq}_D1_表里关系V230101_向上_任意_任意_0"],
    })


# ============================================================
# 策略 1：一买策略 — 趋势反转
# ============================================================

def build_first_buy_position(symbol: str) -> Position:
    """一买开多：cxt_first_buy_V221126 触发一买信号"""
    open_event = Event.load({
        "name": "一买_开多",
        "operate": "开多",
        "signals_all": [f"{F30}_D1B_BUY1_一买_任意_任意_0"],
        "signals_not": [_ZDT_30],
    })
    return Position(
        name="一买策略_多头",
        symbol=symbol,
        opens=[open_event],
        exits=[_exit_bi_down(F30)],
        interval=3600 * 8,
        timeout=16 * 60,
        stop_loss=500,
        t0=False,
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
            base.append({"name": "cxt_first_buy_V221126", "freq": F30, "di": 1})
        return base


# ============================================================
# 策略 2：二买策略 — 趋势确认
# ============================================================

def build_second_buy_position(symbol: str) -> Position:
    """二买开多：cxt_second_bs_V230320 + SMA21 辅助"""
    open_event = Event.load({
        "name": "二买_开多",
        "operate": "开多",
        "signals_all": [f"{F30}_D1#SMA#21_BS2辅助V230320_二买_任意_任意_0"],
        "signals_not": [_ZDT_30],
    })
    return Position(
        name="二买策略_多头",
        symbol=symbol,
        opens=[open_event],
        exits=[_exit_bi_down(F30)],
        interval=3600 * 4,
        timeout=16 * 40,
        stop_loss=400,
        t0=False,
    )


class SecondBuyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_second_buy_position(self.symbol)]


# ============================================================
# 策略 3：三买策略 — 中枢突破
# ============================================================

def build_third_buy_position(symbol: str) -> Position:
    """三买开多：三种三买信号 OR 触发"""
    opens = [
        Event.load({
            "name": "纯笔三买_开多",
            "operate": "开多",
            "signals_all": [f"{F30}_D1_三买辅助V230228_三买_任意_任意_0"],
            "signals_not": [_ZDT_30],
        }),
        Event.load({
            "name": "均线三买_开多",
            "operate": "开多",
            "signals_all": [f"{F30}_D1#SMA#34_BS3辅助V230318_三买_任意_任意_0"],
            "signals_not": [_ZDT_30],
        }),
    ]
    return Position(
        name="三买策略_多头",
        symbol=symbol,
        opens=opens,
        exits=[_exit_bi_down(F30)],
        interval=3600 * 4,
        timeout=16 * 30,
        stop_loss=300,
        t0=False,
    )


class ThirdBuyStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_third_buy_position(self.symbol)]


# ============================================================
# 策略 4：笔趋势跟踪 — 非多即空
# ============================================================

def build_bi_trend_position(symbol: str) -> Position:
    """笔向上开多、笔向下开空，反手即平"""
    opens = [
        Event.load({
            "name": "笔向上_开多",
            "operate": "开多",
            "signals_all": [_BI_UP_30],
            "signals_not": [_ZDT_30],
        }),
        Event.load({
            "name": "笔向下_开空",
            "operate": "开空",
            "signals_all": [_BI_DOWN_30],
            "signals_not": [_DIETING_30],
        }),
    ]
    return Position(
        name="笔趋势_非多即空",
        symbol=symbol,
        opens=opens,
        exits=[],
        interval=3600 * 4,
        timeout=16 * 30,
        stop_loss=500,
    )


class BiTrendStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_bi_trend_position(self.symbol)]


# ============================================================
# 策略 5：背驰策略 — 五笔形态背驰
# ============================================================

def build_divergence_position(symbol: str) -> Position:
    """五笔底背驰开多，笔向下平多"""
    opens = [
        Event.load({
            "name": "aAb底背驰_开多",
            "operate": "开多",
            "signals_all": [f"{F30}_D1五笔_形态V230619_aAb式底背驰_任意_任意_0"],
            "signals_not": [_ZDT_30],
        }),
        Event.load({
            "name": "类三买_开多",
            "operate": "开多",
            "signals_all": [f"{F30}_D1五笔_形态V230619_类三买_任意_任意_0"],
            "signals_not": [_ZDT_30],
        }),
    ]
    return Position(
        name="背驰策略_多头",
        symbol=symbol,
        opens=opens,
        exits=[_exit_bi_down(F30)],
        interval=3600 * 8,
        timeout=16 * 50,
        stop_loss=500,
        t0=False,
    )


class DivergenceStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_divergence_position(self.symbol)]


# ============================================================
# 策略 6：多级别联立 — 60分钟方向 + 30分钟入场
# ============================================================

def build_multi_level_position(symbol: str) -> Position:
    """大级别（60分钟）笔向上时，小级别（30分钟）三买入场"""
    open_event = Event.load({
        "name": "60min向上+30min三买_开多",
        "operate": "开多",
        "signals_all": [
            _BI_UP_60,
            f"{F30}_D1_三买辅助V230228_三买_任意_任意_0",
        ],
        "signals_not": [_ZDT_30],
    })
    return Position(
        name="多级别联立_多头",
        symbol=symbol,
        opens=[open_event],
        exits=[_exit_bi_down(F30)],
        interval=3600 * 4,
        timeout=16 * 30,
        stop_loss=300,
        t0=False,
    )


class MultiLevelStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [build_multi_level_position(self.symbol)]


# ============================================================
# 统一回测流程
# ============================================================

ALL_STRATEGIES = [
    ("1_一买策略", FirstBuyStrategy),
    ("2_二买策略", SecondBuyStrategy),
    ("3_三买策略", ThirdBuyStrategy),
    ("4_笔趋势跟踪", BiTrendStrategy),
    ("5_背驰策略", DivergenceStrategy),
    ("6_多级别联立", MultiLevelStrategy),
]


def holds_to_weight_df(holds: pd.DataFrame) -> pd.DataFrame:
    df = holds[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"})
    if df.duplicated(subset=["dt", "symbol"]).any():
        df = df.groupby(["dt", "symbol"], as_index=False).agg(
            weight=("weight", "mean"),
            price=("price", "first"),
        )
    return df[["dt", "symbol", "weight", "price"]]


def run_strategy(tag: str, strategy: CzscStrategyBase, bars: list, sdt: str) -> dict | None:
    """运行单个策略的回测"""
    print(f"\n{'='*60}")
    print(f"  [{tag}]")
    print(f"{'='*60}")
    print(f"  base_freq = {strategy.base_freq}")
    print(f"  freqs = {strategy.freqs}")
    print(f"  signals_config = {len(strategy.signals_config)} 项")
    for cfg in strategy.signals_config:
        print(f"    {cfg}")

    t0 = time.time()
    try:
        res = strategy.backtest(bars, sdt=sdt)
    except Exception as e:
        print(f"  [ERROR] 回测失败: {e}")
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    pairs = res.pairs_df()
    holds = res.holds_df()
    print(f"  回测耗时: {elapsed:.1f}s")
    print(f"  pairs: {len(pairs)} 笔交易  |  holds: {len(holds)} 条持仓")

    if holds.empty or len(holds) < 10:
        print(f"  [WARN] 持仓数据不足，跳过绩效计算")
        return None

    dfw = holds_to_weight_df(holds)
    wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)

    stats = wb.stats
    stats["tag"] = tag
    stats["pairs_count"] = len(pairs)
    stats["elapsed_s"] = round(elapsed, 1)

    # 交易对统计
    if not pairs.empty and "盈亏比例" in pairs.columns:
        profits = pairs["盈亏比例"]
        stats["avg_profit"] = round(float(profits.mean() * 100), 2)
        stats["median_profit"] = round(float(profits.median() * 100), 2)
        stats["win_count"] = int((profits > 0).sum())
        stats["loss_count"] = int((profits <= 0).sum())
        stats["pair_win_rate"] = round(float((profits > 0).sum() / len(profits) * 100), 1) if len(profits) > 0 else 0
        winning = profits[profits > 0]
        losing = profits[profits <= 0]
        avg_win = float(winning.mean()) if len(winning) > 0 else 0
        avg_loss = abs(float(losing.mean())) if len(losing) > 0 else 1e-9
        stats["profit_loss_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    print(f"  核心指标:")
    for k in ["年化收益率", "夏普比率", "最大回撤", "卡尔马比率", "pair_win_rate", "profit_loss_ratio", "pairs_count"]:
        if k in stats:
            print(f"    {k}: {stats[k]}")

    # HTML 报告
    try:
        out_html = OUTPUT_DIR / f"{tag}.html"
        generate_backtest_report(
            df=dfw,
            output_path=str(out_html),
            title=f"缠论策略 - {tag}",
            fee_rate=FEE_RATE,
            weight_type="ts",
            yearly_days=252,
        )
        print(f"  HTML: {out_html.name} ({out_html.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  [WARN] HTML 报告生成失败: {e}")

    return stats


def main():
    print("=" * 60)
    print("  缠论 6 大策略统一回测")
    print("=" * 60)

    df = generate_symbol_kines(SYMBOL, F30, SDT_DATA, EDT_DATA, seed=SEED)
    bars = format_standard_kline(df, freq=F30)
    print(f"[数据] {len(bars)} 根 {F30} K 线 ({bars[0].dt} ~ {bars[-1].dt})")

    all_stats = []
    for tag, strategy_cls in ALL_STRATEGIES:
        strategy = strategy_cls(symbol=SYMBOL)
        stats = run_strategy(tag, strategy, bars, SDT_BT)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] 所有策略都未产生有效结果")
        return

    # 汇总对比
    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 80)
    print("  策略对比汇总")
    print("=" * 80)

    display_cols = [c for c in [
        "pairs_count", "pair_win_rate", "profit_loss_ratio", "avg_profit",
        "年化收益率", "夏普比率", "最大回撤", "卡尔马比率", "elapsed_s"
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    # 保存为 JSON
    with open(OUTPUT_DIR / "comparison_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR / 'comparison_stats.json'}")
    print(f"[完成] 所有策略回测完成")


if __name__ == "__main__":
    main()
