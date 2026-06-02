"""板块联动验证：对比"加板块过滤 vs 不加"对策略效果的影响

验证方法：
- 标的池：机械基件行业 116 只个股（002896 所属行业）
- 基础策略：笔趋势动量加速（BiTrendAccelStrategy）
- 板块强度指标：行业内全部个股 20 日平均涨幅
- 对比维度：
  A. 原版策略（不加板块过滤）
  B. 板块强势过滤（板块 20 日平均涨幅 > 0 时才允许开仓）
  C. 板块领涨过滤（个股 20 日涨幅 > 板块均值时才允许开仓）
"""

from __future__ import annotations

import json
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

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "sector_correlation"
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

_BI_UPTREND = f"{FREQ}_D4N1笔趋势_高低点辅助判断V230913_上升趋势_任意_任意_0"
_MACD_STRONG = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_强势_任意_任意_0"
_PRICE_ACC = f"{FREQ}_D1W10_加速V221110_上涨_任意_任意_0"
_MACD_WEAK = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_弱势_任意_任意_0"

SECTOR_CODES = [
    "000530.SZ", "000777.SZ", "001268.SZ", "001306.SZ", "001379.SZ",
    "002026.SZ", "002046.SZ", "002150.SZ", "002164.SZ", "002184.SZ",
    "002272.SZ", "002282.SZ", "002342.SZ", "002347.SZ", "002438.SZ",
    "002514.SZ", "002598.SZ", "002633.SZ", "002747.SZ", "002760.SZ",
    "002795.SZ", "002823.SZ", "002871.SZ", "002877.SZ", "002884.SZ",
    "002896.SZ", "002931.SZ", "300091.SZ", "300145.SZ", "300154.SZ",
    "300257.SZ", "300260.SZ", "300277.SZ", "300420.SZ", "300421.SZ",
    "300464.SZ", "300470.SZ", "300488.SZ", "300503.SZ", "300606.SZ",
    "300718.SZ", "300780.SZ", "300817.SZ", "300828.SZ", "300838.SZ",
    "300850.SZ", "300885.SZ", "300943.SZ", "300946.SZ", "300971.SZ",
    "300984.SZ", "300988.SZ", "300992.SZ", "301040.SZ", "301043.SZ",
    "301070.SZ", "301079.SZ", "301107.SZ", "301125.SZ", "301137.SZ",
    "301151.SZ", "301160.SZ", "301202.SZ", "301232.SZ", "301252.SZ",
    "301255.SZ", "301261.SZ", "301268.SZ", "301273.SZ", "301309.SZ",
    "301311.SZ", "301317.SZ", "301368.SZ", "301377.SZ", "301399.SZ",
    "301446.SZ", "301448.SZ", "301548.SZ", "301596.SZ", "301616.SZ",
    "600114.SH", "600343.SH", "600592.SH", "600889.SH", "601002.SH",
    "601177.SH", "601369.SH", "603092.SH", "603112.SH", "603173.SH",
    "603187.SH", "603248.SH", "603269.SH", "603270.SH", "603308.SH",
    "603400.SH", "603617.SH", "603667.SH", "603699.SH", "603757.SH",
    "603915.SH", "605060.SH", "605100.SH", "605389.SH", "688017.SH",
    "688028.SH", "688059.SH", "688255.SH", "688257.SH", "688308.SH",
    "688333.SH", "688355.SH", "688360.SH", "688379.SH", "688395.SH",
    "688455.SH",
]

SECTOR_STRENGTH_WINDOW = 20


def _exit_bi_down() -> Event:
    return Event.load({"name": "笔向下_平多", "operate": "平多", "signals_all": [_BI_DOWN]})


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


def _load_stock_df(ts_code: str) -> pd.DataFrame | None:
    """加载单只股票日线数据"""
    pq = DATA_DIR / f"{ts_code}.parquet"
    if not pq.exists():
        return None
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return None
    if len(df) < MIN_BARS:
        return None
    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def build_sector_strength() -> pd.DataFrame:
    """构建板块强度时间序列

    Returns:
        DataFrame with columns: dt, sector_ret20 (板块 20 日平均涨幅),
        每只个股的 20 日涨幅列 (symbol -> ret20)
    """
    print("[Phase 1] 构建板块强度指标 ...")
    stock_returns = {}
    for code in SECTOR_CODES:
        df = _load_stock_df(code)
        if df is None:
            continue
        r = df[["dt", "close"]].copy()
        r["ret20"] = r["close"].pct_change(SECTOR_STRENGTH_WINDOW)
        r = r[["dt", "ret20"]].rename(columns={"ret20": code})
        stock_returns[code] = r.set_index("dt")

    if not stock_returns:
        raise RuntimeError("无有效个股数据")

    combined = pd.concat(stock_returns.values(), axis=1)
    result = pd.DataFrame(index=combined.index)
    result["sector_ret20"] = combined.mean(axis=1)

    for code in SECTOR_CODES:
        if code in combined.columns:
            result[code] = combined[code]

    result = result.dropna(subset=["sector_ret20"]).reset_index()
    print(f"  板块强度序列: {len(result)} 个交易日")
    print(f"  板块 20 日均涨幅范围: [{result['sector_ret20'].min():.4f}, {result['sector_ret20'].max():.4f}]")
    print(f"  板块强势天数占比 (ret20>0): {(result['sector_ret20'] > 0).mean():.1%}")
    return result


def _backtest_one_stock(ts_code: str, bars, sdt: str):
    """对单只股票跑策略，返回 holds_df"""
    symbol = bars[0].symbol
    strategy = BiTrendAccelStrategy(symbol=symbol)
    res = strategy.backtest(bars, sdt=sdt)
    pairs = res.pairs_df()
    holds = res.holds_df()
    return pairs, holds


def _filter_holds(holds: pd.DataFrame, sector_strength: pd.DataFrame,
                  mode: str, ts_code: str) -> pd.DataFrame:
    """按板块过滤条件修改 holds 中的仓位

    mode:
      "baseline": 不做任何过滤
      "sector_bull": 仅板块 20 日均涨幅 > 0 时保留仓位
      "stock_leader": 仅个股 20 日涨幅 > 板块均值时保留仓位
    """
    if mode == "baseline" or holds.empty:
        return holds

    h = holds.copy()
    h["dt"] = pd.to_datetime(h["dt"])

    ss = sector_strength[["dt", "sector_ret20"]].copy()
    if ts_code in sector_strength.columns:
        ss["stock_ret20"] = sector_strength[ts_code].values

    h = h.merge(ss, on="dt", how="left")

    if mode == "sector_bull":
        mask = h["sector_ret20"].fillna(0) <= 0
        h.loc[mask, "pos"] = 0
    elif mode == "stock_leader":
        if "stock_ret20" in h.columns:
            mask = h["stock_ret20"].fillna(0) <= h["sector_ret20"].fillna(0)
            h.loc[mask, "pos"] = 0
        else:
            mask = h["sector_ret20"].fillna(0) <= 0
            h.loc[mask, "pos"] = 0

    return h[["dt", "symbol", "pos", "price"]]


def run_comparison(sector_strength: pd.DataFrame):
    """运行三组对比回测"""
    print("\n[Phase 2] 逐股回测 ...")

    modes = ["baseline", "sector_bull", "stock_leader"]
    mode_names = {
        "baseline": "A_无板块过滤",
        "sector_bull": "B_板块强势过滤",
        "stock_leader": "C_板块领涨过滤",
    }

    all_holds: dict[str, list[pd.DataFrame]] = {m: [] for m in modes}
    all_pairs: dict[str, dict] = {m: {"win": 0, "loss": 0, "pairs": 0, "stocks": 0} for m in modes}

    valid = 0
    for i, code in enumerate(SECTOR_CODES, 1):
        df = _load_stock_df(code)
        if df is None:
            continue

        try:
            bars = format_standard_kline(df, freq=FREQ)
        except Exception:
            continue

        n_bars = len(bars)
        sdt = bars[n_bars // 4].dt.strftime("%Y-%m-%d")

        try:
            pairs, holds = _backtest_one_stock(code, bars, sdt)
        except Exception:
            continue

        valid += 1

        for mode in modes:
            filtered_holds = _filter_holds(holds, sector_strength, mode, code)

            if not filtered_holds.empty:
                h = filtered_holds[["dt", "symbol", "pos", "price"]].copy()
                h["tag"] = mode_names[mode]
                all_holds[mode].append(h)

            if mode == "baseline" and not pairs.empty and "盈亏比例" in pairs.columns:
                profits = pairs["盈亏比例"]
                for m in modes:
                    all_pairs[m]["pairs"] += len(pairs)
                    all_pairs[m]["win"] += int((profits > 0).sum())
                    all_pairs[m]["loss"] += int((profits <= 0).sum())
                    all_pairs[m]["stocks"] += 1

        if i % 20 == 0:
            print(f"  [{i}/{len(SECTOR_CODES)}] 有效 {valid}")

    print(f"\n  回测完成: {valid}/{len(SECTOR_CODES)} 只有效")
    return modes, mode_names, all_holds, all_pairs


def generate_report(modes, mode_names, all_holds, all_pairs):
    """汇总并输出对比报告"""
    print("\n[Phase 3] 汇总分析 ...")

    all_stats = []
    for mode in modes:
        tag = mode_names[mode]
        holds_list = all_holds[mode]

        print(f"\n{'='*60}")
        print(f"  [{tag}]")
        print(f"{'='*60}")

        if not holds_list:
            print("  无持仓数据")
            continue

        dfw = pd.concat(holds_list, ignore_index=True)
        dfw = dfw[["dt", "symbol", "pos", "price"]].rename(columns={"pos": "weight"})

        if dfw.duplicated(subset=["dt", "symbol"]).any():
            dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
                weight=("weight", "mean"), price=("price", "first"),
            )
        dfw = dfw[["dt", "symbol", "weight", "price"]]

        try:
            wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
            stats = wb.stats
        except Exception as e:
            print(f"  WeightBacktest 失败: {e}")
            continue

        stats["tag"] = tag
        ps = all_pairs[mode]

        total = ps["win"] + ps["loss"]
        if total > 0:
            stats["pair_win_rate"] = round(ps["win"] / total * 100, 1)

        for k in ["年化收益", "夏普比率", "最大回撤", "卡玛比率", "pair_win_rate"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            out_html = OUTPUT_DIR / f"{tag}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"板块联动验证 - {tag}（机械基件行业）",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"    HTML: {out_html.name}")
        except Exception as e:
            print(f"    HTML 报告生成失败: {e}")

        all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] 所有组别均无结果")
        return

    cmp = pd.DataFrame(all_stats).set_index("tag")

    print("\n\n" + "=" * 80)
    print("  板块联动验证 — 策略对比汇总")
    print("  行业: 机械基件 | 基础策略: 笔趋势动量加速")
    print("=" * 80)
    display_cols = [c for c in [
        "年化收益", "夏普比率", "最大回撤", "卡玛比率", "pair_win_rate",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    if len(all_stats) >= 2:
        baseline = all_stats[0]
        print("\n--- 板块过滤效果 ---")
        for s in all_stats[1:]:
            tag = s["tag"]
            for metric in ["年化收益", "夏普比率", "最大回撤"]:
                b_val = baseline.get(metric, 0)
                s_val = s.get(metric, 0)
                try:
                    b_f = float(str(b_val).rstrip("%"))
                    s_f = float(str(s_val).rstrip("%"))
                    diff = s_f - b_f
                    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                    print(f"  {tag} vs 无过滤 | {metric}: {b_val} → {s_val} ({arrow}{abs(diff):.2f})")
                except (ValueError, TypeError):
                    print(f"  {tag} vs 无过滤 | {metric}: {b_val} → {s_val}")

    with open(OUTPUT_DIR / "sector_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[文件] {OUTPUT_DIR / 'sector_comparison.json'}")


def main():
    print("=" * 70)
    print("  板块联动验证 — 机械基件行业（笔趋势动量加速策略）")
    print("=" * 70)
    t0 = time.time()

    sector_strength = build_sector_strength()
    modes, mode_names, all_holds, all_pairs = run_comparison(sector_strength)
    generate_report(modes, mode_names, all_holds, all_pairs)

    print(f"\n[总耗时] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
