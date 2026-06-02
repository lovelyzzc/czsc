"""全 A 股最强标的选择器 — 5 日持有期最优因子研究

核心问题：如何选出未来 5 个交易日收益最高的标的？

研究框架：
- 每周一（每 5 个交易日）对全市场做截面排序
- 按不同因子选出 Top 20 标的，等权持有 5 日
- 对比各因子组合的年化收益 / 夏普 / 最大回撤

测试因子：
  F1. 20 日动量 (ret20)           — 中期趋势强度
  F2. 5 日动量 (ret5)             — 短期爆发力
  F3. 量比 (vol_ratio)            — 近 5 日均量 / 20 日均量
  F4. 突破 MA20 幅度 (ma20_prem)  — 价格相对均线位置
  F5. 波动收敛后突破 (vol_squeeze) — ATR 收窄 + 价格创新高
  F6. 综合排名 (composite)        — F1+F3+F4 等权加权

数据源：~/.ts_data_cache/a_stock_daily_qfq/ 下的日线 parquet
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "stock_selector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"

TOP_N = 20
HOLD_DAYS = 5
MIN_BARS = 120
MIN_PRICE = 2.0
MAX_PCT_CHG = 9.8


def load_all_stocks() -> pd.DataFrame:
    """加载全 A 股日线数据到统一 DataFrame"""
    print("[1/4] 加载全 A 股日线数据 ...")
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    print(f"  文件数: {len(parquets)}")

    parts = []
    skipped = 0
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
        except Exception:
            skipped += 1
            continue
        if len(df) < MIN_BARS:
            skipped += 1
            continue

        code = df["ts_code"].iloc[0]
        if code.startswith(("688", "920", "83", "43")):
            skipped += 1
            continue

        df = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    print(f"  有效股票: {len(parts)} 只 (跳过 {skipped})")
    print(f"  总行数: {len(all_df):,}")
    print(f"  时间范围: {all_df['trade_date'].min().date()} ~ {all_df['trade_date'].max().date()}")
    return all_df


def compute_factors(all_df: pd.DataFrame) -> pd.DataFrame:
    """计算各选股因子"""
    print("\n[2/4] 计算选股因子 ...")
    t0 = time.time()

    g = all_df.groupby("ts_code", sort=False)

    all_df["ret5"] = g["close"].transform(lambda x: x.pct_change(5))
    all_df["ret10"] = g["close"].transform(lambda x: x.pct_change(10))
    all_df["ret20"] = g["close"].transform(lambda x: x.pct_change(20))

    all_df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
    all_df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
    all_df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())

    all_df["ma20_prem"] = (all_df["close"] - all_df["ma20"]) / all_df["ma20"]

    all_df["ma_bull"] = ((all_df["ma5"] > all_df["ma10"]) & (all_df["ma10"] > all_df["ma20"])).astype(int)

    all_df["vol_ma5"] = g["vol"].transform(lambda x: x.rolling(5).mean())
    all_df["vol_ma20"] = g["vol"].transform(lambda x: x.rolling(20).mean())
    all_df["vol_ratio"] = all_df["vol_ma5"] / all_df["vol_ma20"]

    all_df["high20"] = g["high"].transform(lambda x: x.rolling(20).max())
    all_df["near_high20"] = all_df["close"] / all_df["high20"]

    def _atr(sub):
        h = sub["high"]
        l = sub["low"]
        pc = sub["close"].shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.rolling(20).mean()

    all_df["atr20"] = g.apply(_atr, include_groups=False).reset_index(level=0, drop=True)
    all_df["atr5"] = g.apply(
        lambda sub: pd.concat([
            sub["high"] - sub["low"],
            (sub["high"] - sub["close"].shift(1)).abs(),
            (sub["low"] - sub["close"].shift(1)).abs()
        ], axis=1).max(axis=1).rolling(5).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    all_df["vol_squeeze"] = (all_df["atr5"] < all_df["atr20"] * 0.7).astype(int)

    all_df["fwd_ret5"] = g["close"].transform(lambda x: x.shift(-HOLD_DAYS) / x - 1)

    all_df["is_limit"] = all_df["pct_chg"].abs() >= MAX_PCT_CHG

    print(f"  因子计算完成 ({time.time() - t0:.1f}s)")
    return all_df


def run_factor_backtest(all_df: pd.DataFrame) -> dict:
    """按因子排序选股，回测 5 日持有期收益"""
    print("\n[3/4] 因子选股回测 ...")

    trade_dates = sorted(all_df["trade_date"].unique())
    rebalance_dates = trade_dates[60::HOLD_DAYS]
    print(f"  调仓日数: {len(rebalance_dates)} (每 {HOLD_DAYS} 日)")

    factor_defs = {
        "F1_动量20日_做多最强": ("ret20", "top"),
        "F2_动量5日_做多最强": ("ret5", "top"),
        "F3_反转5日_做多最弱": ("ret5", "bottom"),
        "F4_反转10日_做多最弱": ("ret10", "bottom"),
        "F5_创20日新高": ("near_high20", "top"),
        "F6_量比_最高": ("vol_ratio", "top"),
        "F7_量比_最低": ("vol_ratio", "bottom"),
    }

    results = {name: [] for name in factor_defs}
    results["F8_近高位+MA多头"] = []
    results["F9_近高位+低波动"] = []
    results["F10_反转+缩量"] = []
    results["F11_稳健强势"] = []
    results["benchmark"] = []

    for dt in rebalance_dates:
        day_df = all_df[all_df["trade_date"] == dt].copy()

        day_df = day_df[
            (day_df["close"] >= MIN_PRICE)
            & (~day_df["is_limit"])
            & (day_df["vol"] > 0)
            & (day_df["fwd_ret5"].notna())
            & (day_df["ret20"].notna())
        ]

        if len(day_df) < TOP_N * 2:
            continue

        bench_ret = day_df["fwd_ret5"].mean()
        results["benchmark"].append({"dt": dt, "ret": bench_ret})

        for name, (col, direction) in factor_defs.items():
            if col not in day_df.columns or day_df[col].isna().all():
                continue
            if direction == "top":
                top = day_df.nlargest(TOP_N, col)
            else:
                top = day_df.nsmallest(TOP_N, col)
            results[name].append({
                "dt": dt, "ret": top["fwd_ret5"].mean(),
                "stocks": top["ts_code"].tolist(),
            })

        # F8: 近高位 + MA 多头排列
        cond_f8 = day_df[(day_df["ma_bull"] == 1) & (day_df["near_high20"] > 0.95)]
        if len(cond_f8) >= TOP_N:
            top_f8 = cond_f8.nlargest(TOP_N, "near_high20")
            results["F8_近高位+MA多头"].append({
                "dt": dt, "ret": top_f8["fwd_ret5"].mean(),
                "stocks": top_f8["ts_code"].tolist(),
            })

        # F9: 近高位 + 低波动（稳步上涨、波动小的标的）
        cond_f9 = day_df[day_df["near_high20"] > 0.95].copy()
        if len(cond_f9) >= TOP_N and "atr20" in cond_f9.columns:
            cond_f9["atr_pct"] = cond_f9["atr20"] / cond_f9["close"]
            top_f9 = cond_f9.nsmallest(TOP_N, "atr_pct")
            results["F9_近高位+低波动"].append({
                "dt": dt, "ret": top_f9["fwd_ret5"].mean(),
                "stocks": top_f9["ts_code"].tolist(),
            })

        # F10: 短期反转 + 缩量（超跌 + 卖压衰竭）
        cond_f10 = day_df[(day_df["vol_ratio"] < 0.8)].copy()
        if len(cond_f10) >= TOP_N:
            top_f10 = cond_f10.nsmallest(TOP_N, "ret5")
            results["F10_反转+缩量"].append({
                "dt": dt, "ret": top_f10["fwd_ret5"].mean(),
                "stocks": top_f10["ts_code"].tolist(),
            })

        # F11: 稳健强势 = 20日动量>0 + 近高位>0.92 + 波动率不极端
        cond_f11 = day_df[
            (day_df["ret20"] > 0)
            & (day_df["near_high20"] > 0.92)
            & (day_df["ma_bull"] == 1)
        ].copy()
        if len(cond_f11) >= TOP_N:
            for c in ["near_high20", "ret20"]:
                cond_f11[f"rank_{c}"] = cond_f11[c].rank(pct=True)
            cond_f11["steady_score"] = cond_f11["rank_near_high20"] * 0.6 + cond_f11["rank_ret20"] * 0.4
            top_f11 = cond_f11.nlargest(TOP_N, "steady_score")
            results["F11_稳健强势"].append({
                "dt": dt, "ret": top_f11["fwd_ret5"].mean(),
                "stocks": top_f11["ts_code"].tolist(),
            })

    return results


def analyze_results(results: dict):
    """分析各因子回测结果"""
    print("\n[4/4] 结果分析 ...")

    summary = []
    for name, records in results.items():
        if not records:
            continue
        rets = pd.Series([r["ret"] for r in records])
        cum = (1 + rets).cumprod()

        n_periods = len(rets)
        periods_per_year = 252 / HOLD_DAYS
        total_ret = cum.iloc[-1] - 1

        ann_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0
        ann_vol = rets.std() * np.sqrt(periods_per_year) if rets.std() > 0 else 0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (cum / cum.cummax() - 1).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        win_rate = (rets > 0).mean()
        avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
        avg_loss = rets[rets <= 0].mean() if (rets <= 0).any() else 0
        profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        info = {
            "因子": name,
            "调仓次数": n_periods,
            "累计收益": f"{total_ret:.1%}",
            "年化收益": f"{ann_ret:.1%}",
            "年化波动": f"{ann_vol:.1%}",
            "夏普比率": round(sharpe, 2),
            "最大回撤": f"{max_dd:.1%}",
            "卡玛比率": round(calmar, 2),
            "胜率": f"{win_rate:.1%}",
            "盈亏比": round(profit_ratio, 2),
        }
        summary.append(info)

        dates = [r["dt"] for r in records]
        cum_df = pd.DataFrame({"dt": dates, "cum_ret": cum.values, "period_ret": rets.values})
        cum_df.to_csv(OUTPUT_DIR / f"{name}_cumulative.csv", index=False)

    df_summary = pd.DataFrame(summary)
    print("\n" + "=" * 100)
    print("  全 A 股最强标的选择器 — 5 日持有期因子对比")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    df_summary.to_csv(OUTPUT_DIR / "factor_comparison.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "factor_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR}")
    return df_summary


def generate_html_report(results: dict, df_summary: pd.DataFrame):
    """生成可视化 HTML 报告"""
    chart_data = {}
    for name, records in results.items():
        if not records:
            continue
        rets = pd.Series([r["ret"] for r in records])
        cum = (1 + rets).cumprod()
        dates = [pd.Timestamp(r["dt"]).strftime("%Y-%m-%d") for r in records]
        chart_data[name] = {"dates": dates, "values": [round(v, 4) for v in cum.values]}

    table_rows = ""
    for _, row in df_summary.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in df_summary.columns)
        table_rows += f"<tr>{cells}</tr>\n"
    table_headers = "".join(f"<th>{c}</th>" for c in df_summary.columns)

    colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#95a5a6",
    ]

    datasets_js = []
    for i, (name, data) in enumerate(chart_data.items()):
        color = colors[i % len(colors)]
        width = 3 if name in ("F7_综合排名", "F8_动量+量比+MA多头") else 1.5
        dash = "undefined" if name != "benchmark" else "[5,5]"
        datasets_js.append(f"""{{
            label: '{name}',
            data: {json.dumps(data['values'])},
            borderColor: '{color}',
            borderWidth: {width},
            borderDash: {dash},
            fill: false,
            pointRadius: 0,
        }}""")

    labels_js = json.dumps(chart_data.get("benchmark", list(chart_data.values())[0])["dates"])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>全 A 股最强标的选择器 — 5 日持有期因子对比</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a1a; color: #e0e0e0; padding: 24px; }}
  h1 {{ text-align: center; margin-bottom: 8px; color: #fff; font-size: 24px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 24px; font-size: 14px; }}
  .chart-container {{ background: #111; border-radius: 12px; padding: 20px;
                      margin-bottom: 24px; max-width: 1200px; margin: 0 auto 24px; }}
  table {{ width: 100%; max-width: 1200px; margin: 0 auto; border-collapse: collapse; font-size: 14px; }}
  th {{ background: #1a1a2e; padding: 10px 8px; text-align: center; border-bottom: 2px solid #333; }}
  td {{ padding: 8px; text-align: center; border-bottom: 1px solid #222; }}
  tr:hover {{ background: #1a1a2e; }}
  .note {{ max-width: 1200px; margin: 24px auto; padding: 16px; background: #111;
           border-radius: 8px; font-size: 13px; line-height: 1.8; color: #aaa; }}
  .note strong {{ color: #f39c12; }}
</style>
</head>
<body>
<h1>全 A 股最强标的选择器 — 因子对比研究</h1>
<p class="subtitle">持有期: {HOLD_DAYS} 个交易日 | 每期选 Top {TOP_N} 只 | 等权持有</p>

<div class="chart-container">
  <canvas id="cumChart" height="400"></canvas>
</div>

<table>
  <thead><tr>{table_headers}</tr></thead>
  <tbody>{table_rows}</tbody>
</table>

<div class="note">
  <p><strong>因子说明：</strong></p>
  <p>F1 动量20日做多最强：过去 20 日涨幅最大的 Top {TOP_N}</p>
  <p>F2 动量5日做多最强：过去 5 日涨幅最大的 Top {TOP_N}</p>
  <p>F3 反转5日做多最弱：过去 5 日跌幅最大的 Top {TOP_N}（反转策略）</p>
  <p>F4 反转10日做多最弱：过去 10 日跌幅最大的 Top {TOP_N}</p>
  <p>F5 创20日新高：收盘价 / 20日最高价 最接近 1 的 Top {TOP_N}</p>
  <p>F6 量比最高：近 5 日均量 / 20 日均量最大（放量）</p>
  <p>F7 量比最低：近 5 日均量 / 20 日均量最小（缩量）</p>
  <p>F8 近高位+MA多头：MA 多头排列 + 收盘接近20日高点</p>
  <p>F9 近高位+低波动：接近20日高点 + ATR/价格最低（稳步上涨）</p>
  <p>F10 反转+缩量：5日跌幅最大 + 量比&lt;0.8（超跌+卖压衰竭）</p>
  <p>F11 稳健强势：20日动量>0 + 近高位 + MA多头（综合评分）</p>
  <p>benchmark：全市场等权平均收益</p>
</div>

<script>
new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels: {labels_js},
    datasets: [{','.join(datasets_js)}]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      title: {{ display: true, text: '累计净值曲线', color: '#fff', font: {{ size: 16 }} }},
      legend: {{ labels: {{ color: '#ccc', font: {{ size: 12 }} }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#888', maxTicksLimit: 20 }}, grid: {{ color: '#222' }} }},
      y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }},
           title: {{ display: true, text: '累计净值', color: '#888' }} }},
    }},
  }},
}});
</script>
</body>
</html>"""

    out = OUTPUT_DIR / "factor_comparison.html"
    out.write_text(html, encoding="utf-8")
    print(f"[HTML] {out}")


def main():
    print("=" * 70)
    print("  全 A 股最强标的选择器 — 5 日持有期最优因子研究")
    print("=" * 70)
    t0 = time.time()

    all_df = load_all_stocks()
    all_df = compute_factors(all_df)
    results = run_factor_backtest(all_df)
    df_summary = analyze_results(results)
    generate_html_report(results, df_summary)

    print(f"\n[总耗时] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
