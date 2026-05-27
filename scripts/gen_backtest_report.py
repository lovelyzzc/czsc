"""生成「涨停突破中枢上沿」回测统计 HTML 报告"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT = Path(__file__).resolve().parent / "_output"

with open(OUTPUT / "zs_breakout_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

df = pd.read_csv(OUTPUT / "zs_breakout_signals.csv")

# 按年统计
df["year"] = df["date"].astype(str).str[:4].astype(int)
yearly = []
for yr, g in df.groupby("year"):
    v20 = g[g["ret_20d"].notna()]
    if len(v20) < 10:
        continue
    yearly.append({
        "year": int(yr),
        "count": len(g),
        "win_rate": round((v20["ret_20d"] > 0).sum() / len(v20) * 100, 1),
        "mean_ret": round(v20["ret_20d"].mean(), 2),
        "bullish": round((v20["ret_20d"] > 5).sum() / len(v20) * 100, 1),
        "neutral": round(((v20["ret_20d"] >= -5) & (v20["ret_20d"] <= 5)).sum() / len(v20) * 100, 1),
        "bearish": round((v20["ret_20d"] < -5).sum() / len(v20) * 100, 1),
    })

yearly_rows = ""
for y in yearly:
    color_ret = "var(--red)" if y["mean_ret"] > 0 else "var(--green-stock)"
    yearly_rows += f"""<tr>
      <td class="num">{y['year']}</td>
      <td class="num">{y['count']}</td>
      <td class="num">{y['win_rate']}%</td>
      <td class="num" style="color:{color_ret}">{y['mean_ret']:+.2f}%</td>
      <td class="num">{y['bullish']}%</td>
      <td class="num">{y['neutral']}%</td>
      <td class="num">{y['bearish']}%</td>
    </tr>"""

# 002896 在样本中的记录
df_002896 = df[df["symbol"] == "002896.SZ"].sort_values("date")
rows_002896 = ""
for _, r in df_002896.iterrows():
    ret5 = f"{r['ret_5d']:+.1f}%" if pd.notna(r['ret_5d']) else "-"
    ret10 = f"{r['ret_10d']:+.1f}%" if pd.notna(r['ret_10d']) else "-"
    ret20 = f"{r['ret_20d']:+.1f}%" if pd.notna(r['ret_20d']) else "待定"
    rows_002896 += f"""<tr>
      <td>{r['date']}</td>
      <td class="num">{r['close']}</td>
      <td class="num" style="color:var(--red)">+{r['pct_chg']:.1f}%</td>
      <td class="num">{r['zs_zg']}</td>
      <td class="num">{r['breakout_pct']:+.1f}%</td>
      <td class="num">{ret5}</td><td class="num">{ret10}</td><td class="num">{ret20}</td>
    </tr>"""

# 收益分布直方图数据
hist_data_js = {}
for n in [5, 10, 20]:
    col = f"ret_{n}d"
    valid = df[df[col].notna()][col].values.tolist()
    hist_data_js[n] = valid

s5 = stats["5d"]
s10 = stats["10d"]
s20 = stats["20d"]

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>涨停突破中枢 · 回测统计报告</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg:#faf8f5;--text:#1a1a1a;--muted:#8a8580;--accent:#c0392b;
  --green-stock:#27ae60;--red:#c0392b;--blue:#2980b9;--orange:#e67e22;
  --card:#fff;--border:#e8e4df;--highlight:#fff8e1;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Sans','PingFang SC',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;max-width:1020px;margin:0 auto;padding:40px 24px 80px}}
.mono{{font-family:'IBM Plex Mono',monospace}}
h1{{font-size:28px;font-weight:700;margin-bottom:8px}}
h2{{font-size:20px;font-weight:600;margin:40px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}}
h3{{font-size:16px;font-weight:600;margin:24px 0 12px}}
.subtitle{{color:var(--muted);font-size:14px;margin-bottom:32px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.card--green{{border-left:4px solid var(--green-stock)}}
.card--red{{border-left:4px solid var(--red)}}
.card--blue{{border-left:4px solid var(--blue)}}
.card--orange{{border-left:4px solid var(--orange)}}
.card--highlight{{background:var(--highlight)}}
.tag{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;letter-spacing:.04em}}
.tag--green{{background:#e8f5e9;color:#2e7d32}}.tag--red{{background:#ffebee;color:#c62828}}
.tag--blue{{background:#e3f2fd;color:#1565c0}}.tag--orange{{background:#fff3e0;color:#e65100}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}}
th{{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}
.num{{font-family:'IBM Plex Mono',monospace}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:640px){{.grid-3,.grid-2{{grid-template-columns:1fr}}}}
.stat-box{{text-align:center;padding:16px}}
.stat-box .label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}}
.stat-box .value{{font-size:32px;font-weight:700;font-family:'IBM Plex Mono',monospace}}
.stat-box .sub{{font-size:12px;color:var(--muted);margin-top:4px}}
.bar-container{{display:flex;height:28px;border-radius:4px;overflow:hidden;margin:8px 0}}
.bar-green{{background:#27ae60}}.bar-blue{{background:#2980b9}}.bar-red{{background:#c0392b}}
.bar-container span{{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:600}}
.prob-row{{display:flex;align-items:center;gap:12px;margin:6px 0}}
.prob-row .lbl{{width:100px;font-size:13px;font-weight:500}}
.prob-row .fill{{height:20px;border-radius:3px;display:flex;align-items:center;padding:0 8px;font-size:11px;color:#fff;font-weight:600}}
canvas{{width:100%;height:200px}}
.warning{{background:#fff3e0;border:1px solid #ffe0b2;border-radius:6px;padding:12px 16px;font-size:13px;color:#e65100;margin:16px 0}}
.footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center}}
ul{{padding-left:20px}}li{{margin:4px 0;font-size:14px}}
.compare-old{{text-decoration:line-through;color:var(--muted);font-size:13px}}
.compare-new{{color:var(--accent);font-weight:600}}
</style>
</head><body>

<h1>涨停突破中枢 · 历史回测统计</h1>
<div class="subtitle">
  信号定义: 日线涨停 (&ge;9.5%) &amp; 收盘价突破最近完成中枢上沿 (ZG) &amp; 前一日收盘在 ZG 附近/下方
  &nbsp;|&nbsp; 样本 {stats['total']:,} 次 &nbsp;|&nbsp; 覆盖 {stats['stocks']:,} 只 A 股 &nbsp;|&nbsp; {stats['date_range'][0]} ~ {stats['date_range'][1]}
</div>

<!-- ===== 核心数据 ===== -->
<h2>一、核心统计概览</h2>

<div class="grid-3">
  <div class="card"><div class="stat-box">
    <div class="label">信号总数</div>
    <div class="value">{stats['total']:,}</div>
    <div class="sub">覆盖 {stats['stocks']:,} 只股票</div>
  </div></div>
  <div class="card"><div class="stat-box">
    <div class="label">20日平均收益</div>
    <div class="value" style="color:var(--red)">{s20['mean_ret']:+.2f}%</div>
    <div class="sub">中位数 {s20['median_ret']:+.2f}%</div>
  </div></div>
  <div class="card"><div class="stat-box">
    <div class="label">20日胜率</div>
    <div class="value">{s20['win_rate']}%</div>
    <div class="sub">收盘价 > 信号日收盘价</div>
  </div></div>
</div>

<!-- ===== 走势分类概率 ===== -->
<h2>二、走势分类概率（数据驱动）</h2>

<div class="card card--highlight">
  <h3>与此前主观估计的对比</h3>
  <table>
    <tr>
      <th>走势类型</th><th>定义</th>
      <th>主观估计</th><th>5日实际</th><th>10日实际</th><th>20日实际</th>
    </tr>
    <tr>
      <td><span class="tag tag--green">上涨 (突破)</span></td>
      <td>区间收益 &gt; +5%</td>
      <td class="compare-old">35%</td>
      <td class="num compare-new">{s5['bullish_pct']}%</td>
      <td class="num compare-new">{s10['bullish_pct']}%</td>
      <td class="num compare-new">{s20['bullish_pct']}%</td>
    </tr>
    <tr>
      <td><span class="tag tag--blue">震荡 (盘整)</span></td>
      <td>区间收益 &plusmn;5%</td>
      <td class="compare-old">40%</td>
      <td class="num compare-new">{s5['neutral_pct']}%</td>
      <td class="num compare-new">{s10['neutral_pct']}%</td>
      <td class="num compare-new">{s20['neutral_pct']}%</td>
    </tr>
    <tr>
      <td><span class="tag tag--red">下跌 (回落)</span></td>
      <td>区间收益 &lt; -5%</td>
      <td class="compare-old">25%</td>
      <td class="num compare-new">{s5['bearish_pct']}%</td>
      <td class="num compare-new">{s10['bearish_pct']}%</td>
      <td class="num compare-new">{s20['bearish_pct']}%</td>
    </tr>
  </table>
</div>

<h3>20日后走势分布</h3>
<div class="bar-container">
  <span class="bar-green" style="width:{s20['bullish_pct']}%">{s20['bullish_pct']}% 上涨</span>
  <span class="bar-blue" style="width:{s20['neutral_pct']}%">{s20['neutral_pct']}% 震荡</span>
  <span class="bar-red" style="width:{s20['bearish_pct']}%">{s20['bearish_pct']}% 下跌</span>
</div>

<h3>10日后走势分布</h3>
<div class="bar-container">
  <span class="bar-green" style="width:{s10['bullish_pct']}%">{s10['bullish_pct']}%</span>
  <span class="bar-blue" style="width:{s10['neutral_pct']}%">{s10['neutral_pct']}%</span>
  <span class="bar-red" style="width:{s10['bearish_pct']}%">{s10['bearish_pct']}%</span>
</div>

<h3>5日后走势分布</h3>
<div class="bar-container">
  <span class="bar-green" style="width:{s5['bullish_pct']}%">{s5['bullish_pct']}%</span>
  <span class="bar-blue" style="width:{s5['neutral_pct']}%">{s5['neutral_pct']}%</span>
  <span class="bar-red" style="width:{s5['bearish_pct']}%">{s5['bearish_pct']}%</span>
</div>

<div class="card card--red">
  <h3>关键发现：主观估计存在显著偏差</h3>
  <ul>
    <li><strong>下跌概率被严重低估</strong>：主观估计 25%，实际 20 日数据为 <strong>41.7%</strong>，几乎翻倍</li>
    <li><strong>上涨概率被高估</strong>：主观估计 35%，实际为 <strong>32.0%</strong>（20日），短期仅 25.5%（5日）</li>
    <li><strong>胜率不足五成</strong>：20日胜率仅 {s20['win_rate']}%，中位收益为负（{s20['median_ret']:+.2f}%）</li>
    <li><strong>「跌回中枢」概率高</strong>：20日内价格跌回ZG以下的概率为 <strong>{s20['back_below_zg_pct']}%</strong></li>
    <li><strong>右偏分布</strong>：均值为正（{s20['mean_ret']:+.2f}%），但中位数为负 → 少数大涨拉高了平均值，多数情况是下跌</li>
  </ul>
</div>

<!-- ===== 详细统计 ===== -->
<h2>三、收益分位数详情</h2>

<div class="card">
  <table>
    <tr>
      <th>持有期</th><th>样本数</th>
      <th>P10</th><th>P25</th><th>P50 (中位)</th><th>P75</th><th>P90</th>
      <th>平均最大涨幅</th><th>平均最大回撤</th>
    </tr>
    <tr>
      <td>5日</td><td class="num">{s5['samples']:,}</td>
      <td class="num">{s5['quantiles']['10']:+.2f}%</td>
      <td class="num">{s5['quantiles']['25']:+.2f}%</td>
      <td class="num">{s5['quantiles']['50']:+.2f}%</td>
      <td class="num">{s5['quantiles']['75']:+.2f}%</td>
      <td class="num">{s5['quantiles']['90']:+.2f}%</td>
      <td class="num" style="color:var(--red)">{s5['avg_max_gain']:+.2f}%</td>
      <td class="num" style="color:var(--green-stock)">{s5['avg_max_dd']:.2f}%</td>
    </tr>
    <tr>
      <td>10日</td><td class="num">{s10['samples']:,}</td>
      <td class="num">{s10['quantiles']['10']:+.2f}%</td>
      <td class="num">{s10['quantiles']['25']:+.2f}%</td>
      <td class="num">{s10['quantiles']['50']:+.2f}%</td>
      <td class="num">{s10['quantiles']['75']:+.2f}%</td>
      <td class="num">{s10['quantiles']['90']:+.2f}%</td>
      <td class="num" style="color:var(--red)">{s10['avg_max_gain']:+.2f}%</td>
      <td class="num" style="color:var(--green-stock)">{s10['avg_max_dd']:.2f}%</td>
    </tr>
    <tr>
      <td>20日</td><td class="num">{s20['samples']:,}</td>
      <td class="num">{s20['quantiles']['10']:+.2f}%</td>
      <td class="num">{s20['quantiles']['25']:+.2f}%</td>
      <td class="num">{s20['quantiles']['50']:+.2f}%</td>
      <td class="num">{s20['quantiles']['75']:+.2f}%</td>
      <td class="num">{s20['quantiles']['90']:+.2f}%</td>
      <td class="num" style="color:var(--red)">{s20['avg_max_gain']:+.2f}%</td>
      <td class="num" style="color:var(--green-stock)">{s20['avg_max_dd']:.2f}%</td>
    </tr>
  </table>
</div>

<!-- ===== 分年统计 ===== -->
<h2>四、分年度统计（20日口径）</h2>

<div class="card">
  <table>
    <tr><th>年份</th><th>信号次数</th><th>胜率</th><th>平均收益</th>
        <th>上涨(&gt;5%)</th><th>震荡(&plusmn;5%)</th><th>下跌(&lt;-5%)</th></tr>
    {yearly_rows}
  </table>
</div>

<!-- ===== 002896 历史信号 ===== -->
<h2>五、002896.SZ 中大力德 历史信号</h2>

<div class="card card--orange">
{"<p>002896.SZ 在本数据集中共触发 <strong>" + str(len(df_002896)) + "</strong> 次「涨停突破中枢」信号：</p>" if len(df_002896) > 0 else "<p>002896.SZ 在筛选条件下未发现历史信号（最近的 5/26 涨停为最新数据，将作为新样本追踪）</p>"}
  {"<table><tr><th>日期</th><th>收盘价</th><th>涨幅</th><th>ZG</th><th>突破幅度</th><th>5日后</th><th>10日后</th><th>20日后</th></tr>" + rows_002896 + "</table>" if len(df_002896) > 0 else ""}
</div>

<!-- ===== 修正后的交易建议 ===== -->
<h2>六、基于数据的交易建议修正</h2>

<div class="card card--highlight">
  <h3>原始交易计划需要修正的要点</h3>
  <table>
    <tr><th>原始建议</th><th>数据修正</th></tr>
    <tr>
      <td>情景 A (突破上涨) 概率 35%</td>
      <td>实际约 <strong>25~32%</strong>（5日口径更低），需降低预期</td>
    </tr>
    <tr>
      <td>情景 C (下跌回落) 概率 25%</td>
      <td>实际约 <strong>34~42%</strong>，下行风险远高于预期，这是最可能的结果</td>
    </tr>
    <tr>
      <td>策略1：追涨买入</td>
      <td>数据不支持涨停后直接追涨，5日中位收益 -1.66%，短期下行概率更大</td>
    </tr>
    <tr>
      <td>策略2：回踩低吸（推荐）</td>
      <td>仍然是更优策略。数据显示 {s20['back_below_zg_pct']}% 的概率会跌回ZG，等待回踩再入场更合理</td>
    </tr>
    <tr>
      <td>止损设在 75.75</td>
      <td>可以保留。P10 的 20日亏损约 -18%，止损 ~15% 能覆盖多数情况</td>
    </tr>
  </table>
</div>

<div class="card card--green">
  <h3>修正后的概率判断（002896 @ 89.68）</h3>
  <div class="prob-row">
    <span class="lbl">A. 突破上涨</span>
    <span class="fill bar-green" style="width:{s20['bullish_pct']*2.8}px">{s20['bullish_pct']}%</span>
  </div>
  <div class="prob-row">
    <span class="lbl">B. 区间震荡</span>
    <span class="fill bar-blue" style="width:{s20['neutral_pct']*2.8}px">{s20['neutral_pct']}%</span>
  </div>
  <div class="prob-row">
    <span class="lbl">C. 下跌回落</span>
    <span class="fill bar-red" style="width:{s20['bearish_pct']*2.8}px">{s20['bearish_pct']}%</span>
  </div>
  <p style="margin-top:12px;font-size:13px;color:var(--muted)">
    基于全 A 股 {stats['total']:,} 次同类信号的 20 日统计。002896 可能因个股特性（机器人板块、前期涨幅大）有偏差，数据仅供参考。
  </p>
</div>

<div class="card">
  <h3>数据驱动的操作建议</h3>
  <ul>
    <li><strong>不建议追涨</strong>：涨停次日直接买入的期望收益为负（5日中位 -1.66%），短期冲高回落是大概率事件</li>
    <li><strong>等待回踩</strong>：{s20['back_below_zg_pct']}% 的概率会在 20 日内跌回 ZG 以下，耐心等回踩 82~86 区间再考虑介入</li>
    <li><strong>严格止损</strong>：如果入场，止损不超过总资金 3%。42% 的概率会亏损超过 5%</li>
    <li><strong>关注少数大涨</strong>：均值为正 (+{s20['mean_ret']}%) 说明少数情况会大涨（P90 = +27%），但多数情况是亏损</li>
    <li><strong>缩短持有期</strong>：如果做短线，5日口径下震荡（40.6%）是主旋律，可考虑快进快出</li>
  </ul>
</div>

<div class="warning">
  <strong>免责声明：</strong>以上分析基于 {stats['date_range'][0]}~{stats['date_range'][1]} 期间全 A 股同类信号的历史统计，不构成投资建议。
  过去表现不代表未来，个股基本面、市场情绪、政策面等因素均会影响走势。投资有风险，入市需谨慎。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 全 A 股回测 · {stats['total']:,} 样本 · 生成于 2026-05-27
</div>

</body></html>"""

out = OUTPUT / "002896_backtest_report.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
