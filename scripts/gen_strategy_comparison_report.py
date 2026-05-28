"""生成缠论 5 大策略对比 HTML 报告（真实 A 股日线数据 + WeightBacktest）"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "_output" / "strategy_comparison"

with open(OUTPUT / "comparison_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

total_stocks = max(s.get("stocks_count", 0) for s in stats)
FREQ = "日线"
FEE_RATE = 0.0002


def _pct(v, digits=1):
    if v is None or v != v:
        return "-"
    return f"{v*100:.{digits}f}%"


def _num(v, digits=2):
    if v is None or v != v:
        return "-"
    return f"{v:.{digits}f}"


def _comma(v):
    if v is None:
        return "-"
    return f"{int(v):,}"


def _color(v, good_positive=True):
    if v is None or v != v:
        return ""
    if good_positive:
        return "color:var(--green)" if v > 0 else "color:var(--red)" if v < 0 else ""
    else:
        return "color:var(--red)" if v > 0.3 else "color:var(--green)" if v < 0.1 else ""


by_sharpe = sorted(stats, key=lambda x: x.get("夏普比率", -99), reverse=True)
by_winrate = sorted(stats, key=lambda x: x.get("pair_win_rate", 0) or 0, reverse=True)
by_drawdown = sorted(stats, key=lambda x: x.get("最大回撤", 1))
by_calmar = sorted(stats, key=lambda x: x.get("卡玛比率", -99), reverse=True)

rows_main = ""
for s in stats:
    tag = s["tag"]
    rows_main += f"""<tr>
      <td><strong>{tag}</strong></td>
      <td class="num">{_comma(s.get('stocks_count'))}</td>
      <td class="num">{_comma(s.get('pairs_count'))}</td>
      <td class="num">{_num(s.get('pair_win_rate'),1)}%</td>
      <td class="num">{_num(s.get('单笔盈亏比'))}</td>
      <td class="num" style="{_color(s.get('年化收益'))}">{_pct(s.get('年化收益'))}</td>
      <td class="num">{_num(s.get('夏普比率'))}</td>
      <td class="num" style="{_color(s.get('最大回撤'), False)}">{_pct(s.get('最大回撤'))}</td>
      <td class="num">{_num(s.get('卡玛比率'))}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>缠论策略回测对比 — 全 A 股日线</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#faf8f5;--text:#1a1a1a;--muted:#8a8580;--green:#27ae60;--red:#c0392b;
--blue:#2980b9;--orange:#e67e22;--gold:#f39c12;--card:#fff;--border:#e8e4df}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Sans','PingFang SC',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;max-width:1100px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:28px;font-weight:700;margin-bottom:8px}}
h2{{font-size:20px;font-weight:600;margin:40px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}}
h3{{font-size:16px;font-weight:600;margin:24px 0 12px}}
.subtitle{{color:var(--muted);font-size:14px;margin-bottom:32px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.card--gold{{border-left:4px solid var(--gold);background:#fffdf5}}
.card--green{{border-left:4px solid var(--green)}}
.card--red{{border-left:4px solid var(--red)}}
.card--blue{{border-left:4px solid var(--blue)}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}}
th{{font-weight:600;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
.num{{font-family:'IBM Plex Mono',monospace}}
.tag{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}}
.tag--gold{{background:#fff8e1;color:#e65100}}.tag--green{{background:#e8f5e9;color:#2e7d32}}
.tag--red{{background:#ffebee;color:#c62828}}.tag--blue{{background:#e3f2fd;color:#1565c0}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:640px){{.grid-3,.grid-2{{grid-template-columns:1fr}}}}
.stat-box{{text-align:center;padding:16px}}
.stat-box .label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}}
.stat-box .value{{font-size:28px;font-weight:700;font-family:'IBM Plex Mono',monospace}}
.stat-box .sub{{font-size:12px;color:var(--muted);margin-top:4px}}
.rank{{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:14px}}
.rank .medal{{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff}}
.rank .gold{{background:var(--gold)}}.rank .silver{{background:#95a5a6}}.rank .bronze{{background:#cd7f32}}
ul{{padding-left:20px}}li{{margin:4px 0;font-size:14px}}
.strategy-card{{border:1px solid var(--border);border-radius:8px;padding:16px;background:#fff;margin:8px 0}}
.warning{{background:#fff3e0;border:1px solid #ffe0b2;border-radius:6px;padding:12px 16px;font-size:13px;color:#e65100;margin:16px 0}}
.footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:12px;text-align:center}}
.kpi .v{{font-size:22px;font-weight:700;font-family:'IBM Plex Mono',monospace}}
.kpi .l{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:4px}}
</style>
</head><body>

<h1>缠论 5 大策略回测对比</h1>
<div class="subtitle">
  数据: 全 A 股日线前复权 {_comma(total_stocks)} 只 (Tushare) &nbsp;|&nbsp;
  回测区间: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')} &nbsp;|&nbsp;
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP &nbsp;|&nbsp;
  耗时: {stats[0].get('elapsed_s',0):.0f}s
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="v">{_comma(total_stocks)}</div><div class="l">有效股票</div></div>
  <div class="kpi"><div class="v">{_comma(sum(s.get('pairs_count',0) for s in stats))}</div><div class="l">总交易笔数</div></div>
  <div class="kpi"><div class="v">5</div><div class="l">策略数</div></div>
  <div class="kpi"><div class="v">{FREQ}</div><div class="l">K线频率</div></div>
</div>

<!-- ===== 冠军 ===== -->
<h2>综合排名</h2>

<div class="grid-3">
  <div class="card card--gold">
    <div class="stat-box">
      <div class="label">最高夏普</div>
      <div class="value" style="color:var(--gold)">{by_sharpe[0]['tag']}</div>
      <div class="sub">夏普 {_num(by_sharpe[0].get('夏普比率'))}</div>
    </div>
  </div>
  <div class="card card--green">
    <div class="stat-box">
      <div class="label">最高卡尔马</div>
      <div class="value" style="color:var(--green)">{by_calmar[0]['tag']}</div>
      <div class="sub">卡尔马 {_num(by_calmar[0].get('卡玛比率'))}</div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">最小回撤</div>
      <div class="value" style="color:var(--blue)">{by_drawdown[0]['tag']}</div>
      <div class="sub">最大回撤 {_pct(by_drawdown[0].get('最大回撤'))}</div>
    </div>
  </div>
</div>

<!-- ===== 核心指标表 ===== -->
<h2>一、核心指标横向对比</h2>

<div class="card">
  <table>
    <tr>
      <th>策略</th><th>覆盖股票</th><th>交易笔数</th><th>胜率</th>
      <th>盈亏比</th><th>年化收益</th><th>夏普</th><th>最大回撤</th><th>卡尔马</th>
    </tr>
    {rows_main}
  </table>
</div>

<!-- ===== 分维度排名 ===== -->
<h2>二、分维度排名</h2>

<div class="grid-2">
  <div class="card">
    <h3>夏普比率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("夏普比率"))}</span></div>' for i, s in enumerate(by_sharpe))}
  </div>
  <div class="card">
    <h3>卡尔马比率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("卡玛比率"))}</span></div>' for i, s in enumerate(by_calmar))}
  </div>
  <div class="card">
    <h3>最大回撤排名 (越小越好)</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_pct(s.get("最大回撤"))}</span></div>' for i, s in enumerate(by_drawdown))}
  </div>
  <div class="card">
    <h3>交易胜率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("pair_win_rate"),1)}%</span></div>' for i, s in enumerate(by_winrate))}
  </div>
</div>

<!-- ===== 策略特征解读 ===== -->
<h2>三、策略特征解读</h2>"""

for i, s in enumerate(stats):
    tag = s["tag"]
    descriptions = [
        ("一买策略", "cxt_first_buy_V221126（纯笔结构一买）", "下跌趋势末端识别背驰，抄底入场", "var(--gold)"),
        ("二买策略", "cxt_second_bs_V230320 + SMA21 辅助", "一买后回踩不破前低，确认趋势反转", "var(--blue)"),
        ("三买策略", "cxt_third_buy_V230228 + cxt_third_bs_V230318 OR", "价格离开中枢后回踩不入中枢", "var(--orange)"),
        ("笔趋势跟踪", "cxt_bi_status_V230101（笔方向）", "笔向上开多、笔向下开空，始终持仓", "var(--red)"),
        ("背驰策略", "cxt_five_bi_V230619（aAb底背驰 + 类三买）", "通过多笔力度对比识别趋势衰竭", "var(--green)"),
    ]
    _, signal, logic, color = descriptions[i]
    sharpe = s.get("夏普比率", 0)
    calmar = s.get("卡玛比率", 0)

    if sharpe > 0.8:
        verdict = '<span class="tag tag--gold">推荐 — 夏普 > 0.8</span>'
    elif sharpe > 0.2:
        verdict = '<span class="tag tag--blue">可用 — 夏普正</span>'
    elif sharpe > -0.1:
        verdict = '<span class="tag tag--red">需改进 — 夏普 ~0</span>'
    else:
        verdict = '<span class="tag tag--red">不推荐 — 夏普为负</span>'

    html += f"""
<div class="strategy-card" style="border-left:4px solid {color}">
  <h3>{tag}</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：{signal}</p>
      <p><strong>逻辑</strong>：{logic}</p>
      <p><strong>覆盖</strong>：{_comma(s.get('stocks_count'))} 只 / {_comma(s.get('pairs_count'))} 笔交易</p>
    </div>
    <div>
      <p><strong>年化</strong>：{_pct(s.get('年化收益'))} &nbsp; <strong>夏普</strong>：{_num(s.get('夏普比率'))} &nbsp; <strong>卡尔马</strong>：{_num(s.get('卡玛比率'))}</p>
      <p><strong>回撤</strong>：{_pct(s.get('最大回撤'))} &nbsp; <strong>胜率</strong>：{_num(s.get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(s.get('单笔盈亏比'))}</p>
      <p><strong>波动率</strong>：{_pct(s.get('年化波动率'))} &nbsp; <strong>年胜率</strong>：{_pct(s.get('年胜率'))}</p>
      <p>{verdict}</p>
    </div>
  </div>
</div>"""

html += f"""

<!-- ===== 核心结论 ===== -->
<h2>四、核心结论</h2>

<div class="card card--gold">
  <h3>数据驱动的策略选择</h3>
  <ul>
    <li><strong>背驰策略（夏普 0.91）和一买策略（夏普 0.89）</strong>是唯二夏普 > 0.8 的策略，年胜率 100%，回撤 < 5%，是个人交易系统的核心策略。</li>
    <li><strong>二买策略（夏普 0.30）</strong>可作为辅助确认信号，单独使用收益有限。</li>
    <li><strong>三买策略（夏普 -0.02）</strong>在日线级别效果接近零，中枢突破假信号过多，需增加量能/均线过滤。</li>
    <li><strong>笔趋势跟踪（夏普 -0.47）</strong>明确不可用，日线非多即空交易噪音太大。</li>
  </ul>
</div>

<div class="card">
  <h3>推荐策略组合</h3>
  <table>
    <tr><th>定位</th><th>策略</th><th>核心指标</th></tr>
    <tr><td><strong>核心</strong></td><td>背驰策略</td><td>夏普 0.91 / 卡尔马 1.28 / 回撤 4.9%</td></tr>
    <tr><td><strong>核心</strong></td><td>一买策略</td><td>夏普 0.89 / 卡尔马 1.44 / 回撤 4.5%</td></tr>
    <tr><td><strong>辅助</strong></td><td>二买（叠加确认）</td><td>夏普 0.30 / 盈亏比 1.90</td></tr>
    <tr><td><strong>待优化</strong></td><td>三买 + 量能过滤</td><td>当前夏普 ~0，需增加确认条件</td></tr>
    <tr><td><strong>避免</strong></td><td>笔趋势跟踪</td><td>夏普 -0.47 / 回撤 36.7%</td></tr>
  </table>
</div>

<div class="warning">
  回测数据: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')}，全 A 股 {_comma(total_stocks)} 只日线前复权。
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP。以上不构成投资建议。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 全 A 股 {FREQ} · 5 大策略回测对比 · 2026-05-28
</div>

</body></html>"""

out = OUTPUT / "strategy_comparison.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
