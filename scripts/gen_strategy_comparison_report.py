"""生成趋势起爆到加速策略对比 HTML 报告"""

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
by_calmar = sorted(stats, key=lambda x: x.get("卡玛比率", -99), reverse=True)
by_drawdown = sorted(stats, key=lambda x: x.get("最大回撤", 1))
by_winrate = sorted(stats, key=lambda x: x.get("pair_win_rate", 0) or 0, reverse=True)

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

STRATEGY_META = [
    {
        "tag": "1_MACD转势起爆",
        "phase": "起爆",
        "signals": "tas_dif_layer（DIF 零轴附近）+ tas_macd_direct（MACD 向上）+ bar_polyfit（加速上涨）",
        "logic": "DIF 穿越零轴标志趋势转多，MACD 柱子方向向上确认动量增强，价格拟合显示加速",
        "exit": "MACD 方向转下 或 笔向下",
        "color": "var(--gold)",
    },
    {
        "tag": "2_三连阳放量起爆",
        "phase": "起爆",
        "signals": "bar_triple（三K 新高涨 + 依次放量）+ tas_ma_system（均线多头排列）",
        "logic": "连续三根阳线创新高且成交量依次放大 — 最直观的起爆 K 线形态，均线多头排列确认方向",
        "exit": "笔向下",
        "color": "var(--orange)",
    },
    {
        "tag": "3_笔趋势动量加速",
        "phase": "起爆→加速",
        "signals": "cxt_bi_trend（笔上升趋势）+ tas_macd_power（MACD 强势）+ bar_accelerate（价格加速）",
        "logic": "笔高低点形成上升趋势（结构确认）+ MACD 处于强势档（非超强，避免追高）+ 价格加速上涨",
        "exit": "MACD 转弱 或 笔向下",
        "color": "var(--green)",
    },
    {
        "tag": "4_趋势动量加速",
        "phase": "加速",
        "signals": "bar_trend（60 日趋势多头）+ bar_bpm（绝对动量强势）+ bar_polyfit（加速上涨）",
        "logic": "60 日趋势跟踪确认多头方向 + 绝对动量达到强势水平 + 价格拟合显示加速上涨",
        "exit": "趋势转空 或 笔向下",
        "color": "var(--blue)",
    },
]


def _verdict(sharpe, calmar, pairs_count):
    if pairs_count < 500:
        return '<span class="tag tag--gray">样本偏少</span>'
    if calmar > 2:
        return '<span class="tag tag--gold">优秀 — 卡尔马 > 2</span>'
    if sharpe > 0.5:
        return '<span class="tag tag--green">推荐 — 夏普 > 0.5</span>'
    elif sharpe > 0.2:
        return '<span class="tag tag--blue">可用 — 夏普正</span>'
    elif sharpe > -0.1:
        return '<span class="tag tag--gray">需改进</span>'
    else:
        return '<span class="tag tag--red">不推荐</span>'


html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>趋势起爆到加速 — 策略回测</title>
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
.tag--gray{{background:#f5f5f5;color:#616161}}
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
.phase{{display:inline-flex;align-items:center;gap:6px;font-size:12px;margin:8px 0}}
.phase-dot{{width:8px;height:8px;border-radius:50%}}
.phase-label{{color:var(--muted);font-weight:500}}
</style>
</head><body>

<h1>趋势起爆到加速 — 策略回测</h1>
<div class="subtitle">
  聚焦趋势生命周期的"起爆→加速"窗口，用 4 种不同维度捕捉同一现象 &nbsp;|&nbsp;
  数据: 全 A 股 {_comma(total_stocks)} 只日线前复权 &nbsp;|&nbsp;
  回测区间: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')} &nbsp;|&nbsp;
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP
</div>

<div class="card" style="background:#f8f6f0;border:none;margin-bottom:32px">
  <h3 style="margin-top:0">趋势生命周期模型</h3>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;text-align:center;padding:16px 0;letter-spacing:1px;line-height:2">
    <span style="color:var(--muted)">底部整理</span>
    <span style="color:var(--muted);margin:0 4px">&rarr;</span>
    <span style="background:var(--gold);color:#fff;padding:4px 12px;border-radius:4px;font-weight:700">起爆</span>
    <span style="color:var(--muted);margin:0 4px">&rarr;</span>
    <span style="background:var(--green);color:#fff;padding:4px 12px;border-radius:4px;font-weight:700">加速</span>
    <span style="color:var(--muted);margin:0 4px">&rarr;</span>
    <span style="color:var(--muted)">巅峰</span>
    <span style="color:var(--muted);margin:0 4px">&rarr;</span>
    <span style="color:var(--muted)">衰竭</span>
    <br/>
    <span style="font-size:11px;color:var(--muted)">入场窗口 &uarr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;持有区间 &uarr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;出场 &uarr;</span>
  </div>
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="v">{_comma(total_stocks)}</div><div class="l">有效股票</div></div>
  <div class="kpi"><div class="v">{_comma(sum(s.get('pairs_count',0) for s in stats))}</div><div class="l">总交易笔数</div></div>
  <div class="kpi"><div class="v">4</div><div class="l">策略数</div></div>
  <div class="kpi"><div class="v">{FREQ}</div><div class="l">K线频率</div></div>
</div>

<!-- ===== 冠军 ===== -->
<h2>综合排名</h2>

<div class="grid-3">
  <div class="card card--gold">
    <div class="stat-box">
      <div class="label">最高卡尔马</div>
      <div class="value" style="color:var(--gold)">{by_calmar[0]['tag'].split('_',1)[1]}</div>
      <div class="sub">卡尔马 {_num(by_calmar[0].get('卡玛比率'))}</div>
    </div>
  </div>
  <div class="card card--green">
    <div class="stat-box">
      <div class="label">最高夏普</div>
      <div class="value" style="color:var(--green)">{by_sharpe[0]['tag'].split('_',1)[1]}</div>
      <div class="sub">夏普 {_num(by_sharpe[0].get('夏普比率'))}</div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">最低回撤</div>
      <div class="value" style="color:var(--blue)">{by_drawdown[0]['tag'].split('_',1)[1]}</div>
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

<div class="grid-2">
  <div class="card">
    <h3>夏普比率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("夏普比率"))}</span></div>' for i, s in enumerate(by_sharpe))}
  </div>
  <div class="card">
    <h3>卡尔马比率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("卡玛比率"))}</span></div>' for i, s in enumerate(by_calmar))}
  </div>
</div>

<!-- ===== 策略解读 ===== -->
<h2>二、策略详解</h2>"""

for i, s in enumerate(stats):
    tag = s["tag"]
    meta = STRATEGY_META[i]
    v = _verdict(s.get("夏普比率", 0), s.get("卡玛比率", 0), s.get("pairs_count", 0))

    html += f"""
<div class="strategy-card" style="border-left:4px solid {meta['color']}">
  <h3>{tag}
    <span class="tag" style="background:{meta['color']}20;color:{meta['color']};margin-left:8px">{meta['phase']}</span>
  </h3>
  <div class="grid-2">
    <div>
      <p><strong>信号组合</strong>：{meta['signals']}</p>
      <p><strong>逻辑</strong>：{meta['logic']}</p>
      <p><strong>出场</strong>：{meta['exit']}</p>
      <p><strong>覆盖</strong>：{_comma(s.get('stocks_count'))} 只 / {_comma(s.get('pairs_count'))} 笔交易</p>
    </div>
    <div>
      <p><strong>年化</strong>：{_pct(s.get('年化收益'))} &nbsp; <strong>夏普</strong>：{_num(s.get('夏普比率'))} &nbsp; <strong>卡尔马</strong>：{_num(s.get('卡玛比率'))}</p>
      <p><strong>回撤</strong>：{_pct(s.get('最大回撤'))} &nbsp; <strong>胜率</strong>：{_num(s.get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(s.get('单笔盈亏比'))}</p>
      <p><strong>波动率</strong>：{_pct(s.get('年化波动率'))} &nbsp; <strong>年胜率</strong>：{_pct(s.get('年胜率'))} &nbsp; <strong>持仓</strong>：{_num(s.get('持仓K线数'),1)} 日</p>
      <p>{v}</p>
    </div>
  </div>
</div>"""

html += f"""

<!-- ===== 核心发现 ===== -->
<h2>三、核心发现</h2>

<div class="card card--gold">
  <h3>「笔趋势动量加速」是最优策略</h3>
  <ul>
    <li><strong>卡尔马比率 8.80</strong> — 收益风险比极优秀。年化 3.73%，最大回撤仅 0.42%。</li>
    <li><strong>核心逻辑</strong>：用缠论笔结构（高低点递升）确认趋势已形成，用 MACD 动量强度（强势而非超强）确保还在加速而非到顶，用价格加速信号确认动量正在增强。三者共振准确捕捉"起爆→加速"窗口。</li>
    <li><strong>为什么 MACD「强势」而非「超强」</strong>：超强 = 已在巅峰，追高风险大；强势 = 正在加速，是最佳入场区间。</li>
    <li><strong>样本量</strong>：1,839 只股票、2,279 笔交易。虽然总量偏低但覆盖面广，统计有参考意义。</li>
  </ul>
</div>

<div class="card card--green">
  <h3>「MACD 转势起爆」是覆盖面最广的实用策略</h3>
  <ul>
    <li><strong>夏普 0.42 / 卡尔马 0.38</strong>，覆盖 5,267 只股票、48,072 笔交易 — 最具统计代表性。</li>
    <li>DIF 穿越零轴是经典趋势转多信号。叠加 MACD 方向和价格加速过滤掉了大量假突破。</li>
    <li>年化 1.51%，回撤 3.96%，适合作为基线策略或与其他策略组合使用。</li>
  </ul>
</div>

<div class="card">
  <h3>策略对比分析</h3>
  <table>
    <tr><th>维度</th><th>策略 1 MACD 起爆</th><th>策略 3 笔趋势加速</th></tr>
    <tr><td>识别阶段</td><td>起爆（DIF 穿零轴）</td><td>起爆→加速（笔趋势+MACD强）</td></tr>
    <tr><td>信号频率</td><td>高（48K 笔）</td><td>低（2.3K 笔）</td></tr>
    <tr><td>精度</td><td>中（夏普 0.42）</td><td>高（夏普 0.63）</td></tr>
    <tr><td>风控</td><td>中（回撤 4.0%）</td><td>极优（回撤 0.4%）</td></tr>
    <tr><td>适用场景</td><td>广覆盖、高频交易</td><td>精选、低频高质</td></tr>
  </table>
</div>

<div class="card card--blue">
  <h3>关键洞察：为什么「三连阳放量」效果差？</h3>
  <ul>
    <li>三连阳放量（策略 2）胜率仅 34.9%，夏普 0.03 — 几乎无效。</li>
    <li><strong>原因</strong>：三连阳放量是一个"描述性"而非"预测性"信号。它描述了已经发生的起爆，但价格往往已经涨过了最佳入场点。</li>
    <li><strong>对比</strong>：MACD 零轴 + 加速（策略 1）和笔趋势 + MACD 强势（策略 3）是"预测性"信号 — 它们通过趋势结构转变和动量层级变化，在起爆发生前或发生初期就给出信号。</li>
    <li><strong>启示</strong>：做趋势起爆策略，"结构+动量"组合优于"K线形态+量能"组合。</li>
  </ul>
</div>

<div class="card">
  <h3>推荐策略组合</h3>
  <table>
    <tr><th>定位</th><th>策略</th><th>核心指标</th><th>资金配比建议</th></tr>
    <tr><td><strong>核心</strong></td><td>笔趋势动量加速</td><td>卡尔马 8.80 / 回撤 0.42%</td><td>60%</td></tr>
    <tr><td><strong>卫星</strong></td><td>MACD 转势起爆</td><td>夏普 0.42 / 覆盖 5267 只</td><td>30%</td></tr>
    <tr><td><strong>观察</strong></td><td>趋势动量加速</td><td>夏普 0.28 / 回撤 1.3%</td><td>10%</td></tr>
    <tr><td><strong>剔除</strong></td><td>三连阳放量起爆</td><td>夏普 0.03 / 无效</td><td>0%</td></tr>
  </table>
</div>

<div class="warning">
  回测数据: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')}，全 A 股 {_comma(total_stocks)} 只日线前复权。
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP。以上不构成投资建议。
</div>

<div class="footer">
  czsc 趋势起爆到加速策略研究 · 全 A 股 {FREQ} · 4 大策略回测对比 · 2026-05-28
</div>

</body></html>"""

out = OUTPUT / "strategy_comparison.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
