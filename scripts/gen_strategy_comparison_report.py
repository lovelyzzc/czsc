"""生成新策略研究对比 HTML 报告（全 A 股日线数据 + WeightBacktest）"""

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
        "tag": "1_波动率压缩突破",
        "signals": "bar_zfzd_V241013（窄幅震荡）+ bar_break_V240428（收盘新高）+ vol_ti_suo_V221216（梯量价升）",
        "logic": "低波动窄幅整理后放量突破 20 日新高，捕捉压缩-爆发行情",
        "category": "突破",
        "color": "var(--orange)",
    },
    {
        "tag": "2_MACD零轴共振",
        "signals": "tas_cross_status_V230625（零轴下二次金叉）+ tas_dif_layer_V241010（DIF 零轴附近）+ tas_ma_system_V230513（均线多头排列）",
        "logic": "DIF 零轴附近第 2 次金叉配合 MA5/10/20 多头排列，趋势加速启动",
        "category": "趋势",
        "color": "var(--gold)",
    },
    {
        "tag": "3_TD序列反转",
        "signals": "bar_td9_V240616（神奇九转买点）+ tas_rsi_base_V230227（RSI 超卖）",
        "logic": "TD 计数到 9 标记趋势衰竭 + RSI<30 确认超卖，精准抄底",
        "category": "反转",
        "color": "var(--blue)",
    },
    {
        "tag": "4_量价背离反转",
        "signals": "bar_vol_bs1_V230224（量价极值买点）+ jcc_ten_mo_V221028（看涨吞没）",
        "logic": "成交量极值 + 看涨吞没形态双重确认底部反转",
        "category": "反转",
        "color": "var(--green)",
    },
    {
        "tag": "5_布林KDJ超卖反弹",
        "signals": "tas_boll_power_V221112（布林空头超强）+ tas_kdj_evc_V230401（KDJ 超卖转多）",
        "logic": "价格触及布林下轨极值 + KDJ 超卖金叉，均值回归反弹",
        "category": "均值回归",
        "color": "var(--red)",
    },
    {
        "tag": "6_双均线趋势回踩",
        "signals": "tas_double_ma_V221203（SMA5/20 强势多头）+ pressure_support_V240402（支撑位）+ xl_bar_position_V240328（相对低点）",
        "logic": "均线强势多头中回踩分型支撑位 + 处于 20 日相对低位，趋势回踩入场",
        "category": "趋势回踩",
        "color": "#8e44ad",
    },
]


def _verdict(sharpe, pairs_count):
    if pairs_count < 500:
        return '<span class="tag tag--gray">样本不足</span>'
    if sharpe > 0.7:
        return '<span class="tag tag--gold">推荐 — 夏普 > 0.7</span>'
    elif sharpe > 0.2:
        return '<span class="tag tag--blue">可用 — 夏普正</span>'
    elif sharpe > -0.1:
        return '<span class="tag tag--gray">需改进 — 夏普 ~0</span>'
    else:
        return '<span class="tag tag--red">不推荐 — 夏普为负</span>'


html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>新策略研究回测 — 全 A 股日线</title>
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
</style>
</head><body>

<h1>新策略研究回测 — 6 大策略</h1>
<div class="subtitle">
  告别传统缠论买卖点，基于技术指标多重确认构建全新策略体系 &nbsp;|&nbsp;
  数据: 全 A 股 {_comma(total_stocks)} 只日线前复权 &nbsp;|&nbsp;
  回测区间: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')} &nbsp;|&nbsp;
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP
</div>

<div class="kpi-grid">
  <div class="kpi"><div class="v">{_comma(total_stocks)}</div><div class="l">有效股票</div></div>
  <div class="kpi"><div class="v">{_comma(sum(s.get('pairs_count',0) for s in stats))}</div><div class="l">总交易笔数</div></div>
  <div class="kpi"><div class="v">6</div><div class="l">策略数</div></div>
  <div class="kpi"><div class="v">{FREQ}</div><div class="l">K线频率</div></div>
</div>

<!-- ===== 冠军 ===== -->
<h2>综合排名</h2>

<div class="grid-3">
  <div class="card card--gold">
    <div class="stat-box">
      <div class="label">最高夏普</div>
      <div class="value" style="color:var(--gold)">{by_sharpe[0]['tag'].split('_',1)[1]}</div>
      <div class="sub">夏普 {_num(by_sharpe[0].get('夏普比率'))}</div>
    </div>
  </div>
  <div class="card card--green">
    <div class="stat-box">
      <div class="label">最高卡尔马</div>
      <div class="value" style="color:var(--green)">{by_calmar[0]['tag'].split('_',1)[1]}</div>
      <div class="sub">卡尔马 {_num(by_calmar[0].get('卡玛比率'))}</div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">最高胜率</div>
      <div class="value" style="color:var(--blue)">{by_winrate[0]['tag'].split('_',1)[1]}</div>
      <div class="sub">交易胜率 {_num(by_winrate[0].get('pair_win_rate'),1)}%</div>
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
    meta = STRATEGY_META[i]
    v = _verdict(s.get("夏普比率", 0), s.get("pairs_count", 0))

    html += f"""
<div class="strategy-card" style="border-left:4px solid {meta['color']}">
  <h3>{tag} <span class="tag" style="background:{meta['color']}20;color:{meta['color']};margin-left:8px">{meta['category']}</span></h3>
  <div class="grid-2">
    <div>
      <p><strong>信号组合</strong>：{meta['signals']}</p>
      <p><strong>逻辑</strong>：{meta['logic']}</p>
      <p><strong>覆盖</strong>：{_comma(s.get('stocks_count'))} 只 / {_comma(s.get('pairs_count'))} 笔交易</p>
    </div>
    <div>
      <p><strong>年化</strong>：{_pct(s.get('年化收益'))} &nbsp; <strong>夏普</strong>：{_num(s.get('夏普比率'))} &nbsp; <strong>卡尔马</strong>：{_num(s.get('卡玛比率'))}</p>
      <p><strong>回撤</strong>：{_pct(s.get('最大回撤'))} &nbsp; <strong>胜率</strong>：{_num(s.get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(s.get('单笔盈亏比'))}</p>
      <p><strong>波动率</strong>：{_pct(s.get('年化波动率'))} &nbsp; <strong>年胜率</strong>：{_pct(s.get('年胜率'))} &nbsp; <strong>持仓周期</strong>：{_num(s.get('持仓K线数'),1)} 日</p>
      <p>{v}</p>
    </div>
  </div>
</div>"""

html += f"""

<!-- ===== 核心发现 ===== -->
<h2>四、核心发现</h2>

<div class="card card--gold">
  <h3>数据驱动的策略评估</h3>
  <ul>
    <li><strong>MACD 零轴共振（夏普 0.81）</strong>是唯一夏普 > 0.7 的策略，年化 1.16%、年胜率 100%、回撤仅 2.07%。零轴附近二次金叉 + 均线多头排列的组合逻辑最可靠。</li>
    <li><strong>量价背离反转（卡尔马 1.04）</strong>风险回报比最优，但 AND 条件过严导致仅 173 笔交易（170 只股票），样本不足以得出统计可靠结论。</li>
    <li><strong>TD 序列反转（胜率 65.0%）</strong>胜率最高但盈亏比仅 0.56 — 典型的"赢小亏大"结构，需增加利润保护机制。</li>
    <li><strong>双均线趋势回踩（夏普 0.37）</strong>回撤仅 0.38%，是最保守的策略，适合叠加其他策略做风控确认。</li>
    <li><strong>波动率压缩突破（夏普 0.21）</strong>逻辑成立但日线级别信号频率低（9857 笔），更适合分钟线级别。</li>
    <li><strong>布林 KDJ 超卖反弹（夏普 -0.31）</strong>明确失败 — 均值回归在日线级别不可靠，盈亏比 0.67 表明亏损笔远大于盈利笔。</li>
  </ul>
</div>

<div class="card">
  <h3>策略流派对比总结</h3>
  <table>
    <tr><th>流派</th><th>策略</th><th>结论</th><th>核心指标</th></tr>
    <tr><td><strong>趋势</strong></td><td>MACD 零轴共振</td><td style="color:var(--green)">推荐</td><td>夏普 0.81 / 卡尔马 0.56 / 回撤 2.1%</td></tr>
    <tr><td><strong>趋势回踩</strong></td><td>双均线趋势回踩</td><td style="color:var(--blue)">可用</td><td>夏普 0.37 / 回撤 0.4% / 胜率 49.2%</td></tr>
    <tr><td><strong>反转</strong></td><td>量价背离反转</td><td style="color:var(--muted)">样本不足</td><td>卡尔马 1.04 / 仅 173 笔</td></tr>
    <tr><td><strong>反转</strong></td><td>TD 序列反转</td><td style="color:var(--muted)">需优化</td><td>胜率 65% 但盈亏比 0.56</td></tr>
    <tr><td><strong>突破</strong></td><td>波动率压缩突破</td><td style="color:var(--muted)">需优化</td><td>夏普 0.21 / 建议用分钟线</td></tr>
    <tr><td><strong>均值回归</strong></td><td>布林 KDJ 超卖反弹</td><td style="color:var(--red)">失败</td><td>夏普 -0.31 / 盈亏比 0.67</td></tr>
  </table>
</div>

<div class="card card--green">
  <h3>下一步优化方向</h3>
  <ul>
    <li><strong>MACD 零轴共振</strong>：尝试增加成交量过滤（vol_ti_suo 梯量）或布林带通道确认提升入场精度</li>
    <li><strong>TD 序列反转</strong>：增加 pos_take / pos_holds 信号做移动止盈/保本，改善盈亏比</li>
    <li><strong>量价背离反转</strong>：放宽 AND 条件（去掉看涨吞没或改为 signals_any），增加交易频率</li>
    <li><strong>波动率压缩突破</strong>：尝试 30 分钟/60 分钟数据，日线信号频率太低</li>
    <li><strong>多策略组合</strong>：MACD 零轴共振（核心盈利）+ 双均线趋势回踩（风控确认）构建复合策略</li>
  </ul>
</div>

<div class="warning">
  回测数据: {stats[0].get('开始日期','')} ~ {stats[0].get('结束日期','')}，全 A 股 {_comma(total_stocks)} 只日线前复权。
  手续费: 双边 {FEE_RATE*2*10000:.0f}BP。以上不构成投资建议。
</div>

<div class="footer">
  czsc 新策略研究 · 全 A 股 {FREQ} · 6 大策略回测对比 · 2026-05-28
</div>

</body></html>"""

out = OUTPUT / "strategy_comparison.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
