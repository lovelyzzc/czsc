"""生成缠论 5 大策略对比 HTML 报告（真实 A 股日线数据）"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "_output" / "strategy_comparison"

with open(OUTPUT / "comparison_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

total_stocks = max(s.get("stocks_count", 0) for s in stats)
FREQ = "日线"
STOP_LOSS = 500
TIMEOUT = 120
FEE_RATE = 0.0002


def _pct(v, digits=1):
    if v is None or v != v:
        return "-"
    return f"{v*100:.{digits}f}%" if abs(v) < 1 else f"{v:.{digits}f}%"


def _num(v, digits=2):
    if v is None or v != v:
        return "-"
    return f"{v:.{digits}f}"


def _comma(v):
    if v is None:
        return "-"
    return f"{int(v):,}"


# 排名
by_winrate = sorted(stats, key=lambda x: x.get("pair_win_rate", 0) or 0, reverse=True)
by_plr = sorted(stats, key=lambda x: x.get("profit_loss_ratio", 0) or 0, reverse=True)
by_drawdown = sorted(stats, key=lambda x: x.get("avg_max_drawdown", 1))
by_return = sorted(stats, key=lambda x: x.get("avg_return_pct", -1e9) or -1e9, reverse=True)

# 主表行
rows_main = ""
for s in stats:
    tag = s["tag"]
    wr = s.get("pair_win_rate")
    plr = s.get("profit_loss_ratio")
    ret = s.get("avg_return_pct")
    dd = s.get("avg_max_drawdown")
    ret_color = "color:var(--green)" if ret and ret > 0 else "color:var(--red)" if ret and ret < 0 else ""
    rows_main += f"""<tr>
      <td><strong>{tag}</strong></td>
      <td class="num">{_comma(s.get('stocks_count'))}</td>
      <td class="num">{_comma(s.get('pairs_count'))}</td>
      <td class="num">{_num(wr, 1)}%</td>
      <td class="num">{_num(plr)}</td>
      <td class="num" style="{ret_color}">{_num(ret, 0)} BP</td>
      <td class="num">{_pct(dd)}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>缠论策略回测对比 — 全 A 股日线</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#faf8f5;--text:#1a1a1a;--muted:#8a8580;--green:#27ae60;--red:#c0392b;
--blue:#2980b9;--orange:#e67e22;--gold:#f39c12;--card:#fff;--border:#e8e4df;--highlight:#fff8e1}}
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
  数据: 全 A 股日线前复权 {_comma(total_stocks)} 只 &nbsp;|&nbsp;
  数据源: ~/.ts_data_cache/a_stock_daily_qfq/ (Tushare) &nbsp;|&nbsp;
  手续费: 双边 4BP &nbsp;|&nbsp;
  耗时: {stats[0].get('elapsed_s', 0):.0f}s
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
      <div class="label">最高胜率</div>
      <div class="value" style="color:var(--gold)">{by_winrate[0]['tag']}</div>
      <div class="sub">胜率 {_num(by_winrate[0].get('pair_win_rate'),1)}%</div>
    </div>
  </div>
  <div class="card card--green">
    <div class="stat-box">
      <div class="label">最高盈亏比</div>
      <div class="value" style="color:var(--green)">{by_plr[0]['tag']}</div>
      <div class="sub">盈亏比 {_num(by_plr[0].get('profit_loss_ratio'))}</div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">最小回撤</div>
      <div class="value" style="color:var(--blue)">{by_drawdown[0]['tag']}</div>
      <div class="sub">平均回撤 {_pct(by_drawdown[0].get('avg_max_drawdown'))}</div>
    </div>
  </div>
</div>

<!-- ===== 核心指标表 ===== -->
<h2>一、核心指标横向对比</h2>

<div class="card">
  <table>
    <tr>
      <th>策略</th><th>覆盖股票</th><th>交易笔数</th><th>胜率</th>
      <th>盈亏比</th><th>单笔均收</th><th>平均回撤</th>
    </tr>
    {rows_main}
  </table>
  <div style="font-size:11px;color:var(--muted);margin-top:8px">
    * 单笔均收单位为 BP (basis points, 万分之一)；回撤为各股票平均最大回撤
  </div>
</div>

<!-- ===== 分维度排名 ===== -->
<h2>二、分维度排名</h2>

<div class="grid-2">
  <div class="card">
    <h3>交易胜率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("pair_win_rate"),1)}%</span></div>' for i, s in enumerate(by_winrate))}
  </div>
  <div class="card">
    <h3>盈亏比排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("profit_loss_ratio"))}</span></div>' for i, s in enumerate(by_plr))}
  </div>
  <div class="card">
    <h3>平均回撤排名 (越小越好)</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_pct(s.get("avg_max_drawdown"))}</span></div>' for i, s in enumerate(by_drawdown))}
  </div>
  <div class="card">
    <h3>单笔收益排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("avg_return_pct",0),0)} BP</span></div>' for i, s in enumerate(by_return))}
  </div>
</div>

<!-- ===== 策略特征解读 ===== -->
<h2>三、策略特征解读</h2>

<div class="strategy-card" style="border-left:4px solid var(--gold)">
  <h3>1. 一买策略 — 逆势抄底</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_first_buy_V221126（纯笔结构一买）</p>
      <p><strong>逻辑</strong>：下跌趋势末端识别背驰，抄底入场</p>
      <p><strong>平仓</strong>：笔向下即平</p>
      <p><strong>覆盖</strong>：{_comma(stats[0].get('stocks_count'))} 只股票 / {_comma(stats[0].get('pairs_count'))} 笔交易</p>
    </div>
    <div>
      <p><strong>胜率</strong>：{_num(stats[0].get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(stats[0].get('profit_loss_ratio'))}</p>
      <p><strong>单笔均收</strong>：{_num(stats[0].get('avg_return_pct'),0)} BP &nbsp; <strong>回撤</strong>：{_pct(stats[0].get('avg_max_drawdown'))}</p>
      <p><strong>诊断</strong>：胜率 41.7% 配合盈亏比 2.07，在全 A 验证下表现超预期。回撤最低 (9.3%)。逆势策略的假信号被大样本平均化。</p>
      <p><span class="tag tag--gold">惊喜 — 高盈亏比 + 低回撤</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--blue)">
  <h3>2. 二买策略 — 趋势确认回踩</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_second_bs_V230320 + SMA21 辅助</p>
      <p><strong>逻辑</strong>：一买后回踩不破前低，确认趋势反转</p>
      <p><strong>覆盖</strong>：{_comma(stats[1].get('stocks_count'))} 只 / {_comma(stats[1].get('pairs_count'))} 笔</p>
    </div>
    <div>
      <p><strong>胜率</strong>：{_num(stats[1].get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(stats[1].get('profit_loss_ratio'))}</p>
      <p><strong>单笔均收</strong>：{_num(stats[1].get('avg_return_pct'),0)} BP &nbsp; <strong>回撤</strong>：{_pct(stats[1].get('avg_max_drawdown'))}</p>
      <p><strong>诊断</strong>：交易频率较高（10.7 万笔），盈亏比 2.01 稳健，但胜率偏低拉低了整体收益。适合做辅助确认信号。</p>
      <p><span class="tag tag--blue">稳健 — 可作为辅助</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--orange)">
  <h3>3. 三买策略 — 中枢突破</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_third_buy_V230228 + cxt_third_bs_V230318（OR 触发）</p>
      <p><strong>逻辑</strong>：价格离开中枢后回踩不入中枢，确认上移</p>
      <p><strong>覆盖</strong>：{_comma(stats[2].get('stocks_count'))} 只 / {_comma(stats[2].get('pairs_count'))} 笔</p>
    </div>
    <div>
      <p><strong>胜率</strong>：{_num(stats[2].get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(stats[2].get('profit_loss_ratio'))}</p>
      <p><strong>单笔均收</strong>：{_num(stats[2].get('avg_return_pct'),0)} BP &nbsp; <strong>回撤</strong>：{_pct(stats[2].get('avg_max_drawdown'))}</p>
      <p><strong>诊断</strong>：在真实数据上单笔均收转负，说明日线级别三买容易被假突破欺骗。中枢突破需要配合量能确认或多周期过滤。</p>
      <p><span class="tag tag--red">需改进 — 日线假突破多</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--red)">
  <h3>4. 笔趋势跟踪 — 非多即空</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_bi_status_V230101（笔方向）</p>
      <p><strong>逻辑</strong>：笔向上开多、笔向下开空，始终持仓</p>
      <p><strong>覆盖</strong>：{_comma(stats[3].get('stocks_count'))} 只 / {_comma(stats[3].get('pairs_count'))} 笔</p>
    </div>
    <div>
      <p><strong>胜率</strong>：{_num(stats[3].get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(stats[3].get('profit_loss_ratio'))}</p>
      <p><strong>单笔均收</strong>：{_num(stats[3].get('avg_return_pct'),0)} BP &nbsp; <strong>回撤</strong>：{_pct(stats[3].get('avg_max_drawdown'))}</p>
      <p><strong>诊断</strong>：34.8 万笔交易，回撤 77.6% 灾难性。日线级别笔的切换过于频繁，手续费和滑点磨损严重。</p>
      <p><span class="tag tag--red">不推荐 — 回撤灾难</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--green)">
  <h3>5. 背驰策略 — 五笔形态</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_five_bi_V230619（aAb底背驰 + 类三买）</p>
      <p><strong>逻辑</strong>：通过多笔力度对比识别趋势衰竭</p>
      <p><strong>覆盖</strong>：{_comma(stats[4].get('stocks_count'))} 只 / {_comma(stats[4].get('pairs_count'))} 笔</p>
    </div>
    <div>
      <p><strong>胜率</strong>：{_num(stats[4].get('pair_win_rate'),1)}% &nbsp; <strong>盈亏比</strong>：{_num(stats[4].get('profit_loss_ratio'))}</p>
      <p><strong>单笔均收</strong>：{_num(stats[4].get('avg_return_pct'),0)} BP &nbsp; <strong>回撤</strong>：{_pct(stats[4].get('avg_max_drawdown'))}</p>
      <p><strong>诊断</strong>：盈亏比最高 (2.17)，单笔均收 11,040 BP。五笔形态识别的背驰位置质量较高，是最值得深入优化的策略。</p>
      <p><span class="tag tag--green">最佳推荐 — 盈亏比最高</span></p>
    </div>
  </div>
</div>

<!-- ===== 核心结论 ===== -->
<h2>四、核心结论（真实数据验证）</h2>

<div class="card card--gold">
  <h3>数据驱动的发现</h3>
  <ul>
    <li><strong>一买 + 背驰是核心策略</strong>：在全 A 股验证下，一买（盈亏比 2.07 / 回撤 9.3%）和背驰（盈亏比 2.17 / 单笔均收最高）表现最好，说明逆势抄底类策略在大样本下有正期望。</li>
    <li><strong>三买假突破严重</strong>：模拟数据中三买表现优秀（夏普 0.63），但真实数据单笔均收转负。日线级别中枢边界噪音大，需增加量能、均线等过滤条件。</li>
    <li><strong>笔趋势跟踪不可用</strong>：回撤 77.6%，交易 34.8 万笔，纯粹的噪音交易。日线级别不适合如此高频的策略。</li>
    <li><strong>胜率普遍 35~43%</strong>：缠论买点的胜率本身不高，但盈亏比 > 2.0 的策略（一买、二买、背驰）仍有正期望。核心在于「截断亏损、让利润奔跑」。</li>
  </ul>
</div>

<div class="card">
  <h3>推荐策略组合（个人交易系统）</h3>
  <table>
    <tr><th>定位</th><th>策略</th><th>理由</th></tr>
    <tr>
      <td><strong>核心策略</strong></td>
      <td>背驰策略（五笔形态）</td>
      <td>盈亏比 2.17 最高，单笔均收 11,040 BP，形态识别质量好</td>
    </tr>
    <tr>
      <td><strong>辅助策略</strong></td>
      <td>一买策略</td>
      <td>回撤仅 9.3% 全场最低，盈亏比 2.07，适合控制总回撤</td>
    </tr>
    <tr>
      <td><strong>确认信号</strong></td>
      <td>二买（叠加使用）</td>
      <td>盈亏比 2.01 稳健，可与一买/背驰交叉确认提升胜率</td>
    </tr>
    <tr>
      <td><strong>待优化</strong></td>
      <td>三买 + 量能过滤</td>
      <td>需增加成交量放大、均线多头等确认条件过滤假突破</td>
    </tr>
    <tr>
      <td><strong>避免</strong></td>
      <td>纯笔趋势跟踪</td>
      <td>日线级别噪音过大，回撤不可接受</td>
    </tr>
  </table>
</div>

<div class="card card--green">
  <h3>模拟 vs 真实数据差异</h3>
  <table>
    <tr><th>策略</th><th>模拟数据结论</th><th>真实数据结论</th><th>差异原因</th></tr>
    <tr>
      <td>一买</td><td><span class="tag tag--red">不推荐</span></td><td><span class="tag tag--gold">推荐</span></td>
      <td>真实市场的趋势反转更有规律，大样本平均化了假信号</td>
    </tr>
    <tr>
      <td>三买</td><td><span class="tag tag--green">推荐</span></td><td><span class="tag tag--red">需改进</span></td>
      <td>真实市场假突破远多于模拟数据，中枢边界噪音大</td>
    </tr>
    <tr>
      <td>背驰</td><td><span class="tag tag--blue">中性</span></td><td><span class="tag tag--green">最佳</span></td>
      <td>五笔形态在真实市场更具结构化，识别精度更高</td>
    </tr>
  </table>
</div>

<div class="warning">
  <strong>注意</strong>：回测数据为 2021-2026 年 A 股日线前复权（Tushare），覆盖 {_comma(total_stocks)} 只个股。
  结果基于特定的风控参数（止损 {STOP_LOSS} BP、超时 {TIMEOUT} 根）和手续费假设（双边 {FEE_RATE*2*10000:.0f} BP）。
  实际交易需考虑滑点、涨跌停限制、停牌等因素。以上不构成投资建议。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 全 A 股日线 5 大策略回测对比 · 2026-05-28
</div>

</body></html>"""

out = OUTPUT / "strategy_comparison.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
