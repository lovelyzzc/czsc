"""生成缠论 6 大策略对比 HTML 报告"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "_output" / "strategy_comparison"

with open(OUTPUT / "comparison_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

def _pct(v, digits=1):
    if v is None or v != v:
        return "-"
    return f"{v*100:.{digits}f}%" if abs(v) < 10 else f"{v:.{digits}f}%"

def _num(v, digits=2):
    if v is None or v != v:
        return "-"
    return f"{v:.{digits}f}"

def _color(v, good_positive=True):
    if v is None or v != v:
        return ""
    if good_positive:
        return "color:var(--green)" if v > 0 else "color:var(--red)" if v < 0 else ""
    else:
        return "color:var(--red)" if v > 0.5 else "color:var(--green)" if v < 0.3 else ""

rows_main = ""
for s in stats:
    tag = s["tag"]
    rows_main += f"""<tr>
      <td><strong>{tag}</strong></td>
      <td class="num">{s['pairs_count']}</td>
      <td class="num">{_num(s.get('pair_win_rate'))}%</td>
      <td class="num">{_num(s.get('profit_loss_ratio'))}</td>
      <td class="num" style="{_color(s.get('年化收益'))}">{_pct(s.get('年化收益'))}</td>
      <td class="num">{_num(s.get('夏普比率'))}</td>
      <td class="num" style="{_color(s.get('最大回撤'), False)}">{_pct(s.get('最大回撤'))}</td>
      <td class="num">{_num(s.get('卡玛比率'))}</td>
      <td class="num">{_pct(s.get('年化波动率'))}</td>
    </tr>"""

# 排名
by_sharpe = sorted(stats, key=lambda x: x.get("夏普比率", 0), reverse=True)
by_winrate = sorted(stats, key=lambda x: x.get("pair_win_rate", 0) or 0, reverse=True)
by_drawdown = sorted(stats, key=lambda x: x.get("最大回撤", 1))
by_plr = sorted(stats, key=lambda x: x.get("profit_loss_ratio", 0) or 0, reverse=True)

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>缠论 6 大策略回测对比</title>
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
.bar-h{{display:flex;align-items:center;height:24px;border-radius:3px;overflow:hidden;margin:4px 0}}
.bar-h span{{display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:600;min-width:32px}}
.strategy-card{{border:1px solid var(--border);border-radius:8px;padding:16px;background:#fff;margin:8px 0}}
.warning{{background:#fff3e0;border:1px solid #ffe0b2;border-radius:6px;padding:12px 16px;font-size:13px;color:#e65100;margin:16px 0}}
.footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center}}
</style>
</head><body>

<h1>缠论 6 大策略回测对比</h1>
<div class="subtitle">
  数据: 模拟 30 分钟 K 线 21,920 根 (2018-01-01 ~ 2024-01-01) &nbsp;|&nbsp;
  回测区间: 2019-01-01 ~ 2024-01-01 &nbsp;|&nbsp; 手续费: 双边 4BP
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
      <div class="label">最高胜率</div>
      <div class="value" style="color:var(--green)">{by_winrate[0]['tag']}</div>
      <div class="sub">胜率 {_num(by_winrate[0].get('pair_win_rate'))}%</div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">最小回撤</div>
      <div class="value" style="color:var(--blue)">{by_drawdown[0]['tag']}</div>
      <div class="sub">回撤 {_pct(by_drawdown[0].get('最大回撤'))}</div>
    </div>
  </div>
</div>

<!-- ===== 核心指标表 ===== -->
<h2>一、核心指标横向对比</h2>

<div class="card">
  <table>
    <tr>
      <th>策略</th><th>交易次数</th><th>胜率</th><th>盈亏比</th>
      <th>年化收益</th><th>夏普</th><th>最大回撤</th><th>卡尔马</th><th>波动率</th>
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
    <h3>交易胜率排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("pair_win_rate"))}%</span></div>' for i, s in enumerate(by_winrate))}
  </div>
  <div class="card">
    <h3>最大回撤排名 (越小越好)</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_pct(s.get("最大回撤"))}</span></div>' for i, s in enumerate(by_drawdown))}
  </div>
  <div class="card">
    <h3>盈亏比排名</h3>
    {"".join(f'<div class="rank"><span class="medal {"gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""}">{i+1}</span>{s["tag"]} <span class="num" style="margin-left:auto">{_num(s.get("profit_loss_ratio"))}</span></div>' for i, s in enumerate(by_plr))}
  </div>
</div>

<!-- ===== 策略特征解读 ===== -->
<h2>三、策略特征解读</h2>

<div class="strategy-card" style="border-left:4px solid var(--red)">
  <h3>1. 一买策略 — 逆势抄底</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_first_buy_V221126（纯笔结构一买）</p>
      <p><strong>逻辑</strong>：下跌趋势末端识别背驰，抄底入场</p>
      <p><strong>平仓</strong>：笔向下即平</p>
    </div>
    <div>
      <p><strong>结果</strong>：夏普 {_num(stats[0].get('夏普比率'))}，年化 {_pct(stats[0].get('年化收益'))}</p>
      <p><strong>诊断</strong>：胜率仅 33.5%，逆势操作假信号多，趋势可能继续延伸。盈亏比 1.6 不足以弥补低胜率。</p>
      <p><span class="tag tag--red">不推荐单独使用</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--blue)">
  <h3>2. 二买策略 — 趋势确认回踩</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_second_bs_V230320 + SMA21 辅助</p>
      <p><strong>逻辑</strong>：一买后回踩不破前低，确认趋势反转</p>
      <p><strong>平仓</strong>：笔向下即平</p>
    </div>
    <div>
      <p><strong>结果</strong>：夏普 {_num(stats[1].get('夏普比率'))}，盈亏比 {_num(stats[1].get('profit_loss_ratio'))}</p>
      <p><strong>诊断</strong>：盈亏比 2.04 较好，但胜率 35.5% 仍偏低。适合作为确认信号与其他策略组合。</p>
      <p><span class="tag tag--blue">可作为辅助确认</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--green)">
  <h3>3. 三买策略 — 中枢突破</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_third_buy_V230228 + cxt_third_bs_V230318（OR 触发）</p>
      <p><strong>逻辑</strong>：价格离开中枢后回踩不入中枢，确认上移</p>
      <p><strong>平仓</strong>：笔向下即平</p>
    </div>
    <div>
      <p><strong>结果</strong>：夏普 <strong>{_num(stats[2].get('夏普比率'))}</strong>，年化 <strong>{_pct(stats[2].get('年化收益'))}</strong>，盈亏比 <strong>{_num(stats[2].get('profit_loss_ratio'))}</strong></p>
      <p><strong>诊断</strong>：综合表现第二好。盈亏比最高 2.23，顺势操作逻辑清晰。</p>
      <p><span class="tag tag--green">推荐 — 高盈亏比策略</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--orange)">
  <h3>4. 笔趋势跟踪 — 非多即空</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_bi_status_V230101（笔方向）</p>
      <p><strong>逻辑</strong>：笔向上开多、笔向下开空，始终持仓</p>
      <p><strong>平仓</strong>：反向开仓即平（反手）</p>
    </div>
    <div>
      <p><strong>结果</strong>：胜率 44.3%（最高之一），但回撤 <strong>{_pct(stats[3].get('最大回撤'))}</strong> 灾难性</p>
      <p><strong>诊断</strong>：交易频率过高（1447 笔），手续费和滑点磨损严重。波动率最大。</p>
      <p><span class="tag tag--red">不推荐 — 回撤过大</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid #9b59b6">
  <h3>5. 背驰策略 — 五笔形态</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：cxt_five_bi_V230619（aAb底背驰 + 类三买）</p>
      <p><strong>逻辑</strong>：通过多笔力度对比识别趋势衰竭</p>
      <p><strong>平仓</strong>：笔向下即平</p>
    </div>
    <div>
      <p><strong>结果</strong>：盈亏比 {_num(stats[4].get('profit_loss_ratio'))}，胜率 34.9%</p>
      <p><strong>诊断</strong>：与一买/二买类似的逆势特征，但盈亏比更好。适合有经验的交易者精选使用。</p>
      <p><span class="tag tag--blue">中性 — 需配合其他过滤</span></p>
    </div>
  </div>
</div>

<div class="strategy-card" style="border-left:4px solid var(--gold)">
  <h3>6. 多级别联立 — 大周期定方向 + 小周期入场</h3>
  <div class="grid-2">
    <div>
      <p><strong>信号</strong>：60分钟笔向上（方向过滤）+ 30分钟三买（入场）</p>
      <p><strong>逻辑</strong>：大级别趋势确认后，等小级别出现中枢突破买入</p>
      <p><strong>平仓</strong>：笔向下即平</p>
    </div>
    <div>
      <p><strong>结果</strong>：夏普 <strong>{_num(stats[5].get('夏普比率'))}</strong>（最高），回撤 <strong>{_pct(stats[5].get('最大回撤'))}</strong>（最小），胜率 <strong>{_num(stats[5].get('pair_win_rate'))}%</strong>（最高）</p>
      <p><strong>诊断</strong>：全面最优。多重过滤显著提升了信号质量，代价是交易次数最少（130笔）。</p>
      <p><span class="tag tag--gold">最佳推荐 — 综合最优</span></p>
    </div>
  </div>
</div>

<!-- ===== 核心结论 ===== -->
<h2>四、核心结论</h2>

<div class="card card--gold">
  <h3>数据驱动的策略选择建议</h3>
  <ul>
    <li><strong>最佳综合策略：多级别联立</strong> — 夏普最高 (0.66)、回撤最低 (29.3%)、胜率最高 (49.2%)。通过大级别方向过滤 + 小级别精准入场，有效过滤假信号。</li>
    <li><strong>最佳盈亏比策略：三买</strong> — 盈亏比 2.23 领先，夏普 0.63 接近最优，逻辑清晰易执行。</li>
    <li><strong>避免单独使用：一买 / 笔趋势跟踪</strong> — 一买逆势操作胜率太低，笔趋势回撤灾难性。</li>
    <li><strong>提升方向</strong>：所有策略的胜率都偏低（33~49%），说明单一买点信号不够强。建议组合多个维度（缠论结构 + 量价 + 均线）进行交叉确认。</li>
  </ul>
</div>

<div class="card">
  <h3>策略组合建议（个人交易系统）</h3>
  <table>
    <tr><th>场景</th><th>推荐策略</th><th>理由</th></tr>
    <tr>
      <td>主策略</td>
      <td><strong>多级别联立（60+30分钟）</strong></td>
      <td>综合指标最优，适合作为核心仓位</td>
    </tr>
    <tr>
      <td>辅助加仓</td>
      <td><strong>三买策略</strong></td>
      <td>盈亏比最高，在主策略持仓期间叠加三买信号加仓</td>
    </tr>
    <tr>
      <td>底部抄底（小仓位）</td>
      <td><strong>二买 + 背驰组合</strong></td>
      <td>盈亏比 > 2.0，限制仓位在总资金 10~20% 以下</td>
    </tr>
    <tr>
      <td>避免</td>
      <td>纯一买、纯笔趋势</td>
      <td>单独使用效果差，风险收益不匹配</td>
    </tr>
  </table>
</div>

<div class="warning">
  <strong>注意</strong>：回测使用模拟数据（mock K线），仅用于验证策略逻辑和框架可行性。
  实际交易中需使用真实历史数据、考虑滑点冲击成本、分品种测试稳健性。以上不构成投资建议。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 6 大策略回测对比 · 2026-05-28
</div>

</body></html>"""

out = OUTPUT / "strategy_comparison.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
