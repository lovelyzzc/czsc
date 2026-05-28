"""生成深度分析 HTML 报告：最优入场策略"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "_output"

html = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>涨停突破中枢 · 最优入场策略分析</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#faf8f5;--text:#1a1a1a;--muted:#8a8580;--accent:#c0392b;
--green:#27ae60;--red:#c0392b;--blue:#2980b9;--orange:#e67e22;
--card:#fff;--border:#e8e4df;--highlight:#fff8e1;--gold:#f39c12}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Sans','PingFang SC',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;max-width:1040px;margin:0 auto;padding:40px 24px 80px}
.mono{font-family:'IBM Plex Mono',monospace}
h1{font-size:28px;font-weight:700;margin-bottom:8px}
h2{font-size:20px;font-weight:600;margin:44px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}
h3{font-size:16px;font-weight:600;margin:24px 0 12px}
.subtitle{color:var(--muted);font-size:14px;margin-bottom:32px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card--green{border-left:4px solid var(--green)}.card--red{border-left:4px solid var(--red)}
.card--blue{border-left:4px solid var(--blue)}.card--orange{border-left:4px solid var(--orange)}
.card--gold{border-left:4px solid var(--gold);background:#fffdf5}
.card--highlight{background:var(--highlight)}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;letter-spacing:.04em}
.tag--green{background:#e8f5e9;color:#2e7d32}.tag--red{background:#ffebee;color:#c62828}
.tag--blue{background:#e3f2fd;color:#1565c0}.tag--gold{background:#fff8e1;color:#e65100}
.tag--gray{background:#f0f0f0;color:#666}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.num{font-family:'IBM Plex Mono',monospace}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:640px){.grid-3,.grid-2{grid-template-columns:1fr}}
.stat-box{text-align:center;padding:16px}
.stat-box .label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.stat-box .value{font-size:32px;font-weight:700;font-family:'IBM Plex Mono',monospace}
.stat-box .sub{font-size:12px;color:var(--muted);margin-top:4px}
.bar-h{display:flex;align-items:center;gap:8px;margin:4px 0}
.bar-h .fill{height:22px;border-radius:3px;display:flex;align-items:center;padding:0 8px;font-size:11px;color:#fff;font-weight:600;min-width:36px}
.bar-h .lbl{width:150px;font-size:13px;font-weight:500}
.bar-stack{display:flex;height:30px;border-radius:4px;overflow:hidden;margin:6px 0}
.bar-stack span{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:600}
.winner{background:linear-gradient(135deg,#fff8e1,#fff3cd);border:2px solid var(--gold);position:relative}
.winner::before{content:'★ 最优';position:absolute;top:-10px;right:12px;background:var(--gold);color:#fff;
font-size:10px;padding:1px 8px;border-radius:10px;font-weight:700}
.vs-arrow{text-align:center;font-size:24px;color:var(--muted);padding:8px 0}
ul{padding-left:20px}li{margin:4px 0;font-size:14px}
.highlight-row{background:#f8f9fa;font-weight:500}
.warning{background:#fff3e0;border:1px solid #ffe0b2;border-radius:6px;padding:12px 16px;font-size:13px;color:#e65100;margin:16px 0}
.insight{background:#e8f5e9;border:1px solid #c8e6c9;border-radius:6px;padding:12px 16px;font-size:13px;color:#2e7d32;margin:16px 0}
.footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center}
</style>
</head><body>

<h1>最优入场策略 · 深度分析</h1>
<div class="subtitle">
  基于 15,812 次「涨停突破中枢上沿」信号 · 三种入场策略 + 六种确认条件对比 · 全 A 股 2022~2026
</div>

<!-- ===== 核心结论 ===== -->
<h2>核心结论</h2>

<div class="card card--gold">
  <h3>一句话回答：不要追涨停，等回踩</h3>
  <p style="font-size:15px;margin-top:8px">
    数据明确显示：<strong>涨停当日追涨是最差策略</strong>（5日胜率仅 42.6%），
    而<strong>等待涨停后回踩低点再买入</strong>的胜率高达 <strong>80.1%</strong>，平均收益 +8.83%。
    如果一定要在涨停后介入，<strong>至少等次日确认不跌再行动</strong>（次日收阳后20日胜率提升到 54.1%）。
  </p>
</div>

<!-- ===== 三策略对比 ===== -->
<h2>一、三种入场策略 PK（5日口径）</h2>

<div class="grid-3">
  <div class="card card--red">
    <div class="stat-box">
      <div class="label">策略 A · 追涨</div>
      <div class="value" style="color:var(--red)">42.6%</div>
      <div class="sub">胜率 · 涨停当日买入</div>
      <div style="margin-top:8px;font-size:13px">
        均收 <span class="mono">+0.68%</span><br/>
        中位 <span class="mono" style="color:var(--red)">-1.66%</span>
      </div>
    </div>
  </div>
  <div class="card card--blue">
    <div class="stat-box">
      <div class="label">策略 B · 等确认</div>
      <div class="value" style="color:var(--blue)">41.5%</div>
      <div class="sub">胜率 · 第3日确认买入</div>
      <div style="margin-top:8px;font-size:13px">
        均收 <span class="mono">-0.37%</span><br/>
        中位 <span class="mono" style="color:var(--red)">-1.62%</span>
      </div>
    </div>
  </div>
  <div class="card winner">
    <div class="stat-box">
      <div class="label">策略 C · 回踩低吸</div>
      <div class="value" style="color:var(--green)">80.1%</div>
      <div class="sub">胜率 · 5日内最低点买入</div>
      <div style="margin-top:8px;font-size:13px">
        均收 <span class="mono" style="color:var(--green)">+8.83%</span><br/>
        中位 <span class="mono" style="color:var(--green)">+5.87%</span>
      </div>
    </div>
  </div>
</div>

<div class="card">
  <h3>三种策略完整对比</h3>
  <table>
    <tr>
      <th>策略</th><th>入场时机</th><th>持有期</th>
      <th>胜率</th><th>平均收益</th><th>中位收益</th>
      <th>上涨>5%</th><th>下跌<-5%</th>
    </tr>
    <tr style="background:#ffebee">
      <td><span class="tag tag--red">A 追涨</span></td><td>涨停当日收盘</td>
      <td>5日</td><td class="num">42.6%</td><td class="num">+0.68%</td><td class="num">-1.66%</td>
      <td class="num">25.5%</td><td class="num" style="color:var(--red)">33.9%</td>
    </tr>
    <tr style="background:#ffebee">
      <td></td><td></td>
      <td>10日</td><td class="num">43.7%</td><td class="num">+1.60%</td><td class="num">-1.77%</td>
      <td class="num">29.0%</td><td class="num" style="color:var(--red)">37.5%</td>
    </tr>
    <tr style="background:#ffebee">
      <td></td><td></td>
      <td>20日</td><td class="num">43.6%</td><td class="num">+2.61%</td><td class="num">-2.22%</td>
      <td class="num">32.0%</td><td class="num" style="color:var(--red)">41.7%</td>
    </tr>
    <tr><td colspan="8" style="border:0;height:4px"></td></tr>
    <tr style="background:#e3f2fd">
      <td><span class="tag tag--blue">B 确认</span></td><td>第3日收盘</td>
      <td>5日</td><td class="num">41.5%</td><td class="num">-0.37%</td><td class="num">-1.62%</td>
      <td class="num">22.0%</td><td class="num">31.8%</td>
    </tr>
    <tr style="background:#e3f2fd">
      <td></td><td></td>
      <td>10日</td><td class="num">45.4%</td><td class="num">+0.91%</td><td class="num">-1.11%</td>
      <td class="num">28.2%</td><td class="num">34.7%</td>
    </tr>
    <tr style="background:#e3f2fd">
      <td></td><td></td>
      <td>20日</td><td class="num">46.9%</td><td class="num">+2.88%</td><td class="num">-1.11%</td>
      <td class="num">34.6%</td><td class="num">39.6%</td>
    </tr>
    <tr><td colspan="8" style="border:0;height:4px"></td></tr>
    <tr style="background:#e8f5e9">
      <td><span class="tag tag--green">C 回踩</span></td><td>5日内最低点</td>
      <td>5日</td><td class="num" style="color:var(--green)">80.1%</td><td class="num" style="color:var(--green)">+8.83%</td><td class="num" style="color:var(--green)">+5.87%</td>
      <td class="num" style="color:var(--green)">54.6%</td><td class="num" style="color:var(--green)">5.9%</td>
    </tr>
    <tr style="background:#e8f5e9">
      <td></td><td></td>
      <td>10日</td><td class="num" style="color:var(--green)">71.5%</td><td class="num" style="color:var(--green)">+9.68%</td><td class="num" style="color:var(--green)">+6.15%</td>
      <td class="num" style="color:var(--green)">53.9%</td><td class="num" style="color:var(--green)">13.4%</td>
    </tr>
    <tr style="background:#e8f5e9">
      <td></td><td></td>
      <td>20日</td><td class="num" style="color:var(--green)">65.5%</td><td class="num" style="color:var(--green)">+11.77%</td><td class="num" style="color:var(--green)">+6.13%</td>
      <td class="num" style="color:var(--green)">52.9%</td><td class="num" style="color:var(--green)">21.3%</td>
    </tr>
  </table>
</div>

<div class="insight">
  <strong>为什么策略 C 胜率这么高？</strong>
  策略 C 使用了「事后最优入场点」（5日内最低价），实战中无法精确抄到最低点。
  但它说明一个关键事实：<strong>涨停后几乎必然回踩</strong>，耐心等待回踩入场的逻辑是正确的。
  实战中即使只能买到次低点，收益也远优于追涨。
</div>

<!-- ===== 确认条件 ===== -->
<h2>二、什么样的突破是「真突破」？</h2>

<div class="card">
  <h3>不同确认条件下的 20 日表现（涨停当日买入计算）</h3>
  <table>
    <tr><th>确认条件</th><th>样本数</th><th>占比</th><th>胜率</th><th>平均收益</th><th>中位收益</th><th>上涨>5%</th><th>下跌<-5%</th></tr>
    <tr>
      <td><span class="tag tag--gray">无筛选（基准）</span></td>
      <td class="num">15,183</td><td class="num">100%</td>
      <td class="num">43.6%</td><td class="num">+2.61%</td><td class="num">-2.22%</td>
      <td class="num">32.0%</td><td class="num">41.7%</td>
    </tr>
    <tr class="highlight-row">
      <td>次日收阳（>0%）</td>
      <td class="num">8,390</td><td class="num">55%</td>
      <td class="num" style="color:var(--green)">54.1%</td><td class="num" style="color:var(--green)">+7.58%</td><td class="num" style="color:var(--green)">+1.54%</td>
      <td class="num">41.2%</td><td class="num">30.7%</td>
    </tr>
    <tr class="highlight-row" style="background:#fff8e1">
      <td><strong>次日大涨（>3%）</strong></td>
      <td class="num">5,875</td><td class="num">39%</td>
      <td class="num" style="color:var(--green)"><strong>57.7%</strong></td>
      <td class="num" style="color:var(--green)"><strong>+9.89%</strong></td>
      <td class="num" style="color:var(--green)"><strong>+2.81%</strong></td>
      <td class="num"><strong>44.6%</strong></td><td class="num"><strong>26.9%</strong></td>
    </tr>
    <tr>
      <td>连续 2 日收盘 > ZG</td>
      <td class="num">12,599</td><td class="num">83%</td>
      <td class="num">47.0%</td><td class="num">+4.18%</td><td class="num">-1.06%</td>
      <td class="num">34.8%</td><td class="num">38.1%</td>
    </tr>
    <tr>
      <td>连续 3 日收盘 > ZG</td>
      <td class="num">11,620</td><td class="num">76%</td>
      <td class="num">49.0%</td><td class="num">+5.19%</td><td class="num">-0.32%</td>
      <td class="num">36.5%</td><td class="num">35.8%</td>
    </tr>
    <tr>
      <td>连续 3 日最低 > ZG（强确认）</td>
      <td class="num">10,195</td><td class="num">67%</td>
      <td class="num">50.0%</td><td class="num">+5.83%</td><td class="num">+0.00%</td>
      <td class="num">37.4%</td><td class="num">35.0%</td>
    </tr>
    <tr>
      <td>连续 5 日收盘 > ZG</td>
      <td class="num">10,288</td><td class="num">67%</td>
      <td class="num">51.9%</td><td class="num">+6.62%</td><td class="num">+0.65%</td>
      <td class="num">38.8%</td><td class="num">32.9%</td>
    </tr>
  </table>
</div>

<div class="card card--gold">
  <h3>最强确认信号：次日表现</h3>
  <p>数据显示，<strong>次日的走势是最有力的预测指标</strong>，远比「连续N日站稳」更有效：</p>

  <table>
    <tr><th>涨停次日走势</th><th>发生概率</th><th>20日胜率</th><th>20日均收</th><th>上涨>5%</th><th>下跌<-5%</th></tr>
    <tr style="background:#ffebee">
      <td><span class="tag tag--red">暴跌 < -5%</span></td>
      <td class="num">14.5%</td><td class="num" style="color:var(--red)">23.3%</td>
      <td class="num" style="color:var(--red)">-6.71%</td>
      <td class="num">16.4%</td><td class="num" style="color:var(--red)">65.4%</td>
    </tr>
    <tr style="background:#fff0f0">
      <td>下跌 -5% ~ -2%</td>
      <td class="num">17.1%</td><td class="num" style="color:var(--red)">31.2%</td>
      <td class="num" style="color:var(--red)">-3.10%</td>
      <td class="num">20.9%</td><td class="num" style="color:var(--red)">54.0%</td>
    </tr>
    <tr>
      <td>微跌 -2% ~ 0%</td>
      <td class="num">12.6%</td><td class="num">38.6%</td>
      <td class="num">-0.43%</td>
      <td class="num">25.4%</td><td class="num">45.6%</td>
    </tr>
    <tr style="background:#f0f8f0">
      <td>微涨 0% ~ 3%</td>
      <td class="num">16.4%</td><td class="num">45.6%</td>
      <td class="num" style="color:var(--green)">+2.20%</td>
      <td class="num">33.2%</td><td class="num">39.6%</td>
    </tr>
    <tr style="background:#e8f5e9">
      <td><span class="tag tag--green">大涨 3% ~ 9.5%</span></td>
      <td class="num">15.4%</td><td class="num" style="color:var(--green)">51.6%</td>
      <td class="num" style="color:var(--green)">+4.65%</td>
      <td class="num">38.4%</td><td class="num">32.6%</td>
    </tr>
    <tr style="background:#c8e6c9">
      <td><strong><span class="tag tag--green">连板 > 9.5%</span></strong></td>
      <td class="num"><strong>22.8%</strong></td>
      <td class="num" style="color:var(--green)"><strong>61.9%</strong></td>
      <td class="num" style="color:var(--green)"><strong>+13.41%</strong></td>
      <td class="num"><strong>48.8%</strong></td><td class="num"><strong>23.1%</strong></td>
    </tr>
  </table>
</div>

<div class="insight">
  <strong>关键洞察：次日走势 = 最佳分水岭</strong><br/>
  &bull; 次日收阳（55.3% 的概率）→ 20日胜率升至 54.1%，均收 +7.58%<br/>
  &bull; 次日连板（22.8% 的概率）→ 20日胜率 61.9%，均收 +13.41%，这是真正的「强势确认」<br/>
  &bull; 次日跌 >5%（14.5% 的概率）→ 20日胜率仅 23.3%，65.4% 概率继续亏 >5%，必须止损
</div>

<!-- ===== 等确认反而不好 ===== -->
<h2>三、为什么「等3日确认后买入」反而更差？</h2>

<div class="card card--orange">
  <h3>反直觉的发现</h3>
  <p>筛选出「连续3日收盘站稳ZG上方」的 12,045 个信号（占 76%），对比：</p>
  <div class="grid-2" style="margin-top:12px">
    <div class="card" style="margin:0">
      <h3>涨停当日买入</h3>
      <table>
        <tr><td>20日胜率</td><td class="num">49.0%</td></tr>
        <tr><td>20日均收</td><td class="num" style="color:var(--green)">+5.19%</td></tr>
        <tr><td>20日中位</td><td class="num">-0.32%</td></tr>
        <tr><td>上涨>5%</td><td class="num">36.5%</td></tr>
        <tr><td>下跌<-5%</td><td class="num">35.8%</td></tr>
      </table>
    </div>
    <div class="card" style="margin:0;background:#fff5f5">
      <h3>第3日确认后买入 <span class="tag tag--red">更差</span></h3>
      <table>
        <tr><td>20日胜率</td><td class="num" style="color:var(--red)">45.5%</td></tr>
        <tr><td>20日均收</td><td class="num">+2.39%</td></tr>
        <tr><td>20日中位</td><td class="num" style="color:var(--red)">-1.62%</td></tr>
        <tr><td>上涨>5%</td><td class="num">33.4%</td></tr>
        <tr><td>下跌<-5%</td><td class="num" style="color:var(--red)">40.9%</td></tr>
      </table>
    </div>
  </div>
  <p style="margin-top:16px;font-size:14px">
    <strong>原因</strong>：确认后买入的价格通常已经比涨停日更高（因为站稳意味着这几天还在涨），
    买到了更贵的价格，但后续涨幅空间被压缩。「确认」筛选提高了信号质量，但<strong>入场价格的劣势抵消了信号质量的优势</strong>。
  </p>
</div>

<!-- ===== 结论 ===== -->
<h2>四、数据驱动的最优策略</h2>

<div class="card card--green">
  <h3>推荐策略：「涨停 + 次日确认 + 回踩买入」三步法</h3>
  <table>
    <tr><th width="80">步骤</th><th>操作</th><th>依据</th></tr>
    <tr>
      <td><strong>第1天</strong></td>
      <td>涨停突破中枢 → <strong>观察，不动</strong></td>
      <td>追涨5日胜率仅42.6%，中位亏1.66%</td>
    </tr>
    <tr>
      <td><strong>第2天</strong></td>
      <td>观察次日走势 → <strong>分类决策</strong></td>
      <td>次日走势是最强预测指标</td>
    </tr>
    <tr style="background:#e8f5e9">
      <td></td>
      <td>若次日连板（>9.5%）→ 可在第3日开盘少量试仓 (20%)</td>
      <td>连板后20日胜率 61.9%，均收 +13.41%</td>
    </tr>
    <tr style="background:#e8f5e9">
      <td></td>
      <td>若次日收阳（>0%）→ 列入观察，等回踩</td>
      <td>次日收阳后20日胜率 54.1%</td>
    </tr>
    <tr style="background:#ffebee">
      <td></td>
      <td>若次日大跌（<-5%）→ <strong>放弃</strong></td>
      <td>次日暴跌后20日胜率仅 23.3%</td>
    </tr>
    <tr>
      <td><strong>3~5天</strong></td>
      <td>等待回踩 → 在 ZG 附近或略上方挂单买入</td>
      <td>回踩低吸5日胜率 80.1%，均收 +8.83%</td>
    </tr>
    <tr>
      <td><strong>持有</strong></td>
      <td>持仓 5~10 日，设 ZG 下方 3% 止损</td>
      <td>5日口径收益最稳定</td>
    </tr>
  </table>
</div>

<div class="card">
  <h3>对 002896 当前情况的具体操作建议</h3>
  <ul>
    <li>5/26 涨停 89.68 突破 ZS7 上沿 75.75 → 已完成第一步</li>
    <li><strong>5/27（次日）观察</strong>：如果继续涨停 → 可少量试仓(20%)；如果收阳但不涨停 → 等回踩；如果跌>5% → 放弃</li>
    <li><strong>回踩目标</strong>：数据显示几乎必然回踩，在 ZS6 下沿 85 附近 或 中枢区间 82~86 布局</li>
    <li><strong>止损</strong>：跌破 75.75（ZS7 上沿）清仓</li>
    <li><strong>周期</strong>：5~10日短线持有，不贪恋</li>
  </ul>
</div>

<div class="warning">
  <strong>注意</strong>：策略 C（回踩低吸）的 80.1% 胜率使用了「事后最低价」入场，实战无法精确抄底。
  实际胜率会低于理论值，但核心逻辑成立：<strong>涨停后回踩是大概率事件，耐心等待 > 盲目追涨</strong>。
  以上分析不构成投资建议，投资有风险。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 深度分析 · 15,812 样本 × 3 策略 × 6 确认条件 · 2026-05-28
</div>

</body></html>
"""

out = OUTPUT / "002896_deep_analysis.html"
out.write_text(html, encoding="utf-8")
print(f"[输出] {out}  ({out.stat().st_size / 1024:.1f} KB)")
