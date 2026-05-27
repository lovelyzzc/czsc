"""002896.SZ 中大力德 · 走势分类 + 交易计划 HTML 报告"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "_output" / "002896_trading_plan.html"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>002896.SZ 走势分类与交易计划</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #faf8f5; --text: #1a1a1a; --muted: #8a8580; --accent: #c0392b;
    --green: #27ae60; --red: #c0392b; --blue: #2980b9; --orange: #e67e22;
    --card: #fff; --border: #e8e4df; --highlight: #fff8e1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'IBM Plex Sans', 'PingFang SC', system-ui, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    max-width: 960px; margin: 0 auto; padding: 40px 24px 80px;
  }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
  h2 { font-size: 20px; font-weight: 600; margin: 40px 0 16px; padding-bottom: 8px;
       border-bottom: 2px solid var(--border); }
  h3 { font-size: 16px; font-weight: 600; margin: 24px 0 12px; }
  .subtitle { color: var(--muted); font-size: 14px; margin-bottom: 32px; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px 24px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .card--green { border-left: 4px solid var(--green); }
  .card--red { border-left: 4px solid var(--red); }
  .card--blue { border-left: 4px solid var(--blue); }
  .card--orange { border-left: 4px solid var(--orange); }
  .card--highlight { background: var(--highlight); }
  .tag {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.04em;
  }
  .tag--green { background: #e8f5e9; color: #2e7d32; }
  .tag--red { background: #ffebee; color: #c62828; }
  .tag--blue { background: #e3f2fd; color: #1565c0; }
  .tag--orange { background: #fff3e0; color: #e65100; }
  .tag--gray { background: #f5f5f5; color: #616161; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.08em; }
  .num { font-family: 'IBM Plex Mono', monospace; }
  .up { color: var(--red); }
  .down { color: var(--green); }
  .key-level { background: #f9f9f7; font-weight: 500; }
  .scenario { display: grid; grid-template-columns: 120px 1fr; gap: 12px; margin: 8px 0; }
  .scenario dt { font-weight: 600; color: var(--muted); font-size: 13px; }
  .scenario dd { font-size: 14px; }
  .prob-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .prob-bar__fill { height: 8px; border-radius: 4px; }
  .prob-bar__label { font-size: 12px; font-weight: 600; }
  .divider { border: 0; border-top: 1px dashed var(--border); margin: 24px 0; }
  .warning { background: #fff3e0; border: 1px solid #ffe0b2; border-radius: 6px;
             padding: 12px 16px; font-size: 13px; color: #e65100; margin: 16px 0; }
  .footer { margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border);
            color: var(--muted); font-size: 12px; text-align: center; }
  ul { padding-left: 20px; }
  li { margin: 4px 0; font-size: 14px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } }
  .price-box { text-align: center; padding: 16px; }
  .price-box .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                      letter-spacing: 0.1em; }
  .price-box .value { font-size: 32px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
</style>
</head>
<body>

<h1>002896.SZ 中大力德</h1>
<div class="subtitle">日线缠论走势分类与交易计划 &nbsp;|&nbsp; 数据截止 2026-05-26 &nbsp;|&nbsp; 最新收盘 <span class="mono up">89.68</span> (+10.00%)</div>

<!-- ==================== 第一部分：当前格局 ==================== -->
<h2>一、当前格局概述</h2>

<div class="grid-2">
  <div class="card">
    <div class="price-box">
      <div class="label">最新收盘</div>
      <div class="value up">89.68</div>
      <div style="color:var(--red);font-size:13px;margin-top:4px">涨停 +10.00%</div>
    </div>
  </div>
  <div class="card">
    <div class="price-box">
      <div class="label">历史最高 (前复权)</div>
      <div class="value" style="color:var(--muted)">109.35</div>
      <div style="color:var(--muted);font-size:13px;margin-top:4px">2025-09-18</div>
    </div>
  </div>
</div>

<div class="card">
  <h3>缠论结构摘要</h3>
  <table>
    <tr><th>要素</th><th>状态</th><th>备注</th></tr>
    <tr><td>已完成笔数</td><td class="num">49</td><td>最后完成笔 #48：向下，75.75 → 67.41</td></tr>
    <tr><td>当前运动</td><td><span class="tag tag--green">构建向上笔</span></td><td>从 67.41 低点反弹中，尚未形成完成笔</td></tr>
    <tr><td>最近中枢 (ZS7)</td><td class="num">ZD=70.30 ~ ZG=75.75</td><td>2026-03-20 ~ 2026-04-10，4笔构成</td></tr>
    <tr><td>前一个中枢 (ZS6)</td><td class="num">ZD=85.00 ~ ZG=92.58</td><td>2025-07-22 ~ 2025-09-04，15笔大中枢</td></tr>
    <tr><td>价格 vs ZS7</td><td class="up">大幅突破上沿 (+13.93)</td><td>涨停直接穿越</td></tr>
    <tr><td>价格 vs ZS6</td><td class="up">进入区间</td><td>89.68 处于 85.00~92.58 之间</td></tr>
    <tr><td>均线</td><td><span class="tag tag--green">多头排列</span></td><td>MA5 > MA10 > MA20 > MA60</td></tr>
    <tr><td>量比</td><td class="num up">2.64x</td><td>显著放量（20日均量的2.64倍）</td></tr>
  </table>
</div>

<div class="card card--highlight">
  <h3>大级别走势定位</h3>
  <p>从大级别看，002896 经历了一波完整的<strong>上涨-回调</strong>周期：</p>
  <ul>
    <li><strong>主升浪</strong>：2024-02 低点 16.62 → 2025-09 高点 109.35（+558%，历时 19 个月）</li>
    <li><strong>中继调整</strong>：2025-09 高点 109.35 → 2026-04 低点 67.41（-38.3%，历时 7 个月）</li>
    <li><strong>当前位置</strong>：调整后的反弹段，5/26 涨停至 89.68，进入前大中枢 ZS6 区间</li>
  </ul>
  <p style="margin-top:8px">回调幅度 38.3% 属于<strong>正常的黄金分割回调</strong>（0.382 位在 73.55 附近，实际低点 67.41 略深），当前反弹力度强劲。</p>
</div>

<!-- ==================== 第二部分：走势分类 ==================== -->
<h2>二、未来走势分类（三种可能性）</h2>

<!-- 情景 A -->
<div class="card card--green">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3 style="margin:0">情景 A：突破 ZS6 上沿，开启新一轮上攻</h3>
    <span class="tag tag--green">概率 35%</span>
  </div>
  <div class="prob-bar">
    <div class="prob-bar__fill" style="width:35%;background:var(--green)"></div>
    <span class="prob-bar__label" style="color:var(--green)">35%</span>
  </div>

  <dl class="scenario">
    <dt>触发条件</dt>
    <dd>放量突破 92.58（ZS6 上沿），且回踩不破 85.00（ZS6 下沿），形成<strong>第三类买点</strong></dd>
    <dt>目标区间</dt>
    <dd>第一目标 100-105 元（前高 109.35 附近 0.618 位）；强势则挑战 109.35 历史高点</dd>
    <dt>走势形态</dt>
    <dd>中枢上移格局确立 → ZS7 (70-75) → ZS6 (85-92) → 新高中枢</dd>
    <dt>验证信号</dt>
    <dd>
      <ul>
        <li>连续 2-3 日站稳 92.58 上方，不跌回</li>
        <li>突破时成交量维持在 20 日均量 1.5 倍以上</li>
        <li>回踩 85-88 区间时缩量企稳（量缩价稳）</li>
      </ul>
    </dd>
    <dt>支撑逻辑</dt>
    <dd>主升浪的回调幅度已达 38.2%，属于强势修正；涨停突破 ZS7 的力度非常强，资金明确做多意图</dd>
  </dl>
</div>

<!-- 情景 B -->
<div class="card card--blue">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3 style="margin:0">情景 B：在 ZS6 区间内震荡构筑新中枢</h3>
    <span class="tag tag--blue">概率 40%</span>
  </div>
  <div class="prob-bar">
    <div class="prob-bar__fill" style="width:40%;background:var(--blue)"></div>
    <span class="prob-bar__label" style="color:var(--blue)">40%</span>
  </div>

  <dl class="scenario">
    <dt>触发条件</dt>
    <dd>价格在 85.00~92.58 区间反复震荡，形成新的笔和中枢，但无法有效突破 92.58</dd>
    <dt>运行区间</dt>
    <dd>核心区间 82~95 元，上沿 92-95 承压，下沿 82-85 支撑</dd>
    <dt>走势形态</dt>
    <dd>进入 ZS6 区间后，至少构筑 3 笔以上的新中枢 → 方向选择延后</dd>
    <dt>验证信号</dt>
    <dd>
      <ul>
        <li>涨停后次日冲高回落，无法持续拉升</li>
        <li>在 85~93 区间反复震荡超过 2 周</li>
        <li>成交量逐步回落到均量水平</li>
        <li>均线走平，MA5 与 MA20 纠缠</li>
      </ul>
    </dd>
    <dt>后续演变</dt>
    <dd>震荡后方向二选一：向上突破走情景 A，向下破位走情景 C。此阶段以观望为主。</dd>
  </dl>
</div>

<!-- 情景 C -->
<div class="card card--red">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3 style="margin:0">情景 C：反弹结束，重新回落测试低点</h3>
    <span class="tag tag--red">概率 25%</span>
  </div>
  <div class="prob-bar">
    <div class="prob-bar__fill" style="width:25%;background:var(--red)"></div>
    <span class="prob-bar__label" style="color:var(--red)">25%</span>
  </div>

  <dl class="scenario">
    <dt>触发条件</dt>
    <dd>涨停后快速回落，跌破 81.50（ZS7 的 GG 位），随后跌破 75.75（ZS7 上沿）</dd>
    <dt>目标区间</dt>
    <dd>第一支撑 70-75（ZS7 区间）；若破则看 67.41 前低；极端情况测试 60 元</dd>
    <dt>走势形态</dt>
    <dd>5/26 涨停属于反弹末端的诱多，随后形成新的向下笔 → 继续下跌趋势</dd>
    <dt>验证信号</dt>
    <dd>
      <ul>
        <li>涨停次日即低开或高开低走，放量阴线</li>
        <li>3 日内跌破 82 元（MA5 附近），5 日内跌破 76 元</li>
        <li>成交量在下跌时持续放大</li>
        <li>出现顶分型后快速形成向下笔</li>
      </ul>
    </dd>
    <dt>风险因素</dt>
    <dd>涨停可能只是消息面驱动的脉冲行情（需关注是否有基本面利好）；回调 38% 后的反弹通常以 0.5 或 0.618 位为阻力，0.5 位在 88.38，已被穿越但未站稳</dd>
  </dl>
</div>

<!-- ==================== 第三部分：关键价位 ==================== -->
<h2>三、关键价位地图</h2>

<div class="card">
  <table>
    <tr><th>价位</th><th>性质</th><th>说明</th></tr>
    <tr class="key-level"><td class="num up">109.35</td><td>历史最高</td><td>2025-09-18 前复权高点，终极阻力</td></tr>
    <tr><td class="num">100 ~ 105</td><td>强阻力区</td><td>前高的 0.618 回撤 + 心理整数关口</td></tr>
    <tr class="key-level"><td class="num up">92.58</td><td>ZS6 上沿</td><td>突破确认 = 中枢上移，第三类买点成立</td></tr>
    <tr><td class="num" style="color:var(--orange)">89.68</td><td>当前价格</td><td>涨停收盘价，处于 ZS6 区间内</td></tr>
    <tr><td class="num">88.38</td><td>回调 0.5 位</td><td>109.35 → 67.41 的 50% 回撤位</td></tr>
    <tr class="key-level"><td class="num">85.00</td><td>ZS6 下沿</td><td>中枢区间底部，强支撑</td></tr>
    <tr><td class="num">81.50</td><td>ZS7 最高点 (GG)</td><td>回落第一防线</td></tr>
    <tr class="key-level"><td class="num down">75.75</td><td>ZS7 上沿</td><td>跌破 = 突破失败，需止损</td></tr>
    <tr><td class="num">70.30</td><td>ZS7 下沿</td><td>中枢底部</td></tr>
    <tr class="key-level"><td class="num down">67.41</td><td>本轮低点</td><td>2026-04-29 底分型，终极止损参考</td></tr>
  </table>
</div>

<!-- ==================== 第四部分：交易计划 ==================== -->
<h2>四、交易计划</h2>

<h3>策略 1：右侧突破策略（追涨型）</h3>
<div class="card card--green">
  <table>
    <tr><th width="100">项目</th><th>内容</th></tr>
    <tr><td>适用情景</td><td>情景 A</td></tr>
    <tr><td>入场条件</td><td>放量突破 <strong>92.58</strong>（ZS6 上沿），且次日确认不回落到 92 以下</td></tr>
    <tr><td>入场价格</td><td>93 ~ 95 元区间</td></tr>
    <tr><td>仓位</td><td>首次建仓 <strong>30%</strong> → 站稳 95 以上加仓至 <strong>50%</strong></td></tr>
    <tr><td>止损位</td><td><strong>85.00</strong>（ZS6 下沿），亏损幅度约 8~10%</td></tr>
    <tr><td>止盈位</td><td>第一目标 <strong>105</strong>（+11%）→ 第二目标 <strong>109</strong>（前高）→ 破前高后移动止盈</td></tr>
    <tr><td>盈亏比</td><td>约 <strong>1.3 : 1</strong>（第一目标）~ <strong>1.7 : 1</strong>（第二目标）</td></tr>
    <tr><td>持仓周期</td><td>预计 2~4 周</td></tr>
  </table>
</div>

<h3>策略 2：回踩买入策略（低吸型）⭐ 推荐</h3>
<div class="card card--blue">
  <table>
    <tr><th width="100">项目</th><th>内容</th></tr>
    <tr><td>适用情景</td><td>情景 A / B 的回踩阶段</td></tr>
    <tr><td>入场条件</td><td>涨停后回踩 <strong>82~86</strong> 区间（ZS6 下沿附近），出现<strong>底分型</strong>且缩量企稳</td></tr>
    <tr><td>入场价格</td><td>82 ~ 86 元区间</td></tr>
    <tr><td>仓位</td><td>82~86 分批建仓：86 元买 <strong>20%</strong>，84 元买 <strong>20%</strong>，82 元买 <strong>20%</strong>（最多 60%）</td></tr>
    <tr><td>止损位</td><td><strong>75.75</strong>（ZS7 上沿，跌破 = 突破失败），亏损幅度约 8~12%</td></tr>
    <tr><td>止盈位</td><td>第一目标 <strong>93</strong>（ZS6 上沿）→ 突破后持有目标 <strong>105</strong></td></tr>
    <tr><td>盈亏比</td><td>约 <strong>1.5 : 1</strong>（保守）~ <strong>2.5 : 1</strong>（突破后）</td></tr>
    <tr><td>持仓周期</td><td>预计 1~4 周</td></tr>
  </table>
</div>

<h3>策略 3：观望 / 防守策略</h3>
<div class="card card--orange">
  <table>
    <tr><th width="100">项目</th><th>内容</th></tr>
    <tr><td>适用情景</td><td>情景 B（震荡）/ 情景 C（下跌）</td></tr>
    <tr><td>操作</td><td>若已持仓：设定 <strong>75.75</strong> 为硬止损；在 85~93 区间内可做 T 降低成本</td></tr>
    <tr><td>空仓应对</td><td>价格在 85~93 震荡时不追涨，等待方向明确</td></tr>
    <tr><td>做空条件</td><td>跌破 75.75 且放量，可考虑融券（如可操作），目标 67~70</td></tr>
    <tr><td>重新评估</td><td>若跌到 67~70 区间再次出现底分型 + 放量，可视为新的低吸机会</td></tr>
  </table>
</div>

<!-- ==================== 第五部分：执行检查表 ==================== -->
<h2>五、每日跟踪检查表</h2>

<div class="card">
  <table>
    <tr><th>检查项</th><th>看多信号</th><th>看空信号</th></tr>
    <tr>
      <td>开盘表现</td>
      <td>高开 > 89 且不回补缺口</td>
      <td>低开 < 86 或高开低走</td>
    </tr>
    <tr>
      <td>成交量</td>
      <td>维持 15 万手以上（> 均量 1.3 倍）</td>
      <td>放量阴线（量增价跌）</td>
    </tr>
    <tr>
      <td>日内走势</td>
      <td>分时均线上方运行，尾盘拉升</td>
      <td>快速下杀，跌幅 > 5%</td>
    </tr>
    <tr>
      <td>K 线形态</td>
      <td>收中大阳线，或十字星后阳线</td>
      <td>长上影线 / 大阴线</td>
    </tr>
    <tr>
      <td>均线系统</td>
      <td>MA5 向上发散，MA20 拐头上行</td>
      <td>跌破 MA5 且 MA5 拐头向下</td>
    </tr>
    <tr>
      <td>缠论结构</td>
      <td>出现新的底分型 → 向上笔延伸</td>
      <td>出现顶分型 → 向下笔启动</td>
    </tr>
  </table>
</div>

<!-- ==================== 第六部分：风控 ==================== -->
<h2>六、风险控制</h2>

<div class="card card--red">
  <h3>硬性止损规则</h3>
  <ul>
    <li><strong>绝对止损</strong>：任何仓位，跌破 <strong>75.75</strong>（ZS7 上沿）立即清仓，不犹豫</li>
    <li><strong>时间止损</strong>：持仓超过 4 周仍未突破 92.58，减仓至 20% 以下</li>
    <li><strong>浮盈保护</strong>：浮盈超过 10% 后，回撤 50% 浮盈则减半仓位</li>
    <li><strong>单笔亏损</strong>：单次交易亏损不超过总资金的 <strong>3%</strong></li>
  </ul>
</div>

<div class="card">
  <h3>仓位管理</h3>
  <table>
    <tr><th>总资金占比</th><th>条件</th></tr>
    <tr><td class="num">20%</td><td>初始试探仓（回踩 85 附近入场）</td></tr>
    <tr><td class="num">40%</td><td>确认企稳（底分型出现 + 站上 MA5）</td></tr>
    <tr><td class="num">60%</td><td>突破 92.58 确认（三买信号成立）</td></tr>
    <tr><td class="num">≤ 60%</td><td>单只个股最大仓位上限，不满仓博弈</td></tr>
  </table>
</div>

<div class="warning">
  <strong>免责声明：</strong>以上分析基于缠论技术框架的结构性研判，仅供学习交流，不构成投资建议。
  股市有风险，投资需谨慎。实际交易前请结合基本面、资金面、市场情绪等因素综合判断。
</div>

<div class="footer">
  czsc 缠论量化分析框架 · 生成于 2026-05-27 · 数据来源 Tushare
</div>

</body>
</html>
"""

OUTPUT.write_text(HTML, encoding="utf-8")
print(f"[输出] {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.1f} KB)")
