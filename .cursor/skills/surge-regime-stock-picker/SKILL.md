---
name: surge-regime-stock-picker
description: >-
  基于缠论11态状态机的主升浪结构观察：展示最近十日仍处上行家族的因果状态、止损和退出信息。
  用于主升浪结构复盘、状态机扫描、缠论走势状态观察或追踪已有候选；不用于策略候选或实盘筛选。
  用户要求减少候选、运行策略候选、存活确认买点或前向验证时，应改用 surge-delay5-stock-picker。
---

# 主升浪结构观察（缠论走势状态机）

旧 `surge-wave-stock-picker` 用 S1-S7 等权打分 + 全量 CZSC 后 `bi.edt<=dt` 过滤（轻微未来函数泄漏）；
本 skill 用 `scripts/trend_regime.py` 的流式 11 态状态机 + 原生 `ZS` 中枢，**对「今日」无未来数据、严格因果**，
是其改进版（两者并存）。

## 走势 11 态与买卖点

`0 不可交易 · 1 下跌 · 2 一买 · 3 二买 · 4 中枢构造 · 5 向上离开中枢 · 6 三买 ·
7 主升延续 · 8 加速主升 · 9 背驰衰竭 · 10 结构破坏`

- **主升浪启动（买点）**：
  - **确认追入**：状态跳变进入 `7/8` + 门控（量比≤0.8、MA5-MA20 散度≥3%）+ 启动前走过 `4→5`；
  - **启动埋伏**：状态跳变进入 `5` + 更强门控（再加 20 日涨幅≥8%）+ 走过 `4`。
- **卖点（统一）**：进入 `9 背驰`（减仓）/ `10 结构破坏`（清仓），叠加笔结构止损 SL2。

## 工作流（三步）

### Step 1: 同步最新数据

```bash
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/scripts/_sync_daily_data.py
```

确认输出含"完成"或"数据已是最新"后继续。

### Step 2: 运行每日扫描

```bash
# S0 模式（默认）— 仅主升浪候选
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-regime-stock-picker/scripts/daily_scan.py

# S7 分层策略（推荐）— S2b 核心(sp≥12) 全天候 + S2c 增量(sp≥10) 仅牛市，无均值回复
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-regime-stock-picker/scripts/daily_scan.py --strategy s7

# S4 环境自适应模式 — 牛市用 surge，熊/震荡用 reversion
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-regime-stock-picker/scripts/daily_scan.py --strategy s4

# 全信号模式 — 同时报告 surge + reversion 两类
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-regime-stock-picker/scripts/daily_scan.py --strategy all
```

全 A 股流式因果扫描约 1-2 分钟。

- `--strategy s0`（默认）：输出全部主升浪候选
- `--strategy s7`（推荐）：分层策略，S2b 核心（vr≤0.8, sp≥12）全天候展示，S2c 增量（sp≥10）仅在牛市补位。
  回测验证 OOS 5d +2.70%, Sharpe 1.775, 最大回撤 -6.94%（三种备选中最小）。报告含"层级"列（核心/增量）。
- `--strategy s4`：根据市场环境自动切换：牛市 = surge 追涨，熊/震荡 = reversion 均值回复
- `--strategy all`：同时报告 surge + reversion 两类信号

硬过滤会剔除 ST/退市风险、最近一日成交额 < 1 亿、止损幅度不在 8%-20% 的标的；阈值可用
`SURGE_PICKER_MIN_AMOUNT_E` / `SURGE_PICKER_STOP_MIN_PCT` / `SURGE_PICKER_STOP_MAX_PCT` 调整。

### Step 3: 汇报结果

呈现结构观察池的优先级前 20 只，并明确它们是状态观察对象、不是策略候选。
当用户要求可执行候选或减少候选数量时，运行
`.cursor/skills/surge-delay5-stock-picker/scripts/delay5_scan.py`；它会使用存活确认、市场状态门
和固定硬过滤，市场门开启日历史平均候选数为 8.68（P90=17）。

| 列 | 说明 |
|---|---|
| 等级 | A(优先级≥75 且通过硬过滤) / B(60-74 且通过硬过滤) / C(观察池或硬过滤未通过) |
| 当前状态 | 今日所处走势类型（5 离开/6 三买/7 主升/8 加速） |
| 启动方式 | 确认追入(进 7/8) / 启动埋伏(进 5) |
| 优先级 | 综合评分 = 主升强度(35) + 止损可控(25) + 信号新鲜度(20) + 状态质量(20) |
| score | 主升强度 0-100（量比/散度/涨幅/笔角度合成，当日值） |
| 量比/散度%/ret20% | 当日结构特征（注：为今日值，启动当日通常更强） |
| 成交额亿 | 最近一日成交额，默认要求 ≥1 |
| 推荐止损 | SL2 笔结构止损价（最近向下笔低点）/ 退化用中枢下沿 |
| 止损幅度% | (收盘价 − 推荐止损)/收盘价；默认要求 8%-20%，≤0 表示已穿透 |
| 过滤原因 | 进入观察池的硬过滤原因 |
| 新鲜度 | 距启动信号的交易日数（0=今日启动） |

## 止损止盈（数据驱动，源自 909 次主升浪研究）

| 阶段 | 规则 |
|---|---|
| 主升中段(7) | 笔结构止损 SL2 / 中枢下沿托底，让利润奔跑 |
| 加速段(8) | 浮盈后切换 ~18% 跟踪止损（主升浪区间内回撤 P75≈18%） |
| 背驰(9) | 减仓（首次背驰离场可锁定约 63% 峰值涨幅） |
| 结构破坏(10) | 清仓 |

## 边界

默认观察池的默认门控、优先级和十日回看没有通过实盘镜像的选股 alpha 检验，因此不能把它
当作交易建议或在其上继续调阈值。用它观察结构、核验止损和管理已有持仓；用
`surge-delay5-stock-picker` 生成策略候选。

## 输出文件

- S0/S4/all 扫描结果保存在 `scripts/_output/surge_regime_picks/picks_YYYY-MM-DD.parquet`
- S7 扫描结果保存在 `scripts/_output/surge_regime_picks/picks_s7_YYYY-MM-DD.parquet`（含"层级"列）
- 策略候选和前向日志由 delay5 skill 写入同目录的 `picks_exp_delay5_YYYY-MM-DD.parquet` 与
  `market_state_live.parquet`
