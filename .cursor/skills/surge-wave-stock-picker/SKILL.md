---
name: surge-wave-stock-picker
description: >-
  主升浪每日选股入口。默认委托到 surge-regime-stock-picker 的 11 态因果状态机扫描；
  历史 S1-S7 等权打分脚本仅作为研究对照，不再作为主推荐口径。
  触发场景：用户提到"主升浪选股"、"追涨选股"、"趋势选股"、"主升浪扫描"。
---

# 主升浪每日选股

> 当前主推荐口径已切换到 `surge-regime-stock-picker`：流式 11 态状态机 + 原生中枢，
> 对今日信号无未来函数。旧 S1-S7 等权版回测漂亮，但存在全量 CZSC 后按日期过滤的轻微泄漏风险，
> 仅保留作研究对照。

## 工作流

执行以下三步，每步完成后再进入下一步：

### Step 1: 同步最新数据

```bash
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/scripts/_sync_daily_data.py
```

确认输出包含"完成"或"数据已是最新"后继续。

### Step 2: 运行每日扫描

```bash
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-regime-stock-picker/scripts/daily_scan.py
```

扫描全 A 股，输出最近 10 个交易日内出现主升浪启动信号、当前仍处于 5/6/7/8 主升家族，
并通过实盘硬过滤的标的。硬过滤默认剔除 ST/退市风险、近一日成交额 < 1 亿、止损幅度不在 8%-20% 的标的；
可用环境变量 `SURGE_PICKER_MIN_AMOUNT_E` / `SURGE_PICKER_STOP_MIN_PCT` / `SURGE_PICKER_STOP_MAX_PCT` 调整。

### Step 3: 汇报结果

脚本会自动进行优先级评分并精选 Top 20 可操作标的。将精选结果呈现给用户：

| 列 | 说明 |
|---|---|
| 等级 | A(优先级≥75 且通过硬过滤) / B(60-74 且通过硬过滤) / C(观察池或硬过滤未通过) |
| 代码 | 股票代码 |
| 名称 | 股票名称 |
| 行业 | 所属行业 |
| 收盘价 | 最新收盘价 |
| 当前状态 | 5 向上离开 / 6 三买 / 7 主升延续 / 8 加速主升 |
| 启动方式 | 确认追入 / 启动埋伏 |
| 优先级 | 主升强度 + 止损可控 + 新鲜度 + 状态质量 |
| score | 主升强度 0-100 |
| 量比 / MA散度 / ret20 | 状态机今日因果特征 |
| 成交额亿 | 最近一日成交额，默认要求 >= 1 |
| 推荐止损 | SL2 笔结构止损价（最近向下笔低点），退化用中枢下沿 |
| 止损幅度% | (收盘价 - 推荐止损) / 收盘价，默认要求 8%-20% |
| 过滤原因 | 未进入精选时的硬过滤原因 |

## 旧 S1-S7 等权版

历史脚本仍在 `surge-wave-stock-picker/scripts/daily_scan.py`，用于复盘 S1-S7 等权特征：
S1 笔力加速、S2 力度比、S3 脱离中枢、S4 低点抬升、S5 MA 扩散、S6 DIF 加速、S7 涨幅确认。
不要把旧脚本输出作为默认交易清单。

## 输出文件

主推荐扫描结果保存在 `scripts/_output/surge_regime_picks/picks_YYYY-MM-DD.parquet`。
旧 S1-S7 对照结果保存在 `.cursor/scripts/_output/surge_wave_picks/picks_YYYY-MM-DD.parquet`。
