---
name: surge-delay5-stock-picker
description: >-
  主升浪终态策略（anticipate + delay5 存活确认 + 市场状态门）的每日筛选：anticipate 信号后
  第 5 个交易日收盘仍在上行家族则次日开盘入场，叠加市场门与固定硬过滤，广度优先 30 槽。
  五轮研究定格口径，转正前仅记录非买入建议。触发场景：用户提到"delay5 选股"、"存活确认买点"、
  "终态策略每日筛选"、"主升浪回踩筛选"、"前向转正进度"。
---

# delay5 存活确认买点（终态口径）每日筛选

本 skill 是五轮 surge regime 研究（2026-06-10 → 06-12）定格的**终态策略**专用入口。
选股逻辑 100% 复用 `surge-regime-stock-picker` 的扫描引擎（同一代码路径，与研究镜像
逐字节一致），共用同一前向日志；本 skill 额外提供前向转正进度读数。旧 skill 保留
全量状态观察功能，两者并存。

## 策略定义（全部固定，不吃 `SURGE_PICKER_*` 环境变量）

1. **信号**：anticipate 启动（状态跳入 5 + 门控 量比≥1.2 / 散度≥3% / 立于中枢上方 / ret20≥8）；
2. **决策**：信号后第 5 个交易日收盘仍处上行家族 {5,6,7,8}（存活确认）；
3. **入场**：次日开盘（开盘逼近涨停 ≥板限-0.3% 则放弃）；
4. **市场状态门**：`high20_ratio > 0.12` 且 等权指数 > MA20，门关则只记录不开仓；
5. **硬过滤**：成交额≥1亿、止损带 8-20%、剔除 ST/退市风险；
6. **退出**：SL2 笔结构止损 + 浮盈后 18% 跟踪 + 背驰(9)减仓/破坏(10)清仓（FULL，第四轮证实不可再优化）；
7. **组合**：广度优先 **30 槽**（第五轮容量研究按预声明规则由 20 更新）、单笔小、不补仓。

## 工作流（三步）

### Step 1: 同步最新数据

```bash
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/scripts/_sync_daily_data.py
```

### Step 2: 运行每日筛选

```bash
PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python /home/lovelyzzc/czsc/.cursor/skills/surge-delay5-stock-picker/scripts/delay5_scan.py
```

全 A 股因果扫描约 1-2 分钟。输出三段：市场状态门（开/关）、delay5 候选表
（按优先级排序，含止损/幅度/过滤原因/可操作标记）、前向转正进度。

### Step 3: 汇报结果

- 先报**市场状态门**：门关（2026 环境常态）→ 明确"今日不开新仓，候选仅记录"；
- 门开 → 报可操作候选（代码/名称/收盘/状态/优先级/推荐止损/止损幅度），
  按 30 槽口径提示分散与单笔仓位；
- 报**前向转正进度** X/60 笔；达标时提示触发预声明重检（研究脚本算超额 t 与中位数）。

## 诚实定位（五轮研究结论，转正前必须随结果一并呈现）

- **肥尾彩票型分布**：2024+ OOS 毛超额均值 +7.58%（t=2.96）但**中位数为负**（-0.19%），
  top-10 笔占正收益 56%——典型一笔是小亏的，均值靠尾部彩票，不可把均值当预期；
- **无选股 alpha 证明**：随机对照（同日同成交额十分位）下仅 2025 年有显著正超额；
  "哪只会主升"在日线量价空间内不可预测（walk-forward AUC≈0.5）；
- **全日历诚实年化（20 槽口径）**：全史仅 +2.8%/年、最大回撤 43.9%（IS -17.4%、OOS +28.2%
  含微盘 beta）——本策略是「市场门开启期的环境押注」，不是全天候策略；
- **四层全部冻结**：入场阈值（轮二）、选股因子（轮三）、退出规则（轮四）、组合容量（轮五）
  均已按预声明判定定格，任何改动需先过 镜像 + 预声明标准；
- **转正标准（预声明）**：前向 ≥60 笔可操作样本、超额 t≥2 **且中位数>0**；达标前仅记录。

## 输出文件（与 surge-regime-stock-picker 共用）

- `scripts/_output/surge_regime_picks/picks_exp_delay5_YYYY-MM-DD.parquet`：当日全部
  结构候选 + 市场门/过滤布尔列（前向日志，按决策日命名，重复运行幂等）；
- `scripts/_output/surge_regime_picks/market_state_live.parquet`：市场状态前向审计日志。

## 研究文档索引

| 轮 | 文档（`scripts/`） | 结论 |
|---|---|---|
| 一 | `SURGE_REGIME_DELAY5_MIRROR_2026-06-11.md` | delay5 稳健 + 工程镜像 239/239 一致 |
| 二 | `SURGE_REGIME_SELECTION_AUDIT_2026-06-11.md` | 9 个选股条件零触发优化 → 阈值冻结 |
| 三 | `SURGE_REGIME_FACTOR_RESEARCH_2026-06-11.md` | 17 因子无增量、主升不可预测 → 因子冻结 |
| 四 | `SURGE_REGIME_EXIT_RESEARCH_2026-06-11.md` | 12 退出变体全败 → 退出冻结 |
| 五 | `SURGE_REGIME_CAPACITY_RESEARCH_2026-06-12.md` | 槽数 20→30、bd_confirm2 不转正、诚实年化 +2.8% |
