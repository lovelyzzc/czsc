---
name: surge-delay5-stock-picker
description: >-
  主升浪 delay5 的推荐每日研究工作流：anticipate + 存活确认 + 市场状态门 + 固定硬过滤。
  用于运行研究候选扫描、检查存活确认、留存零候选日志或查看前向验证进度时。
  不用于默认十日结构观察池；不允许叠加未经验证的个股静态过滤或固定候选数量上限。
---

# delay5 存活确认策略候选

把本 skill 作为**唯一推荐的 delay5 研究扫描入口**。默认 `surge-regime-stock-picker` 只是
十日滚动结构观察池，不能当作实盘候选池。当前 `live_authorized=false`，本工作流只留档，
不产生交易授权。

本工作流只调用共享 `surge_live.detect_delay5`，与旧 surge/reversion 扫描解耦；所有股票统一
使用 active manifest 的安全截止交易日。候选、真实零候选日和市场状态均持续留档。

## 固定候选规则

1. **信号**：anticipate 启动（状态跳入 5 + 门控 量比≤0.8 / 散度≥3% / ret20≥8）；
2. **决策**：信号后第 5 个交易日收盘仍处上行家族 {5,6,7,8}（存活确认）；
3. **入场**：次日开盘（开盘逼近涨停 ≥板限-0.3% 则放弃）；
4. **市场状态门**：`high20_ratio > 0.12` 且 等权指数 > MA20，门关则只记录不开仓；
5. **硬过滤**：成交额≥1亿、止损带 8-20%、剔除 ST/退市风险；
6. **历史 FULL 退出**：SL2 + 收盘峰值 18% 跟踪 + 状态 9/10 后次日开盘全退 + 60 根上限；
7. **组合**：30 槽只属于旧容量敏感性，不是当前实盘配置。

不要加入 `ret5`、决策日量额比、状态仅限 7/8、市场广度 >15% 等额外静态门。这些规则虽能
减少 21%–50% 的候选，但低收益审计显示它们会在 IS/OOS 某一段降低槽位组合超额或删掉尾部赢家。
`priority` 只用于同日槽位不足时的执行顺序，不是收益预测器，也不是额外过滤条件。

## 工作流（三步）

以下命令均从仓库根目录运行；若依赖尚未安装，先执行 `uv sync --extra dev`。

### Step 1: 预检并发布最新数据

先运行不写 active manifest 的 dry-run，生成并验证增量计划、数据完整性和安全截止日；该模式只记录
工作树状态，不验证远端发布绑定：

```bash
PYTHONUNBUFFERED=1 uv run --no-sync python scripts/_sync_daily_data.py
```

确认计划无误后再用 `--apply`；正式发布会额外强制工作树 clean、当前提交已 push、分支/upstream/
remote URL 与远端 ref 全部匹配，全部通过才原子更新 active manifest：

```bash
PYTHONUNBUFFERED=1 uv run --no-sync python scripts/_sync_daily_data.py --apply
```

若预检失败，不得跳过门槛直接扫描；先修复失败项，再重新执行 dry-run。

### Step 2: 运行每日筛选

```bash
PYTHONUNBUFFERED=1 uv run --no-sync python .cursor/skills/surge-delay5-stock-picker/scripts/delay5_scan.py
```

全 A 股因果扫描约 1-2 分钟。输出市场状态门、全部 delay5 候选表和前向转正进度。
候选按 `priority` 展示；完整候选写入前向日志，不用固定数量上限截断。

候选数可以为零；专用入口仍会写出带固定 schema 的空 parquet，因此可区分“真实零候选”与
“扫描未运行”。停牌或陈旧股票不会把旧信号混入当前交易日。

### Step 3: 汇报结果

- 先报**市场状态门**：门关 → 明确“今日候选仅记录”；
- 门开 → 报全部可操作候选（代码、名称、收盘、状态、优先级、推荐止损、止损幅度），
  但不得把“可操作”解释成已经授权交易；
- 报**前向转正进度** X/60 笔；仅在达到 60 笔后触发预声明重检，不能提前把回测均值当作盈利预期。

## 必须随结果呈现的风险边界

- **生产共同期限未确认**：2024+ 行业 + 当前股本规模代理控制后，H5/H20/H60 的 HAC t
  分别为 1.186/0.785/1.569，期望块长 10 的 stationary-bootstrap 区间均跨零；
- **点时精确市值不解锁结果**：production 请求已完成 47,431/47,431 键、170/170 日期和
  286/286 处理票闭包；同行业 exact-mcap 1.5× K10/min5 支持全期 236/286、2024+ 143/174；
  但 ret20、vol20、价格和流动性匹配后 `|SMD|` 仍高于 0.1，审计状态为
  `BALANCE_INSUFFICIENT_OUTCOMES_NOT_EVALUATED`，不能查看 exact H5/H20/H60 或宣称因果零效应；
- **旧正值不可外推**：amount-only 和容量模拟口径未应用完整生产门控，且历史正值受少数长趋势、
  涨停/连板路径影响，不能作为生产选股 alpha；
- **规则冻结**：禁止依据本次候选或 2025 尾部继续调门控；下一轮只做 outcome-blind 协变量平衡，
  未通过覆盖率、positivity/ESS 与余额门之前继续锁定结果；
- **转正标准（预声明）**：前向 ≥60 笔可操作样本、超额 t≥2 **且中位数>0**；达标前仅记录。

## 输出文件（与 surge-regime-stock-picker 共用）

- `scripts/_output/surge_regime_picks/picks_exp_delay5_YYYY-MM-DD.parquet`：当日全部
  结构候选 + 市场门/过滤布尔列（前向日志，按决策日命名，重复运行幂等）；
- `scripts/_output/surge_regime_picks/market_state_live.parquet`：市场状态前向审计日志。

## 研究依据

- `scripts/S8_INCREMENTAL_VALIDATION_2026-08-11.md`：当前跨设备总报告与正式结论；
- `scripts/surge_delay5_production_cohort_audit.py`：生产顺序漏斗和成熟度身份；
- `scripts/delay5_common_horizon_att.py`：固定 5/20/60 日共同期限匹配；
- `scripts/delay5_factor_balance_audit.py`：决策日前动量、波动、价格与涨停路径余额审计；
- `scripts/delay5_pit_exact_mcap_plan.py`：点时精确市值 outcome-blind 冻结计划；
- `scripts/delay5_pit_exact_mcap_collector.py`：按决策日完整截面采集、内容寻址缓存与验证；
- `scripts/delay5_pit_exact_mcap_balance_audit.py`：单一 exact-mcap 主规格的结果盲覆盖与余额门；
- `scripts/s2b_same_fsm_path_audit.py`：历史精确支持集的同-FSM与路径敏感性。

## 研究文档索引

| 证据层 | 当前结论 |
|---|---|
| 生产漏斗 | 只把硬门、市场门、成交门和完整退出都通过的样本称为成熟 |
| 共同期限 | 行业/规模代理控制后 H5/H20/H60 全部未确认 |
| 因子余额 | 两套 K5/min3 规格全部期限区间跨零，且 ret20/vol20 仍未平衡 |
| 点时市值 | 47,431 键完整闭包；exact K10/min5 覆盖超过 80%，但余额失败且 outcome 未加载 |
| 同-FSM | 冻结 top5 仍为正，但继续占超额总和约 58%，极端涨停路径集中 |
| 总判定 | `live_authorized=false`；等待充分余额与独立前向成熟样本 |
