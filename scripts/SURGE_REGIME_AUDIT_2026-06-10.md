# Surge Regime 策略审计报告（2026-06-10）—— 交接给下一轮迭代

> 本文是对「缠论 11 态走势状态机 + 主升浪启动选股」策略的完整审计结论与交接说明。
> 阅读对象：下一轮迭代的 agent/开发者。所有数字可由文末「复现命令」重新生成。

---

## 一、最终判定（预声明标准，不可回溯修改）

**判定标准（实验前声明）**：实盘镜像回测的 OOS 毛超额（vs 同日同成交额十分位随机对照）
≤0 或 |t|<2 → 收益主体为规模/市场 beta，不存在选股 alpha。

**判定结果：两种买点均未通过 → 该策略当前形态无可证明的选股 alpha，不构成可交易的盈利预期。**

| 入场 | 段 | 成交数(10槽) | 胜率 | 单笔净均值 | 毛超额 | t | 判定 |
|---|---|---|---|---|---|---|---|
| confirm 确认追入 | IS ≤2023 | 409 | 27.9% | -2.94% | -0.19% | -0.30 | — |
| confirm 确认追入 | OOS ≥2024 | 394 | 33.0% | +0.04% | +0.88% | 1.05 | **beta** |
| anticipate 启动埋伏 | IS ≤2023 | 522 | 29.3% | -2.80% | -0.69% | -1.61 | — |
| anticipate 启动埋伏 | OOS ≥2024 | 490 | 33.5% | -1.09% | -0.26% | -0.48 | **beta** |

组合级（槽位模型，净收益口径）：confirm 10 槽全期净年化 **-24.3%**、最大回撤 84%；
anticipate 更差（-34.8%）。20 槽结论相同。

**分年超额（confirm，10 槽）**——唯一的亮点与最重要的线索：

| 年 | n | 超额均值 | t |
|---|---|---|---|
| 2021 | 74 | -0.39% | -0.27 |
| 2022 | 194 | -1.29% | -1.43 |
| 2023 | 141 | +1.42% | 1.23 |
| 2024 | 180 | -0.71% | -0.89 |
| **2025** | **131** | **+4.98%** | **+2.50** |
| 2026YTD | 83 | -2.14% | -1.31 |

→ 信号只在 2025 类市场环境有超额，其余年份无效甚至为负。**强市场状态依赖**。

## 二、旧结论为何作废

历史版本曾报告「OOS 年化 240% / 夏普 6.35 / 卡玛 10+」（`surge_regime_backtest.py` 口径），
是**四重假象叠加**，逐项已被实证拆解：

1. **WeightBacktest 稀疏组合年化外推假象**：偶发持仓的集中篮子年化数值退化（已知坑）；
2. **零成本**：pair 级收益是裸价差；按买 0.15% / 卖 0.25%（含印花税）计入后，
   OOS 单笔均值从 +0.74% 砍到约 +0.35%（毛→净）；
3. **回测人群 ≠ 实盘人群**：回测跑全部信号等权，实盘是 top-N 优先级 + 硬过滤；
   镜像之后 OOS 单笔净均值只剩 +0.04%（见下「priority 反向」）；
4. **数据缺陷**（本轮已全部修复，见第四节）：生存者偏差（无退市股）、qfq 价格接缝、
   创业板 10cm 阈值错杀。

另有一个早期疑点被实证**排除**：「涨停一字开盘买不进导致回测虚高」不成立
（开盘跳空 ≥9% 的入场仅 0.1-3.9% 且收益为负，剔除后整体均值反而略升）。

## 三、信号层实证结论（来自 `surge_signal_analyses.py` + 组合回测交叉验证）

1. **priority_score 排序反向**：按优先级取 top-N 选出的交易（OOS 净 +0.04%）**差于**
   全候选平均（OOS 净 +0.74%）。优先级打分（主升强度35+止损25+新鲜度20+状态质量20）
   不是收益预测器，只能当展示排序用。**下轮迭代不要在它上面叠加逻辑**。
2. **新鲜度不衰减**：延迟 0/1/2/3/5/7/10 天入场，OOS 净均值基本持平
   （confirm：+0.74 → +0.30；anticipate 反而上升：+0.73 → d=7 时 +1.98）。
   「越新鲜越好」假设不成立；埋伏模式延迟 5-7 天（回踩后）更优——这是一条
   可深挖的线索（向上离开中枢后的回踩买点 vs 立即追入）。
3. **门控不敏感**：量比/散度/ret20 阈值 ±20% 网格扫描，IS 恒负（-2.5~-2.7%）、
   OOS 恒小正（+0.5~+0.9%），无刀刃效应。稳健，但说明**门控不是收益来源**，
   继续调门控阈值没有意义。
4. **退出结构**：state（背驰/破坏）退出占 ~75%，trail18 ~13%，sl2 ~6%。
   旧结论「FULL 退出优于仅 SL2 的相对排序」仍成立，风险出清规则保留价值。
5. **收益随成交额单调递减（IS/OOS 一致）**：信号日成交额 <1亿 的单笔最好
   （test +3.7~+5.2%），>10亿 最差（-0.7~-1.2%）→ 所谓 OOS 转正主要是
   2024-2025 微盘股 beta；而实盘流动性过滤（≥1亿）恰好切掉了最好的桶。

## 四、本轮完成的工程修复（已验证，下轮不要重做）

| 修复 | 内容 | 验证 |
|---|---|---|
| qfq 接缝清零 | 693 只增量追加接缝全量重下 + 182 只数据源配股/缩股复权缺陷 pre_close 链式修平 | 全库 5719 只 `close[i]/close[i-1]-1 == pct_chg[i]` 通过；`_sync_daily_data.py` 已接自动修平 |
| 退市股回补 | `sync_a_stock_daily.py --list-status D` 回补 198 只（130 只早年退市无复权因子，failures 记录在 manifest.json） | 生存者偏差缓解（2023-06 前退市的仍因 MIN_BARS=500 缺席，残余偏差方向=高估） |
| 创业板 20cm | `trend_regime.limit_pct_for()`：300/301/302→19.8，其余 9.8 | `--self-check 000636.SZ / 300750.SZ` 因果断言通过 |
| priority 单一真源 | `trend_regime.priority_score()`，daily_scan 与回测共用；止损带满分区间与硬过滤对齐 8-20% | daily_scan import 冒烟通过 |
| TAIL 快路径 | tail=160 vs 全量重放：末根 regime 与 onset 判定一致率 100%（247 有效样本） | `check_tail_consistency.py` |
| ST 历史判定 | `~/.ts_data_cache/namechange.parquet`（2010 起全部改名记录，4993 行）→ 回测中按区间剔除 ST/退 | 组合回测剔除 137/158 笔 |

## 五、基础设施地图（复用入口）

```
数据：~/.ts_data_cache/a_stock_daily_qfq/*.parquet   5719 只，2021-05→2026-06，接缝清零
      ~/.ts_data_cache/namechange.parquet            历史改名（ST 区间判定）

共享中间数据（重跑一次 ~12 分钟，下游全部秒级）：
  scripts/_output/surge_candidates/candidates.parquet   121.7 万行 = 门控前候选 × 7 个延迟，
      含独立模拟的 FULL 退出结果（毛收益）、信号/决策双时点特征、sl_pct、gap、成交额
  scripts/_output/surge_candidates/panel.parquet        dt×symbol open/close/amount 面板

脚本：
  scripts/trend_regime.py              11 态 FSM 核心（limit_pct_for / priority_score / surge_onset）
  scripts/surge_candidates_dump.py     候选抽取（改信号定义后需重跑）
  scripts/surge_portfolio_backtest.py  实盘镜像组合回测 + beta 对照（改组合规则只跑这个）
  scripts/surge_signal_analyses.py     新鲜度/门控敏感性（纯读 candidates）
  scripts/check_tail_consistency.py    TAIL 快路径一致性
  scripts/_repair_qfq_seams.py         数据接缝检测/修复

结果存档：
  scripts/_output/surge_portfolio/summary.json + trades_*.parquet + equity_*.html
```

## 六、给下一轮迭代的方向（按证据支持度排序）

**不要做的**（已被本轮证据否定）：
- ❌ 调门控阈值（不敏感，非收益来源）
- ❌ 在 priority_score 上叠加因子（排序反向）
- ❌ 相信任何「全信号等权 + 零成本」口径的回测数字
- ❌ 用 WeightBacktest 给稀疏篮子算年化

**值得做的**：
1. **市场状态过滤器（最高优先级，唯一有数据支持的方向）**：找出区分
   「2025（超额 t=2.5）vs 2024/2026（负超额）」的可观测状态变量——候选：全市场中位
   ret20、20 日新高家数占比、微盘/大盘相对强弱、指数与其 MA20 关系。
   **现成验证路径**：直接在 candidates.parquet 上按状态变量分桶算超额（秒级，不用重跑
   dump），状态门打开时段的 OOS 超额 t≥2 才算通过。注意：状态变量本身要因果（仅用 ≤t 数据）。
2. **埋伏模式的回踩买点**：anticipate 延迟 5-7 天入场 OOS 净均值 ~2 倍于立即入场
   （+1.98 vs +0.73），且样本充足（n≈4.6k）。可把「离开中枢后 5-7 天 + 仍在上行家族」
   做成显式买点变体，走同样的镜像回测 + 对照判定。
3. **流动性悖论的处置**：收益集中在 <1亿 成交额桶（微盘），实盘过滤又必须 ≥1亿。
   要么接受小资金专做 1-3亿 桶（test +3.78%，次优桶）并验证其超额是否独立于微盘 beta
   （对照已按成交额十分位匹配，1-3亿桶的超额可单独算），要么放弃。
4. **退出规则单独保值**：背驰/破坏出清 + SL2 + 18% 跟踪的退出框架在两段排序一致，
   可作为其他入场策略的退出模块复用。

**方法论约束（对 codex 的硬要求，沿用本轮标准）**：
- 任何新变体先在 candidates.parquet 上做 pair 级分析，再过 `surge_portfolio_backtest.py`
  的实盘镜像 + 随机对照，判定标准预声明（OOS 超额 t≥2）；
- 全程因果：流式重放（iter_states）、次日开盘成交、决策只用 ≤t 数据；
- 成本口径：买 0.15% / 卖 0.25%；
- 改了 `surge_onset` / FSM / 数据 → 必须重跑 `surge_candidates_dump.py`；
- 改名/新增阈值不得按回测指标精调（只允许 a-priori 取值 + 敏感性验证）。

## 七、复现命令

```bash
# 数据体检（应输出 0 接缝）
.venv/bin/python scripts/_repair_qfq_seams.py

# 因果自检
uv run --no-sync python scripts/trend_regime.py --self-check 000636.SZ
uv run --no-sync python scripts/trend_regime.py --self-check 300750.SZ

# 全链路（依次，候选抽取 ~12 分钟，其余秒级）
uv run --no-sync python scripts/surge_candidates_dump.py
uv run --no-sync python scripts/surge_portfolio_backtest.py
uv run --no-sync python scripts/surge_signal_analyses.py
uv run --no-sync python scripts/check_tail_consistency.py 160 300
```
