# 缠论走势类型划分（11 态）+ 买卖点回测 + 主升浪特征研究

> **2026-06-10 审计完结**：完整结论、证据与下一轮迭代交接见
> [`SURGE_REGIME_AUDIT_2026-06-10.md`](SURGE_REGIME_AUDIT_2026-06-10.md)。
> 一句话判定：实盘镜像 + 随机对照下当前形态**无选股 alpha**。

把个股日线走势划分为 0..10 共 11 个走势类型，标注各状态的买卖点，做样本外（OOS）
回测对比，并研究主升浪个股的特征与阶段、给出止损止盈策略。**全程不使用未来函数，
不按指标网格搜最优（防过拟合）。**

## 三个脚本

| 脚本 | 作用 | 运行 |
|---|---|---|
| `trend_regime.py` | 因果安全的 11 态缠论状态机（核心模块） | `uv run --no-sync python scripts/trend_regime.py --self-check 000636.SZ` |
| `trend_regime_backtest.py` | 右侧买点 + 止损止盈矩阵 + OOS 回测 | `uv run --no-sync python scripts/trend_regime_backtest.py` |
| `surge_characteristics.py` | 主升浪特征/阶段研究报告 | `uv run --no-sync python scripts/surge_characteristics.py` |
| `surge_regime_backtest.py` | 主升浪启动 OOS 回测（全信号等权，**结论已被下行替代**） | `uv run --no-sync python scripts/surge_regime_backtest.py` |
| `surge_candidates_dump.py` | 门控前候选 × 多延迟模拟一次性抽取（下游共享数据） | `uv run --no-sync python scripts/surge_candidates_dump.py` |
| `surge_portfolio_backtest.py` | **实盘镜像组合回测**（top-N+硬过滤+成本）+ 随机对照 beta 剥离 | `uv run --no-sync python scripts/surge_portfolio_backtest.py` |
| `surge_signal_analyses.py` | 新鲜度衰减 + 门控敏感性分析 | `uv run --no-sync python scripts/surge_signal_analyses.py` |
| `check_tail_consistency.py` | daily_scan TAIL 快路径 vs 全量重放一致性 | `uv run --no-sync python scripts/check_tail_consistency.py 160 300` |
| `_repair_qfq_seams.py` | qfq 缓存接缝检测/重下/链式修平 | `python scripts/_repair_qfq_seams.py --fix` |

输出落 `scripts/_output/{trend_regime, surge_regime, surge_candidates, surge_portfolio}/`。

## 主升浪启动信号 + 每日选股

`trend_regime.surge_onset(prev, regime, feats, prior, mode)` 是共享的因果「主升浪启动」信号
（`confirm` 确认进 7/8 / `anticipate` 刚离开中枢 5，均带放量+均线发散门控），被回测与选股 skill 共用。
每日选股见 `.cursor/skills/surge-regime-stock-picker/`（`iter_states(tail=160)` 快路径，全市场扫描 ~2 分钟；
快路径与全量重放一致率 100%）。扫描输出拆为两层：`picks_*.parquet` 保留完整结构池与前向样本，
`review_*.parquet` 只保留今日新增且通过硬过滤的最多 5 只人工复盘样本；后者优先覆盖不同行业、
启动方式和当前状态，不改变回测或组合口径。优先级排序走 `trend_regime.priority_score`
（选股与回测单一真源，但只作展示排序，不视为收益预测器）。

**2026-06-10 实盘镜像重测结论（替代旧 surge_regime_backtest 结论）**：top-N 优先级选股 +
硬过滤 + 成本 + 退市股回补后，两种买点 OOS 超额（vs 同日同成交额十分位随机对照）均不显著
→ **收益主体为规模/市场 beta，未证明选股 alpha**；组合净年化为负。仅 2025 年有显著正超额，
2024/2026 为负（市场状态依赖）。`priority_score` 选出的交易差于全候选平均，仅作展示排序。
详见 `.cursor/skills/surge-regime-stock-picker/SKILL.md` 的回测表现一节与
`scripts/_output/surge_portfolio/summary.json`。

## 11 个走势类型

`0 不可交易 · 1 下跌 · 2 一买观察 · 3 二买转强 · 4 中枢构造 · 5 向上离开中枢 ·
6 三买确认 · 7 主升延续 · 8 加速主升 · 9 背驰衰竭 · 10 结构破坏`

判据全部基于缠论原语（`bi_list` 笔、原生 `ZS` 中枢、`power/angle` 力度、MACD-DIF 面积
背驰、MA 排列），按 `prev_regime` 上下文做有限状态转移（FSM），保证状态有序演化、互斥。

- **买点**（右侧确认为主）：进入 `5 向上离开中枢` / `6 三买确认`。
- **卖点**（统一）：进入 `9 背驰衰竭` / `10 结构破坏`，叠加笔结构止损 SL2。

## 不使用未来函数的三道保险

1. **流式重放**：逐 bar `CZSC.update()`，每个 bar 只用截至当前的笔结构分类。已在
   `--self-check` 中**断言**「流式 `bi_list` == 全量 `CZSC(bars[:t+1])` 逐字节一致」，
   消除「最后一笔用未来 bar 确认」这一常见泄漏（旧脚本用 `bi.edt<=dt` 过滤即有此泄漏）。
2. **次日开盘成交**：信号收盘确认，次日开盘价成交；止损按当日触及价、跳空按开盘。
3. **涨停跳过**：按板阈值（主板 9.8% / 创业板 19.8%，`limit_pct_for`）当日置为
   不可交易，无法成交即不入场。

## 数据层（2026-06-10 修复）

- **qfq 接缝清零**：增量同步曾把未复权行追加到前复权历史（693 只），且数据源对
  配股/缩股类除权的复权因子有缺陷（182 只）——`_repair_qfq_seams.py --fix` 重下 +
  pre_close 链式修平，全库 `close[i]/close[i-1]-1 == pct_chg[i]` 校验通过；
  `_sync_daily_data.py` 同步后自动修平新接缝。
- **退市股回补**：`sync_a_stock_daily.py --list-status D` 回补 198 只退市股
  （130 只早年退市无复权因子失败，记录于 manifest），缓解生存者偏差；
  全库 5719 只。

## 关键结论

### 回测（全 A 5522 只，train≤2023 / test≥2024）

退出方式的**排序在样本内/外一致**（这是比绝对数值更可信的稳健性证据）：

| 组合（买点=离开+三买 W56） | IS 夏普 | IS 卡玛 | OOS 夏普 | OOS 卡玛 | OOS 回撤 |
|---|---|---|---|---|---|
| STATE（纯状态退出） | 0.77 | 0.45 | 2.29 | 1.89 | 31.7% |
| **STATE_SL2（状态+结构止损）** | **1.28** | **0.89** | **3.25** | **3.45** | 24.8% |
| SL2（仅结构止损） | 0.77 | 0.51 | 2.49 | 3.79 | 18.0% |
| **ATR（波动止损）** | **1.29** | **1.00** | 3.10 | 3.43 | 25.0% |
| TRAIL（SL2+跟踪） | 1.04 | 0.81 | 2.82 | 3.61 | 21.3% |

- **结构/波动止损（SL2 / ATR）让利润奔跑，风险调整收益优于「一见风吹草动就走」的快速状态止损。**
- `三买(6)` 太稀疏（全市场约 0.3% bar）单用样本不足，故主买点用更宽的 `{5,6}`（用户选定）。
  `T6_仅三买` 的年化 200%~600%、卡玛被截到 20 是**稀疏组合年化外推的统计假象**，
  其诚实读数是「单笔平均收益 / 胜率 / 盈亏比」，不是组合年化。

> **诚实的注意事项**：OOS(2024+) 明显强于 IS，部分原因是测试窗口正好覆盖 2024Q4 的
> A 股大涨——这里有**市场 beta**，不能全记为策略 alpha。可信的是「跨期一致的排序」，
> 不是某个绝对年化数。
>
> **2026-06-10 更新**：上表数字产生于数据修复前（qfq 接缝、无退市股、创业板 10cm 错杀）
> 且为全信号等权 + 零成本口径，**绝对数值不可用于交易决策**；退出方式的相对排序结论
> （结构/波动止损 > 快速状态止损）仍被实盘镜像重测支持。主升浪买点的最终判定见上节：
> 无显著选股 alpha。

### 主升浪特征（909 次主升浪 vs 130 万对照样本）

- **阶段路径**：95% 主升浪启动前走过 `5 向上离开中枢`，87% 走过 `4 中枢构造`；
  典型路径 **中枢构造 → 向上离开 → 主升**。启动当根 74% 紧邻 `向上离开中枢`。
- **判别特征**（主升组 vs 对照组中位）：MA5-MA20 散度 5.1% vs 2.2%、20 日涨幅 11.9% vs 6.5%、
  量比 1.31 vs 0.90、中枢更多更宽。即「**放量、均线发散加速、立于中枢上方**」。
- **止损止盈**：主升浪区间内最大回撤中位 13.4%、P75 18.3% → 跟踪止损宜 ≈18%（过紧会被洗下车）；
  58% 主升浪以 `背驰/结构破坏` 收尾，**首次背驰/破坏离场可锁定约 63% 的峰值涨幅**。

**综合止损止盈**：主升中段（7）用 SL2 / 中枢下沿托底、让利润奔跑；加速段（8）切换 ≈18% 跟踪止损；
`背驰(9)` 减仓、`结构破坏(10)` 清仓。

## 可复用入口

`from trend_regime import iter_states, classify_fsm, Regime, load_stock, extract_zs_list`
—— `iter_states(df, with_features=True)` 返回每根 bar 的因果状态快照，可直接接每日选股
（输出当日各状态标的清单）或实盘信号。
