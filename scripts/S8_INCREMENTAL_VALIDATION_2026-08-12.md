# S8 / production Delay5 增量验证补充（2026-08-12）

## 结论

截至 2026-08-12，production Delay5 的原始队列、成交语义、点时精确流通市值匹配、结果盲余额门、
H5/H20/H60 结果推断与五段时间稳定性研究审计已经按顺序完成。精确 PIT 控制后，历史样本与 2024+
样本均未确认稳健选股 alpha；时间切片也没有任何稳健单元。因此研究阶段可立即做出降级/停止投入决策，
不需要等待 60 日真前瞻；S8/Delay5 仍不具备实盘授权。

新的独立前瞻协议已经冻结，但当前行情只到 2026-08-11，后冻结交易日为 0/60，不能提前读取结果。
XS-Chan Stage3 的协议、源码锚点和 genesis ledger 有效，但尚无任何前瞻 decision 或 label，确认链仍未开始。

## 1. 原始队列与执行语义

`surge_candidates_dump.py` 现在保存所有因果可判定的结构候选，不再因未来 entry/FULL 结果缺失而删除
右尾记录。close/state 触发退出拆为 signal 与下一观察开盘 fill；盘中止损仍为当日成交，max-hold 为固定
第 60 日收盘。

重建结果：

- 数据源文件 5,753 个，全部分类；不可读与未知来源均为 0。
- 原始候选 1,255,256 行，其中 FULL 完整 1,247,319 行。
- panel 5,644,485 行。
- production Delay5 raw 101,486，fill 298，固定 H60 mature 286。
- raw 最大 decision 为 2026-08-11；完整 FULL 最大 decision 为 2026-08-06，证明右尾现已保留。
- 候选 SHA256：`0c913fc34478461de52c087450dd1e666ff3e032d5da77f8d754a89db5646b31`。
- panel SHA256：`dd38f70fc02f74c7bf35b38576374de58c8cdde7f8b37f5a5f891eba20154fb0`。
- production cohort SHA256：`895132678775dcfd7ee022d1e172c9ff18183b48f7c99a733c980a878b272224`。

production 审计 schema 升为 v3；raw/fill/maturity 只由 panel、市场交易日历与因果规则重建，明确
`outcome_fields_used_for_cohort=false`。

## 2. Outcome-blind 点时精确市值平衡门

主规格在任何 H5/H20/H60 结果读取前冻结为：同年度冻结行业、精确 `circ_mv` 2 倍 inclusive 卡尺、
完整因果协变量、至少 5 个控制；卡尺内全部控制使用冻结条件熵权重。全期和 2024+ 同时校准以下 9 项：

`log_exact_mcap, ret5, ret20, ret60, vol20, vol60, log_price, log_amount, liq20`。

卡尺 1.5/2.0/3.0 的选择只看协变量、coverage 和 positivity；2.0 是满足全部预设门槛的最小卡尺。
冻结计划 SHA256 为 `45e123ae019603a3690735099db3242540dff9594b3ab94853c562f0328db0e6`。

| 范围 | 源交易 | 支持交易 | 支持率 | ESS p05 | 最大单边权重 | 全局 ESS/交易 | 最大绝对 SMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全期 | 286 | 253 | 88.46% | 5.572 | 0.443 | 16.747 | 1.26e-14 |
| 2024+ | 174 | 154 | 88.51% | 6.209 | 0.348 | 18.823 | 5.65e-15 |

余额 verdict 为 `BALANCE_GATE_PASSED_OUTCOME_EVALUATION_PERMITTED`，且 `outcomes_loaded=false`。
冻结匹配边 16,086 条，SHA256 为
`1e5f10fbff11f5b9864a533d9aaaf1634b2915d845969e73e8e8478c280c0c17`，没有任何未来结果列。

## 3. H5/H20/H60 精确 PIT 结果

正式结果阶段绑定上述计划、余额 verdict、匹配对 SHA 和 production schedule；不允许重匹配。主估计量为
等权处理交易 ATT，控制腿使用逐交易归一的冻结条件熵权重。推断同时使用：

- 完整交易日历 ratio-influence Newey-West，固定 lags 4/19/59；
- 期望块长 5/20/60 的平稳 bootstrap，10,000 次，seed 42；
- H5/H20/H60 单侧 Holm 校正；
- treated-symbol × control-symbol 双向聚类协方差；
- 等权控制均值、控制中位数和毛收益作为非主敏感性。

### 2024+ 主范围（154 笔、96 个 decision dates）

| 期限 | 净 ATT 均值 | HAC t | HAC Holm p | bootstrap 95% CI | bootstrap Holm p | 双向聚类 p |
|---|---:|---:|---:|---:|---:|---:|
| H5 | +0.252% | 0.433 | 0.857 | [-0.907%, +1.453%] | 0.895 | 0.344 |
| H20 | -1.166% | -1.410 | 0.921 | [-2.787%, +0.493%] | 0.924 | 0.891 |
| H60 | +0.841% | 0.566 | 0.857 | [-2.078%, +3.491%] | 0.895 | 0.333 |

### 全期（253 笔、162 个 decision dates）

| 期限 | 净 ATT 均值 | HAC t | HAC Holm p | bootstrap 95% CI | bootstrap Holm p | 双向聚类 p |
|---|---:|---:|---:|---:|---:|---:|
| H5 | -0.0003% | -0.001 | 1.000 | [-0.789%, +0.815%] | 1.000 | 0.500 |
| H20 | -0.089% | -0.146 | 1.000 | [-1.314%, +1.128%] | 1.000 | 0.550 |
| H60 | +0.628% | 0.626 | 0.797 | [-1.226%, +2.479%] | 0.808 | 0.319 |

三期限全部未通过冻结稳健门。控制股票复用明显：3,273 个控制股票中 2,495 个被重复使用，单票最多进入
21 笔处理交易；H60 路径峰值同时活跃 38 笔。双向聚类后仍无显著结果。

正式 verdict：`HISTORICAL_EXACT_PIT_EDGE_NOT_ROBUSTLY_CONFIRMED`，`live_authorized=false`。

## 4. 独立 60 交易日前瞻协议

`surge_delay5_forward_protocol_2026-08-12.json` 在上海 2026-08-12 收盘后冻结。genesis 数据只到
2026-08-11；为避免把冻结当日误算为前瞻样本，decision cutoff 独立顺延到 2026-08-12。规则如下：

- 只接纳 decision 严格晚于 2026-08-12 的交易；
- 首 60 个后冻结官方交易日为固定 accrual window；
- 使用历史冻结 feature scale 与 `all + 2024plus` 熵系数，不在前瞻样本重拟合；
- 每个入组 decision 必须在结果访问前归档 causal decision surface 与完整 `daily_basic` 截面；
- accrual 关闭、最后 decision 的 next-session fill 可观察、全部入组成交达到固定 H60 后，才允许一次性评估；
- 统计和 Holm/双向聚类门与历史正式阶段一致，不延长样本、不挑期限。

当前状态：`WAITING_FOR_FIRST_POST_FREEZE_SESSION`，观察 0/60，入组 raw/fill/mature 均为 0，
`outcome_evaluation_permitted=false`。这一步受真实时间与新数据约束，不能在 2026-08-12 完成结果验证。

为落实主协议中的逐日归档要求，另行冻结了 outcome-blind、append-only 的补充捕获链：

- 捕获脚本 SHA256：`55031619841cb7f33eb186029c2d00583e1faa1de5294f3b43319dceed285596`；
- 补充协议 SHA256：`9d317c6c9696c891ffed6588b99f0d355eaec0c7372a55361a4d267de9fb7161`；
- ledger ID：`1e3c1a5bdf0845b8ed1407b5b9f319999316f0bd61af0e3752ff884baf4b5930`；
- genesis record hash：`71fa3da21f2346134b329e431bb6737574228a93504eb275e0f2d2bf46ca97c8`。

前 60 个 accrual session 每日封存完整 `daily_basic`、因果特征面、production projection、市场状态与
open/close 路径输入；其后持续封存 H60 所需路径和队列状态。对象按内容寻址，ledger 哈希串联；漏日追补
永久标为 `LATE_EXCLUDED`。每个 session 的唯一 head anchor 还必须保持 clean、进入 commit、可从实时远端
分支到达，且 commit 时间早于下一官方交易日 09:30。该远端锚点是操作性证据，不是独立 TSA，协议明确
保留这项强度限制。

当前链只有 genesis，session capture 仍为 0；`combined_outcome_evaluation_permitted=false`、
`outcomes_loaded=false`、`live_authorized=false`。最终解锁不仅要求主 H60 状态 ready，还要求从首个
accrual 日到最后一笔入组成交 H60 终点的每一日对象与远端锚点全部通过。

### 4.1 研究阶段即时结论：五段时间稳定性

为了避免把实盘确认周期错误地当成研究阻塞，新增
`delay5_pit_temporal_stability_audit.py`。该审计严格复用已冻结的 253 笔 trade ATT 和 16,086 条 control
pairs，不重匹配、不重拟合权重；按自然时间边界固定拆为 2021–2022、2023、2024、2025、2026 partial。
这是一项诚实标注的 post-hoc 时间稳定性诊断，不冒充 untouched holdout。

| 时间段 | 交易数 | H5 净 ATT | H20 净 ATT | H60 净 ATT |
|---|---:|---:|---:|---:|
| 2021–2022 | 62 | -0.579% | +1.447% | +0.995% |
| 2023 | 37 | -0.082% | +1.823% | -0.871% |
| 2024 | 37 | -0.643% | -1.879% | -0.621% |
| 2025 | 82 | +0.712% | -1.910% | +1.458% |
| 2026 partial | 35 | +0.121% | +1.330% | +0.940% |

五段 × 三期限构成固定的 15 项单侧检验族；经全局 Holm、对应块长的平稳 bootstrap 和
treated-symbol × control-symbol 双向聚类联合判定，稳健单元为 **0/15**。各期限正收益时间段分别为
H5 2/5、H20 3/5、H60 3/5，符号和幅度明显随时期改变。留一时间段结果中，H5 与 H20 的总体符号均会
翻转；H60 虽保持为正，但原正式推断和各局部单元均不显著。

研究 verdict 为 `RESEARCH_STAGE_EDGE_NOT_TEMPORALLY_STABLE_DEPRIORITIZE`：

- `research_decision_available_now=true`；
- `wait_for_60_day_forward_before_research_decision=false`；
- 60 日真前瞻只保留为未来 live authorization 的必要门，不再阻塞当前研究取舍；
- 当前合理动作是降级/归档该固定假设；若提出新信号或新机制，必须作为新的研究协议重新开始，不能用
  当前前瞻样本反复调参。

## 5. XS-Chan Stage3 可复现性

恢复了清理时删除、但 Stage1 import 和旧单元测试仍需要的只读 V2 研究内核与协议。外部复现审计不修改
已锚定的 Stage3 源码，分别验证工作区、Git 父对象和 ledger：

- Stage3 source SHA256 `4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea` 与锚点一致。
- spec SHA256 `3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa` 与锚点一致。
- protocol anchor 已在 commit `990c863769a48d46055271fd883a83011e8c7050` 推送。
- 冻结 Stage2 spec/source/results 均可从 parent commit
  `836816716bfbe0dccc6766c7478ce79d8a8e781b` 按原 SHA 恢复。
- 三份 Stage1 ignored parquet（ranked surface、buffered memberships、gate proposals）既不在工作区也从未进入 Git；
  因此旧 `status` 原命令继续 fail-closed，不能伪造完整 rehydration。
- append-only ledger 只有 1 条 genesis，decision=0，label=0，52 周剩余 52 周。

Stage3 verdict：`STAGE3_PROTOCOL_LEDGER_VALID_NO_PROSPECTIVE_DECISIONS`；确认链为 `NOT_STARTED`，
禁止 efficacy 输出与实盘授权。

## 6. 复现命令

```bash
uv run --no-sync python scripts/surge_candidates_dump.py
uv run --no-sync python scripts/surge_delay5_production_cohort_audit.py
uv run --no-sync python scripts/delay5_pit_exact_mcap_plan.py
uv run --no-sync python scripts/delay5_pit_exact_mcap_balance_audit.py
uv run --no-sync python scripts/delay5_pit_exact_mcap_outcome_audit.py
uv run --no-sync python scripts/delay5_pit_temporal_stability_audit.py
uv run --no-sync python scripts/surge_delay5_forward_audit.py
uv run --no-sync python scripts/surge_delay5_forward_capture.py append
uv run --no-sync python scripts/surge_delay5_forward_capture.py status
uv run --no-sync python scripts/xs_chan_stage3_repro_audit.py
```

当前研究决策保持不变：**S8/Delay5 关闭，XS-Chan Stage3 未开始确认，全部 live authorization 为 false。**
