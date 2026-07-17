# 横截面因子 × 缠论 Pilot V2 启动就绪度审计（2026-07-17）

## 结论

当前结论为 **NOT READY / FAIL-CLOSED**。

V2 已形成独立的预注册协议、数据契约、因果状态缓存、PIT 特征装配、缓冲选股与精确随机对照、
确认性统计原语和本地哈希链骨架。当前正式判定入口会强制返回
`INVALID_DATA_OR_ENGINEERING`，本地账本即使结构上写满 52 周也不会自行设置
`confirmatory_oos=true`。

这轮工作没有产生新的 alpha 证据，也没有改变 V1 的 `REJECT_OR_REWRITE` 结论。本地历史数据与
全市场状态缓存只能标记为 `CONTAMINATED_STRESS_ONLY`，不得用于宣称 V2 通过。

机器协议：`xs_chan_protocol_v2.json`

- protocol id：`xs_chan_pilot_v2_preregistered_20260717`
- status：`LOCKED_PENDING_CLEAN_DATA`
- SHA256：`0ec5cb260f2aec78a6dce838880140981fdd1a64b5c9a2dcf64be010ba5c8b62`

## 已完成的工程闭环

| 组件 | 当前结果 | 正式含义 |
|---|---|---|
| 预注册协议 | 已冻结 50/75、周频、次 session 开盘、因子/门控/成本/统计口径 | 可审计，不允许历史结果后调参 |
| 数据 bundle | 12 项 artifact、逐文件 hash/schema/时间、公司行为和 L/D/P 对账 | 契约已完成；干净源数据尚未取得 |
| 状态缓存 | 全市场 0–10 投影、内容寻址、100×20 prefix/full 精确审计 | 仅证明当前 qfq 压力数据上的工程因果性 |
| 特征装配 | market-session 时钟、停牌价格 carry、PIT 缺失不填、正式起始周规则 | 正式入口受数据/工程门阻断 |
| 选择与对照 | 50/75 缓冲、受阻卖出占槽、R_match/FGR/FMGR、20 seed | RNG 已绑定协议、arm、seed index、决策周 |
| 统计 | HAC lag 4、4 周 circular bootstrap、Top10、IR/MDD | 初始 NAV、零标准差和 linear 分位口径已冻结 |
| 本地账本 | 排他写、hash chain、决策/执行时窗、精确依赖闭包、首 52 周身份 | 只证明本地结构完整性，不证明 artifact 语义或外部时间 |
| 正式判定 | 纯判定树保留为私有 helper | 语义 replay verifier 缺失时公开入口恒 fail-closed |

## 全市场状态工程审计

本次对 `~/.ts_data_cache/a_stock_daily_qfq/` 的 5,719 个 parquet 完成了全量构建。输入共
6,453,156 行，日期为 2021-01-04 至 2026-07-08；无文件跳过、无校验错误。上市首条记录因没有
前收盘而出现的唯一 `pct_chg=null` 被限定为“仅最早 observed bar 可因果归零”，其他位置仍拒绝。

发布目录：

`~/.ts_data_cache/xs_chan_state_cache_v2/CHAN_STATE_CACHE_473b92f92f548ca638c953506dad8c2906f9f981f268d88db6f3b88c75f4a54b/`

| 产物 | SHA256 |
|---|---|
| `states.parquet` | `665121b375ab80539744639370a653180f90cda423bb74a769003f4fa8d8fb65` |
| `state_audit.parquet` | `02148ba6273fd764f619d26f3718da7ebc418771ed1e27a8938e610414636cbb` |
| `state_audit.json` | `f697361609afec2b3a89131245304c88aea4f396241696e066fe9a96f4704133` |
| `source_manifest.json` | `4153ae578bd6c5c3f2c4d0b54d67101093c59455cd3a52301b062632a5f22796` |

状态投影物理 schema 精确为 `symbol:string, dt:timestamp[ns], regime:int8`，无未来字段。固定 seed
20260717 抽取 100 只，每只 20 个历史截断点，共 2,000 次 prefix/full 比较；`symbol/dt/regime`
逐字段 mismatch 均为 0。

该审计仍不是正式 V2 状态证据：输入是会被后续公司行为重写的 qfq 文件，不是 bundle 内的
`raw OHLC × 当日 adj_factor`；当前状态时钟也尚未用官方停牌证据区分“真实停牌”和“采集缺口”。

## 红队发现与本轮修复

已修复：

1. 最大回撤加入初始 NAV=1，覆盖首周 -50% 的反例；
2. execution 必须在执行日 09:30（含）至 15:00（不含）写入；
3. 退市换股强制非空 `consideration_ts_code`，同一退市键的 cash/share/writeoff 互斥；
4. 随机流绑定 `protocol + arm + seed index + decision date + algorithm version` 并加入 golden vector；
5. `DecisionPlan` 拒绝 `"False" -> True`、浮点 rank 截断等有损身份转换；
6. genesis 依赖闭包精确冻结七项，周度研究/成交 engine hash 必须匹配；
7. 首 52 周的 104 个 record hash、日期、协议、依赖、日历和第 52 周 chain head 形成稳定身份；
8. 数据验证器的 malformed source entry 改为返回失败报告，不再抛未捕获 `KeyError`；
9. board 只接受 `MAIN/CHINEXT/STAR/BSE`；formal start、winsor linear 插值和零标准差失效规则写入协议；
10. 合法 hash 和自报 `true` 不再使本地账本或公开判定入口产生确认性通过。

## 验证结果

V1 冻结回归与全部 V2 单元测试联合运行：`115 passed`。其中 V2 覆盖数据门、特征装配、研究内核、
账本和状态缓存；V1 的 12 项原有回归保持通过。新增 Python 文件同时通过 Ruff、`py_compile`、
JSON 严格解析和 `git diff --check`。全市场缓存 manifest 中的 builder SHA256 与当前
`xs_chan_state_cache.py` 一致。

## 仍为 false 的正式工程门

以下门已在验证器和机器协议中逐项登记，当前全部为 false：

1. `state_recompute_engineering`
2. `feature_prefix_engineering`
3. `source_archive_reconciliation_engineering`
4. `suspension_reconciliation_engineering`
5. `calendar_ledger_binding_engineering`
6. `execution_replay_engineering`
7. `corporate_action_replay_engineering`
8. `statistics_replay_engineering`
9. `artifact_semantic_verifier_engineering`
10. `external_timestamp_anchor_engineering`

因此，不能通过只修改一个 `state_recompute` 开关来启动正式链。

## 正式启动前的最短路径

1. 保存不可变的原始请求、分页响应和分页元数据，由验证器从响应独立重建 L/D/P、逐日行数、
   symbol 集和公司行为覆盖；加入官方停复牌数据，缺行不能默认视为停牌。
2. 用 bundle 内 raw OHLC 与同日 adj factor 重建 valid-bar 状态，并由绑定的冻结 native/engine 做
   100×20 独立前缀重放；不能复用当前 qfq 压力缓存升级为正式证据。
3. 实现 FC/FMA/FGR/FMGR 全 arm、20 seed、gross/1×/2×/容量场景的成交与组合 replay，覆盖
   涨跌停、部分成交、费用、公司行为、退市、pending sell 和现金/持仓不变量。
4. 让账本 hash 指向不可变 artifact store；verifier 读取实际文件、核对 schema/digest，并从订单、
   成交、费用和持仓重新计算 invariant 与前 52 周收益矩阵。
5. 将 bundle 官方日历与 genesis calendar digest 绑定，并把每个 decision 前的 chain head 提交到
   RFC3161、WORM 对象存储或外部透明日志。
6. 以上十个工程门全部自动变绿后，才允许初始化新的未来链；最早连续完整 52 周结束后再运行唯一
   确认性判定，不延长、不换窗口、不回填。

## 可声明与不可声明

可以声明：V2 的预注册规范和 fail-closed 工程骨架已建立；当前压力数据上的全市场状态投影通过
2,000 次因果一致性审计。

不可声明：V2 已完成正式回测、已消除幸存者偏差、已有选股 alpha、缠论择时有效、可以启动 52 周
确认链或可以进入实盘。
