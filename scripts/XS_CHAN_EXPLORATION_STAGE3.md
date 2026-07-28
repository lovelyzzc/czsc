# 横截面因子 × 缠论 Stage 3：状态 3 前向机制复制

- 机器规格：`scripts/xs_chan_exploration_stage3.json`
- 研究 ID：`xs_chan_exploration_stage3_state3_forward_20260728`
- 模式：`FORWARD_EXPLORATION_ONLY`
- 确认链：`NOT_STARTED`
- 首个可计数决策周：2026-07-31

## 为什么不继续历史回测

Stage 2 已经在 2022–2026 历史样本中证伪状态 7 增量和风险归因后残差，也证明失败画像
不能稳定复制。状态 3 只是在 11 个状态、两个重叠视图的事后扫描中表现突出：

- FC proposal：+1.167%/周，132 个事件、56 个非空周；
- F membership：+1.184%/周，189 个事件、65 个非空周。

这不是独立复制，而且是多重扫描后挑出的结果。Stage 3 不把状态 3 加入允许状态，也不
再改变门、权重或阈值；它只用未来数据回答一个提前冻结的问题。

## 时间边界

Stage 1 最后一个 primary 决策日是 2026-06-05，但其 delay 与 20-session 诊断读取了
截至 2026-07-07 的行情；Stage 2 审计还读取了截至 2026-07-08 的缓存覆盖。因此
2026-06-12、06-18、06-26 三周只能作为 `PRE_GENESIS_RETROSPECTIVE_PIPELINE_FIXTURE`
演练数据，永远不得计入 Stage 3。

协议在 2026-07-28 本地冻结。首个可计数周固定为协议冻结后的
2026-07-31。要保留“前向”表述，规格、源码和 anchor 必须在该决策日 15:00
（Asia/Shanghai）前提交并推送；否则本链为 `INVALID_PROSPECTIVE_FREEZE`。

## 唯一主问题

每周沿用冻结的 FC 基准路径：50/75 buffer，只对新提案应用状态 5–8 门，拒绝后不
继续补位。主 population 是该路径门前的 proposal：

- treated：可交易且 `regime == 3`；
- control：同周、同因子排名十分位、可交易且 `regime != 3`；
- 每个十分位的 control 均值按受支持的 treated 数加权；
- 没有状态 3 的周，主周差固定为 0，不能删除；
- 目标量是连续前 52 周周差的简单平均。

它仍是前向关联，不是缠论状态的因果效应。

## 两阶段账本

每周必须产生两个不同的追加记录：

1. `decision_freeze`：在决策日收盘后、下一交易日开盘前写入。只能包含当时可得特征、
   排名、基准持仓和 proposal，禁止任何未来收益字段。
2. `label_completion`：在入口后完成 5 个持有 session 的下一开盘（含入口为第 6 个
   open）对应退出日收盘后追加，只能引用已经冻结的
   decision hash，不得回写特征或成员身份。

记录使用 canonical JSON、sequence、previous hash、logical event key 和 payload hash
组成的 SHA256 链；精确重试幂等，同一 key 的不同 payload 冲突并保存失败证据。账本
HEAD 由 records 重算，不信任可变指针。

每个 decision/label 记录还必须运行 `export-head-anchor`，在
`scripts/xs_chan_exploration_stage3_ledger_anchors/` 生成唯一的小型 head 承诺并提交、
推送；导出前会用当时磁盘上的 reference/raw 闭包重新生成当前 decision projection 或
label observation，逐字节一致才允许产生 anchor。已有 anchor 永不改写。完整 projection 保存在 content-addressed 本地对象中，
需要同步备份整个 Stage 3 `_output` 目录。Git anchor 能证明之后未无痕改写，但本地
SHA256 本身仍不是第三方时间戳或数字签名。
生产 CLI 不允许注入 `recorded_at`，但本地系统时钟与 Git commit time 仍可由机器
所有者影响；所以本研究只称 `LOCAL_PROSPECTIVE`，不声称具备受托第三方时间认证。

正式操作顺序是固定的：

1. 首周前运行 `anchor`，提交并推送规格、源码和 protocol anchor；
2. 运行 `init`、`export-head-anchor`，再提交并推送 genesis anchor；
3. 每周只在合法窗口运行 `freeze-decision`，随后立刻
   `export-head-anchor`、提交并推送；
4. 退出日收盘后运行 `complete-label`，随后同样导出、提交并推送 head anchor；
5. 下一事件若看不到上一个 head 的已推送 anchor，collector 会拒绝追加。

首个 2026-07-31 decision 由
`scripts/xs_chan_stage3_first_week.py` 作为外部操作保护层执行。它不替换或修改冻结
collector/spec，也不改变 study identity；它只把首周的数据准备、完整八日期
preflight、锁内最终复核以及 decision head anchor 导出做成 fail-closed 编排。
因此首周步骤 3 的 `freeze-decision` 与紧随其后的 `export-head-anchor` 由该操作器一次
完成，但导出的 anchor 仍必须在 entry open 前提交并推送。
首周不得直接调用冻结 collector 自带的 `freeze-decision` CLI；该旧入口仅因研究身份与
重放兼容性而保留，绕过保护层属于操作违规。

## 样本门与盲态

窗口固定为前 52 个连续官方周，不允许延长：

- 至少 26 周出现状态 3；
- 至少 52 个状态 3 事件；
- matched treated coverage 至少 95%；
- 至少 20 个不同股票；
- 单一股票不超过状态 3 事件的 10%；
- 前、后 26 周各至少 13 个状态 3 presence 周。

第 26 周只允许查看完整性、事件数和匹配覆盖。第 52 周以前禁止输出收益、方向、NAV、
胜率、置信区间、bootstrap 或图表；没有提前成功或提前失败。

这里的“盲态”只约束脚本的正式 `status` / `evaluate` 输出，不是密码学或权限隔离：
append-only 标签为便于最终重放而保存明文，操作者也能从公开行情自行算出收益。因此它
只能防止流程意外产生 interim efficacy 报告，不能阻止有动机的人主动偷看。若发生人工
偷看，本研究应标记为操作者非盲，而不能继续声称独立盲态复制。

## 最终一次性规则

唯一主端点使用 HAC lag 4，以及冻结种子的 4 周 circular block bootstrap
20,000 次。经济门为年化 2%，对应周度
`(1.02)^(1/52)-1 = 0.00038089227674453774`。

- `FORWARD_SUPPORTED_EXPLORATORY`：HAC 单侧下界和 bootstrap 5% 分位都严格高于
  SESOI，且前后 26 周点估计都大于 0；
- `FALSIFIED_FOR_REGISTERED_SESOI`：HAC 单侧上界和 bootstrap 95% 分位都严格低于
  SESOI；
- 其他有效结果：`INCONCLUSIVE`；
- 数据或匹配门失败：`INSUFFICIENT_DATA` / `INSUFFICIENT_MATCHING`；
- 链、日历、时间或语义失败：`INVALID_CHAIN`。

市场状态、F membership 和 shadow path 只能在最终解盲后作为
`NON_DECISION_DIAGNOSTIC` 输出，不能改变主状态，也不能在 Stage 3 内选门。

## 结论边界

即使通过，也只能说明冻结 FC proposal population 中，状态 3 的未来五日相对关联值得
另立更晚、互不重叠的组合政策试验。它不能启动或修改 V2.1、不能确认组合 alpha、
不能证明缠论普遍有效，也不授权实盘。
