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

每个 decision/label 记录都由外部操作器在同一个 ledger lock 内自动导出唯一的 head
anchor；不得再单独运行 collector 的 `export-head-anchor` 拼接正式证据。导出前会用
当时磁盘上的 reference/raw 闭包重新生成当前 decision projection 或 label observation，
逐字节一致才允许产生 anchor。已有 anchor 永不改写。完整 projection 保存在
content-addressed 本地对象中，需要同步备份整个 Stage 3 `_output` 目录。Git anchor
能证明之后未无痕改写，但本地 SHA256 本身仍不是第三方时间戳或数字签名。
生产 CLI 不允许注入 `recorded_at`，但本地系统时钟与 Git commit time 仍可由机器
所有者影响；所以本研究只称 `LOCAL_PROSPECTIVE`，不声称具备受托第三方时间认证。

正式操作顺序是固定的：

1. 首周前运行 `anchor`，提交并推送规格、源码和 protocol anchor；
2. 运行 `init`、`export-head-anchor`，再提交并推送 genesis anchor；
3. 每个决策窗口使用 `scripts/xs_chan_stage3_first_week.py` 准备并冻结唯一的下一
   decision；操作器在追加前写 authorization，并在追加后自动导出 matching sidecar
   与 anchor；
4. 退出日 18:00（Asia/Shanghai）以后使用 `scripts/xs_chan_stage3_weekly.py` 完成
   最老的未标签 decision；它同样执行 authorization-before-record，并自动导出 matching
   sidecar 与 anchor；
5. 每个事件的 sidecar 与 anchor 必须是一个单父 Git commit 中仅有的两个变更，立即
   推送后才能开始下一正式事件。

`xs_chan_stage3_first_week.py` 是历史兼容文件名，现在是覆盖 D1–D52 的通用 decision
操作器。省略全局 `--decision-date` 时，它从语义账本和官方 SSE 日历自动推导唯一的下一
decision；提供该参数时只作为相等性断言，不能选周。D1 使用冻结的完整八日期 bootstrap
reference；D2–D52 每次只追加精确的下一决策日，并继承上一 decision 的成员身份与
`daily_basic` baseline。窗口恰为 52 个 decision；D52 之后只能补齐 outstanding labels
并最终评价，D53 永久禁止。

日历优先级高于标签到期顺序：完成任一 label 前，所有决策日不晚于该 label 退出日的
decision 都必须已经授权、锚定、提交并推送。因此 D2（2026-08-07）必须先于 L1
（退出日 2026-08-10）出现；遇到短交易周时，可能有更多 later decisions 先于同一
label。label 操作器始终只允许完成最老的未标签前缀，且 raw、state 与 observation
必须精确止于退出日；退出日 18:00 前不允许生成或追加 label。

两个操作器不替换或修改冻结 collector/spec，也不参与 study identity 计算。冻结值
保持为 spec
`3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa`、
collector
`4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea`、
study identity
`f0afe61d932fbdf68b5b5ee242b68c7eed75bc932e4f6cf785bd7c9af51fb607`。
冻结 collector 保留原始 `freeze-decision`、`complete-label` 和
`export-head-anchor` 入口只为研究身份与重放兼容；正式链严禁直接调用。绕过操作器产生
的 record 没有 append-before-record authorization，不能事后补造 sidecar 洗白。

通用 decision 操作器还固定以下与结果无关的完整性门：

- 每个新增 raw session 的 daily 股票全集至少 4,000 个；
- 相对上一个 active session 至少保留 95%，对称集合变化不超过 5%；
- daily 中每个股票必须有同日 `adj_factor`；factor-only 股票允许存在，但必须计数并
  哈希；
- 八日期 `daily_basic` bridge 使用相同的 4,000 / 95% / 5% 门，正式目标日还必须
  完整覆盖 raw daily 股票全集，额外股票不超过 raw 全集的 5%。

这些证据被纳入 raw execution binding、audit、每周 preflight 和最终 authorization。
正式发布在目录交换前还会从旧 raw snapshot 的实际 parquet 行以及内容寻址的
calendar/daily/adj-factor CSV 独立重算整份报告，并要求与 binding 和 audit 顶层逐字节
等价。正式 binding 固定当前研究分支、upstream、fetch/push URL 和真实远端 ref，并在
交换紧前直接查询远端 SHA；active/candidate 在交换前后、audit commit 前和清理旧目录前
都必须保持绑定的双侧闭包。snapshot root 不得位于 active raw 内，candidate、snapshot
及 API/full-qfq 证据都拒绝 symlink/非普通文件，并在正式可见前同步文件和目录项。门是在
恢复时只有固定 audit 根下、哈希文件名与 canonical JSON 相符且精确绑定
data_dir/before/after/binding 的文件才是 commit marker。门是在 2026-07-31 数据获取前
冻结的；旧 audit 不会被追溯性补写成已证明。

label 的 tracked sidecar 是不含 events、returns 或其他 outcome 的最小承诺，只保存
record/authorization/operator/Git 绑定；完整 observation 与 authorization 留在本地
content-addressed 对象中。每次 formal append 前都会重放从 genesis 到当前 HEAD 的全部
D/L pair，并证明上一 pair commit 是下一 authorization Git HEAD 的祖先；最新 pair 还
必须位于直接查询到的真实远端历史中，不能只信任 tracking ref。

若 record 已获预授权但进程在 sidecar、anchor、双文件 commit 或 push 前后崩溃，恢复
会区分未导出、dirty exact pair、已提交未推送和已推送四种状态。只有当前 head 的精确
authorization 及预期两个证据路径可恢复；已提交未推送的 exact pair 必须原样 push，
已推送 pair 的重试只做幂等验证，不要求旧 raw 闭包仍是当前 active 数据。decision 的
合法追加仍须满足 entry-open 截止；label 的合法追加仍须满足退出日 18:00 gate。所有
collector 证据写入都经过 complete-or-absent 的同文件系统原子发布。两个 `status`
命令只做本地重放、不调用网络或市场 API；正式 mutation 与恢复才直接核对真实远端。

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
