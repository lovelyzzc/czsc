# 横截面因子 × 缠论入场时机 Pilot V2.1（独立预注册协议）

## 1. 协议身份、边界与当前状态

V2.1 是一个新协议，不是 V2 的补丁，也不允许续写 V2 的账本。冻结的 V1、V2 文件、历史结果和链均
保持原样。机器真源是 `xs_chan_protocol_v2_1.json`，协议标识固定为
`xs_chan_pilot_v2_1_preregistered_20260720`，当前状态固定为
`LOCKED_PENDING_ENGINEERING_AND_FORWARD_DATA`。
机器协议按 `UTF-8 + sort_keys + separators=(",", ":") + ensure_ascii=false + allow_nan=false`
规范化后的 SHA256 为 `8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203`。

V2.1 解决 V2 审计发现的八类歧义：组合会计、七比较判定闭环、最小经济效应与真正的证伪、有效干预数、
暴露/换手/蒙特卡洛误差、唯一主链与失败重启、增量官方日历、逐日成交证据和退市归零证据。它没有把尚未
完成的工程门写成通过，也没有启动前向链。

本协议最多支持两个很窄的命题：

1. 冻结的 `120−20` 动量和低波动等权因子，在行业、自由流通市值、流动性与留存结构匹配后，是否存在
   足够大的前向主动收益；
2. 冻结的状态 `{5,6,7,8}` 新仓准入门，是否比相同配额随机门和固定 SMA20 门多出足够大的前向价值。

状态引擎同时使用缠论结构、均线、MACD 和 20 个 session 收益。即使最终通过，也只能解释为“这个冻结的
混合分类器及其 entry-veto 规则有效”，不能外推为“缠论理论普遍有效”。任何历史运行仍只能标记
`CONTAMINATED_STRESS_ONLY` 或 `NON_AUTHORITATIVE_SIMULATION`。

## 2. 不变的经济规格

- 每个完整交易周最后一个官方 session 收盘后决策，下一官方 session 开盘执行。
- 因子目标 50 个槽位，退出缓冲排名 75，预留 0.5% 现金，不加杠杆、不卖空。
- 因子仍为 50% `mom_120_20` 与 50% `lowvol_60`；1%/99% 线性 winsor；分别对 PIT 行业和
  `log(自由流通市值)` 做 OLS 残差化，再做横截面百分位排名。
- 因子物理价格为 `raw close × same-day adj_factor`；执行、成本、整手和 NAV 使用未复权官方 OHLC。
- FC 只对新仓应用 `{5,6,7,8}`；已有目标仓不因状态关闭而强平；不从第 51 名以后回填。
- FMA 只对新仓使用 `adjusted close > SMA20`，其余语义与 FC 相同。
- R_match、FGR 和 FMGR 均用 20 个冻结种子。无法精确匹配时整条确认链无效，不能扩池、换种子或丢周。

V2.1 的随机根种子改为 `20260720`，算法标识为
`xs_chan_rng_v2_1_sha256_seedsequence_v1`。任何代码仍使用 V2 的协议、种子或账本，都不是 V2.1 证据。

## 3. 唯一组合会计

### 3.1 初始状态

每个 arm、每个 gross/1×/2×/容量场景必须独立从空账本开始：参考场景现金恰为 1,000 万元，容量场景
恰为 1 亿元；股票、pending sell、应收现金、递延分红税、其他资产和负债均为空，借款为零，初始 NAV
等于现金。Genesis 必须绑定每本账的初始状态哈希，禁止继承历史仓位或在链外预装仓位。

### 3.2 sizing NAV 和槽位

每个 arm/scenario 在决策 session 的官方收盘完成公司行为、税务记账和估值后，冻结自己的
`sizing_nav`：

```text
target_invested_notional = sizing_nav × 0.995
slot_notional            = target_invested_notional / 50
requested_new_shares     = floor(slot_notional / raw_open / board_lot) × board_lot
```

决策不能使用未来开盘或成交价；`raw_open` 只在执行记录中把冻结的槽位金额变成请求股数。留存仓位不做
周度增持或减持，只允许公司行为改股数；其权重漂移必须报告。这样 50/75 buffer 真正减少无谓交易，且不会
产生“执行器 A 重配留存仓、执行器 B 不重配”的双重解释。

执行优先级唯一固定为：

1. 旧 pending full exit，按最初退出决策日期、股票代码排序；
2. 本周新 full exit，按股票代码排序；
3. 新仓，按 buffer 后因子计划顺序、股票代码排序。

受阻卖出的剩余量进入 pending，并在每个后续官方 session 开盘重试，直至成交或发生官方终止行为；送转后
只按官方比例调整剩余股数。受阻或参与率截断的买单余量在当次开盘执行结束时取消，不跨日重试。现金不足时，
把买量降到“成交价和全部费用计入后仍不负现金”的最大整手，余量取消。零成交也必须显式留痕。

开盘容量同时受两个上限约束：`5% × 决策时已知 ADV20` 与
`10% × 当日官方开盘集合竞价实际成交额`，取二者较小值。集合竞价成交额只能在开盘后用于截断 fill，不能
提前进入决策；缺失或为零时该股票当次开盘成交量为零。全天成交额、预测量和连续竞价首笔均不得冒充集合竞价
容量。正式 bundle 因而必须包含逐股票、逐执行 session 对账的官方 `open_auction` artifact。

最低佣金固定按“股票 × 方向 × execution session”收取 5 元，1× 和 2× 场景都保持 5 元，即 multiplier
固定为 1。2× 只放大佣金费率、过户费、印花税、基础滑点和冲击，不能把最低佣金 floor 也乘二。

### 3.3 现金、分红和股数变化

闲置现金按面值持有，利率固定为零；不允许融资、负现金或外部申赎。

本研究冻结的纳税人身份为境内个人证券账户。现金分红按官方 record date 的 settled shares 确权，在官方
payment date 由应收转为 gross cash。分红税按 FIFO 原始买入 lot 管理：持股不超过一个自然月税率 20%，
超过一个自然月至一个自然年（含）税率 10%，超过一个自然年为 0%；“月/年”严格按自然月/年边界，不得用
30/365 日近似。持有期从取得交割日至转让交割日前一日，卖出/终止时追缴；未处分 lot 每个 EOD 以“若当日
处分”的税率重估负债，不能在确认窗端点把未缴税当成零。税务依据固定为国家税务总局
[财税〔2015〕101 号](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5203902/content.html)及其沿用的
[财税〔2012〕85 号操作规则](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5204415/content.html)。

股数变化必须区分 `stock_dividend / capital_reserve_conversion / split / consolidation`。送股和资本公积
转增是增量 child lot：原 lot 的数量与取得日不变，child 数量为 `parent × (post/pre−1)`，取得日为官方新增
股份登记/到账日；拆股或并股是替换 lot，变后股数为 `parent × post/pre`，继承原取得日。按股票、账户、
交割日的官方 EOD 净增减做 FIFO。总 cost basis 除官方碎股现金外保持不变并按变后每股均摊；碎股必须使用
官方 cash-in-lieu，否则该周期无效。应税送股的递延税 entitlement 仍挂在 record-date parent FIFO lot，
必须有官方每股应税股息基数；资本公积转增/拆并必须有官方零税基 subtype。数据无法区分时禁止猜测。

### 3.4 NAV、周收益和终点

可交易股票按官方未复权收盘估值；停牌股票沿用最后官方未复权收盘，直至恢复交易或官方终止行为。应收现金
按官方金额不贴现，递延税按全额负债计入：

```text
NAV = cash + position market value + receivables - tax and other liabilities
```

一个确认周期从上一个 cycle 的决策收盘 NAV 开始，到当前完整周最后一个 session 的收盘 NAV 结束：

```text
weekly_return = end_nav / start_nav - 1
active_return = arm_weekly_return - comparator_weekly_return
```

因此 52 个收益需要一个先行 anchor decision close，以及随后恰好 52 个连续 cycle close。所有交易成本、
分红、税和终止行为都落在这两个端点之间。第 52 个端点不做假想强平；未卖仓位按上述规则估值，pending sell
仍是股票。确认窗后必须继续按真实日度执行规则 unwind 并报告天数和成本，但该诊断不得改变确认结论。

## 4. 状态的可解释映射

| code | 名称 | 中文 | 新仓放行 | 可解释含义 |
|---:|---|---|:---:|---|
| 0 | NotTradable | 不可交易 | 否 | 无官方有效 bar 或 warm-up |
| 1 | Downtrend | 下跌走势 | 否 | 空头结构 |
| 2 | FirstBuy | 一买观察 | 否 | 早期反转观察，未形成注册确认 |
| 3 | SecondBuy | 二买转强 | 否 | 转强但不是注册准入态 |
| 4 | PivotBuilding | 中枢构造 | 否 | 盘整/中枢构造 |
| 5 | UpwardDeparture | 向上离开中枢 | 是 | 向上离开中枢 |
| 6 | ThirdBuy | 三买确认 | 是 | 三买确认 |
| 7 | MainUptrend | 主升延续 | 是 | 主升趋势延续 |
| 8 | Acceleration | 加速主升 | 是 | 上升趋势加速 |
| 9 | Divergence | 背驰衰竭 | 否 | 趋势衰竭 |
| 10 | Breakdown | 结构破坏 | 否 | 多头结构破坏 |

每个状态必须报告候选新仓数、放行数、成交数、平均入场权重、持有期和前向收益分布；但 V2.1 不允许根据这些
诊断事后挑选状态子集。状态 6 稀少或状态 7 占主导，不会自动证明或否定整体门，只会限制对“三买”的单独表述。

## 5. 七个注册比较与 SESOI

SESOI 是“最小有经济意义的年化主动收益”，不是零。每项比较都必须生成同一 52 周窗口、无重复无缺失的
完整 artifact，并得到 `PASS / FALSIFIED / INCONCLUSIVE` 三分状态。

共同规则如下：

- `PASS`：HAC 单侧 95% 下界和 4 周 circular block bootstrap 5% 分位都严格高于本项 SESOI，周主动
  收益中位数严格大于 0，最好 10 个正主动周贡献不超过 50%；
- `FALSIFIED`：HAC 单侧 95% 上界和 bootstrap 95% 分位都严格低于本项 SESOI；
- 其他全部为 `INCONCLUSIVE`。尤其是“不显著”“样本标准差为零”或两个推断器不一致，均不是证伪。

| 比较 | 角色 | 成本 | SESOI（年化） | 决定总状态 |
|---|---|---:|---:|:---:|
| `F_2x_minus_R_match_2x` | 因子主检验 | 2× | 3.0% | 是 |
| `FC_gross_minus_FGR_gross` | 缠论身份主检验 | gross | 2.0% | 是 |
| `FC_2x_minus_FGR_2x` | 缠论身份成本稳健性 | 2× | 1.5% | 是 |
| `FC_gross_minus_FMA_gross` | 缠论相对 SMA20 特异性 | gross | 1.0% | 是 |
| `FC_2x_minus_FMA_2x` | 特异性的成本稳健性 | 2× | 1.0% | 是 |
| `FMA_gross_minus_FMGR_gross` | SMA20 身份诊断 | gross | 1.0% | 否，必须产出 |
| `FMA_2x_minus_FMGR_2x` | SMA20 成本诊断 | 2× | 1.0% | 否，必须产出 |

最后两项不再是“注册了但没人读取”：它们明确分类普通趋势门是否也有作用，但不单独决定 V2.1 是否通过。
若它们通过而 FC−FMA 不通过，证据支持简单趋势解释而不是 Chan 增量；若它们被证伪，也不能自动验证 FC。
若 net 比较通过但对应 gross 身份比较不通过，只能解释为成本/换手管理，不能称为预测 alpha。

## 6. MDE、功效和可证伪边界

V2.1 固定单侧 α=5%、目标 power=80%、52 周。IID 规划近似中，相对 SESOI 的年化最小可检测距离为：

```text
MDE_excess = (z0.95 + z0.80) × weekly_active_std × sqrt(52)
```

| 周主动收益标准差 | 相对 SESOI 的年化 MDE |
|---:|---:|
| 0.5% | 8.9651% |
| 1.0% | 17.9302% |
| 1.5% | 26.8953% |

例如因子 SESOI 为 3%，周主动波动 0.5% 时，80% power 近似要求真实效应达到约 11.97%。HAC 与块依赖只会
让有效 MDE 更大。这意味着 V2.1 故意不把普通的“不显著”冒充证伪：只有两个上界都低于 SESOI，才能
否定“至少有经济意义”这一窄命题；置信区间跨过 SESOI 时只能停止投入或继续登记新协议，不能在原链加周。

SESOI、MDE、波动网格和 power curve 必须随最终报告公开，不能根据这 52 周重新选择。

## 7. 有效干预、暴露、换手与 MCSE

收益比较之前先验证实验真的发生过。每个注册比较均要求：

- 52 周中至少 26 周，arm 与 comparator 在冻结配额后的新仓身份至少有一次替换；
- 新仓集合对称差的一半累计至少 260 个 symbol assignment；
- 整窗至少 260 个可执行的新仓门机会。

这些计数只看身份，禁止用非零收益定义“有效干预”。前五项决策比较未达到门槛时，总状态为
`INSUFFICIENT_INTERVENTION`，不是 `FALSIFIED`。最后两项不足时其诊断为不可判定，但仍须产出 artifact。

暴露固定为每日官方收盘 `gross long market value / NAV`。20-seed comparator 必须先逐日等权平均；随后计算
arm 与 comparator 暴露的绝对差，不能用可相互抵消的有符号均值代替。前五项比较必须同时满足：均值不超过
1%、P95 不超过 3%、最大值不超过 5%。

周单边换手固定为 `(|buy filled notional| + |sell filled notional|) / (2 × pre-trade NAV)`。F/R_match、
FC/FGR 和 FMA/FMGR 的匹配控制要求周换手绝对差均值不超过 2.5%、P95 不超过 10%。FC/FMA 的换手差本身
可能是干预机制，故强制报告但不作为 control validity gate。

20-seed 控制的逐周 MCSE 为 seed 收益样本标准差（ddof=1）除以 `sqrt(20)`；52 周年化 MCSE 为逐周
MCSE 平方和的平方根，最多 0.5%。20,000 次 bootstrap 固定拆成 20 个 1,000-draw 批次，批次分位数的
`sd(ddof=1)/sqrt(20)` 年化后最多 0.25%。超限返回 `INVALID_CONTROL`；不能看结果后增加 seed 或 draw。

## 8. 公司行为与退市归零

终止类型互斥：`delist_cash` 只能使用官方现金对价；`delist_share` 必须使用官方换股比例且目标证券必须存在
于 security master；`delist_writeoff` 只表示权威来源明确证明现金和股份回收均为零。

每个 writeoff 必须绑定股票、与 master 一致的生效日、权威机构、文档编号、文档 SHA256、source-asof、
retrieved-at 和“零回收”原文片段的 SHA256。活跃股票不得有终止事件；执行器不得因“资料没抓到”“长期停牌”
或“价格为 NaN”自行生成 writeoff。终值缺失或含糊时是 invalid cycle，绝不是零值 fallback。

同样地，所有与 bundle 日历相交的历史 L/D/P 证券都必须有 raw daily 响应或可验证的权威零行/停牌响应；
名称历史必须有完整 effective-dated 记录或权威“从未变更”响应。用当前名称回填历史和漏掉退市股票都会关门。

## 9. 增量官方日历

Genesis 不再假装能预知未来 52 个完整官方交易周。它只绑定官方日历来源和创建时已经正式发布、至少覆盖首个
执行周的 session。之后使用 `calendar_extension` 逐段追加，每段必须在首个新增 session 开盘前完成外部时间
锚，并绑定官方文档、发布时间、抓取时间、有序 session/week 列表、前后 calendar head。

日历段必须连续且不重叠；任何 decision、execution 或 valuation session 在使用前都必须已有锚定覆盖。已覆盖
session 的官方修订或删除会终止主链为 `INVALID_CHAIN`，不能静默改历史，也不能拿预测日历冒充官方日历。

## 10. 逐日证据链

合法记录类型只有：`genesis`、`calendar_extension`、`decision`、`session_open_execution`、
`session_eod_valuation`、`cycle_close`、`chain_abort` 和 `final_evaluation`。

- `decision`：决策日 15:00 后、下一 session 09:30 前；
- `session_open_execution`：当日 09:30（含）后、15:00 前，只绑定开盘时已知数据、公司行为和真实 fills，
  禁止声称此时已有完整日 OHLC；
- `session_eod_valuation`：15:00 后、下一 session 09:30 前，绑定完整官方 OHLC/status、应收/税务、lot、
  cash、positions、pending sells、NAV、暴露和换手；
- `cycle_close`：完整周最后一个 EOD valuation 后、下一 decision 前。

每个覆盖 session 即使没有订单，也必须有 open execution heartbeat 和 EOD valuation。pending sell 每天重试，
每天产生新的 open execution 证据。一个有效周必须包含一个决策、该周全部官方 session 的 open/EOD 记录、
一个 cycle close，并被独立语义重放；本地 hash 连续或 payload 自报 `valid=true` 永远不够。

## 11. 唯一主链和失败重启

`trial_id = sha256(canonical protocol bytes)`。通过全部工程门后，第一个写入不可变外部 trial registry 且获得
可信时间锚的合法 genesis，是该 protocol 唯一 primary chain。同一 protocol 的第二个 primary genesis
一律非法。

Genesis 除协议和依赖闭包外，还必须绑定 data bundle、readiness report、每个工程门证据、每个 arm/scenario
空初始账本、RNG、当前 calendar head、最早合法起点证据和外部锚回执。任何 invalid cycle、covered calendar
修订、语义重放失败或依赖变化都终止主链；不丢周、不补周、不换起点、不延长。

失败后不得用 V2.1 再试。若团队仍要继续，必须新建 V2.2 或更高 protocol id/trial id，并公开绑定 V2.1
失败链 head 和 abort reason。所有尝试都要披露，禁止从多条链中挑一条成功的。

## 12. 正式状态与优先判定

生命周期只允许：

1. `LOCKED_PENDING_ENGINEERING_AND_FORWARD_DATA`
2. `READY_TO_ANCHOR_PRIMARY_GENESIS`
3. `PRIMARY_FORWARD_COLLECTION_ACTIVE`
4. `PRIMARY_CHAIN_TERMINATED_INVALID`
5. `PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY`
6. `EVALUATED`

正式 evaluation 按优先级只能返回：

1. 数据/工程门失败：`INVALID_DATA_OR_ENGINEERING`；
2. 主链不唯一、已终止或重放失败：`INVALID_CHAIN`；
3. 少于 52 个完整收益周期：`FORWARD_COLLECTION_REQUIRED`；
4. 匹配、暴露、换手或 MCSE 失败：`INVALID_CONTROL`；
5. 前五项比较干预不足：`INSUFFICIENT_INTERVENTION`；
6. 因子主比较 `FALSIFIED`：`FALSIFIED_FACTOR`；
7. 因子主比较 `INCONCLUSIVE`：`INCONCLUSIVE_FACTOR`；
8. 因子通过但四项时机门任一 `FALSIFIED`：`FACTOR_PASSED_TIMING_FALSIFIED`；
9. 因子通过、时机门无 falsified 但至少一项 inconclusive：`FACTOR_PASSED_TIMING_INCONCLUSIVE`；
10. 因子和四项时机门全部 `PASS`：`FORWARD_EVIDENCE_PASSED_SHADOW_ONLY`。

最后一个状态只解锁 shadow/capacity 验证，不直接上线。未知状态、`REJECT_TIMING` 等未登记字符串禁止输出。

## 13. 当前仍然关闭的工程门

V2.1 在 genesis 前要求验证器自动推导以下十三门全部为 true：状态重算、特征 prefix、源响应全市场对账、
停牌/采集缺口区分、官方集合竞价对账、增量日历验证、成交重放、组合会计重放、公司行为重放、统计重放、
artifact 语义验证、唯一主链 registry 和外部可信时间锚。人工 `ready=true` 无效。

协议文件、文档或单元测试完成，不等于这些工程门已经完成。正式链只能在实现、恶意反例、golden replay、
外部 registry/anchor 和完整干净 bundle 全部就绪后初始化。

## 14. 禁止的事后选择

禁止改变因子方向/权重、N/缓冲、频率、状态子集、SMA 窗口、seed/draw、SESOI/MDE、有效干预门、暴露/
换手/MCSE 门、比较角色、起止点或统计量；禁止丢掉退市/停牌/无价股票；禁止用下一排名补失败买单；禁止
在看过结果后延长 52 周、增加随机种子、重启同一协议或选择最佳链。任何改变都必须是新 protocol id、
新 trial id 和新的唯一前向主链。
