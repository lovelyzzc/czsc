# XS-Chan Stage 3 状态：2026-07-28

## 当前结论

- 协议状态：`REGISTERED_LOCAL_FORWARD_EXPLORATION`
- 研究身份：`f0afe61d932fbdf68b5b5ee242b68c7eed75bc932e4f6cf785bd7c9af51fb607`
- 首个可计数决策日：`2026-07-31`
- 已计数前瞻周：`0 / 52`
- 确认链：`NOT_STARTED`
- 实盘授权：`false`
- efficacy 输出：未生成

Stage 3 只研究一个事后提出的状态 3 机制假设。当前没有新的前瞻结果，也没有改变
FC 门、V2.1、组合政策或实盘状态。

## 冻结字节

- Stage 3 spec SHA256：
  `3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa`
- collector source SHA256：
  `4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea`
- 初始 reference manifest SHA256：
  `0fc45300c1cfde24d65b7d71dd3c6c576d655f80f621ad7013b3a6c9ca36f6ac`
- 初始 rehearsal bundle SHA256：
  `307bf37955ac08d8f97adefdfa76d4c8d3bb08e8ad839f7efaf7d54e7c0dfa82`
- 初始 raw source closure SHA256：
  `21f848dd5fe1d986fb1eeec51d75099bfd81170a4be47b5e8f4e7349d5a12fc5`

Reference、projection、raw closure 和 label observation 都保存在研究身份目录下的
content-addressed 本地对象中。rehearsal bundle 还内嵌了 study identity、spec hash 和
collector source hash，不能被其他源码身份冒用。

## 通用每周操作保护层

`scripts/xs_chan_stage3_first_week.py` 保留了历史文件名，但现在是覆盖 D1–D52 的
fail-closed 通用 decision 操作器；`scripts/xs_chan_stage3_weekly.py` 是对应的通用
label 操作器。它们位于冻结 collector 外部，不是新的研究协议，也不参与 study identity
计算。它们没有修改 `scripts/xs_chan_exploration_stage3.json` 或
`scripts/xs_chan_exploration_stage3.py`；因此冻结值仍为：

- spec SHA256：
  `3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa`
- collector SHA256：
  `4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea`
- study identity：
  `f0afe61d932fbdf68b5b5ee242b68c7eed75bc932e4f6cf785bd7c9af51fb607`

decision 操作器省略全局 `--decision-date` 时，会从语义账本与官方 SSE 日历自动推导
唯一的下一 decision；提供该参数时只做相等性断言，不能跳周或回选。D1 使用完整八日期
bootstrap reference；D2–D52 各自只追加精确下一决策日，并继承上一 decision 的成员身份
和 `daily_basic` baseline。窗口固定为 52 周；D52 之后只允许完成 outstanding labels
和最终评价，D53 永久禁止。

两个操作器的共同保护包括：

- 两个 `status`、`preflight-decision` 和所有不带 apply flag 的命令都不写正式
  ledger；`status` 只重放本地账本、对象、anchor、sidecar 和 tracking 状态，不调用网络
  或市场 API；
- decision 的 `prepare-decision --apply-data` 只更新 raw、state、reference 并生成
  content-addressed preflight receipt；`freeze-decision --apply-ledger` 必须同时给出
  receipt 和 expected head；
- label 的 `prepare-label --apply-data` 只把 raw/state 推进到最老未标签 decision 的
  精确 exit；`complete-label --apply-ledger` 必须给出 exit state manifest 和 expected
  head，并且只能在 exit 日 18:00（Asia/Shanghai）以后追加；
- decision receipt、锁内重新生成的 bridge path 与最终 payload 的 path hash 必须完全
  一致；label preflight、raw/state closure、observation 与最终 payload 也必须完全一致；
- 正式 raw sync 与八日期 `daily_basic` 都执行预先固定的 4,000 个股票绝对下限、
  相邻 session 95% 保留率和 5% 对称变化门；
- daily 股票必须全部具有同日 `adj_factor`，目标 `daily_basic` 必须完整覆盖 raw
  daily 股票全集；
- 在 ledger append lock 内重新检查固定分支、upstream、远端 URL、实际远端 HEAD、
  新鲜时间、适用时间门、raw/state/reference closure 和 expected head；
- 在 record 出现前先写入精确 anticipated record hash 的 content-addressed
  authorization；事后对已有 record 调用授权存储会被拒绝；
- record 追加成功后，在同一个 ledger lock 内以 complete-or-absent 原子写立即导出
  唯一 head anchor 和 matching authorization sidecar。

Preflight receipt 会绑定操作器当时的源码 SHA256；该 SHA 不冒充或替代冻结研究身份。
冻结 collector 仍保留原始 `freeze-decision`、`complete-label` 和
`export-head-anchor` 入口以维持源码身份和重放能力，但正式链严禁直接调用。没有
append-before-record authorization 的 record 不能事后恢复或补造 sidecar 洗白。

每个事件的授权 sidecar 与 head anchor 必须在一个只包含这两个路径的单父 Git commit
中提交并推送；其父提交必须精确等于 authorization 绑定的
`mine/feat/surge-wave-strategy` HEAD，fetch/push URL 必须都是
`git@github.com:lovelyzzc/czsc.git`。label 的 tracked sidecar 是不含 events、returns
或其他 outcome 的最小承诺；完整 observation 与 authorization 只保存在本地
content-addressed 对象中。历史 sidecar 使用其授权时的 operator blob 验证，不会因
未来 operator 合法升级而失效。

每次 formal D/L 事件都重放从 genesis 到当前 HEAD 的完整历史 pair：所有旧 decision
和 label 都必须有 exact anchor+sidecar pair，上一 pair commit 必须是下一 authorization
Git HEAD 的祖先，最新 pair 必须存在于直接查询的真实远端历史中。label 还强制最老未标签
前缀与日历优先；凡决策日不晚于该 label exit 的 decision 都必须先完成。因此 D2
（2026-08-07）必须先于 L1（exit 2026-08-10），短交易周也可能要求更多 decision
先于一个到期 label。

崩溃恢复会区分未导出、dirty exact pair、已提交未推送和已推送状态。只有当前 HEAD
已预授权 record 的 matching sidecar 与 anchor 可以恢复；已提交未推送的 exact pair
只能原样 push，已推送 pair 的重试只做幂等验证，不会错误要求 active raw 回退到旧闭包。
decision record 必须在 entry open 前合法追加；label record 必须在 exit 日 18:00 后
合法追加。

## 三周管线演练

以下三周均为 `PRE_GENESIS_RETROSPECTIVE_PIPELINE_FIXTURE_NEVER_COUNTS`：

| 决策日 | Entry | Exit | Eligible | 前成员 | Retained | Proposals | Gate pass | 当前成员 | State 3 | Matched |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-12 | 06-15 | 06-23 | 3,682 | 22 | 14 | 36 | 3 | 17 | 0 | 0 |
| 2026-06-18 | 06-22 | 06-29 | 3,633 | 17 | 14 | 36 | 5 | 19 | 2 | 2 |
| 2026-06-26 | 06-29 | 07-06 | 3,619 | 19 | 13 | 37 | 2 | 15 | 5 | 5 |

演练验证了：

- 初始 FC 成员精确来自 Stage 1 的 2026-06-05 快照；
- proposal 是剔除 retained 后的固定 slots，gate 拒绝后不向后补位；
- decision projection 不包含 outcome 字段；
- 109 个 proposals 均有 label event，没有压制或 null label；
- 状态、因子、官方 SSE session、entry/exit 和 raw open 标签能够重放；
- 7 个状态 3 事件都有同周、同因子排名十分位 control。

这些数字只证明 collector 管线接线正确，不能用来评价状态 3 是否有效。

## 7 月 27 日首周就绪快照

2026-07-28 使用官方 SSE calendar 将当日 18:00 前的安全截止日固定为
2026-07-27。同步过程先比较每日 `adj_factor`：

- 464 个复权因子发生变化的既有标的完整重抓 `pro_bar(adj=qfq)`；
- 24 个新代码完整补建上市以来 qfq 历史；
- 其余标的仅合并因子恒定窗口内的 `daily` 行；
- 11 个供应端残余接缝只在 staging 中链式修平；
- 更新前后各自固化为独立 raw snapshot，最后使用
  `renameat2(RENAME_EXCHANGE)` 原子交换整个 active 目录。

本轮 raw 同步证据：

- raw sync audit SHA256：
  `1eebd4c8aa1b5e0275262e695bb690740af72bb35ced849676d72975cefb72ef`
- 更新前 parquet closure：
  `f29a1a88ecba2f7ffdad2f80f1847820cce131d8065731b5d2d082c2904c3435`
- 更新后 active parquet closure：
  `2378679e96e178b77b9e1ef9f33716dde89a6be5d1f43a656bfe7b8fa853652d`
- 把 auxiliary bytes 一并纳入身份后的 v2 更新前 snapshot closure：
  `1991767b231c0aa90ad927ec80087f6f085847051d09196af516ab203c27a61f`
- v2 更新后 snapshot closure：
  `9464dccab24cfa82d383a125b2ef6f1a1f3bedbf1f669dd4c57fd660db0131a0`
- active raw：5,743 个 parquet，最大交易日 2026-07-27。

旧版 raw audit 没有记录原进程实际加载的源码 bytes，因此不追溯性补写或冒充
execution proof。内容寻址的 post-hoc attestation
`abe0cfe8714690c7d2baa38efb1a03ee1458ce63e55bed817c7f033293b0e559`
只证明当前可见 artifact 完整性：audit 自哈希、更新前后 closure、5,537 个 after
文件 hash、5,513 个 before 文件 hash、517 个冻结 API/qfq 对象，以及
464 changed + 24 new + 0 missing = 488 full refresh 的路由重算全部一致；它明确把
原执行源码标为 `UNKNOWN_NOT_ATTESTED`。

从该 active raw 新建了独立的 Stage 3 state cache：

- cache ID：
  `3594b922cfd0580dd795a762ee6b192cbefad112ce6b071418cf552e1e847850`
- source manifest SHA256：
  `162ed1a2b2f53f73c565f18ea11740b562f3952a260d0d8a7fe964ac12842ac7`
- state projection SHA256：
  `b66e74cba9f7834fab41368c57d9c2974ae04214e3adf371c1bf779dfca49b9a`
- 6,558,670 行，最大日期 2026-07-27；
- 100 个标的 × 20 个截点，共 2,000 次 prefix/full 精确比较，mismatch 为 0，
  `AUDIT PASS`。

截至 07-24 的七日期 bridge readiness reference 为
`c2eece7dc73f0a368ca47e900ee79faf4bc396f103d7bee89a0264315ec60d3d`，
其 raw source closure 为
`f30d234c560981aaeb23d94daebe20fb54f0106548e970ca31f19a1da8e7045c`。
用该 reference 重跑 rehearsal，bundle 为
`bc0c7cacfe27766d4421445632c69d1c1b3fdea91d26d10896342655637dc4b4`：
仍为 109 proposals、7 个状态 3、7 个 matched，且
`prospective_week_count == 0`。七周 bridge path dry-run 共有 262 个 proposals，
不含 outcome 字段，也不写正式 ledger。

## 完整性边界

18 项单元/负向回归覆盖 hash 链、幂等、冲突留痕、fork/gap、FC 不补位、日历前缀、
六月误计数、future outcome 字段、伪造 label return、任意 ledger identity、对象删除和
terminal 后追加。

另有 70 项 raw 同步/凭证回归覆盖全 inventory 扫描、闭包幂等、重叠日替换、任意复权因子变化
触发全刷、首行引用空值边界、断点对象 hash 绑定、staging 接缝修复、新代码 fail-closed、
parquet 与 auxiliary 联合快照身份、journal/audit 的文件及目录 `fsync`、无 journal
候选拒删、execution binding 参数敏感性、源码漂移回滚，以及 post-hoc 凭证禁止越权声称
原执行源码已证明；还覆盖严重缩量、异常扩张、缺失 factor、正常小幅增删与 completeness
binding 篡改、固定分支/upstream/URL/真实远端 SHA、snapshot 路径隔离、symlink 拒绝、
文件级持久化、目录交换双侧闭包竞态，以及伪造 audit commit marker 拒绝。

下一次 `--apply` 会要求干净且已推送的固定研究分支，并绑定 upstream、fetch/push URL
与直接 `ls-remote` 得到的真实远端 SHA；网络抓取前还会冻结 updater、下载器、接缝
修复器、`uv.lock`、依赖版本和规范化运行参数。目录交换前会从 old v2 snapshot 的实际
目标日 parquet 行重建 baseline，并从冻结的 calendar/daily/adj-factor CSV 重建完整
session 链和 completeness report；重算结果必须同时等于 binding 与 audit 顶层报告。
snapshot root 必须在 active raw 外，正式 payload 必须是 no-follow 普通文件并完成
file/directory `fsync`。交换前后、audit commit 前和旧目录清理前均重核 active/candidate
双侧闭包；journal 在交换前持久化，audit 才是 commit marker。

两个操作器的 `status` 必须从磁盘重扫账本并重放对象、pair 与完整 D/L Git 因果链，
但明确不查询网络；正式 append/recovery 才直接核对真实远端。每个新 head 导出 Git
anchor 前，还会用当时磁盘上的 raw/reference closure 重新生成当前 decision 或 label。
它仍是本地完整性控制：本地系统时钟、Git commit time 和数据供应方内容都不是受托
第三方签名。标签为可重放而明文保存，所以“盲态”只约束正式输出，不能阻止操作者主动
查看公开行情。

通用 decision/label 回归覆盖自动 next decision、D53 拒绝、D2-before-L1 与短周日历
优先、完整历史 D/L pair 因果链、dry-run 账本字节不变、精确 exit 18:00 gate、
raw/state/reference 漂移、expected-head 竞争、append-lock 内最终复核、
append-before-record 授权、直接 collector CLI 绕过拒绝、双文件 Git parent 绑定、
崩溃恢复、已提交未推送与已推送幂等路径、最小无 outcome label sidecar 和原子证据写；
state cache 的 prefix/full audit 失败时禁止发布 cache。

## 当前操作状态与下一步

当前操作状态为 `WAIT_DECISION_DATE`：

- active raw：5,743 个 parquet，最大交易日 `2026-07-27`；
- 正式 ledger：1 条 genesis，head
  `3f206409bb83021141448c61c190288fab57a09afae0ad9e517ccb596ecc9a71`；
- decision / label / 状态 3 事件：全部为 0；
- 已计数前瞻周：`0 / 52`；
- efficacy 输出：未生成；
- 首周 decision / entry / exit：
  `2026-07-31` / `2026-08-03` / `2026-08-10`。

正式 ledger 仍只有 genesis；当前没有任何可提交的 decision 或 label。现在只运行只读
状态或 dry-run：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py status
uv run --no-sync python scripts/xs_chan_stage3_first_week.py prepare-decision
uv run --no-sync python scripts/xs_chan_stage3_weekly.py status
```

`status` 不调用网络或市场 API，也不写正式 ledger。

## 通用 decision 命令

`--decision-date` 与 `--data-dir` 是全局参数，若使用必须放在子命令之前。通常应省略
`--decision-date`，让操作器自动推导唯一 next decision；显式值只作相等性断言，例如：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  --decision-date 2026-07-31 status
```

2026-07-31 18:00 Asia/Shanghai 日线发布后，且严格早于
2026-08-03 07:30 Asia/Shanghai，运行数据准备：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  prepare-decision --apply-data
```

保存输出中的 `reference_manifest_path`、`state_manifest_path`、
`preflight_report_path` 和 `expected_head`。D1 的最终 reference 必须精确覆盖
06-12、06-18、06-26、07-03、07-10、07-17、07-24、07-31；本页七日期
readiness reference 不能用于正式 decision。D2–D52 使用同一组命令，但操作器只构造
自动推导出的单个新决策日 reference，并继承上一 decision 的成员身份与
`daily_basic` baseline。

先做不写账本的完整验证：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  preflight-decision \
  --reference-manifest <FINAL_REF> \
  --state-manifest <STATE_MANIFEST>

uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  freeze-decision \
  --reference-manifest <FINAL_REF> \
  --state-manifest <STATE_MANIFEST>
```

验证通过后，才显式追加该 decision：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  freeze-decision \
  --apply-ledger \
  --reference-manifest <FINAL_REF> \
  --state-manifest <STATE_MANIFEST> \
  --preflight-report <PREFLIGHT_REPORT> \
  --expected-head <EXPECTED_HEAD>
```

该命令会先持久化 append authorization，再追加 record，并自动导出 decision head
anchor；不要再单独重复 `export-head-anchor`。应立即在同一个单父 commit 中只提交输出
指定的 decision anchor 和 matching authorization sidecar，然后推送；这个 commit 的
父提交必须仍是命令绑定的远端 HEAD。D1 必须严格早于
2026-08-03 09:30 Asia/Shanghai 完成。D52 以后操作器返回 decision window complete，
不允许 D53。

## 日历优先与通用 label 命令

账本事件按官方日历优先级推进。第二个 prospective decision 是
`2026-08-07`，它必须在 `2026-08-10 09:30 Asia/Shanghai` 前冻结、导出 anchor、
提交并推送；之后才能在 `2026-08-10 18:00 Asia/Shanghai` 后完成 L1。操作器会拒绝
先写 L1；同理，完成任一 label 前，所有 decision date 不晚于该 label exit 的 decisions
都必须已经形成已推送 pair。短交易周可能要求多个 later decisions 先完成。

`scripts/xs_chan_stage3_weekly.py` 自动选择最老的未标签 decision；全局
`--decision-date` 仍只作相等性断言。`--decision-date`、`--data-dir` 和
`--state-manifest` 都必须放在子命令之前。先用无网络、无写入的状态命令确定下一动作：

```bash
uv run --no-sync python scripts/xs_chan_stage3_weekly.py status
uv run --no-sync python scripts/xs_chan_stage3_weekly.py \
  --decision-date 2026-07-31 \
  --state-manifest <EXIT_MANIFEST> \
  status
```

退出日 18:00 后才可把 raw/state 准备到精确 exit；active raw 早于或晚于该 exit 都不能
正式追加：

```bash
uv run --no-sync python scripts/xs_chan_stage3_weekly.py \
  --decision-date 2026-07-31 \
  prepare-label --apply-data
```

保存输出中的 exit `state_manifest_path` 与当前 `expected_head`，再完成一个且仅一个最老
label：

```bash
uv run --no-sync python scripts/xs_chan_stage3_weekly.py \
  --decision-date 2026-07-31 \
  --state-manifest <EXIT_MANIFEST> \
  complete-label --apply-ledger --expected-head <EXPECTED_HEAD>
```

该命令先写精确 anticipated label 的 append authorization，再追加 record，并自动导出
最小无 outcome sidecar 与 matching head anchor。立即把这两个路径作为单父 commit 的
唯一变更提交并推送；任何旧 D/L pair 缺失、未推送或不在完整 Git 因果链中，下一事件
都会被拒绝。

若 append 已获授权但在证据导出、commit 或 push 附近崩溃，使用当前 label record hash
恢复；已存在的 exact commit 不得重写：

```bash
uv run --no-sync python scripts/xs_chan_stage3_weekly.py \
  --decision-date 2026-07-31 \
  recover-label --apply-ledger --record-hash <LABEL_RECORD_HASH>
```

恢复返回 `LABEL_EVIDENCE_COMMIT_AWAITING_PUSH` 时只 push 当前 exact commit；返回已推送
状态时不再要求旧 raw/state 成为 active。

## 盲态输出边界

所有 decision/label preflight 与正式操作凭证都固定
`efficacy_output: FORBIDDEN`，tracked label sidecar 不含 events、returns、收益方向、
NAV、胜率、置信区间、bootstrap 或图表。D52 以前不得生成 interim efficacy；D52 后也
必须先完成全部 outstanding labels，才能运行冻结的一次性最终评价。append-only label
records 为最终重放保留明文，因此这是操作输出约束，不是密码学盲化。
