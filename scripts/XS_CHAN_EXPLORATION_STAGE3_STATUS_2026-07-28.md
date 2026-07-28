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

## 首周操作保护层

`scripts/xs_chan_stage3_first_week.py` 是冻结 collector 外部的 fail-closed
操作保护层，不是新的研究协议，也不参与 study identity 计算。它没有修改
`scripts/xs_chan_exploration_stage3.json` 或
`scripts/xs_chan_exploration_stage3.py`；因此冻结值仍为：

- spec SHA256：
  `3f01620964300440ee1d02ea374e7a42ea16ae3a48379675cc2543a5f64862fa`
- collector SHA256：
  `4f4a97407973246083e9d08c31044f21a1c4ecafde15dd43e81445ef9ba475ea`
- study identity：
  `f0afe61d932fbdf68b5b5ee242b68c7eed75bc932e4f6cf785bd7c9af51fb607`

保护层只编排首个 2026-07-31 decision：

- `status`、`preflight-decision` 和不带 apply flag 的命令不写正式 ledger；
- `prepare-decision --apply-data` 只更新 raw、state、八日期 reference，并生成
  content-addressed preflight receipt；
- `freeze-decision --apply-ledger` 必须同时给出 receipt 和 expected head；
- receipt、冻结前重新生成的 bridge path 与最终 decision payload 的 path hash
  必须完全一致；
- 在 ledger append lock 内重新检查新鲜时间、两小时安全余量、raw closure、
  reference manifest 和 expected head；
- decision 追加成功后，在同一个 ledger lock 内立即导出唯一 head anchor。

Preflight receipt 会绑定操作器当时的源码 SHA256；该 SHA 不冒充或替代冻结研究身份。
冻结 collector 仍保留原始 `freeze-decision` 入口以维持源码身份和重放能力，但首周严禁
直接调用它；直接调用会绕过上述外部保护层，应视为操作违规。

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

另有 17 项 raw 同步/凭证回归覆盖全 inventory 扫描、闭包幂等、重叠日替换、任意复权因子变化
触发全刷、首行引用空值边界、断点对象 hash 绑定、staging 接缝修复、新代码 fail-closed、
parquet 与 auxiliary 联合快照身份、journal/audit 的文件及目录 `fsync`、无 journal
候选拒删、execution binding 参数敏感性、源码漂移回滚，以及 post-hoc 凭证禁止越权声称
原执行源码已证明。

下一次 `--apply` 会要求干净且已推送的 Git HEAD，并在网络抓取前冻结 updater、
下载器、接缝修复器、`uv.lock`、依赖版本和规范化运行参数；交换前与 audit 发布前
再次验证 binding。journal 在目录交换前持久化，audit 只有在文件和目录项都
`fsync` 后才成为 commit marker。

`status` / `evaluate` 必须从磁盘重扫账本并重放对象。每个新 head 导出 Git anchor 前，
还会用当时磁盘上的 raw/reference 闭包重新生成当前 decision 或 label。它仍是本地
完整性控制：本地系统时钟、Git commit time 和数据供应方内容都不是受托第三方签名。
标签为可重放而明文保存，所以“盲态”只约束正式输出，不能阻止操作者主动查看公开行情。

首周操作器的回归还覆盖时间窗口边界、dry-run 账本字节不变、八日期 reference
完整性、raw/state/reference 漂移、expected-head 竞争、append-lock 内最终时钟复核、
preflight/decision path 绑定和 anchor 自动导出；state cache 的 prefix/full audit
失败时禁止发布 cache。

## 首周操作状态与下一步

当前操作状态为 `WAIT_DECISION_DATE`：

- active raw：5,743 个 parquet，最大交易日 `2026-07-27`；
- 正式 ledger：1 条 genesis，head
  `3f206409bb83021141448c61c190288fab57a09afae0ad9e517ccb596ecc9a71`；
- decision / label / 状态 3 事件：全部为 0；
- 已计数前瞻周：`0 / 52`；
- efficacy 输出：未生成；
- 首周 decision / entry / exit：
  `2026-07-31` / `2026-08-03` / `2026-08-10`。

现在只运行只读状态或 dry-run：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py status
uv run --no-sync python scripts/xs_chan_stage3_first_week.py prepare-decision
```

2026-07-31 18:00 Asia/Shanghai 日线发布后，且严格早于
2026-08-03 07:30 Asia/Shanghai，运行数据准备：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  prepare-decision --apply-data
```

保存输出中的 `reference_manifest_path`、`state_manifest_path`、
`preflight_report_path` 和 `expected_head`。最终 reference 必须精确覆盖
06-12、06-18、06-26、07-03、07-10、07-17、07-24、07-31；本页七日期
readiness reference 不能用于正式 decision。

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

验证通过后，才显式追加首个 decision：

```bash
uv run --no-sync python scripts/xs_chan_stage3_first_week.py \
  freeze-decision \
  --apply-ledger \
  --reference-manifest <FINAL_REF> \
  --state-manifest <STATE_MANIFEST> \
  --preflight-report <PREFLIGHT_REPORT> \
  --expected-head <EXPECTED_HEAD>
```

该命令已经自动导出 decision head anchor，不要再单独重复
`export-head-anchor`。应立即只提交并推送该 anchor，并严格早于
2026-08-03 09:30 Asia/Shanghai 完成。

2026-08-10 退出日收盘后，label 仍由冻结 collector 完成：

```bash
uv run --no-sync python scripts/xs_chan_exploration_stage3.py \
  complete-label 2026-07-31 --state-manifest <EXIT_MANIFEST>
uv run --no-sync python scripts/xs_chan_exploration_stage3.py \
  export-head-anchor
```

随后提交并推送新的 label head anchor。任何前置 anchor 未提交并推送，下一事件都会
被拒绝。
