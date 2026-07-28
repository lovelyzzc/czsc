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
- reference manifest SHA256：
  `0fc45300c1cfde24d65b7d71dd3c6c576d655f80f621ad7013b3a6c9ca36f6ac`
- rehearsal bundle SHA256：
  `307bf37955ac08d8f97adefdfa76d4c8d3bb08e8ad839f7efaf7d54e7c0dfa82`
- raw source closure SHA256：
  `21f848dd5fe1d986fb1eeec51d75099bfd81170a4be47b5e8f4e7349d5a12fc5`

Reference、projection、raw closure 和 label observation 都保存在研究身份目录下的
content-addressed 本地对象中。rehearsal bundle 还内嵌了 study identity、spec hash 和
collector source hash，不能被其他源码身份冒用。

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

## 完整性边界

18 项单元/负向回归覆盖 hash 链、幂等、冲突留痕、fork/gap、FC 不补位、日历前缀、
六月误计数、future outcome 字段、伪造 label return、任意 ledger identity、对象删除和
terminal 后追加。

`status` / `evaluate` 必须从磁盘重扫账本并重放对象。每个新 head 导出 Git anchor 前，
还会用当时磁盘上的 raw/reference 闭包重新生成当前 decision 或 label。它仍是本地
完整性控制：本地系统时钟、Git commit time 和数据供应方内容都不是受托第三方签名。
标签为可重放而明文保存，所以“盲态”只约束正式输出，不能阻止操作者主动查看公开行情。

## 首周前仍需完成

当前 raw/state cache 只覆盖到 2026-07-08，尚不能生成 2026-07-31 决策。首周需要：

1. 更新并审计 raw qfq 与 state cache，使其覆盖 2026-07-31 收盘；
2. 为 07-03、07-10、07-17、07-24、07-31 以及既有三段桥接周冻结 daily-basic 与
   官方 SSE calendar；
3. 在 07-31 15:00 Asia/Shanghai 后、下一交易日 09:30 前运行
   `freeze-decision`；
4. 立即运行 `export-head-anchor`，提交并推送该唯一 head anchor；
5. 退出日收盘后追加 label，再导出、提交并推送新 head anchor。

任何一个前置 anchor 未提交并推送，下一事件都会被 collector 拒绝。
