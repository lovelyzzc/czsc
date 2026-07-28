# 横截面因子 × 缠论 Stage 2：历史证伪压力测试

- 机器规格：`scripts/xs_chan_exploration_stage2.json`
- 研究 ID：`xs_chan_exploration_stage2_hypothesis_stress_20260727`
- 模式：`EXPLORATORY_ONLY`
确认链：`NOT_STARTED`

## 为什么需要 Stage 2

Stage 1 的完成后语义审计发现：

- 行业规模随机对照没有保持因子候选支持；
- H1 混用了 F 当前成员、FC 新入选提案和全 ranked universe；
- H3 的两个字段由全样本选择，不能再把 2024 年以后称为时间外验证；
- 条件提案随机化没有递归重建后续持仓路径；
- 固定 50 槽、空槽现金与成本没有成为主路径口径。

Stage 2 只用于修正这些辨识问题、淘汰机制和产生下一轮假设。Stage 1 已经看过
2022–2026 全样本，且协议冻结前的独立审计已经计算过部分 Stage 2 指标，所以本阶段
不是未见样本预注册，也不是确认。

## 固定时间切分

- Development：2022-01-14 至 2023-12-29，共 99 周。
- Historical validation：2024-01-05 至 2026-06-05，共 125 周。
- V1：2024-01-05 至 2025-03-14，共 62 周。
- V2：2025-03-21 至 2026-06-05，共 63 周。

这些边界由 Stage 1 的 224 个已分析周冻结，不能按结果移动。缓存中只有 3 个 Stage 1
未使用且 5-session 标签完整的周，少于最低 26 周，因此不能构成新的独立样本。

## 研究内容

1. H1 在 FC proposal 的共同决策周上计算状态 7 减 SMA20 的配对 cohort 周差；另把
   F buffered membership 作为独立敏感性 estimand，禁止拼接。这里的共同周不是
   名称级共同支持，也不具有因果含义。
2. H1 对照只在同一 FC proposal population 内匹配周、申万一级行业、规模五分位、
   MA20 状态和因子排名十分位；treated 与 control 必须来自相同精确 strata 并保持
   相同匹配数，匹配不足必须报告为不足。
3. H2 逐周复刻 Stage 1 横截面 OLS，并把原 `market` 标签改称
   `cross_sectional_common_return`。六分量必须逐周精确重构。
4. H3 原两个字段永久标记为 post-selection diagnostic；另用 development-only
   选择两个字段，在 historical validation 中一次性评价。
5. 补交周内底部五分位 secondary failure、失败标签置换和确定性噪声负对照。
6. F、FC、FMA 使用固定 50 槽、0.5% 现金缓冲、买 15bp、卖 25bp；空槽和不可观测
   槽位按现金代理并单独报告完整率。
7. 递归随机缓冲路径会让随机持仓反馈到下一周保留和提案，不再条件于真实 FC 路径。
8. 状态 0–10 全部作为安慰剂报告；稀有状态换手在完整 224 周日历上插入空集。
9. 市场状态只用两个固定决策时变量：eligible MA20 breadth 的 50% 分割、eligible
   median momentum 的零分割。不得选择表现最好的门。

## 判定含义

允许的状态包括：

- `PASS_EXPLORATORY_HISTORICAL`
- `FALSIFIED_RETROSPECTIVE`
- `INCONCLUSIVE`
- `INSUFFICIENT_DATA`
- `INVALID_POST_SELECTION`
- `INVALID_DATA`

即使出现 PASS，也只能表示历史复用样本支持继续收集数据，不能确认 alpha。控制失败
不能挽救被有效端点证伪的假设；哈希、重构或时间语义失败则整项为无效。

## 产物完整性

默认输出是：

```text
scripts/_output/xs_chan_exploration_stage2/STAGE2_<identity_sha256>/
```

身份绑定机器规格和 Stage 2 源码字节。研究先写同文件系统临时目录，全部实验完成并生成
SHA256、大小、parquet 行数和 schema 清单后才原子发布；manifest 另有 detached
SHA256。校验器会用当前规格、源码和冻结输入重算当前身份，并把 revision history 中的
旧身份明确标为 `VERIFIED_SUPERSEDED`。已存在身份拒绝覆盖；失败运行改名为
`FAILED_<identity>_<timestamp>` 保留。Detached digest 是同目录损坏检测，不是外部
签名；Revision 3 以前的旧身份没有该 digest，其 superseded 验证只证明旧 payload
清单自洽与生命周期已登记，不能由当前工作树重建旧源码身份。

## 命令

```bash
uv run --no-sync python scripts/xs_chan_exploration_stage2.py check
uv run --no-sync python scripts/xs_chan_exploration_stage2.py run
uv run --no-sync python scripts/xs_chan_exploration_stage2.py verify
```
