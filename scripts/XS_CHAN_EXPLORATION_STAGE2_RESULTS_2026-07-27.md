# 横截面因子 × 缠论 Stage 2：结果冻结

- 日期：2026-07-27
- 研究 ID：`xs_chan_exploration_stage2_hypothesis_stress_20260727`
- 有效内容身份：`1a92a698ecb6b042e96496139a461ad673a99235a9d4449833b1da224121ae46`
- 性质：`RETROSPECTIVE_FALSIFICATION_ONLY`
确认链：`NOT_STARTED`

## 最终判断

不应继续在当前历史样本上优化状态 5–8 门、因子权重或失败画像阈值。修正
Stage 1 的对照人群、时间泄漏和递归路径问题后：

- H1 状态 7 独立增量：`FALSIFIED_RETROSPECTIVE`；
- H2 风险归因后因子残差：`FALSIFIED_RETROSPECTIVE`；
- 原 H3 两个失败画像：`INVALID_POST_SELECTION`；
- early-only 选择、validation-only 评价的嵌套 H3：
  `FALSIFIED_RETROSPECTIVE`。

这些是已经看过全样本后的历史证伪，不是独立 OOS。缓存中只有 3 个 Stage 1 未使用
且 5-session 标签完整的周，20-session 诊断完整周为 0，无法启动新确认。

## H1：同一 proposal population 上的状态 7

FC proposal 的共同决策周口径中，状态 7 减 SMA20：

| 分段 | 周差 |
| --- | ---: |
| Development | -0.403% |
| Historical validation | +0.016% |
| V1 | -0.199% |
| V2 | +0.227% |

验证段均值接近零，V1、V2 反号，因此命中冻结的证伪规则。F buffered membership
敏感性口径在验证段为 -0.021%/周，V1、V2 分别为 -0.018%、-0.024%。

“共同决策周”不是名称级共同支持；状态 7 与 SMA20 cohort 也可能重叠，因此这个端点
不能解释为缠论状态的因果增量。

精确匹配端点只能匹配 5.05% 的状态 7 事件，远低于 95% 数据门。它显示的
+1.510%/周不得解释，也不得计算有效 p 值。Revision 3 已确保 treated 与 control
都只来自相同精确 strata 和相同匹配数。保持周度配额的 proposal 身份随机化中，
状态 7 减随机组为 -0.122%/周。

## H2：横截面归因残差

Stage 2 逐周重做 OLS；最大六分量重构误差为
`5.55e-17`，eligible universe 残差均值最大绝对值为 `1.35e-16`。
原 Stage 1 的 `market` 分量正式改称“横截面共同收益”，不再表述为指数 beta。

| 分段 | 因子池未解释残差 | HAC t |
| --- | ---: | ---: |
| Development | -0.058%/周 | -0.85 |
| Historical validation | -0.208%/周 | -2.51 |
| V1 | -0.143%/周 | -1.06 |
| V2 | -0.273%/周 | -2.89 |

随机 50 身份残差空值接近 0；实际 validation 残差减空值为 -0.208%/周，单侧尾位置
为 0.999。结果不支持正残差。

## H3：失败画像

`distance_to_sma20`、`volume_ratio` 使用全样本选择，不能把 2024 年以后重新称为
时间外验证。它们只保留诊断，validation SMD 分别为 +0.103、+0.077。

真正按代码隔离的 development-only 选择得到：

| 字段 | Development SMD | Validation SMD |
| --- | ---: | ---: |
| `mom_120_20` | +0.105 | -0.045 |
| `factor_score` | -0.082 | +0.015 |

两个字段都反号且幅度低于 0.10，嵌套诊断被证伪。周内底部五分位 secondary
failure 已补交；其中 turnover、距 SMA20 等差异只作为新的 post-hoc 描述，不升级为
过滤器。

## 固定 50 槽、现金与成本

使用 0.5% 现金缓冲、买 15bp、卖 25bp、空槽现金代理：

| 路径 | Historical validation 净周均 | 平均槽位 |
| --- | ---: | ---: |
| F | +0.103% | 50.0 |
| FC | -0.056% | 28.6 |
| FMA | +0.004% | 33.8 |
| FC − F | -0.159% | — |
| FC − FMA | -0.060% | — |

500 条递归随机缓冲路径会让随机持仓反馈到下一周保留和提案。随机 FC 空值为
+0.041%/周，实际 FC 减空值为 -0.097%/周，尾位置为 1.0。当前状态门既严重
欠配，也没有优于随机状态身份。这个比较包含身份变化和更高随机填仓/暴露的总机制，
不能单独解释成纯 identity alpha。

## 安慰剂与市场状态

11 个状态全部报告，未选择最好状态。状态 3 在 historical validation 的事后
安慰剂表中较突出：

- FC proposal：+1.167%/周，HAC t = 2.48，132 个事件；
- F membership：+1.184%/周，HAC t = 2.42，189 个事件。

它们分别只有 56、65 个非空状态周，是有状态周的原始 cohort 均值，不是 125 周含现金
的固定槽路径，也不是风险调整增量。

它来自多状态事后扫描，样本和路径也未形成预先冻结的新策略，因此不能据此把允许状态
从 5–8 换成 3。若未来取得独立数据，只能先单独登记“状态 3 是否代表反转机制”的
新假设。

两个预先固定的市场变量没有强而可操作的关系。validation Spearman 绝对值最大只有
0.127；MA20 breadth 对 FC−F 几乎为 0。没有足够证据加入市场状态过滤器。

## 完整性与版本

- V2.1 的物理、canonical 和文档 SHA256 均保持不变；正式确认链未启动。
- Stage 2 规格、源码和 10 个 Stage 1 直接输入均被哈希绑定。
- 10/10 实验完成；35 个 payload 经 SHA、大小、parquet 行数和 schema 校验，manifest
  另有 detached SHA256，共核验 37 个文件。
- 校验器重算当前规格、源码和冻结输入身份；当前目录为 `VERIFIED_CURRENT`，旧目录为
  `VERIFIED_SUPERSEDED`。
- Detached digest 是同目录损坏检测，不是外部签名。两个旧身份发布时没有该 digest；
  对它们的验证只证明旧 payload 清单自洽和生命周期已登记，不能用当前工作树重建其
  旧源码/规格身份。
- 输出目录拒绝覆盖；再次运行同一身份已验证会直接失败。
- 首个身份
  `14cd6318aa878103bd200b3c239ca26a44257bc191ee25f4acb8368d21328155`
  被保留。它正确标记匹配端点数据不足，但仍把该端点的尾概率放入 Holm 表，因此
  被 Revision 2 取代；效应值和假设状态没有改变。
- Revision 2 身份
  `dda1f6e18efa219d292dd5c8a87d968858bcca7cfcadeef99b0a8edda4df80fa`
  也被保留。独立审计发现它的 matched treated 均值仍混入未匹配 strata，分段事件数和
  槽位元数据重复全样本值，且 verifier 未区分当前与旧身份，因此由 Revision 3 取代；
  四项假设状态均未改变。

有效机器产物位于：

```text
scripts/_output/xs_chan_exploration_stage2/
  STAGE2_1a92a698ecb6b042e96496139a461ad673a99235a9d4449833b1da224121ae46/
```

## 下一步

1. 停止同样本状态门、因子权重和阈值优化。
2. 不把 Stage 1 或 Stage 2 的失败画像升级为过滤器。
3. 收集至少 26 个全新、不可变、5-session 标签完整的周；更稳妥的机制评价需要
   52 周。
4. 新数据到达前，只登记状态 3 反转机制和市场状态机制的候选问题，不运行选择性
   回测，也不启动 V2.1。
