# 周频行业/规模中性超跌反转：预注册回顾性检验

- 状态：**HYPOTHESIS_FALSIFIED_DEVELOPMENT**
- 冻结协议：`weekly_residual_reversal_protocol_2026-08-12.json`
- 边界：锁定验证仍是历史时间切分；通过也只能进入日频自融资复核与独立前向测试。

## 固定策略

每周用过去 5 个股票观察日收益，对申万一级行业和流通市值做横截面中性化；买入残差跌幅最大的
50 只，排名 75 内保留，下一市场日开盘成交，固定 5 日开盘持有。主成本为买 15bp、卖 25bp。

## 开发期（2022–2023）

- 周数：99；组合扣费周均：-0.697%；全市场基准：-0.129%。
- 主端点主动周均：-0.568%；HAC t=-3.137；
  block-bootstrap q05=-0.861%。
- 目标槽可观察率：99.737%；复利最大回撤：-52.932%。
- 开发门：**FAIL**。

- `minimum_active_effect`: FAIL
- `hac_positive`: FAIL
- `bootstrap_q05_positive`: FAIL
- `both_halves_positive`: FAIL
- `candidate_net_mean_positive`: FAIL
- `observable_rate`: PASS
- `drawdown_cap`: FAIL

## 结论

开发门未全部通过，因此未计算或写出 2024+ 收益与统计。这个精确的连续超跌反转假设被
回顾性否决，不调整回看期、中性化、持仓数、缓冲排名或持有期救援。
