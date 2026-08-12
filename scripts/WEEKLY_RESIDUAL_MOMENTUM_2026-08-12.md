# 周频行业/规模中性短期动量：锁定时间验证

- 状态：**HYPOTHESIS_FALSIFIED_LOCKED_VALIDATION**
- 冻结协议：`weekly_residual_momentum_protocol_2026-08-12.json`
- 起源：方向来自已冻结超跌反转策略在开发期的失败，因此开发期统计只作描述，不能确认。
- 边界：2024+ 是首次锁定检验，但仍属历史时间切分；通过也不能授权实盘。

## 固定策略

过去 5 个股票观察日收益经申万一级行业和流通市值中性化后，选择正残差最大的 50 只，
排名 75 内保留；次一市场日开盘进入，固定 5 日开盘持有，主成本买 15bp / 卖 25bp。

## 开发期描述（禁止确认）

- 主动周均：-2.361%；HAC t=-7.673；
  bootstrap q05=-2.896%。

## 锁定验证期（2024+）

- 周数：121；组合扣费周均：-1.596%；
  全市场基准：0.485%。
- 主动周均：-2.081%；HAC t=-5.393；
  bootstrap q05=-2.695%。
- 两个锁定半段主动均值：-3.234%, -1.241%。
- 目标槽可观察率：99.669%；
  复利最大回撤：-90.185%。
- 验证门：**FAIL**；
  60bp 压力门：**FAIL**。

### 验证门逐项

- `minimum_active_effect`: FAIL
- `hac_positive`: FAIL
- `bootstrap_q05_positive`: FAIL
- `both_halves_positive`: FAIL
- `candidate_net_mean_positive`: FAIL
- `observable_rate`: PASS
- `drawdown_cap`: FAIL

## 结论

锁定验证或 60bp 成本压力失败；该精确短期动量策略被否决，不做参数救援。
