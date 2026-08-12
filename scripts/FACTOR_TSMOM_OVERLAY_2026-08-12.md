# 因子 F 的 20 周时间序列动量仓位覆盖层

- 状态：**HYPOTHESIS_FALSIFIED_DEVELOPMENT**
- 冻结协议：`factor_tsmom_overlay_protocol_2026-08-12.json`
- 边界：2024+ 仍是历史时间切分；任何通过只允许进入不可变前向测试。

## 固定机制

F 股票选择完全不变。每个周决策收盘，用 F 已实现净值相对 20 个决策日前的涨跌决定下一开盘
持有完整 F 或全部现金；阈值为 0，无迟滞、无杠杆。

## 开发期

- 风险开启率：72.15%。
- 年化收益：F 9.15%，TSMOM -7.29%；
  Sharpe 改善 -1.185。
- 最大回撤：F -13.19%，TSMOM -14.98%；
  相对改善 -13.58%。
- 开发门：**FAIL**。

- `risk_on_rate_floor`: PASS
- `risk_on_rate_cap`: PASS
- `candidate_annual_return_positive`: FAIL
- `drawdown_reduction`: FAIL
- `sharpe_improvement`: FAIL
- `annual_return_noninferiority`: FAIL
- `weekly_hac_noninferiority`: FAIL

## 结论

开发门未全部通过，因此未计算或写出 2024+ 路径与统计。该精确 20 周零阈值覆盖层被否决，
不调整回看期、阈值、迟滞或杠杆救援。
