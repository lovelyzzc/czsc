# 因子 F 的 20 周回撤再进入覆盖层

- 状态：**HYPOTHESIS_FALSIFIED_LOCKED_VALIDATION**
- 冻结协议：`factor_drawdown_reentry_protocol_2026-08-13.json`
- 起源：规则方向来自正趋势覆盖层在开发期的失败；开发期数字禁止用于确认。
- 边界：2024+ 是首次锁定时间检验，但仍非独立前向样本。

## 固定机制

F 选股完全不变。仅当 F 的已实现收盘 NAV 不高于 20 个周决策前时，在下一开盘持有完整 F；
其余时间现金。前 20 个决策为空仓，无迟滞、无杠杆。

## 开发期描述（禁止确认）

- 风险开启率 27.85%；年化收益 17.22%；
  Sharpe 改善 1.568；回撤相对改善 46.51%。

## 锁定验证期（2024+）

- 风险开启率：35.54%。
- 年化收益：F 6.80%，回撤再进入 8.91%；
  Sharpe 改善 0.219。
- 最大回撤：F -24.05%，候选 -24.05%；
  相对改善 -0.01%。
- 验证门：**FAIL**；
  60bp 压力门：**PASS**。

### 验证门逐项

- `risk_on_rate_floor`: PASS
- `risk_on_rate_cap`: PASS
- `candidate_annual_return_positive`: PASS
- `drawdown_reduction`: FAIL
- `sharpe_improvement`: PASS
- `annual_return_noninferiority`: PASS
- `weekly_hac_noninferiority`: FAIL
- `locked_validation_subperiods`: FAIL

## 结论

锁定验证或 60bp 压力失败；该精确回撤再进入规则被否决，不做参数救援。
