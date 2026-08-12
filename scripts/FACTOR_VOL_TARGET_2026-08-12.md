# 因子 F 的 20 日 / 15% 目标波动率覆盖层

- 状态：**HYPOTHESIS_FALSIFIED_DEVELOPMENT_STOP_SAME_SAMPLE_MINING**
- 冻结协议：`factor_vol_target_protocol_2026-08-12.json`
- 边界：全为历史样本；失败后停止同样本挖掘，等待新不可变数据。

## 开发期

- 平均目标仓位：89.31%。
- 年化收益：F 0.95%，VolTarget -0.13%；
  Sharpe 改善 -0.076。
- 最大回撤：F -22.10%，VolTarget -16.46%；
  相对改善 25.51%。
- 开发门：**FAIL**。

- `exposure_floor`: PASS
- `exposure_cap`: PASS
- `positive_return`: FAIL
- `drawdown_reduction`: PASS
- `sharpe_improvement`: FAIL
- `return_noninferiority`: FAIL
- `hac_noninferiority`: FAIL

## 结论

开发门未全部通过，2024+ 未打开。该精确波动率覆盖层被否决；按协议应停止在同一历史样本
继续挖掘，转入新不可变数据收集。
