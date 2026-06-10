# Surge Regime 市场状态过滤器前推验证（2026-06-10）

本文是 `SURGE_REGIME_AUDIT_2026-06-10.md` 之后的下一轮迭代记录。

## 目标

审计报告指出：主升浪信号只在 2025 类市场环境有超额，下一步最高优先级是验证
「市场状态过滤器」能否区分有效 / 失效阶段。

本轮不修改 FSM、不修改 `surge_onset`，只在现成
`scripts/_output/surge_candidates/candidates.parquet` 与 `panel.parquet` 上做后置过滤验证。

## 新增脚本

- `scripts/surge_market_state_filter.py`
  - 计算因果市场状态变量；
  - 对固定的 a-priori 状态门做候选级分桶与实盘镜像组合回测；
  - 使用稳定随机对照：同一笔交易无论出现在哪个过滤器里，都匹配同一组
    同日、同成交额十分位随机样本。
- `scripts/surge_market_state_walkforward.py`
  - 前推验证；
  - 每个测试年只允许用更早年份选择过滤门；
  - 默认 OOS 起点为 2024，和审计报告口径一致。

## 固定状态变量

全部变量只使用决策日收盘前可见数据：

- `mkt_ret20_median`：全市场 20 日收益中位数；
- `mkt_ret20_pos_ratio`：20 日收益为正的股票占比；
- `high20_ratio`：创 20 日新高股票占比；
- `ew_index_above_ma20`：等权市场指数是否在 MA20 上方；
- `small_minus_large_ret20`：低成交额组相对高成交额组的 20 日收益差。

## 组合层 OOS 复核

命令：

```bash
uv run --no-sync python scripts/surge_market_state_filter.py
```

输出：

- `scripts/_output/surge_market_state_filter/report.md`
- `scripts/_output/surge_market_state_filter/summary.json`
- `scripts/_output/surge_market_state_filter/market_state.parquet`

关键结果（10 槽，2024+ OOS）：

| 模式 | 过滤门 | OOS n | 毛超额 | t | 单笔净均值 | 净年化 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| confirm | baseline | 394 | +1.10% | 1.32 | +0.04% | -8.1% | fail |
| confirm | `mkt_ret20_median > 0` | 242 | +2.48% | 2.22 | +1.01% | +6.5% | pass |
| confirm | `high20_ratio > 0.12` | 221 | +2.33% | 2.10 | +0.54% | +0.2% | pass |
| anticipate | baseline | 490 | -0.26% | -0.48 | -1.09% | -24.9% | fail |
| anticipate | `mkt_ret20_pos_ratio > 0.55` | 280 | +2.28% | 2.20 | +1.28% | +10.1% | pass |
| anticipate | `high20_ratio > 0.12` | 290 | +2.16% | 2.36 | +0.57% | +1.6% | pass |

结论：市场状态门确实能把 baseline 的失效状态过滤掉一部分，但复杂组合门并不优于简单门。

## 前推验证

命令：

```bash
uv run --no-sync python scripts/surge_market_state_walkforward.py
```

选择规则：

1. 每个测试年只看更早年份；
2. 在固定的 a-priori 状态门中，选历史毛超额 t 值最高且均值为正的非 baseline 门；
3. 单门历史样本少于 60 笔时不参与选择；
4. 若没有合格门，则退回 baseline。

输出：

- `scripts/_output/surge_market_state_walkforward/report.md`
- `scripts/_output/surge_market_state_walkforward/summary.json`

2024+ 前推聚合结果：

| 模式 | 组合 | 年份 | n | 毛超额 | t | 单笔净均值 | 净年化 | 最大回撤 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| confirm | 前推选择门 | 2024-2026 | 226 | +1.87% | 1.64 | +0.49% | +0.4% | 28.5% |
| confirm | baseline | 2024-2026 | 394 | +1.10% | 1.32 | +0.04% | -8.1% | 51.3% |
| anticipate | 前推选择门 | 2024-2026 | 290 | +2.16% | 2.36 | +0.57% | +1.6% | 32.5% |
| anticipate | baseline | 2024-2026 | 490 | -0.26% | -0.48 | -1.09% | -24.9% | 49.0% |

逐年看：

- `confirm`：2025 有效，2024 / 2026 仍弱，前推聚合未过 t>=2。
- `anticipate`：前推选择稳定落在 `high20_ratio > 0.12 & ew_index_above_ma20`；
  2024 为正但不显著，2025 显著，2026 仍为正超额但 t 很弱、净收益为负。

## 本轮判定

市场状态过滤器方向 **通过研究层验证，但尚不建议直接默认实盘启用**。

理由：

- `anticipate` 前推聚合穿过预声明 OOS 超额 t>=2；
- 但净年化只有 +1.6%，最大回撤仍有 32.5%；
- 2026YTD 净收益仍为负，说明状态过滤解决了部分 beta 依赖，但没有完全解决入场时点问题。

## 下一步建议

继续走审计报告中的第二优先级方向：

1. 保留 `anticipate` 市场状态门作为外层过滤；
2. 在其内部验证「离开中枢后 5-7 天 + 仍在上行家族」的回踩买点；
3. 用同一套前推选择规则验证，不允许按 2025 结果精调；
4. 若回踩买点不能把净年化 / 回撤显著改善，则不要并入 daily_scan 默认逻辑。
