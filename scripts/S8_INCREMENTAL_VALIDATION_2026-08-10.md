# S8 增量验证复核与跨设备研究交接（2026-08-10）

> 本文是 `S8_INCREMENTAL_VALIDATION_2026-07-31.md` 的最新复核结论。
> 结论已纳入 Git；原始行情缓存和 `scripts/_output/` 生成物没有上传，另一台设备必须按本文的
> 数据身份和复现步骤重新准备。

## 一、结论

**S8 继续关闭，研究基线回落到 S2b Only（A）/ S2b+Cash（E），不授权实盘。**

更新至 2026-08-07 的数据没有提供可复现的 S2c 卫星增量 alpha：

- S8a（强制替换）与 S8b（独立预算）均只通过预注册判据 2/6；
- S8c/F（允许卫星挤占核心）只通过 1/6，并违反核心交易不变性 P6；
- 三个方案全部未通过硬条款 P2（增量显著）与 P3（反彩票）；
- walk-forward、多重检验修正、架构新鲜段和随机挤占对照均不支持上线。

本轮新增日期尚未覆盖最长 60 个交易日的完整退出周期，不构成新的完整独立 OOS。
主结论仍应以架构新鲜段、冻结至 2026-07-27 的留出段、walk-forward 和随机对照为准。

## 二、数据与执行身份

| 项目 | 值 |
| --- | --- |
| 请求结束日 | 2026-08-10 |
| 安全截止交易日 | **2026-08-07** |
| active 股票文件数 | 5,750 |
| 更新文件数 | 5,539 |
| 新增股票数 | 7 |
| qfq 接缝修复数 | 0 |
| active inventory SHA256 | `868388405f0417a7a9933af363a14b13efc6b3e2287cc2826ca97fd5fc566002` |
| 同步审计 SHA256 | `3335faf67a77fcd28eb5920de79f0b24fc32b29cf6aba6a61aab0ef3c1ad0277` |
| 同步执行源码提交 | `ed5ae3cc446595e59e8093678bd974cbd4f70ae2` |
| 同步分支 | `feat/surge-wave-strategy` |

发布审计文件名、文件内容 SHA256、审计中的 `after_inventory`、
`expected_active_inventory_sha256` 与 active 目录实测 inventory 完全一致。

本地内容寻址快照身份：

- 发布前：`RAW_9464dccab24cfa82d383a125b2ef6f1a1f3bedbf1f669dd4c57fd660db0131a0`
- 发布后：`RAW_1555248c805489936ea5e78260c6cdb09c610ed0cce17d1d16572a311d81972f`

这些快照位于本机 `~/.ts_data_cache/xs_chan_raw_snapshots/`，不在 Git 仓库中。

## 三、重建产物

执行 `scripts/surge_candidates_dump.py` 后：

| 产物 | 规模 |
| --- | ---: |
| 研究股票宇宙 | 5,750 只 |
| `candidates.parquet` | 1,252,250 行 |
| `panel.parquet` | 5,634,356 行 |
| d=0 候选 | 264,791 行 |
| S2b 候选 | 1,269 |
| S2c 候选 | 3,557 |
| S2c-only | 2,288 |
| S2b \ S2c | 0 |

上述 Parquet 位于 `scripts/_output/surge_candidates/`，被 `.gitignore` 排除，未随本文上传。

## 四、最新 S8 结果

### 4.1 主窗（2024-01-01 至 2026-08-07，仅作历史对齐）

主窗已经受选择过程污染，只能用于解释机制，不能作为确认性证据。

| 臂 | Sharpe | CAGR | MDD |
| --- | ---: | ---: | ---: |
| A · S2b Only | 0.878 | 20.38% | -37.89% |
| E · S2b+Cash | 0.910 | 21.34% | -37.71% |
| S8a · 强制替换 | 0.780 | 20.37% | -37.28% |
| S8b · 独立预算 n_sat=5 | 1.149 | 29.18% | -26.41% |
| S8c/F · 允许挤占 | 1.170 | 34.27% | -25.03% |

三通道分解：

| 通道 | ΔCAGR | ΔSharpe | 解释 |
| --- | ---: | ---: | --- |
| S8a - A | -0.01% | **-0.098** | 核心集合不变时，纯加卫星没有贡献 |
| F - S8a | +13.90% | +0.391 | 样本内优势来自挤占核心交易 |
| overlay - blend | +14.21% | 0.000 | 纯暴露/杠杆效应 |

挤占通道的 HAC 单边 p=0.0988，block-bootstrap 95% CI 跨零，不能声称显著。

### 4.2 架构新鲜段（2021-07 至 2023-12）

| 臂 | Sharpe | CAGR | MDD |
| --- | ---: | ---: | ---: |
| A | 0.253 | 2.87% | -22.08% |
| S8a | **-0.709** | -16.32% | -39.94% |
| S8b n_sat=5 | 0.198 | 1.94% | -20.59% |
| S8c/F | **-0.651** | -15.08% | -35.45% |

S8a 与 F 在架构新鲜段大幅劣于 A；S8b 也没有超过 A。

### 4.3 2026 留出段（冻结至 2026-07-27）

| 臂 | Sharpe | CAGR | MDD |
| --- | ---: | ---: | ---: |
| A | -1.994 | -45.37% | -37.57% |
| S8a | -1.404 | -39.04% | -36.94% |
| S8b n_sat=5 | -0.984 | -28.11% | -32.13% |
| S8c/F | -0.145 | -9.22% | -24.83% |

该窗口全臂为负且样本较薄，只能用于方向检查。F 的相对改善同时伴随 P6 违规，不能解释为
“相同核心上的增量 alpha”。

### 4.4 Walk-forward 与预注册判定

| 臂 | 正折数 | 平均 ΔSharpe | P1-P6 | P2 日均差 | P3 市场超额中位数 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S8a | 2/8 | -0.282 | 2/6 | -1.720 bps | -1.320% | 关闭 |
| S8b | 3/8 | -0.200 | 2/6 | -1.597 bps | -1.411% | 关闭 |
| S8c/F | 4/8 | -0.140 | 1/6 | +0.417 bps | -1.504% | 关闭 |

P2 的 HAC t 值分别为 -0.878、-0.878、+0.139，三个 block-bootstrap 区间全部跨零。
walk-forward 年化 Sharpe 分别为 0.477、0.600、0.702，均低于 365/430 次试验对应的
DSR 门槛。

### 4.5 随机挤占对照

- F 在主窗挤掉 A 的 154/332 笔核心交易，并额外加入 27 笔核心交易；
- F 留存交易均值 2.693%，随机分层剔除零分布为 2.147% ± 0.357%；
- 选择层分位为 **93.3%**，低于预注册 95% 门槛；
- 组合层 F 纯核心 Sharpe=0.701，虽在随机零分布 98.6 分位，但仍低于不挤占的 A=0.878；
- 因而“优于随机挤占”仍不等于“优于不挤占”，且该统计是事后搜索结果。

## 五、工程镜像诊断

最近 10 个交易日的市场状态重算与研究面板完全一致。TAIL=160 的独立抽样检查结果：

- 有效样本 243 只；
- 末根 regime 一致 243/243；
- 最近 10 根 onset 集合一致 243/243。

`surge_delay5_mirror_check.py` 当前存在一个待修的比较器口径问题：dump 侧仍使用旧门控
`sig_vol_ratio >= 0.8 && sig_above_zg == 1`，而当前 Rust/live 默认门控已经是
`sig_vol_ratio <= 0.8` 且不使用 `above_zg`。

因此原脚本报告的 `live=18 / dump=173 / matched=0` 是旧门控造成的假警报。保持其余条件不变，
按当前 Rust 门控重算 dump 后得到：

- matched：18/18；
- live_only：0；
- dump_only：0。

这个问题不影响 S8 主审计：`_s2c_core.py` 的 S2b/S2c 门控使用当前的 `<=` 方向。
后续应先修正镜像脚本，再把其生成报告作为正式工程证据。

## 六、另一台设备的复现顺序

```bash
git fetch mine feat/surge-wave-strategy
git switch feat/surge-wave-strategy
git pull --ff-only mine feat/surge-wave-strategy

# 精确复现本文身份时固定到 2026-08-07；默认是不发布的 dry-run
uv run --no-sync python scripts/_sync_daily_data.py --end-date 20260807

# 只有满足 clean/pushed、远端绑定和完整性门槛后才正式发布
uv run --no-sync python scripts/_sync_daily_data.py --end-date 20260807 --apply

# 数据改变后必须依次重建
uv run --no-sync python scripts/surge_candidates_dump.py
uv run --no-sync python scripts/s8_incremental_validation.py
uv run --no-sync python scripts/surge_market_state_filter.py
uv run --no-sync python scripts/check_tail_consistency.py 160 300
```

如果另一台设备执行时已经晚于 2026-08-10，应把 `--end-date` 改为当日，并记录新的
safe end、inventory SHA256 与审计 SHA256；新结果应另建日期报告，不能覆盖本文身份。

## 七、下一步研究边界

1. **冻结当前参数**，不要用 2026-08 新增尾部重新调 S2b/S2c 门控。
2. 等最新信号覆盖完整 60 个交易日退出周期后，再形成真正的新前向 OOS。
3. 修正 Delay5 镜像比较器的旧门控，并重新要求选择集合、字段和市场状态同时通过。
4. 新方案必须相对 A/E 做配对增量检验；漂亮的绝对净值不能替代增量证据。
5. 在 P2、P3、P5 和 P6 同时通过前，保持 `live_authorized=false`。
