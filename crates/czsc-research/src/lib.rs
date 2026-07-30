//! czsc-research —— xs_chan 研究引擎的计算层。
//!
//! 将 Stage 2/3 和 V2.1 管线中的性能关键计算从 Python 迁移至 Rust：
//! - `state_cache`: 并行状态投影 + 前缀审计
//! - `features`: 因子面板（动量/低波/ADV/SMA/排名）
//! - `portfolio`: FC/F/FMA 组合模拟
//! - `execution`: 开盘竞价/成交/NAV/公司行为
//! - `statistics`: HAC + circular block bootstrap
//! - `benchmark`: EW 市场基准
//! - `slot_backtest`: 日频贪心槽位回测引擎
//! - `random_control`: 随机对照 beta 剥离

pub mod benchmark;
pub mod execution;
pub mod features;
pub mod portfolio;
pub mod random_control;
pub mod slot_backtest;
pub mod state_cache;
pub mod statistics;

#[cfg(feature = "python")]
pub mod python;
