//! 日频贪心槽位组合模拟。
//!
//! 等价于 Python `surge_portfolio_backtest.py` 中的 `simulate_slots()` + `_daily_returns()` + 统计汇总，
//! 但在 Rust 端一体化完成，支持分层门控和多种填充模式。

use std::collections::HashMap;

use chrono::{DateTime, Datelike, Utc};
use czsc_trend_regime::GateLevel;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SlotBacktestError {
    #[error("slot backtest error: {0}")]
    Config(String),
    #[error("data error: {0}")]
    Data(String),
}

pub type Result<T> = std::result::Result<T, SlotBacktestError>;

// ─── 填充模式 ────────────────────────────────────────────────────

/// 槽位填充模式，决定门控放宽策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillMode {
    /// 仅 Full gate（3/3）候选可入场。
    Strict,
    /// Full + Partial（≥2/3）。
    PartialFill,
    /// Full + Partial + Weak（≥1/3）。
    AllFill,
    /// 按 `gate_confidence` 加权仓位，不设硬性门槛。
    ConfidenceWeighted,
}

impl FillMode {
    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "strict" => Ok(Self::Strict),
            "partial_fill" => Ok(Self::PartialFill),
            "all_fill" => Ok(Self::AllFill),
            "confidence_weighted" => Ok(Self::ConfidenceWeighted),
            other => Err(format!("unknown fill_mode: {other}")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::PartialFill => "partial_fill",
            Self::AllFill => "all_fill",
            Self::ConfidenceWeighted => "confidence_weighted",
        }
    }

    pub fn min_gate_level(&self) -> GateLevel {
        match self {
            Self::Strict => GateLevel::Full,
            Self::PartialFill => GateLevel::Partial,
            Self::AllFill => GateLevel::Weak,
            Self::ConfidenceWeighted => GateLevel::None,
        }
    }
}

// ─── 配置 & 输入 ─────────────────────────────────────────────────

/// 槽位回测配置。
#[derive(Debug, Clone)]
pub struct SlotBacktestConfig {
    pub n_slots: usize,
    pub buy_cost: f64,
    pub sell_cost: f64,
    pub fill_mode: FillMode,
}

impl Default for SlotBacktestConfig {
    fn default() -> Self {
        Self {
            n_slots: 10,
            buy_cost: 0.0015,
            sell_cost: 0.0025,
            fill_mode: FillMode::Strict,
        }
    }
}

/// 单笔候选记录（来自 `surge_candidates_dump` 输出）。
#[derive(Debug, Clone)]
pub struct CandidateRow {
    pub symbol: String,
    pub entry_dt: DateTime<Utc>,
    pub exit_dt: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub gate_level: GateLevel,
    pub gate_confidence: f64,
    pub priority: f64,
    pub ret_gross_pct: f64,
    pub hold_days: usize,
    pub seg: String,
    pub year: i32,
}

/// 收盘价面板：symbol × date → close。
#[derive(Debug, Clone)]
pub struct ClosePanel {
    symbol_idx: HashMap<String, usize>,
    date_idx: HashMap<DateTime<Utc>, usize>,
    dates_sorted: Vec<DateTime<Utc>>,
    data: Vec<Vec<f64>>,
}

impl ClosePanel {
    /// 从 parquet 文件读取（列: symbol, dt, close）。
    pub fn from_parquet(path: &std::path::Path, value_col: &str) -> Result<Self> {
        use polars::prelude::*;
        let pl_path = PlPath::new(path.to_string_lossy().as_ref());
        let df = LazyFrame::scan_parquet(pl_path, Default::default())
            .map_err(|e| SlotBacktestError::Data(format!("read parquet: {e}")))?
            .select([col("symbol"), col("dt"), col(value_col)])
            .collect()
            .map_err(|e| SlotBacktestError::Data(format!("collect: {e}")))?;

        let sym_col = df.column("symbol")
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?
            .str()
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?;
        let dt_col = df.column("dt")
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?
            .datetime()
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?;
        let val_col_data = df.column(value_col)
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?
            .f64()
            .map_err(|e| SlotBacktestError::Data(e.to_string()))?;

        let ts_phys = dt_col.physical();
        let tu = dt_col.time_unit();

        let mut symbol_idx: HashMap<String, usize> = HashMap::new();
        let mut symbols: Vec<String> = Vec::new();
        let mut dates_set: std::collections::BTreeSet<DateTime<Utc>> = std::collections::BTreeSet::new();
        let mut raw: Vec<(usize, DateTime<Utc>, f64)> = Vec::with_capacity(df.height());

        for i in 0..df.height() {
            let sym = sym_col.get(i).unwrap_or("");
            let ts_raw = ts_phys.get(i).unwrap_or(0);
            let val = val_col_data.get(i).unwrap_or(f64::NAN);
            if val.is_nan() { continue; }

            let ts_ms = match tu {
                TimeUnit::Nanoseconds => ts_raw / 1_000_000,
                TimeUnit::Microseconds => ts_raw / 1_000,
                TimeUnit::Milliseconds => ts_raw,
            };
            let dt = DateTime::from_timestamp_millis(ts_ms)
                .unwrap_or_default();
            let si = *symbol_idx.entry(sym.to_string()).or_insert_with(|| {
                let idx = symbols.len();
                symbols.push(sym.to_string());
                idx
            });
            dates_set.insert(dt);
            raw.push((si, dt, val));
        }

        let dates_sorted: Vec<DateTime<Utc>> = dates_set.into_iter().collect();
        let date_idx: HashMap<DateTime<Utc>, usize> =
            dates_sorted.iter().enumerate().map(|(i, d)| (*d, i)).collect();
        let mut data = vec![vec![f64::NAN; dates_sorted.len()]; symbols.len()];
        for (si, dt, close) in &raw {
            if let Some(&di) = date_idx.get(dt) {
                data[*si][di] = *close;
            }
        }

        Ok(Self { symbol_idx, date_idx, dates_sorted, data })
    }

    /// 从 (symbol, date, close) 三元组列表构造。
    pub fn from_triples(triples: &[(String, DateTime<Utc>, f64)]) -> Self {
        let mut symbols: Vec<String> = Vec::new();
        let mut dates_set: std::collections::BTreeSet<DateTime<Utc>> = std::collections::BTreeSet::new();
        let mut raw: HashMap<(String, DateTime<Utc>), f64> = HashMap::new();

        for (sym, dt, close) in triples {
            if !symbols.contains(sym) {
                symbols.push(sym.clone());
            }
            dates_set.insert(*dt);
            raw.insert((sym.clone(), *dt), *close);
        }

        let dates_sorted: Vec<DateTime<Utc>> = dates_set.into_iter().collect();
        let symbol_idx: HashMap<String, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let date_idx: HashMap<DateTime<Utc>, usize> =
            dates_sorted.iter().enumerate().map(|(i, d)| (*d, i)).collect();

        let mut data = vec![vec![f64::NAN; dates_sorted.len()]; symbols.len()];
        for ((sym, dt), close) in &raw {
            if let (Some(&si), Some(&di)) = (symbol_idx.get(sym), date_idx.get(dt)) {
                data[si][di] = *close;
            }
        }

        Self {
            symbol_idx,
            date_idx,
            dates_sorted,
            data,
        }
    }

    pub fn close(&self, symbol: &str, date: &DateTime<Utc>) -> Option<f64> {
        let si = *self.symbol_idx.get(symbol)?;
        let di = *self.date_idx.get(date)?;
        let v = self.data[si][di];
        if v.is_nan() { None } else { Some(v) }
    }

    /// 返回 [start..=end] 范围内的 (date, close) 列表。
    pub fn close_range(
        &self,
        symbol: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Vec<(DateTime<Utc>, f64)> {
        let si = match self.symbol_idx.get(symbol) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let start_pos = self.dates_sorted.partition_point(|d| d < start);
        let end_pos = self.dates_sorted.partition_point(|d| d <= end);
        let mut out = Vec::with_capacity(end_pos - start_pos);
        for di in start_pos..end_pos {
            let v = self.data[si][di];
            if !v.is_nan() {
                out.push((self.dates_sorted[di], v));
            }
        }
        out
    }

    pub fn dates(&self) -> &[DateTime<Utc>] {
        &self.dates_sorted
    }
}

// ─── 输出 ────────────────────────────────────────────────────────

/// 单笔成交记录。
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub symbol: String,
    pub entry_dt: DateTime<Utc>,
    pub exit_dt: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub ret_gross_pct: f64,
    pub ret_net_pct: f64,
    pub gate_level: GateLevel,
    pub gate_confidence: f64,
    pub priority: f64,
    pub hold_days: usize,
    pub seg: String,
    pub year: i32,
    pub position_weight: f64,
}

/// 组合权益统计。
#[derive(Debug, Clone, Default)]
pub struct CurveStats {
    pub annual_return_pct: f64,
    pub sharpe: f64,
    pub max_drawdown_pct: f64,
    pub calmar: f64,
    pub trading_days: usize,
}

/// 配对交易统计。
#[derive(Debug, Clone, Default)]
pub struct PairStats {
    pub n_trades: usize,
    pub win_rate_pct: f64,
    pub profit_loss_ratio: f64,
    pub net_mean_pct: f64,
    pub net_median_pct: f64,
    pub gross_mean_pct: f64,
    pub avg_hold_days: f64,
}

/// 回测结果。
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<TradeRecord>,
    pub daily_returns: Vec<(DateTime<Utc>, f64)>,
    pub curve_stats: CurveStats,
    pub pair_stats: PairStats,
}

// ─── 核心模拟 ────────────────────────────────────────────────────

/// 贪心槽位分配 + 日频 MTM 收益聚合。
pub fn simulate_slots(
    candidates: &[CandidateRow],
    panel: &ClosePanel,
    config: &SlotBacktestConfig,
) -> Result<BacktestResult> {
    if config.n_slots == 0 {
        return Err(SlotBacktestError::Config("n_slots must be > 0".into()));
    }

    let min_level = config.fill_mode.min_gate_level();
    let is_confidence = config.fill_mode == FillMode::ConfidenceWeighted;

    // 1. 按 entry_dt 分组，组内按 priority 降序
    let mut by_entry: std::collections::BTreeMap<DateTime<Utc>, Vec<&CandidateRow>> =
        std::collections::BTreeMap::new();
    for c in candidates {
        if c.gate_level >= min_level || is_confidence {
            by_entry.entry(c.entry_dt).or_default().push(c);
        }
    }
    for group in by_entry.values_mut() {
        group.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
    }

    // 2. 贪心分配
    let mut trades: Vec<TradeRecord> = Vec::new();
    // symbol → exit_dt（exit 当日仍占槽，次日释放）
    let mut open_until: HashMap<String, DateTime<Utc>> = HashMap::new();

    for (entry_dt, day_candidates) in &by_entry {
        open_until.retain(|_, exit_dt| *exit_dt >= *entry_dt);
        let mut free = config.n_slots.saturating_sub(open_until.len());
        if free == 0 {
            continue;
        }

        // 分层填充：Strict 模式下只取 Full; PartialFill 先 Full 再 Partial; ...
        let sorted_by_level: Vec<&CandidateRow> = if is_confidence {
            day_candidates.clone()
        } else {
            let mut sorted = day_candidates.clone();
            sorted.sort_by(|a, b| {
                b.gate_level
                    .cmp(&a.gate_level)
                    .then_with(|| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal))
            });
            sorted
        };

        for c in sorted_by_level {
            if free == 0 {
                break;
            }
            if open_until.contains_key(&c.symbol) {
                continue;
            }
            if !is_confidence && c.gate_level < min_level {
                continue;
            }

            let weight = if is_confidence {
                c.gate_confidence.max(0.0).min(1.0)
            } else {
                1.0
            };

            let ret_net_pct =
                ((1.0 + c.ret_gross_pct / 100.0) * (1.0 - config.sell_cost) / (1.0 + config.buy_cost) - 1.0) * 100.0;

            trades.push(TradeRecord {
                symbol: c.symbol.clone(),
                entry_dt: c.entry_dt,
                exit_dt: c.exit_dt,
                entry_price: c.entry_price,
                exit_price: c.exit_price,
                ret_gross_pct: c.ret_gross_pct,
                ret_net_pct,
                gate_level: c.gate_level,
                gate_confidence: c.gate_confidence,
                priority: c.priority,
                hold_days: c.hold_days,
                seg: c.seg.clone(),
                year: c.year,
                position_weight: weight,
            });
            open_until.insert(c.symbol.clone(), c.exit_dt);
            free -= 1;
        }
    }

    // 3. 日收益聚合
    let daily_returns = compute_daily_returns(&trades, panel, config)?;

    // 4. 统计汇总
    let curve_stats = compute_curve_stats(&daily_returns);
    let pair_stats = compute_pair_stats(&trades);

    Ok(BacktestResult {
        trades,
        daily_returns,
        curve_stats,
        pair_stats,
    })
}

/// 从成交明细 + 收盘价面板计算日收益。
fn compute_daily_returns(
    trades: &[TradeRecord],
    panel: &ClosePanel,
    config: &SlotBacktestConfig,
) -> Result<Vec<(DateTime<Utc>, f64)>> {
    let n_slots = config.n_slots as f64;
    let is_confidence = config.fill_mode == FillMode::ConfidenceWeighted;

    let mut acc: std::collections::BTreeMap<DateTime<Utc>, f64> = std::collections::BTreeMap::new();
    let mut cnt: std::collections::BTreeMap<DateTime<Utc>, usize> = std::collections::BTreeMap::new();

    for tr in trades {
        let prices = panel.close_range(&tr.symbol, &tr.entry_dt, &tr.exit_dt);
        if prices.is_empty() {
            continue;
        }

        let entry_eff = tr.entry_price * (1.0 + config.buy_cost);
        let exit_eff = tr.exit_price * (1.0 - config.sell_cost);
        let weight = if is_confidence { tr.position_weight } else { 1.0 };

        for (k, (dt, px)) in prices.iter().enumerate() {
            let ret = if prices.len() == 1 {
                exit_eff / entry_eff - 1.0
            } else if k == 0 {
                px / entry_eff - 1.0
            } else if k == prices.len() - 1 {
                exit_eff / prices[k - 1].1 - 1.0
            } else {
                px / prices[k - 1].1 - 1.0
            };
            *acc.entry(*dt).or_insert(0.0) += ret * weight;
            *cnt.entry(*dt).or_insert(0) += 1;
        }
    }

    // 槽位守恒检查
    if let Some((&max_dt, &max_cnt)) = cnt.iter().max_by_key(|(_, c)| *c) {
        if max_cnt > config.n_slots {
            return Err(SlotBacktestError::Data(format!(
                "slot conservation violated: {} slots occupied on {:?}, max is {}",
                max_cnt, max_dt, config.n_slots
            )));
        }
    }

    let daily: Vec<(DateTime<Utc>, f64)> = acc.into_iter().map(|(dt, sum)| (dt, sum / n_slots)).collect();
    Ok(daily)
}

/// 计算权益曲线统计。
fn compute_curve_stats(daily_returns: &[(DateTime<Utc>, f64)]) -> CurveStats {
    let n = daily_returns.len();
    if n < 20 {
        return CurveStats::default();
    }

    let rets: Vec<f64> = daily_returns.iter().map(|(_, r)| *r).collect();
    let mut equity = Vec::with_capacity(n);
    let mut cum = 1.0;
    for r in &rets {
        cum *= 1.0 + r;
        equity.push(cum);
    }

    let total = equity.last().copied().unwrap_or(1.0) - 1.0;
    let years = n as f64 / 252.0;
    let ann = if years > 0.0 {
        (1.0 + total).powf(1.0 / years) - 1.0
    } else {
        f64::NAN
    };

    let mean = rets.iter().sum::<f64>() / n as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();
    let sharpe = if std > 0.0 { mean / std * 252.0_f64.sqrt() } else { f64::NAN };

    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in &equity {
        if e > peak {
            peak = e;
        }
        let dd = 1.0 - e / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    let calmar = if max_dd > 0.0 { ann / max_dd } else { f64::NAN };

    CurveStats {
        annual_return_pct: round2(ann * 100.0, 1),
        sharpe: round2(sharpe, 2),
        max_drawdown_pct: round2(max_dd * 100.0, 1),
        calmar: round2(calmar, 2),
        trading_days: n,
    }
}

/// 计算配对交易统计。
fn compute_pair_stats(trades: &[TradeRecord]) -> PairStats {
    if trades.is_empty() {
        return PairStats::default();
    }

    let n = trades.len();
    let rets: Vec<f64> = trades.iter().map(|t| t.ret_net_pct).collect();
    let wins: Vec<f64> = rets.iter().filter(|r| **r > 0.0).copied().collect();
    let losses: Vec<f64> = rets.iter().filter(|r| **r <= 0.0).copied().collect();

    let win_rate = wins.len() as f64 / n as f64 * 100.0;
    let pl_ratio = if !wins.is_empty() && !losses.is_empty() {
        let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        (avg_win / avg_loss).abs()
    } else {
        f64::NAN
    };

    let net_mean = rets.iter().sum::<f64>() / n as f64;
    let mut sorted_rets = rets.clone();
    sorted_rets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let net_median = median_sorted(&sorted_rets);

    let gross_mean = trades.iter().map(|t| t.ret_gross_pct).sum::<f64>() / n as f64;
    let avg_hold = trades.iter().map(|t| t.hold_days as f64).sum::<f64>() / n as f64;

    PairStats {
        n_trades: n,
        win_rate_pct: round2(win_rate, 1),
        profit_loss_ratio: round2(pl_ratio, 2),
        net_mean_pct: round2(net_mean, 2),
        net_median_pct: round2(net_median, 2),
        gross_mean_pct: round2(gross_mean, 2),
        avg_hold_days: round2(avg_hold, 1),
    }
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn round2(v: f64, decimals: u32) -> f64 {
    if v.is_nan() || v.is_infinite() {
        return v;
    }
    let factor = 10.0_f64.powi(decimals as i32);
    (v * factor).round() / factor
}

// ─── 分段统计 ────────────────────────────────────────────────────

/// 分段（ALL / IS / OOS / 年度）汇总。
#[derive(Debug, Clone)]
pub struct SegmentStats {
    pub label: String,
    pub curve: CurveStats,
    pub pair: PairStats,
}

/// 对回测结果按段分切统计。
pub fn segment_stats(
    result: &BacktestResult,
    train_end_year: i32,
) -> Vec<SegmentStats> {
    let mut out = Vec::new();

    // ALL
    out.push(SegmentStats {
        label: "ALL".into(),
        curve: result.curve_stats.clone(),
        pair: result.pair_stats.clone(),
    });

    // IS / OOS
    let is_trades: Vec<TradeRecord> = result.trades.iter().filter(|t| t.seg == "train").cloned().collect();
    let oos_trades: Vec<TradeRecord> = result.trades.iter().filter(|t| t.seg == "test").cloned().collect();

    let is_daily: Vec<(DateTime<Utc>, f64)> = result
        .daily_returns
        .iter()
        .filter(|(dt, _)| dt.date_naive().year() <= train_end_year)
        .cloned()
        .collect();
    let oos_daily: Vec<(DateTime<Utc>, f64)> = result
        .daily_returns
        .iter()
        .filter(|(dt, _)| dt.date_naive().year() > train_end_year)
        .cloned()
        .collect();

    out.push(SegmentStats {
        label: format!("IS(≤{})", train_end_year),
        curve: compute_curve_stats(&is_daily),
        pair: compute_pair_stats(&is_trades),
    });
    out.push(SegmentStats {
        label: format!("OOS(≥{})", train_end_year + 1),
        curve: compute_curve_stats(&oos_daily),
        pair: compute_pair_stats(&oos_trades),
    });

    // 年度
    let mut years: Vec<i32> = result.trades.iter().map(|t| t.year).collect();
    years.sort();
    years.dedup();

    for y in years {
        let yr_trades: Vec<TradeRecord> = result.trades.iter().filter(|t| t.year == y).cloned().collect();
        let yr_daily: Vec<(DateTime<Utc>, f64)> = result
            .daily_returns
            .iter()
            .filter(|(dt, _)| dt.date_naive().year() == y)
            .cloned()
            .collect();
        out.push(SegmentStats {
            label: y.to_string(),
            curve: compute_curve_stats(&yr_daily),
            pair: compute_pair_stats(&yr_trades),
        });
    }

    out
}

// ─── 测试 ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn make_candidate(
        sym: &str,
        entry: DateTime<Utc>,
        exit: DateTime<Utc>,
        entry_px: f64,
        exit_px: f64,
        level: GateLevel,
        confidence: f64,
        priority: f64,
    ) -> CandidateRow {
        let ret_gross = (exit_px / entry_px - 1.0) * 100.0;
        CandidateRow {
            symbol: sym.into(),
            entry_dt: entry,
            exit_dt: exit,
            entry_price: entry_px,
            exit_price: exit_px,
            gate_level: level,
            gate_confidence: confidence,
            priority,
            ret_gross_pct: ret_gross,
            hold_days: 3,
            seg: "train".into(),
            year: 2023,
        }
    }

    fn make_panel(data: &[(&str, DateTime<Utc>, f64)]) -> ClosePanel {
        let triples: Vec<(String, DateTime<Utc>, f64)> =
            data.iter().map(|(s, d, c)| (s.to_string(), *d, *c)).collect();
        ClosePanel::from_triples(&triples)
    }

    #[test]
    fn test_strict_mode_rejects_partial() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);

        let candidates = vec![
            make_candidate("A", d1, d3, 10.0, 11.0, GateLevel::Full, 1.0, 80.0),
            make_candidate("B", d1, d3, 10.0, 10.5, GateLevel::Partial, 0.7, 70.0),
        ];
        let panel = make_panel(&[
            ("A", d1, 10.0),
            ("A", utc(2023, 1, 3), 10.5),
            ("A", d3, 11.0),
            ("B", d1, 10.0),
            ("B", utc(2023, 1, 3), 10.2),
            ("B", d3, 10.5),
        ]);
        let config = SlotBacktestConfig {
            n_slots: 2,
            fill_mode: FillMode::Strict,
            ..Default::default()
        };
        let result = simulate_slots(&candidates, &panel, &config).unwrap();
        assert_eq!(result.trades.len(), 1, "Strict should reject Partial");
        assert_eq!(result.trades[0].symbol, "A");
    }

    #[test]
    fn test_partial_fill_accepts_partial() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);

        let candidates = vec![
            make_candidate("A", d1, d3, 10.0, 11.0, GateLevel::Full, 1.0, 80.0),
            make_candidate("B", d1, d3, 10.0, 10.5, GateLevel::Partial, 0.7, 70.0),
        ];
        let panel = make_panel(&[
            ("A", d1, 10.0),
            ("A", utc(2023, 1, 3), 10.5),
            ("A", d3, 11.0),
            ("B", d1, 10.0),
            ("B", utc(2023, 1, 3), 10.2),
            ("B", d3, 10.5),
        ]);
        let config = SlotBacktestConfig {
            n_slots: 2,
            fill_mode: FillMode::PartialFill,
            ..Default::default()
        };
        let result = simulate_slots(&candidates, &panel, &config).unwrap();
        assert_eq!(result.trades.len(), 2, "PartialFill should accept both");
    }

    #[test]
    fn test_slot_conservation() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);

        let candidates: Vec<CandidateRow> = (0..5)
            .map(|i| {
                make_candidate(
                    &format!("S{i}"),
                    d1,
                    d3,
                    10.0,
                    11.0,
                    GateLevel::Full,
                    1.0,
                    50.0 - i as f64,
                )
            })
            .collect();
        let mut panel_data = Vec::new();
        for c in &candidates {
            panel_data.push((c.symbol.as_str(), d1, 10.0));
            panel_data.push((c.symbol.as_str(), utc(2023, 1, 3), 10.5));
            panel_data.push((c.symbol.as_str(), d3, 11.0));
        }
        let panel = make_panel(&panel_data);
        let config = SlotBacktestConfig {
            n_slots: 2,
            fill_mode: FillMode::Strict,
            ..Default::default()
        };
        let result = simulate_slots(&candidates, &panel, &config).unwrap();
        assert_eq!(result.trades.len(), 2, "only 2 slots available");
    }

    #[test]
    fn test_daily_returns_positive() {
        let d1 = utc(2023, 1, 2);
        let d2 = utc(2023, 1, 3);
        let d3 = utc(2023, 1, 4);

        let candidates = vec![
            make_candidate("A", d1, d3, 10.0, 11.0, GateLevel::Full, 1.0, 80.0),
        ];
        let panel = make_panel(&[("A", d1, 10.0), ("A", d2, 10.5), ("A", d3, 11.0)]);
        let config = SlotBacktestConfig {
            n_slots: 1,
            fill_mode: FillMode::Strict,
            ..Default::default()
        };
        let result = simulate_slots(&candidates, &panel, &config).unwrap();
        let total_ret: f64 = result.daily_returns.iter().map(|(_, r)| *r).sum();
        assert!(total_ret > 0.0, "positive price movement should give positive returns");
    }

    #[test]
    fn test_curve_stats_basic() {
        let daily: Vec<(DateTime<Utc>, f64)> = (0..252)
            .map(|i| (utc(2023, 1, 1) + chrono::Duration::days(i), 0.001))
            .collect();
        let stats = compute_curve_stats(&daily);
        assert!(stats.annual_return_pct > 0.0);
        assert!(stats.sharpe > 0.0);
        assert!(stats.trading_days == 252);
    }

    #[test]
    fn test_pair_stats_basic() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);
        let trades = vec![
            TradeRecord {
                symbol: "A".into(),
                entry_dt: d1,
                exit_dt: d3,
                entry_price: 10.0,
                exit_price: 11.0,
                ret_gross_pct: 10.0,
                ret_net_pct: 9.0,
                gate_level: GateLevel::Full,
                gate_confidence: 1.0,
                priority: 80.0,
                hold_days: 3,
                seg: "train".into(),
                year: 2023,
                position_weight: 1.0,
            },
            TradeRecord {
                symbol: "B".into(),
                entry_dt: d1,
                exit_dt: d3,
                entry_price: 10.0,
                exit_price: 9.5,
                ret_gross_pct: -5.0,
                ret_net_pct: -5.5,
                gate_level: GateLevel::Full,
                gate_confidence: 1.0,
                priority: 70.0,
                hold_days: 3,
                seg: "train".into(),
                year: 2023,
                position_weight: 1.0,
            },
        ];
        let stats = compute_pair_stats(&trades);
        assert_eq!(stats.n_trades, 2);
        assert_eq!(stats.win_rate_pct, 50.0);
        assert!(stats.profit_loss_ratio > 0.0);
    }

    #[test]
    fn test_fill_mode_str_roundtrip() {
        for mode in [FillMode::Strict, FillMode::PartialFill, FillMode::AllFill, FillMode::ConfidenceWeighted] {
            let parsed = FillMode::from_str(mode.as_str()).unwrap();
            assert_eq!(parsed, mode);
        }
    }
}
