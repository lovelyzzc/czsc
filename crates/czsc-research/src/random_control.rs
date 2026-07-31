//! 随机对照 beta 剥离。
//!
//! 等价于 Python `surge_portfolio_backtest.py` 中的 `ControlSampler`，
//! 但批量处理所有 trades，利用 Rust 的性能优势。
//!
//! 逻辑：每笔成交配同决策日、同成交额十分位的随机 K 只对照
//! （同日历持有期、同次日开盘入），超额 = 策略毛收益 − 对照中位毛收益。

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use thiserror::Error;

use crate::slot_backtest::TradeRecord;

#[derive(Error, Debug)]
pub enum ControlError {
    #[error("control error: {0}")]
    Data(String),
}

pub type Result<T> = std::result::Result<T, ControlError>;

/// 随机对照配置。
#[derive(Debug, Clone)]
pub struct ControlConfig {
    pub k: usize,
    pub min_valid: usize,
    pub seed: u64,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            k: 50,
            min_valid: 10,
            seed: 42,
        }
    }
}

/// 成交额面板：symbol × date → amount_e（亿元）。
#[derive(Debug, Clone)]
pub struct AmountPanel {
    symbol_idx: HashMap<String, usize>,
    date_idx: HashMap<DateTime<Utc>, usize>,
    symbols: Vec<String>,
    data: Vec<Vec<f64>>,
}

impl AmountPanel {
    pub fn from_triples(triples: &[(String, DateTime<Utc>, f64)]) -> Self {
        let mut symbols: Vec<String> = Vec::new();
        let mut dates_set: std::collections::BTreeSet<DateTime<Utc>> = std::collections::BTreeSet::new();
        let mut raw: HashMap<(String, DateTime<Utc>), f64> = HashMap::new();

        for (sym, dt, amt) in triples {
            if !symbols.contains(sym) {
                symbols.push(sym.clone());
            }
            dates_set.insert(*dt);
            raw.insert((sym.clone(), *dt), *amt);
        }

        let dates_sorted: Vec<DateTime<Utc>> = dates_set.into_iter().collect();
        let symbol_idx: HashMap<String, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let date_idx: HashMap<DateTime<Utc>, usize> =
            dates_sorted.iter().enumerate().map(|(i, d)| (*d, i)).collect();

        let mut data = vec![vec![f64::NAN; dates_sorted.len()]; symbols.len()];
        for ((sym, dt), amt) in &raw {
            if let (Some(&si), Some(&di)) = (symbol_idx.get(sym), date_idx.get(dt)) {
                data[si][di] = *amt;
            }
        }

        Self {
            symbol_idx,
            date_idx,
            symbols,
            data,
        }
    }

    /// 返回某日所有股票的成交额十分位。
    fn decile_on_date(&self, date: &DateTime<Utc>) -> HashMap<usize, u8> {
        let di = match self.date_idx.get(date) {
            Some(&i) => i,
            None => return HashMap::new(),
        };

        let mut vals: Vec<(usize, f64)> = self
            .data
            .iter()
            .enumerate()
            .filter_map(|(si, row)| {
                let v = row[di];
                if v.is_nan() || v <= 0.0 { None } else { Some((si, v)) }
            })
            .collect();
        vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = vals.len();
        let mut out = HashMap::with_capacity(n);
        for (rank, (si, _)) in vals.iter().enumerate() {
            let pct = rank as f64 / n.max(1) as f64;
            let decile = (pct * 10.0).min(9.0) as u8;
            out.insert(*si, decile);
        }
        out
    }
}

/// 开盘价面板：symbol × date → open。
#[derive(Debug, Clone)]
pub struct OpenPanel {
    pub(crate) symbol_idx: HashMap<String, usize>,
    date_idx: HashMap<DateTime<Utc>, usize>,
    data: Vec<Vec<f64>>,
}

impl OpenPanel {
    pub fn from_triples(triples: &[(String, DateTime<Utc>, f64)]) -> Self {
        let mut symbols: Vec<String> = Vec::new();
        let mut dates_set: std::collections::BTreeSet<DateTime<Utc>> = std::collections::BTreeSet::new();
        let mut raw: HashMap<(String, DateTime<Utc>), f64> = HashMap::new();

        for (sym, dt, px) in triples {
            if !symbols.contains(sym) {
                symbols.push(sym.clone());
            }
            dates_set.insert(*dt);
            raw.insert((sym.clone(), *dt), *px);
        }

        let dates_sorted: Vec<DateTime<Utc>> = dates_set.into_iter().collect();
        let symbol_idx: HashMap<String, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let date_idx: HashMap<DateTime<Utc>, usize> =
            dates_sorted.iter().enumerate().map(|(i, d)| (*d, i)).collect();

        let mut data = vec![vec![f64::NAN; dates_sorted.len()]; symbols.len()];
        for ((sym, dt), px) in &raw {
            if let (Some(&si), Some(&di)) = (symbol_idx.get(sym), date_idx.get(dt)) {
                data[si][di] = *px;
            }
        }

        Self {
            symbol_idx,
            date_idx,
            data,
        }
    }

    fn get(&self, si: usize, date: &DateTime<Utc>) -> Option<f64> {
        let di = *self.date_idx.get(date)?;
        let v = self.data[si][di];
        if v.is_nan() { None } else { Some(v) }
    }
}

/// 收盘价面板的内部视图（复用 slot_backtest::ClosePanel 中的 symbol_idx）。
#[derive(Debug, Clone)]
pub struct ClosePanelView {
    pub(crate) symbol_idx: HashMap<String, usize>,
    date_idx: HashMap<DateTime<Utc>, usize>,
    data: Vec<Vec<f64>>,
}

impl ClosePanelView {
    pub fn from_triples(triples: &[(String, DateTime<Utc>, f64)]) -> Self {
        let mut symbols: Vec<String> = Vec::new();
        let mut dates_set: std::collections::BTreeSet<DateTime<Utc>> = std::collections::BTreeSet::new();
        let mut raw: HashMap<(String, DateTime<Utc>), f64> = HashMap::new();

        for (sym, dt, px) in triples {
            if !symbols.contains(sym) {
                symbols.push(sym.clone());
            }
            dates_set.insert(*dt);
            raw.insert((sym.clone(), *dt), *px);
        }

        let dates_sorted: Vec<DateTime<Utc>> = dates_set.into_iter().collect();
        let symbol_idx: HashMap<String, usize> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let date_idx: HashMap<DateTime<Utc>, usize> =
            dates_sorted.iter().enumerate().map(|(i, d)| (*d, i)).collect();

        let mut data = vec![vec![f64::NAN; dates_sorted.len()]; symbols.len()];
        for ((sym, dt), px) in &raw {
            if let (Some(&si), Some(&di)) = (symbol_idx.get(sym), date_idx.get(dt)) {
                data[si][di] = *px;
            }
        }

        Self {
            symbol_idx,
            date_idx,
            data,
        }
    }

    fn get(&self, si: usize, date: &DateTime<Utc>) -> Option<f64> {
        let di = *self.date_idx.get(date)?;
        let v = self.data[si][di];
        if v.is_nan() { None } else { Some(v) }
    }
}

/// 单笔超额记录。
#[derive(Debug, Clone)]
pub struct ExcessRecord {
    pub symbol: String,
    pub entry_dt: DateTime<Utc>,
    pub ret_gross_pct: f64,
    pub control_median_pct: f64,
    pub excess_pct: f64,
    pub control_n: usize,
    pub seg: String,
    pub year: i32,
}

/// 超额汇总统计（按段）。
#[derive(Debug, Clone)]
pub struct ExcessSegment {
    pub label: String,
    pub n: usize,
    pub excess_mean_pct: f64,
    pub excess_median_pct: f64,
    pub t_stat: f64,
    pub positive_rate_pct: f64,
}

/// 批量计算所有 trades 的随机对照超额。
pub fn compute_excess(
    trades: &[TradeRecord],
    open_panel: &OpenPanel,
    close_panel: &ClosePanelView,
    amount_panel: &AmountPanel,
    config: &ControlConfig,
) -> Vec<ExcessRecord> {
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut results = Vec::with_capacity(trades.len());

    for tr in trades {
        let deciles = amount_panel.decile_on_date(&tr.entry_dt);
        let my_si = match amount_panel.symbol_idx.get(&tr.symbol) {
            Some(&si) => si,
            None => {
                results.push(ExcessRecord {
                    symbol: tr.symbol.clone(),
                    entry_dt: tr.entry_dt,
                    ret_gross_pct: tr.ret_gross_pct,
                    control_median_pct: f64::NAN,
                    excess_pct: f64::NAN,
                    control_n: 0,
                    seg: tr.seg.clone(),
                    year: tr.year,
                });
                continue;
            }
        };
        let my_decile = match deciles.get(&my_si) {
            Some(&d) => d,
            None => {
                results.push(ExcessRecord {
                    symbol: tr.symbol.clone(),
                    entry_dt: tr.entry_dt,
                    ret_gross_pct: tr.ret_gross_pct,
                    control_median_pct: f64::NAN,
                    excess_pct: f64::NAN,
                    control_n: 0,
                    seg: tr.seg.clone(),
                    year: tr.year,
                });
                continue;
            }
        };

        // 同 decile 且非自身（收集 symbol index in amount_panel）
        let pool: Vec<usize> = deciles
            .iter()
            .filter(|(si, dec)| **dec == my_decile && **si != my_si)
            .map(|(si, _)| *si)
            .collect();

        if pool.len() < config.min_valid {
            results.push(ExcessRecord {
                symbol: tr.symbol.clone(),
                entry_dt: tr.entry_dt,
                ret_gross_pct: tr.ret_gross_pct,
                control_median_pct: f64::NAN,
                excess_pct: f64::NAN,
                control_n: pool.len(),
                seg: tr.seg.clone(),
                year: tr.year,
            });
            continue;
        }

        let pick_n = config.k.min(pool.len());
        let mut pick = pool.clone();
        pick.shuffle(&mut rng);
        pick.truncate(pick_n);

        // 从 amount_panel 的 si 映射回 symbol name，再在 open/close panel 中查找
        let mut control_rets: Vec<f64> = Vec::new();
        for &si in &pick {
            let sym = &amount_panel.symbols[si];
            let o_si = open_panel.symbol_idx.get(sym).copied();
            let c_si = close_panel.symbol_idx.get(sym).copied();
            let o = o_si.and_then(|idx| open_panel.get(idx, &tr.entry_dt));
            let c = c_si.and_then(|idx| close_panel.get(idx, &tr.exit_dt));
            if let (Some(open), Some(close)) = (o, c) {
                if open > 0.0 {
                    control_rets.push((close / open - 1.0) * 100.0);
                }
            }
        }

        if control_rets.len() < config.min_valid {
            results.push(ExcessRecord {
                symbol: tr.symbol.clone(),
                entry_dt: tr.entry_dt,
                ret_gross_pct: tr.ret_gross_pct,
                control_median_pct: f64::NAN,
                excess_pct: f64::NAN,
                control_n: control_rets.len(),
                seg: tr.seg.clone(),
                year: tr.year,
            });
            continue;
        }

        control_rets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if control_rets.len() % 2 == 1 {
            control_rets[control_rets.len() / 2]
        } else {
            (control_rets[control_rets.len() / 2 - 1] + control_rets[control_rets.len() / 2]) / 2.0
        };

        results.push(ExcessRecord {
            symbol: tr.symbol.clone(),
            entry_dt: tr.entry_dt,
            ret_gross_pct: tr.ret_gross_pct,
            control_median_pct: median,
            excess_pct: tr.ret_gross_pct - median,
            control_n: control_rets.len(),
            seg: tr.seg.clone(),
            year: tr.year,
        });
    }

    results
}

/// 分段汇总超额统计。
pub fn summarize_excess(records: &[ExcessRecord], min_n: usize) -> Vec<ExcessSegment> {
    let mut out = Vec::new();
    let all_valid: Vec<&ExcessRecord> = records.iter().filter(|r| !r.excess_pct.is_nan()).collect();

    let groups: Vec<(&str, Vec<&ExcessRecord>)> = vec![
        ("ALL", all_valid.clone()),
        (
            "IS(≤2023)",
            all_valid.iter().filter(|r| r.seg == "train").copied().collect(),
        ),
        (
            "OOS(≥2024)",
            all_valid.iter().filter(|r| r.seg == "test").copied().collect(),
        ),
    ];

    let mut year_groups: std::collections::BTreeMap<i32, Vec<&ExcessRecord>> = std::collections::BTreeMap::new();
    for r in &all_valid {
        year_groups.entry(r.year).or_default().push(r);
    }

    for (label, group) in groups {
        if group.len() >= min_n {
            out.push(make_segment(label, &group));
        }
    }

    for (year, group) in &year_groups {
        if group.len() >= min_n {
            out.push(make_segment(&year.to_string(), group));
        }
    }

    out
}

fn make_segment(label: &str, group: &[&ExcessRecord]) -> ExcessSegment {
    let n = group.len();
    let vals: Vec<f64> = group.iter().map(|r| r.excess_pct).collect();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let se = (var / n as f64).sqrt();
    let t = if se > 0.0 { mean / se } else { f64::NAN };

    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    let positive_rate = vals.iter().filter(|v| **v > 0.0).count() as f64 / n as f64 * 100.0;

    ExcessSegment {
        label: label.into(),
        n,
        excess_mean_pct: round2(mean, 2),
        excess_median_pct: round2(median, 2),
        t_stat: round2(t, 2),
        positive_rate_pct: round2(positive_rate, 1),
    }
}

fn round2(v: f64, decimals: u32) -> f64 {
    if v.is_nan() || v.is_infinite() {
        return v;
    }
    let factor = 10.0_f64.powi(decimals as i32);
    (v * factor).round() / factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use czsc_trend_regime::GateLevel;

    fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn make_trade(sym: &str, entry: DateTime<Utc>, exit: DateTime<Utc>, ret: f64) -> TradeRecord {
        TradeRecord {
            symbol: sym.into(),
            entry_dt: entry,
            exit_dt: exit,
            entry_price: 10.0,
            exit_price: 10.0 * (1.0 + ret / 100.0),
            ret_gross_pct: ret,
            ret_net_pct: ret - 0.4,
            gate_level: GateLevel::Full,
            gate_confidence: 1.0,
            priority: 80.0,
            hold_days: 3,
            seg: "train".into(),
            year: 2023,
            position_weight: 1.0,
            entry_regime: 7,
        }
    }

    #[test]
    fn test_excess_computation_with_sufficient_pool() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);

        let trades = vec![make_trade("A", d1, d3, 5.0)];

        // 100 pool symbols ensures each decile has ~10 entries (> min_valid=5)
        let mut open_triples = vec![("A".into(), d1, 10.0)];
        let mut close_triples = vec![("A".into(), d3, 10.5)];
        let mut amount_triples = vec![("A".into(), d1, 50.0)];

        for i in 1..=100 {
            let sym = format!("S{i}");
            open_triples.push((sym.clone(), d1, 10.0));
            close_triples.push((sym.clone(), d3, 10.0 + i as f64 * 0.05));
            amount_triples.push((sym.clone(), d1, i as f64));
        }

        let open_panel = OpenPanel::from_triples(&open_triples);
        let close_panel = ClosePanelView::from_triples(&close_triples);
        let amount_panel = AmountPanel::from_triples(&amount_triples);
        let config = ControlConfig { k: 10, min_valid: 5, seed: 42 };

        let results = compute_excess(&trades, &open_panel, &close_panel, &amount_panel, &config);
        assert_eq!(results.len(), 1);
        assert!(!results[0].excess_pct.is_nan(), "should have valid excess");
        assert!(results[0].control_n >= config.min_valid);
    }

    #[test]
    fn test_excess_nan_for_insufficient_pool() {
        let d1 = utc(2023, 1, 2);
        let d3 = utc(2023, 1, 4);

        let trades = vec![make_trade("A", d1, d3, 5.0)];
        let open_triples = vec![("A".into(), d1, 10.0)];
        let close_triples = vec![("A".into(), d3, 10.5)];
        let amount_triples = vec![("A".into(), d1, 1.0)];

        let open_panel = OpenPanel::from_triples(&open_triples);
        let close_panel = ClosePanelView::from_triples(&close_triples);
        let amount_panel = AmountPanel::from_triples(&amount_triples);
        let config = ControlConfig { k: 50, min_valid: 10, seed: 42 };

        let results = compute_excess(&trades, &open_panel, &close_panel, &amount_panel, &config);
        assert_eq!(results.len(), 1);
        assert!(results[0].excess_pct.is_nan(), "should be NaN without sufficient pool");
    }

    #[test]
    fn test_summarize_excess() {
        let records: Vec<ExcessRecord> = (0..50)
            .map(|i| ExcessRecord {
                symbol: format!("S{i}"),
                entry_dt: utc(2023, 1, 2),
                ret_gross_pct: 5.0,
                control_median_pct: 3.0,
                excess_pct: 2.0 + (i as f64 * 0.1 - 2.5),
                control_n: 50,
                seg: "train".into(),
                year: 2023,
            })
            .collect();

        let segments = summarize_excess(&records, 10);
        assert!(!segments.is_empty());
        let all = segments.iter().find(|s| s.label == "ALL").unwrap();
        assert_eq!(all.n, 50);
    }
}
