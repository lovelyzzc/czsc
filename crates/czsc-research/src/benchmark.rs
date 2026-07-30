//! Equal-weight market benchmark computation.
//!
//! Computes EW-All weekly returns using the same 5-session open-to-open
//! return window as the xs_chan strategy.

use chrono::{Datelike, NaiveDate};
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("benchmark error: {0}")]
    Computation(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
}

pub type Result<T> = std::result::Result<T, BenchmarkError>;

pub const HOLDING_SESSIONS: i32 = 5;
pub const MIN_TRADABLE_STOCKS: usize = 200;
pub const MIN_OPEN_PRICE: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct WeeklyBenchmark {
    pub decision_dt: String,
    pub ew_all_return: f64,
    pub ew_all_median_return: f64,
    pub ew_all_std: f64,
    pub tradable_count: usize,
    pub observable_exit_count: usize,
    pub observable_exit_rate: f64,
}

/// Load all A-stock daily data from parquet files into a single panel.
pub fn load_daily_panel(raw_dir: &Path) -> Result<DataFrame> {
    let pattern = raw_dir.join("*.parquet");
    let pattern_str = pattern.to_string_lossy();
    let lf = LazyFrame::scan_parquet(PlPath::new(pattern_str.as_ref()), ScanArgsParquet::default())?;
    Ok(lf
        .select([
            col("ts_code").alias("symbol"),
            col("trade_date").cast(DataType::String).alias("dt_str"),
            col("open").cast(DataType::Float64),
            col("close").cast(DataType::Float64),
            col("amount").cast(DataType::Float64),
        ])
        .collect()?)
}

/// Build market session calendar from daily data.
pub fn build_calendar(panel: &DataFrame) -> Result<Vec<String>> {
    let dt_col = panel.column("dt_str")?;
    let dates: Vec<String> = dt_col.str()?.into_no_null_iter().map(|s| s.to_string()).collect();
    let mut unique_dates: Vec<String> = dates.into_iter().collect::<HashSet<_>>().into_iter().collect();
    unique_dates.sort();
    Ok(unique_dates)
}

/// Build W-FRI decision dates from calendar.
pub fn build_decision_dates(
    calendar: &[String],
    start_date: &str,
    end_date: Option<&str>,
) -> Vec<(usize, String)> {
    let mut weeks: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();

    for (session_no, date_str) in calendar.iter().enumerate() {
        if date_str.as_str() < start_date {
            continue;
        }
        if let Some(end) = end_date {
            if date_str.as_str() > end {
                continue;
            }
        }
        if date_str.len() != 8 {
            continue;
        }
        let Ok(y) = date_str[..4].parse::<i32>() else { continue };
        let Ok(m) = date_str[4..6].parse::<u32>() else { continue };
        let Ok(d) = date_str[6..8].parse::<u32>() else { continue };
        let Some(date) = NaiveDate::from_ymd_opt(y, m, d) else { continue };

        let days_until_fri = (5 - date.weekday().num_days_from_monday() as i32 + 7) % 7;
        let week_end = date + chrono::Duration::days(days_until_fri as i64);
        let week_key = week_end.format("%Y-W%W-FRI").to_string();
        weeks.entry(week_key).or_default().push((session_no, date_str.clone()));
    }

    weeks.values().filter_map(|sessions| sessions.last().cloned()).collect()
}

/// Compute equal-weight weekly returns for all tradable stocks.
pub fn compute_ew_benchmark(
    raw_dir: &Path,
    decision_dates: &[(usize, String)],
    holding_sessions: i32,
) -> Result<Vec<WeeklyBenchmark>> {
    let panel = load_daily_panel(raw_dir)?;
    let calendar = build_calendar(&panel)?;
    let max_session = calendar.len();
    let mut results = Vec::with_capacity(decision_dates.len());

    for &(dec_session, ref dec_dt) in decision_dates {
        let entry_session = dec_session + 1;
        let exit_session = entry_session + holding_sessions as usize;
        if exit_session >= max_session {
            continue;
        }

        let entry_dt = &calendar[entry_session];
        let exit_dt = &calendar[exit_session];
        let entry_mask = panel.column("dt_str")?.str()?.equal(entry_dt.as_str());
        let exit_mask = panel.column("dt_str")?.str()?.equal(exit_dt.as_str());
        let entry_data = panel.filter(&entry_mask.into())?;
        let exit_data = panel.filter(&exit_mask.into())?;
        if entry_data.height() < MIN_TRADABLE_STOCKS {
            continue;
        }

        let entry_sym = entry_data.column("symbol")?.str()?;
        let entry_open = entry_data.column("open")?.f64()?;
        let entry_amt = entry_data.column("amount")?.f64()?;
        let exit_sym = exit_data.column("symbol")?.str()?;
        let exit_open_col = exit_data.column("open")?.f64()?;

        let mut exit_open_by_symbol = HashMap::with_capacity(exit_data.height());
        for j in 0..exit_data.height() {
            if let (Some(sym), Some(x_open)) = (exit_sym.get(j), exit_open_col.get(j)) {
                exit_open_by_symbol.insert(sym, x_open);
            }
        }

        let mut returns = Vec::new();
        let mut observable_exits = 0usize;
        for i in 0..entry_data.height() {
            let Some(sym) = entry_sym.get(i) else { continue };
            let Some(e_open) = entry_open.get(i) else { continue };
            let e_amt = entry_amt.get(i).unwrap_or(0.0);
            if e_open <= MIN_OPEN_PRICE || !e_open.is_finite() || e_amt <= 0.0 {
                continue;
            }
            if let Some(&x_open) = exit_open_by_symbol.get(sym) {
                if x_open > 0.0 && x_open.is_finite() {
                    returns.push(x_open / e_open - 1.0);
                    observable_exits += 1;
                } else {
                    returns.push(0.0);
                }
            } else {
                returns.push(0.0);
            }
        }
        if returns.len() < MIN_TRADABLE_STOCKS {
            continue;
        }

        let n = returns.len();
        let mean_ret = returns.iter().sum::<f64>() / n as f64;
        let mut sorted = returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ret = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1) as f64;

        results.push(WeeklyBenchmark {
            decision_dt: dec_dt.clone(),
            ew_all_return: mean_ret,
            ew_all_median_return: median_ret,
            ew_all_std: var.sqrt(),
            tradable_count: n,
            observable_exit_count: observable_exits,
            observable_exit_rate: observable_exits as f64 / n as f64,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::io::parquet::write::ParquetWriter;
    use std::fs::File;
    use tempfile::tempdir;

    fn sample_panel() -> DataFrame {
        df! {
            "symbol" => ["000001.SZ", "000002.SZ", "000003.SZ", "000001.SZ", "000002.SZ", "000003.SZ"],
            "dt_str" => ["20240102", "20240102", "20240102", "20240103", "20240103", "20240103"],
            "open" => [10.0_f64, 20.0, 30.0, 10.5, 21.0, 31.5],
            "close" => [10.0_f64, 20.0, 30.0, 10.5, 21.0, 31.5],
            "amount" => [1_000_000.0; 6],
        }
        .unwrap()
    }

    #[test]
    fn build_calendar_unique_sorted() {
        let calendar = build_calendar(&sample_panel()).unwrap();
        assert_eq!(calendar, vec!["20240102".to_string(), "20240103".to_string()]);
    }

    #[test]
    fn build_decision_dates_picks_last_session_of_week() {
        let calendar = vec![
            "20240102".to_string(),
            "20240103".to_string(),
            "20240104".to_string(),
            "20240105".to_string(),
        ];
        let decisions = build_decision_dates(&calendar, "20240101", None);
        assert_eq!(decisions.last().map(|(_, dt)| dt.as_str()), Some("20240105"));
    }

    #[test]
    fn load_daily_panel_reads_parquet_glob() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("000001.SZ.parquet");
        let mut frame = sample_panel()
            .lazy()
            .select([
                col("symbol").alias("ts_code"),
                col("dt_str").alias("trade_date"),
                col("open"),
                col("close"),
                col("amount"),
            ])
            .collect()
            .unwrap();
        let mut file = File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut frame).unwrap();

        let loaded = load_daily_panel(dir.path()).unwrap();
        assert_eq!(loaded.height(), 6);
        assert!(loaded.get_column_names().iter().any(|name| name.as_str() == "symbol"));
    }
}
