//! Parallel state projection + prefix audit.
//!
//! Ports the pure computation from `scripts/xs_chan_state_cache.py` into Rust.
//! CLI / manifest JSON ceremony remains in Python.

use chrono::{DateTime, NaiveDate, Utc};
use czsc_core::analyze::utils::format_standard_kline;
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::freq::Freq;
use czsc_trend_regime::{iter_states, MIN_BIS, WARMUP_BARS as TREND_WARMUP_BARS};
use polars::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const WARMUP_BARS: usize = 120;
pub const SCHEMA_VERSION: &str = "xs_chan_state_cache_v1";
const STATE_COLUMNS: [&str; 3] = ["symbol", "dt", "regime"];
const REQUIRED_SOURCE_COLUMNS: [&str; 9] = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "pct_chg",
];
const STRICT_NON_NULL_SOURCE_COLUMNS: [&str; 8] = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
];
const NUMERIC_SOURCE_COLUMNS: [&str; 7] =
    ["open", "high", "low", "close", "vol", "amount", "pct_chg"];
const LIMIT_PCT_MAIN: f64 = 9.8;
const LIMIT_PCT_CHINEXT: f64 = 19.8;
const CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Error, Debug)]
pub enum StateCacheError {
    #[error("input error: {0}")]
    Input(String),
    #[error("projection error: {0}")]
    Projection(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
}

pub type Result<T> = std::result::Result<T, StateCacheError>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateCacheManifest {
    pub schema_version: String,
    pub source_count: usize,
    pub total_rows: usize,
    pub regime_counts: HashMap<u8, usize>,
    pub source_hashes: Vec<SourceHash>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceHash {
    pub path: String,
    pub sha256: String,
    pub symbol: String,
    pub rows: usize,
}

#[derive(Debug, Clone)]
pub struct AuditResult {
    pub passed: bool,
    pub audited_symbols: usize,
    pub comparisons: usize,
    pub mismatches: HashMap<String, usize>,
}

/// Return the limit-up/limit-down threshold for a symbol (ChiNext 20cm vs main 10cm).
pub fn limit_pct_for(code: &str) -> f64 {
    if code.starts_with("300") || code.starts_with("301") || code.starts_with("302") {
        LIMIT_PCT_CHINEXT
    } else {
        LIMIT_PCT_MAIN
    }
}

/// Read a source parquet and enforce the frozen input contract.
pub fn load_source_parquet(path: &Path) -> Result<DataFrame> {
    if !path.is_file() {
        return Err(StateCacheError::Input(format!("source is not a file: {}", path.display())));
    }

    let mut frame = ParquetReader::new(File::open(path)?)
        .finish()
        .map_err(|err| StateCacheError::Input(format!("cannot read parquet data: {}: {err}", path.display())))?;
    let available: Vec<&str> = frame.get_column_names().iter().map(|name| name.as_str()).collect();
    let missing: Vec<_> = REQUIRED_SOURCE_COLUMNS
        .iter()
        .filter(|col| !available.contains(col))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(StateCacheError::Input(format!(
            "{}: missing required columns {missing:?}",
            path.display()
        )));
    }
    frame = frame.select(REQUIRED_SOURCE_COLUMNS)?;

    if frame.is_empty() {
        return Err(StateCacheError::Input(format!("{}: empty source", path.display())));
    }

    for column in STRICT_NON_NULL_SOURCE_COLUMNS {
        if frame.column(column)?.null_count() > 0 {
            return Err(StateCacheError::Input(format!(
                "{}: null required values in {column}",
                path.display()
            )));
        }
    }

    let ts_codes = frame.column("ts_code")?.str()?;
    let unique: std::collections::HashSet<_> = ts_codes.into_iter().flatten().collect();
    if unique.len() != 1 {
        return Err(StateCacheError::Input(format!(
            "{}: expected exactly one non-empty ts_code, got {unique:?}",
            path.display()
        )));
    }
    let symbol = unique.into_iter().next().unwrap().to_string();
    if symbol.is_empty() {
        return Err(StateCacheError::Input(format!(
            "{}: expected exactly one non-empty ts_code, got empty",
            path.display()
        )));
    }

    let parsed_dates = parse_trade_dates(frame.column("trade_date")?, path)?;
    frame.with_column(parsed_dates).map_err(StateCacheError::Polars)?;

    let dt_col = frame.column("trade_date")?.datetime()?;
    let mut seen = std::collections::HashSet::new();
    for i in 0..dt_col.len() {
        let ts = dt_col.phys.get(i).ok_or_else(|| {
            StateCacheError::Input(format!("{}: null trade_date values", path.display()))
        })?;
        if !seen.insert(ts) {
            return Err(StateCacheError::Input(format!(
                "{}: duplicate trade_date values",
                path.display()
            )));
        }
    }

    frame = frame.sort(["trade_date"], SortMultipleOptions::default())?;

    let pct_nulls = frame.column("pct_chg")?.null_count();
    if pct_nulls > 0 {
        let null_rows: Vec<usize> = frame
            .column("pct_chg")?
            .is_null()
            .into_iter()
            .enumerate()
            .filter_map(|(idx, is_null)| is_null.unwrap_or(false).then_some(idx))
            .collect();
        if null_rows != [0] {
            return Err(StateCacheError::Input(format!(
                "{}: pct_chg may be null only on the earliest observed bar, got rows {:?}",
                path.display(),
                &null_rows[..null_rows.len().min(5)]
            )));
        }
        let pct = frame.column("pct_chg")?.f64()?;
        let mut values: Vec<Option<f64>> = pct.into_iter().collect();
        values[0] = Some(0.0);
        let filled = Series::new("pct_chg".into(), values);
        frame.with_column(filled).map_err(StateCacheError::Polars)?;
    }

    for column in NUMERIC_SOURCE_COLUMNS {
        validate_numeric_series(frame.column(column)?, path, column)?;
    }

    validate_positive_ohlc(&frame, path)?;
    validate_non_negative_vol_amount(&frame, path)?;

    frame = frame
        .lazy()
        .with_columns([
            lit(symbol).alias("symbol"),
            col("trade_date").alias("dt"),
        ])
        .select([
            col("symbol"),
            col("dt"),
            col("open"),
            col("high"),
            col("low"),
            col("close"),
            col("vol"),
            col("amount"),
            col("pct_chg"),
        ])
        .collect()?;
    frame = frame.sort(["dt"], SortMultipleOptions::default())?;
    Ok(frame)
}

/// Project causal 0..=10 regimes for one symbol from already-converted bars.
pub fn project_single_stock(bars: &[RawBar], warmup_bars: usize) -> Result<DataFrame> {
    if bars.is_empty() {
        return Err(StateCacheError::Projection("empty bars".into()));
    }
    if warmup_bars != WARMUP_BARS || warmup_bars != TREND_WARMUP_BARS {
        return Err(StateCacheError::Projection(format!(
            "warmup_bars is frozen at {WARMUP_BARS}"
        )));
    }

    let symbol = bars[0].symbol.to_string();
    let n = bars.len();
    let warmup_rows = n.min(warmup_bars);

    let mut symbols = Vec::with_capacity(n);
    let mut dts = Vec::with_capacity(n);
    let mut regimes = Vec::with_capacity(n);

    for bar in &bars[..warmup_rows] {
        symbols.push(symbol.clone());
        dts.push(bar.dt);
        regimes.push(0i8);
    }

    if n > warmup_bars {
        let limit_pct = limit_pct_for(&symbol);
        let snapshots = iter_states(bars, Freq::D, false, None, limit_pct, warmup_bars, MIN_BIS);
        for snap in snapshots {
            symbols.push(symbol.clone());
            dts.push(snap.dt);
            regimes.push(i8::try_from(snap.regime).map_err(|_| {
                StateCacheError::Projection(format!("illegal regime value: {}", snap.regime))
            })?);
        }
    }

    if symbols.len() != n {
        return Err(StateCacheError::Projection(format!(
            "{symbol}: expected {n} state rows, got {}",
            symbols.len()
        )));
    }

    for (idx, regime) in regimes.iter().enumerate() {
        if !valid_regime(*regime) {
            return Err(StateCacheError::Projection(format!(
                "{symbol}: illegal regime value {} at row {idx}",
                *regime
            )));
        }
        if bars[idx].dt != dts[idx] {
            return Err(StateCacheError::Projection(format!(
                "{symbol}: state dates do not exactly cover every observed source bar"
            )));
        }
    }

    build_projection_frame(&symbols, &dts, &regimes)
}

/// Build the merged projection and manifest from a directory of source parquets.
pub fn build_state_cache_parallel(
    raw_dir: &Path,
    warmup_bars: usize,
    workers: Option<usize>,
) -> Result<(DataFrame, StateCacheManifest)> {
    if !raw_dir.is_dir() {
        return Err(StateCacheError::Input(format!(
            "raw_dir is not a directory: {}",
            raw_dir.display()
        )));
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(raw_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(StateCacheError::Input(format!(
            "no parquet sources found in {}",
            raw_dir.display()
        )));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.unwrap_or_else(rayon::current_num_threads))
        .stack_size(64 * 1024 * 1024)
        .build()
        .map_err(|err| StateCacheError::Input(format!("failed to build rayon pool: {err}")))?;

    let results: Result<Vec<_>> = pool.install(|| {
        paths
            .par_iter()
            .map(|path| process_source_file(path, warmup_bars))
            .collect()
    });
    let results = results?;

    let mut seen_symbols = std::collections::HashSet::new();
    for item in &results {
        if !seen_symbols.insert(item.symbol.clone()) {
            return Err(StateCacheError::Input(format!(
                "multiple source files resolve to the same symbol: {}",
                item.symbol
            )));
        }
    }

    let mut regime_counts = HashMap::new();
    for regime in 0..=10u8 {
        regime_counts.insert(regime, 0);
    }

    let mut parts = Vec::with_capacity(results.len());
    let mut source_hashes = Vec::with_capacity(results.len());
    let mut total_rows = 0usize;

    for item in results {
        total_rows += item.rows;
        for (regime, count) in item.regime_counts {
            *regime_counts.entry(regime).or_insert(0) += count;
        }
        source_hashes.push(SourceHash {
            path: item.path,
            sha256: item.sha256,
            symbol: item.symbol,
            rows: item.rows,
        });
        parts.push(item.projection);
    }

    let merged = concat_projection_parts(parts)?;
    let manifest = StateCacheManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        source_count: source_hashes.len(),
        total_rows,
        regime_counts,
        source_hashes,
    };
    Ok((merged, manifest))
}

/// Compute SHA256 hex digest of a file, reading in 1 MiB chunks.
pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; CHUNK_SIZE];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Prefix/full causality audit for one source frame against its full projection.
pub fn prefix_audit(source: &DataFrame, full_projection: &DataFrame, checkpoints: usize) -> Result<AuditResult> {
    if checkpoints == 0 {
        return Err(StateCacheError::Input("checkpoints must be positive".into()));
    }
    if source.height() <= WARMUP_BARS {
        return Err(StateCacheError::Input(format!(
            "prefix audit requires more than {WARMUP_BARS} rows"
        )));
    }

    let full = validate_projection_frame(full_projection, false)?;
    let symbol = source_symbol(source)?;
    let full_symbol = source_symbol(&full)?;
    if symbol != full_symbol {
        return Err(StateCacheError::Input(format!(
            "source symbol {symbol} does not match projection symbol {full_symbol}"
        )));
    }

    let checkpoint_rows = checkpoint_lengths(source.height(), checkpoints, WARMUP_BARS)?;
    let mut mismatches = HashMap::from([
        ("symbol".to_string(), 0usize),
        ("dt".to_string(), 0usize),
        ("regime".to_string(), 0usize),
    ]);
    let mut passed = true;

    for prefix_rows in checkpoint_rows {
        let prefix = source.head(Some(prefix_rows));
        let actual = generate_projection_from_frame(&prefix, WARMUP_BARS)?;
        let cutoff = prefix_dt_at(&prefix, prefix_rows - 1)?;
        let expected = filter_projection_up_to(&full, cutoff)?;
        let field_mismatches = field_mismatches(&expected, &actual);
        if field_mismatches.values().any(|count| *count > 0) {
            passed = false;
        }
        for (field, count) in field_mismatches {
            *mismatches.entry(field).or_insert(0) += count;
        }
    }

    Ok(AuditResult {
        passed,
        audited_symbols: 1,
        comparisons: checkpoints,
        mismatches,
    })
}

struct ProcessedSource {
    path: String,
    sha256: String,
    symbol: String,
    rows: usize,
    regime_counts: HashMap<u8, usize>,
    projection: DataFrame,
}

fn process_source_file(path: &Path, warmup_bars: usize) -> Result<ProcessedSource> {
    let sha256 = sha256_file(path)?;
    let frame = load_source_parquet(path)?;
    let symbol = source_symbol(&frame)?;
    let rows = frame.height();
    let bars = bars_from_source_frame(&frame)?;
    let projection = project_single_stock(&bars, warmup_bars)?;
    validate_projection_frame(&projection, false)?;
    let regime_counts = regime_counts_from_projection(&projection);
    Ok(ProcessedSource {
        path: path.display().to_string(),
        sha256,
        symbol,
        rows,
        regime_counts,
        projection,
    })
}

fn generate_projection_from_frame(frame: &DataFrame, warmup_bars: usize) -> Result<DataFrame> {
    if frame.is_empty() {
        return build_projection_frame(&[], &[], &[]);
    }
    let bars = bars_from_source_frame(frame)?;
    project_single_stock(&bars, warmup_bars)
}

pub fn bars_from_source_frame(frame: &DataFrame) -> Result<Vec<RawBar>> {
    format_standard_kline(frame.clone(), Freq::D).map_err(|err| {
        StateCacheError::Projection(format!("failed to convert frame to RawBar: {err}"))
    })
}

fn build_projection_frame(symbols: &[String], dts: &[DateTime<Utc>], regimes: &[i8]) -> Result<DataFrame> {
    let dt_ns: Vec<i64> = dts
        .iter()
        .map(|dt| dt.timestamp_nanos_opt().unwrap_or(0))
        .collect();
    let dt_series = Series::new("dt".into(), dt_ns).cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;
    let frame = DataFrame::new(vec![
        Series::new("symbol".into(), symbols.to_vec()).into(),
        dt_series.into(),
        Series::new("regime".into(), regimes.to_vec()).into(),
    ])?;
    validate_projection_frame(&frame, symbols.is_empty())
}

fn validate_projection_frame(frame: &DataFrame, allow_empty: bool) -> Result<DataFrame> {
    if frame.width() != STATE_COLUMNS.len()
        || !STATE_COLUMNS
            .iter()
            .enumerate()
            .all(|(idx, name)| frame.get_column_names()[idx].as_str() == *name)
    {
        return Err(StateCacheError::Projection(format!(
            "projection columns must be exactly {STATE_COLUMNS:?}, got {:?}",
            frame.get_column_names()
        )));
    }
    if frame.is_empty() {
        if !allow_empty {
            return Err(StateCacheError::Projection("projection is empty".into()));
        }
        return build_projection_frame(&[], &[], &[]);
    }
    if frame.get_columns().iter().any(|column| column.null_count() > 0) {
        return Err(StateCacheError::Projection("projection contains null values".into()));
    }

    let symbols = frame.column("symbol")?.str()?;
    if symbols.into_iter().flatten().any(str::is_empty) {
        return Err(StateCacheError::Projection("projection contains an empty symbol".into()));
    }

    let regimes = frame.column("regime")?.i8()?;
    for i in 0..regimes.len() {
        let regime = regimes.get(i).ok_or_else(|| {
            StateCacheError::Projection("projection contains an invalid regime".into())
        })?;
        if !valid_regime(regime) {
            return Err(StateCacheError::Projection(format!(
                "illegal regime values: [{regime}]"
            )));
        }
    }

    let mut seen = std::collections::HashSet::new();
    let dt_col = frame.column("dt")?.datetime()?;
    for i in 0..frame.height() {
        let dt = dt_col.phys.get(i).ok_or_else(|| {
            StateCacheError::Projection("projection contains an invalid dt".into())
        })?;
        let symbol = symbols.get(i).ok_or_else(|| {
            StateCacheError::Projection("projection contains an empty symbol".into())
        })?;
        if !seen.insert((symbol.to_string(), dt)) {
            return Err(StateCacheError::Projection(
                "projection contains duplicate symbol/dt rows".into(),
            ));
        }
    }

    let sorted = frame.clone().sort(["symbol", "dt"], SortMultipleOptions::default())?;
    Ok(sorted)
}

fn concat_projection_parts(parts: Vec<DataFrame>) -> Result<DataFrame> {
    if parts.is_empty() {
        return Err(StateCacheError::Projection("no source projection was generated".into()));
    }
    let mut merged = parts[0].clone();
    for part in parts.into_iter().skip(1) {
        merged = merged.vstack(&part).map_err(StateCacheError::Polars)?;
    }
    merged.sort(["symbol", "dt"], SortMultipleOptions::default())
        .map_err(StateCacheError::Polars)
}

fn regime_counts_from_projection(frame: &DataFrame) -> HashMap<u8, usize> {
    let mut counts = HashMap::new();
    for regime in 0..=10u8 {
        counts.insert(regime, 0);
    }
    if frame.is_empty() {
        return counts;
    }
    if let Ok(regimes) = frame.column("regime").and_then(|col| col.i8()) {
        for i in 0..regimes.len() {
            if let Some(regime) = regimes.get(i) {
                *counts.entry(regime as u8).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn checkpoint_lengths(row_count: usize, checkpoint_count: usize, warmup_bars: usize) -> Result<Vec<usize>> {
    let available = row_count.saturating_sub(warmup_bars);
    if available < checkpoint_count {
        return Err(StateCacheError::Input(format!(
            "{row_count} rows provide only {available} unique post-warmup checkpoints; need {checkpoint_count}"
        )));
    }
    let start = warmup_bars + 1;
    let stop = row_count;
    let mut checkpoints = Vec::with_capacity(checkpoint_count);
    if checkpoint_count == 1 {
        checkpoints.push(start);
    } else {
        for i in 0..checkpoint_count {
            let t = i as f64 / (checkpoint_count - 1) as f64;
            let value = start as f64 + t * (stop - start) as f64;
            checkpoints.push(value.round() as usize);
        }
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();
    if checkpoints.len() != checkpoint_count {
        return Err(StateCacheError::Input(
            "checkpoint construction did not produce the configured count".into(),
        ));
    }
    Ok(checkpoints)
}

fn field_mismatches(expected: &DataFrame, actual: &DataFrame) -> HashMap<String, usize> {
    let left = expected;
    let right = actual;
    let common = left.height().min(right.height());
    let length_gap = left.height().abs_diff(right.height());
    let mut mismatches = HashMap::new();

    for column in STATE_COLUMNS {
        let mut count = length_gap;
        if common > 0 {
            count += match column {
                "symbol" => compare_str_columns(left.column(column).ok(), right.column(column).ok(), common),
                "dt" => compare_dt_columns(left.column(column).ok(), right.column(column).ok(), common),
                "regime" => compare_i8_columns(left.column(column).ok(), right.column(column).ok(), common),
                _ => 0,
            };
        }
        mismatches.insert(column.to_string(), count);
    }
    mismatches
}

fn compare_str_columns(left: Option<&Column>, right: Option<&Column>, common: usize) -> usize {
    match (left.and_then(|s| s.str().ok()), right.and_then(|s| s.str().ok())) {
        (Some(l), Some(r)) => (0..common)
            .filter(|idx| l.get(*idx) != r.get(*idx))
            .count(),
        _ => common,
    }
}

fn compare_dt_columns(left: Option<&Column>, right: Option<&Column>, common: usize) -> usize {
    match (left.and_then(|s| s.datetime().ok()), right.and_then(|s| s.datetime().ok())) {
        (Some(l), Some(r)) => (0..common)
            .filter(|idx| l.phys.get(*idx) != r.phys.get(*idx))
            .count(),
        _ => common,
    }
}

fn compare_i8_columns(left: Option<&Column>, right: Option<&Column>, common: usize) -> usize {
    match (left.and_then(|s| s.i8().ok()), right.and_then(|s| s.i8().ok())) {
        (Some(l), Some(r)) => (0..common).filter(|idx| l.get(*idx) != r.get(*idx)).count(),
        _ => common,
    }
}

fn filter_projection_up_to(frame: &DataFrame, cutoff: i64) -> Result<DataFrame> {
    let filtered = frame
        .clone()
        .lazy()
        .filter(col("dt").lt_eq(lit(cutoff)))
        .collect()?;
    validate_projection_frame(&filtered, filtered.is_empty())
}

fn prefix_dt_at(frame: &DataFrame, row_idx: usize) -> Result<i64> {
    frame
        .column("dt")?
        .datetime()?
        .phys
        .get(row_idx)
        .ok_or_else(|| StateCacheError::Projection("prefix cutoff dt is null".into()))
}

fn source_symbol(frame: &DataFrame) -> Result<String> {
    let symbols = frame.column("symbol")?.str()?;
    let unique: std::collections::HashSet<_> = symbols.into_iter().flatten().collect();
    if unique.len() != 1 {
        return Err(StateCacheError::Input(format!(
            "expected exactly one symbol, got {unique:?}"
        )));
    }
    Ok(unique.into_iter().next().unwrap().to_string())
}

fn parse_trade_dates(column: &Column, path: &Path) -> Result<Series> {
    if let Ok(dt) = column.datetime() {
        if dt.null_count() > 0 {
            return Err(StateCacheError::Input(format!(
                "{}: null trade_date values",
                path.display()
            )));
        }
        return Ok(column.as_materialized_series().clone().with_name("trade_date".into()));
    }

    let str_values = column.cast(&DataType::String)?.str()?.clone();
    let texts: Vec<String> = str_values
        .into_iter()
        .map(|value| value.unwrap_or("").trim().to_string())
        .collect();
    let all_yyyymmdd = texts
        .iter()
        .all(|text| text.len() == 8 && text.chars().all(|ch| ch.is_ascii_digit()));

    if all_yyyymmdd {
        let mut parsed_ns = Vec::with_capacity(texts.len());
        for text in texts {
            let date = NaiveDate::parse_from_str(&text, "%Y%m%d").map_err(|_| {
                StateCacheError::Input(format!("{}: invalid trade_date values", path.display()))
            })?;
            let dt = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            parsed_ns.push(dt.timestamp_nanos_opt().unwrap_or(0));
        }
        return Ok(Series::new("trade_date".into(), parsed_ns).cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?);
    }

    column
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .map(|s| s.as_materialized_series().clone().with_name("trade_date".into()))
        .map_err(|_| StateCacheError::Input(format!("{}: invalid trade_date values", path.display())))
}

fn validate_numeric_series(column: &Column, path: &Path, column_name: &str) -> Result<()> {
    let casted = column.cast(&DataType::Float64)?;
    let values = casted.f64()?;
    for i in 0..values.len() {
        let value = values.get(i).ok_or_else(|| {
            StateCacheError::Input(format!("{}: non-numeric {column_name}", path.display()))
        })?;
        if !value.is_finite() {
            return Err(StateCacheError::Input(format!(
                "{}: non-finite {column_name}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_positive_ohlc(frame: &DataFrame, path: &Path) -> Result<()> {
    for column in ["open", "high", "low", "close"] {
        let values = frame.column(column)?.f64()?;
        if values.into_iter().flatten().any(|value| value <= 0.0) {
            return Err(StateCacheError::Input(format!(
                "{}: non-positive OHLC value",
                path.display()
            )));
        }
    }
    Ok(())
}

fn validate_non_negative_vol_amount(frame: &DataFrame, path: &Path) -> Result<()> {
    for column in ["vol", "amount"] {
        let values = frame.column(column)?.f64()?;
        if values.into_iter().flatten().any(|value| value < 0.0) {
            return Err(StateCacheError::Input(format!(
                "{}: negative volume or amount",
                path.display()
            )));
        }
    }
    Ok(())
}

const fn valid_regime(regime: i8) -> bool {
    regime >= 0 && regime <= 10
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, NaiveDate};
    use czsc_core::objects::bar::RawBarBuilder;
    use std::sync::Arc;

    fn write_test_parquet(path: &Path, mut frame: DataFrame) {
        let mut file = File::create(path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut frame).unwrap();
    }

    fn sample_source_frame(rows: usize) -> DataFrame {
        let start = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let mut dates = Vec::with_capacity(rows);
        let mut opens = Vec::with_capacity(rows);
        for i in 0..rows {
            dates.push((start + Duration::days(i as i64)).format("%Y%m%d").to_string());
            opens.push(10.0 + i as f64 * 0.1);
        }
        df! {
            "ts_code" => std::iter::repeat("000001.SZ").take(rows).collect::<Vec<_>>(),
            "trade_date" => dates,
            "open" => opens.clone(),
            "high" => opens.iter().map(|v| v + 0.5).collect::<Vec<_>>(),
            "low" => opens.iter().map(|v| v - 0.5).collect::<Vec<_>>(),
            "close" => opens.iter().map(|v| v + 0.2).collect::<Vec<_>>(),
            "vol" => std::iter::repeat(1000.0_f64).take(rows).collect::<Vec<_>>(),
            "amount" => std::iter::repeat(10000.0_f64).take(rows).collect::<Vec<_>>(),
            "pct_chg" => std::iter::repeat(0.5_f64).take(rows).collect::<Vec<_>>(),
        }
        .unwrap()
    }

    fn raw_bar(symbol: &str, idx: i32, close: f64) -> RawBar {
        let start = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let date = start + Duration::days(idx as i64);
        RawBarBuilder::default()
            .symbol(Arc::<str>::from(symbol))
            .dt(date.and_hms_opt(0, 0, 0).unwrap().and_utc())
            .freq(Freq::D)
            .id(idx)
            .open(close - 0.2)
            .close(close)
            .high(close + 0.3)
            .low(close - 0.3)
            .vol(1000.0)
            .amount(10_000.0)
            .build()
            .unwrap()
    }

    #[test]
    fn load_source_parquet_renames_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.SZ.parquet");
        write_test_parquet(&path, sample_source_frame(5));
        let loaded = load_source_parquet(&path).unwrap();
        assert_eq!(
            loaded.get_column_names(),
            &["symbol", "dt", "open", "high", "low", "close", "vol", "amount", "pct_chg"]
        );
        assert_eq!(loaded.height(), 5);
    }

    #[test]
    fn project_single_stock_warmup_is_regime_zero() {
        let bars: Vec<_> = (0..130).map(|i| raw_bar("000001.SZ", i, 10.0 + i as f64)).collect();
        let projection = project_single_stock(&bars, WARMUP_BARS).unwrap();
        assert_eq!(projection.height(), 130);
        let regimes = projection.column("regime").unwrap().i8().unwrap();
        for i in 0..120 {
            assert_eq!(regimes.get(i), Some(0));
        }
    }

    #[test]
    fn prefix_audit_passes_for_generated_projection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.SZ.parquet");
        write_test_parquet(&path, sample_source_frame(WARMUP_BARS + 40));
        let frame = load_source_parquet(&path).unwrap();
        let full = generate_projection_from_frame(&frame, WARMUP_BARS).unwrap();
        let audit = prefix_audit(&frame, &full, 5).unwrap();
        assert!(audit.passed, "mismatches={:?}", audit.mismatches);
        assert_eq!(audit.comparisons, 5);
    }

    #[test]
    fn sha256_file_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        std::fs::write(&path, b"hello").unwrap();
        let first = sha256_file(&path).unwrap();
        let second = sha256_file(&path).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }
}
