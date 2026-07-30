//! Factor surface: momentum, low-vol, ADV, SMA, ranking.
//!
//! Computes the decision surface for a single week's decision point:
//! adjusted prices, momentum, volatility, average daily value, SMA,
//! industry + market-cap OLS neutralization, and percentile ranking.

use polars::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("feature computation error: {0}")]
    Computation(String),
    #[error("insufficient data: {0}")]
    InsufficientData(String),
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
}

pub type Result<T> = std::result::Result<T, FeatureError>;

/// Output of one week's factor surface computation.
#[derive(Debug, Clone)]
pub struct DecisionSurface {
    /// DataFrame with columns: symbol, momentum, low_vol, adv20, sma20,
    /// momentum_neutral, low_vol_neutral, momentum_rank, low_vol_rank,
    /// composite_score, factor_rank
    pub ranked: DataFrame,
    /// Number of stocks eligible for selection
    pub eligible_count: usize,
}

/// Compute adjusted close prices with suspension forward-fill.
///
/// # Arguments
/// * `close` - Raw close prices (f64 series)
/// * `adj_factor` - Adjustment factors (f64 series)
/// * `amount` - Daily trading amount (f64 series, 0 = suspended)
pub fn compute_adj_close(close: &[f64], adj_factor: &[f64], amount: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut adj = vec![f64::NAN; n];
    let mut last_valid = f64::NAN;

    for i in 0..n {
        let raw = close[i] * adj_factor[i];
        if amount[i] > 0.0 && raw.is_finite() {
            adj[i] = raw;
            last_valid = raw;
        } else if last_valid.is_finite() {
            adj[i] = last_valid;
        }
    }
    adj
}

/// Compute log momentum: ln(P_{t-short}) - ln(P_{t-long}).
///
/// Matches Python ``log_price.shift(short) - log_price.shift(long)`` on dense
/// session indices (not calendar days). Default: short=20, long=120.
pub fn compute_momentum(adj_close: &[f64], short: usize, long: usize) -> Vec<f64> {
    let n = adj_close.len();
    let mut mom = vec![f64::NAN; n];
    for i in long..n {
        let p_short = adj_close[i - short];
        let p_long = adj_close[i - long];
        if p_short > 0.0 && p_long > 0.0 {
            mom[i] = p_short.ln() - p_long.ln();
        }
    }
    mom
}

/// Compute low-volatility factor: -std(log_returns, window).
///
/// Default window=60 sessions.
pub fn compute_low_vol(adj_close: &[f64], window: usize) -> Vec<f64> {
    let n = adj_close.len();
    let mut result = vec![f64::NAN; n];

    if n < 2 {
        return result;
    }

    let mut log_rets = vec![f64::NAN; n];
    for i in 1..n {
        if adj_close[i] > 0.0 && adj_close[i - 1] > 0.0 {
            log_rets[i] = (adj_close[i] / adj_close[i - 1]).ln();
        }
    }

    for i in window..n {
        let slice = &log_rets[i + 1 - window..=i];
        let valid: Vec<f64> = slice.iter().filter(|x| x.is_finite()).copied().collect();
        // min_periods=window, ddof=1 — matches pandas rolling(..., min_periods=window).std(ddof=1)
        if valid.len() >= window {
            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let var = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (valid.len() - 1) as f64;
            result[i] = -var.sqrt();
        }
    }
    result
}

/// Compute rolling mean (e.g., ADV20, SMA20).
pub fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    if n < window {
        return result;
    }

    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..n {
        if values[i].is_finite() {
            sum += values[i];
            count += 1;
        }
        if i >= window {
            if values[i - window].is_finite() {
                sum -= values[i - window];
                count -= 1;
            }
        }
        if i >= window - 1 && count == window {
            result[i] = sum / count as f64;
        }
    }
    result
}

/// Simple OLS neutralization: factor ~ industry_dummies + ln(mcap).
///
/// Returns residuals for each stock.
pub fn ols_neutralize(factor: &[f64], industry: &[u32], ln_mcap: &[f64]) -> Vec<f64> {
    let n = factor.len();
    if n == 0 {
        return vec![];
    }

    // Find valid observations
    let valid_mask: Vec<bool> = (0..n)
        .map(|i| factor[i].is_finite() && ln_mcap[i].is_finite())
        .collect();
    let valid_count = valid_mask.iter().filter(|&&v| v).count();

    if valid_count < 2 {
        return vec![f64::NAN; n];
    }

    // Collect unique industries
    let mut industry_set: Vec<u32> = industry
        .iter()
        .zip(valid_mask.iter())
        .filter_map(|(&ind, &valid)| valid.then_some(ind))
        .collect();
    industry_set.sort_unstable();
    industry_set.dedup();
    let industry_map: HashMap<u32, usize> = industry_set.iter().enumerate().map(|(i, &ind)| (ind, i)).collect();

    // Number of regressors: industry dummies (n_ind - 1 for intercept absorption) + ln_mcap + intercept
    // Simplified: intercept + ln_mcap + (n_industries - 1) dummies
    let n_ind = industry_set.len();
    let k = 2 + n_ind.saturating_sub(1); // intercept + ln_mcap + (n_ind-1) dummies

    if valid_count < k {
        return vec![f64::NAN; n];
    }

    // Build X matrix and y vector for valid observations only
    let mut x_data: Vec<f64> = Vec::with_capacity(valid_count * k);
    let mut y_data: Vec<f64> = Vec::with_capacity(valid_count);

    for i in 0..n {
        if !valid_mask[i] {
            continue;
        }
        // intercept
        x_data.push(1.0);
        // ln_mcap
        x_data.push(ln_mcap[i]);
        // industry dummies (skip first as reference)
        let ind_idx = industry_map[&industry[i]];
        for j in 1..n_ind {
            x_data.push(if ind_idx == j { 1.0 } else { 0.0 });
        }
        y_data.push(factor[i]);
    }

    // Solve OLS via normal equations: beta = (X'X)^{-1} X'y
    let beta = solve_ols_normal(&x_data, &y_data, valid_count, k);

    // Compute residuals for ALL observations
    let mut residuals = vec![f64::NAN; n];
    for i in 0..n {
        if !valid_mask[i] {
            continue;
        }
        let ind_idx = industry_map[&industry[i]];
        let mut fitted = beta[0] + beta[1] * ln_mcap[i];
        for j in 1..n_ind {
            if ind_idx == j {
                fitted += beta[2 + j - 1];
            }
        }
        residuals[i] = factor[i] - fitted;
    }

    residuals
}

/// Solve OLS normal equations: beta = (X'X)^{-1} X'y
fn solve_ols_normal(x_flat: &[f64], y: &[f64], n: usize, k: usize) -> Vec<f64> {
    // X'X
    let mut xtx = vec![0.0; k * k];
    for i in 0..n {
        let row = &x_flat[i * k..(i + 1) * k];
        for a in 0..k {
            for b in 0..k {
                xtx[a * k + b] += row[a] * row[b];
            }
        }
    }

    // X'y
    let mut xty = vec![0.0; k];
    for i in 0..n {
        let row = &x_flat[i * k..(i + 1) * k];
        for a in 0..k {
            xty[a] += row[a] * y[i];
        }
    }

    // Solve via Cholesky or fallback to Gaussian elimination
    cholesky_solve(&xtx, &xty, k).unwrap_or_else(|| gauss_solve(&xtx, &xty, k))
}

fn cholesky_solve(ata: &[f64], atb: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];

    for i in 0..k {
        for j in 0..=i {
            let mut s = 0.0;
            for p in 0..j {
                s += l[i * k + p] * l[j * k + p];
            }
            if i == j {
                let diag = ata[i * k + i] - s;
                if diag <= 0.0 {
                    return None;
                }
                l[i * k + j] = diag.sqrt();
            } else {
                l[i * k + j] = (ata[i * k + j] - s) / l[j * k + j];
            }
        }
    }

    // Forward solve L*z = atb
    let mut z = vec![0.0; k];
    for i in 0..k {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * k + j] * z[j];
        }
        z[i] = (atb[i] - s) / l[i * k + i];
    }

    // Back solve L'*x = z
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = 0.0;
        for j in (i + 1)..k {
            s += l[j * k + i] * x[j];
        }
        x[i] = (z[i] - s) / l[i * k + i];
    }

    Some(x)
}

fn gauss_solve(ata: &[f64], atb: &[f64], k: usize) -> Vec<f64> {
    let mut aug = vec![0.0; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            aug[i * (k + 1) + j] = ata[i * k + j];
        }
        aug[i * (k + 1) + k] = atb[i];
    }

    for col in 0..k {
        let mut max_row = col;
        let mut max_val = aug[col * (k + 1) + col].abs();
        for row in col + 1..k {
            let val = aug[row * (k + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            for j in 0..=k {
                let temp = aug[col * (k + 1) + j];
                aug[col * (k + 1) + j] = aug[max_row * (k + 1) + j];
                aug[max_row * (k + 1) + j] = temp;
            }
        }

        let pivot = aug[col * (k + 1) + col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in col + 1..k {
            let factor = aug[row * (k + 1) + col] / pivot;
            for j in col..=k {
                aug[row * (k + 1) + j] -= factor * aug[col * (k + 1) + j];
            }
        }
    }

    let mut result = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = aug[i * (k + 1) + k];
        for j in i + 1..k {
            s -= aug[i * (k + 1) + j] * result[j];
        }
        let diag = aug[i * (k + 1) + i];
        result[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    result
}

/// Compute percentile rank (0.0 to 1.0) for a set of values.
/// NaN values get NaN rank.
pub fn percentile_rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];

    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.is_finite().then_some((i, v)))
        .collect();

    let valid_n = indexed.len();
    if valid_n < 2 {
        if valid_n == 1 {
            result[indexed[0].0] = 0.5;
        }
        return result;
    }

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = rank as f64 / (valid_n - 1) as f64;
    }

    result
}

/// Compute the full decision surface for a cross-section of stocks.
///
/// This is the main entry point for weekly factor computation.
///
/// # Arguments
/// * `symbols` - Stock symbols
/// * `adj_closes` - Map of symbol -> adjusted close price time series
/// * `amounts` - Map of symbol -> daily amount time series
/// * `industries` - Map of symbol -> industry code
/// * `ln_mcaps` - Map of symbol -> ln(market_cap) at decision date
/// * `momentum_short` - Short window for momentum (default: 20)
/// * `momentum_long` - Long window for momentum (default: 120)
/// * `vol_window` - Volatility window (default: 60)
/// * `adv_window` - ADV window (default: 20)
/// * `sma_window` - SMA window (default: 20)
pub fn compute_decision_surface(
    symbols: &[String],
    adj_closes: &HashMap<String, Vec<f64>>,
    amounts: &HashMap<String, Vec<f64>>,
    industries: &HashMap<String, u32>,
    ln_mcaps: &HashMap<String, f64>,
    momentum_short: usize,
    momentum_long: usize,
    vol_window: usize,
    adv_window: usize,
    sma_window: usize,
) -> Result<DecisionSurface> {
    let n = symbols.len();
    if n == 0 {
        return Err(FeatureError::InsufficientData("no symbols provided".into()));
    }

    let mut mom_vals = Vec::with_capacity(n);
    let mut vol_vals = Vec::with_capacity(n);
    let mut adv_vals = Vec::with_capacity(n);
    let mut sma_vals = Vec::with_capacity(n);
    let mut ind_codes = Vec::with_capacity(n);
    let mut mcap_vals = Vec::with_capacity(n);

    for sym in symbols {
        let ac = adj_closes
            .get(sym)
            .ok_or_else(|| FeatureError::Computation(format!("missing adj_close for {sym}")))?;
        let amt = amounts
            .get(sym)
            .ok_or_else(|| FeatureError::Computation(format!("missing amount for {sym}")))?;

        let mom = compute_momentum(ac, momentum_short, momentum_long);
        let vol = compute_low_vol(ac, vol_window);
        let adv = rolling_mean(amt, adv_window);
        let sma = rolling_mean(ac, sma_window);

        mom_vals.push(*mom.last().unwrap_or(&f64::NAN));
        vol_vals.push(*vol.last().unwrap_or(&f64::NAN));
        adv_vals.push(*adv.last().unwrap_or(&f64::NAN));
        sma_vals.push(*sma.last().unwrap_or(&f64::NAN));
        ind_codes.push(*industries.get(sym).unwrap_or(&0));
        mcap_vals.push(*ln_mcaps.get(sym).unwrap_or(&f64::NAN));
    }

    // Neutralize momentum and low_vol
    let mom_neutral = ols_neutralize(&mom_vals, &ind_codes, &mcap_vals);
    let vol_neutral = ols_neutralize(&vol_vals, &ind_codes, &mcap_vals);

    // Percentile rank
    let mom_rank = percentile_rank(&mom_neutral);
    let vol_rank = percentile_rank(&vol_neutral);

    // Composite score: equal weight
    let composite: Vec<f64> = (0..n)
        .map(|i| {
            if mom_rank[i].is_finite() && vol_rank[i].is_finite() {
                (mom_rank[i] + vol_rank[i]) / 2.0
            } else {
                f64::NAN
            }
        })
        .collect();

    let factor_rank = percentile_rank(&composite);
    let eligible_count = composite.iter().filter(|v| v.is_finite()).count();

    // Build DataFrame
    let df = df!(
        "symbol" => symbols,
        "momentum" => &mom_vals,
        "low_vol" => &vol_vals,
        "adv20" => &adv_vals,
        "sma20" => &sma_vals,
        "momentum_neutral" => &mom_neutral,
        "low_vol_neutral" => &vol_neutral,
        "momentum_rank" => &mom_rank,
        "low_vol_rank" => &vol_rank,
        "composite_score" => &composite,
        "factor_rank" => &factor_rank,
    )?;

    Ok(DecisionSurface {
        ranked: df,
        eligible_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adj_close_forward_fills_on_suspension() {
        let close = vec![10.0, 10.0, 10.0];
        let adj_factor = vec![1.0, 1.0, 1.0];
        let amount = vec![100.0, 0.0, 50.0];
        let adj = compute_adj_close(&close, &adj_factor, &amount);
        assert_eq!(adj[0], 10.0);
        assert_eq!(adj[1], 10.0);
        assert_eq!(adj[2], 10.0);
    }

    #[test]
    fn momentum_requires_long_window() {
        let prices: Vec<f64> = (0..130).map(|i| 100.0 + i as f64).collect();
        let mom = compute_momentum(&prices, 20, 120);
        assert!(mom[119].is_nan());
        assert!(mom[120].is_finite());
    }

    #[test]
    fn momentum_matches_python_shift_formula() {
        let prices = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let mom = compute_momentum(&prices, 2, 4);
        // at i=4: ln(P_2) - ln(P_0) = ln(120/100)
        let expected = (120.0_f64 / 100.0).ln();
        assert!((mom[4] - expected).abs() < 1e-12);
    }

    #[test]
    fn percentile_rank_spans_zero_to_one() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let ranks = percentile_rank(&values);
        assert!((ranks[0] - 0.0).abs() < 1e-10);
        assert!((ranks[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn decision_surface_produces_dataframe() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let prices: Vec<f64> = (0..150).map(|i| 100.0 + i as f64 * 0.1).collect();
        let amounts: Vec<f64> = vec![1_000_000.0; 150];
        let mut adj_closes = HashMap::new();
        let mut amounts_map = HashMap::new();
        let mut industries = HashMap::new();
        let mut ln_mcaps = HashMap::new();
        for (idx, sym) in symbols.iter().enumerate() {
            adj_closes.insert(sym.clone(), prices.clone());
            amounts_map.insert(sym.clone(), amounts.clone());
            industries.insert(sym.clone(), 1);
            ln_mcaps.insert(sym.clone(), 20.0 + idx as f64);
        }

        let surface = compute_decision_surface(
            &symbols,
            &adj_closes,
            &amounts_map,
            &industries,
            &ln_mcaps,
            20,
            120,
            60,
            20,
            20,
        )
        .expect("decision surface should compute");

        assert_eq!(surface.eligible_count, 2);
        assert_eq!(surface.ranked.height(), 2);
    }
}
