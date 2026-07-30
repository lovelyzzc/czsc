//! HAC standard error + circular block bootstrap.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StatisticsError {
    #[error("statistics error: {0}")]
    Computation(String),
    #[error("degenerate inference: {0}")]
    DegenerateInference(String),
}

pub type Result<T> = std::result::Result<T, StatisticsError>;

#[derive(Debug, Clone, PartialEq)]
pub struct HacResult {
    pub mean: f64,
    pub se: f64,
    pub t_stat: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub n: usize,
    pub lag: usize,
    pub annualized_mean: f64,
    pub annualized_se: f64,
}

/// Newey-West HAC standard error for weekly time series differences.
pub fn newey_west_hac(weekly_diffs: &[f64], lag: usize, confidence: f64) -> Result<HacResult> {
    let n = weekly_diffs.len();
    if n < 3 {
        return Err(StatisticsError::DegenerateInference(format!(
            "need >= 3 observations, got {n}"
        )));
    }
    if lag >= n {
        return Err(StatisticsError::DegenerateInference(format!(
            "lag {lag} must be < n {n}"
        )));
    }
    if !(0.5..1.0).contains(&confidence) {
        return Err(StatisticsError::DegenerateInference(
            "confidence must be in (0.5, 1)".into(),
        ));
    }
    if weekly_diffs.iter().any(|x| !x.is_finite()) {
        return Err(StatisticsError::DegenerateInference(
            "weekly_diffs must be finite".into(),
        ));
    }

    let mean = weekly_diffs.iter().sum::<f64>() / n as f64;
    let residuals: Vec<f64> = weekly_diffs.iter().map(|x| x - mean).collect();
    let gamma_0 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;

    let mut nw_var = gamma_0;
    for j in 1..=lag.min(n - 1) {
        let gamma_j: f64 = residuals[j..]
            .iter()
            .zip(residuals[..n - j].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n as f64;
        let weight = 1.0 - j as f64 / (lag as f64 + 1.0);
        nw_var += 2.0 * weight * gamma_j;
    }

    let scale = gamma_0.max(f64::MIN_POSITIVE);
    if nw_var < -1e-12 * scale {
        return Err(StatisticsError::DegenerateInference(
            "Newey-West long-run variance is materially negative".into(),
        ));
    }
    if nw_var <= f64::EPSILON * scale.max(1.0) {
        return Err(StatisticsError::DegenerateInference(
            "Newey-West variance estimate is non-positive".into(),
        ));
    }

    let se = (nw_var / n as f64).sqrt();
    let t_stat = mean / se;
    let z = normal_quantile((1.0 + confidence) / 2.0);
    let lower = mean - z * se;
    let upper = mean + z * se;

    Ok(HacResult {
        mean,
        se,
        t_stat,
        lower_95: lower,
        upper_95: upper,
        n,
        lag,
        annualized_mean: mean * 52.0,
        annualized_se: se * 52.0_f64.sqrt(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapResult {
    pub mean: f64,
    pub se: f64,
    pub q05: f64,
    pub q50: f64,
    pub q95: f64,
    pub n_draws: usize,
    pub block_size: usize,
    pub seed: u64,
    pub annualized_mean: f64,
    pub annualized_q05: f64,
    pub annualized_q95: f64,
}

/// Circular block bootstrap for weekly time series.
pub fn circular_block_bootstrap(
    weekly_diffs: &[f64],
    block_size: usize,
    n_draws: usize,
    seed: u64,
) -> Result<BootstrapResult> {
    let n = weekly_diffs.len();
    if n < block_size {
        return Err(StatisticsError::Computation(format!(
            "series length {n} < block_size {block_size}"
        )));
    }
    if n_draws == 0 {
        return Err(StatisticsError::Computation("n_draws must be > 0".into()));
    }
    if weekly_diffs.iter().any(|x| !x.is_finite()) {
        return Err(StatisticsError::Computation(
            "weekly_diffs must be finite".into(),
        ));
    }

    let blocks_needed = (n + block_size - 1) / block_size;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(n_draws);

    for _ in 0..n_draws {
        let mut sample = Vec::with_capacity(blocks_needed * block_size);
        for _ in 0..blocks_needed {
            let start = rng.random_range(0..n);
            for j in 0..block_size {
                sample.push(weekly_diffs[(start + j) % n]);
            }
        }
        means.push(sample[..n].iter().sum::<f64>() / n as f64);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = means.iter().sum::<f64>() / n_draws as f64;
    let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (n_draws - 1) as f64;
    let se = var.sqrt();

    Ok(BootstrapResult {
        mean,
        se,
        q05: quantile(&means, 0.05),
        q50: quantile(&means, 0.50),
        q95: quantile(&means, 0.95),
        n_draws,
        block_size,
        seed,
        annualized_mean: mean * 52.0,
        annualized_q05: quantile(&means, 0.05) * 52.0,
        annualized_q95: quantile(&means, 0.95) * 52.0,
    })
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newey_west_hac_basic() {
        let diffs: Vec<f64> = (0..52).map(|i| 0.001 + (i as f64 * 0.0001).sin() * 0.002).collect();
        let result = newey_west_hac(&diffs, 4, 0.95).unwrap();
        assert_eq!(result.n, 52);
        assert!(result.se > 0.0);
        assert!(result.lower_95 < result.mean && result.upper_95 > result.mean);
    }

    #[test]
    fn circular_block_bootstrap_constant_series() {
        let diffs = vec![0.01; 52];
        let result = circular_block_bootstrap(&diffs, 4, 1_000, 42).unwrap();
        assert!((result.mean - 0.01).abs() < 1e-12);
        assert_eq!(result.n_draws, 1_000);
    }
}
