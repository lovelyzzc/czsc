//! PyO3 bindings for czsc-research.
//!
//! Exposes the research engine's computational functions to Python via
//! `czsc._native.research.*`. Python ceremony (hash chains, Git anchors,
//! CLI) stays in Python; only pure compute is exposed here.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::{benchmark, execution, features, statistics};

// ─── Statistics ─────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (weekly_diffs, lag=4, confidence=0.95))]
fn newey_west_hac<'py>(
    py: Python<'py>,
    weekly_diffs: Vec<f64>,
    lag: usize,
    confidence: f64,
) -> PyResult<Py<PyAny>> {
    let result = statistics::newey_west_hac(&weekly_diffs, lag, confidence)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("mean", result.mean)?;
    dict.set_item("se", result.se)?;
    dict.set_item("t_stat", result.t_stat)?;
    dict.set_item("lower_95", result.lower_95)?;
    dict.set_item("upper_95", result.upper_95)?;
    dict.set_item("n", result.n)?;
    dict.set_item("lag", result.lag)?;
    dict.set_item("annualized_mean", result.annualized_mean)?;
    dict.set_item("annualized_se", result.annualized_se)?;
    Ok(dict.into_any().unbind())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (weekly_diffs, block_size=4, n_draws=20_000, seed=42))]
fn circular_block_bootstrap<'py>(
    py: Python<'py>,
    weekly_diffs: Vec<f64>,
    block_size: usize,
    n_draws: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let result = statistics::circular_block_bootstrap(&weekly_diffs, block_size, n_draws, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("mean", result.mean)?;
    dict.set_item("se", result.se)?;
    dict.set_item("q05", result.q05)?;
    dict.set_item("q50", result.q50)?;
    dict.set_item("q95", result.q95)?;
    dict.set_item("n_draws", result.n_draws)?;
    dict.set_item("block_size", result.block_size)?;
    dict.set_item("seed", result.seed)?;
    dict.set_item("annualized_mean", result.annualized_mean)?;
    dict.set_item("annualized_q05", result.annualized_q05)?;
    dict.set_item("annualized_q95", result.annualized_q95)?;
    Ok(dict.into_any().unbind())
}

// ─── Features ───────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (close, adj_factor, amount))]
fn compute_adj_close(close: Vec<f64>, adj_factor: Vec<f64>, amount: Vec<f64>) -> Vec<f64> {
    features::compute_adj_close(&close, &adj_factor, &amount)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (adj_close, short=20, long=120))]
fn compute_momentum(adj_close: Vec<f64>, short: usize, long: usize) -> Vec<f64> {
    features::compute_momentum(&adj_close, short, long)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (adj_close, window=60))]
fn compute_low_vol(adj_close: Vec<f64>, window: usize) -> Vec<f64> {
    features::compute_low_vol(&adj_close, window)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (values, window))]
fn rolling_mean(values: Vec<f64>, window: usize) -> Vec<f64> {
    features::rolling_mean(&values, window)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (factor, industry, ln_mcap))]
fn ols_neutralize(factor: Vec<f64>, industry: Vec<u32>, ln_mcap: Vec<f64>) -> Vec<f64> {
    features::ols_neutralize(&factor, &industry, &ln_mcap)
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (values,))]
fn percentile_rank(values: Vec<f64>) -> Vec<f64> {
    features::percentile_rank(&values)
}

// ─── Benchmark ──────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (raw_dir, start_date="20220101", end_date=None, holding_sessions=5))]
fn compute_ew_benchmark<'py>(
    py: Python<'py>,
    raw_dir: &str,
    start_date: &str,
    end_date: Option<&str>,
    holding_sessions: i32,
) -> PyResult<Py<PyAny>> {
    let path = PathBuf::from(raw_dir);

    let panel = benchmark::load_daily_panel(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let calendar = benchmark::build_calendar(&panel)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let decision_dates = benchmark::build_decision_dates(&calendar, start_date, end_date);
    let results = benchmark::compute_ew_benchmark(&path, &decision_dates, holding_sessions)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let list = PyList::empty(py);
    for wb in &results {
        let dict = PyDict::new(py);
        dict.set_item("decision_dt", &wb.decision_dt)?;
        dict.set_item("ew_all_return", wb.ew_all_return)?;
        dict.set_item("ew_all_median_return", wb.ew_all_median_return)?;
        dict.set_item("ew_all_std", wb.ew_all_std)?;
        dict.set_item("tradable_count", wb.tradable_count)?;
        dict.set_item("observable_exit_count", wb.observable_exit_count)?;
        dict.set_item("observable_exit_rate", wb.observable_exit_rate)?;
        list.append(dict)?;
    }
    Ok(list.into_any().unbind())
}

// ─── Execution ──────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (notional_cny, is_sell, adv20, participation_rate=0.01,
                    commission_bps=2.5, transfer_bps=0.1, stamp_duty_bps_sell=5.0,
                    base_slippage_bps=10.0, impact_bps_at_1pct=15.0, cost_multiplier=1.0))]
#[allow(clippy::too_many_arguments)]
fn compute_transaction_cost(
    notional_cny: f64,
    is_sell: bool,
    adv20: f64,
    participation_rate: f64,
    commission_bps: f64,
    transfer_bps: f64,
    stamp_duty_bps_sell: f64,
    base_slippage_bps: f64,
    impact_bps_at_1pct: f64,
    cost_multiplier: f64,
) -> f64 {
    let cost_model = execution::CostModel {
        commission_bps,
        transfer_bps,
        min_commission_cny: 5.0,
        stamp_duty_bps_sell,
        base_slippage_bps,
        impact_bps_at_1pct,
        max_adv_participation: 0.05,
        auction_participation_cap: 0.10,
        cost_multiplier,
    };
    execution::compute_transaction_cost(&cost_model, notional_cny, is_sell, adv20, participation_rate)
}

// ─── Portfolio ───────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (ranked_symbols, target_slots=50, retention_max_rank=75,
                    gate_regimes=None, use_chan_gate=false, use_sma_gate=false))]
fn simulate_fc_path<'py>(
    py: Python<'py>,
    ranked_symbols: Vec<Vec<String>>,
    target_slots: usize,
    retention_max_rank: usize,
    gate_regimes: Option<Vec<u8>>,
    use_chan_gate: bool,
    use_sma_gate: bool,
) -> PyResult<Py<PyAny>> {
    use crate::portfolio::{self, PortfolioConfig, WeeklyDecision};
    use std::collections::HashMap;

    let gate = gate_regimes
        .map(|v| v.into_iter().collect::<HashSet<_>>())
        .unwrap_or_else(|| [5, 6, 7, 8].iter().copied().collect());

    let config = PortfolioConfig {
        target_slots,
        retention_max_rank,
        gate_regimes: gate,
        use_chan_gate,
        use_sma_gate,
    };

    let decisions: Vec<WeeklyDecision> = ranked_symbols
        .into_iter()
        .enumerate()
        .map(|(i, symbols)| WeeklyDecision {
            week_label: format!("week_{i}"),
            ranked_symbols: symbols,
            factor_ranks: HashMap::new(),
            regimes: HashMap::new(),
            above_sma20: HashMap::new(),
            blocked_exits: HashSet::new(),
        })
        .collect();

    let results = portfolio::simulate_portfolio_path(&decisions, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let list = PyList::empty(py);
    for wm in &results {
        let dict = PyDict::new(py);
        dict.set_item("week_label", &wm.week_label)?;
        dict.set_item("holdings", &wm.holdings)?;
        dict.set_item("new_entries", &wm.new_entries)?;
        dict.set_item("exits", &wm.exits)?;
        dict.set_item("retained", &wm.retained)?;
        dict.set_item("blocked_exits", &wm.blocked_exits)?;
        list.append(dict)?;
    }
    Ok(list.into_any().unbind())
}

// ─── Module Registration ────────────────────────────────────────

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let research = PyModule::new(py, "research")?;
    research.setattr("__name__", "czsc._native.research")?;

    // Statistics
    research.add_function(wrap_pyfunction!(newey_west_hac, &research)?)?;
    research.add_function(wrap_pyfunction!(circular_block_bootstrap, &research)?)?;

    // Features
    research.add_function(wrap_pyfunction!(compute_adj_close, &research)?)?;
    research.add_function(wrap_pyfunction!(compute_momentum, &research)?)?;
    research.add_function(wrap_pyfunction!(compute_low_vol, &research)?)?;
    research.add_function(wrap_pyfunction!(rolling_mean, &research)?)?;
    research.add_function(wrap_pyfunction!(ols_neutralize, &research)?)?;
    research.add_function(wrap_pyfunction!(percentile_rank, &research)?)?;

    // Benchmark
    research.add_function(wrap_pyfunction!(compute_ew_benchmark, &research)?)?;

    // Execution
    research.add_function(wrap_pyfunction!(compute_transaction_cost, &research)?)?;

    // Portfolio
    research.add_function(wrap_pyfunction!(simulate_fc_path, &research)?)?;

    let sys = py.import("sys")?;
    let py_modules = sys.getattr("modules")?;
    py_modules.set_item("czsc._native.research", &research)?;
    parent.add("research", &research)?;

    Ok(())
}
