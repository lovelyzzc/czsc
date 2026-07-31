//! PyO3 bindings for czsc-research.
//!
//! Exposes the research engine's computational functions to Python via
//! `czsc._native.research.*`. Python ceremony (hash chains, Git anchors,
//! CLI) stays in Python; only pure compute is exposed here.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::{benchmark, execution, features, random_control, slot_backtest, state_cache, statistics};

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

// ─── Slot Backtest ──────────────────────────────────────────────

/// 日频贪心槽位组合回测。
///
/// 输入为 candidates（字典列表）和 close panel（三元组列表），
/// 返回回测结果字典（trades, daily_returns, curve_stats, pair_stats）。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (candidates, close_triples, n_slots=10, fill_mode="strict",
                    buy_cost=0.0015, sell_cost=0.0025, train_end_year=2023))]
#[allow(clippy::too_many_arguments)]
fn simulate_surge_backtest<'py>(
    py: Python<'py>,
    candidates: Vec<HashMap<String, Py<PyAny>>>,
    close_triples: Vec<(String, String, f64)>,
    n_slots: usize,
    fill_mode: &str,
    buy_cost: f64,
    sell_cost: f64,
    train_end_year: i32,
) -> PyResult<Py<PyAny>> {
    use chrono::NaiveDate;
    use slot_backtest::{CandidateRow, ClosePanel, FillMode, SlotBacktestConfig};

    let fm = FillMode::from_str(fill_mode)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let config = SlotBacktestConfig {
        n_slots,
        buy_cost,
        sell_cost,
        fill_mode: fm,
    };

    let parse_dt = |s: &str| -> PyResult<chrono::DateTime<chrono::Utc>> {
        let nd = NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(s, "%Y%m%d"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad date: {e}")))?;
        Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc())
    };

    let get_str = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<String> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<String>(py)
    };
    let get_f64 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<f64> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<f64>(py)
    };
    let get_i32 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<i32> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<i32>(py)
    };
    let get_usize = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<usize> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<usize>(py)
    };
    let get_u8 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<u8> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<u8>(py)
    };

    let mut cands = Vec::with_capacity(candidates.len());
    for d in &candidates {
        let entry_dt_str = get_str(d, "entry_dt", py)?;
        let exit_dt_str = get_str(d, "exit_dt", py)?;
        let gate_level_val = get_u8(d, "gate_level", py).unwrap_or(3);

        cands.push(CandidateRow {
            symbol: get_str(d, "symbol", py)?,
            entry_dt: parse_dt(&entry_dt_str)?,
            exit_dt: parse_dt(&exit_dt_str)?,
            entry_price: get_f64(d, "entry_price", py)?,
            exit_price: get_f64(d, "exit_price", py)?,
            gate_level: czsc_trend_regime::GateLevel::from_u8(gate_level_val),
            gate_confidence: get_f64(d, "gate_confidence", py).unwrap_or(1.0),
            priority: get_f64(d, "priority", py)?,
            ret_gross_pct: get_f64(d, "ret_gross_pct", py)?,
            hold_days: get_usize(d, "hold_days", py).unwrap_or(0),
            seg: get_str(d, "seg", py).unwrap_or_default(),
            year: get_i32(d, "year", py).unwrap_or(0),
            entry_regime: get_u8(d, "entry_regime", py).unwrap_or(0),
        });
    }

    let panel_triples: Vec<(String, chrono::DateTime<chrono::Utc>, f64)> = close_triples
        .into_iter()
        .map(|(sym, dt_str, close)| {
            let dt = parse_dt(&dt_str)?;
            Ok((sym, dt, close))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let panel = ClosePanel::from_triples(&panel_triples);

    let result = slot_backtest::simulate_slots(&cands, &panel, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let segments = slot_backtest::segment_stats(&result, train_end_year);

    let dict = PyDict::new(py);

    // trades as list of dicts
    let trades_list = PyList::empty(py);
    for tr in &result.trades {
        let td = PyDict::new(py);
        td.set_item("symbol", &tr.symbol)?;
        td.set_item("entry_dt", tr.entry_dt.format("%Y-%m-%d").to_string())?;
        td.set_item("exit_dt", tr.exit_dt.format("%Y-%m-%d").to_string())?;
        td.set_item("entry_price", tr.entry_price)?;
        td.set_item("exit_price", tr.exit_price)?;
        td.set_item("ret_gross_pct", tr.ret_gross_pct)?;
        td.set_item("ret_net_pct", tr.ret_net_pct)?;
        td.set_item("gate_level", tr.gate_level.as_str())?;
        td.set_item("gate_confidence", tr.gate_confidence)?;
        td.set_item("priority", tr.priority)?;
        td.set_item("hold_days", tr.hold_days)?;
        td.set_item("seg", &tr.seg)?;
        td.set_item("year", tr.year)?;
        td.set_item("position_weight", tr.position_weight)?;
        td.set_item("entry_regime", tr.entry_regime)?;
        trades_list.append(td)?;
    }
    dict.set_item("trades", trades_list)?;

    // daily_returns as list of (date_str, return)
    let dr_list = PyList::empty(py);
    for (dt, ret) in &result.daily_returns {
        let pair = PyList::empty(py);
        pair.append(dt.format("%Y-%m-%d").to_string())?;
        pair.append(*ret)?;
        dr_list.append(pair)?;
    }
    dict.set_item("daily_returns", dr_list)?;

    // segments as list of dicts
    let seg_list = PyList::empty(py);
    for seg in &segments {
        let sd = PyDict::new(py);
        sd.set_item("label", &seg.label)?;
        let cd = PyDict::new(py);
        cd.set_item("annual_return_pct", seg.curve.annual_return_pct)?;
        cd.set_item("sharpe", seg.curve.sharpe)?;
        cd.set_item("max_drawdown_pct", seg.curve.max_drawdown_pct)?;
        cd.set_item("calmar", seg.curve.calmar)?;
        cd.set_item("trading_days", seg.curve.trading_days)?;
        sd.set_item("curve", cd)?;
        let pd = PyDict::new(py);
        pd.set_item("n_trades", seg.pair.n_trades)?;
        pd.set_item("win_rate_pct", seg.pair.win_rate_pct)?;
        pd.set_item("profit_loss_ratio", seg.pair.profit_loss_ratio)?;
        pd.set_item("net_mean_pct", seg.pair.net_mean_pct)?;
        pd.set_item("net_median_pct", seg.pair.net_median_pct)?;
        pd.set_item("gross_mean_pct", seg.pair.gross_mean_pct)?;
        pd.set_item("avg_hold_days", seg.pair.avg_hold_days)?;
        sd.set_item("pair", pd)?;
        seg_list.append(sd)?;
    }
    dict.set_item("segments", seg_list)?;

    Ok(dict.into_any().unbind())
}

/// Helper: 将 BacktestResult + segments 序列化为 Python dict。
fn backtest_result_to_pydict<'py>(
    py: Python<'py>,
    result: &slot_backtest::BacktestResult,
    segments: &[slot_backtest::SegmentStats],
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    let trades_list = PyList::empty(py);
    for tr in &result.trades {
        let td = PyDict::new(py);
        td.set_item("symbol", &tr.symbol)?;
        td.set_item("entry_dt", tr.entry_dt.format("%Y-%m-%d").to_string())?;
        td.set_item("exit_dt", tr.exit_dt.format("%Y-%m-%d").to_string())?;
        td.set_item("entry_price", tr.entry_price)?;
        td.set_item("exit_price", tr.exit_price)?;
        td.set_item("ret_gross_pct", tr.ret_gross_pct)?;
        td.set_item("ret_net_pct", tr.ret_net_pct)?;
        td.set_item("gate_level", tr.gate_level.as_str())?;
        td.set_item("gate_confidence", tr.gate_confidence)?;
        td.set_item("priority", tr.priority)?;
        td.set_item("hold_days", tr.hold_days)?;
        td.set_item("seg", &tr.seg)?;
        td.set_item("year", tr.year)?;
        td.set_item("position_weight", tr.position_weight)?;
        td.set_item("entry_regime", tr.entry_regime)?;
        trades_list.append(td)?;
    }
    dict.set_item("trades", trades_list)?;

    let dr_list = PyList::empty(py);
    for (dt, ret) in &result.daily_returns {
        let pair = PyList::empty(py);
        pair.append(dt.format("%Y-%m-%d").to_string())?;
        pair.append(*ret)?;
        dr_list.append(pair)?;
    }
    dict.set_item("daily_returns", dr_list)?;

    let seg_list = PyList::empty(py);
    for seg in segments {
        let sd = PyDict::new(py);
        sd.set_item("label", &seg.label)?;
        let cd = PyDict::new(py);
        cd.set_item("annual_return_pct", seg.curve.annual_return_pct)?;
        cd.set_item("sharpe", seg.curve.sharpe)?;
        cd.set_item("max_drawdown_pct", seg.curve.max_drawdown_pct)?;
        cd.set_item("calmar", seg.curve.calmar)?;
        cd.set_item("trading_days", seg.curve.trading_days)?;
        sd.set_item("curve", cd)?;
        let pd = PyDict::new(py);
        pd.set_item("n_trades", seg.pair.n_trades)?;
        pd.set_item("win_rate_pct", seg.pair.win_rate_pct)?;
        pd.set_item("profit_loss_ratio", seg.pair.profit_loss_ratio)?;
        pd.set_item("net_mean_pct", seg.pair.net_mean_pct)?;
        pd.set_item("net_median_pct", seg.pair.net_median_pct)?;
        pd.set_item("gross_mean_pct", seg.pair.gross_mean_pct)?;
        pd.set_item("avg_hold_days", seg.pair.avg_hold_days)?;
        sd.set_item("pair", pd)?;
        seg_list.append(sd)?;
    }
    dict.set_item("segments", seg_list)?;

    Ok(dict.into_any().unbind())
}

/// 日频贪心槽位组合回测（从 parquet 路径读取面板）。
///
/// 与 `simulate_surge_backtest` 功能相同，但面板从 parquet 文件路径读取，
/// 避免 Python→Rust 传递 540 万行三元组的开销。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (candidates, panel_path, n_slots=10, fill_mode="strict",
                    buy_cost=0.0015, sell_cost=0.0025, train_end_year=2023))]
#[allow(clippy::too_many_arguments)]
fn simulate_surge_backtest_parquet<'py>(
    py: Python<'py>,
    candidates: Vec<HashMap<String, Py<PyAny>>>,
    panel_path: &str,
    n_slots: usize,
    fill_mode: &str,
    buy_cost: f64,
    sell_cost: f64,
    train_end_year: i32,
) -> PyResult<Py<PyAny>> {
    use chrono::NaiveDate;
    use slot_backtest::{CandidateRow, ClosePanel, FillMode, SlotBacktestConfig};

    let fm = FillMode::from_str(fill_mode)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let config = SlotBacktestConfig {
        n_slots,
        buy_cost,
        sell_cost,
        fill_mode: fm,
    };

    let parse_dt = |s: &str| -> PyResult<chrono::DateTime<chrono::Utc>> {
        let nd = NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(s, "%Y%m%d"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad date: {e}")))?;
        Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc())
    };

    let get_str = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<String> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<String>(py)
    };
    let get_f64 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<f64> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<f64>(py)
    };
    let get_i32 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<i32> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<i32>(py)
    };
    let get_usize = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<usize> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<usize>(py)
    };
    let get_u8 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<u8> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<u8>(py)
    };

    let mut cands = Vec::with_capacity(candidates.len());
    for d in &candidates {
        let entry_dt_str = get_str(d, "entry_dt", py)?;
        let exit_dt_str = get_str(d, "exit_dt", py)?;
        let gate_level_val = get_u8(d, "gate_level", py).unwrap_or(3);

        cands.push(CandidateRow {
            symbol: get_str(d, "symbol", py)?,
            entry_dt: parse_dt(&entry_dt_str)?,
            exit_dt: parse_dt(&exit_dt_str)?,
            entry_price: get_f64(d, "entry_price", py)?,
            exit_price: get_f64(d, "exit_price", py)?,
            gate_level: czsc_trend_regime::GateLevel::from_u8(gate_level_val),
            gate_confidence: get_f64(d, "gate_confidence", py).unwrap_or(1.0),
            priority: get_f64(d, "priority", py)?,
            ret_gross_pct: get_f64(d, "ret_gross_pct", py)?,
            hold_days: get_usize(d, "hold_days", py).unwrap_or(0),
            seg: get_str(d, "seg", py).unwrap_or_default(),
            year: get_i32(d, "year", py).unwrap_or(0),
            entry_regime: get_u8(d, "entry_regime", py).unwrap_or(0),
        });
    }

    let panel_path = std::path::Path::new(panel_path);
    let panel = ClosePanel::from_parquet(panel_path, "close")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let result = slot_backtest::simulate_slots(&cands, &panel, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let segments = slot_backtest::segment_stats(&result, train_end_year);

    backtest_result_to_pydict(py, &result, &segments)
}

/// 随机对照 beta 剥离（从 parquet 路径读取面板）。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (trades, panel_path, k=50, min_valid=10, seed=42, min_segment_n=30))]
#[allow(clippy::too_many_arguments)]
fn compute_surge_excess_parquet<'py>(
    py: Python<'py>,
    trades: Vec<HashMap<String, Py<PyAny>>>,
    panel_path: &str,
    k: usize,
    min_valid: usize,
    seed: u64,
    min_segment_n: usize,
) -> PyResult<Py<PyAny>> {
    use chrono::NaiveDate;
    use random_control::{AmountPanel, ClosePanelView, ControlConfig, OpenPanel};
    use slot_backtest::TradeRecord;

    let parse_dt = |s: &str| -> PyResult<chrono::DateTime<chrono::Utc>> {
        let nd = NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(s, "%Y%m%d"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad date: {e}")))?;
        Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc())
    };

    let get_str = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<String> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<String>(py)
    };
    let get_f64 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<f64> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<f64>(py)
    };

    let mut trade_records = Vec::with_capacity(trades.len());
    for d in &trades {
        let entry_dt_str = get_str(d, "entry_dt", py)?;
        let exit_dt_str = get_str(d, "exit_dt", py)?;
        trade_records.push(TradeRecord {
            symbol: get_str(d, "symbol", py)?,
            entry_dt: parse_dt(&entry_dt_str)?,
            exit_dt: parse_dt(&exit_dt_str)?,
            entry_price: get_f64(d, "entry_price", py)?,
            exit_price: get_f64(d, "exit_price", py)?,
            ret_gross_pct: get_f64(d, "ret_gross_pct", py)?,
            ret_net_pct: get_f64(d, "ret_net_pct", py).unwrap_or(0.0),
            gate_level: czsc_trend_regime::GateLevel::Full,
            gate_confidence: 1.0,
            priority: get_f64(d, "priority", py).unwrap_or(0.0),
            hold_days: d.get("hold_days")
                .and_then(|v| v.extract::<usize>(py).ok())
                .unwrap_or(0),
            seg: get_str(d, "seg", py).unwrap_or_default(),
            year: d.get("year")
                .and_then(|v| v.extract::<i32>(py).ok())
                .unwrap_or(0),
            position_weight: 1.0,
            entry_regime: d.get("entry_regime")
                .and_then(|v| v.extract::<u8>(py).ok())
                .unwrap_or(0),
        });
    }

    use polars::prelude::*;
    let pl_path = PlPath::new(panel_path);
    let df = LazyFrame::scan_parquet(pl_path, Default::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("scan: {e}")))?
        .select([col("symbol"), col("dt"), col("open"), col("close"), col("amount_e")])
        .collect()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("collect: {e}")))?;

    let sym_col = df.column("symbol").unwrap().str().unwrap();
    let dt_col = df.column("dt").unwrap().datetime().unwrap();
    let ts_phys = dt_col.physical();
    let tu = dt_col.time_unit();

    let build_triples = |val_name: &str| -> Vec<(String, chrono::DateTime<chrono::Utc>, f64)> {
        let val_col = df.column(val_name).unwrap().f64().unwrap();
        let mut out = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            let sym = sym_col.get(i).unwrap_or("").to_string();
            let ts_raw = ts_phys.get(i).unwrap_or(0);
            let val = val_col.get(i).unwrap_or(f64::NAN);
            if val.is_nan() { continue; }
            let ts_ms = match tu {
                TimeUnit::Nanoseconds => ts_raw / 1_000_000,
                TimeUnit::Microseconds => ts_raw / 1_000,
                TimeUnit::Milliseconds => ts_raw,
            };
            let dt = chrono::DateTime::from_timestamp_millis(ts_ms)
                .unwrap_or_default();
            out.push((sym, dt, val));
        }
        out
    };

    let open_triples = build_triples("open");
    let close_triples = build_triples("close");
    let amount_triples = build_triples("amount_e");

    let open_p = OpenPanel::from_triples(&open_triples);
    let close_p = ClosePanelView::from_triples(&close_triples);
    let amount_p = AmountPanel::from_triples(&amount_triples);

    let config = ControlConfig { k, min_valid, seed };
    let records = random_control::compute_excess(&trade_records, &open_p, &close_p, &amount_p, &config);
    let segments = random_control::summarize_excess(&records, min_segment_n);

    let dict = PyDict::new(py);

    let rec_list = PyList::empty(py);
    for r in &records {
        let rd = PyDict::new(py);
        rd.set_item("symbol", &r.symbol)?;
        rd.set_item("entry_dt", r.entry_dt.format("%Y-%m-%d").to_string())?;
        rd.set_item("ret_gross_pct", r.ret_gross_pct)?;
        rd.set_item("control_median_pct", r.control_median_pct)?;
        rd.set_item("excess_pct", r.excess_pct)?;
        rd.set_item("control_n", r.control_n)?;
        rd.set_item("seg", &r.seg)?;
        rd.set_item("year", r.year)?;
        rec_list.append(rd)?;
    }
    dict.set_item("records", rec_list)?;

    let seg_list = PyList::empty(py);
    for seg in &segments {
        let sd = PyDict::new(py);
        sd.set_item("label", &seg.label)?;
        sd.set_item("n", seg.n)?;
        sd.set_item("excess_mean_pct", seg.excess_mean_pct)?;
        sd.set_item("excess_median_pct", seg.excess_median_pct)?;
        sd.set_item("t_stat", seg.t_stat)?;
        sd.set_item("positive_rate_pct", seg.positive_rate_pct)?;
        seg_list.append(sd)?;
    }
    dict.set_item("segments", seg_list)?;

    let oos = segments.iter().find(|s| s.label.contains("OOS"));
    let verdict = match oos {
        Some(s) if s.excess_mean_pct > 0.0 && s.t_stat.abs() >= 2.0 => {
            "OOS 超额显著为正 → 有选股 alpha"
        }
        _ => "OOS 超额不显著/为负 → 收益主体为规模/市场 beta",
    };
    dict.set_item("verdict", verdict)?;

    Ok(dict.into_any().unbind())
}

/// 随机对照 beta 剥离。
///
/// 输入 trades（字典列表）+ open/close/amount 面板（三元组列表），
/// 返回超额统计字典。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (trades, open_triples, close_triples, amount_triples,
                    k=50, min_valid=10, seed=42, min_segment_n=30))]
#[allow(clippy::too_many_arguments)]
fn compute_surge_excess<'py>(
    py: Python<'py>,
    trades: Vec<HashMap<String, Py<PyAny>>>,
    open_triples: Vec<(String, String, f64)>,
    close_triples: Vec<(String, String, f64)>,
    amount_triples: Vec<(String, String, f64)>,
    k: usize,
    min_valid: usize,
    seed: u64,
    min_segment_n: usize,
) -> PyResult<Py<PyAny>> {
    use chrono::NaiveDate;
    use random_control::{AmountPanel, ClosePanelView, ControlConfig, OpenPanel};
    use slot_backtest::TradeRecord;

    let parse_dt = |s: &str| -> PyResult<chrono::DateTime<chrono::Utc>> {
        let nd = NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(s, "%Y%m%d"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad date: {e}")))?;
        Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc())
    };

    let get_str = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<String> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<String>(py)
    };
    let get_f64 = |d: &HashMap<String, Py<PyAny>>, key: &str, py: Python<'_>| -> PyResult<f64> {
        d.get(key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
            .extract::<f64>(py)
    };

    let mut trade_records = Vec::with_capacity(trades.len());
    for d in &trades {
        let entry_dt_str = get_str(d, "entry_dt", py)?;
        let exit_dt_str = get_str(d, "exit_dt", py)?;
        trade_records.push(TradeRecord {
            symbol: get_str(d, "symbol", py)?,
            entry_dt: parse_dt(&entry_dt_str)?,
            exit_dt: parse_dt(&exit_dt_str)?,
            entry_price: get_f64(d, "entry_price", py)?,
            exit_price: get_f64(d, "exit_price", py)?,
            ret_gross_pct: get_f64(d, "ret_gross_pct", py)?,
            ret_net_pct: get_f64(d, "ret_net_pct", py).unwrap_or(0.0),
            gate_level: czsc_trend_regime::GateLevel::Full,
            gate_confidence: 1.0,
            priority: get_f64(d, "priority", py).unwrap_or(0.0),
            hold_days: d.get("hold_days")
                .and_then(|v| v.extract::<usize>(py).ok())
                .unwrap_or(0),
            seg: get_str(d, "seg", py).unwrap_or_default(),
            year: d.get("year")
                .and_then(|v| v.extract::<i32>(py).ok())
                .unwrap_or(0),
            position_weight: 1.0,
            entry_regime: d.get("entry_regime")
                .and_then(|v| v.extract::<u8>(py).ok())
                .unwrap_or(0),
        });
    }

    let to_panel_triples = |triples: Vec<(String, String, f64)>| -> PyResult<Vec<(String, chrono::DateTime<chrono::Utc>, f64)>> {
        triples
            .into_iter()
            .map(|(sym, dt_str, val)| {
                let dt = parse_dt(&dt_str)?;
                Ok((sym, dt, val))
            })
            .collect()
    };

    let open_p = OpenPanel::from_triples(&to_panel_triples(open_triples)?);
    let close_p = ClosePanelView::from_triples(&to_panel_triples(close_triples)?);
    let amount_p = AmountPanel::from_triples(&to_panel_triples(amount_triples)?);

    let config = ControlConfig { k, min_valid, seed };
    let records = random_control::compute_excess(&trade_records, &open_p, &close_p, &amount_p, &config);
    let segments = random_control::summarize_excess(&records, min_segment_n);

    let dict = PyDict::new(py);

    // individual excess records
    let rec_list = PyList::empty(py);
    for r in &records {
        let rd = PyDict::new(py);
        rd.set_item("symbol", &r.symbol)?;
        rd.set_item("entry_dt", r.entry_dt.format("%Y-%m-%d").to_string())?;
        rd.set_item("ret_gross_pct", r.ret_gross_pct)?;
        rd.set_item("control_median_pct", r.control_median_pct)?;
        rd.set_item("excess_pct", r.excess_pct)?;
        rd.set_item("control_n", r.control_n)?;
        rd.set_item("seg", &r.seg)?;
        rd.set_item("year", r.year)?;
        rec_list.append(rd)?;
    }
    dict.set_item("records", rec_list)?;

    // segment summaries
    let seg_list = PyList::empty(py);
    for seg in &segments {
        let sd = PyDict::new(py);
        sd.set_item("label", &seg.label)?;
        sd.set_item("n", seg.n)?;
        sd.set_item("excess_mean_pct", seg.excess_mean_pct)?;
        sd.set_item("excess_median_pct", seg.excess_median_pct)?;
        sd.set_item("t_stat", seg.t_stat)?;
        sd.set_item("positive_rate_pct", seg.positive_rate_pct)?;
        seg_list.append(sd)?;
    }
    dict.set_item("segments", seg_list)?;

    // verdict
    let oos = segments.iter().find(|s| s.label.contains("OOS"));
    let verdict = match oos {
        Some(s) if s.excess_mean_pct > 0.0 && s.t_stat.abs() >= 2.0 => {
            "OOS 超额显著为正 → 有选股 alpha"
        }
        _ => "OOS 超额不显著/为负 → 收益主体为规模/市场 beta",
    };
    dict.set_item("verdict", verdict)?;

    Ok(dict.into_any().unbind())
}

// ─── State Cache ────────────────────────────────────────────────

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (raw_dir, output_path, warmup_bars=120, batch_size=200))]
fn build_state_cache<'py>(
    py: Python<'py>,
    raw_dir: &str,
    output_path: &str,
    warmup_bars: usize,
    batch_size: usize,
) -> PyResult<Py<PyAny>> {
    use polars::prelude::*;
    use std::fs;
    use std::path::Path;

    let raw_path = Path::new(raw_dir);
    let out_path = Path::new(output_path);

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot create output dir: {e}")))?;
    }

    let mut paths: Vec<std::path::PathBuf> = fs::read_dir(raw_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot read dir: {e}")))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    paths.sort();

    if paths.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no parquet files found"));
    }

    let batch_sz = batch_size.max(1);
    let mut all_parts: Vec<DataFrame> = Vec::new();
    let mut total_rows = 0usize;
    let mut source_count = 0usize;
    let mut regime_counts = std::collections::HashMap::<u8, usize>::new();
    for r in 0..=10u8 {
        regime_counts.insert(r, 0);
    }

    for chunk in paths.chunks(batch_sz) {
        let chunk_paths: Vec<std::path::PathBuf> = chunk.to_vec();
        let chunk_result: std::result::Result<Vec<_>, _> = chunk_paths
            .iter()
            .map(|p| {
                let frame = state_cache::load_source_parquet(p)
                    .map_err(|e| format!("{}: {e}", p.display()))?;
                let bars = state_cache::bars_from_source_frame(&frame)
                    .map_err(|e| format!("{}: {e}", p.display()))?;
                state_cache::project_single_stock(&bars, warmup_bars)
                    .map_err(|e| format!("{}: {e}", p.display()))
            })
            .collect();

        let parts = chunk_result
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        for part in parts {
            let n = part.height();
            total_rows += n;
            source_count += 1;
            if let Ok(regimes) = part.column("regime").and_then(|c| c.i8()) {
                for i in 0..regimes.len() {
                    if let Some(r) = regimes.get(i) {
                        *regime_counts.entry(r as u8).or_insert(0) += 1;
                    }
                }
            }
            all_parts.push(part);
        }
    }

    let mut merged = all_parts[0].clone();
    for part in all_parts.into_iter().skip(1) {
        merged = merged.vstack(&part)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vstack: {e}")))?;
    }
    merged = merged.sort(["symbol", "dt"], SortMultipleOptions::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("sort: {e}")))?;

    let mut file = fs::File::create(out_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("cannot create output: {e}")))?;
    ParquetWriter::new(&mut file)
        .finish(&mut merged)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet write: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("output_path", output_path)?;
    dict.set_item("source_count", source_count)?;
    dict.set_item("total_rows", total_rows)?;
    dict.set_item("schema_version", state_cache::SCHEMA_VERSION)?;

    let regime_dict = PyDict::new(py);
    for (regime, count) in &regime_counts {
        regime_dict.set_item(*regime, *count)?;
    }
    dict.set_item("regime_counts", regime_dict)?;
    Ok(dict.into_any().unbind())
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

    // Slot Backtest
    research.add_function(wrap_pyfunction!(simulate_surge_backtest, &research)?)?;
    research.add_function(wrap_pyfunction!(simulate_surge_backtest_parquet, &research)?)?;

    // Random Control / Excess
    research.add_function(wrap_pyfunction!(compute_surge_excess, &research)?)?;
    research.add_function(wrap_pyfunction!(compute_surge_excess_parquet, &research)?)?;

    // State Cache
    research.add_function(wrap_pyfunction!(build_state_cache, &research)?)?;

    let sys = py.import("sys")?;
    let py_modules = sys.getattr("modules")?;
    py_modules.set_item("czsc._native.research", &research)?;
    parent.add("research", &research)?;

    Ok(())
}
