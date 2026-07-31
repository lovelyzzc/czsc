//! czsc-trend-regime 的 PyO3 binding。通过 `python` feature 来开关。

use czsc_core::objects::{bar::RawBar, freq::Freq};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::trend_regime;

/// 流式重放个股日线，返回每个 bar 的因果走势类型状态快照。
///
/// 将整个 CZSC 暖机 + 逐 bar FSM 分类循环在 Rust 端完成，
/// 消除 Python-Rust 逐 bar 边界穿越。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (bars, freq=Freq::D, with_features=false, tail=None, limit_pct=9.8,
                    warmup_bars=120, min_bis=6))]
#[allow(clippy::too_many_arguments)]
fn iter_regime_states(
    bars: Vec<RawBar>,
    freq: Freq,
    with_features: bool,
    tail: Option<usize>,
    limit_pct: f64,
    warmup_bars: usize,
    min_bis: usize,
) -> Vec<trend_regime::StateSnapshot> {
    trend_regime::iter_states(&bars, freq, with_features, tail, limit_pct, warmup_bars, min_bis)
}

/// 因果「主升浪启动」检测。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (prev, regime, feats=None, prior_regimes=vec![], mode="confirm"))]
fn py_surge_onset(
    prev: u8,
    regime: u8,
    feats: Option<trend_regime::FeatureSnapshot>,
    prior_regimes: Vec<u8>,
    mode: &str,
) -> bool {
    trend_regime::surge_onset(prev, regime, feats.as_ref(), &prior_regimes, mode)
}

/// 因果「主升浪启动」检测，支持可配置的最低门控层级。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (prev, regime, feats=None, prior_regimes=vec![], mode="confirm", min_level="full"))]
fn py_surge_onset_with_level(
    prev: u8,
    regime: u8,
    feats: Option<trend_regime::FeatureSnapshot>,
    prior_regimes: Vec<u8>,
    mode: &str,
    min_level: &str,
) -> bool {
    let level = trend_regime::GateLevel::from_str(min_level);
    trend_regime::surge_onset_with_level(
        prev,
        regime,
        feats.as_ref(),
        &prior_regimes,
        mode,
        level,
    )
}

/// 可配置门控的主升浪启动检测。
///
/// `gate_config` 为 dict，可选键：
/// - `vol_ratio_threshold` (float, 默认 1.2)
/// - `vol_ratio_direction` (str: "gte"/"lte", 默认 "gte")
/// - `ma_spread_threshold` (float, 默认 3.0)
/// - `ma_spread_direction` (str: "gte"/"lte", 默认 "gte")
/// - `use_above_zg` (bool, 默认 true)
/// - `ret20_threshold` (float, 默认 8.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (prev, regime, feats=None, prior_regimes=vec![], mode="confirm",
                    min_level="full", gate_config=None))]
#[allow(clippy::too_many_arguments)]
fn py_surge_onset_with_config(
    prev: u8,
    regime: u8,
    feats: Option<trend_regime::FeatureSnapshot>,
    prior_regimes: Vec<u8>,
    mode: &str,
    min_level: &str,
    gate_config: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<bool> {
    let level = trend_regime::GateLevel::from_str(min_level);
    let cfg = parse_gate_config(gate_config)?;
    Ok(trend_regime::surge_onset_with_config(
        prev,
        regime,
        feats.as_ref(),
        &prior_regimes,
        mode,
        level,
        &cfg,
    ))
}

/// 三维门控分层：返回 "full" / "partial" / "weak" / "none"。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats,))]
fn py_classify_gate_level(feats: &trend_regime::FeatureSnapshot) -> &'static str {
    trend_regime::classify_gate_level(feats).as_str()
}

/// 可配置门控分层。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats, gate_config=None))]
fn py_classify_gate_level_with_config(
    feats: &trend_regime::FeatureSnapshot,
    gate_config: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<&'static str> {
    let cfg = parse_gate_config(gate_config)?;
    Ok(trend_regime::classify_gate_level_with_config(feats, &cfg).as_str())
}

/// 连续门控置信度 0.0–1.0。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats,))]
fn py_gate_confidence(feats: &trend_regime::FeatureSnapshot) -> f64 {
    trend_regime::gate_confidence(feats)
}

/// 可配置的连续门控置信度 0.0–1.0。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats, gate_config=None))]
fn py_gate_confidence_with_config(
    feats: &trend_regime::FeatureSnapshot,
    gate_config: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<f64> {
    let cfg = parse_gate_config(gate_config)?;
    Ok(trend_regime::gate_confidence_with_config(feats, &cfg))
}

/// 主升浪强度打分 0..100。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats=None))]
fn py_surge_score(feats: Option<trend_regime::FeatureSnapshot>) -> f64 {
    trend_regime::surge_score(feats.as_ref())
}

/// 综合优先级评分。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (score, sl_pct, freshness, regime, scan_window=10))]
fn py_priority_score(
    score: f64,
    sl_pct: f64,
    freshness: usize,
    regime: u8,
    scan_window: usize,
) -> f64 {
    trend_regime::priority_score(score, sl_pct, freshness, regime, scan_window)
}

/// 均值回复启动检测。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (prev, regime, feats=None, prior_regimes=vec![],
                    min_down_days=5, max_down_days=20))]
fn py_reversion_onset(
    prev: u8,
    regime: u8,
    feats: Option<trend_regime::FeatureSnapshot>,
    prior_regimes: Vec<u8>,
    min_down_days: usize,
    max_down_days: usize,
) -> bool {
    trend_regime::reversion_onset(
        prev,
        regime,
        feats.as_ref(),
        &prior_regimes,
        min_down_days,
        max_down_days,
    )
}

/// 均值回复评分 0..100。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (feats=None, down_days=0))]
fn py_reversion_score(feats: Option<trend_regime::FeatureSnapshot>, down_days: usize) -> f64 {
    trend_regime::reversion_score(feats.as_ref(), down_days)
}

/// 基于市场指数收盘价分类市场环境（bull/sideways/bear）。
///
/// 返回与 `index_closes` 等长的字符串列表。
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (index_closes, lookback=60, bull_threshold=10.0, bear_threshold=-10.0))]
fn py_classify_market_regime(
    index_closes: Vec<f64>,
    lookback: usize,
    bull_threshold: f64,
    bear_threshold: f64,
) -> Vec<&'static str> {
    trend_regime::classify_market_regime(&index_closes, lookback, bull_threshold, bear_threshold)
        .into_iter()
        .map(|r| r.as_str())
        .collect()
}

// ─── 辅助：从 Python dict 解析 GateConfig ─────────────────────────

fn parse_gate_config(
    dict: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<trend_regime::GateConfig> {
    use pyo3::types::PyDictMethods;
    let mut cfg = trend_regime::GateConfig::default();
    if let Some(d) = dict {
        if let Some(v) = d.get_item("vol_ratio_threshold")? {
            cfg.vol_ratio_threshold = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("vol_ratio_direction")? {
            let s: String = v.extract()?;
            cfg.vol_ratio_direction = trend_regime::GateDirection::from_str(&s);
        }
        if let Some(v) = d.get_item("ma_spread_threshold")? {
            cfg.ma_spread_threshold = v.extract::<f64>()?;
        }
        if let Some(v) = d.get_item("ma_spread_direction")? {
            let s: String = v.extract()?;
            cfg.ma_spread_direction = trend_regime::GateDirection::from_str(&s);
        }
        if let Some(v) = d.get_item("use_above_zg")? {
            cfg.use_above_zg = v.extract::<bool>()?;
        }
        if let Some(v) = d.get_item("ret20_threshold")? {
            cfg.ret20_threshold = v.extract::<f64>()?;
        }
    }
    Ok(cfg)
}

/// 在父 `_native` 模块上注册 `trend_regime` 子模块。
///
/// 同时将子模块注入 `sys.modules`，使
/// `from czsc._native.trend_regime import Regime` 等 dotted import 可用。
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let tr = PyModule::new(py, "trend_regime")?;
    tr.setattr("__name__", "czsc._native.trend_regime")?;

    tr.add_class::<trend_regime::Regime>()?;
    tr.add_class::<trend_regime::StateSnapshot>()?;
    tr.add_class::<trend_regime::FeatureSnapshot>()?;
    tr.add_function(wrap_pyfunction!(iter_regime_states, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_surge_onset, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_surge_onset_with_level, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_surge_onset_with_config, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_classify_gate_level, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_classify_gate_level_with_config, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_gate_confidence, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_gate_confidence_with_config, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_surge_score, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_priority_score, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_reversion_onset, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_reversion_score, &tr)?)?;
    tr.add_function(wrap_pyfunction!(py_classify_market_regime, &tr)?)?;

    let sys = py.import("sys")?;
    let py_modules = sys.getattr("modules")?;
    py_modules.set_item("czsc._native.trend_regime", &tr)?;
    parent.add("trend_regime", &tr)?;

    Ok(())
}
