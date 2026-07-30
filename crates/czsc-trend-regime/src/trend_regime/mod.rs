//! 走势类型划分（11 态）—— 因果安全的缠论状态机。
//!
//! 把个股日线走势分类为 0..10 共 11 个走势类型，对应 [`Regime`] 枚举。
//! 所有函数仅使用 ≤ 当前 bar 的数据，确保因果安全（无未来函数）。

mod features;
mod fsm;
mod indicators;
mod scoring;

pub use features::{compute_features, FeatureSnapshot};
pub use fsm::{classify_fsm, extract_zs_list, seed_regime};
pub use indicators::TrendIndicators;
pub use scoring::{
    classify_gate_level, gate_confidence, gates_pass, priority_score, surge_onset,
    surge_onset_with_level, surge_score, GateLevel,
};

use chrono::{DateTime, Utc};
#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

// ─── 模块级常量 ─────────────────────────────────────────────────
pub const MIN_BARS: usize = 500;
pub const WARMUP_BARS: usize = 120;
pub const MIN_BIS: usize = 6;
pub const ZS_MIN_BIS: usize = 3;
pub const ACCEL_SPREAD: f64 = 15.0;
pub const ACCEL_RET20: f64 = 15.0;
pub const SURGE_GATE_VOL_RATIO: f64 = 1.2;
pub const SURGE_GATE_MA_SPREAD: f64 = 3.0;
pub const SURGE_GATE_RET20: f64 = 8.0;

// ─── Regime 枚举 ────────────────────────────────────────────────

/// 11 种走势类型。数值即编号 0..10。
#[cfg_attr(feature = "python", gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(from_py_object, eq, eq_int, frozen, hash, module = "czsc._native"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Regime {
    NotTradable = 0,
    Downtrend = 1,
    FirstBuy = 2,
    SecondBuy = 3,
    PivotBuilding = 4,
    UpwardDeparture = 5,
    ThirdBuy = 6,
    MainUptrend = 7,
    Acceleration = 8,
    Divergence = 9,
    Breakdown = 10,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl Regime {
    /// 从整数构造 Regime。
    #[staticmethod]
    #[pyo3(name = "from_int")]
    fn py_from_int(v: u8) -> Self {
        Self::from_u8(v)
    }

    fn __int__(&self) -> u8 {
        *self as u8
    }

    fn __repr__(&self) -> String {
        format!("Regime.{:?}", self)
    }

    /// 迭代所有 Regime 变体（兼容 `for r in Regime` 用法）。
    #[classmethod]
    #[gen_stub(skip)]
    fn __iter__(_cls: &pyo3::Bound<'_, pyo3::types::PyType>) -> RegimeIter {
        RegimeIter { idx: 0 }
    }
}

#[cfg(feature = "python")]
#[pyclass(module = "czsc._native", skip_from_py_object)]
struct RegimeIter {
    idx: u8,
}

#[cfg(feature = "python")]
#[pymethods]
impl RegimeIter {
    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Regime> {
        if self.idx <= 10 {
            let r = Regime::from_u8(self.idx);
            self.idx += 1;
            Some(r)
        } else {
            None
        }
    }
}

impl Regime {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::NotTradable,
            1 => Self::Downtrend,
            2 => Self::FirstBuy,
            3 => Self::SecondBuy,
            4 => Self::PivotBuilding,
            5 => Self::UpwardDeparture,
            6 => Self::ThirdBuy,
            7 => Self::MainUptrend,
            8 => Self::Acceleration,
            9 => Self::Divergence,
            10 => Self::Breakdown,
            _ => Self::NotTradable,
        }
    }
}

// ─── StateSnapshot ──────────────────────────────────────────────

/// 单个 bar 的因果分类快照（全部字段仅含 ≤idx 信息）。
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass(from_py_object, module = "czsc._native"))]
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub idx: usize,
    pub dt: DateTime<Utc>,
    pub regime: u8,
    pub prev_regime: u8,
    pub close: f64,
    pub next_open: f64,
    pub sl_ref: f64,
    pub zg: f64,
    pub zd: f64,
    pub feats: Option<FeatureSnapshot>,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl StateSnapshot {
    #[getter]
    fn idx(&self) -> usize {
        self.idx
    }

    #[getter]
    fn dt(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        czsc_core::utils::common::create_naive_pandas_timestamp(py, self.dt)
    }

    #[getter]
    fn regime(&self) -> u8 {
        self.regime
    }

    #[getter]
    fn prev_regime(&self) -> u8 {
        self.prev_regime
    }

    #[getter]
    fn close(&self) -> f64 {
        self.close
    }

    #[getter]
    fn next_open(&self) -> f64 {
        self.next_open
    }

    #[getter]
    fn sl_ref(&self) -> f64 {
        self.sl_ref
    }

    #[getter]
    fn zg(&self) -> f64 {
        self.zg
    }

    #[getter]
    fn zd(&self) -> f64 {
        self.zd
    }

    #[getter]
    fn feats(&self) -> Option<FeatureSnapshot> {
        self.feats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateSnapshot(idx={}, regime={}, close={:.2})",
            self.idx, self.regime, self.close
        )
    }
}

// ─── bi_list 变更检测缓存 ────────────────────────────────────────

use czsc_core::analyze::CZSC;
use czsc_core::objects::bar::RawBar;
use czsc_core::objects::direction::Direction;
use czsc_core::objects::freq::Freq;
use czsc_core::objects::zs::ZS;

/// 缓存 `extract_zs_list` 结果和 dn_last_low，仅当 bi_list 发生变更时重算。
///
/// 指纹 `(bi_count, last_bi_edt)` 可捕获 push / pop / drain 全部变更场景。
struct BiCache {
    bi_count: usize,
    last_bi_edt: Option<DateTime<Utc>>,
    zs_list: Vec<ZS>,
    dn_last_low: f64,
}

impl BiCache {
    fn new() -> Self {
        Self {
            bi_count: 0,
            last_bi_edt: None,
            zs_list: Vec::new(),
            dn_last_low: f64::NAN,
        }
    }

    #[inline]
    fn needs_update(&self, bis: &[czsc_core::objects::bi::BI]) -> bool {
        let count = bis.len();
        let edt = bis.last().map(|b| b.end_dt());
        count != self.bi_count || edt != self.last_bi_edt
    }

    fn update(&mut self, bis: &[czsc_core::objects::bi::BI]) {
        self.bi_count = bis.len();
        self.last_bi_edt = bis.last().map(|b| b.end_dt());
        self.zs_list = extract_zs_list(bis);
        self.dn_last_low = bis
            .iter()
            .rev()
            .find(|b| b.direction == Direction::Down)
            .map_or(f64::NAN, |b| b.get_low());
    }
}

// ─── iter_states 主入口 ─────────────────────────────────────────

/// 流式重放整只个股，返回每个 bar 的因果状态快照。
///
/// - `with_features`：为每个可交易 bar 附 `feats`。
/// - `tail`：仅暖机到 `n-tail`、流式分类最后 `tail` 根（选股快路径）；
///   `None` 则全程从 warmup 开始（回测用）。
/// - `limit_pct`：涨跌停判定阈值。
///
/// 性能优化：
/// 1. `BiCache` —— 仅当 bi_list 发生变化时重算中枢列表（跳过 ~80% 的 bar）
/// 2. 单遍 up/dn 分区 —— 消除 `classify_fsm` + `compute_features` 的重复扫描
pub fn iter_states(
    bars: &[RawBar],
    _freq: Freq,
    with_features: bool,
    tail: Option<usize>,
    limit_pct: f64,
    warmup_bars: usize,
    min_bis: usize,
) -> Vec<StateSnapshot> {
    let ind = TrendIndicators::from_bars(bars);
    let n = ind.n;
    if n <= warmup_bars {
        return Vec::new();
    }

    let start = match tail {
        None => warmup_bars,
        Some(t) => warmup_bars.max(n.saturating_sub(t)),
    };

    let mut czsc = CZSC::new(bars[..start].to_vec(), 50, 6);
    let mut out: Vec<StateSnapshot> = Vec::with_capacity(n - start);
    let mut prev = Regime::NotTradable;
    let mut seeded = false;
    let mut cache = BiCache::new();

    for idx in start..n {
        czsc.update_bar(bars[idx].clone());
        let bis = &czsc.bi_list;

        let pct = ind.pct_chg[idx];
        let not_tradable =
            bis.len() < min_bis || pct.abs() >= limit_pct || ind.vol[idx] <= 0.0;

        let (regime, dn_last_low, feats, snap_zg, snap_zd) = if not_tradable {
            (Regime::NotTradable, f64::NAN, None, f64::NAN, f64::NAN)
        } else {
            if cache.needs_update(bis) {
                cache.update(bis);
            }

            let (up, dn): (Vec<&_>, Vec<&_>) =
                bis.iter().partition(|b| b.direction == Direction::Up);
            let last_bi = bis.last().unwrap();

            if !seeded {
                prev = seed_regime(bis, &cache.zs_list, &ind, idx);
                seeded = true;
            }
            let regime = classify_fsm(prev, &up, &dn, last_bi, &cache.zs_list, &ind, idx);
            let feats = if with_features {
                Some(compute_features(&up, &dn, &cache.zs_list, &ind, idx))
            } else {
                None
            };
            let zs = cache.zs_list.last();
            (
                regime,
                cache.dn_last_low,
                feats,
                zs.map_or(f64::NAN, |z| z.zg),
                zs.map_or(f64::NAN, |z| z.zd),
            )
        };

        out.push(StateSnapshot {
            idx,
            dt: ind.dates[idx],
            regime: regime as u8,
            prev_regime: prev as u8,
            close: ind.close[idx],
            next_open: if idx + 1 < n { ind.open[idx + 1] } else { f64::NAN },
            sl_ref: dn_last_low,
            zg: snap_zg,
            zd: snap_zd,
            feats,
        });

        if regime != Regime::NotTradable {
            prev = regime;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_from_u8_roundtrip() {
        for v in 0..=10u8 {
            let r = Regime::from_u8(v);
            assert_eq!(r as u8, v, "from_u8({v}) roundtrip failed");
        }
        assert_eq!(Regime::from_u8(255), Regime::NotTradable);
    }

    #[test]
    fn test_surge_score_none_feats() {
        assert_eq!(surge_score(None), 0.0);
    }

    #[test]
    fn test_surge_score_basic() {
        let f = FeatureSnapshot {
            up_dn_power_ratio: 1.5,
            last_up_angle: 30.0,
            ma_spread_pct: 10.0,
            dif: 0.5,
            vol_ratio: 1.5,
            n_pivots: 2,
            pivot_width_pct: 5.0,
            ret20: 15.0,
            above_zg: true,
        };
        let s = surge_score(Some(&f));
        assert!(s > 0.0 && s <= 100.0, "score={s}");
    }

    #[test]
    fn test_priority_score_nan_sl() {
        let p = priority_score(50.0, f64::NAN, 0, 7, 10);
        assert!(p >= 0.0, "priority_score with NaN sl should not be negative: {p}");
    }

    #[test]
    fn test_surge_onset_no_feats() {
        assert!(!surge_onset(1, 7, None, &[], "confirm"));
        assert!(!surge_onset(1, 5, None, &[], "anticipate"));
    }

    #[test]
    fn test_gates_pass_basic() {
        let f = FeatureSnapshot {
            up_dn_power_ratio: 1.5,
            last_up_angle: 30.0,
            ma_spread_pct: 5.0,
            dif: 0.5,
            vol_ratio: 1.5,
            n_pivots: 2,
            pivot_width_pct: 5.0,
            ret20: 15.0,
            above_zg: true,
        };
        assert!(gates_pass(&f));

        let f_low_vol = FeatureSnapshot {
            vol_ratio: 0.5,
            ..f.clone()
        };
        assert!(!gates_pass(&f_low_vol));
    }

    #[test]
    fn test_classify_gate_level() {
        let full = FeatureSnapshot {
            up_dn_power_ratio: 1.5,
            last_up_angle: 30.0,
            ma_spread_pct: 5.0,
            dif: 0.5,
            vol_ratio: 1.5,
            n_pivots: 2,
            pivot_width_pct: 5.0,
            ret20: 15.0,
            above_zg: true,
        };
        assert_eq!(classify_gate_level(&full), GateLevel::Full);
        assert!(gates_pass(&full));

        // 2/3: low vol → Partial
        let partial = FeatureSnapshot { vol_ratio: 0.5, ..full.clone() };
        assert_eq!(classify_gate_level(&partial), GateLevel::Partial);
        assert!(!gates_pass(&partial));

        // 1/3: low vol + low spread → Weak
        let weak = FeatureSnapshot {
            vol_ratio: 0.5,
            ma_spread_pct: 1.0,
            ..full.clone()
        };
        assert_eq!(classify_gate_level(&weak), GateLevel::Weak);

        // 0/3: everything fails → None
        let none = FeatureSnapshot {
            vol_ratio: 0.5,
            ma_spread_pct: 1.0,
            above_zg: false,
            ..full.clone()
        };
        assert_eq!(classify_gate_level(&none), GateLevel::None);
    }

    #[test]
    fn test_gate_level_ordering() {
        assert!(GateLevel::Full > GateLevel::Partial);
        assert!(GateLevel::Partial > GateLevel::Weak);
        assert!(GateLevel::Weak > GateLevel::None);
    }

    #[test]
    fn test_gate_confidence_range() {
        let f = FeatureSnapshot {
            up_dn_power_ratio: 1.5,
            last_up_angle: 30.0,
            ma_spread_pct: 5.0,
            dif: 0.5,
            vol_ratio: 1.5,
            n_pivots: 2,
            pivot_width_pct: 5.0,
            ret20: 15.0,
            above_zg: true,
        };
        let c = gate_confidence(&f);
        assert!(c > 0.0 && c <= 1.0, "confidence={c}");

        let f_zero = FeatureSnapshot {
            vol_ratio: 0.0,
            ma_spread_pct: 0.0,
            above_zg: false,
            ..f.clone()
        };
        assert_eq!(gate_confidence(&f_zero), 0.0);

        // NaN fields should yield 0 contribution
        let f_nan = FeatureSnapshot {
            vol_ratio: f64::NAN,
            ma_spread_pct: f64::NAN,
            above_zg: false,
            ..f
        };
        assert_eq!(gate_confidence(&f_nan), 0.0);
    }

    #[test]
    fn test_gate_level_str_roundtrip() {
        for level in [GateLevel::None, GateLevel::Weak, GateLevel::Partial, GateLevel::Full] {
            assert_eq!(GateLevel::from_str(level.as_str()), level);
        }
    }

    #[test]
    fn test_surge_onset_with_level_relaxed() {
        let f = FeatureSnapshot {
            up_dn_power_ratio: 1.5,
            last_up_angle: 30.0,
            ma_spread_pct: 5.0,
            dif: 0.5,
            vol_ratio: 0.8, // below threshold → only Partial
            n_pivots: 2,
            pivot_width_pct: 5.0,
            ret20: 15.0,
            above_zg: true,
        };
        let prior = vec![
            Regime::PivotBuilding as u8,
            Regime::UpwardDeparture as u8,
        ];
        // Full level should reject (vol_ratio < 1.2)
        assert!(!surge_onset(
            Regime::UpwardDeparture as u8,
            Regime::MainUptrend as u8,
            Some(&f),
            &prior,
            "confirm",
        ));
        // Partial level should accept
        assert!(surge_onset_with_level(
            Regime::UpwardDeparture as u8,
            Regime::MainUptrend as u8,
            Some(&f),
            &prior,
            "confirm",
            GateLevel::Partial,
        ));
    }
}
