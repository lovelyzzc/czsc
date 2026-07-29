//! 因果结构特征提取。

use czsc_core::objects::bi::BI;
use czsc_core::objects::zs::ZS;

use super::indicators::TrendIndicators;

#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods, IntoPyObject};
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// 单 bar 的因果结构特征。
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass(from_py_object, module = "czsc._native"))]
#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    pub up_dn_power_ratio: f64,
    pub last_up_angle: f64,
    pub ma_spread_pct: f64,
    pub dif: f64,
    pub vol_ratio: f64,
    pub n_pivots: usize,
    pub pivot_width_pct: f64,
    pub ret20: f64,
    pub above_zg: bool,
}

#[cfg(feature = "python")]
#[gen_stub_pymethods]
#[pymethods]
impl FeatureSnapshot {
    #[getter]
    fn up_dn_power_ratio(&self) -> f64 {
        self.up_dn_power_ratio
    }

    #[getter]
    fn last_up_angle(&self) -> f64 {
        self.last_up_angle
    }

    #[getter]
    fn ma_spread_pct(&self) -> f64 {
        self.ma_spread_pct
    }

    #[getter]
    fn dif(&self) -> f64 {
        self.dif
    }

    #[getter]
    fn vol_ratio(&self) -> f64 {
        self.vol_ratio
    }

    #[getter]
    fn n_pivots(&self) -> usize {
        self.n_pivots
    }

    #[getter]
    fn pivot_width_pct(&self) -> f64 {
        self.pivot_width_pct
    }

    #[getter]
    fn ret20(&self) -> f64 {
        self.ret20
    }

    #[getter]
    fn above_zg(&self) -> bool {
        self.above_zg
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureSnapshot(vol_ratio={:.2}, ma_spread={:.2}%, ret20={:.2}%)",
            self.vol_ratio, self.ma_spread_pct, self.ret20
        )
    }

    /// 转换为 Python dict（向后兼容旧 Python dict feats 格式）。
    fn to_dict(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyDict>> {
        use pyo3::types::{PyDict, PyDictMethods};
        let d = PyDict::new(py);
        d.set_item("up_dn_power_ratio", self.up_dn_power_ratio)?;
        d.set_item("last_up_angle", self.last_up_angle)?;
        d.set_item("ma_spread_pct", self.ma_spread_pct)?;
        d.set_item("dif", self.dif)?;
        d.set_item("vol_ratio", self.vol_ratio)?;
        d.set_item("n_pivots", self.n_pivots)?;
        d.set_item("pivot_width_pct", self.pivot_width_pct)?;
        d.set_item("ret20", self.ret20)?;
        d.set_item("above_zg", self.above_zg as i32)?;
        Ok(d.unbind())
    }

    /// 获取特征值（向后兼容旧 Python dict .get() 访问模式）。
    #[pyo3(name = "get")]
    fn py_get(&self, py: pyo3::Python<'_>, key: &str) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        let val: pyo3::Py<pyo3::PyAny> = match key {
            "up_dn_power_ratio" => self.up_dn_power_ratio.into_pyobject(py)?.into_any().unbind(),
            "last_up_angle" => self.last_up_angle.into_pyobject(py)?.into_any().unbind(),
            "ma_spread_pct" => self.ma_spread_pct.into_pyobject(py)?.into_any().unbind(),
            "dif" => self.dif.into_pyobject(py)?.into_any().unbind(),
            "vol_ratio" => self.vol_ratio.into_pyobject(py)?.into_any().unbind(),
            "n_pivots" => self.n_pivots.into_pyobject(py)?.into_any().unbind(),
            "pivot_width_pct" => self.pivot_width_pct.into_pyobject(py)?.into_any().unbind(),
            "ret20" => self.ret20.into_pyobject(py)?.into_any().unbind(),
            "above_zg" => (self.above_zg as i32).into_pyobject(py)?.into_any().unbind(),
            _ => py.None(),
        };
        Ok(val)
    }
}

/// 该 bar 的因果结构特征（全部仅用 ≤idx 数据）。
///
/// `up`/`dn` 由调用方一次性分区后传入，避免重复 O(n) 扫描。
pub fn compute_features(
    up: &[&BI],
    dn: &[&BI],
    zs_list: &[ZS],
    ind: &TrendIndicators,
    idx: usize,
) -> FeatureSnapshot {
    let zs = zs_list.last();
    let ma5 = ind.ma5[idx];
    let ma20 = ind.ma20[idx];

    let up_pow = if up.is_empty() {
        0.0
    } else {
        let tail: Vec<f64> = up.iter().rev().take(3).map(|b| b.get_power()).collect();
        tail.iter().sum::<f64>() / tail.len() as f64
    };
    let dn_pow = if dn.is_empty() {
        0.0
    } else {
        let tail: Vec<f64> = dn.iter().rev().take(3).map(|b| b.get_power()).collect();
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    let vol_ratio = if idx >= 20 {
        let window = &ind.vol[idx - 20..idx];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        if mean > 0.0 {
            ind.vol[idx] / mean
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    FeatureSnapshot {
        up_dn_power_ratio: if dn_pow > 0.0 {
            round3(up_pow / dn_pow)
        } else {
            f64::NAN
        },
        last_up_angle: if !up.is_empty() {
            round2(up.last().unwrap().get_angle())
        } else {
            f64::NAN
        },
        ma_spread_pct: if !ma20.is_nan() && ma20 > 0.0 && !ma5.is_nan() {
            round2((ma5 - ma20) / ma20 * 100.0)
        } else {
            f64::NAN
        },
        dif: round4(ind.dif[idx]),
        vol_ratio: if vol_ratio.is_nan() {
            f64::NAN
        } else {
            round2(vol_ratio)
        },
        n_pivots: zs_list.len(),
        pivot_width_pct: match zs {
            Some(z) if z.zd > 0.0 => round2((z.zg - z.zd) / z.zd * 100.0),
            _ => f64::NAN,
        },
        ret20: round2(ind.ret20[idx]),
        above_zg: zs.is_some() && ind.close[idx] > zs.unwrap().zg,
    }
}

#[inline]
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[inline]
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[inline]
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}
