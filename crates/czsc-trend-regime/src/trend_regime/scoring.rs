//! 主升浪启动检测 + 强度/优先级评分。

use super::features::FeatureSnapshot;
use super::{Regime, SURGE_GATE_MA_SPREAD, SURGE_GATE_RET20, SURGE_GATE_VOL_RATIO};

// ─── 分层门控 ────────────────────────────────────────────────────

/// 门控通过层级：`Full`(3/3) > `Partial`(2/3) > `Weak`(1/3) > `None`(0/3)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum GateLevel {
    None = 0,
    Weak = 1,
    Partial = 2,
    Full = 3,
}

impl GateLevel {
    pub fn from_u8(v: u8) -> Self {
        match v {
            3 => Self::Full,
            2 => Self::Partial,
            1 => Self::Weak,
            _ => Self::None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "full" => Self::Full,
            "partial" => Self::Partial,
            "weak" => Self::Weak,
            _ => Self::None,
        }
    }
}

/// 三维门控各自的通过状态，然后汇总为 [`GateLevel`]。
pub fn classify_gate_level(feats: &FeatureSnapshot) -> GateLevel {
    let vr_ok = !feats.vol_ratio.is_nan() && feats.vol_ratio >= SURGE_GATE_VOL_RATIO;
    let sp_ok = !feats.ma_spread_pct.is_nan() && feats.ma_spread_pct >= SURGE_GATE_MA_SPREAD;
    let zg_ok = feats.above_zg;
    let pass_count = vr_ok as u8 + sp_ok as u8 + zg_ok as u8;
    GateLevel::from_u8(pass_count)
}

/// 连续置信度 0.0–1.0，将三个门控条件映射为归一化分数后加权。
///
/// - `vol_ratio`:  `(value / threshold).clamp(0, 2) / 2` — 权重 0.4
/// - `ma_spread`:  `(value / threshold).clamp(0, 2) / 2` — 权重 0.35
/// - `above_zg`:   1.0 / 0.0                              — 权重 0.25
pub fn gate_confidence(feats: &FeatureSnapshot) -> f64 {
    let vr = feats.vol_ratio;
    let sp = feats.ma_spread_pct;

    let vr_score = if vr.is_nan() || vr <= 0.0 {
        0.0
    } else {
        (vr / SURGE_GATE_VOL_RATIO).min(2.0) / 2.0
    };
    let sp_score = if sp.is_nan() || sp <= 0.0 {
        0.0
    } else {
        (sp / SURGE_GATE_MA_SPREAD).min(2.0) / 2.0
    };
    let zg_score = if feats.above_zg { 1.0 } else { 0.0 };

    (vr_score * 0.4 + sp_score * 0.35 + zg_score * 0.25).min(1.0)
}

/// 原始主升浪门控（全部 3 条件通过）。
///
/// 语义等价于 `classify_gate_level(feats) == GateLevel::Full`，
/// 保留为独立函数以保持向后兼容。
pub fn gates_pass(feats: &FeatureSnapshot) -> bool {
    classify_gate_level(feats) == GateLevel::Full
}

// ─── 主升浪启动检测 ──────────────────────────────────────────────

/// 因果「主升浪启动」检测，支持可配置的最低门控层级。
pub fn surge_onset_with_level(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    mode: &str,
    min_level: GateLevel,
) -> bool {
    let prior: std::collections::HashSet<u8> = prior_regimes.iter().copied().collect();
    let feats = match feats {
        Some(f) => f,
        None => return false,
    };

    let gate_ok = classify_gate_level(feats) >= min_level;

    match mode {
        "confirm" => {
            let entered = prev != Regime::MainUptrend as u8
                && prev != Regime::Acceleration as u8
                && (regime == Regime::MainUptrend as u8 || regime == Regime::Acceleration as u8);
            let path_ok = prior.contains(&(Regime::PivotBuilding as u8))
                && prior.contains(&(Regime::UpwardDeparture as u8));
            entered && path_ok && gate_ok
        }
        "anticipate" => {
            let entered =
                prev != Regime::UpwardDeparture as u8 && regime == Regime::UpwardDeparture as u8;
            let path_ok = prior.contains(&(Regime::PivotBuilding as u8));
            let ret_ok = feats.ret20 >= SURGE_GATE_RET20;
            entered && path_ok && ret_ok && gate_ok
        }
        _ => false,
    }
}

/// 因果「主升浪启动」检测（仅用 ≤t 数据）。
///
/// 等价于 `surge_onset_with_level(..., GateLevel::Full)`，保留原始签名。
pub fn surge_onset(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    mode: &str,
) -> bool {
    surge_onset_with_level(prev, regime, feats, prior_regimes, mode, GateLevel::Full)
}

/// 主升浪强度打分 0..100。
pub fn surge_score(feats: Option<&FeatureSnapshot>) -> f64 {
    let feats = match feats {
        Some(f) => f,
        None => return 0.0,
    };

    let v = |x: f64, default: f64| -> f64 {
        if x.is_nan() { default } else { x }
    };

    let s = (v(feats.vol_ratio, 0.0) / 2.0).min(1.0) * 30.0
        + (v(feats.ma_spread_pct, 0.0) / 15.0).min(1.0) * 30.0
        + (v(feats.ret20, 0.0).max(0.0) / 30.0).min(1.0) * 20.0
        + (v(feats.last_up_angle, 0.0).max(0.0) / 45.0).min(1.0) * 20.0;
    (s * 10.0).round() / 10.0
}

const REGIME_QUALITY: [(Regime, f64); 4] = [
    (Regime::Acceleration, 20.0),
    (Regime::ThirdBuy, 18.0),
    (Regime::MainUptrend, 16.0),
    (Regime::UpwardDeparture, 12.0),
];
const STOP_FULL_BAND: (f64, f64) = (8.0, 20.0);
const STOP_PARTIAL_BAND: (f64, f64) = (5.0, 30.0);

/// 综合优先级 = 主升强度(35) + 止损可控(25) + 新鲜度(20) + 状态质量(20)。
pub fn priority_score(
    score: f64,
    sl_pct: f64,
    freshness: usize,
    regime: u8,
    scan_window: usize,
) -> f64 {
    let mut p = score.min(100.0) * 0.35;

    if sl_pct.is_nan() || sl_pct <= 0.0 {
        p -= 30.0;
    } else if sl_pct >= STOP_FULL_BAND.0 && sl_pct <= STOP_FULL_BAND.1 {
        p += 25.0;
    } else if (sl_pct >= STOP_PARTIAL_BAND.0 && sl_pct < STOP_FULL_BAND.0)
        || (sl_pct > STOP_FULL_BAND.1 && sl_pct <= STOP_PARTIAL_BAND.1)
    {
        p += 15.0;
    } else {
        p += 5.0;
    }

    p += (scan_window.saturating_sub(freshness) as f64 / scan_window as f64).max(0.0) * 20.0;

    let rq = REGIME_QUALITY
        .iter()
        .find(|(r, _)| *r as u8 == regime)
        .map(|(_, q)| *q)
        .unwrap_or(10.0);
    p += rq;

    (p.max(0.0) * 10.0).round() / 10.0
}
