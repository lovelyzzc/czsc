//! 主升浪启动检测 + 强度/优先级评分。

use super::features::FeatureSnapshot;
use super::{Regime, SURGE_GATE_MA_SPREAD, SURGE_GATE_RET20, SURGE_GATE_VOL_RATIO};

/// 主升浪启动的特征门控（放量 + 均线发散 + 立于中枢上方）。
pub fn gates_pass(feats: &FeatureSnapshot) -> bool {
    let vr = feats.vol_ratio;
    let sp = feats.ma_spread_pct;
    !vr.is_nan()
        && vr >= SURGE_GATE_VOL_RATIO
        && !sp.is_nan()
        && sp >= SURGE_GATE_MA_SPREAD
        && feats.above_zg
}

/// 因果「主升浪启动」检测（仅用 ≤t 数据）。
pub fn surge_onset(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    mode: &str,
) -> bool {
    let prior: std::collections::HashSet<u8> = prior_regimes.iter().copied().collect();
    let feats = match feats {
        Some(f) => f,
        None => return false,
    };

    match mode {
        "confirm" => {
            let entered = prev != Regime::MainUptrend as u8
                && prev != Regime::Acceleration as u8
                && (regime == Regime::MainUptrend as u8 || regime == Regime::Acceleration as u8);
            let path_ok = prior.contains(&(Regime::PivotBuilding as u8))
                && prior.contains(&(Regime::UpwardDeparture as u8));
            entered && path_ok && gates_pass(feats)
        }
        "anticipate" => {
            let entered =
                prev != Regime::UpwardDeparture as u8 && regime == Regime::UpwardDeparture as u8;
            let path_ok = prior.contains(&(Regime::PivotBuilding as u8));
            let ret_ok = feats.ret20 >= SURGE_GATE_RET20;
            entered && path_ok && ret_ok && gates_pass(feats)
        }
        _ => false,
    }
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
