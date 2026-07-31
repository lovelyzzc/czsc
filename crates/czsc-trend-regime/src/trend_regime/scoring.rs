//! 主升浪启动检测 + 强度/优先级评分。

use super::features::FeatureSnapshot;
use super::{Regime, SURGE_GATE_MA_SPREAD, SURGE_GATE_RET20, SURGE_GATE_VOL_RATIO};

// ─── 门控方向 ────────────────────────────────────────────────────

/// 门控条件的比较方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateDirection {
    /// 特征值 >= 阈值视为通过（原始行为）。
    Gte,
    /// 特征值 <= 阈值视为通过（翻转方向）。
    Lte,
}

impl GateDirection {
    pub fn from_str(s: &str) -> Self {
        match s {
            "lte" | "le" | "<=" => Self::Lte,
            _ => Self::Gte,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gte => "gte",
            Self::Lte => "lte",
        }
    }

    #[inline]
    pub(crate) fn passes(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Gte => value >= threshold,
            Self::Lte => value <= threshold,
        }
    }

    #[inline]
    pub(crate) fn score(self, value: f64, threshold: f64) -> f64 {
        match self {
            Self::Gte => (value / threshold).clamp(0.0, 2.0) / 2.0,
            Self::Lte => {
                if threshold <= 0.0 || value <= 0.0 {
                    return 0.0;
                }
                (threshold / value).clamp(0.0, 2.0) / 2.0
            }
        }
    }
}

// ─── 门控配置 ────────────────────────────────────────────────────

/// 可配置的门控参数：阈值 + 方向，支持网格搜索优化。
#[derive(Debug, Clone)]
pub struct GateConfig {
    pub vol_ratio_threshold: f64,
    pub vol_ratio_direction: GateDirection,
    pub ma_spread_threshold: f64,
    pub ma_spread_direction: GateDirection,
    pub use_above_zg: bool,
    pub ret20_threshold: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            vol_ratio_threshold: SURGE_GATE_VOL_RATIO,
            vol_ratio_direction: GateDirection::Gte,
            ma_spread_threshold: SURGE_GATE_MA_SPREAD,
            ma_spread_direction: GateDirection::Gte,
            use_above_zg: true,
            ret20_threshold: SURGE_GATE_RET20,
        }
    }
}

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

/// 可配置门控分层：按 `GateConfig` 中的阈值和方向判断。
pub fn classify_gate_level_with_config(feats: &FeatureSnapshot, cfg: &GateConfig) -> GateLevel {
    let vr_ok = !feats.vol_ratio.is_nan()
        && cfg
            .vol_ratio_direction
            .passes(feats.vol_ratio, cfg.vol_ratio_threshold);
    let sp_ok = !feats.ma_spread_pct.is_nan()
        && cfg
            .ma_spread_direction
            .passes(feats.ma_spread_pct, cfg.ma_spread_threshold);
    let zg_ok = if cfg.use_above_zg {
        feats.above_zg
    } else {
        true
    };
    let pass_count = vr_ok as u8 + sp_ok as u8 + zg_ok as u8;
    GateLevel::from_u8(pass_count)
}

/// 三维门控各自的通过状态，然后汇总为 [`GateLevel`]（使用默认阈值）。
pub fn classify_gate_level(feats: &FeatureSnapshot) -> GateLevel {
    classify_gate_level_with_config(feats, &GateConfig::default())
}

/// 可配置的连续置信度 0.0–1.0，支持方向翻转。
pub fn gate_confidence_with_config(feats: &FeatureSnapshot, cfg: &GateConfig) -> f64 {
    let vr = feats.vol_ratio;
    let sp = feats.ma_spread_pct;

    let vr_score = if vr.is_nan() || vr <= 0.0 {
        0.0
    } else {
        cfg.vol_ratio_direction
            .score(vr, cfg.vol_ratio_threshold)
    };
    let sp_score = if sp.is_nan() || sp <= 0.0 {
        0.0
    } else {
        cfg.ma_spread_direction
            .score(sp, cfg.ma_spread_threshold)
    };
    let zg_score = if cfg.use_above_zg {
        if feats.above_zg {
            1.0
        } else {
            0.0
        }
    } else {
        1.0
    };

    (vr_score * 0.4 + sp_score * 0.35 + zg_score * 0.25).min(1.0)
}

/// 连续置信度 0.0–1.0（使用默认阈值和方向）。
pub fn gate_confidence(feats: &FeatureSnapshot) -> f64 {
    gate_confidence_with_config(feats, &GateConfig::default())
}

/// 原始主升浪门控（全部 3 条件通过）。
///
/// 语义等价于 `classify_gate_level(feats) == GateLevel::Full`，
/// 保留为独立函数以保持向后兼容。
pub fn gates_pass(feats: &FeatureSnapshot) -> bool {
    classify_gate_level(feats) == GateLevel::Full
}

// ─── 主升浪启动检测 ──────────────────────────────────────────────

/// 因果「主升浪启动」检测，支持可配置的最低门控层级和门控参数。
pub fn surge_onset_with_config(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    mode: &str,
    min_level: GateLevel,
    cfg: &GateConfig,
) -> bool {
    let prior: std::collections::HashSet<u8> = prior_regimes.iter().copied().collect();
    let feats = match feats {
        Some(f) => f,
        None => return false,
    };

    let gate_ok = classify_gate_level_with_config(feats, cfg) >= min_level;

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
            let ret_ok = feats.ret20 >= cfg.ret20_threshold;
            entered && path_ok && ret_ok && gate_ok
        }
        _ => false,
    }
}

/// 因果「主升浪启动」检测，支持可配置的最低门控层级（使用默认阈值）。
pub fn surge_onset_with_level(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    mode: &str,
    min_level: GateLevel,
) -> bool {
    surge_onset_with_config(prev, regime, feats, prior_regimes, mode, min_level, &GateConfig::default())
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

// ─── 均值回复信号 ────────────────────────────────────────────────

/// 均值回复启动检测：买入"正在从下跌态脱离"的股票。
///
/// 触发条件：
/// - `prev_regime` 为 Downtrend(1)
/// - `regime` 转入 PivotBuilding(4) / FirstBuy(2) / SecondBuy(3) 之一
/// - 下跌态持续时间在 `min_down_days..=max_down_days` 范围内
/// - 如果提供 `feats`，优先选低 vol_ratio（Q0-Q2 对应正超额）
pub fn reversion_onset(
    prev: u8,
    regime: u8,
    feats: Option<&FeatureSnapshot>,
    prior_regimes: &[u8],
    min_down_days: usize,
    max_down_days: usize,
) -> bool {
    let from_downtrend = prev == Regime::Downtrend as u8;
    let to_recovery = regime == Regime::PivotBuilding as u8
        || regime == Regime::FirstBuy as u8
        || regime == Regime::SecondBuy as u8;

    if !from_downtrend || !to_recovery {
        return false;
    }

    let down_days = prior_regimes
        .iter()
        .rev()
        .take_while(|&&r| r == Regime::Downtrend as u8)
        .count();

    if down_days < min_down_days || down_days > max_down_days {
        return false;
    }

    if let Some(f) = feats {
        if f.vol_ratio.is_nan() {
            return false;
        }
    }

    true
}

/// 均值回复评分 0..100。
///
/// 权重设计（与主升浪相反）：
/// - 低 vol_ratio → 高分（缩量企稳优于放量）
/// - 下跌态持续天数在甜点区间 → 加分
/// - ret20 为负且绝对值适中 → 均值回复空间大
pub fn reversion_score(feats: Option<&FeatureSnapshot>, down_days: usize) -> f64 {
    let feats = match feats {
        Some(f) => f,
        None => return 0.0,
    };

    let v = |x: f64, default: f64| -> f64 {
        if x.is_nan() {
            default
        } else {
            x
        }
    };

    let vr = v(feats.vol_ratio, 1.0);
    let vr_score = (2.0 - vr).clamp(0.0, 2.0) / 2.0 * 35.0;

    let down_score = if (5..=20).contains(&down_days) {
        let optimal = 10.0_f64;
        let dist = (down_days as f64 - optimal).abs();
        (1.0 - dist / 10.0).clamp(0.0, 1.0) * 25.0
    } else {
        5.0
    };

    let ret = v(feats.ret20, 0.0);
    let ret_score = if ret < 0.0 {
        (ret.abs() / 30.0).clamp(0.0, 1.0) * 20.0
    } else {
        5.0
    };

    let spread = v(feats.ma_spread_pct, 0.0);
    let spread_score = if spread < 0.0 {
        (spread.abs() / 10.0).clamp(0.0, 1.0) * 20.0
    } else {
        5.0
    };

    let s = vr_score + down_score + ret_score + spread_score;
    (s.clamp(0.0, 100.0) * 10.0).round() / 10.0
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

// ─── 市场环境分类 ────────────────────────────────────────────────

/// 市场环境分类：基于市场指数（等权/宽基）的中期收益率。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MarketRegime {
    Bull = 0,
    Sideways = 1,
    Bear = 2,
}

impl MarketRegime {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Bull,
            1 => Self::Sideways,
            _ => Self::Bear,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bull => "bull",
            Self::Sideways => "sideways",
            Self::Bear => "bear",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "bull" => Self::Bull,
            "sideways" => Self::Sideways,
            "bear" => Self::Bear,
            _ => Self::Sideways,
        }
    }
}

/// 基于市场指数收盘价序列的 60 日收益率分类市场环境。
///
/// - `index_closes`：市场指数收盘价序列（等权/沪深300等）
/// - `lookback`：回看天数（默认 60）
/// - `bull_threshold`：牛市阈值（收益率 %，默认 10.0）
/// - `bear_threshold`：熊市阈值（收益率 %，默认 -10.0）
///
/// 返回与 `index_closes` 等长的 `MarketRegime` 序列，前 `lookback` 根为 Sideways。
pub fn classify_market_regime(
    index_closes: &[f64],
    lookback: usize,
    bull_threshold: f64,
    bear_threshold: f64,
) -> Vec<MarketRegime> {
    let n = index_closes.len();
    let mut out = vec![MarketRegime::Sideways; n];
    if n <= lookback {
        return out;
    }

    for i in lookback..n {
        let prev = index_closes[i - lookback];
        let curr = index_closes[i];
        if prev <= 0.0 || prev.is_nan() || curr.is_nan() {
            continue;
        }
        let ret_pct = (curr / prev - 1.0) * 100.0;
        out[i] = if ret_pct >= bull_threshold {
            MarketRegime::Bull
        } else if ret_pct <= bear_threshold {
            MarketRegime::Bear
        } else {
            MarketRegime::Sideways
        };
    }
    out
}

/// 对日期-指数收盘价列表分类市场环境。
///
/// 返回 `(date, MarketRegime)` 的列表。
pub fn classify_market_regime_series(
    dates: &[chrono::DateTime<chrono::Utc>],
    index_closes: &[f64],
    lookback: usize,
    bull_threshold: f64,
    bear_threshold: f64,
) -> Vec<(chrono::DateTime<chrono::Utc>, MarketRegime)> {
    let regimes = classify_market_regime(index_closes, lookback, bull_threshold, bear_threshold);
    dates.iter().copied().zip(regimes).collect()
}
