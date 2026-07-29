//! 中枢提取 + 11 态 FSM 状态分类。

use czsc_core::objects::bi::BI;
use czsc_core::objects::direction::Direction;
use czsc_core::objects::zs::ZS;

use super::indicators::TrendIndicators;
use super::{ACCEL_RET20, ACCEL_SPREAD, Regime, ZS_MIN_BIS};

// ─── 中枢提取 ───────────────────────────────────────────────────

/// 把笔列表切成非重叠中枢，按时间升序返回。
pub fn extract_zs_list(bis: &[BI]) -> Vec<ZS> {
    let mut zs_list = Vec::new();
    let n = bis.len();
    let mut i = 0;
    while i + ZS_MIN_BIS <= n {
        let zs = ZS::new(bis[i..i + ZS_MIN_BIS].to_vec());
        if !zs.is_valid() {
            i += 1;
            continue;
        }
        let mut k = i + ZS_MIN_BIS;
        let mut best_zs = zs;
        while k < n {
            let grown = ZS::new(bis[i..=k].to_vec());
            if grown.is_valid() {
                best_zs = grown;
                k += 1;
            } else {
                break;
            }
        }
        zs_list.push(best_zs);
        i = k;
    }
    zs_list
}

// ─── 辅助 ───────────────────────────────────────────────────────

#[inline]
fn is_up(bi: &BI) -> bool {
    bi.direction == Direction::Up
}

fn dif_extreme_over(
    dates: &[chrono::DateTime<chrono::Utc>],
    dif: &[f64],
    sdt: chrono::DateTime<chrono::Utc>,
    edt: chrono::DateTime<chrono::Utc>,
    want_max: bool,
) -> f64 {
    let lo = dates.partition_point(|d| *d < sdt);
    let hi = dates.partition_point(|d| *d <= edt);
    if hi <= lo {
        return dif[lo.min(dif.len() - 1)];
    }
    let seg = &dif[lo..hi];
    if want_max {
        seg.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        seg.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

// ─── FSM 主函数 ─────────────────────────────────────────────────

/// 给定前一状态 + 截至 idx 的因果笔结构，返回当前走势类型。
pub fn classify_fsm(
    prev: Regime,
    bis: &[BI],
    zs_list: &[ZS],
    ind: &TrendIndicators,
    idx: usize,
) -> Regime {
    let up: Vec<&BI> = bis.iter().filter(|b| is_up(b)).collect();
    let dn: Vec<&BI> = bis.iter().filter(|b| !is_up(b)).collect();
    let last = &bis[bis.len() - 1];
    let zs = zs_list.last();

    let c = ind.close[idx];
    let ma5 = ind.ma5[idx];
    let ma10 = ind.ma10[idx];
    let ma20 = ind.ma20[idx];
    let dif = &ind.dif;
    let dates = &ind.dates;

    // ── 条件 ──────────────────────────────────────────────

    let top_div = if up.len() >= 2 {
        let u_last = up[up.len() - 1];
        let u_prev = up[up.len() - 2];
        if u_last.get_high() > u_prev.get_high() && u_last.get_power() < u_prev.get_power() {
            let dif_now =
                dif_extreme_over(dates, dif, u_last.start_dt(), u_last.end_dt(), true);
            let dif_prev =
                dif_extreme_over(dates, dif, u_prev.start_dt(), u_prev.end_dt(), true);
            dif_now < dif_prev
        } else {
            false
        }
    } else {
        false
    };

    let bot_div = if dn.len() >= 2 {
        let d_last = dn[dn.len() - 1];
        let d_prev = dn[dn.len() - 2];
        if d_last.get_low() < d_prev.get_low() && d_last.get_power() < d_prev.get_power() {
            let dif_now =
                dif_extreme_over(dates, dif, d_last.start_dt(), d_last.end_dt(), false);
            let dif_prev =
                dif_extreme_over(dates, dif, d_prev.start_dt(), d_prev.end_dt(), false);
            dif_now > dif_prev
        } else {
            false
        }
    } else {
        false
    };

    let second_buy =
        dn.len() >= 2 && dn.last().unwrap().get_low() > dn[dn.len() - 2].get_low() && is_up(last);

    let third_buy = zs.is_some()
        && !dn.is_empty()
        && dn.last().unwrap().get_low() > zs.unwrap().zg
        && is_up(last);

    let in_pivot = zs.is_some() && {
        let z = zs.unwrap();
        z.zd <= c && c <= z.zg
    };

    let breakout_up = zs.is_some() && c > zs.unwrap().zg;

    let bull_stack =
        !ma5.is_nan() && !ma10.is_nan() && !ma20.is_nan() && ma5 > ma10 && ma10 > ma20;
    let up_dom = if !up.is_empty() && !dn.is_empty() {
        let dn_tail: Vec<f64> = dn.iter().rev().take(2).map(|b| b.get_power()).collect();
        let dn_mean = dn_tail.iter().sum::<f64>() / dn_tail.len() as f64;
        up.last().unwrap().get_power() >= dn_mean
    } else {
        false
    };
    let main_up = bull_stack && up_dom && (zs.is_none() || c > zs.unwrap().zg);

    let accel = up.len() >= 2 && {
        let u_last = up[up.len() - 1];
        let u_prev = up[up.len() - 2];
        (u_last.get_power() > u_prev.get_power() || u_last.get_angle() > u_prev.get_angle())
            && ma20 > 0.0
            && !ma5.is_nan()
            && !ma20.is_nan()
            && (ma5 - ma20) / ma20 * 100.0 > ACCEL_SPREAD
            && ind.ret20[idx] > ACCEL_RET20
    };

    let trend_rev = !dn.is_empty()
        && !up.is_empty()
        && dn.last().unwrap().get_power() > up.last().unwrap().get_power()
        && !is_up(last);

    let down_resolve = zs.is_some() && c < zs.unwrap().zd;

    let bd_up = (zs.is_some() && c < zs.unwrap().zg) || trend_rev;

    // ── 有限状态转移 ──────────────────────────────────────

    match prev {
        Regime::Downtrend | Regime::FirstBuy | Regime::SecondBuy => {
            if breakout_up && third_buy {
                Regime::ThirdBuy
            } else if breakout_up {
                Regime::UpwardDeparture
            } else if in_pivot {
                Regime::PivotBuilding
            } else if bot_div {
                Regime::FirstBuy
            } else if second_buy {
                Regime::SecondBuy
            } else {
                Regime::Downtrend
            }
        }

        Regime::PivotBuilding => {
            if breakout_up {
                Regime::UpwardDeparture
            } else if down_resolve {
                Regime::Downtrend
            } else {
                Regime::PivotBuilding
            }
        }

        Regime::UpwardDeparture => {
            if bd_up {
                Regime::Breakdown
            } else if third_buy {
                Regime::ThirdBuy
            } else if accel {
                Regime::Acceleration
            } else if main_up {
                Regime::MainUptrend
            } else {
                Regime::UpwardDeparture
            }
        }

        Regime::ThirdBuy | Regime::MainUptrend | Regime::Acceleration => {
            if bd_up {
                Regime::Breakdown
            } else if top_div {
                Regime::Divergence
            } else if accel {
                Regime::Acceleration
            } else if main_up {
                Regime::MainUptrend
            } else if prev != Regime::ThirdBuy {
                prev
            } else {
                Regime::MainUptrend
            }
        }

        Regime::Divergence => {
            if bd_up {
                Regime::Breakdown
            } else if accel && !top_div {
                Regime::Acceleration
            } else if main_up && !top_div {
                Regime::MainUptrend
            } else {
                Regime::Divergence
            }
        }

        Regime::Breakdown => {
            if breakout_up && third_buy {
                Regime::ThirdBuy
            } else if breakout_up {
                Regime::UpwardDeparture
            } else if in_pivot {
                Regime::PivotBuilding
            } else {
                Regime::Downtrend
            }
        }

        _ => Regime::Downtrend,
    }
}

/// 暖机结束时的初始状态：用一次无状态启发式播种。
pub fn seed_regime(
    _bis: &[BI],
    zs_list: &[ZS],
    ind: &TrendIndicators,
    idx: usize,
) -> Regime {
    let zs = zs_list.last();
    let c = ind.close[idx];
    let ma5 = ind.ma5[idx];
    let ma10 = ind.ma10[idx];
    let ma20 = ind.ma20[idx];

    if let Some(z) = zs {
        if z.zd <= c && c <= z.zg {
            return Regime::PivotBuilding;
        }
    }
    if !ma5.is_nan()
        && !ma10.is_nan()
        && !ma20.is_nan()
        && ma5 > ma10
        && ma10 > ma20
        && (zs.is_none() || c > zs.unwrap().zg)
    {
        return Regime::MainUptrend;
    }
    Regime::Downtrend
}
