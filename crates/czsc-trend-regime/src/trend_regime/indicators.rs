//! 预计算因果指标（rolling/ewm 仅用过去值，天然因果）。

use chrono::{DateTime, Utc};
use czsc_core::objects::bar::RawBar;
use czsc_ta::pure::sma;

/// 预计算的指标数组（一次性分配，按下标对齐 bars）。
pub struct TrendIndicators {
    pub close: Vec<f64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub vol: Vec<f64>,
    pub pct_chg: Vec<f64>,
    pub dates: Vec<DateTime<Utc>>,
    pub dif: Vec<f64>,
    pub ma5: Vec<f64>,
    pub ma10: Vec<f64>,
    pub ma20: Vec<f64>,
    pub ret20: Vec<f64>,
    pub n: usize,
}

/// pandas 兼容的 EWM（`adjust=False`，以 `series[0]` 为种子）。
fn ewm_pandas(series: &[f64], span: usize) -> Vec<f64> {
    let n = series.len();
    if n == 0 || span == 0 {
        return vec![];
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0; n];
    out[0] = series[0];
    for i in 1..n {
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1];
    }
    out
}

impl TrendIndicators {
    /// 从 `RawBar` 切片提取 OHLCV 并预计算 dif/ma5/ma10/ma20/ret20。
    pub fn from_bars(bars: &[RawBar]) -> Self {
        let n = bars.len();
        let mut close = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut vol = Vec::with_capacity(n);
        let mut dates = Vec::with_capacity(n);

        for b in bars {
            close.push(b.close);
            open.push(b.open);
            high.push(b.high);
            low.push(b.low);
            vol.push(b.vol);
            dates.push(b.dt);
        }

        let mut pct_chg = vec![0.0; n];
        for i in 1..n {
            if close[i - 1] != 0.0 {
                pct_chg[i] = (close[i] / close[i - 1] - 1.0) * 100.0;
            }
        }

        let ema12 = ewm_pandas(&close, 12);
        let ema26 = ewm_pandas(&close, 26);
        let mut dif = vec![0.0; n];
        for i in 0..n {
            dif[i] = ema12[i] - ema26[i];
        }

        let ma5 = sma(&close, 5);
        let ma10 = sma(&close, 10);
        let ma20 = sma(&close, 20);

        let mut ret20 = vec![0.0; n];
        if n > 20 {
            for i in 20..n {
                if close[i - 20] != 0.0 {
                    ret20[i] = (close[i] / close[i - 20] - 1.0) * 100.0;
                }
            }
        }

        Self {
            close,
            open,
            high,
            low,
            vol,
            pct_chg,
            dates,
            dif,
            ma5,
            ma10,
            ma20,
            ret20,
            n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewm_pandas_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewm_pandas(&data, 3);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);
    }
}
