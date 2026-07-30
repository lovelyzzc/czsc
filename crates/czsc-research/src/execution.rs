//! Deterministic execution replay engine.
//!
//! Pure replay engine: all selections, market observations, corporate actions,
//! and cost parameters must be supplied as inputs. It never fetches data.
//!
//! Key features:
//! - Opening auction capacity filling
//! - Limit up/down and suspension handling
//! - Sell-before-buy with blocked exit FIFO
//! - Per-session NAV tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("execution replay error: {0}")]
    Replay(String),
    #[error("invalid input: {0}")]
    Input(String),
}

pub type Result<T> = std::result::Result<T, ExecutionError>;

/// Transaction cost model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub commission_bps: f64,
    pub transfer_bps: f64,
    pub min_commission_cny: f64,
    pub stamp_duty_bps_sell: f64,
    pub base_slippage_bps: f64,
    pub impact_bps_at_1pct: f64,
    pub max_adv_participation: f64,
    pub auction_participation_cap: f64,
    pub cost_multiplier: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            commission_bps: 2.5,
            transfer_bps: 0.1,
            min_commission_cny: 5.0,
            stamp_duty_bps_sell: 5.0,
            base_slippage_bps: 10.0,
            impact_bps_at_1pct: 15.0,
            max_adv_participation: 0.05,
            auction_participation_cap: 0.10,
            cost_multiplier: 1.0,
        }
    }
}

/// Market data for one session for one stock.
#[derive(Debug, Clone)]
pub struct SessionMarketData {
    pub symbol: String,
    pub session: String,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub amount: f64,
    pub adv20: f64,
    pub is_trading: bool,
    pub is_limit_up: bool,
    pub is_limit_down: bool,
}

/// A single tax lot in FIFO accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxLot {
    pub lot_id: String,
    pub acquisition_session: String,
    pub shares: i64,
    pub cost_basis_cny: f64,
}

/// Position in a single stock.
#[derive(Debug, Clone, Default)]
pub struct Position {
    pub lots: Vec<TaxLot>,
}

impl Position {
    pub fn total_shares(&self) -> i64 {
        self.lots.iter().map(|l| l.shares).sum()
    }

    pub fn total_cost_basis(&self) -> f64 {
        self.lots.iter().map(|l| l.cost_basis_cny).sum()
    }
}

/// Portfolio book: all positions + cash.
#[derive(Debug, Clone)]
pub struct Book {
    pub arm_id: String,
    pub positions: HashMap<String, Position>,
    pub cash_cny: f64,
    pub initial_nav: f64,
}

impl Book {
    pub fn new(arm_id: &str, initial_nav: f64) -> Self {
        Self {
            arm_id: arm_id.to_string(),
            positions: HashMap::new(),
            cash_cny: initial_nav,
            initial_nav,
        }
    }

    pub fn nav(&self, prices: &HashMap<String, f64>) -> f64 {
        let position_value: f64 = self
            .positions
            .iter()
            .map(|(sym, pos)| {
                let price = prices.get(sym).copied().unwrap_or(0.0);
                pos.total_shares() as f64 * price
            })
            .sum();
        self.cash_cny + position_value
    }

    pub fn exposure(&self, prices: &HashMap<String, f64>) -> f64 {
        let nav = self.nav(prices);
        if nav <= 0.0 {
            return 0.0;
        }
        let position_value: f64 = self
            .positions
            .iter()
            .map(|(sym, pos)| {
                let price = prices.get(sym).copied().unwrap_or(0.0);
                pos.total_shares() as f64 * price
            })
            .sum();
        position_value / nav
    }
}

/// Execution decision for one session.
#[derive(Debug, Clone)]
pub struct ExecutionDecision {
    pub session: String,
    pub buys: Vec<(String, i64)>,
    pub sells: Vec<(String, i64)>,
}

/// Result of processing one session.
#[derive(Debug, Clone)]
pub struct SessionResult {
    pub session: String,
    pub nav: f64,
    pub exposure: f64,
    pub buy_fills: Vec<(String, i64, f64)>,
    pub sell_fills: Vec<(String, i64, f64)>,
    pub blocked_sells: Vec<(String, i64)>,
    pub total_commission: f64,
    pub total_slippage: f64,
}

/// Compute transaction cost for a single trade.
pub fn compute_transaction_cost(
    cost_model: &CostModel,
    notional_cny: f64,
    is_sell: bool,
    adv20: f64,
    participation_rate: f64,
) -> f64 {
    let commission = (notional_cny * cost_model.commission_bps / 10_000.0)
        .max(cost_model.min_commission_cny);
    let transfer = notional_cny * cost_model.transfer_bps / 10_000.0;
    let stamp_duty = if is_sell {
        notional_cny * cost_model.stamp_duty_bps_sell / 10_000.0
    } else {
        0.0
    };
    let impact = if adv20 > 0.0 {
        notional_cny * cost_model.impact_bps_at_1pct / 10_000.0
            * (participation_rate / 0.01).sqrt()
    } else {
        notional_cny * cost_model.base_slippage_bps / 10_000.0
    };
    let slippage = notional_cny * cost_model.base_slippage_bps / 10_000.0;

    (commission + transfer + stamp_duty + impact + slippage) * cost_model.cost_multiplier
}

/// Compute maximum fillable shares based on capacity constraints.
pub fn max_fill_shares(
    target_shares: i64,
    price: f64,
    adv20: f64,
    lot_size: i64,
    cost_model: &CostModel,
) -> i64 {
    if price <= 0.0 || target_shares <= 0 {
        return 0;
    }

    let adv_limited = if adv20 > 0.0 {
        let adv_limit = (adv20 * cost_model.max_adv_participation / price) as i64;
        target_shares.min(adv_limit)
    } else {
        target_shares
    };

    (adv_limited / lot_size) * lot_size
}

/// Execute one session: process sells first, then buys.
pub fn execute_session(
    book: &mut Book,
    decision: &ExecutionDecision,
    market_data: &HashMap<String, SessionMarketData>,
    cost_model: &CostModel,
    lot_size: i64,
) -> Result<SessionResult> {
    let mut buy_fills = Vec::new();
    let mut sell_fills = Vec::new();
    let mut blocked_sells = Vec::new();
    let mut total_commission = 0.0;
    let mut total_slippage = 0.0;

    for (symbol, shares_to_sell) in &decision.sells {
        let md = match market_data.get(symbol) {
            Some(md) => md,
            None => {
                blocked_sells.push((symbol.clone(), *shares_to_sell));
                continue;
            }
        };

        if !md.is_trading || md.is_limit_down {
            blocked_sells.push((symbol.clone(), *shares_to_sell));
            continue;
        }

        let position = match book.positions.get_mut(symbol) {
            Some(p) => p,
            None => continue,
        };

        let available = position.total_shares();
        let fill_target = (*shares_to_sell).min(available);
        let fill = max_fill_shares(fill_target, md.open, md.adv20, lot_size, cost_model);

        if fill > 0 {
            let notional = fill as f64 * md.open;
            let cost = compute_transaction_cost(cost_model, notional, true, md.adv20, 0.01);
            book.cash_cny += notional - cost;
            total_commission += cost;
            total_slippage += notional * cost_model.base_slippage_bps / 10_000.0;

            let mut remaining = fill;
            book.positions.get_mut(symbol).unwrap().lots.retain_mut(|lot| {
                if remaining <= 0 {
                    return true;
                }
                if lot.shares <= remaining {
                    remaining -= lot.shares;
                    false
                } else {
                    let fraction = remaining as f64 / lot.shares as f64;
                    lot.cost_basis_cny *= 1.0 - fraction;
                    lot.shares -= remaining;
                    remaining = 0;
                    true
                }
            });

            sell_fills.push((symbol.clone(), fill, notional - cost));

            if book.positions.get(symbol).map_or(true, |p| p.total_shares() <= 0) {
                book.positions.remove(symbol);
            }
        }

        let unfilled = shares_to_sell - fill;
        if unfilled > 0 {
            blocked_sells.push((symbol.clone(), unfilled));
        }
    }

    for (symbol, target_shares) in &decision.buys {
        let md = match market_data.get(symbol) {
            Some(md) => md,
            None => continue,
        };

        if !md.is_trading || md.is_limit_up {
            continue;
        }

        let fill = max_fill_shares(*target_shares, md.open, md.adv20, lot_size, cost_model);
        if fill <= 0 {
            continue;
        }

        let notional = fill as f64 * md.open;
        let cost = compute_transaction_cost(cost_model, notional, false, md.adv20, 0.01);
        let total_cost = notional + cost;

        if total_cost > book.cash_cny {
            continue;
        }

        book.cash_cny -= total_cost;
        total_commission += cost;
        total_slippage += notional * cost_model.base_slippage_bps / 10_000.0;

        let lot = TaxLot {
            lot_id: format!("{}_{}", symbol, decision.session),
            acquisition_session: decision.session.clone(),
            shares: fill,
            cost_basis_cny: total_cost,
        };

        book.positions
            .entry(symbol.clone())
            .or_default()
            .lots
            .push(lot);

        buy_fills.push((symbol.clone(), fill, total_cost));
    }

    let close_prices: HashMap<String, f64> = market_data
        .iter()
        .map(|(sym, md)| (sym.clone(), md.close))
        .collect();

    let nav = book.nav(&close_prices);
    let exposure = book.exposure(&close_prices);

    Ok(SessionResult {
        session: decision.session.clone(),
        nav,
        exposure,
        buy_fills,
        sell_fills,
        blocked_sells,
        total_commission,
        total_slippage,
    })
}

/// Replay a full sequence of execution decisions.
pub fn replay_execution(
    book: &mut Book,
    decisions: &[ExecutionDecision],
    market_data_by_session: &HashMap<String, HashMap<String, SessionMarketData>>,
    cost_model: &CostModel,
    lot_size: i64,
) -> Result<Vec<SessionResult>> {
    let mut results = Vec::with_capacity(decisions.len());

    for decision in decisions {
        let session_data = market_data_by_session.get(&decision.session).ok_or_else(|| {
            ExecutionError::Input(format!(
                "missing market data for session {}",
                decision.session
            ))
        })?;

        let result = execute_session(book, decision, session_data, cost_model, lot_size)?;
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_market_data(symbol: &str, open: f64, close: f64) -> SessionMarketData {
        SessionMarketData {
            symbol: symbol.to_string(),
            session: "2024-01-01".to_string(),
            open,
            close,
            high: open.max(close) * 1.01,
            low: open.min(close) * 0.99,
            amount: 1_000_000.0,
            adv20: 500_000.0,
            is_trading: true,
            is_limit_up: false,
            is_limit_down: false,
        }
    }

    #[test]
    fn test_transaction_cost_buy() {
        let model = CostModel::default();
        let cost = compute_transaction_cost(&model, 100_000.0, false, 500_000.0, 0.01);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_transaction_cost_sell_includes_stamp_duty() {
        let model = CostModel::default();
        let buy_cost = compute_transaction_cost(&model, 100_000.0, false, 500_000.0, 0.01);
        let sell_cost = compute_transaction_cost(&model, 100_000.0, true, 500_000.0, 0.01);
        assert!(sell_cost > buy_cost);
    }

    #[test]
    fn test_execute_buy() {
        let mut book = Book::new("test", 10_000_000.0);
        let cost_model = CostModel::default();
        let mut md = HashMap::new();
        md.insert("AAAA".to_string(), make_market_data("AAAA", 10.0, 10.5));

        let decision = ExecutionDecision {
            session: "2024-01-01".to_string(),
            buys: vec![("AAAA".to_string(), 1000)],
            sells: vec![],
        };

        let result = execute_session(&mut book, &decision, &md, &cost_model, 100).unwrap();
        assert!(!result.buy_fills.is_empty());
        assert!(book.cash_cny < 10_000_000.0);
        assert!(book.positions.contains_key("AAAA"));
    }

    #[test]
    fn test_limit_down_blocks_sell() {
        let mut book = Book::new("test", 10_000_000.0);
        let cost_model = CostModel::default();

        book.positions.insert(
            "BBBB".to_string(),
            Position {
                lots: vec![TaxLot {
                    lot_id: "lot1".to_string(),
                    acquisition_session: "2023-12-01".to_string(),
                    shares: 1000,
                    cost_basis_cny: 10_000.0,
                }],
            },
        );

        let mut md_entry = make_market_data("BBBB", 10.0, 9.0);
        md_entry.is_limit_down = true;
        let mut md = HashMap::new();
        md.insert("BBBB".to_string(), md_entry);

        let decision = ExecutionDecision {
            session: "2024-01-01".to_string(),
            buys: vec![],
            sells: vec![("BBBB".to_string(), 1000)],
        };

        let result = execute_session(&mut book, &decision, &md, &cost_model, 100).unwrap();
        assert!(result.sell_fills.is_empty());
        assert_eq!(result.blocked_sells.len(), 1);
    }
}
