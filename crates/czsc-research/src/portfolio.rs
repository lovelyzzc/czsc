//! FC/F/FMA portfolio simulation with 50/75 buffer.
//!
//! Simulates weekly rebalanced equal-weight portfolios with different
//! feature and gate combinations:
//! - F: factor-only selection (top N by composite score)
//! - FC: factor + Chan regime gate (new entries require regime in {5,6,7,8})
//! - FMA: factor + moving average gate (SMA20 filter)

use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PortfolioError {
    #[error("portfolio error: {0}")]
    Computation(String),
}

pub type Result<T> = std::result::Result<T, PortfolioError>;

/// Weekly membership for one portfolio arm.
#[derive(Debug, Clone)]
pub struct WeeklyMembership {
    pub week_label: String,
    pub holdings: Vec<String>,
    pub new_entries: Vec<String>,
    pub exits: Vec<String>,
    pub retained: Vec<String>,
    pub blocked_exits: Vec<String>,
}

/// Configuration for portfolio simulation.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    pub target_slots: usize,
    pub retention_max_rank: usize,
    pub gate_regimes: HashSet<u8>,
    pub use_chan_gate: bool,
    pub use_sma_gate: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            target_slots: 50,
            retention_max_rank: 75,
            gate_regimes: [5, 6, 7, 8].iter().copied().collect(),
            use_chan_gate: false,
            use_sma_gate: false,
        }
    }
}

/// Input for one week's decision.
#[derive(Debug, Clone)]
pub struct WeeklyDecision {
    pub week_label: String,
    pub ranked_symbols: Vec<String>,
    pub factor_ranks: HashMap<String, f64>,
    pub regimes: HashMap<String, u8>,
    pub above_sma20: HashMap<String, bool>,
    pub blocked_exits: HashSet<String>,
}

/// Simulate a single portfolio arm path through weekly decisions.
///
/// Implements the 50/75 buffer retention logic:
/// 1. Existing holdings ranked within top retention_max_rank are retained
/// 2. Blocked exits (suspended/limit-down) are force-retained
/// 3. New entries fill remaining slots from top-ranked candidates
/// 4. New entries must pass the configured gate (Chan regime / SMA20)
pub fn simulate_portfolio_path(
    weekly_decisions: &[WeeklyDecision],
    config: &PortfolioConfig,
) -> Result<Vec<WeeklyMembership>> {
    let mut results = Vec::with_capacity(weekly_decisions.len());
    let mut current_holdings: BTreeSet<String> = BTreeSet::new();

    for decision in weekly_decisions {
        let mut retained = Vec::new();
        let mut blocked = Vec::new();
        let mut exits = Vec::new();

        for symbol in &current_holdings {
            if decision.blocked_exits.contains(symbol) {
                blocked.push(symbol.clone());
                retained.push(symbol.clone());
                continue;
            }

            let rank_position = decision
                .ranked_symbols
                .iter()
                .position(|s| s == symbol);

            match rank_position {
                Some(pos) if pos < config.retention_max_rank => {
                    retained.push(symbol.clone());
                }
                _ => {
                    exits.push(symbol.clone());
                }
            }
        }

        let available_slots = config.target_slots.saturating_sub(retained.len());
        let mut new_entries = Vec::new();

        for symbol in &decision.ranked_symbols {
            if new_entries.len() >= available_slots {
                break;
            }
            if retained.contains(symbol) {
                continue;
            }

            let gate_pass = if config.use_chan_gate {
                decision
                    .regimes
                    .get(symbol)
                    .map_or(false, |r| config.gate_regimes.contains(r))
            } else if config.use_sma_gate {
                decision.above_sma20.get(symbol).copied().unwrap_or(false)
            } else {
                true
            };

            if gate_pass {
                new_entries.push(symbol.clone());
            }
        }

        current_holdings.clear();
        for s in &retained {
            current_holdings.insert(s.clone());
        }
        for s in &new_entries {
            current_holdings.insert(s.clone());
        }

        let holdings: Vec<String> = current_holdings.iter().cloned().collect();

        results.push(WeeklyMembership {
            week_label: decision.week_label.clone(),
            holdings,
            new_entries,
            exits,
            retained,
            blocked_exits: blocked,
        });
    }

    Ok(results)
}

/// Simulate multiple portfolio arms (F, FC, FMA) and random controls.
pub fn simulate_all_arms(
    weekly_decisions: &[WeeklyDecision],
    target_slots: usize,
    retention_max_rank: usize,
    gate_regimes: &HashSet<u8>,
    random_seeds: &[u64],
) -> Result<HashMap<String, Vec<WeeklyMembership>>> {
    let mut all_results = HashMap::new();

    let f_config = PortfolioConfig {
        target_slots,
        retention_max_rank,
        use_chan_gate: false,
        use_sma_gate: false,
        ..Default::default()
    };
    all_results.insert(
        "F".to_string(),
        simulate_portfolio_path(weekly_decisions, &f_config)?,
    );

    let fc_config = PortfolioConfig {
        target_slots,
        retention_max_rank,
        gate_regimes: gate_regimes.clone(),
        use_chan_gate: true,
        use_sma_gate: false,
    };
    all_results.insert(
        "FC".to_string(),
        simulate_portfolio_path(weekly_decisions, &fc_config)?,
    );

    let fma_config = PortfolioConfig {
        target_slots,
        retention_max_rank,
        use_chan_gate: false,
        use_sma_gate: true,
        ..Default::default()
    };
    all_results.insert(
        "FMA".to_string(),
        simulate_portfolio_path(weekly_decisions, &fma_config)?,
    );

    for &seed in random_seeds {
        use rand::seq::SliceRandom;
        use rand::rngs::SmallRng;
        use rand::SeedableRng;

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut shuffled_decisions: Vec<WeeklyDecision> = weekly_decisions.to_vec();

        for decision in &mut shuffled_decisions {
            decision.ranked_symbols.shuffle(&mut rng);
        }

        let r_config = PortfolioConfig {
            target_slots,
            retention_max_rank,
            use_chan_gate: false,
            use_sma_gate: false,
            ..Default::default()
        };

        all_results.insert(
            format!("R_match_{seed}"),
            simulate_portfolio_path(&shuffled_decisions, &r_config)?,
        );
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decision(week: usize, symbols: Vec<&str>) -> WeeklyDecision {
        WeeklyDecision {
            week_label: format!("week_{week}"),
            ranked_symbols: symbols.iter().map(|s| s.to_string()).collect(),
            factor_ranks: HashMap::new(),
            regimes: HashMap::new(),
            above_sma20: HashMap::new(),
            blocked_exits: HashSet::new(),
        }
    }

    #[test]
    fn test_empty_decisions() {
        let config = PortfolioConfig::default();
        let result = simulate_portfolio_path(&[], &config).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fills_to_target() {
        let config = PortfolioConfig {
            target_slots: 3,
            retention_max_rank: 5,
            ..Default::default()
        };
        let decisions = vec![
            make_decision(0, vec!["A", "B", "C", "D", "E"]),
        ];
        let result = simulate_portfolio_path(&decisions, &config).unwrap();
        assert_eq!(result[0].holdings.len(), 3);
        assert_eq!(result[0].new_entries.len(), 3);
    }

    #[test]
    fn test_retention_buffer() {
        let config = PortfolioConfig {
            target_slots: 2,
            retention_max_rank: 4,
            ..Default::default()
        };
        let decisions = vec![
            make_decision(0, vec!["A", "B", "C"]),
            make_decision(1, vec!["C", "B", "A", "D"]),
        ];
        let result = simulate_portfolio_path(&decisions, &config).unwrap();
        assert_eq!(result[0].holdings.len(), 2);
        // A and B were in top-4 ranks, so retained; no room for new entries
        assert!(result[1].retained.contains(&"A".to_string()));
        assert!(result[1].retained.contains(&"B".to_string()));
    }

    #[test]
    fn test_chan_gate_blocks_entry() {
        let mut decision = make_decision(0, vec!["A", "B", "C"]);
        decision.regimes.insert("A".to_string(), 1); // not in gate
        decision.regimes.insert("B".to_string(), 5); // in gate
        decision.regimes.insert("C".to_string(), 7); // in gate

        let config = PortfolioConfig {
            target_slots: 3,
            retention_max_rank: 5,
            gate_regimes: [5, 6, 7, 8].iter().copied().collect(),
            use_chan_gate: true,
            use_sma_gate: false,
        };
        let result = simulate_portfolio_path(&[decision], &config).unwrap();
        assert!(!result[0].new_entries.contains(&"A".to_string()));
        assert!(result[0].new_entries.contains(&"B".to_string()));
        assert!(result[0].new_entries.contains(&"C".to_string()));
    }
}
