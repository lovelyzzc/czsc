from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "scripts" / "xs_chan_protocol_v2_1.json"
DOCUMENT_PATH = ROOT / "scripts" / "XS_CHAN_PILOT_PROTOCOL_V2_1.md"

EXPECTED_PROTOCOL_ID = "xs_chan_pilot_v2_1_preregistered_20260720"
EXPECTED_PROTOCOL_STATUS = "LOCKED_PENDING_ENGINEERING_AND_FORWARD_DATA"
EXPECTED_CANONICAL_SHA256 = "8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203"

COMPARISON_IDS = [
    "F_2x_minus_R_match_2x",
    "FC_gross_minus_FGR_gross",
    "FC_2x_minus_FGR_2x",
    "FC_gross_minus_FMA_gross",
    "FC_2x_minus_FMA_2x",
    "FMA_gross_minus_FMGR_gross",
    "FMA_2x_minus_FMGR_2x",
]

COMPARISON_DESIGN = {
    "F_2x_minus_R_match_2x": ("factor_primary", 0.03, True, True),
    "FC_gross_minus_FGR_gross": ("chan_identity_primary", 0.02, True, True),
    "FC_2x_minus_FGR_2x": ("chan_cost_robustness", 0.015, True, True),
    "FC_gross_minus_FMA_gross": ("chan_specificity_primary", 0.01, True, False),
    "FC_2x_minus_FMA_2x": ("chan_specificity_cost_robustness", 0.01, True, False),
    "FMA_gross_minus_FMGR_gross": ("ma_placebo_identity_diagnostic", 0.01, False, True),
    "FMA_2x_minus_FMGR_2x": ("ma_placebo_cost_diagnostic", 0.01, False, True),
}

LIFECYCLE_STATUSES = [
    "LOCKED_PENDING_ENGINEERING_AND_FORWARD_DATA",
    "READY_TO_ANCHOR_PRIMARY_GENESIS",
    "PRIMARY_FORWARD_COLLECTION_ACTIVE",
    "PRIMARY_CHAIN_TERMINATED_INVALID",
    "PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY",
    "EVALUATED",
]

EVALUATION_STATUSES = [
    "INVALID_DATA_OR_ENGINEERING",
    "INVALID_CHAIN",
    "FORWARD_COLLECTION_REQUIRED",
    "INVALID_CONTROL",
    "INSUFFICIENT_INTERVENTION",
    "FALSIFIED_FACTOR",
    "INCONCLUSIVE_FACTOR",
    "FACTOR_PASSED_TIMING_FALSIFIED",
    "FACTOR_PASSED_TIMING_INCONCLUSIVE",
    "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY",
]


def _load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _canonical_bytes(protocol: dict) -> bytes:
    return json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def test_protocol_identity_predecessor_and_canonical_hash_are_frozen():
    protocol = _load_protocol()

    assert protocol["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert protocol["protocol_status"] == EXPECTED_PROTOCOL_STATUS
    assert protocol["frozen_at_utc"] == "2026-07-20T00:00:00Z"
    assert protocol["canonicalization"] == (
        "utf8_json_sort_keys_true_separators_comma_colon_ensure_ascii_false_allow_nan_false"
    )
    assert protocol["predecessor"] == {
        "protocol_id": "xs_chan_pilot_v2_preregistered_20260717",
        "protocol_sha256": "0ec5cb260f2aec78a6dce838880140981fdd1a64b5c9a2dcf64be010ba5c8b62",
        "relationship": "new_protocol_and_new_chain_not_an_amendment_or_continuation",
    }
    assert hashlib.sha256(_canonical_bytes(protocol)).hexdigest() == EXPECTED_CANONICAL_SHA256


def test_portfolio_accounting_has_one_empty_initial_state_and_one_sizing_rule():
    protocol = _load_protocol()
    selection = protocol["selection"]
    accounting = protocol["portfolio_accounting"]
    initial = accounting["initial_state"]
    sizing = accounting["sizing_nav"]

    assert selection["target_size"] == 50
    assert selection["cash_buffer"] == 0.005
    assert selection["weighting"] == "entry_equal_slot_no_periodic_rebalance"
    assert selection["retained_position_rebalance"] == "none"
    assert initial["reference_cash_cny"] == 10_000_000.0
    assert initial["capacity_cash_cny"] == 100_000_000.0
    assert initial["positions"] == []
    assert initial["pending_sells"] == []
    assert initial["cash_receivables"] == []
    assert initial["dividend_tax_liabilities"] == []
    assert initial["borrowed_cash_cny"] == 0.0
    assert initial["initial_nav_equals_cash"] is True
    assert initial["inherited_or_preloaded_book_forbidden"] is True

    assert sizing["endpoint"] == (
        "decision_session_official_close_after_all_eod_corporate_actions_tax_postings_and_valuation"
    )
    assert sizing["target_invested_notional"] == "sizing_nav_times_0_995"
    assert sizing["slot_notional"] == "target_invested_notional_divided_by_50"
    assert sizing["new_entry_requested_shares"] == "floor_slot_notional_divided_by_raw_open_to_board_lot"
    assert accounting["retained_positions"]["weekly_top_up"] is False
    assert accounting["retained_positions"]["weekly_trim"] is False
    assert accounting["order_priority"] == [
        "oldest_pending_full_exits_then_symbol",
        "new_full_exits_then_symbol",
        "new_entries_by_buffered_factor_plan_order_then_symbol",
    ]


def test_partial_fills_cash_dividend_tax_and_share_change_lots_are_unambiguous():
    accounting = _load_protocol()["portfolio_accounting"]
    partial = accounting["partial_fill_policy"]
    cash = accounting["cash_policy"]
    tax = accounting["cash_dividend_and_tax"]
    share_change = accounting["share_change_lot_accounting"]

    assert partial["sell_unfilled_remainder"].startswith("append_pending_sell_and_retry")
    assert partial["buy_unfilled_remainder"] == "cancel_at_end_of_open_execution_no_intraday_or_next_day_retry"
    assert partial["cash_limited_buy"].startswith("reduce_to_largest_board_lot")
    assert partial["fees_use_actual_filled_quantity_and_notional"] is True
    assert partial["zero_fill_is_explicitly_recorded"] is True
    assert cash == {
        "cash_interest_rate": 0.0,
        "cash_interest_compounding": "none",
        "negative_cash_allowed": False,
        "margin_or_shorting_allowed": False,
        "uninvested_cash_carries_at_par": True,
    }

    assert tax["dividend_credit_date"] == "official_cash_payment_date"
    assert tax["taxpayer_profile"] == "mainland_china_individual_securities_account"
    assert tax["policy_authority"] == "cai_shui_2015_101_with_operational_rules_from_cai_shui_2012_85"
    assert tax["lot_matching"] == "fifo_by_original_acquisition_date_then_lot_id"
    assert tax["holding_period"] == (
        "from_official_acquisition_settlement_date_through_day_before_official_disposal_settlement_date"
    )
    assert tax["period_boundary"] == "natural_calendar_month_and_year_not_fixed_30_or_365_day_approximation"
    assert tax["deferred_tax_settlement"] == "on_sale_or_terminal_disposition_of_each_entitled_share_lot"
    assert tax["open_lot_liability_valuation"] == ("remeasure_each_eod_at_tax_rate_if_disposed_on_that_valuation_date")
    assert tax["tax_rate_schedule"] == [
        {
            "condition": "disposal_settlement_date_lte_acquisition_settlement_date_plus_one_calendar_month",
            "rate": 0.2,
        },
        {
            "condition": "disposal_settlement_date_gt_acquisition_settlement_date_plus_one_calendar_month_and_lte_plus_one_calendar_year",
            "rate": 0.1,
        },
        {
            "condition": "disposal_settlement_date_gt_acquisition_settlement_date_plus_one_calendar_year",
            "rate": 0.0,
        },
    ]
    assert share_change["allowed_action_subtypes"] == [
        "stock_dividend",
        "capital_reserve_conversion",
        "split",
        "consolidation",
    ]
    assert share_change["stock_dividend_or_capital_reserve_child_acquisition_date"] == (
        "official_new_share_registration_or_credit_date"
    )
    assert share_change["split_or_consolidation_acquisition_date"] == ("inherit_original_parent_lot_acquisition_date")
    assert share_change["daily_netting_before_fifo"].startswith("use_official_eod_net_increase_or_decrease")
    assert share_change["total_cost_basis"] == "preserved_except_official_cash_in_lieu"
    assert share_change["missing_action_subtype_or_tax_basis"] == "invalidate_cycle_no_executor_inference"


def test_weekly_return_endpoints_and_terminal_treatment_are_unique():
    accounting = _load_protocol()["portfolio_accounting"]
    weekly = accounting["weekly_return_accounting"]
    terminal = accounting["terminal_policy"]
    valuation = accounting["valuation"]

    assert weekly["start_endpoint"] == "prior_cycle_decision_close_nav_after_eod_postings"
    assert weekly["end_endpoint"] == (
        "current_cycle_close_nav_at_last_completed_week_official_close_after_eod_postings"
    )
    assert weekly["cycle_return"] == "end_nav_divided_by_start_nav_minus_one"
    assert weekly["active_return"] == ("arm_arithmetic_cycle_return_minus_comparator_arithmetic_cycle_return")
    assert weekly["external_cash_flows"] == "forbidden_after_genesis"
    assert weekly["confirmation_returns"] == "first_exactly_52_consecutive_semantically_valid_cycle_returns"
    assert terminal["forced_or_synthetic_liquidation_in_confirmatory_return"] is False
    assert terminal["post_window_unwind_role"] == ("mandatory_tradability_diagnostic_never_changes_confirmatory_status")
    assert valuation["suspended_position_price"].startswith("last_official_raw_unadjusted_close")
    assert valuation["missing_price_or_unverified_terminal_value"] == "invalidate_cycle_not_drop_position"


def test_opening_capacity_and_minimum_commission_are_frozen():
    protocol = _load_protocol()
    cost = protocol["cost_model"]
    artifacts = protocol["data_contract"]["required_artifacts"]

    assert cost["max_adv_participation"] == 0.05
    assert cost["official_open_auction_turnover_participation_cap"] == 0.1
    assert cost["opening_fill_notional_cap"] == (
        "minimum_of_0_05_times_decision_known_adv20_cny_and_0_10_times_same_session_official_opening_call_auction_turnover_cny"
    )
    assert cost["missing_or_zero_official_open_auction_turnover"] == "zero_fill_for_symbol_at_that_open"
    assert cost["official_open_auction_turnover_substitute"].startswith("forbidden")
    assert cost["minimum_commission_multiplier"] == 1.0
    assert cost["minimum_commission_cny_under_1x_and_2x"] == 5.0
    assert "open_auction" in artifacts
    assert "open_auction_reconciliation" in artifacts
    assert "open_auction_reconciliation_engineering" in protocol["engineering_readiness"]["required_gates"]


def test_all_seven_comparisons_have_explicit_role_sesoi_and_three_way_rule():
    statistics = _load_protocol()["statistics"]
    registry = statistics["comparison_registry"]

    assert statistics["confirmatory_comparisons"] == COMPARISON_IDS
    assert list(registry) == COMPARISON_IDS
    assert set(registry) == set(COMPARISON_DESIGN)
    assert statistics["comparison_statuses"] == ["PASS", "FALSIFIED", "INCONCLUSIVE"]

    required_fields = {
        "role",
        "claim",
        "arm",
        "comparator",
        "cost_basis",
        "decision_gating",
        "effect_metric",
        "sesoi_annualized",
        "minimum_identity_disagreement_weeks",
        "minimum_differing_symbol_assignments",
        "seed_mcse_required",
        "pass_rule",
        "falsified_rule",
        "otherwise_status",
    }
    for comparison_id, (role, sesoi, gating, mcse_required) in COMPARISON_DESIGN.items():
        comparison = registry[comparison_id]
        assert set(comparison) == required_fields
        assert comparison["role"] == role
        assert comparison["sesoi_annualized"] == sesoi
        assert comparison["decision_gating"] is gating
        assert comparison["seed_mcse_required"] is mcse_required
        assert comparison["effect_metric"] == "annualized_mean_weekly_active_return"
        assert comparison["minimum_identity_disagreement_weeks"] == 26
        assert comparison["minimum_differing_symbol_assignments"] == 260
        assert "hac_lower_95_and_bootstrap_q05_strictly_greater_than_sesoi" in comparison["pass_rule"]
        assert comparison["falsified_rule"] == ("hac_upper_95_and_bootstrap_q95_strictly_less_than_sesoi")
        assert comparison["otherwise_status"] == "INCONCLUSIVE"


def test_sesoi_mde_and_failure_to_pass_semantics_are_numerically_frozen():
    mde = _load_protocol()["statistics"]["sesoi_and_mde"]

    assert mde["alpha_one_sided"] == 0.05
    assert mde["target_power"] == 0.8
    assert mde["weeks"] == 52
    assert mde["failure_to_pass_is_not_falsification"] is True
    assert mde["falsification_boundary"] == ("both_registered_upper_95_bounds_strictly_below_comparison_sesoi")
    expected = [
        (mde["z_one_minus_alpha"] + mde["z_power"]) * weekly_std * math.sqrt(mde["weeks"])
        for weekly_std in mde["weekly_active_std_grid"]
    ]
    assert len(expected) == len(mde["mde_annualized_excess_over_sesoi_grid"])
    for actual, frozen in zip(expected, mde["mde_annualized_excess_over_sesoi_grid"], strict=True):
        assert math.isclose(actual, frozen, rel_tol=0.0, abs_tol=1e-15)


def test_regime_registry_maps_all_states_and_only_5_to_8_allow_entry():
    registry = _load_protocol()["timing_arms"]["regime_registry"]
    names = [
        "NotTradable",
        "Downtrend",
        "FirstBuy",
        "SecondBuy",
        "PivotBuilding",
        "UpwardDeparture",
        "ThirdBuy",
        "MainUptrend",
        "Acceleration",
        "Divergence",
        "Breakdown",
    ]

    assert list(registry) == [str(code) for code in range(11)]
    assert [registry[str(code)]["name"] for code in range(11)] == names
    assert {code for code in range(11) if registry[str(code)]["entry_allowed"]} == {5, 6, 7, 8}
    assert _load_protocol()["timing_arms"]["chan"]["claim_scope"] == "frozen_hybrid_state_classifier_only"


def test_intervention_exposure_turnover_and_mcse_gates_are_operational():
    validity = _load_protocol()["control_validity"]
    intervention = validity["intervention"]
    exposure = validity["exposure"]
    turnover = validity["turnover"]
    mcse = validity["monte_carlo_standard_error"]

    assert intervention["minimum_identity_disagreement_weeks"] == 26
    assert intervention["minimum_differing_symbol_assignments"] == 260
    assert intervention["minimum_gate_eligible_new_entry_opportunities"] == 260
    assert intervention["insufficient_policy"] == "INSUFFICIENT_INTERVENTION_not_falsified"
    assert intervention["realized_return_must_not_define_intervention"] is True

    assert exposure["measure"] == "daily_official_close_gross_long_market_value_divided_by_nav"
    assert exposure["gap_measure"] == "absolute_arm_minus_comparator_exposure"
    assert (exposure["mean_absolute_gap_max"], exposure["p95_absolute_gap_max"]) == (0.01, 0.03)
    assert exposure["maximum_absolute_gap_max"] == 0.05
    assert exposure["signed_mean_gap_is_not_a_valid_substitute"] is True

    assert turnover["mean_absolute_weekly_gap_max"] == 0.025
    assert turnover["p95_absolute_weekly_gap_max"] == 0.1
    assert turnover["fc_vs_fma_policy"].startswith("mandatory_report_not_a_control_validity_gate")
    assert len(turnover["matched_control_pairs"]) == 5

    assert mcse["maximum_annualized_seed_mcse"] == 0.005
    assert mcse["bootstrap_batches"] == 20
    assert mcse["bootstrap_draws_per_batch"] == 1000
    assert mcse["maximum_bootstrap_quantile_mcse_annualized"] == 0.0025
    assert mcse["failed_mcse_policy"] == "INVALID_CONTROL_no_seed_or_draw_extension_after_results"


def test_calendar_is_incremental_and_primary_chain_cannot_be_restarted_or_selected():
    protocol = _load_protocol()
    calendar = protocol["calendar_commitment"]
    chain = protocol["chain_governance"]

    assert calendar["mode"] == "incremental_official_calendar_chunks"
    assert calendar["future_52_week_calendar_at_genesis_forbidden"] is True
    assert calendar["extension_record"] == "calendar_extension"
    assert calendar["extension_must_be_externally_anchored_before"] == "first_newly_covered_session_open"
    assert calendar["contiguous_nonoverlapping_chunks_required"] is True
    assert calendar["covered_session_revision_or_deletion"] == "terminate_primary_chain_as_INVALID_CHAIN"

    assert chain["trial_id"] == "sha256_of_canonical_protocol_bytes"
    assert chain["primary_chain"] == ("first_readiness_valid_genesis_registered_and_externally_anchored_for_trial_id")
    assert chain["second_primary_genesis_for_same_protocol_forbidden"] is True
    assert chain["same_protocol_restart_after_abort"] == "forbidden"
    assert chain["restart_requirement"].startswith("new_protocol_id_new_trial_id_new_primary_chain")
    assert chain["cross_chain_selection"].startswith("forbidden")
    assert chain["invalid_cycle_policy"] == "terminate_primary_chain_no_drop_replacement_or_extension"
    assert "initial_state_sha256_for_every_arm_and_scenario" in chain["genesis_must_bind"]
    assert "each_engineering_gate_evidence_sha256" in chain["genesis_must_bind"]


def test_daily_open_and_eod_records_separate_information_by_availability():
    ledger = _load_protocol()["ledger"]
    daily = ledger["daily_record_requirements"]

    assert ledger["record_types"] == [
        "genesis",
        "calendar_extension",
        "decision",
        "session_open_execution",
        "session_eod_valuation",
        "cycle_close",
        "chain_abort",
        "final_evaluation",
    ]
    assert ledger["record_time_rules"]["decision"].startswith("after_decision_session_15_00")
    assert ledger["record_time_rules"]["session_open_execution"].startswith("at_or_after_session_09_30")
    assert ledger["record_time_rules"]["session_eod_valuation"].startswith("at_or_after_session_15_00")
    assert daily["every_covered_session_requires_open_execution_even_when_no_orders"] is True
    assert daily["every_covered_session_requires_eod_valuation"] is True
    assert "official_opening_call_auction_turnover_snapshot_hash" in daily["open_execution_fields"]
    assert "official_raw_ohlc_volume_amount_and_status_snapshot_hash" in daily["eod_valuation_fields"]
    assert daily["full_raw_ohlc_in_open_execution_forbidden"] is True
    assert daily["pending_sell_retry"].startswith("new_session_open_execution_record_each_official_session")
    assert ledger["cycle_completeness"].startswith("one_valid_decision_all_official_session_open_and_eod_records")
    assert ledger["confirmation_window"] == "first_exactly_52_consecutive_complete_cycle_returns_no_extension"


def test_delist_writeoff_requires_authoritative_zero_recovery_evidence():
    contract = _load_protocol()["data_contract"]
    coverage = contract["coverage_semantics"]
    writeoff = contract["delist_writeoff_evidence"]

    assert coverage["active_security_terminal_event_forbidden"] is True
    assert coverage["terminal_effective_date_must_equal_security_master_delist_date"] is True
    assert coverage["delist_share_target_must_exist_in_security_master"] is True
    assert coverage["current_name_backfill_for_missing_history_forbidden"] is True
    assert writeoff["meaning"] == ("authoritative_source_explicitly_establishes_zero_cash_and_zero_share_recovery")
    assert set(writeoff["required_fields"]) == {
        "ts_code",
        "effective_date",
        "authority",
        "source_document_id",
        "source_document_sha256",
        "source_asof",
        "retrieved_at",
        "zero_recovery_text_sha256",
    }
    assert writeoff["missing_or_ambiguous_terminal_information"] == "invalid_not_writeoff"
    assert writeoff["executor_generated_writeoff_forbidden"] is True
    assert writeoff["mutually_exclusive_with_delist_cash_or_delist_share"] is True


def test_all_formal_states_and_priority_matrix_are_closed():
    protocol = _load_protocol()
    registry = protocol["formal_state_registry"]
    matrix = protocol["decision_matrix"]

    assert registry["lifecycle"] == LIFECYCLE_STATUSES
    assert registry["evaluation"] == EVALUATION_STATUSES
    assert registry["non_authoritative"] == ["NON_AUTHORITATIVE_SIMULATION", "CONTAMINATED_STRESS_ONLY"]
    assert registry["unknown_status_forbidden"] is True
    assert [row["priority"] for row in matrix] == list(range(1, 11))
    assert [row["status"] for row in matrix] == EVALUATION_STATUSES
    assert len({row["condition"] for row in matrix}) == len(matrix)
    assert "REJECT_TIMING" not in json.dumps(protocol, ensure_ascii=False)


def test_explanatory_document_tracks_machine_protocol():
    document = DOCUMENT_PATH.read_text(encoding="utf-8")

    assert EXPECTED_PROTOCOL_ID in document
    assert EXPECTED_PROTOCOL_STATUS in document
    for comparison_id in COMPARISON_IDS:
        assert comparison_id in document
    for status in EVALUATION_STATUSES:
        assert status in document
    assert "同一 protocol 的第二个 primary genesis" in document
    assert "资料没抓到" in document
    assert "不显著" in document
