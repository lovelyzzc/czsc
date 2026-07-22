from __future__ import annotations

# ruff: noqa: E402, I001

from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_statistics_v2_1 as statistics


PROTOCOL = json.loads((SCRIPTS_DIR / "xs_chan_protocol_v2_1.json").read_text(encoding="utf-8"))
PROTOCOL_SHA256 = hashlib.sha256(statistics.canonical_json_bytes(PROTOCOL)).hexdigest()
WEEK_LABELS = [f"2026-W{index:02d}" for index in range(1, 53)]
DAILY_LABELS = [f"2026-D{index:03d}" for index in range(260)]
SEEDED_ARMS = {
    arm_id
    for arm_id in statistics.EXPECTED_RISK_ARM_IDS
    if any(arm_id.startswith(f"{family}_") for family in statistics.EXPECTED_SEEDED_FAMILIES)
}


def _pending_settlements_root(count: int, tag: str) -> str:
    if count == 0:
        return statistics.EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256
    return statistics.object_sha256({"pending_settlement_count": count, "tag": tag})


def _bind_unwind_metadata(
    unwind: dict,
    *,
    starting_pending_count: int = 0,
    pending_counts: list[int] | None = None,
) -> dict:
    sessions = unwind["sessions"]
    if pending_counts is None:
        pending_counts = [0] * len(sessions)
    assert len(pending_counts) == len(sessions)
    unwind["starting_pending_settlement_count"] = starting_pending_count
    unwind["starting_pending_settlements_root_sha256"] = _pending_settlements_root(starting_pending_count, "start")
    for index, (session, pending_count) in enumerate(zip(sessions, pending_counts, strict=True)):
        session["pending_settlement_count"] = pending_count
        session["pending_settlements_root_sha256"] = _pending_settlements_root(pending_count, f"session:{index}")
        session["session_event_count"] = (
            len(session["sell_orders"])
            + len(session["terminal_dispositions"])
            + len(session["terminal_share_receipts"])
        )
        session["session_events_root_sha256"] = statistics.object_sha256(
            {"session": session["session"], "event_count": session["session_event_count"]}
        )
    starts_complete = (
        not unwind["starting_positions"] and not unwind["pending_share_conversions"] and starting_pending_count == 0
    )
    completion_session = unwind["confirmation_end_session"] if starts_complete else sessions[-1]["session"]
    completion_state = {
        "nav_cny": 10_000_000.0,
        "cash_cny": 10_000_000.0,
        "cash_yield_cny": 0.0,
        "position_value_cny": 0.0,
        "cash_receivable_value_cny": 0.0,
        "other_asset_value_cny": 0.0,
        "other_liability_value_cny": 0.0,
        "dividend_tax_liability_cny": 0.0,
        "gross_exposure": 0.0,
        "holding_count": 0,
        "pending_sell_count": 0,
        "pending_sell_symbols": [],
        "pending_settlements_root_sha256": statistics.EMPTY_PENDING_SETTLEMENTS_ROOT_SHA256,
        "pending_settlement_count": 0,
        "contingent_slot_symbols": [],
        "holdings": [],
        "pending_sells_sha256": statistics.object_sha256({}),
    }
    unwind.update(
        {
            "completion_session": completion_session,
            "completion_economic_state": completion_state,
            "completion_economic_state_sha256": statistics.object_sha256(completion_state),
            "completion_state_sha256": statistics.object_sha256({"completion_session": completion_session}),
            "post_completion_heartbeats": [],
        }
    )
    return unwind


def _path_payload(
    returns: np.ndarray,
    *,
    arm_id: str,
    exposure: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    selection_prefix: str,
    opportunities: int = 10,
) -> dict:
    exposure = np.full(len(DAILY_LABELS), 0.50) if exposure is None else np.asarray(exposure, dtype=float)
    turnover = np.full(52, 0.10) if turnover is None else np.asarray(turnover, dtype=float)
    selections = [[f"{selection_prefix}-{slot:02d}"] for slot in range(10)]
    # Ten different identities per week make the symmetric-difference count
    # exactly ten assignments when two prefixes are compared.
    weekly_selections = [[item[0] for item in selections] for _ in range(52)]
    scenario = next(
        item
        for item in sorted(statistics.EXPECTED_RISK_SCENARIOS, key=len, reverse=True)
        if arm_id.endswith(f"_{item}")
    )
    cost_scale = {"gross": 0.0, "1x": 1.0, "2x": 2.0, "capacity_1x": 1.0}[scenario]
    pretrade_nav = 100_000_000.0 if scenario == "capacity_1x" else 10_000_000.0
    costs = [
        {
            "commission_cny": 5.0 * cost_scale,
            "transfer_fee_cny": 1.0 * cost_scale,
            "stamp_duty_cny": 2.0 * cost_scale,
            "slippage_cost_cny": 10.0 * cost_scale,
        }
        for _ in range(52)
    ]
    order_attempts = []
    state_observations = []
    if arm_id.startswith("FC_"):
        for week_index, week_label in enumerate(WEEK_LABELS):
            entry_index = week_index * 5
            entry_session = DAILY_LABELS[entry_index]
            exit_session = DAILY_LABELS[entry_index + 1]
            for slot in range(opportunities):
                symbol = f"FC-{week_index:02d}-{slot:02d}"
                order_attempts.append(
                    {
                        "session": entry_session,
                        "symbol": symbol,
                        "side": "buy",
                        "requested_shares": 200,
                        "filled_shares": 100,
                        "fill_notional_cny": 1_000.0,
                        "adv20_cny_asof_decision": 100_000.0,
                        "open_auction_turnover_cny": 20_000.0,
                    }
                )
                state_observations.append(
                    {
                        "week_label": week_label,
                        "execution_session": entry_session,
                        "symbol": symbol,
                        "regime": 5,
                        "eligible_new_entry": True,
                        "allowed_new_entry": True,
                        "requested_shares": 200,
                        "filled_shares": 100,
                        "entry_fill_notional_cny": 1_000.0,
                        "pretrade_nav_cny": pretrade_nav,
                        "entry_price": 10.0,
                        "exit_or_window_session": exit_session,
                        "exit_or_window_price": 11.0,
                    }
                )
    else:
        order_attempts.append(
            {
                "session": DAILY_LABELS[0],
                "symbol": "GENERIC",
                "side": "buy",
                "requested_shares": 200,
                "filled_shares": 100,
                "fill_notional_cny": 1_000.0,
                "adv20_cny_asof_decision": 100_000.0,
                "open_auction_turnover_cny": 20_000.0,
            }
        )
    order_attempts.sort(key=lambda item: (item["session"], item["symbol"], item["side"]))
    return {
        "weekly_returns": np.asarray(returns, dtype=float).tolist(),
        "daily_post_close_gross_exposure": exposure.tolist(),
        "weekly_one_way_turnover": turnover.tolist(),
        "weekly_new_entry_identity_sets": weekly_selections,
        "weekly_gate_eligible_new_entry_opportunities": [opportunities] * 52,
        "weekly_pretrade_nav_cny": [pretrade_nav] * 52,
        "weekly_explicit_costs_cny": costs,
        "order_attempts": order_attempts,
        "state_gate_observations": state_observations,
        "post_window_unwind": _bind_unwind_metadata(
            {
                "confirmation_end_session": DAILY_LABELS[-1],
                "starting_positions": [{"symbol": "UNWIND", "shares": 100}],
                "pending_share_conversions": [],
                "sessions": [
                    {
                        "session": "2027-D001",
                        "sell_orders": [{"symbol": "UNWIND", "requested_shares": 100, "filled_shares": 100}],
                        "terminal_dispositions": [],
                        "terminal_share_receipts": [],
                        "remaining_positions": [],
                        "fill_notional_cny": 1_100.0,
                        "commission_cny": 5.0 * cost_scale,
                        "transfer_fee_cny": 1.0 * cost_scale,
                        "stamp_duty_cny": 2.0 * cost_scale,
                        "slippage_cost_cny": 10.0 * cost_scale,
                    }
                ],
            }
        ),
    }


def _arm_path(
    arm_id: str,
    returns: np.ndarray,
    *,
    seeded: bool | None = None,
    exposure: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    seed_offsets: np.ndarray | None = None,
    selection_prefix: str | None = None,
) -> dict:
    seeded = arm_id in SEEDED_ARMS if seeded is None else seeded
    prefix = selection_prefix or arm_id
    if not seeded:
        paths = {
            "primary": _path_payload(
                returns,
                arm_id=arm_id,
                exposure=exposure,
                turnover=turnover,
                selection_prefix=prefix,
            )
        }
        aggregation = "single_path"
    else:
        offsets = np.zeros(20) if seed_offsets is None else np.asarray(seed_offsets, dtype=float)
        assert len(offsets) == 20
        paths = {
            str(seed): _path_payload(
                returns + offsets[index],
                arm_id=arm_id,
                exposure=exposure,
                turnover=turnover,
                selection_prefix=f"{prefix}-seed{index:02d}",
            )
            for index, seed in enumerate(statistics.EXPECTED_RANDOM_SEEDS)
        }
        aggregation = "equal_mean_all_20_frozen_seeds_before_statistics"
    return {
        "schema": statistics.WEEKLY_PATH_SCHEMA,
        "arm_id": arm_id,
        "week_labels": list(WEEK_LABELS),
        "daily_labels": list(DAILY_LABELS),
        "aggregation": aggregation,
        "paths": paths,
    }


def _return_vectors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260720)
    market = 0.001 + rng.normal(0.0, 0.002, 52)
    factor_noise = rng.normal(0.0, 0.00010, 52)
    chan_gross_noise = rng.normal(0.0, 0.00011, 52)
    chan_2x_noise = rng.normal(0.0, 0.00012, 52)
    ma_diag_noise = rng.normal(0.0, 0.00008, 52)
    confirmatory = {
        "F_2x": market + 0.0050 + factor_noise,
        "R_match_2x": market,
        "FC_gross": market + 0.0050 + chan_gross_noise,
        "FGR_gross": market,
        "FC_2x": market + 0.0050 + chan_2x_noise,
        "FGR_2x": market,
        "FMA_gross": market,
        "FMA_2x": market,
        # The two SMA placebo diagnostics are deliberately FALSIFIED.  Their
        # non-gating role must not veto five passing confirmatory gates.
        "FMGR_gross": market + 0.0010 + ma_diag_noise,
        "FMGR_2x": market + 0.0010 - ma_diag_noise,
    }
    result: dict[str, np.ndarray] = {}
    for family in statistics.EXPECTED_RISK_FAMILIES:
        gross_key = f"{family}_gross"
        two_x_key = f"{family}_2x"
        gross = confirmatory[gross_key] if gross_key in confirmatory else confirmatory[two_x_key] + 0.00030
        two_x = confirmatory[two_x_key] if two_x_key in confirmatory else gross - 0.00030
        one_x = (gross + two_x) / 2.0
        result[gross_key] = gross
        result[f"{family}_1x"] = one_x
        result[two_x_key] = two_x
        result[f"{family}_capacity_1x"] = one_x - 0.00002
    return result


def _statistics_input() -> dict:
    vectors = _return_vectors()
    return {
        "schema": statistics.STATISTICS_INPUT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_id": PROTOCOL_SHA256,
        "window_sha256": hashlib.sha256(b"frozen-first-52-weeks").hexdigest(),
        "comparison_registry": deepcopy(PROTOCOL["statistics"]["comparison_registry"]),
        "arm_paths": {arm: _arm_path(arm, vectors[arm]) for arm in statistics.EXPECTED_RISK_ARM_IDS},
        "controls": deepcopy(statistics.EXPECTED_CONTROL_THRESHOLDS),
        "bootstrap": {
            "method_seed": 20260720,
            "block_length_weeks": 4,
            "draws": 20_000,
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
        },
    }


def test_exact_seven_comparison_registry_order_roles_and_frozen_window_are_enforced():
    value = _statistics_input()
    result = statistics.evaluate_statistics(value)

    assert result["comparison_order"] == list(statistics.EXPECTED_COMPARISONS)
    assert list(result["comparisons"]) == list(statistics.EXPECTED_COMPARISONS)
    assert result["complete_weeks"] == 52
    assert [result["comparisons"][identity]["role"] for identity in statistics.EXPECTED_COMPARISONS] == [
        PROTOCOL["statistics"]["comparison_registry"][identity]["role"] for identity in statistics.EXPECTED_COMPARISONS
    ]

    missing = _statistics_input()
    missing["comparison_registry"].pop(statistics.EXPECTED_COMPARISONS[-1])
    with pytest.raises(statistics.StatisticsError, match="exact seven|order"):
        statistics.evaluate_statistics(missing)

    reordered = _statistics_input()
    first = reordered["comparison_registry"].pop(statistics.EXPECTED_COMPARISONS[0])
    reordered["comparison_registry"][statistics.EXPECTED_COMPARISONS[0]] = first
    with pytest.raises(statistics.StatisticsError, match="order"):
        statistics.evaluate_statistics(reordered)

    short = _statistics_input()
    short["arm_paths"]["F_2x"]["week_labels"].pop()
    short["arm_paths"]["F_2x"]["paths"]["primary"]["weekly_returns"].pop()
    with pytest.raises(statistics.StatisticsError, match="exactly 52"):
        statistics.evaluate_statistics(short)


def test_seeded_control_is_equal_mean_of_all_and_only_the_twenty_frozen_seeds():
    base = np.linspace(-0.01, 0.01, 52)
    offsets = np.linspace(-0.0095, 0.0095, 20)
    path = _arm_path("R_match_2x", base, seed_offsets=offsets)

    parsed = statistics._parse_arm_path("R_match_2x", path)

    np.testing.assert_allclose(parsed.returns, base + offsets.mean(), atol=1e-15)
    assert parsed.seed_ids == statistics.EXPECTED_RANDOM_SEEDS
    assert parsed.seed_returns is not None and parsed.seed_returns.shape == (20, 52)

    missing = deepcopy(path)
    missing["paths"].pop(str(statistics.EXPECTED_RANDOM_SEEDS[-1]))
    with pytest.raises(statistics.StatisticsError, match="exactly seeds"):
        statistics._parse_arm_path("R_match_2x", missing)

    extra = deepcopy(path)
    extra["paths"]["999"] = deepcopy(next(iter(extra["paths"].values())))
    with pytest.raises(statistics.StatisticsError, match="exactly seeds"):
        statistics._parse_arm_path("R_match_2x", extra)


def test_degenerate_hac_fails_closed_both_directly_and_in_full_evaluation():
    with pytest.raises(statistics.DegenerateInferenceError, match="non-positive"):
        statistics.newey_west_mean_interval([0.0] * 52)

    value = _statistics_input()
    market = _return_vectors()["R_match_2x"]
    # Constant non-zero active return has zero long-run variance.  It must be
    # INCONCLUSIVE, never a fabricated infinite t statistic or a serializer crash.
    value["arm_paths"]["F_2x"] = _arm_path("F_2x", market + 0.005)
    result = statistics.evaluate_statistics(value)
    comparison = result["comparisons"]["F_2x_minus_R_match_2x"]

    assert comparison["status"] == "INCONCLUSIVE"
    assert comparison["status_reasons"] == ["degenerate_inference"]
    assert comparison["hac"] is None
    assert comparison["inference_error"]


def test_sesoi_decision_is_strictly_three_way_and_uses_median_and_both_intervals():
    spec = PROTOCOL["statistics"]["comparison_registry"]["F_2x_minus_R_match_2x"]
    interventions = {
        "identity_disagreement_weeks": 52,
        "differing_symbol_assignments": 520,
        "gate_eligible_new_entry_opportunities": 520,
    }

    passed = statistics._comparison_status(
        spec=spec,
        hac={"annualized_lower": 0.040, "annualized_upper": 0.060},
        bootstrap={"annualized_lower": 0.041, "annualized_upper": 0.061},
        concentration=0.25,
        weekly_median=0.001,
        maximum_top10_share=0.50,
        interventions=interventions,
        minimum_gate_opportunities=260,
    )
    falsified = statistics._comparison_status(
        spec=spec,
        hac={"annualized_lower": -0.010, "annualized_upper": 0.020},
        bootstrap={"annualized_lower": -0.012, "annualized_upper": 0.025},
        concentration=0.0,
        weekly_median=-0.001,
        maximum_top10_share=0.50,
        interventions=interventions,
        minimum_gate_opportunities=260,
    )
    inconclusive = statistics._comparison_status(
        spec=spec,
        hac={"annualized_lower": 0.020, "annualized_upper": 0.040},
        bootstrap={"annualized_lower": 0.025, "annualized_upper": 0.045},
        concentration=0.25,
        weekly_median=0.001,
        maximum_top10_share=0.50,
        interventions=interventions,
        minimum_gate_opportunities=260,
    )
    negative_median = statistics._comparison_status(
        spec=spec,
        hac={"annualized_lower": 0.040, "annualized_upper": 0.060},
        bootstrap={"annualized_lower": 0.041, "annualized_upper": 0.061},
        concentration=0.25,
        weekly_median=-0.0001,
        maximum_top10_share=0.50,
        interventions=interventions,
        minimum_gate_opportunities=260,
    )

    assert passed == ("PASS", [], True)
    assert falsified == ("FALSIFIED", [], True)
    assert inconclusive[0] == "INCONCLUSIVE"
    assert negative_median[0] == "INCONCLUSIVE"
    assert "weekly_active_median_not_positive" in negative_median[1]


def test_two_placebo_diagnostics_are_reported_but_do_not_change_overall_status():
    result = statistics.evaluate_statistics(_statistics_input())

    assert [result["comparisons"][identity]["status"] for identity in statistics.EXPECTED_COMPARISONS[:5]] == [
        "PASS"
    ] * 5
    assert [result["comparisons"][identity]["status"] for identity in statistics.EXPECTED_COMPARISONS[5:]] == [
        "FALSIFIED",
        "FALSIFIED",
    ]
    assert result["overall_status"] == "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY"


def test_daily_exposure_control_uses_mean_p95_and_max_not_a_signed_weekly_proxy():
    value = _statistics_input()
    exposure = np.full(len(DAILY_LABELS), 0.50)
    exposure[:15] = 0.54
    exposure[15] = 0.56
    value["arm_paths"]["F_2x"] = _arm_path("F_2x", _return_vectors()["F_2x"], exposure=exposure)

    result = statistics.evaluate_statistics(value)
    comparison = result["comparisons"]["F_2x_minus_R_match_2x"]

    assert comparison["mean_absolute_post_close_exposure_gap"] < 0.01
    assert comparison["p95_absolute_post_close_exposure_gap"] == pytest.approx(0.04)
    assert comparison["maximum_absolute_post_close_exposure_gap"] == pytest.approx(0.06)
    assert "mean_absolute_exposure_gap" not in comparison["control_failures"]
    assert "p95_absolute_exposure_gap" in comparison["control_failures"]
    assert "maximum_absolute_exposure_gap" in comparison["control_failures"]
    assert result["overall_status"] == "INVALID_CONTROL"


def test_turnover_gate_applies_only_to_preregistered_matched_pairs():
    value = _statistics_input()
    high_turnover = np.full(52, 0.20)
    low_turnover = np.zeros(52)
    vectors = _return_vectors()
    for arm in ("FC_gross", "FGR_gross", "FC_2x", "FGR_2x"):
        value["arm_paths"][arm] = _arm_path(arm, vectors[arm], turnover=high_turnover)
    for arm in ("FMA_gross", "FMGR_gross", "FMA_2x", "FMGR_2x"):
        value["arm_paths"][arm] = _arm_path(arm, vectors[arm], turnover=low_turnover)

    result = statistics.evaluate_statistics(value)
    matched = result["comparisons"]["FC_gross_minus_FGR_gross"]
    intervention = result["comparisons"]["FC_gross_minus_FMA_gross"]

    assert matched["turnover_control_applies"] is True
    assert matched["mean_absolute_one_way_turnover_gap"] == pytest.approx(0.0, abs=1e-15)
    assert intervention["turnover_control_applies"] is False
    assert intervention["mean_absolute_one_way_turnover_gap"] == pytest.approx(0.20)
    assert not any("turnover" in item for item in intervention["control_failures"])
    assert result["overall_status"] == "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY"


def test_seed_mcse_breach_is_invalid_control_without_adding_seeds_posthoc():
    value = _statistics_input()
    vectors = _return_vectors()
    offsets = np.linspace(-0.01, 0.01, 20)
    value["arm_paths"]["R_match_2x"] = _arm_path("R_match_2x", vectors["R_match_2x"], seed_offsets=offsets)

    result = statistics.evaluate_statistics(value)

    assert result["random_control_diagnostics"]["R_match_2x"]["seed_count"] == 20
    assert result["random_control_diagnostics"]["R_match_2x"]["annualized_seed_mean_mcse"] > 0.005
    assert "R_match_2x.annualized_seed_mean_mcse" in result["control_failures"]
    assert result["overall_status"] == "INVALID_CONTROL"


def test_insufficient_identity_intervention_is_not_mislabelled_as_falsification():
    value = _statistics_input()
    factor_symbols = value["arm_paths"]["F_2x"]["paths"]["primary"]["weekly_new_entry_identity_sets"]
    for seed_path in value["arm_paths"]["R_match_2x"]["paths"].values():
        seed_path["weekly_new_entry_identity_sets"] = deepcopy(factor_symbols)

    result = statistics.evaluate_statistics(value)
    factor = result["comparisons"]["F_2x_minus_R_match_2x"]

    assert factor["interventions"]["identity_disagreement_weeks"] == 0
    assert factor["interventions"]["differing_symbol_assignments"] == 0
    assert factor["intervention_sufficient"] is False
    assert factor["status"] == "INCONCLUSIVE"
    assert result["overall_status"] == "INSUFFICIENT_INTERVENTION"


def test_statistics_semantic_verifier_rejects_forged_overall_status():
    value = _statistics_input()
    result = statistics.evaluate_statistics(value)
    result["overall_status"] = "FALSIFIED_FACTOR"

    with pytest.raises(statistics.StatisticsError, match="semantic replay"):
        statistics.verify_statistics_result(result, value)


def test_risk_and_mechanism_report_is_recomputed_from_raw_paths():
    result = statistics.evaluate_statistics(_statistics_input())
    report = result["risk_and_mechanism_report"]

    assert report["schema"] == statistics.RISK_MECHANISM_REPORT_SCHEMA
    assert report["arm_order"] == list(statistics.EXPECTED_RISK_ARM_IDS)
    assert report["report_sha256"] == statistics.object_sha256(
        {key: item for key, item in report.items() if key != "report_sha256"}
    )
    fc = report["arms"]["FC_1x"]
    assert fc["daily_exposure_distribution"]["mean"] == pytest.approx(0.50)
    assert fc["weekly_turnover_distribution"]["p95"] == pytest.approx(0.10)
    assert fc["fill_ratio"]["share_weighted_fill_ratio"] == pytest.approx(0.50)
    assert fc["capacity_usage_distribution"]["mean"] == pytest.approx(0.50)
    state_five = fc["state_specific_gate_and_fill"]["5"]
    assert state_five["eligible_new_entry_count"] == 520
    assert state_five["allowed_new_entry_count"] == 520
    assert state_five["fill_count"] == 520
    assert state_five["holding_session_distribution"]["median"] == pytest.approx(2.0)
    assert state_five["forward_return_distribution"]["mean"] == pytest.approx(0.10)
    unwind = fc["post_window_unwind"]
    assert unwind["official_sessions_to_complete_distribution"]["mean"] == pytest.approx(1.0)
    assert unwind["cost_cny_distributions"]["total"]["mean"] == pytest.approx(18.0)

    attribution = report["gross_to_1x_to_2x_cost_attribution"]["FC"]
    assert attribution["annualized_explicit_cost_rate_by_scenario"]["gross"]["total"] == 0.0
    assert attribution["annualized_explicit_cost_rate_by_scenario"]["1x"]["total"] == pytest.approx(9.36e-5)
    assert attribution["annualized_explicit_cost_rate_by_scenario"]["2x"]["total"] == pytest.approx(1.872e-4)
    assert "risk_and_power" in result["comparisons"]["FC_2x_minus_FGR_2x"]


def test_risk_path_schema_rejects_precomputed_flags_missing_arms_and_capacity_tampering():
    precomputed = _statistics_input()
    precomputed["arm_paths"]["F_gross"]["paths"]["primary"]["fill_ratio_pass"] = True
    with pytest.raises(statistics.StatisticsError, match="exact V2.1 path keys"):
        statistics.evaluate_statistics(precomputed)

    missing = _statistics_input()
    missing["arm_paths"].pop("F_capacity_1x")
    with pytest.raises(statistics.StatisticsError, match="exact 24-arm"):
        statistics.evaluate_statistics(missing)

    over_capacity = _statistics_input()
    order = over_capacity["arm_paths"]["F_1x"]["paths"]["primary"]["order_attempts"][0]
    order["fill_notional_cny"] = 2_001.0
    with pytest.raises(statistics.StatisticsError, match="exceeds the frozen dual opening capacity"):
        statistics.evaluate_statistics(over_capacity)


def test_state_records_and_post_window_unwind_fail_closed_under_adversarial_edits():
    invalid_state = _statistics_input()
    observation = invalid_state["arm_paths"]["FC_1x"]["paths"]["primary"]["state_gate_observations"][0]
    observation["regime"] = 3
    with pytest.raises(statistics.StatisticsError, match="changes the frozen Chan state gate"):
        statistics.evaluate_statistics(invalid_state)

    hidden_buy = _statistics_input()
    observation = hidden_buy["arm_paths"]["FC_1x"]["paths"]["primary"]["state_gate_observations"][0]
    observation.update(
        requested_shares=0,
        filled_shares=0,
        entry_fill_notional_cny=0.0,
        entry_price=None,
        exit_or_window_session=None,
        exit_or_window_price=None,
    )
    result = statistics.evaluate_statistics(hidden_buy)
    assert (
        result["risk_and_mechanism_report"]["arms"]["FC_1x"]["state_specific_gate_and_fill"]["5"]["fill_count"] == 519
    )

    fabricated_state_buy = _statistics_input()
    observation = fabricated_state_buy["arm_paths"]["FC_1x"]["paths"]["primary"]["state_gate_observations"][0]
    observation["symbol"] = "ABSENT-FROM-RAW-ORDERS"
    with pytest.raises(statistics.StatisticsError, match="does not bind to its raw buy order"):
        statistics.evaluate_statistics(fabricated_state_buy)

    incomplete_unwind = _statistics_input()
    session = incomplete_unwind["arm_paths"]["F_1x"]["paths"]["primary"]["post_window_unwind"]["sessions"][0]
    session["sell_orders"][0]["filled_shares"] = 50
    session["remaining_positions"] = [{"symbol": "UNWIND", "shares": 50}]
    session["fill_notional_cny"] = 550.0
    with pytest.raises(statistics.StatisticsError, match="continue through complete real fill"):
        statistics.evaluate_statistics(incomplete_unwind)


def test_symbol_level_unwind_conserves_a_delist_share_conversion_before_selling_target():
    unwind = {
        "confirmation_end_session": DAILY_LABELS[-1],
        "starting_positions": [{"symbol": "A", "shares": 100}],
        "pending_share_conversions": [],
        "sessions": [
            {
                "session": "2027-D001",
                "sell_orders": [],
                "terminal_dispositions": [
                    {
                        "event_index": 100,
                        "action_id": "swap-A-B",
                        "action_type": "delist_share",
                        "symbol": "A",
                        "disposed_shares": 100,
                    }
                ],
                "terminal_share_receipts": [],
                "remaining_positions": [],
                "fill_notional_cny": 0.0,
                "commission_cny": 0.0,
                "transfer_fee_cny": 0.0,
                "stamp_duty_cny": 0.0,
                "slippage_cost_cny": 0.0,
            },
            {
                "session": "2027-D002",
                "sell_orders": [{"symbol": "B", "requested_shares": 80, "filled_shares": 80}],
                "terminal_dispositions": [],
                "terminal_share_receipts": [
                    {
                        "event_index": 110,
                        "action_id": "swap-A-B",
                        "source_symbol": "A",
                        "target_symbol": "B",
                        "received_shares": 80,
                    }
                ],
                "remaining_positions": [],
                "fill_notional_cny": 880.0,
                "commission_cny": 5.0,
                "transfer_fee_cny": 1.0,
                "stamp_duty_cny": 2.0,
                "slippage_cost_cny": 10.0,
            },
        ],
    }
    _bind_unwind_metadata(unwind, pending_counts=[1, 0])

    summary = statistics._parse_unwind(unwind, "unwind", DAILY_LABELS[-1])
    assert summary.starting_shares == 100
    assert summary.official_sessions_to_complete == 2
    assert summary.fill_notional_cny == 880.0

    forged = deepcopy(unwind)
    forged["sessions"][1]["terminal_share_receipts"][0]["source_symbol"] = "FORGED"
    with pytest.raises(statistics.StatisticsError, match="pending conversion"):
        statistics._parse_unwind(forged, "unwind", DAILY_LABELS[-1])

    partial_disposition = deepcopy(unwind)
    partial_disposition["sessions"][0]["terminal_dispositions"][0]["disposed_shares"] = 50
    partial_disposition["sessions"][0]["remaining_positions"] = [{"symbol": "A", "shares": 50}]
    with pytest.raises(statistics.StatisticsError, match="complete symbol position"):
        statistics._parse_unwind(partial_disposition, "unwind", DAILY_LABELS[-1])

    already_pending = {
        "confirmation_end_session": DAILY_LABELS[-1],
        "starting_positions": [],
        "pending_share_conversions": [{"action_id": "pre-window-swap", "source_symbol": "A"}],
        "sessions": [
            {
                "session": "2027-D001",
                "sell_orders": [{"symbol": "B", "requested_shares": 80, "filled_shares": 80}],
                "terminal_dispositions": [],
                "terminal_share_receipts": [
                    {
                        "event_index": 120,
                        "action_id": "pre-window-swap",
                        "source_symbol": "A",
                        "target_symbol": "B",
                        "received_shares": 80,
                    }
                ],
                "remaining_positions": [],
                "fill_notional_cny": 880.0,
                "commission_cny": 5.0,
                "transfer_fee_cny": 1.0,
                "stamp_duty_cny": 2.0,
                "slippage_cost_cny": 10.0,
            }
        ],
    }
    _bind_unwind_metadata(already_pending, starting_pending_count=1, pending_counts=[0])
    pending_summary = statistics._parse_unwind(already_pending, "unwind", DAILY_LABELS[-1])
    assert pending_summary.starting_shares == 0
    assert pending_summary.official_sessions_to_complete == 1


def test_post_completion_heartbeat_is_economically_inert_and_not_counted_as_unwind_time():
    unwind = deepcopy(_statistics_input()["arm_paths"]["F_1x"]["paths"]["primary"]["post_window_unwind"])
    completion_state = deepcopy(unwind["completion_economic_state"])
    heartbeat = {
        "session": "2027-D002",
        "state_sha256": unwind["completion_state_sha256"],
        **completion_state,
        "economic_state_sha256": unwind["completion_economic_state_sha256"],
        "administrative_events": [],
        "session_events_root_sha256": statistics.object_sha256([]),
        "session_event_count": 0,
    }
    unwind["post_completion_heartbeats"] = [heartbeat]
    summary = statistics._parse_unwind(unwind, "unwind", DAILY_LABELS[-1])
    assert summary.official_sessions_to_complete == 1

    forged = deepcopy(unwind)
    forged_heartbeat = forged["post_completion_heartbeats"][0]
    forged_heartbeat["cash_cny"] += 1.0
    forged_heartbeat["economic_state_sha256"] = statistics.object_sha256(
        {field: forged_heartbeat[field] for field in completion_state}
    )
    with pytest.raises(statistics.StatisticsError, match="changes economic state"):
        statistics._parse_unwind(forged, "unwind", DAILY_LABELS[-1])


def test_semantic_verifier_rejects_forged_risk_or_mechanism_output():
    value = _statistics_input()
    result = statistics.evaluate_statistics(value)
    result["risk_and_mechanism_report"]["arms"]["FC_1x"]["fill_ratio"]["share_weighted_fill_ratio"] = 1.0

    with pytest.raises(statistics.StatisticsError, match="semantic replay"):
        statistics.verify_statistics_result(result, value)
