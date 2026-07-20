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
SEEDED_ARMS = {"R_match_2x", "FGR_gross", "FGR_2x", "FMGR_gross", "FMGR_2x"}


def _path_payload(
    returns: np.ndarray,
    *,
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
    return {
        "weekly_returns": np.asarray(returns, dtype=float).tolist(),
        "daily_post_close_gross_exposure": exposure.tolist(),
        "weekly_one_way_turnover": turnover.tolist(),
        "weekly_new_entry_identity_sets": weekly_selections,
        "weekly_gate_eligible_new_entry_opportunities": [opportunities] * 52,
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
    return {
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


def _statistics_input() -> dict:
    vectors = _return_vectors()
    return {
        "schema": statistics.STATISTICS_INPUT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "trial_id": PROTOCOL_SHA256,
        "window_sha256": hashlib.sha256(b"frozen-first-52-weeks").hexdigest(),
        "comparison_registry": deepcopy(PROTOCOL["statistics"]["comparison_registry"]),
        "arm_paths": {arm: _arm_path(arm, values) for arm, values in vectors.items()},
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
