"""Preregistered V2.1 weekly statistics and three-way evidence decisions.

The functions in this module operate on already replayed portfolio paths.  They
do not accept precomputed t statistics, confidence intervals, or pass/fail
booleans.  Every formal comparison is reconstructed from the first 52 frozen
weekly returns, including equal-weight aggregation across every frozen random
seed before time-series inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

STATISTICS_INPUT_SCHEMA = "xs_chan_statistics_input_v2_1"
STATISTICS_RESULT_SCHEMA = "xs_chan_statistics_result_v2_1"
WEEKLY_PATH_SCHEMA = "xs_chan_weekly_path_v2_1"
EXPECTED_COMPLETE_WEEKS = 52
EXPECTED_RANDOM_SEEDS = tuple(range(20260720, 20260740))
EXPECTED_COMPARISONS = (
    "F_2x_minus_R_match_2x",
    "FC_gross_minus_FGR_gross",
    "FC_2x_minus_FGR_2x",
    "FC_gross_minus_FMA_gross",
    "FC_2x_minus_FMA_2x",
    "FMA_gross_minus_FMGR_gross",
    "FMA_2x_minus_FMGR_2x",
)
EXPECTED_CONTROL_THRESHOLDS = {
    "maximum_top10_positive_weeks_share": 0.50,
    "maximum_mean_absolute_exposure_gap": 0.01,
    "maximum_p95_absolute_exposure_gap": 0.03,
    "maximum_absolute_exposure_gap": 0.05,
    "maximum_mean_absolute_turnover_gap": 0.025,
    "maximum_p95_absolute_turnover_gap": 0.10,
    "maximum_random_annualized_seed_mean_mcse": 0.005,
    "maximum_bootstrap_quantile_mcse_annualized": 0.0025,
    "minimum_gate_eligible_new_entry_opportunities": 260.0,
}
EXPECTED_COMPARISON_CORE = {
    "F_2x_minus_R_match_2x": ("factor_primary", "F_2x", "R_match_2x", "2x", True, 0.03, True),
    "FC_gross_minus_FGR_gross": (
        "chan_identity_primary",
        "FC_gross",
        "FGR_gross",
        "gross",
        True,
        0.02,
        True,
    ),
    "FC_2x_minus_FGR_2x": ("chan_cost_robustness", "FC_2x", "FGR_2x", "2x", True, 0.015, True),
    "FC_gross_minus_FMA_gross": (
        "chan_specificity_primary",
        "FC_gross",
        "FMA_gross",
        "gross",
        True,
        0.01,
        False,
    ),
    "FC_2x_minus_FMA_2x": (
        "chan_specificity_cost_robustness",
        "FC_2x",
        "FMA_2x",
        "2x",
        True,
        0.01,
        False,
    ),
    "FMA_gross_minus_FMGR_gross": (
        "ma_placebo_identity_diagnostic",
        "FMA_gross",
        "FMGR_gross",
        "gross",
        False,
        0.01,
        True,
    ),
    "FMA_2x_minus_FMGR_2x": (
        "ma_placebo_cost_diagnostic",
        "FMA_2x",
        "FMGR_2x",
        "2x",
        False,
        0.01,
        True,
    ),
}
EXPECTED_PASS_RULE = (
    "hac_lower_95_and_bootstrap_q05_strictly_greater_than_sesoi_with_positive_weekly_median_and_top10_share_lte_0_50"
)
EXPECTED_FALSIFIED_RULE = "hac_upper_95_and_bootstrap_q95_strictly_less_than_sesoi"


class StatisticsError(ValueError):
    """Raised when formal statistical input is incomplete or not preregistered."""


class DegenerateInferenceError(StatisticsError):
    """Raised when a HAC/variance calculation cannot support valid inference."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StatisticsError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StatisticsError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticsError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number):
        raise StatisticsError(f"{label} must be finite numeric")
    return number


def _as_weekly_array(value: Sequence[Any], label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result <= -1.0):
        raise StatisticsError(f"{label} contains a return <= -100%")
    return result


def _as_exposure_array(value: Sequence[Any], label: str, expected_length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_length:
        raise StatisticsError(f"{label} must contain exactly {expected_length} daily observations")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result < -1e-12) or np.any(result > 1.0 + 1e-12):
        raise StatisticsError(f"{label} must be post-close gross exposure in [0, 1]")
    return result


def _as_turnover_array(value: Sequence[Any], label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result = np.asarray([_finite(item, f"{label}[{index}]") for index, item in enumerate(value)], dtype=float)
    if np.any(result < 0):
        raise StatisticsError(f"{label} must contain non-negative one-way turnover")
    return result


def _validate_week_labels(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} labels")
    labels = tuple(str(item) for item in value)
    if any(not item for item in labels) or len(labels) != len(set(labels)) or labels != tuple(sorted(labels)):
        raise StatisticsError(f"{label} must be unique, non-empty, and increasing")
    return labels


def _validate_daily_labels(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain the full daily confirmation window")
    labels = tuple(str(item) for item in value)
    if any(not item for item in labels) or len(labels) != len(set(labels)) or labels != tuple(sorted(labels)):
        raise StatisticsError(f"{label} must be unique, non-empty, and increasing")
    return labels


def _validate_selection_sets(value: Any, label: str) -> tuple[frozenset[str], ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_COMPLETE_WEEKS:
        raise StatisticsError(f"{label} must contain exactly {EXPECTED_COMPLETE_WEEKS} weeks")
    result: list[frozenset[str]] = []
    for week, symbols in enumerate(value):
        if not isinstance(symbols, list) or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise StatisticsError(f"{label}[{week}] must be a list of symbols")
        if len(symbols) != len(set(symbols)):
            raise StatisticsError(f"{label}[{week}] contains duplicate symbols")
        result.append(frozenset(symbols))
    return tuple(result)


@dataclass(frozen=True)
class ArmPath:
    arm_id: str
    week_labels: tuple[str, ...]
    daily_labels: tuple[str, ...]
    returns: np.ndarray
    exposure: np.ndarray
    turnover: np.ndarray
    selections: tuple[frozenset[str], ...]
    gate_opportunities: np.ndarray
    seed_returns: np.ndarray | None
    seed_exposure: np.ndarray | None
    seed_turnover: np.ndarray | None
    seed_selections: tuple[tuple[frozenset[str], ...], ...] | None
    seed_gate_opportunities: np.ndarray | None
    seed_ids: tuple[int, ...]

    @property
    def seeded(self) -> bool:
        return self.seed_returns is not None


def _parse_arm_path(arm_id: str, value: Any) -> ArmPath:
    if not isinstance(value, Mapping):
        raise StatisticsError(f"arm_paths.{arm_id} must be an object")
    common = {"schema", "arm_id", "week_labels", "daily_labels", "aggregation", "paths"}
    if set(value) != common or value["schema"] != WEEKLY_PATH_SCHEMA or str(value["arm_id"]) != arm_id:
        raise StatisticsError(f"arm_paths.{arm_id} has an invalid V2.1 schema")
    labels = _validate_week_labels(value["week_labels"], f"arm_paths.{arm_id}.week_labels")
    daily_labels = _validate_daily_labels(value["daily_labels"], f"arm_paths.{arm_id}.daily_labels")
    aggregation = str(value["aggregation"])
    paths = value["paths"]
    if not isinstance(paths, Mapping):
        raise StatisticsError(f"arm_paths.{arm_id}.paths must be an object")

    def parse_one(
        path: Any, path_label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[frozenset[str], ...], np.ndarray]:
        exact = {
            "weekly_returns",
            "daily_post_close_gross_exposure",
            "weekly_one_way_turnover",
            "weekly_new_entry_identity_sets",
            "weekly_gate_eligible_new_entry_opportunities",
        }
        if not isinstance(path, Mapping) or set(path) != exact:
            raise StatisticsError(f"{path_label} must have exact V2.1 path keys")
        opportunities_raw = path["weekly_gate_eligible_new_entry_opportunities"]
        if (
            not isinstance(opportunities_raw, list)
            or len(opportunities_raw) != EXPECTED_COMPLETE_WEEKS
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in opportunities_raw)
        ):
            raise StatisticsError(f"{path_label}.weekly_gate_eligible_new_entry_opportunities is invalid")
        return (
            _as_weekly_array(path["weekly_returns"], f"{path_label}.weekly_returns"),
            _as_exposure_array(
                path["daily_post_close_gross_exposure"],
                f"{path_label}.daily_post_close_gross_exposure",
                len(daily_labels),
            ),
            _as_turnover_array(path["weekly_one_way_turnover"], f"{path_label}.weekly_one_way_turnover"),
            _validate_selection_sets(
                path["weekly_new_entry_identity_sets"], f"{path_label}.weekly_new_entry_identity_sets"
            ),
            np.asarray(opportunities_raw, dtype=int),
        )

    if aggregation == "single_path":
        if set(paths) != {"primary"}:
            raise StatisticsError(f"single arm {arm_id} must contain only paths.primary")
        returns, exposure, turnover, selections, gate_opportunities = parse_one(
            paths["primary"], f"arm_paths.{arm_id}.paths.primary"
        )
        return ArmPath(
            arm_id,
            labels,
            daily_labels,
            returns,
            exposure,
            turnover,
            selections,
            gate_opportunities,
            None,
            None,
            None,
            None,
            None,
            (),
        )
    if aggregation != "equal_mean_all_20_frozen_seeds_before_statistics":
        raise StatisticsError(f"unknown aggregation for arm {arm_id}")
    expected_keys = {str(seed) for seed in EXPECTED_RANDOM_SEEDS}
    if set(paths) != expected_keys:
        raise StatisticsError(f"seeded arm {arm_id} must contain exactly seeds 20260720..20260739")
    parsed = [parse_one(paths[str(seed)], f"arm_paths.{arm_id}.paths.{seed}") for seed in EXPECTED_RANDOM_SEEDS]
    seed_returns = np.stack([item[0] for item in parsed], axis=0)
    seed_exposure = np.stack([item[1] for item in parsed], axis=0)
    seed_turnover = np.stack([item[2] for item in parsed], axis=0)
    seed_selections = tuple(item[3] for item in parsed)
    seed_gate_opportunities = np.stack([item[4] for item in parsed], axis=0)
    # Equal seed means are computed here, never accepted from a caller.
    return ArmPath(
        arm_id,
        labels,
        daily_labels,
        seed_returns.mean(axis=0),
        seed_exposure.mean(axis=0),
        seed_turnover.mean(axis=0),
        tuple(frozenset().union(*(seed[week] for seed in seed_selections)) for week in range(EXPECTED_COMPLETE_WEEKS)),
        seed_gate_opportunities.mean(axis=0),
        seed_returns,
        seed_exposure,
        seed_turnover,
        seed_selections,
        seed_gate_opportunities,
        EXPECTED_RANDOM_SEEDS,
    )


def newey_west_mean_interval(
    values: Sequence[float], *, lag: int = 4, one_sided_confidence: float = 0.95
) -> dict[str, float]:
    """Return a PSD Newey-West interval for a weekly mean.

    Bartlett weights yield a positive-semidefinite estimator.  A materially
    non-positive long-run variance is an invalid inference condition; it is not
    silently converted into a zero t statistic.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) <= lag + 1 or not np.isfinite(x).all():
        raise DegenerateInferenceError("HAC input is too short or non-finite")
    if lag < 0 or lag >= len(x):
        raise DegenerateInferenceError("HAC lag is outside the valid range")
    mean = float(x.mean())
    centered = x - mean
    n = len(x)
    gamma0 = float(centered @ centered / n)
    lrv = gamma0
    for offset in range(1, lag + 1):
        gamma = float(centered[offset:] @ centered[:-offset] / n)
        lrv += 2.0 * (1.0 - offset / (lag + 1.0)) * gamma
    scale = max(gamma0, np.finfo(float).tiny)
    if lrv < -1e-12 * scale:
        raise DegenerateInferenceError("Newey-West long-run variance is materially negative")
    if lrv <= np.finfo(float).eps * max(scale, 1.0):
        raise DegenerateInferenceError("Newey-West long-run variance is non-positive")
    standard_error = math.sqrt(lrv / n)
    if not 0.5 < one_sided_confidence < 1.0:
        raise DegenerateInferenceError("one-sided confidence must be in (0.5, 1)")
    z = NormalDist().inv_cdf(one_sided_confidence)
    return {
        "weekly_mean": mean,
        "long_run_variance": lrv,
        "weekly_standard_error": standard_error,
        "z_score": mean / standard_error,
        "annualized_mean": mean * 52.0,
        "annualized_lower": (mean - z * standard_error) * 52.0,
        "annualized_upper": (mean + z * standard_error) * 52.0,
        "one_sided_confidence": one_sided_confidence,
        "lag": lag,
    }


def _derived_bootstrap_seed(
    *, method_seed: int, identity: str, window_sha256: str, block_length: int, draws: int
) -> int:
    material = {
        "domain": "xs_chan_v2_1_circular_block_bootstrap",
        "method_seed": int(method_seed),
        "identity": identity,
        "window_sha256": window_sha256,
        "block_length": int(block_length),
        "draws": int(draws),
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(material)).digest()[:8], "big", signed=False)


def circular_block_bootstrap_interval(
    values: Sequence[float],
    *,
    identity: str,
    window_sha256: str,
    method_seed: int = 20260720,
    block_length: int = 4,
    draws: int = 20_000,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, float | int]:
    """Circular block bootstrap with domain-separated deterministic RNG."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) != EXPECTED_COMPLETE_WEEKS or not np.isfinite(x).all():
        raise StatisticsError("bootstrap requires exactly 52 finite weekly values")
    if not isinstance(block_length, int) or block_length <= 0 or block_length > len(x):
        raise StatisticsError("bootstrap block_length is invalid")
    if not isinstance(draws, int) or draws < 1_000:
        raise StatisticsError("bootstrap draws must be at least 1000")
    if not 0 < lower_quantile < upper_quantile < 1:
        raise StatisticsError("bootstrap quantiles are invalid")
    _require_sha256(window_sha256, "window_sha256")
    derived_seed = _derived_bootstrap_seed(
        method_seed=method_seed,
        identity=identity,
        window_sha256=window_sha256,
        block_length=block_length,
        draws=draws,
    )
    rng = np.random.default_rng(derived_seed)
    blocks_needed = math.ceil(len(x) / block_length)
    offsets = np.arange(block_length, dtype=int)
    means = np.empty(draws, dtype=float)
    # Chunking keeps the formal 20k draw run fast without a large transient cube.
    chunk = 1_000
    for first in range(0, draws, chunk):
        size = min(chunk, draws - first)
        starts = rng.integers(0, len(x), size=(size, blocks_needed))
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(x)
        samples = x[indices.reshape(size, -1)[:, : len(x)]]
        means[first : first + size] = samples.mean(axis=1) * 52.0
    lower, upper = np.quantile(means, [lower_quantile, upper_quantile], method="linear")
    # Quantile Monte-Carlo error is estimated from fixed contiguous batches.  It
    # is diagnostic and can be bounded by the protocol without a second RNG.
    batch_count = min(20, draws // 250)
    batch_size = draws // batch_count
    trimmed = means[: batch_count * batch_size].reshape(batch_count, batch_size)
    batch_quantiles = np.quantile(trimmed, [lower_quantile, upper_quantile], axis=1, method="linear")
    lower_mcse = float(np.std(batch_quantiles[0], ddof=1) / math.sqrt(batch_count))
    upper_mcse = float(np.std(batch_quantiles[1], ddof=1) / math.sqrt(batch_count))
    return {
        "annualized_lower": float(lower),
        "annualized_upper": float(upper),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "block_length": block_length,
        "draws": draws,
        "derived_seed": derived_seed,
        "lower_quantile_mcse": lower_mcse,
        "upper_quantile_mcse": upper_mcse,
    }


def top_positive_concentration(values: Sequence[float], count: int = 10) -> float:
    positive = np.sort(np.asarray(values, dtype=float)[np.asarray(values, dtype=float) > 0])[::-1]
    if len(positive) == 0:
        return 0.0
    return float(positive[:count].sum() / positive.sum())


def _active_risk_and_power(values: np.ndarray, sesoi: float) -> dict[str, Any]:
    sample_std = float(np.std(values, ddof=1))
    ratio: float | None = float(math.sqrt(52.0) * np.mean(values) / sample_std) if sample_std > 0 else None
    active_nav = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    active_drawdown = active_nav / np.maximum.accumulate(active_nav) - 1.0
    z_sum = 1.6448536269514722 + 0.8416212335729143
    mde_excess = z_sum * sample_std * math.sqrt(52.0)
    effect_grid = [sesoi, sesoi + mde_excess / 2.0, sesoi + mde_excess, sesoi + 2.0 * mde_excess]
    power_curve: list[dict[str, float | None]] = []
    for annualized_effect in effect_grid:
        if sample_std <= 0:
            power = None
        else:
            noncentral = (annualized_effect - sesoi) / (sample_std * math.sqrt(52.0))
            power = NormalDist().cdf(noncentral - 1.6448536269514722)
        power_curve.append({"annualized_effect": annualized_effect, "iid_planning_power": power})
    return {
        "active_information_ratio": ratio,
        "active_path_maximum_drawdown_including_initial_nav": float(np.min(active_drawdown)),
        "observed_weekly_active_sample_std": sample_std,
        "iid_mde_annualized_excess_over_sesoi_at_80pct_power": mde_excess,
        "iid_minimum_effect_for_80pct_power": sesoi + mde_excess,
        "power_curve": power_curve,
        "dependence_warning": "HAC/block dependence can reduce effective power; curve is IID planning only",
    }


def _random_control_diagnostics(path: ArmPath) -> dict[str, float | int | bool]:
    if not path.seeded or path.seed_returns is None:
        return {
            "seed_count": 0,
            "annualized_seed_mean_mcse": 0.0,
            "maximum_weekly_seed_mean_mcse": 0.0,
        }
    weekly_mcse = path.seed_returns.std(axis=0, ddof=1) / math.sqrt(path.seed_returns.shape[0])
    return {
        "seed_count": int(path.seed_returns.shape[0]),
        "annualized_seed_mean_mcse": float(math.sqrt(float(np.sum(np.square(weekly_mcse))))),
        "maximum_weekly_seed_mean_mcse": float(weekly_mcse.max()),
    }


def _intervention_counts(left: ArmPath, right: ArmPath) -> dict[str, float | int]:
    disagreement_weeks = 0
    assignments = 0.0
    if right.seed_selections is None:
        for week in range(EXPECTED_COMPLETE_WEEKS):
            difference = left.selections[week].symmetric_difference(right.selections[week])
            assignments += len(difference) / 2.0
            disagreement_weeks += int(bool(difference))
    else:
        for week in range(EXPECTED_COMPLETE_WEEKS):
            per_seed_difference: list[float] = []
            for seed in right.seed_selections:
                per_seed_difference.append(len(left.selections[week].symmetric_difference(seed[week])) / 2.0)
            assignments += float(np.mean(per_seed_difference))
            disagreement_weeks += int(any(value > 0 for value in per_seed_difference))
    return {
        "identity_disagreement_weeks": disagreement_weeks,
        "differing_symbol_assignments": assignments,
        "gate_eligible_new_entry_opportunities": int(np.sum(left.gate_opportunities)),
    }


def _validate_comparison_registry(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or tuple(value.keys()) != EXPECTED_COMPARISONS:
        raise StatisticsError("comparison_registry must preserve the exact seven preregistered comparison order")
    required = {
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
    result: dict[str, dict[str, Any]] = {}
    for identity in EXPECTED_COMPARISONS:
        spec = value[identity]
        if not isinstance(spec, Mapping) or set(spec) != required:
            raise StatisticsError(f"comparison_registry.{identity} must have exact V2.1 fields")
        if spec["effect_metric"] != "annualized_mean_weekly_active_return":
            raise StatisticsError(f"comparison_registry.{identity} has an unregistered effect metric")
        if spec["otherwise_status"] != "INCONCLUSIVE":
            raise StatisticsError(f"comparison_registry.{identity} must use INCONCLUSIVE fallback")
        if type(spec["decision_gating"]) is not bool:
            raise StatisticsError(f"comparison_registry.{identity}.decision_gating must be boolean")
        if type(spec["seed_mcse_required"]) is not bool:
            raise StatisticsError(f"comparison_registry.{identity}.seed_mcse_required must be boolean")
        sesoi = _finite(spec["sesoi_annualized"], f"comparison_registry.{identity}.sesoi_annualized")
        if sesoi <= 0:
            raise StatisticsError(f"comparison_registry.{identity}.sesoi_annualized must be positive")
        for key in ("minimum_identity_disagreement_weeks", "minimum_differing_symbol_assignments"):
            if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] <= 0:
                raise StatisticsError(f"comparison_registry.{identity}.{key} must be positive integer")
        if not str(spec["arm"]) or not str(spec["comparator"]):
            raise StatisticsError(f"comparison_registry.{identity} arm/comparator must be non-empty")
        actual_core = (
            str(spec["role"]),
            str(spec["arm"]),
            str(spec["comparator"]),
            str(spec["cost_basis"]),
            bool(spec["decision_gating"]),
            float(spec["sesoi_annualized"]),
            bool(spec["seed_mcse_required"]),
        )
        if actual_core != EXPECTED_COMPARISON_CORE[identity]:
            raise StatisticsError(f"comparison_registry.{identity} differs from the frozen comparison core")
        if spec["pass_rule"] != EXPECTED_PASS_RULE or spec["falsified_rule"] != EXPECTED_FALSIFIED_RULE:
            raise StatisticsError(f"comparison_registry.{identity} changes a frozen decision rule")
        if spec["minimum_identity_disagreement_weeks"] != 26 or spec["minimum_differing_symbol_assignments"] != 260:
            raise StatisticsError(f"comparison_registry.{identity} changes frozen intervention thresholds")
        result[identity] = dict(spec)
    if [identity for identity in EXPECTED_COMPARISONS if result[identity]["decision_gating"]] != list(
        EXPECTED_COMPARISONS[:5]
    ):
        raise StatisticsError("exactly the first five comparisons must be decision-gating")
    return result


def _comparison_status(
    *,
    spec: Mapping[str, Any],
    hac: Mapping[str, float],
    bootstrap: Mapping[str, float | int],
    concentration: float,
    weekly_median: float,
    maximum_top10_share: float,
    interventions: Mapping[str, int],
    minimum_gate_opportunities: int,
) -> tuple[str, list[str], bool]:
    support_failures: list[str] = []
    if interventions["identity_disagreement_weeks"] < int(spec["minimum_identity_disagreement_weeks"]):
        support_failures.append("minimum_identity_disagreement_weeks")
    if interventions["differing_symbol_assignments"] < int(spec["minimum_differing_symbol_assignments"]):
        support_failures.append("minimum_differing_symbol_assignments")
    if interventions["gate_eligible_new_entry_opportunities"] < minimum_gate_opportunities:
        support_failures.append("minimum_gate_eligible_new_entry_opportunities")
    if support_failures:
        return "INCONCLUSIVE", support_failures, False
    sesoi = float(spec["sesoi_annualized"])
    lower = min(float(hac["annualized_lower"]), float(bootstrap["annualized_lower"]))
    upper = max(float(hac["annualized_upper"]), float(bootstrap["annualized_upper"]))
    if lower > sesoi and concentration <= maximum_top10_share and weekly_median > 0:
        return "PASS", [], True
    if upper < sesoi:
        return "FALSIFIED", [], True
    failures = []
    if lower <= sesoi:
        failures.append("lower_bound_not_above_sesoi")
    if concentration > maximum_top10_share:
        failures.append("positive_week_concentration")
    if weekly_median <= 0:
        failures.append("weekly_active_median_not_positive")
    return "INCONCLUSIVE", failures, True


def evaluate_statistics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all seven comparisons and the preregistered overall status."""

    exact = {
        "schema",
        "protocol_sha256",
        "trial_id",
        "window_sha256",
        "comparison_registry",
        "arm_paths",
        "controls",
        "bootstrap",
    }
    if not isinstance(value, Mapping) or set(value) != exact:
        raise StatisticsError("statistics input must have exact V2.1 top-level keys")
    if value["schema"] != STATISTICS_INPUT_SCHEMA:
        raise StatisticsError(f"statistics schema must be {STATISTICS_INPUT_SCHEMA!r}")
    _require_sha256(value["protocol_sha256"], "protocol_sha256")
    _require_sha256(value["window_sha256"], "window_sha256")
    if not str(value["trial_id"]):
        raise StatisticsError("trial_id must be non-empty")
    registry = _validate_comparison_registry(value["comparison_registry"])
    if not isinstance(value["arm_paths"], Mapping):
        raise StatisticsError("arm_paths must be an object")
    needed_arms = {str(spec[key]) for spec in registry.values() for key in ("arm", "comparator")}
    if set(value["arm_paths"]) != needed_arms:
        raise StatisticsError("arm_paths must equal the exact arm domain used by the seven comparisons")
    paths = {arm: _parse_arm_path(arm, value["arm_paths"][arm]) for arm in sorted(needed_arms)}
    labels = {path.week_labels for path in paths.values()}
    if len(labels) != 1:
        raise StatisticsError("all arm paths must use the identical frozen 52-week labels")
    daily_labels = {path.daily_labels for path in paths.values()}
    if len(daily_labels) != 1:
        raise StatisticsError("all arm paths must use identical daily exposure labels")

    controls_expected = {
        "maximum_top10_positive_weeks_share",
        "maximum_mean_absolute_exposure_gap",
        "maximum_p95_absolute_exposure_gap",
        "maximum_absolute_exposure_gap",
        "maximum_mean_absolute_turnover_gap",
        "maximum_p95_absolute_turnover_gap",
        "maximum_random_annualized_seed_mean_mcse",
        "maximum_bootstrap_quantile_mcse_annualized",
        "minimum_gate_eligible_new_entry_opportunities",
    }
    controls = value["controls"]
    if not isinstance(controls, Mapping) or set(controls) != controls_expected:
        raise StatisticsError("controls must have exact V2.1 fields")
    thresholds = {key: _finite(controls[key], f"controls.{key}") for key in controls_expected}
    if any(number < 0 for number in thresholds.values()):
        raise StatisticsError("control thresholds must be non-negative")
    if thresholds["maximum_top10_positive_weeks_share"] > 1:
        raise StatisticsError("maximum_top10_positive_weeks_share must be <= 1")
    if thresholds != EXPECTED_CONTROL_THRESHOLDS:
        raise StatisticsError("control thresholds differ from the frozen V2.1 protocol")

    bootstrap_expected = {"method_seed", "block_length_weeks", "draws", "lower_quantile", "upper_quantile"}
    bootstrap_config = value["bootstrap"]
    if not isinstance(bootstrap_config, Mapping) or set(bootstrap_config) != bootstrap_expected:
        raise StatisticsError("bootstrap must have exact V2.1 fields")
    method_seed = int(bootstrap_config["method_seed"])
    block_length = int(bootstrap_config["block_length_weeks"])
    draws = int(bootstrap_config["draws"])
    lower_quantile = _finite(bootstrap_config["lower_quantile"], "bootstrap.lower_quantile")
    upper_quantile = _finite(bootstrap_config["upper_quantile"], "bootstrap.upper_quantile")
    if (method_seed, block_length, draws, lower_quantile, upper_quantile) != (20260720, 4, 20_000, 0.05, 0.95):
        raise StatisticsError("bootstrap configuration differs from the frozen V2.1 protocol")

    random_diagnostics = {arm: _random_control_diagnostics(path) for arm, path in paths.items() if path.seeded}
    random_failures: list[str] = []
    for arm, diagnostic in random_diagnostics.items():
        if diagnostic["seed_count"] != len(EXPECTED_RANDOM_SEEDS):
            random_failures.append(f"{arm}.seed_count")
        if diagnostic["annualized_seed_mean_mcse"] > thresholds["maximum_random_annualized_seed_mean_mcse"]:
            random_failures.append(f"{arm}.annualized_seed_mean_mcse")

    comparisons: dict[str, dict[str, Any]] = {}
    global_control_failures = list(random_failures)
    for identity in EXPECTED_COMPARISONS:
        spec = registry[identity]
        left = paths[str(spec["arm"])]
        right = paths[str(spec["comparator"])]
        active = left.returns - right.returns
        exposure_gaps = np.abs(left.exposure - right.exposure)
        exposure_gap = float(np.mean(exposure_gaps))
        exposure_p95 = float(np.quantile(exposure_gaps, 0.95, method="linear"))
        exposure_max = float(np.max(exposure_gaps))
        turnover_gaps = np.abs(left.turnover - right.turnover)
        turnover_gap = float(np.mean(turnover_gaps))
        turnover_p95 = float(np.quantile(turnover_gaps, 0.95, method="linear"))
        interventions = _intervention_counts(left, right)
        comparison_control_failures: list[str] = []
        if bool(spec["decision_gating"]):
            if exposure_gap > thresholds["maximum_mean_absolute_exposure_gap"]:
                comparison_control_failures.append("mean_absolute_exposure_gap")
            if exposure_p95 > thresholds["maximum_p95_absolute_exposure_gap"]:
                comparison_control_failures.append("p95_absolute_exposure_gap")
            if exposure_max > thresholds["maximum_absolute_exposure_gap"]:
                comparison_control_failures.append("maximum_absolute_exposure_gap")
        matched_turnover = identity in {
            "F_2x_minus_R_match_2x",
            "FC_gross_minus_FGR_gross",
            "FC_2x_minus_FGR_2x",
            "FMA_gross_minus_FMGR_gross",
            "FMA_2x_minus_FMGR_2x",
        }
        if matched_turnover:
            if turnover_gap > thresholds["maximum_mean_absolute_turnover_gap"]:
                comparison_control_failures.append("mean_absolute_turnover_gap")
            if turnover_p95 > thresholds["maximum_p95_absolute_turnover_gap"]:
                comparison_control_failures.append("p95_absolute_turnover_gap")
        try:
            hac = newey_west_mean_interval(active, lag=4, one_sided_confidence=0.95)
            bootstrap = circular_block_bootstrap_interval(
                active,
                identity=identity,
                window_sha256=str(value["window_sha256"]),
                method_seed=method_seed,
                block_length=block_length,
                draws=draws,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            if (
                max(float(bootstrap["lower_quantile_mcse"]), float(bootstrap["upper_quantile_mcse"]))
                > thresholds["maximum_bootstrap_quantile_mcse_annualized"]
            ):
                comparison_control_failures.append("bootstrap_quantile_mcse")
            concentration = top_positive_concentration(active)
            weekly_median = float(np.median(active))
            status, status_reasons, intervention_sufficient = _comparison_status(
                spec=spec,
                hac=hac,
                bootstrap=bootstrap,
                concentration=concentration,
                weekly_median=weekly_median,
                maximum_top10_share=thresholds["maximum_top10_positive_weeks_share"],
                interventions=interventions,
                minimum_gate_opportunities=int(thresholds["minimum_gate_eligible_new_entry_opportunities"]),
            )
            inference_error = None
        except DegenerateInferenceError as exc:
            hac = None
            bootstrap = None
            concentration = top_positive_concentration(active)
            status = "INCONCLUSIVE"
            status_reasons = ["degenerate_inference"]
            intervention_sufficient = (
                interventions["identity_disagreement_weeks"] >= int(spec["minimum_identity_disagreement_weeks"])
                and interventions["differing_symbol_assignments"] >= int(spec["minimum_differing_symbol_assignments"])
                and interventions["gate_eligible_new_entry_opportunities"]
                >= int(thresholds["minimum_gate_eligible_new_entry_opportunities"])
            )
            inference_error = str(exc)
        if comparison_control_failures:
            global_control_failures.extend(f"{identity}.{reason}" for reason in comparison_control_failures)
        comparisons[identity] = {
            "role": spec["role"],
            "decision_gating": bool(spec["decision_gating"]),
            "arm": spec["arm"],
            "comparator": spec["comparator"],
            "sesoi_annualized": float(spec["sesoi_annualized"]),
            "annualized_active_arithmetic": float(active.mean() * 52.0),
            "risk_and_power": _active_risk_and_power(active, float(spec["sesoi_annualized"])),
            "weekly_active_median": float(np.median(active)),
            "top10_positive_weeks_share": concentration,
            "hac": hac,
            "block_bootstrap": bootstrap,
            "mean_absolute_post_close_exposure_gap": exposure_gap,
            "p95_absolute_post_close_exposure_gap": exposure_p95,
            "maximum_absolute_post_close_exposure_gap": exposure_max,
            "mean_absolute_one_way_turnover_gap": turnover_gap,
            "p95_absolute_one_way_turnover_gap": turnover_p95,
            "turnover_control_applies": matched_turnover,
            "interventions": interventions,
            "intervention_sufficient": intervention_sufficient,
            "control_failures": comparison_control_failures,
            "status": status,
            "status_reasons": status_reasons,
            "inference_error": inference_error,
        }

    if global_control_failures:
        overall_status = "INVALID_CONTROL"
    else:
        gating = [comparisons[identity] for identity in EXPECTED_COMPARISONS[:5]]
        factor_status = gating[0]["status"]
        timing_statuses = [item["status"] for item in gating[1:]]
        if any(not comparisons[identity]["intervention_sufficient"] for identity in EXPECTED_COMPARISONS[:5]):
            overall_status = "INSUFFICIENT_INTERVENTION"
        elif factor_status == "FALSIFIED":
            overall_status = "FALSIFIED_FACTOR"
        elif factor_status != "PASS":
            overall_status = "INCONCLUSIVE_FACTOR"
        elif "FALSIFIED" in timing_statuses:
            overall_status = "FACTOR_PASSED_TIMING_FALSIFIED"
        elif any(status != "PASS" for status in timing_statuses):
            overall_status = "FACTOR_PASSED_TIMING_INCONCLUSIVE"
        else:
            overall_status = "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY"

    result = {
        "schema": STATISTICS_RESULT_SCHEMA,
        "protocol_sha256": value["protocol_sha256"],
        "trial_id": str(value["trial_id"]),
        "window_sha256": value["window_sha256"],
        "input_sha256": object_sha256(value),
        "complete_weeks": EXPECTED_COMPLETE_WEEKS,
        "comparison_order": list(EXPECTED_COMPARISONS),
        "comparisons": comparisons,
        "random_control_diagnostics": random_diagnostics,
        "control_failures": sorted(set(global_control_failures)),
        "overall_status": overall_status,
        "alpha_validated": False,
        "live_trading_allowed": False,
    }
    result["result_sha256"] = object_sha256(result)
    return result


def verify_statistics_result(value: Mapping[str, Any], expected_input: Mapping[str, Any]) -> dict[str, Any]:
    replayed = evaluate_statistics(expected_input)
    if canonical_json_bytes(value) != canonical_json_bytes(replayed):
        raise StatisticsError("statistics result differs from deterministic semantic replay")
    return replayed


__all__ = [
    "EXPECTED_COMPARISONS",
    "EXPECTED_COMPLETE_WEEKS",
    "EXPECTED_RANDOM_SEEDS",
    "STATISTICS_INPUT_SCHEMA",
    "STATISTICS_RESULT_SCHEMA",
    "StatisticsError",
    "DegenerateInferenceError",
    "WEEKLY_PATH_SCHEMA",
    "canonical_json_bytes",
    "circular_block_bootstrap_interval",
    "evaluate_statistics",
    "newey_west_mean_interval",
    "object_sha256",
    "top_positive_concentration",
    "verify_statistics_result",
]
