"""Run the append-only Stage 2 retrospective falsification study.

Stage 2 reuses the already-seen Stage 1 historical artifacts.  It is therefore
allowed to invalidate mechanisms and generate hypotheses, but it cannot start
the V2.1 confirmation chain or make an independent out-of-sample claim.

Usage:

    uv run --no-sync python scripts/xs_chan_exploration_stage2.py check
    uv run --no-sync python scripts/xs_chan_exploration_stage2.py run
    uv run --no-sync python scripts/xs_chan_exploration_stage2.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import xs_chan_exploration_stage1 as stage1
from numpy.random import PCG64DXSM, Generator

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SPEC_PATH = SCRIPTS_DIR / "xs_chan_exploration_stage2.json"
SOURCE_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPTS_DIR / "_output" / "xs_chan_exploration_stage2"
V21_PROTOCOL_PATH = SCRIPTS_DIR / "xs_chan_protocol_v2_1.json"
V21_DOCUMENT_PATH = SCRIPTS_DIR / "XS_CHAN_PILOT_PROTOCOL_V2_1.md"

EXPECTED_EXPERIMENT_IDS = (
    "S2E00_INPUT_AND_INTEGRITY",
    "S2E01_H1_COMMON_SUPPORT",
    "S2E02_H1_CONTROLS",
    "S2E03_H2_RESIDUAL_STABILITY",
    "S2E04_H2_RANDOM_IDENTITY",
    "S2E05_H3_PROFILE_REPLICATION",
    "S2E06_H3_CONTROLS_AND_SECONDARY",
    "S2E07_FIXED_SLOT_PATH_STRESS",
    "S2E08_PLACEBO_AND_MARKET_STATE",
    "S2E09_DECISION_MATRIX",
)
DECISION_TIME_FEATURES = (
    "factor_score",
    "factor_rank",
    "mom_120_20",
    "lowvol_60",
    "log_free_float_mcap",
    "adv20",
    "turnover_rate",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "distance_to_sma20",
)
STATUS_VALUES = {
    "PASS_EXPLORATORY_HISTORICAL",
    "FALSIFIED_RETROSPECTIVE",
    "INCONCLUSIVE",
    "INSUFFICIENT_DATA",
    "INVALID_POST_SELECTION",
    "INVALID_DATA",
}


class Stage2Error(RuntimeError):
    """Raised when Stage 2 cannot produce an integrity-preserving result."""


@dataclass(frozen=True)
class InputBundle:
    """Validated Stage 1 event-level inputs."""

    ranked: pd.DataFrame
    memberships: pd.DataFrame
    proposals: pd.DataFrame
    attribution_weekly: pd.DataFrame
    failure_sample: pd.DataFrame
    decision_dates: pd.DatetimeIndex
    paths: dict[str, Path]


@dataclass
class ExperimentRecord:
    """One experiment status persisted after every transition."""

    experiment_id: str
    deliverable: str
    status: str = "PLANNED"
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_seconds: float | None = None
    output_rows: dict[str, int] | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_tail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "deliverable": self.deliverable,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "duration_seconds": self.duration_seconds,
            "output_rows": self.output_rows or {},
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_tail": self.traceback_tail,
        }


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    """Read a strict JSON object."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(Stage2Error(f"invalid JSON constant: {token}")),
        )
    except OSError as exc:
        raise Stage2Error(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Stage2Error(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write strict, stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_json(payload: Any) -> bytes:
    """Return canonical JSON bytes."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_and_validate_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    """Validate frozen scope, no-tuning rules, V2.1 and every Stage 1 byte input."""

    spec = read_json(path)
    if (
        spec.get("mode") != "EXPLORATORY_ONLY"
        or spec.get("study_type") != "RETROSPECTIVE_FALSIFICATION_ONLY"
        or spec.get("confirmation_chain") != "NOT_STARTED"
    ):
        raise Stage2Error("Stage 2 must remain retrospective exploration with no confirmation chain")
    provenance = spec.get("design_provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("stage1_full_sample_already_seen"):
        raise Stage2Error("Stage 2 must disclose that the full Stage 1 sample was already seen")
    if provenance.get("stage2_is_preregistered_confirmation") is not False:
        raise Stage2Error("Stage 2 must not claim untouched preregistration")

    no_tuning = spec.get("no_tuning")
    false_keys = (
        "parameter_search",
        "threshold_optimization",
        "best_variant_selection",
        "failed_experiment_suppression",
        "post_result_endpoint_substitution",
        "matching_relaxation_after_results",
        "seed_extension_after_results",
    )
    if not isinstance(no_tuning, Mapping) or any(no_tuning.get(key) is not False for key in false_keys):
        raise Stage2Error("all frozen anti-tuning flags must be literal false")
    planned = spec.get("planned_experiments")
    if not isinstance(planned, list):
        raise Stage2Error("planned_experiments must be a list")
    ids = tuple(row.get("id") for row in planned if isinstance(row, Mapping))
    if ids != EXPECTED_EXPERIMENT_IDS:
        raise Stage2Error("Stage 2 planned experiment registry differs from the frozen order")

    baseline = spec.get("frozen_v2_1_baseline")
    if not isinstance(baseline, Mapping):
        raise Stage2Error("missing V2.1 baseline")
    if sha256_file(V21_PROTOCOL_PATH) != baseline.get("protocol_physical_sha256"):
        raise Stage2Error("V2.1 physical protocol changed")
    protocol = read_json(V21_PROTOCOL_PATH)
    if hashlib.sha256(canonical_json(protocol)).hexdigest() != baseline.get("protocol_canonical_sha256"):
        raise Stage2Error("V2.1 canonical protocol changed")
    if protocol.get("protocol_id") != baseline.get("protocol_id"):
        raise Stage2Error("V2.1 protocol ID changed")
    if protocol.get("protocol_status") != baseline.get("required_protocol_status"):
        raise Stage2Error("V2.1 protocol status changed")
    if sha256_file(V21_DOCUMENT_PATH) != baseline.get("protocol_document_sha256"):
        raise Stage2Error("V2.1 explanatory document changed")

    frozen_inputs = spec.get("frozen_stage1_inputs")
    if not isinstance(frozen_inputs, list) or not frozen_inputs:
        raise Stage2Error("frozen_stage1_inputs must be non-empty")
    names: set[str] = set()
    for row in frozen_inputs:
        if not isinstance(row, Mapping):
            raise Stage2Error("each frozen Stage 1 input must be an object")
        name = str(row.get("name", ""))
        relative = Path(str(row.get("path", "")))
        expected = str(row.get("sha256", ""))
        if not name or name in names or relative.is_absolute() or len(expected) != 64:
            raise Stage2Error(f"invalid frozen input declaration: {row}")
        names.add(name)
        absolute = REPO_ROOT / relative
        if not absolute.is_file():
            raise Stage2Error(f"missing frozen Stage 1 input: {absolute}")
        actual = sha256_file(absolute)
        if actual != expected:
            raise Stage2Error(f"frozen Stage 1 input changed: {name} expected={expected} actual={actual}")
    if tuple(spec["h3_failure_profile"]["nested_candidate_features"]) != DECISION_TIME_FEATURES:
        raise Stage2Error("nested H3 candidate set changed")
    if (
        spec["h1_state7_increment"]["matched_control"].get("treated_support")
        != "same_exact_strata_and_same_matched_count_as_sampled_controls"
    ):
        raise Stage2Error("matched H1 treated support is not frozen to the sampled control support")
    integrity = spec.get("output_integrity")
    required_integrity_flags = (
        "atomic_publish",
        "refuse_overwrite",
        "preserve_failed_run_directory",
        "inventory_output_sha256_size_rows_schema",
        "detached_manifest_sha256",
        "verify_current_identity_against_spec_source_and_frozen_inputs",
        "classify_revision_history_identities_as_superseded",
    )
    if not isinstance(integrity, Mapping) or any(integrity.get(key) is not True for key in required_integrity_flags):
        raise Stage2Error("Stage 2 output-integrity controls must remain enabled")
    return spec


def study_identity(spec: Mapping[str, Any], source_path: Path = SOURCE_PATH) -> str:
    """Bind the output directory to the complete spec and source bytes."""

    payload = canonical_json(spec) + b"\0" + source_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def output_path_for(spec: Mapping[str, Any], output_root: Path = OUTPUT_ROOT) -> Path:
    """Resolve the immutable content-addressed publication path."""

    return output_root / f"{spec['output_integrity']['directory_prefix']}{study_identity(spec)}"


def rng_for(spec: Mapping[str, Any], domain: str, index: int = 0) -> Generator:
    """Create a domain-separated PCG64DXSM generator."""

    root = int(spec["statistics"]["random_seed_root"])
    digest = hashlib.sha256(f"{spec['study_id']}|{domain}|{index}".encode()).digest()
    derived = int.from_bytes(digest[:8], "big")
    return Generator(PCG64DXSM(np.random.SeedSequence([root, derived, int(index)])))


def _frozen_input_paths(spec: Mapping[str, Any]) -> dict[str, Path]:
    return {str(row["name"]): REPO_ROOT / str(row["path"]) for row in spec["frozen_stage1_inputs"]}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := required - set(frame):
        raise Stage2Error(f"{label} missing columns: {sorted(missing)}")


def load_input_bundle(spec: Mapping[str, Any]) -> InputBundle:
    """Load compact Stage 1 artifacts and fail closed on schema or identity drift."""

    paths = _frozen_input_paths(spec)
    ranked_columns = [
        "decision_dt",
        "symbol",
        "factor_rank",
        "factor_score",
        "regime",
        "ma_allowed",
        "entry_tradable",
        "fwd_5d_open_return",
        "industry_code",
        "mcap_bucket",
        "log_free_float_mcap",
        "lowvol_60",
        "mom_120_20",
    ]
    ranked = pd.read_parquet(paths["ranked_surface"], columns=ranked_columns)
    memberships = pd.read_parquet(paths["buffered_memberships"])
    proposals = pd.read_parquet(paths["gate_proposals"])
    attribution = pd.read_parquet(paths["attribution_weekly"])
    failure = pd.read_parquet(paths["failure_sample"])
    _require_columns(ranked, set(ranked_columns), "ranked_surface")
    _require_columns(
        memberships,
        {
            "decision_dt",
            "arm",
            "symbol",
            "membership_role",
            "factor_rank",
            "regime",
            "ma_allowed",
            "entry_tradable",
            "fwd_5d_open_return",
        },
        "buffered_memberships",
    )
    _require_columns(
        proposals,
        {
            "decision_dt",
            "arm",
            "symbol",
            "regime",
            "ma_allowed",
            "industry_code",
            "mcap_bucket",
            "factor_rank",
            "entry_tradable",
            "fwd_5d_open_return",
        },
        "gate_proposals",
    )
    _require_columns(
        attribution,
        {
            "decision_dt",
            "selected_return",
            "market",
            "size",
            "industry",
            "low_volatility",
            "momentum",
            "unexplained_residual",
            "reconstruction_error",
        },
        "attribution_weekly",
    )
    _require_columns(
        failure,
        {
            "decision_dt",
            "symbol",
            "failure_primary",
            "failure_bottom_quintile",
            *DECISION_TIME_FEATURES,
        },
        "failure_sample",
    )
    for frame in (ranked, memberships, proposals, attribution, failure):
        frame["decision_dt"] = pd.to_datetime(frame["decision_dt"])
    if ranked.duplicated(["decision_dt", "symbol"]).any():
        raise Stage2Error("ranked_surface has duplicate weekly identities")
    if memberships.duplicated(["decision_dt", "arm", "symbol"]).any():
        raise Stage2Error("buffered_memberships has duplicate arm identities")
    if proposals.duplicated(["decision_dt", "arm", "symbol"]).any():
        raise Stage2Error("gate_proposals has duplicate arm identities")
    if attribution.duplicated(["decision_dt"]).any():
        raise Stage2Error("attribution_weekly has duplicate dates")
    if failure.duplicated(["decision_dt", "symbol"]).any():
        raise Stage2Error("failure_sample has duplicate identities")
    dates = pd.DatetimeIndex(sorted(ranked["decision_dt"].unique()))
    split = spec["time_split"]
    expected_dates = int(split["expected_development_weeks"]) + int(split["expected_historical_validation_weeks"])
    expected_start = pd.Timestamp(split["development_start"])
    expected_end = pd.Timestamp(split["historical_validation_end"])
    if len(dates) != expected_dates or dates[0] != expected_start or dates[-1] != expected_end:
        raise Stage2Error(
            "Stage 1 analyzed date identity changed: "
            f"expected={expected_dates}/{expected_start.date()}/{expected_end.date()} "
            f"actual={len(dates)}/{dates[0].date()}/{dates[-1].date()}"
        )
    if not pd.DatetimeIndex(sorted(attribution["decision_dt"].unique())).equals(dates):
        raise Stage2Error("attribution dates differ from ranked dates")
    return InputBundle(
        ranked=ranked,
        memberships=memberships,
        proposals=proposals,
        attribution_weekly=attribution,
        failure_sample=failure,
        decision_dates=dates,
        paths=paths,
    )


def segment_mask(dates: pd.Series | pd.DatetimeIndex, spec: Mapping[str, Any], segment: str) -> np.ndarray:
    """Return the exact frozen time-segment mask."""

    values = pd.DatetimeIndex(pd.to_datetime(dates))
    split = spec["time_split"]
    bounds = {
        "DEVELOPMENT": (split["development_start"], split["development_end"]),
        "HISTORICAL_VALIDATION": (split["historical_validation_start"], split["historical_validation_end"]),
        "V1": (split["validation_half_1_start"], split["validation_half_1_end"]),
        "V2": (split["validation_half_2_start"], split["validation_half_2_end"]),
    }
    if segment == "FULL":
        return np.ones(len(values), dtype=bool)
    if segment.startswith("YEAR_"):
        return values.year == int(segment.removeprefix("YEAR_"))
    if segment not in bounds:
        raise Stage2Error(f"unknown frozen segment: {segment}")
    start, end = map(pd.Timestamp, bounds[segment])
    return np.asarray((values >= start) & (values <= end), dtype=bool)


def _finite(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def circular_block_ci(
    values: Sequence[float] | pd.Series | np.ndarray,
    spec: Mapping[str, Any],
    identity: str,
) -> tuple[float | None, float | None]:
    """Return the frozen circular-block bootstrap confidence interval for a mean."""

    vector = _finite(values)
    if not len(vector):
        return None, None
    block = int(spec["statistics"]["circular_block_weeks"])
    draws = int(spec["statistics"]["bootstrap_draws"])
    confidence = float(spec["statistics"]["bootstrap_confidence"])
    blocks = math.ceil(len(vector) / block)
    rng = rng_for(spec, f"bootstrap|{identity}")
    starts = rng.integers(0, len(vector), size=(draws, blocks))
    offsets = np.arange(block, dtype=int)
    indices = (starts[..., None] + offsets) % len(vector)
    samples = vector[indices.reshape(draws, -1)[:, : len(vector)]]
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha, method="linear")),
        float(np.quantile(means, 1.0 - alpha, method="linear")),
    )


def weekly_stats(
    values: Sequence[float] | pd.Series | np.ndarray,
    spec: Mapping[str, Any],
    identity: str,
) -> dict[str, Any]:
    """Summarize one weekly vector with HAC and frozen block-bootstrap uncertainty."""

    vector = _finite(values)
    if not len(vector):
        return {
            "n_weeks": 0,
            "mean": None,
            "median": None,
            "std": None,
            "win_rate": None,
            "hac_t": None,
            "hac_se": None,
            "hac_ci_low": None,
            "hac_ci_high": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "best_10_week_contribution_share": None,
        }
    count = len(vector)
    centered = vector - float(vector.mean())
    lag_max = min(int(spec["statistics"]["hac_lag"]), count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    configured_lag = int(spec["statistics"]["hac_lag"])
    for lag in range(1, lag_max + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * (1.0 - lag / (configured_lag + 1.0)) * covariance
    hac_se = math.sqrt(max(long_run_variance, 0.0) / count)
    mean = float(vector.mean())
    hac_t = mean / hac_se if hac_se > 0 else 0.0
    boot_low, boot_high = circular_block_ci(vector, spec, identity)
    total = float(vector.sum())
    best_count = min(10, count)
    contribution = None if abs(total) < 1e-12 else float(np.sort(vector)[-best_count:].sum() / total)
    return {
        "n_weeks": int(count),
        "mean": mean,
        "median": float(np.median(vector)),
        "std": float(np.std(vector, ddof=1)) if count > 1 else None,
        "win_rate": float(np.mean(vector > 0)),
        "hac_t": float(hac_t),
        "hac_se": float(hac_se),
        "hac_ci_low": float(mean - 1.96 * hac_se),
        "hac_ci_high": float(mean + 1.96 * hac_se),
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
        "best_10_week_contribution_share": contribution,
    }


def summarize_weekly_frame(
    weekly: pd.DataFrame,
    value_column: str,
    spec: Mapping[str, Any],
    identity: str,
    *,
    extra: Mapping[str, Any] | None = None,
    segment_sums: Mapping[str, str] | None = None,
    segment_means: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize full, frozen splits and calendar years."""

    segments = ["FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"]
    segments.extend(f"YEAR_{year}" for year in sorted(weekly["decision_dt"].dt.year.unique()))
    rows: list[dict[str, Any]] = []
    for segment in segments:
        mask = segment_mask(weekly["decision_dt"], spec, segment)
        row = {
            "segment": segment,
            **weekly_stats(weekly.loc[mask, value_column], spec, f"{identity}|{segment}"),
        }
        if extra:
            row.update(extra)
        if segment_sums:
            for output_column, source_column in segment_sums.items():
                row[output_column] = int(pd.to_numeric(weekly.loc[mask, source_column], errors="coerce").sum())
        if segment_means:
            for output_column, source_column in segment_means.items():
                values = _finite(pd.to_numeric(weekly.loc[mask, source_column], errors="coerce"))
                row[output_column] = float(values.mean()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def normal_two_sided_p(hac_t: float | None) -> float | None:
    """Return a descriptive normal-reference two-sided p-value."""

    if hac_t is None or not math.isfinite(float(hac_t)):
        return None
    return float(math.erfc(abs(float(hac_t)) / math.sqrt(2.0)))


def holm_adjust(
    p_values: Mapping[str, float | None],
    *,
    family_size: int | None = None,
) -> dict[str, float | None]:
    """Apply Holm adjustment without dropping named missing endpoints."""

    finite = [(key, float(value)) for key, value in p_values.items() if value is not None and math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    count = len(finite) if family_size is None else int(family_size)
    if count < len(finite):
        raise Stage2Error("Holm family size cannot be smaller than the finite endpoint count")
    adjusted: dict[str, float | None] = dict.fromkeys(p_values)
    running = 0.0
    for rank, (key, value) in enumerate(finite):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def execute_experiment(
    spec: Mapping[str, Any],
    records: dict[str, ExperimentRecord],
    work_dir: Path,
    experiment_id: str,
    function: Callable[[], Any],
    row_counter: Callable[[Any], Mapping[str, int]],
) -> Any:
    """Execute one registered experiment and persist both success and failure."""

    record = records[experiment_id]
    record.status = "RUNNING"
    record.started_at_utc = utc_now()
    started = time.monotonic()
    write_registry(spec, records, work_dir)
    try:
        result = function()
    except Exception as exc:
        record.status = "FAILED"
        record.completed_at_utc = utc_now()
        record.duration_seconds = round(time.monotonic() - started, 6)
        record.error_type = type(exc).__name__
        record.error_message = str(exc)
        record.traceback_tail = "\n".join(traceback.format_exc().splitlines()[-16:])
        write_registry(spec, records, work_dir)
        raise
    record.status = "COMPLETED"
    record.completed_at_utc = utc_now()
    record.duration_seconds = round(time.monotonic() - started, 6)
    record.output_rows = {str(key): int(value) for key, value in row_counter(result).items()}
    write_registry(spec, records, work_dir)
    return result


def write_registry(
    spec: Mapping[str, Any],
    records: Mapping[str, ExperimentRecord],
    work_dir: Path,
) -> None:
    """Write the current experiment registry."""

    write_json(
        work_dir / "experiment_registry.json",
        {
            "study_id": spec["study_id"],
            "mode": spec["mode"],
            "updated_at_utc": utc_now(),
            "experiments": [records[key].as_dict() for key in EXPECTED_EXPERIMENT_IDS],
        },
    )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping or fail with a Stage 2 integrity error."""

    if not isinstance(value, Mapping):
        raise Stage2Error(f"{label} must be an object")
    return value


def _repo_relative_file(raw_path: Any, label: str) -> Path:
    """Resolve an existing repository-relative audit input without traversal."""

    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise Stage2Error(f"{label} must be repository-relative")
    resolved = REPO_ROOT / relative
    if not resolved.is_file():
        raise Stage2Error(f"{label} is missing: {resolved}")
    return resolved


def _validate_stage1_algorithm_revision_claims(
    spec: Mapping[str, Any],
    audit: Mapping[str, Any],
    stage1_spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate every semantic claim that permits the revised Stage 1 artifacts."""

    required = _require_mapping(spec.get("stage1_algorithm_revision"), "Stage 1 algorithm-revision contract")
    if audit.get("audit_id") != required.get("required_audit_id"):
        raise Stage2Error("Stage 1 algorithm-revision audit ID changed")
    if audit.get("audit_type") != required.get("required_audit_type"):
        raise Stage2Error("Stage 1 algorithm-revision audit type changed")
    if audit.get("audited_study_id") != stage1_spec.get("study_id"):
        raise Stage2Error("Stage 1 algorithm-revision audit study ID changed")
    if audit.get("confirmation_chain") != "NOT_STARTED":
        raise Stage2Error("Stage 1 algorithm-revision audit cannot start a confirmation chain")

    conclusion = _require_mapping(audit.get("overall_conclusion"), "Stage 1 algorithm-revision conclusion")
    required_conclusion = {
        "alpha_confirmation": "NOT_SUPPORTED",
        "direction_of_stage1_negative_conclusion": "REQUIRES_STAGE2_RECALCULATION",
        "stage1_result_status": "SUPERSEDED_IN_PART",
    }
    observed_conclusion = {key: conclusion.get(key) for key in required_conclusion}
    if observed_conclusion != required_conclusion:
        raise Stage2Error(f"Stage 1 algorithm-revision conclusion changed: {observed_conclusion}")

    prior = _require_mapping(audit.get("prior_semantic_audit"), "prior Stage 1 semantic audit")
    if prior.get("carry_forward") is not True:
        raise Stage2Error("prior Stage 1 semantic findings are not carried forward")
    if prior.get("path") != required.get("required_prior_audit_path"):
        raise Stage2Error("prior Stage 1 semantic audit path changed")
    if prior.get("sha256") != required.get("required_prior_audit_sha256"):
        raise Stage2Error("prior Stage 1 semantic audit hash changed")
    if prior.get("findings") != required.get("required_prior_findings"):
        raise Stage2Error("prior Stage 1 semantic finding set changed")

    revision = _require_mapping(audit.get("algorithm_revision"), "Stage 1 algorithm revision")
    exact_revision_fields = {
        "source_commit": "required_source_commit",
        "old_state_cache_identity": "required_old_state_cache_identity",
        "old_state_sha256": "required_old_state_sha256",
        "old_state_manifest_sha256": "required_old_state_manifest_sha256",
        "new_state_cache_identity": "required_new_state_cache_identity",
        "new_state_sha256": "required_new_state_sha256",
        "new_state_manifest_sha256": "required_new_state_manifest_sha256",
        "native_extension_sha256": "required_native_extension_sha256",
        "old_source_files": "required_old_source_files",
        "new_source_files": "required_new_source_files",
        "overlap_rows": "required_overlap_rows",
        "regime_mismatches": "required_regime_mismatches",
        "new_state_rows": "required_new_state_rows",
        "prefix_causality_audit": "required_prefix_causality_audit",
    }
    for audit_key, required_key in exact_revision_fields.items():
        if revision.get(audit_key) != required.get(required_key):
            raise Stage2Error(
                f"Stage 1 algorithm-revision field changed: {audit_key} "
                f"expected={required.get(required_key)} actual={revision.get(audit_key)}"
            )
    overlap_rows = int(revision["overlap_rows"])
    regime_mismatches = int(revision["regime_mismatches"])
    expected_rate = regime_mismatches / overlap_rows
    if not math.isclose(float(revision.get("regime_mismatch_rate", math.nan)), expected_rate, rel_tol=0, abs_tol=1e-15):
        raise Stage2Error("Stage 1 algorithm-revision mismatch rate is inconsistent")

    comparison = _require_mapping(revision.get("comparison"), "Stage 1 state comparison")
    comparison_requirements = {
        "keys": "required_comparison_keys",
        "join": "required_comparison_join",
        "mismatch_predicate": "required_mismatch_predicate",
        "method": "required_comparison_method",
    }
    for audit_key, required_key in comparison_requirements.items():
        if comparison.get(audit_key) != required.get(required_key):
            raise Stage2Error(f"Stage 1 state-comparison contract changed: {audit_key}")
    if comparison.get("new_state_path_source") != "stage1_data_manifest.inputs.state_path":
        raise Stage2Error("Stage 1 new-state path source changed")
    return required, revision, prior, comparison


def _state_revision_counts(
    old_state_path: Path,
    new_state_path: Path,
    keys: Sequence[str],
) -> tuple[int, int]:
    """Recompute the exact overlapping rows and regime mismatches."""

    old = pl.scan_parquet(str(old_state_path)).select(
        *keys,
        pl.col("regime").alias("old_regime"),
    )
    new = pl.scan_parquet(str(new_state_path)).select(
        *keys,
        pl.col("regime").alias("new_regime"),
    )
    result = (
        old.join(new, on=list(keys), how="inner")
        .select(
            pl.len().alias("overlap_rows"),
            (pl.col("old_regime") != pl.col("new_regime")).sum().alias("regime_mismatches"),
        )
        .collect(engine="streaming")
    )
    return int(result["overlap_rows"][0]), int(result["regime_mismatches"][0])


def _verify_stage1_algorithm_revision_files(
    bundle: InputBundle,
    required: Mapping[str, Any],
    revision: Mapping[str, Any],
    prior: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> tuple[int, int]:
    """Verify both state-cache generations and independently recompute their difference."""

    prior_path = _repo_relative_file(prior["path"], "prior Stage 1 semantic audit")
    if sha256_file(prior_path) != prior["sha256"]:
        raise Stage2Error("prior Stage 1 semantic audit file changed")

    old_state_path = _repo_relative_file(comparison.get("old_state_path"), "old Stage 1 state projection")
    old_manifest_path = _repo_relative_file(
        comparison.get("old_state_manifest_path"),
        "old Stage 1 state manifest",
    )
    if sha256_file(old_state_path) != revision["old_state_sha256"]:
        raise Stage2Error("old Stage 1 state projection changed")
    if sha256_file(old_manifest_path) != revision["old_state_manifest_sha256"]:
        raise Stage2Error("old Stage 1 state manifest changed")
    old_manifest = read_json(old_manifest_path)
    old_projection = _require_mapping(old_manifest.get("projection"), "old state projection manifest")
    if old_manifest.get("cache_id") != revision["old_state_cache_identity"]:
        raise Stage2Error("old Stage 1 state-cache identity changed")
    if old_projection.get("sha256") != revision["old_state_sha256"]:
        raise Stage2Error("old Stage 1 state manifest does not bind its projection")
    if len(old_manifest.get("sources", [])) != int(revision["old_source_files"]):
        raise Stage2Error("old Stage 1 state-cache source count changed")

    stage1_manifest = read_json(bundle.paths["stage1_data_manifest"])
    stage1_inputs = _require_mapping(stage1_manifest.get("inputs"), "Stage 1 manifest inputs")
    new_state_path = Path(str(stage1_inputs.get("state_path", "")))
    new_manifest_path = new_state_path.with_name("source_manifest.json")
    if not new_state_path.is_file() or not new_manifest_path.is_file():
        raise Stage2Error("new Stage 1 state-cache files are missing")
    if stage1_inputs.get("state_sha256") != revision["new_state_sha256"]:
        raise Stage2Error("Stage 1 manifest does not bind the audited new state projection")
    if stage1_inputs.get("state_manifest_sha256") != revision["new_state_manifest_sha256"]:
        raise Stage2Error("Stage 1 manifest does not bind the audited new state manifest")
    if sha256_file(new_state_path) != revision["new_state_sha256"]:
        raise Stage2Error("new Stage 1 state projection changed")
    if sha256_file(new_manifest_path) != revision["new_state_manifest_sha256"]:
        raise Stage2Error("new Stage 1 state manifest changed")

    new_manifest = read_json(new_manifest_path)
    new_projection = _require_mapping(new_manifest.get("projection"), "new state projection manifest")
    new_engine = _require_mapping(new_manifest.get("engine"), "new state-cache engine")
    new_native = _require_mapping(new_engine.get("native_extension"), "new state-cache native engine")
    if new_manifest.get("cache_id") != revision["new_state_cache_identity"]:
        raise Stage2Error("new Stage 1 state-cache identity changed")
    if new_projection.get("sha256") != revision["new_state_sha256"]:
        raise Stage2Error("new Stage 1 state manifest does not bind its projection")
    if int(new_projection.get("rows", -1)) != int(revision["new_state_rows"]):
        raise Stage2Error("new Stage 1 state row count changed")
    if len(new_manifest.get("sources", [])) != int(revision["new_source_files"]):
        raise Stage2Error("new Stage 1 state-cache source count changed")
    if new_native.get("sha256") != revision["native_extension_sha256"]:
        raise Stage2Error("new Stage 1 native-extension identity changed")
    state_audit_entry = _require_mapping(new_manifest.get("state_audit"), "new state-cache causality audit")
    state_audit_path = new_manifest_path.parent / str(state_audit_entry.get("path", ""))
    if not state_audit_path.is_file() or sha256_file(state_audit_path) != state_audit_entry.get("sha256"):
        raise Stage2Error("new Stage 1 state-cache causality audit changed")
    state_audit = read_json(state_audit_path)
    if state_audit.get("passed") is not True or int(state_audit.get("mismatch_count", -1)) != 0:
        raise Stage2Error("new Stage 1 state-cache causality audit did not pass")

    overlap_rows, regime_mismatches = _state_revision_counts(
        old_state_path,
        new_state_path,
        tuple(str(key) for key in comparison["keys"]),
    )
    if overlap_rows != int(required["required_overlap_rows"]):
        raise Stage2Error(f"Stage 1 state overlap changed: {overlap_rows}")
    if regime_mismatches != int(required["required_regime_mismatches"]):
        raise Stage2Error(f"Stage 1 state mismatch count changed: {regime_mismatches}")
    return overlap_rows, regime_mismatches


def run_input_audit(spec: Mapping[str, Any], bundle: InputBundle) -> dict[str, Any]:
    """Prove frozen time coverage and quantify genuinely unused complete weeks."""

    split = spec["time_split"]
    counts = {
        "development": int(segment_mask(bundle.decision_dates, spec, "DEVELOPMENT").sum()),
        "historical_validation": int(segment_mask(bundle.decision_dates, spec, "HISTORICAL_VALIDATION").sum()),
        "validation_half_1": int(segment_mask(bundle.decision_dates, spec, "V1").sum()),
        "validation_half_2": int(segment_mask(bundle.decision_dates, spec, "V2").sum()),
    }
    expected = {
        "development": int(split["expected_development_weeks"]),
        "historical_validation": int(split["expected_historical_validation_weeks"]),
        "validation_half_1": int(split["expected_validation_half_1_weeks"]),
        "validation_half_2": int(split["expected_validation_half_2_weeks"]),
    }
    if counts != expected:
        raise Stage2Error(f"frozen time split changed: observed={counts} expected={expected}")

    stage1_spec = read_json(bundle.paths["stage1_spec"])
    calendar = stage1.load_market_calendar(stage1_spec["sample"]["end_date"])
    table = pd.DataFrame({"dt": calendar, "session_no": np.arange(len(calendar), dtype=int)})
    table["week"] = table["dt"].dt.to_period("W-FRI")
    all_decisions = table.groupby("week", sort=True, observed=True)["session_no"].max().to_numpy(dtype=int)
    last_used = calendar.get_loc(bundle.decision_dates[-1])
    unused = all_decisions[all_decisions > last_used]
    primary_offset = 1 + int(stage1_spec["sample"]["primary_holding_sessions"])
    diagnostic_offset = 1 + int(stage1_spec["sample"]["diagnostic_holding_sessions"])
    primary_complete = unused[unused + primary_offset < len(calendar)]
    diagnostic_complete = unused[unused + diagnostic_offset < len(calendar)]
    audit = read_json(bundle.paths["stage1_audit"])
    required_revision, revision, prior_audit, comparison = _validate_stage1_algorithm_revision_claims(
        spec,
        audit,
        stage1_spec,
    )
    audited_artifacts = audit.get("audited_artifacts")
    if not isinstance(audited_artifacts, Mapping):
        raise Stage2Error("Stage 1 semantic audit artifact inventory is missing")
    audit_bindings = {
        "stage1_spec_sha256": "stage1_spec",
        "stage1_source_sha256": "stage1_source",
        "data_manifest_sha256": "stage1_data_manifest",
        "next_hypotheses_sha256": "stage1_next_hypotheses",
        "ranked_surface_sha256": "ranked_surface",
        "buffered_memberships_sha256": "buffered_memberships",
        "gate_proposals_sha256": "gate_proposals",
        "attribution_weekly_sha256": "attribution_weekly",
        "failure_sample_sha256": "failure_sample",
    }
    for audit_key, path_key in audit_bindings.items():
        expected_digest = str(audited_artifacts.get(audit_key, ""))
        actual_digest = sha256_file(bundle.paths[path_key])
        if expected_digest != actual_digest:
            raise Stage2Error(
                f"Stage 1 semantic audit does not bind current {path_key}: "
                f"expected={expected_digest or 'MISSING'} actual={actual_digest}"
            )
    overlap_rows, regime_mismatches = _verify_stage1_algorithm_revision_files(
        bundle,
        required_revision,
        revision,
        prior_audit,
        comparison,
    )
    return {
        "study_id": spec["study_id"],
        "integrity_status": "PASSED",
        "stage1_semantic_status": "SUPERSEDED_IN_PART",
        "stage1_algorithm_revision": {
            "source_commit": revision["source_commit"],
            "old_state_cache_identity": revision["old_state_cache_identity"],
            "new_state_cache_identity": revision["new_state_cache_identity"],
            "comparison_keys": list(comparison["keys"]),
            "comparison_join": comparison["join"],
            "overlap_rows": overlap_rows,
            "regime_mismatches": regime_mismatches,
            "regime_mismatch_rate": regime_mismatches / overlap_rows,
            "prefix_causality_audit": revision["prefix_causality_audit"],
            "status": "VERIFIED_ALGORITHM_REVISION",
        },
        "decision_dates": {
            "rows": int(len(bundle.decision_dates)),
            "start": bundle.decision_dates[0].date().isoformat(),
            "end": bundle.decision_dates[-1].date().isoformat(),
            "split_counts": counts,
        },
        "fresh_sample_audit": {
            "calendar_end": calendar[-1].date().isoformat(),
            "unused_primary_complete_weeks": int(len(primary_complete)),
            "unused_primary_decision_dates": [calendar[index].date().isoformat() for index in primary_complete],
            "unused_20_session_diagnostic_complete_weeks": int(len(diagnostic_complete)),
            "minimum_independent_weeks": 26,
            "independent_sample_sufficient": bool(len(primary_complete) >= 26),
        },
        "input_rows": {
            "ranked_surface": int(len(bundle.ranked)),
            "buffered_memberships": int(len(bundle.memberships)),
            "gate_proposals": int(len(bundle.proposals)),
            "attribution_weekly": int(len(bundle.attribution_weekly)),
            "failure_sample": int(len(bundle.failure_sample)),
        },
        "v2_1_unchanged": True,
        "confirmation_chain": "NOT_STARTED",
    }


def paired_gate_weekly(
    frame: pd.DataFrame,
    left_mask: pd.Series | np.ndarray,
    right_mask: pd.Series | np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Return equal-week paired cohort means on common decision weeks."""

    scoped = frame[
        frame["entry_tradable"].fillna(False) & np.isfinite(pd.to_numeric(frame["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    left_index = pd.Series(left_mask, index=frame.index).reindex(scoped.index).fillna(False).astype(bool)
    right_index = pd.Series(right_mask, index=frame.index).reindex(scoped.index).fillna(False).astype(bool)
    left = (
        scoped[left_index].groupby("decision_dt", sort=True, observed=True)["fwd_5d_open_return"].agg(["mean", "size"])
    )
    left.columns = [f"{left_name}_return", f"{left_name}_events"]
    right = (
        scoped[right_index].groupby("decision_dt", sort=True, observed=True)["fwd_5d_open_return"].agg(["mean", "size"])
    )
    right.columns = [f"{right_name}_return", f"{right_name}_events"]
    weekly = left.join(right, how="inner").reset_index()
    weekly["weekly_difference"] = weekly[f"{left_name}_return"] - weekly[f"{right_name}_return"]
    return weekly.sort_values("decision_dt", kind="mergesort").reset_index(drop=True)


def run_h1_common_support(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep proposal and buffered-membership H1 estimands strictly separate."""

    proposal_scope = bundle.proposals[bundle.proposals["arm"].eq("FC")].copy()
    proposal_weekly = paired_gate_weekly(
        proposal_scope,
        proposal_scope["regime"].eq(7),
        proposal_scope["ma_allowed"].fillna(False),
        left_name="state7",
        right_name="sma20",
    )
    proposal_weekly["estimand"] = "proposal_common_support"
    membership_scope = bundle.memberships[bundle.memberships["arm"].eq("F")].copy()
    membership_weekly = paired_gate_weekly(
        membership_scope,
        membership_scope["regime"].eq(7),
        membership_scope["ma_allowed"].fillna(False),
        left_name="state7",
        right_name="sma20",
    )
    membership_weekly["estimand"] = "buffered_membership_common_support"
    weekly = pd.concat([proposal_weekly, membership_weekly], ignore_index=True)
    summaries = []
    for estimand, group in weekly.groupby("estimand", sort=True, observed=True):
        summaries.append(
            summarize_weekly_frame(
                group,
                "weekly_difference",
                spec,
                f"h1|{estimand}",
                extra={"estimand": estimand},
                segment_sums={
                    "state7_events": "state7_events",
                    "sma20_events": "sma20_events",
                },
            )
        )
    return weekly, pd.concat(summaries, ignore_index=True)


def _segment_mean(weekly: pd.DataFrame, value_column: str, spec: Mapping[str, Any], segment: str) -> float:
    mask = segment_mask(weekly["decision_dt"], spec, segment)
    values = _finite(weekly.loc[mask, value_column])
    return float(values.mean()) if len(values) else math.nan


def _segment_count(frame: pd.DataFrame, spec: Mapping[str, Any], segment: str) -> int:
    return int(segment_mask(frame["decision_dt"], spec, segment).sum())


def _one_sided_upper_tail(actual: float, null_values: Sequence[float]) -> float:
    values = _finite(null_values)
    return float((1 + np.sum(values >= actual)) / (1 + len(values))) if len(values) else math.nan


def _paired_control_gte_treated_probability(
    treated_values: Sequence[float],
    control_values: Sequence[float],
) -> float:
    """Return the finite-pair share where a sampled control is no worse than treated."""

    treated = np.asarray(treated_values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    valid = np.isfinite(treated) & np.isfinite(control)
    return float((1 + np.sum(control[valid] >= treated[valid])) / (1 + valid.sum())) if valid.any() else math.nan


def run_h1_controls(
    spec: Mapping[str, Any],
    bundle: InputBundle,
    common_weekly: pd.DataFrame,
    common_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run factor-supported matching and proposal-identity randomization for H1."""

    config = spec["h1_state7_increment"]
    scoped = bundle.proposals[
        bundle.proposals["arm"].eq("FC")
        & bundle.proposals["entry_tradable"].fillna(False)
        & np.isfinite(pd.to_numeric(bundle.proposals["fwd_5d_open_return"], errors="coerce"))
    ].copy()
    edges = list(map(int, config["matched_control"]["factor_rank_decile_edges"]))
    scoped["factor_rank_decile"] = pd.cut(
        scoped["factor_rank"],
        bins=edges,
        labels=False,
        right=True,
        include_lowest=True,
    )
    actual = scoped[scoped["regime"].eq(7)].copy()
    control_pool = scoped[~scoped["regime"].eq(7)].copy()
    strata = ["decision_dt", "industry_code", "mcap_bucket", "ma_allowed", "factor_rank_decile"]
    actual_groups = list(actual.groupby(strata, sort=True, observed=True, dropna=False))
    control_groups = {
        key: group["fwd_5d_open_return"].to_numpy(dtype=float)
        for key, group in control_pool.groupby(strata, sort=True, observed=True, dropna=False)
    }
    actual_weekly = (
        actual.groupby("decision_dt", sort=True, observed=True)["fwd_5d_open_return"].mean().rename("state7_return")
    )
    repetitions = int(config["matched_control"]["repetitions"])
    requested_by_segment = {
        segment: int(segment_mask(actual["decision_dt"], spec, segment).sum())
        for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2")
    }
    matched_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = rng_for(spec, "h1_matched", repetition)
        weekly_treated_sum: dict[pd.Timestamp, float] = {}
        weekly_control_sum: dict[pd.Timestamp, float] = {}
        weekly_count: dict[pd.Timestamp, int] = {}
        matched_dates: list[pd.Timestamp] = []
        for key, cohort in actual_groups:
            date = pd.Timestamp(key[0])
            pool = control_groups.get(key)
            if pool is None or not len(pool):
                continue
            take = min(len(cohort), len(pool))
            treated_positions = (
                np.arange(len(cohort), dtype=int)
                if take == len(cohort)
                else rng.choice(len(cohort), size=take, replace=False)
            )
            control_positions = rng.choice(len(pool), size=take, replace=False)
            treated = cohort.iloc[treated_positions]["fwd_5d_open_return"].to_numpy(dtype=float)
            controls = pool[control_positions]
            weekly_treated_sum[date] = weekly_treated_sum.get(date, 0.0) + float(treated.sum())
            weekly_control_sum[date] = weekly_control_sum.get(date, 0.0) + float(controls.sum())
            weekly_count[date] = weekly_count.get(date, 0) + int(take)
            matched_dates.extend([date] * int(take))
        treated_weekly = pd.Series(
            {date: weekly_treated_sum[date] / weekly_count[date] for date in weekly_treated_sum},
            name="matched_state7_return",
            dtype=float,
        )
        control_weekly = pd.Series(
            {date: weekly_control_sum[date] / weekly_count[date] for date in weekly_control_sum},
            name="matched_return",
            dtype=float,
        )
        paired = pd.concat([treated_weekly, control_weekly], axis=1, join="inner").dropna().reset_index()
        paired = paired.rename(columns={"index": "decision_dt"})
        paired["weekly_difference"] = paired["matched_state7_return"] - paired["matched_return"]
        matched_date_series = pd.Series(pd.to_datetime(matched_dates), dtype="datetime64[ns]")
        row: dict[str, Any] = {"repetition": int(repetition)}
        for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
            row[f"state7_minus_matched_{segment}"] = _segment_mean(paired, "weekly_difference", spec, segment)
            row[f"matched_state7_mean_{segment}"] = _segment_mean(paired, "matched_state7_return", spec, segment)
            row[f"matched_control_mean_{segment}"] = _segment_mean(paired, "matched_return", spec, segment)
            matched_count = (
                int(segment_mask(matched_date_series, spec, segment).sum()) if len(matched_date_series) else 0
            )
            requested = requested_by_segment[segment]
            row[f"match_rate_{segment}"] = float(matched_count / requested) if requested else math.nan
            row[f"paired_weeks_{segment}"] = _segment_count(paired, spec, segment)
        matched_rows.append(row)
    matched_distribution = pd.DataFrame(matched_rows)

    week_groups = list(scoped.groupby("decision_dt", sort=True, observed=True))
    random_rows: list[dict[str, Any]] = []
    for repetition in range(int(config["proposal_identity_randomization"]["repetitions"])):
        rng = rng_for(spec, "h1_proposal_identity_randomization", repetition)
        rows: list[dict[str, Any]] = []
        for decision_dt, week in week_groups:
            count = int(week["regime"].eq(7).sum())
            if count <= 0:
                continue
            chosen = rng.choice(len(week), size=count, replace=False)
            rows.append(
                {
                    "decision_dt": pd.Timestamp(decision_dt),
                    "random_return": float(week.iloc[chosen]["fwd_5d_open_return"].mean()),
                }
            )
        random_weekly = pd.DataFrame(rows)
        paired = (
            actual_weekly.rename_axis("decision_dt")
            .reset_index()
            .merge(
                random_weekly,
                on="decision_dt",
                how="inner",
                validate="one_to_one",
            )
        )
        paired["weekly_difference"] = paired["state7_return"] - paired["random_return"]
        row = {"repetition": int(repetition)}
        for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
            row[f"state7_minus_random_{segment}"] = _segment_mean(paired, "weekly_difference", spec, segment)
            row[f"random_mean_{segment}"] = _segment_mean(paired, "random_return", spec, segment)
            row[f"paired_weeks_{segment}"] = _segment_count(paired, spec, segment)
        random_rows.append(row)
    random_distribution = pd.DataFrame(random_rows)

    proposal_validation = common_summary[
        common_summary["estimand"].eq("proposal_common_support") & common_summary["segment"].eq("HISTORICAL_VALIDATION")
    ].iloc[0]
    proposal_v1 = common_summary[
        common_summary["estimand"].eq("proposal_common_support") & common_summary["segment"].eq("V1")
    ].iloc[0]
    proposal_v2 = common_summary[
        common_summary["estimand"].eq("proposal_common_support") & common_summary["segment"].eq("V2")
    ].iloc[0]
    matched_delta = float(matched_distribution["state7_minus_matched_HISTORICAL_VALIDATION"].mean())
    matched_v1 = float(matched_distribution["state7_minus_matched_V1"].mean())
    matched_v2 = float(matched_distribution["state7_minus_matched_V2"].mean())
    matched_mcse = float(
        matched_distribution["state7_minus_matched_HISTORICAL_VALIDATION"].std(ddof=1)
        / math.sqrt(len(matched_distribution))
    )
    match_rate = float(matched_distribution["match_rate_HISTORICAL_VALIDATION"].mean())
    match_rate_v1 = float(matched_distribution["match_rate_V1"].mean())
    match_rate_v2 = float(matched_distribution["match_rate_V2"].mean())
    actual_validation_mean = float(
        actual_weekly.loc[segment_mask(actual_weekly.index, spec, "HISTORICAL_VALIDATION")].mean()
    )
    summary = pd.DataFrame(
        [
            {
                "control": "sma20_common_proposal_weeks",
                "validation_effect": float(proposal_validation["mean"]),
                "v1_effect": float(proposal_v1["mean"]),
                "v2_effect": float(proposal_v2["mean"]),
                "validation_common_weeks": int(proposal_validation["n_weeks"]),
                "validation_match_rate": 1.0,
                "null_mcse": None,
                "descriptive_p_value": normal_two_sided_p(proposal_validation["hac_t"]),
                "descriptive_p_value_type": "two_sided_normal_hac_reference",
            },
            {
                "control": "exact_factor_supported_matched",
                "validation_effect": matched_delta,
                "v1_effect": matched_v1,
                "v2_effect": matched_v2,
                "validation_common_weeks": int(matched_distribution["paired_weeks_HISTORICAL_VALIDATION"].median()),
                "validation_match_rate": match_rate,
                "null_mcse": matched_mcse,
                "descriptive_p_value": _paired_control_gte_treated_probability(
                    matched_distribution["matched_state7_mean_HISTORICAL_VALIDATION"],
                    matched_distribution["matched_control_mean_HISTORICAL_VALIDATION"],
                ),
                "descriptive_p_value_type": "one_sided_paired_matching_draw_tail",
            },
            {
                "control": "weekly_quota_proposal_identity_randomization",
                "validation_effect": float(random_distribution["state7_minus_random_HISTORICAL_VALIDATION"].mean()),
                "v1_effect": float(random_distribution["state7_minus_random_V1"].mean()),
                "v2_effect": float(random_distribution["state7_minus_random_V2"].mean()),
                "validation_common_weeks": int(random_distribution["paired_weeks_HISTORICAL_VALIDATION"].median()),
                "validation_match_rate": 1.0,
                "null_mcse": float(
                    random_distribution["state7_minus_random_HISTORICAL_VALIDATION"].std(ddof=1)
                    / math.sqrt(len(random_distribution))
                ),
                "descriptive_p_value": _one_sided_upper_tail(
                    actual_validation_mean,
                    random_distribution["random_mean_HISTORICAL_VALIDATION"],
                ),
                "descriptive_p_value_type": "one_sided_weekly_identity_randomization_tail",
            },
        ]
    )

    state7_validation_events = int(segment_mask(actual["decision_dt"], spec, "HISTORICAL_VALIDATION").sum())
    sma_sufficient = (
        int(proposal_validation["n_weeks"]) >= int(config["minimum_validation_common_weeks"])
        and int(proposal_v1["n_weeks"]) >= int(config["minimum_each_half_common_weeks"])
        and int(proposal_v2["n_weeks"]) >= int(config["minimum_each_half_common_weeks"])
        and state7_validation_events >= int(config["minimum_validation_state7_events"])
    )
    matched_sufficient = (
        match_rate >= float(config["matched_control"]["minimum_full_match_rate"])
        and match_rate_v1 >= float(config["matched_control"]["minimum_half_match_rate"])
        and match_rate_v2 >= float(config["matched_control"]["minimum_half_match_rate"])
        and matched_mcse <= float(config["matched_control"]["maximum_null_mcse_weekly"])
        and int(matched_distribution["paired_weeks_HISTORICAL_VALIDATION"].median())
        >= int(config["minimum_validation_common_weeks"])
    )
    summary["endpoint_sufficient"] = True
    summary.loc[summary["control"].eq("sma20_common_proposal_weeks"), "endpoint_sufficient"] = bool(sma_sufficient)
    summary.loc[summary["control"].eq("exact_factor_supported_matched"), "endpoint_sufficient"] = bool(
        matched_sufficient
    )
    if not matched_sufficient:
        summary.loc[summary["control"].eq("exact_factor_supported_matched"), "descriptive_p_value"] = None
    endpoints = [
        (
            "state7_minus_sma20",
            sma_sufficient,
            float(proposal_validation["mean"]),
            float(proposal_v1["mean"]),
            float(proposal_v2["mean"]),
        ),
        ("state7_minus_matched", matched_sufficient, matched_delta, matched_v1, matched_v2),
    ]
    falsifying = [
        name
        for name, sufficient, full, half1, half2 in endpoints
        if sufficient and (full <= 0 or half1 <= 0 or half2 <= 0)
    ]
    passing = [
        name
        for name, sufficient, full, half1, half2 in endpoints
        if sufficient and full >= float(spec["statistics"]["effect_threshold_weekly"]) and half1 > 0 and half2 > 0
    ]
    if falsifying:
        status = "FALSIFIED_RETROSPECTIVE"
        reason = f"sufficient endpoint failed sign rule: {', '.join(falsifying)}"
    elif len(passing) == len(endpoints):
        status = "PASS_EXPLORATORY_HISTORICAL"
        reason = "both sufficient endpoints met the frozen effect and half-period rules"
    elif not sma_sufficient or not matched_sufficient:
        status = "INSUFFICIENT_DATA"
        reason = "no sufficient endpoint falsified, but at least one required data gate failed"
    else:
        status = "INCONCLUSIVE"
        reason = "sufficient endpoints were positive but did not both reach the frozen threshold"
    decision = {
        "hypothesis_id": "H1_STATE7_HAS_CHAN_SPECIFIC_INCREMENT",
        "status": status,
        "reason": reason,
        "proposal_endpoint_sufficient": bool(sma_sufficient),
        "matched_endpoint_sufficient": bool(matched_sufficient),
        "validation_state7_events": state7_validation_events,
        "matched_validation_rate": match_rate,
        "matched_v1_rate": match_rate_v1,
        "matched_v2_rate": match_rate_v2,
        "matched_null_mcse": matched_mcse,
        "positive_result_claim_allowed": False,
    }
    return summary, matched_distribution, random_distribution, decision


def _standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    std = float(numeric.std(ddof=0))
    return (numeric - float(numeric.mean())) / std if math.isfinite(std) and std > 0 else numeric * 0.0


def recompute_attribution(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Stage 1 OLS and retain identity-level residuals for null controls."""

    selected = bundle.memberships[bundle.memberships["arm"].eq("F")][["decision_dt", "symbol"]]
    weekly_rows: list[dict[str, Any]] = []
    residual_parts: list[pd.DataFrame] = []
    for decision_dt, universe in bundle.ranked.groupby("decision_dt", sort=True, observed=True):
        universe = universe[
            universe["entry_tradable"].fillna(False)
            & np.isfinite(pd.to_numeric(universe["fwd_5d_open_return"], errors="coerce"))
        ].copy()
        pool_symbols = set(selected.loc[selected["decision_dt"].eq(pd.Timestamp(decision_dt)), "symbol"].astype(str))
        if len(universe) < int(spec["h2_factor_residual"]["minimum_weekly_universe"]):
            continue
        universe["size_z"] = _standardize(universe["log_free_float_mcap"])
        universe["lowvol_z"] = _standardize(universe["lowvol_60"])
        universe["momentum_z"] = _standardize(universe["mom_120_20"])
        dummies = pd.get_dummies(universe["industry_code"].astype(str), prefix="industry", dtype=float)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        dummies = dummies - dummies.mean(axis=0)
        if len(dummies.columns):
            dummies = dummies.iloc[:, 1:]
        design = pd.concat(
            [
                pd.Series(1.0, index=universe.index, name="intercept"),
                universe[["size_z", "lowvol_z", "momentum_z"]],
                dummies,
            ],
            axis=1,
        )
        y = universe["fwd_5d_open_return"].astype(float)
        matrix = design.to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(matrix, y.to_numpy(dtype=float), rcond=None)
        fitted = matrix @ beta
        residual = y.to_numpy(dtype=float) - fitted
        selected_mask = universe["symbol"].isin(pool_symbols).to_numpy()
        if int(selected_mask.sum()) < int(spec["h2_factor_residual"]["minimum_weekly_factor_pool"]):
            continue
        contributions = matrix[selected_mask] * beta
        selected_y = y.to_numpy(dtype=float)[selected_mask]
        industry_start = 4
        row = {
            "decision_dt": pd.Timestamp(decision_dt),
            "n_universe": int(len(universe)),
            "n_selected": int(selected_mask.sum()),
            "selected_return": float(selected_y.mean()),
            "cross_sectional_common_return": float(contributions[:, 0].mean()),
            "size": float(contributions[:, 1].mean()),
            "low_volatility": float(contributions[:, 2].mean()),
            "momentum": float(contributions[:, 3].mean()),
            "industry": float(contributions[:, industry_start:].sum(axis=1).mean())
            if contributions.shape[1] > industry_start
            else 0.0,
            "unexplained_residual": float((selected_y - contributions.sum(axis=1)).mean()),
            "universe_residual_mean": float(residual.mean()),
        }
        row["selected_minus_common_return"] = row["selected_return"] - row["cross_sectional_common_return"]
        row["reconstruction_error"] = float(
            row["selected_return"]
            - sum(
                row[key]
                for key in (
                    "cross_sectional_common_return",
                    "size",
                    "industry",
                    "low_volatility",
                    "momentum",
                    "unexplained_residual",
                )
            )
        )
        weekly_rows.append(row)
        residual_parts.append(
            pd.DataFrame(
                {
                    "decision_dt": pd.Timestamp(decision_dt),
                    "symbol": universe["symbol"].astype(str).to_numpy(),
                    "residual": residual,
                    "is_factor_member": selected_mask,
                }
            )
        )
    weekly = pd.DataFrame(weekly_rows)
    residual_surface = pd.concat(residual_parts, ignore_index=True)
    maximum_error = float(weekly["reconstruction_error"].abs().max())
    maximum_universe_residual = float(weekly["universe_residual_mean"].abs().max())
    if maximum_error > float(spec["h2_factor_residual"]["maximum_reconstruction_error"]):
        raise Stage2Error(f"attribution reconstruction failed: {maximum_error}")
    if maximum_universe_residual > 1e-10:
        raise Stage2Error(f"weekly universe residual mean is not centered: {maximum_universe_residual}")

    frozen = bundle.attribution_weekly.rename(columns={"market": "cross_sectional_common_return"}).copy()
    compare_columns = [
        "selected_return",
        "cross_sectional_common_return",
        "size",
        "industry",
        "low_volatility",
        "momentum",
        "unexplained_residual",
        "reconstruction_error",
    ]
    comparison = weekly[["decision_dt", *compare_columns]].merge(
        frozen[["decision_dt", *compare_columns]],
        on="decision_dt",
        suffixes=("_recomputed", "_frozen"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise Stage2Error("recomputed attribution dates differ from frozen Stage 1")
    maximum_frozen_difference = max(
        float((comparison[f"{column}_recomputed"] - comparison[f"{column}_frozen"]).abs().max())
        for column in compare_columns
    )
    if maximum_frozen_difference > 1e-12:
        raise Stage2Error(f"recomputed attribution differs from frozen Stage 1: {maximum_frozen_difference}")
    return weekly, residual_surface


def run_h2_residual_stability(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recompute H2 and apply the frozen validation/half-period rule."""

    weekly, residual_surface = recompute_attribution(spec, bundle)
    summary_parts = []
    for column in (
        "selected_return",
        "cross_sectional_common_return",
        "selected_minus_common_return",
        "size",
        "industry",
        "low_volatility",
        "momentum",
        "unexplained_residual",
    ):
        summary_parts.append(
            summarize_weekly_frame(
                weekly,
                column,
                spec,
                f"h2|{column}",
                extra={"component": column},
            )
        )
    summary = pd.concat(summary_parts, ignore_index=True)
    residual_rows = summary[summary["component"].eq("unexplained_residual")].set_index("segment")
    validation = residual_rows.loc["HISTORICAL_VALIDATION"]
    v1 = residual_rows.loc["V1"]
    v2 = residual_rows.loc["V2"]
    config = spec["h2_factor_residual"]
    sufficient = (
        int(validation["n_weeks"]) >= int(config["minimum_validation_weeks"])
        and int(v1["n_weeks"]) >= int(config["minimum_each_half_weeks"])
        and int(v2["n_weeks"]) >= int(config["minimum_each_half_weeks"])
        and bool((weekly["n_universe"] >= int(config["minimum_weekly_universe"])).all())
        and bool((weekly["n_selected"] >= int(config["minimum_weekly_factor_pool"])).all())
    )
    if not sufficient:
        status = "INSUFFICIENT_DATA"
        reason = "one or more frozen weekly coverage gates failed"
    elif float(validation["mean"]) <= 0 or float(v1["mean"]) <= 0 or float(v2["mean"]) <= 0:
        status = "FALSIFIED_RETROSPECTIVE"
        reason = "validation residual or at least one validation half is non-positive"
    elif (
        float(validation["mean"]) >= float(spec["statistics"]["effect_threshold_weekly"])
        and float(v1["mean"]) > 0
        and float(v2["mean"]) > 0
    ):
        status = "PASS_EXPLORATORY_HISTORICAL"
        reason = "historical validation and both halves met the frozen rule"
    else:
        status = "INCONCLUSIVE"
        reason = "residual was positive but below the frozen effect threshold"
    decision = {
        "hypothesis_id": "H2_FACTOR_HAS_RESIDUAL_AFTER_RISK_ATTRIBUTION",
        "status": status,
        "reason": reason,
        "validation_mean": validation["mean"],
        "validation_hac_t": validation["hac_t"],
        "v1_mean": v1["mean"],
        "v2_mean": v2["mean"],
        "maximum_reconstruction_error": float(weekly["reconstruction_error"].abs().max()),
        "maximum_universe_residual_mean": float(weekly["universe_residual_mean"].abs().max()),
        "positive_result_claim_allowed": False,
    }
    return weekly, summary, residual_surface, decision


def run_h2_random_identity(
    spec: Mapping[str, Any],
    residual_surface: pd.DataFrame,
    attribution_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the factor-pool residual with random weekly 50-name residual baskets."""

    config = spec["h2_factor_residual"]["random_identity_control"]
    repetitions = int(config["repetitions"])
    sample_size = int(config["weekly_sample_size"])
    groups = [
        (pd.Timestamp(date), group["residual"].to_numpy(dtype=float))
        for date, group in residual_surface.groupby("decision_dt", sort=True, observed=True)
    ]
    matrices = np.empty((repetitions, len(groups)), dtype=float)
    for repetition in range(repetitions):
        rng = rng_for(spec, "h2_random_identity", repetition)
        for column, (_date, values) in enumerate(groups):
            if len(values) < sample_size:
                raise Stage2Error("random residual control has fewer names than the frozen sample size")
            chosen = rng.choice(len(values), size=sample_size, replace=False)
            matrices[repetition, column] = float(values[chosen].mean())
    dates = pd.DatetimeIndex([date for date, _values in groups])
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        row = {"repetition": int(repetition)}
        for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
            mask = segment_mask(dates, spec, segment)
            row[f"random_residual_mean_{segment}"] = float(matrices[repetition, mask].mean())
        rows.append(row)
    distribution = pd.DataFrame(rows)
    actual = attribution_weekly.set_index("decision_dt")["unexplained_residual"]
    actual_validation = float(actual.loc[segment_mask(actual.index, spec, "HISTORICAL_VALIDATION")].mean())
    null_values = distribution["random_residual_mean_HISTORICAL_VALIDATION"]
    summary = pd.DataFrame(
        [
            {
                "control": "weekly_random_50_identity_residual",
                "repetitions": repetitions,
                "actual_validation_residual": actual_validation,
                "null_validation_mean": float(null_values.mean()),
                "actual_minus_null": float(actual_validation - null_values.mean()),
                "null_mcse": float(null_values.std(ddof=1) / math.sqrt(repetitions)),
                "one_sided_tail_probability": _one_sided_upper_tail(actual_validation, null_values),
            }
        ]
    )
    return summary, distribution


def feature_failure_profile(
    frame: pd.DataFrame,
    feature: str,
    failure_column: str,
) -> dict[str, Any]:
    """Compute one pooled failure-minus-profit standardized difference."""

    values = pd.to_numeric(frame[feature], errors="coerce").astype(float)
    labels = frame[failure_column].astype(bool)
    valid = np.isfinite(values.to_numpy(dtype=float))
    scoped_values = values[valid]
    scoped_labels = labels[valid]
    failure = scoped_values[scoped_labels]
    profit = scoped_values[~scoped_labels]
    std = float(scoped_values.std(ddof=0))
    difference = float(failure.mean() - profit.mean()) if len(failure) and len(profit) else math.nan
    return {
        "feature": feature,
        "failure_column": failure_column,
        "n_total": int(len(frame)),
        "n_non_missing": int(valid.sum()),
        "non_missing_rate": float(valid.mean()) if len(valid) else math.nan,
        "n_failure": int(len(failure)),
        "n_profit": int(len(profit)),
        "failure_mean": float(failure.mean()) if len(failure) else math.nan,
        "profit_mean": float(profit.mean()) if len(profit) else math.nan,
        "mean_difference": difference,
        "standardized_difference": difference / std if math.isfinite(std) and std > 0 else math.nan,
    }


def _profile_segments(
    frame: pd.DataFrame,
    features: Sequence[str],
    failure_column: str,
    spec: Mapping[str, Any],
    *,
    profile_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in ("DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
        scoped = frame[segment_mask(frame["decision_dt"], spec, segment)]
        for feature in features:
            rows.append(
                {
                    "profile_type": profile_type,
                    "segment": segment,
                    **feature_failure_profile(scoped, feature, failure_column),
                }
            )
    return pd.DataFrame(rows)


def _validation_tail_lift(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    feature: str,
    failure_column: str,
) -> dict[str, Any]:
    profile = feature_failure_profile(development, feature, failure_column)
    direction = 1 if float(profile["standardized_difference"]) >= 0 else -1
    development_values = pd.to_numeric(development[feature], errors="coerce").astype(float)
    quantile = 0.75 if direction > 0 else 0.25
    threshold = float(development_values.quantile(quantile))
    validation_values = pd.to_numeric(validation[feature], errors="coerce").astype(float)
    valid = validation_values.notna()
    risk_tail = validation_values.ge(threshold) if direction > 0 else validation_values.le(threshold)
    labels = validation[failure_column].astype(bool)
    tail_rate = float(labels[valid & risk_tail].mean())
    remainder_rate = float(labels[valid & ~risk_tail].mean())
    return {
        "feature": feature,
        "development_direction": direction,
        "development_quantile": quantile,
        "development_threshold": threshold,
        "validation_tail_n": int((valid & risk_tail).sum()),
        "validation_remainder_n": int((valid & ~risk_tail).sum()),
        "validation_tail_failure_rate": tail_rate,
        "validation_remainder_failure_rate": remainder_rate,
        "validation_failure_rate_lift": tail_rate - remainder_rate,
    }


def run_h3_profile_replication(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Separate invalid post-selection features from an early-only nested diagnostic."""

    config = spec["h3_failure_profile"]
    frame = bundle.failure_sample.copy()
    registered_features = list(config["registered_post_selection_features"])
    registered = _profile_segments(
        frame,
        registered_features,
        "failure_primary",
        spec,
        profile_type="registered_post_selection_diagnostic",
    )
    development = frame[segment_mask(frame["decision_dt"], spec, "DEVELOPMENT")]
    candidate_rows = [
        feature_failure_profile(development, feature, "failure_primary")
        for feature in config["nested_candidate_features"]
    ]
    development_candidates = pd.DataFrame(candidate_rows)
    development_candidates["abs_standardized_difference"] = development_candidates["standardized_difference"].abs()
    selected_features = (
        development_candidates.sort_values(
            ["abs_standardized_difference", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(2)["feature"]
        .astype(str)
        .tolist()
    )
    nested = _profile_segments(
        frame,
        selected_features,
        "failure_primary",
        spec,
        profile_type="development_only_nested_diagnostic",
    )
    validation = frame[segment_mask(frame["decision_dt"], spec, "HISTORICAL_VALIDATION")]
    tail_lifts = pd.DataFrame(
        [_validation_tail_lift(development, validation, feature, "failure_primary") for feature in selected_features]
    )

    validation_rows = nested[nested["segment"].eq("HISTORICAL_VALIDATION")].set_index("feature")
    development_rows = nested[nested["segment"].eq("DEVELOPMENT")].set_index("feature")
    data_sufficient = True
    data_reasons: list[str] = []
    effects: list[tuple[str, float, float]] = []
    for feature in selected_features:
        dev = development_rows.loc[feature]
        val = validation_rows.loc[feature]
        if (
            float(val["non_missing_rate"]) < float(config["minimum_validation_non_missing_rate"])
            or int(val["n_failure"]) < int(config["minimum_validation_failure_rows"])
            or int(val["n_profit"]) < int(config["minimum_validation_profit_rows"])
            or not math.isfinite(float(val["standardized_difference"]))
        ):
            data_sufficient = False
            data_reasons.append(f"{feature} failed non-missing/group-size/variance gate")
        effects.append(
            (
                feature,
                float(dev["standardized_difference"]),
                float(val["standardized_difference"]),
            )
        )
    threshold = float(spec["statistics"]["failure_profile_smd_threshold"])
    reversals = [
        feature for feature, development_smd, validation_smd in effects if development_smd * validation_smd <= 0
    ]
    weak = [feature for feature, _development_smd, validation_smd in effects if abs(validation_smd) < threshold]
    if not data_sufficient:
        nested_status = "INSUFFICIENT_DATA"
        nested_reason = "; ".join(data_reasons)
    elif reversals or len(weak) == len(effects):
        nested_status = "FALSIFIED_RETROSPECTIVE"
        nested_reason = f"direction reversals={reversals}; validation effects below {threshold:.2f}={weak}"
    elif all(
        development_smd * validation_smd > 0 and abs(validation_smd) >= threshold
        for _feature, development_smd, validation_smd in effects
    ):
        nested_status = "PASS_EXPLORATORY_HISTORICAL"
        nested_reason = "both development-selected features replicated direction and frozen magnitude"
    else:
        nested_status = "INCONCLUSIVE"
        nested_reason = "one feature replicated strongly and one remained weak without reversing"
    registered_decision = {
        "hypothesis_id": "H3_FAILURE_PROFILE_REPLICATES_OUT_OF_TIME",
        "status": "INVALID_POST_SELECTION",
        "reason": "the registered features were selected using the full sample that includes the validation period",
        "features": registered_features,
        "positive_result_claim_allowed": False,
    }
    nested_decision = {
        "hypothesis_id": "H3_NESTED_FAILURE_PROFILE_DIAGNOSTIC",
        "status": nested_status,
        "reason": nested_reason,
        "selected_development_features": selected_features,
        "effects": [
            {
                "feature": feature,
                "development_smd": development_smd,
                "historical_validation_smd": validation_smd,
            }
            for feature, development_smd, validation_smd in effects
        ],
        "positive_result_claim_allowed": False,
    }
    return registered, development_candidates, nested, tail_lifts, registered_decision, nested_decision


def _smd_from_arrays(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    scoped = values[valid]
    scoped_labels = labels[valid]
    if not len(scoped) or not scoped_labels.any() or scoped_labels.all():
        return math.nan
    std = float(np.std(scoped, ddof=0))
    if std <= 0 or not math.isfinite(std):
        return math.nan
    return float((scoped[scoped_labels].mean() - scoped[~scoped_labels].mean()) / std)


def _deterministic_noise(symbol: str, decision_dt: pd.Timestamp, study_id: str) -> float:
    digest = hashlib.sha256(f"{symbol}|{pd.Timestamp(decision_dt).date().isoformat()}|{study_id}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    uniform = (integer + 0.5) / (2**64)
    return float(NormalDist().inv_cdf(uniform))


def run_h3_controls_and_secondary(
    spec: Mapping[str, Any],
    bundle: InputBundle,
    nested_profiles: pd.DataFrame,
    development_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run label permutations, deterministic noise and the omitted secondary profile."""

    config = spec["h3_failure_profile"]
    selected_features = (
        development_candidates.sort_values(
            ["abs_standardized_difference", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(2)["feature"]
        .astype(str)
        .tolist()
    )
    validation = bundle.failure_sample[
        segment_mask(bundle.failure_sample["decision_dt"], spec, "HISTORICAL_VALIDATION")
    ].reset_index(drop=True)
    labels = validation["failure_primary"].astype(bool).to_numpy()
    group_indices = [
        group.index.to_numpy(dtype=int) for _date, group in validation.groupby("decision_dt", sort=True, observed=True)
    ]
    value_arrays = {
        feature: pd.to_numeric(validation[feature], errors="coerce").to_numpy(dtype=float)
        for feature in selected_features
    }
    permutation_rows: list[dict[str, Any]] = []
    for repetition in range(int(config["permutation_control_repetitions"])):
        rng = rng_for(spec, "h3_failure_label_permutation", repetition)
        permuted = labels.copy()
        for indices in group_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        for feature, values in value_arrays.items():
            permutation_rows.append(
                {
                    "repetition": int(repetition),
                    "feature": feature,
                    "standardized_difference": _smd_from_arrays(values, permuted),
                }
            )
    permutation = pd.DataFrame(permutation_rows)
    observed = nested_profiles[nested_profiles["segment"].eq("HISTORICAL_VALIDATION")].set_index("feature")
    control_rows = []
    for feature in selected_features:
        null_values = permutation.loc[permutation["feature"].eq(feature), "standardized_difference"].to_numpy(
            dtype=float
        )
        observed_smd = float(observed.loc[feature, "standardized_difference"])
        control_rows.append(
            {
                "feature": feature,
                "observed_smd": observed_smd,
                "null_mean": float(np.nanmean(null_values)),
                "null_std": float(np.nanstd(null_values, ddof=1)),
                "two_sided_tail_probability": float(
                    (1 + np.sum(np.abs(null_values) >= abs(observed_smd))) / (1 + len(null_values))
                ),
            }
        )
    permutation_summary = pd.DataFrame(control_rows)

    noise_frame = bundle.failure_sample.copy()
    noise_frame["synthetic_sha256_noise"] = [
        _deterministic_noise(symbol, decision_dt, spec["study_id"])
        for symbol, decision_dt in zip(
            noise_frame["symbol"],
            noise_frame["decision_dt"],
            strict=True,
        )
    ]
    noise_profile = _profile_segments(
        noise_frame,
        ["synthetic_sha256_noise"],
        "failure_primary",
        spec,
        profile_type="deterministic_noise_negative_control",
    )
    secondary = _profile_segments(
        bundle.failure_sample,
        list(config["nested_candidate_features"]),
        "failure_bottom_quintile",
        spec,
        profile_type="secondary_bottom_quintile_profile",
    )
    return permutation_summary, permutation, noise_profile, secondary


def _cost_columns(roundtrip: float, base_buy: float, base_sell: float) -> tuple[float, float]:
    if math.isclose(roundtrip, base_buy + base_sell, rel_tol=0, abs_tol=1e-12):
        return base_buy, base_sell
    return roundtrip / 2.0, roundtrip / 2.0


def build_arm_weekly_paths(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> pd.DataFrame:
    """Build fixed-50-slot cash-scaled arm returns with explicit buys and sells."""

    config = spec["portfolio_path_stress"]
    target = int(config["target_slots"])
    deployable = 1.0 - float(config["cash_buffer"])
    dates = bundle.decision_dates
    rows: list[dict[str, Any]] = []
    for arm in config["arms"]:
        scoped = bundle.memberships[bundle.memberships["arm"].eq(arm)].copy()
        by_date = {
            pd.Timestamp(date): group.set_index("symbol", drop=False)
            for date, group in scoped.groupby("decision_dt", sort=True, observed=True)
        }
        previous: set[str] = set()
        arm_rows: list[dict[str, Any]] = []
        for decision_dt in dates:
            week = by_date.get(pd.Timestamp(decision_dt))
            if week is None:
                week = scoped.iloc[:0].set_index("symbol", drop=False)
            current = set(week.index.astype(str))
            buys = current - previous
            sells = previous - current
            returns = pd.to_numeric(week["fwd_5d_open_return"], errors="coerce")
            observable = week["entry_tradable"].fillna(False) & np.isfinite(returns)
            gross_return = deployable * float(returns.where(observable, 0.0).sum()) / target
            arm_rows.append(
                {
                    "decision_dt": pd.Timestamp(decision_dt),
                    "arm": str(arm),
                    "slots": int(len(current)),
                    "observable_slots": int(observable.sum()),
                    "buy_count": int(len(buys)),
                    "sell_count": int(len(sells)),
                    "terminal_sell_count": 0,
                    "gross_return": gross_return,
                }
            )
            previous = current
        if arm_rows and bool(config["terminal_liquidation"]):
            arm_rows[-1]["terminal_sell_count"] = int(len(previous))
        rows.extend(arm_rows)
    weekly = pd.DataFrame(rows)
    base_buy = float(config["buy_cost"])
    base_sell = float(config["sell_cost"])
    for roundtrip in map(float, config["roundtrip_cost_stress"]):
        buy_cost, sell_cost = _cost_columns(roundtrip, base_buy, base_sell)
        suffix = f"{int(roundtrip * 10_000)}bps"
        weekly[f"cost_{suffix}"] = (
            deployable
            * (weekly["buy_count"] * buy_cost + (weekly["sell_count"] + weekly["terminal_sell_count"]) * sell_cost)
            / target
        )
        weekly[f"net_return_{suffix}"] = weekly["gross_return"] - weekly[f"cost_{suffix}"]
    weekly["net_return"] = weekly["net_return_40bps"]
    weekly["slot_fill_rate"] = weekly["slots"] / target
    weekly["observable_target_slot_rate"] = weekly["observable_slots"] / target
    return weekly


def summarize_arm_paths(
    spec: Mapping[str, Any],
    arm_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize every arm and the paired FC differences without selecting variants."""

    parts: list[pd.DataFrame] = []
    return_columns = ["gross_return", "net_return_0bps", "net_return_20bps", "net_return_40bps", "net_return_60bps"]
    for arm, group in arm_weekly.groupby("arm", sort=True, observed=True):
        for column in return_columns:
            summary = summarize_weekly_frame(
                group,
                column,
                spec,
                f"path|{arm}|{column}",
                extra={
                    "path": arm,
                    "metric": column,
                },
                segment_means={
                    "mean_slots": "slots",
                    "mean_slot_fill_rate": "slot_fill_rate",
                    "observable_target_slot_rate": "observable_target_slot_rate",
                },
            )
            parts.append(summary)
    pivot = arm_weekly.pivot(index="decision_dt", columns="arm", values="net_return_40bps")
    for comparator in ("F", "FMA"):
        paired = pivot[["FC", comparator]].dropna().reset_index()
        paired["weekly_difference"] = paired["FC"] - paired[comparator]
        parts.append(
            summarize_weekly_frame(
                paired,
                "weekly_difference",
                spec,
                f"path|FC_minus_{comparator}",
                extra={
                    "path": f"FC_minus_{comparator}",
                    "metric": "paired_net_return_40bps",
                },
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_recursive_randomized_paths(
    spec: Mapping[str, Any],
    bundle: InputBundle,
    actual_arm_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recursively rebuild random Chan paths so holdings affect later proposals."""

    config = spec["portfolio_path_stress"]
    random_config = config["recursive_randomized_chan_paths"]
    repetitions = int(random_config["repetitions"])
    target = int(config["target_slots"])
    retention_rank = 75
    deployable = 1.0 - float(config["cash_buffer"])
    buy_cost = float(config["buy_cost"])
    sell_cost = float(config["sell_cost"])
    week_data: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for decision_dt, week in bundle.ranked.groupby("decision_dt", sort=True, observed=True):
        scoped = week[week["factor_rank"].le(retention_rank)].sort_values(["factor_rank", "symbol"], kind="mergesort")
        if len(scoped) < target:
            raise Stage2Error("recursive path has fewer than 50 rank<=75 candidates")
        week_data.append((pd.Timestamp(decision_dt), scoped.reset_index(drop=True)))

    distribution_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = rng_for(spec, "recursive_randomized_chan_path", repetition)
        previous: set[str] = set()
        weekly_rows: list[dict[str, Any]] = []
        for decision_dt, week in week_data:
            symbols = week["symbol"].astype(str).to_numpy()
            symbol_to_position = {symbol: index for index, symbol in enumerate(symbols)}
            retained = {
                symbol
                for symbol in previous
                if symbol in symbol_to_position
                and int(week.iloc[symbol_to_position[symbol]]["factor_rank"]) <= retention_rank
            }
            slots = target - len(retained)
            candidates = [symbol for symbol in symbols if symbol not in retained][:slots]
            allowed = week["regime"].isin([5, 6, 7, 8]).to_numpy(dtype=bool)
            permuted_allowed = rng.permutation(allowed)
            accepted = {symbol for symbol in candidates if bool(permuted_allowed[symbol_to_position[symbol]])}
            current = retained | accepted
            buys = current - previous
            sells = previous - current
            positions = np.fromiter((symbol_to_position[symbol] for symbol in current), dtype=int)
            position_returns = (
                pd.to_numeric(week.iloc[positions]["fwd_5d_open_return"], errors="coerce").to_numpy(dtype=float)
                if len(positions)
                else np.array([], dtype=float)
            )
            position_tradable = (
                week.iloc[positions]["entry_tradable"].fillna(False).to_numpy(dtype=bool)
                if len(positions)
                else np.array([], dtype=bool)
            )
            valid = position_tradable & np.isfinite(position_returns)
            gross = deployable * float(np.where(valid, position_returns, 0.0).sum()) / target
            cost = deployable * (len(buys) * buy_cost + len(sells) * sell_cost) / target
            weekly_rows.append(
                {
                    "decision_dt": decision_dt,
                    "gross_return": gross,
                    "net_return": gross - cost,
                    "slots": int(len(current)),
                    "observable_slots": int(valid.sum()),
                    "buy_count": int(len(buys)),
                    "sell_count": int(len(sells)),
                }
            )
            previous = current
        if weekly_rows and bool(config["terminal_liquidation"]):
            terminal_cost = deployable * len(previous) * sell_cost / target
            weekly_rows[-1]["net_return"] -= terminal_cost
            weekly_rows[-1]["sell_count"] += len(previous)
        weekly = pd.DataFrame(weekly_rows)
        row: dict[str, Any] = {
            "repetition": int(repetition),
            "mean_slots": float(weekly["slots"].mean()),
            "mean_observable_slots": float(weekly["observable_slots"].mean()),
        }
        for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
            row[f"net_mean_{segment}"] = _segment_mean(weekly, "net_return", spec, segment)
            row[f"gross_mean_{segment}"] = _segment_mean(weekly, "gross_return", spec, segment)
        distribution_rows.append(row)
    distribution = pd.DataFrame(distribution_rows)
    actual_fc = actual_arm_weekly[actual_arm_weekly["arm"].eq("FC")]
    actual_validation = _segment_mean(actual_fc, "net_return", spec, "HISTORICAL_VALIDATION")
    null_values = distribution["net_mean_HISTORICAL_VALIDATION"]
    summary = pd.DataFrame(
        [
            {
                "control": "recursive_weekly_allowed_identity_permutation",
                "repetitions": repetitions,
                "actual_fc_validation_net_return": actual_validation,
                "null_validation_mean": float(null_values.mean()),
                "actual_minus_null": float(actual_validation - null_values.mean()),
                "one_sided_tail_probability": _one_sided_upper_tail(actual_validation, null_values),
                "actual_mean_slots": float(actual_fc["slots"].mean()),
                "null_mean_slots": float(distribution["mean_slots"].mean()),
            }
        ]
    )
    return summary, distribution


def run_path_stress(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run fixed-slot, fixed-cost and recursive random-path diagnostics."""

    weekly = build_arm_weekly_paths(spec, bundle)
    summary = summarize_arm_paths(spec, weekly)
    random_summary, random_distribution = run_recursive_randomized_paths(spec, bundle, weekly)
    return weekly, summary, random_summary, random_distribution


def full_calendar_turnover(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Compute Jaccard turnover on every calendar week, including empty weeks."""

    by_date = {
        pd.Timestamp(date): set(group["symbol"].astype(str))
        for date, group in frame.groupby("decision_dt", sort=True, observed=True)
    }
    previous: set[str] = set()
    turnovers: list[float] = []
    entries: list[int] = []
    exits: list[int] = []
    present: list[bool] = []
    for position, decision_dt in enumerate(dates):
        current = by_date.get(pd.Timestamp(decision_dt), set())
        present.append(bool(current))
        if position > 0:
            union = current | previous
            turnovers.append(1.0 - len(current & previous) / len(union) if union else 0.0)
            entries.append(len(current - previous))
            exits.append(len(previous - current))
        previous = current
    runs: list[int] = []
    current_run = 0
    for value in present:
        if value:
            current_run += 1
        elif current_run:
            runs.append(current_run)
            current_run = 0
    if current_run:
        runs.append(current_run)
    return {
        "full_calendar_jaccard_turnover": float(np.mean(turnovers)) if turnovers else math.nan,
        "mean_weekly_entries": float(np.mean(entries)) if entries else math.nan,
        "mean_weekly_exits": float(np.mean(exits)) if exits else math.nan,
        "presence_weeks": int(sum(present)),
        "presence_run_count": int(len(runs)),
        "median_presence_run_weeks": float(np.median(runs)) if runs else 0.0,
        "max_presence_run_weeks": int(max(runs)) if runs else 0,
    }


def run_state_placebos(
    spec: Mapping[str, Any],
    bundle: InputBundle,
) -> pd.DataFrame:
    """Report every state on proposal and membership views without choosing a winner."""

    views = {
        "FC_proposal": bundle.proposals[bundle.proposals["arm"].eq("FC")],
        "F_membership": bundle.memberships[bundle.memberships["arm"].eq("F")],
    }
    rows: list[dict[str, Any]] = []
    for view, raw in views.items():
        raw = raw[
            raw["entry_tradable"].fillna(False) & np.isfinite(pd.to_numeric(raw["fwd_5d_open_return"], errors="coerce"))
        ].copy()
        for state in spec["placebo_and_market_state"]["placebo_states"]:
            cohort = raw[raw["regime"].eq(int(state))].copy()
            weekly = (
                cohort.groupby("decision_dt", sort=True, observed=True)["fwd_5d_open_return"]
                .mean()
                .rename("weekly_return")
                .reset_index()
            )
            turnover = full_calendar_turnover(cohort, bundle.decision_dates)
            for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
                mask = segment_mask(weekly["decision_dt"], spec, segment) if len(weekly) else np.array([], dtype=bool)
                stats = weekly_stats(
                    weekly.loc[mask, "weekly_return"] if len(weekly) else [],
                    spec,
                    f"placebo|{view}|{state}|{segment}",
                )
                rows.append(
                    {
                        "view": view,
                        "state": int(state),
                        "state_name": stage1.STATE_NAMES[int(state)],
                        "segment": segment,
                        "n_events": int(segment_mask(cohort["decision_dt"], spec, segment).sum()) if len(cohort) else 0,
                        "minimum_event_gate_passed": bool(
                            int(segment_mask(cohort["decision_dt"], spec, segment).sum())
                            >= int(spec["placebo_and_market_state"]["minimum_placebo_events"])
                        )
                        if len(cohort)
                        else False,
                        **stats,
                        **turnover,
                    }
                )
    return pd.DataFrame(rows)


def build_market_state_weekly(bundle: InputBundle) -> pd.DataFrame:
    """Build two fixed decision-time market-state variables from eligible ranks."""

    return (
        bundle.ranked.groupby("decision_dt", sort=True, observed=True)
        .agg(
            eligible_ma20_breadth=("ma_allowed", "mean"),
            eligible_median_momentum=("mom_120_20", "median"),
            eligible_names=("symbol", "size"),
        )
        .reset_index()
    )


def run_market_state_mechanisms(
    spec: Mapping[str, Any],
    bundle: InputBundle,
    h1_common_weekly: pd.DataFrame,
    arm_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Condition two fixed outcomes on two fixed, decision-time market states."""

    state_weekly = build_market_state_weekly(bundle)
    proposal = h1_common_weekly[h1_common_weekly["estimand"].eq("proposal_common_support")][
        ["decision_dt", "weekly_difference"]
    ].rename(columns={"weekly_difference": "proposal_state7_minus_sma20_return"})
    path_pivot = arm_weekly.pivot(index="decision_dt", columns="arm", values="net_return_40bps").reset_index()
    path_pivot["FC_minus_F_fixed_50_slot_net_return"] = path_pivot["FC"] - path_pivot["F"]
    weekly = (
        state_weekly.merge(
            path_pivot[["decision_dt", "FC_minus_F_fixed_50_slot_net_return"]],
            on="decision_dt",
            how="left",
            validate="one_to_one",
        )
        .merge(proposal, on="decision_dt", how="left", validate="one_to_one")
        .sort_values("decision_dt", kind="mergesort")
    )
    weekly["eligible_ma20_breadth_bin"] = np.where(
        weekly["eligible_ma20_breadth"].ge(0.50),
        "gte_0.50",
        "lt_0.50",
    )
    weekly["eligible_median_momentum_bin"] = np.where(
        weekly["eligible_median_momentum"].gt(0),
        "gt_0",
        "lte_0",
    )
    outcomes = list(spec["placebo_and_market_state"]["outcomes"])
    summary_rows: list[dict[str, Any]] = []
    for variable in ("eligible_ma20_breadth", "eligible_median_momentum"):
        bin_column = f"{variable}_bin"
        for bin_name, group in weekly.groupby(bin_column, sort=True, observed=True):
            for outcome in outcomes:
                for segment in ("DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
                    mask = segment_mask(group["decision_dt"], spec, segment)
                    summary_rows.append(
                        {
                            "market_state_variable": variable,
                            "market_state_bin": bin_name,
                            "outcome": outcome,
                            "segment": segment,
                            **weekly_stats(
                                group.loc[mask, outcome],
                                spec,
                                f"market_state|{variable}|{bin_name}|{outcome}|{segment}",
                            ),
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    association_rows: list[dict[str, Any]] = []
    for variable in ("eligible_ma20_breadth", "eligible_median_momentum"):
        for outcome in outcomes:
            for segment in ("FULL", "DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"):
                scoped = weekly.loc[segment_mask(weekly["decision_dt"], spec, segment), [variable, outcome]].dropna()
                association_rows.append(
                    {
                        "market_state_variable": variable,
                        "outcome": outcome,
                        "segment": segment,
                        "n_weeks": int(len(scoped)),
                        "spearman_correlation": float(scoped[variable].corr(scoped[outcome], method="spearman"))
                        if len(scoped) >= 3
                        else math.nan,
                    }
                )
    return weekly, summary, pd.DataFrame(association_rows)


def run_placebo_and_market_state(
    spec: Mapping[str, Any],
    bundle: InputBundle,
    h1_common_weekly: pd.DataFrame,
    arm_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all-state placebo and fixed market-state mechanism diagnostics."""

    placebos = run_state_placebos(spec, bundle)
    market_weekly, market_summary, market_associations = run_market_state_mechanisms(
        spec,
        bundle,
        h1_common_weekly,
        arm_weekly,
    )
    return placebos, market_weekly, market_summary, market_associations


def derive_decision_matrix(
    spec: Mapping[str, Any],
    h1_decision: Mapping[str, Any],
    h2_decision: Mapping[str, Any],
    h3_registered_decision: Mapping[str, Any],
    h3_nested_decision: Mapping[str, Any],
    h1_common_summary: pd.DataFrame,
    h1_control_summary: pd.DataFrame,
    h2_summary: pd.DataFrame,
    h3_permutation_summary: pd.DataFrame,
    arm_summary: pd.DataFrame,
    market_associations: pd.DataFrame,
) -> dict[str, Any]:
    """Freeze hypothesis statuses, descriptive Holm values and next actions."""

    proposal = h1_common_summary[
        h1_common_summary["estimand"].eq("proposal_common_support")
        & h1_common_summary["segment"].eq("HISTORICAL_VALIDATION")
    ].iloc[0]
    matched = h1_control_summary[h1_control_summary["control"].eq("exact_factor_supported_matched")].iloc[0]
    residual = h2_summary[
        h2_summary["component"].eq("unexplained_residual") & h2_summary["segment"].eq("HISTORICAL_VALIDATION")
    ].iloc[0]
    matched_endpoint_sufficient = bool(h1_decision["matched_endpoint_sufficient"])
    p_values: dict[str, float | None] = {
        "H1_state7_minus_sma20": normal_two_sided_p(proposal["hac_t"]),
        "H1_state7_minus_matched": (
            None
            if not matched_endpoint_sufficient or pd.isna(matched["descriptive_p_value"])
            else float(matched["descriptive_p_value"])
        ),
        "H2_unexplained_residual": normal_two_sided_p(residual["hac_t"]),
    }
    for index, row in enumerate(
        h3_permutation_summary.sort_values("feature", kind="mergesort").itertuples(index=False),
        1,
    ):
        p_values[f"H3_nested_feature_{index}_{row.feature}"] = float(row.two_sided_tail_probability)
    holm = holm_adjust(
        p_values,
        family_size=int(spec["statistics"]["holm_endpoint_count"]),
    )
    endpoints = [
        {
            "endpoint": key,
            "descriptive_p": value,
            "holm_adjusted_p": holm[key],
            "endpoint_sufficient": (matched_endpoint_sufficient if key == "H1_state7_minus_matched" else True),
            "confirmation_use": "FORBIDDEN",
        }
        for key, value in p_values.items()
    ]

    fc_minus_f = arm_summary[
        arm_summary["path"].eq("FC_minus_F")
        & arm_summary["metric"].eq("paired_net_return_40bps")
        & arm_summary["segment"].eq("HISTORICAL_VALIDATION")
    ].iloc[0]
    strongest_market_association = (
        market_associations[market_associations["segment"].eq("HISTORICAL_VALIDATION")]
        .assign(abs_correlation=lambda frame: frame["spearman_correlation"].abs())
        .sort_values(
            ["abs_correlation", "market_state_variable", "outcome"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .iloc[0]
    )
    decisions = [
        dict(h1_decision),
        dict(h2_decision),
        dict(h3_registered_decision),
        dict(h3_nested_decision),
    ]
    for row in decisions:
        if row["status"] not in STATUS_VALUES:
            raise Stage2Error(f"unknown decision status: {row['status']}")
    return {
        "study_id": spec["study_id"],
        "mode": spec["mode"],
        "study_type": spec["study_type"],
        "hypothesis_decisions": decisions,
        "five_endpoint_descriptive_multiplicity": endpoints,
        "portfolio_path_evidence": {
            "fc_minus_f_validation_net_weekly_return": fc_minus_f["mean"],
            "fc_minus_f_validation_hac_t": fc_minus_f["hac_t"],
            "cost_assumption": "buy_15bp_sell_25bp_fixed_50_slots",
        },
        "market_state_mechanism": {
            "strongest_fixed_validation_association": {
                "variable": strongest_market_association["market_state_variable"],
                "outcome": strongest_market_association["outcome"],
                "spearman": strongest_market_association["spearman_correlation"],
                "weeks": strongest_market_association["n_weeks"],
            },
            "action": "HYPOTHESIS_GENERATION_ONLY_DO_NOT_SELECT_FILTER",
        },
        "next_actions": [
            {
                "priority": 1,
                "action": "STOP_SAME_SAMPLE_STATE_AND_FACTOR_PARAMETER_OPTIMIZATION",
                "reason": "H1/H2 and fixed-slot path evidence do not support a stable positive increment.",
            },
            {
                "priority": 2,
                "action": "DO_NOT_PROMOTE_STAGE1_FAILURE_FEATURES_TO_FILTERS",
                "reason": "The registered H3 test is post-selection invalid; the nested diagnostic is not confirmation.",
            },
            {
                "priority": 3,
                "action": "COLLECT_AT_LEAST_26_NEW_IMMUTABLE_PRIMARY_WEEKS_BEFORE_ANY_PILOT_DECISION",
                "reason": "The current cache contains fewer than 26 genuinely unused complete primary weeks.",
            },
            {
                "priority": 4,
                "action": "REGISTER_ONE_FIXED_MARKET_STATE_MECHANISM_TEST_ONLY_AFTER_FRESH_DATA_EXISTS",
                "reason": "The two fixed market-state associations are retrospective and cannot justify a filter now.",
            },
        ],
        "confirmation_chain": "NOT_STARTED",
        "live_trading_authorized": False,
    }


def _fmt_pct(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    spec: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    h1_common_summary: pd.DataFrame,
    h1_control_summary: pd.DataFrame,
    h2_summary: pd.DataFrame,
    h3_registered: pd.DataFrame,
    h3_nested: pd.DataFrame,
    arm_summary: pd.DataFrame,
    recursive_summary: pd.DataFrame,
    placebo: pd.DataFrame,
    market_associations: pd.DataFrame,
    decision_matrix: Mapping[str, Any],
    path: Path,
) -> None:
    """Write the human-readable Stage 2 report."""

    h1_rows = h1_common_summary[h1_common_summary["segment"].isin(["DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"])]
    h2_rows = h2_summary[
        h2_summary["component"].eq("unexplained_residual")
        & h2_summary["segment"].isin(["DEVELOPMENT", "HISTORICAL_VALIDATION", "V1", "V2"])
    ]
    registered_rows = h3_registered[h3_registered["segment"].eq("HISTORICAL_VALIDATION")]
    nested_rows = h3_nested[h3_nested["segment"].isin(["DEVELOPMENT", "HISTORICAL_VALIDATION"])]
    path_rows = arm_summary[
        arm_summary["metric"].isin(["net_return_40bps", "paired_net_return_40bps"])
        & arm_summary["segment"].eq("HISTORICAL_VALIDATION")
    ]
    valid_placebos = placebo[
        placebo["segment"].eq("HISTORICAL_VALIDATION") & placebo["minimum_event_gate_passed"]
    ].sort_values(["view", "mean"], ascending=[True, False], kind="mergesort")
    association = market_associations[market_associations["segment"].eq("HISTORICAL_VALIDATION")]
    lines = [
        f"# 横截面因子 × 缠论 Stage 2 历史证伪压力测试（{spec['study_id']}）",
        "",
        "## 结论边界",
        "",
        "Stage 1 已经看过全部 2022–2026 样本。本报告是历史复用的证伪压力测试，不是独立 OOS，"
        "不修改 V2.1、不启动确认链，也不授权实盘。完成前独立语义审计已经看过部分 Stage 2 指标；"
        "该事实已写入机器规格。",
        "",
        f"- Stage 1 决策周：{input_audit['decision_dates']['rows']}",
        f"- 真正未使用且 5-session 标签完整的周：{input_audit['fresh_sample_audit']['unused_primary_complete_weeks']}",
        f"- 真正未使用且 20-session 诊断完整的周："
        f"{input_audit['fresh_sample_audit']['unused_20_session_diagnostic_complete_weeks']}",
        "- Stage 1 语义状态：`SUPERSEDED_IN_PART`",
        "",
        "## H1：状态 7 相对 SMA20 的共同周配对 cohort 对照",
        "",
        "| estimand | segment | 周差 | HAC t | 周数 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in h1_rows.itertuples(index=False):
        lines.append(
            f"| {row.estimand} | {row.segment} | {_fmt_pct(row.mean)} | {_fmt_num(row.hac_t)} | {int(row.n_weeks)} |"
        )
    lines.extend(
        [
            "",
            "这里的 `common_support` 只表示两组都存在的共同决策周；它不是名称级共同支持，"
            "两个 cohort 也可能重叠，因此不得解释为缠论状态的因果增量。",
            "",
            "| 对照 | 验证段增量 | V1 | V2 | 匹配率 | 数据门 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in h1_control_summary.itertuples(index=False):
        lines.append(
            f"| {row.control} | {_fmt_pct(row.validation_effect)} | {_fmt_pct(row.v1_effect)} | "
            f"{_fmt_pct(row.v2_effect)} | {_fmt_pct(row.validation_match_rate)} | "
            f"{'通过' if bool(row.endpoint_sufficient) else '不足，不解释效应/p值'} |"
        )
    h1_decision = next(
        row
        for row in decision_matrix["hypothesis_decisions"]
        if row["hypothesis_id"] == "H1_STATE7_HAS_CHAN_SPECIFIC_INCREMENT"
    )
    lines.extend(["", f"判定：`{h1_decision['status']}`。{h1_decision['reason']}", ""])

    lines.extend(
        [
            "## H2：风险归因后的因子残差",
            "",
            "| segment | 残差周均 | HAC t | 95% block CI |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in h2_rows.itertuples(index=False):
        lines.append(
            f"| {row.segment} | {_fmt_pct(row.mean)} | {_fmt_num(row.hac_t)} | "
            f"[{_fmt_pct(row.bootstrap_ci_low)}, {_fmt_pct(row.bootstrap_ci_high)}] |"
        )
    h2_decision = next(
        row
        for row in decision_matrix["hypothesis_decisions"]
        if row["hypothesis_id"] == "H2_FACTOR_HAS_RESIDUAL_AFTER_RISK_ATTRIBUTION"
    )
    lines.extend(
        [
            "",
            f"判定：`{h2_decision['status']}`。{h2_decision['reason']}",
            "",
            "原 Stage 1 的 `market` 标签在这里改称 `cross_sectional_common_return`；"
            "它是横截面共同收益，不是指数 beta。",
            "",
            "## H3：失败画像",
            "",
            "原登记字段使用全样本选择，因此状态固定为 `INVALID_POST_SELECTION`。它们只保留为诊断：",
            "",
            "| 字段 | validation SMD |",
            "| --- | ---: |",
        ]
    )
    for row in registered_rows.itertuples(index=False):
        lines.append(f"| {row.feature} | {_fmt_num(row.standardized_difference)} |")
    lines.extend(
        [
            "",
            "另做 early-only 选择、validation-only 评价的嵌套诊断：",
            "",
            "| 字段 | segment | SMD |",
            "| --- | --- | ---: |",
        ]
    )
    for row in nested_rows.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.segment} | {_fmt_num(row.standardized_difference)} |")
    h3_nested_decision = next(
        row
        for row in decision_matrix["hypothesis_decisions"]
        if row["hypothesis_id"] == "H3_NESTED_FAILURE_PROFILE_DIAGNOSTIC"
    )
    lines.extend(
        [
            "",
            f"嵌套诊断：`{h3_nested_decision['status']}`。{h3_nested_decision['reason']}",
            "",
            "## 固定 50 槽、现金与成本",
            "",
            "| 路径 | validation 周均 | HAC t | 平均槽位 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in path_rows.itertuples(index=False):
        lines.append(f"| {row.path} | {_fmt_pct(row.mean)} | {_fmt_num(row.hac_t)} | {_fmt_num(row.mean_slots, 1)} |")
    recursive = recursive_summary.iloc[0]
    lines.extend(
        [
            "",
            f"递归随机缓冲路径：实际 FC 减随机空值为 "
            f"{_fmt_pct(recursive['actual_minus_null'])}/周；这次随机持仓会反馈到后续保留和提案。"
            "该比较包含状态身份与填仓/暴露变化的总机制，不能单独解释为纯 identity alpha。",
            "",
            "## 安慰剂状态与市场状态",
            "",
            f"达到 100 事件门槛的状态/视图单元共 {len(valid_placebos)} 个；全部保留，未选择最好状态。"
            "这些是有状态周的原始 cohort 均值，不是 125 周含现金的固定槽路径，也不是风险调整增量。",
            "两个固定市场变量只用于机制生成。验证段关联如下：",
            "",
            "| 市场变量 | 结果 | Spearman | 周数 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in association.itertuples(index=False):
        lines.append(
            f"| {row.market_state_variable} | {row.outcome} | "
            f"{_fmt_num(row.spearman_correlation)} | {int(row.n_weeks)} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
        ]
    )
    for action in decision_matrix["next_actions"]:
        lines.append(f"{action['priority']}. `{action['action']}`：{action['reason']}")
    lines.extend(
        [
            "",
            "## 仍然存在的限制",
            "",
            "- qfq 行情和状态缓存受未来公司行为污染。",
            "- 没有官方停复牌、集合竞价、完整公司行为、退市终值与正式账户会计。",
            "- 成本路径是固定代理，不是可成交证明。",
            "- 所有 p 值和区间都只是历史描述；Holm 调整不把本研究变成确认。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def output_inventory(directory: Path) -> list[dict[str, Any]]:
    """Inventory payloads, excluding the manifest and its detached digest."""

    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in {"data_manifest.json", "data_manifest.sha256"}:
            continue
        row: dict[str, Any] = {
            "path": relative,
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "type": path.suffix.removeprefix(".") or "file",
        }
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            row["rows"] = int(parquet.metadata.num_rows)
            row["schema"] = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
        rows.append(row)
    return rows


def write_manifest_with_digest(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest plus a detached digest that detects manifest mutation."""

    manifest_path = directory / "data_manifest.json"
    write_json(manifest_path, manifest)
    (directory / "data_manifest.sha256").write_text(f"{sha256_file(manifest_path)}\n", encoding="ascii")


def runtime_manifest() -> dict[str, Any]:
    """Record the research runtime without importing the public CZSC API."""

    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "polars", "statsmodels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def verify_published_directory(
    path: Path,
    *,
    require_identity_name: bool = True,
    verify_current_identity: bool = True,
    spec: Mapping[str, Any] | None = None,
    spec_path: Path = SPEC_PATH,
    source_path: Path = SOURCE_PATH,
) -> dict[str, Any]:
    """Verify payload integrity and classify the identity against the current study."""

    manifest_path = path / "data_manifest.json"
    if not manifest_path.is_file():
        raise Stage2Error(f"missing Stage 2 manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    identity = str(manifest.get("study_identity", ""))
    if require_identity_name and path.name != f"STAGE2_{identity}":
        raise Stage2Error("published directory name does not match manifest identity")

    lifecycle_status = "NOT_CHECKED"
    current_spec = spec
    if verify_current_identity:
        current_spec = current_spec or load_and_validate_spec(spec_path)
        current_identity = study_identity(current_spec, source_path)
        superseded = {
            str(row.get("superseded_identity"))
            for row in current_spec.get("revision_history", [])
            if isinstance(row, Mapping) and row.get("superseded_identity")
        }
        if identity == current_identity:
            lifecycle_status = "CURRENT"
        elif identity in superseded:
            lifecycle_status = "SUPERSEDED"
        else:
            raise Stage2Error(
                f"published identity is neither current nor registered as superseded: "
                f"published={identity} current={current_identity}"
            )

    digest_path = path / "data_manifest.sha256"
    if digest_path.is_file():
        expected_manifest_digest = digest_path.read_text(encoding="ascii").strip()
        if len(expected_manifest_digest) != 64 or sha256_file(manifest_path) != expected_manifest_digest:
            raise Stage2Error("published manifest digest changed")
        manifest_digest_status = "VERIFIED"
    elif lifecycle_status == "CURRENT":
        raise Stage2Error("current published identity is missing its detached manifest digest")
    else:
        manifest_digest_status = "LEGACY_NOT_AVAILABLE"

    if lifecycle_status == "CURRENT":
        assert current_spec is not None
        expected_spec_physical = sha256_file(spec_path)
        expected_spec_canonical = hashlib.sha256(canonical_json(current_spec)).hexdigest()
        expected_source = sha256_file(source_path)
        if manifest.get("study_id") != current_spec.get("study_id"):
            raise Stage2Error("current manifest study ID differs from the current spec")
        if manifest.get("mode") != current_spec.get("mode") or manifest.get("study_type") != current_spec.get(
            "study_type"
        ):
            raise Stage2Error("current manifest study mode differs from the current spec")
        if manifest.get("spec", {}).get("physical_sha256") != expected_spec_physical:
            raise Stage2Error("current manifest physical spec digest differs from the current spec")
        if manifest.get("spec", {}).get("canonical_sha256") != expected_spec_canonical:
            raise Stage2Error("current manifest canonical spec digest differs from the current spec")
        if manifest.get("source", {}).get("sha256") != expected_source:
            raise Stage2Error("current manifest source digest differs from the current source")
        expected_frozen_inputs = [
            {
                "name": row["name"],
                "path": row["path"],
                "sha256": row["sha256"],
                "size": int((REPO_ROOT / row["path"]).stat().st_size),
            }
            for row in current_spec["frozen_stage1_inputs"]
        ]
        if manifest.get("frozen_inputs") != expected_frozen_inputs:
            raise Stage2Error("current manifest frozen-input declarations differ from the current spec")
        for row in expected_frozen_inputs:
            input_path = REPO_ROOT / row["path"]
            if sha256_file(input_path) != row["sha256"] or int(input_path.stat().st_size) != int(row["size"]):
                raise Stage2Error(f"current frozen input differs from the manifest: {row['name']}")
        if manifest.get("confirmation_chain") != "NOT_STARTED" or manifest.get("live_trading_authorized") is not False:
            raise Stage2Error("current manifest changed confirmation or live-trading governance")

    expected_rows = manifest.get("output_inventory")
    if not isinstance(expected_rows, list):
        raise Stage2Error("manifest output inventory is missing")
    expected_paths = {str(row["path"]) for row in expected_rows}
    if len(expected_paths) != len(expected_rows):
        raise Stage2Error("manifest output inventory contains duplicate paths")
    actual_paths = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name not in {"data_manifest.json", "data_manifest.sha256"}
    }
    if actual_paths != expected_paths:
        raise Stage2Error(
            f"published output set changed: missing={sorted(expected_paths - actual_paths)} "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    for row in expected_rows:
        file_path = path / str(row["path"])
        if int(file_path.stat().st_size) != int(row["size"]):
            raise Stage2Error(f"output size changed: {row['path']}")
        if sha256_file(file_path) != row["sha256"]:
            raise Stage2Error(f"output digest changed: {row['path']}")
        if file_path.suffix == ".parquet":
            parquet = pq.ParquetFile(file_path)
            if int(parquet.metadata.num_rows) != int(row["rows"]):
                raise Stage2Error(f"output parquet row count changed: {row['path']}")
            schema = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
            if schema != row["schema"]:
                raise Stage2Error(f"output parquet schema changed: {row['path']}")
    registry = read_json(path / "experiment_registry.json")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or [row.get("experiment_id") for row in experiments] != list(
        EXPECTED_EXPERIMENT_IDS
    ):
        raise Stage2Error("published experiment registry differs from the frozen plan")
    if any(row.get("status") != "COMPLETED" for row in experiments):
        raise Stage2Error("published experiment registry is not fully completed")
    if manifest.get("experiment_registry") != experiments:
        raise Stage2Error("manifest experiment registry differs from the published registry")
    decision_matrix = read_json(path / "decision_matrix.json")
    if manifest.get("hypothesis_decisions") != decision_matrix.get("hypothesis_decisions"):
        raise Stage2Error("manifest decisions differ from the published decision matrix")
    status = {
        "CURRENT": "VERIFIED_CURRENT",
        "SUPERSEDED": "VERIFIED_SUPERSEDED",
        "NOT_CHECKED": "VERIFIED_INTEGRITY_ONLY",
    }[lifecycle_status]
    return {
        "study_id": manifest["study_id"],
        "study_identity": identity,
        "path": str(path),
        "files_verified": int(len(expected_rows) + 1 + int(digest_path.is_file())),
        "status": status,
        "integrity_status": "VERIFIED",
        "lifecycle_status": lifecycle_status,
        "manifest_digest_status": manifest_digest_status,
        "confirmation_chain": manifest["confirmation_chain"],
    }


def _failure_directory(output_root: Path, identity: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return output_root / f"FAILED_{identity}_{stamp}"


def run_study(
    spec: Mapping[str, Any],
    *,
    output_root: Path = OUTPUT_ROOT,
    spec_path: Path = SPEC_PATH,
    source_path: Path = SOURCE_PATH,
) -> Path:
    """Run all experiments in a temporary directory and atomically publish once."""

    identity = study_identity(spec, source_path)
    final_path = output_root / f"{spec['output_integrity']['directory_prefix']}{identity}"
    if final_path.exists():
        raise Stage2Error(f"refusing to overwrite published Stage 2 identity: {final_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".tmp_{identity}_", dir=output_root))
    records = {
        row["id"]: ExperimentRecord(experiment_id=row["id"], deliverable=row["deliverable"])
        for row in spec["planned_experiments"]
    }
    write_registry(spec, records, work_dir)
    state: dict[str, Any] = {}
    try:

        def e00() -> dict[str, Any]:
            bundle = load_input_bundle(spec)
            audit = run_input_audit(spec, bundle)
            state["bundle"] = bundle
            write_json(work_dir / "input_audit.json", audit)
            return audit

        input_audit = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E00_INPUT_AND_INTEGRITY",
            e00,
            lambda result: {"audit_records": 1, "input_tables": len(result["input_rows"])},
        )
        bundle: InputBundle = state["bundle"]

        def e01() -> tuple[pd.DataFrame, pd.DataFrame]:
            result = run_h1_common_support(spec, bundle)
            result[0].to_parquet(work_dir / "h1_common_support_weekly.parquet", index=False)
            result[1].to_parquet(work_dir / "h1_common_support_summary.parquet", index=False)
            return result

        h1_common_weekly, h1_common_summary = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E01_H1_COMMON_SUPPORT",
            e01,
            lambda result: {"weekly_rows": len(result[0]), "summary_rows": len(result[1])},
        )

        def e02() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
            result = run_h1_controls(spec, bundle, h1_common_weekly, h1_common_summary)
            result[0].to_parquet(work_dir / "h1_control_summary.parquet", index=False)
            result[1].to_parquet(work_dir / "h1_matched_distribution.parquet", index=False)
            result[2].to_parquet(work_dir / "h1_randomized_distribution.parquet", index=False)
            write_json(work_dir / "h1_decision.json", result[3])
            return result

        h1_control_summary, h1_matched_distribution, h1_random_distribution, h1_decision = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E02_H1_CONTROLS",
            e02,
            lambda result: {
                "summary_rows": len(result[0]),
                "matched_repetitions": len(result[1]),
                "randomized_repetitions": len(result[2]),
                "decisions": 1,
            },
        )

        def e03() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
            result = run_h2_residual_stability(spec, bundle)
            result[0].to_parquet(work_dir / "h2_attribution_weekly.parquet", index=False)
            result[1].to_parquet(work_dir / "h2_attribution_summary.parquet", index=False)
            result[2].to_parquet(work_dir / "h2_residual_surface.parquet", index=False)
            write_json(work_dir / "h2_decision.json", result[3])
            return result

        h2_weekly, h2_summary, residual_surface, h2_decision = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E03_H2_RESIDUAL_STABILITY",
            e03,
            lambda result: {
                "weekly_rows": len(result[0]),
                "summary_rows": len(result[1]),
                "residual_rows": len(result[2]),
                "decisions": 1,
            },
        )

        def e04() -> tuple[pd.DataFrame, pd.DataFrame]:
            result = run_h2_random_identity(spec, residual_surface, h2_weekly)
            result[0].to_parquet(work_dir / "h2_random_identity_summary.parquet", index=False)
            result[1].to_parquet(work_dir / "h2_random_identity_distribution.parquet", index=False)
            return result

        h2_random_summary, h2_random_distribution = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E04_H2_RANDOM_IDENTITY",
            e04,
            lambda result: {"summary_rows": len(result[0]), "repetitions": len(result[1])},
        )

        def e05() -> tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            dict[str, Any],
            dict[str, Any],
        ]:
            result = run_h3_profile_replication(spec, bundle)
            result[0].to_parquet(work_dir / "h3_registered_profiles.parquet", index=False)
            result[1].to_parquet(work_dir / "h3_development_candidates.parquet", index=False)
            result[2].to_parquet(work_dir / "h3_nested_profiles.parquet", index=False)
            result[3].to_parquet(work_dir / "h3_tail_lifts.parquet", index=False)
            write_json(work_dir / "h3_registered_decision.json", result[4])
            write_json(work_dir / "h3_nested_decision.json", result[5])
            return result

        (
            h3_registered,
            h3_development_candidates,
            h3_nested,
            h3_tail_lifts,
            h3_registered_decision,
            h3_nested_decision,
        ) = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E05_H3_PROFILE_REPLICATION",
            e05,
            lambda result: {
                "registered_rows": len(result[0]),
                "development_candidates": len(result[1]),
                "nested_rows": len(result[2]),
                "tail_lifts": len(result[3]),
                "decisions": 2,
            },
        )

        def e06() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            result = run_h3_controls_and_secondary(
                spec,
                bundle,
                h3_nested,
                h3_development_candidates,
            )
            result[0].to_parquet(work_dir / "h3_permutation_summary.parquet", index=False)
            result[1].to_parquet(work_dir / "h3_permutation_distribution.parquet", index=False)
            result[2].to_parquet(work_dir / "h3_noise_profile.parquet", index=False)
            result[3].to_parquet(work_dir / "h3_secondary_failure_profiles.parquet", index=False)
            return result

        h3_permutation_summary, h3_permutation, h3_noise, h3_secondary = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E06_H3_CONTROLS_AND_SECONDARY",
            e06,
            lambda result: {
                "permutation_summary_rows": len(result[0]),
                "permutation_rows": len(result[1]),
                "noise_rows": len(result[2]),
                "secondary_rows": len(result[3]),
            },
        )

        def e07() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            result = run_path_stress(spec, bundle)
            result[0].to_parquet(work_dir / "path_weekly.parquet", index=False)
            result[1].to_parquet(work_dir / "path_summary.parquet", index=False)
            result[2].to_parquet(work_dir / "recursive_random_path_summary.parquet", index=False)
            result[3].to_parquet(work_dir / "recursive_random_path_distribution.parquet", index=False)
            return result

        arm_weekly, arm_summary, recursive_summary, recursive_distribution = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E07_FIXED_SLOT_PATH_STRESS",
            e07,
            lambda result: {
                "weekly_rows": len(result[0]),
                "summary_rows": len(result[1]),
                "recursive_summary_rows": len(result[2]),
                "recursive_repetitions": len(result[3]),
            },
        )

        def e08() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            result = run_placebo_and_market_state(
                spec,
                bundle,
                h1_common_weekly,
                arm_weekly,
            )
            result[0].to_parquet(work_dir / "state_placebos.parquet", index=False)
            result[1].to_parquet(work_dir / "market_state_weekly.parquet", index=False)
            result[2].to_parquet(work_dir / "market_state_summary.parquet", index=False)
            result[3].to_parquet(work_dir / "market_state_associations.parquet", index=False)
            return result

        state_placebos, market_weekly, market_summary, market_associations = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E08_PLACEBO_AND_MARKET_STATE",
            e08,
            lambda result: {
                "placebo_rows": len(result[0]),
                "market_weekly_rows": len(result[1]),
                "market_summary_rows": len(result[2]),
                "market_association_rows": len(result[3]),
            },
        )

        def e09() -> dict[str, Any]:
            decision = derive_decision_matrix(
                spec,
                h1_decision,
                h2_decision,
                h3_registered_decision,
                h3_nested_decision,
                h1_common_summary,
                h1_control_summary,
                h2_summary,
                h3_permutation_summary,
                arm_summary,
                market_associations,
            )
            write_json(work_dir / "decision_matrix.json", decision)
            summary = {
                "study_id": spec["study_id"],
                "mode": spec["mode"],
                "study_type": spec["study_type"],
                "input_audit": input_audit,
                "decision_matrix": decision,
                "h1_common_support": h1_common_summary.to_dict("records"),
                "h1_controls": h1_control_summary.to_dict("records"),
                "h2_attribution": h2_summary.to_dict("records"),
                "h2_random_identity": h2_random_summary.to_dict("records"),
                "h3_registered_profiles": h3_registered.to_dict("records"),
                "h3_nested_profiles": h3_nested.to_dict("records"),
                "h3_tail_lifts": h3_tail_lifts.to_dict("records"),
                "h3_permutation": h3_permutation_summary.to_dict("records"),
                "path_summary": arm_summary.to_dict("records"),
                "recursive_random_path": recursive_summary.to_dict("records"),
                "market_state_associations": market_associations.to_dict("records"),
                "confirmation_chain": "NOT_STARTED",
            }
            write_json(work_dir / "summary.json", summary)
            write_report(
                spec,
                input_audit,
                h1_common_summary,
                h1_control_summary,
                h2_summary,
                h3_registered,
                h3_nested,
                arm_summary,
                recursive_summary,
                state_placebos,
                market_associations,
                decision,
                work_dir / "report.md",
            )
            return decision

        decision_matrix = execute_experiment(
            spec,
            records,
            work_dir,
            "S2E09_DECISION_MATRIX",
            e09,
            lambda result: {
                "hypothesis_decisions": len(result["hypothesis_decisions"]),
                "next_actions": len(result["next_actions"]),
            },
        )
        if any(record.status != "COMPLETED" for record in records.values()):
            raise Stage2Error("not every Stage 2 experiment completed")
        inventory = output_inventory(work_dir)
        manifest = {
            "study_id": spec["study_id"],
            "study_identity": identity,
            "mode": spec["mode"],
            "study_type": spec["study_type"],
            "confirmation_chain": "NOT_STARTED",
            "completed_at_utc": utc_now(),
            "spec": {
                "path": str(spec_path),
                "physical_sha256": sha256_file(spec_path),
                "canonical_sha256": hashlib.sha256(canonical_json(spec)).hexdigest(),
            },
            "source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "frozen_inputs": [
                {
                    "name": row["name"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "size": int((REPO_ROOT / row["path"]).stat().st_size),
                }
                for row in spec["frozen_stage1_inputs"]
            ],
            "runtime": runtime_manifest(),
            "input_audit": input_audit,
            "experiment_registry": [records[key].as_dict() for key in EXPECTED_EXPERIMENT_IDS],
            "hypothesis_decisions": decision_matrix["hypothesis_decisions"],
            "output_inventory": inventory,
            "limitations": [
                "full_stage1_sample_already_seen",
                "pre_protocol_audit_peeked_stage2_metrics",
                "contaminated_qfq_history",
                "contaminated_state_cache",
                "no_official_suspension_or_open_auction_evidence",
                "no_complete_corporate_action_or_delist_terminal_evidence",
                "fixed_cost_and_cash_execution_proxy_only",
            ],
            "live_trading_authorized": False,
        }
        write_manifest_with_digest(work_dir, manifest)
        verify_published_directory(
            work_dir,
            require_identity_name=False,
            spec=spec,
            spec_path=spec_path,
            source_path=source_path,
        )
        if final_path.exists():
            raise Stage2Error(f"publication race: final identity already exists: {final_path}")
        os.rename(work_dir, final_path)
        verify_published_directory(
            final_path,
            spec=spec,
            spec_path=spec_path,
            source_path=source_path,
        )
        return final_path
    except Exception:
        if work_dir.exists():
            failed_path = _failure_directory(output_root, identity)
            os.rename(work_dir, failed_path)
            print(f"[stage2] failed evidence preserved at {failed_path}", file=sys.stderr, flush=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate frozen inputs, schemas, splits and fresh-sample coverage")
    subparsers.add_parser("run", help="Run once and atomically publish a content-addressed result")
    verify = subparsers.add_parser("verify", help="Verify a published result from its manifest")
    verify.add_argument("--path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_and_validate_spec(args.spec)
    if args.command == "check":
        bundle = load_input_bundle(spec)
        audit = run_input_audit(spec, bundle)
        print(
            json.dumps(
                {
                    "study_id": spec["study_id"],
                    "study_identity": study_identity(spec),
                    "output_path": str(output_path_for(spec, args.output_root)),
                    "output_exists": output_path_for(spec, args.output_root).exists(),
                    "audit": audit,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        path = run_study(spec, output_root=args.output_root, spec_path=args.spec)
        print(
            json.dumps(
                verify_published_directory(path, spec=spec, spec_path=args.spec),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    verify_path = args.path or output_path_for(spec, args.output_root)
    print(
        json.dumps(
            verify_published_directory(verify_path, spec=spec, spec_path=args.spec),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
