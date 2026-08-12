"""Freeze the outcome-blind production delay5 point-in-time market-cap design.

This planner never downloads vendor data and never evaluates realised returns.  It
rebuilds the compact request identities from the bound common-horizon artifacts,
checks every upstream file identity, and compares the result with the tracked plan.

    uv run --no-sync python scripts/delay5_pit_exact_mcap_plan.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXPECTED_PLAN_PATH = SCRIPT_DIR / "delay5_pit_exact_mcap_plan_2026-08-12.json"

COMMON_DIR = SCRIPT_DIR / "_output" / "delay5_common_horizon_att"
COMMON_AUDIT_PATH = COMMON_DIR / "audit.json"
REQUEST_PATH = COMMON_DIR / "exact_mcap_request_manifest.json"
TREATED_PATH = COMMON_DIR / "treated_common_support.parquet"
PRODUCTION_DIR = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort"

ARTIFACT_PATHS = {
    "common_audit": COMMON_AUDIT_PATH,
    "exact_mcap_request_manifest": REQUEST_PATH,
    "treated_common_support": TREATED_PATH,
    "production_audit": PRODUCTION_DIR / "audit.json",
    "production_cohort": PRODUCTION_DIR / "cohort.parquet",
    "candidates": SCRIPT_DIR / "_output" / "surge_candidates" / "candidates.parquet",
    "candidate_manifest": SCRIPT_DIR / "_output" / "surge_candidates" / "manifest.json",
    "panel": SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet",
    "market_state": SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet",
    "namechange": Path.home() / ".ts_data_cache" / "namechange.parquet",
}

SCRIPT_PATHS = {
    "common_horizon": SCRIPT_DIR / "delay5_common_horizon_att.py",
    "production_cohort": SCRIPT_DIR / "surge_delay5_production_cohort_audit.py",
    "candidate_dump": SCRIPT_DIR / "surge_candidates_dump.py",
    "market_state": SCRIPT_DIR / "surge_market_state_filter.py",
    "industry_snapshot": SCRIPT_DIR / "s2b_industry_size_proxy_audit.py",
}

INDUSTRY_ANCHORS = {
    2021: "20210701",
    2022: "20220104",
    2023: "20230103",
    2024: "20240102",
    2025: "20250102",
    2026: "20260105",
}
INDUSTRY_COLUMNS = ("symbol", "code", "updateDate", "industry", "industryClassification")
CANONICALIZATION = (
    "top-level JSON array; positional records; lexicographic sort; json.dumps ensure_ascii=False "
    "separators=(',', ':') allow_nan=False; UTF-8; no trailing newline"
)
FORBIDDEN_RESULT_KEYS = {
    "att",
    "att_results",
    "confidence_intervals",
    "effect_estimates",
    "outcomes",
    "p_values",
    "realised_returns",
    "realized_returns",
    "return_values",
    "summaries",
    "t_statistics",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 and fail when the input is absent."""

    if not path.is_file():
        raise FileNotFoundError(f"required plan input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload(records: Sequence[Any]) -> bytes:
    """Encode a sorted compact JSON array without a trailing newline."""

    return json.dumps(
        sorted(records),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_identity(records: Sequence[Any]) -> dict[str, Any]:
    """Return the compact identity used by the request producer."""

    ordered = sorted(records)
    payload = canonical_payload(ordered)
    return {
        "count": len(ordered),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first3": ordered[:3],
        "last3": ordered[-3:] if ordered else [],
    }


def _repo_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        if resolved == (Path.home() / ".ts_data_cache" / "namechange.parquet").resolve():
            return "$HOME/.ts_data_cache/namechange.parquet"
        raise RuntimeError(f"plan input has no portable locator: {path}") from None


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _industry_path(year: int) -> Path:
    anchor = INDUSTRY_ANCHORS[year]
    return (
        SCRIPT_DIR
        / "_output"
        / "s2b_industry_size_proxy"
        / "reference"
        / (f"baostock_industry_{year}_{anchor}.parquet")
    )


def canonical_industry_sha256(frame: pd.DataFrame) -> str:
    """Hash industry content independently of Parquet writer metadata."""

    if missing := set(INDUSTRY_COLUMNS) - set(frame):
        raise RuntimeError(f"industry snapshot lacks canonical columns: {sorted(missing)}")
    canonical = frame.loc[:, INDUSTRY_COLUMNS].fillna("").astype(str)
    canonical = canonical.sort_values(list(INDUSTRY_COLUMNS), kind="mergesort")
    return hashlib.sha256(canonical_payload(canonical.to_numpy().tolist())).hexdigest()


class _HashTextWriter:
    """把确定性 CSV 文本直接流入 SHA256，避免大型中间文件。"""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, value: str) -> int:
        payload = value.encode("utf-8")
        self.digest.update(payload)
        self.bytes += len(payload)
        return len(value)


def canonical_frame_identity(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
    date_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """对因果投影做 Parquet writer 与 row-group 无关的内容身份。"""

    if missing := set(columns) - set(frame):
        raise RuntimeError(f"canonical frame lacks columns: {sorted(missing)}")
    scoped = frame.loc[:, list(columns)].copy()
    for column in date_columns:
        values = pd.to_datetime(scoped[column], errors="coerce")
        scoped[column] = values.dt.strftime("%Y-%m-%d").fillna("<NA>")
    scoped = scoped.sort_values(list(sort_by), kind="mergesort", na_position="last").reset_index(drop=True)
    writer = _HashTextWriter()
    scoped.to_csv(
        writer,
        index=False,
        header=True,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    )
    return {
        "schema": "sorted CSV UTF-8; header; LF; <NA>; float %.17g",
        "columns": list(columns),
        "sort_by": list(sort_by),
        "rows": int(len(scoped)),
        "bytes": int(writer.bytes),
        "sha256": writer.digest.hexdigest(),
    }


def _production_causal_identities() -> dict[str, Any]:
    """从 production stage-fill 与市场日历重建结果盲身份。"""

    cohort = pd.read_parquet(
        ARTIFACT_PATHS["production_cohort"], columns=["symbol", "dec_dt", "entry_dt", "stage_fill"]
    )
    market = pd.read_parquet(ARTIFACT_PATHS["market_state"], columns=["dt"])
    calendar = pd.DatetimeIndex(pd.to_datetime(market["dt"])).normalize().sort_values().unique()
    scoped = cohort[cohort["stage_fill"].astype(bool)].loc[:, ["symbol", "dec_dt", "entry_dt"]].copy()
    for column in ("dec_dt", "entry_dt"):
        scoped[column] = pd.to_datetime(scoped[column]).dt.normalize()
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    scoped = scoped[scoped["entry_dt"].eq(scoped["dec_dt"].map(next_session))].copy()
    all_filled_records = [
        [str(row.symbol), row.dec_dt.strftime("%Y-%m-%d"), row.entry_dt.strftime("%Y-%m-%d")]
        for row in scoped.itertuples(index=False)
    ]
    positions = pd.Series(range(len(calendar)), index=calendar)
    scoped["entry_position"] = scoped["entry_dt"].map(positions)
    mature = scoped[scoped["entry_position"].notna() & scoped["entry_position"].le(len(calendar) - 60)].copy()
    schedules: list[list[str]] = []
    for row in mature.itertuples(index=False):
        position = int(row.entry_position)
        schedules.append(
            [
                str(row.symbol),
                row.dec_dt.strftime("%Y-%m-%d"),
                row.entry_dt.strftime("%Y-%m-%d"),
                pd.Timestamp(calendar[position + 4]).strftime("%Y-%m-%d"),
                pd.Timestamp(calendar[position + 19]).strftime("%Y-%m-%d"),
                pd.Timestamp(calendar[position + 59]).strftime("%Y-%m-%d"),
            ]
        )
    return {
        "all_filled_identity": payload_identity(all_filled_records),
        "common60_schedule_identity": payload_identity(schedules),
    }


def _reproducible_source_identity() -> dict[str, Any]:
    """构造 balance 实际读取字段的跨设备 canonical 身份。"""

    panel = pd.read_parquet(ARTIFACT_PATHS["panel"], columns=["symbol", "dt", "close", "amount_e"])
    market = pd.read_parquet(ARTIFACT_PATHS["market_state"], columns=["dt"])
    namechange = pd.read_parquet(ARTIFACT_PATHS["namechange"])
    return {
        "panel_causal_projection": canonical_frame_identity(
            panel,
            columns=["symbol", "dt", "close", "amount_e"],
            sort_by=["symbol", "dt"],
            date_columns=["dt"],
        ),
        "market_calendar": canonical_frame_identity(
            market,
            columns=["dt"],
            sort_by=["dt"],
            date_columns=["dt"],
        ),
        "historical_st": canonical_frame_identity(
            namechange,
            columns=["ts_code", "name", "start_date", "end_date"],
            sort_by=["ts_code", "start_date", "end_date", "name"],
        ),
        "production": _production_causal_identities(),
    }


def _require_identity(actual: Mapping[str, Any], recorded: Mapping[str, Any], label: str) -> None:
    for field in ("count", "bytes", "sha256", "first3", "last3"):
        if actual.get(field) != recorded.get(field):
            raise RuntimeError(f"{label} identity drift on {field}")


def _validate_request_manifest(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("schema") != "delay5_exact_mcap_request_manifest_v2":
        raise RuntimeError("exact-mcap request must use schema v2")
    if request.get("canonicalization") != (
        "JSON array of [symbol, YYYY-MM-DD] records; lexicographic sort; ensure_ascii=False; "
        "separators=(',', ':'); UTF-8; no trailing newline"
    ):
        raise RuntimeError("exact-mcap request canonicalization drift")
    request_records = request.get("request_records")
    treated_records = request.get("treated_request_records")
    dates = request.get("request_dates")
    symbols = request.get("request_symbols")
    if not all(isinstance(value, list) for value in (request_records, treated_records, dates, symbols)):
        raise RuntimeError("exact-mcap request lacks canonical record arrays")

    identities = {
        "request": payload_identity(request_records),
        "treated": payload_identity(treated_records),
        "dates": payload_identity(dates),
        "symbols": payload_identity(symbols),
    }
    recorded = {
        "request": request.get("request_identity", {}),
        "treated": request.get("treated_request_identity", {}),
        "dates": request.get("request_dates_identity", {}),
        "symbols": request.get("request_symbols_identity", {}),
    }
    for label in identities:
        _require_identity(identities[label], recorded[label], f"request {label}")

    request_keys = {tuple(row) for row in request_records}
    treated_keys = {tuple(row) for row in treated_records}
    closure = request.get("closure", {})
    if len(request_keys) != len(request_records) or len(treated_keys) != len(treated_records):
        raise RuntimeError("exact-mcap request keys are not unique")
    if not treated_keys.issubset(request_keys):
        raise RuntimeError("exact-mcap request omits treated keys")
    if not all(
        closure.get(field) is True for field in ("request_keys_unique", "treated_keys_unique", "all_treated_requested")
    ):
        raise RuntimeError("exact-mcap request producer did not close all gates")
    cohort = request.get("cohort", {})
    if cohort.get("treated_trades") != identities["treated"]["count"]:
        raise RuntimeError("request cohort treated count drift")
    if cohort.get("decision_dates") != identities["dates"]["count"]:
        raise RuntimeError("request cohort date count drift")
    return identities


def _treated_identity(expected_treated: Mapping[str, Any]) -> dict[str, Any]:
    columns = ["symbol", "dec_dt", "entry_dt", "h5_dt", "h20_dt", "h60_dt"]
    treated = pd.read_parquet(TREATED_PATH, columns=columns)
    if treated.duplicated(["symbol", "dec_dt"]).any():
        raise RuntimeError("treated common support is not unique by symbol and decision date")

    def date_text(value: Any) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    treated_pairs = [[row.symbol, date_text(row.dec_dt)] for row in treated.itertuples(index=False)]
    pair_identity = payload_identity(treated_pairs)
    _require_identity(pair_identity, expected_treated, "treated parquet/request")
    schedules = [
        [
            row.symbol,
            date_text(row.dec_dt),
            date_text(row.entry_dt),
            date_text(row.h5_dt),
            date_text(row.h20_dt),
            date_text(row.h60_dt),
        ]
        for row in treated.itertuples(index=False)
    ]
    return {
        "rows": len(treated),
        "decision_dates": int(pd.to_datetime(treated["dec_dt"]).nunique()),
        "trades_2024plus": int(pd.to_datetime(treated["dec_dt"]).dt.year.ge(2024).sum()),
        "schedule_identity": payload_identity(schedules),
    }


def _validate_common_bindings(common: Mapping[str, Any], request_identity: Mapping[str, Any]) -> None:
    if common.get("schema") != "delay5_common_horizon_att_audit_v1":
        raise RuntimeError("unexpected common-horizon audit schema")
    if common.get("verdict", {}).get("live_authorized") is not False:
        raise RuntimeError("common-horizon audit must not authorize live trading")
    for field in ("count", "bytes", "sha256"):
        if common.get("exact_mcap_request", {}).get(field) != request_identity.get(field):
            raise RuntimeError(f"common/request identity drift on {field}")
    output_records = common.get("outputs", {})
    expected_outputs = {
        "treated_common_support": TREATED_PATH,
        "exact_mcap_request_manifest": REQUEST_PATH,
    }
    for label, path in expected_outputs.items():
        record = output_records.get(label, {})
        if record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"common output binding drift: {label}")

    upstream = common.get("data_identity", {})
    mapping = {
        "candidates": "candidates",
        "candidate_manifest": "candidate_manifest",
        "panel": "panel",
        "market_state": "market_state",
        "namechange": "namechange",
        "production_cohort": "production_cohort",
    }
    for common_label, artifact_label in mapping.items():
        record = upstream.get(common_label, {})
        if record.get("sha256") != sha256_file(ARTIFACT_PATHS[artifact_label]):
            raise RuntimeError(f"common upstream binding drift: {common_label}")
    production = upstream.get("production_cohort", {})
    if production.get("upstream_schema") != "surge_delay5_production_cohort_audit_v3":
        raise RuntimeError("unexpected production cohort audit schema")
    if production.get("upstream_audit_sha256") != sha256_file(ARTIFACT_PATHS["production_audit"]):
        raise RuntimeError("production audit binding drift")

    industries = upstream.get("industry_snapshots", {})
    for year in INDUSTRY_ANCHORS:
        record = industries.get(str(year), {})
        path = _industry_path(year)
        if record.get("file_sha256") != sha256_file(path):
            raise RuntimeError(f"industry {year} file binding drift")
        frame = pd.read_parquet(path)
        if record.get("canonical_content_sha256") != canonical_industry_sha256(frame):
            raise RuntimeError(f"industry {year} canonical binding drift")


def _artifact_identities() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, path in ARTIFACT_PATHS.items():
        record: dict[str, Any] = {"locator": _repo_locator(path), "sha256": sha256_file(path)}
        if path.suffix == ".parquet":
            record["rows"] = _parquet_rows(path)
        result[label] = record
    return result


def portable_source_identity(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    """只保留跨设备可稳定复核的计划身份；物理 JSON/Parquet SHA 留作本机证据。"""

    industries = source_identity.get("industry_snapshots", {})
    portable_industries = {
        str(year): {
            "rows": record.get("rows"),
            "canonical_content_sha256": record.get("canonical_content_sha256"),
            "effective_dates": record.get("effective_dates"),
        }
        for year, record in industries.items()
    }
    return {
        "baseline_parent_commit": source_identity.get("baseline_parent_commit"),
        "scripts": source_identity.get("scripts"),
        "reproducible_identity": source_identity.get("reproducible_identity"),
        "industry_snapshots": portable_industries,
    }


def _industry_identities() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for year in INDUSTRY_ANCHORS:
        path = _industry_path(year)
        frame = pd.read_parquet(path)
        result[str(year)] = {
            "locator": _repo_locator(path),
            "rows": len(frame),
            "file_sha256": sha256_file(path),
            "canonical_content_sha256": canonical_industry_sha256(frame),
            "effective_dates": sorted(frame["updateDate"].fillna("").astype(str).unique().tolist()),
        }
    return result


def _design() -> dict[str, Any]:
    return {
        "outcome_blind": True,
        "data_source": {
            "provider": "Tushare/Tinyshare",
            "endpoint": "daily_basic",
            "query": "one full-market query per request date using trade_date=YYYYMMDD",
            "fields": [
                "ts_code",
                "trade_date",
                "close",
                "total_share",
                "float_share",
                "total_mv",
                "circ_mv",
            ],
            "exact_mcap_cny": "circ_mv * 10000",
            "close_times_float_share": "quality-control diagnostic only",
            "mix_sources_within_date": False,
            "fallback_for_missing_exact_mcap": False,
        },
        "cache_contract": {
            "root": "CZSC_DELAY5_EXACT_MCAP_CACHE or $HOME/.ts_data_cache/delay5_exact_mcap_v1",
            "object_scope": "complete daily_basic cross-section projected to the declared fields",
            "object_canonicalization": CANONICALIZATION,
            "content_addressed": True,
            "atomic_writes": True,
            "resume_requires_sha256_verification": True,
            "closure_gates": [
                "all 170 request dates have one verified object",
                "all rows belong to the queried trade date",
                "no duplicate ts_code/trade_date keys",
                "4000 <= full-market rows < 6000",
                "circ_mv is finite and positive for every requested key",
                "all 47431 requested keys and all 286 treated keys are materialized",
            ],
        },
        "matching": {
            "primary": {
                "name": "pit_exact_industry_mcap2_conditional_entropy_v1",
                "industry": "same frozen annual industry; missing treated industry is unsupported",
                "candidate_pool": (
                    "all controls with complete frozen causal balance features inside the inclusive exact "
                    "point-in-time circ_mv caliper"
                ),
                "caliper_ratio": 2.0,
                "min_controls": 5,
                "weighting": (
                    "conditional entropy tilting with one all-scope and one 2024plus-scope coefficient per balance "
                    "feature; weights normalize to one within each treated trade"
                ),
                "solver": {
                    "method": "scipy.optimize.least_squares",
                    "max_nfev": 2000,
                    "parameter_bounds": [-50.0, 50.0],
                    "initial_parameters": "all zeros",
                    "xtol_ftol_gtol": 1e-12,
                },
                "tie_break": "match_distance, symbol",
            },
            "causal_cutoff": "every feature and eligibility input has max timestamp <= decision date",
            "exclude": "treated symbol, every production-treated symbol on that date, historical ST",
            "outcome_blind_design_ledger": {
                "candidate_calipers": [1.5, 2.0, 3.0],
                "selection_rule": (
                    "choose the smallest caliper with both-scope coverage >=0.80, all frozen SMD gates passing, "
                    "per-trade ESS p05 >=5, max pair weight <=0.50 and global pair ESS/trade >=5"
                ),
                "selected_caliper": 2.0,
                "outcomes_loaded_during_selection": False,
                "note": "weighting was designed after exact covariates were available but before exact H5/H20/H60 outcomes",
            },
        },
        "balance_gate": {
            "evaluated_before_outcome_loading": True,
            "scopes": {"all": 286, "2024plus": 174},
            "features": [
                "log_exact_mcap",
                "ret5",
                "ret20",
                "ret60",
                "vol20",
                "vol60",
                "log_price",
                "log_amount",
                "liq20",
            ],
            "smd_thresholds": {"log_exact_mcap": 0.05, "all_other_features": 0.1},
            "criterion": (
                "finite abs(SMD)<0.05 for log_exact_mcap and <0.1 for every other feature, "
                "in both all and 2024plus scopes"
            ),
            "coverage": {
                "treated_exact_mcap": "286/286",
                "all": "D_supported/D_source >= 0.80",
                "2024plus": "D_supported/D_source >= 0.80",
            },
            "positivity": {
                "all_and_2024plus": {
                    "per_trade_ess_p05_gte": 5.0,
                    "max_pair_weight_lte": 0.5,
                    "global_pair_ess_per_trade_gte": 5.0,
                }
            },
            "fail_closed": True,
            "failure_action": "stop before loading or computing any realised outcome",
        },
        "outcome_authorization": {
            "allowed_in_this_plan": False,
            "unlock_condition": "exact-data closure and the frozen balance gate both pass",
        },
        "outcome_design": {
            "horizons_sessions": [5, 20, 60],
            "common_support": "entry day is day one; endpoints use offsets 4/19/59; identical controls at every horizon",
            "primary_contrast": "treated net return minus frozen conditional-entropy weighted control mean",
            "sensitivity_contrast": "equal-weight mean and median controls, reported as non-primary sensitivities",
            "path_policy": "no-open or gap-at-limit control stays cash without replacement; post-entry suspension uses LOCF",
            "terminal_bounds": "missing terminal settlement reports pessimistic -100% and optimistic LOCF bounds",
            "costs": {"buy": 0.0015, "sell": 0.0025, "cash": 0.0},
            "inference": {
                "primary_estimand": "equal-weight treated-trade ATT",
                "hac": ("complete decision-date calendar ratio-influence Newey-West, lags H5/H20/H60 = 4/19/59"),
                "bootstrap": (
                    "stationary bootstrap on complete decision-date calendar, expected blocks H5/H20/H60 = "
                    "5/20/60, 10000 replications, seed 42"
                ),
                "multiple_testing": "Holm-adjusted one-sided H5/H20/H60 family; no horizon cherry-picking",
                "secondary_estimand": "equal-decision-date mean, reported separately from trade-weight ATT",
                "network_dependence": "report control reuse and overlapping-path diagnostics; HAC alone is not a proof",
            },
        },
    }


def assert_outcome_blind(manifest: Mapping[str, Any]) -> None:
    """Reject embedded realised-result fields while allowing a frozen outcome protocol."""

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_RESULT_KEYS:
                    raise RuntimeError(f"outcome-blind plan contains result field: {'.'.join((*path, str(key)))}")
                walk(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(manifest)


def validate_design_contract(manifest: Mapping[str, Any]) -> None:
    """Fail closed when an authorization, source, matching, or balance rule drifts."""

    assert_outcome_blind(manifest)
    design = manifest.get("design", {})
    source = design.get("data_source", {})
    primary = design.get("matching", {}).get("primary", {})
    balance = design.get("balance_gate", {})
    authorization = design.get("outcome_authorization", {})
    inference = design.get("outcome_design", {}).get("inference", {})
    verdict = manifest.get("verdict", {})
    required = (
        design.get("outcome_blind") is True,
        source.get("endpoint") == "daily_basic",
        source.get("fields") == ["ts_code", "trade_date", "close", "total_share", "float_share", "total_mv", "circ_mv"],
        source.get("exact_mcap_cny") == "circ_mv * 10000",
        source.get("mix_sources_within_date") is False,
        source.get("fallback_for_missing_exact_mcap") is False,
        primary.get("name") == "pit_exact_industry_mcap2_conditional_entropy_v1",
        primary.get("candidate_pool")
        == (
            "all controls with complete frozen causal balance features inside the inclusive exact point-in-time "
            "circ_mv caliper"
        ),
        primary.get("caliper_ratio") == 2.0,
        primary.get("min_controls") == 5,
        primary.get("weighting")
        == (
            "conditional entropy tilting with one all-scope and one 2024plus-scope coefficient per balance "
            "feature; weights normalize to one within each treated trade"
        ),
        primary.get("solver")
        == {
            "method": "scipy.optimize.least_squares",
            "max_nfev": 2000,
            "parameter_bounds": [-50.0, 50.0],
            "initial_parameters": "all zeros",
            "xtol_ftol_gtol": 1e-12,
        },
        primary.get("tie_break") == "match_distance, symbol",
        balance.get("evaluated_before_outcome_loading") is True,
        balance.get("scopes") == {"all": 286, "2024plus": 174},
        balance.get("features")
        == [
            "log_exact_mcap",
            "ret5",
            "ret20",
            "ret60",
            "vol20",
            "vol60",
            "log_price",
            "log_amount",
            "liq20",
        ],
        balance.get("smd_thresholds") == {"log_exact_mcap": 0.05, "all_other_features": 0.1},
        balance.get("coverage")
        == {
            "treated_exact_mcap": "286/286",
            "all": "D_supported/D_source >= 0.80",
            "2024plus": "D_supported/D_source >= 0.80",
        },
        balance.get("positivity")
        == {
            "all_and_2024plus": {
                "per_trade_ess_p05_gte": 5.0,
                "max_pair_weight_lte": 0.5,
                "global_pair_ess_per_trade_gte": 5.0,
            }
        },
        balance.get("fail_closed") is True,
        authorization.get("allowed_in_this_plan") is False,
        inference.get("primary_estimand") == "equal-weight treated-trade ATT",
        inference.get("multiple_testing") == "Holm-adjusted one-sided H5/H20/H60 family; no horizon cherry-picking",
        verdict.get("outcome_evaluation_authorized") is False,
        verdict.get("live_authorized") is False,
    )
    if not all(required):
        raise RuntimeError("PIT exact-mcap design contract is incomplete or drifted")


def build_manifest() -> dict[str, Any]:
    """Rebuild the deterministic, lightweight, outcome-blind plan."""

    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    common = json.loads(COMMON_AUDIT_PATH.read_text(encoding="utf-8"))
    request_identities = _validate_request_manifest(request)
    _validate_common_bindings(common, request_identities["request"])
    treated = _treated_identity(request_identities["treated"])
    if treated["rows"] != request_identities["treated"]["count"]:
        raise RuntimeError("treated common-support denominator drift")

    production_audit = json.loads(ARTIFACT_PATHS["production_audit"].read_text(encoding="utf-8"))
    if production_audit.get("schema") != "surge_delay5_production_cohort_audit_v3":
        raise RuntimeError("production audit schema drift")

    manifest = {
        "schema": "delay5_pit_exact_mcap_plan_manifest_v1",
        "canonicalization": CANONICALIZATION,
        "source_identity": {
            "baseline_parent_commit": "52b6fc75ecba6519f2b641c150343da81d21e324",
            "baseline_parent_note": (
                "research baseline before the frozen PIT plan; current HEAD is intentionally not compared"
            ),
            "scripts": {
                label: {"locator": _repo_locator(path), "sha256": sha256_file(path)}
                for label, path in SCRIPT_PATHS.items()
            },
            "reproducible_identity": _reproducible_source_identity(),
            "artifacts": _artifact_identities(),
            "industry_snapshots": _industry_identities(),
        },
        "request_identity": {
            **request_identities,
            "missing_frozen_industry": request.get("cohort", {}).get("missing_frozen_industry", []),
        },
        "treated_common_support": treated,
        "design": _design(),
        "verdict": {
            "status": "OUTCOME_EVALUATION_LOCKED_PENDING_ENTROPY_BALANCE_AUDIT",
            "exact_data_identity_frozen": True,
            "balance_passed": False,
            "outcome_evaluation_authorized": False,
            "live_authorized": False,
        },
    }
    if (
        manifest["source_identity"]["reproducible_identity"]["production"]["common60_schedule_identity"]
        != treated["schedule_identity"]
    ):
        raise RuntimeError("production/common treated schedule identity drift")
    validate_design_contract(manifest)
    return manifest


def identity_checks(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, bool]:
    """Compare every frozen plan section; callers must require all checks."""

    fields = (
        "schema",
        "canonicalization",
        "request_identity",
        "treated_common_support",
        "design",
        "verdict",
    )
    checks = {field: actual.get(field) == expected.get(field) for field in fields}
    checks["portable_source_identity"] = portable_source_identity(actual.get("source_identity", {})) == (
        portable_source_identity(expected.get("source_identity", {}))
    )
    return checks


def run_audit() -> dict[str, Any]:
    """Reproduce the plan and fail closed on any tracked identity drift."""

    if not EXPECTED_PLAN_PATH.is_file():
        raise FileNotFoundError(f"tracked PIT plan is missing: {EXPECTED_PLAN_PATH}")
    expected = json.loads(EXPECTED_PLAN_PATH.read_text(encoding="utf-8"))
    validate_design_contract(expected)
    actual = build_manifest()
    checks = identity_checks(actual, expected)
    return {
        "schema": "delay5_pit_exact_mcap_plan_audit_v1",
        "expected_plan_sha256": sha256_file(EXPECTED_PLAN_PATH),
        "checks": checks,
        "verdict": {
            "status": "OUTCOME_BLIND_PLAN_REPRODUCED" if all(checks.values()) else "PLAN_IDENTITY_DRIFT",
            "outcome_evaluation_authorized": False,
            "live_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="atomically refresh the tracked outcome-blind plan from current verified inputs",
    )
    args = parser.parse_args()
    if args.write_expected:
        manifest = build_manifest()
        temporary = EXPECTED_PLAN_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(EXPECTED_PLAN_PATH)
    result = run_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
