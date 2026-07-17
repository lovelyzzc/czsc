from __future__ import annotations

# ruff: noqa: E402, I001

import json
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_data_v2 as data_v2
import xs_chan_oos_ledger as ledger


N_SYMBOLS = data_v2.MIN_STATE_AUDIT_SYMBOLS
N_DATES = data_v2.MIN_HISTORY_SESSIONS + 10
ENGINE_SHA = data_v2.sha256_bytes(b"frozen-state-engine")


def _symbols() -> list[str]:
    return [f"{index:06d}.SZ" for index in range(N_SYMBOLS)]


def _source_entry(api: str, params: dict, retrieved_at: str) -> dict:
    return {
        "provider": "unit-test-offline-source",
        "api": api,
        "parameters": params,
        "retrieved_at_utc": retrieved_at,
    }


def _source_metadata(
    retrieved_at: str,
    start: str,
    end: str,
    *,
    security_symbols: list[str],
    raw_symbols: list[str],
    state_cache_manifest_sha256: str,
    engine_combination_sha256: str,
    raw_input_evidence_sha256: str,
) -> dict:
    return {
        "calendar": _source_entry(
            "trade_cal",
            {"exchange": "SSE", "start_date": start, "end_date": end},
            retrieved_at,
        ),
        "security_master": _source_entry(
            "stock_basic",
            {
                "list_statuses": ["L", "D", "P"],
                "symbol_count": len(set(security_symbols)),
                "symbol_set_sha256": data_v2.symbol_set_sha256(security_symbols),
            },
            retrieved_at,
        ),
        "namechange": _source_entry(
            "namechange",
            {"start_date": start, "end_date": end},
            retrieved_at,
        ),
        "raw_daily": _source_entry(
            "daily",
            {
                "start_date": start,
                "end_date": end,
                "adjustment": "none",
                "symbol_count": len(set(raw_symbols)),
                "symbol_set_sha256": data_v2.symbol_set_sha256(raw_symbols),
            },
            retrieved_at,
        ),
        "adj_factor": _source_entry(
            "adj_factor",
            {"start_date": start, "end_date": end},
            retrieved_at,
        ),
        "daily_basic": _source_entry(
            "daily_basic",
            {"start_date": start, "end_date": end, "fields": ["free_share", "circ_mv"]},
            retrieved_at,
        ),
        "stk_limit": _source_entry(
            "stk_limit",
            {"start_date": start, "end_date": end, "fields": ["up_limit", "down_limit"]},
            retrieved_at,
        ),
        "industry_membership": _source_entry(
            "index_member",
            {
                "start_date": start,
                "end_date": end,
                "classification": "SW2021",
                "membership_scope": "historical",
            },
            retrieved_at,
        ),
        "corporate_actions": _source_entry(
            "dividend_and_terminal_actions",
            {
                "start_date": start,
                "end_date": end,
                "action_types": list(data_v2.CORPORATE_ACTION_TYPES),
                "coverage_scope": "all_raw_symbols_for_full_date_range",
                "queried_symbol_count": len(set(raw_symbols)),
                "queried_symbol_set_sha256": data_v2.symbol_set_sha256(raw_symbols),
            },
            retrieved_at,
        ),
        "universe_reconciliation": _source_entry(
            "offline_query_reconciliation",
            {"required_partitions": list(data_v2.RECONCILIATION_PARTITIONS)},
            retrieved_at,
        ),
        "chan_states": _source_entry(
            "trend_regime.iter_states",
            {
                "engine_sha256": engine_combination_sha256,
                "state_domain": list(data_v2.STATE_DOMAIN),
                "price_track": "raw_times_adj_factor",
                "state_cache_manifest_sha256": state_cache_manifest_sha256,
                "engine_combination_sha256": engine_combination_sha256,
                "raw_input_evidence_sha256": raw_input_evidence_sha256,
            },
            retrieved_at,
        ),
        "state_audit": _source_entry(
            "state_prefix_audit",
            {
                "minimum_symbols": data_v2.MIN_STATE_AUDIT_SYMBOLS,
                "minimum_cutoffs": data_v2.MIN_STATE_AUDIT_CUTOFFS,
                "state_cache_manifest_sha256": state_cache_manifest_sha256,
            },
            retrieved_at,
        ),
    }


def _write_state_cache_manifest(root: Path, symbols: list[str]) -> tuple[str, str, str]:
    algorithm_hashes = {
        component: data_v2.sha256_bytes(f"engine:{component}".encode()) for component in data_v2.STATE_ENGINE_COMPONENTS
    }
    input_sources = [
        {
            "name": f"{symbol}.parquet",
            "size": 10_000 + index,
            "sha256": data_v2.sha256_bytes(f"raw-input:{symbol}".encode()),
        }
        for index, symbol in enumerate(symbols)
    ]
    identity = {
        "schema_version": "xs_chan_state_cache_v1",
        "config": {
            "audit_symbols": data_v2.MIN_STATE_AUDIT_SYMBOLS,
            "audit_checkpoints": data_v2.MIN_STATE_AUDIT_CUTOFFS,
            "audit_seed": 20260717,
            "warmup_bars": 120,
        },
        "projection_columns": ["symbol", "dt", "regime"],
        "valid_regimes": list(data_v2.STATE_DOMAIN),
        "algorithm_hashes": algorithm_hashes,
        "runtime_versions": {"python": "3.12-test"},
        "sources": input_sources,
    }
    identity_digest = data_v2.sha256_json(identity)
    projection_path = root / "chan_states.parquet"
    audit_path = root / "state_audit.parquet"
    source_records = [
        {
            **fingerprint,
            "symbol": symbol,
            "status": "generated",
            "source_rows": N_DATES,
            "state_rows": N_DATES,
        }
        for fingerprint, symbol in zip(input_sources, symbols, strict=True)
    ]
    state_manifest = {
        "schema_version": "xs_chan_state_cache_v1",
        "cache_id": identity_digest,
        "content_digest_sha256": identity_digest,
        "identity": identity,
        "engine": {
            **{
                component: {"path": f"/{component}", "sha256": algorithm_hashes[component]}
                for component in data_v2.STATE_ENGINE_COMPONENTS
            },
            "versions": {"python": "3.12-test"},
        },
        "config": identity["config"],
        "projection": {
            "path": "states.parquet",
            "sha256": data_v2.sha256_file(projection_path),
            "columns": ["symbol", "dt", "regime"],
            "rows": N_SYMBOLS * N_DATES,
            "symbols": N_SYMBOLS,
        },
        "state_audit": {"path": "state_audit.json", "sha256": data_v2.sha256_bytes(b"audit-summary")},
        "state_audit_details": {
            "path": "state_audit.parquet",
            "sha256": data_v2.sha256_file(audit_path),
            "columns": list(data_v2.DATASET_SPECS["state_audit"].required_columns),
            "rows": data_v2.MIN_STATE_AUDIT_SYMBOLS * data_v2.MIN_STATE_AUDIT_CUTOFFS,
        },
        "sources": source_records,
    }
    path = root / data_v2.DEFAULT_STATE_CACHE_MANIFEST
    path.write_text(json.dumps(state_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return (
        data_v2.sha256_file(path),
        data_v2.sha256_json(algorithm_hashes),
        data_v2.sha256_json(sorted(input_sources, key=lambda item: item["name"])),
    )


def _write_reconciliation(
    root: Path,
    security: pd.DataFrame,
    raw: pd.DataFrame,
    source_asof: pd.Timestamp,
    ingested_at: pd.Timestamp,
) -> None:
    rows = []
    for partition in data_v2.RECONCILIATION_PARTITIONS:
        dataset = partition.split(":", 1)[0]
        if dataset == "security_master":
            status = partition.rsplit("=", 1)[-1]
            symbols = security.loc[security["list_status"].eq(status), "ts_code"]
        else:
            status = None
            symbols = raw["ts_code"].drop_duplicates()
        count = int(symbols.nunique())
        digest = data_v2.symbol_set_sha256(symbols)
        rows.append(
            {
                "dataset": dataset,
                "partition": partition,
                "trade_date": pd.NaT,
                "list_status": status,
                "expected_count": count,
                "received_count": count,
                "expected_symbol_set_sha256": digest,
                "received_symbol_set_sha256": digest,
                "request_sha256": data_v2.sha256_bytes(f"request:{partition}".encode()),
                "response_sha256": data_v2.sha256_bytes(f"response:{partition}".encode()),
                "missing_count": 0,
                "source_asof": source_asof,
                "ingested_at": ingested_at,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "universe_reconciliation.parquet", index=False)


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    symbols = _symbols()
    dates = pd.bdate_range("2024-01-02", periods=N_DATES)
    source_asof = pd.Timestamp(dates[-1], tz="UTC") + pd.Timedelta(days=2)
    ingested_at = source_asof + pd.Timedelta(hours=1)
    retrieved_at = ingested_at.isoformat().replace("+00:00", "Z")

    calendar = pd.DataFrame(
        {
            "trade_date": dates,
            "is_open": True,
            "prev_trade_date": pd.Series(dates).shift(1),
            "next_trade_date": pd.Series(dates).shift(-1),
            "source_asof": source_asof,
            "ingested_at": ingested_at,
        }
    )
    calendar.to_parquet(root / "calendar.parquet", index=False)

    list_statuses = ["L"] * (N_SYMBOLS - 2) + ["D", "P"]
    delist_dates = [pd.NaT] * (N_SYMBOLS - 2) + [pd.Timestamp("2025-01-02"), pd.NaT]
    security = pd.DataFrame(
        {
            "ts_code": symbols,
            "name": [f"测试{index}" for index in range(N_SYMBOLS)],
            "exchange": "SZSE",
            "board": "MAIN",
            "list_date": pd.Timestamp("2020-01-02"),
            "delist_date": delist_dates,
            "list_status": list_statuses,
            "source_asof": source_asof,
            "ingested_at": ingested_at,
        }
    )
    security.to_parquet(root / "security_master.parquet", index=False)

    namechange = pd.DataFrame(
        {
            "ts_code": [symbols[0]],
            "name": ["测试股份"],
            "effective_from": [pd.Timestamp("2020-01-02")],
            "effective_to": [pd.NaT],
            "source_asof": [source_asof],
            "ingested_at": [ingested_at],
        }
    )
    namechange.to_parquet(root / "namechange.parquet", index=False)

    symbol_values = np.repeat(symbols, N_DATES)
    date_values = np.tile(dates.to_numpy(), N_SYMBOLS)
    symbol_number = np.repeat(np.arange(N_SYMBOLS), N_DATES)
    bar_number = np.tile(np.arange(N_DATES), N_SYMBOLS)
    open_price = 10.0 + symbol_number * 0.01 + bar_number * 0.001
    close_price = open_price * 1.001
    raw = pd.DataFrame(
        {
            "ts_code": symbol_values,
            "trade_date": date_values,
            "open": open_price,
            "high": open_price * 1.02,
            "low": open_price * 0.98,
            "close": close_price,
            "pre_close": open_price,
            "vol": 1_000_000.0,
            "amount": 100_000_000.0,
            "pct_chg": 0.1,
            "source_asof": source_asof,
            "ingested_at": ingested_at,
        }
    )
    raw.to_parquet(root / "raw_daily.parquet", index=False)

    key = raw[["ts_code", "trade_date", "source_asof", "ingested_at"]].copy()
    factor = key.copy()
    factor["adj_factor"] = 1.0
    factor.to_parquet(root / "adj_factor.parquet", index=False)

    daily_basic = key.copy()
    daily_basic["free_share"] = 100_000.0
    daily_basic.to_parquet(root / "daily_basic.parquet", index=False)

    limits = key.copy()
    limits["up_limit"] = open_price * 1.1
    limits["down_limit"] = open_price * 0.9
    limits.to_parquet(root / "stk_limit.parquet", index=False)

    industry = pd.DataFrame(
        {
            "ts_code": symbols,
            "industry_code": [f"I{index % 10:02d}" for index in range(N_SYMBOLS)],
            "industry_name": [f"行业{index % 10}" for index in range(N_SYMBOLS)],
            "classification_version": "SW2021",
            "effective_from": pd.Timestamp("2020-01-02"),
            "effective_to": pd.NaT,
            "source_asof": source_asof,
            "ingested_at": ingested_at,
        }
    )
    industry.to_parquet(root / "industry_membership.parquet", index=False)

    corporate_actions = pd.DataFrame(
        {
            "ts_code": [symbols[0]],
            "effective_date": [dates[100]],
            "action_type": ["cash_dividend"],
            "cash_per_pre_action_share": [0.10],
            "post_to_pre_share_ratio": [np.nan],
            "official_disposal_cash_per_pre_action_share": [np.nan],
            "consideration_ts_code": pd.Series([pd.NA], dtype="string"),
            "source_asof": [source_asof],
            "ingested_at": [ingested_at],
        }
    )
    corporate_actions.to_parquet(root / "corporate_actions.parquet", index=False)
    _write_reconciliation(root, security, raw, source_asof, ingested_at)

    warmup = bar_number < 120
    regimes = np.where(warmup, 0, (bar_number - 120) % 10 + 1)
    states = pd.DataFrame(
        {
            "symbol": symbol_values,
            "dt": date_values,
            "regime": regimes.astype(np.int8),
        }
    )
    states.to_parquet(root / "chan_states.parquet", index=False)

    checkpoints = dates[-data_v2.MIN_STATE_AUDIT_CUTOFFS :]
    audit_rows = []
    for symbol in symbols:
        for checkpoint in checkpoints:
            prefix_rows = int(dates.get_loc(checkpoint)) + 1
            state_hash = data_v2.sha256_bytes(f"{symbol}|{checkpoint.date()}|state".encode())
            audit_rows.append(
                {
                    "symbol": symbol,
                    "checkpoint_dt": checkpoint,
                    "prefix_rows": prefix_rows,
                    "expected_rows": prefix_rows,
                    "actual_rows": prefix_rows,
                    "expected_sha256": state_hash,
                    "actual_sha256": state_hash,
                    "symbol_mismatches": 0,
                    "dt_mismatches": 0,
                    "regime_mismatches": 0,
                    "passed": True,
                }
            )
    pd.DataFrame(audit_rows).to_parquet(root / "state_audit.parquet", index=False)

    state_cache_sha, engine_combination_sha, raw_input_sha = _write_state_cache_manifest(root, symbols)
    metadata = _source_metadata(
        retrieved_at,
        "20100101",
        dates[-1].strftime("%Y%m%d"),
        security_symbols=symbols,
        raw_symbols=symbols,
        state_cache_manifest_sha256=state_cache_sha,
        engine_combination_sha256=engine_combination_sha,
        raw_input_evidence_sha256=raw_input_sha,
    )
    (root / data_v2.DEFAULT_SOURCE_METADATA).write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(data_v2.FROZEN_PROTOCOL_PATH, root / data_v2.DEFAULT_PROTOCOL)
    data_v2.build_manifest(root)


@pytest.fixture(scope="module")
def pristine_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("xs_chan_data_v2") / "bundle"
    _write_bundle(root)
    return root


def _copy_bundle(pristine: Path, tmp_path: Path) -> Path:
    target = tmp_path / "bundle"
    shutil.copytree(pristine, target)
    return target


def _rewrite_manifest(root: Path, mutate) -> dict:
    path = root / data_v2.DEFAULT_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["data_evidence_sha256"] = data_v2.compute_data_evidence_sha256(manifest)
    manifest["manifest_sha256"] = data_v2.compute_manifest_sha256(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def _business_days(start: date, count: int) -> list[date]:
    sessions: list[date] = []
    current = start
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def _add_external_ledger(
    path: Path,
    bundle: Path,
    *,
    complete_weeks: int,
    protocol_sha256: str | None = None,
    data_evidence_sha256: str | None = None,
) -> Path:
    if complete_weeks:
        raise ValueError("data-contract tests leave weekly payload validation to xs_chan_oos_ledger tests")
    manifest = json.loads((bundle / data_v2.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    sessions = _business_days(date(2026, 7, 20), 55 * 5 + 1)
    original_clock = ledger._utc_now
    ledger._utc_now = lambda: datetime(2026, 7, 17, 12, tzinfo=UTC)
    try:
        dependencies = {name: data_v2.sha256_bytes(f"test-{name}".encode()) for name in ledger.EXPECTED_DEPENDENCY_KEYS}
        dependencies["data_evidence"] = data_evidence_sha256 or manifest["data_evidence_sha256"]
        dependencies["state_engine"] = ENGINE_SHA
        ledger.init_ledger(
            path,
            protocol_sha256=protocol_sha256 or manifest["protocol"]["sha256"],
            dependency_sha256=dependencies,
            calendar_sessions=sessions,
            oos_start_date=sessions[0],
        )
    finally:
        ledger._utc_now = original_clock
    return path


def test_evidence_complete_bundle_stays_closed_without_independent_state_replay(pristine_bundle: Path):
    report = data_v2.validate_bundle(pristine_bundle)

    assert report.valid is False, report.to_dict()
    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert report.gates["forward_chain_complete"] is False
    assert report.gates["state_cache_provenance"] is True
    assert report.gates["state_recompute_engineering"] is False
    assert all(report.gates[name] is False for name in data_v2.REQUIRED_ENGINEERING_GATES)
    assert any(issue.code == "state_recompute_unavailable" for issue in report.issues)
    assert any(issue.code == "formal_replay_stack_unavailable" for issue in report.issues)
    manifest = json.loads((pristine_bundle / data_v2.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    assert not data_v2.RESERVED_READINESS_KEYS.intersection(manifest)
    assert manifest["protocol"]["sha256"] == data_v2.frozen_protocol_sha256()
    assert manifest["datasets"]["raw_daily"]["semantics"]["volume_unit"] == "lots"
    assert manifest["datasets"]["raw_daily"]["semantics"]["amount_unit"] == "thousand_CNY"
    protocol = json.loads((pristine_bundle / data_v2.DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
    assert protocol["data_contract"]["corporate_action_types"] == list(data_v2.CORPORATE_ACTION_TYPES)
    assert protocol["data_contract"]["universe_reconciliation_partitions"] == list(data_v2.RECONCILIATION_PARTITIONS)


def test_missing_manifest_is_fail_closed(tmp_path: Path):
    report = data_v2.validate_bundle(tmp_path)

    assert report.valid is False
    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert any(issue.code == "manifest_load" for issue in report.issues)


def test_manifest_rejects_traversal_even_with_recomputed_hash(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)

    _rewrite_manifest(root, lambda manifest: manifest["datasets"]["calendar"].update(path="../calendar.parquet"))
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert any(issue.code == "unsafe_path" and issue.dataset == "calendar" for issue in report.issues)


def test_manifest_rejects_symlink_alias(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    (root / "calendar-link.parquet").symlink_to(root / "calendar.parquet")

    _rewrite_manifest(root, lambda manifest: manifest["datasets"]["calendar"].update(path="calendar-link.parquet"))
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert any(issue.code == "unsafe_path" for issue in report.issues)


def test_physical_hash_tamper_closes_all_gates(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    with (root / "raw_daily.parquet").open("ab") as stream:
        stream.write(b"tamper")

    report = data_v2.validate_bundle(root)

    assert report.valid is False
    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert any(issue.code == "file_hash" and issue.dataset == "raw_daily" for issue in report.issues)


def test_duplicate_daily_basic_key_is_evidence_not_manual_override(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "daily_basic.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert report.gates["dataset_daily_basic"] is False
    assert any(issue.code == "duplicate_key" and issue.dataset == "daily_basic" for issue in report.issues)


def test_manual_readiness_and_semantic_relabel_are_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)

    def mutate(manifest: dict) -> None:
        manifest["forward_start_allowed"] = True
        manifest["datasets"]["raw_daily"]["semantics"]["price_basis"] = "qfq"

    _rewrite_manifest(root, mutate)
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert any(issue.code == "manual_readiness" for issue in report.issues)
    assert any(issue.code == "semantic_declaration" and issue.dataset == "raw_daily" for issue in report.issues)


def test_malformed_dataset_source_entry_returns_a_closed_report_instead_of_raising(
    pristine_bundle: Path,
    tmp_path: Path,
):
    root = _copy_bundle(pristine_bundle, tmp_path)

    def mutate(manifest: dict) -> None:
        del manifest["datasets"]["raw_daily"]["source"]["retrieved_at_utc"]

    _rewrite_manifest(root, mutate)
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert report.gates["dataset_raw_daily"] is False
    assert any(issue.code == "source_metadata" and issue.dataset == "raw_daily" for issue in report.issues)


def test_manifest_rejects_embedded_second_forward_schema(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)

    _rewrite_manifest(root, lambda manifest: manifest.update(forward_chain={"ready": True}))
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert any(issue.code == "manifest_fields" for issue in report.issues)


def test_future_state_payload_column_is_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "chan_states.parquet"
    frame = pd.read_parquet(path)
    frame["next_open"] = 10.0
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert any(issue.code == "unsafe_columns" and issue.dataset == "chan_states" for issue in report.issues)


def test_state_audit_mismatch_closes_forward_start(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "state_audit.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "regime_mismatches"] = 1
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert report.gates["state_audit_semantics"] is False
    assert any(issue.code == "state_mismatch" for issue in report.issues)


def test_empty_corporate_actions_passes_only_with_full_coverage_reconciliation(
    pristine_bundle: Path,
    tmp_path: Path,
):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "corporate_actions.parquet"
    pd.read_parquet(path).iloc[0:0].to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["dataset_corporate_actions"] is True
    assert report.gates["corporate_actions_semantics"] is True
    assert report.gates["reconciliation_binds_physical_symbols"] is True
    assert report.gates["source_symbol_and_action_coverage_binding"] is True


def test_invalid_corporate_action_cash_is_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "corporate_actions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "cash_per_pre_action_share"] = -0.01
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["corporate_actions_semantics"] is False
    assert any(issue.code == "corporate_action_cash" for issue in report.issues)


def test_delist_share_requires_consideration_symbol(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "corporate_actions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "action_type"] = "delist_share"
    frame.loc[0, "cash_per_pre_action_share"] = np.nan
    frame.loc[0, "post_to_pre_share_ratio"] = 0.5
    frame.loc[0, "consideration_ts_code"] = pd.NA
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["corporate_actions_semantics"] is False
    assert any(issue.code == "corporate_action_applicability" for issue in report.issues)


def test_effective_delisting_requires_terminal_action(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "security_master.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["list_status"].eq("D"), "delist_date"] = pd.Timestamp("2024-12-30")
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["terminal_actions_cover_effective_delistings"] is False
    assert any(issue.code == "terminal_action_missing" for issue in report.issues)


def test_terminal_action_types_are_mutually_exclusive_per_delisting(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    master_path = root / "security_master.parquet"
    master = pd.read_parquet(master_path)
    delisted_symbol = master.loc[master["list_status"].eq("D"), "ts_code"].iloc[0]
    effective_date = pd.Timestamp("2024-12-30")
    master.loc[master["ts_code"].eq(delisted_symbol), "delist_date"] = effective_date
    master.to_parquet(master_path, index=False)

    action_path = root / "corporate_actions.parquet"
    actions = pd.read_parquet(action_path)
    provenance = actions.iloc[0][["source_asof", "ingested_at"]].to_dict()
    terminal = pd.DataFrame(
        [
            {
                "ts_code": delisted_symbol,
                "effective_date": effective_date,
                "action_type": "delist_cash",
                "cash_per_pre_action_share": 1.0,
                "post_to_pre_share_ratio": np.nan,
                "official_disposal_cash_per_pre_action_share": np.nan,
                "consideration_ts_code": pd.NA,
                **provenance,
            },
            {
                "ts_code": delisted_symbol,
                "effective_date": effective_date,
                "action_type": "delist_writeoff",
                "cash_per_pre_action_share": np.nan,
                "post_to_pre_share_ratio": np.nan,
                "official_disposal_cash_per_pre_action_share": np.nan,
                "consideration_ts_code": pd.NA,
                **provenance,
            },
        ]
    )
    pd.concat([actions, terminal], ignore_index=True).to_parquet(action_path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["terminal_actions_cover_effective_delistings"] is False
    assert any(issue.code == "terminal_action_conflict" for issue in report.issues)


def test_security_board_requires_canonical_enum(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "security_master.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "board"] = "科创板"
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["security_master_semantics"] is False
    assert any(issue.code == "security_board" for issue in report.issues)


def test_namechange_gap_cannot_fall_back_to_current_master_name(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "namechange.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "effective_from"] = pd.Timestamp("2024-02-01")
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["namechange_semantics"] is True
    assert report.gates["namechange_gapless_when_present"] is False
    assert any(issue.code == "namechange_gap" for issue in report.issues)


def test_overlapping_namechange_history_is_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "namechange.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "effective_to"] = pd.Timestamp("2024-06-30")
    second = frame.iloc[[0]].copy()
    second["name"] = "测试新名"
    second["effective_from"] = pd.Timestamp("2024-06-01")
    second["effective_to"] = pd.NaT
    pd.concat([frame, second], ignore_index=True).to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["namechange_semantics"] is False
    assert report.forward_start_allowed is False
    assert any(issue.code == "overlapping_interval" and issue.dataset == "namechange" for issue in report.issues)


def test_self_consistent_all_l_reconciliation_is_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    security_path = root / "security_master.parquet"
    security = pd.read_parquet(security_path)
    security["list_status"] = "L"
    security["delist_date"] = pd.NaT
    security.to_parquet(security_path, index=False)

    reconciliation_path = root / "universe_reconciliation.parquet"
    reconciliation = pd.read_parquet(reconciliation_path)
    for status in ("L", "D", "P"):
        partition = f"security_master:list_status={status}"
        selected = security.loc[security["list_status"].eq(status), "ts_code"]
        digest = data_v2.symbol_set_sha256(selected)
        row = reconciliation["partition"].eq(partition)
        reconciliation.loc[row, ["expected_count", "received_count"]] = int(selected.nunique())
        reconciliation.loc[row, ["expected_symbol_set_sha256", "received_symbol_set_sha256"]] = digest
    reconciliation.to_parquet(reconciliation_path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["reconciliation_binds_physical_symbols"] is True
    assert report.gates["universe_reconciliation_semantics"] is False
    assert any(issue.code == "reconciliation_mismatch" for issue in report.issues)


def test_survivor_subset_claim_cannot_override_physical_symbol_set(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    subset = _symbols()[:-1]
    subset_digest = data_v2.symbol_set_sha256(subset)
    reconciliation_path = root / "universe_reconciliation.parquet"
    reconciliation = pd.read_parquet(reconciliation_path)
    subset_rows = reconciliation["partition"].isin(
        ["raw_daily:bundle_symbol_set", "corporate_actions:coverage_universe"]
    )
    reconciliation.loc[subset_rows, ["expected_count", "received_count"]] = len(subset)
    reconciliation.loc[
        subset_rows,
        ["expected_symbol_set_sha256", "received_symbol_set_sha256"],
    ] = subset_digest
    reconciliation.to_parquet(reconciliation_path, index=False)

    source_path = root / data_v2.DEFAULT_SOURCE_METADATA
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["raw_daily"]["parameters"].update(symbol_count=len(subset), symbol_set_sha256=subset_digest)
    source["corporate_actions"]["parameters"].update(
        queried_symbol_count=len(subset),
        queried_symbol_set_sha256=subset_digest,
    )
    source_path.write_text(json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["universe_reconciliation_semantics"] is True
    assert report.gates["reconciliation_binds_physical_symbols"] is False
    assert report.gates["source_symbol_and_action_coverage_binding"] is False


def test_evidence_after_manifest_cutoff_is_rejected(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    path = root / "raw_daily.parquet"
    frame = pd.read_parquet(path)
    frame["source_asof"] = pd.Timestamp("2100-01-01", tz="UTC")
    frame["ingested_at"] = pd.Timestamp("2100-01-02", tz="UTC")
    frame.to_parquet(path, index=False)
    data_v2.build_manifest(root)

    report = data_v2.validate_bundle(root)

    assert report.gates["manifest_time_cutoff"] is False
    assert any(issue.code == "manifest_time_cutoff" for issue in report.issues)


def test_state_cache_manifest_must_bind_physical_outputs(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    state_path = root / data_v2.DEFAULT_STATE_CACHE_MANIFEST
    state_manifest = json.loads(state_path.read_text(encoding="utf-8"))
    state_manifest["projection"]["sha256"] = "f" * 64
    state_path.write_text(json.dumps(state_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def bind_tampered_state_manifest(manifest: dict) -> None:
        entry = manifest["state_cache_manifest"]
        entry["sha256"] = data_v2.sha256_file(state_path)
        entry["size_bytes"] = state_path.stat().st_size

    _rewrite_manifest(root, bind_tampered_state_manifest)
    report = data_v2.validate_bundle(root)

    assert report.gates["state_cache_provenance"] is False
    assert any(issue.code in {"state_cache_source_binding", "state_cache_output_binding"} for issue in report.issues)


def test_source_metadata_cannot_contain_credentials(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    source_path = root / data_v2.DEFAULT_SOURCE_METADATA
    metadata = json.loads(source_path.read_text(encoding="utf-8"))
    metadata["raw_daily"]["parameters"]["token"] = "must-not-enter-a-bundle"
    source_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(data_v2.ContractError, match="credential"):
        data_v2.build_manifest(root)


def test_protocol_copy_must_match_repository_bytes(pristine_bundle: Path, tmp_path: Path):
    root = _copy_bundle(pristine_bundle, tmp_path)
    protocol_path = root / data_v2.DEFAULT_PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["eligibility"]["min_market_sessions_since_listing"] = 251
    protocol_path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def bind_modified_protocol(manifest: dict) -> None:
        entry = manifest["protocol"]
        entry["sha256"] = data_v2.sha256_file(protocol_path)
        entry["size_bytes"] = protocol_path.stat().st_size

    _rewrite_manifest(root, bind_modified_protocol)
    report = data_v2.validate_bundle(root)

    assert report.forward_start_allowed is False
    assert report.gates["protocol_frozen"] is False
    assert any(issue.code == "protocol_repository_hash" for issue in report.issues)


def test_verified_external_ledger_freeze_is_bound_but_not_yet_confirmatory(
    pristine_bundle: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    chain = _add_external_ledger(
        tmp_path / "external-ledger",
        pristine_bundle,
        complete_weeks=0,
    )

    report = data_v2.validate_bundle(pristine_bundle, forward_chain_path=chain)

    assert report.valid is False, report.to_dict()
    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert report.gates["forward_chain_complete"] is False
    assert report.evidence["forward_chain"]["valid"] is True
    assert report.evidence["forward_chain"]["completed_weeks"] == 0
    assert (
        data_v2.main(
            [
                "validate",
                "--bundle-root",
                str(pristine_bundle),
                "--forward-chain",
                str(chain),
                "--require",
                "confirmatory-oos",
            ]
        )
        == 1
    )
    capsys.readouterr()


def test_external_ledger_must_bind_data_evidence(pristine_bundle: Path, tmp_path: Path):
    chain = _add_external_ledger(
        tmp_path / "wrong-dependency-ledger",
        pristine_bundle,
        complete_weeks=0,
        data_evidence_sha256="f" * 64,
    )

    report = data_v2.validate_bundle(pristine_bundle, forward_chain_path=chain)

    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert report.valid is False
    assert any(issue.code == "forward_dependency" for issue in report.issues)


def test_external_ledger_must_bind_frozen_protocol(pristine_bundle: Path, tmp_path: Path):
    chain = _add_external_ledger(
        tmp_path / "wrong-protocol-ledger",
        pristine_bundle,
        complete_weeks=0,
        protocol_sha256="e" * 64,
    )

    report = data_v2.validate_bundle(pristine_bundle, forward_chain_path=chain)

    assert report.forward_start_allowed is False
    assert report.confirmatory_oos is False
    assert report.valid is False
    assert any(issue.code == "forward_protocol" for issue in report.issues)


def test_cli_validate_defaults_to_forward_start_gate(pristine_bundle: Path):
    assert data_v2.main(["validate", "--bundle-root", str(pristine_bundle)]) == 1
