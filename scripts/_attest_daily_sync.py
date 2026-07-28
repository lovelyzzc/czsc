"""为旧版全市场日线同步审计生成诚实的事后完整性凭证。

该工具只验证现存输入、输出、快照与路由证据的一致性。它不会、也不能把旧审计
追溯性地升级为“已绑定原始执行源码”的证明。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from _sync_daily_data import (
    SNAPSHOT_MANIFEST_FILE_NAME,
    DailySyncError,
    atomic_write_bytes,
    canonical_json,
    identify_factor_refresh_symbols,
    inspect_inventory,
    sha256_bytes,
    sha256_file,
)

CLAIM = "POST_HOC_ARTIFACT_INTEGRITY_ONLY"
UNKNOWN_EXECUTION_SOURCE = "UNKNOWN_NOT_ATTESTED"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DailySyncError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise DailySyncError(f"JSON object must be a mapping: {path}")
    return payload


def _verify_file_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    expected = str(record["sha256"])
    if not path.is_file() or sha256_file(path) != expected:
        raise DailySyncError(f"frozen object differs from its recorded hash: {path}")


def _flatten_api_objects(objects: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for value in objects.values():
        if isinstance(value, Mapping) and "path" in value:
            records.append(value)
            continue
        if not isinstance(value, Mapping):
            raise DailySyncError("API response object registry is malformed")
        for record in value.values():
            if not isinstance(record, Mapping):
                raise DailySyncError("API response object record is malformed")
            records.append(record)
    return records


def inspect_v2_snapshot(path: Path) -> dict[str, Any]:
    manifest = _load_json(path / SNAPSHOT_MANIFEST_FILE_NAME)
    if manifest.get("schema") != "a_stock_daily_qfq_raw_snapshot_v2":
        raise DailySyncError(f"canonical snapshot is not schema v2: {path}")
    payload_files = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.iterdir())
        if item.is_file() and item.name != SNAPSHOT_MANIFEST_FILE_NAME
    ]
    payload_sha = sha256_bytes(canonical_json(payload_files))
    parquet_inventory = inspect_inventory(path)
    parquet_sha = str(parquet_inventory["content_inventory_sha256"])
    if (
        path.name != f"RAW_{payload_sha}"
        or manifest.get("closure_sha256") != payload_sha
        or manifest.get("payload_closure_sha256") != payload_sha
        or manifest.get("parquet_content_inventory_sha256") != parquet_sha
    ):
        raise DailySyncError(f"canonical snapshot identity differs from its bytes: {path}")
    return {
        "name": path.name,
        "payload_closure_sha256": payload_sha,
        "parquet_content_inventory_sha256": parquet_sha,
        "file_count": parquet_inventory["file_count"],
        "min_dt": parquet_inventory["min_dt"],
        "max_dt": parquet_inventory["max_dt"],
    }


def validate_attestation_claims(attestation: Mapping[str, Any]) -> None:
    if attestation.get("claim") != CLAIM:
        raise DailySyncError("post-hoc attestation must use the limited integrity-only claim")
    limitations = attestation.get("limitations")
    if not isinstance(limitations, Mapping):
        raise DailySyncError("post-hoc attestation must state its limitations")
    if limitations.get("exact_execution_source") != UNKNOWN_EXECUTION_SOURCE:
        raise DailySyncError("post-hoc attestation cannot claim exact original execution source")
    if attestation.get("exact_execution_attested") is not None:
        raise DailySyncError("post-hoc attestation cannot contain exact_execution_attested")


def build_attestation(
    audit_path: Path,
    active_data_dir: Path,
    canonical_old_snapshot: Path,
    canonical_new_snapshot: Path,
) -> dict[str, Any]:
    audit_sha = sha256_file(audit_path)
    if audit_path.stem != audit_sha:
        raise DailySyncError("subject audit filename does not match its SHA256")
    audit = _load_json(audit_path)

    after_inventory = inspect_inventory(active_data_dir)
    expected_after_sha = str(audit["after_inventory"]["content_inventory_sha256"])
    if after_inventory["content_inventory_sha256"] != expected_after_sha:
        raise DailySyncError("active raw closure differs from the subject audit")

    original_old = Path(str(audit["old_raw_snapshot"]["path"]))
    original_new = Path(str(audit["new_raw_snapshot"]["path"]))
    old_inventory = inspect_inventory(original_old)
    new_inventory = inspect_inventory(original_new)
    expected_old_sha = str(audit["before_inventory"]["content_inventory_sha256"])
    if old_inventory["content_inventory_sha256"] != expected_old_sha:
        raise DailySyncError("original old snapshot differs from the subject audit")
    if new_inventory["content_inventory_sha256"] != expected_after_sha:
        raise DailySyncError("original new snapshot differs from the subject audit")

    canonical_old = inspect_v2_snapshot(canonical_old_snapshot)
    canonical_new = inspect_v2_snapshot(canonical_new_snapshot)
    if canonical_old["parquet_content_inventory_sha256"] != expected_old_sha:
        raise DailySyncError("canonical old snapshot has the wrong parquet closure")
    if canonical_new["parquet_content_inventory_sha256"] != expected_after_sha:
        raise DailySyncError("canonical new snapshot has the wrong parquet closure")

    updates = audit["updates"]
    if not isinstance(updates, list):
        raise DailySyncError("subject audit updates must be a list")
    after_hash_matches = 0
    before_hash_matches = 0
    for record in updates:
        if not isinstance(record, Mapping):
            raise DailySyncError("subject audit update record is malformed")
        active_path = active_data_dir / str(record["name"])
        if sha256_file(active_path) != record["after_sha256"]:
            raise DailySyncError(f"active update hash differs: {active_path.name}")
        after_hash_matches += 1
        if record.get("had_original"):
            old_path = original_old / str(record["name"])
            if sha256_file(old_path) != record["before_sha256"]:
                raise DailySyncError(f"old update hash differs: {old_path.name}")
            before_hash_matches += 1
    if after_hash_matches != int(audit["updated_file_count"]):
        raise DailySyncError("verified update count differs from the subject audit")

    api_objects = _flatten_api_objects(audit["api_response_objects"])
    full_qfq_objects = audit["full_qfq_refresh_objects"]
    if not isinstance(full_qfq_objects, list):
        raise DailySyncError("full qfq object registry must be a list")
    for record in api_objects:
        _verify_file_record(record)
    for record in full_qfq_objects:
        if not isinstance(record, Mapping):
            raise DailySyncError("full qfq object record is malformed")
        _verify_file_record({"path": record["object_path"], "sha256": record["object_sha256"]})

    daily_registry = audit["api_response_objects"]["daily"]
    factor_registry = audit["api_response_objects"]["adj_factor"]
    daily = pd.concat(
        [pd.read_csv(Path(str(record["path"]))) for record in daily_registry.values()],
        ignore_index=True,
    )
    factors = pd.concat(
        [pd.read_csv(Path(str(record["path"]))) for record in factor_registry.values()],
        ignore_index=True,
    )
    daily["trade_date"] = daily["trade_date"].astype(str)
    factors["trade_date"] = factors["trade_date"].astype(str)
    trade_dates = [str(value) for value in audit["trade_dates"]]
    existing_symbols = {path.stem for path in original_old.glob("*.parquet")}
    daily_symbols = set(daily["ts_code"].astype(str))
    new_symbols = sorted(daily_symbols - existing_symbols)
    if new_symbols != audit["new_symbols"]:
        raise DailySyncError("recomputed new-symbol routing differs from the subject audit")
    factor_refresh = identify_factor_refresh_symbols(factors, trade_dates, existing_symbols)
    if factor_refresh != audit["factor_refresh_symbols"]:
        raise DailySyncError("recomputed factor-change routing differs from the subject audit")
    daily_existing = daily_symbols & existing_symbols
    factors_by_symbol = {
        str(symbol): set(group["trade_date"])
        for symbol, group in factors.loc[factors["ts_code"].isin(daily_existing)].groupby("ts_code")
    }
    missing_factor = sorted(
        symbol for symbol in daily_existing if factors_by_symbol.get(symbol, set()) != set(trade_dates)
    )
    if missing_factor != audit["missing_factor_symbols"]:
        raise DailySyncError("recomputed missing-factor routing differs from the subject audit")
    full_refresh = sorted(set(new_symbols) | set(factor_refresh) | set(missing_factor))
    recorded_full_refresh = sorted(str(record["ts_code"]) for record in full_qfq_objects)
    if full_refresh != recorded_full_refresh:
        raise DailySyncError("recomputed full-qfq routing differs from the frozen object registry")

    verifier_path = Path(__file__).resolve()
    supporting_path = verifier_path.with_name("_sync_daily_data.py")
    attestation = {
        "schema": "a_stock_daily_qfq_posthoc_attestation_v1",
        "claim": CLAIM,
        "subject_audit_sha256": audit_sha,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_now": {
            "audit_self_hash": True,
            "active_after_closure_match": True,
            "original_snapshot_parquet_closures_match": True,
            "canonical_v2_snapshots": {
                "old": canonical_old,
                "new": canonical_new,
            },
            "updated_after_file_hashes_match": after_hash_matches,
            "updated_before_file_hashes_match": before_hash_matches,
            "api_and_full_qfq_object_hashes_match": len(api_objects) + len(full_qfq_objects),
            "factor_routing_recomputed": {
                "changed": len(factor_refresh),
                "new": len(new_symbols),
                "missing_factor": len(missing_factor),
                "full_refresh": len(full_refresh),
            },
        },
        "verification_method": {
            "verifier_source_sha256": sha256_file(verifier_path),
            "supporting_source_sha256": sha256_file(supporting_path),
            "python": sys.version.split()[0],
        },
        "limitations": {
            "exact_execution_source": UNKNOWN_EXECUTION_SOURCE,
            "exact_execution_parameters": "PARTIAL_FROM_SUBJECT_AUDIT_ONLY",
            "does_not_attest": [
                "exact originally executed source bytes",
                "exact loaded dependency bytes",
                "original process or crash history",
            ],
        },
    }
    validate_attestation_claims(attestation)
    return attestation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--active-data-dir", type=Path, required=True)
    parser.add_argument("--canonical-old-snapshot", type=Path, required=True)
    parser.add_argument("--canonical-new-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    attestation = build_attestation(
        args.audit.expanduser().resolve(),
        args.active_data_dir.expanduser().resolve(),
        args.canonical_old_snapshot.expanduser().resolve(),
        args.canonical_new_snapshot.expanduser().resolve(),
    )
    raw = canonical_json(attestation) + b"\n"
    digest = sha256_bytes(raw)
    output_path = args.output_dir.expanduser().resolve() / f"{digest}.json"
    atomic_write_bytes(output_path, raw, replace_existing=False)
    print(
        json.dumps(
            {
                "attestation_path": str(output_path),
                "attestation_sha256": digest,
                "claim": CLAIM,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
