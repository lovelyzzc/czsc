"""Point-in-time exact-mcap collector integrity and resume tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_pit_exact_mcap_collector as collector  # noqa: E402

DATES = ("2024-01-02", "2024-01-03")
REQUESTS = (("000001.SZ", DATES[0]), ("000002.SZ", DATES[1]))


def _request_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    request_records = [list(record) for record in REQUESTS]
    dates = list(DATES)
    symbols = sorted({record[0] for record in REQUESTS})
    manifest = {
        "schema": collector.REQUEST_SCHEMA,
        "canonicalization": "test fixture uses producer canonical identities",
        "cohort": {
            "treated_trades": len(request_records),
            "decision_dates": len(dates),
            "year_counts": {"2024": len(request_records)},
            "missing_frozen_industry": [],
        },
        "request_identity": collector.payload_identity(request_records),
        "request_records": request_records,
        "treated_request_identity": collector.payload_identity(request_records),
        "treated_request_records": request_records,
        "request_dates_identity": collector.payload_identity(dates),
        "request_dates": dates,
        "request_symbols_identity": collector.payload_identity(symbols),
        "request_symbols": symbols,
        "closure": {
            "request_keys_unique": True,
            "treated_keys_unique": True,
            "all_treated_requested": True,
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _cross_section(date_text: str, *, rows: int = collector.MIN_CROSS_SECTION_ROWS) -> pd.DataFrame:
    requested = "000001.SZ" if date_text == DATES[0] else "000002.SZ"
    symbols = [requested] + [f"{index:06d}.SH" for index in range(100_000, 100_000 + rows - 1)]
    return pd.DataFrame(
        {
            "ts_code": symbols,
            "trade_date": [date_text.replace("-", "")] * rows,
            "close": np.arange(rows, dtype=float) / 100 + 1,
            "total_share": np.full(rows, 1_000.0),
            "float_share": np.full(rows, 800.0),
            "total_mv": np.full(rows, 10_000.0),
            "circ_mv": np.arange(rows, dtype=float) + 12_345.0,
        }
    )


class FakePro:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, str]] = []

    def daily_basic(self, *, trade_date: str, fields: str) -> pd.DataFrame:
        self.calls.append((trade_date, fields))
        return self.frames[trade_date].copy()


def _frames() -> dict[str, pd.DataFrame]:
    return {date.replace("-", ""): _cross_section(date) for date in DATES}


def _collect(tmp_path: Path, pro: Any | None = None) -> tuple[Path, Path, Any, dict[str, Any]]:
    request_path = _request_manifest(tmp_path / "request.json")
    cache_root = tmp_path / "cache"
    client = pro if pro is not None else FakePro(_frames())
    result = collector.collect_exact_mcap(
        request_path=request_path,
        cache_root=cache_root,
        pro=client,
        rate_hz=0,
        max_attempts=1,
    )
    return request_path, cache_root, client, result


def _downgrade_to_legacy(cache_root: Path, result: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = cache_root / "manifests" / f"exact_mcap_{result['request']['sha256']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    portable_path = cache_root / manifest.pop("portable_request")["path"]
    portable_path.unlink()
    manifest["provenance"].pop("request_file_sha_is_local_diagnostic")
    manifest["request"].pop("dates_sha256")
    manifest["request"].pop("symbols_sha256")
    collector._atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def test_canonical_object_is_invariant_to_provider_row_order() -> None:
    frame = _cross_section(DATES[0])
    first, _ = collector.normalise_cross_section(
        frame,
        trade_date=DATES[0],
        requested_symbols={"000001.SZ"},
    )
    second, _ = collector.normalise_cross_section(
        frame.sample(frac=1, random_state=42),
        trade_date=DATES[0],
        requested_symbols={"000001.SZ"},
    )

    assert collector.canonical_json(first) == collector.canonical_json(second)


def test_collection_materializes_explicit_units_and_portable_verified_manifest(tmp_path: Path) -> None:
    request_path, cache_root, client, result = _collect(tmp_path)
    materialized = pd.read_parquet(cache_root / result["materialized"]["path"])

    assert len(client.calls) == len(DATES)
    assert client.calls[0][1] == ",".join(collector.DAILY_BASIC_FIELDS)
    assert list(materialized.columns) == list(collector.MATERIALIZED_COLUMNS)
    assert materialized["exact_circ_mv_cny"].equals(materialized["circ_mv_10k_cny"] * 10_000)
    assert materialized["source"].eq(collector.SOURCE).all()
    assert not Path(result["materialized"]["path"]).is_absolute()
    provenance = result["provenance"]
    assert provenance["request_manifest_file_sha256"] == collector.sha256_file(request_path)
    assert (
        provenance["query_canonical_sha256"]
        == collector.query_identity(collector.load_request_manifest(request_path))["sha256"]
    )
    assert provenance["credential_recorded"] is False
    assert set(provenance["runtime_versions"]) == {"python", "pandas", "tinyshare", "tushare"}
    assert "revision-vintage" in provenance["historical_exactness"]
    assert collector.verify_collection(request_path=request_path, cache_root=cache_root)["verified"] is True


def test_non_requested_nulls_are_diagnostics_only() -> None:
    frame = _cross_section(DATES[0])
    frame.loc[1, "float_share"] = np.nan

    _, diagnostics = collector.normalise_cross_section(
        frame,
        trade_date=DATES[0],
        requested_symbols={"000001.SZ"},
    )

    assert diagnostics["non_requested_null_counts"] == {"float_share": 1}


def test_valid_resume_makes_zero_provider_calls(tmp_path: Path) -> None:
    request_path, cache_root, _, first = _collect(tmp_path)
    no_call = FakePro({})

    second = collector.collect_exact_mcap(
        request_path=request_path,
        cache_root=cache_root,
        pro=no_call,
        rate_hz=0,
        max_attempts=1,
    )

    assert no_call.calls == []
    assert second["materialized"]["sha256"] == first["materialized"]["sha256"]


def test_verify_accepts_same_canonical_request_with_different_physical_json(tmp_path: Path) -> None:
    request_path, cache_root, _, result = _collect(tmp_path)
    collected_file_sha = result["provenance"]["request_manifest_file_sha256"]
    request_document = json.loads(request_path.read_text(encoding="utf-8"))
    request_document["local_exact_cache"] = {
        "available": True,
        "path": "/different-device/local-only/cache.parquet",
    }
    request_path.write_text(json.dumps(request_document, ensure_ascii=False, indent=4), encoding="utf-8")

    assert collector.sha256_file(request_path) != collected_file_sha
    report = collector.verify_collection(request_path=request_path, cache_root=cache_root)
    assert report["verified"] is True
    assert report["request_file_sha_is_local_diagnostic"] is True
    portable_path = cache_root / report["artifacts"]["portable_request"]
    assert not Path(report["artifacts"]["portable_request"]).is_absolute()
    assert "local_exact_cache" not in json.loads(portable_path.read_text(encoding="utf-8"))


def test_legacy_manifest_migration_is_atomic_and_does_not_rewrite_data(tmp_path: Path) -> None:
    request_path, cache_root, _, result = _collect(tmp_path)
    manifest_path, _ = _downgrade_to_legacy(cache_root, result)
    parquet_path = cache_root / result["materialized"]["path"]
    parquet_sha_before = collector.sha256_file(parquet_path)
    object_shas_before = {
        entry["object_path"]: collector.sha256_file(cache_root / entry["object_path"]) for entry in result["objects"]
    }

    migration = collector.upgrade_portable_manifest(request_path=request_path, cache_root=cache_root)

    assert migration["upgraded"] is True
    assert migration["manifest_sha256"] == collector.sha256_file(manifest_path)
    assert parquet_sha_before == collector.sha256_file(parquet_path)
    assert object_shas_before == {path: collector.sha256_file(cache_root / path) for path in object_shas_before}
    assert collector.verify_collection(request_path=request_path, cache_root=cache_root)["verified"] is True


def test_legacy_manifest_migration_fails_before_replacement_on_projection_tamper(tmp_path: Path) -> None:
    request_path, cache_root, _, result = _collect(tmp_path)
    manifest_path, legacy = _downgrade_to_legacy(cache_root, result)
    parquet_path = cache_root / legacy["materialized"]["path"]
    frame = pd.read_parquet(parquet_path)
    frame.loc[0, "close_cny"] += 1.0
    frame.to_parquet(parquet_path, index=False)
    legacy["materialized"]["sha256"] = collector.sha256_file(parquet_path)
    collector._atomic_json(manifest_path, legacy)
    legacy_bytes = manifest_path.read_bytes()

    with pytest.raises(collector.VerificationError, match="materialized/object value mismatch in close_cny"):
        collector.upgrade_portable_manifest(request_path=request_path, cache_root=cache_root)

    assert manifest_path.read_bytes() == legacy_bytes


def test_corrupt_object_is_refetched_without_refetching_valid_date(tmp_path: Path) -> None:
    request_path, cache_root, _, result = _collect(tmp_path)
    corrupt = result["objects"][1]
    (cache_root / corrupt["object_path"]).write_bytes(b"tampered")
    resumed = FakePro(_frames())

    collector.collect_exact_mcap(
        request_path=request_path,
        cache_root=cache_root,
        pro=resumed,
        rate_hz=0,
        max_attempts=1,
    )

    assert [call[0] for call in resumed.calls] == [DATES[1].replace("-", "")]
    assert collector.verify_collection(request_path=request_path, cache_root=cache_root)["verified"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_date", "wrong or invalid trade_date"),
        ("missing_field", "missing fields"),
        ("duplicate", "duplicate symbols"),
        ("too_small", "fewer than"),
        ("too_large", "at least"),
        ("missing_requested", "requested symbols are missing"),
        ("invalid_circ_mv", "invalid circ_mv"),
    ],
)
def test_per_date_hard_gates(mutation: str, message: str) -> None:
    frame = _cross_section(DATES[0])
    if mutation == "wrong_date":
        frame.loc[0, "trade_date"] = "20240103"
    elif mutation == "missing_field":
        frame = frame.drop(columns="total_mv")
    elif mutation == "duplicate":
        frame.loc[1, "ts_code"] = frame.loc[0, "ts_code"]
    elif mutation == "too_small":
        frame = frame.iloc[:-1]
    elif mutation == "too_large":
        frame = _cross_section(DATES[0], rows=collector.MAX_CROSS_SECTION_ROWS_EXCLUSIVE)
    elif mutation == "missing_requested":
        frame.loc[0, "ts_code"] = "999999.SH"
    elif mutation == "invalid_circ_mv":
        frame.loc[0, "circ_mv"] = np.nan

    with pytest.raises(collector.CrossSectionError, match=message):
        collector.normalise_cross_section(
            frame,
            trade_date=DATES[0],
            requested_symbols={"000001.SZ"},
        )


def test_tamper_rejection_and_failure_does_not_publish_final_artifacts(tmp_path: Path) -> None:
    request_path, cache_root, _, result = _collect(tmp_path / "complete")
    object_path = cache_root / result["objects"][0]["object_path"]
    object_path.write_bytes(object_path.read_bytes() + b" ")
    with pytest.raises(collector.VerificationError, match="hash mismatch"):
        collector.verify_collection(request_path=request_path, cache_root=cache_root, write_report=False)

    failed_root = tmp_path / "failed" / "cache"
    failed_request = _request_manifest(tmp_path / "failed_request.json")
    bad_frames = _frames()
    bad_frames[DATES[1].replace("-", "")] = bad_frames[DATES[1].replace("-", "")].iloc[:-1]
    with pytest.raises(collector.CollectionError):
        collector.collect_exact_mcap(
            request_path=failed_request,
            cache_root=failed_root,
            pro=FakePro(bad_frames),
            rate_hz=0,
            max_attempts=1,
        )
    assert not (failed_root / "materialized").exists()
    assert not (failed_root / "manifests").exists()
    assert not (failed_root / "reports").exists()


@pytest.mark.parametrize("column", ["close_cny", "circ_mv_10k_cny"])
def test_parquet_tamper_rejected_even_when_manifest_file_hash_is_updated(tmp_path: Path, column: str) -> None:
    request_path, cache_root, _, result = _collect(tmp_path)
    parquet_path = cache_root / result["materialized"]["path"]
    frame = pd.read_parquet(parquet_path)
    frame.loc[0, column] += 1.0
    if column == "circ_mv_10k_cny":
        frame.loc[0, "exact_circ_mv_cny"] = frame.loc[0, column] * 10_000
    frame.to_parquet(parquet_path, index=False)

    manifest_path = cache_root / "manifests" / f"exact_mcap_{result['request']['sha256']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["materialized"]["sha256"] = collector.sha256_file(parquet_path)
    collector._atomic_json(manifest_path, manifest)

    with pytest.raises(collector.VerificationError, match=f"materialized/object value mismatch in {column}"):
        collector.verify_collection(request_path=request_path, cache_root=cache_root, write_report=False)


def test_v2_manifest_identity_tampering_is_rejected(tmp_path: Path) -> None:
    request_path = _request_manifest(tmp_path / "request.json")
    manifest = json.loads(request_path.read_text(encoding="utf-8"))
    manifest["request_identity"]["count"] += 1
    request_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(collector.ManifestError, match="request_identity"):
        collector.load_request_manifest(request_path)
