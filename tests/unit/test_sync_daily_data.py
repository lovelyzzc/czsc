from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _sync_daily_data as sync  # noqa: E402
from _attest_daily_sync import validate_attestation_claims  # noqa: E402


def _frame(symbol: str, dates: list[str], closes: list[float] | None = None) -> pd.DataFrame:
    selected_closes = closes or [10.0 + index for index in range(len(dates))]
    rows = []
    for index, (trade_date, close) in enumerate(zip(dates, selected_closes, strict=True)):
        previous = selected_closes[index - 1] if index else close
        pct_chg = 0.0 if index == 0 else (close / previous - 1.0) * 100.0
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "pre_close": previous,
                "change": close - previous,
                "pct_chg": pct_chg,
                "vol": 100.0,
                "amount": 1_000.0,
            }
        )
    return pd.DataFrame(rows, columns=sync.DAILY_COLUMNS)


def _write_symbol(root: Path, symbol: str, dates: list[str], closes: list[float] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol}.parquet"
    _frame(symbol, dates, closes).to_parquet(path, index=False)
    return path


def _execution_binding(data_dir: Path, snapshot_root: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "data_dir": data_dir,
        "snapshot_root": snapshot_root,
        "end_date": "20260727",
        "min_daily_rows": 1,
        "new_symbol_sleep_seconds": 0.0,
        "apply": False,
    }
    values.update(overrides)
    return sync.build_execution_binding(
        Namespace(**values),
        data_dir,
        snapshot_root,
        "20260727",
        require_clean_and_pushed=False,
    )


def test_inventory_scans_every_parquet_not_only_first_ten(tmp_path: Path) -> None:
    for index in range(11):
        symbol = f"{index:06d}.SZ"
        end = "20260709" if index == 10 else "20260708"
        _write_symbol(tmp_path, symbol, ["20260707", end])

    inventory = sync.inspect_inventory(tmp_path)

    assert inventory["file_count"] == 11
    assert inventory["max_dt"] == "20260709"
    assert len(inventory["records"]) == 11
    assert {row["name"] for row in inventory["records"]} == {f"{index:06d}.SZ.parquet" for index in range(11)}


def test_active_manifest_only_matches_exact_current_closure(tmp_path: Path) -> None:
    _write_symbol(tmp_path, "000001.SZ", ["20260727"])
    inventory = sync.inspect_inventory(tmp_path)
    sync.atomic_write_json(
        tmp_path / "manifest.json",
        {
            "schema": "a_stock_daily_qfq_active_snapshot_v1",
            "safe_end_date": "20260727",
            "content_inventory_sha256": inventory["content_inventory_sha256"],
            "file_count": inventory["file_count"],
        },
    )

    assert sync.active_manifest_matches(tmp_path, inventory, "20260727")
    assert not sync.active_manifest_matches(tmp_path, inventory, "20260728")

    _write_symbol(tmp_path, "000001.SZ", ["20260727", "20260728"])
    mutated = sync.inspect_inventory(tmp_path)
    assert not sync.active_manifest_matches(tmp_path, mutated, "20260727")


def test_inclusive_overlap_remote_row_replaces_local() -> None:
    local = _frame("000001.SZ", ["20260708", "20260709"], [10.0, 999.0])
    incoming = _frame("000001.SZ", ["20260709", "20260710"], [11.0, 12.0])

    merged = sync.merge_incremental_rows(local, incoming, "000001.SZ")

    assert merged["trade_date"].tolist() == ["20260708", "20260709", "20260710"]
    assert merged.loc[merged["trade_date"].eq("20260709"), "close"].item() == 11.0
    assert merged.loc[merged["trade_date"].eq("20260708"), "close"].item() == 10.0


def test_any_factor_change_or_missing_date_requires_full_refresh() -> None:
    factors = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260708", "adj_factor": 1.0},
            {"ts_code": "000001.SZ", "trade_date": "20260709", "adj_factor": 1.0},
            {"ts_code": "000002.SZ", "trade_date": "20260708", "adj_factor": 2.0},
            {"ts_code": "000002.SZ", "trade_date": "20260709", "adj_factor": 2.000001},
            {"ts_code": "000003.SZ", "trade_date": "20260709", "adj_factor": 3.0},
        ]
    )

    refresh = sync.identify_factor_refresh_symbols(
        factors,
        ["20260708", "20260709"],
        {"000001.SZ", "000002.SZ", "000003.SZ"},
    )

    assert refresh == ["000002.SZ", "000003.SZ"]


def test_full_qfq_allows_reference_nulls_only_on_earliest_row() -> None:
    frame = _frame("920576.BJ", ["20210428", "20210610"], [8.63, 8.63])
    frame.loc[0, ["pre_close", "change", "pct_chg"]] = [float("nan")] * 3

    validated = sync._validate_full_qfq(frame.iloc[::-1], "920576.BJ", "20260727")

    assert validated["trade_date"].tolist() == ["20210428", "20210610"]
    assert validated.loc[0, ["pre_close", "change", "pct_chg"]].isna().all()

    contaminated = frame.copy()
    contaminated.loc[1, "pre_close"] = float("nan")
    with pytest.raises(sync.DailySyncError, match="outside the earliest row"):
        sync._validate_full_qfq(contaminated, "920576.BJ", "20260727")


def test_full_qfq_resume_cache_is_query_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _frame("000001.SZ", ["20260727"])
    monkeypatch.setattr(sync, "download_one", lambda *_: history.copy())
    _, first_evidence = sync.fetch_new_symbol_histories(
        ["000001.SZ"],
        "20260727",
        sleep_seconds=0,
        expected_max_dates={"000001.SZ": "20260727"},
        object_root=tmp_path,
    )
    assert first_evidence[0]["reused"] is False

    def unexpected_download(*_: object) -> pd.DataFrame:
        raise AssertionError("hash-bound cache should be reused without a network call")

    monkeypatch.setattr(sync, "download_one", unexpected_download)
    _, second_evidence = sync.fetch_new_symbol_histories(
        ["000001.SZ"],
        "20260727",
        sleep_seconds=0,
        expected_max_dates={"000001.SZ": "20260727"},
        object_root=tmp_path,
    )
    assert second_evidence[0]["reused"] is True

    cache_path = tmp_path / "full_qfq" / "20260727" / "000001.SZ.parquet"
    contaminated = history.copy()
    contaminated.loc[0, "close"] = 99.0
    contaminated.to_parquet(cache_path, index=False)
    with pytest.raises(sync.DailySyncError, match="cache object hash differs"):
        sync.fetch_new_symbol_histories(
            ["000001.SZ"],
            "20260727",
            sleep_seconds=0,
            expected_max_dates={"000001.SZ": "20260727"},
            object_root=tmp_path,
        )


def test_qfq_seam_is_repaired_only_in_staging(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    live = _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
    before_bytes = live.read_bytes()
    incoming = _frame("000001.SZ", ["20260709"], [5.0])
    incoming.loc[0, ["pre_close", "change", "pct_chg"]] = [5.0, 0.0, 0.0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    updates, seams_fixed = sync.stage_updates(data_dir, incoming, {}, run_dir)

    staged = run_dir / "staged" / live.name
    repaired = pd.read_parquet(staged)
    assert seams_fixed == 1
    assert updates[0]["qfq_seams_fixed"] == 1
    assert sync.find_seams(staged) == []
    assert repaired.loc[0, "close"] == 5.0
    assert live.read_bytes() == before_bytes


def test_new_symbol_without_full_qfq_history_fails_before_publish(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    live = _write_symbol(data_dir, "000001.SZ", ["20260708"])
    before_bytes = live.read_bytes()
    daily = pd.concat(
        [
            _frame("000001.SZ", ["20260709"]),
            _frame("000002.SZ", ["20260709"]),
        ],
        ignore_index=True,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(sync.DailySyncError, match="new symbols have no full qfq history"):
        sync.stage_updates(data_dir, daily, {}, run_dir)

    assert live.read_bytes() == before_bytes
    assert not (data_dir / "000002.SZ.parquet").exists()


def test_raw_snapshot_is_independent_and_content_addressed(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    first = _write_symbol(data_dir, "000001.SZ", ["20260708"])
    _write_symbol(data_dir, "000002.SZ", ["20260708"])
    (data_dir / "manifest.json").write_text('{"legacy": true}\n', encoding="utf-8")
    inventory = sync.inspect_inventory(data_dir)
    snapshot_root = tmp_path / "snapshots"

    snapshot, archived = sync.archive_raw_snapshot(data_dir, inventory, snapshot_root)

    assert snapshot.name == f"RAW_{archived['snapshot_payload_sha256']}"
    assert archived["content_inventory_sha256"] == inventory["content_inventory_sha256"]
    assert (snapshot / "manifest.json").read_text(encoding="utf-8") == '{"legacy": true}\n'
    snapshot_manifest = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert snapshot_manifest["schema"] == "a_stock_daily_qfq_raw_snapshot_v2"
    assert snapshot_manifest["closure_sha256"] == archived["snapshot_payload_sha256"]
    assert snapshot_manifest["parquet_content_inventory_sha256"] == inventory["content_inventory_sha256"]
    first.unlink()
    assert (snapshot / first.name).is_file()
    assert sync.sha256_file(snapshot / first.name) == inventory["records"][0]["sha256"]


def test_raw_snapshot_identity_includes_auxiliary_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_parquet = _write_symbol(first, "000001.SZ", ["20260708"])
    second.mkdir()
    (second / first_parquet.name).write_bytes(first_parquet.read_bytes())
    (first / "manifest.json").write_text('{"version": "A"}\n', encoding="utf-8")
    (second / "manifest.json").write_text('{"version": "B"}\n', encoding="utf-8")
    first_inventory = sync.inspect_inventory(first)
    second_inventory = sync.inspect_inventory(second)
    assert first_inventory["content_inventory_sha256"] == second_inventory["content_inventory_sha256"]

    snapshot_root = tmp_path / "snapshots"
    first_snapshot, _ = sync.archive_raw_snapshot(first, first_inventory, snapshot_root)
    second_snapshot, _ = sync.archive_raw_snapshot(second, second_inventory, snapshot_root)

    assert first_snapshot != second_snapshot
    assert (first_snapshot / "manifest.json").read_text(encoding="utf-8") == '{"version": "A"}\n'
    assert (second_snapshot / "manifest.json").read_text(encoding="utf-8") == '{"version": "B"}\n'


def test_atomic_write_json_syncs_the_final_directory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(sync, "fsync_directory", lambda path: synced.append(path))

    target = tmp_path / "journal.json"
    sync.atomic_write_json(target, {"phase": "PREPARED"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"phase": "PREPARED"}
    assert tmp_path in synced


def test_execution_binding_changes_when_a_result_parameter_changes(tmp_path: Path) -> None:
    first = _execution_binding(tmp_path / "live", tmp_path / "snapshots", min_daily_rows=1)
    second = _execution_binding(tmp_path / "live", tmp_path / "snapshots", min_daily_rows=2)

    assert first["binding_sha256"] != second["binding_sha256"]
    full_qfq_retry = first["binding"]["parameters"]["retry_policy"]["full_qfq"]  # type: ignore[index]
    assert full_qfq_retry == {
        "max_attempts": 4,
        "exponential_backoff_seconds": [1, 2, 4],
        "backoff_cap_seconds": 8,
    }


def test_posthoc_attestation_rejects_an_exact_execution_claim() -> None:
    payload = {
        "claim": "POST_HOC_ARTIFACT_INTEGRITY_ONLY",
        "exact_execution_attested": True,
        "limitations": {"exact_execution_source": "UNKNOWN_NOT_ATTESTED"},
    }

    with pytest.raises(sync.DailySyncError, match="cannot contain exact_execution_attested"):
        validate_attestation_claims(payload)


def test_recovery_refuses_to_delete_a_candidate_without_a_journal(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    _write_symbol(data_dir, "000001.SZ", ["20260708"])
    candidate = tmp_path / f"{sync.RUN_PREFIX}orphan" / "candidate_active"
    _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"])

    with pytest.raises(sync.DailySyncError, match="no durable publication journal"):
        sync.recover_orphaned_runs(data_dir)

    assert data_dir.is_dir()
    assert candidate.is_dir()


def test_directory_exchange_syncs_both_parent_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_parent_fsync_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        left = root / "active_parent" / "live"
        right = root / "run_parent" / "candidate"
        _write_symbol(left, "000001.SZ", ["20260708"])
        _write_symbol(right, "000001.SZ", ["20260709"])
        synced: list[Path] = []
        monkeypatch.setattr(sync, "fsync_directory", lambda path: synced.append(path))

        sync.atomic_exchange_directories(left, right)

        assert synced == [left.parent, right.parent]


def test_directory_exchange_publication_rolls_back_after_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pytest's configured tmp_path is on DrvFS here, which deliberately rejects
    # renameat2(RENAME_EXCHANGE); use the same Linux filesystem as the real cache.
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_exchange_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / "run" / "candidate_active"
        run_dir = candidate.parent
        old = _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
        _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        old_bytes = old.read_bytes()
        before = sync.inspect_inventory(data_dir)
        after = sync.inspect_inventory(candidate)
        audit_payload = {
            "before_inventory": {key: value for key, value in before.items() if key != "records"},
            "expected_active_inventory_sha256": after["content_inventory_sha256"],
            "execution_binding": _execution_binding(data_dir, root / "snapshots"),
        }
        real_replace = os.replace

        def fail_audit_replace(source: str | bytes | os.PathLike[str] | os.PathLike[bytes], target: object) -> None:
            target_path = Path(target)  # type: ignore[arg-type]
            if target_path.parent.name == sync.AUDIT_DIR_NAME:
                raise OSError("injected audit publication failure")
            real_replace(source, target)  # type: ignore[arg-type]

        monkeypatch.setattr(sync.os, "replace", fail_audit_replace)

        with pytest.raises(OSError, match="injected audit publication failure"):
            sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

        assert (data_dir / old.name).read_bytes() == old_bytes
        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == before["content_inventory_sha256"]
        assert sync.inspect_inventory(candidate)["content_inventory_sha256"] == after["content_inventory_sha256"]


def test_directory_exchange_rolls_back_when_execution_sources_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_binding_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / "run" / "candidate_active"
        run_dir = candidate.parent
        old = _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
        _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        old_bytes = old.read_bytes()
        before = sync.inspect_inventory(data_dir)
        after = sync.inspect_inventory(candidate)
        audit_payload = {
            "before_inventory": {key: value for key, value in before.items() if key != "records"},
            "expected_active_inventory_sha256": after["content_inventory_sha256"],
            "execution_binding": _execution_binding(data_dir, root / "snapshots"),
        }
        checks = 0

        def fail_after_exchange(_binding: object) -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise sync.DailySyncError("execution source bytes changed during synchronization")

        monkeypatch.setattr(sync, "assert_execution_binding_current", fail_after_exchange)

        with pytest.raises(sync.DailySyncError, match="execution source bytes changed"):
            sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

        assert checks == 2
        assert (data_dir / old.name).read_bytes() == old_bytes
        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == before["content_inventory_sha256"]
