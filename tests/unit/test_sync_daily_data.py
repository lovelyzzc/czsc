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


def _source_frames(
    symbols_by_date: dict[str, set[str]],
    *,
    factor_only_by_date: dict[str, set[str]] | None = None,
    factor_missing_by_date: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_only_by_date = factor_only_by_date or {}
    factor_missing_by_date = factor_missing_by_date or {}
    daily_rows = [
        {"ts_code": symbol, "trade_date": trade_date}
        for trade_date, symbols in symbols_by_date.items()
        for symbol in sorted(symbols)
    ]
    factor_rows = [
        {
            "ts_code": symbol,
            "trade_date": trade_date,
            "adj_factor": 1.0,
        }
        for trade_date, symbols in symbols_by_date.items()
        for symbol in sorted(
            (symbols - factor_missing_by_date.get(trade_date, set())) | factor_only_by_date.get(trade_date, set())
        )
    ]
    return pd.DataFrame(daily_rows), pd.DataFrame(factor_rows)


def _test_source_completeness_report() -> dict[str, object]:
    baseline = {f"{index:06d}.SZ" for index in range(sync.SOURCE_SYMBOL_ABSOLUTE_MINIMUM)}
    daily, factors = _source_frames({"20260727": baseline})
    return sync.validate_source_symbol_completeness(
        daily,
        factors,
        ["20260727"],
        previous_trade_date="20260727",
        previous_symbols=baseline,
    )


def _execution_binding(data_dir: Path, snapshot_root: Path, **overrides: object) -> dict[str, object]:
    source_symbol_completeness = (
        overrides.pop("source_symbol_completeness")
        if "source_symbol_completeness" in overrides
        else _test_source_completeness_report()
    )
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
        source_symbol_completeness=source_symbol_completeness,  # type: ignore[arg-type]
    )


def _formalize_execution_binding(binding: dict[str, object]) -> dict[str, object]:
    formal = json.loads(json.dumps(binding))
    formal["binding"]["parameters"]["apply"] = True
    git = formal["binding"]["git"]
    git["branch"] = sync.REQUIRED_GIT_BRANCH
    git["upstream"] = sync.REQUIRED_GIT_UPSTREAM
    git["upstream_head"] = git["head"]
    git["remote_name"] = sync.REQUIRED_GIT_REMOTE_NAME
    git["remote_ref"] = sync.REQUIRED_GIT_REMOTE_REF
    git["remote_head"] = git["head"]
    git["remote_fetch_url"] = sync.REQUIRED_GIT_REMOTE_URL
    git["remote_push_url"] = sync.REQUIRED_GIT_REMOTE_URL
    git["remote_verified"] = True
    git["worktree_clean"] = True
    formal["binding_sha256"] = sync.sha256_bytes(sync.canonical_json(formal["binding"]))
    return formal


def _full_daily_source_frame(symbols: set[str], trade_date: str) -> pd.DataFrame:
    ordered = sorted(symbols)
    payload: dict[str, object] = {
        "ts_code": ordered,
        "trade_date": [trade_date] * len(ordered),
    }
    for column in sync.NUMERIC_COLUMNS:
        payload[column] = [1.0] * len(ordered)
    return pd.DataFrame(payload, columns=sync.DAILY_COLUMNS)


def _frozen_source_fixture(
    data_dir: Path,
    symbols: set[str],
    trade_date: str = "20260727",
) -> tuple[dict[str, object], dict[str, object]]:
    daily = _full_daily_source_frame(symbols, trade_date)
    factors = pd.DataFrame(
        {
            "ts_code": sorted(symbols),
            "trade_date": [trade_date] * len(symbols),
            "adj_factor": [1.0] * len(symbols),
        }
    )
    calendar = pd.DataFrame(
        [
            {
                "exchange": "SSE",
                "cal_date": trade_date,
                "is_open": 1,
                "pretrade_date": "20260724",
            }
        ]
    )
    objects = sync.freeze_api_response_objects(
        data_dir.parent / sync.API_OBJECT_DIR_NAME,
        calendar,
        daily,
        factors,
    )
    audit_fragment: dict[str, object] = {
        "trade_dates": [trade_date],
        "official_calendar_response": {
            "rows": objects["official_calendar"]["rows"],
            "canonical_csv_sha256": objects["official_calendar"]["sha256"],
        },
        "daily_responses": [
            {
                "trade_date": trade_date,
                "rows": objects["daily"][trade_date]["rows"],
                "canonical_csv_sha256": objects["daily"][trade_date]["sha256"],
            }
        ],
        "adjustment_factor_responses": [
            {
                "trade_date": trade_date,
                "rows": objects["adj_factor"][trade_date]["rows"],
                "canonical_csv_sha256": objects["adj_factor"][trade_date]["sha256"],
            }
        ],
        "api_response_objects": objects,
    }
    report = sync.validate_source_symbol_completeness(
        daily,
        factors,
        [trade_date],
        previous_trade_date=trade_date,
        previous_symbols=symbols,
    )
    return audit_fragment, report


def _patch_clean_git(
    monkeypatch: pytest.MonkeyPatch,
    execution_binding: dict[str, object],
    *,
    remote_head: str | None = None,
) -> None:
    real_git_output = sync._git_output
    head = execution_binding["binding"]["git"]["head"]  # type: ignore[index]

    def clean_git_output(repo_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return str(head)
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if args == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return sync.REQUIRED_GIT_BRANCH
        if args == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return sync.REQUIRED_GIT_UPSTREAM
        if args == ("rev-parse", "@{upstream}"):
            return str(head)
        if args == ("remote", "get-url", sync.REQUIRED_GIT_REMOTE_NAME):
            return sync.REQUIRED_GIT_REMOTE_URL
        if args == ("remote", "get-url", "--push", sync.REQUIRED_GIT_REMOTE_NAME):
            return sync.REQUIRED_GIT_REMOTE_URL
        if args == (
            "ls-remote",
            "--heads",
            sync.REQUIRED_GIT_REMOTE_NAME,
            sync.REQUIRED_GIT_REMOTE_REF,
        ):
            resolved_remote_head = str(head) if remote_head is None else remote_head
            return f"{resolved_remote_head}\t{sync.REQUIRED_GIT_REMOTE_REF}"
        return real_git_output(repo_root, *args)

    monkeypatch.setattr(sync, "_git_output", clean_git_output)


def _formal_publication_payload(
    data_dir: Path,
    before: dict[str, object],
    after: dict[str, object],
    snapshot_root: Path,
    frozen_audit: dict[str, object],
    report: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    execution_binding = _formalize_execution_binding(
        _execution_binding(
            data_dir,
            snapshot_root,
            apply=True,
            min_daily_rows=1_000,
            source_symbol_completeness=report,
        )
    )
    audit_payload: dict[str, object] = {
        "schema": "a_stock_daily_qfq_sync_audit_v2",
        "data_dir": str(data_dir.resolve()),
        "requested_end_date": "20260727",
        "safe_end_date": "20260727",
        "overlap_start_date": "20260727",
        "before_inventory": {key: value for key, value in before.items() if key != "records"},
        "expected_active_inventory_sha256": after["content_inventory_sha256"],
        "execution_binding": execution_binding,
        "source_symbol_completeness": report,
        "old_raw_snapshot": {
            "path": str(snapshot_root / f"{sync.SNAPSHOT_PREFIX}{'a' * 64}"),
            "closure_sha256": "a" * 64,
            "parquet_content_inventory_sha256": before["content_inventory_sha256"],
        },
        "new_raw_snapshot": {
            "path": str(snapshot_root / f"{sync.SNAPSHOT_PREFIX}{'b' * 64}"),
            "closure_sha256": "b" * 64,
            "parquet_content_inventory_sha256": after["content_inventory_sha256"],
        },
        **frozen_audit,
    }
    return audit_payload, execution_binding


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


def test_active_session_symbols_come_from_files_present_on_inventory_max_date(
    tmp_path: Path,
) -> None:
    _write_symbol(tmp_path, "000001.SZ", ["20260726", "20260727"])
    _write_symbol(tmp_path, "000002.SZ", ["20260726"])
    _write_symbol(tmp_path, "000003.SZ", ["20260727"])

    trade_date, symbols = sync.active_session_symbols(tmp_path, sync.inspect_inventory(tmp_path))

    assert trade_date == "20260727"
    assert symbols == {"000001.SZ", "000003.SZ"}


def test_active_session_symbols_reject_filename_row_identity_mismatch(tmp_path: Path) -> None:
    path = _write_symbol(tmp_path, "000001.SZ", ["20260727"])
    mismatched = pd.read_parquet(path)
    mismatched["ts_code"] = "000002.SZ"
    mismatched.to_parquet(path, index=False)
    inventory = sync.inspect_inventory(tmp_path)

    with pytest.raises(sync.DailySyncError, match="exactly one matching"):
        sync.active_session_symbols(tmp_path, inventory)


@pytest.mark.parametrize("invalid", [None, "", "   ", "nan", "000001.XX", "1.SZ"])
def test_daily_source_rejects_null_blank_or_noncanonical_ts_code(invalid: object) -> None:
    frame = _frame("000001.SZ", ["20260727"])
    frame.loc[0, "ts_code"] = invalid

    with pytest.raises(sync.DailySyncError, match="ts_code"):
        sync._validate_daily_frame(frame, "20260727", 1)


@pytest.mark.parametrize("invalid", [None, "", "   ", "nan", "000001.XX", "1.SZ"])
def test_adjustment_factor_rejects_null_blank_or_noncanonical_ts_code(invalid: object) -> None:
    frame = pd.DataFrame([{"ts_code": invalid, "trade_date": "20260727", "adj_factor": 1.0}])

    with pytest.raises(sync.DailySyncError, match="ts_code"):
        sync._validate_adjustment_factor_frame(frame, "20260727", 1)


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


def test_source_completeness_rejects_a_severely_truncated_response_above_legacy_floor() -> None:
    baseline = {f"{index:06d}.SZ" for index in range(5_000)}
    truncated = set(sorted(baseline)[:1_000])
    daily, factors = _source_frames(
        {
            "20260727": baseline,
            "20260728": truncated,
        }
    )

    with pytest.raises(sync.DailySyncError, match="retained only 1000/5000"):
        sync.validate_source_symbol_completeness(
            daily,
            factors,
            ["20260727", "20260728"],
            previous_trade_date="20260727",
            previous_symbols=baseline,
            absolute_minimum=1_000,
            minimum_previous_coverage=0.95,
            maximum_symmetric_change_share=0.05,
        )


def test_source_completeness_accepts_small_additions_deletions_and_factor_only_symbols() -> None:
    baseline = {f"{index:06d}.SZ" for index in range(100)}
    removed = set(sorted(baseline)[:2])
    added = {"900001.BJ", "900002.BJ", "900003.BJ"}
    next_symbols = (baseline - removed) | added
    daily, factors = _source_frames(
        {
            "20260727": baseline,
            "20260728": next_symbols,
        },
        factor_only_by_date={
            "20260727": {"800001.SZ"},
            "20260728": {"800002.SZ"},
        },
    )

    report = sync.validate_source_symbol_completeness(
        daily,
        factors,
        ["20260727", "20260728"],
        previous_trade_date="20260727",
        previous_symbols=baseline,
        absolute_minimum=50,
        minimum_previous_coverage=0.95,
        maximum_symmetric_change_share=0.05,
    )

    assert report["passed"] is True
    assert len(report["report_sha256"]) == 64
    second = report["sessions"][1]
    assert second["daily_adj_factor_aligned_exact"] is True
    assert second["raw_adj_factor_only"]["count"] == 1
    assert second["daily_without_adj_factor"]["count"] == 0
    assert second["added_since_previous"]["count"] == 3
    assert second["removed_since_previous"]["count"] == 2
    assert second["symmetric_change_count"] == 5
    assert second["symmetric_change_share"] == pytest.approx(0.05)
    assert len(second["daily_symbols"]["sha256"]) == 64


def test_source_completeness_rejects_daily_symbol_without_adjustment_factor() -> None:
    baseline = {f"{index:06d}.SZ" for index in range(100)}
    missing = {sorted(baseline)[0]}
    daily, factors = _source_frames(
        {
            "20260727": baseline,
            "20260728": baseline,
        },
        factor_missing_by_date={"20260728": missing},
    )

    with pytest.raises(sync.DailySyncError, match="daily_without_factor=1"):
        sync.validate_source_symbol_completeness(
            daily,
            factors,
            ["20260727", "20260728"],
            previous_trade_date="20260727",
            previous_symbols=baseline,
            absolute_minimum=50,
        )


def test_source_completeness_rejects_abnormal_symbol_expansion() -> None:
    baseline = {f"{index:06d}.SZ" for index in range(100)}
    expanded = baseline | {f"90000{index}.BJ" for index in range(6)}
    daily, factors = _source_frames(
        {
            "20260727": baseline,
            "20260728": expanded,
        }
    )

    with pytest.raises(sync.DailySyncError, match="changed 6/100"):
        sync.validate_source_symbol_completeness(
            daily,
            factors,
            ["20260727", "20260728"],
            previous_trade_date="20260727",
            previous_symbols=baseline,
            absolute_minimum=50,
        )


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


def test_full_qfq_resume_cache_rejects_symlink_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _frame("000001.SZ", ["20260727"])
    monkeypatch.setattr(sync, "download_one", lambda *_: history.copy())
    sync.fetch_new_symbol_histories(
        ["000001.SZ"],
        "20260727",
        sleep_seconds=0,
        expected_max_dates={"000001.SZ": "20260727"},
        object_root=tmp_path,
    )
    cache_path = tmp_path / "full_qfq" / "20260727" / "000001.SZ.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(cache_path.read_bytes())
    cache_path.unlink()
    cache_path.symlink_to(outside)

    with pytest.raises(sync.DailySyncError, match="regular file|symlink"):
        sync.fetch_new_symbol_histories(
            ["000001.SZ"],
            "20260727",
            sleep_seconds=0,
            expected_max_dates={"000001.SZ": "20260727"},
            object_root=tmp_path,
        )


def test_frozen_api_object_rejects_existing_symlink_payload(tmp_path: Path) -> None:
    calendar = pd.DataFrame([{"exchange": "SSE", "cal_date": "20260727", "is_open": 1, "pretrade_date": "20260724"}])
    daily = _frame("000001.SZ", ["20260727"])
    factors = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260727", "adj_factor": 1.0}])
    object_root = tmp_path / "objects"
    objects = sync.freeze_api_response_objects(object_root, calendar, daily, factors)
    object_path = Path(objects["official_calendar"]["path"])
    outside = tmp_path / "outside.csv"
    outside.write_bytes(object_path.read_bytes())
    object_path.unlink()
    object_path.symlink_to(outside)

    with pytest.raises(sync.DailySyncError, match="regular file|symlink"):
        sync.freeze_api_response_objects(object_root, calendar, daily, factors)


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
    snapshot_evidence = {
        "path": str(snapshot),
        "closure_sha256": archived["snapshot_payload_sha256"],
        "parquet_content_inventory_sha256": inventory["content_inventory_sha256"],
    }
    verified_path, verified_inventory = sync._validated_snapshot_for_publication(
        snapshot_evidence,
        expected_root=snapshot_root,
        label="test",
    )
    assert verified_path == snapshot.resolve()
    assert verified_inventory["content_inventory_sha256"] == inventory["content_inventory_sha256"]
    with pytest.raises(sync.DailySyncError, match="outside its bound"):
        sync._validated_snapshot_for_publication(
            snapshot_evidence,
            expected_root=tmp_path / "another-snapshot-root",
            label="test",
        )
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


def test_raw_snapshot_rejects_symlink_source_payload(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    _write_symbol(data_dir, "000001.SZ", ["20260708"])
    inventory = sync.inspect_inventory(data_dir)
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside": true}\n', encoding="utf-8")
    (data_dir / "manifest.json").symlink_to(outside)

    with pytest.raises(sync.DailySyncError, match="symlink payload"):
        sync.archive_raw_snapshot(data_dir, inventory, tmp_path / "snapshots")


def test_existing_raw_snapshot_rejects_symlink_payload(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    source = _write_symbol(data_dir, "000001.SZ", ["20260708"])
    inventory = sync.inspect_inventory(data_dir)
    snapshot_root = tmp_path / "snapshots"
    snapshot, _ = sync.archive_raw_snapshot(data_dir, inventory, snapshot_root)
    payload = snapshot / source.name
    payload.unlink()
    payload.symlink_to(source)

    with pytest.raises(sync.DailySyncError, match="non-regular payload|symlink"):
        sync.archive_raw_snapshot(data_dir, inventory, snapshot_root)


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


def test_open_regular_file_closes_both_descriptors_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"payload")
    opened: list[int] = []
    real_open = os.open
    real_fstat = os.fstat
    calls = 0

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    def fail_second_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(sync.os, "open", tracked_open)
    monkeypatch.setattr(sync.os, "fstat", fail_second_fstat)
    with pytest.raises(OSError, match="injected"):
        sync._open_regular_file_nofollow(path)

    assert len(opened) == 2
    for descriptor in opened:
        with pytest.raises(OSError):
            real_fstat(descriptor)


def test_copy_closes_source_descriptors_when_destination_directory_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    destination_parent = tmp_path / "destination"
    destination_parent.mkdir()
    original = sync._open_directory_nofollow
    before = set(os.listdir("/proc/self/fd"))

    def fail_destination(path: Path) -> int:
        if path == destination_parent:
            raise OSError("injected destination failure")
        return original(path)

    monkeypatch.setattr(sync, "_open_directory_nofollow", fail_destination)
    with pytest.raises(OSError, match="injected"):
        sync._copy_regular_file_nofollow(source, destination_parent / "payload")

    assert set(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("nested", [False, True], ids=["same", "nested"])
def test_run_sync_rejects_snapshot_root_inside_data_before_creation(tmp_path: Path, nested: bool) -> None:
    data_dir = tmp_path / "not-created"
    snapshot_root = data_dir / "snapshots" if nested else data_dir
    args = Namespace(
        data_dir=data_dir,
        snapshot_root=snapshot_root,
        end_date="20260727",
        min_daily_rows=1,
        new_symbol_sleep_seconds=0.0,
        apply=False,
    )

    with pytest.raises(sync.DailySyncError, match="snapshot root must be outside"):
        sync.run_sync(args)

    assert not data_dir.exists()


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
    rule = first["binding"]["parameters"]["source_symbol_completeness_rule"]  # type: ignore[index]
    assert rule == sync.source_symbol_completeness_rule()
    assert rule["absolute_minimum_symbols"] == 4_000
    assert rule["minimum_previous_session_coverage"] == 0.95
    assert rule["maximum_symmetric_change_share"] == 0.05


def test_execution_binding_records_and_authenticates_source_completeness_evidence(
    tmp_path: Path,
) -> None:
    baseline = {f"{index:06d}.SZ" for index in range(4_000)}
    daily, factors = _source_frames({"20260727": baseline})
    report = sync.validate_source_symbol_completeness(
        daily,
        factors,
        ["20260727"],
        previous_trade_date="20260727",
        previous_symbols=baseline,
    )
    binding = _execution_binding(
        tmp_path / "live",
        tmp_path / "snapshots",
        source_symbol_completeness=report,
    )

    evidence = binding["binding"]["input_evidence"]  # type: ignore[index]
    assert evidence["source_symbol_completeness"] == report
    assert evidence["source_symbol_completeness_sha256"] == sync.sha256_bytes(sync.canonical_json(report))
    sync.assert_execution_binding_current(binding)

    tampered = json.loads(json.dumps(binding))
    tampered["binding"]["input_evidence"]["source_symbol_completeness"]["baseline"]["symbols"]["count"] -= 1
    tampered["binding_sha256"] = sync.sha256_bytes(sync.canonical_json(tampered["binding"]))
    with pytest.raises(sync.DailySyncError, match="completeness evidence is malformed"):
        sync.assert_execution_binding_current(tampered)


def test_formal_execution_binding_rejects_missing_source_completeness(
    tmp_path: Path,
) -> None:
    binding = _execution_binding(
        tmp_path / "live",
        tmp_path / "snapshots",
        source_symbol_completeness=None,
    )

    sync.assert_execution_binding_current(binding)
    with pytest.raises(sync.DailySyncError, match="formal publication requires"):
        sync.assert_execution_binding_current(
            binding,
            require_source_completeness=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("branch", "another-branch"),
        ("upstream", "origin/feat/surge-wave-strategy"),
        ("remote_name", "origin"),
        ("remote_ref", "refs/heads/another-branch"),
        ("remote_fetch_url", "git@example.invalid:other/repo.git"),
        ("remote_push_url", "git@example.invalid:other/repo.git"),
    ],
)
def test_formal_execution_binding_is_fixed_to_the_research_remote(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    binding = _formalize_execution_binding(_execution_binding(tmp_path / "live", tmp_path / "snapshots"))
    binding["binding"]["git"][field] = value  # type: ignore[index]
    binding["binding_sha256"] = sync.sha256_bytes(sync.canonical_json(binding["binding"]))

    with pytest.raises(sync.DailySyncError, match="frozen branch, upstream, URL, and remote ref"):
        sync.assert_execution_binding_current(
            binding,
            require_source_completeness=True,
            require_formal_publication=True,
        )


def test_formal_execution_binding_rejects_actual_remote_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _formalize_execution_binding(_execution_binding(tmp_path / "live", tmp_path / "snapshots"))
    _patch_clean_git(monkeypatch, binding, remote_head="0" * 40)

    with pytest.raises(sync.DailySyncError, match="actual remote branch"):
        sync.assert_execution_binding_current(
            binding,
            require_source_completeness=True,
            require_formal_publication=True,
        )


def test_frozen_source_objects_recompute_the_exact_bound_completeness_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "live"
    symbols = {f"{index:06d}.SZ" for index in range(sync.SOURCE_SYMBOL_ABSOLUTE_MINIMUM)}
    audit_fragment, report = _frozen_source_fixture(data_dir, symbols)

    recomputed = sync._recompute_frozen_source_completeness(
        audit_fragment,
        data_dir=data_dir,
        safe_end="20260727",
        previous_trade_date="20260727",
        previous_symbols=symbols,
        minimum_rows=1_000,
    )

    assert recomputed == report


@pytest.mark.parametrize("invalid_binding", ["apply_false", "dirty", "unpushed"])
def test_publish_rejects_nonformal_binding_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_binding: str,
) -> None:
    data_dir = tmp_path / "live"
    snapshot_root = tmp_path / "snapshots"
    execution_binding = _formalize_execution_binding(_execution_binding(data_dir, snapshot_root))
    if invalid_binding == "apply_false":
        execution_binding["binding"]["parameters"]["apply"] = False  # type: ignore[index]
    elif invalid_binding == "dirty":
        execution_binding["binding"]["git"]["worktree_clean"] = False  # type: ignore[index]
    else:
        execution_binding["binding"]["git"]["upstream_head"] = "0" * 40  # type: ignore[index]
    execution_binding["binding_sha256"] = sync.sha256_bytes(  # type: ignore[index]
        sync.canonical_json(execution_binding["binding"])
    )
    audit_payload = {
        "before_inventory": {"content_inventory_sha256": "a" * 64},
        "expected_active_inventory_sha256": "b" * 64,
        "execution_binding": execution_binding,
    }
    exchanged = False

    def unexpected_exchange(*_args: object) -> None:
        nonlocal exchanged
        exchanged = True

    monkeypatch.setattr(sync, "atomic_exchange_directories", unexpected_exchange)

    with pytest.raises(sync.DailySyncError, match="apply=true clean binding"):
        sync.publish_candidate(data_dir, tmp_path / "candidate", tmp_path, audit_payload)

    assert exchanged is False
    assert not (tmp_path / sync.JOURNAL_FILE_NAME).exists()


def test_publish_rechecks_the_current_worktree_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "live"
    execution_binding = _formalize_execution_binding(_execution_binding(data_dir, tmp_path / "snapshots"))
    real_git_output = sync._git_output
    head = execution_binding["binding"]["git"]["head"]  # type: ignore[index]

    def dirty_git_output(repo_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M unrelated.txt"
        if args == ("rev-parse", "@{upstream}"):
            return str(head)
        return real_git_output(repo_root, *args)

    monkeypatch.setattr(sync, "_git_output", dirty_git_output)
    exchanged = False

    def unexpected_exchange(*_args: object) -> None:
        nonlocal exchanged
        exchanged = True

    monkeypatch.setattr(sync, "atomic_exchange_directories", unexpected_exchange)
    audit_payload = {
        "before_inventory": {"content_inventory_sha256": "a" * 64},
        "expected_active_inventory_sha256": "b" * 64,
        "execution_binding": execution_binding,
    }

    with pytest.raises(sync.DailySyncError, match="remain clean"):
        sync.publish_candidate(data_dir, tmp_path / "candidate", tmp_path, audit_payload)

    assert exchanged is False
    assert not (tmp_path / sync.JOURNAL_FILE_NAME).exists()


def test_publish_rechecks_the_actual_remote_immediately_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0
    exchanged = False

    def check_binding(*_args: object, **_kwargs: object) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise sync.DailySyncError("actual remote branch changed")

    def unexpected_exchange(*_args: object) -> None:
        nonlocal exchanged
        exchanged = True

    monkeypatch.setattr(sync, "assert_execution_binding_current", check_binding)
    monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sync, "atomic_exchange_directories", unexpected_exchange)
    audit_payload = {
        "before_inventory": {"content_inventory_sha256": "a" * 64},
        "expected_active_inventory_sha256": "b" * 64,
        "execution_binding": {},
    }

    with pytest.raises(sync.DailySyncError, match="actual remote branch changed"):
        sync.publish_candidate(tmp_path / "live", tmp_path / "candidate", tmp_path, audit_payload)

    assert checks == 2
    assert exchanged is False
    assert not (tmp_path / sync.JOURNAL_FILE_NAME).exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_objects",
        "missing_top_level_report",
        "empty_sessions",
        "forged_report_hash",
        "wrong_data_path",
        "wrong_safe_end",
    ],
)
def test_publish_recomputes_frozen_evidence_and_rejects_tampering_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    data_dir = tmp_path / "live"
    candidate = tmp_path / "run" / "candidate_active"
    run_dir = candidate.parent
    _write_symbol(data_dir, "000001.SZ", ["20260727"], [10.0])
    _write_symbol(candidate, "000001.SZ", ["20260727"], [11.0])
    before = sync.inspect_inventory(data_dir)
    after = sync.inspect_inventory(candidate)
    symbols = {f"{index:06d}.SZ" for index in range(sync.SOURCE_SYMBOL_ABSOLUTE_MINIMUM)}
    frozen_audit, valid_report = _frozen_source_fixture(data_dir, symbols)
    report = valid_report
    if tamper == "empty_sessions":
        report = json.loads(json.dumps(valid_report))
        report["sessions"] = []
        report["report_sha256"] = sync.sha256_bytes(
            sync.canonical_json({key: value for key, value in report.items() if key != "report_sha256"})
        )
    elif tamper == "forged_report_hash":
        report = {**valid_report, "report_sha256": "0" * 64}
    snapshot_root = tmp_path / "snapshots"
    audit_payload, execution_binding = _formal_publication_payload(
        data_dir,
        before,
        after,
        snapshot_root,
        frozen_audit,
        report,
    )
    if tamper == "missing_objects":
        audit_payload.pop("api_response_objects")
    elif tamper == "missing_top_level_report":
        audit_payload.pop("source_symbol_completeness")
    elif tamper == "wrong_data_path":
        audit_payload["data_dir"] = str(tmp_path / "another-live")
    elif tamper == "wrong_safe_end":
        audit_payload["safe_end_date"] = "20260728"

    def fake_snapshot(
        snapshot: object,
        *,
        expected_root: Path,
        label: str,
    ) -> tuple[Path, dict[str, object]]:
        assert isinstance(snapshot, dict)
        assert expected_root == snapshot_root
        inventory = before if label == "old raw" else after
        return Path(str(snapshot["path"])), inventory

    monkeypatch.setattr(sync, "_validated_snapshot_for_publication", fake_snapshot)
    monkeypatch.setattr(
        sync,
        "active_session_symbols",
        lambda *_args, **_kwargs: ("20260727", symbols),
    )
    _patch_clean_git(monkeypatch, execution_binding)
    exchanged = False

    def unexpected_exchange(*_args: object) -> None:
        nonlocal exchanged
        exchanged = True

    monkeypatch.setattr(sync, "atomic_exchange_directories", unexpected_exchange)

    with pytest.raises(
        sync.DailySyncError,
        match="exact frozen|differs from recomputed|completeness evidence is malformed|data path",
    ):
        sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

    assert exchanged is False
    assert not (run_dir / sync.JOURNAL_FILE_NAME).exists()


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


def test_recovery_rejects_an_unbound_hash_named_file_as_a_commit_marker() -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_forged_audit_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / f"{sync.RUN_PREFIX}forged-audit" / "candidate_active"
        run_dir = candidate.parent
        _write_symbol(data_dir, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        _write_symbol(candidate, "000001.SZ", ["20260708"], [10.0])
        new_inventory = sync.inspect_inventory(data_dir)
        old_inventory = sync.inspect_inventory(candidate)
        forged_raw = b"{}\n"
        forged_path = root / "unbound" / f"{sync.sha256_bytes(forged_raw)}.json"
        sync.atomic_write_bytes(forged_path, forged_raw)
        sync.atomic_write_json(
            run_dir / sync.JOURNAL_FILE_NAME,
            {
                "schema": "a_stock_daily_qfq_directory_exchange_journal_v2",
                "phase": "AUDIT_PREPARED",
                "data_dir": str(data_dir),
                "candidate_path": str(candidate),
                "old_inventory_sha256": old_inventory["content_inventory_sha256"],
                "new_inventory_sha256": new_inventory["content_inventory_sha256"],
                "audit_path": str(forged_path),
            },
        )

        recovered = sync.recover_orphaned_runs(data_dir)

        assert recovered == [f"rolled_back:{run_dir.name}"]
        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == old_inventory["content_inventory_sha256"]
        assert not run_dir.exists()


def test_recovery_accepts_only_a_bound_canonical_audit_commit_marker() -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_bound_audit_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / f"{sync.RUN_PREFIX}bound-audit" / "candidate_active"
        run_dir = candidate.parent
        _write_symbol(data_dir, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        _write_symbol(candidate, "000001.SZ", ["20260708"], [10.0])
        new_inventory = sync.inspect_inventory(data_dir)
        old_inventory = sync.inspect_inventory(candidate)
        audit = {
            "schema": "a_stock_daily_qfq_sync_audit_v2",
            "data_dir": str(data_dir),
            "before_inventory": {
                "content_inventory_sha256": old_inventory["content_inventory_sha256"],
            },
            "expected_active_inventory_sha256": new_inventory["content_inventory_sha256"],
            "after_inventory": {
                "content_inventory_sha256": new_inventory["content_inventory_sha256"],
            },
            "execution_binding": _execution_binding(data_dir, root / "snapshots"),
        }
        audit_raw = (
            json.dumps(
                audit,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        audit_path = root / sync.AUDIT_DIR_NAME / f"{sync.sha256_bytes(audit_raw)}.json"
        sync.atomic_write_bytes(audit_path, audit_raw, replace_existing=False)
        sync.atomic_write_json(
            run_dir / sync.JOURNAL_FILE_NAME,
            {
                "schema": "a_stock_daily_qfq_directory_exchange_journal_v2",
                "phase": "AUDIT_PREPARED",
                "data_dir": str(data_dir),
                "candidate_path": str(candidate),
                "old_inventory_sha256": old_inventory["content_inventory_sha256"],
                "new_inventory_sha256": new_inventory["content_inventory_sha256"],
                "audit_path": str(audit_path),
            },
        )

        recovered = sync.recover_orphaned_runs(data_dir)

        assert recovered == [f"cleaned_committed:{run_dir.name}"]
        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == new_inventory["content_inventory_sha256"]
        assert not run_dir.exists()


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
        real_link = os.link

        def fail_audit_link(
            source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            target: object,
            **kwargs: object,
        ) -> None:
            target_path = Path(target)  # type: ignore[arg-type]
            if target_path.parent.name == sync.AUDIT_DIR_NAME:
                raise OSError("injected audit publication failure")
            real_link(source, target, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(sync.os, "link", fail_audit_link)
        monkeypatch.setattr(sync, "assert_execution_binding_current", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)

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

        def fail_after_exchange(
            _binding: object,
            **_kwargs: object,
        ) -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise sync.DailySyncError("execution source bytes changed during synchronization")

        monkeypatch.setattr(sync, "assert_execution_binding_current", fail_after_exchange)
        monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)

        with pytest.raises(sync.DailySyncError, match="execution source bytes changed"):
            sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

        assert checks == 2
        assert (data_dir / old.name).read_bytes() == old_bytes
        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == before["content_inventory_sha256"]


@pytest.mark.parametrize("drifted_side", ["active", "candidate"])
def test_publish_rechecks_both_closures_immediately_before_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_side: str,
) -> None:
    data_dir = tmp_path / "live"
    candidate = tmp_path / "run" / "candidate_active"
    run_dir = candidate.parent
    _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
    _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
    before = sync.inspect_inventory(data_dir)
    after = sync.inspect_inventory(candidate)
    audit_payload = {
        "before_inventory": {key: value for key, value in before.items() if key != "records"},
        "expected_active_inventory_sha256": after["content_inventory_sha256"],
        "execution_binding": _execution_binding(data_dir, tmp_path / "snapshots"),
    }
    checks = 0

    def drift_after_evidence(_binding: object, **_kwargs: object) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            target = data_dir if drifted_side == "active" else candidate
            _write_symbol(target, "000001.SZ", ["20260708"], [99.0])

    exchanged = False

    def unexpected_exchange(*_args: object) -> None:
        nonlocal exchanged
        exchanged = True

    monkeypatch.setattr(sync, "assert_execution_binding_current", drift_after_evidence)
    monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sync, "atomic_exchange_directories", unexpected_exchange)

    with pytest.raises(sync.DailySyncError, match="immediately before atomic exchange"):
        sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

    assert checks == 2
    assert exchanged is False
    assert (run_dir / sync.JOURNAL_FILE_NAME).is_file()


def test_publish_detects_active_old_drift_across_the_exchange_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_exchange_race_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / "run" / "candidate_active"
        run_dir = candidate.parent
        _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
        _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        before = sync.inspect_inventory(data_dir)
        after = sync.inspect_inventory(candidate)
        audit_payload = {
            "before_inventory": {key: value for key, value in before.items() if key != "records"},
            "expected_active_inventory_sha256": after["content_inventory_sha256"],
            "execution_binding": _execution_binding(data_dir, root / "snapshots"),
        }
        real_exchange = sync.atomic_exchange_directories

        def drift_active_then_exchange(left: Path, right: Path) -> None:
            _write_symbol(left, "000001.SZ", ["20260708"], [99.0])
            real_exchange(left, right)

        monkeypatch.setattr(sync, "assert_execution_binding_current", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sync, "atomic_exchange_directories", drift_active_then_exchange)

        with pytest.raises(sync.DailySyncError, match="immediately after atomic exchange"):
            sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

        assert sync.inspect_inventory(data_dir)["content_inventory_sha256"] == after["content_inventory_sha256"]
        assert sync.inspect_inventory(candidate)["content_inventory_sha256"] not in {
            before["content_inventory_sha256"],
            after["content_inventory_sha256"],
        }
        assert (run_dir / sync.JOURNAL_FILE_NAME).is_file()
        assert not list((root / sync.AUDIT_DIR_NAME).glob("*.json"))


def test_late_old_candidate_drift_is_not_attested_or_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xs_chan_late_old_write_test_", dir=cache_root) as temporary:
        root = Path(temporary)
        data_dir = root / "live"
        candidate = root / f"{sync.RUN_PREFIX}late-old-write" / "candidate_active"
        run_dir = candidate.parent
        _write_symbol(data_dir, "000001.SZ", ["20260708"], [10.0])
        _write_symbol(candidate, "000001.SZ", ["20260708", "20260709"], [10.0, 11.0])
        before = sync.inspect_inventory(data_dir)
        after = sync.inspect_inventory(candidate)
        audit_payload = {
            "before_inventory": {key: value for key, value in before.items() if key != "records"},
            "expected_active_inventory_sha256": after["content_inventory_sha256"],
            "execution_binding": _execution_binding(data_dir, root / "snapshots"),
        }
        real_atomic_write = sync.atomic_write_bytes

        def write_audit_then_finish_old_write(
            path: Path,
            raw: bytes,
            *,
            replace_existing: bool = True,
        ) -> None:
            real_atomic_write(path, raw, replace_existing=replace_existing)
            if path.parent.name == sync.AUDIT_DIR_NAME:
                _write_symbol(candidate, "000001.SZ", ["20260708"], [99.0])

        monkeypatch.setattr(sync, "assert_execution_binding_current", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sync, "_validate_publication_evidence", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sync, "atomic_write_bytes", write_audit_then_finish_old_write)

        with pytest.raises(sync.DailySyncError, match="before successful publication return"):
            sync.publish_candidate(data_dir, candidate, run_dir, audit_payload)

        journal = json.loads((run_dir / sync.JOURNAL_FILE_NAME).read_text(encoding="utf-8"))
        audit_path = Path(journal["audit_path"])
        assert audit_path.is_file()
        assert candidate.is_dir()
        assert sync.inspect_inventory(candidate)["content_inventory_sha256"] != before["content_inventory_sha256"]
        with pytest.raises(
            sync.DailySyncError,
            match="cannot (?:clean committed|automatically recover)",
        ):
            sync.recover_orphaned_runs(data_dir)
        assert candidate.is_dir()
