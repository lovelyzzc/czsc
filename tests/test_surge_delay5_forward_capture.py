"""Tests for the append-only Delay5 forward capture supplement."""

from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_delay5_forward_capture as capture  # noqa: E402


def test_protocol_binds_main_protocol_and_capture_script() -> None:
    protocol = capture.build_protocol()
    capture.validate_protocol(protocol)

    drifted = copy.deepcopy(protocol)
    drifted["capture_script"]["sha256"] = "0" * 64
    with pytest.raises(capture.CaptureError, match="drifted"):
        capture.validate_protocol(drifted)

    assert protocol["outcome_lock"]["outcomes_loaded"] is False
    assert protocol["outcome_lock"]["live_authorized"] is False


def test_latest_only_classification_is_fail_closed() -> None:
    latest = pd.Timestamp("2026-08-14")
    assert capture.capture_classification(latest, latest) == "CAPTURED_LATEST_OBSERVED_SESSION"
    assert capture.capture_classification(pd.Timestamp("2026-08-13"), latest) == "LATE_EXCLUDED"


def test_panel_projection_is_canonical_and_session_scoped() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["B", "A", "A"],
            "dt": pd.to_datetime(["2026-08-13", "2026-08-13", "2026-08-12"]),
            "open": [2.0, 1.0, 0.5],
            "close": [2.2, 1.1, 0.6],
            "amount_e": [20.0, 10.0, 5.0],
        }
    )
    result = capture.build_panel_projection(panel, pd.Timestamp("2026-08-13"))

    assert result["columns"] == list(capture.PANEL_FIELDS)
    assert result["records"] == [
        ["A", "2026-08-13", 1.0, 1.1, 10.0],
        ["B", "2026-08-13", 2.0, 2.2, 20.0],
    ]


def test_append_only_chain_is_idempotent_and_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = {"protocol": "unit-test"}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(capture.canonical_json(protocol))
    monkeypatch.setattr(capture, "PROTOCOL_PATH", protocol_path)
    root = tmp_path / "ledger"
    genesis_payload = {
        "schema": capture.LEDGER_SCHEMA,
        "capture_protocol_sha256": capture.sha256_file(protocol_path),
    }
    genesis = capture._append_record(
        root,
        protocol,
        record_type="genesis",
        session=None,
        classification="OUTCOME_BLIND_GENESIS",
        payload=genesis_payload,
        captured_at_utc="2026-08-12T08:30:00Z",
    )
    first = capture._append_record(
        root,
        protocol,
        record_type="session_capture",
        session="2026-08-13",
        classification="CAPTURED_LATEST_OBSERVED_SESSION",
        payload={"snapshot": {"sha256": "x"}, "outcomes_loaded": False},
        captured_at_utc="2026-08-13T07:30:00Z",
    )
    repeated = capture._append_record(
        root,
        protocol,
        record_type="session_capture",
        session="2026-08-13",
        classification="CAPTURED_LATEST_OBSERVED_SESSION",
        payload={"snapshot": {"sha256": "x"}, "outcomes_loaded": False},
        captured_at_utc="2026-08-13T08:00:00Z",
    )

    records = capture.scan_records(root)
    capture.validate_ledger(records, protocol)
    assert records == (genesis, first)
    assert repeated.path == first.path

    raw = bytearray(first.path.read_bytes())
    raw[-1] = ord(" ")
    first.path.write_bytes(bytes(raw))
    with pytest.raises(capture.CaptureError, match="canonicalization drift"):
        capture.scan_records(root)


def test_content_addressed_session_objects_detect_drift(tmp_path: Path) -> None:
    objects = {
        name: capture._store_object(tmp_path, name, {"name": name})
        for name in ("market_panel_projection", "production_projection", "market_projection")
    }
    snapshot = {
        "schema": capture.SNAPSHOT_SCHEMA,
        "session": "2026-08-13",
        "capture_classification": "CAPTURED_LATEST_OBSERVED_SESSION",
        "accrual_session": False,
        "objects": objects,
        "outcomes_loaded": False,
    }
    snapshot_record = capture._store_object(tmp_path, "session_snapshot", snapshot)
    record = capture.CaptureRecord(
        tmp_path / "unused.json",
        {
            "record_type": "session_capture",
            "session": "2026-08-13",
            "payload": {"snapshot": snapshot_record, "outcomes_loaded": False},
        },
        b"",
    )
    assert capture.verify_session_objects(tmp_path, record) == snapshot

    child = tmp_path / objects["market_projection"]["object"]
    child.write_text("{}", encoding="utf-8")
    with pytest.raises(capture.CaptureError, match="identity drift"):
        capture.verify_session_objects(tmp_path, record)


def test_exact_cross_section_capture_enforces_complete_provider_object() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def daily_basic(self, **kwargs: str) -> pd.DataFrame:
            self.calls.append(kwargs)
            symbols = [f"{index:06d}.SZ" for index in range(1, 4_001)]
            return pd.DataFrame(
                {
                    "ts_code": symbols,
                    "trade_date": "20260813",
                    "close": 10.0,
                    "total_share": 100.0,
                    "float_share": 80.0,
                    "total_mv": 1_000.0,
                    "circ_mv": 800.0,
                }
            )

    provider = FakeProvider()
    content, diagnostics = capture._fetch_exact_object(
        provider,
        pd.Timestamp("2026-08-13"),
        {"000001.SZ"},
    )

    assert content["trade_date"] == "2026-08-13"
    assert len(content["rows"]) == 4_000
    assert diagnostics == {"rows": 4_000, "requested_symbols": 1, "non_requested_null_counts": {}, "attempts": 1}
    assert provider.calls[0]["trade_date"] == "20260813"


def test_untracked_anchor_cannot_be_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(capture, "ANCHOR_DIR", tmp_path / "anchors")
    record = capture.CaptureRecord(
        tmp_path / "record.json",
        {"sequence": 1, "record_hash": "a" * 64},
        b"",
    )
    anchor = capture.ANCHOR_DIR / f"000001_{'a' * 64}.json"
    anchor.parent.mkdir()
    anchor.write_text("{}", encoding="utf-8")

    result = capture.validate_pushed_anchor(record, pd.Timestamp("2026-08-14"))
    assert result["available"] is True
    assert result["tracked"] is False
    assert result["pushed"] is False
    assert result["before_next_open"] is False


def test_capture_source_never_reads_return_or_exit_outcomes() -> None:
    source = inspect.getsource(capture)
    forbidden = ("ret_gross_pct", "ret_net_pct", "exit_price", "exit_reason")
    assert not any(token in source for token in forbidden)
    assert 'outcomes_loaded": False' in source
    assert 'live_authorized": False' in source
