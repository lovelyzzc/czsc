from __future__ import annotations

import base64
import hashlib
import inspect
import json
import shutil
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_execution_v2_1 as execution  # noqa: E402
import xs_chan_oos_ledger_v2_1 as ledger  # noqa: E402

# Test-only 1024-bit RSA key; production ledger code only receives N and E.
_P = 12060072689906285720137231624945355253539818774429660663930157103511652948867046809262822711405803847709721164087280652053303791883070380543746355376250459
_Q = 12297000748699417999801620737686606274978151199065837760279798349489235946473170008161873811805608221555612241727435509179103883637864579203342456821357897
_N = _P * _Q
_E = 65537
_D = pow(_E, -1, (_P - 1) * (_Q - 1))
_PROVIDER = "TEST_EXTERNAL_TIMESTAMP_AUTHORITY"
_KEY_ID = "test-rsa-key-2026"
_GENESIS_NOW = datetime(2026, 7, 15, 10, tzinfo=UTC)
_CALENDAR_NOW = datetime(2026, 7, 16, 10, tzinfo=UTC)
_OOS_START = date(2026, 7, 20)


class _Clock:
    def __init__(self, value: datetime):
        self.value = value

    def set(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    value = _Clock(_GENESIS_NOW)
    monkeypatch.setattr(ledger, "_utc_now", value)
    return value


@pytest.fixture
def verifier() -> ledger.RsaPkcs1v15Sha256ReceiptVerifier:
    return ledger.RsaPkcs1v15Sha256ReceiptVerifier(
        provider=_PROVIDER,
        key_id=_KEY_ID,
        modulus_hex=f"{_N:x}",
        public_exponent=_E,
    )


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(UTC)


def _sign_receipt(subject: str, issued_at: datetime, nonce: str) -> dict[str, Any]:
    fields = {
        "schema": ledger.ANCHOR_RECEIPT_SCHEMA,
        "algorithm": ledger.ANCHOR_ALGORITHM,
        "provider": _PROVIDER,
        "key_id": _KEY_ID,
        "subject_sha256": subject,
        "issued_at_utc": _utc(issued_at),
        "nonce": nonce,
    }
    digest_info = ledger._SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(ledger._canonical_bytes(fields)).digest()
    width = (_N.bit_length() + 7) // 8
    encoded = b"\x00\x01" + b"\xff" * (width - len(digest_info) - 3) + b"\x00" + digest_info
    signature = pow(int.from_bytes(encoded, "big"), _D, _N).to_bytes(width, "big")
    return {**fields, "signature_base64": base64.b64encode(signature).decode("ascii")}


def _dependencies() -> dict[str, str]:
    return {name: _hash(f"dependency:{name}") for name in ledger.EXPECTED_DEPENDENCY_KEYS}


def _readiness() -> dict[str, Any]:
    return {
        name: {"status": "PASSED", "evidence_sha256": _hash(f"readiness:{name}")} for name in ledger.READINESS_GATE_KEYS
    }


def _config() -> dict[str, Any]:
    initial_state = execution.empty_portfolio(10_000_000.0).as_dict()
    return {
        "schema": ledger.GENESIS_CONFIG_SCHEMA,
        "protocol_id": ledger.PROTOCOL_ID,
        "protocol_sha256": ledger.CANONICAL_PROTOCOL_SHA256,
        "data_bundle_sha256": _hash("clean-data-bundle"),
        "dependency_sha256": _dependencies(),
        "readiness_gates": _readiness(),
        "initial_portfolio_state": initial_state,
        "initial_portfolio_state_sha256": execution.object_sha256(initial_state),
        "trial_id": ledger.CANONICAL_PROTOCOL_SHA256,
        "trial_registry_namespace": "czsc/xs-chan/confirmatory",
        "trial_registry_authority_sha256": _hash("registry-authority"),
        "chain_role": "PRIMARY",
        "calendar_source_id": "SSE_SZSE_OFFICIAL_CALENDAR",
        "oos_start_date": _OOS_START.isoformat(),
        "min_complete_weeks": 52,
        "exchange_timezone": "Asia/Shanghai",
        "decision_close_local": "15:00:00",
        "execution_open_local": "09:30:00",
        "open_execution_deadline_local": "10:00:00",
        "session_close_local": "15:00:00",
    }


def _init(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    *,
    root_name: str = "ledger",
    registry_name: str = "registry",
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / root_name
    registry = tmp_path / registry_name
    config = _config()
    commitment = ledger.genesis_commitment_sha256(config, receipt_verifier=verifier)
    anchor = _sign_receipt(commitment, _GENESIS_NOW - timedelta(minutes=30), "genesis-anchor-0001")
    clock.set(_GENESIS_NOW)
    ledger.init_ledger(
        root,
        genesis_config=config,
        trial_registry_root=registry,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    return root, registry, config


def _business_sessions(start: date, end: date) -> list[date]:
    sessions = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def _segment(index: int, start: date, end: date, retrieved_at: datetime) -> dict[str, Any]:
    return {
        "schema": ledger.CALENDAR_SEGMENT_SCHEMA,
        "segment_index": index,
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "sessions": [item.isoformat() for item in _business_sessions(start, end)],
        "source_id": "SSE_SZSE_OFFICIAL_CALENDAR",
        "source_uri": f"https://official.example/calendar/{index}.json",
        "source_sha256": _hash(f"official-calendar:{index}:{start}:{end}"),
        "source_retrieved_at_utc": _utc(retrieved_at),
    }


def _append_calendar(
    root: Path,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    clock: _Clock,
    segment: dict[str, Any],
    append_at: datetime,
) -> Path:
    prepared = ledger.prepare_calendar_extension(root, segment=segment)
    anchor = _sign_receipt(
        prepared["record_commitment_sha256"],
        append_at - timedelta(minutes=1),
        f"calendar-segment-{segment['segment_index']:04d}",
    )
    clock.set(append_at)
    return ledger.append_calendar_segment(
        root,
        segment=segment,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )


def _record_hash(path: Path) -> str:
    return json.loads(path.read_bytes())["record_hash"]


def _state(root: Path) -> ledger._ReplayState:
    return ledger._validate_chain(root, receipt_verifier=None)


def _week_specs(root: Path) -> list[ledger.WeekSpec]:
    return ledger._week_specs(_state(root))


def _anchor_cycle_event(
    root: Path,
    record_type: str,
    week_id: str,
    data: dict[str, Any],
    append_at: datetime,
) -> dict[str, Any]:
    prepared = ledger.prepare_cycle_event(
        root,
        record_type=record_type,
        week_id=week_id,
        data=data,
    )
    return _sign_receipt(
        prepared["record_commitment_sha256"],
        append_at - timedelta(seconds=1),
        f"{record_type}-record-{len(ledger.read_ledger_records(root)):06d}",
    )


def _decision_data(state: ledger._ReplayState, week: ledger.WeekSpec, tag: str) -> dict[str, Any]:
    return {
        "schema": ledger.DECISION_SCHEMA,
        "snapshot_cutoff_utc": _utc(_local(week.decision_dt, 15)),
        "engine_sha256": state.config["dependency_sha256"]["research_engine"],
        "decision_artifact_sha256": _hash(f"decision:{tag}"),
        "portfolio_before_sha256": state.portfolio_sha256,
        "pending_sells_before_sha256": state.pending_sells_sha256,
        "calendar_state_sha256": week.calendar_state_sha256,
    }


def _open_data(
    state: ledger._ReplayState,
    session: date,
    portfolio_after: str,
    pending_after: str,
    pending_count_after: int,
    tag: str,
) -> dict[str, Any]:
    assert state.decision_record is not None
    return {
        "schema": ledger.OPEN_EXECUTION_SCHEMA,
        "snapshot_cutoff_utc": _utc(_local(session, 9, 30)),
        "engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "decision_record_sha256": state.decision_record.record_hash,
        "previous_eod_record_sha256": state.last_eod_record.record_hash if state.last_eod_record else ledger.ZERO_HASH,
        "portfolio_before_sha256": state.portfolio_sha256,
        "pending_sells_before_sha256": state.pending_sells_sha256,
        "pending_sell_count_before": state.pending_sell_count,
        "open_prices_sha256": _hash(f"open:{tag}"),
        "opening_auction_turnover_sha256": _hash(f"auction:{tag}"),
        "limit_state_sha256": _hash(f"limit:{tag}"),
        "corporate_actions_sha256": _hash(f"actions:{tag}"),
        "requested_orders_sha256": _hash(f"orders:{tag}"),
        "fills_sha256": _hash(f"fills:{tag}"),
        "fees_sha256": _hash(f"fees:{tag}"),
        "execution_result_sha256": _hash(f"result:{tag}"),
        "portfolio_after_sha256": portfolio_after,
        "pending_sells_after_sha256": pending_after,
        "pending_sell_count_after": pending_count_after,
    }


def _eod_data(
    state: ledger._ReplayState,
    session: date,
    portfolio_after: str,
    tag: str,
) -> dict[str, Any]:
    assert state.last_open_record is not None
    return {
        "schema": ledger.EOD_VALUATION_SCHEMA,
        "snapshot_cutoff_utc": _utc(_local(session, 15)),
        "engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "session_open_execution_sha256": state.last_open_record.record_hash,
        "raw_close_snapshot_sha256": _hash(f"close:{tag}"),
        "corporate_actions_sha256": _hash(f"eod-actions:{tag}"),
        "portfolio_before_eod_sha256": state.portfolio_sha256,
        "portfolio_state_sha256": portfolio_after,
        "pending_sells_sha256": state.pending_sells_sha256,
        "pending_sell_count": state.pending_sell_count,
        "nav_artifact_sha256": _hash(f"nav:{tag}"),
    }


def _close_data(state: ledger._ReplayState, recorded_at: datetime, tag: str) -> dict[str, Any]:
    assert state.last_eod_record is not None
    return {
        "schema": ledger.WEEKLY_CLOSE_SCHEMA,
        "snapshot_cutoff_utc": _utc(recorded_at),
        "execution_engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "statistics_engine_sha256": state.config["dependency_sha256"]["statistics_engine"],
        "eod_record_sha256": state.last_eod_record.record_hash,
        "portfolio_state_sha256": state.portfolio_sha256,
        "pending_sells_sha256": state.pending_sells_sha256,
        "pending_sell_count": state.pending_sell_count,
        "weekly_returns_sha256": _hash(f"returns:{tag}"),
        "statistics_input_sha256": _hash(f"stats-input:{tag}"),
    }


def _append_public_cycle(
    root: Path,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    clock: _Clock,
    week: ledger.WeekSpec,
    tag: str,
) -> None:
    state = _state(root)
    decision_data = _decision_data(state, week, tag)
    decision_at = _local(week.decision_dt, 16)
    anchor = _anchor_cycle_event(root, "decision", week.week_id, decision_data, decision_at)
    clock.set(decision_at)
    ledger.append_decision(
        root,
        week_id=week.week_id,
        data=decision_data,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    for index, session in enumerate(week.sessions):
        state = _state(root)
        portfolio_open = _hash(f"portfolio:{tag}:{session}:open")
        pending = _hash(f"pending:{tag}:{session}")
        open_data = _open_data(state, session, portfolio_open, pending, 0, f"{tag}:{session}")
        open_at = _local(session, 9, 40)
        anchor = _anchor_cycle_event(root, "session_open_execution", week.week_id, open_data, open_at)
        clock.set(open_at)
        ledger.append_open_execution(
            root,
            week_id=week.week_id,
            data=open_data,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )
        state = _state(root)
        portfolio_eod = _hash(f"portfolio:{tag}:{session}:eod")
        eod_data = _eod_data(state, session, portfolio_eod, f"{tag}:{session}")
        eod_at = _local(session, 16)
        anchor = _anchor_cycle_event(root, "session_eod_valuation", week.week_id, eod_data, eod_at)
        clock.set(eod_at)
        ledger.append_eod_valuation(
            root,
            week_id=week.week_id,
            data=eod_data,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )
        assert index == _state(root).next_session_index - 1 or session == week.sessions[-1]
    state = _state(root)
    close_at = _local(week.close_dt, 16, 10)
    close_data = _close_data(state, close_at, tag)
    anchor = _anchor_cycle_event(root, "cycle_close", week.week_id, close_data, close_at)
    clock.set(close_at)
    ledger.append_weekly_close(
        root,
        week_id=week.week_id,
        data=close_data,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )


def test_genesis_matches_execution_state_and_binds_13_gates(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier)
    frozen = ledger.read_ledger_records(root)[0]["payload"]["config"]
    expected_state = execution.empty_portfolio(10_000_000.0).as_dict()
    assert frozen["protocol_sha256"] == ledger.CANONICAL_PROTOCOL_SHA256
    assert frozen["trial_id"] == frozen["protocol_sha256"]
    assert frozen["initial_portfolio_state"] == expected_state
    assert frozen["initial_portfolio_state_sha256"] == execution.object_sha256(expected_state)
    assert set(frozen["readiness_gates"]) == ledger.READINESS_GATE_KEYS
    assert len(frozen["readiness_gates"]) == 13
    report = ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)
    assert report["valid"] is True
    assert report["confirmatory_oos"] is False


@pytest.mark.parametrize(
    "mutation",
    ["trial", "protocol_hash", "missing_gate", "failed_gate", "nonempty", "state_hash"],
)
def test_genesis_exact_schema_is_fail_closed(
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    mutation: str,
):
    config = _config()
    if mutation == "trial":
        config["trial_id"] = _hash("another-trial")
    elif mutation == "protocol_hash":
        config["protocol_sha256"] = _hash("raw-file-not-canonical")
    elif mutation == "missing_gate":
        config["readiness_gates"].pop(next(iter(ledger.READINESS_GATE_KEYS)))
    elif mutation == "failed_gate":
        config["readiness_gates"][next(iter(ledger.READINESS_GATE_KEYS))]["status"] = "FAILED"
    elif mutation == "nonempty":
        config["initial_portfolio_state"]["positions"] = {"000001.SZ": {}}
    else:
        config["initial_portfolio_state_sha256"] = _hash("wrong-state")
    with pytest.raises(ledger.LedgerValidationError):
        ledger.genesis_commitment_sha256(config, receipt_verifier=verifier)


def test_receipt_is_cryptographic_and_missing_verifier_stays_closed(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    closed = ledger.verify_ledger(root, trial_registry_root=registry)
    assert closed["structurally_valid"] is True
    assert closed["external_receipts_verified"] is False
    assert closed["valid"] is False

    config = _config()
    commitment = ledger.genesis_commitment_sha256(config, receipt_verifier=verifier)
    bad = _sign_receipt(commitment, _GENESIS_NOW - timedelta(minutes=1), "bad-signature-0001")
    signature = bytearray(base64.b64decode(bad["signature_base64"]))
    signature[-1] ^= 1
    bad["signature_base64"] = base64.b64encode(signature).decode("ascii")
    with pytest.raises(ledger.ReceiptVerificationError, match="signature is invalid"):
        verifier.verify(bad, expected_subject_sha256=commitment)
    assert "valid" not in inspect.signature(verifier.verify).parameters


def test_primary_registry_forbids_same_protocol_restart(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier)
    commitment = ledger.genesis_commitment_sha256(config, receipt_verifier=verifier)
    anchor = _sign_receipt(commitment, _GENESIS_NOW - timedelta(minutes=30), "genesis-anchor-0001")
    clock.set(_GENESIS_NOW)
    assert ledger.init_ledger(
        root,
        genesis_config=config,
        trial_registry_root=registry,
        external_anchor=anchor,
        receipt_verifier=verifier,
    ) == next(root.glob("000000_*.json"))
    with pytest.raises(ledger.LedgerConflictError, match="primary chain"):
        ledger.init_ledger(
            tmp_path / "restart",
            genesis_config=config,
            trial_registry_root=registry,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )


def test_incremental_calendar_requires_contiguity_source_and_signed_preopen_anchor(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    first = _segment(0, date(2026, 7, 17), date(2026, 8, 2), _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, first, _CALENDAR_NOW)
    assert len(_week_specs(root)) == 2
    second = _segment(1, date(2026, 8, 3), date(2026, 8, 30), _CALENDAR_NOW + timedelta(hours=1))
    _append_calendar(root, verifier, clock, second, _CALENDAR_NOW + timedelta(hours=2))
    report = ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)
    assert report["calendar_segment_count"] == 2
    assert report["external_receipts_verified"] is True

    gap = _segment(2, date(2026, 9, 1), date(2026, 9, 30), _CALENDAR_NOW + timedelta(hours=3))
    prepared = ledger.prepare_calendar_extension(root, segment=gap)
    anchor = _sign_receipt(
        prepared["record_commitment_sha256"], _CALENDAR_NOW + timedelta(hours=4), "calendar-gap-0001"
    )
    clock.set(_CALENDAR_NOW + timedelta(hours=5))
    with pytest.raises(ledger.LedgerValidationError, match="without overlap or gaps"):
        ledger.append_calendar_segment(root, segment=gap, external_anchor=anchor, receipt_verifier=verifier)


def test_every_record_is_anchored_and_every_session_has_open_then_eod(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    segment = _segment(0, date(2026, 7, 17), date(2026, 8, 9), _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    week = _week_specs(root)[0]
    _append_public_cycle(root, verifier, clock, week, "public")
    records = ledger.read_ledger_records(root)
    types = [item["record_type"] for item in records]
    assert types.count("session_open_execution") == len(week.sessions)
    assert types.count("session_eod_valuation") == len(week.sessions)
    assert "pending_sell_update" not in types
    assert types[-1] == "cycle_close"
    for record in records[1:]:
        assert set(record["payload"]) == ledger._ANCHORED_PAYLOAD_KEYS
        assert record["payload"]["external_anchor"]["subject_sha256"] == record["payload"]["record_commitment_sha256"]
    report = ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)
    assert report["completed_weeks"] == 1
    assert report["next_required_event"] == "decision"
    assert report["confirmatory_oos"] is False


def test_pending_sell_retry_occurs_only_at_next_session_open(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, _, _ = _init(tmp_path, clock, verifier)
    segment = _segment(0, date(2026, 7, 17), date(2026, 8, 9), _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    week = _week_specs(root)[0]

    state = _state(root)
    data = _decision_data(state, week, "pending")
    at = _local(week.decision_dt, 16)
    anchor = _anchor_cycle_event(root, "decision", week.week_id, data, at)
    clock.set(at)
    ledger.append_decision(root, week_id=week.week_id, data=data, external_anchor=anchor, receipt_verifier=verifier)
    state = _state(root)
    first_session = week.sessions[0]
    first_open = _open_data(state, first_session, _hash("open-p"), _hash("pending-one"), 1, "first")
    at = _local(first_session, 9, 40)
    anchor = _anchor_cycle_event(root, "session_open_execution", week.week_id, first_open, at)
    clock.set(at)
    ledger.append_open_execution(
        root, week_id=week.week_id, data=first_open, external_anchor=anchor, receipt_verifier=verifier
    )
    assert ledger.verify_ledger(root, receipt_verifier=verifier)["next_required_event"] == "session_eod_valuation"

    state = _state(root)
    first_eod = _eod_data(state, first_session, _hash("eod-p"), "first")
    at = _local(first_session, 16)
    anchor = _anchor_cycle_event(root, "session_eod_valuation", week.week_id, first_eod, at)
    clock.set(at)
    ledger.append_eod_valuation(
        root, week_id=week.week_id, data=first_eod, external_anchor=anchor, receipt_verifier=verifier
    )
    assert ledger.verify_ledger(root, receipt_verifier=verifier)["next_required_event"] == "session_open_execution"

    state = _state(root)
    second = week.sessions[1]
    retry = _open_data(state, second, _hash("retry-p"), _hash("pending-zero"), 0, "retry")
    at = _local(second, 9, 40)
    anchor = _anchor_cycle_event(root, "session_open_execution", week.week_id, retry, at)
    clock.set(at)
    ledger.append_open_execution(
        root, week_id=week.week_id, data=retry, external_anchor=anchor, receipt_verifier=verifier
    )
    assert _state(root).pending_sell_count == 0


@pytest.mark.parametrize(
    ("record_type", "at_factory"),
    [
        ("decision", lambda week: _local(week.decision_dt, 14, 59)),
        ("session_open_execution", lambda week: _local(week.execution_dt, 10)),
    ],
)
def test_market_event_windows_are_strict(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    record_type: str,
    at_factory: Any,
):
    root, _, _ = _init(tmp_path, clock, verifier)
    segment = _segment(0, date(2026, 7, 17), date(2026, 8, 9), _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    week = _week_specs(root)[0]
    if record_type == "decision":
        state = _state(root)
        data = _decision_data(state, week, "early")
        prepared = ledger.prepare_cycle_event(root, record_type="decision", week_id=week.week_id, data=data)
        at = at_factory(week)
        anchor = _sign_receipt(prepared["record_commitment_sha256"], at, "early-decision-0001")
        clock.set(at)
        with pytest.raises(ledger.LedgerValidationError):
            ledger.append_decision(
                root, week_id=week.week_id, data=data, external_anchor=anchor, receipt_verifier=verifier
            )
        return
    state = _state(root)
    decision = _decision_data(state, week, "late-open")
    decision_at = _local(week.decision_dt, 16)
    anchor = _anchor_cycle_event(root, "decision", week.week_id, decision, decision_at)
    clock.set(decision_at)
    ledger.append_decision(root, week_id=week.week_id, data=decision, external_anchor=anchor, receipt_verifier=verifier)
    state = _state(root)
    data = _open_data(state, week.execution_dt, _hash("late-p"), _hash("late-s"), 0, "late")
    prepared = ledger.prepare_cycle_event(root, record_type="session_open_execution", week_id=week.week_id, data=data)
    at = at_factory(week)
    anchor = _sign_receipt(prepared["record_commitment_sha256"], at, "late-open-anchor-1")
    clock.set(at)
    with pytest.raises(ledger.LedgerValidationError, match="MISSED_SESSION_OPEN_EXECUTION_DEADLINE"):
        ledger.append_open_execution(
            root, week_id=week.week_id, data=data, external_anchor=anchor, receipt_verifier=verifier
        )


def test_missed_deadline_and_abort_are_irreversible(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    segment = _segment(0, date(2026, 7, 17), date(2026, 8, 9), _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    week = _week_specs(root)[0]
    at = _local(week.execution_dt, 9, 30)
    event = {
        "schema": ledger.TERMINAL_SCHEMA,
        "outcome": "FAILED",
        "reason_code": "MISSED_DECISION_DEADLINE",
        "evidence_sha256": _hash("missed"),
        "details_sha256": _hash("missed-details"),
    }
    prepared = ledger.prepare_record_commitment(root, record_type="chain_abort", event_payload=event)
    anchor = _sign_receipt(prepared["record_commitment_sha256"], at, "deadline-abort-0001")
    clock.set(at)
    ledger.append_terminal(
        root,
        outcome="FAILED",
        reason_code="MISSED_DECISION_DEADLINE",
        evidence_sha256=event["evidence_sha256"],
        details_sha256=event["details_sha256"],
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    assert (
        ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)["operational_status"]
        == "FAILED"
    )
    with pytest.raises(ledger.LedgerValidationError, match="terminal"):
        ledger.append_terminal(
            root,
            outcome="FAILED",
            reason_code="SECOND_ABORT",
            evidence_sha256=_hash("again"),
            details_sha256=_hash("again-details"),
            external_anchor=anchor,
            receipt_verifier=verifier,
        )


def test_tamper_fork_and_copied_root_are_detected(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    segment = _segment(0, date(2026, 7, 17), date(2026, 8, 9), _CALENDAR_NOW - timedelta(hours=2))
    path = _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    with pytest.raises(ledger.LedgerValidationError, match="registry claim differs"):
        ledger.verify_ledger(copied, receipt_verifier=verifier, trial_registry_root=registry)
    fork = root / f"000001_{'f' * 64}.json"
    shutil.copyfile(path, fork)
    with pytest.raises(ledger.LedgerValidationError, match="fork detected"):
        ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)
    fork.unlink()
    value = json.loads(path.read_bytes())
    value["payload"]["event"]["segment"]["source_uri"] = "https://evil.example/tamper"
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ledger.LedgerValidationError, match="record hash mismatch"):
        ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)


def test_public_api_has_no_backdating_and_writes_with_o_excl(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    monkeypatch: pytest.MonkeyPatch,
):
    for function in (
        ledger.init_ledger,
        ledger.append_calendar_segment,
        ledger.append_decision,
        ledger.append_open_execution,
        ledger.append_eod_valuation,
        ledger.append_weekly_close,
        ledger.append_terminal,
        ledger.verify_ledger,
    ):
        assert "now" not in inspect.signature(function).parameters
        assert "recorded_at_utc" not in inspect.signature(function).parameters
    assert "recorded-at" not in inspect.getsource(ledger.build_parser)

    flags: list[int] = []
    original = ledger.os.open

    def wrapped(path: Any, value: int, mode: int = 0o777) -> int:
        if value & ledger.os.O_CREAT:
            flags.append(value)
        return original(path, value, mode)

    monkeypatch.setattr(ledger.os, "open", wrapped)
    _init(tmp_path, clock, verifier)
    assert len(flags) == 2
    assert all(value & ledger.os.O_EXCL for value in flags)


def _direct_append(
    state: ledger._ReplayState,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    record_type: str,
    event: dict[str, Any],
    recorded_at: datetime,
) -> ledger.LedgerRecord:
    previous = state.records[-1].record_hash
    commitment = ledger._record_commitment_sha256(
        primary_chain_id=ledger._primary_chain_id(state.config),
        previous_record_hash=previous,
        record_type=record_type,
        event_payload=event,
    )
    receipt = _sign_receipt(commitment, recorded_at - timedelta(seconds=1), f"direct-{len(state.records):06d}-anchor")
    wrapper = {
        "event": event,
        "record_commitment_sha256": commitment,
        "external_anchor": receipt,
    }
    data, raw = ledger._build_record(len(state.records), record_type, recorded_at, previous, wrapper)
    stored = ledger.LedgerRecord(
        path=ledger._record_path(state.genesis.path.parent, len(state.records), data["record_hash"]),
        data=data,
        raw=raw,
    )
    event_record, _, _ = ledger._unwrap_anchored_record(state, stored, receipt_verifier=verifier)
    if record_type == "decision":
        ledger._apply_decision(state, event_record)
    elif record_type == "session_open_execution":
        ledger._apply_open_execution(state, event_record)
    elif record_type == "session_eod_valuation":
        ledger._apply_eod(state, event_record)
    elif record_type == "cycle_close":
        ledger._apply_weekly_close(state, event_record)
    else:
        raise AssertionError(record_type)
    ledger._exclusive_write(stored.path, raw)
    state.records.append(stored)
    return event_record


def test_52_actual_cycles_form_stable_identity_but_never_self_confirm(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier)
    end = _OOS_START + timedelta(weeks=53, days=6)
    segment = _segment(0, date(2026, 7, 17), end, _CALENDAR_NOW - timedelta(hours=2))
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    state = ledger._validate_chain(root, receipt_verifier=verifier)

    for index in range(52):
        week = ledger._expected_week(state)
        assert week is not None
        tag = f"cycle-{index}"
        decision_data = _decision_data(state, week, tag)
        decision_event = {
            "week_id": week.week_id,
            "decision_dt": week.decision_dt.isoformat(),
            "execution_dt": week.execution_dt.isoformat(),
            "close_dt": week.close_dt.isoformat(),
            "data": decision_data,
        }
        decision_at = _local(week.decision_dt, 16) if index == 0 else _local(week.decision_dt, 16, 20)
        _direct_append(state, verifier, "decision", decision_event, decision_at)
        for session in week.sessions:
            portfolio_open = _hash(f"{tag}:{session}:open")
            pending = _hash(f"{tag}:{session}:pending")
            open_data = _open_data(state, session, portfolio_open, pending, 0, f"{tag}:{session}")
            open_event = {
                "week_id": week.week_id,
                "session_dt": session.isoformat(),
                "decision_hash": state.decision_record.record_hash,
                "data": open_data,
            }
            _direct_append(state, verifier, "session_open_execution", open_event, _local(session, 9, 40))
            portfolio_eod = _hash(f"{tag}:{session}:eod")
            eod_data = _eod_data(state, session, portfolio_eod, f"{tag}:{session}")
            eod_event = {"week_id": week.week_id, "session_dt": session.isoformat(), "data": eod_data}
            _direct_append(state, verifier, "session_eod_valuation", eod_event, _local(session, 16))
        close_at = _local(week.close_dt, 16, 10)
        close_data = _close_data(state, close_at, tag)
        close_event = {"week_id": week.week_id, "close_dt": week.close_dt.isoformat(), "data": close_data}
        _direct_append(state, verifier, "cycle_close", close_event, close_at)

    clock.set(state.records[-1].recorded_at_utc + timedelta(seconds=1))
    report = ledger.verify_ledger(root, receipt_verifier=verifier, trial_registry_root=registry)
    assert report["completed_weeks"] == 52
    assert report["structural_window_complete"] is True
    assert report["semantic_replay_verified"] is False
    assert report["confirmatory_oos"] is False
    identity = report["confirmation_window"]
    assert len(identity["weeks"]) == 52
    assert len({item["week_id"] for item in identity["weeks"]}) == 52
    assert identity["weeks"][0]["week_start"] == _OOS_START.isoformat()
    assert identity["confirmation_head_sha256"] == report["chain_head_sha256"]
    assert len(identity["identity_sha256"]) == 64
