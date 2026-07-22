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
_REGISTRY_PROVIDER = "TEST_EXTERNAL_PRIMARY_REGISTRY"
_REGISTRY_KEY_ID = "test-registry-rsa-key-2026"
_GENESIS_NOW = datetime(2026, 7, 17, 7, 30, tzinfo=UTC)
_CALENDAR_NOW = datetime(2026, 7, 17, 7, 45, tzinfo=UTC)
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


@pytest.fixture
def registry_verifier() -> ledger.RsaPkcs1v15Sha256ReceiptVerifier:
    return ledger.RsaPkcs1v15Sha256ReceiptVerifier(
        provider=_REGISTRY_PROVIDER,
        key_id=_REGISTRY_KEY_ID,
        modulus_hex=f"{_N:x}",
        public_exponent=_E,
    )


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _local(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(UTC)


def _sign_receipt(
    subject: str,
    issued_at: datetime,
    nonce: str,
    *,
    provider: str = _PROVIDER,
    key_id: str = _KEY_ID,
) -> dict[str, Any]:
    fields = {
        "schema": ledger.ANCHOR_RECEIPT_SCHEMA,
        "algorithm": ledger.ANCHOR_ALGORITHM,
        "provider": provider,
        "key_id": key_id,
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


def _segment(
    index: int,
    start: date,
    end: date,
    retrieved_at: datetime,
    *,
    previous_head: str,
) -> dict[str, Any]:
    sessions = _business_sessions(start, end)
    value = {
        "schema": ledger.CALENDAR_SEGMENT_SCHEMA,
        "segment_index": index,
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "sessions": [item.isoformat() for item in sessions],
        "week_labels": [ledger._week_id(item) for item in sessions],
        "source_id": "SSE_SZSE_OFFICIAL_CALENDAR",
        "source_uri": f"https://official.example/calendar/{index}.json",
        "source_sha256": _hash(f"official-calendar:{index}:{start}:{end}"),
        "source_published_at_utc": _utc(retrieved_at - timedelta(hours=1)),
        "source_retrieved_at_utc": _utc(retrieved_at),
        "previous_calendar_head_sha256": previous_head,
        "new_calendar_head_sha256": ledger.ZERO_HASH,
    }
    value["new_calendar_head_sha256"] = ledger.calendar_segment_head_sha256(value)
    return value


def _genesis_segment() -> dict[str, Any]:
    return _segment(
        0,
        date(2026, 7, 17),
        date(2026, 7, 26),
        _GENESIS_NOW - timedelta(days=1),
        previous_head=ledger.ZERO_HASH,
    )


def _config(registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier) -> dict[str, Any]:
    genesis_segment = _genesis_segment()
    initial_states = ledger.canonical_initial_state_sha256_by_book_scenario()
    data_bundle_sha256 = _hash("clean-data-bundle")
    data_contract_identity_sha256 = _hash("data-contract-identity")
    genesis_data_validation_report_sha256 = _hash("genesis-data-validation-report")
    genesis_data_chain_head_sha256 = ledger.data_chain_transition_sha256(
        data_contract_identity_sha256=data_contract_identity_sha256,
        previous_data_chain_head_sha256=ledger.ZERO_HASH,
        decision_session="2026-07-17",
        snapshot_cutoff_utc=_utc(_local(date(2026, 7, 17), 15)),
        data_manifest_sha256=data_bundle_sha256,
        data_validation_report_sha256=genesis_data_validation_report_sha256,
    )
    config = {
        "schema": ledger.GENESIS_CONFIG_SCHEMA,
        "protocol_id": ledger.PROTOCOL_ID,
        "protocol_sha256": ledger.CANONICAL_PROTOCOL_SHA256,
        "predecessor_protocol_sha256": ledger.PREDECESSOR_PROTOCOL_SHA256,
        "data_bundle_sha256": data_bundle_sha256,
        "data_contract_identity_sha256": data_contract_identity_sha256,
        "genesis_data_chain_head_sha256": genesis_data_chain_head_sha256,
        "genesis_data_validation_report_sha256": genesis_data_validation_report_sha256,
        "dependency_sha256": _dependencies(),
        "readiness_gates": _readiness(),
        "readiness_report_sha256": _hash("readiness-report"),
        "initial_state_sha256_by_book_scenario": initial_states,
        "initial_state_root_sha256": ledger._object_sha256(initial_states),
        "rng_identity": {
            "algorithm_version": ledger.RNG_ALGORITHM_VERSION,
            "root_seed": ledger.RNG_ROOT_SEED,
            "seed_count": len(ledger.RNG_SEEDS),
            "seeds": list(ledger.RNG_SEEDS),
            "identity_freeze": "all_arm_and_seed_symbol_identities_frozen_before_any_cost_or_capacity_replay",
        },
        "formal_start_eligibility": {
            "schema": "xs_chan_v2_1_formal_start_eligibility_v2",
            "readiness_report_sha256": _hash("readiness-report"),
            "readiness_anchor_receipt_sha256": _hash("readiness-anchor-receipt"),
            "readiness_completed_at_utc": _utc(_local(date(2026, 7, 17), 14, 30)),
            "eligibility_cutoff_utc": _utc(_local(date(2026, 7, 17), 15)),
            "candidate_decision_session": "2026-07-17",
            "candidate_execution_week": ledger._week_id(_OOS_START),
            "candidate_eligible_count": 50,
            "minimum_eligible_symbol_count": 50,
            "earlier_completed_weeks": [],
            "calendar_head_sha256": genesis_segment["new_calendar_head_sha256"],
            "data_snapshot_sha256": data_bundle_sha256,
            "planner_verifier_sha256": _hash("dependency:research_engine"),
            "evidence_sha256": _hash("formal-start-eligibility"),
        },
        "genesis_calendar_segment": genesis_segment,
        "genesis_calendar_head_sha256": genesis_segment["new_calendar_head_sha256"],
        "trial_id": ledger.CANONICAL_PROTOCOL_SHA256,
        "trial_registry_namespace": "czsc/xs-chan/confirmatory",
        "trial_registry_authority_sha256": ledger._object_sha256(registry_verifier.identity),
        "chain_role": "PRIMARY",
        "calendar_source_id": "SSE_SZSE_OFFICIAL_CALENDAR",
        "oos_start_date": _OOS_START.isoformat(),
        "min_complete_weeks": 52,
        "exchange_timezone": "Asia/Shanghai",
        "decision_close_local": "15:00:00",
        "execution_open_local": "09:30:00",
        "open_execution_deadline_local": "15:00:00",
        "session_close_local": "15:00:00",
    }
    config["formal_start_eligibility"]["evidence_sha256"] = ledger.formal_start_eligibility_evidence_sha256(
        config["formal_start_eligibility"]
    )
    return config


def _readiness_commitment(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = {
        "schema": ledger.READINESS_COMMITMENT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "trial_id": config["trial_id"],
        "data_contract_identity_sha256": config["data_contract_identity_sha256"],
        "created_at_utc": _utc(_local(date(2026, 7, 17), 14, 29)),
        "dependency_sha256": config["dependency_sha256"],
        "engineering_gates": config["readiness_gates"],
    }
    report["report_sha256"] = ledger._object_sha256(report)
    anchor = _sign_receipt(
        report["report_sha256"],
        _local(date(2026, 7, 17), 14, 30),
        "readiness-anchor-0001",
    )
    config["readiness_report_sha256"] = report["report_sha256"]
    eligibility = config["formal_start_eligibility"]
    eligibility["readiness_report_sha256"] = report["report_sha256"]
    eligibility["readiness_anchor_receipt_sha256"] = ledger._sha256_bytes(ledger._canonical_bytes(anchor))
    eligibility["readiness_completed_at_utc"] = anchor["issued_at_utc"]
    eligibility["evidence_sha256"] = ledger.formal_start_eligibility_evidence_sha256(eligibility)
    return report, anchor


def _registry_anchor(
    config: dict[str, Any],
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
) -> dict[str, Any]:
    claim = ledger.primary_registry_claim_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_receipt_verifier=registry_verifier,
    )
    return _sign_receipt(
        claim,
        _GENESIS_NOW - timedelta(minutes=20),
        "registry-claim-0001",
        provider=_REGISTRY_PROVIDER,
        key_id=_REGISTRY_KEY_ID,
    )


def _init(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    *,
    root_name: str = "ledger",
    registry_name: str = "registry",
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / root_name
    registry = tmp_path / registry_name
    config = _config(registry_verifier)
    readiness_report, readiness_anchor = _readiness_commitment(config)
    registry_anchor = _registry_anchor(config, verifier, registry_verifier)
    commitment = ledger.genesis_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
    )
    anchor = _sign_receipt(commitment, _GENESIS_NOW - timedelta(minutes=10), "genesis-anchor-0001")
    clock.set(_GENESIS_NOW)
    ledger.init_ledger(
        root,
        genesis_config=config,
        readiness_report=readiness_report,
        readiness_external_anchor=readiness_anchor,
        trial_registry_root=registry,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
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
    snapshot_cutoff_utc = _utc(_local(week.decision_dt, 15))
    data_manifest_sha256 = _hash(f"data-manifest:{tag}")
    data_validation_report_sha256 = _hash(f"data-validation:{tag}")
    data_chain_head_sha256 = ledger.data_chain_transition_sha256(
        data_contract_identity_sha256=state.config["data_contract_identity_sha256"],
        previous_data_chain_head_sha256=state.data_chain_head_sha256,
        decision_session=week.decision_dt,
        snapshot_cutoff_utc=snapshot_cutoff_utc,
        data_manifest_sha256=data_manifest_sha256,
        data_validation_report_sha256=data_validation_report_sha256,
    )
    return {
        "schema": ledger.DECISION_SCHEMA,
        "snapshot_cutoff_utc": snapshot_cutoff_utc,
        "engine_sha256": state.config["dependency_sha256"]["research_engine"],
        "decision_artifact_sha256": _hash(f"decision:{tag}"),
        "book_state_before_root_sha256": state.portfolio_sha256,
        "pending_sells_before_root_sha256": state.pending_sells_sha256,
        "calendar_state_sha256": week.calendar_state_sha256,
        "eligible_symbol_count": 50,
        "eligibility_evidence_sha256": _hash(f"eligibility:{tag}"),
        "data_manifest_sha256": data_manifest_sha256,
        "previous_data_chain_head_sha256": state.data_chain_head_sha256,
        "data_validation_report_sha256": data_validation_report_sha256,
        "data_chain_head_sha256": data_chain_head_sha256,
        "planner_artifact_sha256": _hash(f"planner:{tag}"),
    }


def _open_data(
    state: ledger._ReplayState,
    session: date,
    portfolio_after: str,
    pending_after: str,
    pending_count_after: int,
    tag: str,
    position_count_after: int = 0,
    terminal_share_receipt_count: int = 0,
) -> dict[str, Any]:
    assert state.decision_record is not None
    return {
        "schema": ledger.OPEN_EXECUTION_SCHEMA,
        "snapshot_cutoff_utc": _utc(_local(session, 9, 30)),
        "engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "decision_record_sha256": state.decision_record.record_hash,
        "previous_eod_record_sha256": state.last_eod_record.record_hash if state.last_eod_record else ledger.ZERO_HASH,
        "book_state_before_root_sha256": state.portfolio_sha256,
        "pending_sells_before_root_sha256": state.pending_sells_sha256,
        "pending_sell_count_before": state.pending_sell_count,
        "aggregate_position_count_before": state.position_count,
        "open_prices_sha256": _hash(f"open:{tag}"),
        "opening_auction_turnover_sha256": _hash(f"auction:{tag}"),
        "limit_state_sha256": _hash(f"limit:{tag}"),
        "corporate_actions_sha256": _hash(f"actions:{tag}"),
        "terminal_share_receipts_root_sha256": _hash(f"terminal-share-receipts:{tag}"),
        "terminal_share_receipt_count": terminal_share_receipt_count,
        "requested_orders_root_sha256": _hash(f"orders:{tag}"),
        "fills_root_sha256": _hash(f"fills:{tag}"),
        "fees_root_sha256": _hash(f"fees:{tag}"),
        "execution_result_root_sha256": _hash(f"result:{tag}"),
        "book_state_after_root_sha256": portfolio_after,
        "pending_sells_after_root_sha256": pending_after,
        "pending_sell_count_after": pending_count_after,
        "aggregate_position_count_after": position_count_after,
    }


def _eod_data(
    state: ledger._ReplayState,
    session: date,
    portfolio_after: str,
    tag: str,
    *,
    pending_settlement_count: int = 0,
) -> dict[str, Any]:
    assert state.last_open_record is not None
    cutoff = _utc(_local(session, 15))
    data = {
        "schema": (
            ledger.UNWIND_EOD_VALUATION_SCHEMA
            if len(state.completed_weeks) >= ledger.MIN_COMPLETE_WEEKS
            else ledger.EOD_VALUATION_SCHEMA
        ),
        "snapshot_cutoff_utc": cutoff,
        "engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "session_open_execution_sha256": state.last_open_record.record_hash,
        "raw_close_snapshot_sha256": _hash(f"close:{tag}"),
        "corporate_actions_sha256": _hash(f"eod-actions:{tag}"),
        "book_state_before_eod_root_sha256": state.portfolio_sha256,
        "book_state_root_sha256": portfolio_after,
        "pending_sells_root_sha256": state.pending_sells_sha256,
        "pending_sell_count": state.pending_sell_count,
        "aggregate_position_count": state.position_count,
        "nav_root_sha256": _hash(f"nav:{tag}"),
        "exposure_root_sha256": _hash(f"exposure:{tag}"),
        "turnover_root_sha256": _hash(f"turnover:{tag}"),
    }
    if len(state.completed_weeks) >= ledger.MIN_COMPLETE_WEEKS:
        manifest_sha = _hash(f"data-manifest:{tag}:unwind-eod")
        report_sha = _hash(f"data-validation:{tag}:unwind-eod")
        data.update(
            {
                "data_manifest_sha256": manifest_sha,
                "previous_data_chain_head_sha256": state.data_chain_head_sha256,
                "data_validation_report_sha256": report_sha,
                "data_chain_head_sha256": ledger.data_chain_transition_sha256(
                    data_contract_identity_sha256=state.config["data_contract_identity_sha256"],
                    previous_data_chain_head_sha256=state.data_chain_head_sha256,
                    decision_session=session,
                    snapshot_cutoff_utc=cutoff,
                    data_manifest_sha256=manifest_sha,
                    data_validation_report_sha256=report_sha,
                ),
                "aggregate_pending_settlements_root_sha256": _hash(f"pending-settlements:{tag}"),
                "aggregate_pending_settlement_count": pending_settlement_count,
            }
        )
    return data


def _close_data(state: ledger._ReplayState, recorded_at: datetime, tag: str) -> dict[str, Any]:
    assert state.last_eod_record is not None
    return {
        "schema": ledger.WEEKLY_CLOSE_SCHEMA,
        "snapshot_cutoff_utc": _utc(recorded_at),
        "execution_engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "statistics_engine_sha256": state.config["dependency_sha256"]["statistics_engine"],
        "eod_record_sha256": state.last_eod_record.record_hash,
        "book_state_root_sha256": state.portfolio_sha256,
        "pending_sells_root_sha256": state.pending_sells_sha256,
        "pending_sell_count": state.pending_sell_count,
        "aggregate_position_count": state.position_count,
        "weekly_returns_root_sha256": _hash(f"returns:{tag}"),
        "statistics_input_root_sha256": _hash(f"stats-input:{tag}"),
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
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    frozen = ledger.read_ledger_records(root)[0]["payload"]["config"]
    assert frozen["protocol_sha256"] == ledger.CANONICAL_PROTOCOL_SHA256
    assert frozen["predecessor_protocol_sha256"] == ledger.PREDECESSOR_PROTOCOL_SHA256
    assert frozen["trial_id"] == frozen["protocol_sha256"]
    assert set(frozen["initial_state_sha256_by_book_scenario"]) == set(ledger.BOOK_IDS)
    assert all(set(row) == set(ledger.SCENARIO_IDS) for row in frozen["initial_state_sha256_by_book_scenario"].values())
    assert frozen["initial_state_root_sha256"] == ledger.canonical_initial_state_root_sha256()
    assert frozen["initial_state_sha256_by_book_scenario"]["F"]["capacity_1x"] == execution.object_sha256(
        execution.empty_portfolio(100_000_000.0).as_dict()
    )
    assert set(frozen["readiness_gates"]) == ledger.READINESS_GATE_KEYS
    assert len(frozen["readiness_gates"]) == 13
    report = ledger.verify_ledger(
        root,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert report["valid"] is True
    assert report["readiness_report_sha256"] == config["readiness_report_sha256"]
    assert report["dependency_sha256"]["ledger_engine"] == config["dependency_sha256"]["ledger_engine"]
    assert report["calendar_segment_count"] == 1
    assert report["data_contract_identity_sha256"] == config["data_contract_identity_sha256"]
    assert report["genesis_data_chain_head_sha256"] == config["genesis_data_chain_head_sha256"]
    assert report["current_data_chain_head_sha256"] == config["genesis_data_chain_head_sha256"]
    assert report["confirmatory_oos"] is False


def test_invalid_readiness_cannot_create_ledger_or_registry(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
) -> None:
    root = tmp_path / "invalid-readiness-ledger"
    registry = tmp_path / "invalid-readiness-registry"
    config = _config(registry_verifier)
    readiness_report, readiness_anchor = _readiness_commitment(config)
    config["formal_start_eligibility"]["readiness_anchor_receipt_sha256"] = "9" * 64
    config["formal_start_eligibility"]["evidence_sha256"] = ledger.formal_start_eligibility_evidence_sha256(
        config["formal_start_eligibility"]
    )
    registry_anchor = _registry_anchor(config, verifier, registry_verifier)
    commitment = ledger.genesis_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
    )
    genesis_anchor = _sign_receipt(
        commitment,
        _GENESIS_NOW - timedelta(minutes=10),
        "invalid-readiness-genesis-anchor",
    )
    clock.set(_GENESIS_NOW)
    with pytest.raises(ledger.ReceiptVerificationError, match="different readiness receipt"):
        ledger.init_ledger(
            root,
            genesis_config=config,
            readiness_report=readiness_report,
            readiness_external_anchor=readiness_anchor,
            trial_registry_root=registry,
            registry_external_anchor=registry_anchor,
            registry_receipt_verifier=registry_verifier,
            external_anchor=genesis_anchor,
            receipt_verifier=verifier,
        )
    assert not root.exists()
    assert not registry.exists()


@pytest.mark.parametrize(
    "mutation",
    ["trial", "protocol_hash", "missing_gate", "failed_gate", "missing_book", "state_hash"],
)
def test_genesis_exact_schema_is_fail_closed(
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    mutation: str,
):
    config = _config(registry_verifier)
    if mutation == "trial":
        config["trial_id"] = _hash("another-trial")
    elif mutation == "protocol_hash":
        config["protocol_sha256"] = _hash("raw-file-not-canonical")
    elif mutation == "missing_gate":
        config["readiness_gates"].pop(next(iter(ledger.READINESS_GATE_KEYS)))
    elif mutation == "failed_gate":
        config["readiness_gates"][next(iter(ledger.READINESS_GATE_KEYS))]["status"] = "FAILED"
    elif mutation == "missing_book":
        config["initial_state_sha256_by_book_scenario"].pop("F")
    else:
        config["initial_state_sha256_by_book_scenario"]["F"]["gross"] = _hash("wrong-state")
    with pytest.raises(ledger.LedgerValidationError):
        ledger.primary_registry_claim_commitment_sha256(
            config,
            receipt_verifier=verifier,
            registry_receipt_verifier=registry_verifier,
        )


def test_genesis_rejects_preloaded_52_week_calendar(
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    config = _config(registry_verifier)
    preloaded = _segment(
        0,
        date(2026, 7, 17),
        _OOS_START + timedelta(weeks=52, days=6),
        _GENESIS_NOW - timedelta(days=1),
        previous_head=ledger.ZERO_HASH,
    )
    config["genesis_calendar_segment"] = preloaded
    config["genesis_calendar_head_sha256"] = preloaded["new_calendar_head_sha256"]
    config["formal_start_eligibility"]["calendar_head_sha256"] = preloaded["new_calendar_head_sha256"]
    with pytest.raises(ledger.LedgerValidationError, match="may not preload"):
        ledger.primary_registry_claim_commitment_sha256(
            config,
            receipt_verifier=verifier,
            registry_receipt_verifier=registry_verifier,
        )


def test_formal_start_cannot_omit_earlier_post_readiness_weeks(
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    config = _config(registry_verifier)
    longer_history = _segment(
        0,
        date(2026, 7, 3),
        date(2026, 7, 26),
        _GENESIS_NOW - timedelta(days=1),
        previous_head=ledger.ZERO_HASH,
    )
    config["genesis_calendar_segment"] = longer_history
    config["genesis_calendar_head_sha256"] = longer_history["new_calendar_head_sha256"]
    eligibility = config["formal_start_eligibility"]
    eligibility["calendar_head_sha256"] = longer_history["new_calendar_head_sha256"]
    eligibility["readiness_completed_at_utc"] = _utc(_local(date(2026, 7, 3), 14, 30))
    with pytest.raises(ledger.LedgerValidationError, match="omits or adds"):
        ledger.primary_registry_claim_commitment_sha256(
            config,
            receipt_verifier=verifier,
            registry_receipt_verifier=registry_verifier,
        )


def test_external_registry_receipt_cannot_be_replaced_by_local_or_timestamp_identity(
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    config = _config(registry_verifier)
    claim = ledger.primary_registry_claim_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_receipt_verifier=registry_verifier,
    )
    wrong_authority_receipt = _sign_receipt(
        claim,
        _GENESIS_NOW - timedelta(minutes=20),
        "wrong-registry-authority-0001",
    )
    with pytest.raises(ledger.ReceiptVerificationError, match="signer differs"):
        ledger.genesis_commitment_sha256(
            config,
            receipt_verifier=verifier,
            registry_external_anchor=wrong_authority_receipt,
            registry_receipt_verifier=registry_verifier,
        )


def test_decision_below_50_eligible_symbols_can_only_abort(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, _, _ = _init(tmp_path, clock, verifier, registry_verifier)
    state = _state(root)
    week = ledger._expected_week(state)
    assert week is not None
    data = _decision_data(state, week, "shortfall")
    data["eligible_symbol_count"] = 49
    at = _local(week.decision_dt, 16)
    anchor = _anchor_cycle_event(root, "decision", week.week_id, data, at)
    clock.set(at)
    with pytest.raises(ledger.LedgerValidationError, match="requires chain_abort"):
        ledger.append_decision(
            root,
            week_id=week.week_id,
            data=data,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "previous_data_chain_head_sha256",
            _hash("attacker-selected-previous-data-head"),
            "previous data-chain head differs",
        ),
        (
            "data_chain_head_sha256",
            _hash("attacker-selected-new-data-head"),
            "data-chain head is not derived",
        ),
    ],
)
def test_decision_data_chain_rejects_reparenting_and_forged_heads(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    field: str,
    replacement: str,
    message: str,
):
    root, _, _ = _init(tmp_path, clock, verifier, registry_verifier)
    state = _state(root)
    week = ledger._expected_week(state)
    assert week is not None
    data = _decision_data(state, week, f"forged-{field}")
    data[field] = replacement
    at = _local(week.decision_dt, 16)
    anchor = _anchor_cycle_event(root, "decision", week.week_id, data, at)
    clock.set(at)
    with pytest.raises(ledger.LedgerValidationError, match=message):
        ledger.append_decision(
            root,
            week_id=week.week_id,
            data=data,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )


def test_receipt_is_cryptographic_and_missing_verifier_stays_closed(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, _ = _init(tmp_path, clock, verifier, registry_verifier)
    closed = ledger.verify_ledger(root, trial_registry_root=registry)
    assert closed["structurally_valid"] is True
    assert closed["external_receipts_verified"] is False
    assert closed["valid"] is False

    config = _config(registry_verifier)
    registry_anchor = _registry_anchor(config, verifier, registry_verifier)
    commitment = ledger.genesis_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
    )
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
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    readiness_report, readiness_anchor = _readiness_commitment(config)
    registry_anchor = _registry_anchor(config, verifier, registry_verifier)
    commitment = ledger.genesis_commitment_sha256(
        config,
        receipt_verifier=verifier,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
    )
    anchor = _sign_receipt(commitment, _GENESIS_NOW - timedelta(minutes=10), "genesis-anchor-0001")
    clock.set(_GENESIS_NOW + timedelta(hours=1))
    assert ledger.init_ledger(
        root,
        genesis_config=config,
        readiness_report=readiness_report,
        readiness_external_anchor=readiness_anchor,
        trial_registry_root=registry,
        registry_external_anchor=registry_anchor,
        registry_receipt_verifier=registry_verifier,
        external_anchor=anchor,
        receipt_verifier=verifier,
    ) == next(root.glob("000000_*.json"))
    with pytest.raises(ledger.LedgerConflictError, match="primary chain"):
        ledger.init_ledger(
            tmp_path / "restart",
            genesis_config=config,
            readiness_report=readiness_report,
            readiness_external_anchor=readiness_anchor,
            trial_registry_root=registry,
            registry_external_anchor=registry_anchor,
            registry_receipt_verifier=registry_verifier,
            external_anchor=anchor,
            receipt_verifier=verifier,
        )


def test_incremental_calendar_requires_contiguity_source_and_signed_preopen_anchor(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    first = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 2),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
    _append_calendar(root, verifier, clock, first, _CALENDAR_NOW)
    assert len(_week_specs(root)) == 2
    second = _segment(
        2,
        date(2026, 8, 3),
        date(2026, 8, 30),
        _CALENDAR_NOW + timedelta(hours=1),
        previous_head=first["new_calendar_head_sha256"],
    )
    _append_calendar(root, verifier, clock, second, _CALENDAR_NOW + timedelta(hours=2))
    report = ledger.verify_ledger(
        root,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert report["calendar_segment_count"] == 3
    assert report["external_receipts_verified"] is True

    wrong_head = _segment(
        3,
        date(2026, 8, 31),
        date(2026, 9, 6),
        _CALENDAR_NOW + timedelta(hours=3),
        previous_head=ledger.ZERO_HASH,
    )
    with pytest.raises(ledger.LedgerValidationError, match="previous head"):
        ledger.prepare_calendar_extension(root, segment=wrong_head)

    gap = _segment(
        3,
        date(2026, 9, 1),
        date(2026, 9, 30),
        _CALENDAR_NOW + timedelta(hours=3),
        previous_head=second["new_calendar_head_sha256"],
    )
    with pytest.raises(ledger.LedgerValidationError, match="without overlap or gaps"):
        ledger.prepare_calendar_extension(root, segment=gap)


def test_every_record_is_anchored_and_every_session_has_open_then_eod(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    segment = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 9),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
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
    report = ledger.verify_ledger(
        root,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert report["completed_weeks"] == 1
    assert report["next_required_event"] == "decision"
    assert report["confirmatory_oos"] is False


def test_pending_sell_retry_occurs_only_at_next_session_open(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, _, config = _init(tmp_path, clock, verifier, registry_verifier)
    segment = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 9),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
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
        ("session_open_execution", lambda week: _local(week.execution_dt, 15)),
    ],
)
def test_market_event_windows_are_strict(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    record_type: str,
    at_factory: Any,
):
    root, _, config = _init(tmp_path, clock, verifier, registry_verifier)
    segment = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 9),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
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
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    segment = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 9),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
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
        ledger.verify_ledger(
            root,
            receipt_verifier=verifier,
            trial_registry_root=registry,
            registry_receipt_verifier=registry_verifier,
        )["lifecycle_status"]
        == "PRIMARY_CHAIN_TERMINATED_INVALID"
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


def test_tamper_and_fork_are_detected_but_chain_identity_is_path_portable(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    segment = _segment(
        1,
        date(2026, 7, 27),
        date(2026, 8, 9),
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
    path = _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    copied_report = ledger.verify_ledger(
        copied,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert copied_report["valid"] is True
    fork = root / f"000001_{'f' * 64}.json"
    shutil.copyfile(path, fork)
    with pytest.raises(ledger.LedgerValidationError, match="fork detected"):
        ledger.verify_ledger(
            root,
            receipt_verifier=verifier,
            trial_registry_root=registry,
            registry_receipt_verifier=registry_verifier,
        )
    fork.unlink()
    value = json.loads(path.read_bytes())
    value["payload"]["event"]["segment"]["source_uri"] = "https://evil.example/tamper"
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ledger.LedgerValidationError, match="record hash mismatch"):
        ledger.verify_ledger(
            root,
            receipt_verifier=verifier,
            trial_registry_root=registry,
            registry_receipt_verifier=registry_verifier,
        )


def test_public_api_has_no_backdating_and_writes_with_o_excl(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
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
        ledger.append_final_evaluation,
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
    _init(tmp_path, clock, verifier, registry_verifier)
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


def test_segment_end_eod_and_cycle_close_do_not_require_future_calendar(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, _, _ = _init(tmp_path, clock, verifier, registry_verifier)
    state = ledger._validate_chain(root, receipt_verifier=verifier)
    week = ledger._expected_week(state)
    assert week is not None
    decision_data = _decision_data(state, week, "segment-end")
    _direct_append(
        state,
        verifier,
        "decision",
        {
            "week_id": week.week_id,
            "decision_kind": "CONFIRMATORY_CYCLE",
            "decision_dt": week.decision_dt.isoformat(),
            "execution_dt": week.execution_dt.isoformat(),
            "close_dt": week.close_dt.isoformat(),
            "data": decision_data,
        },
        _local(week.decision_dt, 16),
    )
    for session in week.sessions:
        open_data = _open_data(
            state,
            session,
            _hash(f"segment-end:{session}:open"),
            _hash(f"segment-end:{session}:pending"),
            0,
            f"segment-end:{session}",
        )
        _direct_append(
            state,
            verifier,
            "session_open_execution",
            {
                "week_id": week.week_id,
                "session_dt": session.isoformat(),
                "decision_hash": state.decision_record.record_hash,
                "data": open_data,
            },
            _local(session, 9, 40),
        )
        eod_data = _eod_data(
            state,
            session,
            _hash(f"segment-end:{session}:eod"),
            f"segment-end:{session}",
        )
        _direct_append(
            state,
            verifier,
            "session_eod_valuation",
            {"week_id": week.week_id, "session_dt": session.isoformat(), "data": eod_data},
            _local(session, 16),
        )
    event, _, deadline = ledger._event_window(state)
    assert event == "cycle_close"
    assert deadline == datetime.max.replace(tzinfo=UTC)
    close_at = _local(week.close_dt + timedelta(days=1), 10)
    close_data = _close_data(state, close_at, "segment-end")
    _direct_append(
        state,
        verifier,
        "cycle_close",
        {"week_id": week.week_id, "close_dt": week.close_dt.isoformat(), "data": close_data},
        close_at,
    )
    assert len(state.completed_weeks) == 1


def test_52_actual_cycles_form_stable_identity_but_never_self_confirm(
    tmp_path: Path,
    clock: _Clock,
    verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
    registry_verifier: ledger.RsaPkcs1v15Sha256ReceiptVerifier,
):
    root, registry, config = _init(tmp_path, clock, verifier, registry_verifier)
    end = _OOS_START + timedelta(weeks=53, days=6)
    segment = _segment(
        1,
        date(2026, 7, 27),
        end,
        _CALENDAR_NOW - timedelta(hours=2),
        previous_head=config["genesis_calendar_head_sha256"],
    )
    _append_calendar(root, verifier, clock, segment, _CALENDAR_NOW)
    state = ledger._validate_chain(root, receipt_verifier=verifier)

    for index in range(52):
        week = ledger._expected_week(state)
        assert week is not None
        tag = f"cycle-{index}"
        decision_data = _decision_data(state, week, tag)
        decision_event = {
            "week_id": week.week_id,
            "decision_kind": "CONFIRMATORY_CYCLE",
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
            open_data = _open_data(
                state,
                session,
                portfolio_open,
                pending,
                0,
                f"{tag}:{session}",
                position_count_after=1 if index == 51 else 0,
            )
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
    report = ledger.verify_ledger(
        root,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert report["completed_weeks"] == 52
    assert report["structural_window_complete"] is True
    assert report["semantic_replay_verified"] is False
    assert report["confirmatory_oos"] is False
    identity = report["confirmation_window"]
    assert len(identity["weeks"]) == 52
    assert len({item["week_id"] for item in identity["weeks"]}) == 52
    assert identity["weeks"][0]["week_start"] == _OOS_START.isoformat()
    assert identity["confirmation_head_sha256"] == report["chain_head_sha256"]
    assert identity["confirmation_data_chain_head_sha256"] == report["current_data_chain_head_sha256"]
    assert len(identity["identity_sha256"]) == 64

    state = _state(root)
    unwind_session = ledger._next_unwind_session(state)
    assert unwind_session is not None
    decision_at = state.records[-1].recorded_at_utc + timedelta(minutes=10)
    exit_orders_root = _hash("unwind-exit-orders")
    unwind_data_manifest_sha256 = _hash("unwind-data-manifest")
    unwind_data_validation_report_sha256 = _hash("unwind-data-validation")
    unwind_data_chain_head_sha256 = ledger.data_chain_transition_sha256(
        data_contract_identity_sha256=state.config["data_contract_identity_sha256"],
        previous_data_chain_head_sha256=state.data_chain_head_sha256,
        decision_session=state.completed_weeks[-1]["close_dt"],
        snapshot_cutoff_utc=_utc(decision_at),
        data_manifest_sha256=unwind_data_manifest_sha256,
        data_validation_report_sha256=unwind_data_validation_report_sha256,
    )
    unwind_decision = {
        "schema": ledger.UNWIND_DECISION_SCHEMA,
        "snapshot_cutoff_utc": _utc(decision_at),
        "engine_sha256": state.config["dependency_sha256"]["execution_engine"],
        "data_manifest_sha256": unwind_data_manifest_sha256,
        "previous_data_chain_head_sha256": state.data_chain_head_sha256,
        "data_validation_report_sha256": unwind_data_validation_report_sha256,
        "data_chain_head_sha256": unwind_data_chain_head_sha256,
        "confirmation_window_identity_sha256": identity["identity_sha256"],
        "book_state_before_root_sha256": state.portfolio_sha256,
        "pending_sells_before_root_sha256": state.pending_sells_sha256,
        "aggregate_position_count_before": state.position_count,
        "aggregate_pending_sell_count_before": state.pending_sell_count,
        "exit_orders_root_sha256": exit_orders_root,
        "full_exit_coverage_root_sha256": ledger._object_sha256(
            {
                "book_state_before_root_sha256": state.portfolio_sha256,
                "aggregate_position_count_before": state.position_count,
                "exit_orders_root_sha256": exit_orders_root,
            }
        ),
        "book_state_after_decision_root_sha256": _hash("unwind-decision-book-root"),
        "pending_sells_after_decision_root_sha256": _hash("unwind-decision-pending-root"),
        "aggregate_pending_sell_count_after_decision": state.position_count,
        "unwind_evidence_root_sha256": _hash("unwind-decision-evidence"),
    }
    anchor = _anchor_cycle_event(
        root,
        "decision",
        ledger.UNWIND_WEEK_ID,
        unwind_decision,
        decision_at,
    )
    clock.set(decision_at)
    ledger.append_decision(
        root,
        week_id=ledger.UNWIND_WEEK_ID,
        data=unwind_decision,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )

    state = _state(root)
    spoof_at = decision_at + timedelta(minutes=1)
    spoof_final = {
        "schema": ledger.FINAL_EVALUATION_SCHEMA,
        "snapshot_cutoff_utc": _utc(spoof_at),
        "confirmation_window_identity_sha256": identity["identity_sha256"],
        "statistics_result_sha256": _hash("spoof-statistics"),
        "semantic_replay_evidence_sha256": _hash("spoof-replay"),
        "evaluation_status": "INCONCLUSIVE_FACTOR",
        "post_window_unwind_completed": True,
        "post_window_unwind_session_count": 0,
        "post_window_unwind_cost_cny": 0.0,
        "remaining_position_count": 0,
        "remaining_pending_sell_count": 0,
        "remaining_pending_settlement_count": 0,
        "unwind_completion_book_state_root_sha256": state.portfolio_sha256,
        "unwind_completion_pending_sells_root_sha256": state.pending_sells_sha256,
        "unwind_completion_pending_settlements_root_sha256": _hash("spoof-pending-settlements"),
        "unwind_evidence_sha256": _hash("caller-asserted-unwind"),
        "final_evaluation_artifact_sha256": _hash("spoof-final"),
    }
    prepared = ledger.prepare_final_evaluation(root, data=spoof_final)
    spoof_anchor = _sign_receipt(
        prepared["record_commitment_sha256"],
        spoof_at - timedelta(seconds=1),
        "spoof-final-anchor-0001",
    )
    clock.set(spoof_at)
    with pytest.raises(ledger.LedgerValidationError, match="irreversible all-book unwind-completion"):
        ledger.append_final_evaluation(
            root,
            data=spoof_final,
            external_anchor=spoof_anchor,
            receipt_verifier=verifier,
        )

    state = _state(root)
    unwind_open = _open_data(
        state,
        unwind_session,
        _hash("unwind-open-book-root"),
        _hash("unwind-open-pending-root"),
        0,
        "unwind",
        position_count_after=0,
    )
    open_at = _local(unwind_session, 9, 40)
    anchor = _anchor_cycle_event(
        root,
        "session_open_execution",
        ledger.UNWIND_WEEK_ID,
        unwind_open,
        open_at,
    )
    clock.set(open_at)
    ledger.append_open_execution(
        root,
        week_id=ledger.UNWIND_WEEK_ID,
        data=unwind_open,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    state = _state(root)
    unwind_eod = _eod_data(
        state,
        unwind_session,
        _hash("unwind-eod-book-root"),
        "unwind",
        pending_settlement_count=1,
    )
    eod_at = _local(unwind_session, 16)
    anchor = _anchor_cycle_event(
        root,
        "session_eod_valuation",
        ledger.UNWIND_WEEK_ID,
        unwind_eod,
        eod_at,
    )
    clock.set(eod_at)
    ledger.append_eod_valuation(
        root,
        week_id=ledger.UNWIND_WEEK_ID,
        data=unwind_eod,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )

    state = _state(root)
    assert state.phase == "UNWIND_AWAIT_SESSION_OPEN"
    assert state.pending_settlement_count == 1
    settlement_session = ledger._next_unwind_session(state)
    assert settlement_session is not None
    settlement_open = _open_data(
        state,
        settlement_session,
        _hash("unwind-settlement-open-book-root"),
        state.pending_sells_sha256,
        0,
        "unwind-settlement",
        position_count_after=0,
    )
    settlement_open_at = _local(settlement_session, 9, 40)
    anchor = _anchor_cycle_event(
        root,
        "session_open_execution",
        ledger.UNWIND_WEEK_ID,
        settlement_open,
        settlement_open_at,
    )
    clock.set(settlement_open_at)
    ledger.append_open_execution(
        root,
        week_id=ledger.UNWIND_WEEK_ID,
        data=settlement_open,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    state = _state(root)
    settlement_eod = _eod_data(
        state,
        settlement_session,
        _hash("unwind-settlement-eod-book-root"),
        "unwind-settlement",
    )
    settlement_eod_at = _local(settlement_session, 16)
    anchor = _anchor_cycle_event(
        root,
        "session_eod_valuation",
        ledger.UNWIND_WEEK_ID,
        settlement_eod,
        settlement_eod_at,
    )
    clock.set(settlement_eod_at)
    ledger.append_eod_valuation(
        root,
        week_id=ledger.UNWIND_WEEK_ID,
        data=settlement_eod,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )

    state = _state(root)
    assert state.phase == "UNWIND_COMPLETE_PENDING_REPLAY"
    assert state.pending_settlement_count == 0
    assert ledger._event_window(state) is None
    with pytest.raises(ledger.LedgerValidationError, match="permits only final evaluation or abort"):
        ledger._apply_calendar_segment(
            state,
            state.records[-1],
            external_anchor={},
            verified_receipt=None,
        )
    derived_unwind_evidence = ledger._object_sha256(
        {
            "decision_unwind_evidence_root_sha256": state.unwind_evidence_root_sha256,
            "completion_book_state_root_sha256": state.portfolio_sha256,
            "completion_pending_sells_root_sha256": state.pending_sells_sha256,
            "remaining_position_count": state.position_count,
            "remaining_pending_sell_count": state.pending_sell_count,
            "remaining_pending_settlement_count": state.pending_settlement_count,
            "completion_pending_settlements_root_sha256": state.pending_settlements_sha256,
            "unwind_session_count": state.unwind_session_count,
            "post_window_unwind_cost_cny": 123.45,
        }
    )
    final_at = settlement_eod_at + timedelta(minutes=10)
    final_data = {
        "schema": ledger.FINAL_EVALUATION_SCHEMA,
        "snapshot_cutoff_utc": _utc(final_at),
        "confirmation_window_identity_sha256": identity["identity_sha256"],
        "statistics_result_sha256": _hash("statistics-result"),
        "semantic_replay_evidence_sha256": _hash("semantic-replay-evidence"),
        "evaluation_status": "INCONCLUSIVE_FACTOR",
        "post_window_unwind_completed": True,
        "post_window_unwind_session_count": 2,
        "post_window_unwind_cost_cny": 123.45,
        "remaining_position_count": 0,
        "remaining_pending_sell_count": 0,
        "remaining_pending_settlement_count": 0,
        "unwind_completion_book_state_root_sha256": state.portfolio_sha256,
        "unwind_completion_pending_sells_root_sha256": state.pending_sells_sha256,
        "unwind_completion_pending_settlements_root_sha256": state.pending_settlements_sha256,
        "unwind_evidence_sha256": derived_unwind_evidence,
        "final_evaluation_artifact_sha256": _hash("final-evaluation-artifact"),
    }
    stale_final_data = {**final_data, "snapshot_cutoff_utc": _utc(eod_at)}
    stale_prepared = ledger.prepare_final_evaluation(root, data=stale_final_data)
    stale_anchor = _sign_receipt(
        stale_prepared["record_commitment_sha256"],
        final_at - timedelta(seconds=1),
        "stale-final-evaluation-anchor-0001",
    )
    clock.set(final_at)
    with pytest.raises(ledger.LedgerTimingError, match="precedes the causal market boundary"):
        ledger.append_final_evaluation(
            root,
            data=stale_final_data,
            external_anchor=stale_anchor,
            receipt_verifier=verifier,
        )
    prepared = ledger.prepare_final_evaluation(root, data=final_data)
    anchor = _sign_receipt(
        prepared["record_commitment_sha256"],
        final_at - timedelta(seconds=1),
        "final-evaluation-anchor-0001",
    )
    clock.set(final_at)
    ledger.append_final_evaluation(
        root,
        data=final_data,
        external_anchor=anchor,
        receipt_verifier=verifier,
    )
    final_report = ledger.verify_ledger(
        root,
        receipt_verifier=verifier,
        trial_registry_root=registry,
        registry_receipt_verifier=registry_verifier,
    )
    assert final_report["lifecycle_status"] == "PRIMARY_WINDOW_COMPLETE_PENDING_REPLAY"
    assert final_report["declared_lifecycle_status"] == "EVALUATED"
    assert final_report["final_evaluation_recorded"] is True
    assert final_report["final_evaluation_verified"] is False
    assert final_report["confirmation_head_sha256"] == identity["confirmation_head_sha256"]
    assert final_report["chain_head_sha256"] != identity["confirmation_head_sha256"]
    assert final_report["current_data_chain_head_sha256"] == settlement_eod["data_chain_head_sha256"]
    assert final_report["current_data_chain_head_sha256"] != identity["confirmation_data_chain_head_sha256"]
    assert final_report["unwind_summary"]["session_count"] == 2
