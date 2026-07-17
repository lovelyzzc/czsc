from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_oos_ledger as ledger  # noqa: E402

PROTOCOL_HASH = "a" * 64
DEPENDENCY_HASHES = {name: hashlib.sha256(name.encode()).hexdigest() for name in ledger.EXPECTED_DEPENDENCY_KEYS}
ENGINE_HASH = DEPENDENCY_HASHES["research_engine"]
EXECUTION_ENGINE_HASH = DEPENDENCY_HASHES["execution_engine"]
FREEZE_NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


class _TestClock:
    def __init__(self, value: datetime):
        self.value = value

    def set(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _TestClock:
    fake = _TestClock(FREEZE_NOW)
    monkeypatch.setattr(ledger, "_utc_now", fake)
    return fake


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _business_days(start: date, count: int) -> list[date]:
    sessions: list[date] = []
    current = start
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def _calendar(weeks: int = 55) -> list[date]:
    return _business_days(date(2026, 7, 20), weeks * 5 + 1)


def _decision_close(decision_dt: date) -> datetime:
    return datetime.combine(decision_dt, datetime.min.time(), tzinfo=UTC).replace(hour=7)


def _decision_now(decision_dt: date) -> datetime:
    # 16:00 Asia/Shanghai, after the frozen 15:00 close.
    return datetime.combine(decision_dt, datetime.min.time(), tzinfo=UTC).replace(hour=8)


def _execution_open(exec_dt: date) -> datetime:
    return datetime.combine(exec_dt, datetime.min.time(), tzinfo=UTC).replace(hour=1, minute=30)


def _execution_now(exec_dt: date) -> datetime:
    # 10:00 Asia/Shanghai, after the frozen 09:30 execution time.
    return datetime.combine(exec_dt, datetime.min.time(), tzinfo=UTC).replace(hour=2)


def _execution_close(exec_dt: date) -> datetime:
    return datetime.combine(exec_dt, datetime.min.time(), tzinfo=UTC).replace(hour=7)


def _record_hash(path: Path) -> str:
    return str(json.loads(path.read_bytes())["record_hash"])


def _week_pair(sessions: list[date], week: int) -> tuple[date, date]:
    return sessions[4 + week * 5], sessions[5 + week * 5]


def _seeded_identities(arm: str, tag: str) -> dict[str, str]:
    return {str(seed): _hash(f"{arm}:{seed}:{tag}") for seed in ledger.EXPECTED_SEED_IDS}


def _decision_data(decision_dt: date, tag: str = "base") -> dict[str, Any]:
    return {
        "schema": ledger.DECISION_DATA_SCHEMA,
        "asof_snapshot_sha256": _hash(f"asof:{tag}"),
        "asof_cutoff_utc": _utc_text(_decision_close(decision_dt)),
        "engine_sha256": ENGINE_HASH,
        "plan_sha256": _hash(f"plan:{tag}"),
        "gate_identity_sha256": _hash(f"gate:{tag}"),
        "reference_books_sha256": _hash(f"books:{tag}"),
        "adv_snapshot_sha256": _hash(f"adv:{tag}"),
        "state_snapshot_sha256": _hash(f"state:{tag}"),
        "ma_snapshot_sha256": _hash(f"ma:{tag}"),
        "seed_ids": list(ledger.EXPECTED_SEED_IDS),
        "arm_identity_sha256": {
            "F": _hash(f"F:{tag}"),
            "FC": _hash(f"FC:{tag}"),
            "FMA": _hash(f"FMA:{tag}"),
            "R_MATCH": _seeded_identities("R_MATCH", tag),
            "FGR": _seeded_identities("FGR", tag),
            "FMGR": _seeded_identities("FMGR", tag),
        },
        "invariants": {
            "blocked_slots_applied": True,
            "target_count_lte_50": True,
            "identities_frozen_before_cost_replay": True,
        },
    }


def _execution_data(exec_dt: date, gate_hash: str, tag: str = "base") -> dict[str, Any]:
    return {
        "schema": ledger.EXECUTION_DATA_SCHEMA,
        "engine_sha256": EXECUTION_ENGINE_HASH,
        "execution_snapshot_sha256": _hash(f"execution:{tag}"),
        "snapshot_cutoff_utc": _utc_text(_execution_open(exec_dt)),
        "raw_ohlc_sha256": _hash(f"ohlc:{tag}"),
        "stk_limit_sha256": _hash(f"limits:{tag}"),
        "corporate_actions_sha256": _hash(f"actions:{tag}"),
        "requested_orders_sha256": _hash(f"orders:{tag}"),
        "fills_sha256": _hash(f"fills:{tag}"),
        "fees_sha256": _hash(f"fees:{tag}"),
        "portfolio_state_sha256": _hash(f"portfolio:{tag}"),
        "pending_sells_sha256": _hash(f"pending:{tag}"),
        "gate_identity_sha256": gate_hash,
        "scenario_replay_sha256": _hash(f"replay:{tag}"),
        "invariants": {
            "all_arms_present": True,
            "no_negative_cash": True,
            "max_positions_lte_50": True,
            "same_identity_all_cost_scenarios": True,
            "exact_seed_set": True,
            "corporate_actions_applied": True,
            "terminal_events_resolved": True,
        },
    }


def _init(
    root: Path,
    clock: _TestClock,
    *,
    weeks: int = 55,
    sessions: list[date] | None = None,
    start_index: int = 0,
) -> list[date]:
    calendar = sessions or _calendar(weeks)
    clock.set(FREEZE_NOW)
    ledger.init_ledger(
        root,
        protocol_sha256=PROTOCOL_HASH,
        dependency_sha256=DEPENDENCY_HASHES,
        calendar_sessions=calendar,
        oos_start_date=calendar[start_index],
    )
    return calendar


def _append_complete_week(
    root: Path,
    sessions: list[date],
    week: int,
    clock: _TestClock,
) -> tuple[Path, Path]:
    decision_dt, exec_dt = _week_pair(sessions, week)
    decision_data = _decision_data(decision_dt, tag=f"week-{week}")
    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(
        root,
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        data=decision_data,
    )
    clock.set(_execution_now(exec_dt))
    execution = ledger.append_execution(
        root,
        decision_hash=_record_hash(decision),
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        data=_execution_data(exec_dt, decision_data["gate_identity_sha256"], tag=f"week-{week}"),
    )
    return decision, execution


def test_freeze_locks_hashes_calendar_and_is_only_byte_idempotent(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _calendar()
    first = _init(root, clock, sessions=sessions)
    freeze_path = next(root.glob("000000_*.json"))

    record = json.loads(freeze_path.read_bytes())
    assert record["record_type"] == "freeze"
    assert record["payload"]["protocol_sha256"] == PROTOCOL_HASH
    assert record["payload"]["dependency_sha256"] == DEPENDENCY_HASHES
    assert record["payload"]["decision_time_local"] == "15:00:00"
    assert record["payload"]["execution_record_deadline_local"] == "15:00:00"
    verified = ledger.verify_ledger(root)
    assert record["payload"]["calendar_sha256"] == verified["calendar_sha256"]
    assert verified["dependency_sha256"] == DEPENDENCY_HASHES

    clock.set(FREEZE_NOW)
    repeated = ledger.init_ledger(
        root,
        protocol_sha256=PROTOCOL_HASH,
        dependency_sha256=DEPENDENCY_HASHES,
        calendar_sessions=sessions,
        oos_start_date=sessions[0],
    )
    assert repeated == freeze_path
    assert len(list(root.glob("*.json"))) == 1
    assert first == sessions

    clock.set(FREEZE_NOW + timedelta(seconds=1))
    with pytest.raises(ledger.LedgerConflictError, match="different freeze bytes"):
        ledger.init_ledger(
            root,
            protocol_sha256=PROTOCOL_HASH,
            dependency_sha256=DEPENDENCY_HASHES,
            calendar_sessions=sessions,
            oos_start_date=sessions[0],
        )


def test_freeze_requires_exactly_52_available_weekly_pairs(tmp_path: Path, clock: _TestClock):
    short_calendar = _calendar(weeks=10)
    with pytest.raises(ledger.LedgerValidationError, match="52 complete weekly"):
        _init(tmp_path / "short", clock, sessions=short_calendar)

    sessions = _calendar()
    clock.set(FREEZE_NOW)
    with pytest.raises(ledger.LedgerValidationError, match="must be frozen at 52"):
        ledger.init_ledger(
            tmp_path / "lower-threshold",
            protocol_sha256=PROTOCOL_HASH,
            dependency_sha256=DEPENDENCY_HASHES,
            calendar_sessions=sessions,
            oos_start_date=sessions[0],
            min_complete_weeks=51,
        )
    with pytest.raises(ledger.LedgerValidationError, match="exact frozen dependency closure"):
        ledger.init_ledger(
            tmp_path / "missing-dependency",
            protocol_sha256=PROTOCOL_HASH,
            dependency_sha256={name: digest for name, digest in DEPENDENCY_HASHES.items() if name != "replay_verifier"},
            calendar_sessions=sessions,
            oos_start_date=sessions[0],
        )


def test_freeze_rejects_nonfrozen_market_clock_and_late_genesis(tmp_path: Path, clock: _TestClock):
    sessions = _calendar()
    common = {
        "protocol_sha256": PROTOCOL_HASH,
        "dependency_sha256": DEPENDENCY_HASHES,
        "calendar_sessions": sessions,
        "oos_start_date": sessions[0],
    }

    clock.set(FREEZE_NOW)
    with pytest.raises(ledger.LedgerValidationError, match="exchange_timezone must be frozen at Asia/Shanghai"):
        ledger.init_ledger(tmp_path / "utc", exchange_timezone="UTC", **common)
    with pytest.raises(ledger.LedgerValidationError, match="execution_time_local must be frozen at 09:30:00"):
        ledger.init_ledger(tmp_path / "late-execution", execution_time_local="09:31:00", **common)

    first_decision_dt, _ = _week_pair(sessions, 0)
    clock.set(_decision_close(first_decision_dt))
    with pytest.raises(ledger.LedgerTimingError, match="before the first OOS decision close"):
        ledger.init_ledger(tmp_path / "at-first-close", **common)

    # 2026-07-20 00:00:00 Asia/Shanghai: the genesis cannot be created
    # after OOS has already started, even before the first Friday close.
    clock.set(datetime(2026, 7, 19, 16, tzinfo=UTC))
    with pytest.raises(ledger.LedgerTimingError, match="before oos_start_date begins"):
        ledger.init_ledger(tmp_path / "at-oos-start", **common)


def test_verifier_revalidates_frozen_market_clock_and_genesis(tmp_path: Path, clock: _TestClock):
    source = tmp_path / "source"
    _init(source, clock)
    freeze = json.loads(next(source.glob("000000_*.json")).read_bytes())

    cases = [
        (
            "timezone",
            FREEZE_NOW,
            {**freeze["payload"], "exchange_timezone": "UTC"},
            "exchange_timezone must be frozen at Asia/Shanghai",
        ),
        (
            "execution-time",
            FREEZE_NOW,
            {**freeze["payload"], "execution_time_local": "09:31:00"},
            "execution_time_local must be frozen at 09:30:00",
        ),
        (
            "late-genesis",
            datetime(2026, 7, 19, 16, tzinfo=UTC),
            freeze["payload"],
            "before oos_start_date begins",
        ),
    ]
    for name, recorded_at, payload, match in cases:
        root = tmp_path / name
        root.mkdir()
        record, raw = ledger._build_record(0, "freeze", recorded_at, ledger.ZERO_HASH, payload)
        (root / f"000000_{record['record_hash']}.json").write_bytes(raw)
        with pytest.raises(ledger.LedgerValidationError, match=match):
            ledger.verify_ledger(root)


def test_decision_execution_two_phase_and_exact_byte_idempotency(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    decision_data = _decision_data(decision_dt)
    execution_data = _execution_data(exec_dt, decision_data["gate_identity_sha256"])

    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    assert ledger.verify_ledger(root)["pending_decision_hash"] == _record_hash(decision)
    clock.set(_execution_now(exec_dt))
    execution = ledger.append_execution(
        root,
        decision_hash=_record_hash(decision),
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        data=execution_data,
    )
    assert ledger.verify_ledger(root)["completed_weeks"] == 1

    clock.set(_decision_now(decision_dt))
    assert ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data) == decision
    clock.set(_execution_now(exec_dt))
    assert (
        ledger.append_execution(
            root,
            decision_hash=_record_hash(decision),
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=execution_data,
        )
        == execution
    )
    changed = copy.deepcopy(decision_data)
    changed["plan_sha256"] = _hash("changed-plan")
    clock.set(_decision_now(decision_dt))
    with pytest.raises(ledger.LedgerConflictError, match="bytes differ"):
        ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=changed)


def test_timing_is_checked_on_both_sides_of_execution_session(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "late-decision"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    decision_data = _decision_data(decision_dt)

    clock.set(_decision_close(decision_dt) - timedelta(microseconds=1))
    with pytest.raises(ledger.LedgerTimingError, match="before the decision session close"):
        ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    clock.set(_execution_open(exec_dt))
    with pytest.raises(ledger.LedgerTimingError, match="before the execution session"):
        ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)

    root = tmp_path / "early-execution"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    decision_data = _decision_data(decision_dt)
    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    clock.set(_execution_open(exec_dt) - timedelta(microseconds=1))
    with pytest.raises(ledger.LedgerTimingError, match="cannot be recorded before"):
        ledger.append_execution(
            root,
            decision_hash=_record_hash(decision),
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=_execution_data(exec_dt, decision_data["gate_identity_sha256"]),
        )

    clock.set(_execution_close(exec_dt))
    with pytest.raises(ledger.LedgerTimingError, match="before the execution session close"):
        ledger.append_execution(
            root,
            decision_hash=_record_hash(decision),
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=_execution_data(exec_dt, decision_data["gate_identity_sha256"]),
        )


def test_snapshot_cutoffs_are_bounded_by_market_event_and_record_time(tmp_path: Path, clock: _TestClock):
    sessions = _calendar()
    decision_dt, exec_dt = _week_pair(sessions, 0)
    for name, cutoff, match in (
        ("before-close", _decision_close(decision_dt) - timedelta(seconds=1), "precedes"),
        ("after-record", _decision_now(decision_dt) + timedelta(seconds=1), "exceeds"),
    ):
        root = tmp_path / name
        _init(root, clock, sessions=sessions)
        data = _decision_data(decision_dt)
        data["asof_cutoff_utc"] = _utc_text(cutoff)
        clock.set(_decision_now(decision_dt))
        with pytest.raises(ledger.LedgerTimingError, match=match):
            ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=data)

    for name, cutoff, match in (
        ("before-open", _execution_open(exec_dt) - timedelta(seconds=1), "precedes"),
        ("execution-after-record", _execution_now(exec_dt) + timedelta(seconds=1), "exceeds"),
    ):
        root = tmp_path / name
        _init(root, clock, sessions=sessions)
        decision_data = _decision_data(decision_dt, tag=name)
        clock.set(_decision_now(decision_dt))
        decision = ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
        execution_data = _execution_data(exec_dt, decision_data["gate_identity_sha256"], tag=name)
        execution_data["snapshot_cutoff_utc"] = _utc_text(cutoff)
        clock.set(_execution_now(exec_dt))
        with pytest.raises(ledger.LedgerTimingError, match=match):
            ledger.append_execution(
                root,
                decision_hash=_record_hash(decision),
                decision_dt=decision_dt,
                exec_dt=exec_dt,
                data=execution_data,
            )


def test_recorded_timestamps_cannot_be_backdated_before_chain_head(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    clock.set(FREEZE_NOW - timedelta(seconds=1))
    with pytest.raises(ledger.LedgerValidationError, match="strictly increasing"):
        ledger.append_decision(
            root,
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=_decision_data(decision_dt),
        )


def test_next_session_reference_and_state_machine_are_fail_closed(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    clock.set(_decision_now(decision_dt))
    with pytest.raises(ledger.LedgerValidationError, match="next exchange session"):
        ledger.append_decision(
            root,
            decision_dt=decision_dt,
            exec_dt=sessions[6],
            data=_decision_data(decision_dt),
        )

    decision_data = _decision_data(decision_dt)
    ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    next_decision, next_exec = _week_pair(sessions, 1)
    clock.set(_decision_now(next_decision))
    with pytest.raises(ledger.LedgerValidationError, match="execution is pending"):
        ledger.append_decision(
            root,
            decision_dt=next_decision,
            exec_dt=next_exec,
            data=_decision_data(next_decision),
        )
    clock.set(_execution_now(exec_dt))
    with pytest.raises(ledger.LedgerValidationError, match="does not reference"):
        ledger.append_execution(
            root,
            decision_hash="c" * 64,
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=_execution_data(exec_dt, decision_data["gate_identity_sha256"]),
        )


def test_frozen_week_sequence_rejects_first_and_later_skips(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "first-skip"
    sessions = _init(root, clock)
    skipped_decision, skipped_exec = _week_pair(sessions, 1)
    clock.set(_decision_now(skipped_decision))
    with pytest.raises(ledger.LedgerValidationError, match="next complete frozen-calendar week"):
        ledger.append_decision(
            root,
            decision_dt=skipped_decision,
            exec_dt=skipped_exec,
            data=_decision_data(skipped_decision),
        )

    root = tmp_path / "later-skip"
    sessions = _init(root, clock)
    _append_complete_week(root, sessions, 0, clock)
    skipped_decision, skipped_exec = _week_pair(sessions, 2)
    clock.set(_decision_now(skipped_decision))
    with pytest.raises(ledger.LedgerValidationError, match="next complete frozen-calendar week"):
        ledger.append_decision(
            root,
            decision_dt=skipped_decision,
            exec_dt=skipped_exec,
            data=_decision_data(skipped_decision),
        )


def test_partial_start_and_holiday_weeks_follow_frozen_calendar(tmp_path: Path, clock: _TestClock):
    sessions = _calendar()
    root = tmp_path / "partial-start"
    _init(root, clock, sessions=sessions, start_index=2)
    partial_decision, partial_exec = _week_pair(sessions, 0)
    clock.set(_decision_now(partial_decision))
    with pytest.raises(ledger.LedgerValidationError, match="next complete frozen-calendar week"):
        ledger.append_decision(
            root,
            decision_dt=partial_decision,
            exec_dt=partial_exec,
            data=_decision_data(partial_decision),
        )

    first_decision, first_exec = _week_pair(sessions, 1)
    clock.set(_decision_now(first_decision))
    ledger.append_decision(
        root,
        decision_dt=first_decision,
        exec_dt=first_exec,
        data=_decision_data(first_decision),
    )

    holiday_sessions = _calendar()
    holiday_sessions.remove(date(2026, 7, 24))
    root = tmp_path / "holiday-week"
    _init(root, clock, sessions=holiday_sessions)
    decision_dt, exec_dt = date(2026, 7, 23), date(2026, 7, 27)
    decision_data = _decision_data(decision_dt, tag="holiday")
    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    clock.set(_execution_now(exec_dt))
    ledger.append_execution(
        root,
        decision_hash=_record_hash(decision),
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        data=_execution_data(exec_dt, decision_data["gate_identity_sha256"], tag="holiday"),
    )
    assert ledger.verify_ledger(root)["completed_weeks"] == 1


def test_strictly_increasing_decisions_reject_backfill(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _init(root, clock)
    _, latest_execution = _append_complete_week(root, sessions, 0, clock)
    older_decision, older_exec = sessions[3], sessions[4]
    chain_head = datetime.fromisoformat(
        str(json.loads(latest_execution.read_bytes())["recorded_at_utc"]).replace("Z", "+00:00")
    )
    clock.set(chain_head + timedelta(seconds=1))
    with pytest.raises(ledger.LedgerValidationError, match="strictly later"):
        ledger.append_decision(
            root,
            decision_dt=older_decision,
            exec_dt=older_exec,
            data=_decision_data(older_decision),
        )


def test_verifier_detects_tampering_missing_sequence_and_fork(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "tamper"
    sessions = _init(root, clock)
    decision, _ = _append_complete_week(root, sessions, 0, clock)
    value = json.loads(decision.read_bytes())
    value["payload"]["data"]["plan_sha256"] = _hash("tampered")
    decision.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ledger.LedgerValidationError, match="record hash mismatch"):
        ledger.verify_ledger(root)

    root = tmp_path / "missing"
    sessions = _init(root, clock)
    decision, _ = _append_complete_week(root, sessions, 0, clock)
    decision.unlink()
    with pytest.raises(ledger.LedgerValidationError, match="missing ledger sequence"):
        ledger.verify_ledger(root)

    root = tmp_path / "fork"
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(
        root,
        decision_dt=decision_dt,
        exec_dt=exec_dt,
        data=_decision_data(decision_dt),
    )
    shutil.copyfile(decision, root / f"000001_{'f' * 64}.json")
    with pytest.raises(ledger.LedgerValidationError, match="fork detected"):
        ledger.verify_ledger(root)


def _mutate_empty(_: dict[str, Any]) -> dict[str, Any]:
    return {}


def _mutate_seed(data: dict[str, Any]) -> dict[str, Any]:
    data["seed_ids"] = data["seed_ids"][:-1]
    return data


def _mutate_hash(data: dict[str, Any]) -> dict[str, Any]:
    data["plan_sha256"] = "ABC"
    return data


def _mutate_invariant(data: dict[str, Any]) -> dict[str, Any]:
    data["invariants"]["blocked_slots_applied"] = "true"
    return data


def _mutate_arm_map(data: dict[str, Any]) -> dict[str, Any]:
    data["arm_identity_sha256"]["FGR"].pop(str(ledger.EXPECTED_SEED_IDS[-1]))
    return data


def _mutate_extra(data: dict[str, Any]) -> dict[str, Any]:
    data["unexpected"] = True
    return data


def _mutate_schema(data: dict[str, Any]) -> dict[str, Any]:
    data["schema"] = "wrong"
    return data


def _mutate_engine(data: dict[str, Any]) -> dict[str, Any]:
    data["engine_sha256"] = _hash("unfrozen-engine")
    return data


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (_mutate_empty, "fields mismatch"),
        (_mutate_seed, "seed_ids"),
        (_mutate_hash, "lowercase SHA256"),
        (_mutate_invariant, "literal true"),
        (_mutate_arm_map, "fields mismatch"),
        (_mutate_extra, "fields mismatch"),
        (_mutate_schema, "schema must be"),
        (_mutate_engine, "frozen research engine"),
    ],
)
def test_decision_exact_schema_rejects_malformed_evidence(
    tmp_path: Path,
    clock: _TestClock,
    mutator: Any,
    match: str,
):
    root = tmp_path / mutator.__name__
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    data = mutator(copy.deepcopy(_decision_data(decision_dt)))
    clock.set(_decision_now(decision_dt))
    with pytest.raises(ledger.LedgerValidationError, match=match):
        ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=data)
    assert ledger.verify_ledger(root)["record_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("empty", "fields mismatch"),
        ("bad_hash", "lowercase SHA256"),
        ("string_bool", "literal true"),
        ("wrong_gate", "does not match"),
        ("extra", "fields mismatch"),
        ("wrong_engine", "frozen execution engine"),
    ],
)
def test_execution_exact_schema_rejects_malformed_evidence(
    tmp_path: Path,
    clock: _TestClock,
    mutation: str,
    match: str,
):
    root = tmp_path / mutation
    sessions = _init(root, clock)
    decision_dt, exec_dt = _week_pair(sessions, 0)
    decision_data = _decision_data(decision_dt, tag=mutation)
    clock.set(_decision_now(decision_dt))
    decision = ledger.append_decision(root, decision_dt=decision_dt, exec_dt=exec_dt, data=decision_data)
    data = _execution_data(exec_dt, decision_data["gate_identity_sha256"], tag=mutation)
    if mutation == "empty":
        data = {}
    elif mutation == "bad_hash":
        data["fills_sha256"] = "not-a-hash"
    elif mutation == "string_bool":
        data["invariants"]["no_negative_cash"] = "true"
    elif mutation == "wrong_engine":
        data["engine_sha256"] = _hash("unfrozen-execution-engine")
    else:
        if mutation == "wrong_gate":
            data["gate_identity_sha256"] = _hash("wrong-gate")
        else:
            data["unexpected"] = True
    clock.set(_execution_now(exec_dt))
    with pytest.raises(ledger.LedgerValidationError, match=match):
        ledger.append_execution(
            root,
            decision_hash=_record_hash(decision),
            decision_dt=decision_dt,
            exec_dt=exec_dt,
            data=data,
        )
    verified = ledger.verify_ledger(root)
    assert verified["completed_weeks"] == 0
    assert verified["pending_decision_hash"] == _record_hash(decision)


def test_public_api_and_cli_do_not_expose_a_historical_clock_override():
    for function in (ledger.init_ledger, ledger.append_decision, ledger.append_execution, ledger.main):
        assert "now" not in inspect.signature(function).parameters
    help_text = ledger.build_parser().format_help()
    assert "recorded-at" not in help_text


def test_52_hash_only_weeks_are_structurally_complete_but_never_confirmatory(tmp_path: Path, clock: _TestClock):
    root = tmp_path / "ledger"
    sessions = _init(root, clock, weeks=53)
    for week in range(51):
        _append_complete_week(root, sessions, week, clock)
    before = ledger.verify_ledger(root)
    assert before["completed_weeks"] == 51
    assert before["confirmatory_oos"] is False
    assert before["structural_window_complete"] is False
    assert before["confirmation_window"] is None

    _append_complete_week(root, sessions, 51, clock)
    after = ledger.verify_ledger(root)
    assert after["completed_weeks"] == 52
    assert after["min_complete_weeks"] == 52
    assert after["structural_window_complete"] is True
    assert after["semantic_replay_verified"] is False
    assert after["confirmatory_oos"] is False
    window = after["confirmation_window"]
    assert window["schema"] == "xs_chan_v2_confirmation_window_identity_v1"
    assert len(window["record_hashes"]) == 104
    assert len(window["decision_dates"]) == len(window["execution_dates"]) == 52
    assert window["confirmation_head_sha256"] == window["record_hashes"][-1]
    assert len(window["identity_sha256"]) == 64

    frozen_identity = window["identity_sha256"]
    _append_complete_week(root, sessions, 52, clock)
    extended = ledger.verify_ledger(root)
    assert extended["completed_weeks"] == 53
    assert extended["confirmation_window"]["identity_sha256"] == frozen_identity


def test_cli_supports_init_append_and_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clock: _TestClock,
):
    root = tmp_path / "ledger"
    protocol = tmp_path / "protocol.json"
    calendar_path = tmp_path / "calendar.json"
    decision_payload = tmp_path / "decision.json"
    execution_payload = tmp_path / "execution.json"
    sessions = _calendar()
    decision_dt, exec_dt = _week_pair(sessions, 0)
    protocol.write_text('{"protocol_id":"v2"}\n', encoding="utf-8")
    dependency_paths = {}
    for name in sorted(ledger.EXPECTED_DEPENDENCY_KEYS):
        path = tmp_path / f"{name}.bin"
        path.write_text(f"frozen {name}\n", encoding="utf-8")
        dependency_paths[name] = path
    decision_data = _decision_data(decision_dt, tag="cli")
    decision_data["engine_sha256"] = ledger.sha256_file(dependency_paths["research_engine"])
    execution_data = _execution_data(exec_dt, decision_data["gate_identity_sha256"], tag="cli")
    execution_data["engine_sha256"] = ledger.sha256_file(dependency_paths["execution_engine"])
    calendar_path.write_text(json.dumps([item.isoformat() for item in sessions]), encoding="utf-8")
    decision_payload.write_text(json.dumps(decision_data), encoding="utf-8")
    execution_payload.write_text(json.dumps(execution_data), encoding="utf-8")

    clock.set(FREEZE_NOW)
    assert (
        ledger.main(
            [
                "init",
                str(root),
                "--protocol",
                str(protocol),
                *[
                    item
                    for name, path in sorted(dependency_paths.items())
                    for item in ("--dependency", f"{name}={path}")
                ],
                "--calendar",
                str(calendar_path),
                "--oos-start-date",
                sessions[0].isoformat(),
            ]
        )
        == 0
    )
    capsys.readouterr()
    clock.set(_decision_now(decision_dt))
    assert (
        ledger.main(
            [
                "append",
                "decision",
                str(root),
                "--decision-dt",
                decision_dt.isoformat(),
                "--exec-dt",
                exec_dt.isoformat(),
                "--payload",
                str(decision_payload),
            ]
        )
        == 0
    )
    decision_output = json.loads(capsys.readouterr().out)
    clock.set(_execution_now(exec_dt))
    assert (
        ledger.main(
            [
                "append",
                "execution",
                str(root),
                "--decision-hash",
                decision_output["record_hash"],
                "--decision-dt",
                decision_dt.isoformat(),
                "--exec-dt",
                exec_dt.isoformat(),
                "--payload",
                str(execution_payload),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert ledger.main(["verify", str(root)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True
    assert verified["completed_weeks"] == 1
