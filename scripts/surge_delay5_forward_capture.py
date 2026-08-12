"""Append-only, content-addressed capture chain for the Delay5 forward trial.

The tracked 2026-08-12 forward protocol is immutable, so this outcome-blind
supplement implements its per-session capture contract without changing the
original script bytes.  Every observed post-freeze session stores:

* the causal feature surface through that close;
* one complete ``daily_basic`` cross-section;
* the production cohort stage projection; and
* the causal market-state row.

Each ledger head is exported as a unique tracked anchor.  A session is eligible
for the primary forward sample only when that anchor was committed and pushed
before the next official session opened.  Missing, late, dirty, or unpushed
anchors can never unlock outcomes.

Commands::

    uv run --no-sync python scripts/surge_delay5_forward_capture.py --write-protocol
    uv run --no-sync python scripts/surge_delay5_forward_capture.py init
    uv run --with tinyshare==0.1028.0 python scripts/surge_delay5_forward_capture.py append
    uv run --no-sync python scripts/surge_delay5_forward_capture.py status
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import delay5_pit_exact_mcap_balance_audit as balance
import delay5_pit_exact_mcap_collector as collector
import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as industry_source
import surge_delay5_forward_audit as forward
import surge_portfolio_backtest as portfolio

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "surge_delay5_forward_capture_protocol_2026-08-12.json"
MAIN_PROTOCOL_PATH = forward.PROTOCOL_PATH
OUTPUT_ROOT = SCRIPT_DIR / "_output" / "surge_delay5_forward_capture"
ANCHOR_DIR = SCRIPT_DIR / "surge_delay5_forward_capture_ledger_anchors"
PANEL_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
COHORT_PATH = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort" / "cohort.parquet"
PRODUCTION_AUDIT_PATH = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort" / "audit.json"
CANDIDATE_MANIFEST_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "manifest.json"
STATUS_PATH = OUTPUT_ROOT / "status.json"

PROTOCOL_SCHEMA = "surge_delay5_forward_capture_protocol_v1"
LEDGER_SCHEMA = "surge_delay5_forward_capture_ledger_v1"
RECORD_SCHEMA = "surge_delay5_forward_capture_record_v1"
STATUS_SCHEMA = "surge_delay5_forward_capture_status_v1"
SURFACE_SCHEMA = "surge_delay5_forward_causal_surface_v1"
SNAPSHOT_SCHEMA = "surge_delay5_forward_session_snapshot_v1"
ANCHOR_SCHEMA = "surge_delay5_forward_capture_head_anchor_v1"
ZERO_HASH = "0" * 64
EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")
FEATURES = (
    "log_exact_mcap",
    "ret5",
    "ret20",
    "ret60",
    "vol20",
    "vol60",
    "log_price",
    "log_amount",
    "liq20",
)
SURFACE_FIELDS = (
    "symbol",
    "decision_close",
    "decision_amount_e",
    "industry",
    "is_st",
    *FEATURES,
)
PANEL_FIELDS = ("symbol", "dt", "open", "close", "amount_e")
RECORD_KEYS = {
    "schema",
    "ledger_id",
    "sequence",
    "previous_hash",
    "record_type",
    "session",
    "capture_classification",
    "captured_at_utc",
    "payload",
    "payload_sha256",
    "record_hash",
}


@dataclass(frozen=True)
class CaptureRecord:
    path: Path
    data: dict[str, Any]
    raw: bytes


class CaptureError(RuntimeError):
    """Raised when the append-only forward capture contract cannot close."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CaptureError(f"path is outside repository: {path}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must be a JSON object")
    return value


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        if path.read_bytes() == raw:
            return
        raise CaptureError(f"immutable artifact already exists with different bytes: {path}") from exc
    with os.fdopen(descriptor, "wb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())


def _store_object(root: Path, kind: str, value: Any) -> dict[str, Any]:
    raw = canonical_json(value)
    digest = sha256_bytes(raw)
    path = root / "objects" / kind / f"{digest}.json"
    _exclusive_bytes(path, raw)
    return {
        "object": path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": len(raw),
    }


def _read_object(root: Path, record: Mapping[str, Any]) -> Any:
    relative = Path(str(record.get("object", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise CaptureError("object locator is not portable")
    path = root / relative
    expected = str(record.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected:
        raise CaptureError(f"capture object identity drift: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json(value) != path.read_bytes():
        raise CaptureError(f"capture object is not canonical: {relative}")
    return value


def build_protocol() -> dict[str, Any]:
    """Freeze the capture supplement before any post-freeze session exists."""

    main = _load_json(MAIN_PROTOCOL_PATH, "main forward protocol")
    forward.validate_protocol(main)
    status = forward.run_status()["progress"]
    if status["post_freeze_sessions_observed"] != 0 or status["enrolled_identity"]["rows"] != 0:
        raise CaptureError("capture supplement must be frozen before the first post-freeze session")
    return {
        "schema": PROTOCOL_SCHEMA,
        "frozen_at_utc": "2026-08-12T08:25:35Z",
        "main_forward_protocol": {
            "locator": _repo_locator(MAIN_PROTOCOL_PATH),
            "sha256": sha256_file(MAIN_PROTOCOL_PATH),
        },
        "capture_script": {
            "locator": _repo_locator(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "decision_cutoff_inclusive": forward.DECISION_CUTOFF.strftime("%Y-%m-%d"),
        "capture_scope": "every observed post-freeze official session through completion of all enrolled H60 paths",
        "objects": {
            "every_observed_session": {
                "market_panel_projection": list(PANEL_FIELDS),
                "production_projection": list(forward.FORWARD_COHORT_COLUMNS),
                "market_projection": ["dt", "high20_ratio", "ew_index_above_ma20"],
            },
            "first_60_accrual_sessions_only": {
                "causal_surface": list(SURFACE_FIELDS),
                "daily_basic": list(collector.DAILY_BASIC_FIELDS),
            },
        },
        "timeliness": {
            "capture_requirement": "session is the latest observed market session when appended",
            "remote_anchor_proxy": (
                "the unique ledger-head anchor must be clean, committed, reachable from the configured live "
                "remote branch, and have a commit timestamp before the next official session opens at 09:30 "
                "Asia/Shanghai; this is an operational remote anchor, not an independent timestamp authority"
            ),
            "late_or_unanchored_action": "permanently exclude that decision session from the primary forward sample",
        },
        "chain": {
            "ledger_schema": LEDGER_SCHEMA,
            "record_schema": RECORD_SCHEMA,
            "zero_hash": ZERO_HASH,
            "objects_content_addressed": True,
            "records_append_only": True,
        },
        "outcome_lock": {
            "combined_unlock": (
                "main forward protocol is ready AND every required session through the final enrolled H60 endpoint "
                "has a verified timely remote anchor AND every required session object verifies"
            ),
            "outcomes_loaded": False,
            "live_authorized": False,
        },
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    main = _load_json(MAIN_PROTOCOL_PATH, "main forward protocol")
    forward.validate_protocol(main)
    required = (
        protocol.get("schema") == PROTOCOL_SCHEMA,
        protocol.get("main_forward_protocol", {}).get("sha256") == sha256_file(MAIN_PROTOCOL_PATH),
        protocol.get("capture_script", {}).get("sha256") == sha256_file(Path(__file__)),
        protocol.get("decision_cutoff_inclusive") == forward.DECISION_CUTOFF.strftime("%Y-%m-%d"),
        protocol.get("chain", {}).get("zero_hash") == ZERO_HASH,
        protocol.get("outcome_lock", {}).get("outcomes_loaded") is False,
        protocol.get("outcome_lock", {}).get("live_authorized") is False,
    )
    if not all(required):
        raise CaptureError("forward capture protocol is incomplete or drifted")


def ledger_identity(protocol: Mapping[str, Any]) -> str:
    material = b"surge-delay5-forward-capture-ledger-v1\0" + canonical_json(protocol)
    return sha256_bytes(material)


def ledger_root(protocol: Mapping[str, Any]) -> Path:
    return OUTPUT_ROOT / f"LEDGER_{ledger_identity(protocol)}"


def _record_hash(body: Mapping[str, Any]) -> str:
    return sha256_bytes(b"surge-delay5-forward-capture-record-v1\0" + canonical_json(body))


def scan_records(root: Path) -> tuple[CaptureRecord, ...]:
    records_dir = root / "records"
    if not records_dir.exists():
        return ()
    paths = sorted(path for path in records_dir.iterdir() if path.is_file())
    result: list[CaptureRecord] = []
    previous = ZERO_HASH
    identity: str | None = None
    sessions: set[str] = set()
    for expected_sequence, path in enumerate(paths):
        raw = path.read_bytes()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaptureError(f"capture record schema/canonicalization drift: {path.name}") from exc
        if not isinstance(value, dict) or set(value) != RECORD_KEYS or canonical_json(value) != raw:
            raise CaptureError(f"capture record schema/canonicalization drift: {path.name}")
        if value["schema"] != RECORD_SCHEMA or value["sequence"] != expected_sequence:
            raise CaptureError(f"capture record sequence drift: {path.name}")
        if value["previous_hash"] != previous:
            raise CaptureError(f"capture record previous hash drift: {path.name}")
        if identity is None:
            identity = str(value["ledger_id"])
        elif value["ledger_id"] != identity:
            raise CaptureError(f"capture ledger identity changed: {path.name}")
        body = {key: child for key, child in value.items() if key != "record_hash"}
        if value["record_hash"] != _record_hash(body):
            raise CaptureError(f"capture record hash drift: {path.name}")
        if value["payload_sha256"] != sha256_bytes(canonical_json(value["payload"])):
            raise CaptureError(f"capture payload hash drift: {path.name}")
        expected_name = f"{expected_sequence:06d}_{value['record_hash']}.json"
        if path.name != expected_name:
            raise CaptureError(f"capture record filename drift: {path.name}")
        session = value["session"]
        if session is not None:
            if session in sessions:
                raise CaptureError(f"duplicate captured session: {session}")
            sessions.add(str(session))
        previous = str(value["record_hash"])
        result.append(CaptureRecord(path, value, raw))
    return tuple(result)


def validate_ledger(records: Sequence[CaptureRecord], protocol: Mapping[str, Any]) -> None:
    """Bind the append-only chain to its protocol and unique genesis."""

    if not records:
        raise CaptureError("capture ledger is empty")
    expected_identity = ledger_identity(protocol)
    if any(record.data["ledger_id"] != expected_identity for record in records):
        raise CaptureError("capture ledger is not bound to the active protocol")
    genesis = records[0]
    if (
        genesis.data["record_type"] != "genesis"
        or genesis.data["session"] is not None
        or genesis.data["capture_classification"] != "OUTCOME_BLIND_GENESIS"
        or genesis.data["payload"].get("schema") != LEDGER_SCHEMA
        or genesis.data["payload"].get("capture_protocol_sha256") != sha256_file(PROTOCOL_PATH)
    ):
        raise CaptureError("capture ledger genesis contract drift")
    if any(record.data["record_type"] == "genesis" for record in records[1:]):
        raise CaptureError("capture ledger contains a second genesis")
    if any(record.data["record_type"] not in {"genesis", "session_capture"} for record in records):
        raise CaptureError("capture ledger contains an unknown record type")


def _append_record(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    record_type: str,
    session: str | None,
    classification: str,
    payload: Mapping[str, Any],
    captured_at_utc: str,
) -> CaptureRecord:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".append.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = scan_records(root)
        if session is not None:
            existing = [record for record in records if record.data["session"] == session]
            if existing:
                expected_payload = sha256_bytes(canonical_json(payload))
                record = existing[0]
                if (
                    record.data["record_type"] == record_type
                    and record.data["capture_classification"] == classification
                    and record.data["payload_sha256"] == expected_payload
                ):
                    return record
                raise CaptureError(f"session already captured with different immutable content: {session}")
        sequence = len(records)
        previous = records[-1].data["record_hash"] if records else ZERO_HASH
        body = {
            "schema": RECORD_SCHEMA,
            "ledger_id": ledger_identity(protocol),
            "sequence": sequence,
            "previous_hash": previous,
            "record_type": record_type,
            "session": session,
            "capture_classification": classification,
            "captured_at_utc": captured_at_utc,
            "payload": dict(payload),
            "payload_sha256": sha256_bytes(canonical_json(payload)),
        }
        value = {**body, "record_hash": _record_hash(body)}
        raw = canonical_json(value)
        path = root / "records" / f"{sequence:06d}_{value['record_hash']}.json"
        _exclusive_bytes(path, raw)
        return CaptureRecord(path, value, raw)


def _export_head_anchor(protocol: Mapping[str, Any], record: CaptureRecord) -> Path:
    payload = {
        "schema": ANCHOR_SCHEMA,
        "ledger_id": ledger_identity(protocol),
        "sequence": record.data["sequence"],
        "record_type": record.data["record_type"],
        "session": record.data["session"],
        "record_hash": record.data["record_hash"],
        "previous_hash": record.data["previous_hash"],
        "captured_at_utc": record.data["captured_at_utc"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "capture_script_sha256": sha256_file(Path(__file__)),
        "required_follow_up": "commit_and_push_this_unique_anchor_before_the_next_official_session_open",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path = ANCHOR_DIR / f"{record.data['sequence']:06d}_{record.data['record_hash']}.json"
    _exclusive_bytes(path, raw)
    return path


def initialize(protocol: Mapping[str, Any], *, captured_at_utc: str | None = None) -> CaptureRecord:
    validate_protocol(protocol)
    root = ledger_root(protocol)
    records = scan_records(root)
    if records:
        validate_ledger(records, protocol)
        return records[0]
    timestamp = captured_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "schema": LEDGER_SCHEMA,
        "main_forward_protocol_sha256": sha256_file(MAIN_PROTOCOL_PATH),
        "capture_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "decision_cutoff_inclusive": forward.DECISION_CUTOFF.strftime("%Y-%m-%d"),
        "post_freeze_sessions_at_genesis": 0,
        "outcomes_loaded": False,
        "live_authorized": False,
    }
    record = _append_record(
        root,
        protocol,
        record_type="genesis",
        session=None,
        classification="OUTCOME_BLIND_GENESIS",
        payload=payload,
        captured_at_utc=timestamp,
    )
    _export_head_anchor(protocol, record)
    return record


def _date_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _finite_or_none(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if isinstance(value, (int, np.integer)) else number


def _cohort_projection(cohort: pd.DataFrame, session: pd.Timestamp) -> dict[str, Any]:
    scoped = cohort[pd.to_datetime(cohort["dec_dt"]).dt.normalize().eq(session)].copy()
    records: list[list[Any]] = []
    for row in scoped.sort_values(["symbol", "sig_dt"], kind="mergesort").itertuples(index=False):
        records.append(
            [
                str(row.symbol),
                _date_value(row.sig_dt),
                _date_value(row.dec_dt),
                _date_value(row.entry_dt),
                _date_value(row.entry_fill_dt),
                _date_value(row.common_h60_dt),
                *[
                    bool(getattr(row, name))
                    for name in (
                        "stage_raw",
                        "stage_gate",
                        "stage_hard",
                        "stage_st",
                        "stage_market",
                        "stage_fill",
                        "stage_mature",
                    )
                ],
            ]
        )
    return {
        "schema": "surge_delay5_forward_cohort_projection_v1",
        "session": session.strftime("%Y-%m-%d"),
        "columns": list(forward.FORWARD_COHORT_COLUMNS),
        "records": records,
    }


def build_panel_projection(panel: pd.DataFrame, session: pd.Timestamp) -> dict[str, Any]:
    """Freeze the open/close path inputs observed on one official session."""

    scoped = panel[pd.to_datetime(panel["dt"]).dt.normalize().eq(session)].copy()
    records = [
        [
            str(row.symbol),
            _date_value(row.dt),
            _finite_or_none(row.open),
            _finite_or_none(row.close),
            _finite_or_none(row.amount_e),
        ]
        for row in scoped.sort_values("symbol", kind="mergesort").itertuples(index=False)
    ]
    return {
        "schema": "surge_delay5_forward_market_panel_projection_v1",
        "session": session.strftime("%Y-%m-%d"),
        "columns": list(PANEL_FIELDS),
        "records": records,
    }


def capture_classification(session: pd.Timestamp, latest: pd.Timestamp) -> str:
    """Only the latest observed session can enter the primary forward chain."""

    return "CAPTURED_LATEST_OBSERVED_SESSION" if session == latest else "LATE_EXCLUDED"


def build_causal_surface(
    session: pd.Timestamp,
    *,
    feature_store: balance.CausalFeatureStore,
    exact_object: Mapping[str, Any],
    industry: pd.Series,
    st_intervals: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact decision-close surface needed for transported matching."""

    snapshot = feature_store.snapshot(session).copy()
    exact_rows = {str(row["ts_code"]): row for row in exact_object["rows"]}
    records: list[list[Any]] = []
    for symbol, row in snapshot.sort_index().iterrows():
        exact = exact_rows.get(str(symbol), {})
        circ_mv = _finite_or_none(exact.get("circ_mv"))
        exact_cny = float(circ_mv) * 10_000 if circ_mv is not None and circ_mv > 0 else None
        values = {
            "symbol": str(symbol),
            "decision_close": _finite_or_none(feature_store.raw_close.at[session, symbol]),
            "decision_amount_e": _finite_or_none(feature_store.amount.at[session, symbol]),
            "industry": None if pd.isna(industry.get(symbol, np.nan)) else str(industry.get(symbol)),
            "is_st": bool(portfolio.is_st_on(st_intervals, str(symbol), session)),
            "log_exact_mcap": math.log(exact_cny) if exact_cny is not None and exact_cny > 0 else None,
            **{feature: _finite_or_none(row.get(feature)) for feature in FEATURES if feature != "log_exact_mcap"},
        }
        records.append([values[field] for field in SURFACE_FIELDS])
    return {
        "schema": SURFACE_SCHEMA,
        "session": session.strftime("%Y-%m-%d"),
        "columns": list(SURFACE_FIELDS),
        "records": records,
    }


def _fetch_exact_object(
    pro: Any,
    session: pd.Timestamp,
    requested_symbols: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    content, diagnostics, attempts = collector._fetch_one_date(
        pro,
        trade_date=session.strftime("%Y-%m-%d"),
        requested_symbols=frozenset(requested_symbols),
        max_attempts=collector.DEFAULT_MAX_ATTEMPTS,
        retry_sleep=lambda _seconds: None,
        retry_delay=0,
    )
    return content, {**diagnostics, "attempts": attempts}


def capture_sessions(
    protocol: Mapping[str, Any],
    *,
    pro: Any | None = None,
    captured_at_utc: str | None = None,
) -> tuple[CaptureRecord, ...]:
    """Append every uncaptured observed session; older missed sessions are permanently late."""

    validate_protocol(protocol)
    root = ledger_root(protocol)
    initialize(protocol, captured_at_utc=captured_at_utc)
    candidate_manifest = _load_json(CANDIDATE_MANIFEST_PATH, "candidate manifest")
    production_audit = _load_json(PRODUCTION_AUDIT_PATH, "production audit")
    if candidate_manifest.get("raw_completeness_proven") is not True:
        raise CaptureError("candidate manifest does not prove raw completeness")
    if production_audit.get("raw_completeness_proven") is not True:
        raise CaptureError("production audit does not prove raw completeness")
    if production_audit.get("design", {}).get("outcome_fields_used_for_cohort") is not False:
        raise CaptureError("production cohort is not outcome blind")

    market = pd.read_parquet(MARKET_PATH)
    market["dt"] = pd.to_datetime(market["dt"]).dt.normalize()
    calendar = balance.normalise_calendar(market["dt"])
    post_sessions = calendar[calendar > forward.DECISION_CUTOFF]
    existing = {str(record.data["session"]) for record in scan_records(root) if record.data["session"] is not None}
    missing = [session for session in post_sessions if session.strftime("%Y-%m-%d") not in existing]
    if not missing:
        return scan_records(root)
    accrual = post_sessions[: forward.ACCRUAL_SESSIONS]
    missing_accrual = [session for session in missing if session in accrual]
    panel = pd.read_parquet(PANEL_PATH, columns=list(PANEL_FIELDS))
    st_intervals: Mapping[str, Any] = {}
    feature_store: balance.CausalFeatureStore | None = None
    if missing_accrual:
        if pro is None:
            pro = collector.create_env_client()
        st_intervals = portfolio.load_st_intervals()
        if not st_intervals:
            raise CaptureError("historical ST intervals are unavailable")
        feature_store = balance.CausalFeatureStore.from_panel(panel, calendar, st_intervals)
    cohort = pd.read_parquet(COHORT_PATH, columns=list(forward.FORWARD_COHORT_COLUMNS))
    industry_maps = {
        int(year): pd.read_parquet(industry_source._industry_path(int(year))).set_index("symbol")["industry"]
        for year in sorted({int(session.year) for session in missing_accrual})
    }
    timestamp = captured_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    latest = post_sessions.max()
    for session in missing:
        date_text = session.strftime("%Y-%m-%d")
        classification = capture_classification(session, latest)
        day_cohort = _cohort_projection(cohort, session)
        panel_projection = build_panel_projection(panel, session)
        market_row = market.loc[market["dt"].eq(session), ["dt", "high20_ratio", "ew_index_above_ma20"]].iloc[0]
        market_projection = {
            "schema": "surge_delay5_forward_market_projection_v1",
            "session": date_text,
            "high20_ratio": _finite_or_none(market_row["high20_ratio"]),
            "ew_index_above_ma20": bool(market_row["ew_index_above_ma20"] > 0),
        }
        objects = {
            "market_panel_projection": _store_object(root, "market_panel_projection", panel_projection),
            "production_projection": _store_object(root, "production_projection", day_cohort),
            "market_projection": _store_object(root, "market_projection", market_projection),
        }
        exact_diagnostics: dict[str, Any] | None = None
        surface_rows: int | None = None
        if session in accrual:
            if feature_store is None or pro is None:
                raise CaptureError("accrual capture dependencies are unavailable")
            requested = {
                str(record[0])
                for record in day_cohort["records"]
                if bool(record[10])  # stage_market in the positional projection
            }
            exact_object, exact_diagnostics = _fetch_exact_object(pro, session, requested)
            surface = build_causal_surface(
                session,
                feature_store=feature_store,
                exact_object=exact_object,
                industry=industry_maps[int(session.year)],
                st_intervals=st_intervals,
            )
            objects["causal_surface"] = _store_object(root, "causal_surface", surface)
            objects["daily_basic"] = _store_object(root, "daily_basic", exact_object)
            surface_rows = len(surface["records"])
        snapshot = {
            "schema": SNAPSHOT_SCHEMA,
            "session": date_text,
            "capture_classification": classification,
            "accrual_session": bool(session in accrual),
            "objects": objects,
            "diagnostics": {
                "panel_rows": len(panel_projection["records"]),
                "surface_rows": surface_rows,
                "production_rows": len(day_cohort["records"]),
                "exact": exact_diagnostics,
            },
            "source_identity": {
                "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST_PATH),
                "production_audit_sha256": sha256_file(PRODUCTION_AUDIT_PATH),
                "cohort_sha256": sha256_file(COHORT_PATH),
                "panel_sha256": sha256_file(PANEL_PATH),
                "market_sha256": sha256_file(MARKET_PATH),
            },
            "outcomes_loaded": False,
        }
        snapshot_record = _store_object(root, "session_snapshot", snapshot)
        record = _append_record(
            root,
            protocol,
            record_type="session_capture",
            session=date_text,
            classification=classification,
            payload={"snapshot": snapshot_record, "outcomes_loaded": False},
            captured_at_utc=timestamp,
        )
        _export_head_anchor(protocol, record)
    return scan_records(root)


def verify_session_objects(root: Path, record: CaptureRecord) -> dict[str, Any]:
    if record.data["record_type"] != "session_capture" or record.data["payload"].get("outcomes_loaded") is not False:
        raise CaptureError(f"session record outcome lock drift: {record.data['session']}")
    snapshot = _read_object(root, record.data["payload"]["snapshot"])
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("session") != record.data["session"]:
        raise CaptureError(f"session snapshot identity drift: {record.data['session']}")
    if snapshot.get("outcomes_loaded") is not False:
        raise CaptureError(f"session snapshot outcome lock drift: {record.data['session']}")
    objects = snapshot.get("objects", {})
    required = {"market_panel_projection", "production_projection", "market_projection"}
    if snapshot.get("accrual_session") is True:
        required.update({"causal_surface", "daily_basic"})
    if not isinstance(objects, dict) or set(objects) != required:
        raise CaptureError(f"session snapshot object contract drift: {record.data['session']}")
    for object_record in objects.values():
        _read_object(root, object_record)
    return snapshot


def _git_output(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_pushed_anchor(record: CaptureRecord, next_session: pd.Timestamp | None) -> dict[str, Any]:
    """Prove a unique clean anchor was pushed before the next session opened."""

    path = ANCHOR_DIR / f"{record.data['sequence']:06d}_{record.data['record_hash']}.json"
    result = {
        "available": path.is_file(),
        "tracked": False,
        "clean": False,
        "committed": False,
        "pushed": False,
        "before_next_open": False,
        "commit": None,
        "remote_ref": None,
        "remote_head": None,
        "timestamp_strength": "COMMITTER_TIMESTAMP_NOT_INDEPENDENT_TSA",
    }
    if not path.is_file():
        return result
    relative = _repo_locator(path)
    if _git_output("ls-files", "--error-unmatch", "--", relative) is None:
        return result
    result["tracked"] = True
    status_output = _git_output("status", "--porcelain=v1", "--", relative)
    result["clean"] = status_output == ""
    commit = _git_output("log", "-1", "--format=%H", "--", relative)
    if not commit:
        return result
    result["committed"] = True
    result["commit"] = commit
    committed = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, check=False, capture_output=True)
    if committed.returncode != 0 or committed.stdout != path.read_bytes():
        return result
    branch = _git_output("branch", "--show-current")
    remote = _git_output("config", f"branch.{branch}.remote") if branch else None
    merge_ref = _git_output("config", f"branch.{branch}.merge") if branch else None
    if remote and merge_ref:
        remote_result = subprocess.run(
            ["git", "ls-remote", "--exit-code", remote, merge_ref],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        remote_head = remote_result.stdout.split(maxsplit=1)[0] if remote_result.returncode == 0 else None
        result["remote_ref"] = f"{remote}:{merge_ref}"
        result["remote_head"] = remote_head
    else:
        remote_head = None
    if remote_head:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, remote_head], cwd=REPO_ROOT, check=False
        )
        result["pushed"] = ancestor.returncode == 0
    if next_session is not None:
        committed_at = _git_output("show", "-s", "--format=%cI", commit)
        if committed_at:
            commit_time = datetime.fromisoformat(committed_at)
            next_open = pd.Timestamp(next_session).tz_localize(EXCHANGE_TZ).replace(hour=9, minute=30).to_pydatetime()
            result["before_next_open"] = commit_time < next_open
    return result


def build_status(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Combine the main H60 lock with object and external-anchor closure."""

    validate_protocol(protocol)
    root = ledger_root(protocol)
    records = scan_records(root)
    validate_ledger(records, protocol)
    main = forward.run_status()["progress"]
    market = pd.read_parquet(MARKET_PATH, columns=["dt"])
    sessions = balance.normalise_calendar(market["dt"])
    post_sessions = sessions[sessions > forward.DECISION_CUTOFF]
    accrual = post_sessions[: forward.ACCRUAL_SESSIONS]
    by_session = {
        pd.Timestamp(record.data["session"]): record
        for record in records
        if record.data["record_type"] == "session_capture"
    }
    unexpected = sorted(session.strftime("%Y-%m-%d") for session in set(by_session) - set(post_sessions))
    if unexpected:
        raise CaptureError(f"captured sessions disappeared from the official calendar: {unexpected[:5]}")
    verified = {session: verify_session_objects(root, record) for session, record in by_session.items()}

    if main["outcome_evaluation_permitted"]:
        cohort = pd.read_parquet(COHORT_PATH, columns=list(forward.FORWARD_COHORT_COLUMNS))
        cohort["dec_dt"] = pd.to_datetime(cohort["dec_dt"]).dt.normalize()
        cohort["common_h60_dt"] = pd.to_datetime(cohort["common_h60_dt"]).dt.normalize()
        enrolled = cohort[cohort["dec_dt"].isin(accrual) & cohort["stage_fill"].astype(bool)]
        maturity_dates = enrolled["common_h60_dt"].dropna()
        if maturity_dates.empty:
            raise CaptureError("main outcome lock opened without a fixed H60 endpoint")
        required_end = maturity_dates.max()
        required_sessions = post_sessions[post_sessions <= required_end]
    else:
        required_end = post_sessions.max() if len(post_sessions) else None
        required_sessions = post_sessions

    session_evidence: dict[str, Any] = {}
    primary_sessions: list[str] = []
    positions = {session: index for index, session in enumerate(post_sessions)}
    for session in required_sessions:
        date_text = session.strftime("%Y-%m-%d")
        record = by_session.get(session)
        if record is None:
            session_evidence[date_text] = {"captured": False, "primary_eligible": False}
            continue
        index = positions[session]
        next_session = post_sessions[index + 1] if index + 1 < len(post_sessions) else None
        anchor = validate_pushed_anchor(record, next_session)
        eligible = bool(
            record.data["capture_classification"] == "CAPTURED_LATEST_OBSERVED_SESSION"
            and anchor["tracked"]
            and anchor["clean"]
            and anchor["committed"]
            and anchor["pushed"]
            and anchor["before_next_open"]
        )
        session_evidence[date_text] = {
            "captured": True,
            "capture_classification": record.data["capture_classification"],
            "objects_verified": session in verified,
            "anchor": anchor,
            "primary_eligible": eligible,
        }
        if eligible:
            primary_sessions.append(date_text)

    required_capture_complete = bool(
        len(required_sessions) > 0
        and len(primary_sessions) == len(required_sessions)
        and all(session in verified for session in required_sessions)
    )
    primary_accrual = [session for session in accrual if session.strftime("%Y-%m-%d") in primary_sessions]
    accrual_capture_complete = bool(
        len(accrual) == forward.ACCRUAL_SESSIONS and len(primary_accrual) == forward.ACCRUAL_SESSIONS
    )
    observed_capture_complete = len(by_session) == len(post_sessions) and all(
        session in verified for session in post_sessions
    )
    combined = bool(main["outcome_evaluation_permitted"] and required_capture_complete)
    return {
        "schema": STATUS_SCHEMA,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "ledger_id": ledger_identity(protocol),
        "ledger_records": len(records),
        "captured_sessions": len(by_session),
        "observed_post_freeze_sessions": len(post_sessions),
        "accrual_sessions_observed": len(accrual),
        "required_capture_end": required_end.strftime("%Y-%m-%d") if required_end is not None else None,
        "required_sessions": len(required_sessions),
        "primary_anchored_required_sessions": len(primary_sessions),
        "primary_anchored_accrual_sessions": len(primary_accrual),
        "observed_capture_complete": observed_capture_complete,
        "required_capture_complete": required_capture_complete,
        "accrual_capture_complete": accrual_capture_complete,
        "main_forward_progress": main,
        "session_evidence": session_evidence,
        "combined_outcome_evaluation_permitted": combined,
        "outcomes_loaded": False,
        "live_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("command", nargs="?", choices=("init", "append", "status"), default="status")
    args = parser.parse_args()
    if args.write_protocol:
        if PROTOCOL_PATH.exists():
            raise FileExistsError(f"capture protocol already exists and is immutable: {PROTOCOL_PATH}")
        protocol = build_protocol()
        PROTOCOL_PATH.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"schema": protocol["schema"], "path": str(PROTOCOL_PATH)}, indent=2))
        return
    protocol = _load_json(PROTOCOL_PATH, "capture protocol")
    if args.command == "init":
        record = initialize(protocol)
        print(json.dumps(record.data, ensure_ascii=False, indent=2))
        return
    if args.command == "append":
        records = capture_sessions(protocol)
        print(json.dumps({"records": len(records), "head": records[-1].data["record_hash"]}, indent=2))
        return
    status = build_status(protocol)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(STATUS_PATH, json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
    print(
        json.dumps(
            {
                "captured_sessions": status["captured_sessions"],
                "observed_post_freeze_sessions": status["observed_post_freeze_sessions"],
                "primary_anchored_accrual_sessions": status["primary_anchored_accrual_sessions"],
                "primary_anchored_required_sessions": status["primary_anchored_required_sessions"],
                "combined_outcome_evaluation_permitted": status["combined_outcome_evaluation_permitted"],
                "live_authorized": status["live_authorized"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[output] {STATUS_PATH}")


if __name__ == "__main__":
    main()
