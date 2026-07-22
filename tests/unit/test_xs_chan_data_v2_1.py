from __future__ import annotations

# ruff: noqa: E402, I001

import io
import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_data_v2_1 as data_v21
import xs_chan_execution_v2_1 as execution_v21
import xs_chan_oos_ledger_v2_1 as ledger_v21


SOURCE_ASOF = "2026-01-07T00:00:00Z"
REQUESTED_AT = "2026-01-07T00:01:00Z"
RESPONDED_AT = "2026-01-07T00:02:00Z"
INGESTED_AT = "2026-01-07T00:03:00Z"
COMPUTED_AT = "2026-01-07T00:04:00Z"
CREATED_AT = "2026-01-07T00:05:00Z"
DATES = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
ACTIVE = "000001.SZ"
DELISTED = "000002.SZ"


def _typed_frame(name: str, rows: list[dict]) -> pd.DataFrame:
    spec = data_v21.ALL_SPECS[name]
    columns: dict[str, pd.Series] = {}
    for column, kind in spec.columns:
        values = [row.get(column) for row in rows]
        if kind == "string":
            columns[column] = pd.Series(values, dtype="string")
        elif kind == "date":
            columns[column] = pd.Series(pd.to_datetime(values), dtype="datetime64[ns]")
        elif kind == "utc_datetime":
            columns[column] = pd.Series(pd.to_datetime(values, utc=True), dtype="datetime64[ns, UTC]")
        elif kind == "float64":
            columns[column] = pd.Series(values, dtype="float64")
        elif kind == "int64":
            columns[column] = pd.Series(values, dtype="int64")
        elif kind == "bool":
            columns[column] = pd.Series(values, dtype="bool")
        else:  # pragma: no cover - contract additions must update this fixture
            raise AssertionError(kind)
    return pd.DataFrame(columns, columns=spec.names)


def _source_fields(**values: object) -> dict[str, object]:
    return {**values, data_v21.SOURCE_ASOF: SOURCE_ASOF, data_v21.INGESTED_AT: INGESTED_AT}


def _base_source_frames(*, delisted_fully_suspended: bool = False) -> dict[str, pd.DataFrame]:
    calendar = _typed_frame(
        "calendar",
        [
            _source_fields(
                trade_date=value,
                is_open=True,
                prev_trade_date=None if index == 0 else DATES[index - 1],
                next_trade_date=None if index == len(DATES) - 1 else DATES[index + 1],
            )
            for index, value in enumerate(DATES)
        ],
    )
    master = _typed_frame(
        "security_master",
        [
            _source_fields(
                ts_code=ACTIVE,
                name="Active Co",
                exchange="SZSE",
                board="MAIN",
                list_status="L",
                list_date="2020-01-01",
                delist_date=None,
            ),
            _source_fields(
                ts_code=DELISTED,
                name="Old Co",
                exchange="SZSE",
                board="MAIN",
                list_status="D",
                list_date="2020-01-01",
                delist_date="2026-01-06",
            ),
        ],
    )
    raw_rows: list[dict] = []
    for offset, value in enumerate(DATES):
        price = 10.0 + offset
        raw_rows.append(
            _source_fields(
                ts_code=ACTIVE,
                trade_date=value,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                pre_close=price - 0.5,
                vol=1_000.0,
                amount=10_000.0,
                pct_chg=1.0,
            )
        )
    if not delisted_fully_suspended:
        raw_rows.append(
            _source_fields(
                ts_code=DELISTED,
                trade_date=DATES[0],
                open=20.0,
                high=21.0,
                low=19.0,
                close=20.5,
                pre_close=19.5,
                vol=1_000.0,
                amount=20_000.0,
                pct_chg=1.0,
            )
        )
    raw = _typed_frame("raw_daily", sorted(raw_rows, key=lambda row: (row["ts_code"], row["trade_date"])))
    adj = _typed_frame(
        "adj_factor",
        [
            _source_fields(ts_code=row.ts_code, trade_date=row.trade_date, adj_factor=1.25)
            for row in raw.itertuples(index=False)
        ],
    )
    names = _typed_frame(
        "namechange",
        [
            _source_fields(
                ts_code=DELISTED,
                name="Old Co",
                effective_from="2020-01-01",
                effective_to="2026-01-05",
            )
        ],
    )
    actions = _typed_frame(
        "corporate_actions",
        [
            _source_fields(
                action_id="action-1",
                ts_code=DELISTED,
                record_date=None,
                effective_date="2026-01-06",
                payment_date=None,
                action_type="delist_share",
                action_subtype=None,
                gross_cash_per_share=None,
                taxable_dividend_per_pre_action_share=None,
                new_share_registration_date=None,
                post_to_pre_ratio=None,
                fractional_cash_price=10.0,
                official_disposal_proceeds_per_entitled_share=None,
                disposal_settlement_date="2026-01-06",
                target_symbol=ACTIVE,
                target_share_ratio=0.5,
                terminal_cash_per_share=None,
                terminal_reason="official share consideration",
                no_value_evidence_sha256=None,
            )
        ],
    )
    status_rows = [_source_fields(ts_code=ACTIVE, trade_date=value, official_status="TRADING") for value in DATES]
    status_rows.extend(
        [
            _source_fields(
                ts_code=DELISTED,
                trade_date=DATES[0],
                official_status="SUSPENDED" if delisted_fully_suspended else "TRADING",
            ),
            _source_fields(ts_code=DELISTED, trade_date=DATES[1], official_status="SUSPENDED"),
        ]
    )
    status = _typed_frame(
        "official_daily_status", sorted(status_rows, key=lambda row: (row["ts_code"], row["trade_date"]))
    )
    daily_basic = _typed_frame(
        "daily_basic",
        [
            _source_fields(ts_code=row.ts_code, trade_date=row.trade_date, free_share=100_000.0)
            for row in raw.itertuples(index=False)
        ],
    )
    stk_limit = _typed_frame(
        "stk_limit",
        [
            _source_fields(ts_code=row.ts_code, trade_date=row.trade_date, up_limit=30.0, down_limit=1.0)
            for row in raw.itertuples(index=False)
        ],
    )
    industry = _typed_frame(
        "industry_membership",
        [
            _source_fields(
                ts_code=symbol,
                industry_code="801010",
                industry_name="Agriculture",
                classification_version="SW2021",
                effective_from="2020-01-01",
                effective_to=None,
            )
            for symbol in (ACTIVE, DELISTED)
        ],
    )
    auction = _typed_frame("open_auction", [])
    action_evidence = _typed_frame(
        "corporate_action_evidence",
        [
            _source_fields(
                action_id="action-1",
                ts_code=DELISTED,
                effective_date="2026-01-06",
                action_type="delist_share",
                authority="unit-test-exchange",
                source_document_id="doc-1",
                source_document_sha256=data_v21.sha256_bytes(b"official-document"),
                zero_recovery_text_sha256=None,
            )
        ],
    )
    states = _typed_frame(
        "chan_states",
        [{"symbol": row.ts_code, "dt": row.trade_date, "regime": 0} for row in raw.itertuples(index=False)],
    )
    audit_rows = []
    for symbol, group in raw.groupby("ts_code", sort=True):
        digest = data_v21.sha256_bytes(f"prefix:{symbol}".encode())
        audit_rows.append(
            {
                "symbol": symbol,
                "checkpoint_dt": group["trade_date"].max(),
                "prefix_rows": len(group),
                "expected_rows": len(group),
                "actual_rows": len(group),
                "expected_sha256": digest,
                "actual_sha256": digest,
                "symbol_mismatches": 0,
                "dt_mismatches": 0,
                "regime_mismatches": 0,
                "passed": True,
            }
        )
    state_audit = _typed_frame("state_audit", audit_rows)
    universe_rows = []
    for list_status in ("D", "L", "P"):
        symbols = sorted(master.loc[master["list_status"].eq(list_status), "ts_code"].tolist())
        digest = data_v21.sha256_bytes(data_v21.canonical_json_bytes(symbols))
        universe_rows.append(
            _source_fields(
                list_status=list_status,
                expected_count=len(symbols),
                received_count=len(symbols),
                expected_symbol_set_sha256=digest,
                received_symbol_set_sha256=digest,
                request_sha256=data_v21.sha256_bytes(f"request:{list_status}".encode()),
                response_sha256=data_v21.sha256_bytes(f"response:{list_status}".encode()),
            )
        )
    universe = _typed_frame("universe_reconciliation", universe_rows)
    return {
        "calendar": calendar,
        "security_master": master,
        "raw_daily": raw,
        "adj_factor": adj,
        "namechange": names,
        "corporate_actions": actions,
        "official_daily_status": status,
        "open_auction": auction,
        "daily_basic": daily_basic,
        "stk_limit": stk_limit,
        "industry_membership": industry,
        "corporate_action_evidence": action_evidence,
        "universe_reconciliation": universe,
        "chan_states": states,
        "state_audit": state_audit,
    }


def _truncate_frames_to_observation(
    frames: dict[str, pd.DataFrame], observation: pd.Timestamp, *, retain_future_chan_state: bool = False
) -> None:
    dated = {
        "raw_daily": "trade_date",
        "adj_factor": "trade_date",
        "official_daily_status": "trade_date",
        "open_auction": "trade_date",
        "daily_basic": "trade_date",
        "stk_limit": "trade_date",
        "chan_states": "dt",
    }
    future_chan = frames["chan_states"].loc[frames["chan_states"]["dt"].gt(observation)].head(1)
    for artifact, column in dated.items():
        frames[artifact] = frames[artifact].loc[frames[artifact][column].le(observation)].reset_index(drop=True)
    if retain_future_chan_state and not future_chan.empty:
        frames["chan_states"] = _typed_frame(
            "chan_states",
            sorted(
                [*frames["chan_states"].to_dict(orient="records"), *future_chan.to_dict(orient="records")],
                key=lambda row: (row["symbol"], row["dt"]),
            ),
        )

    actions = frames["corporate_actions"]
    entitlement = actions["action_type"].isin({"cash_dividend", "share_change", "rights_issue"})
    observed = (entitlement & actions["record_date"].le(observation)) | (
        ~entitlement & actions["effective_date"].le(observation)
    )
    frames["corporate_actions"] = actions.loc[observed].reset_index(drop=True)
    action_ids = set(frames["corporate_actions"]["action_id"])
    frames["corporate_action_evidence"] = (
        frames["corporate_action_evidence"]
        .loc[frames["corporate_action_evidence"]["action_id"].isin(action_ids)]
        .reset_index(drop=True)
    )

    audit_rows = []
    for symbol, group in frames["raw_daily"].groupby("ts_code", sort=True):
        digest = data_v21.sha256_bytes(f"observation-prefix:{symbol}".encode())
        audit_rows.append(
            {
                "symbol": symbol,
                "checkpoint_dt": group["trade_date"].max(),
                "prefix_rows": len(group),
                "expected_rows": len(group),
                "actual_rows": len(group),
                "expected_sha256": digest,
                "actual_sha256": digest,
                "symbol_mismatches": 0,
                "dt_mismatches": 0,
                "regime_mismatches": 0,
                "passed": True,
            }
        )
    frames["state_audit"] = _typed_frame("state_audit", audit_rows)


def _state_binding(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = frames["raw_daily"].set_index(["ts_code", "trade_date"], drop=False)
    adj = frames["adj_factor"].set_index(["ts_code", "trade_date"], drop=False)
    status = frames["official_daily_status"]
    rows: list[dict] = []
    for status_row in status.to_dict(orient="records"):
        key = (status_row["ts_code"], status_row["trade_date"])
        trading = status_row["official_status"] == "TRADING"
        row: dict[str, object] = {
            "ts_code": key[0],
            "trade_date": key[1],
            "official_status": status_row["official_status"],
            "included_in_state_engine": trading,
            "status_row_sha256": data_v21.canonical_row_sha256("official_daily_status", status_row),
            data_v21.COMPUTED_AT: COMPUTED_AT,
        }
        if trading:
            raw_row = raw.loc[key].to_dict()
            adj_row = adj.loc[key].to_dict()
            factor = float(adj_row["adj_factor"])
            for column in ("open", "high", "low", "close"):
                row[f"raw_{column}"] = float(raw_row[column])
                row[f"adjusted_{column}"] = float(raw_row[column]) * factor
            row["adj_factor"] = factor
            row["raw_row_sha256"] = data_v21.canonical_row_sha256("raw_daily", raw_row)
            row["adj_factor_row_sha256"] = data_v21.canonical_row_sha256("adj_factor", adj_row)
        else:
            for column in (
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "adj_factor",
                "adjusted_open",
                "adjusted_high",
                "adjusted_low",
                "adjusted_close",
                "raw_row_sha256",
                "adj_factor_row_sha256",
            ):
                row[column] = None
        row["input_row_sha256"] = data_v21.state_input_row_sha256(
            {name: row[name] for name in data_v21.STATE_INPUT_HASH_COLUMNS}
        )
        rows.append(row)
    return _typed_frame("state_input_binding", rows)


def _response(
    endpoint: str,
    symbol: str | None,
    frame: pd.DataFrame,
    *,
    start_date: str = "2026-01-02",
    end_date: str = "2026-01-06",
) -> dict:
    rows = [data_v21.canonical_source_row(endpoint, row) for row in frame.to_dict(orient="records")]
    return {
        "schema_version": data_v21.SOURCE_RESPONSE_SCHEMA,
        "provider": "unit-test-official-source",
        "endpoint": endpoint,
        "scope": {"ts_code": symbol, "start_date": start_date, "end_date": end_date},
        "source_asof_utc": SOURCE_ASOF,
        "requested_at_utc": REQUESTED_AT,
        "responded_at_utc": RESPONDED_AT,
        "ingested_at_utc": INGESTED_AT,
        "kind": "POSITIVE" if rows else "ZERO",
        "rows": rows,
    }


def _responses(frames: dict[str, pd.DataFrame], observation_through_session: str) -> list[dict]:
    start = frames["calendar"]["trade_date"].min().strftime("%Y-%m-%d")
    calendar_end = frames["calendar"]["trade_date"].max().strftime("%Y-%m-%d")
    result = [
        _response(
            endpoint,
            None,
            frames[endpoint],
            start_date=start,
            end_date=calendar_end if endpoint == "calendar" else observation_through_session,
        )
        for endpoint in sorted(data_v21.GLOBAL_ENDPOINTS)
    ]
    for endpoint in sorted(data_v21.SYMBOL_ENDPOINTS):
        artifact = frames[endpoint]
        for symbol in (ACTIVE, DELISTED):
            subset = artifact.loc[artifact["ts_code"].eq(symbol)].reset_index(drop=True)
            result.append(
                _response(
                    endpoint,
                    symbol,
                    subset,
                    start_date=start,
                    end_date=observation_through_session,
                )
            )
    return result


def _response_map(responses: list[dict]) -> dict[tuple[str, str | None], tuple[str, dict]]:
    return {
        (response["endpoint"], response["scope"]["ts_code"]): (
            data_v21.sha256_bytes(data_v21.canonical_json_bytes(response)),
            response,
        )
        for response in responses
    }


def _add_reconciliation_frames(
    frames: dict[str, pd.DataFrame], responses: list[dict], observation_through_session: str
) -> None:
    response_map = _response_map(responses)
    requested_start = frames["calendar"]["trade_date"].min().strftime("%Y-%m-%d")
    common_rows: dict[str, list[dict]] = {
        "namechange_reconciliation": [],
        "open_auction_reconciliation": [],
    }
    for artifact, endpoint in (
        ("namechange_reconciliation", "namechange"),
        ("open_auction_reconciliation", "open_auction"),
    ):
        for symbol in (ACTIVE, DELISTED):
            digest, response = response_map[(endpoint, symbol)]
            common_rows[artifact].append(
                _source_fields(
                    ts_code=symbol,
                    response_kind=response["kind"],
                    response_sha256=digest,
                    source_authority=response["provider"],
                    requested_start=requested_start,
                    requested_end=observation_through_session,
                )
            )
        frames[artifact] = _typed_frame(artifact, common_rows[artifact])
    raw_rows = []
    for symbol in (ACTIVE, DELISTED):
        raw_digest, raw_response = response_map[("raw_daily", symbol)]
        status_digest, _ = response_map[("official_daily_status", symbol)]
        raw_rows.append(
            _source_fields(
                ts_code=symbol,
                response_kind=raw_response["kind"],
                response_sha256=raw_digest,
                status_response_sha256=status_digest,
                source_authority=raw_response["provider"],
                requested_start=requested_start,
                requested_end=observation_through_session,
            )
        )
    frames["raw_daily_reconciliation"] = _typed_frame("raw_daily_reconciliation", raw_rows)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _write_object(root: Path, payload: bytes) -> tuple[str, int]:
    digest = data_v21.sha256_bytes(payload)
    (root / "objects" / "sha256" / digest).write_bytes(payload)
    return digest, len(payload)


def _build_snapshot(
    tmp_path: Path,
    *,
    observation_through_session: str = "2026-01-06",
    delisted_fully_suspended: bool = False,
    mutate_frames: Callable[[dict[str, pd.DataFrame]], None] | None = None,
    mutate_responses: Callable[[list[dict]], None] | None = None,
    artifact_blob_override: Mapping[str, bytes] | None = None,
    mutate_manifest: Callable[[dict], None] | None = None,
) -> tuple[Path, dict[str, pd.DataFrame], dict]:
    root = tmp_path / "snapshot"
    (root / "objects" / "sha256").mkdir(parents=True)
    (root / "manifests").mkdir()
    frames = _base_source_frames(delisted_fully_suspended=delisted_fully_suspended)
    if mutate_frames:
        mutate_frames(frames)
    frames["state_input_binding"] = _state_binding(frames)
    responses = _responses(frames, observation_through_session)
    _add_reconciliation_frames(frames, responses, observation_through_session)
    protocol_sha256 = data_v21.frozen_protocol_sha256()
    artifacts: dict[str, dict] = {}
    evidence_objects: dict[str, dict] = {}
    overrides = dict(artifact_blob_override or {})
    for name in data_v21.REQUIRED_ARTIFACTS:
        payload = overrides.get(name, _parquet_bytes(frames[name]))
        digest, size = _write_object(root, payload)
        artifacts[name] = {
            "sha256": digest,
            "size": size,
            "media_type": data_v21.PARQUET_MEDIA_TYPE,
            "authority": data_v21.OBJECT_AUTHORITIES[name],
            "provenance_sha256": data_v21.object_provenance_sha256(name, digest, protocol_sha256),
        }
    for name in data_v21.REQUIRED_EVIDENCE_OBJECTS:
        payload = overrides.get(name, _parquet_bytes(frames[name]))
        digest, size = _write_object(root, payload)
        evidence_objects[name] = {
            "sha256": digest,
            "size": size,
            "media_type": data_v21.PARQUET_MEDIA_TYPE,
            "authority": data_v21.OBJECT_AUTHORITIES[name],
            "provenance_sha256": data_v21.object_provenance_sha256(name, digest, protocol_sha256),
        }
    if mutate_responses:
        mutate_responses(responses)
    source_entries = []
    for response in responses:
        payload = data_v21.canonical_json_bytes(response)
        digest, size = _write_object(root, payload)
        source_entries.append(
            {
                "sha256": digest,
                "size": size,
                "media_type": data_v21.SOURCE_RESPONSE_MEDIA_TYPE,
            }
        )
    manifest = {
        "contract_id": data_v21.CONTRACT_ID,
        "manifest_version": data_v21.MANIFEST_VERSION,
        "protocol_id": data_v21.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "created_at_utc": CREATED_AT,
        "observation_through_session": observation_through_session,
        "artifacts": artifacts,
        "evidence_objects": evidence_objects,
        "source_objects": sorted(source_entries, key=lambda entry: entry["sha256"]),
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    payload = data_v21.canonical_json_bytes(manifest)
    digest = data_v21.sha256_bytes(payload)
    path = root / "manifests" / f"{digest}.json"
    path.write_bytes(payload)
    return path, frames, manifest


def _clone_verified_snapshot(
    snapshot: data_v21.VerifiedDataSnapshot,
    *,
    created_at_utc: str = "2026-01-07T00:06:00Z",
    replace_frame: tuple[str, pd.DataFrame] | None = None,
    remove_source_sha256: str | None = None,
    mutate_manifest: Callable[[dict], None] | None = None,
) -> data_v21.VerifiedDataSnapshot:
    manifest = snapshot.manifest_dict()
    artifact_blobs = {name: snapshot.read_artifact_bytes(name) for name in snapshot.artifact_names}
    evidence_blobs = {name: bytes(snapshot._evidence_blobs[name]) for name in snapshot.evidence_object_names}
    responses = {digest: snapshot.read_source_response(digest) for digest in snapshot.source_response_sha256}
    manifest["created_at_utc"] = created_at_utc
    if replace_frame is not None:
        name, frame = replace_frame
        payload = _parquet_bytes(frame)
        section = "artifacts" if name in data_v21.ARTIFACT_SPECS else "evidence_objects"
        if section == "artifacts":
            artifact_blobs[name] = payload
        else:
            evidence_blobs[name] = payload
        digest = data_v21.sha256_bytes(payload)
        manifest[section][name].update(
            sha256=digest,
            size=len(payload),
            provenance_sha256=data_v21.object_provenance_sha256(name, digest, manifest["protocol_sha256"]),
        )
    if remove_source_sha256 is not None:
        responses.pop(remove_source_sha256)
        manifest["source_objects"] = [
            entry for entry in manifest["source_objects"] if entry["sha256"] != remove_source_sha256
        ]
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_sha256 = data_v21.sha256_bytes(data_v21.canonical_json_bytes(manifest))
    return data_v21.VerifiedDataSnapshot(
        manifest_sha256=manifest_sha256,
        created_at_utc=created_at_utc,
        observation_through_session=manifest["observation_through_session"],
        manifest=manifest,
        artifact_blobs=artifact_blobs,
        evidence_blobs=evidence_blobs,
        responses=responses,
    )


def _issue_codes(report: data_v21.DataValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def _market_binding_case(
    *,
    auction_kind: str = "ZERO",
    include_session_auction: bool = False,
    terminal_action: bool = False,
    cash_dividend: bool = False,
) -> tuple[data_v21.VerifiedDataSnapshot, dict, dict[str, pd.DataFrame]]:
    if terminal_action and cash_dividend:
        raise AssertionError("test fixture supports one corporate action at a time")
    dates = pd.bdate_range("2025-11-24", periods=24).normalize()
    session = dates[21]
    settlement = dates[22]
    decision = dates[20]
    calendar = _typed_frame(
        "calendar",
        [
            _source_fields(
                trade_date=value,
                is_open=True,
                prev_trade_date=None if index == 0 else dates[index - 1],
                next_trade_date=None if index == len(dates) - 1 else dates[index + 1],
            )
            for index, value in enumerate(dates)
        ],
    )
    master = _typed_frame(
        "security_master",
        [
            _source_fields(
                ts_code=ACTIVE,
                name="Active Co",
                exchange="SZSE",
                board="MAIN",
                list_status="D" if terminal_action else "L",
                list_date="2020-01-01",
                delist_date=session if terminal_action else None,
            )
        ],
    )
    observed_dates = dates[:-1]
    if terminal_action:
        observed_dates = observed_dates[observed_dates < session]
    raw = _typed_frame(
        "raw_daily",
        [
            _source_fields(
                ts_code=ACTIVE,
                trade_date=value,
                open=10.0 + index / 10,
                high=11.0 + index / 10,
                low=9.0 + index / 10,
                close=10.5 + index / 10,
                pre_close=9.5 + index / 10,
                vol=1_000.0 + index,
                amount=10_000.0 + index,
                pct_chg=1.0,
            )
            for index, value in enumerate(observed_dates)
        ],
    )
    status = _typed_frame(
        "official_daily_status",
        [_source_fields(ts_code=ACTIVE, trade_date=value, official_status="TRADING") for value in observed_dates],
    )
    limits = _typed_frame(
        "stk_limit",
        [_source_fields(ts_code=ACTIVE, trade_date=value, up_limit=30.0, down_limit=1.0) for value in observed_dates],
    )
    auction_rows = []
    if auction_kind == "POSITIVE":
        auction_date = session if include_session_auction else (decision if terminal_action else dates[0])
        auction_rows.append(
            _source_fields(
                ts_code=ACTIVE,
                trade_date=auction_date,
                auction_price=12.1,
                auction_volume=0.0 if terminal_action and not include_session_auction else 1_000.0,
                auction_amount=0.0 if terminal_action and not include_session_auction else 12_100.0,
            )
        )
    auction = _typed_frame("open_auction", auction_rows)
    actions = _typed_frame("corporate_actions", [])
    action_evidence = _typed_frame("corporate_action_evidence", [])
    if terminal_action:
        actions = _typed_frame(
            "corporate_actions",
            [
                _source_fields(
                    action_id="terminal-1",
                    ts_code=ACTIVE,
                    record_date=None,
                    effective_date=session,
                    payment_date=None,
                    action_type="delist_cash",
                    action_subtype=None,
                    gross_cash_per_share=None,
                    taxable_dividend_per_pre_action_share=None,
                    new_share_registration_date=None,
                    post_to_pre_ratio=None,
                    fractional_cash_price=None,
                    official_disposal_proceeds_per_entitled_share=None,
                    disposal_settlement_date=settlement,
                    target_symbol=None,
                    target_share_ratio=None,
                    terminal_cash_per_share=8.25,
                    terminal_reason="official cash consideration",
                    no_value_evidence_sha256=None,
                )
            ],
        )
        action_evidence = _typed_frame(
            "corporate_action_evidence",
            [
                _source_fields(
                    action_id="terminal-1",
                    ts_code=ACTIVE,
                    effective_date=session,
                    action_type="delist_cash",
                    authority="unit-test-exchange",
                    source_document_id="terminal-doc-1",
                    source_document_sha256=data_v21.sha256_bytes(b"terminal-document"),
                    zero_recovery_text_sha256=None,
                )
            ],
        )
    elif cash_dividend:
        actions = _typed_frame(
            "corporate_actions",
            [
                _source_fields(
                    action_id="dividend-1",
                    ts_code=ACTIVE,
                    record_date=session,
                    effective_date=settlement,
                    payment_date=dates[23],
                    action_type="cash_dividend",
                    action_subtype=None,
                    gross_cash_per_share=0.25,
                    taxable_dividend_per_pre_action_share=None,
                    new_share_registration_date=None,
                    post_to_pre_ratio=None,
                    fractional_cash_price=None,
                    official_disposal_proceeds_per_entitled_share=None,
                    disposal_settlement_date=None,
                    target_symbol=None,
                    target_share_ratio=None,
                    terminal_cash_per_share=None,
                    terminal_reason=None,
                    no_value_evidence_sha256=None,
                )
            ],
        )
        action_evidence = _typed_frame(
            "corporate_action_evidence",
            [
                _source_fields(
                    action_id="dividend-1",
                    ts_code=ACTIVE,
                    effective_date=settlement,
                    action_type="cash_dividend",
                    authority="unit-test-exchange",
                    source_document_id="dividend-doc-1",
                    source_document_sha256=data_v21.sha256_bytes(b"dividend-document"),
                    zero_recovery_text_sha256=None,
                )
            ],
        )
    frames = {
        "calendar": calendar,
        "security_master": master,
        "raw_daily": raw,
        "official_daily_status": status,
        "stk_limit": limits,
        "open_auction": auction,
        "corporate_actions": actions,
        "corporate_action_evidence": action_evidence,
    }
    response = _response("open_auction", ACTIVE, auction)
    response["scope"] = {
        "ts_code": ACTIVE,
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
    }
    response["kind"] = auction_kind
    response_sha = data_v21.sha256_bytes(data_v21.canonical_json_bytes(response))
    snapshot_sha = "b" * 64
    observation_through_session = (session if terminal_action else observed_dates.max()).strftime("%Y-%m-%d")
    artifact_blobs = {name: _parquet_bytes(frame) for name, frame in frames.items() if name != "official_daily_status"}
    snapshot = data_v21.VerifiedDataSnapshot(
        manifest_sha256=snapshot_sha,
        created_at_utc=CREATED_AT,
        observation_through_session=observation_through_session,
        manifest={
            "protocol_sha256": data_v21.frozen_protocol_sha256(),
            "observation_through_session": observation_through_session,
        },
        artifact_blobs=artifact_blobs,
        evidence_blobs={"official_daily_status": _parquet_bytes(status)},
        responses={response_sha: response},
    )
    if terminal_action:
        session_raw = session_status = session_limit = None
        execution_status = "delisted"
        execution_open = execution_pre_close = execution_close = execution_up = execution_down = 0.0
        status_sha = raw_sha = limit_sha = None
    else:
        session_raw = raw.loc[raw["trade_date"].eq(session)].iloc[0].to_dict()
        session_status = status.loc[status["trade_date"].eq(session)].iloc[0].to_dict()
        session_limit = limits.loc[limits["trade_date"].eq(session)].iloc[0].to_dict()
        execution_status = "trading"
        execution_open = float(session_raw["open"])
        execution_pre_close = float(session_raw["pre_close"])
        execution_close = float(session_raw["close"])
        execution_up = float(session_limit["up_limit"])
        execution_down = float(session_limit["down_limit"])
        status_sha = data_v21.canonical_row_sha256("official_daily_status", session_status)
        raw_sha = data_v21.canonical_row_sha256("raw_daily", session_raw)
        limit_sha = data_v21.canonical_row_sha256("stk_limit", session_limit)
    adv_rows = raw.loc[raw["trade_date"].le(decision)].tail(20).to_dict(orient="records")
    adv_cny = float(pd.Series([row["amount"] for row in adv_rows]).rolling(20, min_periods=20).mean().iloc[-1]) * 1000
    auction_at_session = auction.loc[auction["trade_date"].eq(session)]
    auction_row_sha = (
        None
        if auction_at_session.empty
        else data_v21.canonical_row_sha256("open_auction", auction_at_session.iloc[0].to_dict())
    )
    terminal_action_source_sha: list[str] = []
    if terminal_action:
        terminal_action_row = actions.iloc[0].to_dict()
        terminal_evidence_row = action_evidence.iloc[0].to_dict()
        terminal_source = {
            "data_snapshot_sha256": snapshot_sha,
            "action_id": "terminal-1",
            "corporate_action_row_sha256": data_v21.canonical_row_sha256("corporate_actions", terminal_action_row),
            "corporate_action_evidence_row_sha256": data_v21.canonical_row_sha256(
                "corporate_action_evidence", terminal_evidence_row
            ),
            "source_document_sha256": terminal_evidence_row["source_document_sha256"],
        }
        terminal_action_source_sha.append(data_v21.execution_source_row_sha256("corporate_action", terminal_source))
    open_source = {
        "data_snapshot_sha256": snapshot_sha,
        "symbol": ACTIVE,
        "session": session.strftime("%Y-%m-%d"),
        "decision_session": decision.strftime("%Y-%m-%d"),
        "official_status_row_sha256": status_sha,
        "raw_daily_row_sha256": raw_sha,
        "stk_limit_row_sha256": limit_sha,
        "adv20_raw_row_sha256": [data_v21.canonical_row_sha256("raw_daily", row) for row in adv_rows],
        "open_auction_row_sha256": auction_row_sha,
        "open_auction_response_sha256": response_sha,
        "terminal_action_source_sha256": terminal_action_source_sha,
    }
    eod_source = {
        "data_snapshot_sha256": snapshot_sha,
        "symbol": ACTIVE,
        "session": session.strftime("%Y-%m-%d"),
        "official_status_row_sha256": status_sha,
        "raw_daily_row_sha256": raw_sha,
        "terminal_action_source_sha256": terminal_action_source_sha,
    }
    corporate_actions: list[dict] = []
    if terminal_action:
        corporate_actions.append(
            {
                "action_id": "terminal-1",
                "symbol": ACTIVE,
                "effective_session": session.strftime("%Y-%m-%d"),
                "action_type": "delist_cash",
                "source_row_sha256": terminal_action_source_sha[0],
                "cash_per_share": 8.25,
                "terminal_reason": "official cash consideration",
                "disposal_settlement_session": settlement.strftime("%Y-%m-%d"),
            }
        )
    elif cash_dividend:
        action_row = actions.iloc[0].to_dict()
        evidence_row = action_evidence.iloc[0].to_dict()
        action_source = {
            "data_snapshot_sha256": snapshot_sha,
            "action_id": "dividend-1",
            "corporate_action_row_sha256": data_v21.canonical_row_sha256("corporate_actions", action_row),
            "corporate_action_evidence_row_sha256": data_v21.canonical_row_sha256(
                "corporate_action_evidence", evidence_row
            ),
            "source_document_sha256": evidence_row["source_document_sha256"],
        }
        corporate_actions.append(
            {
                "action_id": "dividend-1",
                "symbol": ACTIVE,
                "record_session": session.strftime("%Y-%m-%d"),
                "effective_session": settlement.strftime("%Y-%m-%d"),
                "action_type": "cash_dividend",
                "source_row_sha256": data_v21.execution_source_row_sha256("corporate_action", action_source),
                "gross_cash_per_share": 0.25,
                "payment_session": dates[23].strftime("%Y-%m-%d"),
            }
        )
    execution_input = {
        "schema": "xs_chan_execution_input_v2_1",
        "protocol_sha256": data_v21.frozen_protocol_sha256(),
        "data_snapshot_sha256": snapshot_sha,
        "trial_id": "trial-v2-1",
        "arm_id": "F",
        "seed_id": None,
        "scenario_id": "gross",
        "initial_capital_cny": 100_000_000.0,
        "annual_cash_yield": 0.0,
        "terminal_policy": "mark_to_market_no_forced_liquidation",
        "cost_model": asdict(execution_v21.CostModelV21()),
        "sessions": [
            {
                "session": session.strftime("%Y-%m-%d"),
                "settlement_session": settlement.strftime("%Y-%m-%d"),
                "open_snapshot": [
                    {
                        "symbol": ACTIVE,
                        "open": execution_open,
                        "pre_close": execution_pre_close,
                        "status": execution_status,
                        "limit_up": execution_up,
                        "limit_down": execution_down,
                        "adv20_cny_asof_decision": adv_cny,
                        "open_auction_turnover_cny": (
                            float(auction_at_session.iloc[0]["auction_amount"]) * 1000
                            if not auction_at_session.empty
                            else 0.0
                        ),
                        "lot_size": 100,
                        "source_row_sha256": data_v21.execution_source_row_sha256("open", open_source),
                    }
                ],
                "eod_snapshot": [
                    {
                        "symbol": ACTIVE,
                        "close": execution_close,
                        "status": execution_status,
                        "source_row_sha256": data_v21.execution_source_row_sha256("eod", eod_source),
                    }
                ],
                "corporate_actions": corporate_actions,
                "decision": {
                    "decision_session": decision.strftime("%Y-%m-%d"),
                    "execution_session": session.strftime("%Y-%m-%d"),
                    "ordered_symbols": [],
                    "new_entry_symbols": [],
                    "gate_eligible_new_entry_opportunities": 0,
                    "slots": 50,
                    "cash_buffer_fraction": 0.005,
                    "sizing_nav_cny": 100_000_000.0,
                    "sizing_nav_record_sha256": data_v21.sha256_bytes(b"sizing-nav"),
                    "selection_identity_sha256": execution_v21.selection_identity(
                        decision.strftime("%Y-%m-%d"), session.strftime("%Y-%m-%d"), []
                    ),
                    "decision_record_sha256": data_v21.sha256_bytes(b"decision-record"),
                },
            }
        ],
    }
    if terminal_action:
        _prepend_anchor_session(snapshot, execution_input, frames)
    return snapshot, execution_input, frames


def _prepend_anchor_session(
    snapshot: data_v21.VerifiedDataSnapshot,
    execution_input: dict,
    frames: dict[str, pd.DataFrame],
) -> None:
    execution_session = execution_input["sessions"][0]
    anchor = pd.Timestamp(execution_session["decision"]["decision_session"])
    calendar = frames["calendar"]
    calendar_row = calendar.loc[calendar["trade_date"].eq(anchor)].iloc[0]
    adv_decision = pd.Timestamp(calendar_row["prev_trade_date"])
    raw = frames["raw_daily"]
    status = frames["official_daily_status"]
    limits = frames["stk_limit"]
    raw_row = raw.loc[raw["trade_date"].eq(anchor)].iloc[0].to_dict()
    status_row = status.loc[status["trade_date"].eq(anchor)].iloc[0].to_dict()
    limit_row = limits.loc[limits["trade_date"].eq(anchor)].iloc[0].to_dict()
    adv_rows = raw.loc[raw["trade_date"].le(adv_decision)].tail(20).to_dict(orient="records")
    adv_cny = float(pd.Series([row["amount"] for row in adv_rows]).rolling(20, min_periods=20).mean().iloc[-1]) * 1000
    response_sha = snapshot.source_response_sha256[0]
    auction = frames["open_auction"]
    anchor_auction = auction.loc[auction["trade_date"].eq(anchor)]
    anchor_auction_sha = (
        None
        if anchor_auction.empty
        else data_v21.canonical_row_sha256("open_auction", anchor_auction.iloc[0].to_dict())
    )
    anchor_auction_cny = 0.0 if anchor_auction.empty else float(anchor_auction.iloc[0]["auction_amount"]) * 1000
    open_source = {
        "data_snapshot_sha256": snapshot.manifest_sha256,
        "symbol": ACTIVE,
        "session": anchor.strftime("%Y-%m-%d"),
        "decision_session": adv_decision.strftime("%Y-%m-%d"),
        "official_status_row_sha256": data_v21.canonical_row_sha256("official_daily_status", status_row),
        "raw_daily_row_sha256": data_v21.canonical_row_sha256("raw_daily", raw_row),
        "stk_limit_row_sha256": data_v21.canonical_row_sha256("stk_limit", limit_row),
        "adv20_raw_row_sha256": [data_v21.canonical_row_sha256("raw_daily", row) for row in adv_rows],
        "open_auction_row_sha256": anchor_auction_sha,
        "open_auction_response_sha256": response_sha,
        "terminal_action_source_sha256": [],
    }
    eod_source = {
        "data_snapshot_sha256": snapshot.manifest_sha256,
        "symbol": ACTIVE,
        "session": anchor.strftime("%Y-%m-%d"),
        "official_status_row_sha256": data_v21.canonical_row_sha256("official_daily_status", status_row),
        "raw_daily_row_sha256": data_v21.canonical_row_sha256("raw_daily", raw_row),
        "terminal_action_source_sha256": [],
    }
    execution_input["sessions"].insert(
        0,
        {
            "session": anchor.strftime("%Y-%m-%d"),
            "settlement_session": execution_session["session"],
            "open_snapshot": [
                {
                    "symbol": ACTIVE,
                    "open": float(raw_row["open"]),
                    "pre_close": float(raw_row["pre_close"]),
                    "status": "trading",
                    "limit_up": float(limit_row["up_limit"]),
                    "limit_down": float(limit_row["down_limit"]),
                    "adv20_cny_asof_decision": adv_cny,
                    "open_auction_turnover_cny": anchor_auction_cny,
                    "lot_size": 100,
                    "source_row_sha256": data_v21.execution_source_row_sha256("open", open_source),
                }
            ],
            "eod_snapshot": [
                {
                    "symbol": ACTIVE,
                    "close": float(raw_row["close"]),
                    "status": "trading",
                    "source_row_sha256": data_v21.execution_source_row_sha256("eod", eod_source),
                }
            ],
            "corporate_actions": [],
            "decision": None,
        },
    )


def _replace_parquet_entry(root: Path, manifest: dict, section: str, name: str, frame: pd.DataFrame) -> Path:
    payload = _parquet_bytes(frame)
    digest, size = _write_object(root, payload)
    manifest[section][name].update({"sha256": digest, "size": size})
    manifest[section][name]["provenance_sha256"] = data_v21.object_provenance_sha256(
        name, digest, manifest["protocol_sha256"]
    )
    manifest_payload = data_v21.canonical_json_bytes(manifest)
    new_path = root / "manifests" / f"{data_v21.sha256_bytes(manifest_payload)}.json"
    new_path.write_bytes(manifest_payload)
    return new_path


def test_contract_exactly_matches_authoritative_protocol() -> None:
    protocol = json.loads(data_v21.FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert data_v21.MANIFEST_VERSION == protocol["data_contract"]["manifest_version"] == 3
    assert set(data_v21.REQUIRED_ARTIFACTS) == set(protocol["data_contract"]["required_artifacts"])
    assert len(data_v21.REQUIRED_ARTIFACTS) == 17
    assert data_v21.frozen_protocol_sha256() == "8501bdd242cdd961b13cb4688cc7e2fd6d621342f85039a3223a18c4579ea203"
    assert data_v21.protocol_file_sha256() != data_v21.frozen_protocol_sha256()


def test_data_contract_identity_and_validation_report_digest_are_stable() -> None:
    assert (
        data_v21.data_contract_identity_sha256() == "20400f8cc40aef988b13969eac8923d900604d8c0b389bd717b2d84e9e30f294"
    )

    report = data_v21.DataValidationReport(manifest_sha256="1" * 64)
    mapping = report.to_dict()
    expected = hashlib.sha256(
        json.dumps(
            mapping,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert data_v21.data_validation_report_sha256(report) == expected
    assert data_v21.data_validation_report_sha256(mapping) == expected
    assert data_v21.data_validation_report_sha256(mapping) == ledger_v21._object_sha256(mapping)
    with pytest.raises(data_v21.ContractError, match="must be a DataValidationReport or mapping"):
        data_v21.data_validation_report_sha256([])


def test_data_chain_transition_is_byte_identical_to_ledger() -> None:
    inputs = {
        "data_contract_identity_sha256": data_v21.data_contract_identity_sha256(),
        "previous_data_chain_head_sha256": "0" * 64,
        "decision_session": "2026-07-20",
        "snapshot_cutoff_utc": "2026-07-20T22:00:00.000000Z",
        "data_manifest_sha256": "1" * 64,
        "data_validation_report_sha256": "2" * 64,
    }

    assert data_v21.data_chain_transition_sha256(**inputs) == ledger_v21.data_chain_transition_sha256(**inputs)
    assert data_v21.data_chain_transition_sha256(**inputs) == (
        "5ab80bbc7aa0a6ba4b1bc7f26926ea70e0245ca99085031161f7400535fb63d0"
    )
    with pytest.raises(data_v21.ContractError, match="canonical UTC Z format with microseconds"):
        data_v21.data_chain_transition_sha256(**{**inputs, "snapshot_cutoff_utc": "2026-07-20T22:00:00Z"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.pop("observation_through_session"),
        lambda manifest: manifest.update(observation_through_session="2026-1-6"),
    ],
)
def test_manifest_requires_canonical_observation_horizon(tmp_path: Path, mutation: Callable[[dict], None]) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_manifest=mutation)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert "formal_entry_closed" in _issue_codes(report)


def test_future_calendar_session_is_allowed_without_future_observations(tmp_path: Path) -> None:
    observation = DATES[1]
    observation_text = observation.strftime("%Y-%m-%d")
    manifest_path, _, _ = _build_snapshot(
        tmp_path,
        observation_through_session=observation_text,
        mutate_frames=lambda frames: _truncate_frames_to_observation(frames, observation),
    )

    report, snapshot = data_v21.load_verified_snapshot(manifest_path)

    assert report.valid is True
    assert snapshot is not None
    assert snapshot.observation_through_session == observation_text
    assert snapshot.read_parquet("calendar")["trade_date"].max() == DATES[2]
    assert snapshot.read_parquet("raw_daily")["trade_date"].max() == observation
    assert snapshot.read_evidence_parquet("official_daily_status")["trade_date"].max() == observation
    assert snapshot.read_parquet("corporate_actions").empty
    assert report.evidence["observation_through_session"] == observation_text


def test_record_date_entitlement_may_bind_later_effective_date(tmp_path: Path) -> None:
    observation = DATES[1]

    def prepare(frames: dict[str, pd.DataFrame]) -> None:
        _truncate_frames_to_observation(frames, observation)
        frames["corporate_actions"] = _typed_frame(
            "corporate_actions",
            [
                _source_fields(
                    action_id="dividend-at-t",
                    ts_code=ACTIVE,
                    record_date=observation,
                    effective_date=DATES[2],
                    payment_date="2026-01-07",
                    action_type="cash_dividend",
                    action_subtype=None,
                    gross_cash_per_share=0.25,
                    taxable_dividend_per_pre_action_share=None,
                    new_share_registration_date=None,
                    post_to_pre_ratio=None,
                    fractional_cash_price=None,
                    official_disposal_proceeds_per_entitled_share=None,
                    disposal_settlement_date=None,
                    target_symbol=None,
                    target_share_ratio=None,
                    terminal_cash_per_share=None,
                    terminal_reason=None,
                    no_value_evidence_sha256=None,
                )
            ],
        )
        frames["corporate_action_evidence"] = _typed_frame(
            "corporate_action_evidence",
            [
                _source_fields(
                    action_id="dividend-at-t",
                    ts_code=ACTIVE,
                    effective_date=DATES[2],
                    action_type="cash_dividend",
                    authority="unit-test-exchange",
                    source_document_id="dividend-doc-at-t",
                    source_document_sha256=data_v21.sha256_bytes(b"dividend-at-t-document"),
                    zero_recovery_text_sha256=None,
                )
            ],
        )

    manifest_path, _, _ = _build_snapshot(
        tmp_path,
        observation_through_session=observation.strftime("%Y-%m-%d"),
        mutate_frames=prepare,
    )

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is True


def test_post_horizon_observation_is_rejected_even_when_calendar_contains_session(tmp_path: Path) -> None:
    observation = DATES[1]
    manifest_path, _, _ = _build_snapshot(
        tmp_path,
        observation_through_session=observation.strftime("%Y-%m-%d"),
        mutate_frames=lambda frames: _truncate_frames_to_observation(
            frames, observation, retain_future_chan_state=True
        ),
    )

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["artifact_semantics"] is False
    leaks = [issue for issue in report.issues if issue.code == "observation_horizon_leakage"]
    assert any(issue.artifact == "chan_states" and "2026-01-06" in issue.message for issue in leaks)


def test_snapshot_extension_accepts_only_new_primary_keys(tmp_path: Path) -> None:
    manifest_path, frames, _ = _build_snapshot(tmp_path)
    loaded, previous = data_v21.load_verified_snapshot(manifest_path)
    assert loaded.valid and previous is not None
    appended_states = _typed_frame(
        "chan_states",
        [
            *frames["chan_states"].to_dict(orient="records"),
            {"symbol": "999999.SZ", "dt": "2026-01-06", "regime": 1},
        ],
    )
    current = _clone_verified_snapshot(previous, replace_frame=("chan_states", appended_states))

    report = data_v21.verify_snapshot_extension(previous, current)

    assert report["schema"] == data_v21.SNAPSHOT_EXTENSION_SCHEMA
    assert report["valid"] is True
    assert report["row_counts"]["chan_states"] == {
        "previous": len(frames["chan_states"]),
        "current": len(frames["chan_states"]) + 1,
        "appended": 1,
    }
    assert report["appended_source_response_count"] == 0
    expected = hashlib.sha256(
        json.dumps(
            {key: value for key, value in report.items() if key != "verification_sha256"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert report["verification_sha256"] == expected


def test_snapshot_extension_rejects_row_mutation_deletion_and_time_rewind(tmp_path: Path) -> None:
    manifest_path, frames, _ = _build_snapshot(tmp_path)
    loaded, previous = data_v21.load_verified_snapshot(manifest_path)
    assert loaded.valid and previous is not None

    changed_records = frames["raw_daily"].to_dict(orient="records")
    changed_records[0]["close"] += 0.125
    changed = _clone_verified_snapshot(
        previous, replace_frame=("raw_daily", _typed_frame("raw_daily", changed_records))
    )
    changed_report = data_v21.verify_snapshot_extension(previous, changed)
    assert changed_report["valid"] is False
    assert "raw_daily modified old primary-key row" in changed_report["errors"][0]

    shortened = _clone_verified_snapshot(
        previous,
        replace_frame=("raw_daily", _typed_frame("raw_daily", frames["raw_daily"].to_dict(orient="records")[1:])),
    )
    shortened_report = data_v21.verify_snapshot_extension(previous, shortened)
    assert shortened_report["valid"] is False
    assert "raw_daily deleted old primary-key row" in shortened_report["errors"][0]

    rewind = _clone_verified_snapshot(previous, created_at_utc="2026-01-07T00:04:59Z")
    rewind_report = data_v21.verify_snapshot_extension(previous, rewind)
    assert rewind_report["valid"] is False
    assert "precedes the previous snapshot" in rewind_report["errors"][0]


def test_snapshot_extension_allows_rolling_response_set_but_rejects_digest_mutation(tmp_path: Path) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path)
    loaded, previous = data_v21.load_verified_snapshot(manifest_path)
    assert loaded.valid and previous is not None
    digest = previous.source_response_sha256[0]

    removed = _clone_verified_snapshot(previous, remove_source_sha256=digest)
    removed_report = data_v21.verify_snapshot_extension(previous, removed)
    assert removed_report["valid"] is True
    assert removed_report["retired_source_response_count"] == 1

    modified = _clone_verified_snapshot(previous)
    modified._responses[digest]["provider"] = "forged-provider"
    modified_report = data_v21.verify_snapshot_extension(previous, modified)
    assert modified_report["valid"] is False
    assert "source response is not content-addressed" in modified_report["errors"][0]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda manifest: manifest.update(contract_id="different-contract"), "contract identity/version mismatch"),
        (lambda manifest: manifest.update(protocol_sha256="f" * 64), "does not bind the authoritative V2.1 protocol"),
    ],
)
def test_snapshot_extension_rejects_contract_or_protocol_change(
    tmp_path: Path, mutation: Callable[[dict], None], error: str
) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path)
    loaded, previous = data_v21.load_verified_snapshot(manifest_path)
    assert loaded.valid and previous is not None
    current = _clone_verified_snapshot(previous, mutate_manifest=mutation)

    report = data_v21.verify_snapshot_extension(previous, current)

    assert report["valid"] is False
    assert error in report["errors"][0]


def test_valid_content_addressed_snapshot_and_replay_api(tmp_path: Path) -> None:
    manifest_path, frames, _ = _build_snapshot(tmp_path)

    report, snapshot = data_v21.load_verified_snapshot(manifest_path)

    assert report.valid is True
    assert report.data_ready is True
    assert all(report.gates.values())
    assert snapshot is not None
    assert snapshot.observation_through_session == "2026-01-06"
    assert snapshot.manifest_sha256 == manifest_path.stem
    assert snapshot.artifact_names == tuple(sorted(data_v21.REQUIRED_ARTIFACTS))
    assert snapshot.evidence_object_names == tuple(sorted(data_v21.REQUIRED_EVIDENCE_OBJECTS))
    assert snapshot.read_parquet("raw_daily")["close"].tolist() == frames["raw_daily"]["close"].tolist()
    assert snapshot.artifact_sha256("raw_daily") == report.evidence.get("unused", snapshot.artifact_sha256("raw_daily"))
    json.dumps(report.to_dict(), allow_nan=False)

    # Structural data validity is deliberately distinct from the expensive
    # formal 100×20 independent state-engine gate.
    state_verification = data_v21.verify_state_recompute(snapshot)
    assert state_verification["valid"] is False
    assert state_verification["source_symbol_count"] == 2
    assert state_verification["recomputed_symbol_count"] == 2
    assert state_verification["required_prefix_audit_symbol_count"] == 100
    assert state_verification["prefix_audit_symbol_count"] == 0
    assert state_verification["cutoffs_per_prefix_audit_symbol"] == 20
    assert state_verification["mismatch_count"] == 0
    assert "requires 100 symbols" in state_verification["errors"][0]


def test_state_recompute_rejects_tamper_outside_prefix_audit_sample() -> None:
    import xs_chan_state_cache as state_cache

    symbols = ("100001.SZ", "100002.SZ")
    dates = pd.bdate_range("2025-01-02", periods=state_cache.WARMUP_BARS + 20).normalize()
    raw_rows: list[dict] = []
    binding_rows: list[dict] = []
    source_frames: dict[str, pd.DataFrame] = {}
    for symbol_index, symbol in enumerate(symbols):
        source_rows: list[dict] = []
        for row_index, session in enumerate(dates):
            price = 10.0 + symbol_index + row_index / 100.0
            raw_rows.append(
                _source_fields(
                    ts_code=symbol,
                    trade_date=session,
                    open=price,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price + 0.1,
                    pre_close=price - 0.1,
                    vol=1_000.0 + row_index,
                    amount=10_000.0 + row_index,
                    pct_chg=0.1,
                )
            )
            row_token = data_v21.sha256_bytes(f"{symbol}:{session.date()}".encode())
            binding_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": session,
                    "official_status": "TRADING",
                    "raw_open": price,
                    "raw_high": price + 0.5,
                    "raw_low": price - 0.5,
                    "raw_close": price + 0.1,
                    "adj_factor": 1.0,
                    "adjusted_open": price,
                    "adjusted_high": price + 0.5,
                    "adjusted_low": price - 0.5,
                    "adjusted_close": price + 0.1,
                    "included_in_state_engine": True,
                    "raw_row_sha256": row_token,
                    "adj_factor_row_sha256": row_token,
                    "status_row_sha256": row_token,
                    data_v21.COMPUTED_AT: COMPUTED_AT,
                    "input_row_sha256": row_token,
                }
            )
            source_rows.append(
                {
                    "symbol": symbol,
                    "dt": session,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.1,
                    "vol": 1_000.0 + row_index,
                    "amount": 10_000.0 + row_index,
                    "pct_chg": 0.1,
                }
            )
        source_frames[symbol] = pd.DataFrame(source_rows)

    protocol_sha = data_v21.frozen_protocol_sha256()
    selected = min(symbols, key=lambda symbol: data_v21.sha256_bytes(f"{protocol_sha}:state-prefix:{symbol}".encode()))
    unsampled = next(symbol for symbol in symbols if symbol != selected)
    projections = {symbol: state_cache.generate_projection(frame) for symbol, frame in source_frames.items()}
    state_rows = pd.concat(list(projections.values()), ignore_index=True).to_dict(orient="records")
    tampered = next(row for row in state_rows if row["symbol"] == unsampled)
    tampered["regime"] = (int(tampered["regime"]) + 1) % 11

    audit = state_cache.audit_frame_prefix_invariance(source_frames[selected], checkpoints=20)
    audit_rows: list[dict] = []
    for comparison in audit["comparisons"]:
        mismatches = comparison["field_mismatches"]
        audit_rows.append(
            {
                "symbol": selected,
                "checkpoint_dt": comparison["cutoff"],
                "prefix_rows": comparison["prefix_rows"],
                "expected_rows": comparison["expected_rows"],
                "actual_rows": comparison["actual_rows"],
                "expected_sha256": comparison["expected_sha256"],
                "actual_sha256": comparison["actual_sha256"],
                "symbol_mismatches": mismatches["symbol"],
                "dt_mismatches": mismatches["dt"],
                "regime_mismatches": mismatches["regime"],
                "passed": comparison["passed"],
            }
        )

    observation = dates[-1].strftime("%Y-%m-%d")
    snapshot = data_v21.VerifiedDataSnapshot(
        manifest_sha256="a" * 64,
        created_at_utc=CREATED_AT,
        observation_through_session=observation,
        manifest={"observation_through_session": observation},
        artifact_blobs={
            "raw_daily": _parquet_bytes(_typed_frame("raw_daily", raw_rows)),
            "chan_states": _parquet_bytes(_typed_frame("chan_states", state_rows)),
            "state_audit": _parquet_bytes(_typed_frame("state_audit", audit_rows)),
        },
        evidence_blobs={
            "state_input_binding": _parquet_bytes(_typed_frame("state_input_binding", binding_rows)),
        },
        responses={},
    )

    verification = data_v21._verify_state_recompute(
        snapshot,
        prefix_audit_symbol_target=1,
        cutoffs_per_prefix_audit_symbol=20,
    )

    assert verification["valid"] is False
    assert verification["source_symbol_count"] == 2
    assert verification["recomputed_symbol_count"] == 2
    assert verification["prefix_audit_symbol_count"] == 1
    assert verification["cutoffs_per_prefix_audit_symbol"] == 20
    assert verification["mismatch_count"] == 1, verification
    assert any(f"full state projection differs for {unsampled}" in error for error in verification["errors"])


def test_zero_raw_response_requires_and_accepts_full_suspension(tmp_path: Path) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path, delisted_fully_suspended=True)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is True


def test_missing_ldp_raw_response_closes_formal_gate(tmp_path: Path) -> None:
    def remove_raw_response(responses: list[dict]) -> None:
        responses[:] = [
            response
            for response in responses
            if not (response["endpoint"] == "raw_daily" and response["scope"]["ts_code"] == DELISTED)
        ]

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_responses=remove_raw_response)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["universe_response_coverage"] is False
    assert "response_coverage" in _issue_codes(report)


def test_zero_raw_response_without_suspension_evidence_is_rejected(tmp_path: Path) -> None:
    def forge_zero(responses: list[dict]) -> None:
        response = next(
            item for item in responses if item["endpoint"] == "raw_daily" and item["scope"]["ts_code"] == DELISTED
        )
        response["kind"] = "ZERO"
        response["rows"] = []

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_responses=forge_zero)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["universe_response_coverage"] is False
    assert "raw_zero_unexplained" in _issue_codes(report)


def test_missing_namechange_negative_response_cannot_use_current_name(tmp_path: Path) -> None:
    def remove_negative_response(responses: list[dict]) -> None:
        responses[:] = [
            response
            for response in responses
            if not (response["endpoint"] == "namechange" and response["scope"]["ts_code"] == ACTIVE)
        ]

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_responses=remove_negative_response)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["namechange_response_evidence"] is False
    assert "namechange_missing_response" in _issue_codes(report)


@pytest.mark.parametrize("mode", ["active_terminal", "wrong_date", "self_target"])
def test_terminal_actions_are_exact_and_share_target_is_alive(tmp_path: Path, mode: str) -> None:
    def mutate(frames: dict[str, pd.DataFrame]) -> None:
        actions = frames["corporate_actions"].to_dict(orient="records")
        if mode == "active_terminal":
            actions.append(
                _source_fields(
                    action_id="action-2",
                    ts_code=ACTIVE,
                    record_date=None,
                    effective_date="2026-01-05",
                    payment_date=None,
                    action_type="delist_writeoff",
                    action_subtype=None,
                    gross_cash_per_share=None,
                    taxable_dividend_per_pre_action_share=None,
                    new_share_registration_date=None,
                    post_to_pre_ratio=None,
                    fractional_cash_price=None,
                    official_disposal_proceeds_per_entitled_share=None,
                    disposal_settlement_date="2026-01-05",
                    target_symbol=None,
                    target_share_ratio=None,
                    terminal_cash_per_share=None,
                    terminal_reason="official zero recovery",
                    no_value_evidence_sha256=data_v21.sha256_bytes(b"zero-recovery"),
                )
            )
        elif mode == "wrong_date":
            actions[0]["effective_date"] = pd.Timestamp("2026-01-05")
        else:
            actions[0]["target_symbol"] = DELISTED
        frames["corporate_actions"] = _typed_frame(
            "corporate_actions",
            sorted(actions, key=lambda row: row["action_id"]),
        )

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=mutate)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["terminal_actions_exact"] is False
    expected = "delist_share_target_lifecycle" if mode == "self_target" else "terminal_action_equality"
    assert expected in _issue_codes(report)


def test_unknown_delist_share_target_is_rejected(tmp_path: Path) -> None:
    def mutate(frames: dict[str, pd.DataFrame]) -> None:
        frames["corporate_actions"].loc[0, "target_symbol"] = "999999.SZ"

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=mutate)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert "delist_share_target" in _issue_codes(report)


@pytest.mark.parametrize(
    "required_field",
    ["disposal_settlement_date", "target_symbol", "target_share_ratio", "fractional_cash_price", "terminal_reason"],
)
def test_action_type_required_field_is_fail_closed(tmp_path: Path, required_field: str) -> None:
    def remove_required_field(frames: dict[str, pd.DataFrame]) -> None:
        frames["corporate_actions"].loc[0, required_field] = None

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=remove_required_field)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["artifact_semantics"] is False
    assert "corporate_action_applicability" in _issue_codes(report)


def test_corporate_action_physical_schema_missing_column_is_fail_closed(tmp_path: Path) -> None:
    frames = _base_source_frames()
    malformed = frames["corporate_actions"].drop(columns=["record_date"])
    manifest_path, _, _ = _build_snapshot(
        tmp_path, artifact_blob_override={"corporate_actions": _parquet_bytes(malformed)}
    )

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["artifact_schema"] is False
    assert "artifact_schema" in _issue_codes(report)


def test_corporate_action_without_exact_authority_evidence_is_rejected(tmp_path: Path) -> None:
    def remove_evidence(frames: dict[str, pd.DataFrame]) -> None:
        frames["corporate_action_evidence"] = _typed_frame("corporate_action_evidence", [])

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=remove_evidence)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["terminal_actions_exact"] is False
    assert "corporate_action_evidence_equality" in _issue_codes(report)


def test_raw_dependent_artifact_must_have_exact_key_coverage(tmp_path: Path) -> None:
    def drop_daily_basic_row(frames: dict[str, pd.DataFrame]) -> None:
        frames["daily_basic"] = frames["daily_basic"].iloc[:-1].reset_index(drop=True)

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=drop_daily_basic_row)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["universe_response_coverage"] is False
    assert "raw_dependent_coverage" in _issue_codes(report)


def test_reconciliation_digest_must_bind_archived_response(tmp_path: Path) -> None:
    manifest_path, _, manifest = _build_snapshot(tmp_path)
    root = manifest_path.parent.parent
    entry = manifest["artifacts"]["raw_daily_reconciliation"]
    reconciliation = pd.read_parquet(root / "objects" / "sha256" / entry["sha256"])
    reconciliation.loc[0, "response_sha256"] = "0" * 64
    new_path = _replace_parquet_entry(root, manifest, "artifacts", "raw_daily_reconciliation", reconciliation)

    report = data_v21.validate_data_manifest(new_path)

    assert report.valid is False
    assert report.gates["universe_response_coverage"] is False
    assert "raw_reconciliation_binding" in _issue_codes(report)


@pytest.mark.parametrize("column", ["adjusted_close", "status_row_sha256", "official_status"])
def test_state_input_binding_rejects_value_hash_and_status_mismatch(tmp_path: Path, column: str) -> None:
    manifest_path, _, manifest = _build_snapshot(tmp_path)
    root = manifest_path.parent.parent
    state_entry = manifest["evidence_objects"]["state_input_binding"]
    state = pd.read_parquet(root / "objects" / "sha256" / state_entry["sha256"])
    if column == "adjusted_close":
        state.loc[0, column] += 0.01
    elif column == "status_row_sha256":
        state.loc[0, column] = "0" * 64
    else:
        state.loc[0, column] = "SUSPENDED"
    new_path = _replace_parquet_entry(root, manifest, "evidence_objects", "state_input_binding", state)

    report = data_v21.validate_data_manifest(new_path)

    assert report.valid is False
    assert report.gates["state_input_row_binding"] is False


def test_reverse_source_time_order_is_closed_not_raised(tmp_path: Path) -> None:
    def reverse_time(responses: list[dict]) -> None:
        responses[0]["requested_at_utc"] = "2025-12-31T00:00:00Z"

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_responses=reverse_time)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["source_time_order"] is False
    assert report.gates["source_archive_binding"] is False
    assert "source_archive" in _issue_codes(report)


def test_malformed_parquet_dtype_is_closed_not_raised(tmp_path: Path) -> None:
    frames = _base_source_frames()
    malformed = frames["raw_daily"].copy()
    malformed["open"] = malformed["open"].astype("int64")
    manifest_path, _, _ = _build_snapshot(tmp_path, artifact_blob_override={"raw_daily": _parquet_bytes(malformed)})

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["artifact_schema"] is False
    assert "artifact_schema" in _issue_codes(report)


def test_malformed_semantics_is_closed_not_raised(tmp_path: Path) -> None:
    def mutate(frames: dict[str, pd.DataFrame]) -> None:
        frames["raw_daily"].loc[0, "high"] = -1.0

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_frames=mutate)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["artifact_semantics"] is False
    assert "raw_ohlc" in _issue_codes(report)


def test_formal_entry_rejects_loose_directory_and_non_addressed_manifest(tmp_path: Path) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path)

    loose_report = data_v21.validate_data_manifest(manifest_path.parent.parent)
    renamed = manifest_path.with_name(f"{'0' * 64}.json")
    renamed.write_bytes(manifest_path.read_bytes())
    renamed_report = data_v21.validate_data_manifest(renamed)

    assert loose_report.valid is False
    assert renamed_report.valid is False
    assert "formal_entry_closed" in _issue_codes(loose_report)
    assert "formal_entry_closed" in _issue_codes(renamed_report)


@pytest.mark.parametrize("mode", ["missing_artifact", "missing_evidence", "wrong_protocol", "wrong_provenance"])
def test_manifest_exact_sets_and_protocol_binding_are_fail_closed(tmp_path: Path, mode: str) -> None:
    def mutate(manifest: dict) -> None:
        if mode == "missing_artifact":
            manifest["artifacts"].pop("daily_basic")
        elif mode == "missing_evidence":
            manifest["evidence_objects"].pop("official_daily_status")
        elif mode == "wrong_protocol":
            manifest["protocol_sha256"] = "0" * 64
        else:
            manifest["artifacts"]["chan_states"]["authority"] = "unregistered-engine"

    manifest_path, _, _ = _build_snapshot(tmp_path, mutate_manifest=mutate)
    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert report.gates["content_addressed_manifest"] is False
    assert "formal_entry_closed" in _issue_codes(report)


def test_verified_snapshot_uses_held_bytes_after_object_replacement(tmp_path: Path) -> None:
    manifest_path, frames, manifest = _build_snapshot(tmp_path)
    report, snapshot = data_v21.load_verified_snapshot(manifest_path)
    assert report.valid and snapshot is not None
    raw_digest = manifest["artifacts"]["raw_daily"]["sha256"]
    raw_path = manifest_path.parent.parent / "objects" / "sha256" / raw_digest

    raw_path.write_bytes(b"changed-after-validation")

    assert snapshot.read_parquet("raw_daily")["close"].tolist() == frames["raw_daily"]["close"].tolist()
    assert data_v21.validate_data_manifest(manifest_path).valid is False


def test_symlinked_content_object_is_rejected(tmp_path: Path) -> None:
    manifest_path, _, manifest = _build_snapshot(tmp_path)
    root = manifest_path.parent.parent
    digest = manifest["artifacts"]["raw_daily"]["sha256"]
    object_path = root / "objects" / "sha256" / digest
    backup = root / "raw-backup"
    object_path.rename(backup)
    os.symlink(backup, object_path)

    report = data_v21.validate_data_manifest(manifest_path)

    assert report.valid is False
    assert "formal_entry_closed" in _issue_codes(report)


def test_report_and_snapshot_accessors_return_copies(tmp_path: Path) -> None:
    manifest_path, _, _ = _build_snapshot(tmp_path)
    report, snapshot = data_v21.load_verified_snapshot(manifest_path)
    assert report.valid and snapshot is not None

    manifest_copy = snapshot.manifest_dict()
    manifest_copy["contract_id"] = "mutated"
    response_copy = snapshot.read_source_response(snapshot.source_response_sha256[0])
    response_copy["provider"] = "mutated"

    assert snapshot.manifest_dict()["contract_id"] == data_v21.CONTRACT_ID
    assert snapshot.read_source_response(snapshot.source_response_sha256[0])["provider"] == "unit-test-official-source"


@pytest.mark.parametrize(
    ("auction_kind", "include_session_auction", "terminal_action"),
    [("ZERO", False, False), ("POSITIVE", True, False), ("POSITIVE", False, True)],
)
def test_execution_market_inputs_bind_every_source_row(
    auction_kind: str, include_session_auction: bool, terminal_action: bool
) -> None:
    snapshot, execution_input, _ = _market_binding_case(
        auction_kind=auction_kind,
        include_session_auction=include_session_auction,
        terminal_action=terminal_action,
    )

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input, deepcopy(execution_input)])

    assert report["schema"] == data_v21.EXECUTION_MARKET_BINDING_SCHEMA
    assert report["valid"] is True
    assert report["execution_input_count"] == 2
    sessions_per_input = 2 if terminal_action else 1
    assert report["official_session_count"] == sessions_per_input
    assert report["bound_open_row_count"] == 2 * sessions_per_input
    assert report["bound_eod_row_count"] == 2 * sessions_per_input
    assert report["bound_corporate_action_count"] == (2 if terminal_action else 0)
    assert report["verification_sha256"] == data_v21.sha256_bytes(
        data_v21.canonical_json_bytes({key: value for key, value in report.items() if key != "verification_sha256"})
    )


@pytest.mark.parametrize(
    ("mutate", "error_text"),
    [
        (
            lambda value: value["sessions"][0]["open_snapshot"][0].update(open=999.0),
            "open differs from the immutable data snapshot",
        ),
        (
            lambda value: value["sessions"][0].update(settlement_session="2099-01-01"),
            "settlement_session is not the official next trade session",
        ),
        (
            lambda value: value["sessions"][0]["eod_snapshot"][0].update(status="suspended"),
            "execution status differs from official status",
        ),
        (
            lambda value: value["sessions"][0]["open_snapshot"][0].update(source_row_sha256="0" * 64),
            "opening source digest is not snapshot-bound",
        ),
    ],
)
def test_execution_market_inputs_reject_self_reported_or_wrong_market_facts(
    mutate: Callable[[dict], None], error_text: str
) -> None:
    snapshot, execution_input, _ = _market_binding_case()
    mutate(execution_input)

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input])

    assert report["valid"] is False
    assert error_text in report["errors"][0]


def test_positive_open_auction_partition_cannot_mean_zero_for_an_omitted_session() -> None:
    snapshot, execution_input, _ = _market_binding_case(auction_kind="POSITIVE", include_session_auction=False)

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input])

    assert report["valid"] is False
    assert "POSITIVE open-auction response omits execution row" in report["errors"][0]


def test_execution_market_inputs_require_one_common_timeline_and_identical_overlap() -> None:
    snapshot, first, _ = _market_binding_case()
    second = deepcopy(first)
    second["trial_id"] = "another-arm"
    second["sessions"][0]["open_snapshot"][0]["open"] += 0.01

    report = data_v21.verify_execution_market_inputs(snapshot, [first, second])

    assert report["valid"] is False
    assert "differs from the immutable data snapshot" in report["errors"][0]


def test_execution_corporate_action_digest_binds_authority_evidence() -> None:
    snapshot, execution_input, _ = _market_binding_case(terminal_action=True)
    execution_input["sessions"][-1]["corporate_actions"][0]["source_row_sha256"] = data_v21.sha256_bytes(
        b"self-reported-action"
    )

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input])

    assert report["valid"] is False
    assert "source_row_sha256 differs from the immutable data snapshot" in report["errors"][0]


def test_anchor_session_uses_official_previous_session_then_requires_explicit_decision() -> None:
    snapshot, execution_input, frames = _market_binding_case()
    _prepend_anchor_session(snapshot, execution_input, frames)

    accepted = data_v21.verify_execution_market_inputs(snapshot, [execution_input])

    assert accepted["valid"] is True
    assert accepted["official_session_count"] == 2
    assert execution_input["sessions"][0]["decision"] is None
    assert execution_input["sessions"][1]["decision"] is not None

    execution_input["sessions"][1]["decision"] = None
    rejected = data_v21.verify_execution_market_inputs(snapshot, [execution_input])
    assert rejected["valid"] is False
    assert "must carry the first explicit post-anchor decision" in rejected["errors"][0]


def test_terminal_lifecycle_derives_delisted_zero_row_and_replays() -> None:
    snapshot, execution_input, _ = _market_binding_case(
        terminal_action=True, auction_kind="POSITIVE", include_session_auction=False
    )
    terminal_session = execution_input["sessions"][-1]

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input])
    replayed = execution_v21.replay_execution(execution_input)

    assert terminal_session["open_snapshot"][0]["status"] == "delisted"
    assert terminal_session["open_snapshot"][0]["open"] == 0
    assert terminal_session["open_snapshot"][0]["open_auction_turnover_cny"] == 0
    assert terminal_session["eod_snapshot"][0]["close"] == 0
    assert snapshot.read_source_response(snapshot.source_response_sha256[0])["kind"] == "POSITIVE"
    assert report["valid"] is True
    assert replayed["schema"] == execution_v21.EXECUTION_RESULT_SCHEMA

    terminal_session["open_snapshot"][0]["status"] = "trading"
    rejected = data_v21.verify_execution_market_inputs(snapshot, [execution_input])
    assert rejected["valid"] is False
    assert "execution status differs from official status" in rejected["errors"][0]


def test_entitlement_action_is_bound_and_emitted_on_record_before_effective_session() -> None:
    snapshot, execution_input, _ = _market_binding_case(cash_dividend=True)
    session = execution_input["sessions"][0]
    action = session["corporate_actions"][0]

    report = data_v21.verify_execution_market_inputs(snapshot, [execution_input])

    assert action["record_session"] == session["session"]
    assert action["record_session"] < action["effective_session"] < action["payment_session"]
    assert report["valid"] is True
    assert report["bound_corporate_action_count"] == 1

    # Moving the entitlement to its later effective date would omit the
    # record-date event and must close the formal binding gate.
    session["corporate_actions"] = []
    rejected = data_v21.verify_execution_market_inputs(snapshot, [execution_input])
    assert rejected["valid"] is False
    assert "corporate actions are not exact" in rejected["errors"][0]
