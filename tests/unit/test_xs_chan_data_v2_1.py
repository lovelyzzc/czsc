from __future__ import annotations

# ruff: noqa: E402, I001

import io
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_data_v2_1 as data_v21


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


def _response(endpoint: str, symbol: str | None, frame: pd.DataFrame) -> dict:
    rows = [data_v21.canonical_source_row(endpoint, row) for row in frame.to_dict(orient="records")]
    return {
        "schema_version": data_v21.SOURCE_RESPONSE_SCHEMA,
        "provider": "unit-test-official-source",
        "endpoint": endpoint,
        "scope": {"ts_code": symbol, "start_date": "2026-01-02", "end_date": "2026-01-06"},
        "source_asof_utc": SOURCE_ASOF,
        "requested_at_utc": REQUESTED_AT,
        "responded_at_utc": RESPONDED_AT,
        "ingested_at_utc": INGESTED_AT,
        "kind": "POSITIVE" if rows else "ZERO",
        "rows": rows,
    }


def _responses(frames: dict[str, pd.DataFrame]) -> list[dict]:
    result = [_response(endpoint, None, frames[endpoint]) for endpoint in sorted(data_v21.GLOBAL_ENDPOINTS)]
    for endpoint in sorted(data_v21.SYMBOL_ENDPOINTS):
        artifact = frames[endpoint]
        for symbol in (ACTIVE, DELISTED):
            subset = artifact.loc[artifact["ts_code"].eq(symbol)].reset_index(drop=True)
            result.append(_response(endpoint, symbol, subset))
    return result


def _response_map(responses: list[dict]) -> dict[tuple[str, str | None], tuple[str, dict]]:
    return {
        (response["endpoint"], response["scope"]["ts_code"]): (
            data_v21.sha256_bytes(data_v21.canonical_json_bytes(response)),
            response,
        )
        for response in responses
    }


def _add_reconciliation_frames(frames: dict[str, pd.DataFrame], responses: list[dict]) -> None:
    response_map = _response_map(responses)
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
                    requested_start="2026-01-02",
                    requested_end="2026-01-06",
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
                requested_start="2026-01-02",
                requested_end="2026-01-06",
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
    responses = _responses(frames)
    _add_reconciliation_frames(frames, responses)
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


def _issue_codes(report: data_v21.DataValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


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


def test_valid_content_addressed_snapshot_and_replay_api(tmp_path: Path) -> None:
    manifest_path, frames, _ = _build_snapshot(tmp_path)

    report, snapshot = data_v21.load_verified_snapshot(manifest_path)

    assert report.valid is True
    assert report.data_ready is True
    assert all(report.gates.values())
    assert snapshot is not None
    assert snapshot.manifest_sha256 == manifest_path.stem
    assert snapshot.artifact_names == tuple(sorted(data_v21.REQUIRED_ARTIFACTS))
    assert snapshot.evidence_object_names == tuple(sorted(data_v21.REQUIRED_EVIDENCE_OBJECTS))
    assert snapshot.read_parquet("raw_daily")["close"].tolist() == frames["raw_daily"]["close"].tolist()
    assert snapshot.artifact_sha256("raw_daily") == report.evidence.get("unused", snapshot.artifact_sha256("raw_daily"))
    json.dumps(report.to_dict(), allow_nan=False)


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
