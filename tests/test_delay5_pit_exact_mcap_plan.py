"""Ratchets for the outcome-blind production delay5 PIT exact-mcap plan."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_pit_exact_mcap_plan as plan  # noqa: E402


def load_expected() -> dict:
    return json.loads(plan.EXPECTED_PLAN_PATH.read_text(encoding="utf-8"))


def test_canonical_payload_is_sorted_compact_and_stable() -> None:
    records = [["b", "2025-01-02"], ["a", "2025-01-01"]]
    payload = plan.canonical_payload(records)
    assert payload == b'[["a","2025-01-01"],["b","2025-01-02"]]'
    assert not payload.endswith(b"\n")
    assert plan.payload_identity(records)["sha256"] == hashlib.sha256(payload).hexdigest()


def test_tracked_request_and_treated_identities_are_frozen() -> None:
    manifest = load_expected()
    identities = manifest["request_identity"]
    expected = {
        "request": (47_431, "71f0543b5e64d44041fc27c1291630d64e7b7931855aea25ca205f0df7e6bdc9"),
        "dates": (170, "52476a7f768588037a5e5ab0b5d4339098ebb71d98cde6fdb743bfb1bb79079c"),
        "symbols": (4_089, "bdb4b3a943e1599fa4800556d96e919040c96cdcfe8e58ab2ca8c6b95a17493b"),
        "treated": (286, "57167f6739ec25e0569349802074771e1e7f5763f629d428708729c6c903e3b3"),
    }
    assert {key: (identities[key]["count"], identities[key]["sha256"]) for key in expected} == expected
    assert manifest["treated_common_support"]["trades_2024plus"] == 174
    assert identities["missing_frozen_industry"] == [{"symbol": "301308.SZ", "dec_dt": "2023-11-14"}]


def test_tracked_design_is_outcome_blind_and_fail_closed() -> None:
    manifest = load_expected()
    plan.validate_design_contract(manifest)
    design = manifest["design"]
    assert design["data_source"]["exact_mcap_cny"] == "circ_mv * 10000"
    assert design["matching"]["primary"] == {
        "name": "pit_exact_industry_mcap_k10_min5_c1p5",
        "industry": "same frozen annual industry; missing treated industry is unsupported",
        "metric": "nearest absolute log(exact point-in-time circ_mv CNY) distance",
        "caliper_ratio": 1.5,
        "k": 10,
        "min_controls": 5,
        "tie_break": "match_distance, symbol",
    }
    assert design["balance_gate"]["scopes"] == {"all": 286, "2024plus": 174}
    assert design["balance_gate"]["smd_thresholds"] == {
        "log_exact_mcap": 0.05,
        "all_other_features": 0.1,
    }
    assert design["balance_gate"]["coverage"]["treated_exact_mcap"] == "286/286"
    assert design["outcome_authorization"]["allowed_in_this_plan"] is False
    assert manifest["verdict"]["live_authorized"] is False


def test_identity_drift_is_detected() -> None:
    expected = load_expected()
    actual = copy.deepcopy(expected)
    actual["request_identity"]["request"]["sha256"] = "0" * 64
    checks = plan.identity_checks(actual, expected)
    assert checks["request_identity"] is False
    assert not all(checks.values())


def test_local_generated_artifact_sha_does_not_break_portable_plan_identity() -> None:
    expected = load_expected()
    actual = copy.deepcopy(expected)
    actual["source_identity"]["artifacts"]["common_audit"]["sha256"] = "0" * 64
    actual["source_identity"]["artifacts"]["exact_mcap_request_manifest"]["sha256"] = "1" * 64
    actual["source_identity"]["artifacts"]["treated_common_support"]["sha256"] = "2" * 64

    checks = plan.identity_checks(actual, expected)

    assert checks["portable_source_identity"] is True
    assert all(checks.values())


def test_portable_upstream_or_script_drift_still_fails() -> None:
    expected = load_expected()
    actual = copy.deepcopy(expected)
    actual["source_identity"]["reproducible_identity"]["panel_causal_projection"]["sha256"] = "0" * 64

    assert plan.identity_checks(actual, expected)["portable_source_identity"] is False


def test_canonical_frame_identity_is_row_order_and_writer_independent() -> None:
    frame = plan.pd.DataFrame(
        {
            "symbol": ["B", "A"],
            "dt": plan.pd.to_datetime(["2025-01-02", "2025-01-01"]),
            "close": [2.0, 1.0],
            "amount_e": [20.0, 10.0],
        }
    )

    first = plan.canonical_frame_identity(
        frame,
        columns=["symbol", "dt", "close", "amount_e"],
        sort_by=["symbol", "dt"],
        date_columns=["dt"],
    )
    second = plan.canonical_frame_identity(
        frame.iloc[::-1],
        columns=["symbol", "dt", "close", "amount_e"],
        sort_by=["symbol", "dt"],
        date_columns=["dt"],
    )

    assert first == second


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("att_results", {"h20": 1.0}), "result field"),
        (("balance_gate", None), "design contract"),
    ],
)
def test_outcome_result_or_missing_balance_gate_fails_closed(mutation: tuple[str, object], message: str) -> None:
    manifest = load_expected()
    field, value = mutation
    if field == "balance_gate":
        manifest["design"].pop(field)
    else:
        manifest[field] = value
    with pytest.raises(RuntimeError, match=message):
        plan.validate_design_contract(manifest)
