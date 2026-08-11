"""S2b 全行业 exact-mcap 规划身份纯函数测试。"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_full_exact_match_plan as plan  # noqa: E402


def test_canonical_payload_is_sorted_compact_utf8_without_newline() -> None:
    records = [["b", "2025-01-02"], ["a", "2025-01-01"]]
    payload = plan.canonical_payload(records)
    assert payload == b'[["a","2025-01-01"],["b","2025-01-02"]]'
    assert not payload.endswith(b"\n")
    identity = plan.payload_identity(records)
    assert identity["sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["first3"][0][0] == "a"


def test_tracked_manifest_freezes_expected_request_identities() -> None:
    manifest = json.loads(plan.EXPECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["cohort"]["capacity_entry_keys"]["count"] == 329
    assert manifest["cohort"]["capacity_entry_keys"]["sha256"] == (
        "78f24cccbc71a47618163c689dc9d1223a9fa3f57b3729ad5eb57dc795ac6142"
    )
    assert manifest["cohort"]["full_trade_records"]["count"] == 329
    assert manifest["cohort"]["full_trade_records"]["sha256"] == (
        "30c057c04d6c34418700ea0ce67267e4f9ee69b541f6fe75dffe74a1e65fc9b3"
    )
    assert manifest["cohort"]["decision_dates"]["count"] == 219
    assert manifest["cohort"]["decision_dates"]["sha256"] == (
        "1a19000783e686f83a37846c26f06da88bd127c24c637e68b06fed8b4ecfabf4"
    )

    expected_universes = {
        "legacy_future_valid": (
            60_028,
            "8a8643abb3fbfe4e459e3fece879d4dc4f01585b48bfff42d49fe16ac569674e",
        ),
        "decision_date_only": (
            60_102,
            "a1f01f61efc127e3878eac2c686ed67f929a2a8e8bc9735060819ddf2fd30296",
        ),
        "annual_industry_full": (
            60_900,
            "b7255e7ab2a53c132b856b91793f7d1a93184dc6ffcd5c8c5e1df1a2f8983253",
        ),
    }
    for name, (count, sha256) in expected_universes.items():
        assert manifest["request_universes"][name]["count"] == count
        assert manifest["request_universes"][name]["sha256"] == sha256

    expected_industry_sha = {
        "2024": "4b4d33bd78215aa808a2c21128a9266e8247157049fd3877f5975f555920e231",
        "2025": "db11f8c6ac3789bbcd19d108296ff8c520754af4648bee19ac55190e49b1d1fa",
        "2026": "a829f1aa817514f9b09a07800dc5b85d4eb6717d3f6f09f4e64f2ee30d8b2f58",
    }
    assert {
        year: snapshot["canonical_content_sha256"] for year, snapshot in manifest["industry_snapshots"].items()
    } == expected_industry_sha

    fetch_plan = manifest["fetch_plan"]
    assert fetch_plan["git_only_required_calls"] == 219
    assert fetch_plan["calls_if_two_named_ignored_objects_are_copied"] == 217
    assert fetch_plan["preferred_request_universe"] == "decision_date_only"
    assert fetch_plan["full_exact_matching_implemented"] is False
    assert manifest["local_ignored_artifacts"]["git_portable"] is False


def test_identity_checks_require_schema_canonicalization_and_fetch_plan() -> None:
    expected = json.loads(plan.EXPECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
    mutations = {
        "schema": "drifted-schema",
        "canonicalization": "drifted-canonicalization",
        "fetch_plan": {**expected["fetch_plan"], "git_only_required_calls": 218},
    }
    for field, value in mutations.items():
        actual = copy.deepcopy(expected)
        actual[field] = value
        checks = plan._identity_checks(actual, expected)
        assert checks[field] is False
