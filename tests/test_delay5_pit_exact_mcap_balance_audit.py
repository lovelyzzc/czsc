"""PIT exact-mcap 结果盲匹配、因果窗口、余额权重和门控测试。"""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_pit_exact_mcap_balance_audit as audit  # noqa: E402


def test_exact_mcap_caliper_is_inclusive_and_ties_break_by_symbol() -> None:
    target = 150.0
    pool = pd.DataFrame(
        {
            "symbol": ["000004.SZ", "000002.SZ", "000001.SZ", "000003.SZ", "000005.SZ"],
            "exact_circ_mv_cny": [225.0, 100.0, 225.0, 99.999, 225.001],
        }
    )

    matched, eligible_n = audit.nearest_exact_mcap_controls(pool, treated_mcap=target, k=3)

    assert eligible_n == 3
    assert matched["symbol"].tolist() == ["000002.SZ", "000001.SZ", "000004.SZ"]
    assert matched["exact_circ_mv_cny"].between(target / 1.5, target * 1.5, inclusive="both").all()


def test_exact_mcap_k_and_min_support_are_separate() -> None:
    pool = pd.DataFrame(
        {
            "symbol": [f"{value:06d}.SZ" for value in range(1, 15)],
            "exact_circ_mv_cny": np.linspace(90.0, 110.0, 14),
        }
    )
    matched, eligible_n = audit.nearest_exact_mcap_controls(pool, treated_mcap=100.0)
    assert eligible_n == 14
    assert len(matched) == audit.K == 10
    assert audit.MIN_CONTROLS == 5


def _small_match_inputs(*, missing_industry: bool = False) -> tuple:
    calendar = pd.bdate_range("2025-01-02", periods=70)
    symbols = ["T.SZ", *[f"C{value}.SZ" for value in range(1, 7)]]
    rows = []
    for index, date in enumerate(calendar):
        for offset, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "dt": date,
                    "close": 10.0 + index * 0.01 + offset * 0.001,
                    "amount_e": 100.0 + offset,
                }
            )
    panel = pd.DataFrame(rows)
    decision = calendar[-1]
    treated = pd.DataFrame(
        {"trade_id": [f"T.SZ|{decision.date()}"], "symbol": ["T.SZ"], "dec_dt": [decision], "year": [2025]}
    )
    all_filled = treated.loc[:, ["symbol", "dec_dt"]].copy()
    exact = pd.DataFrame(
        {
            "symbol": symbols,
            "dec_dt": decision,
            "exact_circ_mv_cny": [100.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0],
        }
    )
    industry = pd.Series("A", index=symbols)
    if missing_industry:
        industry.loc["T.SZ"] = np.nan
    return treated, all_filled, audit.CausalFeatureStore.from_panel(panel, calendar), exact, {2025: industry}


def test_missing_industry_treated_is_an_explicit_unsupported_attempt() -> None:
    treated, all_filled, store, exact, industries = _small_match_inputs(missing_industry=True)

    pairs, attempts = audit.build_exact_matches(
        treated,
        all_filled,
        feature_store=store,
        exact_mcap=exact,
        industry_maps=industries,
        st_intervals={},
    )

    assert pairs.empty
    assert len(attempts) == 1
    assert not bool(attempts.loc[0, "support_ok"])
    assert attempts.loc[0, "unsupported_reason"] == "missing_treated_industry"


def test_future_price_mutation_does_not_change_pair_identity_hash() -> None:
    treated, all_filled, store, exact, industries = _small_match_inputs()
    pairs_before, _ = audit.build_exact_matches(
        treated,
        all_filled,
        feature_store=store,
        exact_mcap=exact,
        industry_maps=industries,
        st_intervals={},
    )
    decision = treated.loc[0, "dec_dt"]
    future = store.panel.iloc[:7].copy()
    future["dt"] = decision + pd.Timedelta(days=1)
    future["close"] = 1_000_000_000.0
    future["amount_e"] = 1_000_000_000.0
    mutated_store = audit.CausalFeatureStore.from_panel(
        pd.concat([store.panel, future], ignore_index=True),
        [*store.calendar, decision + pd.Timedelta(days=1)],
    )
    pairs_after, _ = audit.build_exact_matches(
        treated,
        all_filled,
        feature_store=mutated_store,
        exact_mcap=exact,
        industry_maps=industries,
        st_intervals={},
    )
    columns = ["trade_id", "control_symbol", "dec_dt", "rank", "match_distance"]
    assert audit.frame_identity(pairs_before, columns) == audit.frame_identity(pairs_after, columns)


def test_balance_weights_each_treated_one_and_controls_one_over_k() -> None:
    pairs = pd.DataFrame(
        {
            "trade_id": ["T1", "T2", "T2"],
            "treated_x": [0.0, 2.0, 2.0],
            "control_x": [0.0, 1.0, 3.0],
        }
    )

    result, missing = audit.balance_statistics(pairs, ["x"])

    assert missing == []
    assert result["x"]["treated_mean"] == pytest.approx(1.0)
    assert result["x"]["control_mean"] == pytest.approx(1.0)
    assert result["x"]["smd"] == pytest.approx(0.0)


def test_balance_nan_fails_closed_instead_of_row_dropping() -> None:
    pairs = pd.DataFrame({"trade_id": ["T1"], "treated_x": [np.nan], "control_x": [1.0]})
    result, missing = audit.balance_statistics(pairs, ["x"])
    assert result == {}
    assert missing == ["x"]


def test_coverage_denominators_and_verdict_are_fixed() -> None:
    attempts = pd.DataFrame(
        {
            "year": [2023, 2024, 2024, 2025, 2025],
            "treated_exact_mcap_available": [True, True, True, True, False],
            "attempt_eligible": [True, True, True, True, False],
            "support_ok": [True, True, False, True, False],
        }
    )
    coverage = audit.coverage_summary(attempts)
    assert coverage["all"]["D_source"] == 5
    assert coverage["all"]["D_attempt"] == 4
    assert coverage["2024plus"]["D_source"] == 4
    assert coverage["2024plus"]["collector_coverage"] == pytest.approx(0.75)
    assert coverage["2024plus"]["support_rate"] == pytest.approx(0.5)
    assert coverage["2024plus"]["source_support_rate"] == pytest.approx(0.5)
    assert coverage["2024plus"]["attempt_support_rate"] == pytest.approx(2 / 3)
    assert set(coverage["by_year"]) == {"2023", "2024", "2025"}

    balanced = {
        scope: {
            "statistics": {"log_exact_mcap": {"passes": True}},
            "missing_features": [],
        }
        for scope in audit.BALANCE_SCOPES
    }
    coverage["all"]["D_supported"] = 4
    coverage["all"]["D_source"] = 5
    coverage["2024plus"]["D_supported"] = 3
    coverage["2024plus"]["D_source"] = 4
    verdict = audit.build_verdict(coverage, balanced, required_features=["log_exact_mcap"])
    assert verdict["status"] == "COLLECTOR_COVERAGE_INSUFFICIENT_OUTCOMES_NOT_EVALUATED"
    assert verdict["outcomes_loaded"] is False
    assert verdict["live_authorized"] is False


def test_balance_gate_uses_stricter_log_exact_mcap_threshold() -> None:
    stats = {
        scope: {
            "statistics": {
                "log_exact_mcap": {"passes": False, "smd": 0.07},
                "ret5": {"passes": True, "smd": 0.07},
            },
            "missing_features": [],
        }
        for scope in audit.BALANCE_SCOPES
    }
    coverage = {
        scope: {
            "collector_coverage": 1.0,
            "attempt_coverage": 1.0,
            "support_rate": 1.0,
            "D_supported": 10,
            "D_source": 10,
        }
        for scope in audit.BALANCE_SCOPES
    }
    verdict = audit.build_verdict(coverage, stats, required_features=["log_exact_mcap", "ret5"])
    assert verdict["status"] == "BALANCE_INSUFFICIENT_OUTCOMES_NOT_EVALUATED"
    assert verdict["failed_balance_features"]["all"] == ["log_exact_mcap"]


def test_pairs_attempts_reject_future_outcome_columns() -> None:
    with pytest.raises(RuntimeError, match="future outcome columns"):
        audit.assert_outcome_blind_columns(pd.DataFrame({"gross_h20_pct": [1.0]}), "pairs")


def test_unbound_exact_mcap_override_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manifest_path = root / "manifests" / "final.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": audit.collector.FINAL_SCHEMA,
                "materialized": {
                    "path": "materialized/bound.parquet",
                    "sha256": "0" * 64,
                    "rows": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    verification = {
        "artifacts": {
            "manifest": "manifests/final.json",
            "materialized": "materialized/bound.parquet",
        }
    }

    with pytest.raises(RuntimeError, match="unbound exact_mcap_path override"):
        audit.load_bound_collection(
            cache_root=root,
            verification=verification,
            exact_mcap_path=root / "materialized" / "unbound.parquet",
        )


def test_reproduced_plan_upstream_drift_fails_closed() -> None:
    fields = (
        "schema",
        "canonicalization",
        "source_identity",
        "request_identity",
        "treated_common_support",
        "design",
        "verdict",
    )
    expected = {field: {"frozen": field} for field in fields}
    expected["source_identity"] = {
        "baseline_parent_commit": "a" * 40,
        "scripts": {},
        "artifacts": {},
        "industry_snapshots": {},
    }
    reproduced = copy.deepcopy(expected)
    reproduced["source_identity"]["baseline_parent_commit"] = "b" * 40

    with pytest.raises(RuntimeError, match="source_identity"):
        audit.validate_reproduced_plan(expected, reproduced)


def test_independently_rebuilt_production_identities_are_bound_to_plan() -> None:
    all_filled = pd.DataFrame(
        {
            "symbol": ["A.SZ", "B.SZ"],
            "dec_dt": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "entry_dt": pd.to_datetime(["2025-01-03", "2025-01-06"]),
        }
    )
    treated = all_filled.iloc[[0]].copy()
    treated["h5_dt"] = pd.Timestamp("2025-01-09")
    treated["h20_dt"] = pd.Timestamp("2025-01-30")
    treated["h60_dt"] = pd.Timestamp("2025-03-27")
    expected = {
        "all_filled_identity": audit.pit_plan.payload_identity(
            [["A.SZ", "2025-01-02", "2025-01-03"], ["B.SZ", "2025-01-03", "2025-01-06"]]
        ),
        "common60_schedule_identity": audit.pit_plan.payload_identity(
            [["A.SZ", "2025-01-02", "2025-01-03", "2025-01-09", "2025-01-30", "2025-03-27"]]
        ),
    }
    plan = {"source_identity": {"reproducible_identity": {"production": expected}}}

    assert audit.validate_rebuilt_production_identities(treated, all_filled, plan) == expected
    drifted = copy.deepcopy(plan)
    drifted["source_identity"]["reproducible_identity"]["production"]["all_filled_identity"]["count"] = 3
    with pytest.raises(RuntimeError, match="all_filled_identity"):
        audit.validate_rebuilt_production_identities(treated, all_filled, drifted)


def test_output_records_use_repo_relative_locator() -> None:
    record = audit.output_record(audit.BALANCE_SCRIPT_PATH)

    assert record["locator"] == "scripts/delay5_pit_exact_mcap_balance_audit.py"
    assert not Path(record["locator"]).is_absolute()
    assert "path" not in record


def test_cross_provider_qa_checks_units_without_changing_matching() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    exact = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "dec_dt": dates,
            "exact_circ_mv_cny": [100_000_000.0, 202_000_000.0],
        }
    )
    baostock = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "dec_dt": dates,
            "exact_circ_mv": [100_000_000.0, 200_000_000.0],
        }
    )

    result = audit.cross_provider_unit_qa(exact, baostock)

    assert result["overlap_rows"] == 2
    assert result["ratio_median"] == pytest.approx(1.005)
    assert result["absolute_deviation_pct"]["max"] == pytest.approx(1.0)
    assert result["absolute_deviation_pct"]["gt_5pct"] == 0
    assert "never used for matching" in result["role"]


def test_source_ratchet_has_no_outcome_loader_or_att_builder() -> None:
    source = inspect.getsource(audit)
    forbidden = ("add_treated_outcomes", "build_trade_att", "treated_common_support.parquet")
    assert not any(token in source for token in forbidden)
    assert 'columns=["symbol", "dec_dt", "entry_dt", "stage_fill"]' in source
    assert '"outcomes_loaded": False' in source
