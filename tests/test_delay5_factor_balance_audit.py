"""Delay5 因子余额审计的因果窗口、匹配、权重与 fail-closed 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_factor_balance_audit as audit  # noqa: E402


def test_decision_features_use_fixed_endpoints_and_ignore_future_prices() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=30)
    raw = pd.DataFrame({"000001.SZ": np.arange(1.0, 31.0)}, index=calendar)
    mark = raw.ffill()
    limitup = pd.DataFrame(False, index=calendar, columns=raw.columns)
    limitup.iloc[[20, 21, 23], 0] = True
    decision = calendar[24]

    before = audit.decision_feature_snapshot(raw, mark, limitup, decision)
    mutated = raw.copy()
    mutated.loc[calendar[25] :, "000001.SZ"] = 1_000_000.0
    after = audit.decision_feature_snapshot(mutated, mutated.ffill(), limitup, decision)

    assert before.loc["000001.SZ", "ret5"] == pytest.approx((25 / 20 - 1) * 100)
    assert before.loc["000001.SZ", "ret20"] == pytest.approx((25 / 5 - 1) * 100)
    assert before.loc["000001.SZ", "limitup_count20"] == 3
    assert before.loc["000001.SZ", "limitup_maxrun20"] == 2
    pd.testing.assert_series_equal(before.loc["000001.SZ"], after.loc["000001.SZ"])


def test_vol20_uses_locf_common_session_returns_without_backfill() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=21)
    raw = pd.DataFrame({"000001.SZ": [np.nan, *([10.0] * 19), np.nan]}, index=calendar)
    mark = raw.ffill()
    limitup = pd.DataFrame(False, index=calendar, columns=raw.columns)

    snapshot = audit.decision_feature_snapshot(raw, mark, limitup, calendar[-1])

    assert snapshot.loc["000001.SZ", "vol20"] == pytest.approx(0.0)
    assert pd.isna(snapshot.loc["000001.SZ", "log_price"])
    assert pd.isna(mark.iloc[0, 0])


def test_limitup_thresholds_are_historical_and_suspension_breaks_run() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=4)
    raw = pd.DataFrame(
        {
            "000001.SZ": [100.0, 105.0, np.nan, 110.25],
            "300001.SZ": [100.0, 110.0, 110.0, 132.0],
        },
        index=calendar,
    )
    mark = raw.ffill()
    st = {"000001.SZ": [("20260105", "99999999", True), ("20260108", "99999999", False)]}
    thresholds = {"000001.SZ": 9.8, "300001.SZ": 19.8}

    result = audit.build_limitup_matrix(raw, mark, st, base_limit_pct=thresholds)

    assert result["000001.SZ"].tolist() == [False, True, False, False]
    assert result["300001.SZ"].tolist() == [False, False, False, True]
    assert audit.longest_true_run(result.to_numpy()).tolist() == [1, 1]


def test_standardized_nearest_is_deterministic_and_exact_bins_are_enforced() -> None:
    pool = pd.DataFrame(
        {
            "symbol": ["000002.SZ", "000001.SZ", "000003.SZ"],
            "x": [-1.0, 1.0, 0.1],
            "limitup_count20": [1, 1, 2],
        }
    )
    target = pd.Series({"x": 0.0, "limitup_count20": 1})

    matched, eligible_n, zero_variance = audit.standardized_nearest_controls(
        pool,
        target,
        continuous=("x",),
        exact=("limitup_count20",),
        k=5,
    )

    assert eligible_n == 2
    assert zero_variance == []
    assert matched["symbol"].tolist() == ["000001.SZ", "000002.SZ"]


def test_zero_variance_features_do_not_create_nondeterministic_distance() -> None:
    pool = pd.DataFrame({"symbol": ["000002.SZ", "000001.SZ", "000003.SZ"], "x": [1.0, 1.0, 1.0]})
    target = pd.Series({"x": 1.0})

    matched, eligible_n, zero_variance = audit.standardized_nearest_controls(
        pool,
        target,
        continuous=("x",),
        k=2,
    )

    assert eligible_n == 3
    assert zero_variance == ["x"]
    assert matched["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
    assert matched["match_distance"].eq(0).all()


def test_balance_weights_each_treated_equally_when_control_counts_differ() -> None:
    pairs = pd.DataFrame(
        {
            "trade_id": ["T1", "T2", "T2"],
            "treated_ret5": [0.0, 2.0, 2.0],
            "control_ret5": [0.0, 1.0, 3.0],
        }
    )

    result = audit.balance_statistics(pairs, features=("ret5",))["ret5"]

    assert result["treated_mean"] == pytest.approx(1.0)
    assert result["control_mean"] == pytest.approx(1.0)
    assert result["smd"] == pytest.approx(0.0)


def test_scoped_balance_reports_all_and_2024plus_separately() -> None:
    pairs = pd.DataFrame(
        {
            "trade_id": ["old", "new"],
            "year": [2023, 2024],
            **{f"treated_{feature}": [0.0, 2.0] for feature in audit.BALANCE_FEATURES},
            **{f"control_{feature}": [0.0, 1.0] for feature in audit.BALANCE_FEATURES},
        }
    )

    result = audit.scoped_balance_statistics(pairs)

    assert set(result) == {"all", "2024plus"}
    assert result["all"]["ret5"]["treated_mean"] == pytest.approx(1.0)
    assert result["2024plus"]["ret5"]["treated_mean"] == pytest.approx(2.0)


def test_stale_sha_binding_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bound.bin"
    path.write_bytes(b"v1")
    expected = audit.sha256_file(path)
    path.write_bytes(b"v2")

    with pytest.raises(RuntimeError, match="stale test binding"):
        audit.verify_bound_file(path, expected, "test")
    with pytest.raises(RuntimeError, match="lacks sha256"):
        audit.verify_bound_file(path, None, "test")


def test_common_output_path_sha_and_rows_are_all_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {
        "treated_common_support": tmp_path / "treated.parquet",
        "matched_pairs": tmp_path / "pairs.parquet",
        "trade_att": tmp_path / "trade.parquet",
        "exact_mcap_request_manifest": tmp_path / "request.json",
    }
    for label, path in paths.items():
        path.write_bytes(label.encode())
    monkeypatch.setattr(audit, "COMMON_OUTPUT_PATHS", {"audit": tmp_path / "audit.json", **paths})
    treated = pd.DataFrame(index=range(2))
    pairs = pd.DataFrame(index=range(3))
    trade = pd.DataFrame(index=range(1))
    manifest = {"request_identity": {"count": 4}}
    rows = {"treated_common_support": 2, "matched_pairs": 3, "trade_att": 1, "exact_mcap_request_manifest": 4}
    common_audit = {
        "outputs": {
            label: {"path": str(path), "sha256": audit.sha256_file(path), "rows": rows[label]}
            for label, path in paths.items()
        }
    }

    verified = audit.validate_common_output_bindings(common_audit, treated, pairs, trade, manifest)

    assert set(verified) == set(paths)
    paths["matched_pairs"].write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="stale common output matched_pairs binding"):
        audit.validate_common_output_bindings(common_audit, treated, pairs, trade, manifest)


def test_support_min3_and_2024plus_denominator_are_fixed() -> None:
    attempts = pd.DataFrame(
        {
            "specification": [audit.SPEC_ALL, audit.SPEC_ALL, audit.SPEC_EXACT, audit.SPEC_EXACT],
            "year": [2023, 2024, 2024, 2025],
            "support_ok": [False, True, False, True],
            "selected_controls_n": [2, 3, 2, 5],
            "proxy_caliper_pool_n": [2, 3, 2, 5],
        }
    )
    pairs = pd.DataFrame(
        {
            "specification": [audit.SPEC_ALL] * 5 + [audit.SPEC_EXACT] * 7,
            "support_ok": [False, False, True, True, True, False, False, True, True, True, True, True],
        }
    )

    result = audit.support_summary(attempts, pairs)

    assert result[audit.SPEC_ALL]["all"]["support_rate_pct"] == 50.0
    assert result[audit.SPEC_ALL]["2024plus"]["support_rate_pct"] == 100.0
    assert result[audit.SPEC_EXACT]["2024plus"]["support_rate_pct"] == 50.0


@pytest.mark.parametrize("balanced", [True, False])
def test_exploratory_verdict_never_authorizes_live_trading(balanced: bool) -> None:
    smd = 0.01 if balanced else 0.2
    balance = {
        specification: {
            scope: {feature: {"smd": smd} for feature in audit.BALANCE_FEATURES} for scope in audit.BALANCE_SCOPES
        }
        for specification in audit.SPECIFICATIONS
    }
    verdict = audit.build_verdict(balance)

    assert verdict["edge_confirmed"] is False
    assert verdict["live_authorized"] is False
    assert "NOT_CONFIRMED" in verdict["status"]


def test_verdict_fails_closed_when_a_formal_balance_scope_is_missing() -> None:
    incomplete = {
        specification: {
            "all": {feature: {"smd": 0.0} for feature in audit.BALANCE_FEATURES},
        }
        for specification in audit.SPECIFICATIONS
    }

    with pytest.raises(RuntimeError, match="missing scope"):
        audit.build_verdict(incomplete)
