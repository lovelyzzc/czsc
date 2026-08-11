"""S2b 尾部机制审计纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_tail_mechanism_audit as audit  # noqa: E402


def test_hypergeom_tail_matches_trail18_enrichment() -> None:
    probability = audit.hypergeom_tail(total=117, successes=18, draws=5, observed=3)
    assert probability == pytest.approx(0.0254845169, abs=1e-10)


def test_nominal_gate_requires_both_thresholds() -> None:
    passed = {"daily_hac": {"t_stat": 2.1}, "daily_block_bootstrap": {"ci_lo": 0.01}}
    assert audit.nominal_gate(passed)
    assert not audit.nominal_gate({**passed, "daily_hac": {"t_stat": 1.99}})
    assert not audit.nominal_gate({**passed, "daily_block_bootstrap": {"ci_lo": 0.0}})


def test_classify_descriptive_breadth_regime_uses_median_and_breadth() -> None:
    frame = pd.DataFrame(
        {
            "mkt_ret20_median": [0.01, -0.01, 0.01],
            "mkt_ret20_pos_ratio": [0.60, 0.40, 0.50],
        }
    )
    assert audit.classify_descriptive_breadth_regime(frame).tolist() == ["bull", "bear", "sideways"]


def test_legacy_hard_filter_masks_keep_gap_separate() -> None:
    frame = pd.DataFrame(
        {
            "amount_e": [1.0, 0.9, 1.2],
            "sl_pct": [8.0, 10.0, 21.0],
            "gap_pct": [9.7, 1.0, 1.0],
            "limit_pct": [10.0, 10.0, 10.0],
        }
    )
    st_mask = pd.Series([False, False, False])
    masks = audit.legacy_hard_filter_masks(frame, st_mask)
    assert masks["current_hard_without_gap"].tolist() == [True, False, False]
    assert masks["legacy_hard_with_gap"].tolist() == [False, False, False]


def test_group_breakdown_drops_global_top5_within_each_group() -> None:
    frame = pd.DataFrame(
        {
            "bucket": ["a", "a", "b"],
            "exact_excess_pct": [10.0, 2.0, -1.0],
            "is_global_top5": [True, False, False],
        }
    )
    rows = {row["group"]: row for row in audit.group_breakdown(frame, "bucket")}
    assert rows["a"]["full"]["mean_pct"] == 6.0
    assert rows["a"]["after_drop_global_top5"]["mean_pct"] == 2.0


def test_validate_annual_reference_binds_schema_sha_rows_year_and_status() -> None:
    annual = {
        "schema": "s2b_annual_exact_mcap_support_audit_v2",
        "design": {"year": 2025},
        "trade_output": {"sha256": "abc", "rows": 117},
        "coverage": {"exact_supported_trades": 117},
        "verdict": {"status": "2025_EXACT_SUPPORT_EDGE_REMAINS_TAIL_DEPENDENT"},
    }
    assert (
        audit.validate_annual_reference(annual, trade_sha256="abc", trade_rows=117)
        == "2025_EXACT_SUPPORT_EDGE_REMAINS_TAIL_DEPENDENT"
    )
    with pytest.raises(RuntimeError, match="sha256"):
        audit.validate_annual_reference(annual, trade_sha256="different", trade_rows=117)
    with pytest.raises(RuntimeError, match="rows mismatch"):
        audit.validate_annual_reference(annual, trade_sha256="abc", trade_rows=116)


def test_mechanism_label_only_describes_checked_cross_month_h2_structure() -> None:
    label = audit.derive_mechanism_label(
        tail_dependent=True,
        all_month_lomo_nominal=True,
        distinct_top_months=5,
        h2_global_top_count=4,
        h2_after_global_top_mean=-0.072,
    )
    assert label == "2025_CROSS_MONTH_SPARSE_TAIL_WITH_H2_COLLAPSE"
    assert (
        audit.derive_mechanism_label(
            tail_dependent=True,
            all_month_lomo_nominal=True,
            distinct_top_months=4,
            h2_global_top_count=4,
            h2_after_global_top_mean=-0.072,
        )
        == "2025_TAIL_STRUCTURE_REQUIRES_REVIEW"
    )


def test_market_feature_diagnostics_exposes_common_background() -> None:
    trades = pd.DataFrame(
        {
            "small_minus_large_ret20": [-0.1, -0.2, -0.3],
            "ew_index_above_ma20": [1.0, 1.0, 0.0],
            "mkt_ret20_pos_ratio": [0.6, 0.4, 0.7],
            "high20_ratio": [0.2, 0.1, 0.3],
            "mkt_ret20_median": [0.01, -0.01, 0.02],
        }
    )
    result = audit.market_feature_diagnostics(trades, trades.head(2))
    assert result["small_minus_large_ret20_negative"]["all_count"] == 3
    assert result["small_minus_large_ret20_negative"]["top5_count"] == 2
    assert result["small_minus_large_ret20_negative"]["descriptive_probability_all_top5_given_margin"] == 1.0


def test_stable_top_n_uses_identity_keys_for_ties() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["B.SZ", "A.SZ", "C.SZ", "A.SZ"],
            "dec_dt": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"]),
            "exact_excess_pct": [5.0, 5.0, 6.0, 5.0],
        }
    )
    top = audit.stable_top_n(frame, 3)
    assert list(zip(top["symbol"], top["dec_dt"].dt.strftime("%Y-%m-%d"), strict=True)) == [
        ("C.SZ", "2025-01-03"),
        ("A.SZ", "2025-01-01"),
        ("A.SZ", "2025-01-02"),
    ]


def test_own_tail_concentration_drops_own_top3_and_top5(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "symbol": list("GFABCDE"),
            "exact_excess_pct": [-1.0, 1.0, 10.0, 8.0, 6.0, 4.0, 2.0],
        }
    )
    monkeypatch.setattr(audit, "inference_summary", lambda scoped: {"n_trades": len(scoped)})
    result = audit.own_tail_concentration(frame)
    assert result["drop_own_top3"]["descriptive"] == {
        "n": 4,
        "mean_pct": 1.5,
        "median_pct": 1.5,
        "positive_pct": 75.0,
        "total_pct": 6.0,
    }
    assert result["drop_own_top3"]["removed_share_of_positive_excess_pct"] == 77.42
    assert result["drop_own_top5"]["descriptive"]["n"] == 2
    assert result["drop_own_top5"]["descriptive"]["total_pct"] == 0.0
    assert result["drop_own_top5"]["removed_share_of_positive_excess_pct"] == 96.77


def test_leave_one_period_out_excludes_each_period(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "month": ["2025-02", "2025-01", "2025-02", "2025-03"],
            "exact_excess_pct": [2.0, 1.0, 3.0, 4.0],
        }
    )
    monkeypatch.setattr(
        audit,
        "inference_summary",
        lambda scoped: {"n_trades": len(scoped), "total_pct": float(scoped["exact_excess_pct"].sum())},
    )
    monkeypatch.setattr(audit, "nominal_gate", lambda summary: summary["n_trades"] >= 3)
    rows = audit.leave_one_period_out(frame, "month")
    assert [row["excluded"] for row in rows] == ["2025-01", "2025-02", "2025-03"]
    assert [row["summary"] for row in rows] == [
        {"n_trades": 3, "total_pct": 9.0},
        {"n_trades": 2, "total_pct": 5.0},
        {"n_trades": 3, "total_pct": 6.0},
    ]
    assert [row["nominal_gate"] for row in rows] == [True, False, True]
