"""Production delay5 common-horizon ATT audit pure-function tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import delay5_common_horizon_att as audit  # noqa: E402


def _visible_frame(n: int = 12) -> pd.DataFrame:
    symbols = [f"S{i:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "close": np.full(n, 10.0),
            "amount_e": np.arange(1, n + 1, dtype=float),
            "industry": ["A"] * n,
            "mcap_proxy": np.arange(100, 100 + n, dtype=float),
            "is_st": [False] * n,
        }
    )


def test_horizon_endpoints_count_entry_as_day_one() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=65)
    endpoints = audit.horizon_endpoints(calendar, calendar[1])

    assert endpoints == {5: calendar[5], 20: calendar[20], 60: calendar[60]}


def test_common_maturity_uses_same_60_session_support() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=60)
    frame = pd.DataFrame({"entry_dt": [calendar[0], calendar[1]], "dec_dt": [calendar[0], calendar[0]]})

    mature, immature = audit.add_common_horizon_schedule(frame, calendar)

    assert mature.index.tolist() == [0]
    assert immature.index.tolist() == [1]
    assert mature.iloc[0]["h5_dt"] == calendar[4]
    assert mature.iloc[0]["h20_dt"] == calendar[19]
    assert mature.iloc[0]["h60_dt"] == calendar[59]


def test_signal_gate_locks_lte_volume_and_ignores_above_zg() -> None:
    frame = pd.DataFrame(
        {
            "sig_vol_ratio": [audit.tr.SURGE_GATE_VOL_RATIO, audit.tr.SURGE_GATE_VOL_RATIO + 0.001],
            "sig_ma_spread_pct": [audit.tr.SURGE_GATE_MA_SPREAD] * 2,
            "sig_ret20": [audit.tr.SURGE_GATE_RET20] * 2,
            "sig_above_zg": [0.0, 1.0],
        }
    )

    assert audit.signal_gate(frame).tolist() == [True, False]


def test_deciles_precede_exclusions_and_all_treated_are_removed() -> None:
    visible = _visible_frame()
    visible.loc[2, "is_st"] = True
    expected = np.floor(visible["amount_e"].rank(pct=True).mul(10).clip(upper=9.999)).astype(int)

    full, pool = audit.prepare_decision_pool(visible, treated_symbols={"S00", "S01"})

    assert full["amount_decile"].tolist() == expected.tolist()
    assert not {"S00", "S01", "S02"} & set(pool.index)


def test_nearest_match_tie_break_and_future_price_invariance() -> None:
    visible = _visible_frame(6)
    visible["mcap_proxy"] = [100.0, 100.0, 100.0, 110.0, 90.0, 130.0]
    full, pool = audit.prepare_decision_pool(visible, treated_symbols=set())
    pool["future_close"] = [1, 2, 3, 4, 5, 6]
    first = audit.nearest_controls(pool, metric="mcap_proxy", target=100.0, k=3)
    pool["future_close"] = [600, 500, 400, 300, 200, 100]
    second = audit.nearest_controls(pool, metric="mcap_proxy", target=100.0, k=3)

    assert first["symbol"].tolist() == ["S00", "S01", "S02"]
    assert second["symbol"].tolist() == first["symbol"].tolist()
    assert full.index.tolist() == visible["symbol"].tolist()


def test_amount_match_uses_treated_full_decile() -> None:
    visible = _visible_frame(20)
    full, pool = audit.prepare_decision_pool(visible, treated_symbols={"S19"})

    matched, eligible = audit.match_amount_controls(
        full,
        pool,
        treated_symbol="S19",
        treated_amount=20.0,
        k=5,
    )

    assert eligible == 2
    assert matched["symbol"].tolist() == ["S18", "S17"]


def test_proxy_match_fails_closed_without_industry() -> None:
    _, pool = audit.prepare_decision_pool(_visible_frame(), treated_symbols=set())

    matched, eligible = audit.match_proxy_controls(
        pool,
        treated_industry=np.nan,
        treated_mcap_proxy=105.0,
    )

    assert matched.empty
    assert eligible == 0


def test_exact_mcap_requests_always_include_treated_when_industry_is_missing() -> None:
    pool = pd.DataFrame(
        {"industry": ["A", "B"]},
        index=pd.Index(["CONTROL_A", "CONTROL_B"], name="symbol"),
    )
    requests: set[tuple[str, str]] = set()

    audit.add_exact_mcap_request_pairs(
        requests,
        treated_symbol="TREATED_MISSING",
        dec_dt=pd.Timestamp("2023-11-14"),
        treated_industry=np.nan,
        pool=pool,
    )
    audit.add_exact_mcap_request_pairs(
        requests,
        treated_symbol="TREATED_A",
        dec_dt=pd.Timestamp("2024-01-02"),
        treated_industry="A",
        pool=pool,
    )

    assert ("TREATED_MISSING", "2023-11-14") in requests
    assert ("TREATED_A", "2024-01-02") in requests
    assert ("CONTROL_A", "2024-01-02") in requests
    assert ("CONTROL_B", "2024-01-02") not in requests


def test_request_identity_is_order_invariant_for_pairs_and_scalars() -> None:
    pairs = [["B", "2024-01-02"], ["A", "2024-01-01"]]
    dates = ["2024-01-02", "2024-01-01"]

    assert audit._payload_identity(pairs) == audit._payload_identity(list(reversed(pairs)))
    assert audit._payload_identity(dates) == audit._payload_identity(list(reversed(dates)))


def test_control_no_open_and_gap_abandon_are_cash_without_cost() -> None:
    marks = {5: 110.0, 20: 110.0, 60: 110.0}
    terminal = {5: False, 20: False, 60: False}

    no_open = audit.evaluate_control_path(
        decision_close=100.0,
        entry_open=np.nan,
        horizon_marks=marks,
        terminal_by_horizon=terminal,
        limit_pct=10.0,
    )
    abandoned = audit.evaluate_control_path(
        decision_close=100.0,
        entry_open=109.8,
        horizon_marks=marks,
        terminal_by_horizon=terminal,
        limit_pct=10.0,
    )

    assert no_open["entry_status"] == "cash_no_open"
    assert abandoned["entry_status"] == "cash_gap_abandoned"
    for result in (no_open, abandoned):
        assert result["gross_h60_pct"] == 0.0
        assert result["net_h60_pct"] == 0.0


def test_control_locf_terminal_bounds_and_costs() -> None:
    result = audit.evaluate_control_path(
        decision_close=100.0,
        entry_open=100.0,
        horizon_marks={5: 110.0, 20: 105.0, 60: 90.0},
        terminal_by_horizon={5: False, 20: False, 60: True},
        limit_pct=10.0,
    )

    assert result["entry_status"] == "filled"
    assert result["gross_h5_pct"] == pytest.approx(10.0)
    assert result["gross_h60_pct"] == pytest.approx(-10.0)
    assert result["gross_pessimistic_h60_pct"] == -100.0
    assert result["gross_optimistic_h60_pct"] == pytest.approx(-10.0)
    assert result["net_h5_pct"] == pytest.approx(audit.net_return_pct(10.0, filled=True))


def test_net_cost_applies_only_to_filled_legs() -> None:
    expected = ((1.10 * (1 - audit.portfolio.SELL_COST) / (1 + audit.portfolio.BUY_COST)) - 1) * 100

    assert audit.net_return_pct(10.0, filled=True) == pytest.approx(expected)
    assert audit.net_return_pct(10.0, filled=False) == 0.0


def test_production_cohort_requires_exact_next_session_and_maturity() -> None:
    calendar = pd.bdate_range("2026-01-05", periods=61)
    candidate = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "mode": ["anticipate", "anticipate"],
            "delay": [5, 5],
            "sig_dt": [calendar[0], calendar[0]],
            "dec_dt": [calendar[0], calendar[0]],
            "entry_dt": [calendar[1], calendar[2]],
            "exit_dt": [calendar[10], calendar[10]],
            "sig_vol_ratio": [audit.tr.SURGE_GATE_VOL_RATIO] * 2,
            "sig_ma_spread_pct": [audit.tr.SURGE_GATE_MA_SPREAD] * 2,
            "sig_ret20": [audit.tr.SURGE_GATE_RET20] * 2,
            "amount_e": [2.0, 2.0],
            "sl_pct": [10.0, 10.0],
            "gap_pct": [0.0, 0.0],
            "limit_pct": [10.0, 10.0],
        }
    )
    market = pd.DataFrame(
        {
            "dt": calendar,
            "high20_ratio": [0.2] * len(calendar),
            "ew_index_above_ma20": [1.0] * len(calendar),
        }
    )

    mature, filled, funnel = audit.build_production_delay5_cohort(
        candidate,
        market,
        calendar,
        st_intervals={"DUMMY": []},
    )

    assert filled["symbol"].tolist() == ["000001.SZ"]
    assert mature["symbol"].tolist() == ["000001.SZ"]
    assert funnel["strict_next_market_session"]["rows"] == 1


def test_upstream_production_cohort_identity_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calendar = pd.bdate_range("2026-01-05", periods=61)
    candidate_path = tmp_path / "candidates.parquet"
    panel_path = tmp_path / "panel.parquet"
    market_path = tmp_path / "market.parquet"
    st_path = tmp_path / "namechange.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    upstream_path = tmp_path / "audit.json"
    candidate_path.write_bytes(b"candidate-v1")
    panel_path.write_bytes(b"panel-v1")
    market_path.write_bytes(b"market-v1")
    st_path.write_bytes(b"st-v1")
    cohort = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "sig_dt": [calendar[0]],
            "dec_dt": [calendar[0]],
            "entry_dt": [calendar[1]],
            "exit_dt": [calendar[10]],
            "state_fill_dt": [calendar[10]],
            "state_fill_open": [10.0],
            "state_fill_observed": pd.array([True], dtype="boolean"),
            **{f"stage_{name}": [True] for name in ["raw", "gate", "hard", "st", "market", "fill", "mature"]},
        }
    )
    cohort.to_parquet(cohort_path, index=False)
    upstream = {
        "schema": "surge_delay5_production_cohort_audit_v2",
        "inputs": {
            "candidates": {"sha256": audit.proxy.sha256_file(candidate_path)},
            "panel": {"sha256": audit.proxy.sha256_file(panel_path)},
            "market_state": {"sha256": audit.proxy.sha256_file(market_path)},
            "historical_st": {"sha256": audit.proxy.sha256_file(st_path)},
        },
        "outputs": {"cohort": {"sha256": audit.proxy.sha256_file(cohort_path), "rows": 1}},
    }
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
    monkeypatch.setattr(audit, "CANDIDATES_PATH", candidate_path)
    monkeypatch.setattr(audit, "PANEL_PATH", panel_path)
    monkeypatch.setattr(audit, "MARKET_PATH", market_path)
    monkeypatch.setattr(audit, "PRODUCTION_COHORT_PATH", cohort_path)
    monkeypatch.setattr(audit, "PRODUCTION_AUDIT_PATH", upstream_path)
    monkeypatch.setattr(audit.portfolio, "NAMECHANGE_PATH", st_path)

    mature, filled, _, _ = audit.load_production_common_support(calendar)

    assert len(mature) == len(filled) == 1
    panel_path.write_bytes(b"panel-mutated")
    with pytest.raises(RuntimeError, match="stale panel binding"):
        audit.load_production_common_support(calendar)
    panel_path.write_bytes(b"panel-v1")

    candidate_path.write_bytes(b"candidate-mutated")
    with pytest.raises(RuntimeError, match="stale candidates binding"):
        audit.load_production_common_support(calendar)
