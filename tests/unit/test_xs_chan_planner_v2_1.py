from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_planner_v2_1 as planner  # noqa: E402


def _frames(symbol_count: int = 100) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    sessions = pd.bdate_range("2025-06-02", periods=281)
    decision, execution = [
        (left, right)
        for left, right in zip(sessions[:-1], sessions[1:], strict=True)
        if left.isocalendar()[:2] != right.isocalendar()[:2]
    ][-1]
    sessions = sessions[sessions <= execution]
    symbols = [f"S{index:03d}" for index in range(symbol_count)]
    calendar = pd.DataFrame({"trade_date": sessions, "is_open": True})
    master = pd.DataFrame(
        {
            "ts_code": symbols,
            "name": symbols,
            "board": "MAIN",
            "list_status": "L",
            "list_date": sessions[0],
            "delist_date": pd.NaT,
        }
    )
    clock = pd.MultiIndex.from_product([symbols, sessions], names=["ts_code", "trade_date"]).to_frame(index=False)
    symbol_index = clock["ts_code"].str[1:].astype(int)
    session_index = clock.groupby("ts_code", sort=False).cumcount()
    close = 10.0 + symbol_index * 0.015 + session_index * 0.002 + np.sin(session_index / 9 + symbol_index) * 0.12
    raw = clock.assign(close=close, vol=100_000.0, amount=100_000.0 + symbol_index * 100.0)
    factors = clock.assign(adj_factor=np.where(session_index < 180, 1.0, 1.1 + symbol_index * 0.0001))
    basic = clock.assign(free_share=100_000.0 + symbol_index * 200.0)
    status = clock.assign(official_status="TRADING")
    states = clock.rename(columns={"ts_code": "symbol", "trade_date": "dt"}).assign(
        regime=(symbol_index % 11).astype(int)
    )[["symbol", "dt", "regime"]]
    industries = pd.DataFrame(
        {
            "ts_code": symbols,
            "industry_code": [f"I{index % 5}" for index in range(symbol_count)],
            "classification_version": "SW2021",
            "effective_from": sessions[0],
            "effective_to": pd.NaT,
        }
    )
    names = pd.DataFrame(columns=["ts_code", "name", "effective_from", "effective_to"])
    return (
        {
            "calendar": calendar,
            "security_master": master,
            "namechange": names,
            "raw_daily": raw,
            "adj_factor": factors,
            "daily_basic": basic,
            "industry_membership": industries,
            "official_daily_status": status,
            "chan_states": states,
        },
        pd.Timestamp(decision),
        pd.Timestamp(execution),
    )


@pytest.fixture(scope="module")
def formal_case():
    frames, decision, execution = _frames()
    books = planner.empty_reference_books()
    artifact = planner.build_planning_artifact_from_frames(
        frames=frames,
        decision_dt=decision,
        exec_dt=execution,
        reference_books=books,
    )
    return frames, decision, execution, books, artifact


def test_artifact_freezes_all_252_family_seed_scenario_decisions(formal_case):
    frames, _, _, _, artifact = formal_case
    assert artifact["frozen_seeds"] == list(range(20260720, 20260740))
    assert len(artifact["arms"]) == 63
    assert len(artifact["execution_decisions"]) == 252
    assert planner.verify_planning_artifact(artifact, frames=frames).valid
    for arm in artifact["arms"].values():
        assert len(set(arm["cost_scenario_selection_identity_sha256"].values())) == 1
        assert set(arm["cost_scenario_selection_identity_sha256"]) == set(planner.SCENARIOS)


def test_feature_surface_is_prefix_invariant_and_uses_same_day_factor(formal_case):
    frames, decision, _, _, _ = formal_case
    full = planner.compute_decision_surface(frames, decision)
    prefix = {}
    for name, frame in frames.items():
        if "trade_date" in frame:
            prefix[name] = frame.loc[pd.to_datetime(frame["trade_date"]).le(decision)].copy()
        elif "dt" in frame:
            prefix[name] = frame.loc[pd.to_datetime(frame["dt"]).le(decision)].copy()
        elif "effective_from" in frame:
            prefix[name] = frame.loc[pd.to_datetime(frame["effective_from"]).le(decision)].copy()
        else:
            prefix[name] = frame.copy()
    prefix["calendar"] = frames["calendar"].loc[frames["calendar"]["trade_date"].le(decision)].copy()
    causal = planner.compute_decision_surface(prefix, decision)
    pd.testing.assert_frame_equal(full, causal)
    symbol = str(full.iloc[0]["symbol"])
    raw = frames["raw_daily"].query("ts_code == @symbol and trade_date == @decision").iloc[0]
    factor = frames["adj_factor"].query("ts_code == @symbol and trade_date == @decision").iloc[0]
    assert full.set_index("symbol").loc[symbol, "adjusted_close"] == pytest.approx(raw["close"] * factor["adj_factor"])


def test_factor_rank_stays_integral_and_semantic_replay_rejects_float_tamper(formal_case):
    frames, decision, _, _, artifact = formal_case
    surface = planner.compute_decision_surface(frames, decision)
    assert pd.api.types.is_integer_dtype(surface["factor_rank"])
    assert surface["factor_rank"].tolist() == list(range(1, len(surface) + 1))

    forged = copy.deepcopy(artifact)
    forged["ranked_surface"][0]["factor_rank"] = 1.0
    forged.pop("artifact_sha256")
    forged["artifact_sha256"] = planner.object_sha256(forged)
    result = planner.verify_planning_artifact(forged, frames=frames)
    assert not result.valid
    assert "semantic replay differs" in result.errors[0]


def test_state_projection_is_physical_three_columns_and_missing_state_denies_entry(formal_case):
    frames, decision, _, _, _ = formal_case
    extra = {name: frame.copy() for name, frame in frames.items()}
    extra["chan_states"]["future_return"] = 999.0
    with pytest.raises(planner.PlanningError, match="only"):
        planner.compute_decision_surface(extra, decision)

    baseline = planner.compute_decision_surface(frames, decision)
    symbol = str(baseline.iloc[0]["symbol"])
    missing = {name: frame.copy() for name, frame in frames.items()}
    missing["chan_states"] = missing["chan_states"].loc[
        ~(missing["chan_states"]["symbol"].eq(symbol) & pd.to_datetime(missing["chan_states"]["dt"]).eq(decision))
    ]
    denied = planner.compute_decision_surface(missing, decision).set_index("symbol").loc[symbol]
    assert bool(denied["state_missing"])
    assert not bool(denied["chan_allowed"])


def test_random_control_exact_matching_and_sparse_incumbent_pool_fails_closed(formal_case):
    frames, decision, execution, books, artifact = formal_case
    f_strata = Counter(tuple(value) for value in artifact["arms"]["F"]["stratum_by_symbol"].values())
    for seed in range(20260720, 20260740):
        control = artifact["arms"][f"R_match@{seed}"]
        assert Counter(tuple(value) for value in control["stratum_by_symbol"].values()) == f_strata

    prior = tuple(artifact["arms"]["F"]["ordered_symbols"][:10])
    with pytest.raises(planner.ExactControlError, match="incumbents"):
        planner.build_planning_artifact_from_frames(
            frames=frames,
            decision_dt=decision,
            exec_dt=execution,
            reference_books=books,
            previous_factor_targets=prior,
            previous_r_match_targets_by_seed=dict.fromkeys(range(20260720, 20260740), ()),
        )


def test_blocked_exit_occupies_a_reference_slot_before_chan_quota(formal_case):
    frames, decision, execution, books, _ = formal_case
    changed = dict(books)
    base = books["FC"]
    changed["FC"] = planner.ReferenceBook(
        actual_holdings=("BLOCKED.OUT",),
        blocked_exit_symbols=("BLOCKED.OUT",),
        sizing_nav_cny_by_scenario=base.sizing_nav_cny_by_scenario,
        sizing_nav_record_sha256_by_scenario=base.sizing_nav_record_sha256_by_scenario,
    )
    artifact = planner.build_planning_artifact_from_frames(
        frames=frames,
        decision_dt=decision,
        exec_dt=execution,
        reference_books=changed,
    )
    fc = artifact["arms"]["FC"]
    assert fc["blocked_exit_placeholders"] == ["BLOCKED.OUT"]
    assert fc["gate_candidate_new_entry_count"] == 49
    assert len(fc["ordered_symbols"]) + len(fc["blocked_exit_placeholders"]) <= 50


def test_planner_input_is_strict_and_content_addressed(formal_case):
    _, _, _, _, artifact = formal_case
    planner_input = planner.planner_input_from_artifact(artifact)
    assert planner_input["schema"] == planner.PLANNER_INPUT_SCHEMA
    forged = copy.deepcopy(planner_input)
    forged["decision_dt"] = "2099-01-01"
    with pytest.raises(planner.PlanningError, match="content hash"):
        planner._strict_planner_input(forged)
