from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_research_v2 as v2  # noqa: E402


def _load_generate_symbol_kines():
    """直接加载 czsc.mock，避开当前工作区旧 native 的顶层导入不匹配。"""
    path = ROOT / "czsc" / "mock.py"
    spec = importlib.util.spec_from_file_location("czsc.mock", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_symbol_kines


generate_symbol_kines = _load_generate_symbol_kines()


def _ranked_group(
    dt: str, ranks: dict[str, int], strata: dict[str, tuple[str, int, int]] | None = None
) -> pd.DataFrame:
    strata = strata or dict.fromkeys(ranks, ("I1", 0, 0))
    rows = []
    for symbol, rank in ranks.items():
        industry, mcap_bucket, adv_bucket = strata[symbol]
        rows.append(
            {
                "symbol": symbol,
                "dt": pd.Timestamp(dt),
                "factor_rank": rank,
                "factor_score": 1.0 - rank / 100.0,
                "adv20": 100_000.0,
                "mom_120_20_rank": 1.0 - rank / 110.0,
                "lowvol_60_rank": 1.0 - rank / 120.0,
                "industry_code": industry,
                "mcap_bucket": mcap_bucket,
                "adv_bucket": adv_bucket,
                "chan_allowed": rank % 2 == 0,
                "ma_allowed": rank % 3 == 0,
            }
        )
    return pd.DataFrame(rows)


def _plan(
    dt: str,
    symbols: tuple[str, ...],
    strata: dict[str, tuple[str, int, int]] | None = None,
) -> v2.DecisionPlan:
    strata = strata or dict.fromkeys(symbols, ("I1", 0, 0))
    return v2.DecisionPlan(
        decision_dt=pd.Timestamp(dt),
        exec_dt=pd.Timestamp(dt) + pd.Timedelta(days=1),
        symbols=symbols,
        rank_by_symbol={symbol: i + 1 for i, symbol in enumerate(symbols)},
        selection_reason=dict.fromkeys(symbols, "rank_entry"),
        adv_cny=dict.fromkeys(symbols, 100_000_000.0),
        chan_allowed=dict.fromkeys(symbols, True),
        ma_allowed=dict.fromkeys(symbols, True),
        stratum_by_symbol={symbol: strata[symbol] for symbol in symbols},
    )


def test_frozen_protocol_is_strict_and_machine_readable(tmp_path: Path):
    config, protocol = v2.load_protocol()

    assert config.selection == v2.SelectionSpec(target_size=50, retention_max_rank=75, ma_window=20)
    assert config.allowed_chan_regimes == (5, 6, 7, 8)
    assert protocol["factor_model"]["weights"] == {"mom_120_20": 0.5, "lowvol_60": 0.5}
    assert protocol["factor_model"]["winsor_quantile_method"] == "linear"
    assert protocol["eligibility"]["canonical_board_values"] == ["MAIN", "CHINEXT", "STAR", "BSE"]
    assert (
        protocol["forward_validation"]["formal_start_rule"]
        == "first_completed_week_with_eligible_count_gte_target_size_then_no_later_shortfall"
    )
    assert protocol["forward_validation"]["required_engineering_gates"] == list(v2.REQUIRED_ENGINEERING_GATES)
    assert protocol["statistics"]["confirmatory_comparisons"] == list(v2.CONFIRMATORY_COMPARISONS)
    assert protocol["data_contract"]["corporate_action_types"] == list(v2.CORPORATE_ACTION_TYPES)
    assert protocol["data_contract"]["universe_reconciliation_partitions"] == list(
        v2.UNIVERSE_RECONCILIATION_PARTITIONS
    )

    changed = dict(protocol)
    changed["selection"] = {**protocol["selection"], "target_size": 20}
    path = tmp_path / "changed.json"
    path.write_text(v2.canonical_json(changed).decode(), encoding="utf-8")
    with pytest.raises(v2.ProtocolError, match="byte-for-byte"):
        v2.load_protocol(path)


def test_adjusted_features_are_prefix_invariant_and_ma_uses_only_history():
    raw = generate_symbol_kines("000001.SZ", "日线", "20200101", "20221231", seed=17)
    raw["adj_factor"] = np.where(np.arange(len(raw)) < len(raw) // 2, 1.0, 1.25)
    required = raw[["symbol", "dt", "close", "adj_factor", "amount"]].copy()
    cutoff = len(required) - 30

    prefix = v2.compute_adjusted_features(
        required.iloc[:cutoff], required.iloc[:cutoff]["dt"], asof=required.iloc[cutoff - 1]["dt"]
    ).reset_index(drop=True)
    full = v2.compute_adjusted_features(required, required["dt"], asof=required.iloc[-1]["dt"])
    full = full.iloc[:cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        prefix[["mom_120_20", "lowvol_60", "sma20", "adv20", "ma_allowed"]],
        full[["mom_120_20", "lowvol_60", "sma20", "adv20", "ma_allowed"]],
    )
    expected = required["close"] * required["adj_factor"]
    assert np.allclose(full["adjusted_close"], expected.iloc[:cutoff])

    suspended_dt = required.iloc[80]["dt"]
    observed = required.drop(index=80)
    dense = v2.compute_adjusted_features(observed, required["dt"], asof=required.iloc[-1]["dt"]).set_index("dt")
    assert dense.loc[suspended_dt, "amount"] == 0
    assert not bool(dense.loc[suspended_dt, "is_valid_trade"])
    assert dense.loc[suspended_dt, "adjusted_close"] == pytest.approx(dense.iloc[79]["adjusted_close"])


def test_rank_cross_section_neutralizes_pit_fields_and_fails_closed_on_missing():
    rows = []
    dt = pd.Timestamp("2026-07-17")
    for i in range(20):
        rows.append(
            {
                "symbol": f"S{i:02d}",
                "dt": dt,
                "eligible": True,
                "mom_120_20": i / 20,
                "lowvol_60": (20 - i) / 20,
                "adv20": 60_000 + i * 100,
                "industry_code": "A" if i < 10 else "B",
                "free_float_mcap": 1e9 * (i + 1),
                "adjusted_close": 10 + i,
                "sma20": 11 + i / 2,
            }
        )
    features = pd.DataFrame(rows)

    first = v2.rank_cross_section_v2(features)
    second = v2.rank_cross_section_v2(features.sample(frac=1, random_state=7))

    columns = ["symbol", "factor_rank", "factor_score", "mcap_bucket", "adv_bucket", "ma_allowed"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values("symbol").reset_index(drop=True),
        second[columns].sort_values("symbol").reset_index(drop=True),
    )
    assert first["factor_rank"].tolist() == list(range(1, 21))

    broken = features.copy()
    broken.loc[0, "industry_code"] = None
    with pytest.raises(ValueError, match="industry_code"):
        v2.rank_cross_section_v2(broken)


def test_joint_buckets_are_deterministic_when_values_tie():
    dt = pd.Timestamp("2026-07-17")
    features = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(10)],
            "dt": dt,
            "eligible": True,
            "mom_120_20": np.linspace(0.0, 0.9, 10),
            "lowvol_60": np.linspace(0.9, 0.0, 10),
            "adv20": 100_000.0,
            "industry_code": "I1",
            "free_float_mcap": 1_000_000_000.0,
            "adjusted_close": 10.0,
            "sma20": 9.0,
        }
    )

    first = v2.rank_cross_section_v2(features)
    second = v2.rank_cross_section_v2(features.sample(frac=1, random_state=19))
    columns = ["symbol", "mcap_bucket", "adv_bucket"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values("symbol").reset_index(drop=True),
        second[columns].sort_values("symbol").reset_index(drop=True),
    )


def test_eligibility_is_rederived_and_excludes_st_board_new_and_illiquid_rows():
    config, _ = v2.load_protocol()
    rows = []
    for symbol in ["OK", "ST", "STAR", "NEW", "ILLIQ"]:
        rows.append(
            {
                "symbol": symbol,
                "dt": pd.Timestamp("2026-07-17"),
                "eligible": True,
                "pit_universe_member": True,
                "market_sessions_since_listing": 300,
                "valid_sessions_60": 60,
                "is_st": False,
                "board": "MAIN",
                "adv20": 60_000.0,
                "mom_120_20": 0.1,
                "lowvol_60": -0.01,
                "industry_code": "I1",
                "free_float_mcap": 1e9,
                "adjusted_close": 10.0,
                "sma20": 9.0,
            }
        )
    frame = pd.DataFrame(rows).set_index("symbol", drop=False)
    frame.loc["ST", "is_st"] = True
    frame.loc["STAR", "board"] = "STAR"
    frame.loc["NEW", "market_sessions_since_listing"] = 249
    frame.loc["ILLIQ", "adv20"] = 49_999.0

    derived = v2.derive_eligibility(frame.reset_index(drop=True), config.eligibility).set_index("symbol")

    assert derived["eligible"].to_dict() == {"OK": True, "ST": False, "STAR": False, "NEW": False, "ILLIQ": False}
    assert "st" in derived.loc["ST", "ineligibility_reasons"]
    assert "excluded_board" in derived.loc["STAR", "ineligibility_reasons"]


def test_select_with_50_75_buffer_retains_then_replaces_deterministically():
    ranks = {f"S{i:03d}": i + 1 for i in range(100)}
    group = _ranked_group("2026-07-17", ranks)
    previous = tuple(["S060", "S074", "S075"] + [f"S{i:03d}" for i in range(47)])

    selected = v2.select_with_rank_buffer(group, previous, target_size=50, retention_max_rank=75)

    assert "S060" in selected.symbols
    assert "S074" in selected.symbols
    assert "S075" not in selected.symbols
    assert len(selected.symbols) == 50
    assert selected.reason_by_symbol["S060"] == "buffer_retain"
    assert selected.symbols[:2] == ("S000", "S001")


def test_buffered_plans_require_current_adv_for_ineligible_exits():
    day1 = _ranked_group("2026-07-10", {"A": 1, "B": 2, "C": 3})
    day2 = _ranked_group("2026-07-17", {"B": 1, "C": 2, "D": 3})
    ranked = pd.concat([day1, day2], ignore_index=True)
    schedule = pd.DataFrame(
        {
            "decision_dt": pd.to_datetime(["2026-07-10", "2026-07-17"]),
            "exec_dt": pd.to_datetime(["2026-07-13", "2026-07-20"]),
        }
    )
    all_adv = pd.DataFrame(
        [
            {"symbol": symbol, "dt": dt, "adv20": 100_000.0}
            for dt, symbols in [("2026-07-10", "ABC"), ("2026-07-17", "ABCD")]
            for symbol in symbols
        ]
    )
    calendar = pd.bdate_range("2026-07-06", "2026-07-20")
    plans = v2.build_buffered_target_plans(ranked, schedule, v2.SelectionSpec(2, 3, 20), all_adv, calendar)

    assert plans[pd.Timestamp("2026-07-10")].symbols == ("A", "B")
    assert plans[pd.Timestamp("2026-07-17")].symbols == ("B", "C")
    assert "A" in plans[pd.Timestamp("2026-07-17")].adv_cny

    with pytest.raises(ValueError, match="exits/targets"):
        v2.build_buffered_target_plans(
            ranked,
            schedule,
            v2.SelectionSpec(2, 3, 20),
            all_adv[~((all_adv["symbol"] == "A") & (all_adv["dt"] == "2026-07-17"))],
            calendar,
        )

    missing_schedule = schedule.iloc[:1]
    with pytest.raises(ValueError, match="ranked/schedule decision dates differ"):
        v2.build_buffered_target_plans(ranked, missing_schedule, v2.SelectionSpec(2, 3, 20), all_adv, calendar)

    skipped = pd.DataFrame(
        {
            "decision_dt": pd.to_datetime(["2026-07-10", "2026-07-24"]),
            "exec_dt": pd.to_datetime(["2026-07-13", "2026-07-27"]),
        }
    )
    with pytest.raises(ValueError, match="skip"):
        v2.validate_weekly_schedule(skipped, pd.bdate_range("2026-07-06", "2026-07-27"))


def test_gate_uses_actual_holdings_and_random_control_matches_exact_quota():
    config, _ = v2.load_protocol()
    plan = _plan("2026-07-17", ("A", "B", "C", "D"))
    plan = v2.DecisionPlan(
        **{
            **plan.__dict__,
            "chan_allowed": {"A": False, "B": False, "C": True, "D": False},
        }
    )

    accepted, quota = v2.derive_gate_decision(plan, held_symbols={"A"}, gate="chan")

    assert accepted == ("A", "C")
    assert quota == v2.GateQuota(total=2, retained=1, new=1)
    random = v2.apply_matched_random_gate(
        plan.symbols,
        {"A"},
        quota,
        seed=9,
        protocol_id=config.protocol_id,
        arm="FGR",
        seed_index=0,
        decision_dt=plan.decision_dt,
    )
    assert len(random) == 2
    assert random[0] == "A"
    with pytest.raises(v2.ControlMatchError, match="retained quota"):
        v2.apply_matched_random_gate(
            plan.symbols,
            set(),
            quota,
            seed=9,
            protocol_id=config.protocol_id,
            arm="FGR",
            seed_index=0,
            decision_dt=plan.decision_dt,
        )

    # B 上周即使曾在目标中，只要没有实际成交，本周仍需重新过门。
    accepted_without_fill, _ = v2.derive_gate_decision(plan, held_symbols={"A"}, gate="chan")
    assert "B" not in accepted_without_fill


def test_plan_and_all_gate_control_identities_are_deeply_frozen_before_replay():
    config, _ = v2.load_protocol()
    plan = _plan("2026-07-17", ("A", "B", "C", "D"))
    plan = v2.DecisionPlan(
        **{
            **plan.__dict__,
            "chan_allowed": {"A": False, "B": False, "C": True, "D": False},
            "ma_allowed": {"A": False, "B": False, "C": False, "D": True},
        }
    )
    seeds = tuple(range(config.random_seed, config.random_seed + config.random_seed_count))

    frozen = v2.freeze_gate_identities(
        plan,
        held_fc={"A"},
        held_fma={"A"},
        held_fgr_by_seed={seed: {"A"} for seed in seeds},
        held_fmgr_by_seed={seed: {"A"} for seed in seeds},
        config=config,
    )

    assert frozen.fc_symbols == ("A", "C")
    assert frozen.fma_symbols == ("A", "D")
    assert set(frozen.fgr_symbols_by_seed) == set(seeds)
    assert len(frozen.identity_sha256) == 64
    with pytest.raises(TypeError):
        plan.chan_allowed["A"] = True
    with pytest.raises(TypeError):
        frozen.fgr_symbols_by_seed[seeds[0]] = ("A",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chan_allowed", {"A": "False"}, "literal booleans"),
        ("rank_by_symbol", {"A": 1.5}, "positive integers without coercion"),
    ],
)
def test_decision_plan_rejects_lossy_identity_coercion(field: str, value: dict, message: str):
    kwargs = {
        "decision_dt": pd.Timestamp("2026-07-17"),
        "exec_dt": pd.Timestamp("2026-07-20"),
        "symbols": ("A",),
        "rank_by_symbol": {"A": 1},
        "selection_reason": {"A": "rank_entry"},
        "adv_cny": {"A": 100_000_000.0},
        "chan_allowed": {"A": True},
        "ma_allowed": {"A": True},
        "stratum_by_symbol": {"A": ("I1", 0, 0)},
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        v2.DecisionPlan(**kwargs)


def test_blocked_exit_consumes_reference_slot_before_gate_quota_freeze():
    plan = _plan("2026-07-17", ("A", "B", "C", "D"))

    budget = v2.apply_reference_slot_budget(
        plan,
        actual_held_symbols={"A", "X"},
        blocked_exit_symbols={"X"},
        maximum_positions=4,
    )

    assert budget.blocked_outside_targets == ("X",)
    assert budget.held_factor_targets == ("A",)
    assert budget.admitted_new_targets == ("B", "C")
    assert budget.effective_plan.symbols == ("A", "B", "C")
    assert len(budget.effective_plan.symbols) + len(budget.blocked_outside_targets) == 4
    with pytest.raises(ValueError, match="subset"):
        v2.apply_reference_slot_budget(
            plan,
            actual_held_symbols={"A"},
            blocked_exit_symbols={"X"},
            maximum_positions=4,
        )


def test_rng_domain_separation_binds_protocol_arm_seed_index_and_decision_date():
    identity = (v2.RNG_ALGORITHM_VERSION, "protocol", "FGR", 0, "2026-07-24", 20260717)
    first = v2._rng_for(*identity).integers(0, 2**31, size=8)
    repeated = v2._rng_for(*identity).integers(0, 2**31, size=8)
    other_arm = v2._rng_for(v2.RNG_ALGORITHM_VERSION, "protocol", "FMGR", 0, "2026-07-24", 20260717).integers(
        0, 2**31, size=8
    )
    other_week = v2._rng_for(v2.RNG_ALGORITHM_VERSION, "protocol", "FGR", 0, "2026-07-31", 20260717).integers(
        0, 2**31, size=8
    )

    assert np.array_equal(first, repeated)
    assert first.tolist() == [1435807938, 1985068693, 775879264, 938090014, 1202982539, 835374612, 699688240, 195327961]
    assert not np.array_equal(first, other_arm)
    assert not np.array_equal(first, other_week)


def test_r_match_exactly_matches_joint_strata_and_retention():
    symbols = [f"S{i}" for i in range(8)]
    strata = dict.fromkeys(symbols, ("I1", 0, 0))
    day1 = _ranked_group("2026-07-10", {symbol: i + 1 for i, symbol in enumerate(symbols)}, strata)
    day2 = _ranked_group("2026-07-17", {symbol: i + 1 for i, symbol in enumerate(symbols[1:] + symbols[:1])}, strata)
    ranked = pd.concat([day1, day2], ignore_index=True)
    factor_plans = {
        pd.Timestamp("2026-07-10"): _plan("2026-07-10", ("S0", "S1"), strata),
        pd.Timestamp("2026-07-17"): _plan("2026-07-17", ("S1", "S2"), strata),
    }
    all_adv = pd.DataFrame(
        [{"symbol": symbol, "dt": dt, "adv20": 100_000.0} for dt in ["2026-07-10", "2026-07-17"] for symbol in symbols]
    )

    config, _ = v2.load_protocol()
    controls = v2.build_matched_random_plans(
        ranked,
        factor_plans,
        all_adv,
        seed=11,
        protocol_id=config.protocol_id,
        seed_index=0,
    )
    first = set(controls[pd.Timestamp("2026-07-10")].symbols)
    second = set(controls[pd.Timestamp("2026-07-17")].symbols)

    assert len(first) == len(second) == 2
    assert len(first & second) == 1
    assert Counter(controls[pd.Timestamp("2026-07-17")].stratum_by_symbol.values()) == Counter(
        strata[s] for s in ("S1", "S2")
    )


def test_state_projection_is_physically_restricted_and_missing_state_denies():
    ranked = _ranked_group("2026-07-17", {"A": 1, "B": 2})
    states = pd.DataFrame({"symbol": ["A"], "dt": pd.to_datetime(["2026-07-17"]), "regime": [6]})

    attached = v2.attach_chan_gate(ranked, states, (5, 6, 7, 8))

    assert attached.set_index("symbol")["chan_allowed"].to_dict() == {"A": True, "B": False}
    with pytest.raises(ValueError, match="exactly"):
        v2.attach_chan_gate(ranked, states.assign(next_open=10.0), (5, 6, 7, 8))
    with pytest.raises(ValueError, match="integral"):
        v2.attach_chan_gate(ranked, states.assign(regime=5.5), (5, 6, 7, 8))


def test_factor_diagnostics_invalidates_missing_labels_without_survivor_drop_or_backfill():
    ranked = _ranked_group("2026-07-17", {f"S{i}": i + 1 for i in range(6)})
    labels = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(6)],
            "dt": pd.Timestamp("2026-07-17"),
            "forward_return": [np.nan, 0.1, 0.9, 0.0, -0.1, 0.2],
        }
    )

    diagnostics = v2.factor_diagnostics_v2(ranked, labels, topn=2).iloc[0]

    assert diagnostics["topn"] == 2
    assert diagnostics["topn_label_coverage"] == 0.5
    assert diagnostics["period_valid"] is np.False_ or diagnostics["period_valid"] is False
    assert np.isnan(diagnostics["topn_equal_weight_return"])
    assert diagnostics["invalid_reason"] == "missing_or_non_finite_forward_label_no_survivor_drop"


def test_seed_aggregation_and_first_52_week_statistics_are_frozen_and_deterministic():
    config, _ = v2.load_protocol()
    index = pd.date_range("2026-07-17", periods=52, freq="W-FRI")
    seeds = range(config.random_seed, config.random_seed + config.random_seed_count)
    weekly = {
        seed: pd.Series(np.linspace(-0.01, 0.02, 52) + offset * 1e-5, index=index) for offset, seed in enumerate(seeds)
    }

    aggregated = v2.aggregate_seed_weekly_v2(weekly, config)
    first = v2.active_statistics_v2(aggregated, config, "F_2x_minus_R_match_2x")
    second = v2.active_statistics_v2(aggregated, config, "F_2x_minus_R_match_2x")

    assert first == second
    assert first["annualized_active_arithmetic"] > 0
    assert 0 <= first["top10_positive_weeks_share"] <= 1
    with pytest.raises(ValueError, match="exactly the first 52"):
        v2.active_statistics_v2(aggregated.iloc[:-1], config, "F_2x_minus_R_match_2x")
    with pytest.raises(ValueError, match="unregistered confirmatory comparison"):
        v2.active_statistics_v2(aggregated, config, "posthoc_comparison")
    with pytest.raises(ValueError, match="zero weekly sample standard deviation"):
        v2.active_statistics_v2(np.zeros(52), config, "F_2x_minus_R_match_2x")
    with pytest.raises(ValueError, match="all 20"):
        v2.aggregate_seed_weekly_v2({seed: weekly[seed] for seed in list(seeds)[:-1]}, config)


def test_ir_or_drawdown_improvement_uses_exact_aligned_first_52_week_paths():
    config, _ = v2.load_protocol()
    index = pd.date_range("2026-07-17", periods=52, freq="W-FRI")
    factor = pd.Series(np.tile([0.02, -0.02], 26), index=index)
    fc = pd.Series(np.tile([0.001, 0.002], 26), index=index)

    result = v2.ir_mdd_improvement_v2(fc, factor, config)

    assert result["FC_improves_ir_or_drawdown"] is True
    assert result["fc_maximum_drawdown"] == 0.0
    assert result["f_maximum_drawdown"] < 0.0
    with pytest.raises(ValueError, match="exactly 52 weeks"):
        v2.ir_mdd_improvement_v2(fc.iloc[:-1], factor.iloc[:-1], config)
    with pytest.raises(ValueError, match="identical indices"):
        v2.ir_mdd_improvement_v2(fc, factor.set_axis(index + pd.Timedelta(days=1)), config)
    with pytest.raises(ValueError, match="zero weekly sample standard deviation"):
        v2.ir_mdd_improvement_v2(pd.Series(np.zeros(52), index=index), factor, config)


def test_maximum_drawdown_includes_initial_nav_before_a_first_week_loss():
    config, _ = v2.load_protocol()
    index = pd.date_range("2026-07-17", periods=52, freq="W-FRI")
    fc = pd.Series([-0.5, *([0.0] * 51)], index=index)
    factor = pd.Series([0.0, -0.1, *([0.01] * 50)], index=index)

    result = v2.ir_mdd_improvement_v2(fc, factor, config)

    assert result["fc_maximum_drawdown"] == pytest.approx(-0.5)
    assert result["f_maximum_drawdown"] == pytest.approx(-0.1)
    assert result["FC_improves_ir_or_drawdown"] is False


def _active(annual: float = 0.1, median: float = 0.01, t_value: float = 2.0, lower: float = 0.01):
    return {
        "annualized_active_arithmetic": annual,
        "weekly_active_median": median,
        "top10_positive_weeks_share": 0.4,
        "hac_t_weekly_mean_lag4": t_value,
        "block_bootstrap_lower": lower,
    }


def test_private_decision_tree_enforces_data_then_factor_then_timing_hierarchy():
    config, _ = v2.load_protocol()
    engineering = {"all_passed": True, "failed_checks": []}
    seeds = list(range(config.random_seed, config.random_seed + config.random_seed_count))
    controls = {
        "all_exact": True,
        "fc_fgr_average_exposure_gap": 0.005,
        "r_match_seed_ids": seeds,
        "fgr_seed_ids": seeds,
        "fmgr_seed_ids": seeds,
        "seed_aggregation": "per_week_equal_seed_mean_then_compute_preregistered_statistics",
        "identity_frozen_before_cost_replay": True,
        "same_identity_all_cost_scenarios": True,
        "first_frozen_window_only": True,
        "evaluation_window_weeks": 52,
        "gate_identity_hash_count": 52,
        "invalid_period_count": 0,
    }
    metrics = {
        "F_2x_minus_R_match_2x": _active(),
        "FC_gross_minus_FGR_gross": _active(),
        "FC_2x_minus_FGR_2x": _active(),
        "FC_gross_minus_FMA_gross": _active(),
        "FC_2x_minus_FMA_2x": _active(),
        "FC_improves_ir_or_drawdown": True,
    }

    invalid = v2._evaluate_v2_logic(
        {"forward_start_allowed": False, "failed_gates": ["pit_industry"]}, engineering, controls, metrics, config
    )
    assert invalid["status"] == "INVALID_DATA_OR_ENGINEERING"

    collecting = v2._evaluate_v2_logic(
        {"forward_start_allowed": True, "confirmatory_oos": False, "complete_weeks": 17},
        engineering,
        controls,
        metrics,
        config,
    )
    assert collecting["status"] == "FORWARD_COLLECTION_REQUIRED"

    rejected_metrics = {**metrics, "F_2x_minus_R_match_2x": _active(annual=-0.01)}
    rejected = v2._evaluate_v2_logic(
        {"forward_start_allowed": True, "confirmatory_oos": True},
        engineering,
        controls,
        rejected_metrics,
        config,
    )
    assert rejected["status"] == "REJECT_FACTOR"
    assert rejected["timing_evaluated"] is False

    passed = v2._evaluate_v2_logic(
        {"forward_start_allowed": True, "confirmatory_oos": True}, engineering, controls, metrics, config
    )
    assert passed["status"] == "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY"
    assert passed["live_trading_allowed"] is False

    string_boolean = v2._evaluate_v2_logic(
        {"forward_start_allowed": "true", "confirmatory_oos": True}, engineering, controls, metrics, config
    )
    assert string_boolean["status"] == "INVALID_DATA_OR_ENGINEERING"
    nan_gap = v2._evaluate_v2_logic(
        {"forward_start_allowed": True, "confirmatory_oos": True},
        engineering,
        {**controls, "fc_fgr_average_exposure_gap": np.nan},
        metrics,
        config,
    )
    assert nan_gap["status"] == "INVALID_CONTROL"

    public = v2.evaluate_v2(
        {"forward_start_allowed": True, "confirmatory_oos": True},
        engineering,
        controls,
        metrics,
        config,
    )
    assert public["status"] == "INVALID_DATA_OR_ENGINEERING"
    assert public["failed_engineering_checks"] == ["semantic_artifact_replay_verifier_unavailable"]

    malformed_public = v2.evaluate_v2({}, {}, {}, {}, config)
    assert malformed_public == public
