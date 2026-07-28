from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_exploration_stage2 as stage2  # noqa: E402


def _spec() -> dict:
    return copy.deepcopy(stage2.load_and_validate_spec())


def _empty_bundle(
    *,
    ranked: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    proposals: pd.DataFrame | None = None,
    failure_sample: pd.DataFrame | None = None,
) -> stage2.InputBundle:
    dates = pd.DatetimeIndex(
        sorted(
            set(
                pd.concat(
                    [
                        frame["decision_dt"]
                        for frame in (ranked, memberships, proposals, failure_sample)
                        if frame is not None and len(frame)
                    ],
                    ignore_index=True,
                )
            )
        )
    )
    return stage2.InputBundle(
        ranked=ranked if ranked is not None else pd.DataFrame(),
        memberships=memberships if memberships is not None else pd.DataFrame(),
        proposals=proposals if proposals is not None else pd.DataFrame(),
        attribution_weekly=pd.DataFrame(),
        failure_sample=failure_sample if failure_sample is not None else pd.DataFrame(),
        decision_dates=dates,
        paths={},
    )


def test_stage2_spec_discloses_reuse_and_locks_all_inputs():
    spec = stage2.load_and_validate_spec()

    assert spec["mode"] == "EXPLORATORY_ONLY"
    assert spec["study_type"] == "RETROSPECTIVE_FALSIFICATION_ONLY"
    assert spec["confirmation_chain"] == "NOT_STARTED"
    assert spec["design_provenance"]["stage1_full_sample_already_seen"] is True
    assert spec["design_provenance"]["stage2_is_preregistered_confirmation"] is False
    assert [row["id"] for row in spec["planned_experiments"]] == list(stage2.EXPECTED_EXPERIMENT_IDS)
    assert len(spec["frozen_stage1_inputs"]) == 10


def test_study_identity_binds_spec_and_source(tmp_path: Path):
    spec = _spec()
    source = tmp_path / "study.py"
    source.write_text("first\n", encoding="utf-8")
    first = stage2.study_identity(spec, source)

    source.write_text("second\n", encoding="utf-8")
    second = stage2.study_identity(spec, source)
    changed = copy.deepcopy(spec)
    changed["study_id"] = "changed"
    third = stage2.study_identity(changed, source)

    assert first != second
    assert second != third
    assert len(first) == 64


def test_paired_gate_weekly_is_common_week_and_week_weighted():
    frame = pd.DataFrame(
        [
            {
                "decision_dt": pd.Timestamp("2024-01-05"),
                "entry_tradable": True,
                "fwd_5d_open_return": 1.0,
                "left": True,
                "right": False,
            },
            {
                "decision_dt": pd.Timestamp("2024-01-05"),
                "entry_tradable": True,
                "fwd_5d_open_return": 1.0,
                "left": True,
                "right": False,
            },
            {
                "decision_dt": pd.Timestamp("2024-01-05"),
                "entry_tradable": True,
                "fwd_5d_open_return": 0.0,
                "left": False,
                "right": True,
            },
            {
                "decision_dt": pd.Timestamp("2024-01-12"),
                "entry_tradable": True,
                "fwd_5d_open_return": -1.0,
                "left": True,
                "right": False,
            },
            {
                "decision_dt": pd.Timestamp("2024-01-12"),
                "entry_tradable": True,
                "fwd_5d_open_return": 0.0,
                "left": False,
                "right": True,
            },
            {
                "decision_dt": pd.Timestamp("2024-01-19"),
                "entry_tradable": True,
                "fwd_5d_open_return": 9.0,
                "left": True,
                "right": False,
            },
        ]
    )

    weekly = stage2.paired_gate_weekly(
        frame,
        frame["left"],
        frame["right"],
        left_name="left",
        right_name="right",
    )

    assert weekly["decision_dt"].tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert weekly["weekly_difference"].tolist() == pytest.approx([1.0, -1.0])
    assert weekly["weekly_difference"].mean() == pytest.approx(0.0)


def test_h1_controls_are_deterministic_on_exact_common_support():
    spec = _spec()
    spec["h1_state7_increment"]["matched_control"]["repetitions"] = 4
    spec["h1_state7_increment"]["proposal_identity_randomization"]["repetitions"] = 4
    spec["h1_state7_increment"]["minimum_validation_common_weeks"] = 1
    spec["h1_state7_increment"]["minimum_each_half_common_weeks"] = 0
    spec["h1_state7_increment"]["minimum_validation_state7_events"] = 1
    spec["h1_state7_increment"]["matched_control"]["minimum_full_match_rate"] = 0
    spec["h1_state7_increment"]["matched_control"]["minimum_half_match_rate"] = 0
    rows = []
    for week, decision_dt in enumerate([pd.Timestamp("2024-01-05"), pd.Timestamp("2025-03-21")]):
        for index in range(4):
            rows.append(
                {
                    "decision_dt": decision_dt,
                    "arm": "FC",
                    "symbol": f"S{week}{index}",
                    "regime": 7 if index == 0 else 4,
                    "ma_allowed": index in {0, 1},
                    "industry_code": "I1",
                    "mcap_bucket": 2,
                    "factor_rank": index + 1,
                    "entry_tradable": True,
                    "fwd_5d_open_return": (index - 1) / 100,
                }
            )
    proposals = pd.DataFrame(rows)
    bundle = _empty_bundle(proposals=proposals)
    common_weekly = stage2.paired_gate_weekly(
        proposals,
        proposals["regime"].eq(7),
        proposals["ma_allowed"],
        left_name="state7",
        right_name="sma20",
    )
    common_weekly["estimand"] = "proposal_common_support"
    common_summary = stage2.summarize_weekly_frame(
        common_weekly,
        "weekly_difference",
        spec,
        "test",
        extra={"estimand": "proposal_common_support", "state7_events": 2, "sma20_events": 4},
    )

    first = stage2.run_h1_controls(spec, bundle, common_weekly, common_summary)
    second = stage2.run_h1_controls(spec, bundle, common_weekly, common_summary)

    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[3] == second[3]


def test_h1_segment_event_counts_are_not_full_sample_repeats():
    spec = _spec()
    proposal_rows = []
    membership_rows = []
    for decision_dt, state7_count in [
        (pd.Timestamp("2023-12-29"), 2),
        (pd.Timestamp("2024-01-05"), 1),
    ]:
        week = [
            {
                "decision_dt": decision_dt,
                "arm": "FC",
                "symbol": f"P{decision_dt:%Y%m%d}M",
                "regime": 4,
                "ma_allowed": True,
                "entry_tradable": True,
                "fwd_5d_open_return": 0.0,
            }
        ]
        week.extend(
            {
                "decision_dt": decision_dt,
                "arm": "FC",
                "symbol": f"P{decision_dt:%Y%m%d}S{index}",
                "regime": 7,
                "ma_allowed": False,
                "entry_tradable": True,
                "fwd_5d_open_return": 0.01,
            }
            for index in range(state7_count)
        )
        proposal_rows.extend(week)
        membership_rows.extend({**row, "arm": "F"} for row in week)
    bundle = _empty_bundle(
        proposals=pd.DataFrame(proposal_rows),
        memberships=pd.DataFrame(membership_rows),
    )

    _, summary = stage2.run_h1_common_support(spec, bundle)
    proposal = summary[summary["estimand"].eq("proposal_common_support")].set_index("segment")

    assert proposal.loc["FULL", "state7_events"] == 3
    assert proposal.loc["DEVELOPMENT", "state7_events"] == 2
    assert proposal.loc["HISTORICAL_VALIDATION", "state7_events"] == 1


def test_h1_exact_match_excludes_treated_events_without_matching_strata():
    spec = _spec()
    spec["h1_state7_increment"]["matched_control"]["repetitions"] = 2
    spec["h1_state7_increment"]["proposal_identity_randomization"]["repetitions"] = 2
    spec["h1_state7_increment"]["minimum_validation_common_weeks"] = 1
    spec["h1_state7_increment"]["minimum_each_half_common_weeks"] = 0
    spec["h1_state7_increment"]["minimum_validation_state7_events"] = 1
    spec["h1_state7_increment"]["matched_control"]["minimum_full_match_rate"] = 0
    spec["h1_state7_increment"]["matched_control"]["minimum_half_match_rate"] = 0
    rows = []
    for week, decision_dt in enumerate([pd.Timestamp("2024-01-05"), pd.Timestamp("2025-03-21")]):
        rows.extend(
            [
                {
                    "decision_dt": decision_dt,
                    "arm": "FC",
                    "symbol": f"M{week}",
                    "regime": 7,
                    "ma_allowed": False,
                    "industry_code": "MATCHED",
                    "mcap_bucket": 1,
                    "factor_rank": 1,
                    "entry_tradable": True,
                    "fwd_5d_open_return": 0.10,
                },
                {
                    "decision_dt": decision_dt,
                    "arm": "FC",
                    "symbol": f"U{week}",
                    "regime": 7,
                    "ma_allowed": False,
                    "industry_code": "UNMATCHED",
                    "mcap_bucket": 1,
                    "factor_rank": 2,
                    "entry_tradable": True,
                    "fwd_5d_open_return": 9.0,
                },
                {
                    "decision_dt": decision_dt,
                    "arm": "FC",
                    "symbol": f"C{week}",
                    "regime": 4,
                    "ma_allowed": False,
                    "industry_code": "MATCHED",
                    "mcap_bucket": 1,
                    "factor_rank": 3,
                    "entry_tradable": True,
                    "fwd_5d_open_return": 0.0,
                },
                {
                    "decision_dt": decision_dt,
                    "arm": "FC",
                    "symbol": f"SMA{week}",
                    "regime": 4,
                    "ma_allowed": True,
                    "industry_code": "SMA_ONLY",
                    "mcap_bucket": 1,
                    "factor_rank": 4,
                    "entry_tradable": True,
                    "fwd_5d_open_return": 0.0,
                },
            ]
        )
    proposals = pd.DataFrame(rows)
    common_weekly = stage2.paired_gate_weekly(
        proposals,
        proposals["regime"].eq(7),
        proposals["ma_allowed"],
        left_name="state7",
        right_name="sma20",
    )
    common_weekly["estimand"] = "proposal_common_support"
    common_summary = stage2.summarize_weekly_frame(
        common_weekly,
        "weekly_difference",
        spec,
        "test",
        extra={"estimand": "proposal_common_support"},
        segment_sums={"state7_events": "state7_events", "sma20_events": "sma20_events"},
    )

    _, matched, _, _ = stage2.run_h1_controls(spec, _empty_bundle(proposals=proposals), common_weekly, common_summary)

    assert matched["matched_state7_mean_V1"].tolist() == pytest.approx([0.10, 0.10])
    assert matched["matched_state7_mean_V2"].tolist() == pytest.approx([0.10, 0.10])
    assert matched["state7_minus_matched_FULL"].tolist() == pytest.approx([0.10, 0.10])


def test_h3_nested_feature_selection_uses_development_only():
    spec = _spec()
    rows = []
    for segment_start, is_validation in [("2023-01-06", False), ("2024-01-05", True)]:
        for index in range(40):
            failure = index % 2 == 0
            row = {
                "decision_dt": pd.Timestamp(segment_start) + pd.Timedelta(weeks=index % 20),
                "symbol": f"{'V' if is_validation else 'D'}{index}",
                "failure_primary": failure,
                "failure_bottom_quintile": index % 5 == 0,
            }
            for feature_no, feature in enumerate(stage2.DECISION_TIME_FEATURES):
                row[feature] = float(index + feature_no)
            row["mom_120_20"] = float(failure) * 10 + index / 100
            row["factor_score"] = -float(failure) * 8 + index / 100
            if is_validation:
                row["distance_to_sma20"] = 1000 - index
                row["volume_ratio"] = -1000 + index
            rows.append(row)
    failure_sample = pd.DataFrame(rows)
    bundle = _empty_bundle(failure_sample=failure_sample)

    first = stage2.run_h3_profile_replication(spec, bundle)
    changed = failure_sample.copy()
    validation_mask = changed["decision_dt"].ge("2024-01-01")
    changed.loc[validation_mask, list(stage2.DECISION_TIME_FEATURES)] *= -100
    second = stage2.run_h3_profile_replication(spec, _empty_bundle(failure_sample=changed))

    first_selected = first[5]["selected_development_features"]
    second_selected = second[5]["selected_development_features"]
    assert first_selected == second_selected == ["mom_120_20", "factor_score"]
    assert first[4]["status"] == "INVALID_POST_SELECTION"


def test_fixed_slot_costs_charge_buy_and_terminal_sell():
    spec = _spec()
    dates = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    rows = []
    for arm in ("F", "FC", "FMA"):
        for position, decision_dt in enumerate(dates):
            rows.append(
                {
                    "decision_dt": decision_dt,
                    "arm": arm,
                    "symbol": "A",
                    "membership_role": "new_entry" if position == 0 else "retained",
                    "entry_tradable": True,
                    "fwd_5d_open_return": 0.10,
                }
            )
    memberships = pd.DataFrame(rows)
    bundle = _empty_bundle(memberships=memberships)

    weekly = stage2.build_arm_weekly_paths(spec, bundle)
    factor = weekly[weekly["arm"].eq("F")].reset_index(drop=True)

    deployable = 0.995
    assert factor.loc[0, "gross_return"] == pytest.approx(deployable * 0.10 / 50)
    assert factor.loc[0, "cost_40bps"] == pytest.approx(deployable * 0.0015 / 50)
    assert factor.loc[1, "cost_40bps"] == pytest.approx(deployable * 0.0025 / 50)
    assert factor["buy_count"].tolist() == [1, 0]
    assert factor["terminal_sell_count"].tolist() == [0, 1]


def test_path_slot_metadata_is_computed_inside_each_segment():
    spec = _spec()
    rows = []
    for arm in ("F", "FC", "FMA"):
        for decision_dt, slots in [
            (pd.Timestamp("2023-12-29"), 50),
            (pd.Timestamp("2024-01-05"), 10),
        ]:
            rows.append(
                {
                    "decision_dt": decision_dt,
                    "arm": arm,
                    "slots": slots,
                    "slot_fill_rate": slots / 50,
                    "observable_target_slot_rate": slots / 50,
                    "gross_return": 0.0,
                    "net_return_0bps": 0.0,
                    "net_return_20bps": 0.0,
                    "net_return_40bps": 0.0,
                    "net_return_60bps": 0.0,
                }
            )

    summary = stage2.summarize_arm_paths(spec, pd.DataFrame(rows))
    factor = summary[summary["path"].eq("F") & summary["metric"].eq("net_return_40bps")].set_index("segment")

    assert factor.loc["FULL", "mean_slots"] == pytest.approx(30)
    assert factor.loc["DEVELOPMENT", "mean_slots"] == pytest.approx(50)
    assert factor.loc["HISTORICAL_VALIDATION", "mean_slots"] == pytest.approx(10)


def test_full_calendar_turnover_includes_empty_weeks():
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    frame = pd.DataFrame(
        [
            {"decision_dt": dates[0], "symbol": "A"},
            {"decision_dt": dates[3], "symbol": "A"},
        ]
    )

    result = stage2.full_calendar_turnover(frame, dates)

    assert result["presence_weeks"] == 2
    assert result["full_calendar_jaccard_turnover"] == pytest.approx(2 / 3)
    assert result["median_presence_run_weeks"] == 1


def test_weekly_stats_and_holm_keep_named_endpoints():
    spec = _spec()
    stats = stage2.weekly_stats([0.01, 0.02, -0.01, 0.03, 0.0], spec, "unit")
    adjusted = stage2.holm_adjust({"a": 0.01, "b": 0.03, "missing": None})
    planned_family = stage2.holm_adjust({"a": 0.01, "missing": None}, family_size=2)

    assert stats["n_weeks"] == 5
    assert stats["mean"] == pytest.approx(0.01)
    assert stats["bootstrap_ci_low"] <= stats["bootstrap_ci_high"]
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.03)
    assert adjusted["missing"] is None
    assert planned_family["a"] == pytest.approx(0.02)


def test_execute_experiment_persists_failure(tmp_path: Path):
    spec = _spec()
    records = {row["id"]: stage2.ExperimentRecord(row["id"], row["deliverable"]) for row in spec["planned_experiments"]}

    with pytest.raises(RuntimeError, match="boom"):
        stage2.execute_experiment(
            spec,
            records,
            tmp_path,
            "S2E00_INPUT_AND_INTEGRITY",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            lambda _result: {},
        )

    registry = stage2.read_json(tmp_path / "experiment_registry.json")
    first = registry["experiments"][0]
    assert first["status"] == "FAILED"
    assert first["error_type"] == "RuntimeError"
    assert first["error_message"] == "boom"


def test_published_inventory_detects_mutation(tmp_path: Path):
    identity = "a" * 64
    publish = tmp_path / f"STAGE2_{identity}"
    publish.mkdir()
    payload = publish / "result.txt"
    payload.write_text("stable", encoding="utf-8")
    records = [
        stage2.ExperimentRecord(experiment_id, experiment_id, status="COMPLETED").as_dict()
        for experiment_id in stage2.EXPECTED_EXPERIMENT_IDS
    ]
    stage2.write_json(
        publish / "experiment_registry.json",
        {"study_id": "test", "mode": "EXPLORATORY_ONLY", "experiments": records},
    )
    stage2.write_json(publish / "decision_matrix.json", {"hypothesis_decisions": []})
    inventory = stage2.output_inventory(publish)
    stage2.write_manifest_with_digest(
        publish,
        {
            "study_id": "test",
            "study_identity": identity,
            "confirmation_chain": "NOT_STARTED",
            "experiment_registry": records,
            "hypothesis_decisions": [],
            "output_inventory": inventory,
        },
    )

    result = stage2.verify_published_directory(publish, verify_current_identity=False)
    assert result["status"] == "VERIFIED_INTEGRITY_ONLY"

    payload.write_text("mutated", encoding="utf-8")
    with pytest.raises(stage2.Stage2Error, match="size changed|digest changed"):
        stage2.verify_published_directory(publish, verify_current_identity=False)


def test_verifier_proves_current_identity_and_classifies_superseded(tmp_path: Path):
    source_path = tmp_path / "study.py"
    source_path.write_text("stable source\n", encoding="utf-8")
    spec_path = tmp_path / "study.json"
    spec = {
        "study_id": "unit",
        "mode": "EXPLORATORY_ONLY",
        "study_type": "RETROSPECTIVE_FALSIFICATION_ONLY",
        "confirmation_chain": "NOT_STARTED",
        "revision_history": [],
        "frozen_stage1_inputs": [],
    }
    stage2.write_json(spec_path, spec)
    identity = stage2.study_identity(spec, source_path)
    publish = tmp_path / f"STAGE2_{identity}"
    publish.mkdir()
    records = [
        stage2.ExperimentRecord(experiment_id, experiment_id, status="COMPLETED").as_dict()
        for experiment_id in stage2.EXPECTED_EXPERIMENT_IDS
    ]
    stage2.write_json(
        publish / "experiment_registry.json",
        {"study_id": "unit", "mode": "EXPLORATORY_ONLY", "experiments": records},
    )
    stage2.write_json(publish / "decision_matrix.json", {"hypothesis_decisions": []})
    inventory = stage2.output_inventory(publish)
    manifest = {
        "study_id": "unit",
        "study_identity": identity,
        "mode": spec["mode"],
        "study_type": spec["study_type"],
        "confirmation_chain": "NOT_STARTED",
        "live_trading_authorized": False,
        "spec": {
            "physical_sha256": stage2.sha256_file(spec_path),
            "canonical_sha256": hashlib.sha256(stage2.canonical_json(spec)).hexdigest(),
        },
        "source": {"sha256": stage2.sha256_file(source_path)},
        "frozen_inputs": [],
        "experiment_registry": records,
        "hypothesis_decisions": [],
        "output_inventory": inventory,
    }
    stage2.write_manifest_with_digest(publish, manifest)

    current = stage2.verify_published_directory(
        publish,
        spec=spec,
        spec_path=spec_path,
        source_path=source_path,
    )
    assert current["status"] == "VERIFIED_CURRENT"
    assert current["manifest_digest_status"] == "VERIFIED"

    revised = copy.deepcopy(spec)
    revised["revision_history"] = [{"revision": 2, "superseded_identity": identity}]
    stage2.write_json(spec_path, revised)
    superseded = stage2.verify_published_directory(
        publish,
        spec=revised,
        spec_path=spec_path,
        source_path=source_path,
    )
    assert superseded["status"] == "VERIFIED_SUPERSEDED"

    tampered = copy.deepcopy(manifest)
    tampered["source"]["sha256"] = "0" * 64
    stage2.write_manifest_with_digest(publish, tampered)
    stage2.write_json(spec_path, spec)
    with pytest.raises(stage2.Stage2Error, match="source digest"):
        stage2.verify_published_directory(
            publish,
            spec=spec,
            spec_path=spec_path,
            source_path=source_path,
        )
