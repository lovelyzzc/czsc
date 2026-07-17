from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_features_v2 as features_v2  # noqa: E402
import xs_chan_research_v2 as research_v2  # noqa: E402


def _synthetic_inputs(periods: int = 330) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
    schedule = features_v2.build_weekly_schedule(calendar, dates[-1])
    suspension_dt = schedule.loc[schedule["decision_dt"] >= dates[260], "decision_dt"].iloc[0]
    st_dt = schedule["decision_dt"].iloc[-1]
    st_position = int(dates.get_loc(st_dt))
    symbols = ("A.SZ", "B.SZ")
    security = pd.DataFrame(
        {
            "ts_code": symbols,
            "name": ["甲股份", "乙股份"],
            "board": ["MAIN", "MAIN"],
            "list_date": [dates[0], dates[0]],
            "delist_date": [pd.NaT, pd.NaT],
            "list_status": ["L", "L"],
        }
    )

    rows = []
    for symbol_no, symbol in enumerate(symbols):
        for index, dt in enumerate(dates):
            if symbol == "A.SZ" and dt == suspension_dt:
                continue
            close = 10.0 + symbol_no + index * 0.01
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": dt,
                    "close": close,
                    "vol": 1_000_000.0,
                    "amount": 60_000.0,
                }
            )
    raw = pd.DataFrame(rows)
    factor = raw[["ts_code", "trade_date"]].assign(adj_factor=1.0)
    basic = raw[["ts_code", "trade_date"]].assign(free_share=100_000.0)
    namechange = pd.DataFrame(
        {
            "ts_code": ["A.SZ", "A.SZ"],
            "name": ["甲股份", "*ST甲"],
            "effective_from": [dates[0], st_dt],
            "effective_to": [dates[st_position - 1], pd.NaT],
        }
    )
    industry = pd.DataFrame(
        {
            "ts_code": ["A.SZ", "B.SZ"],
            "industry_code": ["I01", "I02"],
            "classification_version": ["SW2021", "SW2021"],
            "effective_from": [dates[0], dates[0]],
            "effective_to": [pd.NaT, pd.NaT],
        }
    )
    return {
        "calendar": calendar,
        "security_master": security,
        "namechange": namechange,
        "raw_daily": raw,
        "adj_factor": factor,
        "daily_basic": basic,
        "industry_membership": industry,
        "suspension_dt": suspension_dt,
        "st_dt": st_dt,
    }


def _assemble(inputs: dict[str, pd.DataFrame]) -> features_v2.FeatureFrames:
    return features_v2.assemble_feature_frames(
        calendar=inputs["calendar"],
        security_master=inputs["security_master"],
        namechange=inputs["namechange"],
        raw_daily=inputs["raw_daily"],
        adj_factor=inputs["adj_factor"],
        daily_basic=inputs["daily_basic"],
        industry_membership=inputs["industry_membership"],
        eligibility_spec=research_v2.EligibilitySpec(250, 57, 50_000.0, ("STAR", "BSE")),
        target_size=1,
    )


def test_dense_session_features_carry_price_zero_amount_and_do_not_fill_pit():
    inputs = _synthetic_inputs()
    outputs = _assemble(inputs)
    decision = pd.Timestamp(inputs["suspension_dt"])
    row = outputs.features.set_index(["symbol", "dt"]).loc[("A.SZ", decision)]

    raw_a = inputs["raw_daily"][inputs["raw_daily"]["ts_code"].eq("A.SZ")].set_index("trade_date")
    prior_close = float(raw_a.loc[raw_a.index < decision, "close"].iloc[-1])
    assert row["adjusted_close"] == pytest.approx(prior_close)
    assert row["adv20"] == pytest.approx(57_000.0)
    assert row["valid_sessions_60"] == 59
    assert pd.isna(row["free_float_mcap"])
    assert row["eligible"] == np.bool_(False)
    assert "missing_pit_or_factor" in row["ineligibility_reasons"]

    dense = pd.Series(index=pd.DatetimeIndex(inputs["calendar"]["trade_date"]), dtype=float)
    dense.loc[raw_a.index] = raw_a["close"]
    dense = dense.ffill()
    expected_momentum = np.log(dense.shift(20).loc[decision]) - np.log(dense.shift(120).loc[decision])
    expected_lowvol = -np.log(dense).diff().rolling(60, min_periods=60).std(ddof=1).loc[decision]
    assert row["mom_120_20"] == pytest.approx(expected_momentum)
    assert row["lowvol_60"] == pytest.approx(expected_lowvol)


def test_schedule_starts_at_first_sufficient_pool_and_st_is_effective_dated():
    inputs = _synthetic_inputs()
    outputs = _assemble(inputs)
    indexed = outputs.features.set_index(["symbol", "dt"])
    mature_dt = outputs.schedule.loc[
        (outputs.schedule["decision_dt"] > inputs["suspension_dt"])
        & (outputs.schedule["decision_dt"] < inputs["st_dt"]),
        "decision_dt",
    ].iloc[0]

    normal = indexed.loc[("A.SZ", mature_dt)]
    st_row = indexed.loc[("A.SZ", pd.Timestamp(inputs["st_dt"]))]
    raw_schedule = features_v2.build_weekly_schedule(inputs["calendar"], inputs["raw_daily"]["trade_date"].max())
    counts = outputs.features.groupby("dt")["eligible"].sum()
    assert outputs.schedule["decision_dt"].iloc[0] > raw_schedule["decision_dt"].iloc[0]
    assert counts.min() >= 1
    assert normal["market_sessions_since_listing"] >= 250
    assert normal["eligible"] == np.bool_(True)
    assert st_row["is_st"] == np.bool_(True)
    assert st_row["eligible"] == np.bool_(False)
    assert "st" in st_row["ineligibility_reasons"]


def test_namechange_history_gap_is_unknown_and_never_filled_from_current_master_name():
    inputs = _synthetic_inputs()
    raw_schedule = features_v2.build_weekly_schedule(inputs["calendar"], inputs["raw_daily"]["trade_date"].max())
    gap_dt = raw_schedule["decision_dt"].iloc[-2]
    dates = pd.DatetimeIndex(inputs["calendar"]["trade_date"])
    gap_position = int(dates.get_loc(gap_dt))
    changes = inputs["namechange"].copy()
    changes.loc[changes["name"].eq("甲股份"), "effective_to"] = dates[gap_position - 1]
    inputs["namechange"] = changes

    outputs = _assemble(inputs)
    row = outputs.features.set_index(["symbol", "dt"]).loc[("A.SZ", gap_dt)]
    assert pd.isna(row["is_st"])
    assert row["eligible"] == np.bool_(False)
    assert "st" in row["ineligibility_reasons"]


def test_later_week_below_target_size_fails_closed():
    inputs = _synthetic_inputs()
    changes = inputs["namechange"].copy()
    changes = pd.concat(
        [
            changes,
            pd.DataFrame(
                {
                    "ts_code": ["B.SZ"],
                    "name": ["*ST乙"],
                    "effective_from": [inputs["st_dt"]],
                    "effective_to": [pd.NaT],
                }
            ),
        ],
        ignore_index=True,
    )
    inputs["namechange"] = changes
    with pytest.raises(features_v2.FeatureAssemblyError, match="fell below target_size"):
        _assemble(inputs)


def test_assembly_is_deterministic_and_prefix_invariant():
    inputs = _synthetic_inputs()
    full = _assemble(inputs)
    cutoff_exec = full.schedule.iloc[-6]["exec_dt"]
    prefix_inputs = dict(inputs)
    for name in ("raw_daily", "adj_factor", "daily_basic"):
        prefix_inputs[name] = inputs[name][inputs[name]["trade_date"] <= cutoff_exec].copy()
    prefix = _assemble(prefix_inputs)
    cutoff_decision = prefix.schedule["decision_dt"].max()

    expected = full.features[full.features["dt"] <= cutoff_decision].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix.features, expected)
    pd.testing.assert_frame_equal(
        prefix.all_adv,
        full.all_adv[full.all_adv["dt"] <= cutoff_decision].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        _assemble(inputs).features,
        full.features,
    )


def test_content_addressed_publish_is_exclusive_and_hashes_every_output(tmp_path: Path):
    frames = _assemble(_synthetic_inputs())
    identity = {
        "schema_version": features_v2.SCHEMA_VERSION,
        "data_evidence_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
    }
    result = features_v2.publish_feature_frames(frames, identity, tmp_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.output_dir.name == f"{features_v2.OUTPUT_PREFIX}{result.run_id}"
    assert manifest["identity"] == {**identity, "feature_coverage": manifest["coverage"]}
    assert manifest["coverage"]["first_decision_dt"] == str(frames.schedule["decision_dt"].iloc[0].date())
    assert manifest["coverage"]["last_decision_dt"] == str(frames.schedule["decision_dt"].iloc[-1].date())
    assert manifest["coverage"]["eligible_count_min"] >= frames.target_size
    assert manifest["outputs"]["features"]["rows"] == len(frames.features)
    for entry in manifest["outputs"].values():
        path = result.output_dir / entry["path"]
        assert features_v2._sha256_file(path) == entry["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        features_v2.publish_feature_frames(frames, identity, tmp_path)


def test_publish_rejects_any_schedule_feature_adv_date_misalignment(tmp_path: Path):
    frames = _assemble(_synthetic_inputs())
    last_decision = frames.schedule["decision_dt"].iloc[-1]
    misaligned = features_v2.FeatureFrames(
        schedule=frames.schedule,
        features=frames.features,
        all_adv=frames.all_adv[frames.all_adv["dt"].ne(last_decision)].copy(),
        target_size=frames.target_size,
    )

    with pytest.raises(features_v2.FeatureAssemblyError, match="exactly the same dates"):
        features_v2.publish_feature_frames(misaligned, {"case": "misaligned"}, tmp_path)


def test_formal_entry_rejects_a_bundle_whose_data_gate_is_not_green(tmp_path: Path):
    bundle = tmp_path / "not_ready"
    bundle.mkdir()
    with pytest.raises(features_v2.FeatureAssemblyError, match="not forward-start ready"):
        features_v2.build_feature_artifacts(bundle, tmp_path / "output")
