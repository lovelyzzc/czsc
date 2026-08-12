from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "fsm_weekly_reversal_discovery.py"
SPEC = importlib.util.spec_from_file_location("fsm_weekly_reversal_discovery", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_holm_adjust_is_monotone_and_named() -> None:
    adjusted = study.holm_adjust({"b": 0.02, "a": 0.01, "c": 0.5})
    assert set(adjusted) == {"a", "b", "c"}
    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.04
    assert adjusted["c"] == 0.5


def test_endpoint_map_counts_entry_as_day_one() -> None:
    calendar = pd.date_range("2026-01-01", periods=8, freq="B")
    mapping = study._endpoint_map(calendar, 5)
    # Decision Thursday -> entry Friday, and H5 closes the following Thursday.
    assert mapping[pd.Timestamp("2026-01-01")] == pd.Timestamp("2026-01-08")


def test_blind_artifact_rejects_outcome_columns() -> None:
    frame = pd.DataFrame({"trade_id": ["x"], "future_return": [0.1]})
    try:
        study.assert_blind(frame, "toy")
    except RuntimeError as exc:
        assert "outcome-like" in str(exc)
    else:
        raise AssertionError("outcome field was not rejected")


def test_cell_id_is_stable() -> None:
    assert study.cell_id(2, 5, 7) == "R2_D05_07"
