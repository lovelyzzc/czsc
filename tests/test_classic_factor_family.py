from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "classic_factor_family.py"
SPEC = importlib.util.spec_from_file_location("classic_factor_family", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_raw_factor_signs_are_frozen() -> None:
    frame = pd.DataFrame(
        {
            "pb": [1.0, 2.0],
            "pe_ttm": [10.0, 20.0],
            "turnover_rate": [1.0, 2.0],
            "volume_ratio": [1.0, 2.0],
            "log_free_float_mcap": [10.0, 11.0],
        }
    )
    assert study.raw_factor(frame, "VALUE_PB").iloc[0] > study.raw_factor(frame, "VALUE_PB").iloc[1]
    assert study.raw_factor(frame, "EARNINGS_YIELD").iloc[0] > study.raw_factor(frame, "EARNINGS_YIELD").iloc[1]
    assert study.raw_factor(frame, "LOW_TURNOVER").iloc[0] > study.raw_factor(frame, "LOW_TURNOVER").iloc[1]
    assert study.raw_factor(frame, "NORMAL_VOLUME").iloc[0] > study.raw_factor(frame, "NORMAL_VOLUME").iloc[1]
    assert study.raw_factor(frame, "SMALL_CAP").iloc[0] > study.raw_factor(frame, "SMALL_CAP").iloc[1]


def test_holm_adjust_is_named_and_monotone() -> None:
    adjusted = study.holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}


def test_invalid_value_inputs_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "pb": [0.0],
            "pe_ttm": [-1.0],
            "turnover_rate": [0.0],
            "volume_ratio": [0.0],
            "log_free_float_mcap": [10.0],
        }
    )
    for factor in ("VALUE_PB", "EARNINGS_YIELD", "LOW_TURNOVER", "NORMAL_VOLUME"):
        assert np.isnan(study.raw_factor(frame, factor).iloc[0])


def test_protocol_freezes_five_candidates_and_forbids_live() -> None:
    protocol = study.load_and_verify_protocol()
    assert len(protocol["family"]) == 5
    assert protocol["statistics"]["development_multiplicity"].startswith("Holm-adjusted")
    assert protocol["live_trading_authorized"] is False
