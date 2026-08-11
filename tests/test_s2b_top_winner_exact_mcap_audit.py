"""S2b 最大赢家精确流通市值审计纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_top_winner_exact_mcap_audit as audit  # noqa: E402


def test_derive_exact_circ_mv_uses_turnover_percent() -> None:
    assert audit.derive_exact_circ_mv(close=10, volume=1_000_000, turnover_pct=2) == 500_000_000
    assert np.isnan(audit.derive_exact_circ_mv(close=10, volume=1_000_000, turnover_pct=0))


def test_exact_caliper_mask_is_symmetric() -> None:
    controls = pd.Series({"a": 60.0, "b": 67.0, "c": 149.0, "d": 151.0, "e": np.nan})
    mask = audit.exact_caliper_mask(100.0, controls, ratio=1.5)
    assert mask.to_dict() == {"a": False, "b": True, "c": True, "d": False, "e": False}


def test_baostock_code_conversion() -> None:
    assert audit.baostock_code("600000.SH") == "sh.600000"
    assert audit.baostock_code("000001.SZ") == "sz.000001"
    with pytest.raises(ValueError, match="unsupported exchange"):
        audit.baostock_code("920001.BJ")
