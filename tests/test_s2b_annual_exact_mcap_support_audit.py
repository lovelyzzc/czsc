"""S2b 年度精确市值支持集审计纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_annual_exact_mcap_support_audit as audit  # noqa: E402


def test_recompute_excess_uses_median_and_minimum_support() -> None:
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert audit.recompute_excess(10.0, returns) == 7.0
    assert np.isnan(audit.recompute_excess(10.0, returns[:4]))


def test_run_audit_rejects_unconfigured_year_before_loading_data() -> None:
    with pytest.raises(ValueError, match="year must be one of"):
        audit.run_audit(2030)
