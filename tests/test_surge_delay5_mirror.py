"""Delay5 工程镜像口径回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_delay5_mirror_check as mirror  # noqa: E402


def test_signal_gate_matches_current_rust_default() -> None:
    """锁定 Lte(0.8) / Gte(3.0) / no-above-zg / ret20 Gte(8.0)。"""

    frame = pd.DataFrame(
        {
            "sig_vol_ratio": [0.8, 0.81, 0.8, 0.8, 0.8, None],
            "sig_ma_spread_pct": [3.0, 3.0, 2.99, 3.0, 3.0, 3.0],
            "sig_above_zg": [0, 1, 1, 1, 0, 1],
            "sig_ret20": [8.0, 8.0, 8.0, 7.99, 8.0, 8.0],
        }
    )

    assert mirror._signal_gate(frame).tolist() == [True, False, False, False, True, False]


def test_build_verdict_requires_matched_values() -> None:
    """集合一致但字段有差异时不得误报 PASS。"""

    common = {
        "total_live": 18,
        "total_dump": 18,
        "total_matched": 18,
        "total_set_mismatches": 0,
        "gates_ok": True,
        "gate_note": "all ok",
    }

    assert mirror.build_verdict(total_value_diffs=0, **common).startswith("PASS")
    assert mirror.build_verdict(total_value_diffs=1, **common).startswith("ATTENTION")
