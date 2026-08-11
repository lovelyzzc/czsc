"""S2b 匹配对照审计的纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_matched_control_audit as audit  # noqa: E402


def test_censored_tail_mask_only_marks_short_max_hold() -> None:
    frame = pd.DataFrame(
        {
            "exit_reason": ["max_hold", "max_hold", "state", "trail18"],
            "hold_days": [58, 59, 10, 20],
        }
    )

    assert audit.censored_tail_mask(frame).tolist() == [True, False, False, False]


def test_summarize_matched_excess_clusters_by_decision_date() -> None:
    frame = pd.DataFrame(
        {
            "dec_dt": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"]),
            "matched_excess_pct": [1.0, 3.0, -1.0, 2.0],
            "ret_gross_pct": [2.0, 4.0, 0.0, 3.0],
        }
    )

    summary = audit.summarize_matched_excess(frame, n_boot=100)

    assert summary["n_trades"] == 4
    assert summary["n_decision_dates"] == 3
    assert summary["mean_excess_pct"] == 1.25
    assert summary["median_excess_pct"] == 1.5
    assert summary["daily_hac"]["n"] == 3
