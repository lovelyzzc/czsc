import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import surge_review as sr  # noqa: E402


def test_select_diverse_rows_prefers_coverage_before_priority():
    rows = [
        {"代码": "A", "行业": "电子", "启动方式": "确认", "当前状态": "主升", "优先级": 99},
        {"代码": "B", "行业": "电子", "启动方式": "确认", "当前状态": "主升", "优先级": 98},
        {"代码": "C", "行业": "医药", "启动方式": "埋伏", "当前状态": "离开", "优先级": 80},
        {"代码": "D", "行业": "机械", "启动方式": "确认", "当前状态": "主升", "优先级": 70},
    ]

    selected = sr.select_diverse_rows(
        rows,
        3,
        priority_key="优先级",
        diversity_keys=("行业", "启动方式", "当前状态"),
        code_key="代码",
    )

    assert [row["代码"] for row in selected] == ["A", "C", "D"]


def test_select_diverse_rows_is_bounded_deterministic_and_non_mutating():
    rows = [
        {"代码": "B", "行业": "电子", "优先级": 80},
        {"代码": "A", "行业": "电子", "优先级": 80},
    ]
    original = list(rows)

    selected = sr.select_diverse_rows(
        rows,
        1,
        priority_key="优先级",
        diversity_keys=("行业",),
        code_key="代码",
    )

    assert [row["代码"] for row in selected] == ["A"]
    assert rows == original
    assert sr.select_diverse_rows(rows, 0, priority_key="优先级", diversity_keys=("行业",), code_key="代码") == []
