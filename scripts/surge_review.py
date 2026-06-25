"""主升浪扫描的人工复盘抽样工具。

这里只控制展示规模，不改变完整候选日志、前向样本或组合回测口径。
"""

from collections.abc import Iterable
from typing import Any


def select_diverse_rows(
    rows: Iterable[dict[str, Any]],
    limit: int,
    *,
    priority_key: str,
    diversity_keys: tuple[str, ...],
    code_key: str,
) -> list[dict[str, Any]]:
    """从既定候选中抽取小规模、确定性的多样化复盘样本。

    每轮优先选择能带来更多新行业/新启动方式/新状态的行；覆盖度相同再按现有
    priority 排序，代码仅用于平手时保证结果稳定。priority 在这里只是展示排序，
    不是收益预测器。
    """
    if limit <= 0:
        return []

    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    seen: dict[str, set[Any]] = {key: set() for key in diversity_keys}

    while remaining and len(selected) < limit:

        def sort_key(row: dict[str, Any]) -> tuple:
            novelty = sum(
                1 for key in diversity_keys if row.get(key) not in (None, "") and row.get(key) not in seen[key]
            )
            priority = float(row.get(priority_key) or 0)
            return (-novelty, -priority, str(row.get(code_key) or ""))

        remaining.sort(key=sort_key)
        chosen = remaining.pop(0)
        selected.append(chosen)
        for key in diversity_keys:
            value = chosen.get(key)
            if value not in (None, ""):
                seen[key].add(value)

    return selected
