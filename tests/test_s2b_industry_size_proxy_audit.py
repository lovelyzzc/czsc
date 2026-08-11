"""S2b 行业与规模代理审计的纯函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import s2b_industry_size_proxy_audit as audit  # noqa: E402


def test_symbol_converters_cover_three_exchanges() -> None:
    assert audit.tencent_quote_code("600000.SH") == "sh600000"
    assert audit.tencent_quote_code("000001.SZ") == "sz000001"
    assert audit.tencent_quote_code("920001.BJ") == "bj920001"
    assert audit.baostock_symbol("sh.600000") == "600000.SH"
    assert audit.baostock_symbol("sz.000001") == "000001.SZ"
    assert audit._quote_float("1.25", scale=100) == 125
    assert np.isnan(audit._quote_float("--"))


def test_canonical_industry_sha_is_independent_of_row_and_column_order() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "B.SZ",
                "code": "sz.B",
                "updateDate": "2025-01-01",
                "industry": "B",
                "industryClassification": "证监会行业分类",
            },
            {
                "symbol": "A.SH",
                "code": "sh.A",
                "updateDate": "2025-01-01",
                "industry": "A",
                "industryClassification": "证监会行业分类",
            },
        ]
    )
    reordered = frame.iloc[::-1][list(reversed(frame.columns))]
    assert audit.canonical_industry_sha256(frame) == audit.canonical_industry_sha256(reordered)
    assert audit.INDUSTRY_SOURCE_COLUMNS == ("code", "updateDate", "industryClassification", "industry")


def test_nearest_control_symbols_is_deterministic_and_excludes_treated() -> None:
    sizes = pd.Series({"T": 100.0, "B": 105.0, "A": 95.0, "C": 160.0, "D": 50.0})

    selected = audit.nearest_control_symbols(
        sizes,
        treated_symbol="T",
        eligible_symbols=pd.Index(["D", "C", "B", "A", "T"]),
        k=3,
    )

    assert selected == ["B", "A", "C"]


def test_nearest_control_symbols_applies_symmetric_ratio_caliper() -> None:
    sizes = pd.Series({"T": 100.0, "A": 67.0, "B": 149.0, "C": 151.0, "D": 60.0})

    selected = audit.nearest_control_symbols(
        sizes,
        treated_symbol="T",
        eligible_symbols=pd.Index(sizes.index),
        k=10,
        caliper_ratio=1.5,
    )

    assert selected == ["B", "A"]
