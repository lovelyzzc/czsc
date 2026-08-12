"""Delay5 工程镜像口径回归测试。"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_daily_scan_native_type_boundaries() -> None:
    """生产扫描使用 Rust 枚举转换，并把原生 FeatureSnapshot 传给评分函数。"""

    path = SCRIPTS_DIR.parent / ".cursor" / "skills" / "surge-regime-stock-picker" / "scripts" / "daily_scan.py"
    spec = importlib.util.spec_from_file_location("surge_daily_scan_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._regime_name(5) == mirror.tr.REGIME_CN[mirror.tr.Regime.UpwardDeparture]
    assert module._regime_name(8) == mirror.tr.REGIME_CN[mirror.tr.Regime.Acceleration]
    assert {5, 6, 7, 8} == module.UPTREND_FAMILY
    assert module.surge_score(None) == 0.0
    assert module.reversion_score(None, 5) == 0.0

    source = path.read_text(encoding="utf-8")
    assert "surge_score(feat_snapshot)" in source
    assert "reversion_score(feat_snapshot, down_days)" in source


def test_daily_scan_integer_regime_reaches_legacy_surge_without_delay5(monkeypatch) -> None:
    """整数状态必须进入 legacy surge，但该入口不得再调用或写入 delay5。"""

    path = SCRIPTS_DIR.parent / ".cursor" / "skills" / "surge-regime-stock-picker" / "scripts" / "daily_scan.py"
    spec = importlib.util.spec_from_file_location("surge_daily_scan_membership_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    states = [SimpleNamespace(regime=0, feats=None) for _ in range(12)]
    states[-1].regime = 5
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "amount": [200_000.0]})
    calls = []

    monkeypatch.setattr(module.tr, "load_stock", lambda _path: frame)
    monkeypatch.setattr(module.tr, "iter_states", lambda *_args, **_kwargs: states)
    monkeypatch.setattr(module, "surge_onset", lambda *_args, **_kwargs: calls.append(states) or False)
    monkeypatch.setattr(module, "reversion_onset", lambda *_args, **_kwargs: False)

    result = module._scan_one("ignored.parquet")

    assert calls and all(value is states for value in calls)
    assert result is None


def test_legacy_main_cannot_run_or_write_delay5() -> None:
    """旧十日观察池入口不得重新接入 delay5 检测或前向日志。"""

    path = SCRIPTS_DIR.parent / ".cursor" / "skills" / "surge-regime-stock-picker" / "scripts" / "daily_scan.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = set()
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)

    assert "_report_experimental" not in calls
    assert "detect_delay5" not in calls
