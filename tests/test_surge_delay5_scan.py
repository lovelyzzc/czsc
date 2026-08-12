"""Delay5 专用每日扫描的目标日与空日志回归测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCAN_PATH = REPO / ".cursor" / "skills" / "surge-delay5-stock-picker" / "scripts" / "delay5_scan.py"


def _load_scan_module():
    name = "surge_delay5_scan_test"
    spec = importlib.util.spec_from_file_location(name, SCAN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_target_date_requires_active_manifest_schema(tmp_path: Path, monkeypatch) -> None:
    module = _load_scan_module()
    monkeypatch.setattr(module.tr, "DATA_DIR", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"a_stock_daily_qfq_active_snapshot_v1","safe_end_date":"20260810"}',
        encoding="utf-8",
    )

    assert module._target_date_from_manifest() == pd.Timestamp("2026-08-10")

    manifest.write_text('{"schema":"wrong","safe_end_date":"20260810"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected active manifest schema"):
        module._target_date_from_manifest()


def test_dedicated_scan_rejects_stale_stock_and_uses_shared_detector(monkeypatch) -> None:
    module = _load_scan_module()
    target = pd.Timestamp("2026-08-10")
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "dt": pd.to_datetime(["2026-08-07", "2026-08-10"]),
            "amount": [100_000.0, 200_000.0],
        }
    )
    states = [SimpleNamespace(regime=5)]
    calls: list[object] = []

    monkeypatch.setattr(module.tr, "load_stock", lambda _path: frame)
    monkeypatch.setattr(module.tr, "iter_states", lambda *_args, **_kwargs: states)
    monkeypatch.setattr(
        module.sl,
        "detect_delay5",
        lambda value: calls.append(value) or {"dec_dt": target, "sl_pct": 10.0, "priority": 1.0},
    )

    hit = module._scan_delay5_one(("ignored.parquet", target.isoformat()))
    assert calls == [states]
    assert hit is not None
    assert hit["代码"] == "000001.SZ"
    assert hit["成交额亿"] == 2.0

    stale = frame.iloc[:1].copy()
    monkeypatch.setattr(module.tr, "load_stock", lambda _path: stale)
    calls.clear()
    assert module._scan_delay5_one(("ignored.parquet", target.isoformat())) is None
    assert calls == []


def test_empty_scan_log_counts_as_completed_day(tmp_path: Path, monkeypatch) -> None:
    module = _load_scan_module()
    monkeypatch.setattr(module.legacy, "OUTPUT_DIR", tmp_path)

    path = module._write_empty_scan_log(pd.Timestamp("2026-08-10"))
    progress = module._forward_progress()
    assert path.exists()
    assert list(pd.read_parquet(path).columns) == ["代码", "dec_dt", "可操作"]
    assert progress["days"] == 1
    assert progress["total"] == 0
    assert progress["first"] == pd.Timestamp("2026-08-10").date()
    assert progress["last"] == pd.Timestamp("2026-08-10").date()


def test_market_state_must_match_manifest_target(monkeypatch) -> None:
    """市场面板陈旧时必须在写日志和生成候选前 fail closed。"""

    module = _load_scan_module()
    state = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2026-08-10"]),
            "high20_ratio": [0.2],
            "ew_index_above_ma20": [1],
        }
    )
    monkeypatch.setattr(module.legacy.sl, "build_live_panel", lambda: pd.DataFrame())
    monkeypatch.setattr(module.legacy.sl, "live_market_state", lambda _panel: state)
    monkeypatch.setattr(module.legacy.sl, "market_gate_open", lambda _row: True)
    monkeypatch.setattr(module.legacy.sl, "append_market_state_log", lambda _state: None)

    module.legacy._report_experimental([], {}, {}, False, expected_dec_dt=pd.Timestamp("2026-08-10"))

    state.loc[0, "dt"] = pd.Timestamp("2026-08-07")
    with pytest.raises(RuntimeError, match="market state is stale"):
        module.legacy._report_experimental([], {}, {}, False, expected_dec_dt=pd.Timestamp("2026-08-10"))


def test_detect_delay5_passes_nan_to_native_priority_when_stop_missing(monkeypatch) -> None:
    module = _load_scan_module()
    states = [
        SimpleNamespace(
            regime=5,
            feats=None,
            sl_ref=float("nan"),
            zd=float("nan"),
            close=10.0,
            dt=pd.Timestamp("2026-08-04") + pd.Timedelta(days=i),
        )
        for i in range(7)
    ]
    seen: list[float] = []

    monkeypatch.setattr(module.sl.tr, "surge_onset", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module.sl.tr, "surge_score", lambda _features: 0.0)
    monkeypatch.setattr(
        module.sl.tr,
        "priority_score",
        lambda _score, stop, _freshness, _regime: seen.append(stop) or 0.0,
    )

    result = module.sl.detect_delay5(states)

    assert result is not None and result["sl_pct"] is None
    assert len(seen) == 1 and pd.isna(seen[0])
