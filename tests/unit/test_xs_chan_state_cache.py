from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_state_cache as state_cache  # noqa: E402


def _raw_bars(symbol: str, rows: int = 126) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    phase = np.linspace(0.0, 8.0 * np.pi, rows)
    close = 10.0 + np.linspace(0.0, 2.0, rows) + 0.8 * np.sin(phase)
    previous = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "vol": 1_000_000.0,
            "amount": 100_000.0,
            "pct_chg": (close / previous - 1.0) * 100.0,
        }
    )


def _write_source(directory: Path, symbol: str, rows: int = 126) -> Path:
    path = directory / f"{symbol}.parquet"
    _raw_bars(symbol, rows).to_parquet(path, index=False)
    return path


def _standard_frame(symbol: str = "000001.SZ", rows: int = 126) -> pd.DataFrame:
    raw = _raw_bars(symbol, rows).rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    raw["dt"] = pd.to_datetime(raw["dt"], format="%Y%m%d")
    return raw


def _causal_generator(frame: pd.DataFrame) -> pd.DataFrame:
    tail = frame.iloc[state_cache.WARMUP_BARS :]
    return state_cache.validate_projection(
        pd.DataFrame(
            {
                "symbol": frame["symbol"].iloc[0],
                "dt": tail["dt"].to_numpy(),
                "regime": np.arange(len(tail), dtype=np.int8) % 11,
            }
        )
    )


def _future_dependent_generator(frame: pd.DataFrame) -> pd.DataFrame:
    projection = _causal_generator(frame)
    projection["regime"] = np.int8(len(frame) % 11)
    return state_cache.validate_projection(projection)


def test_shim_loads_all_eleven_regimes_without_importing_czsc_top_level():
    previous = sys.modules.get("czsc")
    trend_regime = state_cache.get_trend_regime()

    assert {int(regime) for regime in trend_regime.ALL_REGIMES} == set(range(11))
    assert trend_regime.WARMUP_BARS == 120
    assert callable(trend_regime.format_standard_kline)
    assert sys.modules.get("czsc") is previous


@pytest.mark.parametrize("defect", ["missing_column", "duplicate_date"])
def test_source_input_contract_fails_closed(tmp_path: Path, defect: str):
    frame = _raw_bars("000001.SZ")
    if defect == "missing_column":
        frame = frame.drop(columns="amount")
    else:
        frame.loc[1, "trade_date"] = frame.loc[0, "trade_date"]
    path = tmp_path / "bad.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(state_cache.StateCacheInputError):
        state_cache.load_source_frame(path)


def test_only_first_observed_pct_change_may_be_null(tmp_path: Path):
    first_null = _raw_bars("920001.BJ")
    first_null.loc[0, "pct_chg"] = np.nan
    first_path = tmp_path / "first-null.parquet"
    first_null.to_parquet(first_path, index=False)

    loaded = state_cache.load_source_frame(first_path)
    assert loaded.loc[0, "pct_chg"] == 0.0
    assert loaded["pct_chg"].notna().all()

    later_null = _raw_bars("920002.BJ")
    later_null.loc[1, "pct_chg"] = np.nan
    later_path = tmp_path / "later-null.parquet"
    later_null.to_parquet(later_path, index=False)

    with pytest.raises(state_cache.StateCacheInputError, match="only on the earliest observed bar"):
        state_cache.load_source_frame(later_path)


def test_projection_is_strict_and_retains_every_state_from_zero_to_ten():
    dates = pd.bdate_range("2024-01-02", periods=11)
    snapshots = [SimpleNamespace(dt=dt, regime=regime, next_open=999.0) for regime, dt in enumerate(dates)]

    projection = state_cache.project_snapshots("000001.SZ", snapshots)

    assert tuple(projection.columns) == ("symbol", "dt", "regime")
    assert projection["regime"].dtype == np.dtype("int8")
    assert projection["regime"].tolist() == list(range(11))
    assert "next_open" not in projection


@pytest.mark.parametrize("regime", [-1, 11, 1.5])
def test_illegal_regime_fails_closed(regime: float):
    projection = pd.DataFrame({"symbol": ["000001.SZ"], "dt": [pd.Timestamp("2024-01-02")], "regime": [regime]})

    with pytest.raises(state_cache.StateProjectionError):
        state_cache.validate_projection(projection)


def test_generation_covers_short_and_long_sources_with_explicit_warmup():
    new_listing = _standard_frame(rows=5)
    exactly_warmup = _standard_frame(rows=120)
    first_eligible = _standard_frame(rows=121)

    new_projection = state_cache.generate_projection(new_listing)
    assert len(new_projection) == 5
    assert new_projection["regime"].eq(0).all()
    warmup_projection = state_cache.generate_projection(exactly_warmup)
    assert len(warmup_projection) == 120
    assert warmup_projection["regime"].eq(0).all()
    projection = state_cache.generate_projection(first_eligible)
    assert len(projection) == 121
    assert projection.iloc[:120]["regime"].eq(0).all()
    assert projection.iloc[-1]["dt"] == first_eligible.iloc[-1]["dt"]


def test_small_prefix_full_audit_has_exact_zero_difference():
    assert state_cache.DEFAULT_AUDIT_SYMBOLS == 100
    assert state_cache.DEFAULT_AUDIT_CHECKPOINTS == 20
    assert state_cache.DEFAULT_AUDIT_SYMBOLS * state_cache.DEFAULT_AUDIT_CHECKPOINTS == 2_000

    audit = state_cache.audit_frame_prefix_invariance(
        _standard_frame(rows=132),
        checkpoints=3,
        generator=_causal_generator,
    )

    assert audit["passed"] is True
    assert audit["comparison_count"] == 3
    assert audit["mismatch_count"] == 0
    assert audit["field_mismatches"] == {"symbol": 0, "dt": 0, "regime": 0}


def test_prefix_full_audit_detects_future_dependent_states():
    audit = state_cache.audit_frame_prefix_invariance(
        _standard_frame(rows=132),
        checkpoints=3,
        generator=_future_dependent_generator,
    )

    assert audit["passed"] is False
    assert audit["mismatch_count"] > 0
    assert audit["field_mismatches"]["regime"] > 0


def test_failed_prefix_audit_is_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_dir = tmp_path / "data"
    output_root = tmp_path / "output"
    data_dir.mkdir()
    _write_source(data_dir, "000001.SZ", rows=126)
    monkeypatch.setattr(
        state_cache,
        "run_prefix_audit",
        lambda *_args, **_kwargs: {"passed": False, "mismatch_count": 1},
    )

    with pytest.raises(state_cache.StateProjectionError, match="refusing to publish"):
        state_cache.build_state_cache(
            data_dir,
            output_root,
            workers=1,
            config=state_cache.StateCacheConfig(audit_symbols=1, audit_checkpoints=1),
        )

    assert not list(output_root.glob("CHAN_STATE_CACHE_*"))
    assert not list(output_root.glob(".tmp_*"))


def test_content_addressed_cache_manifest_audit_and_no_overwrite(tmp_path: Path):
    data_dir = tmp_path / "data"
    output_root = tmp_path / "output"
    data_dir.mkdir()
    source = _write_source(data_dir, "000001.SZ", rows=126)
    config = state_cache.StateCacheConfig(audit_symbols=1, audit_checkpoints=2, audit_seed=17)

    result = state_cache.build_state_cache(
        data_dir,
        output_root,
        workers=1,
        config=config,
    )

    assert result.audit_passed is True
    assert len(result.cache_id) == 64
    assert result.cache_dir.name == f"CHAN_STATE_CACHE_{result.cache_id}"
    assert sorted(path.name for path in result.cache_dir.iterdir()) == [
        "source_manifest.json",
        "state_audit.json",
        "state_audit.parquet",
        "states.parquet",
    ]
    projection = pd.read_parquet(result.projection_path)
    assert tuple(projection.columns) == ("symbol", "dt", "regime")
    assert len(projection) == 126
    assert projection.iloc[:120]["regime"].eq(0).all()
    assert projection["regime"].between(0, 10).all()

    audit_details = pd.read_parquet(result.audit_parquet_path)
    assert tuple(audit_details.columns) == state_cache.AUDIT_DETAIL_COLUMNS
    assert len(audit_details) == 2
    assert audit_details["passed"].all()
    assert audit_details[["symbol_mismatches", "dt_mismatches", "regime_mismatches"]].eq(0).all().all()

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["data_gate_ready"] is True
    assert audit["mismatch_count"] == 0
    assert audit["projection_contract"]["generation_rule"] == "every_non_empty_source"
    assert audit["projection_contract"]["whole_file_min_bars_filter_used"] is False
    assert set(audit["projection"]["regime_counts"]) == {str(regime) for regime in range(11)}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_id"] == result.cache_id
    assert manifest["projection"]["sha256"] == state_cache.sha256_file(result.projection_path)
    assert manifest["sources"][0]["sha256"] == state_cache.sha256_file(source)
    assert manifest["sources"][0]["source_rows"] == 126
    assert manifest["state_audit_details"] == audit["audit_details"]
    assert manifest["state_audit_details"]["path"] == "state_audit.parquet"
    assert manifest["state_audit_details"]["sha256"] == state_cache.sha256_file(result.audit_parquet_path)
    assert manifest["state_audit_details"]["rows"] == 2
    assert manifest["state_audit_details"]["columns"] == list(state_cache.AUDIT_DETAIL_COLUMNS)

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        state_cache.build_state_cache(
            data_dir,
            output_root,
            workers=1,
            config=config,
        )


def test_frozen_source_hash_detects_mutation(tmp_path: Path):
    source = _write_source(tmp_path, "000001.SZ", rows=121)
    expected = state_cache.sha256_file(source)
    changed = _raw_bars("000001.SZ", rows=122)
    changed.to_parquet(source, index=False)

    with pytest.raises(state_cache.StateCacheInputError, match="source mutated"):
        state_cache._assert_source_hash(source, expected, "unit test")


def test_files_at_or_below_warmup_are_fully_published_as_regime_zero(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_source(data_dir, "000001.SZ", rows=120)
    _write_source(data_dir, "000002.SZ", rows=126)

    result = state_cache.build_state_cache(
        data_dir,
        tmp_path / "output",
        workers=1,
        config=state_cache.StateCacheConfig(audit_symbols=1, audit_checkpoints=1),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    records = {item["symbol"]: item for item in manifest["sources"]}
    projection = pd.read_parquet(result.projection_path)

    assert {symbol: item["status"] for symbol, item in records.items()} == {
        "000001.SZ": "generated",
        "000002.SZ": "generated",
    }
    assert records["000001.SZ"]["state_rows"] == 120
    assert records["000002.SZ"]["state_rows"] == 126
    short_states = projection[projection["symbol"] == "000001.SZ"]
    assert len(short_states) == 120
    assert short_states["regime"].eq(0).all()
    assert len(projection) == 246


def test_duplicate_symbol_across_source_files_fails_closed(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = _raw_bars("000001.SZ")
    frame.to_parquet(data_dir / "first.parquet", index=False)
    frame.to_parquet(data_dir / "second.parquet", index=False)

    with pytest.raises(state_cache.StateCacheInputError, match="same symbol"):
        state_cache.build_state_cache(
            data_dir,
            tmp_path / "output",
            workers=1,
            config=state_cache.StateCacheConfig(audit_symbols=1, audit_checkpoints=1),
        )

    assert not list((tmp_path / "output").glob("CHAN_STATE_CACHE_*"))


def test_multiprocessing_build_produces_a_passing_two_symbol_audit(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_source(data_dir, "000001.SZ", rows=124)
    _write_source(data_dir, "000002.SZ", rows=124)

    result = state_cache.build_state_cache(
        data_dir,
        tmp_path / "output",
        workers=2,
        config=state_cache.StateCacheConfig(audit_symbols=2, audit_checkpoints=1, audit_seed=23),
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert result.audit_passed is True
    assert audit["prefix_full_audit"]["audited_symbols"] == 2
    assert audit["prefix_full_audit"]["comparison_count"] == 2
    assert audit["prefix_full_audit"]["mismatch_count"] == 0
