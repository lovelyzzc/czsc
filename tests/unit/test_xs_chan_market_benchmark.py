from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_excess_return_report as report  # noqa: E402
import xs_chan_market_benchmark as market  # noqa: E402


def _synthetic_weekly_paths(n_weeks: int = 20) -> pd.DataFrame:
    """Generate synthetic path_weekly data spanning FULL/DEV/VAL segments."""

    np.random.seed(42)
    start = pd.Timestamp("2022-01-14")
    dates = pd.date_range(start, periods=n_weeks, freq="W-FRI")

    rows = []
    for arm in ("F", "FC", "FMA"):
        for dt in dates:
            gross = np.random.normal(0.005, 0.03)
            cost = 0.002
            rows.append(
                {
                    "decision_dt": dt,
                    "arm": arm,
                    "slots": 50,
                    "buy_count": 10,
                    "sell_count": 8,
                    "gross_return": gross,
                    "net_return": gross - cost,
                    "net_return_0bps": gross,
                    "net_return_20bps": gross - 0.001,
                    "net_return_40bps": gross - cost,
                    "net_return_60bps": gross - 0.003,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_benchmark(weekly: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic market benchmark aligned to weekly decision dates."""

    np.random.seed(123)
    dates = sorted(weekly["decision_dt"].unique())
    tradable_count = np.random.randint(3000, 5000, len(dates))
    missing_exit_count = np.random.randint(0, 10, len(dates))
    observable_exit_count = tradable_count - missing_exit_count
    return pd.DataFrame(
        {
            "decision_dt": pd.to_datetime(dates),
            "ew_all_return": np.random.normal(0.003, 0.025, len(dates)),
            "ew_all_median_return": np.random.normal(0.002, 0.02, len(dates)),
            "ew_all_std": np.abs(np.random.normal(0.04, 0.01, len(dates))),
            "tradable_count": tradable_count,
            "observable_exit_count": observable_exit_count,
            "observable_exit_rate": observable_exit_count / tradable_count,
            "000852_return": np.random.normal(0.004, 0.03, len(dates)),
            "000905_return": np.random.normal(0.003, 0.028, len(dates)),
        }
    )


class TestBuildMarketExcessTables:
    def test_produces_curves_and_metrics_for_all_available_pairs(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        curves, metrics = report.build_market_excess_tables(weekly, benchmark)

        assert not curves.empty
        assert not metrics.empty
        assert set(metrics["pair"].unique()) == set(report.MARKET_PAIR_SPECS.keys())

    def test_handles_missing_benchmark_column_gracefully(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        benchmark = benchmark.drop(columns=["000852_return"])

        curves, metrics = report.build_market_excess_tables(weekly, benchmark)

        present_pairs = set(metrics["pair"].unique())
        assert "FC_minus_CSI1000" not in present_pairs
        assert "F_minus_CSI1000" not in present_pairs
        assert "FC_minus_EW" in present_pairs
        assert "F_minus_EW" in present_pairs

    def test_rejects_incomplete_date_coverage(self):
        weekly = _synthetic_weekly_paths(n_weeks=5)
        benchmark = pd.DataFrame(
            {
                "decision_dt": pd.date_range("2030-01-01", periods=5, freq="W-FRI"),
                "ew_all_return": [0.01] * 5,
            }
        )
        with pytest.raises(report.ExcessReportError, match="coverage is incomplete"):
            report.build_market_excess_tables(weekly, benchmark)

    def test_relative_nav_is_exact_ratio_not_compounded_active(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        curves, metrics = report.build_market_excess_tables(weekly, benchmark)

        for pair in report.MARKET_PAIR_SPECS:
            pair_curves = curves[(curves["pair"] == pair) & (curves["segment"] == "FULL")]
            if pair_curves.empty:
                continue
            s_nav = pair_curves["strategy_nav"].to_numpy()
            b_nav = pair_curves["benchmark_nav"].to_numpy()
            expected_relative = s_nav / b_nav
            assert pair_curves["relative_nav"].to_numpy() == pytest.approx(expected_relative, rel=1e-12)

    def test_active_return_is_arithmetic_difference(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        curves, _ = report.build_market_excess_tables(weekly, benchmark)

        for pair in report.MARKET_PAIR_SPECS:
            pair_curves = curves[(curves["pair"] == pair) & (curves["segment"] == "FULL")]
            if pair_curves.empty:
                continue
            expected_active = pair_curves["strategy_net_return"] - pair_curves["benchmark_net_return"]
            assert pair_curves["active_return"].to_numpy() == pytest.approx(expected_active.to_numpy(), rel=1e-12)

    def test_metrics_annualized_return_uses_geometric(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        for _, row in metrics.iterrows():
            if row["weeks"] < 2:
                continue
            expected_ann = (1.0 + row["relative_wealth_return"]) ** (report.WEEKS_PER_YEAR / row["weeks"]) - 1.0
            assert row["annualized_relative_return"] == pytest.approx(expected_ann, rel=1e-10)

    def test_f_minus_ew_plus_fc_minus_f_decomposes_fc_minus_ew(self):
        """Three-layer attribution: FC-EW = (FC-F) + (F-EW)."""
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        for segment in ("FULL",):
            seg_metrics = metrics[metrics["segment"] == segment].set_index("pair")
            if "FC_minus_EW" not in seg_metrics.index or "F_minus_EW" not in seg_metrics.index:
                continue

            fc_ew_mean = seg_metrics.loc["FC_minus_EW", "active_weekly_mean"]
            f_ew_mean = seg_metrics.loc["F_minus_EW", "active_weekly_mean"]

            fc_arm = weekly[weekly["arm"] == "FC"].sort_values("decision_dt")["net_return_40bps"]
            f_arm = weekly[weekly["arm"] == "F"].sort_values("decision_dt")["net_return_40bps"]
            fc_minus_f_mean = float((fc_arm.values - f_arm.values).mean())

            assert fc_ew_mean == pytest.approx(f_ew_mean + fc_minus_f_mean, abs=1e-10)

    def test_ew_exit_coverage_is_carried_into_metrics(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly)
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        ew = metrics[(metrics["segment"] == "FULL") & (metrics["pair"] == "FC_minus_EW")].iloc[0]
        assert ew["benchmark_observable_exit_rate_mean"] == pytest.approx(benchmark["observable_exit_rate"].mean())
        assert ew["benchmark_observable_exit_rate_min"] == pytest.approx(benchmark["observable_exit_rate"].min())
        assert int(ew["benchmark_missing_exit_observations"]) == int(
            (benchmark["tradable_count"] - benchmark["observable_exit_count"]).sum()
        )


class TestRenderMarketBenchmarkSection:
    def test_render_empty_produces_fallback_message(self):
        text = report.render_market_benchmark_section(pd.DataFrame())
        assert "市场基准数据不可用" in text

    def test_render_with_data_contains_key_sections(self):
        weekly = _synthetic_weekly_paths(n_weeks=200)
        benchmark = _synthetic_benchmark(weekly)
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        text = report.render_market_benchmark_section(metrics)
        assert "相对市场基准收益" in text
        assert "不能称为 alpha" in text
        assert "因果归因" in text
        assert "全A等权" in text
        assert "中证1000" in text
        assert "口径说明" in text
        assert "退出开盘覆盖" in text

    def test_render_includes_both_segments(self):
        weekly = _synthetic_weekly_paths(n_weeks=200)
        benchmark = _synthetic_benchmark(weekly)
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        text = report.render_market_benchmark_section(metrics)
        assert "Historical validation" in text
        assert "全样本" in text

    def test_render_explicitly_discloses_unavailable_csi1000(self):
        weekly = _synthetic_weekly_paths(n_weeks=20)
        benchmark = _synthetic_benchmark(weekly).drop(columns=["000852_return", "000905_return"])
        _, metrics = report.build_market_excess_tables(weekly, benchmark)

        text = report.render_market_benchmark_section(metrics)
        assert "CSI1000 本轮不可用" in text
        assert "未计算、未展示" in text


class TestMarketPairSpecsComplete:
    def test_all_market_pairs_include_required_fields(self):
        for pair, spec in report.MARKET_PAIR_SPECS.items():
            assert "strategy" in spec, f"{pair} missing strategy"
            assert "benchmark" in spec, f"{pair} missing benchmark"
            assert "name" in spec, f"{pair} missing name"
            assert "comparison_role" in spec, f"{pair} missing comparison_role"
            assert spec["strategy"] in ("F", "FC", "FMA")

    def test_f_minus_ew_pair_exists_for_factor_alpha_decomposition(self):
        assert "F_minus_EW" in report.MARKET_PAIR_SPECS
        assert report.MARKET_PAIR_SPECS["F_minus_EW"]["comparison_role"] == "FACTOR_MARKET_RELATIVE"


def test_index_benchmark_uses_exact_strategy_open_window():
    calendar = pd.bdate_range("2024-01-01", periods=10)
    decision_dates = pd.DatetimeIndex([calendar[1]])
    index_daily = pd.DataFrame(
        {
            "trade_date": calendar.strftime("%Y%m%d"),
            "open": np.arange(10, 20, dtype=float),
            "close": np.arange(100, 110, dtype=float),
        }
    )

    result = market.compute_index_weekly_returns(
        index_daily,
        calendar,
        decision_dates,
        "000852.SH",
    )

    expected = index_daily.loc[7, "open"] / index_daily.loc[2, "open"] - 1.0
    assert result["000852_return"].iloc[0] == pytest.approx(expected)


def test_benchmark_verifier_rejects_output_and_raw_input_tampering(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "benchmark"
    raw_dir.mkdir()
    output_dir.mkdir()
    raw_path = raw_dir / "000001.SZ.parquet"
    pd.DataFrame({"trade_date": ["20240101"], "open": [10.0]}).to_parquet(raw_path, index=False)
    monkeypatch.setattr(market, "DATA_DIR", raw_dir)

    weekly_path = output_dir / "market_benchmark_weekly.parquet"
    weekly = pd.DataFrame(
        {
            "decision_dt": pd.to_datetime(["2024-01-05"]),
            "ew_all_return": [0.01],
            "tradable_count": [1000],
            "observable_exit_count": [999],
            "observable_exit_rate": [0.999],
        }
    )
    weekly.to_parquet(weekly_path, index=False)
    manifest = {
        "schema": market.MANIFEST_SCHEMA,
        "generator_sha256": market._sha256_file(market.SOURCE_PATH),
        "output_sha256": market._sha256_file(weekly_path),
        "raw_data_closure": market._physical_data_closure(),
        "index_inputs": [],
        "columns": list(weekly.columns),
        "output_weeks": len(weekly),
        "decision_start_date": "2024-01-05",
        "decision_end_date": "2024-01-05",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    market.verify_benchmark(output_dir)

    weekly.assign(ew_all_return=0.02).to_parquet(weekly_path, index=False)
    with pytest.raises(RuntimeError, match="parquet hash"):
        market.verify_benchmark(output_dir)

    weekly.to_parquet(weekly_path, index=False)
    pd.DataFrame({"trade_date": ["20240101"], "open": [11.0]}).to_parquet(raw_path, index=False)
    with pytest.raises(RuntimeError, match="input closure"):
        market.verify_benchmark(output_dir)
