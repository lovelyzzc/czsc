from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import xs_chan_excess_return_report as report  # noqa: E402


def _weekly_paths() -> pd.DataFrame:
    dates = pd.to_datetime(["2022-01-14", "2023-12-29", "2024-01-05", "2026-06-05"])
    returns = {
        "F": [0.10, 0.00, 0.10, -0.05],
        "FC": [0.05, 0.02, 0.00, -0.10],
        "FMA": [0.06, 0.01, 0.02, -0.08],
    }
    slots = {"F": 50, "FC": 25, "FMA": 35}
    rows = []
    for arm, arm_returns in returns.items():
        for index, (decision_dt, net_return) in enumerate(zip(dates, arm_returns, strict=True)):
            cost_drag = 0.002 + index * 0.0001
            gross_return = net_return + cost_drag
            rows.append(
                {
                    "decision_dt": decision_dt,
                    "arm": arm,
                    "slots": slots[arm],
                    "buy_count": 10 + index,
                    "sell_count": 8 + index,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "net_return_0bps": gross_return,
                    "net_return_20bps": net_return + cost_drag / 2,
                    "net_return_40bps": net_return,
                    "net_return_60bps": net_return - cost_drag / 2,
                }
            )
    return pd.DataFrame(rows)


def _registered_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, pair_spec in report.PAIR_SPECS.items():
        strategy = weekly[weekly["arm"].eq(pair_spec["strategy"])][["decision_dt", "net_return"]].rename(
            columns={"net_return": "strategy"}
        )
        benchmark = weekly[weekly["arm"].eq(pair_spec["benchmark"])][["decision_dt", "net_return"]].rename(
            columns={"net_return": "benchmark"}
        )
        paired = strategy.merge(benchmark, on="decision_dt", validate="one_to_one")
        paired["active"] = paired["strategy"] - paired["benchmark"]
        for segment, (lo, hi) in report.SEGMENTS.items():
            active = paired.loc[paired["decision_dt"].between(lo, hi), "active"].to_numpy()
            rows.append(
                {
                    "segment": segment,
                    "path": pair,
                    "metric": report.REGISTERED_METRIC,
                    "n_weeks": len(active),
                    "mean": float(np.mean(active)),
                    "median": float(np.median(active)),
                    "std": float(np.std(active, ddof=1)),
                    "win_rate": float(np.mean(active > 0)),
                    "hac_t": -0.5,
                    "hac_se": 0.01,
                    "hac_ci_low": -0.03,
                    "hac_ci_high": 0.01,
                    "bootstrap_ci_low": -0.04,
                    "bootstrap_ci_high": 0.02,
                }
            )
    return pd.DataFrame(rows)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = _weekly_paths()
    return report.build_excess_tables(
        weekly,
        _registered_summary(weekly),
        expected_weeks=None,
    )


def test_relative_nav_uses_exact_nav_ratio_not_compounded_active_difference():
    curves, metrics = _tables()
    curve = curves[curves["segment"].eq("HISTORICAL_VALIDATION") & curves["pair"].eq("FC_minus_F")].sort_values(
        "decision_dt"
    )
    strategy_nav = float(np.prod(1.0 + curve["strategy_net_return"]))
    benchmark_nav = float(np.prod(1.0 + curve["benchmark_net_return"]))
    expected_relative = strategy_nav / benchmark_nav
    wrong_active_compound = float(np.prod(1.0 + curve["active_return"]))

    assert curve["relative_nav"].iloc[-1] == pytest.approx(expected_relative)
    assert curve["relative_nav"].iloc[-1] != pytest.approx(wrong_active_compound)
    metric = metrics[metrics["segment"].eq("HISTORICAL_VALIDATION") & metrics["pair"].eq("FC_minus_F")].iloc[0]
    assert metric["relative_wealth_return"] == pytest.approx(expected_relative - 1.0)


def test_registered_paired_mean_mismatch_is_rejected():
    weekly = _weekly_paths()
    registered = _registered_summary(weekly)
    mask = registered["segment"].eq("FULL") & registered["path"].eq("FC_minus_F")
    registered.loc[mask, "mean"] += 0.001

    with pytest.raises(report.ExcessReportError, match="mean mismatch"):
        report.build_excess_tables(weekly, registered, expected_weeks=None)


def test_active_attribution_is_an_exact_diagnostic_identity():
    _, metrics = _tables()
    primary = metrics[metrics["segment"].eq("HISTORICAL_VALIDATION") & metrics["pair"].eq("FC_minus_F")].iloc[0]

    assert (primary["exposure_effect_weekly_mean"] + primary["selection_gate_residual_weekly_mean"]) == pytest.approx(
        primary["gross_active_weekly_mean"]
    )
    assert (primary["gross_active_weekly_mean"] + primary["differential_cost_advantage_weekly_mean"]) == pytest.approx(
        primary["active_weekly_mean"]
    )


def test_active_risk_metrics_and_cost_curves_are_segment_rebased():
    curves, metrics = _tables()
    curve = curves[curves["segment"].eq("HISTORICAL_VALIDATION") & curves["pair"].eq("FC_minus_F")].sort_values(
        "decision_dt"
    )
    metric = metrics[metrics["segment"].eq("HISTORICAL_VALIDATION") & metrics["pair"].eq("FC_minus_F")].iloc[0]
    active = curve["active_return"].to_numpy()
    active_std = float(np.std(active, ddof=1))
    relative = curve["relative_nav"].to_numpy()
    relative_with_base = np.r_[1.0, relative]
    expected_drawdown = relative / np.maximum.accumulate(relative_with_base)[1:] - 1.0

    assert metric["tracking_error"] == pytest.approx(active_std * np.sqrt(report.WEEKS_PER_YEAR))
    assert metric["information_ratio"] == pytest.approx(np.mean(active) / active_std * np.sqrt(report.WEEKS_PER_YEAR))
    assert metric["annualized_relative_return"] == pytest.approx(
        relative[-1] ** (report.WEEKS_PER_YEAR / len(curve)) - 1.0
    )
    assert metric["active_max_drawdown"] == pytest.approx(expected_drawdown.min())
    for suffix in ("0bps", "20bps", "40bps", "60bps"):
        expected = (1.0 + curve[f"strategy_return_{suffix}"]).cumprod() / (
            1.0 + curve[f"benchmark_return_{suffix}"]
        ).cumprod()
        assert curve[f"relative_nav_{suffix}"].to_numpy() == pytest.approx(expected.to_numpy())


def test_registered_net_return_alias_is_required():
    weekly = _weekly_paths()
    weekly.loc[0, "net_return_40bps"] += 0.001

    with pytest.raises(report.ExcessReportError, match="registered 40bps"):
        report.validate_weekly_paths(weekly, expected_weeks=None)


def test_markdown_prioritizes_excess_and_preserves_claim_boundary():
    curves, metrics = _tables()
    markdown = report.render_markdown(metrics, curves, identity="a" * 64)

    assert "把超额收益放在绝对收益之前" in markdown
    assert "NAV_FC / NAV_comparator" in markdown
    assert "不是市场指数" in markdown
    assert "不能声称统计显著" in markdown
    assert "实盘授权：`false`" in markdown
    assert "Stage 3 正式前瞻样本仍为 0/52" in markdown


def test_derived_output_inventory_fails_closed_on_tamper_and_extra_file(tmp_path: Path):
    contents = {
        "excess_curve.csv": "date,value\n2024-01-05,1.0\n",
        "excess_metrics.csv": "pair,value\nFC_minus_F,-0.1\n",
        "report.md": "# report\n",
    }
    for name, content in contents.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    manifest = {
        "outputs": [
            {
                "path": name,
                "size": (tmp_path / name).stat().st_size,
                "sha256": report.sha256_file(tmp_path / name),
            }
            for name in sorted(contents)
        ]
    }
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    assert report._verify_output_files(tmp_path, manifest) == 3

    (tmp_path / "report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(report.ExcessReportError, match="size mismatch|hash mismatch"):
        report._verify_output_files(tmp_path, manifest)

    (tmp_path / "report.md").write_text(contents["report.md"], encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("stale\n", encoding="utf-8")
    with pytest.raises(report.ExcessReportError, match="output set changed"):
        report._verify_output_files(tmp_path, manifest)
