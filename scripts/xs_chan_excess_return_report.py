"""Build an excess-return-first report from the frozen Stage 2 portfolio paths.

The primary research comparator is F (the same factor path without the Chan
gate). FMA is a secondary comparator. Neither comparator is a market index, so
the resulting active returns must not be described as market alpha.

The module also supports market-benchmark comparisons (EW-All, CSI1000) when
the ``xs_chan_market_benchmark`` module has pre-computed benchmark returns.
These comparisons use a separate code path that does NOT require registered
HAC statistics, and are clearly labeled as "市场超额" in the output.

This module is deliberately separate from ``xs_chan_exploration_stage2.py``:
the Stage 2 publication identity binds that source file, and a derived report
must not mutate the frozen study or its confirmation governance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xs_chan_exploration_stage2 as stage2

SOURCE_PATH = Path(__file__).resolve()
REPO_ROOT = SOURCE_PATH.parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scripts/_output/xs_chan_excess_return_report"
REPORT_SCHEMA = "xs_chan_excess_return_report_v1"
REGISTERED_METRIC = "paired_net_return_40bps"
WEEKS_PER_YEAR = 52.0
TARGET_SLOTS = 50.0
DERIVED_OUTPUT_FILES = {"excess_curve.csv", "excess_metrics.csv", "report.md"}

MARKET_PAIR_SPECS: dict[str, dict[str, str]] = {
    "FC_minus_EW": {
        "strategy": "FC",
        "benchmark": "ew_all_return",
        "name": "FC − EW-All · 缠论策略相对全A等权",
        "comparison_role": "MARKET_BENCHMARK",
    },
    "F_minus_EW": {
        "strategy": "F",
        "benchmark": "ew_all_return",
        "name": "F − EW-All · 纯因子相对全A等权",
        "comparison_role": "FACTOR_MARKET_ALPHA",
    },
    "FC_minus_CSI1000": {
        "strategy": "FC",
        "benchmark": "000852_return",
        "name": "FC − CSI1000 · 缠论策略相对中证1000",
        "comparison_role": "INDEX_REFERENCE",
    },
    "F_minus_CSI1000": {
        "strategy": "F",
        "benchmark": "000852_return",
        "name": "F − CSI1000 · 纯因子相对中证1000",
        "comparison_role": "INDEX_REFERENCE",
    },
}

PAIR_SPECS: dict[str, dict[str, str]] = {
    "FC_minus_F": {
        "strategy": "FC",
        "benchmark": "F",
        "name": "FC − F · 缠论门控相对纯因子",
        "comparison_role": "PRIMARY_RESEARCH_COMPARATOR",
    },
    "FC_minus_FMA": {
        "strategy": "FC",
        "benchmark": "FMA",
        "name": "FC − FMA · 缠论门控相对 SMA20 门",
        "comparison_role": "SECONDARY_RESEARCH_COMPARATOR",
    },
}

SEGMENTS = {
    "FULL": (pd.Timestamp("2022-01-14"), pd.Timestamp("2026-06-05")),
    "DEVELOPMENT": (pd.Timestamp("2022-01-14"), pd.Timestamp("2023-12-29")),
    "HISTORICAL_VALIDATION": (pd.Timestamp("2024-01-05"), pd.Timestamp("2026-06-05")),
}

COST_COLUMNS = (
    "net_return_0bps",
    "net_return_20bps",
    "net_return_40bps",
    "net_return_60bps",
)

REQUIRED_WEEKLY_COLUMNS = {
    "decision_dt",
    "arm",
    "slots",
    "buy_count",
    "sell_count",
    "gross_return",
    "net_return",
    *COST_COLUMNS,
}


class ExcessReportError(RuntimeError):
    """Raised when a source or derived-report invariant is violated."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _pp(value: float, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f} pp"


def _num(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _bps(value: float, digits: int = 2) -> str:
    return f"{float(value) * 10_000:.{digits}f} bps"


def validate_weekly_paths(weekly: pd.DataFrame, *, expected_weeks: int | None = 224) -> pd.DataFrame:
    """Validate and normalize the registered fixed-slot weekly path table."""

    missing = REQUIRED_WEEKLY_COLUMNS - set(weekly.columns)
    if missing:
        raise ExcessReportError(f"path_weekly missing columns: {sorted(missing)}")
    normalized = weekly.copy()
    normalized["decision_dt"] = pd.to_datetime(normalized["decision_dt"])
    normalized = normalized.sort_values(["decision_dt", "arm"]).reset_index(drop=True)
    if set(normalized["arm"]) != {"F", "FC", "FMA"}:
        raise ExcessReportError(f"unexpected arms: {sorted(set(normalized['arm']))}")
    counts = normalized.groupby("arm").size()
    if counts.nunique() != 1:
        raise ExcessReportError(f"path arms are not date-balanced: {counts.to_dict()}")
    if expected_weeks is not None and not counts.eq(expected_weeks).all():
        raise ExcessReportError(f"expected {expected_weeks} weeks per arm, got {counts.to_dict()}")
    if normalized.duplicated(["decision_dt", "arm"]).any():
        raise ExcessReportError("path_weekly contains duplicate arm/date rows")
    if not np.allclose(
        normalized["net_return"],
        normalized["net_return_40bps"],
        rtol=0,
        atol=1e-15,
    ):
        raise ExcessReportError("net_return is not the registered 40bps path")
    return normalized


def _paired_path(weekly: pd.DataFrame, pair: str) -> pd.DataFrame:
    spec = PAIR_SPECS[pair]
    strategy = weekly[weekly["arm"].eq(spec["strategy"])].sort_values("decision_dt")
    benchmark = weekly[weekly["arm"].eq(spec["benchmark"])].sort_values("decision_dt")
    paired = strategy.merge(
        benchmark,
        on="decision_dt",
        how="inner",
        validate="one_to_one",
        suffixes=("_strategy", "_benchmark"),
    )
    if len(paired) != len(strategy) or len(paired) != len(benchmark):
        raise ExcessReportError(f"{pair} does not have identical strategy and comparator dates")
    return paired


def _registered_row(
    registered_summary: pd.DataFrame,
    *,
    pair: str,
    segment: str,
) -> pd.Series:
    rows = registered_summary[
        registered_summary["metric"].eq(REGISTERED_METRIC)
        & registered_summary["path"].eq(pair)
        & registered_summary["segment"].eq(segment)
    ]
    if len(rows) != 1:
        raise ExcessReportError(f"expected one registered paired row for {pair}/{segment}, got {len(rows)}")
    return rows.iloc[0]


def _build_segment_curve(paired: pd.DataFrame, pair: str, segment: str) -> pd.DataFrame:
    lo, hi = SEGMENTS[segment]
    scoped = paired[paired["decision_dt"].between(lo, hi)].copy()
    if scoped.empty:
        raise ExcessReportError(f"{pair}/{segment} has no paired observations")

    spec = PAIR_SPECS[pair]
    curve = pd.DataFrame(
        {
            "decision_dt": scoped["decision_dt"],
            "segment": segment,
            "pair": pair,
            "pair_name": spec["name"],
            "strategy": spec["strategy"],
            "benchmark": spec["benchmark"],
            "strategy_net_return": scoped["net_return_strategy"],
            "benchmark_net_return": scoped["net_return_benchmark"],
            "strategy_gross_return": scoped["gross_return_strategy"],
            "benchmark_gross_return": scoped["gross_return_benchmark"],
            "strategy_exposure": scoped["slots_strategy"] / TARGET_SLOTS,
            "benchmark_exposure": scoped["slots_benchmark"] / TARGET_SLOTS,
            "strategy_turnover": (scoped["buy_count_strategy"] + scoped["sell_count_strategy"]) / (2.0 * TARGET_SLOTS),
            "benchmark_turnover": (scoped["buy_count_benchmark"] + scoped["sell_count_benchmark"])
            / (2.0 * TARGET_SLOTS),
        }
    )
    curve["active_return"] = curve["strategy_net_return"] - curve["benchmark_net_return"]
    curve["period_relative_return"] = (1.0 + curve["strategy_net_return"]) / (1.0 + curve["benchmark_net_return"]) - 1.0
    curve["gross_active_return"] = curve["strategy_gross_return"] - curve["benchmark_gross_return"]
    curve["strategy_cost_drag"] = curve["strategy_gross_return"] - curve["strategy_net_return"]
    curve["benchmark_cost_drag"] = curve["benchmark_gross_return"] - curve["benchmark_net_return"]
    curve["differential_cost_advantage"] = curve["benchmark_cost_drag"] - curve["strategy_cost_drag"]
    curve["exposure_gap"] = curve["strategy_exposure"] - curve["benchmark_exposure"]
    curve["exposure_effect"] = curve["exposure_gap"] * curve["benchmark_gross_return"]
    curve["selection_gate_residual"] = curve["gross_active_return"] - curve["exposure_effect"]
    curve["strategy_nav"] = (1.0 + curve["strategy_net_return"]).cumprod()
    curve["benchmark_nav"] = (1.0 + curve["benchmark_net_return"]).cumprod()
    curve["relative_nav"] = curve["strategy_nav"] / curve["benchmark_nav"]
    relative_with_base = np.r_[1.0, curve["relative_nav"].to_numpy(dtype=float)]
    curve["relative_drawdown"] = (
        curve["relative_nav"].to_numpy(dtype=float) / np.maximum.accumulate(relative_with_base)[1:] - 1.0
    )
    curve["cumulative_return_gap"] = curve["strategy_nav"] - curve["benchmark_nav"]

    for column in COST_COLUMNS:
        suffix = column.removeprefix("net_return_")
        strategy_return = scoped[f"{column}_strategy"].reset_index(drop=True)
        benchmark_return = scoped[f"{column}_benchmark"].reset_index(drop=True)
        curve[f"strategy_return_{suffix}"] = strategy_return.to_numpy()
        curve[f"benchmark_return_{suffix}"] = benchmark_return.to_numpy()
        curve[f"relative_nav_{suffix}"] = (
            (1.0 + strategy_return).cumprod() / (1.0 + benchmark_return).cumprod()
        ).to_numpy()
    return curve.reset_index(drop=True)


def _segment_metrics(
    curve: pd.DataFrame,
    registered: pd.Series,
    *,
    comparison_role: str,
) -> dict[str, Any]:
    active = curve["active_return"].to_numpy(dtype=float)
    active_std = float(np.std(active, ddof=1))
    n = len(curve)
    registered_checks = {
        "n_weeks": n,
        "mean": float(np.mean(active)),
        "median": float(np.median(active)),
        "std": active_std,
        "win_rate": float(np.mean(active > 0)),
    }
    for field, actual in registered_checks.items():
        expected = float(registered[field])
        if field == "n_weeks":
            if int(expected) != int(actual):
                raise ExcessReportError(
                    f"registered {curve['pair'].iloc[0]}/{curve['segment'].iloc[0]} {field} mismatch"
                )
        elif not np.isclose(expected, actual, rtol=0, atol=1e-15):
            raise ExcessReportError(f"registered {curve['pair'].iloc[0]}/{curve['segment'].iloc[0]} {field} mismatch")

    relative_nav = curve["relative_nav"].to_numpy(dtype=float)
    return {
        "segment": curve["segment"].iloc[0],
        "pair": curve["pair"].iloc[0],
        "pair_name": curve["pair_name"].iloc[0],
        "comparison_role": comparison_role,
        "strategy": curve["strategy"].iloc[0],
        "benchmark": curve["benchmark"].iloc[0],
        "start": curve["decision_dt"].min(),
        "end": curve["decision_dt"].max(),
        "weeks": n,
        "strategy_cumulative_return": float(curve["strategy_nav"].iloc[-1] - 1.0),
        "benchmark_cumulative_return": float(curve["benchmark_nav"].iloc[-1] - 1.0),
        "cumulative_return_gap": float(curve["cumulative_return_gap"].iloc[-1]),
        "relative_wealth_return": float(relative_nav[-1] - 1.0),
        "annualized_relative_return": float(relative_nav[-1] ** (WEEKS_PER_YEAR / n) - 1.0),
        "active_weekly_mean": float(np.mean(active)),
        "annualized_active_mean": float(np.mean(active) * WEEKS_PER_YEAR),
        "tracking_error": float(active_std * math.sqrt(WEEKS_PER_YEAR)),
        "information_ratio": (
            float(np.mean(active) / active_std * math.sqrt(WEEKS_PER_YEAR)) if active_std > 0 else None
        ),
        "active_max_drawdown": float(curve["relative_drawdown"].min()),
        "active_hit_rate": float(np.mean(active > 0)),
        "hac_t": float(registered["hac_t"]),
        "hac_se": float(registered["hac_se"]),
        "hac_ci_low": float(registered["hac_ci_low"]),
        "hac_ci_high": float(registered["hac_ci_high"]),
        "bootstrap_ci_low": float(registered["bootstrap_ci_low"]),
        "bootstrap_ci_high": float(registered["bootstrap_ci_high"]),
        "strategy_mean_exposure": float(curve["strategy_exposure"].mean()),
        "benchmark_mean_exposure": float(curve["benchmark_exposure"].mean()),
        "mean_exposure_gap": float(curve["exposure_gap"].mean()),
        "strategy_mean_turnover": float(curve["strategy_turnover"].mean()),
        "benchmark_mean_turnover": float(curve["benchmark_turnover"].mean()),
        "gross_active_weekly_mean": float(curve["gross_active_return"].mean()),
        "exposure_effect_weekly_mean": float(curve["exposure_effect"].mean()),
        "selection_gate_residual_weekly_mean": float(curve["selection_gate_residual"].mean()),
        "differential_cost_advantage_weekly_mean": float(curve["differential_cost_advantage"].mean()),
    }


def build_excess_tables(
    weekly: pd.DataFrame,
    registered_summary: pd.DataFrame,
    *,
    expected_weeks: int | None = 224,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build segment-reset exact relative curves and registered paired metrics."""

    normalized = validate_weekly_paths(weekly, expected_weeks=expected_weeks)
    curve_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for pair, spec in PAIR_SPECS.items():
        paired = _paired_path(normalized, pair)
        for segment in SEGMENTS:
            curve = _build_segment_curve(paired, pair, segment)
            registered = _registered_row(registered_summary, pair=pair, segment=segment)
            curve_frames.append(curve)
            metric_rows.append(
                _segment_metrics(
                    curve,
                    registered,
                    comparison_role=spec["comparison_role"],
                )
            )
    curves = pd.concat(curve_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    return curves, metrics


def _metrics_table(metrics: pd.DataFrame, segment: str) -> str:
    rows = [
        "| 比较 | 累计收益差 | 相对财富收益 | 年化超额 | 周均主动 | 跟踪误差 | IR | 主动最大回撤 | 主动胜率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scoped = metrics[metrics["segment"].eq(segment)].set_index("pair")
    for pair in PAIR_SPECS:
        row = scoped.loc[pair]
        rows.append(
            "| "
            + " | ".join(
                [
                    PAIR_SPECS[pair]["name"],
                    _pp(row["cumulative_return_gap"]),
                    _pct(row["relative_wealth_return"]),
                    _pct(row["annualized_relative_return"]),
                    _pct(row["active_weekly_mean"], 3),
                    _pct(row["tracking_error"]),
                    _num(row["information_ratio"]),
                    _pct(row["active_max_drawdown"]),
                    _pct(row["active_hit_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def render_markdown(metrics: pd.DataFrame, curves: pd.DataFrame, *, identity: str) -> str:
    """Render the deterministic human report."""

    primary = metrics[metrics["segment"].eq("HISTORICAL_VALIDATION") & metrics["pair"].eq("FC_minus_F")].iloc[0]
    primary_curve = curves[curves["segment"].eq("HISTORICAL_VALIDATION") & curves["pair"].eq("FC_minus_F")].sort_values(
        "decision_dt"
    )
    cost_rows = []
    for label, column in (
        ("0 bps", "relative_nav_0bps"),
        ("20 bps", "relative_nav_20bps"),
        ("40 bps（注册主口径）", "relative_nav_40bps"),
        ("60 bps", "relative_nav_60bps"),
    ):
        cost_rows.append(f"| {label} | {_pct(float(primary_curve[column].iloc[-1] - 1.0))} |")

    return f"""# XS-Chan 超额收益优先回测报告

- Stage 2 内容身份：`{identity}`
- 研究性质：`RETROSPECTIVE_FALSIFICATION_ONLY`
- 实盘授权：`false`
- 主研究对照：F（纯因子路径）
- 辅助研究对照：FMA（SMA20 门）

## 口径

本报告把超额收益放在绝对收益之前。周主动收益定义为
`net_return_FC - net_return_comparator`；TE、IR 与 HAC 都使用这条配对周序列。
相对净值严格定义为 `NAV_FC / NAV_comparator`，不对周主动收益直接复利。

F 与 FMA 都是冻结研究对照，不是市场指数；因此这里的“超额”不是市场 alpha。
FC − F 衡量门控总机制，包含候选选择、欠配、现金暴露和差异成本。

## Historical validation（2024-01-05 至 2026-06-05）

{_metrics_table(metrics, "HISTORICAL_VALIDATION")}

主口径 FC − F 的周主动收益 HAC 95% CI 为
`[{_pct(primary["hac_ci_low"], 3)}, {_pct(primary["hac_ci_high"], 3)}]`，
HAC t 为 `{_num(primary["hac_t"])}`。点估计在经济上不利，但区间跨零，不能声称统计显著。

## 周均主动收益诊断

| 项目 | 验证段周均 |
| --- | ---: |
| 暴露/现金效应 | {_bps(primary["exposure_effect_weekly_mean"])} |
| 选择/门控残差 | {_bps(primary["selection_gate_residual_weekly_mean"])} |
| 毛主动收益 | {_bps(primary["gross_active_weekly_mean"])} |
| 差异成本优势 | {_bps(primary["differential_cost_advantage_weekly_mean"])} |
| 净主动收益 | {_bps(primary["active_weekly_mean"])} |

暴露效应按每周 `(FC 暴露 − F 暴露) × F 毛收益` 计算。选择/门控残差只是保持恒等式
成立的描述性诊断，不是因果 alpha；空槽收益按 0 处理。

## FC / F 相对成本敏感性

| 成本情景 | 验证段相对财富收益 |
| --- | ---: |
{chr(10).join(cost_rows)}

较高成本会让 FC 相对 F 看起来稍好，因为 F 换手更高；这不表示 FC 的绝对净值随成本
增加而改善。

## 全样本（224周）

{_metrics_table(metrics, "FULL")}

## 边界

- 历史验证段已被研究过程看过，不是独立 OOS。
- qfq 开盘只是路径代理，缺少完整集合竞价、停复牌、公司行为、退市终值和逐日账户会计。
- 当前资产没有沪深 300、中证 500/1000 等外部全收益指数，禁止写成“市场超额”。
- Stage 3 正式前瞻样本仍为 0/52；本报告不修改 Stage 2/3 身份、账本或授权状态。
"""


def _output_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(DERIVED_OUTPUT_FILES):
        path = output_dir / name
        row: dict[str, Any] = {
            "path": name,
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix == ".csv":
            row["rows"] = int(len(pd.read_csv(path)))
        rows.append(row)
    return rows


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _verify_output_files(path: Path, manifest: Mapping[str, Any]) -> int:
    output_rows = manifest.get("outputs")
    if not isinstance(output_rows, list):
        raise ExcessReportError("report manifest outputs must be a list")
    expected_names = {str(row.get("path")) for row in output_rows if isinstance(row, Mapping)}
    if expected_names != DERIVED_OUTPUT_FILES or len(output_rows) != len(DERIVED_OUTPUT_FILES):
        raise ExcessReportError(f"report manifest output set mismatch: {sorted(expected_names)}")
    actual_names = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
    expected_all = DERIVED_OUTPUT_FILES | {"manifest.json"}
    if actual_names != expected_all:
        raise ExcessReportError(
            f"report output set changed: missing={sorted(expected_all - actual_names)} "
            f"extra={sorted(actual_names - expected_all)}"
        )
    for row in output_rows:
        output = path / str(row["path"])
        if int(output.stat().st_size) != int(row["size"]):
            raise ExcessReportError(f"report output size mismatch: {output}")
        if sha256_file(output) != row["sha256"]:
            raise ExcessReportError(f"report output hash mismatch: {output}")
    return len(output_rows)


def generate_report(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Generate a verified derived report from the current Stage 2 publication."""

    spec = stage2.load_and_validate_spec()
    stage2_dir = stage2.output_path_for(spec)
    verification = stage2.verify_published_directory(stage2_dir, spec=spec)
    if verification["status"] != "VERIFIED_CURRENT":
        raise ExcessReportError(f"Stage 2 publication is not current: {verification}")

    weekly_path = stage2_dir / "path_weekly.parquet"
    summary_path = stage2_dir / "path_summary.parquet"
    weekly = pd.read_parquet(weekly_path)
    registered_summary = pd.read_parquet(summary_path)
    curves, metrics = build_excess_tables(weekly, registered_summary)

    identity = str(verification["study_identity"])
    output_dir = output_root / f"EXCESS_{identity}"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {item.name for item in output_dir.iterdir() if item.is_file()}
    allowed_names = DERIVED_OUTPUT_FILES | {"manifest.json"}
    if unexpected := existing_names - allowed_names:
        raise ExcessReportError(f"refusing to overwrite directory with unexpected files: {sorted(unexpected)}")
    _atomic_write_csv(output_dir / "excess_curve.csv", curves)
    _atomic_write_csv(output_dir / "excess_metrics.csv", metrics)
    _atomic_write_text(
        output_dir / "report.md",
        render_markdown(metrics, curves, identity=identity),
    )
    manifest = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "study_boundary": "RETROSPECTIVE_FALSIFICATION_ONLY",
        "live_trading_authorized": False,
        "evaluation_framework": "PAIRED_EXCESS_RETURN_FIRST",
        "primary_research_comparator": "F",
        "secondary_research_comparator": "FMA",
        "external_market_benchmark": None,
        "active_return_formula": "net_return_FC - net_return_comparator",
        "relative_nav_formula": "NAV_FC / NAV_comparator",
        "report_source_sha256": sha256_file(SOURCE_PATH),
        "stage2_verification": verification,
        "inputs": {
            "path_weekly": {"sha256": sha256_file(weekly_path), "rows": int(len(weekly))},
            "path_summary": {
                "sha256": sha256_file(summary_path),
                "rows": int(len(registered_summary)),
            },
        },
        "outputs": _output_inventory(output_dir),
    }
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return output_dir


def verify_report(path: Path) -> dict[str, Any]:
    """Verify a generated report manifest and its source/output hashes."""

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ExcessReportError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REPORT_SCHEMA:
        raise ExcessReportError(f"unexpected report schema: {manifest.get('schema')}")
    if manifest.get("live_trading_authorized") is not False:
        raise ExcessReportError("derived report must not authorize live trading")
    if manifest.get("report_source_sha256") != sha256_file(SOURCE_PATH):
        raise ExcessReportError("report generator source changed")
    files_verified = _verify_output_files(path, manifest)

    spec = stage2.load_and_validate_spec()
    stage2_dir = stage2.output_path_for(spec)
    verification = stage2.verify_published_directory(stage2_dir, spec=spec)
    if verification["study_identity"] != manifest["stage2_verification"]["study_identity"]:
        raise ExcessReportError("report Stage 2 identity is not current")
    for name, source in (
        ("path_weekly", stage2_dir / "path_weekly.parquet"),
        ("path_summary", stage2_dir / "path_summary.parquet"),
    ):
        if sha256_file(source) != manifest["inputs"][name]["sha256"]:
            raise ExcessReportError(f"report input changed: {name}")
    return {
        "status": "VERIFIED_EXCESS_REPORT",
        "path": str(path),
        "stage2_identity": verification["study_identity"],
        "files_verified": files_verified + 3,
        "live_trading_authorized": False,
    }


def _market_segment_curve(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    decision_dates: pd.DatetimeIndex,
    pair: str,
    segment: str,
) -> pd.DataFrame:
    """Build a segment curve for market benchmark comparison (no registered stats required)."""

    lo, hi = SEGMENTS[segment]
    mask = (decision_dates >= lo) & (decision_dates <= hi)
    if not mask.any():
        return pd.DataFrame()

    s_ret = strategy_returns[mask].to_numpy(dtype=float)
    b_ret = benchmark_returns[mask].to_numpy(dtype=float)
    dates = decision_dates[mask]
    spec = MARKET_PAIR_SPECS[pair]

    active = s_ret - b_ret
    s_nav = np.cumprod(1.0 + s_ret)
    b_nav = np.cumprod(1.0 + b_ret)
    relative_nav = s_nav / b_nav
    relative_with_base = np.r_[1.0, relative_nav]
    relative_dd = relative_nav / np.maximum.accumulate(relative_with_base)[1:] - 1.0

    return pd.DataFrame(
        {
            "decision_dt": dates,
            "segment": segment,
            "pair": pair,
            "pair_name": spec["name"],
            "strategy_net_return": s_ret,
            "benchmark_net_return": b_ret,
            "active_return": active,
            "strategy_nav": s_nav,
            "benchmark_nav": b_nav,
            "relative_nav": relative_nav,
            "relative_drawdown": relative_dd,
            "cumulative_return_gap": s_nav - b_nav,
        }
    )


def _market_segment_metrics(curve: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics for a market benchmark segment (no HAC registration)."""

    if curve.empty:
        return {}

    active = curve["active_return"].to_numpy(dtype=float)
    active_std = float(np.std(active, ddof=1)) if len(active) > 1 else 0.0
    n = len(curve)
    relative_nav = curve["relative_nav"].to_numpy(dtype=float)

    return {
        "segment": curve["segment"].iloc[0],
        "pair": curve["pair"].iloc[0],
        "pair_name": curve["pair_name"].iloc[0],
        "comparison_role": MARKET_PAIR_SPECS[curve["pair"].iloc[0]]["comparison_role"],
        "start": curve["decision_dt"].min(),
        "end": curve["decision_dt"].max(),
        "weeks": n,
        "strategy_cumulative_return": float(curve["strategy_nav"].iloc[-1] - 1.0),
        "benchmark_cumulative_return": float(curve["benchmark_nav"].iloc[-1] - 1.0),
        "cumulative_return_gap": float(curve["cumulative_return_gap"].iloc[-1]),
        "relative_wealth_return": float(relative_nav[-1] - 1.0),
        "annualized_relative_return": float(relative_nav[-1] ** (WEEKS_PER_YEAR / n) - 1.0),
        "active_weekly_mean": float(np.mean(active)),
        "annualized_active_mean": float(np.mean(active) * WEEKS_PER_YEAR),
        "tracking_error": float(active_std * math.sqrt(WEEKS_PER_YEAR)) if active_std > 0 else 0.0,
        "information_ratio": (
            float(np.mean(active) / active_std * math.sqrt(WEEKS_PER_YEAR)) if active_std > 0 else None
        ),
        "active_max_drawdown": float(curve["relative_drawdown"].min()),
        "active_hit_rate": float(np.mean(active > 0)),
    }


def build_market_excess_tables(
    weekly: pd.DataFrame,
    benchmark_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build market-benchmark excess tables from path_weekly and market benchmark data.

    Parameters
    ----------
    weekly : path_weekly from Stage 2 (arms F, FC, FMA with net_return_40bps)
    benchmark_weekly : from xs_chan_market_benchmark (decision_dt, ew_all_return, 000852_return, ...)
    """

    weekly = weekly.copy()
    weekly["decision_dt"] = pd.to_datetime(weekly["decision_dt"])
    benchmark_weekly = benchmark_weekly.copy()
    benchmark_weekly["decision_dt"] = pd.to_datetime(benchmark_weekly["decision_dt"])

    curve_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for pair, spec in MARKET_PAIR_SPECS.items():
        arm_name = spec["strategy"]
        bench_col = spec["benchmark"]

        if bench_col not in benchmark_weekly.columns:
            continue

        arm_data = weekly[weekly["arm"].eq(arm_name)].sort_values("decision_dt").reset_index(drop=True)
        merged = arm_data.merge(benchmark_weekly[["decision_dt", bench_col]], on="decision_dt", how="inner")

        if merged.empty:
            continue

        strategy_returns = merged["net_return_40bps"].astype(float)
        bench_returns = merged[bench_col].astype(float)
        decision_dates = pd.DatetimeIndex(merged["decision_dt"])

        valid = np.isfinite(strategy_returns) & np.isfinite(bench_returns)
        strategy_returns = strategy_returns[valid].reset_index(drop=True)
        bench_returns = bench_returns[valid].reset_index(drop=True)
        decision_dates = decision_dates[valid]

        for segment in SEGMENTS:
            curve = _market_segment_curve(strategy_returns, bench_returns, decision_dates, pair, segment)
            if curve.empty:
                continue
            metrics = _market_segment_metrics(curve)
            if metrics:
                curve_frames.append(curve)
                metric_rows.append(metrics)

    if not curve_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(metric_rows)


def _market_metrics_table(metrics: pd.DataFrame, segment: str) -> str:
    """Render a markdown table for market benchmark metrics."""

    rows = [
        "| 比较 | 策略累计 | 基准累计 | 累计收益差 | 相对财富收益 | 年化超额 | IR | 主动最大回撤 | 主动胜率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scoped = metrics[metrics["segment"].eq(segment)]
    for _, row in scoped.iterrows():
        ir_str = _num(row["information_ratio"]) if row["information_ratio"] is not None else "N/A"
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["pair_name"]),
                    _pct(row["strategy_cumulative_return"]),
                    _pct(row["benchmark_cumulative_return"]),
                    _pp(row["cumulative_return_gap"]),
                    _pct(row["relative_wealth_return"]),
                    _pct(row["annualized_relative_return"]),
                    ir_str,
                    _pct(row["active_max_drawdown"]),
                    _pct(row["active_hit_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def render_market_benchmark_section(market_metrics: pd.DataFrame) -> str:
    """Render the market benchmark section for the report."""

    if market_metrics.empty:
        return "\n## 市场超额收益\n\n市场基准数据不可用。请先运行：\n```\nuv run --no-sync python scripts/xs_chan_market_benchmark.py generate\n```\n"

    lines = [
        "",
        "## 市场超额收益",
        "",
        "以下使用外部市场基准衡量策略的绝对 alpha。「全A等权」是最公平的 null hypothesis：",
        "如果等权随机持有全部可交易A股，收益如何。中证1000 是小盘市值加权参照。",
        "",
        "**注意**：「F − EW-All」衡量的是纯因子选股本身的市场超额；「FC − F」（上文）衡量",
        "缠论门控在因子基础上的增量。三层归因：",
        "",
        "```",
        "策略总收益 = 市场 beta + 因子 alpha + 门控增量",
        "市场 beta  ≈ EW-All 收益",
        "因子 alpha ≈ F − EW-All",
        "门控增量   ≈ FC − F",
        "```",
        "",
        "### Historical validation（2024-01-05 至 2026-06-05）",
        "",
    ]

    if "HISTORICAL_VALIDATION" in market_metrics["segment"].values:
        lines.append(_market_metrics_table(market_metrics, "HISTORICAL_VALIDATION"))
    else:
        lines.append("（验证段数据不可用）")

    lines.extend(["", "### 全样本", ""])
    if "FULL" in market_metrics["segment"].values:
        lines.append(_market_metrics_table(market_metrics, "FULL"))
    else:
        lines.append("（全样本数据不可用）")

    lines.extend(
        [
            "",
            "### 口径说明",
            "",
            "- 全A等权(EW-All)：每周所有可交易A股（开盘价>1元、有成交）的等权 5-session open-to-open 收益均值",
            "- 中证1000(CSI1000)：官方指数同窗口 close-to-close 收益",
            "- 策略使用 net_return_40bps（含买 15bps + 卖 25bps 成本），基准为零成本",
            "- 基准不扣除分红再投资，策略侧也无分红调整，二者口径一致",
            "",
        ]
    )
    return "\n".join(lines)


def generate_market_report(
    weekly: pd.DataFrame,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate market benchmark excess report as a standalone addon.

    Returns a dict with status and file paths. Designed to be called after
    the main generate_report() or independently.
    """

    try:
        import xs_chan_market_benchmark as mkt
    except ImportError:
        import importlib.util

        spec_obj = importlib.util.spec_from_file_location(
            "xs_chan_market_benchmark",
            Path(__file__).resolve().parent / "xs_chan_market_benchmark.py",
        )
        if spec_obj is None or spec_obj.loader is None:
            return {"status": "MARKET_BENCHMARK_MODULE_NOT_FOUND"}
        mkt = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mkt)

    try:
        benchmark_weekly = mkt.load_benchmark_weekly()
    except FileNotFoundError:
        return {"status": "MARKET_BENCHMARK_NOT_GENERATED"}

    market_curves, market_metrics = build_market_excess_tables(weekly, benchmark_weekly)

    if market_metrics.empty:
        return {"status": "NO_OVERLAPPING_DATES"}

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / "market_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_curves.to_csv(output_dir / "market_excess_curve.csv", index=False)
    market_metrics.to_csv(output_dir / "market_excess_metrics.csv", index=False)

    report_text = render_market_benchmark_section(market_metrics)
    (output_dir / "market_benchmark_report.md").write_text(report_text, encoding="utf-8")

    return {
        "status": "GENERATED",
        "output_dir": str(output_dir),
        "weeks_matched": int(market_metrics["weeks"].max()) if not market_metrics.empty else 0,
        "pairs": list(market_metrics["pair"].unique()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate the current excess-return report")
    generate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    generate.add_argument("--with-market", action="store_true", help="Also generate market benchmark section")
    verify = subparsers.add_parser("verify", help="Verify a generated excess-return report")
    verify.add_argument("--path", type=Path)
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    market = subparsers.add_parser("market", help="Generate market-benchmark-only report")
    market.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        path = generate_report(output_root=args.output_root)
        result = verify_report(path)
        if getattr(args, "with_market", False):
            spec = stage2.load_and_validate_spec()
            stage2_dir = stage2.output_path_for(spec)
            weekly = pd.read_parquet(stage2_dir / "path_weekly.parquet")
            market_result = generate_market_report(weekly, output_dir=path / "market_benchmark")
            result["market_benchmark"] = market_result
    elif args.command == "market":
        spec = stage2.load_and_validate_spec()
        stage2_dir = stage2.output_path_for(spec)
        weekly = pd.read_parquet(stage2_dir / "path_weekly.parquet")
        result = generate_market_report(weekly, output_dir=args.output_root / "market_benchmark")
    else:
        if args.path is None:
            spec = stage2.load_and_validate_spec()
            args.path = args.output_root / f"EXCESS_{stage2.study_identity(spec)}"
        result = verify_report(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
