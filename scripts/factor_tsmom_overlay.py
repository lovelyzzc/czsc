"""Test a 20-decision time-series-momentum exposure overlay on factor F.

The stock identities remain the frozen F path.  The only new decision is full
F exposure versus cash, using the causal close NAV of the F baseline.  The
protocol is frozen in ``factor_tsmom_overlay_protocol_2026-08-12.json``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "factor_tsmom_overlay_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "factor_tsmom_overlay"
REPORT_PATH = SCRIPT_DIR / "FACTOR_TSMOM_OVERLAY_2026-08-12.md"
PRIMARY_BPS = 40

ENGINE_SPEC = importlib.util.spec_from_file_location(
    "factor_exit_path_engine_for_tsmom", SCRIPT_DIR / "fsm_factor_exit_overlay.py"
)
assert ENGINE_SPEC and ENGINE_SPEC.loader
engine = importlib.util.module_from_spec(ENGINE_SPEC)
ENGINE_SPEC.loader.exec_module(engine)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(locator: str) -> Path:
    path = Path(locator).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_and_verify_protocol() -> dict[str, Any]:
    """Load the frozen protocol and bind all source files."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "factor_tsmom_overlay_protocol_v1":
        raise RuntimeError("unexpected factor TSMOM protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    for name in ("ranked_surface", "price_panel", "historical_stage2_f_path"):
        record = protocol["inputs"][name]
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    return protocol


def load_price_market(protocol: dict[str, Any], symbols: list[str]) -> Any:
    """Load dense prices for fixed F identities; state data is intentionally absent."""

    price_path = _resolve(protocol["inputs"]["price_panel"]["path"])
    calendar = pd.DatetimeIndex(
        pl.scan_parquet(price_path).select("dt").unique().sort("dt").collect().to_series().to_list()
    ).normalize()
    price = (
        pl.scan_parquet(price_path)
        .filter(pl.col("symbol").is_in(symbols))
        .select("symbol", "dt", "open", "close")
        .collect()
        .to_pandas()
    )
    price["dt"] = pd.to_datetime(price["dt"])
    open_frame = price.pivot(index="dt", columns="symbol", values="open").reindex(calendar)
    close_frame = price.pivot(index="dt", columns="symbol", values="close").reindex(calendar)
    return engine.MarketData(
        calendar=calendar,
        open=open_frame,
        close=close_frame,
        regime=pd.DataFrame(index=calendar, columns=open_frame.columns, dtype=float),
    )


def build_tsmom_targets(
    memberships: pd.DataFrame,
    baseline_path: pd.DataFrame,
    *,
    lookback: int = 20,
    end_date: pd.Timestamp,
    risk_on_positive: bool = True,
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    """Create causal full-F-or-cash targets from baseline close NAV."""

    all_targets = engine._targets_by_date(memberships)
    decisions = [date for date in sorted(all_targets) if date <= end_date]
    nav = baseline_path.set_index("dt")["nav"]
    targets: dict[pd.Timestamp, list[str]] = {}
    rows = []
    for index, decision_dt in enumerate(decisions):
        if decision_dt not in nav.index:
            raise RuntimeError(f"baseline NAV missing decision date {decision_dt.date()}")
        if index < lookback:
            trailing_return = np.nan
            risk_on = False
        else:
            anchor = decisions[index - lookback]
            trailing_return = float(nav.at[decision_dt] / nav.at[anchor] - 1)
            risk_on = trailing_return > 0 if risk_on_positive else trailing_return <= 0
        targets[decision_dt] = list(all_targets[decision_dt]) if risk_on else []
        rows.append(
            {
                "decision_dt": decision_dt,
                "lookback_anchor_dt": pd.NaT if index < lookback else decisions[index - lookback],
                "trailing_f_nav_return": trailing_return,
                "risk_on": risk_on,
                "target_slots": 50 if risk_on else 0,
            }
        )
    return targets, pd.DataFrame(rows)


def summarize_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    signals: pd.DataFrame,
    protocol: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Summarize the fixed risk-control comparison."""

    annualization = int(protocol["sample"]["annualization_sessions"])
    base = engine.segment_metrics(baseline, start, end, annualization)
    tsmom = engine.segment_metrics(candidate, start, end, annualization)
    hac = engine.active_hac(
        baseline,
        candidate,
        start,
        end,
        lag=int(protocol["sample"]["weekly_hac_lag"]),
        margin_weekly=float(protocol["statistics"]["weekly_noninferiority_margin"]),
    )
    scoped_signals = signals[signals["decision_dt"].between(start, end)]
    return {
        "baseline": base,
        "candidate": tsmom,
        "risk_on_decisions": int(scoped_signals["risk_on"].sum()),
        "decision_count": int(len(scoped_signals)),
        "risk_on_decision_rate": float(scoped_signals["risk_on"].mean()),
        "annualized_return_difference": tsmom["annualized_return"] - base["annualized_return"],
        "sharpe_improvement": tsmom["sharpe"] - base["sharpe"],
        "relative_drawdown_reduction": (abs(base["maximum_drawdown"]) - abs(tsmom["maximum_drawdown"]))
        / abs(base["maximum_drawdown"]),
        "weekly_active_hac": hac,
    }


def evaluate_gate(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the common development and validation primary gate."""

    stats = protocol["statistics"]
    checks = {
        "risk_on_rate_floor": summary["risk_on_decision_rate"] >= float(stats["minimum_risk_on_decision_rate"]),
        "risk_on_rate_cap": summary["risk_on_decision_rate"] <= float(stats["maximum_risk_on_decision_rate"]),
        "candidate_annual_return_positive": summary["candidate"]["annualized_return"] > 0,
        "drawdown_reduction": summary["relative_drawdown_reduction"]
        >= float(stats["minimum_relative_drawdown_reduction"]),
        "sharpe_improvement": summary["sharpe_improvement"] >= float(stats["minimum_sharpe_improvement"]),
        "annual_return_noninferiority": summary["annualized_return_difference"]
        >= -float(stats["maximum_annual_return_underperformance"]),
        "weekly_hac_noninferiority": summary["weekly_active_hac"]["noninferiority_one_sided_p"]
        < float(stats["one_sided_alpha"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def validation_subperiods(
    baseline: pd.DataFrame, candidate: pd.DataFrame, protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compute the two locked validation subperiod path returns."""

    rows = []
    for label, start, end in protocol["sample"]["validation_subperiods"]:
        base = engine.segment_metrics(baseline, pd.Timestamp(start), pd.Timestamp(end))
        gate = engine.segment_metrics(candidate, pd.Timestamp(start), pd.Timestamp(end))
        rows.append(
            {
                "period": label,
                "baseline_total_return": base["total_return"],
                "candidate_total_return": gate["total_return"],
                "paired_total_return": gate["total_return"] - base["total_return"],
            }
        )
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _pct(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"


def write_report(audit: dict[str, Any]) -> None:
    """Write the temporal-gate report."""

    dev = audit["development"]["summary_40bps"]
    lines = [
        "# 因子 F 的 20 周时间序列动量仓位覆盖层",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 边界：2024+ 仍是历史时间切分；任何通过只允许进入不可变前向测试。",
        "",
        "## 固定机制",
        "",
        "F 股票选择完全不变。每个周决策收盘，用 F 已实现净值相对 20 个决策日前的涨跌决定下一开盘",
        "持有完整 F 或全部现金；阈值为 0，无迟滞、无杠杆。",
        "",
        "## 开发期",
        "",
        f"- 风险开启率：{_pct(dev['risk_on_decision_rate'])}。",
        f"- 年化收益：F {_pct(dev['baseline']['annualized_return'])}，TSMOM {_pct(dev['candidate']['annualized_return'])}；",
        f"  Sharpe 改善 {dev['sharpe_improvement']:.3f}。",
        f"- 最大回撤：F {_pct(dev['baseline']['maximum_drawdown'])}，TSMOM {_pct(dev['candidate']['maximum_drawdown'])}；",
        f"  相对改善 {_pct(dev['relative_drawdown_reduction'])}。",
        f"- 开发门：**{'PASS' if audit['development']['gate']['passed'] else 'FAIL'}**。",
        "",
    ]
    for name, passed in audit["development"]["gate"]["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    if not audit["validation_opened"]:
        lines.extend(
            [
                "",
                "## 结论",
                "",
                "开发门未全部通过，因此未计算或写出 2024+ 路径与统计。该精确 20 周零阈值覆盖层被否决，",
                "不调整回看期、阈值、迟滞或杠杆救援。",
            ]
        )
    else:
        val = audit["validation"]["summary_40bps"]
        lines.extend(
            [
                "",
                "## 锁定验证期",
                "",
                f"- 风险开启率：{_pct(val['risk_on_decision_rate'])}。",
                f"- 年化收益：F {_pct(val['baseline']['annualized_return'])}，TSMOM {_pct(val['candidate']['annualized_return'])}；",
                f"  Sharpe 改善 {val['sharpe_improvement']:.3f}。",
                f"- 最大回撤相对改善：{_pct(val['relative_drawdown_reduction'])}。",
                f"- 验证门：**{'PASS' if audit['validation']['gate']['passed'] else 'FAIL'}**；",
                f"  60bp 压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
                "",
                "## 结论",
                "",
                audit["conclusion"],
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run development first and open validation only on a full pass."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_parquet(
        _resolve(protocol["inputs"]["ranked_surface"]["path"]),
        columns=[
            "decision_dt",
            "symbol",
            "factor_rank",
            "entry_tradable",
            "fwd_5d_open_return",
            "exit_5d_dt",
        ],
    )
    ranked["decision_dt"] = pd.to_datetime(ranked["decision_dt"])
    memberships = engine.build_target_memberships(ranked)
    memberships.to_parquet(OUTPUT_DIR / protocol["outputs"]["target_memberships"], index=False)
    frozen = pd.read_parquet(_resolve(protocol["inputs"]["historical_stage2_f_path"]["path"]))
    reconstruction = engine.reconstruct_stage2_proxy(ranked, memberships, frozen)
    market = load_price_market(protocol, sorted(memberships["symbol"].unique()))
    dev_start = pd.Timestamp(protocol["sample"]["development"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    val_start = pd.Timestamp(protocol["sample"]["validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["validation"][1])
    terminal_date = pd.Timestamp(ranked["exit_5d_dt"].max())
    if terminal_date != val_end:
        raise RuntimeError("terminal date drift")

    base40_dev, _, _ = engine.simulate_path(
        memberships,
        market,
        overlay=False,
        buy_cost=0.0015,
        sell_cost=0.0025,
        end_date=dev_end,
        terminal_date=terminal_date,
    )
    dev_targets, dev_signals = build_tsmom_targets(
        memberships, base40_dev, lookback=int(protocol["overlay"]["lookback_decisions"]), end_date=dev_end
    )
    development: dict[int, dict[str, Any]] = {}
    for bps in (20, 40, 60):
        buy_cost, sell_cost = engine.cost_pair(bps)
        base_path, _, base_complete = engine.simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=dev_end,
            terminal_date=terminal_date,
        )
        gate_path, _, gate_complete = engine.simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=dev_end,
            terminal_date=terminal_date,
            decision_targets=dev_targets,
        )
        development[bps] = {
            "baseline_path": base_path,
            "candidate_path": gate_path,
            "baseline_completeness": base_complete,
            "candidate_completeness": gate_complete,
            "summary": summarize_comparison(base_path, gate_path, dev_signals, protocol, dev_start, dev_end),
        }
    dev_gate = evaluate_gate(development[40]["summary"], protocol)
    dev_signals.to_parquet(OUTPUT_DIR / protocol["outputs"]["development_signals"], index=False)
    pd.concat(
        [
            development[40]["baseline_path"].assign(arm="F_BASELINE"),
            development[40]["candidate_path"].assign(arm="F_TSMOM"),
        ],
        ignore_index=True,
    ).to_parquet(OUTPUT_DIR / protocol["outputs"]["development_paths"], index=False)
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "reconstruction": reconstruction,
        "development": {
            "summary_20bps": development[20]["summary"],
            "summary_40bps": development[40]["summary"],
            "summary_60bps": development[60]["summary"],
            "gate": dev_gate,
        },
        "validation_opened": bool(dev_gate["passed"]),
        "live_trading_authorized": False,
    }
    if not dev_gate["passed"]:
        audit["status"] = "HYPOTHESIS_FALSIFIED_DEVELOPMENT"
        audit["conclusion"] = "开发门失败；锁定验证未打开。"
    else:
        base40_full, _, _ = engine.simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=0.0015,
            sell_cost=0.0025,
            end_date=terminal_date,
            terminal_date=terminal_date,
        )
        full_targets, full_signals = build_tsmom_targets(
            memberships,
            base40_full,
            lookback=int(protocol["overlay"]["lookback_decisions"]),
            end_date=terminal_date,
        )
        full_runs: dict[int, dict[str, Any]] = {}
        for bps in (20, 40, 60):
            buy_cost, sell_cost = engine.cost_pair(bps)
            base_path, _, base_complete = engine.simulate_path(
                memberships,
                market,
                overlay=False,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                end_date=terminal_date,
                terminal_date=terminal_date,
            )
            gate_path, _, gate_complete = engine.simulate_path(
                memberships,
                market,
                overlay=False,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                end_date=terminal_date,
                terminal_date=terminal_date,
                decision_targets=full_targets,
            )
            full_runs[bps] = {
                "baseline_path": base_path,
                "candidate_path": gate_path,
                "baseline_completeness": base_complete,
                "candidate_completeness": gate_complete,
                "summary": summarize_comparison(base_path, gate_path, full_signals, protocol, val_start, val_end),
            }
        val_gate = evaluate_gate(full_runs[40]["summary"], protocol)
        subperiods = validation_subperiods(
            full_runs[40]["baseline_path"], full_runs[40]["candidate_path"], protocol
        )
        subperiod_check = all(
            row["candidate_total_return"] > 0 and row["paired_total_return"] >= -0.02 for row in subperiods
        )
        val_gate["checks"]["validation_subperiods"] = subperiod_check
        val_gate["passed"] = all(val_gate["checks"].values())
        stress_checks = {
            "development_60bps_candidate_return_positive": development[60]["summary"]["candidate"][
                "annualized_return"
            ]
            > 0,
            "development_60bps_drawdown_lower": development[60]["summary"]["relative_drawdown_reduction"] > 0,
            "validation_60bps_candidate_return_positive": full_runs[60]["summary"]["candidate"][
                "annualized_return"
            ]
            > 0,
            "validation_60bps_drawdown_lower": full_runs[60]["summary"]["relative_drawdown_reduction"] > 0,
        }
        stress_gate = {"passed": all(stress_checks.values()), "checks": stress_checks}
        full_signals[full_signals["decision_dt"] >= val_start].to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_signals"], index=False
        )
        pd.concat(
            [
                full_runs[40]["baseline_path"].assign(arm="F_BASELINE"),
                full_runs[40]["candidate_path"].assign(arm="F_TSMOM"),
            ],
            ignore_index=True,
        ).query("dt >= @val_start").to_parquet(OUTPUT_DIR / protocol["outputs"]["validation_paths"], index=False)
        passed = bool(val_gate["passed"] and stress_gate["passed"])
        audit["validation"] = {
            "summary_20bps": full_runs[20]["summary"],
            "summary_40bps": full_runs[40]["summary"],
            "summary_60bps": full_runs[60]["summary"],
            "subperiods": subperiods,
            "gate": val_gate,
        }
        audit["cost_stress"] = stress_gate
        audit["status"] = (
            "RETROSPECTIVE_TSMOM_CANDIDATE_FORWARD_TEST_REQUIRED"
            if passed
            else "HYPOTHESIS_FALSIFIED_VALIDATION_OR_COST_STRESS"
        )
        audit["conclusion"] = (
            "开发、锁定验证和 60bp 压力均通过；该规则只能进入不可变独立前向测试。"
            if passed
            else "锁定验证或 60bp 压力失败；该精确覆盖层被否决，不做参数救援。"
        )
    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit)
    print(json.dumps({"status": audit["status"], "validation_opened": audit["validation_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
