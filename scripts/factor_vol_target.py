"""Test a fixed 20-session, 15% target-volatility overlay on factor F."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "factor_vol_target_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "factor_vol_target"
REPORT_PATH = SCRIPT_DIR / "FACTOR_VOL_TARGET_2026-08-12.md"

CORE_SPEC = importlib.util.spec_from_file_location(
    "factor_tsmom_core_for_vol_target", SCRIPT_DIR / "factor_tsmom_overlay.py"
)
assert CORE_SPEC and CORE_SPEC.loader
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)
engine = core.engine


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
    """Load the final same-sample protocol and bind all inputs."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "factor_vol_target_protocol_v1":
        raise RuntimeError("unexpected vol-target protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    for name, record in protocol["inputs"].items():
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    return protocol


def build_exposure_signals(
    memberships: pd.DataFrame,
    baseline: pd.DataFrame,
    protocol: dict[str, Any],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute weekly causal realized volatility and next-open target exposure."""

    lookback = int(protocol["overlay"]["lookback_market_sessions"])
    annualization = int(protocol["overlay"]["annualization_sessions"])
    target_vol = float(protocol["overlay"]["target_annualized_volatility"])
    decisions = sorted(date for date in memberships["decision_dt"].unique() if date <= end_date)
    path = baseline.set_index("dt").sort_index()
    returns = path["nav"].pct_change()
    calendar = list(path.index)
    next_session = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(calendar[:-1], calendar[1:], strict=False)}
    rows = []
    for decision_dt in decisions:
        history = returns.loc[:decision_dt].dropna().tail(lookback)
        if len(history) < lookback:
            realized_vol = math.nan
            exposure = 0.0
        else:
            realized_vol = float(history.std(ddof=1) * math.sqrt(annualization))
            exposure = min(1.0, target_vol / realized_vol) if realized_vol > 0 else 1.0
        rows.append(
            {
                "decision_dt": pd.Timestamp(decision_dt),
                "fill_dt": next_session.get(pd.Timestamp(decision_dt), pd.NaT),
                "realized_volatility": realized_vol,
                "target_exposure": exposure,
            }
        )
    return pd.DataFrame(rows)


def simulate_sleeve(
    baseline: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    buy_cost: float,
    sell_cost: float,
    terminal_date: pd.Timestamp,
) -> pd.DataFrame:
    """Scale an investable F sleeve at next-open fills with exact cost algebra."""

    targets = {
        pd.Timestamp(row.fill_dt): float(row.target_exposure)
        for row in signals.itertuples(index=False)
        if pd.notna(row.fill_dt)
    }
    targets[terminal_date] = 0.0
    ordered = baseline.sort_values("dt").reset_index(drop=True)
    cash = 1.0
    sleeve = 0.0
    previous_close_nav: float | None = None
    rows = []
    for row in ordered.itertuples(index=False):
        dt = pd.Timestamp(row.dt)
        if previous_close_nav is not None:
            sleeve *= float(row.open_nav) / previous_close_nav
        transaction_cost = 0.0
        target = targets.get(dt)
        if target is not None:
            total = cash + sleeve
            desired = float(target)
            if sleeve < desired * total:
                purchase = (desired * total - sleeve) / (1 + desired * buy_cost)
                transaction_cost = purchase * buy_cost
                sleeve += purchase
                cash -= purchase + transaction_cost
            elif sleeve > desired * total:
                sale = (sleeve - desired * total) / (1 - desired * sell_cost)
                transaction_cost = sale * sell_cost
                sleeve -= sale
                cash += sale - transaction_cost
        sleeve *= float(row.nav) / float(row.open_nav)
        total = cash + sleeve
        if cash < -1e-12 or total <= 0:
            raise RuntimeError(f"invalid vol-target accounting on {dt.date()}: cash={cash} nav={total}")
        rows.append(
            {
                "dt": dt,
                "nav": total,
                "exposure": sleeve / total,
                "target_exposure": np.nan if target is None else target,
                "transaction_cost": transaction_cost,
            }
        )
        previous_close_nav = float(row.nav)
    return pd.DataFrame(rows)


def summarize(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    signals: pd.DataFrame,
    protocol: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Summarize the fixed return/risk comparison."""

    base = engine.segment_metrics(baseline, start, end, int(protocol["sample"]["annualization_sessions"]))
    gate = engine.segment_metrics(candidate, start, end, int(protocol["sample"]["annualization_sessions"]))
    hac = engine.active_hac(
        baseline,
        candidate,
        start,
        end,
        lag=int(protocol["sample"]["weekly_hac_lag"]),
        margin_weekly=float(protocol["statistics"]["weekly_noninferiority_margin"]),
    )
    scoped = signals[signals["decision_dt"].between(start, end)]
    return {
        "baseline": base,
        "candidate": gate,
        "average_target_exposure": float(scoped["target_exposure"].mean()),
        "annualized_return_difference": gate["annualized_return"] - base["annualized_return"],
        "sharpe_improvement": gate["sharpe"] - base["sharpe"],
        "relative_drawdown_reduction": (abs(base["maximum_drawdown"]) - abs(gate["maximum_drawdown"]))
        / abs(base["maximum_drawdown"]),
        "weekly_active_hac": hac,
    }


def evaluate_gate(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the common development and validation gate."""

    stats = protocol["statistics"]
    checks = {
        "exposure_floor": summary["average_target_exposure"] >= float(stats["minimum_average_exposure"]),
        "exposure_cap": summary["average_target_exposure"] <= float(stats["maximum_average_exposure"]),
        "positive_return": summary["candidate"]["annualized_return"] > 0,
        "drawdown_reduction": summary["relative_drawdown_reduction"]
        >= float(stats["minimum_relative_drawdown_reduction"]),
        "sharpe_improvement": summary["sharpe_improvement"] >= float(stats["minimum_sharpe_improvement"]),
        "return_noninferiority": summary["annualized_return_difference"]
        >= -float(stats["maximum_annual_return_underperformance"]),
        "hac_noninferiority": summary["weekly_active_hac"]["noninferiority_one_sided_p"]
        < float(stats["one_sided_alpha"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def subperiods(baseline: pd.DataFrame, candidate: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute locked validation subperiod returns."""

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
    """Write the final same-sample research report."""

    dev = audit["development"]["summary_40bps"]
    lines = [
        "# 因子 F 的 20 日 / 15% 目标波动率覆盖层",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 边界：全为历史样本；失败后停止同样本挖掘，等待新不可变数据。",
        "",
        "## 开发期",
        "",
        f"- 平均目标仓位：{_pct(dev['average_target_exposure'])}。",
        f"- 年化收益：F {_pct(dev['baseline']['annualized_return'])}，VolTarget {_pct(dev['candidate']['annualized_return'])}；",
        f"  Sharpe 改善 {dev['sharpe_improvement']:.3f}。",
        f"- 最大回撤：F {_pct(dev['baseline']['maximum_drawdown'])}，VolTarget {_pct(dev['candidate']['maximum_drawdown'])}；",
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
                "开发门未全部通过，2024+ 未打开。该精确波动率覆盖层被否决；按协议应停止在同一历史样本",
                "继续挖掘，转入新不可变数据收集。",
            ]
        )
    else:
        val = audit["validation"]["summary_40bps"]
        lines.extend(
            [
                "",
                "## 锁定验证期",
                "",
                f"- 年化收益：F {_pct(val['baseline']['annualized_return'])}，VolTarget {_pct(val['candidate']['annualized_return'])}；",
                f"  Sharpe 改善 {val['sharpe_improvement']:.3f}；回撤相对改善 {_pct(val['relative_drawdown_reduction'])}。",
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
    """Run development first; validation is fail-closed."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_parquet(
        _resolve(protocol["inputs"]["ranked_surface"]["path"]),
        columns=["decision_dt", "symbol", "factor_rank", "entry_tradable", "fwd_5d_open_return", "exit_5d_dt"],
    )
    ranked["decision_dt"] = pd.to_datetime(ranked["decision_dt"])
    memberships = engine.build_target_memberships(ranked)
    frozen = pd.read_parquet(_resolve(protocol["inputs"]["historical_stage2_f_path"]["path"]))
    reconstruction = engine.reconstruct_stage2_proxy(ranked, memberships, frozen)
    market = core.load_price_market(protocol, sorted(memberships["symbol"].unique()))
    dev_start = pd.Timestamp(protocol["sample"]["development"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    val_start = pd.Timestamp(protocol["sample"]["validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["validation"][1])
    terminal = pd.Timestamp(ranked["exit_5d_dt"].max())
    base40_dev, _, _ = engine.simulate_path(
        memberships, market, overlay=False, buy_cost=0.0015, sell_cost=0.0025, end_date=dev_end, terminal_date=terminal
    )
    dev_signals = build_exposure_signals(memberships, base40_dev, protocol, dev_end)
    development = {}
    for bps in (20, 40, 60):
        buy_cost, sell_cost = engine.cost_pair(bps)
        baseline, _, _ = engine.simulate_path(
            memberships, market, overlay=False, buy_cost=buy_cost, sell_cost=sell_cost, end_date=dev_end, terminal_date=terminal
        )
        candidate = simulate_sleeve(baseline, dev_signals, buy_cost=buy_cost, sell_cost=sell_cost, terminal_date=terminal)
        development[bps] = {"baseline": baseline, "candidate": candidate, "summary": summarize(baseline, candidate, dev_signals, protocol, dev_start, dev_end)}
    dev_gate = evaluate_gate(development[40]["summary"], protocol)
    dev_signals.to_parquet(OUTPUT_DIR / protocol["outputs"]["development_signals"], index=False)
    pd.concat([development[40]["baseline"].assign(arm="F_BASELINE"), development[40]["candidate"].assign(arm="F_VOL_TARGET")], ignore_index=True).to_parquet(
        OUTPUT_DIR / protocol["outputs"]["development_paths"], index=False
    )
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "reconstruction": reconstruction,
        "development": {**{f"summary_{bps}bps": development[bps]["summary"] for bps in (20, 40, 60)}, "gate": dev_gate},
        "validation_opened": bool(dev_gate["passed"]),
        "live_trading_authorized": False,
    }
    if not dev_gate["passed"]:
        audit["status"] = "HYPOTHESIS_FALSIFIED_DEVELOPMENT_STOP_SAME_SAMPLE_MINING"
        audit["conclusion"] = "开发门失败；锁定验证未打开；停止同样本挖掘。"
    else:
        base40_full, _, _ = engine.simulate_path(
            memberships, market, overlay=False, buy_cost=0.0015, sell_cost=0.0025, end_date=terminal, terminal_date=terminal
        )
        signals = build_exposure_signals(memberships, base40_full, protocol, terminal)
        full = {}
        for bps in (20, 40, 60):
            buy_cost, sell_cost = engine.cost_pair(bps)
            baseline, _, _ = engine.simulate_path(
                memberships, market, overlay=False, buy_cost=buy_cost, sell_cost=sell_cost, end_date=terminal, terminal_date=terminal
            )
            candidate = simulate_sleeve(baseline, signals, buy_cost=buy_cost, sell_cost=sell_cost, terminal_date=terminal)
            full[bps] = {"baseline": baseline, "candidate": candidate, "summary": summarize(baseline, candidate, signals, protocol, val_start, val_end)}
        val_gate = evaluate_gate(full[40]["summary"], protocol)
        periods = subperiods(full[40]["baseline"], full[40]["candidate"], protocol)
        period_ok = all(row["candidate_total_return"] > 0 and row["paired_total_return"] >= -0.02 for row in periods)
        val_gate["checks"]["validation_subperiods"] = period_ok
        val_gate["passed"] = all(val_gate["checks"].values())
        stress_checks = {
            "development_60bps_positive": development[60]["summary"]["candidate"]["annualized_return"] > 0,
            "development_60bps_drawdown_lower": development[60]["summary"]["relative_drawdown_reduction"] > 0,
            "validation_60bps_positive": full[60]["summary"]["candidate"]["annualized_return"] > 0,
            "validation_60bps_drawdown_lower": full[60]["summary"]["relative_drawdown_reduction"] > 0,
        }
        stress = {"passed": all(stress_checks.values()), "checks": stress_checks}
        signals[signals["decision_dt"] >= val_start].to_parquet(OUTPUT_DIR / protocol["outputs"]["validation_signals"], index=False)
        pd.concat([full[40]["baseline"].assign(arm="F_BASELINE"), full[40]["candidate"].assign(arm="F_VOL_TARGET")], ignore_index=True).query("dt >= @val_start").to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_paths"], index=False
        )
        passed = bool(val_gate["passed"] and stress["passed"])
        audit["validation"] = {**{f"summary_{bps}bps": full[bps]["summary"] for bps in (20, 40, 60)}, "subperiods": periods, "gate": val_gate}
        audit["cost_stress"] = stress
        audit["status"] = "RETROSPECTIVE_VOL_TARGET_CANDIDATE_FORWARD_TEST_REQUIRED" if passed else "HYPOTHESIS_FALSIFIED_VALIDATION_STOP_SAME_SAMPLE_MINING"
        audit["conclusion"] = "所有门通过；只允许进入独立前向测试。" if passed else "锁定验证失败；停止同样本挖掘。"
    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(audit)
    print(json.dumps({"status": audit["status"], "validation_opened": audit["validation_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
