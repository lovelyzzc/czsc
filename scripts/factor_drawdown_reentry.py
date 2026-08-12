"""Validate the post-development inverse 20-decision factor exposure rule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "factor_drawdown_reentry_protocol_2026-08-13.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "factor_drawdown_reentry"
REPORT_PATH = SCRIPT_DIR / "FACTOR_DRAWDOWN_REENTRY_2026-08-13.md"

CORE_SPEC = importlib.util.spec_from_file_location(
    "factor_tsmom_core_for_drawdown_reentry", SCRIPT_DIR / "factor_tsmom_overlay.py"
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
    """Bind the exact inverse-rule protocol, source data, and genesis audit."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "factor_drawdown_reentry_protocol_v1":
        raise RuntimeError("unexpected drawdown-reentry protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    records = {**protocol["inputs"], "genesis_audit": protocol["genesis"]["audit"]}
    for name, record in records.items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    genesis = json.loads(_resolve(protocol["genesis"]["audit"]["path"]).read_text(encoding="utf-8"))
    if genesis["status"] != "HYPOTHESIS_FALSIFIED_DEVELOPMENT" or genesis["validation_opened"] is not False:
        raise RuntimeError("drawdown-reentry genesis status drift")
    return protocol


def analysis_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Map renamed sample labels onto the shared metric engine."""

    mapped = dict(protocol)
    mapped["sample"] = {
        **protocol["sample"],
        "annualization_sessions": protocol["sample"]["annualization_sessions"],
        "weekly_hac_lag": protocol["sample"]["weekly_hac_lag"],
        "validation_subperiods": protocol["sample"]["locked_validation_subperiods"],
    }
    return mapped


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
    """Write the locked inverse-rule validation report."""

    dev = audit["development_descriptive"]["summary_40bps"]
    val = audit["locked_validation"]["summary_40bps"]
    lines = [
        "# 因子 F 的 20 周回撤再进入覆盖层",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 起源：规则方向来自正趋势覆盖层在开发期的失败；开发期数字禁止用于确认。",
        "- 边界：2024+ 是首次锁定时间检验，但仍非独立前向样本。",
        "",
        "## 固定机制",
        "",
        "F 选股完全不变。仅当 F 的已实现收盘 NAV 不高于 20 个周决策前时，在下一开盘持有完整 F；",
        "其余时间现金。前 20 个决策为空仓，无迟滞、无杠杆。",
        "",
        "## 开发期描述（禁止确认）",
        "",
        f"- 风险开启率 {_pct(dev['risk_on_decision_rate'])}；年化收益 {_pct(dev['candidate']['annualized_return'])}；",
        f"  Sharpe 改善 {dev['sharpe_improvement']:.3f}；回撤相对改善 {_pct(dev['relative_drawdown_reduction'])}。",
        "",
        "## 锁定验证期（2024+）",
        "",
        f"- 风险开启率：{_pct(val['risk_on_decision_rate'])}。",
        f"- 年化收益：F {_pct(val['baseline']['annualized_return'])}，回撤再进入 {_pct(val['candidate']['annualized_return'])}；",
        f"  Sharpe 改善 {val['sharpe_improvement']:.3f}。",
        f"- 最大回撤：F {_pct(val['baseline']['maximum_drawdown'])}，候选 {_pct(val['candidate']['maximum_drawdown'])}；",
        f"  相对改善 {_pct(val['relative_drawdown_reduction'])}。",
        f"- 验证门：**{'PASS' if audit['locked_validation']['gate']['passed'] else 'FAIL'}**；",
        f"  60bp 压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
        "",
        "### 验证门逐项",
        "",
    ]
    for name, passed in audit["locked_validation"]["gate"]["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## 结论", "", audit["conclusion"]])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the inverse direction once on the previously unopened temporal path."""

    protocol = load_and_verify_protocol()
    metrics_protocol = analysis_protocol(protocol)
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
    market = core.load_price_market(protocol, sorted(memberships["symbol"].unique()))
    terminal_date = pd.Timestamp(ranked["exit_5d_dt"].max())
    base40, _, _ = engine.simulate_path(
        memberships,
        market,
        overlay=False,
        buy_cost=0.0015,
        sell_cost=0.0025,
        end_date=terminal_date,
        terminal_date=terminal_date,
    )
    targets, signals = core.build_tsmom_targets(
        memberships,
        base40,
        lookback=int(protocol["overlay"]["lookback_decisions"]),
        end_date=terminal_date,
        risk_on_positive=False,
    )
    signals.to_parquet(OUTPUT_DIR / protocol["outputs"]["signals"], index=False)
    runs: dict[int, dict[str, Any]] = {}
    dev_start = pd.Timestamp(protocol["sample"]["development_descriptive"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development_descriptive"][1])
    val_start = pd.Timestamp(protocol["sample"]["locked_validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["locked_validation"][1])
    for bps in (20, 40, 60):
        buy_cost, sell_cost = engine.cost_pair(bps)
        baseline, _, base_complete = engine.simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=terminal_date,
            terminal_date=terminal_date,
        )
        candidate, _, candidate_complete = engine.simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=terminal_date,
            terminal_date=terminal_date,
            decision_targets=targets,
        )
        runs[bps] = {
            "baseline_path": baseline,
            "candidate_path": candidate,
            "baseline_completeness": base_complete,
            "candidate_completeness": candidate_complete,
            "development": core.summarize_comparison(
                baseline, candidate, signals, metrics_protocol, dev_start, dev_end
            ),
            "validation": core.summarize_comparison(
                baseline, candidate, signals, metrics_protocol, val_start, val_end
            ),
        }
    pd.concat(
        [runs[40]["baseline_path"].assign(arm="F_BASELINE"), runs[40]["candidate_path"].assign(arm="F_DRAWDOWN_REENTRY")],
        ignore_index=True,
    ).to_parquet(OUTPUT_DIR / protocol["outputs"]["paths"], index=False)
    val_gate = core.evaluate_gate(runs[40]["validation"], metrics_protocol)
    subperiods = core.validation_subperiods(
        runs[40]["baseline_path"], runs[40]["candidate_path"], metrics_protocol
    )
    subperiod_check = all(
        row["candidate_total_return"] > 0 and row["paired_total_return"] >= -0.02 for row in subperiods
    )
    val_gate["checks"]["locked_validation_subperiods"] = subperiod_check
    val_gate["passed"] = all(val_gate["checks"].values())
    stress_checks = {
        "validation_60bps_candidate_return_positive": runs[60]["validation"]["candidate"]["annualized_return"] > 0,
        "validation_60bps_drawdown_lower": runs[60]["validation"]["relative_drawdown_reduction"] > 0,
    }
    stress_gate = {"passed": all(stress_checks.values()), "checks": stress_checks}
    passed = bool(val_gate["passed"] and stress_gate["passed"])
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "reconstruction": reconstruction,
        "development_is_confirmatory": False,
        "development_descriptive": {
            "summary_20bps": runs[20]["development"],
            "summary_40bps": runs[40]["development"],
            "summary_60bps": runs[60]["development"],
        },
        "locked_validation": {
            "summary_20bps": runs[20]["validation"],
            "summary_40bps": runs[40]["validation"],
            "summary_60bps": runs[60]["validation"],
            "subperiods": subperiods,
            "gate": val_gate,
        },
        "cost_stress": stress_gate,
        "live_trading_authorized": False,
        "status": (
            "RETROSPECTIVE_DRAWDOWN_REENTRY_CANDIDATE_FORWARD_TEST_REQUIRED"
            if passed
            else "HYPOTHESIS_FALSIFIED_LOCKED_VALIDATION"
        ),
        "conclusion": (
            "锁定验证和 60bp 压力通过；该开发后生成规则只能进入不可变独立前向测试。"
            if passed
            else "锁定验证或 60bp 压力失败；该精确回撤再进入规则被否决，不做参数救援。"
        ),
    }
    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit)
    print(json.dumps({"status": audit["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
