"""Discover one classic weekly factor with Holm-corrected development tests."""

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
PROTOCOL_PATH = SCRIPT_DIR / "classic_factor_family_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "classic_factor_family"
REPORT_PATH = SCRIPT_DIR / "CLASSIC_FACTOR_FAMILY_2026-08-12.md"

CORE_SPEC = importlib.util.spec_from_file_location(
    "weekly_factor_core", SCRIPT_DIR / "weekly_residual_reversal.py"
)
assert CORE_SPEC and CORE_SPEC.loader
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)


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
    """Load the frozen family and bind the source surface."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "classic_factor_family_protocol_v1":
        raise RuntimeError("unexpected classic-factor protocol schema")
    if protocol.get("live_trading_authorized") is not False or len(protocol["family"]) != 5:
        raise RuntimeError("classic-factor family boundary drift")
    record = protocol["inputs"]["ranked_surface"]
    actual = sha256_file(_resolve(record["path"]))
    if actual != record["sha256"]:
        raise RuntimeError(f"frozen ranked-surface drift: expected={record['sha256']} actual={actual}")
    return protocol


def raw_factor(frame: pd.DataFrame, factor_id: str) -> pd.Series:
    """Return the exact frozen raw factor, leaving invalid inputs missing."""

    if factor_id == "VALUE_PB":
        return -np.log(frame["pb"].where(frame["pb"] > 0))
    if factor_id == "EARNINGS_YIELD":
        return (1 / frame["pe_ttm"]).where(frame["pe_ttm"] > 0)
    if factor_id == "LOW_TURNOVER":
        return -np.log(frame["turnover_rate"].where(frame["turnover_rate"] > 0))
    if factor_id == "NORMAL_VOLUME":
        return -np.log(frame["volume_ratio"].where(frame["volume_ratio"] > 0)).abs()
    if factor_id == "SMALL_CAP":
        return -frame["log_free_float_mcap"]
    raise ValueError(f"unknown factor: {factor_id}")


def rank_factor_family(protocol: dict[str, Any]) -> pd.DataFrame:
    """Rank all five causal candidates before joining any forward outcome."""

    source = pd.read_parquet(
        _resolve(protocol["inputs"]["ranked_surface"]["path"]),
        columns=[
            "decision_dt",
            "symbol",
            "industry_code",
            "log_free_float_mcap",
            "pb",
            "pe_ttm",
            "turnover_rate",
            "volume_ratio",
        ],
    )
    source["decision_dt"] = pd.to_datetime(source["decision_dt"])
    minimum = int(protocol["signal"]["minimum_complete_weekly_universe"])
    parts = []
    for family_index, definition in enumerate(protocol["family"]):
        factor_id = definition["id"]
        candidate = source.assign(raw_score=raw_factor(source, factor_id)).dropna(
            subset=["raw_score", "industry_code", "log_free_float_mcap"]
        )
        rows = []
        for decision_dt, week in candidate.groupby("decision_dt", sort=True, observed=True):
            week = week.sort_values("symbol", kind="mergesort").copy()
            if len(week) < minimum:
                raise RuntimeError(f"{factor_id} universe below {minimum} on {pd.Timestamp(decision_dt).date()}")
            low, high = week["raw_score"].quantile([0.01, 0.99])
            y = week["raw_score"].clip(low, high).to_numpy(dtype=float)
            industries = pd.Categorical(
                week["industry_code"].astype(str), categories=sorted(week["industry_code"].astype(str).unique())
            )
            dummies = pd.get_dummies(industries, drop_first=True, dtype=float).to_numpy(dtype=float)
            columns = [np.ones(len(week))]
            if factor_id != "SMALL_CAP":
                mcap = week["log_free_float_mcap"].to_numpy(dtype=float)
                columns.append((mcap - np.mean(mcap)) / np.std(mcap, ddof=0))
            columns.append(dummies)
            design = np.column_stack(columns)
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            week["factor_score"] = y - design @ beta
            week = week.sort_values(["factor_score", "symbol"], ascending=[False, True], kind="mergesort")
            week["factor_rank"] = np.arange(1, len(week) + 1, dtype=np.int64)
            week["factor_id"] = factor_id
            week["family_index"] = family_index
            rows.append(week[["decision_dt", "symbol", "factor_id", "family_index", "raw_score", "factor_score", "factor_rank"]])
        parts.append(pd.concat(rows, ignore_index=True))
    return pd.concat(parts, ignore_index=True).sort_values(
        ["factor_id", "decision_dt", "factor_rank", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def build_family_memberships(blind: pd.DataFrame) -> pd.DataFrame:
    """Build an independent 50/75 recursive path for every candidate."""

    parts = []
    for factor_id, group in blind.groupby("factor_id", sort=False, observed=True):
        group = group.assign(residual_return_5obs=group["factor_score"])
        memberships = core.build_memberships(group, rank_column="factor_rank")
        memberships["factor_id"] = factor_id
        parts.append(memberships)
    return pd.concat(parts, ignore_index=True)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return named Holm step-down adjusted p-values."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def outcome_paths(
    protocol: dict[str, Any], memberships: pd.DataFrame, *, end_date: pd.Timestamp, terminal: bool
) -> pd.DataFrame:
    """Build each candidate path only after blind ranks are fixed."""

    parts = []
    for factor_id, group in memberships.groupby("factor_id", sort=False, observed=True):
        parts.append(
            core.build_weekly_outcomes(protocol, group, end_date=end_date, terminal_liquidation=terminal).assign(
                factor_id=factor_id
            )
        )
    return pd.concat(parts, ignore_index=True)


def summarize_family(
    weekly: pd.DataFrame,
    protocol: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    halves: list[list[str]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Summarize all candidates and costs without selecting a best result."""

    result = {}
    family_order = {row["id"]: index for index, row in enumerate(protocol["family"])}
    for factor_id, group in weekly.groupby("factor_id", sort=False, observed=True):
        mapped = {**protocol, "statistics": dict(protocol["statistics"])}
        mapped["statistics"]["circular_block_weeks"] = protocol["statistics"]["block_size"]
        mapped["statistics"]["bootstrap_seed"] = int(protocol["statistics"]["bootstrap_seed_root"]) + family_order[
            factor_id
        ]
        result[factor_id] = {
            bps: core.summarize_segment(group, mapped, start, end, bps=bps, halves=halves)
            for bps in (20, 40, 60)
        }
    return result


def development_decision(summaries: dict[str, dict[int, dict[str, Any]]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply Holm correction and select at most one development winner."""

    p_values = {name: result[40]["hac"]["one_sided_positive_p"] for name, result in summaries.items()}
    adjusted = holm_adjust(p_values)
    rows = []
    passing = []
    stats = protocol["statistics"]
    for factor_id, cost_map in summaries.items():
        summary = cost_map[40]
        checks = {
            "minimum_active_effect": summary["active_mean"] >= float(stats["minimum_active_weekly_effect"]),
            "holm_p": adjusted[factor_id] < float(stats["one_sided_alpha"]),
            "bootstrap_q05": summary["bootstrap"]["q05"] > 0,
            "both_halves": all(row["active_mean"] > 0 for row in summary["half_means"]),
            "net_mean": summary["candidate_net_mean"] > 0,
            "observable": summary["observable_target_slot_rate"]
            >= float(stats["minimum_observable_target_slot_rate"]),
            "drawdown": abs(summary["compounded_candidate_max_drawdown"])
            <= float(stats["maximum_compounded_drawdown"]),
        }
        passed = all(checks.values())
        row = {
            "factor_id": factor_id,
            "raw_p": p_values[factor_id],
            "holm_p": adjusted[factor_id],
            "passed": passed,
            "checks": checks,
        }
        rows.append(row)
        if passed:
            passing.append(factor_id)
    selected = None
    if passing:
        selected = sorted(
            passing,
            key=lambda name: (-summaries[name][40]["bootstrap"]["q05"], -summaries[name][40]["active_mean"], name),
        )[0]
    return {"family": rows, "selected_factor": selected, "passed": selected is not None}


def validation_gate(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the locked single-winner validation rule."""

    stats = protocol["statistics"]
    checks = {
        "minimum_active_effect": summary["active_mean"] >= float(stats["minimum_active_weekly_effect"]),
        "hac_p": summary["hac"]["one_sided_positive_p"] < float(stats["one_sided_alpha"]),
        "bootstrap_q05": summary["bootstrap"]["q05"] > 0,
        "both_halves": all(row["active_mean"] > 0 for row in summary["half_means"]),
        "net_mean": summary["candidate_net_mean"] > 0,
        "observable": summary["observable_target_slot_rate"] >= float(stats["minimum_observable_target_slot_rate"]),
        "drawdown": abs(summary["compounded_candidate_max_drawdown"])
        <= float(stats["maximum_compounded_drawdown"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


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
    return "NA" if value is None or not np.isfinite(value) else f"{value * 100:.3f}%"


def write_report(audit: dict[str, Any]) -> None:
    """Write the multiplicity-aware family report."""

    lines = [
        "# 五个经典横截面因子：多重校正发现与锁定验证",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 边界：全部是历史样本；开发期对 5 个候选做 Holm 校正，最多一个进入 2024+。",
        "",
        "## 开发期（2022–2023，主成本 40bp）",
        "",
        "| 因子 | 主动周均 | HAC t | 原始 p | Holm p | bootstrap q05 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["development_decision"]["family"]:
        summary = audit["development_summaries"][row["factor_id"]]["40"]
        lines.append(
            f"| {row['factor_id']} | {_pct(summary['active_mean'])} | {summary['hac']['t_stat']:.3f} | "
            f"{row['raw_p']:.4f} | {row['holm_p']:.4f} | {_pct(summary['bootstrap']['q05'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    if not audit["validation_opened"]:
        lines.extend(
            [
                "",
                "## 结论",
                "",
                "没有候选通过完整开发门，故未打开 2024+ 结果。这个五因子族被回顾性否决，不调整符号、",
                "变换、中性化、持仓数、缓冲或持有期救援。",
            ]
        )
    else:
        selected = audit["development_decision"]["selected_factor"]
        val = audit["validation_summaries"][selected]["40"]
        lines.extend(
            [
                "",
                f"## 锁定验证：{selected}",
                "",
                f"- 主动周均 {_pct(val['active_mean'])}；HAC t={val['hac']['t_stat']:.3f}；",
                f"  bootstrap q05={_pct(val['bootstrap']['q05'])}。",
                f"- 验证门：**{'PASS' if audit['validation_gate']['passed'] else 'FAIL'}**；",
                f"  60bp 压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
                "",
                "## 结论",
                "",
                audit["conclusion"],
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run development family selection and open one validation path at most."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blind = rank_factor_family(protocol)
    blind.to_parquet(OUTPUT_DIR / protocol["outputs"]["blind_ranks"], index=False)
    memberships = build_family_memberships(blind)
    dev_start = pd.Timestamp(protocol["sample"]["development"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    val_start = pd.Timestamp(protocol["sample"]["validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["validation"][1])
    memberships[memberships["decision_dt"].between(dev_start, dev_end)].to_parquet(
        OUTPUT_DIR / protocol["outputs"]["development_memberships"], index=False
    )
    dev_weekly = outcome_paths(protocol, memberships, end_date=dev_end, terminal=False)
    dev_weekly.to_parquet(OUTPUT_DIR / protocol["outputs"]["development_weekly"], index=False)
    dev_summaries = summarize_family(
        dev_weekly, protocol, dev_start, dev_end, protocol["sample"]["development_halves"]
    )
    decision = development_decision(dev_summaries, protocol)
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "blind_rows": int(len(blind)),
        "development_summaries": dev_summaries,
        "development_decision": decision,
        "validation_opened": bool(decision["passed"]),
        "live_trading_authorized": False,
    }
    if not decision["passed"]:
        audit["status"] = "FAMILY_FALSIFIED_DEVELOPMENT"
        audit["conclusion"] = "无开发期 Holm 合格候选；锁定验证未打开。"
    else:
        selected = decision["selected_factor"]
        selected_memberships = memberships[memberships["factor_id"].eq(selected)]
        selected_memberships[selected_memberships["decision_dt"] >= val_start].to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_memberships"], index=False
        )
        val_weekly = outcome_paths(protocol, selected_memberships, end_date=val_end, terminal=True)
        val_weekly[val_weekly["decision_dt"] >= val_start].to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_weekly"], index=False
        )
        val_summaries = summarize_family(
            val_weekly, protocol, val_start, val_end, protocol["sample"]["validation_halves"]
        )
        gate = validation_gate(val_summaries[selected][40], protocol)
        stress_checks = {
            "development_60bps_active_mean_positive": dev_summaries[selected][60]["active_mean"] > 0,
            "development_60bps_bootstrap_q05_positive": dev_summaries[selected][60]["bootstrap"]["q05"] > 0,
            "validation_60bps_active_mean_positive": val_summaries[selected][60]["active_mean"] > 0,
            "validation_60bps_bootstrap_q05_positive": val_summaries[selected][60]["bootstrap"]["q05"] > 0,
        }
        stress = {"passed": all(stress_checks.values()), "checks": stress_checks}
        passed = bool(gate["passed"] and stress["passed"])
        audit["validation_summaries"] = val_summaries
        audit["validation_gate"] = gate
        audit["cost_stress"] = stress
        audit["status"] = (
            "RETROSPECTIVE_CLASSIC_FACTOR_CANDIDATE_FORWARD_TEST_REQUIRED"
            if passed
            else "FAMILY_WINNER_FALSIFIED_VALIDATION_OR_COST_STRESS"
        )
        audit["conclusion"] = (
            "开发、锁定验证和 60bp 压力通过；该经典因子只能进入独立前向测试。"
            if passed
            else "开发胜者未通过锁定验证或成本压力；本族被否决，不做参数救援。"
        )
    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit)
    print(json.dumps({"status": audit["status"], "validation_opened": audit["validation_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
