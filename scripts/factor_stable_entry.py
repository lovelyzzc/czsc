"""Test an entry-only short-term idiosyncratic-shock filter on factor F.

This full-sample retrospective interaction was generated after both tails of a
weekly industry/size-neutral return signal failed.  No historical result is
independent OOS; a pass can create only a prospective forward-test candidate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from czsc._native.research import circular_block_bootstrap, newey_west_hac

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "factor_stable_entry_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "factor_stable_entry"
REPORT_PATH = SCRIPT_DIR / "FACTOR_STABLE_ENTRY_2026-08-12.md"

CORE_SPEC = importlib.util.spec_from_file_location(
    "weekly_residual_reversal_core_for_stable", SCRIPT_DIR / "weekly_residual_reversal.py"
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
    """Load the frozen full-sample-generated protocol and bind every input."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "factor_stable_entry_protocol_v1":
        raise RuntimeError("unexpected stable-entry protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    records = {
        **protocol["inputs"],
        "reversal_audit": protocol["genesis"]["reversal_audit"],
        "momentum_audit": protocol["genesis"]["momentum_audit"],
    }
    for name, record in records.items():
        if "path" not in record:
            continue
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    return protocol


def build_blind_shock_surface(protocol: dict[str, Any]) -> pd.DataFrame:
    """Attach the causal absolute weekly residual and fixed stable-pool flag."""

    core_protocol = dict(protocol)
    core_protocol["signal"] = {
        "minimum_complete_weekly_universe": protocol["shock_measure"]["minimum_complete_weekly_universe"]
    }
    neutralized = core.rank_reversal_signal(core.build_blind_surface(core_protocol))
    factor = pd.read_parquet(
        _resolve(protocol["inputs"]["ranked_surface"]["path"]),
        columns=["decision_dt", "symbol", "factor_rank"],
    )
    factor["decision_dt"] = pd.to_datetime(factor["decision_dt"])
    shock_columns = [
        "decision_dt",
        "symbol",
        "raw_return_5obs",
        "residual_return_5obs",
    ]
    blind = factor.merge(
        neutralized[shock_columns],
        on=["decision_dt", "symbol"],
        how="left",
        validate="one_to_one",
    )
    blind["absolute_shock"] = blind["residual_return_5obs"].abs()
    cutoff = float(protocol["shock_measure"]["cutoff"])
    blind["stable_cutoff"] = blind.groupby("decision_dt", observed=True)["absolute_shock"].transform(
        lambda values: values.quantile(cutoff)
    )
    blind["stable_new_entry"] = (blind["absolute_shock"] <= blind["stable_cutoff"]).fillna(False)
    return blind.sort_values(["decision_dt", "factor_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def build_arm_memberships(blind: pd.DataFrame, *, target: int = 50, retention_rank: int = 75) -> pd.DataFrame:
    """Build F and entry-filtered F paths with identical retention semantics."""

    previous = {"F_BASELINE": set(), "F_STABLE_ENTRY": set()}
    rows: list[dict[str, Any]] = []
    for decision_dt, week in blind.groupby("decision_dt", sort=True, observed=True):
        week = week.sort_values(["factor_rank", "symbol"], kind="mergesort")
        indexed = week.set_index("symbol", drop=False)
        available = set(indexed.index.astype(str))
        for arm in ("F_BASELINE", "F_STABLE_ENTRY"):
            retained = {
                symbol
                for symbol in previous[arm]
                if symbol in available and int(indexed.at[symbol, "factor_rank"]) <= retention_rank
            }
            candidates = [str(symbol) for symbol in week["symbol"] if str(symbol) not in retained]
            if arm == "F_STABLE_ENTRY":
                candidates = [symbol for symbol in candidates if bool(indexed.at[symbol, "stable_new_entry"])]
            current = retained | set(candidates[: target - len(retained)])
            if len(current) != target:
                raise RuntimeError(f"{arm} underfilled on {pd.Timestamp(decision_dt).date()}")
            for symbol in sorted(current, key=lambda item: (int(indexed.at[item, "factor_rank"]), item)):
                rows.append(
                    {
                        "decision_dt": pd.Timestamp(decision_dt),
                        "arm": arm,
                        "symbol": symbol,
                        "factor_rank": int(indexed.at[symbol, "factor_rank"]),
                        "absolute_shock": float(indexed.at[symbol, "absolute_shock"]),
                        "stable_new_entry": bool(indexed.at[symbol, "stable_new_entry"]),
                        "membership_role": "RETAINED" if symbol in retained else "NEW",
                    }
                )
            previous[arm] = current
    return pd.DataFrame(rows)


def build_paired_weekly(
    protocol: dict[str, Any], memberships: pd.DataFrame, end_date: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build both outcome paths and prove the F arm reconstructs Stage-2."""

    arm_frames = []
    for arm in ("F_BASELINE", "F_STABLE_ENTRY"):
        weekly = core.build_weekly_outcomes(
            protocol,
            memberships[memberships["arm"].eq(arm)],
            end_date=end_date,
            terminal_liquidation=True,
        ).assign(arm=arm)
        arm_frames.append(weekly)
    combined = pd.concat(arm_frames, ignore_index=True)
    base = combined[combined["arm"].eq("F_BASELINE")].sort_values("decision_dt")
    frozen = pd.read_parquet(_resolve(protocol["inputs"]["historical_stage2_f_path"]["path"]))
    frozen = frozen[frozen["arm"].eq("F")].sort_values("decision_dt")
    if len(base) != len(frozen):
        raise RuntimeError("F reconstruction decision-date count drift")
    checks = {
        "slots": np.array_equal(base["slots"].to_numpy(), frozen["slots"].to_numpy()),
        "buys": np.array_equal(base["buy_count"].to_numpy(), frozen["buy_count"].to_numpy()),
        "sells": np.array_equal(base["sell_count"].to_numpy(), frozen["sell_count"].to_numpy()),
        "gross": np.allclose(base["gross_return"], frozen["gross_return"], rtol=0, atol=1e-15),
    }
    if not all(checks.values()):
        raise RuntimeError(f"F reconstruction failed: {checks}")
    reconstruction = {
        "status": "PASSED",
        "checks": checks,
        "weeks": int(len(base)),
        "maximum_gross_return_error": float(np.max(np.abs(base["gross_return"] - frozen["gross_return"]))),
    }
    return combined, reconstruction


def summarize_pair(weekly: pd.DataFrame, protocol: dict[str, Any], *, bps: int) -> dict[str, Any]:
    """Summarize the fixed paired improvement and four consistency periods."""

    pivot = weekly.pivot(index="decision_dt", columns="arm", values=f"net_return_{bps}bps").sort_index()
    paired = (pivot["F_STABLE_ENTRY"] - pivot["F_BASELINE"]).to_numpy(dtype=float)
    hac = {key: float(value) for key, value in newey_west_hac(paired.tolist(), lag=4).items()}
    hac["one_sided_positive_p"] = float(norm.sf(hac["t_stat"]))
    bootstrap = {
        key: float(value)
        for key, value in circular_block_bootstrap(
            paired.tolist(),
            block_size=int(protocol["statistics"]["circular_block_weeks"]),
            n_draws=int(protocol["statistics"]["bootstrap_draws"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        ).items()
    }
    stable = pivot["F_STABLE_ENTRY"].to_numpy(dtype=float)
    wealth = np.cumprod(1 + stable)
    running_max = np.maximum.accumulate(np.r_[1.0, wealth])
    max_drawdown = float(np.min(np.r_[1.0, wealth] / running_max - 1))
    subperiods = []
    for label, start, end in protocol["sample"]["fixed_subperiods"]:
        part = pivot[pivot.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()]
        subperiods.append(
            {
                "period": label,
                "weeks": int(len(part)),
                "paired_mean": float((part["F_STABLE_ENTRY"] - part["F_BASELINE"]).mean()),
                "stable_net_mean": float(part["F_STABLE_ENTRY"].mean()),
                "baseline_net_mean": float(part["F_BASELINE"].mean()),
            }
        )
    stable_rows = weekly[weekly["arm"].eq("F_STABLE_ENTRY")]
    return {
        "weeks": int(len(pivot)),
        "paired_mean": float(np.mean(paired)),
        "stable_net_mean": float(np.mean(stable)),
        "baseline_net_mean": float(pivot["F_BASELINE"].mean()),
        "paired_win_rate": float(np.mean(paired > 0)),
        "stable_observable_target_slot_rate": float(stable_rows["observable_target_slot_rate"].mean()),
        "stable_compounded_return": float(wealth[-1] - 1),
        "stable_compounded_max_drawdown": max_drawdown,
        "hac": hac,
        "bootstrap": bootstrap,
        "subperiods": subperiods,
    }


def evaluate_candidate(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen descriptive consistency rule."""

    stats = protocol["statistics"]
    checks = {
        "minimum_paired_improvement": summary["paired_mean"] >= float(stats["minimum_weekly_improvement"]),
        "hac_positive": summary["hac"]["one_sided_positive_p"] < float(stats["one_sided_alpha"]),
        "bootstrap_q05_positive": summary["bootstrap"]["q05"] > 0,
        "all_subperiod_paired_positive": all(row["paired_mean"] > 0 for row in summary["subperiods"]),
        "all_subperiod_stable_net_positive": all(row["stable_net_mean"] > 0 for row in summary["subperiods"]),
        "observable_rate": summary["stable_observable_target_slot_rate"]
        >= float(stats["minimum_observable_target_slot_rate"]),
        "drawdown_cap": abs(summary["stable_compounded_max_drawdown"])
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
    """Write a report that preserves the full-sample post-selection boundary."""

    summary = audit["summary_40bps"]
    lines = [
        "# 因子 F 的短期冲击稳定开仓过滤：全样本回顾性检验",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        "- 边界：机制由已查看的正负两侧历史失败生成；所有显著性只作一致性诊断，不能当作 OOS。",
        "",
        "## 固定策略",
        "",
        "F 的排名、50 个槽位和 75 名保留线完全不变。仅在补充新仓时，排除行业/规模中性 5 日收益",
        "绝对残差最高的 20%；已有仓位不因冲击被强制卖出。",
        "",
        "## 结果（主成本 40bp）",
        "",
        f"- F-Stable 周均：{_pct(summary['stable_net_mean'])}；F 周均：{_pct(summary['baseline_net_mean'])}；",
        f"  配对改善：{_pct(summary['paired_mean'])}。",
        f"- HAC t={summary['hac']['t_stat']:.3f}；bootstrap q05={_pct(summary['bootstrap']['q05'])}；",
        f"  F-Stable 最大回撤={_pct(summary['stable_compounded_max_drawdown'])}。",
        "",
        "| 子段 | 周数 | 配对改善 | F-Stable 周均 | F 周均 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["subperiods"]:
        lines.append(
            f"| {row['period']} | {row['weeks']} | {_pct(row['paired_mean'])} | "
            f"{_pct(row['stable_net_mean'])} | {_pct(row['baseline_net_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"回顾性候选门：**{'PASS' if audit['candidate_gate']['passed'] else 'FAIL'}**；",
            f"60bp 压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
            "",
            "## 结论",
            "",
            audit["conclusion"],
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the exact full-sample-generated interaction once."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blind = build_blind_shock_surface(protocol)
    blind.to_parquet(OUTPUT_DIR / protocol["outputs"]["blind_shock_surface"], index=False)
    memberships = build_arm_memberships(blind)
    memberships.to_parquet(OUTPUT_DIR / protocol["outputs"]["memberships"], index=False)
    end_date = pd.Timestamp(protocol["sample"]["full"][1])
    weekly, reconstruction = build_paired_weekly(protocol, memberships, end_date)
    weekly.to_parquet(OUTPUT_DIR / protocol["outputs"]["weekly"], index=False)
    summaries = {bps: summarize_pair(weekly, protocol, bps=bps) for bps in (20, 40, 60)}
    candidate_gate = evaluate_candidate(summaries[40], protocol)
    stress_checks = {
        "full_60bps_paired_mean_positive": summaries[60]["paired_mean"] > 0,
        "full_60bps_bootstrap_q05_positive": summaries[60]["bootstrap"]["q05"] > 0,
        "all_60bps_subperiod_stable_net_positive": all(
            row["stable_net_mean"] > 0 for row in summaries[60]["subperiods"]
        ),
    }
    stress_gate = {"passed": all(stress_checks.values()), "checks": stress_checks}
    passed = bool(candidate_gate["passed"] and stress_gate["passed"])
    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "reconstruction": reconstruction,
        "blind_surface_rows": int(len(blind)),
        "membership_rows": int(len(memberships)),
        "summary_20bps": summaries[20],
        "summary_40bps": summaries[40],
        "summary_60bps": summaries[60],
        "candidate_gate": candidate_gate,
        "cost_stress": stress_gate,
        "live_trading_authorized": False,
        "status": (
            "RETROSPECTIVE_STABLE_ENTRY_CANDIDATE_FORWARD_TEST_REQUIRED"
            if passed
            else "HYPOTHESIS_FALSIFIED_RETROSPECTIVE"
        ),
        "conclusion": (
            "四个历史子段和 60bp 压力均一致通过；由于全样本生成，该策略只能进入不可变前向测试。"
            if passed
            else "回顾性候选门或 60bp 压力门失败；该精确稳定开仓过滤被否决，不做参数救援。"
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
