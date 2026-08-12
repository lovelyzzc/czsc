"""冻结 PIT 精确流通市值匹配对的 H5/H20/H60 结果与稳健推断。

本脚本只有在 ``delay5_pit_exact_mcap_balance_audit`` 明确授权后才读取未来路径。
主估计量是等权处理交易 ATT；每笔交易的控制腿使用余额阶段冻结的条件熵权重。
推断使用完整交易日历上的 ratio-influence Newey-West、固定期望块长的平稳
bootstrap、三期限单侧 Holm 校正，以及处理/控制股票双向聚类协方差。

运行：``uv run --no-sync python scripts/delay5_pit_exact_mcap_outcome_audit.py``。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import delay5_common_horizon_att as common
import delay5_pit_exact_mcap_balance_audit as balance_audit
import delay5_pit_exact_mcap_plan as pit_plan
import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as identity_utils

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLAN_PATH = pit_plan.EXPECTED_PLAN_PATH
BALANCE_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_balance_audit"
BALANCE_AUDIT_PATH = BALANCE_DIR / "audit.json"
FROZEN_PAIRS_PATH = BALANCE_DIR / "matched_pairs.parquet"
PANEL_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_outcome_audit"
TREATED_PATH = OUTPUT_DIR / "treated_outcomes.parquet"
PAIR_OUTCOMES_PATH = OUTPUT_DIR / "control_outcome_pairs.parquet"
TRADE_ATT_PATH = OUTPUT_DIR / "trade_att.parquet"
AUDIT_PATH = OUTPUT_DIR / "audit.json"

SCHEMA = "delay5_pit_exact_mcap_outcome_audit_v1"
EXPECTED_BALANCE_SCHEMA = "delay5_pit_exact_mcap_balance_audit_v2"
HORIZONS = common.HORIZONS
HAC_LAGS = {5: 4, 20: 19, 60: 59}
BOOTSTRAP_BLOCKS = {5: 5, 20: 20, 60: 60}
N_BOOT = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
PRIMARY_SCOPE = "2024plus"


def _repo_locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository: {path}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def validate_outcome_authorization(
    plan: Mapping[str, Any],
    balance: Mapping[str, Any],
) -> dict[str, Any]:
    """绑定冻结计划、余额产物与 outcome 解锁判定；任一漂移即停止。"""

    pit_plan.validate_design_contract(plan)
    if balance.get("schema") != EXPECTED_BALANCE_SCHEMA:
        raise RuntimeError("unexpected exact-mcap balance audit schema")
    if balance.get("outcomes_loaded") is not False:
        raise RuntimeError("balance audit is not outcome blind")
    verdict = balance.get("verdict", {})
    required_verdict = (
        verdict.get("status") == "BALANCE_GATE_PASSED_OUTCOME_EVALUATION_PERMITTED",
        verdict.get("input_complete") is True,
        verdict.get("coverage_sufficient") is True,
        verdict.get("balance_sufficient") is True,
        verdict.get("positivity_sufficient") is True,
        verdict.get("outcome_evaluation_permitted") is True,
        verdict.get("outcomes_loaded") is False,
        verdict.get("live_authorized") is False,
    )
    if not all(required_verdict):
        raise RuntimeError("exact-mcap balance gate does not authorize outcome evaluation")
    plan_sha = identity_utils.sha256_file(PLAN_PATH)
    balance_plan = balance.get("input_identity", {}).get("tracked_plan", {})
    if balance_plan.get("sha256") != plan_sha:
        raise RuntimeError("balance audit is not bound to the current tracked plan")
    pair_record = balance.get("outputs", {}).get("matched_pairs", {})
    if pair_record.get("sha256") != identity_utils.sha256_file(FROZEN_PAIRS_PATH):
        raise RuntimeError("frozen matched-pair identity drift")
    current_artifacts: dict[str, dict[str, Any]] = {}
    for label, path in pit_plan.ARTIFACT_PATHS.items():
        expected = plan.get("source_identity", {}).get("artifacts", {}).get(label, {})
        if expected.get("sha256") != identity_utils.sha256_file(path):
            raise RuntimeError(f"tracked outcome input drift: {label}")
        current_artifacts[label] = {
            "locator": expected.get("locator"),
            "sha256": expected.get("sha256"),
        }
    return {
        "tracked_plan": {"locator": _repo_locator(PLAN_PATH), "sha256": plan_sha},
        "balance_audit": {
            "locator": _repo_locator(BALANCE_AUDIT_PATH),
            "sha256": identity_utils.sha256_file(BALANCE_AUDIT_PATH),
            "schema": balance.get("schema"),
        },
        "frozen_pairs": {
            "locator": _repo_locator(FROZEN_PAIRS_PATH),
            "sha256": pair_record.get("sha256"),
            "rows": int(pair_record.get("rows", -1)),
        },
        "upstream_artifacts": current_artifacts,
    }


def validate_frozen_pairs(pairs: pd.DataFrame, expected_rows: int) -> None:
    """验证权重、键和结果盲列，确保结果阶段没有重匹配。"""

    required = {
        "trade_id",
        "treated_symbol",
        "control_symbol",
        "dec_dt",
        "year",
        "rank",
        "support_ok",
        "control_weight",
    }
    if missing := required - set(pairs):
        raise RuntimeError(f"frozen pairs lack columns: {sorted(missing)}")
    if len(pairs) != expected_rows:
        raise RuntimeError("frozen pair row count drift")
    balance_audit.assert_outcome_blind_columns(pairs, "outcome-stage frozen pairs")
    if not pairs["support_ok"].astype(bool).all():
        raise RuntimeError("outcome-stage frozen pairs contain unsupported edges")
    if pairs.duplicated(["trade_id", "control_symbol"]).any():
        raise RuntimeError("frozen pairs contain duplicate treated/control edges")
    weights = pd.to_numeric(pairs["control_weight"], errors="coerce")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError("frozen control weights are invalid")
    sums = pairs.assign(_weight=weights).groupby("trade_id", sort=False)["_weight"].sum()
    if not np.allclose(sums.to_numpy(float), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("frozen control weights do not sum to one by trade")


def add_control_outcomes(
    pairs: pd.DataFrame,
    treated: pd.DataFrame,
    *,
    open_w: pd.DataFrame,
    close_w: pd.DataFrame,
    mark_w: pd.DataFrame,
    last_dt: pd.Series,
) -> pd.DataFrame:
    """只给冻结边附加未来路径；不新增、删除或替换控制票。"""

    treated_by_id = treated.set_index("trade_id", drop=False)
    if missing := set(pairs["trade_id"]) - set(treated_by_id.index):
        raise RuntimeError(f"frozen pairs reference unknown treated trades: {sorted(missing)[:3]}")
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        treated_row = treated_by_id.loc[pair.trade_id]
        path = common._control_path_from_wide(
            str(pair.control_symbol),
            treated_row,
            open_w=open_w,
            close_w=close_w,
            mark_w=mark_w,
            last_dt=last_dt,
        )
        record: dict[str, Any] = {
            "entry_dt": treated_row["entry_dt"],
            **{f"h{horizon}_dt": treated_row[f"h{horizon}_dt"] for horizon in HORIZONS},
            **path,
        }
        for horizon in HORIZONS:
            for basis in ("gross", "net"):
                record[f"treated_{basis}_h{horizon}_pct"] = float(treated_row[f"{basis}_h{horizon}_pct"])
                record[f"treated_{basis}_pessimistic_h{horizon}_pct"] = float(
                    treated_row[f"{basis}_pessimistic_h{horizon}_pct"]
                )
                record[f"treated_{basis}_optimistic_h{horizon}_pct"] = float(
                    treated_row[f"{basis}_optimistic_h{horizon}_pct"]
                )
        rows.append(record)
    outcome_columns = pd.DataFrame(rows, index=pairs.index)
    return pd.concat([pairs.copy(), outcome_columns], axis=1)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weight_values = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.isfinite(weight_values).all():
        raise RuntimeError("outcome aggregation contains non-finite values")
    return float(np.dot(numeric, weight_values))


def build_trade_att(treated: pd.DataFrame, outcome_pairs: pd.DataFrame) -> pd.DataFrame:
    """聚合为每笔处理交易的冻结加权主 ATT 与等权/中位数敏感性。"""

    treated_by_id = treated.set_index("trade_id", drop=False)
    rows: list[dict[str, Any]] = []
    for trade_id, group in outcome_pairs.groupby("trade_id", sort=True):
        treated_row = treated_by_id.loc[trade_id]
        weights = group["control_weight"]
        record: dict[str, Any] = {
            "trade_id": str(trade_id),
            "symbol": str(treated_row["symbol"]),
            "dec_dt": treated_row["dec_dt"],
            "entry_dt": treated_row["entry_dt"],
            "year": int(treated_row["dec_dt"].year),
            "n_controls": int(len(group)),
            "control_filled_n": int(group["control_filled"].sum()),
            "control_cash_n": int((~group["control_filled"]).sum()),
            "control_weight_ess": float(1 / np.square(weights.to_numpy(float)).sum()),
        }
        for horizon in HORIZONS:
            record[f"control_terminal_h{horizon}_n"] = int(group[f"terminal_h{horizon}"].sum())
            for basis in ("gross", "net"):
                treated_nominal = float(treated_row[f"{basis}_h{horizon}_pct"])
                treated_pessimistic = float(treated_row[f"{basis}_pessimistic_h{horizon}_pct"])
                treated_optimistic = float(treated_row[f"{basis}_optimistic_h{horizon}_pct"])
                nominal = group[f"{basis}_h{horizon}_pct"]
                pessimistic = group[f"{basis}_pessimistic_h{horizon}_pct"]
                optimistic = group[f"{basis}_optimistic_h{horizon}_pct"]
                weighted = _weighted_average(nominal, weights)
                equal_mean = float(nominal.mean())
                median = float(nominal.median())
                record[f"treated_{basis}_h{horizon}_pct"] = treated_nominal
                record[f"control_weighted_{basis}_h{horizon}_pct"] = weighted
                record[f"control_equal_mean_{basis}_h{horizon}_pct"] = equal_mean
                record[f"control_median_{basis}_h{horizon}_pct"] = median
                record[f"att_weighted_{basis}_h{horizon}_pct"] = treated_nominal - weighted
                record[f"att_equal_mean_{basis}_h{horizon}_pct"] = treated_nominal - equal_mean
                record[f"att_median_{basis}_h{horizon}_pct"] = treated_nominal - median
                record[f"att_weighted_{basis}_h{horizon}_lower_pct"] = treated_pessimistic - _weighted_average(
                    optimistic, weights
                )
                record[f"att_weighted_{basis}_h{horizon}_upper_pct"] = treated_optimistic - _weighted_average(
                    pessimistic, weights
                )
        rows.append(record)
    result = pd.DataFrame(rows).sort_values(["dec_dt", "symbol"], kind="mergesort").reset_index(drop=True)
    if len(result) != outcome_pairs["trade_id"].nunique():
        raise AssertionError("trade ATT aggregation changed supported trade count")
    return result


def _normal_one_sided_p(t_stat: float) -> float:
    if not math.isfinite(t_stat):
        return 1.0
    return float(0.5 * math.erfc(t_stat / math.sqrt(2.0)))


def ratio_influence_hac(
    frame: pd.DataFrame,
    value_column: str,
    calendar: Sequence[Any],
    *,
    lag: int,
) -> dict[str, Any]:
    """完整交易日历上的 trade-weight ratio estimator 与固定 lag Newey-West。"""

    valid = frame.dropna(subset=[value_column]).copy()
    if valid.empty:
        raise ValueError("ratio-influence HAC has no observations")
    valid["dec_dt"] = pd.to_datetime(valid["dec_dt"]).dt.normalize()
    sessions = common._normalise_dates(calendar)
    sessions = sessions[(sessions >= valid["dec_dt"].min()) & (sessions <= valid["dec_dt"].max())]
    grouped = valid.groupby("dec_dt", sort=True)[value_column].agg(["sum", "size"])
    numerator = grouped["sum"].reindex(sessions, fill_value=0.0).to_numpy(float)
    denominator = grouped["size"].reindex(sessions, fill_value=0.0).to_numpy(float)
    mean_denominator = float(denominator.mean())
    estimate = float(numerator.sum() / denominator.sum())
    influence = (numerator - estimate * denominator) / mean_denominator
    sample_size = len(influence)
    centered = influence - influence.mean()
    lrv = float(centered @ centered / sample_size)
    effective_lag = min(int(lag), sample_size - 1)
    for offset in range(1, effective_lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / sample_size)
        lrv += 2 * (1 - offset / (effective_lag + 1)) * covariance
    lrv = max(lrv, 0.0)
    standard_error = math.sqrt(lrv / sample_size)
    t_stat = estimate / standard_error if standard_error > 0 else math.copysign(math.inf, estimate)
    return {
        "estimand": "equal-weight treated-trade ATT",
        "mean_att_pct": estimate,
        "standard_error_pct": standard_error,
        "t_stat": t_stat,
        "p_value_one_sided": _normal_one_sided_p(t_stat),
        "ci_95_pct": [estimate - 1.959963984540054 * standard_error, estimate + 1.959963984540054 * standard_error],
        "n_trades": int(len(valid)),
        "n_decision_dates": int(valid["dec_dt"].nunique()),
        "calendar_sessions": int(sample_size),
        "calendar_start": sessions.min().strftime("%Y-%m-%d"),
        "calendar_end": sessions.max().strftime("%Y-%m-%d"),
        "hac_lag": int(effective_lag),
    }


def _stationary_bootstrap_indices(
    n: int,
    n_boot: int,
    expected_block: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / expected_block
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < restart_probability
    restart[:, 0] = True
    result = np.empty((n_boot, n), dtype=np.int32)
    current = starts[:, 0].copy()
    for index in range(n):
        current = np.where(restart[:, index], starts[:, index], (current + 1) % n)
        result[:, index] = current
    return result


def stationary_ratio_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    calendar: Sequence[Any],
    *,
    expected_block: int,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """在完整交易日历上对 ratio estimator 做 Politis-Romano 平稳 bootstrap。"""

    valid = frame.dropna(subset=[value_column]).copy()
    if valid.empty or n_boot <= 0:
        raise ValueError("stationary ratio bootstrap requires observations and positive replications")
    valid["dec_dt"] = pd.to_datetime(valid["dec_dt"]).dt.normalize()
    sessions = common._normalise_dates(calendar)
    sessions = sessions[(sessions >= valid["dec_dt"].min()) & (sessions <= valid["dec_dt"].max())]
    grouped = valid.groupby("dec_dt", sort=True)[value_column].agg(["sum", "size"])
    numerator = grouped["sum"].reindex(sessions, fill_value=0.0).to_numpy(float)
    denominator = grouped["size"].reindex(sessions, fill_value=0.0).to_numpy(float)
    indices = _stationary_bootstrap_indices(len(sessions), n_boot, expected_block, seed)
    boot_numerator = numerator[indices].sum(axis=1)
    boot_denominator = denominator[indices].sum(axis=1)
    estimates = np.divide(
        boot_numerator,
        boot_denominator,
        out=np.full(n_boot, np.nan, dtype=float),
        where=boot_denominator > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < max(100, int(n_boot * 0.99)):
        raise RuntimeError("stationary ratio bootstrap produced too many empty-calendar replicates")
    estimate = float(valid[value_column].mean())
    return {
        "mean_att_pct": estimate,
        "ci_95_pct": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "p_value_one_sided": float((np.count_nonzero(estimates <= 0) + 1) / (len(estimates) + 1)),
        "n_boot_requested": int(n_boot),
        "n_boot_valid": int(len(estimates)),
        "expected_block_sessions": int(expected_block),
        "seed": int(seed),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """对具名的固定三期限单侧 p 值做 Holm step-down 校正。"""

    ordered = sorted((str(key), float(value)) for key, value in p_values.items())
    if any(not math.isfinite(value) or value < 0 or value > 1 for _, value in ordered):
        raise ValueError("Holm adjustment received invalid p-values")
    ranked = sorted(ordered, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ranked)
    for rank, (key, value) in enumerate(ranked):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return {key: adjusted[key] for key, _ in ordered}


def two_way_symbol_cluster(
    pair_outcomes: pd.DataFrame,
    *,
    treated_column: str,
    control_column: str,
) -> dict[str, Any]:
    """处理股票×控制股票双向聚类 sandwich，显式处理股票复用网络。"""

    required = {"trade_id", "treated_symbol", "control_symbol", "control_weight", treated_column, control_column}
    if missing := required - set(pair_outcomes):
        raise ValueError(f"two-way cluster input lacks columns: {sorted(missing)}")
    frame = pair_outcomes.loc[:, sorted(required)].copy()
    frame["edge_difference"] = frame[treated_column].astype(float) - frame[control_column].astype(float)
    trade_att = frame.groupby("trade_id", sort=False).apply(
        lambda group: np.dot(group["edge_difference"], group["control_weight"]), include_groups=False
    )
    estimate = float(trade_att.mean())
    frame["score"] = frame["control_weight"] * (frame["edge_difference"] - estimate)

    def corrected_meat(keys: list[str]) -> tuple[float, int]:
        sums = frame.groupby(keys, sort=False)["score"].sum().to_numpy(float)
        groups = len(sums)
        correction = groups / (groups - 1) if groups > 1 else 1.0
        return float(correction * np.square(sums).sum()), groups

    treated_meat, treated_groups = corrected_meat(["treated_symbol"])
    control_meat, control_groups = corrected_meat(["control_symbol"])
    intersection_meat, intersection_groups = corrected_meat(["treated_symbol", "control_symbol"])
    raw_variance = (treated_meat + control_meat - intersection_meat) / len(trade_att) ** 2
    standard_error = math.sqrt(raw_variance) if raw_variance > 0 else math.nan
    t_stat = estimate / standard_error if math.isfinite(standard_error) and standard_error > 0 else math.nan
    return {
        "estimand": "equal-weight treated-trade ATT",
        "cluster_dimensions": ["treated_symbol", "control_symbol"],
        "mean_att_pct": estimate,
        "standard_error_pct": standard_error,
        "variance_raw": raw_variance,
        "t_stat": t_stat,
        "p_value_one_sided": _normal_one_sided_p(t_stat),
        "ci_95_pct": (
            [estimate - 1.959963984540054 * standard_error, estimate + 1.959963984540054 * standard_error]
            if math.isfinite(standard_error)
            else [None, None]
        ),
        "n_trades": int(len(trade_att)),
        "treated_symbol_clusters": int(treated_groups),
        "control_symbol_clusters": int(control_groups),
        "intersection_clusters": int(intersection_groups),
    }


def _scope_summary(
    trade_att: pd.DataFrame,
    pair_outcomes: pd.DataFrame,
    calendar: Sequence[Any],
    *,
    n_boot: int,
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    hac_p: dict[str, float] = {}
    bootstrap_p: dict[str, float] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        primary_column = f"att_weighted_net_h{horizon}_pct"
        hac = ratio_influence_hac(trade_att, primary_column, calendar, lag=HAC_LAGS[horizon])
        bootstrap = stationary_ratio_bootstrap(
            trade_att,
            primary_column,
            calendar,
            expected_block=BOOTSTRAP_BLOCKS[horizon],
            n_boot=n_boot,
        )
        network = two_way_symbol_cluster(
            pair_outcomes,
            treated_column=f"treated_net_h{horizon}_pct",
            control_column=f"net_h{horizon}_pct",
        )
        horizons[key] = {
            "primary": {
                "mean_att_pct": float(trade_att[primary_column].mean()),
                "median_att_pct": float(trade_att[primary_column].median()),
                "positive_att_pct": float(trade_att[primary_column].gt(0).mean() * 100),
                "ratio_influence_hac": hac,
                "stationary_ratio_bootstrap": bootstrap,
                "treated_control_two_way_cluster": network,
                "terminal_bound_mean_pct": {
                    "lower": float(trade_att[f"att_weighted_net_h{horizon}_lower_pct"].mean()),
                    "upper": float(trade_att[f"att_weighted_net_h{horizon}_upper_pct"].mean()),
                },
            },
            "sensitivities": {
                "equal_control_mean_att_pct": float(trade_att[f"att_equal_mean_net_h{horizon}_pct"].mean()),
                "control_median_att_pct": float(trade_att[f"att_median_net_h{horizon}_pct"].mean()),
                "gross_weighted_att_pct": float(trade_att[f"att_weighted_gross_h{horizon}_pct"].mean()),
            },
        }
        hac_p[key] = float(hac["p_value_one_sided"])
        bootstrap_p[key] = float(bootstrap["p_value_one_sided"])
    hac_holm = holm_adjust(hac_p)
    bootstrap_holm = holm_adjust(bootstrap_p)
    robust_horizons: list[int] = []
    for horizon in HORIZONS:
        key = str(horizon)
        primary = horizons[key]["primary"]
        primary["holm_adjusted_one_sided"] = {
            "hac": hac_holm[key],
            "stationary_bootstrap": bootstrap_holm[key],
        }
        passes = bool(
            primary["mean_att_pct"] > 0
            and hac_holm[key] < ALPHA
            and bootstrap_holm[key] < ALPHA
            and primary["treated_control_two_way_cluster"]["p_value_one_sided"] < ALPHA
        )
        primary["historical_robust_gate"] = passes
        if passes:
            robust_horizons.append(horizon)
    daily = trade_att.groupby("dec_dt", sort=True)[[f"att_weighted_net_h{h}_pct" for h in HORIZONS]].mean()
    return {
        "n_trades": int(len(trade_att)),
        "n_decision_dates": int(trade_att["dec_dt"].nunique()),
        "equal_decision_date_mean_pct": {
            str(horizon): float(daily[f"att_weighted_net_h{horizon}_pct"].mean()) for horizon in HORIZONS
        },
        "holm_family": "one-sided H5/H20/H60; family size 3",
        "historical_robust_horizons": robust_horizons,
        "horizons": horizons,
    }


def build_inference(
    trade_att: pd.DataFrame,
    pair_outcomes: pd.DataFrame,
    calendar: Sequence[Any],
    *,
    n_boot: int,
) -> dict[str, Any]:
    """全期与 2024+ 使用同一冻结统计协议。"""

    scopes = {
        "all": trade_att.index,
        "2024plus": trade_att.index[trade_att["year"].ge(2024)],
    }
    result: dict[str, Any] = {}
    for name, indices in scopes.items():
        trades = trade_att.loc[indices].copy()
        ids = set(trades["trade_id"])
        pairs = pair_outcomes[pair_outcomes["trade_id"].isin(ids)].copy()
        if trades.empty or pairs.empty:
            raise RuntimeError(f"outcome inference scope is empty: {name}")
        result[name] = _scope_summary(trades, pairs, calendar, n_boot=n_boot)
    return result


def dependence_diagnostics(
    trade_att: pd.DataFrame,
    pair_outcomes: pd.DataFrame,
    calendar: Sequence[Any],
) -> dict[str, Any]:
    """量化重复股票与重叠 60 日路径，不把 HAC 当作完整网络证明。"""

    control_reuse = pair_outcomes.groupby("control_symbol")["trade_id"].nunique()
    treated_reuse = trade_att.groupby("symbol")["trade_id"].nunique()
    weighted_exposure = pair_outcomes.groupby("control_symbol")["control_weight"].sum()
    sessions = common._normalise_dates(calendar)
    active = pd.Series(0, index=sessions, dtype=int)
    schedule = pair_outcomes.drop_duplicates("trade_id", keep="first")
    for row in schedule.itertuples(index=False):
        mask = (sessions >= row.entry_dt) & (sessions <= row.h60_dt)
        active.loc[sessions[mask]] += 1
    return {
        "treated_symbol_reuse": {
            "unique_symbols": int(len(treated_reuse)),
            "symbols_used_more_than_once": int(treated_reuse.gt(1).sum()),
            "max_trades_per_symbol": int(treated_reuse.max()),
        },
        "control_symbol_reuse": {
            "unique_symbols": int(len(control_reuse)),
            "symbols_used_more_than_once": int(control_reuse.gt(1).sum()),
            "max_trades_per_symbol": int(control_reuse.max()),
            "max_total_frozen_weight": float(weighted_exposure.max()),
        },
        "overlapping_h60_paths": {
            "calendar_sessions": int(len(active)),
            "sessions_with_active_trade": int(active.gt(0).sum()),
            "mean_active_on_nonzero_sessions": float(active[active.gt(0)].mean()),
            "p95_active_on_nonzero_sessions": float(active[active.gt(0)].quantile(0.95)),
            "max_active_trades": int(active.max()),
        },
    }


def _output_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    result = {"locator": _repo_locator(path), "sha256": identity_utils.sha256_file(path)}
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def run_audit(*, n_boot: int = N_BOOT, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """验证授权、附加结果、执行冻结推断并写出内容寻址证据。"""

    started = time.time()
    plan = _load_json(PLAN_PATH, "tracked outcome-blind plan")
    balance = _load_json(BALANCE_AUDIT_PATH, "exact-mcap balance audit")
    input_identity = validate_outcome_authorization(plan, balance)
    pairs = pd.read_parquet(FROZEN_PAIRS_PATH)
    validate_frozen_pairs(pairs, input_identity["frozen_pairs"]["rows"])

    market = pd.read_parquet(MARKET_PATH, columns=["dt"])
    calendar = common._normalise_dates(market["dt"])
    treated, all_filled, funnel, production_audit = common.load_production_common_support(calendar)
    rebuilt = balance_audit.validate_rebuilt_production_identities(treated, all_filled, plan)
    if rebuilt["common60_schedule_identity"] != plan["treated_common_support"]["schedule_identity"]:
        raise RuntimeError("outcome-stage treated schedule drift")
    panel = pd.read_parquet(PANEL_PATH)
    open_w, close_w, _, mark_w, last_dt = common._wide_panel(panel, calendar)
    treated = common.add_treated_outcomes(
        treated,
        open_w=open_w,
        close_w=close_w,
        mark_w=mark_w,
        last_dt=last_dt,
    )
    supported_ids = set(pairs["trade_id"])
    treated = treated[treated["trade_id"].isin(supported_ids)].copy()
    if len(treated) != pairs["trade_id"].nunique():
        raise RuntimeError("supported treated identity mismatch")
    pair_outcomes = add_control_outcomes(
        pairs,
        treated,
        open_w=open_w,
        close_w=close_w,
        mark_w=mark_w,
        last_dt=last_dt,
    )
    trade_att = build_trade_att(treated, pair_outcomes)
    inference = build_inference(trade_att, pair_outcomes, calendar, n_boot=n_boot)
    dependence = dependence_diagnostics(trade_att, pair_outcomes, calendar)

    output_dir.mkdir(parents=True, exist_ok=True)
    treated_path = output_dir / TREATED_PATH.name
    pair_path = output_dir / PAIR_OUTCOMES_PATH.name
    trade_path = output_dir / TRADE_ATT_PATH.name
    audit_path = output_dir / AUDIT_PATH.name
    treated.to_parquet(treated_path, index=False)
    pair_outcomes.to_parquet(pair_path, index=False)
    trade_att.to_parquet(trade_path, index=False)
    primary_horizons = inference[PRIMARY_SCOPE]["historical_robust_horizons"]
    verdict = {
        "status": (
            "HISTORICAL_EXACT_PIT_EDGE_ROBUST_FORWARD_VALIDATION_REQUIRED"
            if primary_horizons
            else "HISTORICAL_EXACT_PIT_EDGE_NOT_ROBUSTLY_CONFIRMED"
        ),
        "primary_scope": PRIMARY_SCOPE,
        "historical_robust_horizons": primary_horizons,
        "forward_validation_required": True,
        "live_authorized": False,
    }
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "design": {
            "outcome_unlocked_by_frozen_balance_gate": True,
            "horizons_sessions": list(HORIZONS),
            "primary_estimand": "equal-weight treated-trade ATT",
            "control_aggregation": "frozen within-trade conditional-entropy weights",
            "sensitivities": ["equal control mean", "control median", "gross weighted ATT"],
            "path_policy": plan["design"]["outcome_design"]["path_policy"],
            "costs": plan["design"]["outcome_design"]["costs"],
            "hac_lags": {str(key): value for key, value in HAC_LAGS.items()},
            "stationary_bootstrap": {
                "expected_blocks": {str(key): value for key, value in BOOTSTRAP_BLOCKS.items()},
                "replications": int(n_boot),
                "seed": BOOTSTRAP_SEED,
            },
            "multiple_testing": "Holm-adjusted one-sided H5/H20/H60 family",
            "network_inference": "treated_symbol × control_symbol two-way cluster sandwich",
            "historical_robust_gate": (
                "positive ATT and HAC-Holm<0.05 and stationary-bootstrap-Holm<0.05 and two-way-cluster p<0.05"
            ),
            "parameter_search": "none after balance authorization",
        },
        "input_identity": {
            **input_identity,
            "outcome_script": {
                "locator": _repo_locator(Path(__file__)),
                "sha256": identity_utils.sha256_file(Path(__file__)),
            },
            "production_audit_schema": production_audit.get("schema"),
        },
        "funnel": funnel,
        "support": {
            "source_mature_trades": int(plan["treated_common_support"]["rows"]),
            "supported_trades": int(len(trade_att)),
            "supported_trades_2024plus": int(trade_att["year"].ge(2024).sum()),
            "frozen_pairs": int(len(pair_outcomes)),
            "control_cash_pairs": int((~pair_outcomes["control_filled"]).sum()),
            "control_terminal_h60_pairs": int(pair_outcomes["terminal_h60"].sum()),
        },
        "inference": inference,
        "dependence_diagnostics": dependence,
        "outputs": {
            "treated_outcomes": _output_record(treated_path, len(treated)),
            "control_outcome_pairs": _output_record(pair_path, len(pair_outcomes)),
            "trade_att": _output_record(trade_path, len(trade_att)),
        },
        "limitations": [
            "Historical significance cannot authorize live use; an untouched forward sample is still required.",
            "Two-way symbol clustering addresses repeated treated/control names but not every overlapping-path mechanism.",
            "Terminal settlement remains bounded by pessimistic -100% versus optimistic LOCF conventions.",
        ],
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    clean = _json_clean(audit)
    audit_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    audit = run_audit(n_boot=args.n_boot)
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
