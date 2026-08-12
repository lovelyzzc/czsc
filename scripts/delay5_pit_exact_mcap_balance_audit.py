"""Delay5 点时精确流通市值的结果盲匹配与余额门。

本模块只读取 production cohort 的身份/成交阶段、决策日及以前的行情、冻结年度
行业、历史 ST 和点时 ``circ_mv``。它不加载处理票或控制票的未来收益；匹配对、
尝试表和余额报告落盘并哈希后，才给出是否允许进入独立 outcome 阶段的判定。

运行：``uv run --no-sync python scripts/delay5_pit_exact_mcap_balance_audit.py``。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import delay5_pit_exact_mcap_collector as collector
import delay5_pit_exact_mcap_plan as pit_plan
import numpy as np
import pandas as pd
import surge_portfolio_backtest as portfolio
import trend_regime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REQUEST_PATH = SCRIPT_DIR / "_output" / "delay5_common_horizon_att" / "exact_mcap_request_manifest.json"
PLAN_PATH = SCRIPT_DIR / "delay5_pit_exact_mcap_plan_2026-08-11.json"
PLAN_SCRIPT_PATH = SCRIPT_DIR / "delay5_pit_exact_mcap_plan.py"
COLLECTOR_SCRIPT_PATH = SCRIPT_DIR / "delay5_pit_exact_mcap_collector.py"
BALANCE_SCRIPT_PATH = Path(__file__).resolve()
PRODUCTION_COHORT_PATH = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort" / "cohort.parquet"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
PANEL_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"
PROXY_PAIRS_PATH = SCRIPT_DIR / "_output" / "delay5_common_horizon_att" / "matched_pairs.parquet"
BAOSTOCK_EXACT_PATH = SCRIPT_DIR / "_output" / "s2b_top_winner_exact_mcap" / "baostock_exact_mcap.parquet"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "delay5_pit_exact_mcap_balance_audit"
PAIRS_PATH = OUTPUT_DIR / "matched_pairs.parquet"
ATTEMPTS_PATH = OUTPUT_DIR / "match_attempts.parquet"
BALANCE_PATH = OUTPUT_DIR / "balance.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"

SCHEMA = "delay5_pit_exact_mcap_balance_audit_v1"
EXPECTED_REQUEST_SCHEMA = "delay5_exact_mcap_request_manifest_v2"
EXPECTED_PLAN_SCHEMA = "delay5_pit_exact_mcap_plan_manifest_v1"
SPECIFICATION = "pit_exact_industry_mcap_k10_min5_c1p5"
K = 10
MIN_CONTROLS = 5
CALIPER_RATIO = 1.5
COMMON_HORIZON = 60
BALANCE_THRESHOLD = 0.10
MCAP_BALANCE_THRESHOLD = 0.05
COVERAGE_THRESHOLD = 0.80
BALANCE_SCOPES = ("all", "2024plus")

# 这些字段均可由 decision-day panel 的因果前缀可靠构造。若计划预注册更多字段，
# ``resolve_balance_features`` 会保留并把不可构造字段列入 missing，而不是静默删除。
AVAILABLE_BALANCE_FEATURES = (
    "log_exact_mcap",
    "ret5",
    "ret20",
    "ret60",
    "vol20",
    "vol60",
    "log_price",
    "log_amount",
    "liq20",
    "limitup_count20",
    "limitup_maxrun20",
)
DEFAULT_BALANCE_FEATURES = AVAILABLE_BALANCE_FEATURES
FORBIDDEN_FUTURE_TOKENS = ("gross_", "net_", "ret_gross", "exit_", "h5_", "h20_", "h60_")
EXPECTED_PRIMARY_PLAN = {
    "name": SPECIFICATION,
    "industry": "same frozen annual industry; missing treated industry is unsupported",
    "metric": "nearest absolute log(exact point-in-time circ_mv CNY) distance",
    "caliper_ratio": CALIPER_RATIO,
    "k": K,
    "min_controls": MIN_CONTROLS,
    "tie_break": "match_distance, symbol",
}
CACHE_LOCATOR_PREFIX = "delay5-exact-mcap-cache://"


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """返回排序、紧凑、UTF-8 JSON 的稳定哈希。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def repo_locator(path: Path) -> str:
    """把仓库内路径写成跨设备可用的相对 locator。"""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository: {path}") from exc


def cache_locator(path: Path, cache_root: Path) -> str:
    """把 cache artifact 写成不泄露本机绝对目录的 locator。"""

    try:
        relative = path.resolve().relative_to(cache_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"cache artifact is outside the verified cache root: {path}") from exc
    return f"{CACHE_LOCATOR_PREFIX}{relative.as_posix()}"


def resolve_cache_artifact(cache_root: Path, relative_path: Any, label: str) -> Path:
    """安全解析 collector manifest 中的 portable relative path。"""

    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise RuntimeError(f"{label} lacks a portable relative path")
    parts = Path(relative_path).parts
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"{label} contains an unsafe relative path")
    path = cache_root.joinpath(*parts)
    try:
        path.resolve().relative_to(cache_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the verified cache root") from exc
    return path


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """对指定列的排序记录做与文件格式无关的身份哈希。"""

    scoped = frame.loc[:, list(columns)].copy()
    for column in scoped:
        if pd.api.types.is_datetime64_any_dtype(scoped[column]):
            scoped[column] = scoped[column].dt.strftime("%Y-%m-%d")
    records = scoped.sort_values(list(columns), kind="mergesort").to_dict("records")
    return {"rows": int(len(records)), "sha256": canonical_sha256(records)}


def assert_outcome_blind_columns(frame: pd.DataFrame, label: str) -> None:
    """禁止结果路径字段进入匹配中间产物。"""

    forbidden = [column for column in frame if any(token in column.lower() for token in FORBIDDEN_FUTURE_TOKENS)]
    if forbidden:
        raise RuntimeError(f"{label} contains future outcome columns: {sorted(forbidden)}")


def normalise_calendar(values: Sequence[Any]) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(list(values))).normalize().sort_values().unique()
    if len(calendar) == 0:
        raise ValueError("calendar is empty")
    return pd.DatetimeIndex(calendar)


def build_common60_identities(cohort: pd.DataFrame, calendar: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只从 ``stage_fill`` 身份和日历重建共同 H60 处理集及同日排除集。"""

    required = {"symbol", "dec_dt", "entry_dt", "stage_fill"}
    if missing := required - set(cohort):
        raise ValueError(f"production cohort lacks identity columns: {sorted(missing)}")
    scoped = cohort.loc[:, sorted(required)].copy()
    scoped["symbol"] = scoped["symbol"].astype(str)
    for column in ("dec_dt", "entry_dt"):
        scoped[column] = pd.to_datetime(scoped[column]).dt.normalize()
    if scoped.duplicated(["symbol", "dec_dt"]).any():
        raise RuntimeError("production cohort is not unique by symbol/dec_dt")
    sessions = normalise_calendar(calendar)
    next_session = pd.Series(sessions[1:], index=sessions[:-1])
    filled = scoped[scoped["stage_fill"].astype(bool)].copy()
    filled = filled[filled["entry_dt"].eq(filled["dec_dt"].map(next_session))].copy()
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    filled["entry_position"] = filled["entry_dt"].map(position)
    treated = filled[
        filled["entry_position"].notna() & filled["entry_position"].le(len(sessions) - COMMON_HORIZON)
    ].copy()
    for horizon in (5, 20, 60):
        treated[f"h{horizon}_dt"] = [sessions[int(position) + horizon - 1] for position in treated["entry_position"]]
    treated["trade_id"] = treated["symbol"] + "|" + treated["dec_dt"].dt.strftime("%Y-%m-%d")
    treated["year"] = treated["dec_dt"].dt.year.astype(int)
    treated = treated.sort_values(["dec_dt", "symbol"], kind="mergesort").reset_index(drop=True)
    return treated, filled.sort_values(["dec_dt", "symbol"], kind="mergesort").reset_index(drop=True)


def nearest_exact_mcap_controls(
    pool: pd.DataFrame,
    *,
    treated_mcap: float,
    k: int = K,
    caliper_ratio: float = CALIPER_RATIO,
) -> tuple[pd.DataFrame, int]:
    """在 inclusive 规模卡尺内按 ``|log ratio|, symbol`` 确定性匹配。"""

    if k <= 0:
        raise ValueError("k must be positive")
    if caliper_ratio <= 1:
        raise ValueError("caliper_ratio must be greater than one")
    if not np.isfinite(treated_mcap) or treated_mcap <= 0:
        return pool.iloc[:0].assign(match_distance=pd.Series(dtype=float)), 0
    required = {"symbol", "exact_circ_mv_cny"}
    if missing := required - set(pool):
        raise ValueError(f"exact-mcap pool lacks columns: {sorted(missing)}")
    values = pd.to_numeric(pool["exact_circ_mv_cny"], errors="coerce")
    eligible = pool[
        np.isfinite(values)
        & values.gt(0)
        & values.ge(treated_mcap / caliper_ratio)
        & values.le(treated_mcap * caliper_ratio)
    ].copy()
    eligible["match_distance"] = (
        np.log(eligible["exact_circ_mv_cny"].astype(float)) - math.log(float(treated_mcap))
    ).abs()
    eligible = eligible.sort_values(["match_distance", "symbol"], kind="mergesort")
    return eligible.head(k).copy(), int(len(eligible))


@dataclass
class CausalFeatureStore:
    """只在每个决策日的 ``panel[dt <= dec_dt]`` 前缀上计算特征。"""

    panel: pd.DataFrame
    calendar: pd.DatetimeIndex
    raw_close: pd.DataFrame
    amount: pd.DataFrame
    mark_close: pd.DataFrame
    limitup: pd.DataFrame
    _cache: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_panel(
        cls,
        panel: pd.DataFrame,
        calendar: Sequence[Any],
        st_intervals: Mapping[str, Sequence[tuple[str, str, bool]]] | None = None,
    ) -> CausalFeatureStore:
        required = {"symbol", "dt", "close", "amount_e"}
        if missing := required - set(panel):
            raise ValueError(f"panel lacks causal feature columns: {sorted(missing)}")
        scoped = panel.loc[:, ["symbol", "dt", "close", "amount_e"]].copy()
        scoped["symbol"] = scoped["symbol"].astype(str)
        scoped["dt"] = pd.to_datetime(scoped["dt"]).dt.normalize()
        if scoped.duplicated(["dt", "symbol"]).any():
            raise RuntimeError("panel contains duplicate symbol/date rows")
        sessions = normalise_calendar(calendar)
        raw_close = scoped.pivot(index="dt", columns="symbol", values="close").reindex(sessions)
        amount = scoped.pivot(index="dt", columns="symbol", values="amount_e").reindex(sessions)
        mark_close = raw_close.ffill()
        limitup = build_limitup_matrix(raw_close, mark_close, st_intervals or {})
        return cls(scoped, sessions, raw_close, amount, mark_close, limitup)

    def snapshot(self, dec_dt: Any) -> pd.DataFrame:
        decision = pd.Timestamp(dec_dt).normalize()
        if decision not in self._cache:
            # The explicit upper bound is a causal ratchet: future rows never enter a
            # fill operation, rolling statistic, candidate pool, or feature value.
            self._cache[decision] = decision_feature_snapshot(
                self.raw_close.loc[:decision],
                self.amount.loc[:decision],
                self.mark_close.loc[:decision],
                self.limitup.loc[:decision],
                decision,
            )
        return self._cache[decision]


def build_limitup_matrix(
    raw_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    st_intervals: Mapping[str, Sequence[tuple[str, str, bool]]],
) -> pd.DataFrame:
    """构造只依赖当日实际收盘和前一日 LOCF 收盘的历史涨停代理。"""

    if not raw_close.index.equals(mark_close.index) or not raw_close.columns.equals(mark_close.columns):
        raise ValueError("raw_close and mark_close axes differ")
    thresholds = pd.Series(
        {str(symbol): trend_regime.limit_pct_for(str(symbol)) for symbol in raw_close.columns}, dtype=float
    ).reindex(raw_close.columns)
    previous = mark_close.shift(1)
    returns_pct = (raw_close / previous - 1) * 100
    observed = raw_close.notna() & raw_close.gt(0) & previous.notna() & previous.gt(0)
    result = observed & returns_pct.ge(thresholds, axis="columns")
    date_keys = np.asarray(raw_close.index.strftime("%Y%m%d"))
    for symbol, rows in st_intervals.items():
        if symbol not in result.columns or not rows:
            continue
        ordered = sorted(rows, key=lambda item: str(item[0]))
        starts = np.asarray([str(item[0]) for item in ordered])
        flags = np.asarray([bool(item[2]) for item in ordered])
        positions = np.searchsorted(starts, date_keys, side="right") - 1
        has_state = positions >= 0
        is_st = np.zeros(len(date_keys), dtype=bool)
        is_st[has_state] = flags[positions[has_state]]
        st_hits = observed[symbol].to_numpy(bool) & returns_pct[symbol].ge(4.8).to_numpy(bool)
        values = result[symbol].to_numpy(bool)
        values[is_st] = st_hits[is_st]
        result[symbol] = values
    return result.astype(bool)


def longest_true_run(window: np.ndarray) -> np.ndarray:
    """按列返回布尔窗口的最长连续 True。"""

    values = np.asarray(window, dtype=bool)
    if values.ndim != 2:
        raise ValueError("limit-up window must be two-dimensional")
    running = np.zeros(values.shape[1], dtype=int)
    longest = np.zeros(values.shape[1], dtype=int)
    for row in values:
        running = np.where(row, running + 1, 0)
        longest = np.maximum(longest, running)
    return longest


def decision_feature_snapshot(
    raw_close: pd.DataFrame,
    amount: pd.DataFrame,
    mark_close: pd.DataFrame,
    limitup: pd.DataFrame,
    dec_dt: Any,
) -> pd.DataFrame:
    """从已截断行情前缀计算决策日可见的价格、流动性和路径特征。"""

    decision = pd.Timestamp(dec_dt).normalize()
    matrices = (raw_close, amount, mark_close, limitup)
    if any(matrix.index[-1] != decision for matrix in matrices):
        raise ValueError("decision date is absent from causal calendar prefix")
    if not all(raw_close.index.equals(matrix.index) for matrix in matrices[1:]) or not all(
        raw_close.columns.equals(matrix.columns) for matrix in matrices[1:]
    ):
        raise ValueError("causal feature matrix axes differ")
    current_mark = mark_close.iloc[-1]
    actual_close = raw_close.iloc[-1]
    actual_amount = amount.iloc[-1]
    result = pd.DataFrame(index=raw_close.columns)
    result.index.name = "symbol"
    for horizon in (5, 20, 60):
        result[f"ret{horizon}"] = (
            (current_mark / mark_close.iloc[-1 - horizon] - 1) * 100 if len(mark_close) > horizon else np.nan
        )
    for horizon in (20, 60):
        if len(mark_close) > horizon:
            daily = mark_close.iloc[-1 - horizon :].pct_change(fill_method=None).iloc[1:]
            result[f"vol{horizon}"] = daily.std(axis=0, ddof=1) * math.sqrt(252) * 100
        else:
            result[f"vol{horizon}"] = np.nan
    result["log_price"] = np.log(actual_close.where(actual_close.gt(0)))
    result["log_amount"] = np.log(actual_amount.where(actual_amount.gt(0)))
    trailing_amount = amount.iloc[-20:] if len(amount) >= 20 else amount.iloc[:0]
    result["liq20"] = np.log(trailing_amount.mean(axis=0).where(lambda value: value.gt(0)))
    if len(limitup) >= 20:
        window = limitup.iloc[-20:].to_numpy(dtype=bool)
        result["limitup_count20"] = window.sum(axis=0).astype(int)
        result["limitup_maxrun20"] = longest_true_run(window)
    else:
        result["limitup_count20"] = np.nan
        result["limitup_maxrun20"] = np.nan
    result["feature_max_dt"] = decision
    return result


def build_exact_matches(
    treated: pd.DataFrame,
    all_filled: pd.DataFrame,
    *,
    feature_store: CausalFeatureStore,
    exact_mcap: pd.DataFrame,
    industry_maps: Mapping[int, pd.Series],
    st_intervals: Mapping[str, Any],
    balance_features: Sequence[str] = DEFAULT_BALANCE_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造单一预注册 exact-mcap 匹配；不接触任何未来结果列。"""

    required_treated = {"trade_id", "symbol", "dec_dt", "year"}
    if missing := required_treated - set(treated):
        raise ValueError(f"treated identities lack columns: {sorted(missing)}")
    required_mcap = {"symbol", "dec_dt", "exact_circ_mv_cny"}
    if missing := required_mcap - set(exact_mcap):
        raise ValueError(f"materialized exact mcap lacks columns: {sorted(missing)}")
    mcap = exact_mcap.loc[:, sorted(required_mcap)].copy()
    mcap["symbol"] = mcap["symbol"].astype(str)
    mcap["dec_dt"] = pd.to_datetime(mcap["dec_dt"]).dt.normalize()
    if mcap.duplicated(["symbol", "dec_dt"]).any():
        raise RuntimeError("materialized exact mcap has duplicate keys")
    treated_by_date = {
        pd.Timestamp(date): set(group["symbol"].astype(str)) for date, group in all_filled.groupby("dec_dt", sort=True)
    }
    pair_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for dec_dt, day_treated in treated.groupby("dec_dt", sort=True):
        decision = pd.Timestamp(dec_dt).normalize()
        year = int(decision.year)
        snapshot = feature_store.snapshot(decision).copy()
        snapshot.index.name = None
        snapshot = snapshot.join(mcap[mcap["dec_dt"].eq(decision)].set_index("symbol")[["exact_circ_mv_cny"]])
        snapshot["symbol"] = snapshot.index.astype(str)
        snapshot["log_exact_mcap"] = np.log(snapshot["exact_circ_mv_cny"].where(snapshot["exact_circ_mv_cny"].gt(0)))
        industry = industry_maps.get(year)
        if industry is None:
            raise RuntimeError(f"frozen industry map is unavailable for {year}")
        snapshot["industry"] = snapshot.index.to_series().map(industry)
        snapshot["is_st"] = [portfolio.is_st_on(st_intervals, symbol, decision) for symbol in snapshot.index]
        current_valid = np.isfinite(pd.to_numeric(snapshot["log_price"], errors="coerce")) & np.isfinite(
            pd.to_numeric(snapshot["log_amount"], errors="coerce")
        )
        excluded = treated_by_date.get(decision, set())
        base_pool = snapshot[current_valid & ~snapshot.index.isin(excluded) & ~snapshot["is_st"].astype(bool)].copy()
        for row in day_treated.itertuples(index=False):
            reason: str | None = None
            treated_decision_visible = bool(row.symbol in snapshot.index and current_valid.get(row.symbol, False))
            target_industry: Any = industry.get(row.symbol, np.nan)
            if not treated_decision_visible:
                reason = "treated_not_decision_visible"
            elif target_industry is None or pd.isna(target_industry) or not str(target_industry).strip():
                reason = "missing_treated_industry"
            target_mcap = (
                float(snapshot.at[row.symbol, "exact_circ_mv_cny"]) if row.symbol in snapshot.index else np.nan
            )
            if reason is None and (not np.isfinite(target_mcap) or target_mcap <= 0):
                reason = "missing_treated_exact_mcap"
            attempt_eligible = reason is None
            industry_pool = base_pool.iloc[:0]
            matched = base_pool.iloc[:0].assign(match_distance=pd.Series(dtype=float))
            eligible_n = 0
            if reason is None:
                industry_pool = base_pool[base_pool["industry"].eq(target_industry)].copy()
                matched, eligible_n = nearest_exact_mcap_controls(industry_pool, treated_mcap=target_mcap)
                if len(matched) < MIN_CONTROLS:
                    reason = "fewer_than_min_controls"
            support_ok = reason is None
            attempt_rows.append(
                {
                    "trade_id": row.trade_id,
                    "specification": SPECIFICATION,
                    "symbol": row.symbol,
                    "dec_dt": decision,
                    "year": year,
                    "treated_industry": None if pd.isna(target_industry) else str(target_industry),
                    "treated_decision_visible": treated_decision_visible,
                    "treated_exact_mcap_available": bool(np.isfinite(target_mcap) and target_mcap > 0),
                    "attempt_eligible": bool(attempt_eligible),
                    "industry_pool_n": int(len(industry_pool)),
                    "caliper_eligible_pool_n": int(eligible_n),
                    "selected_controls_n": int(len(matched)),
                    "support_ok": bool(support_ok),
                    "unsupported_reason": reason,
                }
            )
            for rank, (control_symbol, control) in enumerate(matched.iterrows(), start=1):
                record: dict[str, Any] = {
                    "trade_id": row.trade_id,
                    "specification": SPECIFICATION,
                    "support_ok": bool(support_ok),
                    "treated_symbol": row.symbol,
                    "control_symbol": str(control_symbol),
                    "dec_dt": decision,
                    "year": year,
                    "rank": rank,
                    "eligible_pool_n": int(eligible_n),
                    "selected_controls_n": int(len(matched)),
                    "treated_industry": str(target_industry),
                    "treated_exact_circ_mv_cny": target_mcap,
                    "control_exact_circ_mv_cny": float(control["exact_circ_mv_cny"]),
                    "mcap_ratio": float(control["exact_circ_mv_cny"]) / target_mcap,
                    "match_distance": float(control["match_distance"]),
                }
                target = snapshot.loc[row.symbol]
                for feature in balance_features:
                    record[f"treated_{feature}"] = float(target.get(feature, np.nan))
                    record[f"control_{feature}"] = float(control.get(feature, np.nan))
                pair_rows.append(record)
    attempts = pd.DataFrame(attempt_rows).sort_values(["dec_dt", "symbol"], kind="mergesort").reset_index(drop=True)
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["dec_dt", "treated_symbol", "rank"], kind="mergesort").reset_index(drop=True)
    assert_outcome_blind_columns(attempts, "attempts")
    assert_outcome_blind_columns(pairs, "pairs")
    return pairs, attempts


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or len(values) != len(weights) or not np.isfinite(values).all():
        raise ValueError("weighted moments received invalid values")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weighted moments received invalid weights")
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return mean, variance


def balance_statistics(pairs: pd.DataFrame, features: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """treated=1、每个 control=1/K，任何缺值都 fail closed。"""

    if pairs.empty:
        return {}, list(features)
    treated = pairs.drop_duplicates("trade_id", keep="first")
    weights = 1.0 / pairs.groupby("trade_id")["trade_id"].transform("size").to_numpy(float)
    result: dict[str, Any] = {}
    missing: list[str] = []
    for feature in features:
        treated_column = f"treated_{feature}"
        control_column = f"control_{feature}"
        if treated_column not in pairs or control_column not in pairs:
            missing.append(str(feature))
            continue
        treated_values = pd.to_numeric(treated[treated_column], errors="coerce").to_numpy(float)
        control_values = pd.to_numeric(pairs[control_column], errors="coerce").to_numpy(float)
        if not np.isfinite(treated_values).all() or not np.isfinite(control_values).all():
            missing.append(str(feature))
            continue
        treated_mean, treated_var = _weighted_moments(treated_values, np.ones(len(treated_values)))
        control_mean, control_var = _weighted_moments(control_values, weights)
        pooled = math.sqrt((treated_var + control_var) / 2)
        smd = (
            0.0
            if pooled == 0 and treated_mean == control_mean
            else (
                math.copysign(math.inf, treated_mean - control_mean)
                if pooled == 0
                else (treated_mean - control_mean) / pooled
            )
        )
        threshold = MCAP_BALANCE_THRESHOLD if feature == "log_exact_mcap" else BALANCE_THRESHOLD
        result[str(feature)] = {
            "treated_mean": treated_mean,
            "control_mean": control_mean,
            "smd": float(smd),
            "threshold_abs_smd": threshold,
            "passes": bool(np.isfinite(smd) and abs(smd) < threshold),
        }
    return result, missing


def scoped_balance(pairs: pd.DataFrame, features: Sequence[str]) -> dict[str, Any]:
    supported = pairs[pairs["support_ok"].astype(bool)].copy()
    scopes = {"all": supported, "2024plus": supported[supported["year"].ge(2024)]}
    return {
        name: {
            "supported_trades": int(scope["trade_id"].nunique()),
            "statistics": balance_statistics(scope, features)[0],
            "missing_features": balance_statistics(scope, features)[1],
        }
        for name, scope in scopes.items()
    }


def coverage_summary(attempts: pd.DataFrame) -> dict[str, Any]:
    """报告 D_source/D_attempt、全期/2024+/逐年支持。"""

    def summarize(scope: pd.DataFrame) -> dict[str, Any]:
        source = int(len(scope))
        collector_available = int(scope["treated_exact_mcap_available"].sum())
        attempted = int(scope["attempt_eligible"].sum())
        supported = int(scope["support_ok"].sum())
        return {
            "D_source": source,
            "D_attempt": attempted,
            "D_supported": supported,
            "collector_available": collector_available,
            "collector_coverage": float(collector_available / source) if source else 0.0,
            "attempt_coverage": float(attempted / source) if source else 0.0,
            "support_rate": float(supported / source) if source else 0.0,
            "source_support_rate": float(supported / source) if source else 0.0,
            "attempt_support_rate": float(supported / attempted) if attempted else 0.0,
        }

    return {
        "all": summarize(attempts),
        "2024plus": summarize(attempts[attempts["year"].ge(2024)]),
        "by_year": {str(int(year)): summarize(group) for year, group in attempts.groupby("year", sort=True)},
    }


def proxy_identical_support_crosstab(attempts: pd.DataFrame, proxy_pairs: pd.DataFrame | None) -> dict[str, Any]:
    """仅比较 trade identity/support；不读取 proxy 的结果列。"""

    if proxy_pairs is None or proxy_pairs.empty:
        return {"available": False}
    required = {"trade_id", "support_ok"}
    if missing := required - set(proxy_pairs):
        raise ValueError(f"proxy identity/support lacks columns: {sorted(missing)}")
    proxy = proxy_pairs.loc[:, ["trade_id", "support_ok"]].groupby("trade_id", as_index=False)["support_ok"].max()
    exact = attempts.loc[:, ["trade_id", "support_ok"]].rename(columns={"support_ok": "exact_support"})
    merged = exact.merge(proxy.rename(columns={"support_ok": "proxy_support"}), on="trade_id", how="left")
    table = pd.crosstab(
        merged["exact_support"].astype(bool),
        merged["proxy_support"].eq(True),
        dropna=False,
    )
    return {
        "available": True,
        "identical_trade_ids": int(merged["proxy_support"].notna().sum()),
        "crosstab": {
            f"exact_{str(bool(exact_value)).lower()}__proxy_{str(bool(proxy_value)).lower()}": int(value)
            for exact_value, row in table.iterrows()
            for proxy_value, value in row.items()
        },
    }


def cross_provider_unit_qa(exact_mcap: pd.DataFrame, baostock: pd.DataFrame | None) -> dict[str, Any]:
    """用已有 BaoStock 重叠点核对单位和方向；诊断结果绝不参与匹配。"""

    if baostock is None or baostock.empty:
        return {"available": False}
    exact_required = {"symbol", "dec_dt", "exact_circ_mv_cny"}
    baostock_required = {"symbol", "dec_dt", "exact_circ_mv"}
    if missing := exact_required - set(exact_mcap):
        raise ValueError(f"exact-mcap QA input lacks columns: {sorted(missing)}")
    if missing := baostock_required - set(baostock):
        raise ValueError(f"BaoStock QA input lacks columns: {sorted(missing)}")
    left = exact_mcap.loc[:, sorted(exact_required)].copy()
    right = baostock.loc[:, sorted(baostock_required)].copy()
    for frame in (left, right):
        frame["dec_dt"] = pd.to_datetime(frame["dec_dt"]).dt.normalize()
    merged = left.merge(right, on=["symbol", "dec_dt"], how="inner", validate="one_to_one")
    valid = (
        np.isfinite(pd.to_numeric(merged["exact_circ_mv_cny"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["exact_circ_mv"], errors="coerce"))
        & merged["exact_circ_mv_cny"].gt(0)
        & merged["exact_circ_mv"].gt(0)
    )
    merged = merged[valid].copy()
    if merged.empty:
        return {"available": False, "overlap_rows": 0}
    merged["ratio"] = merged["exact_circ_mv_cny"] / merged["exact_circ_mv"]
    merged["absolute_deviation_pct"] = (merged["ratio"] - 1).abs() * 100
    records = [
        [
            str(row.symbol),
            pd.Timestamp(row.dec_dt).strftime("%Y-%m-%d"),
            round(float(row.ratio), 12),
            round(float(row.absolute_deviation_pct), 9),
        ]
        for row in merged.sort_values(["symbol", "dec_dt"], kind="mergesort").itertuples(index=False)
    ]
    return {
        "available": True,
        "role": "unit/direction QA only; never used for matching or fallback",
        "overlap_rows": int(len(merged)),
        "overlap_dates": int(merged["dec_dt"].nunique()),
        "overlap_symbols": int(merged["symbol"].nunique()),
        "ratio_median": float(merged["ratio"].median()),
        "ratio_min": float(merged["ratio"].min()),
        "ratio_max": float(merged["ratio"].max()),
        "absolute_deviation_pct": {
            "median": float(merged["absolute_deviation_pct"].median()),
            "p95": float(merged["absolute_deviation_pct"].quantile(0.95)),
            "max": float(merged["absolute_deviation_pct"].max()),
            "gt_5pct": int(merged["absolute_deviation_pct"].gt(5).sum()),
        },
        "canonical_records_sha256": canonical_sha256(records),
    }


def build_verdict(
    coverage: Mapping[str, Mapping[str, Any]],
    balance: Mapping[str, Mapping[str, Any]],
    *,
    required_features: Sequence[str],
    input_complete: bool = True,
) -> dict[str, Any]:
    """余额/coverage 未过则显式阻断 outcome 阶段；实盘永不授权。"""

    coverage_ok = input_complete and all(
        float(coverage[scope]["collector_coverage"]) >= COVERAGE_THRESHOLD
        and float(coverage[scope]["D_supported"]) / max(float(coverage[scope]["D_source"]), 1.0) >= COVERAGE_THRESHOLD
        for scope in BALANCE_SCOPES
    )
    failed: dict[str, list[str]] = {}
    for scope in BALANCE_SCOPES:
        scope_balance = balance.get(scope, {})
        stats = scope_balance.get("statistics", {})
        missing = list(scope_balance.get("missing_features", []))
        failed[scope] = sorted(
            set(missing)
            | {
                feature
                for feature in required_features
                if feature not in stats or not bool(stats[feature].get("passes", False))
            }
        )
    balance_ok = not any(failed.values())
    if not input_complete:
        status = "INPUT_INCOMPLETE_OUTCOMES_NOT_EVALUATED"
    elif not coverage_ok:
        status = "COLLECTOR_COVERAGE_INSUFFICIENT_OUTCOMES_NOT_EVALUATED"
    elif not balance_ok:
        status = "BALANCE_INSUFFICIENT_OUTCOMES_NOT_EVALUATED"
    else:
        status = "BALANCE_GATE_PASSED_OUTCOME_EVALUATION_PERMITTED"
    return {
        "status": status,
        "input_complete": bool(input_complete),
        "coverage_sufficient": bool(coverage_ok),
        "balance_sufficient": bool(balance_ok),
        "failed_balance_features": failed,
        "outcome_evaluation_permitted": bool(coverage_ok and balance_ok),
        "outcomes_loaded": False,
        "live_authorized": False,
    }


def resolve_balance_features(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], list[str]]:
    """读取计划预注册特征；不可构造项原样列为 missing。"""

    design = plan.get("design", {})
    configured = design.get("balance_gate", {}).get("features") if isinstance(design, Mapping) else None
    features = tuple(str(value) for value in configured) if configured else DEFAULT_BALANCE_FEATURES
    missing = [feature for feature in features if feature not in AVAILABLE_BALANCE_FEATURES]
    return features, missing


def validate_plan_contract(plan: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """绑定 tracked plan 的 schema、request 身份、主规格和 outcome 锁。"""

    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise RuntimeError("unexpected tracked PIT exact-mcap plan schema")
    request_identity = plan.get("request_identity")
    if not isinstance(request_identity, Mapping):
        raise RuntimeError("tracked plan lacks request identity")
    expected_identities = {
        "request": request.get("request_identity"),
        "treated": request.get("treated_request_identity"),
        "dates": request.get("request_dates_identity"),
        "symbols": request.get("request_symbols_identity"),
    }
    for label, expected in expected_identities.items():
        if request_identity.get(label) != expected:
            raise RuntimeError(f"tracked plan request identity drift: {label}")
    design = plan.get("design")
    if not isinstance(design, Mapping) or design.get("outcome_blind") is not True:
        raise RuntimeError("tracked plan is not outcome blind")
    source = design.get("data_source", {})
    if not (
        source.get("endpoint") == "daily_basic"
        and source.get("exact_mcap_cny") == "circ_mv * 10000"
        and source.get("mix_sources_within_date") is False
        and source.get("fallback_for_missing_exact_mcap") is False
    ):
        raise RuntimeError("tracked plan exact-mcap source contract drift")
    primary = design.get("matching", {}).get("primary")
    if primary != EXPECTED_PRIMARY_PLAN:
        raise RuntimeError("tracked plan primary matching specification drift")
    balance = design.get("balance_gate", {})
    if not (
        balance.get("evaluated_before_outcome_loading") is True
        and balance.get("scopes") == {"all": 286, "2024plus": 174}
        and balance.get("smd_thresholds")
        == {"log_exact_mcap": MCAP_BALANCE_THRESHOLD, "all_other_features": BALANCE_THRESHOLD}
        and balance.get("coverage")
        == {
            "treated_exact_mcap": "286/286",
            "all": "D_supported/D_source >= 0.80",
            "2024plus": "D_supported/D_source >= 0.80",
        }
        and balance.get("fail_closed") is True
    ):
        raise RuntimeError("tracked plan balance gate drift")
    authorization = design.get("outcome_authorization", {})
    verdict = plan.get("verdict", {})
    if authorization.get("allowed_in_this_plan") is not False or verdict.get("live_authorized") is not False:
        raise RuntimeError("tracked plan authorization lock drift")
    return {
        "primary_specification_sha256": canonical_sha256(EXPECTED_PRIMARY_PLAN),
    }


def validate_reproduced_plan(expected: Mapping[str, Any], reproduced: Mapping[str, Any]) -> dict[str, bool]:
    """要求 planner 复算出的全部冻结 section 与 tracked plan 完全一致。"""

    checks = pit_plan.identity_checks(reproduced, expected)
    failed = sorted(label for label, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"tracked plan upstream identity drift: {failed}")
    return checks


def validate_rebuilt_production_identities(
    treated: pd.DataFrame,
    all_filled: pd.DataFrame,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """将本模块独立重建的 production 身份绑定到 tracked canonical 身份。"""

    expected = plan.get("source_identity", {}).get("reproducible_identity", {}).get("production")
    if not isinstance(expected, Mapping):
        raise RuntimeError("tracked plan lacks reproducible production identities")

    def date_text(value: Any) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    all_filled_records = [
        [str(row.symbol), date_text(row.dec_dt), date_text(row.entry_dt)] for row in all_filled.itertuples(index=False)
    ]
    common60_records = [
        [
            str(row.symbol),
            date_text(row.dec_dt),
            date_text(row.entry_dt),
            date_text(row.h5_dt),
            date_text(row.h20_dt),
            date_text(row.h60_dt),
        ]
        for row in treated.itertuples(index=False)
    ]
    actual = {
        "all_filled_identity": pit_plan.payload_identity(all_filled_records),
        "common60_schedule_identity": pit_plan.payload_identity(common60_records),
    }
    failed = sorted(label for label, identity in actual.items() if identity != expected.get(label))
    if failed:
        raise RuntimeError(f"independently rebuilt production identity drift: {failed}")
    return actual


def verify_tracked_plan(plan: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """复算 plan 的 source/artifact/script/industry 全链身份并 fail closed。"""

    local_contract = validate_plan_contract(plan, request)
    pit_plan.validate_design_contract(plan)
    reproduced = pit_plan.build_manifest()
    checks = validate_reproduced_plan(plan, reproduced)
    return {
        "locator": repo_locator(PLAN_PATH),
        "sha256": sha256_file(PLAN_PATH),
        "reproduction_checks": checks,
        "reproduced_source_identity_sha256": canonical_sha256(reproduced["source_identity"]),
        **local_contract,
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def load_bound_collection(
    *,
    cache_root: Path,
    verification: Mapping[str, Any],
    exact_mcap_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    """读取 collector final manifest，并严格绑定 materialized path/SHA/rows。"""

    artifacts = verification.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("collector verification lacks artifact bindings")
    final_manifest_path = resolve_cache_artifact(cache_root, artifacts.get("manifest"), "collector final manifest")
    final_manifest = load_json(final_manifest_path, "collector final manifest")
    if final_manifest.get("schema") != collector.FINAL_SCHEMA:
        raise RuntimeError("unexpected collector final manifest schema")
    materialized_meta = final_manifest.get("materialized")
    if not isinstance(materialized_meta, Mapping):
        raise RuntimeError("collector final manifest lacks materialized binding")
    bound_path = resolve_cache_artifact(cache_root, materialized_meta.get("path"), "collector materialized parquet")
    report_path = resolve_cache_artifact(cache_root, artifacts.get("materialized"), "verified materialized parquet")
    if report_path.resolve() != bound_path.resolve():
        raise RuntimeError("collector verification/final-manifest materialized path drift")
    if exact_mcap_path is not None and Path(exact_mcap_path).resolve() != bound_path.resolve():
        raise RuntimeError("unbound exact_mcap_path override")
    expected_sha = materialized_meta.get("sha256")
    expected_rows = materialized_meta.get("rows")
    if not isinstance(expected_sha, str) or not isinstance(expected_rows, int) or expected_rows <= 0:
        raise RuntimeError("collector materialized SHA/rows binding is malformed")
    if not bound_path.is_file() or sha256_file(bound_path) != expected_sha:
        raise RuntimeError("collector materialized parquet SHA binding drift")
    if not final_manifest_path.is_file():
        raise RuntimeError("collector final manifest is missing")
    identity = {
        "cache_contract": f"{collector.CACHE_ENV} or $HOME/.ts_data_cache/delay5_exact_mcap_v1",
        "final_manifest": {
            "locator": cache_locator(final_manifest_path, cache_root),
            "sha256": sha256_file(final_manifest_path),
        },
        "materialized": {
            "locator": cache_locator(bound_path, cache_root),
            "sha256": expected_sha,
            "rows": expected_rows,
        },
    }
    return bound_path, identity


def validate_materialized_rows(frame: pd.DataFrame, collection_identity: Mapping[str, Any]) -> None:
    """确保实际加载行数仍与 collector final manifest 一致。"""

    expected_rows = collection_identity.get("materialized", {}).get("rows")
    if not isinstance(expected_rows, int) or len(frame) != expected_rows:
        raise RuntimeError("collector materialized parquet row binding drift")


def build_input_identity(
    *,
    request: Mapping[str, Any],
    tracked_plan: Mapping[str, Any],
    collection: Mapping[str, Any],
    proxy_pairs: pd.DataFrame | None,
    baostock_exact: pd.DataFrame | None,
) -> dict[str, Any]:
    """生成不含绝对 cache 路径的完整审计输入身份。"""

    result: dict[str, Any] = {
        "request_manifest": {
            "locator": repo_locator(REQUEST_PATH),
            "file_sha256": sha256_file(REQUEST_PATH),
            "file_sha256_role": "local evidence only; not a cross-device hard gate",
            "canonical_identity": {
                "request": request.get("request_identity"),
                "treated": request.get("treated_request_identity"),
                "dates": request.get("request_dates_identity"),
                "symbols": request.get("request_symbols_identity"),
            },
        },
        "tracked_plan": dict(tracked_plan),
        "collection": dict(collection),
        "scripts": {
            "collector": {
                "locator": repo_locator(COLLECTOR_SCRIPT_PATH),
                "sha256": sha256_file(COLLECTOR_SCRIPT_PATH),
            },
            "plan": {"locator": repo_locator(PLAN_SCRIPT_PATH), "sha256": sha256_file(PLAN_SCRIPT_PATH)},
            "balance": {
                "locator": repo_locator(BALANCE_SCRIPT_PATH),
                "sha256": sha256_file(BALANCE_SCRIPT_PATH),
            },
        },
    }
    if proxy_pairs is not None:
        result["proxy_pairs_identity_qa"] = {
            "locator": repo_locator(PROXY_PAIRS_PATH),
            "sha256": sha256_file(PROXY_PAIRS_PATH),
            "rows_loaded": int(len(proxy_pairs)),
        }
    if baostock_exact is not None:
        result["baostock_exact_unit_qa"] = {
            "locator": repo_locator(BAOSTOCK_EXACT_PATH),
            "sha256": sha256_file(BAOSTOCK_EXACT_PATH),
            "rows": int(len(baostock_exact)),
        }
    return result


def output_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    """返回仓库相对 output locator 与物理内容身份。"""

    record: dict[str, Any] = {"locator": repo_locator(path), "sha256": sha256_file(path)}
    if rows is not None:
        record["rows"] = int(rows)
    return record


def run_audit(
    *,
    exact_mcap_path: Path | None = None,
    cache_root: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """运行结果盲 exact-mcap matching/balance；本函数从不读取未来收益列。"""

    started = time.time()
    request = load_json(REQUEST_PATH, "common exact-mcap request")
    if request.get("schema") != EXPECTED_REQUEST_SCHEMA:
        raise RuntimeError("unexpected common exact-mcap request schema")
    closure = request.get("closure", {})
    if not all(
        closure.get(key) is True for key in ("request_keys_unique", "treated_keys_unique", "all_treated_requested")
    ):
        raise RuntimeError("common exact-mcap request closure is incomplete")
    plan = load_json(PLAN_PATH, "tracked PIT exact-mcap plan")
    plan_identity = verify_tracked_plan(plan, request)
    required_features, prereg_missing = resolve_balance_features(plan)

    # Column projection is deliberate: outcome-bearing production columns never enter memory.
    cohort = pd.read_parquet(PRODUCTION_COHORT_PATH, columns=["symbol", "dec_dt", "entry_dt", "stage_fill"])
    market = pd.read_parquet(MARKET_PATH, columns=["dt"])
    calendar = normalise_calendar(market["dt"])
    treated, all_filled = build_common60_identities(cohort, calendar)
    rebuilt_production_identity = validate_rebuilt_production_identities(treated, all_filled, plan)
    rebuilt_schedule_identity = rebuilt_production_identity["common60_schedule_identity"]
    expected_schedule_identity = plan.get("treated_common_support", {}).get("schedule_identity")
    if rebuilt_schedule_identity != expected_schedule_identity:
        raise RuntimeError("rebuilt common60 schedule identity drift")
    panel = pd.read_parquet(PANEL_PATH, columns=["symbol", "dt", "close", "amount_e"])
    st_intervals = portfolio.load_st_intervals()
    if not st_intervals:
        raise RuntimeError("historical ST intervals are unavailable")
    feature_store = CausalFeatureStore.from_panel(panel, calendar, st_intervals)

    root = Path(cache_root) if cache_root is not None else collector.default_cache_root()
    verified = collector.verify_collection(request_path=REQUEST_PATH, cache_root=root)
    materialized_path, collection_identity = load_bound_collection(
        cache_root=root,
        verification=verified,
        exact_mcap_path=exact_mcap_path,
    )
    exact_mcap = pd.read_parquet(materialized_path)
    validate_materialized_rows(exact_mcap, collection_identity)
    years = sorted(treated["year"].unique())
    import s2b_industry_size_proxy_audit as industry_source

    industry_maps = {
        int(year): pd.read_parquet(industry_source._industry_path(int(year))).set_index("symbol")["industry"]
        for year in years
    }
    pairs, attempts = build_exact_matches(
        treated,
        all_filled,
        feature_store=feature_store,
        exact_mcap=exact_mcap,
        industry_maps=industry_maps,
        st_intervals=st_intervals,
        balance_features=required_features,
    )
    coverage = coverage_summary(attempts)
    balance = scoped_balance(pairs, required_features)
    for scope in BALANCE_SCOPES:
        balance[scope]["missing_features"] = sorted(set(balance[scope]["missing_features"]) | set(prereg_missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / PAIRS_PATH.name
    attempts_path = output_dir / ATTEMPTS_PATH.name
    balance_path = output_dir / BALANCE_PATH.name
    pairs.to_parquet(pairs_path, index=False)
    attempts.to_parquet(attempts_path, index=False)
    balance_payload = {
        "schema": f"{SCHEMA}_balance_v1",
        "outcomes_loaded": False,
        "required_features": list(required_features),
        "scopes": balance,
    }
    balance_path.write_text(
        json.dumps(balance_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    verdict = build_verdict(coverage, balance, required_features=required_features)
    proxy_pairs = None
    if PROXY_PAIRS_PATH.is_file():
        proxy_pairs = pd.read_parquet(PROXY_PAIRS_PATH, columns=["trade_id", "specification", "support_ok"])
        proxy_pairs = proxy_pairs[proxy_pairs["specification"].eq("industry_mcap_proxy_k10")]
    proxy_crosstab = proxy_identical_support_crosstab(attempts, proxy_pairs)
    baostock_exact = pd.read_parquet(BAOSTOCK_EXACT_PATH) if BAOSTOCK_EXACT_PATH.is_file() else None
    provider_qa = cross_provider_unit_qa(exact_mcap, baostock_exact)
    input_identity = build_input_identity(
        request=request,
        tracked_plan=plan_identity,
        collection=collection_identity,
        proxy_pairs=proxy_pairs,
        baostock_exact=baostock_exact,
    )
    audit = {
        "schema": SCHEMA,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "outcome_blind": True,
        "outcomes_loaded": False,
        "specification": SPECIFICATION,
        "parameters": {"k": K, "min_controls": MIN_CONTROLS, "caliper_ratio": CALIPER_RATIO},
        "cohort": {
            "D_source": int(len(treated)),
            "D_source_2024plus": int(treated["year"].ge(2024).sum()),
            "decision_dates": int(treated["dec_dt"].nunique()),
            "identity": frame_identity(treated, ["trade_id", "symbol", "dec_dt"]),
            "schedule_identity": rebuilt_schedule_identity,
            "all_filled_identity": rebuilt_production_identity["all_filled_identity"],
        },
        "collector_closure": verified,
        "input_identity": input_identity,
        "coverage": coverage,
        "proxy_identical_support": proxy_crosstab,
        "cross_provider_unit_qa": provider_qa,
        "balance": balance,
        "outputs": {
            "matched_pairs": output_record(pairs_path, rows=len(pairs)),
            "match_attempts": output_record(attempts_path, rows=len(attempts)),
            "balance": output_record(balance_path),
        },
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    audit_path = output_dir / AUDIT_PATH.name
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    audit = run_audit()
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
