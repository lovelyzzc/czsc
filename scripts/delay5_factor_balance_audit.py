"""Production delay5 的决策日前因子余额与探索性共同期限审计。

本脚本不搜索参数。它严格绑定 ``delay5_common_horizon_att`` 的最新输出及其上游，冻结复现两套
曾用于证伪的匹配规格：

1. 行业 + 当前股本市值代理 1.5x 池内，全特征标准化最近邻 K5/min3；
2. 同一池内精确匹配近 20 日涨停次数与最长连板，再做标准化最近邻 K5/min3。

六项行情/路径特征只使用决策日及以前的 panel 行情；规模卡尺与距离仍含当前股本这一非点时
代理。结果是 post-hoc 探索性证伪，余额不足或任何名义显著都不能触发实盘授权。

    uv run --no-sync python scripts/delay5_factor_balance_audit.py
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

import delay5_common_horizon_att as common
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR / "_output" / "delay5_common_horizon_att"
COMMON_AUDIT_PATH = COMMON_DIR / "audit.json"
COMMON_OUTPUT_PATHS = {
    "audit": COMMON_AUDIT_PATH,
    "treated_common_support": COMMON_DIR / "treated_common_support.parquet",
    "matched_pairs": COMMON_DIR / "matched_pairs.parquet",
    "trade_att": COMMON_DIR / "trade_att.parquet",
    "exact_mcap_request_manifest": COMMON_DIR / "exact_mcap_request_manifest.json",
}
OUTPUT_DIR = SCRIPT_DIR / "_output" / "delay5_factor_balance_audit"
PAIRS_PATH = OUTPUT_DIR / "matched_pairs.parquet"
TRADE_ATT_PATH = OUTPUT_DIR / "trade_att.parquet"
AUDIT_PATH = OUTPUT_DIR / "audit.json"

EXPECTED_COMMON_SCHEMA = "delay5_common_horizon_att_audit_v1"
EXPECTED_EXACT_SCHEMA = "delay5_exact_mcap_request_manifest_v2"
SCHEMA = "delay5_factor_balance_audit_v2"

SPEC_ALL = "factor_std_all_k5"
SPEC_EXACT = "factor_exact_limitup_k5"
SPECIFICATIONS = (SPEC_ALL, SPEC_EXACT)
K = 5
MIN_CONTROLS = 3
MCAP_CALIPER_RATIO = 1.5
N_BOOT = 10_000
BLOCK_LEN = 10
BALANCE_THRESHOLD = 0.10
BALANCE_SCOPES = ("all", "2024plus")

BALANCE_FEATURES = (
    "ret5",
    "ret20",
    "vol20",
    "log_price",
    "limitup_count20",
    "limitup_maxrun20",
)
DISTANCE_CONTINUOUS = ("ret5", "ret20", "vol20", "log_price", "log_mcap_proxy")
DISTANCE_ALL = (*DISTANCE_CONTINUOUS, "limitup_count20", "limitup_maxrun20")
EXACT_LIMITUP = ("limitup_count20", "limitup_maxrun20")
DATE_COLUMNS = ("sig_dt", "dec_dt", "entry_dt", "exit_dt", "h5_dt", "h20_dt", "h60_dt")

SPECIFICATION_PAYLOAD = {
    "causal_cutoff": "feature_max_dt <= dec_dt",
    "returns": {
        "ret5": "100*(LOCF_close[t]/LOCF_close[t-5]-1)",
        "ret20": "100*(LOCF_close[t]/LOCF_close[t-20]-1)",
    },
    "vol20": "std(ddof=1) of 20 LOCF common-session returns ending t, annualized sqrt(252)*100",
    "price": "natural log of actual qfq close on decision date; no backfill",
    "limitup": {
        "proxy": "actual close / prior LOCF close; actual-close missing is false and breaks runs",
        "window": "20 common sessions inclusive of decision date",
        "thresholds_pct": {"main": 9.8, "chinext": 19.8, "historical_st": 4.8},
        "historical_st_semantics": "latest name-change state whose start_date <= session",
    },
    "base_pool": "same frozen annual industry and current-share mcap proxy within 1.5x",
    "standardization": "per-treated z scale fit on treated plus its decision-date industry/caliper pool",
    "zero_variance": "dimension contributes zero distance and is retained in audit",
    "matching": {
        SPEC_ALL: {"continuous": DISTANCE_ALL, "exact": (), "k": K, "min_controls": MIN_CONTROLS},
        SPEC_EXACT: {
            "continuous": DISTANCE_CONTINUOUS,
            "exact": EXACT_LIMITUP,
            "k": K,
            "min_controls": MIN_CONTROLS,
        },
        "tie_break": "match_distance, symbol",
    },
    "control_reducer": "median, matching the bound common-horizon audit; this is a matched-median contrast",
    "balance": {
        "scopes": BALANCE_SCOPES,
        "threshold": "all audited features must have finite abs(SMD) < 0.1 in both scopes",
    },
    "inference": {
        "descriptive_estimand": "equal-weighted treated-trade matched-median contrast",
        "cluster_estimand": "unweighted decision-date mean of within-date treated contrasts",
        "hac": "Newey-West automatic lag",
        "bootstrap": {
            "method": "stationary bootstrap",
            "expected_block_len": BLOCK_LEN,
            "replications": N_BOOT,
            "seed": 42,
        },
    },
}


def sha256_file(path: Path) -> str:
    """使用 common 审计相同的流式 SHA256。"""

    return common.proxy.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    """对规格 JSON 做稳定哈希。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")


def verify_bound_file(path: Path, expected_sha256: str | None, label: str) -> dict[str, Any]:
    """验证绑定文件路径和 SHA；任何缺失身份都 fail closed。"""

    require_file(path, label)
    if not expected_sha256:
        raise RuntimeError(f"{label} binding lacks sha256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"stale {label} binding: expected {expected_sha256}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _verify_recorded_path(record: Mapping[str, Any], expected_path: Path, label: str) -> None:
    recorded = record.get("path")
    if not recorded or Path(str(recorded)).resolve() != expected_path.resolve():
        raise RuntimeError(f"{label} path binding differs from expected local input")


def validate_common_upstream(common_audit: Mapping[str, Any]) -> dict[str, Any]:
    """复核 common audit 中记录的 production、行情、ST、股本和行业快照身份。"""

    if common_audit.get("schema") != EXPECTED_COMMON_SCHEMA:
        raise RuntimeError("unexpected common-horizon audit schema")
    if common_audit.get("verdict", {}).get("live_authorized") is not False:
        raise RuntimeError("bound common-horizon audit must not authorize live trading")
    identities = common_audit.get("data_identity")
    if not isinstance(identities, Mapping):
        raise RuntimeError("common-horizon audit lacks data_identity")

    expected = {
        "candidates": common.CANDIDATES_PATH,
        "panel": common.PANEL_PATH,
        "market_state": common.MARKET_PATH,
        "namechange": common.portfolio.NAMECHANGE_PATH,
        "capital_proxy": common.proxy.CAPITAL_PATH,
        "production_cohort": common.PRODUCTION_COHORT_PATH,
    }
    verified: dict[str, Any] = {}
    for label, path in expected.items():
        record = identities.get(label)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"common-horizon audit lacks {label} identity")
        _verify_recorded_path(record, path, label)
        verified[label] = verify_bound_file(path, record.get("sha256"), label)

    production = identities["production_cohort"]
    if production.get("upstream_schema") != "surge_delay5_production_cohort_audit_v2":
        raise RuntimeError("unexpected bound production cohort schema")
    _verify_recorded_path(
        {"path": production.get("upstream_audit_path")}, common.PRODUCTION_AUDIT_PATH, "production audit"
    )
    verified["production_audit"] = verify_bound_file(
        common.PRODUCTION_AUDIT_PATH,
        production.get("upstream_audit_sha256"),
        "production audit",
    )

    industry_records = identities.get("industry_snapshots")
    if not isinstance(industry_records, Mapping) or not industry_records:
        raise RuntimeError("common-horizon audit lacks frozen industry identities")
    verified_industries: dict[str, Any] = {}
    for year_text, record in industry_records.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"industry identity is malformed for {year_text}")
        path = common.proxy._industry_path(int(year_text))
        _verify_recorded_path(record, path, f"industry {year_text}")
        identity = verify_bound_file(path, record.get("file_sha256"), f"industry {year_text}")
        frame = pd.read_parquet(path)
        canonical = common.proxy.canonical_industry_sha256(frame)
        if canonical != record.get("canonical_content_sha256"):
            raise RuntimeError(f"stale canonical industry binding for {year_text}")
        identity["canonical_content_sha256"] = canonical
        verified_industries[str(year_text)] = identity
    verified["industry_snapshots"] = verified_industries
    return verified


def load_bound_common_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """读取并交叉验证 common 的五项输出，返回 treated、pairs、trade_att 和身份对象。"""

    for label, path in COMMON_OUTPUT_PATHS.items():
        require_file(path, f"common {label}")
    common_audit = json.loads(COMMON_AUDIT_PATH.read_text(encoding="utf-8"))
    upstream = validate_common_upstream(common_audit)
    exact_manifest = json.loads(COMMON_OUTPUT_PATHS["exact_mcap_request_manifest"].read_text(encoding="utf-8"))
    validate_exact_request_closure(exact_manifest)

    treated = pd.read_parquet(COMMON_OUTPUT_PATHS["treated_common_support"])
    pairs = pd.read_parquet(COMMON_OUTPUT_PATHS["matched_pairs"])
    trade_att = pd.read_parquet(COMMON_OUTPUT_PATHS["trade_att"])
    for frame in (treated, pairs, trade_att):
        for column in DATE_COLUMNS:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    if "trade_id" not in treated or treated["trade_id"].duplicated().any():
        raise RuntimeError("common treated output lacks a unique trade_id")
    if len(treated) != int(exact_manifest.get("cohort", {}).get("treated_trades", -1)):
        raise RuntimeError("common treated rows differ from exact-mcap manifest")
    required_pairs = {
        "trade_id",
        "specification",
        "support_ok",
        "treated_symbol",
        "control_symbol",
        "dec_dt",
    }
    if missing := required_pairs - set(pairs):
        raise RuntimeError(f"common matched pairs lack columns: {sorted(missing)}")
    if common.SPEC_PROXY not in set(pairs["specification"]):
        raise RuntimeError("common matched pairs lack the frozen proxy specification")
    manifest_request = exact_manifest.get("request_identity", {})
    audit_request = common_audit.get("exact_mcap_request", {})
    for key in ("count", "bytes", "sha256"):
        if manifest_request.get(key) != audit_request.get(key):
            raise RuntimeError(f"exact-mcap request identity mismatch on {key}")

    output_identity = validate_common_output_bindings(common_audit, treated, pairs, trade_att, exact_manifest)
    output_identity["audit"] = {
        "path": str(COMMON_AUDIT_PATH),
        "sha256": sha256_file(COMMON_AUDIT_PATH),
        "self_bound_by_common_audit": False,
    }
    return (
        treated,
        pairs,
        trade_att,
        {
            "common_outputs": output_identity,
            "common_upstream": upstream,
            "common_audit": common_audit,
            "exact_manifest": exact_manifest,
        },
    )


def validate_common_output_bindings(
    common_audit: Mapping[str, Any],
    treated: pd.DataFrame,
    pairs: pd.DataFrame,
    trade_att: pd.DataFrame,
    exact_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """逐一验证 common audit 已记录的四个非循环输出 path/SHA/rows。"""

    records = common_audit.get("outputs")
    if not isinstance(records, Mapping):
        raise RuntimeError("common-horizon audit lacks bound outputs")
    row_counts = {
        "treated_common_support": len(treated),
        "matched_pairs": len(pairs),
        "trade_att": len(trade_att),
        "exact_mcap_request_manifest": exact_manifest.get("request_identity", {}).get("count"),
    }
    verified: dict[str, Any] = {}
    for label, rows in row_counts.items():
        record = records.get(label)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"common-horizon audit lacks {label} output identity")
        path = COMMON_OUTPUT_PATHS[label]
        _verify_recorded_path(record, path, f"common output {label}")
        identity = verify_bound_file(path, record.get("sha256"), f"common output {label}")
        if rows is None or int(record.get("rows", -1)) != int(rows):
            raise RuntimeError(f"common output {label} row count differs from audit")
        identity["rows"] = int(rows)
        verified[label] = identity
    return verified


def validate_exact_request_closure(exact_manifest: Mapping[str, Any]) -> None:
    """要求 common exact-mcap v2 请求对 treated 身份形成完整闭包。"""

    if exact_manifest.get("schema") != EXPECTED_EXACT_SCHEMA:
        raise RuntimeError("unexpected exact-mcap request schema")
    closure = exact_manifest.get("closure", {})
    if not all(
        closure.get(field) is True for field in ("request_keys_unique", "treated_keys_unique", "all_treated_requested")
    ):
        raise RuntimeError("exact-mcap request closure is incomplete")


def build_limitup_matrix(
    raw_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    st_intervals: Mapping[str, Sequence[tuple[str, str, bool]]],
    *,
    base_limit_pct: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """构造仅用当日实际收盘和前一日 LOCF 收盘的涨停收盘代理。"""

    if not raw_close.index.equals(mark_close.index) or not raw_close.columns.equals(mark_close.columns):
        raise ValueError("raw_close and mark_close axes differ")
    thresholds = pd.Series(
        base_limit_pct or {str(symbol): common.tr.limit_pct_for(str(symbol)) for symbol in raw_close.columns},
        dtype=float,
    ).reindex(raw_close.columns)
    if thresholds.isna().any():
        raise ValueError("base limit threshold is missing for panel symbols")
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
        st_hits = observed[symbol].to_numpy(dtype=bool) & returns_pct[symbol].ge(4.8).to_numpy(dtype=bool)
        values = result[symbol].to_numpy(dtype=bool)
        values[is_st] = st_hits[is_st]
        result[symbol] = values
    return result.astype(bool)


def longest_true_run(window: np.ndarray) -> np.ndarray:
    """按列返回布尔窗口的最长连续 True；False（含停牌）会中断。"""

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
    mark_close: pd.DataFrame,
    limitup: pd.DataFrame,
    dec_dt: pd.Timestamp,
) -> pd.DataFrame:
    """构造单个决策日截面的冻结特征；切片上界始终为 dec_dt。"""

    if not raw_close.index.equals(mark_close.index) or not raw_close.index.equals(limitup.index):
        raise ValueError("feature matrix calendars differ")
    if not raw_close.columns.equals(mark_close.columns) or not raw_close.columns.equals(limitup.columns):
        raise ValueError("feature matrix symbols differ")
    decision = pd.Timestamp(dec_dt).normalize()
    if decision not in raw_close.index:
        raise ValueError(f"decision date is absent from calendar: {decision.date()}")
    position = int(raw_close.index.get_loc(decision))
    symbols = raw_close.columns
    current = mark_close.iloc[position]
    actual = raw_close.iloc[position]
    result = pd.DataFrame(index=symbols)
    result.index.name = "symbol"
    result["ret5"] = (current / mark_close.iloc[position - 5] - 1) * 100 if position >= 5 else np.nan
    result["ret20"] = (current / mark_close.iloc[position - 20] - 1) * 100 if position >= 20 else np.nan
    if position >= 20:
        marks = mark_close.iloc[position - 20 : position + 1]
        daily = marks.pct_change(fill_method=None).iloc[1:]
        result["vol20"] = daily.std(axis=0, ddof=1) * math.sqrt(252) * 100
    else:
        result["vol20"] = np.nan
    result["log_price"] = np.log(actual.where(actual.gt(0)))
    if position >= 19:
        window = limitup.iloc[position - 19 : position + 1].to_numpy(dtype=bool)
        result["limitup_count20"] = window.sum(axis=0).astype(int)
        result["limitup_maxrun20"] = longest_true_run(window)
    else:
        result["limitup_count20"] = np.nan
        result["limitup_maxrun20"] = np.nan
    result["feature_max_dt"] = decision
    return result


@dataclass
class DecisionFeatureStore:
    """缓存按决策日生成的因果截面。"""

    raw_close: pd.DataFrame
    mark_close: pd.DataFrame
    limitup: pd.DataFrame
    _cache: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_panel(
        cls,
        panel: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        st_intervals: Mapping[str, Sequence[tuple[str, str, bool]]],
    ) -> DecisionFeatureStore:
        required = {"symbol", "dt", "close"}
        if missing := required - set(panel):
            raise ValueError(f"panel lacks feature columns: {sorted(missing)}")
        scoped = panel.loc[:, ["symbol", "dt", "close"]].copy()
        scoped["dt"] = pd.to_datetime(scoped["dt"]).dt.normalize()
        if scoped.duplicated(["dt", "symbol"]).any():
            raise ValueError("panel has duplicate symbol/date rows")
        raw_close = scoped.pivot(index="dt", columns="symbol", values="close").reindex(calendar)
        mark_close = raw_close.ffill()
        limitup = build_limitup_matrix(raw_close, mark_close, st_intervals)
        return cls(raw_close=raw_close, mark_close=mark_close, limitup=limitup)

    def snapshot(self, dec_dt: pd.Timestamp) -> pd.DataFrame:
        decision = pd.Timestamp(dec_dt).normalize()
        if decision not in self._cache:
            self._cache[decision] = decision_feature_snapshot(
                self.raw_close,
                self.mark_close,
                self.limitup,
                decision,
            )
        return self._cache[decision]


def attach_pair_features(pairs: pd.DataFrame, store: DecisionFeatureStore) -> pd.DataFrame:
    """给匹配对附加 treated/control 的决策日特征。"""

    parts: list[pd.DataFrame] = []
    for dec_dt, group in pairs.groupby("dec_dt", sort=True):
        snapshot = store.snapshot(pd.Timestamp(dec_dt))
        scoped = group.copy()
        for feature in BALANCE_FEATURES:
            scoped[f"treated_{feature}"] = scoped["treated_symbol"].map(snapshot[feature])
            scoped[f"control_{feature}"] = scoped["control_symbol"].map(snapshot[feature])
        parts.append(scoped)
    if not parts:
        return pairs.copy()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["specification", "dec_dt", "treated_symbol", "rank"], kind="mergesort"
    )


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or len(values) != len(weights) or not np.isfinite(values).all():
        raise ValueError("weighted moments received invalid values")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weighted moments received invalid weights")
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def balance_statistics(pairs: pd.DataFrame, features: Sequence[str] = BALANCE_FEATURES) -> dict[str, Any]:
    """每个 treated 权重为 1，其控制各权重 1/K_i，报告 pooled-SD SMD。"""

    if pairs.empty:
        raise ValueError("cannot compute balance on empty pairs")
    if pairs["trade_id"].isna().any():
        raise ValueError("balance pairs lack trade_id")
    treated = pairs.drop_duplicates("trade_id", keep="first")
    control_weights = 1.0 / pairs.groupby("trade_id")["trade_id"].transform("size").to_numpy(dtype=float)
    result: dict[str, Any] = {}
    for feature in features:
        treated_values = pd.to_numeric(treated[f"treated_{feature}"], errors="coerce").to_numpy(dtype=float)
        control_values = pd.to_numeric(pairs[f"control_{feature}"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(treated_values).all() or not np.isfinite(control_values).all():
            raise RuntimeError(f"supported balance sample has missing {feature}")
        treated_mean, treated_var = _weighted_moments(treated_values, np.ones(len(treated_values)))
        control_mean, control_var = _weighted_moments(control_values, control_weights)
        pooled = math.sqrt((treated_var + control_var) / 2)
        if pooled == 0:
            smd = 0.0 if treated_mean == control_mean else math.copysign(math.inf, treated_mean - control_mean)
        else:
            smd = (treated_mean - control_mean) / pooled
        result[str(feature)] = {
            "treated_mean": round(treated_mean, 6),
            "control_mean": round(control_mean, 6),
            "smd": round(float(smd), 6),
            "abs_smd_lt_0_1": bool(abs(smd) < BALANCE_THRESHOLD),
        }
    return result


def standardized_nearest_controls(
    pool: pd.DataFrame,
    target: pd.Series,
    *,
    continuous: Sequence[str],
    exact: Sequence[str] = (),
    k: int = K,
) -> tuple[pd.DataFrame, int, list[str]]:
    """在单个 treated 的因果候选池内拟合尺度，按距离和 symbol 确定性取邻居。"""

    required = {"symbol", *continuous, *exact}
    if missing := required - set(pool):
        raise ValueError(f"factor pool lacks columns: {sorted(missing)}")
    target_values = pd.to_numeric(target[list(continuous)], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        return pool.iloc[:0].assign(match_distance=pd.Series(dtype=float)), 0, []
    eligible = pool.copy()
    for column in exact:
        if pd.isna(target.get(column)):
            return pool.iloc[:0].assign(match_distance=pd.Series(dtype=float)), 0, []
        eligible = eligible[eligible[column].eq(target[column])].copy()
    matrix = eligible.loc[:, list(continuous)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1)
    eligible = eligible.loc[finite].copy()
    matrix = matrix.loc[finite]
    if eligible.empty:
        return eligible.assign(match_distance=pd.Series(dtype=float)), 0, []
    scale_matrix = np.vstack([target_values, matrix.to_numpy(dtype=float)])
    scale = scale_matrix.std(axis=0, ddof=0)
    active = np.isfinite(scale) & (scale > 0)
    zero_variance = [str(name) for name, enabled in zip(continuous, active, strict=False) if not enabled]
    if active.any():
        deltas = (matrix.to_numpy(dtype=float)[:, active] - target_values[active]) / scale[active]
        distance = np.sqrt(np.square(deltas).sum(axis=1))
    else:
        distance = np.zeros(len(eligible), dtype=float)
    eligible["match_distance"] = distance
    matched = eligible.sort_values(["match_distance", "symbol"], kind="mergesort").head(k).copy()
    return matched, int(len(eligible)), zero_variance


def build_factor_matches(
    treated: pd.DataFrame,
    all_filled: pd.DataFrame,
    *,
    store: DecisionFeatureStore,
    open_w: pd.DataFrame,
    close_w: pd.DataFrame,
    amount_w: pd.DataFrame,
    mark_w: pd.DataFrame,
    last_dt: pd.Series,
    industry_maps: Mapping[int, pd.Series],
    shares: pd.Series,
    st_intervals: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造两套冻结 factor 匹配；收益路径完全复用 common 执行语义。"""

    treated_by_date = {
        pd.Timestamp(dec_dt): set(group["symbol"].astype(str)) for dec_dt, group in all_filled.groupby("dec_dt")
    }
    pair_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    configs = {
        SPEC_ALL: (DISTANCE_ALL, ()),
        SPEC_EXACT: (DISTANCE_CONTINUOUS, EXACT_LIMITUP),
    }
    for dec_dt, day_trades in treated.groupby("dec_dt", sort=True):
        decision = pd.Timestamp(dec_dt)
        industry = industry_maps[int(decision.year)]
        full, pool = common._decision_snapshot(
            decision,
            close_w=close_w,
            amount_w=amount_w,
            industry=industry,
            shares=shares,
            st_intervals=st_intervals,
            treated_symbols=treated_by_date[decision],
        )
        snapshot = store.snapshot(decision)
        full = full.join(snapshot[list(BALANCE_FEATURES)], how="left")
        pool = pool.join(snapshot[list(BALANCE_FEATURES)], how="left")
        full["log_mcap_proxy"] = np.log(full["mcap_proxy"].where(full["mcap_proxy"].gt(0)))
        pool["log_mcap_proxy"] = np.log(pool["mcap_proxy"].where(pool["mcap_proxy"].gt(0)))
        for row in day_trades.itertuples(index=False):
            target = full.loc[row.symbol] if row.symbol in full.index else pd.Series(dtype=float)
            target_industry = target.get("industry")
            target_mcap = float(target.get("mcap_proxy", np.nan))
            base = pool.iloc[:0].copy()
            if (
                target_industry is not None
                and not pd.isna(target_industry)
                and np.isfinite(target_mcap)
                and target_mcap > 0
            ):
                base = pool[
                    pool["industry"].eq(target_industry)
                    & pool["mcap_proxy"].between(
                        target_mcap / MCAP_CALIPER_RATIO,
                        target_mcap * MCAP_CALIPER_RATIO,
                    )
                ].copy()
            for specification, (continuous, exact) in configs.items():
                matched, eligible_n, zero_variance = standardized_nearest_controls(
                    base,
                    target,
                    continuous=continuous,
                    exact=exact,
                    k=K,
                )
                support_ok = len(matched) >= MIN_CONTROLS
                attempt_rows.append(
                    {
                        "trade_id": row.trade_id,
                        "specification": specification,
                        "symbol": row.symbol,
                        "dec_dt": decision,
                        "year": int(decision.year),
                        "proxy_caliper_pool_n": int(len(base)),
                        "feature_eligible_pool_n": eligible_n,
                        "selected_controls_n": int(len(matched)),
                        "support_ok": support_ok,
                        "zero_variance_features": json.dumps(zero_variance, ensure_ascii=False),
                    }
                )
                for rank, control in enumerate(matched.itertuples(), start=1):
                    record: dict[str, Any] = {
                        "trade_id": row.trade_id,
                        "specification": specification,
                        "support_ok": support_ok,
                        "treated_symbol": row.symbol,
                        "control_symbol": control.symbol,
                        "dec_dt": decision,
                        "entry_dt": row.entry_dt,
                        **{f"h{h}_dt": getattr(row, f"h{h}_dt") for h in common.HORIZONS},
                        "year": int(decision.year),
                        "rank": rank,
                        "proxy_caliper_pool_n": int(len(base)),
                        "eligible_pool_n": eligible_n,
                        "selected_controls_n": int(len(matched)),
                        "treated_industry": None if pd.isna(target_industry) else str(target_industry),
                        "match_distance": float(control.match_distance),
                    }
                    for feature in (*BALANCE_FEATURES, "log_mcap_proxy"):
                        record[f"treated_{feature}"] = float(target.get(feature, np.nan))
                        record[f"control_{feature}"] = float(getattr(control, feature))
                    record.update(
                        common._control_path_from_wide(
                            control.symbol,
                            row,
                            open_w=open_w,
                            close_w=close_w,
                            mark_w=mark_w,
                            last_dt=last_dt,
                        )
                    )
                    pair_rows.append(record)
    if not pair_rows:
        raise RuntimeError("factor matching produced no candidate pairs")
    pairs = pd.DataFrame(pair_rows).sort_values(["specification", "dec_dt", "treated_symbol", "rank"], kind="mergesort")
    attempts = pd.DataFrame(attempt_rows).sort_values(["specification", "dec_dt", "symbol"], kind="mergesort")
    return pairs.reset_index(drop=True), attempts.reset_index(drop=True)


def support_summary(attempts: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for specification in SPECIFICATIONS:
        scope = attempts[attempts["specification"].eq(specification)]
        supported = scope[scope["support_ok"]]
        pair_scope = pairs[pairs["specification"].eq(specification)]
        supported_pairs = pair_scope[pair_scope["support_ok"]]
        recent = scope[scope["year"].ge(2024)]
        recent_supported = recent[recent["support_ok"]]
        controls = supported["selected_controls_n"].value_counts().sort_index()
        result[specification] = {
            "all": {
                "source_trades": int(len(scope)),
                "supported_trades": int(len(supported)),
                "support_rate_pct": round(float(len(supported) / len(scope) * 100), 2),
            },
            "2024plus": {
                "source_trades": int(len(recent)),
                "supported_trades": int(len(recent_supported)),
                "support_rate_pct": round(float(len(recent_supported) / len(recent) * 100), 2),
            },
            "selected_pairs": int(len(pair_scope)),
            "supported_pairs": int(len(supported_pairs)),
            "supported_control_count_distribution": {str(int(key)): int(value) for key, value in controls.items()},
            "proxy_caliper_pool": {
                "p05": round(float(scope["proxy_caliper_pool_n"].quantile(0.05)), 3),
                "median": round(float(scope["proxy_caliper_pool_n"].median()), 3),
                "p95": round(float(scope["proxy_caliper_pool_n"].quantile(0.95)), 3),
            },
        }
    return result


def summarize_2024plus(trade_att: pd.DataFrame, *, n_boot: int) -> dict[str, Any]:
    """报告 2024+ 三期限 matched-median contrast 的 HAC 与期望块长 10 的平稳 bootstrap。"""

    result: dict[str, Any] = {}
    for specification in SPECIFICATIONS:
        scope = trade_att[trade_att["specification"].eq(specification) & trade_att["year"].ge(2024)]
        horizons: dict[str, Any] = {}
        for horizon in common.HORIZONS:
            gross = common.summarize_att(scope, f"att_gross_h{horizon}_pct", n_boot=n_boot)
            net = common.summarize_att(scope, f"att_net_h{horizon}_pct", n_boot=n_boot)
            horizons[str(horizon)] = {
                "gross": gross,
                "net": net,
                "gross_nominal_gate": common._nominal_gate(gross),
                "net_nominal_gate": common._nominal_gate(net),
            }
        result[specification] = {
            "n_trades": int(len(scope)),
            "n_decision_dates": int(scope["dec_dt"].nunique()),
            "estimands": {
                "mean_att_pct": "equal-weighted treated-trade descriptive mean",
                "daily_hac_and_bootstrap_mean": "unweighted decision-date mean; not the trade-weighted mean",
            },
            "horizons": horizons,
        }
    return result


def scoped_balance_statistics(pairs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """分别输出全期与 2024+ SMD；任一正式范围无支持时 fail closed。"""

    result: dict[str, dict[str, Any]] = {}
    scopes = {
        "all": pairs,
        "2024plus": pairs[pairs["year"].ge(2024)],
    }
    for scope_name, scope in scopes.items():
        if scope.empty:
            raise RuntimeError(f"factor balance scope has no supported pairs: {scope_name}")
        result[scope_name] = balance_statistics(scope)
    return result


def build_verdict(post_match_balance: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """探索审计永不授权；当前余额不足时显式列出失败特征。"""

    failures: dict[str, dict[str, list[str]]] = {}
    for specification in SPECIFICATIONS:
        if specification not in post_match_balance:
            raise RuntimeError(f"factor balance is missing specification: {specification}")
        failures[specification] = {}
        for scope_name in BALANCE_SCOPES:
            if scope_name not in post_match_balance[specification]:
                raise RuntimeError(f"factor balance is missing scope: {specification}/{scope_name}")
            balance = post_match_balance[specification][scope_name]
            failed: list[str] = []
            for feature in BALANCE_FEATURES:
                values = balance.get(feature)
                smd = math.inf if values is None else float(values.get("smd", math.inf))
                if not math.isfinite(smd) or abs(smd) >= BALANCE_THRESHOLD:
                    failed.append(feature)
            failures[specification][scope_name] = failed
    any_failure = any(failed for scopes in failures.values() for failed in scopes.values())
    return {
        "status": (
            "EXPLORATORY_FACTOR_BALANCE_EDGE_NOT_CONFIRMED_BALANCE_INSUFFICIENT"
            if any_failure
            else "EXPLORATORY_FACTOR_BALANCE_EDGE_NOT_CONFIRMED_POST_HOC_ONLY"
        ),
        "edge_confirmed": False,
        "balance_threshold_abs_smd": BALANCE_THRESHOLD,
        "balance_sufficient": not any_failure,
        "failed_balance_features": failures,
        "confirmatory": False,
        "live_authorized": False,
    }


def _json_clean(value: Any) -> Any:
    return common._json_clean(value)


def run_audit(*, n_boot: int = N_BOOT, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """运行完整只读输入审计并写出 ignored 复现产物。"""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    started = time.time()
    treated, common_pairs, _common_trade_att, identities = load_bound_common_outputs()
    market_state = pd.read_parquet(common.MARKET_PATH)
    calendar = common._normalise_dates(market_state["dt"])
    upstream_treated, all_filled, production_funnel, production_audit = common.load_production_common_support(calendar)
    if set(treated["trade_id"]) != set(
        upstream_treated["symbol"] + "|" + upstream_treated["dec_dt"].dt.strftime("%Y-%m-%d")
    ):
        raise RuntimeError("common treated identities differ from current bound production support")

    panel = pd.read_parquet(common.PANEL_PATH)
    st_intervals = common.portfolio.load_st_intervals()
    if not st_intervals:
        raise RuntimeError("historical ST intervals are unavailable")
    open_w, close_w, amount_w, mark_w, last_dt = common._wide_panel(panel, calendar)
    store = DecisionFeatureStore.from_panel(panel, calendar, st_intervals)
    industry_maps, industry_identities = common._load_industry_maps(treated["dec_dt"].dt.year.unique())
    capital = pd.read_parquet(common.proxy.CAPITAL_PATH)
    shares = capital.set_index("symbol")["current_float_shares"]

    existing_proxy = common_pairs[
        common_pairs["specification"].eq(common.SPEC_PROXY) & common_pairs["support_ok"].astype(bool)
    ].copy()
    existing_proxy = attach_pair_features(existing_proxy, store)
    existing_proxy_balance = balance_statistics(existing_proxy)

    factor_pairs, attempts = build_factor_matches(
        treated,
        all_filled,
        store=store,
        open_w=open_w,
        close_w=close_w,
        amount_w=amount_w,
        mark_w=mark_w,
        last_dt=last_dt,
        industry_maps=industry_maps,
        shares=shares,
        st_intervals=st_intervals,
    )
    trade_att = common.build_trade_att(treated, factor_pairs)
    post_match_balance = {
        specification: scoped_balance_statistics(
            factor_pairs[factor_pairs["specification"].eq(specification) & factor_pairs["support_ok"].astype(bool)]
        )
        for specification in SPECIFICATIONS
    }
    support = support_summary(attempts, factor_pairs)
    inference = summarize_2024plus(trade_att, n_boot=n_boot)
    verdict = build_verdict(post_match_balance)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / PAIRS_PATH.name
    trade_path = output_dir / TRADE_ATT_PATH.name
    factor_pairs.to_parquet(pairs_path, index=False)
    trade_att.to_parquet(trade_path, index=False)
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "exploratory": True,
        "post_outcome_known": True,
        "parameters_frozen_before_this_audit": True,
        "parameter_search_performed": False,
        "specification": SPECIFICATION_PAYLOAD,
        "specification_sha256": canonical_sha256(SPECIFICATION_PAYLOAD),
        "cohort": {
            "common_60_session_treated": int(len(treated)),
            "decision_dates": int(treated["dec_dt"].nunique()),
            "year_counts": common._year_counts(treated),
            "production_funnel": production_funnel,
        },
        "existing_proxy_balance": {
            "supported_trades": int(existing_proxy["trade_id"].nunique()),
            "supported_pairs": int(len(existing_proxy)),
            "features": existing_proxy_balance,
        },
        "exploratory_matches": {
            "support": support,
            "post_match_balance": post_match_balance,
            "inference_2024plus": inference,
        },
        "input_identity": {
            **identities,
            "production_schema": production_audit["schema"],
            "industry_snapshots_reloaded": industry_identities,
        },
        "outputs": {
            "matched_pairs": {
                "path": str(pairs_path),
                "sha256": sha256_file(pairs_path),
                "rows": int(len(factor_pairs)),
            },
            "trade_att": {"path": str(trade_path), "sha256": sha256_file(trade_path), "rows": int(len(trade_att))},
        },
        "limitations": [
            "This is a post-outcome exploratory falsification audit, not a pre-registered confirmatory design.",
            "qfq log price and qfq close times current float shares are not exact point-in-time covariates.",
            "Limit-up history is a close-limit proxy and cannot identify intraday touches.",
            "The common audit uses median control outcomes, so reported ATT fields are matched-median contrasts.",
            "Trade-weighted descriptive means and unweighted decision-date HAC/bootstrap means are different estimands.",
            "The stationary bootstrap expected block length 10 is short relative to H60 overlap; control reuse adds dependence.",
            "Raw cohort completeness remains unproven; matching balance cannot repair selection into the dump.",
        ],
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    cleaned = _json_clean(audit)
    (output_dir / AUDIT_PATH.name).write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return cleaned


def main() -> None:
    audit = run_audit()
    print(json.dumps(audit["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
