"""Production delay5 固定 5/20/60 日共同期限 matched-median contrast 审计。

严格复原 ``raw → signal gate → hard → ST → market → fill → mature``，仅让从
入场日起已经完整观察 60 个全市场交易日的成交候选进入共同支持集。5/20/60 日使用
同一处理票和同一批预先匹配控制票；控制池只使用决策日收盘可见信息。

冻结两套规格：同成交额十分位最近 log-amount K50/min10，以及冻结年度行业内最近
log(决策日 qfq 收盘价×当前流通股本) K10/min5、1.5 倍卡尺。后者含事后股本信息，
只能作稳健性反证，不能替代点时 ``circ_mv``。脚本同时输出完整精确市值请求身份。

控制票次日无开盘或开盘逼近涨停时保持现金且不补票；入场后停牌按 LOCF 持价；
期限前终止而无结算数据时，同时输出悲观 -100% 和乐观 LOCF 边界。

运行：``uv run --no-sync python scripts/delay5_common_horizon_att.py``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import _s2c_core as core
import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as proxy
import surge_live as live
import surge_portfolio_backtest as portfolio
import trend_regime as tr

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "candidates.parquet"
PANEL_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"
MARKET_PATH = SCRIPT_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
PRODUCTION_DIR = SCRIPT_DIR / "_output" / "surge_delay5_production_cohort"
PRODUCTION_COHORT_PATH = PRODUCTION_DIR / "cohort.parquet"
PRODUCTION_AUDIT_PATH = PRODUCTION_DIR / "audit.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "delay5_common_horizon_att"
TREATED_PATH = OUTPUT_DIR / "treated_common_support.parquet"
PAIRS_PATH = OUTPUT_DIR / "matched_pairs.parquet"
TRADE_ATT_PATH = OUTPUT_DIR / "trade_att.parquet"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
EXACT_REQUEST_PATH = OUTPUT_DIR / "exact_mcap_request_manifest.json"
EXACT_CACHE_PATH = SCRIPT_DIR / "_output" / "s2b_top_winner_exact_mcap" / "baostock_exact_mcap.parquet"

HORIZONS = (5, 20, 60)
COMMON_HORIZON = max(HORIZONS)
AMOUNT_K, AMOUNT_MIN_VALID = 50, 10
PROXY_K, PROXY_MIN_VALID, PROXY_CALIPER_RATIO = 10, 5, 1.5
N_BOOT, BLOCK_LEN = 10_000, 10
SPEC_AMOUNT = "amount_k50"
SPEC_PROXY = "industry_mcap_proxy_k10"
SPEC_MIN_VALID = {SPEC_AMOUNT: AMOUNT_MIN_VALID, SPEC_PROXY: PROXY_MIN_VALID}
DATE_COLUMNS = ("sig_dt", "dec_dt", "entry_dt", "exit_dt")
MARKET_COLUMNS = ("high20_ratio", "ew_index_above_ma20")


def _normalise_dates(values: Iterable[Any]) -> pd.DatetimeIndex:
    """返回严格递增、无重复、归一到午夜的交易日历。"""

    calendar = pd.DatetimeIndex(pd.to_datetime(list(values))).normalize()
    if calendar.has_duplicates:
        raise ValueError("calendar contains duplicate sessions")
    if not calendar.is_monotonic_increasing:
        calendar = calendar.sort_values()
    if calendar.empty:
        raise ValueError("calendar is empty")
    return calendar


def horizon_endpoints(
    calendar: Sequence[Any] | pd.DatetimeIndex,
    entry_dt: Any,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, pd.Timestamp]:
    """固定期限终点；入场日是第 1 日，所以 H 日终点偏移为 ``H-1``。"""

    sessions = _normalise_dates(calendar)
    entry = pd.Timestamp(entry_dt).normalize()
    position = int(sessions.get_indexer([entry])[0])
    if position < 0:
        raise ValueError(f"entry date is not an official session: {entry.date()}")
    result: dict[int, pd.Timestamp] = {}
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        end = position + int(horizon) - 1
        if end >= len(sessions):
            raise ValueError(f"horizon {horizon} is not mature for entry {entry.date()}")
        result[int(horizon)] = pd.Timestamp(sessions[end])
    return result


def add_common_horizon_schedule(
    frame: pd.DataFrame,
    calendar: Sequence[Any] | pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把已成交 cohort 拆成 60 日成熟集和右侧未成熟集。"""

    sessions = _normalise_dates(calendar)
    positions = pd.Series(np.arange(len(sessions), dtype=np.int32), index=sessions)
    scoped = frame.copy()
    scoped["entry_pos"] = pd.to_datetime(scoped["entry_dt"]).dt.normalize().map(positions)
    scoped["common_mature"] = scoped["entry_pos"].notna() & scoped["entry_pos"].le(len(sessions) - COMMON_HORIZON)
    for horizon in HORIZONS:
        scoped[f"h{horizon}_dt"] = [
            sessions[int(pos) + horizon - 1] if pd.notna(pos) and int(pos) + horizon - 1 < len(sessions) else pd.NaT
            for pos in scoped["entry_pos"]
        ]
    return scoped[scoped["common_mature"]].copy(), scoped[~scoped["common_mature"]].copy()


def signal_gate(frame: pd.DataFrame) -> pd.Series:
    """当前 Rust/live 默认 anticipate 信号门；不再使用 ``above_zg``。"""

    return (
        frame["sig_vol_ratio"].le(tr.SURGE_GATE_VOL_RATIO)
        & frame["sig_ma_spread_pct"].ge(tr.SURGE_GATE_MA_SPREAD)
        & frame["sig_ret20"].ge(tr.SURGE_GATE_RET20)
    ).fillna(False)


def net_return_pct(
    gross_pct: float,
    *,
    filled: bool,
    buy_cost: float = portfolio.BUY_COST,
    sell_cost: float = portfolio.SELL_COST,
) -> float:
    """对真实成交腿计买卖成本；现金路径收益和成本都为 0。"""

    if not filled:
        return 0.0
    return ((1 + float(gross_pct) / 100) * (1 - sell_cost) / (1 + buy_cost) - 1) * 100


def prepare_decision_pool(
    visible: pd.DataFrame,
    *,
    treated_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造决策日可见全集和控制池；十分位在任何剔除动作之前计算。"""

    required = {"symbol", "close", "amount_e", "industry", "mcap_proxy", "is_st"}
    if missing := required - set(visible):
        raise ValueError(f"decision snapshot missing columns: {sorted(missing)}")
    if visible["symbol"].duplicated().any():
        raise ValueError("decision snapshot contains duplicate symbols")
    full = visible.copy()
    finite = (
        np.isfinite(pd.to_numeric(full["close"], errors="coerce"))
        & np.isfinite(pd.to_numeric(full["amount_e"], errors="coerce"))
        & full["close"].gt(0)
        & full["amount_e"].gt(0)
    )
    full = full[finite].copy()
    full["amount_decile"] = np.floor(full["amount_e"].rank(pct=True).mul(10).clip(upper=9.999)).astype(int)
    full = full.sort_values("symbol", kind="mergesort").set_index("symbol", drop=False)
    full.index.name = None
    pool = full[~full.index.isin(treated_symbols) & ~full["is_st"].astype(bool)].copy()
    return full, pool


def nearest_controls(
    pool: pd.DataFrame,
    *,
    metric: str,
    target: float,
    k: int,
    caliper_ratio: float | None = None,
) -> pd.DataFrame:
    """按 log 距离取确定性最近邻；symbol 是显式 tie-break。"""

    if metric not in pool:
        raise ValueError(f"matching metric is absent: {metric}")
    if not np.isfinite(target) or target <= 0:
        return pool.iloc[:0].assign(match_distance=pd.Series(dtype=float))
    values = pd.to_numeric(pool[metric], errors="coerce")
    eligible = pool[np.isfinite(values) & values.gt(0)].copy()
    if caliper_ratio is not None:
        if caliper_ratio <= 1:
            raise ValueError("caliper_ratio must be greater than one")
        eligible = eligible[eligible[metric].between(target / caliper_ratio, target * caliper_ratio)].copy()
    eligible["match_distance"] = (np.log(eligible[metric]) - math.log(target)).abs()
    return eligible.sort_values(["match_distance", "symbol"], kind="mergesort").head(k).copy()


def match_amount_controls(
    full: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    treated_symbol: str,
    treated_amount: float,
    k: int = AMOUNT_K,
) -> tuple[pd.DataFrame, int]:
    """同决策日成交额十分位内的最近 log-amount 控制。"""

    if treated_symbol not in full.index:
        raise ValueError(f"treated symbol is not decision-visible: {treated_symbol}")
    decile = int(full.at[treated_symbol, "amount_decile"])
    eligible = pool[pool["amount_decile"].eq(decile)].copy()
    return nearest_controls(eligible, metric="amount_e", target=treated_amount, k=k), int(len(eligible))


def match_proxy_controls(
    pool: pd.DataFrame,
    *,
    treated_industry: Any,
    treated_mcap_proxy: float,
    k: int = PROXY_K,
    caliper_ratio: float = PROXY_CALIPER_RATIO,
) -> tuple[pd.DataFrame, int]:
    """同行业、当前股本规模代理最近邻；行业缺失时 fail closed。"""

    if treated_industry is None or pd.isna(treated_industry) or not str(treated_industry).strip():
        return pool.iloc[:0].assign(match_distance=pd.Series(dtype=float)), 0
    eligible = pool[pool["industry"].eq(treated_industry)].copy()
    matched = nearest_controls(
        eligible,
        metric="mcap_proxy",
        target=treated_mcap_proxy,
        k=k,
        caliper_ratio=caliper_ratio,
    )
    metrics = pd.to_numeric(eligible["mcap_proxy"], errors="coerce")
    valid = metrics[np.isfinite(metrics) & metrics.gt(0)]
    in_caliper = valid.between(treated_mcap_proxy / caliper_ratio, treated_mcap_proxy * caliper_ratio)
    return matched, int(in_caliper.sum())


def evaluate_control_path(
    *,
    decision_close: float,
    entry_open: float,
    horizon_marks: Mapping[int, float],
    terminal_by_horizon: Mapping[int, bool],
    limit_pct: float,
    gap_margin_pct: float = portfolio.GAP_LIMIT_MARGIN,
) -> dict[str, Any]:
    """评估预先匹配控制票；缺开盘/涨停不可买均保持现金且不补票。"""

    no_open = not np.isfinite(decision_close) or decision_close <= 0 or not np.isfinite(entry_open) or entry_open <= 0
    if no_open:
        status, filled, gap_pct = "cash_no_open", False, None
    else:
        gap_pct = (float(entry_open) / float(decision_close) - 1) * 100
        filled = bool(gap_pct < float(limit_pct) - float(gap_margin_pct))
        status = "filled" if filled else "cash_gap_abandoned"
    result: dict[str, Any] = {"entry_status": status, "control_filled": filled, "gap_pct": gap_pct}
    for horizon in HORIZONS:
        terminal = bool(terminal_by_horizon.get(horizon, False))
        result[f"terminal_h{horizon}"] = terminal
        if not filled:
            locf_gross = pessimistic_gross = optimistic_gross = 0.0
        else:
            mark = float(horizon_marks.get(horizon, np.nan))
            if not np.isfinite(mark) or mark <= 0:
                if not terminal:
                    raise ValueError(f"non-terminal filled control lacks h{horizon} mark")
                locf_gross = -100.0
            else:
                locf_gross = (mark / float(entry_open) - 1) * 100
            pessimistic_gross = -100.0 if terminal else locf_gross
            optimistic_gross = locf_gross
        for label, gross in (
            ("gross", locf_gross),
            ("gross_pessimistic", pessimistic_gross),
            ("gross_optimistic", optimistic_gross),
        ):
            result[f"{label}_h{horizon}_pct"] = gross
            result[f"{label.replace('gross', 'net')}_h{horizon}_pct"] = net_return_pct(gross, filled=filled)
    return result


def _year_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    years = pd.to_datetime(frame["dec_dt"]).dt.year
    return {str(int(year)): int(count) for year, count in frame.groupby(years, sort=True).size().items()}


def _stage(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "decision_dates": int(pd.to_datetime(frame["dec_dt"]).nunique()) if len(frame) else 0,
        "year_counts": _year_counts(frame),
    }


def build_production_delay5_cohort(
    candidates: pd.DataFrame,
    market_state: pd.DataFrame,
    calendar: Sequence[Any] | pd.DatetimeIndex,
    st_intervals: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """复原 raw→gate→hard→ST→market→fill→60-day-mature 漏斗。"""

    if not st_intervals:
        raise RuntimeError("historical ST intervals are unavailable; production cohort must fail closed")
    scoped = candidates.copy()
    for column in DATE_COLUMNS:
        if column in scoped:
            scoped[column] = pd.to_datetime(scoped[column]).dt.normalize()
    raw = scoped[scoped["mode"].eq("anticipate") & scoped["delay"].eq(live.EXP_DELAY)].copy()
    if raw.duplicated(["symbol", "dec_dt"]).any():
        raise RuntimeError("delay5 candidate identity is not unique by symbol and decision date")
    gated = raw[signal_gate(raw)].copy()
    hard = gated[
        gated["amount_e"].ge(portfolio.MIN_AMOUNT_E)
        & gated["sl_pct"].between(portfolio.STOP_MIN_PCT, portfolio.STOP_MAX_PCT)
    ].copy()
    hard["is_st"] = [
        portfolio.is_st_on(st_intervals, symbol, dec_dt)
        for symbol, dec_dt in zip(hard["symbol"], hard["dec_dt"], strict=False)
    ]
    non_st = hard[~hard["is_st"]].copy()
    state = market_state.copy()
    state["dt"] = pd.to_datetime(state["dt"]).dt.normalize()
    merged = non_st.merge(
        state[["dt", *MARKET_COLUMNS]].rename(columns={"dt": "dec_dt"}),
        on="dec_dt",
        how="left",
        validate="many_to_one",
    )
    if merged[list(MARKET_COLUMNS)].isna().any(axis=1).any():
        raise RuntimeError("production candidates have missing causal market-state rows")
    market = merged[merged["high20_ratio"].gt(live.MARKET_GATE_HIGH20) & merged["ew_index_above_ma20"].gt(0)].copy()
    sessions = _normalise_dates(calendar)
    next_session = pd.Series(sessions[1:], index=sessions[:-1])
    market["next_market_dt"] = market["dec_dt"].map(next_session)
    next_day = market[market["entry_dt"].eq(market["next_market_dt"])].copy()
    filled = next_day[next_day["gap_pct"].lt(next_day["limit_pct"] - portfolio.GAP_LIMIT_MARGIN)].copy()
    mature, immature = add_common_horizon_schedule(filled, sessions)
    mature = mature.sort_values(["dec_dt", "symbol", "entry_dt"], kind="mergesort").reset_index(drop=True)
    filled = filled.sort_values(["dec_dt", "symbol", "entry_dt"], kind="mergesort").reset_index(drop=True)
    funnel = {
        "raw": _stage(raw),
        "signal_gate": _stage(gated),
        "hard_amount_stop": _stage(hard),
        "historical_non_st": _stage(non_st),
        "market_gate": _stage(market),
        "strict_next_market_session": _stage(next_day),
        "gap_filled": _stage(filled),
        "common_60_session_mature": _stage(mature),
        "right_tail_immature": _stage(immature),
    }
    return mature, filled, funnel


def load_production_common_support(
    calendar: Sequence[Any] | pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """加载并验证上游 production-cohort 审计，再构造 60 日共同成熟支持。"""

    for path, label in (
        (PRODUCTION_COHORT_PATH, "production cohort"),
        (PRODUCTION_AUDIT_PATH, "production cohort audit"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required {label} is missing: {path}")
    upstream = json.loads(PRODUCTION_AUDIT_PATH.read_text(encoding="utf-8"))
    if upstream.get("schema") != "surge_delay5_production_cohort_audit_v2":
        raise RuntimeError("unexpected production cohort audit schema")
    expected_output = upstream.get("outputs", {}).get("cohort", {})
    if expected_output.get("sha256") != proxy.sha256_file(PRODUCTION_COHORT_PATH):
        raise RuntimeError("production cohort parquet no longer matches its audit identity")
    bound_inputs = {
        "candidates": CANDIDATES_PATH,
        "panel": PANEL_PATH,
        "market_state": MARKET_PATH,
        "historical_st": portfolio.NAMECHANGE_PATH,
    }
    for label, path in bound_inputs.items():
        expected = upstream.get("inputs", {}).get(label, {}).get("sha256")
        if not path.is_file() or expected != proxy.sha256_file(path):
            raise RuntimeError(f"production cohort audit has stale {label} binding")

    cohort = pd.read_parquet(PRODUCTION_COHORT_PATH)
    if len(cohort) != int(expected_output.get("rows", -1)):
        raise RuntimeError("production cohort row count differs from its audit")
    required = {
        "stage_raw",
        "stage_gate",
        "stage_hard",
        "stage_st",
        "stage_market",
        "stage_fill",
        "stage_mature",
        "state_fill_dt",
        "state_fill_open",
        "state_fill_observed",
    }
    if missing := required - set(cohort):
        raise RuntimeError(f"production cohort lacks stage columns: {sorted(missing)}")
    for column in DATE_COLUMNS:
        if column in cohort:
            cohort[column] = pd.to_datetime(cohort[column]).dt.normalize()
    sessions = _normalise_dates(calendar)
    next_session = pd.Series(sessions[1:], index=sessions[:-1])
    filled = cohort[cohort["stage_fill"]].copy()
    filled["next_market_dt"] = filled["dec_dt"].map(next_session)
    strict = filled[filled["entry_dt"].eq(filled["next_market_dt"])].copy()
    mature, immature = add_common_horizon_schedule(strict, sessions)
    mature = mature.sort_values(["dec_dt", "symbol", "entry_dt"], kind="mergesort").reset_index(drop=True)
    strict = strict.sort_values(["dec_dt", "symbol", "entry_dt"], kind="mergesort").reset_index(drop=True)
    stage_map = {
        "raw": "stage_raw",
        "signal_gate": "stage_gate",
        "hard_amount_stop": "stage_hard",
        "historical_non_st": "stage_st",
        "market_gate": "stage_market",
        "gap_filled": "stage_fill",
        "production_full_mature": "stage_mature",
    }
    funnel = {name: _stage(cohort[cohort[column]]) for name, column in stage_map.items()}
    funnel["strict_next_market_session"] = _stage(strict)
    funnel["common_60_session_mature"] = _stage(mature)
    funnel["right_tail_immature"] = _stage(immature)
    return mature, strict, funnel, upstream


def _wide_panel(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    scoped = panel.copy()
    scoped["dt"] = pd.to_datetime(scoped["dt"]).dt.normalize()
    if scoped.duplicated(["dt", "symbol"]).any():
        raise RuntimeError("panel contains duplicate symbol/date rows")
    open_w = scoped.pivot(index="dt", columns="symbol", values="open").reindex(calendar)
    close_w = scoped.pivot(index="dt", columns="symbol", values="close").reindex(calendar)
    amount_w = scoped.pivot(index="dt", columns="symbol", values="amount_e").reindex(calendar)
    return open_w, close_w, amount_w, close_w.ffill(), scoped.groupby("symbol", sort=False)["dt"].max()


def add_treated_outcomes(
    treated: pd.DataFrame,
    *,
    open_w: pd.DataFrame,
    close_w: pd.DataFrame,
    mark_w: pd.DataFrame,
    last_dt: pd.Series,
) -> pd.DataFrame:
    """给共同支持处理集附加三期限 LOCF、终止上下界及毛/净收益。"""

    result = treated.copy()
    result["entry_open"] = [float(open_w.at[row.entry_dt, row.symbol]) for row in result.itertuples()]
    if (~np.isfinite(result["entry_open"]) | result["entry_open"].le(0)).any():
        raise RuntimeError("filled treated cohort contains invalid entry open")
    result["candidate_entry_open_abs_diff"] = (result["entry_open"] - result["entry_price"]).abs()
    for horizon in HORIZONS:
        gross: list[float] = []
        pessimistic: list[float] = []
        exact_missing: list[bool] = []
        terminal_flags: list[bool] = []
        for row in result.itertuples():
            endpoint = getattr(row, f"h{horizon}_dt")
            mark = mark_w.at[endpoint, row.symbol]
            terminal = bool(last_dt.get(row.symbol, pd.NaT) < endpoint)
            if not np.isfinite(mark) or mark <= 0:
                raise RuntimeError(f"treated trade lacks LOCF mark: {row.symbol} {endpoint}")
            value = (float(mark) / float(row.entry_open) - 1) * 100
            gross.append(value)
            pessimistic.append(-100.0 if terminal else value)
            exact_missing.append(bool(pd.isna(close_w.at[endpoint, row.symbol])))
            terminal_flags.append(terminal)
        result[f"gross_h{horizon}_pct"] = gross
        result[f"gross_pessimistic_h{horizon}_pct"] = pessimistic
        result[f"gross_optimistic_h{horizon}_pct"] = gross
        result[f"net_h{horizon}_pct"] = [net_return_pct(value, filled=True) for value in gross]
        result[f"net_pessimistic_h{horizon}_pct"] = [net_return_pct(value, filled=True) for value in pessimistic]
        result[f"net_optimistic_h{horizon}_pct"] = result[f"net_h{horizon}_pct"]
        result[f"endpoint_exact_missing_h{horizon}"] = exact_missing
        result[f"terminal_h{horizon}"] = terminal_flags
    result["trade_id"] = result["symbol"] + "|" + result["dec_dt"].dt.strftime("%Y-%m-%d")
    return result


def _load_industry_maps(years: Iterable[int]) -> tuple[dict[int, pd.Series], dict[str, Any]]:
    maps: dict[int, pd.Series] = {}
    identities: dict[str, Any] = {}
    for year in sorted({int(value) for value in years}):
        path = proxy._industry_path(year)
        if not path.exists():
            raise FileNotFoundError(f"missing frozen industry snapshot: {path}")
        frame = pd.read_parquet(path)
        maps[year] = frame.set_index("symbol")["industry"]
        identities[str(year)] = {
            "path": str(path),
            "rows": int(len(frame)),
            "requested_anchor": proxy.INDUSTRY_ANCHORS[year],
            "effective_dates": sorted(frame["updateDate"].astype(str).unique().tolist()),
            "canonical_content_sha256": proxy.canonical_industry_sha256(frame),
            "file_sha256": proxy.sha256_file(path),
        }
    return maps, identities


def _decision_snapshot(
    dec_dt: pd.Timestamp,
    *,
    close_w: pd.DataFrame,
    amount_w: pd.DataFrame,
    industry: pd.Series,
    shares: pd.Series,
    st_intervals: Mapping[str, Any],
    treated_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = close_w.columns
    visible = pd.DataFrame(
        {
            "symbol": symbols,
            "close": close_w.loc[dec_dt].to_numpy(dtype=float),
            "amount_e": amount_w.loc[dec_dt].to_numpy(dtype=float),
        }
    )
    visible["industry"] = visible["symbol"].map(industry)
    visible["mcap_proxy"] = visible["close"] * visible["symbol"].map(shares)
    visible["is_st"] = [portfolio.is_st_on(st_intervals, symbol, dec_dt) for symbol in visible["symbol"]]
    return prepare_decision_pool(visible, treated_symbols=treated_symbols)


def _control_path_from_wide(
    symbol: str,
    treated_row: Any,
    *,
    open_w: pd.DataFrame,
    close_w: pd.DataFrame,
    mark_w: pd.DataFrame,
    last_dt: pd.Series,
) -> dict[str, Any]:
    marks = {h: mark_w.at[getattr(treated_row, f"h{h}_dt"), symbol] for h in HORIZONS}
    terminal = {h: bool(last_dt.get(symbol, pd.NaT) < getattr(treated_row, f"h{h}_dt")) for h in HORIZONS}
    return evaluate_control_path(
        decision_close=float(close_w.at[treated_row.dec_dt, symbol]),
        entry_open=float(open_w.at[treated_row.entry_dt, symbol]),
        horizon_marks=marks,
        terminal_by_horizon=terminal,
        limit_pct=tr.limit_pct_for(symbol),
    )


def _canonical_payload(records: list[Any]) -> bytes:
    return json.dumps(sorted(records), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _payload_identity(records: list[Any]) -> dict[str, Any]:
    ordered = sorted(records)
    payload = _canonical_payload(ordered)
    return {
        "count": int(len(ordered)),
        "bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first3": ordered[:3],
        "last3": ordered[-3:] if ordered else [],
    }


def _exact_cache_coverage(treated: pd.DataFrame, request_records: list[list[str]]) -> dict[str, Any]:
    if not EXACT_CACHE_PATH.exists():
        return {"available": False, "treated_valid": 0, "request_valid": 0}
    cache = pd.read_parquet(EXACT_CACHE_PATH)
    cache["dec_dt"] = pd.to_datetime(cache["dec_dt"]).dt.normalize()
    valid = cache[cache["exact_circ_mv"].notna()]
    valid_keys = set(zip(valid["symbol"], valid["dec_dt"].dt.strftime("%Y-%m-%d"), strict=False))
    treated_keys = set(zip(treated["symbol"], treated["dec_dt"].dt.strftime("%Y-%m-%d"), strict=False))
    request_keys = {(symbol, date_text) for symbol, date_text in request_records}
    return {
        "available": True,
        "path": str(EXACT_CACHE_PATH),
        "sha256": proxy.sha256_file(EXACT_CACHE_PATH),
        "rows": int(len(cache)),
        "valid_rows": int(len(valid)),
        "decision_dates": int(cache["dec_dt"].nunique()),
        "treated_valid": int(len(treated_keys & valid_keys)),
        "request_valid": int(len(request_keys & valid_keys)),
    }


def add_exact_mcap_request_pairs(
    request_pairs: set[tuple[str, str]],
    *,
    treated_symbol: str,
    dec_dt: pd.Timestamp,
    treated_industry: Any,
    pool: pd.DataFrame,
) -> None:
    """加入点时市值请求；treated 永远保留，行业缺失只关闭控制匹配。"""

    date_text = pd.Timestamp(dec_dt).strftime("%Y-%m-%d")
    request_pairs.add((str(treated_symbol), date_text))
    if treated_industry is None or pd.isna(treated_industry) or not str(treated_industry).strip():
        return
    controls = pool.index[pool["industry"].eq(treated_industry)].tolist()
    request_pairs.update((str(symbol), date_text) for symbol in controls)


def build_matches(
    treated: pd.DataFrame,
    all_filled: pd.DataFrame,
    *,
    open_w: pd.DataFrame,
    close_w: pd.DataFrame,
    amount_w: pd.DataFrame,
    mark_w: pd.DataFrame,
    last_dt: pd.Series,
    industry_maps: Mapping[int, pd.Series],
    shares: pd.Series,
    st_intervals: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[list[str]]]:
    """构建两套冻结匹配、评估控制路径并返回精确市值请求记录。"""

    treated_by_date = {
        pd.Timestamp(dec_dt): set(group["symbol"].astype(str)) for dec_dt, group in all_filled.groupby("dec_dt")
    }
    pairs: list[dict[str, Any]] = []
    request_pairs: set[tuple[str, str]] = set()
    missing_industry: list[dict[str, str]] = []
    pool_sizes: dict[str, list[int]] = {SPEC_AMOUNT: [], SPEC_PROXY: []}
    for dec_dt, day_trades in treated.groupby("dec_dt", sort=True):
        year = int(pd.Timestamp(dec_dt).year)
        industry_map = industry_maps[year]
        full, pool = _decision_snapshot(
            pd.Timestamp(dec_dt),
            close_w=close_w,
            amount_w=amount_w,
            industry=industry_map,
            shares=shares,
            st_intervals=st_intervals,
            treated_symbols=treated_by_date[pd.Timestamp(dec_dt)],
        )
        for row in day_trades.itertuples(index=False):
            treated_industry = industry_map.get(row.symbol, np.nan)
            add_exact_mcap_request_pairs(
                request_pairs,
                treated_symbol=row.symbol,
                dec_dt=row.dec_dt,
                treated_industry=treated_industry,
                pool=pool,
            )
            treated_proxy = float(close_w.at[row.dec_dt, row.symbol] * shares.get(row.symbol, np.nan))
            amount_matches, amount_pool_n = match_amount_controls(
                full, pool, treated_symbol=row.symbol, treated_amount=float(row.amount_e)
            )
            proxy_matches, proxy_pool_n = match_proxy_controls(
                pool, treated_industry=treated_industry, treated_mcap_proxy=treated_proxy
            )
            pool_sizes[SPEC_AMOUNT].append(amount_pool_n)
            pool_sizes[SPEC_PROXY].append(proxy_pool_n)
            if treated_industry is None or pd.isna(treated_industry) or not str(treated_industry).strip():
                missing_industry.append({"symbol": row.symbol, "dec_dt": row.dec_dt.strftime("%Y-%m-%d")})
            specifications = (
                (SPEC_AMOUNT, amount_matches, amount_pool_n, float(row.amount_e), "amount_e"),
                (SPEC_PROXY, proxy_matches, proxy_pool_n, treated_proxy, "mcap_proxy"),
            )
            for specification, matches, eligible_n, treated_metric, metric in specifications:
                support_ok = len(matches) >= SPEC_MIN_VALID[specification]
                for rank, control in enumerate(matches.itertuples(), start=1):
                    control_metric = float(getattr(control, metric))
                    record = {
                        "trade_id": row.trade_id,
                        "specification": specification,
                        "support_ok": support_ok,
                        "treated_symbol": row.symbol,
                        "control_symbol": control.symbol,
                        "dec_dt": row.dec_dt,
                        "entry_dt": row.entry_dt,
                        **{f"h{h}_dt": getattr(row, f"h{h}_dt") for h in HORIZONS},
                        "year": int(row.dec_dt.year),
                        "rank": rank,
                        "eligible_pool_n": int(eligible_n),
                        "selected_controls_n": int(len(matches)),
                        "treated_industry": None if pd.isna(treated_industry) else str(treated_industry),
                        "treated_metric": treated_metric,
                        "control_metric": control_metric,
                        "metric_ratio": control_metric / treated_metric if treated_metric > 0 else np.nan,
                        "match_distance": float(control.match_distance),
                    }
                    record.update(
                        _control_path_from_wide(
                            control.symbol,
                            row,
                            open_w=open_w,
                            close_w=close_w,
                            mark_w=mark_w,
                            last_dt=last_dt,
                        )
                    )
                    pairs.append(record)
    pair_frame = pd.DataFrame(pairs).sort_values(
        ["specification", "dec_dt", "treated_symbol", "rank"], kind="mergesort"
    )
    diagnostics = {
        "missing_treated_industry": missing_industry,
        "eligible_pool": {
            spec: {
                "p05": round(float(np.percentile(values, 5)), 3),
                "median": round(float(np.median(values)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
            }
            for spec, values in pool_sizes.items()
        },
    }
    return pair_frame, diagnostics, [list(pair) for pair in sorted(request_pairs)]


def build_trade_att(treated: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """把控制路径聚合为逐处理票毛/净 ATT 及终止上下界。"""

    treated_by_id = treated.set_index("trade_id", drop=False)
    rows: list[dict[str, Any]] = []
    supported = pairs[pairs["support_ok"]]
    for (specification, trade_id), group in supported.groupby(["specification", "trade_id"], sort=True):
        treated_row = treated_by_id.loc[trade_id]
        record: dict[str, Any] = {
            "trade_id": trade_id,
            "specification": specification,
            "symbol": treated_row["symbol"],
            "dec_dt": treated_row["dec_dt"],
            "entry_dt": treated_row["entry_dt"],
            "year": int(treated_row["dec_dt"].year),
            "n_controls": int(len(group)),
            "control_filled_n": int(group["control_filled"].sum()),
            "control_cash_n": int((~group["control_filled"]).sum()),
        }
        for horizon in HORIZONS:
            record[f"control_terminal_h{horizon}_n"] = int(group[f"terminal_h{horizon}"].sum())
            for basis in ("gross", "net"):
                treated_base = float(treated_row[f"{basis}_h{horizon}_pct"])
                treated_pessimistic = float(treated_row[f"{basis}_pessimistic_h{horizon}_pct"])
                treated_optimistic = float(treated_row[f"{basis}_optimistic_h{horizon}_pct"])
                control_base = float(group[f"{basis}_h{horizon}_pct"].median())
                control_pessimistic = float(group[f"{basis}_pessimistic_h{horizon}_pct"].median())
                control_optimistic = float(group[f"{basis}_optimistic_h{horizon}_pct"].median())
                record[f"treated_{basis}_h{horizon}_pct"] = treated_base
                record[f"control_median_{basis}_h{horizon}_pct"] = control_base
                record[f"att_{basis}_h{horizon}_pct"] = treated_base - control_base
                record[f"att_{basis}_h{horizon}_lower_pct"] = treated_pessimistic - control_optimistic
                record[f"att_{basis}_h{horizon}_upper_pct"] = treated_optimistic - control_pessimistic
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["specification", "dec_dt", "symbol"], kind="mergesort")


def summarize_att(frame: pd.DataFrame, column: str, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """按决策日先聚类均值，再做 HAC 与平稳分块 bootstrap。"""

    valid = frame.dropna(subset=[column])
    if valid.empty:
        return {"n_trades": 0, "n_decision_dates": 0}
    values = valid[column].astype(float)
    daily = valid.groupby("dec_dt", sort=True)[column].mean().astype(float)
    return {
        "n_trades": int(len(valid)),
        "n_decision_dates": int(len(daily)),
        "mean_att_pct": round(float(values.mean()), 3),
        "median_att_pct": round(float(values.median()), 3),
        "positive_att_pct": round(float(values.gt(0).mean() * 100), 1),
        "daily_hac": core.newey_west_tstat(daily.to_numpy()),
        "daily_block_bootstrap": core.block_bootstrap_mean(daily.to_numpy(), n_boot=n_boot, block_len=BLOCK_LEN),
    }


def _nominal_gate(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("daily_hac", {}).get("t_stat", -np.inf) >= 2
        and summary.get("daily_block_bootstrap", {}).get("ci_lo", -np.inf) > 0
    )


def build_summaries(trade_att: pd.DataFrame, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """报告全样本、2024 前、2024+ 与逐年共同支持结果。"""

    summaries: dict[str, Any] = {}
    for specification, spec_frame in trade_att.groupby("specification", sort=True):
        scopes: list[tuple[str, pd.DataFrame]] = [
            ("all", spec_frame),
            ("pre_2024", spec_frame[spec_frame["year"].le(2023)]),
            ("2024plus", spec_frame[spec_frame["year"].ge(2024)]),
        ]
        scopes.extend((str(int(year)), group) for year, group in spec_frame.groupby("year", sort=True))
        spec_result: dict[str, Any] = {}
        for name, scope in scopes:
            horizon_result: dict[str, Any] = {}
            for horizon in HORIZONS:
                gross = summarize_att(scope, f"att_gross_h{horizon}_pct", n_boot=n_boot)
                net = summarize_att(scope, f"att_net_h{horizon}_pct", n_boot=n_boot)
                horizon_result[str(horizon)] = {
                    "gross": gross,
                    "net": net,
                    "gross_nominal_gate": _nominal_gate(gross),
                    "net_nominal_gate": _nominal_gate(net),
                    "gross_bound_mean_pct": {
                        "lower": round(float(scope[f"att_gross_h{horizon}_lower_pct"].mean()), 3)
                        if len(scope)
                        else None,
                        "upper": round(float(scope[f"att_gross_h{horizon}_upper_pct"].mean()), 3)
                        if len(scope)
                        else None,
                    },
                }
            spec_result[name] = {
                "n_trades": int(len(scope)),
                "n_decision_dates": int(scope["dec_dt"].nunique()) if len(scope) else 0,
                "horizons": horizon_result,
            }
        summaries[str(specification)] = spec_result
    return summaries


def _support_summary(treated: pd.DataFrame, trade_att: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for specification in (SPEC_AMOUNT, SPEC_PROXY):
        supported = trade_att[trade_att["specification"].eq(specification)]
        selected = pairs[pairs["specification"].eq(specification)]
        result[specification] = {
            "source_trades": int(len(treated)),
            "supported_trades": int(len(supported)),
            "support_rate_pct": round(float(len(supported) / len(treated) * 100), 2),
            "selected_pairs": int(len(selected)),
            "control_cash_pairs": int((~selected["control_filled"]).sum()),
            "control_terminal_h60_pairs": int(selected["terminal_h60"].sum()),
            "year_support": {
                str(int(year)): {
                    "source": int(len(source)),
                    "supported": int(supported["year"].eq(year).sum()),
                    "support_rate_pct": round(float(supported["year"].eq(year).sum() / len(source) * 100), 2),
                }
                for year, source in treated.groupby(treated["dec_dt"].dt.year, sort=True)
            },
        }
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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def run_audit(*, n_boot: int = N_BOOT, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """运行完整审计、写入 ignored artifacts 并返回 JSON 对象。"""

    started = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    market_state = pd.read_parquet(MARKET_PATH)
    calendar = _normalise_dates(market_state["dt"])
    st_intervals = portfolio.load_st_intervals()
    if not st_intervals:
        raise RuntimeError("historical ST intervals are unavailable")
    treated, all_filled, funnel, production_audit = load_production_common_support(calendar)
    open_w, close_w, amount_w, mark_w, last_dt = _wide_panel(panel, calendar)
    treated = add_treated_outcomes(treated, open_w=open_w, close_w=close_w, mark_w=mark_w, last_dt=last_dt)
    industry_maps, industry_identities = _load_industry_maps(treated["dec_dt"].dt.year.unique())
    if not proxy.CAPITAL_PATH.exists():
        raise FileNotFoundError(f"missing current-capital proxy snapshot: {proxy.CAPITAL_PATH}")
    capital = pd.read_parquet(proxy.CAPITAL_PATH)
    shares = capital.set_index("symbol")["current_float_shares"]
    pairs, match_diagnostics, request_records = build_matches(
        treated,
        all_filled,
        open_w=open_w,
        close_w=close_w,
        amount_w=amount_w,
        mark_w=mark_w,
        last_dt=last_dt,
        industry_maps=industry_maps,
        shares=shares,
        st_intervals=st_intervals,
    )
    trade_att = build_trade_att(treated, pairs)
    summaries = build_summaries(trade_att, n_boot=n_boot)
    support = _support_summary(treated, trade_att, pairs)
    request_identity = _payload_identity(request_records)
    treated_request_records = sorted(
        {
            (str(row.symbol), pd.Timestamp(row.dec_dt).strftime("%Y-%m-%d"))
            for row in treated[["symbol", "dec_dt"]].itertuples(index=False)
        }
    )
    request_keys = {tuple(record) for record in request_records}
    treated_request_keys = set(treated_request_records)
    if len(treated_request_records) != len(treated):
        raise RuntimeError("treated exact-mcap request keys are not unique")
    if not treated_request_keys.issubset(request_keys):
        raise RuntimeError("exact-mcap request does not contain every treated key")
    request_dates = sorted({date_text for _, date_text in request_keys})
    request_symbols = sorted({symbol for symbol, _ in request_keys})
    exact_manifest = {
        "schema": "delay5_exact_mcap_request_manifest_v2",
        "canonicalization": (
            "JSON array of [symbol, YYYY-MM-DD] records; lexicographic sort; ensure_ascii=False; "
            "separators=(',', ':'); UTF-8; no trailing newline"
        ),
        "cohort": {
            "treated_trades": int(len(treated)),
            "decision_dates": int(treated["dec_dt"].nunique()),
            "year_counts": _year_counts(treated),
            "missing_frozen_industry": match_diagnostics["missing_treated_industry"],
        },
        "request_identity": request_identity,
        "request_records": sorted(request_records),
        "treated_request_identity": _payload_identity([list(record) for record in treated_request_records]),
        "treated_request_records": [list(record) for record in treated_request_records],
        "request_dates_identity": _payload_identity(request_dates),
        "request_dates": request_dates,
        "request_symbols_identity": _payload_identity(request_symbols),
        "request_symbols": request_symbols,
        "closure": {
            "request_keys_unique": len(request_keys) == len(request_records),
            "treated_keys_unique": len(treated_request_keys) == len(treated_request_records),
            "all_treated_requested": treated_request_keys.issubset(request_keys),
        },
        "local_exact_cache": _exact_cache_coverage(treated, request_records),
    }
    proxy_2024plus = summaries[SPEC_PROXY]["2024plus"]
    proxy_confirmed = any(proxy_2024plus["horizons"][str(h)]["net_nominal_gate"] for h in HORIZONS)
    audit: dict[str, Any] = {
        "schema": "delay5_common_horizon_att_audit_v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": {
            "production_cohort_source": "validated surge_delay5_production_cohort audit artifact",
            "production_funnel": "raw→signal gate→amount/stop→historical ST→market→next-session fill→mature",
            "signal_gate": {
                "sig_vol_ratio_lte": tr.SURGE_GATE_VOL_RATIO,
                "sig_ma_spread_pct_gte": tr.SURGE_GATE_MA_SPREAD,
                "sig_ret20_gte": tr.SURGE_GATE_RET20,
            },
            "market_gate": {"high20_ratio_gt": live.MARKET_GATE_HIGH20, "ew_index_above_ma20": True},
            "horizons": {
                "sessions": list(HORIZONS),
                "entry_session_is_day_one": True,
                "endpoint_offsets": {str(h): h - 1 for h in HORIZONS},
                "common_support": "entry_pos+59 available; same treated/control identities for every horizon",
            },
            "amount_match": (
                f"decision-visible amount decile; nearest log amount; K={AMOUNT_K}; min={AMOUNT_MIN_VALID}; "
                "symbol tie-break"
            ),
            "proxy_match": (
                f"frozen annual industry; nearest log(qfq close*current float shares); K={PROXY_K}; "
                f"min={PROXY_MIN_VALID}; caliper={PROXY_CALIPER_RATIO}x"
            ),
            "control_pool": (
                "decision-date close/amount only; amount decile computed before exclusions; all same-date production "
                "treated and historical ST excluded; matching never checks future availability"
            ),
            "path_policy": (
                "next-session no-open or gap>=board-limit-0.3pct => cash zero/no replacement; post-entry suspension "
                "uses LOCF; terminal missing settlement reports pessimistic -100pct and optimistic LOCF bounds"
            ),
            "costs": {"buy": portfolio.BUY_COST, "sell": portfolio.SELL_COST, "cash_cost": 0.0},
            "inference": f"trade ATT; decision-date mean; HAC and stationary bootstrap block={BLOCK_LEN}",
            "parameter_search": "none; frozen production delay5 and two predeclared matching sensitivities",
        },
        "funnel": funnel,
        "support": support,
        "match_diagnostics": match_diagnostics,
        "summaries": summaries,
        "exact_mcap_request": {
            "path": str(output_dir / EXACT_REQUEST_PATH.name),
            **request_identity,
            "local_cache": exact_manifest["local_exact_cache"],
        },
        "data_identity": {
            "production_cohort": {
                "path": str(PRODUCTION_COHORT_PATH),
                "sha256": proxy.sha256_file(PRODUCTION_COHORT_PATH),
                "upstream_schema": production_audit["schema"],
                "upstream_audit_path": str(PRODUCTION_AUDIT_PATH),
                "upstream_audit_sha256": proxy.sha256_file(PRODUCTION_AUDIT_PATH),
                "raw_completeness_proven": bool(production_audit["raw_completeness_proven"]),
            },
            "candidates": {"path": str(CANDIDATES_PATH), "sha256": proxy.sha256_file(CANDIDATES_PATH)},
            "panel": {"path": str(PANEL_PATH), "sha256": proxy.sha256_file(PANEL_PATH)},
            "market_state": {"path": str(MARKET_PATH), "sha256": proxy.sha256_file(MARKET_PATH)},
            "namechange": {
                "path": str(portfolio.NAMECHANGE_PATH),
                "sha256": proxy.sha256_file(portfolio.NAMECHANGE_PATH),
            },
            "capital_proxy": {
                "path": str(proxy.CAPITAL_PATH),
                "sha256": proxy.sha256_file(proxy.CAPITAL_PATH),
                "quote_timestamps": sorted(capital["quote_timestamp"].astype(str).unique().tolist()),
                "warning": "current float shares are ex-post and are not point-in-time circ_mv",
            },
            "industry_snapshots": industry_identities,
        },
        "limitations": [
            "Industry/current-share proxy matching is not an exact point-in-time market-cap design.",
            "The local exact-mcap cache does not currently cover treated delay5 trades.",
            "Control reuse and overlapping 60-session paths create network dependence beyond decision-date HAC.",
            "Actual delisting settlement is absent; LOCF/-100pct bounds do not replace settlement data.",
            "Next falsification: decision-visible ret5/20, vol20, price and limit-up-path balance.",
        ],
        "verdict": {
            "status": (
                "PRODUCTION_DELAY5_PROXY_EDGE_REQUIRES_REVIEW"
                if proxy_confirmed
                else "PRODUCTION_DELAY5_COMMON_HORIZON_EDGE_NOT_CONFIRMED_AFTER_INDUSTRY_SIZE_CONTROL"
            ),
            "industry_size_proxy_any_2024plus_net_horizon_confirmed": proxy_confirmed,
            "exact_point_in_time_matching_complete": False,
            "live_authorized": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    treated_path = output_dir / TREATED_PATH.name
    pairs_path = output_dir / PAIRS_PATH.name
    trade_att_path = output_dir / TRADE_ATT_PATH.name
    request_path = output_dir / EXACT_REQUEST_PATH.name
    audit_path = output_dir / AUDIT_PATH.name
    treated.to_parquet(treated_path, index=False)
    pairs.to_parquet(pairs_path, index=False)
    trade_att.to_parquet(trade_att_path, index=False)
    request_path.write_text(
        json.dumps(_json_clean(exact_manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit["outputs"] = {
        "treated_common_support": {
            "path": str(treated_path),
            "rows": int(len(treated)),
            "sha256": proxy.sha256_file(treated_path),
        },
        "matched_pairs": {
            "path": str(pairs_path),
            "rows": int(len(pairs)),
            "sha256": proxy.sha256_file(pairs_path),
        },
        "trade_att": {
            "path": str(trade_att_path),
            "rows": int(len(trade_att)),
            "sha256": proxy.sha256_file(trade_att_path),
        },
        "exact_mcap_request_manifest": {
            "path": str(request_path),
            "rows": int(len(request_records)),
            "sha256": proxy.sha256_file(request_path),
        },
    }
    clean = _json_clean(audit)
    audit_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    result = run_audit(n_boot=args.n_boot)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    print(f"[output] {AUDIT_PATH}")


if __name__ == "__main__":
    main()
