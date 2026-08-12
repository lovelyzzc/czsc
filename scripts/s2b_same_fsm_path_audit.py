"""冻结 S2b 精确支持集的同 FULL-FSM 对照与逐日路径机制审计。

本脚本只做 2025 年事后敏感性诊断，不重新匹配控制组，也不搜索参数：

1. 校验 ``s2b_annual_exact_mcap_support_audit_v2`` 的交易文件 SHA256 与行数；
2. 对冻结的 ``exact_caliper_control_symbols`` 使用各自点时状态与止损参考重放 FULL 退出；
3. 比较原“处理票退出日固定收益”与同 FULL-FSM 控制收益；
4. 报告处理票 top5、其余处理票及控制实例的 MFE/MAE、到达 MFE 时间、涨停代理、
   最长连板代理和最大单腿收益贡献。

上游控制池曾要求控制票在处理票的未来退出日仍有价格，因此是 future-selected controls；
随机控制自身的 ``sl_ref`` 也不等同于处理票风险预算。结果不具确认性，不授权实盘。

    uv run --no-sync python scripts/s2b_same_fsm_path_audit.py
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trend_regime as tr
from surge_candidates_dump import MAX_HOLD_DAYS, SELL_SET, TRAIL_STOP

SCRIPT_DIR = Path(__file__).resolve().parent
ANNUAL_ROOT = SCRIPT_DIR / "_output" / "s2b_annual_exact_mcap_support"
OUTPUT_PATH = SCRIPT_DIR / "_output" / "s2b_same_fsm_path_audit.json"
YEAR = 2025
ANNUAL_SCHEMA = "s2b_annual_exact_mcap_support_audit_v2"
SCHEMA = "s2b_same_fsm_path_audit_v1"
MIN_VALID_CONTROLS = 5
RETURN_TOLERANCE_PCT = 0.001
DECISION_LOOKBACK_BARS = 20
PATH_METRICS = (
    "mfe_pct",
    "close_mfe_pct",
    "mae_pct",
    "time_to_mfe_bars",
    "time_to_mae_bars",
    "limit_up_days_held",
    "max_limit_up_streak_held",
    "decision_lookback_20_inclusive_limit_up_days",
    "max_single_leg_return_pct",
    "max_single_leg_contribution_pct",
    "max_single_leg_share_of_gross_pct",
    "hold_signal_bars",
)


@dataclass(frozen=True)
class FullExitResult:
    """FULL 退出的信号时点与真实成交时点。"""

    entry_idx: int
    entry_dt: pd.Timestamp
    entry_price: float
    exit_signal_idx: int
    exit_signal_dt: pd.Timestamp
    exit_fill_idx: int
    exit_fill_dt: pd.Timestamp
    exit_price: float
    reason: str

    @property
    def hold_signal_bars(self) -> int:
        return self.exit_signal_idx - self.entry_idx

    @property
    def hold_fill_bars(self) -> int:
        return self.exit_fill_idx - self.entry_idx

    @property
    def ret_gross_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0


@dataclass
class StockReplay:
    """单票原始日线、状态快照及索引。"""

    frame: pd.DataFrame
    states: list[Any]
    indicators: dict[str, Any]
    state_pos_by_dt: dict[pd.Timestamp, int]
    regime_by_idx: dict[int, int]


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_true_streak(flags: Iterable[bool]) -> int:
    """返回布尔序列中最长连续 True 长度。"""

    longest = current = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return longest


def simulate_full_exit(
    decision_state: Any,
    *,
    opens: np.ndarray,
    closes: np.ndarray,
    lows: np.ndarray,
    dates: Iterable[Any],
    regime_by_idx: dict[int, int],
    max_hold_days: int = MAX_HOLD_DAYS,
    trail_stop: float = TRAIL_STOP,
    sell_set: frozenset[int] = SELL_SET,
) -> FullExitResult | None:
    """严格重放候选脚本的 FULL 退出，并显式区分 state 信号日与次开盘成交日。"""

    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)
    lows = np.asarray(lows, dtype=float)
    dates = [pd.Timestamp(value) for value in dates]
    if not (len(opens) == len(closes) == len(lows) == len(dates)):
        raise ValueError("FULL exit arrays must have identical lengths")
    if max_hold_days <= 0:
        raise ValueError("max_hold_days must be positive")

    entry_idx = int(decision_state.idx) + 1
    entry_price = float(decision_state.next_open)
    if entry_idx >= len(closes) or not np.isfinite(entry_price) or entry_price <= 0:
        return None
    sl_ref = float(decision_state.sl_ref)
    peak = entry_price
    last_idx = min(entry_idx + max_hold_days, len(closes)) - 1

    for idx in range(entry_idx, last_idx + 1):
        peak = max(peak, float(closes[idx]))
        if idx == entry_idx:
            continue
        if np.isfinite(sl_ref) and lows[idx] <= sl_ref:
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=dates[entry_idx],
                entry_price=entry_price,
                exit_signal_idx=idx,
                exit_signal_dt=dates[idx],
                exit_fill_idx=idx,
                exit_fill_dt=dates[idx],
                exit_price=float(min(opens[idx], sl_ref)),
                reason="sl2",
            )
        if peak > entry_price and (peak - closes[idx]) / peak >= trail_stop:
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=dates[entry_idx],
                entry_price=entry_price,
                exit_signal_idx=idx,
                exit_signal_dt=dates[idx],
                exit_fill_idx=idx,
                exit_fill_dt=dates[idx],
                exit_price=float(closes[idx]),
                reason="trail18",
            )
        if regime_by_idx.get(idx) in sell_set:
            fill_idx = idx + 1 if idx + 1 < len(closes) else idx
            fill_price = opens[fill_idx] if fill_idx != idx else closes[idx]
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=dates[entry_idx],
                entry_price=entry_price,
                exit_signal_idx=idx,
                exit_signal_dt=dates[idx],
                exit_fill_idx=fill_idx,
                exit_fill_dt=dates[fill_idx],
                exit_price=float(fill_price),
                reason="state",
            )

    return FullExitResult(
        entry_idx=entry_idx,
        entry_dt=dates[entry_idx],
        entry_price=entry_price,
        exit_signal_idx=last_idx,
        exit_signal_dt=dates[last_idx],
        exit_fill_idx=last_idx,
        exit_fill_dt=dates[last_idx],
        exit_price=float(closes[last_idx]),
        reason="max_hold",
    )


def _executable_legs(frame: pd.DataFrame, result: FullExitResult) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造 entry-open 到真实退出成交价的可乘收益腿与可加价格贡献。"""

    prices = [result.entry_price]
    labels: list[str] = []
    for idx in range(result.entry_idx, result.exit_signal_idx + 1):
        row = frame.iloc[idx]
        if idx == result.exit_signal_idx and result.reason == "sl2":
            price = result.exit_price
            suffix = "sl_fill"
        else:
            price = float(row["close"])
            suffix = "close"
        prices.append(price)
        labels.append(f"{pd.Timestamp(row['dt']).date()}_{suffix}")
    if result.reason == "state" and result.exit_fill_idx != result.exit_signal_idx:
        prices.append(result.exit_price)
        labels.append(f"{result.exit_fill_dt.date()}_next_open_state_fill")

    prices_array = np.asarray(prices, dtype=float)
    leg_returns = prices_array[1:] / prices_array[:-1] - 1.0
    additive_contributions = np.diff(prices_array) / result.entry_price
    return leg_returns, additive_contributions, labels


def compute_path_metrics(
    frame: pd.DataFrame,
    result: FullExitResult,
    *,
    decision_idx: int,
    limit_threshold_pct: float,
) -> dict[str, Any]:
    """计算一笔 FULL 交易的路径指标；决策回看含决策日在内共 20 根。"""

    held = frame.iloc[result.entry_idx : result.exit_signal_idx + 1]
    if held.empty:
        raise ValueError("path has no held bars")
    highs = held["high"].to_numpy(dtype=float)
    lows = held["low"].to_numpy(dtype=float)
    closes = held["close"].to_numpy(dtype=float)
    mfe_pos = int(np.nanargmax(highs))
    mae_pos = int(np.nanargmin(lows))
    close_mfe_pos = int(np.nanargmax(closes))
    limit_flags = held["pct_chg"].astype(float).ge(limit_threshold_pct).to_numpy()
    lookback_start = max(0, decision_idx - DECISION_LOOKBACK_BARS + 1)
    lookback = frame.iloc[lookback_start : decision_idx + 1]
    lookback_limit_flags = lookback["pct_chg"].astype(float).ge(limit_threshold_pct)
    leg_returns, additive_contributions, leg_labels = _executable_legs(frame, result)
    max_return_pos = int(np.nanargmax(leg_returns))
    max_contribution_pos = int(np.nanargmax(additive_contributions))
    gross = result.ret_gross_pct
    signal_close = float(frame.iloc[result.exit_signal_idx]["close"])
    state_gap = (result.exit_price / signal_close - 1.0) * 100.0 if result.reason == "state" else None
    state_effect = (result.exit_price - signal_close) / result.entry_price * 100.0 if result.reason == "state" else None
    return {
        "mfe_pct": (highs[mfe_pos] / result.entry_price - 1.0) * 100.0,
        "mfe_dt": pd.Timestamp(held.iloc[mfe_pos]["dt"]),
        "time_to_mfe_bars": mfe_pos,
        "close_mfe_pct": (closes[close_mfe_pos] / result.entry_price - 1.0) * 100.0,
        "mae_pct": (lows[mae_pos] / result.entry_price - 1.0) * 100.0,
        "mae_dt": pd.Timestamp(held.iloc[mae_pos]["dt"]),
        "time_to_mae_bars": mae_pos,
        "limit_threshold_pct": float(limit_threshold_pct),
        "limit_up_days_held": int(limit_flags.sum()),
        "max_limit_up_streak_held": maximum_true_streak(limit_flags),
        "decision_lookback_20_inclusive_limit_up_days": int(lookback_limit_flags.sum()),
        "max_single_leg_return_pct": float(leg_returns[max_return_pos] * 100.0),
        "max_single_leg_return_label": leg_labels[max_return_pos],
        "max_single_leg_contribution_pct": float(additive_contributions[max_contribution_pos] * 100.0),
        "max_single_leg_share_of_gross_pct": (
            float(additive_contributions[max_contribution_pos] * 100.0 / gross * 100.0) if gross > 0 else None
        ),
        "max_single_leg_contribution_label": leg_labels[max_contribution_pos],
        "hold_signal_bars": result.hold_signal_bars,
        "hold_fill_bars": result.hold_fill_bars,
        "state_overnight_gap_pct": state_gap,
        "state_fill_effect_on_trade_pct": state_effect,
    }


def validate_annual_binding(
    annual: dict[str, Any],
    *,
    trade_sha256: str,
    trade_rows: int,
    year: int = YEAR,
) -> str:
    """绑定 annual v2 schema、年份、交易 SHA/行数和 future-selection 警告。"""

    if annual.get("schema") != ANNUAL_SCHEMA:
        raise RuntimeError(f"annual schema mismatch: {annual.get('schema')!r}")
    if int(annual.get("design", {}).get("year", -1)) != year:
        raise RuntimeError("annual design year mismatch")
    trade_output = annual.get("trade_output", {})
    if trade_output.get("sha256") != trade_sha256:
        raise RuntimeError("annual trade sha256 mismatch")
    if int(trade_output.get("rows", -1)) != trade_rows:
        raise RuntimeError("annual trade rows mismatch")
    if int(annual.get("coverage", {}).get("exact_supported_trades", -1)) != trade_rows:
        raise RuntimeError("annual exact-supported rows mismatch")
    warning = str(annual.get("design", {}).get("outcome_warning", ""))
    if "future outcome availability" not in warning or "path-dependent FULL exits" not in warning:
        raise RuntimeError("annual outcome warning no longer records future-selected asymmetric controls")
    return str(annual.get("verdict", {}).get("status", "UNKNOWN"))


def validate_trade_frame(frame: pd.DataFrame, *, year: int = YEAR) -> None:
    """校验审计所需的冻结交易字段与身份约束。"""

    required = {
        "symbol",
        "dec_dt",
        "entry_dt",
        "exit_dt",
        "ret_gross_pct",
        "hold_days",
        "exit_reason",
        "exact_excess_pct",
        "exact_control_median_pct",
        "exact_caliper_control_symbols",
    }
    if missing := required - set(frame):
        raise RuntimeError(f"annual trades missing columns: {sorted(missing)}")
    if frame.duplicated(["symbol", "dec_dt"]).any():
        raise RuntimeError("annual trades contain duplicate symbol/decision identities")
    if not frame["dec_dt"].dt.year.eq(year).all():
        raise RuntimeError("annual trades contain rows outside the bound year")
    control_counts = frame["exact_caliper_control_symbols"].map(lambda value: len(list(value)))
    if control_counts.lt(MIN_VALID_CONTROLS).any():
        raise RuntimeError("annual trades contain fewer than five exact controls")
    reconstructed = frame["ret_gross_pct"] - frame["exact_control_median_pct"]
    if not np.allclose(reconstructed, frame["exact_excess_pct"], atol=1e-9, rtol=0):
        raise RuntimeError("annual fixed-exit excess no longer reconciles")


@cache
def load_stock_replay(symbol: str) -> StockReplay | None:
    """加载并因果重放单票状态；同一进程按股票缓存。"""

    frame = tr.load_stock(tr.DATA_DIR / f"{symbol}.parquet")
    if frame is None:
        return None
    states = tr.iter_states(frame, with_features=True)
    indicators = tr.compute_indicators(frame)
    return StockReplay(
        frame=frame,
        states=states,
        indicators=indicators,
        state_pos_by_dt={pd.Timestamp(state.dt): pos for pos, state in enumerate(states)},
        regime_by_idx={int(state.idx): int(state.regime) for state in states},
    )


def replay_symbol(
    symbol: str,
    decision_dt: pd.Timestamp,
    *,
    expected_entry_dt: pd.Timestamp,
) -> tuple[dict[str, Any] | None, str | None]:
    """从指定决策日重放一票；入场日与处理票不一致时拒绝。"""

    replay = load_stock_replay(symbol)
    if replay is None:
        return None, "stock_unavailable"
    state_pos = replay.state_pos_by_dt.get(pd.Timestamp(decision_dt))
    if state_pos is None:
        return None, "decision_state_unavailable"
    decision_state = replay.states[state_pos]
    result = simulate_full_exit(
        decision_state,
        opens=replay.indicators["open"],
        closes=replay.indicators["close"],
        lows=replay.indicators["low"],
        dates=replay.indicators["dates"],
        regime_by_idx=replay.regime_by_idx,
    )
    if result is None:
        return None, "entry_unavailable"
    if result.entry_dt != pd.Timestamp(expected_entry_dt):
        return None, "entry_calendar_mismatch"
    sl_ref = float(decision_state.sl_ref)
    if not np.isfinite(sl_ref):
        stop_geometry = "missing"
        stop_distance_pct = None
    elif sl_ref >= result.entry_price:
        stop_geometry = "at_or_above_entry"
        stop_distance_pct = (result.entry_price - sl_ref) / result.entry_price * 100.0
    else:
        stop_geometry = "below_entry"
        stop_distance_pct = (result.entry_price - sl_ref) / result.entry_price * 100.0
    path = compute_path_metrics(
        replay.frame,
        result,
        decision_idx=int(decision_state.idx),
        limit_threshold_pct=tr.limit_pct_for(symbol),
    )
    record = {
        "symbol": symbol,
        "dec_dt": pd.Timestamp(decision_dt),
        "entry_dt": result.entry_dt,
        "entry_price": result.entry_price,
        "exit_signal_dt": result.exit_signal_dt,
        "exit_fill_dt": result.exit_fill_dt,
        "exit_price": result.exit_price,
        "exit_reason": result.reason,
        "ret_gross_pct": result.ret_gross_pct,
        "decision_regime": int(decision_state.regime),
        "sl_ref": sl_ref if np.isfinite(sl_ref) else None,
        "stop_distance_pct": stop_distance_pct,
        "stop_geometry": stop_geometry,
        **path,
    }
    return record, None


def stable_top_n(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """按超额、代码和日期稳定选择 top N。"""

    return frame.sort_values(
        ["exact_excess_pct", "symbol", "dec_dt"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(n)


def descriptive(values: Iterable[Any]) -> dict[str, Any]:
    """返回有限值的事后描述统计。"""

    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna().astype(float)
    if series.empty:
        return {"n": 0}
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
        "positive_pct": float(series.gt(0).mean() * 100.0),
    }


def path_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一组处理票或匹配实例的路径指标。"""

    if not records:
        return {"n": 0, "metrics": {}}
    frame = pd.DataFrame(records)
    return {
        "n": int(len(frame)),
        "metrics": {metric: descriptive(frame[metric]) for metric in PATH_METRICS},
        "rates_pct": {
            "any_limit_up_held": float(frame["limit_up_days_held"].gt(0).mean() * 100.0),
            "limit_up_streak_ge_2": float(frame["max_limit_up_streak_held"].ge(2).mean() * 100.0),
            "any_limit_up_in_decision_lookback": float(
                frame["decision_lookback_20_inclusive_limit_up_days"].gt(0).mean() * 100.0
            ),
        },
        "exit_reason_counts": {str(key): int(value) for key, value in frame["exit_reason"].value_counts().items()},
    }


def comparison_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """汇总固定退出与同 FSM 控制的差异。"""

    valid = frame.dropna(subset=["same_fsm_excess_pct"])
    return {
        "n": int(len(valid)),
        "fixed_exact_excess_pct": descriptive(valid["exact_excess_pct"]),
        "same_fsm_excess_pct": descriptive(valid["same_fsm_excess_pct"]),
        "same_minus_fixed_pct": descriptive(valid["same_minus_fixed_pct"]),
        "fixed_control_median_pct": descriptive(valid["exact_control_median_pct"]),
        "same_fsm_control_median_pct": descriptive(valid["same_fsm_control_median_pct"]),
    }


def derive_same_fsm_verdict(comparison: pd.DataFrame, *, expected_top_n: int = 5) -> dict[str, Any]:
    """由支持完整性与冻结 top-N 的同 FSM 正值条件生成 fail-closed 结论。"""

    required = {"is_top5", "same_fsm_excess_pct"}
    if missing := required - set(comparison):
        raise ValueError(f"comparison missing verdict columns: {sorted(missing)}")
    if expected_top_n <= 0:
        raise ValueError("expected_top_n must be positive")

    supported = comparison["same_fsm_excess_pct"].notna()
    top = comparison[comparison["is_top5"].astype(bool)]
    top_supported = top["same_fsm_excess_pct"].notna()
    complete_support = bool(len(comparison) > 0 and supported.all())
    complete_top_support = bool(len(top) == expected_top_n and top_supported.all())
    top_all_positive = bool(complete_top_support and top["same_fsm_excess_pct"].gt(0).all())

    all_sum = float(comparison.loc[supported, "same_fsm_excess_pct"].sum())
    top_sum = float(top.loc[top_supported, "same_fsm_excess_pct"].sum())
    top_share = float(top_sum / all_sum * 100.0) if complete_support and all_sum > 0 else None

    if not complete_support or not complete_top_support:
        status = "SAME_FSM_SENSITIVITY_INCOMPLETE_SUPPORT"
        reason = (
            "No preservation claim is made because same-FSM support is incomplete or the frozen top5 set "
            "does not contain exactly five supported trades."
        )
    elif top_all_positive:
        status = "SAME_FSM_SENSITIVITY_PRESERVES_FROZEN_TOP5_POSITIVITY"
        reason = (
            "All five frozen historical top5 trades retain positive excess under own-FULL controls; this is a "
            "post-selection sensitivity result, not evidence of causal or temporally robust alpha."
        )
    else:
        status = "SAME_FSM_SENSITIVITY_DOES_NOT_PRESERVE_FROZEN_TOP5_POSITIVITY"
        reason = (
            "At least one fully supported frozen top5 trade no longer has positive excess under own-FULL controls; "
            "the same-FSM sensitivity therefore does not preserve the frozen top5 positivity pattern."
        )

    return {
        "status": status,
        "treated_trades": int(len(comparison)),
        "same_fsm_supported_trades": int(supported.sum()),
        "complete_same_fsm_support": complete_support,
        "frozen_top5_trades": int(len(top)),
        "frozen_top5_supported_trades": int(top_supported.sum()),
        "frozen_top5_all_positive": top_all_positive if complete_top_support else None,
        "same_fsm_preserves_frozen_top5_positivity": bool(
            complete_support and complete_top_support and top_all_positive
        ),
        "same_fsm_removes_frozen_top5_positivity": (not top_all_positive) if complete_top_support else None,
        "all_same_fsm_excess_sum_pct": all_sum,
        "frozen_top5_same_fsm_excess_sum_pct": top_sum,
        "frozen_top5_share_of_same_fsm_excess_sum_pct": top_share,
        "confirmation": False,
        "live_authorized": False,
        "reason": reason,
    }


def _validate_treated_replay(trade: Any, record: dict[str, Any]) -> None:
    """确保新审计重放与候选文件的 legacy 信号日口径逐笔一致。"""

    if record["exit_reason"] != str(trade.exit_reason):
        raise RuntimeError(f"treated exit reason drift: {trade.symbol} {trade.dec_dt}")
    if record["exit_signal_dt"] != pd.Timestamp(trade.exit_dt):
        raise RuntimeError(f"treated exit signal date drift: {trade.symbol} {trade.dec_dt}")
    if int(record["hold_signal_bars"]) != int(trade.hold_days):
        raise RuntimeError(f"treated hold drift: {trade.symbol} {trade.dec_dt}")
    if abs(float(record["ret_gross_pct"]) - float(trade.ret_gross_pct)) > RETURN_TOLERANCE_PCT:
        raise RuntimeError(f"treated return drift: {trade.symbol} {trade.dec_dt}")


def _raw_closure(symbols: set[str]) -> dict[str, Any]:
    """计算本次状态重放实际依赖的原始文件内容闭包摘要。"""

    entries: list[list[Any]] = []
    total_bytes = 0
    for symbol in sorted(symbols):
        path = tr.DATA_DIR / f"{symbol}.parquet"
        if not path.exists():
            continue
        size = path.stat().st_size
        entries.append([symbol, sha256_file(path), size])
        total_bytes += size
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(tr.DATA_DIR),
        "requested_symbols": int(len(symbols)),
        "bound_files": int(len(entries)),
        "total_bytes": int(total_bytes),
        "canonical_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_safe(value: Any) -> Any:
    """把 numpy/pandas 标量递归转换为严格 JSON。"""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def run_audit(*, year: int = YEAR) -> dict[str, Any]:
    """运行冻结年度的同 FSM 与路径审计。"""

    annual_path = ANNUAL_ROOT / str(year) / "audit.json"
    trades_path = ANNUAL_ROOT / str(year) / "trades.parquet"
    annual = json.loads(annual_path.read_text(encoding="utf-8"))
    trades_sha = sha256_file(trades_path)
    trades = pd.read_parquet(trades_path)
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        trades[column] = pd.to_datetime(trades[column])
    upstream_status = validate_annual_binding(
        annual,
        trade_sha256=trades_sha,
        trade_rows=len(trades),
        year=year,
    )
    validate_trade_frame(trades, year=year)

    top = stable_top_n(trades, 5)
    top_keys = set(zip(top["symbol"], top["dec_dt"], strict=True))
    treated_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    raw_symbols = set(trades["symbol"].astype(str))

    for trade in trades.itertuples():
        parent_key = (str(trade.symbol), pd.Timestamp(trade.dec_dt))
        parent_is_top5 = parent_key in top_keys
        treated, error = replay_symbol(
            str(trade.symbol),
            pd.Timestamp(trade.dec_dt),
            expected_entry_dt=pd.Timestamp(trade.entry_dt),
        )
        if treated is None:
            raise RuntimeError(f"treated replay failed: {trade.symbol} {trade.dec_dt}: {error}")
        _validate_treated_replay(trade, treated)
        treated.update(
            {
                "exact_excess_pct": float(trade.exact_excess_pct),
                "is_top5": parent_is_top5,
            }
        )
        treated_records.append(treated)

        control_returns: list[float] = []
        parent_controls: list[dict[str, Any]] = []
        for control_symbol in map(str, list(trade.exact_caliper_control_symbols)):
            raw_symbols.add(control_symbol)
            control, control_error = replay_symbol(
                control_symbol,
                pd.Timestamp(trade.dec_dt),
                expected_entry_dt=pd.Timestamp(trade.entry_dt),
            )
            if control is None:
                failures[str(control_error)] += 1
                continue
            control.update(
                {
                    "parent_symbol": str(trade.symbol),
                    "parent_dec_dt": pd.Timestamp(trade.dec_dt),
                    "parent_is_top5": parent_is_top5,
                }
            )
            control_returns.append(float(control["ret_gross_pct"]))
            parent_controls.append(control)
        control_records.extend(parent_controls)
        if len(control_returns) >= MIN_VALID_CONTROLS:
            same_median = float(np.median(control_returns))
            same_excess = float(trade.ret_gross_pct - same_median)
        else:
            same_median = same_excess = np.nan
        comparison_rows.append(
            {
                "symbol": str(trade.symbol),
                "dec_dt": pd.Timestamp(trade.dec_dt),
                "is_top5": parent_is_top5,
                "ret_gross_pct": float(trade.ret_gross_pct),
                "exact_excess_pct": float(trade.exact_excess_pct),
                "exact_control_median_pct": float(trade.exact_control_median_pct),
                "same_fsm_valid_controls": int(len(control_returns)),
                "same_fsm_control_median_pct": same_median,
                "same_fsm_excess_pct": same_excess,
                "same_minus_fixed_pct": same_excess - float(trade.exact_excess_pct),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    controls = pd.DataFrame(control_records)
    treated_frame = pd.DataFrame(treated_records)
    valid_comparison = comparison.dropna(subset=["same_fsm_excess_pct"])
    exit_counts = controls["exit_reason"].value_counts()
    stop_counts = controls["stop_geometry"].value_counts()
    state_treated = treated_frame[treated_frame["exit_reason"].eq("state")]
    state_controls = controls[controls["exit_reason"].eq("state")]
    verdict = derive_same_fsm_verdict(comparison)

    top5_details: list[dict[str, Any]] = []
    for treated in treated_records:
        if not treated["is_top5"]:
            continue
        row = comparison[comparison["symbol"].eq(treated["symbol"]) & comparison["dec_dt"].eq(treated["dec_dt"])].iloc[
            0
        ]
        top5_details.append(
            {
                "symbol": treated["symbol"],
                "dec_dt": treated["dec_dt"],
                "fixed_exact_excess_pct": row["exact_excess_pct"],
                "same_fsm_excess_pct": row["same_fsm_excess_pct"],
                "same_fsm_control_median_pct": row["same_fsm_control_median_pct"],
                "path": {key: treated[key] for key in PATH_METRICS if key in treated},
                "exit_reason": treated["exit_reason"],
                "exit_signal_dt": treated["exit_signal_dt"],
                "exit_fill_dt": treated["exit_fill_dt"],
            }
        )

    result = {
        "schema": SCHEMA,
        "generated_at": pd.Timestamp.now(tz="UTC"),
        "design": {
            "year": year,
            "treated": "2025 annual-v2 exact-supported 10-slot capacity-simulated S2b trades",
            "controls": "frozen exact_caliper_control_symbols; no rematching or parameter search",
            "same_fsm": (
                "each control uses its own causal decision state, own sl_ref, next-open entry, SL2, "
                "18% close-peak trail, state 9/10 next-open exit, and 60-bar cap"
            ),
            "limit_up_proxy": "pct_chg >= 9.8% main-board or >=19.8% ChiNext; historical ST limits not reconstructed",
            "decision_lookback": "20 trading bars ending on the decision date, inclusive",
            "inference": "descriptive post-selection sensitivity only; no confirmatory p-value or live gate",
            "future_selected_controls": True,
            "own_stop_geometry_not_risk_matched": True,
        },
        "bindings": {
            "annual_audit": {
                "path": str(annual_path),
                "sha256": sha256_file(annual_path),
                "schema": annual["schema"],
                "status": upstream_status,
            },
            "annual_trades": {"path": str(trades_path), "sha256": trades_sha, "rows": int(len(trades))},
            "raw_qfq_closure": _raw_closure(raw_symbols),
        },
        "counts": {
            "treated_trades": int(len(trades)),
            "top5_trades": int(comparison["is_top5"].sum()),
            "control_instances_requested": int(
                trades["exact_caliper_control_symbols"].map(lambda value: len(list(value))).sum()
            ),
            "control_instances_replayed": int(len(controls)),
            "same_fsm_supported_trades": int(len(valid_comparison)),
            "same_fsm_min_controls": int(valid_comparison["same_fsm_valid_controls"].min()),
            "same_fsm_control_failures": {key: int(value) for key, value in sorted(failures.items())},
        },
        "fixed_vs_same_fsm": {
            "all": comparison_summary(comparison),
            "top5": comparison_summary(comparison[comparison["is_top5"]]),
            "rest": comparison_summary(comparison[~comparison["is_top5"]]),
            "top5_details": top5_details,
        },
        "control_diagnostics": {
            "exit_distribution": {
                "n": int(len(controls)),
                "counts": {str(key): int(value) for key, value in exit_counts.items()},
                "pct": {str(key): float(value / len(controls) * 100.0) for key, value in exit_counts.items()},
            },
            "stop_geometry": {
                "counts": {str(key): int(value) for key, value in stop_counts.items()},
                "at_or_above_entry_warning": (
                    "own sl_ref at/above entry can mechanically trigger an early SL2; this arm is not risk-budget matched"
                ),
            },
            "state_signal_fill_mismatch": {
                "treated_state_exits": int(len(state_treated)),
                "treated_mean_fill_effect_pct": float(state_treated["state_fill_effect_on_trade_pct"].mean()),
                "control_state_exits": int(len(state_controls)),
                "control_mean_fill_effect_pct": float(state_controls["state_fill_effect_on_trade_pct"].mean()),
                "legacy_warning": (
                    "upstream exit_dt is the state signal day although exit_price is the following trading day's open"
                ),
            },
        },
        "path_diagnostics": {
            "treated_top5": path_group_summary([row for row in treated_records if row["is_top5"]]),
            "treated_rest": path_group_summary([row for row in treated_records if not row["is_top5"]]),
            "controls_for_top5": path_group_summary([row for row in control_records if row["parent_is_top5"]]),
            "controls_for_rest": path_group_summary([row for row in control_records if not row["parent_is_top5"]]),
            "all_control_instances": path_group_summary(control_records),
        },
        "limitations": [
            (
                "FUTURE_SELECTED_CONTROLS: upstream matching required control prices at each treated trade's future "
                "exit date; replaying those frozen symbols cannot repair survivorship/outcome-availability selection."
            ),
            (
                "STOP_GEOMETRY: controls use their own state sl_ref, including non-candidate regimes and stops at or "
                "above entry; this tests own-structure FULL sensitivity, not a shared risk-budget exit policy."
            ),
            (
                "EXIT_TIME: legacy state exits label the signal day while filling at the next trading day's open; "
                "the audit reports both dates, but upstream matching and slot occupancy used the legacy signal date."
            ),
            (
                "PATH_MEASUREMENT: MFE/MAE use daily high/low and do not identify intraday event order; FULL trail "
                "itself uses closing-price peaks."
            ),
            ("LIMIT_UP_PROXY: constant board thresholds omit historical ST limits and exchange price rounding."),
            "NON_CONFIRMATORY: the year, S2b gate, exact support and top5 are all post-selection historical diagnostics.",
        ],
        "verdict": verdict,
    }
    return _json_safe(result)


def main() -> None:
    started = time.time()
    result = run_audit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {OUTPUT_PATH} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
