"""Test FSM states 9/10 as an exit-only overlay on the frozen factor F path.

The single hypothesis and every decision threshold are frozen in
``fsm_factor_exit_overlay_protocol_2026-08-12.json``.  The script computes the
development segment first and does not open the 2024+ temporal validation
unless every development gate passes.  This is retrospective research and can
never authorize live trading.

Run::

    uv run --no-sync python scripts/fsm_factor_exit_overlay.py
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from czsc._native.research import newey_west_hac

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROTOCOL_PATH = SCRIPT_DIR / "fsm_factor_exit_overlay_protocol_2026-08-12.json"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "fsm_factor_exit_overlay"
REPORT_PATH = SCRIPT_DIR / "FSM_FACTOR_EXIT_OVERLAY_2026-08-12.md"
PRIMARY_COST_BPS = 40


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
    """Load the frozen protocol and fail closed on input drift."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "fsm_factor_exit_overlay_protocol_v1":
        raise RuntimeError("unexpected exit-overlay protocol schema")
    if protocol.get("live_trading_authorized") is not False:
        raise RuntimeError("retrospective protocol must not authorize live trading")
    for name in ("ranked_surface", "historical_stage2_f_path", "state_cache", "price_panel"):
        record = protocol["inputs"][name]
        actual = sha256_file(_resolve(record["path"]))
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen input drift: {name} expected={record['sha256']} actual={actual}")
    return protocol


def cost_pair(roundtrip_bps: int) -> tuple[float, float]:
    """Map the frozen roundtrip stress to buy and sell costs."""

    if roundtrip_bps == PRIMARY_COST_BPS:
        return 0.0015, 0.0025
    one_way = roundtrip_bps / 20_000
    return one_way, one_way


def build_target_memberships(ranked: pd.DataFrame, *, target: int = 50, retention_rank: int = 75) -> pd.DataFrame:
    """Recursively reconstruct the frozen F identities without using outcomes."""

    required = {"decision_dt", "symbol", "factor_rank"}
    missing = required - set(ranked.columns)
    if missing:
        raise RuntimeError(f"ranked surface lacks membership columns: {sorted(missing)}")
    previous: set[str] = set()
    rows: list[dict[str, Any]] = []
    for decision_dt, week in ranked.groupby("decision_dt", sort=True, observed=True):
        week = week.sort_values(["factor_rank", "symbol"], kind="mergesort")
        indexed = week.set_index("symbol", drop=False)
        available = set(indexed.index.astype(str))
        retained = {
            symbol
            for symbol in previous
            if symbol in available and int(indexed.at[symbol, "factor_rank"]) <= retention_rank
        }
        fill = [str(symbol) for symbol in week["symbol"] if str(symbol) not in retained][: target - len(retained)]
        current = retained | set(fill)
        if len(current) != target:
            raise RuntimeError(f"F target underfilled on {pd.Timestamp(decision_dt).date()}: {len(current)}/{target}")
        for symbol in sorted(current, key=lambda item: (int(indexed.at[item, "factor_rank"]), item)):
            rows.append(
                {
                    "decision_dt": pd.Timestamp(decision_dt),
                    "symbol": symbol,
                    "factor_rank": int(indexed.at[symbol, "factor_rank"]),
                    "membership_role": "RETAINED" if symbol in retained else "NEW",
                }
            )
        previous = current
    result = pd.DataFrame(rows)
    if result.duplicated(["decision_dt", "symbol"]).any():
        raise RuntimeError("reconstructed F membership has duplicate identities")
    return result


def reconstruct_stage2_proxy(
    ranked: pd.DataFrame,
    memberships: pd.DataFrame,
    frozen_path: pd.DataFrame,
) -> dict[str, Any]:
    """Prove that the rebuilt 50/75 F path is the historical Stage-2 F path."""

    indexed = ranked.set_index(["decision_dt", "symbol"], drop=False)
    previous: set[str] = set()
    rows: list[dict[str, Any]] = []
    for decision_dt, group in memberships.groupby("decision_dt", sort=True, observed=True):
        current = set(group["symbol"].astype(str))
        week = indexed.loc[[(pd.Timestamp(decision_dt), symbol) for symbol in sorted(current)]]
        returns = pd.to_numeric(week["fwd_5d_open_return"], errors="coerce")
        observable = week["entry_tradable"].fillna(False) & np.isfinite(returns)
        rows.append(
            {
                "decision_dt": pd.Timestamp(decision_dt),
                "slots": len(current),
                "buy_count": len(current - previous),
                "sell_count": len(previous - current),
                "gross_return": 0.995 * float(returns.where(observable, 0.0).sum()) / 50,
            }
        )
        previous = current
    rebuilt = pd.DataFrame(rows)
    expected = frozen_path[frozen_path["arm"].eq("F")].copy()
    expected["decision_dt"] = pd.to_datetime(expected["decision_dt"])
    merged = rebuilt.merge(expected, on="decision_dt", suffixes=("_rebuilt", "_frozen"), validate="one_to_one")
    checks = {
        "decision_dates": len(merged) == len(rebuilt) == len(expected),
        "slots": np.array_equal(merged["slots_rebuilt"], merged["slots_frozen"]),
        "buys": np.array_equal(merged["buy_count_rebuilt"], merged["buy_count_frozen"]),
        "sells": np.array_equal(merged["sell_count_rebuilt"], merged["sell_count_frozen"]),
        "gross_return": np.allclose(
            merged["gross_return_rebuilt"], merged["gross_return_frozen"], rtol=0, atol=1e-15
        ),
    }
    maximum_error = float(np.max(np.abs(merged["gross_return_rebuilt"] - merged["gross_return_frozen"])))
    if not all(checks.values()):
        raise RuntimeError(f"frozen F reconstruction failed: {checks}")
    return {
        "status": "PASSED",
        "checks": checks,
        "weeks": int(len(merged)),
        "maximum_gross_return_error": maximum_error,
        "unique_symbols": int(memberships["symbol"].nunique()),
    }


@dataclass
class MarketData:
    """Dense price/state matrices for the fixed target identities."""

    calendar: pd.DatetimeIndex
    open: pd.DataFrame
    close: pd.DataFrame
    regime: pd.DataFrame


def load_market_data(protocol: dict[str, Any], symbols: list[str]) -> MarketData:
    """Load only symbols that ever enter the frozen F target path."""

    price_path = _resolve(protocol["inputs"]["price_panel"]["path"])
    state_path = _resolve(protocol["inputs"]["state_cache"]["path"])
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
    states = (
        pl.scan_parquet(state_path)
        .filter(pl.col("symbol").is_in(symbols))
        .select("symbol", "dt", "regime")
        .collect()
        .to_pandas()
    )
    price["dt"] = pd.to_datetime(price["dt"])
    states["dt"] = pd.to_datetime(states["dt"])
    return MarketData(
        calendar=calendar,
        open=price.pivot(index="dt", columns="symbol", values="open").reindex(calendar),
        close=price.pivot(index="dt", columns="symbol", values="close").reindex(calendar),
        regime=states.pivot(index="dt", columns="symbol", values="regime").reindex(calendar),
    )


def _finite_value(frame: pd.DataFrame, dt: pd.Timestamp, symbol: str) -> float | None:
    if dt not in frame.index or symbol not in frame.columns:
        return None
    value = frame.at[dt, symbol]
    return float(value) if pd.notna(value) and np.isfinite(value) and float(value) > 0 else None


def _targets_by_date(memberships: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    return {
        pd.Timestamp(dt): list(group.sort_values(["factor_rank", "symbol"], kind="mergesort")["symbol"].astype(str))
        for dt, group in memberships.groupby("decision_dt", sort=True, observed=True)
    }


def simulate_path(
    memberships: pd.DataFrame,
    market: MarketData,
    *,
    overlay: bool,
    buy_cost: float,
    sell_cost: float,
    end_date: pd.Timestamp,
    terminal_date: pd.Timestamp,
    decision_targets: dict[pd.Timestamp, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the self-financing 50-slot path using only causal close-to-next-open actions."""

    targets = _targets_by_date(memberships) if decision_targets is None else decision_targets
    start_date = min(targets)
    sessions = market.calendar[(market.calendar >= start_date) & (market.calendar <= end_date)]
    next_session = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(market.calendar[:-1], market.calendar[1:], strict=False)}
    slots: list[dict[str, Any]] = [
        {
            "cash": 0.995 / 50,
            "symbol": None,
            "shares": 0.0,
            "last_close": None,
            "pending_sell": None,
            "signal_dt": None,
            "planned_buy": None,
            "planned_buy_dt": None,
        }
        for _ in range(50)
    ]
    events: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    missing_terminal: list[str] = []

    def event(dt: pd.Timestamp, slot_no: int, action: str, reason: str, symbol: str, **extra: Any) -> None:
        events.append(
            {
                "dt": dt,
                "slot": slot_no,
                "action": action,
                "reason": reason,
                "symbol": symbol,
                **extra,
            }
        )

    for dt in sessions:
        dt = pd.Timestamp(dt)
        terminal = dt == terminal_date and end_date >= terminal_date

        # Fills are processed at the open; sells always precede same-open buys.
        for slot_no, slot in enumerate(slots):
            symbol = slot["symbol"]
            if symbol is None:
                continue
            if terminal and slot["pending_sell"] is None:
                slot["pending_sell"] = "TERMINAL"
            reason = slot["pending_sell"]
            if reason is None:
                continue
            price = _finite_value(market.open, dt, symbol)
            if price is None:
                if terminal:
                    missing_terminal.append(symbol)
                continue
            gross_value = float(slot["shares"]) * price
            cost = gross_value * sell_cost
            slot["cash"] = gross_value - cost
            slot["symbol"] = None
            slot["shares"] = 0.0
            slot["last_close"] = None
            slot["pending_sell"] = None
            event(dt, slot_no, "SELL", str(reason), symbol, price=price, transaction_cost=cost)

        if not terminal:
            held = {str(slot["symbol"]) for slot in slots if slot["symbol"] is not None}
            for slot_no, slot in enumerate(slots):
                planned = slot["planned_buy"]
                planned_dt = slot["planned_buy_dt"]
                if planned is None or planned_dt != dt:
                    continue
                if slot["symbol"] is None and planned not in held:
                    price = _finite_value(market.open, dt, planned)
                    if price is not None:
                        capital = float(slot["cash"])
                        shares = capital / (price * (1 + buy_cost))
                        cost = shares * price * buy_cost
                        slot["shares"] = shares
                        slot["cash"] = 0.0
                        slot["symbol"] = planned
                        slot["last_close"] = price
                        held.add(planned)
                        event(dt, slot_no, "BUY", "WEEKLY_TARGET", planned, price=price, transaction_cost=cost)
                    else:
                        event(dt, slot_no, "SKIP_BUY", "MISSING_STRICT_NEXT_OPEN", planned)
                slot["planned_buy"] = None
                slot["planned_buy_dt"] = None

        open_slot_values = []
        for slot in slots:
            if slot["symbol"] is None:
                open_slot_values.append(float(slot["cash"]))
            else:
                open_price = _finite_value(market.open, dt, str(slot["symbol"]))
                mark = open_price if open_price is not None else slot["last_close"]
                if mark is None:
                    raise RuntimeError("held slot has no observable open mark")
                open_slot_values.append(float(slot["shares"]) * float(mark))
        open_nav = 0.005 + float(sum(open_slot_values))

        # Mark held positions at the close, carrying the last mark through suspensions.
        for slot in slots:
            symbol = slot["symbol"]
            if symbol is None:
                continue
            close = _finite_value(market.close, dt, str(symbol))
            if close is not None:
                slot["last_close"] = close

        # The overlay is close-confirmed and can only fill at a later open.
        if overlay and not terminal:
            for slot_no, slot in enumerate(slots):
                symbol = slot["symbol"]
                if symbol is None or slot["pending_sell"] is not None:
                    continue
                regime = market.regime.at[dt, symbol] if dt in market.regime.index and symbol in market.regime.columns else np.nan
                if pd.notna(regime) and int(regime) in (9, 10):
                    slot["pending_sell"] = "STATE_9_10"
                    slot["signal_dt"] = dt
                    event(dt, slot_no, "SIGNAL", "STATE_9_10", str(symbol), regime=int(regime))

        # Weekly membership changes are decided after the close and fill next session.
        if dt in targets and not terminal:
            fill_dt = next_session.get(dt)
            if fill_dt is None:
                raise RuntimeError(f"no next market session after decision {dt.date()}")
            target_order = targets[dt]
            target_set = set(target_order)
            retained: set[str] = set()
            blocked_same_decision: set[str] = set()
            available_slots: list[int] = []
            for slot_no, slot in enumerate(slots):
                symbol = slot["symbol"]
                signal_dt = slot["signal_dt"]
                if symbol is None:
                    if signal_dt is None or pd.Timestamp(signal_dt) < dt:
                        available_slots.append(slot_no)
                    continue
                symbol = str(symbol)
                if slot["pending_sell"] == "STATE_9_10" and signal_dt == dt:
                    blocked_same_decision.add(symbol)
                    continue
                if symbol in target_set and slot["pending_sell"] is None:
                    retained.add(symbol)
                    continue
                if slot["pending_sell"] is None:
                    slot["pending_sell"] = "WEEKLY_DROP"
                available_slots.append(slot_no)
            missing = [symbol for symbol in target_order if symbol not in retained and symbol not in blocked_same_decision]
            for slot_no, symbol in zip(sorted(available_slots), missing, strict=False):
                slots[slot_no]["planned_buy"] = symbol
                slots[slot_no]["planned_buy_dt"] = fill_dt
            for symbol in missing[len(available_slots) :]:
                event(dt, -1, "SKIP_TARGET", "SAME_DECISION_STATE_COOLDOWN", symbol)

        slot_values = []
        for slot in slots:
            if slot["symbol"] is None:
                slot_values.append(float(slot["cash"]))
            else:
                last_close = slot["last_close"]
                if last_close is None:
                    raise RuntimeError("held slot has no observable mark")
                slot_values.append(float(slot["shares"]) * float(last_close))
        held_symbols = [str(slot["symbol"]) for slot in slots if slot["symbol"] is not None]
        if len(held_symbols) != len(set(held_symbols)):
            raise RuntimeError(f"duplicate live holding on {dt.date()}")
        path_rows.append(
            {
                "dt": dt,
                "open_nav": open_nav,
                "nav": 0.005 + float(sum(slot_values)),
                "held_slots": len(held_symbols),
                "cash_slots": sum(slot["symbol"] is None for slot in slots),
                "pending_sells": sum(slot["pending_sell"] is not None for slot in slots),
            }
        )

    final_held = [str(slot["symbol"]) for slot in slots if slot["symbol"] is not None]
    completeness = {
        "terminal_liquidation_requested": bool(end_date >= terminal_date),
        "final_held_slots": len(final_held),
        "final_held_symbols": final_held,
        "missing_terminal_symbols": sorted(set(missing_terminal)),
        "maximum_held_slots": max(row["held_slots"] for row in path_rows),
        "minimum_held_slots": min(row["held_slots"] for row in path_rows),
    }
    path = pd.DataFrame(path_rows)
    event_frame = pd.DataFrame(events)
    return path, event_frame, completeness


def segment_metrics(path: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, annualization: int = 242) -> dict[str, float]:
    """Compute path statistics with the last pre-segment NAV as the anchor."""

    ordered = path.sort_values("dt").copy()
    prior = ordered[ordered["dt"] < start].tail(1)
    scoped = ordered[(ordered["dt"] >= start) & (ordered["dt"] <= end)]
    if not prior.empty:
        scoped = pd.concat([prior, scoped], ignore_index=True)
    if len(scoped) < 2:
        raise RuntimeError(f"insufficient path rows for {start.date()}..{end.date()}")
    nav = scoped["nav"].to_numpy(dtype=float)
    daily = pd.Series(nav).pct_change().dropna().to_numpy(dtype=float)
    total = float(nav[-1] / nav[0] - 1)
    ann_return = float((nav[-1] / nav[0]) ** (annualization / len(daily)) - 1)
    ann_vol = float(np.std(daily, ddof=1) * math.sqrt(annualization)) if len(daily) > 1 else math.nan
    sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(annualization)) if ann_vol > 0 else math.nan
    running_max = np.maximum.accumulate(nav)
    max_drawdown = float(np.min(nav / running_max - 1))
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else math.nan
    return {
        "start_nav": float(nav[0]),
        "end_nav": float(nav[-1]),
        "sessions": int(len(daily)),
        "total_return": total,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "maximum_drawdown": max_drawdown,
        "calmar": float(calmar),
    }


def active_hac(
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lag: int,
    margin_weekly: float,
) -> dict[str, float]:
    """Compute weekly active returns and the frozen noninferiority test."""

    merged = baseline[["dt", "nav"]].merge(overlay[["dt", "nav"]], on="dt", suffixes=("_base", "_overlay"))
    merged = merged.set_index("dt").sort_index()
    anchor = merged[merged.index < start].tail(1)
    scoped = merged[(merged.index >= start) & (merged.index <= end)]
    if not anchor.empty:
        scoped = pd.concat([anchor, scoped])
    weekly = scoped.resample("W-FRI").last().pct_change().dropna()
    active = (weekly["nav_overlay"] - weekly["nav_base"]).to_numpy(dtype=float)
    result = {key: float(value) for key, value in newey_west_hac(active.tolist(), lag=lag).items()}
    z = (result["mean"] + margin_weekly) / result["se"] if result["se"] > 0 else math.inf
    result["noninferiority_margin_weekly"] = margin_weekly
    result["noninferiority_z"] = float(z)
    result["noninferiority_one_sided_p"] = float(norm.sf(z))
    return result


def summarize_comparison(
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
    overlay_events: pd.DataFrame,
    protocol: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Summarize a predeclared segment without selecting variants."""

    annualization = int(protocol["sample"]["annualization_sessions"])
    base = segment_metrics(baseline, start, end, annualization)
    over = segment_metrics(overlay, start, end, annualization)
    exits = overlay_events[
        overlay_events.get("action", pd.Series(dtype=str)).eq("SELL")
        & overlay_events.get("reason", pd.Series(dtype=str)).eq("STATE_9_10")
        & pd.to_datetime(overlay_events.get("dt", pd.Series(dtype="datetime64[ns]")).values).to_series().between(
            start, end
        ).to_numpy()
    ] if not overlay_events.empty else overlay_events
    hac = active_hac(
        baseline,
        overlay,
        start,
        end,
        lag=int(protocol["sample"]["weekly_hac_lag"]),
        margin_weekly=float(protocol["statistics"]["noninferiority_margin_weekly_mean"]),
    )
    return {
        "baseline": base,
        "overlay": over,
        "overlay_state_exit_fills": int(len(exits)),
        "annualized_return_difference": over["annualized_return"] - base["annualized_return"],
        "sharpe_improvement": over["sharpe"] - base["sharpe"],
        "relative_drawdown_reduction": (abs(base["maximum_drawdown"]) - abs(over["maximum_drawdown"]))
        / abs(base["maximum_drawdown"]),
        "weekly_active_hac": hac,
    }


def evaluate_primary_gate(summary: dict[str, Any], protocol: dict[str, Any], *, validation: bool) -> dict[str, Any]:
    """Apply the frozen development or validation decision rule."""

    stats = protocol["statistics"]
    minimum_exits = int(
        stats["minimum_validation_overlay_exits"] if validation else stats["minimum_development_overlay_exits"]
    )
    checks = {
        "minimum_exit_fills": summary["overlay_state_exit_fills"] >= minimum_exits,
        "drawdown_reduction": summary["relative_drawdown_reduction"] >= float(
            stats["minimum_relative_drawdown_reduction"]
        ),
        "sharpe_improvement": summary["sharpe_improvement"] >= float(stats["minimum_sharpe_improvement"]),
        "annual_return_noninferiority": summary["annualized_return_difference"]
        >= -float(stats["noninferiority_margin_annual_return"]),
        "weekly_hac_noninferiority": summary["weekly_active_hac"]["noninferiority_one_sided_p"]
        < float(stats["one_sided_alpha"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def calendar_subperiods(
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the three frozen validation calendar subperiods."""

    periods = [
        ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ("2026_YTD", pd.Timestamp("2026-01-01"), pd.Timestamp(protocol["sample"]["validation"][1])),
    ]
    rows = []
    for label, start, end in periods:
        base = segment_metrics(baseline, start, end, int(protocol["sample"]["annualization_sessions"]))
        over = segment_metrics(overlay, start, end, int(protocol["sample"]["annualization_sessions"]))
        rows.append(
            {
                "period": label,
                "baseline_total_return": base["total_return"],
                "overlay_total_return": over["total_return"],
                "active_total_return": over["total_return"] - base["total_return"],
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _pct(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"


def write_report(audit: dict[str, Any]) -> None:
    """Write the concise, failure-preserving research report."""

    dev = audit["development"]["summary"]
    lines = [
        "# FSM 作为因子 F 退出覆盖层：预注册回顾性检验",
        "",
        f"- 冻结协议：`{PROTOCOL_PATH.name}`",
        f"- 状态：**{audit['status']}**",
        "- 边界：历史样本已被既有研究看过；即使通过也只能进入前向测试，不能授权实盘。",
        "",
        "## 固定机制",
        "",
        "每周仍按纯因子 F 的 50/75 规则选择同一批股票；覆盖层只在持仓收盘进入 9（背驰衰竭）或",
        "10（结构破坏）时发出退出信号，并在下一可观察开盘卖出。卖出后不在周中补仓。",
        "",
        "## 基线完整性",
        "",
        f"- 复现周数：{audit['reconstruction']['weeks']}；唯一股票：{audit['reconstruction']['unique_symbols']}；",
        f"  最大周收益误差：{audit['reconstruction']['maximum_gross_return_error']:.3e}。",
        "- 冻结 F 的槽位、买入、卖出和周收益代理均通过复现门。",
        "",
        "## 开发期（2022–2023）",
        "",
        "| 指标 | F 基线 | F + 状态退出 | 差异 |",
        "|---|---:|---:|---:|",
        f"| 年化收益 | {_pct(dev['baseline']['annualized_return'])} | {_pct(dev['overlay']['annualized_return'])} | {_pct(dev['annualized_return_difference'])} |",
        f"| Sharpe | {dev['baseline']['sharpe']:.3f} | {dev['overlay']['sharpe']:.3f} | {dev['sharpe_improvement']:.3f} |",
        f"| 最大回撤 | {_pct(dev['baseline']['maximum_drawdown'])} | {_pct(dev['overlay']['maximum_drawdown'])} | {_pct(dev['relative_drawdown_reduction'])} 相对改善 |",
        f"| 状态退出成交 | — | {dev['overlay_state_exit_fills']} | — |",
        "",
        f"开发门：**{'PASS' if audit['development']['gate']['passed'] else 'FAIL'}**；逐项：",
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
                "开发门未全部通过，因此按协议没有计算或打开 2024+ 验证结果。这个精确的 9/10 退出覆盖层",
                "被回顾性否决，不再用拆分状态、延迟、冷却期或成本阈值救援。",
            ]
        )
    else:
        val = audit["validation"]["summary"]
        lines.extend(
            [
                "",
                "## 锁定验证期（2024+）",
                "",
                "| 指标 | F 基线 | F + 状态退出 | 差异 |",
                "|---|---:|---:|---:|",
                f"| 年化收益 | {_pct(val['baseline']['annualized_return'])} | {_pct(val['overlay']['annualized_return'])} | {_pct(val['annualized_return_difference'])} |",
                f"| Sharpe | {val['baseline']['sharpe']:.3f} | {val['overlay']['sharpe']:.3f} | {val['sharpe_improvement']:.3f} |",
                f"| 最大回撤 | {_pct(val['baseline']['maximum_drawdown'])} | {_pct(val['overlay']['maximum_drawdown'])} | {_pct(val['relative_drawdown_reduction'])} 相对改善 |",
                f"| 状态退出成交 | — | {val['overlay_state_exit_fills']} | — |",
                "",
                f"验证门：**{'PASS' if audit['validation']['gate']['passed'] else 'FAIL'}**。",
                f"60bp 成本压力门：**{'PASS' if audit['cost_stress']['passed'] else 'FAIL'}**。",
                "",
                "## 结论",
                "",
                audit["conclusion"],
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the frozen study, opening validation only after the development gate."""

    protocol = load_and_verify_protocol()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked_path = _resolve(protocol["inputs"]["ranked_surface"]["path"])
    ranked = pd.read_parquet(
        ranked_path,
        columns=[
            "decision_dt",
            "symbol",
            "factor_rank",
            "entry_tradable",
            "fwd_5d_open_return",
            "entry_dt",
            "exit_5d_dt",
        ],
    )
    ranked["decision_dt"] = pd.to_datetime(ranked["decision_dt"])
    memberships = build_target_memberships(ranked)
    memberships.to_parquet(OUTPUT_DIR / protocol["outputs"]["target_memberships"], index=False)
    frozen_path = pd.read_parquet(_resolve(protocol["inputs"]["historical_stage2_f_path"]["path"]))
    reconstruction = reconstruct_stage2_proxy(ranked, memberships, frozen_path)
    market = load_market_data(protocol, sorted(memberships["symbol"].unique()))
    dev_start = pd.Timestamp(protocol["sample"]["development"][0])
    dev_end = pd.Timestamp(protocol["sample"]["development"][1])
    val_start = pd.Timestamp(protocol["sample"]["validation"][0])
    val_end = pd.Timestamp(protocol["sample"]["validation"][1])
    terminal_date = pd.Timestamp(ranked["exit_5d_dt"].max())
    if terminal_date != val_end:
        raise RuntimeError(f"terminal date drift: protocol={val_end.date()} ranked={terminal_date.date()}")

    development_runs: dict[int, dict[str, Any]] = {}
    for bps in (20, 40, 60):
        buy_cost, sell_cost = cost_pair(bps)
        base_path, base_events, base_complete = simulate_path(
            memberships,
            market,
            overlay=False,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=dev_end,
            terminal_date=terminal_date,
        )
        over_path, over_events, over_complete = simulate_path(
            memberships,
            market,
            overlay=True,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            end_date=dev_end,
            terminal_date=terminal_date,
        )
        development_runs[bps] = {
            "baseline_path": base_path,
            "baseline_events": base_events,
            "baseline_completeness": base_complete,
            "overlay_path": over_path,
            "overlay_events": over_events,
            "overlay_completeness": over_complete,
            "summary": summarize_comparison(base_path, over_path, over_events, protocol, dev_start, dev_end),
        }

    primary_dev = development_runs[PRIMARY_COST_BPS]
    dev_gate = evaluate_primary_gate(primary_dev["summary"], protocol, validation=False)
    primary_dev["baseline_path"].assign(arm="F_BASELINE").pipe(
        lambda frame: pd.concat([frame, primary_dev["overlay_path"].assign(arm="F_EXIT_9_10")], ignore_index=True)
    ).to_parquet(OUTPUT_DIR / protocol["outputs"]["development_daily_paths"], index=False)
    primary_dev["baseline_events"].assign(arm="F_BASELINE").pipe(
        lambda frame: pd.concat([frame, primary_dev["overlay_events"].assign(arm="F_EXIT_9_10")], ignore_index=True)
    ).to_parquet(OUTPUT_DIR / protocol["outputs"]["development_events"], index=False)

    audit: dict[str, Any] = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "reconstruction": reconstruction,
        "development": {
            "summary": primary_dev["summary"],
            "gate": dev_gate,
            "cost_stress": {str(bps): development_runs[bps]["summary"] for bps in (20, 40, 60)},
            "baseline_completeness": primary_dev["baseline_completeness"],
            "overlay_completeness": primary_dev["overlay_completeness"],
        },
        "validation_opened": bool(dev_gate["passed"]),
        "live_trading_authorized": False,
    }

    if not dev_gate["passed"]:
        audit["status"] = "HYPOTHESIS_FALSIFIED_DEVELOPMENT"
        audit["conclusion"] = "开发门失败；锁定验证未打开。"
    else:
        full_runs: dict[int, dict[str, Any]] = {}
        for bps in (20, 40, 60):
            buy_cost, sell_cost = cost_pair(bps)
            base_path, base_events, base_complete = simulate_path(
                memberships,
                market,
                overlay=False,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                end_date=terminal_date,
                terminal_date=terminal_date,
            )
            over_path, over_events, over_complete = simulate_path(
                memberships,
                market,
                overlay=True,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                end_date=terminal_date,
                terminal_date=terminal_date,
            )
            if base_complete["final_held_slots"] or over_complete["final_held_slots"]:
                raise RuntimeError("terminal liquidation was incomplete")
            full_runs[bps] = {
                "baseline_path": base_path,
                "baseline_events": base_events,
                "baseline_completeness": base_complete,
                "overlay_path": over_path,
                "overlay_events": over_events,
                "overlay_completeness": over_complete,
                "summary": summarize_comparison(base_path, over_path, over_events, protocol, val_start, val_end),
            }
        primary_val = full_runs[PRIMARY_COST_BPS]
        val_gate = evaluate_primary_gate(primary_val["summary"], protocol, validation=True)
        subperiods = calendar_subperiods(primary_val["baseline_path"], primary_val["overlay_path"], protocol)
        calendar_gate = all(
            row["active_total_return"] >= -float(protocol["statistics"]["maximum_calendar_year_active_underperformance"])
            for row in subperiods
        )
        val_gate["checks"]["calendar_subperiod_floor"] = calendar_gate
        val_gate["passed"] = all(val_gate["checks"].values())
        dev_60 = development_runs[60]["summary"]
        val_60 = full_runs[60]["summary"]
        cost_checks = {
            "development_60bps_return_noninferiority_3pp": dev_60["annualized_return_difference"] >= -0.03,
            "development_60bps_drawdown_lower": dev_60["relative_drawdown_reduction"] > 0,
            "validation_60bps_return_noninferiority_3pp": val_60["annualized_return_difference"] >= -0.03,
            "validation_60bps_drawdown_lower": val_60["relative_drawdown_reduction"] > 0,
        }
        cost_gate = {"passed": all(cost_checks.values()), "checks": cost_checks}
        primary_val["baseline_path"].assign(arm="F_BASELINE").pipe(
            lambda frame: pd.concat([frame, primary_val["overlay_path"].assign(arm="F_EXIT_9_10")], ignore_index=True)
        ).query("dt >= @val_start").to_parquet(
            OUTPUT_DIR / protocol["outputs"]["validation_daily_paths"], index=False
        )
        primary_val["baseline_events"].assign(arm="F_BASELINE").pipe(
            lambda frame: pd.concat([frame, primary_val["overlay_events"].assign(arm="F_EXIT_9_10")], ignore_index=True)
        ).query("dt >= @val_start").to_parquet(OUTPUT_DIR / protocol["outputs"]["validation_events"], index=False)
        audit["validation"] = {
            "summary": primary_val["summary"],
            "gate": val_gate,
            "calendar_subperiods": subperiods,
            "cost_stress": {str(bps): full_runs[bps]["summary"] for bps in (20, 40, 60)},
            "baseline_completeness": primary_val["baseline_completeness"],
            "overlay_completeness": primary_val["overlay_completeness"],
        }
        audit["cost_stress"] = cost_gate
        passed = bool(val_gate["passed"] and cost_gate["passed"])
        audit["status"] = (
            "RETROSPECTIVE_RISK_OVERLAY_CANDIDATE_FORWARD_TEST_REQUIRED"
            if passed
            else "HYPOTHESIS_FALSIFIED_VALIDATION_OR_COST_STRESS"
        )
        audit["conclusion"] = (
            "开发、锁定验证和成本压力门均通过；该覆盖层只能进入独立前向测试。"
            if passed
            else "锁定验证或成本压力门失败；该精确覆盖层被回顾性否决，不做参数救援。"
        )

    audit = _jsonable(audit)
    (OUTPUT_DIR / protocol["outputs"]["audit"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit)
    print(json.dumps({"status": audit["status"], "validation_opened": audit["validation_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
