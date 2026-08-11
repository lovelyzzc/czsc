"""对 2025 S2b 最大 5 笔匹配超额赢家做精确点时流通市值证伪。

这是事后诊断，不是确认性检验。它只检查上一轮结论最脆弱的环节：让 2025 显著性对
``drop-top-5`` 敏感的五笔交易，是否因为“当前股本 × 历史前复权价格”的规模代理失真，
从而匹配到了决策日规模差异过大的对照。

精确流通市值由 BaoStock 决策日未复权收盘价、成交量和换手率推导：

``circulating_shares = volume / (turnover_pct / 100)``
``circ_mv = close * circulating_shares``

    uv run --with baostock==0.9.3 python scripts/s2b_top_winner_exact_mcap_audit.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import _s2c_core as core
import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as proxy
import s2b_matched_control_audit as base
from surge_market_state_filter import StableControlSampler

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s2b_top_winner_exact_mcap"
REFERENCE_PATH = OUTPUT_DIR / "baostock_exact_mcap.parquet"
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
TOP_N = 5
EXACT_CALIPER_RATIO = 1.5
MIN_VALID = 5


def baostock_code(symbol: str) -> str:
    """把仓库代码转换为 BaoStock 代码。"""

    code, suffix = str(symbol).split(".", maxsplit=1)
    if suffix == "SH":
        return f"sh.{code}"
    if suffix == "SZ":
        return f"sz.{code}"
    raise ValueError(f"unsupported exchange suffix: {suffix}")


def derive_exact_circ_mv(close: float, volume: float, turnover_pct: float) -> float:
    """由未复权收盘价、成交股数和换手率推导流通市值。"""

    values = np.asarray([close, volume, turnover_pct], dtype=float)
    if not np.isfinite(values).all() or close <= 0 or volume <= 0 or turnover_pct <= 0:
        return np.nan
    circulating_shares = volume / (turnover_pct / 100.0)
    return float(close * circulating_shares)


def exact_caliper_mask(treated_mcap: float, control_mcaps: pd.Series, ratio: float = EXACT_CALIPER_RATIO) -> pd.Series:
    """返回满足对称市值倍数卡尺的控制组掩码。"""

    valid = control_mcaps.notna() & control_mcaps.gt(0)
    if not np.isfinite(treated_mcap) or treated_mcap <= 0:
        return pd.Series(False, index=control_mcaps.index)
    size_ratio = control_mcaps / treated_mcap
    return valid & size_ratio.between(1 / ratio, ratio)


def _load_year_matches(year: int, *, top_n: int | None = None) -> tuple[pd.DataFrame, StableControlSampler]:
    capital = proxy.fetch_capital_snapshot()
    industries = proxy.fetch_industry_snapshots()
    context = core.load_context(verbose=True)
    d0 = context["d0"]
    s2b = d0.loc[core.gate_mask(d0, core.S2B_VR, core.S2B_SP)].copy()
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        s2b[column] = pd.to_datetime(s2b[column])
    s2b = s2b[~base.censored_tail_mask(s2b)].copy()
    execution_keys = base._main_execution_keys(context)
    s2b["executed_main"] = [
        (symbol, entry_dt) in execution_keys for symbol, entry_dt in zip(s2b["symbol"], s2b["entry_dt"], strict=False)
    ]
    s2b = s2b[s2b["executed_main"] & s2b["dec_dt"].dt.year.eq(year)].copy()

    sampler = StableControlSampler()
    shares = capital.set_index("symbol")["current_float_shares"].reindex(sampler.close_w.columns)
    industry_map = industries[year].set_index("symbol")["industry"].reindex(sampler.close_w.columns)
    rows: list[dict[str, Any]] = []
    for trade in s2b.itertuples():
        match = proxy._match_trade(
            trade,
            sampler=sampler,
            shares=shares,
            industries=industry_map,
            caliper_ratio=proxy.CALIPER_RATIO,
        )
        if pd.isna(match["excess_pct"]):
            continue
        rows.append(
            {
                "symbol": trade.symbol,
                "dec_dt": pd.Timestamp(trade.dec_dt),
                "entry_dt": pd.Timestamp(trade.entry_dt),
                "exit_dt": pd.Timestamp(trade.exit_dt),
                "industry": industry_map.get(trade.symbol),
                "ret_gross_pct": float(trade.ret_gross_pct),
                "proxy_excess_pct": float(match["excess_pct"]),
                "control_symbols": list(match["control_symbols"]),
                "eligible_symbols": industry_map.index[
                    industry_map.eq(industry_map.get(trade.symbol))
                    & sampler.open_w.loc[trade.entry_dt].notna()
                    & sampler.close_w.loc[trade.exit_dt].notna()
                ]
                .difference(pd.Index([trade.symbol]))
                .tolist(),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["proxy_excess_pct", "symbol"], ascending=[False, True])
    if top_n is not None:
        if len(frame) < top_n:
            raise RuntimeError(f"only {len(frame)} valid {year} executed matches; need {top_n}")
        frame = frame.head(top_n)
    return frame.reset_index(drop=True), sampler


def _fetch_one(result: Any, *, symbol: str, dec_dt: pd.Timestamp) -> dict[str, Any]:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    record: dict[str, Any] = {
        "symbol": symbol,
        "dec_dt": pd.Timestamp(dec_dt),
        "error_code": result.error_code,
        "error_message": result.error_msg,
        "close": np.nan,
        "volume": np.nan,
        "turnover_pct": np.nan,
        "exact_circ_mv": np.nan,
    }
    if result.error_code != "0" or not rows:
        return record
    values = dict(zip(result.fields, rows[-1], strict=False))
    close = pd.to_numeric(values.get("close"), errors="coerce")
    volume = pd.to_numeric(values.get("volume"), errors="coerce")
    turnover = pd.to_numeric(values.get("turn"), errors="coerce")
    record.update(
        {
            "close": close,
            "volume": volume,
            "turnover_pct": turnover,
            "exact_circ_mv": derive_exact_circ_mv(close, volume, turnover),
        }
    )
    return record


def fetch_exact_mcaps(pairs: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    """逐代码/决策日获取精确流通市值，支持增量缓存。"""

    requested = pairs[["symbol", "dec_dt"]].drop_duplicates().copy()
    requested["dec_dt"] = pd.to_datetime(requested["dec_dt"])
    if REFERENCE_PATH.exists() and not refresh:
        cached = pd.read_parquet(REFERENCE_PATH)
        cached["dec_dt"] = pd.to_datetime(cached["dec_dt"])
    else:
        cached = pd.DataFrame(columns=["symbol", "dec_dt"])
    existing = set(zip(cached.get("symbol", []), cached.get("dec_dt", []), strict=False))
    missing = requested[
        [
            (symbol, dec_dt) not in existing
            for symbol, dec_dt in zip(requested["symbol"], requested["dec_dt"], strict=False)
        ]
    ]
    if not missing.empty:
        try:
            import baostock as bs
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("missing baostock; run with `uv run --with baostock==0.9.3`") from exc

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
        fetched: list[dict[str, Any]] = []
        try:
            for index, row in enumerate(missing.itertuples(index=False), 1):
                date_text = pd.Timestamp(row.dec_dt).strftime("%Y-%m-%d")
                result = bs.query_history_k_data_plus(
                    baostock_code(row.symbol),
                    "date,code,close,volume,turn,tradestatus",
                    start_date=date_text,
                    end_date=date_text,
                    frequency="d",
                    adjustflag="3",
                )
                fetched.append(_fetch_one(result, symbol=row.symbol, dec_dt=row.dec_dt))
                if index % 25 == 0 or index == len(missing):
                    print(f"[exact-mcap] {index}/{len(missing)}", flush=True)
        finally:
            bs.logout()
        fetched_frame = pd.DataFrame(fetched)
        cached = fetched_frame if cached.empty else pd.concat([cached, fetched_frame], ignore_index=True)
        cached = cached.drop_duplicates(["symbol", "dec_dt"], keep="last")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cached.sort_values(["dec_dt", "symbol"]).to_parquet(REFERENCE_PATH, index=False)
    return requested.merge(cached, on=["symbol", "dec_dt"], how="left", validate="one_to_one")


def run_audit(*, refresh_reference: bool = False) -> dict[str, Any]:
    """运行最大赢家精确市值诊断。"""

    top, sampler = _load_year_matches(2025, top_n=TOP_N)
    pair_rows: list[dict[str, Any]] = []
    for trade in top.itertuples():
        pair_rows.append({"symbol": trade.symbol, "dec_dt": trade.dec_dt})
        pair_rows.extend({"symbol": symbol, "dec_dt": trade.dec_dt} for symbol in trade.eligible_symbols)
    exact = fetch_exact_mcaps(pd.DataFrame(pair_rows), refresh=refresh_reference)
    exact_map = exact.set_index(["symbol", "dec_dt"])["exact_circ_mv"]

    trade_rows: list[dict[str, Any]] = []
    for trade in top.itertuples():
        treated_mcap = exact_map.get((trade.symbol, trade.dec_dt), np.nan)
        control_mcaps = pd.Series(
            {symbol: exact_map.get((symbol, trade.dec_dt), np.nan) for symbol in trade.eligible_symbols}, dtype=float
        )
        mask = exact_caliper_mask(treated_mcap, control_mcaps)
        exact_sizes = pd.concat([control_mcaps, pd.Series({trade.symbol: treated_mcap})])
        retained = proxy.nearest_control_symbols(
            exact_sizes,
            treated_symbol=trade.symbol,
            eligible_symbols=control_mcaps.index[mask],
            k=proxy.MATCH_K,
            caliper_ratio=EXACT_CALIPER_RATIO,
        )
        returns = (
            sampler.close_w.loc[trade.exit_dt, retained].to_numpy(float)
            / sampler.open_w.loc[trade.entry_dt, retained].to_numpy(float)
            - 1
            if retained
            else np.array([], dtype=float)
        )
        returns = returns[np.isfinite(returns)]
        exact_excess = trade.ret_gross_pct - float(np.median(returns)) * 100 if len(returns) >= MIN_VALID else np.nan
        ratios = control_mcaps.loc[retained] / treated_mcap if retained else pd.Series(dtype=float)
        trade_rows.append(
            {
                "symbol": trade.symbol,
                "dec_dt": str(trade.dec_dt.date()),
                "industry": trade.industry,
                "ret_gross_pct": round(float(trade.ret_gross_pct), 6),
                "proxy_excess_pct": round(float(trade.proxy_excess_pct), 6),
                "treated_exact_circ_mv": float(treated_mcap) if np.isfinite(treated_mcap) else None,
                "proxy_controls": list(trade.control_symbols),
                "exact_industry_pool_n": int(len(control_mcaps)),
                "exact_mcap_covered_controls": int(control_mcaps.notna().sum()),
                "exact_caliper_available_n": int(mask.sum()),
                "exact_caliper_controls": retained,
                "exact_caliper_n": int(len(returns)),
                "exact_max_size_ratio": (
                    round(float(np.maximum(ratios, 1 / ratios).max()), 6) if not ratios.empty else None
                ),
                "exact_excess_pct": round(float(exact_excess), 6) if np.isfinite(exact_excess) else None,
            }
        )

    valid_exact = [row for row in trade_rows if row["exact_excess_pct"] is not None]
    original_sum = float(sum(row["proxy_excess_pct"] for row in trade_rows))
    exact_sum = float(sum(row["exact_excess_pct"] for row in valid_exact)) if valid_exact else np.nan
    all_valid = len(valid_exact) == TOP_N
    if all_valid and exact_sum > 0:
        status = "TOP_WINNER_PROXY_MISMATCH_NOT_PRIMARY_ARTIFACT"
    elif all_valid:
        status = "TOP_WINNER_EDGE_COLLAPSES_UNDER_EXACT_MCAP"
    else:
        status = "TOP_WINNER_EXACT_CALIPER_COVERAGE_INSUFFICIENT"
    return {
        "schema": "s2b_top_winner_exact_mcap_audit_v1",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "design": {
            "scope": "post-hoc diagnostic of the five largest 2025 strict-match excess winners",
            "exact_mcap": "BaoStock unadjusted close * volume / (turnover_pct / 100) on decision date",
            "control_selection": "entire same-industry tradable pool; exact 1.5x mcap caliper; exact-nearest K=10",
            "minimum_valid_controls": MIN_VALID,
            "confirmation_use": "none; diagnostic falsification only",
        },
        "reference": {
            "path": str(REFERENCE_PATH),
            "sha256": proxy.sha256_file(REFERENCE_PATH),
            "requested_pairs": int(len(exact)),
            "valid_exact_mcap": int(exact["exact_circ_mv"].notna().sum()),
        },
        "trades": trade_rows,
        "summary": {
            "top_n": TOP_N,
            "trades_with_min_exact_controls": len(valid_exact),
            "original_proxy_excess_sum_pct": round(original_sum, 6),
            "exact_caliper_excess_sum_pct": round(exact_sum, 6) if np.isfinite(exact_sum) else None,
            "exact_to_proxy_sum_ratio": round(exact_sum / original_sum, 6)
            if np.isfinite(exact_sum) and original_sum != 0
            else None,
        },
        "verdict": {
            "status": status,
            "live_authorized": False,
            "reason": (
                "This targeted post-hoc audit can falsify proxy-size mismatch for the largest winners, but cannot "
                "replace a full-sample point-in-time mcap match or a new frozen forward window."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reference", action="store_true")
    args = parser.parse_args()
    started = time.time()
    result = run_audit(refresh_reference=args.refresh_reference)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {OUTPUT_PATH} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
