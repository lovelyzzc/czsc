"""逐年度复核 S2b 10 槽容量模拟入选交易的代理近邻是否满足精确点时市值卡尺。

该审计固定上一轮的同行业、规模代理最近邻 K=10，不重新搜索控制组。随后用 BaoStock
决策日未复权收盘价、成交量与换手率推导点时流通市值，删除不满足 1.5 倍精确市值卡尺
的控制，至少保留 5 个才进入统计。

这是支持集稳健性检查：精确卡尺不通过的交易会退出，因此必须同时报告覆盖率和同支持集
的原代理结果，不能把它解释成全样本确认。

    uv run --with baostock==0.9.3 python scripts/s2b_annual_exact_mcap_support_audit.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import s2b_industry_size_proxy_audit as proxy
import s2b_matched_control_audit as base
import s2b_top_winner_exact_mcap_audit as exact

OUTPUT_ROOT = Path(__file__).resolve().parent / "_output" / "s2b_annual_exact_mcap_support"
MIN_COVERAGE = 0.8


def output_dir(year: int) -> Path:
    """返回年度审计输出目录。"""

    return OUTPUT_ROOT / str(year)


def recompute_excess(ret_gross_pct: float, control_returns: np.ndarray, *, min_valid: int = exact.MIN_VALID) -> float:
    """使用通过精确卡尺的控制收益中位数重算毛超额。"""

    values = np.asarray(control_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < min_valid:
        return np.nan
    return float(ret_gross_pct - np.median(values) * 100)


def _summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    scoped = frame.rename(columns={column: "matched_excess_pct"})
    return base.summarize_matched_excess(scoped)


def run_audit(year: int, *, refresh_reference: bool = False) -> dict[str, Any]:
    """运行指定年度容量模拟入选集精确卡尺支持审计。"""

    if year not in proxy.INDUSTRY_ANCHORS:
        raise ValueError(f"year must be one of {sorted(proxy.INDUSTRY_ANCHORS)}")
    matches, sampler = exact._load_year_matches(year, top_n=None)
    capacity_simulated_trades = int(matches.attrs.get("capacity_simulated_trades", len(matches)))
    pair_rows: list[dict[str, Any]] = []
    for trade in matches.itertuples():
        pair_rows.append({"symbol": trade.symbol, "dec_dt": trade.dec_dt})
        pair_rows.extend({"symbol": symbol, "dec_dt": trade.dec_dt} for symbol in trade.control_symbols)
    exact_mcaps = exact.fetch_exact_mcaps(pd.DataFrame(pair_rows), refresh=refresh_reference)
    exact_map = exact_mcaps.set_index(["symbol", "dec_dt"])["exact_circ_mv"]

    rows: list[dict[str, Any]] = []
    for trade in matches.itertuples():
        treated_mcap = exact_map.get((trade.symbol, trade.dec_dt), np.nan)
        control_mcaps = pd.Series(
            {symbol: exact_map.get((symbol, trade.dec_dt), np.nan) for symbol in trade.control_symbols}, dtype=float
        )
        mask = exact.exact_caliper_mask(treated_mcap, control_mcaps)
        retained = control_mcaps.index[mask].tolist()
        control_returns = (
            sampler.close_w.loc[trade.exit_dt, retained].to_numpy(float)
            / sampler.open_w.loc[trade.entry_dt, retained].to_numpy(float)
            - 1
            if retained
            else np.array([], dtype=float)
        )
        exact_excess = recompute_excess(trade.ret_gross_pct, control_returns)
        finite_control_returns = control_returns[np.isfinite(control_returns)]
        rows.append(
            {
                "symbol": trade.symbol,
                "dec_dt": trade.dec_dt,
                "entry_dt": trade.entry_dt,
                "exit_dt": trade.exit_dt,
                "ret_gross_pct": float(trade.ret_gross_pct),
                "treated_industry": trade.industry,
                "hold_days": int(trade.hold_days),
                "exit_reason": trade.exit_reason,
                "dec_regime": int(trade.dec_regime),
                "score": float(trade.score),
                "amount_e": float(trade.amount_e),
                "sl_pct": float(trade.sl_pct),
                "sig_vol_ratio": float(trade.sig_vol_ratio),
                "sig_ma_spread_pct": float(trade.sig_ma_spread_pct),
                "sig_ret20": float(trade.sig_ret20),
                "proxy_excess_pct": float(trade.proxy_excess_pct),
                "exact_excess_pct": exact_excess,
                "exact_control_median_pct": (
                    float(np.median(finite_control_returns) * 100) if len(finite_control_returns) else np.nan
                ),
                "exact_mcap_covered_controls": int(control_mcaps.notna().sum()),
                "exact_caliper_controls": int(len(retained)),
                "exact_caliper_control_symbols": retained,
            }
        )
    frame = pd.DataFrame(rows)
    supported = frame[frame["exact_excess_pct"].notna()].copy()
    coverage = len(supported) / len(frame) if len(frame) else 0.0
    capacity_coverage = len(supported) / capacity_simulated_trades if capacity_simulated_trades else 0.0
    trade_output_path = output_dir(year) / "trades.parquet"
    trade_output_path.parent.mkdir(parents=True, exist_ok=True)
    supported.sort_values(["dec_dt", "symbol"]).to_parquet(trade_output_path, index=False)
    exact_summary = _summary(supported, "exact_excess_pct")
    proxy_on_support = _summary(supported, "proxy_excess_pct")
    concentration = proxy.concentration_audit(supported, "exact_excess_pct")
    nominal_positive = bool(
        exact_summary.get("daily_hac", {}).get("t_stat", -np.inf) >= 2
        and exact_summary.get("daily_block_bootstrap", {}).get("ci_lo", -np.inf) > 0
    )
    drop_top5_nominal_positive = bool(
        concentration.get("drop_top_5_trades", {}).get("daily_hac", {}).get("t_stat", -np.inf) >= 2
        and concentration.get("drop_top_5_trades", {}).get("daily_block_bootstrap", {}).get("ci_lo", -np.inf) > 0
    )
    if capacity_coverage < MIN_COVERAGE:
        status = "EXACT_CALIPER_SUPPORT_INSUFFICIENT_VS_CAPACITY_SET"
    elif nominal_positive and not drop_top5_nominal_positive:
        status = f"{year}_EXACT_SUPPORT_EDGE_REMAINS_TAIL_DEPENDENT"
    elif nominal_positive:
        status = f"{year}_EXACT_SUPPORT_NOMINAL_ASSOCIATION"
    else:
        status = f"{year}_NO_NOMINAL_EXACT_SUPPORT_ASSOCIATION"
    return {
        "schema": "s2b_annual_exact_mcap_support_audit_v2",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "design": {
            "year": year,
            "scope": f"{year} S2b d0 structural candidates selected by the 10-slot capacity simulation",
            "controls": "frozen same-industry proxy-nearest K=10; exact 1.5x mcap filter; min 5",
            "exact_mcap": "BaoStock unadjusted close * volume / (turnover_pct / 100)",
            "support_warning": "trades failing exact control coverage are excluded; proxy is reported on identical support",
            "execution_scope_warning": (
                "The capacity simulation does not apply the production delay5 signal, hard filters, ST history, "
                "market-state gate, or fill rules; it is not a live execution mirror."
            ),
            "outcome_warning": (
                "Treated trades use path-dependent FULL exits, while controls use the treated exit date; "
                "future outcome availability is also required when forming the control pool."
            ),
            "inference_warning": "annual splits are post-selection historical diagnostics, not independent OOS tests",
        },
        "reference": {
            "path": str(exact.REFERENCE_PATH),
            "sha256": proxy.sha256_file(exact.REFERENCE_PATH),
            "requested_pairs": int(len(exact_mcaps)),
            "valid_exact_mcap": int(exact_mcaps["exact_circ_mv"].notna().sum()),
        },
        "coverage": {
            "capacity_simulated_trades": capacity_simulated_trades,
            "proxy_matched_trades": int(len(frame)),
            "exact_supported_trades": int(len(supported)),
            "exact_coverage_of_proxy_pct": round(float(coverage * 100), 2),
            "exact_coverage_of_capacity_pct": round(float(capacity_coverage * 100), 2),
            "exact_controls_median": round(float(supported["exact_caliper_controls"].median()), 2)
            if len(supported)
            else None,
            "exact_controls_min": int(supported["exact_caliper_controls"].min()) if len(supported) else None,
        },
        "trade_output": {
            "path": str(trade_output_path),
            "sha256": proxy.sha256_file(trade_output_path),
            "rows": int(len(supported)),
        },
        "summaries": {
            "exact_mcap_support": exact_summary,
            "proxy_on_identical_support": proxy_on_support,
        },
        "concentration": concentration,
        "verdict": {
            "status": status,
            "exact_support_nominal_gate": nominal_positive,
            "drop_top5_nominal_gate": drop_top5_nominal_positive,
            "interpretation": "nominal post-selection historical association only",
            "live_authorized": False,
            "reason": (
                f"The {year} window is a supported-subset audit. It can test proxy mcap "
                "fidelity, but cannot establish temporal robustness or replace a new frozen forward sample."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=sorted(proxy.INDUSTRY_ANCHORS), default=2025)
    parser.add_argument("--refresh-reference", action="store_true")
    args = parser.parse_args()
    started = time.time()
    result = run_audit(args.year, refresh_reference=args.refresh_reference)
    year_output_dir = output_dir(args.year)
    output_path = year_output_dir / "audit.json"
    year_output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {output_path} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
