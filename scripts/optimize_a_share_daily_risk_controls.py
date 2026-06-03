"""Scan risk controls for the A-share daily classified strategy.

This script starts from the best alpha families found by
``optimize_a_share_daily_strategies.py`` and tests portfolio risk overlays:
market weak exits, individual invalid exits, board caps, volatility targeting,
and drawdown throttles.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import a_share_daily_classified_strategy as base
import optimize_a_share_daily_strategies as opt


OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "a_share_daily_risk_optimization"

BOARD_OTHER = 0
BOARD_BJ = 1
BOARD_STAR = 2
BOARD_CHINEXT = 3


@dataclass(frozen=True)
class AlphaCase:
    family: str
    score: str
    top_n: int
    rebalance_days: int


@dataclass(frozen=True)
class RiskCase:
    market_policy: str
    invalid_policy: str
    cap_policy: str
    vol_target: str
    drawdown_guard: str
    cash_if_empty: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--fee-rate", type=float, default=0.0007)
    parser.add_argument("--max-ranked", type=int, default=600)
    parser.add_argument("--preset", choices=["smoke", "focused", "wide"], default="focused")
    return parser.parse_args()


def board_code(symbol: str) -> int:
    if symbol.endswith(".BJ") or symbol.startswith(("43", "83", "87", "88", "92")):
        return BOARD_BJ
    if symbol.startswith("688"):
        return BOARD_STAR
    if symbol.startswith(("300", "301")):
        return BOARD_CHINEXT
    return BOARD_OTHER


def board_name(code: int) -> str:
    return {
        BOARD_OTHER: "other",
        BOARD_BJ: "bj",
        BOARD_STAR: "star",
        BOARD_CHINEXT: "chinext",
    }[int(code)]


def cap_limits(policy: str, top_n: int) -> dict[str, int]:
    if policy == "none":
        return {}
    if policy == "bj40":
        return {"bj": max(1, int(top_n * 0.40))}
    if policy == "bj30":
        return {"bj": max(1, int(top_n * 0.30))}
    if policy == "bj20":
        return {"bj": max(1, int(top_n * 0.20))}
    if policy == "bj30_growth70":
        return {"bj": max(1, int(top_n * 0.30)), "growth": max(1, int(top_n * 0.70))}
    if policy == "growth60":
        return {"growth": max(1, int(top_n * 0.60))}
    raise ValueError(f"unknown cap_policy: {policy}")


def select_with_caps(candidates: np.ndarray, top_n: int, board_by_idx: np.ndarray, policy: str) -> np.ndarray:
    if len(candidates) == 0:
        return candidates[:0].astype(np.int32, copy=False)
    limits = cap_limits(policy, top_n)
    if not limits:
        return candidates[:top_n].astype(np.int32, copy=False)

    selected: list[int] = []
    bj_count = 0
    growth_count = 0
    for idx in candidates:
        board = int(board_by_idx[idx])
        is_bj = board == BOARD_BJ
        is_growth = board in (BOARD_BJ, BOARD_STAR, BOARD_CHINEXT)
        if "bj" in limits and is_bj and bj_count >= limits["bj"]:
            continue
        if "growth" in limits and is_growth and growth_count >= limits["growth"]:
            continue

        selected.append(int(idx))
        if is_bj:
            bj_count += 1
        if is_growth:
            growth_count += 1
        if len(selected) >= top_n:
            break

    return np.array(selected, dtype=np.int32)


def exposure_from_market(policy: str, breadth: float, c1_pct: float) -> float:
    if policy == "carry":
        return 1.0
    if policy == "breadth_cash":
        return 1.0 if breadth > 0.38 else 0.0
    if policy == "breadth_step":
        if breadth > 0.45:
            return 1.0
        return 0.5 if breadth > 0.38 else 0.0
    if policy == "c1_80_cash":
        return 0.0 if c1_pct > 0.80 else 1.0
    if policy == "c1_step":
        if c1_pct <= 0.60:
            return 1.0
        return 0.5 if c1_pct <= 0.80 else 0.0
    if policy == "c1_soft":
        if c1_pct <= 0.75:
            return 1.0
        return 0.3 if c1_pct <= 0.90 else 0.0
    raise ValueError(f"unknown market_policy: {policy}")


def apply_vol_target(policy: str, exposure: float, prior_rets: list[float]) -> float:
    if policy == "none" or exposure <= 0:
        return exposure
    target_ann = {"ann20": 0.20, "ann25": 0.25, "ann30": 0.30}[policy]
    if len(prior_rets) < 20:
        return exposure
    realized_daily = float(np.std(prior_rets[-20:], ddof=0))
    if realized_daily <= 1e-8:
        return exposure
    target_daily = target_ann / math.sqrt(252)
    return min(exposure, target_daily / realized_daily)


def apply_drawdown_guard(policy: str, exposure: float, equity: float, peak: float) -> float:
    if policy == "none" or exposure <= 0 or peak <= 0:
        return exposure
    drawdown = equity / peak - 1
    if policy == "dd15_half":
        return exposure * 0.5 if drawdown <= -0.15 else exposure
    if policy == "dd20_half":
        return exposure * 0.5 if drawdown <= -0.20 else exposure
    if policy == "dd20_quarter":
        return exposure * 0.25 if drawdown <= -0.20 else exposure
    raise ValueError(f"unknown drawdown_guard: {policy}")


def calc_weight_turnover(previous_idx: np.ndarray, new_idx: np.ndarray, previous_exposure: float, new_exposure: float) -> float:
    if (len(previous_idx) == 0 or previous_exposure <= 0) and (len(new_idx) == 0 or new_exposure <= 0):
        return 0.0
    if len(previous_idx) == 0 or previous_exposure <= 0:
        return float(new_exposure)
    if len(new_idx) == 0 or new_exposure <= 0:
        return float(previous_exposure)

    prev_w = previous_exposure / len(previous_idx)
    new_w = new_exposure / len(new_idx)
    overlap = len(np.intersect1d(previous_idx, new_idx, assume_unique=False))
    return (
        overlap * abs(new_w - prev_w)
        + (len(new_idx) - overlap) * new_w
        + (len(previous_idx) - overlap) * prev_w
    )


def calc_metrics(rets: list[float], turns: list[float], exposures: list[float], counts: list[int]) -> dict[str, float]:
    metrics = opt.calc_metrics(rets, turns)
    metrics["avg_exposure"] = float(np.mean(exposures)) if exposures else 0.0
    metrics["avg_names"] = float(np.mean(counts)) if counts else 0.0
    metrics["active_days"] = int(sum(x > 0 for x in exposures))
    metrics["max_daily_loss"] = float(np.min(rets)) if rets else 0.0
    return metrics


def build_valid_matrices(df: pd.DataFrame, trade_dates: list[str], symbols: pd.Index) -> dict[str, np.ndarray]:
    policies = {
        "ma60": (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.98),
        "ma60_ret20": (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"] * 0.98) & (df["ret20"] > -0.15),
        "strict": (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]) & (df["ret20"] > -0.12),
    }
    out: dict[str, np.ndarray] = {
        "none": np.ones((len(trade_dates), len(symbols)), dtype=bool),
    }
    for name, valid in policies.items():
        tmp = df.loc[:, ["trade_date", "ts_code"]].copy()
        tmp["valid"] = valid.astype("float32")
        matrix = tmp.pivot_table(index="trade_date", columns="ts_code", values="valid", aggfunc="last")
        matrix = matrix.reindex(index=trade_dates, columns=symbols).fillna(0)
        out[name] = matrix.to_numpy(dtype=np.float32, copy=False) > 0.5
    return out


def build_market_state(df: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
    hist = df["ma250"].notna()
    liquid = (df["amt20"] >= 20_000) & (df["close"] >= 2) & (df["open"] > 0)
    market_ok = df["breadth120"] > 0.38
    c1 = hist & liquid & ~market_ok
    state = pd.DataFrame({
        "breadth120": df.groupby("trade_date")["breadth120"].first(),
        "c1_count": c1.groupby(df["trade_date"]).sum(),
        "all_count": df.groupby("trade_date").size(),
    })
    state["c1_pct"] = state["c1_count"] / state["all_count"].replace(0, np.nan)
    state = state.reindex(trade_dates).fillna({"breadth120": 0, "c1_count": 0, "all_count": 0, "c1_pct": 1})
    return state


def run_case(
    ret_values: np.ndarray,
    trade_dates: list[str],
    ranked_candidates: dict[str, np.ndarray],
    board_by_idx: np.ndarray,
    valid_matrix: np.ndarray,
    market_state: pd.DataFrame,
    alpha: AlphaCase,
    risk: RiskCase,
    fee_rate: float,
) -> dict[str, float]:
    previous_idx = np.array([], dtype=np.int32)
    current_idx = np.array([], dtype=np.int32)
    previous_exposure = 0.0
    equity = 1.0
    peak = 1.0
    rets: list[float] = []
    turns: list[float] = []
    exposures: list[float] = []
    counts: list[int] = []

    for pos in range(0, len(trade_dates) - 2):
        signal_date = trade_dates[pos]
        if pos % alpha.rebalance_days == 0:
            candidates = ranked_candidates.get(signal_date, np.array([], dtype=np.int32))
            selected = select_with_caps(candidates, alpha.top_n, board_by_idx, risk.cap_policy)
            if len(selected) or risk.cash_if_empty:
                current_idx = selected

        if len(current_idx):
            current_idx = current_idx[valid_matrix[pos, current_idx]]

        row = market_state.iloc[pos]
        exposure = exposure_from_market(risk.market_policy, float(row["breadth120"]), float(row["c1_pct"]))
        exposure = apply_vol_target(risk.vol_target, exposure, rets)
        exposure = apply_drawdown_guard(risk.drawdown_guard, exposure, equity, peak)
        if len(current_idx) == 0:
            exposure = 0.0

        turnover = calc_weight_turnover(previous_idx, current_idx, previous_exposure, exposure)
        raw_ret = float(ret_values[pos + 1, current_idx].mean()) * exposure if exposure > 0 and len(current_idx) else 0.0
        strategy_ret = raw_ret - turnover * fee_rate

        equity *= 1 + strategy_ret
        peak = max(peak, equity)
        rets.append(strategy_ret)
        turns.append(turnover)
        exposures.append(exposure)
        counts.append(len(current_idx) if exposure > 0 else 0)
        previous_idx = current_idx
        previous_exposure = exposure

    metrics = calc_metrics(rets, turns, exposures, counts)
    metrics["candidate_days"] = len(ranked_candidates)
    return metrics


def alpha_cases(preset: str) -> list[AlphaCase]:
    if preset == "smoke":
        return [
            AlphaCase("small_trend_mid", "momentum", 120, 20),
            AlphaCase("lowvol_trend", "lowvol", 160, 15),
        ]

    cases: list[AlphaCase] = []
    seeds = [
        ("small_trend_mid", "momentum", [80, 120, 160], [15, 20]),
        ("small_trend_mid", "lowvol", [120, 160, 220], [15, 20]),
        ("small_trend_mid", "quality", [120, 160, 220], [15, 20]),
        ("small_trend", "momentum", [120, 160], [15, 20]),
        ("lowvol_trend", "lowvol", [120, 160, 220, 300], [15, 20]),
        ("lowvol_trend", "quality", [120, 160, 220, 300], [15, 20]),
        ("near_ma20_momentum", "short_momentum", [60, 80, 160], [15, 20]),
        ("wide_trend", "baseline", [120, 160, 220], [15, 20]),
    ]
    if preset == "wide":
        seeds.extend([
            ("breakout_60", "lowvol", [60, 80, 120], [10, 15, 20]),
            ("trend_pullback", "quality", [80, 120, 160], [10, 15, 20]),
            ("short_reversal", "short_momentum", [60, 80, 120], [10, 15, 20]),
        ])
    for family, score, top_ns, rebalances in seeds:
        for top_n in top_ns:
            for rebalance_days in rebalances:
                cases.append(AlphaCase(family, score, top_n, rebalance_days))
    return cases


def risk_cases(preset: str) -> list[RiskCase]:
    if preset == "smoke":
        return [
            RiskCase("carry", "none", "none", "none", "none"),
            RiskCase("c1_step", "ma60", "bj30_growth70", "none", "none"),
        ]

    cases: list[RiskCase] = []
    for market_policy in ["carry", "breadth_step", "c1_80_cash", "c1_step", "c1_soft"]:
        for invalid_policy in ["none", "ma60", "ma60_ret20"]:
            for cap_policy in ["none", "bj40", "bj30", "bj30_growth70", "growth60"]:
                cases.append(RiskCase(market_policy, invalid_policy, cap_policy, "none", "none"))

    # Add selected portfolio-level overlays without exploding the grid.
    overlay_roots = [
        ("carry", "none", "none"),
        ("carry", "ma60", "bj30_growth70"),
        ("c1_step", "none", "none"),
        ("c1_step", "ma60", "bj30_growth70"),
        ("c1_soft", "ma60_ret20", "bj30"),
    ]
    for market_policy, invalid_policy, cap_policy in overlay_roots:
        for vol_target in ["ann25", "ann30"]:
            cases.append(RiskCase(market_policy, invalid_policy, cap_policy, vol_target, "none"))
        for drawdown_guard in ["dd15_half", "dd20_half", "dd20_quarter"]:
            cases.append(RiskCase(market_policy, invalid_policy, cap_policy, "none", drawdown_guard))

    return cases


def write_summary(output_dir: Path, results: pd.DataFrame, manifest: dict) -> None:
    top_calmar = results.sort_values(["calmar", "annual_return"], ascending=False).head(20)
    top_low_dd = results[results["annual_return"].gt(0.12)].sort_values(["max_drawdown", "annual_return"], ascending=[False, False]).head(20)
    top_sharpe = results.sort_values(["sharpe", "annual_return"], ascending=False).head(20)
    top_current_family = results[results["family"].eq("small_trend_mid") & results["score"].eq("momentum")].sort_values(["calmar", "annual_return"], ascending=False).head(20)

    lines = [
        "# A 股日线风险控制优化扫描",
        "",
        f"- 数据区间：`{manifest.get('start_date')}` 至 `{manifest.get('expected_trade_end_date', manifest.get('end_date'))}`",
        f"- 股票数：`{manifest.get('total_symbols')}`",
        "- 成本：换手成本 `0.0007`",
        "",
        "## Top Calmar",
        "",
        top_calmar.to_markdown(index=False),
        "",
        "## 年化大于 12% 的低回撤 Top",
        "",
        top_low_dd.to_markdown(index=False),
        "",
        "## Top Sharpe",
        "",
        top_sharpe.to_markdown(index=False),
        "",
        "## 当前 Alpha 家族 Top",
        "",
        top_current_family.to_markdown(index=False),
        "",
        "## 结论提示",
        "",
        "- `market_policy` 控制弱市时是否继续持有、降仓或清仓。",
        "- `invalid_policy` 在每日信号日检查旧持仓是否继续有效；不再只在调仓日换股。",
        "- `cap_policy` 限制北交所或高弹性成长板块过度集中。",
        "- `vol_target` 与 `drawdown_guard` 是组合层风险叠加，不改变选股排序。",
    ]
    (output_dir / "risk_optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.load_manifest(args.data_dir)
    end_date = args.end_date or manifest.get("expected_trade_end_date") or manifest.get("end_date")
    cfg = base.StrategyConfig(end_date=end_date)

    t0 = time.time()
    print(f"[INFO] loading features from {args.data_dir}", flush=True)
    df = base.load_features(args.data_dir, end_date=end_date, max_symbols=args.max_symbols)
    df = base.add_market_breadth(df, cfg)
    df = opt.add_optimizer_features(df)
    print(f"[INFO] feature rows={len(df)}, symbols={df['ts_code'].nunique()}, elapsed={time.time() - t0:.1f}s", flush=True)

    trade_dates = sorted(d for d in df["trade_date"].unique() if d >= args.start_date and (not end_date or d <= end_date))
    ret_matrix = df.pivot_table(index="trade_date", columns="ts_code", values="oo_ret", aggfunc="last")
    ret_matrix = ret_matrix.reindex(trade_dates).fillna(0).astype("float32")
    ret_values = ret_matrix.to_numpy(dtype=np.float32, copy=False)
    symbols = ret_matrix.columns
    symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}
    board_by_idx = np.array([board_code(symbol) for symbol in symbols], dtype=np.int8)
    valid_matrices = build_valid_matrices(df, trade_dates, symbols)
    market_state = build_market_state(df, trade_dates)

    scores = opt.calc_scores(df)
    families = opt.make_family_masks(df)
    alpha_grid = alpha_cases(args.preset)
    risk_grid = risk_cases(args.preset)
    max_ranked = max(args.max_ranked, max(a.top_n for a in alpha_grid))
    print(f"[INFO] alpha_cases={len(alpha_grid)}, risk_cases={len(risk_grid)}, max_ranked={max_ranked}", flush=True)

    ranked_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    case_count = 0
    for alpha in alpha_grid:
        cache_key = (alpha.family, alpha.score)
        if cache_key not in ranked_cache:
            ranked_cache[cache_key] = opt.build_ranked_candidates(
                df=df,
                mask=families[alpha.family],
                score=scores[alpha.score],
                max_top_n=max_ranked,
                symbol_to_idx=symbol_to_idx,
            )
        ranked_candidates = ranked_cache[cache_key]
        candidate_rows = int(families[alpha.family].sum())
        if not ranked_candidates:
            continue

        for risk in risk_grid:
            metrics = run_case(
                ret_values=ret_values,
                trade_dates=trade_dates,
                ranked_candidates=ranked_candidates,
                board_by_idx=board_by_idx,
                valid_matrix=valid_matrices[risk.invalid_policy],
                market_state=market_state,
                alpha=alpha,
                risk=risk,
                fee_rate=args.fee_rate,
            )
            rows.append({
                **asdict(alpha),
                **asdict(risk),
                "candidate_rows": candidate_rows,
                "candidate_days": len(ranked_candidates),
                **metrics,
            })
            case_count += 1
        if case_count % max(len(risk_grid), 1) == 0:
            print(f"[INFO] cases={case_count}, last_alpha={alpha}, elapsed={time.time() - t0:.1f}s", flush=True)

    results = pd.DataFrame(rows)
    results = results.sort_values(["calmar", "annual_return"], ascending=False).reset_index(drop=True)
    results.to_csv(args.output_dir / "risk_optimization_results.csv", index=False)
    write_summary(args.output_dir, results, manifest)
    payload = {
        "manifest": manifest,
        "preset": args.preset,
        "cases": len(results),
        "top_calmar": results.head(30).to_dict("records"),
        "top_low_dd": results[results["annual_return"].gt(0.12)].sort_values(["max_drawdown", "annual_return"], ascending=[False, False]).head(30).to_dict("records"),
    }
    (args.output_dir / "risk_optimization_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default), encoding="utf-8")

    cols = [
        "family", "score", "top_n", "rebalance_days",
        "market_policy", "invalid_policy", "cap_policy", "vol_target", "drawdown_guard",
        "total_return", "annual_return", "sharpe", "max_drawdown", "calmar",
        "avg_turnover", "avg_exposure", "avg_names", "active_days", "max_daily_loss",
    ]
    print("[INFO] Top Calmar")
    print(results[cols].head(20).to_string(index=False))
    print("[INFO] Top low drawdown with annual_return > 12%")
    low_dd = results[results["annual_return"].gt(0.12)].sort_values(["max_drawdown", "annual_return"], ascending=[False, False])
    print(low_dd[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
