"""S2c 增量验证研究 — 严格回答 S2c 是否真正补强 S2b

对照策略：
  A: S2b Only (全天候核心，空仓持现金)
  B: S7 (S2b 核心 + S2c 牛市增量补位)
  C: S2c Only (仅增量信号，牛市)
  D: S2b + 随机补位 (500 次蒙特卡洛)
  E: S2b + 现金收益 (空闲资金年化 2%)
  F: S2b + 放宽无牛市过滤

输出到 scripts/_output/s2c_validation/

    uv run --no-sync python scripts/s2c_incremental_validation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _s2c_core as C  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Core Delegation
# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载 / regime / 门控 / 槽位模拟 / 指标 / 统计全部下沉到 _s2c_core，本文件只保留
# S2c 专属分析（归因、边际贡献、质量分层、MC 补位、压力网格）。
#
# 与 2026-07-31 首版的两处差异（首版是 bug，不是口径选择）：
#   1. 首版 simulate_detailed 接收 cost_pct 但函数体从未使用 → 全部读数是零成本的。
#      现在默认计入 40bps 往返成本，Sharpe/CAGR 会低于首版 report.json。
#      设 COST_PCT = 0.0 可复现首版数值（已验证逐笔一致）。
#   2. 首版 deflated_sharpe 把年化 Sharpe 代入日频 SE 公式，SE 低估 ~sqrt(252)。
#      现在用 _s2c_core.deflated_sharpe(legacy=False)；legacy=True 可复现旧值。
# 结论方向不受影响：成本只会让 S7 更差。详见 S8_INCREMENTAL_VALIDATION_2026-07-31.md

TRAIN_END = C.TRAIN_END
N_SLOTS = C.N_SLOTS
TOTAL_COST_PCT = C.TOTAL_COST_PCT
COST_PCT = C.TOTAL_COST_PCT          # 设 0.0 可复现首版零成本读数
LOOKBACK, BULL_TH, BEAR_TH = C.LOOKBACK, C.BULL_TH, C.BEAR_TH
ANN_DAYS = C.ANN_DAYS
CASH_RATE_DAILY = C.CASH_RATE_DAILY
MC_N = 500
BOOT_N = 1000

COST_STRESS_BPS = [10, 25, 40, 50, 75, 100]
SLOT_STRESS = [3, 5, 8, 10, 15, 20]
REGIME_LAGS = [0, 1, 3, 5]
VR_GRID = [0.75, 0.80, 0.85, 0.90, 0.95]
SP_GRID = [9, 10, 11, 12, 13]

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s2c_validation"

gate_mask = C.gate_mask
build_market_regime = C.build_market_regime
_base_entries = C.base_entries
bootstrap_ci = C.block_bootstrap_mean


def load_data():
    """加载数据。补上本文件专用的 ctx 键（core 不提供，因为只有随机补位 MC 需要）。"""
    ctx = C.load_context()
    ctx["oos_dates"] = ctx["eval_dates"]
    ctx["oos_di"] = ctx["eval_di"]
    # 每个交易日「面板上有报价的标的」索引，供 run_monte_carlo 随机补位抽样
    close_mx, dt2i = ctx["close_mx"], ctx["dt2i"]
    ctx["panel_stocks_by_date"] = {
        d: np.where(~np.isnan(close_mx[:, dt2i[d]]))[0] for d in ctx["eval_dates"]
    }
    return ctx


def simulate_detailed(entries_df, ctx, n_slots=N_SLOTS, cost_pct=COST_PCT,
                      cash_daily=0.0, track_details=True):
    """委托 _s2c_core.simulate（passive = 现 S7/F 行为），返回旧的三元组形态。"""
    r = C.simulate(entries_df, ctx, policy="passive", n_slots=n_slots,
                   cost_pct=cost_pct, cash_daily=cash_daily, track_details=track_details)
    trades = r["trades"]
    for t in trades:
        # 兼容垫片（只服务本文件下游，不污染 core）：
        #   score —— 首版 trade 记录里写了 score，但入场表已把 score 重命名成 priority，
        #            row.get("score") 永远取不到 → 首版该列实际全是 NaN。这里沿用 priority
        #            的原值（减去核心层 +100 的排序加成后即原 score）。
        #   exit_dt —— 首版存的是"解析后"的计划退出日；core 的 meta 里可能是 NaT，回落到实际退出。
        if "score" not in t:
            pr = t.get("priority")
            t["score"] = (pr - 100.0) if (pr is not None and t.get("layer") == C.LAYER_CORE) else pr
        if t.get("exit_dt") is None or pd.isna(t.get("exit_dt")):
            t["exit_dt"] = t.get("actual_exit_dt")
    return trades, r["rejected"], r["nav"]


def simulate_fast_nav(entries_df, ctx, n_slots=N_SLOTS, cash_daily=0.0, cost_pct=COST_PCT):
    r = C.simulate(entries_df, ctx, policy="passive", n_slots=n_slots, cost_pct=cost_pct,
                   cash_daily=cash_daily, track_details=False)
    return np.cumprod(1 + r["nav"]["daily_ret"].values)


def compute_metrics(nav_df, label="", rf_daily=0.0, slots=N_SLOTS):
    return C.compute_metrics(nav_df, label=label, rf_daily=rf_daily, slots=slots)


def compute_metrics_from_nav(nav_arr, oos_dates, label=""):
    rets = np.diff(nav_arr, prepend=1.0) / np.concatenate([[1.0], nav_arr[:-1]])
    nav_df = pd.DataFrame({"dt": list(oos_dates)[:len(rets)], "daily_ret": rets, "n_pos": np.nan})
    m = C.compute_metrics(nav_df, label=label)
    return {k: m[k] for k in ("label", "sharpe", "cagr_pct", "max_dd_pct", "calmar",
                              "total_ret_pct") if k in m}


def deflated_sharpe(sharpe_obs, n_trials, n_obs, skew=0, kurt=3):
    """默认改用修正版（首版量纲有误）。kurt 入参沿用旧的"超额峰度"含义。"""
    return C.deflated_sharpe(sharpe_obs, n_trials, n_obs, skew=skew, kurt_excess=kurt)


def build_a(d0, s2b_mask):
    return C.build_core(d0, s2b_mask)


def build_b(d0, s2b_mask, s2c_mask, mkt_map):
    core = C.build_core(d0, s2b_mask, priority_boost=100.0)
    sat = C.build_satellite(d0, s2b_mask, s2c_mask, mkt_map=mkt_map, regime="bull")
    return pd.concat([core, sat], ignore_index=True)


def build_c(d0, s2b_mask, s2c_mask, mkt_map):
    return C.build_satellite(d0, s2b_mask, s2c_mask, mkt_map=mkt_map, regime="bull")


def build_e(d0, s2b_mask):
    return build_a(d0, s2b_mask)


def build_f(d0, s2b_mask, s2c_mask):
    core = C.build_core(d0, s2b_mask, priority_boost=100.0)
    sat = C.build_satellite(d0, s2b_mask, s2c_mask)
    return pd.concat([core, sat], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Attribution
# ═══════════════════════════════════════════════════════════════════════════════

def compute_attribution(trades_a, trades_b, rejected_b, d0, s2b_mask):
    ta = pd.DataFrame(trades_a) if trades_a else pd.DataFrame()
    tb = pd.DataFrame(trades_b) if trades_b else pd.DataFrame()
    rb = pd.DataFrame(rejected_b) if rejected_b else pd.DataFrame()

    result = {
        "s2b_executed_in_a": len(ta),
        "total_executed_in_b": len(tb),
        "core_s2b_in_b": 0, "satellite_s2c_in_b": 0,
        "missed_s2b_count": 0, "missed_s2b_trades": [],
        "s2c_no_conflict": 0, "s2c_occupied_before_s2b": 0,
    }
    if tb.empty:
        return result

    result["core_s2b_in_b"] = int((tb["layer"] == "CORE_S2B").sum())
    result["satellite_s2c_in_b"] = int((tb["layer"] == "SATELLITE_S2C").sum())

    if not ta.empty and not tb.empty:
        a_keys = set(zip(ta["symbol"], ta["entry_dt"], strict=True))
        b_keys = set(zip(tb["symbol"], tb["entry_dt"], strict=True))
        missed_keys = a_keys - b_keys
        result["missed_s2b_count"] = len(missed_keys)

        if missed_keys and not rb.empty:
            missed_list = []
            for sym, edt in missed_keys:
                rej_row = rb[(rb["symbol"] == sym) & (rb["dt"] == edt)]
                reason = rej_row["reason"].iloc[0] if len(rej_row) > 0 else "unknown"
                a_trade = ta[(ta["symbol"] == sym) & (ta["entry_dt"] == edt)]
                ret = a_trade["ret_gross_pct"].iloc[0] if len(a_trade) > 0 and "ret_gross_pct" in a_trade.columns else np.nan
                missed_list.append({"symbol": sym, "entry_dt": str(edt)[:10],
                                    "reason": reason, "ret_gross_pct": round(ret, 2) if not np.isnan(ret) else None})
            result["missed_s2b_trades"] = missed_list

    if not rb.empty:
        s2b_rej = rb[rb["layer"] == "CORE_S2B"]
        conflict = 0
        for _, rr in s2b_rej.iterrows():
            if rr["reason"] == "slot_full":
                held_s2c_on_date = tb[(tb["layer"] == "SATELLITE_S2C") &
                                      (tb["entry_dt"] <= rr["dt"]) &
                                      (tb["exit_dt"] >= rr["dt"])]
                if len(held_s2c_on_date) > 0:
                    conflict += 1
        result["s2c_occupied_before_s2b"] = conflict
        result["s2c_no_conflict"] = result["satellite_s2c_in_b"] - conflict

    return result


def compute_marginal_contribution(trades_b, trades_a, s2b_avg_ret):
    tb = pd.DataFrame(trades_b) if trades_b else pd.DataFrame()
    ta = pd.DataFrame(trades_a) if trades_a else pd.DataFrame()

    s2c_trades = tb[tb["layer"] == "SATELLITE_S2C"] if not tb.empty else pd.DataFrame()
    if s2c_trades.empty:
        return {"s2c_realized_pct": 0, "n_s2c": 0}

    n_s2c = len(s2c_trades)
    s2c_rets = s2c_trades["ret_gross_pct"].dropna()
    s2c_realized = s2c_rets.sum()
    s2c_cost = n_s2c * TOTAL_COST_PCT

    a_keys = set(zip(ta["symbol"], ta["entry_dt"], strict=True)) if not ta.empty else set()
    b_keys = set(zip(tb["symbol"], tb["entry_dt"], strict=True)) if not tb.empty else set()
    missed_keys = a_keys - b_keys

    opp_cost_actual = 0.0
    opp_cost_avg = len(missed_keys) * s2b_avg_ret
    opp_cost_best = 0.0
    for sym, edt in missed_keys:
        a_row = ta[(ta["symbol"] == sym) & (ta["entry_dt"] == edt)]
        if not a_row.empty and "ret_gross_pct" in a_row.columns:
            r = a_row["ret_gross_pct"].iloc[0]
            if not np.isnan(r):
                opp_cost_actual += r
                opp_cost_best = max(opp_cost_best, r)

    return {
        "n_s2c": n_s2c,
        "s2c_realized_pct": round(s2c_realized, 2),
        "s2c_cost_pct": round(s2c_cost, 2),
        "missed_s2b_count": len(missed_keys),
        "opp_cost_actual_pct": round(opp_cost_actual, 2),
        "opp_cost_avg_pct": round(opp_cost_avg, 2),
        "opp_cost_best_pct": round(opp_cost_best, 2),
        "net_marginal_actual": round(s2c_realized - s2c_cost - opp_cost_actual, 2),
        "net_marginal_avg": round(s2c_realized - s2c_cost - opp_cost_avg, 2),
        "net_marginal_best": round(s2c_realized - s2c_cost - opp_cost_best, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# S2c Quality Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_s2c_quality(trades_b, ctx, mkt_map):
    tb = pd.DataFrame(trades_b) if trades_b else pd.DataFrame()
    s2c = tb[tb["layer"] == "SATELLITE_S2C"].copy() if not tb.empty else pd.DataFrame()

    if s2c.empty or len(s2c) < 3:
        return {"error": "insufficient S2C trades", "n": len(s2c)}

    rets = s2c["ret_gross_pct"].dropna()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    basic = {
        "n_trades": len(s2c),
        "mean_ret": round(rets.mean(), 2),
        "median_ret": round(rets.median(), 2),
        "std_ret": round(rets.std(), 2),
        "win_rate": round(len(wins) / len(rets) * 100, 1) if len(rets) > 0 else 0,
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "win_loss_ratio": round(abs(wins.mean() / losses.mean()), 2) if len(losses) > 0 and losses.mean() != 0 else np.inf,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if losses.sum() != 0 else np.inf,
        "max_win": round(rets.max(), 2),
        "max_loss": round(rets.min(), 2),
    }

    sorted_rets = rets.sort_values(ascending=False)
    n = len(sorted_rets)
    total = sorted_rets.sum()
    top5_contrib = sorted_rets.iloc[:max(1, int(n * 0.05))].sum() / total * 100 if total != 0 else 0
    top10_contrib = sorted_rets.iloc[:max(1, int(n * 0.10))].sum() / total * 100 if total != 0 else 0
    ex_top1 = sorted_rets.iloc[max(1, int(n * 0.01)):].mean()
    ex_top5 = sorted_rets.iloc[max(1, int(n * 0.05)):].mean()
    ex_top10 = sorted_rets.iloc[max(1, int(n * 0.10)):].mean()

    tail = {
        "top5pct_contrib_pct": round(top5_contrib, 1),
        "top10pct_contrib_pct": round(top10_contrib, 1),
        "mean_ex_top1pct": round(ex_top1, 2),
        "mean_ex_top5pct": round(ex_top5, 2),
        "mean_ex_top10pct": round(ex_top10, 2),
    }

    fwd_windows = [1, 3, 5, 10, 20]
    multi_period = {}
    for w in fwd_windows:
        fwd_rets = []
        for _, row in s2c.iterrows():
            si = ctx["sym2i"].get(row["symbol"])
            di = ctx["dt2i"].get(row["entry_dt"])
            if si is None or di is None:
                continue
            fwd_di = di + w
            if fwd_di >= len(ctx["dates"]):
                continue
            c0 = ctx["close_mx"][si, di]
            c1 = ctx["close_mx"][si, fwd_di]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                fwd_rets.append((c1 / c0 - 1) * 100)
        if fwd_rets:
            arr = np.array(fwd_rets)
            multi_period[f"fwd_{w}d"] = {
                "mean": round(arr.mean(), 2),
                "median": round(np.median(arr), 2),
                "win_rate": round((arr > 0).mean() * 100, 1),
                "n": len(arr),
            }

    strat_layers = {}
    for col, bins, labels in [
        ("sig_ma_spread_pct", [9.999, 11, 12, 100], ["sp10-11", "sp11-12", "sp12+"]),
        ("sig_vol_ratio", [0.0, 0.80, 0.85, 0.9001], ["vr<=0.80", "vr0.80-0.85", "vr0.85-0.90"]),
    ]:
        if col in s2c.columns:
            s2c["_bin"] = pd.cut(s2c[col], bins=bins, labels=labels, right=True)
            for lbl in labels:
                sub = s2c[s2c["_bin"] == lbl]
                if len(sub) > 0:
                    sr = sub["ret_gross_pct"].dropna()
                    strat_layers[f"{col}={lbl}"] = {
                        "n": len(sr), "mean": round(sr.mean(), 2),
                        "win_rate": round((sr > 0).mean() * 100, 1),
                    }
            s2c.drop(columns=["_bin"], inplace=True, errors="ignore")

    s2c["market"] = s2c["entry_dt"].map(mkt_map)
    regime_breakdown = {}
    for regime in ["bull", "bear", "sideways"]:
        sub = s2c[s2c["market"] == regime]
        if len(sub) > 0:
            sr = sub["ret_gross_pct"].dropna()
            regime_breakdown[regime] = {"n": len(sr), "mean": round(sr.mean(), 2),
                                        "win_rate": round((sr > 0).mean() * 100, 1)}

    yearly = {}
    s2c["year"] = pd.to_datetime(s2c["entry_dt"]).dt.year
    for yr, grp in s2c.groupby("year"):
        sr = grp["ret_gross_pct"].dropna()
        yearly[str(yr)] = {"n": len(sr), "mean": round(sr.mean(), 2),
                           "win_rate": round((sr > 0).mean() * 100, 1)}

    if "amount_e" in s2c.columns:
        q33 = s2c["amount_e"].quantile(0.33)
        q66 = s2c["amount_e"].quantile(0.66)
        for lbl, mask in [("small", s2c["amount_e"] <= q33),
                          ("mid", (s2c["amount_e"] > q33) & (s2c["amount_e"] <= q66)),
                          ("large", s2c["amount_e"] > q66)]:
            sub = s2c[mask]
            if len(sub) > 0:
                sr = sub["ret_gross_pct"].dropna()
                strat_layers[f"size={lbl}"] = {"n": len(sr), "mean": round(sr.mean(), 2),
                                                "win_rate": round((sr > 0).mean() * 100, 1)}

    return {
        "basic": basic, "tail_dependency": tail, "multi_period": multi_period,
        "stratification": strat_layers, "regime_breakdown": regime_breakdown,
        "yearly": yearly,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Random Fill
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(ctx, entries_a, trades_b, mkt_map, n_iter=MC_N, n_slots=N_SLOTS):
    print(f"[MC] 蒙特卡洛随机补位 × {n_iter} ...")
    tb = pd.DataFrame(trades_b) if trades_b else pd.DataFrame()
    s2c_fills = tb[tb["layer"] == "SATELLITE_S2C"].copy() if not tb.empty else pd.DataFrame()

    if s2c_fills.empty:
        print("  无 S2C 补位，跳过 MC")
        return {"skipped": True, "reason": "no_s2c_fills"}

    fill_schedule = []
    for dt, grp in s2c_fills.groupby("entry_dt"):
        for _, row in grp.iterrows():
            exit_dt = row.get("exit_dt", dt + pd.Timedelta(days=22))
            fill_schedule.append({"dt": dt, "exit_dt": exit_dt})

    sym2i = ctx["sym2i"]
    symbols = ctx["symbols"]
    oos_dates = ctx["oos_dates"]
    panel_stocks = ctx["panel_stocks_by_date"]

    s2b_si_set = set()
    for _, row in entries_a.iterrows():
        si = sym2i.get(row["symbol"])
        if si is not None:
            s2b_si_set.add(si)

    mc_results = []
    rng = np.random.default_rng(42)

    for it in range(n_iter):
        random_entries = []
        for fill in fill_schedule:
            dt = fill["dt"]
            available = panel_stocks.get(dt, np.array([]))
            if len(available) == 0:
                continue
            pool = np.setdiff1d(available, list(s2b_si_set))
            if len(pool) == 0:
                continue
            si = rng.choice(pool)
            random_entries.append({
                "symbol": symbols[si], "dt": dt, "priority": 0.0,
                "layer": "RANDOM", "exit_dt": fill["exit_dt"],
            })
        random_df = pd.DataFrame(random_entries)
        combined = pd.concat([entries_a, random_df], ignore_index=True)
        nav = simulate_fast_nav(combined, ctx, n_slots=n_slots)
        m = compute_metrics_from_nav(nav, oos_dates, label=f"MC_{it}")
        mc_results.append(m)

        if (it + 1) % 100 == 0:
            print(f"  MC {it+1}/{n_iter} done")

    mc_df = pd.DataFrame(mc_results)
    return {
        "n_iter": n_iter,
        "sharpe_mean": round(mc_df["sharpe"].mean(), 3),
        "sharpe_std": round(mc_df["sharpe"].std(), 3),
        "sharpe_p5": round(mc_df["sharpe"].quantile(0.05), 3),
        "sharpe_p95": round(mc_df["sharpe"].quantile(0.95), 3),
        "cagr_mean": round(mc_df["cagr_pct"].mean(), 2),
        "max_dd_mean": round(mc_df["max_dd_pct"].mean(), 2),
        "calmar_mean": round(mc_df["calmar"].mean(), 3),
        "mc_df": mc_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness: Cost Stress
# ═══════════════════════════════════════════════════════════════════════════════

def run_cost_stress(ctx, entries_a, entries_b):
    print("[压力] 成本敏感性 ...")
    results = []
    # 首版把成本摊成"每日固定拖累"（两臂同一个数），换手差异被抹平 → 边际 Sharpe 对成本
    # 几乎不敏感，正是成本敏感性检验最不该有的性质。现在按**逐笔**扣（卫星换手更高会更疼）。
    # COST_STRESS_BPS 沿用首版语义：单边 bps，往返 pct = bps * 2 / 100。
    for cost_bps in COST_STRESS_BPS:
        cost_pct = cost_bps * 2 / 100.0
        for label, entries in [("A_S2bOnly", entries_a), ("B_S7", entries_b)]:
            nav = simulate_fast_nav(entries, ctx, n_slots=N_SLOTS, cost_pct=cost_pct)
            m = compute_metrics_from_nav(nav, ctx["oos_dates"], label=f"{label}_cost{cost_bps}bp")
            m["cost_bps"] = cost_bps
            m["strategy"] = label
            results.append(m)

    marginal_sharpe = []
    for cost_bps in COST_STRESS_BPS:
        a_m = next(r for r in results if r["strategy"] == "A_S2bOnly" and r["cost_bps"] == cost_bps)
        b_m = next(r for r in results if r["strategy"] == "B_S7" and r["cost_bps"] == cost_bps)
        marginal_sharpe.append({
            "cost_bps": cost_bps,
            "a_sharpe": a_m["sharpe"], "b_sharpe": b_m["sharpe"],
            "s2c_marginal_sharpe": round(b_m["sharpe"] - a_m["sharpe"], 3),
            "a_cagr": a_m["cagr_pct"], "b_cagr": b_m["cagr_pct"],
            "s2c_marginal_cagr": round(b_m["cagr_pct"] - a_m["cagr_pct"], 2),
        })

    breakeven = None
    for ms in marginal_sharpe:
        if ms["s2c_marginal_cagr"] <= 0:
            breakeven = ms["cost_bps"]
            break

    return {"details": results, "marginal": marginal_sharpe, "breakeven_cost_bps": breakeven}


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness: Slot Stress
# ═══════════════════════════════════════════════════════════════════════════════

def run_slot_stress(ctx, entries_a, entries_b):
    print("[压力] 槽位敏感性 ...")
    results = []
    for ns in SLOT_STRESS:
        for label, entries in [("A_S2bOnly", entries_a), ("B_S7", entries_b)]:
            nav = simulate_fast_nav(entries, ctx, n_slots=ns)
            m = compute_metrics_from_nav(nav, ctx["oos_dates"], label=f"{label}_slots{ns}")
            m["n_slots"] = ns
            m["strategy"] = label
            results.append(m)

    marginal = []
    for ns in SLOT_STRESS:
        a_m = next(r for r in results if r["strategy"] == "A_S2bOnly" and r["n_slots"] == ns)
        b_m = next(r for r in results if r["strategy"] == "B_S7" and r["n_slots"] == ns)
        marginal.append({
            "n_slots": ns,
            "s2c_marginal_sharpe": round(b_m["sharpe"] - a_m["sharpe"], 3),
            "s2c_marginal_cagr": round(b_m["cagr_pct"] - a_m["cagr_pct"], 2),
            "s2c_marginal_dd": round(b_m["max_dd_pct"] - a_m["max_dd_pct"], 2),
        })

    return {"details": results, "marginal": marginal}


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness: Regime Label Lag
# ═══════════════════════════════════════════════════════════════════════════════

def run_regime_lag(ctx, d0, s2b_mask, s2c_mask):
    print("[压力] 牛市标签延迟敏感性 ...")
    results = []
    for lag in REGIME_LAGS:
        mkt = build_market_regime(ctx, lag_days=lag)
        entries = build_b(d0, s2b_mask, s2c_mask, mkt)
        nav = simulate_fast_nav(entries, ctx, n_slots=N_SLOTS)
        m = compute_metrics_from_nav(nav, ctx["oos_dates"], label=f"S7_lag{lag}d")
        m["lag_days"] = lag
        n_sat = len(entries[entries["layer"] == "SATELLITE_S2C"])
        m["n_s2c_entries"] = n_sat
        results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness: Parameter Neighborhood
# ═══════════════════════════════════════════════════════════════════════════════

def run_param_neighborhood(ctx, d0, s2b_mask, mkt_map):
    print("[压力] 参数邻域稳定性 (5×5) ...")
    results = []
    for vr in VR_GRID:
        for sp in SP_GRID:
            s2c_m = gate_mask(d0, vr, sp)
            incr_m = s2c_m & ~s2b_mask
            sat = _base_entries(d0, incr_m, "SATELLITE_S2C")
            sat["market"] = sat["dt"].map(mkt_map)
            sat_bull = sat[sat["market"] == "bull"].drop(columns=["market"])

            core = _base_entries(d0, s2b_mask, "CORE_S2B")
            core["priority"] = core["priority"] + 100
            combined = pd.concat([core, sat_bull], ignore_index=True)

            nav = simulate_fast_nav(combined, ctx, n_slots=N_SLOTS)
            m = compute_metrics_from_nav(nav, ctx["oos_dates"], label=f"vr{vr}_sp{sp}")
            m["vr_th"] = vr
            m["sp_th"] = sp
            m["n_increment"] = len(sat_bull)
            results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════

# bootstrap_ci / deflated_sharpe 已上移为 _s2c_core 委托（见文件头部），此处不再重复定义


def compute_mc_percentile(s7_metric, mc_df, metric_col):
    mc_vals = mc_df[metric_col].values
    pct = (mc_vals < s7_metric).mean() * 100
    return {"s7_value": round(s7_metric, 3), "mc_mean": round(mc_vals.mean(), 3),
            "mc_std": round(mc_vals.std(), 3), "percentile": round(pct, 1)}


# ═══════════════════════════════════════════════════════════════════════════════
# Regime Breakdown (for daily NAV)
# ═══════════════════════════════════════════════════════════════════════════════

def regime_breakdown(nav_df, mkt_map):
    nav_df = nav_df.copy()
    nav_df["regime"] = nav_df["dt"].map(mkt_map)
    results = {}
    for regime in ["bull", "bear", "sideways"]:
        sub = nav_df[nav_df["regime"] == regime]
        if len(sub) > 0:
            rets = sub["daily_ret"].values
            results[regime] = {
                "n_days": len(sub),
                "mean_daily_ret_pct": round(rets.mean() * 100, 4),
                "cum_ret_pct": round(((1 + rets).prod() - 1) * 100, 2),
                "sharpe": round(rets.mean() / rets.std() * np.sqrt(ANN_DAYS), 3) if rets.std() > 0 else 0,
            }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Factor Regression (lightweight)
# ═══════════════════════════════════════════════════════════════════════════════

def lightweight_regression(trades_df, ctx, mkt_map):
    """Trade-level regression using available features as factor proxies."""
    if len(trades_df) < 10:
        return {"error": "insufficient trades for regression"}

    df = trades_df.copy()
    df = df.dropna(subset=["ret_gross_pct"])
    if len(df) < 10:
        return {"error": "insufficient non-null returns"}

    y = df["ret_gross_pct"].values

    eq_close = np.nanmean(ctx["close_mx"], axis=0)
    dt2i = ctx["dt2i"]
    mkt_rets = []
    for _, row in df.iterrows():
        di = dt2i.get(row["entry_dt"])
        if di is not None and di > 0:
            mr = (eq_close[di] / eq_close[di - 1] - 1) * 100
            mkt_rets.append(mr)
        else:
            mkt_rets.append(0.0)
    mkt_rets = np.array(mkt_rets)

    features = {"market_ret": mkt_rets}
    for col in ["sig_ret20", "sig_vol_ratio", "sig_ma_spread_pct", "amount_e"]:
        if col in df.columns:
            vals = df[col].fillna(df[col].median()).values
            if col == "amount_e":
                vals = sp_stats.rankdata(vals) / len(vals)
            features[col] = vals

    X = np.column_stack([np.ones(len(y))] + list(features.values()))
    feat_names = ["intercept"] + list(features.keys())

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        sse = np.sum(resid ** 2)
        mse = sse / (len(y) - len(beta))
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_vals = beta / se
        p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_vals), df=len(y) - len(beta)))

        result = {"r_squared": round(1 - sse / np.sum((y - y.mean()) ** 2), 4), "n": len(y)}
        for i, name in enumerate(feat_names):
            result[name] = {"coef": round(beta[i], 4), "t": round(t_vals[i], 2), "p": round(p_vals[i], 4)}
        result["alpha_pct"] = round(beta[0], 2)
        result["alpha_t"] = round(t_vals[0], 2)
        return result
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)[:10]
    if isinstance(obj, pd.DataFrame):
        return "DataFrame_omitted"
    return str(obj)


def generate_report(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {k: v for k, v in results.items() if k != "mc_df" and k != "nav_dfs"}
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_jsonable)

    if "trade_attribution_df" in results:
        results["trade_attribution_df"].to_csv(output_dir / "trade_attribution.csv", index=False)

    if "missed_s2b_df" in results:
        results["missed_s2b_df"].to_csv(output_dir / "missed_s2b.csv", index=False)

    if "nav_dfs" in results:
        nav_combined = None
        for label, ndf in results["nav_dfs"].items():
            col_df = ndf[["dt"]].copy()
            col_df[f"nav_{label}"] = ndf["nav"]
            col_df[f"ret_{label}"] = ndf["daily_ret"]
            col_df[f"npos_{label}"] = ndf["n_pos"]
            nav_combined = (col_df if nav_combined is None
                            else nav_combined.merge(col_df, on="dt", how="outer"))
        if nav_combined is not None:
            nav_combined.to_csv(output_dir / "daily_nav.csv", index=False)

    if "mc_results" in results and "mc_df" in results["mc_results"]:
        results["mc_results"]["mc_df"].to_csv(output_dir / "random_fill_mc.csv", index=False)

    if "cost_stress" in results:
        pd.DataFrame(results["cost_stress"]["marginal"]).to_csv(output_dir / "cost_stress.csv", index=False)

    if "slot_stress" in results:
        pd.DataFrame(results["slot_stress"]["marginal"]).to_csv(output_dir / "slot_stress.csv", index=False)

    if "param_neighborhood" in results:
        pd.DataFrame(results["param_neighborhood"]).to_csv(output_dir / "param_stability.csv", index=False)

    if "regime_breakdowns" in results:
        rows = []
        for strat, rb in results["regime_breakdowns"].items():
            for regime, data in rb.items():
                rows.append({"strategy": strat, "regime": regime, **data})
        pd.DataFrame(rows).to_csv(output_dir / "regime_breakdown.csv", index=False)

    print(f"\n[输出] 所有文件 → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = load_data()
    d0 = ctx["d0"]

    print("\n[构建] 市场环境 ...")
    mkt_map = build_market_regime(ctx)
    bull_days = sum(1 for v in mkt_map.values() if v == "bull")
    print(f"  牛市: {bull_days} 日")

    print("[构建] 门控掩码 ...")
    s2b_mask = gate_mask(d0, 0.8, 12.0)
    s2c_mask = gate_mask(d0, 0.9, 10.0)
    print(f"  S2b: {s2b_mask.sum()} | S2c: {s2c_mask.sum()} | S2c-only: {(s2c_mask & ~s2b_mask).sum()}")

    # ─── Build strategy entries ────────────────────────────────────
    print("\n[策略] 构建对照组入场信号 ...")
    entries_a = build_a(d0, s2b_mask)
    entries_b = build_b(d0, s2b_mask, s2c_mask, mkt_map)
    entries_c = build_c(d0, s2b_mask, s2c_mask, mkt_map)
    entries_e = build_e(d0, s2b_mask)
    entries_f = build_f(d0, s2b_mask, s2c_mask)
    print(f"  A(S2b): {len(entries_a)} | B(S7): {len(entries_b)} | C(S2c): {len(entries_c)} "
          f"| E(S2b+Cash): {len(entries_e)} | F(S2b+RelaxNoBull): {len(entries_f)}")

    # ─── Run detailed simulations ─────────────────────────────────
    print("\n[模拟] 运行组合模拟 ...")
    results = {}
    nav_dfs = {}

    for label, entries, cash in [
        ("A_S2bOnly", entries_a, 0.0),
        ("B_S7", entries_b, 0.0),
        ("C_S2cOnly", entries_c, 0.0),
        ("E_S2bCash", entries_e, CASH_RATE_DAILY),
        ("F_S2bRelaxed", entries_f, 0.0),
    ]:
        print(f"  [{label}] ...", end=" ")
        trades, rejected, nav_df = simulate_detailed(entries, ctx, cash_daily=cash)
        metrics = compute_metrics(nav_df, label=label)
        print(f"trades={len(trades)} | Sharpe={metrics['sharpe']} | CAGR={metrics['cagr_pct']}% "
              f"| MaxDD={metrics['max_dd_pct']}%")
        results[f"metrics_{label}"] = metrics
        results[f"trades_{label}"] = trades
        results[f"rejected_{label}"] = rejected
        nav_dfs[label] = nav_df

    results["nav_dfs"] = nav_dfs

    # ─── Trade Attribution ─────────────────────────────────────────
    print("\n[归因] 交易归因分析 ...")
    attribution = compute_attribution(
        results["trades_A_S2bOnly"], results["trades_B_S7"],
        results["rejected_B_S7"], d0, s2b_mask
    )
    results["attribution"] = attribution
    print(f"  S2b in A: {attribution['s2b_executed_in_a']} | S2b in B: {attribution['core_s2b_in_b']} "
          f"| S2c in B: {attribution['satellite_s2c_in_b']} | Missed S2b: {attribution['missed_s2b_count']}")

    tb = pd.DataFrame(results["trades_B_S7"])
    if not tb.empty:
        results["trade_attribution_df"] = tb[["symbol", "entry_dt", "exit_dt", "layer",
            "ret_gross_pct", "hold_days", "exit_reason", "score",
            "sig_vol_ratio", "sig_ma_spread_pct", "amount_e"]].copy()

    if attribution["missed_s2b_trades"]:
        results["missed_s2b_df"] = pd.DataFrame(attribution["missed_s2b_trades"])

    # ─── Marginal Contribution ─────────────────────────────────────
    print("[归因] S2c 净边际贡献 ...")
    s2b_avg = 0.0
    ta = pd.DataFrame(results["trades_A_S2bOnly"])
    if not ta.empty and "ret_gross_pct" in ta.columns:
        s2b_avg = ta["ret_gross_pct"].dropna().mean()
    marginal = compute_marginal_contribution(results["trades_B_S7"], results["trades_A_S2bOnly"], s2b_avg)
    results["marginal_contribution"] = marginal
    print(f"  S2c 已实现: {marginal['s2c_realized_pct']}% | 成本: {marginal['s2c_cost_pct']}% "
          f"| 机会成本(实际): {marginal['opp_cost_actual_pct']}% | 净贡献(实际): {marginal['net_marginal_actual']}%")

    # ─── S2c Quality ───────────────────────────────────────────────
    print("\n[质量] S2c 独有交易分析 ...")
    s2c_quality = analyze_s2c_quality(results["trades_B_S7"], ctx, mkt_map)
    results["s2c_quality"] = s2c_quality
    if "basic" in s2c_quality:
        b = s2c_quality["basic"]
        print(f"  n={b['n_trades']} | mean={b['mean_ret']}% | wr={b['win_rate']}% "
              f"| PF={b['profit_factor']} | W/L={b['win_loss_ratio']}")

    # ─── Factor Regression ─────────────────────────────────────────
    print("[回归] 轻量因子回归 ...")
    s2c_trades_df = pd.DataFrame(results["trades_B_S7"])
    if not s2c_trades_df.empty:
        s2c_only = s2c_trades_df[s2c_trades_df["layer"] == "SATELLITE_S2C"]
        regression = lightweight_regression(s2c_only, ctx, mkt_map)
        results["factor_regression"] = regression
        if "alpha_pct" in regression:
            print(f"  Alpha: {regression['alpha_pct']}% (t={regression['alpha_t']}) | R²={regression.get('r_squared', 'N/A')}")

    # ─── Monte Carlo Random Fill ───────────────────────────────────
    print()
    mc_results = run_monte_carlo(ctx, entries_a, results["trades_B_S7"], mkt_map)
    results["mc_results"] = mc_results

    if "mc_df" in mc_results:
        mc_df = mc_results["mc_df"]
        b_metrics = results["metrics_B_S7"]
        results["mc_percentile"] = {
            "sharpe": compute_mc_percentile(b_metrics["sharpe"], mc_df, "sharpe"),
            "cagr": compute_mc_percentile(b_metrics["cagr_pct"], mc_df, "cagr_pct"),
            "max_dd": compute_mc_percentile(b_metrics["max_dd_pct"], mc_df, "max_dd_pct"),
            "calmar": compute_mc_percentile(b_metrics["calmar"], mc_df, "calmar"),
        }
        print(f"  S7 Sharpe 百分位: {results['mc_percentile']['sharpe']['percentile']}% | "
              f"CAGR 百分位: {results['mc_percentile']['cagr']['percentile']}%")

    # ─── Regime Breakdown ──────────────────────────────────────────
    print("\n[分层] 市场状态分解 ...")
    regime_bds = {}
    for label, nav_df in nav_dfs.items():
        regime_bds[label] = regime_breakdown(nav_df, mkt_map)
    results["regime_breakdowns"] = regime_bds

    # ─── Robustness Tests ──────────────────────────────────────────
    print()
    results["cost_stress"] = run_cost_stress(ctx, entries_a, entries_b)
    results["slot_stress"] = run_slot_stress(ctx, entries_a, entries_b)
    results["regime_lag"] = run_regime_lag(ctx, d0, s2b_mask, s2c_mask)
    results["param_neighborhood"] = run_param_neighborhood(ctx, d0, s2b_mask, mkt_map)

    # ─── Statistical Tests ─────────────────────────────────────────
    print("\n[统计] Bootstrap CI + Deflated Sharpe ...")
    s2c_trades = pd.DataFrame(results["trades_B_S7"])
    s2c_rets = s2c_trades.loc[s2c_trades["layer"] == "SATELLITE_S2C", "ret_gross_pct"].dropna().values if not s2c_trades.empty else np.array([])

    if len(s2c_rets) > 5:
        results["bootstrap_s2c_mean"] = bootstrap_ci(s2c_rets)
        print(f"  S2c mean: {results['bootstrap_s2c_mean']['mean']}% "
              f"[{results['bootstrap_s2c_mean']['ci_lo']}, {results['bootstrap_s2c_mean']['ci_hi']}]")
    else:
        results["bootstrap_s2c_mean"] = {"error": "insufficient S2C trades"}

    b_sharpe = results["metrics_B_S7"]["sharpe"]
    n_obs = results["metrics_B_S7"]["n_days"]
    skew = results["metrics_B_S7"]["skewness"]
    kurt = results["metrics_B_S7"]["kurtosis"] + 3
    results["deflated_sharpe"] = deflated_sharpe(b_sharpe, n_trials=77, n_obs=n_obs, skew=skew, kurt=kurt)
    print(f"  Deflated Sharpe: DSR={results['deflated_sharpe']['dsr']} (p={results['deflated_sharpe']['p_value']})")

    # ─── Executive Summary ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("                         S2c 增量验证 — 执行摘要")
    print("=" * 100)

    a_m = results["metrics_A_S2bOnly"]
    b_m = results["metrics_B_S7"]
    c_m = results["metrics_C_S2cOnly"]
    e_m = results["metrics_E_S2bCash"]
    f_m = results["metrics_F_S2bRelaxed"]

    header = f"{'指标':<25} {'A:S2bOnly':>12} {'B:S7':>12} {'C:S2cOnly':>12} {'E:S2b+Cash':>12} {'F:Relaxed':>12}"
    print(header)
    print("-" * 100)
    for metric, fmt in [
        ("cagr_pct", ".2f"), ("sharpe", ".3f"), ("sortino", ".3f"), ("calmar", ".3f"),
        ("max_dd_pct", ".2f"), ("ann_vol_pct", ".2f"), ("var_95_pct", ".3f"),
        ("cvar_95_pct", ".3f"), ("worst_month_pct", ".2f"), ("max_consec_loss", "d"),
        ("avg_positions", ".1f"), ("utilization_pct", ".1f"),
    ]:
        vals = [a_m.get(metric, 0), b_m.get(metric, 0), c_m.get(metric, 0),
                e_m.get(metric, 0), f_m.get(metric, 0)]
        line = f"{metric:<25}"
        for v in vals:
            line += f" {v:>12{fmt}}" if fmt != "d" else f" {int(v):>12d}"
        print(line)

    print("\n--- S2c 边际贡献 ---")
    print(f"  S2c 已实现总收益: {marginal['s2c_realized_pct']}%")
    print(f"  S2c 交易成本: {marginal['s2c_cost_pct']}%")
    print(f"  错失 S2b 机会成本(实际): {marginal['opp_cost_actual_pct']}%")
    print(f"  净边际贡献(实际口径): {marginal['net_marginal_actual']}%")
    print(f"  净边际贡献(均值口径): {marginal['net_marginal_avg']}%")

    if "mc_percentile" in results:
        mp = results["mc_percentile"]
        print(f"\n--- 随机补位对照 (MC={mc_results.get('n_iter', 0)}) ---")
        print(f"  S7 Sharpe 超过随机补位: {mp['sharpe']['percentile']}%")
        print(f"  S7 CAGR 超过随机补位: {mp['cagr']['percentile']}%")

    print("\n--- 稳健性 ---")
    if results.get("cost_stress", {}).get("breakeven_cost_bps"):
        print(f"  成本盈亏平衡点: {results['cost_stress']['breakeven_cost_bps']} bp (单边)")
    else:
        print("  成本盈亏平衡点: 在测试范围内 S2c 始终有效")

    if results.get("regime_lag"):
        for lag_r in results["regime_lag"]:
            print(f"  标签延迟 {lag_r['lag_days']}日: Sharpe={lag_r['sharpe']} CAGR={lag_r['cagr_pct']}% S2c数={lag_r['n_s2c_entries']}")

    if "deflated_sharpe" in results:
        ds = results["deflated_sharpe"]
        print("\n--- 多重检验 ---")
        print(f"  Deflated Sharpe: {ds['dsr']} (p={ds['p_value']}, 搜索过 {ds['n_trials']} 个配置)")

    # ─── Final Judgment ────────────────────────────────────────────
    verdict_score = 0
    reasons = []

    if b_m["cagr_pct"] > a_m["cagr_pct"]:
        verdict_score += 1
        reasons.append(f"CAGR: B({b_m['cagr_pct']}%) > A({a_m['cagr_pct']}%)")
    else:
        reasons.append(f"CAGR: B({b_m['cagr_pct']}%) <= A({a_m['cagr_pct']}%)")

    if b_m["sharpe"] >= a_m["sharpe"] - 0.05:
        verdict_score += 1
        reasons.append(f"Sharpe: B({b_m['sharpe']}) >= A({a_m['sharpe']})-0.05")
    else:
        reasons.append(f"Sharpe: B({b_m['sharpe']}) < A({a_m['sharpe']})-0.05")

    if abs(b_m["max_dd_pct"]) <= abs(a_m["max_dd_pct"]) * 1.2:
        verdict_score += 1
        reasons.append(f"MaxDD: B({b_m['max_dd_pct']}%) 未显著恶化 (A={a_m['max_dd_pct']}%)")
    else:
        reasons.append(f"MaxDD: B({b_m['max_dd_pct']}%) 显著恶化 (A={a_m['max_dd_pct']}%)")

    if marginal["net_marginal_actual"] > 0:
        verdict_score += 1
        reasons.append(f"净边际贡献(实际): {marginal['net_marginal_actual']}% > 0")
    else:
        reasons.append(f"净边际贡献(实际): {marginal['net_marginal_actual']}% <= 0")

    if "mc_percentile" in results and results["mc_percentile"]["sharpe"]["percentile"] > 60:
        verdict_score += 1
        reasons.append(f"S7 Sharpe > {results['mc_percentile']['sharpe']['percentile']}% 随机补位")
    elif "mc_percentile" in results:
        reasons.append(f"S7 Sharpe 仅超 {results['mc_percentile']['sharpe']['percentile']}% 随机补位")

    if "bootstrap_s2c_mean" in results and "ci_lo" in results["bootstrap_s2c_mean"]:
        if results["bootstrap_s2c_mean"]["ci_lo"] > 0:
            verdict_score += 1
            reasons.append(f"S2c 均值 Bootstrap CI: [{results['bootstrap_s2c_mean']['ci_lo']}, {results['bootstrap_s2c_mean']['ci_hi']}] > 0")
        else:
            reasons.append(f"S2c 均值 Bootstrap CI 包含 0: [{results['bootstrap_s2c_mean']['ci_lo']}, {results['bootstrap_s2c_mean']['ci_hi']}]")

    if verdict_score >= 5:
        verdict = "A: S2c 明确补强 S2b"
        confidence = "高"
    elif verdict_score >= 3:
        verdict = "B: S2c 提高仓位利用率，但补强 Alpha 证据有限"
        confidence = "中"
    elif verdict_score >= 1:
        verdict = "C: S2c 稀释 S2b 或证据不足"
        confidence = "低"
    else:
        verdict = "D: 证据不足"
        confidence = "低"

    results["verdict"] = {"judgment": verdict, "confidence": confidence, "score": verdict_score,
                          "max_score": 6, "reasons": reasons}

    print(f"\n{'=' * 100}")
    print(f"  最终判定: {verdict}")
    print(f"  置信度: {confidence} ({verdict_score}/6)")
    print("  依据:")
    for r in reasons:
        print(f"    - {r}")
    print(f"{'=' * 100}")

    # ─── Save ──────────────────────────────────────────────────────
    generate_report(results, OUTPUT_DIR)
    print(f"\n[完成] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
