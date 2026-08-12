"""S2b/S2c/S8 组合验证公共核心

从 s2c_incremental_validation.py 抽出，供 s2c_/s8_incremental_validation.py 共用。

相对原实现的三处修正（均为原实现的 bug，非行为偏好）：
  1. **交易成本未计入 NAV**：原 simulate_detailed 接收 cost_pct 但函数体从未使用它，
     所有 Sharpe/CAGR 都是零成本读数。本模块在开仓/平仓时逐笔扣减。
  2. **deflated_sharpe 量纲错配**：原实现把年化 Sharpe 代入日频 SE 公式，SE 低估 ~sqrt(252)，
     任何策略都被判死。本模块提供 legacy=True 复现旧值以便对齐历史报告。
  3. **exit_dt 解析表达式歧义**：原 `pd.isna(x) if isinstance(x, float) else ...` 三元嵌套
     在 Timestamp 分支上行为不明，改为显式 helper。

槽位模拟器支持三种 policy：
  - passive : 卫星可长期占用槽位，S2b 信号到达无空槽即被拒（= 现 S7/F 行为）
  - preempt : S2b 信号到达无空槽时逐出优先级最低的卫星持仓（S8-a，保证 S2b 集合不变）
  - budget  : 核心池/卫星池各自独立槽位，互不干扰（S8-b，S2b 集合天然不变）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from surge_candidates_dump import completed_outcomes

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = pd.Timestamp("2023-12-31")
N_SLOTS = 10
BUY_COST = 0.0015
SELL_COST = 0.0025
TOTAL_COST_PCT = (BUY_COST + SELL_COST) * 100
LOOKBACK = 30
BULL_TH = 8.0
BEAR_TH = -8.0
ANN_DAYS = 252
CASH_RATE_ANNUAL = 0.02
CASH_RATE_DAILY = CASH_RATE_ANNUAL / ANN_DAYS

LAYER_CORE = "CORE_S2B"
LAYER_SAT = "SATELLITE_S2C"

S2B_VR, S2B_SP = 0.8, 12.0
S2C_VR, S2C_SP = 0.9, 10.0

_SCRIPT_DIR = Path(__file__).resolve().parent
CAND_PATH = _SCRIPT_DIR / "_output" / "surge_candidates" / "candidates.parquet"
PANEL_PATH = _SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"

_ENTRY_COLS = [
    "symbol",
    "entry_dt",
    "exit_dt",
    "score",
    "ret_gross_pct",
    "hold_days",
    "exit_reason",
    "sig_vol_ratio",
    "sig_ma_spread_pct",
    "amount_e",
]

# 随交易记录透传的元数据列（entry 侧原样带出，便于事后分层）
_META_COLS = ["exit_dt", "ret_gross_pct", "hold_days", "exit_reason", "sig_vol_ratio", "sig_ma_spread_pct", "amount_e"]

# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_context(eval_start=None, eval_end=None, verbose=True):
    """加载候选/面板并建成矩阵。eval_start/eval_end 界定评估窗（含端点）。

    eval_start 默认 TRAIN_END 之后第一天；eval_end 默认面板最后一天。
    """
    if verbose:
        print("[加载] 数据 ...")
    cand = completed_outcomes(pd.read_parquet(CAND_PATH))
    panel = pd.read_parquet(PANEL_PATH)
    d0 = cand[cand["delay"] == 0].copy()

    ps = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)
    ps["daily_ret"] = ps.groupby("symbol")["close"].pct_change()

    symbols = sorted(ps["symbol"].unique())
    # 统一成 pd.Timestamp：原实现里 dates 是 np.datetime64 而 groupby 键是 Timestamp，
    # 靠两者 hash 相等才没出错，这里显式归一，不依赖该巧合。
    dates = [pd.Timestamp(d) for d in sorted(ps["dt"].unique())]
    sym2i = {s: i for i, s in enumerate(symbols)}
    dt2i = {d: i for i, d in enumerate(dates)}

    n_sym, n_dt = len(symbols), len(dates)
    close_mx = np.full((n_sym, n_dt), np.nan)
    ret_mx = np.full((n_sym, n_dt), np.nan)
    si = ps["symbol"].map(sym2i).values
    di = ps["dt"].map(dt2i).values
    close_mx[si, di] = ps["close"].values
    ret_mx[si, di] = ps["daily_ret"].values

    ctx = {
        "d0": d0,
        "panel": ps,
        "symbols": symbols,
        "dates": dates,
        "sym2i": sym2i,
        "dt2i": dt2i,
        "close_mx": close_mx,
        "ret_mx": ret_mx,
    }
    set_eval_window(ctx, eval_start, eval_end)
    if verbose:
        print(f"  候选(d=0): {len(d0)} | 矩阵: {n_sym} symbols × {n_dt} dates")
        print(
            f"  评估窗: {str(ctx['eval_dates'][0])[:10]} → {str(ctx['eval_dates'][-1])[:10]}"
            f" ({len(ctx['eval_dates'])} 日)"
        )
    return ctx


def set_eval_window(ctx, eval_start=None, eval_end=None):
    """就地重设评估窗，不重新加载数据（walk-forward 逐折调用）。"""
    dates = ctx["dates"]
    lo = pd.Timestamp(eval_start) if eval_start is not None else TRAIN_END + pd.Timedelta(days=1)
    hi = pd.Timestamp(eval_end) if eval_end is not None else pd.Timestamp(dates[-1])
    eval_dates = [d for d in dates if lo <= d <= hi]
    if len(eval_dates) == 0:
        raise ValueError(f"评估窗为空: {lo} → {hi}")
    ctx["eval_dates"] = eval_dates
    ctx["eval_date_set"] = set(eval_dates)
    ctx["eval_di"] = [ctx["dt2i"][d] for d in eval_dates]
    return ctx


def attach_forward_returns(ctx, windows=(5, 20)):
    """给 d0 附上 fwd_{w}d（按面板 close），供逐折重推门控用。"""
    ps = ctx["panel"]
    cols = []
    for w in windows:
        col = f"fwd_{w}d"
        if col not in ps.columns:
            ps[col] = ps.groupby("symbol")["close"].transform(lambda s, w=w: s.shift(-w) / s - 1) * 100
        cols.append(col)
    d0 = ctx["d0"]
    if all(c in d0.columns for c in cols):
        return ctx
    merged = d0.merge(
        ps[["symbol", "dt"] + cols],
        left_on=["symbol", "entry_dt"],
        right_on=["symbol", "dt"],
        how="left",
        suffixes=("", "_p"),
    )
    merged.drop(columns=[c for c in ("dt", "dt_p") if c in merged.columns], inplace=True)
    ctx["d0"] = merged
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# Market Regime
# ═══════════════════════════════════════════════════════════════════════════════


def build_market_regime(ctx, lag_days=0):
    close_mx, dates = ctx["close_mx"], ctx["dates"]
    eq_close = np.nanmean(close_mx, axis=0)
    regime_map = {}
    for i, d in enumerate(dates):
        if i < LOOKBACK:
            regime_map[d] = "sideways"
        else:
            ret = (eq_close[i] / eq_close[i - LOOKBACK] - 1) * 100
            regime_map[d] = "bull" if ret >= BULL_TH else ("bear" if ret <= BEAR_TH else "sideways")
    if lag_days > 0:
        return {d: regime_map[dates[max(0, i - lag_days)]] for i, d in enumerate(dates)}
    return regime_map


# ═══════════════════════════════════════════════════════════════════════════════
# Gates & Entry Builders
# ═══════════════════════════════════════════════════════════════════════════════


def gate_mask(df, vr_th, sp_th):
    vr, sp = df["sig_vol_ratio"], df["sig_ma_spread_pct"]
    return (vr <= vr_th) & vr.notna() & (sp >= sp_th) & sp.notna()


def base_entries(d0, mask, layer):
    cols = [c for c in _ENTRY_COLS if c in d0.columns]
    df = d0.loc[mask, cols].copy()
    df = df.rename(columns={"entry_dt": "dt", "score": "priority"})
    df["layer"] = layer
    return df


def build_core(d0, s2b_mask, priority_boost=0.0):
    core = base_entries(d0, s2b_mask, LAYER_CORE)
    if priority_boost:
        core["priority"] = core["priority"] + priority_boost
    return core


def build_satellite(d0, s2b_mask, s2c_mask, mkt_map=None, regime=None):
    """卫星集合 = S2c 独有信号（严格排除 S2b）。regime 非 None 时才按市场环境过滤。"""
    sat = base_entries(d0, s2c_mask & ~s2b_mask, LAYER_SAT)
    if regime is not None:
        if mkt_map is None:
            raise ValueError("按 regime 过滤需要 mkt_map")
        sat = sat[sat["dt"].map(mkt_map) == regime].copy()
    return sat


# ═══════════════════════════════════════════════════════════════════════════════
# Slot Simulator
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_exit_di(exit_dt, entry_dt, di, ctx, default_days=22):
    """把 exit_dt 解析成日期索引。缺失时回落到 entry_dt + default_days 之后首个交易日。"""
    if exit_dt is None or exit_dt is pd.NaT or (isinstance(exit_dt, float) and np.isnan(exit_dt)):
        exit_dt = entry_dt + pd.Timedelta(days=default_days)
    exit_dt = pd.Timestamp(exit_dt)
    exit_di = ctx["dt2i"].get(exit_dt)
    if exit_di is not None:
        return exit_di
    pos = np.searchsorted(ctx["dates"], exit_dt, side="left")
    return int(pos) if pos < len(ctx["dates"]) else di + default_days


def simulate(
    entries_df,
    ctx,
    policy="passive",
    n_slots=N_SLOTS,
    n_sat_slots=None,
    cost_pct=TOTAL_COST_PCT,
    cash_daily=0.0,
    track_details=True,
):
    """槽位组合模拟。

    policy:
      passive : 单池，卫星与核心抢同一批槽位（现 S7/F 行为）
      preempt : 单池，核心信号无槽时逐出优先级最低的卫星（S8-a）
      budget  : 双池，核心 n_slots + 卫星 n_sat_slots，互不干扰（S8-b）

    返回 dict：trades / rejected / nav（DataFrame）。budget 下 nav 含
    blend（等资本，核心与卫星按槽位数加权，总资本不变）与
    overlay（加资本，核心保持满仓、卫星额外加资金 → 总敞口 >100%，杠杆读数）两列。
    """
    if policy not in ("passive", "preempt", "budget"):
        raise ValueError(f"未知 policy: {policy}")
    if policy == "budget" and not n_sat_slots:
        raise ValueError("policy=budget 需要 n_sat_slots")

    sym2i, dt2i = ctx["sym2i"], ctx["dt2i"]
    close_mx, ret_mx = ctx["close_mx"], ctx["ret_mx"]
    eval_dates = ctx["eval_dates"]
    # cost_pct 是往返总成本（百分点），按 BUY/SELL 原比例拆分。
    # cost_pct=TOTAL_COST_PCT 时 buy/sell 恰好还原为 BUY_COST/SELL_COST。
    _cf = cost_pct / 100.0
    _rt = BUY_COST + SELL_COST
    buy_frac = _cf * (BUY_COST / _rt)
    sell_frac = _cf * (SELL_COST / _rt)

    pools = {"core": n_slots, "sat": n_sat_slots} if policy == "budget" else {"main": n_slots}
    # 不可用 dict.fromkeys(pools, [])：那样所有池共享同一个 list，核心与卫星账本会被熔在一起
    held = {k: [] for k in pools}  # noqa: C420
    trades, rejected, daily = [], [], []
    nav = dict.fromkeys(pools, 1.0)
    nav_blend = nav_overlay = 1.0

    edf = entries_df.copy()
    if edf.empty:
        edf = pd.DataFrame(columns=["symbol", "dt", "priority", "layer"])
    else:
        edf = edf[edf["dt"].isin(ctx["eval_date_set"])]
        edf = edf.sort_values(["dt", "priority"], ascending=[True, False])
    entries_by_dt = {pd.Timestamp(dt): g for dt, g in edf.groupby("dt")} if not edf.empty else {}

    pending_cost = dict.fromkeys(pools, 0.0)
    total_slots = sum(pools.values())

    def _pool_of(layer):
        if policy != "budget":
            return "main"
        return "core" if layer == LAYER_CORE else "sat"

    def _record(p, exit_di_actual, preempted=False, still_open=False):
        if not track_details:
            return
        c0, ce = p["entry_price"], close_mx[p["si"], min(exit_di_actual, len(ctx["dates"]) - 1)]
        row = dict(p["meta"])
        row.update(
            {
                "symbol": p["symbol"],
                "entry_dt": p["entry_dt"],
                "layer": p["layer"],
                "priority": p["priority"],
                "preempted": preempted,
                "still_open": still_open,
                "actual_exit_dt": ctx["dates"][min(exit_di_actual, len(ctx["dates"]) - 1)],
                "hold_days_actual": exit_di_actual - p["entry_di"],
                "close_ret_pct": ((ce / c0) - 1) * 100 if not np.isnan(ce) and c0 > 0 else np.nan,
                "cost_pct": cost_pct,
            }
        )
        row["close_ret_net_pct"] = row["close_ret_pct"] - cost_pct
        trades.append(row)

    for dt in eval_dates:
        di = dt2i[dt]

        for k in pools:  # exit_di 当日仍持有，次日移除
            keep = []
            for p in held[k]:
                if p["exit_di"] >= di:
                    keep.append(p)
                else:
                    _record(p, p["exit_di"])
            held[k] = keep

        pool_ret = {}
        for k, cap in pools.items():
            ret_sum, cost_sum = 0.0, pending_cost[k]
            pending_cost[k] = 0.0
            for p in held[k]:
                if p["entry_di"] < di:
                    r = ret_mx[p["si"], di]
                    if not np.isnan(r):
                        ret_sum += r
                if p["exit_di"] == di:
                    cost_sum += sell_frac
            n_empty = cap - len(held[k])
            pool_ret[k] = (ret_sum + n_empty * cash_daily - cost_sum) / cap
            nav[k] *= 1 + pool_ret[k]

        n_core = sum(1 for k in held for p in held[k] if p["layer"] == LAYER_CORE)
        n_sat = sum(1 for k in held for p in held[k] if p["layer"] != LAYER_CORE)
        row = {"dt": dt, "n_pos": n_core + n_sat, "n_core": n_core, "n_satellite": n_sat}
        if policy == "budget":
            r_blend = (pools["core"] * pool_ret["core"] + pools["sat"] * pool_ret["sat"]) / total_slots
            r_overlay = pool_ret["core"] + pools["sat"] / pools["core"] * pool_ret["sat"]
            nav_blend *= 1 + r_blend
            nav_overlay *= 1 + r_overlay
            row.update(
                {
                    "nav": nav_blend,
                    "daily_ret": r_blend,
                    "nav_blend": nav_blend,
                    "daily_ret_blend": r_blend,
                    "nav_overlay": nav_overlay,
                    "daily_ret_overlay": r_overlay,
                    "nav_core": nav["core"],
                    "daily_ret_core": pool_ret["core"],
                    "nav_sat": nav["sat"],
                    "daily_ret_sat": pool_ret["sat"],
                }
            )
        else:
            row.update({"nav": nav["main"], "daily_ret": pool_ret["main"]})
        daily.append(row)

        grp = entries_by_dt.get(dt)
        if grp is None:
            continue

        for _, ent in grp.iterrows():
            sym, layer = ent["symbol"], ent.get("layer", LAYER_CORE)
            k = _pool_of(layer)
            si = sym2i.get(sym)
            if si is None:
                continue
            c = close_mx[si, di]
            if np.isnan(c) or c <= 0:
                continue

            # preempt 下核心遇到"该名被卫星占着"也必须让位，否则核心集合就与基准 A 不一致
            # （A 里该名未被持有 → 能进；这里若因卫星占名被拒，P6 不变性即破）
            if policy == "preempt" and layer == LAYER_CORE:
                same_name_sat = [p for p in held[k] if p["layer"] != LAYER_CORE and p["symbol"] == sym]
                for victim in same_name_sat:
                    held[k].remove(victim)
                    _record(victim, di, preempted=True)
                    pending_cost[k] += sell_frac

            # 同名去重：budget 下卫星也不许与核心持仓同名（避免同一标的双份敞口）
            occupied = {p["symbol"] for p in held[k]}
            if policy == "budget" and layer != LAYER_CORE:
                occupied |= {p["symbol"] for p in held["core"]}
            if sym in occupied:
                if track_details:
                    rejected.append(
                        {
                            "symbol": sym,
                            "dt": dt,
                            "layer": layer,
                            "priority": ent["priority"],
                            "reason": "symbol_occupied",
                        }
                    )
                continue

            if len(held[k]) >= pools[k]:
                evicted = False
                if policy == "preempt" and layer == LAYER_CORE:
                    sats = [p for p in held[k] if p["layer"] != LAYER_CORE]
                    if sats:
                        victim = min(sats, key=lambda p: p["priority"])
                        held[k].remove(victim)
                        _record(victim, di, preempted=True)
                        pending_cost[k] += sell_frac
                        evicted = True
                if not evicted:
                    if track_details:
                        rejected.append(
                            {
                                "symbol": sym,
                                "dt": dt,
                                "layer": layer,
                                "priority": ent["priority"],
                                "reason": "slot_full",
                            }
                        )
                    continue

            exit_di = _resolve_exit_di(ent.get("exit_dt"), dt, di, ctx)
            meta = {c: ent.get(c, np.nan) for c in _META_COLS if c in ent.index}
            held[k].append(
                {
                    "symbol": sym,
                    "si": si,
                    "entry_dt": dt,
                    "entry_di": di,
                    "entry_price": c,
                    "exit_di": exit_di,
                    "layer": layer,
                    "priority": ent["priority"],
                    "meta": meta,
                }
            )
            pending_cost[k] += buy_frac

    # 评估窗末仍持有的仓位也要记一笔（否则 trades 数与入场数不一致）
    last_di = ctx["dt2i"][eval_dates[-1]]
    for k in pools:
        for p in held[k]:
            _record(p, min(p["exit_di"], last_di), still_open=True)

    nav_df = pd.DataFrame(daily)
    if nav_df.empty:
        nav_df = pd.DataFrame(
            {"dt": eval_dates, "nav": 1.0, "daily_ret": 0.0, "n_pos": 0, "n_core": 0, "n_satellite": 0}
        )
    return {
        "trades": trades,
        "rejected": rejected,
        "nav": nav_df,
        "policy": policy,
        "n_slots": n_slots,
        "n_sat_slots": n_sat_slots,
        "total_slots": total_slots,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_metrics(nav_df, label="", rf_daily=0.0, slots=N_SLOTS, ret_col="daily_ret"):
    rets = nav_df[ret_col].values
    nav = np.cumprod(1 + rets)
    n = len(rets)
    if n < 2:
        return {"label": label, "n_days": n, "error": "insufficient data"}

    years = n / ANN_DAYS
    cagr = (nav[-1] ** (1 / years) - 1) * 100 if years > 0 and nav[-1] > 0 else 0
    excess = rets - rf_daily
    sd = np.std(excess, ddof=1)
    ann_vol = np.std(rets, ddof=1) * np.sqrt(ANN_DAYS) * 100
    sharpe = np.mean(excess) / sd * np.sqrt(ANN_DAYS) if sd > 0 else 0
    down = rets[rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-9
    sortino = np.mean(excess) / down_std * np.sqrt(ANN_DAYS) if down_std > 0 else 0

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    dd_start, max_dd_dur = None, 0
    for i, v in enumerate(dd):
        if v < 0:
            dd_start = i if dd_start is None else dd_start
        elif dd_start is not None:
            max_dd_dur = max(max_dd_dur, i - dd_start)
            dd_start = None
    if dd_start is not None:
        max_dd_dur = max(max_dd_dur, len(dd) - dd_start)

    q5 = np.percentile(rets, 5)
    tail = rets[rets <= q5]
    tmp = pd.DataFrame({"dt": pd.to_datetime(nav_df["dt"]), "r": rets})
    iso = tmp["dt"].dt.isocalendar()
    week_rets = tmp.groupby([iso["year"].values, iso["week"].values])["r"].apply(lambda x: (1 + x).prod() - 1)
    month_rets = tmp.groupby(tmp["dt"].dt.to_period("M"))["r"].apply(lambda x: (1 + x).prod() - 1)

    consec, max_consec = 0, 0
    for r in rets:
        consec = consec + 1 if r < 0 else 0
        max_consec = max(max_consec, consec)

    avg_pos = nav_df["n_pos"].mean() if "n_pos" in nav_df.columns else np.nan
    return {
        "label": label,
        "n_days": n,
        "total_ret_pct": round((nav[-1] - 1) * 100, 2),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd_pct": round(max_dd, 2),
        "max_dd_duration_days": int(max_dd_dur),
        "var_95_pct": round(q5 * 100, 3),
        "cvar_95_pct": round(tail.mean() * 100, 3) if len(tail) else 0.0,
        "worst_day_pct": round(rets.min() * 100, 3),
        "worst_week_pct": round(week_rets.min() * 100, 2) if len(week_rets) else 0.0,
        "worst_month_pct": round(month_rets.min() * 100, 2) if len(month_rets) else 0.0,
        "max_consec_loss": int(max_consec),
        "avg_positions": round(avg_pos, 1) if not np.isnan(avg_pos) else None,
        "utilization_pct": round(avg_pos / slots * 100, 1) if slots and not np.isnan(avg_pos) else None,
        "skewness": round(float(sp_stats.skew(rets)), 3),
        "kurtosis": round(float(sp_stats.kurtosis(rets)), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════


def newey_west_tstat(x, lags=None):
    """H0: mean(x)=0，HAC(Newey-West) 稳健 t。lags 默认 floor(4*(n/100)^(2/9))。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "t_stat": float("nan"), "p_value": 1.0, "n": n}
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    gamma0 = xc @ xc / n
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        gl = xc[lag:] @ xc[:-lag] / n
        var += 2 * (1 - lag / (lags + 1)) * gl
    var = max(var, 1e-18)
    se = np.sqrt(var / n)
    t = x.mean() / se
    return {
        "mean": float(x.mean()),
        "t_stat": round(float(t), 3),
        "n": n,
        "hac_lags": int(lags),
        "p_value": round(float(2 * (1 - sp_stats.norm.cdf(abs(t)))), 4),
        "p_value_one_sided": round(float(1 - sp_stats.norm.cdf(t)), 4),
    }


def _stationary_bootstrap_idx(n, n_boot, block_len, rng):
    """Politis-Romano 平稳分块 bootstrap 索引矩阵 (n_boot, n)，保留自相关结构。"""
    p = 1.0 / max(block_len, 1)
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < p
    jump[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    cur = starts[:, 0].copy()
    for t in range(n):
        cur = np.where(jump[:, t], starts[:, t], (cur + 1) % n)
        idx[:, t] = cur
    return idx


def block_bootstrap_mean(x, n_boot=10000, block_len=10, seed=42, alpha=0.05):
    """均值的平稳分块 bootstrap CI（日频序列有自相关，iid bootstrap 会低估宽度）。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_idx(n, n_boot, block_len, rng)
    means = x[idx].mean(axis=1)
    return {
        "mean": round(float(x.mean()), 6),
        "ci_lo": round(float(np.percentile(means, 100 * alpha / 2)), 6),
        "ci_hi": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 6),
        "p_boot_le_zero": round(float((means <= 0).mean()), 4),
        "n": n,
        "n_boot": n_boot,
        "block_len": block_len,
    }


def block_bootstrap_sharpe_diff(ra, rb, n_boot=10000, block_len=10, seed=42, alpha=0.05):
    """Sharpe 差 (a-b) 的分块 bootstrap CI。同索引重抽样，保留两序列同期相关。"""
    ra, rb = np.asarray(ra, dtype=float), np.asarray(rb, dtype=float)
    m = ~(np.isnan(ra) | np.isnan(rb))
    ra, rb = ra[m], rb[m]
    n = len(ra)
    if n < 5:
        return {"diff": float("nan"), "n": n}

    def _sr(x):
        sd = x.std(axis=-1, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)
        return x.mean(axis=-1) / sd * np.sqrt(ANN_DAYS)

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_idx(n, n_boot, block_len, rng)
    diffs = _sr(ra[idx]) - _sr(rb[idx])
    diffs = diffs[~np.isnan(diffs)]
    obs = float(_sr(ra) - _sr(rb))
    return {
        "diff": round(obs, 4),
        "sharpe_a": round(float(_sr(ra)), 4),
        "sharpe_b": round(float(_sr(rb)), 4),
        "ci_lo": round(float(np.percentile(diffs, 100 * alpha / 2)), 4),
        "ci_hi": round(float(np.percentile(diffs, 100 * (1 - alpha / 2))), 4),
        "p_boot_le_zero": round(float((diffs <= 0).mean()), 4),
        "n": n,
        "n_boot": n_boot,
        "block_len": block_len,
    }


def jkm_sharpe_test(ra, rb):
    """Jobson-Korkie / Memmel 修正的 Sharpe 相等性检验（H0: SR_a = SR_b）。

    两个策略共用同一段日历、且高度相关（S8 与 A 共享全部核心交易），必须用配对检验，
    独立样本 Sharpe 比较会严重高估显著性。
    """
    ra, rb = np.asarray(ra, dtype=float), np.asarray(rb, dtype=float)
    m = ~(np.isnan(ra) | np.isnan(rb))
    ra, rb = ra[m], rb[m]
    n = len(ra)
    if n < 5:
        return {"n": n, "error": "insufficient"}
    sa, sb = ra.std(ddof=1), rb.std(ddof=1)
    if sa <= 0 or sb <= 0:
        return {"n": n, "error": "zero variance"}
    sr_a, sr_b = ra.mean() / sa, rb.mean() / sb
    rho = float(np.corrcoef(ra, rb)[0, 1])
    var = (1 / n) * (2 * (1 - rho) + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2))
    if var <= 0:
        return {"n": n, "error": "non-positive variance"}
    z = (sr_a - sr_b) / np.sqrt(var)
    return {
        "n": n,
        "rho": round(rho, 4),
        "sharpe_a_ann": round(sr_a * np.sqrt(ANN_DAYS), 4),
        "sharpe_b_ann": round(sr_b * np.sqrt(ANN_DAYS), 4),
        "diff_ann": round((sr_a - sr_b) * np.sqrt(ANN_DAYS), 4),
        "z_stat": round(float(z), 3),
        "p_value_one_sided": round(float(1 - sp_stats.norm.cdf(z)), 4),
        "p_value_two_sided": round(float(2 * (1 - sp_stats.norm.cdf(abs(z)))), 4),
    }


_EULER = 0.5772156649015329


def expected_max_sharpe(n_trials, var_sr_daily=1.0):
    """N 次独立试验下 Sharpe 最大值的期望（Bailey & Lopez de Prado 2014 式 (5)）。"""
    if n_trials <= 1:
        return 0.0
    nrm = sp_stats.norm
    z1 = nrm.ppf(1 - 1.0 / n_trials)
    z2 = nrm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr_daily) * ((1 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe(sharpe_ann, n_trials, n_obs, skew=0.0, kurt_excess=0.0, legacy=False):
    """Deflated Sharpe Ratio。

    单位约定：入参 sharpe_ann 为**年化** Sharpe，内部换算成日频后代入 Bailey 的 SE 公式
    （该公式以「每观测期」Sharpe 为单位）。原实现直接把年化值代入日频公式，SE 低估 ~sqrt(252)，
    任何策略都被判死；legacy=True 复现旧值仅供与历史报告对齐。

    kurt_excess 传 scipy.stats.kurtosis 的**超额**峰度，内部 +3 还原为 Pearson 峰度。
    """
    nrm = sp_stats.norm
    if legacy:
        e_max = nrm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0.0
        se = np.sqrt((1 - skew * sharpe_ann + (kurt_excess - 1) / 4 * sharpe_ann**2) / n_obs)
        if se == 0:
            return {"dsr": 0.0, "p_value": 1.0}
        z = (sharpe_ann - e_max) / se
        return {
            "dsr": round(float(z), 3),
            "p_value": round(float(1 - nrm.cdf(z)), 4),
            "e_max_sharpe": round(float(e_max), 3),
            "sharpe_obs": round(sharpe_ann, 3),
            "n_trials": int(n_trials),
            "legacy": True,
        }

    sr_d = sharpe_ann / np.sqrt(ANN_DAYS)
    kurt = kurt_excess + 3.0
    se = np.sqrt(max((1 - skew * sr_d + (kurt - 1) / 4 * sr_d**2) / (n_obs - 1), 1e-24))
    e_max_d = expected_max_sharpe(n_trials, var_sr_daily=se**2)
    z = (sr_d - e_max_d) / se
    return {
        "dsr": round(float(z), 3),
        "p_value": round(float(1 - nrm.cdf(z)), 4),
        "sharpe_obs_ann": round(sharpe_ann, 3),
        "hurdle_sharpe_ann": round(float(e_max_d * np.sqrt(ANN_DAYS)), 3),
        "se_daily": round(float(se), 6),
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "passed": bool(z > sp_stats.norm.ppf(0.95)),
        "legacy": False,
    }
