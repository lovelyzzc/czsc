"""S8 增量验证 — S2b 核心 + S2c 全天候卫星（取消牛市过滤）

回答一个问题：**在 S2b 交易集合完全不变的前提下，S2c 增量是否还有正贡献。**
成立 → F 的高收益是真增量 alpha；不成立 → F 的收益主体是挤占运气 + 暴露上升。

策略臂：
  A       : S2b Only（基准，空仓持现金 0%）
  E       : S2b + 现金年化 2%
  S8a     : S2b 核心 + S2c 全天候卫星，**强制替换**（核心无槽时逐出卫星）
  S8b     : S2b 核心 + S2c 全天候卫星，**独立风险预算**（双池，n_sat 扫描）
  S8c     : = 现 F（passive，卫星可挤占核心），污染参照点

三窗口：架构新鲜段(2021-07→2023-12) / walk-forward(逐折重推门控) / 2026 留出段

    uv run --no-sync python scripts/s8_incremental_validation.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _s2c_core as C  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s8_validation"

# ─── 预注册参数（跑之前定死） ──────────────────────────────────────────────
HEADLINE_N_SAT = 5           # S8b 头条口径：卫星槽 = 核心槽的一半
N_SAT_SWEEP = [3, 5, 10]
BLOCK_LEN = 10               # 平稳分块 bootstrap 块长（日）
N_BOOT = 10000
MC_N = 500                   # 挤占运气零分布次数
COST_STRESS_BPS = [10, 25, 40, 50, 75, 100]

# 三窗口
ARCH_FRESH = (pd.Timestamp("2021-07-01"), pd.Timestamp("2023-12-31"))
HOLDOUT_2026 = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-27"))
WF_TEST_MONTHS = 6
WF_MIN_TRAIN_MONTHS = 18

# 门控重推网格（与 gate_threshold_grid_search.py 的 refine 网格一致 = 77 格）
VR_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
SP_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]

# 门控密度下限（首轮 GATE_MIN_N=30 过松，搜索选出 vr0.5/sp15 这类每折只有 1~2 笔成交的
# 退化门控，逐折 Sharpe 纯噪声）。冻结 S2b 在各训练窗保留约 0.45%~0.57% 的候选
# （180/31587、417/87472、1039/230791），故按 0.4% + 绝对下限 150 设闸。
# 卫星须比核心多 ≥2.0 倍（冻结 S2c/S2b = 2.81），否则"增量"薄到无意义。
# 这三条只约束门控规模、与结果方向无关，不构成对结论的调参。
GATE_MIN_N = 150
GATE_MIN_CORE_FRAC = 0.004
MIN_SAT_RATIO = 2.0

# 预注册通过线
P2_MIN_T = 2.0
P3_MIN_MEDIAN = 0.0
P4_MIN_CONSISTENCY = 0.60
P1_MIN_SAT_TRADES = 60

# ═══════════════════════════════════════════════════════════════════════════════
# Trial Ledger — 多重检验分母的逐项来源（约束 6）
# ═══════════════════════════════════════════════════════════════════════════════

def build_trial_ledger():
    """把「搜索过多少配置」摊开写清楚。原报告用 77 是低估。"""
    items = [
        {"source": "gate_threshold_grid_search.py (full)", "n": 288,
         "detail": "vr 6 × vr_dir 2 × sp 6 × sp_dir 2 × zg 2"},
        {"source": "gate_threshold_grid_search.py (refine)", "n": 77,
         "detail": "vr 7 × sp 11（zg=False, vr=lte, sp=gte）"},
        {"source": "comprehensive_backtest.py 策略臂", "n": 13,
         "detail": "S0/S0L/S2/S2a/S2b/S2c/S3/S4/S4b/S4c/S7/S5/BM_Random"},
        {"source": "s2c_incremental_validation.py 策略臂", "n": 6,
         "detail": "A/B/C/D(MC)/E/F"},
        {"source": "压力网格", "n": 41,
         "detail": "成本 6 + 槽位 6 + regime lag 4 + vr×sp 邻域 25"},
        {"source": "本轮 S8 新增臂", "n": 5,
         "detail": "S8a + S8b×3(n_sat 扫描) + S8c"},
    ]
    total = sum(i["n"] for i in items)
    increment_variants = [
        {"variant": "B (S2c 牛市增量)", "note": "已废弃"},
        {"variant": "S8a (强制替换)", "note": "本轮"},
        {"variant": "S8b n_sat=3/5/10", "note": "本轮，计 3"},
        {"variant": "S8c = F (无牛市过滤 passive)", "note": "上轮发现"},
    ]
    n_increment = 6
    return {
        "items": items, "total_trials": total,
        "increment_variants": increment_variants, "n_increment_trials": n_increment,
        "note": ("绝对 Sharpe 的 DSR 用 total_trials；'增量是否有用' 的 DSR 用 "
                 "n_increment_trials —— 门控搜索的 365 格是为选 S2b 花的，不该重复罚增量检验。"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Market Excess (反彩票条款的基准)
# ═══════════════════════════════════════════════════════════════════════════════

def build_market_cum(ctx):
    """面板等权市场累计净值（与策略同一 universe，用于 pair 级超额）。"""
    with warnings.catch_warnings():   # 早期日期全 NaN 切片属正常，等权收益按 0 处理
        warnings.simplefilter("ignore", RuntimeWarning)
        ew = np.nanmean(ctx["ret_mx"], axis=0)
    ew = np.nan_to_num(ew, nan=0.0)
    return np.cumprod(1 + ew)


def attach_market_excess(trades_df, ctx, mkt_cum):
    """给交易加 mkt_ret_pct / excess_net_pct（净超额 = 净收益 − 同期等权市场）。"""
    if trades_df.empty:
        for c in ("mkt_ret_pct", "excess_net_pct"):
            trades_df[c] = np.nan
        return trades_df
    dt2i = ctx["dt2i"]
    e_di = trades_df["entry_dt"].map(dt2i).values
    x_di = trades_df["actual_exit_dt"].map(dt2i).values
    last = len(mkt_cum) - 1
    e_di = np.clip(e_di.astype(float), 0, last).astype(int)
    x_di = np.clip(x_di.astype(float), 0, last).astype(int)
    trades_df["mkt_ret_pct"] = (mkt_cum[x_di] / mkt_cum[e_di] - 1) * 100
    trades_df["excess_net_pct"] = trades_df["close_ret_net_pct"] - trades_df["mkt_ret_pct"]
    return trades_df


# ═══════════════════════════════════════════════════════════════════════════════
# Arm Runner
# ═══════════════════════════════════════════════════════════════════════════════

def build_arms(d0, s2b_mask, s2c_mask):
    """核心/卫星入场表。核心 priority +100 保证同日排在卫星之前（约束 4 的排序侧）。"""
    core = C.build_core(d0, s2b_mask)
    core_boost = C.build_core(d0, s2b_mask, priority_boost=100.0)
    sat = C.build_satellite(d0, s2b_mask, s2c_mask)   # 全天候，无牛市过滤（约束 1）
    combined = pd.concat([core_boost, sat], ignore_index=True)
    return {"core": core, "core_boost": core_boost, "sat": sat, "combined": combined}


def run_arms(ctx, arms, cost_pct=C.TOTAL_COST_PCT, n_sat_sweep=N_SAT_SWEEP, verbose=True):
    """跑 A/E/S8a/S8b(扫描)/S8c，返回 {label: sim_result}。"""
    out = {}
    out["A"] = C.simulate(arms["core"], ctx, policy="passive", cost_pct=cost_pct)
    out["E"] = C.simulate(arms["core"], ctx, policy="passive", cost_pct=cost_pct,
                          cash_daily=C.CASH_RATE_DAILY)
    out["S8a"] = C.simulate(arms["combined"], ctx, policy="preempt", cost_pct=cost_pct)
    out["S8c_F"] = C.simulate(arms["combined"], ctx, policy="passive", cost_pct=cost_pct)
    for ns in n_sat_sweep:
        out[f"S8b_ns{ns}"] = C.simulate(arms["combined"], ctx, policy="budget",
                                        n_sat_slots=ns, cost_pct=cost_pct)
    if verbose:
        for k, r in out.items():
            slots = r["total_slots"]
            m = C.compute_metrics(r["nav"], label=k, slots=slots)
            n_sat = sum(1 for t in r["trades"] if t["layer"] != C.LAYER_CORE)
            print(f"  {k:<12} trades={len(r['trades']):4d} (sat={n_sat:4d}) "
                  f"Sharpe={m['sharpe']:6.3f} CAGR={m['cagr_pct']:7.2f}% MDD={m['max_dd_pct']:7.2f}%")
    return out


def core_key_set(sim):
    return {(t["symbol"], pd.Timestamp(t["entry_dt"]))
            for t in sim["trades"] if t["layer"] == C.LAYER_CORE}


def assert_p6_invariance(sims, strict_labels=("S8a", "S8b_ns3", "S8b_ns5", "S8b_ns10")):
    """P6：S8a/S8b 的核心集合必须与 A 逐笔相等，不等即报错退出（约束 3）。"""
    base = core_key_set(sims["A"])
    report = {}
    failures = []
    for lbl in strict_labels:
        if lbl not in sims:
            continue
        ks = core_key_set(sims[lbl])
        missing, extra = base - ks, ks - base
        report[lbl] = {"n_core": len(ks), "n_base": len(base),
                       "missing": len(missing), "extra": len(extra), "equal": ks == base}
        if ks != base:
            failures.append(f"{lbl}: 缺 {len(missing)} 多 {len(extra)}")
    # S8c 作为污染参照点，本就会偏离，只记录不判死
    if "S8c_F" in sims:
        ks = core_key_set(sims["S8c_F"])
        report["S8c_F"] = {"n_core": len(ks), "n_base": len(base),
                           "missing": len(base - ks), "extra": len(ks - base),
                           "equal": ks == base, "expected_to_differ": True}
    if failures:
        raise AssertionError("P6 不变性失败（约束 3 被破坏）: " + "; ".join(failures))
    report["p6_passed"] = True
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Three-Channel Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_channels(sims, headline_ns=HEADLINE_N_SAT):
    """把 F − A 拆成可加的两段 + 单列暴露通道。

    精确恒等式（同一总资本、同一日历）：
        (F − A) = (S8a − A) + (F − S8a)
                = 纯卫星贡献(核心集合不变) + 挤占通道
    暴露通道单列：S8b overlay − S8b blend（同一批交易，只差资本口径）。
    """
    def rets(lbl, col="daily_ret"):
        return sims[lbl]["nav"][col].values

    a, s8a, f = rets("A"), rets("S8a"), rets("S8c_F")
    bl = f"S8b_ns{headline_ns}"
    blend, overlay = rets(bl, "daily_ret_blend"), rets(bl, "daily_ret_overlay")

    def _ann(x):
        return (np.prod(1 + x) ** (C.ANN_DAYS / len(x)) - 1) * 100

    ch = {
        "total_F_minus_A": {
            "cagr_delta_pct": round(_ann(f) - _ann(a), 2),
            "sharpe_delta": round(_sharpe(f) - _sharpe(a), 3),
        },
        "satellite_channel_S8a_minus_A": {
            "desc": "核心集合逐笔不变，纯加卫星（强制替换）",
            "cagr_delta_pct": round(_ann(s8a) - _ann(a), 2),
            "sharpe_delta": round(_sharpe(s8a) - _sharpe(a), 3),
            "paired": C.newey_west_tstat(s8a - a),
            "boot": C.block_bootstrap_mean(s8a - a, n_boot=N_BOOT, block_len=BLOCK_LEN),
        },
        "blocking_channel_F_minus_S8a": {
            "desc": "唯一差别 = 核心是否被卫星挤占；正值即挤占在赚钱",
            "cagr_delta_pct": round(_ann(f) - _ann(s8a), 2),
            "sharpe_delta": round(_sharpe(f) - _sharpe(s8a), 3),
            "paired": C.newey_west_tstat(f - s8a),
            "boot": C.block_bootstrap_mean(f - s8a, n_boot=N_BOOT, block_len=BLOCK_LEN),
        },
        "exposure_channel_overlay_minus_blend": {
            "desc": f"S8b(n_sat={headline_ns}) 加资本 − 等资本；总敞口 >100%，杠杆读数",
            "cagr_delta_pct": round(_ann(overlay) - _ann(blend), 2),
            "sharpe_delta": round(_sharpe(overlay) - _sharpe(blend), 3),
        },
        "identity_check": {
            "sum_of_two_channels": round((_ann(s8a) - _ann(a)) + (_ann(f) - _ann(s8a)), 2),
            "total": round(_ann(f) - _ann(a), 2),
        },
    }
    return ch


def _sharpe(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(C.ANN_DAYS)) if sd > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Blocking-Luck Null Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def blocking_luck_null(ctx, arms, sims, n_iter=MC_N, seed=42, cost_pct=C.TOTAL_COST_PCT,
                       n_select_iter=2000):
    """F 挤掉了 A 的一部分核心交易，且挤掉的恰是较差的一半。这是运气还是本事？

    两个检验：
      1. **选择检验（主）**：把 A 的 329 笔**已执行**交易当固定总体，F 留下了其中一部分。
         零分布 = 按（年月 × priority 十分位）分层随机留下同等数量，比较留存集均值/中位数。
         纯选择问题，无槽位回填等混淆。
      2. **组合检验（辅）**：从核心入场表里剔除 A 的这些已执行交易，重跑组合。
         剔除会让原先被拒的信号回填（F 就多出 27 笔 A 没有的核心交易），故与 F 可比。

    注意：抽样池必须是 A 的**已执行交易**，不是 1258 行入场表 —— 后者绝大多数从未成交，
    随机剔除它们对组合几乎无影响，会造出一个虚假的弱零分布。
    """
    base_keys = core_key_set(sims["A"])
    f_keys = core_key_set(sims["S8c_F"])
    blocked = base_keys - f_keys
    kept = base_keys & f_keys
    n_block, n_keep = len(blocked), len(kept)

    ta = pd.DataFrame(sims["A"]["trades"])
    ta["entry_dt"] = pd.to_datetime(ta["entry_dt"])
    ta["_key"] = list(zip(ta["symbol"], ta["entry_dt"], strict=True))
    ta["_ym"] = ta["entry_dt"].dt.to_period("M").astype(str)
    ta["_pd"] = pd.qcut(ta["priority"].rank(method="first"), 10, labels=False, duplicates="drop")
    ta["_blocked"] = ta["_key"].isin(blocked)

    ret_col = "close_ret_net_pct"
    strata_counts = ta[ta["_blocked"]].groupby(["_ym", "_pd"]).size().to_dict()
    pools = {k: g.index.to_numpy() for k, g in ta.groupby(["_ym", "_pd"])}

    rng_sel = np.random.default_rng(seed)
    all_idx = ta.index.to_numpy()
    sel_rows = []
    for _ in range(n_select_iter):
        drop_idx = set()
        for stratum, need in strata_counts.items():
            pool = [i for i in pools.get(stratum, []) if i not in drop_idx]
            if not pool:
                continue
            take = min(need, len(pool))
            drop_idx |= {pool[i] for i in rng_sel.choice(len(pool), size=take, replace=False)}
        if len(drop_idx) < n_block:
            rest = [i for i in all_idx if i not in drop_idx]
            extra = rng_sel.choice(len(rest), size=min(n_block - len(drop_idx), len(rest)),
                                   replace=False)
            drop_idx |= {rest[i] for i in extra}
        keep_mask = ~ta.index.isin(list(drop_idx))
        kr = ta.loc[keep_mask, ret_col].dropna()
        sel_rows.append({"mean": kr.mean(), "median": kr.median()})

    sel_null = pd.DataFrame(sel_rows)
    f_kept = ta.loc[ta["_key"].isin(kept), ret_col].dropna()
    a_all = ta[ret_col].dropna()
    selection = {
        "n_universe": len(ta), "n_blocked": n_block, "n_kept": n_keep,
        "n_iter": n_select_iter, "ret_col": ret_col,
        "a_all_mean": round(float(a_all.mean()), 3),
        "a_all_median": round(float(a_all.median()), 3),
        "f_kept_mean": round(float(f_kept.mean()), 3),
        "f_kept_median": round(float(f_kept.median()), 3),
        "blocked_mean": round(float(ta.loc[ta["_blocked"], ret_col].dropna().mean()), 3),
        "null_mean_avg": round(float(sel_null["mean"].mean()), 3),
        "null_mean_std": round(float(sel_null["mean"].std()), 3),
        "pctile_mean": round(float((sel_null["mean"].values < f_kept.mean()).mean() * 100), 1),
        "pctile_median": round(float((sel_null["median"].values < f_kept.median()).mean() * 100), 1),
    }
    selection["verdict"] = ("F 的挤占选择超越随机零分布 95 分位（疑似有选择能力）"
                            if selection["pctile_mean"] >= 95
                            else "F 的挤占选择未超越随机零分布 95 分位（运气）")

    core = arms["core"].copy()
    core["dt"] = pd.to_datetime(core["dt"])
    core["_key"] = list(zip(core["symbol"], core["dt"], strict=True))
    rng = np.random.default_rng(seed)
    # 抽样池 = A 的已执行交易（按分层），不是全入场表
    by_stratum = {k: g["_key"].tolist() for k, g in ta.groupby(["_ym", "_pd"])}
    exec_keys = ta["_key"].tolist()
    rows = []
    for it in range(n_iter):
        drop = set()
        for stratum, need in strata_counts.items():
            pool = [k for k in by_stratum.get(stratum, []) if k not in drop]
            if not pool:
                continue
            take = min(need, len(pool))
            picks = rng.choice(len(pool), size=take, replace=False)
            drop |= {pool[i] for i in picks}
        # 分层池不足时用已执行交易补齐，保证剔除数量与 F 一致
        if len(drop) < n_block:
            rest = [k for k in exec_keys if k not in drop]
            extra = rng.choice(len(rest), size=min(n_block - len(drop), len(rest)), replace=False)
            drop |= {rest[i] for i in extra}

        kept = core[~core["_key"].isin(drop)].drop(columns=["_key"])
        r = C.simulate(kept, ctx, policy="passive", cost_pct=cost_pct, track_details=False)
        m = C.compute_metrics(r["nav"], label=f"mc{it}")
        rows.append({"iter": it, "n_dropped": len(drop), "sharpe": m["sharpe"],
                     "cagr_pct": m["cagr_pct"], "max_dd_pct": m["max_dd_pct"],
                     "calmar": m["calmar"]})
        if (it + 1) % 100 == 0:
            print(f"    MC {it + 1}/{n_iter} ...")

    mc = pd.DataFrame(rows)

    # F 的实际核心组合：剔除被 F 挤掉的那批已执行交易后重跑（允许回填，与 F 同构）
    f_core_entries = core[~core["_key"].isin(blocked)].drop(columns=["_key"])
    r_f = C.simulate(f_core_entries, ctx, policy="passive", cost_pct=cost_pct)
    m_f = C.compute_metrics(r_f["nav"], label="F_core_only")
    m_a = C.compute_metrics(sims["A"]["nav"], label="A")

    pct = {}
    for col in ("sharpe", "cagr_pct", "calmar", "max_dd_pct"):
        v = m_f[col]
        pct[col] = {"f_core_only": v, "mc_mean": round(float(mc[col].mean()), 3),
                    "mc_std": round(float(mc[col].std()), 3),
                    "percentile": round(float((mc[col].values < v).mean() * 100), 1),
                    "a_full": m_a[col]}
    return {"n_blocked": n_block, "n_iter": n_iter, "mc_df": mc,
            "f_core_only_metrics": m_f, "percentiles": pct, "selection_test": selection,
            "verdict": ("挤占的组合级收益在随机零分布中不显著（运气）"
                        if pct["sharpe"]["percentile"] < 95
                        else "挤占的组合级收益超越随机零分布 95 分位")}


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Re-derivation (walk-forward 逐折)
# ═══════════════════════════════════════════════════════════════════════════════

def derive_gates(d0, train_lo, train_hi, fwd_col="fwd_5d"):
    """在训练窗内用与 gate_threshold_grid_search 相同的判据（fwd_5d 均值超额）重推门控。

    核心 = 超额最高者；卫星 = 在「vr 更松且 sp 更松」的合法嵌套配置中超额最高者，
    保证 core ⊂ satellite（与 S2b ⊂ S2c 的历史关系同构）。
    """
    tr = d0[(d0["entry_dt"] >= train_lo) & (d0["entry_dt"] <= train_hi)]
    tr = tr[tr[fwd_col].notna()]
    if len(tr) < GATE_MIN_N:
        return None
    base = tr[fwd_col].mean()
    min_n = max(GATE_MIN_N, int(len(tr) * GATE_MIN_CORE_FRAC))
    recs = []
    for vr in VR_GRID:
        for sp in SP_GRID:
            m = C.gate_mask(tr, vr, sp)
            n = int(m.sum())
            if n < min_n:
                continue
            recs.append({"vr": vr, "sp": sp, "n": n,
                         "excess": float(tr.loc[m, fwd_col].mean() - base)})
    if not recs:
        return None
    g = pd.DataFrame(recs).sort_values("excess", ascending=False)
    core = g.iloc[0]
    # 卫星须严格更松（嵌套 core ⊂ sat）且候选量 ≥ MIN_SAT_RATIO × core
    nested = g[(g["vr"] >= core["vr"]) & (g["sp"] <= core["sp"])]
    nested = nested[(nested["vr"] > core["vr"]) | (nested["sp"] < core["sp"])]
    nested = nested[nested["n"] >= MIN_SAT_RATIO * core["n"]]
    if nested.empty:
        return None
    sat = nested.iloc[0]
    return {"core_vr": float(core["vr"]), "core_sp": float(core["sp"]),
            "sat_vr": float(sat["vr"]), "sat_sp": float(sat["sp"]),
            "core_excess": round(float(core["excess"]), 4),
            "sat_excess": round(float(sat["excess"]), 4),
            "n_core_train": int(core["n"]), "n_sat_train": int(sat["n"]),
            "sat_ratio": round(float(sat["n"] / max(core["n"], 1)), 2),
            "min_n_applied": min_n, "n_train": len(tr),
            "baseline_fwd": round(float(base), 4), "n_grid_evaluated": len(recs)}


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward (主口径：逐折重推门控，门控+架构都是真 OOS)
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward(ctx, mkt_cum, headline_ns=HEADLINE_N_SAT, cost_pct=C.TOTAL_COST_PCT):
    d0 = ctx["d0"]
    all_dates = ctx["dates"]
    first, last = all_dates[0], all_dates[-1]
    fold_start = (first + pd.DateOffset(months=WF_MIN_TRAIN_MONTHS)).to_period("M").to_timestamp()

    folds, fold_rows = [], []
    cur = fold_start
    while cur < last:
        test_lo = cur
        test_hi = min((cur + pd.DateOffset(months=WF_TEST_MONTHS) - pd.Timedelta(days=1)), last)
        if len([d for d in all_dates if test_lo <= d <= test_hi]) < 40:
            break
        folds.append((first, test_lo - pd.Timedelta(days=1), test_lo, test_hi))
        cur = cur + pd.DateOffset(months=WF_TEST_MONTHS)

    sat_trades_all = []
    stitch = {"A": [], "S8a": [], "S8b": [], "S8c_F": [], "dt": []}
    for i, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(folds):
        gates = derive_gates(d0, tr_lo, tr_hi)
        if gates is None:
            print(f"    fold{i} {str(te_lo)[:7]}: 训练样本不足，跳过")
            continue
        C.set_eval_window(ctx, te_lo, te_hi)
        s2b = C.gate_mask(d0, gates["core_vr"], gates["core_sp"])
        s2c = C.gate_mask(d0, gates["sat_vr"], gates["sat_sp"])
        arms = build_arms(d0, s2b, s2c)

        a = C.simulate(arms["core"], ctx, policy="passive", cost_pct=cost_pct)
        s8a = C.simulate(arms["combined"], ctx, policy="preempt", cost_pct=cost_pct)
        s8b = C.simulate(arms["combined"], ctx, policy="budget", n_sat_slots=headline_ns,
                         cost_pct=cost_pct)
        # F 也进 walk-forward：约束 6 要求把 F 纳入同一套多重检验，
        # 而 F 的历史读数落在门控被拟合的那段窗口里，必须有一个诚实的 OOS 数
        s8c = C.simulate(arms["combined"], ctx, policy="passive", cost_pct=cost_pct)
        # 逐折也校验 P6
        ok_a = core_key_set(a) == core_key_set(s8a)
        ok_b = core_key_set(a) == core_key_set(s8b)

        ra = a["nav"]["daily_ret"].values
        r8a = s8a["nav"]["daily_ret"].values
        r8b = s8b["nav"]["daily_ret_blend"].values
        r8c = s8c["nav"]["daily_ret"].values
        m_a = C.compute_metrics(a["nav"], slots=10)
        m_8a = C.compute_metrics(s8a["nav"], slots=10)
        m_8b = C.compute_metrics(s8b["nav"], slots=10 + headline_ns, ret_col="daily_ret_blend")
        m_8c = C.compute_metrics(s8c["nav"], slots=10)

        for arm_lbl, sim in (("S8a", s8a), ("S8b", s8b), ("S8c_F", s8c)):
            st = pd.DataFrame([t for t in sim["trades"] if t["layer"] != C.LAYER_CORE])
            if not st.empty:
                st = attach_market_excess(st, ctx, mkt_cum)
                st["fold"] = i
                st["arm"] = arm_lbl
                sat_trades_all.append(st)

        # 逐折日收益缝合成连续 walk-forward 序列（P2 的主检验样本）
        stitch["A"].append(ra)
        stitch["S8a"].append(r8a)
        stitch["S8b"].append(r8b)
        stitch["S8c_F"].append(r8c)
        stitch["dt"].append(a["nav"]["dt"].values)

        fold_rows.append({
            "fold": i, "train_hi": str(tr_hi)[:10], "test_lo": str(te_lo)[:10],
            "test_hi": str(te_hi)[:10], "n_days": len(ra),
            **{f"gate_{k}": v for k, v in gates.items() if k.startswith(("core_", "sat_"))},
            "n_core_trades": len(core_key_set(a)),
            "n_sat_trades_S8a": int(sum(1 for t in s8a["trades"] if t["layer"] != C.LAYER_CORE)),
            "p6_ok_S8a": ok_a, "p6_ok_S8b": ok_b,
            "p6_ok_S8c_F": core_key_set(a) == core_key_set(s8c),
            "sharpe_A": m_a["sharpe"], "sharpe_S8a": m_8a["sharpe"], "sharpe_S8b": m_8b["sharpe"],
            "sharpe_S8c_F": m_8c["sharpe"],
            "cagr_A": m_a["cagr_pct"], "cagr_S8a": m_8a["cagr_pct"], "cagr_S8b": m_8b["cagr_pct"],
            "cagr_S8c_F": m_8c["cagr_pct"],
            "d_sharpe_S8a": round(m_8a["sharpe"] - m_a["sharpe"], 3),
            "d_sharpe_S8b": round(m_8b["sharpe"] - m_a["sharpe"], 3),
            "d_sharpe_S8c_F": round(m_8c["sharpe"] - m_a["sharpe"], 3),
            "mean_daily_diff_S8a_bps": round(float(np.mean(r8a - ra) * 1e4), 3),
            "mean_daily_diff_S8b_bps": round(float(np.mean(r8b - ra) * 1e4), 3),
            "mean_daily_diff_S8c_F_bps": round(float(np.mean(r8c - ra) * 1e4), 3),
        })
        print(f"    fold{i} {str(te_lo)[:7]}→{str(te_hi)[:7]} "
              f"gate core(vr{gates['core_vr']},sp{gates['core_sp']}) "
              f"sat(vr{gates['sat_vr']},sp{gates['sat_sp']}) | ΔSharpe "
              f"S8a={m_8a['sharpe'] - m_a['sharpe']:+.3f} "
              f"S8b={m_8b['sharpe'] - m_a['sharpe']:+.3f} "
              f"F={m_8c['sharpe'] - m_a['sharpe']:+.3f} | P6={ok_a and ok_b}")

    fdf = pd.DataFrame(fold_rows)
    sat_df = pd.concat(sat_trades_all, ignore_index=True) if sat_trades_all else pd.DataFrame()
    summary = {}
    if not fdf.empty:
        for arm in ("S8a", "S8b", "S8c_F"):
            d = fdf[f"d_sharpe_{arm}"].values
            summary[arm] = {
                "n_folds": len(d),
                "n_positive": int((d > 0).sum()),
                "consistency": round(float((d > 0).mean()), 3),
                "mean_d_sharpe": round(float(d.mean()), 3),
                "median_d_sharpe": round(float(np.median(d)), 3),
                "p6_all_ok": bool(fdf[f"p6_ok_{arm}"].all()),
            }
    stitched = {k: (np.concatenate(v) if v else np.array([])) for k, v in stitch.items()}
    return {"folds_df": fdf, "summary": summary, "sat_trades": sat_df,
            "n_folds": len(fold_rows), "stitched": stitched}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-Window Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_window(ctx, mkt_cum, lo, hi, label, headline_ns=HEADLINE_N_SAT,
                cost_pct=C.TOTAL_COST_PCT, run_p6=True):
    """在固定窗内跑全部臂（门控沿用冻结的 S2b/S2c），返回指标 + 配对检验 + 卫星交易。"""
    C.set_eval_window(ctx, lo, hi)
    d0 = ctx["d0"]
    s2b = C.gate_mask(d0, C.S2B_VR, C.S2B_SP)
    s2c = C.gate_mask(d0, C.S2C_VR, C.S2C_SP)
    arms = build_arms(d0, s2b, s2c)
    print(f"  [{label}] {str(lo)[:10]} → {str(hi)[:10]}")
    sims = run_arms(ctx, arms, cost_pct=cost_pct)
    p6 = assert_p6_invariance(sims) if run_p6 else {}

    bl = f"S8b_ns{headline_ns}"
    metrics, ret_map = {}, {}
    for lbl, sim in sims.items():
        col = "daily_ret_blend" if sim["policy"] == "budget" else "daily_ret"
        metrics[lbl] = C.compute_metrics(sim["nav"], label=lbl, slots=sim["total_slots"], ret_col=col)
        ret_map[lbl] = sim["nav"][col].values
        if sim["policy"] == "budget":
            metrics[lbl + "_overlay"] = C.compute_metrics(
                sim["nav"], label=lbl + "_overlay", slots=sim["total_slots"],
                ret_col="daily_ret_overlay")
            ret_map[lbl + "_overlay"] = sim["nav"]["daily_ret_overlay"].values

    ra = ret_map["A"]
    paired = {}
    for lbl in ("S8a", bl, "S8c_F"):
        d = ret_map[lbl] - ra
        paired[lbl] = {
            "hac": C.newey_west_tstat(d),
            "boot_mean_diff": C.block_bootstrap_mean(d, n_boot=N_BOOT, block_len=BLOCK_LEN),
            "boot_sharpe_diff": C.block_bootstrap_sharpe_diff(ret_map[lbl], ra, n_boot=N_BOOT,
                                                              block_len=BLOCK_LEN),
            "jkm": C.jkm_sharpe_test(ret_map[lbl], ra),
        }

    sat_trades = {}
    for lbl in ("S8a", bl, "S8c_F"):
        st = pd.DataFrame([t for t in sims[lbl]["trades"] if t["layer"] != C.LAYER_CORE])
        if not st.empty:
            st = attach_market_excess(st, ctx, mkt_cum)
        sat_trades[lbl] = st

    return {"label": label, "window": [str(lo)[:10], str(hi)[:10]], "sims": sims,
            "metrics": metrics, "paired": paired, "p6": p6, "sat_trades": sat_trades,
            "ret_map": ret_map, "arms": arms}


def cost_stress(ctx, mkt_cum, lo, hi, headline_ns=HEADLINE_N_SAT):
    """成本压力：卫星换手更高，成本敏感性必须单列。"""
    rows = []
    for bps in COST_STRESS_BPS:
        cp = bps / 100.0  # bps → 百分点（40bps = 0.40%）
        C.set_eval_window(ctx, lo, hi)
        d0 = ctx["d0"]
        arms = build_arms(d0, C.gate_mask(d0, C.S2B_VR, C.S2B_SP),
                          C.gate_mask(d0, C.S2C_VR, C.S2C_SP))
        a = C.simulate(arms["core"], ctx, policy="passive", cost_pct=cp, track_details=False)
        s8a = C.simulate(arms["combined"], ctx, policy="preempt", cost_pct=cp, track_details=False)
        s8b = C.simulate(arms["combined"], ctx, policy="budget", n_sat_slots=headline_ns,
                         cost_pct=cp, track_details=False)
        m_a = C.compute_metrics(a["nav"], slots=10)
        m_8a = C.compute_metrics(s8a["nav"], slots=10)
        m_8b = C.compute_metrics(s8b["nav"], slots=10 + headline_ns, ret_col="daily_ret_blend")
        rows.append({"cost_bps": bps, "sharpe_A": m_a["sharpe"], "sharpe_S8a": m_8a["sharpe"],
                     "sharpe_S8b": m_8b["sharpe"], "cagr_A": m_a["cagr_pct"],
                     "cagr_S8a": m_8a["cagr_pct"], "cagr_S8b": m_8b["cagr_pct"],
                     "d_sharpe_S8a": round(m_8a["sharpe"] - m_a["sharpe"], 3),
                     "d_sharpe_S8b": round(m_8b["sharpe"] - m_a["sharpe"], 3)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-registered Verdict (P1–P6)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_arm(arm_key, wf, windows, ledger, headline_ns=HEADLINE_N_SAT):
    """对单个 S8 臂逐条判定 P1–P6。主口径 = walk-forward 缝合序列。"""
    st = wf["stitched"]
    ra, rs = st["A"], st[arm_key]
    n_obs = len(ra)
    d = rs - ra

    hac = C.newey_west_tstat(d)
    boot = C.block_bootstrap_mean(d, n_boot=N_BOOT, block_len=BLOCK_LEN)
    boot_sr = C.block_bootstrap_sharpe_diff(rs, ra, n_boot=N_BOOT, block_len=BLOCK_LEN)
    jkm = C.jkm_sharpe_test(rs, ra)

    sat = wf["sat_trades"]
    if not sat.empty and "arm" in sat.columns:
        sat = sat[sat["arm"] == arm_key]
    n_sat = len(sat)
    med_excess = float(sat["excess_net_pct"].median()) if n_sat else float("nan")
    mean_excess = float(sat["excess_net_pct"].mean()) if n_sat else float("nan")
    sat_hac = C.newey_west_tstat(sat["excess_net_pct"].values) if n_sat else {}

    # P4：walk-forward 逐折方向 + 架构新鲜段方向
    wf_sum = wf["summary"].get(arm_key, {})
    arch = windows.get("arch_fresh")
    arch_key = {"S8a": "S8a", "S8b": f"S8b_ns{headline_ns}", "S8c_F": "S8c_F"}[arm_key]
    arch_d = None
    if arch:
        arch_d = arch["metrics"][arch_key]["sharpe"] - arch["metrics"]["A"]["sharpe"]
    n_pos = wf_sum.get("n_positive", 0) + (1 if (arch_d or 0) > 0 else 0)
    n_tot = wf_sum.get("n_folds", 0) + (1 if arch_d is not None else 0)
    consistency = n_pos / n_tot if n_tot else 0.0

    sr_ann = jkm.get("sharpe_a_ann", 0.0)
    skew = float(pd.Series(rs).skew())
    kurt = float(pd.Series(rs).kurt())
    dsr_tiers = {}
    for tier, nt in [("77", 77), ("365", 365), ("total", ledger["total_trials"])]:
        dsr_tiers[tier] = C.deflated_sharpe(sr_ann, nt, n_obs, skew=skew, kurt_excess=kurt)
    dsr_legacy = C.deflated_sharpe(sr_ann, 77, n_obs, skew=skew, kurt_excess=kurt, legacy=True)

    # 增量 DSR：把「增量本身」当作被搜索的对象，分母 = 增量变体数
    sd = float(np.std(d, ddof=1))
    incr_sharpe_ann = float(np.mean(d) / sd * np.sqrt(C.ANN_DAYS)) if sd > 0 else 0.0
    dsr_incr = C.deflated_sharpe(incr_sharpe_ann, ledger["n_increment_trials"], n_obs,
                                 skew=float(pd.Series(d).skew()),
                                 kurt_excess=float(pd.Series(d).kurt()))

    # P6 逐臂读该臂自己的核心集合比对（F 天生会偏离 → 必然不过，这正是约束 3 的判死点）
    p6_ok = True
    p6_detail = {}
    for wname, w in windows.items():
        rec = (w.get("p6") or {}).get(arch_key)
        if rec is not None:
            p6_detail[wname] = {"missing": rec["missing"], "extra": rec["extra"]}
            p6_ok = p6_ok and bool(rec["equal"])
    p6_ok = p6_ok and bool(wf_sum.get("p6_all_ok", False))

    checks = {
        "P1_sample": {"pass": n_sat >= P1_MIN_SAT_TRADES, "n_sat_trades": n_sat,
                      "threshold": P1_MIN_SAT_TRADES},
        "P2_increment_significant": {
            "pass": bool(hac["mean"] > 0 and hac["t_stat"] >= P2_MIN_T and boot["ci_lo"] > 0),
            "mean_daily_diff_bps": round(hac["mean"] * 1e4, 3), "hac_t": hac["t_stat"],
            "hac_p_one_sided": hac["p_value_one_sided"], "boot_ci": [boot["ci_lo"], boot["ci_hi"]],
            "sharpe_diff_ann": jkm.get("diff_ann"), "jkm_z": jkm.get("z_stat"),
            "boot_sharpe_diff_ci": [boot_sr.get("ci_lo"), boot_sr.get("ci_hi")],
            "threshold": f"t>={P2_MIN_T} 且 boot CI 下界>0"},
        "P3_anti_lottery": {
            "pass": bool(n_sat > 0 and med_excess > P3_MIN_MEDIAN),
            "median_excess_net_pct": round(med_excess, 3) if n_sat else None,
            "mean_excess_net_pct": round(mean_excess, 3) if n_sat else None,
            "excess_hac_t": sat_hac.get("t_stat"), "threshold": "中位数>0"},
        "P4_cross_window_consistency": {
            "pass": consistency >= P4_MIN_CONSISTENCY,
            "consistency": round(consistency, 3), "n_positive": n_pos, "n_windows": n_tot,
            "wf_fold_consistency": wf_sum.get("consistency"),
            "arch_fresh_d_sharpe": round(arch_d, 3) if arch_d is not None else None,
            "threshold": P4_MIN_CONSISTENCY},
        "P5_multiple_testing": {
            "pass": bool(dsr_tiers["365"]["passed"]),
            "sharpe_ann_wf": round(sr_ann, 3), "n_obs": n_obs,
            "dsr_by_tier": dsr_tiers, "dsr_increment": dsr_incr,
            "dsr_legacy_buggy_77": dsr_legacy,
            "threshold": "修正版 DSR @ n_trials=365 通过 (z>1.645)"},
        "P6_core_invariance": {"pass": p6_ok, "by_window": p6_detail,
                               "threshold": "核心集合与 A 逐笔相等（约束 3）"},
    }

    hard = ["P2_increment_significant", "P3_anti_lottery", "P6_core_invariance"]
    if not all(checks[k]["pass"] for k in hard):
        failed = [k for k in hard if not checks[k]["pass"]]
        judgment, action = f"关闭 S8（硬条款不过: {', '.join(failed)}）", "回落 A/E"
    elif all(checks[k]["pass"] for k in checks):
        judgment, action = "提升 S8 为推荐策略", "接入 comprehensive_backtest / daily_scan"
    elif dsr_tiers["77"]["passed"]:
        judgment, action = "暂定，待前向数据（P5 仅在 n_trials=77 档通过）", "不进 daily_scan 默认路径"
    else:
        failed = [k for k, v in checks.items() if not v["pass"]]
        judgment, action = f"不提升（未过: {', '.join(failed)}）", "回落 A/E"

    return {"arm": arm_key, "checks": checks, "judgment": judgment, "action": action,
            "n_pass": sum(1 for v in checks.values() if v["pass"]), "n_checks": len(checks)}


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj))[:10]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = C.load_context()
    C.attach_forward_returns(ctx, windows=(5, 20))
    mkt_cum = build_market_cum(ctx)
    ledger = build_trial_ledger()
    print(f"[试验账本] 总计 {ledger['total_trials']} 次搜索 | 增量变体 {ledger['n_increment_trials']}")

    d0 = ctx["d0"]
    s2b = C.gate_mask(d0, C.S2B_VR, C.S2B_SP)
    s2c = C.gate_mask(d0, C.S2C_VR, C.S2C_SP)
    print(f"[门控] S2b={int(s2b.sum())} S2c={int(s2c.sum())} "
          f"S2c-only={int((s2c & ~s2b).sum())} | S2b\\S2c={int((s2b & ~s2c).sum())} (须为 0)")
    assert int((s2b & ~s2c).sum()) == 0, "S2b ⊄ S2c，卫星集合定义前提不成立"

    results = {"trial_ledger": ledger, "config": {
        "headline_n_sat": HEADLINE_N_SAT, "n_sat_sweep": N_SAT_SWEEP,
        "cost_pct": C.TOTAL_COST_PCT, "block_len": BLOCK_LEN, "n_boot": N_BOOT,
        "mc_n": MC_N, "gates": {"s2b": [C.S2B_VR, C.S2B_SP], "s2c": [C.S2C_VR, C.S2C_SP]},
        "prereg": {"P1_min_sat": P1_MIN_SAT_TRADES, "P2_min_t": P2_MIN_T,
                   "P3_min_median": P3_MIN_MEDIAN, "P4_min_consistency": P4_MIN_CONSISTENCY},
    }}

    # ─── 窗口 1/3：主窗（2024+，选择性污染，仅作对齐历史用） ────────────
    print("\n[窗口] 主窗 2024+（选择性污染，与历史报告对齐）...")
    win_main = eval_window(ctx, mkt_cum, C.TRAIN_END + pd.Timedelta(days=1),
                           pd.Timestamp(ctx["dates"][-1]), "main_2024plus")

    # ─── 窗口 2/3：架构新鲜段 ──────────────────────────────────────────
    print("\n[窗口] 架构新鲜段 2021-07 → 2023-12（组合模拟从未在此跑过）...")
    win_arch = eval_window(ctx, mkt_cum, *ARCH_FRESH, label="arch_fresh")

    # ─── 窗口 3/3：2026 留出段 ────────────────────────────────────────
    print("\n[窗口] 2026 留出段（方向检查，样本薄）...")
    win_2026 = eval_window(ctx, mkt_cum, *HOLDOUT_2026, label="holdout_2026")

    windows = {"main_2024plus": win_main, "arch_fresh": win_arch, "holdout_2026": win_2026}
    for k, w in windows.items():
        results[f"window_{k}"] = {"window": w["window"], "metrics": w["metrics"],
                                  "paired": w["paired"], "p6": w["p6"]}

    # ─── 三通道分解（主窗，因为 F 的历史读数在此） ──────────────────────
    print("\n[分解] 三通道 ...")
    C.set_eval_window(ctx, C.TRAIN_END + pd.Timedelta(days=1), pd.Timestamp(ctx["dates"][-1]))
    sims_main = win_main["sims"]
    channels = decompose_channels(sims_main)
    results["channel_decomposition"] = channels
    for k, v in channels.items():
        if isinstance(v, dict) and "cagr_delta_pct" in v:
            print(f"  {k:<42} ΔCAGR={v['cagr_delta_pct']:+7.2f}% ΔSharpe={v['sharpe_delta']:+.3f}")

    # ─── 挤占运气零分布 ───────────────────────────────────────────────
    print(f"\n[MC] 挤占运气零分布 × {MC_N} ...")
    blk = blocking_luck_null(ctx, win_main["arms"], sims_main, n_iter=MC_N)
    results["blocking_luck"] = {k: v for k, v in blk.items() if k != "mc_df"}
    sel = blk["selection_test"]
    print(f"  F 挤掉 {blk['n_blocked']}/{sel['n_universe']} 笔已执行核心 "
          f"(被挤掉均值 {sel['blocked_mean']}% vs 全体 {sel['a_all_mean']}%)")
    print(f"  [选择检验] F 留存均值 {sel['f_kept_mean']}% vs 随机零分布 "
          f"{sel['null_mean_avg']}±{sel['null_mean_std']}% → 分位 {sel['pctile_mean']}% | {sel['verdict']}")
    print(f"  [组合检验] F 纯核心 Sharpe 分位={blk['percentiles']['sharpe']['percentile']}% "
          f"(MC 均值 {blk['percentiles']['sharpe']['mc_mean']}, A={blk['percentiles']['sharpe']['a_full']})"
          f" | {blk['verdict']}")

    # ─── walk-forward（主口径） ───────────────────────────────────────
    print("\n[walk-forward] 逐折重推门控 ...")
    wf = walk_forward(ctx, mkt_cum)
    results["walk_forward"] = {"summary": wf["summary"], "n_folds": wf["n_folds"],
                               "folds": wf["folds_df"]}
    for arm, s in wf["summary"].items():
        print(f"  {arm}: {s['n_positive']}/{s['n_folds']} 折为正 (一致性 {s['consistency']}) "
              f"| 均值ΔSharpe={s['mean_d_sharpe']:+.3f} | P6 全过={s['p6_all_ok']}")

    # ─── 成本压力 ────────────────────────────────────────────────────
    print("\n[压力] 成本 10→100 bps ...")
    cs = cost_stress(ctx, mkt_cum, C.TRAIN_END + pd.Timedelta(days=1), pd.Timestamp(ctx["dates"][-1]))
    results["cost_stress"] = cs
    print(cs.to_string(index=False))

    # ─── 预注册判定 ──────────────────────────────────────────────────
    print("\n[判定] 预注册 P1–P6 ...")
    verdicts = {}
    for arm in ("S8a", "S8b", "S8c_F"):
        v = evaluate_arm(arm, wf, windows, ledger)
        verdicts[arm] = v
        print(f"\n  ── {arm} ({v['n_pass']}/{v['n_checks']} 过) → {v['judgment']}")
        for name, ck in v["checks"].items():
            mark = "✓" if ck["pass"] else "✗"
            extra = {k: val for k, val in ck.items()
                     if k not in ("pass", "threshold", "dsr_by_tier", "dsr_increment",
                                  "dsr_legacy_buggy_77", "by_window")}
            print(f"     {mark} {name}: {extra}")
    results["verdicts"] = verdicts

    # ─── 落盘 ────────────────────────────────────────────────────────
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=1), encoding="utf-8")
    wf["folds_df"].to_csv(OUTPUT_DIR / "walkforward_folds.csv", index=False)
    cs.to_csv(OUTPUT_DIR / "cost_stress.csv", index=False)
    blk["mc_df"].to_csv(OUTPUT_DIR / "blocking_mc.csv", index=False)
    pd.DataFrame([{"channel": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}}
                  for k, v in channels.items() if isinstance(v, dict)]).to_csv(
        OUTPUT_DIR / "channel_decomposition.csv", index=False)
    if not wf["sat_trades"].empty:
        wf["sat_trades"].to_csv(OUTPUT_DIR / "wf_satellite_trades.csv", index=False)
    nav_out = []
    for k, w in windows.items():
        for lbl, r in w["ret_map"].items():
            nav_out.append(pd.DataFrame({"window": k, "arm": lbl,
                                         "dt": w["sims"]["A"]["nav"]["dt"].values,
                                         "daily_ret": r}))
    pd.concat(nav_out, ignore_index=True).to_csv(OUTPUT_DIR / "daily_nav.csv", index=False)
    (OUTPUT_DIR / "trial_ledger.json").write_text(
        json.dumps(_jsonable(ledger), ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[输出] {OUTPUT_DIR} | 耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


# <!-- S8-CHUNK-7 -->
