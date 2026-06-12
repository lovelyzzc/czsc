"""组合容量层研究：槽数曲线 + bd_confirm2@20 预注册重检 + 全日历诚实年化（第五轮）

第四轮预注册重检触发条件已满足（终态配置=20 槽，轮二广度优先结论）。本脚本回答：

1. bd_confirm2 在 20 槽下是否通过（10 槽败因 = 持有期 16→28 天吃掉容量）；
2. 广度在哪里饱和：槽数曲线 N∈{5,10,20,30,40} + 每日候选供给 vs 取用分析；
3. 终态配置的诚实年化：日收益 reindex 到完整交易日历（空仓日=0，含市场门
   关闭期与无候选期），报年化/夏普/回撤/卡玛 + 空仓日占比 + 平均占用槽数。

预声明判定（先于运行写定，跑完不许改）：

- **bd_confirm2@20 通过** = 20 槽镜像 OOS 超额中位数 > 0 且 均值 > 0 且 t ≥ 2
  且 IS 中位数 ≥ full@20 的 IS 中位数（第四轮判定式原样平移到 20 槽）；
  不通过 → bd_confirm2 关闭，只剩前向 ≥60 笔一个重开条件；
- **槽数默认值改变** = 仅当某 N∈{5,10,30,40} 同时满足：OOS t ≥ t(20) 且
  中位数 ≥ median(20) 且 均值 ≥ 0.8×mean(20)（a-priori，禁止事后挑点）；
- 年化部分纯描述性（无通过/失败），全日历口径强制。

交叉校验（结论前置条件）：full@10 须复现第四轮既知数（n=210 / 6.84 / -0.38 / t=2.69）；
full@20 的 OOS t 应与轮二既知 ≈3.82 同量级；entry 定位失败 0；漂移占比 ≈0.6%。

已知口径混合：日收益曲线的持仓中段用 panel 收盘（dump 时 qfq），入/出场价用当前
cache 重放价，影响上限 = 漂移占比（|Δ|>0.5% 仅 0.6%）。

    uv run --no-sync python scripts/surge_capacity_research.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import surge_exit_research as ser
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import surge_selection_audit as ssa

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_capacity"
VARIANTS = {name: ser.VARIANTS[name] for name in ("full", "bd_confirm2")}
SLOT_GRID = [5, 10, 20, 30, 40]
DEFAULT_SLOTS = 20
OOS_START = pd.Timestamp("2024-01-01")


def slot_point(frame: pd.DataFrame, n_slots: int) -> tuple[dict, pd.DataFrame, pd.Series]:
    trades, daily = spb.simulate_slots(frame, n_slots)
    point = {
        "oos": ser.seg_stats(trades, "test"),
        "is": ser.seg_stats(trades, "train"),
        "taken": int(len(trades)),
        "taken_share": msf.round_float(len(trades) / len(frame) * 100, 1),
    }
    return point, trades, daily


def supply_analysis(frame: pd.DataFrame, trades_by_n: dict[int, pd.DataFrame]) -> dict:
    """每日候选供给分布 + 各 N 下的受限日占比（taken < 当日候选 ⇒ 槽位/同票约束）。"""
    cand_daily = frame.groupby("entry_dt").size()
    dist = {
        "active_days": int(len(cand_daily)),
        "mean": msf.round_float(cand_daily.mean(), 1),
        "median": msf.round_float(cand_daily.median(), 1),
        "p90": msf.round_float(cand_daily.quantile(0.9), 1),
        "max": int(cand_daily.max()),
    }
    rows = []
    for n_slots, trades in trades_by_n.items():
        taken_daily = trades.groupby("entry_dt").size().reindex(cand_daily.index).fillna(0)
        constrained = taken_daily < cand_daily
        rows.append(
            {
                "slots": n_slots,
                "constrained_days%": msf.round_float(constrained.mean() * 100, 1),
                "untaken_cands%": msf.round_float((1 - taken_daily.sum() / cand_daily.sum()) * 100, 1),
            }
        )
    return {"daily_candidates": dist, "by_slots": rows}


def honest_curve(daily: pd.Series, trades: pd.DataFrame, calendar: np.ndarray, lo, hi, n_slots: int) -> dict:
    """全日历口径曲线统计：空仓日=0；平均占用槽 = 持仓·日总量 / 窗口天数。"""
    cal = calendar[(calendar >= np.datetime64(lo)) & (calendar <= np.datetime64(hi))]
    if len(cal) < 20:
        return {}
    idx = pd.DatetimeIndex(cal)
    filled = daily.reindex(idx).fillna(0.0)
    stats = spb.curve_stats(filled)
    stats["空仓日占比%"] = msf.round_float(float((~idx.isin(daily.index)).mean()) * 100, 1)
    sub = trades[(trades["entry_dt"] >= idx[0]) & (trades["entry_dt"] <= idx[-1])]
    pos_days = (
        np.searchsorted(cal, sub["exit_dt"].to_numpy(), side="right")
        - np.searchsorted(cal, sub["entry_dt"].to_numpy(), side="left")
    ).sum()
    stats["平均占用槽"] = msf.round_float(float(pos_days) / len(cal), 1)
    stats["满仓上限"] = n_slots
    return stats


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uni = ssa.load_universe()
    uni = ssa.with_excess(uni)
    pop = uni[ssa.cond_mask(uni, ssa.BASE)].copy()
    print(f"[人群] delay5 基线 {len(pop)} 笔 / {pop['symbol'].nunique()} 只")
    meta_cols = ["symbol", "sig_dt", "dec_dt", "entry_dt", "seg", "year", "priority", "score", "sl_pct", "amount_e"]
    meta = pop[meta_cols].copy()
    stored = pop[["symbol", "sig_dt", "ret_gross_pct"]].rename(columns={"ret_gross_pct": "stored_gross"})

    tasks = [
        (sym, g[["sig_dt", "dec_dt", "entry_dt"]].to_dict("records")) for sym, g in pop.groupby("symbol", sort=True)
    ]
    worker = partial(ser._replay_symbol, variants=VARIANTS)
    ctx = mp.get_context("spawn")
    all_rows, total_miss = [], 0
    with ctx.Pool(min(mp.cpu_count(), 8)) as pool:
        for i, (rows, miss) in enumerate(pool.imap_unordered(worker, tasks, chunksize=8), 1):
            all_rows.extend(rows)
            total_miss += miss
            if i % 400 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] 行 {len(all_rows)} | 定位失败 {total_miss} | {time.time() - t0:.0f}s")
    sim = pd.DataFrame(all_rows)
    print(f"[重放] {len(sim)} 行 | entry 定位失败 {total_miss} 笔（应≈0）")

    full_raw = sim[sim["variant"] == "full"].merge(stored, on=["symbol", "sig_dt"], how="left")
    drift_share = float(((full_raw["ret_gross_pct"] - full_raw["stored_gross"]).abs() > 0.5).mean())
    print(f"[漂移] |重算-存储|>0.5% 占比 {drift_share * 100:.1f}%（应≈0.6%）")

    sampler = msf.StableControlSampler()
    frames = {name: ser.variant_frame(sim, meta, name, sampler) for name in VARIANTS}

    # —— 槽数曲线 ——
    curve: dict[str, dict[int, dict]] = {}
    trades_store: dict[tuple[str, int], pd.DataFrame] = {}
    daily_store: dict[tuple[str, int], pd.Series] = {}
    for name, frame in frames.items():
        curve[name] = {}
        for n_slots in SLOT_GRID:
            point, trades, daily = slot_point(frame, n_slots)
            curve[name][n_slots] = point
            trades_store[(name, n_slots)] = trades
            daily_store[(name, n_slots)] = daily
            print(f"  {name:<12} N={n_slots:<3} OOS={point['oos']} taken={point['taken']}")

    full10 = curve["full"][10]["oos"]
    full20 = curve["full"][DEFAULT_SLOTS]
    print(f"[基线复现] full@10 OOS: {full10}（第四轮既知 n=210/6.84/-0.38/t=2.69）")
    print(f"[轮二对照] full@20 OOS t={full20['oos'].get('t')}（既知 ≈3.82 量级）")

    # —— 预声明判定 1：bd_confirm2@20 ——
    bd20 = curve["bd_confirm2"][DEFAULT_SLOTS]
    full20_is_median = full20["is"].get("excess_median") or 0.0
    bd_pass = bool(
        (bd20["oos"].get("excess_median") or -99) > 0
        and (bd20["oos"].get("excess_mean") or 0) > 0
        and (bd20["oos"].get("t") or 0) >= 2
        and (bd20["is"].get("excess_median") or -99) >= full20_is_median
    )
    print(
        f"[判定] bd_confirm2@20 通过={bd_pass} | OOS={bd20['oos']} | IS中位 {bd20['is'].get('excess_median')} vs full {full20_is_median}"
    )

    # —— 预声明判定 2：槽数默认值 ——
    t20 = full20["oos"].get("t") or 0
    med20 = full20["oos"].get("excess_median")
    mean20 = full20["oos"].get("excess_mean") or 0
    slot_winner = None
    for n_slots in SLOT_GRID:
        if n_slots == DEFAULT_SLOTS:
            continue
        o = curve["full"][n_slots]["oos"]
        if (
            (o.get("t") or 0) >= t20
            and (o.get("excess_median") or -99) >= (med20 if med20 is not None else -99)
            and (o.get("excess_mean") or 0) >= 0.8 * mean20
        ):
            slot_winner = n_slots if slot_winner is None else slot_winner
    print(f"[判定] 槽数默认值改变: {slot_winner if slot_winner else '维持 20'}")

    # —— 供给/饱和 ——
    supply = supply_analysis(frames["full"], {n: trades_store[("full", n)] for n in SLOT_GRID})

    # —— 全日历诚实年化（终态配置；bd_confirm2@20 作对照一并报，描述性）——
    calendar = np.sort(pd.read_parquet(spb.CAND_DIR / "panel.parquet", columns=["dt"])["dt"].unique())
    honest = {}
    for name in VARIANTS:
        trades, daily = trades_store[(name, DEFAULT_SLOTS)], daily_store[(name, DEFAULT_SLOTS)]
        first_entry = trades["entry_dt"].min()
        honest[name] = {
            "ALL": honest_curve(daily, trades, calendar, first_entry, calendar[-1], DEFAULT_SLOTS),
            "IS": honest_curve(daily, trades, calendar, first_entry, OOS_START - pd.Timedelta(days=1), DEFAULT_SLOTS),
            "OOS": honest_curve(daily, trades, calendar, OOS_START, calendar[-1], DEFAULT_SLOTS),
        }

    summary = {
        "population": int(len(pop)),
        "entry_miss": int(total_miss),
        "drift_share_gt_0p5": msf.round_float(drift_share * 100, 1),
        "slot_curve": {name: {str(n): pt for n, pt in pts.items()} for name, pts in curve.items()},
        "supply": supply,
        "honest_annualization": honest,
        "verdicts": {"bd_confirm2_at_20": bd_pass, "slot_default_change": slot_winner},
        "full20_is_median": full20_is_median,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


def write_report(s: dict) -> None:
    lines = [
        "# Capacity research (slot curve + bd_confirm2@20 pre-registered recheck + honest annualization)",
        "",
        f"Population {s['population']}; entry-locate misses {s['entry_miss']}; "
        f"drift share {s['drift_share_gt_0p5']}% (expect ~0.6%).",
        "",
        "Pre-declared: bd_confirm2@20 passes iff slot-OOS median>0 AND mean>0 AND t>=2 AND "
        f"IS median >= full@20 IS median ({s['full20_is_median']}). "
        "Slot default changes iff some N has t>=t(20), median>=median(20), mean>=0.8*mean(20).",
        "",
        "## Slot curve",
    ]
    rows = []
    for name, pts in s["slot_curve"].items():
        for n, pt in pts.items():
            o, i = pt["oos"], pt["is"]
            rows.append(
                {
                    "variant": name,
                    "slots": n,
                    "oos_n": o.get("n"),
                    "oos_mean": o.get("excess_mean"),
                    "oos_median": o.get("excess_median"),
                    "oos_t": o.get("t"),
                    "oos_net": o.get("net_mean"),
                    "is_median": i.get("excess_median"),
                    "taken_share%": pt["taken_share"],
                }
            )
    lines += msf.markdown_table(
        rows, ["variant", "slots", "oos_n", "oos_mean", "oos_median", "oos_t", "oos_net", "is_median", "taken_share%"]
    )

    lines += [
        "",
        "## Daily candidate supply (full variant)",
        "",
        f"```json\n{json.dumps(s['supply']['daily_candidates'], ensure_ascii=False)}\n```",
        "",
    ]
    lines += msf.markdown_table(s["supply"]["by_slots"], ["slots", "constrained_days%", "untaken_cands%"])

    lines += ["", "## Honest annualization (full calendar, idle days = 0, 20 slots)"]
    hrows = []
    for name, segs in s["honest_annualization"].items():
        for seg, st in segs.items():
            if st:
                hrows.append({"variant": name, "seg": seg, **st})
    cols = ["variant", "seg", "年化%", "夏普", "最大回撤%", "卡玛", "交易日数", "空仓日占比%", "平均占用槽"]
    lines += msf.markdown_table(hrows, cols)

    lines += ["", "## Verdicts", "", f"```json\n{json.dumps(s['verdicts'], ensure_ascii=False, indent=2)}\n```", ""]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
