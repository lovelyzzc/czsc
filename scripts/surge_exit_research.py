"""退出/持仓规则研究：修肥尾中位数（time-stop）+ 防假破坏护尾部（破坏确认）

人群 = delay5 基线 3185 笔（成员资格沿用已验证 dump），价格/状态/止损参考全部从
**当前 cache** 重放重导（qfq 漂移免疫），退出循环复刻 `surge_candidates_dump._simulate_full`
的日内优先序（盘中 SL2 → 收盘 trail → state 次日开盘 → max_hold 60），变体注入：

| 变体 | 定义 |
|---|---|
| full              | 基线复算（漂移校准基准） |
| full_ts5/_ts10    | + 第 k 个持有日收盘仍 ≤ 入场价 → 次日开盘离场（当日各检查之后评估） |
| bd_confirm2       | 破坏(10)需连续 2 bar 处于 {9,10} 才出清；背驰(9) 单 bar 即出 |
| sell_below_ma20   | state 出清(9/10) 仅当收盘 < MA20 才执行 |
| trail12/trail24   | 跟踪 18%→12%/24%（敏感性） |
| no_trail          | 去掉跟踪（诊断 trail 是否砍尾部） |
| combo_*           | ts × 防假破坏 的 4 个组合（仅预声明规则选中的那个参与判定，其余列附录） |

预声明判定（先于运行）：变体通过 = 10 槽镜像 OOS 超额中位数 > 0 且 均值 > 0 且 t ≥ 2
且 IS 段中位数 ≥ 重算基线 IS 中位数。多变体通过取机制最简（ts > state 改动 > trail 改动）。
组合仅当 一个 ts 与 一个防假破坏 变体都单独通过时，取各自 t 最高者的组合作确认检验。
全部失败 → 「中位数问题源于入场人群，退出不可救」，如实记录。

交叉校验：重算 full 的 10 槽 OOS 应近似复现 n≈210 / +7.58 / 中位数 -0.19；
漂移 |重算-存储| ret_gross > 0.5% 占比应 < 5%；entry_dt 定位匹配率 ≈ 100%。

    uv run --no-sync python scripts/surge_exit_research.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import surge_selection_audit as ssa
import trend_regime as tr

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_exit"
MAX_HOLD = 60
SELL_SET = {9, 10}
FIRST_TEST_YEAR = 2024

# (trail_pct | None, state_mode, ts_k | None)
VARIANTS: dict[str, tuple[float | None, str, int | None]] = {
    "full": (0.18, "base", None),
    "full_ts5": (0.18, "base", 5),
    "full_ts10": (0.18, "base", 10),
    "bd_confirm2": (0.18, "bd2", None),
    "sell_below_ma20": (0.18, "ma20", None),
    "trail12": (0.12, "base", None),
    "trail24": (0.24, "base", None),
    "no_trail": (None, "base", None),
    "combo_ts5_bd2": (0.18, "bd2", 5),
    "combo_ts5_ma20": (0.18, "ma20", 5),
    "combo_ts10_bd2": (0.18, "bd2", 10),
    "combo_ts10_ma20": (0.18, "ma20", 10),
}
TS_VARIANTS = ["full_ts5", "full_ts10"]
ANTIFB_VARIANTS = ["bd_confirm2", "sell_below_ma20"]


def _state_exit_today(j: int, mode: str, regime_by_idx: dict, c: np.ndarray, ma20: np.ndarray) -> bool:
    rg = regime_by_idx.get(j)
    if rg not in SELL_SET:
        return False
    if mode == "base":
        return True
    if mode == "bd2":  # 背驰 9 单 bar 即出；破坏 10 需前一 bar 也在 {9,10}
        return rg == 9 or regime_by_idx.get(j - 1) in SELL_SET
    if mode == "ma20":
        return bool(ma20[j] == ma20[j] and c[j] < ma20[j])
    raise ValueError(mode)


def _simulate(entry_idx: int, entry_price: float, sl_ref: float, params: tuple, ind: dict, regime_by_idx: dict):
    """复刻 surge_candidates_dump._simulate_full 的优先序（SL2→trail→state→[ts]→max_hold）。"""
    trail_pct, state_mode, ts_k = params
    n = ind["n"]
    o, c, lo, ma20 = ind["open"], ind["close"], ind["low"], ind["ma20"]
    peak = entry_price
    last_j = min(entry_idx + MAX_HOLD, n) - 1
    for j in range(entry_idx, last_j + 1):
        peak = max(peak, c[j])
        if j == entry_idx:
            continue
        if sl_ref == sl_ref and lo[j] <= sl_ref:
            return j, min(o[j], sl_ref), "sl2"
        if trail_pct is not None and peak > entry_price and (peak - c[j]) / peak >= trail_pct:
            return j, c[j], "trail"
        if _state_exit_today(j, state_mode, regime_by_idx, c, ma20):
            return (j, o[j + 1], "state") if j + 1 < n else (j, c[j], "state")
        if ts_k is not None and j == entry_idx + ts_k and c[j] <= entry_price:
            return (j, o[j + 1], "ts") if j + 1 < n else (j, c[j], "ts")
    return last_j, c[last_j], "max_hold"


def _replay_symbol(args: tuple, variants: dict) -> tuple[list[dict], int]:
    """单只股票：重放全部交易 × 全部变体。返回 (rows, entry定位失败数)。"""
    symbol, trades = args
    path = tr.DATA_DIR / f"{symbol}.parquet"
    df = tr.load_stock(path)
    if df is None:
        return [], len(trades)
    states = tr.iter_states(df)
    if not states:
        return [], len(trades)
    ind = tr.compute_indicators(df)
    dates = ind["dates"]
    regime_by_idx = {s.idx: s.regime for s in states}
    snap_by_idx = {s.idx: s for s in states}

    rows, misses = [], 0
    for t in trades:
        e_dt = np.datetime64(t["entry_dt"])
        pos = int(np.searchsorted(dates, e_dt))
        if pos >= len(dates) or dates[pos] != e_dt:
            misses += 1
            continue
        snap = snap_by_idx.get(pos - 1)  # 决策日快照
        if snap is None:
            misses += 1
            continue
        sl = snap.sl_ref if snap.sl_ref == snap.sl_ref else snap.zd
        sl = sl if (sl == sl and sl > 0) else np.nan
        entry_price = float(ind["open"][pos])
        for name, params in variants.items():
            exit_idx, exit_price, reason = _simulate(pos, entry_price, sl, params, ind, regime_by_idx)
            rows.append(
                {
                    "symbol": symbol,
                    "sig_dt": t["sig_dt"],
                    "dec_dt": t["dec_dt"],
                    "entry_dt": t["entry_dt"],
                    "variant": name,
                    "exit_dt": pd.Timestamp(dates[exit_idx]),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(float(exit_price), 4),
                    "ret_gross_pct": round((exit_price / entry_price - 1) * 100, 3),
                    "hold_days": exit_idx - pos,
                    "exit_reason": reason,
                }
            )
    return rows, misses


def variant_frame(sim: pd.DataFrame, meta: pd.DataFrame, name: str, sampler) -> pd.DataFrame:
    v = sim[sim["variant"] == name].drop(columns=["variant"])
    v = v.merge(meta, on=["symbol", "sig_dt", "dec_dt", "entry_dt"], how="inner")
    v["ret_net_pct"] = ((1 + v["ret_gross_pct"] / 100) * (1 - spb.SELL_COST) / (1 + spb.BUY_COST) - 1) * 100
    v = msf.add_excess(v, sampler)
    return v.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)


def seg_stats(trades: pd.DataFrame, seg: str) -> dict:
    sub = trades[trades["seg"] == seg]
    v = sub["excess_pct"].dropna()
    out = {"n": int(len(sub)), "excess_valid": int(len(v))}
    if len(v) >= 30:
        out.update(
            {
                "excess_mean": msf.round_float(v.mean()),
                "excess_median": msf.round_float(v.median()),
                "t": msf.round_float(msf.t_stat(v)),
            }
        )
    if len(sub):
        out["net_mean"] = msf.round_float(sub["ret_net_pct"].mean())
        out["net_median"] = msf.round_float(sub["ret_net_pct"].median())
        out["hold"] = msf.round_float(sub["hold_days"].mean(), 1)
    return out


def evaluate_variant(vdf: pd.DataFrame, slots: int = 10) -> dict:
    trades, _ = spb.simulate_slots(vdf, slots)
    return {
        "pair_oos": seg_stats(vdf, "test"),
        "slot_oos": seg_stats(trades, "test"),
        "slot_is": seg_stats(trades, "train"),
        "exit_dist": vdf["exit_reason"].value_counts(normalize=True).round(3).to_dict(),
        "slot_trades": trades,
    }


def tail_retention(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """以重算 full 的年内 top20% 超额笔为尾部集，看各变体对这些笔的影响。"""
    base = frames["full"].copy()
    base = base[base["excess_pct"].notna()]
    base["y"] = base.groupby("year")["excess_pct"].rank(pct=True)
    tail_keys = base.loc[base["y"] >= 0.8, ["symbol", "sig_dt"]]
    rows = []
    for name, vdf in frames.items():
        sub = vdf.merge(tail_keys, on=["symbol", "sig_dt"], how="inner")
        rows.append(
            {
                "variant": name,
                "tail_n": int(len(sub)),
                "tail_gross_mean": msf.round_float(sub["ret_gross_pct"].mean()),
                "tail_hold": msf.round_float(sub["hold_days"].mean(), 1),
                "tail_exit_ts%": msf.round_float((sub["exit_reason"] == "ts").mean() * 100, 1),
            }
        )
    return rows


def passes(metrics: dict, base_is_median: float) -> bool:
    oos, is_ = metrics["slot_oos"], metrics["slot_is"]
    return bool(
        (oos.get("excess_median") or -99) > 0
        and (oos.get("excess_mean") or 0) > 0
        and (oos.get("t") or 0) >= 2
        and (is_.get("excess_median") or -99) >= base_is_median
    )


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
    worker = partial(_replay_symbol, variants=VARIANTS)
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

    # —— 漂移校准（full vs dump 存储）——
    full_raw = sim[sim["variant"] == "full"].merge(stored, on=["symbol", "sig_dt"], how="left")
    drift = (full_raw["ret_gross_pct"] - full_raw["stored_gross"]).abs()
    drift_share = float((drift > 0.5).mean())
    print(f"[漂移] |重算-存储|>0.5% 占比 {drift_share * 100:.1f}%（门槛 <5%）")

    sampler = msf.StableControlSampler()
    frames, results = {}, {}
    for name in VARIANTS:
        frames[name] = variant_frame(sim, meta, name, sampler)
        results[name] = evaluate_variant(frames[name])
        m = results[name]
        print(f"  {name:<18} slotOOS={m['slot_oos']} | pairOOS n={m['pair_oos']['n']}")

    base_oos = results["full"]["slot_oos"]
    base_is_median = results["full"]["slot_is"].get("excess_median") or 0.0
    print(f"[基线复现] 重算 full 10槽 OOS: {base_oos}（既知 n=210/+7.58/中位 -0.19）")

    # —— 预声明判定 ——
    core_names = ["full_ts5", "full_ts10", "bd_confirm2", "sell_below_ma20", "trail12", "trail24", "no_trail"]
    verdicts = {name: passes(results[name], base_is_median) for name in core_names}
    ts_pass = [n for n in TS_VARIANTS if verdicts[n]]
    fb_pass = [n for n in ANTIFB_VARIANTS if verdicts[n]]
    combo_judged = None
    if ts_pass and fb_pass:
        best_ts = max(ts_pass, key=lambda n: results[n]["slot_oos"].get("t") or -99)
        best_fb = max(fb_pass, key=lambda n: results[n]["slot_oos"].get("t") or -99)
        combo_judged = f"combo_{best_ts.replace('full_', '')}_{'bd2' if best_fb == 'bd_confirm2' else 'ma20'}"
        verdicts[combo_judged] = passes(results[combo_judged], base_is_median)
    print(f"[判定] {verdicts} | 组合参与判定: {combo_judged}")

    # —— 通过项的 20 槽广度数字 ——
    slots20 = {}
    for name, ok in verdicts.items():
        if ok:
            trades20, _ = spb.simulate_slots(frames[name], 20)
            slots20[name] = seg_stats(trades20, "test")

    tail_rows = tail_retention(frames)

    summary = {
        "population": int(len(pop)),
        "entry_miss": int(total_miss),
        "drift_share_gt_0p5": msf.round_float(drift_share * 100, 1),
        "baseline_slot_oos": base_oos,
        "baseline_is_median": base_is_median,
        "variants": {
            name: {
                "slot_oos": results[name]["slot_oos"],
                "slot_is": results[name]["slot_is"],
                "pair_oos": results[name]["pair_oos"],
                "exit_dist": results[name]["exit_dist"],
            }
            for name in VARIANTS
        },
        "verdicts": verdicts,
        "combo_judged": combo_judged,
        "slots20_for_passing": slots20,
        "tail_retention": tail_rows,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


def write_report(s: dict) -> None:
    lines = [
        "# Exit-rule research (delay5 baseline population, replayed from current cache)",
        "",
        f"Population {s['population']}; entry-locate misses {s['entry_miss']}; "
        f"drift share |recomputed-stored|>0.5%: {s['drift_share_gt_0p5']}% (gate <5%).",
        "",
        "Pre-declared pass: slot-mirror OOS excess median>0 AND mean>0 AND t>=2 AND IS median >= baseline IS median "
        f"({s['baseline_is_median']}).",
        "",
        "## Variants (10-slot mirror)",
    ]
    rows = []
    for name, v in s["variants"].items():
        oos, is_ = v["slot_oos"], v["slot_is"]
        rows.append(
            {
                "variant": name,
                "oos_n": oos.get("n"),
                "oos_mean": oos.get("excess_mean"),
                "oos_median": oos.get("excess_median"),
                "oos_t": oos.get("t"),
                "oos_net": oos.get("net_mean"),
                "hold": oos.get("hold"),
                "is_median": is_.get("excess_median"),
                "pass": s["verdicts"].get(name, ""),
            }
        )
    lines.extend(
        msf.markdown_table(
            rows, ["variant", "oos_n", "oos_mean", "oos_median", "oos_t", "oos_net", "hold", "is_median", "pass"]
        )
    )
    lines.extend(["", "## Pair-level OOS (all candidates, no slot interaction)"])
    prow = [
        {
            "variant": name,
            **{k: v["pair_oos"].get(k) for k in ["n", "excess_mean", "excess_median", "t", "net_mean", "hold"]},
        }
        for name, v in s["variants"].items()
    ]
    lines.extend(msf.markdown_table(prow, ["variant", "n", "excess_mean", "excess_median", "t", "net_mean", "hold"]))
    lines.extend(["", "## Exit-reason distribution"])
    erow = [{"variant": name, **v["exit_dist"]} for name, v in s["variants"].items()]
    keys = sorted({k for r in erow for k in r if k != "variant"})
    lines.extend(msf.markdown_table(erow, ["variant", *keys]))
    lines.extend(["", "## Tail retention (full-variant within-year top-20% excess trades)"])
    lines.extend(
        msf.markdown_table(s["tail_retention"], ["variant", "tail_n", "tail_gross_mean", "tail_hold", "tail_exit_ts%"])
    )
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            "```json",
            json.dumps(
                {
                    "verdicts": s["verdicts"],
                    "combo_judged": s["combo_judged"],
                    "slots20_for_passing": s["slots20_for_passing"],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "```",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
