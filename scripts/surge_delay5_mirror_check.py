"""Delay5 实验模式的工程镜像交叉核对（daily_scan live 路径 vs 离线 candidates.parquet）

对最近 K 个 candidates.parquet 覆盖的交易日 T，逐日做「真实候选等待 5 天」的状态管理验证：

- live 侧：每只股票 df 截断到 dt ≤ T，走与 daily_scan 实验段完全相同的路径
  （iter_states(tail=160) + surge_live.detect_delay5）→ 当日实验候选集；
- dump 侧：candidates.parquet 过滤 mode=anticipate & delay=5 & dec_dt=T，
  叠加信号门 + 硬过滤（成交额≥1亿、止损带 8-20%；**不含 gap 过滤**——它用 T+1 开盘，
  属执行层规则，live 当日不可知）+ 历史 ST 剔除；
- 同时核对 live 市场状态（surge_live.build_live_panel 尾部面板）与研究
  market_state.parquet 在重叠日期的数值与门判定。

预期：选股集合匹配率 ~100%；不匹配逐笔归因（次日无 bar 致 dump 缺行 / qfq 复权漂移 /
满 500 根宇宙漂移 / 逻辑 bug——最后一类必须为 0）。

    uv run --no-sync python scripts/surge_delay5_mirror_check.py [--days 10]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import pandas as pd
import surge_live as sl
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import surge_pullback_entry_research as spe
import trend_regime as tr

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_delay5_mirror"
TAIL = 160  # 与 daily_scan 一致的快路径窗口
SL_TOL = 0.011  # sl_pct 数值比对容差（双方均 round 2）
AMT_TOL = 0.0011  # amount_e 容差


def _check_one(parquet_path: str, t_list: list[pd.Timestamp]) -> list[dict]:
    """单只股票：对每个 T 截断重放，输出 live 实验候选（与 daily_scan 同路径）。"""
    df = tr.load_stock(parquet_path)
    if df is None:
        return []
    rows = []
    dts = df["dt"]
    for t in t_list:
        if not (dts == t).any():  # T 日无 bar（停牌/未上市）→ 双侧都不会有
            continue
        df_t = df[dts <= t]
        states = tr.iter_states(df_t, with_features=True, tail=TAIL)
        if not states:
            continue
        hit = sl.detect_delay5(states)
        if hit is None or hit["dec_dt"] != t:
            continue
        amount = float(df_t["amount"].iloc[-1]) if "amount" in df_t.columns else 0.0
        hit["symbol"] = df["symbol"].iloc[0]
        hit["amount_e"] = sl.amount_to_e(amount)  # 与 daily_scan / dump 同一 helper，杜绝舍入口径漂移
        hit["n_bars_total"] = int(len(df))
        rows.append(hit)
    return rows


def dump_side(cand: pd.DataFrame, t_list: list[pd.Timestamp], st_intervals: dict) -> pd.DataFrame:
    df = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5) & (cand["dec_dt"].isin(t_list))].copy()
    df = df[spe._signal_gate(df)]
    df["pass_hard"] = (df["amount_e"] >= spb.MIN_AMOUNT_E) & df["sl_pct"].between(spb.STOP_MIN_PCT, spb.STOP_MAX_PCT)
    if st_intervals:
        st_mask = df.apply(lambda r: spb.is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
        df = df[~st_mask]
    return df


def live_side(live_rows: pd.DataFrame, st_intervals: dict) -> pd.DataFrame:
    df = live_rows.copy()
    df["pass_hard"] = [not sl.hard_filter_reasons(s, a) for s, a in zip(df["sl_pct"], df["amount_e"], strict=False)]
    if st_intervals:
        st_mask = df.apply(lambda r: spb.is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
        df = df[~st_mask]
    return df


def compare_day(t: pd.Timestamp, live: pd.DataFrame, dump: pd.DataFrame, cand_raw: pd.DataFrame) -> dict:
    lv = live[live["dec_dt"] == t].set_index("symbol")
    dp = dump[dump["dec_dt"] == t].set_index("symbol")
    matched = sorted(set(lv.index) & set(dp.index))
    live_only = sorted(set(lv.index) - set(dp.index))
    dump_only = sorted(set(dp.index) - set(lv.index))

    value_diffs = []
    for s in matched:
        lr, dr = lv.loc[s], dp.loc[s]
        diffs = {}
        if pd.Timestamp(lr["sig_dt"]) != pd.Timestamp(dr["sig_dt"]):
            diffs["sig_dt"] = (str(lr["sig_dt"]), str(dr["sig_dt"]))
        l_sl = lr["sl_pct"] if lr["sl_pct"] is not None else float("nan")
        if abs((l_sl if l_sl == l_sl else -999) - (dr["sl_pct"] if dr["sl_pct"] == dr["sl_pct"] else -999)) > SL_TOL:
            diffs["sl_pct"] = (l_sl, float(dr["sl_pct"]))
        if abs(lr["amount_e"] - dr["amount_e"]) > AMT_TOL:
            diffs["amount_e"] = (float(lr["amount_e"]), float(dr["amount_e"]))
        if bool(lr["pass_hard"]) != bool(dr["pass_hard"]):
            diffs["pass_hard"] = (bool(lr["pass_hard"]), bool(dr["pass_hard"]))
        if diffs:
            value_diffs.append({"symbol": s, **{k: str(v) for k, v in diffs.items()}})

    attributions = []
    raw_t = cand_raw[
        (cand_raw["mode"] == "anticipate") & (cand_raw["delay"] == 5) & (cand_raw["dec_dt"] == t)
    ].set_index("symbol")
    for s in live_only:
        if s in raw_t.index:
            reason = "dump原始行存在但被信号门拒绝 → 特征值漂移(qfq)"
        elif int(lv.loc[s, "n_bars_total"]) <= tr.MIN_BARS + 10:
            reason = f"宇宙漂移：当前 {int(lv.loc[s, 'n_bars_total'])} 根，dump 时可能 <{tr.MIN_BARS}"
        else:
            reason = "dump 无原始行（dump 时 T 为末根无次日 bar，或结构判定漂移）→ 需逐笔核查"
        attributions.append({"symbol": s, "side": "live_only", "reason": reason})
    for s in dump_only:
        attributions.append(
            {"symbol": s, "side": "dump_only", "reason": "live 未检出 → tail 路径或 qfq 漂移，需逐笔核查"}
        )

    return {
        "date": str(t.date()),
        "live_n": int(len(lv)),
        "dump_n": int(len(dp)),
        "matched": len(matched),
        "live_only": live_only,
        "dump_only": dump_only,
        "value_diffs": value_diffs,
        "attributions": attributions,
    }


def compare_market_state(t_list: list[pd.Timestamp]) -> list[dict]:
    print("[市场状态] live 尾部面板重算 vs 研究 market_state.parquet")
    panel = sl.build_live_panel()
    live_state = sl.live_market_state(panel).set_index("dt")
    ref = pd.read_parquet(msf.OUTPUT_DIR / "market_state.parquet").set_index("dt")
    rows = []
    for t in t_list:
        if t not in live_state.index or t not in ref.index:
            rows.append({"date": str(t.date()), "note": "missing"})
            continue
        lr, rr = live_state.loc[t], ref.loc[t]
        rows.append(
            {
                "date": str(t.date()),
                "high20_ratio_live": round(float(lr["high20_ratio"]), 5),
                "high20_ratio_ref": round(float(rr["high20_ratio"]), 5),
                "high20_diff": round(abs(float(lr["high20_ratio"]) - float(rr["high20_ratio"])), 5),
                "ret20med_diff": round(abs(float(lr["mkt_ret20_median"]) - float(rr["mkt_ret20_median"])), 5),
                "index_above_ma20_live": int(lr["ew_index_above_ma20"]),
                "index_above_ma20_ref": int(rr["ew_index_above_ma20"]),
                "gate_live": sl.market_gate_open(lr),
                "gate_ref": sl.market_gate_open(rr),
                "gate_match": sl.market_gate_open(lr) == sl.market_gate_open(rr),
            }
        )
    return rows


def write_report(summary: dict) -> None:
    lines = [
        "# Delay5 engineering mirror check",
        "",
        f"Window: last {len(summary['dates'])} trading days covered by candidates.parquet "
        f"({summary['dates'][0]} → {summary['dates'][-1]}).",
        "Live path = truncate-to-T + iter_states(tail=160) + surge_live.detect_delay5 (identical to daily_scan).",
        "Dump path = candidates.parquet (anticipate, delay=5) + signal gate + hard filters (no gap) + ST.",
        "",
        "## Selection-set comparison",
    ]
    day_rows = [
        {
            "date": d["date"],
            "live_n": d["live_n"],
            "dump_n": d["dump_n"],
            "matched": d["matched"],
            "live_only": len(d["live_only"]),
            "dump_only": len(d["dump_only"]),
            "value_diffs": len(d["value_diffs"]),
        }
        for d in summary["days"]
    ]
    lines.extend(
        msf.markdown_table(day_rows, ["date", "live_n", "dump_n", "matched", "live_only", "dump_only", "value_diffs"])
    )
    lines.extend(
        [
            "",
            f"**Totals**: matched {summary['total_matched']} / live {summary['total_live']} / dump {summary['total_dump']}; mismatches {summary['total_mismatch']}.",
        ]
    )
    if summary["all_attributions"]:
        lines.extend(["", "## Mismatch attributions"])
        lines.extend(msf.markdown_table(summary["all_attributions"], ["symbol", "side", "reason"]))
    if summary["all_value_diffs"]:
        lines.extend(["", "## Matched-row value diffs"])
        for d in summary["all_value_diffs"]:
            lines.append(f"- {d}")
    lines.extend(["", "## Market-state comparison (live recompute vs research parquet)"])
    lines.extend(
        msf.markdown_table(
            summary["market_state"],
            [
                "date",
                "high20_ratio_live",
                "high20_ratio_ref",
                "high20_diff",
                "ret20med_diff",
                "index_above_ma20_live",
                "index_above_ma20_ref",
                "gate_match",
            ],
        )
    )
    lines.extend(["", f"**Verdict**: {summary['verdict']}"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    cand_raw = pd.read_parquet(msf.CAND_DIR / "candidates.parquet")
    cand_max = cand_raw["dec_dt"].max()
    ref_state = pd.read_parquet(msf.OUTPUT_DIR / "market_state.parquet")
    t_list = [pd.Timestamp(x) for x in ref_state.loc[ref_state["dt"] <= cand_max, "dt"].tail(args.days)]
    print(
        f"[窗口] {t_list[0].date()} → {t_list[-1].date()}（{len(t_list)} 个交易日，candidates 覆盖至 {cand_max.date()}）"
    )

    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[live] {len(files)} 只 × {len(t_list)} 日截断重放（tail={TAIL}）...")
    ctx = mp.get_context("spawn")
    live_rows = []
    with ctx.Pool(n_workers) as pool:
        worker = partial(_check_one, t_list=t_list)
        for i, res in enumerate(pool.imap_unordered(worker, files, chunksize=10), 1):
            live_rows.extend(res)
            if i % 500 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] live 候选 {len(live_rows)} | {time.time() - t0:.0f}s")
    live = (
        pd.DataFrame(live_rows)
        if live_rows
        else pd.DataFrame(columns=["symbol", "sig_dt", "dec_dt", "sl_pct", "amount_e", "n_bars_total"])
    )

    st_intervals = spb.load_st_intervals()
    live = live_side(live, st_intervals) if len(live) else live.assign(pass_hard=[])
    dump = dump_side(cand_raw, t_list, st_intervals)
    print(f"[对比] live {len(live)} 行 vs dump {len(dump)} 行")

    days = [compare_day(t, live, dump, cand_raw) for t in t_list]
    market_rows = compare_market_state(t_list)

    total_live = sum(d["live_n"] for d in days)
    total_dump = sum(d["dump_n"] for d in days)
    total_matched = sum(d["matched"] for d in days)
    total_mismatch = sum(len(d["live_only"]) + len(d["dump_only"]) for d in days)
    # 市场状态必须每个 T 都有有效对比行：缺失日不允许靠 all(空集)=True 误判 PASS
    valid_market = [r for r in market_rows if "gate_match" in r]
    missing_market = len(t_list) - len(valid_market)
    gates_ok = missing_market == 0 and all(r["gate_match"] for r in valid_market)
    gate_note = (
        "all ok" if gates_ok else (f"{missing_market} day(s) MISSING" if missing_market else "DIVERGED")
    )
    verdict = (
        "PASS — selection sets identical and market-state gates agree"
        if total_mismatch == 0 and total_matched == total_live == total_dump and gates_ok
        else f"ATTENTION — {total_mismatch} set mismatches (see attributions), market-state={gate_note}"
    )
    summary = {
        "dates": [str(t.date()) for t in t_list],
        "days": days,
        "total_live": total_live,
        "total_dump": total_dump,
        "total_matched": total_matched,
        "total_mismatch": total_mismatch,
        "all_attributions": [a for d in days for a in d["attributions"]],
        "all_value_diffs": [v for d in days for v in d["value_diffs"]],
        "market_state": market_rows,
        "verdict": verdict,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"\n[判定] {verdict}")
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
