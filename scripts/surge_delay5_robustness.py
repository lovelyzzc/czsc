"""Delay5 回踩买点稳健性审计（接续 SURGE_REGIME_PULLBACK_ENTRY_2026-06-10.md）

`anticipate + 市场门 + delay5` 在进入 daily_scan 工程镜像前的敏感性体检。
全部计算只读现成 candidates.parquet / panel.parquet / market_state.parquet，
不改 FSM、不改 surge_onset、不引入任何新阈值。

预声明判定标准（先于运行声明，跑完不许改）：

- R1 delay 曲线 {0,1,2,3,5,7,10} × 市场门：delay3 OOS 超额 > 0；
  若 delay3 ≤ 0 且仅 delay5 一点 t≥2 → 刀刃红旗。
- R2 槽位敏感性 delay5 × {5,10,20} 槽：20 槽 OOS 超额 > 0 且 t ≥ 1.5。
- R3 排序敏感性 delay5 × 5 个 seed 随机排序（替代 priority 排序）：
  5 次 OOS 超额均值全部 > 0，min t ≥ 1.0。
- R4 集中度（delay5 10 槽 OOS）：超额中位数 > 0 且 10% 截尾均值 > 0。
- R5 机制诊断（不设门槛，仅记录）：(a) 信号日→决策日收益分桶看超额来源；
  (b) state_delay0 集合到 delay5 的存活率分解。

    uv run --no-sync python scripts/surge_delay5_robustness.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import surge_market_state_filter as msf
import surge_portfolio_backtest as spb
import surge_pullback_entry_research as spe

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_delay5_robustness"

DELAY_GRID = [0, 1, 2, 3, 5, 7, 10]
SLOT_GRID = [5, 10, 20]
ORDER_SEEDS = [11, 22, 33, 44, 55]
FIRST_TEST_YEAR = 2024
MARKET_GATE = "high20_and_index"  # 沿用前推验证选定门，不另选


def trimmed_mean(values: pd.Series, trim: float = 0.1) -> float:
    v = np.sort(values.dropna().to_numpy(dtype=float))
    k = int(len(v) * trim)
    v = v[k : len(v) - k] if len(v) > 2 * k else v
    return float(v.mean()) if len(v) else np.nan


def metrics_row(name: str, metrics: dict, candidates: int | None = None) -> dict:
    row = spe._row_from_metrics(name, "", metrics, candidates)
    row.pop("label", None)
    return row


def scoped_metrics(record: dict, years: list[int]) -> dict:
    return spe._trade_metrics(record["trades"], record["daily"], years)


def run_delay_curve(base: pd.DataFrame, sampler, oos_years: list[int], is_years: list[int]) -> dict:
    print("[R1] delay 曲线 × 市场门")
    rows_oos, rows_is, records = [], [], {}
    for d in DELAY_GRID:
        variant = spe.EntryVariant(f"state_delay{d}", (d,), MARKET_GATE)
        candidates = spe.variant_candidates(base, variant)
        record = spe.simulate_variant(candidates, sampler, 10)
        records[d] = record
        rows_oos.append(metrics_row(f"delay{d}", scoped_metrics(record, oos_years), record["candidates"]))
        rows_is.append(metrics_row(f"delay{d}", scoped_metrics(record, is_years)))
        print(f"  delay{d:>2}: cand={record['candidates']:>5} OOS={rows_oos[-1]}")
    return {"oos_rows": rows_oos, "is_rows": rows_is, "records": records}


def run_slot_sensitivity(base: pd.DataFrame, sampler, oos_years: list[int]) -> list[dict]:
    print("[R2] delay5 槽位敏感性")
    variant = spe.EntryVariant("state_delay5", (5,), MARKET_GATE)
    candidates = spe.variant_candidates(base, variant)
    rows = []
    for slots in SLOT_GRID:
        record = spe.simulate_variant(candidates, sampler, slots)
        row = metrics_row(f"slots{slots}", scoped_metrics(record, oos_years), record["candidates"])
        rows.append(row)
        print(f"  slots={slots:>2}: {row}")
    return rows


def run_order_sensitivity(base: pd.DataFrame, sampler, oos_years: list[int]) -> list[dict]:
    print("[R3] delay5 排序敏感性（随机排序替代 priority）")
    variant = spe.EntryVariant("state_delay5", (5,), MARKET_GATE)
    candidates = spe.variant_candidates(base, variant)
    rows = []
    for seed in ORDER_SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = candidates.copy()
        shuffled["priority"] = rng.random(len(shuffled))
        shuffled = shuffled.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)
        trades, daily = spb.simulate_slots(shuffled, 10)
        trades = msf.add_excess(trades, sampler) if len(trades) else trades.assign(excess_pct=[])
        row = metrics_row(f"seed{seed}", spe._trade_metrics(trades, daily, oos_years))
        rows.append(row)
        print(f"  seed={seed}: {row}")
    return rows


def run_concentration(delay5_record: dict, oos_years: list[int]) -> dict:
    print("[R4] delay5 集中度（10 槽 OOS）")
    trades = delay5_record["trades"]
    oos = trades[trades["year"].isin(oos_years)].copy()
    excess = oos["excess_pct"].dropna()
    net = oos["ret_net_pct"].dropna().sort_values(ascending=False)
    pos_sum = float(net[net > 0].sum())
    top10_sum = float(net.head(10).sum())
    out = {
        "n": int(len(oos)),
        "excess_median_pct": msf.round_float(excess.median()),
        "excess_trimmed_mean_pct": msf.round_float(trimmed_mean(excess)),
        "excess_mean_pct": msf.round_float(excess.mean()),
        "net_top10_sum_pct": msf.round_float(top10_sum),
        "net_positive_sum_pct": msf.round_float(pos_sum),
        "net_top10_share_of_positive": msf.round_float(top10_sum / pos_sum, 3) if pos_sum > 0 else None,
        "net_total_sum_pct": msf.round_float(net.sum()),
    }
    print(f"  {out}")
    return out


def run_pullback_mechanism(base: pd.DataFrame, panel: pd.DataFrame, sampler, oos_years: list[int]) -> list[dict]:
    """R5a：delay5 候选按 信号日→决策日 收益分桶，看超额集中在真回踩还是续涨。"""
    print("[R5a] 机制诊断：信号日→决策日 收益分桶（候选级）")
    variant = spe.EntryVariant("state_delay5", (5,), MARKET_GATE)
    candidates = spe.variant_candidates(base, variant)
    closes = panel[["symbol", "dt", "close"]]
    candidates = candidates.merge(
        closes.rename(columns={"dt": "sig_dt", "close": "sig_close"}), on=["symbol", "sig_dt"], how="left"
    )
    candidates = candidates.merge(
        closes.rename(columns={"dt": "dec_dt", "close": "dec_close"}), on=["symbol", "dec_dt"], how="left"
    )
    candidates["sig_to_dec_pct"] = (candidates["dec_close"] / candidates["sig_close"] - 1) * 100
    candidates = msf.add_excess(candidates, sampler)
    scoped = candidates[candidates["year"].isin(oos_years)].copy()
    edges = [-np.inf, -5, -2, 0, 2, 5, np.inf]
    labels = ["<-5%", "-5~-2%", "-2~0%", "0~2%", "2~5%", ">5%"]
    scoped["bucket"] = pd.cut(scoped["sig_to_dec_pct"], bins=edges, labels=labels)
    rows = []
    for bucket, group in scoped.groupby("bucket", observed=True):
        values = group["excess_pct"].dropna()
        if len(values) < 30:
            rows.append({"bucket": str(bucket), "n": int(len(values))})
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "n": int(len(values)),
                "excess_mean_pct": msf.round_float(values.mean()),
                "t": msf.round_float(msf.t_stat(values)),
                "net_mean_pct": msf.round_float(group["ret_net_pct"].mean()),
            }
        )
        print(f"  {rows[-1]}")
    return rows


def run_survival(base: pd.DataFrame, cand_raw: pd.DataFrame, oos_years: list[int]) -> dict:
    """R5b：state_delay0 候选（信号门+硬过滤+市场门）到 delay5 的存活率分解。"""
    print("[R5b] 机制诊断：delay0 → delay5 存活率（OOS 候选级）")
    gates = spe.market_gate_specs()
    d0 = base[(base["delay"] == 0) & gates[MARKET_GATE](base)]
    d0 = d0[d0["year"].isin(oos_years)][["symbol", "sig_dt"]].drop_duplicates()

    raw5 = cand_raw[(cand_raw["mode"] == "anticipate") & (cand_raw["delay"] == 5)][["symbol", "sig_dt"]]
    raw5 = raw5.drop_duplicates().assign(s1_exists=True)
    base5 = base[base["delay"] == 5][["symbol", "sig_dt"]].drop_duplicates().assign(s2_hard=True)
    gate5 = base[(base["delay"] == 5) & gates[MARKET_GATE](base)]
    gate5 = gate5[["symbol", "sig_dt"]].drop_duplicates().assign(s3_market=True)

    j = d0.merge(raw5, on=["symbol", "sig_dt"], how="left")
    j = j.merge(base5, on=["symbol", "sig_dt"], how="left")
    j = j.merge(gate5, on=["symbol", "sig_dt"], how="left")
    n = len(j)
    out = {
        "delay0_signals": n,
        "s1_still_in_family_pct": msf.round_float(j["s1_exists"].fillna(False).mean() * 100, 1),
        "s2_plus_hard_filters_pct": msf.round_float(j["s2_hard"].fillna(False).mean() * 100, 1),
        "s3_plus_market_gate_pct": msf.round_float(j["s3_market"].fillna(False).mean() * 100, 1),
    }
    # 反向：delay5 集合中有多少 (symbol, sig_dt) 不在 delay0 集合（新进者 vs 存活者）
    g5 = gate5.merge(d0.assign(in_d0=True), on=["symbol", "sig_dt"], how="left")
    out["delay5_signals"] = int(len(gate5))
    out["delay5_from_delay0_pct"] = msf.round_float(g5["in_d0"].fillna(False).mean() * 100, 1)
    print(f"  {out}")
    return out


def evaluate_verdicts(summary: dict) -> dict:
    oos = {row["name"]: row for row in summary["r1_delay_curve_oos"]}
    d3 = oos.get("delay3", {})
    d5 = oos.get("delay5", {})
    r1 = (d3.get("excess") or 0) > 0
    slots = {row["name"]: row for row in summary["r2_slot_rows"]}
    s20 = slots.get("slots20", {})
    r2 = (s20.get("excess") or 0) > 0 and (s20.get("t") or 0) >= 1.5
    order_rows = summary["r3_order_rows"]
    r3 = (
        all((row.get("excess") or 0) > 0 for row in order_rows)
        and min((row.get("t") or -99) for row in order_rows) >= 1.0
    )
    conc = summary["r4_concentration"]
    r4 = (conc.get("excess_median_pct") or 0) > 0 and (conc.get("excess_trimmed_mean_pct") or 0) > 0
    knife_edge = not r1 and (d5.get("t") or 0) >= 2
    return {
        "R1_delay3_positive": bool(r1),
        "R1_knife_edge_flag": bool(knife_edge),
        "R2_slots20": bool(r2),
        "R3_order_all_positive": bool(r3),
        "R4_concentration": bool(r4),
        "all_pass": bool(r1 and r2 and r3 and r4),
    }


def write_report(summary: dict) -> None:
    lines = [
        "# Delay5 robustness audit",
        "",
        "Pre-declared criteria: R1 delay3 OOS excess > 0; R2 slots20 excess > 0 & t >= 1.5; "
        "R3 all 5 random orderings excess > 0 & min t >= 1.0; R4 excess median > 0 & 10% trimmed mean > 0.",
        "",
        "## R1 Delay curve (market gate, 10 slots, OOS)",
    ]
    lines.extend(
        msf.markdown_table(
            summary["r1_delay_curve_oos"], ["name", "candidates", "n", "excess", "t", "net", "annual", "mdd"]
        )
    )
    lines.extend(["", "## R1 Delay curve (IS <=2023)"])
    lines.extend(msf.markdown_table(summary["r1_delay_curve_is"], ["name", "n", "excess", "t", "net", "annual", "mdd"]))
    lines.extend(["", "## R2 Slot sensitivity (delay5, OOS)"])
    lines.extend(
        msf.markdown_table(summary["r2_slot_rows"], ["name", "candidates", "n", "excess", "t", "net", "annual", "mdd"])
    )
    lines.extend(["", "## R3 Ordering sensitivity (delay5, 10 slots, OOS)"])
    lines.extend(msf.markdown_table(summary["r3_order_rows"], ["name", "n", "excess", "t", "net", "annual", "mdd"]))
    lines.extend(
        [
            "",
            "## R4 Concentration (delay5, 10 slots, OOS)",
            "",
            "```json",
            json.dumps(summary["r4_concentration"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    lines.extend(["", "## R5a Signal->decision return buckets (candidate level, OOS)"])
    lines.extend(
        msf.markdown_table(summary["r5a_pullback_buckets"], ["bucket", "n", "excess_mean_pct", "t", "net_mean_pct"])
    )
    lines.extend(
        [
            "",
            "## R5b Survival decomposition (OOS)",
            "",
            "```json",
            json.dumps(summary["r5b_survival"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    lines.extend(
        ["", "## Verdicts", "", "```json", json.dumps(summary["verdicts"], ensure_ascii=False, indent=2), "```"]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market_path = msf.OUTPUT_DIR / "market_state.parquet"
    market = pd.read_parquet(market_path) if market_path.exists() else msf.load_market_state()
    cand_raw = pd.read_parquet(msf.CAND_DIR / "candidates.parquet")
    panel = pd.read_parquet(msf.CAND_DIR / "panel.parquet", columns=["symbol", "dt", "close"])
    st_intervals = spb.load_st_intervals()
    sampler = msf.StableControlSampler()

    print("[prepare] anticipate base（信号门+硬过滤+ST，全 delay）")
    base = spe._prepare_base(cand_raw, market, st_intervals)
    years = sorted(int(y) for y in base["year"].dropna().unique())
    oos_years = [y for y in years if y >= FIRST_TEST_YEAR]
    is_years = [y for y in years if y < FIRST_TEST_YEAR]
    print(f"  base={len(base)} 行 | IS {is_years} | OOS {oos_years}")

    r1 = run_delay_curve(base, sampler, oos_years, is_years)
    summary = {
        "market_gate": MARKET_GATE,
        "oos_years": oos_years,
        "r1_delay_curve_oos": r1["oos_rows"],
        "r1_delay_curve_is": r1["is_rows"],
        "r2_slot_rows": run_slot_sensitivity(base, sampler, oos_years),
        "r3_order_rows": run_order_sensitivity(base, sampler, oos_years),
        "r4_concentration": run_concentration(r1["records"][5], oos_years),
        "r5a_pullback_buckets": run_pullback_mechanism(base, panel, sampler, oos_years),
        "r5b_survival": run_survival(base, cand_raw, oos_years),
    }
    summary["verdicts"] = evaluate_verdicts(summary)
    print(f"\n[verdicts] {summary['verdicts']}")

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
