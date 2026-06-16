"""Industry-linkage research for the surge-regime delay5 strategy.

Motivation:
    Previous daily price/structure factors failed to identify which delay5
    candidates become tail winners. This script tests the lowest-cost new data
    source proposed in SURGE_REGIME_NEXT_DIRECTIONS_2026-06-12.md:
    industry/sector linkage from stock_basic.industry plus the existing daily
    cache.

Pre-declared protocol:
- Industry classification: tinyshare stock_basic(exchange="", list_status="L,D")
  ``industry`` field, cached locally.
- Base strategy universe: existing anticipate + delay5 candidate dump, current
  BASE gates (signal gates, hard filters, market gate, ST/gap filters).
- Industry features are causal at decision close T:
  industry ret20 mean/median, industry high20 ratio, industry limit-up ratio,
  and same-day industry delay5 candidate density.
- Candidate-layer screen: OOS excess by industry feature buckets.
- Portfolio-layer screen: 10-slot mirror using existing priority order and
  candidate-level excess cache.

Pass criterion for moving an industry gate to a fuller mirror/walk-forward:
- OOS 10-slot trade count >= 60;
- OOS trade excess mean >= baseline mean + 1.5 pp;
- OOS trade excess median >= baseline median;
- OOS t >= 2;
- IS trade excess mean >= baseline IS mean.

This is a first-stage research gate. It does not change live strategy behavior.

    .venv/bin/python scripts/surge_industry_linkage_research.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPTS_DIR / "_output" / "surge_industry_linkage"
REPORT_PATH = SCRIPTS_DIR / "SURGE_REGIME_INDUSTRY_LINKAGE_RESEARCH_2026-06-16.md"
STOCK_BASIC_CACHE = OUTPUT_DIR / "stock_basic_industry.parquet"
INDUSTRY_FEATURE_CACHE = OUTPUT_DIR / "industry_features.parquet"

CAND_DIR = SCRIPTS_DIR / "_output" / "surge_candidates"
MARKET_STATE_PATH = SCRIPTS_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
EXCESS_CACHE_PATH = SCRIPTS_DIR / "_output" / "surge_selection_audit" / "excess_cache.parquet"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

FIRST_TEST_YEAR = 2024
SLOTS = 10
MIN_OOS_TRADES = 60
MEAN_GAIN_GATE = 1.5

SURGE_GATE_VOL_RATIO = 1.2
SURGE_GATE_MA_SPREAD = 3.0
SURGE_GATE_RET20 = 8.0
MIN_AMOUNT_E = 1.0
STOP_MIN_PCT = 8.0
STOP_MAX_PCT = 20.0
GAP_LIMIT_MARGIN = 0.3
BUY_COST = 0.0015
SELL_COST = 0.0025

REGIME_QUALITY = {8: 20, 6: 18, 7: 16, 5: 12}


def round_float(value: float | int | None, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def t_stat(values: pd.Series) -> float | None:
    v = values.dropna()
    if len(v) < 2:
        return None
    std = v.std()
    if std <= 0 or pd.isna(std):
        return None
    return float(v.mean() / (std / np.sqrt(len(v))))


def limit_pct_for(symbol: str) -> float:
    return 19.8 if str(symbol).startswith(("300", "301", "302")) else 9.8


def priority_score(score: float, sl_pct: float | None, regime: int) -> float:
    p = min(float(score or 0), 100.0) * 0.35
    if sl_pct is None or pd.isna(sl_pct) or sl_pct <= 0:
        p -= 30
    elif 8.0 <= sl_pct <= 20.0:
        p += 25
    elif 5.0 <= sl_pct < 8.0 or 20.0 < sl_pct <= 30.0:
        p += 15
    else:
        p += 5
    p += 20  # freshness=0 in delayed-entry mirror
    p += REGIME_QUALITY.get(int(regime), 10)
    return round(max(p, 0.0), 1)


def _is_st_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "退" in text


def load_st_intervals() -> dict[str, list[tuple[str, str, bool]]]:
    if not NAMECHANGE_PATH.exists():
        return {}
    nc = pd.read_parquet(NAMECHANGE_PATH)
    nc["end_date"] = nc["end_date"].fillna("99999999")
    out: dict[str, list[tuple[str, str, bool]]] = {}
    for code, group in nc.groupby("ts_code"):
        group = group.sort_values("start_date")
        out[str(code)] = list(
            zip(group["start_date"], group["end_date"], [_is_st_name(x) for x in group["name"]], strict=False)
        )
    return out


def is_st_on(intervals: dict[str, list[tuple[str, str, bool]]], symbol: str, date: pd.Timestamp) -> bool:
    rows = intervals.get(str(symbol))
    if not rows:
        return False
    d = pd.Timestamp(date).strftime("%Y%m%d")
    best = None
    for start, _end, is_st in rows:
        if start <= d:
            best = is_st
    return bool(best) if best is not None else False


def load_stock_basic() -> pd.DataFrame:
    if STOCK_BASIC_CACHE.exists():
        return pd.read_parquet(STOCK_BASIC_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPTS_DIR))
    from sync_a_stock_daily import get_all_stocks

    basic = get_all_stocks("L,D")
    basic = basic[["ts_code", "name", "industry", "list_status", "list_date"]].copy()
    basic["industry"] = basic["industry"].replace("", "未知").fillna("未知")
    basic.to_parquet(STOCK_BASIC_CACHE, index=False)
    return basic


def load_industry_features(force: bool = False) -> pd.DataFrame:
    if INDUSTRY_FEATURE_CACHE.exists() and not force:
        print(f"[industry] cache hit: {INDUSTRY_FEATURE_CACHE}")
        return pd.read_parquet(INDUSTRY_FEATURE_CACHE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    basic = load_stock_basic()[["ts_code", "industry"]].rename(columns={"ts_code": "symbol"})
    panel = pd.read_parquet(CAND_DIR / "panel.parquet")
    panel = panel.merge(basic, on="symbol", how="left")
    panel["industry"] = panel["industry"].fillna("未知")
    panel = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)

    close = panel.groupby("symbol")["close"]
    panel["ret1"] = close.pct_change()
    panel["ret20"] = close.pct_change(20)
    panel["high20"] = close.transform(lambda s: s.rolling(20, min_periods=20).max())
    panel["is_high20"] = (panel["close"] >= panel["high20"]).astype(float)
    panel["limit_pct"] = panel["symbol"].map(limit_pct_for)
    panel["limit_up"] = (panel["ret1"] * 100 >= panel["limit_pct"] - 0.3).astype(float)

    features = (
        panel.groupby(["dt", "industry"])
        .agg(
            ind_n=("symbol", "nunique"),
            ind_ret20_mean=("ret20", "mean"),
            ind_ret20_median=("ret20", "median"),
            ind_high20_ratio=("is_high20", "mean"),
            ind_limit_up_ratio=("limit_up", "mean"),
            ind_amount_mean=("amount_e", "mean"),
        )
        .reset_index()
    )

    # Same-day industry delay5 candidate density. These are known at decision
    # close T and are used only for next-open decisions.
    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    delay5 = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5)].copy()
    delay5 = delay5.merge(basic, on="symbol", how="left")
    delay5["industry"] = delay5["industry"].fillna("未知")
    struct_counts = delay5.groupby(["dec_dt", "industry"]).size().rename("ind_struct_candidates").reset_index()
    features = features.merge(struct_counts, left_on=["dt", "industry"], right_on=["dec_dt", "industry"], how="left")
    features = features.drop(columns=["dec_dt"])
    features["ind_struct_candidates"] = features["ind_struct_candidates"].fillna(0)
    features["ind_struct_density"] = features["ind_struct_candidates"] / features["ind_n"].replace(0, np.nan)

    # Feature ranks are cross-industry ranks on each date.
    rank_cols = ["ind_ret20_mean", "ind_ret20_median", "ind_high20_ratio", "ind_limit_up_ratio", "ind_struct_density"]
    for col in rank_cols:
        features[f"{col}_rank"] = features.groupby("dt")[col].rank(pct=True)
    features["ind_composite_rank"] = features[
        ["ind_ret20_mean_rank", "ind_high20_ratio_rank", "ind_limit_up_ratio_rank", "ind_struct_density_rank"]
    ].mean(axis=1)

    float_cols = [c for c in features.columns if c.startswith("ind_") and c not in {"industry"}]
    for col in float_cols:
        if features[col].dtype == "float64":
            features[col] = features[col].astype("float32")
    features.to_parquet(INDUSTRY_FEATURE_CACHE, index=False)
    print(f"[industry] wrote {len(features)} rows -> {INDUSTRY_FEATURE_CACHE}")
    return features


def load_universe() -> pd.DataFrame:
    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    df = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5)].copy()
    market = pd.read_parquet(MARKET_STATE_PATH, columns=["dt", "high20_ratio", "ew_index_above_ma20"])
    df = df.merge(market, left_on="dec_dt", right_on="dt", how="left").drop(columns=["dt"])

    basic = load_stock_basic()[["ts_code", "name", "industry"]].rename(columns={"ts_code": "symbol"})
    df = df.merge(basic, on="symbol", how="left")
    df["industry"] = df["industry"].fillna("未知")

    features = load_industry_features()
    df = df.merge(features, left_on=["dec_dt", "industry"], right_on=["dt", "industry"], how="left").drop(columns=["dt"])

    st_intervals = load_st_intervals()
    if st_intervals:
        df["is_st"] = df.apply(lambda r: is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
    else:
        df["is_st"] = False
    df["gap_ok"] = (df["gap_pct"] < df["limit_pct"] - GAP_LIMIT_MARGIN).fillna(False)
    df["priority"] = [priority_score(s, sl, rg) for s, sl, rg in zip(df["score"], df["sl_pct"], df["dec_regime"], strict=False)]
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - SELL_COST) / (1 + BUY_COST) - 1) * 100

    keys = ["symbol", "sig_dt", "dec_dt", "entry_dt", "exit_dt"]
    if not EXCESS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing excess cache: {EXCESS_CACHE_PATH}")
    excess = pd.read_parquet(EXCESS_CACHE_PATH)
    df = df.merge(excess, on=keys, how="left")
    return df.reset_index(drop=True)


def base_mask(df: pd.DataFrame) -> pd.Series:
    mask = (
        (~df["is_st"])
        & df["gap_ok"]
        & (df["sig_vol_ratio"] >= SURGE_GATE_VOL_RATIO)
        & (df["sig_ma_spread_pct"] >= SURGE_GATE_MA_SPREAD)
        & (df["sig_ret20"] >= SURGE_GATE_RET20)
        & (df["sig_above_zg"] == 1)
        & (df["amount_e"] >= MIN_AMOUNT_E)
        & df["sl_pct"].between(STOP_MIN_PCT, STOP_MAX_PCT)
        & (df["high20_ratio"] > 0.12)
        & (df["ew_index_above_ma20"] > 0)
    )
    return mask.fillna(False)


def simulate_slots(df: pd.DataFrame, slots: int = SLOTS) -> pd.DataFrame:
    ordered = df.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)
    taken = []
    open_until: dict[str, pd.Timestamp] = {}
    for entry_dt, group in ordered.groupby("entry_dt", sort=True):
        open_until = {s: x for s, x in open_until.items() if x >= entry_dt}
        free = slots - len(open_until)
        if free <= 0:
            continue
        for _, row in group.iterrows():
            if free <= 0:
                break
            if row["symbol"] in open_until:
                continue
            open_until[row["symbol"]] = row["exit_dt"]
            taken.append(row)
            free -= 1
    return pd.DataFrame(taken).reset_index(drop=True)


def stat_values(values: pd.Series) -> dict[str, Any]:
    v = values.dropna()
    if len(v) < 30:
        return {"n": int(len(v))}
    return {
        "n": int(len(v)),
        "mean": round_float(v.mean()),
        "median": round_float(v.median()),
        "t": round_float(t_stat(v)),
        "pos_pct": round_float((v > 0).mean() * 100, 1),
    }


def trade_stats(trades: pd.DataFrame, seg: str) -> dict[str, Any]:
    scoped = trades[trades["seg"] == seg]
    out = stat_values(scoped["excess_pct"])
    if len(scoped):
        out["net_mean"] = round_float(scoped["ret_net_pct"].mean())
        out["net_median"] = round_float(scoped["ret_net_pct"].median())
        out["industries"] = int(scoped["industry"].nunique())
    return out


def candidate_stats(df: pd.DataFrame, mask: pd.Series, seg: str) -> dict[str, Any]:
    scoped = df[mask & (df["seg"] == seg)]
    out = stat_values(scoped["excess_pct"])
    if len(scoped):
        out["industries"] = int(scoped["industry"].nunique())
    return out


def gate_specs() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        ("baseline", "current base strategy, no industry gate", lambda df: pd.Series(True, index=df.index)),
        ("ind_ret20_gt0", "industry mean ret20 > 0", lambda df: df["ind_ret20_mean"] > 0),
        ("ind_ret20_rank_top50", "industry mean ret20 cross-industry rank >= 50%", lambda df: df["ind_ret20_mean_rank"] >= 0.50),
        ("ind_ret20_rank_top70", "industry mean ret20 cross-industry rank >= 70%", lambda df: df["ind_ret20_mean_rank"] >= 0.70),
        ("ind_high20_gt012", "industry high20 ratio > 12%", lambda df: df["ind_high20_ratio"] > 0.12),
        ("ind_high20_gt_market", "industry high20 ratio > market high20 ratio", lambda df: df["ind_high20_ratio"] > df["high20_ratio"]),
        ("ind_limit_up_rank_top70", "industry limit-up ratio rank >= 70%", lambda df: df["ind_limit_up_ratio_rank"] >= 0.70),
        ("ind_struct_density_top70", "industry delay5 structural density rank >= 70%", lambda df: df["ind_struct_density_rank"] >= 0.70),
        ("ind_composite_top50", "industry composite rank >= 50%", lambda df: df["ind_composite_rank"] >= 0.50),
        ("ind_composite_top70", "industry composite rank >= 70%", lambda df: df["ind_composite_rank"] >= 0.70),
    ]


def evaluate_gate(df: pd.DataFrame, gate_name: str, rule: str, gate_mask: pd.Series, base: pd.Series, baseline: dict | None) -> dict:
    mask = base & gate_mask.fillna(False)
    candidates = df[mask].copy()
    trades = simulate_slots(candidates)
    out = {
        "gate": gate_name,
        "rule": rule,
        "candidates": int(len(candidates)),
        "oos_candidates": int((candidates["seg"] == "test").sum()),
        "trade_count": int(len(trades)),
        "candidate_is": candidate_stats(df, mask, "train"),
        "candidate_oos": candidate_stats(df, mask, "test"),
        "trade_is": trade_stats(trades, "train"),
        "trade_oos": trade_stats(trades, "test"),
    }
    if baseline is None or gate_name == "baseline":
        out["verdict"] = {"passed": False, "failed": ["baseline"]}
        return out
    oos = out["trade_oos"]
    is_ = out["trade_is"]
    b_oos = baseline["trade_oos"]
    b_is = baseline["trade_is"]
    failed = []
    if (oos.get("n") or 0) < MIN_OOS_TRADES:
        failed.append("oos_n")
    if (oos.get("mean") or -999) < (b_oos.get("mean") or 0) + MEAN_GAIN_GATE:
        failed.append("mean_gain")
    if (oos.get("median") or -999) < (b_oos.get("median") or -999):
        failed.append("median")
    if (oos.get("t") or 0) < 2:
        failed.append("t")
    if (is_.get("mean") or -999) < (b_is.get("mean") or -999):
        failed.append("is_mean")
    out["verdict"] = {"passed": not failed, "failed": failed}
    return out


def bucket_report(df: pd.DataFrame, mask: pd.Series, feature: str) -> list[dict[str, Any]]:
    scoped = df[mask & (df["seg"] == "test") & df[feature].notna()].copy()
    if len(scoped) < 100:
        return []
    scoped["bucket"] = pd.qcut(scoped[feature], 5, duplicates="drop")
    rows = []
    for bucket, group in scoped.groupby("bucket", observed=True):
        stats = stat_values(group["excess_pct"])
        rows.append(
            {
                "bucket": str(bucket),
                "n": stats.get("n"),
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "t": stats.get("t"),
                "t1_rate": round_float(group.get("t1_px30", pd.Series(dtype=float)).mean() * 100, 1)
                if "t1_px30" in group
                else None,
            }
        )
    return rows


def maybe_add_t1_label(df: pd.DataFrame) -> pd.DataFrame:
    bars_path = SCRIPTS_DIR / "_output" / "surge_factors" / "bars.parquet"
    if not bars_path.exists():
        return df
    bars = pd.read_parquet(bars_path, columns=["symbol", "dt", "t1_px30", "fwd40_max"])
    return df.merge(bars.rename(columns={"dt": "dec_dt"}), on=["symbol", "dec_dt"], how="left")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Surge Regime industry-linkage research (2026-06-16)",
        "",
        "First-stage test of industry/sector linkage for the delay5 surge-regime strategy.",
        "This uses stock_basic.industry plus the existing daily panel; it does not change live behavior.",
        "",
        "## Pass Criterion",
        "",
        f"- OOS 10-slot trades >= {MIN_OOS_TRADES}",
        f"- OOS trade excess mean >= baseline + {MEAN_GAIN_GATE} pp",
        "- OOS trade excess median >= baseline median",
        "- OOS t >= 2",
        "- IS trade excess mean >= baseline IS mean",
        "",
        "## Universe",
        "",
        "```json",
        json.dumps(summary["meta"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Gate Results",
    ]
    rows = []
    for row in summary["gates"]:
        oos = row["trade_oos"]
        is_ = row["trade_is"]
        rows.append(
            {
                "gate": row["gate"],
                "passed": row["verdict"]["passed"],
                "cand": row["candidates"],
                "oos_n": oos.get("n"),
                "oos_mean": oos.get("mean"),
                "oos_med": oos.get("median"),
                "oos_t": oos.get("t"),
                "is_mean": is_.get("mean"),
                "failed": ",".join(row["verdict"]["failed"]),
            }
        )
    lines.extend(markdown_table(rows, ["gate", "passed", "cand", "oos_n", "oos_mean", "oos_med", "oos_t", "is_mean", "failed"]))

    passing = [r["gate"] for r in summary["gates"] if r["verdict"]["passed"]]
    lines.extend(["", "## Conclusion", ""])
    if passing:
        lines.append(f"Industry gate(s) passed first-stage screen: {', '.join(passing)}.")
    else:
        lines.append(
            "No industry-linkage gate passed the pre-declared first-stage screen. "
            "Industry information shows descriptive structure but is not yet a deployable filter for delay5."
        )

    lines.extend(["", "## OOS Candidate Buckets"])
    for feature, rows_ in summary["buckets"].items():
        lines.extend(["", f"### {feature}"])
        lines.extend(markdown_table(rows_, ["bucket", "n", "mean", "median", "t", "t1_rate"]))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] building universe with industry features")
    df = maybe_add_t1_label(load_universe())
    base = base_mask(df)
    print(f"[base] candidates={int(base.sum())} universe={len(df)}")

    gate_rows = []
    baseline_row = None
    for name, rule, fn in gate_specs():
        row = evaluate_gate(df, name, rule, fn(df), base, baseline_row)
        if name == "baseline":
            baseline_row = row
        gate_rows.append(row)
        print(
            f"[{name}] cand={row['candidates']} oos_n={row['trade_oos'].get('n')} "
            f"mean={row['trade_oos'].get('mean')} med={row['trade_oos'].get('median')} "
            f"pass={row['verdict']['passed']}"
        )

    bucket_features = [
        "ind_ret20_mean_rank",
        "ind_high20_ratio_rank",
        "ind_limit_up_ratio_rank",
        "ind_struct_density_rank",
        "ind_composite_rank",
    ]
    buckets = {feature: bucket_report(df, base, feature) for feature in bucket_features}

    meta = {
        "universe_rows": int(len(df)),
        "base_candidates": int(base.sum()),
        "base_oos_candidates": int((base & (df["seg"] == "test")).sum()),
        "industries_total": int(df["industry"].nunique()),
        "industries_base": int(df.loc[base, "industry"].nunique()),
        "first_test_year": FIRST_TEST_YEAR,
        "slots": SLOTS,
    }
    summary = {"meta": meta, "gates": gate_rows, "buckets": buckets}
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    write_report(summary)
    print(f"[done] {time.time() - t0:.0f}s -> {OUTPUT_DIR}")
    print(f"[report] {REPORT_PATH}")


if __name__ == "__main__":
    main()
