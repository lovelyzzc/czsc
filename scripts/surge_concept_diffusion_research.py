"""Concept/theme diffusion research for the surge-regime delay5 strategy.

This is the finer-grained follow-up after stock_basic.industry linkage failed.
It uses same-day 同花顺概念板块 (ths_index type=N) membership and the existing
daily panel to test whether concept/theme diffusion helps identify delay5 tail
winners.

Pre-declared protocol:
- Concept source: tinyshare ths_index(type="N") + ths_member. Members are
  cached locally. Broad index/style baskets are excluded by name and member
  count to keep the universe closer to thematic concepts.
- Base universe: existing anticipate + delay5 candidate dump and current BASE
  gates (signal gates, hard filters, market gate, ST/gap filters).
- Causal concept features at decision close T:
  concept ret20 mean, high20 ratio, limit-up ratio, and same-day delay5
  structural candidate density. For a candidate with multiple concepts, use
  max/mean ranks and hot-concept counts across its concepts.
- Candidate-layer screen: OOS excess by concept diffusion buckets.
- Portfolio-layer screen: 10-slot mirror using existing priority order and
  candidate-level excess cache.

Pass criterion for moving a concept gate to fuller mirror/walk-forward:
- OOS 10-slot trade count >= 60;
- OOS trade excess mean >= baseline mean + 1.5 pp;
- OOS trade excess median >= baseline median;
- OOS t >= 2;
- IS trade excess mean >= baseline IS mean.

This is a first-stage research gate. It does not change live strategy behavior.

    .venv/bin/python scripts/surge_concept_diffusion_research.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPTS_DIR / "_output" / "surge_concept_diffusion"
REPORT_PATH = SCRIPTS_DIR / "SURGE_REGIME_CONCEPT_DIFFUSION_RESEARCH_2026-06-16.md"

CONCEPT_INDEX_CACHE = OUTPUT_DIR / "ths_concept_index.parquet"
CONCEPT_MEMBERS_CACHE = OUTPUT_DIR / "ths_concept_members.parquet"
CONCEPT_FEATURE_CACHE = OUTPUT_DIR / "concept_features.parquet"
CONCEPT_CANDIDATE_CACHE = OUTPUT_DIR / "candidate_concept_features.parquet"

TINYSHARE_TOKEN = "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b"

FIRST_TEST_YEAR = 2024
SLOTS = 10
MIN_OOS_TRADES = 60
MEAN_GAIN_GATE = 1.5
MIN_CONCEPT_MEMBERS = 8
MAX_CONCEPT_MEMBERS = 180
THS_MEMBER_INTERVAL = 0.75

BROAD_NAME_RE = re.compile(
    r"样本股|成份股|成分股|沪深|中证|上证|深证|创业板指|科创|指数|"
    r"融资融券|转融券|沪股通|深股通|港股通|MSCI|富时|标普|证金|基金重仓|机构重仓|"
    r"同花顺漂亮|同花顺出海|金仓"
)


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


def load_concept_index(force: bool = False) -> pd.DataFrame:
    if CONCEPT_INDEX_CACHE.exists() and not force:
        return pd.read_parquet(CONCEPT_INDEX_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import tinyshare as ts

    ts.set_token(TINYSHARE_TOKEN)
    pro = ts.pro_api()
    concepts = pro.ths_index(type="N")
    if concepts is None or concepts.empty:
        raise RuntimeError("ths_index(type=N) returned empty result")
    concepts = concepts.dropna(subset=["ts_code"]).copy()
    concepts["name"] = concepts["name"].astype(str)
    concepts.to_parquet(CONCEPT_INDEX_CACHE, index=False)
    return concepts


def load_concept_members(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    concepts = load_concept_index(force=force)
    existing = None
    if CONCEPT_MEMBERS_CACHE.exists() and not force:
        existing = pd.read_parquet(CONCEPT_MEMBERS_CACHE)
        missing = sorted(set(concepts["ts_code"].astype(str)) - set(existing["ts_code"].astype(str)))
        if not missing:
            return concepts, existing
        print(f"[members] cache has {existing['ts_code'].nunique()} concepts; retrying {len(missing)} missing concepts")
        concepts_to_fetch = concepts[concepts["ts_code"].isin(missing)].copy()
    else:
        concepts_to_fetch = concepts.copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import tinyshare as ts

    ts.set_token(TINYSHARE_TOKEN)
    pro = ts.pro_api()

    rows = []
    t0 = time.time()
    for i, rec in enumerate(concepts_to_fetch.to_dict("records"), 1):
        code = rec["ts_code"]
        mem = None
        for attempt in range(3):
            try:
                mem = pro.ths_member(ts_code=code)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [member warn] {code} attempt={attempt + 1}: {exc}")
                time.sleep(5 + attempt * 5)
        if mem is not None and not mem.empty and "con_code" in mem.columns:
            tmp = mem[["ts_code", "con_code", "con_name"]].copy()
            tmp["concept_name"] = rec["name"]
            rows.append(tmp)
        if i % 25 == 0 or i == len(concepts_to_fetch):
            print(
                f"  [members] {i}/{len(concepts_to_fetch)} concepts, "
                f"new_rows={sum(len(x) for x in rows)} elapsed={time.time() - t0:.0f}s"
            )
        time.sleep(THS_MEMBER_INTERVAL)

    if not rows and existing is None:
        raise RuntimeError("No concept members fetched")
    new_members = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ts_code", "con_code", "con_name", "concept_name"])
    if existing is not None:
        members = pd.concat([existing, new_members], ignore_index=True)
    else:
        members = new_members
    members = members.drop_duplicates(["ts_code", "con_code"])
    members.to_parquet(CONCEPT_MEMBERS_CACHE, index=False)
    return concepts, members


def fine_concept_members(force: bool = False) -> pd.DataFrame:
    concepts, members = load_concept_members(force=force)
    counts = members.groupby("ts_code")["con_code"].nunique().rename("member_count").reset_index()
    meta = concepts[["ts_code", "name"]].merge(counts, on="ts_code", how="left")
    meta["member_count"] = meta["member_count"].fillna(0)
    fine = meta[
        (meta["member_count"].between(MIN_CONCEPT_MEMBERS, MAX_CONCEPT_MEMBERS))
        & ~meta["name"].astype(str).str.contains(BROAD_NAME_RE, regex=True)
    ].copy()
    out = members[members["ts_code"].isin(fine["ts_code"])].copy()
    out = out.rename(columns={"ts_code": "concept_code", "con_code": "symbol"})
    out = out.merge(fine.rename(columns={"ts_code": "concept_code", "name": "concept_name"})[["concept_code", "concept_name", "member_count"]], on=["concept_code", "concept_name"], how="left")
    return out[["concept_code", "concept_name", "symbol", "con_name", "member_count"]].drop_duplicates()


def load_base_universe() -> pd.DataFrame:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import surge_industry_linkage_research as ind

    return ind.maybe_add_t1_label(ind.load_universe())


def base_mask(df: pd.DataFrame) -> pd.Series:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import surge_industry_linkage_research as ind

    return ind.base_mask(df)


def simulate_slots(df: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import surge_industry_linkage_research as ind

    return ind.simulate_slots(df, slots=SLOTS)


def load_concept_features(force: bool = False) -> pd.DataFrame:
    if CONCEPT_FEATURE_CACHE.exists() and not force:
        print(f"[concept] cache hit: {CONCEPT_FEATURE_CACHE}")
        return pd.read_parquet(CONCEPT_FEATURE_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    members = fine_concept_members(force=force)
    panel = pd.read_parquet(SCRIPTS_DIR / "_output" / "surge_candidates" / "panel.parquet")
    panel = panel.sort_values(["symbol", "dt"]).reset_index(drop=True)

    close = panel.groupby("symbol")["close"]
    panel["ret1"] = close.pct_change()
    panel["ret20"] = close.pct_change(20)
    panel["high20"] = close.transform(lambda s: s.rolling(20, min_periods=20).max())
    panel["is_high20"] = (panel["close"] >= panel["high20"]).astype(float)
    panel["limit_pct"] = np.where(panel["symbol"].astype(str).str.startswith(("300", "301", "302")), 19.8, 9.8)
    panel["limit_up"] = (panel["ret1"] * 100 >= panel["limit_pct"] - 0.3).astype(float)

    long = panel[["symbol", "dt", "ret20", "is_high20", "limit_up", "amount_e"]].merge(
        members[["concept_code", "concept_name", "symbol", "member_count"]],
        on="symbol",
        how="inner",
    )
    features = (
        long.groupby(["dt", "concept_code", "concept_name"])
        .agg(
            cpt_n=("symbol", "nunique"),
            cpt_ret20_mean=("ret20", "mean"),
            cpt_ret20_median=("ret20", "median"),
            cpt_high20_ratio=("is_high20", "mean"),
            cpt_limit_up_ratio=("limit_up", "mean"),
            cpt_amount_mean=("amount_e", "mean"),
            cpt_member_count=("member_count", "first"),
        )
        .reset_index()
    )

    cand = pd.read_parquet(SCRIPTS_DIR / "_output" / "surge_candidates" / "candidates.parquet")
    delay5 = cand[(cand["mode"] == "anticipate") & (cand["delay"] == 5)][["symbol", "dec_dt"]].copy()
    cands_long = delay5.merge(members[["concept_code", "concept_name", "symbol"]], on="symbol", how="inner")
    struct_counts = (
        cands_long.groupby(["dec_dt", "concept_code", "concept_name"]).size().rename("cpt_struct_candidates").reset_index()
    )
    features = features.merge(
        struct_counts,
        left_on=["dt", "concept_code", "concept_name"],
        right_on=["dec_dt", "concept_code", "concept_name"],
        how="left",
    ).drop(columns=["dec_dt"])
    features["cpt_struct_candidates"] = features["cpt_struct_candidates"].fillna(0)
    features["cpt_struct_density"] = features["cpt_struct_candidates"] / features["cpt_member_count"].replace(0, np.nan)

    rank_cols = ["cpt_ret20_mean", "cpt_high20_ratio", "cpt_limit_up_ratio", "cpt_struct_density"]
    for col in rank_cols:
        features[f"{col}_rank"] = features.groupby("dt")[col].rank(pct=True)
    features["cpt_composite_rank"] = features[[f"{c}_rank" for c in rank_cols]].mean(axis=1)

    for col in [c for c in features.columns if c.startswith("cpt_") and features[c].dtype == "float64"]:
        features[col] = features[col].astype("float32")
    features.to_parquet(CONCEPT_FEATURE_CACHE, index=False)
    print(f"[concept] wrote {len(features)} feature rows from {members['concept_code'].nunique()} fine concepts")
    return features


def candidate_concept_features(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if CONCEPT_CANDIDATE_CACHE.exists() and not force:
        print(f"[candidate concepts] cache hit: {CONCEPT_CANDIDATE_CACHE}")
        return pd.read_parquet(CONCEPT_CANDIDATE_CACHE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    members = fine_concept_members(force=False)
    features = load_concept_features(force=force)

    keys = df[["symbol", "dec_dt"]].drop_duplicates().reset_index(drop=True)
    keys["row_key"] = np.arange(len(keys))
    long = keys.merge(members[["concept_code", "concept_name", "symbol"]], on="symbol", how="left")
    long = long.merge(
        features,
        left_on=["dec_dt", "concept_code", "concept_name"],
        right_on=["dt", "concept_code", "concept_name"],
        how="left",
    )
    long = long.drop(columns=["dt"])
    rank_cols = [
        "cpt_ret20_mean_rank",
        "cpt_high20_ratio_rank",
        "cpt_limit_up_ratio_rank",
        "cpt_struct_density_rank",
        "cpt_composite_rank",
    ]
    agg_spec: dict[str, tuple[str, str]] = {}
    for col in rank_cols:
        agg_spec[f"{col}_max"] = (col, "max")
        agg_spec[f"{col}_mean"] = (col, "mean")
    extra = {
        "concept_count": ("concept_code", "nunique"),
        "hot70_count": ("cpt_composite_rank", lambda x: int((x >= 0.70).sum())),
        "hot85_count": ("cpt_composite_rank", lambda x: int((x >= 0.85).sum())),
        "density70_count": ("cpt_struct_density_rank", lambda x: int((x >= 0.70).sum())),
        "best_concept_rank": ("cpt_composite_rank", "max"),
    }
    agg = long.groupby("row_key").agg(**agg_spec, **extra).reset_index()

    best = long.sort_values(["row_key", "cpt_composite_rank"], ascending=[True, False]).drop_duplicates("row_key")
    best = best[["row_key", "concept_code", "concept_name", "cpt_composite_rank"]].rename(
        columns={
            "concept_code": "best_concept_code",
            "concept_name": "best_concept_name",
            "cpt_composite_rank": "best_concept_composite_rank",
        }
    )
    out = keys.merge(agg, on="row_key", how="left").merge(best, on="row_key", how="left")
    out["hot70_share"] = out["hot70_count"] / out["concept_count"].replace(0, np.nan)
    out["hot85_share"] = out["hot85_count"] / out["concept_count"].replace(0, np.nan)
    out.to_parquet(CONCEPT_CANDIDATE_CACHE, index=False)
    return out


def attach_concept_features(df: pd.DataFrame) -> pd.DataFrame:
    cfeat = candidate_concept_features(df)
    return df.merge(cfeat.drop(columns=["row_key"]), on=["symbol", "dec_dt"], how="left")


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
        out["best_concepts"] = int(scoped["best_concept_name"].nunique())
    return out


def candidate_stats(df: pd.DataFrame, mask: pd.Series, seg: str) -> dict[str, Any]:
    scoped = df[mask & (df["seg"] == seg)]
    out = stat_values(scoped["excess_pct"])
    if len(scoped):
        out["best_concepts"] = int(scoped["best_concept_name"].nunique())
    return out


def gate_specs() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        ("baseline", "current base strategy, no concept gate", lambda df: pd.Series(True, index=df.index)),
        ("cpt_composite_max_top70", "max concept composite rank >= 70%", lambda df: df["cpt_composite_rank_max"] >= 0.70),
        ("cpt_composite_max_top85", "max concept composite rank >= 85%", lambda df: df["cpt_composite_rank_max"] >= 0.85),
        ("cpt_composite_mean_top60", "mean concept composite rank >= 60%", lambda df: df["cpt_composite_rank_mean"] >= 0.60),
        ("cpt_hot70_count_ge2", "at least two concepts with composite rank >=70%", lambda df: df["hot70_count"] >= 2),
        ("cpt_hot70_share_ge50", "at least half candidate concepts are hot70", lambda df: df["hot70_share"] >= 0.50),
        ("cpt_density_max_top80", "max concept structural-density rank >=80%", lambda df: df["cpt_struct_density_rank_max"] >= 0.80),
        ("cpt_limit_up_max_top80", "max concept limit-up rank >=80%", lambda df: df["cpt_limit_up_ratio_rank_max"] >= 0.80),
        ("cpt_ret20_max_top80", "max concept ret20 rank >=80%", lambda df: df["cpt_ret20_mean_rank_max"] >= 0.80),
        ("cpt_high20_max_top80", "max concept high20 rank >=80%", lambda df: df["cpt_high20_ratio_rank_max"] >= 0.80),
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
                "t1_rate": round_float(group["t1_px30"].mean() * 100, 1) if "t1_px30" in group else None,
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Surge Regime concept-diffusion research (2026-06-16)",
        "",
        "First-stage test of fine-grained 同花顺概念 diffusion for the delay5 surge-regime strategy.",
        "It uses ths_index(type=N) membership plus the existing daily panel; it does not change live behavior.",
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
        lines.append(f"Concept gate(s) passed first-stage screen: {', '.join(passing)}.")
    else:
        lines.append(
            "No concept-diffusion gate passed the pre-declared first-stage screen. "
            "Fine-grained concepts show descriptive heat but are not yet a deployable delay5 filter."
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
    print("[load] delay5 universe")
    df = load_base_universe()
    base = base_mask(df)
    print(f"[base] candidates={int(base.sum())} universe={len(df)}")
    df = attach_concept_features(df)
    fine_members = fine_concept_members()
    print(
        f"[concepts] fine_concepts={fine_members['concept_code'].nunique()} "
        f"member_links={len(fine_members)} base_with_concepts={int(df.loc[base, 'concept_count'].notna().sum())}"
    )

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
        "cpt_composite_rank_max",
        "cpt_composite_rank_mean",
        "hot70_count",
        "hot70_share",
        "cpt_struct_density_rank_max",
        "cpt_limit_up_ratio_rank_max",
        "cpt_ret20_mean_rank_max",
        "cpt_high20_ratio_rank_max",
    ]
    buckets = {feature: bucket_report(df, base, feature) for feature in bucket_features}
    meta = {
        "universe_rows": int(len(df)),
        "base_candidates": int(base.sum()),
        "base_oos_candidates": int((base & (df["seg"] == "test")).sum()),
        "all_concepts": int(load_concept_index()["ts_code"].nunique()),
        "fine_concepts": int(fine_members["concept_code"].nunique()),
        "fine_member_links": int(len(fine_members)),
        "base_candidates_with_concepts": int(df.loc[base, "concept_count"].notna().sum()),
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
