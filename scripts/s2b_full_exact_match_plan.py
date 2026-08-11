"""冻结全行业点时精确市值匹配的请求身份与跨设备取数计划。

本脚本不执行完整匹配，也不下载 Tushare 数据。它从当前候选、年度行业快照和价格面板重建
329 笔容量模拟 cohort，并分别冻结三种股票×决策日请求宇宙。若本机存在 ignored 的 BaoStock
缓存和两份 Stage3 ``daily_basic`` 对象，还会复算有限的跨供应商口径 sanity check。

    uv run --no-sync python scripts/s2b_full_exact_match_plan.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import _s2c_core as core
import pandas as pd
import s2b_industry_size_proxy_audit as proxy
import s2b_matched_control_audit as base
import s2b_top_winner_exact_mcap_audit as exact
from surge_market_state_filter import StableControlSampler

YEARS = (2024, 2025, 2026)
EXPECTED_MANIFEST_PATH = Path(__file__).resolve().parent / "s2b_full_exact_match_plan_2026-08-10.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s2b_full_exact_match_plan"
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
STAGE3_ROOT = Path(__file__).resolve().parent / "_output" / "xs_chan_exploration_stage3"
TUSHARE_OBJECTS = {
    "2026-06-26": "5121f47e3ebc2a810e75595de68a4d69234093bb6b6d5dda78c04c6bb47be106",
    "2026-07-03": "0f715365b4b4d36d3e5a8803ca4acfb084bc1fbd1b4fcf99536022c817888859",
}


def canonical_payload(records: list[Any]) -> bytes:
    """按清单约定编码排序 JSON 数组：UTF-8、无空白、无末尾换行。"""

    ordered = sorted(records)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def payload_identity(records: list[Any]) -> dict[str, Any]:
    """返回 canonical payload 的计数、字节数、SHA256 与首末哨兵。"""

    ordered = sorted(records)
    payload = canonical_payload(ordered)
    return {
        "count": int(len(ordered)),
        "bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first3": ordered[:3],
        "last3": ordered[-3:] if ordered else [],
    }


def _load_industry_snapshots() -> dict[int, pd.DataFrame]:
    """只读加载冻结行业快照；规划脚本本身不触发网络下载。"""

    frames: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        path = proxy._industry_path(year)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; first run s2b_industry_size_proxy_audit.py with BaoStock available"
            )
        frames[year] = pd.read_parquet(path)
    return frames


def _load_capacity_cohort() -> pd.DataFrame:
    context = core.load_context(verbose=False)
    d0 = context["d0"].copy()
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        d0[column] = pd.to_datetime(d0[column]).dt.normalize()
    candidates = d0.loc[core.gate_mask(d0, core.S2B_VR, core.S2B_SP)].copy()
    candidates = candidates[~base.censored_tail_mask(candidates)]
    capacity_keys = base._main_execution_keys(context)
    candidates = candidates[
        [
            (symbol, entry_dt) in capacity_keys
            for symbol, entry_dt in zip(candidates["symbol"], candidates["entry_dt"], strict=False)
        ]
    ]
    cohort = candidates[candidates["dec_dt"].dt.year.isin(YEARS)].copy()
    cohort = cohort.sort_values(["dec_dt", "symbol", "entry_dt"]).reset_index(drop=True)
    if cohort.duplicated(["symbol", "entry_dt"]).any():
        raise RuntimeError("capacity cohort is not unique by symbol and entry_dt")
    return cohort


def _add_pairs(target: set[tuple[str, str]], symbols: pd.Index, dec_dt: pd.Timestamp) -> None:
    date_text = pd.Timestamp(dec_dt).strftime("%Y-%m-%d")
    target.update((str(symbol), date_text) for symbol in symbols)


def build_request_identities() -> tuple[dict[str, Any], dict[str, set[tuple[str, str]]]]:
    """重建 cohort 与三种全行业请求宇宙。"""

    cohort = _load_capacity_cohort()
    sampler = StableControlSampler()
    industry_frames = _load_industry_snapshots()
    industry_maps = {
        year: frame.set_index("symbol")["industry"].reindex(sampler.close_w.columns)
        for year, frame in industry_frames.items()
    }
    pair_sets: dict[str, set[tuple[str, str]]] = {
        "legacy_future_valid": set(),
        "decision_date_only": set(),
        "annual_industry_full": set(),
    }
    for trade in cohort.itertuples():
        industry_map = industry_maps[int(trade.dec_dt.year)]
        treated_industry = industry_map.get(trade.symbol)
        if treated_industry is None or pd.isna(treated_industry) or not str(treated_industry):
            raise RuntimeError(f"missing frozen industry for treated trade {trade.symbol} {trade.dec_dt}")
        same_industry = industry_map.index[industry_map.eq(treated_industry)].difference(pd.Index([trade.symbol]))
        decision_visible = same_industry.intersection(
            sampler.close_w.columns[sampler.close_w.loc[trade.dec_dt].notna()]
        )
        future_valid = sampler.open_w.loc[trade.entry_dt].notna() & sampler.close_w.loc[trade.exit_dt].notna()
        legacy = same_industry.intersection(future_valid.index[future_valid])
        treated = pd.Index([trade.symbol])
        _add_pairs(pair_sets["annual_industry_full"], same_industry.union(treated), trade.dec_dt)
        _add_pairs(pair_sets["decision_date_only"], decision_visible.union(treated), trade.dec_dt)
        _add_pairs(pair_sets["legacy_future_valid"], legacy.union(treated), trade.dec_dt)

    entry_keys = sorted(
        [[row.symbol, pd.Timestamp(row.entry_dt).strftime("%Y-%m-%d")] for row in cohort.itertuples(index=False)]
    )
    trade_records = sorted(
        [
            [
                row.symbol,
                pd.Timestamp(row.dec_dt).strftime("%Y-%m-%d"),
                pd.Timestamp(row.entry_dt).strftime("%Y-%m-%d"),
                pd.Timestamp(row.exit_dt).strftime("%Y-%m-%d"),
            ]
            for row in cohort.itertuples(index=False)
        ]
    )
    decision_dates = sorted(cohort["dec_dt"].dt.strftime("%Y-%m-%d").unique().tolist())
    identity = {
        "cohort": {
            "year_counts": {
                str(int(year)): int(count)
                for year, count in cohort.groupby(cohort["dec_dt"].dt.year, sort=True).size().items()
            },
            "capacity_entry_keys": payload_identity(entry_keys),
            "full_trade_records": payload_identity(trade_records),
            "decision_dates": payload_identity(decision_dates),
        },
        "request_universes": {
            name: payload_identity([list(pair) for pair in pairs]) for name, pairs in pair_sets.items()
        },
        "industry_snapshots": {
            str(year): {
                "requested_anchor": proxy.INDUSTRY_ANCHORS[year],
                "effective_dates": sorted(industry_frames[year]["updateDate"].astype(str).unique().tolist()),
                "rows": int(len(industry_frames[year])),
                "canonical_content_sha256": proxy.canonical_industry_sha256(industry_frames[year]),
            }
            for year in YEARS
        },
    }
    return identity, pair_sets


def exact_cache_identity() -> dict[str, Any]:
    """记录 ignored BaoStock pair cache；缺失时不伪装成已随 Git 同步。"""

    if not exact.REFERENCE_PATH.exists():
        return {"available": False}
    frame = pd.read_parquet(exact.REFERENCE_PATH)
    return {
        "available": True,
        "sha256": proxy.sha256_file(exact.REFERENCE_PATH),
        "rows": int(len(frame)),
        "valid_exact_mcap": int(frame["exact_circ_mv"].notna().sum()),
        "symbols": int(frame["symbol"].nunique()),
        "decision_dates": int(pd.to_datetime(frame["dec_dt"]).nunique()),
    }


def _find_daily_basic_object(object_sha: str) -> Path | None:
    paths = sorted(STAGE3_ROOT.glob(f"LEDGER_*/objects/daily_basic/{object_sha}.json"))
    if not paths:
        return None
    for path in paths:
        if proxy.sha256_file(path) != object_sha:
            raise RuntimeError(f"daily_basic object content hash mismatch: {path}")
    return paths[0]


def cross_provider_sanity() -> dict[str, Any]:
    """在两个已知 2026 日期上复算 Tushare/BaoStock 点时市值差异。"""

    if not exact.REFERENCE_PATH.exists():
        return {"available": False, "reason": "BaoStock exact-mcap cache is absent"}
    records: list[dict[str, Any]] = []
    objects: dict[str, Any] = {}
    for date_text, object_sha in TUSHARE_OBJECTS.items():
        path = _find_daily_basic_object(object_sha)
        if path is None:
            return {"available": False, "reason": f"ignored daily_basic object {object_sha} is absent"}
        rows = json.loads(path.read_text(encoding="utf-8"))
        records.extend(rows)
        objects[date_text] = {"sha256": object_sha, "rows": int(len(rows))}
    tushare = pd.DataFrame(records)
    tushare["symbol"] = tushare["ts_code"].astype(str)
    tushare["dec_dt"] = pd.to_datetime(tushare["trade_date"], format="%Y%m%d")
    tushare["tushare_circ_mv_yuan"] = pd.to_numeric(tushare["circ_mv"], errors="coerce") * 10_000
    baostock = pd.read_parquet(exact.REFERENCE_PATH)
    baostock["dec_dt"] = pd.to_datetime(baostock["dec_dt"])
    merged = baostock.merge(
        tushare[["symbol", "dec_dt", "tushare_circ_mv_yuan"]],
        on=["symbol", "dec_dt"],
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["exact_circ_mv", "tushare_circ_mv_yuan"])
    ratio = merged["tushare_circ_mv_yuan"] / merged["exact_circ_mv"]
    absolute_pct = ratio.sub(1).abs().mul(100)
    if merged.empty:
        raise RuntimeError("known daily_basic objects have no overlap with BaoStock cache")
    max_row = merged.loc[absolute_pct.idxmax()]
    return {
        "available": True,
        "scope": "two 2026 dates only; supplier-unit sanity check, not full-period equivalence evidence",
        "objects": objects,
        "unit_conversion": "Tushare circ_mv (10k CNY) * 10,000 = CNY",
        "overlap_pairs": int(len(merged)),
        "ratio_median": round(float(ratio.median()), 9),
        "absolute_difference_pct_median": round(float(absolute_pct.median()), 6),
        "absolute_difference_pct_p95": round(float(absolute_pct.quantile(0.95)), 6),
        "absolute_difference_pct_max": round(float(absolute_pct.max()), 6),
        "pairs_above_5pct": int(absolute_pct.gt(5).sum()),
        "max_difference_pair": {
            "symbol": str(max_row["symbol"]),
            "dec_dt": str(pd.Timestamp(max_row["dec_dt"]).date()),
        },
    }


def build_manifest() -> dict[str, Any]:
    """构建无生成时间戳的确定性规划清单。"""

    identity, _ = build_request_identities()
    return {
        "schema": "s2b_full_exact_match_plan_manifest_v1",
        "canonicalization": (
            "top-level JSON array; positional records; dates YYYY-MM-DD; lexicographic sort; "
            "json.dumps ensure_ascii=False separators=(',', ':') allow_nan=False; UTF-8; no trailing newline"
        ),
        **identity,
        "local_ignored_artifacts": {
            "baostock_exact_mcap": exact_cache_identity(),
            "tushare_baostock_sanity": cross_provider_sanity(),
            "git_portable": False,
        },
        "fetch_plan": {
            "recommended_source": "Tushare daily_basic by decision date; use circ_mv without per-symbol source mixing",
            "git_only_required_calls": 219,
            "calls_if_two_named_ignored_objects_are_copied": 217,
            "approximate_rows_git_only": 1_200_000,
            "full_exact_matching_implemented": False,
            "preferred_request_universe": "decision_date_only",
            "legacy_future_valid_warning": (
                "60,028 pairs condition on entry/treated-exit price availability; use the 60,102 decision-date-only "
                "universe for the next design and handle later missing outcomes explicitly"
            ),
        },
    }


def _identity_checks(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "schema": actual.get("schema") == expected.get("schema"),
        "canonicalization": actual.get("canonicalization") == expected.get("canonicalization"),
        "cohort": actual["cohort"] == expected["cohort"],
        "request_universes": actual["request_universes"] == expected["request_universes"],
        "industry_snapshots": actual["industry_snapshots"] == expected["industry_snapshots"],
        "fetch_plan": actual["fetch_plan"] == expected["fetch_plan"],
    }
    expected_local = expected["local_ignored_artifacts"]
    actual_local = actual["local_ignored_artifacts"]
    checks["baostock_cache_if_available"] = (
        actual_local["baostock_exact_mcap"] == expected_local["baostock_exact_mcap"]
        if actual_local["baostock_exact_mcap"].get("available")
        else None
    )
    checks["cross_provider_sanity_if_available"] = (
        actual_local["tushare_baostock_sanity"] == expected_local["tushare_baostock_sanity"]
        if actual_local["tushare_baostock_sanity"].get("available")
        else None
    )
    return checks


def run_audit() -> dict[str, Any]:
    """重建规划身份并与 Git 中的冻结清单比对。"""

    if not EXPECTED_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"missing tracked expected manifest: {EXPECTED_MANIFEST_PATH}")
    expected = json.loads(EXPECTED_MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = build_manifest()
    checks = _identity_checks(actual, expected)
    required_names = (
        "schema",
        "canonicalization",
        "cohort",
        "request_universes",
        "industry_snapshots",
        "fetch_plan",
    )
    required = [checks[name] for name in required_names]
    return {
        "schema": "s2b_full_exact_match_plan_audit_v1",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "expected_manifest": {
            "path": str(EXPECTED_MANIFEST_PATH),
            "sha256": proxy.sha256_file(EXPECTED_MANIFEST_PATH),
        },
        "checks": checks,
        "actual": actual,
        "verdict": {
            "status": "PLAN_IDENTITY_REPRODUCED" if all(required) else "PLAN_INPUT_IDENTITY_DRIFT",
            "full_exact_matching_implemented": False,
            "live_authorized": False,
        },
    }


def main() -> None:
    started = time.time()
    result = run_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {OUTPUT_PATH} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
