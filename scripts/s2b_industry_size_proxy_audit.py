"""冻结 S2b 的行业与流通市值代理匹配稳健性审计。

本审计不搜索门控参数，只回答：此前同日同成交额十分位的历史正超额，在进一步控制
点时行业和更细规模后是否仍然存在。

数据边界：

- 行业使用 BaoStock 在每年首个预声明锚点可见的历史行业快照，并在当年内冻结；
- 规模使用腾讯当前报价中的流通股本，再乘本地决策日的前复权
  收盘价，得到 ``float_mcap_proxy``；
- 当前股本包含事后信息，且不能精确处理历史股本变化。因此结果只能作为稳健性证据，
  不能替代点时 ``daily_basic.circ_mv`` 的确认性检验，也不能授权实盘。

首次运行需要临时 BaoStock 依赖，缓存写入 ``scripts/_output``：

    uv run --with baostock==0.9.3 python scripts/s2b_industry_size_proxy_audit.py

后续缓存命中时可直接：

    uv run --no-sync python scripts/s2b_industry_size_proxy_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import _s2c_core as core
import numpy as np
import pandas as pd
import requests
import s2b_matched_control_audit as base
from surge_market_state_filter import StableControlSampler

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "s2b_industry_size_proxy"
REFERENCE_DIR = OUTPUT_DIR / "reference"
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
CAPITAL_PATH = REFERENCE_DIR / "tencent_current_capital.parquet"
REFERENCE_MANIFEST_PATH = REFERENCE_DIR / "manifest.json"

INDUSTRY_ANCHORS = {
    2021: "2021-07-01",
    2022: "2022-01-04",
    2023: "2023-01-03",
    2024: "2024-01-02",
    2025: "2025-01-02",
    2026: "2026-01-05",
}
TENCENT_URL = "https://qt.gtimg.cn/q="
TENCENT_BATCH_SIZE = 50
MATCH_K = 10
MIN_VALID = 5
CALIPER_RATIO = 1.5


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tencent_quote_code(symbol: str) -> str:
    """把仓库代码转换为腾讯报价代码。"""

    code, suffix = str(symbol).split(".", maxsplit=1)
    return f"{suffix.lower()}{code}"


def _quote_float(value: str, *, scale: float = 1.0) -> float:
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return np.nan


def _get_tencent_quotes(codes: list[str], retries: int = 4) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                TENCENT_URL + ",".join(codes),
                headers={"Referer": "https://gu.qq.com/", "User-Agent": "Mozilla/5.0"},
                timeout=(5, 20),
            )
            response.raise_for_status()
            response.encoding = "gbk"
            if not response.text.strip():
                raise RuntimeError("Tencent quote response is empty")
            return response.text
        except Exception as exc:  # pragma: no cover - network failures vary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Tencent quote request failed after {retries} attempts: {last_error}") from last_error


def fetch_capital_snapshot(*, refresh: bool = False) -> pd.DataFrame:
    """获取一次全市场当前流通股本快照并缓存。"""

    if CAPITAL_PATH.exists() and not refresh:
        return pd.read_parquet(CAPITAL_PATH)

    panel_symbols = sorted(
        pd.read_parquet(core.PANEL_PATH, columns=["symbol"])["symbol"].drop_duplicates().astype(str).tolist()
    )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(panel_symbols), TENCENT_BATCH_SIZE):
        symbols = panel_symbols[start : start + TENCENT_BATCH_SIZE]
        quote_codes = [tencent_quote_code(symbol) for symbol in symbols]
        symbol_by_quote = dict(zip(quote_codes, symbols, strict=False))
        text = _get_tencent_quotes(quote_codes)
        for line in text.strip().splitlines():
            if '="' not in line:
                continue
            variable, payload = line.split('="', maxsplit=1)
            values = payload.rsplit('"', maxsplit=1)[0].split("~")
            quote_code = variable.removeprefix("v_")
            if len(values) < 74 or quote_code not in symbol_by_quote:
                continue
            rows.append(
                {
                    "symbol": symbol_by_quote[quote_code],
                    "name": values[1],
                    "current_price": values[3],
                    "quote_timestamp": values[30],
                    "circ_mv": _quote_float(values[44], scale=100_000_000),
                    "total_mv": _quote_float(values[45], scale=100_000_000),
                    "current_float_shares": values[72],
                    "current_total_shares": values[73],
                }
            )
        print(
            f"[capital] requested={min(start + TENCENT_BATCH_SIZE, len(panel_symbols))}/{len(panel_symbols)} rows={len(rows)}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    required = {
        "symbol",
        "name",
        "current_price",
        "quote_timestamp",
        "total_mv",
        "circ_mv",
        "current_float_shares",
        "current_total_shares",
    }
    if missing := required - set(frame):
        raise RuntimeError(f"capital snapshot missing fields: {sorted(missing)}")
    for column in ("current_price", "total_mv", "circ_mv", "current_float_shares", "current_total_shares"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    fallback_shares = frame["circ_mv"] / frame["current_price"]
    frame["current_float_shares"] = frame["current_float_shares"].fillna(fallback_shares)
    frame.loc[
        ~np.isfinite(frame["current_float_shares"]) | frame["current_float_shares"].le(0),
        "current_float_shares",
    ] = np.nan
    relative_error = (frame["current_float_shares"] / fallback_shares - 1).abs().replace([np.inf, -np.inf], np.nan)
    if relative_error.notna().mean() < 0.95 or relative_error.quantile(0.95) > 0.02:
        raise RuntimeError("Tencent quote field positions failed float-share / circ-mv consistency check")
    if frame.duplicated("symbol").any():
        raise RuntimeError("capital snapshot contains duplicate symbols")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    frame.sort_values("symbol").to_parquet(CAPITAL_PATH, index=False)
    return frame


def _industry_path(year: int) -> Path:
    return REFERENCE_DIR / f"baostock_industry_{year}_{INDUSTRY_ANCHORS[year].replace('-', '')}.parquet"


def fetch_industry_snapshots(*, refresh: bool = False) -> dict[int, pd.DataFrame]:
    """获取预声明年度锚点的历史行业快照。"""

    missing_years = [year for year in INDUSTRY_ANCHORS if refresh or not _industry_path(year).exists()]
    if missing_years:
        try:
            import baostock as bs
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("missing baostock; run with `uv run --with baostock==0.9.3`") from exc

        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
        try:
            for year in missing_years:
                anchor = INDUSTRY_ANCHORS[year]
                result = bs.query_stock_industry(date=anchor)
                rows: list[list[str]] = []
                while result.error_code == "0" and result.next():
                    rows.append(result.get_row_data())
                if result.error_code != "0":
                    raise RuntimeError(f"BaoStock industry[{anchor}] failed: {result.error_msg}")
                frame = pd.DataFrame(rows, columns=result.fields)
                required = {"updateDate", "code", "industry", "industryClassification"}
                if missing := required - set(frame):
                    raise RuntimeError(f"industry[{anchor}] missing fields: {sorted(missing)}")
                frame = frame[list(required)].copy()
                frame["symbol"] = frame["code"].map(baostock_symbol)
                frame = frame[frame["industry"].astype(str).str.len().gt(0)].copy()
                if frame.duplicated("symbol").any():
                    raise RuntimeError(f"industry[{anchor}] contains duplicate symbols")
                frame.sort_values("symbol").to_parquet(_industry_path(year), index=False)
                print(f"[industry] {anchor}: {len(frame)} rows", flush=True)
        finally:
            bs.logout()

    return {year: pd.read_parquet(_industry_path(year)) for year in INDUSTRY_ANCHORS}


def baostock_symbol(code: str) -> str:
    """把 BaoStock 代码转换为仓库代码。"""

    market, digits = str(code).split(".", maxsplit=1)
    suffix = "BJ" if digits.startswith(("4", "8", "9")) else "SH" if market.lower() == "sh" else "SZ"
    return f"{digits}.{suffix}"


def nearest_control_symbols(
    sizes: pd.Series,
    *,
    treated_symbol: str,
    eligible_symbols: pd.Index,
    k: int = MATCH_K,
    caliper_ratio: float | None = None,
) -> list[str]:
    """按对数规模距离、代码稳定排序选择最近邻。"""

    treated_size = sizes.get(treated_symbol)
    if treated_size is None or not np.isfinite(treated_size) or treated_size <= 0:
        return []
    candidates = sizes.reindex(eligible_symbols).drop(index=treated_symbol, errors="ignore").dropna()
    candidates = candidates[candidates.gt(0)]
    distance = (np.log(candidates) - math.log(float(treated_size))).abs()
    if caliper_ratio is not None:
        distance = distance[distance <= math.log(caliper_ratio)]
    ordered = pd.DataFrame({"symbol": distance.index.astype(str), "distance": distance.to_numpy()}).sort_values(
        ["distance", "symbol"], kind="mergesort"
    )
    return ordered["symbol"].head(k).tolist()


def _match_trade(
    trade: Any,
    *,
    sampler: StableControlSampler,
    shares: pd.Series,
    industries: pd.Series | None,
    caliper_ratio: float | None,
) -> dict[str, Any]:
    dec_dt = pd.Timestamp(trade.dec_dt)
    entry_dt = pd.Timestamp(trade.entry_dt)
    exit_dt = pd.Timestamp(trade.exit_dt)
    if dec_dt not in sampler.dates or entry_dt not in sampler.dates or exit_dt not in sampler.dates:
        return {
            "excess_pct": np.nan,
            "pool_n": 0,
            "used_n": 0,
            "max_size_ratio": np.nan,
            "control_symbols": [],
        }

    sizes = sampler.close_w.loc[dec_dt].mul(shares, fill_value=np.nan)
    eligible = sizes.index
    if industries is not None:
        treated_industry = industries.get(trade.symbol)
        if treated_industry is None or pd.isna(treated_industry) or not str(treated_industry):
            return {
                "excess_pct": np.nan,
                "pool_n": 0,
                "used_n": 0,
                "max_size_ratio": np.nan,
                "control_symbols": [],
            }
        eligible = industries.index[industries.eq(treated_industry)]

    valid_prices = sampler.open_w.loc[entry_dt].notna() & sampler.close_w.loc[exit_dt].notna()
    eligible = eligible.intersection(valid_prices.index[valid_prices])
    treated_size = sizes.get(trade.symbol)
    pool_sizes = sizes.reindex(eligible).drop(index=trade.symbol, errors="ignore").dropna()
    pool_sizes = pool_sizes[pool_sizes.gt(0)]
    if caliper_ratio is not None and treated_size is not None and np.isfinite(treated_size) and treated_size > 0:
        pool_sizes = pool_sizes[(np.log(pool_sizes) - math.log(float(treated_size))).abs() <= math.log(caliper_ratio)]
    symbols = nearest_control_symbols(
        sizes,
        treated_symbol=trade.symbol,
        eligible_symbols=eligible,
        k=MATCH_K,
        caliper_ratio=caliper_ratio,
    )
    pool_n = int(len(pool_sizes))
    if len(symbols) < MIN_VALID:
        return {
            "excess_pct": np.nan,
            "pool_n": pool_n,
            "used_n": len(symbols),
            "max_size_ratio": np.nan,
            "control_symbols": symbols,
        }
    returns = (
        sampler.close_w.loc[exit_dt, symbols].to_numpy(float) / sampler.open_w.loc[entry_dt, symbols].to_numpy(float)
        - 1
    )
    returns = returns[np.isfinite(returns)]
    if len(returns) < MIN_VALID:
        return {
            "excess_pct": np.nan,
            "pool_n": pool_n,
            "used_n": len(returns),
            "max_size_ratio": np.nan,
            "control_symbols": symbols,
        }
    treated_size = float(sizes[trade.symbol])
    ratios = sizes.loc[symbols].to_numpy(float) / treated_size
    max_ratio = float(np.maximum(ratios, 1 / ratios).max())
    return {
        "excess_pct": float(trade.ret_gross_pct - np.median(returns) * 100),
        "pool_n": pool_n,
        "used_n": int(len(returns)),
        "max_size_ratio": max_ratio,
        "control_symbols": symbols,
    }


def _summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    scoped = frame.rename(columns={column: "matched_excess_pct"})
    return base.summarize_matched_excess(scoped)


def _window_summaries(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    arch = frame[frame["dec_dt"].between(*base.ARCH_FRESH)]
    main = frame[frame["dec_dt"] >= base.MAIN_START]
    holdout = frame[frame["dec_dt"].between(*base.HOLDOUT_2026)]
    return {
        "all_mature": _summary(frame, column),
        "architecture_fresh_2021_07_to_2023_12": _summary(arch, column),
        "main_2024plus": _summary(main, column),
        "main_executed_10_slots": _summary(main[main["executed_main"]], column),
        "holdout_2026_to_2026_07_27": _summary(holdout, column),
        "holdout_2026_executed_10_slots": _summary(holdout[holdout["executed_main"]], column),
    }


def _yearly_summaries(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(int(year)): _summary(group, column) for year, group in frame.groupby(frame["dec_dt"].dt.year, sort=True)
    }


def _confirmed(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("daily_hac", {}).get("t_stat", -np.inf) >= 2
        and summary.get("daily_block_bootstrap", {}).get("ci_lo", -np.inf) > 0
    )


def concentration_audit(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    """检查正超额是否由少数交易、日期或行业贡献。"""

    valid = frame.dropna(subset=[column]).sort_values(column, ascending=False).copy()
    if valid.empty:
        return {"n_trades": 0}
    positive = valid.loc[valid[column] > 0, column]
    daily = valid.groupby("dec_dt", sort=True)[column].mean().sort_values(ascending=False)
    industry = (
        valid.groupby("treated_industry", dropna=False)[column]
        .agg([("n", "size"), ("mean_excess_pct", "mean"), ("total_excess", "sum")])
        .sort_values("total_excess", ascending=False)
    )
    top_industry = industry.index[0]
    winsorized = valid.copy()
    winsorized[column] = winsorized[column].clip(valid[column].quantile(0.05), valid[column].quantile(0.95))
    return {
        "n_trades": int(len(valid)),
        "n_dates": int(valid["dec_dt"].nunique()),
        "n_industries": int(valid["treated_industry"].nunique(dropna=True)),
        "top_5_positive_trade_share_pct": (
            round(float(positive.head(5).sum() / positive.sum() * 100), 2) if positive.sum() > 0 else None
        ),
        "top_5_daily_mean_share_pct": (
            round(float(daily.head(5).sum() / daily[daily > 0].sum() * 100), 2) if daily[daily > 0].sum() > 0 else None
        ),
        "full": _summary(valid, column),
        "drop_top_1_trade": _summary(valid.iloc[1:], column),
        "drop_top_5_trades": _summary(valid.iloc[5:], column),
        "drop_top_10_trades": _summary(valid.iloc[10:], column),
        "winsorized_5_95": _summary(winsorized, column),
        "top_industry": str(top_industry),
        "drop_top_industry": _summary(valid[valid["treated_industry"].ne(top_industry)], column),
        "industry_contributions": [
            {
                "industry": str(index),
                "n": int(row["n"]),
                "mean_excess_pct": round(float(row["mean_excess_pct"]), 3),
                "total_excess": round(float(row["total_excess"]), 3),
            }
            for index, row in industry.head(10).iterrows()
        ],
    }


def _reference_manifest(capital: pd.DataFrame, industries: dict[int, pd.DataFrame]) -> dict[str, Any]:
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "capital": {
            "source": TENCENT_URL,
            "rows": int(len(capital)),
            "valid_current_float_shares": int(capital["current_float_shares"].notna().sum()),
            "quote_timestamp_min": str(capital["quote_timestamp"].min()),
            "quote_timestamp_max": str(capital["quote_timestamp"].max()),
            "path": str(CAPITAL_PATH),
            "sha256": sha256_file(CAPITAL_PATH),
        },
        "industry": {
            str(year): {
                "requested_anchor": INDUSTRY_ANCHORS[year],
                "effective_dates": sorted(frame["updateDate"].astype(str).unique().tolist()),
                "rows": int(len(frame)),
                "industries": int(frame["industry"].nunique()),
                "path": str(_industry_path(year)),
                "sha256": sha256_file(_industry_path(year)),
            }
            for year, frame in industries.items()
        },
    }


def run_audit(*, refresh_reference: bool = False) -> dict[str, Any]:
    """运行冻结稳健性审计。"""

    capital = fetch_capital_snapshot(refresh=refresh_reference)
    industry_frames = fetch_industry_snapshots(refresh=refresh_reference)
    manifest = _reference_manifest(capital, industry_frames)
    REFERENCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    context = core.load_context(verbose=True)
    d0 = context["d0"]
    s2b = d0.loc[core.gate_mask(d0, core.S2B_VR, core.S2B_SP)].copy()
    for column in ("dec_dt", "entry_dt", "exit_dt"):
        s2b[column] = pd.to_datetime(s2b[column])
    s2b["censored_tail"] = base.censored_tail_mask(s2b)
    execution_keys = base._main_execution_keys(context)
    s2b["executed_main"] = [
        (symbol, entry_dt) in execution_keys for symbol, entry_dt in zip(s2b["symbol"], s2b["entry_dt"], strict=False)
    ]
    mature = s2b[~s2b["censored_tail"]].copy()

    sampler = StableControlSampler()
    shares = capital.set_index("symbol")["current_float_shares"].reindex(sampler.close_w.columns)
    industry_maps = {
        year: frame.set_index("symbol")["industry"].reindex(sampler.close_w.columns)
        for year, frame in industry_frames.items()
    }
    mature["treated_industry"] = [
        industry_maps[int(dec_dt.year)].get(symbol)
        for symbol, dec_dt in zip(mature["symbol"], mature["dec_dt"], strict=False)
    ]

    baseline: list[float] = []
    size_rows: list[dict[str, Any]] = []
    industry_size_rows: list[dict[str, Any]] = []
    caliper_rows: list[dict[str, Any]] = []
    for index, trade in enumerate(mature.itertuples(), 1):
        industries = industry_maps[int(pd.Timestamp(trade.dec_dt).year)]
        baseline.append(sampler.excess_for(trade))
        size_rows.append(_match_trade(trade, sampler=sampler, shares=shares, industries=None, caliper_ratio=None))
        industry_size_rows.append(
            _match_trade(trade, sampler=sampler, shares=shares, industries=industries, caliper_ratio=None)
        )
        caliper_rows.append(
            _match_trade(trade, sampler=sampler, shares=shares, industries=industries, caliper_ratio=CALIPER_RATIO)
        )
        if index % 100 == 0 or index == len(mature):
            print(f"[match] {index}/{len(mature)}", flush=True)

    mature["amount_decile_excess_pct"] = baseline
    specs = {
        "size_proxy_nearest_k10": size_rows,
        "industry_size_proxy_nearest_k10": industry_size_rows,
        "industry_size_proxy_caliper_1_5x_k10": caliper_rows,
    }
    diagnostics: dict[str, Any] = {}
    summaries = {"amount_decile_k50": _window_summaries(mature, "amount_decile_excess_pct")}
    yearly = {
        "amount_decile_k50": {
            "all_candidates": _yearly_summaries(mature, "amount_decile_excess_pct"),
            "executed_10_slots": _yearly_summaries(mature[mature["executed_main"]], "amount_decile_excess_pct"),
        }
    }
    for name, rows in specs.items():
        prefix = name
        result = pd.DataFrame(rows, index=mature.index)
        excess_column = f"{prefix}_excess_pct"
        mature[excess_column] = result["excess_pct"]
        valid = mature[excess_column].notna()
        diagnostics[name] = {
            "matched": int(valid.sum()),
            "coverage_pct": round(float(valid.mean() * 100), 2),
            "pool_n_median": round(float(result.loc[valid, "pool_n"].median()), 1) if valid.any() else None,
            "used_n_median": round(float(result.loc[valid, "used_n"].median()), 1) if valid.any() else None,
            "max_size_ratio_median": round(float(result.loc[valid, "max_size_ratio"].median()), 4)
            if valid.any()
            else None,
            "max_size_ratio_p95": round(float(result.loc[valid, "max_size_ratio"].quantile(0.95)), 4)
            if valid.any()
            else None,
        }
        summaries[name] = _window_summaries(mature, excess_column)
        yearly[name] = {
            "all_candidates": _yearly_summaries(mature, excess_column),
            "executed_10_slots": _yearly_summaries(mature[mature["executed_main"]], excess_column),
        }
        common = mature[valid & mature["amount_decile_excess_pct"].notna()].copy()
        summaries[f"amount_decile_k50_on_{name}_support"] = _window_summaries(common, "amount_decile_excess_pct")

    architecture = summaries["industry_size_proxy_caliper_1_5x_k10"]["architecture_fresh_2021_07_to_2023_12"]
    historical = summaries["industry_size_proxy_caliper_1_5x_k10"]["main_executed_10_slots"]
    current = summaries["industry_size_proxy_caliper_1_5x_k10"]["holdout_2026_executed_10_slots"]
    annual_executed = yearly["industry_size_proxy_caliper_1_5x_k10"]["executed_10_slots"]
    annual_confirmations = {year: _confirmed(summary) for year, summary in annual_executed.items()}
    architecture_confirmed = _confirmed(architecture)
    historical_confirmed = _confirmed(historical)
    current_confirmed = _confirmed(current)
    historical_years_robust = all(annual_confirmations.get(year, False) for year in ("2024", "2025"))
    strict_column = "industry_size_proxy_caliper_1_5x_k10_excess_pct"
    concentration_2025 = concentration_audit(
        mature[mature["executed_main"] & mature["dec_dt"].dt.year.eq(2025)], strict_column
    )
    return {
        "schema": "s2b_industry_size_proxy_audit_v1",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "design": {
            "gate": {"sig_vol_ratio_lte": core.S2B_VR, "sig_ma_spread_pct_gte": core.S2B_SP},
            "industry": "BaoStock historical snapshot at frozen annual anchor; frozen within year",
            "size_proxy": "decision-date qfq close times current float shares from Tencent quote snapshot",
            "matching": f"nearest log-size, K={MATCH_K}, min_valid={MIN_VALID}; sensitivity includes same-industry {CALIPER_RATIO}x caliper",
            "comparison": "gross strategy return minus gross control median, same entry/exit dates",
            "parameter_search": "none; all three matching specifications reported",
            "limitation": "current shares are ex-post and do not exactly represent historical point-in-time circ_mv",
        },
        "references": manifest,
        "counts": {
            "s2b_total": int(len(s2b)),
            "mature": int(len(mature)),
            "capital_covered_treated": int(
                mature["symbol"].isin(capital.loc[capital["current_float_shares"].notna(), "symbol"]).sum()
            ),
        },
        "diagnostics": diagnostics,
        "summaries": summaries,
        "yearly": yearly,
        "concentration_2025_executed": concentration_2025,
        "verdict": {
            "architecture_fresh_proxy_edge": architecture_confirmed,
            "historical_2024plus_executed_proxy_edge": historical_confirmed,
            "historical_2024_and_2025_both_confirmed": historical_years_robust,
            "annual_executed_confirmations": annual_confirmations,
            "holdout_2026_executed_proxy_edge": current_confirmed,
            "status": (
                "EDGE_CONCENTRATED_IN_2025_NOT_TEMPORALLY_ROBUST"
                if historical_confirmed
                and not historical_years_robust
                and annual_confirmations.get("2025", False)
                and not architecture_confirmed
                and not current_confirmed
                else "REVIEW_REQUIRED"
            ),
            "live_authorized": False,
            "reason": (
                "The combined 2024+ edge is concentrated in 2025: strict industry/size-proxy matches do not confirm "
                "2024, the architecture-fresh window, or 2026. The proxy is ex-post; exact point-in-time circ_mv and "
                "a new frozen forward window are still required before any deployment claim."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reference", action="store_true")
    args = parser.parse_args()
    started = time.time()
    result = run_audit(refresh_reference=args.refresh_reference)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {OUTPUT_PATH} | {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
