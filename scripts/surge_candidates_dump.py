"""主升浪候选信号全市场抽取（门控前 + 多延迟模拟）—— 下游分析的共享数据源

对每只股票全量流式重放，记录**门控前**的主升浪候选（仅状态跳变 + 路径条件，
量比/散度/ret20 门控不在此应用，存特征供下游后置过滤 → 门控敏感性扫描零成本）：

- confirm    跳变进入 7/8，且 prior 40 根走过 4 与 5；
- anticipate 跳变进入 5，且 prior 40 根走过 4。

每个候选 × 决策延迟 d∈{0,1,2,3,5,7,10}（信号后第 d 根收盘决策、次日开盘入场，
要求决策日仍处于上行家族 5/6/7/8），先无条件记录只依赖决策日及之前信息的 raw 行，
再在未来成交与完整退出均可观察时附加 FULL 结果。因右端截尾而没有未来入场或退出的 raw 行
不会被丢弃，``full_outcome_complete`` 明确区分可用于历史结果分析的完整样本。

FULL 退出显式区分信号与成交：盘中止损可在当日成交；由收盘确认的 18% 跟踪止损和
状态 9/10 退出统一在下一根可观察 bar 开盘成交；最大持有期使用预先确定的第 60 根收盘。

候选独立模拟、允许同票时间重叠
（组合层会强制单票单仓；pair 级分析接受重叠）。

输出：
- scripts/_output/surge_candidates/candidates.parquet  一行 = (候选, delay)
- scripts/_output/surge_candidates/panel.parquet       全市场 dt×symbol 的 open/close/amount

    uv run --no-sync python scripts/surge_candidates_dump.py
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trend_regime as tr
from trend_regime import Regime

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELAYS = [0, 1, 2, 3, 5, 7, 10]
MAX_HOLD_DAYS = 60
TRAIL_STOP = 0.18
TRAIN_END = pd.Timestamp("2023-12-31")
SELL_SET = tr.SELL_REGIMES  # {9, 10}
UPTREND_FAMILY = {int(Regime.UpwardDeparture), int(Regime.ThirdBuy), int(Regime.MainUptrend), int(Regime.Acceleration)}


_MAIN_UP_OR_ACCEL = {int(Regime.MainUptrend), int(Regime.Acceleration)}
_PIVOT_BUILDING = int(Regime.PivotBuilding)
_UPWARD_DEPARTURE = int(Regime.UpwardDeparture)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory_identity(paths: list[str]) -> dict[str, Any]:
    """内容寻址本次被枚举的本地源文件集合。"""

    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "first": rows[0]["name"] if rows else None,
        "last": rows[-1]["name"] if rows else None,
    }


def _is_candidate(prev: int, regime: int, prior: list[int], mode: str) -> bool:
    """门控前候选：仅状态跳变 + 路径条件（surge_onset 去掉特征门控的部分）。"""
    prior_set = set(prior)
    if mode == "confirm":
        entered = prev not in _MAIN_UP_OR_ACCEL and regime in _MAIN_UP_OR_ACCEL
        return entered and _PIVOT_BUILDING in prior_set and _UPWARD_DEPARTURE in prior_set
    entered = prev != _UPWARD_DEPARTURE and regime == _UPWARD_DEPARTURE
    return entered and _PIVOT_BUILDING in prior_set


@dataclass(frozen=True)
class FullExitResult:
    """一笔具有明确退出信号/成交时点的完整 FULL 结果。"""

    entry_idx: int
    entry_dt: pd.Timestamp
    entry_price: float
    exit_signal_idx: int
    exit_signal_dt: pd.Timestamp
    exit_fill_idx: int
    exit_fill_dt: pd.Timestamp
    exit_price: float
    reason: str

    @property
    def hold_signal_days(self) -> int:
        return self.exit_signal_idx - self.entry_idx

    @property
    def hold_fill_days(self) -> int:
        return self.exit_fill_idx - self.entry_idx


def _next_open_fill(index: int, opens: np.ndarray, dates: Any) -> tuple[int, pd.Timestamp, float] | None:
    """返回收盘信号之后的下一根有限开盘；右端缺失时结果尚未成熟。"""

    fill_idx = index + 1
    if fill_idx >= len(opens) or not np.isfinite(opens[fill_idx]) or opens[fill_idx] <= 0:
        return None
    return fill_idx, pd.Timestamp(dates[fill_idx]), float(opens[fill_idx])


def _simulate_full(p_dec: int, states: list, regime_by_idx: dict, ind: dict) -> FullExitResult | None:
    """从决策 bar 模拟完整 FULL；未来成交或退出不可观察时返回 ``None``。"""

    n = ind["n"]
    o, c, lo = ind["open"], ind["close"], ind["low"]
    dates = ind["dates"]
    sig = states[p_dec]
    entry_idx = sig.idx + 1
    if entry_idx >= n or not np.isfinite(sig.next_open) or sig.next_open <= 0:
        return None
    entry_price = float(sig.next_open)
    sl_ref = sig.sl_ref

    peak = entry_price
    max_hold_idx = entry_idx + MAX_HOLD_DAYS - 1
    observed_last_idx = min(max_hold_idx, n - 1)
    for j in range(entry_idx, observed_last_idx + 1):
        peak = max(peak, c[j])
        if j == entry_idx:
            continue
        if not np.isnan(sl_ref) and lo[j] <= sl_ref:
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=pd.Timestamp(dates[entry_idx]),
                entry_price=entry_price,
                exit_signal_idx=j,
                exit_signal_dt=pd.Timestamp(dates[j]),
                exit_fill_idx=j,
                exit_fill_dt=pd.Timestamp(dates[j]),
                exit_price=float(min(o[j], sl_ref)),
                reason="sl2",
            )
        if peak > entry_price and (peak - c[j]) / peak >= TRAIL_STOP:
            fill = _next_open_fill(j, o, dates)
            if fill is None:
                return None
            fill_idx, fill_dt, fill_price = fill
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=pd.Timestamp(dates[entry_idx]),
                entry_price=entry_price,
                exit_signal_idx=j,
                exit_signal_dt=pd.Timestamp(dates[j]),
                exit_fill_idx=fill_idx,
                exit_fill_dt=fill_dt,
                exit_price=fill_price,
                reason="trail18",
            )
        if regime_by_idx.get(j) in SELL_SET:
            fill = _next_open_fill(j, o, dates)
            if fill is None:
                return None
            fill_idx, fill_dt, fill_price = fill
            return FullExitResult(
                entry_idx=entry_idx,
                entry_dt=pd.Timestamp(dates[entry_idx]),
                entry_price=entry_price,
                exit_signal_idx=j,
                exit_signal_dt=pd.Timestamp(dates[j]),
                exit_fill_idx=fill_idx,
                exit_fill_dt=fill_dt,
                exit_price=fill_price,
                reason="state",
            )

    if max_hold_idx >= n or not np.isfinite(c[max_hold_idx]) or c[max_hold_idx] <= 0:
        return None
    return FullExitResult(
        entry_idx=entry_idx,
        entry_dt=pd.Timestamp(dates[entry_idx]),
        entry_price=entry_price,
        exit_signal_idx=max_hold_idx,
        exit_signal_dt=pd.Timestamp(dates[max_hold_idx]),
        exit_fill_idx=max_hold_idx,
        exit_fill_dt=pd.Timestamp(dates[max_hold_idx]),
        exit_price=float(c[max_hold_idx]),
        reason="max_hold",
    )


def _feat(feats: dict | None, key: str) -> float:
    if not feats:
        return np.nan
    v = feats.get(key)
    return np.nan if v is None else float(v)


def completed_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """返回 FULL 结果完整的历史样本，并拒绝旧 schema 被静默当作 raw-complete。"""

    if "full_outcome_complete" not in frame:
        raise ValueError("candidate data predates the raw-complete schema; rebuild surge_candidates_dump.py")
    return frame.loc[frame["full_outcome_complete"].fillna(False)].copy()


def _outcome_fields(result: FullExitResult | None) -> dict[str, Any]:
    """把完整 FULL 结果转成稳定 schema；未成熟 raw 行保留空结果。"""

    if result is None:
        return {
            "full_outcome_complete": False,
            "entry_idx": None,
            "entry_dt": None,
            "entry_price": None,
            "exit_signal_idx": None,
            "exit_signal_dt": None,
            "exit_fill_idx": None,
            "exit_fill_dt": None,
            "exit_idx": None,
            "exit_dt": None,
            "exit_price": None,
            "ret_gross_pct": None,
            "hold_signal_days": None,
            "hold_days": None,
            "exit_reason": None,
            "gap_pct": None,
        }
    gross = (result.exit_price / result.entry_price - 1) * 100
    return {
        "full_outcome_complete": True,
        "entry_idx": result.entry_idx,
        "entry_dt": result.entry_dt,
        "entry_price": result.entry_price,
        "exit_signal_idx": result.exit_signal_idx,
        "exit_signal_dt": result.exit_signal_dt,
        "exit_fill_idx": result.exit_fill_idx,
        "exit_fill_dt": result.exit_fill_dt,
        # 兼容历史研究字段，但语义统一为真实成交时点。
        "exit_idx": result.exit_fill_idx,
        "exit_dt": result.exit_fill_dt,
        "exit_price": result.exit_price,
        "ret_gross_pct": round(gross, 3),
        "hold_signal_days": result.hold_signal_days,
        "hold_days": result.hold_fill_days,
        "exit_reason": result.reason,
        "gap_pct": None,
    }


def _process(parquet_path: str) -> dict[str, Any]:
    df = tr.load_stock(parquet_path)
    if df is None:
        try:
            raw = pd.read_parquet(parquet_path, columns=["ts_code"])
        except Exception:
            status = "unreadable"
        else:
            if len(raw) < tr.MIN_BARS:
                status = "insufficient_bars"
            elif raw.empty or str(raw["ts_code"].iloc[0]).startswith(tr.EXCLUDE_PREFIX):
                status = "excluded_board"
            else:
                status = "load_rejected_unknown"
        return {"status": status, "path": parquet_path, "rows": []}
    states = tr.iter_states(df, with_features=True)
    if len(states) < 30:
        return {"status": "insufficient_states", "path": parquet_path, "rows": []}
    ind = tr.compute_indicators(df)
    regimes = [s.regime for s in states]
    regime_by_idx = {s.idx: s.regime for s in states}
    symbol = df["symbol"].iloc[0]
    amount = df["amount"].to_numpy(dtype=float) if "amount" in df.columns else np.full(ind["n"], np.nan)
    limit_pct = tr.limit_pct_for(symbol)

    rows = []
    for mode in ("confirm", "anticipate"):
        for p in range(1, len(states)):
            prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
            if not _is_candidate(states[p - 1].regime, states[p].regime, prior, mode):
                continue
            sig_feats = states[p].feats or {}
            for d in DELAYS:
                p_dec = p + d
                if p_dec >= len(states) or states[p_dec].regime not in UPTREND_FAMILY:
                    continue  # 决策日已不在上行家族 → 实盘不会列出
                dec = states[p_dec]
                dec_close = dec.close
                sl = dec.sl_ref if dec.sl_ref == dec.sl_ref else dec.zd
                sl = sl if (sl == sl and sl > 0) else np.nan
                sl_pct = (dec_close - sl) / dec_close * 100 if sl == sl else np.nan
                dec_amt = amount[dec.idx]
                row = {
                    "symbol": symbol,
                    "mode": mode,
                    "delay": d,
                    "sig_dt": states[p].dt,
                    "dec_dt": dec.dt,
                    "dec_idx": int(dec.idx),
                    "dec_close": float(dec_close),
                    "dec_regime": int(dec.regime),
                    "sl_ref": float(sl) if sl == sl else np.nan,
                    "sl_pct": round(sl_pct, 2) if sl_pct == sl_pct else np.nan,
                    "score": tr.surge_score(dec.feats),
                    "amount_e": round(dec_amt / 1e5, 3) if dec_amt == dec_amt else np.nan,
                    "limit_pct": limit_pct,
                    "sig_vol_ratio": _feat(sig_feats, "vol_ratio"),
                    "sig_ma_spread_pct": _feat(sig_feats, "ma_spread_pct"),
                    "sig_ret20": _feat(sig_feats, "ret20"),
                    "sig_above_zg": _feat(sig_feats, "above_zg"),
                }
                outcome = _outcome_fields(_simulate_full(p_dec, states, regime_by_idx, ind))
                entry_idx = int(dec.idx) + 1
                if entry_idx < ind["n"] and np.isfinite(ind["open"][entry_idx]) and ind["open"][entry_idx] > 0:
                    entry_price = float(ind["open"][entry_idx])
                    outcome.update(
                        {
                            "entry_idx": entry_idx,
                            "entry_dt": pd.Timestamp(ind["dates"][entry_idx]),
                            "entry_price": entry_price,
                            "gap_pct": round((entry_price / dec_close - 1) * 100, 2),
                        }
                    )
                row.update(outcome)
                rows.append(row)
    return {"status": "processed", "path": parquet_path, "rows": rows}


def _panel_one(parquet_path: str) -> pd.DataFrame | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "dt": df["dt"],
            "open": df["open"].astype(float),
            "close": df["close"].astype(float),
            "amount_e": (df["amount"].astype(float) / 1e5) if "amount" in df.columns else np.nan,
        }
    )
    return out


def main():
    t0 = time.time()
    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只 | {n_workers} 进程 | 延迟集 {DELAYS}")

    ctx = mp.get_context("spawn")
    all_rows = []
    process_counts = {
        "processed": 0,
        "insufficient_states": 0,
        "insufficient_bars": 0,
        "excluded_board": 0,
        "unreadable": 0,
        "load_rejected_unknown": 0,
    }
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, files, chunksize=20), 1):
            process_counts[res["status"]] += 1
            all_rows.extend(res["rows"])
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 候选行 {len(all_rows)} | {time.time() - t0:.0f}s")

    cand = pd.DataFrame(all_rows)
    if cand.empty:
        print("[错误] 未找到任何候选信号，请检查 iter_states / _is_candidate 逻辑")
        return
    cand["seg"] = np.where(cand["dec_dt"] <= TRAIN_END, "train", "test")
    cand["year"] = cand["dec_dt"].dt.year
    candidate_path = OUTPUT_DIR / "candidates.parquet"
    cand.to_parquet(candidate_path, index=False)
    complete_count = int(cand["full_outcome_complete"].sum())
    print(f"[候选] raw {len(cand)} 行 / 完整 FULL {complete_count} 行 → candidates.parquet")

    print("[面板] 构建 dt×symbol 价格面板 ...")
    panels = []
    with ctx.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_panel_one, files, chunksize=50):
            if res is not None:
                panels.append(res)
    panel = pd.concat(panels, ignore_index=True)
    panel_path = OUTPUT_DIR / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    print(f"[面板] {len(panel)} 行 → panel.parquet")
    inventory = _source_inventory_identity(files)
    eligible_sources = process_counts["processed"] + process_counts["insufficient_states"]
    all_sources_accounted = sum(process_counts.values()) == len(files) and len(panels) == eligible_sources
    hard_failures = process_counts["unreadable"] + process_counts["load_rejected_unknown"]
    manifest = {
        "schema": "surge_candidates_dump_manifest_v2",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_completeness_proven": hard_failures == 0 and all_sources_accounted,
        "source_files": {
            **inventory,
            "processing": process_counts,
            "panel_source_count": len(panels),
            "eligible_source_count": eligible_sources,
            "all_sources_accounted": all_sources_accounted,
            "eligibility": {
                "minimum_bars": tr.MIN_BARS,
                "excluded_prefixes": list(tr.EXCLUDE_PREFIX),
            },
        },
        "outputs": {
            "candidates": {
                "path": str(candidate_path),
                "sha256": _sha256_file(candidate_path),
                "rows": int(len(cand)),
                "full_outcome_complete_rows": complete_count,
            },
            "panel": {"path": str(panel_path), "sha256": _sha256_file(panel_path), "rows": int(len(panel))},
        },
        "semantics": {
            "row_identity": ["symbol", "mode", "delay", "sig_dt", "dec_dt"],
            "raw_row": "persisted from information available through dec_dt even when future entry/exit is absent",
            "legacy_exit_dt": "alias of exit_fill_dt, never exit_signal_dt",
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["raw_completeness_proven"]:
        raise RuntimeError(f"candidate raw completeness failed: {process_counts}")
    print(f"[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
