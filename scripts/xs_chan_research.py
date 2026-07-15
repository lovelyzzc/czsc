"""低维横截面因子 × 缠论入场门的可复现研究 Pilot。

本脚本的历史输出一律标记为 ``CONTAMINATED_STRESS_ONLY``。它用于验证因果时间线、
有限持仓账本、成本和对照实验，不能证明 alpha 或生成实盘委托。冻结规格见
``XS_CHAN_PILOT_PROTOCOL_V1.md``。

默认运行：

    uv run --no-sync python scripts/xs_chan_research.py

快速工程冒烟：

    uv run --no-sync python scripts/xs_chan_research.py --max-symbols 80 --start-date 20250101
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"
CHAN_CACHE_PATH = SCRIPT_DIR / "_output" / "surge_factors" / "bars.parquet"
UNIVERSE_PANEL_PATH = SCRIPT_DIR / "_output" / "surge_candidates" / "panel.parquet"
OUTPUT_DIR = SCRIPT_DIR / "_output" / "xs_chan_research"
PROTOCOL_PATH = SCRIPT_DIR / "xs_chan_protocol_v1.json"

RESEARCH_LABEL = "CONTAMINATED_STRESS_ONLY"
STATE_COLUMNS = ("symbol", "dt", "regime")
PRICE_COLUMNS = ("ts_code", "trade_date", "open", "close", "pre_close", "vol", "amount")
EXCLUDE_PREFIXES = ("688", "689", "920", "83", "43")
CHINEXT_PREFIXES = ("300", "301", "302")
STAMP_CHANGE_DATE = pd.Timestamp("2023-08-28")
LOT_SIZE = 100


@dataclass(frozen=True)
class PilotConfig:
    """机器可复现的唯一 Pilot 主规格。"""

    protocol_id: str
    start_date: str
    common_cutoff: str
    frequency: str
    holdings: int
    initial_capital_cny: float
    min_history_bars: int
    min_valid_bars_60: int
    min_adv20_thousand_cny: float
    liquidity_buckets: int
    winsor_low: float
    winsor_high: float
    random_seed: int
    allowed_chan_regimes: tuple[int, ...]
    cash_buffer: float
    random_seed_count: int = 20


@dataclass(frozen=True)
class CostModel:
    """显式拆分的成交成本假设，所有 bps 以成交名义金额为基数。"""

    commission_bps: float
    transfer_bps: float
    min_commission_cny: float
    stamp_before_bps: float
    stamp_after_bps: float
    base_slippage_bps: float
    impact_bps_at_1pct: float
    max_adv_participation: float
    multiplier: float = 1.0

    def stamp_bps(self, dt: pd.Timestamp) -> float:
        return self.stamp_after_bps if dt >= STAMP_CHANGE_DATE else self.stamp_before_bps


@dataclass(frozen=True)
class TargetPlan:
    """决策层产物；不包含任何成交价。"""

    decision_dt: pd.Timestamp
    exec_dt: pd.Timestamp
    symbols: tuple[str, ...]
    adv_cny: Mapping[str, float]
    gate_allowed: Mapping[str, bool]


@dataclass
class MarketData:
    """组合撮合需要的窄行情矩阵。"""

    dates: pd.DatetimeIndex
    symbols: pd.Index
    open: np.ndarray
    close: np.ndarray
    pre_close: np.ndarray
    is_st: np.ndarray
    date_to_idx: dict[pd.Timestamp, int]
    symbol_to_idx: dict[str, int]

    @classmethod
    def from_long_frame(
        cls,
        prices: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        symbols: Sequence[str],
    ) -> MarketData:
        dates = pd.DatetimeIndex(calendar).sort_values().unique()
        columns = pd.Index(sorted(set(map(str, symbols))), name="symbol")
        frame = prices.drop_duplicates(["dt", "symbol"], keep="last")

        def matrix(col: str, dtype: str = "float64") -> np.ndarray:
            wide = frame.pivot(index="dt", columns="symbol", values=col)
            return wide.reindex(index=dates, columns=columns).to_numpy(dtype=dtype)

        return cls(
            dates=dates,
            symbols=columns,
            open=matrix("open"),
            close=matrix("close"),
            pre_close=matrix("pre_close"),
            is_st=matrix("is_st", dtype="float64") > 0.5,
            date_to_idx={pd.Timestamp(dt): i for i, dt in enumerate(dates)},
            symbol_to_idx={str(symbol): i for i, symbol in enumerate(columns)},
        )


@dataclass(frozen=True)
class PendingSell:
    decision_dt: pd.Timestamp
    target_shares: int
    adv_cny: float


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    intents: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--namechange-path", type=Path, default=NAMECHANGE_PATH)
    parser.add_argument("--chan-cache", type=Path, default=CHAN_CACHE_PATH)
    parser.add_argument("--universe-panel", type=Path, default=UNIVERSE_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-date", default=None, help="覆盖协议起点时只生成 SMOKE_ONLY 产物。")
    parser.add_argument("--cutoff", default=None, help="只能早于协议共同截止；非默认运行标为 smoke。")
    parser.add_argument("--frequency", choices=("weekly", "daily"), default="weekly")
    parser.add_argument("--max-symbols", type=int, default=None, help="仅用于工程冒烟，不得用于收益判断。")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default).encode()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projected_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """对实际进入研究的列/行生成稳定投影指纹，避免 cutoff 后追加改变 run identity。"""
    projected = frame.loc[:, list(columns)].reset_index(drop=True)
    digest = hashlib.sha256(canonical_json({"columns": list(columns), "dtypes": projected.dtypes.astype(str).tolist()}))
    digest.update(pd.util.hash_pandas_object(projected, index=False, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def load_protocol(
    path: Path, start_date: str | None, cutoff: str | None, frequency: str
) -> tuple[PilotConfig, CostModel, dict]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id",
        "start_date",
        "common_cutoff",
        "frequency",
        "holdings",
        "initial_capital_cny",
        "min_history_bars",
        "min_valid_bars_60",
        "min_adv20_thousand_cny",
        "liquidity_buckets",
        "winsor_limits",
        "random_seed",
        "allowed_chan_regimes",
        "cash_buffer",
        "cost_model",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing keys: {sorted(missing)}")
    frozen_contract = {
        "execution": "decision_close_t_to_open_t_plus_1",
        "frequency": "weekly_last_trading_day",
        "timing_policy": "entry_veto_after_factor_topn_no_refill",
        "state_missing_policy": "deny_new_entry",
    }
    for key, expected in frozen_contract.items():
        if protocol.get(key) != expected:
            raise ValueError(f"unsupported protocol {key}: {protocol.get(key)!r}; expected {expected!r}")
    if protocol.get("state_columns_whitelist") != list(STATE_COLUMNS):
        raise ValueError("state_columns_whitelist must exactly match the engine projection")
    if protocol.get("factor_weights") != {"lowvol_60": 0.5, "mom_120_20": 0.5}:
        raise ValueError("Pilot V1 only supports the frozen equal-weight factor blend")
    if protocol.get("cost_stress_multipliers") != [1.0, 2.0]:
        raise ValueError("Pilot V1 requires the frozen 1x and 2x cost stresses")

    frozen_cutoff = pd.Timestamp(protocol["common_cutoff"])
    selected_cutoff = pd.Timestamp(cutoff) if cutoff else frozen_cutoff
    if selected_cutoff > frozen_cutoff:
        raise ValueError(f"cutoff {selected_cutoff.date()} exceeds frozen cutoff {frozen_cutoff.date()}")
    config = PilotConfig(
        protocol_id=str(protocol["protocol_id"]),
        start_date=pd.Timestamp(start_date or protocol["start_date"]).strftime("%Y-%m-%d"),
        common_cutoff=selected_cutoff.strftime("%Y-%m-%d"),
        frequency=frequency,
        holdings=int(protocol["holdings"]),
        initial_capital_cny=float(protocol["initial_capital_cny"]),
        min_history_bars=int(protocol["min_history_bars"]),
        min_valid_bars_60=int(protocol["min_valid_bars_60"]),
        min_adv20_thousand_cny=float(protocol["min_adv20_thousand_cny"]),
        liquidity_buckets=int(protocol["liquidity_buckets"]),
        winsor_low=float(protocol["winsor_limits"][0]),
        winsor_high=float(protocol["winsor_limits"][1]),
        random_seed=int(protocol["random_seed"]),
        allowed_chan_regimes=tuple(map(int, protocol["allowed_chan_regimes"])),
        cash_buffer=float(protocol["cash_buffer"]),
        random_seed_count=int(protocol.get("random_seed_count", 20)),
    )
    raw_cost = protocol["cost_model"]
    cost = CostModel(
        commission_bps=float(raw_cost["commission_bps"]),
        transfer_bps=float(raw_cost["transfer_bps"]),
        min_commission_cny=float(raw_cost["min_commission_cny"]),
        stamp_before_bps=float(raw_cost["stamp_duty_bps_before_2023_08_28"]),
        stamp_after_bps=float(raw_cost["stamp_duty_bps_from_2023_08_28"]),
        base_slippage_bps=float(raw_cost["base_slippage_bps"]),
        impact_bps_at_1pct=float(raw_cost["impact_bps_at_1pct"]),
        max_adv_participation=float(raw_cost["max_adv_participation"]),
    )
    return config, cost, protocol


def resolve_calendar_path(data_dir: Path) -> Path:
    preferred = data_dir / "000001.SZ.parquet"
    path = preferred if preferred.exists() else next(iter(sorted(data_dir.glob("*.parquet"))), None)
    if path is None:
        raise FileNotFoundError(f"no parquet files in {data_dir}")
    return path


def load_calendar(data_dir: Path) -> pd.DatetimeIndex:
    """用长期连续的主板股票建立交易日历，避免按任意回测起点每五日取样。"""
    path = resolve_calendar_path(data_dir)
    dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
    return pd.DatetimeIndex(pd.to_datetime(dates.astype(str))).sort_values().unique()


def make_rebalance_schedule(
    calendar: Iterable[pd.Timestamp],
    start_date: str | pd.Timestamp,
    cutoff: str | pd.Timestamp,
    frequency: str = "weekly",
) -> pd.DataFrame:
    """返回严格的 ``decision_dt -> 下一交易日 exec_dt`` 时间表。"""
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    if len(dates) < 2:
        return pd.DataFrame(columns=["decision_dt", "exec_dt"])
    current, following = dates[:-1], dates[1:]
    if frequency == "weekly":
        is_decision = current.to_period("W-FRI") != following.to_period("W-FRI")
    elif frequency == "daily":
        is_decision = np.ones(len(current), dtype=bool)
    else:
        raise ValueError(f"unsupported frequency: {frequency}")
    start, end = pd.Timestamp(start_date), pd.Timestamp(cutoff)
    mask = is_decision & (current >= start) & (current <= end) & (following <= end)
    return pd.DataFrame({"decision_dt": current[mask], "exec_dt": following[mask]}).reset_index(drop=True)


def load_st_intervals(path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """读取名称变更表中的 PIT ST 区间；缺文件时 fail closed。"""
    if not path.exists():
        raise FileNotFoundError(f"PIT namechange file is required: {path}")
    frame = pd.read_parquet(path, columns=["ts_code", "name", "start_date", "end_date"])
    frame = frame[frame["name"].astype(str).str.contains("ST", case=False, na=False)].copy()
    frame["start"] = pd.to_datetime(frame["start_date"].astype(str), errors="coerce")
    frame["end"] = pd.to_datetime(frame["end_date"].astype(str), errors="coerce").fillna(pd.Timestamp.max.normalize())
    frame = frame.dropna(subset=["start"])
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for symbol, group in frame.groupby("ts_code", sort=False):
        out[str(symbol)] = list(zip(group["start"], group["end"], strict=False))
    return out


def st_mask(
    symbol: str, dates: pd.Series, intervals: Mapping[str, Sequence[tuple[pd.Timestamp, pd.Timestamp]]]
) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    values = pd.DatetimeIndex(dates)
    for start, end in intervals.get(symbol, ()):
        result |= (values >= start) & (values <= end)
    return result


def load_chan_projection(path: Path, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """物理上只读取白名单三列，未来标签不会进入选股调用链。"""
    if not path.exists():
        raise FileNotFoundError(f"Chan cache not found: {path}")
    frame = pd.read_parquet(path, columns=list(STATE_COLUMNS))
    if tuple(frame.columns) != STATE_COLUMNS:
        raise ValueError(f"unexpected projected state columns: {tuple(frame.columns)}")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["dt"] = pd.to_datetime(frame["dt"])
    frame["regime"] = pd.to_numeric(frame["regime"], errors="raise").astype("int8")
    frame = frame[frame["dt"] <= pd.Timestamp(cutoff)]
    return frame.drop_duplicates(["symbol", "dt"], keep="last").reset_index(drop=True)


def load_frozen_universe(path: Path) -> list[str]:
    """读取冻结污染 panel 的 symbol 清单；状态值本身不得决定研究宇宙。"""
    if not path.exists():
        raise FileNotFoundError(f"frozen pilot universe panel not found: {path}")
    frame = pd.read_parquet(path, columns=["symbol"])
    symbols = sorted(
        symbol for symbol in frame["symbol"].dropna().astype(str).unique() if not symbol.startswith(EXCLUDE_PREFIXES)
    )
    if not symbols:
        raise RuntimeError(f"empty frozen universe: {path}")
    return symbols


def compute_symbol_features(
    raw: pd.DataFrame,
    symbol: str,
    intervals: Mapping[str, Sequence[tuple[pd.Timestamp, pd.Timestamp]]],
    config: PilotConfig,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """只使用当前及历史行的因果滚动特征。"""
    frame = raw.rename(columns={"ts_code": "symbol", "trade_date": "dt"}).copy()
    frame["symbol"] = symbol
    frame["dt"] = pd.to_datetime(frame["dt"].astype(str))
    frame = frame.sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)
    for col in ("open", "close", "pre_close", "vol", "amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    frame["observed_bar"] = True
    if calendar is not None and not frame.empty:
        first_dt, last_dt = frame["dt"].min(), frame["dt"].max()
        sessions = pd.DatetimeIndex(calendar)
        sessions = sessions[(sessions >= first_dt) & (sessions <= last_dt)]
        frame = frame.set_index("dt").reindex(sessions).rename_axis("dt").reset_index()
        frame["symbol"] = symbol
        frame["observed_bar"] = frame["observed_bar"].eq(True)
        frame[["vol", "amount"]] = frame[["vol", "amount"]].fillna(0.0)
        # 停牌日按零收益持有计量；open/pre_close 保持 NaN，执行层仍会判定不可成交。
        frame["close"] = frame["close"].ffill()

    close = frame["close"].where(frame["close"] > 0)
    log_close = np.log(close)
    log_ret = log_close.diff()
    valid = frame["observed_bar"] & (frame["amount"] > 0) & close.notna()
    frame["bar_no"] = np.arange(1, len(frame) + 1, dtype=np.int32)
    frame["mom_120_20"] = log_close.shift(20) - log_close.shift(120)
    frame["lowvol_60"] = -log_ret.rolling(60, min_periods=60).std()
    frame["adv20"] = frame["amount"].rolling(20, min_periods=20).mean()
    frame["valid60"] = valid.rolling(60, min_periods=60).sum()
    frame["is_st"] = st_mask(symbol, frame["dt"], intervals)
    board_ok = not symbol.startswith(EXCLUDE_PREFIXES)
    frame["universe_valid"] = (
        board_ok
        & (frame["bar_no"] >= config.min_history_bars)
        & (frame["valid60"] >= config.min_valid_bars_60)
        & (frame["adv20"] >= config.min_adv20_thousand_cny)
        & (~frame["is_st"])
        & frame["observed_bar"]
        & frame["mom_120_20"].notna()
        & frame["lowvol_60"].notna()
    )
    return frame


def load_pilot_data(
    data_dir: Path,
    symbols: Sequence[str],
    signal_dates: pd.DatetimeIndex,
    intervals: Mapping[str, Sequence[tuple[pd.Timestamp, pd.Timestamp]]],
    config: PilotConfig,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    """逐股票读取，返回每日撮合行情、决策日特征和审计 manifest。"""
    price_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    signal_set = set(pd.DatetimeIndex(signal_dates))
    cutoff = pd.Timestamp(config.common_cutoff)
    t0 = time.time()

    for i, symbol in enumerate(symbols, 1):
        path = data_dir / f"{symbol}.parquet"
        if not path.exists():
            missing.append(symbol)
            continue
        try:
            raw = pd.read_parquet(path, columns=list(PRICE_COLUMNS))
        except Exception as exc:
            print(f"[WARN] {path.name}: {exc}", file=sys.stderr)
            missing.append(symbol)
            continue
        frame = compute_symbol_features(raw, symbol, intervals, config, calendar=calendar)
        frame = frame[frame["dt"] <= cutoff]
        if frame.empty:
            missing.append(symbol)
            continue
        manifest.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "used_prefix_sha256": projected_frame_sha256(
                    frame, ["symbol", "dt", "open", "close", "pre_close", "vol", "amount"]
                ),
                "size": path.stat().st_size,
                "rows": len(frame),
                "min_dt": frame["dt"].min(),
                "max_dt": frame["dt"].max(),
            }
        )
        price_parts.append(frame[["symbol", "dt", "open", "close", "pre_close", "is_st"]].copy())
        selected = frame[frame["dt"].isin(signal_set) & frame["universe_valid"]]
        if not selected.empty:
            feature_parts.append(
                selected[["symbol", "dt", "mom_120_20", "lowvol_60", "adv20", "bar_no", "valid60"]].copy()
            )
        if i % 500 == 0 or i == len(symbols):
            print(f"[LOAD] {i}/{len(symbols)} symbols | elapsed={time.time() - t0:.1f}s")

    if not price_parts:
        raise RuntimeError("no usable price data")
    prices = pd.concat(price_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    manifest.sort(key=lambda item: item["path"])
    return prices, features, manifest, missing


def rank_cross_section(features: pd.DataFrame, buckets: int, q_low: float, q_high: float) -> pd.DataFrame:
    """在每个日期、ADV 分层内 winsorize 后做两个因子等权百分位排名。"""
    required = {"symbol", "dt", "mom_120_20", "lowvol_60", "adv20"}
    if missing := required - set(features):
        raise ValueError(f"feature columns missing: {sorted(missing)}")
    parts: list[pd.DataFrame] = []
    for dt, group in features.groupby("dt", sort=True):
        g = group.dropna(subset=["mom_120_20", "lowvol_60", "adv20"]).copy()
        if g.empty:
            continue
        n_buckets = min(buckets, len(g))
        if n_buckets <= 1:
            g["adv_bucket"] = 0
        else:
            g["adv_bucket"] = pd.qcut(g["adv20"].rank(method="first"), n_buckets, labels=False).astype("int8")
        for factor in ("mom_120_20", "lowvol_60"):
            winsor = g.groupby("adv_bucket", observed=True)[factor].transform(
                lambda s: s.clip(s.quantile(q_low), s.quantile(q_high))
            )
            g[f"{factor}_rank"] = winsor.groupby(g["adv_bucket"]).rank(method="average", pct=True)
        g["factor_score"] = 0.5 * g["mom_120_20_rank"] + 0.5 * g["lowvol_60_rank"]
        g = g.sort_values(["factor_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
        g["factor_rank"] = np.arange(1, len(g) + 1, dtype=np.int32)
        g["dt"] = pd.Timestamp(dt)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_chan_gate(ranked: pd.DataFrame, states: pd.DataFrame, allowed: Sequence[int]) -> pd.DataFrame:
    projected = states.loc[:, list(STATE_COLUMNS)].copy()
    out = ranked.merge(projected, how="left", left_on=["symbol", "dt"], right_on=["symbol", "dt"])
    out["gate_allowed"] = out["regime"].isin(set(map(int, allowed)))
    return out


def build_target_plans(
    ranked: pd.DataFrame,
    schedule: pd.DataFrame,
    holdings: int,
) -> dict[pd.Timestamp, TargetPlan]:
    exec_map = schedule.set_index("decision_dt")["exec_dt"].to_dict()
    plans: dict[pd.Timestamp, TargetPlan] = {}
    previous_symbols: set[str] = set()
    for decision_dt, group in ranked.groupby("dt", sort=True):
        decision = pd.Timestamp(decision_dt)
        if decision not in exec_map:
            continue
        top = group.nsmallest(holdings, "factor_rank").sort_values("factor_rank")
        symbols = tuple(top["symbol"].astype(str))
        adv_series = group.set_index("symbol")["adv20"].astype(float) * 1000.0
        needed_adv = [symbol for symbol in set(symbols) | previous_symbols if symbol in adv_series.index]
        plans[pd.Timestamp(exec_map[decision])] = TargetPlan(
            decision_dt=decision,
            exec_dt=pd.Timestamp(exec_map[decision]),
            symbols=symbols,
            # 旧持仓掉出 Top-N 时，卖出容量仍必须使用本决策日已知 ADV，不能变成 NaN 后永久挂单。
            adv_cny={symbol: float(adv_series.loc[symbol]) for symbol in needed_adv},
            gate_allowed=dict(zip(symbols, top["gate_allowed"].astype(bool), strict=False)),
        )
        previous_symbols = set(symbols)
    return plans


def build_liquidity_matched_random_plans(
    ranked: pd.DataFrame,
    schedule: pd.DataFrame,
    holdings: int,
    seed: int,
) -> dict[pd.Timestamp, TargetPlan]:
    """固定种子，匹配 F 的 ADV 分层数量和 Top-N 名单留存数。"""
    exec_map = schedule.set_index("decision_dt")["exec_dt"].to_dict()
    plans: dict[pd.Timestamp, TargetPlan] = {}
    previous_factor: set[str] = set()
    previous_random: set[str] = set()
    for decision_dt, group in ranked.groupby("dt", sort=True):
        decision = pd.Timestamp(decision_dt)
        if decision not in exec_map:
            continue
        top = group.nsmallest(holdings, "factor_rank")
        counts = top["adv_bucket"].value_counts().sort_index()
        rng = np.random.default_rng(seed + int(decision.strftime("%Y%m%d")))
        factor_now = set(top["symbol"].astype(str))
        target_retained = len(factor_now & previous_factor) if previous_factor else 0
        bucket_by_symbol = group.set_index("symbol")["adv_bucket"].to_dict()
        capacity = {int(bucket): int(count) for bucket, count in counts.items()}
        eligible_incumbents = [
            symbol
            for symbol in sorted(previous_random)
            if symbol in bucket_by_symbol and capacity.get(int(bucket_by_symbol[symbol]), 0) > 0
        ]
        rng.shuffle(eligible_incumbents)
        selected_symbols: list[str] = []
        selected_count = dict.fromkeys(capacity, 0)
        for symbol in eligible_incumbents:
            bucket = int(bucket_by_symbol[symbol])
            if len(selected_symbols) >= target_retained:
                break
            if selected_count[bucket] < capacity[bucket]:
                selected_symbols.append(symbol)
                selected_count[bucket] += 1

        for bucket, count in counts.items():
            pool = group[group["adv_bucket"] == bucket].sort_values("symbol")
            pool_symbols = [symbol for symbol in pool["symbol"].astype(str) if symbol not in selected_symbols]
            take = min(int(count) - selected_count[int(bucket)], len(pool_symbols))
            if take > 0:
                loc = np.sort(rng.choice(len(pool_symbols), size=take, replace=False))
                selected_symbols.extend(pool_symbols[i] for i in loc)
        selected_symbols = selected_symbols[:holdings]
        selected = group.set_index("symbol").loc[selected_symbols].reset_index() if selected_symbols else group.head(0)
        symbols = tuple(selected["symbol"].astype(str))
        adv_series = group.set_index("symbol")["adv20"].astype(float) * 1000.0
        needed_adv = [symbol for symbol in set(symbols) | previous_random if symbol in adv_series.index]
        plans[pd.Timestamp(exec_map[decision])] = TargetPlan(
            decision_dt=decision,
            exec_dt=pd.Timestamp(exec_map[decision]),
            symbols=symbols,
            adv_cny={symbol: float(adv_series.loc[symbol]) for symbol in needed_adv},
            gate_allowed=dict.fromkeys(symbols, True),
        )
        previous_factor = factor_now
        previous_random = set(symbols)
    return plans


def apply_entry_gate(
    base_symbols: Sequence[str],
    held_symbols: Iterable[str],
    gate_allowed: Mapping[str, bool],
) -> tuple[str, ...]:
    """只否决新仓、不回填；已有且仍在 Top-N 的股票允许续持。"""
    held = set(held_symbols)
    return tuple(symbol for symbol in base_symbols if symbol in held or bool(gate_allowed.get(symbol, False)))


def apply_random_target_count(
    base_symbols: Sequence[str],
    held_symbols: Iterable[str],
    target_count: int,
    seed: int,
) -> tuple[str, ...]:
    """随机选出精确目标数量；优先保留仍在 Top-N 的旧仓以降低无谓换手。"""
    quota = max(0, min(int(target_count), len(base_symbols)))
    held = set(held_symbols)
    held_top = {symbol for symbol in base_symbols if symbol in held}
    rng = np.random.default_rng(seed)
    if len(held_top) > quota:
        accepted = set(rng.choice(sorted(held_top), size=quota, replace=False))
    else:
        new_candidates = [symbol for symbol in base_symbols if symbol not in held_top]
        take = max(0, min(len(new_candidates), quota - len(held_top)))
        selected_new = set(rng.choice(new_candidates, size=take, replace=False)) if take else set()
        accepted = held_top | selected_new
    return tuple(symbol for symbol in base_symbols if symbol in accepted)


def _board_limit(symbol: str, is_st: bool) -> float:
    if is_st:
        return 0.048
    return 0.198 if symbol.startswith(CHINEXT_PREFIXES) else 0.098


def _fillability(market: MarketData, row: int, col: int, symbol: str, side: str) -> str:
    open_px = market.open[row, col]
    pre_close = market.pre_close[row, col]
    # 开盘撮合不能读取执行日收盘后才完整可知的全天成交额；缺 bar 会自然表现为 open=NaN。
    if not np.isfinite(open_px) or open_px <= 0:
        return "suspended"
    if not np.isfinite(pre_close) or pre_close <= 0:
        return "missing_pre_close"
    gap = open_px / pre_close - 1.0
    limit = _board_limit(symbol, bool(market.is_st[row, col]))
    if side == "buy" and gap >= limit:
        return "limit_up"
    if side == "sell" and gap <= -limit:
        return "limit_down"
    return "fillable"


def _execution_terms(
    side: str,
    open_px: float,
    shares: int,
    adv_cny: float,
    dt: pd.Timestamp,
    cost: CostModel,
) -> dict[str, float]:
    open_notional = open_px * shares
    participation = open_notional / adv_cny if adv_cny > 0 else math.inf
    impact_bps = cost.impact_bps_at_1pct * math.sqrt(max(participation, 0.0) / 0.01)
    slippage_bps = (cost.base_slippage_bps + impact_bps) * cost.multiplier
    fill_px = open_px * (1.0 + slippage_bps / 10_000.0) if side == "buy" else open_px * (1.0 - slippage_bps / 10_000.0)
    notional = fill_px * shares
    commission = max(
        cost.min_commission_cny * cost.multiplier,
        notional * cost.commission_bps * cost.multiplier / 10_000.0,
    )
    transfer = notional * cost.transfer_bps * cost.multiplier / 10_000.0
    stamp = notional * cost.stamp_bps(dt) * cost.multiplier / 10_000.0 if side == "sell" else 0.0
    base_slippage = open_notional * cost.base_slippage_bps * cost.multiplier / 10_000.0
    impact_cost = open_notional * impact_bps * cost.multiplier / 10_000.0
    return {
        "fill_price": fill_px,
        "notional": notional,
        "commission": commission,
        "transfer": transfer,
        "stamp": stamp,
        "base_slippage": base_slippage,
        "impact": impact_cost,
        "participation": participation,
        "total_cost": commission + transfer + stamp + base_slippage + impact_cost,
    }


def _max_fill_shares(requested: int, open_px: float, adv_cny: float, cost: CostModel) -> int:
    if requested <= 0 or not np.isfinite(adv_cny) or adv_cny <= 0:
        return 0
    cap = math.floor((adv_cny * cost.max_adv_participation / open_px) / LOT_SIZE) * LOT_SIZE
    return max(0, min(requested, cap))


def _open_value(
    cash: float,
    holdings: Mapping[str, int],
    last_close: Mapping[str, float],
    market: MarketData,
    row: int,
) -> float:
    value = cash
    for symbol, shares in holdings.items():
        col = market.symbol_to_idx[symbol]
        px = market.open[row, col]
        if not np.isfinite(px) or px <= 0:
            px = last_close.get(symbol, np.nan)
        if np.isfinite(px):
            value += shares * px
    return value


def simulate_portfolio(
    market: MarketData,
    plans: Mapping[pd.Timestamp, TargetPlan],
    strategy: str,
    slots: int,
    initial_capital: float,
    cost: CostModel,
    entry_gate: bool = False,
    cash_buffer: float = 0.005,
    target_count_by_decision: Mapping[pd.Timestamp, int] | None = None,
    random_gate_seed: int | None = None,
) -> BacktestResult:
    """逐日持仓账本；非调仓日不会免费恢复等权。"""
    if not plans:
        raise ValueError("no target plans")
    plans = {pd.Timestamp(k): v for k, v in plans.items()}
    first_decision = min(plan.decision_dt for plan in plans.values())
    start_idx = int(market.dates.searchsorted(first_decision))
    end_idx = len(market.dates) - 1

    cash = float(initial_capital)
    holdings: dict[str, int] = {}
    last_close: dict[str, float] = {}
    last_adv: dict[str, float] = {}
    pending_sells: dict[str, PendingSell] = {}
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    intent_rows: list[dict[str, Any]] = []
    previous_nav: float | None = None

    for row in range(start_idx, end_idx + 1):
        dt = pd.Timestamp(market.dates[row])
        plan = plans.get(dt)
        daily_cost = 0.0
        daily_notional = 0.0
        blocked_buy = 0
        blocked_sell = 0
        desired_shares: dict[str, int] = {}
        accepted_for_plan: tuple[str, ...] = ()

        if plan is not None:
            last_adv.update({symbol: float(value) for symbol, value in plan.adv_cny.items() if np.isfinite(value)})
            if target_count_by_decision is not None:
                quota = int(target_count_by_decision.get(plan.decision_dt, 0))
                accepted = apply_random_target_count(
                    plan.symbols,
                    holdings,
                    quota,
                    (random_gate_seed or 0) + int(plan.decision_dt.strftime("%Y%m%d")),
                )
            else:
                accepted = (
                    apply_entry_gate(plan.symbols, holdings, plan.gate_allowed) if entry_gate else tuple(plan.symbols)
                )
            accepted_for_plan = accepted
            accepted_set = set(accepted)
            nav_open = _open_value(cash, holdings, last_close, market, row)
            slot_notional = nav_open * (1.0 - cash_buffer) / slots
            for symbol in accepted:
                col = market.symbol_to_idx.get(symbol)
                if col is None:
                    continue
                open_px = market.open[row, col]
                if np.isfinite(open_px) and open_px > 0:
                    desired_shares[symbol] = math.floor((slot_notional / open_px) / LOT_SIZE) * LOT_SIZE
                elif symbol in holdings:
                    desired_shares[symbol] = holdings[symbol]

            for symbol, current_shares in list(holdings.items()):
                target = desired_shares.get(symbol, 0)
                if current_shares > target:
                    pending_sells[symbol] = PendingSell(
                        decision_dt=plan.decision_dt,
                        target_shares=target,
                        adv_cny=float(plan.adv_cny.get(symbol, last_adv.get(symbol, math.nan))),
                    )
                else:
                    pending_sells.pop(symbol, None)

            for symbol in plan.symbols:
                was_held = symbol in holdings
                gate_ok = bool(plan.gate_allowed.get(symbol, False))
                intent_rows.append(
                    {
                        "strategy": strategy,
                        "decision_dt": plan.decision_dt,
                        "exec_dt": dt,
                        "symbol": symbol,
                        "was_held": was_held,
                        "gate_allowed": gate_ok,
                        "accepted": symbol in accepted_set,
                        "reason": (
                            "held_topn"
                            if was_held
                            else (
                                "random_quota_pass"
                                if target_count_by_decision is not None and symbol in accepted_set
                                else (
                                    "random_quota_veto"
                                    if target_count_by_decision is not None
                                    else ("gate_pass" if gate_ok or not entry_gate else "gate_veto")
                                )
                            )
                        ),
                    }
                )

        # 卖单逐日重试；先卖后买，卖不出时不会凭空给新仓融资。
        for symbol, pending in list(pending_sells.items()):
            current = holdings.get(symbol, 0)
            requested = current - pending.target_shares
            if requested <= 0:
                pending_sells.pop(symbol, None)
                continue
            col = market.symbol_to_idx[symbol]
            status = _fillability(market, row, col, symbol, "sell")
            if status != "fillable":
                blocked_sell += 1
                trade_rows.append(
                    {
                        "strategy": strategy,
                        "decision_dt": pending.decision_dt,
                        "exec_dt": dt,
                        "symbol": symbol,
                        "side": "sell",
                        "status": status,
                        "requested_shares": requested,
                        "filled_shares": 0,
                    }
                )
                continue
            open_px = float(market.open[row, col])
            fill_shares = _max_fill_shares(requested, open_px, pending.adv_cny, cost)
            if fill_shares <= 0:
                blocked_sell += 1
                status = "participation_cap"
                trade_rows.append(
                    {
                        "strategy": strategy,
                        "decision_dt": pending.decision_dt,
                        "exec_dt": dt,
                        "symbol": symbol,
                        "side": "sell",
                        "status": status,
                        "requested_shares": requested,
                        "filled_shares": 0,
                    }
                )
                continue
            terms = _execution_terms("sell", open_px, fill_shares, pending.adv_cny, dt, cost)
            cash += terms["notional"] - terms["commission"] - terms["transfer"] - terms["stamp"]
            holdings[symbol] = current - fill_shares
            if holdings[symbol] <= 0:
                holdings.pop(symbol, None)
                last_close.pop(symbol, None)
            if holdings.get(symbol, 0) <= pending.target_shares:
                pending_sells.pop(symbol, None)
            daily_cost += terms["total_cost"]
            daily_notional += terms["notional"]
            trade_rows.append(
                {
                    "strategy": strategy,
                    "decision_dt": pending.decision_dt,
                    "exec_dt": dt,
                    "symbol": symbol,
                    "side": "sell",
                    "status": "filled" if fill_shares == requested else "partial",
                    "requested_shares": requested,
                    "filled_shares": fill_shares,
                    "open_price": open_px,
                    **terms,
                }
            )

        if plan is not None:
            for symbol in accepted_for_plan:
                target = desired_shares.get(symbol, 0)
                requested = target - holdings.get(symbol, 0)
                if requested <= 0:
                    continue
                if symbol not in holdings and len(holdings) >= slots:
                    blocked_buy += 1
                    trade_rows.append(
                        {
                            "strategy": strategy,
                            "decision_dt": plan.decision_dt,
                            "exec_dt": dt,
                            "symbol": symbol,
                            "side": "buy",
                            "status": "holding_slot_cap",
                            "requested_shares": requested,
                            "filled_shares": 0,
                        }
                    )
                    continue
                col = market.symbol_to_idx[symbol]
                status = _fillability(market, row, col, symbol, "buy")
                if status != "fillable":
                    blocked_buy += 1
                    trade_rows.append(
                        {
                            "strategy": strategy,
                            "decision_dt": plan.decision_dt,
                            "exec_dt": dt,
                            "symbol": symbol,
                            "side": "buy",
                            "status": status,
                            "requested_shares": requested,
                            "filled_shares": 0,
                        }
                    )
                    continue
                open_px = float(market.open[row, col])
                adv_cny = float(plan.adv_cny.get(symbol, math.nan))
                fill_shares = _max_fill_shares(requested, open_px, adv_cny, cost)
                if fill_shares <= 0:
                    blocked_buy += 1
                    continue
                terms = _execution_terms("buy", open_px, fill_shares, adv_cny, dt, cost)
                total_cash = terms["notional"] + terms["commission"] + terms["transfer"]
                if total_cash > cash:
                    affordable = math.floor((cash / max(terms["fill_price"], 1e-12)) / LOT_SIZE) * LOT_SIZE
                    fill_shares = min(fill_shares, affordable)
                    if fill_shares <= 0:
                        blocked_buy += 1
                        continue
                    terms = _execution_terms("buy", open_px, fill_shares, adv_cny, dt, cost)
                    total_cash = terms["notional"] + terms["commission"] + terms["transfer"]
                    while fill_shares > 0 and total_cash > cash:
                        fill_shares -= LOT_SIZE
                        if fill_shares > 0:
                            terms = _execution_terms("buy", open_px, fill_shares, adv_cny, dt, cost)
                            total_cash = terms["notional"] + terms["commission"] + terms["transfer"]
                    if fill_shares <= 0:
                        blocked_buy += 1
                        continue
                cash -= total_cash
                holdings[symbol] = holdings.get(symbol, 0) + fill_shares
                last_close.setdefault(symbol, open_px)
                daily_cost += terms["total_cost"]
                daily_notional += terms["notional"]
                trade_rows.append(
                    {
                        "strategy": strategy,
                        "decision_dt": plan.decision_dt,
                        "exec_dt": dt,
                        "symbol": symbol,
                        "side": "buy",
                        "status": "filled" if fill_shares == requested else "partial",
                        "requested_shares": requested,
                        "filled_shares": fill_shares,
                        "open_price": open_px,
                        **terms,
                    }
                )

        position_value = 0.0
        marked: dict[str, float] = {}
        for symbol, shares in holdings.items():
            col = market.symbol_to_idx[symbol]
            close_px = market.close[row, col]
            if np.isfinite(close_px) and close_px > 0:
                last_close[symbol] = float(close_px)
            px = last_close.get(symbol, np.nan)
            if np.isfinite(px):
                marked[symbol] = shares * px
                position_value += marked[symbol]
        nav = cash + position_value
        daily_return = 0.0 if previous_nav is None else nav / previous_nav - 1.0
        equity_rows.append(
            {
                "strategy": strategy,
                "dt": dt,
                "nav": nav,
                "return": daily_return,
                "cash": cash,
                "cash_weight": cash / nav if nav else math.nan,
                "exposure": position_value / nav if nav else math.nan,
                "n_holdings": len(holdings),
                "daily_cost": daily_cost,
                "daily_notional": daily_notional,
                "blocked_buy": blocked_buy,
                "blocked_sell": blocked_sell,
                "pending_sells": len(pending_sells),
            }
        )
        for symbol, value in marked.items():
            holding_rows.append(
                {
                    "strategy": strategy,
                    "dt": dt,
                    "symbol": symbol,
                    "shares": holdings[symbol],
                    "value": value,
                    "weight": value / nav if nav else math.nan,
                }
            )
        if cash < -1e-6 or not np.isfinite(nav):
            raise AssertionError(f"portfolio conservation failed on {dt.date()}: cash={cash}, nav={nav}")
        previous_nav = nav

    return BacktestResult(
        equity=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
        holdings=pd.DataFrame(holding_rows),
        intents=pd.DataFrame(intent_rows),
    )


def aggregate_random_backtests(
    results: Sequence[BacktestResult],
    strategy: str,
    initial_capital: float,
) -> BacktestResult:
    """按每日净收益平均固定随机种子，避免事后挑选某个随机基准。"""
    if not results:
        raise ValueError("no random backtests to aggregate")
    columns = [
        "return",
        "cash_weight",
        "exposure",
        "n_holdings",
        "daily_cost",
        "daily_notional",
        "blocked_buy",
        "blocked_sell",
        "pending_sells",
    ]
    indexed = [result.equity.set_index("dt")[columns].astype(float) for result in results]
    common = indexed[0].index
    for frame in indexed[1:]:
        common = common.intersection(frame.index)
    cube = np.stack([frame.loc[common].to_numpy() for frame in indexed], axis=0)
    mean = pd.DataFrame(cube.mean(axis=0), index=common, columns=columns)
    mean["nav"] = initial_capital * (1.0 + mean["return"]).cumprod()
    mean["cash"] = mean["nav"] * mean["cash_weight"]
    mean["strategy"] = strategy
    mean = mean.reset_index().rename(columns={"index": "dt"})
    ordered = [
        "strategy",
        "dt",
        "nav",
        "return",
        "cash",
        "cash_weight",
        "exposure",
        "n_holdings",
        "daily_cost",
        "daily_notional",
        "blocked_buy",
        "blocked_sell",
        "pending_sells",
    ]
    return BacktestResult(
        equity=mean[ordered],
        trades=pd.DataFrame(),
        holdings=pd.DataFrame(),
        intents=pd.DataFrame(),
    )


def portfolio_metrics(result: BacktestResult) -> dict[str, Any]:
    equity = result.equity.copy()
    returns = equity["return"].astype(float)
    n_days = max(len(equity) - 1, 1)
    years = n_days / 242.0
    total_return = equity["nav"].iloc[-1] / equity["nav"].iloc[0] - 1.0
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    annual_vol = returns.std(ddof=1) * math.sqrt(242) if len(returns) > 1 else math.nan
    sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(242) if returns.std(ddof=1) > 0 else math.nan
    drawdown = equity["nav"] / equity["nav"].cummax() - 1.0
    trades = result.trades
    filled = trades[trades.get("filled_shares", pd.Series(dtype=float)).fillna(0) > 0] if not trades.empty else trades

    def trade_sum(column: str) -> float:
        return float(pd.to_numeric(filled[column], errors="coerce").fillna(0).sum()) if column in filled else 0.0

    participation = (
        pd.to_numeric(filled["participation"], errors="coerce").dropna()
        if "participation" in filled
        else pd.Series(dtype=float)
    )
    return {
        "start": equity["dt"].iloc[0],
        "end": equity["dt"].iloc[-1],
        "trading_days": len(equity),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_zero_rf": sharpe,
        "max_drawdown": float(drawdown.min()),
        "average_exposure": float(equity["exposure"].mean()),
        "average_holdings": float(equity["n_holdings"].mean()),
        "max_holdings": int(equity["n_holdings"].max()),
        "total_cost_cny": float(equity["daily_cost"].sum()),
        "commission_cny": trade_sum("commission"),
        "transfer_cny": trade_sum("transfer"),
        "stamp_duty_cny": trade_sum("stamp"),
        "base_slippage_cny": trade_sum("base_slippage"),
        "impact_cny": trade_sum("impact"),
        "participation_p95": float(participation.quantile(0.95)) if len(participation) else math.nan,
        "participation_p99": float(participation.quantile(0.99)) if len(participation) else math.nan,
        "participation_max": float(participation.max()) if len(participation) else math.nan,
        "one_way_turnover": float(equity["daily_notional"].sum() / (2.0 * equity["nav"].mean())),
        "blocked_buy_attempts": int(equity["blocked_buy"].sum()),
        "blocked_sell_attempts": int(equity["blocked_sell"].sum()),
        "filled_trade_rows": int(len(filled)),
    }


def _trimmed_mean(values: pd.Series, proportion: float = 0.1) -> float:
    clean = np.sort(values.dropna().to_numpy(dtype=float))
    if not len(clean):
        return math.nan
    cut = int(len(clean) * proportion)
    core = clean[cut : len(clean) - cut] if cut and len(clean) > 2 * cut else clean
    return float(np.mean(core))


def _newey_west_mean_t(values: pd.Series, max_lag: int = 4) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n <= max_lag + 1:
        return math.nan
    centered = clean - clean.mean()
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, max_lag + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    return float(clean.mean() / standard_error) if standard_error > 0 else math.nan


def _block_bootstrap_annualized_mean_ci(
    values: pd.Series,
    block_length: int = 4,
    samples: int = 5_000,
    seed: int = 20260715,
) -> list[float]:
    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < block_length:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block_length)
    means = np.empty(samples, dtype=float)
    offsets = np.arange(block_length)
    for i in range(samples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
        means[i] = clean[indices].mean() * 52.0
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def active_metrics(left: BacktestResult, right: BacktestResult) -> dict[str, Any]:
    left_returns = left.equity.set_index("dt")["return"].astype(float)
    right_returns = right.equity.set_index("dt")["return"].astype(float)
    aligned = pd.concat([left_returns.rename("left"), right_returns.rename("right")], axis=1).dropna()
    active_daily = aligned["left"] - aligned["right"]
    weekly = (1.0 + aligned).resample("W-FRI").prod() - 1.0
    active_weekly = weekly["left"] - weekly["right"]
    ir = active_weekly.mean() / active_weekly.std(ddof=1) * math.sqrt(52) if active_weekly.std(ddof=1) > 0 else math.nan
    positive = active_weekly[active_weekly > 0].sort_values(ascending=False)
    top10_share = float(positive.head(10).sum() / positive.sum()) if positive.sum() > 0 else math.nan
    yearly_left = aligned["left"].groupby(aligned.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
    yearly_right = aligned["right"].groupby(aligned.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
    yearly = yearly_left - yearly_right
    return {
        "annualized_active_arithmetic": float(active_daily.mean() * 242),
        "information_ratio_weekly": float(ir),
        "hac_t_weekly_mean_lag4": _newey_west_mean_t(active_weekly, max_lag=4),
        "block_bootstrap_annualized_active_95pct": _block_bootstrap_annualized_mean_ci(active_weekly),
        "weekly_active_median": float(active_weekly.median()),
        "weekly_active_trimmed_mean_10pct": _trimmed_mean(active_weekly),
        "top10_positive_weeks_share": top10_share,
        "positive_years": int((yearly > 0).sum()),
        "year_count": int(len(yearly)),
        "yearly_active": {str(int(year)): float(value) for year, value in yearly.items()},
    }


def factor_ic_table(ranked: pd.DataFrame, schedule: pd.DataFrame, market: MarketData) -> pd.DataFrame:
    """标签在排名完成后单独生成，绝不参与可选池。"""
    schedule = schedule.sort_values("decision_dt").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i in range(len(schedule) - 1):
        decision = pd.Timestamp(schedule.loc[i, "decision_dt"])
        exec_dt = pd.Timestamp(schedule.loc[i, "exec_dt"])
        next_exec = pd.Timestamp(schedule.loc[i + 1, "exec_dt"])
        if exec_dt not in market.date_to_idx or next_exec not in market.date_to_idx:
            continue
        group = ranked[ranked["dt"] == decision].copy()
        if len(group) < 20:
            continue
        cols = group["symbol"].map(market.symbol_to_idx)
        valid_col = cols.notna()
        group = group[valid_col].copy()
        idx = cols[valid_col].astype(int).to_numpy()
        open0 = market.open[market.date_to_idx[exec_dt], idx]
        open1 = market.open[market.date_to_idx[next_exec], idx]
        fwd = np.full(len(open0), np.nan, dtype=float)
        valid_price = np.isfinite(open0) & (open0 > 0) & np.isfinite(open1) & (open1 > 0)
        fwd[valid_price] = open1[valid_price] / open0[valid_price] - 1.0
        group["fwd_open_to_open"] = fwd
        valid = np.isfinite(fwd)
        frozen_top20 = group.nsmallest(20, "factor_rank")
        top20_valid = frozen_top20["fwd_open_to_open"].notna()
        labeled = group.loc[valid].copy()
        row_out = {
            "decision_dt": decision,
            "n_universe": int(len(group)),
            "n_labeled": int(valid.sum()),
            "label_coverage": float(valid.mean()),
            "n_missing_future_open": int((~valid).sum()),
            "score_rank_ic": labeled["factor_score"].corr(labeled["fwd_open_to_open"], method="spearman"),
            "momentum_rank_ic": labeled["mom_120_20"].corr(labeled["fwd_open_to_open"], method="spearman"),
            "lowvol_rank_ic": labeled["lowvol_60"].corr(labeled["fwd_open_to_open"], method="spearman"),
            "top20_equal_weight_return": frozen_top20.loc[top20_valid, "fwd_open_to_open"].mean(),
            "top20_label_coverage": float(top20_valid.mean()),
            "top20_missing_future_open": int((~top20_valid).sum()),
        }
        if len(labeled) >= 5:
            labeled["score_quintile"] = pd.qcut(labeled["factor_score"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            quintiles = labeled.groupby("score_quintile", observed=True)["fwd_open_to_open"].mean()
            row_out.update({f"score_q{int(q)}_return": float(value) for q, value in quintiles.items()})
        rows.append(row_out)
    return pd.DataFrame(rows)


def data_limitations() -> dict[str, Any]:
    limitations = {
        "pit_industry": False,
        "pit_float_mcap": False,
        "raw_unadjusted_execution_price": False,
        "full_0_to_10_chan_state_cache": False,
        "survivorship_clean_universe": False,
        "confirmatory_oos": False,
    }
    return {
        **limitations,
        "upgrade_allowed": all(limitations.values()),
        "permitted_conclusion": "engineering_pass_or_hypothesis_rejection_only",
        "research_label": RESEARCH_LABEL,
    }


def evaluate_pilot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """预声明的非对称证伪器；永远不会返回“alpha 已验证”。"""
    f2 = metrics["F_2x_minus_R_liq_2x"]
    fc2 = metrics["FC_2x_minus_R_liq_2x"]
    fc_increment = metrics["FC_minus_F"]
    chan_identity = metrics["FC_minus_FGR"]
    fc_mdd = abs(float(metrics["FC_1x"]["max_drawdown"]))
    f_mdd = abs(float(metrics["F_1x"]["max_drawdown"]))
    fgr_mdd = abs(float(metrics["FGR_1x"]["max_drawdown"]))

    checks = {
        "f_2x_active_not_positive": float(f2["annualized_active_arithmetic"]) <= 0,
        "fc_2x_active_not_positive": float(fc2["annualized_active_arithmetic"]) <= 0,
        "f_weekly_active_median_not_positive": float(metrics["F_minus_R_liq"]["weekly_active_median"]) <= 0,
        "fc_weekly_active_median_not_positive": float(metrics["FC_minus_R_liq"]["weekly_active_median"]) <= 0,
        "f_top10_positive_weeks_share_gt_50pct": float(metrics["F_minus_R_liq"]["top10_positive_weeks_share"]) > 0.50,
        "fc_top10_positive_weeks_share_gt_50pct": float(metrics["FC_minus_R_liq"]["top10_positive_weeks_share"]) > 0.50,
        "f_at_least_four_negative_years": (
            int(metrics["F_minus_R_liq"]["year_count"]) >= 4
            and int(metrics["F_minus_R_liq"]["positive_years"]) <= int(metrics["F_minus_R_liq"]["year_count"]) - 4
        ),
        "fc_at_least_four_negative_years": (
            int(metrics["FC_minus_R_liq"]["year_count"]) >= 4
            and int(metrics["FC_minus_R_liq"]["positive_years"]) <= int(metrics["FC_minus_R_liq"]["year_count"]) - 4
        ),
        "chan_neither_improves_ir_nor_drawdown": (
            float(fc_increment["information_ratio_weekly"]) <= 0 and fc_mdd >= f_mdd
        ),
        "chan_state_identity_neither_improves_ir_nor_drawdown_vs_random_gate": (
            float(chan_identity["information_ratio_weekly"]) <= 0 and fc_mdd >= fgr_mdd
        ),
    }
    rejected = [name for name, failed in checks.items() if bool(failed)]
    return {
        "status": "REJECT_OR_REWRITE" if rejected else "NOT_REJECTED_BUT_NOT_VALIDATED",
        "rejection_triggers": rejected,
        "checks": checks,
        "alpha_validated": False,
        "upgrade_allowed": False,
    }


def render_summary(config: PilotConfig, metrics: Mapping[str, Any], limitations: Mapping[str, Any]) -> str:
    lines = [
        f"# {RESEARCH_LABEL}: 横截面因子 × 缠论择时 Pilot",
        "",
        "> 该结果只能用于工程验收和证伪，不能证明 alpha、不能上线。",
        "",
        f"- 协议：`{config.protocol_id}`",
        f"- 区间：{config.start_date} 至 {config.common_cutoff}",
        f"- 调仓：{config.frequency}，Top {config.holdings}，次日开盘执行",
        f"- 正式升级：`{limitations['upgrade_allowed']}`",
        "",
        "## 组合描述统计",
        "",
        "| 组合 | 成本 | 年化 | 最大回撤 | Sharpe | 平均暴露 | 总成本(元) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "R_liq_gross",
        "F_gross",
        "FGR_gross",
        "FC_gross",
        "R_liq_1x",
        "F_1x",
        "FGR_1x",
        "FC_1x",
        "R_liq_2x",
        "F_2x",
        "FGR_2x",
        "FC_2x",
    ):
        item = metrics[key]
        lines.append(
            f"| {key.rsplit('_', 1)[0]} | {key.rsplit('_', 1)[1]} | {item['annual_return']:.2%} | "
            f"{item['max_drawdown']:.2%} | {item['sharpe_zero_rf']:.2f} | "
            f"{item['average_exposure']:.2%} | {item['total_cost_cny']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## 主动描述统计",
            "",
            "- F − R_liq：" + json.dumps(metrics["F_minus_R_liq"], ensure_ascii=False, default=json_default),
            "- FC − R_liq：" + json.dumps(metrics["FC_minus_R_liq"], ensure_ascii=False, default=json_default),
            "- FC − F（缠论门增量）：" + json.dumps(metrics["FC_minus_F"], ensure_ascii=False, default=json_default),
            "- FC − FGR（相同放行数量的随机门）："
            + json.dumps(metrics["FC_minus_FGR"], ensure_ascii=False, default=json_default),
            "",
            "完整限制见 `data_limitations.json`；输入指纹见 `source_manifest.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config, base_cost, protocol = load_protocol(PROTOCOL_PATH, args.start_date, args.cutoff, args.frequency)
    smoke = (
        args.max_symbols is not None
        or args.cutoff is not None
        or args.frequency != "weekly"
        or config.start_date != pd.Timestamp(protocol["start_date"]).strftime("%Y-%m-%d")
    )
    print(f"[{RESEARCH_LABEL}] protocol={config.protocol_id} | smoke={smoke}")

    states = load_chan_projection(args.chan_cache, config.common_cutoff)
    symbols = load_frozen_universe(args.universe_panel)
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    calendar_all = load_calendar(args.data_dir)
    schedule = make_rebalance_schedule(calendar_all, config.start_date, config.common_cutoff, config.frequency)
    if schedule.empty:
        raise RuntimeError("empty rebalance schedule")
    intervals = load_st_intervals(args.namechange_path)
    prices, features, price_manifest, missing = load_pilot_data(
        args.data_dir,
        symbols,
        pd.DatetimeIndex(schedule["decision_dt"]),
        intervals,
        config,
        calendar_all,
    )
    if features.empty:
        raise RuntimeError("no valid feature snapshots")

    ranked = rank_cross_section(features, config.liquidity_buckets, config.winsor_low, config.winsor_high)
    ranked = attach_chan_gate(ranked, states, config.allowed_chan_regimes)
    f_plans = build_target_plans(ranked, schedule, config.holdings)
    random_plan_sets = [
        (
            config.random_seed + offset,
            build_liquidity_matched_random_plans(
                ranked,
                schedule,
                config.holdings,
                config.random_seed + offset,
            ),
        )
        for offset in range(config.random_seed_count)
    ]
    if not f_plans or any(not plans for _, plans in random_plan_sets):
        raise RuntimeError("no executable target plans")

    calendar = calendar_all[
        (calendar_all >= min(plan.decision_dt for plan in f_plans.values()))
        & (calendar_all <= pd.Timestamp(config.common_cutoff))
    ]
    market = MarketData.from_long_frame(prices, calendar, symbols)
    results: dict[str, BacktestResult] = {}
    random_seed_metrics: dict[str, list[dict[str, Any]]] = {}
    gross_cost = CostModel(
        commission_bps=0.0,
        transfer_bps=0.0,
        min_commission_cny=0.0,
        stamp_before_bps=0.0,
        stamp_after_bps=0.0,
        base_slippage_bps=0.0,
        impact_bps_at_1pct=0.0,
        max_adv_participation=base_cost.max_adv_participation,
    )
    cost_specs = [
        ("gross", gross_cost),
        ("1x", CostModel(**{**asdict(base_cost), "multiplier": 1.0})),
        ("2x", CostModel(**{**asdict(base_cost), "multiplier": 2.0})),
    ]
    for suffix, cost in cost_specs:
        results[f"F_{suffix}"] = simulate_portfolio(
            market,
            f_plans,
            f"F_{suffix}",
            config.holdings,
            config.initial_capital_cny,
            cost,
            entry_gate=False,
            cash_buffer=config.cash_buffer,
        )
        results[f"FC_{suffix}"] = simulate_portfolio(
            market,
            f_plans,
            f"FC_{suffix}",
            config.holdings,
            config.initial_capital_cny,
            cost,
            entry_gate=True,
            cash_buffer=config.cash_buffer,
        )
        accepted_target_count = {
            pd.Timestamp(dt): int(count)
            for dt, count in results[f"FC_{suffix}"].intents.groupby("decision_dt")["accepted"].sum().items()
        }
        random_runs = [
            simulate_portfolio(
                market,
                plans,
                f"R_liq_seed_{seed}_{suffix}",
                config.holdings,
                config.initial_capital_cny,
                cost,
                entry_gate=False,
                cash_buffer=config.cash_buffer,
            )
            for seed, plans in random_plan_sets
        ]
        random_seed_metrics[suffix] = [
            {"seed": seed, **portfolio_metrics(result)}
            for (seed, _), result in zip(random_plan_sets, random_runs, strict=True)
        ]
        results[f"R_liq_{suffix}"] = aggregate_random_backtests(
            random_runs,
            f"R_liq_{suffix}",
            config.initial_capital_cny,
        )
        random_gate_runs = [
            simulate_portfolio(
                market,
                f_plans,
                f"FGR_seed_{seed}_{suffix}",
                config.holdings,
                config.initial_capital_cny,
                cost,
                entry_gate=True,
                cash_buffer=config.cash_buffer,
                target_count_by_decision=accepted_target_count,
                random_gate_seed=seed,
            )
            for seed in range(config.random_seed, config.random_seed + config.random_seed_count)
        ]
        random_seed_metrics[f"gate_{suffix}"] = [
            {"seed": seed, **portfolio_metrics(result)}
            for seed, result in zip(
                range(config.random_seed, config.random_seed + config.random_seed_count),
                random_gate_runs,
                strict=True,
            )
        ]
        results[f"FGR_{suffix}"] = aggregate_random_backtests(
            random_gate_runs,
            f"FGR_{suffix}",
            config.initial_capital_cny,
        )

    metrics = {name: portfolio_metrics(result) for name, result in results.items()}
    metrics["F_minus_R_liq"] = active_metrics(results["F_1x"], results["R_liq_1x"])
    metrics["FC_minus_R_liq"] = active_metrics(results["FC_1x"], results["R_liq_1x"])
    metrics["FC_minus_F"] = active_metrics(results["FC_1x"], results["F_1x"])
    metrics["FC_minus_FGR"] = active_metrics(results["FC_1x"], results["FGR_1x"])
    metrics["F_gross_minus_R_liq_gross"] = active_metrics(results["F_gross"], results["R_liq_gross"])
    metrics["FC_gross_minus_R_liq_gross"] = active_metrics(results["FC_gross"], results["R_liq_gross"])
    metrics["FC_gross_minus_FGR_gross"] = active_metrics(results["FC_gross"], results["FGR_gross"])
    metrics["F_2x_minus_R_liq_2x"] = active_metrics(results["F_2x"], results["R_liq_2x"])
    metrics["FC_2x_minus_R_liq_2x"] = active_metrics(results["FC_2x"], results["R_liq_2x"])
    metrics["FC_2x_minus_FGR_2x"] = active_metrics(results["FC_2x"], results["FGR_2x"])
    pilot_decision = evaluate_pilot(metrics)

    calendar_path = resolve_calendar_path(args.data_dir)
    research_inputs = {
        "engine_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "chan_projection_sha256": projected_frame_sha256(states, STATE_COLUMNS),
        "universe_symbols_sha256": hashlib.sha256(canonical_json(symbols)).hexdigest(),
        "calendar_dates_sha256": projected_frame_sha256(pd.DataFrame({"dt": calendar}), ["dt"]),
        "namechange_sha256": sha256_file(args.namechange_path),
        "price_prefixes": [
            {
                "path": item["path"],
                "used_prefix_sha256": item["used_prefix_sha256"],
                "rows": item["rows"],
                "min_dt": item["min_dt"],
                "max_dt": item["max_dt"],
            }
            for item in price_manifest
        ],
    }
    research_input_digest = hashlib.sha256(canonical_json(research_inputs)).hexdigest()
    source_manifest = {
        "research_label": RESEARCH_LABEL,
        "engine_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "chan_cache": {"path": str(args.chan_cache), "sha256": sha256_file(args.chan_cache)},
        "chan_projection_sha256": research_inputs["chan_projection_sha256"],
        "universe_panel": {
            "path": str(args.universe_panel),
            "sha256": sha256_file(args.universe_panel),
            "symbols_sha256": research_inputs["universe_symbols_sha256"],
        },
        "calendar_source": {
            "path": str(calendar_path),
            "sha256": sha256_file(calendar_path),
            "used_dates_sha256": research_inputs["calendar_dates_sha256"],
        },
        "namechange": {"path": str(args.namechange_path), "sha256": sha256_file(args.namechange_path)},
        "price_files": price_manifest,
        "missing_symbols": missing,
        "selected_symbol_count": len(symbols),
        "loaded_symbol_count": len(price_manifest),
        "common_cutoff": config.common_cutoff,
        "research_input_digest": research_input_digest,
        "prefix_hash_algorithm": "pandas_hash_pandas_object_v1_plus_schema_sha256",
    }
    identity = {
        "config": asdict(config),
        "protocol": protocol,
        "source_digest": research_input_digest,
        "smoke": smoke,
    }
    run_id = hashlib.sha256(canonical_json(identity)).hexdigest()[:16]
    prefix = "SMOKE_ONLY" if smoke else RESEARCH_LABEL
    run_dir = args.output_dir / f"{prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", identity)
    write_json(run_dir / "source_manifest.json", source_manifest)
    limitations = data_limitations()
    write_json(run_dir / "data_limitations.json", limitations)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "random_seed_metrics.json", random_seed_metrics)
    write_json(run_dir / "pilot_decision.json", pilot_decision)
    ranked[ranked["factor_rank"] <= config.holdings].to_parquet(run_dir / "topn_decisions.parquet", index=False)
    factor_ic_table(ranked, schedule, market).to_parquet(run_dir / "factor_ic.parquet", index=False)
    for name, result in results.items():
        result.equity.to_parquet(run_dir / f"equity_{name}.parquet", index=False)
        result.trades.to_parquet(run_dir / f"trades_{name}.parquet", index=False)
        result.holdings.to_parquet(run_dir / f"holdings_{name}.parquet", index=False)
        result.intents.to_parquet(run_dir / f"intents_{name}.parquet", index=False)
    (run_dir / "summary.md").write_text(render_summary(config, metrics, limitations), encoding="utf-8")
    print(f"[DONE] {run_dir}")
    print(json.dumps(metrics["FC_minus_F"], ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
