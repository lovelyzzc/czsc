"""主升浪策略实盘镜像组合回测 + 随机对照 beta 剥离

与 `surge_regime_backtest.py`（全部信号、等权、零 pair 成本）不同，本脚本回测的是
**每日选股 skill 实际执行的策略**：

1. 入选 = daily_scan 镜像：默认门控（量比/散度/中枢上方，anticipate 加 ret20）→
   硬过滤（信号日成交额≥1亿、止损带 8-20%、历史 ST/退市剔除、入场日开盘逼近涨停
   视为买不进）→ `trend_regime.priority_score`（freshness=0）排序；
2. 槽位组合：N 个等资金槽（N∈{10,20}），槽空闲时取当日优先级最高候选，单票单仓，
   权益曲线 = 槽权益之和 / N（闲置槽收益 0）。**不用 WeightBacktest**（规避稀疏组合
   年化外推假象），直接从日收益序列算年化/夏普/最大回撤；
3. 成本：买入 0.15%、卖出 0.25%（含印花税），pair 级报毛/净两套；
4. beta 剥离：每笔成交配同决策日、同成交额十分位的随机 K=50 只对照（同日历持有期、
   同次日开盘入），超额 = 策略毛收益 − 对照中位毛收益（毛对毛，隔离选股能力）。
   预声明判定：OOS 超额 ≤0 或 |t|<2 → 收益主体为规模/市场 beta。

依赖 `surge_candidates_dump.py` 的输出（candidates.parquet / panel.parquet）。

    uv run --no-sync python scripts/surge_portfolio_backtest.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trend_regime as tr

CAND_DIR = Path(__file__).resolve().parent / "_output" / "surge_candidates"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "surge_portfolio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

BUY_COST = 0.0015
SELL_COST = 0.0025
SLOT_COUNTS = [10, 20]
MODES = ["confirm", "anticipate"]
TRAIN_END = pd.Timestamp("2023-12-31")
CONTROL_K = 50
CONTROL_MIN_VALID = 10
RNG_SEED = 42

# 硬过滤阈值（与 daily_scan 一致）
MIN_AMOUNT_E = 1.0
STOP_MIN_PCT, STOP_MAX_PCT = 8.0, 20.0
GAP_LIMIT_MARGIN = 0.3  # 开盘跳空 ≥ 板涨停阈值-此值 → 视为一字/秒板买不进


# --------------------------------------------------------------------------- #
# 历史 ST 判定（namechange 区间）
# --------------------------------------------------------------------------- #
def _is_st_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "退" in text


def load_st_intervals() -> dict[str, list[tuple[str, str, bool]]]:
    """symbol → [(start_date, end_date, is_st), ...] 按 start 升序。无记录 = 从未改名 = 非 ST。"""
    if not NAMECHANGE_PATH.exists():
        print(f"[WARN] {NAMECHANGE_PATH} 不存在，跳过历史 ST 过滤（回测将偏乐观）")
        return {}
    nc = pd.read_parquet(NAMECHANGE_PATH)
    nc["end_date"] = nc["end_date"].fillna("99999999")
    out: dict[str, list] = {}
    for code, g in nc.groupby("ts_code"):
        g = g.sort_values("start_date")
        out[code] = list(zip(g["start_date"], g["end_date"], [_is_st_name(x) for x in g["name"]], strict=False))
    return out


def is_st_on(intervals: dict, symbol: str, date: pd.Timestamp) -> bool:
    rows = intervals.get(symbol)
    if not rows:
        return False
    d = date.strftime("%Y%m%d")
    best = None
    for start, _end, is_st in rows:
        if start <= d:
            best = is_st  # 取 start ≤ d 的最近一次改名
    return bool(best) if best is not None else False  # 首次改名前的名字未知，按非 ST 处理


# --------------------------------------------------------------------------- #
# 候选过滤（daily_scan 镜像）
# --------------------------------------------------------------------------- #
def gated_candidates(cand: pd.DataFrame, mode: str, st_intervals: dict) -> pd.DataFrame:
    df = cand[(cand["mode"] == mode) & (cand["delay"] == 0)].copy()
    # 默认门控（surge_onset 等价）
    g = (
        (df["sig_vol_ratio"] >= tr.SURGE_GATE_VOL_RATIO)
        & (df["sig_ma_spread_pct"] >= tr.SURGE_GATE_MA_SPREAD)
        & (df["sig_above_zg"] == 1)
    )
    if mode == "anticipate":
        g &= df["sig_ret20"] >= tr.SURGE_GATE_RET20
    df = df[g.fillna(False)]
    # 硬过滤
    df = df[
        (df["amount_e"] >= MIN_AMOUNT_E)
        & df["sl_pct"].between(STOP_MIN_PCT, STOP_MAX_PCT)
        & (df["gap_pct"] < df["limit_pct"] - GAP_LIMIT_MARGIN)
    ].copy()
    if st_intervals:
        st_mask = df.apply(lambda r: is_st_on(st_intervals, r["symbol"], r["dec_dt"]), axis=1)
        n_st = int(st_mask.sum())
        df = df[~st_mask]
        print(f"  [{mode}] ST/退市历史过滤剔除 {n_st} 笔")
    df["priority"] = [
        tr.priority_score(s, sl, 0, rg) for s, sl, rg in zip(df["score"], df["sl_pct"], df["dec_regime"], strict=False)
    ]
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - SELL_COST) / (1 + BUY_COST) - 1) * 100
    return df.sort_values(["entry_dt", "priority"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 槽位组合模拟
# --------------------------------------------------------------------------- #
def simulate_slots(df: pd.DataFrame, n_slots: int) -> tuple[pd.DataFrame, pd.Series]:
    """贪心槽位分配：返回 (成交明细, 日收益序列)。"""
    taken_rows = []
    open_until: dict[str, pd.Timestamp] = {}  # symbol → exit_dt（exit 当日仍占槽，次日释放）
    for entry_dt, day_grp in df.groupby("entry_dt", sort=True):
        open_until = {s: x for s, x in open_until.items() if x >= entry_dt}
        free = n_slots - len(open_until)
        if free <= 0:
            continue
        for _, r in day_grp.iterrows():
            if free <= 0:
                break
            if r["symbol"] in open_until:
                continue
            open_until[r["symbol"]] = r["exit_dt"]
            taken_rows.append(r)
            free -= 1
    trades = pd.DataFrame(taken_rows).reset_index(drop=True)
    return trades, _daily_returns(trades, n_slots)


def _daily_returns(trades: pd.DataFrame, n_slots: int) -> pd.Series:
    """由成交明细 + 价格面板生成组合日收益（闲置槽收益 0，满仓上限 n_slots）。"""
    syms = trades["symbol"].unique()
    panel = pd.read_parquet(CAND_DIR / "panel.parquet", columns=["symbol", "dt", "close"])
    panel = panel[panel["symbol"].isin(syms)]
    closes = {s: g.set_index("dt")["close"].sort_index() for s, g in panel.groupby("symbol")}

    acc: dict[pd.Timestamp, float] = {}
    cnt: dict[pd.Timestamp, int] = {}
    for r in trades.itertuples():
        c = closes[r.symbol].loc[r.entry_dt : r.exit_dt]
        dts, px = c.index, c.to_numpy(dtype=float)
        entry_eff = r.entry_price * (1 + BUY_COST)
        exit_eff = r.exit_price * (1 - SELL_COST)
        for k, dt in enumerate(dts):
            if len(dts) == 1:
                ret = exit_eff / entry_eff - 1
            elif k == 0:
                ret = px[0] / entry_eff - 1
            elif k == len(dts) - 1:
                ret = exit_eff / px[k - 1] - 1
            else:
                ret = px[k] / px[k - 1] - 1
            acc[dt] = acc.get(dt, 0.0) + ret
            cnt[dt] = cnt.get(dt, 0) + 1
    assert max(cnt.values()) <= n_slots, "槽位守恒被破坏"
    daily = pd.Series(acc).sort_index() / n_slots
    return daily


def curve_stats(daily: pd.Series) -> dict:
    if len(daily) < 20:
        return {}
    equity = (1 + daily).cumprod()
    total = equity.iloc[-1] - 1
    years = len(daily) / 252
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    mdd = (1 - equity / equity.cummax()).max()
    return {
        "年化%": round(ann * 100, 1),
        "夏普": round(sharpe, 2),
        "最大回撤%": round(mdd * 100, 1),
        "卡玛": round(ann / mdd, 2) if mdd > 0 else np.nan,
        "交易日数": len(daily),
    }


def pair_stats(trades: pd.DataFrame) -> dict:
    if not len(trades):
        return {}
    r = trades["ret_net_pct"].to_numpy()
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "交易数": len(r),
        "胜率%": round(len(wins) / len(r) * 100, 1),
        "盈亏比": round(abs(wins.mean() / losses.mean()), 2) if len(wins) and len(losses) else np.nan,
        "净均值%": round(r.mean(), 2),
        "净中位%": round(float(np.median(r)), 2),
        "毛均值%": round(trades["ret_gross_pct"].mean(), 2),
        "平均持有": round(trades["hold_days"].mean(), 1),
    }


# --------------------------------------------------------------------------- #
# 随机对照（同日、同成交额十分位、同持有期）
# --------------------------------------------------------------------------- #
class ControlSampler:
    def __init__(self):
        panel = pd.read_parquet(CAND_DIR / "panel.parquet")
        self.open_w = panel.pivot_table(index="dt", columns="symbol", values="open")
        self.close_w = panel.pivot_table(index="dt", columns="symbol", values="close")
        amt = panel.pivot_table(index="dt", columns="symbol", values="amount_e")
        self.decile = np.floor(amt.rank(axis=1, pct=True).mul(10).clip(upper=9.999))
        self.dates = self.close_w.index
        self.rng = np.random.default_rng(RNG_SEED)

    def excess_for(self, trade) -> float:
        """单笔交易的毛超额（vs 对照中位）；样本不足返回 NaN。"""
        dec_dt, entry_dt, exit_dt = trade.dec_dt, trade.entry_dt, trade.exit_dt
        if dec_dt not in self.decile.index or entry_dt not in self.dates or exit_dt not in self.dates:
            return np.nan
        row = self.decile.loc[dec_dt]
        my_dec = row.get(trade.symbol)
        if pd.isna(my_dec):
            return np.nan
        pool = row.index[(row == my_dec) & (row.index != trade.symbol)]
        if len(pool) < CONTROL_MIN_VALID:
            return np.nan
        pick = self.rng.choice(pool, size=min(CONTROL_K, len(pool)), replace=False)
        o = self.open_w.loc[entry_dt, pick].to_numpy(dtype=float)
        c = self.close_w.loc[exit_dt, pick].to_numpy(dtype=float)
        ret = c / o - 1
        ret = ret[np.isfinite(ret)]
        if len(ret) < CONTROL_MIN_VALID:
            return np.nan
        return trade.ret_gross_pct - float(np.median(ret)) * 100


def excess_report(trades: pd.DataFrame, sampler: ControlSampler) -> pd.DataFrame:
    ex = np.array([sampler.excess_for(t) for t in trades.itertuples()])
    trades = trades.assign(excess_pct=ex)
    rows = []
    groups = [
        ("ALL", trades),
        ("IS(≤2023)", trades[trades["seg"] == "train"]),
        ("OOS(≥2024)", trades[trades["seg"] == "test"]),
    ]
    groups += [(str(y), g) for y, g in trades.groupby("year")]
    for tag, g in groups:
        v = g["excess_pct"].dropna()
        if len(v) < 30:
            continue
        t_stat = v.mean() / (v.std() / np.sqrt(len(v))) if v.std() > 0 else np.nan
        rows.append(
            {
                "段": tag,
                "n": len(v),
                "超额均值%": round(v.mean(), 2),
                "超额中位%": round(float(v.median()), 2),
                "t": round(t_stat, 2),
                "超额>0占比%": round((v > 0).mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    cand = pd.read_parquet(CAND_DIR / "candidates.parquet")
    print(f"[候选] {len(cand)} 行（含全部 delay）")
    st_intervals = load_st_intervals()
    sampler = ControlSampler()
    print(f"[面板] {sampler.close_w.shape} | ST 区间 {len(st_intervals)} 只")

    summary: dict = {}
    for mode in MODES:
        df = gated_candidates(cand, mode, st_intervals)
        print(f"\n{'=' * 110}\n  {mode} | 过滤后候选 {len(df)} 笔\n{'=' * 110}")
        summary[mode] = {"n_candidates": int(len(df))}

        for n_slots in SLOT_COUNTS:
            trades, daily = simulate_slots(df, n_slots)
            tag = f"N{n_slots}"
            print(f"\n--- {mode} | {n_slots} 槽 | 成交 {len(trades)} 笔 ---")
            seg_rows = {}
            for seg_tag, g in [
                ("ALL", trades),
                ("IS(≤2023)", trades[trades["seg"] == "train"]),
                ("OOS(≥2024)", trades[trades["seg"] == "test"]),
                *[(str(y), g) for y, g in trades.groupby("year")],
            ]:
                ps = pair_stats(g)
                if ps:
                    seg_rows[seg_tag] = ps
            print(pd.DataFrame(seg_rows).T.to_string())
            cs_all = curve_stats(daily)
            cs_is = curve_stats(daily[daily.index <= TRAIN_END])
            cs_oos = curve_stats(daily[daily.index > TRAIN_END])
            print(f"  组合(净): ALL {cs_all} | IS {cs_is} | OOS {cs_oos}")
            summary[mode][tag] = {"pair": seg_rows, "curve": {"ALL": cs_all, "IS": cs_is, "OOS": cs_oos}}

            trades.to_parquet(OUTPUT_DIR / f"trades_{mode}_{tag}.parquet", index=False)
            equity = (1 + daily).cumprod()
            try:
                import plotly.graph_objects as go

                fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, mode="lines"))
                fig.update_layout(title=f"主升浪实盘镜像 {mode} {n_slots}槽（净）", yaxis_type="log")
                fig.write_html(OUTPUT_DIR / f"equity_{mode}_{tag}.html", include_plotlyjs="cdn")
            except Exception as e:
                print(f"  equity HTML 失败: {e}")

            if n_slots == SLOT_COUNTS[0]:
                rep = excess_report(trades, sampler)
                print(f"\n  [beta 剥离] {mode} | {n_slots} 槽 | 毛超额 vs 同日同成交额十分位随机对照(K={CONTROL_K})")
                print(rep.to_string(index=False))
                summary[mode]["excess"] = rep.to_dict("records")
                oos = rep[rep["段"] == "OOS(≥2024)"]
                if len(oos):
                    verdict = (
                        "OOS 超额显著为正 → 有选股 alpha"
                        if (oos["超额均值%"].iloc[0] > 0 and abs(oos["t"].iloc[0]) >= 2)
                        else "OOS 超额不显著/为负 → 收益主体为规模/市场 beta"
                    )
                    print(f"  [判定] {verdict}")
                    summary[mode]["verdict"] = verdict

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[完成] {time.time() - t0:.0f}s | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
