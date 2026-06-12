"""主升浪启动（缠论走势状态机）—— 每日全 A 股因果选股

基于 `trend_regime` 的 11 态走势状态机，用因果「主升浪启动」信号（`surge_onset`）
扫描全 A 股最新一根 K 线，输出当日处于主升浪启动/追入窗口的标的 + 推荐止损。

相比旧 surge-wave-stock-picker（S1-S7 等权 + 全量 CZSC 后 edt 过滤，有轻微未来函数泄漏），
本扫描用流式状态机（`iter_states(tail=...)`）+ 原生中枢，**对「今日」无未来数据，严格因果**。

用法：
    PYTHONUNBUFFERED=1 /home/lovelyzzc/czsc/.venv/bin/python daily_scan.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tinyshare as ts

# 引入仓库 scripts/ 下的 trend_regime（共享因果信号，单一真源）
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))

import surge_live as sl  # noqa: E402
import surge_portfolio_backtest as spb  # noqa: E402  # load_st_intervals / is_st_on（本地 namechange 历史 ST 判定）
import trend_regime as tr  # noqa: E402
from trend_regime import REGIME_CN, Regime, priority_score, surge_onset, surge_score  # noqa: E402

TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
OUTPUT_DIR = REPO / "scripts" / "_output" / "surge_regime_picks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_WINDOW = 10  # 回看最近 N 根找最新启动信号
TAIL = 160  # iter_states 快路径窗口（> SCAN_WINDOW + SURGE_PRIOR_WINDOW）
MIN_AMOUNT_E = float(os.getenv("SURGE_PICKER_MIN_AMOUNT_E", "1.0"))  # 最近一日成交额下限：亿元
STOP_MIN_PCT = float(os.getenv("SURGE_PICKER_STOP_MIN_PCT", "8"))
STOP_MAX_PCT = float(os.getenv("SURGE_PICKER_STOP_MAX_PCT", "20"))
UPTREND_FAMILY = {Regime.UpwardDeparture, Regime.ThirdBuy, Regime.MainUptrend, Regime.Acceleration}
TP_RULE = "SL2/中枢下沿托底 + 浮盈后18%跟踪 + 背驰减仓/破坏清仓"


def _scan_one(parquet_path: str) -> dict | None:
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    states = tr.iter_states(df, with_features=True, tail=TAIL)
    if len(states) < SCAN_WINDOW + 2:
        return None

    regimes = [s.regime for s in states]
    last = states[-1]
    if last.regime not in UPTREND_FAMILY:  # 结构已破坏/已转卖点 → 非追入窗口
        return None

    code = df["symbol"].iloc[0]
    amount = float(df["amount"].iloc[-1]) if "amount" in df.columns else 0.0
    amount_e = sl.amount_to_e(amount)  # round 3，与 dump/镜像核对同口径

    # 实验 · delay5 回踩买点（独立于主表逻辑，不影响默认输出）
    exp = sl.detect_delay5(states)
    if exp:
        exp["代码"] = code
        exp["成交额亿"] = amount_e

    # 最近 SCAN_WINDOW 根内最新的一次主升浪启动
    found = None
    for p in range(len(states) - 1, max(0, len(states) - 1 - SCAN_WINDOW), -1):
        prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
        for mode in ("confirm", "anticipate"):
            if surge_onset(states[p - 1].regime, states[p].regime, states[p].feats, prior, mode):
                found = (p, mode)
                break
        if found:
            break
    if found is None:
        return {"main": None, "exp": exp} if exp else None

    p_onset, variant = found
    freshness = (len(states) - 1) - p_onset
    feats = last.feats or {}
    close = last.close
    sl_ref = last.sl_ref if last.sl_ref == last.sl_ref else last.zd  # NaN 检查；退化用中枢下沿
    sl_ref = sl_ref if (sl_ref == sl_ref and sl_ref > 0) else None
    sl_pct = round((close - sl_ref) / close * 100, 1) if sl_ref else None

    main_rec = {
        "代码": code,
        "名称": "",
        "行业": "",
        "日期": last.dt.strftime("%Y-%m-%d"),
        "收盘价": round(close, 2),
        "当前状态": REGIME_CN[Regime(last.regime)],
        "启动方式": "确认追入" if variant == "confirm" else "启动埋伏",
        "启动日": states[p_onset].dt.strftime("%Y-%m-%d"),
        "新鲜度": freshness,
        "score": surge_score(feats),
        "量比": feats.get("vol_ratio"),
        "MA散度%": feats.get("ma_spread_pct"),
        "ret20%": feats.get("ret20"),
        "成交额亿": amount_e,
        "推荐止损": round(sl_ref, 2) if sl_ref else None,
        "止损幅度%": sl_pct,
        "过滤原因": "",
        "可操作": False,
        "止盈规则": TP_RULE,
    }
    return {"main": main_rec, "exp": exp}


def _priority(r: dict) -> float:
    return priority_score(r["score"], r.get("止损幅度%"), r["新鲜度"], _regime_of(r), scan_window=SCAN_WINDOW)


def _regime_of(r: dict) -> int:
    for rg, cn in REGIME_CN.items():
        if cn == r["当前状态"]:
            return int(rg)
    return int(Regime.MainUptrend)


def _load_stock_basic():
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    return (
        dict(zip(basic["ts_code"], basic["name"], strict=False)),
        dict(zip(basic["ts_code"], basic["industry"], strict=False)),
    )


def _is_st_or_delist(name: str) -> bool:
    text = str(name or "").upper()
    return text.startswith(("ST", "*ST")) or "退" in text


def _filter_reasons(r: dict, *, check_name: bool) -> list[str]:
    reasons = []
    if check_name and _is_st_or_delist(r.get("名称", "")):
        reasons.append("ST/退市风险")
    sl_pct = r.get("止损幅度%")
    if sl_pct is None:
        reasons.append("无有效止损")
    elif sl_pct <= 0:
        reasons.append("止损已穿透")
    elif not STOP_MIN_PCT <= sl_pct <= STOP_MAX_PCT:
        reasons.append(f"止损幅度不在{STOP_MIN_PCT:g}-{STOP_MAX_PCT:g}%")
    if r.get("成交额亿", 0) < MIN_AMOUNT_E:
        reasons.append(f"成交额<{MIN_AMOUNT_E:g}亿")
    return reasons


def main():
    print("=" * 70)
    print("  主升浪启动（缠论状态机）— 每日选股扫描")
    print("=" * 70)
    metadata_available = False
    try:
        name_map, industry_map = _load_stock_basic()
        print(f"[基础] 已加载 {len(name_map)} 只股票名称/行业")
        metadata_available = True
    except Exception as e:
        print(f"[基础] 名称/行业加载失败（仅离线扫描）：{e}")
        name_map, industry_map = {}, {}

    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(
        f"[数据] {len(files)} 只 | {n_workers} 进程 | 门控 量比≥{tr.SURGE_GATE_VOL_RATIO} 散度≥{tr.SURGE_GATE_MA_SPREAD}%\n"
    )

    t0 = time.time()
    results, exp_raw = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_one, files, chunksize=20), 1):
            if res:
                if res.get("main"):
                    results.append(res["main"])
                if res.get("exp"):
                    exp_raw.append(res["exp"])
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 命中 {len(results)} | {time.time() - t0:.0f}s")

    if results:
        _report_default(results, name_map, industry_map, metadata_available)
    else:
        print("\n今日无处于主升浪启动/追入窗口的标的")
    _report_experimental(exp_raw, name_map, industry_map, metadata_available)


def _report_default(results, name_map, industry_map, metadata_available):
    for r in results:
        r["名称"] = name_map.get(r["代码"], "")
        r["行业"] = industry_map.get(r["代码"], "")
        r["优先级"] = _priority(r)
        sl_pct = r.get("止损幅度%")
        reasons = _filter_reasons(r, check_name=metadata_available)
        r["过滤原因"] = "|".join(reasons)
        r["可操作"] = not reasons
        r["等级"] = (
            "C"
            if (sl_pct is None or sl_pct <= 0 or reasons)
            else ("A" if r["优先级"] >= 75 else "B" if r["优先级"] >= 60 else "C")
        )

    results.sort(key=lambda x: -x["优先级"])
    scan_date = results[0]["日期"]
    actionable = [r for r in results if r["可操作"]]
    watchlist = [r for r in results if not r["可操作"]]
    actionable.sort(key=lambda x: -x["优先级"])
    n_a = sum(1 for r in actionable if r["等级"] == "A")

    print(f"\n{'=' * 150}")
    print(
        f"  {scan_date} 主升浪启动精选 | 可操作 {len(actionable)} 只 / A级 {n_a} 只 | "
        f"观察池 {len(watchlist)} 只 | 止损止盈：{TP_RULE}"
    )
    st_filter = "剔除 ST/退市风险、" if metadata_available else "ST过滤未启用（缺少名称/行业元数据）、"
    print(f"  硬过滤：{st_filter}成交额<{MIN_AMOUNT_E:g}亿、止损幅度不在{STOP_MIN_PCT:g}-{STOP_MAX_PCT:g}%")
    print(f"{'=' * 150}")
    header = (
        f"{'序':>3} {'级':>2} {'代码':>11} {'名称':<8} {'行业':<7} {'收盘':>7} {'状态':<7} "
        f"{'方式':<5} {'优先级':>5} {'score':>5} {'量比':>5} {'散度%':>6} {'ret20':>6} {'额亿':>5} {'止损':>7} {'幅度%':>6} {'新鲜':>4}"
    )
    print(header)
    print("-" * 150)
    display_rows = actionable[:20] if actionable else results[:20]
    for i, r in enumerate(display_rows, 1):
        print(
            f"{i:>3} {r['等级']:>2} {r['代码']:>11} {r['名称'][:6]:<8} {r['行业'][:6]:<7} {r['收盘价']:>7.2f} "
            f"{r['当前状态']:<7} {r['启动方式']:<5} {r['优先级']:>5} {r['score']:>5} "
            f"{(r['量比'] or 0):>5.2f} {(r['MA散度%'] or 0):>6.2f} {(r['ret20%'] or 0):>6.1f} "
            f"{r['成交额亿']:>5.2f} {(r['推荐止损'] or 0):>7.2f} "
            f"{(r['止损幅度%'] if r['止损幅度%'] is not None else 0):>6.1f} {r['新鲜度']:>4}"
        )
    if len(actionable) > 20:
        print(f"\n  ... 另有 {len(actionable) - 20} 只可操作标的未显示（优先级较低）")
    if not actionable:
        print("\n  今日无通过硬过滤的可操作标的，上表为观察池优先级前 20。")

    print("\n  分级：A(优先级≥75 且通过硬过滤) / B(60-74 且通过硬过滤) / C(观察池或硬过滤未通过)")
    print("  方式：确认追入=已进主升(7/8) | 启动埋伏=刚向上离开中枢(5) | 新鲜=距启动信号天数")

    out = OUTPUT_DIR / f"picks_{scan_date}.parquet"
    cols = [
        "代码",
        "名称",
        "行业",
        "等级",
        "优先级",
        "日期",
        "收盘价",
        "当前状态",
        "启动方式",
        "启动日",
        "新鲜度",
        "score",
        "量比",
        "MA散度%",
        "ret20%",
        "成交额亿",
        "推荐止损",
        "止损幅度%",
        "可操作",
        "过滤原因",
        "止盈规则",
    ]
    pd.DataFrame(results)[cols].to_parquet(out, index=False)
    print(f"\n[文件] {out}")


def _report_experimental(exp_raw, name_map, industry_map, metadata_available):
    """实验 · 回踩买点 delay5。镜像口径固定（不吃环境变量阈值），仅记录与研究，非默认买入建议。

    口径 = scripts/surge_pullback_entry_research.py 的 state_delay5 变体：
    anticipate 信号（信号日门控）后第 5 个交易日收盘决策、次日开盘入场；
    市场状态门 high20_ratio>0.12 & 等权指数>MA20；硬过滤 成交额≥1亿 + 止损带 8-20% + 剔除 ST。
    """
    print(f"\n{'=' * 150}")
    print("  实验 · 回踩买点 delay5（研究中，非买入建议）— anticipate 信号后第 5 个交易日决策、次日开盘入场")
    print(
        "  机制注记：稳健性审计显示超额来自「5 日存活确认」而非回踩定价；超额中位数为负、肥尾驱动（详见 SURGE_REGIME_DELAY5_MIRROR_2026-06-11.md）"
    )
    print(f"{'=' * 150}")

    t0 = time.time()
    print("[市场状态] 构建全市场尾部面板 ...")
    panel = sl.build_live_panel()
    state = sl.live_market_state(panel)
    row = state.iloc[-1]
    gate = sl.market_gate_open(row)
    sl.append_market_state_log(state.tail(10))
    above = "高于" if row["ew_index_above_ma20"] > 0 else "低于"
    print(
        f"  {pd.Timestamp(row['dt']).date()} high20_ratio={row['high20_ratio']:.3f}（门槛>0.12）| "
        f"等权指数{above} MA20 → 市场状态门【{'开' if gate else '关'}】 | {time.time() - t0:.0f}s"
    )

    if not exp_raw:
        print("  今日无 delay5 结构候选（anticipate 信号后第 5 日仍在上行家族：0 只）")
        return

    # ST 过滤优先用本地 namechange 区间（离线可用），名称元数据仅作补充；
    # 两者都不可用时不允许静默放行——候选一律标记不可操作（与 skill 宣称的固定硬过滤一致）
    st_intervals = spb.load_st_intervals()
    st_filter_ok = bool(st_intervals) or metadata_available
    if not st_filter_ok:
        print("  ⚠ ST 过滤数据不可用（namechange.parquet 与名称元数据均缺失）→ 今日候选一律记为不可操作")
    for r in exp_raw:
        r["名称"] = name_map.get(r["代码"], "")
        r["行业"] = industry_map.get(r["代码"], "")
        reasons = sl.hard_filter_reasons(r["sl_pct"], r["成交额亿"])
        is_st = bool(st_intervals) and spb.is_st_on(st_intervals, r["代码"], pd.Timestamp(r["dec_dt"]))
        if not is_st and metadata_available:
            is_st = _is_st_or_delist(r["名称"])
        if is_st:
            reasons.append("ST/退市风险")
        elif not st_filter_ok:
            reasons.append("ST状态未知")
        r["过滤原因"] = "|".join(reasons)
        r["市场门"] = gate
        r["可操作"] = gate and not reasons

    exp_raw.sort(key=lambda x: -x["priority"])
    actionable = [r for r in exp_raw if r["可操作"]]
    print(
        f"  结构候选 {len(exp_raw)} 只 | 通过硬过滤+市场门 {len(actionable)} 只 | "
        f"入场=次日开盘（开盘逼近涨停 ≥板限-0.3% 则放弃）| 退出=SL2+18%跟踪+背驰/破坏"
    )
    header = (
        f"{'序':>3} {'代码':>11} {'名称':<8} {'收盘':>7} {'状态':<7} {'优先级':>5} {'score':>5} "
        f"{'止损':>7} {'幅度%':>6} {'额亿':>6} {'启动日':<11} {'过滤原因':<18}"
    )
    print(header)
    print("-" * 150)
    for i, r in enumerate(exp_raw[:10], 1):
        print(
            f"{i:>3} {r['代码']:>11} {r['名称'][:6]:<8} {r['close']:>7.2f} "
            f"{REGIME_CN[Regime(r['dec_regime'])]:<7} {r['priority']:>5} {r['score']:>5} "
            f"{(r['sl_ref'] or 0):>7.2f} {(r['sl_pct'] if r['sl_pct'] is not None else 0):>6.1f} "
            f"{r['成交额亿']:>6.2f} {pd.Timestamp(r['sig_dt']).strftime('%Y-%m-%d'):<11} {r['过滤原因']:<18}"
        )
    if len(exp_raw) > 10:
        print(f"  ... 另有 {len(exp_raw) - 10} 只结构候选未显示")
    if not gate:
        print("  ⚠ 市场状态门关闭：以上仅作记录，按策略口径今日不开新仓（2026 环境常态，属系统按设计工作）")

    dec_date = pd.Timestamp(exp_raw[0]["dec_dt"]).strftime("%Y-%m-%d")
    out = OUTPUT_DIR / f"picks_exp_delay5_{dec_date}.parquet"
    df = pd.DataFrame(exp_raw)
    df["sig_dt"] = pd.to_datetime(df["sig_dt"])
    df["dec_dt"] = pd.to_datetime(df["dec_dt"])
    df.to_parquet(out, index=False)
    print(f"  [文件] {out}")


if __name__ == "__main__":
    main()
