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

import trend_regime as tr  # noqa: E402
from trend_regime import REGIME_CN, Regime, surge_onset, surge_score  # noqa: E402

TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
OUTPUT_DIR = REPO / "scripts" / "_output" / "surge_regime_picks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_WINDOW = 10  # 回看最近 N 根找最新启动信号
TAIL = 160  # iter_states 快路径窗口（> SCAN_WINDOW + SURGE_PRIOR_WINDOW）
UPTREND_FAMILY = {Regime.UpwardDeparture, Regime.ThirdBuy, Regime.MainUptrend, Regime.Acceleration}
REGIME_QUALITY = {Regime.Acceleration: 20, Regime.ThirdBuy: 18, Regime.MainUptrend: 16, Regime.UpwardDeparture: 12}
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
        return None

    p_onset, variant = found
    freshness = (len(states) - 1) - p_onset
    feats = last.feats or {}
    close = last.close
    sl = last.sl_ref if last.sl_ref == last.sl_ref else last.zd  # NaN 检查；退化用中枢下沿
    sl = sl if (sl == sl and sl > 0) else None
    sl_pct = round((close - sl) / close * 100, 1) if sl else None
    code = df["symbol"].iloc[0]

    return {
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
        "推荐止损": round(sl, 2) if sl else None,
        "止损幅度%": sl_pct,
        "止盈规则": TP_RULE,
    }


def _priority(r: dict) -> float:
    p = 0.0
    p += min(r["score"], 100) * 0.35  # 主升强度（35）
    sl_pct = r.get("止损幅度%")  # 止损风险可控（25）
    if sl_pct is None or sl_pct <= 0:
        p -= 30  # 止损穿透/无效
    elif 5 <= sl_pct <= 20:
        p += 25
    elif 3 <= sl_pct < 5 or 20 < sl_pct <= 30:
        p += 15
    else:
        p += 5
    p += max(0, (SCAN_WINDOW - r["新鲜度"]) / SCAN_WINDOW) * 20  # 新鲜度（20）
    p += REGIME_QUALITY.get(Regime(_regime_of(r)), 10)  # 当前状态质量（20）
    return round(max(p, 0), 1)


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


def main():
    print("=" * 70)
    print("  主升浪启动（缠论状态机）— 每日选股扫描")
    print("=" * 70)
    try:
        name_map, industry_map = _load_stock_basic()
        print(f"[基础] 已加载 {len(name_map)} 只股票名称/行业")
    except Exception as e:
        print(f"[基础] 名称/行业加载失败（仅离线扫描）：{e}")
        name_map, industry_map = {}, {}

    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(
        f"[数据] {len(files)} 只 | {n_workers} 进程 | 门控 量比≥{tr.SURGE_GATE_VOL_RATIO} 散度≥{tr.SURGE_GATE_MA_SPREAD}%\n"
    )

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_one, files, chunksize=20), 1):
            if res:
                results.append(res)
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 命中 {len(results)} | {time.time() - t0:.0f}s")

    if not results:
        print("\n今日无处于主升浪启动/追入窗口的标的")
        return

    for r in results:
        r["名称"] = name_map.get(r["代码"], "")
        r["行业"] = industry_map.get(r["代码"], "")
        r["优先级"] = _priority(r)
        sl_pct = r.get("止损幅度%")
        r["等级"] = (
            "C"
            if (sl_pct is None or sl_pct <= 0)
            else ("A" if r["优先级"] >= 75 else "B" if r["优先级"] >= 60 else "C")
        )

    results.sort(key=lambda x: -x["优先级"])
    scan_date = results[0]["日期"]
    n_a = sum(1 for r in results if r["等级"] == "A")

    print(f"\n{'=' * 150}")
    print(f"  {scan_date} 主升浪启动精选 | A级 {n_a} 只 | 命中合计 {len(results)} 只 | 止损止盈：{TP_RULE}")
    print(f"{'=' * 150}")
    header = (
        f"{'序':>3} {'级':>2} {'代码':>11} {'名称':<8} {'行业':<7} {'收盘':>7} {'状态':<7} "
        f"{'方式':<5} {'优先级':>5} {'score':>5} {'量比':>5} {'散度%':>6} {'ret20':>6} {'止损':>7} {'幅度%':>6} {'新鲜':>4}"
    )
    print(header)
    print("-" * 150)
    for i, r in enumerate(results[:20], 1):
        print(
            f"{i:>3} {r['等级']:>2} {r['代码']:>11} {r['名称'][:6]:<8} {r['行业'][:6]:<7} {r['收盘价']:>7.2f} "
            f"{r['当前状态']:<7} {r['启动方式']:<5} {r['优先级']:>5} {r['score']:>5} "
            f"{(r['量比'] or 0):>5.2f} {(r['MA散度%'] or 0):>6.2f} {(r['ret20%'] or 0):>6.1f} "
            f"{(r['推荐止损'] or 0):>7.2f} {(r['止损幅度%'] if r['止损幅度%'] is not None else 0):>6.1f} {r['新鲜度']:>4}"
        )
    if len(results) > 20:
        print(f"\n  ... 另有 {len(results) - 20} 只命中未显示（优先级较低）")

    print("\n  分级：A(优先级≥75 可入场) / B(60-74 等回踩确认) / C(<60 或止损穿透 观察)")
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
        "推荐止损",
        "止损幅度%",
        "止盈规则",
    ]
    pd.DataFrame(results)[cols].to_parquet(out, index=False)
    print(f"\n[文件] {out}")


if __name__ == "__main__":
    main()
