"""主升浪追涨 CZSC 特征 — 每日全 A 股扫描选股

对全 A 股最新数据进行 7 项主升浪特征等权打分（>= 5 触发），
标注"可入场"（已回调至 MA10）或"等待回调"（信号已触发但尚未回调），
输出含止损价和止盈规则的选股清单。

策略：B+D 组合（回调入场 + SL2兜底 + 分级跟踪止损）
回测：年化 113.5%、夏普 3.02、卡玛 3.81、最大回撤 29.8%

用法：
    PYTHONUNBUFFERED=1 /path/to/.venv/bin/python daily_scan.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tinyshare as ts

from czsc import CZSC, Freq, format_standard_kline

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "_output" / "surge_wave_picks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 500
SCORE_THRESHOLD = 5
PULLBACK_WINDOW = 10  # 信号触发后等待回调的天数


def _extract_zs(bis_list):
    zs_list = []
    i = 0
    while i < len(bis_list) - 2:
        b1, b2, b3 = bis_list[i], bis_list[i + 1], bis_list[i + 2]
        zg = min(b1.high, b2.high, b3.high)
        zd = max(b1.low, b2.low, b3.low)
        if zg > zd:
            zs = {"zg": zg, "zd": zd, "bis": 3}
            j = i + 3
            while j < len(bis_list):
                bj = bis_list[j]
                if bj.high >= zd and bj.low <= zg:
                    zs["bis"] += 1
                    j += 1
                else:
                    break
            zs_list.append(zs)
            i = j
        else:
            i += 1
    return zs_list


def _score_surge(bis_up_to, zs_up_to, close_val, dif_now, dif_5ago,
                 ma5, ma10, ma20, ret20):
    """主升浪 7 项等权打分，返回 (总分, 明细 dict)"""
    raw = {}
    up_bis = [b for b in bis_up_to if b.direction.value == "向上"]
    dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]

    if len(up_bis) >= 3:
        p1, p2, p3 = up_bis[-3].power, up_bis[-2].power, up_bis[-1].power
        raw["S1_笔力加速"] = int(p3 > p2 > p1 and p1 > 0)
    else:
        raw["S1_笔力加速"] = 0

    if len(up_bis) >= 2 and len(dn_bis) >= 2:
        up_avg = np.mean([b.power for b in up_bis[-3:]])
        dn_avg = np.mean([b.power for b in dn_bis[-3:]])
        raw["S2_力度比"] = int(up_avg > dn_avg * 1.5) if dn_avg > 0 else 0
    else:
        raw["S2_力度比"] = 0

    if zs_up_to:
        raw["S3_脱离中枢"] = int(close_val > zs_up_to[-1]["zg"] * 1.2)
    else:
        raw["S3_脱离中枢"] = 0

    if len(dn_bis) >= 2:
        raw["S4_低点抬升"] = int(dn_bis[-1].low > dn_bis[-2].low)
    else:
        raw["S4_低点抬升"] = 0

    if ma5 > ma10 > ma20:
        spread = (ma5 - ma20) / ma20 * 100
        raw["S5_MA扩散"] = int(spread > 15)
    else:
        raw["S5_MA扩散"] = 0

    raw["S6_DIF加速"] = int(dif_now > 0 and dif_now > dif_5ago)
    raw["S7_涨幅确认"] = int(ret20 > 15)

    total = sum(raw.values())
    return total, raw


def _scan_one_stock(parquet_path: str) -> dict | None:
    """扫描单只股票：最近 PULLBACK_WINDOW+1 天内是否有主升浪信号"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if len(df) < MIN_BARS:
        return None

    code = df["ts_code"].iloc[0]
    if code.startswith(("688", "920", "83", "43")):
        return None

    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    try:
        bars = format_standard_kline(df, freq=Freq.D)
        c = CZSC(bars)
    except Exception:
        return None

    bis = c.bi_list
    if len(bis) < 10:
        return None

    close = df["close"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    # 在最近 PULLBACK_WINDOW+1 天范围内搜索信号
    scan_start = max(120, n - PULLBACK_WINDOW - 1)
    best_signal = None

    for idx in range(scan_start, n):
        if abs(pct_chg[idx]) >= 9.8:
            continue

        dt = dates[idx]
        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dt)]
        if len(bis_up_to) < 8:
            continue

        zs_up_to = _extract_zs(bis_up_to)
        dif_5ago = dif[idx - 5] if idx >= 5 else 0
        ret20 = (close[idx] / close[idx - 20] - 1) * 100 if idx >= 20 else 0

        total, detail = _score_surge(
            bis_up_to, zs_up_to, close[idx],
            dif[idx], dif_5ago, ma5[idx], ma10[idx], ma20[idx], ret20,
        )

        if total >= SCORE_THRESHOLD:
            if best_signal is None or total > best_signal["total"]:
                dn_bis = [b for b in bis_up_to if b.direction.value == "向下"]
                best_signal = {
                    "sig_idx": idx,
                    "sig_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "total": total,
                    "detail": detail,
                    "sl_bi": dn_bis[-1].low if dn_bis else None,
                }

    if best_signal is None:
        return None

    # 判断当前日（最后一天）是否满足回调入场条件
    last_idx = n - 1
    last_close = close[last_idx]
    last_ma10 = ma10[last_idx]
    last_date = pd.Timestamp(dates[last_idx]).strftime("%Y-%m-%d")

    days_since_signal = last_idx - best_signal["sig_idx"]

    if days_since_signal > PULLBACK_WINDOW and best_signal["sig_idx"] < last_idx:
        return None  # 信号超过窗口期

    # 回调入场判断
    if last_close <= last_ma10 * 1.02:
        entry_status = "可入场"
    elif best_signal["sig_idx"] == last_idx:
        entry_status = "今日触发"
    else:
        entry_status = "等待回调"

    # 最新一天重新打分（展示当前最新特征状态）
    bis_latest = [bi for bi in bis if bi.edt <= pd.Timestamp(dates[last_idx])]
    zs_latest = _extract_zs(bis_latest)
    dif_5ago_l = dif[last_idx - 5] if last_idx >= 5 else 0
    ret20_l = (close[last_idx] / close[last_idx - 20] - 1) * 100 if last_idx >= 20 else 0

    current_total, current_detail = _score_surge(
        bis_latest, zs_latest, close[last_idx],
        dif[last_idx], dif_5ago_l, ma5[last_idx], ma10[last_idx], ma20[last_idx], ret20_l,
    )

    # SL2 止损
    dn_bis_latest = [b for b in bis_latest if b.direction.value == "向下"]
    sl_bi = dn_bis_latest[-1].low if dn_bis_latest else None
    sl_pct = round((last_close - sl_bi) / last_close * 100, 1) if sl_bi and sl_bi > 0 else None

    ma_status = "多头" if ma5[last_idx] > ma10[last_idx] > ma20[last_idx] else ""

    # MA10 偏离度：当前价相对 MA10 的位置（负=在MA10下方，正=在MA10上方）
    ma10_dev = round((last_close / last_ma10 - 1) * 100, 2) if last_ma10 > 0 else 0

    # 缩量回调确认：最近 3 日均量 vs 20 日均量
    vol = df["vol"].values
    vol_ma3 = np.mean(vol[-3:]) if len(vol) >= 3 else vol[-1]
    vol_ma20 = np.mean(vol[-20:]) if len(vol) >= 20 else vol[-1]
    vol_shrink = round(vol_ma3 / vol_ma20, 2) if vol_ma20 > 0 else 1.0

    return {
        "代码": code,
        "名称": "",
        "行业": "",
        "日期": last_date,
        "收盘价": round(last_close, 2),
        "得分": best_signal["total"],
        "当前得分": current_total,
        **current_detail,
        "信号状态": entry_status,
        "信号日": best_signal["sig_date"],
        "信号新鲜度": days_since_signal,
        "MA状态": ma_status,
        "MA10": round(last_ma10, 2),
        "MA10偏离%": ma10_dev,
        "回调缩量比": vol_shrink,
        "推荐止损": round(sl_bi, 2) if sl_bi else None,
        "止损幅度%": sl_pct,
        "止盈规则": "浮盈10%启动15%跟踪|浮盈25%收紧到10%跟踪",
    }


def _load_stock_basic() -> tuple[dict, dict]:
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    name_map = dict(zip(basic["ts_code"], basic["name"]))
    industry_map = dict(zip(basic["ts_code"], basic["industry"]))
    return name_map, industry_map


def main():
    print("=" * 70)
    print("  主升浪追涨 CZSC 特征 — 每日选股扫描")
    print("=" * 70)

    print("[基础] 加载股票名称和行业...")
    name_map, industry_map = _load_stock_basic()
    print(f"  已加载 {len(name_map)} 只股票基础信息")

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] {len(parquet_files)} 只个股")

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] {n_workers} 进程")

    t0 = time.time()
    file_list = [str(p) for p in parquet_files]

    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_one_stock, file_list, chunksize=20), 1):
            if res is not None:
                results.append(res)
            if i % 1000 == 0 or i == len(file_list):
                print(f"  [{i}/{len(file_list)}] 符合 {len(results)} 只 | {time.time()-t0:.0f}s")

    for r in results:
        code = r["代码"]
        r["名称"] = name_map.get(code, "")
        r["行业"] = industry_map.get(code, "")

    print(f"\n[完成] 耗时 {time.time()-t0:.0f}s | 初筛 {len(results)} 只 (>= {SCORE_THRESHOLD} 分)")

    if not results:
        print("  今日无符合条件的标的")
        return

    scan_date = results[0]["日期"]

    # ── 入场优先级评分（4 个维度，总分 100） ──
    for r in results:
        priority = 0.0

        # 维度 1: 特征得分（30 分）— 得分越高越好
        priority += min(r["得分"] / 7 * 30, 30)

        # 维度 2: 止损风险可控（25 分）— 止损幅度在 5%-20% 最佳
        sl_pct = r.get("止损幅度%")
        if sl_pct is not None and sl_pct > 0:
            if 5 <= sl_pct <= 20:
                priority += 25
            elif 3 <= sl_pct < 5 or 20 < sl_pct <= 30:
                priority += 15
            elif sl_pct > 30:
                priority += 5  # 止损太宽，风险大
            else:
                priority += 0  # 止损太窄（< 3%），可能噪音
        else:
            priority += 0

        # 维度 3: MA10 贴合度（25 分）— 越接近 MA10 入场价越好
        ma10_dev = r.get("MA10偏离%", 0)
        if r["信号状态"] == "可入场":
            # 在 MA10 下方 0-2%: 最佳; 上方 0-5%: 可接受
            if -5 <= ma10_dev <= 2:
                priority += 25
            elif 2 < ma10_dev <= 5:
                priority += 15
            else:
                priority += 5
        elif r["信号状态"] == "今日触发":
            priority += 10  # 还没回调，先观察
        else:
            priority += 5   # 等待回调

        # 维度 4: 信号新鲜度 + 缩量确认（20 分）
        freshness = r.get("信号新鲜度", 10)
        vol_shrink = r.get("回调缩量比", 1.0)

        fresh_score = max(0, 10 - freshness) / 10 * 10  # 越新越好
        # 缩量回调是好事（量比 < 0.8 说明回调是缩量的）
        shrink_score = 10 if vol_shrink < 0.8 else (7 if vol_shrink < 1.0 else 3)
        priority += fresh_score + shrink_score

        r["优先级"] = round(priority, 1)
        r["等级"] = "A" if priority >= 80 else ("B" if priority >= 65 else "C")

    # ── 硬性过滤 ──
    actionable = [r for r in results
                  if r["信号状态"] == "可入场"
                  and r.get("止损幅度%") is not None
                  and 3 < r["止损幅度%"] <= 30
                  and r["当前得分"] >= SCORE_THRESHOLD]
    actionable.sort(key=lambda x: -x["优先级"])

    watchlist = [r for r in results if r not in actionable]
    watchlist.sort(key=lambda x: -x["优先级"])

    # ── 输出精选结果 ──
    TOP_N = 20

    n_all = len(results)
    n_act = len(actionable)
    n_watch = len(watchlist)

    print(f"\n{'='*170}")
    print(f"  {scan_date} 主升浪精选  |  "
          f"精选可入场 {min(n_act, TOP_N)}/{n_act} 只 | "
          f"观察池 {n_watch} 只 | 初筛总计 {n_all} 只")
    print(f"  策略：B+D 组合（回调入场 + SL2兜底 + 分级跟踪止损）| 回测卡玛 3.81")
    print(f"  排序：优先级综合评分 = 特征得分(30) + 止损可控(25) + MA10贴合(25) + 新鲜度缩量(20)")
    print(f"{'='*170}")

    header = (f"{'序':>3} {'等级':>3} {'代码':>12} {'名称':<8} {'行业':<8} {'收盘价':>7} {'得分':>3} "
              f"{'S1':>3}{'S2':>3}{'S3':>3}{'S4':>3}{'S5':>3}{'S6':>3}{'S7':>3} "
              f"{'优先级':>5} {'MA10偏离':>7} {'缩量比':>5} "
              f"{'止损价':>7} {'幅度%':>6} {'信号日':>12}")
    print(header)
    print("-" * 170)

    for i, r in enumerate(actionable[:TOP_N], 1):
        s1 = r.get("S1_笔力加速", 0)
        s2 = r.get("S2_力度比", 0)
        s3 = r.get("S3_脱离中枢", 0)
        s4 = r.get("S4_低点抬升", 0)
        s5 = r.get("S5_MA扩散", 0)
        s6 = r.get("S6_DIF加速", 0)
        s7 = r.get("S7_涨幅确认", 0)

        sl = r.get("推荐止损")
        sl_s = f"{sl:.2f}" if isinstance(sl, (int, float)) and sl else ""
        sl_pct_v = r.get("止损幅度%")
        sl_pct_s = f"{sl_pct_v:.1f}" if isinstance(sl_pct_v, (int, float)) and sl_pct_v is not None else ""
        ma10_dev = r.get("MA10偏离%", 0)
        vol_shrink = r.get("回调缩量比", 1.0)

        name = r.get("名称", "")[:6]
        industry = r.get("行业", "")[:6]
        tier = r.get("等级", "C")

        print(f" {i:>2}   {tier:>2}  {r['代码']:>12} {name:<8} {industry:<8} {r['收盘价']:>7.2f} {r['得分']:>3} "
              f"{s1:>3}{s2:>3}{s3:>3}{s4:>3}{s5:>3}{s6:>3}{s7:>3} "
              f"{r['优先级']:>5} {ma10_dev:>6.1f}% {vol_shrink:>5.2f} "
              f"{sl_s:>7} {sl_pct_s:>6} {r.get('信号日',''):>12}")

    if n_act > TOP_N:
        print(f"\n  ... 还有 {n_act - TOP_N} 只可入场标的未显示（优先级较低）")

    # ── 入场指引 ──
    print(f"\n{'='*170}")
    print("  入场操作指引")
    print(f"{'='*170}")
    print("  [A 级] 优先级 >= 80: 高确定性，可直接入场")
    print("  [B 级] 优先级 65-79: 关注观察，等尾盘确认或次日低开入场")
    print("  [C 级] 优先级 < 65:  观察池，等更好的回调位置再考虑")
    print()
    print("  入场时机判断：")
    print("    1. MA10偏离 <= 0%: 价格已回调到 MA10 下方，最佳入场位")
    print("    2. MA10偏离 0-2%:  价格贴近 MA10，可入场")
    print("    3. 缩量比 < 0.8:  回调伴随缩量，洗盘特征明确")
    print("    4. 止损幅度 5-15%: 风险可控，盈亏比合理")

    # ── 保存全量 parquet ──
    out_parquet = OUTPUT_DIR / f"picks_{scan_date}.parquet"

    col_order = [
        "代码", "名称", "行业", "等级", "优先级", "日期", "收盘价",
        "得分", "当前得分",
        "S1_笔力加速", "S2_力度比", "S3_脱离中枢", "S4_低点抬升",
        "S5_MA扩散", "S6_DIF加速", "S7_涨幅确认",
        "信号状态", "信号日", "信号新鲜度",
        "MA状态", "MA10", "MA10偏离%", "回调缩量比",
        "推荐止损", "止损幅度%", "止盈规则",
    ]
    out_df = pd.DataFrame(results)
    existing_cols = [c for c in col_order if c in out_df.columns]
    extra_cols = [c for c in out_df.columns if c not in col_order]
    out_df = out_df[existing_cols + extra_cols]
    out_df.to_parquet(out_parquet, index=False)
    print(f"\n[文件] {out_parquet}")


if __name__ == "__main__":
    main()
