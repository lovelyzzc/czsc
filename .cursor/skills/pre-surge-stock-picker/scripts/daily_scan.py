"""主升前 CZSC 特征 — 每日全 A 股扫描选股

对全 A 股最新一天的数据进行 7 项缠论主升前特征打分，
输出得分 >= 4 的标的清单，含得分明细。

默认采用 SL2（笔结构止损）作为推荐止损策略——回测显示该策略
年化 82.9%、夏普 3.02、最大回撤 12.1%、卡玛 6.85，综合最优。
同时保留中枢止损和 ATR 止损作为辅助参考。

用法：
    PYTHONUNBUFFERED=1 /path/to/.venv/bin/python daily_scan.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import os

import numpy as np
import pandas as pd
import tinyshare as ts

from czsc import CZSC, Freq, format_standard_kline

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "_output" / "daily_picks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 500

# 通过 149399 条历史样本（单因子分析 + 逻辑回归）优化得出的特征权重
# C6（缩量后放量）预测能力最强：条件收益差 +1.47%，胜率差 +5.3%
# C1（中枢上移）次之：条件收益差 +0.43%
# 其余特征独立预测能力弱或为负，保留最小权重 0.5 作为组合贡献
FEATURE_WEIGHTS = [1.0, 0.5, 0.5, 0.5, 0.5, 3.0, 0.5]  # C1-C7
SCORE_THRESHOLD = 5.0


def _extract_zs(bis_list):
    """从笔列表提取非重叠中枢"""
    zs_list = []
    i = 0
    while i < len(bis_list) - 2:
        b1, b2, b3 = bis_list[i], bis_list[i + 1], bis_list[i + 2]
        zg = min(b1.high, b2.high, b3.high)
        zd = max(b1.low, b2.low, b3.low)
        if zg > zd:
            zs = {"sdt": b1.sdt, "edt": b3.edt, "zg": zg, "zd": zd, "bis": 3}
            j = i + 3
            while j < len(bis_list):
                bj = bis_list[j]
                if bj.high >= zd and bj.low <= zg:
                    zs["edt"] = bj.edt
                    zs["bis"] += 1
                    j += 1
                else:
                    break
            zs_list.append(zs)
            i = j
        else:
            i += 1
    return zs_list


def _score_stock(bis, zs_list, dif_val, vol_ratio_prev, vol_ratio_now):
    """7 项加权打分，返回 (加权总分, 各项原始得分 dict)"""
    raw = {}

    if len(zs_list) >= 3:
        raw["C1_中枢上移"] = int(zs_list[-1]["zd"] > zs_list[-2]["zd"] > zs_list[-3]["zd"])
    else:
        raw["C1_中枢上移"] = 0

    if zs_list:
        width = (zs_list[-1]["zg"] - zs_list[-1]["zd"]) / zs_list[-1]["zd"]
        raw["C2_窄幅收敛"] = int(width < 0.10)
    else:
        raw["C2_窄幅收敛"] = 0

    up_bis = [b for b in bis if b.direction.value == "向上"]
    if len(up_bis) >= 3:
        recent = up_bis[-1].power
        prev_avg = np.mean([b.power for b in up_bis[-3:-1]])
        raw["C3_笔力突增"] = int(recent > prev_avg * 1.5) if prev_avg > 0 else 0
    else:
        raw["C3_笔力突增"] = 0

    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if len(dn_bis) >= 2:
        last_dn = dn_bis[-1]
        prev_dn = dn_bis[-2]
        last_pct = (last_dn.high / last_dn.low - 1)
        raw["C4_下笔递减"] = int(last_dn.power < prev_dn.power and last_pct < 0.12)
    else:
        raw["C4_下笔递减"] = 0

    if dif_val is not None:
        raw["C5_DIF零轴"] = int(abs(dif_val) < 0.5)
    else:
        raw["C5_DIF零轴"] = 0

    raw["C6_缩放量"] = int(vol_ratio_prev < 0.8 and vol_ratio_now > 1.2)

    if len(zs_list) >= 2 and len(dn_bis) >= 1:
        raw["C7_不回中枢"] = int(dn_bis[-1].low > zs_list[-2]["zg"])
    else:
        raw["C7_不回中枢"] = 0

    weighted_total = sum(w * v for w, v in zip(FEATURE_WEIGHTS, raw.values()))
    return weighted_total, raw


def _calc_stop_losses(close_val, bis, zs_list, atr20_val):
    """计算三种止损参考价"""
    stops = {}

    # 1. 缠论中枢止损：前一个中枢下沿 ZD
    if len(zs_list) >= 2:
        stops["止损_中枢ZD"] = round(zs_list[-2]["zd"], 2)
    elif zs_list:
        stops["止损_中枢ZD"] = round(zs_list[-1]["zd"], 2)
    else:
        stops["止损_中枢ZD"] = None

    # 2. 笔结构止损：最近向下笔的低点
    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if dn_bis:
        stops["止损_笔低点"] = round(dn_bis[-1].low, 2)
    else:
        stops["止损_笔低点"] = None

    # 3. 波动率止损：收盘价 - 2x ATR20
    if atr20_val is not None and atr20_val > 0:
        stops["止损_ATR"] = round(close_val - 2 * atr20_val, 2)
    else:
        stops["止损_ATR"] = None

    return stops


def _scan_one_stock(parquet_path: str) -> dict | None:
    """扫描单只股票最新一天的状态"""
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
    if len(bis) < 8:
        return None

    zs_list = _extract_zs(bis)
    if len(zs_list) < 2:
        return None

    close = df["close"].values
    vol = df["vol"].values
    n = len(df)
    idx = n - 1

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    vr_now = vol_ma5[idx] / vol_ma20[idx] if vol_ma20[idx] > 0 else 1.0
    vr_prev = vol_ma5[idx - 5] / vol_ma20[idx - 5] if idx >= 5 and vol_ma20[idx - 5] > 0 else 1.0

    # ATR20
    high_s = df["high"].values
    low_s = df["low"].values
    tr = np.maximum(high_s[1:] - low_s[1:],
                    np.maximum(np.abs(high_s[1:] - close[:-1]),
                               np.abs(low_s[1:] - close[:-1])))
    atr20 = float(np.mean(tr[-20:])) if len(tr) >= 20 else None

    total, detail = _score_stock(bis, zs_list, dif[idx], vr_prev, vr_now)

    if total < SCORE_THRESHOLD:
        return None

    pct_chg = df["pct_chg"].values[idx] if "pct_chg" in df.columns else 0
    if abs(pct_chg) >= 9.8:
        return None

    stops = _calc_stop_losses(close[idx], bis, zs_list, atr20)

    # 推荐止损 = SL2 笔结构止损
    rec_sl = stops.get("止损_笔低点")
    sl_pct = round((close[idx] - rec_sl) / close[idx] * 100, 1) if rec_sl and rec_sl > 0 else None

    # MA 状态
    ma5 = pd.Series(close).rolling(5).mean().values[idx]
    ma10 = pd.Series(close).rolling(10).mean().values[idx]
    ma20 = pd.Series(close).rolling(20).mean().values[idx]
    ma_bull = "多头" if ma5 > ma10 > ma20 else ""

    last_dt = df["dt"].iloc[-1].strftime("%Y-%m-%d")

    return {
        "代码": code,
        "名称": "",
        "行业": "",
        "日期": last_dt,
        "收盘价": round(close[idx], 2),
        "得分": round(total, 1),
        **detail,
        "MA状态": ma_bull,
        "DIF": round(dif[idx], 3),
        "量比": round(vr_now, 2),
        "推荐止损": rec_sl,
        "止损幅度%": sl_pct,
        **stops,
    }


def _load_stock_basic() -> tuple[dict, dict]:
    """通过 tinyshare 获取全 A 股名称和行业映射"""
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    name_map = dict(zip(basic["ts_code"], basic["name"]))
    industry_map = dict(zip(basic["ts_code"], basic["industry"]))
    return name_map, industry_map


def main():
    print("=" * 70)
    print("  主升前 CZSC 特征 — 每日选股扫描")
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

    # 填充名称和行业
    for r in results:
        code = r["代码"]
        r["名称"] = name_map.get(code, "")
        r["行业"] = industry_map.get(code, "")

    results.sort(key=lambda x: x["得分"], reverse=True)

    print(f"\n[完成] 耗时 {time.time()-t0:.0f}s | 符合条件 {len(results)} 只 (>= {SCORE_THRESHOLD} 分)")

    if not results:
        print("  今日无符合条件的标的")
        return

    scan_date = results[0]["日期"]

    # 打印结果表格
    print(f"\n{'='*150}")
    print(f"  {scan_date} 主升前特征选股 Top {min(len(results), 50)}  |  推荐止损：SL2 笔结构止损（回测卡玛 6.85）")
    print(f"{'='*150}")

    header = (f"{'序':>3} {'代码':>12} {'名称':<8} {'行业':<8} {'收盘价':>7} {'得分':>4} "
              f"{'C1':>3}{'C2':>3}{'C3':>3}{'C4':>3}{'C5':>3}{'C6':>3}{'C7':>3} "
              f"{'MA':>4} {'推荐止损':>8} {'幅度%':>6} "
              f"{'中枢止损':>8} {'ATR止损':>8}")
    print(header)
    print("-" * 150)

    for i, r in enumerate(results[:50], 1):
        c1 = r.get("C1_中枢上移", 0)
        c2 = r.get("C2_窄幅收敛", 0)
        c3 = r.get("C3_笔力突增", 0)
        c4 = r.get("C4_下笔递减", 0)
        c5 = r.get("C5_DIF零轴", 0)
        c6 = r.get("C6_缩放量", 0)
        c7 = r.get("C7_不回中枢", 0)

        rec_sl = r.get("推荐止损", "")
        sl_pct = r.get("止损幅度%", "")
        stop_zs = r.get("止损_中枢ZD", "")
        stop_atr = r.get("止损_ATR", "")

        rec_sl_s = f"{rec_sl:.2f}" if isinstance(rec_sl, (int, float)) and rec_sl else ""
        sl_pct_s = f"{sl_pct:.1f}" if isinstance(sl_pct, (int, float)) and sl_pct is not None else ""
        stop_zs_s = f"{stop_zs:.2f}" if isinstance(stop_zs, (int, float)) and stop_zs else ""
        stop_atr_s = f"{stop_atr:.2f}" if isinstance(stop_atr, (int, float)) and stop_atr else ""

        name = r.get("名称", "")[:6]
        industry = r.get("行业", "")[:6]

        print(f"{i:>3} {r['代码']:>12} {name:<8} {industry:<8} {r['收盘价']:>7.2f} {r['得分']:>4} "
              f"{c1:>3}{c2:>3}{c3:>3}{c4:>3}{c5:>3}{c6:>3}{c7:>3} "
              f"{r.get('MA状态',''):>4} {rec_sl_s:>8} {sl_pct_s:>6} "
              f"{stop_zs_s:>8} {stop_atr_s:>8}")

    # 保存 parquet
    out_parquet = OUTPUT_DIR / f"picks_{scan_date}.parquet"
    pd.DataFrame(results).to_parquet(out_parquet, index=False)
    print(f"\n[文件] {out_parquet}")


if __name__ == "__main__":
    main()
