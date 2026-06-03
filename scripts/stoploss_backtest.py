"""三种止损方式回测对比

使用与 pre_surge_screener 相同的入场信号（7项打分 >= 5），
对比四种退出/止损方式的效果：

  SL0 无止损（基准）  — 固定持有 40 个交易日
  SL1 缠论中枢止损    — 价格跌破前一中枢 ZD 退出
  SL2 笔结构止损      — 价格跌破最近向下笔低点退出
  SL3 波动率止损      — 价格跌破 (入场价 - 2×ATR20) 退出

所有止损策略均设 40 日最大持有期。
数据源：~/.ts_data_cache/a_stock_daily_qfq/
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from wbt import generate_backtest_report

from czsc import CZSC, Freq, WeightBacktest, format_standard_kline

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "stoploss_cmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
MIN_BARS = 500
FEE_RATE = 0.0002
SCORE_THRESHOLD = 5
MAX_HOLD_DAYS = 40


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
    """7 项打分"""
    scores = {}

    if len(zs_list) >= 3:
        scores["C1"] = int(zs_list[-1]["zd"] > zs_list[-2]["zd"] > zs_list[-3]["zd"])
    else:
        scores["C1"] = 0

    if zs_list:
        width = (zs_list[-1]["zg"] - zs_list[-1]["zd"]) / zs_list[-1]["zd"]
        scores["C2"] = int(width < 0.10)
    else:
        scores["C2"] = 0

    up_bis = [b for b in bis if b.direction.value == "向上"]
    if len(up_bis) >= 3:
        recent = up_bis[-1].power
        prev_avg = np.mean([b.power for b in up_bis[-3:-1]])
        scores["C3"] = int(recent > prev_avg * 1.5) if prev_avg > 0 else 0
    else:
        scores["C3"] = 0

    dn_bis = [b for b in bis if b.direction.value == "向下"]
    if len(dn_bis) >= 2:
        last_dn = dn_bis[-1]
        prev_dn = dn_bis[-2]
        last_pct = (last_dn.high / last_dn.low - 1)
        scores["C4"] = int(last_dn.power < prev_dn.power and last_pct < 0.12)
    else:
        scores["C4"] = 0

    if dif_val is not None:
        scores["C5"] = int(abs(dif_val) < 0.5)
    else:
        scores["C5"] = 0

    scores["C6"] = int(vol_ratio_prev < 0.8 and vol_ratio_now > 1.2)

    if len(zs_list) >= 2 and len(dn_bis) >= 1:
        scores["C7"] = int(dn_bis[-1].low > zs_list[-2]["zg"])
    else:
        scores["C7"] = 0

    return sum(scores.values()), scores


def _process_stock(parquet_path: str) -> dict | None:
    """单只股票：生成入场信号 + 四种止损方式的 holds 和 pairs"""
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
    if len(zs_list) < 3:
        return None

    close = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    vol = df["vol"].values
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(len(df))
    dates = df["dt"].values
    symbol = code
    n = len(df)

    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26

    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values

    # 预计算 ATR20 序列
    tr = np.zeros(n)
    for k in range(1, n):
        tr[k] = max(high_arr[k] - low_arr[k],
                    abs(high_arr[k] - close[k - 1]),
                    abs(low_arr[k] - close[k - 1]))
    atr20 = pd.Series(tr).rolling(20).mean().values

    start_idx = max(120, n // 4)

    # 生成买入信号
    signals = []
    for idx in range(start_idx, n):
        dt = dates[idx]
        bis_up_to = [bi for bi in bis if bi.edt <= pd.Timestamp(dt)]
        if len(bis_up_to) < 6:
            continue

        zs_up_to = _extract_zs(bis_up_to)
        if len(zs_up_to) < 3:
            continue

        vr_now = vol_ma5[idx] / vol_ma20[idx] if vol_ma20[idx] > 0 else 1.0
        vr_prev = vol_ma5[idx - 5] / vol_ma20[idx - 5] if idx >= 5 and vol_ma20[idx - 5] > 0 else 1.0

        total, _ = _score_stock(bis_up_to, zs_up_to, dif[idx], vr_prev, vr_now)

        if total >= SCORE_THRESHOLD and abs(pct_chg[idx]) < 9.8:
            # 计算三种止损价
            dn_bis_up = [b for b in bis_up_to if b.direction.value == "向下"]

            sl_zs = zs_up_to[-2]["zd"] if len(zs_up_to) >= 2 else None
            sl_bi = dn_bis_up[-1].low if dn_bis_up else None
            sl_atr = close[idx] - 2 * atr20[idx] if not np.isnan(atr20[idx]) else None

            signals.append({
                "dt": pd.Timestamp(dt),
                "idx": idx,
                "close": close[idx],
                "sl_zs": sl_zs,
                "sl_bi": sl_bi,
                "sl_atr": sl_atr,
            })

    if not signals:
        return None

    # 去重：两信号间距 >= 10 天
    filtered = [signals[0]]
    for s in signals[1:]:
        if s["idx"] - filtered[-1]["idx"] >= 10:
            filtered.append(s)
    signals = filtered

    # 四种止损策略生成 holds 和 pairs
    result = {"symbol": symbol, "n_signals": len(signals)}

    sl_configs = {
        "SL0_无止损": lambda sig: None,
        "SL1_中枢止损": lambda sig: sig["sl_zs"],
        "SL2_笔低止损": lambda sig: sig["sl_bi"],
        "SL3_ATR止损": lambda sig: sig["sl_atr"],
    }

    for tag, get_sl in sl_configs.items():
        holds = []
        pairs = []
        for sig in signals:
            entry_idx = sig["idx"]
            entry_price = sig["close"]
            sl_price = get_sl(sig)

            exit_price = None
            exit_reason = "max_hold"

            for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, n)):
                holds.append({
                    "dt": pd.Timestamp(dates[j]),
                    "symbol": symbol,
                    "pos": 1,
                    "price": close[j],
                })
                if sl_price is not None and low_arr[j] <= sl_price and j > entry_idx:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                    break

            if exit_price is None:
                last_j = min(entry_idx + MAX_HOLD_DAYS - 1, n - 1)
                exit_price = close[last_j]

            ret = (exit_price / entry_price - 1) * 100
            pairs.append({
                "symbol": symbol,
                "entry_dt": str(sig["dt"]),
                "entry_price": entry_price,
                "exit_price": round(exit_price, 2),
                "ret_pct": round(ret, 2),
                "exit_reason": exit_reason,
                "sl_price": round(sl_price, 2) if sl_price else None,
            })

        result[f"holds_{tag}"] = holds
        result[f"pairs_{tag}"] = pairs

    return result


def main():
    print("=" * 70)
    print("  三种止损方式回测对比")
    print("=" * 70)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[数据] {len(parquet_files)} 只个股")

    n_workers = min(mp.cpu_count(), 8)
    print(f"[并行] {n_workers} 进程")

    t0 = time.time()
    file_list = [str(p) for p in parquet_files]

    all_results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_stock, file_list, chunksize=20), 1):
            if res is not None:
                all_results.append(res)
            if i % 500 == 0 or i == len(file_list):
                elapsed = time.time() - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - i) / speed if speed > 0 else 0
                print(f"  [{i}/{len(file_list)}] 有信号 {len(all_results)} | "
                      f"{elapsed:.0f}s | ETA {eta:.0f}s")

    print(f"\n[完成] 耗时 {time.time()-t0:.0f}s | {len(all_results)} 只股票产生信号")

    if not all_results:
        print("[ERROR] 无股票产生买入信号")
        return

    total_signals = sum(r["n_signals"] for r in all_results)
    print(f"  总买入信号数: {total_signals}")

    tags = ["SL0_无止损", "SL1_中枢止损", "SL2_笔低止损", "SL3_ATR止损"]
    all_stats = []

    for tag in tags:
        print(f"\n{'='*60}")
        print(f"  [{tag}]")
        print(f"{'='*60}")

        holds_key = f"holds_{tag}"
        pairs_key = f"pairs_{tag}"

        all_holds = []
        all_pairs = []
        for r in all_results:
            all_holds.extend(r[holds_key])
            all_pairs.extend(r[pairs_key])

        if not all_holds:
            print("  无持仓数据")
            continue

        # 止损触发统计
        n_sl = sum(1 for p in all_pairs if p["exit_reason"] == "stop_loss")
        n_max = sum(1 for p in all_pairs if p["exit_reason"] == "max_hold")

        dfw = pd.DataFrame(all_holds)
        dfw = dfw.rename(columns={"pos": "weight"})
        if dfw.duplicated(subset=["dt", "symbol"]).any():
            dfw = dfw.groupby(["dt", "symbol"], as_index=False).agg(
                weight=("weight", "max"), price=("price", "first"),
            )
        dfw = dfw[["dt", "symbol", "weight", "price"]]

        try:
            wb = WeightBacktest(data=dfw, fee_rate=FEE_RATE, weight_type="ts", yearly_days=252)
            stats = wb.stats
        except Exception as e:
            print(f"  WeightBacktest 失败: {e}")
            continue

        stats["tag"] = tag

        rets = [p["ret_pct"] for p in all_pairs]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]

        stats["交易笔数"] = len(rets)
        stats["止损触发"] = n_sl
        stats["到期退出"] = n_max
        stats["止损率"] = f"{n_sl/len(rets)*100:.1f}%" if rets else "0%"
        stats["胜率"] = f"{len(wins)/len(rets)*100:.1f}%" if rets else "0%"
        stats["平均盈利"] = f"{np.mean(wins):.2f}%" if wins else "0%"
        stats["平均亏损"] = f"{np.mean(losses):.2f}%" if losses else "0%"
        stats["盈亏比"] = round(abs(np.mean(wins) / np.mean(losses)), 2) if losses and wins else 0
        stats["平均收益"] = f"{np.mean(rets):.2f}%"
        stats["收益中位数"] = f"{np.median(rets):.2f}%"

        # 止损笔中亏损统计
        sl_rets = [p["ret_pct"] for p in all_pairs if p["exit_reason"] == "stop_loss"]
        if sl_rets:
            stats["止损平均亏损"] = f"{np.mean(sl_rets):.2f}%"
            stats["止损最大亏损"] = f"{np.min(sl_rets):.2f}%"

        for k in ["交易笔数", "止损触发", "到期退出", "止损率",
                   "胜率", "盈亏比", "平均盈利", "平均亏损", "平均收益", "收益中位数",
                   "止损平均亏损", "止损最大亏损",
                   "年化收益", "夏普比率", "最大回撤", "卡玛比率"]:
            if k in stats:
                print(f"    {k}: {stats[k]}")

        try:
            out_html = OUTPUT_DIR / f"{tag}.html"
            generate_backtest_report(
                df=dfw, output_path=str(out_html),
                title=f"止损回测 - {tag}",
                fee_rate=FEE_RATE, weight_type="ts", yearly_days=252,
            )
            print(f"    HTML: {out_html.name}")
        except Exception as e:
            print(f"    HTML 报告失败: {e}")

        all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] 所有策略均无结果")
        return

    # 输出对比表格
    cmp = pd.DataFrame(all_stats).set_index("tag")
    print("\n\n" + "=" * 100)
    print("  三种止损方式回测对比 — 总览")
    print("=" * 100)
    display_cols = [c for c in [
        "交易笔数", "止损触发", "止损率", "胜率", "盈亏比",
        "平均盈利", "平均亏损", "平均收益", "收益中位数",
        "年化收益", "夏普比率", "最大回撤", "卡玛比率",
    ] if c in cmp.columns]
    print(cmp[display_cols].to_string())

    # 生成汇总 HTML 报告
    _generate_summary_html(all_stats, all_results)

    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[文件] {OUTPUT_DIR}")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s")


def _generate_summary_html(all_stats: list[dict], all_results: list[dict]):
    """生成汇总对比 HTML 页面"""
    rows_html = ""
    for s in all_stats:
        tag = s.get("tag", "")
        rows_html += f"""<tr>
  <td><strong>{tag}</strong></td>
  <td>{s.get('交易笔数','')}</td>
  <td>{s.get('止损触发','')}</td>
  <td>{s.get('止损率','')}</td>
  <td>{s.get('胜率','')}</td>
  <td>{s.get('盈亏比','')}</td>
  <td>{s.get('平均盈利','')}</td>
  <td>{s.get('平均亏损','')}</td>
  <td>{s.get('平均收益','')}</td>
  <td>{s.get('收益中位数','')}</td>
  <td>{s.get('年化收益','')}</td>
  <td>{s.get('夏普比率','')}</td>
  <td>{s.get('最大回撤','')}</td>
  <td>{s.get('卡玛比率','')}</td>
</tr>"""

    links_html = ""
    for s in all_stats:
        tag = s.get("tag", "")
        links_html += f'<a href="{tag}.html" class="btn">{tag} 详细报告</a>\n'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>三种止损方式回测对比</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2em auto; max-width: 1400px; background: #f8f9fa; }}
h1 {{ text-align: center; color: #333; }}
table {{ width: 100%; border-collapse: collapse; margin: 1.5em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
th, td {{ padding: 10px 12px; text-align: center; border-bottom: 1px solid #e9ecef; }}
th {{ background: #343a40; color: white; font-weight: 500; }}
tr:hover {{ background: #f1f3f5; }}
.btn {{ display: inline-block; padding: 8px 20px; margin: 6px; border-radius: 5px; background: #228be6; color: white; text-decoration: none; }}
.btn:hover {{ background: #1c7ed6; }}
.links {{ text-align: center; margin: 2em 0; }}
.note {{ color: #666; text-align: center; margin-top: 1em; font-size: 0.9em; }}
.highlight {{ background: #d4edda !important; }}
</style></head><body>
<h1>三种止损方式回测对比</h1>
<p class="note">入场条件：主升前 CZSC 7 项特征 &ge; 5 分 | 最大持有 {MAX_HOLD_DAYS} 天 | 手续费 {FEE_RATE*10000:.0f}‱</p>

<table>
<tr>
  <th>策略</th><th>交易笔数</th><th>止损触发</th><th>止损率</th>
  <th>胜率</th><th>盈亏比</th><th>平均盈利</th><th>平均亏损</th>
  <th>平均收益</th><th>收益中位数</th>
  <th>年化收益</th><th>夏普比率</th><th>最大回撤</th><th>卡玛比率</th>
</tr>
{rows_html}
</table>

<h2 style="text-align:center">止损方式说明</h2>
<table style="max-width:900px;margin:1em auto">
<tr><th>策略</th><th>止损条件</th><th>说明</th></tr>
<tr><td>SL0 无止损</td><td>无</td><td>固定持有 {MAX_HOLD_DAYS} 天，作为基准对比</td></tr>
<tr><td>SL1 中枢止损</td><td>价格跌破前一中枢 ZD</td><td>缠论结构破坏，趋势上移逻辑失效</td></tr>
<tr><td>SL2 笔低止损</td><td>价格跌破最近向下笔低点</td><td>新低出现，上升笔结构被打破</td></tr>
<tr><td>SL3 ATR止损</td><td>价格跌破 (入场价 - 2×ATR20)</td><td>波动率止损，适合短线风控</td></tr>
</table>

<div class="links">
{links_html}
</div>
</body></html>"""

    with open(OUTPUT_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    汇总页: index.html")


if __name__ == "__main__":
    main()
