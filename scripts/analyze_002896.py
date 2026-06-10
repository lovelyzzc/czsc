"""002896.SZ 缠论结构分析 + 可视化（含中枢）

1. 从本地缓存读取历史日线，通过 tinyshare 补齐最新数据
2. 运行 CZSC 分析，输出分型/笔/中枢结构
3. 生成 lightweight-charts HTML
4. 打印走势分析摘要
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import tinyshare as ts

from czsc import CZSC, ZS, Freq, format_standard_kline
from czsc.utils.plotting.lightweight import plot_czsc

PARQUET = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq" / "002896.SZ.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")


def update_data() -> pd.DataFrame:
    """读取本地缓存 + 在线补齐最新日线"""
    df = pd.read_parquet(PARQUET)
    last_date = df["trade_date"].max()
    print(f"[缓存] 最新日期: {last_date}, 共 {len(df)} 条")

    ts.set_token(TOKEN)
    df_new = ts.pro_bar(
        ts_code="002896.SZ", adj="qfq",
        start_date=last_date, end_date="20260601",
        freq="D", asset="E",
    )
    if df_new is not None and not df_new.empty:
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["trade_date"], keep="last")
        df = df.sort_values("trade_date", ascending=True, ignore_index=True)
        df.to_parquet(PARQUET, index=False)
        print(f"[更新] 补齐至 {df['trade_date'].max()}, 共 {len(df)} 条")
    return df


def to_bars(df: pd.DataFrame):
    """Tushare 日线 DataFrame -> czsc RawBar 列表"""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["ts_code"]
    df["vol"] = (df["vol"] * 100).astype(int)
    df["amount"] = (df["amount"] * 1000).astype(float)
    df = df[["symbol", "dt", "open", "high", "low", "close", "vol", "amount"]]
    return format_standard_kline(df, freq=Freq.D)


def extract_zs_list(c: CZSC) -> list[ZS]:
    """从 bi_list 提取非重叠中枢序列"""
    bi_list = list(c.bi_list)
    result = []
    i = 0
    while i <= len(bi_list) - 3:
        try:
            zs = ZS(bi_list[i : i + 3])
        except Exception:
            i += 1
            continue
        if not zs.is_valid():
            i += 1
            continue
        j = i + 3
        zg, zd = zs.zg, zs.zd
        while j < len(bi_list):
            bi = bi_list[j]
            bi_high = max(float(bi.fx_a.fx), float(bi.fx_b.fx))
            bi_low = min(float(bi.fx_a.fx), float(bi.fx_b.fx))
            if bi_low < zg and bi_high > zd:
                j += 1
            else:
                break
        result.append(zs)
        i = j
    return result


def analyze(c: CZSC, df: pd.DataFrame):
    """打印缠论结构分析"""
    bi_list = list(c.bi_list)
    fx_list = list(c.fx_list)
    zs_list = extract_zs_list(c)

    print("\n" + "=" * 70)
    print("  002896.SZ 中大力德 · 日线缠论结构分析")
    print("=" * 70)

    # 基础信息
    print("\n【基础统计】")
    print(f"  数据范围: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
    print(f"  分型数量: {len(fx_list)}")
    print(f"  完成笔数: {len(bi_list)}")
    print(f"  中枢数量: {len(zs_list)}")
    last_price = float(df.iloc[-1]["close"])
    print(f"  最新收盘: {last_price:.2f}")

    # 最近几笔分析
    print("\n【最近 5 笔走势】")
    for bi in bi_list[-5:]:
        direction = "↑" if str(bi.direction) == "Direction.Up" else "↓"
        power = f"{bi.power:.2f}" if hasattr(bi, "power") else "N/A"
        high = max(float(bi.fx_a.fx), float(bi.fx_b.fx))
        low = min(float(bi.fx_a.fx), float(bi.fx_b.fx))
        change_pct = (high - low) / low * 100
        print(
            f"  {direction} {bi.fx_a.dt.strftime('%Y-%m-%d')} → {bi.fx_b.dt.strftime('%Y-%m-%d')}"
            f"  |  {low:.2f} ~ {high:.2f}  |  幅度 {change_pct:.1f}%  |  力度 {power}"
        )

    # 当前笔状态
    last_bi = bi_list[-1]
    is_up = str(last_bi.direction) == "Direction.Up"
    print("\n【当前笔状态】")
    print(f"  最后一笔方向: {'向上 ↑' if is_up else '向下 ↓'}")
    print(f"  起点: {last_bi.fx_a.dt.strftime('%Y-%m-%d')} @ {float(last_bi.fx_a.fx):.2f}")
    print(f"  终点: {last_bi.fx_b.dt.strftime('%Y-%m-%d')} @ {float(last_bi.fx_b.fx):.2f}")
    print(f"  是否在延伸中: {c.last_bi_extend}")

    # 未完成笔（ubi）
    if c.ubi_fxs:
        ubi_fxs = list(c.ubi_fxs)
        print("\n【未完成笔 (UBI)】")
        for fx in ubi_fxs:
            mark = "顶" if str(fx.mark) == "Mark.G" else "底"
            print(f"  {mark}分型 @ {fx.dt.strftime('%Y-%m-%d')}  价格 {float(fx.fx):.2f}")

    # 最近中枢
    if zs_list:
        print("\n【最近中枢】")
        for zs in zs_list[-3:]:
            bis_in_zs = list(zs.bis)
            print(
                f"  {zs.sdt.strftime('%Y-%m-%d')} ~ {zs.edt.strftime('%Y-%m-%d')}"
                f"  |  ZG={zs.zg:.2f}  ZD={zs.zd:.2f}  (区间 {zs.zg - zs.zd:.2f})"
                f"  |  GG={zs.gg:.2f}  DD={zs.dd:.2f}"
                f"  |  {len(bis_in_zs)}笔"
            )

        last_zs = zs_list[-1]
        if last_price > last_zs.zg:
            pos_vs_zs = f"在最近中枢上方 (+{last_price - last_zs.zg:.2f})"
        elif last_price < last_zs.zd:
            pos_vs_zs = f"在最近中枢下方 (-{last_zs.zd - last_price:.2f})"
        else:
            pos_vs_zs = f"在最近中枢区间内 (ZD={last_zs.zd:.2f} ~ ZG={last_zs.zg:.2f})"
        print(f"  当前价格相对位置: {pos_vs_zs}")

    # 近期量价
    recent = df.tail(10).copy()
    recent["pct_chg"] = recent["close"].pct_change() * 100
    avg_vol_20 = df.tail(20)["vol"].mean()
    latest_vol = float(df.iloc[-1]["vol"])
    vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 0

    print("\n【近期量价特征】")
    print("  最近10日涨跌:")
    for _, row in recent.iterrows():
        pct = row["pct_chg"]
        bar = "▓" * min(int(abs(pct) * 2), 20) if not pd.isna(pct) else ""
        sign = "+" if pct > 0 else ""
        pct_str = f"{sign}{pct:.2f}%" if not pd.isna(pct) else "  N/A"
        print(f"    {row['trade_date']}  收:{row['close']:7.2f}  {pct_str:>8}  {bar}")

    print(f"\n  最新成交量: {latest_vol:.0f} (20日均量: {avg_vol_20:.0f})")
    print(f"  量比: {vol_ratio:.2f}x {'(放量)' if vol_ratio > 1.5 else '(缩量)' if vol_ratio < 0.7 else '(正常)'}")

    # 均线位置
    closes = df["close"].values
    ma5 = closes[-5:].mean()
    ma20 = closes[-20:].mean()
    ma60 = closes[-60:].mean() if len(closes) >= 60 else None

    print("\n【均线系统】")
    print(f"  MA5  = {ma5:.2f}  {'✓ 价格在上' if last_price > ma5 else '✗ 价格在下'}")
    print(f"  MA20 = {ma20:.2f}  {'✓ 价格在上' if last_price > ma20 else '✗ 价格在下'}")
    if ma60:
        print(f"  MA60 = {ma60:.2f}  {'✓ 价格在上' if last_price > ma60 else '✗ 价格在下'}")
    if ma5 > ma20:
        print("  MA5 > MA20 → 短期均线多头排列")
    else:
        print("  MA5 < MA20 → 短期均线空头排列")

    # 综合判断
    print(f"\n{'=' * 70}")
    print("  综合走势判断")
    print("=" * 70)

    signals = []
    if is_up:
        signals.append("最后一笔向上")
    else:
        signals.append("最后一笔向下")

    if c.last_bi_extend:
        signals.append("当前笔仍在延伸")
    else:
        signals.append("当前笔已完成")

    if zs_list:
        if last_price > last_zs.zg:
            signals.append("价格突破最近中枢上沿")
        elif last_price < last_zs.zd:
            signals.append("价格跌破最近中枢下沿")
        else:
            signals.append("价格在最近中枢区间震荡")

    if vol_ratio > 1.5:
        signals.append("伴随放量")
    elif vol_ratio < 0.7:
        signals.append("缩量运行")

    if last_price > ma5 > ma20:
        signals.append("均线多头排列")
    elif last_price < ma5 < ma20:
        signals.append("均线空头排列")

    print(f"\n  技术信号: {' | '.join(signals)}")

    # 关键价位
    if zs_list:
        print("\n  关键价位:")
        print(f"    最近中枢上沿 (ZG): {last_zs.zg:.2f} — 突破确认则看多")
        print(f"    最近中枢下沿 (ZD): {last_zs.zd:.2f} — 跌破则看空")
        print(f"    最近中枢最高 (GG): {last_zs.gg:.2f}")
        print(f"    最近中枢最低 (DD): {last_zs.dd:.2f}")

    if len(bi_list) >= 2:
        prev_bi = bi_list[-2]
        prev_high = max(float(prev_bi.fx_a.fx), float(prev_bi.fx_b.fx))
        prev_low = min(float(prev_bi.fx_a.fx), float(prev_bi.fx_b.fx))
        print(f"    前一笔高点: {prev_high:.2f}")
        print(f"    前一笔低点: {prev_low:.2f}")


def main():
    df = update_data()
    bars = to_bars(df)
    print(f"\n[数据] 共 {len(bars)} 根日线 K 线")

    c = CZSC(bars)
    analyze(c, df)

    out_path = OUTPUT_DIR / "002896_czsc_daily.html"
    plot_czsc(
        c,
        output="html",
        path=out_path,
        title="002896.SZ 中大力德 · 日线缠论结构（含中枢）",
        theme="light",
        show_sma=(5, 20),
    )
    print(f"\n[图表] 已生成: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
