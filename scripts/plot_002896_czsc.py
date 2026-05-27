"""为 002896.SZ（中大力德）生成日线缠论结构图（自包含 HTML）

读取本地缓存的前复权日线数据，使用 czsc 原生的 lightweight-charts 可视化：
- 主图：K线 + SMA5 + SMA20 + 分型 marker + 笔 zigzag
- 副图1：成交量
- 副图2：MACD
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from czsc import CZSC, Freq, format_standard_kline
from czsc.utils.plotting.lightweight import plot_czsc

PARQUET = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq" / "002896.SZ.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_bars():
    df = pd.read_parquet(PARQUET)
    df = df.sort_values("trade_date", ascending=True, ignore_index=True)
    df["dt"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["ts_code"]
    df["vol"] = (df["vol"] * 100).astype(int)
    df["amount"] = (df["amount"] * 1000).astype(float)
    df = df[["symbol", "dt", "open", "high", "low", "close", "vol", "amount"]].copy()
    return format_standard_kline(df, freq=Freq.D)


def main():
    bars = load_daily_bars()
    print(f"[数据] 002896.SZ 日线 共 {len(bars)} 根, {bars[0].dt} ~ {bars[-1].dt}")

    c = CZSC(bars)
    print(f"[CZSC] 分型 {len(c.fx_list)} 个, 完成笔 {len(c.bi_list)} 条")

    out_path = OUTPUT_DIR / "002896_czsc_daily.html"
    plot_czsc(
        c,
        output="html",
        path=out_path,
        title="002896.SZ 中大力德 · 日线缠论结构",
        theme="light",
        show_sma=(5, 20),
    )
    print(f"[输出] {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print("双击 HTML 文件用浏览器打开即可查看。")


if __name__ == "__main__":
    main()
