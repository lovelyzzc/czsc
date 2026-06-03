"""批量同步全 A 股日线数据（增量更新）

使用 daily 接口按交易日批量拉取全市场数据，而非逐只股票调用 pro_bar。
每天 ~5500 只股票只需 1 次 API 调用，4 天增量仅需 4 次调用。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import tinyshare as ts

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
TOKEN = "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b"


def get_trade_dates_to_sync() -> list[str]:
    """找出需要补齐的交易日"""
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"本地无数据文件: {DATA_DIR}")

    sample = pd.read_parquet(files[0], columns=["trade_date"])
    local_max = sample["trade_date"].max()
    for pq in files[1:10]:
        d = pd.read_parquet(pq, columns=["trade_date"])["trade_date"].max()
        if d > local_max:
            local_max = d

    print(f"本地数据截止日: {local_max}")

    ts.set_token(TOKEN)
    pro = ts.pro_api()
    cal = pro.query("trade_cal", exchange="SSE", start_date=local_max, end_date="20260630",
                    fields="cal_date,is_open")
    trade_days = cal[(cal["is_open"] == 1) & (cal["cal_date"] > local_max)]["cal_date"].sort_values().tolist()
    return trade_days


def sync_one_day(trade_date: str, pro) -> int:
    """拉取单个交易日全市场数据并追加到各 parquet 文件"""
    t0 = time.time()
    df = pro.query("daily", trade_date=trade_date)
    if df is None or df.empty:
        print(f"  {trade_date}: 无数据（非交易日或尚未发布）")
        return 0

    print(f"  {trade_date}: 拉取 {len(df)} 只, API耗时 {time.time()-t0:.1f}s", end="")

    updated = 0
    t1 = time.time()
    for ts_code, group in df.groupby("ts_code"):
        pq = DATA_DIR / f"{ts_code}.parquet"
        if not pq.exists():
            continue

        try:
            local = pd.read_parquet(pq)
        except Exception:
            continue

        existing_dates = set(local["trade_date"].values)
        new_rows = group[~group["trade_date"].isin(existing_dates)]
        if new_rows.empty:
            continue

        cols = [c for c in local.columns if c in new_rows.columns]
        new_rows = new_rows[cols]

        combined = pd.concat([local, new_rows], ignore_index=True)
        combined = combined.sort_values("trade_date", ascending=True, ignore_index=True)
        combined.to_parquet(pq, index=False)
        updated += 1

    print(f" → 更新 {updated} 个文件, 写入耗时 {time.time()-t1:.1f}s")
    return updated


def main():
    print("=" * 60)
    print("  全 A 股日线数据增量同步")
    print("=" * 60)

    trade_dates = get_trade_dates_to_sync()
    if not trade_dates:
        print("数据已是最新，无需同步")
        return

    print(f"需要同步 {len(trade_dates)} 个交易日: {trade_dates}")

    ts.set_token(TOKEN)
    pro = ts.pro_api()

    t0 = time.time()
    total_updated = 0
    for td in trade_dates:
        n = sync_one_day(td, pro)
        total_updated += n

    print(f"\n[完成] 总耗时 {time.time()-t0:.1f}s | 同步 {len(trade_dates)} 天 | 更新 {total_updated} 次")


if __name__ == "__main__":
    main()
