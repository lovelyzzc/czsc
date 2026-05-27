"""
批量同步全部A股日线前复权数据（近5年）

使用 tinyshare SDK 下载数据，保存为 parquet 格式。
存储路径：~/.ts_data_cache/a_stock_daily_qfq/

用法：
    python scripts/sync_a_stock_daily.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import tinyshare as ts
from tqdm import tqdm

TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
ts.set_token(TOKEN)
pro = ts.pro_api()

SAVE_DIR = Path(os.getenv("TS_CACHE_PATH", os.path.expanduser("~/.ts_data_cache"))) / "a_stock_daily_qfq"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20210526"
END_DATE = "20260526"


def get_all_stocks():
    """获取全部A股上市股票列表"""
    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date")
    print(f"全部上市A股数量: {len(df)}")
    return df


def download_one(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """下载单只股票的日线前复权数据"""
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start_date,
            end_date=end_date,
            freq="D",
            asset="E",
        )
        if df is not None and not df.empty:
            df = df.sort_values("trade_date", ascending=True, ignore_index=True)
            return df
    except Exception as e:
        print(f"  [ERROR] {ts_code}: {e}")
    return None


def sync_all():
    """同步全部A股日线前复权数据"""
    stocks = get_all_stocks()
    ts_codes = stocks["ts_code"].tolist()

    success_count = 0
    fail_count = 0
    skip_count = 0

    for ts_code in tqdm(ts_codes, desc="同步A股日线(前复权)"):
        save_path = SAVE_DIR / f"{ts_code}.parquet"

        # 如果文件已存在且是今天更新的，跳过
        if save_path.exists():
            mtime = os.path.getmtime(save_path)
            if time.time() - mtime < 86400:
                skip_count += 1
                continue

        df = download_one(ts_code, START_DATE, END_DATE)
        if df is not None:
            df.to_parquet(save_path, index=False)
            success_count += 1
        else:
            fail_count += 1

        # 控制请求频率，避免触发限流
        time.sleep(0.3)

    print(f"\n同步完成！成功: {success_count}, 失败: {fail_count}, 跳过(已是最新): {skip_count}")
    print(f"数据保存目录: {SAVE_DIR}")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")


if __name__ == "__main__":
    sync_all()
