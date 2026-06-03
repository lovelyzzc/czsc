"""批量同步 A 股日线前复权数据。

使用 tinyshare SDK 下载数据，保存为 parquet 格式。
默认存储路径：~/.ts_data_cache/a_stock_daily_qfq/

用法：
    python scripts/sync_a_stock_daily.py --start-date 20210101 --end-date 20260528
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tinyshare as ts
from tqdm import tqdm

TOKEN = os.getenv("TINYSHARE_TOKEN", "8mgRs242h2Bc3mADa8Pfh8YAfZf6ym4vYli84P4uMJb9v5QaKbW5l05sa286040b")
ts.set_token(TOKEN)
pro = ts.pro_api()

DEFAULT_START_DATE = "20210101"
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_SAVE_DIR = Path(os.getenv("TS_CACHE_PATH", os.path.expanduser("~/.ts_data_cache"))) / "a_stock_daily_qfq"
DEFAULT_SLEEP_SECONDS = float(os.getenv("SYNC_A_STOCK_SLEEP_SECONDS", "0.3"))
DAILY_BAR_READY_HOUR = int(os.getenv("SYNC_A_STOCK_DAILY_BAR_READY_HOUR", "18"))


def get_all_stocks(list_status: str = "L") -> pd.DataFrame:
    """获取全部A股上市股票列表"""
    frames = []
    for status in [x.strip() for x in list_status.split(",") if x.strip()]:
        df = pro.stock_basic(exchange="", list_status=status, fields="ts_code,symbol,name,area,industry,list_date")
        df["list_status"] = status
        frames.append(df)

    stocks = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    print(f"全部A股数量: {len(stocks)}; list_status={list_status}")
    return stocks.sort_values("ts_code", ignore_index=True)


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
            if "trade_date" not in df.columns:
                print(f"  [WARN] {ts_code}: 返回数据缺少 trade_date，跳过")
                return None
            df = df.sort_values("trade_date", ascending=True, ignore_index=True)
            return df
    except Exception as e:
        print(f"  [ERROR] {ts_code}: {e}")
    return None


def normalize_trade_date(value: object) -> str:
    """统一 trade_date 表示，兼容 parquet 统计值中的日期/字符串格式。"""
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return text
    return pd.to_datetime(text).strftime("%Y%m%d")


def get_expected_end_date(end_date: str) -> str:
    """获取 end_date 当天或之前的最近一个已闭合 A 股日线交易日。"""
    try:
        now = datetime.now()
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        if end_dt.date() > now.date():
            end_dt = now
        if end_dt.date() == now.date() and now.hour < DAILY_BAR_READY_HOUR:
            end_dt = end_dt - timedelta(days=1)
        effective_end_date = end_dt.strftime("%Y%m%d")
        start_date = (end_dt - timedelta(days=370)).strftime("%Y%m%d")
        cal = pro.trade_cal(exchange="", start_date=start_date, end_date=effective_end_date)
        if cal is None or cal.empty:
            return effective_end_date

        if "is_open" in cal.columns:
            cal = cal[cal["is_open"].astype(str).isin(["1", "True", "true"])]
        if cal.empty or "cal_date" not in cal.columns:
            return effective_end_date
        return normalize_trade_date(cal["cal_date"].max())
    except Exception as e:
        print(f"  [WARN] 获取交易日历失败，按请求结束日判断: {e}")
        return end_date


def get_suspended_symbols(trade_date: str) -> set[str]:
    """获取指定交易日停牌的股票代码集合。"""
    try:
        df = pro.suspend_d(trade_date=trade_date)
        if df is None or df.empty or "ts_code" not in df.columns:
            return set()
        return set(df["ts_code"].dropna().astype(str))
    except Exception as e:
        print(f"  [WARN] 获取停牌列表失败，按实际K线覆盖判断: {e}")
        return set()


def read_trade_date_bounds(path: Path) -> tuple[str, str] | None:
    """从 parquet footer 快速读取 trade_date 的首尾范围。"""
    try:
        parquet_file = pq.ParquetFile(path)
        column_index = parquet_file.schema_arrow.get_field_index("trade_date")
        if column_index < 0:
            return None

        min_dates = []
        max_dates = []
        for row_group_index in range(parquet_file.num_row_groups):
            column = parquet_file.metadata.row_group(row_group_index).column(column_index)
            stats = column.statistics
            if stats is None or not stats.has_min_max:
                return None
            min_dates.append(str(stats.min))
            max_dates.append(str(stats.max))

        if not min_dates or not max_dates:
            return None
        return normalize_trade_date(min(min_dates)), normalize_trade_date(max(max_dates))
    except Exception:
        return None


def file_covers_range(
    path: Path,
    start_date: str,
    end_date: str,
    list_date: str | None = None,
    suspended_on_end: bool = False,
) -> bool:
    """判断已有缓存是否覆盖请求区间。

    起始日期允许节假日和新股上市带来的温和偏差；结束日期必须覆盖最近一个交易日，避免
    中断续跑或盘中半成品文件因为 mtime 是当天而被误判为已完成。
    """
    if not path.exists():
        return False

    bounds = read_trade_date_bounds(path)
    if bounds is None:
        try:
            df = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            return False

        if df.empty:
            return False
        first_value = str(df["trade_date"].min())
        last_value = str(df["trade_date"].max())
    else:
        first_value, last_value = bounds

    first_date = pd.to_datetime(first_value)
    last_date = pd.to_datetime(last_value)
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)
    expected_start = requested_start
    if list_date:
        expected_start = max(expected_start, pd.to_datetime(str(list_date)))

    # 请求起点可能是节假日；新股上市后也可能停牌。给首条交易日一个温和容忍窗口。
    starts_near_expected = first_date <= expected_start or (first_date - expected_start).days <= 31
    ends_near_expected = last_date >= requested_end or suspended_on_end
    return starts_near_expected and ends_near_expected


def write_manifest(save_dir: Path, payload: dict) -> None:
    manifest = save_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def next_date(date_str: str) -> str:
    return (pd.to_datetime(str(date_str)) + timedelta(days=1)).strftime("%Y%m%d")


def merge_incremental_cache(path: Path, new_data: pd.DataFrame) -> pd.DataFrame:
    """合并增量数据；同一交易日保留新下载记录。"""
    if not path.exists():
        return new_data

    old_data = pd.read_parquet(path)
    if old_data.empty:
        return new_data
    merged = pd.concat([old_data, new_data], ignore_index=True)
    return (
        merged.drop_duplicates(subset=["trade_date"], keep="last")
        .sort_values("trade_date", ascending=True, ignore_index=True)
    )


def sync_all(
    start_date: str,
    end_date: str,
    save_dir: Path,
    list_status: str = "L",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    incremental: bool = False,
) -> dict:
    """同步全部A股日线前复权数据"""
    save_dir.mkdir(parents=True, exist_ok=True)
    stocks = get_all_stocks(list_status=list_status)
    stock_records = stocks.to_dict("records")
    expected_end_date = get_expected_end_date(end_date)
    if expected_end_date != end_date:
        print(f"请求结束日 {end_date} 非交易日，按最近交易日 {expected_end_date} 判断覆盖")
    suspended_symbols = get_suspended_symbols(expected_end_date)
    if suspended_symbols:
        print(f"{expected_end_date} 停牌股票数量: {len(suspended_symbols)}")

    success_count = 0
    fail_count = 0
    skip_count = 0
    suspended_skip_count = 0
    incremental_count = 0
    failures = []

    for stock in tqdm(stock_records, desc="同步A股日线(前复权)"):
        ts_code = stock["ts_code"]
        save_path = save_dir / f"{ts_code}.parquet"

        if file_covers_range(
            save_path,
            start_date=start_date,
            end_date=expected_end_date,
            list_date=stock.get("list_date"),
            suspended_on_end=ts_code in suspended_symbols,
        ):
            skip_count += 1
            if ts_code in suspended_symbols:
                suspended_skip_count += 1
            continue

        fetch_start_date = start_date
        if incremental and save_path.exists():
            bounds = read_trade_date_bounds(save_path)
            if bounds is not None:
                fetch_start_date = max(start_date, next_date(bounds[1]))
                if fetch_start_date > expected_end_date:
                    skip_count += 1
                    continue
                incremental_count += 1

        df = download_one(ts_code, fetch_start_date, end_date)
        if df is not None:
            if incremental:
                df = merge_incremental_cache(save_path, df)
            df.to_parquet(save_path, index=False)
            success_count += 1
        else:
            fail_count += 1
            failures.append(ts_code)

        # 控制请求频率，避免触发限流
        time.sleep(sleep_seconds)

    manifest = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": "tinyshare.pro_bar",
        "asset": "E",
        "freq": "D",
        "adjust": "qfq",
        "start_date": start_date,
        "end_date": end_date,
        "expected_trade_end_date": expected_end_date,
        "list_status": list_status,
        "total_symbols": len(stock_records),
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "incremental": incremental,
        "incremental_count": incremental_count,
        "suspended_symbols_count": len(suspended_symbols),
        "suspended_skip_count": suspended_skip_count,
        "save_dir": str(save_dir),
        "failures": failures,
    }
    write_manifest(save_dir, manifest)

    print(f"\n同步完成！成功: {success_count}, 失败: {fail_count}, 跳过(已覆盖): {skip_count}")
    print(f"数据保存目录: {save_dir}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"Manifest: {save_dir / 'manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步全部 A 股日线前复权 parquet 缓存")
    parser.add_argument("--start-date", default=os.getenv("SYNC_A_STOCK_START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.getenv("SYNC_A_STOCK_END_DATE", DEFAULT_END_DATE))
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--list-status",
        default=os.getenv("SYNC_A_STOCK_LIST_STATUS", "L"),
        help="股票状态，默认 L。可用逗号传多个状态，如 L,D 用于包含退市股票。",
    )
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="只补每只股票本地最后交易日之后的数据；速度更快，但前复权历史一致性不如全量刷新。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_all(
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=args.save_dir.expanduser(),
        list_status=args.list_status,
        sleep_seconds=args.sleep_seconds,
        incremental=args.incremental,
    )
