"""检测并修复 qfq 缓存中的价格接缝（增量同步追加未复权行导致）

背景：`_sync_daily_data.py` 把 `daily` 接口的未复权行追加到 `pro_bar(adj="qfq")`
下载的前复权历史上。若追加期间发生除权除息，历史段未重新前复权，序列在除权日
出现价格接缝，会扭曲笔力度 / MACD / 均线等所有依赖价格连续性的计算。

检测原理：一致的前复权序列满足 `close[i]/close[i-1] - 1 ≈ pct_chg[i]`
（pct_chg 是除权后真实涨跌幅）。偏差超过阈值的行即接缝。
阈值 1%：低价股两位小数舍入噪声 ≤~0.7%，1% 以下的微小分红对笔结构影响可忽略。

两类接缝及修法：
1. 追加导致：全量重下 pro_bar qfq 即可修复（--fix 第一阶段）；
2. 数据源自身缺陷：配股/缩股等除权事件 pre_close 已调整但 adj_factor 未覆盖，
   重下后接缝仍在 → 用 pre_close 链式修平（把接缝前的历史段乘以
   pre_close[i]/close[i-1]，等价于补做一次前复权；--fix 第二阶段自动执行）。

用法：
    python scripts/_repair_qfq_seams.py            # 仅检测
    python scripts/_repair_qfq_seams.py --fix      # 重下 + 链式修平 + 复扫断言清零
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
SEAM_THRESHOLD = 0.01  # 序列收益与 pct_chg 的绝对偏差阈值
SLEEP_SECONDS = 0.3


def find_seams(pq_path: Path) -> list[str]:
    """返回该 parquet 中接缝行的 trade_date 列表（无接缝返回空）。"""
    try:
        df = pd.read_parquet(pq_path, columns=["trade_date", "close", "pct_chg"])
    except Exception:
        return []
    if len(df) < 2:
        return []
    df = df.sort_values("trade_date", ignore_index=True)
    close = df["close"].to_numpy(dtype=float)
    pct = df["pct_chg"].to_numpy(dtype=float) / 100.0
    series_ret = close[1:] / close[:-1] - 1.0
    diff = abs(series_ret - pct[1:])
    bad = diff > SEAM_THRESHOLD
    return df["trade_date"].iloc[1:][bad].tolist()


def scan_all() -> dict[str, list[str]]:
    """扫描全部缓存文件，返回 {ts_code: [接缝日期...]}。"""
    affected = {}
    files = sorted(DATA_DIR.glob("*.parquet"))
    t0 = time.time()
    for i, pq in enumerate(files, 1):
        seams = find_seams(pq)
        if seams:
            affected[pq.stem] = seams
        if i % 1000 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] 受影响 {len(affected)} | {time.time() - t0:.0f}s")
    return affected


def flatten_seams(pq_path: Path) -> int:
    """pre_close 链式修平：接缝前历史段统一乘 pre_close[i]/close[i-1]。返回修平的接缝数。"""
    import numpy as np

    df = pd.read_parquet(pq_path).sort_values("trade_date", ignore_index=True)
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    pre_close = df["pre_close"].to_numpy(dtype=float)
    pct = df["pct_chg"].to_numpy(dtype=float) / 100.0

    adj = np.ones(n)
    cum = 1.0
    n_seam = 0
    for i in range(n - 1, 0, -1):
        adj[i] = cum
        series_ret = close[i] / close[i - 1] - 1.0
        if abs(series_ret - pct[i]) > SEAM_THRESHOLD and pre_close[i] > 0 and close[i - 1] > 0:
            cum *= pre_close[i] / close[i - 1]
            n_seam += 1
    adj[0] = cum
    if n_seam == 0:
        return 0

    # 行 i 乘数 adj[i] 含其后全部接缝的累积因子；pre_close 同乘 adj[i]：
    # 接缝处 new_close[i-1] = close[i-1]·adj[i]·(pre_close[i]/close[i-1]) = pre_close[i]·adj[i]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].to_numpy(dtype=float) * adj
    df["pre_close"] = pre_close * adj
    if "change" in df.columns:
        df["change"] = df["close"] - df["pre_close"]
    df.to_parquet(pq_path, index=False)
    return n_seam


def repair(ts_codes: list[str]) -> tuple[int, list[str]]:
    """对受影响代码全量重下 qfq 数据并覆盖写回，返回 (成功数, 失败代码)。"""
    from sync_a_stock_daily import download_one  # 延迟导入：import 时会 set_token

    end_date = datetime.now().strftime("%Y%m%d")
    ok, failed = 0, []
    for i, code in enumerate(ts_codes, 1):
        pq = DATA_DIR / f"{code}.parquet"
        local_cols = list(pd.read_parquet(pq).columns)
        df = download_one(code, "20210101", end_date)
        if df is None or df.empty:
            failed.append(code)
        else:
            cols = [c for c in local_cols if c in df.columns]
            df[cols].to_parquet(pq, index=False)
            ok += 1
        if i % 50 == 0 or i == len(ts_codes):
            print(f"  [{i}/{len(ts_codes)}] 重下成功 {ok} 失败 {len(failed)}")
        time.sleep(SLEEP_SECONDS)
    return ok, failed


def main(argv: list[str]) -> int:
    do_fix = "--fix" in argv
    print(f"[扫描] {DATA_DIR}")
    affected = scan_all()
    print(f"\n[结果] {len(affected)} 只存在接缝")
    if affected:
        sample = list(affected.items())[:10]
        for code, dates in sample:
            print(f"  {code}: {dates[:5]}{' ...' if len(dates) > 5 else ''}")

    if not affected or not do_fix:
        if affected:
            print("\n加 --fix 重下受影响代码")
        return 0 if not affected else 1

    print(f"\n[修复·一] 全量重下 {len(affected)} 只 ...")
    ok, failed = repair(sorted(affected))
    print(f"[修复·一完成] 成功 {ok} | 失败 {failed if failed else 0}")

    print("\n[复扫] 检查重下后残余（数据源自身缺陷）...")
    residual = {k: v for k, v in scan_all().items() if k not in failed}
    if residual:
        print(f"\n[修复·二] pre_close 链式修平 {len(residual)} 只 ...")
        n_flat = sum(flatten_seams(DATA_DIR / f"{code}.parquet") for code in residual)
        print(f"[修复·二完成] 修平 {n_flat} 处接缝")

    print("\n[复扫] 验证接缝清零 ...")
    remaining = {k: v for k, v in scan_all().items() if k not in failed}
    if remaining:
        print(f"[FAIL] 仍有 {len(remaining)} 只存在接缝: {list(remaining)[:10]}")
        return 1
    print("[OK] 接缝清零（失败代码除外）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
