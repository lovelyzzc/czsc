"""终态策略每日筛选：anticipate + delay5 存活确认 + 市场状态门（薄包装，零新选股逻辑）

选股逻辑 100% 复用 surge-regime-stock-picker 的 daily_scan：同一 `_scan_one` /
`_report_experimental` 代码路径（与研究镜像逐字节一致，轮一 239/239 核对），
共用同一前向日志（picks_exp_delay5_*.parquet / market_state_live.parquet），
重复运行幂等。本脚本仅新增：前向转正进度读数 + 组合口径提示（广度优先 30 槽）。

阈值全部固定（不吃 SURGE_PICKER_* 环境变量）：市场门 high20_ratio>0.12 &
等权指数>MA20；硬过滤 成交额≥1亿 + 止损带 8-20% + 剔除 ST。

    PYTHONUNBUFFERED=1 .venv/bin/python .cursor/skills/surge-delay5-stock-picker/scripts/delay5_scan.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
LEGACY_SCRIPTS = REPO / ".cursor" / "skills" / "surge-regime-stock-picker" / "scripts"
sys.path.insert(0, str(LEGACY_SCRIPTS))
sys.path.insert(0, str(REPO / "scripts"))

import daily_scan as legacy  # noqa: E402
import trend_regime as tr  # noqa: E402

FORWARD_TARGET = 60  # 预声明转正标准：≥60 笔前向样本、超额 t≥2 且中位数>0


def _forward_progress() -> dict:
    """累计前向样本：扫描全部 picks_exp_delay5_*.parquet，按（代码, dec_dt）去重。"""
    files = sorted(legacy.OUTPUT_DIR.glob("picks_exp_delay5_*.parquet"))
    if not files:
        return {"days": 0, "total": 0, "actionable": 0, "first": None, "last": None}
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["代码", "dec_dt"])
    dec = pd.to_datetime(df["dec_dt"])
    return {
        "days": len(files),
        "total": int(len(df)),
        "actionable": int(df["可操作"].sum()) if "可操作" in df.columns else 0,
        "first": dec.min().date() if len(df) else None,
        "last": dec.max().date() if len(df) else None,
    }


def _report_forward_progress() -> None:
    p = _forward_progress()
    print(f"\n{'=' * 150}")
    print("  前向转正进度（预声明标准：≥60 笔前向样本、超额 t≥2 且中位数>0；超额统计由研究脚本计算）")
    print(f"{'=' * 150}")
    if p["days"] == 0:
        print("  尚无前向日志（首次运行后开始积累）")
    else:
        print(
            f"  可操作样本 {p['actionable']}/{FORWARD_TARGET} 笔 | 结构候选累计 {p['total']} 笔 | "
            f"日志 {p['days']} 个交易日（{p['first']} → {p['last']}）"
        )
        if p["actionable"] >= FORWARD_TARGET:
            print("  ✅ 样本量已达标 → 触发预声明重检：跑超额 t 与中位数判定（研究脚本），通过才转正")
    print("  组合口径：广度优先 30 槽、单笔小、不补仓、FULL 退出（SL2 + 18%跟踪 + 背驰/破坏）；门关=仅记录属常态")
    print("  ⚠ 转正前本筛选仅作记录与观察，不构成买入建议（肥尾彩票型：均值靠尾部、典型交易中位数为负）")


def main() -> None:
    print("=" * 70)
    print("  delay5 存活确认买点（终态口径）— 每日筛选")
    print("=" * 70)
    metadata_available = False
    try:
        name_map, industry_map = legacy._load_stock_basic()
        print(f"[基础] 已加载 {len(name_map)} 只股票名称/行业")
        metadata_available = True
    except Exception as e:
        print(f"[基础] 名称/行业加载失败（仅离线扫描）：{e}")
        name_map, industry_map = {}, {}

    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] {len(files)} 只 | {n_workers} 进程 | 复用 surge-regime-stock-picker 扫描引擎\n")

    t0 = time.time()
    exp_raw = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(legacy._scan_one, files, chunksize=20), 1):
            if res and res.get("exp"):
                exp_raw.append(res["exp"])
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] delay5 候选 {len(exp_raw)} | {time.time() - t0:.0f}s")

    legacy._report_experimental(exp_raw, name_map, industry_map, metadata_available)
    _report_forward_progress()


if __name__ == "__main__":
    main()
