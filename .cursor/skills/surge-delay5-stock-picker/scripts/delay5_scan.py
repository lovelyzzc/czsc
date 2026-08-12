"""研究态每日筛选：anticipate + delay5 存活确认 + 市场状态门。

候选检测只调用共享的 ``surge_live.detect_delay5``，不经过旧十日 surge/reversion
扫描路径；报告与前向日志复用 ``daily_scan._report_experimental``。所有股票统一要求
最后一根日线等于 active manifest 的 ``safe_end_date``，避免把停牌票旧信号混入今日扫描。

阈值全部固定（不吃 SURGE_PICKER_* 环境变量）：市场门 high20_ratio>0.12 &
等权指数>MA20；硬过滤 成交额≥1亿 + 止损带 8-20% + 剔除 ST。

    PYTHONUNBUFFERED=1 .venv/bin/python .cursor/skills/surge-delay5-stock-picker/scripts/delay5_scan.py
"""

from __future__ import annotations

import json
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
import surge_live as sl  # noqa: E402
import surge_portfolio_backtest as spb  # noqa: E402  # GAP_LIMIT_MARGIN（开盘逼近涨停放弃规则）
import trend_regime as tr  # noqa: E402

FORWARD_TARGET = 60  # 预声明转正标准：≥60 笔前向样本、超额 t≥2 且中位数>0


def _target_date_from_manifest() -> pd.Timestamp:
    """读取正式 active snapshot 的安全截止日；身份不完整时 fail closed。"""
    path = tr.DATA_DIR / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "a_stock_daily_qfq_active_snapshot_v1":
        raise ValueError(f"unexpected active manifest schema: {payload.get('schema')!r}")
    value = str(payload.get("safe_end_date", ""))
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"invalid safe_end_date: {value!r}")
    return pd.to_datetime(value, format="%Y%m%d").normalize()


def _scan_delay5_one(task: tuple[str, str]) -> dict | None:
    """在统一目标日上扫描单票 delay5；不执行旧 surge/reversion 分支。"""
    parquet_path, target_text = task
    frame = tr.load_stock(parquet_path)
    if frame is None:
        return None
    target = pd.Timestamp(target_text).normalize()
    if pd.Timestamp(frame["dt"].iloc[-1]).normalize() != target:
        return None
    states = tr.iter_states(frame, with_features=True, tail=legacy.TAIL)
    hit = sl.detect_delay5(states)
    if hit is None or pd.Timestamp(hit["dec_dt"]).normalize() != target:
        return None
    hit["代码"] = str(frame["symbol"].iloc[0])
    amount = float(frame["amount"].iloc[-1]) if "amount" in frame.columns else 0.0
    hit["成交额亿"] = sl.amount_to_e(amount)
    return hit


def _write_empty_scan_log(target_dt: pd.Timestamp) -> Path:
    """真实零候选也写入带 schema 的空日志，使“已扫描”与“未运行”可区分。"""
    path = legacy.OUTPUT_DIR / f"picks_exp_delay5_{target_dt:%Y-%m-%d}.parquet"
    empty = pd.DataFrame(
        {
            "代码": pd.Series(dtype="string"),
            "dec_dt": pd.Series(dtype="datetime64[ns]"),
            "可操作": pd.Series(dtype="bool"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(path, index=False)
    return path


def _entry_status(sdf: pd.DataFrame | None, dec_dt: pd.Timestamp, code: str) -> str:
    """次日开盘成交模拟：filled=可成交 / pending=次日 bar 未落地 / abandoned=开盘逼近涨停放弃。

    gap 与 dec 收盘价都取自当前本地日线（同一复权序列内比值，免疫 qfq 重刷漂移），
    规则与 surge_candidates_dump / surge_portfolio_backtest 一致：gap ≥ 板限-0.3% 视为买不进。
    """
    if sdf is None:
        return "pending"
    idx = sdf.index[sdf["dt"] == dec_dt]
    if len(idx) == 0 or int(idx[0]) + 1 >= len(sdf):
        return "pending"
    i = int(idx[0])
    gap_pct = (float(sdf["open"].iloc[i + 1]) / float(sdf["close"].iloc[i]) - 1) * 100
    return "abandoned" if gap_pct >= tr.limit_pct_for(code) - spb.GAP_LIMIT_MARGIN else "filled"


def _forward_progress() -> dict:
    """累计前向样本：扫描全部 picks_exp_delay5_*.parquet，按（代码, dec_dt）去重。

    预声明标准要求按「次日开盘成交模拟」累计 ≥60 笔（见 S8_INCREMENTAL_VALIDATION_2026-08-10.md）。
    `可操作` 只代表决策日收盘已过市场门+硬过滤，是样本上限；这里对每笔可操作样本再做
    post-entry 确认，只有次日实际可成交的 filled 才计入转正进度。
    """
    files = sorted(legacy.OUTPUT_DIR.glob("picks_exp_delay5_*.parquet"))
    empty = {
        "days": 0,
        "total": 0,
        "actionable": 0,
        "filled": 0,
        "pending": 0,
        "abandoned": 0,
        "first": None,
        "last": None,
    }
    if not files:
        return empty
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["代码", "dec_dt"])
    act = df[df["可操作"]] if "可操作" in df.columns else df.iloc[0:0]
    filled = pending = abandoned = 0
    for code, g in act.groupby("代码"):
        sdf = tr.load_stock(tr.DATA_DIR / f"{code}.parquet")
        for _, r in g.iterrows():
            status = _entry_status(sdf, pd.Timestamp(r["dec_dt"]), str(code))
            filled += status == "filled"
            pending += status == "pending"
            abandoned += status == "abandoned"
    return {
        **empty,
        "days": len(files),
        "total": int(len(df)),
        "actionable": int(len(act)),
        "filled": filled,
        "pending": pending,
        "abandoned": abandoned,
        "first": pd.to_datetime(files[0].stem.removeprefix("picks_exp_delay5_")).date(),
        "last": pd.to_datetime(files[-1].stem.removeprefix("picks_exp_delay5_")).date(),
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
            f"  已确认可成交 {p['filled']}/{FORWARD_TARGET} 笔（次日开盘成交模拟口径）| "
            f"可操作 {p['actionable']} 笔（决策日收盘口径：待次日确认 {p['pending']}、开盘涨停放弃 {p['abandoned']}）"
        )
        print(f"  结构候选累计 {p['total']} 笔 | 日志 {p['days']} 个交易日（{p['first']} → {p['last']}）")
        if p["filled"] >= FORWARD_TARGET:
            print("  ✅ 样本量已达标 → 触发预声明重检：跑超额 t 与中位数判定（研究脚本），通过才转正")
    print("  研究口径：候选完整留档；市场门关时仅记录。生产共同期限行业/规模控制尚未确认选股超额。")
    print("  ⚠ live_authorized=false；本筛选仅用于前向研究，不构成买入建议。")


def main() -> None:
    print("=" * 70)
    print("  delay5 存活确认（研究口径）— 每日筛选")
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
    target_dt = _target_date_from_manifest()
    n_workers = min(mp.cpu_count(), 8)
    print(f"[数据] 目标日 {target_dt.date()} | {len(files)} 只 | {n_workers} 进程 | delay5 专用扫描\n")

    t0 = time.time()
    exp_raw = []
    tasks = [(path, target_dt.isoformat()) for path in files]
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_delay5_one, tasks, chunksize=20), 1):
            if res:
                exp_raw.append(res)
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] delay5 候选 {len(exp_raw)} | {time.time() - t0:.0f}s")

    legacy._report_experimental(
        exp_raw,
        name_map,
        industry_map,
        metadata_available,
        expected_dec_dt=target_dt,
    )
    if not exp_raw:
        print(f"  [空日志] {_write_empty_scan_log(target_dt)}")
    _report_forward_progress()


if __name__ == "__main__":
    main()
