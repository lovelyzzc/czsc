"""冻结 S2b 的同日同流动性匹配对照与执行缺口审计。

回答两个不调参的问题：

1. S2b 候选按 FULL 退出后的毛收益，是否超过同决策日、同成交额十分位、同持有期的随机对照；
2. 10 槽组合的容量与 priority 选择，是否把候选层的相对优势破坏掉。

最新尾部中 ``max_hold`` 且持有不足 59 个交易日的交易视为右删失，不进入确认性统计。
输出写入 ``scripts/_output/s2b_matched_control_audit.json``。

    uv run --no-sync python scripts/s2b_matched_control_audit.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import _s2c_core as core
import pandas as pd
from surge_market_state_filter import StableControlSampler

OUTPUT_PATH = Path(__file__).resolve().parent / "_output" / "s2b_matched_control_audit.json"
ARCH_FRESH = (pd.Timestamp("2021-07-01"), pd.Timestamp("2023-12-31"))
MAIN_START = pd.Timestamp("2024-01-01")
HOLDOUT_2026 = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-27"))
FULL_MAX_HOLD_DAYS = 59
N_BOOT = 10_000
BLOCK_LEN = 10


def censored_tail_mask(frame: pd.DataFrame) -> pd.Series:
    """标记因数据尾部不足而未完成 60 根模拟窗口的 max-hold 交易。"""

    return frame["exit_reason"].eq("max_hold") & frame["hold_days"].lt(FULL_MAX_HOLD_DAYS)


def summarize_matched_excess(frame: pd.DataFrame, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """按决策日聚类后报告 HAC 与平稳分块 bootstrap。"""

    valid = frame.dropna(subset=["matched_excess_pct"]).copy()
    if valid.empty:
        return {"n_trades": 0, "n_decision_dates": 0}
    daily = valid.groupby("dec_dt", sort=True)["matched_excess_pct"].mean()
    values = valid["matched_excess_pct"]
    return {
        "n_trades": int(len(valid)),
        "n_decision_dates": int(len(daily)),
        "mean_excess_pct": round(float(values.mean()), 3),
        "median_excess_pct": round(float(values.median()), 3),
        "positive_excess_pct": round(float(values.gt(0).mean() * 100), 1),
        "gross_mean_pct": round(float(valid["ret_gross_pct"].mean()), 3),
        "daily_hac": core.newey_west_tstat(daily.to_numpy()),
        "daily_block_bootstrap": core.block_bootstrap_mean(
            daily.to_numpy(),
            n_boot=n_boot,
            block_len=BLOCK_LEN,
        ),
    }


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[frame["dec_dt"].between(start, end)].copy()


def _main_execution_keys(context: dict[str, Any]) -> set[tuple[str, pd.Timestamp]]:
    d0 = context["d0"]
    mask = core.gate_mask(d0, core.S2B_VR, core.S2B_SP)
    result = core.simulate(
        core.build_core(d0, mask),
        context,
        policy="passive",
        n_slots=core.N_SLOTS,
        cost_pct=core.TOTAL_COST_PCT,
    )
    trades = pd.DataFrame(result["trades"])
    return set(zip(trades["symbol"], pd.to_datetime(trades["entry_dt"]), strict=False))


def run_audit() -> dict[str, Any]:
    """执行冻结审计并返回可序列化结果。"""

    context = core.load_context(verbose=True)
    d0 = context["d0"]
    s2b = d0.loc[core.gate_mask(d0, core.S2B_VR, core.S2B_SP)].copy()
    s2b["dec_dt"] = pd.to_datetime(s2b["dec_dt"])
    s2b["entry_dt"] = pd.to_datetime(s2b["entry_dt"])
    if s2b.duplicated(["symbol", "entry_dt"]).any():
        raise ValueError("S2b candidate identity is not unique by symbol and entry_dt")

    s2b["censored_tail"] = censored_tail_mask(s2b)
    execution_keys = _main_execution_keys(context)
    s2b["executed_main"] = [
        (symbol, entry_dt) in execution_keys for symbol, entry_dt in zip(s2b["symbol"], s2b["entry_dt"], strict=False)
    ]

    print(f"[匹配对照] S2b={len(s2b)}，右删失={int(s2b['censored_tail'].sum())}，构建 K=50 对照 ...")
    sampler = StableControlSampler()
    s2b["matched_excess_pct"] = [sampler.excess_for(row) for row in s2b.itertuples()]
    mature = s2b[~s2b["censored_tail"]].copy()

    arch = _window(mature, *ARCH_FRESH)
    main = mature[mature["dec_dt"] >= MAIN_START].copy()
    holdout = _window(mature, *HOLDOUT_2026)
    main_executed = main[main["executed_main"]].copy()
    main_not_executed = main[~main["executed_main"]].copy()
    holdout_executed = holdout[holdout["executed_main"]].copy()
    holdout_not_executed = holdout[~holdout["executed_main"]].copy()

    yearly = {
        str(int(year)): summarize_matched_excess(group)
        for year, group in mature.groupby(mature["dec_dt"].dt.year, sort=True)
    }
    main_priority = main.copy()
    main_priority["priority_quintile"] = pd.qcut(main_priority["score"], 5, labels=False, duplicates="drop")
    priority_quintiles = {
        str(int(quintile)): {
            **summarize_matched_excess(group),
            "score_mean": round(float(group["score"].mean()), 2),
            "executed_pct": round(float(group["executed_main"].mean() * 100), 1),
        }
        for quintile, group in main_priority.groupby("priority_quintile", sort=True)
    }

    summaries = {
        "all_mature": summarize_matched_excess(mature),
        "architecture_fresh_2021_07_to_2023_12": summarize_matched_excess(arch),
        "main_2024plus": summarize_matched_excess(main),
        "holdout_2026_to_2026_07_27": summarize_matched_excess(holdout),
        "main_executed_10_slots": summarize_matched_excess(main_executed),
        "main_not_executed": summarize_matched_excess(main_not_executed),
        "holdout_2026_executed_10_slots": summarize_matched_excess(holdout_executed),
        "holdout_2026_not_executed": summarize_matched_excess(holdout_not_executed),
    }
    historical = summaries["main_executed_10_slots"]
    current = summaries["holdout_2026_executed_10_slots"]
    historical_confirmed = historical["daily_hac"]["t_stat"] >= 2 and historical["daily_block_bootstrap"]["ci_lo"] > 0
    current_confirmed = current["daily_hac"]["t_stat"] >= 2 and current["daily_block_bootstrap"]["ci_lo"] > 0

    return {
        "schema": "s2b_matched_control_audit_v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "design": {
            "gate": {"sig_vol_ratio_lte": core.S2B_VR, "sig_ma_spread_pct_gte": core.S2B_SP},
            "control": "same decision date and amount decile; deterministic K=50; same entry/exit dates",
            "comparison": "gross strategy return minus gross control median",
            "daily_inference": f"decision-date cluster mean; HAC; stationary bootstrap block={BLOCK_LEN}",
            "right_censoring": f"exit_reason=max_hold and hold_days<{FULL_MAX_HOLD_DAYS}",
            "parameter_search": "none; frozen S2b only",
        },
        "counts": {
            "s2b_total": int(len(s2b)),
            "censored_tail": int(s2b["censored_tail"].sum()),
            "mature": int(len(mature)),
            "main_executed": int(len(main_executed)),
            "main_not_executed": int(len(main_not_executed)),
            "holdout_2026_executed": int(len(holdout_executed)),
            "holdout_2026_not_executed": int(len(holdout_not_executed)),
        },
        "summaries": summaries,
        "yearly": yearly,
        "main_priority_quintiles": priority_quintiles,
        "verdict": {
            "historical_main_executed_matched_edge": historical_confirmed,
            "holdout_2026_matched_edge": current_confirmed,
            "status": (
                "HISTORICAL_EDGE_BUT_FORWARD_CONFIRMATION_PENDING"
                if historical_confirmed and not current_confirmed
                else "REVIEW_REQUIRED"
            ),
            "live_authorized": False,
            "reason": (
                "2024+ executed S2b retains matched excess, but the 2026 frozen window does not pass "
                "HAC t>=2 and positive bootstrap lower bound; 2024+ was seen during selection."
            ),
        },
    }


def main() -> None:
    started = time.time()
    result = run_audit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    print(f"[输出] {OUTPUT_PATH} | {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
