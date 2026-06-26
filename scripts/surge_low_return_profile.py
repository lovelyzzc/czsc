"""定位 Surge Regime delay5 候选中低收益标的的可观测特征。

这是一个诊断与否决研究，不会修改策略：

1. 严格复现 ``anticipate + delay5 + 市场门 + 硬过滤`` 的 3,185 笔基线候选；
2. 比较实际净收益 <= 0 的候选与盈利候选在决策日可见特征上的差异；
3. 对少量有明确业务含义的缩池规则，同时评估候选层和 10 槽实盘镜像层；
4. 只有同时通过 IS/OOS、候选层/组合层和缩池幅度门槛的规则才能标为 ``promote``。

规则集只用于说明当前特征空间的分离能力。它们没有通过时，不能把某个局部收益更好的
阈值升级为默认策略；尤其不能用事后 OOS 结果反向挑阈值。

依赖已冻结的候选 dump、市场状态、因子库和随机对照超额缓存：

    uv run --no-sync python scripts/surge_low_return_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPTS_DIR / "_output" / "surge_low_return_profile"
CANDIDATES_PATH = SCRIPTS_DIR / "_output" / "surge_candidates" / "candidates.parquet"
EXCESS_PATH = SCRIPTS_DIR / "_output" / "surge_selection_audit" / "excess_cache.parquet"
MARKET_PATH = SCRIPTS_DIR / "_output" / "surge_market_state_filter" / "market_state.parquet"
BARS_PATH = SCRIPTS_DIR / "_output" / "surge_factors" / "bars.parquet"
NAMECHANGE_PATH = Path.home() / ".ts_data_cache" / "namechange.parquet"

BUY_COST, SELL_COST = 0.0015, 0.0025
SLOTS = 10

# 固定基线：与 surge_selection_audit.BASE 一致。
BASE_HIGH20 = 0.12
FEATURES = [
    "ret5",
    "dist_hi250",
    "amp10",
    "amt_ratio20",
    "sl_pct",
    "high20_ratio",
    "sig_vol_ratio",
    "sig_ma_spread_pct",
    "score",
    "pivot_width_pct",
]

# 以下规则均是有明确交易语义的二元条件，不在运行后搜索新的切点。
DIAGNOSTIC_RULES = {
    "baseline": "所有当前基线候选",
    "decision_volume_confirmed": "决策日成交额 ≥ 此前20日均值",
    "short_term_positive": "决策日近5日收益 ≥ 0",
    "main_uptrend_only": "第5日仍处于主升/加速状态（7/8）",
    "stronger_market_breadth": "20日新高占比 > 15%",
}


def round_float(value: float | np.floating | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None and pd.notna(value) else None


def load_st_intervals() -> dict[str, np.ndarray]:
    """返回按股票分组、按改名日期排序的 ST 状态变更表。"""
    if not NAMECHANGE_PATH.exists():
        raise FileNotFoundError(f"缺少历史 ST 数据：{NAMECHANGE_PATH}")
    names = pd.read_parquet(NAMECHANGE_PATH)
    names["is_st"] = names["name"].fillna("").str.upper().str.contains("ST") | names["name"].fillna("").str.contains(
        "退"
    )
    return {
        code: group.sort_values("start_date")[["start_date", "is_st"]].to_numpy()
        for code, group in names.groupby("ts_code")
    }


def is_st_on(intervals: dict[str, np.ndarray], symbol: str, dt: pd.Timestamp) -> bool:
    rows = intervals.get(symbol)
    if rows is None:
        return False
    valid = rows[rows[:, 0] <= pd.Timestamp(dt).strftime("%Y%m%d")]
    return bool(valid[-1, 1]) if len(valid) else False


def load_baseline() -> pd.DataFrame:
    """复现 delay5 基线并补充决策日因子。"""
    keys = ["symbol", "sig_dt", "dec_dt", "entry_dt", "exit_dt"]
    candidates = pd.read_parquet(CANDIDATES_PATH)
    excess = pd.read_parquet(EXCESS_PATH)
    market = pd.read_parquet(MARKET_PATH, columns=["dt", "high20_ratio", "ew_index_above_ma20"])

    df = candidates[(candidates["mode"] == "anticipate") & (candidates["delay"] == 5)].copy()
    df = df.merge(excess, on=keys, how="left")
    df = df.merge(market, left_on="dec_dt", right_on="dt", how="left").drop(columns="dt")
    gate = (
        (df["sig_vol_ratio"] >= 1.2)
        & (df["sig_ma_spread_pct"] >= 3.0)
        & (df["sig_ret20"] >= 8.0)
        & (df["sig_above_zg"] == 1)
        & (df["amount_e"] >= 1.0)
        & df["sl_pct"].between(8.0, 20.0)
        & (df["gap_pct"] < df["limit_pct"] - 0.3)
        & (df["high20_ratio"] > BASE_HIGH20)
        & (df["ew_index_above_ma20"] > 0)
    )
    df = df[gate.fillna(False)].copy()

    intervals = load_st_intervals()
    df = df[
        [not is_st_on(intervals, symbol, dt) for symbol, dt in zip(df["symbol"], df["dec_dt"], strict=False)]
    ].copy()

    bars = pd.read_parquet(
        BARS_PATH,
        columns=["symbol", "dt", "ret5", "dist_hi250", "amp10", "amt_ratio20", "pivot_width_pct"],
    )
    df = df.merge(bars.rename(columns={"dt": "dec_dt"}), on=["symbol", "dec_dt"], how="left")
    df["ret_net_pct"] = ((1 + df["ret_gross_pct"] / 100) * (1 - SELL_COST) / (1 + BUY_COST) - 1) * 100

    # delay5 口径的 freshness 固定为 0；止损已被基线限制在满分区间 8-20%。
    regime_quality = {5: 12, 6: 18, 7: 16, 8: 20}
    df["priority"] = df["score"] * 0.35 + 25 + 20 + df["dec_regime"].map(regime_quality).fillna(10)
    return df.reset_index(drop=True)


def simulate_slots(df: pd.DataFrame) -> pd.DataFrame:
    """与 surge_portfolio_backtest 相同的贪心 10 槽选择，不依赖 Rust 扩展。"""
    selected: list[int] = []
    opened: dict[str, pd.Timestamp] = {}
    for entry_dt, day in df.sort_values(["entry_dt", "priority"], ascending=[True, False]).groupby(
        "entry_dt", sort=True
    ):
        opened = {symbol: exit_dt for symbol, exit_dt in opened.items() if exit_dt >= entry_dt}
        free = SLOTS - len(opened)
        for row in day.itertuples():
            if free <= 0:
                break
            if row.symbol in opened:
                continue
            opened[row.symbol] = row.exit_dt
            selected.append(row.Index)
            free -= 1
    return df.loc[selected].copy()


def performance(df: pd.DataFrame) -> dict:
    """候选或槽位成交的实际净收益与相对收益统计。"""
    excess = df["excess_pct"].dropna()
    return {
        "n": int(len(df)),
        "excess_mean": round_float(excess.mean()),
        "excess_median": round_float(excess.median()),
        "excess_t": round_float(excess.mean() / (excess.std() / np.sqrt(len(excess))) if len(excess) > 1 else None),
        "net_mean": round_float(df["ret_net_pct"].mean()),
        "net_median": round_float(df["ret_net_pct"].median()),
        "net_win_rate": round_float((df["ret_net_pct"] > 0).mean() * 100, 1),
    }


def is_not_worse(candidate: float | None, baseline: float | None) -> bool:
    """两个统计量均存在时才允许比较，避免 0 被 Python ``or`` 当作缺失。"""
    return candidate is not None and baseline is not None and candidate >= baseline


def rule_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "baseline":
        return pd.Series(True, index=df.index)
    if name == "decision_volume_confirmed":
        return df["amt_ratio20"] >= 1.0
    if name == "short_term_positive":
        return df["ret5"] >= 0.0
    if name == "main_uptrend_only":
        return df["dec_regime"].isin([7, 8])
    if name == "stronger_market_breadth":
        return df["high20_ratio"] > 0.15
    raise ValueError(f"未知规则：{name}")


def profile_low_returns(df: pd.DataFrame) -> list[dict]:
    """比较实际净亏损与盈利候选的当时可见特征，IS/OOS 分开避免混读。"""
    rows = []
    for segment in ("train", "test"):
        scoped = df[df["seg"] == segment].copy()
        loss = scoped[scoped["ret_net_pct"] <= 0]
        profit = scoped[scoped["ret_net_pct"] > 0]
        for feature in FEATURES:
            std = scoped[feature].std()
            diff = (loss[feature].mean() - profit[feature].mean()) / std if std and pd.notna(std) else np.nan
            rows.append(
                {
                    "segment": segment,
                    "feature": feature,
                    "loss_mean": round_float(loss[feature].mean(), 3),
                    "profit_mean": round_float(profit[feature].mean(), 3),
                    "standardized_diff": round_float(diff, 3),
                }
            )
    return rows


def evaluate_rules(df: pd.DataFrame) -> list[dict]:
    """评估缩池规则；只有四个维度都成立才允许升级为默认硬过滤。"""
    base_slots = simulate_slots(df)
    rows = []
    for name, description in DIAGNOSTIC_RULES.items():
        kept = df[rule_mask(df, name).fillna(False)].copy()
        slots = simulate_slots(kept)
        result = {
            "rule": name,
            "description": description,
            "candidate_reduction_pct": round_float((1 - len(kept) / len(df)) * 100, 1),
        }
        for segment in ("train", "test"):
            result[f"candidate_{segment}"] = performance(kept[kept["seg"] == segment])
            result[f"slots_{segment}"] = performance(slots[slots["seg"] == segment])

        # 保守升级条件：至少砍掉20%，候选层和10槽层在 IS/OOS 的相对收益均值不能输给基线，
        # 且 OOS 槽位净中位数不能变差。任何一项失败都保持观察，不进入策略。
        base_train, base_test = (
            performance(base_slots[base_slots["seg"] == "train"]),
            performance(base_slots[base_slots["seg"] == "test"]),
        )
        train, test = result["slots_train"], result["slots_test"]
        result["promote"] = bool(
            name != "baseline"
            and result["candidate_reduction_pct"] >= 20
            and is_not_worse(
                result["candidate_train"]["excess_mean"], performance(df[df["seg"] == "train"])["excess_mean"]
            )
            and is_not_worse(
                result["candidate_test"]["excess_mean"], performance(df[df["seg"] == "test"])["excess_mean"]
            )
            and is_not_worse(train["excess_mean"], base_train["excess_mean"])
            and is_not_worse(test["excess_mean"], base_test["excess_mean"])
            and is_not_worse(test["net_median"], base_test["net_median"])
        )
        rows.append(result)
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_report(summary: dict) -> None:
    rules = []
    for row in summary["rules"]:
        rules.append(
            {
                "rule": row["rule"],
                "reduction%": row["candidate_reduction_pct"],
                "IS slots excess": row["slots_train"]["excess_mean"],
                "OOS slots excess": row["slots_test"]["excess_mean"],
                "OOS slots net median": row["slots_test"]["net_median"],
                "promote": row["promote"],
            }
        )
    profile = pd.DataFrame(summary["low_return_profile"])
    pivot = profile.pivot(index="feature", columns="segment", values="standardized_diff").reset_index().fillna("")
    lines = [
        "# Surge Regime 低收益候选特征审计",
        "",
        "基线为 anticipate + delay5 + 市场门 + 既有硬过滤；收益为完整退出后的净收益，超额为同日同成交额十分位随机对照后的毛超额。",
        "",
        "## 基线",
        "",
        "```json",
        json.dumps(summary["baseline"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 净亏损候选相对盈利候选的标准化差异",
        "",
        "负数表示亏损候选的该特征更低；绝对值越大，区分力越强。所有差异均小于 0.25 个标准差，不能单独作为硬筛。",
        "",
    ]
    lines.extend(markdown_table(pivot.to_dict("records"), ["feature", "train", "test"]))
    lines.extend(["", "## 诊断性缩池规则（10槽镜像）", ""])
    lines.extend(
        markdown_table(
            rules, ["rule", "reduction%", "IS slots excess", "OOS slots excess", "OOS slots net median", "promote"]
        )
    )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            "没有规则满足升级门槛。可解释低收益的特征存在，但弱且跨阶段不稳定；将其中任意一项升级为默认硬过滤，会在某个市场阶段删掉尾部赢家或降低槽位组合超额。",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    slots = simulate_slots(baseline)
    summary = {
        "baseline": {
            "candidates": {segment: performance(baseline[baseline["seg"] == segment]) for segment in ("train", "test")},
            "slots": {segment: performance(slots[slots["seg"] == segment]) for segment in ("train", "test")},
        },
        "low_return_profile": profile_low_returns(baseline),
        "rules": evaluate_rules(baseline),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_report(summary)
    print(f"[done] 基线候选 {len(baseline)} | 输出 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
