"""TAIL 快路径一致性检查：daily_scan 用 iter_states(tail=N) 的信号是否等于全量重放

FSM 路径依赖：tail 模式在 n-tail 处用 `_seed_regime` 启发式播种，状态序列可能与
全量重放（WARMUP 起步）不同。本脚本抽样对比两者在「最后一根的 regime」与
「最近 10 根的 surge_onset 判定」上的一致率，决定 TAIL 取值是否安全。

    uv run --no-sync python scripts/check_tail_consistency.py [TAIL=160] [N_SAMPLE=300]
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from functools import partial

import trend_regime as tr

SCAN_WINDOW = 10  # 与 daily_scan 一致


def _onsets(states) -> set[tuple[str, str]]:
    """最近 SCAN_WINDOW 根内的 (启动日, mode) 集合。"""
    regimes = [s.regime for s in states]
    out = set()
    for p in range(max(1, len(states) - SCAN_WINDOW), len(states)):
        prior = regimes[max(0, p - tr.SURGE_PRIOR_WINDOW) : p]
        for mode in ("confirm", "anticipate"):
            if tr.surge_onset(states[p - 1].regime, states[p].regime, states[p].feats, prior, mode):
                out.add((states[p].dt.strftime("%Y-%m-%d"), mode))
    return out


def _check_one(parquet_path: str, tail: int) -> tuple[bool, bool] | None:
    """返回 (末根 regime 一致, 最近10根 onset 集合一致)。"""
    df = tr.load_stock(parquet_path)
    if df is None:
        return None
    full = tr.iter_states(df, with_features=True)
    fast = tr.iter_states(df, with_features=True, tail=tail)
    if len(full) < SCAN_WINDOW + 2 or len(fast) < SCAN_WINDOW + 2:
        return None
    return (full[-1].regime == fast[-1].regime, _onsets(full) == _onsets(fast))


def main(argv: list[str]) -> int:
    tail = int(argv[0]) if argv else 160
    n_sample = int(argv[1]) if len(argv) > 1 else 300

    files = [str(p) for p in sorted(tr.DATA_DIR.glob("*.parquet"))]
    step = max(1, len(files) // n_sample)
    files = files[::step][:n_sample]
    print(f"[TAIL={tail}] 抽样 {len(files)} 只对比 fast vs full")

    t0 = time.time()
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(mp.cpu_count(), 8)) as pool:
        for r in pool.imap_unordered(partial(_check_one, tail=tail), files, chunksize=5):
            if r is not None:
                res.append(r)
    n = len(res)
    reg_ok = sum(1 for a, _ in res if a)
    onset_ok = sum(1 for _, b in res if b)
    print(
        f"[结果] 有效 {n} | 末根 regime 一致 {reg_ok}/{n} ({reg_ok / n * 100:.1f}%) | "
        f"onset 集合一致 {onset_ok}/{n} ({onset_ok / n * 100:.1f}%) | {time.time() - t0:.0f}s"
    )
    mismatch = 1 - min(reg_ok, onset_ok) / n
    if mismatch > 0.01:
        print(f"[FAIL] mismatch {mismatch * 100:.1f}% > 1%，建议提高 TAIL 后复测")
        return 1
    print("[OK] mismatch ≤ 1%，TAIL 取值安全")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
