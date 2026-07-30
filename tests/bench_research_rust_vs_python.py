"""Benchmark: Rust czsc._native.research vs pure Python implementations.

Usage:
    uv run --no-sync python tests/bench_research_rust_vs_python.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─── 1. Newey-West HAC: Rust vs Python ──────────────────────────

def bench_newey_west():
    _divider("Newey-West HAC (52 weekly diffs, lag=4)")

    rng = np.random.default_rng(42)
    weekly_diffs = (rng.normal(0.002, 0.03, 52)).tolist()
    n_iter = 10_000

    # --- Python (from xs_chan_statistics_v2_1.py) ---
    from statistics import NormalDist

    def _py_newey_west(diffs, lag=4, confidence=0.95):
        n = len(diffs)
        mean = sum(diffs) / n
        resid = [x - mean for x in diffs]
        gamma_0 = sum(r * r for r in resid) / n
        nw_var = gamma_0
        for j in range(1, min(lag, n - 1) + 1):
            gamma_j = sum(resid[j + k] * resid[k] for k in range(n - j)) / n
            weight = 1.0 - j / (lag + 1)
            nw_var += 2 * weight * gamma_j
        se = (nw_var / n) ** 0.5
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        return {"mean": mean, "se": se, "t_stat": mean / se, "lower_95": mean - z * se, "upper_95": mean + z * se}

    t0 = time.perf_counter()
    for _ in range(n_iter):
        py_result = _py_newey_west(weekly_diffs)
    py_elapsed = time.perf_counter() - t0

    # --- Rust ---
    from czsc._native.research import newey_west_hac

    t0 = time.perf_counter()
    for _ in range(n_iter):
        rs_result = newey_west_hac(weekly_diffs, lag=4, confidence=0.95)
    rs_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / rs_elapsed if rs_elapsed > 0 else float("inf")
    print(f"  Python: {py_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Rust:   {rs_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Python mean={py_result['mean']:.6f}  Rust mean={rs_result['mean']:.6f}")
    print(f"  Python SE={py_result['se']:.6f}  Rust SE={rs_result['se']:.6f}")


# ─── 2. Bootstrap: Rust vs Python ───────────────────────────────

def bench_bootstrap():
    _divider("Circular Block Bootstrap (52 weeks, 20k draws)")

    rng = np.random.default_rng(42)
    weekly_diffs = (rng.normal(0.002, 0.03, 52)).tolist()

    # --- Python ---
    def _py_bootstrap(diffs, block_size=4, n_draws=20_000, seed=42):
        n = len(diffs)
        blocks_needed = (n + block_size - 1) // block_size
        rs = np.random.RandomState(seed)
        means = []
        for _ in range(n_draws):
            sample = []
            for _ in range(blocks_needed):
                start = rs.randint(0, n)
                for j in range(block_size):
                    sample.append(diffs[(start + j) % n])
            means.append(sum(sample[:n]) / n)
        means.sort()
        return {
            "mean": sum(means) / n_draws,
            "q05": means[int(0.05 * (n_draws - 1))],
            "q95": means[int(0.95 * (n_draws - 1))],
        }

    t0 = time.perf_counter()
    py_result = _py_bootstrap(weekly_diffs)
    py_elapsed = time.perf_counter() - t0

    # --- Rust ---
    from czsc._native.research import circular_block_bootstrap

    t0 = time.perf_counter()
    rs_result = circular_block_bootstrap(weekly_diffs, block_size=4, n_draws=20_000, seed=42)
    rs_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / rs_elapsed if rs_elapsed > 0 else float("inf")
    print(f"  Python: {py_elapsed:.3f}s")
    print(f"  Rust:   {rs_elapsed:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Python q05={py_result['q05']:.6f}  Rust q05={rs_result['q05']:.6f}")


# ─── 3. Features (momentum/low-vol/rolling): Rust vs NumPy ─────

def bench_features():
    _divider("Feature Computation (5000 stocks x 300 sessions)")

    n_stocks = 5000
    n_sessions = 300
    n_iter = 3
    rng = np.random.default_rng(42)

    # --- Python (numpy) ---
    def _py_momentum(adj_close, short=20, long=120):
        result = np.full(len(adj_close), np.nan)
        for i in range(long, len(adj_close)):
            if adj_close[i] > 0 and adj_close[i - short] > 0 and adj_close[i - long] > 0:
                result[i] = np.log(adj_close[i] / adj_close[i - long]) - np.log(adj_close[i] / adj_close[i - short])
        return result

    from czsc._native.research import compute_momentum, compute_low_vol, rolling_mean

    # Generate test data
    test_data = []
    for _ in range(n_stocks):
        prices = np.cumprod(1 + rng.normal(0.0005, 0.02, n_sessions)) * 10
        test_data.append(prices.tolist())

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for prices in test_data:
            _py_momentum(np.array(prices))
    py_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for prices in test_data:
            compute_momentum(prices)
    rs_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / rs_elapsed if rs_elapsed > 0 else float("inf")
    print(f"  Momentum ({n_stocks} stocks x {n_iter} iters):")
    print(f"    Python/NumPy: {py_elapsed:.3f}s")
    print(f"    Rust:         {rs_elapsed:.3f}s")
    print(f"    Speedup: {speedup:.1f}x")


# ─── 4. OLS Neutralization: Rust vs NumPy ──────────────────────

def bench_ols():
    _divider("OLS Neutralization (3000 stocks, 30 industries)")

    n_stocks = 3000
    n_industries = 30
    n_iter = 100
    rng = np.random.default_rng(42)

    factor = rng.normal(0, 1, n_stocks).tolist()
    industry = rng.integers(0, n_industries, n_stocks).astype(np.uint32).tolist()
    ln_mcap = rng.normal(23, 2, n_stocks).tolist()

    # --- Python (numpy lstsq) ---
    def _py_ols_neutralize(factor_arr, ind_arr, mcap_arr):
        n = len(factor_arr)
        unique_ind = sorted(set(ind_arr))
        k = 2 + len(unique_ind) - 1

        X = np.zeros((n, k))
        X[:, 0] = 1.0
        X[:, 1] = mcap_arr
        for j, ind_val in enumerate(unique_ind[1:], start=2):
            X[:, j] = (np.array(ind_arr) == ind_val).astype(float)

        y = np.array(factor_arr)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        return residuals.tolist()

    from czsc._native.research import ols_neutralize

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _py_ols_neutralize(factor, industry, ln_mcap)
    py_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iter):
        ols_neutralize(factor, industry, ln_mcap)
    rs_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / rs_elapsed if rs_elapsed > 0 else float("inf")
    print(f"  Python/NumPy: {py_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Rust:         {rs_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Speedup: {speedup:.1f}x")


# ─── 5. Portfolio Simulation: Rust vs Python ────────────────────

def bench_portfolio():
    _divider("Portfolio Simulation (52 weeks, 5000 stocks)")

    n_weeks = 52
    n_stocks = 5000
    n_iter = 100
    rng = np.random.default_rng(42)

    symbols = [f"S{i:04d}" for i in range(n_stocks)]

    # Generate ranked symbols per week
    ranked_per_week = []
    for _ in range(n_weeks):
        shuffled = list(symbols)
        rng.shuffle(shuffled)
        ranked_per_week.append(shuffled)

    from czsc._native.research import simulate_fc_path

    # --- Python ---
    def _py_simulate(ranked_weeks, target=50, retention=75):
        current = set()
        results = []
        for ranked in ranked_weeks:
            retained = [s for s in current if s in ranked[:retention]]
            available = target - len(retained)
            new_entries = [s for s in ranked if s not in retained][:max(0, available)]
            current = set(retained) | set(new_entries)
            results.append({"holdings": list(current), "new": new_entries})
        return results

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _py_simulate(ranked_per_week)
    py_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iter):
        simulate_fc_path(ranked_per_week, target_slots=50, retention_max_rank=75)
    rs_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / rs_elapsed if rs_elapsed > 0 else float("inf")
    print(f"  Python: {py_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Rust:   {rs_elapsed:.3f}s  ({n_iter} iters)")
    print(f"  Speedup: {speedup:.1f}x")


def main():
    print("=" * 60)
    print("  czsc-research: Rust vs Python Benchmark")
    print("=" * 60)

    try:
        from czsc._native import research  # noqa: F401
    except ImportError:
        print("\n[ERROR] czsc._native.research not available.")
        print("  Run: uv run --no-sync maturin develop --release")
        return 1

    bench_newey_west()
    bench_bootstrap()
    bench_features()
    bench_ols()
    bench_portfolio()

    print(f"\n{'=' * 60}")
    print("  Benchmark complete.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
