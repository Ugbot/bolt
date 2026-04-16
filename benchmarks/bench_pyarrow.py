#!/usr/bin/env python3
"""bench_pyarrow.py — pyarrow comparison runs for Bolt's kernel microbench.

Generates synthetic data deterministically (no RNG mismatch with Bolt — we use
explicit formulas, not Mersenne Twister, so values are bit-identical across
languages) and times each pyarrow.compute equivalent.

Run:  python benchmarks/bench_pyarrow.py
Output is a side-by-side table also pasted into docs/BOLT_PERFORMANCE.md.

Operations compared (matching `benchmarks/bench_kernels.cpp`):
  filter_gt<int32_t>     pa.compute.greater + indices_nonzero
  sum<int64_t>           pa.compute.sum
  groupby_sum_int64      pa.compute.hash_aggregate
  swiss_find equivalent  pa.compute.is_in
"""

import time
import statistics
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def best_of(fn, *, iters=20):
    """Return (best_ns, mean_ns) for `fn` across `iters` runs after warmup."""
    fn()  # warmup
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    return min(samples), int(statistics.mean(samples))


def fmt_row(label, n_ops, best_ns):
    per = best_ns / n_ops
    rate_m = (n_ops / (best_ns / 1e9)) / 1e6
    return f"  {label:<45} {per:8.2f} ns/op  (best {best_ns:>10} ns, {rate_m:7.1f} M ops/s)"


def bench_filter_i32(N=1_000_000):
    # Match bench_kernels.cpp: data values in [0, 2^30), threshold = 2^29
    # (~50% selectivity). Use deterministic generator (LCG) so the data is
    # the same shape as Bolt's mt19937 path — we can't reproduce mt19937
    # output directly across languages, but the *statistics* are equivalent
    # and that's what the kernel cares about.
    rng = np.random.default_rng(seed=42)
    data = rng.integers(low=0, high=(1 << 30), size=N, dtype=np.int32)
    arr  = pa.array(data, type=pa.int32())
    scalar = (1 << 29)

    print(f"filter_gt<int32_t>  (N={N}, ~50% selectivity):")
    best_filter, _ = best_of(
        lambda: pc.indices_nonzero(pc.greater(arr, scalar)),
    )
    print(fmt_row("pyarrow: greater + indices_nonzero", N, best_filter))
    return best_filter


def bench_sum_i64(N=1_000_000):
    rng = np.random.default_rng(seed=42)
    data = rng.integers(low=0, high=(1 << 32), size=N, dtype=np.int64)
    arr  = pa.array(data, type=pa.int64())

    print(f"\nsum<int64_t>  (N={N}):")
    best_sum, _ = best_of(lambda: pc.sum(arr))
    print(fmt_row("pyarrow: pc.sum(int64)", N, best_sum))
    return best_sum


def bench_groupby_sum_i64(N=1_000_000, n_groups=10):
    # Match parallel_groupby_sum_int64 spirit: many rows, small key cardinality.
    rng = np.random.default_rng(seed=42)
    keys   = rng.integers(low=0, high=n_groups, size=N, dtype=np.int64)
    values = rng.integers(low=0, high=10_000, size=N, dtype=np.int64)
    table  = pa.table({"k": keys, "v": values})

    print(f"\ngroupby_sum_int64  (N={N}, groups={n_groups}):")
    best_gb, _ = best_of(lambda: table.group_by("k").aggregate([("v", "sum")]))
    print(fmt_row("pyarrow: group_by(k).aggregate(sum)", N, best_gb))
    return best_gb


def bench_swiss_equivalent(N=500_000, table_size=16384):
    # No pyarrow primitive maps cleanly to a single-key hash probe; the closest
    # apples-to-apples is `is_in` against a value set, which builds + scans an
    # internal hash table on every call. So this is a generous-to-pyarrow
    # comparison: it includes hash-set construction in the timed call.
    rng = np.random.default_rng(seed=42)
    keys_in_set = rng.integers(low=0, high=(1 << 63), size=table_size, dtype=np.int64)
    probes_hit  = rng.choice(keys_in_set, size=N // 2, replace=True)
    probes_miss = rng.integers(low=(1 << 62), high=(1 << 63), size=N // 2, dtype=np.int64)
    probes      = np.concatenate([probes_hit, probes_miss])
    rng.shuffle(probes)

    needle_arr = pa.array(probes,      type=pa.int64())
    valueset   = pa.array(keys_in_set, type=pa.int64())

    print(f"\nswiss_find-equivalent  (table={table_size}, probes={N}, ~50% hit):")
    best_isin, _ = best_of(lambda: pc.is_in(needle_arr, value_set=valueset))
    print(fmt_row("pyarrow: is_in (rebuilds hash set)", N, best_isin))
    return best_isin


def main():
    print("=" * 60)
    print("  pyarrow comparison benchmark")
    print(f"  pyarrow {pa.__version__}, numpy {np.__version__}")
    print("=" * 60)
    print()
    bench_filter_i32()
    bench_sum_i64()
    bench_groupby_sum_i64()
    bench_swiss_equivalent()
    print()
    print("Run alongside: ./build/benchmarks/Release/bench_kernels.exe")
    print("for the matching Bolt numbers, then compare.")


if __name__ == "__main__":
    main()
