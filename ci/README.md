# ci/

CI tooling that isn't part of the main build. See
`.github/workflows/ci.yml` for how these are wired.

## `perf_check.py`

Bolt perf regression gate. Runs `bench_bolt`, `bench_kernels`,
`bench_tpch_lite`, `bench_1brc` and compares their `ns/op` and `ns/row`
metrics against `perf_baselines.json`. Fails if any metric regresses by
more than 5%.

```bash
# After a release build, check against committed baselines:
python3 ci/perf_check.py --build-dir build/release

# Record new baselines on a trusted machine (commits into baselines file):
python3 ci/perf_check.py --build-dir build/release --mode record
git add ci/perf_baselines.json && git commit -m "perf: rebaseline"
```

Baselines are platform-specific in spirit — today we baseline on the
CI's Linux runner only. `perf_baselines.json` ships empty until a human
records the first set; while empty, the gate passes trivially (first-run
behaviour).

## `perf_baselines.json`

Committed baselines for the perf gate. Shape:

```json
{
  "bench_kernels": {
    "op:filter_gt<int32_t>  branchless scalar kernel": 2.31,
    "op:filter_gt<int32_t>  bmm_* SIMD kernel (compressstore)": 0.17
  }
}
```

Keys are `"<unit>:<bench-line-name>"`; values are ns per unit.
