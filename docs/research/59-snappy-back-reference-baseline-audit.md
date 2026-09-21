# G2PERF-33: Snappy back-reference baseline audit (2026-09-21)

## Decision

Do not change the codec from the ticket's old baseline. The ticket describes
the decoder before Bolt commit `6a8b539f841c923487c40ab0e887372bedeb0abc`,
which is an ancestor of the current Bolt HEAD,
`806bc12e6f58fb42c8e314b82dc5393fd6576882`. The targeted back-reference
optimization already landed, so a fresh current-vs-Arrow comparison is needed
before choosing another decoder change.

That commit removed the `len <= 16` cutoff which sent long matches to the
data-dependent copy loop. `snappy_copy_wide` now handles the Snappy format's
whole 1..64-byte match range in four ordered 16-byte units when `off >= 16`.
The exact-destination contract remains intact:

- `kSnappyFastSlop` is 192 bytes, derived from the largest deferred write.
- The wide lane stops before the exact end of the destination.
- The careful tail checks `op + len <= dst_len` and performs exact copies when
  the wide lane has insufficient headroom.
- Invalid zero or backward offsets still return `false`.

`git diff` reports no current working-tree change in
`include/bolt/ingest/bolt_snappy.h`, `tests/test_bolt_snappy.cpp`, or
`benchmarks/bench_parquet_decode.cpp`. This investigation therefore adds
evidence only; it does not layer another speculative decoder rewrite on top of
the landed fix.

## Landed performance evidence

These are the paired base/new measurements recorded by commit `6a8b539`. They
are historical landed evidence, not measurements rerun on 2026-09-21:

| Real decode workload | Before | After | Gain |
|---|---:|---:|---:|
| ClickBench `URL`, 8.517 GB | 4797 ms | 3261 ms | 1.47x |
| ClickBench `Title`, 8.153 GB | 4772 ms | 3834 ms | 1.24x |
| TPC-H SF10 `l_comment`, 1.830 GB | 1522 ms | 1404 ms | 1.08x |
| ClickBench `UserID`, 0.314 GB | 147 ms | 137 ms | 1.07x |
| TPC-H SF10 lineitem, all 16 columns | 3219 ms | 3102 ms | 1.04x |

The same landed evidence records identical old/new value checksums and an
independent byte comparison against Arrow/Google Snappy on 12 real ClickBench
`URL` pages (11.9 MB). Bolt and Arrow produced the same FNV-1a hash for every
page. The tag census which motivated the fix was also independent of the Bolt
decoder: matches longer than 16 bytes were 24.7% of `URL` back-references but
carried 66.4% of copied bytes.

## Fresh sanitizer and differential verification

The current HEAD was rebuilt in a dedicated ASan+UBSan tree and the complete
Snappy suite was run directly:

```sh
cmake -S extern/bolt -B build/sol-snappy/asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBOLT_BUILD_TESTS=ON \
  -DBOLT_BUILD_BENCHMARKS=OFF \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/sol-snappy/asan --target test_bolt_snappy --parallel 3
ASAN_OPTIONS=abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  build/sol-snappy/asan/tests/test_bolt_snappy --gtest_color=no
```

Result: 7/7 tests passed in 1.702 seconds with no sanitizer finding. The gate
includes two deterministic 2,000-stream differential campaigns against a
byte-at-a-time from-spec oracle, adversarial offsets and lengths, output guard
bytes, malformed input, and every truncated prefix of 24 generated streams.
This exercises back-references which Bolt's own literal-only compressor cannot
produce.

## Fresh real-column value checks

The release verifier was built in `build/sol-snappy/release` and run for three
complete passes over each real file. `bench_parquet_decode` hashes values rather
than buffer addresses: for Utf8/Binary it walks the bytes behind every
`StringView`, including the inline/spilled split.

```sh
build/sol-snappy/release/benchmarks/bench_parquet_decode \
  /Users/bengamble/benchdata/clickbench/hits.parquet 3 64 13,2

build/sol-snappy/release/benchmarks/bench_parquet_decode \
  /Users/bengamble/tpch/sf10/lineitem.parquet 3 16
```

| File and projection | Rows | Stable checksum | Pass times |
|---|---:|---:|---:|
| ClickBench `hits.parquet`, `URL` + `Title` | 99,997,497 | `aaa0b111e1d16f25` | 7607.9, 7431.1, 7408.6 ms |
| TPC-H SF10 `lineitem.parquet`, all 16 columns | 59,986,052 | `b7e5b4e772c460ae` | 3235.9, 3236.4, 3236.1 ms |

The checksums were identical on all three passes. These are fresh correctness
results for real Snappy-compressed pages and their materialized column values.
They are not a fresh old/new comparison.

The same binary's independent reference-compressor corpus mode verified exact-
sized, byte-exact output for all 256 chunks before timing. Seven decode passes
were 123.1..122.8 ms (2.19 GB/s at ratio 3.16). This is supplementary coverage;
the real-file and differential gates above carry the correctness conclusion.

## Timing qualification and remaining risk

The host was contended throughout the fresh run: load averages were
39.52/24.17/16.21 before ClickBench, 21.07/22.70/16.90 before TPC-H, and
19.58/22.17/16.97 before the codec corpus on an 18-thread machine. The fresh
times are diagnostic only and must not be used as a performance gate. There is
also no freshly built pre-`6a8b539` baseline binary, so claiming a new A/B from
these numbers would be invalid. The paired measurements in the landed commit
remain the benefit evidence.

The independent Arrow page-body check was not rerun because its scratch
extractor was not landed. If a new Snappy optimization is proposed, first land
that real-page manifest/checksum harness and establish a paired current-vs-new
baseline on a quiet host. A fresh MSVC/x86 sanitizer run also remains a normal
cross-platform integration gate.

## Resolution

G2PERF-33's specific `len <= 16` fast-lane cutoff has already been addressed,
its exact-buffer and malformed-input behavior survives current
sanitizer/differential testing, and real ClickBench/TPC-H values are stable
across full passes. The historical 1.47x `URL` gain does not prove Arrow parity,
and today's current-only run cannot measure the remaining gap. Record
`6a8b539` as the implemented increment and keep a fresh paired
current-vs-Arrow measurement as the next evidence step before closing or
retargeting the ticket.
