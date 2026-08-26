#!/usr/bin/env python3
"""bolt's parquet writer/reader head-to-head against Arrow (pyarrow).

    cmake --build <build> --target bench_parquet_vs_arrow
    python3 benchmarks/bench_parquet_vs_arrow.py <build>/benchmarks <tmpdir>

HOW THE DATA IS KEPT HONEST. The usual way a writer-vs-writer benchmark goes
wrong is the data: two generators that agree in description and differ in
bytes make every size number meaningless. So no generator is duplicated here.
The C++ half writes ONE uncompressed parquet file; this script READS THAT FILE
and re-writes its contents with pyarrow. Both sides then encode provably
identical logical content, and the script verifies it -- every bolt-written
file is read back with pyarrow and compared cell-for-cell against the same
table, so a size win that came from writing different data would fail before
it could be reported.

WHAT IS CONTROLLED
  * Serial on both sides. pyarrow's writer is single-threaded and its reader
    is pinned with use_threads=False; bolt runs with encode_pool = nullptr.
    This is a comparison of codecs and encodings, not of thread counts --
    bolt's parallel encode is measured separately in bench_parquet_write.
  * Dictionary encoding on for both (pyarrow's default).
  * 1 MiB data pages and one row group on both.

WHAT IS NOT EQUALISED, and belongs beside any number quoted from this:
  * bolt's GZIP is a FIXED-Huffman DEFLATE; Arrow's is zlib with dynamic
    Huffman. bolt's ZSTD emits RAW literals with predefined FSE tables;
    libzstd does Huffman literals and builds custom tables. Both are scope
    choices recorded in their headers, and both cost ratio, not correctness.
    The gap on those two rows IS the measure of what the omissions cost.
"""
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

REPEATS = 5
CODECS = ["none", "snappy", "gzip", "zstd", "lz4"]
PA_CODEC = {"none": "none", "snappy": "snappy", "gzip": "gzip",
            "zstd": "zstd", "lz4": "lz4"}


def best(fn):
    """min-of-REPEATS wall time in ms, plus the last return value."""
    lo = float("inf")
    out = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        lo = min(lo, ms)
    return lo, out


def main():
    if len(sys.argv) < 3:
        print("usage: bench_parquet_vs_arrow.py <bench-dir> <tmp-dir>",
              file=sys.stderr)
        return 2
    bench_dir, tmp = sys.argv[1], sys.argv[2]
    os.makedirs(tmp, exist_ok=True)

    exe = os.path.join(bench_dir, "bench_parquet_vs_arrow")

    def run_bolt():
        p = subprocess.run([exe, tmp], capture_output=True, text=True)
        if p.returncode != 0:
            print("bolt half failed:\n" + p.stdout + p.stderr, file=sys.stderr)
            sys.exit(1)
        own, cross = {}, {}
        for line in p.stdout.splitlines():
            if line.startswith("#") or "|" not in line:
                continue
            who, codec, enc, nbytes, dec = line.split("|")
            rec = (float(enc), int(nbytes), float(dec))
            (own if who == "bolt" else cross)[codec] = rec
        return own, cross

    # 1. bolt writes every codec and reports its own timings. The cross-decode
    #    numbers come from a SECOND invocation further down: on this first
    #    pass pyarrow has not written anything yet, and an earlier version of
    #    this script reported the resulting absent files as "bolt cannot read
    #    arrow" -- a false and damaging claim produced entirely by harness
    #    ordering.
    bolt, _ = run_bolt()

    # 2. The SAME data, taken from bolt's uncompressed file.
    src = os.path.join(tmp, "vsarrow_bolt_none.parquet")
    table = pq.read_table(src)
    nrows = table.num_rows

    # 3. pyarrow writes it at each codec, timed the same way.
    pa_res = {}
    for c in CODECS:
        path = os.path.join(tmp, "vsarrow_pa_%s.parquet" % c)

        def write():
            pq.write_table(table, path, compression=PA_CODEC[c],
                           use_dictionary=True, data_page_size=1 << 20,
                           row_group_size=nrows, write_statistics=True)
        enc_ms, _ = best(write)
        nbytes = os.path.getsize(path)
        dec_ms, _ = best(lambda: pq.read_table(path, use_threads=False))
        pa_res[c] = (enc_ms, nbytes, dec_ms)

    # 4. pyarrow reads what BOLT wrote -- both as a correctness gate and as
    #    the other half of the cross-decode matrix.
    pa_reads_bolt = {}
    ref = table.to_pydict()
    for c in CODECS:
        path = os.path.join(tmp, "vsarrow_bolt_%s.parquet" % c)
        if not os.path.exists(path):
            continue
        dec_ms, got = best(lambda: pq.read_table(path, use_threads=False))
        if got.to_pydict() != ref:
            print("FAIL: pyarrow read bolt's %s file as DIFFERENT data -- "
                  "every size comparison below would be meaningless" % c)
            return 1
        pa_reads_bolt[c] = dec_ms

    # 5. NOW re-run the bolt half so it can time reading pyarrow's files.
    _, bolt_reads_pa = run_bolt()

    # 6. Report.
    raw = bolt["none"][1]
    print()
    print("parquet write: bolt vs Arrow — %d rows x 16 cols, serial both "
          "sides, dictionary on, min-of-%d" % (nrows, REPEATS))
    print("(bolt's uncompressed output is %.1f MB; ratios are against it)"
          % (raw / 1e6))
    print()
    print("                  ENCODE (ms)        SIZE (MB)          "
          "DECODE own file (ms)")
    print("  codec        bolt   arrow  ratio   bolt  arrow  ratio   "
          "bolt   arrow  ratio")
    print("  " + "-" * 74)
    for c in CODECS:
        be, bb, bd = bolt.get(c, (-1, 0, -1))
        ae, ab, ad = pa_res[c]
        print("  %-9s %6.1f %7.1f  %5.2fx  %5.2f %6.2f  %5.2fx  %6.1f %7.1f  %5.2fx"
              % (c, be, ae, (ae / be) if be > 0 else 0.0,
                 bb / 1e6, ab / 1e6, (bb / ab) if ab else 0.0,
                 bd, ad, (ad / bd) if bd > 0 else 0.0))
    print()
    print("  ENCODE/DECODE ratio > 1.00x means bolt is FASTER.")
    print("  SIZE ratio > 1.00x means bolt's file is BIGGER (worse).")
    print()
    print("cross-decode — each reader on the other writer's file (ms):")
    print("  codec        bolt reads arrow   arrow reads bolt")
    print("  " + "-" * 46)
    for c in CODECS:
        br = bolt_reads_pa.get(c, (0, 0, -1))[2]
        ar = pa_reads_bolt.get(c, -1)
        print("  %-9s %14s   %16s"
              % (c,
                 ("%.1f" % br) if br >= 0 else "cannot read",
                 ("%.1f" % ar) if ar >= 0 else "cannot read"))
    print()
    print("  Every bolt-written file above was read back by pyarrow and "
          "compared")
    print("  cell-for-cell against the same table: all %d rows x 16 columns "
          "match." % nrows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
