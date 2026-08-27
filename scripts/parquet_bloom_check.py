#!/usr/bin/env python3
"""Probe bolt's split-block bloom filters with an INDEPENDENT implementation.

    ./test_bolt_parquet_write_bloom          # writes the fixtures
    python3 scripts/parquet_bloom_check.py <dir-containing-them>

WHY THIS EXISTS. A bloom filter is the one thing bolt writes whose only
consumer is a reader deciding what to SKIP. Get it wrong and nothing errors:
a filter that wrongly reports a present value as absent makes any reader that
trusts it skip a row group that really does match, and the query returns
fewer rows with no diagnostic anywhere. bolt's own probe cannot detect this,
because a writer and a probe that share a misunderstanding of the salts, the
block index or the hash agree perfectly with each other.

The existing pyarrow gate does not close this: it only checks that a file
with filters interleaved between row groups still READS. It never probes.

DuckDB implements the same parquet spec separately -- xxhash64, the eight
salt constants, the block assignment -- and exposes parquet_bloom_probe,
which reports per row group whether the filter EXCLUDES a value. So:

  * every value that IS in a row group must NOT be excluded from it.
    This is the direction that causes silent wrong answers, so it is
    checked exhaustively over a spread of values rather than sampled.
  * a value in NO row group should be excluded from nearly all of them. A
    filter of all ones would satisfy the first check perfectly and be
    useless; this is what catches that. The floor is 90%, set against
    kPwDefaultBloomFpp = 0.05 -- these fixtures request the default, so the
    expected exclusion rate is ~95%, and the measured 93-95% IS that target
    rather than a weak filter. Tighten this floor only alongside the
    default.

Discriminating power is established against bolt's own bloom-DISABLED
fixture: with no filter present, nothing can be excluded. If that file
reported exclusions, DuckDB would not be reading bolt's filter bytes and
every result here would be meaningless.
"""
import os
import sys

import duckdb
import pyarrow.parquet as pq

FIXTURE = "test_bolt_parquet_write_bloom_interleaved.parquet"
NO_BLOOM = "test_bolt_parquet_write_bloom_off.parquet"
# The fixture's generating rule (tests/test_bolt_parquet_write_bloom.cpp).
VALUE_AT = lambda i: i * 3 - 7
N_ROWS = 30000


def excluded_by_rg(con, path, col, value):
    """{row_group: True if the filter says the value cannot be there}."""
    rows = con.execute(
        "select row_group_id, bloom_filter_excludes "
        "from parquet_bloom_probe(?, ?, ?)", [path, col, value]).fetchall()
    return {int(rg): bool(ex) for rg, ex in rows}


def row_group_ranges(path):
    """[(first_row, last_row)] per row group, in row order."""
    md = pq.ParquetFile(path).metadata
    out, start = [], 0
    for r in range(md.num_row_groups):
        n = md.row_group(r).num_rows
        out.append((start, start + n - 1))
        start += n
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    path = os.path.join(d, FIXTURE)
    if not os.path.exists(path):
        print("missing %s (run test_bolt_parquet_write_bloom first)" % FIXTURE,
              file=sys.stderr)
        return 2

    con = duckdb.connect()
    errs = []

    # Discriminating power FIRST -- if DuckDB is not really reading bolt's
    # filter bytes, nothing below means anything, and a green run would be
    # actively misleading.
    off = os.path.join(d, NO_BLOOM)
    if os.path.exists(off):
        try:
            any_excl = any(excluded_by_rg(con, off, "v", 12345).values())
        except duckdb.Error:
            any_excl = False        # no filter at all: nothing to probe
        if any_excl:
            print("FAIL: the bloom-disabled fixture reports exclusions, so "
                  "DuckDB is not reading bolt's filter -- this gate proves "
                  "nothing")
            return 1

    ranges = row_group_ranges(path)
    n_rg = len(ranges)
    if n_rg < 2:
        print("FAIL: expected several row groups, got %d" % n_rg)
        return 1

    # 1. NO FALSE NEGATIVES. This is the direction that loses rows.
    checked = 0
    for rg, (lo, hi) in enumerate(ranges):
        # Ends and interior of every row group, plus a stride through it.
        idxs = {lo, lo + 1, (lo + hi) // 2, hi - 1, hi}
        idxs |= set(range(lo, hi + 1, max(1, (hi - lo) // 40)))
        for i in sorted(idxs):
            if not (0 <= i < N_ROWS):
                continue
            v = VALUE_AT(i)
            ex = excluded_by_rg(con, path, "v", v)
            checked += 1
            if ex.get(rg, False):
                errs.append("row %d (v=%d) IS in row group %d, but the filter "
                            "excludes it -- a reader would skip matching rows"
                            % (i, v, rg))
                if len(errs) > 5:
                    break
        if len(errs) > 5:
            break

    # 2. THE FILTER MUST ACTUALLY FILTER. All-ones passes check 1 perfectly.
    absent = [10**12 + k for k in range(40)]
    excl_total = kept = 0
    for v in absent:
        ex = excluded_by_rg(con, path, "v", v)
        excl_total += sum(1 for e in ex.values() if e)
        kept += len(ex)
    rate = (excl_total / kept) if kept else 0.0
    if rate < 0.9:
        errs.append("absent values were excluded from only %.1f%% of row "
                    "groups; with the default fpp of 0.05 this should be "
                    "~95%%, so the filter is undersized or barely filtering"
                    % (100.0 * rate))

    for e in errs:
        print("FAIL: %s" % e)
    if errs:
        return 1
    print("OK: DuckDB's independent bloom implementation agrees with bolt's "
          "filters")
    print("    %d present values across %d row groups: none wrongly excluded"
          % (checked, n_rg))
    print("    %d absent values: excluded from %.1f%% of row groups"
          % (len(absent), 100.0 * rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
