#!/usr/bin/env python3
"""Check bolt's Utf8 parquet writer against pyarrow -- the half bolt cannot check.

    ./test_bolt_parquet_write_varbinary                    # writes the fixtures
    python3 scripts/parquet_varbinary_interop.py <dir-containing-them>

WHY THIS EXISTS. `BoltType::Utf8` occurs in two physical layouts: a StringView
array with a spill buffer (`ColumnFormat::Flat`, what the parquet reader and
every operator produce) and int32 offsets over a packed byte pool
(`ColumnFormat::VarBinary`, what MarbleDB produces -- `txn_batch` serializes
Utf8 that way and `bolt_wire_parse` rebuilds it, so every Utf8 column read out
of a sealed SST block is VarBinary). The writer only ever read the first, and
reading one as the other is a silent garbage read.

Round-tripping through bolt's own reader proves almost nothing about the FILE:
a writer and a reader that share a misreading agree perfectly. So this reads
the fixtures with pyarrow and compares against the model RE-DERIVED HERE from
its generating rules (tests/test_bolt_parquet_write_varbinary.cpp), not
against anything bolt emitted.

The model:
    row i is NULL   when i % 23 == 0
    row i is EMPTY  when i % 11 == 0 (and not null)
    otherwise       "row-<i>-" + ('x' * (i % 30))
    ts[i] == 1000 + i,  id[i] == i

Exit 0 = every fixture matches. Exit 1 = a mismatch. Exit 2 = could not run.
"""
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("SKIP: pyarrow not installed")
    sys.exit(2)

N_ROWS = 300

# Every fixture holds the SAME logical column; only the in-memory layout the
# writer was handed (and the row-group split) differs. A layout is a memory
# representation, not a file-format choice, so all four must agree.
FIXTURES = [
    ("pw_varbinary.parquet", "VarBinary offsets + packed pool (the MarbleDB shape)"),
    ("pw_flat.parquet", "Flat StringView + spill buffer"),
    ("pw_eq_varbinary.parquet", "VarBinary, byte-equality fixture"),
    ("pw_eq_flat.parquet", "Flat, byte-equality fixture"),
    ("pw_varbinary_split.parquet", "VarBinary across 5 row groups"),
]


def model(n=N_ROWS):
    """The expected content column, from the generating rules alone."""
    out = []
    for i in range(n):
        if i % 23 == 0:
            out.append(None)
        elif i % 11 == 0:
            out.append("")
        else:
            out.append("row-%d-" % i + "x" * (i % 30))
    return out


def check(path, note, expect):
    pf = pq.ParquetFile(path)
    n_groups = pf.metadata.num_row_groups
    table = pq.read_table(path)
    if table.num_rows != N_ROWS:
        return ["%s: %d rows, expected %d" % (path, table.num_rows, N_ROWS)]
    names = table.schema.names
    if names != ["ts", "id", "content"]:
        return ["%s: schema is %r" % (path, names)]

    bad = []
    ts = table.column("ts").to_pylist()
    ident = table.column("id").to_pylist()
    got = table.column("content").to_pylist()
    for i in range(N_ROWS):
        if ts[i] != 1000 + i:
            bad.append("%s: ts row %d is %r" % (path, i, ts[i]))
        if ident[i] != i:
            bad.append("%s: id row %d is %r" % (path, i, ident[i]))
        if got[i] != expect[i]:
            bad.append("%s: content row %d is %r, expected %r"
                       % (path, i, got[i], expect[i]))
        if len(bad) > 10:
            bad.append("%s: ... (further mismatches suppressed)" % path)
            return bad
    print("  OK  %-28s %d rows, %d row group(s) -- %s"
          % (os.path.basename(path), table.num_rows,
             n_groups, note))
    return bad


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    expect = model()
    missing = [f for f, _ in FIXTURES if not os.path.exists(os.path.join(d, f))]
    if missing:
        print("ERROR: run ./test_bolt_parquet_write_varbinary in %s first "
              "(missing: %s)" % (d, ", ".join(missing)))
        return 2

    failures = []
    for fname, note in FIXTURES:
        failures += check(os.path.join(d, fname), note, expect)

    # A refused write must leave NO file. Both of these are written up to the
    # point of failure and then removed; a footerless file left on disk has
    # the right name and extension and only fails much later, in whatever
    # tries to read it.
    for fname in ("pw_bad_layout.parquet", "pw_partial.parquet"):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            failures.append("%s: a refused write left a file behind" % p)
        else:
            print("  OK  %-28s absent, as a refused write must be" % fname)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nAll fixtures match the model under pyarrow %s."
          % __import__("pyarrow").__version__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
