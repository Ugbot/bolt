#!/usr/bin/env python3
"""Read bolt-written parquet files with pyarrow and check them value-by-value.

Why this exists as a separate gate: a bolt-to-bolt round trip proves the
writer and reader are self-consistent, which is exactly what you get when
both share a misreading of the spec. Only a reference implementation settles
whether the bytes are parquet. That lesson is already in this tree twice --
see docs/research/parquet-reader-completeness-plan.md.

Run the fixture-producing test first, then point this at its output dir:

    ./test_bolt_parquet_write_dict
    python3 scripts/parquet_write_interop.py build/pq/tests

Checks per fixture:
  * pyarrow opens the file and agrees on row count
  * every VALUE matches the model the C++ test wrote (regenerated here from
    the same closed form, NOT read out of the file)
  * the declared encodings are what the writer intended (a dictionary fixture
    must really carry RLE_DICTIONARY, not silently fall back to PLAIN)
  * the page index, where requested, is FOUND by an independent reader --
    presence only; see check_page_index for why the contents are gated on the
    C++ side instead

This script is itself checked for discriminating power: perturbing the model
by one value, or flipping a fixture's expected encoding or index presence,
must make it fail. If it cannot fail it is not a gate.
"""

import sys
import os

import pyarrow.parquet as pq

# (tag, distinct, nullable, want_dictionary, want_page_index)
# Mirrors the `cases` table in tests/test_bolt_parquet_write_dict.cpp.
ROWS = 8000
FIXTURES = [
    ("interop_dict_snappy",   50, False, True,  True),
    ("interop_dict_plain",     7, False, True,  False),
    ("interop_dict_nullable", 13, True,  True,  True),
    ("interop_split_plain",    0, False, False, True),
    ("interop_dict_bw0",       1, False, True,  True),
]
NULL_EVERY = 5


def model(n, distinct, nullable):
    """The same closed form make_model() uses in the C++ test."""
    ints, strs, valid = [], [], []
    for i in range(n):
        d = (i % distinct) if distinct > 0 else i
        ints.append(d * 7 - 3)
        strs.append("v%d" % d)
        valid.append(not (nullable and (i % NULL_EVERY) == 0))
    return ints, strs, valid


def check_values(path, distinct, nullable, errs):
    tbl = pq.read_table(path)
    ints, strs, valid = model(ROWS, distinct, nullable)
    if tbl.num_rows != ROWS:
        errs.append("%s: pyarrow saw %d rows, want %d"
                    % (path, tbl.num_rows, ROWS))
        return
    got_id = tbl.column("id").to_pylist()
    got_nm = tbl.column("name").to_pylist()
    for i in range(ROWS):
        if not valid[i]:
            if got_id[i] is not None or got_nm[i] is not None:
                errs.append("%s: row %d should be null, got %r/%r"
                            % (path, i, got_id[i], got_nm[i]))
                return
            continue
        if got_id[i] != ints[i]:
            errs.append("%s: id row %d: got %r want %r"
                        % (path, i, got_id[i], ints[i]))
            return
        if got_nm[i] != strs[i]:
            errs.append("%s: name row %d: got %r want %r"
                        % (path, i, got_nm[i], strs[i]))
            return


def check_encoding(path, want_dict, errs):
    md = pq.ParquetFile(path).metadata
    for r in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(r).column(c)
            encs = set(col.encodings)
            has_dict = ("RLE_DICTIONARY" in encs or
                        "PLAIN_DICTIONARY" in encs)
            # BOOLEAN is never dictionary encoded; these fixtures have none.
            if want_dict and not has_dict:
                errs.append("%s: rg%d col%d declared %r, wanted a dictionary"
                            % (path, r, c, sorted(encs)))
            if want_dict and not col.has_dictionary_page:
                errs.append("%s: rg%d col%d has no dictionary page"
                            % (path, r, c))
            if not want_dict and has_dict:
                errs.append("%s: rg%d col%d unexpectedly dictionary encoded"
                            % (path, r, c))


def check_page_index(path, want_index, errs):
    """Presence only, deliberately.

    pyarrow 21 surfaces has_column_index / has_offset_index on the column
    chunk but gives Python no way to READ the two structs, so this checks the
    half that pyarrow can settle: that an independent implementation finds the
    locators exactly where the spec puts them (ColumnChunk fields 4-7), which
    is what makes the index reachable at all.

    The CONTENTS are gated on the C++ side instead, by bolt's own index
    parser -- test_bolt_parquet_write_dict.PageIndexDescribesTheRealPages
    checks page counts agree, row windows tile the chunk without gaps, every
    page's declared bounds bracket that page's actual values, null counts sum
    to the chunk's, and boundary_order is proven rather than guessed.
    """
    md = pq.ParquetFile(path).metadata
    for r in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(r).column(c)
            if col.has_column_index != want_index:
                errs.append("%s: rg%d col%d has_column_index=%s, wanted %s"
                            % (path, r, c, col.has_column_index, want_index))
            if col.has_offset_index != want_index:
                errs.append("%s: rg%d col%d has_offset_index=%s, wanted %s"
                            % (path, r, c, col.has_offset_index, want_index))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    errs = []
    checked = 0
    for tag, distinct, nullable, want_dict, want_idx in FIXTURES:
        path = os.path.join(d, "test_bolt_parquet_write_dict_%s.parquet" % tag)
        if not os.path.exists(path):
            errs.append("missing fixture %s (run test_bolt_parquet_write_dict "
                        "in that directory first)" % path)
            continue
        check_values(path, distinct, nullable, errs)
        check_encoding(path, want_dict, errs)
        check_page_index(path, want_idx, errs)
        checked += 1
        print("checked %s" % os.path.basename(path))
    if errs:
        print("\nFAIL (%d):" % len(errs))
        for e in errs:
            print("  " + e)
        return 1
    print("\nOK: %d fixtures read by pyarrow, values/encodings/page index "
          "all agree" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
