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


# ---- single-column encoding fixtures (test_bolt_parquet_write_enc.cpp) ----
ENC_ROWS = 8000


def _i64(i):
    return i * 7 - 3


def _f64(i):
    return i * 0.25 - 3.0


def _str(i):
    return "com.example.package.name.%08d" % i


ENC_FIXTURES = [
    ("interop_dbp",     "DELTA_BINARY_PACKED",      _i64),
    ("interop_bss_i64", "BYTE_STREAM_SPLIT",        _i64),
    ("interop_bss_f64", "BYTE_STREAM_SPLIT",        _f64),
    ("interop_dba",     "DELTA_BYTE_ARRAY",         _str),
    ("interop_dlba",    "DELTA_LENGTH_BYTE_ARRAY",  _str),
    # DATA_PAGE_V2. The page TYPE is not visible through pyarrow's metadata,
    # but reading it is the assertion that matters: v2 stores the levels raw
    # ahead of separately-compressed values, so a size field that forgets the
    # level bytes, or a codec applied to the levels, fails here and only here.
    ("interop_v2",      "DELTA_BINARY_PACKED",      _i64),
]


def check_encoding_fixture(path, want_enc, gen, errs):
    """Values AND the declared encoding.

    The encoding assertion is not decoration: if the writer silently fell
    back to PLAIN, every value would still match and the test would pass
    while the feature did nothing. pyarrow reporting DELTA_BINARY_PACKED is
    what proves the delta stream was really produced -- and that pyarrow,
    not just bolt, can decode it.
    """
    md = pq.ParquetFile(path).metadata
    for r in range(md.num_row_groups):
        encs = set(md.row_group(r).column(0).encodings)
        if want_enc not in encs:
            errs.append("%s: rg%d declared %r, wanted %s"
                        % (path, r, sorted(encs), want_enc))
            return
    got = pq.read_table(path).column("v").to_pylist()
    if len(got) != ENC_ROWS:
        errs.append("%s: %d rows, want %d" % (path, len(got), ENC_ROWS))
        return
    for i in range(ENC_ROWS):
        want = gen(i)
        if got[i] != want:
            errs.append("%s: row %d: got %r want %r" % (path, i, got[i], want))
            return


def check_bloom_fixture(path, errs):
    """A file whose bloom filters sit BETWEEN row groups still reads.

    The filters are written after each row group's data, so their bytes fall
    inside the span a reader might naively walk as pages. If
    total_compressed_size wrongly counted them, a page walk would run off the
    end of the chunk -- and bolt's own reader is not the one to ask, since it
    would share the mistake. pyarrow is.
    """
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups < 2:
        errs.append("%s: expected several row groups, got %d"
                    % (path, md.num_row_groups))
        return
    got = pq.read_table(path).column("v").to_pylist()
    if len(got) != 30000:
        errs.append("%s: %d rows, want 30000" % (path, len(got)))
        return
    for i in range(30000):
        if got[i] != i * 3 - 7:
            errs.append("%s: row %d: got %r want %r" % (path, i, got[i], i * 3 - 7))
            return


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
    for tag, want_enc, gen in ENC_FIXTURES:
        path = os.path.join(d, "test_bolt_parquet_write_enc_%s.parquet" % tag)
        if not os.path.exists(path):
            errs.append("missing fixture %s (run test_bolt_parquet_write_enc "
                        "in that directory first)" % path)
            continue
        check_encoding_fixture(path, want_enc, gen, errs)
        checked += 1
        print("checked %s" % os.path.basename(path))
    bloom_path = os.path.join(
        d, "test_bolt_parquet_write_bloom_interleaved.parquet")
    if os.path.exists(bloom_path):
        check_bloom_fixture(bloom_path, errs)
        checked += 1
        print("checked %s" % os.path.basename(bloom_path))
    else:
        errs.append("missing fixture %s (run test_bolt_parquet_write_bloom "
                    "in that directory first)" % bloom_path)
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
