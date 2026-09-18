#!/usr/bin/env python3
"""Regenerate the NESTED LIST/MAP parquet fixtures for G2PQ-15 (max_rep >= 2).

Written by pyarrow deliberately, exactly like make_list_fixtures.py: a
fixture bolt produced could only prove bolt self-consistent, and the whole
risk with multi-level Dremel assembly is a shared misreading of the spec.
The C++ test restates these same closed forms and compares against them, so
neither side reads its expectation out of the file.

    python3 scripts/make_list_nested_fixtures.py

Shapes covered:
  * outer list NULL, outer list EMPTY (different values, one leaf slot each)
  * outer list with elements where an ELEMENT's own inner list is itself
    NULL or EMPTY -- max_rep == 2, the case bolt could not assemble before
    G2PQ-15 (list<list<int64>>, column "c")
  * a NULL leaf value inside a present, non-empty inner list
  * a MAP whose VALUE is itself a LIST (map<string, list<int64>>, column
    "mp") -- a different schema shape (repeated key_value -> struct{key,
    value} -> repeated list) than list<list<T>>, exercising the schema walk
    generally rather than only the one shape it was designed against
  * three levels of nested repetition (list<list<list<int64>>>, column
    "c3") -- proves the assembly is not hard-coded to R == 2
  * a flat column in the same file, which must keep reading normally
"""
import pyarrow as pa
import pyarrow.parquet as pq

N = 500


def outer_len(i):
    return (i % 4) + 1


def inner_kind(i, j):
    """0 = NULL inner list, 1 = EMPTY inner list, 2 = has values."""
    s = i + j
    if s % 13 == 0:
        return 0
    if s % 7 == 0:
        return 1
    return 2


def inner_len(i, j):
    return (i + j) % 3 + 1


def leaf_value(i, j, k):
    return i * 1000 + j * 10 + k


def leaf_is_null(i, j, k):
    return (i + j + k) % 17 == 0


def build_c2():
    """column 'c': list<list<int64>>, the R==2 flagship case."""
    rows = []
    for i in range(N):
        if i % 19 == 0:
            rows.append(None)                 # outer list NULL
            continue
        if i % 11 == 0:
            rows.append([])                   # outer list EMPTY (present, 0 elems)
            continue
        outer = []
        for j in range(outer_len(i)):
            kind = inner_kind(i, j)
            if kind == 0:
                outer.append(None)             # inner list NULL
            elif kind == 1:
                outer.append([])               # inner list EMPTY
            else:
                inner = []
                for k in range(inner_len(i, j)):
                    inner.append(None if leaf_is_null(i, j, k)
                                 else leaf_value(i, j, k))
                outer.append(inner)
        rows.append(outer)
    return rows


def build_c3():
    """column 'c3': list<list<list<int64>>>, R==3 -- a deeper stretch check.

    Reuses c2's shape one level further out: row i is NULL/EMPTY by the same
    (i%19, i%11) rule as c2; otherwise it has outer_len(i) elements, each of
    which IS an independent c2-shaped sub-row keyed by (i*31 + j) so the
    values differ from c2's own.
    """
    rows = []
    for i in range(N):
        if i % 19 == 0:
            rows.append(None)
            continue
        if i % 11 == 0:
            rows.append([])
            continue
        outer = []
        for j in range(outer_len(i)):
            mid_key = i * 31 + j
            mkind = inner_kind(i, j)
            if mkind == 0:
                outer.append(None)
                continue
            if mkind == 1:
                outer.append([])
                continue
            mid = []
            for m in range(inner_len(i, j)):
                kind2 = inner_kind(mid_key, m)
                if kind2 == 0:
                    mid.append(None)
                elif kind2 == 1:
                    mid.append([])
                else:
                    leaves = []
                    for k in range(inner_len(mid_key, m)):
                        leaves.append(None if leaf_is_null(mid_key, m, k)
                                      else leaf_value(mid_key, m, k))
                    mid.append(leaves)
            outer.append(mid)
        rows.append(outer)
    return rows


def build_map_of_list():
    """column 'mp': map<string, list<int64>> -- value leaf has max_rep == 2
    via a DIFFERENT schema shape (key_value struct, not element-of-a-list)."""
    rows = []
    for i in range(N):
        if i % 23 == 0:
            rows.append(None)                 # whole map NULL
            continue
        nkeys = i % 4
        entries = []
        for j in range(nkeys):
            # offset from c2's (i, j) by +1 so the kind/length/value sequence
            # differs from column "c"'s -- but consistently, in ALL THREE, so
            # this is just model_inner(i, j + 1) (see
            # tests/test_bolt_parquet_list_nested.cpp's model_mp_value_entry).
            jj = j + 1
            kind = inner_kind(i, jj)
            key = "k%d" % j
            if kind == 0:
                entries.append((key, None))
            elif kind == 1:
                entries.append((key, []))
            else:
                vals = [None if leaf_is_null(i, jj, k) else leaf_value(i, jj, k)
                        for k in range(inner_len(i, jj))]
                entries.append((key, vals))
        rows.append(entries)
    return rows


def build():
    return pa.table({
        "c": pa.array(build_c2(), pa.list_(pa.list_(pa.int64()))),
        "c3": pa.array(build_c3(), pa.list_(pa.list_(pa.list_(pa.int64())))),
        "mp": pa.array(build_map_of_list(), pa.map_(pa.string(), pa.list_(pa.int64()))),
        "flat": pa.array(list(range(N)), pa.int64()),
    })


def build_deep(depth):
    """A single all-required, all-non-null column nested `depth` deep, each
    row holding exactly one element per level down to one leaf value -- just
    enough structure to prove the SCHEMA parses (or is cleanly refused) at
    a given depth; values are not exercised by the refusal test."""
    def ty(d):
        return pa.int64() if d == 0 else pa.list_(ty(d - 1))

    def val(d):
        return 0 if d == 0 else [val(d - 1)]

    return pa.table({"d": pa.array([val(depth) for _ in range(3)], ty(depth))})


def main():
    t = build()
    pq.write_table(t, "tests/data/golden_list_nested2.parquet",
                   compression="snappy", version="2.6", use_dictionary=False)
    pq.write_table(t, "tests/data/golden_list_nested2_dict.parquet",
                   compression="snappy", version="1.0", use_dictionary=True)
    # kPqMaxRepLevels is 8: 10 levels of list nesting must be refused, not
    # misassembled or silently truncated (A2's stated scope boundary).
    deep = build_deep(10)
    pq.write_table(deep, "tests/data/golden_list_too_deep.parquet",
                   compression="snappy", version="2.6", use_dictionary=False)
    print("wrote tests/data/golden_list_nested2{,_dict}.parquet + "
          "golden_list_too_deep.parquet")


if __name__ == "__main__":
    main()
