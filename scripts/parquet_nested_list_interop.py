#!/usr/bin/env python3
"""Check bolt's NESTED LIST writer (G2PQ-27) against pyarrow.

    ./test_bolt_parquet_nested_list_write      # writes the fixtures
    python3 scripts/parquet_nested_list_interop.py <dir-containing-them>

WHY THIS EXISTS. Same argument as scripts/parquet_list_interop.py, one level
deeper: bolt's own reader REFUSES max_rep >= 2 (a list of lists) at the value
-assembly layer, so it cannot be used to check that bolt's WRITER emits a
correct nested Dremel level stream -- there is no bolt round-trip to compare
against, only pyarrow, a completely independent implementation.

The models (tests/test_bolt_parquet_nested_list_write.cpp) restated here:

Depth 2 -- List<List<Int32>>, column "li", n=60 rows:
    outer row r (0..n-1):
        r % 13 == 0                  -> NULL outer list
        r % 7  == 0 (else)           -> EMPTY outer list ([])
        else                         -> OL = (r % 4) + 1 inner lists
    inner list j of outer row r:
        (r + j) % 11 == 0            -> NULL inner list
        (r + j) % 5  == 0 (else)     -> EMPTY inner list ([])
        else                         -> LL = (j % 3) + 1 leaf values,
                                         value k = r*1000 + j*10 + k,
                                         NULL (leaf_nullable file only) when
                                         (r + j + k) % 9 == 0.

Depth 3 -- List<List<List<Int32>>>, column "li3": the depth-2 model above
    with n_mid=12, grouped 3-at-a-time into ceil(12/3)=4 outer-outer rows
    (all present, non-null -- only the *known-nested* levels underneath carry
    the null/empty mix; see the C++ test for the exact grouping).
"""
import os
import sys

import pyarrow.parquet as pq

N_ROWS_D2 = 60
N_MID_D3 = 12


def depth2_model(n, leaf_nullable):
    """[[...],[...],...] per outer row, or None for a NULL outer list."""
    rows = []
    for r in range(n):
        if r % 13 == 0:
            rows.append(None)
            continue
        if r % 7 == 0:
            rows.append([])
            continue
        ol = (r % 4) + 1
        inner = []
        for j in range(ol):
            g = r + j
            if g % 11 == 0:
                inner.append(None)
                continue
            if g % 5 == 0:
                inner.append([])
                continue
            ll = (j % 3) + 1
            vals = []
            for k in range(ll):
                if leaf_nullable and (r + j + k) % 9 == 0:
                    vals.append(None)
                else:
                    vals.append(r * 1000 + j * 10 + k)
            inner.append(vals)
        rows.append(inner)
    return rows


def check_depth2(path, leaf_nullable):
    table = pq.read_table(path)
    if table.num_columns != 1:
        return "expected 1 column, got %d" % table.num_columns
    field = table.schema.field(0)
    # list<list<int32>>
    ft = field.type
    if not str(ft).startswith("list<"):
        return "column is %s, not a list" % ft
    inner_field = ft.value_field
    if not str(inner_field.type).startswith("list<"):
        return "inner type is %s, not a list -- not actually nested" % inner_field.type
    leaf_field = inner_field.type.value_field
    if leaf_field.nullable != leaf_nullable:
        return ("leaf nullability is %s, expected %s"
                % (leaf_field.nullable, leaf_nullable))

    got = table.column(0).to_pylist()
    want = depth2_model(N_ROWS_D2, leaf_nullable)
    if len(got) != len(want):
        return "row count %d, expected %d" % (len(got), len(want))
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return "row %d: pyarrow read %r, model says %r" % (i, g, w)
    return None


def check_depth2_multirg(d):
    name = "test_bolt_parquet_nested_list_d2_multirg.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return "missing %s (run test_bolt_parquet_nested_list_write first)" % name
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups < 2:
        return "%s has %d row groups; the split path never ran" % (name, md.num_row_groups)
    table = pq.read_table(path)
    got = table.column(0).to_pylist()
    want = depth2_model(N_ROWS_D2, True)
    if got != want:
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return "%s row %d: pyarrow read %r, model says %r" % (name, i, g, w)
        return "%s: %d rows, model has %d" % (name, len(got), len(want))
    print("checked %s (%d row groups, %d rows)" % (name, md.num_row_groups, len(got)))
    return None


def required_levels_model(n=20):
    """test_bolt_parquet_required_list_d2.parquet's model:

    Same generator as depth2_model(n, leaf_nullable=True), but the C++ test
    force-clears the outer/inner VALIDITY bitmaps entirely (mirroring a real
    REQUIRED declaration -- a null-bearing column claiming REQUIRED would be
    a caller bug, not what this fixture tests), which turns every would-be
    NULL outer/inner list into an EMPTY one (the offsets already span zero
    elements either way; only the validity bit differed).
    """
    base = depth2_model(n, True)
    out = []
    for row in base:
        if row is None:
            out.append([])
            continue
        out.append([[] if inner is None else inner for inner in row])
    return out


def check_required_levels_d2(d):
    """The regression fixture for the REQUIRED-outer max_def bug (G2PQ-27):
    pre-fix, chunk_write_list's max_def formula ignored the column's own
    nullability and always assumed the outer LIST group was OPTIONAL. Schema
    -only checks (bolt's own parquet_read_meta) cannot catch this: the
    SCHEMA was always written correctly (nullable=false), so a schema walk
    re-derives the correct max_def regardless of what the DATA pages
    actually used -- confirmed by deliberately reintroducing the bug and
    finding the schema-only C++ assertions still passed. Only re-reading the
    VALUES (here, pyarrow, since bolt's own reader refuses max_rep >= 2)
    actually exercises the level bytes at the width the schema declares.
    """
    name = "test_bolt_parquet_required_list_d2.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return "missing %s (run test_bolt_parquet_nested_list_write first)" % name
    field = pq.ParquetFile(path).schema_arrow.field(0)
    if field.nullable:
        return "%s: column declared nullable, expected REQUIRED" % name
    table = pq.read_table(path)
    got = table.column(0).to_pylist()
    want = required_levels_model(20)
    if got != want:
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return "%s row %d: pyarrow read %r, model says %r" % (name, i, g, w)
        return "%s: %d rows, model has %d" % (name, len(got), len(want))
    print("checked %s (%d rows, REQUIRED outer+inner)" % (name, len(got)))
    return None


def check_depth3(d):
    name = "test_bolt_parquet_nested_list_d3.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return "missing %s (run test_bolt_parquet_nested_list_write first)" % name
    table = pq.read_table(path)
    field = table.schema.field(0)
    ft = field.type
    # list<list<list<int32>>>
    for level in range(2):
        if not str(ft).startswith("list<"):
            return "%s: type %s is not nested enough (level %d)" % (name, ft, level)
        ft = ft.value_field.type
    if not str(ft).startswith("list<"):
        return "%s: type %s is not a 3-level list" % (name, ft)

    mid = depth2_model(N_MID_D3, True)
    want = [mid[i:i + 3] for i in range(0, N_MID_D3, 3)]
    got = table.column(0).to_pylist()
    if got != want:
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return "%s row %d: pyarrow read %r, model says %r" % (name, i, g, w)
        return "%s: %d rows, model has %d" % (name, len(got), len(want))
    print("checked %s (%d rows, 3 levels deep)" % (name, len(got)))
    return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [("test_bolt_parquet_nested_list_d2_er.parquet", False),
             ("test_bolt_parquet_nested_list_d2_en.parquet", True)]
    missing = [f for f, _ in files if not os.path.exists(os.path.join(d, f))]
    if missing:
        print("missing fixtures (run test_bolt_parquet_nested_list_write first): %s"
              % ", ".join(missing), file=sys.stderr)
        return 2

    rc = 0
    for name, leaf_nullable in files:
        path = os.path.join(d, name)
        err = check_depth2(path, leaf_nullable)
        if err:
            print("FAIL %s: %s" % (name, err))
            rc = 1
        else:
            print("checked %s (leaf_nullable=%s)" % (name, leaf_nullable))

    for check in (check_depth2_multirg, check_depth3, check_required_levels_d2):
        err = check(d)
        if err:
            print("FAIL: %s" % err)
            rc = 1

    if rc:
        return rc

    # Prove the comparison can fail -- a gate that cannot distinguish a
    # NULL/EMPTY/value mismatch at the INNER level is not actually checking
    # the nested case, only the flat outer shape.
    path = os.path.join(d, files[0][0])
    table = pq.read_table(path)
    want = depth2_model(N_ROWS_D2, False)

    def mutate_inner_empty_to_null(rows):
        out = []
        for row in rows:
            if row is None:
                out.append(None)
                continue
            out.append([None if inner == [] else inner for inner in row])
        return out

    def mutate_inner_null_to_empty(rows):
        out = []
        for row in rows:
            if row is None:
                out.append(None)
                continue
            out.append([[] if inner is None else inner for inner in row])
        return out

    def mutate_drop_last_leaf(rows):
        out = []
        for row in rows:
            if row is None:
                out.append(None)
                continue
            out.append([inner[:-1] if isinstance(inner, list) and inner else inner
                        for inner in row])
        return out

    got = table.column(0).to_pylist()
    injections = [
        ("inner empty read as null", mutate_inner_empty_to_null),
        ("inner null read as empty", mutate_inner_null_to_empty),
        ("one leaf element dropped", mutate_drop_last_leaf),
    ]
    for label, mut in injections:
        mutated_want = mut(want)
        if got == mutated_want:
            print("FAIL: injection '%s' was NOT caught -- this gate is "
                  "not discriminating" % label)
            rc = 1
    if rc == 0:
        print("\nOK: pyarrow agrees with the model on depth-2 (both "
              "nullability shapes), the multi-row-group split, and depth-3; "
              "all 3 injections caught")
    return rc


if __name__ == "__main__":
    sys.exit(main())
