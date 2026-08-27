#!/usr/bin/env python3
"""Check bolt's LIST *writer* against pyarrow -- the half bolt cannot check.

    ./test_bolt_parquet_list            # writes the fixtures
    python3 scripts/parquet_list_interop.py <dir-containing-them>

WHY THIS EXISTS. bolt's list writer generates Dremel repetition/definition
levels by inverting the assembly its own reader performs. Round-tripping
through that reader therefore proves almost nothing about the FILE: an
encoder and a decoder that share a misreading of the level semantics agree
perfectly with each other while every other reader in the world sees
different data. The same argument is already written down for snappy in
bolt_snappy.h, and it applies with far more force here, because a null list,
an EMPTY list, and a list containing a null differ only in level bits -- they
carry the same element counts, so a level bug preserves row counts, list
counts, and every value, and shows up only as lists whose contents moved.

So this reads the written files with pyarrow and compares against the model
RE-DERIVED HERE from its generating rules, not against anything bolt emitted.

The model (tests/test_bolt_parquet_list.cpp, make_list_model):
    row i is NULL      when i % 17 == 0
    row i is EMPTY     when i % 7 == 0   (and not null)
    otherwise length   (i % 5) + 1
    element j of row i is i*10 + j
    element j is NULL  when (i + j) % 11 == 0, and only in the
                       element-nullable file
"""
import os
import sys

import pyarrow.parquet as pq

N_ROWS = 500


def model(elem_nullable, n=N_ROWS):
    """The expected column, derived from the generating rules alone."""
    rows = []
    for i in range(n):
        if i % 17 == 0:
            rows.append(None)
            continue
        if i % 7 == 0:
            rows.append([])
            continue
        vals = []
        for j in range((i % 5) + 1):
            if elem_nullable and (i + j) % 11 == 0:
                vals.append(None)
            else:
                vals.append(i * 10 + j)
        rows.append(vals)
    return rows


def check(path, elem_nullable, mutate=None):
    table = pq.read_table(path)
    if table.num_columns != 1:
        return "expected 1 column, got %d" % table.num_columns

    field = table.schema.field(0)
    if not str(field.type).startswith("list<"):
        return "column is %s, not a list" % field.type
    # The 3-level LIST shape must also carry element nullability, or a
    # consumer building its own schema from ours gets it wrong.
    if field.type.value_field.nullable != elem_nullable:
        return ("element nullability is %s, expected %s"
                % (field.type.value_field.nullable, elem_nullable))

    got = table.column(0).to_pylist()
    want = model(elem_nullable)
    if mutate is not None:
        want = mutate(want)
    if len(got) != len(want):
        return "row count %d, expected %d" % (len(got), len(want))
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            # Spell out the three cases that a level bug confuses, because
            # "[] != None" on its own reads like a formatting difference.
            def what(v):
                if v is None:
                    return "NULL list"
                if v == []:
                    return "EMPTY list"
                return "list%r" % (v,)
            return ("row %d: pyarrow read %s, model says %s"
                    % (i, what(g), what(w)))
    return None


def check_multirg(d):
    """The list column SPLIT ACROSS ROW GROUPS, read by pyarrow.

    This is the shape that exposed a real writer bug: a list column is sliced
    for each row group by advancing its OFFSETS, and the values in those
    offsets are absolute indices into one shared element array. Every row
    group after the first was written from element 0 of the whole column, so
    the page's declared value count disagreed with its payload and pyarrow
    rejected the file outright ("Unexpected end of stream"). Every list
    fixture before this used a single row group, so nothing caught it.
    """
    name = "test_bolt_parquet_list_multirg.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return "missing %s (run test_bolt_parquet_list first)" % name
    table = pq.read_table(path)
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups < 2:
        return "%s has %d row groups; the split path never ran" % (
            name, md.num_row_groups)
    got = table.column(0).to_pylist()
    want = model(True, n=900)
    if got != want:
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return ("%s row %d: pyarrow read %r, model says %r"
                        % (name, i, g, w))
        return "%s: %d rows, model has %d" % (name, len(got), len(want))
    print("checked %s (%d row groups, %d rows)"
          % (name, md.num_row_groups, len(got)))
    return None


def check_json_annotation(d):
    """bolt's JSON logical annotation, read by an independent reader.

    Same blind spot as the lists above, one layer down: bolt writes
    LogicalType field 12 and bolt maps field 12 back to Json, so a wrong
    field number round-trips perfectly through bolt and is simply a
    different type to everyone else. pyarrow surfaces the annotation as an
    `arrow.json` extension type (or, on older builds, records it in the
    parquet schema's logical type), so ask both ways rather than pinning a
    single pyarrow version's spelling.
    """
    name = "test_bolt_parquet_json_written.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return "missing %s (run test_bolt_parquet_json first)" % name
    pf = pq.ParquetFile(path)
    lt = str(pf.schema_arrow.field(0).type)
    logical = str(pf.schema.column(0).logical_type)
    if "json" not in lt.lower() and "json" not in logical.lower():
        return ("column 'j' carries no JSON annotation: arrow type %s, "
                "parquet logical type %s" % (lt, logical))
    print("checked %s (arrow=%s, parquet logical=%s)" % (name, lt, logical))
    return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [("test_bolt_parquet_list_written_er.parquet", False),
             ("test_bolt_parquet_list_written_en.parquet", True)]

    missing = [f for f, _ in files if not os.path.exists(os.path.join(d, f))]
    if missing:
        print("missing fixtures (run test_bolt_parquet_list first): %s"
              % ", ".join(missing), file=sys.stderr)
        return 2

    rc = 0
    for name, elem_nullable in files:
        path = os.path.join(d, name)
        err = check(path, elem_nullable)
        if err:
            print("FAIL %s: %s" % (name, err))
            rc = 1
        else:
            print("checked %s (element_nullable=%s)" % (name, elem_nullable))

    err = check_multirg(d)
    if err:
        print("FAIL: %s" % err)
        rc = 1

    err = check_json_annotation(d)
    if err:
        print("FAIL: %s" % err)
        rc = 1

    if rc:
        return rc

    # Prove the comparison can fail. A gate that cannot distinguish the three
    # list states would pass on a file it never really examined -- and those
    # three states are exactly what a Dremel level bug swaps.
    path = os.path.join(d, files[0][0])
    injections = [
        ("empty read as null", lambda w: [None if v == [] else v for v in w]),
        ("null read as empty", lambda w: [[] if v is None else v for v in w]),
        ("one element dropped",
         lambda w: [v[:-1] if isinstance(v, list) and v else v for v in w]),
    ]
    for label, mut in injections:
        if check(path, files[0][1], mutate=mut) is None:
            print("FAIL: injection '%s' was NOT caught -- this gate is "
                  "not discriminating" % label)
            rc = 1
    if rc == 0:
        print("\nOK: pyarrow agrees with the model on both files; "
              "all 3 injections caught")
    return rc


if __name__ == "__main__":
    sys.exit(main())
