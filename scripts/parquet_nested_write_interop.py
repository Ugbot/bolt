#!/usr/bin/env python3
"""Check bolt's nested STRUCT / LIST / MAP writer against pyarrow (and DuckDB).

    ./test_bolt_parquet_nested_write               # writes fixtures + *.expect.json
    python3 scripts/parquet_nested_write_interop.py <dir>

Every "<fixture>.expect.json" names a parquet file, its expected rows, the
expected PARQUET:field_id of every node (arrow field paths), the expected
leaf column paths with max definition/repetition levels, the row group count
and, optionally, the leaf index RowGroup.sorting_columns must name. pyarrow
reads the file independently and must reproduce all of it; DuckDB (when
importable) must reproduce the rows. Exit 2 when pyarrow is missing.
"""
import datetime
import decimal
import glob
import json
import os
import re
import sys
import uuid

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("SKIP: pyarrow not installed")
    sys.exit(2)

try:
    import duckdb
except ImportError:
    duckdb = None

EPOCH = datetime.datetime(1970, 1, 1)
EPOCH_TZ = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


UUID_TEXT = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# DuckDB hands UUIDs back as canonical text; pyarrow as 16 bytes.
def norm(v, uuid_text=False):
    if isinstance(v, str) and uuid_text and UUID_TEXT.match(v):
        return v.replace("-", "")
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, uuid.UUID):
        return v.bytes.hex()
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, datetime.datetime):
        base = EPOCH_TZ if v.tzinfo is not None else EPOCH
        d = v - base
        return (d.days * 86400 + d.seconds) * 1000000 + d.microseconds
    if isinstance(v, datetime.date):
        return (v - EPOCH.date()).days
    if isinstance(v, dict):
        return {k: norm(x, uuid_text) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [norm(x, uuid_text) for x in v]
    raise TypeError(f"unhandled {type(v)}")


def field_ids(field, path, out):
    md = field.metadata or {}
    if b"PARQUET:field_id" in md:
        out[path] = int(md[b"PARQUET:field_id"])
    t = field.type
    if pa.types.is_struct(t):
        for i in range(t.num_fields):
            f = t.field(i)
            field_ids(f, path + "." + f.name, out)
    elif pa.types.is_map(t):
        field_ids(t.key_field, path + ".key", out)
        field_ids(t.item_field, path + ".value", out)
    elif pa.types.is_list(t) or pa.types.is_large_list(t):
        field_ids(t.value_field, path + ".element", out)


# The present values of one leaf, walked down its parquet path in arrow.
def leaf_values(table, path):
    segs = path.split(".")
    arr = table.column(segs[0]).combine_chunks()
    i = 1
    while i < len(segs):
        t = arr.type
        if pa.types.is_struct(t):
            arr = arr.flatten()[t.get_field_index(segs[i])]
            i += 1
        elif pa.types.is_map(t):
            arr = arr.keys if segs[i + 1] == "key" else arr.items
            i += 2
        else:
            arr = arr.flatten()
            i += 2
    return arr


# Chunk statistics against the decoded data: null_count is the level slots
# without a present value; min/max bracket exactly the present values.
def check_stats(pf, errs):
    import pyarrow.compute as pc
    md = pf.metadata
    for g in range(md.num_row_groups):
        rt = pf.read_row_group(g)
        for i in range(md.num_columns):
            c = md.row_group(g).column(i)
            st = c.statistics
            if st is None:
                continue
            vals = leaf_values(rt, c.path_in_schema)
            present = len(vals) - vals.null_count
            if st.has_null_count and st.null_count != c.num_values - present:
                errs.append(f"rg {g} {c.path_in_schema}: null_count {st.null_count}, "
                            f"slots {c.num_values}, present {present}")
            if not st.has_min_max or present == 0:
                continue
            if pa.types.is_decimal(vals.type) or pa.types.is_fixed_size_binary(vals.type):
                continue
            mm = pc.min_max(vals)
            if (mm["min"].as_py(), mm["max"].as_py()) != (st.min, st.max):
                errs.append(f"rg {g} {c.path_in_schema}: stats {(st.min, st.max)} "
                            f"!= data {(mm['min'].as_py(), mm['max'].as_py())}")


def diff_rows(tag, got, want, errs):
    if len(got) != len(want):
        errs.append(f"{tag}: {len(got)} rows, expected {len(want)}")
        return
    for r, (g, w) in enumerate(zip(got, want)):
        if g != w:
            errs.append(f"{tag}: row {r}\n  got  {g}\n  want {w}")
            return


def check(path_json):
    with open(path_json) as fh:
        exp = json.load(fh)
    fpath = os.path.join(os.path.dirname(path_json), exp["file"])
    errs = []
    pf = pq.ParquetFile(fpath)
    md = pf.metadata
    if md.num_row_groups != exp["row_groups"]:
        errs.append(f"row groups {md.num_row_groups} != {exp['row_groups']}")
    leaves = exp["leaves"]
    if md.num_columns != len(leaves):
        errs.append(f"leaf columns {md.num_columns} != {len(leaves)}")
    for i, lf in enumerate(leaves[: md.num_columns]):
        c = md.schema.column(i)
        got = (c.path, c.max_definition_level, c.max_repetition_level)
        want = (lf["path"], lf["max_def"], lf["max_rep"])
        if got != want:
            errs.append(f"leaf {i}: {got} != {want}")
    if exp.get("sort_leaf", -1) >= 0:
        for g in range(md.num_row_groups):
            sc = md.row_group(g).sorting_columns
            if len(sc) != 1 or sc[0].column_index != exp["sort_leaf"]:
                errs.append(f"rg {g} sorting_columns {sc}")
    table = pf.read()
    ids = {}
    for f in table.schema:
        field_ids(f, f.name, ids)
    if ids != exp["field_ids"]:
        errs.append(f"field ids\n  got  {ids}\n  want {exp['field_ids']}")
    diff_rows("pyarrow", norm(table.to_pylist()), exp["rows"], errs)
    check_stats(pf, errs)
    if duckdb is not None:
        con = duckdb.connect()
        dt = con.execute(f"SELECT * FROM read_parquet('{fpath}')").fetch_arrow_table()
        diff_rows("duckdb", norm(dt.to_pylist(), True), exp["rows"], errs)
    return exp["file"], len(exp["rows"]), errs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "nested_*.expect.json")))
    if not files:
        print(f"FAIL: no nested_*.expect.json in {d}")
        return 1
    failed = 0
    for p in files:
        name, n, errs = check(p)
        if errs:
            failed += 1
            print(f"FAIL {name} ({n} rows)")
            for e in errs:
                print("   ", e)
        else:
            print(f"ok   {name} ({n} rows){' +duckdb' if duckdb else ''}")
    print(f"pyarrow {pa.__version__}, duckdb {duckdb.__version__ if duckdb else 'absent'}: "
          f"{len(files) - failed}/{len(files)} fixtures ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
