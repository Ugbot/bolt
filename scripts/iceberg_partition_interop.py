#!/usr/bin/env python3
"""G2ICE-292: do Iceberg readers that are not bolt agree with bolt's partitioned tables?

    uv run --no-project --python 3.12 --with 'pyiceberg[pyiceberg-core]' --with pyarrow --with 'duckdb==1.4.*' \\
        python scripts/iceberg_partition_interop.py <build>/tests/test_bolt_iceberg_partitioned <scratch dir>

The executable's IcebergPartitionFixture.WriteTables writes one table per (transform, source type) over a 12-column
schema (long, int, string, date, timestamp, timestamptz, time, decimal(9,2), uuid, binary, fixed[4]), a table with
four partition fields, a spec-evolution table (identity(k), then void(k) + day(ts)) and delete / overwrite / compaction
tables, and lists them in tables.txt. For each table this script checks, against a model it computes itself from the
same row formula as the fixture:
  * rows: pyiceberg (StaticTable) and DuckDB (iceberg_scan) return the model's rows;
  * transforms: for every live data file, the partition tuple bolt recorded equals pyiceberg's own transform
    (pyiceberg.transforms, under the file's spec) of every row physically in the file, and the file's spec id is
    the spec it was written under;
  * pruning: for predicates of every operator on the partition source, pyiceberg plan_files never skips a file
    holding a matching row, and skips every file whose tuple pyiceberg's inclusive projection of the predicate
    rejects (computed from the expected tuple, not bolt's); DuckDB answers the same predicates with the model's rows;
  * deletes: every position-delete file carries the spec id and tuple of each data file it names.
Two reader defects are excluded by name, each reproduced on a table pyiceberg itself wrote (not bolt):
  * DuckDB 1.4 answers `<` and `>` on an identity-partitioned date / timestamp / time / decimal column with no rows
    (`<=`, `=`, `>=` are right): those two operators are not asked of those tables;
  * pyiceberg 0.12 binds a uuid literal as bytes while its partition Record holds a uuid.UUID, so plan_files skips
    every file of an identity(uuid) table under `u = <uuid>`: that predicate is not asked of identity(uuid).
Negative controls prove the checks can fail: every table's tuples are checked against a deliberately wrong
transform (bucket[N+1], truncate[W+1], another time unit, ...), a copy of a bucket table's and a truncate table's
metadata whose spec says bucket[5] / truncate[11] is planned, and the DML tables' rows are compared against a model
with the predicate ignored. Each must be reported.

Exit 0 = pass; 1 = a check failed or a negative control did not fail; 77 = pyiceberg or duckdb not installed (SKIP).
"""
import copy
import datetime
import decimal
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

try:
    import duckdb
    import pyarrow.parquet as pq
    from pyiceberg.expressions import (EqualTo, GreaterThan, GreaterThanOrEqual, IsNull, LessThan,
                                       LessThanOrEqual, NotEqualTo, NotNull)
    from pyiceberg.expressions.visitors import expression_evaluator, inclusive_projection
    from pyiceberg.manifest import DataFileContent
    from pyiceberg.schema import Schema
    from pyiceberg.table import StaticTable
    from pyiceberg.transforms import (BucketTransform, DayTransform, HourTransform, IdentityTransform,
                                      MonthTransform, TruncateTransform, VoidTransform, YearTransform)
    from pyiceberg.typedef import Record
except ImportError as e:  # a missing oracle is a SKIP, never a PASS
    print("SKIP: %s" % e)
    sys.exit(77)

COLS = ["id", "k", "s", "d", "ts", "i", "tz", "tm", "dec", "u", "b", "fx"]
DAY = 86400000000
EPOCH = datetime.date(1970, 1, 1)


# ---- the fixture's row formula (tests/test_bolt_iceberg_partitioned.cpp: wide_null / wide_int / wide_bytes) ----

def wide_null(i, c):
    mod = [0, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    rem = [0, 3, 5, 7, 9, 2, 4, 6, 8, 10, 12, 14]
    return c != 0 and i % mod[c] == rem[c]


def wide_value(i, c):
    """Canonical value: ints, days, micros, unscaled decimal, str, bytes."""
    if wide_null(i, c):
        return None
    d = -400 + i * 37
    if c == 0:
        return i
    if c == 1:
        return i % 4
    if c == 2:
        return ("café-" if i % 3 == 0 else "row-") + str(i % 9) + ("-tail-long-enough-to-spill" if i % 2 else "")
    if c == 3:
        return d
    if c == 4:
        return (0 if wide_null(i, 3) else d) * DAY + (i % 24) * 3600000000 - 1
    if c == 5:
        return -50 + i * 3
    if c == 6:
        return 1510871468000000 + (i - 30) * 7777777777
    if c == 7:
        return (i * 1234567891) % 86400000000
    if c == 8:
        return i * 137 - 3000
    if c == 9:
        return bytes((i * 17 + j * 31) % 256 for j in range(16))
    if c == 10:
        return bytes((i + j * 13) % 256 for j in range(i % 5))
    return bytes((i * 3 + j) % 256 for j in range(4))


def model_row(i):
    return tuple(wide_value(i, c) for c in range(len(COLS)))


def col(r, name):
    return r[COLS.index(name)]


def base_rows():
    return [model_row(i) for i in range(60)]


def model_rows(kind):
    rows = base_rows()
    if kind == "delete":
        rows = [r for r in rows if col(r, "k") != 1 and not (col(r, "i") is not None and col(r, "i") < 25)]
    elif kind == "overwrite":
        rows = [r for r in rows if col(r, "s") != "row-4"] + [model_row(i) for i in range(100, 112)]
    elif kind == "compact":
        rows = [r for r in rows if not (col(r, "i") is not None and col(r, "i") < 25)]
    return sorted(rows, key=lambda r: r[0])


def wrong_rows(kind):   # negative control: the DML predicate ignored
    if kind == "overwrite":
        return sorted(base_rows() + [model_row(i) for i in range(100, 112)], key=lambda r: r[0])
    return sorted(base_rows(), key=lambda r: r[0])


# ---- normalising reader values to the canonical form ----

def micros(dt):
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    delta = dt - datetime.datetime(1970, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def canon(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, datetime.datetime):
        return micros(v)
    if isinstance(v, datetime.date):
        return (v - EPOCH).days
    if isinstance(v, datetime.time):
        return ((v.hour * 60 + v.minute) * 60 + v.second) * 1000000 + v.microsecond
    if isinstance(v, decimal.Decimal):
        return int(v.scaleb(2))
    if isinstance(v, uuid.UUID):
        return v.bytes
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    return v


def latest_metadata(table_dir):
    paths = glob.glob(os.path.join(table_dir, "metadata", "v*.metadata.json"))
    return max(paths, key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))


def pyiceberg_rows(meta, row_filter=None):
    scan = StaticTable.from_metadata("file://" + meta).scan(**({"row_filter": row_filter} if row_filter else {}))
    t = scan.to_arrow()
    cols = [t.column(c).to_pylist() for c in COLS]
    return sorted((tuple(canon(x[n]) for x in cols) for n in range(t.num_rows)), key=lambda r: r[0])


_con = None


def duckdb_rows(meta, where=""):
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.execute("INSTALL iceberg; LOAD iceberg;")
    sel = ("id, k, s, d - DATE '1970-01-01', epoch_us(ts), i, epoch_us(tz), CAST(tm AS VARCHAR), "
           "CAST(dec * 100 AS BIGINT), u, b, fx")
    out = []
    for r in _con.execute("SELECT %s FROM iceberg_scan('%s') %s" % (sel, meta, where)).fetchall():
        r = list(r)
        if r[7] is not None:
            r[7] = canon(datetime.time.fromisoformat(r[7]))
        out.append(tuple(canon(x) for x in r))
    return sorted(out, key=lambda r: r[0])


def check_rows(name, meta, want):
    fails = []
    for reader, fn in (("pyiceberg", pyiceberg_rows), ("duckdb", duckdb_rows)):
        try:
            got = fn(meta)
        except Exception as e:  # noqa: BLE001 - any reader error is a failed check
            fails.append("%s: %s read failed: %s" % (name, reader, e))
            continue
        if got != want:
            fails.append("%s: %s rows differ (%d vs %d; first diff %s)"
                         % (name, reader, len(got), len(want),
                            next(((a, b) for a, b in zip(got, want) if a != b), "length")))
    return fails


# ---- transforms ----

def to_pyiceberg(v, t):
    """Canonical value -> pyiceberg's internal representation of source type `t`."""
    if v is None:
        return None
    n = str(t)
    if n.startswith("decimal"):
        return decimal.Decimal(v).scaleb(-t.scale)
    if n == "uuid":
        return uuid.UUID(bytes=v)
    return v


def wrong_transform(tr):
    """A transform that must disagree with `tr` on these rows (negative control)."""
    if isinstance(tr, BucketTransform):
        return BucketTransform(tr.num_buckets + 1)
    if isinstance(tr, TruncateTransform):
        return TruncateTransform(tr.width + 1)
    if isinstance(tr, YearTransform):
        return MonthTransform()
    if isinstance(tr, MonthTransform):
        return YearTransform()
    if isinstance(tr, DayTransform):
        return MonthTransform()
    if isinstance(tr, HourTransform):
        return DayTransform()
    if isinstance(tr, VoidTransform):
        return IdentityTransform()
    return VoidTransform()


def live_entries(tbl):
    snap = tbl.current_snapshot()
    out = []
    for m in snap.manifests(tbl.io):
        for e in m.fetch_manifest_entry(tbl.io, discard_deleted=True):
            out.append(e.data_file)
    return out


def file_rows(path):
    t = pq.read_table(path.replace("file://", ""))
    cols = [t.column(c).to_pylist() for c in COLS]
    return [tuple(canon(x[n]) for x in cols) for n in range(t.num_rows)]


def expected_tuple(tbl, spec, rows, wrong=False):
    """The tuple pyiceberg computes for these rows under `spec` (None when they disagree)."""
    tuples = set()
    for r in rows:
        tup = []
        for f in spec.fields:
            src = tbl.schema().find_field(f.source_id)
            tr = wrong_transform(f.transform) if wrong else f.transform
            fn = tr.transform(src.field_type)
            tup.append(canon(fn(to_pyiceberg(col(r, src.name), src.field_type))))
        tuples.add(tuple(tup))
    return tuples.pop() if len(tuples) == 1 else ("<rows of several tuples>", len(tuples))


def check_transforms(name, meta, wrong=False):
    fails = []
    tbl = StaticTable.from_metadata("file://" + meta)
    specs = tbl.specs()
    data_files = {}
    for df in live_entries(tbl):
        if df.content != DataFileContent.DATA:
            continue
        spec = specs[df.spec_id]
        rows = file_rows(df.file_path)
        recorded = tuple(canon(df.partition[i]) for i in range(len(spec.fields)))
        want = expected_tuple(tbl, spec, rows, wrong)
        data_files[df.file_path] = (df.spec_id, recorded)
        if recorded != want:
            fails.append("%s: %s spec %d records %s, pyiceberg transforms its rows to %s"
                         % (name, os.path.basename(df.file_path), df.spec_id, recorded, want))
    return fails, data_files


def check_delete_scoping(name, meta, data_files):
    fails = []
    tbl = StaticTable.from_metadata("file://" + meta)
    for df in live_entries(tbl):
        if df.content != DataFileContent.POSITION_DELETES:
            continue
        spec = tbl.specs()[df.spec_id]
        mine = (df.spec_id, tuple(canon(df.partition[i]) for i in range(len(spec.fields))))
        targets = set(pq.read_table(df.file_path.replace("file://", "")).column("file_path").to_pylist())
        for t in targets:
            if t in data_files and data_files[t] != mine:
                fails.append("%s: delete file %s is %s but names %s of %s"
                             % (name, os.path.basename(df.file_path), mine, os.path.basename(t), data_files[t]))
    return fails


# ---- pruning ----

def literal_for(name, v, t):
    n = str(t)
    if n.startswith("decimal"):
        return decimal.Decimal(v).scaleb(-t.scale)
    if n == "uuid":
        return uuid.UUID(bytes=v)
    return v


OPS = [("=", EqualTo, lambda a, b: a == b), ("!=", NotEqualTo, lambda a, b: a != b),
       ("<", LessThan, lambda a, b: a < b), ("<=", LessThanOrEqual, lambda a, b: a <= b),
       (">", GreaterThan, lambda a, b: a > b), (">=", GreaterThanOrEqual, lambda a, b: a >= b)]


def predicates(tbl, colname):
    """(label, pyiceberg expression, row predicate) for every operator over literals drawn from the rows."""
    t = tbl.schema().find_field(colname).field_type
    c = COLS.index(colname)
    vals = sorted({wide_value(i, c) for i in (0, 7, 13, 29, 30, 44, 59) if wide_value(i, c) is not None})
    out = [("is null", IsNull(colname), lambda r: col(r, colname) is None),
           ("not null", NotNull(colname), lambda r: col(r, colname) is not None)]
    identity = any(isinstance(f.transform, IdentityTransform) and f.source_id == tbl.schema().find_field(colname).field_id
                   for f in tbl.spec().fields)
    for v in vals:
        for label, ctor, fn in OPS:
            if str(t) == "uuid" and label not in ("=", "!="):
                continue                                                    # uuid order differs between engines
            if str(t) == "uuid" and identity:
                continue                                                    # pyiceberg 0.12 defect, see the header
            out.append(("%s %s %r" % (colname, label, v), ctor(colname, literal_for(colname, v, t)),
                        (lambda fn, v: lambda r: col(r, colname) is not None and fn(col(r, colname), v))(fn, v)))
    return out


def check_pruning(name, meta, counters, model_tuples=True):
    """plan_files under every predicate on a partition source: never skips a file holding a matching row, and (with
    model_tuples) skips every file whose expected tuple -- pyiceberg's transform of the file's rows under the file's
    own spec -- pyiceberg's inclusive projection rejects."""
    fails = []
    tbl = StaticTable.from_metadata("file://" + meta)
    specs = tbl.specs()
    sources = sorted({tbl.schema().find_field(f.source_id).name for sp in specs.values() for f in sp.fields
                      if not isinstance(f.transform, VoidTransform)})
    files = [(df.file_path, df.spec_id, file_rows(df.file_path)) for df in live_entries(tbl)
             if df.content == DataFileContent.DATA]
    part_schema = {sid: Schema(*sp.partition_type(tbl.schema()).fields) for sid, sp in specs.items()}
    tuples = {path: expected_tuple(tbl, specs[sid], rows) for path, sid, rows in files} if model_tuples else {}
    for colname in sources:
        for label, expr, row_pred in predicates(tbl, colname):
            planned = {task.file.file_path for task in tbl.scan(row_filter=expr).plan_files()}
            evaluate = {sid: expression_evaluator(part_schema[sid], inclusive_projection(tbl.schema(), sp)(expr), True)
                        for sid, sp in specs.items()}
            for path, sid, rows in files:
                counters["files"] += 1
                if any(row_pred(r) for r in rows) and path not in planned:
                    fails.append("%s: %s skipped %s, which holds a matching row" % (name, label, os.path.basename(path)))
                tup = tuples.get(path)
                if tup is None or (tup and isinstance(tup[0], str)):
                    continue
                rec = Record(*[to_pyiceberg(x, f.field_type) for x, f in zip(tup, part_schema[sid].fields)])
                if not evaluate[sid](rec):
                    counters["excluded"] += 1
                    if path in planned:
                        fails.append("%s: %s planned %s, whose tuple %s the projection excludes"
                                     % (name, label, os.path.basename(path), tup))
    return fails


def sql_literal(v, t):
    n = str(t)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, bytes):
        return "'" + "".join("\\x%02X" % b for b in v) + "'::BLOB"
    if n == "date":
        return "(DATE '1970-01-01' + %d)" % v
    if n == "timestamp":
        return "make_timestamp(%d)" % v
    if n == "timestamptz":
        return "to_timestamp(%d / 1000000.0)" % v
    if n.startswith("decimal"):
        return "CAST(%s AS DECIMAL(9,2))" % str(decimal.Decimal(v).scaleb(-2))
    if n == "time":
        return "TIME '%s'" % (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=v)).time().isoformat()
    return str(v)


def check_duckdb_filters(name, meta, want):
    fails = []
    tbl = StaticTable.from_metadata("file://" + meta)
    spec = tbl.spec()
    for f in spec.fields:
        src = tbl.schema().find_field(f.source_id)
        c = COLS.index(src.name)
        if str(src.field_type) in ("uuid", "binary", "fixed[4]", "timestamptz"):
            continue                                                        # literal spelling, not pruning, at stake
        strict_broken = isinstance(f.transform, IdentityTransform) and (
            str(src.field_type) in ("date", "timestamp", "time") or str(src.field_type).startswith("decimal"))
        for i in (7, 30):
            v = wide_value(i, c)
            if v is None:
                continue
            for op, _, fn in OPS:
                if strict_broken and op in ("<", ">"):
                    continue                                                # DuckDB 1.4 defect, see the header
                expect = [r for r in want if col(r, src.name) is not None and fn(col(r, src.name), v)]
                try:
                    got = duckdb_rows(meta, "WHERE %s %s %s" % (src.name, op, sql_literal(v, src.field_type)))
                except Exception as e:  # noqa: BLE001
                    fails.append("%s: duckdb WHERE %s %s failed: %s" % (name, src.name, op, e))
                    continue
                if got != expect:
                    fails.append("%s: duckdb WHERE %s %s %r returned %d rows, model %d"
                                 % (name, src.name, op, v, len(got), len(expect)))
    return fails


def tampered_spec(meta, scratch, old, new):
    with open(meta) as f:
        doc = json.load(f)
    doc = copy.deepcopy(doc)
    for s in doc["partition-specs"]:
        for fld in s["fields"]:
            if fld["transform"] == old:
                fld["transform"] = new
    out = os.path.join(scratch, "tampered-%d.metadata.json" % len(os.listdir(scratch)))
    with open(out, "w") as f:
        json.dump(doc, f)
    return out


def run_fixture(exe, out_dir):
    env = dict(os.environ, BOLT_ICEBERG_PART_DIR=out_dir)
    r = subprocess.run([exe, "--gtest_filter=IcebergPartitionFixture.*"], env=env, capture_output=True, text=True)
    if r.returncode != 0 or "[  PASSED  ] 1 test" not in r.stdout:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit("fixture writer failed")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    exe, out_dir = sys.argv[1], os.path.abspath(sys.argv[2])
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    run_fixture(exe, out_dir)
    with open(os.path.join(out_dir, "tables.txt")) as f:
        tables = [line.split() for line in f if line.strip()]
    fails, controls = [], []
    counters = {"files": 0, "excluded": 0}
    scratch = tempfile.mkdtemp(prefix="iceberg_part_neg_")
    for name, kind in tables:
        meta = latest_metadata(os.path.join(out_dir, name))
        want = model_rows(kind)
        fails += check_rows(name, meta, want)
        tf, data_files = check_transforms(name, meta)
        fails += tf
        fails += check_delete_scoping(name, meta, data_files)
        controls.append(("%s tuples vs a wrong transform" % name, check_transforms(name, meta, wrong=True)[0]))
        if kind in ("delete", "overwrite", "compact"):
            controls.append(("%s rows vs predicate-ignored model" % name, check_rows(name, meta, wrong_rows(kind))))
            continue
        fails += check_pruning(name, meta, counters)
        fails += check_duckdb_filters(name, meta, want)
        print("%s: %d rows, %d data files checked" % (name, len(want), len(data_files)))
    for name, old, new in (("bucket_4__id", "bucket[4]", "bucket[5]"), ("truncate_10__id", "truncate[10]", "truncate[11]")):
        meta = tampered_spec(latest_metadata(os.path.join(out_dir, name)), scratch, old, new)
        controls.append(("%s planned under a spec claiming %s" % (name, new),
                         [f for f in check_pruning(name, meta, {"files": 0, "excluded": 0}, model_tuples=False)
                          if "holds a matching row" in f]))
    if counters["excluded"] == 0:
        fails.append("no file was ever excluded by a projection: the pruning check is vacuous")
    for what, found in controls:
        if not found:
            fails.append("negative control did not fail: %s" % what)
    print("negative controls red as they must be: %d of %d" % (sum(1 for _, f in controls if f), len(controls)))
    print("pruning: %d file x predicate checks, %d files excluded by pyiceberg's projection"
          % (counters["files"], counters["excluded"]))
    for f in fails[:60]:
        print("FAIL:", f)
    if fails:
        print("%d failures" % len(fails))
        return 1
    print("PASS: pyiceberg and DuckDB agree with the model; every tuple is pyiceberg's transform of its rows; "
          "plan_files never skips a match and skips what the projection excludes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
