#!/usr/bin/env python3
"""G2ICE-283: do Iceberg readers that are not bolt agree with the delete files bolt writes?

    python scripts/iceberg_delete_writers_interop.py <build>/tests/test_bolt_iceberg_delete_writers <scratch dir>

IcebergDeleteWritersFixture.WriteTables writes equality deletes (by predicate with the identifier field and a string key,
partition-scoped, global under an added unpartitioned spec, by key unpartitioned / bucket-scoped / global, an upsert) and
position deletes carrying the deleted `row`. For each table this checks, against a model computed here:
  * the rows DuckDB returns (iceberg_scan), and pyiceberg's for position-delete tables; pyiceberg must refuse an
    equality-delete table by name (it reads none), which also shows it saw them as equality deletes;
  * every snapshot's summary counters against a recount of its manifests (pyiceberg's manifest reader);
  * each equality delete file's equality_ids, spec and partition against what the delete means;
  * every position delete's `row` column (pyarrow) equals the row at (file_path, pos) in its data file, read back from
    that data file, and has field id 2147483544.
Negative controls: the rows against the deletes-ignored model, and the metadata rolled back one snapshot, must be red.
Exit 0 = pass; 1 = a check failed or a control stayed green; 77 = an oracle is not installed (SKIP).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import pyarrow.parquet as pq
    from pyiceberg.manifest import DataFileContent, ManifestEntryStatus
    from pyiceberg.table import StaticTable
    import duckdb
    import iceberg_dml_interop as dml
except (ImportError, SystemExit) as e:  # a missing oracle is a SKIP, never a PASS
    print("SKIP: %s" % e)
    sys.exit(77)

row_s, base_rows = dml.row_s, dml.base_rows


def ordered(rows):
    return sorted(rows, key=lambda r: (r[0], -2**63 if r[1] is None else r[1], r[2]))


def keep(rows, gone):
    return ordered(r for r in rows if not gone(r))
POS_GONE = lambda r: r[0] in (15, 22)  # noqa: E731


def upsert_rows():
    return [(i, 7, "new-" + row_s(i)) for i in range(5)]


# want: the rows; wrong: the rows had the newest delete been ignored; ops; eq: (equality ids, partition tuple width) of
# every equality delete file; pos_rows: position deletes carry `row`.
TABLES = {
    "eq_pred_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[1] == 1 or r[0] == 27),
        wrong=keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[1] == 1),
        ops=["append"] * 3 + ["delete"] * 4, eq=sorted([([1], 0), ([3], 0)])),
    "eq_scoped_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] < 12), wrong=base_rows(0, 30),
        ops=["append"] * 3 + ["delete"], eq=[([1], 1)] * 3),
    "eq_global_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] >= 25), wrong=base_rows(0, 30),
        ops=["append"] * 3 + ["delete"], eq=[([1], 0)], global_spec=True),
    "keys_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[0] in (1, 2, 28)),
        wrong=keep(base_rows(0, 30), POS_GONE),
        ops=["append"] * 3 + ["delete"] * 3, eq=[([1], 0)]),
    "keys_bucketed_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] in (4, 17, 26)), wrong=base_rows(0, 30),
        ops=["append"] * 3 + ["delete"], eq=None),
    "keys_global_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] in (5, 13, 27)), wrong=base_rows(0, 30),
        ops=["append"] * 3 + ["delete"], eq=[([1], 0)], global_spec=True),
    "upsert_tbl": dict(
        want=ordered(keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[0] < 5) + upsert_rows()),
        wrong=ordered(keep(base_rows(0, 30), POS_GONE) + upsert_rows()),
        ops=["append"] * 3 + ["delete"] * 2 + ["overwrite"], eq=[([1], 0)]),
    "pos_row_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[1] == 1 or r[0] <= 2),
        wrong=keep(base_rows(0, 30), lambda r: POS_GONE(r) or r[1] == 1),
        ops=["append"] * 3 + ["delete"] * 4, eq=[], pos_rows=True),
    # format v3: deletes are deletion vectors merged per file (15, 22, then 27..29 into C's), A removed whole, an equality
    # delete of id 18, then rows 100 and 101 appended (row ids 30 and 31).
    "v3_tbl": dict(
        want=ordered(keep(base_rows(0, 30), lambda r: r[0] in (15, 22, 18) or r[0] >= 27 or r[0] < 10) +
                     [(100, 7, row_s(100)), (101, 7, row_s(101))]),
        wrong=ordered(keep(base_rows(0, 30), lambda r: r[0] in (15, 22, 18) or r[0] >= 27 or r[0] < 10)),
        ops=["append"] * 3 + ["delete"] * 6 + ["append"], eq=[([1], 0)], v3=True),
    "pos_row_part_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] >= 23), wrong=base_rows(0, 30),
        ops=["append"] * 3 + ["delete"], eq=[], pos_rows=True),
}


def has_eq(name):
    return TABLES[name]["eq"] is None or len(TABLES[name]["eq"]) > 0


def duck_rows(meta):
    return dml.duckdb_rows(meta)


def check_rows(name, meta, want):
    fails = []
    readers = [("duckdb", duck_rows)] + ([] if has_eq(name) else [("pyiceberg", dml.pyiceberg_rows)])
    for reader, fn in readers:
        try:
            got = fn(meta)
        except Exception as e:  # noqa: BLE001 - any reader error is a failed check
            fails.append("%s: %s read failed: %s" % (name, reader, e))
            continue
        if got != want:
            fails.append("%s: %s rows differ (%d vs %d; missing %s, extra %s)"
                         % (name, reader, len(got), len(want), sorted(set(want) - set(got))[:5],
                            sorted(set(got) - set(want))[:5]))
    return fails


def pyiceberg_refuses_eq(name, meta):
    try:
        dml.pyiceberg_rows(meta)
    except (NotImplementedError, ValueError) as e:
        return [] if "support equality deletes" in str(e) else ["%s: pyiceberg refused with %s" % (name, e)]
    except Exception as e:  # noqa: BLE001
        return ["%s: pyiceberg failed with %r, not its equality-delete refusal" % (name, e)]
    return ["%s: pyiceberg read an equality-delete table without refusing" % name]


def delete_entries(meta):
    tbl = StaticTable.from_metadata("file://" + meta)
    snap = tbl.current_snapshot()
    out = []
    for m in snap.manifests(tbl.io):
        for e in m.fetch_manifest_entry(tbl.io, discard_deleted=True):
            if e.data_file.content != DataFileContent.DATA:
                out.append((e.data_file, e.status))
    return tbl, out


def check_eq_files(name, meta, t):
    if t["eq"] is None:
        return []
    tbl, ents = delete_entries(meta)
    specs = {s.spec_id: s for s in tbl.metadata.partition_specs}
    got, fails = [], []
    for df, _ in ents:
        if df.content != DataFileContent.EQUALITY_DELETES:
            continue
        got.append((list(df.equality_ids), len(df.partition)))
        spec = specs.get(df.spec_id)
        if spec is None:
            fails.append("%s: delete file spec %d is not a spec of the table" % (name, df.spec_id))
        elif t.get("global_spec") and (len(spec.fields) != 0 or df.spec_id == tbl.metadata.default_spec_id):
            fails.append("%s: a global delete is under spec %d, not an added unpartitioned spec" % (name, df.spec_id))
        cols = pq.read_schema(df.file_path.replace("file://", ""))
        ids = [int(f.metadata[b"PARQUET:field_id"]) for f in cols]
        if ids != list(df.equality_ids):
            fails.append("%s: equality delete columns carry field ids %s, equality_ids %s" % (name, ids, df.equality_ids))
    if sorted(got) != sorted(t["eq"]):
        fails.append("%s: equality delete files %s, want %s" % (name, sorted(got), sorted(t["eq"])))
    return fails


def check_pos_rows(name, meta, t):
    if not t.get("pos_rows"):
        return []
    _, ents = delete_entries(meta)
    fails, seen = [], 0
    cache = {}
    for df, _ in ents:
        if df.content != DataFileContent.POSITION_DELETES:
            continue
        path = df.file_path.replace("file://", "")
        sch = pq.read_schema(path)
        if "row" not in sch.names:
            continue                                  # the base's position deletes (ids 15 and 22) carry no row
        rid = sch.field("row").metadata or {}
        if rid.get(b"PARQUET:field_id") != b"2147483544":
            fails.append("%s: `row` field id %s, want 2147483544" % (name, rid))
        tab = pq.read_table(path).to_pylist()
        keys = [(r["file_path"], r["pos"]) for r in tab]
        if keys != sorted(keys):
            fails.append("%s: position deletes not sorted by (file_path, pos)" % name)
        for r in tab:
            fp = r["file_path"].replace("file://", "")
            if fp not in cache:
                cache[fp] = pq.read_table(fp).to_pylist()
            src = cache[fp][r["pos"]]
            if r["row"] != src:
                fails.append("%s: row at %s#%d is %s, the data file holds %s" % (name, fp, r["pos"], r["row"], src))
            seen += 1
    if seen == 0:
        fails.append("%s: no position delete carries a `row`" % name)
    return fails


def check_counters(name, meta, ops):
    return dml.check_snapshots(name, meta, ops)


def run_fixture(exe, out_dir):
    env = dict(os.environ, BOLT_ICEBERG_DELETE_DIR=out_dir)
    r = subprocess.run([exe, "--gtest_filter=IcebergDeleteWritersFixture.*"], env=env, capture_output=True, text=True)
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
    fails, controls = [], []
    scratch = tempfile.mkdtemp(prefix="iceberg_delete_neg_")
    for name, t in TABLES.items():
        meta = dml.latest_metadata(os.path.join(out_dir, name))
        fails += check_rows(name, meta, t["want"])
        if has_eq(name):
            fails += pyiceberg_refuses_eq(name, meta)
        fails += check_counters(name, meta, t["ops"])
        fails += check_eq_files(name, meta, t)
        fails += check_pos_rows(name, meta, t)
        controls.append(("%s rows vs the deletes-ignored model" % name, check_rows(name, meta, t["wrong"])))
        controls.append(("%s rolled back one snapshot" % name,
                         check_rows(name, dml.tampered(meta, scratch, dml.rolled_back(1)), t["want"])))
        print("%s: %d rows, %d snapshots checked" % (name, len(t["want"]), len(t["ops"])))
    for what, found in controls:
        if not found:
            fails.append("negative control did not fail: %s" % what)
        else:
            print("negative control red as it must be: %s (e.g. %s)" % (what, found[0][:160]))
    for f in fails:
        print("FAIL:", f)
    if fails:
        return 1
    print("PASS: DuckDB (and pyiceberg where it reads) agree with the model; equality ids, specs, `row` values and "
          "summary counters match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
