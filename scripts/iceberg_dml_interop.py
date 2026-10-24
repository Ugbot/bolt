#!/usr/bin/env python3
"""G2ICE-280: do Iceberg readers that are not bolt agree with bolt's table_delete(pred), table_overwrite(pred, batch)
and table_compact?

    uv run --no-project --python 3.12 --with 'pyiceberg[pyiceberg-core]' --with pyarrow --with 'duckdb==1.4.*' \\
        python scripts/iceberg_dml_interop.py <build>/tests/test_bolt_iceberg_dml <scratch dir>

The executable's IcebergDmlFixture.WriteTables writes three tables (delete_tbl, overwrite_tbl, compact_tbl). For each
one this script checks, against a model it computes itself from the same row formula:
  * the rows pyiceberg (StaticTable on the metadata file) and DuckDB (iceberg_scan) return;
  * the snapshot operations, in order, as pyiceberg's inspect.snapshots() reports them;
  * every summary counter of every snapshot against counts recomputed here from that snapshot's manifests.
Negative controls prove the checks can fail: the rows are compared against a model with the predicate ignored, a
copy of the metadata whose current snapshot is the parent of the DML snapshot (the operation "did nothing") is read,
and a copy with one summary counter off by one is checked. Each must be reported as a mismatch.

The two readers are not equally sharp. With the manifest writer made to relabel a carried entry's sequence number
(shared_tbl), pyiceberg returns the position-deleted id 12 again and DuckDB 1.4.5 does not: DuckDB applies position
deletes without comparing sequence numbers. So shared_tbl's verdict rests on pyiceberg.

Exit 0 = pass; 1 = a check failed or a negative control did not fail; 77 = pyiceberg or duckdb not installed (SKIP).
"""
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

try:
    from pyiceberg.manifest import DataFileContent, ManifestEntryStatus
    from pyiceberg.table import StaticTable
    import duckdb
except ImportError as e:  # a missing oracle is a SKIP, never a PASS
    print("SKIP: %s" % e)
    sys.exit(77)

NULL = None


def row_s(i):
    return "row-%d%s" % (i, "-with-a-long-spilled-tail" if i % 3 == 0 else "")


def base_k(i):
    if i == 3:
        return NULL
    if i < 10:
        return 0
    if i < 20:
        return 1
    return 1 if i < 25 else 2


def base_rows(lo, hi):
    return [(i, base_k(i), row_s(i)) for i in range(lo, hi)]


def added_rows():
    return [(i, 7, row_s(i)) for i in range(100, 105)]


def keep(rows, gone):
    return sorted(r for r in rows if not gone(r))


# What each table must hold, what it would hold had the predicate been ignored, and its snapshot operations.
TABLES = {
    "delete_tbl": dict(
        want=keep(base_rows(0, 30), lambda r: r[0] in (15, 22) or r[1] == 1 or r[2] == row_s(27)),
        wrong=keep(base_rows(0, 30), lambda r: r[0] in (15, 22)),
        ops=["append"] * 3 + ["delete"] * 4),   # 2 position deletes, k = 1, s = row_s(27); k = 99 committed nothing
    "overwrite_tbl": dict(
        want=keep(base_rows(0, 30) + added_rows(), lambda r: r[0] in (15, 22) or r[1] == 1),
        wrong=keep(base_rows(0, 30) + added_rows(), lambda r: r[0] in (15, 22)),
        ops=["append"] * 3 + ["delete"] * 2 + ["overwrite"]),
    "compact_tbl": dict(
        want=keep(base_rows(0, 20), lambda r: r[0] in (1, 14, 17)),
        wrong=keep(base_rows(0, 20), lambda r: False),   # deletes not applied by the rewrite
        ops=["append"] * 4 + ["delete"] * 3 + ["replace"],
        # A replace must not change rows, so its parent reads the same; the negative control steps back past the
        # three deletes the rewrite applied instead.
        same_as_parent=True, rollback=4),
    # Two compaction outputs in one manifest; id 12 position-deleted; the first output dropped, which rewrites the
    # shared manifest with the second as EXISTING (its sequence number must survive or id 12 comes back).
    "shared_tbl": dict(
        want=keep(base_rows(0, 20), lambda r: r[0] <= 9 or r[0] == 12),
        wrong=keep(base_rows(0, 20), lambda r: r[0] <= 9),
        ops=["append"] * 4 + ["replace", "delete", "delete"]),
}

COUNTERS = ["added-data-files", "deleted-data-files", "added-delete-files", "removed-delete-files",
            "added-position-delete-files", "removed-position-delete-files", "added-records", "deleted-records",
            "added-position-deletes", "removed-position-deletes", "added-files-size", "removed-files-size",
            "total-records", "total-data-files", "total-delete-files", "total-position-deletes",
            "total-equality-deletes", "total-files-size"]


def latest_metadata(table_dir):
    paths = glob.glob(os.path.join(table_dir, "metadata", "v*.metadata.json"))
    return max(paths, key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))


def norm(rows):
    return sorted((int(i), None if k is None else int(k), s) for i, k, s in rows)


def pyiceberg_rows(meta):
    t = StaticTable.from_metadata("file://" + meta).scan().to_arrow()
    return norm(zip(t.column("id").to_pylist(), t.column("k").to_pylist(), t.column("s").to_pylist()))


_con = None


def duckdb_rows(meta):
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.execute("INSTALL iceberg; LOAD iceberg;")
    return norm(_con.execute("SELECT id, k, s FROM iceberg_scan('%s')" % meta).fetchall())


def check_rows(name, meta, want):
    fails = []
    for reader, fn in (("pyiceberg", pyiceberg_rows), ("duckdb", duckdb_rows)):
        try:
            got = fn(meta)
        except Exception as e:  # noqa: BLE001 - any reader error is a failed check
            fails.append("%s: %s read failed: %s" % (name, reader, e))
            continue
        if got != want:
            missing = sorted(set(want) - set(got))[:5]
            extra = sorted(set(got) - set(want))[:5]
            fails.append("%s: %s rows differ (%d vs %d; missing %s, extra %s)"
                         % (name, reader, len(got), len(want), missing, extra))
    return fails


def recount(tbl, snap):
    """Summary counters of `snap` recomputed from its manifests."""
    c = dict.fromkeys(COUNTERS, 0)
    for m in snap.manifests(tbl.io):
        for e in m.fetch_manifest_entry(tbl.io, discard_deleted=False):
            df = e.data_file
            data = df.content == DataFileContent.DATA
            pos = df.content == DataFileContent.POSITION_DELETES
            mine = e.snapshot_id == snap.snapshot_id
            if e.status == ManifestEntryStatus.ADDED and mine:
                c["added-data-files" if data else "added-delete-files"] += 1
                c["added-records" if data else "added-position-deletes"] += df.record_count if (data or pos) else 0
                c["added-position-delete-files"] += 1 if pos else 0
                c["added-files-size"] += df.file_size_in_bytes
            if e.status == ManifestEntryStatus.DELETED and mine:
                c["deleted-data-files" if data else "removed-delete-files"] += 1
                c["deleted-records" if data else "removed-position-deletes"] += df.record_count if (data or pos) else 0
                c["removed-position-delete-files"] += 1 if pos else 0
                c["removed-files-size"] += df.file_size_in_bytes
            if e.status == ManifestEntryStatus.DELETED:
                continue
            c["total-files-size"] += df.file_size_in_bytes
            if data:
                c["total-data-files"] += 1
                c["total-records"] += df.record_count
            else:
                c["total-delete-files"] += 1
                c["total-position-deletes" if pos else "total-equality-deletes"] += df.record_count
    return c


def check_snapshots(name, meta, ops):
    fails = []
    tbl = StaticTable.from_metadata("file://" + meta)
    rows = tbl.inspect.snapshots().to_pylist()
    got_ops = [r["operation"] for r in sorted(rows, key=lambda r: r["committed_at"])]
    if got_ops != ops:
        fails.append("%s: operations %s, want %s" % (name, got_ops, ops))
    for r in rows:
        summary = dict(r["summary"] or [])
        snap = tbl.snapshot_by_id(r["snapshot_id"])
        for k, v in recount(tbl, snap).items():
            have = summary.get(k)
            if have is None and (k.startswith("total-") or v != 0):
                fails.append("%s: snapshot %d %s missing (manifests: %d)" % (name, r["snapshot_id"], k, v))
            elif have is not None and int(have) != v:
                fails.append("%s: snapshot %d %s = %s, manifests say %d" % (name, r["snapshot_id"], k, have, v))
    return fails


def tampered(meta, scratch, edit):
    with open(meta) as f:
        doc = json.load(f)
    doc = edit(copy.deepcopy(doc))
    out = os.path.join(scratch, "tampered-%d.metadata.json" % len(os.listdir(scratch)))
    with open(out, "w") as f:
        json.dump(doc, f)
    return out


def rolled_back(steps):
    def edit(doc):
        snaps = {s["snapshot-id"]: s for s in doc["snapshots"]}
        cur = doc["current-snapshot-id"]
        for _ in range(steps):
            cur = snaps[cur]["parent-snapshot-id"]
        doc["current-snapshot-id"] = cur
        doc["refs"]["main"]["snapshot-id"] = cur
        return doc
    return edit


def counter_off_by_one(doc):
    cur = [s for s in doc["snapshots"] if s["snapshot-id"] == doc["current-snapshot-id"]][0]
    cur["summary"]["total-records"] = str(int(cur["summary"]["total-records"]) + 1)
    return doc


def run_fixture(exe, out_dir):
    env = dict(os.environ, BOLT_ICEBERG_DML_DIR=out_dir)
    r = subprocess.run([exe, "--gtest_filter=IcebergDmlFixture.*"], env=env, capture_output=True, text=True)
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
    fails = []
    controls = []
    scratch = tempfile.mkdtemp(prefix="iceberg_dml_neg_")
    for name, t in TABLES.items():
        meta = latest_metadata(os.path.join(out_dir, name))
        fails += check_rows(name, meta, t["want"])
        fails += check_snapshots(name, meta, t["ops"])
        # Negative controls: each must produce at least one failure.
        controls.append(("%s rows vs predicate-ignored model" % name, check_rows(name, meta, t["wrong"])))
        if t.get("same_as_parent"):
            fails += check_rows(name + " (parent of the replace)", tampered(meta, scratch, rolled_back(1)), t["want"])
        steps = t.get("rollback", 1)
        controls.append(("%s rolled back %d snapshot(s)" % (name, steps),
                         check_rows(name, tampered(meta, scratch, rolled_back(steps)), t["want"])))
        controls.append(("%s summary counter off by one" % name,
                         check_snapshots(name, tampered(meta, scratch, counter_off_by_one), t["ops"])))
        print("%s: %d rows, %d snapshots checked" % (name, len(t["want"]), len(t["ops"])))
    for what, found in controls:
        # A row control must be red in BOTH readers, so neither oracle is blind to the defect.
        readers = [r for r in ("pyiceberg", "duckdb") if "rows" not in what and "rolled" not in what
                   or any((": %s " % r) in f for f in found)]
        if not found or len(readers) != 2:
            fails.append("negative control did not fail (in every reader): %s -> %s" % (what, found))
        else:
            print("negative control red as it must be: %s (%d findings, e.g. %s)" % (what, len(found), found[0][:160]))
    for f in fails:
        print("FAIL:", f)
    if fails:
        return 1
    print("PASS: pyiceberg and DuckDB agree with the model on every table; every summary counter matches the manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
