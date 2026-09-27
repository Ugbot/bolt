#!/usr/bin/env python3
"""LIMITS L5 scale fixture: a pyiceberg-written Iceberg table with TOTAL one-row
data files over N snapshots (12 identity partition columns): the first append
writes TOTAL - (N - 1) files into one manifest, each later append one file.
The schema is then evolved to 300 columns. Writes <out>/oracle.json with pyiceberg's and DuckDB's
row counts / sums for the current snapshot and several time-travel snapshots,
for tests/test_bolt_lakehouse_scale.cpp to compare bolt's reader against.

usage: l5_iceberg_scale.py <out_dir> [snapshots=5000] [total_files=100000]
"""
import glob, json, os, shutil, sys, time

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import IntegerType, LongType, NestedField

# pyiceberg deep-copies the whole TableMetadata (every snapshot) on each
# metadata update, which makes 5,000 appends quadratic (hours). Its updates
# build new lists rather than mutating, so a shallow copy of the frozen
# metadata models is equivalent; the oracle below re-loads the table from disk
# and cross-checks every answer against DuckDB.
from pyiceberg.table.metadata import TableMetadataV2
_orig_copy = TableMetadataV2.model_copy


def _shallow_copy(self, *, update=None, deep=False):
    return _orig_copy(self, update=update, deep=False)


TableMetadataV2.model_copy = _shallow_copy

N_PART = 12
N_DATA = 6
WIDE = 300


def schema():
    f = [NestedField(1, "id", LongType(), required=False),
         NestedField(2, "v", LongType(), required=False)]
    for j in range(N_PART):
        f.append(NestedField(3 + j, f"p{j}", IntegerType(), required=False))
    for j in range(N_DATA):
        f.append(NestedField(3 + N_PART + j, f"d{j}", LongType(), required=False))
    return Schema(*f)


def spec():
    return PartitionSpec(*[PartitionField(source_id=3 + j, field_id=1000 + j,
                                          transform=IdentityTransform(),
                                          name=f"p{j}") for j in range(N_PART)])


def batch(sch, first, files):
    ids = [first + i for i in range(files)]
    cols = {"id": pa.array(ids, pa.int64()),
            "v": pa.array([i * 3 + 1 for i in ids], pa.int64())}
    for j in range(N_PART):
        # distinct tuple per row -> one data file per row
        cols[f"p{j}"] = pa.array([i if j == 0 else (i * 7 + j) % 97
                                  for i in ids], pa.int32())
    for j in range(N_DATA):
        cols[f"d{j}"] = pa.array([i + j for i in ids], pa.int64())
    return pa.Table.from_pydict(cols, schema=sch.as_arrow())


def main():
    out = os.path.abspath(sys.argv[1])
    n_snap = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    first = total - (n_snap - 1)
    assert first >= 1
    counts = [first] + [1] * (n_snap - 1)
    per_tx = int(os.environ.get("L5_PER_TX", "100"))
    budget_s = float(os.environ.get("L5_TIME_BUDGET_S", "0"))  # 0 = no limit
    wh = os.path.join(out, "wh")
    resume = os.path.exists(os.path.join(out, "cat.db"))
    if not resume:
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(wh)
    cat = SqlCatalog("l5", uri=f"sqlite:///{out}/cat.db", warehouse=f"file://{wh}")
    sch = schema()
    if resume:
        tbl = cat.load_table("default.t")
    else:
        cat.create_namespace("default")
        tbl = cat.create_table(
            "default.t", schema=sch, partition_spec=spec(),
            properties={"write.parquet.compression-codec": "snappy",
                        # keeps each manifest list short; 5,000 fast
                        # appends would be O(n^2) list entries
                        "commit.manifest-merge.enabled": "true",
                        "commit.manifest.min-count-to-merge": "50"})
    t0 = time.time()
    snap = len(tbl.snapshots())
    row = sum(counts[:snap])
    while snap < n_snap:
        if budget_s and time.time() - t0 > budget_s:
            print(f"PARTIAL {snap}/{n_snap}; rerun to resume", flush=True)
            return
        with tbl.transaction() as tx:
            for _ in range(min(per_tx, n_snap - snap)):
                tx.append(batch(sch, row, counts[snap]))
                row += counts[snap]
                snap += 1
        print(f"{snap}/{n_snap} snapshots {time.time() - t0:.0f}s", flush=True)
    if len(tbl.schema().fields) == WIDE and os.path.exists(
            os.path.join(out, "oracle.json")):
        print("already complete")
        return
    # Widen the schema to 300 columns (no new snapshot, no rewrite).
    if len(tbl.schema().fields) < WIDE:
        with tbl.update_schema() as us:
            for c in range(len(tbl.schema().fields), WIDE):
                us.add_column(f"w{c}", LongType())
    tbl = cat.load_table("default.t")
    snaps = tbl.snapshots()
    assert len(snaps) == n_snap, len(snaps)
    assert len(tbl.schema().fields) == WIDE
    picks = sorted({0, 9, n_snap // 2, n_snap - 1})
    oracle = {"metadata": tbl.metadata_location.replace("file://", ""),
              "location": tbl.location().replace("file://", ""),
              "snapshots": len(snaps), "schema_fields": len(tbl.schema().fields),
              "total_files": total, "checks": []}
    import duckdb
    con = duckdb.connect()
    con.execute("LOAD iceberg")
    # Each check reads up to TOTAL files twice; cache them one at a time so a
    # time-boxed run resumes where it stopped.
    part = os.path.join(out, "checks_partial.json")
    done = json.load(open(part)) if os.path.exists(part) else {}
    for k in picks:
        if str(k) in done:
            continue
        if budget_s and time.time() - t0 > budget_s:
            print(f"PARTIAL oracle {len(done)}/{len(picks)}; rerun to resume",
                  flush=True)
            return
        sid = snaps[k].snapshot_id
        at = tbl.scan(snapshot_id=sid, selected_fields=("id", "v")).to_arrow()
        rows_py = at.num_rows
        sum_py = int(pa.compute.sum(at["v"]).as_py() or 0)
        r = con.execute("SELECT count(*), sum(v) FROM iceberg_scan(?, "
                        "snapshot_from_id => ?)",
                        [oracle["metadata"], sid]).fetchone()
        assert rows_py == r[0] == sum(counts[:k + 1]), (k, rows_py, r)
        assert sum_py == int(r[1] or 0), (k, sum_py, r)
        done[str(k)] = {"index": k, "snapshot_id": sid, "rows": rows_py,
                        "sum_v": sum_py, "duckdb_rows": int(r[0]),
                        "duckdb_sum_v": int(r[1] or 0)}
        json.dump(done, open(part, "w"))
        print(f"oracle check {k}: {rows_py} rows {time.time() - t0:.0f}s",
              flush=True)
    oracle["checks"] = [done[str(k)] for k in picks]
    n_files = len(glob.glob(os.path.join(oracle["location"], "data", "**",
                                         "*.parquet"), recursive=True))
    oracle["data_files"] = n_files
    with open(os.path.join(out, "oracle.json"), "w") as f:
        json.dump(oracle, f, indent=1)
    # Line form for the C++ harness: header, then "<snapshot_id> <rows> <sum_v>".
    with open(os.path.join(out, "oracle.txt"), "w") as f:
        f.write(f"{oracle['metadata']}\n{oracle['location']}\n"
                f"{oracle['snapshots']} {oracle['schema_fields']} "
                f"{oracle['data_files']}\n")
        for c in oracle["checks"]:
            f.write(f"{c['snapshot_id']} {c['rows']} {c['sum_v']}\n")
    print(json.dumps({k: oracle[k] for k in ("snapshots", "schema_fields",
                                             "data_files")}))


if __name__ == "__main__":
    main()
