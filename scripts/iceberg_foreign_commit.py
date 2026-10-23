#!/usr/bin/env python3
"""G2ICE-279: pyiceberg makes a table (tag, branch, two-field sort order, custom properties, a statistics entry), bolt
appends to it, pyiceberg reads it back. Every one of those must be unchanged and the new rows present. A v1 table must
be refused by bolt, never rewritten as v2. Exit 2 when pyiceberg is not installed (reported as skipped).

    python3 scripts/iceberg_foreign_commit.py <path to test_bolt_iceberg_foreign_commit> <scratch dir>
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

try:
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.table import StaticTable
    from pyiceberg.table.sorting import SortField, SortOrder
    from pyiceberg.table.statistics import BlobMetadata, StatisticsFile
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType, LongType, NestedField
except ImportError:
    print("SKIP: pyiceberg not installed")
    sys.exit(2)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  -- {detail}"))
    if not ok:
        FAILS.append(name)


SCHEMA = Schema(NestedField(1, "id", LongType(), required=True), NestedField(2, "score", DoubleType(), required=True))
ROWS = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "score": pa.array([1.5, 2.5, 3.5], pa.float64())},
                schema=pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("score", pa.float64(), nullable=False)]))


def hadoop_layout(table) -> str:
    """Expose the table's current metadata as metadata/v1.metadata.json + version-hint.text (the layout bolt opens)."""
    loc = table.metadata.location.replace("file://", "")
    shutil.copyfile(table.metadata_location.replace("file://", ""), os.path.join(loc, "metadata", "v1.metadata.json"))
    with open(os.path.join(loc, "metadata", "version-hint.text"), "w") as f:
        f.write("1")
    return loc


def run_bolt(binary: str, loc: str, mode: str) -> int:
    env = dict(os.environ, BOLT_ICEBERG_FOREIGN_DIR=loc, BOLT_ICEBERG_FOREIGN_MODE=mode)
    p = subprocess.run([binary], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
    return p.returncode


def main() -> int:
    binary, work = sys.argv[1], sys.argv[2]
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cat = SqlCatalog("f", uri=f"sqlite:///{work}/cat.db", warehouse=f"file://{work}/wh")
    cat.create_namespace("ns")

    t = cat.create_table("ns.t2", schema=SCHEMA, properties={"team": "oracle", "write.metadata.metrics.default": "full"},
                         sort_order=SortOrder(SortField(source_id=1, transform=IdentityTransform()),
                                              SortField(source_id=2, transform=IdentityTransform()), order_id=1))
    t.append(ROWS)
    t = cat.load_table("ns.t2")
    sid = t.metadata.current_snapshot_id
    t.manage_snapshots().create_tag(sid, "v1-tag").create_branch(sid, "dev").commit()
    t = cat.load_table("ns.t2")
    stat = StatisticsFile(snapshot_id=sid, statistics_path=f"file://{work}/stats.puffin", file_size_in_bytes=10,
                          file_footer_size_in_bytes=4, key_metadata=None,
                          blob_metadata=[BlobMetadata(type="apache-datasketches-theta-v1", snapshot_id=sid, sequence_number=1, fields=[1])])
    with t.update_statistics() as us:
        us.set_statistics(stat)
    t = cat.load_table("ns.t2")
    before = t.metadata
    loc = hadoop_layout(t)
    check("bolt appends to the pyiceberg v2 table", run_bolt(binary, loc, "append") == 0)
    after_path = os.path.join(loc, "metadata", "v2.metadata.json")
    check("bolt published v2.metadata.json", os.path.exists(after_path))
    if os.path.exists(after_path):
        after = StaticTable.from_metadata(f"file://{after_path}")
        m = after.metadata
        check("the tag survives", "v1-tag" in m.refs and m.refs["v1-tag"].snapshot_id == sid, str(m.refs))
        check("the branch survives", "dev" in m.refs and m.refs["dev"].snapshot_id == sid, str(m.refs))
        check("main moves to the new snapshot", m.refs["main"].snapshot_id == m.current_snapshot_id != sid, str(m.refs))
        check("the two-field sort order survives", [len(o.fields) for o in m.sort_orders if o.order_id == 1] == [2] and m.default_sort_order_id == 1,
              str(m.sort_orders))
        check("custom properties survive", m.properties.get("team") == "oracle" and m.properties.get("write.metadata.metrics.default") == "full",
              str(m.properties))
        check("the statistics entry survives", [s.statistics_path for s in m.statistics] == [s.statistics_path for s in before.statistics],
              str(m.statistics))
        check("the table uuid and format version are unchanged", m.table_uuid == before.table_uuid and m.format_version == 2)
        got = after.scan().to_arrow().to_pydict()
        check("pyiceberg reads the old rows and bolt's new ones", sorted(got["id"]) == [1, 2, 3, 900, 901, 902], str(got))
        try:
            import duckdb
            con = duckdb.connect()
            con.execute("INSTALL iceberg; LOAD iceberg;")
            ids = sorted(r[0] for r in con.execute(f"SELECT id FROM iceberg_scan('{after_path}')").fetchall())
            check("DuckDB reads the same rows", ids == [1, 2, 3, 900, 901, 902], str(ids))
        except ImportError:
            print("  (DuckDB not installed: skipped its read)")

    t1 = cat.create_table("ns.t1", schema=SCHEMA, properties={"format-version": "1"})
    t1.append(ROWS)
    loc1 = hadoop_layout(cat.load_table("ns.t1"))
    check("a v1 table is refused by bolt, not rewritten as v2", run_bolt(binary, loc1, "refuse") == 0)
    check("nothing was published for the v1 table", not glob.glob(os.path.join(loc1, "metadata", "v2.metadata.json")))
    print(f"{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
