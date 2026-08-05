#!/usr/bin/env python3
"""Regenerate `golden_iceberg_table/` — a COMPLETE real Iceberg warehouse.

G2FEAT-76 committed three real manifest .avro files, which is enough to prove
the manifest DECODER. It is not enough to prove a SCAN: a scan needs the
metadata.json, the manifest list, the manifests AND the referenced Parquet data
files, all in one tree. This script produces that tree, and
test_bolt_iceberg_real_scan.cpp reads it end to end.

    pip install "pyiceberg[pyarrow]" sqlalchemy
    python make_iceberg_real_table.py

Everything under golden_iceberg_table/ is written by **pyiceberg 0.11.1** driving
a real `SqlCatalog` warehouse — `create_table` + two `append(pyarrow.Table)`
calls, at pyiceberg's DEFAULT settings (no table properties are set at all).
bolt writes none of it. Format-version 2, identity-partitioned on `sym`,
`"avro.codec":"deflate"` manifests, **`zstd` Parquet data files** (the default),
two snapshots, five data files.

THE TABLE (db.trades — id:long, sym:string, price:double, active:boolean):
    append #1 -> 6 rows  AAPL(101.5,102.25) MSFT(310.0,311.75)
                         GOOG(140.0, NULL price, NULL active)   => 3 files
    append #2 -> 3 rows  AAPL(103.0) TSLA(250.5,251.0)          => 2 files
    current snapshot therefore sees BOTH manifests: 5 files, 9 rows.

ONE DELIBERATE CHOICE, disclosed rather than silently convenient:

1. The tree is committed with the ABSOLUTE `file://` locations baked in by the
   generating machine, exactly as pyiceberg wrote them — they are NOT rewritten
   to be relative. That is the point: reading the fixture from wherever it is
   checked out exercises the scan's real relocation logic (strip the metadata
   `location` prefix, resolve under the table's actual directory), which is the
   same code path a table moved between buckets or restored from backup needs.

HISTORY: this fixture originally set `write.parquet.compression-codec=snappy`,
because bolt's Parquet reader had no zstd and pyiceberg's default is zstd — so
the fixture was the happy path, disclosed but real. G2FEAT-134 gave bolt a
self-contained zstd decoder (include/bolt/ingest/bolt_zstd_dec.h), so the
property is gone and this warehouse is now written at pyiceberg's untouched
defaults. That is the whole point: the table a user actually gets from the most
common Python writer is the table bolt reads.

Not wired into CMake — the tree is committed; this exists so it can be rebuilt
or bumped to a newer Iceberg writer deliberately. If you change the data below
you MUST update test_bolt_iceberg_real_scan.cpp, which asserts VALUES.
"""
import os
import shutil
import sys
import tempfile

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import BooleanType, DoubleType, LongType, NestedField, StringType

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "golden_iceberg_table")

WH = tempfile.mkdtemp(prefix="bolt_iceberg_gen_")
WH_URI = WH.replace("\\", "/")

cat = SqlCatalog("real", uri=f"sqlite:///{WH_URI}/catalog.db", warehouse=f"file://{WH_URI}")
cat.create_namespace("db")

tbl = cat.create_table(
    "db.trades",
    schema=Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "sym", StringType(), required=False),
        NestedField(3, "price", DoubleType(), required=False),
        NestedField(4, "active", BooleanType(), required=False),
    ),
    partition_spec=PartitionSpec(
        PartitionField(source_id=2, field_id=1000,
                       transform=IdentityTransform(), name="sym")),
    # NO properties: pyiceberg's DEFAULTS, which means zstd data files.
)

schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("sym", pa.string()),
    pa.field("price", pa.float64()),
    pa.field("active", pa.bool_()),
])

tbl.append(pa.Table.from_pydict({
    "id":     [1, 2, 3, 4, 5, 6],
    "sym":    ["AAPL", "AAPL", "MSFT", "MSFT", "GOOG", "GOOG"],
    "price":  [101.5, 102.25, 310.0, 311.75, 140.0, None],
    "active": [True, False, True, True, None, False],
}, schema=schema))

tbl.append(pa.Table.from_pydict({
    "id":     [7, 8, 9],
    "sym":    ["AAPL", "TSLA", "TSLA"],
    "price":  [103.0, 250.5, 251.0],
    "active": [True, False, True],
}, schema=schema))

# Ground truth the C++ test asserts against, printed so a regeneration can be
# diffed by eye before it is committed.
tbl = cat.load_table("db.trades")
print("current-snapshot-id:", tbl.metadata.current_snapshot_id)
print("pyiceberg scan:", tbl.scan().to_arrow().to_pydict())

src = os.path.join(WH, "db", "trades")
if os.path.exists(DEST):
    shutil.rmtree(DEST)
shutil.copytree(src, os.path.join(DEST, "db", "trades"))

n = sum(len(f) for _, _, f in os.walk(DEST))
print(f"wrote {n} files to {DEST}")
# The SqlCatalog holds catalog.db open, so a hard rmtree can race it on Windows.
shutil.rmtree(WH, ignore_errors=True)
if os.path.exists(WH):
    print("NOTE: could not remove the scratch warehouse; delete it:", WH,
          file=sys.stderr)
