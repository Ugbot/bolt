#!/usr/bin/env python3
"""Regenerate the golden_iceberg_real_*.avro fixtures.

These fixtures must be produced by a REAL Iceberg implementation — a synthetic
Avro file written by bolt would prove nothing about real-world compatibility,
which is the entire point of test_bolt_iceberg_real_avro.cpp.

    pip install "pyiceberg[pyarrow]" sqlalchemy
    python make_iceberg_real_avro.py

Produced with pyiceberg 0.11.1 / pyarrow 25.0.0: format-version 2, identity
partitioning on `sym`, "avro.codec":"deflate". The table's VALUES are asserted
by the C++ test, so if you change the data below you must update that test.

Not wired into CMake — the fixtures are committed; this exists so they can be
rebuilt or bumped to a newer Iceberg writer deliberately.
"""
import glob
import os
import shutil
import sys

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import BooleanType, DoubleType, LongType, NestedField, StringType

HERE = os.path.dirname(os.path.abspath(__file__))
WH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_iceberg_wh")

if os.path.exists(WH):
    shutil.rmtree(WH)
os.makedirs(WH)

cat = SqlCatalog("real", uri=f"sqlite:///{WH}/catalog.db", warehouse=f"file://{WH}")
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
)

schema = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("sym", pa.string()),
    pa.field("price", pa.float64()),
    pa.field("active", pa.bool_()),
])

# append #1 -> 3 data files (AAPL, GOOG, MSFT); GOOG carries a NULL price so the
# null_value_counts assertions have something real to read.
tbl.append(pa.Table.from_pydict({
    "id":     [1, 2, 3, 4, 5, 6],
    "sym":    ["AAPL", "AAPL", "MSFT", "MSFT", "GOOG", "GOOG"],
    "price":  [101.5, 102.25, 310.0, 311.75, 140.0, None],
    "active": [True, False, True, True, None, False],
}, schema=schema))

# append #2 -> 2 data files (AAPL, TSLA).
tbl.append(pa.Table.from_pydict({
    "id":     [7, 8, 9],
    "sym":    ["AAPL", "TSLA", "TSLA"],
    "price":  [103.0, 250.5, 251.0],
    "active": [True, False, True],
}, schema=schema))

tbl = cat.load_table("db.trades")
cur = tbl.metadata.current_snapshot_id
meta = os.path.join(WH, "db", "trades", "metadata")

# Newest snapshot's manifest list, its manifest, and the first snapshot's
# manifest (the one holding the NULL price).
snap = glob.glob(os.path.join(meta, f"snap-{cur}-*.avro"))[0]
mans = sorted(glob.glob(os.path.join(meta, "*-m0.avro")), key=os.path.getmtime)
shutil.copy(snap, os.path.join(HERE, "golden_iceberg_real_manifest_list.avro"))
shutil.copy(mans[-1], os.path.join(HERE, "golden_iceberg_real_manifest.avro"))
shutil.copy(mans[0], os.path.join(HERE, "golden_iceberg_real_manifest_nulls.avro"))
# The SqlCatalog keeps catalog.db open, so a hard rmtree races it on Windows.
shutil.rmtree(WH, ignore_errors=True)
print("regenerated 3 fixtures in", HERE)
if os.path.exists(WH):
    print("NOTE: could not remove the scratch warehouse; delete it:", WH)
