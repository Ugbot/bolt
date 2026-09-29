#!/usr/bin/env python3
"""Regenerate golden_iceberg_day_table/ (G2ICE-234) with pyiceberg 0.10.0.

A v2 table partitioned by day(ts) and identity(tenant), two appends. pyiceberg
declares the day partition value as ["null", {"type":"int","logicalType":
"date"}] -- the union shape bolt's Avro flattener mis-split into two fields.

    python3 make_iceberg_day_table.py /tmp/wh && cp -R /tmp/wh/db/ev \
        golden_iceberg_day_table/db/
"""
import os, sys, datetime as dt, pyarrow as pa, json, glob, fastavro
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, TimestampType, IntegerType, LongType
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform, IdentityTransform
wh = sys.argv[1]; os.makedirs(wh, exist_ok=True)
cat = SqlCatalog("t", uri=f"sqlite:///{wh}/cat.db", warehouse=f"file://{wh}")
cat.create_namespace("db")
sch = Schema(NestedField(1,"ts",TimestampType(),required=False), NestedField(2,"tenant",IntegerType(),required=False), NestedField(3,"v",LongType(),required=False))
spec = PartitionSpec(PartitionField(1,1000,DayTransform(),"ts_day"), PartitionField(2,1001,IdentityTransform(),"tenant"))
t = cat.create_table("db.ev", schema=sch, partition_spec=spec)
def T(rows): return pa.table({"ts":pa.array([r[0] for r in rows],pa.timestamp("us")),"tenant":pa.array([r[1] for r in rows],pa.int32()),"v":pa.array([r[2] for r in rows],pa.int64())})
d=dt.datetime
t.append(T([(d(2024,3,1,1),7,10),(d(2024,3,1,2),7,11),(d(2024,3,2,5),8,12)]))
t.append(T([(d(2024,5,9,1),9,20),(d(2024,5,10,1),7,21)]))
