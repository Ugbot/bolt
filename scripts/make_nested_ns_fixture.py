#!/usr/bin/env python3
"""Generate the exact-nanosecond nested-reader regression with PyArrow."""
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1] / "tests/data/nested_read"
kind = pa.struct([pa.field("local",pa.timestamp("ns")),pa.field("utc",pa.timestamp("ns",tz="UTC"))])
rows = [{"local":1234567891,"utc":-1},None,{"local":None,"utc":1700000000000000001},{"local":-1,"utc":None}]
path = root / "timestamp_ns.parquet"
pq.write_table(pa.table({"payload":pa.array(rows,type=kind)}),path,row_group_size=2,version="2.6",compression="snappy")
read = pq.read_table(path)["payload"].combine_chunks()
assert read.field("local").cast(pa.int64()).to_pylist() == [1234567891,None,None,-1]
assert read.field("utc").cast(pa.int64()).to_pylist() == [-1,None,1700000000000000001,None]
print(path)
