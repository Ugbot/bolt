#!/usr/bin/env python3
"""Generate FLBA-logical-type parquet fixtures (G2PQ-16).

The parquet spec uses FIXED_LEN_BYTE_ARRAY as the physical storage for
several logical types beyond DECIMAL: UUID (FIXED[16]), FLOAT16 (FIXED[2]),
the deprecated ConvertedType.INTERVAL (FIXED[12]), and plain unannotated
fixed-width binary. bolt is checked against files written by REAL,
independent implementations -- a fixture bolt wrote could only prove bolt
self-consistent.

DuckDB 1.4.x writes a genuine `fixed_len_byte_array(16) ... (UUID)` for its
native UUID type and a genuine `fixed_len_byte_array(12) ... (Interval)` for
its native INTERVAL type (verified by inspecting the actual thrift schema
pyarrow reads back, not by assumption -- both are legal parquet LogicalType/
ConvertedType annotations per LogicalTypes.md). pyarrow writes a genuine
`fixed_len_byte_array(2) ... (Float16)` for pa.float16() and a plain
`fixed_len_byte_array(5)` with NO annotation at all for pa.binary(5).

Two files come out (one per writer):
  golden_flba_duckdb.parquet   id:UUID, iv:INTERVAL, n:int64 (5 rows, some NULL)
  golden_flba_pyarrow.parquet  h:FLOAT16, raw:binary(5) (unannotated), n:int64
"""
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def make_duckdb_fixture():
    # id: NULL at i==1; iv: NULL at i==2. Deterministic UUID string per row
    # so the expected 16 raw bytes are reproducible from Python's uuid module
    # in the test generator / gtest without re-reading the fixture.
    duckdb.sql("""
        SELECT
          CASE WHEN i = 1 THEN NULL ELSE
            CAST(printf('%08x-0000-4000-8000-%012d', i, i) AS UUID)
          END AS id,
          CASE WHEN i = 2 THEN NULL ELSE
            (INTERVAL (i) MONTHS + INTERVAL (i * 2) DAYS +
             INTERVAL (i * 1000) MILLISECONDS)
          END AS iv,
          i AS n
        FROM range(5) t(i)
    """).write_parquet("tests/data/golden_flba_duckdb.parquet")
    print("wrote tests/data/golden_flba_duckdb.parquet")


def make_pyarrow_fixture():
    h = pa.array([1.5, -2.0, 0.0, float("nan"), None], type=pa.float16())
    raw = pa.array([b"AAAAA", b"BBBBB", None, b"DDDDD", b"EEEEE"],
                    type=pa.binary(5))
    n = pa.array(list(range(5)), pa.int64())
    t = pa.table({"h": h, "raw": raw, "n": n})
    pq.write_table(t, "tests/data/golden_flba_pyarrow.parquet")
    print("wrote tests/data/golden_flba_pyarrow.parquet")


def verify():
    """Print the schemas back so a human/CI log can eyeball the annotations
    actually landed (not just that pyarrow can re-read its own output)."""
    for name in ("golden_flba_duckdb.parquet", "golden_flba_pyarrow.parquet"):
        f = pq.ParquetFile(f"tests/data/{name}")
        print(f"--- {name} ---")
        print(f.metadata.schema)


if __name__ == "__main__":
    make_duckdb_fixture()
    make_pyarrow_fixture()
    verify()
