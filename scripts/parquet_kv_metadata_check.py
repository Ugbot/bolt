#!/usr/bin/env python3
"""G2PQ-23: verify FileMetaData.key_value_metadata (thrift field 5) using a
REAL pyarrow.parquet.ParquetFile -- a strictly stronger oracle than a
hand-rolled thrift decoder, since pyarrow.metadata.metadata is the exact
dict Arrow's own reader trusts for ARROW:schema.

    ./test_bolt_parquet_write_kv_metadata     # writes the fixture
    python3 scripts/parquet_kv_metadata_check.py <dir-containing-fixture>

Column-level key_value_metadata (ColumnMetaData field 8) is checked against
bolt's own reader in test_bolt_parquet_write_kv_metadata.cpp's
RoundTripsThroughBoltsOwnReader case instead of here: pyarrow's own writer
has no public API to SET column-level key_value_metadata (confirmed on
pyarrow 21.0.0 -- ParquetFile.metadata.row_group(0).column(0).metadata is
None for any file pyarrow itself writes), so it cannot serve as an
independent oracle for that half; it can still READ the attribute (the
`.metadata` property exists on ColumnChunkMetaData), so this script also
does a best-effort read of it as a secondary check.
"""
import os
import sys

import pyarrow.parquet as pq

ARROW_SCHEMA_B64 = (
    "/////6gAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAIA"
    "AABAAAAABAAAANj///8AAAEFEAAAABgAAAAEAAAAAAAAAAEAAABiAAAABAAEAAQAAAAQABQA"
    "CAAGAAcADAAAABAAEAAAAAAAAQIQAAAAHAAAAAQAAAAAAAAAAQAAAGEAAAAIAAwACAAHAAgA"
    "AAAAAAABQAAAAAAAAAA="
)


def check(path, kv_pairs, expect_col0_field_id=None):
    pf = pq.ParquetFile(path)
    meta = pf.metadata.metadata or {}
    for key, val in kv_pairs.items():
        got = meta.get(key.encode())
        assert got is not None, f"{path}: missing file-level key {key!r} (have {list(meta)})"
        assert got == val.encode(), f"{path}: {key} = {got!r}, want {val!r}"
    print(f"  file-level key_value_metadata OK ({list(meta)})")

    if expect_col0_field_id is not None:
        col = pf.metadata.row_group(0).column(0)
        if col.metadata is not None:
            got = col.metadata.get(b"PARQUET:field_id")
            assert got == expect_col0_field_id.encode(), (
                f"{path}: column 0 PARQUET:field_id = {got!r}, want {expect_col0_field_id!r}"
            )
            print(f"  column-level key_value_metadata OK ({dict(col.metadata)})")
        else:
            print("  (pyarrow's Cython layer did not surface column.metadata on this "
                  "build -- covered independently by the C++ round-trip test instead)")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    d = sys.argv[1]

    fixture = os.path.join(d, "test_bolt_parquet_write_kv_metadata_fixture.parquet")
    roundtrip = os.path.join(d, "test_bolt_parquet_write_kv_metadata_roundtrip.parquet")
    empty = os.path.join(d, "test_bolt_parquet_write_kv_metadata_empty.parquet")

    n = 0
    if os.path.exists(fixture):
        print(f"checking {fixture}")
        check(fixture, {"ARROW:schema": ARROW_SCHEMA_B64}, expect_col0_field_id="1")
        n += 1
    if os.path.exists(roundtrip):
        print(f"checking {roundtrip}")
        check(roundtrip, {"ARROW:schema": ARROW_SCHEMA_B64, "custom.note": "written by bolt"},
              expect_col0_field_id="1")
        n += 1
    if os.path.exists(empty):
        print(f"checking {empty} (must have NO key_value_metadata)")
        pf = pq.ParquetFile(empty)
        meta = pf.metadata.metadata
        assert not meta, f"{empty}: expected no key_value_metadata, got {meta}"
        print("  absent as expected")
        n += 1

    if n == 0:
        print(f"no fixtures found under {d} -- run the gtest binary first", file=sys.stderr)
        return 1
    print(f"{n} fixture(s) verified via real pyarrow.parquet.ParquetFile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
