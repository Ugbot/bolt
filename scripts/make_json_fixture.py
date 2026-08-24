#!/usr/bin/env python3
"""Regenerate the JSON-logical-type parquet fixture.

pyarrow models a JSON column as `extension<arrow.json>` over string storage
and writes it to parquet as a BYTE_ARRAY annotated with the JSON logical type
(and the legacy ConvertedType JSON). That is the reference bolt is checked
against: a fixture bolt wrote could only prove bolt self-consistent.
"""
import pyarrow as pa
import pyarrow.parquet as pq

N = 300


def doc(i):
    return '{"id":%d,"tag":"t%d","nested":{"v":%d}}' % (i, i % 7, i * 3)


def main():
    j = [None if i % 13 == 0 else doc(i) for i in range(N)]
    t = pa.table({
        "j": pa.array(j, pa.json_()),
        "plain": pa.array([doc(i) for i in range(N)], pa.string()),
        "n": pa.array(list(range(N)), pa.int64()),
    })
    pq.write_table(t, "tests/data/golden_json.parquet", compression="snappy")
    print("wrote tests/data/golden_json.parquet")


if __name__ == "__main__":
    main()
