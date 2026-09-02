#!/usr/bin/env python3
"""arrow_ipc_check.py — pyarrow oracle for bolt's Arrow IPC stream writer
(G2ARROW-10).

Opens the fixture test_bolt_arrow_ipc writes with pyarrow.ipc.open_stream
— the REAL external reader, not bolt reading its own bytes — and checks
schema + every value against the generating rule re-derived here in
python (mirrors tests/test_bolt_arrow_ipc.cpp exactly). A size/row-count
check alone would pass on scrambled bytes; values are the gate.

Usage: arrow_ipc_check.py <fixture.arrows>
Exit 0 = every check passed; nonzero with a message otherwise.
"""

import sys

import pyarrow as pa
import pyarrow.ipc as ipc

K_ROWS = 100


def int_val(i: int) -> int:
    return i * 3 - 50


def dbl_val(i: int) -> float:
    return float(i) * 0.5 - 10.0


def row_is_null(i: int) -> bool:
    return (i % 13) == 0


def str_val(i: int) -> str:
    if (i % 7) == 0:
        return f"spilled-string-value-{i}-padpadpad"
    return f"s{i}"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_batch(b: pa.RecordBatch, which: int) -> None:
    if b.num_rows != K_ROWS:
        fail(f"batch {which}: num_rows {b.num_rows} != {K_ROWS}")
    ints, floats, strs = b.column(0), b.column(1), b.column(2)
    for i in range(K_ROWS):
        if ints[i].as_py() != int_val(i):
            fail(f"batch {which} ints[{i}] = {ints[i].as_py()} "
                 f"!= {int_val(i)}")
        if floats[i].as_py() != dbl_val(i):
            fail(f"batch {which} floats[{i}] = {floats[i].as_py()} "
                 f"!= {dbl_val(i)}")
        if row_is_null(i):
            if strs[i].is_valid:
                fail(f"batch {which} strs[{i}] should be NULL, "
                     f"got {strs[i].as_py()!r}")
        else:
            if not strs[i].is_valid:
                fail(f"batch {which} strs[{i}] is NULL, "
                     f"expected {str_val(i)!r}")
            if strs[i].as_py() != str_val(i):
                fail(f"batch {which} strs[{i}] = {strs[i].as_py()!r} "
                     f"!= {str_val(i)!r}")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: arrow_ipc_check.py <fixture.arrows>")
    path = sys.argv[1]
    with open(path, "rb") as fh:
        reader = ipc.open_stream(fh)
        schema = reader.schema
        if schema.names != ["ints", "floats", "strs"]:
            fail(f"schema names {schema.names}")
        if schema.field(0).type != pa.int64():
            fail(f"col0 type {schema.field(0).type} != int64")
        if schema.field(1).type != pa.float64():
            fail(f"col1 type {schema.field(1).type} != float64")
        if schema.field(2).type != pa.utf8():
            fail(f"col2 type {schema.field(2).type} != utf8")
        batches = [b for b in reader]
    if len(batches) != 2:
        fail(f"expected 2 record batches, got {len(batches)}")
    for which, b in enumerate(batches):
        check_batch(b, which)
    n_null = sum(1 for i in range(K_ROWS) if row_is_null(i))
    if batches[0].column(2).null_count != n_null:
        fail(f"strs null_count {batches[0].column(2).null_count} "
             f"!= {n_null}")
    print(f"OK: pyarrow read 2 batches x {K_ROWS} rows, "
          f"all values match ({n_null} utf8 nulls verified)")


if __name__ == "__main__":
    main()
