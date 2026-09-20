#!/usr/bin/env python3
"""arrow_ipc_check.py — pyarrow oracle for bolt's Arrow IPC stream writer
(G2ARROW-10, widened to Bool/Date32/Binary/Decimal128 by G2ARROW-20).

Opens the fixture(s) test_bolt_arrow_ipc writes with pyarrow.ipc.open_stream
— the REAL external reader, not bolt reading its own bytes — and checks
schema + every value against the generating rule re-derived here in
python (mirrors tests/test_bolt_arrow_ipc.cpp exactly). A size/row-count
check alone would pass on scrambled bytes; values are the gate.

Usage: arrow_ipc_check.py <fixture.arrows> [wide_fixture.arrows]
  fixture.arrows       — the original Int64/Float64/Utf8 fixture (G2ARROW-10)
  wide_fixture.arrows  — optional: the +Bool/Date32/Binary/Decimal128
                          fixture (G2ARROW-20); checked when given.
Exit 0 = every check passed; nonzero with a message otherwise.
"""

import sys
from decimal import Decimal

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


def check_fixture(path: str) -> None:
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


# ---------------------------------------------------------------------
# Wide-types fixture (G2ARROW-20): Bool, Date32, Binary, Decimal128.
# Generating rules mirror WideFixture in tests/test_bolt_arrow_ipc.cpp
# exactly, re-derived independently here.
# ---------------------------------------------------------------------

DECIMAL_SCALE = 4


def bool_val(i: int) -> bool:
    return (i % 3) == 0


def bool_is_null(i: int) -> bool:
    return (i % 11) == 0


def date_val(i: int) -> int:
    return i * 5 - 200


def date_is_null(i: int) -> bool:
    return (i % 17) == 0


def decimal_mantissa(i: int) -> int:
    return (i - 50) * 100000007


def decimal_is_null(i: int) -> bool:
    return (i % 19) == 0


def binary_val(i: int) -> bytes:
    b = bytes([0xFF, 0xFE, 0x00, i & 0xFF, 0x80, 0xC0, 0xC1])
    if (i % 7) == 0:
        b += bytes([0x80 + (k & 0x3F) for k in range(10)])
    return b


def binary_is_null(i: int) -> bool:
    return (i % 23) == 0


def check_wide_fixture(path: str) -> None:
    import datetime

    with open(path, "rb") as fh:
        reader = ipc.open_stream(fh)
        schema = reader.schema
        want_names = ["ints", "floats", "strs", "bools", "dates", "bins",
                      "decimals"]
        if schema.names != want_names:
            fail(f"wide schema names {schema.names} != {want_names}")
        if schema.field(0).type != pa.int64():
            fail(f"wide col0 type {schema.field(0).type} != int64")
        if schema.field(1).type != pa.float64():
            fail(f"wide col1 type {schema.field(1).type} != float64")
        if schema.field(2).type != pa.utf8():
            fail(f"wide col2 type {schema.field(2).type} != utf8")
        if schema.field(3).type != pa.bool_():
            fail(f"wide col3 type {schema.field(3).type} != bool")
        if schema.field(4).type != pa.date32():
            fail(f"wide col4 type {schema.field(4).type} != date32")
        if schema.field(5).type != pa.binary():
            fail(f"wide col5 type {schema.field(5).type} != binary")
        if schema.field(6).type != pa.decimal128(38, DECIMAL_SCALE):
            fail(f"wide col6 type {schema.field(6).type} != "
                 f"decimal128(38,{DECIMAL_SCALE})")
        batches = [b for b in reader]
    if len(batches) != 1:
        fail(f"wide: expected 1 record batch, got {len(batches)}")
    b = batches[0]
    if b.num_rows != K_ROWS:
        fail(f"wide: num_rows {b.num_rows} != {K_ROWS}")
    ints, floats, strs = b.column(0), b.column(1), b.column(2)
    bools, dates, bins, decs = b.column(3), b.column(4), b.column(5), b.column(6)

    epoch = datetime.date(1970, 1, 1)
    n_bool_null = n_date_null = n_bin_null = n_dec_null = 0
    for i in range(K_ROWS):
        if ints[i].as_py() != int_val(i):
            fail(f"wide ints[{i}] = {ints[i].as_py()} != {int_val(i)}")
        if floats[i].as_py() != dbl_val(i):
            fail(f"wide floats[{i}] = {floats[i].as_py()} != {dbl_val(i)}")

        if row_is_null(i):
            if strs[i].is_valid:
                fail(f"wide strs[{i}] should be NULL")
        elif strs[i].as_py() != str_val(i):
            fail(f"wide strs[{i}] = {strs[i].as_py()!r} != {str_val(i)!r}")

        if bool_is_null(i):
            n_bool_null += 1
            if bools[i].is_valid:
                fail(f"wide bools[{i}] should be NULL")
        elif bools[i].as_py() != bool_val(i):
            fail(f"wide bools[{i}] = {bools[i].as_py()} != {bool_val(i)}")

        if date_is_null(i):
            n_date_null += 1
            if dates[i].is_valid:
                fail(f"wide dates[{i}] should be NULL")
        else:
            want_date = epoch + datetime.timedelta(days=date_val(i))
            if dates[i].as_py() != want_date:
                fail(f"wide dates[{i}] = {dates[i].as_py()} != {want_date}")

        if binary_is_null(i):
            n_bin_null += 1
            if bins[i].is_valid:
                fail(f"wide bins[{i}] should be NULL")
        elif bins[i].as_py() != binary_val(i):
            fail(f"wide bins[{i}] = {bins[i].as_py()!r} != "
                 f"{binary_val(i)!r}")

        if decimal_is_null(i):
            n_dec_null += 1
            if decs[i].is_valid:
                fail(f"wide decimals[{i}] should be NULL")
        else:
            want_dec = Decimal(decimal_mantissa(i)).scaleb(-DECIMAL_SCALE)
            if decs[i].as_py() != want_dec:
                fail(f"wide decimals[{i}] = {decs[i].as_py()} != {want_dec}")

    if bools.null_count != n_bool_null:
        fail(f"bools null_count {bools.null_count} != {n_bool_null}")
    if dates.null_count != n_date_null:
        fail(f"dates null_count {dates.null_count} != {n_date_null}")
    if bins.null_count != n_bin_null:
        fail(f"bins null_count {bins.null_count} != {n_bin_null}")
    if decs.null_count != n_dec_null:
        fail(f"decimals null_count {decs.null_count} != {n_dec_null}")

    # Sign check: prove two's-complement negative mantissas actually
    # round-tripped (not just their magnitude) — row 1's mantissa is
    # negative by construction ((1-50)*100000007 < 0).
    if decimal_mantissa(1) >= 0:
        fail("test bug: expected row 1 decimal mantissa to be negative")
    if decs[1].as_py() >= 0:
        fail(f"wide decimals[1] = {decs[1].as_py()} should be negative "
             "(two's-complement sign lost)")

    print(f"OK: wide-types pyarrow read {K_ROWS} rows across 7 columns "
          f"(bool={n_bool_null} date32={n_date_null} binary={n_bin_null} "
          f"decimal128={n_dec_null} nulls verified; negative decimal "
          "sign verified)")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        fail("usage: arrow_ipc_check.py <fixture.arrows> [wide_fixture.arrows]")
    check_fixture(sys.argv[1])
    if len(sys.argv) == 3:
        check_wide_fixture(sys.argv[2])


if __name__ == "__main__":
    main()
