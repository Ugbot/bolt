#!/usr/bin/env python3
"""arrow_ipc_check.py — pyarrow oracle for bolt's Arrow IPC stream writer
(G2ARROW-10, widened to Bool/Date32/Binary/Decimal128 by G2ARROW-20, nested
List/Struct by G2ARROW-21).

Opens the fixture(s) test_bolt_arrow_ipc writes with pyarrow.ipc.open_stream
— the REAL external reader, not bolt reading its own bytes — and checks
schema + every value against the generating rule re-derived here in
python (mirrors tests/test_bolt_arrow_ipc.cpp exactly). A size/row-count
check alone would pass on scrambled bytes; values are the gate.

Usage: arrow_ipc_check.py [--timestamp ts_fixture.arrows] <fixture.arrows> [wide_fixture.arrows] [nested_fixture.arrows]
  fixture.arrows        — the original Int64/Float64/Utf8 fixture (G2ARROW-10)
  wide_fixture.arrows   — optional: the +Bool/Date32/Binary/Decimal128
                           fixture (G2ARROW-20); checked when given.
  nested_fixture.arrows — optional: the List<Utf8>/Struct{Int64,Utf8}
                           fixture (G2ARROW-21); checked when given.
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


#  ----------------------------------------------------------------------
# Nested fixture (G2ARROW-21): List<Utf8> "tags" and Struct{id: Int64,
# label: Utf8} "person", alongside a plain Int64 "ids" baseline column.
# Generating rules mirror NestedFixture in tests/test_bolt_arrow_ipc.cpp
# exactly, re-derived independently here — including nulls at the PARENT
# level (a whole list/struct row null) *and* the CHILD level (one
# element/field null while the row itself is present), and an EMPTY
# (non-null) list sharing the same offsets[i]==offsets[i+1] span as a
# null list, distinguished only by the validity bit.
# ----------------------------------------------------------------------


def tag_list_is_null(i: int) -> bool:
    return (i % 13) == 0


def tag_list_is_empty(i: int) -> bool:
    return not tag_list_is_null(i) and (i % 5) == 0


def tag_list_count(i: int) -> int:
    if tag_list_is_null(i) or tag_list_is_empty(i):
        return 0
    return 1 + (i % 4)


def tag_elem_is_null(j: int) -> bool:
    return (j % 3) == 2


def tag_elem_val(i: int, j: int) -> str:
    return f"t{i}_{j}"


def person_is_null(i: int) -> bool:
    return (i % 17) == 0


def person_id_is_null(i: int) -> bool:
    return (i % 29) == 0


def person_label_is_null(i: int) -> bool:
    return (i % 9) == 0


def person_label_val(i: int) -> str:
    return f"p{i}"


def check_nested_fixture(path: str) -> None:
    with open(path, "rb") as fh:
        reader = ipc.open_stream(fh)
        schema = reader.schema
        want_names = ["ids", "tags", "person"]
        if schema.names != want_names:
            fail(f"nested schema names {schema.names} != {want_names}")
        if schema.field(0).type != pa.int64():
            fail(f"nested col0 type {schema.field(0).type} != int64")
        tags_t = schema.field(1).type
        if not pa.types.is_list(tags_t):
            fail(f"nested col1 type {tags_t} is not a list type")
        if tags_t.value_type != pa.utf8():
            fail(f"nested col1 value_type {tags_t.value_type} != utf8")
        person_t = schema.field(2).type
        if not pa.types.is_struct(person_t):
            fail(f"nested col2 type {person_t} is not a struct type")
        if person_t.num_fields != 2:
            fail(f"nested col2 num_fields {person_t.num_fields} != 2")
        if person_t.field(0).name != "id" or person_t.field(0).type != pa.int64():
            fail(f"nested col2 field0 {person_t.field(0)} != id: int64")
        if person_t.field(1).name != "label" or person_t.field(1).type != pa.utf8():
            fail(f"nested col2 field1 {person_t.field(1)} != label: utf8")
        batches = [b for b in reader]
    if len(batches) != 1:
        fail(f"nested: expected 1 record batch, got {len(batches)}")
    b = batches[0]
    if b.num_rows != K_ROWS:
        fail(f"nested: num_rows {b.num_rows} != {K_ROWS}")

    ids, tags, person = b.column(0), b.column(1), b.column(2)
    n_list_null = n_struct_null = n_id_null = n_label_null = 0
    saw_empty_nonnull_list = False

    for i in range(K_ROWS):
        if ids[i].as_py() != int_val(i):
            fail(f"nested ids[{i}] = {ids[i].as_py()} != {int_val(i)}")

        # --- List<Utf8> "tags" ---
        if tag_list_is_null(i):
            n_list_null += 1
            if tags[i].is_valid:
                fail(f"nested tags[{i}] should be NULL (parent-level)")
        else:
            if not tags[i].is_valid:
                fail(f"nested tags[{i}] should NOT be NULL")
            got = tags[i].as_py()
            want_count = tag_list_count(i)
            if len(got) != want_count:
                fail(f"nested tags[{i}] len {len(got)} != {want_count}")
            if tag_list_is_empty(i):
                if len(got) != 0:
                    fail(f"nested tags[{i}] should be EMPTY (not null)")
                saw_empty_nonnull_list = True
            for j in range(want_count):
                if tag_elem_is_null(j):
                    if got[j] is not None:
                        fail(f"nested tags[{i}][{j}] should be NULL "
                             f"(child-level), got {got[j]!r}")
                else:
                    want = tag_elem_val(i, j)
                    if got[j] != want:
                        fail(f"nested tags[{i}][{j}] = {got[j]!r} != {want!r}")

        # --- Struct{id, label} "person" ---
        if person_is_null(i):
            n_struct_null += 1
            if person[i].is_valid:
                fail(f"nested person[{i}] should be NULL (parent-level)")
        else:
            if not person[i].is_valid:
                fail(f"nested person[{i}] should NOT be NULL")
            got = person[i].as_py()
            if person_id_is_null(i):
                n_id_null += 1
                if got["id"] is not None:
                    fail(f"nested person[{i}].id should be NULL "
                         f"(child-level), got {got['id']!r}")
            elif got["id"] != int_val(i):
                fail(f"nested person[{i}].id = {got['id']} != {int_val(i)}")
            if person_label_is_null(i):
                n_label_null += 1
                if got["label"] is not None:
                    fail(f"nested person[{i}].label should be NULL "
                         f"(child-level), got {got['label']!r}")
            else:
                want = person_label_val(i)
                if got["label"] != want:
                    fail(f"nested person[{i}].label = {got['label']!r} "
                         f"!= {want!r}")

    if tags.null_count != n_list_null:
        fail(f"nested tags null_count {tags.null_count} != {n_list_null}")
    if person.null_count != n_struct_null:
        fail(f"nested person null_count {person.null_count} != {n_struct_null}")
    if not saw_empty_nonnull_list:
        fail("test bug: expected at least one EMPTY (non-null) list row")

    print(f"OK: nested pyarrow read {K_ROWS} rows (List<Utf8> "
          f"null={n_list_null} parent-nulls verified + empty-vs-null "
          f"distinction verified; Struct null={n_struct_null} parent-nulls, "
          f"id child-null={n_id_null}, label child-null={n_label_null} "
          "verified)")


def ts_val(i: int) -> int:
    return (i - 10) * 86400000000 + i * 1234567


def check_timestamp_fixture(path: str) -> None:
    with pa.OSFile(path, "rb") as f:
        reader = ipc.open_stream(f)
        schema = reader.schema
        batches = list(reader)
    if schema.names != ["id", "ts"]:
        fail(f"timestamp schema names {schema.names}")
    if schema.field(1).type != pa.timestamp("us"):
        fail(f"timestamp col1 type {schema.field(1).type} != timestamp[us]")
    b = batches[0]
    if len(batches) != 1 or b.num_rows != K_ROWS:
        fail("timestamp: expected one batch of K_ROWS")
    ts = b.column(1).cast(pa.int64())
    n_null = 0
    for i in range(K_ROWS):
        if i % 7 == 3:
            n_null += 1
            if ts[i].is_valid:
                fail(f"timestamp ts[{i}] should be NULL")
        elif ts[i].as_py() != ts_val(i):
            fail(f"timestamp ts[{i}] = {ts[i].as_py()} != {ts_val(i)}")
    print(f"OK: timestamp[us] pyarrow read {K_ROWS} rows ({n_null} NULL)")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--timestamp":
        check_timestamp_fixture(sys.argv[2])
        del sys.argv[1:3]
        if len(sys.argv) == 1:
            return
    if len(sys.argv) not in (2, 3, 4):
        fail("usage: arrow_ipc_check.py <fixture.arrows> "
             "[wide_fixture.arrows] [nested_fixture.arrows]")
    check_fixture(sys.argv[1])
    if len(sys.argv) >= 3:
        check_wide_fixture(sys.argv[2])
    if len(sys.argv) == 4:
        check_nested_fixture(sys.argv[3])


if __name__ == "__main__":
    main()
