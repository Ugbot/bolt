#!/usr/bin/env python3
"""G2PQ-26 (spec B6): verify RowGroup.sorting_columns against pyarrow.

Unlike encoding_stats or the file_offset/total_compressed_size/ordinal
fields, pyarrow DOES expose this one directly:
ParquetFile(...).metadata.row_group(i).sorting_columns -- confirmed on
pyarrow 21.0.0 by direct inspection, not assumed. So pyarrow is the primary
oracle here, not a from-scratch thrift reader.

    ./test_bolt_parquet_write_sorting_columns
    python3 scripts/parquet_sorting_columns_check.py <dir-containing-fixtures>
"""
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("SKIP: pyarrow not installed")
    sys.exit(0)


def check_asc(path):
    rg = pq.ParquetFile(path).metadata.row_group(0)
    errs = []
    if len(rg.sorting_columns) != 1:
        errs.append(f"expected 1 sorting_columns entry, got {len(rg.sorting_columns)}")
        return errs
    sc = rg.sorting_columns[0]
    if sc.column_index != 0:
        errs.append(f"column_index: got {sc.column_index}, want 0")
    if sc.descending is not False:
        errs.append(f"descending: got {sc.descending}, want False")
    if sc.nulls_first is not False:
        errs.append(f"nulls_first: got {sc.nulls_first}, want False")
    return errs


def check_desc(path):
    rg = pq.ParquetFile(path).metadata.row_group(0)
    errs = []
    if len(rg.sorting_columns) != 1:
        errs.append(f"expected 1 sorting_columns entry, got {len(rg.sorting_columns)}")
        return errs
    sc = rg.sorting_columns[0]
    if sc.column_index != 0:
        errs.append(f"column_index: got {sc.column_index}, want 0")
    if sc.descending is not True:
        errs.append(f"descending: got {sc.descending}, want True")
    if sc.nulls_first is not False:
        errs.append(f"nulls_first: got {sc.nulls_first}, want False")
    return errs


def check_nullsfirst(path):
    rg = pq.ParquetFile(path).metadata.row_group(0)
    errs = []
    if len(rg.sorting_columns) != 1:
        errs.append(f"expected 1 sorting_columns entry, got {len(rg.sorting_columns)}")
        return errs
    sc = rg.sorting_columns[0]
    if sc.column_index != 0:
        errs.append(f"column_index: got {sc.column_index}, want 0")
    if sc.descending is not False:
        errs.append(f"descending: got {sc.descending}, want False")
    if sc.nulls_first is not True:
        errs.append(f"nulls_first: got {sc.nulls_first}, want True")
    return errs


def check_no_claim(path):
    # A row group written with NO caller claim must carry NO
    # sorting_columns entries at all -- the "no claim" baseline this
    # ticket's fail-loud policy must never accidentally populate.
    rg = pq.ParquetFile(path).metadata.row_group(0)
    errs = []
    if len(rg.sorting_columns) != 0:
        errs.append(f"expected 0 sorting_columns entries, got {len(rg.sorting_columns)}")
    return errs


CHECKS = {
    "test_bolt_parquet_write_sorting_columns_asc.parquet": check_asc,
    "test_bolt_parquet_write_sorting_columns_desc.parquet": check_desc,
    "test_bolt_parquet_write_sorting_columns_nullsfirst.parquet": check_nullsfirst,
    "test_bolt_parquet_write_sorting_columns_no_claim.parquet": check_no_claim,
}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    all_errs = []
    for name, fn in CHECKS.items():
        path = os.path.join(d, name)
        if not os.path.exists(path):
            print(f"FAIL: missing fixture {path}")
            return 1
        errs = fn(path)
        if errs:
            all_errs.extend(f"{name}: {e}" for e in errs)
        else:
            print(f"OK: {name}")

    if all_errs:
        for e in all_errs:
            print(f"FAIL: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
