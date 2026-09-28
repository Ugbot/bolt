#!/usr/bin/env python3
"""Limits review L6: bolt's wide-column files against a real reader (pyarrow).

    ./test_bolt_l6_wide_columns            # writes the two bolt fixtures
    python3 scripts/l6_wide_columns_check.py <dir-with-fixtures> <test-binary>

1. pyarrow reads l6_wide_bolt.parquet (4,096 columns, 3 row groups) and every
   cell matches the formulas in test_bolt_l6_wide_columns.cpp.
2. pyarrow reads l6_wide.arrows (a 4,096-column Arrow IPC stream) the same way.
3. pyarrow writes a 5,000-column Parquet file and bolt reads it back
   (L6WideColumns.ReadsPyarrowWideFile): readers accept what writers emit, in
   both directions.
Exits 1 on any mismatch.
"""
import os
import subprocess
import sys
import tempfile

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError:
    print("SKIP: pyarrow not installed")
    sys.exit(77)

WIDE, ROWS = 4096, 300


def expect(col, ty, ipc_types):
    if ipc_types and col % 4 == 3:
        ty = "i64"
    out = []
    for r in range(ROWS):
        if col % 5 == 4 and (r + col) % 7 == 0:
            out.append(None)
        elif ty == "i64":
            out.append(r * 1000003 - col * 7)
        elif ty == "f64":
            out.append(r * 0.25 + col)
        elif ty == "utf8":
            out.append(f"long-value-c{col}-r{r}" if r % 3 == 0 else f"c{col}r{r}")
        else:
            out.append(r * 3 - col)
    return out


def check_table(table, what, ipc_types):
    assert table.num_columns == WIDE, f"{what}: {table.num_columns} columns"
    assert table.num_rows == ROWS, f"{what}: {table.num_rows} rows"
    kinds = ["i64", "f64", "utf8", "i32"]
    bad = 0
    for c in range(WIDE):
        name = table.column_names[c]
        assert name == f"col{c:05d}", f"{what}: column {c} named {name}"
        got = table.column(c).to_pylist()
        want = expect(c, kinds[c % 4], ipc_types)
        if got != want:
            bad += 1
            if bad < 5:
                print(f"  {what}: column {c} differs: {got[:4]} vs {want[:4]}")
    assert bad == 0, f"{what}: {bad} columns differ"
    print(f"  {what}: {WIDE} columns x {ROWS} rows exact")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    d, test_bin = sys.argv[1], sys.argv[2]
    pf = pq.ParquetFile(os.path.join(d, "l6_wide_bolt.parquet"))
    assert pf.metadata.num_row_groups == 3, pf.metadata.num_row_groups
    check_table(pf.read(), "pyarrow <- bolt parquet", False)
    with open(os.path.join(d, "l6_wide.arrows"), "rb") as f:
        check_table(ipc.open_stream(f).read_all(), "pyarrow <- bolt arrow ipc", True)

    n = 5000
    table = pa.table({f"p{c}": pa.array([r * c for r in range(40)], pa.int64())
                      for c in range(n)})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "l6_pyarrow_wide.parquet")
        pq.write_table(table, path, row_group_size=16)
        env = dict(os.environ, BOLT_L6_PYARROW_WIDE=path)
        run = subprocess.run([test_bin, "--gtest_filter=L6WideColumns.ReadsPyarrowWideFile"],
                             env=env, capture_output=True, text=True)
        print("  " + "\n  ".join(l for l in run.stdout.splitlines()
                                 if "read pyarrow" in l or "OK" in l or "FAIL" in l))
        assert run.returncode == 0 and "[  PASSED  ] 1 test" in run.stdout, run.stdout[-2000:]
    print("L6 wide columns: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
