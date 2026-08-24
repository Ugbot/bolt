#!/usr/bin/env python3
"""Regenerate the LIST/MAP parquet fixtures for test_bolt_parquet_list.

Written by pyarrow deliberately: a fixture bolt produced could only prove
bolt self-consistent, and the whole risk with Dremel levels is a shared
misreading of the spec. The C++ test restates these same closed forms and
compares against them, so neither side reads its expectation out of the file.

    python3 scripts/make_list_fixtures.py

Shapes covered, and why each is here:
  * a NULL list and an EMPTY list  -- different values that a null bitmap
    alone cannot tell apart; they differ only in a definition level
  * NULL elements inside a present list -- a third case sharing a level with
    "list present"
  * strings both under and over the 12-byte StringView inline limit
  * a MAP, whose key and value leaves must agree on row boundaries
  * a flat column in the same file, which must keep reading normally
"""
import pyarrow as pa
import pyarrow.parquet as pq

N = 500


def build():
    ints, strs = [], []
    for i in range(N):
        if i % 17 == 0:                       # NULL list
            ints.append(None)
            strs.append(None)
        elif i % 7 == 0:                      # EMPTY list
            ints.append([])
            strs.append([])
        else:
            k = (i % 5) + 1
            ints.append([i * 10 + j for j in range(k)])
            strs.append([("s%d-%d" % (i, j)) if (i + j) % 11 else None
                         for j in range(k)])
    maps = [None if i % 23 == 0
            else [(("k%d" % j), i * 100 + j) for j in range(i % 4)]
            for i in range(N)]
    return pa.table({
        "li": pa.array(ints, pa.list_(pa.int64())),
        "ls": pa.array(strs, pa.list_(pa.string())),
        "mp": pa.array(maps, pa.map_(pa.string(), pa.int64())),
        "flat": pa.array(list(range(N)), pa.int64()),
    })


def main():
    t = build()
    # Two encodings, so two independent decode paths are exercised.
    pq.write_table(t, "tests/data/golden_list.parquet", compression="snappy",
                   version="2.6", use_dictionary=False)
    pq.write_table(t, "tests/data/golden_list_dict.parquet",
                   compression="snappy", version="1.0", use_dictionary=True)
    print("wrote tests/data/golden_list.parquet + golden_list_dict.parquet")


if __name__ == "__main__":
    main()
