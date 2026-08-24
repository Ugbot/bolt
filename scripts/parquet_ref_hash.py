#!/usr/bin/env python3
"""Reference per-column value hash for a parquet file, computed with pyarrow.

The counterpart of benchmarks/bench_parquet_decode's parquet mode: BOTH sides
hash the same canonical byte form, so a disagreement is a decode bug and
nothing rests on float formatting or text rendering.

  8-byte types : the 8 little-endian bytes
  4-byte types : the 4 little-endian bytes
  Utf8/Binary  : 4-byte LE length, then the bytes

Used to verify BYTE_STREAM_SPLIT, the three delta encodings, DATA_PAGE_V2 and
the GZIP/LZ4_RAW codecs against a reference writer -- bolt's own writer cannot
emit any of them, so a round-trip through it proves nothing about interop.

  python3 scripts/parquet_ref_hash.py FILE.parquet
"""
# The same canonical hash, computed from pyarrow. Any disagreement is a decode bug.
import struct, sys
import pyarrow.parquet as pq
def fnv(h, b):
    for x in b:
        h ^= x; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
t = pq.read_table(sys.argv[1])
for c, name in enumerate(t.column_names):
    col = t.column(c).combine_chunks()
    h = 0xcbf29ce484222325
    ty = str(col.type)
    n = 0
    for v in col.to_pylist():
        n += 1
        if ty in ('string','large_string','binary'):
            bs = v.encode() if isinstance(v, str) else v
            h = fnv(h, struct.pack('<I', len(bs))); h = fnv(h, bs)
        elif ty in ('double','float'):
            h = fnv(h, struct.pack('<d', float(v)))       # bolt widens f32 -> f64
        elif ty in ('int64','uint64','timestamp[us]','timestamp[ns]','duration[ns]'):
            h = fnv(h, struct.pack('<q', int(v)))
        elif ty in ('int32','uint32','date32[day]'):
            h = fnv(h, struct.pack('<i', int(v)))
        else:
            print(f"col{c} {name}: UNHANDLED type {ty}"); break
    else:
        print(f"col{c} rows={n} hash={h:016x}   ({name}: {ty})")
