#!/usr/bin/env python3
"""Hand bolt's DEFLATE and GZIP output to python's zlib.

This is the gate that actually settles the compressor. bolt inflating its own
deflate output proves only that the encoder and decoder share a reading of the
format -- and DEFLATE is unusually exposed to that failure, because bits go
into bytes LSB-first while a Huffman CODE is transmitted MSB-first. An encoder
that gets both wrong in the same direction as its decoder round-trips
perfectly and produces a stream nothing else can read.

Usage:

    ./test_bolt_deflate_dump <file>        # written by the C++ side
    python3 scripts/deflate_zlib_check.py <file>

or, more usually, via the test target which does both.

Each record in the dump is:

    kind u32, raw_len u32, deflate_len u32, gzip_len u32
    raw bytes, deflate bytes, gzip bytes

Both encodings of every input are checked: the raw stream with zlib's
-15 window (no wrapper) and the gzip container with 16+15.
"""
import struct
import sys
import zlib


def main():
    if len(sys.argv) < 2:
        print("usage: deflate_zlib_check.py <dump-file>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "rb") as fh:
        f = fh.read()
    off = 0
    ok = 0
    fails = []
    while off < len(f):
        if off + 16 > len(f):
            fails.append(("truncated header", off))
            break
        kind, n, rn, gn = struct.unpack_from("<IIII", f, off)
        off += 16
        raw = f[off:off + n]; off += n
        deflated = f[off:off + rn]; off += rn
        gzipped = f[off:off + gn]; off += gn
        if len(raw) != n or len(deflated) != rn or len(gzipped) != gn:
            fails.append(("truncated body", kind, n))
            break
        try:
            got = zlib.decompress(deflated, -15)        # raw DEFLATE
        except Exception as e:                          # noqa: BLE001
            fails.append(("zlib rejected deflate", kind, n, str(e)))
            continue
        if got != raw:
            fails.append(("deflate decoded to wrong bytes", kind, n))
            continue
        try:
            got2 = zlib.decompress(gzipped, 16 + 15)    # gzip container
        except Exception as e:                          # noqa: BLE001
            fails.append(("zlib rejected gzip", kind, n, str(e)))
            continue
        if got2 != raw:
            fails.append(("gzip decoded to wrong bytes", kind, n))
            continue
        ok += 1
    if fails:
        print("FAIL (%d):" % len(fails))
        for x in fails[:20]:
            print("  ", x)
        return 1
    print("OK: python zlib accepted %d bolt streams "
          "(raw DEFLATE and gzip container)" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
