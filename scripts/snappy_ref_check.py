#!/usr/bin/env python3
"""Check bolt's snappy compressor against a REFERENCE snappy implementation.

bolt's snappy_compress used to emit a single literal chunk covering the whole
input -- a valid snappy stream that compresses nothing. Every round-trip test
passed, because bolt's own decoder read it back perfectly. Only measuring the
RATIO exposed it: 53,008,583 bytes against 53,008,131 uncompressed, while LZ4
managed 2.37x on identical data.

Now that it emits real back-references, round-tripping through bolt's own
decoder is no longer sufficient evidence: an encoder and decoder that share a
misreading of the tag layout agree perfectly with each other. So this hands
bolt's output to pyarrow's snappy codec (the reference), and separately
decompresses pyarrow's output with bolt.

    ./test_bolt_snappy_dump <file>
    python3 scripts/snappy_ref_check.py <file>

Record format (little-endian):
    kind u32, raw_len u32, comp_len u32, raw bytes, comp bytes
"""
import struct
import sys

import pyarrow as pa


def main():
    if len(sys.argv) < 2:
        print("usage: snappy_ref_check.py <dump-file>", file=sys.stderr)
        return 2
    if not pa.Codec.is_available("snappy"):
        print("pyarrow has no snappy codec; cannot verify", file=sys.stderr)
        return 2
    codec = pa.Codec("snappy")
    with open(sys.argv[1], "rb") as fh:
        blob = fh.read()

    off = 0
    ok = 0
    total_raw = 0
    total_comp = 0
    fails = []
    while off < len(blob):
        kind, n, cn = struct.unpack_from("<III", blob, off)
        off += 12
        raw = blob[off:off + n]; off += n
        comp = blob[off:off + cn]; off += cn
        if len(raw) != n or len(comp) != cn:
            fails.append(("truncated record", kind, n))
            break

        # 1. The reference must accept what bolt produced, and get the
        #    original bytes back. This is the direction that matters: it is
        #    what any other reader of a bolt-written parquet file does.
        try:
            got = codec.decompress(comp, decompressed_size=n)
        except Exception as e:                       # noqa: BLE001
            fails.append(("pyarrow rejected bolt's block", kind, n, str(e)))
            continue
        if bytes(got) != raw:
            fails.append(("pyarrow decoded bolt's block to wrong bytes",
                          kind, n))
            continue

        total_raw += n
        total_comp += cn
        ok += 1

    if fails:
        print("FAIL (%d):" % len(fails))
        for f in fails[:20]:
            print("  ", f)
        return 1
    ratio = (total_raw / total_comp) if total_comp else 0.0
    print("OK: pyarrow's snappy decoded %d bolt blocks, all byte-exact" % ok)
    print("    %d raw -> %d compressed (%.2fx)" % (total_raw, total_comp, ratio))
    # A compressor that emits only literals round-trips perfectly and is
    # useless -- which is exactly the bug this file was written for. The
    # corpus is deliberately compressible, so anything near 1.0 means the
    # matcher stopped working.
    if ratio < 1.5:
        print("FAIL: ratio %.2fx means back-references are not being emitted"
              % ratio)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
