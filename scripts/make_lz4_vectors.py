#!/usr/bin/env python3
"""Emit LZ4 block-format test vectors produced by the REFERENCE liblz4.

bolt's decoder agreeing with bolt's compressor proves only that the two share
a reading of the format -- which is exactly the failure worth ruling out for a
codec, and the reason this file exists rather than a round-trip test alone.
These vectors are compressed by liblz4 itself, so decoding them is a genuine
interoperability claim.

    python3 scripts/make_lz4_vectors.py     # writes tests/data/lz4_vectors.bin

liblz4 is loaded through ctypes rather than linked, so nothing in bolt's build
gains a dependency: the reference is needed to GENERATE the committed vectors,
never to consume them.

File format (little-endian). The raw input is re-derived on the C++ side from
its seed rather than stored -- committing ~1 MB of random bytes to prove a
codec works is a poor trade, and a generator both sides implement is also a
check that they agree about the input:

    magic   "BLZ4"                     4 bytes
    count   u32                        number of vectors
    then `count` records:
        kind      u32    input generator id (see gen() below)
        raw_len   u32    length of the ORIGINAL input
        seed      u32    generator seed
        comp_len  u32    length of the liblz4-compressed block
        comp      bytes  comp_len bytes
"""
import ctypes
import os
import struct
import sys

CANDIDATES = [
    "/opt/homebrew/lib/liblz4.dylib",
    "/usr/local/lib/liblz4.dylib",
    "/usr/lib/x86_64-linux-gnu/liblz4.so.1",
    "liblz4.so.1",
]


def load_lz4():
    for c in CANDIDATES:
        try:
            lib = ctypes.CDLL(c)
        except OSError:
            continue
        lib.LZ4_compress_default.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.c_int, ctypes.c_int]
        lib.LZ4_compress_default.restype = ctypes.c_int
        lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        lib.LZ4_compressBound.restype = ctypes.c_int
        return lib
    return None


class Rng:
    """xorshift64, bit-identical to the C++ side's Rng."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = s
        return (s >> 32) & 0xFFFFFFFF


def gen(kind, n, seed):
    """Must match gen_vector() in tests/test_bolt_lz4_raw.cpp exactly."""
    r = Rng(seed)
    out = bytearray(n)
    if kind == 0:                     # incompressible
        for i in range(n):
            out[i] = r.next() & 0xFF
    elif kind == 1:                   # one long run
        for i in range(n):
            out[i] = 0x5A
    elif kind == 2:                   # short period -> overlapping matches
        for i in range(n):
            out[i] = i % 3
    elif kind == 3:                   # repeated 64-byte blocks
        for i in range(n):
            out[i] = (i // 64) % 7
    elif kind == 4:                   # low-entropy text
        for i in range(n):
            out[i] = ord('a') + (r.next() % 6)
    else:                             # mostly one byte, rare noise
        for i in range(n):
            a = r.next()
            b = r.next()
            out[i] = (b & 0xFF) if (a % 32 == 0) else 0xC3
    return bytes(out)


# Sizes straddle the format's two end-of-block rules (the final 5 bytes are
# always literals; the last match starts >= 12 bytes from the end) and the
# all-literals short-block path.
SIZES = [0, 1, 4, 5, 12, 13, 17, 64, 255, 256, 1023, 4096, 65535, 70000]


def main():
    lib = load_lz4()
    if lib is None:
        print("liblz4 not found; cannot produce reference vectors",
              file=sys.stderr)
        return 1
    recs = []
    for kind in range(6):
        for n in SIZES:
            seed = (0x9E3779B9 ^ (kind * 1000003) ^ n) & 0xFFFFFFFF
            raw = gen(kind, n, seed)
            cap = lib.LZ4_compressBound(max(n, 1))
            buf = ctypes.create_string_buffer(cap)
            got = lib.LZ4_compress_default(raw, buf, n, cap)
            if got <= 0 and n != 0:
                print("liblz4 refused kind=%d n=%d" % (kind, n), file=sys.stderr)
                return 1
            recs.append((kind, n, seed, buf.raw[:got]))
    os.makedirs("tests/data", exist_ok=True)
    with open("tests/data/lz4_vectors.bin", "wb") as f:
        f.write(b"BLZ4")
        f.write(struct.pack("<I", len(recs)))
        for kind, n, seed, comp in recs:
            f.write(struct.pack("<IIII", kind, n, seed, len(comp)))
            f.write(comp)
    total = sum(len(c) for _, _, _, c in recs)
    print("wrote tests/data/lz4_vectors.bin: %d vectors, %d compressed bytes"
          % (len(recs), total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
