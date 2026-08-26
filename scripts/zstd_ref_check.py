#!/usr/bin/env python3
"""Hand bolt's zstd frames to the REFERENCE libzstd.

This is the only evidence that matters for a zstd encoder. bolt has its own
zstd decoder, but an encoder and decoder written from the same reading of the
spec agree with each other whether or not that reading is right -- and zstd's
sequence section is FSE entropy coding written in reverse into a stream the
decoder walks backwards, which is not a place to trust self-consistency.

libzstd is loaded through ctypes, so bolt's build gains no dependency: the
reference is needed to CHECK the frames, never to produce or consume them in
production.

    ./test_bolt_zstd_enc_dump <file>
    python3 scripts/zstd_ref_check.py <file>

Record format (little-endian):
    kind u32, raw_len u32, comp_len u32, raw bytes, comp bytes
"""
import ctypes
import struct
import sys

CANDIDATES = [
    "/opt/homebrew/lib/libzstd.dylib",
    "/usr/local/lib/libzstd.dylib",
    "/usr/lib/x86_64-linux-gnu/libzstd.so.1",
    "libzstd.so.1",
]


def load():
    for c in CANDIDATES:
        try:
            lib = ctypes.CDLL(c)
        except OSError:
            continue
        lib.ZSTD_decompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.c_char_p, ctypes.c_size_t]
        lib.ZSTD_decompress.restype = ctypes.c_size_t
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        return lib
    return None


def main():
    if len(sys.argv) < 2:
        print("usage: zstd_ref_check.py <dump-file>", file=sys.stderr)
        return 2
    lib = load()
    if lib is None:
        print("libzstd not found; cannot verify", file=sys.stderr)
        return 2
    blob = open(sys.argv[1], "rb").read()
    off = 0
    ok = 0
    traw = 0
    tcmp = 0
    fails = []
    while off < len(blob):
        kind, n, cn = struct.unpack_from("<III", blob, off)
        off += 12
        raw = blob[off:off + n]; off += n
        comp = blob[off:off + cn]; off += cn
        buf = ctypes.create_string_buffer(max(n, 1))
        r = lib.ZSTD_decompress(buf, max(n, 1), comp, cn)
        if lib.ZSTD_isError(r):
            fails.append((kind, n, lib.ZSTD_getErrorName(r).decode()))
            continue
        if r != n or buf.raw[:n] != raw:
            fails.append((kind, n, "decoded to wrong bytes"))
            continue
        ok += 1
        traw += n
        tcmp += cn
    if fails:
        print("FAIL (%d):" % len(fails))
        for f in fails[:20]:
            print("   kind=%d n=%d: %s" % f)
        return 1
    ratio = (traw / tcmp) if tcmp else 0.0
    print("OK: libzstd accepted %d bolt frames, all byte-exact" % ok)
    print("    %d raw -> %d compressed (%.2fx)" % (traw, tcmp, ratio))
    # An encoder that emitted only Raw blocks would round-trip perfectly and
    # compress nothing -- the exact shape of the snappy bug found earlier.
    if ratio < 1.5:
        print("FAIL: ratio %.2fx means sequences are not being emitted" % ratio)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
