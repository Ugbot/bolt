"""Generate tests/data/zstd_vectors.bin -- byte-exact reference vectors for
bolt's self-contained zstd decoder (G2FEAT-134).

WHY THE PATTERN INDIRECTION: the reference OUTPUT is the big half of a codec
fixture (megabytes). Rather than commit it, most vectors describe their raw
bytes as a deterministic PATTERN that tests/test_bolt_zstd_dec.cpp regenerates
itself; only the COMPRESSED bytes are committed. The comparison is still
byte-exact against what python-zstandard produced -- the compressed bytes came
from compressing exactly that pattern -- but the fixture stays ~100 KB instead
of ~6 MB. A few real parquet pages, which have no closed form, store their raw
bytes inline (they are small).

The pattern PRNG is a plain 64-bit LCG so C++ reproduces it exactly; do not
change the constants without regenerating and updating the C++ side.

COVERAGE (verified by the printout at the end -- every zstd block type,
literal mode and sequence mode a real writer emits must appear):
  block    : Raw, RLE, Compressed
  literals : Raw, RLE, Compressed (Huffman) 1-stream and 4-stream, Treeless
  sequences: Predefined, RLE, FSE_Compressed  (+ Repeat when the writer picks it)

Run from this directory:  python make_zstd_vectors.py
"""
import os
import struct
import sys

import zstandard as zstd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "zstd_vectors.bin")

# kinds -- must match the enum in tests/test_bolt_zstd_dec.cpp
K_INLINE, K_ZEROS, K_CONST, K_REPEAT, K_LCG, K_ALPHA, K_CHUNKS = range(7)

MUL = 6364136223846793005
INC = 1442695040888963407
M64 = (1 << 64) - 1


def lcg_bytes(n, seed):
    out = bytearray(n)
    s = seed & M64
    for i in range(n):
        s = (s * MUL + INC) & M64
        out[i] = (s >> 33) & 0xFF
    return bytes(out)


def alpha_bytes(n, seed, k):
    out = bytearray(n)
    s = seed & M64
    for i in range(n):
        s = (s * MUL + INC) & M64
        out[i] = ord('a') + ((s >> 33) % k)
    return bytes(out)


def repeat_bytes(n, pat):
    if not pat:
        return b""
    reps = n // len(pat) + 1
    return (pat * reps)[:n]


def chunk_bytes(seed, k, chunk_len, reps, lit):
    """A dictionary of `k` random chunks, then `reps` records of
    (one constant literal byte + one whole chunk picked pseudo-randomly).

    This shape is what makes zstd emit the rare modes: every literal is the
    same byte (-> RLE literals) and every sequence has the same literal length
    and match length (-> RLE sequence tables), while the offsets still vary.
    A plain periodic input will NOT produce them -- zstd collapses it to a
    single long match."""
    chunks = [lcg_bytes(chunk_len, (seed + 1 + i) & M64) for i in range(k)]
    out = bytearray()
    for c in chunks:
        out += c
    s = seed & M64
    for _ in range(reps):
        s = (s * MUL + INC) & M64
        out += bytes([lit]) + chunks[(s >> 33) % k]
    return bytes(out)


def unpack_chunk_pat(pat):
    cl, reps = struct.unpack_from("<II", pat, 0)
    return cl, reps, pat[8]


def materialize(kind, raw_len, seed, aux, pat):
    if kind == K_ZEROS:
        return b"\x00" * raw_len
    if kind == K_CONST:
        return bytes([seed & 0xFF]) * raw_len
    if kind == K_REPEAT:
        return repeat_bytes(raw_len, pat)
    if kind == K_LCG:
        return lcg_bytes(raw_len, seed)
    if kind == K_ALPHA:
        return alpha_bytes(raw_len, seed, aux)
    if kind == K_CHUNKS:
        cl, reps, lit = unpack_chunk_pat(pat)
        return chunk_bytes(seed, aux, cl, reps, lit)
    raise ValueError(kind)


VEC = []


def add(name, kind, raw_len, seed=0, aux=0, pat=b"", **kw):
    raw = materialize(kind, raw_len, seed, aux, pat)
    if kind == K_CHUNKS:
        raw_len = len(raw)            # derived, not caller-specified
    assert len(raw) == raw_len, f"{name}: {len(raw)} != {raw_len}"
    c = zstd.ZstdCompressor(**kw).compress(raw)
    VEC.append((name, kind, raw_len, seed, aux, pat, b"", c))
    return raw


def add_inline(name, raw, comp):
    VEC.append((name, K_INLINE, len(raw), 0, 0, b"", raw, comp))


def add_raw_frames(name, raw, comp):
    """A vector whose compressed side is caller-supplied (multi-frame etc.)."""
    VEC.append((name, K_INLINE, len(raw), 0, 0, b"", raw, comp))


# --- degenerate / boundary --------------------------------------------------
add("empty", K_ZEROS, 0)
add("one_byte", K_CONST, 1, seed=ord('A'))
add("const_small", K_CONST, 300, seed=ord('z'))
add("zeros_100k", K_ZEROS, 100000)                  # long match / RLE
add("const_64k", K_CONST, 65536, seed=0x5A)         # RLE literals

# --- RLE blocks: a whole block of one repeated byte --------------------------
add("zeros_300k", K_ZEROS, 300000)
add("const_200k", K_CONST, 200000, seed=ord('A'))

# --- RLE literals + RLE sequence tables (see chunk_bytes' comment) ----------
_CH = struct.pack("<II", 64, 8000) + bytes([ord('a')])
# seed 20260805 @ level 19 is the config that makes zstd pick RLE LITERALS --
# it is order-sensitive, so do not retune the seed without re-checking that
# 'RLE' still appears in the literal-mode coverage printout.
add("chunks_rle_l19", K_CHUNKS, 0, seed=20260805, aux=256, pat=_CH, level=19)
_CH2 = struct.pack("<II", 32, 8000) + bytes([ord('a')])
add("chunks_rle_l1", K_CHUNKS, 0, seed=99, aux=16, pat=_CH2, level=1)

# --- Repeat sequence mode (a block reusing the previous block's FSE table) ---
add("alpha2_400k", K_ALPHA, 400000, seed=3, aux=2, level=9)

# --- incompressible: Raw blocks / Raw literals ------------------------------
add("lcg_40k", K_LCG, 40000, seed=12345)          # Raw block / Raw literals

# --- Huffman literals -------------------------------------------------------
add("alpha16_512", K_ALPHA, 512, seed=7, aux=16)    # small -> 1-stream Huffman
add("alpha16_32k", K_ALPHA, 32768, seed=7, aux=16)  # 4-stream Huffman + FSE
add("alpha8_60k", K_ALPHA, 60000, seed=5, aux=8)        # skewed weights
add("alpha4_48k", K_ALPHA, 48000, seed=77, aux=4)
add("alpha16_32k_l19", K_ALPHA, 32768, seed=7, aux=16, level=19)
add("alpha16_32k_l1", K_ALPHA, 32768, seed=7, aux=16, level=1)

# --- text: Raw literals + predefined sequences ------------------------------
TXT = b"the quick brown fox jumps over the lazy dog. "
add("text_135", K_REPEAT, 135, pat=TXT)
add("text_225k", K_REPEAT, 225000, pat=TXT)
add("text_4k", K_REPEAT, 4096, pat=TXT)

# --- frame options ----------------------------------------------------------
add("cksum", K_REPEAT, 40000, pat=TXT, write_checksum=True)
add("with_size", K_REPEAT, 28000, pat=TXT, write_content_size=True)
add("no_size", K_REPEAT, 32000, pat=TXT, write_content_size=False)

# --- repeat offsets / periodic ---------------------------------------------
add("periodic_256", K_REPEAT, 512000, pat=bytes(range(256)))
add("periodic_3", K_REPEAT, 90000, pat=b"abc")

# --- multi-frame (two concatenated frames) ---------------------------------
_r1 = repeat_bytes(20000, b"frame one ")
_r2 = repeat_bytes(20000, b"frame two ")
add_raw_frames(
    "multi_frame", _r1 + _r2,
    zstd.ZstdCompressor().compress(_r1)
    + zstd.ZstdCompressor(write_checksum=True).compress(_r2))

# --- skippable frame before real data --------------------------------------
_r3 = repeat_bytes(5000, TXT)
_skip = struct.pack("<II", 0x184D2A50, 8) + b"SKIPPED!"
add_raw_frames("skippable_then_data", _r3,
               _skip + zstd.ZstdCompressor().compress(_r3))

# --- REAL parquet pages (no closed form -> raw stored inline) --------------
try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(20260805)
    tbl = pa.table({
        "i": pa.array(rng.integers(0, 1 << 20, 4000), type=pa.int64()),
        "f": pa.array(rng.random(4000)),
        "s": pa.array([f"sym{int(v)%97:03d}" for v in rng.integers(0, 1000, 4000)]),
    })
    tmp = os.path.join(HERE, "_tmp_zstd_pages.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    blob = open(tmp, "rb").read()
    md = pq.ParquetFile(tmp).metadata
    dctx = zstd.ZstdDecompressor()
    got = 0
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            cc = md.row_group(rg).column(c)
            off = cc.dictionary_page_offset or cc.data_page_offset
            region = blob[off:off + cc.total_compressed_size]
            i = 0
            while got < 8:
                j = region.find(b"\x28\xb5\x2f\xfd", i)
                if j < 0:
                    break
                tail = region[j:]
                try:
                    ob = dctx.decompressobj()
                    raw = ob.decompress(tail)
                    flen = len(tail) - len(ob.unused_data)
                except Exception:
                    i = j + 4
                    continue
                if flen <= 0 or not raw or len(raw) > 200000:
                    i = j + max(flen, 4)
                    continue
                add_inline(f"pq_page_{got}", raw, tail[:flen])
                got += 1
                i = j + flen
    os.remove(tmp)
    print(f"captured {got} real parquet pages")
except ImportError:
    print("pyarrow/numpy absent -- skipping real parquet page vectors",
          file=sys.stderr)

# --- golden_zstd.parquet: a REAL zstd parquet file, end to end -------------
# Values are closed-form so tests/test_bolt_zstd_dec.cpp can assert every one
# without committing an expectation blob. Before G2FEAT-134 bolt's parquet
# reader rejected this file outright (PqCodec::Zstd hit the `return false`
# default arm), so the test that reads it fails on the old code by
# construction -- that is the point of it.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    N = 5000
    ids = list(range(N))
    vs = [(i * 2654435761) % 1000003 for i in range(N)]
    fs = [i * 0.25 for i in range(N)]
    ss = [f"row{i:05d}" for i in range(N)]
    gt = pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "v": pa.array(vs, type=pa.int64()),
        "f": pa.array(fs, type=pa.float64()),
        "s": pa.array(ss, type=pa.string()),
    })
    gpath = os.path.join(HERE, "golden_zstd.parquet")
    pq.write_table(gt, gpath, compression="zstd", row_group_size=1500)
    print(f"wrote {gpath} ({os.path.getsize(gpath)} bytes, {N} rows, "
          f"{pq.ParquetFile(gpath).metadata.num_row_groups} row groups)")
except ImportError:
    print("pyarrow absent -- skipping golden_zstd.parquet", file=sys.stderr)

# --- write ------------------------------------------------------------------
with open(OUT, "wb") as f:
    f.write(struct.pack("<I", len(VEC)))
    for (name, kind, raw_len, seed, aux, pat, inline, comp) in VEC:
        nb = name.encode()
        f.write(struct.pack("<I", len(nb))); f.write(nb)
        f.write(struct.pack("<I", kind))
        f.write(struct.pack("<Q", raw_len))
        f.write(struct.pack("<Q", seed))
        f.write(struct.pack("<I", aux))
        f.write(struct.pack("<I", len(pat))); f.write(pat)
        f.write(struct.pack("<Q", len(inline))); f.write(inline)
        f.write(struct.pack("<Q", len(comp))); f.write(comp)

size = os.path.getsize(OUT)
print(f"wrote {len(VEC)} vectors -> {OUT}  ({size} bytes)")
