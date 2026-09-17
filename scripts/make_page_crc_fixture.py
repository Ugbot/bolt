#!/usr/bin/env python3
"""Generate PageHeader.crc parquet fixtures (G2PQ-19).

The parquet spec's PageHeader.crc (field 4, optional) is "calculated on the
compressed page data ... used to indicate the corruption of the page" --
see extern/bolt/docs/research/parquet-spec-conformance.md A7. Most writers
(including bolt's own) never emit it, so it must stay optional; but when a
writer DOES emit it, a reader that ignores it will happily decode corrupted
bytes instead of refusing.

pyarrow does not expose a way to opt into writing page CRCs, so -- following
the exact precedent make_legacy_list_fixtures.py already established for
G2PQ-14 -- this hand-constructs a minimal single-column, single-page,
single-row-group file from the thrift compact-protocol spec directly, no
parquet library involved. Two files come out of one page body:

  page_crc_ok.parquet    correct crc for the page bytes -- must read fine
  page_crc_bad.parquet   byte-identical EXCEPT one byte of the page's value
                          data is flipped after the crc was computed, so the
                          stored crc now provably does not match -- must be
                          REFUSED, not silently decoded.

Both files are verified against a real external reader (pyarrow) where
applicable: page_crc_ok.parquet must open in pyarrow with the expected 5
rows; page_crc_bad.parquet is deliberately NOT valid deflate/plain-decodable
garbage in a way that would fool pyarrow either (it's plain data, so pyarrow
--which does not check PageHeader.crc at all as of pyarrow 21-- will happily
return the CORRUPTED values; this is exactly the gap this fixture exists to
prove bolt closes that pyarrow does not).
"""
import os
import struct
import zlib

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")

PHYS_I64 = 2  # parquet Type.INT64


class TC:
    """Minimal thrift compact protocol writer (from the spec, no library)."""
    I32, BINARY, LIST, STRUCT = 5, 8, 9, 12

    def __init__(self):
        self.b = bytearray()
        self.last_fid = [0]

    def _varint(self, v):
        assert v >= 0
        while True:
            if v < 0x80:
                self.b.append(v)
                return
            self.b.append((v & 0x7F) | 0x80)
            v >>= 7

    def _zigzag(self, v):
        self._varint((v << 1) ^ (v >> 63) if v >= 0 else ((-v) * 2 - 1))

    def field(self, fid, ftype):
        delta = fid - self.last_fid[-1]
        if 0 < delta <= 15:
            self.b.append((delta << 4) | ftype)
        else:
            self.b.append(ftype)
            self._zigzag(fid)
        self.last_fid[-1] = fid

    def i32(self, fid, v):
        self.field(fid, self.I32)
        self._zigzag(v)

    def i64(self, fid, v):
        self.field(fid, 6)
        self._zigzag(v)

    def binary(self, fid, data):
        if isinstance(data, str):
            data = data.encode()
        self.field(fid, self.BINARY)
        self._varint(len(data))
        self.b += data

    def list_begin(self, fid, etype, n):
        self.field(fid, self.LIST)
        if n < 15:
            self.b.append((n << 4) | etype)
        else:
            self.b.append(0xF0 | etype)
            self._varint(n)

    def struct_begin(self, fid=None):
        if fid is not None:
            self.field(fid, self.STRUCT)
        self.last_fid.append(0)

    def struct_end(self):
        self.b.append(0)
        self.last_fid.pop()


def schema_element(tc, name, repetition, phys=None, num_children=None):
    tc.struct_begin()
    if phys is not None:
        tc.i32(1, phys)
    tc.i32(3, repetition)
    tc.binary(4, name)
    if num_children is not None:
        tc.i32(5, num_children)
    tc.struct_end()


def page_header_bytes(body, crc):
    """PageHeader{1 type=DATA_PAGE, 2 unc, 3 cmp, [4 crc], 5 data_page_header}.

    codec is UNCOMPRESSED throughout this fixture, so uncompressed_page_size
    == compressed_page_size == len(body), and the crc (when present) is
    computed over exactly `body` -- the same bytes the spec names ("the
    compressed page data").
    """
    hdr = TC()
    hdr.struct_begin()
    hdr.i32(1, 0)              # type = DATA_PAGE
    hdr.i32(2, len(body))      # uncompressed_page_size
    hdr.i32(3, len(body))      # compressed_page_size
    if crc is not None:
        hdr.i32(4, crc)        # G2PQ-19: the field under test
    hdr.struct_begin(5)        # data_page_header
    hdr.i32(1, len(body) // 8)  # num_values (8 bytes/value, INT64 PLAIN)
    hdr.i32(2, 0)               # encoding = PLAIN
    hdr.i32(3, 0)               # definition_level_encoding = unused (required)
    hdr.i32(4, 0)               # repetition_level_encoding = unused (required)
    hdr.struct_end()
    hdr.struct_end()
    return bytes(hdr.b)


def write_file(path, body, crc):
    """One row group, one column ('v', required INT64), one data page."""
    out = bytearray(b"PAR1")
    page = page_header_bytes(body, crc) + body
    offset = len(out)
    out += page
    size = len(page)

    meta = TC()
    meta.struct_begin()
    meta.i32(1, 1)                                   # version
    n_rows = len(body) // 8
    meta.list_begin(2, TC.STRUCT, 2)                  # schema: root + 1 leaf
    schema_element(meta, "root", 0, num_children=1)   # repetition unused @root
    schema_element(meta, "v", 0, phys=PHYS_I64)       # 0 = REQUIRED
    meta.i64(3, n_rows)
    meta.list_begin(4, TC.STRUCT, 1)                  # row_groups
    meta.struct_begin()
    meta.list_begin(1, TC.STRUCT, 1)                  # columns
    meta.struct_begin()
    meta.i64(2, offset)                               # file_offset
    meta.struct_begin(3)                              # ColumnMetaData
    meta.i32(1, PHYS_I64)
    meta.list_begin(2, TC.I32, 1)
    meta._zigzag(0)                                   # encodings = [PLAIN]
    meta.list_begin(3, TC.BINARY, 1)
    meta._varint(1)
    meta.b += b"v"
    meta.i32(4, 0)                                    # codec = UNCOMPRESSED
    meta.i64(5, n_rows)                                # num_values
    meta.i64(6, size)                                  # total_uncompressed_size
    meta.i64(7, size)                                  # total_compressed_size
    meta.i64(9, offset)                                # data_page_offset
    meta.struct_end()
    meta.struct_end()
    meta.i64(2, size)                                  # total_byte_size
    meta.i64(3, n_rows)
    meta.struct_end()
    meta.struct_end()
    md = bytes(meta.b)
    out += md
    out += struct.pack("<I", len(md))
    out += b"PAR1"
    with open(path, "wb") as f:
        f.write(out)
    return offset, len(page)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    values = [10, 20, 30, 40, 50]
    body = b"".join(struct.pack("<q", v) for v in values)
    correct_crc = zlib.crc32(body) & 0xFFFFFFFF
    # bolt's crc32_ieee(data,len,0) is byte-for-byte the conventional CRC-32
    # (init 0xFFFFFFFF, final XOR 0xFFFFFFFF) -- identical to zlib.crc32.

    ok_path = os.path.join(OUT_DIR, "page_crc_ok.parquet")
    write_file(ok_path, body, correct_crc)

    # Corrupt ONE byte of the page's value data after the crc was already
    # computed from the original bytes -- the stored crc now provably does
    # not match what's actually on disk, exactly the "bit rot in transit"
    # scenario PageHeader.crc exists to catch.
    corrupted = bytearray(body)
    corrupted[0] ^= 0xFF
    bad_path = os.path.join(OUT_DIR, "page_crc_bad.parquet")
    write_file(bad_path, bytes(corrupted), correct_crc)

    # A file with NO crc field at all -- the overwhelmingly common case
    # (parquet-mr's default writer, bolt's own writer) -- must be completely
    # unaffected by this feature.
    none_path = os.path.join(OUT_DIR, "page_crc_absent.parquet")
    write_file(none_path, body, None)

    print(f"wrote {ok_path} (crc=0x{correct_crc:08x}, values={values})")
    print(f"wrote {bad_path} (stored crc=0x{correct_crc:08x}, "
          f"real crc=0x{zlib.crc32(bytes(corrupted)) & 0xFFFFFFFF:08x} -- "
          f"MISMATCH by construction)")
    print(f"wrote {none_path} (no crc field)")


def verify(inject=False):
    """Self-check: pyarrow must read page_crc_ok.parquet's real values, and
    (the point of `inject`) must NOT notice page_crc_bad.parquet's corruption
    on its own -- proving this fixture's discriminating power depends on
    bolt's own crc check, not on the corruption being independently obvious.
    """
    import pyarrow.parquet as pq
    ok_path = os.path.join(OUT_DIR, "page_crc_ok.parquet")
    got = pq.read_table(ok_path).column("v").to_pylist()
    want = [10, 20, 30, 40, 50]
    if inject:
        want = [999]  # deliberately wrong, to prove this check can fail
    if got != want:
        print(f"FAIL: page_crc_ok.parquet got {got}, want {want}")
        return False
    print(f"OK: page_crc_ok.parquet read back {got} via pyarrow")

    bad_path = os.path.join(OUT_DIR, "page_crc_bad.parquet")
    bad_got = pq.read_table(bad_path).column("v").to_pylist()
    print(f"OK: pyarrow (which ignores PageHeader.crc) reads "
          f"page_crc_bad.parquet as {bad_got} without complaint -- "
          f"confirms bolt's crc check is the only thing that can catch this")
    return True


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        ok = verify(inject="--inject" in sys.argv)
        sys.exit(0 if ok else 1)
    main()
