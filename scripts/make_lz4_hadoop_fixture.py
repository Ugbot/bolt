#!/usr/bin/env python3
"""Generate legacy Hadoop-framed-LZ4 parquet fixtures (G2PQ-18).

parquet's CompressionCodec.LZ4 (value 5) is DEPRECATED in favor of LZ4_RAW
(value 7, added later) but still legal to read -- see
extern/bolt/docs/research/parquet-spec-conformance.md A6. It wraps parquet-mr's
Hadoop Lz4Codec framing: the page payload is one or more sub-blocks, each
prefixed with [4B big-endian decompressedSize][4B big-endian compressedSize],
followed by a BARE LZ4 block (NOT the LZ4 frame format) for that sub-block.

Neither pyarrow nor DuckDB write this deprecated codec (checked first-hand:
pyarrow.parquet.write_table's `compression` enum has no way to select it --
passing "LZ4" or "LZ4_RAW" both land on LZ4_RAW under the hood, verified by
decoding the written file's own ColumnMetaData.codec field, see
`verify_no_tool_emits_legacy_lz4()` below), so -- following the exact
precedent make_page_crc_fixture.py established for G2PQ-19 -- this
hand-constructs the parquet footer from the thrift compact-protocol spec
directly, no parquet library involved. The page payload's LZ4 sub-blocks are
produced by Python's own `lz4.block` module (liblz4 under the hood, `pip
install lz4`) with `store_size=False`, i.e. the SAME bare-block format
LZ4_RAW/bolt's own lz4_raw_decompress already speaks and that
test_bolt_lz4_raw.cpp already validates against a liblz4 reference -- so this
script only has to hand-add the Hadoop 8-byte-per-block length prefix, not
reimplement LZ4 itself.

Two files:
  lz4_hadoop_single.parquet   one sub-block (5 int64 values, matches the
                               page_crc fixture's value set for an easy
                               independent eyeball check)
  lz4_hadoop_multi.parquet    two sub-blocks (200 int64 values split down the
                               middle), proving the reader's per-page framing
                               loop actually iterates instead of only handling
                               the single-frame case
"""
import os
import struct

import lz4.block

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")

PHYS_I64 = 2  # parquet Type.INT64
CODEC_LZ4 = 5  # parquet CompressionCodec.LZ4 (deprecated, Hadoop-framed)


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


def hadoop_frame(chunk):
    """One Hadoop Lz4Codec sub-block: [4B BE unc][4B BE cmp][bare LZ4 block]."""
    comp = lz4.block.compress(chunk, store_size=False)
    return struct.pack(">II", len(chunk), len(comp)) + comp


def page_header_bytes(unc_size, cmp_size, num_values):
    """PageHeader{1 type=DATA_PAGE, 2 unc, 3 cmp, 5 data_page_header}."""
    hdr = TC()
    hdr.struct_begin()
    hdr.i32(1, 0)               # type = DATA_PAGE
    hdr.i32(2, unc_size)        # uncompressed_page_size
    hdr.i32(3, cmp_size)        # compressed_page_size
    hdr.struct_begin(5)         # data_page_header
    hdr.i32(1, num_values)
    hdr.i32(2, 0)                # encoding = PLAIN
    hdr.i32(3, 0)                # definition_level_encoding (unused, required)
    hdr.i32(4, 0)                # repetition_level_encoding (unused, required)
    hdr.struct_end()
    hdr.struct_end()
    return bytes(hdr.b)


def write_file(path, body, frames):
    """One row group, one column ('v', required INT64), one data page.

    `body` is the plain (uncompressed) column bytes -- used only to derive
    num_values here, never written to disk. `frames` is the already Hadoop-
    framed, LZ4-block-compressed page payload that actually goes on disk.
    """
    n_rows = len(body) // 8
    page_body = b"".join(frames)
    hdr = page_header_bytes(len(body), len(page_body), n_rows)

    out = bytearray(b"PAR1")
    offset = len(out)
    out += hdr
    out += page_body
    size = len(out) - offset

    meta = TC()
    meta.struct_begin()
    meta.i32(1, 1)                                    # version
    meta.i64(3, n_rows)
    meta.list_begin(2, TC.STRUCT, 2)                   # schema: root + 1 leaf
    schema_element(meta, "root", 0, num_children=1)
    schema_element(meta, "v", 0, phys=PHYS_I64)        # 0 = REQUIRED
    meta.list_begin(4, TC.STRUCT, 1)                   # row_groups
    meta.struct_begin()
    meta.list_begin(1, TC.STRUCT, 1)                   # columns
    meta.struct_begin()
    meta.i64(2, offset)                                # file_offset
    meta.struct_begin(3)                               # ColumnMetaData
    meta.i32(1, PHYS_I64)
    meta.list_begin(2, TC.I32, 1)
    meta._zigzag(0)                                    # encodings = [PLAIN]
    meta.list_begin(3, TC.BINARY, 1)
    meta._varint(1)
    meta.b += b"v"
    meta.i32(4, CODEC_LZ4)                             # codec = LZ4 (5, legacy)
    meta.i64(5, n_rows)                                 # num_values
    # NOTE: the spec says these two exclude page headers, but arrow-cpp's
    # reader uses total_compressed_size to size the buffered read for the
    # WHOLE column chunk starting at data_page_offset (which points at the
    # first page's HEADER, not its body) -- excluding the header here made
    # arrow's SerializedPageReader read the header, then come up 17 bytes
    # short on the body ("Page was smaller (18) than expected (35)"), fixed
    # by including it, matching make_page_crc_fixture.py's precedent.
    meta.i64(6, size)                                   # total_uncompressed_size
    meta.i64(7, size)                                   # total_compressed_size
    meta.i64(9, offset)                                 # data_page_offset
    meta.struct_end()
    meta.struct_end()
    meta.i64(2, size)                                   # total_byte_size
    meta.i64(3, n_rows)
    meta.struct_end()
    meta.struct_end()
    md = bytes(meta.b)
    out += md
    out += struct.pack("<I", len(md))
    out += b"PAR1"
    with open(path, "wb") as f:
        f.write(out)


def verify_no_tool_emits_legacy_lz4():
    """Confirm pyarrow's `compression="LZ4"` actually writes LZ4_RAW (7), not
    the deprecated Hadoop-framed LZ4 (5), by decoding the real codec enum out
    of the file it writes -- not by trusting the option name (arrow C++'s own
    parquet/thrift_internal.h ToThrift() maps arrow's generic Compression::LZ4
    to thrift CompressionCodec::LZ4_RAW=7, and reserves thrift
    CompressionCodec::LZ4=5 for a SEPARATE arrow-level enumerator,
    Compression::LZ4_HADOOP, that pyarrow's parquet `compression=` kwarg has
    no string spelling for -- confirmed by reading arrow's own source, then
    proven here against the actual on-disk bytes rather than trusted).

    A generic (not fixture-shape-specific) thrift compact-protocol struct
    walker over the real footer, printing every field it visits, so the
    codec value comes from structurally locating ColumnMetaData.codec (field
    4 of the row_groups[0].columns[0].meta_data struct), not from a byte
    offset or string search.
    """
    import io
    import struct as _struct
    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.table({"v": pa.array(list(range(50)), type=pa.int64())})
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="LZ4")
    data = buf.getvalue()
    assert data[:4] == b"PAR1" and data[-4:] == b"PAR1"
    md_len = _struct.unpack("<I", data[-8:-4])[0]
    md = data[-8 - md_len:-8]

    pos = [0]
    depth = [0]
    last_id_stack = [[0]]
    codec_path = []   # (depth, field_id) stack tracking whose "field 4" this is
    found = {}         # depth -> codec i32 value seen at "the right" struct

    def read_varint():
        result, shift = 0, 0
        while True:
            b = md[pos[0]]
            pos[0] += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def read_zigzag():
        v = read_varint()
        return (v >> 1) ^ -(v & 1)

    # root: FileMetaData -> field4 row_groups (list<struct>)
    #   RowGroup -> field1 columns (list<struct>)
    #     ColumnChunk -> field3 meta_data (struct)
    #       ColumnMetaData -> field4 codec (i32)   <-- what we want
    # Track the (field-id path) of open structs so "field 4, i32" is only
    # trusted when it's nested exactly where ColumnMetaData.codec lives.
    path = []  # list of field ids of currently-open structs, root-to-leaf

    while pos[0] < len(md):
        ctrl = md[pos[0]]
        pos[0] += 1
        if ctrl == 0:                              # STRUCT_END
            last_id_stack.pop()
            if path:
                path.pop()
            continue
        ftype = ctrl & 0x0F
        delta = (ctrl >> 4) & 0x0F
        fid = read_zigzag() if delta == 0 else last_id_stack[-1][-1] + delta
        last_id_stack[-1][-1] = fid
        if ftype == TC.I32 or ftype == 6:           # i32 or i64
            v = read_zigzag()
            if ftype == TC.I32 and fid == 4 and path == [4, 1, 3]:
                found["codec"] = v
        elif ftype == TC.BINARY:
            n = read_varint()
            pos[0] += n
        elif ftype in (TC.LIST, 10):                # list or set
            b0 = md[pos[0]]
            pos[0] += 1
            etype = b0 & 0x0F
            n = b0 >> 4
            if n == 15:
                n = read_varint()
            if etype == TC.STRUCT:
                # A list-of-struct element has NO per-element header byte in
                # compact protocol -- it's just that struct's own field
                # sequence, terminated by its own STOP(0), back to back n
                # times. Pushing all n contexts here (rather than one at a
                # time as each element is "entered") looks unusual but is
                # correct: every pushed context starts at field-id 0 with an
                # empty path frame, so it doesn't matter which of the n
                # equivalent frames a given element's fields delta-track
                # against or which frame a given STOP pops -- one push per
                # element, one pop per STOP, LIFO order is irrelevant when
                # every frame is identical at push time.
                for _ in range(n):
                    last_id_stack.append([0])
                    path.append(fid)
            else:
                for _ in range(n):
                    if etype in (TC.I32, 6):
                        read_zigzag()
                    elif etype == TC.BINARY:
                        ln = read_varint()
                        pos[0] += ln
        elif ftype == TC.STRUCT:
            path.append(fid)
            last_id_stack.append([0])

    assert "codec" in found, "could not structurally locate ColumnMetaData.codec"
    codec = found["codec"]
    assert codec != 5, (
        f"pyarrow actually wrote codec=5 (legacy Hadoop-framed LZ4)! "
        f"scope assumption was wrong -- re-derive the fixture strategy")
    assert codec == 7, f"expected LZ4_RAW(7), got {codec}"
    print(f"confirmed via raw footer bytes: pyarrow compression='LZ4' writes "
          f"thrift CompressionCodec={codec} (LZ4_RAW), never the deprecated "
          f"5 -- no external tool on this box can produce the true legacy "
          f"Hadoop-framed fixture; hand-construction below is required")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # single sub-block: same 5 values as the page_crc fixture, for an easy
    # independent eyeball cross-check.
    values = [10, 20, 30, 40, 50]
    body = b"".join(struct.pack("<q", v) for v in values)
    write_file(os.path.join(OUT_DIR, "lz4_hadoop_single.parquet"), body,
               [hadoop_frame(body)])

    # two sub-blocks: split 200 values down the middle, each half framed and
    # LZ4-block-compressed independently -- exercises the reader's per-page
    # multi-frame loop, not just the single-frame case.
    values2 = list(range(200))
    body2 = b"".join(struct.pack("<q", v) for v in values2)
    half = len(body2) // 2
    # split on an 8-byte boundary so each half is a whole number of int64s
    half -= half % 8
    frames = [hadoop_frame(body2[:half]), hadoop_frame(body2[half:])]
    write_file(os.path.join(OUT_DIR, "lz4_hadoop_multi.parquet"), body2,
               frames)

    print(f"wrote lz4_hadoop_single.parquet ({len(values)} values, "
          f"1 sub-block)")
    print(f"wrote lz4_hadoop_multi.parquet ({len(values2)} values, "
          f"2 sub-blocks, split at byte {half})")


if __name__ == "__main__":
    import sys
    if "--verify-no-tool" in sys.argv:
        verify_no_tool_emits_legacy_lz4()
        sys.exit(0)
    main()
