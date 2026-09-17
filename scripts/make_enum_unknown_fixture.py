#!/usr/bin/env python3
"""Generate ENUM / UNKNOWN logical-type parquet fixtures (G2PQ-17).

ENUM: neither pyarrow nor any Python parquet library exposes a way to opt
into writing ConvertedType.ENUM / LogicalType.ENUM (verified: pyarrow's
dictionary_encode() writes plain STRING, not ENUM; there is no `pa.enum_()`).
So -- following the exact hand-thrift-construction precedent
make_page_crc_fixture.py set for G2PQ-19 -- this hand-builds a minimal
single-column, single-row-group, PLAIN-encoded BYTE_ARRAY file with
ConvertedType.ENUM (field 6 = 4) and NO LogicalType (field 10) at all: the
shape a genuinely legacy (pre parquet-format LogicalType, e.g. old Hive/Pig)
writer would emit. This exercises bolt's ConvertedType-only fallback path
(derive_logical_from_converted) with no self-consistency involved -- no bolt
code writes this file.

UNKNOWN: pyarrow CAN write this one -- `pa.null()` becomes an
`optional int32 ... (Null)` column, i.e. LogicalType.UNKNOWN (NullType) over
an INT32 physical carrier, every value null. That's a real external oracle,
used here in preference to hand-construction.
"""
import os
import struct

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")

PHYS_BYTE_ARRAY = 6  # parquet Type.BYTE_ARRAY
CONV_ENUM = 4        # parquet ConvertedType.ENUM


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


def plain_byte_array_body(values):
    out = bytearray()
    for s in values:
        enc = s.encode()
        out += struct.pack("<I", len(enc))
        out += enc
    return bytes(out)


def page_header_bytes(body, n_values):
    """DATA_PAGE, PLAIN, required column -- no def/rep levels."""
    hdr = TC()
    hdr.struct_begin()
    hdr.i32(1, 0)               # type = DATA_PAGE
    hdr.i32(2, len(body))       # uncompressed_page_size
    hdr.i32(3, len(body))       # compressed_page_size
    hdr.struct_begin(5)         # data_page_header
    hdr.i32(1, n_values)        # num_values
    hdr.i32(2, 0)               # encoding = PLAIN
    hdr.i32(3, 0)               # definition_level_encoding (unused, required)
    hdr.i32(4, 0)               # repetition_level_encoding (unused, required)
    hdr.struct_end()
    hdr.struct_end()
    return bytes(hdr.b)


def write_enum_legacy_file(path, values):
    """One row group, one REQUIRED BYTE_ARRAY column 'status', ConvertedType
    ENUM only -- no LogicalType field at all (the pre-LogicalType writer
    shape)."""
    out = bytearray(b"PAR1")
    body = plain_byte_array_body(values)
    page = page_header_bytes(body, len(values)) + body
    offset = len(out)
    out += page
    size = len(page)

    meta = TC()
    meta.struct_begin()
    meta.i32(1, 1)                                    # version
    meta.list_begin(2, TC.STRUCT, 2)                   # schema: root + 1 leaf
    meta.struct_begin()                                # root
    meta.i32(3, 0)
    meta.binary(4, "root")
    meta.i32(5, 1)
    meta.struct_end()
    meta.struct_begin()                                # leaf "status"
    meta.i32(1, PHYS_BYTE_ARRAY)
    meta.i32(3, 0)                                     # REQUIRED
    meta.binary(4, "status")
    meta.i32(6, CONV_ENUM)                             # converted_type ENUM
    # deliberately NO field 10 (logicalType) -- the legacy shape under test
    meta.struct_end()
    meta.i64(3, len(values))                           # num_rows
    meta.list_begin(4, TC.STRUCT, 1)                    # row_groups
    meta.struct_begin()
    meta.list_begin(1, TC.STRUCT, 1)                    # columns
    meta.struct_begin()
    meta.i64(2, offset)                                 # file_offset
    meta.struct_begin(3)                                # ColumnMetaData
    meta.i32(1, PHYS_BYTE_ARRAY)
    meta.list_begin(2, TC.I32, 1)
    meta._zigzag(0)                                     # encodings = [PLAIN]
    meta.list_begin(3, TC.BINARY, 1)
    meta._varint(6)
    meta.b += b"status"
    meta.i32(4, 0)                                      # codec = UNCOMPRESSED
    meta.i64(5, len(values))                            # num_values
    meta.i64(6, size)                                   # total_uncompressed_size
    meta.i64(7, size)                                   # total_compressed_size
    meta.i64(9, offset)                                 # data_page_offset
    meta.struct_end()
    meta.struct_end()
    meta.i64(2, size)                                   # total_byte_size
    meta.i64(3, len(values))
    meta.struct_end()
    meta.struct_end()
    md = bytes(meta.b)
    out += md
    out += struct.pack("<I", len(md))
    out += b"PAR1"
    with open(path, "wb") as f:
        f.write(out)


def write_unknown_pyarrow_file(path, n):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({
        "u": pa.array([None] * n, pa.null()),
        "n": pa.array(list(range(n)), pa.int64()),
    })
    pq.write_table(t, path)


ENUM_VALUES = ["ACTIVE", "INACTIVE", "ACTIVE", "PENDING", "ACTIVE"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    enum_path = os.path.join(OUT_DIR, "enum_converted_legacy.parquet")
    write_enum_legacy_file(enum_path, ENUM_VALUES)
    print(f"wrote {enum_path} (values={ENUM_VALUES})")

    unknown_path = os.path.join(OUT_DIR, "unknown_null_pyarrow.parquet")
    write_unknown_pyarrow_file(unknown_path, 7)
    print(f"wrote {unknown_path} (7 rows, all null)")


def verify():
    """pyarrow must open both fixtures without complaint -- proves the
    hand-crafted ENUM file is well-formed parquet even though pyarrow does
    not understand ConvertedType.ENUM specially (it just reads BYTE_ARRAY
    bytes as strings, exactly bolt's own blind-passthrough behaviour)."""
    import pyarrow.parquet as pq
    enum_path = os.path.join(OUT_DIR, "enum_converted_legacy.parquet")
    got = pq.read_table(enum_path).column("status").to_pylist()
    got = [v.decode() if isinstance(v, bytes) else v for v in got]
    ok1 = got == ENUM_VALUES
    print(f"{'OK' if ok1 else 'FAIL'}: enum_converted_legacy.parquet -> {got}")

    unknown_path = os.path.join(OUT_DIR, "unknown_null_pyarrow.parquet")
    t = pq.read_table(unknown_path)
    got_u = t.column("u").to_pylist()
    ok2 = got_u == [None] * 7 and str(t.schema.field("u").type) == "null"
    print(f"{'OK' if ok2 else 'FAIL'}: unknown_null_pyarrow.parquet -> "
          f"{got_u}, type={t.schema.field('u').type}")
    return ok1 and ok2


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    main()
