#!/usr/bin/env python3
"""Generate LEGACY 2-level LIST parquet fixtures (G2PQ-14).

LogicalTypes.md's backward-compatibility rules cover list shapes that predate
the 3-level `group (LIST) { repeated group list { element } }` convention:

  1. a `repeated` field with no LIST annotation is itself a list of its type;
  2. inside a LIST group, a repeated LEAF is the element (2-level);
  3. inside a LIST group, a repeated GROUP with more than one field IS the
     element (a struct), not a 3-level middle;
  4. a single-field repeated group named `array` or `<name>_tuple` IS the
     element too (struct-of-one), despite looking like a 3-level middle.

pyarrow cannot WRITE any of these (use_compliant_nested_type=False only
changes the 3-level element name to `item`), so the fixtures are constructed
here from the spec: a minimal thrift-compact + PLAIN + RLE writer with no
parquet library involved. pyarrow then READS every produced file back and the
values are asserted against the generating model -- so the fixtures are
certified by the ecosystem's implementation of the backward-compat rules
before bolt is ever tested against them. A deliberately mis-modelled
injection run proves the oracle check can fail.

Outputs (tests/data/):
  legacy2_bare.parquet        case 1: root { repeated int64 nums; id int64 }
  legacy2_annotated.parquet   case 2: optional group l (LIST) { repeated int32 element }
  legacy2_multifield.parquet  case 3: repeated group element { a int64; b int64 }
  legacy2_array.parquet       case 4: repeated group array { required int32 x }
  legacy2_equiv3.parquet      pyarrow-written 3-level file with the SAME rows
                              as legacy2_annotated (the value-equality oracle)
"""
import struct
import sys
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")

# ---------------------------------------------------------------- thrift compact
class TC:
    """Minimal thrift compact protocol writer (from the spec, no library)."""
    BOOL_TRUE, BOOL_FALSE, BYTE, I16, I32, I64 = 1, 2, 3, 4, 5, 6
    DOUBLE, BINARY, LIST, SET, MAP, STRUCT = 7, 8, 9, 10, 11, 12

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
        self.field(fid, self.I64)
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
        self.b.append(0)  # STOP
        self.last_fid.pop()


# ------------------------------------------------------------------- RLE levels
def rle_levels(levels, bit_width):
    """Encode a level stream as RLE runs (RLE/bit-packed hybrid, RLE only)."""
    assert bit_width <= 8
    out = bytearray()
    i = 0
    while i < len(levels):
        j = i
        while j < len(levels) and levels[j] == levels[i]:
            j += 1
        run = j - i
        header = run << 1  # low bit 0 = RLE run
        while header >= 0x80:
            out.append((header & 0x7F) | 0x80)
            header >>= 7
        out.append(header)
        out.append(levels[i])  # value in ceil(bw/8) = 1 byte
        i = j
    return bytes(out)


# ------------------------------------------------------------ file construction
PHYS_I32, PHYS_I64 = 1, 2
REQUIRED, OPTIONAL, REPEATED = 0, 1, 2
CONV_LIST = 3


def schema_element(tc, fid_ctx, name, repetition, phys=None, num_children=None,
                   converted=None):
    tc.struct_begin()
    if phys is not None:
        tc.i32(1, phys)
    tc.i32(3, repetition)
    tc.binary(4, name)
    if num_children is not None:
        tc.i32(5, num_children)
    if converted is not None:
        tc.i32(6, converted)
    tc.struct_end()


def data_page_v1(values_bytes, rep_levels, def_levels, max_rep, max_def,
                 num_values):
    """[rep RLE + u32 len][def RLE + u32 len][PLAIN values], v1 page header."""
    body = bytearray()
    if max_rep > 0:
        r = rle_levels(rep_levels, 1)
        body += struct.pack("<I", len(r)) + r
    if max_def > 0:
        d = rle_levels(def_levels, max(1, (max_def).bit_length()))
        body += struct.pack("<I", len(d)) + d
    body += values_bytes
    hdr = TC()
    hdr.struct_begin()
    hdr.i32(1, 0)                    # type = DATA_PAGE
    hdr.i32(2, len(body))            # uncompressed_page_size
    hdr.i32(3, len(body))            # compressed_page_size (UNCOMPRESSED)
    hdr.struct_begin(5)              # data_page_header
    hdr.i32(1, num_values)
    hdr.i32(2, 0)                    # encoding = PLAIN
    hdr.i32(3, 3)                    # definition_level_encoding = RLE
    hdr.i32(4, 3)                    # repetition_level_encoding = RLE
    hdr.struct_end()
    hdr.struct_end()
    return bytes(hdr.b) + bytes(body)


class Leaf:
    def __init__(self, path, phys, values, rep, deflev, max_rep, max_def):
        self.path = path            # list of path components
        self.phys = phys
        self.values = values        # python ints, already filtered to present
        self.rep = rep
        self.deflev = deflev
        self.max_rep = max_rep
        self.max_def = max_def


def write_file(path, schema_emit, leaves, num_rows):
    """schema_emit(tc) writes the SchemaElement list entries AFTER the root."""
    out = bytearray(b"PAR1")
    chunks = []
    for lf in leaves:
        fmt = "<i" if lf.phys == PHYS_I32 else "<q"
        vals = b"".join(struct.pack(fmt, v) for v in lf.values)
        page = data_page_v1(vals, lf.rep, lf.deflev, lf.max_rep, lf.max_def,
                            len(lf.rep) if lf.max_rep > 0 else len(lf.values))
        offset = len(out)
        out += page
        chunks.append((lf, offset, len(page)))

    meta = TC()
    meta.struct_begin()
    meta.i32(1, 1)                                   # version
    n_elems, emit = schema_emit
    meta.list_begin(2, TC.STRUCT, n_elems)
    emit(meta)
    meta.i64(3, num_rows)
    meta.list_begin(4, TC.STRUCT, 1)                 # row_groups
    meta.struct_begin()
    meta.list_begin(1, TC.STRUCT, len(chunks))       # columns
    total = 0
    for lf, off, size in chunks:
        total += size
        meta.struct_begin()
        meta.i64(2, off)                             # file_offset
        meta.struct_begin(3)                         # ColumnMetaData
        meta.i32(1, lf.phys)
        meta.list_begin(2, TC.I32, 1)
        meta._zigzag(0)                              # encodings = [PLAIN]
        meta.list_begin(3, TC.BINARY, len(lf.path))
        for comp in lf.path:
            meta._varint(len(comp))
            meta.b += comp.encode()
        meta.i32(4, 0)                               # codec = UNCOMPRESSED
        meta.i64(5, len(lf.rep) if lf.max_rep > 0 else len(lf.values))
        meta.i64(6, size)                            # total_uncompressed_size
        meta.i64(7, size)                            # total_compressed_size
        meta.i64(9, off)                             # data_page_offset
        meta.struct_end()
        meta.struct_end()
    meta.i64(2, total)                               # total_byte_size
    meta.i64(3, num_rows)
    meta.struct_end()
    meta.struct_end()
    md = bytes(meta.b)
    out += md
    out += struct.pack("<I", len(md))
    out += b"PAR1"
    with open(path, "wb") as f:
        f.write(out)


# ------------------------------------------------------------------- the model
# Shared row model for the LIST fixtures: 6 rows exercising null list, empty
# list, and 1..3-element lists. None = null list, [] = empty.
ROWS = [[10, 11, 12], None, [], [13], [14, 15], None]


def levels_for(rows, def_present, def_empty, def_null):
    """(rep, def, values) for one repeated leaf under max_rep=1."""
    rep, deflev, values = [], [], []
    for r in rows:
        if r is None:
            rep.append(0)
            deflev.append(def_null)
        elif len(r) == 0:
            rep.append(0)
            deflev.append(def_empty)
        else:
            for j, v in enumerate(r):
                rep.append(0 if j == 0 else 1)
                deflev.append(def_present)
                values.append(v)
    return rep, deflev, values


def make_bare(path):
    # message m { repeated int64 nums; required int64 id; }
    # A bare repeated field cannot be NULL, only empty.
    rows = [[1, 2, 3], [], [7], [8, 9], [], [42]]
    rep, deflev, values = levels_for(rows, 1, 0, None)
    ids = [100 + i for i in range(len(rows))]
    nums = Leaf(["nums"], PHYS_I64, values, rep, deflev, 1, 1)
    idl = Leaf(["id"], PHYS_I64, ids, [], [], 0, 0)

    def emit(tc):
        schema_element(tc, None, "m", REQUIRED, num_children=2)
        schema_element(tc, None, "nums", REPEATED, phys=PHYS_I64)
        schema_element(tc, None, "id", REQUIRED, phys=PHYS_I64)

    write_file(path, (3, emit), [nums, idl], len(rows))
    return rows, ids


def make_annotated(path):
    # message m { optional group l (LIST) { repeated int32 element; } }
    rep, deflev, values = levels_for(ROWS, 2, 1, 0)
    lf = Leaf(["l", "element"], PHYS_I32, values, rep, deflev, 1, 2)

    def emit(tc):
        schema_element(tc, None, "m", REQUIRED, num_children=1)
        schema_element(tc, None, "l", OPTIONAL, num_children=1,
                       converted=CONV_LIST)
        schema_element(tc, None, "element", REPEATED, phys=PHYS_I32)

    write_file(path, (3, emit), [lf], len(ROWS))
    return ROWS


def make_multifield(path):
    # message m { optional group l (LIST) {
    #   repeated group element { required int64 a; required int64 b; } } }
    # The repeated group has TWO fields -> it IS the element (a struct).
    rows = [[(1, 2), (3, 4)], None, [], [(5, 6)], [(7, 8), (9, 10)], None]
    scalar = [[p[0] for p in r] if r is not None else None for r in rows]
    rep, deflev, avals = levels_for(scalar, 2, 1, 0)
    bvals = [p[1] for r in rows if r for p in r]
    la = Leaf(["l", "element", "a"], PHYS_I64, avals, rep, deflev, 1, 2)
    lb = Leaf(["l", "element", "b"], PHYS_I64, bvals, rep, deflev, 1, 2)

    def emit(tc):
        schema_element(tc, None, "m", REQUIRED, num_children=1)
        schema_element(tc, None, "l", OPTIONAL, num_children=1,
                       converted=CONV_LIST)
        schema_element(tc, None, "element", REPEATED, num_children=2)
        schema_element(tc, None, "a", REQUIRED, phys=PHYS_I64)
        schema_element(tc, None, "b", REQUIRED, phys=PHYS_I64)

    write_file(path, (5, emit), [la, lb], len(rows))
    return rows


def make_array_named(path):
    # message m { optional group l (LIST) {
    #   repeated group array { required int32 x; } } }
    # Single-field repeated group, but the name `array` means the group IS the
    # element (list<struct<x>>), NOT a 3-level middle.
    rows = [[(20,), (21,)], None, [], [(22,)], [(23,), (24,), (25,)], None]
    scalar = [[p[0] for p in r] if r is not None else None for r in rows]
    rep, deflev, values = levels_for(scalar, 2, 1, 0)
    lf = Leaf(["l", "array", "x"], PHYS_I32, values, rep, deflev, 1, 2)

    def emit(tc):
        schema_element(tc, None, "m", REQUIRED, num_children=1)
        schema_element(tc, None, "l", OPTIONAL, num_children=1,
                       converted=CONV_LIST)
        schema_element(tc, None, "array", REPEATED, num_children=1)
        schema_element(tc, None, "x", REQUIRED, phys=PHYS_I32)

    write_file(path, (4, emit), [lf], len(rows))
    return rows


def make_equiv3(path):
    # The SAME rows as legacy2_annotated, written by pyarrow as a modern
    # 3-level file. Value equality between this and the legacy file is the
    # tracker's acceptance bar.
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table({"l": pa.array(ROWS, type=pa.list_(pa.int32()))})
    pq.write_table(t, path, compression="none", use_dictionary=False)


# -------------------------------------------------------------------- verify
def verify(inject=False):
    """Read every fixture back with pyarrow and assert the model.

    inject=True flips one expected value to prove this check can fail.
    """
    import pyarrow.parquet as pq

    def eq(name, got, want):
        if got != want:
            raise SystemExit(f"ORACLE MISMATCH {name}: got {got!r} want {want!r}")

    t = pq.read_table(os.path.join(OUT_DIR, "legacy2_bare.parquet"))
    want_rows = [[1, 2, 3], [], [7], [8, 9], [], [42]]
    if inject:
        want_rows[0] = [1, 2, 999]
    eq("bare.nums", t.column("nums").to_pylist(), want_rows)
    eq("bare.id", t.column("id").to_pylist(), [100 + i for i in range(6)])

    t = pq.read_table(os.path.join(OUT_DIR, "legacy2_annotated.parquet"))
    eq("annotated.l", t.column("l").to_pylist(), ROWS)

    t = pq.read_table(os.path.join(OUT_DIR, "legacy2_multifield.parquet"))
    want = [[{"a": a, "b": b} for (a, b) in r] if r is not None else None
            for r in [[(1, 2), (3, 4)], None, [], [(5, 6)],
                      [(7, 8), (9, 10)], None]]
    eq("multifield.l", t.column("l").to_pylist(), want)
    # pyarrow must agree this is list<struct<a,b>>, i.e. it applied the
    # multi-field rule instead of reading a 3-level middle.
    typ = t.schema.field("l").type
    import pyarrow as pa
    assert pa.types.is_list(typ) and pa.types.is_struct(typ.value_type), typ

    t = pq.read_table(os.path.join(OUT_DIR, "legacy2_array.parquet"))
    want = [[{"x": x[0]} for x in r] if r is not None else None
            for r in [[(20,), (21,)], None, [], [(22,)],
                      [(23,), (24,), (25,)], None]]
    eq("array.l", t.column("l").to_pylist(), want)
    typ = t.schema.field("l").type
    assert pa.types.is_list(typ) and pa.types.is_struct(typ.value_type), \
        f"`array` naming rule not applied: {typ}"

    t = pq.read_table(os.path.join(OUT_DIR, "legacy2_equiv3.parquet"))
    eq("equiv3.l", t.column("l").to_pylist(), ROWS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_bare(os.path.join(OUT_DIR, "legacy2_bare.parquet"))
    make_annotated(os.path.join(OUT_DIR, "legacy2_annotated.parquet"))
    make_multifield(os.path.join(OUT_DIR, "legacy2_multifield.parquet"))
    make_array_named(os.path.join(OUT_DIR, "legacy2_array.parquet"))
    make_equiv3(os.path.join(OUT_DIR, "legacy2_equiv3.parquet"))

    # Discriminating power of the oracle itself: an injected wrong expectation
    # must fail before the real check is trusted.
    try:
        verify(inject=True)
    except SystemExit as e:
        print(f"injection caught as expected: {e}")
    else:
        raise SystemExit("INJECTION NOT CAUGHT -- oracle check is vacuous")

    verify(inject=False)
    print("all legacy 2-level fixtures verified by pyarrow")


if __name__ == "__main__":
    main()
