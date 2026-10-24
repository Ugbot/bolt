#!/usr/bin/env python3
"""Nested STRUCT / LIST / MAP read fixtures for parquet_read_nested_column.

pyarrow writes every file and pyarrow reads every file back to produce the
expected rendering (`<name>.expected`), so the oracle is the ecosystem's
record assembly, never bolt's. test_bolt_parquet_read_nested.cpp renders the
assembled BoltColumn tree in the SAME text form and compares line by line.

    uv run --no-project --python 3.12 --with pyarrow \
        python scripts/make_nested_read_fixtures.py

Expected file lines (tab separated):
    schema <top> <render>   field tree with names and field ids
    row    <top> <render>   one per row, file order, all row groups

Value rendering: null | int | %.17g float | "bytes" (\\xHH escapes) |
D<days> | T<micros> | <mantissa>e-<scale> | [list] | {name:val,...} |
<key=val,...>. Booleans render 0/1 (bolt lands them as Int64).
"""
import datetime as dt
import decimal
import os
import sys
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "nested_read")
LEGACY = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
N = 200
_ids = [0]


def fid():
    _ids[0] += 1
    return {"PARQUET:field_id": str(_ids[0])}


def F(name, t, nullable=True, ident=True):
    return pa.field(name, t, nullable=nullable, metadata=fid() if ident else None)


def L(t, nullable=True, name="element"):
    return pa.list_(F(name, t, nullable))


def M(kt, vt, vnull=True):
    return pa.map_(F("key", kt, False), F("value", vt, vnull))


# ---- deterministic row model ------------------------------------------------

def nul(i, k):
    return (i * 7 + k) % 11 == 0


def mk_list(i, salt, f, n=4):
    """None, [] and populated lists in a fixed rotation."""
    m = (i + salt) % 9
    if m == 0:
        return None
    if m == 4:
        return []
    return [f(i, j) for j in range((i + salt) % n + 1)]


def struct_rows(i):
    if i % 13 == 5:
        return None
    return {
        "a": None if nul(i, 1) else i * 3 - 100,
        "b": "s%d" % (i % 17) + "x" * (i % 15),
        "c": None if i % 10 == 3 else {
            "x": None if nul(i, 2) else i / 4.0,
            "y": mk_list(i, 1, lambda r, j: None if (r + j) % 6 == 0 else r * 10 + j),
        },
        "d": {"e": None if nul(i, 3) else (i % 2 == 0)},
    }


def table_struct():
    _ids[0] = 0
    sch = pa.schema([
        F("id", pa.int64(), False),
        F("s", pa.struct([
            F("a", pa.int32()),
            F("b", pa.string(), False),
            F("c", pa.struct([F("x", pa.float64()), F("y", L(pa.int64()))])),
            F("d", pa.struct([F("e", pa.bool_())]), False),
        ])),
        F("t", pa.struct([F("u", pa.date32()), F("w", pa.decimal128(9, 2)),
                          F("z", pa.decimal128(30, 6)), F("m", pa.timestamp("ms")),
                          F("tm", pa.time64("us")), F("bin", pa.binary()),
                          F("i16", pa.int16())]), False),
        F("flat", pa.string()),
    ])
    rows = []
    for i in range(N):
        rows.append({
            "id": i,
            "s": struct_rows(i),
            "t": {"u": None if nul(i, 4) else dt.date(1970, 1, 1) + dt.timedelta(days=i * 37 - 5000),
                  "w": None if nul(i, 5) else decimal.Decimal(i * 1234 - 99999).scaleb(-2),
                  "z": decimal.Decimal(i * 10**20 + 7).scaleb(-6),
                  "m": None if nul(i, 8) else dt.datetime(2001, 2, 3) + dt.timedelta(milliseconds=i * 7919),
                  "tm": dt.time(i % 24, i % 60, i % 60, (i * 37) % 1000000),
                  "bin": None if nul(i, 9) else bytes(range(i % 7, i % 7 + i % 20)),
                  "i16": i * 97 - 9000},
            "flat": None if nul(i, 6) else "f%d" % (i % 5),
        })
    return pa.Table.from_pylist(rows, schema=sch)


def table_list(item_name):
    _ids[0] = 100
    elem = lambda: pa.struct([F("p", pa.int64()), F("q", pa.string())])
    sch = pa.schema([
        F("id", pa.int32(), False),
        F("ls", L(elem(), name=item_name)),
        F("ll", L(L(pa.int32(), name=item_name), name=item_name)),
        F("li", L(pa.int64(), False, name=item_name), False),
        F("sl", pa.struct([F("l", L(pa.string(), name=item_name)), F("k", pa.int64())])),
        F("lu", L(pa.binary(16), name=item_name)),
        F("lt", L(pa.timestamp("us", tz="UTC"), name=item_name)),
        F("lf", L(pa.float32(), name=item_name)),
        F("lb", L(pa.binary(5), name=item_name)),
        F("ln", L(pa.timestamp("ns"), name=item_name)),
    ])
    rows = []
    for i in range(N):
        rows.append({
            "id": i,
            "ls": mk_list(i, 0, lambda r, j: None if (r + j) % 8 == 0 else
                          {"p": None if (r + j) % 5 == 0 else r * 100 + j,
                           "q": None if (r + j) % 7 == 0 else "q%d" % ((r * j) % 23)}),
            "ll": mk_list(i, 2, lambda r, j: mk_list(r + j, 3, lambda a, b: None if (a + b) % 7 == 0 else a - b)),
            "li": [i + j for j in range(i % 4)],
            "sl": None if i % 17 == 2 else {"l": mk_list(i, 5, lambda r, j: "e%d" % (r + j) * ((r + j) % 4)),
                                             "k": None if nul(i, 7) else -i},
            "lu": mk_list(i, 6, lambda r, j: None if (r + j) % 5 == 1 else ((r * 1000003 + j) * 0x9E3779B97F4A7C15 % (1 << 128)).to_bytes(16, "big")),
            "lt": mk_list(i, 7, lambda r, j: dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=r * 1000003 + j)),
            "lf": mk_list(i, 8, lambda r, j: None if (r + j) % 6 == 2 else (r - j) * 0.25),
            "lb": mk_list(i, 1, lambda r, j: bytes([(r + j + k) % 256 for k in range(5)])),
            "ln": mk_list(i, 3, lambda r, j: dt.datetime(1999, 12, 31) + dt.timedelta(microseconds=r * 333 + j)),
        })
    tab = pa.Table.from_pylist(rows, schema=sch)
    k = tab.schema.get_field_index("lu")
    f = tab.schema.field(k)
    vf = f.type.value_field
    ut = pa.list_(pa.field(vf.name, pa.uuid(), metadata=vf.metadata))
    return tab.set_column(k, f.with_type(ut), tab.column(k).cast(ut))


def table_map():
    _ids[0] = 200
    sch = pa.schema([
        F("id", pa.int64(), False),
        F("ms", M(pa.string(), pa.int64())),
        F("mi", M(pa.int32(), pa.struct([F("u", pa.float64()), F("v", L(pa.string()))]))),
        F("lm", L(M(pa.string(), pa.int32()))),
        F("sm", pa.struct([F("m", M(pa.int64(), pa.string(), False))])),
        F("mm", M(pa.string(), M(pa.int32(), pa.int32()))),
    ])
    rows = []
    for i in range(N):
        rows.append({
            "id": i,
            "ms": None if i % 9 == 0 else [("k%d" % j, None if (i + j) % 5 == 0 else i * j) for j in range(i % 4)],
            "mi": None if i % 11 == 1 else [(j * 3 - i, None if (i + j) % 6 == 0 else
                                            {"u": None if nul(i, j) else i + j / 8.0,
                                             "v": mk_list(i, j, lambda r, k: "v%d" % (r + k))}) for j in range(i % 3)],
            "lm": mk_list(i, 4, lambda r, j: None if (r + j) % 7 == 0 else
                          [("m%d" % k, r + k) for k in range((r + j) % 3)]),
            "sm": None if i % 10 == 9 else {"m": [(i * 10 + j, "z" * (j + 1)) for j in range(i % 3)]},
            "mm": None if i % 8 == 3 else [("o%d" % j, None if (i + j) % 4 == 0 else
                                            [(k, i * k) for k in range((i + j) % 3)]) for j in range(i % 3)],
        })
    return pa.Table.from_pylist(rows, schema=sch)


# ---- rendering (the oracle side) --------------------------------------------

def q(b):
    out = []
    for c in b:
        if 0x20 <= c < 0x7F and c not in (0x22, 0x5C):
            out.append(chr(c))
        else:
            out.append("\\x%02x" % c)
    return '"' + "".join(out) + '"'


EPOCH_D = dt.date(1970, 1, 1)
EPOCH = dt.datetime(1970, 1, 1)
EPOCH_TZ = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def rv(v, t):
    if v is None:
        return "null"
    if pa.types.is_struct(t):
        return "{" + ",".join("%s:%s" % (t.field(k).name, rv(v[t.field(k).name], t.field(k).type))
                              for k in range(t.num_fields)) + "}"
    if pa.types.is_map(t):
        return "<" + ",".join("%s=%s" % (rv(k, t.key_type), rv(x, t.item_type)) for k, x in v) + ">"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "[" + ",".join(rv(e, t.value_type) for e in v) + "]"
    if isinstance(t, pa.BaseExtensionType):
        return rv(v.bytes if isinstance(v, uuid.UUID) else v, t.storage_type)
    if pa.types.is_boolean(t):
        return "1" if v else "0"
    if pa.types.is_integer(t):
        return str(v)
    if pa.types.is_floating(t):
        return "%.17g" % v
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return q(v.encode("utf-8"))
    if pa.types.is_binary(t) or pa.types.is_fixed_size_binary(t):
        return q(v)
    if pa.types.is_decimal(t):
        return "%de-%d" % (int(v.scaleb(t.scale)), t.scale)
    if pa.types.is_date32(t):
        return "D%d" % (v - EPOCH_D).days
    if pa.types.is_time(t):
        return "U%d" % (((v.hour * 60 + v.minute) * 60 + v.second) * 1000000 + v.microsecond)
    if pa.types.is_timestamp(t):
        base = EPOCH_TZ if v.tzinfo is not None else EPOCH
        return "T%d" % ((v - base) // dt.timedelta(microseconds=1))
    raise SystemExit("no render for %s" % t)


def rs(f):
    md = f.metadata or {}
    s = f.name + ("#" + md[b"PARQUET:field_id"].decode() if b"PARQUET:field_id" in md else "")
    t = f.type
    if pa.types.is_struct(t):
        return s + "{" + ",".join(rs(t.field(k)) for k in range(t.num_fields)) + "}"
    if pa.types.is_map(t):
        return s + "<" + rs(t.key_field) + "," + rs(t.item_field) + ">"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return s + "[" + rs(t.value_field) + "]"
    return s


def dump(path, out_path):
    pf = pq.ParquetFile(path)
    tab = pf.read()
    lines = []
    for k, f in enumerate(pf.schema_arrow):
        lines.append("schema\t%d\t%s" % (k, rs(f)))
    for k, f in enumerate(tab.schema):
        for v in tab.column(k).to_pylist():
            lines.append("row\t%d\t%s" % (k, rv(v, f.type)))
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return pf.metadata.num_row_groups


VARIANTS = [
    # name, writer options
    ("v1_dict", dict(data_page_version="1.0", use_dictionary=True, compression="snappy")),
    ("v2_plain", dict(data_page_version="2.0", use_dictionary=False, compression="zstd")),
    ("v2_dict", dict(data_page_version="2.0", use_dictionary=True, compression="none")),
]


def write(name, tab, legacy=False, **opts):
    path = os.path.join(OUT, name + ".parquet")
    pq.write_table(tab, path, row_group_size=110, data_page_size=512,
                   write_batch_size=32, use_compliant_nested_type=not legacy,
                   **opts)
    groups = dump(path, os.path.join(OUT, name + ".expected"))
    assert groups >= 2, name
    print("wrote", name, "row groups", groups)


# ---- hand-built legacy shapes pyarrow cannot write -------------------------
# Built with make_legacy_list_fixtures.py's spec-level writer (no parquet
# library), then read back by pyarrow for the expected dump like the rest.

def _legacy():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy", os.path.join(os.path.dirname(__file__), "make_legacy_list_fixtures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _elem(lg, tc, name, rep, phys=None, n=None, conv=None, logical=None):
    tc.struct_begin()
    if phys is not None:
        tc.i32(1, phys)
    tc.i32(3, rep)
    tc.binary(4, name)
    if n is not None:
        tc.i32(5, n)
    if conv is not None:
        tc.i32(6, conv)
    if logical is not None:            # LogicalType union member, empty struct
        tc.struct_begin(10)
        tc.struct_begin(logical)
        tc.struct_end()
        tc.struct_end()
    tc.struct_end()


def _levels(rows, entry):
    """rep/def/values for one leaf under one optional LIST/MAP level.
    entry(x) -> (def, value-or-None) for a present element."""
    rep, dl, vals = [], [], []
    for r in rows:
        if r is None:
            rep.append(0); dl.append(0)
        elif not r:
            rep.append(0); dl.append(1)
        else:
            for j, x in enumerate(r):
                d, v = entry(x)
                rep.append(0 if j == 0 else 1); dl.append(d)
                if v is not None:
                    vals.append(v)
    return rep, dl, vals


def legacy_shapes():
    lg = _legacy()
    R, O, P = lg.REQUIRED, lg.OPTIONAL, lg.REPEATED
    I32, I64 = lg.PHYS_I32, lg.PHYS_I64
    # MAP_KEY_VALUE on the outer group (pre-MAP writers).
    mrows = [None, [], [(1, 10), (2, None)], [(3, 30)], [(4, None), (5, 50), (6, 60)]]
    kr, kd, kv = _levels(mrows, lambda e: (2, e[0]))
    vr, vd, vv = _levels(mrows, lambda e: (3, e[1]) if e[1] is not None else (2, None))
    leaves = [lg.Leaf(["m", "map", "key"], I64, kv, kr, kd, 1, 2),
              lg.Leaf(["m", "map", "value"], I32, vv, vr, vd, 1, 3)]

    def emit_mkv(tc):
        _elem(lg, tc, "s", R, n=1)
        _elem(lg, tc, "m", O, n=1, conv=2)
        _elem(lg, tc, "map", P, n=2)
        _elem(lg, tc, "key", R, phys=I64)
        _elem(lg, tc, "value", O, phys=I32)
    lg.write_file(os.path.join(OUT, "legacy_map_kv.parquet"), (5, emit_mkv), leaves, len(mrows))
    # parquet-mr / Spark legacy 3-level: repeated group `bag` { optional `array` }.
    lrows = [[1, None, 3], None, [], [4], [None]]
    r, d, v = _levels(lrows, lambda e: (3, e) if e is not None else (2, None))

    def emit_bag(tc):
        _elem(lg, tc, "s", R, n=1)
        _elem(lg, tc, "l", O, n=1, conv=3)
        _elem(lg, tc, "bag", P, n=1)
        _elem(lg, tc, "array", O, phys=I64)
    lg.write_file(os.path.join(OUT, "legacy_bag_array.parquet"), (4, emit_bag),
                  [lg.Leaf(["l", "bag", "array"], I64, v, r, d, 1, 3)], len(lrows))
    # LIST and MAP annotated by LogicalType only (no ConvertedType).
    leaves = [lg.Leaf(["l", "list", "element"], I64, v, r, d, 1, 3),
              lg.Leaf(["mp", "key_value", "key"], I64, kv, kr, kd, 1, 2),
              lg.Leaf(["mp", "key_value", "value"], I32, vv, vr, vd, 1, 3)]

    def emit_logical(tc):
        _elem(lg, tc, "s", R, n=2)
        _elem(lg, tc, "l", O, n=1, logical=3)
        _elem(lg, tc, "list", P, n=1)
        _elem(lg, tc, "element", O, phys=I64)
        _elem(lg, tc, "mp", O, n=1, logical=2)
        _elem(lg, tc, "key_value", P, n=2)
        _elem(lg, tc, "key", R, phys=I64)
        _elem(lg, tc, "value", O, phys=I32)
    assert len(lrows) == len(mrows)
    lg.write_file(os.path.join(OUT, "legacy_logical_only.parquet"), (8, emit_logical),
                  leaves, len(lrows))
    for name in ("legacy_map_kv", "legacy_bag_array", "legacy_logical_only"):
        dump(os.path.join(OUT, name + ".parquet"), os.path.join(OUT, name + ".expected"))
        print("wrote", name)


def main():
    os.makedirs(OUT, exist_ok=True)
    for vname, opts in VARIANTS:
        write("struct_" + vname, table_struct(), **opts)
        write("list_" + vname, table_list("element"), **opts)
        write("map_" + vname, table_map(), **opts)
    # pyarrow's legacy (non-compliant) 3-level naming: list.item
    write("list_item_v1", table_list("item"), legacy=True,
          data_page_version="1.0", use_dictionary=True)
    legacy_shapes()
    # Spec-constructed 2-level legacy shapes from make_legacy_list_fixtures.py.
    for name in ("legacy2_bare", "legacy2_annotated", "legacy2_multifield",
                 "legacy2_array"):
        src = os.path.join(LEGACY, name + ".parquet")
        if os.path.exists(src):
            dump(src, os.path.join(OUT, name + ".expected"))
            print("dumped", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
