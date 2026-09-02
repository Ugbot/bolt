#!/usr/bin/env python3
"""Decode bolt's chunk-level Statistics from the SPEC and check them
(G2PQ-21 unsigned BYTE_ARRAY ordering; G2PQ-22 NaN / signed-zero rules).

    ./test_bolt_parquet_write_stats          # writes the fixture
    python3 scripts/parquet_stats_check.py <dir-containing-it>

WHY THIS EXISTS. The stats a writer emits are consumed by OTHER readers'
pruning: get the ordering or the NaN rules wrong and it is DuckDB / pyarrow /
parquet-mr that silently drop matching rows, while bolt's own round-trip
stays green (a writer and reader sharing a misreading agree perfectly). So
the Statistics struct and FileMetaData.column_orders are decoded here with a
from-spec thrift compact reader — never bolt's parser and never through
pyarrow's Statistics object (whose repr cannot even hold non-UTF8 bounds).

What a stats-trusting reader relies on, all checked:

  * BYTE_ARRAY min/max use UNSIGNED byte-wise ordering, proven on values
    with bytes >= 0x80 where the signed and unsigned orderings disagree on
    every one of the four bounds (an ASCII fixture can never catch this);
  * no NaN is ever written as a min or max; a chunk of only NaNs OMITS the
    statistic (readers treat a missing statistic as "unknown", never as
    "empty range" — the omission depends on that);
  * signed zeros are conservative: a zero min is stored as -0.0 and a zero
    max as +0.0, so BOTH zeros lie inside [min, max] under exact-equality
    pruning;
  * +/-inf are legal exact bounds;
  * the deprecated Statistics.min/max (fields 1/2, whose ordering semantics
    differ from min_value/max_value) are ABSENT — writing them with modern
    semantics would poison a legacy reader;
  * column_orders declares exactly one TYPE_ORDER per column — without it
    parquet-cpp refuses to trust min_value/max_value at all;
  * the all-NaN column's ColumnIndex page entry is an EMPTY byte array with
    null_pages=false (the page has values, just no usable bound), and no
    NaN bytes appear in any ColumnIndex bound.

Every expectation is re-derived from the fixture's generating model (the
values test_bolt_parquet_write_stats.cpp writes), and a set of injections
proves each check can actually fail.
"""
import math
import os
import struct
import sys

# ---- thrift compact protocol (same shape as parquet_pageindex_check.py) ---
T_STOP, T_TRUE, T_FALSE, T_I8, T_I16, T_I32, T_I64 = 0, 1, 2, 3, 4, 5, 6
T_DOUBLE, T_BINARY, T_LIST, T_SET, T_MAP, T_STRUCT = 7, 8, 9, 10, 11, 12


class TC:
    def __init__(self, buf, pos=0):
        self.b, self.p = buf, pos

    def byte(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def uvarint(self):
        shift = out = 0
        while True:
            c = self.byte()
            out |= (c & 0x7F) << shift
            if not c & 0x80:
                return out
            shift += 7

    def zigzag(self):
        n = self.uvarint()
        return (n >> 1) ^ -(n & 1)

    def binary(self):
        n = self.uvarint()
        v = self.b[self.p:self.p + n]
        self.p += n
        return bytes(v)

    def list_header(self):
        h = self.byte()
        size, et = h >> 4, h & 0x0F
        if size == 15:
            size = self.uvarint()
        return size, et

    def skip(self, t):
        if t in (T_TRUE, T_FALSE):
            return
        if t == T_I8:
            self.byte()
        elif t in (T_I16, T_I32, T_I64):
            self.zigzag()
        elif t == T_DOUBLE:
            self.p += 8
        elif t == T_BINARY:
            self.binary()
        elif t in (T_LIST, T_SET):
            n, et = self.list_header()
            for _ in range(n):
                if et in (T_TRUE, T_FALSE):
                    self.byte()
                else:
                    self.skip(et)
        elif t == T_STRUCT:
            self.struct(lambda fid, ft, r: False)
        else:
            raise ValueError("bad thrift type %d" % t)

    def struct(self, visit):
        fid = 0
        while True:
            h = self.byte()
            if h == T_STOP:
                return
            delta, ft = h >> 4, h & 0x0F
            fid = fid + delta if delta else self.zigzag()
            if not visit(fid, ft, self):
                self.skip(ft)


# ---- FileMetaData walk (from parquet.thrift, no library) ------------------
PHYS = {0: "BOOLEAN", 1: "INT32", 2: "INT64", 4: "FLOAT",
        5: "DOUBLE", 6: "BYTE_ARRAY", 7: "FLBA"}


def parse_statistics(rr):
    """Statistics {1 max(dep), 2 min(dep), 3 null_count, 4 distinct,
    5 max_value, 6 min_value}. Returns raw bytes; None = field absent."""
    st = {"dep_max": None, "dep_min": None, "null_count": None,
          "max_value": None, "min_value": None}

    def v(fid, ft, x):
        if fid == 1:
            st["dep_max"] = x.binary()
            return True
        if fid == 2:
            st["dep_min"] = x.binary()
            return True
        if fid == 3:
            st["null_count"] = x.zigzag()
            return True
        if fid == 5:
            st["max_value"] = x.binary()
            return True
        if fid == 6:
            st["min_value"] = x.binary()
            return True
        return False
    rr.struct(v)
    return st


def parse_footer(buf):
    """{columns: [{name, type, stats|None, ci_off, ci_len}],
        n_column_orders, n_type_order}"""
    flen = struct.unpack("<I", buf[-8:-4])[0]
    r = TC(buf, len(buf) - 8 - flen)
    out = {"columns": [], "n_column_orders": 0, "n_type_order": 0}

    def chunk(rr):
        d = {"name": None, "type": None, "stats": None,
             "ci_off": None, "ci_len": None}

        def meta(fid, ft, x):
            if fid == 1:
                d["type"] = PHYS.get(x.zigzag(), "?")
                return True
            if fid == 3:
                n, _t = x.list_header()
                d["name"] = ".".join(x.binary().decode() for _ in range(n))
                return True
            if fid == 12:
                d["stats"] = parse_statistics(x)
                return True
            return False

        def v(fid, ft, x):
            if fid == 3:
                x.struct(meta)
                return True
            if fid == 6:
                d["ci_off"] = x.zigzag()
                return True
            if fid == 7:
                d["ci_len"] = x.zigzag()
                return True
            return False
        rr.struct(v)
        return d

    def row_group(rr):
        def v(fid, ft, x):
            if fid == 1:
                n, _t = x.list_header()
                for _ in range(n):
                    out["columns"].append(chunk(x))
                return True
            return False
        rr.struct(v)

    def order(rr):
        def v(fid, ft, x):
            if fid == 1:                      # TYPE_ORDER (empty struct)
                x.struct(lambda a, b, c: False)
                out["n_type_order"] += 1
                return True
            return False
        rr.struct(v)

    def visit(fid, ft, rr):
        if fid == 4:
            n, _t = rr.list_header()
            for _ in range(n):
                row_group(rr)
            return True
        if fid == 7:
            n, _t = rr.list_header()
            out["n_column_orders"] = n
            for _ in range(n):
                order(rr)
            return True
        return False

    r.struct(visit)
    return out


def parse_column_index(buf, off):
    r = TC(buf, off)
    out = {"null_pages": [], "min": [], "max": []}

    def visit(fid, ft, rr):
        if fid == 1:
            n, _t = rr.list_header()
            out["null_pages"] = [rr.byte() == 1 for _ in range(n)]
            return True
        if fid == 2:
            n, _t = rr.list_header()
            out["min"] = [rr.binary() for _ in range(n)]
            return True
        if fid == 3:
            n, _t = rr.list_header()
            out["max"] = [rr.binary() for _ in range(n)]
            return True
        return False
    r.struct(visit)
    return out


# ---- the generating model (mirrors test_bolt_parquet_write_stats.cpp) -----
NAN, INF = float("nan"), float("inf")
MODEL = {
    "b": [b"\x01", b"A", b"\x7f", b"\x80", b"\xff\x01"],
    "d_mixed": [NAN, 1.5, -2.5, NAN, 3.0],
    "d_allnan": [NAN] * 5,
    "d_zero": [-0.0, +0.0, -0.0, +0.0, -0.0],
    "d_inf": [-INF, 0.5, INF, 0.5, 0.5],
}


def f64(raw):
    return struct.unpack("<d", raw)[0] if raw is not None and len(raw) == 8 \
        else None


def is_neg_zero(x):
    return x == 0.0 and math.copysign(1.0, x) < 0


def check(cols, errs):
    by_name = {c["name"]: c for c in cols}
    for name in MODEL:
        if name not in by_name:
            errs.append("column %r missing from footer" % name)
            return
    # -- G2PQ-21: unsigned byte-wise BYTE_ARRAY ordering ---------------------
    b = by_name["b"]
    st = b["stats"]
    if st is None or st["min_value"] is None or st["max_value"] is None:
        errs.append("b: min_value/max_value absent")
    else:
        umin, umax = min(MODEL["b"]), max(MODEL["b"])   # bytes = unsigned order
        smin = min(MODEL["b"], key=lambda v: [x - 256 if x >= 128 else x
                                              for x in v])
        smax = max(MODEL["b"], key=lambda v: [x - 256 if x >= 128 else x
                                              for x in v])
        assert (umin, umax) != (smin, smax), "fixture lost its discriminator"
        if st["min_value"] != umin:
            errs.append("b: min_value %r != unsigned min %r (signed order "
                        "would pick %r)" % (st["min_value"], umin, smin))
        if st["max_value"] != umax:
            errs.append("b: max_value %r != unsigned max %r (signed order "
                        "would pick %r)" % (st["max_value"], umax, smax))
    # -- G2PQ-22: NaN never a bound; mixed chunk uses finite extremes --------
    st = by_name["d_mixed"]["stats"]
    if st is None or st["min_value"] is None or st["max_value"] is None:
        errs.append("d_mixed: min_value/max_value absent")
    else:
        mn, mx = f64(st["min_value"]), f64(st["max_value"])
        if mn is None or math.isnan(mn) or mn != -2.5:
            errs.append("d_mixed: min %r, want -2.5 (NaN skipped)" % mn)
        if mx is None or math.isnan(mx) or mx != 3.0:
            errs.append("d_mixed: max %r, want 3.0 (NaN skipped)" % mx)
    # -- G2PQ-22: all-NaN chunk OMITS min/max entirely -----------------------
    st = by_name["d_allnan"]["stats"]
    if st is not None and (st["min_value"] is not None or
                           st["max_value"] is not None):
        errs.append("d_allnan: all-NaN chunk wrote min/max %r/%r -- must be "
                    "OMITTED" % (st["min_value"], st["max_value"]))
    # -- G2PQ-22: conservative signed zeros ---------------------------------
    st = by_name["d_zero"]["stats"]
    if st is None or st["min_value"] is None or st["max_value"] is None:
        errs.append("d_zero: min_value/max_value absent")
    else:
        mn, mx = f64(st["min_value"]), f64(st["max_value"])
        if mn != 0.0 or not is_neg_zero(mn):
            errs.append("d_zero: min %r must be -0.0 exactly" % mn)
        if mx != 0.0 or is_neg_zero(mx):
            errs.append("d_zero: max %r must be +0.0 exactly" % mx)
    # -- G2PQ-22: infinities are exact legal bounds --------------------------
    st = by_name["d_inf"]["stats"]
    if st is None or f64(st["min_value"]) != -INF or \
            f64(st["max_value"]) != INF:
        errs.append("d_inf: bounds %r not [-inf, inf]" % (st,))
    # -- deprecated fields 1/2 must be absent on EVERY chunk -----------------
    for c in cols:
        st = c["stats"]
        if st is not None and (st["dep_max"] is not None or
                               st["dep_min"] is not None):
            errs.append("%s: deprecated Statistics.min/max (fields 1/2) "
                        "written -- their legacy ordering differs; omit them"
                        % c["name"])


def check_orders(meta, errs):
    n = len(meta["columns"])
    if meta["n_column_orders"] != n or meta["n_type_order"] != n:
        errs.append("column_orders: %d entries / %d TYPE_ORDER for %d columns"
                    % (meta["n_column_orders"], meta["n_type_order"], n))


def check_column_index(buf, cols, errs):
    """The all-NaN page: empty min/max byte arrays, null_pages=false; and no
    NaN bytes in any double column's ColumnIndex bound."""
    for c in cols:
        if c["ci_off"] is None:
            errs.append("%s: no ColumnIndex (fixture asks for one)"
                        % c["name"])
            continue
        idx = parse_column_index(buf, c["ci_off"])
        for p in range(len(idx["min"])):
            mn, mx = idx["min"][p], idx["max"][p]
            if c["name"] == "d_allnan":
                if idx["null_pages"][p]:
                    errs.append("d_allnan page %d: null_pages=true but the "
                                "page holds NaN VALUES, not nulls" % p)
                if mn != b"" or mx != b"":
                    errs.append("d_allnan page %d: bound %r/%r -- an all-NaN "
                                "page's index entry must be empty" % (p, mn, mx))
            elif c["type"] == "DOUBLE":
                for raw, which in ((mn, "min"), (mx, "max")):
                    v = f64(raw)
                    if v is not None and math.isnan(v):
                        errs.append("%s page %d: NaN written as ColumnIndex %s"
                                    % (c["name"], p, which))


def run(path, mutate=None):
    buf = open(path, "rb").read()
    meta = parse_footer(buf)
    if mutate is not None:
        mutate(meta)
    errs = []
    check(meta["columns"], errs)
    check_orders(meta, errs)
    if mutate is None:                # index bytes can't be mutated via meta
        check_column_index(buf, meta["columns"], errs)
    return errs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    path = os.path.join(d, "test_bolt_parquet_stats_fixture.parquet")
    if not os.path.exists(path):
        print("fixture missing -- run test_bolt_parquet_write_stats first",
              file=sys.stderr)
        return 2

    errs = run(path)
    for e in errs[:20]:
        print("FAIL: %s" % e)
    if errs:
        return 1

    # Injections: each is a REAL writer-bug shape and each must be caught.
    def signed_order(meta):               # the classic memcmp-on-signed bug
        c = [x for x in meta["columns"] if x["name"] == "b"][0]
        c["stats"]["min_value"] = b"\x80"
        c["stats"]["max_value"] = b"\x7f"

    def nan_bound(meta):                  # naive min/max loop over NaNs
        c = [x for x in meta["columns"] if x["name"] == "d_mixed"][0]
        c["stats"]["min_value"] = struct.pack("<d", NAN)

    def allnan_written(meta):             # all-NaN chunk writes a bound
        c = [x for x in meta["columns"] if x["name"] == "d_allnan"][0]
        c["stats"] = {"dep_max": None, "dep_min": None, "null_count": 0,
                      "min_value": struct.pack("<d", NAN),
                      "max_value": struct.pack("<d", NAN)}

    def wrong_zero(meta):                 # +0.0 min: misses -0.0 on equality
        c = [x for x in meta["columns"] if x["name"] == "d_zero"][0]
        c["stats"]["min_value"] = struct.pack("<d", +0.0)

    def deprecated_written(meta):         # fields 1/2 with modern semantics
        c = [x for x in meta["columns"] if x["name"] == "b"][0]
        c["stats"]["dep_min"] = c["stats"]["min_value"]

    def orders_missing(meta):             # readers then distrust min_value
        meta["n_column_orders"] = meta["n_type_order"] = 0

    for label, mut in (("signed BYTE_ARRAY order", signed_order),
                       ("NaN as a bound", nan_bound),
                       ("all-NaN chunk wrote stats", allnan_written),
                       ("non-conservative signed zero", wrong_zero),
                       ("deprecated min/max written", deprecated_written),
                       ("column_orders missing", orders_missing)):
        if not run(path, mutate=mut):
            print("FAIL: injection '%s' was NOT caught -- this gate is not "
                  "discriminating" % label)
            return 1

    print("OK: chunk Statistics prove unsigned BYTE_ARRAY ordering on "
          ">=0x80 bytes, NaN never a bound, all-NaN omission, conservative "
          "signed zeros, exact infinities, deprecated fields absent, "
          "column_orders complete; ColumnIndex all-NaN page entry empty")
    print("    6 injections caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
