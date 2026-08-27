#!/usr/bin/env python3
"""Decode bolt's ColumnIndex/OffsetIndex from the SPEC and check them.

    ./test_bolt_parquet_write_dict           # writes the fixtures
    python3 scripts/parquet_pageindex_check.py <dir-containing-them>

WHY THIS EXISTS, and why it is not done with an existing reader.

A ColumnIndex is a skip structure: per page it declares min/max, a null flag
and a null count, and a reader uses those to decide which pages it can ignore.
Declare a page's bounds too narrow and any reader that trusts them skips a
page that really does match -- fewer rows, no error, nowhere to look.

Nothing external checked the CONTENTS. pyarrow 21 surfaces
has_column_index/has_offset_index but gives Python no way to read either
struct, so parquet_write_interop.py can only confirm the LOCATORS sit where
the spec puts them. That is worth having -- it proves the fields are right --
but it says nothing about the bytes they point at. The contents were left to
bolt's own index parser, and a writer and parser that share a
misunderstanding of the struct agree perfectly with each other.

DuckDB was tried as the oracle first and REJECTED, on evidence: flipping
every byte of the index region and re-running the same range queries returns
the identical answer, so DuckDB does not consult the ColumnIndex here and a
semantic query-level gate built on it would have been vacuous.

So this parses the thrift compact protocol directly, from the format
description, and checks what a skipping reader actually relies on:

  * every page's declared [min, max] BRACKETS every value in that page,
    with the page's row range taken from the OffsetIndex, not assumed;
  * null_pages and null_counts agree with the real data;
  * a declared boundary_order of ASCENDING/DESCENDING actually holds --
    a wrong one makes a reader's binary search skip matching pages.
"""
import os
import struct
import sys

import pyarrow.parquet as pq

# ---- thrift compact protocol ---------------------------------------------
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
        elif t == T_MAP:
            n = self.uvarint()
            if n:
                kt = self.byte()
                for _ in range(n):
                    self.skip(kt >> 4)
                    self.skip(kt & 0x0F)
        elif t == T_STRUCT:
            self.struct(lambda fid, ft, r: False)
        else:
            raise ValueError("bad thrift type %d" % t)

    def struct(self, visit):
        """visit(field_id, type, self) -> True if it consumed the value."""
        fid = 0
        while True:
            h = self.byte()
            if h == T_STOP:
                return
            delta, ft = h >> 4, h & 0x0F
            fid = fid + delta if delta else self.zigzag()
            if not visit(fid, ft, self):
                self.skip(ft)


def read_list(r, et, elem):
    n, t = r.list_header()
    return [elem(r, t) for _ in range(n)]


def parse_column_index(buf, off, ln):
    r = TC(buf, off)
    out = {"null_pages": [], "min": [], "max": [], "order": 0, "nulls": None}

    def visit(fid, ft, rr):
        if fid == 1:
            n, t = rr.list_header()
            # In compact protocol a boolean list element is one byte,
            # 1 = true / 2 = false.
            out["null_pages"] = [rr.byte() == 1 for _ in range(n)]
            return True
        if fid == 2:
            out["min"] = read_list(rr, ft, lambda x, t: x.binary())
            return True
        if fid == 3:
            out["max"] = read_list(rr, ft, lambda x, t: x.binary())
            return True
        if fid == 4:
            out["order"] = rr.zigzag()
            return True
        if fid == 5:
            out["nulls"] = read_list(rr, ft, lambda x, t: x.zigzag())
            return True
        return False

    r.struct(visit)
    return out


def parse_offset_index(buf, off, ln):
    r = TC(buf, off)
    pages = []

    def page(rr, _t):
        d = {}

        def v(fid, ft, x):
            if fid == 1:
                d["offset"] = x.zigzag()
                return True
            if fid == 2:
                d["size"] = x.zigzag()
                return True
            if fid == 3:
                d["first_row"] = x.zigzag()
                return True
            return False
        rr.struct(v)
        return d

    def visit(fid, ft, rr):
        if fid == 1:
            pages.extend(read_list(rr, ft, page))
            return True
        return False

    r.struct(visit)
    return pages


def parse_footer_locators(buf):
    """[(row_group, col, ci_off, ci_len, oi_off, oi_len)] from FileMetaData."""
    flen = struct.unpack("<I", buf[-8:-4])[0]
    r = TC(buf, len(buf) - 8 - flen)
    out = []

    def chunk(rr, _t, rg, ci):
        d = {"rg": rg, "col": ci}

        def v(fid, ft, x):
            for f, key in ((4, "oi_off"), (5, "oi_len"),
                           (6, "ci_off"), (7, "ci_len")):
                if fid == f:
                    d[key] = x.zigzag()
                    return True
            return False
        rr.struct(v)
        return d

    def row_group(rr, _t, rg):
        cols = []

        def v(fid, ft, x):
            if fid == 1:
                n, t = x.list_header()
                for ci in range(n):
                    cols.append(chunk(x, t, rg, ci))
                return True
            return False
        rr.struct(v)
        return cols

    def visit(fid, ft, rr):
        if fid == 4:
            n, t = rr.list_header()
            for rg in range(n):
                out.extend(row_group(rr, t, rg))
            return True
        return False

    r.struct(visit)
    return out


# ---- decoding a statistic to a comparable value ---------------------------
def decode_stat(raw, ptype):
    if ptype == "INT64":
        return struct.unpack("<q", raw)[0]
    if ptype == "INT32":
        return struct.unpack("<i", raw)[0]
    if ptype == "DOUBLE":
        return struct.unpack("<d", raw)[0]
    if ptype == "FLOAT":
        return struct.unpack("<f", raw)[0]
    if ptype == "BOOLEAN":
        return raw[0] != 0
    return raw          # BYTE_ARRAY: unsigned byte-wise, which is bytes' own


def as_key(v, ptype):
    if ptype == "BYTE_ARRAY" and isinstance(v, str):
        return v.encode()
    return v


def check_file(path, errs, corrupt=None):
    buf = open(path, "rb").read()
    locs = parse_footer_locators(buf)
    if not locs:
        return 0
    pf = pq.ParquetFile(path)
    md = pf.metadata
    table = pq.read_table(path)
    checked = 0

    rg_start, s = [], 0
    for r in range(md.num_row_groups):
        rg_start.append(s)
        s += md.row_group(r).num_rows

    for L in locs:
        if not L.get("ci_off") or not L.get("oi_off"):
            continue
        rg, ci = L["rg"], L["col"]
        ptype = str(md.row_group(rg).column(ci).physical_type)
        name = md.row_group(rg).column(ci).path_in_schema
        col = table.column(name).to_pylist()

        idx = parse_column_index(buf, L["ci_off"], L["ci_len"])
        pages = parse_offset_index(buf, L["oi_off"], L["oi_len"])
        if corrupt is not None:
            idx = corrupt(idx, ptype)
        tag = "%s rg%d %s" % (os.path.basename(path), rg, name)

        if len(idx["min"]) != len(pages):
            errs.append("%s: ColumnIndex has %d pages, OffsetIndex %d"
                        % (tag, len(idx["min"]), len(pages)))
            continue

        n_rg = md.row_group(rg).num_rows
        mins = []
        for p, pg in enumerate(pages):
            lo = rg_start[rg] + pg["first_row"]
            hi = (rg_start[rg] + pages[p + 1]["first_row"]
                  if p + 1 < len(pages) else rg_start[rg] + n_rg)
            vals = [v for v in col[lo:hi] if v is not None]
            nulls = (hi - lo) - len(vals)

            if idx["null_pages"][p] != (len(vals) == 0):
                errs.append("%s page %d: null_pages=%s but the page has %d "
                            "non-null values"
                            % (tag, p, idx["null_pages"][p], len(vals)))
            if idx["nulls"] is not None and idx["nulls"][p] != nulls:
                errs.append("%s page %d: null_count=%d, actual %d"
                            % (tag, p, idx["nulls"][p], nulls))
            if not vals:
                continue

            dmin = decode_stat(idx["min"][p], ptype)
            dmax = decode_stat(idx["max"][p], ptype)
            keys = [as_key(v, ptype) for v in vals]
            amin, amax = min(keys), max(keys)
            # THE assertion: bounds must BRACKET the page. Wider is legal
            # (imprecise but safe); narrower makes a reader skip real matches.
            if dmin > amin:
                errs.append("%s page %d: declared min %r > actual min %r -- a "
                            "reader would skip matching rows"
                            % (tag, p, dmin, amin))
            if dmax < amax:
                errs.append("%s page %d: declared max %r < actual max %r -- a "
                            "reader would skip matching rows"
                            % (tag, p, dmax, amax))
            mins.append((dmin, dmax))
            checked += 1

        # A declared ordering must hold, or a binary search over pages lands
        # in the wrong place and silently misses matches.
        order = idx["order"]
        if order == 1 and any(mins[i][0] > mins[i + 1][0]
                              for i in range(len(mins) - 1)):
            errs.append("%s: boundary_order=ASCENDING but page minima are not"
                        % tag)
        if order == 2 and any(mins[i][0] < mins[i + 1][0]
                              for i in range(len(mins) - 1)):
            errs.append("%s: boundary_order=DESCENDING but page minima are not"
                        % tag)
    return checked


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [f for f in sorted(os.listdir(d))
             if f.endswith(".parquet") and "parquet_write" in f]
    if not files:
        print("no fixtures in %s" % d, file=sys.stderr)
        return 2

    errs = []
    total = files_with_index = 0
    for f in files:
        n = check_file(os.path.join(d, f), errs)
        total += n
        if n:
            files_with_index += 1

    for e in errs[:20]:
        print("FAIL: %s" % e)
    if errs:
        return 1
    # Prove the comparison can fail. Every injection below is a real shape of
    # index bug, and each must be caught -- a gate that cannot distinguish a
    # narrowed bound from a correct one would report OK on any file at all.
    def narrow_min(idx, ptype):
        idx = dict(idx)
        idx["min"] = [struct.pack("<q", decode_stat(m, ptype) + 1)
                      if ptype == "INT64" else m for m in idx["min"]]
        return idx

    def narrow_max(idx, ptype):
        idx = dict(idx)
        idx["max"] = [struct.pack("<q", decode_stat(m, ptype) - 1)
                      if ptype == "INT64" else m for m in idx["max"]]
        return idx

    def bad_nulls(idx, ptype):
        idx = dict(idx)
        if idx["nulls"] is not None:
            idx["nulls"] = [n + 1 for n in idx["nulls"]]
        return idx

    def bad_order(idx, ptype):
        idx = dict(idx)
        idx["order"] = 2 if idx["order"] == 1 else 1
        return idx

    for label, mut in (("narrowed min", narrow_min), ("narrowed max", narrow_max),
                       ("wrong null_count", bad_nulls),
                       ("flipped boundary_order", bad_order)):
        inj = []
        for f in files:
            check_file(os.path.join(d, f), inj, corrupt=mut)
            if inj:
                break
        if not inj:
            print("FAIL: injection '%s' was NOT caught -- this gate is not "
                  "discriminating" % label)
            return 1

    if total == 0:
        print("FAIL: no page index was decoded from any fixture -- this gate "
              "did not examine anything")
        return 1
    print("OK: %d pages across %d fixtures; every declared bound brackets its "
          "page, null counts agree, declared orderings hold"
          % (total, files_with_index))
    print("    4 injections caught (narrowed min, narrowed max, wrong "
          "null_count, flipped boundary_order)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
