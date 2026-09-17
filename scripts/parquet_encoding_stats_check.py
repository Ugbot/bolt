#!/usr/bin/env python3
"""G2PQ-24: verify ColumnMetaData.encoding_stats. pyarrow does not expose
this field, so it is decoded here with a from-spec thrift compact reader
(same TC shape as parquet_pageindex_check.py) and cross-checked against the
independently-parsed OffsetIndex page count.

    ./test_bolt_parquet_write_encoding_stats
    python3 scripts/parquet_encoding_stats_check.py <dir-containing-fixtures>
"""
import os
import struct
import sys

T_STOP, T_I32, T_LIST, T_STRUCT = 0, 5, 9, 12
PAGE_TYPE = {0: "DATA_PAGE", 2: "DICTIONARY_PAGE", 3: "DATA_PAGE_V2"}
ENCODING = {0: "PLAIN", 3: "RLE", 8: "RLE_DICTIONARY"}


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
        if t in (1, 2):
            return
        if t == 3:
            self.byte()
        elif t in (4, 5, 6):
            self.zigzag()
        elif t == 7:
            self.p += 8
        elif t == 8:
            self.binary()
        elif t in (9, 10):
            n, et = self.list_header()
            for _ in range(n):
                self.skip(et) if et not in (1, 2) else self.byte()
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


def parse_encoding_stats(r):
    out = []

    def entry(fid, ft, x):
        d.setdefault("page_type", None)
        if fid == 1:
            d["page_type"] = x.zigzag()
            return True
        if fid == 2:
            d["encoding"] = x.zigzag()
            return True
        if fid == 3:
            d["count"] = x.zigzag()
            return True
        return False

    n, et = r.list_header()
    for _ in range(n):
        d = {}
        r.struct(entry)
        out.append((d.get("page_type"), d.get("encoding"), d.get("count")))
    return out


def parse_footer(buf):
    """[{name, encoding_stats, oi_off, oi_len}] across every chunk."""
    flen = struct.unpack("<I", buf[-8:-4])[0]
    r = TC(buf, len(buf) - 8 - flen)
    out = []

    def chunk(rr):
        d = {"name": None, "encoding_stats": None, "oi_off": None, "oi_len": None}

        def meta(fid, ft, x):
            if fid == 3:
                n, _t = x.list_header()
                d["name"] = ".".join(x.binary().decode() for _ in range(n))
                return True
            if fid == 13:
                d["encoding_stats"] = parse_encoding_stats(x)
                return True
            return False

        def v(fid, ft, x):
            if fid == 3:
                x.struct(meta)
                return True
            if fid == 4:
                d["oi_off"] = x.zigzag()
                return True
            if fid == 5:
                d["oi_len"] = x.zigzag()
                return True
            return False
        rr.struct(v)
        out.append(d)

    def row_group(rr):
        def v(fid, ft, x):
            if fid == 1:
                n, _t = x.list_header()
                for _ in range(n):
                    chunk(x)
                return True
            return False
        rr.struct(v)

    def visit(fid, ft, rr):
        if fid == 4:
            n, _t = rr.list_header()
            for _ in range(n):
                row_group(rr)
            return True
        return False
    r.struct(visit)
    return out


def parse_offset_index_page_count(buf, off, ln):
    r = TC(buf, off)
    n_pages = [0]

    def page(rr, _t):
        rr.struct(lambda a, b, c: False)

    def visit(fid, ft, rr):
        if fid == 1:
            n, et = rr.list_header()
            n_pages[0] = n
            for _ in range(n):
                page(rr, et)
            return True
        return False
    r.struct(visit)
    return n_pages[0]


def check(path):
    buf = open(path, "rb").read()
    cols = parse_footer(buf)
    errs = []

    if path.endswith("plain.parquet"):
        want = [(0, 0, 1)]   # DATA_PAGE, PLAIN, 1
        got = cols[0]["encoding_stats"]
        if got != want:
            errs.append(f"plain_one_page: got {got}, want {want}")

    elif path.endswith("dict.parquet"):
        want = [(2, 0, 1), (0, 8, 1)]   # DICTIONARY_PAGE/PLAIN/1, DATA_PAGE/RLE_DICTIONARY/1
        got = cols[0]["encoding_stats"]
        if got != want:
            errs.append(f"dict_col: got {got}, want {want}")

    elif path.endswith("multipage.parquet"):
        c = cols[0]
        stats = c["encoding_stats"] or []
        if len(stats) != 1 or stats[0][0] != 0 or stats[0][1] != 0:
            errs.append(f"multi_page: unexpected encoding_stats shape {stats}")
        else:
            claimed_count = stats[0][2]
            real_count = parse_offset_index_page_count(buf, c["oi_off"], c["oi_len"])
            if claimed_count <= 1:
                errs.append(f"multi_page: expected >1 page, encoding_stats claims {claimed_count}")
            if claimed_count != real_count:
                errs.append(
                    f"multi_page: encoding_stats count {claimed_count} != "
                    f"OffsetIndex page count {real_count}")

    return errs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    fixtures = [
        "test_bolt_parquet_write_encoding_stats_plain.parquet",
        "test_bolt_parquet_write_encoding_stats_dict.parquet",
        "test_bolt_parquet_write_encoding_stats_multipage.parquet",
    ]
    all_errs = []
    for name in fixtures:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            print(f"FAIL: missing fixture {path}")
            return 1
        errs = check(path)
        if errs:
            all_errs.extend(f"{name}: {e}" for e in errs)
        else:
            print(f"OK: {name}")

    if all_errs:
        for e in all_errs:
            print(f"FAIL: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
