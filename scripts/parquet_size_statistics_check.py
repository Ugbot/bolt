#!/usr/bin/env python3
"""G2PQ-25: verify ColumnMetaData.size_statistics (field 16). pyarrow 21's
ColumnChunkMetaData exposes no size_statistics attribute (checked directly,
not assumed), so this hand-decodes thrift the same self-contained way
scripts/parquet_encoding_stats_check.py does for field 13 (encoding_stats).

    ./test_bolt_parquet_write_size_statistics
    python3 scripts/parquet_size_statistics_check.py <dir-containing-fixtures>
"""
import os
import struct
import sys

T_STOP, T_I64, T_LIST, T_STRUCT = 0, 6, 9, 12


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


def parse_i64_list(r):
    n, et = r.list_header()
    assert et == T_I64, "expected list<i64>, got element type %d" % et
    return [r.zigzag() for _ in range(n)]


def parse_size_statistics(r):
    d = {"bytes": None, "rep_hist": None, "def_hist": None}

    def field(fid, ft, x):
        if fid == 1:
            d["bytes"] = x.zigzag()
            return True
        if fid == 2:
            d["rep_hist"] = parse_i64_list(x)
            return True
        if fid == 3:
            d["def_hist"] = parse_i64_list(x)
            return True
        return False
    r.struct(field)
    return d


def parse_footer(buf):
    """[{name, size_statistics}] across every chunk in row group 0."""
    flen = struct.unpack("<I", buf[-8:-4])[0]
    r = TC(buf, len(buf) - 8 - flen)
    out = []

    def chunk(rr):
        d = {"name": None, "size_statistics": None}

        def meta(fid, ft, x):
            if fid == 3:
                n, _t = x.list_header()
                d["name"] = ".".join(x.binary().decode() for _ in range(n))
                return True
            if fid == 16:
                d["size_statistics"] = parse_size_statistics(x)
                return True
            return False

        def v(fid, ft, x):
            if fid == 3:
                x.struct(meta)
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
            # Only row group 0 -- every fixture here writes exactly one.
            if n > 0:
                row_group(rr)
            return True
        return False
    r.struct(visit)
    return out


# ---- expected values, restated from the .cpp fixture builders -------------

def flat_plain_val(i):
    return "x" * ((i % 9) + 1)


def expect_flat_plain():
    return sum(len(flat_plain_val(i)) for i in range(50))


def expect_flat_dict():
    distinct = ["a", "bb", "ccc", "dddd", "eeeee"]
    return sum(len(distinct[i % 5]) for i in range(300))


# The 6-row LIST model shared by both list fixtures.
EXPECT_DEF_HIST = [1, 2, 2, 3]
EXPECT_REP_HIST = [6, 2]
EXPECT_LIST_BYTES = len("hello") + len("ab") + len("cde")   # 10


def check(path):
    buf = open(path, "rb").read()
    cols = parse_footer(buf)
    errs = []
    assert len(cols) == 1, f"{path}: expected 1 column, got {len(cols)}"
    ss = cols[0]["size_statistics"]

    if path.endswith("flat_plain.parquet"):
        want = expect_flat_plain()
        if ss is None:
            errs.append("size_statistics absent, expected byte count")
        else:
            if ss["bytes"] != want:
                errs.append(f"byte_array_data_bytes: got {ss['bytes']}, want {want}")
            if ss["rep_hist"] is not None or ss["def_hist"] is not None:
                errs.append(f"flat column must have no histogram, got {ss}")

    elif path.endswith("flat_dict.parquet"):
        want = expect_flat_dict()
        if ss is None:
            errs.append("size_statistics absent, expected byte count")
        elif ss["bytes"] != want:
            errs.append(
                f"byte_array_data_bytes: got {ss['bytes']}, want {want} "
                "(LOGICAL row count, not distinct dictionary entries)")

    elif path.endswith("flat_int64_absent.parquet"):
        if ss is not None:
            errs.append(f"expected size_statistics entirely absent, got {ss}")

    elif path.endswith("disabled.parquet"):
        if ss is not None:
            errs.append(f"emit_size_statistics=false but got {ss}")

    elif path.endswith("list_int64.parquet"):
        if ss is None:
            errs.append("size_statistics absent, expected histograms")
        else:
            if ss["def_hist"] != EXPECT_DEF_HIST:
                errs.append(f"def_hist: got {ss['def_hist']}, want {EXPECT_DEF_HIST}")
            if ss["rep_hist"] != EXPECT_REP_HIST:
                errs.append(f"rep_hist: got {ss['rep_hist']}, want {EXPECT_REP_HIST}")
            if ss["bytes"] is not None:
                errs.append(f"Int64 elements must have no byte count, got {ss['bytes']}")

    elif path.endswith("list_utf8.parquet"):
        if ss is None:
            errs.append("size_statistics absent, expected histograms + bytes")
        else:
            if ss["def_hist"] != EXPECT_DEF_HIST:
                errs.append(f"def_hist: got {ss['def_hist']}, want {EXPECT_DEF_HIST}")
            if ss["rep_hist"] != EXPECT_REP_HIST:
                errs.append(f"rep_hist: got {ss['rep_hist']}, want {EXPECT_REP_HIST}")
            if ss["bytes"] != EXPECT_LIST_BYTES:
                errs.append(f"byte_array_data_bytes: got {ss['bytes']}, want {EXPECT_LIST_BYTES}")

    return errs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    fixtures = [
        "test_bolt_parquet_write_size_statistics_flat_plain.parquet",
        "test_bolt_parquet_write_size_statistics_flat_dict.parquet",
        "test_bolt_parquet_write_size_statistics_flat_int64_absent.parquet",
        "test_bolt_parquet_write_size_statistics_disabled.parquet",
        "test_bolt_parquet_write_size_statistics_list_int64.parquet",
        "test_bolt_parquet_write_size_statistics_list_utf8.parquet",
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
