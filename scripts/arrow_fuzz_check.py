#!/usr/bin/env python3
"""arrow_fuzz_check.py — G2ARROW-8 (Tier 4): property/fuzz export vs REAL pyarrow.

Drives tests/arrow_fuzz_probe.cpp's `probe_fuzz_case`: for each (seed, iter)
the C++ side deterministically PLANS a randomized bolt::BoltColumn batch
(type x nullability x row count x per-row string-length, explicitly covering
the 0/12/13/large boundary around StringView's 12-byte inline/spill cutoff),
exports it via bolt::arrow::export_column, and hands back live ArrowSchema/
ArrowArray struct addresses. This script imports them with REAL pyarrow
(`pa.Array._import_from_c`), runs `validate(full=True)` (independently checks
offset monotonicity, buffer sizes, null counts), and compares every value
against an INDEPENDENTLY recomputed expectation.

No RNG state crosses the language boundary: every expected value is a pure
function of (seed, iter, row, salt) via mix64() below, which MUST match
arrow_fuzz_probe.cpp's mix64()/mix() byte-for-byte (both are plain 64-bit
integer ops -- add/xor/shift/multiply mod 2**64 -- so this is exact across
languages, no shared PRNG draw-order to keep in sync). If a case fails, the
(seed, iter, type) that produced it is printed so it can be reproduced by
rerunning ARROW_FUZZ_SEED=<seed> ARROW_FUZZ_ITER=<iter> against this script.

Exit codes: 0 = all iterations passed. 1 = a real mismatch was found. 2 = a
dependency (pyarrow, or the probe library) is unavailable -- ctest's
SKIP_RETURN_CODE 2 reports this as SKIPPED, never a silent pass.
"""
import ctypes
import datetime as dt
import decimal
import os
import struct
import sys

try:
    import pyarrow as pa
except ImportError:
    print("SKIP: pyarrow not installed")
    sys.exit(2)

_LIB = os.environ.get("ARROW_FUZZ_PROBE_LIB", "")
if not _LIB or not os.path.exists(_LIB):
    print(f"SKIP: probe library not found (ARROW_FUZZ_PROBE_LIB={_LIB!r})")
    sys.exit(2)

MASK64 = (1 << 64) - 1


def mix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return z


def mix(seed: int, iteration: int, salt: int) -> int:
    h = seed & MASK64
    h = mix64(h ^ ((iteration & MASK64) * 0x9E3779B97F4A7C15) & MASK64)
    h = mix64(h ^ ((salt & MASK64) * 0xC2B2AE3D27D4EB4F) & MASK64)
    return h


TYPES = [
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64", "Bool",
    "Utf8View", "Utf8VarBinary", "Binary",
    "Decimal128", "Date32", "Date64", "Timestamp", "Duration", "UUID",
    "UnsupportedFailClosed",
]
ROW_BUCKETS = [0, 1, 2, 11, 12, 13, 14, 50, None]  # None == "large" (500..1999)
LEN_BUCKETS = [0, 1, 11, 12, 13, 14, 50, None]     # None == "large" (200..499)

INT_TYPE_BITS = {  # name -> (bits, signed)
    "Int8": (8, True), "Int16": (16, True), "Int32": (32, True), "Int64": (64, True),
    "UInt8": (8, False), "UInt16": (16, False), "UInt32": (32, False), "UInt64": (64, False),
}


def wrap_int(v: int, bits: int, signed: bool) -> int:
    v &= (1 << bits) - 1
    if signed and v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def expected_double(seed, iteration, row):
    r = (mix(seed, iteration, row) % 2000000000) - 1000000000
    return r / 1000.0


def expected_float32(seed, iteration, row):
    # Emulate C++'s narrowing (float) cast: round the double to nearest f32.
    d = expected_double(seed, iteration, row)
    return struct.unpack("<f", struct.pack("<f", d))[0]


def gen_byte(seed, iteration, row, char_idx):
    h = mix(seed, iteration, 2000000 + row * 10007 + char_idx)
    return chr(ord("a") + (h % 26))


def pick_str_len(seed, iteration, row, forced_bucket):
    idx = forced_bucket if forced_bucket is not None and forced_bucket >= 0 else (
        mix(seed, iteration, 3000 + row) % len(LEN_BUCKETS)
    )
    v = LEN_BUCKETS[idx]
    if v is None:
        v = 200 + (mix(seed, iteration, 4000 + row) % 300)
    return v


def expected_string(seed, iteration, row, forced_len_bucket):
    n = pick_str_len(seed, iteration, row, forced_len_bucket)
    return "".join(gen_byte(seed, iteration, row, j) for j in range(n))


def pick_row_count(seed, iteration, forced_bucket):
    idx = forced_bucket if forced_bucket is not None and forced_bucket >= 0 else (
        mix(seed, iteration, -2) % len(ROW_BUCKETS)
    )
    v = ROW_BUCKETS[idx]
    if v is None:
        v = 500 + (mix(seed, iteration, -20) % 1500)
    return v


def pick_null_mode(seed, iteration):
    return mix(seed, iteration, -3) % 3


def row_is_null(null_mode, seed, iteration, row):
    if null_mode == 0:
        return False
    if null_mode == 2:
        return True
    return (mix(seed, iteration, row) & 1) == 0


def expected_decimal(seed, iteration, row):
    scale = mix(seed, iteration, -10) % 6
    mantissa = mix(seed, iteration, 5000 + row) % 1000000000
    return decimal.Decimal(mantissa).scaleb(-scale)


# --- ctypes wiring -----------------------------------------------------
L = ctypes.CDLL(_LIB)
L.probe_sizeof_schema.restype = ctypes.c_int64
L.probe_sizeof_array.restype = ctypes.c_int64
L.probe_fuzz_num_types.restype = ctypes.c_int64
NS, NA = L.probe_sizeof_schema(), L.probe_sizeof_array()
assert L.probe_fuzz_num_types() == len(TYPES), "TYPES table drifted from the probe's kTypes[]"

L.probe_fuzz_case.argtypes = [
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
]
L.probe_fuzz_case.restype = ctypes.c_int
L.probe_fuzz_reset_arena.argtypes = []


def pair():
    return ctypes.create_string_buffer(NS), ctypes.create_string_buffer(NA)


class SchemaView(ctypes.Structure):
    _fields_ = [("format", ctypes.c_void_p), ("name", ctypes.c_void_p),
                ("metadata", ctypes.c_void_p), ("flags", ctypes.c_int64),
                ("n_children", ctypes.c_int64), ("children", ctypes.c_void_p),
                ("dictionary", ctypes.c_void_p), ("release", ctypes.c_void_p),
                ("private_data", ctypes.c_void_p)]


fails = []
n_run = 0
n_skip_unsupported = 0


def run_case(seed, iteration, forced_type, forced_row_bucket, forced_len_bucket):
    global n_run, n_skip_unsupported
    sc, ar = pair()
    name_buf = ctypes.create_string_buffer(64)
    row_count = ctypes.c_int64(0)
    null_count = ctypes.c_int64(0)
    rc = L.probe_fuzz_case(
        seed, iteration,
        -1 if forced_type is None else forced_type,
        -1 if forced_row_bucket is None else forced_row_bucket,
        -1 if forced_len_bucket is None else forced_len_bucket,
        ctypes.addressof(sc), ctypes.addressof(ar),
        name_buf, len(name_buf),
        ctypes.byref(row_count), ctypes.byref(null_count),
    )
    type_name = name_buf.value.decode("ascii")
    tag = f"seed={seed} iter={iteration} type={type_name} forced_row={forced_row_bucket} forced_len={forced_len_bucket}"

    if type_name == "UnsupportedFailClosed":
        if rc != 2:
            fails.append(f"{tag}: expected fail-closed (rc=2), got rc={rc}")
        n_skip_unsupported += 1
        return

    if rc != 1:
        fails.append(f"{tag}: probe_fuzz_case returned {rc} (export unexpectedly failed)")
        return
    n_run += 1

    n = row_count.value
    a = pa.Array._import_from_c(ctypes.addressof(ar), ctypes.addressof(sc))
    try:
        a.validate(full=True)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the sweep
        fails.append(f"{tag}: validate(full=True) raised {e!r}")
        return

    if a.null_count != null_count.value:
        fails.append(f"{tag}: null_count mismatch: pyarrow={a.null_count} probe={null_count.value}")
    if len(a) != n:
        fails.append(f"{tag}: length mismatch: pyarrow={len(a)} expected={n}")
        return

    got = a.to_pylist()
    null_mode = pick_null_mode(seed, iteration)

    for i in range(n):
        expect_null = row_is_null(null_mode, seed, iteration, i)
        if expect_null:
            if got[i] is not None:
                fails.append(f"{tag} row={i}: expected null, got {got[i]!r}")
            continue
        if got[i] is None:
            fails.append(f"{tag} row={i}: expected non-null, got None")
            continue

        if type_name in INT_TYPE_BITS:
            bits, signed = INT_TYPE_BITS[type_name]
            want = wrap_int(mix(seed, iteration, i), bits, signed)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]} want {want}")
        elif type_name == "Float64":
            want = expected_double(seed, iteration, i)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Float32":
            want = expected_float32(seed, iteration, i)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Bool":
            want = (mix(seed, iteration, i) & 1) != 0
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]} want {want}")
        elif type_name in ("Utf8View", "Utf8VarBinary"):
            want = expected_string(seed, iteration, i, forced_len_bucket)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Binary":
            want = expected_string(seed, iteration, i, forced_len_bucket).encode("ascii")
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Decimal128":
            want = expected_decimal(seed, iteration, i)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Date32":
            want = dt.date(1970, 1, 1) + dt.timedelta(
                days=wrap_int(mix(seed, iteration, i) % 40000 - 20000, 32, True))
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Date64":
            days = (mix(seed, iteration, i) % 7300) - 3650
            want = dt.date(1970, 1, 1) + dt.timedelta(days=days)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Timestamp":
            us = (mix(seed, iteration, i) % (2 * 315360000000000)) - 315360000000000
            want = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=us)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "Duration":
            us = (mix(seed, iteration, i) % (2 * 315360000000000)) - 315360000000000
            want = dt.timedelta(microseconds=us)
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        elif type_name == "UUID":
            want = bytes(ord(gen_byte(seed, iteration, i, j)) for j in range(16))
            if got[i] != want:
                fails.append(f"{tag} row={i}: got {got[i]!r} want {want!r}")
        else:
            fails.append(f"{tag} row={i}: no comparator wired for type {type_name!r}")

    # Ownership: the export must survive the arena that produced it (release
    # contract + copy-not-alias, same property G2ARROW-3's audit pinned).
    if i_am_last_of_batch(seed, iteration):
        L.probe_fuzz_reset_arena()
        import gc
        gc.collect()
        got_after = a.to_pylist()
        if got_after != got:
            fails.append(f"{tag}: values changed after Arena::reset() (dangling export)")


def i_am_last_of_batch(seed, iteration):
    return iteration % 17 == 0  # periodic spot-check, not every case (keeps the sweep fast)


def main():
    seed = int(os.environ.get("ARROW_FUZZ_SEED", "424242"))
    only_iter = os.environ.get("ARROW_FUZZ_ITER")

    if only_iter is not None:
        # Reproduce a single reported failure exactly.
        it = int(only_iter)
        run_case(seed, it, None, None, None)
        if fails:
            print("REPRO FAILED:")
            for f in fails:
                print("  " + f)
            sys.exit(1)
        print(f"REPRO OK (seed={seed} iter={it})")
        sys.exit(0)

    # Phase 1: MUST-HIT boundary matrix -- every type crossed with every row
    # bucket, and every varlen type crossed with every string-length bucket.
    # Guaranteed by explicit bucket-index forcing, not by hoping random draws
    # land there. This is what makes 0/12/13/large a certainty, not a maybe.
    it = 0
    for type_idx in range(len(TYPES) - 1):  # skip UnsupportedFailClosed here
        for row_bucket in range(len(ROW_BUCKETS)):
            run_case(seed, it, type_idx, row_bucket, None)
            it += 1
    for type_idx in (11, 12, 13):  # Utf8View, Utf8VarBinary, Binary
        for len_bucket in range(len(LEN_BUCKETS)):
            run_case(seed, it, type_idx, 7, len_bucket)  # row bucket 7 == 50 rows
            it += 1
    for _ in range(5):  # the fail-closed path, a few times for good measure
        run_case(seed, it, len(TYPES) - 1, 3, None)
        it += 1

    # Phase 2: randomized sweep -- bounded iteration count so this stays
    # CI-affordable (hundreds, not millions), fully random plan per iteration.
    N_RANDOM = 400
    for i in range(N_RANDOM):
        run_case(seed, it, None, None, None)
        it += 1

    print(f"ran {n_run} exported cases + {n_skip_unsupported} fail-closed cases "
          f"({it} total probe_fuzz_case calls, seed={seed})")
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for f in fails[:50]:
            print("  " + f)
        if len(fails) > 50:
            print(f"  ... and {len(fails) - 50} more")
        print(f"\nReproduce with: ARROW_FUZZ_SEED={seed} ARROW_FUZZ_ITER=<iter> "
              f"ARROW_FUZZ_PROBE_LIB={_LIB} python3 {sys.argv[0]}")
        sys.exit(1)

    print(f"\narrow_fuzz_check: ALL OK (validated against pyarrow {pa.__version__})")


if __name__ == "__main__":
    main()
