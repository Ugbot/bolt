// arrow_fuzz_probe — G2ARROW-8 (Tier 4): property/fuzz export against pyarrow.
//
// Companion to tests/arrow_probe.cpp (Tier 3), which pins a fixed, hand-picked
// set of cases. This probe generates RANDOMIZED bolt::BoltColumn batches --
// varying type, nullability, row count, and (for Utf8/Binary) per-row string
// length, explicitly including the 0/12/13/large boundary around StringView's
// 12-byte inline-vs-spilled threshold (bolt_types.h: `is_inline() { return
// length <= 12; }`) -- exports each through the SAME bolt::arrow::export_column
// the whole Arrow C Data Interface rides on, and lets the Python driver
// (test_arrow_fuzz_pyarrow.py) validate with REAL pyarrow's validate(full=True)
// plus an independent value comparison.
//
// DESIGN: no shared mutable RNG state crosses the C++/Python boundary. Every
// generated value is a PURE FUNCTION of (seed, iter, row, salt) via mix64()
// below (a splitmix64-style avalanche finalizer over 64-bit integer ops --
// bit-identical in C++ and Python, since both use IEEE-754 doubles and
// twos-complement wraparound integer math on any platform this runs on). The
// Python driver reimplements mix64() once and recomputes "what SHOULD be at
// row i" independently, rather than mirroring a stateful sequential PRNG draw
// order -- which is the fragile way two implementations drift apart. This
// also means there is nothing to "consume" per draw: any row's expected value
// can be recomputed standalone, in either language, without replaying prior
// rows.
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_arrow.h"

using namespace bolt;

namespace {

Arena* g_arena = nullptr;
Arena& arena() { if (!g_arena) g_arena = new Arena(); return *g_arena; }

// splitmix64 finalizer -- must byte-for-byte match fuzz_common.py's mix64().
inline uint64_t mix64(uint64_t z) noexcept {
    z += 0x9E3779B97f4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z;
}

// mix(seed, iter, salt) -- the single pure-function generator every value in
// this file derives from. `salt` disambiguates independent draws within one
// (seed, iter) case (plan decisions use small negative-coded salts; per-row
// values use salt = row index or larger offsets for per-row sub-draws).
inline uint64_t mix(int64_t seed, int64_t iter, int64_t salt) noexcept {
    uint64_t h = static_cast<uint64_t>(seed);
    h = mix64(h ^ (static_cast<uint64_t>(iter) * 0x9E3779B97f4A7C15ULL));
    h = mix64(h ^ (static_cast<uint64_t>(salt) * 0xC2B2AE3D27D4EB4FULL));
    return h;
}

// Row-count buckets. Index chosen by mix(); must match fuzz_common.py exactly.
constexpr int64_t kRowBuckets[] = {0, 1, 2, 11, 12, 13, 14, 50, -1 /*large*/};
constexpr int kNumRowBuckets = 9;
// forced_bucket >= 0 pins an EXACT bucket index (used by the Python driver's
// must-hit boundary phase, so 0/12/13/large row counts are GUARANTEED to be
// exercised rather than merely probable over enough random iterations).
int64_t pick_row_count(int64_t seed, int64_t iter, int forced_bucket) {
    int idx = (forced_bucket >= 0) ? forced_bucket
                                    : static_cast<int>(mix(seed, iter, -2) % kNumRowBuckets);
    int64_t v = kRowBuckets[idx];
    if (v < 0) v = 500 + static_cast<int64_t>(mix(seed, iter, -20) % 1500);  // 500..1999
    return v;
}

// Per-row string-length buckets -- explicitly covers the StringView 12/13
// inline/spill boundary and 0 / large per the ticket.
constexpr int64_t kLenBuckets[] = {0, 1, 11, 12, 13, 14, 50, -1 /*large*/};
constexpr int kNumLenBuckets = 8;
// Same guaranteed-coverage mechanism as pick_row_count, but process-global
// since it must reach build_utf8_view/build_varbinary's per-row calls without
// threading a new parameter through every builder signature.
int g_forced_len_bucket = -1;
int64_t pick_str_len(int64_t seed, int64_t iter, int64_t row) {
    int idx = (g_forced_len_bucket >= 0) ? g_forced_len_bucket
                                          : static_cast<int>(mix(seed, iter, 3000 + row) % kNumLenBuckets);
    int64_t v = kLenBuckets[idx];
    if (v < 0) v = 200 + static_cast<int64_t>(mix(seed, iter, 4000 + row) % 300);  // 200..499
    return v;
}

// 0 = no validity buffer (all valid), 1 = mixed (parity of mix()), 2 = all null.
int pick_null_mode(int64_t seed, int64_t iter) {
    return static_cast<int>(mix(seed, iter, -3) % 3);
}
bool row_is_null(int null_mode, int64_t seed, int64_t iter, int64_t row) {
    if (null_mode == 0) return false;
    if (null_mode == 2) return true;
    return (mix(seed, iter, row) & 1u) == 0;
}

// Fill a byte with the SAME printable-ASCII rule as fuzz_common.py's
// expected_byte(): keeps every generated Utf8/Binary payload trivially valid
// UTF-8 too, so the same bytes can be round-tripped through either type.
char gen_byte(int64_t seed, int64_t iter, int64_t row, int64_t char_idx) {
    uint64_t h = mix(seed, iter, 2000000 + row * 10007 + char_idx);
    return static_cast<char>('a' + static_cast<int>(h % 26));
}

uint8_t* build_validity(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_null_count) {
    *out_null_count = 0;
    if (null_mode == 0 || n == 0) return nullptr;
    const size_t nb = static_cast<size_t>((n + 7) / 8);
    auto* v = static_cast<uint8_t*>(arena().allocate(nb, 64));
    std::memset(v, 0xFF, nb);
    for (int64_t i = 0; i < n; ++i) {
        if (row_is_null(null_mode, seed, iter, i)) {
            v[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
            ++(*out_null_count);
        }
    }
    return v;
}

template <typename T>
BoltColumn build_int_like(int64_t n, BoltType type, int null_mode, int64_t seed,
                          int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<T*>(arena().allocate(static_cast<size_t>(n) * sizeof(T), 64));
    for (int64_t i = 0; i < n; ++i) {
        d[i] = static_cast<T>(mix(seed, iter, i));  // truncation IS the type-range mapping
    }
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, type);
}

BoltColumn build_bool(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<uint8_t*>(arena().allocate(static_cast<size_t>(n), 64));
    for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(mix(seed, iter, i) & 1u);
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, BoltType::Bool);
}

// value = ((int64)(mix()%2000000000) - 1000000000) / 1000.0 -- must match
// fuzz_common.py's expected_double() term-for-term (same op order/widths).
double float_formula(int64_t seed, int64_t iter, int64_t row) {
    int64_t r = static_cast<int64_t>(mix(seed, iter, row) % 2000000000ULL) - 1000000000LL;
    return static_cast<double>(r) / 1000.0;
}

BoltColumn build_f64(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<double*>(arena().allocate(static_cast<size_t>(n) * 8, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = float_formula(seed, iter, i);
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, BoltType::Float64);
}
BoltColumn build_f32(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<float*>(arena().allocate(static_cast<size_t>(n) * 4, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(float_formula(seed, iter, i));
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, BoltType::Float32);
}

// Fixed-width time-ish types: int64 magnitude bounded to +-10y-ish so python
// datetime conversions never overflow. Date32 is int32 days.
int64_t bounded_i64(int64_t seed, int64_t iter, int64_t row, int64_t magnitude) {
    int64_t r = static_cast<int64_t>(mix(seed, iter, row) % static_cast<uint64_t>(2 * magnitude));
    return r - magnitude;
}
BoltColumn build_date32(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<int32_t*>(arena().allocate(static_cast<size_t>(n) * 4, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(bounded_i64(seed, iter, i, 20000));
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, BoltType::Date32);
}
BoltColumn build_i64_time(int64_t n, BoltType type, int null_mode, int64_t seed, int64_t iter,
                          int64_t magnitude, int64_t* out_nulls) {
    auto* d = static_cast<int64_t*>(arena().allocate(static_cast<size_t>(n) * 8, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = bounded_i64(seed, iter, i, magnitude);
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, type);
}
// Date64 stores milliseconds but the spec's unit is a whole day, so values
// must land on an 86400000ms boundary -- otherwise pyarrow's day-precision
// decode (python `datetime.date`) and our millisecond-precision expected
// value would silently disagree by construction, not because of a bug.
BoltColumn build_date64(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<int64_t*>(arena().allocate(static_cast<size_t>(n) * 8, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = bounded_i64(seed, iter, i, 3650) * 86400000LL;
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_flat(d, v, n, BoltType::Date64);
}

// Decimal128: small mantissa (fits low 64 bits, high 64 zero/sign-extended by
// omission since we never set them) + a small scale, mirrored exactly by
// fuzz_common.py's expected_decimal().
BoltColumn build_decimal(int64_t n, int null_mode, int64_t seed, int64_t iter, uint8_t* out_scale,
                         int64_t* out_nulls) {
    *out_scale = static_cast<uint8_t>(mix(seed, iter, -10) % 6);
    auto* d = static_cast<unsigned char*>(arena().allocate(static_cast<size_t>(n) * 16, 64));
    std::memset(d, 0, static_cast<size_t>(n) * 16);
    for (int64_t i = 0; i < n; ++i) {
        int64_t m = static_cast<int64_t>(mix(seed, iter, 5000 + i) % 1000000000ULL);
        std::memcpy(d + i * 16, &m, 8);
    }
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    auto c = BoltColumn::make_flat(d, v, n, BoltType::Decimal128);
    c.decimal_scale = *out_scale;
    c.type_size_bytes = 16;
    return c;
}

// UUID: 16 raw printable-ASCII bytes per row (structural FixedSizeBinary(16)
// test, not a real-UUID-format test -- see file header).
BoltColumn build_uuid(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* d = static_cast<unsigned char*>(arena().allocate(static_cast<size_t>(n) * 16, 64));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < 16; ++j) {
            d[i * 16 + j] = static_cast<unsigned char>(gen_byte(seed, iter, i, j));
        }
    }
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    auto c = BoltColumn::make_flat(d, v, n, BoltType::UUID);
    c.type_size_bytes = 16;
    return c;
}

// Utf8 in the Flat/StringView layout (mirrors test_bolt_arrow_export.cpp's
// make_utf8_view / arrow_probe.cpp's probe_utf8_stringview).
BoltColumn build_utf8_view(int64_t n, int null_mode, int64_t seed, int64_t iter, int64_t* out_nulls) {
    auto* views = static_cast<StringView*>(arena().allocate(static_cast<size_t>(n) * sizeof(StringView), 64));
    std::memset(views, 0, static_cast<size_t>(n) * sizeof(StringView));
    std::string spill;
    for (int64_t i = 0; i < n; ++i) {
        int64_t len = pick_str_len(seed, iter, i);
        views[i].length = static_cast<uint32_t>(len);
        if (len <= 12) {
            // prefix[4] + inline_data[8] are contiguous struct members (no
            // padding between them), so a single len<=12 memcpy starting at
            // prefix correctly spans both -- same technique as
            // test_bolt_arrow_export.cpp's make_utf8_view.
            char tmp[12];
            for (int64_t j = 0; j < len; ++j) tmp[j] = gen_byte(seed, iter, i, j);
            std::memcpy(views[i].prefix, tmp, static_cast<size_t>(len));
        } else {
            char tmp4[4];
            for (int64_t j = 0; j < 4; ++j) tmp4[j] = gen_byte(seed, iter, i, j);
            std::memcpy(views[i].prefix, tmp4, 4);
            views[i].ref.buf_idx = 0;
            views[i].ref.offset = static_cast<uint32_t>(spill.size());
            for (int64_t j = 0; j < len; ++j) spill.push_back(gen_byte(seed, iter, i, j));
        }
    }
    auto* base = static_cast<char*>(arena().allocate(spill.size() + 1, 64));
    std::memcpy(base, spill.data(), spill.size());
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    auto c = BoltColumn::make_flat(views, v, n, BoltType::Utf8);
    c.str_overflow_base = base;
    return c;
}

// Utf8/Binary in the VarBinary layout (offsets + packed bytes).
BoltColumn build_varbinary(int64_t n, BoltType type, int null_mode, int64_t seed, int64_t iter,
                           int64_t* out_nulls) {
    auto* offs = static_cast<int32_t*>(arena().allocate(static_cast<size_t>(n + 1) * 4, 64));
    offs[0] = 0;
    std::string packed;
    for (int64_t i = 0; i < n; ++i) {
        int64_t len = pick_str_len(seed, iter, i);
        for (int64_t j = 0; j < len; ++j) packed.push_back(gen_byte(seed, iter, i, j));
        offs[i + 1] = static_cast<int32_t>(packed.size());
    }
    auto* bytes = static_cast<uint8_t*>(arena().allocate(packed.size() + 1, 64));
    std::memcpy(bytes, packed.data(), packed.size());
    uint8_t* v = build_validity(n, null_mode, seed, iter, out_nulls);
    return BoltColumn::make_var_binary(bytes, v, offs, n, type, &arena());
}

// Deliberately-unsupported case: export_column MUST fail closed (return
// false), never export something a consumer will misread.
BoltColumn build_unsupported(int64_t n) {
    auto* d = static_cast<int64_t*>(arena().allocate(static_cast<size_t>(n) * 8, 64));
    for (int64_t i = 0; i < n; ++i) d[i] = i;
    auto c = BoltColumn::make_flat(d, nullptr, n, BoltType::Int64);
    c.format = ColumnFormat::Dictionary;   // not resolvable as Flat/View
    return c;
}

struct TypeEntry { const char* name; };
constexpr TypeEntry kTypes[] = {
    {"Int8"}, {"Int16"}, {"Int32"}, {"Int64"},
    {"UInt8"}, {"UInt16"}, {"UInt32"}, {"UInt64"},
    {"Float32"}, {"Float64"}, {"Bool"},
    {"Utf8View"}, {"Utf8VarBinary"}, {"Binary"},
    {"Decimal128"}, {"Date32"}, {"Date64"}, {"Timestamp"}, {"Duration"}, {"UUID"},
    {"UnsupportedFailClosed"},
};
constexpr int kNumTypes = 21;

}  // namespace

extern "C" {

int64_t probe_fuzz_num_types() { return kNumTypes; }
int64_t probe_fuzz_num_row_buckets() { return kNumRowBuckets; }
int64_t probe_fuzz_num_len_buckets() { return kNumLenBuckets; }

// Force-select a specific type index for a given (seed,iter) -- used by the
// Python driver's first K "must-hit boundary" iterations so the 0/12/13/large
// axis is guaranteed to appear regardless of what the random plan lands on,
// instead of merely hoping enough iterations cover it. type_idx < 0 means
// "let mix() pick it" (the normal randomized path).
int probe_fuzz_case(int64_t seed, int64_t iter, int64_t forced_type_idx,
                    int64_t forced_row_bucket, int64_t forced_len_bucket,
                    ArrowSchema* sc, ArrowArray* ar,
                    char* type_name_out, int type_name_cap,
                    int64_t* row_count_out, int64_t* null_count_out) {
    if (sc == nullptr || ar == nullptr || type_name_out == nullptr ||
        row_count_out == nullptr || null_count_out == nullptr) {
        return 0;
    }
    g_forced_len_bucket = static_cast<int>(forced_len_bucket);
    int type_idx = (forced_type_idx >= 0)
                       ? static_cast<int>(forced_type_idx)
                       : static_cast<int>(mix(seed, iter, -1) % kNumTypes);
    std::snprintf(type_name_out, static_cast<size_t>(type_name_cap), "%s", kTypes[type_idx].name);

    if (type_idx == 20) {  // UnsupportedFailClosed
        int64_t n = pick_row_count(seed, iter, static_cast<int>(forced_row_bucket));
        auto c = build_unsupported(n);
        *row_count_out = n;
        *null_count_out = 0;
        bool exported = bolt::arrow::export_column(c, n, "f", sc, ar);
        return exported ? 0 : 2;   // 2 == correctly refused; 0 would be a real bug here
    }

    int64_t n = pick_row_count(seed, iter, static_cast<int>(forced_row_bucket));
    int null_mode = pick_null_mode(seed, iter);
    int64_t nulls = 0;
    BoltColumn c;
    switch (type_idx) {
        case 0:  c = build_int_like<int8_t>(n, BoltType::Int8, null_mode, seed, iter, &nulls); break;
        case 1:  c = build_int_like<int16_t>(n, BoltType::Int16, null_mode, seed, iter, &nulls); break;
        case 2:  c = build_int_like<int32_t>(n, BoltType::Int32, null_mode, seed, iter, &nulls); break;
        case 3:  c = build_int_like<int64_t>(n, BoltType::Int64, null_mode, seed, iter, &nulls); break;
        case 4:  c = build_int_like<uint8_t>(n, BoltType::UInt8, null_mode, seed, iter, &nulls); break;
        case 5:  c = build_int_like<uint16_t>(n, BoltType::UInt16, null_mode, seed, iter, &nulls); break;
        case 6:  c = build_int_like<uint32_t>(n, BoltType::UInt32, null_mode, seed, iter, &nulls); break;
        case 7:  c = build_int_like<uint64_t>(n, BoltType::UInt64, null_mode, seed, iter, &nulls); break;
        case 8:  c = build_f32(n, null_mode, seed, iter, &nulls); break;
        case 9:  c = build_f64(n, null_mode, seed, iter, &nulls); break;
        case 10: c = build_bool(n, null_mode, seed, iter, &nulls); break;
        case 11: c = build_utf8_view(n, null_mode, seed, iter, &nulls); break;
        case 12: c = build_varbinary(n, BoltType::Utf8, null_mode, seed, iter, &nulls); break;
        case 13: c = build_varbinary(n, BoltType::Binary, null_mode, seed, iter, &nulls); break;
        case 14: { uint8_t sc_; c = build_decimal(n, null_mode, seed, iter, &sc_, &nulls); break; }
        case 15: c = build_date32(n, null_mode, seed, iter, &nulls); break;
        case 16: c = build_date64(n, null_mode, seed, iter, &nulls); break;
        case 17: c = build_i64_time(n, BoltType::Timestamp, null_mode, seed, iter, 315360000000000LL, &nulls); break;
        case 18: c = build_i64_time(n, BoltType::Duration, null_mode, seed, iter, 315360000000000LL, &nulls); break;
        case 19: c = build_uuid(n, null_mode, seed, iter, &nulls); break;
        default: return 0;
    }
    *row_count_out = n;
    *null_count_out = nulls;
    return bolt::arrow::export_column(c, n, "f", sc, ar) ? 1 : 0;
}

int64_t probe_sizeof_schema() { return static_cast<int64_t>(sizeof(ArrowSchema)); }
int64_t probe_sizeof_array()  { return static_cast<int64_t>(sizeof(ArrowArray)); }
void probe_fuzz_reset_arena() { if (g_arena) g_arena->reset(); }

}  // extern "C"
