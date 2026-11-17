// bolt_json_column.h — columnar JSON path extraction (G2CHK-1198, logs).
//
// One compiled JsonPath over a Utf8 column of JSON documents -> a typed
// column plus a validity bitmap, for rows [0,n) or the rows named by a
// selection vector. Members the path does not name are skipped by
// json_path_lookup's byte scanner; no tape, no DOM, no per-row allocation, no
// StringView copies. A row is NULL (validity bit 0, output 0) when its
// document is malformed, the path is missing, or the value has the wrong type
// (strict integer for i64: "404" the string, 4.5 and 1e3 are NULL).
//
// Validity: bit i of `valid` (LSB first) answers output row i. The caller
// supplies ceil(n/8) bytes; the kernel writes every byte (no pre-clear needed)
// and returns the NULL count, so a caller can drop an all-valid bitmap.

#pragma once

#include <cassert>
#include <cstdint>

#include "bolt/bolt_types.h"
#include "bolt/kernels/bolt_json_path.h"
#include "bolt/kernels/bolt_utf8.h"

namespace bolt {
namespace kernels {
namespace json {

namespace detail {

// Shared row loop: `Conv(span, &out) -> bool` decides validity.
template <typename T, typename Conv>
BOLT_FORCE_INLINE int64_t json_column_loop(const JsonPath* path, const StringView* data,
                                           const char* spilled_base, const int32_t* sel,
                                           int64_t n, T* out, uint8_t* valid, Conv conv) noexcept {
    assert(path != nullptr && (n == 0 || (data != nullptr && out != nullptr && valid != nullptr)));
    int64_t nulls = 0;
    for (int64_t b = 0; b < n; b += 8) {                 // bounded: n
        uint8_t bits = 0;
        const int64_t e = (n - b < 8) ? n - b : 8;
        for (int64_t k = 0; k < e; ++k) {
            const int64_t i = b + k;
            const int32_t r = sel != nullptr ? sel[i] : static_cast<int32_t>(i);
            const StringView& sv = data[r];
            JsonSpan span{};
            T v{};
            const bool ok = json_path_lookup(path, utf8::sv_bytes(sv, spilled_base), sv.length, &span) &&
                            conv(span, &v);
            out[i] = ok ? v : T{};
            bits = static_cast<uint8_t>(bits | (static_cast<unsigned>(ok) << k));
            nulls += ok ? 0 : 1;
        }
        valid[b >> 3] = bits;
    }
    assert(nulls >= 0 && nulls <= n);
    return nulls;
}

}  // namespace detail

// json_extract_int over a column. Returns the NULL count.
BOLT_FORCE_INLINE int64_t json_column_extract_i64(const JsonPath* path, const StringView* data,
                                                  const char* spilled_base, const int32_t* sel,
                                                  int64_t n, int64_t* out, uint8_t* valid) noexcept {
    return detail::json_column_loop<int64_t>(path, data, spilled_base, sel, n, out, valid,
        [](const JsonSpan& s, int64_t* v) noexcept { return json_span_to_i64(s, v); });
}

// json_extract_double over a column. Returns the NULL count.
BOLT_FORCE_INLINE int64_t json_column_extract_f64(const JsonPath* path, const StringView* data,
                                                  const char* spilled_base, const int32_t* sel,
                                                  int64_t n, double* out, uint8_t* valid) noexcept {
    return detail::json_column_loop<double>(path, data, spilled_base, sel, n, out, valid,
        [](const JsonSpan& s, double* v) noexcept { return json_span_to_f64(s, v); });
}

// json_exists over a column: out[i] = 1 when the path resolves (a JSON null
// value counts as present), else 0. Never NULL for a non-NULL document.
BOLT_FORCE_INLINE void json_column_exists(const JsonPath* path, const StringView* data,
                                          const char* spilled_base, const int32_t* sel,
                                          int64_t n, int64_t* out) noexcept {
    assert(path != nullptr && (n == 0 || (data != nullptr && out != nullptr)));
    for (int64_t i = 0; i < n; ++i) {                    // bounded: n
        const int32_t r = sel != nullptr ? sel[i] : static_cast<int32_t>(i);
        JsonSpan span{};
        out[i] = json_path_lookup(path, utf8::sv_bytes(data[r], spilled_base), data[r].length, &span) ? 1 : 0;
    }
}

}  // namespace json
}  // namespace kernels
}  // namespace bolt
