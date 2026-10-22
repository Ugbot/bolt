// Strong copied object identity and bounded admission for borrowed versioned I/O.
#pragma once
#include "bolt/lakehouse/object_store/object_store_limits.h"
#include <cstddef>
#include <cstdint>
#include "bolt/bolt_arena.h"

namespace bolt::lakehouse {
struct ObjectVersion {
    uint64_t size;
    char etag[kOsStrongEtagCap]; // exact strong quoted validator, including quotes
};
inline bool os_object_version_valid(const ObjectVersion* version) noexcept {
    if (version == nullptr) return false;
    uint32_t n = 0;
    while (n < kOsStrongEtagCap && version->etag[n] != 0) ++n;
    if (n < 2 || n == kOsStrongEtagCap || version->etag[0] != '"' || version->etag[n - 1] != '"') return false;
    for (uint32_t i = 1; i + 1 < n; ++i) {
        const auto c = static_cast<unsigned char>(version->etag[i]);
        if (c < 0x21 || c == '"' || c == 127) return false;
    }
    return true;
}

namespace detail {
struct ObjectSpan { const void* data; size_t size; };
inline bool object_span_valid(ObjectSpan span) noexcept {
    return span.size == 0 || (span.data != nullptr &&
        span.size <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(span.data));
}
inline bool object_spans_overlap(ObjectSpan a, ObjectSpan b) noexcept {
    if (a.size == 0 || b.size == 0) return false;
    const auto x = reinterpret_cast<uintptr_t>(a.data), y = reinterpret_cast<uintptr_t>(b.data);
    return x < y + b.size && y < x + a.size;
}
inline bool object_key_span(const char* key, uint32_t cap, ObjectSpan* out) noexcept {
    if (!object_span_valid({key, cap})) return false;
    for (uint32_t i = 0; i < cap; ++i) {
        if (key[i] == 0) { *out = {key, static_cast<size_t>(i) + 1u}; return true; }
    }
    return false;
}
// Validate addresses before arithmetic; never write a rejected borrowed span.
// Input spans may share storage; each output is disjoint from all other spans.
inline bool object_io_admit(const ObjectSpan* inputs, uint32_t ni,
                            const ObjectSpan* outputs, uint32_t no, Arena* scratch,
                            const ObjectSpan* protected_spans, uint32_t np) noexcept {
    if (ni > 4 || no > 2 || np > 8) return false;
    for (uint32_t group = 0; group < 2; ++group) {
        const auto* spans = group == 0 ? inputs : outputs;
        const uint32_t count = group == 0 ? ni : no;
        for (uint32_t i = 0; i < count; ++i) {
            if (!object_span_valid(spans[i])) return false;
            if (scratch != nullptr && (object_spans_overlap(spans[i], {scratch, sizeof(Arena)}) ||
                scratch->overlaps_backing(spans[i].data, spans[i].size))) return false;
            for (uint32_t j = 0; j < np; ++j) {
                if (!object_span_valid(protected_spans[j]) ||
                    object_spans_overlap(spans[i], protected_spans[j])) return false;
            }
        }
    }
    for (uint32_t i = 0; i < no; ++i) {
        for (uint32_t j = 0; j < ni; ++j)
            if (object_spans_overlap(outputs[i], inputs[j])) return false;
        for (uint32_t j = 0; j < i; ++j)
            if (object_spans_overlap(outputs[i], outputs[j])) return false;
    }
    return true;
}
}  // namespace detail
}  // namespace bolt::lakehouse
