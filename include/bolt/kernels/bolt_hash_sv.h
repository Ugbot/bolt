// bolt_hash_sv.h — hash_bytes over StringView rows.
//
// hash_bytes_sv(rows, n, overflow, out): out[i] == hash_bytes(bytes of rows[i],
// rows[i].length) for every row, bit for bit, following BOLT_HASH_TIER.
// hash_bytes_sv_wyhash3_v1 selects the fixed persisted hash contract instead.
// An inline row (<= 12 bytes) is read from the view
// as one 8-byte word plus a masked tail, with no byte loop and no pointer
// chase; the length seed is recomputed only when the length changes. A
// spilled row goes through hash_bytes, at overflow + ref.offset: one
// overflow buffer for the whole column (ref.buf_idx is not consulted), the
// layout marbledb's sv_bytes assumes too.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_hash.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace detail {

BOLT_FORCE_INLINE uint64_t hash_sv_seed(uint32_t len) noexcept {
    return 0x9E3779B97F4A7C15ULL ^ swiss_mix_wyhash3(static_cast<uint64_t>(len));
}

template <bool FixedWyhash3>
BOLT_FORCE_INLINE uint64_t hash_sv_finalize(uint64_t state) noexcept {
    if constexpr (FixedWyhash3) return swiss_mix_wyhash3(state);
    else return swiss_mix(state);
}

template <bool FixedWyhash3>
BOLT_FORCE_INLINE uint64_t hash_sv_bytes(const void* bytes, size_t len) noexcept {
    return hash_sv_finalize<FixedWyhash3>(hash_bytes_accumulate_v1(bytes, len));
}

// The view's bytes 4..15 (prefix + inline_data) are always readable.
template <bool FixedWyhash3>
BOLT_FORCE_INLINE uint64_t hash_sv_inline(const StringView& v, uint64_t seed) noexcept {
    assert(v.length <= 12u);
    assert(seed == hash_sv_seed(v.length));
    static_assert(offsetof(StringView, prefix) + 12 == sizeof(StringView), "12 inline bytes");
    const auto* p = reinterpret_cast<const uint8_t*>(v.prefix);
    uint64_t h = seed;
    if (v.length >= 8u) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h = swiss_mix_wyhash3(h ^ w);
        if (v.length > 8u) {
            uint32_t t;
            std::memcpy(&t, p + 8, 4);
            const uint32_t rem = v.length - 8u;
            t &= static_cast<uint32_t>((uint64_t{1} << (rem * 8u)) - 1u);
            h = swiss_mix_wyhash3(h ^ t);
        }
    } else if (v.length != 0u) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        w &= (uint64_t{1} << (v.length * 8u)) - 1u;
        h = swiss_mix_wyhash3(h ^ w);
    }
    return hash_sv_finalize<FixedWyhash3>(h);
}

}  // namespace detail

template <bool FixedWyhash3>
BOLT_FORCE_INLINE uint64_t hash_bytes_sv_impl(const StringView& v, const void* overflow) noexcept {
    assert(v.length <= 12u || overflow != nullptr);
#if defined(BOLT_BIG_ENDIAN)
    const void* p = v.length <= 12u ? static_cast<const void*>(v.prefix)
                                    : static_cast<const uint8_t*>(overflow) + v.ref.offset;
    return detail::hash_sv_bytes<FixedWyhash3>(p, v.length);
#else
    if (v.length <= 12u) return detail::hash_sv_inline<FixedWyhash3>(v, detail::hash_sv_seed(v.length));
    return detail::hash_sv_bytes<FixedWyhash3>(static_cast<const uint8_t*>(overflow) + v.ref.offset, v.length);
#endif
}

template <bool FixedWyhash3>
inline void hash_bytes_sv_impl(const StringView* BOLT_RESTRICT rows, size_t n, const void* overflow,
                          uint64_t* BOLT_RESTRICT out) noexcept {
    assert(rows != nullptr || n == 0);
    assert(out != nullptr || n == 0);
#if defined(BOLT_BIG_ENDIAN)
    for (size_t i = 0; i < n; ++i) out[i] = hash_bytes_sv_impl<FixedWyhash3>(rows[i], overflow);   // bounded: n
#else
    constexpr uint32_t kWord = 8;   // 8-byte keys (encoded integers) take a two-mix path
    const uint64_t seed8 = detail::hash_sv_seed(kWord);
    uint32_t last_len = 0;
    uint64_t seed = detail::hash_sv_seed(0);
    for (size_t i = 0; i < n; ++i) {                     // bounded: n
        const StringView& v = rows[i];
        if (v.length == kWord) {
            uint64_t w;
            std::memcpy(&w, v.prefix, 8);
            out[i] = detail::hash_sv_finalize<FixedWyhash3>(swiss_mix_wyhash3(seed8 ^ w));
            continue;
        }
        if (v.length > 12u) {
            assert(overflow != nullptr);
            out[i] = detail::hash_sv_bytes<FixedWyhash3>(static_cast<const uint8_t*>(overflow) + v.ref.offset, v.length);
            continue;
        }
        if (v.length != last_len) { last_len = v.length; seed = detail::hash_sv_seed(last_len); }
        out[i] = detail::hash_sv_inline<FixedWyhash3>(v, seed);
    }
#endif
}

// Public generic entry points retain the configured digest and the existing
// optimized 8-byte/inline/overflow paths. Fixed v1 selects only the finalizer
// at compile time; there is no per-row runtime dispatch.
BOLT_FORCE_INLINE uint64_t hash_bytes_sv(const StringView& v, const void* overflow) noexcept {
    return hash_bytes_sv_impl<false>(v, overflow);
}
inline void hash_bytes_sv(const StringView* BOLT_RESTRICT rows, size_t n, const void* overflow,
                          uint64_t* BOLT_RESTRICT out) noexcept {
    hash_bytes_sv_impl<false>(rows, n, overflow, out);
}
BOLT_FORCE_INLINE uint64_t hash_bytes_sv_wyhash3_v1(const StringView& v, const void* overflow) noexcept {
    return hash_bytes_sv_impl<true>(v, overflow);
}
inline void hash_bytes_sv_wyhash3_v1(const StringView* BOLT_RESTRICT rows, size_t n, const void* overflow,
                                     uint64_t* BOLT_RESTRICT out) noexcept {
    hash_bytes_sv_impl<true>(rows, n, overflow, out);
}

}  // namespace bolt
