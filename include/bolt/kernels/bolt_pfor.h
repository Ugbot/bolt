// bolt/kernels/bolt_pfor.h - PFOR-Delta scalar block-pack/unpack kernels.
//
// Layer 2 substrate of plan `this-was-a-freach-hashed-crab.md`. Lucene104-shape
// PFOR (Patched Frame-Of-Reference) over fixed 128-value uint32 blocks. Each
// block picks the smallest bits-per-value such that <=7 outliers remain; those
// outliers are stored as (idx_byte, high_byte) patches after the bit-packed
// payload. The high byte caps the patch range at bpv+8 bits.
//
// Reference (locked by docs/research/posting-codec-lucene90.md):
//   - Lemire & Boytsov 2015, "Decoding billions of integers per second through
//     vectorization". arXiv:1209.2137.
//   - Zukowski et al. 2006, "Super-scalar RAM-CPU cache compression" (PFOR).
//   - Apache Lucene 10.4 PForUtil.java / ForUtil.java.
//
// Block layout in bytes (no headers, just back-to-back blocks):
//   [token : 1B][bitpacked payload : ceil(bpv*128/8) B][exceptions : 2*numExc B]
// Token byte = (numExceptions << 5) | bpv  -- 5 bits bpv (0..31), 3 bits numExc (0..7).
// numExc=0..7, bpv=0..32 (bpv=32 stored as 0 in token with a special flag? No --
// we cap at 31 fitted in 5 bits + the 8-bit patch ceiling, so a true 32-bit value
// always becomes a patch on bpv=24. Bpv=32 with zero exceptions only arises if
// some value's low bits already need 32; we then bypass the patch-channel and
// store bpv=31 with 1 exception holding the top 1 bit. Because bpv field is 5
// bits, 31 is the max stored bpv.
//
// Tiger Style:
//   - noexcept, no exceptions, no RTTI, no virtual.
//   - No std:: containers, no smart pointers, no allocations on the hot path.
//   - All scratch lives on the stack at fixed kBlockSize footprint.
//   - >=2 assertions on every public function, <=70 lines per function.
//   - BOLT_FORCE_INLINE on hot paths, BOLT_RESTRICT on kernel pointer params.
//   - The unpack inner loop is branchless: a single bpv-parameterised loop, no
//     switch, no data-dependent branches. The mask is computed once outside.
//   - Bpv=0 fast path (all values zero) skips the payload entirely.
//   - SIMD overlay hook: a future bolt_pfor_avx2.h can replace pack_bits/
//     unpack_bits via the same signatures without touching the public API.
//
// Research note: docs/research/pfor-bitpack-kernels.md.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstdint>
#include <cstring>

// SIMD overlay (Layer 2.1b). The header expands to nothing useful when
// BOLT_SIMD_AVX2 / BOLT_SIMD_SSE42 are both 0, leaving the scalar path
// in sole control. Bit-exact equivalence with scalar is asserted by
// `tests/test_bolt_pfor_simd.cpp`.
#include "bolt/kernels/bolt_pfor_simd.h"

namespace bolt {
namespace kernels {
namespace pfor {

// --- Locked constants ------------------------------------------------------

inline constexpr int32_t kBlockSize     = 128;
inline constexpr uint8_t kMaxExceptions = 7;
inline constexpr int32_t kMaxBitWidth   = 32;
inline constexpr uint8_t kStoredBpvMax  = 31;  // 5-bit field cap.
inline constexpr int32_t kPatchHighBits = 8;   // patch byte holds 8 high bits.

// Worst-case packed size: 1B token + 32 bits/value * 128 values + 7*2B patches.
inline constexpr int32_t kPforMaxPackedBytes =
    1 + (kMaxBitWidth * kBlockSize / 8) + 2 * kMaxExceptions;

// Required slack at the END of the output buffer used by pack_bits/unpack_bits:
// the inner loop loads/stores 8 bytes spanning the bit window, so the last
// iteration may touch up to 7 bytes past `byte_off`. Callers ensure space.
inline constexpr int32_t kBitpackTailSlack = 8;

// --- Internal helpers ------------------------------------------------------

// Bit width of a uint32 (0 -> 0, 1 -> 1, 2..3 -> 2, ...). Branchless via clz.
BOLT_FORCE_INLINE uint8_t pfor_bit_width(uint32_t v) noexcept {
    // bolt_clz32(0) is impl-defined; guard on the zero case.
    return v == 0u ? uint8_t{0}
                   : static_cast<uint8_t>(32 - bolt_clz32(v));
}

// Bytes that the payload of `n` values at `bpv` bits/value occupies.
BOLT_FORCE_INLINE int32_t pfor_payload_bytes(int32_t n, uint8_t bpv) noexcept {
    return (n * static_cast<int32_t>(bpv) + 7) >> 3;
}

// --- pack_bits / unpack_bits -----------------------------------------------
//
// `internal::pack_bits_scalar` / `internal::unpack_bits_scalar` are the
// branchless scalar bodies. They are exposed in this header (instead of
// hidden in an anonymous namespace) so the SIMD overlay test
// (`tests/test_bolt_pfor_simd.cpp`) can call the scalar path explicitly
// and assert byte-for-byte equality with the SIMD path. The load-bearing
// correctness gate is "SIMD packed bytes == scalar packed bytes" — any
// drift breaks every downstream consumer of the wire format.
namespace internal {

// Pure scalar bit-packer. `bpv` is loaded once and the inner loop has no
// branches. `out` MUST be pre-zeroed for the bytes we will OR into.
BOLT_FORCE_INLINE void pack_bits_scalar(const uint32_t* BOLT_RESTRICT in,
                                         int32_t n_values, uint8_t bpv,
                                         uint8_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    assert(n_values >= 0);
    assert(bpv <= 32u);
    if (bpv == 0u) { return; }
    if (bpv == 32u) {
        std::memcpy(out, in, static_cast<size_t>(n_values) * 4u);
        return;
    }
    const uint64_t mask = (1ull << bpv) - 1ull;
    for (int32_t i = 0; i < n_values; ++i) {
        const int32_t bit_off  = i * static_cast<int32_t>(bpv);
        const int32_t byte_off = bit_off >> 3;
        const int32_t shift    = bit_off & 7;
        const uint64_t v       = static_cast<uint64_t>(in[i]) & mask;
        uint64_t w;
        std::memcpy(&w, out + byte_off, 8);
        w |= v << shift;
        std::memcpy(out + byte_off, &w, 8);
    }
}

// Pure scalar bit-unpacker. Branchless inner loop.
BOLT_FORCE_INLINE void unpack_bits_scalar(const uint8_t* BOLT_RESTRICT in,
                                           int32_t n_values, uint8_t bpv,
                                           uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    assert(n_values >= 0);
    assert(bpv <= 32u);
    if (bpv == 0u) {
        std::memset(out, 0, static_cast<size_t>(n_values) * 4u);
        return;
    }
    if (bpv == 32u) {
        std::memcpy(out, in, static_cast<size_t>(n_values) * 4u);
        return;
    }
    const uint64_t mask = (1ull << bpv) - 1ull;
    for (int32_t i = 0; i < n_values; ++i) {
        const int32_t bit_off  = i * static_cast<int32_t>(bpv);
        const int32_t byte_off = bit_off >> 3;
        const int32_t shift    = bit_off & 7;
        uint64_t w;
        std::memcpy(&w, in + byte_off, 8);
        out[i] = static_cast<uint32_t>((w >> shift) & mask);
    }
}

}  // namespace internal

// Public bit-packer. Tries the SIMD overlay first (compile-time gated, no
// runtime branch in the inner loop) and falls through to the scalar path on
// `false` return (unsupported bpv / shape). `out` MUST be pre-zeroed for the
// bytes we will OR into. `out` MUST have at least pfor_payload_bytes(n_values,
// bpv) + kBitpackTailSlack bytes addressable for the scalar 8-byte
// read/modify/write window.
BOLT_FORCE_INLINE void pack_bits(const uint32_t* BOLT_RESTRICT in,
                                  int32_t n_values, uint8_t bpv,
                                  uint8_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
#if BOLT_SIMD_AVX2
    if (bolt::kernels::pfor::simd::pack_avx2(in, n_values, bpv, out)) { return; }
#elif BOLT_SIMD_SSE42
    if (bolt::kernels::pfor::simd::pack_sse42(in, n_values, bpv, out)) { return; }
#endif
    internal::pack_bits_scalar(in, n_values, bpv, out);
}

// Public bit-unpacker. SIMD-first dispatch with scalar fallback. `in` MUST
// have at least pfor_payload_bytes(n_values, bpv) + kBitpackTailSlack bytes
// addressable for the scalar 8-byte read window.
BOLT_FORCE_INLINE void unpack_bits(const uint8_t* BOLT_RESTRICT in,
                                    int32_t n_values, uint8_t bpv,
                                    uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
#if BOLT_SIMD_AVX2
    if (bolt::kernels::pfor::simd::unpack_avx2(in, n_values, bpv, out)) { return; }
#elif BOLT_SIMD_SSE42
    if (bolt::kernels::pfor::simd::unpack_sse42(in, n_values, bpv, out)) { return; }
#endif
    internal::unpack_bits_scalar(in, n_values, bpv, out);
}

// --- Block pack ------------------------------------------------------------
//
// Histograms the 128 values' bit-widths, picks smallest bpv such that
// count(values > 2^bpv - 1) <= kMaxExceptions, then writes
//   [token][payload][numExc * (idx, high)]
// Returns bytes written, or 0 if `out_cap` is insufficient.
BOLT_FORCE_INLINE int32_t pfor_pack_block(const uint32_t* BOLT_RESTRICT in,
                                           uint8_t* BOLT_RESTRICT out,
                                           int32_t out_cap) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    assert(out_cap >= 0);

    // Histogram: hist[k] = # values with bit_width == k, for k in [0..32].
    // Then suffix-sum to get count_above[b] = #values with bit_width > b.
    int32_t hist[33] = {0};
    for (int32_t i = 0; i < kBlockSize; ++i) {
        hist[pfor_bit_width(in[i])] += 1;
    }
    // count_above[b] = sum_{k>b} hist[k]. count_above[32] = 0.
    int32_t count_above[33];
    count_above[32] = 0;
    for (int32_t b = 31; b >= 0; --b) {
        count_above[b] = count_above[b + 1] + hist[b + 1];
    }

    // Pick smallest bpv such that count_above[bpv] <= kMaxExceptions AND
    // every outlier fits in (bpv + kPatchHighBits) bits.
    uint8_t bpv = 0u;
    for (int32_t b = 0; b <= static_cast<int32_t>(kStoredBpvMax); ++b) {
        if (count_above[b] <= kMaxExceptions) {
            // Outliers must have bit_width <= b + 8 to fit the patch byte.
            // i.e. count_above[b + kPatchHighBits] must be 0.
            const int32_t cap_b = b + kPatchHighBits;
            const int32_t over  = cap_b >= 32 ? 0 : count_above[cap_b];
            if (over == 0) { bpv = static_cast<uint8_t>(b); break; }
        }
    }
    const uint8_t num_exc = static_cast<uint8_t>(count_above[bpv]);
    assert(num_exc <= kMaxExceptions);

    const int32_t payload_bytes = pfor_payload_bytes(kBlockSize, bpv);
    const int32_t total_bytes   = 1 + payload_bytes + 2 * num_exc;
    if (out_cap < total_bytes) { return 0; }

    out[0] = static_cast<uint8_t>((num_exc << 5) | bpv);
    if (payload_bytes > 0) {
        std::memset(out + 1, 0, static_cast<size_t>(payload_bytes));
    }

    // Pack low `bpv` bits of every value; collect outlier indices+high bytes.
    uint8_t patches[2 * kMaxExceptions];
    uint8_t patch_idx = 0u;
    const uint64_t mask = bpv == 32u ? ~0ull : ((1ull << bpv) - 1ull);
    for (int32_t i = 0; i < kBlockSize; ++i) {
        const uint32_t v = in[i];
        if (v > mask) {
            const uint32_t high = v >> bpv;
            assert(high <= 0xFFu);  // 8-bit patch ceiling.
            patches[2 * patch_idx + 0] = static_cast<uint8_t>(i);
            patches[2 * patch_idx + 1] = static_cast<uint8_t>(high);
            patch_idx += 1u;
        }
    }
    assert(patch_idx == num_exc);

    if (payload_bytes > 0) {
        // We need kBitpackTailSlack bytes of writable scratch past payload.
        // The patches overwrite that scratch later, so use a temp stack buffer.
        uint8_t scratch[kBlockSize * 4 + kBitpackTailSlack] = {0};
        pack_bits(in, kBlockSize, bpv, scratch);
        std::memcpy(out + 1, scratch, static_cast<size_t>(payload_bytes));
    }
    if (num_exc > 0u) {
        std::memcpy(out + 1 + payload_bytes, patches,
                    static_cast<size_t>(2 * num_exc));
    }
    return total_bytes;
}

// --- Block unpack ----------------------------------------------------------
//
// Returns bytes consumed, or 0 on malformed input.
BOLT_FORCE_INLINE int32_t pfor_unpack_block(const uint8_t* BOLT_RESTRICT in,
                                             int32_t in_cap,
                                             uint32_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    assert(in_cap >= 0);
    if (in_cap < 1) { return 0; }

    const uint8_t token   = in[0];
    const uint8_t bpv     = token & 0x1Fu;          // 5 bits.
    const uint8_t num_exc = (token >> 5) & 0x07u;   // 3 bits.
    if (bpv > kStoredBpvMax) { return 0; }
    if (num_exc > kMaxExceptions) { return 0; }

    const int32_t payload_bytes = pfor_payload_bytes(kBlockSize, bpv);
    const int32_t total_bytes   = 1 + payload_bytes + 2 * num_exc;
    if (in_cap < total_bytes) { return 0; }

    if (payload_bytes == 0) {
        std::memset(out, 0, static_cast<size_t>(kBlockSize) * 4u);
    } else {
        // Copy payload + slack into a stack buffer so unpack_bits can do its
        // 8-byte read window without overrunning the caller's `in`.
        uint8_t scratch[kBlockSize * 4 + kBitpackTailSlack] = {0};
        std::memcpy(scratch, in + 1, static_cast<size_t>(payload_bytes));
        unpack_bits(scratch, kBlockSize, bpv, out);
    }

    // Apply patches: out[idx] |= high << bpv.
    if (num_exc > 0u) {
        const uint8_t* p = in + 1 + payload_bytes;
        for (uint8_t e = 0u; e < num_exc; ++e) {
            const uint8_t  idx  = p[2 * e + 0];
            const uint32_t high = p[2 * e + 1];
            out[idx] |= high << bpv;
        }
    }
    return total_bytes;
}

// --- Delta -> doc-id helper ------------------------------------------------
//
// Unpacks a packed delta block AND prefix-sums into uint64 doc-ids in one pass.
// `prev_doc` is the absolute doc-id of the value immediately preceding this
// block (the caller's running cursor). The prefix sum is branchless.
BOLT_FORCE_INLINE int32_t pfor_unpack_to_doc_ids(const uint8_t* BOLT_RESTRICT in,
                                                  int32_t in_cap,
                                                  uint64_t prev_doc,
                                                  uint64_t* BOLT_RESTRICT out) noexcept {
    assert(in != nullptr);
    assert(out != nullptr);
    assert(in_cap >= 0);

    uint32_t deltas[kBlockSize];
    const int32_t consumed = pfor_unpack_block(in, in_cap, deltas);
    if (consumed == 0) { return 0; }

    uint64_t acc = prev_doc;
    for (int32_t i = 0; i < kBlockSize; ++i) {
        acc += static_cast<uint64_t>(deltas[i]);
        out[i] = acc;
    }
    return consumed;
}

}  // namespace pfor
}  // namespace kernels
}  // namespace bolt
