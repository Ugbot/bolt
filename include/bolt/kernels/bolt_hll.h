// bolt_hll.h — HyperLogLog++ cardinality sketch.
//
// Fixed-size POD sketch for approximate COUNT(DISTINCT) over 64-bit
// hashes. Precision P selects m = 2^P registers (1 byte each, holding
// 6-bit values 0..64). Standard error is ~1.04/sqrt(m); P=14 → ~0.81%
// standard error with a 16 KiB sketch.
//
// Estimator:
//   * Linear counting when raw estimate ≤ 2.5*m and at least one
//     register is still zero (Flajolet 2007 small-range correction).
//   * Classical HLL E = alpha_m * m^2 / sum(2^-R[i]) otherwise.
//   * Large-range correction unnecessary for 64-bit hashes.
//
// HLL++ refinements (Heule et al. 2013) that we intentionally skip:
//   * 6-bit register packing (saves 2x memory; pay later if we need it).
//   * Sparse representation (we always carry the full m bytes).
//   * Empirical bias correction tables (linear counting + raw HLL
//     already gives ≤2% error on the cardinalities we care about with
//     P=14; tables can be added behind a compile-time switch).
//
// Tiger Style: zero allocations, noexcept, POD, ≥2 asserts on hot
// paths, branchless inner loops.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_hash.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace bolt::kernels::hll {

// ---------------------------------------------------------------------------
// Fixed-size HLL++ sketch — m = 2^P single-byte registers.
// ---------------------------------------------------------------------------
template <uint8_t P>
struct HllPp {
    static_assert(P >= 4 && P <= 18, "HLL precision out of range [4, 18]");
    static constexpr uint32_t k_p             = P;
    static constexpr uint32_t k_num_registers = 1u << P;
    static constexpr uint64_t k_idx_mask      = (1ull << P) - 1ull;

    // 1 byte per register; max value is 64 (count of leading zeros in
    // the (64-P) low bits, plus 1). 6 bits would suffice; we trade
    // memory for branchless updates and a trivial merge loop.
    uint8_t registers[k_num_registers];
};

// ---------------------------------------------------------------------------
// alpha_m — Flajolet/Fusy/Gandouet/Meunier 2007 constant.
//   alpha_16 = 0.673
//   alpha_32 = 0.697
//   alpha_64 = 0.709
//   alpha_m  = 0.7213 / (1 + 1.079 / m)   for m >= 128
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE double hll_alpha_m(uint32_t m) noexcept {
    assert(m >= 16);
    if (m == 16) return 0.673;
    if (m == 32) return 0.697;
    if (m == 64) return 0.709;
    return 0.7213 / (1.0 + 1.079 / static_cast<double>(m));
}

// ---------------------------------------------------------------------------
// hll_init — zero all registers. Called once per sketch lifetime.
// ---------------------------------------------------------------------------
template <uint8_t P>
BOLT_FORCE_INLINE void hll_init(HllPp<P>* sketch) noexcept {
    assert(sketch != nullptr);
    std::memset(sketch->registers, 0, HllPp<P>::k_num_registers);
}

// ---------------------------------------------------------------------------
// hll_add_hash — fold a single pre-hashed 64-bit value into the sketch.
//
//   idx = high P bits          ( hash >> (64 - P) )
//   rho = ctlz( hash << P | sentinel ) + 1  in [1, 65-P]
//
// The sentinel bit at position (P-1) of the shifted hash guarantees
// rho is bounded by (64 - P + 1) and the shift never overflows.
// Branchless register update: r = max(old, rho).
// ---------------------------------------------------------------------------
template <uint8_t P>
BOLT_FORCE_INLINE void hll_add_hash(HllPp<P>* sketch, uint64_t hash) noexcept {
    assert(sketch != nullptr);
    constexpr uint32_t shift = 64u - P;
    const uint32_t idx       = static_cast<uint32_t>(hash >> shift);
    // Append a sentinel '1' at bit (shift - 1) so clz never exceeds shift.
    const uint64_t w         = (hash << P) | (1ull << (P - 1));
    const uint8_t  rho       = static_cast<uint8_t>(::bolt_clz64(w) + 1);
    assert(idx < HllPp<P>::k_num_registers);
    assert(rho >= 1 && rho <= (64u - P + 1u));
    const uint8_t old        = sketch->registers[idx];
    sketch->registers[idx]   = (rho > old) ? rho : old;
}

// ---------------------------------------------------------------------------
// hll_add_batch — fold n pre-hashed values. Caller supplies hashes
// produced by bolt::swiss_mix (or any 64-bit hash with good upper-bit
// distribution).
// ---------------------------------------------------------------------------
template <uint8_t P>
void hll_add_batch(HllPp<P>* sketch,
                   const uint64_t* BOLT_RESTRICT hashes,
                   int64_t n) noexcept {
    assert(sketch != nullptr);
    assert(hashes != nullptr || n == 0);
    assert(n >= 0);
    constexpr uint32_t shift = 64u - P;
    constexpr uint64_t sent  = 1ull << (P - 1);
    uint8_t* BOLT_RESTRICT regs = sketch->registers;
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t h   = hashes[i];
        const uint32_t idx = static_cast<uint32_t>(h >> shift);
        const uint64_t w   = (h << P) | sent;
        const uint8_t  rho = static_cast<uint8_t>(::bolt_clz64(w) + 1);
        const uint8_t  old = regs[idx];
        regs[idx]          = (rho > old) ? rho : old;
    }
}

// ---------------------------------------------------------------------------
// hll_merge — register-wise max into dst. Both sketches must share P.
// ---------------------------------------------------------------------------
template <uint8_t P>
void hll_merge(HllPp<P>* dst, const HllPp<P>* src) noexcept {
    assert(dst != nullptr);
    assert(src != nullptr);
    assert(dst != src);
    uint8_t* BOLT_RESTRICT d       = dst->registers;
    const uint8_t* BOLT_RESTRICT s = src->registers;
    constexpr uint32_t m           = HllPp<P>::k_num_registers;
    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t a = d[i];
        const uint8_t b = s[i];
        d[i] = (a > b) ? a : b;
    }
}

// ---------------------------------------------------------------------------
// hll_estimate — cardinality estimate.
//
//   Z       = sum_i 2^(-R[i])
//   E_raw   = alpha_m * m^2 / Z
//   V       = #{i : R[i] == 0}
//   If V > 0 and E_raw <= 2.5*m: return m * ln(m / V)   (linear counting)
//   Else: return E_raw
//
// Branchless inner loop accumulates Z and V in a single pass over the
// register array — ~ 1 cache line per 64 registers, one mul + add per
// register (no FP exp; we precompute 2^-k via reciprocal of a table).
// ---------------------------------------------------------------------------
// 2^-k for k in [0, 64] — process-lifetime table, zero per-call init cost.
// Meyers singleton: thread-safe one-time init, no per-call overhead.
BOLT_FORCE_INLINE const double* hll_inv_pow2_table() noexcept {
    static const double* const tbl = []() noexcept {
        static double t[65];
        t[0] = 1.0;
        for (int k = 1; k <= 64; ++k) t[k] = t[k - 1] * 0.5;
        return t;
    }();
    return tbl;
}

template <uint8_t P>
uint64_t hll_estimate(const HllPp<P>* sketch) noexcept {
    assert(sketch != nullptr);
    constexpr uint32_t m = HllPp<P>::k_num_registers;
    const double* BOLT_RESTRICT inv_pow2 = hll_inv_pow2_table();

    // 8-way unrolled accumulators break the serial dependency on `z`
    // (FP add has 3-4 cycle latency, throughput 1/cycle — a single
    // accumulator caps us at ~12 µs for 16384 regs at 4 GHz).
    const uint8_t* BOLT_RESTRICT regs = sketch->registers;
    static_assert(m >= 16, "tail handling assumes m >= 16");
    double   z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;
    double   z4 = 0.0, z5 = 0.0, z6 = 0.0, z7 = 0.0;
    uint32_t zeros = 0;
    const uint32_t m8 = m & ~7u;
    for (uint32_t i = 0; i < m8; i += 8) {
        const uint8_t r0 = regs[i+0]; z0 += inv_pow2[r0]; zeros += (r0 == 0);
        const uint8_t r1 = regs[i+1]; z1 += inv_pow2[r1]; zeros += (r1 == 0);
        const uint8_t r2 = regs[i+2]; z2 += inv_pow2[r2]; zeros += (r2 == 0);
        const uint8_t r3 = regs[i+3]; z3 += inv_pow2[r3]; zeros += (r3 == 0);
        const uint8_t r4 = regs[i+4]; z4 += inv_pow2[r4]; zeros += (r4 == 0);
        const uint8_t r5 = regs[i+5]; z5 += inv_pow2[r5]; zeros += (r5 == 0);
        const uint8_t r6 = regs[i+6]; z6 += inv_pow2[r6]; zeros += (r6 == 0);
        const uint8_t r7 = regs[i+7]; z7 += inv_pow2[r7]; zeros += (r7 == 0);
    }
    for (uint32_t i = m8; i < m; ++i) {
        const uint8_t r = regs[i];
        z0 += inv_pow2[r]; zeros += (r == 0);
    }
    const double z = ((z0 + z1) + (z2 + z3)) + ((z4 + z5) + (z6 + z7));
    assert(z > 0.0);

    const double alpha = hll_alpha_m(m);
    const double mm    = static_cast<double>(m) * static_cast<double>(m);
    const double e_raw = alpha * mm / z;

    // Small-range correction (linear counting). 2.5*m is Flajolet's
    // empirical threshold; HLL++ uses a per-P table but the difference
    // at P=14 is <0.5% in the worst case.
    double e = e_raw;
    if (zeros != 0u && e_raw <= 2.5 * static_cast<double>(m)) {
        e = static_cast<double>(m) *
            std::log(static_cast<double>(m) / static_cast<double>(zeros));
    }
    if (e < 0.0) e = 0.0;
    return static_cast<uint64_t>(e + 0.5);
}

// ---------------------------------------------------------------------------
// Convenience: hash + add. For when callers have raw uint64 keys and
// want the default bolt mix. String / multi-column keys must be hashed
// upstream with the appropriate kernel.
//
// Note: HLL needs more bit-mixing than SwissTable does — the index
// (top P bits) and the rho-zone (bottom 64-P bits) of each hash must
// be statistically independent of the input. The single wyhash3 mix
// used as SwissTable's default is reversible and leaves residual
// structure in the top bits when callers pass sequential integers.
// Double-mix here (wyhash3 then murmur3-finalizer) to get full
// avalanche; the second mix costs ~3 cycles and is well within the
// 3 ns/row budget. Callers who already hold a well-mixed 64-bit hash
// should call hll_add_hash directly.
// ---------------------------------------------------------------------------
template <uint8_t P>
BOLT_FORCE_INLINE void hll_add_u64(HllPp<P>* sketch, uint64_t key) noexcept {
    hll_add_hash<P>(sketch,
                    ::bolt::swiss_mix_murmur3(::bolt::swiss_mix(key)));
}

}  // namespace bolt::kernels::hll
