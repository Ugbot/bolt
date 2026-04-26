// bolt_ribbon.h — Ribbon filter (Dillinger 2021) — Wave 20D.4 body.
//
// Goal: cut bits/key vs SBBF (~10 bits/key for 1% FPR) on cold L2+
// SSTs where build cost is amortised over many reads. Build cost is
// higher than Bloom (banded GF(2) elimination), so MarbleDB uses
// Ribbon ONLY on level >= 2 SSTs.
//
// STATUS (Wave 20D.4): real body lives in `external/bolt/src/bolt_ribbon.cpp`.
// Parameters: r=32 band width, fp_bits=8 (byte-aligned fingerprint),
// m = ceil(1.05 * n) + r slots. Storage = m bytes ≈ 8.4 bits/key for
// FPR ≈ 1/256 ≈ 0.39%.  Construction may fail with low probability
// (~1%/attempt at 1.05×); callers MUST retry up to `kMaxRetries`
// with monotonically increasing seeds.
//
// Reference:
//   Dillinger, P. C. & Walzer, S. "Ribbon filter: practically
//   smaller than Bloom and Xor." arXiv:2103.02515, 2021.
//
// API shape (frozen here so the writer / reader / test scaffolding
// compiles against a stable surface):
//   - Construction: `bolt::ribbon::build(keys, n, target_fpr, scratch,
//     scratch_bytes, out_buf, out_buf_bytes, seed)`. Returns the
//     bytes written on success, or 0 on failure (caller retries with
//     a different seed; ribbon construction fails ~1% of the time).
//   - Query: `bolt::ribbon::might_contain(filter_bytes, len, key)`.
//     `noexcept`, no allocation, single-cache-line probe.
//   - Sizing: `bolt::ribbon::size_bytes_for(n, target_fpr)` returns
//     the upper bound on `out_buf_bytes` so callers can right-size
//     the arena before calling `build`.
//
// Failure mode: ribbon construction has a per-attempt ~1% failure
// probability (column dependency in the banded matrix). Callers
// MUST retry up to `kMaxRetries = 8` with monotonically increasing
// `seed` values before falling back to SBBF.
//
// RULES (per Bolt CLAUDE.md):
//   - Header-only; POD; noexcept; no RTTI.
//   - No std::string / std::vector / smart pointers.
//   - ≥2 asserts per function.
//   - Caller-owned scratch — no malloc inside `build`.

#pragma once

#include "bolt/bolt_port.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace ribbon {

// ---------------------------------------------------------------------------
// Constants — frozen, on-disk visible.
// ---------------------------------------------------------------------------

// Construction retry cap. ~1% per-attempt failure → 8 retries gives
// ~10^-16 total failure rate. Caller falls back to SBBF on overflow.
inline constexpr uint32_t kMaxRetries = 8u;

// Band width — number of consecutive slots each key's coefficient
// vector spans. r=32 keeps the per-slot pivot row in a u32. The
// in-flight coefficient lives in a u64 register so the leftward
// shift `pivot << pos` (pos up to 31) doesn't lose high bits.
inline constexpr uint32_t kBandWidth = 32u;

// Fingerprint size in bits. Byte-aligned for branchless query.
// FPR ≈ 2^-fp_bits = 1/256 ≈ 0.39%.
inline constexpr uint32_t kFpBits = 8u;

// Over-provisioning factor. m = ceil(kOverProvisionN/kOverProvisionD * n)
// + kBandWidth.  At r=32 the per-attempt failure rate at 1.05× is on
// the cliff — we ship 1.30× to land in the "single-attempt usually
// succeeds" regime even at n=10K.  Storage cost: ~10.5 bits/key vs the
// paper-best 8.4 bits/key — still cheaper than a 1% SBBF (~10 bits/key)
// and the build-side reliability is what matters here (failures fall
// back to SBBF, which would erase the win).
inline constexpr uint32_t kOverProvisionN = 130u;
inline constexpr uint32_t kOverProvisionD = 100u;

// Bits/key reported by the implementation given the kFpBits choice.
// Used for sizing helpers; ~8.4 bits/key for fp=8.
inline constexpr double kRibbonBitsPerKey =
    (static_cast<double>(kFpBits) * kOverProvisionN) /
    static_cast<double>(kOverProvisionD);

// Header word stamped into the first 8 bytes of `out_buf` so the
// query path can self-validate without external metadata.
// Bytes "RIBBON\1\0" little-endian.
inline constexpr uint64_t kMagic = 0x00'01'4E'4F'42'42'49'52ull;

// On-disk header is 16 bytes:
//   [0..7]   uint64 magic        — kMagic
//   [8..11]  uint32 m            — slot count
//   [12]     uint8  r            — band width (== kBandWidth)
//   [13]     uint8  fp_bits      — fingerprint bits (== kFpBits)
//   [14..15] uint16 seed_used    — low 16 bits of the seed used
inline constexpr size_t kHeaderBytes = 16u;

// ---------------------------------------------------------------------------
// Sizing helper. Returns the upper bound on bytes the caller must
// reserve in `out_buf` for `n` keys.  `target_fpr` is currently
// advisory — the implementation ships with a fixed `kFpBits` (FPR ≈
// 1/256). Callers asking for a tighter FPR will need a future variant
// with a wider fingerprint.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE size_t size_bytes_for(int64_t n, double target_fpr) noexcept {
    assert(n >= 0);
    assert(target_fpr > 0.0 && target_fpr < 1.0);
    (void)target_fpr;
    if (n == 0) return kHeaderBytes;
    // m = ceil(n * kOverProvisionN / kOverProvisionD) + kBandWidth.
    const uint64_t m = (static_cast<uint64_t>(n) * kOverProvisionN
                         + kOverProvisionD - 1u) / kOverProvisionD
                       + kBandWidth;
    // Payload = m bytes (one fingerprint per slot, fp_bits=8).
    return kHeaderBytes + static_cast<size_t>(m);
}

// ---------------------------------------------------------------------------
// Scratch sizing helper. Caller passes a scratch buffer of at least
// this size to `build`. No malloc is ever performed inside `build`.
// Layout consumed by the kernel:
//   m × u32  — pivot coefficient row per slot (0 == empty)
//   m × u8   — pivot RHS fingerprint per slot
//   align padding to 4 bytes between regions
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE size_t scratch_bytes_for(int64_t n) noexcept {
    assert(n >= 0);
    if (n == 0) return 0u;
    const uint64_t m = (static_cast<uint64_t>(n) * kOverProvisionN
                         + kOverProvisionD - 1u) / kOverProvisionD
                       + kBandWidth;
    // Bytes: m*4 (coefs) + m (rhs) + 4 alignment slack
    return static_cast<size_t>(m) * 5u + 4u;
}

// ---------------------------------------------------------------------------
// Construction.
//
// On success: writes the filter to `out_buf` (header + payload) and
// returns the byte count actually used (== size_bytes_for(n)). On
// failure (column dependency, scratch too small, out_buf too small,
// n < 0): returns 0. The caller retries with `seed + 1` up to
// `kMaxRetries` times before falling back to SBBF.
// Definition: external/bolt/src/bolt_ribbon.cpp.
// ---------------------------------------------------------------------------
size_t build(const uint64_t* keys, int64_t n,
             double target_fpr,
             void* scratch, size_t scratch_bytes,
             void* out_buf, size_t out_buf_bytes,
             uint64_t seed) noexcept;

// ---------------------------------------------------------------------------
// Query.
//
// Returns true if `key` MAY be present (false positive possible),
// false if `key` is definitely absent. NEVER returns false for a
// member key. Branchless inner loop — bounded ~30 cycles.
// Definition: external/bolt/src/bolt_ribbon.cpp.
// ---------------------------------------------------------------------------
bool might_contain(const void* filter_bytes, size_t len,
                   uint64_t key) noexcept;

// ---------------------------------------------------------------------------
// Header validation. Returns true iff `filter_bytes[0..16)` carries
// the ribbon magic word. Lets the reader detect a corrupted /
// scaffolded-empty filter and fall back to "may be present".
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE bool has_valid_header(const void* filter_bytes,
                                        size_t len) noexcept {
    assert(filter_bytes != nullptr || len == 0);
    if (len < 16u) return false;
    uint64_t magic = 0;
    const auto* p = static_cast<const uint8_t*>(filter_bytes);
    for (uint32_t i = 0; i < 8u; ++i) {
        magic |= static_cast<uint64_t>(p[i]) << (i * 8u);
    }
    return magic == kMagic;
}

}  // namespace ribbon
}  // namespace bolt
