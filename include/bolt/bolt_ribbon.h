// bolt_ribbon.h — Ribbon filter (Dillinger 2021) — SCAFFOLD ONLY.
//
// Goal: ~3 bits/key for 1% FPR (vs SBBF's ~10 bits/key) with 1.05×
// space overhead. Higher build cost than Bloom (Gaussian elimination
// over GF(2)), so MarbleDB plans to use Ribbon ONLY on cold L2+ SSTs
// where build cost is amortised over many reads (Wave 20D.4 plan).
//
// STATUS: SCAFFOLDING ONLY. Body deferred — a correct ribbon
// implementation is 200+ lines of bit-banging (banded matrix
// construction, Gaussian elimination with retry on dependent rows,
// query-time bit-stripe XOR) that exceeds this header's 300-line
// budget AND would need extensive correctness testing before being
// trusted for point-get (false negatives = silent KV miss). Until
// the body lands, callers must NOT route real point-get traffic
// through this filter — the writer side is gated on `level >= 2`
// behind a still-disabled flag (Wave 20D.4 deferral).
//
// Reference (when implemented):
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
// Constants frozen at scaffold time so the writer/reader can compile
// against stable values.
// ---------------------------------------------------------------------------

// Construction retry cap. ~1% per-attempt failure → 8 retries gives
// ~10^-16 total failure rate. Caller falls back to SBBF on overflow.
inline constexpr uint32_t kMaxRetries = 8u;

// Target bits-per-key for 1% FPR. ~3 bits/key matches Dillinger's
// reported figures; SBBF needs ~10 bits/key for the same FPR. The
// 1.05× overhead is the slop above the information-theoretic lower
// bound. When the body lands, this constant becomes a function of
// the requested `target_fpr` — kept as a compile-time scalar here so
// `size_bytes_for` returns deterministic values.
inline constexpr double kRibbonBitsPerKey1Pct = 3.15;  // 3.0 × 1.05

// Header word stamped into the first 16 bytes of `out_buf` so the
// query path can self-validate without external metadata.
inline constexpr uint64_t kMagic = 0x52'49'42'42'4F'4E'01'00ull;  // "RIBBON\1\0"

// ---------------------------------------------------------------------------
// Sizing helper. Returns the upper bound on bytes the caller must
// reserve in `out_buf` for `n` keys at the given target FPR.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE size_t size_bytes_for(int64_t n, double target_fpr) noexcept {
    assert(n >= 0);
    assert(target_fpr > 0.0 && target_fpr < 1.0);
    if (n == 0) return 16u;  // header only
    // Conservative scaffold sizing: 1% target gets the canonical 3.15
    // bits/key; tighter targets scale linearly. Final implementation
    // will use `-log2(fpr) * 1.05 / 8` bytes/key.
    const double bits_per_key = (target_fpr <= 0.01)
        ? kRibbonBitsPerKey1Pct
        : kRibbonBitsPerKey1Pct * 0.01 / target_fpr;
    const size_t payload_bits = static_cast<size_t>(
        static_cast<double>(n) * bits_per_key + 0.5);
    const size_t payload_bytes = (payload_bits + 7u) / 8u;
    return 16u + payload_bytes;  // 16 B header + bit-packed payload
}

// ---------------------------------------------------------------------------
// Scratch sizing helper. Caller passes a scratch buffer of at least
// this size to `build`. No malloc is ever performed inside `build`.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE size_t scratch_bytes_for(int64_t n) noexcept {
    assert(n >= 0);
    if (n == 0) return 0u;
    // Banded matrix: n rows × ~64-bit band → 8 bytes/row, plus a
    // RHS column (1 bit/row, rounded to 8) and per-row hash slot
    // (8 bytes/row) and a row-index permutation (4 bytes/row).
    return static_cast<size_t>(n) * 24u;
}

// ---------------------------------------------------------------------------
// Construction (DEFERRED — scaffold only).
//
// On success: writes the filter to `out_buf` and returns the byte
// count actually used. On failure (column dependency, scratch too
// small, out_buf too small, n < 0): returns 0. The caller retries
// with `seed + 1` up to `kMaxRetries` times before falling back.
//
// The scaffold body always returns 0 — DO NOT WIRE THIS UP TO REAL
// POINT-GET TRAFFIC. The writer-side gate stays on SBBF until the
// body lands and the `test_ribbon_filter` round-trip tests are
// enabled.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE size_t build(const uint64_t* keys, int64_t n,
                                double target_fpr,
                                void* scratch, size_t scratch_bytes,
                                void* out_buf, size_t out_buf_bytes,
                                uint64_t seed) noexcept {
    assert(out_buf != nullptr || out_buf_bytes == 0);
    assert(scratch != nullptr || scratch_bytes == 0);
    (void)keys;
    (void)n;
    (void)target_fpr;
    (void)scratch;
    (void)scratch_bytes;
    (void)out_buf;
    (void)out_buf_bytes;
    (void)seed;
    // SCAFFOLD: body deferred. Returning 0 forces every caller's
    // retry loop to exhaust and fall back to SBBF, preserving point-
    // get correctness while the real implementation is pending.
    // TODO(wave-20D.4-followup): implement banded GF(2) construction
    // per Dillinger 2021 §3.2. Estimate: ~180 lines (banded matrix
    // build + Gaussian elimination + bit-stripe pack).
    return 0u;
}

// ---------------------------------------------------------------------------
// Query (DEFERRED — scaffold only).
//
// Returns true if `key` MAY be present (false positive possible),
// false if `key` is definitely absent. NEVER returns false for a
// member key (the no-false-negative invariant is what makes the
// filter safe in the point-get path).
//
// The scaffold body always returns true (a "no filter at all" answer
// — every query falls through to the actual SST block I/O). This is
// safe but slow; it lets the writer-side gate ship behind a flag
// without breaking any reader that happens to encounter a Ribbon-
// kind file before the body lands.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE bool might_contain(const void* filter_bytes, size_t len,
                                      uint64_t key) noexcept {
    assert(filter_bytes != nullptr || len == 0);
    (void)filter_bytes;
    (void)len;
    (void)key;
    // SCAFFOLD: body deferred. Conservative answer = true (= "may be
    // present"), so the caller falls through to block I/O — no false
    // negatives, just no filtering benefit. Real implementation will
    // probe the banded matrix at `swiss_mix(key)` and XOR-reduce.
    // TODO(wave-20D.4-followup): implement query per Dillinger 2021
    // §3.3. Estimate: ~40 lines.
    return true;
}

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
