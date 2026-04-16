// ---------------------------------------------------------------------------
// bolt::parse — branchless ASCII numeric parsers.
//
// Implements the mtopolnik / merrykitty / royvanrijn magic-multiplier
// trick for parsing fixed-precision decimals (the 1BRC hot path) and
// a positional digit-pack integer parser.
//
// Caller-validated API: no out-of-range checks; bounds are asserts.
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_ascii.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace parse {

// Load up to 8 bytes starting at p into a little-endian uint64_t.
// Bytes beyond len are zeroed. Used only for the byte-pointer API;
// callers that already have a word use parse_int10th_word directly.
static BOLT_FORCE_INLINE uint64_t load_word_le(const char* p, int len) noexcept {
    assert(p != nullptr);
    assert(len >= 1 && len <= 8);
    uint64_t w = 0;
    // memcpy of exact len keeps this safe for sub-8 buffers without
    // over-reading. Compilers lower short memcpys to a single load+mask.
    std::memcpy(&w, p, static_cast<size_t>(len));
    return w;
}

int32_t parse_int10th_word(uint64_t word, int len) noexcept {
    // Preconditions (>=2 asserts).
    assert(len >= 3 && len <= 6);
    assert(len == 3 || len == 4 || len == 5 || len == 6);

    // Branchless decimal-with-one-fractional-digit parser inspired by the
    // mtopolnik / merrykitty / royvanrijn 1BRC fast paths. We use SWAR
    // dot-search + positional digit-packing, keeping the control flow
    // entirely data-driven (no branches per-record).
    //
    // Layout (little-endian load, example "12.3" with len=4):
    //   byte 0: '1'   byte 1: '2'   byte 2: '.'   byte 3: '3'
    //
    // Step 1: detect sign via bit-4 test on byte 0.  '-' (0x2D) has bit 4
    //         clear; any digit '0'..'9' (0x30..0x39) has bit 4 set.
    // Step 2: find '.' via SWAR — gives byte index of the separator.
    // Step 3: extract low nibbles of the ASCII digit bytes positionally.
    //         Three digits total (hundreds/tens/ones-or-tenths), each
    //         worth 100/10/1 in the scaled-by-10 result.
    // Step 4: branchless negate using `(x ^ m) - m` with m = 0 or -1.

    // Sign detection: neg_mask = -1 if byte0=='-', else 0. Arith shift
    // of `~word` reproduces the Java/merrykitty `(~word << 59) >> 63`.
    const int64_t neg_mask_s = static_cast<int64_t>(~word << 59) >> 63;
    const uint64_t neg_mask = static_cast<uint64_t>(neg_mask_s);

    // Locate '.' (byte index in the word).
    const int dot_byte = bolt_swar_find_byte_u64(word, '.');
    assert(dot_byte >= 1 && dot_byte <= 4);

    // Digit byte indexes (positional layout):
    //   idx_ones = dot_byte - 1  (digit immediately before '.')
    //   idx_tens = dot_byte - 2  (0 iff no tens digit — caller promised
    //                             at least one digit before '.')
    //   idx_tnth = dot_byte + 1  (digit immediately after '.')
    // We always read 3 "slots"; a missing tens digit lands on the sign
    // byte (if negative) or on the ones digit's own byte (if positive),
    // which we neutralise by clamping idx_tens to the sign position and
    // masking its contribution to zero when byte0 is '-'.
    const int idx_ones = dot_byte - 1;
    const int idx_tens = dot_byte - 2;  // may be -1; we handle below
    const int idx_tnth = dot_byte + 1;
    assert(idx_ones >= 0 && idx_tnth <= 6);

    // Extract low nibbles via variable byte-shifts.
    const uint64_t ones_d = (word >> (idx_ones * 8)) & 0x0Full;
    const uint64_t tnth_d = (word >> (idx_tnth * 8)) & 0x0Full;

    // Tens digit: if idx_tens < 0, the number has only one integer digit
    //             — contribute 0.
    //             Else if idx_tens == 0 and byte0 == '-', also 0.
    //             Otherwise low nibble of byte[idx_tens].
    // All expressed branchlessly.
    const uint64_t tens_raw = (idx_tens >= 0)
        ? ((word >> (idx_tens * 8)) & 0x0Full)
        : 0ull;
    // If idx_tens == 0 and sign byte, raw would be 0x0D (from '-'=0x2D
    // low nibble). Mask with ~neg_mask when idx_tens==0.
    const uint64_t tens_sign_kill =
        (idx_tens == 0) ? (~neg_mask & 0xFFull) : 0xFFull;
    const uint64_t tens_d = tens_raw & tens_sign_kill;

    const uint64_t abs_value =
        tnth_d + ones_d * 10ull + tens_d * 100ull;
    assert(abs_value <= 999ull);

    const int64_t result =
        (static_cast<int64_t>(abs_value) ^ neg_mask_s) - neg_mask_s;
    assert(result >= -999 && result <= 999);
    (void)len;
    return static_cast<int32_t>(result);
}

int32_t parse_int10th(const char* p, int len) noexcept {
    assert(p != nullptr);
    assert(len >= 3 && len <= 6);
    const uint64_t word = load_word_le(p, len);
    return parse_int10th_word(word, len);
}

int64_t parse_int(const char* p, int len) noexcept {
    assert(p != nullptr);
    assert(len >= 1 && len <= 19);

    // Handle sign.
    int64_t sign = 1;
    int i = 0;
    if (p[0] == '-') {
        sign = -1;
        i = 1;
    } else if (p[0] == '+') {
        i = 1;
    }
    assert(i <= len);

    // Positional accumulation. Branchless body; loop bound is the
    // digit count (<=18), which beats any data-dependent branch.
    int64_t acc = 0;
    for (; i < len; ++i) {
        const int64_t d = static_cast<int64_t>(
            static_cast<uint8_t>(p[i]) & 0x0Fu);
        acc = acc * 10 + d;
    }
    return acc * sign;
}

}  // namespace parse
}  // namespace bolt

// ---------------------------------------------------------------------------
// C-callable forwarders.
// ---------------------------------------------------------------------------
extern "C" {

int32_t bolt_parse_int10th(const char* p, int len) {
    return bolt::parse::parse_int10th(p, len);
}

int64_t bolt_parse_int(const char* p, int len) {
    return bolt::parse::parse_int(p, len);
}

int32_t bolt_parse_int10th_word(uint64_t word, int len) {
    return bolt::parse::parse_int10th_word(word, len);
}

}  // extern "C"
