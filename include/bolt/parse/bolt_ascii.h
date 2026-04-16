#pragma once
// ---------------------------------------------------------------------------
// bolt::parse — compiled-library ASCII numeric parsers.
//
// These functions assume the caller has already validated the byte range
// (length and character set). They are the branchless fast paths used by
// the lakehouse CSV / 1BRC-style ingest. Length bounds:
//   parse_int10th        — [1..6]  (e.g. "-99.9")
//   parse_int            — [1..18]
//   parse_int10th_word   — [1..6]  (word preloaded)
//
// No validation on input. No exceptions. noexcept.
// ---------------------------------------------------------------------------
#include "bolt/bolt_port.h"
#include <cstdint>

namespace bolt {
namespace parse {

// Parse an ASCII signed decimal-with-one-fractional-digit number into
// tenths. "-12.3" -> -123. "0.0" -> 0. "99.9" -> 999. The string need
// NOT be null-terminated; len is the exact byte count [1..6].
//
// Branchless implementation in src/parse/bolt_ascii.cpp using the
// mtopolnik/merrykitty magic-multiplier trick.
int32_t parse_int10th(const char* p, int len) noexcept;

// Parse an ASCII signed decimal integer (no fractional digit) into
// int64_t. Branchless for [1..18] digit inputs. Length is the exact
// byte count.
int64_t parse_int(const char* p, int len) noexcept;

// Word-based variant: caller has already loaded an 8-byte chunk
// containing the ASCII number, with the number occupying bytes [0..len).
// Bytes [len..8) are unused. This avoids a load when SWAR scanning has
// already produced the word.
int32_t parse_int10th_word(uint64_t word, int len) noexcept;

}  // namespace parse
}  // namespace bolt

// C-callable surface for FFI / lakehouse-library consumers.
extern "C" {
int32_t bolt_parse_int10th(const char* p, int len);
int64_t bolt_parse_int(const char* p, int len);
int32_t bolt_parse_int10th_word(uint64_t word, int len);
}
