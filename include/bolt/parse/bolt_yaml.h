#pragma once
// bolt::parse::yaml — YAML onto the fionn tape (G2GRAPH-453).
//
// A bounded reader for the YAML that configuration and Markdown frontmatter
// use. It emits the SAME structural tape as bolt::parse::json (bolt_json.h):
// Begin/EndObject, Begin/EndArray, Key, String, Int64, Float64, BoolTrue,
// BoolFalse, Null, End — so every consumer (value_from_index, iterators with
// skip_to_close, the JSON-LD expander, knowledge ingest) walks one
// representation, and a PathFilter builds only the subtrees asked for.
//
// Architecture (after fionn, docs/research/yaml-tape-architecture.md):
//   Stage 1 — structural pre-scan. One pass finds every line start
//     (word-at-a-time newline detection) and, per line, its indentation
//     (word-at-a-time leading-space count) and class (content / blank /
//     comment / tab-indented). SIMD is a TODO, as in bolt_json; the scalar
//     SWAR path is the reference.
//   Stage 2 — a state machine over the line table: block structure is
//     decided by line indentation looked up in the table, never by
//     rescanning bytes; only scalars and flow collections are scanned
//     within a line.
//   Tape — zero-copy where YAML allows: a plain or quoted single-line scalar
//     whose bytes are already valid JSON string content is an (offset,
//     length) slice of the source; a number is its source slice when that is
//     a JSON number, parsed lazily by iter_int64 / iter_float64. Escaped
//     (YAML-only escapes), folded, multi-line and block scalars, and number
//     spellings JSON lacks (0x1F, +5, .5), are materialised once into an
//     arena buffer (StructuralIndex::aux), JSON-escaped, with
//     Token::flags = kTokenAux.
//
// Semantics: YAML 1.2 core schema. Plain scalars type as null (`~`, `null`,
// empty), bool (`true` / `false` in the spec's three cases), int (decimal,
// `0o`, `0x`), float (decimal / exponent); every other scalar, and every
// quoted one, is a string. Dates stay strings. Duplicate keys are kept in
// order (a consumer's map decides; value_from_index keeps both).
//
// SUPPORTED: block mappings and sequences (compact `- key: v` items,
// indentless sequences under a key, `- - x`), flow mappings and sequences
// (nested, multi-line), plain multi-line scalars, single and double quoted
// scalars (every escape, folding), literal `|` / folded `>` block scalars
// with chomping and indentation indicators, comments, an optional leading
// `---` and a trailing `...`.
//
// REFUSED BY NAME (YamlCode::Unsupported, `what`): anchors and aliases,
// tags, directives, complex (`?`) and non-scalar keys, more than one
// document, tab indentation, non-finite floats, integers beyond int64,
// multi-line plain scalars inside flow collections, single-pair mappings in
// flow sequences.
//
// Budgets are the caller's (Limits registry entries of whoever calls): input
// bytes, nesting depth, value nodes; past one the build fails with kLimit
// naming it. A filtered build also caps depth at json::kJsonMaxDepth (the
// path stack's).
//
// Tiger Style: noexcept, error-return, arena-owned output, no STL containers.

#include <cstdint>

#include "bolt/parse/bolt_json.h"

namespace bolt { class Arena; }

namespace bolt::parse::yaml {

enum class YamlCode : uint8_t { Ok, Syntax, Unsupported, Limit, NoMemory };

struct YamlLimits {
    uint32_t bytes  = 0u;   // input bytes; also the output bound of to_json (<= INT32_MAX)
    uint32_t depth  = 0u;   // nesting of collections (<= 256)
    uint32_t nodes  = 0u;   // values, collections included
};

struct YamlError {
    YamlCode    code;
    uint32_t    line;     // 1-based; 0 when not positional
    uint32_t    col;      // 1-based
    const char* what;     // static string naming the problem / construct / budget
};

// Build the structural tape of `src` (one YAML document; empty = `null`).
// `filter` (nullable, from json::compile_paths) keeps only the subtrees on
// interest paths, exactly as json::build_index_filtered. On success `*out`
// borrows `src` (it must outlive the index) and owns its tape and aux bytes
// in `arena`. On failure `*out` is zeroed.
YamlCode build_index(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                     const json::PathFilter* filter, json::StructuralIndex* out,
                     YamlError* err) noexcept;

// Thin adapter: build_index, then the compact JSON text of the tape
// (json::value_from_index + json::serialize). Output bytes are bounded by
// limits.bytes. On success *out / *out_len describe arena bytes.
YamlCode to_json(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                 const char** out, uint32_t* out_len, YamlError* err) noexcept;

}  // namespace bolt::parse::yaml
