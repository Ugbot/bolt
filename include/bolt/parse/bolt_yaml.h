#pragma once
// bolt::parse::yaml — YAML -> JSON text (G2GRAPH-453).
//
// A bounded reader for the YAML that configuration and Markdown frontmatter
// use, emitting compact JSON that the fionn reader (bolt_json.h) and every
// JSON consumer already accept. YAML 1.2 core schema: plain scalars type as
// null (`~`, `null`, empty), bool (`true`/`false`, any case form the spec
// lists), int (decimal, `0o`, `0x`) and float (decimal / exponent); every
// other scalar, and every quoted scalar, is a string. Dates stay strings.
//
// SUPPORTED: block mappings and sequences (including compact `- key: v`
// items, indentless sequences under a key, `- - x`), flow mappings and
// sequences (nested, multi-line), plain scalars (multi-line, folded), single
// and double quoted scalars (all escapes, folding), literal `|` and folded
// `>` block scalars with chomping and indentation indicators, comments, an
// optional leading `---` and trailing `...`.
//
// REFUSED BY NAME (YamlCode::kUnsupported, `what` names it): anchors and
// aliases, tags, directives, complex (`?`) and non-scalar keys, more than one
// document, tab indentation, non-finite floats, integers beyond int64.
//
// Budgets are the caller's (Limits registry entries of whoever calls):
// input bytes, nesting depth, value nodes, output bytes. Past one the call
// fails with kLimit naming it; output is never truncated.
//
// Tiger Style: noexcept, error-return, arena-owned output, no STL.

#include <cstdint>

namespace bolt { class Arena; }

namespace bolt::parse::yaml {

enum class YamlCode : uint8_t { Ok, Syntax, Unsupported, Limit, NoMemory };

struct YamlLimits {
    uint32_t bytes  = 0u;   // input bytes and output bytes (<= INT32_MAX)
    uint32_t depth  = 0u;   // nesting of collections (<= 256)
    uint32_t nodes  = 0u;   // values, collections included
};

struct YamlError {
    YamlCode    code;
    uint32_t    line;     // 1-based; 0 when not positional
    uint32_t    col;      // 1-based
    const char* what;     // static string naming the problem / construct / budget
};

// Convert `src` (one YAML document) to JSON. An empty document is `null`.
// On success *out / *out_len describe arena-owned bytes (NUL-terminated).
YamlCode to_json(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                 const char** out, uint32_t* out_len, YamlError* err) noexcept;

}  // namespace bolt::parse::yaml
