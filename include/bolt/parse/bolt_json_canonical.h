#pragma once
#include <cstdint>
namespace bolt { class Arena; }
namespace bolt::parse::json {
enum class CanonicalResult : uint8_t { Ok, Invalid, Limit, NoMemory };
struct CanonicalLimits {
    uint32_t bytes = 0u;  // maximum input AND output bytes, <= INT32_MAX
    uint32_t nodes = 0u;  // value nodes, including containers
    uint32_t depth = 0u;  // <= kJsonMaxDepth, root depth=0
};
// Lossless object comparison form, NOT RFC8785/JCS: object keys sort by decoded
// UTF-8 bytes; array order stays significant; strings/keys decode and re-escape
// to ASCII; decoded duplicate keys, invalid Unicode/UTF-8 and bad JSON refuse.
// NUMBER SPELLING IS PRESERVED, without float/int conversion: large integers,
// decimals, exponent text and signed zero cannot lose information. Thus 1.0
// and 1e0 remain different, while object ordering/whitespace/string escapes do
// not affect identity. No Unicode normalization. Existing writer modes unchanged.
// All scratch/output belongs to arena. Input need only survive this call.
// Outputs unchanged on failure; scratch may be consumed. No IO or mutation of
// caller input. Caller supplies finite byte/node/depth budgets.
CanonicalResult canonicalize(const uint8_t*, uint32_t, Arena*, CanonicalLimits,
    const uint8_t** output, uint32_t* output_bytes) noexcept;
} // namespace bolt::parse::json
