// Private Zstandard limits, included inside bolt::ingest anonymous namespace.
// uint32_t is supplied by the including translation unit.
#pragma once

// Kind A: maintained format/entropy-algorithm ceilings and derived table shapes.
// ---------------------------------------------------------------------------
// Bounded caps (Tiger Style: every buffer has a constexpr ceiling)
// ---------------------------------------------------------------------------
constexpr uint32_t kBlockMax      = 1u << 17;  // 128 KiB, zstd's block ceiling
constexpr uint32_t kHufMaxBits    = 11;        // spec: Huffman codes <= 11 bits
constexpr uint32_t kHufTableSize  = 1u << kHufMaxBits;
constexpr uint32_t kMaxSymbols    = 256;
constexpr uint32_t kLLMaxLog      = 9;
constexpr uint32_t kMLMaxLog      = 9;
constexpr uint32_t kOFMaxLog      = 8;
constexpr uint32_t kFseMaxLog     = 9;
constexpr uint32_t kFseTableSize  = 1u << kFseMaxLog;
constexpr uint32_t kWeightLog     = 6;         // Huffman-weight FSE accuracy
constexpr uint32_t kWeightTblSize = 1u << kWeightLog;
constexpr uint32_t kLLSymbolMax   = 35;
constexpr uint32_t kMLSymbolMax   = 52;
constexpr uint32_t kOFSymbolMax   = 31;
// Kind C: frame/block anti-abuse work budgets. A frame is a bounded number of blocks; guards against a crafted infinite
// stream of zero-length blocks.
[[maybe_unused]] constexpr uint32_t kMaxBlocksPerFrame = 1u << 20;
[[maybe_unused]] constexpr uint32_t kMaxFrames         = 1u << 16;

// Kind A: symbol ceilings of the predefined sequence tables.
constexpr uint32_t kLLDefMaxSym = 35;   // kLLDefault has 36 entries
constexpr uint32_t kMLDefMaxSym = 52;   // kMLDefault has 53 entries
constexpr uint32_t kOFDefMaxSym = 28;   // kOFDefault has 29 entries (not 32)
