#pragma once
#include <cstdint>
namespace bolt::ingest {
inline constexpr std::uint32_t kZstdStreamBlockBytes = 128u*1024u;
inline constexpr std::uint32_t kZstdStreamAlignment = 8;
struct ZstdStream;
struct ZstdStreamLimits {
    std::uint64_t window_bytes = 0;
    std::uint64_t max_decoded_bytes = 0; // Logical total limit, not buffer capacity.
};
enum class ZstdStreamStatus : std::uint8_t {
    Progress, NeedInput, NeedOutput, FrameEnd, Finished,
    InvalidArgument, WorkspaceSmall, BadInput, Truncated, ChecksumMismatch,
    UnsupportedDictionaryEncoding, WindowLimit, OutputLimit, CounterOverflow,
    UnsupportedWindow
};
enum class ZstdFrameIntegrity : std::uint8_t { Pending, ChecksumVerified, NoChecksum };
struct ZstdStep {
    ZstdStreamStatus status = ZstdStreamStatus::InvalidArgument;
    std::uint32_t input_consumed = 0, output_produced = 0;
    std::uint64_t frame_index = 0, frame_decoded_bytes = 0, total_decoded_bytes = 0;
    ZstdFrameIntegrity integrity = ZstdFrameIntegrity::Pending;
};
// Counters describe decoded (possibly not yet drained) bytes. frame_index is
// the1-based admitted data-frame index; skippable frames do not increment it.
// Integrity describes that frame. Only FrameEnd certifies complete framing/FCS;
// ChecksumVerified is an integrity check, never source/durability authority.
// Fixed state includes entropy/literal scratch and one encoded/decoded block.
// The caller separately owns window_bytes history. No allocation or IO occurs.
// Admitted entropy profile: offset codes0..31, no dictionary-ID field (even
// encoded zero), frame windows<=UINT32_MAX. Larger advertised windows return
// UnsupportedWindow BEFORE caller WindowLimit; limits never silently clamp.
std::uint64_t zstd_stream_state_size() noexcept;
// All ranges (including out) disjoint, nonwrapping and caller-owned/quiescent.
// State storage is8-byte aligned. Admission errors preserve out and memory.
// Returns Progress on success. Re-init is the only reset after terminal error.
ZstdStreamStatus zstd_stream_init(void* state, std::uint64_t state_bytes,
    std::uint8_t* history, std::uint64_t history_bytes, ZstdStreamLimits limits,
    ZstdStream** out) noexcept;
// At most one bounded action/block per call; at most128KiB consumed/produced.
// Input consumed may be reused immediately; no input/output views are retained.
// Resubmit unconsumed input. final_input means no bytes AFTER this span; keep it
// true when resubmitting a suffix. Repeated empty final calls drain/boundary/EOF.
// Once the final span is wholly consumed, later nonempty input refuses; callers
// may omit final_input on subsequent empty calls (the end is retained).
// FrameEnd is delivered before next frame. Finished requires final input at a
// clean boundary; trailing partial headers/blocks/checksums/skips are Truncated.
// Output is PROVISIONAL until its frame validates. A later checksum failure
// cannot undo prior output. No-checksum frames are never ChecksumVerified.
// Alias/invalid-argument refusal is observational; malformed/limit errors are
// sticky. No output from the action discovering a terminal error is published.
// Callers poll cancellation/deadlines BETWEEN calls; one block is the CPU bound.
// Synchronous IO cancellation belongs to the caller, not this decoder.
ZstdStep zstd_stream_step(ZstdStream*, const std::uint8_t* input,
    std::uint32_t input_bytes, bool final_input, std::uint8_t* output,
    std::uint32_t output_capacity) noexcept;
} // namespace bolt::ingest
