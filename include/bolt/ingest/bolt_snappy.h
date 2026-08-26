// bolt/ingest/bolt_snappy.h — W-PQ: minimal snappy BLOCK decompressor
// (the format parquet SNAPPY pages use — raw block format, NOT the
// framing format). Zero dependencies.
//
// Format (https://github.com/google/snappy/blob/main/format_description.txt):
//   [uvarint uncompressed_length] then a tag stream:
//     tag & 3 == 0 : literal,  len = (tag>>2)+1, or 60..63 => 1..4 extra
//                    length bytes (little-endian)
//     tag & 3 == 1 : copy, len = ((tag>>2)&7)+4, offset = ((tag>>5)<<8)|byte
//     tag & 3 == 2 : copy, len = (tag>>2)+1, offset = 2-byte LE
//     tag & 3 == 3 : copy, len = (tag>>2)+1, offset = 4-byte LE
//   Copies may overlap forward (offset < len) — byte-by-byte semantics.
//
// Safety contract: every read/write is bounds-checked; corrupt input
// returns false, never UB (fuzzed in the parquet test suite under ASAN).
// Tiger Style: noexcept, no allocation, bounded loops, >=2 asserts.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace ingest {

// Parse the uncompressed-length preamble. Returns bytes consumed (1..5)
// or 0 on corrupt/oversized input.
inline uint32_t snappy_uncompressed_len(const uint8_t* src, uint64_t src_len,
                                        uint64_t* out_len) noexcept {
    assert(out_len != nullptr);
    if (src == nullptr || src_len == 0) return 0;
    uint64_t v = 0;
    uint32_t i = 0;
    for (; i < 5 && i < src_len; ++i) {
        v |= static_cast<uint64_t>(src[i] & 0x7Fu) << (7u * i);
        if ((src[i] & 0x80u) == 0) {
            if (v > (1ull << 32)) return 0;   // snappy caps at 2^32
            *out_len = v;
            return i + 1;
        }
    }
    return 0;
}

// Copy an 8-byte word. Fixed size, so this compiles to one load + one
// store — never a call into libc's runtime size-dispatch ladder.
inline void snappy_copy8(uint8_t* dst, const uint8_t* src) noexcept {
    uint64_t w;
    std::memcpy(&w, src, 8);
    std::memcpy(dst, &w, 8);
}

// Copy `len` bytes to dst+op from dst+op-off, where `off` may be SMALLER
// than `len` (a repeating pattern — snappy's overlapping copy).
//
// Why not std::memcpy: a snappy back-reference is 1..64 bytes BY FORMAT, so
// a runtime-length memcpy pays a call plus libc's size dispatch to move a
// handful of bytes. Measured at ~22% of total query CPU on TPC-H SF100
// (dtrace run 7f5b0152, `_platform_memmove` under decompress_page). This
// moves the same bytes with fixed-size 8-byte stores and no call.
//
// It OVERCOPIES up to 7 bytes past the run, so the caller must have proven
// `op + len + 8 <= dst_len`. Only the first `len` bytes are meaningful;
// anything past them is either rewritten by a later tag or lies beyond the
// final `op == dst_len` fill point.
//
// Every 8-byte read lands entirely inside bytes that are already final. For
// off >= 8 that is immediate. For off < 8 the offset is first grown to the
// smallest multiple of `off` that is >= 8 (legal because the source region
// is periodic with period `off`) and the few bytes that multiple needs are
// materialised one at a time — so this never reads uninitialised memory,
// which the naive "just read 16 bytes back" formulation does.
inline void snappy_copy_match(uint8_t* dst, uint64_t op, uint64_t off,
                              uint64_t len) noexcept {
    assert(off > 0 && off <= op);
    assert(len > 0);
    uint64_t k = 0;
    uint64_t d = off;
    if (d < 8) {
        while (d < 8) d += off;              // bounded: <= 8 iterations
        const uint64_t prime = d - off;      // <= 7
        const uint64_t stop = (prime < len) ? prime : len;
        for (; k < stop; ++k) dst[op + k] = dst[op + k - off];
    }
    for (; k < len; k += 8) {                // bounded: len <= 64 => <= 8
        snappy_copy8(dst + op + k, dst + op + k - d);
    }
}

// ---------------------------------------------------------------------------
// Tag decode table.
//
// A snappy tag byte determines, on its own, the element length, any offset
// bits carried inside the tag, and how many further input bytes the tag
// consumes. Deriving those with a `switch (tag & 3)` costs a multi-way branch
// on every element, and at a MEASURED 6.3 output bytes per tag on real parquet
// text the per-tag cost IS the decode cost.
//
// So resolve all three with one L1 lookup instead. 512 bytes, one cache line
// per 32 tags.
//
// GENERATED, not transcribed: the entries are computed from the format rules
// below at compile time, so the table cannot silently disagree with the spec
// the way a hand-copied literal table can (this file already carries the scar
// of exactly that — see bolt_zstd_dec.cpp, where a mis-transcribed RFC 8878
// distribution decoded a 100000-byte block to 38 bytes).
//
// Layout, chosen so `len` needs no shift on the hot path:
//     bits  0..7   element length      (copy: 1..64, literal: 1..60, 0 = long)
//     bits  8..10  offset bits held in the tag byte, in units of 256 (copy-1)
//     bits 11..13  extra input bytes consumed after the tag
struct SnappyTagTable { uint16_t e[256]; };

constexpr SnappyTagTable snappy_make_tag_table() noexcept {
    SnappyTagTable t{};
    for (int i = 0; i < 256; ++i) {
        const uint32_t tag = static_cast<uint32_t>(i);
        uint32_t len = 0, off_hi = 0, extra = 0;
        switch (tag & 3u) {
            case 0: {                       // literal
                const uint32_t inline_len = (tag >> 2) + 1u;
                // 61..64 encode "1..4 length bytes follow" rather than a
                // length. Store 0 as the sentinel that sends the tag to the
                // careful path; the fast loop never handles a long literal.
                len   = (inline_len <= 60u) ? inline_len : 0u;
                extra = (inline_len <= 60u) ? 0u : (inline_len - 60u);
                break;
            }
            case 1:                          // copy, 1 offset byte, 3 in tag
                len    = ((tag >> 2) & 7u) + 4u;
                off_hi = tag >> 5;           // 0..7, scaled by 256 at use
                extra  = 1u;
                break;
            case 2:                          // copy, 2 offset bytes
                len   = (tag >> 2) + 1u;
                extra = 2u;
                break;
            default:                         // copy, 4 offset bytes
                len   = (tag >> 2) + 1u;
                extra = 4u;
                break;
        }
        t.e[i] = static_cast<uint16_t>((extra << 11) | (off_hi << 8) | len);
    }
    return t;
}

inline constexpr SnappyTagTable kSnappyTag = snappy_make_tag_table();

// Offset mask per tag type, so the offset falls out of ONE 4-byte load with no
// branch: literal contributes nothing, copy-1 one byte, copy-2 two, copy-4 all
// four.
//
// MEASURED NEGATIVE: packing these into a single 64-bit constant and shifting
// by `type * 16` — what Google's ExtractOffset does on aarch64, specifically to
// avoid this table load — is slower here. 190.0 ms vs 187.5 on the same 512 MB,
// re-measured at 100% fast-loop coverage after the earlier reading (taken at
// 4.4% coverage, and therefore void) happened to point the same way.
//
// The array does compile to a real load (`ldr w10, [x22, w10, uxtw #2]`), but
// it is 16 bytes, permanently L1-resident and fully pipelined, whereas the
// shift-and-mask chain adds latency to the offset itself — which feeds the
// copy's source address. Worth revisiting only on a machine with a tighter
// load pipeline than this one.
inline constexpr uint32_t kSnappyOffMask[4] = {0u, 0xFFu, 0xFFFFu, 0xFFFFFFFFu};

// Input/output headroom the fast loop keeps in reserve so that NO per-tag
// bounds check is needed inside it.
//
// Worst case is set by the DEFERRED copy: the loop guard tests `op`, but the
// tag being decoded actually lands at `op + deferred_length`, up to 16 bytes
// further on. Writing from there:
//     copy element    op + 16 + 64 + 7   (format max length, plus overcopy)
//     long literal    op + 16 + 60
//     deferred flush  op + 16 + 16
// so 87 is the true bound and 96 is the round number above it. Reading:
//     next tag        ip + 65            (tag + 60-byte literal payload + 1)
//     literal payload ip + 1 + 60
//     offset word     ip + 5
// 96 covers those with room to spare.
//
// Getting this wrong is a heap overflow rather than a wrong answer, so it is
// derived here rather than tuned: the previous value of 80 was correct for the
// undeferred loop and would have been 6 bytes short once deferral moved the
// write point forward.
//
// This is where bolt departs from Google's decoder rather than copying it.
// Theirs requires the CALLER to append 64 slop bytes to the output buffer,
// because their fast loop runs to the end. Bolt cannot: snappy_decompress
// promises an exact fill, and three callers size their destination exactly
// (kafka_wire.cpp passes `needed`; both tests pass exactly-sized vectors), so
// a slop-requiring decoder would be a heap overflow in each. Stopping the fast
// loop 80 bytes early and letting the pre-existing careful loop finish the
// tail buys the same freedom from bounds checks with no contract change.
inline constexpr uint64_t kSnappyFastSlop = 96u;

// Decompress one snappy block into dst (exactly dst_len bytes — the
// caller sizes dst from snappy_uncompressed_len). False on any corrupt
// shape: truncated tags, out-of-range offsets, output over/underflow.
//
// Writes NEVER exceed dst_len. The wide-store fast lanes are taken only
// when there is provably room; the tail falls back to exact copies. That
// keeps the exact-fill contract every caller relies on (kafka_wire sizes
// its output buffer to exactly the decoded length, as do the tests) rather
// than requiring callers to append slop bytes.
inline bool snappy_decompress(const uint8_t* src, uint64_t src_len,
                              uint8_t* dst, uint64_t dst_len) noexcept {
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_len == 0);
    uint64_t hdr_len = 0;
    const uint32_t consumed = snappy_uncompressed_len(src, src_len, &hdr_len);
    if (consumed == 0 || hdr_len != dst_len) return false;
    uint64_t ip = consumed;
    uint64_t op = 0;

    // ---- fast loop -------------------------------------------------------
    // Runs only while both streams have kSnappyFastSlop bytes in hand, which
    // is what lets every per-tag buffer-overflow check go away: the loop
    // condition has already proven the room. The only test left inside is
    // `off > op`, which is not a bounds check but a correctness check on
    // untrusted input (a back-reference before the start of the block), and
    // it predicts perfectly on real data.
    //
    // Everything the careful loop below does, this does identically — it is a
    // faster route to the same bytes, not a different decode. The differential
    // fuzz in test_bolt_snappy.cpp compares both against a naive oracle.
    if (src_len > kSnappyFastSlop && dst_len > kSnappyFastSlop) {
        const uint64_t ip_fast = src_len - kSnappyFastSlop;
        const uint64_t op_fast = dst_len - kSnappyFastSlop;
        // The tag is carried in a REGISTER across iterations. This is the
        // whole point of the loop's shape, and it is worth more than the table
        // or the hoisted bounds checks combined.
        //
        // Written the obvious way, each iteration loads the tag, looks the tag
        // up in the table to learn how far to advance, then loads the next tag
        // from the new position: two DEPENDENT L1 loads, ~10 cycles, serialised
        // and unhideable. Measured at 10.6 cycles/tag, which is the entire
        // decode cost at 6.3 output bytes per tag.
        //
        // The advance below is therefore derived from the tag byte by
        // ARITHMETIC ONLY — never from the table — so the next tag's load
        // depends on nothing but the tag already in hand and issues at once,
        // while the table lookup for length/offset proceeds in parallel with
        // it. One load latency per tag instead of two chained.
        // DEFERRED COPY. A tag's bytes are not written in the iteration that
        // decodes them; they are written in the NEXT one. Two things come off
        // the critical path as a result:
        //
        //  1. `op`. Written directly the recurrence is
        //     op += len, and len comes from the table, which comes from a load
        //     of the tag — so each iteration's op waits on a load. Deferred it
        //     becomes op += deferred_length, whose addend was computed a full
        //     iteration earlier and is already in a register. One add.
        //
        //  2. the store. The copy for tag N issues after tag N+1's decode has
        //     started, so the store no longer sits between the decode and the
        //     next address computation.
        //
        // It also makes literal and copy share ONE copy site: the only thing
        // that differs is where the bytes come from, which is a csel on a
        // pointer rather than a branch.
        //
        // Honest note on attribution: deferral was measured at PARITY (251.6 ms
        // vs 252.3) when it landed, and the convergence it enables measured 2%
        // slower. Both numbers were taken while the fast loop was reached by
        // only 4.4% of tags, so neither described the real work, and neither
        // has been re-measured in isolation since coverage reached 100%. The
        // structure is kept because it is Google's and is sound; the claim that
        // it is what earns the speed is NOT established here. The measured win
        // is the resume fix below.
        //
        // `def_src` starts at a zeroed local so the first flush reads defined
        // bytes. It writes 16 bytes at op with def_len == 0, which the next
        // flush immediately overwrites at the same op — harmless, and it keeps
        // the loop free of a first-iteration special case.
        // MEASURED NEGATIVE — the two-way unroll, tried WITH the deferred copy
        // in place because Google's own comments tie the two together ("the
        // throughput is limited by instructions, unrolling the inner loop twice
        // reduces the amount of instructions checking limits").
        //
        // It is slower here, both ways round: codec 72.8 vs 71.8 ms, real
        // l_comment 962.7 vs 955.1. And it was genuinely unrolled, not silently
        // ignored — a plain `for (u < 2)` did NOT unroll (the tag-table load
        // `ldrh ..., uxtw #1` appeared once in the emitted arm64, and that
        // version was slower still, having added a counter for nothing), so it
        // needed `#pragma clang loop unroll_count(2)` to reach two loads. Both
        // configurations lost.
        //
        // The guard this removes is two compares and a predictable branch per
        // tag. On a machine this wide, with a reorder window this deep, that was
        // already hidden; unrolling only adds register pressure and code
        // footprint. It also forced kSnappyFastSlop from 96 to 192, since the
        // second tag runs before the guard is re-tested.
        //
        // Worth re-testing on a narrower core (and on x86), where the guard is a
        // larger share of the loop. Not here.
        uint8_t safe_src[16] = {0};
        const uint8_t* def_src = safe_src;
        uint64_t def_len = 0;
        uint32_t tag = src[ip];
        while (ip < ip_fast && op < op_fast) {
            const uint32_t type = tag & 3u;
            const uint32_t e    = kSnappyTag.e[tag];
            const uint64_t len  = e & 0xFFu;

            // Both from `tag` alone. Literal advance is 1 tag byte + len
            // payload bytes, and len == (tag >> 2) + 1, hence 2 + (tag >> 2).
            // Copy advance is 1 tag byte + its offset bytes: 1, 2 or 4.
            const uint64_t adv     = (type == 0u)
                                   ? (2u + (tag >> 2))
                                   : (1u + ((type == 3u) ? 4u : type));
            const uint64_t next_ip = ip + adv;
            // In bounds by the slop reservation: adv is at most 65.
            const uint32_t next_tag = src[next_ip];
            // MEASURED NEUTRAL: an explicit BOLT_PREFETCH_READ(src + ip + 128)
            // here, which is what Google's decoder does at this point, changed
            // nothing (codec 72.3 vs 71.8 ms, real l_comment 954.9 vs 955.1).
            // The stream is walked strictly forward at ~2.5 input bytes per
            // tag, so the hardware prefetcher already has the line long before
            // the tag load needs it. Not kept: it would also have made this
            // header depend on bolt_port.h, which it currently does not.

            // Where THIS tag's bytes land: past the copy still owed.
            const uint64_t cur_op = op + def_len;

            // Offset. For a literal the mask is 0 and the tag carries no
            // offset bits, so this yields 0 and is simply unused.
            uint32_t next;
            std::memcpy(&next, src + ip + 1u, 4);
            const uint64_t off = (e & 0x700u) | (next & kSnappyOffMask[type]);

            const bool is_lit = (type == 0u);
            // One branch for everything the straight-line lane cannot do:
            // a long literal (len sentinel 0), anything over 16 bytes, copy-4,
            // an overlapping back-reference, or a corrupt one. False for ~99%
            // of tags.
            if (len == 0u || len > 16u ||
                (!is_lit && (off < 8u || off > cur_op))) {
                snappy_copy8(dst + op, def_src);          // settle the debt
                snappy_copy8(dst + op + 8u, def_src + 8u);
                op = cur_op;
                def_src = safe_src;
                def_len = 0;

                if (is_lit) {
                    uint64_t l = len;
                    uint64_t s = ip + 1u;
                    if (l == 0u) {
                        // Literal of more than 60 bytes: 1..4 little-endian
                        // length bytes follow the tag.
                        const uint64_t nb = e >> 11;
                        uint64_t v = 0;
                        for (uint64_t k = 0; k < nb; ++k) {
                            v |= static_cast<uint64_t>(src[s + k]) << (8u * k);
                        }
                        l = v + 1u;
                        s += nb;
                        // A literal can be arbitrarily long, so this is the ONE
                        // place the fast loop may legitimately hand over — and
                        // handing over must stay rare, because it is permanent
                        // for the rest of the block.
                        //
                        // It was not rare. An earlier version simply broke out
                        // on every long literal, and long literals are ~0.4% of
                        // literals — about 88 per 1 MiB page. The first one
                        // ended the fast loop for that page, so MEASURED, the
                        // fast loop saw 4.4% of tags and the careful loop 95.6%,
                        // with exactly one exit per chunk (512 of 512). Every
                        // fast-loop optimisation was running on a twentieth of
                        // the data and read as noise. Decoding the length here
                        // and continuing took coverage to 100.0% and the
                        // microbenchmark from 252.3 ms to 187.4.
                        //
                        // A "fast path" that quietly stops being taken is worse
                        // than no fast path: it costs the complexity and pays
                        // nothing, and the benchmark still moves enough to look
                        // like progress. Count coverage, do not assume it.
                        if (s + l + kSnappyFastSlop > src_len ||
                            op + l + kSnappyFastSlop > dst_len) {
                            break;                        // ip still at the tag
                        }
                    }
                    std::memcpy(dst + op, src + s, l);
                    op += l;
                    ip  = s + l;
                    tag = src[ip];
                    continue;
                }
                if (off == 0u || off > op) return false;
                snappy_copy_match(dst, op, off, len);
                op += len;
                ip = next_ip;
                tag = next_tag;
                continue;
            }

            // Straight-line lane. The source is the only difference between a
            // literal and a copy, so select it and defer; the two fixed stores
            // below are settling the PREVIOUS tag, not this one.
            //
            // Safe for a copy because off >= 8 and the stores are sequential:
            // by the time the second reads at cur_op + 8 - off, the first has
            // already made cur_op .. cur_op+7 final. And the bytes this reads
            // were written by the flush of the tag before it, which happened in
            // the previous iteration — so they are final by program order.
            const uint8_t* from = is_lit ? (src + ip + 1u)
                                         : (dst + cur_op - off);
            snappy_copy8(dst + op, def_src);
            snappy_copy8(dst + op + 8u, def_src + 8u);
            op      = cur_op;
            def_src = from;
            def_len = len;
            ip      = next_ip;
            tag     = next_tag;
        }
        // Settle whatever the loop still owes before the careful loop resumes.
        if (def_len != 0u) {
            snappy_copy8(dst + op, def_src);
            snappy_copy8(dst + op + 8u, def_src + 8u);
            op += def_len;
        }
    }

    // ---- careful loop: the tail, and any tag the fast lane declined -------
    while (ip < src_len) {                       // bounded: ip advances >=1
        const uint8_t tag = src[ip++];
        if ((tag & 3u) == 0) {                   // literal
            uint64_t len = (tag >> 2) + 1u;
            if (len > 60u) {
                const uint32_t nb = static_cast<uint32_t>(len - 60u);
                if (ip + nb > src_len) return false;
                uint64_t v = 0;
                for (uint32_t k = 0; k < nb; ++k) {
                    v |= static_cast<uint64_t>(src[ip + k]) << (8u * k);
                }
                ip += nb;
                len = v + 1u;
            }
            if (ip + len > src_len || op + len > dst_len) return false;
            // Short literals dominate dictionary/RLE-encoded parquet pages.
            // Two fixed 8-byte moves beat a call whose whole job is to
            // dispatch on a size that is almost always < 16.
            if (len <= 16u && ip + 16u <= src_len && op + 16u <= dst_len) {
                snappy_copy8(dst + op, src + ip);
                snappy_copy8(dst + op + 8u, src + ip + 8u);
            } else {
                std::memcpy(dst + op, src + ip, len);
            }
            ip += len;
            op += len;
            continue;
        }
        uint64_t len = 0, off = 0;
        switch (tag & 3u) {
            case 1:
                if (ip + 1 > src_len) return false;
                len = ((tag >> 2) & 7u) + 4u;
                off = (static_cast<uint64_t>(tag >> 5) << 8) | src[ip];
                ip += 1;
                break;
            case 2:
                if (ip + 2 > src_len) return false;
                len = (tag >> 2) + 1u;
                off = static_cast<uint64_t>(src[ip]) |
                      (static_cast<uint64_t>(src[ip + 1]) << 8);
                ip += 2;
                break;
            default:   // 3
                if (ip + 4 > src_len) return false;
                len = (tag >> 2) + 1u;
                std::uint32_t o32;
                std::memcpy(&o32, src + ip, 4);
                off = o32;
                ip += 4;
                break;
        }
        if (off == 0 || off > op) return false;          // bad back-ref
        if (op + len > dst_len) return false;            // output overflow
        // THE lane. Measured over 512 MB of real TPC-H l_comment text
        // (85.5M tags, 6.3 output bytes per tag, so per-tag cost is
        // everything): 86.7% of tags are matches, and 99.3% of those are
        // len <= 16 with off >= 8. Serve exactly that with two unconditional
        // 8-byte stores — no trip count, no loop, no overlap test.
        //
        // snappy_copy_match below is correct for all of it, but pays a
        // data-dependent `for (k < len; k += 8)` whose exit is unpredictable
        // (75.5% of matches want one iteration, 23.9% want two) plus a branch
        // on the off < 8 overlap case that only 0.11% of matches ever take.
        // Overcopying to a fixed 16 costs nothing here and deletes both.
        //
        // Safe because off >= 8 means the second load starts at op+8-off <= op,
        // and the first store has already made dst[op .. op+7] final. Bytes
        // past op+len are meaningless but in bounds, and a later tag rewrites
        // them — the same argument snappy_copy_match's overcopy already relies
        // on, with a fixed bound instead of a computed one.
        if (len <= 16u && off >= 8u && op + 16u <= dst_len) {
            snappy_copy8(dst + op, dst + op - off);
            snappy_copy8(dst + op + 8u, dst + op + 8u - off);
        } else if (op + len + 8u <= dst_len) {           // room to overcopy
            snappy_copy_match(dst, op, off, len);
        } else if (off >= len) {                         // tail, no overlap
            std::memcpy(dst + op, dst + op - off, len);
        } else {                                         // tail, overlapping
            for (uint64_t k = 0; k < len; ++k) {
                dst[op + k] = dst[op + k - off];
            }
        }
        op += len;
    }
    return op == dst_len;                        // exact fill required
}

// ---------------------------------------------------------------------------
// W-PQ-W: minimal Snappy block COMPRESSOR.
//
// Encodes input as a single literal chunk preceded by the uncompressed-length
// varint. The format permits this — a literal-only stream is a valid Snappy
// block (just no compression benefit). Parquet's SNAPPY codec accepts it,
// and the decoder above round-trips it exactly. This keeps the encoder
// dependency-free and Tiger-Style tiny while still letting us claim
// "compression = 1 / SNAPPY" in the page metadata.
//
// Worst-case output is `src_len + varint_len(src_len) + 5` (one literal
// header per <= 4 GiB chunk). `snappy_max_compressed_len` gives a safe
// upper bound the caller sizes `dst` from.
// ---------------------------------------------------------------------------

inline uint64_t snappy_max_compressed_len(uint64_t src_len) noexcept {
    // Snappy's own bound: 32 + n + n/6. The previous `n + 32` was only valid
    // for a compressor that emitted ONE literal chunk; a real one emits a tag
    // per literal run, and on incompressible input that is a byte per ~60.
    return 32u + src_len + (src_len / 6u);
}

namespace snappy_detail {

// Snappy compresses in independent 64 KiB blocks, so a copy offset always
// fits two bytes and the hash table stays small enough to live in L1. This
// mirrors the reference implementation's block size for the same reasons.
inline constexpr uint64_t kBlockSize = 65536u;
inline constexpr uint32_t kHashBits  = 14u;      // 16 KiB table
inline constexpr uint32_t kHashSize  = 1u << kHashBits;
inline constexpr uint32_t kMinMatch  = 4u;
// The format's own limits: a copy carries at most 64 bytes, and the last 5
// bytes of a block must be literals so the decoder's wide copies stay in
// bounds.
inline constexpr uint32_t kMaxCopy   = 64u;
inline constexpr uint32_t kInputMargin = 15u;

inline uint32_t sn_load32(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint32_t sn_hash(uint32_t v) noexcept {
    return (v * 0x1E35A7BDu) >> (32u - kHashBits);
}

// Emit a literal run of `len` bytes from `lit`. Tag low 2 bits 00; the upper
// 6 bits carry len-1 when it fits, otherwise 60..63 select 1..4 extra
// little-endian length bytes.
inline bool sn_emit_literal(uint8_t* dst, uint64_t cap, uint64_t* op,
                            const uint8_t* lit, uint64_t len) noexcept {
    if (len == 0u) return true;
    const uint64_t n = len - 1u;
    if (n < 60u) {
        if (*op + 1u + len > cap) return false;
        dst[(*op)++] = static_cast<uint8_t>(n << 2);
    } else {
        uint32_t extra = 1u;
        if (n > 0xFFu)     extra = 2u;
        if (n > 0xFFFFu)   extra = 3u;
        if (n > 0xFFFFFFu) extra = 4u;
        if (*op + 1u + extra + len > cap) return false;
        dst[(*op)++] = static_cast<uint8_t>((59u + extra) << 2);
        for (uint32_t k = 0; k < extra; ++k) {
            dst[(*op)++] = static_cast<uint8_t>((n >> (8u * k)) & 0xFFu);
        }
    }
    std::memcpy(dst + *op, lit, len);
    *op += len;
    return true;
}

// Emit a back-reference. Two encodings are used: the 1-byte-offset form
// (tag 01) covers lengths 4..11 at offsets under 2048 in two bytes total,
// and the 2-byte-offset form (tag 10) covers lengths 1..64 at offsets under
// 65536 in three. Longer matches are split, and the split is NOT arbitrary:
// leaving a remainder of 1..3 would need a copy shorter than the 4-byte
// minimum, so a match that would leave such a tail is cut at 60 instead.
inline bool sn_emit_copy(uint8_t* dst, uint64_t cap, uint64_t* op,
                         uint32_t offset, uint32_t len) noexcept {
    assert(offset >= 1u && offset <= 65535u);
    while (len >= 68u) {                       // 64 + at least 4 left over
        if (*op + 3u > cap) return false;
        dst[(*op)++] = static_cast<uint8_t>(((kMaxCopy - 1u) << 2) | 2u);
        dst[(*op)++] = static_cast<uint8_t>(offset & 0xFFu);
        dst[(*op)++] = static_cast<uint8_t>((offset >> 8) & 0xFFu);
        len -= kMaxCopy;
    }
    if (len > kMaxCopy) {                      // 65..67 -> 60 + (5..7)
        if (*op + 3u > cap) return false;
        dst[(*op)++] = static_cast<uint8_t>(((60u - 1u) << 2) | 2u);
        dst[(*op)++] = static_cast<uint8_t>(offset & 0xFFu);
        dst[(*op)++] = static_cast<uint8_t>((offset >> 8) & 0xFFu);
        len -= 60u;
    }
    if (len >= 4u && len <= 11u && offset < 2048u) {
        if (*op + 2u > cap) return false;
        dst[(*op)++] = static_cast<uint8_t>(1u | ((len - 4u) << 2) |
                                            ((offset >> 8) << 5));
        dst[(*op)++] = static_cast<uint8_t>(offset & 0xFFu);
        return true;
    }
    if (*op + 3u > cap) return false;
    dst[(*op)++] = static_cast<uint8_t>(((len - 1u) << 2) | 2u);
    dst[(*op)++] = static_cast<uint8_t>(offset & 0xFFu);
    dst[(*op)++] = static_cast<uint8_t>((offset >> 8) & 0xFFu);
    return true;
}

}  // namespace snappy_detail

// Compress `src` into snappy's block format.
//
// This USED to emit one literal chunk spanning the whole input -- a valid
// snappy stream that compresses nothing. Since SNAPPY is parquet's most
// common codec and bolt's writer default, every "compressed" parquet file
// bolt produced was the same size as an uncompressed one, and the ratio said
// so: 53,008,583 bytes against 53,008,131 uncompressed, while LZ4 managed
// 2.37x on identical input.
//
// Greedy LZ77 over independent 64 KiB blocks, matching the reference
// implementation's structure. `scratch` is optional caller-supplied hash
// table storage; when null a block-local table is used, which is fine for
// page-sized inputs.
inline bool snappy_compress(const uint8_t* src, uint64_t src_len,
                            uint8_t* dst, uint64_t dst_cap,
                            uint64_t* out_len) noexcept {
    using namespace snappy_detail;
    assert(src != nullptr || src_len == 0);
    assert(dst != nullptr || dst_cap == 0);
    assert(out_len != nullptr);
    if (src_len > (uint64_t{1} << 32)) return false;       // snappy caps at 2^32

    uint64_t op = 0;
    {   // preamble: uncompressed length as a varint
        uint64_t v = src_len;
        for (int i = 0; i < 5; ++i) {
            if (op >= dst_cap) return false;
            const uint8_t b = static_cast<uint8_t>(v & 0x7Fu);
            v >>= 7;
            if (v == 0) { dst[op++] = b; break; }
            dst[op++] = static_cast<uint8_t>(b | 0x80u);
        }
    }
    if (src_len == 0) { *out_len = op; return true; }

    uint16_t table[kHashSize];
    for (uint64_t block = 0; block < src_len; block += kBlockSize) {
        const uint64_t blen = (src_len - block < kBlockSize)
            ? (src_len - block) : kBlockSize;
        const uint8_t* base = src + block;
        // A block too short to hold a match is pure literal.
        if (blen < kInputMargin) {
            if (!sn_emit_literal(dst, dst_cap, &op, base, blen)) return false;
            continue;
        }
        std::memset(table, 0, sizeof(table));
        uint32_t ip = 0;        // scan cursor within the block
        uint32_t anchor = 0;    // start of the pending literal run
        const uint32_t limit = static_cast<uint32_t>(blen) - kInputMargin;
        while (ip < limit) {
            // Greedy search with step acceleration: after a run of misses,
            // sample less densely so incompressible input stays linear.
            uint32_t match = 0;
            bool found = false;
            uint32_t skip = 32u;
            while (ip < limit) {
                const uint32_t h = sn_hash(sn_load32(base + ip));
                const uint32_t cand = table[h];
                table[h] = static_cast<uint16_t>(ip);
                if (cand != 0u && ip - cand <= 65535u &&
                    sn_load32(base + cand) == sn_load32(base + ip)) {
                    match = cand;
                    found = true;
                    break;
                }
                ip += (skip++ >> 5);
            }
            if (!found) break;

            // Extend forward, stopping short of the mandatory trailing
            // literals so the decoder's wide copies stay in bounds.
            uint32_t mlen = kMinMatch;
            const uint32_t mmax = static_cast<uint32_t>(blen) - 5u;
            while (ip + mlen < mmax && base[match + mlen] == base[ip + mlen]) {
                ++mlen;
            }
            if (!sn_emit_literal(dst, dst_cap, &op, base + anchor,
                                 ip - anchor)) {
                return false;
            }
            if (!sn_emit_copy(dst, dst_cap, &op, ip - match, mlen)) {
                return false;
            }
            ip += mlen;
            anchor = ip;
            // Index a position inside the match so a following match can
            // reference into it.
            if (ip < limit) {
                table[sn_hash(sn_load32(base + ip - 1u))] =
                    static_cast<uint16_t>(ip - 1u);
            }
        }
        if (!sn_emit_literal(dst, dst_cap, &op, base + anchor,
                             static_cast<uint32_t>(blen) - anchor)) {
            return false;
        }
    }
    *out_len = op;
    return true;
}

}  // namespace ingest
}  // namespace bolt
