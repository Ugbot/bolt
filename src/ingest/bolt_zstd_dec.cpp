// bolt/ingest/bolt_zstd_dec.cpp — self-contained Zstandard decoder (RFC 8878).
//
// Structure, bottom-up:
//   1. bit readers      — forward (FSE header) and backward (entropy streams)
//   2. FSE              — normalized-count reader, decode-table builder
//   3. Huffman          — weight reader (direct + FSE-coded), table builder,
//                         1-stream and 4-stream literal decode
//   4. literals section — Raw / RLE / Compressed / Treeless
//   5. sequences        — 3 interleaved FSE states, repeat offsets
//   6. sequence exec    — literal copy + (overlapping) match copy
//   7. block / frame    — headers, xxhash64 content checksum
//
// SAFETY MODEL (this parses untrusted bytes):
//   * Every read from `src` is bounds-checked against src_len before the read.
//   * Every write to `dst` is bounds-checked against dst_cap before the write.
//   * Over-PEEKING a backward bitstream past its start is legal (zstd's final
//     symbols do it by construction) and shifts in zeros. Over-CONSUMING is
//     reported by zb_overrun(); it is corruption everywhere the symbol count is
//     known in advance, and the normal end-of-stream signal in the Huffman
//     weight decoder. See zb_overrun's comment.
//   * asserts state programmer invariants only — never input validity.

#include "bolt/ingest/bolt_zstd_dec.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace ingest {
namespace {

#include "bolt_zstd_dec_internal.h"

// Parse a frame header. Returns bytes consumed, or 0 on error.
uint32_t parse_frame_header(const uint8_t* p, uint64_t n, bool* checksum,
                            uint64_t* fcs_out, bool* have_fcs,
                            int* err) noexcept {
    assert(checksum != nullptr);
    assert(err != nullptr);
    assert(fcs_out != nullptr && have_fcs != nullptr);
    *err = kZstdOk;
    *have_fcs = false;
    *fcs_out = 0;
    if (n < 5) { *err = kZstdTruncSrc; return 0; }
    uint64_t o = 4;                                  // magic already matched
    const uint8_t fhd = p[o++];
    const uint32_t fcs_flag   = (fhd >> 6) & 3u;
    const uint32_t single_seg = (fhd >> 5) & 1u;
    const uint32_t reserved   = (fhd >> 3) & 1u;
    *checksum = ((fhd >> 2) & 1u) != 0;
    const uint32_t did_flag   = fhd & 3u;
    if (reserved != 0) { *err = kZstdBadInput; return 0; }
    if (!single_seg) {
        if (o >= n) { *err = kZstdTruncSrc; return 0; }
        ++o;                                         // window descriptor
    }
    static const uint32_t kDidLen[4] = {0, 1, 2, 4};
    const uint32_t did = kDidLen[did_flag];
    if (o + did > n) { *err = kZstdTruncSrc; return 0; }
    if (did != 0) {
        // A dictionary frame needs a dictionary we have no API to supply.
        // Refuse loudly rather than decode to wrong bytes.
        *err = kZstdUnsupported;
        return 0;
    }
    const uint32_t kFcsLen[4] = {single_seg ? 1u : 0u, 2u, 4u, 8u};
    const uint32_t fcs = kFcsLen[fcs_flag];
    if (o + fcs > n) { *err = kZstdTruncSrc; return 0; }
    if (fcs != 0) {
        uint64_t v = 0;
        for (uint32_t i = 0; i < fcs; ++i) {
            v |= static_cast<uint64_t>(p[o + i]) << (8u * i);
        }
        if (fcs == 2) v += 256u;          // the 2-byte form is biased
        *fcs_out = v;
        *have_fcs = true;
    }
    o += fcs;
    return static_cast<uint32_t>(o);
}

// Validate a frame's declared size and optional content checksum, consuming
// the 4 checksum bytes from *o when present.
int check_frame_trailer(const uint8_t* p, uint64_t n, uint64_t* o,
                        const uint8_t* dst, uint64_t frame_start, uint64_t wr,
                        bool have_fcs, uint64_t fcs, bool checksum) noexcept {
    assert(o != nullptr);
    assert(wr >= frame_start);
    // When the writer declared the frame's decompressed size, hold it to that.
    // A frame that decodes SHORT is corrupt, and without this check it would
    // be returned as a silently truncated result.
    if (have_fcs && (wr - frame_start) != fcs) return kZstdBadInput;
    if (!checksum) return kZstdOk;
    if (*o + 4 > n) return kZstdTruncSrc;
    const uint32_t want = static_cast<uint32_t>(p[*o]) |
                          (static_cast<uint32_t>(p[*o + 1]) << 8) |
                          (static_cast<uint32_t>(p[*o + 2]) << 16) |
                          (static_cast<uint32_t>(p[*o + 3]) << 24);
    *o += 4;
    const uint64_t got = xxhash64(dst + frame_start, wr - frame_start);
    return (static_cast<uint32_t>(got & 0xFFFFFFFFull) == want)
         ? kZstdOk : kZstdChecksum;
}

// Decode one frame starting at p[0]. *consumed gets the frame's byte length.
int decode_frame(Scratch* sc, const uint8_t* p, uint64_t n,
                 uint8_t* dst, uint64_t cap, uint64_t* wr,
                 uint64_t* consumed) noexcept {
    assert(sc != nullptr);
    assert(consumed != nullptr);
    bool     checksum = false;
    bool     have_fcs = false;
    uint64_t fcs = 0;
    int      err = kZstdOk;
    const uint32_t hdr = parse_frame_header(p, n, &checksum, &fcs, &have_fcs,
                                            &err);
    if (hdr == 0) return (err == kZstdOk) ? kZstdBadInput : err;

    sc->huf.valid = false;                 // Huffman/FSE reuse is frame-scoped
    sc->seq.have_ll = sc->seq.have_of = sc->seq.have_ml = false;
    uint32_t rep[3] = {1, 4, 8};
    const uint64_t frame_start = *wr;

    uint64_t o = hdr;
    for (uint32_t blk = 0; ; ++blk) {
        if (blk >= kMaxBlocksPerFrame) return kZstdBadInput;
        if (o + 3 > n) return kZstdTruncSrc;
        const uint32_t bh = static_cast<uint32_t>(p[o]) |
                            (static_cast<uint32_t>(p[o + 1]) << 8) |
                            (static_cast<uint32_t>(p[o + 2]) << 16);
        o += 3;
        const uint32_t last  = bh & 1u;
        const uint32_t btype = (bh >> 1) & 3u;
        const uint32_t bsize = bh >> 3;
        if (bsize > kBlockMax) return kZstdBadInput;
        if (btype == 3) return kZstdBadInput;          // reserved
        if (btype == 1) {                              // RLE block
            if (o + 1 > n) return kZstdTruncSrc;
            if (*wr + bsize > cap) return kZstdNeedMore;
            memset(dst + *wr, p[o], bsize);
            *wr += bsize;
            o += 1;
        } else {
            if (o + bsize > n) return kZstdTruncSrc;
            if (btype == 0) {                          // Raw block
                if (*wr + bsize > cap) return kZstdNeedMore;
                memcpy(dst + *wr, p + o, bsize);
                *wr += bsize;
            } else {                                   // Compressed block
                const int r = decode_compressed_block(sc, p + o, bsize, dst,
                                                      cap, wr, rep);
                if (r != kZstdOk) return r;
            }
            o += bsize;
        }
        if (last) break;
    }
    const int tr = check_frame_trailer(p, n, &o, dst, frame_start, *wr,
                                       have_fcs, fcs, checksum);
    if (tr != kZstdOk) return tr;
    *consumed = o;
    return kZstdOk;
}

// A skippable frame carries no data; step over it.
bool skippable_len(const uint8_t* p, uint64_t n, uint64_t* out) noexcept {
    assert(out != nullptr);
    if (n < 8) return false;
    const uint32_t sz = static_cast<uint32_t>(p[4]) |
                        (static_cast<uint32_t>(p[5]) << 8) |
                        (static_cast<uint32_t>(p[6]) << 16) |
                        (static_cast<uint32_t>(p[7]) << 24);
    if (static_cast<uint64_t>(sz) + 8u > n) return false;
    *out = static_cast<uint64_t>(sz) + 8u;
    return true;
}

}  // namespace

uint64_t zstd_scratch_size() noexcept {
    return sizeof(Scratch);
}

int zstd_decode_raw(const uint8_t* src, uint64_t src_len,
                    uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                    void* scratch, uint64_t scratch_len) noexcept {
    assert(out_len != nullptr);
    assert(src != nullptr || src_len == 0);
    if (out_len == nullptr) return kZstdBadInput;
    *out_len = 0;
    if (src == nullptr || src_len == 0) return kZstdTruncSrc;
    if (dst == nullptr && dst_cap != 0) return kZstdBadInput;
    if (scratch == nullptr || scratch_len < sizeof(Scratch)) {
        return kZstdScratchSmall;
    }
    if ((reinterpret_cast<uintptr_t>(scratch) & 7u) != 0) return kZstdBadInput;

    Scratch* sc = static_cast<Scratch*>(scratch);
    sc->huf.valid = false;

    uint64_t o  = 0;
    uint64_t wr = 0;
    for (uint32_t f = 0; o < src_len; ++f) {
        if (f >= kMaxFrames) return kZstdBadInput;
        if (o + 4 > src_len) return kZstdTruncSrc;
        const uint32_t magic = static_cast<uint32_t>(src[o]) |
                               (static_cast<uint32_t>(src[o + 1]) << 8) |
                               (static_cast<uint32_t>(src[o + 2]) << 16) |
                               (static_cast<uint32_t>(src[o + 3]) << 24);
        if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {     // skippable frame
            uint64_t sk = 0;
            if (!skippable_len(src + o, src_len - o, &sk)) return kZstdTruncSrc;
            o += sk;
            continue;
        }
        if (magic != 0xFD2FB528u) return kZstdBadInput;
        uint64_t used = 0;
        const int r = decode_frame(sc, src + o, src_len - o, dst, dst_cap,
                                   &wr, &used);
        if (r != kZstdOk) return r;
        if (used == 0 || used > src_len - o) return kZstdBadInput;
        o += used;
    }
    *out_len = wr;
    return kZstdOk;
}

}  // namespace ingest
}  // namespace bolt
