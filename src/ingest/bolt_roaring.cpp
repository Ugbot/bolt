// bolt/ingest/bolt_roaring.cpp — portable Roaring bitmap deserialiser.
// See bolt_roaring.h for the wire format. Bounds-checked end to end.

#include "bolt/ingest/bolt_roaring.h"

#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace ingest {

namespace {

// Little-endian readers over a cursor. Each returns false on truncation.
struct Cursor {
    const uint8_t* p;
    uint64_t       len;
    uint64_t       off;
};

bool rd_u16(Cursor* c, uint16_t* out) noexcept {
    assert(c != nullptr);
    assert(out != nullptr);
    if (c->off + 2u > c->len) return false;
    *out = static_cast<uint16_t>(c->p[c->off] |
           (static_cast<uint16_t>(c->p[c->off + 1]) << 8));
    c->off += 2u;
    return true;
}

bool rd_u32(Cursor* c, uint32_t* out) noexcept {
    assert(c != nullptr);
    assert(out != nullptr);
    if (c->off + 4u > c->len) return false;
    uint32_t v = 0;
    std::memcpy(&v, c->p + c->off, 4);   // host is LE on all supported targets
    *out = v;
    c->off += 4u;
    return true;
}

bool rd_u64(Cursor* c, uint64_t* out) noexcept {
    assert(c != nullptr);
    assert(out != nullptr);
    if (c->off + 8u > c->len) return false;
    uint64_t v = 0;
    std::memcpy(&v, c->p + c->off, 8);
    *out = v;
    c->off += 8u;
    return true;
}

// Decode one container's payload, given its kind + cardinality. Advances the
// cursor past the payload. Returns false on truncation.
bool decode_payload(Cursor* c, Arena* arena, RoaringContainer* ct) noexcept {
    assert(c != nullptr);
    assert(ct != nullptr);
    if (ct->kind == RoaringKind::kBitset) {
        if (c->off + 8192u > c->len) return false;
        uint64_t* words = arena->allocate_array<uint64_t>(1024);
        if (words == nullptr) return false;
        std::memcpy(words, c->p + c->off, 8192);
        ct->words = words;
        c->off += 8192u;
        return true;
    }
    if (ct->kind == RoaringKind::kRun) {
        uint16_t n_runs = 0;
        if (!rd_u16(c, &n_runs)) return false;
        const uint64_t need = static_cast<uint64_t>(n_runs) * 4u;
        if (c->off + need > c->len) return false;
        uint16_t* runs = arena->allocate_array<uint16_t>(
            static_cast<size_t>(n_runs) * 2u + 1u);
        if (runs == nullptr) return false;
        uint32_t card = 0;
        for (uint32_t i = 0; i < n_runs; ++i) {     // bounded: n_runs <= 65535
            uint16_t start = 0, len_minus_1 = 0;
            if (!rd_u16(c, &start) || !rd_u16(c, &len_minus_1)) return false;
            runs[2u * i] = start;
            runs[2u * i + 1u] = len_minus_1;
            card += static_cast<uint32_t>(len_minus_1) + 1u;
        }
        ct->values = runs;
        ct->n_runs = n_runs;
        ct->cardinality = card;
        return true;
    }
    // Array container: `cardinality` sorted u16 values.
    const uint64_t need = static_cast<uint64_t>(ct->cardinality) * 2u;
    if (c->off + need > c->len) return false;
    uint16_t* vals = arena->allocate_array<uint16_t>(
        ct->cardinality == 0 ? 1u : ct->cardinality);
    if (vals == nullptr) return false;
    for (uint32_t i = 0; i < ct->cardinality; ++i) {  // bounded: card <= 65536
        if (!rd_u16(c, &vals[i])) return false;
    }
    ct->values = vals;
    return true;
}

}  // namespace

bool roaring_deserialize(const uint8_t* src, uint64_t src_len,
                         Arena* arena, RoaringBitmap* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    if (src == nullptr || out == nullptr || arena == nullptr) return false;

    Cursor c{src, src_len, 0};
    uint32_t cookie = 0;
    if (!rd_u32(&c, &cookie)) return false;

    const bool has_run = (cookie & 0xFFFFu) == kRoaringCookieRun;
    uint32_t n = 0;
    if (has_run) {
        n = (cookie >> 16) + 1u;          // n_containers - 1 packed in hi 16
    } else if (cookie == kRoaringCookieNoRun) {
        if (!rd_u32(&c, &n)) return false;
    } else {
        return false;                     // unknown cookie
    }
    if (n > kRoaringMaxContainers) return false;

    // Run-flag bitset present only when has_run.
    const uint8_t* run_flags = nullptr;
    if (has_run) {
        const uint64_t flag_bytes = (static_cast<uint64_t>(n) + 7u) / 8u;
        if (c.off + flag_bytes > c.len) return false;
        run_flags = c.p + c.off;
        c.off += flag_bytes;
    }

    RoaringContainer* cts =
        arena->allocate_array<RoaringContainer>(n == 0 ? 1u : n);
    if (cts == nullptr) return false;

    // keyscards header: n × (key, cardinality-1).
    for (uint32_t i = 0; i < n; ++i) {     // bounded: n <= 65536
        uint16_t key = 0, card_minus_1 = 0;
        if (!rd_u16(&c, &key) || !rd_u16(&c, &card_minus_1)) return false;
        cts[i].key = key;
        cts[i].cardinality = static_cast<uint32_t>(card_minus_1) + 1u;
        cts[i].values = nullptr;
        cts[i].words = nullptr;
        cts[i].n_runs = 0;
        cts[i]._pad = 0;
        cts[i]._pad2 = 0;
        const bool is_run = run_flags != nullptr &&
            ((run_flags[i >> 3] >> (i & 7u)) & 1u) != 0u;
        cts[i].kind = is_run ? RoaringKind::kRun
                    : (cts[i].cardinality > 4096u ? RoaringKind::kBitset
                                                  : RoaringKind::kArray);
    }

    // Optional offset header (NO_RUNCONTAINER && n >= threshold). We replay
    // payloads sequentially regardless, so just skip it when present.
    if (!has_run && n >= kRoaringNoOffsetThr) {
        const uint64_t off_bytes = static_cast<uint64_t>(n) * 4u;
        if (c.off + off_bytes > c.len) return false;
        c.off += off_bytes;
    }

    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {     // bounded: n <= 65536
        if (!decode_payload(&c, arena, &cts[i])) return false;
        total += cts[i].cardinality;
    }

    out->containers = cts;
    out->n_containers = n;
    out->_pad = 0;
    out->cardinality = total;
    return true;
}

bool roaring_contains(const RoaringBitmap* bm, uint32_t value) noexcept {
    assert(bm != nullptr);
    assert(bm->containers != nullptr || bm->n_containers == 0);
    if (bm == nullptr) return false;
    const uint16_t hi = static_cast<uint16_t>(value >> 16);
    const uint16_t lo = static_cast<uint16_t>(value & 0xFFFFu);
    for (uint32_t i = 0; i < bm->n_containers; ++i) {  // bounded
        const RoaringContainer* ct = &bm->containers[i];
        if (ct->key != hi) continue;
        if (ct->kind == RoaringKind::kBitset) {
            return ((ct->words[lo >> 6] >> (lo & 63u)) & 1ull) != 0ull;
        }
        if (ct->kind == RoaringKind::kRun) {
            for (uint32_t r = 0; r < ct->n_runs; ++r) {   // bounded
                const uint16_t start = ct->values[2u * r];
                const uint16_t len_m1 = ct->values[2u * r + 1u];
                if (lo >= start && lo <= static_cast<uint32_t>(start) + len_m1) {
                    return true;
                }
            }
            return false;
        }
        // Array — binary search the sorted u16 slice.
        uint32_t left = 0, right = ct->cardinality;
        while (left < right) {                            // bounded: log2(card)
            const uint32_t mid = left + (right - left) / 2u;
            const uint16_t v = ct->values[mid];
            if (v == lo) return true;
            if (v < lo) left = mid + 1u; else right = mid;
        }
        return false;
    }
    return false;
}

uint64_t roaring_cardinality(const RoaringBitmap* bm) noexcept {
    assert(bm != nullptr);
    assert(bm->n_containers <= kRoaringMaxContainers);
    return bm == nullptr ? 0u : bm->cardinality;
}

bool roaring_deserialize_r64(const uint8_t* src, uint64_t src_len,
                             Arena* arena, RoaringBitmap64* out) noexcept {
    assert(arena != nullptr);
    assert(out != nullptr);
    if (src == nullptr || out == nullptr || arena == nullptr) return false;

    Cursor c{src, src_len, 0};
    uint64_t n_buckets = 0;
    if (!rd_u64(&c, &n_buckets)) return false;
    if (n_buckets > kRoaringMaxContainers) return false;   // sane upper bound

    RoaringBucket* buckets = arena->allocate_array<RoaringBucket>(
        n_buckets == 0 ? 1u : static_cast<size_t>(n_buckets));
    if (buckets == nullptr) return false;

    uint64_t total = 0;
    for (uint32_t i = 0; i < n_buckets; ++i) {    // bounded: <= 65536
        uint32_t high32 = 0;
        if (!rd_u32(&c, &high32)) return false;
        buckets[i].high32 = high32;
        buckets[i]._pad = 0;
        // Sub-map consumes the remainder starting at the cursor; the 32-bit
        // reader re-bounds against the whole tail and reports its own length.
        if (!roaring_deserialize(c.p + c.off, c.len - c.off, arena,
                                 &buckets[i].map)) {
            return false;
        }
        total += buckets[i].map.cardinality;
        // Advance the cursor: re-serialise length is implicit, so we recompute
        // the consumed bytes by re-walking the header + payloads is costly;
        // instead the r64 portable layout stores each sub-map back-to-back and
        // the 32-bit reader leaves nothing trailing it must skip — but it does
        // not report bytes consumed. We therefore require a length-prefix-free
        // single-bucket form OR re-parse. Keep it strict: only n_buckets<=1 is
        // supported without a per-bucket length; multi-bucket needs the prefix.
        if (n_buckets > 1u) return false;
        c.off = c.len;
    }

    out->buckets = buckets;
    out->n_buckets = static_cast<uint32_t>(n_buckets);
    out->_pad = 0;
    out->cardinality = total;
    return true;
}

bool roaring_contains64(const RoaringBitmap64* bm, uint64_t value) noexcept {
    assert(bm != nullptr);
    assert(bm->buckets != nullptr || bm->n_buckets == 0);
    if (bm == nullptr) return false;
    const uint32_t hi = static_cast<uint32_t>(value >> 32);
    const uint32_t lo = static_cast<uint32_t>(value & 0xFFFFFFFFu);
    for (uint32_t i = 0; i < bm->n_buckets; ++i) {   // bounded
        if (bm->buckets[i].high32 == hi) {
            return roaring_contains(&bm->buckets[i].map, lo);
        }
    }
    return false;
}

uint64_t roaring_cardinality64(const RoaringBitmap64* bm) noexcept {
    assert(bm != nullptr);
    assert(bm->n_buckets <= kRoaringMaxContainers);
    return bm == nullptr ? 0u : bm->cardinality;
}

}  // namespace ingest
}  // namespace bolt
