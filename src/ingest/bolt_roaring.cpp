// bolt/ingest/bolt_roaring.cpp — portable Roaring bitmap (de)serialiser.
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

// Little-endian writers. Each bounds-checks against `cap` and returns false
// on overflow rather than writing OOB.
bool wr_u16(uint8_t* dst, uint64_t cap, uint64_t off, uint16_t v) noexcept {
    assert(dst != nullptr || cap == 0);
    if (off + 2u > cap) return false;
    dst[off] = static_cast<uint8_t>(v & 0xFFu);
    dst[off + 1u] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    return true;
}

bool wr_u32(uint8_t* dst, uint64_t cap, uint64_t off, uint32_t v) noexcept {
    assert(dst != nullptr || cap == 0);
    if (off + 4u > cap) return false;
    std::memcpy(dst + off, &v, 4);   // host is LE on all supported targets
    return true;
}

bool wr_u64(uint8_t* dst, uint64_t cap, uint64_t off, uint64_t v) noexcept {
    assert(dst != nullptr || cap == 0);
    if (off + 8u > cap) return false;
    std::memcpy(dst + off, &v, 8);
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

    // Offset header: ALWAYS present for the NO_RUNCONTAINER cookie form
    // (regardless of n — including n<4, the common single/few-container
    // shape a small deletion vector produces); present for the run-cookie
    // form only when n >= NO_OFFSET_THRESHOLD. Getting the NO_RUNCONTAINER
    // case wrong (gating it on n>=4 like the run form) made every n<4
    // no-run bitmap fail to skip 4*n real header bytes and misread them as
    // payload — verified against a real `pyroaring` byte dump (G2ICE-80).
    // We replay payloads sequentially regardless, so just skip it.
    if (has_run) {
        if (n >= kRoaringNoOffsetThr) {
            const uint64_t off_bytes = static_cast<uint64_t>(n) * 4u;
            if (c.off + off_bytes > c.len) return false;
            c.off += off_bytes;
        }
    } else {
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

bool roaring_to_sorted_array(const RoaringBitmap* bm, uint32_t* dst,
                             uint64_t dst_cap, uint64_t* out_n) noexcept {
    assert(bm != nullptr);
    assert(out_n != nullptr);
    if (out_n == nullptr) return false;
    *out_n = 0;
    if (bm == nullptr) return false;
    if (dst_cap < bm->cardinality) return false;
    if (dst == nullptr && dst_cap > 0) return false;
    uint64_t p = 0;
    for (uint32_t i = 0; i < bm->n_containers; ++i) {   // bounded
        const RoaringContainer* ct = &bm->containers[i];
        const uint32_t hi = static_cast<uint32_t>(ct->key) << 16;
        if (ct->kind == RoaringKind::kBitset) {
            for (uint32_t w = 0; w < 1024u; ++w) {      // bounded
                uint64_t bits = ct->words[w];
                while (bits != 0ull) {
                    const uint32_t bit = static_cast<uint32_t>(
                        bolt_ctz64(bits));
                    dst[p++] = hi | (w << 6) | bit;
                    bits &= bits - 1ull;
                }
            }
        } else if (ct->kind == RoaringKind::kRun) {
            for (uint32_t r = 0; r < ct->n_runs; ++r) {   // bounded
                const uint16_t start = ct->values[2u * r];
                const uint16_t len_m1 = ct->values[2u * r + 1u];
                for (uint32_t v = start; v <= static_cast<uint32_t>(start) + len_m1; ++v)
                    dst[p++] = hi | v;
            }
        } else {
            for (uint32_t c = 0; c < ct->cardinality; ++c)   // bounded
                dst[p++] = hi | ct->values[c];
        }
    }
    *out_n = p;
    return true;
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

// ---------------------------------------------------------------------------
// Serialisation (write path — G2ICE-80).
// ---------------------------------------------------------------------------

namespace {

struct Group {
    uint32_t key;      // high 16 bits (stored widened for alignment)
    uint64_t start;     // index into the caller's value array
    uint64_t count;
};

// Bounded single pass: `values` must already be strictly increasing (caller
// validates before calling this). Returns the number of high-16-bit groups.
uint64_t count_groups(const uint32_t* values, uint64_t n) noexcept {
    assert(values != nullptr || n == 0);
    if (n == 0) return 0;
    uint64_t k = 1;
    uint16_t cur = static_cast<uint16_t>(values[0] >> 16);
    for (uint64_t i = 1; i < n; ++i) {   // bounded by n
        const uint16_t key = static_cast<uint16_t>(values[i] >> 16);
        if (key != cur) { ++k; cur = key; }
    }
    return k;
}

void fill_groups(const uint32_t* values, uint64_t n, Group* groups,
                 uint64_t k) noexcept {
    assert(values != nullptr && groups != nullptr);
    assert(n > 0 && k > 0);
    uint64_t gi = 0;
    uint64_t start = 0;
    uint16_t cur = static_cast<uint16_t>(values[0] >> 16);
    for (uint64_t i = 1; i <= n; ++i) {   // bounded by n
        const uint16_t key = (i < n) ? static_cast<uint16_t>(values[i] >> 16)
                                     : 0xFFFFu /* sentinel — never matches
                                                  a real key when i==n */;
        const bool boundary = (i == n) || (key != cur);
        if (boundary) {
            assert(gi < k);
            groups[gi].key = cur;
            groups[gi].start = start;
            groups[gi].count = i - start;
            ++gi;
            if (i < n) { start = i; cur = key; }
        }
    }
    assert(gi == k);
}

}  // namespace

uint64_t roaring_serialize_bound(uint64_t n_values) noexcept {
    if (n_values == 0) return 8u;
    const uint64_t k_max = n_values < kRoaringMaxContainers
        ? n_values : static_cast<uint64_t>(kRoaringMaxContainers);
    return 8u + k_max * 8u + k_max * 8192u;
}

bool roaring_serialize(const uint32_t* sorted_unique_values, uint64_t n_values,
                       Arena* scratch, uint8_t* dst, uint64_t dst_cap,
                       uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out_len != nullptr);
    if (out_len == nullptr) return false;
    *out_len = 0;
    if (n_values > 0 && sorted_unique_values == nullptr) return false;
    if (dst == nullptr && dst_cap > 0) return false;
    if (scratch == nullptr) return false;
    for (uint64_t i = 1; i < n_values; ++i) {   // bounded: strictly increasing
        if (sorted_unique_values[i] <= sorted_unique_values[i - 1]) return false;
    }

    const uint64_t k = count_groups(sorted_unique_values, n_values);
    if (k > kRoaringMaxContainers) return false;
    Group* groups = scratch->allocate_array<Group>(k == 0 ? 1u : k);
    if (groups == nullptr) return false;
    if (k > 0) fill_groups(sorted_unique_values, n_values, groups, k);

    uint64_t off = 0;
    if (!wr_u32(dst, dst_cap, off, kRoaringCookieNoRun)) return false;
    off += 4u;
    if (!wr_u32(dst, dst_cap, off, static_cast<uint32_t>(k))) return false;
    off += 4u;
    for (uint64_t g = 0; g < k; ++g) {   // bounded: k <= 65536
        assert(groups[g].count >= 1 && groups[g].count <= 65536u);
        if (!wr_u16(dst, dst_cap, off, static_cast<uint16_t>(groups[g].key)))
            return false;
        off += 2u;
        if (!wr_u16(dst, dst_cap, off,
                    static_cast<uint16_t>(groups[g].count - 1u)))
            return false;
        off += 2u;
    }
    // Offset table: byte offset of each container's payload, measured from
    // the start of this blob (offset 0) — matches real CRoaring output.
    const uint64_t payload_region_start = off + k * 4u;
    uint64_t running = payload_region_start;
    const uint64_t offset_table_at = off;
    for (uint64_t g = 0; g < k; ++g) {
        const uint64_t this_off = offset_table_at + g * 4u;
        if (!wr_u32(dst, dst_cap, this_off, static_cast<uint32_t>(running)))
            return false;
        running += (groups[g].count <= 4096u) ? groups[g].count * 2u : 8192u;
    }
    off = payload_region_start;
    for (uint64_t g = 0; g < k; ++g) {   // bounded: k <= 65536
        const uint64_t cnt = groups[g].count;
        if (cnt <= 4096u) {
            for (uint64_t i = 0; i < cnt; ++i) {   // bounded: <= 4096
                const uint16_t lo = static_cast<uint16_t>(
                    sorted_unique_values[groups[g].start + i] & 0xFFFFu);
                if (!wr_u16(dst, dst_cap, off, lo)) return false;
                off += 2u;
            }
        } else {
            if (off + 8192u > dst_cap) return false;
            std::memset(dst + off, 0, 8192u);
            uint64_t* words = reinterpret_cast<uint64_t*>(dst + off);
            for (uint64_t i = 0; i < cnt; ++i) {   // bounded: <= 65536
                const uint16_t lo = static_cast<uint16_t>(
                    sorted_unique_values[groups[g].start + i] & 0xFFFFu);
                words[lo >> 6] |= (uint64_t{1} << (lo & 63u));
            }
            off += 8192u;
        }
    }
    *out_len = off;
    return true;
}

uint64_t roaring_serialize_r64_bound(uint64_t n_values) noexcept {
    return 8u + (n_values == 0 ? 0u : (4u + roaring_serialize_bound(n_values)));
}

bool roaring_serialize_r64_single_bucket(const uint32_t* sorted_unique_values,
                                         uint64_t n_values, Arena* scratch,
                                         uint8_t* dst, uint64_t dst_cap,
                                         uint64_t* out_len) noexcept {
    assert(scratch != nullptr);
    assert(out_len != nullptr);
    if (out_len == nullptr) return false;
    *out_len = 0;
    if (dst == nullptr && dst_cap > 0) return false;
    const uint64_t n_buckets = (n_values == 0) ? 0u : 1u;
    if (!wr_u64(dst, dst_cap, 0, n_buckets)) return false;
    uint64_t off = 8u;
    if (n_buckets == 1u) {
        if (!wr_u32(dst, dst_cap, off, 0u)) return false;   // high32 key = 0
        off += 4u;
        uint64_t sub_len = 0;
        if (dst_cap < off) return false;
        if (!roaring_serialize(sorted_unique_values, n_values, scratch,
                               dst + off, dst_cap - off, &sub_len))
            return false;
        off += sub_len;
    }
    *out_len = off;
    return true;
}

}  // namespace ingest
}  // namespace bolt
