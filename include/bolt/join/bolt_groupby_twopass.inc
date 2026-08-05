// bolt_groupby_twopass.inc — K-AGG-C: two-pass ingest for the typed
// multi-key GROUP BY kernel. INCLUDED ONLY BY bolt_groupby.h (after the
// GroupbyTypedState definition; shares gb_detail:: and the state POD).
//
// PASS 1 materialises gid[i] (group slot per iterated row) for a whole
// ≤kGbIngestWindow row window, with the key work specialised per WINDOW
// (one classify branch per window, never per row):
//   - int-like keys (Int64/Int32/Date32, 1-2 of them): u64 reads,
//     register hash, u128 equality.
//   - ALL-INLINE Utf8 keys (every length <= 12): the 16-byte StringView
//     packs into two u64s, padding bytes masked via a 13-entry mask
//     table, and the hash is a REGISTER-ONLY byte-exact replica of
//     gb_detail::hash_utf8_content (cross-window mixing with the
//     fallback path REQUIRES identical hashes).
//   - run-length memo: a row whose packed key equals the previous
//     row's reuses the previous gid (clustered keys are common).
// PASS 2 accumulates each aggregate with ONE dispatch per (kind, type)
// per window into a branchless gid-driven loop — no per-row AggKind/
// BoltType dispatch, no memcpy dance on Decimal128.
//
// Anything the specialised shapes can't certify (spilled keys, DISTINCT,
// >2 keys, exotic types/formats) falls back to gb_ingest_fallback — the
// original per-row loop, bit-for-bit (including the SwissTable
// collision-overwrite policy and canonicalised inline-Utf8 stored cells).
//
// Tiger Style: noexcept, <=70-line functions, >=2 asserts, zero
// allocation in ingest (gid scratch is allocated once in _begin).

namespace gb_twopass {

inline constexpr uint64_t kGbHashSeed = 0x9E3779B97F4A7C15ULL;

// Padding masks for an inline StringView packed as two u64s:
//   cell bytes 0..7  = length(4B) + prefix(4B)        -> word `a`
//   cell bytes 8..15 = inline_data(8B)                -> word `b`
// Content bytes for length L (<=12) are cell bytes 4 .. 4+L-1. The mask
// keeps length + content and zeroes the undefined padding bytes.
struct GbSvMask { uint64_t a, b; };

BOLT_FORCE_INLINE GbSvMask gb_sv_mask(uint32_t len) noexcept {
    assert(len <= 12u);
    const uint64_t lo_content =
        (len >= 4u) ? 0xFFFFFFFF00000000ULL
                    : (len == 0u ? 0ULL
                                 : (((1ULL << (8u * len)) - 1u) << 32));
    const uint64_t hi_content =
        (len <= 4u) ? 0ULL
                    : ((len - 4u >= 8u) ? ~0ULL
                                        : ((1ULL << (8u * (len - 4u))) - 1u));
    return GbSvMask{0x00000000FFFFFFFFULL | lo_content, hi_content};
}

// Canonicalise a stored inline-Utf8 cell in place (zero padding bytes)
// so raw-u128 equality from the specialised path matches content
// equality. No-op for spilled views (length > 12).
BOLT_FORCE_INLINE void gb_sv_canon_inline(GbCell16* cell) noexcept {
    assert(cell != nullptr);
    const uint32_t len = static_cast<uint32_t>(
        static_cast<uint64_t>(cell->a) & 0xFFFFFFFFULL);
    if (len > 12u) return;
    const GbSvMask m = gb_sv_mask(len);
    cell->a = static_cast<int64_t>(static_cast<uint64_t>(cell->a) & m.a);
    cell->b = static_cast<int64_t>(static_cast<uint64_t>(cell->b) & m.b);
}

// Register-only replica of gb_detail::hash_utf8_content for an INLINE
// view whose padding is already masked. Byte-exact: same seed, same
// chunk/tail split (full 8-byte chunk when len >= 8, then a tail).
BOLT_FORCE_INLINE uint64_t gb_hash_inline_sv(uint64_t a, uint64_t b,
                                             uint32_t len) noexcept {
    assert(len <= 12u);
    uint64_t h = swiss_mix_wyhash3(kGbHashSeed ^ static_cast<uint64_t>(len));
    const uint64_t first8 = (a >> 32) | (b << 32);   // content bytes 0..7
    if (len >= 8u) {
        h = swiss_mix_wyhash3(h ^ first8);
        if (len > 8u) {
            h = swiss_mix_wyhash3(h ^ (b >> 32));    // masked tail bytes 8..
        }
    } else if (len > 0u) {
        h = swiss_mix_wyhash3(h ^ first8);           // masked tail bytes 0..
    }
    return h;
}

// Window shape after classification.
enum class GbShape : uint8_t { Fallback = 0, IntKeys = 1, InlineSv = 2 };

// Classify the window's key shape: one branch per window. InlineSv also
// certifies (branch-free OR-scan) that every iterated key length <= 12.
inline GbShape gb_classify(const GroupbyTypedState* st,
                           const BoltColumn* keys, const uint32_t* sel,
                           int64_t start, int64_t count) noexcept {
    assert(st != nullptr && keys != nullptr);
    assert(count > 0);
    if (st->any_distinct || st->n_keys > 2u) return GbShape::Fallback;
    bool all_int = true, all_sv = true;
    for (uint32_t k = 0; k < st->n_keys; ++k) {
        const BoltType t = st->key_types[k];
        if (keys[k].format != ColumnFormat::Flat || keys[k].data == nullptr) {
            return GbShape::Fallback;
        }
        if (t != BoltType::Int64 && t != BoltType::Int32 &&
            t != BoltType::Date32) all_int = false;
        if (t != BoltType::Utf8) all_sv = false;
    }
    if (all_int) return GbShape::IntKeys;
    if (!all_sv) return GbShape::Fallback;
    // Certify all-inline: OR-reduce (length > 12) over iterated rows.
    uint32_t spilled = 0;
    for (uint32_t k = 0; k < st->n_keys; ++k) {
        const auto* sv = static_cast<const StringView*>(keys[k].data);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t r = (sel != nullptr)
                ? static_cast<int64_t>(sel[start + i]) : (start + i);
            spilled |= (sv[r].length > 12u) ? 1u : 0u;
        }
        if (spilled != 0u) return GbShape::Fallback;
    }
    return GbShape::InlineSv;
}

// Cold path: insert a new group with canonical key cells. Replicates the
// fallback's insert block exactly (collision policy: caller already saw
// a find-miss or equality-miss; insert OVERWRITES on hash reuse).
// Returns the new slot, or -1 on cap overflow (state->oom set).
BOLT_FORCE_INLINE int32_t gb_pass1_insert(GroupbyTypedState* st, uint64_t h,
                                          const GbCell16* canon) noexcept {
    assert(st != nullptr && canon != nullptr);
    if (st->entry_count >= st->cap) { st->oom = true; return -1; }
    const uint32_t slot = st->entry_count++;
    if (!st->table->insert(h, slot)) { st->oom = true; return -1; }
    GbCell16* krow = st->keys_flat + static_cast<size_t>(slot) * st->n_keys;
    for (uint32_t k = 0; k < st->n_keys; ++k) krow[k] = canon[k];
    for (uint32_t j = 0; j < st->n_aggs; ++j) {
        const size_t off = static_cast<size_t>(j) * st->cap + slot;
        st->accums[off] = gb_detail::agg_identity(st->specs[j].kind,
                                                  st->agg_in_types[j]);
        st->counts[off] = 0;
    }
    return static_cast<int32_t>(slot);
}

// Pack row r's keys as canonical cells. Int-like keys widen exactly like
// read_cell16 ({value, 0}); inline Utf8 keys mask padding.
template <uint32_t NK, bool SV>
BOLT_FORCE_INLINE void gb_pack_row(const GroupbyTypedState* st,
                                   const BoltColumn* keys, int64_t r,
                                   uint64_t* a, uint64_t* b) noexcept {
    for (uint32_t k = 0; k < NK; ++k) {
        if (SV) {
            const auto* sv = static_cast<const StringView*>(keys[k].data);
            uint64_t ra, rb;
            std::memcpy(&ra, &sv[r], 8);
            std::memcpy(&rb, reinterpret_cast<const char*>(&sv[r]) + 8, 8);
            const uint32_t len = static_cast<uint32_t>(ra & 0xFFFFFFFFULL);
            assert(len <= 12u);   // certified by gb_classify
            const GbSvMask m = gb_sv_mask(len);
            a[k] = ra & m.a;
            b[k] = rb & m.b;
        } else {
            const BoltType t = st->key_types[k];
            const int64_t v = (t == BoltType::Int64)
                ? static_cast<const int64_t*>(keys[k].data)[r]
                : static_cast<int64_t>(
                      static_cast<const int32_t*>(keys[k].data)[r]);
            a[k] = static_cast<uint64_t>(v);
            b[k] = 0;
        }
    }
}

// Byte-exact composite hash replica of gb_detail::hash_keys over packed
// canonical cells (int chain: h ^= mix(a); h = mix(h ^ 0); Utf8 chain:
// h = mix(h ^ content_hash)).
template <uint32_t NK, bool SV>
BOLT_FORCE_INLINE uint64_t gb_hash_packed(const uint64_t* a,
                                          const uint64_t* b) noexcept {
    uint64_t h = kGbHashSeed;
    for (uint32_t k = 0; k < NK; ++k) {
        if (SV) {
            const uint32_t len = static_cast<uint32_t>(a[k] & 0xFFFFFFFFULL);
            h = swiss_mix_wyhash3(h ^ gb_hash_inline_sv(a[k], b[k], len));
        } else {
            h ^= swiss_mix_wyhash3(a[k]);
            h  = swiss_mix_wyhash3(h ^ 0ULL);
        }
    }
    return h;
}

// Hash probe for one packed row: 64-entry direct-mapped (hash -> gid)
// cache verified against the stored canonical cells, then SwissTable
// find, then insert. Returns the gid, or -1 on oom.
template <uint32_t NK>
BOLT_FORCE_INLINE int32_t gb_probe_hash(GroupbyTypedState* st, uint64_t h,
                                        const uint64_t* a,
                                        const uint64_t* b) noexcept {
    assert(st != nullptr);
    const uint32_t ci = static_cast<uint32_t>(h & 63u);
    if (st->gid_cache_h[ci] == h &&
        st->gid_cache_g[ci] < static_cast<int32_t>(st->entry_count)) {
        const int32_t cg = st->gid_cache_g[ci];
        const GbCell16* row =
            st->keys_flat + static_cast<size_t>(cg) * st->n_keys;
        bool eq = true;
        for (uint32_t k = 0; k < NK; ++k) {
            eq = eq && (static_cast<uint64_t>(row[k].a) == a[k]) &&
                       (static_cast<uint64_t>(row[k].b) == b[k]);
        }
        if (eq) return cg;
    }
    int32_t found = st->table->find(h);
    if (found >= 0) {
        const GbCell16* row =
            st->keys_flat + static_cast<size_t>(found) * st->n_keys;
        bool eq = true;
        for (uint32_t k = 0; k < NK; ++k) {
            eq = eq && (static_cast<uint64_t>(row[k].a) == a[k]) &&
                       (static_cast<uint64_t>(row[k].b) == b[k]);
        }
        if (!eq) found = -1;
    }
    if (found < 0) {
        GbCell16 canon[NK];
        for (uint32_t k = 0; k < NK; ++k) {
            canon[k] = GbCell16{static_cast<int64_t>(a[k]),
                                static_cast<int64_t>(b[k])};
        }
        found = gb_pass1_insert(st, h, canon);
        if (found < 0) return -1;   // oom
    }
    st->gid_cache_h[ci] = h;
    st->gid_cache_g[ci] = found;
    return found;
}

// Full probe = dense gid index fast path (W-P: single int-like key with
// certified bounds — ONE array load replaces cache + Swiss probe + key
// compare), falling through to the hash probe. A dense MISS takes the
// hash path and PUBLISHES the resulting gid, so groups inserted by any
// other path (per-row fallback windows) can never be duplicated; keys
// outside the certified range (corrupt stats) use the hash path — sound.
template <uint32_t NK>
BOLT_FORCE_INLINE int32_t gb_probe_full(GroupbyTypedState* st, uint64_t h,
                                        const uint64_t* a,
                                        const uint64_t* b) noexcept {
    assert(st != nullptr);
    if (NK == 1 && st->dense_gid != nullptr) {
        const uint64_t idx = a[0] - static_cast<uint64_t>(st->dense_min);
        if (idx < st->dense_range) {
            const int32_t dg = st->dense_gid[idx];
            if (dg >= 0) return dg;
            const int32_t ng = gb_probe_hash<NK>(st, h, a, b);
            if (ng >= 0) st->dense_gid[idx] = ng;
            return ng;
        }
        // G2FEAT-362: out-of-range key (corrupt stats) — inserted hash-only,
        // never published to dense_gid, so the range-sliced dense merge would
        // drop it. Mark the state ineligible for that merge.
        st->dense_overflow = true;
    }
    return gb_probe_hash<NK>(st, h, a, b);
}

// PASS 1, ring variant (general cardinality): a software pipeline packs +
// hashes rows kGbPfDist ahead of the probe into a small stack ring and
// prefetches the SwissTable group, so each row is packed exactly once and
// the table line is resident by the time it is probed. A run-length memo
// short-circuits key repeats. Fills gid[0..count); false on oom.
inline constexpr int64_t kGbPfDist = 8;   // ring size; power of two

template <uint32_t NK, bool SV>
inline bool gb_pass1(GroupbyTypedState* st, const BoltColumn* keys,
                     const uint32_t* sel, int64_t start, int64_t count,
                     int32_t* gid) noexcept {
    static_assert(NK == 1u || NK == 2u, "specialised for 1-2 keys");
    static_assert((kGbPfDist & (kGbPfDist - 1)) == 0, "ring is pow2");
    assert(st != nullptr && gid != nullptr);
    assert(count > 0);
    uint64_t ra[kGbPfDist][NK];
    uint64_t rb[kGbPfDist][NK];
    uint64_t rh[kGbPfDist];
    const int64_t lead = (count < kGbPfDist) ? count : kGbPfDist;
    for (int64_t i = 0; i < lead; ++i) {
        const int64_t r = (sel != nullptr)
            ? static_cast<int64_t>(sel[start + i]) : (start + i);
        gb_pack_row<NK, SV>(st, keys, r, ra[i], rb[i]);
        rh[i] = gb_hash_packed<NK, SV>(ra[i], rb[i]);
        st->table->prefetch(rh[i]);
    }
    uint64_t pa[NK], pb[NK];
    int32_t prev_gid = -1;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t slot = i & (kGbPfDist - 1);
        uint64_t a[NK], b[NK];
        for (uint32_t k = 0; k < NK; ++k) { a[k] = ra[slot][k]; b[k] = rb[slot][k]; }
        const uint64_t h = rh[slot];
        const int64_t nxt = i + kGbPfDist;   // refill the vacated slot
        if (nxt < count) {
            const int64_t rn = (sel != nullptr)
                ? static_cast<int64_t>(sel[start + nxt]) : (start + nxt);
            gb_pack_row<NK, SV>(st, keys, rn, ra[slot], rb[slot]);
            rh[slot] = gb_hash_packed<NK, SV>(ra[slot], rb[slot]);
            st->table->prefetch(rh[slot]);
        }
        bool same = (prev_gid >= 0);
        for (uint32_t k = 0; k < NK; ++k) {
            same = same && (a[k] == pa[k]) && (b[k] == pb[k]);
        }
        if (same) { gid[i] = prev_gid; continue; }
        const int32_t found = gb_probe_full<NK>(st, h, a, b);
        if (found < 0) return false;   // oom
        gid[i] = found;
        for (uint32_t k = 0; k < NK; ++k) { pa[k] = a[k]; pb[k] = b[k]; }
        prev_gid = found;
    }
    return true;
}

// PASS 1, DENSE variant (W-P): a single int-like key with a certified
// dense gid index. The hot path NEVER hashes and NEVER touches the
// SwissTable — it packs the key, computes idx = key - dense_min, and
// reads dense_gid[idx] (one load), software-prefetching that slot
// kGbPfDist rows ahead (the dense array is now the random-access
// structure, so it gets the prefetch the table used to get). The hash
// is computed LAZILY, only on a cold first-touch (dense_gid[idx] < 0)
// or a key outside the certified range (corrupt stats) — those take the
// authoritative hash insert/probe and publish the gid. A run-length
// memo short-circuits key repeats. Fills gid[0..count); false on oom.
// Q17's 200k-group l_partkey aggregate is exactly this shape; the ring
// pass-1's per-row wyhash + table prefetch were its cost.
template <bool HasSel>
inline bool gb_pass1_dense(GroupbyTypedState* st, const BoltColumn* keys,
                           const uint32_t* sel, int64_t start, int64_t count,
                           int32_t* gid) noexcept {
    static_assert((kGbPfDist & (kGbPfDist - 1)) == 0, "ring is pow2");
    assert(st != nullptr && gid != nullptr && st->dense_gid != nullptr);
    assert(st->n_keys == 1u && count > 0);
    const uint64_t dmin   = static_cast<uint64_t>(st->dense_min);
    const uint64_t drange = st->dense_range;
    int32_t* BOLT_RESTRICT didx = st->dense_gid;
    uint64_t rkey[kGbPfDist];     // packed key value (a[0]); b is always 0
    uint64_t ridx[kGbPfDist];     // key - dense_min
    const int64_t lead = (count < kGbPfDist) ? count : kGbPfDist;
    for (int64_t i = 0; i < lead; ++i) {
        const int64_t r = HasSel ? static_cast<int64_t>(sel[start + i])
                                 : (start + i);
        uint64_t a[1], b[1];
        gb_pack_row<1, false>(st, keys, r, a, b);
        rkey[i] = a[0];
        const uint64_t di = a[0] - dmin;
        ridx[i] = di;
        if (di < drange) BOLT_PREFETCH_READ(&didx[di]);
    }
    uint64_t prev_key = 0;
    int32_t  prev_gid = -1;
    bool     have_prev = false;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t slot = i & (kGbPfDist - 1);
        const uint64_t a0 = rkey[slot];
        const uint64_t di = ridx[slot];
        const int64_t nxt = i + kGbPfDist;          // refill the vacated slot
        if (nxt < count) {
            const int64_t rn = HasSel ? static_cast<int64_t>(sel[start + nxt])
                                      : (start + nxt);
            uint64_t a[1], b[1];
            gb_pack_row<1, false>(st, keys, rn, a, b);
            rkey[slot] = a[0];
            const uint64_t dn = a[0] - dmin;
            ridx[slot] = dn;
            if (dn < drange) BOLT_PREFETCH_READ(&didx[dn]);
        }
        if (have_prev && a0 == prev_key) { gid[i] = prev_gid; continue; }
        int32_t found;
        if (di < drange) {
            found = didx[di];
            if (found < 0) {                        // cold first-touch: insert
                uint64_t a[1] = {a0}, b[1] = {0};
                const uint64_t h = gb_hash_packed<1, false>(a, b);
                found = gb_probe_hash<1>(st, h, a, b);
                if (found < 0) return false;        // oom
                didx[di] = found;
            }
        } else {                                    // out of certified range
            // G2FEAT-362: hash-only insert, never published to dense_gid — the
            // range-sliced dense merge would drop this group, so flag the state.
            st->dense_overflow = true;
            uint64_t a[1] = {a0}, b[1] = {0};
            const uint64_t h = gb_hash_packed<1, false>(a, b);
            found = gb_probe_hash<1>(st, h, a, b);
            if (found < 0) return false;            // oom
        }
        gid[i] = found;
        prev_key = a0; prev_gid = found; have_prev = true;
    }
    return true;
}

// PASS 1, MRU variant (low cardinality): when the table holds at most
// kGbMruWindowMax groups at window start, a 4-entry MRU of packed canonical
// keys resolves almost every row WITHOUT hashing or touching the table (the
// table is L1-resident at this size anyway, so the ring/prefetch pipeline
// buys nothing). The scan is BRANCHLESS — all 4 entries compare into a hit
// mask (one predictable branch on "any hit") — because an early-exit scan
// mispredicts on every random key change. Entries start as the sentinel
// (~0, ~0), unreachable by canonical packing in either shape: int keys
// always pack b == 0, inline Utf8 keys pack a length <= 12 in a's low 32
// bits. Exact: equality is on the full canonical cells. Fills
// gid[0..count); false on oom.
inline constexpr uint32_t kGbMruEntries   = 4u;
inline constexpr uint32_t kGbMruWindowMax = 64u;

template <uint32_t NK, bool SV>
inline bool gb_pass1_mru(GroupbyTypedState* st, const BoltColumn* keys,
                         const uint32_t* sel, int64_t start, int64_t count,
                         int32_t* gid) noexcept {
    static_assert(NK == 1u || NK == 2u, "specialised for 1-2 keys");
    assert(st != nullptr && gid != nullptr);
    assert(count > 0);
    uint64_t ma[kGbMruEntries][NK];
    uint64_t mb[kGbMruEntries][NK];
    int32_t  mg[kGbMruEntries];
    for (uint32_t e = 0; e < kGbMruEntries; ++e) {
        for (uint32_t k = 0; k < NK; ++k) { ma[e][k] = ~0ull; mb[e][k] = ~0ull; }
        mg[e] = 0;
    }
    uint32_t rr = 0;   // round-robin victim on miss
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = (sel != nullptr)
            ? static_cast<int64_t>(sel[start + i]) : (start + i);
        uint64_t a[NK], b[NK];
        gb_pack_row<NK, SV>(st, keys, r, a, b);
        uint32_t mask = 0;
        for (uint32_t e = 0; e < kGbMruEntries; ++e) {
            bool eq = true;
            for (uint32_t k = 0; k < NK; ++k) {
                eq = eq & (a[k] == ma[e][k]) & (b[k] == mb[e][k]);
            }
            mask |= static_cast<uint32_t>(eq) << e;
        }
        int32_t found;
        if (mask != 0) {
            found = mg[bolt_ctz64(mask)];
        } else {
            const uint64_t h = gb_hash_packed<NK, SV>(a, b);
            found = gb_probe_full<NK>(st, h, a, b);
            if (found < 0) return false;   // oom
            const uint32_t v = rr++ & (kGbMruEntries - 1u);
            for (uint32_t k = 0; k < NK; ++k) { ma[v][k] = a[k]; mb[v][k] = b[k]; }
            mg[v] = found;
        }
        gid[i] = found;
    }
    return true;
}

// ---- PASS 2: gid-driven branchless accumulate loops ----------------------
//
// One dispatch per (AggKind, BoltType) per WINDOW. `counts` has exactly
// one reader — finalize's Avg divisor — so the specialised loops bump it
// only when Cnt=true (kind==Avg). The fallback still bumps counts for
// every kind; the divergence is benign because nothing reads counts for
// non-Avg aggregates. gb_p2_generic keeps the fallback behaviour (it can
// carry Avg over Float64).

template <bool Sparse, bool HasValid>
BOLT_FORCE_INLINE bool gb_p2_valid(const BoltColumn& pc, int64_t r) noexcept {
    if (!HasValid) return true;
    const int64_t bit = pc.validity_offset + r;
    return ((pc.validity[bit >> 3] >> (bit & 7)) & 1) != 0;
}

template <bool Sparse>
BOLT_FORCE_INLINE int64_t gb_p2_row(const uint32_t* sel, int64_t start,
                                    int64_t i) noexcept {
    return Sparse ? static_cast<int64_t>(sel[start + i]) : (start + i);
}

// In-place 128-bit accumulate on scalar lanes. MSVC passes/returns 16-byte
// structs through stack temps even under __forceinline, which turns the
// by-value d128_add into a store-forwarding chain (~3ns/row measured);
// writing the lanes directly keeps everything in registers.
BOLT_FORCE_INLINE void gb_d128_acc(
        kernels::decimal::Decimal128* BOLT_RESTRICT dst,
        const kernels::decimal::Decimal128& src) noexcept {
    const uint64_t alo = static_cast<uint64_t>(dst->lo);
    const uint64_t lo  = alo + static_cast<uint64_t>(src.lo);
    dst->lo = static_cast<int64_t>(lo);
    dst->hi = dst->hi + src.hi + static_cast<int64_t>((lo < alo) ? 1 : 0);
}

template <bool Sparse, bool HasValid, bool Cnt>
inline void gb_p2_sum_d128(GbCell16* BOLT_RESTRICT acc,
                           int64_t* BOLT_RESTRICT cnt,
                           const BoltColumn& pc, const uint32_t* sel,
                           int64_t start, int64_t count,
                           const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && cnt != nullptr && gid != nullptr);
    static_assert(sizeof(GbCell16) == sizeof(kernels::decimal::Decimal128),
                  "GbCell16 must alias Decimal128");
    const auto* p = static_cast<const kernels::decimal::Decimal128*>(pc.data);
    auto* a = reinterpret_cast<kernels::decimal::Decimal128*>(acc);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        if (valid) gb_d128_acc(&a[g], p[r]);
        if (Cnt) cnt[g] += valid ? 1 : 0;
    }
}

// W-DEC: Decimal64 payload — int64 mantissas accumulated into the FULL
// d128 cell (lo/hi lanes in place), so sums grow past int64 with no
// plan-time proof needed. Columnar like sum_d128; reads 8B/row not 16B.
template <bool Sparse, bool HasValid, bool Cnt>
inline void gb_p2_sum_dec64(GbCell16* BOLT_RESTRICT acc,
                            int64_t* BOLT_RESTRICT cnt,
                            const BoltColumn& pc, const uint32_t* sel,
                            int64_t start, int64_t count,
                            const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && cnt != nullptr && gid != nullptr);
    const auto* p = static_cast<const int64_t*>(pc.data);
    auto* a = reinterpret_cast<kernels::decimal::Decimal128*>(acc);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        if (valid) {
            const int64_t m = p[r];
            const uint64_t alo = static_cast<uint64_t>(a[g].lo);
            const uint64_t lo  = alo + static_cast<uint64_t>(m);
            a[g].lo = static_cast<int64_t>(lo);
            a[g].hi = a[g].hi + (m >> 63) +
                      static_cast<int64_t>((lo < alo) ? 1 : 0);
        }
        if (Cnt) cnt[g] += valid ? 1 : 0;
    }
}

template <bool Sparse, bool HasValid, typename SrcT, bool Cnt>
inline void gb_p2_sum_int(GbCell16* BOLT_RESTRICT acc,
                          int64_t* BOLT_RESTRICT cnt,
                          const BoltColumn& pc, const uint32_t* sel,
                          int64_t start, int64_t count,
                          const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && cnt != nullptr && gid != nullptr);
    const auto* p = static_cast<const SrcT*>(pc.data);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        // G2FEAT-131: 128-bit accumulation across the whole cell. `acc[g].a +=`
        // wrapped, and an integer aggregate never used the high limb for
        // anything, so widening is free in memory and one add-with-carry in
        // time. AVG divides this exactly at finalize; SUM range-checks it.
        const int64_t x  = valid ? static_cast<int64_t>(p[r]) : 0;
        const uint64_t lo = static_cast<uint64_t>(acc[g].a);
        const uint64_t sm = lo + static_cast<uint64_t>(x);
        acc[g].b += ((sm < lo) ? 1 : 0) + (x >> 63);
        acc[g].a  = static_cast<int64_t>(sm);
        if (Cnt) cnt[g] += valid ? 1 : 0;
    }
}

template <bool Sparse, bool HasValid, typename SrcT, bool IsMin>
inline void gb_p2_minmax_int(GbCell16* BOLT_RESTRICT acc,
                             int64_t* BOLT_RESTRICT cnt,
                             const BoltColumn& pc, const uint32_t* sel,
                             int64_t start, int64_t count,
                             const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && cnt != nullptr && gid != nullptr);
    (void)cnt;   // Min/Max never feed an Avg divisor
    const auto* p = static_cast<const SrcT*>(pc.data);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        if (valid) {
            const int64_t v = static_cast<int64_t>(p[r]);
            acc[g].a = IsMin ? branchless::bmin(acc[g].a, v)
                             : branchless::bmax(acc[g].a, v);
        }
    }
}

template <bool Sparse>
inline void gb_p2_count_star(GbCell16* BOLT_RESTRICT acc,
                             int64_t count,
                             const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && gid != nullptr);
    for (int64_t i = 0; i < count; ++i) {
        const int32_t g = gid[i];
        acc[g].a += 1;
    }
}

template <bool Sparse, bool HasValid>
inline void gb_p2_count(GbCell16* BOLT_RESTRICT acc,
                        int64_t* BOLT_RESTRICT cnt,
                        const BoltColumn& pc, const uint32_t* sel,
                        int64_t start, int64_t count,
                        const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(acc != nullptr && cnt != nullptr && gid != nullptr);
    (void)cnt;   // Count's result lives in acc.a; the divisor is unused
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        acc[g].a += valid ? 1 : 0;
    }
}

// ---- Banked variants: low-cardinality windows ----------------------------
//
// With few groups, consecutive rows frequently hit the same accumulator and
// the read-modify-write chain serializes on store-to-load forwarding. Rows
// cycle kGbBankCount independent bank slices (bank = i & 3) so adjacent rows
// never touch the same cell; banks merge into accums once per window.
// Eligible when entry_count <= kGbBankGroups and the window is big enough
// to amortise zeroing the bank slices (gb_pass2_one gates on it).

static_assert((kGbBankCount & (kGbBankCount - 1u)) == 0u, "banks are pow2");

template <bool Cnt>
BOLT_FORCE_INLINE void gb_p2_bank_zero(GroupbyTypedState* st,
                                       uint32_t ec) noexcept {
    assert(st->bank_acc != nullptr);
    assert(ec <= kGbBankGroups);
    for (uint32_t b = 0; b < kGbBankCount; ++b) {
        std::memset(st->bank_acc + static_cast<size_t>(b) * kGbBankGroups, 0,
                    sizeof(GbCell16) * ec);
        if (Cnt) {
            std::memset(st->bank_cnt + static_cast<size_t>(b) * kGbBankGroups,
                        0, sizeof(int64_t) * ec);
        }
    }
}

template <bool Sparse, bool HasValid, bool Cnt>
inline void gb_p2_sum_d128_bank(GroupbyTypedState* st,
                                GbCell16* BOLT_RESTRICT acc,
                                int64_t* BOLT_RESTRICT cnt,
                                const BoltColumn& pc, const uint32_t* sel,
                                int64_t start, int64_t count,
                                const int32_t* BOLT_RESTRICT gid,
                                uint32_t ec) noexcept {
    namespace dec = kernels::decimal;
    assert(acc != nullptr && gid != nullptr);
    gb_p2_bank_zero<Cnt>(st, ec);
    auto* bk = reinterpret_cast<dec::Decimal128*>(st->bank_acc);
    int64_t* bc = st->bank_cnt;
    const auto* p = static_cast<const dec::Decimal128*>(pc.data);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const size_t o = static_cast<size_t>(i & (kGbBankCount - 1)) *
                         kGbBankGroups + static_cast<uint32_t>(gid[i]);
        if (valid) gb_d128_acc(&bk[o], p[r]);
        if (Cnt) bc[o] += valid ? 1 : 0;
    }
    auto* a = reinterpret_cast<dec::Decimal128*>(acc);
    for (uint32_t g = 0; g < ec; ++g) {
        dec::Decimal128 s = bk[g];
        for (uint32_t b = 1; b < kGbBankCount; ++b)
            gb_d128_acc(&s, bk[static_cast<size_t>(b) * kGbBankGroups + g]);
        gb_d128_acc(&a[g], s);
        if (Cnt) {
            int64_t c = bc[g];
            for (uint32_t b = 1; b < kGbBankCount; ++b)
                c += bc[static_cast<size_t>(b) * kGbBankGroups + g];
            cnt[g] += c;
        }
    }
}

template <bool Sparse, bool HasValid, typename SrcT, bool Cnt>
inline void gb_p2_sum_int_bank(GroupbyTypedState* st,
                               GbCell16* BOLT_RESTRICT acc,
                               int64_t* BOLT_RESTRICT cnt,
                               const BoltColumn& pc, const uint32_t* sel,
                               int64_t start, int64_t count,
                               const int32_t* BOLT_RESTRICT gid,
                               uint32_t ec) noexcept {
    assert(acc != nullptr && gid != nullptr);
    gb_p2_bank_zero<Cnt>(st, ec);
    GbCell16* bk = st->bank_acc;
    int64_t*  bc = st->bank_cnt;
    const auto* p = static_cast<const SrcT*>(pc.data);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const size_t o = static_cast<size_t>(i & (kGbBankCount - 1)) *
                         kGbBankGroups + static_cast<uint32_t>(gid[i]);
        bk[o].a += valid ? static_cast<int64_t>(p[r]) : 0;
        if (Cnt) bc[o] += valid ? 1 : 0;
    }
    for (uint32_t g = 0; g < ec; ++g) {
        int64_t s = bk[g].a;
        for (uint32_t b = 1; b < kGbBankCount; ++b)
            s += bk[static_cast<size_t>(b) * kGbBankGroups + g].a;
        acc[g].a += s;
        if (Cnt) {
            int64_t c = bc[g];
            for (uint32_t b = 1; b < kGbBankCount; ++b)
                c += bc[static_cast<size_t>(b) * kGbBankGroups + g];
            cnt[g] += c;
        }
    }
}

// W-DEC inc 4: lane-PROVEN Decimal64 sum, banked. CALLER CONTRACT
// (AggSpec::lane == 1): row_bound × max|mantissa| ≤ INT64_MAX, so every
// partial sum — including each bank's window subtotal, mixed signs and
// all (|partial| ≤ Σ|values| ≤ row_bound × max|value|) — fits an int64.
// The win over gb_p2_sum_dec64 is dropping the per-row carry chain:
// banks accumulate pure int64; the once-per-window merge widens each
// per-group subtotal to d128 via gb_d128_acc, so the PERSISTENT cell
// stays a true Decimal128 and composes freely with lane-0 / fallback
// windows (no cross-path representation hazard, no finalize changes).
template <bool Sparse, bool HasValid, bool Cnt>
inline void gb_p2_sum_dec64_lane_bank(GroupbyTypedState* st,
                                      GbCell16* BOLT_RESTRICT acc,
                                      int64_t* BOLT_RESTRICT cnt,
                                      const BoltColumn& pc,
                                      const uint32_t* sel,
                                      int64_t start, int64_t count,
                                      const int32_t* BOLT_RESTRICT gid,
                                      uint32_t ec) noexcept {
    assert(acc != nullptr && gid != nullptr);
    assert(st->bank_acc != nullptr);
    gb_p2_bank_zero<Cnt>(st, ec);
    GbCell16* bk = st->bank_acc;
    int64_t*  bc = st->bank_cnt;
    const auto* p = static_cast<const int64_t*>(pc.data);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const size_t o = static_cast<size_t>(i & (kGbBankCount - 1)) *
                         kGbBankGroups + static_cast<uint32_t>(gid[i]);
        bk[o].a += valid ? p[r] : 0;
        if (Cnt) bc[o] += valid ? 1 : 0;
    }
    auto* a = reinterpret_cast<kernels::decimal::Decimal128*>(acc);
    for (uint32_t g = 0; g < ec; ++g) {
        int64_t s = bk[g].a;
        for (uint32_t b = 1; b < kGbBankCount; ++b)
            s += bk[static_cast<size_t>(b) * kGbBankGroups + g].a;
        gb_d128_acc(&a[g], kernels::decimal::d128_from_i64(s));
        if (Cnt) {
            int64_t c = bc[g];
            for (uint32_t b = 1; b < kGbBankCount; ++b)
                c += bc[static_cast<size_t>(b) * kGbBankGroups + g];
            cnt[g] += c;
        }
    }
}

template <bool Sparse>
inline void gb_p2_count_star_bank(GroupbyTypedState* st,
                                  GbCell16* BOLT_RESTRICT acc,
                                  int64_t count,
                                  const int32_t* BOLT_RESTRICT gid,
                                  uint32_t ec) noexcept {
    assert(acc != nullptr && gid != nullptr);
    gb_p2_bank_zero<true>(st, ec);
    int64_t* bc = st->bank_cnt;
    for (int64_t i = 0; i < count; ++i) {
        const size_t o = static_cast<size_t>(i & (kGbBankCount - 1)) *
                         kGbBankGroups + static_cast<uint32_t>(gid[i]);
        bc[o] += 1;
    }
    for (uint32_t g = 0; g < ec; ++g) {
        int64_t c = bc[g];
        for (uint32_t b = 1; b < kGbBankCount; ++b)
            c += bc[static_cast<size_t>(b) * kGbBankGroups + g];
        acc[g].a += c;
    }
}

template <bool Sparse, bool HasValid>
inline void gb_p2_count_bank(GroupbyTypedState* st,
                             GbCell16* BOLT_RESTRICT acc,
                             int64_t* BOLT_RESTRICT cnt,
                             const BoltColumn& pc, const uint32_t* sel,
                             int64_t start, int64_t count,
                             const int32_t* BOLT_RESTRICT gid,
                             uint32_t ec) noexcept {
    assert(acc != nullptr && gid != nullptr);
    (void)cnt;
    gb_p2_bank_zero<true>(st, ec);
    int64_t* bc = st->bank_cnt;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const size_t o = static_cast<size_t>(i & (kGbBankCount - 1)) *
                         kGbBankGroups + static_cast<uint32_t>(gid[i]);
        bc[o] += valid ? 1 : 0;
    }
    for (uint32_t g = 0; g < ec; ++g) {
        int64_t c = bc[g];
        for (uint32_t b = 1; b < kGbBankCount; ++b)
            c += bc[static_cast<size_t>(b) * kGbBankGroups + g];
        acc[g].a += c;
    }
}

// Generic gid-driven loop for unspecialised (kind, type) combos —
// Float64 sums (existing bit-pattern semantics preserved, NOT fixed
// silently), Utf8 min/max, Decimal min/max. Still removes the per-row
// hash/probe; dispatch stays per row via gb_detail::apply (parity).
template <bool Sparse, bool HasValid>
inline void gb_p2_generic(GroupbyTypedState* st, uint32_t j,
                          const BoltColumn& pc, const uint32_t* sel,
                          int64_t start, int64_t count,
                          const int32_t* BOLT_RESTRICT gid) noexcept {
    assert(st != nullptr && gid != nullptr);
    GbCell16* acc = st->accums + static_cast<size_t>(j) * st->cap;
    int64_t*  cnt = st->counts + static_cast<size_t>(j) * st->cap;
    const AggKind kind = st->specs[j].kind;
    const BoltType t   = st->agg_in_types[j];
    // G2FEAT-283: MIN/MAX(Utf8) cannot go through the generic gb_detail::apply
    // — it hard-codes nullptr spilled-string bases, which segfaults the
    // moment a real (>12-byte) string is compared. Resolve + deep-copy-on-win
    // via the dedicated helper instead (same fix as gb_ingest_fallback).
    const bool is_utf8_minmax = (t == BoltType::Utf8) &&
        (kind == AggKind::Min || kind == AggKind::Max);
    const char* cand_base = is_utf8_minmax
        ? static_cast<const char*>(pc.str_overflow_base) : nullptr;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = gb_p2_row<Sparse>(sel, start, i);
        const bool valid = gb_p2_valid<Sparse, HasValid>(pc, r);
        const int32_t g = gid[i];
        const GbCell16 v = gb_detail::read_cell16(pc, r);
        if (is_utf8_minmax) {
            if (!gb_agg_utf8_minmax(st, j, kind, &acc[g], v, valid, cand_base)) {
                return;   // st->oom already set; groupby_agg_multi_key_typed_
                          // ingest checks state->oom right after this call
            }
        } else {
            gb_detail::apply(kind, t, &acc[g], v, valid);
        }
        cnt[g] += valid ? 1 : 0;
    }
}

// Sum/Avg dispatch (plain vs banked, per input type). Returns false when
// the (kind, type) combo is not a specialised sum shape.
template <bool Sparse>
inline bool gb_pass2_sum_like(GroupbyTypedState* st, GbCell16* acc,
                              int64_t* cnt, const BoltColumn& pc,
                              const uint32_t* sel, int64_t start,
                              int64_t count, const int32_t* gid,
                              uint32_t ec, bool banked, BoltType t,
                              bool is_avg, uint8_t lane) noexcept {
    assert(st != nullptr && acc != nullptr);
    const bool hv = (pc.validity != nullptr);
#define BOLT_GB_P2(FN, ...)                                                  \
    do {                                                                     \
        if (hv) FN<Sparse, true, ##__VA_ARGS__>(acc, cnt, pc, sel, start,    \
                                                count, gid);                 \
        else    FN<Sparse, false, ##__VA_ARGS__>(acc, cnt, pc, sel, start,   \
                                                 count, gid);                \
    } while (0)
#define BOLT_GB_P2B(FN, ...)                                                 \
    do {                                                                     \
        if (hv) FN<Sparse, true, ##__VA_ARGS__>(st, acc, cnt, pc, sel,       \
                                                start, count, gid, ec);      \
        else    FN<Sparse, false, ##__VA_ARGS__>(st, acc, cnt, pc, sel,      \
                                                 start, count, gid, ec);     \
    } while (0)
    if (t == BoltType::Decimal128) {
        if (banked) {
            if (is_avg) BOLT_GB_P2B(gb_p2_sum_d128_bank, true);
            else        BOLT_GB_P2B(gb_p2_sum_d128_bank, false);
        } else {
            if (is_avg) BOLT_GB_P2(gb_p2_sum_d128, true);
            else        BOLT_GB_P2(gb_p2_sum_d128, false);
        }
        return true;
    }
    if (t == BoltType::Decimal64) {
        // W-DEC inc 4: lane==1 (caller-proved int64 fit) + banked window
        // → pure-int64 bank accumulation with a d128-widening merge (the
        // cell stays a true d128; see gb_p2_sum_dec64_lane_bank). All
        // other windows — unproven, or too few rows/groups to bank —
        // keep the overflow-safe columnar d128 loop, which is already
        // fast for the non-banked shape.
        if (lane == 1u && banked) {
            if (is_avg) BOLT_GB_P2B(gb_p2_sum_dec64_lane_bank, true);
            else        BOLT_GB_P2B(gb_p2_sum_dec64_lane_bank, false);
        } else {
            if (is_avg) BOLT_GB_P2(gb_p2_sum_dec64, true);
            else        BOLT_GB_P2(gb_p2_sum_dec64, false);
        }
        return true;
    }
    // G2FEAT-131: the BANKED integer sum loop is deliberately NOT taken any
    // more. Banking accumulates each of kGbBankCount window slices in a bare
    // int64 and only widens at the once-per-window merge, which is exactly the
    // shape that made `gb_p2_sum_dec64_lane_bank` require the caller's
    // lane==1 PROOF that no partial can overflow. No such proof exists for a
    // plain integer column, so a banked slice could wrap before the merge ever
    // sees it. The plain loop is exact for any input; regaining banking needs
    // a widened bank (2x bank memory) or the same lane certificate, which is
    // the tracked follow-up. TPC-H's sums are Decimal (untouched arms above),
    // and chukonu's own int64 aggregate operator — where the overwhelming
    // majority of integer sums execute — never used this kernel at all.
    if (t == BoltType::Int64) {
        if (is_avg) BOLT_GB_P2(gb_p2_sum_int, int64_t, true);
        else        BOLT_GB_P2(gb_p2_sum_int, int64_t, false);
        return true;
    }
    if (t == BoltType::Int32 || t == BoltType::Date32) {
        if (is_avg) BOLT_GB_P2(gb_p2_sum_int, int32_t, true);
        else        BOLT_GB_P2(gb_p2_sum_int, int32_t, false);
        return true;
    }
#undef BOLT_GB_P2B
#undef BOLT_GB_P2
    return false;
}

// One dispatch per aggregate per window. Banked loops take over when the
// group count is low enough that RMW chains dominate AND the window is big
// enough to amortise zeroing the bank slices (~96 bytes per group).
template <bool Sparse>
inline void gb_pass2_one(GroupbyTypedState* st, uint32_t j,
                         const BoltColumn* payload, const uint32_t* sel,
                         int64_t start, int64_t count,
                         const int32_t* gid) noexcept {
    assert(st != nullptr && gid != nullptr);
    GbCell16* acc = st->accums + static_cast<size_t>(j) * st->cap;
    int64_t*  cnt = st->counts + static_cast<size_t>(j) * st->cap;
    const AggKind kind = st->specs[j].kind;
    const uint32_t ec = st->entry_count;
    const bool banked = (st->bank_acc != nullptr) && (ec <= kGbBankGroups) &&
                        (count >= static_cast<int64_t>(ec) * 64);
    if (kind == AggKind::CountStar) {
        if (banked) gb_p2_count_star_bank<Sparse>(st, acc, count, gid, ec);
        else        gb_p2_count_star<Sparse>(acc, count, gid);
        return;
    }
    const BoltColumn& pc = payload[st->specs[j].in_col];
    const BoltType t = st->agg_in_types[j];
    const bool hv = (pc.validity != nullptr);
    const bool sum_like = (kind == AggKind::Sum || kind == AggKind::Avg);
    if (sum_like &&
        gb_pass2_sum_like<Sparse>(st, acc, cnt, pc, sel, start, count, gid,
                                  ec, banked, t, kind == AggKind::Avg,
                                  st->specs[j].lane)) {
        return;
    }
#define BOLT_GB_P2(FN, ...)                                                  \
    do {                                                                     \
        if (hv) FN<Sparse, true, ##__VA_ARGS__>(acc, cnt, pc, sel, start,    \
                                                count, gid);                 \
        else    FN<Sparse, false, ##__VA_ARGS__>(acc, cnt, pc, sel, start,   \
                                                 count, gid);                \
    } while (0)
    if (kind == AggKind::Count) {
        if (banked) {
            if (hv) gb_p2_count_bank<Sparse, true>(st, acc, cnt, pc, sel,
                                                   start, count, gid, ec);
            else    gb_p2_count_bank<Sparse, false>(st, acc, cnt, pc, sel,
                                                    start, count, gid, ec);
        } else {
            BOLT_GB_P2(gb_p2_count);
        }
    } else if ((kind == AggKind::Min || kind == AggKind::Max) &&
               t == BoltType::Int64) {
        if (kind == AggKind::Min) BOLT_GB_P2(gb_p2_minmax_int, int64_t, true);
        else                      BOLT_GB_P2(gb_p2_minmax_int, int64_t, false);
    } else if ((kind == AggKind::Min || kind == AggKind::Max) &&
               (t == BoltType::Int32 || t == BoltType::Date32)) {
        if (kind == AggKind::Min) BOLT_GB_P2(gb_p2_minmax_int, int32_t, true);
        else                      BOLT_GB_P2(gb_p2_minmax_int, int32_t, false);
    } else {
        if (hv) gb_p2_generic<Sparse, true>(st, j, pc, sel, start, count, gid);
        else    gb_p2_generic<Sparse, false>(st, j, pc, sel, start, count, gid);
    }
#undef BOLT_GB_P2
}

// Try the two-pass path on one window. Returns false when the shape
// requires the fallback. On oom the state flag is set and true is
// returned (the dispatcher checks state->oom).
inline bool gb_try_twopass(GroupbyTypedState* st, const BoltColumn* keys,
                           const BoltColumn* payload, const uint32_t* sel,
                           int64_t start, int64_t count) noexcept {
    assert(st != nullptr);
    assert(count > 0 && count <= static_cast<int64_t>(kGbIngestWindow));
    if (st->gid_scratch == nullptr) return false;
    const GbShape shape = gb_classify(st, keys, sel, start, count);
    if (shape == GbShape::Fallback) return false;
    int32_t* gid = st->gid_scratch;
    const bool mru = (st->entry_count <= kGbMruWindowMax);
    bool ok;
    // W-P dense fast path: a single int-like key with a certified gid
    // index resolves with one array load and never hashes — strictly
    // cheaper than both the ring and MRU variants, so it takes priority
    // for IntKeys whenever enabled (gb_classify already proved the int
    // shape; dense_gid != nullptr proves single-key + certified bounds).
    if (shape == GbShape::IntKeys && st->dense_gid != nullptr &&
        st->n_keys == 1u) {
        ok = (sel != nullptr)
            ? gb_pass1_dense<true>(st, keys, sel, start, count, gid)
            : gb_pass1_dense<false>(st, keys, sel, start, count, gid);
    } else if (shape == GbShape::IntKeys) {
        if (mru) {
            ok = (st->n_keys == 1u)
                ? gb_pass1_mru<1, false>(st, keys, sel, start, count, gid)
                : gb_pass1_mru<2, false>(st, keys, sel, start, count, gid);
        } else {
            ok = (st->n_keys == 1u)
                ? gb_pass1<1, false>(st, keys, sel, start, count, gid)
                : gb_pass1<2, false>(st, keys, sel, start, count, gid);
        }
    } else {
        if (mru) {
            ok = (st->n_keys == 1u)
                ? gb_pass1_mru<1, true>(st, keys, sel, start, count, gid)
                : gb_pass1_mru<2, true>(st, keys, sel, start, count, gid);
        } else {
            ok = (st->n_keys == 1u)
                ? gb_pass1<1, true>(st, keys, sel, start, count, gid)
                : gb_pass1<2, true>(st, keys, sel, start, count, gid);
        }
    }
    if (!ok) { assert(st->oom); return true; }   // oom mid-window: stop
    for (uint32_t j = 0; j < st->n_aggs; ++j) {
        if (sel != nullptr) {
            gb_pass2_one<true>(st, j, payload, sel, start, count, gid);
        } else {
            gb_pass2_one<false>(st, j, payload, sel, start, count, gid);
        }
    }
    return true;
}

}  // namespace gb_twopass
