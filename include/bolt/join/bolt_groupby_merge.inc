// bolt_groupby_merge.inc — W4 partition-parallel partial-state MERGE
// kernels, split out of bolt_groupby.h for file size (the header passed
// 1900 lines). #included by bolt_groupby.h INSIDE `namespace bolt` at the
// position this code originally occupied; NOT a standalone header (no
// include guard by design — single inclusion point, relies on the parent
// header's includes + GroupbyTypedState/gb_detail/gb_grow definitions).
//
//   groupby_agg_multi_key_typed_merge(dst, src)
//
// Folds every group of `src` (a worker-private partial built by _begin/
// _ingest) into `dst`. Each src group's STORED key row is re-hashed with
// the exact content hash `_ingest` uses (spilled Utf8 keys resolve via
// src->key_str_buf), found-or-inserted into dst (spilled bytes deep-copied
// into dst's buffer via gb_key_str_append), and the accumulator cells are
// combined per agg kind:
//   Sum/Avg   : cell add — d128_add for Decimal128/Decimal64 (exact partial-
//               sum merge; lane-1 cells are still true d128 by the lane
//               representation invariant), int64 add otherwise.
//   Min/Max   : gb_detail::apply typed compare (d128 / byte-lex / bmin-bmax).
//   Count/*   : a += a.
//   counts[]  : += (AVG denominators).
// DISTINCT states are NOT mergeable (DistinctCell sets are not unioned in
// v1) — asserted out; callers must gate eligibility upstream.
// Returns false (and sets dst->oom) on dst capacity / arena overflow.
//
// gb_merge_hash_stored is defined in bolt_groupby.h ABOVE the ingest
// section (gb_grow's rehash needs it; the merge here uses the same
// definition).

// Stored-vs-stored composite-key equality. Utf8 compares by CONTENT bytes
// (each side resolves spilled views via its OWN key_str_buf); other types
// compare the 16-byte cells (read_cell16-canonical on both sides).
inline bool gb_merge_keys_equal(const GroupbyTypedState* dst, uint32_t dgid,
                                const GroupbyTypedState* src,
                                uint32_t sgid) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst->n_keys == src->n_keys);
    const uint32_t n_keys = dst->n_keys;
    const GbCell16* drow = dst->keys_flat + static_cast<size_t>(dgid) * n_keys;
    const GbCell16* srow = src->keys_flat + static_cast<size_t>(sgid) * n_keys;
    for (uint32_t k = 0; k < n_keys; ++k) {
        if (dst->key_types[k] == BoltType::Utf8) {
            StringView dsv, ssv;
            std::memcpy(&dsv, &drow[k], 16);
            std::memcpy(&ssv, &srow[k], 16);
            if (dsv.length != ssv.length) return false;
            if (dsv.length == 0) continue;
            const char* db = kernels::utf8::sv_bytes(dsv, dst->key_str_buf[k]);
            const char* sb = kernels::utf8::sv_bytes(ssv, src->key_str_buf[k]);
            if (std::memcmp(db, sb, dsv.length) != 0) return false;
            continue;
        }
        if (drow[k].a != srow[k].a || drow[k].b != srow[k].b) return false;
    }
    return true;
}

// Insert src group `sgid` as a NEW dst group under hash `h`: copy key cells
// (deep-copying spilled Utf8 bytes into dst's buffer), seed identities.
inline bool gb_merge_insert_group(GroupbyTypedState* dst,
                                  const GroupbyTypedState* src,
                                  uint32_t sgid, uint64_t h,
                                  uint32_t* out_slot) noexcept {
    assert(dst != nullptr && src != nullptr && out_slot != nullptr);
    // W-P growth: merge loops read dst state fresh per insert (no cached
    // pointers), so growing here is safe. Legacy oom-at-cap when disabled.
    if (dst->entry_count >= dst->cap && !gb_grow(dst)) {
        dst->oom = true;
        return false;
    }
    const uint32_t slot = dst->entry_count++;
    if (!dst->table->insert(h, slot)) { dst->oom = true; return false; }
    const uint32_t n_keys = dst->n_keys;
    GbCell16* drow = dst->keys_flat + static_cast<size_t>(slot) * n_keys;
    const GbCell16* srow = src->keys_flat + static_cast<size_t>(sgid) * n_keys;
    for (uint32_t k = 0; k < n_keys; ++k) {
        drow[k] = srow[k];
        if (dst->key_types[k] == BoltType::Utf8) {
            StringView* sv = reinterpret_cast<StringView*>(&drow[k]);
            if (sv->length > 12u) {
                const char* sb = kernels::utf8::sv_bytes(*sv,
                                                         src->key_str_buf[k]);
                uint32_t off = 0;
                if (!gb_key_str_append(dst, k, sb, sv->length, &off)) {
                    dst->oom = true;
                    return false;
                }
                sv->ref.buf_idx = 0;
                sv->ref.offset  = off;
            }
            // Inline (<=12) cells are stored canonicalised — copy as-is.
        }
    }
    for (uint32_t j = 0; j < dst->n_aggs; ++j) {
        const size_t off = static_cast<size_t>(j) * dst->cap + slot;
        dst->accums[off] = gb_detail::agg_identity(dst->specs[j].kind,
                                                   dst->agg_in_types[j]);
        dst->counts[off] = 0;
    }
    *out_slot = slot;
    return true;
}

// Combine src group `sgid` into dst slot `dslot`, per agg kind. Returns false
// (dst->oom already set by the callee) only when a Utf8 MIN/MAX deep-copy hits
// arena exhaustion; every other combine kind is allocation-free and always
// succeeds.
inline bool gb_merge_combine(GroupbyTypedState* dst, uint32_t dslot,
                             const GroupbyTypedState* src,
                             uint32_t sgid) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst->n_aggs == src->n_aggs);
    for (uint32_t j = 0; j < dst->n_aggs; ++j) {
        const size_t doff = static_cast<size_t>(j) * dst->cap + dslot;
        const size_t soff = static_cast<size_t>(j) * src->cap + sgid;
        const AggKind k = dst->specs[j].kind;
        if (k == AggKind::Count || k == AggKind::CountStar) {
            // The src cell holds a partial COUNT — add the count itself
            // (apply() would add 1 per call, which is per-ROW semantics).
            dst->accums[doff].a += src->accums[soff].a;
        } else if (dst->agg_in_types[j] == BoltType::Utf8 &&
                   (k == AggKind::Min || k == AggKind::Max)) {
            // G2FEAT-283: src's cell (if spilled) is owned by SRC's OWN
            // agg_str_buf — resolve it there (never a transient producing
            // morsel's buffer, long gone by merge time). If it wins here, it
            // gets deep-copied into DST's own buffer by gb_agg_utf8_minmax.
            StringView ssv;
            std::memcpy(&ssv, &src->accums[soff], sizeof(ssv));
            const char* cand_base =
                (ssv.length > 12u) ? src->agg_str_buf[j] : nullptr;
            if (!gb_agg_utf8_minmax(dst, j, k, &dst->accums[doff],
                                    src->accums[soff], /*valid=*/true,
                                    cand_base)) {
                return false;   // dst->oom already set
            }
        } else {
            // Sum/Avg: apply's add arm over the src PARTIAL cell == exact
            // partial-sum merge (d128_add is associative two's-complement
            // i128). Min/Max: typed compare against the src partial — a
            // src identity cell (zero valid rows) can never win.
            gb_detail::apply(k, dst->agg_in_types[j], &dst->accums[doff],
                             src->accums[soff], /*valid=*/true);
        }
        dst->counts[doff] += src->counts[soff];
    }
    return true;
}

// HASH-CLASS-RESTRICTED merge (chukonu W-J5 parallel partial merge): fold
// only src groups whose stored-key hash falls in class `class_idx` of
// `n_classes`. N workers each folding a DISJOINT class of every partial
// into a PRIVATE dst produce disjoint dst tables whose union equals the
// full merge — merge cost parallelizes as sum_groups / N instead of the
// serial sum (measured 1010 ms for 16 x 169k partials on TPC-H Q17).
// The class projection MUST be the stored-key hash (the same value the
// dst table is keyed by) so every worker classifies a group identically.
// (class_idx 0 of 1 == the full merge; the public merge delegates here.)
inline bool groupby_agg_multi_key_typed_merge_class(
        GroupbyTypedState* dst, const GroupbyTypedState* src,
        uint32_t class_idx, uint32_t n_classes) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(dst->n_keys == src->n_keys);
    assert(dst->n_aggs == src->n_aggs);
    assert(n_classes >= 1 && class_idx < n_classes);
    assert(!dst->any_distinct && !src->any_distinct &&
           "DISTINCT partials are not mergeable (v1)");
    if (dst->oom || src->oom) return false;
    if (dst->any_distinct || src->any_distinct) return false;
#ifndef NDEBUG
    for (uint32_t k = 0; k < dst->n_keys; ++k) {
        assert(dst->key_types[k] == src->key_types[k]);
        assert(dst->key_scales[k] == src->key_scales[k]);
    }
    for (uint32_t j = 0; j < dst->n_aggs; ++j) {
        assert(dst->specs[j].kind == src->specs[j].kind);
        assert(dst->specs[j].in_col == src->specs[j].in_col);
        assert(dst->agg_in_types[j] == src->agg_in_types[j]);
        assert(dst->agg_in_scales[j] == src->agg_in_scales[j]);
    }
#endif
    const uint32_t n_src = src->entry_count;
    for (uint32_t g = 0; g < n_src; ++g) {
        const uint64_t h = gb_merge_hash_stored(src, g);
        if (n_classes > 1u && (h % n_classes) != class_idx) continue;
        int32_t found = dst->table->find(h);
        if (found >= 0 &&
            !gb_merge_keys_equal(dst, static_cast<uint32_t>(found), src, g)) {
            found = -1;   // 64-bit hash collision: mirror ingest (dup entry)
        }
        uint32_t slot;
        if (found >= 0) {
            slot = static_cast<uint32_t>(found);
        } else if (!gb_merge_insert_group(dst, src, g, h, &slot)) {
            return false;
        }
        if (!gb_merge_combine(dst, slot, src, g)) return false;
    }
    // n_rows_in is write-only bookkeeping; credit it once (class 0) so the
    // sum across class states matches the full-merge total.
    if (class_idx == 0) dst->n_rows_in += src->n_rows_in;
    return true;
}

inline bool groupby_agg_multi_key_typed_merge(
        GroupbyTypedState* dst, const GroupbyTypedState* src) noexcept {
    return groupby_agg_multi_key_typed_merge_class(dst, src, 0, 1);
}

// PRE-HASHED LIST merge: fold the src groups listed in gids[0..n) into
// dst, reading each group's stored-key hash from the GID-INDEXED array
// hashes_by_gid (hashes_by_gid[g] for group g). The fast path for the
// parallel class merge: a bucket pre-pass hashes every entry ONCE
// (parallel by partial) and partitions gids by class, so the N class
// workers stop re-hashing all entries N times (measured: the
// rescan-and-rehash variant spent ~5x the insert cost in hashing).
// hashes_by_gid[g] MUST equal gb_merge_hash_stored(src, g).
inline bool groupby_agg_multi_key_typed_merge_list(
        GroupbyTypedState* dst, const GroupbyTypedState* src,
        const uint32_t* gids, const uint64_t* hashes_by_gid,
        uint32_t n) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(gids != nullptr || n == 0);
    assert(hashes_by_gid != nullptr || n == 0);
    assert(dst->n_keys == src->n_keys);
    assert(dst->n_aggs == src->n_aggs);
    assert(!dst->any_distinct && !src->any_distinct &&
           "DISTINCT partials are not mergeable (v1)");
    if (dst->oom || src->oom) return false;
    if (dst->any_distinct || src->any_distinct) return false;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t g = gids[i];
        assert(g < src->entry_count);
        const uint64_t h = hashes_by_gid[g];
        assert(h == gb_merge_hash_stored(src, g));
        int32_t found = dst->table->find(h);
        if (found >= 0 &&
            !gb_merge_keys_equal(dst, static_cast<uint32_t>(found), src, g)) {
            found = -1;   // 64-bit hash collision: mirror ingest (dup entry)
        }
        uint32_t slot;
        if (found >= 0) {
            slot = static_cast<uint32_t>(found);
        } else if (!gb_merge_insert_group(dst, src, g, h, &slot)) {
            return false;
        }
        if (!gb_merge_combine(dst, slot, src, g)) return false;
    }
    return true;
}
