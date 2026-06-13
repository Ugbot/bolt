// bolt_groupby.h — Two-phase group-by aggregate (Tiger Style).
//
// K-AGG-A typed multi-key surface lives at the bottom of this header — see
// `groupby_agg_multi_key_typed` and the `AggKind` / `AggSpec` / `GbCell16`
// types. The legacy `groupby_agg_int64` int64-only path is untouched.
//
// Phase 1: thread-local partial aggregates keyed by group key via a SwissTable.
//          Aggregate payload = {sum, count, min, max}.
// Phase 2: merge partials into a global SwissTable. With a single caller thread
//          the merge step is trivial; the API is still shaped for N-way merge.
//
// Constant-column fast path: if `keys.format == ColumnFormat::Constant` we
// skip hashing entirely and produce a single output row.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_branchless.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_config.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/join/bolt_groupby_distinct.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/kernels/bolt_decimal.h"
#include "bolt/kernels/bolt_utf8.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <climits>

namespace bolt {

struct GroupAggregate {
    int64_t sum;
    int64_t count;
    int64_t min;
    int64_t max;
};

// ---------------------------------------------------------------------------
// A parallel payload table indexed by SwissSlot.value (slot id inside the
// table). We grow it alongside SwissTable.size — one GroupAggregate per live
// group. Arena-allocated, fixed capacity = SwissTable capacity.
// ---------------------------------------------------------------------------
struct GroupByTable {
    SwissTable       ht;             // key -> group_id (slot index into payload)
    GroupAggregate*  payload;        // capacity entries, indexed by group_id
    uint64_t*        group_keys;     // parallel: group_id -> original key
    uint32_t         num_groups;
    uint32_t         capacity;
    // Runtime lookahead-prefetch distance. 0 (default) disables prefetch
    // — best when the active distinct-key working set fits L1 (1BRC's
    // 413-stations case). Set to ~16 when the hot set spills L1
    // (cardinality ≥ ~10K). Caller sets this on the table after create
    // when expected cardinality is known. Compile-time default comes
    // from `BOLT_GROUPBY_PREFETCH_AHEAD`.
    uint16_t         prefetch_ahead;
    uint16_t         _pad;

    // capacity_hint is the MAX expected distinct keys. Tight-sizing means
    // capacity rounds up to the smallest power-of-two ≥ capacity_hint
    // (no 2× oversize). Better cache density for known-cardinality
    // workloads (e.g. 1BRC: 413 stations → 512 slots vs 1024 default).
    // If real cardinality exceeds the hint, inserts return false — Tiger
    // Style: no growth on the hot path; caller must size the hint right.
    static bool create(GroupByTable* out, uint64_t capacity_hint, Arena* arena) noexcept {
        assert(out != nullptr);
        assert(arena != nullptr);
        if (!SwissTable::create_with(&out->ht, capacity_hint, arena,
                                     /*tight_sizing=*/true)) return false;
        uint32_t cap = out->ht.capacity;
        // payload is sized to cap+1 so ingest_two_pass can use slot
        // [cap] as a throwaway dummy sink for missed-key writes.
        // Cost: one extra GroupAggregate (40 B) per partial table.
        out->payload    = arena->allocate_array<GroupAggregate>(cap + 1);
        out->group_keys = arena->allocate_array<uint64_t>(cap);
        if (!out->payload || !out->group_keys) return false;
        memset(out->payload,    0, (cap + 1) * sizeof(GroupAggregate));
        memset(out->group_keys, 0, cap * sizeof(uint64_t));
        // Initialize the dummy sink at index cap with min=MAX, max=MIN
        // so its values can never affect a real group's aggregates if
        // accidentally promoted.
        out->payload[cap] = {0, 0, INT64_MAX, INT64_MIN};
        out->num_groups     = 0;
        out->capacity       = cap;
        out->prefetch_ahead =
            static_cast<uint16_t>(config::kGroupbyPrefetchAhead);
        out->_pad           = 0;
        return true;
    }

    // Unchecked ingest variant: assumes capacity is sufficient (caller
    // pre-sized the table with hint >= max distinct keys). Drops the
    // per-row capacity check — one fewer branch in the hot morsel loop.
    // The hit-vs-insert branch is still here (inherent to a hash table
    // — different work per side) but it's one of the most-predictable
    // branches in the codebase: after the first sample of each key the
    // path is always "hit". Min/max use branchless `bmin`/`bmax`.
    void ingest_unchecked(uint64_t key, int64_t value) noexcept {
        assert(payload != nullptr);
        assert(group_keys != nullptr);
        assert(num_groups < capacity);   // caller's contract
        int32_t existing = ht.find(key);
        if (existing >= 0) {
            GroupAggregate& g = payload[existing];
            g.sum   += value;
            g.count += 1;
            g.min = bolt::branchless::bmin(g.min, value);
            g.max = bolt::branchless::bmax(g.max, value);
            return;
        }
        uint32_t gid = num_groups++;
        ht.insert(key, gid);             // pre-sized; cannot fail
        GroupAggregate& g = payload[gid];
        g.sum = value; g.count = 1; g.min = value; g.max = value;
        group_keys[gid] = key;
    }

    // Two-pass batched ingest — separates probe (Pass A) from
    // accumulate (Pass B), with misses queued for a third small loop
    // (Pass C). The point is to remove the per-row hit-vs-insert
    // branch (BR5) from the hot path: Pass B uses a CMOV-style
    // "select between hit-slot and dummy-slot" so every row writes
    // unconditionally. Misses are corrected in Pass C (rare path
    // after warmup since each new key is only missed once).
    //
    // `existing_scratch` and `miss_idx_scratch` must be ≥ n entries.
    // Caller owns them; typically allocated from the morsel arena.
    //
    // See docs/research/branchless-hashing.md (two-pass hoisted probe).
    void ingest_two_pass(const uint64_t* BOLT_RESTRICT keys,
                         const int64_t*  BOLT_RESTRICT values,
                         int64_t         n,
                         int32_t* BOLT_RESTRICT existing_scratch,
                         uint32_t* BOLT_RESTRICT miss_idx_scratch) noexcept {
        assert(payload != nullptr);
        assert(group_keys != nullptr);
        assert(n >= 0);
        assert(num_groups + n <= capacity);  // worst-case all distinct

        // Pass A — probe all keys; write existing[i] = slot or -1.
        for (int64_t i = 0; i < n; ++i) {
            existing_scratch[i] = ht.find(keys[i]);
        }

        // Pass B — branchless accumulate into the hit slot OR a
        // throwaway dummy slot at index = capacity (we never insert
        // there). Misses get queued.
        // The dummy slot at payload[capacity] requires payload to be
        // allocated with `capacity + 1` entries — we patch this in
        // create() too.
        uint32_t miss_n = 0;
        const int32_t kDummy = static_cast<int32_t>(capacity);
        for (int64_t i = 0; i < n; ++i) {
            const int32_t hit  = existing_scratch[i];
            const int32_t slot = (hit >= 0) ? hit : kDummy;
            const int64_t v    = values[i];
            GroupAggregate& g  = payload[slot];
            g.sum   += v;
            g.count += 1;
            g.min = bolt::branchless::bmin(g.min, v);
            g.max = bolt::branchless::bmax(g.max, v);
            // Queue miss index (branchless conditional store).
            miss_idx_scratch[miss_n] = static_cast<uint32_t>(i);
            miss_n += (hit < 0) ? 1u : 0u;
        }

        // Pass C — handle the misses: insert and re-apply update.
        // Also undo the dummy-slot accumulation for these (we
        // intentionally double-counted them in Pass B; subtract here).
        // For the FIRST miss of a given key only, do the insert; for
        // duplicate misses of the same new key (multiple new rows
        // before any insert), the second one will find it after we
        // process the first — we re-probe per miss.
        for (uint32_t mi = 0; mi < miss_n; ++mi) {
            const uint32_t i = miss_idx_scratch[mi];
            const uint64_t k = keys[i];
            const int64_t  v = values[i];
            int32_t existing = ht.find(k);
            uint32_t gid;
            if (existing >= 0) {
                gid = static_cast<uint32_t>(existing);
            } else {
                gid = num_groups++;
                ht.insert(k, gid);
                payload[gid] = {0, 0, INT64_MAX, INT64_MIN};
                group_keys[gid] = k;
            }
            GroupAggregate& g = payload[gid];
            g.sum   += v;
            g.count += 1;
            g.min = bolt::branchless::bmin(g.min, v);
            g.max = bolt::branchless::bmax(g.max, v);
        }

        // Reset the dummy slot — it was used as a write sink in Pass B
        // and its accumulated values are noise we discard.
        payload[kDummy] = {0, 0, INT64_MAX, INT64_MIN};
    }

    // Checked ingest — kept for callers that don't pre-size or want a
    // hard cap. Returns false on capacity overflow.
    // Branchless min/max via bolt::branchless::bmin/bmax — both compile
    // to CMOV under MSVC /arch:AVX2 and gcc/clang -O2, eliminating the
    // ~50% branch-mispredict tax on the first sample per group.
    bool ingest(uint64_t key, int64_t value) noexcept {
        assert(payload != nullptr);
        assert(group_keys != nullptr);
        int32_t existing = ht.find(key);
        if (existing >= 0) {
            GroupAggregate& g = payload[existing];
            g.sum   += value;
            g.count += 1;
            g.min = bolt::branchless::bmin(g.min, value);
            g.max = bolt::branchless::bmax(g.max, value);
            return true;
        }
        if (num_groups >= capacity) return false;
        uint32_t gid = num_groups++;
        if (!ht.insert(key, gid)) { --num_groups; return false; }
        GroupAggregate& g = payload[gid];
        g.sum = value; g.count = 1; g.min = value; g.max = value;
        group_keys[gid] = key;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Helper: allocate a Flat int64 column in the arena with `length` rows.
// ---------------------------------------------------------------------------
inline bool gb_alloc_flat_i64(BoltColumn* out, int64_t length, Arena* arena) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr);
    *out = BoltColumn::make_flat_alloc(length, BoltType::Int64, arena);
    return out->data != nullptr || length == 0;
}

// ---------------------------------------------------------------------------
// Public entry point — full {sum, count, min, max} aggregate. Returns false
// on OOM or bad shape. Output columns:
//   out_keys:   Flat int64 column with one row per distinct group.
//   out_sums:   Flat int64 column, sum per group.
//   out_counts: nullable. Flat int64 column, row count per group.
//   out_mins:   nullable. Flat int64 column, min value per group.
//   out_maxes:  nullable. Flat int64 column, max value per group.
//
// Pass nullptr for any of the trailing three to skip emitting it. The thin
// shim `groupby_sum_int64` below preserves the v0.1 API by passing nullptrs
// for count/min/max.
// ---------------------------------------------------------------------------
inline bool groupby_agg_int64(const BoltColumn& keys,
                              const BoltColumn& values,
                              Arena* arena,
                              BoltColumn* out_keys,
                              BoltColumn* out_sums,
                              BoltColumn* out_counts,
                              BoltColumn* out_mins,
                              BoltColumn* out_maxes) noexcept {
    assert(arena   != nullptr);
    assert(out_keys != nullptr);
    assert(out_sums != nullptr);
    if (keys.length != values.length) return false;
    if (values.type != BoltType::Int64) return false;

    const int64_t n = keys.length;
    const int64_t* vals = static_cast<const int64_t*>(values.data);

    // ---- Constant-column fast path ------------------------------------
    if (keys.format == ColumnFormat::Constant) {
        if (!gb_alloc_flat_i64(out_keys, 1, arena)) return false;
        if (!gb_alloc_flat_i64(out_sums, 1, arena)) return false;
        if (out_counts && !gb_alloc_flat_i64(out_counts, 1, arena)) return false;
        if (out_mins   && !gb_alloc_flat_i64(out_mins,   1, arena)) return false;
        if (out_maxes  && !gb_alloc_flat_i64(out_maxes,  1, arena)) return false;
        int64_t k = keys.get_constant<int64_t>();
        int64_t total = 0;
        int64_t mn = INT64_MAX, mx = INT64_MIN;
        assert(n >= 0 && n < (1LL << 40));
        if (values.format == ColumnFormat::Constant) {
            int64_t v = values.get_constant<int64_t>();
            total = v * n;
            mn = (n > 0) ? v : 0;
            mx = (n > 0) ? v : 0;
        } else {
            assert(vals != nullptr || n == 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t v = vals[i];
                total += v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (n == 0) { mn = 0; mx = 0; }
        }
        static_cast<int64_t*>(out_keys->data)[0] = k;
        static_cast<int64_t*>(out_sums->data)[0] = total;
        if (out_counts) static_cast<int64_t*>(out_counts->data)[0] = n;
        if (out_mins)   static_cast<int64_t*>(out_mins->data)[0]   = mn;
        if (out_maxes)  static_cast<int64_t*>(out_maxes->data)[0]  = mx;
        return true;
    }

    if (keys.format != ColumnFormat::Flat && keys.format != ColumnFormat::View) {
        return false;
    }
    if (keys.type != BoltType::Int64 && keys.type != BoltType::UInt64) return false;
    if (values.format != ColumnFormat::Flat && values.format != ColumnFormat::View) {
        return false;
    }

    const uint64_t* ks = static_cast<const uint64_t*>(keys.data);

    // ---- Phase 1: single thread-local partial -------------------------
    GroupByTable partial;
    uint64_t hint = n > 0 ? static_cast<uint64_t>(n) : 1;
    if (!GroupByTable::create(&partial, hint, arena)) return false;
    // Runtime-tunable lookahead prefetch via partial.prefetch_ahead.
    // Loop-peeled: the main body has NO per-row bound check on the
    // prefetch lookahead, NO per-row capacity check on ingest. The
    // tail loop (last `pf` rows) skips prefetch only.
    //
    // Default prefetch_ahead = 0 (off) wins on low-cardinality
    // workloads (1BRC-shape: 413 keys, hot set in L1 across repeat
    // probes — prefetch instr is pure overhead). Set to ~16 on the
    // table when cardinality is high enough (~10K+) that the hot set
    // spills L1. See docs/research/design-log.md (Wave M1).
    const int64_t pf = partial.prefetch_ahead;
    if (pf > 0 && n > pf) {
        const int64_t main_n = n - pf;
        for (int64_t i = 0; i < main_n; ++i) {
            partial.ht.prefetch(ks[i + pf]);
            partial.ingest_unchecked(ks[i], vals[i]);
        }
        for (int64_t i = main_n; i < n; ++i) {
            partial.ingest_unchecked(ks[i], vals[i]);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            partial.ingest_unchecked(ks[i], vals[i]);
        }
    }

    // ---- Phase 2: merge. With one partial, emit directly -------------
    // Shape retained for future N-partial merge.
    GroupByTable global;
    if (!GroupByTable::create(&global, partial.num_groups + 1, arena)) return false;
    for (uint32_t g = 0; g < partial.num_groups; ++g) {
        uint64_t k = partial.group_keys[g];
        int32_t existing = global.ht.find(k);
        if (existing >= 0) {
            GroupAggregate& dst = global.payload[existing];
            const GroupAggregate& src = partial.payload[g];
            dst.sum   += src.sum;
            dst.count += src.count;
            if (src.min < dst.min) dst.min = src.min;
            if (src.max > dst.max) dst.max = src.max;
        } else {
            if (global.num_groups >= global.capacity) return false;
            uint32_t gid = global.num_groups++;
            if (!global.ht.insert(k, gid)) { --global.num_groups; return false; }
            global.payload[gid]    = partial.payload[g];
            global.group_keys[gid] = k;
        }
    }

    // ---- Emit output columns ------------------------------------------
    int64_t out_n = static_cast<int64_t>(global.num_groups);
    if (!gb_alloc_flat_i64(out_keys, out_n, arena)) return false;
    if (!gb_alloc_flat_i64(out_sums, out_n, arena)) return false;
    if (out_counts && !gb_alloc_flat_i64(out_counts, out_n, arena)) return false;
    if (out_mins   && !gb_alloc_flat_i64(out_mins,   out_n, arena)) return false;
    if (out_maxes  && !gb_alloc_flat_i64(out_maxes,  out_n, arena)) return false;
    int64_t* ok = static_cast<int64_t*>(out_keys->data);
    int64_t* os = static_cast<int64_t*>(out_sums->data);
    int64_t* oc = out_counts ? static_cast<int64_t*>(out_counts->data) : nullptr;
    int64_t* on = out_mins   ? static_cast<int64_t*>(out_mins->data)   : nullptr;
    int64_t* ox = out_maxes  ? static_cast<int64_t*>(out_maxes->data)  : nullptr;
    for (int64_t i = 0; i < out_n; ++i) {
        ok[i] = static_cast<int64_t>(global.group_keys[i]);
        const GroupAggregate& g = global.payload[i];
        os[i] = g.sum;
        if (oc) oc[i] = g.count;
        if (on) on[i] = g.min;
        if (ox) ox[i] = g.max;
    }
    return true;
}

// ---------------------------------------------------------------------------
// v0.1 sum-only entry point — thin shim over groupby_agg_int64.
// ---------------------------------------------------------------------------
inline bool groupby_sum_int64(const BoltColumn& keys,
                              const BoltColumn& values,
                              Arena* arena,
                              BoltColumn* out_keys,
                              BoltColumn* out_sums) noexcept {
    return groupby_agg_int64(keys, values, arena, out_keys, out_sums,
                             nullptr, nullptr, nullptr);
}

// ---------------------------------------------------------------------------
// groupby_sum_int32 — int32 keys, int64 values.
//
// Implementation: widen keys into an arena-allocated uint64 buffer, delegate
// to groupby_sum_int64 (which already handles SwissTable hashing, Constant
// fast path, and emit). Then narrow the int64 output keys into a Flat int32
// column. This keeps the body tiny and shares all the non-trivial logic.
// ---------------------------------------------------------------------------
inline bool groupby_sum_int32(const BoltColumn& keys_i32,
                              const BoltColumn& values_i64,
                              Arena* arena,
                              BoltColumn* out_keys,
                              BoltColumn* out_sums) noexcept {
    assert(arena   != nullptr);
    assert(out_keys != nullptr);
    assert(out_sums != nullptr);
    if (keys_i32.type != BoltType::Int32) return false;
    if (keys_i32.length != values_i64.length) return false;

    // Constant-key fast path: forward as an Int64 Constant column.
    if (keys_i32.format == ColumnFormat::Constant) {
        int32_t k32 = keys_i32.get_constant<int32_t>();
        BoltColumn kc64 = BoltColumn::make_constant<int64_t>(
            static_cast<int64_t>(k32), keys_i32.length, BoltType::Int64);
        BoltColumn tmp_keys{}, tmp_sums{};
        if (!groupby_sum_int64(kc64, values_i64, arena, &tmp_keys, &tmp_sums)) return false;
        // Narrow output keys to int32.
        *out_keys = BoltColumn::make_flat_alloc(tmp_keys.length, BoltType::Int32, arena);
        if (tmp_keys.length > 0 && out_keys->data == nullptr) return false;
        const int64_t* src = static_cast<const int64_t*>(tmp_keys.data);
        int32_t* dst = static_cast<int32_t*>(out_keys->data);
        for (int64_t i = 0; i < tmp_keys.length; ++i) dst[i] = static_cast<int32_t>(src[i]);
        *out_sums = tmp_sums;
        return true;
    }

    if (keys_i32.format != ColumnFormat::Flat && keys_i32.format != ColumnFormat::View) {
        return false;
    }

    // Widen keys in place in the arena.
    const int64_t n = keys_i32.length;
    uint64_t* widened = arena->allocate_array<uint64_t>(n > 0 ? n : 1);
    if (!widened && n > 0) return false;
    const int32_t* ks32 = static_cast<const int32_t*>(keys_i32.data);
    assert(ks32 != nullptr || n == 0);
    for (int64_t i = 0; i < n; ++i) {
        widened[i] = static_cast<uint64_t>(static_cast<int64_t>(ks32[i]));
    }
    BoltColumn wide_keys = BoltColumn::make_flat(widened, nullptr, n, BoltType::Int64);

    BoltColumn tmp_keys{}, tmp_sums{};
    if (!groupby_sum_int64(wide_keys, values_i64, arena, &tmp_keys, &tmp_sums)) return false;

    // Narrow output keys int64 -> int32.
    *out_keys = BoltColumn::make_flat_alloc(tmp_keys.length, BoltType::Int32, arena);
    if (tmp_keys.length > 0 && out_keys->data == nullptr) return false;
    const int64_t* src = static_cast<const int64_t*>(tmp_keys.data);
    int32_t* dst = static_cast<int32_t*>(out_keys->data);
    for (int64_t i = 0; i < tmp_keys.length; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
    *out_sums = tmp_sums;
    return true;
}

// ---------------------------------------------------------------------------
// Utf8 byte-lex MIN / MAX helpers for typed GROUP BY aggregates.
//
// Semantics: SQL MIN/MAX over Utf8 columns are byte-lexicographic — short
// strings do NOT automatically beat long strings. The German-style 16-byte
// StringView layout (length + 4B prefix + 8B inline OR 8B spilled-ref) gives
// us a cheap screen: the 4-byte prefix decides >99% of comparisons in TPC-H
// without a pointer chase. Only when the prefix-decided sign is 0 (i.e.
// both views agree on min(length, 4) prefix bytes) do we fall back to a full
// byte compare via `sv_compare`.
//
// Branch-free inner loop: both helpers use a predicated select on a bool —
// MSVC /O2 and clang -O2 compile this to a CMOV chain. No `if` in the hot
// path body; the prefix-screen branch sits behind the `cmp_prefix == 0`
// tie-break which is rare on real workloads (the typed agg's hot path stays
// inside the prefix-decided fast path).
//
// ns/row floor (single-thread, RelWithDebInfo, AVX2 tier, inline keys only):
//   ≤ 2.0 ns/row at cardinality 1024 (prefix-decided)
//   ≤ 6.0 ns/row at cardinality 1024 when 100% of pairs tie on prefix
//     (forces fall-back compare; pathological — real strings disagree fast).
//
// Tiger Style: noexcept, ≤70 lines per fn, ≥2 asserts, no allocations, no
// std::string, no smart pointers, raw pointers + POD.
// ---------------------------------------------------------------------------
BOLT_FORCE_INLINE int sv_bytelex_compare(
        const StringView& a, const char* a_base,
        const StringView& b, const char* b_base) noexcept {
    // First: cheap prefix-only screen (length + min(length,4) prefix bytes).
    // Decides without touching the spilled tail in the common case.
    const int pc = StringView::cmp_prefix(a, b);
    if (pc != 0) return pc;
    // Tail-disambiguation: full byte-lex compare. `sv_compare` handles
    // inline vs spilled internally via sv_bytes(). Tiger Style: ≥ 2 asserts.
    assert(a.length <= 12u || a_base != nullptr);
    assert(b.length <= 12u || b_base != nullptr);
    return kernels::utf8::sv_compare(a, a_base, b, b_base);
}

// Column compare of two StringView columns → int64 {0,1} (the engine's boolean
// column), for the column-at-a-time expression evaluator's Utf8 predicates
// (e.g. p_brand = 'Brand#12'). `op`: 0=eq 1=ne 2=lt 3=le 4=gt 5=ge. Spilled
// (>12-byte) views resolve via the per-column base; inline views ignore it.
// Each element runs the same length+prefix-screened sv_bytelex_compare the
// per-row path uses, so ordering is identical. Caller owns `out`.
BOLT_FORCE_INLINE void sv_cmp_col(
        const StringView* BOLT_RESTRICT a, const char* a_base,
        const StringView* BOLT_RESTRICT b, const char* b_base,
        int64_t n, int op, int64_t* BOLT_RESTRICT out) noexcept {
    assert((a != nullptr && b != nullptr && out != nullptr) || n == 0);
    assert(n >= 0 && op >= 0 && op <= 5);
    // BRANCH-FREE final map: `op` is loop-invariant so only one predicate term
    // survives constant-folding (the byte compare itself screens length+prefix).
    for (int64_t i = 0; i < n; ++i) {
        const int c = sv_bytelex_compare(a[i], a_base, b[i], b_base);
        out[i] = static_cast<int64_t>(
            ((op == 0) & (c == 0)) | ((op == 1) & (c != 0)) |
            ((op == 2) & (c <  0)) | ((op == 3) & (c <= 0)) |
            ((op == 4) & (c >  0)) | ((op == 5) & (c >= 0)));
    }
}

// Branch-free byte-lex MIN over two StringViews. Returns the smaller of
// (cur, cand) under byte-lex order. Caller passes spilled bases (nullptr
// when both views are known inline). Predicated select compiles to CMOV.
BOLT_FORCE_INLINE StringView sv_bytelex_min(
        const StringView& cur, const char* cur_base,
        const StringView& cand, const char* cand_base) noexcept {
    assert(cur.length  <= 12u || cur_base  != nullptr);
    assert(cand.length <= 12u || cand_base != nullptr);
    const int c = sv_bytelex_compare(cand, cand_base, cur, cur_base);
    // c < 0  ⇒ cand < cur ⇒ select cand. Branch-free predicated select.
    return (c < 0) ? cand : cur;
}

// Branch-free byte-lex MAX over two StringViews. Symmetric to sv_bytelex_min.
BOLT_FORCE_INLINE StringView sv_bytelex_max(
        const StringView& cur, const char* cur_base,
        const StringView& cand, const char* cand_base) noexcept {
    assert(cur.length  <= 12u || cur_base  != nullptr);
    assert(cand.length <= 12u || cand_base != nullptr);
    const int c = sv_bytelex_compare(cand, cand_base, cur, cur_base);
    return (c > 0) ? cand : cur;
}

// ===========================================================================
// K-AGG-A — typed multi-key GROUP BY + aggregate matrix kernel.
//
// Surface (header-only, noexcept, Tiger Style):
//
//   bool groupby_agg_multi_key_typed(
//       const BoltColumn* keys,     size_t n_keys,
//       const BoltColumn* payload,  size_t n_payload,
//       const AggSpec*    aggs,     size_t n_aggs,
//       int64_t           n_rows,
//       BoltColumn*       out_keys,    // n_keys columns (Flat per key type)
//       BoltColumn*       out_aggs,    // n_aggs columns (Flat per agg out)
//       uint32_t*         ngroups_out,
//       Arena*            arena);
//
// Single-thread, branch-free apply (CMOV-style bmin/bmax for integer MIN/MAX),
// composite-key fixed cap = kGbMaxKeys, agg-matrix fixed cap = kGbMaxAggs,
// per-group cell width = 16 bytes (GbCell16 holds Int64 / Decimal128 / Date32
// / Utf8 inline-StringView identities). All scratch lives in the caller-
// supplied Arena — no hot-path heap.
//
// Deferred to K-AGG-B (NOT in this kernel): Utf8 spilled keys, COUNT(DISTINCT),
// partitioned-parallel variant. The chukonu wrapper at
// `src/operators/impl/hash_agg_typed_op.cpp` keeps owning those features
// until this kernel's surface is extended in K-AGG-B.
// ===========================================================================

// Per-cell payload — 16 bytes covers Int64 (.a only), Decimal128 ({lo,hi}),
// Date32 (.a low 32 bits sign-extended), Utf8 inline StringView (16 bytes).
struct GbCell16 {
    int64_t a;
    int64_t b;
};
static_assert(sizeof(GbCell16) == 16, "GbCell16 must be 16 bytes");

enum class AggKind : uint8_t {
    Sum       = 0,
    Count     = 1,   // counts non-null input values
    CountStar = 2,   // counts every row (in_col ignored)
    Min       = 3,
    Max       = 4,
    Avg       = 5,
};

struct AggSpec {
    AggKind  kind;
    uint8_t  in_col;     // index into the `payload` array; ignored for CountStar
    uint8_t  distinct;   // K-AGG-A.2 item 2: 1 = fold duplicates per (agg, group)
    // W-DEC inc 4: accumulator-lane stamp for Decimal64 Sum/Avg.
    //   0 = auto/unproven (default) — overflow-safe widened-d128 sum loops.
    //   1 = CALLER CONTRACT: the planner PROVED at plan time that
    //       row_bound × max|value| ≤ INT64_MAX, hence EVERY partial sum of
    //       this aggregate (any subset, any sign mix) fits an int64. The
    //       kernel never verifies this in release (debug-asserts at
    //       finalize); it only selects faster pure-int64 window loops.
    // Representation invariant: lane NEVER changes what the accumulator
    // cell stores — it stays a true Decimal128 — so lane-1 windows, lane-0
    // windows, and per-row fallback windows compose freely.
    uint8_t  lane;
    uint8_t  _pad[4];    // explicit padding; POD, layout-stable
};
static_assert(sizeof(AggSpec) == 8, "AggSpec must be 8 bytes");

inline constexpr uint32_t kGbMaxKeys  = 8;   // was 4; TPC-H Q10 GROUP BYs 7 cols
inline constexpr uint32_t kGbMaxAggs  = 16;
inline constexpr uint32_t kGbEntryCap = 1u << 24;   // 16M groups hard cap
// K-AGG-C: ingest processes rows in windows of this size; the gid
// scratch (one int32 per window row) is allocated once in _begin.
#ifndef BOLT_GROUPBY_INGEST_WINDOW
#define BOLT_GROUPBY_INGEST_WINDOW 65536u
#endif
inline constexpr uint32_t kGbIngestWindow = BOLT_GROUPBY_INGEST_WINDOW;
// K-AGG-C: bank-split accumulation. 4 banks of up to 1024 groups; a window
// is bank-eligible when entry_count <= kGbBankGroups and the window is big
// enough that zeroing the bank slices amortises (see gb_pass2_one).
inline constexpr uint32_t kGbBankCount  = 4u;
inline constexpr uint32_t kGbBankGroups = 1024u;

namespace gb_detail {

// Read one row of a column into a 16-byte cell. Branch-free per cell type.
BOLT_FORCE_INLINE GbCell16 read_cell16(const BoltColumn& c, int64_t r) noexcept {
    assert(c.data != nullptr);
    assert(r >= 0);
    GbCell16 out{0, 0};
    switch (c.type) {
        case BoltType::Decimal128: {
            const auto* p = static_cast<const kernels::decimal::Decimal128*>(c.data);
            std::memcpy(&out, &p[r], 16);
            break;
        }
        case BoltType::Int64:    out.a = static_cast<const int64_t*>(c.data)[r]; break;
        case BoltType::Int32:    out.a = static_cast<const int32_t*>(c.data)[r]; break;
        case BoltType::Date32:   out.a = static_cast<const int32_t*>(c.data)[r]; break;
        case BoltType::Decimal64: {
            // W-DEC: widen the int64 mantissa to a full d128 cell (sign-
            // extended) so the Decimal128 accumulate/compare paths apply
            // verbatim — sums can grow past int64 with zero proof needed.
            const int64_t m = static_cast<const int64_t*>(c.data)[r];
            out.a = m;
            out.b = (m < 0) ? -1 : 0;
            break;
        }
        case BoltType::Float64:  std::memcpy(&out.a, &static_cast<const double*>(c.data)[r], 8); break;
        case BoltType::Utf8:     std::memcpy(&out, &static_cast<const StringView*>(c.data)[r], 16); break;
        default:                 out.a = static_cast<const int64_t*>(c.data)[r]; break;
    }
    return out;
}

// Card S: hash a Utf8 key's CONTENT bytes (length + bytes) so spilled and
// inline views with equal content hash equal regardless of where the bytes
// physically live. `base` resolves spilled (>12) views; ignored for inline.
BOLT_FORCE_INLINE uint64_t hash_utf8_content(const StringView& sv,
                                             const char* base) noexcept {
    const char* p = kernels::utf8::sv_bytes(sv, base);
    uint64_t h = swiss_mix_wyhash3(0x9E3779B97F4A7C15ULL ^
                                   static_cast<uint64_t>(sv.length));
    uint32_t i = 0;
    for (; i + 8u <= sv.length; i += 8u) {
        uint64_t chunk;
        std::memcpy(&chunk, p + i, 8);
        h = swiss_mix_wyhash3(h ^ chunk);
    }
    if (i < sv.length) {
        uint64_t tail = 0;
        std::memcpy(&tail, p + i, static_cast<size_t>(sv.length - i));
        h = swiss_mix_wyhash3(h ^ tail);
    }
    return h;
}

// Composite-key hash. Utf8 keys hash by content bytes (Card S) so spilled
// (>12-char) keys group by value; other types hash the 16-byte cell halves.
// Order-sensitive mix over key columns.
BOLT_FORCE_INLINE uint64_t hash_keys(const BoltColumn* keys, uint32_t n_keys,
                                      int64_t r) noexcept {
    assert(keys != nullptr || n_keys == 0);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (uint32_t k = 0; k < n_keys; ++k) {
        if (keys[k].type == BoltType::Utf8) {
            StringView sv;
            std::memcpy(&sv,
                        &static_cast<const StringView*>(keys[k].data)[r], 16);
            h = swiss_mix_wyhash3(h ^ hash_utf8_content(
                sv, static_cast<const char*>(keys[k].str_overflow_base)));
            continue;
        }
        const GbCell16 c = read_cell16(keys[k], r);
        h ^= swiss_mix_wyhash3(static_cast<uint64_t>(c.a));
        h  = swiss_mix_wyhash3(h ^ static_cast<uint64_t>(c.b));
    }
    return h;
}

// True if row `r`'s composite key matches the group stored at `gid`. Utf8 keys
// compare by CONTENT bytes (Card S): the input row resolves via its column's
// str_overflow_base, the stored representative via `stored_bases[k]` (the
// groupby's per-key deep-copy buffer). Other types compare the cell halves.
BOLT_FORCE_INLINE bool keys_equal(const BoltColumn* keys, uint32_t n_keys,
                                   int64_t r, const GbCell16* keys_flat,
                                   uint32_t gid,
                                   char* const* stored_bases) noexcept {
    assert(keys != nullptr || n_keys == 0);
    assert(keys_flat != nullptr);
    const GbCell16* row = keys_flat + static_cast<size_t>(gid) * n_keys;
    for (uint32_t k = 0; k < n_keys; ++k) {
        if (keys[k].type == BoltType::Utf8) {
            StringView in_sv;
            std::memcpy(&in_sv,
                        &static_cast<const StringView*>(keys[k].data)[r], 16);
            StringView st_sv;
            std::memcpy(&st_sv, &row[k], 16);
            if (in_sv.length != st_sv.length) return false;
            if (in_sv.length == 0) continue;
            const char* ib = kernels::utf8::sv_bytes(
                in_sv, static_cast<const char*>(keys[k].str_overflow_base));
            const char* sb = kernels::utf8::sv_bytes(st_sv, stored_bases[k]);
            if (std::memcmp(ib, sb, in_sv.length) != 0) return false;
            continue;
        }
        const GbCell16 c = read_cell16(keys[k], r);
        if (c.a != row[k].a || c.b != row[k].b) return false;
    }
    return true;
}

// Aggregate identity for a (kind, in-type) pair.
//
// Utf8 MIN/MAX identities (K-AGG-A.4-PREREQ):
//   The cell aliases a 16-byte inline StringView {length, prefix[4], inline_data[8]}.
//   `apply()` asserts `cur.length <= 12u` on every step (inline-only invariant),
//   so identities MUST encode as valid inline StringViews.
//   * MIN identity: a 12-byte all-0xFF string — byte-lex maximum of any real
//     ASCII/UTF-8 value (real values are < 0xFF in every byte). Encoded as
//     length=12, prefix=0xFF×4 ⇒ a = (0xFFFFFFFF<<32) | 12; inline_data=0xFF×8
//     ⇒ b = 0xFFFFFFFFFFFFFFFF.
//   * MAX identity: the empty string (length=0, all zeros) — byte-lex minimum.
//     {0, 0} is already correct; share it with the Sum/Count path.
BOLT_FORCE_INLINE GbCell16 agg_identity(AggKind k, BoltType t) noexcept {
    // W-DEC: Decimal64 cells are widened d128 (read_cell16), so their
    // Min/Max identities are the d128 extremes too.
    const bool d = (t == BoltType::Decimal128 || t == BoltType::Decimal64);
    const bool s = (t == BoltType::Utf8);
    switch (k) {
        case AggKind::Sum:
        case AggKind::Avg:
        case AggKind::Count:
        case AggKind::CountStar: return GbCell16{0, 0};
        case AggKind::Min:
            if (s) return GbCell16{static_cast<int64_t>(0xFFFFFFFF0000000CULL),
                                   static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)};
            return d ? GbCell16{static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), INT64_MAX}
                     : GbCell16{INT64_MAX, 0};
        case AggKind::Max:
            if (s) return GbCell16{0, 0};   // empty string — byte-lex min
            return d ? GbCell16{0, INT64_MIN} : GbCell16{INT64_MIN, 0};
    }
    return GbCell16{0, 0};
}

// Apply one value to one accumulator slot. Branch-free for the int64/date32
// fast path; Decimal128 falls through to the typed `d128_*` kernels; Utf8
// MIN/MAX go through `sv_bytelex_{min,max}` (K-AGG-A.2 item 3).
//
// `valid` (K-AGG-A.2 item 1) is the input value's non-NULL status:
//   - CountStar : ignores `valid` (always +1 — counts every row).
//   - Count     : +1 iff valid; +0 otherwise.
//   - Sum/Avg   : skip the value when !valid (no slot mutation).
//   - Min/Max   : skip the value when !valid (identity preserved).
// With `valid = true` always, behaviour is identical to the pre-K-AGG-A.2
// surface (no asserts fire, no extra branches in the hot path because
// the per-call `valid` is hoisted into a constant by the wrapper for
// columns lacking a validity bitmap).
BOLT_FORCE_INLINE void apply(AggKind k, BoltType t,
                             GbCell16* slot, GbCell16 v, bool valid) noexcept {
    assert(slot != nullptr);
    if (k == AggKind::CountStar) { slot->a += 1; return; }
    if (k == AggKind::Count)     { slot->a += valid ? 1 : 0; return; }
    if (!valid) return;   // Sum/Min/Max/Avg ignore NULLs (SQL semantics)
    // W-DEC: Decimal64 cells arrive WIDENED to d128 (read_cell16), so the
    // Decimal128 arm is exact for them — sums grow past int64 safely.
    if (t == BoltType::Decimal128 || t == BoltType::Decimal64) {
        namespace dec = kernels::decimal;
        dec::Decimal128 s, x;
        std::memcpy(&s, slot, 16);
        std::memcpy(&x, &v,   16);
        if (k == AggKind::Sum || k == AggKind::Avg) {
            const dec::Decimal128 r = dec::d128_add(s, x);
            std::memcpy(slot, &r, 16);
            return;
        }
        const int c = dec::d128_cmp(x, s);
        const bool take = (k == AggKind::Min) ? (c < 0) : (c > 0);
        if (take) std::memcpy(slot, &v, 16);
        return;
    }
    if (t == BoltType::Utf8 && (k == AggKind::Min || k == AggKind::Max)) {
        // Inline-only invariant: the typed kernel does not yet resolve
        // spilled keys (>12 byte content), so the cell carries the full
        // 16-byte StringView for inline strings. Asserts in
        // `sv_bytelex_{min,max}` guard the inline-only precondition.
        StringView cur, cand;
        std::memcpy(&cur,  slot, sizeof(cur));
        std::memcpy(&cand, &v,   sizeof(cand));
        assert(cur.length  <= 12u);
        assert(cand.length <= 12u);
        const StringView out = (k == AggKind::Min)
            ? sv_bytelex_min(cur, nullptr, cand, nullptr)
            : sv_bytelex_max(cur, nullptr, cand, nullptr);
        std::memcpy(slot, &out, sizeof(out));
        return;
    }
    switch (k) {
        case AggKind::Sum:
        case AggKind::Avg: slot->a += v.a; break;
        case AggKind::Min: slot->a = branchless::bmin(slot->a, v.a); break;
        case AggKind::Max: slot->a = branchless::bmax(slot->a, v.a); break;
        default: break;
    }
}

// Arrow-convention validity: validity==nullptr means all-valid; otherwise the
// bit at (validity_offset + r) is set when the value is non-NULL.
BOLT_FORCE_INLINE bool cell_valid(const BoltColumn& c, int64_t r) noexcept {
    assert(r >= 0);
    if (c.validity == nullptr) return true;
    const int64_t bit = c.validity_offset + r;
    return ((c.validity[bit >> 3] >> (bit & 7)) & 1) != 0;
}

// Decide output BoltType for one agg given its input type.
BOLT_FORCE_INLINE BoltType out_type(AggKind k, BoltType in) noexcept {
    if (k == AggKind::CountStar || k == AggKind::Count) return BoltType::Int64;
    if (k == AggKind::Avg) {
        // W-DEC: Avg(Decimal64) accumulates in the widened d128 cell and
        // finalizes through the SAME d128 rescale(+4)/div path, so the
        // result is Decimal128 — bit-identical to the pre-Decimal64 output.
        return (in == BoltType::Decimal128 || in == BoltType::Decimal64)
                   ? BoltType::Decimal128 : BoltType::Float64;
    }
    // W-DEC: SUM(Decimal64) can exceed the int64 mantissa with no plan-time
    // proof at this layer — the accumulator is the widened d128 cell and
    // the output WIDENS to Decimal128 (exact, no overflow ever). MIN/MAX
    // never grow: they keep Decimal64.
    if (in == BoltType::Decimal64) {
        return (k == AggKind::Sum) ? BoltType::Decimal128 : BoltType::Decimal64;
    }
    // SUM / MIN / MAX: preserve input type for Decimal128/Utf8/Date32; SUM/MIN/MAX
    // of narrow ints emit Int64.
    if (in == BoltType::Decimal128 || in == BoltType::Utf8 || in == BoltType::Date32) {
        return in;
    }
    return BoltType::Int64;
}

}  // namespace gb_detail

// ---------------------------------------------------------------------------
// K-AGG-A.3 - cross-morsel partial-merge API.
//
// Caller arena-allocates a `GroupbyTypedState`, calls `_begin` once, then
// `_ingest` per input morsel (honoring `sel` / `sel_len`), then `_finalize`
// once to emit. The legacy one-shot `groupby_agg_multi_key_typed` is kept
// intact below for backwards compat; both surfaces share the gb_detail::
// helpers above so any kernel fix propagates to both.
//
// State POD invariants:
//   - Trivially-copyable. All pointers borrow from a caller-supplied Arena.
//   - `begin` is the only function that allocates scratch; `ingest` is
//     hot-path allocation-free; `finalize` allocates output columns.
//   - `key_types` / `agg_in_types` are captured at begin() time from
//     caller-supplied descriptors so per-morsel `_ingest` calls work
//     against stable types (no first-morsel type capture inside ingest).
//   - Hard cap = SwissTable capacity. Inserts past cap cause _ingest to
//     flag `state->oom = true`; caller checks after _ingest.
// ---------------------------------------------------------------------------
struct GroupbyTypedState {
    SwissTable*     table;
    GbCell16*       keys_flat;          // [cap * n_keys]
    GbCell16*       accums;             // [n_aggs * cap]
    int64_t*        counts;             // [n_aggs * cap] - AVG denominators
    int32_t*        gid_scratch;        // K-AGG-C: [kGbIngestWindow] pass-1 gids
    // K-AGG-C: bank-split accumulator scratch for low-cardinality windows
    // (entry_count <= kGbBankGroups). Rows cycle 4 banks so consecutive
    // same-group rows never serialize on the same accumulator; banks merge
    // into accums/counts once per window. Optional: null => plain loops.
    GbCell16*       bank_acc;           // [kGbBankCount * kGbBankGroups]
    int64_t*        bank_cnt;           // [kGbBankCount * kGbBankGroups]
    // K-AGG-C: direct-mapped (hash -> gid) cache. Low-cardinality GROUP
    // BYs (TPC-H Q1's 4-6 groups) hit ~100% after warm-up and skip the
    // SwissTable probe; verification against keys_flat keeps it exact.
    uint64_t        gid_cache_h[64];
    int32_t         gid_cache_g[64];
    DistinctCell*   distinct_cells;     // optional [n_distinct * cap]
    DistinctCell16* distinct_cells16;   // optional [n_distinct16 * cap]
    uint16_t        distinct_idx[kGbMaxAggs];     // 0xFFFF = "not 8B-distinct"
    uint16_t        distinct16_idx[kGbMaxAggs];   // 0xFFFF = "not Utf8-distinct"

    AggSpec         specs[kGbMaxAggs];
    BoltType        key_types[kGbMaxKeys];
    uint8_t         key_scales[kGbMaxKeys];
    BoltType        agg_in_types[kGbMaxAggs];
    uint8_t         agg_in_scales[kGbMaxAggs];

    // Card S: per-Utf8-key spilled-byte buffer. A spilled (>12-char) group key
    // lives in the input column's overflow buffer, which is freed after the
    // morsel; the stored representative key must own its bytes. On new-group
    // insert we deep-copy spilled key bytes here and re-point the stored cell's
    // offset; hash/equal resolve spilled keys by these bytes; finalize hands
    // this base to the output column. nullptr for inline-only / non-Utf8 keys.
    char*           key_str_buf[kGbMaxKeys];
    size_t          key_str_used[kGbMaxKeys];
    size_t          key_str_cap[kGbMaxKeys];

    Arena*          arena;
    uint32_t        cap;                // SwissTable capacity (== keys_flat / accums stride)
    uint32_t        entry_count;
    uint32_t        n_rows_in;          // cumulative across all _ingest calls
    // W-P geometric growth bound: max capacity gb_grow may reach (0 or
    // <= cap disables growth — exact legacy fail-fast-at-cap behavior;
    // chukonu's partition-driver worker clamps rely on that). When
    // enabled, a full table quadruples + rehashes at ingest-window /
    // merge-insert boundaries, so a junk cardinality estimate is only a
    // bad STARTING size, never an oom or a 4M-entry init for 7 groups.
    uint32_t        grow_cap;
    // W-P dense gid index (perfect-hash fast path): for a SINGLE int-like
    // group key with caller-certified [min, min+range) bounds,
    // dense_gid[key - dense_min] maps key -> gid, replacing the gid-cache
    // + SwissTable probe + key compare with ONE array load (the 200k-group
    // TPC-H Q17 aggregate is cache-miss bound in exactly that probe).
    // nullptr = disabled. CONSISTENCY: a dense MISS falls through to the
    // full hash find-or-insert and PUBLISHES the resulting gid, so paths
    // that insert without publishing (per-row fallback windows) can never
    // create duplicate groups — under-coverage only costs first-touch
    // misses. The SwissTable stays authoritative (inserts always go there
    // too: merge rehash + growth depend on it).
    int32_t*        dense_gid;          // [dense_range], -1 = empty
    int64_t         dense_min;
    uint32_t        dense_range;
    uint16_t        n_distinct;
    uint16_t        n_distinct16;
    uint8_t         n_keys;
    uint8_t         n_aggs;
    bool            oom;                // set true if any _ingest hit cap
    bool            any_distinct;
    uint8_t         _pad[2];
};

// Initialise state and allocate all scratch tables in `arena`. `key_descs`
// and `payload_descs` are *type-only* descriptors - only `.type` and
// `.decimal_scale` are consulted (`.data` may be nullptr).
//
// `expected_groups_hint`: 0 -> kernel uses a 1024-default cap; >0 ->
// tight-size SwissTable to the smallest power-of-two >= hint. Hard cap
// is kGbEntryCap (16M).
inline bool groupby_agg_multi_key_typed_begin(
        GroupbyTypedState* state,
        Arena*             arena,
        const BoltColumn*  key_descs,
        uint8_t            n_keys,
        const BoltColumn*  payload_descs,
        uint8_t            n_payload,
        const AggSpec*     specs,
        uint8_t            n_aggs,
        uint64_t           expected_groups_hint,
        uint64_t           grow_cap = 0) noexcept {
    assert(state != nullptr);
    assert(arena != nullptr);
    assert(n_keys >= 1 && n_keys <= kGbMaxKeys);
    assert(n_aggs >= 1 && n_aggs <= kGbMaxAggs);
    if (n_keys == 0 || n_keys > kGbMaxKeys) return false;
    if (n_aggs == 0 || n_aggs > kGbMaxAggs) return false;

    std::memset(state, 0, sizeof(*state));
    for (uint16_t i = 0; i < kGbMaxAggs; ++i) {
        state->distinct_idx[i]   = 0xFFFFu;
        state->distinct16_idx[i] = 0xFFFFu;
    }
    for (uint8_t k = 0; k < n_keys; ++k) {
        state->key_types[k]  = key_descs[k].type;
        state->key_scales[k] = key_descs[k].decimal_scale;
    }
    for (uint8_t j = 0; j < n_aggs; ++j) {
        state->specs[j] = specs[j];
        if (specs[j].kind == AggKind::CountStar) {
            state->agg_in_types[j]  = BoltType::Int64;
            state->agg_in_scales[j] = 0;
            continue;
        }
        if (specs[j].in_col >= n_payload || payload_descs == nullptr) return false;
        state->agg_in_types[j]  = payload_descs[specs[j].in_col].type;
        state->agg_in_scales[j] = payload_descs[specs[j].in_col].decimal_scale;
    }
    uint16_t n_distinct = 0;
    uint16_t n_distinct16 = 0;
    for (uint8_t j = 0; j < n_aggs; ++j) {
        if (specs[j].distinct == 0) continue;
        state->any_distinct = true;
        if (state->agg_in_types[j] == BoltType::Utf8)
            state->distinct16_idx[j] = n_distinct16++;
        else
            state->distinct_idx[j] = n_distinct++;
    }
    state->n_distinct   = n_distinct;
    state->n_distinct16 = n_distinct16;

    const bool tight = (expected_groups_hint > 0);
    const uint64_t raw = tight ? expected_groups_hint : 1024ULL;
    auto* ht = static_cast<SwissTable*>(
        arena->allocate(sizeof(SwissTable), alignof(SwissTable)));
    if (ht == nullptr) return false;
    if (!SwissTable::create_with(ht, raw, arena, tight)) return false;
    const uint32_t cap = ht->capacity;
    assert(cap <= kGbEntryCap);

    GbCell16* keys_flat = arena->allocate_array<GbCell16>(
        static_cast<size_t>(cap) * n_keys);
    GbCell16* accums    = arena->allocate_array<GbCell16>(
        static_cast<size_t>(cap) * n_aggs);
    int64_t*  counts    = arena->allocate_array<int64_t>(
        static_cast<size_t>(cap) * n_aggs);
    // K-AGG-C: per-window gid scratch — the ONLY required two-pass
    // allocation, done once here so _ingest stays allocation-free.
    int32_t*  gid_scratch  = arena->allocate_array<int32_t>(kGbIngestWindow);
    if (keys_flat == nullptr || accums == nullptr || counts == nullptr ||
        gid_scratch == nullptr) return false;
    // Bank scratch is optional: null banks just disable the banked pass-2
    // loops (plain gid-driven loops remain correct).
    GbCell16* bank_acc = arena->allocate_array<GbCell16>(
        static_cast<size_t>(kGbBankCount) * kGbBankGroups);
    int64_t*  bank_cnt = arena->allocate_array<int64_t>(
        static_cast<size_t>(kGbBankCount) * kGbBankGroups);
    if (bank_acc == nullptr || bank_cnt == nullptr) {
        bank_acc = nullptr;
        bank_cnt = nullptr;
    }
    std::memset(counts, 0, sizeof(int64_t) * static_cast<size_t>(cap) * n_aggs);

    DistinctCell*   distinct_cells   = nullptr;
    DistinctCell16* distinct_cells16 = nullptr;
    if (n_distinct > 0) {
        distinct_cells = arena->allocate_array<DistinctCell>(
            static_cast<size_t>(n_distinct) * cap);
        if (distinct_cells == nullptr) return false;
        std::memset(distinct_cells, 0,
                    sizeof(DistinctCell) * static_cast<size_t>(n_distinct) * cap);
    }
    if (n_distinct16 > 0) {
        distinct_cells16 = arena->allocate_array<DistinctCell16>(
            static_cast<size_t>(n_distinct16) * cap);
        if (distinct_cells16 == nullptr) return false;
        std::memset(distinct_cells16, 0,
                    sizeof(DistinctCell16) * static_cast<size_t>(n_distinct16) * cap);
    }

    state->arena            = arena;
    state->table            = ht;
    state->keys_flat        = keys_flat;
    state->accums           = accums;
    state->counts           = counts;
    state->gid_scratch      = gid_scratch;
    state->bank_acc         = bank_acc;
    state->bank_cnt         = bank_cnt;
    state->distinct_cells   = distinct_cells;
    state->distinct_cells16 = distinct_cells16;
    state->cap              = cap;
    state->n_keys           = n_keys;
    state->n_aggs           = n_aggs;
    state->entry_count      = 0;
    state->n_rows_in        = 0;
    state->grow_cap         = (grow_cap > kGbEntryCap)
        ? kGbEntryCap : static_cast<uint32_t>(grow_cap);
    state->oom              = false;
    for (uint8_t k = 0; k < n_keys; ++k) {        // Card S: spilled-key buffers
        state->key_str_buf[k]  = nullptr;
        state->key_str_used[k] = 0;
        state->key_str_cap[k]  = 0;
    }
    return true;
}

// W-P: enable the dense gid index AFTER begin and BEFORE the first
// ingest (see the state-field note). Single int-like key only; `range`
// = key_max - key_min + 1, certified by the caller from load-time
// column stats. Index cost = 4 bytes/slot; the ceiling bounds it at
// 64 MiB. Returns false (dense stays off, hash path unchanged) on any
// ineligible shape.
inline bool groupby_typed_enable_dense(GroupbyTypedState* state,
                                       int64_t key_min,
                                       uint64_t range) noexcept {
    assert(state != nullptr);
    assert(state->entry_count == 0 && "enable dense before the first ingest");
    if (state == nullptr || state->arena == nullptr) return false;
    if (state->n_keys != 1 || range == 0) return false;
    if (range > (1ull << 24)) return false;        // 64 MiB index ceiling
    const BoltType kt = state->key_types[0];
    if (kt != BoltType::Int64 && kt != BoltType::Int32 &&
        kt != BoltType::Date32) return false;
    auto* idx =
        state->arena->allocate_array<int32_t>(static_cast<size_t>(range));
    if (idx == nullptr) return false;
    std::memset(idx, 0xFF, sizeof(int32_t) * static_cast<size_t>(range));
    state->dense_gid   = idx;
    state->dense_min   = key_min;
    state->dense_range = static_cast<uint32_t>(range);
    return true;
}

// Card S: append `len` spilled key bytes to key column k's deep-copy buffer,
// growing geometrically in the arena (one realloc-and-copy per doubling; the
// arena reclaims the old buffer at reset). Writes the byte offset to *out_off.
// Returns false on OOM / 32-bit-offset overflow. Tiger Style: bounded, asserts.
BOLT_FORCE_INLINE bool gb_key_str_append(GroupbyTypedState* state, uint32_t k,
                                         const char* bytes, uint32_t len,
                                         uint32_t* out_off) noexcept {
    assert(state != nullptr && out_off != nullptr);
    assert(bytes != nullptr || len == 0);
    if (state->key_str_used[k] + len > state->key_str_cap[k]) {
        size_t newcap = (state->key_str_cap[k] != 0) ? state->key_str_cap[k] * 2u
                                                      : size_t{4096};
        while (newcap < state->key_str_used[k] + len) newcap *= 2u;
        char* nb = static_cast<char*>(state->arena->allocate(newcap, 16));
        if (nb == nullptr) return false;
        if (state->key_str_used[k] > 0) {
            std::memcpy(nb, state->key_str_buf[k], state->key_str_used[k]);
        }
        state->key_str_buf[k] = nb;
        state->key_str_cap[k] = newcap;
    }
    if (state->key_str_used[k] > 0xFFFFFFFFull) return false;
    *out_off = static_cast<uint32_t>(state->key_str_used[k]);
    std::memcpy(state->key_str_buf[k] + state->key_str_used[k], bytes, len);
    state->key_str_used[k] += len;
    return true;
}

// K-AGG-C: the two-pass specialised ingest (pass-1 gid materialisation +
// pass-2 columnar accumulate). Kept in an .inc so this header stays
// navigable; included only here.
#include "bolt/join/bolt_groupby_twopass.inc"

// K-AGG-C fallback: the original per-row ingest loop, VERBATIM in
// behaviour, operating on the window [start, start+count) of the
// iteration space (dense rows when sel == nullptr, else sel positions).
// This stays the path for spilled Utf8 keys, DISTINCT aggregates,
// >2 keys, and exotic key types/formats. The one addition is
// canonicalise-at-insert for inline Utf8 stored cells (padding bytes
// zeroed) so the specialised path's raw-u128 equality agrees with this
// path's content equality across windows.
inline void gb_ingest_fallback(
        GroupbyTypedState* state,
        const BoltColumn*  keys,
        const BoltColumn*  payload,
        const uint32_t*    sel,
        int64_t            start,
        int64_t            count) noexcept {
    assert(state != nullptr);
    assert(count >= 0);
    const uint32_t n_keys = state->n_keys;
    const uint32_t n_aggs = state->n_aggs;
    const uint32_t cap    = state->cap;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = (sel != nullptr)
            ? static_cast<int64_t>(sel[start + i]) : (start + i);
        const uint64_t h = gb_detail::hash_keys(keys, n_keys, r);
        int32_t found = state->table->find(h);
        if (found >= 0 && !gb_detail::keys_equal(keys, n_keys, r,
                                                 state->keys_flat,
                                                 static_cast<uint32_t>(found),
                                                 state->key_str_buf)) {
            found = -1;
        }
        uint32_t slot;
        if (found >= 0) {
            slot = static_cast<uint32_t>(found);
        } else {
            if (state->entry_count >= cap) { state->oom = true; return; }
            slot = state->entry_count++;
            if (!state->table->insert(h, slot)) { state->oom = true; return; }
            GbCell16* krow = state->keys_flat + static_cast<size_t>(slot) * n_keys;
            for (uint32_t k = 0; k < n_keys; ++k) {
                krow[k] = gb_detail::read_cell16(keys[k], r);
                // Card S: deep-copy a spilled (>12-char) Utf8 key so the stored
                // group representative owns its bytes — the input overflow
                // buffer is freed after this morsel. hash/equal + finalize then
                // resolve via the groupby's own per-key buffer.
                if (keys[k].type == BoltType::Utf8) {
                    StringView* sv = reinterpret_cast<StringView*>(&krow[k]);
                    if (sv->length > 12u) {
                        const char* sb = kernels::utf8::sv_bytes(*sv,
                            static_cast<const char*>(keys[k].str_overflow_base));
                        uint32_t off = 0;
                        if (!gb_key_str_append(state, k, sb, sv->length, &off)) {
                            state->oom = true;
                            return;
                        }
                        sv->ref.buf_idx = 0;
                        sv->ref.offset  = off;
                    } else {
                        // K-AGG-C: canonicalise inline padding (see above).
                        gb_twopass::gb_sv_canon_inline(&krow[k]);
                    }
                }
            }
            for (uint32_t j = 0; j < n_aggs; ++j) {
                const size_t off = static_cast<size_t>(j) * cap + slot;
                state->accums[off] =
                    gb_detail::agg_identity(state->specs[j].kind,
                                            state->agg_in_types[j]);
                state->counts[off] = 0;
            }
        }
        for (uint32_t j = 0; j < n_aggs; ++j) {
            GbCell16 v{0, 0};
            bool valid = true;
            if (state->specs[j].kind != AggKind::CountStar) {
                const BoltColumn& pc = payload[state->specs[j].in_col];
                v     = gb_detail::read_cell16(pc, r);
                valid = gb_detail::cell_valid(pc, r);
            }
            if (valid && state->specs[j].distinct != 0) {
                const uint16_t i16 = state->distinct16_idx[j];
                if (i16 != 0xFFFFu) {
                    const size_t doff = static_cast<size_t>(i16) * cap + slot;
                    std::uint8_t buf[16];
                    std::memcpy(buf, &v, 16);
                    if (!state->distinct_cells16[doff].insert(buf)) continue;
                } else {
                    const uint16_t i8 = state->distinct_idx[j];
                    assert(i8 != 0xFFFFu);
                    const size_t doff = static_cast<size_t>(i8) * cap + slot;
                    if (!state->distinct_cells[doff].insert(v.a)) continue;
                }
            }
            const size_t off = static_cast<size_t>(j) * cap + slot;
            gb_detail::apply(state->specs[j].kind, state->agg_in_types[j],
                             &state->accums[off], v, valid);
            state->counts[off] += (valid ? 1 : 0);
        }
    }
}

// Ingest one morsel. `keys` / `payload` are the live column pointers for
// THIS morsel - types must match descriptors passed to `_begin`. When
// `sel != nullptr` the morsel is sparse and the kernel iterates over
// `sel[0..sel_len]` — even when sel_len == 0 (a filter that matched
// nothing in THIS morsel; the old `&& sel_len > 0` misread that as dense
// and aggregated unfiltered rows — fuzz_vs_duckdb --tpch seed 3).
//
// K-AGG-C: rows are processed in <=kGbIngestWindow windows; each window
// classifies its key shape ONCE and runs the two-pass specialised path
// (gid materialisation + per-(kind,type) columnar accumulate) or the
// verbatim per-row fallback. Hot-path allocation-free. Sets
// `state->oom = true` if the SwissTable hits cap; caller checks.
// Hash one STORED key row of `s` — byte-exact with gb_detail::hash_keys over
// the same content (stored cells ARE read_cell16 results; inline Utf8 cells
// are canonicalised at insert, and content hashing reads only length bytes).
// Used by gb_grow's rehash and the partial-merge section below.
inline uint64_t gb_merge_hash_stored(const GroupbyTypedState* s,
                                     uint32_t gid) noexcept {
    assert(s != nullptr);
    assert(gid < s->entry_count);
    const uint32_t n_keys = s->n_keys;
    const GbCell16* row = s->keys_flat + static_cast<size_t>(gid) * n_keys;
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (uint32_t k = 0; k < n_keys; ++k) {
        if (s->key_types[k] == BoltType::Utf8) {
            StringView sv;
            std::memcpy(&sv, &row[k], 16);
            h = swiss_mix_wyhash3(
                h ^ gb_detail::hash_utf8_content(sv, s->key_str_buf[k]));
            continue;
        }
        h ^= swiss_mix_wyhash3(static_cast<uint64_t>(row[k].a));
        h  = swiss_mix_wyhash3(h ^ static_cast<uint64_t>(row[k].b));
    }
    return h;
}

// W-P geometric growth: quadruple the group table + cap-strided arrays and
// rehash. Opt-in via state->grow_cap (legacy fail-fast-at-cap when 0 / <=
// cap — partition-driver worker clamps depend on it). Gids are PRESERVED
// (entries copy slot-for-slot; only the SwissTable h->gid mapping
// rebuilds), so the direct-mapped gid cache and every outstanding gid stay
// valid. Spilled Utf8 key bytes live in key_str_buf (offset-stable) and
// are NOT touched. Runs only at ingest-window / merge-insert boundaries —
// never mid-window (the two-pass path caches array pointers per window).
inline bool gb_grow(GroupbyTypedState* st) noexcept {
    assert(st != nullptr);
    if (st->grow_cap <= st->cap || st->oom) return false;
    Arena* arena = st->arena;
    if (arena == nullptr) return false;
    uint64_t want = static_cast<uint64_t>(st->cap) * 4u;
    if (want > st->grow_cap) want = st->grow_cap;
    if (want > kGbEntryCap)  want = kGbEntryCap;
    if (want <= st->cap) return false;
    auto* nt = static_cast<SwissTable*>(
        arena->allocate(sizeof(SwissTable), alignof(SwissTable)));
    if (nt == nullptr) return false;
    if (!SwissTable::create_with(nt, want, arena, /*tight=*/true)) return false;
    const uint32_t ncap = nt->capacity;
    assert(ncap > st->cap && ncap <= kGbEntryCap);
    const uint32_t n_keys = st->n_keys;
    const uint32_t n_aggs = st->n_aggs;
    GbCell16* nkeys = arena->allocate_array<GbCell16>(
        static_cast<size_t>(ncap) * n_keys);
    GbCell16* nacc  = arena->allocate_array<GbCell16>(
        static_cast<size_t>(ncap) * n_aggs);
    int64_t*  ncnt  = arena->allocate_array<int64_t>(
        static_cast<size_t>(ncap) * n_aggs);
    if (nkeys == nullptr || nacc == nullptr || ncnt == nullptr) return false;
    std::memset(ncnt, 0, sizeof(int64_t) * static_cast<size_t>(ncap) * n_aggs);
    DistinctCell*   ndc   = nullptr;
    DistinctCell16* ndc16 = nullptr;
    if (st->n_distinct > 0) {
        ndc = arena->allocate_array<DistinctCell>(
            static_cast<size_t>(st->n_distinct) * ncap);
        if (ndc == nullptr) return false;
        std::memset(ndc, 0,
                    sizeof(DistinctCell) * static_cast<size_t>(st->n_distinct) * ncap);
    }
    if (st->n_distinct16 > 0) {
        ndc16 = arena->allocate_array<DistinctCell16>(
            static_cast<size_t>(st->n_distinct16) * ncap);
        if (ndc16 == nullptr) return false;
        std::memset(ndc16, 0,
                    sizeof(DistinctCell16) *
                        static_cast<size_t>(st->n_distinct16) * ncap);
    }
    // Re-stride: keys_flat is gid-major (verbatim block); accums/counts/
    // distinct cells are agg-major with a CAP stride (per-lane copy).
    const uint32_t n = st->entry_count;
    std::memcpy(nkeys, st->keys_flat,
                static_cast<size_t>(n) * n_keys * sizeof(GbCell16));
    for (uint32_t j = 0; j < n_aggs; ++j) {
        std::memcpy(nacc + static_cast<size_t>(j) * ncap,
                    st->accums + static_cast<size_t>(j) * st->cap,
                    static_cast<size_t>(n) * sizeof(GbCell16));
        std::memcpy(ncnt + static_cast<size_t>(j) * ncap,
                    st->counts + static_cast<size_t>(j) * st->cap,
                    static_cast<size_t>(n) * sizeof(int64_t));
    }
    for (uint16_t d = 0; d < st->n_distinct; ++d) {
        std::memcpy(ndc + static_cast<size_t>(d) * ncap,
                    st->distinct_cells + static_cast<size_t>(d) * st->cap,
                    static_cast<size_t>(n) * sizeof(DistinctCell));
    }
    for (uint16_t d = 0; d < st->n_distinct16; ++d) {
        std::memcpy(ndc16 + static_cast<size_t>(d) * ncap,
                    st->distinct_cells16 + static_cast<size_t>(d) * st->cap,
                    static_cast<size_t>(n) * sizeof(DistinctCell16));
    }
    st->table     = nt;
    st->keys_flat = nkeys;
    st->accums    = nacc;
    st->counts    = ncnt;
    if (ndc   != nullptr) st->distinct_cells   = ndc;
    if (ndc16 != nullptr) st->distinct_cells16 = ndc16;
    st->cap       = ncap;
    for (uint32_t g = 0; g < n; ++g) {
        const bool ok = nt->insert(gb_merge_hash_stored(st, g), g);
        assert(ok && "grow rehash cannot overflow a strictly larger table");
        if (!ok) { st->oom = true; return false; }
    }
    return true;
}

inline void groupby_agg_multi_key_typed_ingest(
        GroupbyTypedState* state,
        const BoltColumn*  keys,
        const BoltColumn*  payload,
        const uint32_t*    sel,
        uint32_t           sel_len,
        int64_t            n_rows) noexcept {
    assert(state != nullptr);
    assert(state->table != nullptr);
    if (state->oom) return;
    assert(state->n_keys >= 1 && state->n_keys <= kGbMaxKeys);
    assert(state->n_aggs >= 1 && state->n_aggs <= kGbMaxAggs);

    const bool sparse = (sel != nullptr);
    const int64_t total = sparse ? static_cast<int64_t>(sel_len) : n_rows;
    for (int64_t start = 0; start < total;
         start += static_cast<int64_t>(kGbIngestWindow)) {
        int64_t count = total - start;
        if (count > static_cast<int64_t>(kGbIngestWindow)) {
            count = static_cast<int64_t>(kGbIngestWindow);
        }
        // W-P growth: guarantee this window cannot hit cap mid-flight
        // (worst case every row is a new group; the two-pass path caches
        // gid/array pointers per window, so growth must happen HERE, at
        // the boundary). Trigger at 3/4 LOAD, not at cap: Swiss probe
        // chains degrade sharply past ~75-85% occupancy (measured: the
        // at-cap trigger ran Q18's 1.5M-group agg at ~97% load between
        // growths, +40% wall). Legacy tight tables carry 1.5x headroom
        // (~66% max load) — 3/4 keeps growth-mode probe costs comparable.
        // grow failure falls through to legacy oom-at-cap in the window.
        while (state->grow_cap > state->cap &&
               static_cast<int64_t>(state->entry_count) + count >
                   static_cast<int64_t>(state->cap / 4u * 3u)) {
            if (!gb_grow(state)) break;
        }
#ifndef BOLT_GROUPBY_TWOPASS_DISABLE
        if (!gb_twopass::gb_try_twopass(state, keys, payload,
                                        sparse ? sel : nullptr,
                                        start, count)) {
            gb_ingest_fallback(state, keys, payload,
                               sparse ? sel : nullptr, start, count);
        }
#else
        gb_ingest_fallback(state, keys, payload,
                           sparse ? sel : nullptr, start, count);
#endif
        if (state->oom) return;
    }
    state->n_rows_in += static_cast<uint32_t>(total);
}

// Emit accumulated groups into typed output columns. Allocates output
// buffers from `state->arena`. After `_finalize` the state is consumed.
inline bool groupby_agg_multi_key_typed_finalize(
        GroupbyTypedState* state,
        BoltColumn*        out_keys,
        BoltColumn*        out_aggs,
        uint32_t*          ngroups_out) noexcept {
    assert(state != nullptr);
    assert(ngroups_out != nullptr);
    if (state->oom) return false;
    const uint8_t  n_keys = state->n_keys;
    const uint8_t  n_aggs = state->n_aggs;
    const uint32_t cap    = state->cap;
    const int64_t  out_n  = static_cast<int64_t>(state->entry_count);
    Arena* arena = state->arena;
    assert(arena != nullptr);
    assert(n_keys >= 1 && n_aggs >= 1);

    for (uint8_t k = 0; k < n_keys; ++k) {
        out_keys[k] = BoltColumn::make_flat_alloc(out_n, state->key_types[k], arena);
        out_keys[k].decimal_scale = state->key_scales[k];
        // Card S: spilled Utf8 group keys point into our deep-copy buffer; hand
        // the base to the output column so the sink/sort resolves them (nullptr
        // for non-Utf8 / inline-only keys, which need no base).
        out_keys[k].str_overflow_base = state->key_str_buf[k];
        if (out_n > 0 && out_keys[k].data == nullptr) return false;
    }
    for (uint8_t j = 0; j < n_aggs; ++j) {
        const BoltType ot = gb_detail::out_type(state->specs[j].kind,
                                                state->agg_in_types[j]);
        out_aggs[j] = BoltColumn::make_flat_alloc(out_n, ot, arena);
        if (ot == BoltType::Decimal128 || ot == BoltType::Decimal64) {
            out_aggs[j].decimal_scale = (state->specs[j].kind == AggKind::Avg)
                ? static_cast<uint8_t>(state->agg_in_scales[j] + 4)
                : state->agg_in_scales[j];
        }
        if (out_n > 0 && out_aggs[j].data == nullptr) return false;
    }
    for (int64_t r = 0; r < out_n; ++r) {
        const GbCell16* krow = state->keys_flat + static_cast<size_t>(r) * n_keys;
        for (uint8_t k = 0; k < n_keys; ++k) {
            void* buf = out_keys[k].data;
            switch (state->key_types[k]) {
                case BoltType::Int64:
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
                case BoltType::Int32:
                    static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(krow[k].a); break;
                case BoltType::Date32:
                    static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(krow[k].a); break;
                case BoltType::Decimal128:
                    std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r],
                                &krow[k], 16); break;
                case BoltType::Decimal64:   // W-DEC: cell.a IS the mantissa
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
                case BoltType::Utf8:
                    std::memcpy(&static_cast<StringView*>(buf)[r], &krow[k], 16); break;
                default:
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
            }
        }
    }
    for (uint8_t j = 0; j < n_aggs; ++j) {
        const BoltType ot = gb_detail::out_type(state->specs[j].kind,
                                                state->agg_in_types[j]);
        void* buf = out_aggs[j].data;
        for (int64_t r = 0; r < out_n; ++r) {
            const size_t off = static_cast<size_t>(j) * cap + static_cast<uint32_t>(r);
            const GbCell16 acc = state->accums[off];
            const int64_t  cnt = state->counts[off];
            // W-DEC inc 4: lane=1 caller contract (planner-proved int64
            // fit). Debug-only: the d128 total must be a pure
            // sign-extension of its low limb. Release never verifies.
            assert(state->specs[j].lane != 1 ||
                   state->agg_in_types[j] != BoltType::Decimal64 ||
                   (state->specs[j].kind != AggKind::Sum &&
                    state->specs[j].kind != AggKind::Avg) ||
                   acc.b == (acc.a >> 63));
            if (state->specs[j].kind == AggKind::CountStar ||
                state->specs[j].kind == AggKind::Count) {
                static_cast<int64_t*>(buf)[r] = acc.a;
                continue;
            }
            if (state->specs[j].kind == AggKind::Avg) {
                namespace dec = kernels::decimal;
                if (ot == BoltType::Decimal128) {
                    dec::Decimal128 s; std::memcpy(&s, &acc, 16);
                    const uint8_t in_s  = state->agg_in_scales[j];
                    const uint8_t out_s = static_cast<uint8_t>(in_s + 4);
                    const dec::Decimal128 num = dec::d128_rescale(s, in_s, out_s);
                    const dec::Decimal128 q   = (cnt > 0)
                        ? dec::d128_div(num, dec::d128_from_i64(cnt))
                        : dec::Decimal128{0, 0};
                    std::memcpy(&static_cast<dec::Decimal128*>(buf)[r], &q, 16);
                } else {
                    const double d = (cnt > 0)
                        ? (static_cast<double>(acc.a) / static_cast<double>(cnt))
                        : 0.0;
                    static_cast<double*>(buf)[r] = d;
                }
                continue;
            }
            if (ot == BoltType::Decimal128) {
                std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r], &acc, 16);
            } else if (ot == BoltType::Utf8) {
                std::memcpy(&static_cast<StringView*>(buf)[r], &acc, 16);
            } else if (ot == BoltType::Date32) {
                static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(acc.a);
            } else {
                static_cast<int64_t*>(buf)[r] = acc.a;
            }
        }
    }
    *ngroups_out = state->entry_count;
    return true;
}

// Partial-state merge kernels (W4 serial fold + W-J5 class/list merges) —
// split out for file size. Single inclusion point, inside namespace bolt.
#include "bolt_groupby_merge.inc"


inline bool groupby_agg_multi_key_typed(
        const BoltColumn* keys,    size_t n_keys,
        const BoltColumn* payload, size_t n_payload,
        const AggSpec*    aggs,    size_t n_aggs,
        int64_t           n_rows,
        BoltColumn*       out_keys,
        BoltColumn*       out_aggs,
        uint32_t*         ngroups_out,
        Arena*            arena,
        uint64_t          expected_groups_hint = 0) noexcept {
    assert(arena != nullptr);
    assert(ngroups_out != nullptr);
    assert(out_keys != nullptr || n_keys == 0);
    assert(out_aggs != nullptr || n_aggs == 0);
    if (n_keys == 0 || n_keys > kGbMaxKeys) return false;
    if (n_aggs == 0 || n_aggs > kGbMaxAggs) return false;
    if (n_rows < 0) return false;
    assert(keys != nullptr);
    assert(aggs != nullptr);
    (void)n_payload;
    *ngroups_out = 0;

    // Capture per-key input types (used to size each output_keys column).
    BoltType key_types[kGbMaxKeys];
    uint8_t  key_scales[kGbMaxKeys];
    for (uint32_t k = 0; k < n_keys; ++k) {
        key_types[k]  = keys[k].type;
        key_scales[k] = keys[k].decimal_scale;
        if (keys[k].length != n_rows) return false;
    }
    // Capture per-agg input types from payload (ignored for CountStar).
    BoltType agg_in_types[kGbMaxAggs];
    uint8_t  agg_in_scales[kGbMaxAggs];
    for (uint32_t j = 0; j < n_aggs; ++j) {
        if (aggs[j].kind == AggKind::CountStar) {
            agg_in_types[j]  = BoltType::Int64;
            agg_in_scales[j] = 0;
            continue;
        }
        if (aggs[j].in_col >= n_payload || payload == nullptr) return false;
        const BoltColumn& c = payload[aggs[j].in_col];
        if (c.length != n_rows) return false;
        agg_in_types[j]  = c.type;
        agg_in_scales[j] = c.decimal_scale;
    }

    // ---- Allocate scratch state in the arena ----
    // expected_groups_hint = 0  ⇒ size by n_rows (worst case all-distinct).
    // expected_groups_hint > 0  ⇒ tight-size to power-of-two ≥ hint; the
    //                              kernel asserts if real groups exceed cap.
    const bool   tight = (expected_groups_hint > 0);
    const uint64_t raw = tight ? expected_groups_hint
                               : (n_rows > 0 ? static_cast<uint64_t>(n_rows) : 1ULL);
    SwissTable* ht = static_cast<SwissTable*>(
        arena->allocate(sizeof(SwissTable), alignof(SwissTable)));
    if (ht == nullptr) return false;
    if (!SwissTable::create_with(ht, raw, arena, tight)) return false;
    const uint32_t cap = ht->capacity;
    assert(cap <= kGbEntryCap);   // safety net; bump kGbEntryCap if it fires

    GbCell16* keys_flat = arena->allocate_array<GbCell16>(
        static_cast<size_t>(cap) * n_keys);
    GbCell16* accums    = arena->allocate_array<GbCell16>(
        static_cast<size_t>(cap) * n_aggs);
    int64_t*  counts    = arena->allocate_array<int64_t>(
        static_cast<size_t>(cap) * n_aggs);
    if (keys_flat == nullptr || accums == nullptr || counts == nullptr) return false;

    // K-AGG-A.2 item 2: per-(distinct agg, group) DISTINCT folding cells.
    // Two parallel arrays — 8B cells for non-Utf8 distinct aggs, 16B cells
    // for Utf8 distinct aggs (full StringView dedup). Indexed by a compact
    // remap `distinct_idx[j]` / `distinct16_idx[j]` so total memory is
    // proportional to the count of distinct-flagged aggs, not n_aggs.
    uint16_t distinct_idx[kGbMaxAggs];     // 0xFFFF = "not 8B-distinct"
    uint16_t distinct16_idx[kGbMaxAggs];   // 0xFFFF = "not Utf8-distinct"
    uint16_t n_distinct   = 0;
    uint16_t n_distinct16 = 0;
    bool any_distinct = false;
    for (uint32_t j = 0; j < n_aggs; ++j) {
        distinct_idx[j]   = 0xFFFFu;
        distinct16_idx[j] = 0xFFFFu;
        if (aggs[j].distinct == 0) continue;
        any_distinct = true;
        if (agg_in_types[j] == BoltType::Utf8) distinct16_idx[j] = n_distinct16++;
        else                                   distinct_idx[j]   = n_distinct++;
    }
    DistinctCell*   distinct_cells   = nullptr;
    DistinctCell16* distinct_cells16 = nullptr;
    if (n_distinct > 0) {
        distinct_cells = arena->allocate_array<DistinctCell>(
            static_cast<size_t>(n_distinct) * cap);
        if (distinct_cells == nullptr) return false;
        std::memset(distinct_cells, 0,
                    sizeof(DistinctCell) * static_cast<size_t>(n_distinct) * cap);
    }
    if (n_distinct16 > 0) {
        distinct_cells16 = arena->allocate_array<DistinctCell16>(
            static_cast<size_t>(n_distinct16) * cap);
        if (distinct_cells16 == nullptr) return false;
        std::memset(distinct_cells16, 0,
                    sizeof(DistinctCell16) * static_cast<size_t>(n_distinct16) * cap);
    }
    (void)any_distinct;   // future: zero-overhead skip when no distinct aggs

    // ---- Single-pass probe + accumulate ----
    // K-AGG-A v1: switch-per-cell read; ~20 ns/row at cardinality 1024.
    // The 0.5 ns/row floor requires per-type specialization (Bolt's
    // groupby_agg_int64 hits 0.13 ns/row that way). Specialization is
    // deferred to K-AGG-A.1 — see docs/research/design-log.md.
    uint32_t entry_count = 0;
    // Card S: this one-shot path keeps stored keys pointing into the input
    // columns (alive for the whole call), so a stored key's spilled-byte base
    // IS the input column's base. (Multi-morsel stateful grouping deep-copies
    // instead; see groupby_agg_multi_key_typed_ingest.)
    char* in_bases[kGbMaxKeys];
    for (uint8_t k = 0; k < n_keys; ++k)
        in_bases[k] = static_cast<char*>(keys[k].str_overflow_base);
    for (int64_t r = 0; r < n_rows; ++r) {
        const uint64_t h = gb_detail::hash_keys(keys, static_cast<uint32_t>(n_keys), r);
        int32_t found = ht->find(h);
        if (found >= 0 && !gb_detail::keys_equal(keys, static_cast<uint32_t>(n_keys),
                                                 r, keys_flat,
                                                 static_cast<uint32_t>(found),
                                                 in_bases)) {
            found = -1;   // hash collision on different key — Tiger: drop row
        }
        uint32_t slot;
        if (found >= 0) {
            slot = static_cast<uint32_t>(found);
        } else {
            if (entry_count >= cap) return false;   // hint underestimated
            slot = entry_count++;
            if (!ht->insert(h, slot)) return false;
            GbCell16* krow = keys_flat + static_cast<size_t>(slot) * n_keys;
            for (uint32_t k = 0; k < n_keys; ++k)
                krow[k] = gb_detail::read_cell16(keys[k], r);
            for (uint32_t j = 0; j < n_aggs; ++j) {
                const size_t off = static_cast<size_t>(j) * cap + slot;
                accums[off] = gb_detail::agg_identity(aggs[j].kind, agg_in_types[j]);
                counts[off] = 0;
            }
        }
        for (uint32_t j = 0; j < n_aggs; ++j) {
            GbCell16 v{0, 0};
            bool valid = true;
            if (aggs[j].kind != AggKind::CountStar) {
                const BoltColumn& pc = payload[aggs[j].in_col];
                v     = gb_detail::read_cell16(pc, r);
                valid = gb_detail::cell_valid(pc, r);
            }
            // K-AGG-A.2 item 2: DISTINCT — drop duplicates per (agg, group)
            // before the accumulator sees them. NULLs never enter the
            // dedup set; they short-circuit through the !valid branch in
            // `apply`. Phase-A scope: linear-scan inline cells with hard
            // cap (`k_distinct_cell_cap`); over-cap inserts silently drop.
            if (valid && aggs[j].distinct != 0) {
                const uint16_t i16 = distinct16_idx[j];
                if (i16 != 0xFFFFu) {
                    const size_t doff =
                        static_cast<size_t>(i16) * cap + slot;
                    std::uint8_t buf[16];
                    std::memcpy(buf, &v, 16);
                    if (!distinct_cells16[doff].insert(buf)) continue;
                } else {
                    const uint16_t i8 = distinct_idx[j];
                    assert(i8 != 0xFFFFu);
                    const size_t doff =
                        static_cast<size_t>(i8) * cap + slot;
                    if (!distinct_cells[doff].insert(v.a)) continue;
                }
            }
            const size_t off = static_cast<size_t>(j) * cap + slot;
            gb_detail::apply(aggs[j].kind, agg_in_types[j], &accums[off], v, valid);
            // Avg's row-per-group counter must also skip NULLs (denominator
            // is "rows with non-NULL value", not raw arrival count).
            counts[off] += (valid ? 1 : 0);
        }
    }

    // ---- Emit output columns ----
    const int64_t out_n = static_cast<int64_t>(entry_count);
    for (uint32_t k = 0; k < n_keys; ++k) {
        out_keys[k] = BoltColumn::make_flat_alloc(out_n, key_types[k], arena);
        out_keys[k].decimal_scale = key_scales[k];
        if (out_n > 0 && out_keys[k].data == nullptr) return false;
    }
    for (uint32_t j = 0; j < n_aggs; ++j) {
        const BoltType ot = gb_detail::out_type(aggs[j].kind, agg_in_types[j]);
        out_aggs[j] = BoltColumn::make_flat_alloc(out_n, ot, arena);
        if (ot == BoltType::Decimal128 || ot == BoltType::Decimal64) {
            // AVG widens scale slightly so integer division keeps precision;
            // SUM/MIN/MAX preserve input scale.
            out_aggs[j].decimal_scale = (aggs[j].kind == AggKind::Avg)
                ? static_cast<uint8_t>(agg_in_scales[j] + 4)
                : agg_in_scales[j];
        }
        if (out_n > 0 && out_aggs[j].data == nullptr) return false;
    }

    // ---- Scatter keys ----
    for (int64_t r = 0; r < out_n; ++r) {
        const GbCell16* krow = keys_flat + static_cast<size_t>(r) * n_keys;
        for (uint32_t k = 0; k < n_keys; ++k) {
            void* buf = out_keys[k].data;
            switch (key_types[k]) {
                case BoltType::Int64:
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
                case BoltType::Int32:
                    static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(krow[k].a); break;
                case BoltType::Date32:
                    static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(krow[k].a); break;
                case BoltType::Decimal128:
                    std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r],
                                &krow[k], 16); break;
                case BoltType::Decimal64:   // W-DEC: cell.a IS the mantissa
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
                case BoltType::Utf8:
                    std::memcpy(&static_cast<StringView*>(buf)[r], &krow[k], 16); break;
                default:
                    static_cast<int64_t*>(buf)[r] = krow[k].a; break;
            }
        }
    }

    // ---- Scatter aggs (finalize Avg per cell) ----
    for (uint32_t j = 0; j < n_aggs; ++j) {
        const BoltType ot = gb_detail::out_type(aggs[j].kind, agg_in_types[j]);
        void* buf = out_aggs[j].data;
        for (int64_t r = 0; r < out_n; ++r) {
            const size_t off = static_cast<size_t>(j) * cap + static_cast<uint32_t>(r);
            const GbCell16 acc = accums[off];
            const int64_t  cnt = counts[off];
            if (aggs[j].kind == AggKind::CountStar || aggs[j].kind == AggKind::Count) {
                static_cast<int64_t*>(buf)[r] = acc.a;
                continue;
            }
            if (aggs[j].kind == AggKind::Avg) {
                namespace dec = kernels::decimal;
                if (ot == BoltType::Decimal128) {
                    dec::Decimal128 s; std::memcpy(&s, &acc, 16);
                    const uint8_t in_s  = agg_in_scales[j];
                    const uint8_t out_s = static_cast<uint8_t>(in_s + 4);
                    const dec::Decimal128 num = dec::d128_rescale(s, in_s, out_s);
                    const dec::Decimal128 q   = (cnt > 0)
                        ? dec::d128_div(num, dec::d128_from_i64(cnt))
                        : dec::Decimal128{0, 0};
                    std::memcpy(&static_cast<dec::Decimal128*>(buf)[r], &q, 16);
                } else {
                    const double d = (cnt > 0) ? (static_cast<double>(acc.a) /
                                                  static_cast<double>(cnt)) : 0.0;
                    static_cast<double*>(buf)[r] = d;
                }
                continue;
            }
            // SUM / MIN / MAX — write 16-byte cell or 8-byte int.
            if (ot == BoltType::Decimal128) {
                std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r], &acc, 16);
            } else if (ot == BoltType::Utf8) {
                std::memcpy(&static_cast<StringView*>(buf)[r], &acc, 16);
            } else if (ot == BoltType::Date32) {
                static_cast<int32_t*>(buf)[r] = static_cast<int32_t>(acc.a);
            } else {
                static_cast<int64_t*>(buf)[r] = acc.a;
            }
        }
    }

    *ngroups_out = entry_count;
    return true;
}

}  // namespace bolt
