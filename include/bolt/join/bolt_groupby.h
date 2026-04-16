// bolt_groupby.h — Two-phase group-by aggregate (Tiger Style).
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
#include "bolt/join/bolt_swiss.h"

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

}  // namespace bolt
