// bolt_setop.h — Two-input sorted-merge set operations.
//
// SQL UNION / INTERSECT / EXCEPT (both DISTINCT and ALL variants) over
// pre-sorted input runs. O(n + m) single-pass two-pointer merge. The
// general-case "general-purpose" implementation is the branchless scalar
// path below — it auto-vectorises well on MSVC/Clang and beats the
// hand-SIMD partition variant on mixed inputs (compress-store path is
// data-dependent and stalls on shuffle dependencies; the scalar path
// hits memory bandwidth on int64).
//
// Tiger Style:
//   - POD + free functions, noexcept everywhere.
//   - No heap; caller owns `out`. Caller pre-sizes `out` to na + nb
//     (worst case: UNION ALL or UNION DISTINCT with no overlap).
//   - ≥2 asserts per function; functions ≤70 lines.
//
// Multiplicity semantics (SQL standard):
//   - UNION ALL:        out = a ++ b, multiplicities sum.
//   - UNION DISTINCT:   out = sort-distinct(a ∪ b).
//   - INTERSECT ALL:    multiplicity(x) = min(mult_a(x), mult_b(x)).
//   - INTERSECT DISTINCT: x ∈ out iff x ∈ a AND x ∈ b (mult = 1).
//   - EXCEPT ALL:       multiplicity(x) = max(mult_a(x) - mult_b(x), 0).
//   - EXCEPT DISTINCT:  x ∈ out iff x ∈ a AND x ∉ b (mult = 1).
//
// Inputs MUST be sorted ascending. DISTINCT outputs are sorted + unique.
// ALL outputs are sorted (interleaved by key) and preserve per-key
// multiplicity.
//
// Companion kernel: bolt::kmerge_i64 for k-way (k > 2) merges. The
// two-input setop kernels here are the dedicated SQL set-op path —
// callers should not route through kmerge for k=2.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace setop {

// ---------------------------------------------------------------------------
// Comparator helper. Returns -1 / 0 / +1.
//
// For arithmetic T this lowers to two `cmp` + `sub`s on x86-64 and a
// `subs` + `csel` on ARM64. No branches.
//
// String / StringView keys: these kernels are arithmetic-only. String
// set ops resolve through a dictionary-encode pass upstream (chukonu
// SortMerge enforcer materialises sort keys into int64 surrogates), so
// the hot kernel stays branchless on POD scalars.
// ---------------------------------------------------------------------------
template <typename T>
BOLT_FORCE_INLINE int setop_cmp3(const T& av, const T& bv) noexcept {
    return (av > bv) - (av < bv);
}

// ---------------------------------------------------------------------------
// UNION ALL — concat then merge by sorted order.
//
// Both inputs are sorted; we interleave so the output stays sorted while
// preserving every duplicate. Equivalent to a two-way k-merge but
// specialised for two inputs to drop the heap and the (input_id, key)
// pair output (set ops only need the keys).
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_union_all_sorted(const T* BOLT_RESTRICT a, int64_t na,
                             const T* BOLT_RESTRICT b, int64_t nb,
                             T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    assert(out != nullptr || (na == 0 && nb == 0));
    int64_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        // cmp <= 0 picks av; cmp > 0 picks bv. Branchless predicated store.
        const bool take_a = (cmp <= 0);
        out[k] = take_a ? av : bv;
        i += static_cast<int64_t>(take_a);
        j += static_cast<int64_t>(!take_a);
        ++k;
    }
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
    return k;
}

// ---------------------------------------------------------------------------
// UNION DISTINCT — sorted merge with branchless duplicate suppression.
//
// Same merge as UNION ALL, except the output cursor only advances when
// the just-stored key differs from the previous output key. The
// speculative-store + masked advance is friendlier to OoO than a
// `continue`-branch.
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_union_distinct_sorted(const T* BOLT_RESTRICT a, int64_t na,
                                  const T* BOLT_RESTRICT b, int64_t nb,
                                  T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    assert(out != nullptr || (na == 0 && nb == 0));
    int64_t i = 0, j = 0, k = 0;
    // `last` carries the previous emitted key in a register so the dedup
    // test doesn't reload `out[k-1]` (serial RAW on stores → loads).
    T last{};
    bool have_last = false;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        const bool take_a = (cmp <= 0);
        const bool take_b = (cmp >= 0);  // tie → consume from BOTH
        const T pick = take_a ? av : bv;
        out[k] = pick;
        const bool is_new = !have_last || setop_cmp3<T>(pick, last) != 0;
        k += static_cast<int64_t>(is_new);
        last = pick;
        have_last = true;
        i += static_cast<int64_t>(take_a);
        j += static_cast<int64_t>(take_b);
    }
    while (i < na) {
        const T v = a[i++];
        const bool is_new = !have_last || setop_cmp3<T>(v, last) != 0;
        out[k] = v;
        k += static_cast<int64_t>(is_new);
        last = v;
        have_last = true;
    }
    while (j < nb) {
        const T v = b[j++];
        const bool is_new = !have_last || setop_cmp3<T>(v, last) != 0;
        out[k] = v;
        k += static_cast<int64_t>(is_new);
        last = v;
        have_last = true;
    }
    return k;
}

// ---------------------------------------------------------------------------
// INTERSECT DISTINCT — elements present in BOTH inputs, deduplicated.
//
// Two-pointer: advance the smaller side; on equality emit (if new) and
// advance both sides past the equal run.
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_intersect_sorted(const T* BOLT_RESTRICT a, int64_t na,
                             const T* BOLT_RESTRICT b, int64_t nb,
                             T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    int64_t i = 0, j = 0, k = 0;
    T last{};
    bool have_last = false;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        // Branchless two-pointer: advance i on cmp<=0, j on cmp>=0 (both on
        // tie). Same {<,=,>} → {i++, both++, j++} dispatch as the 3-way branch,
        // but data-independent. Emit only on a match (cmp==0), deduped against
        // the previous emitted key. The dedup state (`last`/`have_last`) updates
        // only when we emit, exactly as the branched version did.
        const bool match  = (cmp == 0);
        const bool is_new = match && (!have_last || setop_cmp3<T>(av, last) != 0);
        out[k] = av;                       // speculative store; consumed iff is_new
        k += static_cast<int64_t>(is_new);
        last = match ? av : last;
        have_last = have_last | match;
        i += static_cast<int64_t>(cmp <= 0);
        j += static_cast<int64_t>(cmp >= 0);
    }
    return k;
}

// ---------------------------------------------------------------------------
// EXCEPT DISTINCT — elements in a but not in b, deduplicated.
//
// Two-pointer: when a[i] < b[j], emit a[i] (deduped) and advance i. When
// equal, advance both (suppress emit). When a[i] > b[j], advance j.
// Tail of a flushes (still deduped against last emit).
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_except_sorted(const T* BOLT_RESTRICT a, int64_t na,
                          const T* BOLT_RESTRICT b, int64_t nb,
                          T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    int64_t i = 0, j = 0, k = 0;
    T last{};
    bool have_last = false;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        // Branchless: i advances on cmp<=0 (both the emit case cmp<0 and the
        // cancel case cmp==0), j advances only on cmp>0. Emit a[i] (deduped)
        // only on the strict-less case. j is NOT advanced on a tie — matching
        // the original comment: equal a-duplicates must keep cancelling against
        // the same b[j] rather than leaking past as cmp<0 emissions.
        const bool emit   = (cmp < 0);
        const bool is_new = emit && (!have_last || setop_cmp3<T>(av, last) != 0);
        out[k] = av;                       // speculative store; consumed iff is_new
        k += static_cast<int64_t>(is_new);
        last = emit ? av : last;
        have_last = have_last | emit;
        i += static_cast<int64_t>(cmp <= 0);
        j += static_cast<int64_t>(cmp > 0);
    }
    while (i < na) {
        const T v = a[i++];
        const bool is_new = !have_last || setop_cmp3<T>(v, last) != 0;
        out[k] = v;
        k += static_cast<int64_t>(is_new);
        last = v;
        have_last = true;
    }
    return k;
}

// ---------------------------------------------------------------------------
// INTERSECT ALL — per-key multiplicity = min(mult_a, mult_b).
//
// When equal: emit one copy and advance both. Otherwise advance the
// smaller side. Duplicates inside one input only count if the other side
// has a matching duplicate (because that's where the pointers align).
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_intersect_all_sorted(const T* BOLT_RESTRICT a, int64_t na,
                                 const T* BOLT_RESTRICT b, int64_t nb,
                                 T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    int64_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        // Branchless: advance i on cmp<=0, j on cmp>=0 (both on tie). Emit one
        // copy per matched pair (cmp==0). Identical {<,=,>} dispatch as the
        // 3-way branch, data-independent.
        const bool match = (cmp == 0);
        out[k] = av;                       // speculative store; consumed iff match
        k += static_cast<int64_t>(match);
        i += static_cast<int64_t>(cmp <= 0);
        j += static_cast<int64_t>(cmp >= 0);
    }
    return k;
}

// ---------------------------------------------------------------------------
// EXCEPT ALL — per-key multiplicity = max(mult_a - mult_b, 0).
//
// When a[i] < b[j]: emit a[i], advance i.
// When equal:        skip one a, advance both (b cancels one of a).
// When a[i] > b[j]:  advance j.
// Tail of a flushes verbatim.
// ---------------------------------------------------------------------------
template <typename T>
int64_t set_except_all_sorted(const T* BOLT_RESTRICT a, int64_t na,
                              const T* BOLT_RESTRICT b, int64_t nb,
                              T*       BOLT_RESTRICT out) noexcept {
    assert(na >= 0 && nb >= 0);
    assert((na == 0 || a != nullptr) && (nb == 0 || b != nullptr));
    int64_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        const T av = a[i];
        const T bv = b[j];
        const int cmp = setop_cmp3<T>(av, bv);
        // Branchless: i advances on cmp<=0 (emit case cmp<0 + cancel case
        // cmp==0), j advances on cmp>=0 (cancel + skip). Emit a[i] only on the
        // strict-less case. Same dispatch as the 3-way branch, data-independent.
        const bool emit = (cmp < 0);
        out[k] = av;                       // speculative store; consumed iff emit
        k += static_cast<int64_t>(emit);
        i += static_cast<int64_t>(cmp <= 0);
        j += static_cast<int64_t>(cmp >= 0);
    }
    while (i < na) out[k++] = a[i++];
    return k;
}

}  // namespace setop
}  // namespace kernels
}  // namespace bolt
