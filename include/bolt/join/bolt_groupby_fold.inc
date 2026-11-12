// bolt_groupby_fold.inc — Spark product / try_sum / try_avg aggregate cells.
// #included by bolt_groupby.h inside `namespace bolt`, after the pick cells; not
// a standalone header.
//
// A fold cell is the accumulator cell plus its counts[] entry: the number of
// non-NULL values folded, or kGbFoldOverflow once the running value left its
// range (sticky, as Spark's NULL sum buffer is). Rows and partials fold through
// one combine, so a partial merge is the same operation as a row:
//   Product : the double product (cell .a holds its bits); the first value is
//             1.0 * x (Spark's coalesce(product, 1.0) * x).
//   TrySum  : an int64 sum (integral input) leaving int64 range, or a d128 sum
//             (decimal input) leaving 38 digits, overflows. Two in-range d128
//             values never wrap the 128-bit cell (|x| < 10^38 < 2^126).
//   TryAvg  : the TrySum d128 state; the result is Spark's decimal average,
//             sum / count rounded HALF_UP to the output scale, NULL past 38
//             digits (DecimalDivideWithOverflowCheck with nullOnOverflow).

inline constexpr int64_t kGbFoldOverflow = -1;   // counts[]: the sum overflowed

BOLT_FORCE_INLINE bool gb_kind_is_fold(AggKind k) noexcept {
    return k == AggKind::Product || k == AggKind::TrySum || k == AggKind::TryAvg;
}

namespace gb_fold {

namespace u = ::bolt::kernels::u128;   // portable 128-bit (MSVC has no __int128)

// 10^38 - 1: the largest DECIMAL(38) magnitude.
inline u::U128 d38_max() noexcept { return u::sub(u::pow10(38), u::U128{0, 1}); }

// A cell's two's-complement value as sign and magnitude, and back.
BOLT_FORCE_INLINE u::U128 cell_mag(GbCell16 c, bool* neg) noexcept {
    const u::U128 v{static_cast<uint64_t>(c.b), static_cast<uint64_t>(c.a)};
    *neg = (v.hi >> 63) != 0;
    return *neg ? u::negate(v) : v;
}
BOLT_FORCE_INLINE GbCell16 mag_cell(u::U128 m, bool neg) noexcept {
    const u::U128 v = neg ? u::negate(m) : m;
    return GbCell16{static_cast<int64_t>(v.lo), static_cast<int64_t>(v.hi)};
}

}  // namespace gb_fold

// Spark's decimal average: sum (scale in_s) / count rounded HALF_UP to out_s.
// False when the quotient leaves 38 digits (Spark: NULL under try_avg).
inline bool gb_fold_d128_avg(GbCell16 sum, int64_t count, uint8_t in_s, uint8_t out_s, GbCell16* out) noexcept {
    assert(count > 0 && out != nullptr);
    assert(out_s >= in_s && out_s <= 38);
    namespace u = gb_fold::u;
    bool neg = false;
    const u::U128 m = gb_fold::cell_mag(sum, &neg);
    const uint64_t c = static_cast<uint64_t>(count);
    uint64_t rem = 0;
    const u::U128 p = u::pow10(static_cast<uint32_t>(out_s - in_s));
    u::U128 whole{}, frac{};
    if (!u::mul(u::div_small(m, c, &rem), p, &whole)) return false;
    if (!u::mul(u::U128{0, rem}, p, &frac)) return false;
    uint64_t frem = 0;
    u::U128 q = u::add(whole, u::div_small(frac, c, &frem));
    if (frem >= c - frem) q = u::add(q, u::U128{0, 1});   // HALF_UP: away from zero at .5 (2 * frem >= c)
    if (u::less(gb_fold::d38_max(), q)) return false;
    *out = gb_fold::mag_cell(q, neg);
    return true;
}

// Fold one non-NULL value (sc == 1) or one partial (cell s, count sc) into a
// fold cell. Allocation-free; never fails.
inline void gb_fold_combine(AggKind k, BoltType t, GbCell16* d, int64_t* dc, GbCell16 s, int64_t sc) noexcept {
    assert(gb_kind_is_fold(k));
    assert(d != nullptr && dc != nullptr && *dc >= kGbFoldOverflow && sc >= kGbFoldOverflow);
    if (sc == 0 || *dc == kGbFoldOverflow) return;
    if (sc == kGbFoldOverflow) { *dc = kGbFoldOverflow; return; }
    if (k == AggKind::Product) {
        double x, acc;
        std::memcpy(&x, &s.a, 8);
        std::memcpy(&acc, &d->a, 8);
        const double r = (*dc == 0 ? 1.0 : acc) * x;
        std::memcpy(&d->a, &r, 8);
        *dc += sc;
        return;
    }
    if (gb_detail::gb_agg_cell_is_d128(t)) {
        namespace u = gb_fold::u;   // both within +-(10^38 - 1): the 128-bit sum cannot wrap
        const u::U128 r = u::add(u::U128{static_cast<uint64_t>(d->b), static_cast<uint64_t>(d->a)},
                                 u::U128{static_cast<uint64_t>(s.b), static_cast<uint64_t>(s.a)});
        bool neg = false;
        const u::U128 mag = gb_fold::cell_mag(GbCell16{static_cast<int64_t>(r.lo), static_cast<int64_t>(r.hi)}, &neg);
        if (u::less(gb_fold::d38_max(), mag)) { *dc = kGbFoldOverflow; return; }
        *d = GbCell16{static_cast<int64_t>(r.lo), static_cast<int64_t>(r.hi)};
        *dc += sc;
        return;
    }
    int64_t r = 0;
    if (__builtin_add_overflow(d->a, s.a, &r)) { *dc = kGbFoldOverflow; return; }
    d->a = r;
    *dc += sc;
}

// Input row r into fold aggregate j of group `slot`.
inline void gb_fold_row(GroupbyTypedState* st, uint32_t j, uint32_t slot, const BoltColumn* payload,
                        int64_t r) noexcept {
    assert(st != nullptr && payload != nullptr && slot < st->cap);
    const AggSpec& sp = st->specs[j];
    assert(gb_kind_is_fold(sp.kind));
    const BoltColumn& pc = payload[sp.in_col];
    if (!gb_detail::cell_valid(pc, r)) return;
    const size_t off = static_cast<size_t>(j) * st->cap + slot;
    gb_fold_combine(sp.kind, st->agg_in_types[j], &st->accums[off], &st->counts[off],
                    gb_detail::read_cell16(pc, r), 1);
}

template <bool Sparse>
inline void gb_fold_pass2(GroupbyTypedState* st, uint32_t j, const BoltColumn* payload, const uint32_t* sel,
                          int64_t start, int64_t count, const int32_t* gid) noexcept {
    assert(st != nullptr && gid != nullptr);
    assert(!Sparse || sel != nullptr);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = Sparse ? static_cast<int64_t>(sel[start + i]) : (start + i);
        gb_fold_row(st, j, static_cast<uint32_t>(gid[i]), payload, r);
    }
}

// Partial merge of fold aggregate j: src group `sgid` into dst slot `dslot`.
inline void gb_fold_merge(GroupbyTypedState* dst, uint32_t dslot, const GroupbyTypedState* src, uint32_t sgid,
                          uint32_t j) noexcept {
    assert(dst != nullptr && src != nullptr);
    assert(gb_kind_is_fold(dst->specs[j].kind) && src->specs[j].kind == dst->specs[j].kind);
    const size_t doff = static_cast<size_t>(j) * dst->cap + dslot;
    const size_t soff = static_cast<size_t>(j) * src->cap + sgid;
    gb_fold_combine(dst->specs[j].kind, dst->agg_in_types[j], &dst->accums[doff], &dst->counts[doff],
                    src->accums[soff], src->counts[soff]);
}

BOLT_FORCE_INLINE BoltType gb_fold_out_type(AggKind k, BoltType in) noexcept {
    assert(gb_kind_is_fold(k));
    if (k == AggKind::Product) return BoltType::Float64;
    if (k == AggKind::TryAvg || gb_detail::gb_agg_cell_is_d128(in)) return BoltType::Decimal128;
    return BoltType::Int64;
}

// Row r of a fold aggregate's output column `buf` (type ot). A cell whose value
// has no answer (nothing folded, overflow, a TryAvg past 38 digits) writes 0
// and leaves *cnt <= 0: the caller's validity pass reads it as SQL NULL.
inline void gb_fold_emit(AggKind k, BoltType ot, uint8_t in_s, uint8_t out_s, GbCell16 acc, int64_t* cnt,
                         void* buf, int64_t r) noexcept {
    assert(gb_kind_is_fold(k) && cnt != nullptr && buf != nullptr);
    assert(r >= 0);
    GbCell16 v = *cnt > 0 ? acc : GbCell16{0, 0};
    if (k == AggKind::TryAvg && *cnt > 0 && !gb_fold_d128_avg(acc, *cnt, in_s, out_s, &v)) {
        *cnt = kGbFoldOverflow;
        v = GbCell16{0, 0};
    }
    if (ot == BoltType::Decimal128) std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r], &v, 16);
    else if (ot == BoltType::Float64) std::memcpy(&static_cast<double*>(buf)[r], &v.a, 8);
    else static_cast<int64_t*>(buf)[r] = v.a;
}
