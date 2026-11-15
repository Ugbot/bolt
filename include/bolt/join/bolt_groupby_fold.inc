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
//   SumComp : a float64 sum with a Neumaier compensation (.a the sum, .b the compensation, both double bits):
//             a row adds x and a partial adds its sum and its compensation, so any split of the input over
//             partials and any merge order give the answer one fold over the rows does (-1e300, -2.25, 1e300
//             sum to -2.25, never 0). An infinite sum answers itself (neumaier_result).
//   AvgComp : the SumComp state divided by the folded count at the end; a sum that left double range answers
//             +-Inf (Prometheus's incremental mean can recover there; a partial merge cannot).
//   TryAvg  : the TrySum d128 state; the result is Spark's decimal average,
//             sum / count rounded HALF_UP to the output scale, NULL past 38
//             digits (DecimalDivideWithOverflowCheck with nullOnOverflow).

inline constexpr int64_t kGbFoldOverflow = -1;   // counts[]: the sum overflowed

BOLT_FORCE_INLINE bool gb_kind_is_comp(AggKind k) noexcept { return k == AggKind::SumComp || k == AggKind::AvgComp; }

BOLT_FORCE_INLINE bool gb_kind_is_fold(AggKind k) noexcept {
    return k == AggKind::Product || k == AggKind::TrySum || k == AggKind::TryAvg || gb_kind_is_comp(k);
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


// AvgComp cell: while the compensated sum is finite it is (.a sum, .b compensation) and the answer is sum / n. Once a
// partial sum leaves double range the cell turns into Prometheus's incremental mean (.a mean, .b its compensation,
// flagged in counts[]): the mean of finite values is finite, so the answer survives where the sum cannot.
inline constexpr int64_t kGbAvgMeanFlag = int64_t{1} << 62;

namespace gb_avg {

namespace gs = ::bolt::kernels::group_stats;

BOLT_FORCE_INLINE int64_t n_of(int64_t c) noexcept { return c & ~kGbAvgMeanFlag; }
BOLT_FORCE_INLINE bool is_mean(int64_t c) noexcept { return (c & kGbAvgMeanFlag) != 0; }

// A finite-sum cell as the mean state of the same values.
BOLT_FORCE_INLINE void to_mean(GbCell16* c, int64_t* cnt) noexcept {
    double a, b;
    std::memcpy(&a, &c->a, 8); std::memcpy(&b, &c->b, 8);
    const double n = static_cast<double>(n_of(*cnt));
    const double m = a / n, mc = b / n;
    std::memcpy(&c->a, &m, 8); std::memcpy(&c->b, &mc, 8);
    *cnt |= kGbAvgMeanFlag;
}

}  // namespace gb_avg

inline void gb_avg_comp_combine(GbCell16* d, int64_t* dc, GbCell16 s, int64_t sc) noexcept {
    namespace gs = gb_avg::gs;
    assert(d != nullptr && dc != nullptr && sc > 0);
    if (*dc == 0) { *d = s; *dc = sc; return; }
    int64_t sn = sc;
    GbCell16 sv = s;
    const int64_t dn = gb_avg::n_of(*dc);
    if (!gb_avg::is_mean(*dc) && !gb_avg::is_mean(sc)) {   // both sums: add, unless the total leaves double range
        double da, db, sa, sb;
        std::memcpy(&da, &d->a, 8); std::memcpy(&db, &d->b, 8);
        std::memcpy(&sa, &s.a, 8);  std::memcpy(&sb, &s.b, 8);
        double ta = da, tb = db;
        gs::neumaier_add(&ta, &tb, sa);
        tb += sb;
        if (gs::gs_is_finite(ta) && gs::gs_is_finite(tb)) {
            std::memcpy(&d->a, &ta, 8); std::memcpy(&d->b, &tb, 8);
            *dc = dn + gb_avg::n_of(sc);
            return;
        }
    }
    if (!gb_avg::is_mean(*dc)) gb_avg::to_mean(d, dc);
    if (!gb_avg::is_mean(sn)) gb_avg::to_mean(&sv, &sn);
    double dm, dmc, sm, smc;
    std::memcpy(&dm, &d->a, 8); std::memcpy(&dmc, &d->b, 8);
    std::memcpy(&sm, &sv.a, 8); std::memcpy(&smc, &sv.b, 8);
    const double n1 = static_cast<double>(dn), n2 = static_cast<double>(gb_avg::n_of(sn));
    double m, mc;
    if (n2 == 1.0 && smc == 0.0) {
        mc = dmc;   // one more value: the incremental update itself
        m = gs::promql_mean_update(dm, &mc, n1 + 1.0, sm);
    } else if (!gs::gs_is_finite(dm) || !gs::gs_is_finite(sm)) {
        m = dm + sm; mc = 0.0;   // an infinite or NaN mean absorbs the other as a sum does
    } else {
        const double w1 = n1 / (n1 + n2), w2 = n2 / (n1 + n2);
        m = dm * w1; mc = dmc * w1 + smc * w2;
        gs::neumaier_add(&m, &mc, sm * w2);
    }
    std::memcpy(&d->a, &m, 8); std::memcpy(&d->b, &mc, 8);
    *dc = (dn + gb_avg::n_of(sn)) | kGbAvgMeanFlag;
}

// Fold one non-NULL value (sc == 1) or one partial (cell s, count sc) into a
// fold cell. Allocation-free; never fails.
inline void gb_fold_combine(AggKind k, BoltType t, GbCell16* d, int64_t* dc, GbCell16 s, int64_t sc) noexcept {
    assert(gb_kind_is_fold(k));
    assert(d != nullptr && dc != nullptr && *dc >= kGbFoldOverflow && sc >= kGbFoldOverflow);
    if (sc == 0 || *dc == kGbFoldOverflow) return;
    if (sc == kGbFoldOverflow) { *dc = kGbFoldOverflow; return; }
    if (k == AggKind::SumComp) {
        assert(t == BoltType::Float64);
        if (*dc == 0) { *d = s; *dc = sc; return; }   // the first value is the sum: -0.0 stays -0.0
        double ds, dcomp, ss, scomp;
        std::memcpy(&ds, &d->a, 8); std::memcpy(&dcomp, &d->b, 8);
        std::memcpy(&ss, &s.a, 8);  std::memcpy(&scomp, &s.b, 8);
        kernels::group_stats::neumaier_add(&ds, &dcomp, ss);
        dcomp += scomp;
        std::memcpy(&d->a, &ds, 8); std::memcpy(&d->b, &dcomp, 8);
        *dc += sc;
        return;
    }
    if (k == AggKind::AvgComp) {
        assert(t == BoltType::Float64);
        gb_avg_comp_combine(d, dc, s, sc);
        return;
    }
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
    if (k == AggKind::Product || gb_kind_is_comp(k)) return BoltType::Float64;
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
    if (gb_kind_is_comp(k) && *cnt > 0) {
        double sum, comp;
        std::memcpy(&sum, &acc.a, 8); std::memcpy(&comp, &acc.b, 8);
        double r = kernels::group_stats::neumaier_result(sum, comp);
        if (k == AggKind::AvgComp) {
            r = gb_avg::is_mean(*cnt) ? (kernels::group_stats::gs_is_finite(sum) ? sum + comp : sum)
                                      : r / static_cast<double>(gb_avg::n_of(*cnt));
        }
        std::memcpy(&v.a, &r, 8);
    }
    if (k == AggKind::TryAvg && *cnt > 0 && !gb_fold_d128_avg(acc, *cnt, in_s, out_s, &v)) {
        *cnt = kGbFoldOverflow;
        v = GbCell16{0, 0};
    }
    if (ot == BoltType::Decimal128) std::memcpy(&static_cast<kernels::decimal::Decimal128*>(buf)[r], &v, 16);
    else if (ot == BoltType::Float64) std::memcpy(&static_cast<double*>(buf)[r], &v.a, 8);
    else static_cast<int64_t*>(buf)[r] = v.a;
}
