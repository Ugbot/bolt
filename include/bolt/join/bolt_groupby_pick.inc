// bolt_groupby_pick.inc — Spark first / last / any_value / max_by / min_by
// aggregate cells. #included by bolt_groupby.h inside `namespace bolt`, after
// gb_agg_str_append; not a standalone header.
//
// A pick cell is the accumulator cell plus its counts[] entry, which holds the
// cell's set state rather than a count. Rows fold in input order and partials
// merge with dst as the EARLIER partial, the shape Spark 4.0.1 gives them:
//   First : the first row's value (first non-NULL with ignoreNulls); a set dst
//           is kept.
//   Last  : the last row's value (last non-NULL with ignoreNulls); a set src
//           replaces dst.
//   MaxBy / MinBy : the value at the extreme ordering. NULL orderings are
//           skipped; the new row (src partial) replaces the cell unless the kept
//           ordering is strictly better, so the LATER of tied rows wins
//           (MaxMinBy's If(old > new, old, new)). Orderings compare as Spark's:
//           strings by unsigned bytes, NaN equal to NaN and above every double,
//           -0.0 equal to 0.0. The ordering lives in a ByOrd companion slot
//           (AggSpec::aux) that only its MaxBy / MinBy writes.
// Spilled (>12-byte) Utf8 values and orderings are deep-copied into the slot's
// own byte buffer, as Utf8 MIN / MAX winners are (G2FEAT-283).

inline constexpr uint8_t kGbPickIgnoreNulls = 1u;   // AggSpec::flags
inline constexpr int64_t kGbPickUnset = 0;           // counts[]: nothing folded
inline constexpr int64_t kGbPickValue = 1;           //   holds a value
inline constexpr int64_t kGbPickNull  = 2;           //   holds SQL NULL

BOLT_FORCE_INLINE bool gb_kind_is_pick(AggKind k) noexcept {
    return k == AggKind::First || k == AggKind::Last || k == AggKind::MaxBy || k == AggKind::MinBy;
}

BOLT_FORCE_INLINE bool gb_kind_is_by(AggKind k) noexcept {
    return k == AggKind::MaxBy || k == AggKind::MinBy;
}

// The arena-backed buffer a pick cell's spilled Utf8 bytes live in.
struct GbPickBytes {
    char**  buf;
    size_t* used;
    size_t* cap;
    Arena*  arena;
};

inline GbPickBytes gb_pick_state_bytes(GroupbyTypedState* st, uint32_t j) noexcept {
    assert(st != nullptr && j < kGbMaxAggs);
    return GbPickBytes{&st->agg_str_buf[j], &st->agg_str_used[j], &st->agg_str_cap[j], st->arena};
}

// Append `len` bytes, growing geometrically; *off = where they landed.
inline bool gb_pick_bytes_append(GbPickBytes s, const char* p, uint32_t len, uint32_t* off) noexcept {
    assert(s.buf != nullptr && s.used != nullptr && s.cap != nullptr && s.arena != nullptr);
    assert(p != nullptr && off != nullptr);
    if (*s.used + len > *s.cap) {
        size_t ncap = *s.cap != 0 ? *s.cap * 2u : size_t{4096};
        for (int i = 0; i < 40 && ncap < *s.used + len; ++i) ncap *= 2u;
        if (ncap < *s.used + len) return false;
        char* nb = static_cast<char*>(s.arena->allocate(ncap, 16));
        if (nb == nullptr) return false;
        if (*s.used > 0) std::memcpy(nb, *s.buf, *s.used);
        *s.buf = nb;
        *s.cap = ncap;
    }
    if (*s.used > 0xFFFFFFFFull) return false;
    *off = static_cast<uint32_t>(*s.used);
    std::memcpy(*s.buf + *s.used, p, len);
    *s.used += len;
    return true;
}

// *dst = v, owning a spilled Utf8 tail (resolved through `base`) in `s`.
inline bool gb_pick_take(BoltType t, GbCell16* dst, GbCell16 v, const char* base, GbPickBytes s) noexcept {
    assert(dst != nullptr);
    if (!::bolt::gb_detail::gb_bytes(t)) { *dst = v; return true; }
    StringView sv;
    std::memcpy(&sv, &v, sizeof(sv));
    if (sv.length <= 12u) { *dst = v; return true; }
    assert(base != nullptr);
    const char* p = kernels::utf8::sv_bytes(sv, base);
    uint32_t off = 0;
    if (p == nullptr || !gb_pick_bytes_append(s, p, sv.length, &off)) return false;
    sv.ref.buf_idx = 0;
    sv.ref.offset  = off;
    std::memcpy(dst, &sv, sizeof(sv));
    return true;
}

// Spark's ordering of two cells of type `t`: <0, 0, >0.
inline int gb_pick_ord_cmp(BoltType t, GbCell16 a, const char* abase, GbCell16 b, const char* bbase) noexcept {
    if (::bolt::gb_detail::gb_bytes(t)) {
        StringView x, y;
        std::memcpy(&x, &a, sizeof(x));
        std::memcpy(&y, &b, sizeof(y));
        const int c = sv_bytelex_compare(x, abase, y, bbase);
        return (c > 0) - (c < 0);
    }
    if (gb_detail::gb_agg_cell_is_d128(t)) {
        kernels::decimal::Decimal128 x, y;
        std::memcpy(&x, &a, 16);
        std::memcpy(&y, &b, 16);
        const int c = kernels::decimal::d128_cmp(x, y);
        return (c > 0) - (c < 0);
    }
    if (gb_detail::is_float_type(t)) {
        double x, y;
        std::memcpy(&x, &a.a, 8);
        std::memcpy(&y, &b.a, 8);
        const bool xn = x != x, yn = y != y;
        if (xn || yn) return static_cast<int>(xn) - static_cast<int>(yn);
        return (x > y) - (x < y);   // -0.0 == 0.0
    }
    return (a.a > b.a) - (a.a < b.a);
}

// Fold one candidate (a row, or a set src partial) into a First / Last cell.
inline bool gb_pick_fold_fl(AggKind k, bool ignore_nulls, BoltType t, GbCell16* cell, int64_t* state,
                            GbCell16 v, bool valid, const char* base, GbPickBytes s) noexcept {
    assert(k == AggKind::First || k == AggKind::Last);
    assert(cell != nullptr && state != nullptr);
    if (k == AggKind::First && *state != kGbPickUnset) return true;
    if (!valid) {
        if (ignore_nulls) return true;
        *cell = GbCell16{0, 0};
        *state = kGbPickNull;
        return true;
    }
    if (!gb_pick_take(t, cell, v, base, s)) return false;
    *state = kGbPickValue;
    return true;
}

// A MaxBy / MinBy cell: the value slot and its ByOrd companion.
struct GbPickBy {
    GbCell16*   val;
    int64_t*    vstate;
    GbCell16*   ord;
    int64_t*    ostate;
    BoltType    vt;
    BoltType    ot;
    GbPickBytes vbytes;
    GbPickBytes obytes;
};

// Fold one candidate (value v, ordering o) into a MaxBy / MinBy cell.
inline bool gb_pick_fold_by(AggKind k, const GbPickBy& c, GbCell16 v, bool v_valid, const char* vbase,
                            GbCell16 o, bool o_valid, const char* obase) noexcept {
    assert(gb_kind_is_by(k));
    assert(c.val != nullptr && c.ord != nullptr && c.vstate != nullptr && c.ostate != nullptr);
    if (!o_valid) return true;
    if (*c.ostate != kGbPickUnset) {
        const int kept = gb_pick_ord_cmp(c.ot, *c.ord, *c.obytes.buf, o, obase);
        if (k == AggKind::MaxBy ? kept > 0 : kept < 0) return true;
    }
    if (!gb_pick_take(c.ot, c.ord, o, obase, c.obytes)) return false;
    *c.ostate = kGbPickValue;
    if (!v_valid) {
        *c.val = GbCell16{0, 0};
        *c.vstate = kGbPickNull;
        return true;
    }
    if (!gb_pick_take(c.vt, c.val, v, vbase, c.vbytes)) return false;
    *c.vstate = kGbPickValue;
    return true;
}

inline GbPickBy gb_pick_by_cell(GroupbyTypedState* st, uint32_t j, uint32_t slot) noexcept {
    assert(st != nullptr && gb_kind_is_by(st->specs[j].kind));
    const uint32_t aux = st->specs[j].aux;
    assert(aux < st->n_aggs && st->specs[aux].kind == AggKind::ByOrd);
    const size_t vo = static_cast<size_t>(j) * st->cap + slot;
    const size_t oo = static_cast<size_t>(aux) * st->cap + slot;
    return GbPickBy{&st->accums[vo], &st->counts[vo], &st->accums[oo], &st->counts[oo],
                    st->agg_in_types[j], st->agg_in_types[aux],
                    gb_pick_state_bytes(st, j), gb_pick_state_bytes(st, aux)};
}

// Input row r into pick aggregate j of group `slot`. False (oom set) when a
// spilled Utf8 copy cannot be allocated.
inline bool gb_pick_row(GroupbyTypedState* st, uint32_t j, uint32_t slot, const BoltColumn* payload,
                        int64_t r) noexcept {
    assert(st != nullptr && payload != nullptr && slot < st->entry_count);
    const AggSpec& sp = st->specs[j];
    assert(gb_kind_is_pick(sp.kind));
    const BoltColumn& pc = payload[sp.in_col];
    const bool valid = gb_detail::cell_valid(pc, r);
    const GbCell16 v = valid ? gb_detail::read_cell16(pc, r) : GbCell16{0, 0};
    const char* vbase = static_cast<const char*>(pc.str_overflow_base);
    bool ok = true;
    if (!gb_kind_is_by(sp.kind)) {
        const size_t off = static_cast<size_t>(j) * st->cap + slot;
        ok = gb_pick_fold_fl(sp.kind, (sp.flags & kGbPickIgnoreNulls) != 0, st->agg_in_types[j], &st->accums[off],
                             &st->counts[off], v, valid, vbase, gb_pick_state_bytes(st, j));
    } else {
        const BoltColumn& oc = payload[st->specs[sp.aux].in_col];
        const bool o_valid = gb_detail::cell_valid(oc, r);
        const GbCell16 o = o_valid ? gb_detail::read_cell16(oc, r) : GbCell16{0, 0};
        ok = gb_pick_fold_by(sp.kind, gb_pick_by_cell(st, j, slot), v, valid, vbase, o, o_valid,
                             static_cast<const char*>(oc.str_overflow_base));
    }
    if (!ok) st->oom = true;
    return ok;
}

// The window's rows in order, their group ids already resolved (pass 1).
template <bool Sparse>
inline void gb_pick_pass2(GroupbyTypedState* st, uint32_t j, const BoltColumn* payload, const uint32_t* sel,
                          int64_t start, int64_t count, const int32_t* gid) noexcept {
    assert(st != nullptr && gid != nullptr);
    assert(!Sparse || sel != nullptr);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t r = Sparse ? static_cast<int64_t>(sel[start + i]) : (start + i);
        if (!gb_pick_row(st, j, static_cast<uint32_t>(gid[i]), payload, r)) return;
    }
}

// Partial merge of pick aggregate j: src group `sgid` (the LATER partial) into
// dst slot `dslot`. False (dst->oom set) on a failed spilled-Utf8 copy.
inline bool gb_pick_merge(GroupbyTypedState* dst, uint32_t dslot, const GroupbyTypedState* src, uint32_t sgid,
                          uint32_t j) noexcept {
    assert(dst != nullptr && src != nullptr);
    const AggSpec& sp = dst->specs[j];
    assert(gb_kind_is_pick(sp.kind) && src->specs[j].kind == sp.kind);
    const size_t soff = static_cast<size_t>(j) * src->cap + sgid;
    const GbCell16 v = src->accums[soff];
    const bool v_valid = src->counts[soff] == kGbPickValue;
    bool ok = true;
    if (!gb_kind_is_by(sp.kind)) {
        if (src->counts[soff] == kGbPickUnset) return true;
        const size_t doff = static_cast<size_t>(j) * dst->cap + dslot;
        ok = gb_pick_fold_fl(sp.kind, (sp.flags & kGbPickIgnoreNulls) != 0, dst->agg_in_types[j],
                             &dst->accums[doff], &dst->counts[doff], v, v_valid, src->agg_str_buf[j],
                             gb_pick_state_bytes(dst, j));
    } else {
        const size_t sord = static_cast<size_t>(sp.aux) * src->cap + sgid;
        if (src->counts[sord] == kGbPickUnset) return true;
        ok = gb_pick_fold_by(sp.kind, gb_pick_by_cell(dst, j, dslot), v, v_valid, src->agg_str_buf[j],
                             src->accums[sord], true, src->agg_str_buf[sp.aux]);
    }
    if (!ok) dst->oom = true;
    return ok;
}
