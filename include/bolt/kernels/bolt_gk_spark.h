// bolt_gk_spark.h -- Spark 4.0.1 QuantileSummaries (percentile_approx /
// approx_percentile), bit for bit: insert through the 50000-value head buffer,
// compress at 10000 samples, merge of two compressed summaries, the query over
// sorted percentiles, and ApproximatePercentile's serialized form.
//
// The head buffer sorts as java.util.Arrays.sort(double[]): -0.0 before 0.0,
// every NaN last. Comparisons inside insert / merge are IEEE, as in Scala.
// No heap: the caller supplies every buffer. Bounded loops, noexcept.
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_argsort.h"

namespace bolt {
namespace kernels {
namespace stat_agg {

struct GkStat {
    double       v;
    std::int64_t g;
    std::int64_t delta;
};

struct GkWork {
    GkStat*        a;           // two sample arrays, each `cap` entries
    GkStat*        b;
    std::uint32_t  cap;         // >= threshold + head_size
    std::uint32_t  head_size;   // Spark defaultHeadSize (50000)
    std::uint32_t  threshold;   // Spark defaultCompressThreshold (10000)
    double*        head;        // head_size values
    std::int64_t*  key;         // head_size
    std::int32_t*  perm;        // head_size
    std::int32_t*  tmp;         // head_size
};

constexpr std::uint32_t k_gk_head_size = 50000;
constexpr std::uint32_t k_gk_compress_threshold = 10000;

// A compressed summary (no head buffer): `len` samples at `cur`.
struct GkState {
    GkStat*        cur;
    std::uint32_t  len;
    std::int64_t   count;
    double         rel_err;
    std::int32_t   threshold;   // compressThreshold; 0 reads as the default
};

// java.util.Arrays.sort(double[]) order as a signed key.
BOLT_FORCE_INLINE std::int64_t gk_java_sort_key(double v) noexcept {
    if (std::isnan(v)) return INT64_MAX;
    std::uint64_t u = 0;
    std::memcpy(&u, &v, 8);
    const std::uint64_t mask = (static_cast<std::uint64_t>(-static_cast<std::int64_t>(u >> 63))) &
                               0x7FFFFFFFFFFFFFFFULL;
    return static_cast<std::int64_t>(u ^ mask);
}

// Sort the head buffer and insert it into the samples (withHeadBufferInserted).
// False when the samples would pass w->cap (compression need not shrink them:
// with 2 * rel_err * count < 1 nothing merges, so up to `count` samples stay).
inline bool gk_flush(GkWork* w, GkState* s, std::uint32_t h) noexcept {
    assert(w != nullptr && s != nullptr);
    assert(h <= w->head_size);
    if (h == 0) return true;
    if (static_cast<std::uint64_t>(s->len) + h > w->cap) return false;
    for (std::uint32_t i = 0; i < h; ++i) w->key[i] = gk_java_sort_key(w->head[i]);
    if (h > 1) sort_indirect_i64_scratch(w->perm, w->key, w->tmp, h, /*ascending=*/true);
    else w->perm[0] = 0;
    GkStat* out = (s->cur == w->a) ? w->b : w->a;
    std::uint32_t o = 0, si = 0;
    std::int64_t cc = s->count;
    for (std::uint32_t oi = 0; oi < h; ++oi) {
        const double cs = w->head[w->perm[oi]];
        while (si < s->len && s->cur[si].v <= cs) out[o++] = s->cur[si++];
        ++cc;
        const bool edge = (o == 0) || (si == s->len && oi == h - 1);
        const std::int64_t delta =
            edge ? 0 : static_cast<std::int64_t>(std::floor(2.0 * s->rel_err * static_cast<double>(cc)));
        out[o++] = GkStat{cs, 1, delta};
        assert(o <= w->cap);
    }
    while (si < s->len) out[o++] = s->cur[si++];
    assert(o <= w->cap);
    s->cur = out;
    s->len = o;
    s->count = cc;
    return true;
}

// compressImmut: merge neighbours (back to front) while g + g' + delta' < threshold.
// Writes at most `len` samples to `out` (disjoint from `in`); returns the count.
inline std::uint32_t gk_compress_into(const GkStat* in, std::uint32_t len, double merge_threshold,
                                      GkStat* out) noexcept {
    assert(in != nullptr && out != nullptr && in != out);
    if (len == 0) return 0;
    std::uint32_t o = 0;            // filled in reverse, then reversed in place
    GkStat head = in[len - 1];
    for (std::int64_t i = static_cast<std::int64_t>(len) - 2; i >= 1; --i) {
        const GkStat s1 = in[i];
        if (static_cast<double>(s1.g + head.g + head.delta) < merge_threshold) {
            head.g += s1.g;
        } else {
            out[o++] = head;
            head = s1;
        }
    }
    out[o++] = head;
    if (in[0].v <= head.v && len > 1) out[o++] = in[0];
    for (std::uint32_t i = 0, j = o - 1; i < j; ++i, --j) { const GkStat t = out[i]; out[i] = out[j]; out[j] = t; }
    assert(o <= len);
    return o;
}

inline void gk_compress(GkWork* w, GkState* s) noexcept {
    assert(w != nullptr && s != nullptr);
    if (s->len == 0) return;
    GkStat* out = (s->cur == w->a) ? w->b : w->a;
    s->len = gk_compress_into(s->cur, s->len, 2.0 * s->rel_err * static_cast<double>(s->count), out);
    s->cur = out;
}

// Feed `n` values (arrival order) into a fresh summary, then compress (the
// serialized partial). Returns false when `w` is too small: cap must hold
// threshold + head_size and, when compression cannot merge, every value.
inline bool gk_build(const double* BOLT_RESTRICT vals, std::int64_t n, double rel_err,
                     GkWork* w, GkState* s) noexcept {
    assert(w != nullptr && s != nullptr);
    assert(vals != nullptr || n == 0);
    if (w->cap < w->threshold + w->head_size) return false;
    s->cur = w->a; s->len = 0; s->count = 0; s->rel_err = rel_err;
    s->threshold = static_cast<std::int32_t>(w->threshold);
    std::uint32_t h = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        w->head[h++] = vals[i];
        if (h >= w->head_size) {
            if (!gk_flush(w, s, h)) return false;
            h = 0;
            if (s->len >= w->threshold) gk_compress(w, s);
        }
    }
    if (!gk_flush(w, s, h)) return false;
    gk_compress(w, s);
    return true;
}

// QuantileSummaries.merge(self = a, other = b) into `out` (a.len + b.len
// entries) using `tmp` (same size) for the interleave. `out` may be a's or b's
// samples (both are read into `tmp` first), never `tmp`.
inline void gk_merge(const GkState* a, const GkState* b, GkStat* tmp, GkStat* out, GkState* res) noexcept {
    assert(a != nullptr && b != nullptr && res != nullptr && tmp != nullptr && out != nullptr);
    assert(tmp != out);
    const GkState* pick = b->count == 0 ? a : (a->count == 0 ? b : nullptr);
    if (pick != nullptr) {
        if (pick->len != 0 && out != pick->cur) std::memmove(out, pick->cur, sizeof(GkStat) * pick->len);
        *res = *pick;
        res->cur = out;
        return;
    }
    const double eps = a->rel_err > b->rel_err ? a->rel_err : b->rel_err;
    const std::int64_t add_self = static_cast<std::int64_t>(std::floor(2.0 * b->rel_err * static_cast<double>(b->count)));
    const std::int64_t add_other = static_cast<std::int64_t>(std::floor(2.0 * a->rel_err * static_cast<double>(a->count)));
    std::uint32_t i = 0, j = 0, o = 0;
    while (i < a->len && j < b->len) {
        if (a->cur[i].v < b->cur[j].v) {
            GkStat s = a->cur[i++];
            s.delta += j > 0 ? add_self : 0;
            tmp[o++] = s;
        } else {
            GkStat s = b->cur[j++];
            s.delta += i > 0 ? add_other : 0;
            tmp[o++] = s;
        }
    }
    while (i < a->len) tmp[o++] = a->cur[i++];
    while (j < b->len) tmp[o++] = b->cur[j++];
    assert(o == a->len + b->len);
    const std::int64_t n = a->count + b->count;
    res->len = gk_compress_into(tmp, o, 2.0 * eps * static_cast<double>(n), out);
    res->cur = out;
    res->count = n;
    res->rel_err = eps;
    res->threshold = b->threshold;
}

// QuantileSummaries.query(percentiles) over a non-empty compressed summary of
// `len` samples read through at(i): percentiles visited in ascending (stable)
// order, the scan resuming where the previous one stopped. `order` is n
// entries of scratch; out[i] answers ps[i].
template <class At>
inline void gk_query_at(At at, std::uint32_t len, std::int64_t count, double rel_err,
                        const double* BOLT_RESTRICT ps, std::uint32_t n,
                        std::uint32_t* BOLT_RESTRICT order, double* BOLT_RESTRICT out) noexcept {
    assert(len > 0 && ps != nullptr);
    assert(order != nullptr && out != nullptr);
    std::int64_t tmax = INT64_MIN;
    for (std::uint32_t i = 0; i < len; ++i) {
        const GkStat st = at(i);
        if (st.delta + st.g > tmax) tmax = st.delta + st.g;
    }
    const double target = static_cast<double>(tmax / 2);
    for (std::uint32_t i = 0; i < n; ++i) {   // stable insertion sort by value
        std::uint32_t k = i;
        while (k > 0 && gk_java_sort_key(ps[order[k - 1]]) > gk_java_sort_key(ps[i])) { order[k] = order[k - 1]; --k; }
        order[k] = i;
    }
    std::uint32_t index = 0;
    std::int64_t min_rank = at(0).g;
    for (std::uint32_t k = 0; k < n; ++k) {
        const double p = ps[order[k]];
        if (p <= rel_err) { out[order[k]] = at(0).v; continue; }
        if (p >= 1.0 - rel_err) { out[order[k]] = at(len - 1).v; continue; }
        const std::int64_t rank = static_cast<std::int64_t>(std::ceil(p * static_cast<double>(count)));
        std::uint32_t i = index;
        std::int64_t mr = min_rank;
        bool found = false;
        GkStat cur = at(i);
        while (i + 1 < len) {
            const std::int64_t max_rank = mr + cur.delta;
            if (static_cast<double>(max_rank) - target <= static_cast<double>(rank) &&
                static_cast<double>(rank) <= static_cast<double>(mr) + target) { found = true; break; }
            ++i;
            cur = at(i);
            mr += cur.g;
        }
        if (!found) { i = len - 1; mr = 0; cur = at(i); }
        index = i;
        min_rank = mr;
        out[order[k]] = cur.v;
    }
}

inline void gk_query_many(const GkState* BOLT_RESTRICT s, const double* BOLT_RESTRICT ps, std::uint32_t n,
                          std::uint32_t* BOLT_RESTRICT order, double* BOLT_RESTRICT out) noexcept {
    assert(s != nullptr && s->len > 0 && s->cur != nullptr);
    const GkStat* c = s->cur;
    gk_query_at([c](std::uint32_t i) { return c[i]; }, s->len, s->count, s->rel_err, ps, n, order, out);
}

inline double gk_query(const GkState* BOLT_RESTRICT s, double q) noexcept {
    assert(s != nullptr && s->len > 0);
    assert(q >= 0.0 && q <= 1.0);
    std::uint32_t order = 0;
    double out = 0.0;
    gk_query_many(s, &q, 1, &order, &out);
    return out;
}

// ---- ApproximatePercentile.PercentileDigestSerializer (big-endian) ----

constexpr std::uint32_t k_gk_ser_header = 4 + 8 + 8 + 4;
constexpr std::uint32_t k_gk_ser_stat = 8 + 8 + 8;

inline std::uint64_t gk_serialized_bytes(std::uint32_t len) noexcept {
    return k_gk_ser_header + static_cast<std::uint64_t>(len) * k_gk_ser_stat;
}

BOLT_FORCE_INLINE void gk_put_be(std::uint8_t* p, std::uint64_t v, unsigned n) noexcept {
    for (unsigned i = 0; i < n; ++i) p[i] = static_cast<std::uint8_t>(v >> (8u * (n - 1u - i)));
}
BOLT_FORCE_INLINE std::uint64_t gk_get_be(const std::uint8_t* p, unsigned n) noexcept {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i) v = (v << 8) | p[i];
    return v;
}

// Writes gk_serialized_bytes(s->len) bytes to `out`.
inline void gk_serialize(const GkState* s, std::uint8_t* out) noexcept {
    assert(s != nullptr && out != nullptr);
    assert(s->len == 0 || s->cur != nullptr);
    std::uint64_t b = 0;
    const std::int32_t thr = s->threshold == 0 ? static_cast<std::int32_t>(k_gk_compress_threshold) : s->threshold;
    gk_put_be(out, static_cast<std::uint32_t>(thr), 4);
    std::memcpy(&b, &s->rel_err, 8);
    gk_put_be(out + 4, b, 8);
    gk_put_be(out + 12, static_cast<std::uint64_t>(s->count), 8);
    gk_put_be(out + 20, s->len, 4);
    std::uint8_t* p = out + k_gk_ser_header;
    for (std::uint32_t i = 0; i < s->len; ++i, p += k_gk_ser_stat) {
        std::memcpy(&b, &s->cur[i].v, 8);
        gk_put_be(p, b, 8);
        gk_put_be(p + 8, static_cast<std::uint64_t>(s->cur[i].g), 8);
        gk_put_be(p + 16, static_cast<std::uint64_t>(s->cur[i].delta), 8);
    }
}

// The sample count of a serialized summary, or -1 when `n` bytes are not one.
inline std::int64_t gk_serialized_len(const std::uint8_t* in, std::uint64_t n) noexcept {
    if (in == nullptr || n < k_gk_ser_header) return -1;
    const std::uint64_t len = gk_get_be(in + 20, 4);
    if (len > 0x7FFFFFFFu || gk_serialized_bytes(static_cast<std::uint32_t>(len)) != n) return -1;
    return static_cast<std::int64_t>(len);
}

// Reads a summary written by gk_serialize into `samples` (cap entries).
inline bool gk_deserialize(const std::uint8_t* in, std::uint64_t n, GkStat* samples, std::uint32_t cap,
                           GkState* s) noexcept {
    assert(s != nullptr);
    const std::int64_t len = gk_serialized_len(in, n);
    if (len < 0 || static_cast<std::uint64_t>(len) > cap || (len > 0 && samples == nullptr)) return false;
    std::uint64_t b = gk_get_be(in + 4, 8);
    std::memcpy(&s->rel_err, &b, 8);
    s->threshold = static_cast<std::int32_t>(gk_get_be(in, 4));
    s->count = static_cast<std::int64_t>(gk_get_be(in + 12, 8));
    s->len = static_cast<std::uint32_t>(len);
    s->cur = samples;
    const std::uint8_t* p = in + k_gk_ser_header;
    for (std::int64_t i = 0; i < len; ++i, p += k_gk_ser_stat) {
        b = gk_get_be(p, 8);
        std::memcpy(&samples[i].v, &b, 8);
        samples[i].g = static_cast<std::int64_t>(gk_get_be(p + 8, 8));
        samples[i].delta = static_cast<std::int64_t>(gk_get_be(p + 16, 8));
    }
    return s->count >= 0 && (s->len == 0) == (s->count == 0);
}

// The same query over a serialized summary (gk_serialize bytes), no copy.
// Returns false when the bytes are not a summary or it is empty (Spark: NULL).
inline bool gk_query_serialized(const std::uint8_t* in, std::uint64_t n, const double* ps, std::uint32_t k,
                                std::uint32_t* order, double* out) noexcept {
    const std::int64_t len = gk_serialized_len(in, n);
    if (len <= 0) return false;
    std::uint64_t b = gk_get_be(in + 4, 8);
    double rel = 0.0;
    std::memcpy(&rel, &b, 8);
    const auto count = static_cast<std::int64_t>(gk_get_be(in + 12, 8));
    if (count <= 0) return false;
    const std::uint8_t* base = in + k_gk_ser_header;
    const auto at = [base](std::uint32_t i) {
        const std::uint8_t* p = base + static_cast<std::uint64_t>(i) * k_gk_ser_stat;
        GkStat st{};
        const std::uint64_t vb = gk_get_be(p, 8);
        std::memcpy(&st.v, &vb, 8);
        st.g = static_cast<std::int64_t>(gk_get_be(p + 8, 8));
        st.delta = static_cast<std::int64_t>(gk_get_be(p + 16, 8));
        return st;
    };
    gk_query_at(at, static_cast<std::uint32_t>(len), count, rel, ps, k, order, out);
    return true;
}

}  // namespace stat_agg
}  // namespace kernels
}  // namespace bolt
