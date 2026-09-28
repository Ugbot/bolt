// G2GRAPH-342: csr_expand_graph's intersection keep term. A walk restricted
// to neighbours present in a second sorted block must write exactly the rows
// the unrestricted walk writes minus the non-members, in the same order,
// under every resume point. The seeded loop at the bottom is the fuzz test:
// random sorted compact CSRs with parallel edges, self loops, labels,
// exclusions, bound destinations and output caps.

#include "bolt/kernels/bolt_binsearch.h"
#include "bolt/kernels/bolt_csr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using bolt::kernels::bolt_gallop_lower_bound_tmpl;
using bolt::kernels::csr_expand_graph;
using bolt::kernels::CsrExpandCursor;
using bolt::kernels::CsrGraph32;
using bolt::kernels::kCsrExpandOutOfRange;

struct G {
    int64_t n = 0;
    std::vector<int64_t> off;
    std::vector<uint32_t> nbr, eid;
    std::vector<int32_t> lbl;
    CsrGraph32 view(int32_t want, bool labels) const {
        CsrGraph32 g{};
        g.off = off.data();
        g.neighbors = nbr.data();
        g.edge_ids = eid.data();
        g.edge_labels = labels ? lbl.data() : nullptr;
        g.n_nodes = n;
        g.want_label = want;
        return g;
    }
};

// Random multigraph with sorted blocks (by neighbour, then edge id).
G random_graph(std::mt19937_64& rng, int64_t n, int64_t max_deg,
               uint32_t eid_base) {
    G g;
    g.n = n;
    g.off.push_back(0);
    uint32_t next_eid = eid_base;
    for (int64_t s = 0; s < n; ++s) {
        const int64_t d = static_cast<int64_t>(rng() % (max_deg + 1));
        std::vector<uint32_t> ns;
        for (int64_t k = 0; k < d; ++k) {
            const uint32_t v = static_cast<uint32_t>(rng() % n);
            ns.push_back(v);
            if (rng() % 4 == 0) ns.push_back(v);      // parallel edge
        }
        std::sort(ns.begin(), ns.end());
        for (uint32_t v : ns) {
            g.nbr.push_back(v);
            g.eid.push_back(next_eid++);
            g.lbl.push_back(static_cast<int32_t>(rng() % 2));
        }
        g.off.push_back(static_cast<int64_t>(g.nbr.size()));
    }
    return g;
}

struct Rows {
    std::vector<int64_t> src, edge, dst;
    bool failed = false;
    bool operator==(const Rows& o) const {
        return src == o.src && edge == o.edge && dst == o.dst &&
               failed == o.failed;
    }
};

Rows run(const CsrGraph32& g, const std::vector<int64_t>& srcs,
         const std::vector<int64_t>& excl, const std::vector<int64_t>* bounds,
         const CsrGraph32* isect, const std::vector<int64_t>* keys,
         int64_t cap) {
    Rows r;
    std::vector<int64_t> os(cap), oe(cap), od(cap);
    CsrExpandCursor cur{0, 0};
    const int64_t n = static_cast<int64_t>(srcs.size());
    int64_t max_calls = n + 2;              // each call emits cap rows or ends
    for (int64_t s : srcs) {
        if (s >= 0 && s < g.n_nodes) max_calls += g.off[s + 1] - g.off[s];
    }
    for (int64_t call = 0; call < max_calls && cur.src_index < n; ++call) {
        const int64_t got = csr_expand_graph(
            srcs.data(), n, &g, excl.empty() ? nullptr : excl.data(),
            static_cast<int32_t>(excl.size()),
            bounds ? bounds->data() : nullptr, os.data(), oe.data(), od.data(),
            cap, &cur, /*dst_sorted=*/true, isect,
            keys ? keys->data() : nullptr);
        if (got == kCsrExpandOutOfRange) { r.failed = true; return r; }
        for (int64_t i = 0; i < got; ++i) {
            r.src.push_back(os[i]);
            r.edge.push_back(oe[i]);
            r.dst.push_back(od[i]);
        }
    }
    EXPECT_EQ(cur.src_index, n);
    return r;
}

bool member(const G& ig, int64_t key, int64_t v) {
    const auto b = ig.nbr.begin() + ig.off[key];
    const auto e = ig.nbr.begin() + ig.off[key + 1];
    return std::binary_search(b, e, static_cast<uint32_t>(v));
}

TEST(CsrIsect, GallopMatchesLowerBound) {
    std::mt19937_64 rng(342);
    for (int t = 0; t < 2000; ++t) {
        const int64_t n = static_cast<int64_t>(rng() % 200);
        std::vector<uint32_t> a(n);
        for (auto& x : a) x = static_cast<uint32_t>(rng() % 300);
        std::sort(a.begin(), a.end());
        const uint32_t key = static_cast<uint32_t>(rng() % 320);
        const int64_t want = std::lower_bound(a.begin(), a.end(), key) - a.begin();
        EXPECT_EQ(bolt_gallop_lower_bound_tmpl<uint32_t>(a.data(), n, key), want);
    }
}

TEST(CsrIsect, TriangleByHand) {
    // 0->1, 0->2, 1->2, 2->0 (twice), plus 1->1 self loop.
    G g;
    g.n = 3;
    g.off = {0, 2, 4, 6};
    g.nbr = {1, 2, 1, 2, 0, 0};
    g.eid = {0, 1, 2, 3, 4, 5};
    g.lbl = {0, 0, 0, 0, 0, 0};
    // Reverse of g: in-neighbours, sorted.
    G rv;
    rv.n = 3;
    rv.off = {0, 2, 4, 6};
    rv.nbr = {2, 2, 0, 1, 0, 1};
    rv.eid = {4, 5, 0, 2, 1, 3};
    rv.lbl = {0, 0, 0, 0, 0, 0};
    const CsrGraph32 gv = g.view(-1, false);
    const CsrGraph32 iv = rv.view(-1, false);
    // Source b=1, closing node a=0: c in out(1)={1,2} AND c->0: in(0)={2,2}.
    const std::vector<int64_t> srcs{1};
    const std::vector<int64_t> keys{0};
    const Rows r = run(gv, srcs, {}, nullptr, &iv, &keys, 4);
    EXPECT_EQ(r.dst, (std::vector<int64_t>{2}));
    EXPECT_EQ(r.edge, (std::vector<int64_t>{3}));
    // Out-of-range key fails closed.
    const std::vector<int64_t> bad{7};
    EXPECT_TRUE(run(gv, srcs, {}, nullptr, &iv, &bad, 4).failed);
}

TEST(CsrIsect, SeededDifferentialFuzz) {
    std::mt19937_64 rng(0x6a2342ull);
    for (int t = 0; t < 1500; ++t) {
        const int64_t n = 1 + static_cast<int64_t>(rng() % 40);
        const G g = random_graph(rng, n, 1 + static_cast<int64_t>(rng() % 12), 0);
        const G ig = random_graph(rng, n, 1 + static_cast<int64_t>(rng() % 12), 100000);
        const bool labels = rng() % 2 == 0;
        const int32_t want = (rng() % 3 == 0) ? -1 : static_cast<int32_t>(rng() % 2);
        const CsrGraph32 gv = g.view(want, labels);
        const CsrGraph32 iv = ig.view(-1, false);
        const int64_t m = 1 + static_cast<int64_t>(rng() % 12);
        std::vector<int64_t> srcs(m), keys(m), bounds(m);
        for (int64_t i = 0; i < m; ++i) {
            srcs[i] = static_cast<int64_t>(rng() % n);
            keys[i] = static_cast<int64_t>(rng() % n);
            bounds[i] = static_cast<int64_t>(rng() % n);
        }
        std::vector<int64_t> excl;
        if (rng() % 3 == 0 && !g.eid.empty()) {
            excl.push_back(g.eid[rng() % g.eid.size()]);
        }
        const bool bound = rng() % 4 == 0;
        const int64_t cap = 1 + static_cast<int64_t>(rng() % 7);
        const std::vector<int64_t>* bp = bound ? &bounds : nullptr;

        const Rows plain = run(gv, srcs, excl, bp, nullptr, nullptr, 64);
        Rows want_rows;
        // Reference: expand each input row alone, keep members of its key.
        for (int64_t i = 0; i < m; ++i) {
            const std::vector<int64_t> one{srcs[i]};
            const std::vector<int64_t> ob{bounds[i]};
            const Rows ri = run(gv, one, excl, bound ? &ob : nullptr, nullptr,
                                nullptr, 64);
            for (size_t k = 0; k < ri.src.size(); ++k) {
                if (!member(ig, keys[i], ri.dst[k])) continue;
                want_rows.src.push_back(ri.src[k]);
                want_rows.edge.push_back(ri.edge[k]);
                want_rows.dst.push_back(ri.dst[k]);
            }
        }
        const Rows got = run(gv, srcs, excl, bp, &iv, &keys, cap);
        ASSERT_TRUE(got == want_rows) << "trial " << t;
        ASSERT_LE(got.src.size(), plain.src.size());
    }
}

}  // namespace
