// G2GRAPH-359 (GF1) — model-based fuzz of the CSR kernels against a plain
// edge list. Every other CSR test compares one kernel with another (sorted
// probe vs linear walk, compact vs int64); here the reference is the edge
// list itself, so a bug both sides share is still caught.
//
// Random graphs carry parallel edges, self loops, isolated nodes, hubs and a
// node window that leaves a prefix and suffix of the id space edgeless.
// csr_build must place every edge in its source's block in insertion order;
// csr_expand_graph over the int64 layout and over the compact uint32 window,
// with labels, relationship exclusions, bound destinations (sorted blocks
// probed by binary search), and every output cap, must emit exactly the rows
// the edge list says, in order, across resumes.
//
// Seeded: BOLT_CSR_FUZZ_SEED / BOLT_CSR_FUZZ_ITERS; a failure prints both.

#include "bolt/kernels/bolt_csr.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

using bolt::kernels::CsrExpandCursor;
using bolt::kernels::CsrGraph32;
using bolt::kernels::CsrGraph64;
using bolt::kernels::csr_build;
using bolt::kernels::csr_expand_graph;
using bolt::kernels::kCsrExpandOutOfRange;

uint64_t env_u64(const char* n, uint64_t d) {
    const char* v = std::getenv(n);
    return (v == nullptr || v[0] == '\0') ? d : std::strtoull(v, nullptr, 10);
}

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int64_t below(int64_t n) { return n <= 0 ? 0 : static_cast<int64_t>(next() % static_cast<uint64_t>(n)); }
};

struct Edge { int64_t src, dst, eid; int32_t lbl; };

struct Model {
    int64_t n_nodes = 0;
    std::vector<Edge> edges;   // insertion order
};

Model random_model(Rng* r) {
    Model m;
    m.n_nodes = 1 + r->below(r->below(4) == 0 ? 3 : 160);
    // Edges only between [lo, hi): the rest of the id space is isolated, so
    // the compact window has a real prefix and suffix to skip.
    const int64_t lo = r->below(m.n_nodes);
    const int64_t hi = lo + 1 + r->below(m.n_nodes - lo);
    const int64_t n_edges = r->below(4) == 0 ? 0 : r->below(900);
    const int64_t hub = lo + r->below(hi - lo);
    int64_t eid = r->below(3) == 0 ? 0 : 5000 + r->below(1000);
    for (int64_t i = 0; i < n_edges; ++i) {
        Edge e{};
        e.src = r->below(4) == 0 ? hub : lo + r->below(hi - lo);
        const int64_t kind = r->below(10);
        if (kind == 0) e.dst = e.src;                                   // self loop
        else if (kind < 3 && !m.edges.empty()) {                        // parallel edge
            const Edge& p = m.edges[static_cast<size_t>(r->below(static_cast<int64_t>(m.edges.size())))];
            e.src = p.src;
            e.dst = p.dst;
        } else e.dst = r->below(m.n_nodes);
        e.eid = eid;
        eid += 1 + r->below(3);
        e.lbl = static_cast<int32_t>(r->below(3));
        m.edges.push_back(e);
    }
    return m;
}

struct Built {
    std::vector<int64_t> off, nbr, eid;
    std::vector<int32_t> lbl;
    bool sorted = false;
};

// csr_build, plus the label array scattered in the same order, and checked
// block by block against the edge list.
Built build_and_check(const Model& m) {
    Built b;
    const size_t n = m.edges.size();
    std::vector<int64_t> src(n), dst(n), eid(n), scratch(static_cast<size_t>(m.n_nodes));
    for (size_t i = 0; i < n; ++i) {
        src[i] = m.edges[i].src;
        dst[i] = m.edges[i].dst;
        eid[i] = m.edges[i].eid;
    }
    b.off.resize(static_cast<size_t>(m.n_nodes) + 1);
    b.nbr.resize(n);
    b.eid.resize(n);
    EXPECT_TRUE(csr_build(src.data(), dst.data(), eid.data(), static_cast<int64_t>(n), m.n_nodes,
                          b.off.data(), b.nbr.data(), b.eid.data(), scratch.data()));
    EXPECT_EQ(b.off[0], 0);
    EXPECT_EQ(b.off[static_cast<size_t>(m.n_nodes)], static_cast<int64_t>(n));
    std::vector<int64_t> cursor(b.off.begin(), b.off.end() - 1);
    b.lbl.resize(n);
    for (const Edge& e : m.edges) {
        const size_t w = static_cast<size_t>(cursor[static_cast<size_t>(e.src)]++);
        EXPECT_EQ(b.nbr[w], e.dst);
        EXPECT_EQ(b.eid[w], e.eid);
        b.lbl[w] = e.lbl;
    }
    for (int64_t s = 0; s < m.n_nodes; ++s) {
        EXPECT_EQ(cursor[static_cast<size_t>(s)], b.off[static_cast<size_t>(s) + 1]);
    }
    return b;
}

// Blocks reordered by (dst, eid): the layout the bound-destination probe needs.
Built sorted_copy(const Built& in, int64_t n_nodes) {
    Built b = in;
    for (int64_t s = 0; s < n_nodes; ++s) {
        const size_t lo = static_cast<size_t>(b.off[static_cast<size_t>(s)]);
        const size_t hi = static_cast<size_t>(b.off[static_cast<size_t>(s) + 1]);
        std::vector<size_t> idx;
        for (size_t j = lo; j < hi; ++j) idx.push_back(j);
        std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
            return in.nbr[x] != in.nbr[y] ? in.nbr[x] < in.nbr[y] : in.eid[x] < in.eid[y];
        });
        for (size_t k = 0; k < idx.size(); ++k) {
            b.nbr[lo + k] = in.nbr[idx[k]];
            b.eid[lo + k] = in.eid[idx[k]];
            b.lbl[lo + k] = in.lbl[idx[k]];
        }
    }
    b.sorted = true;
    return b;
}

struct Query {
    std::vector<int64_t> srcs, bounds, excl;
    int32_t want = -1;
    bool labels = false;
    bool bound = false;
    int64_t cap = 1;
};

struct Row { int64_t s, e, d; };

// The reference: straight from the edge list, block order = insertion order
// or (dst, eid) when the CSR is sorted.
std::vector<Row> reference(const Model& m, const Query& q, bool sorted, bool* out_of_range) {
    std::vector<Row> out;
    *out_of_range = false;
    for (size_t i = 0; i < q.srcs.size(); ++i) {
        const int64_t s = q.srcs[i];
        if (s < 0 || s >= m.n_nodes) { *out_of_range = true; return out; }
        std::vector<Edge> blk;
        for (const Edge& e : m.edges) if (e.src == s) blk.push_back(e);
        if (sorted) {
            std::stable_sort(blk.begin(), blk.end(), [](const Edge& a, const Edge& b) {
                return a.dst != b.dst ? a.dst < b.dst : a.eid < b.eid;
            });
        }
        for (const Edge& e : blk) {
            if (q.labels && q.want >= 0 && e.lbl != q.want) continue;
            if (std::find(q.excl.begin(), q.excl.end(), e.eid) != q.excl.end()) continue;
            if (q.bound && e.dst != q.bounds[i]) continue;
            out.push_back(Row{s, e.eid, e.dst});
        }
    }
    return out;
}

template <typename G>
std::vector<Row> run(const G& g, const Query& q, bool sorted, bool* failed) {
    std::vector<Row> out;
    *failed = false;
    const int64_t n = static_cast<int64_t>(q.srcs.size());
    std::vector<int64_t> os(static_cast<size_t>(q.cap)), oe(os.size()), od(os.size());
    CsrExpandCursor cur{0, 0};
    for (int64_t call = 0; call < 100000 && cur.src_index < n; ++call) {
        const int64_t got = csr_expand_graph(
            q.srcs.data(), n, &g, q.excl.empty() ? nullptr : q.excl.data(),
            static_cast<int32_t>(q.excl.size()), q.bound ? q.bounds.data() : nullptr,
            os.data(), oe.data(), od.data(), q.cap, &cur, sorted);
        if (got == kCsrExpandOutOfRange) { *failed = true; return out; }
        for (int64_t i = 0; i < got; ++i) {
            out.push_back(Row{os[static_cast<size_t>(i)], oe[static_cast<size_t>(i)],
                              od[static_cast<size_t>(i)]});
        }
    }
    EXPECT_EQ(cur.src_index, n);
    return out;
}

Query random_query(const Model& m, Rng* r) {
    Query q;
    const int64_t n = r->below(24);
    for (int64_t i = 0; i < n; ++i) {
        int64_t s = r->below(m.n_nodes);
        if (!m.edges.empty() && r->below(2) == 0) {
            s = m.edges[static_cast<size_t>(r->below(static_cast<int64_t>(m.edges.size())))].src;
        }
        q.srcs.push_back(s);
        int64_t d = r->below(m.n_nodes);
        for (const Edge& e : m.edges) {
            if (e.src == s && r->below(3) == 0) { d = e.dst; break; }
        }
        q.bounds.push_back(d);
    }
    q.bound = r->below(2) == 0;
    q.labels = r->below(3) == 0;
    q.want = static_cast<int32_t>(r->below(4)) - 1;
    const int64_t n_ex = r->below(4);
    for (int64_t k = 0; k < n_ex && !m.edges.empty(); ++k) {
        q.excl.push_back(m.edges[static_cast<size_t>(r->below(static_cast<int64_t>(m.edges.size())))].eid);
    }
    const int64_t caps[5] = {1, 2, 3, 7, 4096};
    q.cap = caps[r->below(5)];
    return q;
}

void expect_same(const std::vector<Row>& want, const std::vector<Row>& got, const char* what) {
    ASSERT_EQ(want.size(), got.size()) << what;
    for (size_t i = 0; i < want.size(); ++i) {
        ASSERT_EQ(want[i].s, got[i].s) << what << " row " << i;
        ASSERT_EQ(want[i].e, got[i].e) << what << " row " << i;
        ASSERT_EQ(want[i].d, got[i].d) << what << " row " << i;
    }
}

// The compact layout over the tightest window that holds every edge source.
struct Compact {
    std::vector<int64_t> off;
    std::vector<uint32_t> nbr, eid;
    CsrGraph32 g;
};

Compact compact_of(const Built& b, int64_t n_nodes, const Query& q) {
    Compact c;
    int64_t lo = n_nodes, hi = 0;
    for (int64_t s = 0; s < n_nodes; ++s) {
        if (b.off[static_cast<size_t>(s) + 1] == b.off[static_cast<size_t>(s)]) continue;
        lo = std::min(lo, s);
        hi = s + 1;
    }
    if (lo >= hi) { lo = 0; hi = 0; }
    for (int64_t s = lo; s <= hi; ++s) c.off.push_back(b.off[static_cast<size_t>(s)] - b.off[static_cast<size_t>(lo)]);
    if (c.off.empty()) c.off.push_back(0);
    for (size_t j = 0; j < b.nbr.size(); ++j) {
        c.nbr.push_back(static_cast<uint32_t>(b.nbr[j]));
        c.eid.push_back(static_cast<uint32_t>(b.eid[j]));
    }
    c.g.off = c.off.data();
    c.g.neighbors = c.nbr.data();
    c.g.edge_ids = c.eid.data();
    c.g.edge_labels = q.labels ? b.lbl.data() : nullptr;
    c.g.n_nodes = n_nodes;
    c.g.want_label = q.labels ? q.want : -1;
    c.g.node_lo = lo;
    c.g.n_win = hi - lo;
    return c;
}

void one_case(uint64_t seed, uint64_t it) {
    Rng r{seed * 0x100000001B3ull + it};
    const Model m = random_model(&r);
    const Built b = build_and_check(m);
    const Built bs = sorted_copy(b, m.n_nodes);
    for (int k = 0; k < 6; ++k) {
        Query q = random_query(m, &r);
        if (r.below(20) == 0 && !q.srcs.empty()) q.srcs[static_cast<size_t>(r.below(static_cast<int64_t>(q.srcs.size())))] = m.n_nodes + r.below(3);
        for (int layout = 0; layout < 2; ++layout) {
            const Built& bb = layout == 0 ? b : bs;
            bool oor = false;
            const std::vector<Row> want = reference(m, q, bb.sorted, &oor);
            CsrGraph64 g64{};
            g64.off = bb.off.data();
            g64.neighbors = bb.nbr.data();
            g64.edge_ids = bb.eid.data();
            g64.edge_labels = q.labels ? bb.lbl.data() : nullptr;
            g64.n_nodes = m.n_nodes;
            g64.want_label = q.labels ? q.want : -1;
            bool failed = false;
            // The fail-closed sentinel fires when the walk REACHES the bad
            // source, so only the rows before it are compared.
            const std::vector<Row> got = run(g64, q, bb.sorted, &failed);
            ASSERT_EQ(failed, oor) << "seed " << seed << " it " << it;
            if (!oor) expect_same(want, got, layout == 0 ? "int64 insertion" : "int64 sorted");
            const Compact c = compact_of(bb, m.n_nodes, q);
            const std::vector<Row> got32 = run(c.g, q, bb.sorted, &failed);
            ASSERT_EQ(failed, oor) << "seed " << seed << " it " << it;
            if (!oor) expect_same(want, got32, layout == 0 ? "compact insertion" : "compact sorted");
            if (::testing::Test::HasFatalFailure()) {
                ADD_FAILURE() << "replay: BOLT_CSR_FUZZ_SEED=" << seed << " first iteration " << it;
                return;
            }
        }
    }
}

TEST(CsrFuzz, KernelsMatchTheEdgeList) {
    const uint64_t seed = env_u64("BOLT_CSR_FUZZ_SEED", 1);
    const uint64_t iters = env_u64("BOLT_CSR_FUZZ_ITERS", 2000);
    const uint64_t first = env_u64("BOLT_CSR_FUZZ_FIRST", 0);
    for (uint64_t it = first; it < first + iters; ++it) {
        one_case(seed, it);
        if (HasFailure()) {
            ADD_FAILURE() << "replay: BOLT_CSR_FUZZ_SEED=" << seed << " BOLT_CSR_FUZZ_FIRST=" << it
                          << " BOLT_CSR_FUZZ_ITERS=1";
            return;
        }
    }
}

}  // namespace
