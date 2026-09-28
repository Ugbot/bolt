// graph_algo_fuzz_harness.cpp — driven by tests/graph_algo_fuzz.py.
//
// Reads random graphs from stdin, runs every CSR algorithm kernel and the
// four path semantics over them, prints the answers for the Python side to
// check against networkx and a brute-force path enumerator. Two checks need
// no oracle and are done here: the compact u32 layout must answer exactly
// like the int64 one, and a traversal chopped into tiny out_cap / budget
// slices must equal the one-shot run (resumability). A disagreement prints a
// MISMATCH line, which the driver treats as a failure.
//
// Input, one case at a time:
//   case <id> <n> <m> <want_label> <sorted> <max_hops>
//   <u> <v> <label> <weight>        (m lines)
// Test-only: std::vector is fine here.

#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_bfs.h"
#include "bolt/kernels/bolt_csr_components.h"
#include "bolt/kernels/bolt_csr_dijkstra.h"
#include "bolt/kernels/bolt_csr_pagerank.h"
#include "bolt/kernels/bolt_csr_similarity.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bk = bolt::kernels;

namespace {

struct Case {
    int64_t id = 0, n = 0, m = 0;
    int32_t want = -1;
    int sorted = 0, max_hops = 0;
    std::vector<int64_t> u, v;
    std::vector<int32_t> lbl;
    std::vector<double> w;
};

struct Csr {
    std::vector<int64_t> off, nbr, eid;
    std::vector<int32_t> lbl;
    std::vector<double> w;
    std::vector<uint32_t> nbr32, eid32;
    bk::CsrGraph64 g64() const {
        bk::CsrGraph64 g{};
        g.off = off.data(); g.neighbors = nbr.data(); g.edge_ids = eid.data();
        g.edge_labels = lbl.data(); g.n_nodes = static_cast<int64_t>(off.size()) - 1;
        return g;
    }
    bk::CsrGraph32 g32() const {
        bk::CsrGraph32 g{};
        g.off = off.data(); g.neighbors = nbr32.data(); g.edge_ids = eid32.data();
        g.edge_labels = lbl.data(); g.n_nodes = static_cast<int64_t>(off.size()) - 1;
        return g;
    }
};

Csr build(const Case& c, bool reverse) {
    std::vector<int64_t> order(static_cast<size_t>(c.m));
    for (int64_t i = 0; i < c.m; ++i) order[static_cast<size_t>(i)] = i;
    const auto& src = reverse ? c.v : c.u;
    const auto& dst = reverse ? c.u : c.v;
    if (c.sorted)
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return src[a] != src[b] ? src[a] < src[b] : dst[a] < dst[b];
        });
    std::vector<int64_t> s(order.size()), d(order.size()), e(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
        s[k] = src[static_cast<size_t>(order[k])];
        d[k] = dst[static_cast<size_t>(order[k])];
        e[k] = order[k];
    }
    Csr r;
    const size_t n = static_cast<size_t>(c.n), m = static_cast<size_t>(c.m);
    r.off.assign(n + 1, 0); r.nbr.assign(m ? m : 1, 0); r.eid.assign(m ? m : 1, 0);
    std::vector<int64_t> scratch(n, 0);
    if (!bk::csr_build(s.data(), d.data(), e.data(), c.m, c.n, r.off.data(),
                       r.nbr.data(), r.eid.data(), scratch.data()))
        std::printf("MISMATCH csr_build refused\n");
    r.lbl.resize(r.nbr.size()); r.w.resize(r.nbr.size());
    r.nbr32.resize(r.nbr.size()); r.eid32.resize(r.nbr.size());
    for (size_t j = 0; j < m; ++j) {
        r.lbl[j] = c.lbl[static_cast<size_t>(r.eid[j])];
        r.w[j] = c.w[static_cast<size_t>(r.eid[j])];
        r.nbr32[j] = static_cast<uint32_t>(r.nbr[j]);
        r.eid32[j] = static_cast<uint32_t>(r.eid[j]);
    }
    return r;
}

void print_vec(const char* tag, const std::vector<int64_t>& a) {
    std::printf("%s", tag);
    for (int64_t x : a) std::printf(" %" PRId64, x);
    std::printf("\n");
}

template <typename G>
std::vector<int64_t> degrees(const G& g) {
    std::vector<int64_t> d(static_cast<size_t>(g.n_nodes));
    if (bk::csr_degree(&g, d.data()) != bk::CsrAlgoStatus::Ok) d.assign(1, -999);
    return d;
}

template <typename G>
void run_components(const G& g, std::vector<int64_t>* wcc, std::vector<int64_t>* scc) {
    const size_t n = static_cast<size_t>(g.n_nodes);
    wcc->assign(n + 1, 0); scc->assign(n + 1, 0);
    int64_t nw = 0, ns = 0;
    if (bk::csr_wcc(&g, wcc->data() + 1, &nw) != bk::CsrAlgoStatus::Ok) nw = -1;
    std::vector<int64_t> ix(n), lo(n), st(n), cn(n), ce(n);
    std::vector<uint64_t> on((n + 63) / 64);
    bk::CsrSccScratch sc{ix.data(), lo.data(), st.data(), cn.data(), ce.data(), on.data()};
    if (bk::csr_scc(&g, &sc, scc->data() + 1, &ns) != bk::CsrAlgoStatus::Ok) ns = -1;
    (*wcc)[0] = nw; (*scc)[0] = ns;
}

template <typename G>
std::vector<double> pagerank(const G& g, int32_t* iters) {
    std::vector<double> r(static_cast<size_t>(g.n_nodes)), s(r.size());
    const bk::CsrAlgoStatus st = bk::csr_pagerank(&g, 0.85, 10000, 1e-13,
                                                  r.data(), s.data(), iters);
    if (st != bk::CsrAlgoStatus::Ok) *iters = -static_cast<int32_t>(st);
    return r;
}

template <typename G>
std::vector<double> dijkstra(const G& g, const Csr& c, int64_t s, int* status) {
    const size_t n = static_cast<size_t>(g.n_nodes);
    std::vector<double> dist(n);
    std::vector<int64_t> par(n), ps(n), scratch(2 * n);
    int64_t settled = 0;
    *status = static_cast<int>(bk::csr_dijkstra(&g, c.w.data(), s, -1, dist.data(),
                                                par.data(), ps.data(),
                                                scratch.data(), &settled));
    // The recorded tree must realise every distance.
    for (size_t v = 0; v < n && *status == 0; ++v) {
        if (static_cast<int64_t>(v) == s || !std::isfinite(dist[v])) continue;
        const int64_t j = ps[v];
        if (j < 0 || dist[static_cast<size_t>(par[v])] + c.w[static_cast<size_t>(j)] != dist[v])
            std::printf("MISMATCH dijkstra tree node %zu\n", v);
    }
    return dist;
}

template <typename G>
std::vector<double> similarity(const G& g, bool sorted, bk::CsrSimilarity k) {
    const int64_t n = g.n_nodes;
    std::vector<int64_t> us, vs;
    for (int64_t a = 0; a < n; ++a)
        for (int64_t b = 0; b < n; ++b) { us.push_back(a); vs.push_back(b); }
    const int64_t cap = bk::csr_max_degree(&g) + 1;
    std::vector<int64_t> ba(static_cast<size_t>(cap)), bb(ba.size()), bt(ba.size());
    std::vector<double> out(us.size());
    if (bk::csr_node_similarity(&g, us.data(), vs.data(),
                                static_cast<int64_t>(us.size()), k, sorted,
                                ba.data(), bb.data(), bt.data(), cap,
                                out.data()) != bk::CsrAlgoStatus::Ok)
        out.assign(1, -1.0);
    return out;
}

// All rows of one semantics from every source, chopped into slices.
template <typename G>
std::vector<std::pair<int64_t, int64_t>> paths(const G& g, bk::CsrPathSemantics sem,
                                               int32_t lo, int32_t hi,
                                               int64_t slice, int64_t budget,
                                               int* status) {
    const int64_t n = g.n_nodes;
    std::vector<int64_t> src(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) src[static_cast<size_t>(i)] = i;
    bk::CsrBfsParams p{};
    p.per_source_cap = INT64_MAX; p.visit_budget = budget;
    p.min_hops = lo; p.max_hops = hi; p.semantics = sem;
    const int32_t slots = (sem == bk::CsrPathSemantics::Reach) ? 1 : hi + 1;
    std::vector<int64_t> path(static_cast<size_t>(slots)), ep(path.size()), it(path.size());
    std::vector<uint64_t> seen(static_cast<size_t>(bk::csr_reach_seen_words(n)), 0);
    std::vector<int64_t> queue(static_cast<size_t>(n));
    bk::CsrBfsScratch sc{path.data(), ep.data(), it.data(), seen.data(), queue.data()};
    bk::CsrBfsCursor cur{};
    std::vector<int64_t> os(static_cast<size_t>(slice)), od(os.size());
    std::vector<std::pair<int64_t, int64_t>> rows;
    *status = 0;
    for (int64_t guard = 0; !bk::csr_bfs_done(&cur, n); ++guard) {
        if (guard > (int64_t{1} << 26)) { *status = 99; break; }
        int64_t got = 0;
        const bk::CsrBfsStatus s = bk::csr_bfs_expand(src.data(), n, &g, &p, &sc,
                                                      os.data(), od.data(), nullptr,
                                                      slice, &cur, &got);
        for (int64_t r = 0; r < got; ++r)
            rows.emplace_back(os[static_cast<size_t>(r)], od[static_cast<size_t>(r)]);
        if (s != bk::CsrBfsStatus::Ok) { *status = static_cast<int>(s); break; }
    }
    for (uint64_t x : seen)
        if (x != 0 && *status == 0) std::printf("MISMATCH reach left seen bits set\n");
    return rows;
}

void print_rows(const char* tag, int st,
                const std::vector<std::pair<int64_t, int64_t>>& rows) {
    std::printf("%s %d", tag, st);
    for (const auto& r : rows) std::printf(" %" PRId64 ":%" PRId64, r.first, r.second);
    std::printf("\n");
}

void run_paths(const Case& c, const Csr& f) {
    bk::CsrGraph64 g = f.g64();
    bk::CsrGraph32 g32 = f.g32();
    g.want_label = c.want; g32.want_label = c.want;
    const char* names[4] = {"walk", "trail", "simple", "reach"};
    for (int sem = 0; sem < 4; ++sem) {
        for (int32_t lo = 0; lo <= 1; ++lo) {
            const auto S = static_cast<bk::CsrPathSemantics>(sem);
            const int32_t hi = c.max_hops < lo ? lo : c.max_hops;
            int st = 0, st2 = 0, st3 = 0;
            const auto rows = paths(g, S, lo, hi, 1 << 16, 1 << 20, &st);
            const auto chop = paths(g, S, lo, hi, 1 + (c.id % 3), 1 + (c.id % 4), &st2);
            const auto rows32 = paths(g32, S, lo, hi, 7, 5, &st3);
            if (rows != chop || rows != rows32 || st != st2 || st != st3)
                std::printf("MISMATCH %s resumable/compact lo=%d\n", names[sem], lo);
            char tag[64];
            std::snprintf(tag, sizeof(tag), "path %s %d %d", names[sem], lo, hi);
            print_rows(tag, st, rows);
            if (sem == 3) {
                int su = 0;
                const auto unb = paths(g, S, lo, bk::k_csr_reach_unbounded,
                                       1 << 16, 1 << 20, &su);
                std::snprintf(tag, sizeof(tag), "path reach %d -1", lo);
                print_rows(tag, su, unb);
            }
        }
    }
}

void run_case(const Case& c) {
    std::printf("case %" PRId64 "\n", c.id);
    const Csr f = build(c, false);
    const Csr r = build(c, true);
    bk::CsrGraph64 g = f.g64(), gr = r.g64();
    bk::CsrGraph32 g32 = f.g32();
    g.want_label = gr.want_label = g32.want_label = c.want;
    const auto od = degrees(g), id = degrees(gr);
    if (od != degrees(g32)) std::printf("MISMATCH degree compact\n");
    print_vec("outdeg", od);
    print_vec("indeg", id);
    std::vector<int64_t> w, s, w32, s32;
    run_components(g, &w, &s);
    run_components(g32, &w32, &s32);
    if (w != w32 || s != s32) std::printf("MISMATCH components compact\n");
    print_vec("wcc", w);
    print_vec("scc", s);
    int32_t it = 0, it32 = 0;
    const auto pr = pagerank(g, &it), pr32 = pagerank(g32, &it32);
    if (pr != pr32 || it != it32) std::printf("MISMATCH pagerank compact\n");
    std::printf("pr %d", it);
    for (double x : pr) std::printf(" %.17g", x);
    std::printf("\n");
    for (int64_t src = 0; src < c.n; ++src) {
        int st = 0, st32 = 0;
        const auto d = dijkstra(g, f, src, &st);
        if (d != dijkstra(g32, f, src, &st32) || st != st32)
            std::printf("MISMATCH dijkstra compact\n");
        std::printf("dj %" PRId64 " %d", src, st);
        for (double x : d) std::printf(" %.17g", x);
        std::printf("\n");
    }
    for (int k = 0; k < 2; ++k) {
        const auto K = static_cast<bk::CsrSimilarity>(k);
        const auto sim = similarity(g, c.sorted != 0, K);
        if (sim != similarity(g32, c.sorted != 0, K))
            std::printf("MISMATCH similarity compact\n");
        std::printf(k == 0 ? "jac" : "ovl");
        for (double x : sim) std::printf(" %.17g", x);
        std::printf("\n");
    }
    run_paths(c, f);
    std::printf("end\n");
}

}  // namespace

int main() {
    char word[16];
    Case c;
    while (std::scanf("%15s", word) == 1) {
        if (std::strcmp(word, "case") != 0) return 2;
        if (std::scanf("%" SCNd64 " %" SCNd64 " %" SCNd64 " %d %d %d", &c.id, &c.n,
                       &c.m, &c.want, &c.sorted, &c.max_hops) != 6)
            return 2;
        const size_t m = static_cast<size_t>(c.m);
        c.u.resize(m); c.v.resize(m); c.lbl.resize(m); c.w.resize(m);
        for (size_t i = 0; i < m; ++i)
            if (std::scanf("%" SCNd64 " %" SCNd64 " %d %lf", &c.u[i], &c.v[i],
                           &c.lbl[i], &c.w[i]) != 4)
                return 2;
        run_case(c);
        std::fflush(stdout);
    }
    return 0;
}
