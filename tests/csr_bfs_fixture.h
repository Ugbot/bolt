// csr_bfs_fixture.h — shared fixtures for the CSR traversal kernel tests.
//
// Two things live here so test_bolt_csr_bfs.cpp and test_bolt_csr_shortest.cpp
// agree on them exactly:
//   - KnownBfsCsr : the 5-node / 6-edge deterministic graph, laid out in CSR
//     form. It MUST mirror csr_bfs_enumerate.py's NODES/EDGES/LABELS — that
//     script is the oracle for every literal the pins assert.
//   - RandomGraph : loader for data/csr_bfs_random.txt, the 1,000-node /
//     10,000-edge graph whose expectations come from networkx.
//
// Test-only code: std::vector / ifstream are fine here (they are not on any
// bolt hot path), but the kernels under test still receive raw pointers.

#pragma once

#include "bolt/kernels/bolt_csr.h"
#include "bolt/kernels/bolt_csr_bfs.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The deterministic fixture. Mirrors csr_bfs_enumerate.py exactly:
//   0 -100-> 1 ; 0 -101-> 1 ; 1 -102-> 2 ; 2 -103-> 0 ; 2 -105-> 4 ; 3 -104-> 3
// Parallel 100/101 separates TRAIL from SIMPLE; the 0->1->2->0 cycle separates
// WALK from SIMPLE; the 3 -104-> 3 self-loop is the shape SIMPLE drops.
// ---------------------------------------------------------------------------
struct KnownBfsCsr {
    static constexpr int64_t kNodes = 5;
    int64_t off[6] = {0, 2, 3, 5, 6, 6};
    int64_t nbr[6] = {1, 1, 2, 0, 4, 3};
    int64_t eid[6] = {100, 101, 102, 103, 105, 104};
    int32_t lbl[6] = {7, 9, 7, 7, 7, 7};

    bolt::kernels::CsrBfsGraph graph(int32_t want_label = -1,
                                     bool labeled = false) const {
        bolt::kernels::CsrBfsGraph g{};
        g.off         = off;
        g.neighbors   = nbr;
        g.edge_ids    = eid;
        g.edge_labels = labeled ? lbl : nullptr;
        g.n_nodes     = kNodes;
        g.want_label  = want_label;
        return g;
    }
};

// Scratch big enough for any hop window this suite uses.
struct BfsScratchBuf {
    int64_t path[bolt::kernels::k_csr_bfs_max_hops + 1]  = {0};
    int64_t epath[bolt::kernels::k_csr_bfs_max_hops + 1] = {0};
    int64_t iter[bolt::kernels::k_csr_bfs_max_hops + 1]  = {0};

    bolt::kernels::CsrBfsScratch view() {
        bolt::kernels::CsrBfsScratch sc{};
        sc.path  = path;
        sc.epath = epath;
        sc.iter  = iter;
        return sc;
    }
};

inline bolt::kernels::CsrBfsParams bfs_params(
        int32_t min_hops, int32_t max_hops,
        bolt::kernels::CsrPathSemantics sem,
        int64_t per_source_cap = 1 << 20,
        int64_t visit_budget = 1 << 30) {
    bolt::kernels::CsrBfsParams p{};
    p.per_source_cap = per_source_cap;
    p.visit_budget   = visit_budget;
    p.min_hops       = min_hops;
    p.max_hops       = max_hops;
    p.semantics      = sem;
    return p;
}

// ---------------------------------------------------------------------------
// data/csr_bfs_random.txt — the networkx-oracled random graph.
// ---------------------------------------------------------------------------
struct ReachExpect { int64_t src; int32_t hops; int64_t count; };
struct PathsExpect { int64_t src; int32_t hops; int64_t walk, trail, simple; };
struct SpExpect    { int64_t s, t; int32_t len; };

struct RandomGraph {
    int64_t n_nodes = 0;
    std::vector<int64_t> src, dst, eid;
    std::vector<int64_t> off, nbr, ids, scratch;        // forward CSR
    std::vector<int64_t> roff, rnbr, rids, rscratch;    // reverse CSR
    std::vector<ReachExpect> reach;
    std::vector<PathsExpect> paths;
    std::vector<SpExpect>    sp;
    bool loaded = false;

    bolt::kernels::CsrBfsGraph fwd() const {
        bolt::kernels::CsrBfsGraph g{};
        g.off = off.data(); g.neighbors = nbr.data(); g.edge_ids = ids.data();
        g.edge_labels = nullptr; g.n_nodes = n_nodes; g.want_label = -1;
        return g;
    }
    bolt::kernels::CsrBfsGraph rev() const {
        bolt::kernels::CsrBfsGraph g{};
        g.off = roff.data(); g.neighbors = rnbr.data(); g.edge_ids = rids.data();
        g.edge_labels = nullptr; g.n_nodes = n_nodes; g.want_label = -1;
        return g;
    }
};

// Reads the generated file. Returns false (and leaves `loaded` false) when the
// file is missing — the caller must FAIL rather than skip, so a lost fixture
// can never become a silently vacuous pass.
inline bool load_random_graph(const char* path, RandomGraph* out) {
    std::ifstream in(path);
    if (!in.good()) return false;
    std::string tag;
    int64_t n_edges = 0;
    while (in >> tag) {
        if (tag[0] == '#') { std::getline(in, tag); continue; }
        if (tag == "NODES") { in >> out->n_nodes; continue; }
        if (tag == "EDGES") { in >> n_edges; continue; }
        if (tag == "E") {
            int64_t s = 0, d = 0;
            in >> s >> d;
            out->eid.push_back(static_cast<int64_t>(out->src.size()));
            out->src.push_back(s);
            out->dst.push_back(d);
            continue;
        }
        if (tag == "REACH") { ReachExpect r{}; in >> r.src >> r.hops >> r.count;
                              out->reach.push_back(r); continue; }
        if (tag == "PATHS") { PathsExpect p{};
                              in >> p.src >> p.hops >> p.walk >> p.trail >> p.simple;
                              out->paths.push_back(p); continue; }
        if (tag == "SP")    { SpExpect s{}; in >> s.s >> s.t >> s.len;
                              out->sp.push_back(s); continue; }
        std::getline(in, tag);   // unknown line: skip
    }
    if (out->n_nodes <= 0 || out->src.empty()) return false;
    if (static_cast<int64_t>(out->src.size()) != n_edges) return false;

    const int64_t n = out->n_nodes, m = static_cast<int64_t>(out->src.size());
    out->off.assign(static_cast<size_t>(n + 1), 0);
    out->nbr.assign(static_cast<size_t>(m), 0);
    out->ids.assign(static_cast<size_t>(m), 0);
    out->scratch.assign(static_cast<size_t>(n), 0);
    out->roff.assign(static_cast<size_t>(n + 1), 0);
    out->rnbr.assign(static_cast<size_t>(m), 0);
    out->rids.assign(static_cast<size_t>(m), 0);
    out->rscratch.assign(static_cast<size_t>(n), 0);
    if (!bolt::kernels::csr_build(out->src.data(), out->dst.data(),
                                  out->eid.data(), m, n, out->off.data(),
                                  out->nbr.data(), out->ids.data(),
                                  out->scratch.data()))
        return false;
    // Reverse CSR: same edge set, same edge ids, endpoints swapped.
    if (!bolt::kernels::csr_build(out->dst.data(), out->src.data(),
                                  out->eid.data(), m, n, out->roff.data(),
                                  out->rnbr.data(), out->rids.data(),
                                  out->rscratch.data()))
        return false;
    out->loaded = true;
    return true;
}

inline std::string random_graph_path() {
#ifdef BOLT_TEST_DATA_DIR
    return std::string(BOLT_TEST_DATA_DIR) + "/csr_bfs_random.txt";
#else
    return std::string("data/csr_bfs_random.txt");
#endif
}
