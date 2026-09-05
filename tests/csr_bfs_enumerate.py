#!/usr/bin/env python3
"""Independent oracle for bolt's CSR traversal kernels (bolt_csr_bfs.h /
bolt_csr_shortest.h).

Two jobs, both about PROVENANCE — the C++ pins and their expected values must
not both come from one author reading one engine:

  1. `python3 csr_bfs_enumerate.py`
     Re-derives, from the graph definition alone, every literal the
     deterministic C++ pins assert: the exact DFS pre-order (src, dst, hops)
     rows for WALK / TRAIL / SIMPLE at several hop windows, plus shortest-path
     lengths. Print it and diff against test_bolt_csr_bfs.cpp.

  2. `python3 csr_bfs_enumerate.py --emit-random <path>`
     Writes data/csr_bfs_random.txt: a deterministic 1,000-node / 10,000-edge
     random graph plus expected reachability, path counts and shortest-path
     lengths. Reachability and shortest paths are taken from **networkx**, a
     third-party engine that shares no code with bolt — the engine-independent
     re-derivation the oracle law requires. Path COUNTS under WALK/TRAIL/SIMPLE
     have no networkx equivalent (networkx enumerates simple paths only), so
     they come from the pure-Python DFS below, which is written from the
     semantics definition rather than from bolt's implementation.

The C++ test reads the emitted file; it never re-derives anything itself.

Follows the tests/cypher/graph_fixture_enumerate.py pattern.
"""

import sys

# ---------------------------------------------------------------------------
# The deterministic fixture (mirrors KnownBfsCsr in test_bolt_csr_bfs.cpp)
# ---------------------------------------------------------------------------
# Node ids 0..4, edge ids offset by 100 so no edge id can collide with a node
# id. Edges are listed in CSR insertion order: grouped by source, and within a
# source in the order csr_build would scatter them.
#
#   0 -100-> 1     parallel with 101: separates TRAIL from SIMPLE
#   0 -101-> 1
#   1 -102-> 2
#   2 -103-> 0     closes the 0->1->2->0 cycle: separates WALK from SIMPLE
#   2 -105-> 4
#   3 -104-> 3     self-loop: the shape SIMPLE wrongly dropped (G2GRAPH-L3a)
NODES = 5
EDGES = [(0, 100, 1), (0, 101, 1), (1, 102, 2),
         (2, 103, 0), (2, 105, 4), (3, 104, 3)]
LABELS = {100: 7, 101: 9, 102: 7, 103: 7, 105: 7, 104: 7}

WALK, TRAIL, SIMPLE = 0, 1, 2
SEM_NAME = {WALK: "WALK", TRAIL: "TRAIL", SIMPLE: "SIMPLE"}


def adjacency(edges, n_nodes):
    """CSR-order adjacency: adj[u] is the list of (edge_id, dst) out of u, in
    the order csr_build places them (stable by input index)."""
    adj = [[] for _ in range(n_nodes)]
    for (s, e, d) in edges:
        adj[s].append((e, d))
    return adj


def expand(adj, src, min_hops, max_hops, semantics, want_label=None,
           labels=None):
    """Every (src, endpoint, hops) row bolt_csr_bfs.h's csr_bfs_expand emits,
    in the SAME order: depth-first pre-order, neighbours in CSR order.

    Written from the semantics definition:
      WALK   forbids nothing
      TRAIL  forbids reusing a relationship already on the current path
      SIMPLE forbids revisiting a node already on the current path
    """
    out = []

    def step(node, depth, path, epath):
        if depth >= min_hops:
            out.append((src, node, depth))
        if depth >= max_hops:
            return
        for (eid, nb) in adj[node]:
            if want_label is not None and labels is not None:
                if labels[eid] != want_label:
                    continue
            if semantics == TRAIL and eid in epath:
                continue
            if semantics == SIMPLE and nb in path:
                continue
            step(nb, depth + 1, path + [nb], epath + [eid])

    step(src, 0, [src], [])
    return out


def bfs_layers(adj, src, max_hops):
    """{node: distance} for every node within max_hops of src (src included).
    Plain textbook BFS — no bolt code involved."""
    dist = {src: 0}
    frontier = [src]
    d = 0
    while frontier and d < max_hops:
        d += 1
        nxt = []
        for u in frontier:
            for (_e, v) in adj[u]:
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def shortest_len(adj, s, t, max_hops):
    """Hop distance s->t, or -1 if unreachable within max_hops."""
    if s == t:
        return 0
    dist = bfs_layers(adj, s, max_hops)
    return dist.get(t, -1)


# ---------------------------------------------------------------------------
# 1. the deterministic fixture report
# ---------------------------------------------------------------------------

def report_fixture():
    adj = adjacency(EDGES, NODES)
    print("fixture: %d nodes, %d edges" % (NODES, len(EDGES)))
    print("  CSR adjacency (insertion order):")
    for u in range(NODES):
        print("    %d -> %s" % (u, adj[u]))
    print()

    for sem in (WALK, TRAIL, SIMPLE):
        for (lo, hi) in ((0, 1), (1, 2), (1, 3), (0, 0)):
            for src in range(NODES):
                rows = expand(adj, src, lo, hi, sem)
                if not rows:
                    continue
                print("%-6s [*%d..%d] src=%d -> %d rows: %s"
                      % (SEM_NAME[sem], lo, hi, src, len(rows), rows))
        print()

    print("label-filtered (want_label=7) WALK [*1..2]:")
    for src in range(NODES):
        rows = expand(adj, src, 1, 2, WALK, want_label=7, labels=LABELS)
        print("    src=%d -> %d rows: %s" % (src, len(rows), rows))
    print()

    print("csr_bfs_count == len(expand) per source, WALK [*1..2]:")
    print("    %s" % [len(expand(adj, s, 1, 2, WALK)) for s in range(NODES)])
    print()

    print("shortest path lengths (max_hops=8), -1 = unreachable:")
    for s in range(NODES):
        print("    from %d: %s"
              % (s, [shortest_len(adj, s, t, 8) for t in range(NODES)]))


# ---------------------------------------------------------------------------
# 2. the randomized graph + networkx cross-check
# ---------------------------------------------------------------------------

RAND_NODES = 1000
RAND_EDGES = 10000
RAND_SEED = 0x5EED1234
# Node ids [RAND_CONNECTED, RAND_NODES) are ISOLATED — no edge touches them.
# Without them 10k edges over 1k nodes give a strongly connected graph, so no
# SP pair would ever be unreachable and the -1 branch would go untested.
RAND_CONNECTED = 980


def lcg(seed):
    """Deterministic 64-bit LCG (Knuth MMIX constants). Reproducible on any
    Python; the C++ side never generates the graph, it reads the file."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) \
            & 0xFFFFFFFFFFFFFFFF
        yield state >> 11


def build_random():
    rnd = lcg(RAND_SEED)
    raw = []
    for _ in range(RAND_EDGES):
        s = next(rnd) % RAND_CONNECTED
        d = next(rnd) % RAND_CONNECTED
        raw.append((s, d))
    # Stable sort by source: this IS the CSR insertion order, so the C++ side
    # can hand the file straight to csr_build and get the same layout.
    raw.sort(key=lambda p: p[0])
    return [(s, i, d) for i, (s, d) in enumerate(raw)]


def emit_random(path):
    import networkx as nx

    edges = build_random()
    adj = adjacency(edges, RAND_NODES)

    g = nx.MultiDiGraph()
    g.add_nodes_from(range(RAND_NODES))
    for (s, _e, d) in edges:
        g.add_edge(s, d)

    rnd = lcg(RAND_SEED ^ 0xABCDEF)
    reach_srcs = sorted({next(rnd) % RAND_NODES for _ in range(24)})
    path_srcs = sorted({next(rnd) % RAND_NODES for _ in range(12)})
    pairs = [(next(rnd) % RAND_NODES, next(rnd) % RAND_NODES)
             for _ in range(300)]

    lines = []
    lines.append("# csr_bfs_random.txt - GENERATED by tests/csr_bfs_enumerate.py")
    lines.append("# oracle: networkx %s (REACH, SP) + this file's own DFS (PATHS)"
                 % nx.__version__)
    lines.append("# do not hand-edit; regenerate with --emit-random")
    lines.append("NODES %d" % RAND_NODES)
    lines.append("EDGES %d" % len(edges))
    for (s, _e, d) in edges:
        lines.append("E %d %d" % (s, d))

    # REACH: |{v : dist(src, v) <= h}| including src itself. networkx.
    for src in reach_srcs:
        for h in (1, 2, 3):
            got = nx.single_source_shortest_path_length(g, src, cutoff=h)
            lines.append("REACH %d %d %d" % (src, h, len(got)))

    # PATHS: WALK / TRAIL / SIMPLE path counts for hop-length in [1, h].
    for src in path_srcs:
        for h in (2, 3):
            w = len(expand(adj, src, 1, h, WALK))
            t = len(expand(adj, src, 1, h, TRAIL))
            s2 = len(expand(adj, src, 1, h, SIMPLE))
            lines.append("PATHS %d %d %d %d %d" % (src, h, w, t, s2))

    # SP: shortest path length, -1 unreachable. networkx.
    for (a, b) in pairs:
        try:
            ln = nx.shortest_path_length(g, a, b)
        except nx.NetworkXNoPath:
            ln = -1
        lines.append("SP %d %d %d" % (a, b, ln))

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s: %d nodes, %d edges, %d REACH, %d PATHS, %d SP"
          % (path, RAND_NODES, len(edges), len(reach_srcs) * 3,
             len(path_srcs) * 2, len(pairs)))


def main(argv):
    if len(argv) >= 3 and argv[1] == "--emit-random":
        emit_random(argv[2])
        return 0
    sys.setrecursionlimit(10000)
    report_fixture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
