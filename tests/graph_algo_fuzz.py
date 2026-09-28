#!/usr/bin/env python3
"""Seeded random-graph property fuzz for bolt's CSR algorithm kernels.

Generates small random multigraphs (empty edge sets, self-loops, parallel
edges, disconnected pieces, two relationship labels), feeds them to
graph_algo_fuzz_harness, and checks every answer against an engine that
shares no code with bolt:

  degree, pagerank, wcc, scc, dijkstra   networkx (MultiDiGraph)
  jaccard / overlap                      set arithmetic on successor sets
  walk / trail / simple / reach paths    a brute-force path enumerator

Replay one case with --seed S --cases 1 --first K (the failing line prints
the case's own seed). Exit 0 = all agree, 1 = disagreement, 2 = setup error.

  ctest:     graph_algo_fuzz.py --harness H --seed 20260928 --cases 200
  campaign:  /tmp/g2locks/with-lock fuzz 300 -- \
             graph_algo_fuzz.py --harness H --seed $RANDOM --cases 20000
"""

import argparse
import math
import random
import subprocess
import sys
from collections import Counter

try:
    import networkx as nx
except ImportError:  # the oracle is mandatory; a missing one is not a pass
    print("graph_algo_fuzz: networkx not installed", file=sys.stderr)
    sys.exit(2)


def gen_case(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    shape = rng.random()
    if shape < 0.1:
        m = 0
    elif shape < 0.2:
        m = rng.randint(1, 3)
    else:
        m = rng.randint(1, 18)
    edges = []
    for _ in range(m):
        u = rng.randrange(n)
        r = rng.random()
        if r < 0.12:
            v = u                                   # self-loop
        elif r < 0.25 and edges:
            u, v = edges[rng.randrange(len(edges))][:2]   # parallel edge
        else:
            v = rng.randrange(min(n, u + 3) if rng.random() < 0.3 else n)
        lbl = rng.randint(0, 1)
        w = rng.choice([0.0, 0.5, 1.0, 2.0, 3.25, 7.0, rng.randint(0, 20) / 4])
        edges.append((u, v, lbl, w))
    want = rng.choice([-1, -1, 0, 1])
    return {"n": n, "edges": edges, "want": want,
            "sorted": rng.randint(0, 1), "max_hops": rng.randint(0, 4)}


def kept(case):
    return [e for e in case["edges"] if case["want"] < 0 or e[2] == case["want"]]


def multidigraph(case):
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(case["n"]))
    for u, v, _, w in kept(case):
        g.add_edge(u, v, weight=w)
    return g


def pagerank_oracle(g, **kw):
    try:
        import scipy  # noqa: F401
        return nx.pagerank(g, **kw)
    except ImportError:  # networkx's pure-Python power iteration
        from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python
        return _pagerank_python(g, **kw)


def comps_from_sets(n, sets):
    out = [0] * n
    for s in sets:
        mn = min(s)
        for x in s:
            out[x] = mn
    return out


def enum_paths(case, sem, lo, hi):
    """Brute force: every path from every source, per semantics."""
    n = case["n"]
    es = [(i, e) for i, e in enumerate(case["edges"])
          if case["want"] < 0 or e[2] == case["want"]]
    adj = {u: [] for u in range(n)}
    for i, (u, v, _, _) in es:
        adj[u].append((v, i))
    rows = Counter()
    if sem == "reach":
        # Set semantics: bounded = the distinct endpoints of the walks;
        # unbounded = networkx descendants, the start only via a real cycle.
        if hi >= 0:
            return Counter(set(enum_paths(case, "walk", lo, hi)))
        g = multidigraph(case)
        for s in range(n):
            desc = nx.descendants(g, s)
            back = any(p == s or p in desc for p in g.predecessors(s))
            for d in desc | ({s} if lo == 0 or back else set()):
                rows[(s, d)] += 1
        return rows

    def dfs(s, node, depth, nodes, eids):
        if depth >= lo:
            rows[(s, node)] += 1
        if depth == hi:
            return
        for y, i in adj[node]:
            if sem == "trail" and i in eids:
                continue
            if sem == "simple" and y in nodes:
                continue
            dfs(s, y, depth + 1, nodes | {y}, eids | {i})

    for s in range(n):
        dfs(s, s, 0, frozenset([s]), frozenset())
    return rows


def parse_output(text):
    cases, cur = [], None
    for line in text.splitlines():
        if line.startswith("MISMATCH"):
            cur.setdefault("mismatch", []).append(line)
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "case":
            cur = {"id": int(parts[1]), "dj": {}, "paths": {}}
        elif parts[0] == "end":
            cases.append(cur)
        elif parts[0] == "dj":
            cur["dj"][int(parts[1])] = (int(parts[2]), [float(x) for x in parts[3:]])
        elif parts[0] == "path":
            key = (parts[1], int(parts[2]), int(parts[3]))
            rows = Counter(tuple(map(int, r.split(":"))) for r in parts[5:])
            cur["paths"][key] = (int(parts[4]), rows)
        elif parts[0] in ("pr", "jac", "ovl"):
            cur[parts[0]] = [float(x) for x in parts[1:]]
        else:
            cur[parts[0]] = [int(x) for x in parts[1:]]
    return cases


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def check(case, got):
    errs = list(got.get("mismatch", []))
    g = multidigraph(case)
    n = case["n"]
    if got["outdeg"] != [g.out_degree(i) for i in range(n)]:
        errs.append(f"outdeg {got['outdeg']}")
    if got["indeg"] != [g.in_degree(i) for i in range(n)]:
        errs.append(f"indeg {got['indeg']}")
    wcc = comps_from_sets(n, nx.weakly_connected_components(g))
    scc = comps_from_sets(n, nx.strongly_connected_components(g))
    if got["wcc"] != [len(set(wcc))] + wcc:
        errs.append(f"wcc {got['wcc']} want {wcc}")
    if got["scc"] != [len(set(scc))] + scc:
        errs.append(f"scc {got['scc']} want {scc}")
    pr = pagerank_oracle(g, alpha=0.85, tol=1e-13, max_iter=10000, weight=None)
    if got["pr"][0] <= 0 or any(not close(got["pr"][1 + i], pr[i], 1e-9)
                                for i in range(n)):
        errs.append(f"pagerank {got['pr']} want {[pr[i] for i in range(n)]}")
    for s in range(n):
        st, dist = got["dj"][s]
        want = nx.single_source_dijkstra_path_length(g, s, weight="weight")
        exp = [want.get(i, math.inf) for i in range(n)]
        if st != 0 or any(not (d == e or close(d, e, 1e-12))
                          for d, e in zip(dist, exp)):
            errs.append(f"dijkstra from {s}: {st} {dist} want {exp}")
    succ = [set(v for _, v in g.out_edges(i)) for i in range(n)]
    k = 0
    for a in range(n):
        for b in range(n):
            i = len(succ[a] & succ[b])
            un = len(succ[a] | succ[b])
            mn = min(len(succ[a]), len(succ[b]))
            ej = i / un if un else 0.0
            eo = i / mn if mn else 0.0
            if not close(got["jac"][k], ej, 1e-15) or not close(got["ovl"][k], eo, 1e-15):
                errs.append(f"similarity ({a},{b}) {got['jac'][k]} {got['ovl'][k]}")
            k += 1
    for (sem, lo, hi), (st, rows) in got["paths"].items():
        want = enum_paths(case, sem, lo, hi)
        if st != 0 or rows != want:
            errs.append(f"paths {sem} lo={lo} hi={hi} st={st}: "
                        f"got-want={dict(rows - want)} want-got={dict(want - rows)}")
    return errs


def render(cid, case):
    lines = [f"case {cid} {case['n']} {len(case['edges'])} {case['want']} "
             f"{case['sorted']} {case['max_hops']}"]
    lines += [f"{u} {v} {l} {w!r}" for u, v, l, w in case["edges"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()
    bad = 0
    for b0 in range(args.first, args.first + args.cases, args.batch):
        ids = range(b0, min(b0 + args.batch, args.first + args.cases))
        cases = {i: gen_case(args.seed * 1_000_003 + i) for i in ids}
        text = "".join(render(i, cases[i]) for i in ids)
        proc = subprocess.run([args.harness], input=text, capture_output=True,
                              text=True, timeout=600)
        if proc.returncode != 0:
            print(f"harness exit {proc.returncode}: {proc.stderr[-2000:]}")
            return 1
        out = parse_output(proc.stdout)
        if len(out) != len(cases):
            print(f"harness answered {len(out)} of {len(cases)} cases")
            return 1
        for got in out:
            errs = check(cases[got["id"]], got)
            if errs:
                bad += 1
                c = cases[got["id"]]
                print(f"FAIL seed={args.seed} first={got['id']} n={c['n']} "
                      f"want={c['want']} sorted={c['sorted']} hops={c['max_hops']} "
                      f"edges={c['edges']}")
                for e in errs[:6]:
                    print("   ", e)
                if bad >= 5:
                    return 1
    print(f"graph_algo_fuzz: seed={args.seed} cases={args.cases} failures={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
