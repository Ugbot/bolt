#!/usr/bin/env python3
"""Cross-check: does bolt_csr_bfs.h's WALK semantics match the SHIPPED chukonu
operator it is meant to replace?

W1-L1 lands the kernel; W2-L1 rewires `PhysicalBFSExpand` onto it. A kernel
that is internally consistent but disagrees with the operator would turn that
rewire into a silent answer change, so the agreement is MEASURED here rather
than argued from a side-by-side read of the two sources.

Three parties, one number per cell:
  1. csr_bfs_enumerate.py — the pure-Python re-derivation the bolt kernel's
     unit-test literals come from (so this stands in for the kernel).
  2. `chukonu_cli` running real Cypher `-[:R*lo..hi]->` over the SAME graph
     loaded through `.graph` — the shipped engine, unmodified.
  3. Five graph VARIANTS x six hop windows, so a compensating error in one
     structural feature (self-loop, parallel edge, cycle) cannot survive:
     removing each feature must move the expected totals, and it does.

Not part of ctest: it needs a built chukonu_cli with the Cypher frontend on.

    python3 extern/bolt/tests/csr_bfs_vs_chukonu_cli.py [path/to/chukonu_cli]
"""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csr_bfs_enumerate as E   # noqa: E402

# The KnownBfsCsr fixture, plus four structural variants.
BASE = [(0, 100, 1), (0, 101, 1), (1, 102, 2),
        (2, 103, 0), (2, 105, 4), (3, 104, 3)]
VARIANTS = {
    "base":        BASE,
    "no_selfloop": [e for e in BASE if e != (3, 104, 3)],
    "no_parallel": [e for e in BASE if e != (0, 101, 1)],
    "no_cycle":    [e for e in BASE if e != (2, 103, 0)],
    "extra_loop":  BASE + [(2, 106, 2)],
}
WINDOWS = [(1, 1), (0, 1), (1, 2), (1, 3), (0, 3), (2, 3)]

DEFAULT_CLI = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../build/coverage/extern/chukonu/chukonu_cli")


def engine_counts(cli, workdir, edges):
    """Run every hop window through chukonu_cli's Cypher frontend."""
    epath = os.path.join(workdir, "ev.csv")
    with open(epath, "w") as fh:
        fh.write("src,dst\n" + "".join("%d,%d\n" % (s, d) for (s, _e, d) in edges))
    script = ".graph ev.csv NODE n FROM n.csv EDGE R\n.lang cypher\n"
    script += "".join("MATCH (a:n)-[:R*%d..%d]->(b:n) RETURN count(*);\n" % w
                      for w in WINDOWS)
    out = subprocess.run([cli, "--repl"], input=script, capture_output=True,
                         text=True, cwd=workdir).stdout
    return [int(l.strip().strip("│ ")) for l in out.splitlines()
            if l.strip().startswith("│") and l.strip().strip("│ ").isdigit()]


def main(argv):
    cli = argv[1] if len(argv) > 1 else DEFAULT_CLI
    if not os.path.exists(cli):
        print("chukonu_cli not found at %s — build it first" % cli)
        return 2
    workdir = tempfile.mkdtemp(prefix="csr_bfs_xcheck_")
    with open(os.path.join(workdir, "n.csv"), "w") as fh:
        fh.write("id\n0\n1\n2\n3\n4\n")

    agree = disagree = 0
    for name, edges in VARIANTS.items():
        adj = E.adjacency(edges, 5)
        want = [sum(len(E.expand(adj, s, lo, hi, E.WALK)) for s in range(5))
                for (lo, hi) in WINDOWS]
        got = engine_counts(cli, workdir, edges)
        ok = (got == want)
        agree += ok
        disagree += (not ok)
        print("%-12s windows %s" % (name, WINDOWS))
        print("             enumerator (== bolt kernel pins) %s" % want)
        print("             chukonu engine                   %s   %s"
              % (got, "OK" if ok else "MISMATCH"))
    print("\n%d/%d variants agree" % (agree, agree + disagree))
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
