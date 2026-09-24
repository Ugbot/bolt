#!/usr/bin/env python3
"""G2ICE-163 oracle: read the table that four concurrent *processes* committed
to (tests/test_bolt_iceberg_concurrent_commit.cpp, RetryingProcessesLoseNothing)
with pyiceberg AND DuckDB, and compare against the generating rule re-derived
here — never against anything bolt emitted:

    writer w in [0,4), commit c in [0,12), row r in [0,4):
        id = w * 1_000_000 + c * 1000 + r

Every one of the 48 commits reported success after retrying on conflict, so
the table must hold all 192 ids exactly once, in 48 snapshots whose parent
chain is linear.

    python3 scripts/iceberg_concurrent_commit_oracle.py <table_dir>

Exit 0 = pass, 1 = mismatch, 2 = neither oracle installed (SKIP).
"""
import glob
import os
import re
import sys

WRITERS, COMMITS, ROWS = 4, 12, 4


def model():
    return sorted(w * 1_000_000 + c * 1000 + r
                  for w in range(WRITERS) for c in range(COMMITS) for r in range(ROWS))


def latest_metadata(root):
    best, best_v = None, -1
    for p in glob.glob(os.path.join(root, "metadata", "v*.metadata.json")):
        m = re.search(r"/v(\d+)\.metadata\.json$", p)
        if m and int(m.group(1)) > best_v:
            best, best_v = p, int(m.group(1))
    return best


def compare(name, got, errs):
    want = model()
    if sorted(got) != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        errs.append("%s: %d ids (want %d); missing %r extra %r"
                    % (name, len(got), len(want), missing[:8], extra[:8]))


def check_pyiceberg(root, errs, mutate=None):
    from pyiceberg.table import StaticTable
    t = StaticTable.from_metadata("file://" + latest_metadata(root))
    snaps = t.metadata.snapshots
    if len(snaps) != WRITERS * COMMITS:
        errs.append("pyiceberg: %d snapshots, want %d" % (len(snaps), WRITERS * COMMITS))
    ids = {s.snapshot_id for s in snaps}
    if len(ids) != len(snaps):
        errs.append("pyiceberg: duplicate snapshot ids")
    # Linear history: exactly one root, every other parent is a real snapshot,
    # and no snapshot has two children (a fork = a lost update).
    parents = [s.parent_snapshot_id for s in snaps]
    if sum(1 for p in parents if p is None) != 1:
        errs.append("pyiceberg: history does not have exactly one root")
    if any(p is not None and p not in ids for p in parents):
        errs.append("pyiceberg: a parent snapshot id names no snapshot")
    if len([p for p in parents if p is not None]) != len({p for p in parents if p is not None}):
        errs.append("pyiceberg: forked history (two commits on one parent)")
    got = t.scan().to_arrow().column("id").to_pylist()
    if mutate is not None:
        got = mutate(got)
    compare("pyiceberg", got, errs)


def check_duckdb(root, errs, mutate=None):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL iceberg; LOAD iceberg;")
    got = [r[0] for r in con.execute(
        "SELECT id FROM iceberg_scan('%s')" % latest_metadata(root)).fetchall()]
    if mutate is not None:
        got = mutate(got)
    compare("duckdb", got, errs)


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    if latest_metadata(root) is None:
        print("FAIL: no vN.metadata.json under %s/metadata" % root)
        return 1
    oracles = []
    try:
        import pyiceberg  # noqa: F401
        oracles.append(("pyiceberg", check_pyiceberg))
    except ImportError:
        print("note: pyiceberg not installed")
    try:
        import duckdb  # noqa: F401
        oracles.append(("duckdb", check_duckdb))
    except ImportError:
        print("note: duckdb not installed")
    if not oracles:
        print("SKIP: neither pyiceberg nor duckdb installed")
        return 2

    errs = []
    for _, fn in oracles:
        try:
            fn(root, errs)
        except Exception as e:  # an unreadable table is a failure, not a skip
            errs.append("%s raised: %s" % (fn.__name__, str(e).splitlines()[0][:200]))
    if errs:
        print("FAIL: %s" % root)
        for e in errs:
            print("  " + e)
        return 1

    # Discriminating power: a lost commit and a duplicated commit must both
    # be caught by every oracle.
    for label, mut in [("one commit lost", lambda g: sorted(g)[ROWS:]),
                       ("one commit duplicated", lambda g: list(g) + sorted(g)[:ROWS])]:
        for name, fn in oracles:
            e = []
            fn(root, e, mutate=mut)
            if not e:
                print("FAIL: %s did not catch injection: %s" % (name, label))
                return 1

    print("PASS: %s read all %d ids from %d concurrent-process commits, exact"
          % ("+".join(n for n, _ in oracles), len(model()), WRITERS * COMMITS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
