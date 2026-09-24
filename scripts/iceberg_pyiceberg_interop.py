#!/usr/bin/env python3
"""Check bolt's Iceberg *writer* against pyiceberg -- the half bolt cannot check.

    ./test_bolt_iceberg_pyiceberg_interop        # writes the fixture
    python3 scripts/iceberg_pyiceberg_interop.py [dir]

WHY THIS EXISTS (G2ICE-84). bolt has both an Iceberg writer and an Iceberg
reader, and every write test before this one closed the loop through bolt's own
reader. That proves almost nothing about the TABLE, and here it proved less
than usual: `iceberg_scan.cpp`'s `read_ref` tries THREE rebasings of every
recorded path in turn, so it resolves a manifest list whether the writer
recorded `metadata/snap-1.avro` or `/warehouse/t/metadata/snap-1.avro`. A
reader that forgiving structurally cannot detect a writer recording the wrong
thing. pyiceberg resolves the string literally.

Three defects this script found on bolt's committed writer, none of which
bolt's own reader could see:

  1. Every recorded location was RELATIVE, so pyiceberg opened the manifest
     list under its own working directory:
         FileNotFoundError: /private/tmp/metadata/snap-1788...avro
     for a table living in /tmp/ice84_fix.
  2. No field ids and no name mapping:
         ValueError: Parquet file does not have field-ids and the Iceberg
         table does not have 'schema.name-mapping.default' defined
  3. Each snapshot's manifest list held only the manifest that commit ADDED,
     so an append REPLACED the table: two 6-row commits read back as 6 rows.
     bolt's reader followed the same one-entry list and agreed, and the write
     tests asserted snapshot COUNTS, so nothing in-tree could see it.

The expected values are re-derived here from the generating rules in
tests/test_bolt_iceberg_pyiceberg_interop.cpp, never read out of anything bolt
emitted:

    12 rows written as two 6-row commits; for row i (0-based)
        id    = 1000 + i
        score = i * 0.5 - 3.0

Exit 0 = pass, 1 = a real mismatch, 2 = the oracle is not installed (SKIP).
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile

N_ROWS = 12
# G2ICE-82: after the two appends the fixture commits an Iceberg v2 POSITIONAL
# delete for this 0-based row. It must be absent from what pyiceberg reads.
# Re-derived here from the same rule the C++ side states, never read back out
# of anything bolt emitted.
DELETED_ROW = 7


def model():
    """The expected table, from the generating rules alone."""
    live = [i for i in range(N_ROWS) if i != DELETED_ROW]
    return {
        "id": [1000 + i for i in live],
        "score": [i * 0.5 - 3.0 for i in live],
    }


def latest_metadata(root):
    """The newest vN.metadata.json, ranked by N (not by mtime)."""
    best, best_v = None, -1
    for p in glob.glob(os.path.join(root, "metadata", "v*.metadata.json")):
        m = re.search(r"v(\d+)\.metadata\.json$", p)
        if m and int(m.group(1)) > best_v:
            best, best_v = p, int(m.group(1))
    return best


# G2ICE-6: live row count after each snapshot, from the generating rules --
# two 6-row appends, then the positional delete.
SNAPSHOT_ROWS = [6, 12, 11]


def check_history(root, meta):
    """snapshot-log / metadata-log, read through pyiceberg's own model and used
    for time travel by pyiceberg and DuckDB. Returns [] or errors."""
    from pyiceberg.table import StaticTable

    errs = []
    table = StaticTable.from_metadata(meta)
    md = table.metadata
    snaps = [s.snapshot_id for s in md.snapshots]
    hist = table.history()
    if len(hist) != len(SNAPSHOT_ROWS):
        errs.append("snapshot-log has %d entries, expected %d"
                    % (len(hist), len(SNAPSHOT_ROWS)))
        return errs
    if [h.snapshot_id for h in hist] != snaps:
        errs.append("snapshot-log order %r != commit order %r"
                    % ([h.snapshot_id for h in hist], snaps))
    ts = [h.timestamp_ms for h in hist]
    for i, h in enumerate(hist):
        # Ties at one millisecond resolve to the newest snapshot at that ms.
        j = max(k for k in range(len(ts)) if ts[k] == h.timestamp_ms)
        s = table.snapshot_as_of_timestamp(h.timestamp_ms)
        if s is None:
            errs.append("snapshot_as_of_timestamp(%d) found nothing"
                        % h.timestamp_ms)
            continue
        n = table.scan(snapshot_id=s.snapshot_id).to_arrow().num_rows
        if n != SNAPSHOT_ROWS[j]:
            errs.append("as-of %d: %d rows, expected %d"
                        % (h.timestamp_ms, n, SNAPSHOT_ROWS[j]))

    latest_v = int(re.search(r"v(\d+)\.metadata\.json$", meta).group(1))
    mlog = md.metadata_log
    if len(mlog) != latest_v - 1:
        errs.append("metadata-log has %d entries, expected %d (v1..v%d)"
                    % (len(mlog), latest_v - 1, latest_v - 1))
        return errs
    prev_ts = 0
    for k, e in enumerate(mlog, start=1):
        want = os.path.join(root, "metadata", "v%d.metadata.json" % k)
        if os.path.normpath(e.metadata_file) != os.path.normpath(want):
            errs.append("metadata-log[%d] = %s, expected %s"
                        % (k - 1, e.metadata_file, want))
            continue
        if not os.path.exists(want):
            errs.append("metadata-log names a missing file: %s" % want)
            continue
        with open(want) as f:
            own = json.load(f)["last-updated-ms"]
        if e.timestamp_ms != own:
            errs.append("metadata-log[%d] timestamp %d != that file's "
                        "last-updated-ms %d" % (k - 1, e.timestamp_ms, own))
        if e.timestamp_ms < prev_ts:
            errs.append("metadata-log not in time order at %d" % (k - 1))
        prev_ts = e.timestamp_ms
    if prev_ts > md.last_updated_ms:
        errs.append("metadata-log newer than last-updated-ms")
    return errs


def check_duckdb_history(meta):
    """DuckDB time travel to every snapshot. None = not installed."""
    try:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL iceberg; LOAD iceberg;")
    except Exception:
        return None
    with open(meta) as f:
        snaps = [s["snapshot-id"] for s in json.load(f)["snapshots"]]
    errs = []
    for i, sid in enumerate(snaps):
        n = con.execute("SELECT count(*) FROM iceberg_scan('%s', "
                        "snapshot_from_id=%d)" % (meta, sid)).fetchone()[0]
        if n != SNAPSHOT_ROWS[i]:
            errs.append("duckdb snapshot %d: %d rows, expected %d"
                        % (sid, n, SNAPSHOT_ROWS[i]))
    return errs


def history_stripped(root, meta):
    """The pre-G2ICE-6 shape: no snapshot-log, no metadata-log."""
    with open(meta) as f:
        doc = json.load(f)
    doc.pop("snapshot-log", None)
    doc.pop("metadata-log", None)
    tmp = tempfile.mkdtemp(prefix="bolt_ice_hist_")
    path = os.path.join(tmp, os.path.basename(meta))
    with open(path, "w") as f:
        json.dump(doc, f)
    return tmp, path


def check(root, mutate=None):
    """Read `root` with pyiceberg and compare to the model. Returns [] or errors."""
    from pyiceberg.table import StaticTable

    meta = latest_metadata(root)
    if meta is None:
        return ["no vN.metadata.json under %s/metadata" % root]
    table = StaticTable.from_metadata(meta)

    errs = []
    # The locations must be absolute. pyiceberg would resolve a relative one
    # against ITS cwd, so a table that happens to be scanned from its own
    # directory would pass on values while still being unreadable anywhere
    # else -- exactly the bug, invisible to a values-only check.
    for snap in table.metadata.snapshots:
        if not os.path.isabs(snap.manifest_list) and "://" not in snap.manifest_list:
            errs.append("manifest-list is relative: %s" % snap.manifest_list)

    got = table.scan().to_arrow().to_pydict()
    # The deleted row must be gone, and gone because the DELETE was applied --
    # not because the whole file it lived in vanished. Checked explicitly, and
    # ahead of the multiset comparison, so a failure names the real cause.
    if (1000 + DELETED_ROW) in (got.get("id") or []):
        errs.append("RESURRECTION: id=%d was positionally deleted but "
                    "pyiceberg still returns it" % (1000 + DELETED_ROW))
    if mutate is not None:
        got = mutate(got)
    want = model()

    for col, expected in want.items():
        if col not in got:
            errs.append("missing column %r (have %r)" % (col, sorted(got)))
            continue
        actual = got[col]
        if len(actual) != len(expected):
            errs.append("%s: %d rows, expected %d" % (col, len(actual), len(expected)))
            continue
        # Order is not guaranteed across data files, so compare as multisets.
        if sorted(actual) != sorted(expected):
            errs.append("%s: values differ\n  got  %r\n  want %r"
                        % (col, actual, expected))
    return errs


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "BOLT_ICEBERG_INTEROP_DIR", "/tmp/bolt_iceberg_interop")
    try:
        import pyiceberg  # noqa: F401
    except ImportError:
        print("SKIP: pyiceberg not installed (pip install pyiceberg)")
        return 2

    errs = check(root)
    if errs:
        print("FAIL: %s" % root)
        for e in errs:
            print("  " + e)
        return 1

    # Discriminating power: the comparison must reject data that is wrong in
    # each of the ways the real defects were wrong. A green run above means
    # nothing unless these three go red.
    injections = [
        ("dropped half the rows (the one-entry manifest-list bug)",
         lambda d: {k: v[: len(v) // 2] for k, v in d.items()}),
        ("swapped two columns' values (a field-id mis-binding)",
         lambda d: {"id": d["score"], "score": d["id"]}),
        ("perturbed one value",
         lambda d: {k: ([v[0] + 1] + list(v[1:])) for k, v in d.items()}),
        # G2ICE-82: the delete must be what removes the row. If the reader
        # silently ignored the delete file, the deleted id comes back and the
        # row count is 12 -- both caught here.
        ("resurrected the deleted row (delete file ignored)",
         lambda d: {"id": list(d["id"]) + [1000 + DELETED_ROW],
                    "score": list(d["score"]) + [DELETED_ROW * 0.5 - 3.0]}),
    ]
    for name, mut in injections:
        if not check(root, mutate=mut):
            print("FAIL: injection not caught -- %s" % name)
            return 1

    meta = latest_metadata(root)
    errs = check_history(root, meta)
    if errs:
        print("FAIL: history logs of %s" % meta)
        for e in errs:
            print("  " + e)
        return 1
    tmp, stripped = history_stripped(root, meta)
    try:
        if not check_history(root, stripped):
            print("FAIL: injection not caught -- history logs removed")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    derrs = check_duckdb_history(meta)
    if derrs:
        print("FAIL: duckdb time travel")
        for e in derrs:
            print("  " + e)
        return 1
    print("PASS: snapshot-log/metadata-log drive pyiceberg as-of reads%s"
          % ("" if derrs is None else " and duckdb time travel"))

    print("PASS: pyiceberg read %d rows value-for-value from %s" % (N_ROWS - 1, root))
    print("      (deleted row %d absent; 4 injections caught)" % (1000 + DELETED_ROW))
    return 0


if __name__ == "__main__":
    sys.exit(main())
