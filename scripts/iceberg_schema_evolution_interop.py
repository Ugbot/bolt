#!/usr/bin/env python3
"""G2ICE-89: does a time-travel read of a PRE-evolution snapshot see the
schema it was actually written under, or does it incorrectly reflect a later
`table_add_column`?

    ./test_bolt_iceberg_metadata_interop \
        --gtest_filter=IcebergMetadataInterop.SnapshotsRecordTheSchemaTheyWereCommittedUnder
    python3 scripts/iceberg_schema_evolution_interop.py <printed dir> <snap0-id>

Two independent implementations of the Iceberg spec are the oracles here
(pyiceberg's StaticTable + DuckDB's iceberg_scan), exactly as in
iceberg_metadata_interop.py (G2ICE-50) — and, as there, DISCRIMINATING POWER
is established first: a synthetic "pre-fix" variant that collapses every
schema evolution bolt ever wrote into the single ever-mutating schema-id the
old (buggy) `table_add_column` produced. If an oracle is blind to THAT shape,
its green result on the real table proves nothing about this defect.

Exit 0 = both installed oracles show the old snapshot correctly reading its
original (pre-evolution) columns; 1 = at least one shows the bug; 2 = no
oracle installed (a SKIP, never a PASS).
"""
import copy
import json
import os
import shutil
import sys
import tempfile


def latest_metadata(table_dir):
    hint = os.path.join(table_dir, "metadata", "version-hint.text")
    if os.path.exists(hint):
        with open(hint) as f:
            v = f.read().strip()
        p = os.path.join(table_dir, "metadata", "v%s.metadata.json" % v)
        if os.path.exists(p):
            return p
    cand = sorted(
        os.path.join(table_dir, "metadata", n)
        for n in os.listdir(os.path.join(table_dir, "metadata"))
        if n.endswith(".metadata.json"))
    if not cand:
        raise SystemExit("no metadata.json under %s" % table_dir)
    return cand[-1]


def collapse_to_single_mutating_schema(doc):
    """Re-derive the PRE-FIX shape from the REAL (fixed) metadata: one
    schema object (the widest -- i.e. the final, current one), every
    snapshot's "schema-id" rewritten to point at it. This is exactly what
    the old `table_add_column` produced: n_schemas stuck at 1,
    current-schema-id constant, every snapshot sharing that one id whose
    fields kept growing underneath it."""
    d = copy.deepcopy(doc)
    widest = max(d["schemas"], key=lambda s: len(s["fields"]))
    d["schemas"] = [widest]
    d["current-schema-id"] = widest["schema-id"]
    for snap in d["snapshots"]:
        snap["schema-id"] = widest["schema-id"]
    return d


def write_variant(table_dir, doc):
    tmp = tempfile.mkdtemp(prefix="bolt_ice_evo_variant_")
    dst = os.path.join(tmp, "trades")
    shutil.copytree(table_dir, dst)
    tgt = latest_metadata(dst)
    with open(tgt, "w") as f:
        json.dump(doc, f)
    return tmp, dst, tgt


def oracle_pyiceberg_snapshot0_fields(metadata_path, snap0_id):
    """(state, detail): state True/False = field NAMES resolved for
    snapshot 0 via pyiceberg's own schemas()[snapshot.schema_id] lookup and
    whether "qty" is (wrongly) among them; None = unavailable."""
    try:
        from pyiceberg.table import StaticTable
    except Exception as e:                                   # pragma: no cover
        return None, "pyiceberg unavailable: %s" % e
    try:
        t = StaticTable.from_metadata("file://" + metadata_path)
        snap0 = next(s for s in t.metadata.snapshots
                     if s.snapshot_id == snap0_id)
        resolved = t.schemas()[snap0.schema_id]
        names = [f.name for f in resolved.fields]
        has_qty = "qty" in names
        return (not has_qty), ("schemas()[snapshot0.schema_id] = %r "
                               "(has 'qty': %s)" % (names, has_qty))
    except Exception as e:
        return False, "pyiceberg rejected it: %s" % str(e).strip()[:200]


def oracle_pyiceberg_scan_values(metadata_path, snap0_id):
    """The stronger check on the REAL table: an actual time-travel SCAN of
    snapshot 0 must return exactly the (id, price) rows written before the
    evolution -- not merely a schema OBJECT that happens to be short."""
    try:
        from pyiceberg.table import StaticTable
    except Exception as e:                                   # pragma: no cover
        return None, "pyiceberg unavailable: %s" % e
    try:
        t = StaticTable.from_metadata("file://" + metadata_path)
        df = t.scan(snapshot_id=snap0_id).to_arrow().to_pandas()
        if list(df.columns) != ["id", "price"]:
            return False, "columns = %r, want ['id', 'price']" % list(df.columns)
        # Generating rule in the writing test: id=100+i, price=1.5*i.
        want = [(100 + i, 1.5 * i) for i in range(len(df))]
        got = list(df.itertuples(index=False, name=None))
        if got != want:
            return False, "rows = %r, want %r" % (got, want)
        return True, "scan(snapshot_id=%d) read %d rows, exact" % (
            snap0_id, len(df))
    except Exception as e:
        return False, "pyiceberg scan failed: %s" % str(e).strip()[:200]


def oracle_duckdb_snapshot0_values(table_dir, snap0_id):
    try:
        import duckdb
    except Exception as e:                                   # pragma: no cover
        return None, "duckdb unavailable: %s" % e
    con = duckdb.connect()
    try:
        con.execute("INSTALL iceberg; LOAD iceberg;")
    except Exception as e:                                   # pragma: no cover
        return None, "duckdb iceberg extension unavailable: %s" % e
    try:
        cols = con.execute(
            "DESCRIBE SELECT * FROM iceberg_scan('%s', snapshot_from_id=%d)"
            % (table_dir, snap0_id)).fetchdf()["column_name"].tolist()
        if cols != ["id", "price"]:
            return False, "columns = %r, want ['id', 'price']" % cols
        rows = con.execute(
            "SELECT id, price FROM iceberg_scan('%s', snapshot_from_id=%d) "
            "ORDER BY id" % (table_dir, snap0_id)).fetchall()
        want = [(100 + i, 1.5 * i) for i in range(len(rows))]
        if rows != want:
            return False, "rows = %r, want %r" % (rows, want)
        return True, ("iceberg_scan(snapshot_from_id=%d) read %d rows, "
                      "exact, columns %r" % (snap0_id, len(rows), cols))
    except Exception as e:
        return False, "duckdb rejected it: %s" % str(e).strip().splitlines()[0][:200]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    table_dir = os.path.abspath(sys.argv[1])
    snap0_id = int(sys.argv[2])
    meta = latest_metadata(table_dir)
    with open(meta) as f:
        doc = json.load(f)

    print("table:    %s" % table_dir)
    print("metadata: %s" % os.path.basename(meta))
    print("snapshot0-id: %d" % snap0_id)

    rc = 0

    # --- discriminating power, established BEFORE the real run -------------
    tmp, vdir, vmeta = write_variant(table_dir, collapse_to_single_mutating_schema(doc))
    try:
        pv, pd_ = oracle_pyiceberg_snapshot0_fields(vmeta, snap0_id)
        dv, dd = oracle_duckdb_snapshot0_values(vdir, snap0_id)
        sensitive = 0
        for name, v, d in (("pyiceberg", pv, pd_), ("duckdb", dv, dd)):
            if v is None:
                print("SKIP  %-9s not installed" % name)
            elif v is False:
                print("ok    %-9s REJECTS the pre-fix (collapsed) shape: %s"
                     % (name, d))
                sensitive += 1
            else:
                print("note  %-9s tolerates the pre-fix shape (%s) -- blind "
                     "to this defect, its green below proves nothing about it"
                     % (name, d))
        if sensitive == 0:
            print("RED   no installed oracle can witness this defect -- a "
                 "green run below would be vacuous")
            rc = 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- the real run --------------------------------------------------------
    any_oracle = False

    ok, detail = oracle_pyiceberg_snapshot0_fields(meta, snap0_id)
    if ok is None:
        print("SKIP  pyiceberg (schema resolution): %s" % detail)
    else:
        any_oracle = True
        print("%s pyiceberg (schema resolution): %s"
             % ("GREEN" if ok else "RED  ", detail))
        rc = rc or (0 if ok else 1)

    ok, detail = oracle_pyiceberg_scan_values(meta, snap0_id)
    if ok is None:
        print("SKIP  pyiceberg (scan values):    %s" % detail)
    else:
        any_oracle = True
        print("%s pyiceberg (scan values):    %s"
             % ("GREEN" if ok else "RED  ", detail))
        rc = rc or (0 if ok else 1)

    ok, detail = oracle_duckdb_snapshot0_values(table_dir, snap0_id)
    if ok is None:
        print("SKIP  duckdb:                     %s" % detail)
    else:
        any_oracle = True
        print("%s duckdb:                     %s"
             % ("GREEN" if ok else "RED  ", detail))
        rc = rc or (0 if ok else 1)

    if not any_oracle:
        print("no oracle installed -- SKIP, not PASS")
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
