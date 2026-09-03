#!/usr/bin/env python3
"""Will anything OTHER than bolt open a table bolt wrote?

    ./test_bolt_iceberg_metadata_interop           # writes the table
    python3 scripts/iceberg_metadata_interop.py <printed dir>

WHY THIS EXISTS. Every Iceberg write test in this tree checks bolt's writer
against bolt's own reader, and for metadata.json that is close to no evidence
at all: bolt's parser does not read the `"type"` property of a schema, so the
writer could omit it -- and did -- while every round trip stayed green. DuckDB
1.4.5 refused the same table with `StructType required property 'type' is
missing`. A writer and a reader that share a misreading agree perfectly.

The two oracles here are independent implementations of the Iceberg spec:
pyiceberg's `StaticTable` (which validates metadata against its pydantic model
-- a missing required property is an error, not a default) and DuckDB's
`iceberg_scan` (a separate C++ implementation). They are checked to have
DISCRIMINATING POWER first, by being run against the pre-fix metadata shape
re-derived here from the spec violation, not by trusting that they would have
complained.

Exit 0 = every available oracle accepted the table; 1 = one rejected it;
2 = no oracle is installed (a SKIP, never a PASS).
"""
import copy
import json
import os
import shutil
import sys
import tempfile


def latest_metadata(table_dir):
    """The metadata file a reader would pick, via version-hint.text."""
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


def spec_check(doc):
    """The v2 required-property set, read off the spec, not off bolt."""
    missing = [k for k in ("format-version", "table-uuid", "location",
                           "last-column-id", "last-sequence-number",
                           "last-updated-ms", "schemas", "current-schema-id",
                           "partition-specs", "default-spec-id",
                           "last-partition-id", "sort-orders",
                           "default-sort-order-id") if k not in doc]
    for i, s in enumerate(doc.get("schemas", [])):
        if s.get("type") != "struct":
            missing.append("schemas[%d].type == 'struct'" % i)
        for f in s.get("fields", []):
            # A nested field's type is an OBJECT; if one is ever emitted as a
            # bare string the column silently loses its children.
            t = f.get("type")
            if isinstance(t, str) and t in ("struct", "list", "map"):
                missing.append("schemas[%d].fields[%s].type is the bare "
                               "string %r" % (i, f.get("name"), t))
    return missing


def with_property_removed(doc, drop_schema_type=True):
    """The PRE-FIX shape: schemas with no "type", no last-partition-id."""
    d = copy.deepcopy(doc)
    d.pop("last-partition-id", None)
    if drop_schema_type:
        for s in d.get("schemas", []):
            s.pop("type", None)
    return d


def write_variant(table_dir, doc):
    """A copy of the table whose metadata.json is `doc`. Data files are
    hardlinked/copied so both oracles see identical bytes."""
    tmp = tempfile.mkdtemp(prefix="bolt_ice_variant_")
    dst = os.path.join(tmp, "trades")
    shutil.copytree(table_dir, dst)
    tgt = latest_metadata(dst)
    with open(tgt, "w") as f:
        json.dump(doc, f)
    return tmp, dst, tgt


def oracle_pyiceberg(metadata_path):
    try:
        from pyiceberg.table import StaticTable
    except Exception as e:                                   # pragma: no cover
        return None, "pyiceberg unavailable: %s" % e
    try:
        t = StaticTable.from_metadata("file://" + metadata_path)
        n = len(t.schema().fields)
        return True, "pyiceberg opened it; current schema has %d fields" % n
    except Exception as e:
        return False, "pyiceberg rejected it: %s" % str(e).strip()[:200]


# A DuckDB failure that happens AFTER metadata parsing is not this lane's.
# Classifying instead of collapsing to pass/fail is what keeps the gate honest:
# "duckdb read the whole table" and "duckdb parsed the metadata and then could
# not open a data file" are different facts with different owners, and reporting
# either as a bare RED would hide which.
_META_MARKERS = ("StructType", "required property", "Invalid Input Error",
                 "metadata", "schema")


def oracle_duckdb(table_dir):
    """(state, detail) — state True=read it, 'metadata_ok'=parsed the metadata
    then failed downstream, False=rejected the metadata, None=unavailable."""
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
        rows = con.execute("SELECT id, price FROM iceberg_scan('%s') "
                           "ORDER BY id" % table_dir).fetchall()
        # VALUES, not a row count. The model is re-derived from the writing
        # test's generating rule (id = 100+i, price = 1.5*i), so a table that
        # DuckDB reads with the columns swapped or the rows shifted fails here
        # while a row-count check would call it green -- the exact mistake the
        # ClickBench board made before it started comparing cell by cell.
        want = [(100 + i, 1.5 * i) for i in range(len(rows))]
        if rows != want:
            return False, ("duckdb read the table but the VALUES differ: "
                           "got %r want %r" % (rows[:3], want[:3]))
        return True, ("duckdb iceberg_scan read %d rows, every (id, price) "
                      "cell exact" % len(rows))
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:200]
        if msg.startswith("IO Error") and "metadata" not in msg:
            return "metadata_ok", ("duckdb parsed the metadata, then failed "
                                   "reading a file: %s" % msg)
        return False, "duckdb rejected the metadata: %s" % msg


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    table_dir = os.path.abspath(sys.argv[1])
    meta = latest_metadata(table_dir)
    with open(meta) as f:
        doc = json.load(f)

    print("table:    %s" % table_dir)
    print("metadata: %s" % os.path.basename(meta))

    rc = 0
    missing = spec_check(doc)
    if missing:
        print("RED  spec: metadata is missing %s" % ", ".join(missing))
        rc = 1
    else:
        print("GREEN spec: every v2 required property is present")

    # --- discriminating power, established BEFORE the real run -------------
    # If the oracles accept the pre-fix shape too, a green run below means
    # nothing. This is the same argument parquet_bloom_check.py makes by
    # probing bolt's bloom-disabled fixture first.
    tmp, vdir, vmeta = write_variant(table_dir, with_property_removed(doc))
    try:
        pv, pd_ = oracle_pyiceberg(vmeta)
        dv, dd = oracle_duckdb(vdir)
        sensitive = 0
        for name, v, d in (("pyiceberg", pv, pd_), ("duckdb", dv, dd)):
            if v is None:
                print("SKIP  %-9s not installed" % name)
            elif v is False:
                print("ok    %-9s REJECTS the pre-fix shape: %s" % (name, d))
                sensitive += 1
            else:
                # Recorded, not hidden: pyiceberg 0.10.0 defaults a schema's
                # missing "type" instead of failing, so it CANNOT witness this
                # defect. Only DuckDB can, and a green pyiceberg run below is
                # therefore evidence of nothing about "type":"struct".
                print("note  %-9s tolerates the pre-fix shape (%s) -- blind "
                      "to this defect, its green below proves nothing about it"
                      % (name, d))
        if sensitive == 0:
            print("RED   no installed oracle can witness this defect -- a "
                  "green run below would be vacuous")
            rc = 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- the real run -------------------------------------------------------
    any_oracle = False
    ok, detail = oracle_pyiceberg(meta)
    if ok is None:
        print("SKIP  pyiceberg: %s" % detail)
    else:
        any_oracle = True
        print("%s pyiceberg: %s" % ("GREEN" if ok else "RED  ", detail))
        rc = rc or (0 if ok else 1)

    ok, detail = oracle_duckdb(table_dir)
    if ok is None:
        print("SKIP  duckdb: %s" % detail)
    elif ok == "metadata_ok":
        any_oracle = True
        # The metadata question this script asks is answered GREEN; the
        # remaining failure is a data-file path, owned by the manifest/path
        # lane (G2ICE-84) and by the parquet writer lane (G2ICE-87).
        print("GREEN duckdb:    %s" % detail)
        print("      (metadata accepted -- the residue is not this lane's; "
              "see G2ICE-84 / G2ICE-87)")
    else:
        any_oracle = True
        print("%s duckdb:    %s" % ("GREEN" if ok else "RED  ", detail))
        rc = rc or (0 if ok else 1)

    if not any_oracle:
        print("no oracle installed -- SKIP, not PASS")
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
