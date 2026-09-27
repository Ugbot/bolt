#!/usr/bin/env python3
"""Refuse raw 16-byte compares of StringViews.

A spilled (>12-byte) StringView holds a buffer offset, so memcmp of the view
compares WHERE the bytes live, not what they are (G2CHK-295). Compare content
(sv_bytes / sv_compare / DistinctSetUtf8) instead. A compare of cells that are
canonical by construction (inline bytes, zero padding, or fixed 16-byte
decimals) is allowed with a trailing `// sv-memcmp-ok: <why>` comment.

usage: lint_sv_memcmp.py [ROOT...]   (default: this repo's include/ and src/)
"""
import os
import re
import sys

PAT = re.compile(r"memcmp\s*\([^;]*(,\s*16\s*\)|sizeof\s*\(\s*(bolt::)?StringView\s*\))")
EXTS = (".h", ".hpp", ".inc", ".cpp", ".cc")


def scan(root):
    bad = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(EXTS):
                continue
            path = os.path.join(dirpath, f)
            with open(path, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if PAT.search(line) and "sv-memcmp-ok" not in line:
                        bad.append(f"{path}:{n}: {line.strip()}")
    return bad


def main(argv):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = argv[1:] or [os.path.join(here, "include"), os.path.join(here, "src")]
    bad = []
    for r in roots:
        bad += scan(r)
    for b in bad:
        print(b)
    if bad:
        print(f"lint_sv_memcmp: {len(bad)} raw 16-byte compare(s); compare "
              "StringView content, or mark a canonical cell with "
              "`// sv-memcmp-ok: <why>`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
