#!/usr/bin/env python3
"""One zstd interface: bolt/ingest/bolt_zstd.h (+ bolt_zstd_stream.h for the streaming decoder).

Fails when a C/C++ file outside the wrapper includes <zstd.h>, <zstd_errors.h> or bolt's own
encoder / decoder headers (bolt_zstd_enc.h, bolt_zstd_dec.h, bolt_zstd_backends.h), or calls a
ZSTD_* function.

  zstd_iface_lint.py <root> [<root> ...]    scan the trees (build*/ third_party/ .git skipped)
  zstd_iface_lint.py --self-test
"""
import os
import re
import sys

# Path suffixes that may see the implementations: the wrapper itself, and the tests / benchmarks
# whose subject is one implementation.
ALLOW = (
    "src/ingest/bolt_zstd.cpp",
    "src/ingest/bolt_zstd_backends.h",
    "src/ingest/bolt_zstd_dec.cpp",
    "src/ingest/bolt_zstd_dec.h",
    "src/ingest/bolt_zstd_dec_internal.h",
    "src/ingest/bolt_zstd_enc.cpp",
    "src/ingest/bolt_zstd_enc.h",
    "src/ingest/bolt_zstd_stream.cpp",
    "tests/test_bolt_zstd_dec.cpp",
    "tests/test_bolt_zstd_enc_dump.cpp",
    "tests/test_bolt_zstd_backends.cpp",
    "benchmarks/bench_zstd_decoders.cpp",
)
EXTS = (".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".inc", ".ipp")
SKIP_DIRS = {".git", "third_party", "node_modules", "_deps", ".claude", ".scratch", "extern"}
BAD = re.compile(
    r'^\s*#\s*include\s*[<"](?:[^">]*/)?(?:zstd\.h|zstd_errors\.h|bolt_zstd_enc\.h|bolt_zstd_dec\.h'
    r'|bolt_zstd_backends\.h)[">]|\bZSTD_[A-Za-z0-9_]+\s*\(')


def scan_text(path, text):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("//", 1)[0] if not line.lstrip().startswith("#") else line
        if BAD.search(code):
            out.append("%s:%d: %s" % (path, i, line.strip()))
    return out


def scan_root(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS
                   and not x.startswith("build") and not x.startswith("cmake-build")]
        for f in files:
            if not f.endswith(EXTS):
                continue
            p = os.path.join(d, f)
            rel = p.replace(os.sep, "/")
            if rel.endswith(ALLOW):
                continue
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    out += scan_text(rel, fh.read())
            except OSError as e:
                out.append("%s: unreadable: %s" % (rel, e))
    return out


def self_test():
    bad = ['#include <zstd.h>', '  #include "bolt/ingest/bolt_zstd_dec.h"', '#include "bolt_zstd_enc.h"',
           'size_t r = ZSTD_decompress(a, b, c, d);', '# include <zstd_errors.h>']
    good = ['#include "bolt/ingest/bolt_zstd.h"', '#include "bolt/ingest/bolt_zstd_stream.h"',
            '// #include <zstd.h>', 'int kCodecZstd = 6;  // ZSTD_decompress( in a comment',
            'zstd_decompress(a, b, c, d);', '#include "bolt_zstd_dec_limits.h"']
    for t in bad:
        if not scan_text("x.cpp", t):
            print("self-test: missed: " + t)
            return 1
    for t in good:
        if scan_text("x.cpp", t):
            print("self-test: false hit: " + t)
            return 1
    print("zstd_iface_lint self-test ok (%d bad, %d good)" % (len(bad), len(good)))
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) < 2:
        print(__doc__)
        return 2
    hits = []
    for root in argv[1:]:
        if not os.path.isdir(root):
            print("not a directory: " + root)
            return 2
        hits += scan_root(root)
    for h in hits:
        print(h)
    if hits:
        print("%d zstd call(s) outside the interface (bolt/ingest/bolt_zstd.h)" % len(hits))
        return 1
    print("zstd_iface_lint ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
