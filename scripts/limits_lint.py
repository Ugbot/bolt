#!/usr/bin/env python3
"""limits_lint.py — two ratchets from the limits review (G2CHK-330).

stack-arrays   No function-scope array larger than --stack-bytes (16 KiB) on the
               stack. static / thread_local / constexpr arrays are not stack.
               The G2ICE-112 and G2CHK-12 crashes were exactly this: a cap was
               raised and a local array sized by it overflowed a thread stack.
literal-caps   No NEW compile-time capacity constant (kMax*, k*Cap*, k_max_*,
               k_*_cap*, ...) outside a *limits.h header. Existing ones are in
               the baseline; the list may only shrink.

Both checks compare against a baseline file. A finding not in the baseline
fails (new debt). A baseline entry that no longer exists also fails, so the
baseline is shrunk in the same commit that pays the debt down (ratchet).

    limits_lint.py --root . --src include --src src --baseline-dir scripts/limits_lint_baseline
    limits_lint.py ... --update       # rewrite the baselines (review the diff)
    limits_lint.py ... --self-test    # prove each check can fail
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

EXTS = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx", ".inc", ".ipp"}

PRIM_SIZES = {
    "char": 1, "signed char": 1, "unsigned char": 1, "bool": 1, "int8_t": 1,
    "uint8_t": 1, "std::int8_t": 1, "std::uint8_t": 1, "std::byte": 1, "byte": 1,
    "char8_t": 1, "short": 2, "int16_t": 2, "uint16_t": 2, "std::int16_t": 2,
    "std::uint16_t": 2, "char16_t": 2, "int": 4, "unsigned": 4, "unsigned int": 4,
    "int32_t": 4, "uint32_t": 4, "std::int32_t": 4, "std::uint32_t": 4, "float": 4,
    "char32_t": 4, "wchar_t": 4, "long": 8, "unsigned long": 8, "long long": 8,
    "unsigned long long": 8, "int64_t": 8, "uint64_t": 8, "std::int64_t": 8,
    "std::uint64_t": 8, "double": 8, "size_t": 8, "std::size_t": 8,
    "ssize_t": 8, "ptrdiff_t": 8, "std::ptrdiff_t": 8, "uintptr_t": 8,
    "intptr_t": 8, "long double": 16,
}
# Unknown element types (structs) are estimated at this many bytes: enough to
# catch a struct array, low enough not to flag small ones.
UNKNOWN_ELEM_BYTES = 16

CAP_NAME = re.compile(
    r"^(k_?[A-Z]?\w*?(?:Max|MAX|Cap|CAP|Limit|LIMIT)\w*|k_max_\w+|k_\w+_(?:cap|max|limit)\w*"
    r"|k\w+(?:Cap|Max|Limit))$")
LITERAL_CAP = re.compile(
    r"\b(?:static\s+|inline\s+)*constexpr\s+(?:const\s+)?(?:std::)?"
    r"(?:u?int(?:8|16|32|64)_t|size_t|unsigned(?:\s+int|\s+long(?:\s+long)?)?|int|long)\s+"
    r"(\w+)\s*=\s*([^;{}]+);")


def strip_code(text: str) -> str:
    """Blank comments, string and char literals; keep newlines and length."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == "R" and i + 1 < n and text[i + 1] == '"':
            m = re.match(r'R"([^(\s]*)\(', text[i:])
            if m:
                end = text.find(")" + m.group(1) + '"', i)
                end = n if end < 0 else end + len(m.group(1)) + 2
                out.append(re.sub(r"[^\n]", " ", text[i:end]))
                i = end
            else:
                out.append(c)
                i += 1
        elif c in "\"'":
            if c == "'" and i > 0 and (text[i - 1].isalnum()):
                out.append(c)   # digit separator 1'000
                i += 1
                continue
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + " " * max(0, j - i - 2) + (c if j - i >= 2 else ""))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def eval_int(expr: str, consts: dict) -> int | None:
    e = expr.strip()
    e = re.sub(r"(?<=\d)'(?=\d)", "", e)
    e = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)(?:[uUlL]{1,3})\b", r"\1", e)
    e = re.sub(r"static_cast<[^>]*>", "", e)
    e = re.sub(r"\b(?:std::)?(?:u?int\d+_t|size_t)\s*\(", "(", e)

    def sub(m):
        name = m.group(0)
        if name in consts:
            return str(consts[name])
        tail = name.split("::")[-1]
        if tail in consts:
            return str(consts[tail])
        raise KeyError(name)

    try:
        e = re.sub(r"[A-Za-z_][\w:]*", sub, e)
    except KeyError:
        return None
    if not re.fullmatch(r"[\d\s+\-*/%()<>|&^xXa-fA-F]*", e):
        return None
    try:
        v = eval(e.replace("/", "//"), {"__builtins__": {}}, {})
    except Exception:
        return None
    return v if isinstance(v, int) and v >= 0 else None


def iter_sources(root: Path, srcs, excludes):
    for s in srcs:
        base = root / s
        if base.is_file():
            yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel = Path(dirpath).relative_to(root).as_posix()
            dirnames[:] = sorted(d for d in dirnames
                                 if not any((rel + "/" + d).startswith(x) for x in excludes)
                                 and d not in ("build", ".git", "_deps", "third_party"))
            for f in sorted(filenames):
                if Path(f).suffix in EXTS:
                    yield Path(dirpath) / f


def raw_consts(code: str) -> dict:
    raw: dict = {}
    for m in re.finditer(r"\b(?:constexpr|const)\s+[\w:<>\s]*?\b(\w+)\s*(?:=\s*([^;{}]+)|\{([^;{}]+)\})\s*;", code):
        raw.setdefault(m.group(1), []).append(m.group(2) or m.group(3))
    for m in re.finditer(r"^\s*#\s*define\s+(\w+)\s+([^\n]+)$", code, re.M):
        raw.setdefault(m.group(1), []).append(m.group(2))
    for m in re.finditer(r"\benum\b[^{;]*\{([^}]*)\}", code):
        for em in re.finditer(r"(\w+)\s*=\s*([^,]+)", m.group(1)):
            raw.setdefault(em.group(1), []).append(em.group(2))
    return raw


def resolve(raw: dict, base: dict) -> dict:
    """Evaluate raw[name] -> [expr] to ints. A name defined with different
    values is ambiguous: keep the smallest (fewer false alarms)."""
    out: dict = {}
    for _ in range(6):
        env = dict(base)
        env.update(out)
        changed = False
        for name, exprs in raw.items():
            vals = [v for v in (eval_int(x, env) for x in exprs) if v is not None]
            if vals and out.get(name) != min(vals):
                out[name] = min(vals)
                changed = True
        if not changed:
            break
    return out


def collect_consts(files) -> dict:
    """Global table; each file's own definitions override it (see scan)."""
    raw: dict = {}
    for _, code in files:
        for k, v in raw_consts(code).items():
            raw.setdefault(k, []).extend(v)
    return resolve(raw, {})


SCOPE_TYPE = re.compile(r"\b(struct|class|union|enum|namespace)\b|\bextern\s*\"\s*\"\s*$")
BLOCK_TAIL = re.compile(
    r"(?:\)|\]|\}|\belse|\bdo|\btry|:|\bconst|\bnoexcept|\boverride|\bfinal|\bmutable|"
    r"->\s*[\w:<>,\s*&]+)\s*$")
DECL = re.compile(
    r"^(?P<quals>(?:(?:const|volatile|unsigned|signed|alignas\([^)]*\)|struct|typename)\s+)*)"
    r"(?P<type>[A-Za-z_][\w:]*(?:\s*<[^;{}]*?>)?(?:\s+(?:long|int|char|short))*)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?P<dims>(?:\[[^\]\[]+\]\s*)+)\s*(?:=.*|\{.*)?$", re.S)
NOT_TYPES = {"return", "delete", "new", "throw", "case", "goto", "sizeof", "using",
             "typedef", "else", "do", "co_return", "co_yield", "co_await"}


def elem_size(t: str) -> int:
    t = re.sub(r"\s+", " ", t.strip())
    if t in PRIM_SIZES:
        return PRIM_SIZES[t]
    if t.endswith("*"):
        return 8
    return UNKNOWN_ELEM_BYTES


def classify_brace(head: str, in_func: bool) -> str:
    if head and SCOPE_TYPE.search(head) and not head.endswith(")"):
        return "type"
    if head == "":
        return "func" if in_func else "init"
    if head.endswith("="):
        return "init"
    if BLOCK_TAIL.search(head):
        return "func"
    return "init"


def scan_stack_arrays(path: str, code: str, consts: dict, limit: int):
    """Yield (key, bytes) for function-scope arrays larger than limit."""
    stack = []          # scope kinds: 'type', 'func', 'init'
    stmt_start = 0
    paren = 0
    func_depth = 0
    for i, c in enumerate(code):
        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(0, paren - 1)
        elif c == "{" and paren == 0:
            in_init = bool(stack) and stack[-1] == "init"
            kind = "init" if in_init else classify_brace(
                code[stmt_start:i].strip(), func_depth > 0)
            stack.append(kind)
            if kind == "func":
                func_depth += 1
            if kind != "init":
                stmt_start = i + 1
        elif c == "}" and paren == 0:
            kind = stack.pop() if stack else "type"
            if kind == "func":
                func_depth -= 1
            if kind != "init":
                stmt_start = i + 1
        elif c == ";" and paren == 0 and (not stack or stack[-1] != "init"):
            if func_depth > 0 and stack and stack[-1] == "func":
                hit = check_decl(code[stmt_start:i].strip(), consts, limit)
                if hit:
                    yield f"{path}::{hit[0]}::{hit[1]}", hit[2]
            stmt_start = i + 1


def check_decl(stmt: str, consts: dict, limit: int):
    s = re.sub(r"\s+", " ", stmt)
    s = re.sub(r"^(?:\[\[[^\]]*\]\]\s*)+", "", s)
    if re.search(r"\b(static|thread_local|constexpr|extern|typedef|using)\b", s):
        return None
    m = DECL.match(s)
    if not m or m.group("type").split()[0] in NOT_TYPES:
        return None
    dims = re.findall(r"\[([^\]\[]+)\]", m.group("dims"))
    count = 1
    for d in dims:
        v = eval_int(d, consts)
        if v is None:
            return None
        count *= v
    total = count * elem_size(m.group("type"))
    if total <= limit:
        return None
    dim_txt = "][".join(d.strip() for d in dims)
    return m.group("name"), f"{m.group('type').strip()}[{dim_txt}]", total


def scan_literal_caps(path: str, code: str):
    if re.search(r"limits(?:_\w+)?\.h$", path):
        return
    for m in LITERAL_CAP.finditer(code):
        name, value = m.group(1), m.group(2).strip()
        if not CAP_NAME.match(name):
            continue
        if not re.fullmatch(r"[\d\s'xXa-fA-FuUlL+\-*/()<>]+", value):
            continue   # derived from another constant: not a literal
        if not re.search(r"\d", value):
            continue
        yield f"{path}::{name}"


def run(root: Path, srcs, excludes, stack_limit: int):
    files = []
    for p in iter_sources(root, srcs, excludes):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append((p.relative_to(root).as_posix(), strip_code(text)))
    consts = collect_consts(files)
    stack_hits = {}
    caps = set()
    for rel, code in files:
        local = dict(consts)
        local.update(resolve(raw_consts(code), consts))
        for key, size in scan_stack_arrays(rel, code, local, stack_limit):
            stack_hits[key] = size
        caps.update(scan_literal_caps(rel, code))
    return stack_hits, caps


def load_baseline(p: Path) -> set:
    if not p.exists():
        return set()
    return {ln.split("\t")[0].strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}


def compare(name: str, found: set, base: set) -> list:
    errs = []
    for k in sorted(found - base):
        errs.append(f"[{name}] NEW: {k}")
    for k in sorted(base - found):
        errs.append(f"[{name}] GONE (shrink the baseline): {k}")
    return errs


HEADER = {
    "stack_arrays": "# function-scope arrays > the stack budget (key<TAB>bytes). Only shrinks.\n",
    "literal_caps": "# compile-time capacity literals outside *limits.h. Only shrinks.\n",
}


def self_test(stack_limit: int) -> int:
    code = strip_code("""
        constexpr int kSmallCap = 16;
        constexpr int kWideCap = 4096;
        namespace x { struct S { char member[65536]; }; }
        int f(int a) {
            char ok[kSmallCap];
            static char st[1 << 20];
            thread_local char tl[1 << 20];
            uint64_t bad[kWideCap];                // 32 KiB
            for (int i = 0; i < 3; ++i) { double worse[kWideCap * 2]; }
            const char* s = "int fake[99999];";
            return a;
        }
        struct Q { uint64_t field[kWideCap]; void g() { char local[kWideCap * 8]; } };
    """)
    consts = collect_consts([("t.cpp", code)])
    hits = dict(scan_stack_arrays("t.cpp", code, consts, stack_limit))
    want = {"t.cpp::bad::uint64_t[kWideCap]", "t.cpp::worse::double[kWideCap * 2]",
            "t.cpp::local::char[kWideCap * 8]"}
    caps = set(scan_literal_caps("t.cpp", code))
    ok = set(hits) == want and caps == {"t.cpp::kSmallCap", "t.cpp::kWideCap"}
    if not ok:
        print("self-test FAILED\n  stack:", sorted(hits), "\n  caps:", sorted(caps))
        return 1
    print("self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--baseline-dir", required=False)
    ap.add_argument("--stack-bytes", type=int, default=16384)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test(a.stack_bytes)
    root = Path(a.root).resolve()
    stack_hits, caps = run(root, a.src or ["include", "src"], a.exclude, a.stack_bytes)
    bdir = root / (a.baseline_dir or "scripts/limits_lint_baseline")
    sp, cp = bdir / "stack_arrays.txt", bdir / "literal_caps.txt"
    if a.update:
        bdir.mkdir(parents=True, exist_ok=True)
        sp.write_text(HEADER["stack_arrays"] +
                      "".join(f"{k}\t{stack_hits[k]}\n" for k in sorted(stack_hits)))
        cp.write_text(HEADER["literal_caps"] + "".join(f"{k}\n" for k in sorted(caps)))
        print(f"baselines written: {len(stack_hits)} stack arrays, {len(caps)} literal caps")
        return 0
    errs = compare("stack-arrays", set(stack_hits), load_baseline(sp))
    errs += compare("literal-caps", caps, load_baseline(cp))
    if errs:
        print("\n".join(errs))
        print(f"\n{len(errs)} limits-lint finding(s). New stack arrays: move to an arena, "
              "thread_local scratch or ArenaVec. New caps: put them in <repo>_limits.h "
              "with their kind. Paid-down debt: rerun with --update.")
        return 1
    print(f"limits-lint ok: {len(stack_hits)} baselined stack arrays, "
          f"{len(caps)} baselined literal caps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
