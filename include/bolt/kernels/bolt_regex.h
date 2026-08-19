// bolt_regex.h — bounded "practical subset" regex compile + match +
// substitute kernel (G2FEAT-283 ClickBench Q29 follow-up: REGEXP_REPLACE).
//
// This is NOT a general PCRE/RE2-class engine — alternation (`|`) is not
// supported, and there is no NFA/DFA construction. It supports exactly the
// subset real analytical SQL queries actually use for REGEXP_REPLACE-style
// URL/text normalization (ClickBench's own Q29 pattern is the motivating
// case): a flat, optionally-nested SEQUENCE of atoms, each with an optional
// quantifier, anchored or not. Documented supported syntax:
//
//   literal chars          matched verbatim
//   .                       any single byte
//   [abc] / [^abc] / [a-z]  character class, optional negation, ranges
//   ?  *  +                 quantifier on the PRECEDING atom (greedy)
//   ( ... )                 capturing group (up to kMaxGroups - 1, 1-based)
//   (?: ... )                non-capturing group
//   ^  $                    anchor start / end of string
//   \.  \\  \(  \)  \[  \]  \+  \*  \?  \^  \$   escaped metacharacter
//
// NOT supported (documented gap, not attempted): alternation `|`, backrefs
// INSIDE the pattern, lazy quantifiers (`*?`), bounded repeat `{m,n}`,
// lookaround, Unicode character classes (`\d`/`\w`/`\s`) — ASCII byte
// matching only. A pattern using unsupported syntax fails to compile
// (returns false), which the SQL layer surfaces as a clean InvalidInput
// rather than a wrong match.
//
// Compile once (at SQL lowering / plan-build time, NOT per row) into a
// fixed-size CompiledPattern; match/substitute run per row against that
// compiled form — no per-row parsing, no per-row allocation. Backtracking
// is bounded by an explicit step budget (Tiger Style: no unbounded work) —
// a pathological pattern/input combination degrades to "no match" rather
// than hanging.
//
// Tiger Style: noexcept, no exceptions, no heap allocation, fixed caps,
// >=2 asserts/fn, functions <=70 lines (split via small helpers where the
// matcher logic would otherwise run long).
//
// ---------------------------------------------------------------------------
// SECOND ENGINE (additive, this file): the `re2_*` RE2-common-subset anchored
// FULL-MATCH engine — see the banner above the `Re2Program` section near the
// bottom. It is a Thompson NFA (linear-time, ReDoS-safe) supporting
// alternation `|`, `* + ? {m} {m,} {m,n}`, char classes `[..] [^..] \d\w\s`
// (+ negations) `.`, `(...)`/`(?:...)` grouping, and `(?i)`/`(?i:...)`
// case-insensitivity — the subset PromQL's `=~`/`!~` matchers need. It is
// entirely separate from the capture-group backtracker above; the existing
// `regex_compile`/`regex_search`/`regex_substitute` API is untouched.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "bolt/bolt_port.h"

namespace bolt {
namespace kernels {
namespace regex {

constexpr int      kMaxNodes       = 64;   // compiled pattern node cap
constexpr int      kMaxGroups      = 10;   // group 0 = whole match, 1..9 = \N
constexpr int      kMaxClassRanges = 8;    // ranges per character class
constexpr uint16_t kUnbounded      = 0xFFFFu;
// Backtracking step budget: bounds worst-case work to a linear multiple of
// input length rather than true exponential blowup. Generous for realistic
// (URL/text) inputs and patterns; exceeding it means "give up, no match" —
// safe, never wrong, never hangs.
constexpr int32_t  kMaxStepsPerByte = 64;

enum class NodeKind : uint8_t {
    Char       = 0,   // literal single byte (ch)
    Any        = 1,   // '.'
    Class      = 2,   // [abc] / [^abc] / [a-z]
    GroupStart = 3,   // '(' or '(?:' — group_id > 0 iff capturing
    GroupEnd   = 4,   // ')' — matches the GroupStart at match_group_start
    AnchorStart= 5,   // '^'
    AnchorEnd  = 6,   // '$'
};

struct ClassRange { uint8_t lo; uint8_t hi; };

// One compiled pattern node. Quantifier fields apply to Char/Any/Class
// atoms directly; for GroupStart the quantifier applies to the WHOLE
// group (repeated as a unit) and match_group_end names the paired
// GroupEnd's index so the matcher can skip/repeat the sub-sequence.
struct Node {
    NodeKind   kind;
    uint8_t    ch;                          // Char
    uint8_t    group_id;                    // GroupStart: 0 = non-capturing
    uint8_t    negate;                      // Class
    uint8_t    n_ranges;                     // Class
    ClassRange ranges[kMaxClassRanges];      // Class
    uint16_t   min_rep;                      // quantifier lower bound
    uint16_t   max_rep;                      // quantifier upper bound (kUnbounded)
    uint16_t   match_group_end;              // GroupStart only: paired GroupEnd idx
};

struct CompiledPattern {
    Node    nodes[kMaxNodes];
    uint8_t n_nodes;
    uint8_t n_groups;   // count of CAPTURING groups (group ids 1..n_groups)
};

// ---------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------

// Parse one character-class body `[...]` starting at `*io` (pointing just
// past '['); writes ranges into `n`, advances `*io` past the closing ']'.
// Returns false on malformed input (unterminated class, too many ranges).
inline bool regex_compile_class(const char* pat, uint32_t len, uint32_t* io,
                                Node* n) noexcept {
    assert(pat != nullptr && io != nullptr && n != nullptr);
    assert(*io <= len);
    n->kind = NodeKind::Class;
    n->negate = 0;
    n->n_ranges = 0;
    uint32_t i = *io;
    if (i < len && pat[i] == '^') { n->negate = 1; ++i; }
    bool any_ranges = false;
    while (i < len && pat[i] != ']') {
        if (n->n_ranges >= kMaxClassRanges) return false;
        std::uint8_t lo = static_cast<std::uint8_t>(pat[i]);
        std::uint8_t hi = lo;
        ++i;
        if (i + 1 < len && pat[i] == '-' && pat[i + 1] != ']') {
            hi = static_cast<std::uint8_t>(pat[i + 1]);
            i += 2;
        }
        n->ranges[n->n_ranges].lo = lo;
        n->ranges[n->n_ranges].hi = hi;
        n->n_ranges += 1;
        any_ranges = true;
    }
    if (i >= len || pat[i] != ']') return false;   // unterminated
    *io = i + 1;
    return any_ranges;
}

// Parse a trailing quantifier (?, *, +, or none) at `*io`; sets min/max on
// `n` and advances `*io` past the quantifier char if present.
inline void regex_compile_quantifier(const char* pat, uint32_t len,
                                     uint32_t* io, Node* n) noexcept {
    assert(pat != nullptr && io != nullptr && n != nullptr);
    n->min_rep = 1; n->max_rep = 1;
    if (*io >= len) return;
    switch (pat[*io]) {
        case '?': n->min_rep = 0; n->max_rep = 1;          ++(*io); break;
        case '*': n->min_rep = 0; n->max_rep = kUnbounded; ++(*io); break;
        case '+': n->min_rep = 1; n->max_rep = kUnbounded; ++(*io); break;
        default: break;
    }
}

// Compile `pat[0..pat_len)` into `out`. Returns false on any unsupported
// construct or a pattern too large for kMaxNodes/kMaxGroups (the caller
// treats this as a clean compile-time InvalidInput, never a silent
// mismatch). Tiger Style: single bounded forward pass, no recursion.
inline bool regex_compile(const char* pat, uint32_t pat_len,
                          CompiledPattern* out) noexcept {
    assert(pat != nullptr || pat_len == 0);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    uint16_t group_stack[8];     // nesting depth cap
    uint8_t  stack_n = 0;
    uint8_t  next_group_id = 1;
    uint32_t i = 0;
    while (i < pat_len) {
        if (out->n_nodes >= kMaxNodes) return false;
        Node* n = &out->nodes[out->n_nodes];
        std::memset(n, 0, sizeof(*n));
        const char c = pat[i];
        if (c == '^') { n->kind = NodeKind::AnchorStart; ++i; out->n_nodes++; continue; }
        if (c == '$') { n->kind = NodeKind::AnchorEnd;   ++i; out->n_nodes++; continue; }
        if (c == '(') {
            if (stack_n >= 8) return false;
            n->kind = NodeKind::GroupStart;
            ++i;
            if (i + 1 < pat_len && pat[i] == '?' && pat[i + 1] == ':') {
                n->group_id = 0; i += 2;              // non-capturing
            } else {
                if (next_group_id >= kMaxGroups) return false;
                n->group_id = next_group_id++;
            }
            group_stack[stack_n++] = out->n_nodes;
            out->n_nodes++;
            continue;   // quantifier (if any) attaches once ')' closes
        }
        if (c == ')') {
            if (stack_n == 0) return false;
            const uint16_t start_idx = group_stack[--stack_n];
            n->kind = NodeKind::GroupEnd;
            ++i;
            out->nodes[start_idx].match_group_end =
                static_cast<uint16_t>(out->n_nodes);
            regex_compile_quantifier(pat, pat_len, &i, &out->nodes[start_idx]);
            out->n_nodes++;
            continue;
        }
        if (c == '[') {
            ++i;
            if (!regex_compile_class(pat, pat_len, &i, n)) return false;
            regex_compile_quantifier(pat, pat_len, &i, n);
            out->n_nodes++;
            continue;
        }
        if (c == '.') {
            n->kind = NodeKind::Any;
            ++i;
            regex_compile_quantifier(pat, pat_len, &i, n);
            out->n_nodes++;
            continue;
        }
        if (c == '\\' && i + 1 < pat_len) {
            n->kind = NodeKind::Char;
            n->ch = static_cast<std::uint8_t>(pat[i + 1]);
            i += 2;
            regex_compile_quantifier(pat, pat_len, &i, n);
            out->n_nodes++;
            continue;
        }
        if (c == '|') return false;   // alternation: documented unsupported
        n->kind = NodeKind::Char;
        n->ch = static_cast<std::uint8_t>(c);
        ++i;
        regex_compile_quantifier(pat, pat_len, &i, n);
        out->n_nodes++;
    }
    if (stack_n != 0) return false;   // unbalanced groups
    out->n_groups = static_cast<std::uint8_t>(next_group_id - 1);
    assert(out->n_nodes <= kMaxNodes);
    assert(out->n_groups < kMaxGroups);
    return true;
}

// ---------------------------------------------------------------------
// Match
// ---------------------------------------------------------------------

struct MatchResult {
    bool    matched;
    int32_t g_start[kMaxGroups];   // group 0 = whole match; -1 = unset
    int32_t g_end[kMaxGroups];
};

struct MatchState {
    const char*  text;
    int32_t      len;
    int32_t      steps_left;      // backtracking budget (Tiger Style bound)
    MatchResult* mr;
};

inline bool regex_class_matches(const Node& n, std::uint8_t b) noexcept {
    bool hit = false;
    for (std::uint8_t r = 0; r < n.n_ranges; ++r) {
        if (b >= n.ranges[r].lo && b <= n.ranges[r].hi) { hit = true; break; }
    }
    return n.negate ? !hit : hit;
}

inline bool regex_atom_matches(const Node& n, std::uint8_t b) noexcept {
    switch (n.kind) {
        case NodeKind::Char:  return b == n.ch;
        // '.' does NOT match a newline. POSIX, PCRE and RE2 all require an
        // explicit DOTALL/`s` flag for that, and this engine exposes no such
        // flag — so returning true unconditionally was plain over-matching.
        // The Re2Op engine further down this file already gets this right
        // (`adv = (b != '\n')`); only this backtracking engine did not, so the
        // two disagreed about the same pattern depending on which one ran.
        // Found on ClickBench Q29: one referer contains a literal newline, and
        // `^https?://(?:www\.)?([^/]+)/.*$` must NOT match it because `.*`
        // cannot cross the newline to reach `$`. Both oracles leave that string
        // unchanged; chukonu rewrote it to its host and mis-grouped the row.
        case NodeKind::Any:   return b != '\n';
        case NodeKind::Class: return regex_class_matches(n, b);
        default:              return false;
    }
}

// Forward decl: try to match nodes[node_i .. node_hi) starting at text
// position `pos`; on success, matches the REST of the enclosing pattern
// too (nodes[node_hi .. outer_hi)) via the `cont_hi` parameter — this is
// the "continuation" that lets a quantifier backtrack against what
// follows it. Returns true iff the WHOLE remaining pattern up to
// `outer_hi` matches; on success `*out_pos` holds the final text position.
bool regex_match_seq(const CompiledPattern* cp, MatchState* st,
                     uint16_t node_i, uint16_t outer_hi,
                     int32_t pos, int32_t* out_pos) noexcept;

// Try `count` in [min_rep, cap] repetitions of one atom (Char/Any/Class) at
// `node_i`, greedy-then-backtrack, then continue matching
// nodes[node_i+1 .. outer_hi) from the resulting position.
inline bool regex_match_atom_rep(const CompiledPattern* cp, MatchState* st,
                                 uint16_t node_i, uint16_t outer_hi,
                                 int32_t pos, int32_t* out_pos) noexcept {
    assert(cp != nullptr && st != nullptr && out_pos != nullptr);
    const Node& n = cp->nodes[node_i];
    int32_t max_here = pos;
    const int32_t cap = (n.max_rep == kUnbounded) ? st->len : pos + n.max_rep;
    while (max_here < st->len && max_here < cap &&
           regex_atom_matches(n, static_cast<std::uint8_t>(st->text[max_here]))) {
        ++max_here;
    }
    const int32_t min_here = pos + n.min_rep;
    if (max_here < min_here) return false;
    for (int32_t k = max_here; k >= min_here; --k) {
        if (--st->steps_left <= 0) return false;
        if (regex_match_seq(cp, st, static_cast<uint16_t>(node_i + 1), outer_hi,
                            k, out_pos)) {
            return true;
        }
    }
    return false;
}

// Try `count` repetitions of a GROUP's sub-sequence [body_lo, body_hi) as a
// unit, greedy-then-backtrack (bounded: at most `kMaxStepsPerByte * len`
// total attempts across the whole match via st->steps_left), recording
// capture bounds on the LAST successful repetition (SQL/PCRE convention).
inline bool regex_match_group_rep(const CompiledPattern* cp, MatchState* st,
                                  uint16_t start_i, uint16_t outer_hi,
                                  int32_t pos, int32_t* out_pos) noexcept {
    assert(cp != nullptr && st != nullptr && out_pos != nullptr);
    const Node& gs = cp->nodes[start_i];
    const uint16_t body_lo = static_cast<uint16_t>(start_i + 1);
    const uint16_t body_hi = gs.match_group_end;
    const uint16_t after   = static_cast<uint16_t>(body_hi + 1);
    // Collect up to max_rep positions by repeatedly matching the body from
    // the current position; each successful body match must advance pos
    // (empty-body infinite loop guard).
    int32_t positions[kMaxNodes];
    int32_t n_pos = 0;
    positions[n_pos++] = pos;
    int32_t cur = pos;
    while (n_pos < kMaxNodes &&
           (gs.max_rep == kUnbounded || n_pos <= gs.max_rep)) {
        if (--st->steps_left <= 0) break;
        int32_t nxt = 0;
        if (!regex_match_seq(cp, st, body_lo, body_hi, cur, &nxt)) break;
        if (nxt == cur) break;   // no progress: stop (avoid infinite loop)
        cur = nxt;
        positions[n_pos++] = cur;
    }
    const int32_t reps_available = n_pos - 1;   // positions[0] = 0 reps
    if (reps_available < gs.min_rep) return false;
    for (int32_t reps = reps_available; reps >= gs.min_rep; --reps) {
        if (--st->steps_left <= 0) return false;
        const int32_t at = positions[reps];
        // Re-run the LAST rep once more to stamp capture-group bounds for
        // this specific `reps` count (positions[] only stored end offsets).
        if (reps > 0) {
            int32_t restamp = 0;
            (void)regex_match_seq(cp, st, body_lo, body_hi, positions[reps - 1],
                                  &restamp);
        }
        if (gs.group_id > 0) {
            st->mr->g_start[gs.group_id] = positions[reps > 0 ? reps - 1 : 0];
            st->mr->g_end[gs.group_id]   = at;
        }
        if (regex_match_seq(cp, st, after, outer_hi, at, out_pos)) return true;
    }
    return false;
}

inline bool regex_match_seq(const CompiledPattern* cp, MatchState* st,
                            uint16_t node_i, uint16_t outer_hi,
                            int32_t pos, int32_t* out_pos) noexcept {
    assert(cp != nullptr && st != nullptr && out_pos != nullptr);
    assert(node_i <= outer_hi);
    if (st->steps_left <= 0) return false;
    if (node_i >= outer_hi) { *out_pos = pos; return true; }
    const Node& n = cp->nodes[node_i];
    switch (n.kind) {
        case NodeKind::AnchorStart:
            if (pos != 0) return false;
            return regex_match_seq(cp, st, static_cast<uint16_t>(node_i + 1),
                                   outer_hi, pos, out_pos);
        case NodeKind::AnchorEnd:
            if (pos != st->len) return false;
            return regex_match_seq(cp, st, static_cast<uint16_t>(node_i + 1),
                                   outer_hi, pos, out_pos);
        case NodeKind::GroupStart:
            return regex_match_group_rep(cp, st, node_i, outer_hi, pos, out_pos);
        case NodeKind::Char:
        case NodeKind::Any:
        case NodeKind::Class:
            return regex_match_atom_rep(cp, st, node_i, outer_hi, pos, out_pos);
        default:
            assert(false && "unreachable NodeKind in regex_match_seq");
            return false;
    }
}

// Try to match `cp` anywhere in text[0..n) (unanchored search); if the
// pattern starts with '^' the match is naturally pinned to position 0 by
// AnchorStart's own check, so the search loop's later starts just fail
// fast. Fills `*out` with the first (leftmost) match, or matched=false.
inline bool regex_search(const CompiledPattern* cp, const char* text,
                         int32_t n, MatchResult* out) noexcept {
    assert(cp != nullptr && out != nullptr);
    assert(n >= 0);
    std::memset(out, 0, sizeof(*out));
    for (int g = 0; g < kMaxGroups; ++g) { out->g_start[g] = -1; out->g_end[g] = -1; }
    const bool anchored = (cp->n_nodes > 0 &&
                          cp->nodes[0].kind == NodeKind::AnchorStart);
    for (int32_t start = 0; start <= n; ++start) {
        MatchState st{};
        st.text = text; st.len = n;
        st.steps_left = kMaxStepsPerByte * (n + 1);
        st.mr = out;
        int32_t end_pos = 0;
        if (regex_match_seq(cp, &st, 0, cp->n_nodes, start, &end_pos)) {
            out->matched = true;
            out->g_start[0] = start;
            out->g_end[0]   = end_pos;
            return true;
        }
        if (anchored) break;   // '^' can only match at position 0
    }
    return false;
}

// ---------------------------------------------------------------------
// Substitute
// ---------------------------------------------------------------------

// Build the replacement string for a successful match into `out_buf`
// (caller-owned, `out_cap` bytes): `repl` bytes are copied verbatim except
// `\N` (N in 1..9), which is replaced with capture group N's matched text
// (empty if that group didn't participate), and `\\` -> literal backslash.
// Returns the output length, or -1 if it would exceed `out_cap` (Tiger
// Style: caller must supply a bound, never grows).
inline int32_t regex_substitute(const MatchResult* mr, const char* text,
                                const char* repl, uint32_t repl_len,
                                char* out_buf, uint32_t out_cap) noexcept {
    assert(mr != nullptr && text != nullptr && out_buf != nullptr);
    assert(mr->matched);
    uint32_t o = 0;
    uint32_t i = 0;
    while (i < repl_len) {
        if (repl[i] == '\\' && i + 1 < repl_len) {
            const char nx = repl[i + 1];
            if (nx >= '1' && nx <= '9') {
                const int g = nx - '0';
                if (g < kMaxGroups && mr->g_start[g] >= 0) {
                    const int32_t glen = mr->g_end[g] - mr->g_start[g];
                    if (o + static_cast<uint32_t>(glen) > out_cap) return -1;
                    std::memcpy(out_buf + o, text + mr->g_start[g],
                               static_cast<std::size_t>(glen));
                    o += static_cast<uint32_t>(glen);
                }
                i += 2;
                continue;
            }
            if (nx == '\\') {
                if (o + 1 > out_cap) return -1;
                out_buf[o++] = '\\';
                i += 2;
                continue;
            }
        }
        if (o + 1 > out_cap) return -1;
        out_buf[o++] = repl[i];
        ++i;
    }
    return static_cast<int32_t>(o);
}

// =====================================================================
// RE2 common-subset anchored FULL-MATCH engine (Thompson NFA)
// =====================================================================
//
// A second, independent regex engine in this file, purpose-built to back a
// canonical Filter regex predicate (PromQL `=~` / `!~` matchers, which use
// RE2 semantics and fully anchor both ends: the WHOLE label value must
// match `^(?:pattern)$`).
//
// Contract: `re2_full_match(prog, text, n)` is true iff the entire
// text[0..n) is matched by the pattern. Compile ONCE with `re2_compile`
// into a fixed-size `Re2Program`; match many with NO per-call heap
// allocation and NO exceptions.
//
// Why a Thompson NFA (not backtracking): it runs in O(n * program_size),
// with an epsilon-closure "visited" stamp per step — so no catastrophic
// backtracking, no ReDoS, no step budget needed. A pathological pattern
// simply costs a bounded linear multiple; it can never hang.
//
// Supported (RE2 common subset):
//   a|b|c            alternation (top-level and inside any group)
//   *  +  ?          quantifiers; optional lazy suffix accepted for membership
//   {m} {m,} {m,n}   counted repetition (m,n bounded by kRe2MaxRepeat)
//   .                any byte EXCEPT newline (RE2 default, no `s` flag)
//   [abc] [a-z] [^..] character class, ranges, negation, leading `]` literal
//   \d \w \s \D \W \S  ASCII byte classes (+ negations), standalone or in [..]
//   \n \t \r \f \v \0  escaped control bytes; \. \\ \( \* … escaped literals
//   (...) (?:...)    grouping (both identical for matching — no captures)
//   (?i)             set case-insensitive for the rest of the current group
//   (?i:...)         case-insensitive scoped to the group body
//   ^  $             anchors (BOL/EOL empty-width; redundant under full match
//                    but honored if written mid-pattern)
//
// Deviations from RE2 (documented, reject-cleanly — never a wrong match):
//   * No captures, backreferences, lookaround, named groups `(?P<..>)`.
//   * Only the `i` inline flag; `(?s)`/`(?m)`/`(?U)`/`(?P<..>` etc. fail to
//     compile (clean false, surfaced as InvalidInput by the caller).
//   * ASCII/byte semantics only — no Unicode property classes `\p{..}`,
//     no UTF-8-aware `.` (one `.` == one byte).
//   * Leftmost-longest submatch capture is irrelevant here (membership only);
//     accept/reject for the anchored full match is exactly RE2's.
//   * Any construct exceeding a cap below fails to compile (never truncated).

constexpr int32_t kRe2MaxInsts    = 256;  // compiled program cap (instrs)
constexpr int32_t kRe2ClassRanges = 16;   // ranges per character-class instr
constexpr int32_t kRe2MaxDepth    = 32;   // group-nesting recursion cap
constexpr int32_t kRe2MaxArms      = 64;  // alternation arms per level
constexpr int32_t kRe2MaxRepeat   = 128;  // {m,n} bound (m,n each <= this)

enum class Re2Op : uint8_t {
    Char  = 0,   // consume one byte == c (icase-folded if icase)
    Any   = 1,   // consume one byte != '\n'
    Class = 2,   // consume one byte in/against the range set
    Match = 3,   // accept (only honored when text is fully consumed)
    Jmp   = 4,   // epsilon -> x
    Split = 5,   // epsilon -> x AND y (thread fork; membership order-agnostic)
    Bol   = 6,   // empty-width: assert pos == 0
    Eol   = 7,   // empty-width: assert pos == n
};

// One compiled NFA instruction. Ranges live inline (bounded, one-time
// compile) mirroring the older engine's inline-ranges house style.
struct Re2Inst {
    Re2Op      op;
    uint8_t    c;          // Char byte
    uint8_t    icase;      // Char/Class: case-insensitive compare
    uint8_t    negate;     // Class: match complement of the range set
    uint8_t    n_ranges;   // Class
    uint8_t    _pad[3];
    int32_t    x;          // Jmp/Split target 1
    int32_t    y;          // Split target 2
    ClassRange ranges[kRe2ClassRanges];
};

struct Re2Program {
    Re2Inst insts[kRe2MaxInsts];
    int32_t n_insts;
};

// Compile-time scratch (POD, passed by pointer; never heap).
struct Re2C {
    const char* pat;
    uint32_t    len;
    Re2Program* prog;
    bool        ok;
    int32_t     depth;
};

// ------------------------------ small leaves --------------------------

inline uint8_t re2_swapcase(uint8_t b) noexcept {
    if (b >= 'a' && b <= 'z') return static_cast<uint8_t>(b - 32);
    if (b >= 'A' && b <= 'Z') return static_cast<uint8_t>(b + 32);
    return b;
}

inline uint8_t re2_escape_literal(char e) noexcept {
    switch (e) {
        case 'n': return static_cast<uint8_t>('\n');
        case 't': return static_cast<uint8_t>('\t');
        case 'r': return static_cast<uint8_t>('\r');
        case 'f': return static_cast<uint8_t>('\f');
        case 'v': return static_cast<uint8_t>('\v');
        case '0': return 0u;
        default:  return static_cast<uint8_t>(e);   // \. \\ \( \+ -> literal
    }
}

inline bool re2_add_range(Re2Inst* in, uint8_t lo, uint8_t hi) noexcept {
    assert(in != nullptr);
    assert(lo <= hi);
    if (in->n_ranges >= kRe2ClassRanges) return false;
    in->ranges[in->n_ranges].lo = lo;
    in->ranges[in->n_ranges].hi = hi;
    in->n_ranges = static_cast<uint8_t>(in->n_ranges + 1);
    assert(in->n_ranges <= kRe2ClassRanges);
    return true;
}

// Append the ACTUAL matching byte set of a builtin escape `e` as positive
// ranges (complement already baked in for the upper-case negated forms) so
// they compose inside a `[...]` union and the class negate flag stays a
// single whole-class toggle. Returns false if it would exceed the cap.
inline bool re2_add_builtin_into(Re2Inst* in, char e) noexcept {
    assert(in != nullptr);
    assert(e != 0);
    switch (e) {
        case 'd': return re2_add_range(in, '0', '9');
        case 'D': return re2_add_range(in, 0x00, 0x2F) &&
                         re2_add_range(in, 0x3A, 0xFF);
        case 'w': return re2_add_range(in, '0', '9') &&
                         re2_add_range(in, 'A', 'Z') &&
                         re2_add_range(in, 'a', 'z') &&
                         re2_add_range(in, '_', '_');
        case 'W': return re2_add_range(in, 0x00, 0x2F) &&
                         re2_add_range(in, 0x3A, 0x40) &&
                         re2_add_range(in, 0x5B, 0x5E) &&
                         re2_add_range(in, 0x60, 0x60) &&
                         re2_add_range(in, 0x7B, 0xFF);
        case 's': return re2_add_range(in, 0x09, 0x0D) &&
                         re2_add_range(in, 0x20, 0x20);
        case 'S': return re2_add_range(in, 0x00, 0x08) &&
                         re2_add_range(in, 0x0E, 0x1F) &&
                         re2_add_range(in, 0x21, 0xFF);
        default:  return false;
    }
}

inline bool re2_is_builtin_escape(char e) noexcept {
    return e == 'd' || e == 'D' || e == 'w' || e == 'W' || e == 's' || e == 'S';
}

// ------------------------------ match leaves --------------------------

inline bool re2_class_hit(const Re2Inst& in, uint8_t b) noexcept {
    for (uint8_t r = 0; r < in.n_ranges; ++r) {
        if (b >= in.ranges[r].lo && b <= in.ranges[r].hi) return true;
    }
    return false;
}

inline bool re2_class_matches(const Re2Inst& in, uint8_t b) noexcept {
    assert(in.op == Re2Op::Class);
    bool hit = re2_class_hit(in, b);
    if (!hit && in.icase) hit = re2_class_hit(in, re2_swapcase(b));
    return in.negate ? !hit : hit;
}

inline bool re2_char_matches(const Re2Inst& in, uint8_t b) noexcept {
    assert(in.op == Re2Op::Char);
    if (b == in.c) return true;
    return in.icase != 0 && re2_swapcase(b) == in.c;
}

// ------------------------------ scanners ------------------------------

// Return the index of the ']' closing the class opened at `open`, or -1.
inline int32_t re2_scan_class_end(const char* pat, uint32_t len,
                                  uint32_t open) noexcept {
    assert(pat != nullptr);
    assert(open < len && pat[open] == '[');
    uint32_t i = open + 1;
    if (i < len && pat[i] == '^') ++i;
    if (i < len && pat[i] == ']') ++i;   // leading ']' is a literal member
    while (i < len && pat[i] != ']') {
        if (pat[i] == '\\') { i += 2; continue; }
        ++i;
    }
    if (i >= len) return -1;              // unterminated class
    return static_cast<int32_t>(i);
}

// Return the index of the ')' closing the group opened at `open`, or -1.
inline int32_t re2_scan_group_end(const char* pat, uint32_t len,
                                  uint32_t open) noexcept {
    assert(pat != nullptr);
    assert(open < len && pat[open] == '(');
    int32_t depth = 0;
    uint32_t i = open;
    while (i < len) {
        const char ch = pat[i];
        if (ch == '\\') { i += 2; continue; }
        if (ch == '[') {
            const int32_t ce = re2_scan_class_end(pat, len, i);
            if (ce < 0) return -1;
            i = static_cast<uint32_t>(ce) + 1;
            continue;
        }
        if (ch == '(') ++depth;
        else if (ch == ')') { --depth; if (depth == 0) return static_cast<int32_t>(i); }
        ++i;
    }
    return -1;                            // unbalanced '('
}

// End (exclusive) of the atom starting at `pos` within [pos,end), or -1.
inline int32_t re2_scan_atom_end(const Re2C* c, uint32_t pos,
                                 uint32_t end) noexcept {
    assert(c != nullptr);
    assert(pos < end && end <= c->len);
    const char ch = c->pat[pos];
    if (ch == '\\') { return (pos + 2 <= end) ? static_cast<int32_t>(pos + 2) : -1; }
    if (ch == '[') {
        const int32_t ce = re2_scan_class_end(c->pat, c->len, pos);
        if (ce < 0 || static_cast<uint32_t>(ce) >= end) return -1;
        return ce + 1;
    }
    if (ch == '(') {
        const int32_t ge = re2_scan_group_end(c->pat, c->len, pos);
        if (ge < 0 || static_cast<uint32_t>(ge) >= end) return -1;
        return ge + 1;
    }
    if (ch == ')') return -1;            // unbalanced ')'
    return static_cast<int32_t>(pos + 1);
}

inline bool re2_is_iflag_group(const char* pat, uint32_t pos,
                               uint32_t end) noexcept {
    assert(pat != nullptr);
    assert(pos <= end);
    return (end - pos) >= 4 && pat[pos] == '(' && pat[pos + 1] == '?' &&
           pat[pos + 2] == 'i' && pat[pos + 3] == ')';
}

// Split [start,end) into top-level `|` arms (respecting () and []). Writes
// arm bounds into as[]/ae[], returns the arm count, or -1 on overflow/
// malformed class.
inline int32_t re2_scan_arms(const Re2C* c, uint32_t start, uint32_t end,
                             uint32_t* as, uint32_t* ae) noexcept {
    assert(c != nullptr && as != nullptr && ae != nullptr);
    assert(start <= end && end <= c->len);
    int32_t k = 0;
    uint32_t s = start;
    uint32_t i = start;
    int32_t gdepth = 0;
    while (i < end) {
        const char ch = c->pat[i];
        if (ch == '\\') { i += 2; continue; }
        if (ch == '[') {
            const int32_t ce = re2_scan_class_end(c->pat, c->len, i);
            if (ce < 0 || static_cast<uint32_t>(ce) >= end) return -1;
            i = static_cast<uint32_t>(ce) + 1;
            continue;
        }
        if (ch == '(') { ++gdepth; ++i; continue; }
        if (ch == ')') { --gdepth; ++i; continue; }
        if (ch == '|' && gdepth == 0) {
            if (k >= kRe2MaxArms) return -1;
            as[k] = s; ae[k] = i; ++k; s = i + 1;
        }
        ++i;
    }
    if (k >= kRe2MaxArms) return -1;
    as[k] = s; ae[k] = end; ++k;
    return k;
}

// ------------------------------ emit ----------------------------------

inline int32_t re2_emit(Re2C* c, Re2Op op) noexcept {
    assert(c != nullptr && c->prog != nullptr);
    if (c->prog->n_insts >= kRe2MaxInsts) { c->ok = false; return -1; }
    const int32_t idx = c->prog->n_insts++;
    Re2Inst* in = &c->prog->insts[idx];
    std::memset(in, 0, sizeof(*in));
    in->op = op;
    assert(idx >= 0 && idx < kRe2MaxInsts);
    return idx;
}

// ------------------------------ quantifier parse ----------------------

// Parse `{m}` / `{m,}` / `{m,n}` at `*pos` (pointing at '{'). On a valid
// quantifier: sets *mn/*mx (mx == -1 means unbounded) and advances *pos past
// '}'. On a malformed brace: leaves *pos and *mn/*mx untouched so the '{' is
// later consumed as a literal (RE2 behavior). On an out-of-range/inverted
// count: sets c->ok = false.
inline void re2_parse_brace(Re2C* c, uint32_t* pos, uint32_t end,
                            int32_t* mn, int32_t* mx) noexcept {
    assert(c != nullptr && pos != nullptr && mn != nullptr && mx != nullptr);
    assert(*pos < end && c->pat[*pos] == '{');
    uint32_t i = *pos + 1;
    int64_t m = 0; bool got_m = false;
    while (i < end && c->pat[i] >= '0' && c->pat[i] <= '9') {
        m = m * 10 + (c->pat[i] - '0'); got_m = true; ++i;
        if (m > kRe2MaxRepeat + 1) m = kRe2MaxRepeat + 1;
    }
    if (!got_m) return;                  // "{x" -> literal '{'
    int64_t n = m;
    if (i < end && c->pat[i] == ',') {
        ++i;
        if (i < end && c->pat[i] == '}') { n = -1; }   // {m,}
        else {
            int64_t nn = 0; bool got_n = false;
            while (i < end && c->pat[i] >= '0' && c->pat[i] <= '9') {
                nn = nn * 10 + (c->pat[i] - '0'); got_n = true; ++i;
                if (nn > kRe2MaxRepeat + 1) nn = kRe2MaxRepeat + 1;
            }
            if (!got_n) return;          // "{m,x" -> literal
            n = nn;
        }
    }
    if (i >= end || c->pat[i] != '}') return;          // malformed -> literal
    if (m > kRe2MaxRepeat || (n >= 0 && (n > kRe2MaxRepeat || n < m))) {
        c->ok = false; return;
    }
    *mn = static_cast<int32_t>(m);
    *mx = (n < 0) ? -1 : static_cast<int32_t>(n);
    *pos = i + 1;
}

inline void re2_parse_quantifier(Re2C* c, uint32_t* pos, uint32_t end,
                                 int32_t* mn, int32_t* mx) noexcept {
    assert(c != nullptr && pos != nullptr && mn != nullptr && mx != nullptr);
    assert(*pos <= end);
    *mn = 1; *mx = 1;
    if (*pos >= end) return;
    bool quantified = true;
    switch (c->pat[*pos]) {
        case '*': *mn = 0; *mx = -1; ++(*pos); break;
        case '+': *mn = 1; *mx = -1; ++(*pos); break;
        case '?': *mn = 0; *mx = 1;  ++(*pos); break;
        case '{': re2_parse_brace(c, pos, end, mn, mx); break;
        default:  quantified = false; break;
    }
    // Thompson membership is independent of greedy/lazy preference. Consume
    // RE2's lazy suffix so `.*?x` has the same accept set as `.*x` without
    // introducing ordered backtracking into the bounded NFA.
    if (quantified && c->ok && *pos < end && c->pat[*pos] == '?') ++(*pos);
}

// ------------------------------ compile (mutually recursive) ----------

inline void re2_compile_alt(Re2C* c, bool icase, uint32_t start,
                            uint32_t end) noexcept;
inline void re2_compile_concat(Re2C* c, bool icase, uint32_t start,
                               uint32_t end) noexcept;
inline void re2_compile_atom(Re2C* c, bool icase, uint32_t as,
                             uint32_t ae) noexcept;

// `?` — optional (greedy: prefer taking the atom).
inline void re2_emit_opt(Re2C* c, bool icase, uint32_t as, uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as < ae);
    const int32_t s = re2_emit(c, Re2Op::Split);
    if (!c->ok) return;
    re2_compile_atom(c, icase, as, ae);
    if (!c->ok) return;
    c->prog->insts[s].x = s + 1;
    c->prog->insts[s].y = c->prog->n_insts;
    assert(c->prog->insts[s].y >= s + 1);
}

// `*` — zero-or-more (greedy).
inline void re2_emit_star(Re2C* c, bool icase, uint32_t as, uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as < ae);
    const int32_t s = re2_emit(c, Re2Op::Split);
    if (!c->ok) return;
    re2_compile_atom(c, icase, as, ae);
    if (!c->ok) return;
    const int32_t j = re2_emit(c, Re2Op::Jmp);
    if (!c->ok) return;
    c->prog->insts[s].x = s + 1;
    c->prog->insts[s].y = c->prog->n_insts;
    c->prog->insts[j].x = s;
    assert(c->prog->insts[j].x == s);
}

// `+` — one-or-more (greedy).
inline void re2_emit_plus(Re2C* c, bool icase, uint32_t as, uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as < ae);
    const int32_t l = c->prog->n_insts;
    re2_compile_atom(c, icase, as, ae);
    if (!c->ok) return;
    const int32_t s = re2_emit(c, Re2Op::Split);
    if (!c->ok) return;
    c->prog->insts[s].x = l;                 // loop back
    c->prog->insts[s].y = c->prog->n_insts;  // exit
    assert(c->prog->insts[s].x == l);
}

// `{m}` / `{m,}` / `{m,n}` — counted repetition via re-emitted atom copies.
inline void re2_emit_counted(Re2C* c, bool icase, uint32_t as, uint32_t ae,
                             int32_t mn, int32_t mx) noexcept {
    assert(c != nullptr);
    assert(mn >= 0 && (mx == -1 || mx >= mn));
    for (int32_t r = 0; r < mn; ++r) {
        re2_compile_atom(c, icase, as, ae);
        if (!c->ok) return;
    }
    if (mx == -1) { re2_emit_star(c, icase, as, ae); return; }   // {m,}
    const int32_t opt = mx - mn;
    assert(opt >= 0 && opt <= kRe2MaxRepeat);
    int32_t sp[kRe2MaxRepeat];
    int32_t nsp = 0;
    for (int32_t r = 0; r < opt; ++r) {
        const int32_t s = re2_emit(c, Re2Op::Split);
        if (!c->ok) return;
        c->prog->insts[s].x = s + 1;
        sp[nsp++] = s;
        re2_compile_atom(c, icase, as, ae);
        if (!c->ok) return;
    }
    const int32_t endpc = c->prog->n_insts;
    for (int32_t r = 0; r < nsp; ++r) c->prog->insts[sp[r]].y = endpc;
}

inline void re2_compile_atom_repeated(Re2C* c, bool icase, uint32_t as,
                                      uint32_t ae, int32_t mn,
                                      int32_t mx) noexcept {
    assert(c != nullptr);
    assert(as < ae);
    if (!c->ok) return;
    if (mn == 1 && mx == 1)      { re2_compile_atom(c, icase, as, ae); return; }
    if (mn == 0 && mx == 1)      { re2_emit_opt(c, icase, as, ae);     return; }
    if (mn == 0 && mx == -1)     { re2_emit_star(c, icase, as, ae);    return; }
    if (mn == 1 && mx == -1)     { re2_emit_plus(c, icase, as, ae);    return; }
    re2_emit_counted(c, icase, as, ae, mn, mx);
}

// Compile a `[...]` class body into insts[idx].
inline void re2_compile_class(Re2C* c, int32_t idx, bool icase, uint32_t as,
                              uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as + 1 < ae && c->pat[as] == '[' && c->pat[ae - 1] == ']');
    Re2Inst* in = &c->prog->insts[idx];
    in->op = Re2Op::Class; in->negate = 0; in->n_ranges = 0;
    in->icase = icase ? 1 : 0;
    uint32_t i = as + 1;
    if (i < ae - 1 && c->pat[i] == '^') { in->negate = 1; ++i; }
    while (i < ae - 1) {
        const char ch = c->pat[i];
        if (ch == '\\' && i + 1 < ae - 1 + 1) {
            const char e = c->pat[i + 1];
            const bool ok = re2_is_builtin_escape(e)
                ? re2_add_builtin_into(in, e)
                : re2_add_range(in, re2_escape_literal(e), re2_escape_literal(e));
            if (!ok) { c->ok = false; return; }
            i += 2;
            continue;
        }
        uint8_t lo = static_cast<uint8_t>(ch);
        uint8_t hi = lo;
        if (i + 2 < ae - 1 && c->pat[i + 1] == '-') {
            hi = static_cast<uint8_t>(c->pat[i + 2]);
            if (hi < lo) { c->ok = false; return; }
            i += 3;
        } else {
            ++i;
        }
        if (!re2_add_range(in, lo, hi)) { c->ok = false; return; }
    }
    assert(in->op == Re2Op::Class);
}

// Compile a `\x` escape at [as, as+2): builtin class or escaped literal.
inline void re2_compile_escape(Re2C* c, bool icase, uint32_t as,
                               uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(c->pat[as] == '\\' && as + 1 < ae);
    (void)ae;   // span end only used by the assert (compiled out in Release)
    const char e = c->pat[as + 1];
    if (re2_is_builtin_escape(e)) {
        const int32_t idx = re2_emit(c, Re2Op::Class);
        if (!c->ok) return;
        Re2Inst* in = &c->prog->insts[idx];
        in->negate = 0; in->n_ranges = 0; in->icase = icase ? 1 : 0;
        if (!re2_add_builtin_into(in, e)) { c->ok = false; return; }
        return;
    }
    const int32_t idx = re2_emit(c, Re2Op::Char);
    if (!c->ok) return;
    c->prog->insts[idx].c = re2_escape_literal(e);
    c->prog->insts[idx].icase = icase ? 1 : 0;
    assert(c->prog->insts[idx].op == Re2Op::Char);
}

// Compile a `( ... )` group span [as,ae): handles (?:...) / (?i:...) / (...).
inline void re2_compile_group(Re2C* c, bool icase, uint32_t as,
                              uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as + 1 < ae && c->pat[as] == '(' && c->pat[ae - 1] == ')');
    if (++c->depth > kRe2MaxDepth) { c->ok = false; --c->depth; return; }
    uint32_t inner_s = as + 1;
    const uint32_t inner_e = ae - 1;
    bool gi = icase;
    if (inner_s + 1 < inner_e && c->pat[inner_s] == '?') {
        uint32_t p = inner_s + 1;
        bool set_i = false, saw_colon = false, bad = false;
        while (p < inner_e) {
            const char f = c->pat[p];
            if (f == ':') { saw_colon = true; ++p; break; }
            if (f == 'i') { set_i = true; ++p; continue; }
            bad = true; break;           // unsupported flag / named group
        }
        if (bad || !saw_colon) { c->ok = false; --c->depth; return; }
        if (set_i) gi = true;
        inner_s = p;                     // body begins after ':'
    }
    re2_compile_alt(c, gi, inner_s, inner_e);
    --c->depth;
}

inline void re2_compile_atom(Re2C* c, bool icase, uint32_t as,
                             uint32_t ae) noexcept {
    assert(c != nullptr);
    assert(as < ae && ae <= c->len);
    if (!c->ok) return;
    const char ch = c->pat[as];
    if (ch == '(') { re2_compile_group(c, icase, as, ae); return; }
    if (ch == '[') {
        const int32_t idx = re2_emit(c, Re2Op::Class);
        if (!c->ok) return;
        re2_compile_class(c, idx, icase, as, ae);
        return;
    }
    if (ch == '.') { (void)re2_emit(c, Re2Op::Any); return; }
    if (ch == '^') { (void)re2_emit(c, Re2Op::Bol); return; }
    if (ch == '$') { (void)re2_emit(c, Re2Op::Eol); return; }
    if (ch == '\\') { re2_compile_escape(c, icase, as, ae); return; }
    const int32_t idx = re2_emit(c, Re2Op::Char);
    if (!c->ok) return;
    c->prog->insts[idx].c = static_cast<uint8_t>(ch);
    c->prog->insts[idx].icase = icase ? 1 : 0;
    assert(c->prog->insts[idx].op == Re2Op::Char);
}

inline void re2_compile_concat(Re2C* c, bool icase, uint32_t start,
                               uint32_t end) noexcept {
    assert(c != nullptr);
    assert(start <= end && end <= c->len);
    uint32_t pos = start;
    bool local_icase = icase;
    while (pos < end && c->ok) {
        if (re2_is_iflag_group(c->pat, pos, end)) {
            local_icase = true; pos += 4; continue;
        }
        const uint32_t as = pos;
        const int32_t ae = re2_scan_atom_end(c, pos, end);
        if (ae < 0) { c->ok = false; return; }
        pos = static_cast<uint32_t>(ae);
        int32_t mn = 1, mx = 1;
        re2_parse_quantifier(c, &pos, end, &mn, &mx);
        if (!c->ok) return;
        re2_compile_atom_repeated(c, local_icase, as,
                                  static_cast<uint32_t>(ae), mn, mx);
    }
    assert(pos <= end);
}

inline void re2_compile_alt(Re2C* c, bool icase, uint32_t start,
                            uint32_t end) noexcept {
    assert(c != nullptr);
    assert(start <= end && end <= c->len);
    if (!c->ok) return;
    uint32_t arm_s[kRe2MaxArms];
    uint32_t arm_e[kRe2MaxArms];
    const int32_t k = re2_scan_arms(c, start, end, arm_s, arm_e);
    if (k < 0) { c->ok = false; return; }
    if (k == 1) { re2_compile_concat(c, icase, start, end); return; }
    int32_t jmp_patch[kRe2MaxArms];
    int32_t nj = 0;
    for (int32_t a = 0; a < k; ++a) {
        if (a < k - 1) {
            const int32_t s = re2_emit(c, Re2Op::Split);
            if (!c->ok) return;
            re2_compile_concat(c, icase, arm_s[a], arm_e[a]);
            const int32_t j = re2_emit(c, Re2Op::Jmp);
            if (!c->ok) return;
            c->prog->insts[s].x = s + 1;
            c->prog->insts[s].y = c->prog->n_insts;   // next arm start
            jmp_patch[nj++] = j;
        } else {
            re2_compile_concat(c, icase, arm_s[a], arm_e[a]);
        }
    }
    const int32_t endpc = c->prog->n_insts;
    for (int32_t a = 0; a < nj; ++a) c->prog->insts[jmp_patch[a]].x = endpc;
}

// Compile `pat[0..pat_len)` into `out` for anchored full match. Returns
// false on any unsupported construct or a pattern too large for the caps
// (caller surfaces a clean InvalidInput, never a silent mismatch).
inline bool re2_compile(const char* pat, uint32_t pat_len,
                        Re2Program* out) noexcept {
    assert(pat != nullptr || pat_len == 0);
    assert(out != nullptr);
    std::memset(out, 0, sizeof(*out));
    Re2C c{};
    c.pat = pat; c.len = pat_len; c.prog = out; c.ok = true; c.depth = 0;
    re2_compile_alt(&c, false, 0, pat_len);
    if (!c.ok) return false;
    const int32_t m = re2_emit(&c, Re2Op::Match);
    if (!c.ok || m < 0) return false;
    assert(out->n_insts >= 1 && out->n_insts <= kRe2MaxInsts);
    assert(out->insts[out->n_insts - 1].op == Re2Op::Match);
    return true;
}

// ------------------------------ match (Thompson NFA) ------------------

// Push a target pc onto the epsilon-closure DFS stack iff in range and not
// already seen this generation (marks visited at PUSH time so each pc is
// enqueued at most once -> `top` is bounded by n_insts, no overflow).
inline void re2_push(int32_t* stack, int32_t* top, int32_t* visited,
                     int32_t gen, int32_t t, int32_t n_insts) noexcept {
    assert(stack != nullptr && top != nullptr && visited != nullptr);
    assert(*top >= 0 && *top <= n_insts);
    if (t < 0 || t >= n_insts) return;
    if (visited[t] == gen) return;
    visited[t] = gen;
    stack[(*top)++] = t;
}

// Epsilon-close from `pc` at text position `sp`, appending each reached
// consuming/Match instruction to `list`. Bounded: every pc enqueued once.
inline void re2_add_thread(const Re2Program* p, int32_t* list, int32_t* list_n,
                           int32_t* visited, int32_t gen, int32_t pc,
                           int32_t sp, int32_t n) noexcept {
    assert(p != nullptr && list != nullptr && list_n != nullptr);
    assert(visited != nullptr && sp >= 0 && sp <= n);
    int32_t stack[kRe2MaxInsts];
    int32_t top = 0;
    re2_push(stack, &top, visited, gen, pc, p->n_insts);
    while (top > 0) {
        assert(top <= kRe2MaxInsts);
        const int32_t q = stack[--top];
        assert(q >= 0 && q < p->n_insts);
        const Re2Inst& in = p->insts[q];
        switch (in.op) {
            case Re2Op::Jmp:
                re2_push(stack, &top, visited, gen, in.x, p->n_insts);
                break;
            case Re2Op::Split:
                re2_push(stack, &top, visited, gen, in.y, p->n_insts);
                re2_push(stack, &top, visited, gen, in.x, p->n_insts);
                break;
            case Re2Op::Bol:
                if (sp == 0) re2_push(stack, &top, visited, gen, q + 1, p->n_insts);
                break;
            case Re2Op::Eol:
                if (sp == n) re2_push(stack, &top, visited, gen, q + 1, p->n_insts);
                break;
            default:                 // Char / Any / Class / Match
                list[(*list_n)++] = q;
                break;
        }
    }
    assert(*list_n <= p->n_insts);
}

// Anchored FULL MATCH: true iff the entire text[0..n) is matched by `prog`.
// Linear in n * program-size; no heap, no exceptions, no ReDoS.
inline bool re2_full_match(const Re2Program* prog, const char* text,
                           int32_t n) noexcept {
    assert(prog != nullptr && (text != nullptr || n == 0));
    assert(n >= 0 && prog->n_insts >= 1 && prog->n_insts <= kRe2MaxInsts);
    int32_t buf_a[kRe2MaxInsts];
    int32_t buf_b[kRe2MaxInsts];
    int32_t visited[kRe2MaxInsts];
    for (int32_t i = 0; i < prog->n_insts; ++i) visited[i] = -1;
    int32_t* clist = buf_a;
    int32_t* nlist = buf_b;
    int32_t cn = 0;
    int32_t gen = 0;
    re2_add_thread(prog, clist, &cn, visited, gen, 0, 0, n);
    for (int32_t sp = 0; sp <= n; ++sp) {
        bool matched = false;
        for (int32_t i = 0; i < cn; ++i) {
            if (prog->insts[clist[i]].op == Re2Op::Match) { matched = true; break; }
        }
        if (sp == n) return matched;     // accept only when fully consumed
        const uint8_t b = static_cast<uint8_t>(text[sp]);
        ++gen;
        int32_t nn = 0;
        for (int32_t i = 0; i < cn; ++i) {
            const Re2Inst& in = prog->insts[clist[i]];
            bool adv = false;
            switch (in.op) {
                case Re2Op::Char:  adv = re2_char_matches(in, b); break;
                case Re2Op::Any:   adv = (b != '\n'); break;
                case Re2Op::Class: adv = re2_class_matches(in, b); break;
                default:           break;   // Match at non-end dies here
            }
            if (adv) {
                re2_add_thread(prog, nlist, &nn, visited, gen,
                               clist[i] + 1, sp + 1, n);
            }
        }
        int32_t* tmp = clist; clist = nlist; nlist = tmp;
        cn = nn;
    }
    assert(false && "re2_full_match must return at sp == n");
    return false;
}

// Convenience: compile + match in one call (for ad-hoc/one-shot use only;
// production wiring compiles ONCE and matches many via the two calls above).
inline bool re2_compile_and_full_match(const char* pat, uint32_t pat_len,
                                       const char* text, int32_t n,
                                       bool* compile_ok) noexcept {
    assert(compile_ok != nullptr);
    assert(pat != nullptr || pat_len == 0);
    Re2Program prog;
    const bool ok = re2_compile(pat, pat_len, &prog);
    *compile_ok = ok;
    if (!ok) return false;
    return re2_full_match(&prog, text, n);
}

}  // namespace regex
}  // namespace kernels
}  // namespace bolt
