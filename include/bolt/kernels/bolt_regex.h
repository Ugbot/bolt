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
        case NodeKind::Any:   return true;
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

}  // namespace regex
}  // namespace kernels
}  // namespace bolt
