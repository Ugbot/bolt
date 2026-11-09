// bolt_query_options.h — one spelling of the per-query controls for every
// query language.
//
//   ... TIME LIMIT 5s
//   ... OPTIONS (time_limit = '5s', memory_limit = '2GB', workers = 4,
//                morsel_rows = 8192, priority = 'low')
//
// The clause trails the statement, so it is read here, off the tail, before
// any language's parser runs. A forward scan skips strings and comments in the
// dialect's own terms, then looks at the last top-level tokens. Nothing here is
// reserved: a column, alias or function called time, limit or options keeps
// working, because the clause only matches in a shape those cannot take
// (`LIMIT 5` has no unit; `OPTIONS (...)` needs known keys after a complete
// statement).
//
// Any Limits-registry knob is also a query option (G2LAUNCH-156):
//   SET dist_broadcast_mb = 256          -- the session
//   ... OPTIONS (dist_skew_permille = 20) -- this statement
//   params.options {"join_wave_mb": "64"} -- a gateway request
// A query carries its overrides in QueryOptions::knobs; resolution is query >
// session > process (set / env) > default. query_limit_value() reads the
// running query's value from its QueryCtx wherever the code runs, a worker
// included (the plan ships the overrides).
//
// Cold path: runs once per statement, no allocation.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "bolt/bolt_limits.h"
#include "bolt/bolt_query_ctx.h"

namespace bolt {

inline constexpr uint32_t kQueryKnobMax = 32;         // registry overrides one query carries
inline constexpr uint32_t kQueryKnobNameBytes = 48;
inline constexpr uint64_t kQueryKnobReset = UINT64_MAX;   // RESET name: drop the override

enum class QueryKnobSource : uint8_t { Unset = 0, Session = 1, Query = 2, Optimizer = 3 };

struct QueryKnob {
    char            name[kQueryKnobNameBytes];
    uint64_t        value;
    QueryKnobSource source;
    uint8_t         _pad[7];
};

struct QueryKnobs {
    uint32_t  count = 0;
    uint32_t  _pad = 0;
    QueryKnob knobs[kQueryKnobMax];
};

inline int32_t query_knob_index(const QueryKnobs& k, const char* name, size_t n) noexcept {
    for (uint32_t i = 0; i < k.count && i < kQueryKnobMax; ++i)
        if (std::strncmp(k.knobs[i].name, name, n) == 0 && k.knobs[i].name[n] == '\0') return static_cast<int32_t>(i);
    return -1;
}

/// The override of `name`, if this set carries one.
inline bool query_knob_get(const QueryKnobs& k, const char* name, uint64_t* value,
                           QueryKnobSource* source = nullptr) noexcept {
    const int32_t i = query_knob_index(k, name, std::strlen(name));
    if (i < 0) return false;
    *value = k.knobs[i].value;
    if (source != nullptr) *source = k.knobs[i].source;
    return true;
}

/// The registry row a knob name names: a Budget or FixedAtAlloc limit, not an
/// invariant. False for anything else.
inline bool query_knob_limit(const char* name, size_t n, const LimitTable** t, uint32_t* i) noexcept {
    if (n == 0 || n >= kQueryKnobNameBytes) return false;
    char z[kQueryKnobNameBytes];
    for (size_t j = 0; j < n; ++j) z[j] = (name[j] >= 'A' && name[j] <= 'Z') ? static_cast<char>(name[j] + 32) : name[j];
    z[n] = '\0';
    return limits_find(z, t, i) && (*t)->descs[*i].kind != LimitKind::kInvariant;
}

/// Set (or with kQueryKnobReset, drop) one override. False when the name is no
/// registry knob, the value is out of the row's range, or the set is full.
inline bool query_knob_put(QueryKnobs* k, const char* name, size_t n, uint64_t value,
                           QueryKnobSource source) noexcept {
    const LimitTable* t = nullptr;
    uint32_t row = 0;
    if (!query_knob_limit(name, n, &t, &row)) return false;
    char z[kQueryKnobNameBytes];
    std::snprintf(z, sizeof(z), "%s", t->descs[row].name);
    const int32_t at = query_knob_index(*k, z, std::strlen(z));
    if (value == kQueryKnobReset) {
        if (at >= 0) { k->knobs[at] = k->knobs[k->count - 1u]; --k->count; }
        return true;
    }
    if (!limits_in_range(t->descs[row], value)) return false;
    QueryKnob* slot = at >= 0 ? &k->knobs[at] : (k->count < kQueryKnobMax ? &k->knobs[k->count++] : nullptr);
    if (slot == nullptr) return false;
    std::snprintf(slot->name, sizeof(slot->name), "%s", z);
    slot->value = value;
    slot->source = source;
    return true;
}

/// `over` wins name by name; a reset entry drops the name from `base`.
inline void query_knobs_overlay(QueryKnobs* base, const QueryKnobs& over) noexcept {
    for (uint32_t i = 0; i < over.count && i < kQueryKnobMax; ++i) {
        const QueryKnob& o = over.knobs[i];
        (void)query_knob_put(base, o.name, std::strlen(o.name), o.value, o.source);
    }
}

/// Every entry's source set to `source` (a session's overrides, a query's).
inline void query_knobs_mark(QueryKnobs* k, QueryKnobSource source) noexcept {
    for (uint32_t i = 0; i < k->count && i < kQueryKnobMax; ++i) k->knobs[i].source = source;
}

/// Registry row `i` of `t` for the running query: its override when the
/// query (QueryCtx on this thread) carries one, else the process value.
inline uint64_t query_limit_value(const LimitTable& t, uint32_t i) noexcept {
    const QueryCtx* q = current_query_ctx();
    uint64_t v = 0;
    if (q != nullptr && q->knobs != nullptr && query_knob_get(*q->knobs, t.descs[i].name, &v) && v != kQueryKnobReset)
        return v;
    return limits_value(t, i);
}

enum class QueryDialect : uint8_t { Sql = 0, Cypher, Sparql, Promql };
enum class QueryPriorityClass : uint8_t { Unset = 0, Low, Normal, High };
enum class QueryNodes : uint8_t { Unset = 0, Local, Any, Workers };

/// Each field is set when its bit is. Zero is a real value for none of them.
struct QueryOptions {
    enum Bit : uint32_t {
        kTimeLimit = 1u, kMemoryLimit = 2u, kRowLimit = 4u, kWorkers = 8u,
        kMorselRows = 16u, kPriority = 32u, kNodes = 64u, kCancelOnDisconnect = 128u,
        kKnobs = 256u,
    };
    uint32_t set = 0;
    uint32_t workers = 0;
    uint64_t time_limit_ms = 0;     // 0 with the bit set: no limit (still under the cap)
    uint64_t memory_bytes = 0;
    uint64_t row_limit = 0;
    int64_t  morsel_rows = 0;
    QueryPriorityClass priority = QueryPriorityClass::Unset;
    QueryNodes nodes = QueryNodes::Unset;
    uint8_t  node_count = 0;        // nodes = N: candidates considered
    uint8_t  cancel_on_disconnect = 1;
    QueryKnobs knobs;               // Limits-registry overrides (kKnobs)
};

/// Field-wise: what `over` sets wins.
inline void query_options_overlay(QueryOptions* base, const QueryOptions& over) noexcept {
    const uint32_t s = over.set;
    if ((s & QueryOptions::kTimeLimit) != 0u) base->time_limit_ms = over.time_limit_ms;
    if ((s & QueryOptions::kMemoryLimit) != 0u) base->memory_bytes = over.memory_bytes;
    if ((s & QueryOptions::kRowLimit) != 0u) base->row_limit = over.row_limit;
    if ((s & QueryOptions::kWorkers) != 0u) base->workers = over.workers;
    if ((s & QueryOptions::kMorselRows) != 0u) base->morsel_rows = over.morsel_rows;
    if ((s & QueryOptions::kPriority) != 0u) base->priority = over.priority;
    if ((s & QueryOptions::kNodes) != 0u) { base->nodes = over.nodes; base->node_count = over.node_count; }
    if ((s & QueryOptions::kCancelOnDisconnect) != 0u) base->cancel_on_disconnect = over.cancel_on_disconnect;
    if ((s & QueryOptions::kKnobs) != 0u) query_knobs_overlay(&base->knobs, over.knobs);
    base->set |= s;
}

namespace query_options_detail {

inline char lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }
inline bool ident_start(char c) noexcept { const char l = lower(c); return (l >= 'a' && l <= 'z') || c == '_'; }
inline bool digit(char c) noexcept { return c >= '0' && c <= '9'; }
inline bool ident_char(char c) noexcept { return ident_start(c) || digit(c); }
inline bool space(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'; }

inline bool word_is(const char* s, size_t n, const char* word) noexcept {
    size_t i = 0;
    for (; i < n && word[i] != '\0'; ++i) if (lower(s[i]) != word[i]) return false;
    return i == n && word[i] == '\0';
}

// One digit run with an optional fraction, then the unit. Bounded by the caller.
inline bool number_prefix(const char* s, size_t n, uint64_t* whole, uint64_t* frac, uint64_t* frac_scale, size_t* used) noexcept {
    size_t i = 0;
    uint64_t v = 0;
    if (n == 0 || !digit(s[0])) return false;
    for (; i < n && digit(s[i]); ++i) {
        const uint64_t d = static_cast<uint64_t>(s[i] - '0');
        if (v > (UINT64_MAX - d) / 10u) return false;
        v = v * 10u + d;
    }
    uint64_t f = 0, scale = 1;
    if (i < n && s[i] == '.') {
        ++i;
        for (; i < n && digit(s[i]); ++i) {
            if (scale >= 1000000000ull) continue;
            f = f * 10u + static_cast<uint64_t>(s[i] - '0');
            scale *= 10u;
        }
    }
    *whole = v; *frac = f; *frac_scale = scale; *used = i;
    return true;
}

inline bool scaled(uint64_t whole, uint64_t frac, uint64_t scale, uint64_t unit, uint64_t* out) noexcept {
    if (whole != 0 && unit > UINT64_MAX / whole) return false;
    const uint64_t a = whole * unit;
    const uint64_t b = static_cast<uint64_t>(static_cast<double>(frac) / static_cast<double>(scale) * static_cast<double>(unit));
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

}  // namespace query_options_detail

/// "500ms", "5s", "2m", "1h", "1.5s". A bare number is refused: a time without
/// a unit is the mistake this clause exists to avoid.
inline bool parse_query_duration_ms(const char* s, size_t n, uint64_t* out_ms) noexcept {
    using namespace query_options_detail;
    uint64_t w = 0, f = 0, sc = 1;
    size_t i = 0;
    if (!number_prefix(s, n, &w, &f, &sc, &i) || i == n) return false;
    const char* u = s + i;
    const size_t un = n - i;
    uint64_t unit = 0;
    if (word_is(u, un, "ms")) unit = 1;
    else if (word_is(u, un, "s") || word_is(u, un, "sec")) unit = 1000;
    else if (word_is(u, un, "m") || word_is(u, un, "min")) unit = 60000;
    else if (word_is(u, un, "h")) unit = 3600000;
    else return false;
    return scaled(w, f, sc, unit, out_ms);
}

/// "2GB", "512MB", "64kb", "1TiB", or plain bytes. Binary units, as Postgres reads them.
inline bool parse_query_size_bytes(const char* s, size_t n, uint64_t* out) noexcept {
    using namespace query_options_detail;
    uint64_t w = 0, f = 0, sc = 1;
    size_t i = 0;
    if (!number_prefix(s, n, &w, &f, &sc, &i)) return false;
    const char* u = s + i;
    const size_t un = n - i;
    uint64_t unit = 0;
    if (un == 0 || word_is(u, un, "b")) unit = 1;
    else if (word_is(u, un, "kb") || word_is(u, un, "kib")) unit = 1ull << 10;
    else if (word_is(u, un, "mb") || word_is(u, un, "mib")) unit = 1ull << 20;
    else if (word_is(u, un, "gb") || word_is(u, un, "gib")) unit = 1ull << 30;
    else if (word_is(u, un, "tb") || word_is(u, un, "tib")) unit = 1ull << 40;
    else return false;
    return scaled(w, f, sc, unit, out);
}

enum class QueryOptionResult : uint8_t { Ok, UnknownKey, BadValue };

inline bool query_option_builtin(const char* k, size_t n) noexcept;

inline bool query_option_key_known(const char* k, size_t n) noexcept {
    const LimitTable* t = nullptr;
    uint32_t i = 0;
    return query_option_builtin(k, n) || query_knob_limit(k, n, &t, &i);
}

inline bool query_option_builtin(const char* k, size_t n) noexcept {
    using query_options_detail::word_is;
    return word_is(k, n, "time_limit") || word_is(k, n, "statement_timeout") ||
           word_is(k, n, "memory_limit") || word_is(k, n, "row_limit") ||
           word_is(k, n, "workers") || word_is(k, n, "morsel_rows") ||
           word_is(k, n, "priority") || word_is(k, n, "nodes") ||
           word_is(k, n, "cancel_on_disconnect");
}

inline uint32_t query_option_bit(const char* k, size_t n) noexcept {
    using query_options_detail::word_is;
    if (word_is(k, n, "time_limit") || word_is(k, n, "statement_timeout")) return QueryOptions::kTimeLimit;
    if (word_is(k, n, "memory_limit")) return QueryOptions::kMemoryLimit;
    if (word_is(k, n, "row_limit")) return QueryOptions::kRowLimit;
    if (word_is(k, n, "workers")) return QueryOptions::kWorkers;
    if (word_is(k, n, "morsel_rows")) return QueryOptions::kMorselRows;
    if (word_is(k, n, "priority")) return QueryOptions::kPriority;
    if (word_is(k, n, "nodes")) return QueryOptions::kNodes;
    if (word_is(k, n, "cancel_on_disconnect")) return QueryOptions::kCancelOnDisconnect;
    if (!query_option_builtin(k, n)) return QueryOptions::kKnobs;
    return 0u;
}

inline bool parse_query_uint(const char* v, size_t n, uint64_t* out) noexcept {
    using namespace query_options_detail;
    uint64_t w = 0, f = 0, sc = 1;
    size_t used = 0;
    if (!number_prefix(v, n, &w, &f, &sc, &used) || used != n || sc != 1) return false;
    *out = w;
    return true;
}

/// Apply one key = value. `value` is the text with its quotes removed.
inline QueryOptionResult apply_query_option(const char* key, size_t kn, const char* value, size_t vn,
                                            QueryOptions* o) noexcept {
    using query_options_detail::word_is;
    uint64_t n = 0;
    if (word_is(key, kn, "time_limit") || word_is(key, kn, "statement_timeout")) {
        // Postgres: a bare statement_timeout is milliseconds, 0 is off.
        if (word_is(key, kn, "statement_timeout") && parse_query_uint(value, vn, &n)) o->time_limit_ms = n;
        else if (word_is(value, vn, "none") || (vn == 1 && value[0] == '0')) o->time_limit_ms = 0;
        else if (!parse_query_duration_ms(value, vn, &o->time_limit_ms)) return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kTimeLimit;
    } else if (word_is(key, kn, "memory_limit")) {
        if (!parse_query_size_bytes(value, vn, &o->memory_bytes) || o->memory_bytes == 0) return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kMemoryLimit;
    } else if (word_is(key, kn, "row_limit")) {
        if (!parse_query_uint(value, vn, &o->row_limit)) return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kRowLimit;
    } else if (word_is(key, kn, "workers")) {
        if (!parse_query_uint(value, vn, &n) || n == 0 || n > UINT32_MAX) return QueryOptionResult::BadValue;
        o->workers = static_cast<uint32_t>(n);
        o->set |= QueryOptions::kWorkers;
    } else if (word_is(key, kn, "morsel_rows")) {
        if (!parse_query_uint(value, vn, &n) || n < 1024u || n > 65536u) return QueryOptionResult::BadValue;
        o->morsel_rows = static_cast<int64_t>(n);
        o->set |= QueryOptions::kMorselRows;
    } else if (word_is(key, kn, "priority")) {
        if (word_is(value, vn, "low")) o->priority = QueryPriorityClass::Low;
        else if (word_is(value, vn, "normal")) o->priority = QueryPriorityClass::Normal;
        else if (word_is(value, vn, "high")) o->priority = QueryPriorityClass::High;
        else return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kPriority;
    } else if (word_is(key, kn, "nodes")) {
        o->node_count = 0;
        if (word_is(value, vn, "local")) o->nodes = QueryNodes::Local;
        else if (word_is(value, vn, "any") || word_is(value, vn, "auto")) o->nodes = QueryNodes::Any;
        else if (word_is(value, vn, "workers")) o->nodes = QueryNodes::Workers;
        else if (parse_query_uint(value, vn, &n) && n >= 1u && n <= 255u) {
            o->nodes = n == 1u ? QueryNodes::Local : QueryNodes::Any;
            o->node_count = static_cast<uint8_t>(n);
        } else return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kNodes;
    } else if (word_is(key, kn, "cancel_on_disconnect")) {
        if (word_is(value, vn, "true") || word_is(value, vn, "on")) o->cancel_on_disconnect = 1;
        else if (word_is(value, vn, "false") || word_is(value, vn, "off")) o->cancel_on_disconnect = 0;
        else return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kCancelOnDisconnect;
    } else {
        const LimitTable* t = nullptr;
        uint32_t row = 0;
        if (!query_knob_limit(key, kn, &t, &row)) return QueryOptionResult::UnknownKey;
        if (word_is(value, vn, "default")) n = kQueryKnobReset;
        else if (!parse_query_uint(value, vn, &n)) return QueryOptionResult::BadValue;
        if (!query_knob_put(&o->knobs, key, kn, n, QueryKnobSource::Query)) return QueryOptionResult::BadValue;
        o->set |= QueryOptions::kKnobs;
    }
    return QueryOptionResult::Ok;
}

enum class QueryClauseResult : uint8_t {
    None,      // no clause: *body_len is the whole text
    Found,     // options filled, *body_len excludes the clause
    Invalid,   // a clause with a bad key or value; `why` says which
};

namespace query_options_detail {

enum class Tok : uint8_t { Word, Number, String, Group, Punct };
struct Token { uint32_t begin; uint32_t end; Tok kind; };

constexpr uint32_t kTail = 8;

struct TailRing {
    Token    t[kTail];
    uint32_t count = 0;   // tokens seen
    void push(Token x) noexcept { t[count++ % kTail] = x; }
    // i = 0 is the last token.
    const Token* back(uint32_t i) const noexcept { return i < count && i < kTail ? &t[(count - 1 - i) % kTail] : nullptr; }
};

// Past a quoted run opened at s[i]; len when unterminated.
inline size_t skip_quoted(const char* s, size_t len, size_t i, char q, bool backslash) noexcept {
    for (++i; i < len; ++i) {
        if (backslash && s[i] == '\\') { ++i; continue; }
        if (s[i] != q) continue;
        if (!backslash && i + 1 < len && s[i + 1] == q) { ++i; continue; }
        return i + 1;
    }
    return len + 1;
}

// False when the text does not lex cleanly: the language's parser reports that.
inline bool scan_tail(const char* s, size_t len, QueryDialect d, TailRing* ring) noexcept {
    const bool hash_comment = d == QueryDialect::Sparql || d == QueryDialect::Promql;
    const bool slash_comment = d == QueryDialect::Cypher;
    const bool dash_comment = d == QueryDialect::Sql;
    const bool backslash = d != QueryDialect::Sql;
    uint32_t depth = 0;
    uint32_t group_begin = 0;
    size_t i = 0;
    while (i < len) {
        const char c = s[i];
        if (space(c)) { ++i; continue; }
        if ((hash_comment && c == '#') || (slash_comment && c == '/' && i + 1 < len && s[i + 1] == '/') ||
            (dash_comment && c == '-' && i + 1 < len && s[i + 1] == '-')) {
            while (i < len && s[i] != '\n') ++i;
            continue;
        }
        if (d != QueryDialect::Promql && c == '/' && i + 1 < len && s[i + 1] == '*') {
            i += 2;
            while (i + 1 < len && !(s[i] == '*' && s[i + 1] == '/')) ++i;
            if (i + 1 >= len) return false;
            i += 2;
            continue;
        }
        const size_t begin = i;
        Tok kind = Tok::Punct;
        if (c == '\'' || c == '"' || c == '`') {
            i = skip_quoted(s, len, i, c, backslash && c != '`');
            if (i > len) return false;
            kind = Tok::String;
        } else if (d == QueryDialect::Sql && c == '$' && i + 1 < len && s[i + 1] == '$') {
            i += 2;
            while (i + 1 < len && !(s[i] == '$' && s[i + 1] == '$')) ++i;
            if (i + 1 >= len) return false;
            i += 2;
            kind = Tok::String;
        } else if (d == QueryDialect::Sparql && c == '<') {
            // An IRI runs to '>' with no space; otherwise this is less-than.
            size_t j = i + 1;
            while (j < len && s[j] != '>' && !space(s[j])) ++j;
            i = (j < len && s[j] == '>') ? j + 1 : i + 1;
        } else if (ident_start(c)) {
            while (i < len && ident_char(s[i])) ++i;
            kind = Tok::Word;
        } else if (digit(c)) {
            while (i < len && (ident_char(s[i]) || s[i] == '.')) ++i;
            kind = Tok::Number;
        } else {
            ++i;
        }
        if (c == '(' && kind == Tok::Punct) {
            if (depth++ == 0) group_begin = static_cast<uint32_t>(begin);
            continue;
        }
        if (c == ')' && kind == Tok::Punct) {
            if (depth == 0) return false;
            if (--depth == 0) ring->push({group_begin, static_cast<uint32_t>(i), Tok::Group});
            continue;
        }
        if (depth == 0) ring->push({static_cast<uint32_t>(begin), static_cast<uint32_t>(i), kind});
    }
    return depth == 0;
}

// A word that wants an expression after it: `SELECT options (a = 1)` is a call.
inline bool expects_operand(const char* s, const Token& t) noexcept {
    if (t.kind == Tok::Punct) {
        const char c = s[t.begin];
        return c != '}' && c != ']' && c != '*' && c != '?';
    }
    if (t.kind != Tok::Word) return false;
    const char* w = s + t.begin;
    const size_t n = t.end - t.begin;
    static constexpr const char* kWords[] = {
        "select", "from", "join", "where", "and", "or", "not", "on", "by", "return", "with",
        "distinct", "having", "union", "as", "in", "set", "values", "match", "call", "unwind",
    };
    for (const char* k : kWords) if (word_is(w, n, k)) return true;
    return false;
}

inline void set_why(char* why, size_t cap, const char* what, const char* s, size_t n) noexcept {
    if (why == nullptr || cap == 0) return;
    std::snprintf(why, cap, "%s '%.*s'", what, static_cast<int>(n < 64 ? n : 64), s);
}

inline const char* unquote(const char* s, size_t* n) noexcept {
    if (*n >= 2 && (s[0] == '\'' || s[0] == '"') && s[*n - 1] == s[0]) { *n -= 2; return s + 1; }
    return s;
}

// `( k = v , ... )` between begin and end. None: not a clause, leave it alone.
inline QueryClauseResult parse_group(const char* s, const Token& g, QueryOptions* out, char* why, size_t cap) noexcept {
    size_t i = g.begin + 1;
    const size_t end = g.end - 1;
    QueryOptions acc{};
    bool any_known = false;
    bool first_bad_set = false;
    QueryClauseResult bad = QueryClauseResult::None;
    while (true) {
        while (i < end && space(s[i])) ++i;
        const size_t kb = i;
        while (i < end && ident_char(s[i])) ++i;
        const size_t kn = i - kb;
        if (kn == 0) return QueryClauseResult::None;
        while (i < end && space(s[i])) ++i;
        if (i >= end || s[i] != '=') return QueryClauseResult::None;
        ++i;
        while (i < end && space(s[i])) ++i;
        const size_t vb = i;
        if (i < end && (s[i] == '\'' || s[i] == '"')) {
            i = skip_quoted(s, end, i, s[i], false);
            if (i > end) return QueryClauseResult::None;
        } else {
            while (i < end && (ident_char(s[i]) || s[i] == '.')) ++i;
        }
        size_t vn = i - vb;
        if (vn == 0) return QueryClauseResult::None;
        const char* v = unquote(s + vb, &vn);
        const QueryOptionResult r = apply_query_option(s + kb, kn, v, vn, &acc);
        if (query_option_key_known(s + kb, kn)) any_known = true;
        if (r != QueryOptionResult::Ok && !first_bad_set) {
            first_bad_set = true;
            bad = QueryClauseResult::Invalid;
            if (r == QueryOptionResult::UnknownKey) set_why(why, cap, "unknown query option", s + kb, kn);
            else set_why(why, cap, "bad value for query option", s + kb, kn);
        }
        while (i < end && space(s[i])) ++i;
        if (i >= end) break;
        if (s[i] != ',') return QueryClauseResult::None;
        ++i;
    }
    if (!any_known) return QueryClauseResult::None;
    if (bad == QueryClauseResult::Invalid) return bad;
    query_options_overlay(out, acc);
    return QueryClauseResult::Found;
}

}  // namespace query_options_detail

/// Read the trailing clause of `text`. A trailing `;` stays out of the body.
/// The first clause read (the last in the text) wins a key both set.
inline QueryClauseResult split_query_options(const char* text, size_t len, QueryDialect dialect,
                                             QueryOptions* out, size_t* body_len,
                                             char* why, size_t why_cap) noexcept {
    using namespace query_options_detail;
    *body_len = len;
    TailRing ring;
    if (!scan_tail(text, len, dialect, &ring)) return QueryClauseResult::None;
    uint32_t at = 0;
    const Token* last = ring.back(0);
    if (last != nullptr && last->kind == Tok::Punct && text[last->begin] == ';') at = 1;
    QueryClauseResult result = QueryClauseResult::None;
    QueryOptions acc{};
    for (uint32_t round = 0; round < 2; ++round) {
        const Token* t0 = ring.back(at);
        const Token* t1 = ring.back(at + 1);
        const Token* t2 = ring.back(at + 2);
        const Token* t3 = ring.back(at + 3);
        if (t0 == nullptr || t1 == nullptr) break;
        if (t0->kind == Tok::Group && t1->kind == Tok::Word && t2 != nullptr &&
            word_is(text + t1->begin, t1->end - t1->begin, "options") && !expects_operand(text, *t2)) {
            QueryOptions one{};
            const QueryClauseResult r = parse_group(text, *t0, &one, why, why_cap);
            if (r == QueryClauseResult::Invalid) return r;
            if (r == QueryClauseResult::None) break;
            QueryOptions merged = one;
            query_options_overlay(&merged, acc);
            acc = merged;
            *body_len = t1->begin;
            at += 2;
            result = QueryClauseResult::Found;
            continue;
        }
        if (t2 != nullptr && t3 != nullptr && t1->kind == Tok::Word && t2->kind == Tok::Word &&
            word_is(text + t1->begin, t1->end - t1->begin, "limit") &&
            word_is(text + t2->begin, t2->end - t2->begin, "time") &&
            (t0->kind == Tok::Number || t0->kind == Tok::String) && !expects_operand(text, *t3)) {
            size_t vn = t0->end - t0->begin;
            const char* v = unquote(text + t0->begin, &vn);
            uint64_t ms = 0;
            if (!parse_query_duration_ms(v, vn, &ms)) {
                // `ORDER BY time LIMIT 5` is SQL. Only a string here was meant as a clause.
                if (t0->kind == Tok::Number) break;
                set_why(why, why_cap, "bad TIME LIMIT", v, vn);
                return QueryClauseResult::Invalid;
            }
            if ((acc.set & QueryOptions::kTimeLimit) == 0u) {
                acc.time_limit_ms = ms;
                acc.set |= QueryOptions::kTimeLimit;
            }
            *body_len = t2->begin;
            at += 3;
            result = QueryClauseResult::Found;
            continue;
        }
        break;
    }
    if (result == QueryClauseResult::Found) {
        query_options_overlay(out, acc);
        while (*body_len > 0 && space(text[*body_len - 1])) --*body_len;
    }
    return result;
}

enum class QuerySetResult : uint8_t { NotSet, Applied, Reset, Invalid };

/// `SET time_limit = '5s'`, `SET statement_timeout TO 5000`, `RESET time_limit`.
/// NotSet for anything else, including every Cypher SET clause. Reset names the
/// key's bit in *reset_bits (all bits for RESET ALL).
inline QuerySetResult parse_query_set(const char* s, size_t n, QueryOptions* o, uint32_t* reset_bits,
                                      char* why, size_t why_cap) noexcept {
    using namespace query_options_detail;
    size_t i = 0;
    while (i < n && space(s[i])) ++i;
    const size_t wb = i;
    while (i < n && ident_char(s[i])) ++i;
    const bool is_set = word_is(s + wb, i - wb, "set");
    const bool is_reset = word_is(s + wb, i - wb, "reset");
    if (!is_set && !is_reset) return QuerySetResult::NotSet;
    while (i < n && space(s[i])) ++i;
    const size_t kb = i;
    while (i < n && ident_char(s[i])) ++i;
    const size_t kn = i - kb;
    size_t end = n;
    while (end > i && (space(s[end - 1]) || s[end - 1] == ';')) --end;
    if (is_reset && word_is(s + kb, kn, "all") && i == end) { *reset_bits = UINT32_MAX; return QuerySetResult::Reset; }
    if (!query_option_key_known(s + kb, kn)) return QuerySetResult::NotSet;
    if (is_reset && !query_option_builtin(s + kb, kn)) {
        if (i != end) return QuerySetResult::NotSet;
        (void)query_knob_put(&o->knobs, s + kb, kn, kQueryKnobReset, QueryKnobSource::Session);
        o->set |= QueryOptions::kKnobs;
        return QuerySetResult::Applied;
    }
    if (is_reset) {
        if (i != end) return QuerySetResult::NotSet;
        *reset_bits = query_option_bit(s + kb, kn);
        return QuerySetResult::Reset;
    }
    while (i < end && space(s[i])) ++i;
    if (i < end && s[i] == '=') ++i;
    else if (i + 1 < end && word_is(s + i, 2, "to")) i += 2;
    else return QuerySetResult::NotSet;
    while (i < end && space(s[i])) ++i;
    size_t vn = end - i;
    const char* v = unquote(s + i, &vn);
    if (word_is(v, vn, "default") && query_option_builtin(s + kb, kn)) {
        *reset_bits = query_option_bit(s + kb, kn);
        return QuerySetResult::Reset;
    }
    if (apply_query_option(s + kb, kn, v, vn, o) != QueryOptionResult::Ok) {
        set_why(why, why_cap, "bad value for query option", s + kb, kn);
        return QuerySetResult::Invalid;
    }
    return QuerySetResult::Applied;
}

}  // namespace bolt
