// bolt_yaml_scalars.inc — scalars onto the tape (included by bolt_yaml.cpp).
//
// Plain scalars are typed by the YAML 1.2 core schema. A token's bytes stay a
// source slice whenever they already are what a JSON token holds (a JSON
// number, `null` / `true` / `false`, JSON string content); anything else is
// materialised once into aux. The keywords live at fixed aux offsets.

constexpr uint32_t kAuxNull = 0, kAuxTrue = 4, kAuxFalse = 8, kAuxKeywords = 13;

bool word_in(const char* b, uint32_t n, const char* const* words, uint32_t count) noexcept {
    for (uint32_t w = 0; w < count; ++w) {                        // bounded by count
        const uint32_t l = static_cast<uint32_t>(std::strlen(words[w]));
        if (l == n && std::memcmp(b, words[w], n) == 0) return true;
    }
    return false;
}

bool all_digits(const char* b, uint32_t from, uint32_t n) noexcept {
    if (from >= n) return false;
    for (uint32_t k = from; k < n; ++k) if (b[k] < '0' || b[k] > '9') return false;
    return true;
}

bool core_int(const char* b, uint32_t n) noexcept {                // [-+]?[0-9]+
    return all_digits(b, (n > 0 && (b[0] == '-' || b[0] == '+')) ? 1u : 0u, n);
}

bool json_int(const char* b, uint32_t n) noexcept {                // -?(0|[1-9][0-9]*)
    const uint32_t k = (n > 0 && b[0] == '-') ? 1u : 0u;
    return all_digits(b, k, n) && (b[k] != '0' || n == k + 1);
}

// [-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?; *json when also a JSON number.
bool core_float(const char* b, uint32_t n, bool* json) noexcept {
    uint32_t k = (n > 0 && (b[0] == '-' || b[0] == '+')) ? 1u : 0u;
    const uint32_t int_start = k;
    uint32_t idig = 0, fdig = 0;
    while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++idig; }
    bool dot = false;
    if (k < n && b[k] == '.') { dot = true; ++k; while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++fdig; } }
    if (idig + fdig == 0) return false;
    bool exp = false;
    if (k < n && (b[k] == 'e' || b[k] == 'E')) {
        exp = true;
        ++k;
        if (k < n && (b[k] == '-' || b[k] == '+')) ++k;
        uint32_t ed = 0;
        while (k < n && b[k] >= '0' && b[k] <= '9') { ++k; ++ed; }
        if (ed == 0) return false;
    }
    *json = b[0] != '+' && idig > 0 && (!dot || fdig > 0) && (dot || exp) &&
            (b[int_start] != '0' || idig == 1);
    return k == n;
}

bool emit_number(P* p, const Text& t, bj::TokenType type, int base) noexcept {
    const char* b = text_ptr(p, t);
    const uint32_t n = text_len(t);
    char tmp[400];
    if (n >= sizeof(tmp)) return fail(p, YamlCode::Unsupported, type == bj::TokenType::Int64 ? "integer beyond int64" : "float literal too long");
    std::memcpy(tmp, b, n);
    tmp[n] = '\0';
    bool json = false;
    if (type == bj::TokenType::Int64) {
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(base == 10 ? tmp : tmp + 2, &end, base);
        if (errno == ERANGE || end != tmp + n) return fail(p, YamlCode::Unsupported, "integer beyond int64");
        json = base == 10 && json_int(b, n);
        if (!json) (void)std::snprintf(tmp, sizeof(tmp), "%lld", v);
    } else {
        const double v = std::strtod(tmp, nullptr);
        if (!std::isfinite(v)) return fail(p, YamlCode::Unsupported, "non-finite float");
        (void)core_float(b, n, &json);
        if (!json) {
            int w = std::snprintf(tmp, sizeof(tmp), "%.17g", v);
            bool frac = false;
            for (int k = 0; k < w; ++k) frac = frac || tmp[k] == '.' || tmp[k] == 'e';
            if (!frac) { tmp[w++] = '.'; tmp[w++] = '0'; tmp[w] = '\0'; }
        }
    }
    if (!p->emit) return true;
    if (json && t.slice) return tok(p, type, t.b, n, 0);
    uint32_t off = 0;
    const uint32_t len = json ? n : static_cast<uint32_t>(std::strlen(tmp));
    return aux_raw(p, json ? b : tmp, len, &off) && tok(p, type, off, len, bj::kTokenAux);
}

bool emit_keyword(P* p, const Text& t, bj::TokenType type, const char* word, uint32_t aux_off) noexcept {
    const uint32_t n = text_len(t);
    if (t.slice && n == std::strlen(word) && std::memcmp(p->s + t.b, word, n) == 0) return tok(p, type, t.b, n, 0);
    return tok(p, type, aux_off, static_cast<uint32_t>(std::strlen(word)), bj::kTokenAux);
}

// A plain scalar under the YAML 1.2 core schema.
bool emit_plain(P* p, const Text& t) noexcept {
    static const char* const kNull[]  = {"~", "null", "Null", "NULL"};
    static const char* const kTrue[]  = {"true", "True", "TRUE"};
    static const char* const kFalse[] = {"false", "False", "FALSE"};
    static const char* const kInf[]   = {".inf", ".Inf", ".INF", "+.inf", "+.Inf", "+.INF",
                                         "-.inf", "-.Inf", "-.INF", ".nan", ".NaN", ".NAN"};
    if (!count_node(p)) return false;
    const char* b = text_ptr(p, t);
    const uint32_t n = text_len(t);
    // Only these first bytes can start a typed scalar: everything else is a string.
    const char c0 = n == 0 ? '~' : b[0];
    const bool maybe = (c0 >= '0' && c0 <= '9') || c0 == '-' || c0 == '+' || c0 == '.' || c0 == '~' ||
                       c0 == 'n' || c0 == 'N' || c0 == 't' || c0 == 'T' || c0 == 'f' || c0 == 'F';
    if (!maybe) return emit_text(p, bj::TokenType::String, t);
    if (n == 0 || word_in(b, n, kNull, 4)) return emit_keyword(p, t, bj::TokenType::Null, "null", kAuxNull);
    if (word_in(b, n, kTrue, 3)) return emit_keyword(p, t, bj::TokenType::BoolTrue, "true", kAuxTrue);
    if (word_in(b, n, kFalse, 3)) return emit_keyword(p, t, bj::TokenType::BoolFalse, "false", kAuxFalse);
    if (core_int(b, n)) return emit_number(p, t, bj::TokenType::Int64, 10);
    if (n > 2 && b[0] == '0' && b[1] == 'o') {
        bool ok = true;
        for (uint32_t k = 2; k < n; ++k) ok = ok && b[k] >= '0' && b[k] <= '7';
        if (ok) return emit_number(p, t, bj::TokenType::Int64, 8);
    }
    if (n > 2 && b[0] == '0' && b[1] == 'x') {
        bool ok = true;
        for (uint32_t k = 2; k < n; ++k) {
            const char c = b[k];
            ok = ok && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
        }
        if (ok) return emit_number(p, t, bj::TokenType::Int64, 16);
    }
    if (word_in(b, n, kInf, 12)) return fail(p, YamlCode::Unsupported, "non-finite float");
    bool json = false;
    if (core_float(b, n, &json)) return emit_number(p, t, bj::TokenType::Float64, 10);
    return emit_text(p, bj::TokenType::String, t);
}

bool flow_indicator(char c) noexcept { return c == ',' || c == '[' || c == ']' || c == '{' || c == '}'; }

// 0x80 in each byte below `lim` (lim <= 0x80), exact.
inline uint64_t lt_mask(uint64_t w, uint8_t lim) noexcept {
    return ~(((w & kLow7) + kOnes * static_cast<uint8_t>(0x80u - lim)) | w) & kHigh;
}

// Where a plain scalar ends on its line: before " #", ": " (or ':' at end),
// and in flow context before an indicator; trailing spaces trimmed. Word at a
// time: a word with no candidate byte is skipped whole. *json reports whether
// the bytes are JSON string content as they stand (no '"', '\\', control byte).
uint32_t plain_scan(const P* p, uint32_t k, bool flow, bool* json) noexcept {
    const uint32_t le = line_end(p, k);
    uint64_t unsafe = 0;
    uint32_t e = k;
    while (e < le) {                                              // bounded by line end
        const uint64_t w = load8(p->s + e, le - e, 'x');
        uint64_t cand = eq_mask(w, ':') | eq_mask(w, '#');
        if (flow) cand |= eq_mask(w, ',') | eq_mask(w, '[') | eq_mask(w, ']') | eq_mask(w, '{') | eq_mask(w, '}');
        const uint64_t bad = eq_mask(w, '"') | eq_mask(w, '\\') | lt_mask(w, 0x20);
        if (cand == 0) { unsafe |= bad; e += 8; continue; }
        const uint32_t b = static_cast<uint32_t>(std::countr_zero(cand)) >> 3;
        unsafe |= bad & ((b == 0 ? 0ull : (~0ull >> (64u - 8u * b))));
        const uint32_t pos = e + b;
        const char c = p->s[pos];
        const bool stop = (c == '#' && pos > k && is_ws(p->s[pos - 1])) ||
                          (c == ':' && (pos + 1 >= le || is_ws(p->s[pos + 1]) || (flow && flow_indicator(p->s[pos + 1])))) ||
                          (flow && flow_indicator(c));
        if (stop) { e = pos; break; }
        e = pos + 1u;
    }
    if (e > le) e = le;
    while (e > k && is_ws(p->s[e - 1])) --e;
    *json = unsafe == 0;
    return e;
}

uint32_t plain_end(const P* p, uint32_t k, bool flow) noexcept {
    bool json = false;
    return plain_scan(p, k, flow, &json);
}

bool refused_start(P* p, char c) noexcept {
    if (c == '&' || c == '*') return fail(p, YamlCode::Unsupported, "anchors and aliases");
    if (c == '!') return fail(p, YamlCode::Unsupported, "tags");
    if (c == '%') return fail(p, YamlCode::Unsupported, "directives");
    if (c == '?' && blank_or_brk(at(p, p->i + 1))) return fail(p, YamlCode::Unsupported, "complex keys");
    if (c == '@' || c == '`') return fail(p, YamlCode::Syntax, "reserved indicator");
    return true;
}

// Multi-line plain scalar in block context (continuations indented more than
// `parent`). Single-line: a source slice; folded: materialised.
bool block_plain(P* p, int parent, Text* t) noexcept {
    bool json = false;
    const uint32_t e = plain_scan(p, p->i, false, &json);
    *t = Text{p->i, e, true, json, {}};
    p->i = e;
    for (;;) {                                                    // bounded: advances lines
        const uint32_t save = p->i;
        if (!rest_empty(p) || at_end(p)) { p->i = save; return true; }
        uint32_t l = ln_of(p, p->i) + 1u, blanks = 0;
        while (l < p->nlines && p->lines[l].kind == kLineBlank) { ++blanks; ++l; }   // bounded by lines
        if (l >= p->nlines) { p->i = save; return true; }
        const Line& L = p->lines[l];
        if (L.kind != kLineContent || static_cast<int>(L.content - L.start) <= parent ||
            doc_boundary(p, L.start)) {
            p->i = save;
            return true;
        }
        const uint32_t pe = plain_end(p, L.content, false);
        if (pe < L.end && p->s[pe] == ':') { p->i = L.content; return fail(p, YamlCode::Syntax, "mapping value not allowed here"); }
        if (t->slice) {
            t->slice = false;
            if (!buf_put(p, &t->buf, p->s + t->b, t->e - t->b)) return false;
        }
        if (blanks == 0 && !buf_c(p, &t->buf, ' ')) return false;
        for (uint32_t k = 0; k < blanks; ++k) if (!buf_c(p, &t->buf, '\n')) return false;
        if (!buf_put(p, &t->buf, p->s + L.content, pe - L.content)) return false;
        p->i = pe;
    }
}

// ---- quoted scalars ------------------------------------------------------------

// Fold a line break inside a quoted scalar at p->i ('\n').
bool quoted_break(P* p, Buf* out, uint32_t keep) noexcept {
    while (out->n > keep && is_ws(out->b[out->n - 1])) --out->n;   // escaped white space stays
    uint32_t blanks = 0;
    ++p->i;
    for (;;) {                                                    // bounded by n
        skip_ws(p);
        if (cur(p) == '\r') ++p->i;
        if (cur(p) != '\n') break;
        ++blanks;
        ++p->i;
    }
    if (blanks == 0) return buf_c(p, out, ' ');
    for (uint32_t b = 0; b < blanks; ++b) if (!buf_c(p, out, '\n')) return false;
    return true;
}

bool utf8_put(P* p, Buf* out, uint32_t cp) noexcept {
    char u[4];
    uint32_t n = 0;
    if (cp < 0x80) { u[n++] = static_cast<char>(cp); }
    else if (cp < 0x800) { u[n++] = static_cast<char>(0xC0 | (cp >> 6)); u[n++] = static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        u[n++] = static_cast<char>(0xE0 | (cp >> 12)); u[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        u[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        u[n++] = static_cast<char>(0xF0 | (cp >> 18)); u[n++] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        u[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); u[n++] = static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        return fail(p, YamlCode::Syntax, "invalid escape");
    }
    return buf_put(p, out, u, n);
}

bool hex_escape(P* p, Buf* out, uint32_t digits) noexcept {
    uint32_t cp = 0;
    for (uint32_t k = 0; k < digits; ++k) {                       // bounded: <= 8
        const char c = at(p, p->i + 1 + k);
        const int v = (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                    : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return fail(p, YamlCode::Syntax, "invalid escape");
        cp = cp * 16u + static_cast<uint32_t>(v);
    }
    p->i += digits + 1u;
    return utf8_put(p, out, cp);
}

bool dq_escape(P* p, Buf* out) noexcept {
    ++p->i;                                                       // past '\\'
    const char c = cur(p);
    static constexpr char kFrom[] = {'0', 'a', 'b', 't', 'n', 'v', 'f', 'r', 'e', ' ', '"', '/', '\\', '\t'};
    static constexpr char kTo[]   = {'\0', '\a', '\b', '\t', '\n', '\v', '\f', '\r', '\x1b', ' ', '"', '/', '\\', '\t'};
    if (c == '\n') { ++p->i; skip_ws(p); return true; }           // escaped break joins
    if (c == 'x') return hex_escape(p, out, 2);
    if (c == 'u') return hex_escape(p, out, 4);
    if (c == 'U') return hex_escape(p, out, 8);
    if (c == 'N') { ++p->i; return utf8_put(p, out, 0x85); }
    if (c == '_') { ++p->i; return utf8_put(p, out, 0xA0); }
    if (c == 'L') { ++p->i; return utf8_put(p, out, 0x2028); }
    if (c == 'P') { ++p->i; return utf8_put(p, out, 0x2029); }
    for (uint32_t k = 0; k < sizeof(kFrom); ++k) {                // bounded: table
        if (kFrom[k] == c) { ++p->i; return buf_c(p, out, kTo[k]); }
    }
    return fail(p, YamlCode::Syntax, "invalid escape");
}

// Can the quoted body at p->i (after the opening quote) stay a source slice?
// Single line, no `''`, and JSON content as written: for double quotes every
// escape must also be a JSON escape (\" \\ \/ \b \f \n \r \t \uXXXX).
bool quoted_slice(const P* p, char q, uint32_t* close) noexcept {
    const uint32_t le = line_end(p, p->i);
    for (uint32_t k = p->i; k < le; ++k) {                        // bounded by line end
        const auto c = static_cast<unsigned char>(p->s[k]);
        if (c == static_cast<unsigned char>(q)) {
            if (q == '\'' && at(p, k + 1) == '\'') return false;
            *close = k;
            return true;
        }
        if (c < 0x20) return false;
        if (q == '\'' && (c == '"' || c == '\\')) return false;
        if (q == '"' && c == '\\') {
            const char e = at(p, k + 1);
            if (e == 'u') {
                for (uint32_t h = 2; h < 6; ++h) if (!std::isxdigit(static_cast<unsigned char>(at(p, k + h)))) return false;
                k += 5;
                continue;
            }
            if (e != '"' && e != '\\' && e != '/' && e != 'b' && e != 'f' && e != 'n' && e != 'r' && e != 't') return false;
            ++k;
        }
    }
    return false;
}

bool quoted(P* p, Text* t) noexcept {
    const char q = cur(p);
    ++p->i;
    uint32_t close = 0;
    if (quoted_slice(p, q, &close)) {
        *t = Text{p->i, close, true, true, {}};
        p->i = close + 1u;
        return true;
    }
    *t = Text{0, 0, false, false, {}};
    uint32_t keep = 0;   // bytes produced by escapes are never folded away
    for (;;) {                                                    // bounded by n
        if (at_end(p)) return fail(p, YamlCode::Syntax, "unterminated quoted scalar");
        const char c = cur(p);
        if (c == q) {
            if (q == '\'' && at(p, p->i + 1) == '\'') { if (!buf_c(p, &t->buf, '\'')) return false; p->i += 2; continue; }
            ++p->i;
            return true;
        }
        if (c == '\\' && q == '"') { if (!dq_escape(p, &t->buf)) return false; keep = t->buf.n; continue; }
        if (c == '\r' && at(p, p->i + 1) == '\n') { ++p->i; continue; }
        if (c == '\n') { if (!quoted_break(p, &t->buf, keep)) return false; continue; }
        if (!buf_c(p, &t->buf, c)) return false;
        ++p->i;
    }
}

// ---- block scalars ------------------------------------------------------------

struct BlockHeader { char style; char chomp; uint32_t explicit_indent; };

bool block_header(P* p, BlockHeader* h) noexcept {
    h->style = cur(p);
    h->chomp = 'c';
    h->explicit_indent = 0;
    ++p->i;
    for (int k = 0; k < 2; ++k) {                                 // bounded: two indicators
        const char c = cur(p);
        if ((c == '+' || c == '-') && h->chomp == 'c') { h->chomp = c; ++p->i; continue; }
        if (c >= '1' && c <= '9' && h->explicit_indent == 0) { h->explicit_indent = static_cast<uint32_t>(c - '0'); ++p->i; continue; }
    }
    if (!rest_empty(p)) return fail(p, YamlCode::Syntax, "block scalar header");
    return true;
}

// Content indent: explicit, or the first non-blank line's (from the table).
uint32_t block_indent(const P* p, uint32_t first, int parent, const BlockHeader& h) noexcept {
    if (h.explicit_indent != 0) return static_cast<uint32_t>(parent < 0 ? 0 : parent) + h.explicit_indent;
    uint32_t ind = 0, longest = 0;
    bool content = false;
    for (uint32_t l = first; l < p->nlines; ++l) {                 // bounded by lines
        const Line& L = p->lines[l];
        if (L.kind == kLineBlank) {
            const uint32_t sp = L.content - L.start;
            longest = sp > longest ? sp : longest;
            continue;
        }
        ind = L.content - L.start;
        content = true;
        break;
    }
    if (!content) ind = longest;   // no content line: the longest empty line (YAML 1.2 8.1.1.1)
    return static_cast<int>(ind) <= parent ? static_cast<uint32_t>(parent + 1) : ind;
}

bool block_line(P* p, const BlockHeader& h, const Line& L, uint32_t ind, uint32_t* trailing, bool* any,
                bool* prev_more, Buf* out) noexcept {
    const uint32_t sp = L.content - L.start;
    const bool more = sp > ind || p->s[L.start + ind] == '\t';
    const bool keep_breaks = h.style == '|' || more || *prev_more;
    if (*any && keep_breaks) {
        if (!buf_c(p, out, '\n')) return false;
        for (uint32_t b = 0; b < *trailing; ++b) if (!buf_c(p, out, '\n')) return false;
    } else if (*any && *trailing == 0) {
        if (!buf_c(p, out, ' ')) return false;
    } else {
        for (uint32_t b = 0; b < *trailing; ++b) if (!buf_c(p, out, '\n')) return false;
    }
    *trailing = 0;
    if (!buf_put(p, out, p->s + L.start + ind, L.end - (L.start + ind))) return false;
    *any = true;
    *prev_more = more;
    return true;
}

// Literal `|` / folded `>` content: lines with indent >= the content indent, or blank.
bool block_scalar(P* p, int parent, Text* t) noexcept {
    BlockHeader h{};
    if (!block_header(p, &h)) return false;
    *t = Text{0, 0, false, false, {}};
    const uint32_t first = ln_of(p, p->i) + 1u;
    const uint32_t ind = block_indent(p, first, parent, h);
    uint32_t trailing = 0, l = first;
    bool any = false, prev_more = false;
    for (; l < p->nlines; ++l) {                                  // bounded by lines
        const Line& L = p->lines[l];
        if (L.start >= p->n) break;                               // the empty tail after a final '\n'
        // A whitespace-only line is empty unless it reaches past the content
        // indent: then its extra spaces / tabs are content (YAML 1.2 8.1.1.2).
        if (L.kind == kLineBlank && L.end - L.start <= ind) { ++trailing; continue; }
        if (L.kind != kLineBlank && L.content - L.start < ind && L.kind != kLineTab) break;
        if (!block_line(p, h, L, ind, &trailing, &any, &prev_more, &t->buf)) return false;
    }
    if (h.chomp != '-' && any && !buf_c(p, &t->buf, '\n')) return false;
    if (h.chomp == '+') for (uint32_t b = 0; b < trailing; ++b) if (!buf_c(p, &t->buf, '\n')) return false;
    // End of the last line the scalar owns (blank lines after it included).
    p->i = l == 0 ? 0 : p->lines[l - 1].end;
    return true;
}
