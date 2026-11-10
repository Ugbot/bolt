// bolt_yaml_blocks.inc — collections and the document (included by bolt_yaml.cpp).
//
// Block collections are decided by the line table: a sibling is the next
// content line at the collection's column; a child is a deeper one. Every
// container emits Begin/End tokens; a key's subtree is emitted only when the
// path filter (if any) admits it, as json::build_index_filtered does.

// ---- path filter -------------------------------------------------------------------

// Push one path segment (JSON-inner bytes); returns the emit state to restore.
bool push_seg(P* p, const char* b, uint32_t n) noexcept {
    const bool saved = p->emit;
    if (p->filter == nullptr || !p->emit) return saved;
    if (n > static_cast<uint32_t>(bj::kJsonMaxPathBytes)) n = static_cast<uint32_t>(bj::kJsonMaxPathBytes);
    if (p->nseg < bj::kJsonMaxDepth &&
        p->path_len + 1 + static_cast<int32_t>(n) <= static_cast<int32_t>(sizeof(p->path))) {
        p->seg_start[p->nseg++] = p->path_len;
        p->path[p->path_len++] = '/';
        std::memcpy(p->path + p->path_len, b, n);
        p->path_len += static_cast<int32_t>(n);
    }
    p->emit = bj::path_filter_admits(p->filter, p->path, p->path_len);
    return saved;
}

void pop_seg(P* p, bool saved) noexcept {
    if (p->filter != nullptr && saved && p->nseg > 0) p->path_len = p->seg_start[--p->nseg];
    p->emit = saved;
}

bool push_key(P* p, const Text& key, bool* saved) noexcept {
    if (p->filter == nullptr || !p->emit) { *saved = p->emit; return true; }
    if (key.slice && (key.json || json_safe(p->s + key.b, key.e - key.b))) {
        *saved = push_seg(p, p->s + key.b, key.e - key.b);
        return true;
    }
    const uint32_t mark = p->naux;                                // escaped key, scratch only
    uint32_t off = 0, len = 0;
    if (!aux_escaped(p, text_ptr(p, key), text_len(key), &off, &len)) return false;
    *saved = push_seg(p, p->aux + off, len);
    p->naux = mark;
    return true;
}

bool push_index(P* p, uint32_t idx, bool* saved) noexcept {
    if (p->filter == nullptr || !p->emit) { *saved = p->emit; return true; }
    char b[12];
    const int n = std::snprintf(b, sizeof(b), "%u", idx);
    *saved = push_seg(p, b, static_cast<uint32_t>(n));
    return true;
}

bool enter(P* p, bj::TokenType begin) noexcept {
    if (++p->depth > p->lim.depth || (p->filter != nullptr && p->depth > static_cast<uint32_t>(bj::kJsonMaxDepth))) {
        return fail(p, YamlCode::Limit, "depth");
    }
    return count_node(p) && tok(p, begin, p->i, 0, 0);
}

bool leave(P* p, bj::TokenType end) noexcept {
    --p->depth;
    return tok(p, end, p->i, 0, 0);
}

// ---- flow collections -------------------------------------------------------------

bool flow_node(P* p) noexcept;

void flow_ws(P* p) noexcept {
    for (;;) {                                                    // bounded by n
        while (!at_end(p) && (is_ws(cur(p)) || cur(p) == '\n' || cur(p) == '\r')) ++p->i;
        if (cur(p) == '#' && (p->i == 0 || blank_or_brk(at(p, p->i - 1)))) { p->i = line_end(p, p->i); continue; }
        return;
    }
}

bool flow_text(P* p, Text* t, bool as_key) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '"' || c == '\'') return quoted(p, t);
    bool json = false;
    const uint32_t e = plain_scan(p, p->i, true, &json);
    if (e == p->i && !as_key) return fail(p, YamlCode::Syntax, "empty flow entry");
    *t = Text{p->i, e, true, json, {}};
    p->i = e;
    uint32_t k = p->i;
    while (k < p->n && is_ws(p->s[k])) ++k;
    if (k < p->n && is_brk(p->s[k])) {
        p->i = k;
        flow_ws(p);
        if (!at_end(p) && !flow_indicator(cur(p)) && cur(p) != ':') {
            return fail(p, YamlCode::Unsupported, "multi-line plain scalar in a flow collection");
        }
    }
    return true;
}

bool flow_scalar(P* p) noexcept {
    const char c = cur(p);
    Text t{};
    if (!flow_text(p, &t, false)) return false;
    if (c == '"' || c == '\'') return count_node(p) && emit_text(p, bj::TokenType::String, t);
    return emit_plain(p, t);
}

bool flow_sep(P* p, char close, bool first, bool* done) noexcept {
    flow_ws(p);
    *done = false;
    if (at_end(p)) return fail(p, YamlCode::Syntax, close == ']' ? "unterminated flow sequence" : "unterminated flow mapping");
    if (cur(p) == close) { ++p->i; *done = true; return true; }
    if (first) return true;
    if (cur(p) != ',') return fail(p, YamlCode::Syntax, close == ']' ? "expected ',' or ']'" : "expected ',' or '}'");
    ++p->i;
    flow_ws(p);
    if (at_end(p)) return fail(p, YamlCode::Syntax, close == ']' ? "unterminated flow sequence" : "unterminated flow mapping");
    if (cur(p) == close) { ++p->i; *done = true; }
    return true;
}

bool flow_seq(P* p) noexcept {
    if (!enter(p, bj::TokenType::BeginArray)) return false;
    ++p->i;
    for (uint32_t idx = 0;; ++idx) {                              // bounded by n
        bool done = false, saved = false;
        if (!flow_sep(p, ']', idx == 0, &done)) return false;
        if (done) break;
        if (!push_index(p, idx, &saved)) return false;
        const bool ok = flow_node(p);
        pop_seg(p, saved);
        if (!ok) return false;
        flow_ws(p);
        if (cur(p) == ':') return fail(p, YamlCode::Unsupported, "single-pair mapping in a flow sequence");
    }
    return leave(p, bj::TokenType::EndArray);
}

bool null_value(P* p) noexcept {
    return count_node(p) && tok(p, bj::TokenType::Null, kAuxNull, 4, bj::kTokenAux);
}

bool flow_map(P* p) noexcept {
    if (!enter(p, bj::TokenType::BeginObject)) return false;
    ++p->i;
    for (bool first = true;; first = false) {                     // bounded by n
        bool done = false, saved = false;
        if (!flow_sep(p, '}', first, &done)) return false;
        if (done) break;
        if (cur(p) == '[' || cur(p) == '{') return fail(p, YamlCode::Unsupported, "complex keys");
        Text key{};
        if (!flow_text(p, &key, true) || !push_key(p, key, &saved)) return false;
        bool ok = emit_text(p, bj::TokenType::Key, key);
        flow_ws(p);
        if (ok && cur(p) == ':') {
            ++p->i;
            flow_ws(p);
            ok = (cur(p) == ',' || cur(p) == '}') ? null_value(p) : flow_node(p);
        } else if (ok) {
            ok = null_value(p);
        }
        pop_seg(p, saved);
        if (!ok) return false;
    }
    return leave(p, bj::TokenType::EndObject);
}

bool flow_node(P* p) noexcept {
    if (cur(p) == '[') return flow_seq(p);
    if (cur(p) == '{') return flow_map(p);
    return flow_scalar(p);
}

// ---- block collections ------------------------------------------------------------

bool block_node(P* p, int parent) noexcept;
bool block_map(P* p, uint32_t col) noexcept;
bool block_seq(P* p, uint32_t col) noexcept;

// Is the content at p->i a mapping key on this line (`key:` + blank/EOL)?
bool looks_like_key(const P* p) noexcept {
    uint32_t k = p->i;
    const uint32_t le = line_end(p, k);
    const char c = at(p, k);
    if (c == '"' || c == '\'') {
        for (++k; k < le; ++k) {                                  // bounded by line end
            if (p->s[k] == '\\' && c == '"') { ++k; continue; }
            if (p->s[k] != c) continue;
            if (c == '\'' && at(p, k + 1) == '\'') { ++k; continue; }
            break;
        }
        if (k >= le) return false;
        ++k;
        while (k < le && is_ws(p->s[k])) ++k;
        return k < le && p->s[k] == ':' && (k + 1 >= le || is_ws(p->s[k + 1]));
    }
    if (c == '[' || c == '{' || c == '|' || c == '>' || c == '#') return false;
    if (c == '-' && blank_or_brk(at(p, k + 1))) return false;
    bool json = false;
    const uint32_t e = plain_scan(p, k, false, &json);
    P* m = const_cast<P*>(p);                                     // remember the scan for map_key
    m->key_pos = k; m->key_end = e; m->key_json = json;
    uint32_t c2 = e;
    while (c2 < le && is_ws(p->s[c2])) ++c2;                      // `key   :`
    return c2 < le && p->s[c2] == ':' && e > k;
}

// A value written after `key:` / `- ` on the same line.
bool inline_value(P* p, int parent) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '[' || c == '{') {
        if (!flow_node(p)) return false;
        if (!rest_empty(p)) return fail(p, YamlCode::Syntax, "content after a flow collection");
        return true;
    }
    Text t{};
    if (c == '|' || c == '>') return block_scalar(p, parent, &t) && count_node(p) && emit_text(p, bj::TokenType::String, t);
    if (c == '"' || c == '\'') {
        if (!quoted(p, &t)) return false;
        const uint32_t save = p->i;
        if (!rest_empty(p)) { p->i = save; return fail(p, YamlCode::Syntax, "content after a quoted scalar"); }
        return count_node(p) && emit_text(p, bj::TokenType::String, t);
    }
    if (!block_plain(p, parent, &t)) return false;
    const uint32_t save = p->i;
    if (!rest_empty(p)) {
        p->i = save;
        return fail(p, YamlCode::Syntax, cur(p) == ':' ? "mapping value not allowed here" : "unexpected content");
    }
    return emit_plain(p, t);
}

// The value of `key:` / `- ` whose rest of line is empty: a block node on the
// following lines (or an indentless sequence at `col` under a mapping key).
bool value_below(P* p, uint32_t col, bool indentless_ok) noexcept {
    uint32_t pos = 0, ind = 0;
    if (!next_content(p, p->i, &pos, &ind)) return false;
    const bool seq_here = indentless_ok && pos < p->n && ind == col && p->s[pos] == '-' &&
                          blank_or_brk(at(p, pos + 1));
    if (pos >= p->n || (ind <= col && !seq_here) || doc_boundary(p, pos)) return null_value(p);
    p->i = pos;
    if (seq_here) return block_seq(p, col);
    return block_node(p, static_cast<int>(col));
}

bool map_key(P* p, Text* key) noexcept {
    const char c = cur(p);
    if (!refused_start(p, c)) return false;
    if (c == '"' || c == '\'') {
        if (!quoted(p, key)) return false;
        skip_ws(p);
    } else {
        bool json = false;
        const uint32_t e = p->key_pos == p->i ? (json = p->key_json, p->key_end) : plain_scan(p, p->i, false, &json);
        *key = Text{p->i, e, true, json, {}};
        p->i = e;
        skip_ws(p);                                               // `key   : value`
    }
    if (cur(p) != ':') return fail(p, YamlCode::Syntax, "expected ':' after a mapping key");
    ++p->i;
    return true;
}

// Advance to the next sibling at `col`: *done when the collection ends.
bool next_sibling(P* p, uint32_t col, bool* done) noexcept {
    uint32_t pos = 0, ind = 0;
    *done = true;
    if (!rest_empty(p)) return fail(p, YamlCode::Syntax, "unexpected content");
    if (!next_content(p, p->i, &pos, &ind)) return false;
    if (pos >= p->n || ind < col || doc_boundary(p, pos)) {
        p->i = pos >= p->n ? p->n : p->lines[ln_of(p, pos)].start - 1u;
        return true;
    }
    if (ind > col) { p->i = pos; return fail(p, YamlCode::Syntax, "bad indentation"); }
    p->i = pos;
    *done = false;
    return true;
}

bool map_entry(P* p, uint32_t col) noexcept {
    if (!refused_start(p, cur(p))) return false;
    if (!looks_like_key(p)) return fail(p, YamlCode::Syntax, "expected a mapping key");
    Text key{};
    bool saved = false;
    if (!map_key(p, &key) || !push_key(p, key, &saved)) return false;
    bool ok = emit_text(p, bj::TokenType::Key, key);
    skip_ws(p);
    const uint32_t save = p->i;
    if (ok && rest_empty(p)) {
        ok = value_below(p, col, true);
    } else if (ok) {
        p->i = save;
        ok = inline_value(p, static_cast<int>(col));
    }
    pop_seg(p, saved);
    return ok;
}

bool block_map(P* p, uint32_t col) noexcept {
    if (!enter(p, bj::TokenType::BeginObject)) return false;
    for (;;) {                                                    // bounded: advances lines
        if (!map_entry(p, col)) return false;
        bool done = false;
        if (!next_sibling(p, col, &done)) return false;
        if (done) break;
        if (cur(p) == '-' && blank_or_brk(at(p, p->i + 1))) {
            return fail(p, YamlCode::Syntax, "sequence entry where a mapping key was expected");
        }
    }
    return leave(p, bj::TokenType::EndObject);
}

bool seq_item(P* p, uint32_t col, uint32_t idx) noexcept {
    bool saved = false;
    if (!push_index(p, idx, &saved)) return false;
    ++p->i;                                                       // past '-'
    skip_ws(p);
    const uint32_t save = p->i;
    bool ok = false;
    if (rest_empty(p)) {
        ok = value_below(p, col, false);
    } else {
        p->i = save;
        ok = block_node(p, static_cast<int>(col));
    }
    pop_seg(p, saved);
    return ok;
}

bool block_seq(P* p, uint32_t col) noexcept {
    if (!enter(p, bj::TokenType::BeginArray)) return false;
    for (uint32_t idx = 0;; ++idx) {                              // bounded: advances lines
        if (!seq_item(p, col, idx)) return false;
        bool done = false;
        if (!next_sibling(p, col, &done)) return false;
        if (done) break;
        if (!(cur(p) == '-' && blank_or_brk(at(p, p->i + 1)))) {  // a key: the parent mapping's
            p->i = p->lines[ln_of(p, p->i)].start - 1u;
            break;
        }
    }
    return leave(p, bj::TokenType::EndArray);
}

// A node starting at p->i inside a block collection indented `parent` (-1 at top).
bool block_node(P* p, int parent) noexcept {
    const char c = cur(p);
    const uint32_t col = column(p, p->i);
    if (c == '-' && blank_or_brk(at(p, p->i + 1))) return block_seq(p, col);
    if (!refused_start(p, c)) return false;
    if (looks_like_key(p)) return block_map(p, col);
    return inline_value(p, parent);
}

// ---- document --------------------------------------------------------------------

bool first_content(P* p, uint32_t* pos) noexcept {
    for (uint32_t l = 0; l < p->nlines; ++l) {                    // bounded by lines
        const Line& L = p->lines[l];
        if (L.kind == kLineBlank || L.kind == kLineComment) continue;
        if (L.kind == kLineTab) { p->i = L.content; return fail(p, YamlCode::Unsupported, "tab indentation"); }
        *pos = L.content;
        return true;
    }
    *pos = p->n;
    return true;
}

bool finish_document(P* p) noexcept;

bool document(P* p) noexcept {
    uint32_t pos = 0, ind = 0;
    if (!first_content(p, &pos)) return false;
    if (pos < p->n && p->s[pos] == '%') { p->i = pos; return fail(p, YamlCode::Unsupported, "directives"); }
    if (pos < p->n && doc_marker(p, pos, "---")) {
        p->i = pos + 3;
        const uint32_t after = p->i;
        if (!rest_empty(p)) {                                     // `--- value` / `--- |`
            p->i = after;
            skip_ws(p);
            if (!inline_value(p, -1)) return false;
            return finish_document(p);
        }
        if (!next_content(p, p->i, &pos, &ind)) return false;
    }
    if (pos >= p->n || doc_marker(p, pos, "...")) return null_value(p);
    p->i = pos;
    return block_node(p, -1) && finish_document(p);
}

// After the root node: an optional `...`, then nothing.
bool finish_document(P* p) noexcept {
    uint32_t pos = 0, ind = 0;
    if (!rest_empty(p)) return fail(p, YamlCode::Syntax, "unexpected content");
    if (!next_content(p, p->i, &pos, &ind)) return false;
    if (pos < p->n && doc_marker(p, pos, "...")) {
        p->i = pos + 3;
        if (!next_content(p, p->i, &pos, &ind)) return false;
    }
    if (pos < p->n) {
        p->i = pos;
        return fail(p, YamlCode::Unsupported, doc_marker(p, pos, "---") ? "more than one document" : "content after the document");
    }
    return true;
}
