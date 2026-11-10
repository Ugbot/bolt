// bolt_yaml_api.inc — the public entry points (included by bolt_yaml.cpp).

YamlCode build_index(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                     const json::PathFilter* filter, json::StructuralIndex* out, YamlError* err) noexcept {
    assert(arena != nullptr && out != nullptr && err != nullptr);
    *err = YamlError{YamlCode::Ok, 0, 0, ""};
    std::memset(out, 0, sizeof(*out));
    if (src == nullptr && len != 0) { err->code = YamlCode::Syntax; err->what = "null input"; return err->code; }
    if (limits.bytes == 0 || limits.bytes > 0x7FFFFFFFu || limits.depth == 0 || limits.depth > kMaxDepth ||
        limits.nodes == 0) {
        err->code = YamlCode::Limit; err->what = "invalid limits"; return err->code;
    }
    if (len > limits.bytes) { err->code = YamlCode::Limit; err->what = "input bytes"; return err->code; }
    P* p = arena->allocate_array<P>(1);
    if (p == nullptr) { err->code = YamlCode::NoMemory; err->what = "arena exhausted"; return err->code; }
    std::memset(p, 0, offsetof(P, seg_start));                    // the path stack is written before read
    p->s = src == nullptr ? "" : src;
    p->n = len;
    p->arena = arena;
    p->lim = limits;
    p->err = err;
    p->emit = true;
    p->filter = (filter != nullptr && filter->impl != nullptr) ? filter : nullptr;
    uint32_t kw = 0;
    const bool ok = prescan(p) && aux_raw(p, "nulltruefalse", kAuxKeywords, &kw) && document(p) &&
                    tok(p, json::TokenType::End, p->n, 0, 0);
    if (!ok) {
        if (err->code == YamlCode::Ok) { err->code = YamlCode::Syntax; err->what = "invalid YAML"; }
        return err->code;
    }
    assert(kw == kAuxNull);
    out->src = reinterpret_cast<const uint8_t*>(p->s);
    out->src_len = static_cast<int32_t>(p->n);
    out->tokens = p->tape;
    out->token_count = static_cast<int32_t>(p->ntok);
    out->aux = reinterpret_cast<const uint8_t*>(p->aux);
    out->aux_len = static_cast<int32_t>(p->naux);
    return YamlCode::Ok;
}

YamlCode to_json(const char* src, uint32_t len, Arena* arena, YamlLimits limits,
                 const char** out, uint32_t* out_len, YamlError* err) noexcept {
    assert(arena != nullptr && out != nullptr && out_len != nullptr && err != nullptr);
    *out = nullptr;
    *out_len = 0;
    json::StructuralIndex idx{};
    const YamlCode c = build_index(src, len, arena, limits, nullptr, &idx, err);
    if (c != YamlCode::Ok) return c;
    json::JsonValue* root = nullptr;
    const uint8_t* bytes = nullptr;
    int32_t n = 0;
    if (!json::value_from_index(&idx, arena, &root) || !json::serialize(root, arena, &bytes, &n)) {
        err->code = YamlCode::NoMemory;
        err->what = "arena exhausted";
        return err->code;
    }
    if (static_cast<uint32_t>(n) > limits.bytes) {
        err->code = YamlCode::Limit;
        err->what = "output bytes";
        return err->code;
    }
    char* z = static_cast<char*>(arena->allocate(static_cast<size_t>(n) + 1u, 1));
    if (z == nullptr) { err->code = YamlCode::NoMemory; err->what = "arena exhausted"; return err->code; }
    std::memcpy(z, bytes, static_cast<size_t>(n));
    z[n] = '\0';
    *out = z;
    *out_len = static_cast<uint32_t>(n);
    return YamlCode::Ok;
}
