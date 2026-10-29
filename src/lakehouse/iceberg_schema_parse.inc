// Included inside iceberg_metadata.cpp's anonymous namespace. Nested types
// retain their real shape, IDs and requiredness, with a bounded recursion depth.
bool parse_field(const bj::StructuralIndex*, bj::Iterator*, SchemaField*, Arena*, Budget*, uint32_t) noexcept;

bool schema_read_id(const bj::StructuralIndex* idx, bj::Iterator* it, int32_t* out) noexcept {
    assert(idx && it); assert(out);
    int64_t value = 0;
    if (!read_exact_integer(idx,it,&value) || value <= 0 || value > INT32_MAX) return false;
    *out = static_cast<int32_t>(value); return true;
}
bool schema_read_bool(bj::Iterator* it, bool* out) noexcept {
    assert(it && it->idx); assert(out);
    const auto token = bj::iter_peek(it);
    if (token != bj::TokenType::BoolTrue && token != bj::TokenType::BoolFalse) return false;
    *out = token == bj::TokenType::BoolTrue; return bj::iter_advance(it);
}
bool schema_type_name(const bj::StructuralIndex* idx, bj::Iterator* it, char* out, uint32_t cap) noexcept {
    assert(idx && it); assert(out && cap);
    if (bj::iter_peek(it) != bj::TokenType::String) return false;
    const auto& token=idx->tokens[it->cursor];
    if (token.start<1 || token.length<0 || token.length>=idx->src_len || token.start>=idx->src_len-token.length) return false;
    const bolt::kernels::json::JsonSpan span{reinterpret_cast<const char*>(idx->src+token.start-1),static_cast<uint32_t>(token.length)+2u};
    const auto n=bolt::kernels::json::json_span_unquote(span,out,cap-1u);
    if (n<0 || std::memchr(out,0,static_cast<size_t>(n))) return false;
    out[n]=0;return bj::iter_advance(it);
}
bool nested_type_name(const bj::StructuralIndex* idx, bj::Iterator it, char* out, uint32_t cap) noexcept {
    assert(idx && it.idx==idx); assert(out && cap);
    if (bj::iter_peek(&it) != bj::TokenType::BeginObject || !bj::iter_advance(&it)) return false;
    bool found = false;
    for (uint32_t g=0; bj::iter_peek(&it)==bj::TokenType::Key && g<kIterGuard; ++g) {
        const auto key=it.cursor; if (!bj::iter_advance(&it)) return false;
        if (tok_eq(idx,key,"type")) {
            if (found || !schema_type_name(idx,&it,out,cap)) return false;
            found=true;
        } else if (!skip_value(&it)) return false;
    }
    return found && bj::iter_peek(&it)==bj::TokenType::EndObject;
}
bool parse_schema_type(const bj::StructuralIndex*,bj::Iterator*,SchemaField*,Arena*,Budget*,uint32_t) noexcept;

bool parse_struct_children(const bj::StructuralIndex* idx,bj::Iterator* it,SchemaField* out,
                          Arena* arena,Budget* budget,uint32_t depth) noexcept {
    assert(idx && it && out); assert(arena && depth<=kIcebergSchemaMaxDepth);
    if (bj::iter_peek(it)!=bj::TokenType::BeginArray || !bj::iter_advance(it)) return false;
    const uint32_t bound=array_guard(idx);
    for (uint32_t g=0; bj::iter_peek(it)==bj::TokenType::BeginObject && g<bound; ++g) {
        SchemaField child{};
        if (!parse_field(idx,it,&child,arena,budget,depth+1u) ||
            !lake_grow(arena,budget,&out->children,out->n_children,&out->cap_children,out->n_children+1u)) return false;
        out->children[out->n_children++]=child;
    }
    return bj::iter_peek(it)==bj::TokenType::EndArray && bj::iter_advance(it);
}

bool parse_collection_member(const bj::StructuralIndex* idx,bj::Iterator* it,SchemaField* out,
                             int32_t key,Arena* arena,Budget* budget,uint32_t depth,uint32_t* seen) noexcept {
    assert(idx && it && out); assert(arena && seen);
    const bool list=std::strcmp(out->type,"list")==0;
    const char* names[5]={list?"element-id":"key-id",list?"element-required":"key",
                         list?"element":"value-id","value-required","value"};
    for (uint32_t i=0; i<(list?3u:5u); ++i) {
        if (!tok_eq(idx,key,names[i])) continue;
        if ((*seen & (1u<<i)) != 0u) return false;
        *seen |= 1u<<i;
        auto& child=out->children[list || i<2u?0u:1u];
        if (i==0u || (!list && i==2u)) return schema_read_id(idx,it,&child.id);
        if ((list && i==1u) || (!list && i==3u)) return schema_read_bool(it,&child.required);
        return parse_schema_type(idx,it,&child,arena,budget,depth+1u);
    }
    return skip_value(it);
}

bool parse_schema_type(const bj::StructuralIndex* idx,bj::Iterator* it,SchemaField* out,
                       Arena* arena,Budget* budget,uint32_t depth) noexcept {
    assert(idx && it && out); assert(arena);
    if (depth>kIcebergSchemaMaxDepth) return false;
    if (bj::iter_peek(it)==bj::TokenType::String) {
        if (!schema_type_name(idx,it,out->type,sizeof(out->type))) return false;
        return std::strcmp(out->type,"struct")!=0 && std::strcmp(out->type,"list")!=0 && std::strcmp(out->type,"map")!=0;
    }
    if (!nested_type_name(idx,*it,out->type,sizeof(out->type))) return false;
    const bool structure=std::strcmp(out->type,"struct")==0;
    const bool list=std::strcmp(out->type,"list")==0;
    if (!structure && !list && std::strcmp(out->type,"map")!=0) return false;
    if (!structure) {
        out->n_children=list?1u:2u;
        if (!lake_grow(arena,budget,&out->children,0u,&out->cap_children,out->n_children)) return false;
        std::memset(out->children,0,sizeof(SchemaField)*out->n_children);
        std::strcpy(out->children[0].name,list?"element":"key");
        if (!list) { out->children[0].required=true;std::strcpy(out->children[1].name,"value"); }
    }
    if (!bj::iter_advance(it)) return false;
    uint32_t seen=0;bool fields=false;
    for (uint32_t g=0; bj::iter_peek(it)==bj::TokenType::Key && g<kIterGuard; ++g) {
        const auto key=it->cursor;if (!bj::iter_advance(it)) return false;
        if (tok_eq(idx,key,"type")) { if (!skip_value(it)) return false; }
        else if (structure && tok_eq(idx,key,"fields")) {
            if (fields || !parse_struct_children(idx,it,out,arena,budget,depth)) return false;
            fields=true;
        } else if (structure) { if (!skip_value(it)) return false; }
        else if (!parse_collection_member(idx,it,out,key,arena,budget,depth,&seen)) return false;
    }
    if ((structure && !fields) || (!structure && seen!=(list?7u:31u))) return false;
    return bj::iter_peek(it)==bj::TokenType::EndObject && bj::iter_advance(it);
}

bool parse_field(const bj::StructuralIndex* idx,bj::Iterator* it,SchemaField* out,
                 Arena* arena,Budget* budget,uint32_t depth) noexcept {
    assert(idx && it && out); assert(arena && it->idx==idx);
    if (depth>kIcebergSchemaMaxDepth || bj::iter_peek(it)!=bj::TokenType::BeginObject || !bj::iter_advance(it)) return false;
    *out=SchemaField{};uint32_t seen=0;
    for (uint32_t g=0; bj::iter_peek(it)==bj::TokenType::Key && g<kIterGuard; ++g) {
        const auto key=it->cursor;if (!bj::iter_advance(it)) return false;
        uint32_t bit=0;
        if (tok_eq(idx,key,"id")) bit=1u;
        else if (tok_eq(idx,key,"name")) bit=2u;
        else if (tok_eq(idx,key,"type")) bit=4u;
        else if (tok_eq(idx,key,"required")) bit=8u;
        if (bit && (seen&bit)) return false;
        seen|=bit;
        if (bit==1u) { if (!schema_read_id(idx,it,&out->id)) return false; }
        else if (bit==2u) { if (!schema_type_name(idx,it,out->name,sizeof(out->name))) return false; }
        else if (bit==4u) { if (!parse_schema_type(idx,it,out,arena,budget,depth)) return false; }
        else if (bit==8u) { if (!schema_read_bool(it,&out->required)) return false; }
        else if (tok_eq(idx,key,"initial-default")) read_default(idx,it,out);
        else if (!skip_value(it)) return false;
    }
    return (seen&7u)==7u && bj::iter_peek(it)==bj::TokenType::EndObject && bj::iter_advance(it);
}
