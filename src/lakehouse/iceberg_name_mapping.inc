struct NameMappingRead { NameMapEntry* entries=nullptr;uint32_t count=0,cap=0;Arena* arena=nullptr; };
bool read_name_mapping_array(const bj::StructuralIndex*,bj::Iterator*,NameMappingRead&,uint32_t) noexcept;
bool read_name_mapping_names(const bj::StructuralIndex* idx,bj::Iterator* it,NameMapEntry& entry) noexcept {
    assert(idx && it);assert(entry.n_names==0u);
    if (bj::iter_peek(it)!=bj::TokenType::BeginArray || !bj::iter_advance(it)) return false;
    for (uint32_t i=0;bj::iter_peek(it)==bj::TokenType::String && i<kNameMapMaxNames;++i) {
        if (!schema_type_name(idx,it,entry.names[entry.n_names],kIcebergMaxFieldName)) return false;
        ++entry.n_names;
    }
    return bj::iter_peek(it)==bj::TokenType::EndArray && bj::iter_advance(it);
}
bool read_name_mapping_field(const bj::StructuralIndex* idx,bj::Iterator* it,NameMappingRead& out,uint32_t depth) noexcept {
    assert(idx && it);assert(out.arena && depth<=kIcebergSchemaMaxDepth);
    NameMapEntry entry{};entry.field_id=-1;uint32_t seen=0;
    if (!bj::iter_advance(it)) return false;
    for (uint32_t i=0;bj::iter_peek(it)==bj::TokenType::Key && i<kIterGuard;++i) {
        const auto key=it->cursor;if (!bj::iter_advance(it)) return false;
        if (tok_eq(idx,key,"field-id")) {
            if (seen&1u) return false;seen|=1u;
            if (bj::iter_peek(it)==bj::TokenType::Null) {if (!bj::iter_advance(it)) return false;}
            else if (!schema_read_id(idx,it,&entry.field_id)) return false;
        } else if (tok_eq(idx,key,"names")) {
            if ((seen&2u) || !read_name_mapping_names(idx,it,entry)) return false;seen|=2u;
        } else if (tok_eq(idx,key,"fields")) {
            if ((seen&4u) || !read_name_mapping_array(idx,it,out,depth+1u)) return false;seen|=4u;
        } else if (!skip_value(it)) return false;
    }
    if (bj::iter_peek(it)!=bj::TokenType::EndObject || !bj::iter_advance(it)) return false;
    if (entry.field_id>0) {
        if (!lake_grow(out.arena,nullptr,&out.entries,out.count,&out.cap,out.count+1u)) return false;
        out.entries[out.count++]=entry;
    }
    return true;
}
bool read_name_mapping_array(const bj::StructuralIndex* idx,bj::Iterator* it,NameMappingRead& out,uint32_t depth) noexcept {
    assert(idx && it);assert(out.arena);
    if (depth>kIcebergSchemaMaxDepth || bj::iter_peek(it)!=bj::TokenType::BeginArray || !bj::iter_advance(it)) return false;
    const auto bound=array_guard(idx);
    for (uint32_t i=0;bj::iter_peek(it)==bj::TokenType::BeginObject && i<bound;++i)
        if (!read_name_mapping_field(idx,it,out,depth)) return false;
    return bj::iter_peek(it)==bj::TokenType::EndArray && bj::iter_advance(it);
}
