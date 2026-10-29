// Included after the Buf helpers. One recursive type emitter is shared by
// table metadata and the schema embedded in manifests.
bool emit_schema_field(Buf*,const SchemaField&,uint32_t) noexcept;
bool emit_schema_type(Buf* out,const SchemaField& field,uint32_t depth) noexcept {
    assert(out);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if (depth>kIcebergSchemaMaxDepth) return false;
    const bool structure=std::strcmp(field.type,"struct")==0;
    const bool list=std::strcmp(field.type,"list")==0;
    const bool map=std::strcmp(field.type,"map")==0;
    if (!structure && !list && !map)
        return buf_put(out,"\"") && buf_put_jstr(out,field.type) && buf_put(out,"\"");
    if (field.n_children && !field.children) return false;
    if (!buf_put(out,"{") || !buf_kv_str(out,"type",field.type,true)) return false;
    if (structure) {
        if (!buf_put(out,"\"fields\":[")) return false;
        for (uint32_t i=0;i<field.n_children;++i) {
            if ((i && !buf_put(out,",")) || !emit_schema_field(out,field.children[i],depth+1u)) return false;
        }
        return buf_put(out,"]}");
    }
    if (field.n_children!=(list?1u:2u)) return false;
    const auto& first=field.children[0];
    if (!buf_fmt(out,list?"\"element-id\":%d,\"element-required\":%s,\"element\":"
                         :"\"key-id\":%d,\"value-required\":%s,\"key\":",first.id,
                 (list?first.required:field.children[1].required)?"true":"false") ||
        !emit_schema_type(out,first,depth+1u)) return false;
    if (map) {
        if (!first.required || !buf_fmt(out,",\"value-id\":%d,\"value\":",field.children[1].id) ||
            !emit_schema_type(out,field.children[1],depth+1u)) return false;
    }
    return buf_put(out,"}");
}
bool emit_schema_field(Buf* out,const SchemaField& field,uint32_t depth) noexcept {
    assert(out);assert(field.id>0);
    return buf_fmt(out,"{\"id\":%d,",field.id) && buf_kv_str(out,"name",field.name,true) &&
        buf_fmt(out,"\"required\":%s,\"type\":",field.required?"true":"false") &&
        emit_schema_type(out,field,depth) && buf_put(out,"}");
}
