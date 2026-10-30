// Iceberg unknown fields have no physical Parquet column. Logical descriptors
// retain them; only file schemas/batches project them away. No dummy fields.
bool physical_has_unknown(const SchemaField* fields,uint32_t count,uint32_t depth) noexcept {
    assert(depth<=kIcebergSchemaMaxDepth+1u);assert(fields || count==0);
    if(depth>kIcebergSchemaMaxDepth || count>kMaxColumns)return true;
    for(uint32_t i=0;i<count;++i) {
        if(!std::strcmp(fields[i].type,"unknown"))return true;
        if(fields[i].n_children && (!fields[i].children || physical_has_unknown(fields[i].children,fields[i].n_children,depth+1u)))return true;
    }
    return false;
}
bool physical_schema_field(const SchemaField& source,Arena* arena,SchemaField* out,uint32_t depth) noexcept {
    assert(arena && out);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if(depth>kIcebergSchemaMaxDepth || source.n_children>kMaxColumns || (source.n_children && !source.children))return false;
    *out=source;
    if(!source.n_children)return std::strcmp(source.type,"struct")!=0;
    auto* children=arena->allocate_array<SchemaField>(source.n_children);if(!children)return false;
    uint32_t count=0;
    for(uint32_t i=0;i<source.n_children;++i) {
        if(!std::strcmp(source.children[i].type,"unknown")) {
            if(source.children[i].required || std::strcmp(source.type,"struct"))return false;
            continue;
        }
        if(!physical_schema_field(source.children[i],arena,&children[count++],depth+1u))return false;
    }
    if(!count)return false;
    out->children=children;out->n_children=count;out->cap_children=count;return true;
}
bool physical_schema(const Schema& source,Arena* arena,Schema* out) noexcept {
    assert(arena && out);assert(kMaxColumns>0);
    if(!source.n_fields || source.n_fields>kMaxColumns || !source.fields)return false;
    if(!physical_has_unknown(source.fields,source.n_fields,0)) {*out=source;return true;}
    auto* fields=arena->allocate_array<SchemaField>(source.n_fields);if(!fields)return false;
    uint32_t count=0;
    for(uint32_t i=0;i<source.n_fields;++i) {
        if(!std::strcmp(source.fields[i].type,"unknown")) {if(source.fields[i].required)return false;continue;}
        if(!physical_schema_field(source.fields[i],arena,&fields[count++],0))return false;
    }
    if(!count)return false;
    *out=source;out->fields=fields;out->n_fields=count;out->cap_fields=count;return true;
}
bool physical_unknown(const BoltColumn& value) noexcept {
    assert(value.length>=0);assert(sizeof(int64_t)>=sizeof(int32_t));
    if(value.length>INT32_MAX)return false;
    for(int64_t r=0;r<value.length;++r)if(!value.is_null(r))return false;
    return true;
}
bool physical_column(const SchemaField& field,const BoltColumn& source,Arena* arena,BoltColumn* out,uint32_t depth) noexcept {
    assert(arena && out);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if(depth>kIcebergSchemaMaxDepth || field.n_children>kMaxColumns)return false;
    *out=source;if(!field.n_children)return std::strcmp(field.type,"struct")!=0;
    const bool structure=!std::strcmp(field.type,"struct"),map=!std::strcmp(field.type,"map");
    const auto* parent=structure?&source:source.list_element();
    if(source.format!=ColumnFormat::Nested || !parent || (structure && source.type!=BoltType::Struct))return false;
    if((structure || map) && (parent->type!=BoltType::Struct || parent->child_count()!=field.n_children))return false;
    auto* children=arena->allocate_array<BoltColumn>(field.n_children);if(!children)return false;
    uint32_t count=0;
    for(uint32_t i=0;i<field.n_children;++i) {
        const auto* value=structure || map?parent->child_at(i):parent;if(!value)return false;
        if(!std::strcmp(field.children[i].type,"unknown")) {
            if(!structure || !physical_unknown(*value))return false;continue;
        }
        if(!physical_column(field.children[i],*value,arena,&children[count++],depth+1u))return false;
    }
    if(!count)return false;
    auto selected=*parent;
    if(structure || map) {selected.data=children;selected.seq_offset=count;}else selected=children[0];
    if(structure) {*out=selected;return true;}
    auto result=BoltColumn::make_list(&selected,source.list_offsets(),source.length,source.validity,arena);
    if(!result.data)return false;out->data=result.data;return true;
}
bool physical_batch(const Schema& schema,const BoltBatch& source,Arena* arena,BoltBatch* out) noexcept {
    assert(arena && out);assert(schema.n_fields<=kMaxColumns);
    if(source.num_cols!=schema.n_fields || source.schema.num_fields!=source.num_cols || source.read_epoch>1 ||
        !source.columns[source.read_epoch] || source.num_rows<0 || source.num_rows>INT32_MAX)return false;
    BoltBatch::init_empty(out);out->arena=arena;out->num_rows=source.num_rows;
    if(!BoltBatch::alloc_columns(out,arena,schema.n_fields))return false;
    out->num_cols=0;
    for(uint32_t i=0;i<schema.n_fields;++i) {
        const auto& field=schema.fields[i];const auto& value=source.col(i);
        if(std::strcmp(field.name,source.schema.fields[i].name) || value.length!=source.num_rows)return false;
        if(!std::strcmp(field.type,"unknown")) {if(field.required || !physical_unknown(value))return false;continue;}
        BoltColumn selected{};if(!physical_column(field,value,arena,&selected,0))return false;
        if(!out->schema.add_field(field.name,selected.type,!field.required))return false;
        out->schema.fields[out->num_cols]=source.schema.fields[i];out->columns[0][out->num_cols++]=selected;
    }
    return out->num_cols>0;
}
bool physical_row_group(ingest::parquet::ParquetWriter* writer,const Schema& schema,
    const BoltBatch& batch,Arena* arena) noexcept {
    assert(writer && arena);assert(schema.n_fields<=kMaxColumns);
    if(!physical_has_unknown(schema.fields,schema.n_fields,0))return ingest::parquet::parquet_write_row_group(writer,&batch);
    BoltBatch physical{};
    return physical_batch(schema,batch,arena,&physical) && ingest::parquet::parquet_write_row_group(writer,&physical);
}
