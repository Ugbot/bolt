struct BatchLeaf {
    const BoltColumn* value=nullptr;
    const BoltColumn* parents[kIcebergSchemaMaxDepth+1u]{};
    uint32_t depth=0;
};
bool batch_leaf(const BoltBatch& batch,const char* root,const uint32_t* path,uint32_t depth,BatchLeaf* out) noexcept {
    assert(root && path && out);assert(depth<=kIcebergSchemaMaxDepth);
    if(batch.num_cols>kMaxColumns || batch.schema.num_fields<batch.num_cols || batch.read_epoch>1 ||
        (batch.num_cols && (!batch.schema.fields || !batch.columns[batch.read_epoch])) || batch.num_rows<0 || batch.num_rows>INT32_MAX)return false;
    const BoltColumn* value=nullptr;
    for(uint32_t i=0;i<batch.num_cols;++i) {
        if(std::strcmp(batch.schema.fields[i].name,root))continue;
        if(value)return false;value=&batch.col(i);
    }
    if(!value)return false;
    for(uint32_t i=0;i<depth;++i) {
        if(value->length!=batch.num_rows || value->type!=BoltType::Struct || value->format!=ColumnFormat::Nested || path[i]>=value->child_count())return false;
        out->parents[i]=value;value=value->child_at(path[i]);if(!value)return false;
    }
    if(value->length!=batch.num_rows)return false;
    out->value=value;out->depth=depth;return true;
}
bool batch_leaf_metrics(const BoltBatch* const* batches,uint32_t n_batches,const char* root,const SchemaField& field,
    const uint32_t* path,uint32_t depth,const MetricsModeSpec& mode,FileStats* out,Arena* arena,bool* refused) noexcept {
    assert(batches && out && arena && refused);assert(depth<=kIcebergSchemaMaxDepth);
    ValueFold fold{};if(!iceberg_type_parse(field.type,&fold.t))return false;
    fold.mode=mode;fold.a=arena;int64_t values=0,nulls=0;
    for(uint32_t i=0;i<n_batches;++i) {
        if(!batches[i])return false;const auto& batch=*batches[i];BatchLeaf leaf{};
        if(!batch_leaf(batch,root,path,depth,&leaf))return false;
        values+=batch.num_rows;
        for(int64_t row=0;row<batch.num_rows;++row) {
            bool missing=leaf.value->is_null(row);
            for(uint32_t p=0;p<leaf.depth;++p)missing|=leaf.parents[p]->is_null(row);
            if(missing)++nulls;else fold_row(&fold,*leaf.value,row);
        }
    }
    auto* entry=stats_add_col(out,arena,field.id);if(!entry) {*refused=true;return false;}
    entry->null_count=nulls;entry->has_value_count=true;entry->value_count=values;
    if((fold.t.type==IcebergType::kFloat || fold.t.type==IcebergType::kDouble) && fold.ok && !fold.refused) {
        entry->has_nan_count=true;entry->nan_count=fold.nan;
    }
    if(!emit_value_bounds(fold,entry,arena))*refused=true;
    return !*refused;
}
bool batch_field_metrics(const BoltBatch* const* batches,uint32_t n_batches,const char* root,const SchemaField& field,
    uint32_t* path,uint32_t depth,uint32_t* visited,const MetricsModeSpec& mode,FileStats* out,Arena* arena) noexcept {
    assert(path && visited);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if(depth>kIcebergSchemaMaxDepth || *visited==kMaxColumns)return false;++*visited;
    if(!std::strcmp(field.type,"struct")) {
        if(field.n_children>kMaxColumns || (field.n_children && !field.children))return false;
        for(uint32_t i=0;i<field.n_children;++i) {
            path[depth]=i;
            if(!batch_field_metrics(batches,n_batches,root,field.children[i],path,depth+1u,visited,mode,out,arena))return false;
        }
        return true;
    }
    // Repeated leaves do not have row-level counts/bounds. Column sizes belong
    // to the actual file; the batch API cannot supply them.
    if(!std::strcmp(field.type,"list") || !std::strcmp(field.type,"map"))return true;
    bool refused=false;(void)batch_leaf_metrics(batches,n_batches,root,field,path,depth,mode,out,arena,&refused);
    return !refused;
}
