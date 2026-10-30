// SchemaField -> maintained Parquet node table; map entry wrappers are owned by
// the Parquet writer, so the map node names only its real key/value children.
bool nested_pq_options(const SchemaField& root,Arena* arena,ingest::parquet::ParquetWriteColumn* out) noexcept {
    assert(arena && out);assert(ingest::parquet::kPwMaxNestedNodes<=UINT16_MAX);
    namespace p=ingest::parquet;constexpr auto cap=p::kPwMaxNestedNodes;
    auto* nodes=arena->allocate_array<p::ParquetNestedNode>(cap);if(!nodes)return false;
    const SchemaField* fields[cap]{};fields[0]=&root;uint32_t count=1;
    for(uint32_t i=0;i<count;++i) {
        const auto& f=*fields[i];auto& n=nodes[i];n={};
        if(!std::memchr(f.name,0,sizeof(f.name)) || std::strlen(f.name)>=sizeof(n.name))return false;
        std::strcpy(n.name,f.name);n.nullable=!f.required;n.has_field_id=true;n.field_id=f.id;
        const bool st=!std::strcmp(f.type,"struct"),list=!std::strcmp(f.type,"list"),map=!std::strcmp(f.type,"map");
        if(st || list || map) {
            if(!f.n_children || !f.children || f.n_children>cap-count)return false;
            n.type=st?BoltType::Struct:list?BoltType::List:BoltType::Map;
            n.first_child=static_cast<uint16_t>(count);n.n_children=static_cast<uint16_t>(f.n_children);
            for(uint32_t j=0;j<f.n_children;++j)fields[count++]=&f.children[j];
            continue;
        }
        p::ParquetWriteColumn leaf{};if(!pq_iceberg_type(f.type,&leaf,&n.type))return false;
        n.precision=leaf.precision;n.scale=leaf.scale;n.logical=leaf.logical;
        n.ts_local=leaf.ts_local;n.timestamp_unit=leaf.timestamp_unit;n.time_micros=leaf.time_micros;
        n.integer_bits=leaf.integer_bits;n.integer_signed=leaf.integer_signed;n.fixed_len=leaf.fixed_len;
        n.geo=leaf.geo;n.geo_algorithm=leaf.geo_algorithm;n.geo_crs=leaf.geo_crs;n.geo_crs_len=leaf.geo_crs_len;
    }
    out->nested=nodes;out->n_nested=count;return true;
}
