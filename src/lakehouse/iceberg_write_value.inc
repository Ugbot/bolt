S write_fixed(const IcebergTypeInfo& type,Arena* arena,BoltColumn* value) noexcept {
    assert(arena && value);assert(type.type==IcebergType::kFixed);
    if(value->type==BoltType::Binary)return S::Ok;
    if(value->type!=BoltType::FixedSizeBinary || value->format!=ColumnFormat::Flat || value->fixed_width!=type.fixed_len ||
        value->type_size_bytes!=16 || type.fixed_len>16 || !value->data)return S::Invalid;
    const auto rows=static_cast<uint32_t>(value->length);const uint64_t total=uint64_t{rows}*type.fixed_len;
    if(total>INT32_MAX)return S::Capacity;
    auto* bytes=arena->allocate_array<uint8_t>(static_cast<size_t>(total)+1u);
    auto* offsets=arena->allocate_array<int32_t>(static_cast<size_t>(rows)+1u);if(!bytes || !offsets)return S::Capacity;
    uint32_t at=0;offsets[0]=0;
    for(uint32_t i=0;i<rows;++i) {
        if(!value->is_null(i)) {std::memcpy(bytes+at,static_cast<const uint8_t*>(value->data)+uint64_t{i}*16u,type.fixed_len);at+=type.fixed_len;}
        offsets[i+1]=static_cast<int32_t>(at);
    }
    auto out=BoltColumn::make_var_binary(bytes,value->validity,offsets,rows,BoltType::Binary,arena);
    out.validity_offset=value->validity_offset;*value=out;return S::Ok;
}
S write_numeric(const IcebergTypeInfo& type,Arena* arena,BoltColumn* value) noexcept {
    assert(arena && value);assert(value->length>=0 && value->length<=INT32_MAX);
    using T=IcebergType;const auto count=static_cast<size_t>(value->length);
    if(value->format!=ColumnFormat::Flat || (count && !value->data))return S::Invalid;
    if(type.type==T::kFloat && value->type==BoltType::Float64 && value->type_size_bytes==8) {
        auto* floats=arena->allocate_array<float>(count+1u);if(!floats)return S::Capacity;
        for(size_t i=0;i<count;++i) {
            if(value->is_null(static_cast<int64_t>(i))) {floats[i]=0;continue;}
            const auto number=static_cast<const double*>(value->data)[i];
            if(std::isfinite(number) && std::fabs(number)>std::numeric_limits<float>::max())return S::Invalid;
            floats[i]=static_cast<float>(number);
        }
        value->data=floats;value->type=BoltType::Float32;value->type_size_bytes=4;return S::Ok;
    }
    if(type.type==T::kDecimal && value->type==BoltType::Decimal64 && value->type_size_bytes==8) {
        auto* wide=arena->allocate_array<int64_t>((count+1u)*2u);if(!wide)return S::Capacity;
        for(size_t i=0;i<count;++i) {wide[2*i]=value->is_null(static_cast<int64_t>(i))?0:static_cast<const int64_t*>(value->data)[i];wide[2*i+1]=wide[2*i]<0?-1:0;}
        value->data=wide;value->type=BoltType::Decimal128;value->type_size_bytes=16;return S::Ok;
    }
    const bool timestamp=type.type==T::kTimestamp || type.type==T::kTimestampTz || type.type==T::kTimestampNs || type.type==T::kTimestampTzNs;
    if(timestamp && value->type_size_bytes==8 && (value->type==BoltType::Int64 || value->type==BoltType::Timestamp)) {value->type=BoltType::Timestamp;return S::Ok;}
    if(type.type==T::kTime && value->type_size_bytes==8 && (value->type==BoltType::Duration || value->type==BoltType::Int64)) {value->type=BoltType::Int64;return S::Ok;}
    return S::Ok; // Remaining declared lanes are checked by the maintained writer.
}
S write_child_presence(const BoltColumn& parent,Arena* arena,BoltColumn* child) noexcept {
    assert(arena && child);assert(parent.length>=0 && parent.length<=INT32_MAX);
    if(child->length!=parent.length)return S::Invalid;
    if(!parent.validity)return S::Ok;
    auto* validity=static_cast<uint8_t*>(arena->allocate_zeroed((static_cast<size_t>(parent.length)+7u)/8u+1u));
    if(!validity)return S::Capacity;uint32_t nulls=0;
    for(int64_t i=0;i<parent.length;++i) {
        const bool present=!parent.is_null(i) && !child->is_null(i);
        validity[i>>3]|=static_cast<uint8_t>(static_cast<uint8_t>(present)<<(i&7));nulls+=present?0u:1u;
    }
    child->validity=nulls?validity:nullptr;child->validity_offset=0;child->stats.all_valid=nulls==0;child->stats.null_count=nulls;
    return S::Ok;
}
S write_value_prepare(const SchemaField& field,Arena* arena,BoltColumn* value,uint32_t depth) noexcept {
    assert(arena && value);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if(depth>kIcebergSchemaMaxDepth || value->length<0 || value->length>INT32_MAX)return S::Invalid;
    const bool st=!std::strcmp(field.type,"struct"),list=!std::strcmp(field.type,"list"),map=!std::strcmp(field.type,"map");
    if(st || list || map) {
        if(st && !field.n_children)return S::Unsupported;
        const auto type=st?BoltType::Struct:list?BoltType::List:BoltType::Map;
        if(value->type!=type || value->format!=ColumnFormat::Nested || !value->data)return S::Invalid;
        const auto count=st?field.n_children:1u;
        if(value->child_count()!=count)return S::Invalid;
        auto* children=arena->allocate_array<BoltColumn>(count?count:1u);if(!children)return S::Capacity;
        for(uint32_t i=0;i<count;++i) {
            children[i]=*value->child_at(i);
            if(st) {const auto status=write_child_presence(*value,arena,&children[i]);if(status!=S::Ok)return status;}
        }
        if(map) {
            if(children[0].type!=BoltType::Struct || children[0].child_count()!=2)return S::Invalid;
            for(int64_t r=0;r<children[0].length;++r)if(children[0].is_null(r))return S::Invalid;
            auto* entry=arena->allocate_array<BoltColumn>(2);if(!entry)return S::Capacity;
            for(uint32_t i=0;i<2;++i) {entry[i]=*children[0].child_at(i);const auto status=write_value_prepare(field.children[i],arena,&entry[i],depth+1);if(status!=S::Ok)return status;}
            children[0].data=entry;
        } else for(uint32_t i=0;i<count;++i) {const auto status=write_value_prepare(field.children[i],arena,&children[i],depth+1);if(status!=S::Ok)return status;}
        value->data=children;return S::Ok;
    }
    IcebergTypeInfo type{};if(!iceberg_type_parse(field.type,&type))return S::Unsupported;
    if(type.type==IcebergType::kFixed)return write_fixed(type,arena,value);
    if(value->type==BoltType::Utf8 || value->type==BoltType::Binary || value->type==BoltType::UUID)return S::Ok;
    return write_numeric(type,arena,value);
}
