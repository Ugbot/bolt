bool leaf_node(const SchemaField& field,nr::TypeNode& node) noexcept {
    assert(std::memchr(field.type,0,sizeof(field.type)));assert(node.n_children==0);
    IcebergTypeInfo type{};using T=IcebergType;
    if(!iceberg_type_parse(field.type,&type)) {
        if(std::strcmp(field.type,"unknown") && std::strcmp(field.type,"variant"))return false;
        node.type=BoltType::Binary;return true;
    }
    switch(type.type) {
        case T::kBoolean:node.type=BoltType::Bool;break;
        case T::kInt:node.type=BoltType::Int32;break;
        case T::kLong:case T::kTimestampNs:case T::kTimestampTzNs:node.type=BoltType::Int64;break;
        case T::kFloat:case T::kDouble:node.type=BoltType::Float64;break;
        case T::kDate:node.type=BoltType::Date32;break;
        case T::kTime:node.type=BoltType::Duration;break;
        case T::kTimestamp:case T::kTimestampTz:node.type=BoltType::Timestamp;break;
        case T::kString:node.type=BoltType::Utf8;return true;
        case T::kBinary:case T::kGeometry:case T::kGeography:node.type=BoltType::Binary;return true;
        case T::kUuid:node.type=BoltType::UUID;break;
        case T::kDecimal:node.type=type.precision<=18?BoltType::Decimal64:BoltType::Decimal128;node.scale=type.scale;break;
        case T::kFixed:
            node.type=type.fixed_len<=16?BoltType::FixedSizeBinary:BoltType::Binary;
            if(type.fixed_len>16)return true;
            node.fixed_width=static_cast<uint8_t>(type.fixed_len);break;
        default:return false;
    }
    node.fixed_bytes=static_cast<uint16_t>(bolt::type_size(node.type));return node.fixed_bytes!=0;
}
bool type_tree(State& state,const SchemaField& field,nr::TypeTree& tree,uint32_t at,uint32_t depth) noexcept {
    assert(at<tree.n);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !std::memchr(field.type,0,sizeof(field.type)))return state.capacity();
    auto& node=tree.nodes[at];node={};node.nullable=1;
    const bool structure=!std::strcmp(field.type,"struct"),list=!std::strcmp(field.type,"list"),map=!std::strcmp(field.type,"map");
    if(!structure && !list && !map)return field.n_children==0 && leaf_node(field,node);
    if((field.n_children && !field.children) || (list && field.n_children!=1) || (map && (field.n_children!=2 || !field.children[0].required)))return false;
    const auto extra=field.n_children+(map?1u:0u);
    if(extra>nr::kMaxNodes-tree.n)return state.capacity();
    node.type=structure?BoltType::Struct:list?BoltType::List:BoltType::Map;
    const auto first=tree.n;tree.n+=extra;node.first_child=static_cast<uint16_t>(first);
    node.n_children=static_cast<uint16_t>(map?1u:field.n_children);
    if(map) {auto& entry=tree.nodes[first];entry={};entry.type=BoltType::Struct;entry.nullable=1;
        entry.first_child=static_cast<uint16_t>(first+1);entry.n_children=2;}
    for(uint32_t i=0;i<field.n_children;++i)
        if(!type_tree(state,field.children[i],tree,first+(map?1u:0u)+i,depth+(map?2u:1u)))return false;
    return true;
}
struct Scalar {uint8_t cell[16]{};const uint8_t* bytes=nullptr;uint32_t size=0;};
bool scalar_number(State& state,const IcebergTypeInfo& type,const J* value,Scalar& out) noexcept {
    assert(value);assert(out.bytes==nullptr);
    using T=IcebergType;int64_t integer=0;
    if(type.type==T::kBoolean) {
        if(value->kind!=j::ValueKind::Literal)return false;
        if(value->len==4 && !std::memcmp(value->bytes,"true",4))integer=1;
        else if(value->len!=5 || std::memcmp(value->bytes,"false",5))return false;
        out.cell[0]=static_cast<uint8_t>(integer);return true;
    }
    if(type.type==T::kInt || type.type==T::kLong) {
        if(!literal_i64(value,&integer) || (type.type==T::kInt && (integer<INT32_MIN || integer>INT32_MAX)))return false;
        if(type.type==T::kInt) {const auto v=static_cast<int32_t>(integer);std::memcpy(out.cell,&v,4);}
        else std::memcpy(out.cell,&integer,8);return true;
    }
    if(type.type!=T::kFloat && type.type!=T::kDouble)return false;
    if(value->kind!=j::ValueKind::Literal || value->len<1)return false;
    double number=0;
    if(!kernels::json::json_span_to_f64({value->bytes,static_cast<uint32_t>(value->len)},&number) || !std::isfinite(number))return false;
    if(type.type==T::kFloat) {
        if(std::fabs(number)>std::numeric_limits<float>::max())return false;
        number=static_cast<float>(number);
    }
    std::memcpy(out.cell,&number,8);(void)state;return true;
}
bool scalar_parse(State& state,const SchemaField& field,const J* value,Scalar& out) noexcept {
    assert(value && !is_null(value));assert(state.arena);
    IcebergTypeInfo type{};if(!iceberg_type_parse(field.type,&type))return state.unsupported();
    using T=IcebergType;
    if(type.type==T::kBoolean || type.type==T::kInt || type.type==T::kLong || type.type==T::kFloat || type.type==T::kDouble)
        return scalar_number(state,type,value,out);
    if(type.type==T::kGeometry || type.type==T::kGeography)return state.unsupported();
    const uint8_t* text=nullptr;uint32_t size=0;if(!raw_string(state,value,&text,&size))return false;
    if(type.type==T::kString) {out.bytes=text;out.size=size;return true;}
    if(type.type==T::kBinary || type.type==T::kFixed || type.type==T::kUuid) {
        if(!hex_bytes(state,text,size,type.type==T::kUuid,&out.bytes,&out.size))return false;
        if(type.type==T::kFixed && out.size!=type.fixed_len)return false;
        if(type.type==T::kUuid || (type.type==T::kFixed && type.fixed_len<=16))std::memcpy(out.cell,out.bytes,out.size);
        return true;
    }
    if(type.type==T::kDecimal) {
        dec::Decimal128 number{};if(!decimal_parse(text,size,type,&number))return false;
        std::memcpy(out.cell,&number,type.precision<=18?8u:16u);return true;
    }
    int64_t ticks=0;if(!temporal_parse(type.type,text,size,&ticks))return false;
    if(type.type==T::kDate) {const auto days=static_cast<int32_t>(ticks);std::memcpy(out.cell,&days,4);}
    else std::memcpy(out.cell,&ticks,8);return true;
}
