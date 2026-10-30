J* text_value(State& state,const char* text,uint32_t size) noexcept {
    assert(text || !size);assert(state.arena);
    auto* bytes=state.arena->allocate_array<char>(static_cast<size_t>(size)+1u);
    if(!bytes) {state.capacity();return nullptr;}
    if(size)std::memcpy(bytes,text,size);bytes[size]=0;
    auto* value=j::make_string(state.arena,bytes,static_cast<int32_t>(size));
    if(!value)state.capacity();return value;
}
bool column_bytes(const BoltColumn& column,int64_t row,const uint8_t** bytes,uint32_t* size) noexcept {
    assert(bytes && size);assert(row>=0 && row<column.length);
    if(column.type==BoltType::UUID || column.type==BoltType::FixedSizeBinary) {
        const uint32_t width=column.type==BoltType::UUID?16u:column.fixed_width;
        if(column.format!=ColumnFormat::Flat || column.type_size_bytes!=16 || width>16 || !width || !column.data)return false;
        *bytes=static_cast<const uint8_t*>(column.data)+row*16;*size=width;return true;
    }
    if(column.type!=BoltType::Utf8 && column.type!=BoltType::Binary)return false;
    if(column.format!=ColumnFormat::Flat && column.format!=ColumnFormat::VarBinary)return false;
    if(column.format==ColumnFormat::Flat && (column.type_size_bytes!=sizeof(StringView) || !column.data))return false;
    if(column.format==ColumnFormat::VarBinary && (!column.dict_child || !column.dict_child->data || column.dict_child->length<column.length+1))return false;
    int32_t count=0;column.utf8_at(row,bytes,&count);
    if(count<0 || (count && !*bytes))return false;*size=static_cast<uint32_t>(count);return true;
}
bool column_integer(const BoltColumn& column,int64_t row,int64_t* out) noexcept {
    assert(out);assert(row>=0 && row<column.length);
    if(column.format!=ColumnFormat::Flat || !column.data)return false;
    if(column.type==BoltType::Bool && column.type_size_bytes==1) {*out=static_cast<const uint8_t*>(column.data)[row];return true;}
    if((column.type==BoltType::Int32 || column.type==BoltType::Date32) && column.type_size_bytes==4) {
        int32_t value=0;std::memcpy(&value,static_cast<const uint8_t*>(column.data)+row*4,4);*out=value;return true;
    }
    if((column.type==BoltType::Int64 || column.type==BoltType::Timestamp || column.type==BoltType::Duration) && column.type_size_bytes==8) {
        std::memcpy(out,static_cast<const uint8_t*>(column.data)+row*8,8);return true;
    }
    return false;
}
J* encoded_hex(State& state,const uint8_t* bytes,uint32_t size,bool uuid) noexcept {
    assert(bytes || !size);assert(!uuid || size==16);
    const uint64_t length=uint64_t{size}*2u+(uuid?4u:0u);
    if(length>state.limits.json_bytes) {state.capacity();return nullptr;}
    auto* text=state.arena->allocate_array<char>(static_cast<size_t>(length)+1u);if(!text) {state.capacity();return nullptr;}
    constexpr char digits[]="0123456789abcdef";uint32_t at=0;
    for(uint32_t i=0;i<size;++i) {
        if(uuid && (i==4 || i==6 || i==8 || i==10))text[at++]='-';
        text[at++]=digits[bytes[i]>>4];text[at++]=digits[bytes[i]&15u];
    }
    auto* out=j::make_string(state.arena,text,static_cast<int32_t>(at));if(!out)state.capacity();return out;
}
bool emit_geo(State& state,const BoltColumn& column,int64_t row,J** out) noexcept {
    assert(out);assert(row>=0 && row<column.length);
    const uint8_t* bytes=nullptr;uint32_t size=0;if(!column_bytes(column,row,&bytes,&size))return false;
    const auto limits=geo_limits(state);uint32_t needed=0;
    auto status=kernels::wkt::from_wkb(bytes,size,nullptr,0,&needed,limits);
    if(!geo_status(state,status))return false;
    if(needed>state.limits.json_bytes)return state.capacity();
    auto* text=state.arena->allocate_array<char>(static_cast<size_t>(needed)+1u);if(!text)return state.capacity();
    status=kernels::wkt::from_wkb(bytes,size,text,needed,&needed,limits);
    if(!geo_status(state,status))return false;
    text[needed]=0;*out=j::make_string(state.arena,text,static_cast<int32_t>(needed));return *out || state.capacity();
}
bool emit_scalar(State& state,const SchemaField& field,const BoltColumn& column,int64_t row,J** out) noexcept {
    assert(out);assert(row>=0 && row<column.length);
    IcebergTypeInfo type{};if(!iceberg_type_parse(field.type,&type))return state.unsupported();
    nr::TypeNode expected{};if(!leaf_node(field,expected))return state.unsupported();
    if(column.type!=expected.type && !(expected.type==BoltType::Float64 && column.type==BoltType::Float32))return false;
    using T=IcebergType;int64_t integer=0;char text[96]{};
    if(type.type==T::kBoolean || type.type==T::kInt || type.type==T::kLong) {
        if(!column_integer(column,row,&integer) || (type.type==T::kInt && (integer<INT32_MIN || integer>INT32_MAX)))return false;
        if(type.type==T::kBoolean) {if(integer!=0 && integer!=1)return false;*out=j::make_bool(state.arena,integer!=0);}
        else *out=j::make_int(state.arena,integer);
    } else if(type.type==T::kFloat || type.type==T::kDouble) {
        if(column.format!=ColumnFormat::Flat || !column.data)return false;double value=0;
        if(column.type==BoltType::Float64 && column.type_size_bytes==8)std::memcpy(&value,static_cast<const uint8_t*>(column.data)+row*8,8);
        else if(column.type==BoltType::Float32 && column.type_size_bytes==4) {float v=0;std::memcpy(&v,static_cast<const uint8_t*>(column.data)+row*4,4);value=v;}
        else return false;
        if(!std::isfinite(value) || (type.type==T::kFloat && std::fabs(value)>std::numeric_limits<float>::max()))return false;
        if(type.type==T::kFloat)value=static_cast<float>(value);*out=j::make_double(state.arena,value);
    } else if(type.type==T::kDecimal) {
        dec::Decimal128 value{};
        if(column.format!=ColumnFormat::Flat || !column.data || column.decimal_scale!=type.scale)return false;
        if(column.type==BoltType::Decimal64 && column.type_size_bytes==8) {std::memcpy(&value.lo,static_cast<const uint8_t*>(column.data)+row*8,8);value.hi=value.lo<0?-1:0;}
        else if(column.type==BoltType::Decimal128 && column.type_size_bytes==16)std::memcpy(&value,static_cast<const uint8_t*>(column.data)+row*16,16);
        else return false;
        if(!dec::d128_column_fits_precision(&value,1,nullptr,0,type.precision))return false;
        const auto size=decimal_format(value,type.scale,text,sizeof(text));if(!size)return false;*out=text_value(state,text,size);
    } else if(type.type==T::kGeometry || type.type==T::kGeography)return emit_geo(state,column,row,out);
    else if(type.type==T::kString || type.type==T::kBinary || type.type==T::kFixed || type.type==T::kUuid) {
        const uint8_t* bytes=nullptr;uint32_t size=0;if(!column_bytes(column,row,&bytes,&size))return false;
        if(type.type==T::kString) {if(column.type!=BoltType::Utf8)return false;*out=j::make_string(state.arena,reinterpret_cast<const char*>(bytes),static_cast<int32_t>(size));}
        else {
            if(type.type==T::kUuid && column.type==BoltType::Utf8 && !hex_bytes(state,bytes,size,true,&bytes,&size))return false;
            if((type.type==T::kUuid && size!=16) || (type.type==T::kFixed && size!=type.fixed_len))return false;
            *out=encoded_hex(state,bytes,size,type.type==T::kUuid);
        }
    } else {
        if(!column_integer(column,row,&integer))return false;
        const auto size=temporal_format(type.type,integer,text,sizeof(text));if(!size)return false;*out=text_value(state,text,size);
    }
    return *out!=nullptr || state.capacity();
}
bool emit_value(State&,const SchemaField&,const BoltColumn&,int64_t,uint32_t,J**) noexcept;
bool emit_struct(State& state,const SchemaField& field,const BoltColumn& column,int64_t row,uint32_t depth,J** out) noexcept {
    assert(out);assert(column.type==BoltType::Struct);
    if(column.format!=ColumnFormat::Nested || column.child_count()!=field.n_children)return false;
    *out=j::make_object(state.arena);if(!*out)return state.capacity();
    for(uint32_t i=0;i<field.n_children;++i) {
        const auto* child=column.child_at(i);if(!child || child->length!=column.length)return false;
        J* value=nullptr;if(!emit_value(state,field.children[i],*child,row,depth+1,&value))return false;
        if(child->is_null(row))continue;
        char name[16];const auto size=std::snprintf(name,sizeof(name),"%d",field.children[i].id);
        if(size<1 || static_cast<size_t>(size)>=sizeof(name))return false;
        if(!j::object_add(state.arena,*out,name,size,value))return state.capacity();
    }
    return true;
}
bool emit_sequence(State& state,const SchemaField& field,const BoltColumn& column,int64_t row,uint32_t depth,J** out) noexcept {
    assert(out);assert(column.type==BoltType::List || column.type==BoltType::Map);
    const bool map=column.type==BoltType::Map;const auto* child=column.list_element();const auto* offsets=column.list_offsets();
    if(column.format!=ColumnFormat::Nested || !child || !offsets || column.child_count()!=1)return false;
    const auto begin=offsets[row],end=offsets[row+1];
    if(begin<0 || end<begin || end>child->length)return false;
    if(static_cast<uint32_t>(end-begin)>state.limits.values-state.values)return state.capacity();
    J* values=j::make_array(state.arena),*keys=map?j::make_array(state.arena):nullptr;
    if(!values || (map && !keys))return state.capacity();
    if(map && (child->type!=BoltType::Struct || child->child_count()!=2))return false;
    for(int32_t i=begin;i<end;++i) {
        J* value=nullptr;
        if(map) {
            if(child->is_null(i) || !child->child_at(0) || !child->child_at(1))return false;
            if(!state.count() || !emit_value(state,field.children[0],*child->child_at(0),i,depth+2,&value))return false;
            if(!j::array_add(state.arena,keys,value))return state.capacity();
        }
        const auto* source=map?child->child_at(1):child;
        if(!source || !emit_value(state,field.children[map?1:0],*source,i,depth+(map?2u:1u),&value))return false;
        if(!j::array_add(state.arena,values,value))return state.capacity();
    }
    if(!map) {*out=values;return true;}
    *out=j::make_object(state.arena);
    if(!*out || !j::object_add(state.arena,*out,"keys",4,keys) || !j::object_add(state.arena,*out,"values",6,values))return state.capacity();
    return true;
}
bool emit_value(State& state,const SchemaField& field,const BoltColumn& column,int64_t row,uint32_t depth,J** out) noexcept {
    assert(out);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !state.count())return state.capacity();
    if(row<0 || row>=column.length)return false;
    if(column.is_null(row)) {if(field.required)return false;*out=j::make_null(state.arena);return *out || state.capacity();}
    if(!std::strcmp(field.type,"struct"))return column.type==BoltType::Struct && emit_struct(state,field,column,row,depth,out);
    if(!std::strcmp(field.type,"list") || !std::strcmp(field.type,"map")) {
        if(column.type!=(!std::strcmp(field.type,"list")?BoltType::List:BoltType::Map))return false;
        return emit_sequence(state,field,column,row,depth,out);
    }
    return emit_scalar(state,field,column,row,out);
}
