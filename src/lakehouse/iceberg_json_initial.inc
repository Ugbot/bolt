// Compare values projected onto surviving stable IDs. New children acquire their
// own initial defaults; removed children cannot change a surviving child's value.
int32_t initial_child(const SchemaField& field,int32_t id) noexcept {
    assert(field.n_children<=nr::kMaxNodes);assert(id>0);
    for(uint32_t i=0;i<field.n_children;++i)if(field.children[i].id==id)return static_cast<int32_t>(i);
    return -1;
}
bool initial_leaf(State& state,const SchemaField& source,const SchemaField& target,
    const BoltColumn& column,int64_t row,bool newer,J** out) noexcept {
    assert(out);assert(row>=0 && row<column.length);
    auto value=column;int64_t integer=0;dec::Decimal128 decimal{};
    if(!newer && std::strcmp(source.type,target.type)) {
        IcebergTypeInfo from{},to{};
        if(!iceberg_type_parse(source.type,&from) || !iceberg_type_parse(target.type,&to))return false;
        using T=IcebergType;
        if(from.type==T::kInt && to.type==T::kLong) {
            if(!column_integer(column,row,&integer))return false;
            value.type=BoltType::Int64;value.type_size_bytes=8;value.data=&integer;row=0;
        } else if(from.type==T::kFloat && to.type==T::kDouble) {
            // The source decoder already rounded to float32, held in Float64.
        } else if(from.type==T::kDecimal && to.type==T::kDecimal && from.scale==to.scale && from.precision<=to.precision) {
            if(column.type==BoltType::Decimal64 && to.precision>18) {
                if(column.format!=ColumnFormat::Flat || !column.data)return false;
                std::memcpy(&decimal.lo,static_cast<const uint8_t*>(column.data)+row*8,8);decimal.hi=decimal.lo<0?-1:0;
                value.type=BoltType::Decimal128;value.type_size_bytes=16;value.data=&decimal;row=0;
            }
        } else return false;
    }
    return emit_scalar(state,target,value,row,out);
}
bool initial_value(State&,const SchemaField&,const SchemaField&,const BoltColumn&,int64_t,bool,uint32_t,J**) noexcept;
bool initial_struct(State& state,const SchemaField& source,const SchemaField& target,
    const BoltColumn& column,int64_t row,bool newer,uint32_t depth,J** out) noexcept {
    assert(out);assert(source.n_children<=nr::kMaxNodes);
    if(column.type!=BoltType::Struct || column.format!=ColumnFormat::Nested ||
        column.child_count()!=(newer?target.n_children:source.n_children))return false;
    *out=j::make_object(state.arena);if(!*out)return state.capacity();
    for(uint32_t i=0;i<source.n_children;++i) {
        const auto at=initial_child(target,source.children[i].id);if(at<0)continue;
        const auto* child=column.child_at(newer?static_cast<uint32_t>(at):i);J* value=nullptr;
        if(!child || !initial_value(state,source.children[i],target.children[at],*child,row,newer,depth+1u,&value))return false;
        char name[16];const auto n=std::snprintf(name,sizeof(name),"%d",source.children[i].id);
        if(n<1 || static_cast<size_t>(n)>=sizeof(name))return false;
        if(!j::object_add(state.arena,*out,name,n,value))return state.capacity();
    }
    return true;
}
bool initial_key_shape(const SchemaField& source,const SchemaField& target,uint32_t depth) noexcept {
    assert(depth<=nr::kMaxDepth+1u);assert(source.n_children<=nr::kMaxNodes && target.n_children<=nr::kMaxNodes);
    if(depth>nr::kMaxDepth || source.id!=target.id || source.n_children!=target.n_children)return false;
    for(uint32_t i=0;i<source.n_children;++i) {
        const auto at=initial_child(target,source.children[i].id);
        if(at<0 || !initial_key_shape(source.children[i],target.children[at],depth+1u))return false;
    }
    return true;
}
struct InitialEntry {const uint8_t* key=nullptr;int32_t size=0;J* key_value=nullptr;J* value=nullptr;};
bool initial_entry_less(const InitialEntry& a,const InitialEntry& b) noexcept {
    assert(a.size>=0 && b.size>=0);assert(a.key && b.key);
    const auto shared=std::min(a.size,b.size);const auto cmp=std::memcmp(a.key,b.key,static_cast<size_t>(shared));
    return cmp<0 || (cmp==0 && a.size<b.size);
}
bool initial_map(State& state,const SchemaField& source,const SchemaField& target,
    const BoltColumn& element,int32_t begin,uint32_t count,bool newer,uint32_t depth,J** out) noexcept {
    assert(out);assert(source.n_children==2 && target.n_children==2);
    if(element.type!=BoltType::Struct || element.child_count()!=2 || !initial_key_shape(source.children[0],target.children[0],depth))return false;
    auto* entries=state.arena->allocate_array<InitialEntry>(count+1u);if(!entries)return state.capacity();
    for(uint32_t i=0;i<count;++i) {
        entries[i]={};const auto row=static_cast<int64_t>(begin)+i;
        if(element.is_null(row) || !element.child_at(0) || !element.child_at(1))return false;
        if(!initial_value(state,source.children[0],target.children[0],*element.child_at(0),row,newer,depth+2u,&entries[i].key_value) ||
            !initial_value(state,source.children[1],target.children[1],*element.child_at(1),row,newer,depth+2u,&entries[i].value))return false;
        if(!j::serialize(entries[i].key_value,state.arena,j::EmitOptions{},&entries[i].key,&entries[i].size))return state.capacity();
    }
    std::sort(entries,entries+count,initial_entry_less);
    auto* keys=j::make_array(state.arena);auto* values=j::make_array(state.arena);
    if(!keys || !values)return state.capacity();
    for(uint32_t i=0;i<count;++i) {
        if(!j::array_add(state.arena,keys,entries[i].key_value) || !j::array_add(state.arena,values,entries[i].value))return state.capacity();
    }
    *out=j::make_object(state.arena);
    return (*out && j::object_add(state.arena,*out,"keys",4,keys) && j::object_add(state.arena,*out,"values",6,values)) || state.capacity();
}
bool initial_sequence(State& state,const SchemaField& source,const SchemaField& target,
    const BoltColumn& column,int64_t row,bool newer,uint32_t depth,J** out) noexcept {
    assert(out);assert(source.n_children>0 && target.n_children>0);
    const bool map=!std::strcmp(source.type,"map");const auto* offsets=column.list_offsets();const auto* element=column.list_element();
    if(column.format!=ColumnFormat::Nested || !offsets || !element)return false;
    const auto begin=offsets[row],end=offsets[row+1];if(begin<0 || end<begin || end>element->length)return false;
    const auto count=static_cast<uint32_t>(end-begin);if(count>state.limits.values-state.values)return state.capacity();
    if(map)return initial_map(state,source,target,*element,begin,count,newer,depth,out);
    *out=j::make_array(state.arena);if(!*out)return state.capacity();
    for(uint32_t i=0;i<count;++i) {
        J* value=nullptr;
        if(!initial_value(state,source.children[0],target.children[0],*element,static_cast<int64_t>(begin)+i,newer,depth+1u,&value))return false;
        if(!j::array_add(state.arena,*out,value))return state.capacity();
    }
    return true;
}
bool initial_value(State& state,const SchemaField& source,const SchemaField& target,
    const BoltColumn& column,int64_t row,bool newer,uint32_t depth,J** out) noexcept {
    assert(out);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !state.count())return state.capacity();
    if(row<0 || row>=column.length)return false;
    if(column.is_null(row)) {*out=j::make_null(state.arena);return *out || state.capacity();}
    if(!std::strcmp(source.type,"struct")) {
        if(std::strcmp(target.type,"struct"))return false;
        return initial_struct(state,source,target,column,row,newer,depth,out);
    }
    if(!std::strcmp(source.type,"list") || !std::strcmp(source.type,"map")) {
        if(std::strcmp(source.type,target.type) || source.n_children!=target.n_children)return false;
        return initial_sequence(state,source,target,column,row,newer,depth,out);
    }
    return initial_leaf(state,source,target,column,row,newer,out);
}
bool initial_canonical(State& state,const SchemaField& source,const SchemaField& target,bool newer,
    const uint8_t** out,int32_t* size) noexcept {
    assert(out && size);assert(state.arena);
    auto field=newer?target:source;field.required=false; // A null declaration is unset even on a required field.
    BoltColumn value{};const auto status=default_value_decode(field,DefaultValueMode::Initial,state.arena,&value,state.limits);
    if(status!=S::Ok) {state.status=status;return false;}
    J* normalized=nullptr;if(!initial_value(state,source,target,value,0,newer,0,&normalized))return false;
    if(!j::serialize(normalized,state.arena,j::EmitOptions{},out,size) || *size<0)return state.capacity();
    if(static_cast<uint32_t>(*size)>state.limits.json_bytes)return state.capacity();
    return true;
}
bool initial_fields(State& state,const SchemaField* source,uint32_t count,
    const SchemaField* target,uint32_t n_target,uint32_t depth) noexcept {
    assert(depth<=nr::kMaxDepth+1u);assert((source || count==0) && (target || n_target==0));
    if(depth>nr::kMaxDepth)return state.capacity();
    for(uint32_t i=0;i<count;++i) {
        for(uint32_t k=0;k<n_target;++k) {
            if(source[i].id!=target[k].id)continue;
            bool same=false;const auto status=initial_default_equal(source[i],target[k],state.arena,&same,state.limits);
            if(status!=S::Ok) {state.status=status;return false;}
            if(!same)return false;
            if(!initial_fields(state,source[i].children,source[i].n_children,target[k].children,target[k].n_children,depth+1u))return false;
            break;
        }
    }
    return true;
}
