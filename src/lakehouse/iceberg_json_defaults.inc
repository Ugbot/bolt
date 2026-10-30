struct DefaultState {
    State& state;DefaultValueMode mode;uint32_t parsed=0;
};
bool default_json(DefaultState& ctx,const SchemaField& field,J** out) noexcept {
    assert(out);assert(ctx.state.arena);
    const auto value=ctx.mode==DefaultValueMode::Initial?field.initial_default_value:field.write_default_value;
    if((value.json==nullptr)!=(value.size==0))return false;
    if(!value.json) {*out=j::make_null(ctx.state.arena);return *out || ctx.state.capacity();}
    if(value.size>ctx.state.limits.json_bytes-ctx.parsed)return ctx.state.capacity();
    ctx.parsed+=value.size;
    const j::SaxHandler syntax{};j::StructuralIndex index{};
    if(!j::sax_parse(value.json,static_cast<int32_t>(value.size),&syntax))return false;
    if(!j::build_index(value.json,static_cast<int32_t>(value.size),ctx.state.arena,&index) ||
        !j::value_from_index(&index,ctx.state.arena,out))return ctx.state.capacity();
    return true;
}
bool effective_value(DefaultState&,const SchemaField&,const J*,bool,uint32_t,J**) noexcept;
bool effective_struct(DefaultState& ctx,const SchemaField& field,const J* value,bool stored,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(field.n_children<=nr::kMaxNodes);
    const J* fields[nr::kMaxNodes]{};
    if(!object_fields(field,value,fields) || (stored && value->count!=0))return false;
    *out=j::make_object(ctx.state.arena);if(!*out)return ctx.state.capacity();
    for(uint32_t i=0;i<field.n_children;++i) {
        J* child=nullptr;
        if(!effective_value(ctx,field.children[i],fields[i],fields[i]==nullptr,depth+1u,&child))return false;
        char name[16];const auto size=std::snprintf(name,sizeof(name),"%d",field.children[i].id);
        if(size<1 || static_cast<size_t>(size)>=sizeof(name))return false;
        if(!j::object_add(ctx.state.arena,*out,name,size,child))return ctx.state.capacity();
    }
    return true;
}
bool effective_list(DefaultState& ctx,const SchemaField& field,const J* value,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(field.n_children==1);
    if(value->kind!=j::ValueKind::Array || value->count<0)return false;
    *out=j::make_array(ctx.state.arena);if(!*out)return ctx.state.capacity();const auto* node=value->head;
    for(int32_t i=0;i<value->count;++i) {
        if(!node)return false;J* child=nullptr;
        if(!effective_value(ctx,field.children[0],node->value,false,depth+1u,&child))return false;
        if(!j::array_add(ctx.state.arena,*out,child))return ctx.state.capacity();node=node->next;
    }
    return !node;
}
bool effective_map(DefaultState& ctx,const SchemaField& field,const J* value,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(field.n_children==2);
    const J* keys=nullptr,*values=nullptr;if(!map_arrays(value,&keys,&values))return false;
    auto* key_array=j::make_array(ctx.state.arena);auto* value_array=j::make_array(ctx.state.arena);
    if(!key_array || !value_array)return ctx.state.capacity();const auto* key=keys->head;const auto* val=values->head;
    for(int32_t i=0;i<keys->count;++i) {
        if(!key || !val)return false;J* k=nullptr,*v=nullptr;
        if(!effective_value(ctx,field.children[0],key->value,false,depth+2u,&k) ||
            !effective_value(ctx,field.children[1],val->value,false,depth+2u,&v))return false;
        if(!j::array_add(ctx.state.arena,key_array,k) || !j::array_add(ctx.state.arena,value_array,v))return ctx.state.capacity();
        key=key->next;val=val->next;
    }
    if(key || val)return false;
    *out=j::make_object(ctx.state.arena);
    if(!*out || !j::object_add(ctx.state.arena,*out,"keys",4,key_array) ||
        !j::object_add(ctx.state.arena,*out,"values",6,value_array))return ctx.state.capacity();
    return true;
}
bool effective_value(DefaultState& ctx,const SchemaField& field,const J* value,bool stored,uint32_t depth,J** out) noexcept {
    assert(out);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !ctx.state.count())return ctx.state.capacity();
    J* selected=nullptr;if(stored) {if(!default_json(ctx,field,&selected))return false;value=selected;}
    if(!value)return false;
    if(is_null(value)) {*out=const_cast<J*>(value);return true;}
    IcebergTypeInfo type{};
    if(!std::strcmp(field.type,"unknown") || !std::strcmp(field.type,"variant") ||
        (iceberg_type_parse(field.type,&type) && (type.type==IcebergType::kGeometry || type.type==IcebergType::kGeography)))return false;
    if(!std::strcmp(field.type,"struct"))return effective_struct(ctx,field,value,stored,depth,out);
    if(!std::strcmp(field.type,"list"))return effective_list(ctx,field,value,depth,out);
    if(!std::strcmp(field.type,"map"))return effective_map(ctx,field,value,depth,out);
    *out=const_cast<J*>(value);return true;
}
bool validate_default_fields(State& state,const SchemaField& field,uint32_t depth) noexcept {
    assert(state.arena);assert(depth<=kIcebergSchemaMaxDepth+1u);
    if(depth>kIcebergSchemaMaxDepth)return state.capacity();
    constexpr DefaultValueMode modes[]{DefaultValueMode::Initial,DefaultValueMode::Write};
    for(const auto mode:modes) {
        const auto value=mode==DefaultValueMode::Initial?field.initial_default_value:field.write_default_value;
        if(!value.json)continue;
        DefaultState check{state,mode};J* raw=nullptr;
        if(!default_json(check,field,&raw))return false;
        if(is_null(raw))continue; // A null declaration is unset; consumption enforces requiredness.
        BoltColumn decoded{};const auto result=default_value_decode(field,mode,state.arena,&decoded,state.limits);
        if(result!=S::Ok) {state.status=result;return false;}
    }
    for(uint32_t i=0;i<field.n_children;++i)if(!validate_default_fields(state,field.children[i],depth+1u))return false;
    return true;
}
