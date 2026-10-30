bool rebind_shape(const SchemaField& source,const SchemaField& target,uint32_t depth) noexcept {
    assert(depth<=nr::kMaxDepth+1u);assert(nr::kMaxNodes>0);
    if(depth>nr::kMaxDepth || std::strcmp(source.type,target.type) || source.required!=target.required ||
        source.n_children!=target.n_children)return false;
    for(uint32_t i=0;i<source.n_children;++i)if(!rebind_shape(source.children[i],target.children[i],depth+1u))return false;
    return true;
}
bool rebind_value(State&,const SchemaField&,const SchemaField&,const J*,uint32_t,J**) noexcept;
bool rebind_struct(State& state,const SchemaField& source,const SchemaField& target,const J* value,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(source.n_children<=nr::kMaxNodes);
    const J* fields[nr::kMaxNodes]{};if(!object_fields(source,value,fields))return false;
    *out=j::make_object(state.arena);if(!*out)return state.capacity();
    for(uint32_t i=0;i<source.n_children;++i) {
        if(!fields[i])continue;J* child=nullptr;
        if(!rebind_value(state,source.children[i],target.children[i],fields[i],depth+1u,&child))return false;
        char name[16];const auto n=std::snprintf(name,sizeof(name),"%d",target.children[i].id);
        if(n<1 || static_cast<size_t>(n)>=sizeof(name))return false;
        if(!j::object_add(state.arena,*out,name,n,child))return state.capacity();
    }
    return true;
}
bool rebind_array(State& state,const SchemaField& source,const SchemaField& target,const J* value,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(depth<=nr::kMaxDepth+1u);
    if(value->kind!=j::ValueKind::Array || value->count<0)return false;
    *out=j::make_array(state.arena);if(!*out)return state.capacity();const auto* item=value->head;
    for(int32_t i=0;i<value->count;++i) {
        if(!item)return false;J* child=nullptr;
        if(!rebind_value(state,source,target,item->value,depth,&child))return false;
        if(!j::array_add(state.arena,*out,child))return state.capacity();item=item->next;
    }
    return !item;
}
bool rebind_value(State& state,const SchemaField& source,const SchemaField& target,const J* value,uint32_t depth,J** out) noexcept {
    assert(value && out);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !state.count())return state.capacity();
    if(std::strcmp(source.type,target.type) || source.n_children!=target.n_children || source.required!=target.required)return false;
    if(is_null(value)) {*out=const_cast<J*>(value);return true;}
    if(!std::strcmp(source.type,"struct"))return rebind_struct(state,source,target,value,depth,out);
    if(!std::strcmp(source.type,"list"))return rebind_array(state,source.children[0],target.children[0],value,depth+1u,out);
    if(!std::strcmp(source.type,"map")) {
        const J* keys=nullptr,*values=nullptr;if(!map_arrays(value,&keys,&values))return false;
        J* k=nullptr,*v=nullptr;
        if(!rebind_array(state,source.children[0],target.children[0],keys,depth+2u,&k) ||
            !rebind_array(state,source.children[1],target.children[1],values,depth+2u,&v))return false;
        *out=j::make_object(state.arena);
        if(!*out || !j::object_add(state.arena,*out,"keys",4,k) || !j::object_add(state.arena,*out,"values",6,v))return state.capacity();
        return true;
    }
    *out=const_cast<J*>(value);return true;
}
