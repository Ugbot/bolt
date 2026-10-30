struct RowWriter {
    State& state;uint8_t* bytes;uint32_t capacity;uint32_t n=0;
    bool put(const void* source,uint32_t count) noexcept {
        if(count>state.limits.encoded_bytes-n)return state.capacity();
        if(bytes) {if(count>capacity-n)return false;if(count)std::memcpy(bytes+n,source,count);}
        n+=count;return true;
    }
    bool byte(uint8_t value) noexcept {return put(&value,1);}
    bool count(uint32_t value) noexcept {return put(&value,4);}
};
bool key_name(const j::JsonNode& node,const char* name) noexcept {
    assert(name);assert(node.key && node.key_len>=0);
    char text[64]{};const auto length=static_cast<uint32_t>(node.key_len);
    if(length>6u*std::strlen(name) || length+2u>sizeof(text))return false;
    text[0]='"';std::memcpy(text+1,node.key,length);text[length+1]='"';char decoded[64]{};
    const auto n=kernels::json::json_span_unquote({text,length+2u},decoded,sizeof(decoded));
    return n>=0 && static_cast<size_t>(n)==std::strlen(name) && !std::memcmp(decoded,name,static_cast<size_t>(n));
}
bool object_fields(const SchemaField& field,const J* value,const J** fields) noexcept {
    assert(value && fields);assert(field.n_children<=nr::kMaxNodes);
    if(value->kind!=j::ValueKind::Object || value->count<0 || static_cast<uint32_t>(value->count)>field.n_children)return false;
    const auto* node=value->head;
    for(int32_t i=0;i<value->count;++i) {
        if(!node)return false;
        bool found=false;
        for(uint32_t child=0;child<field.n_children;++child) {
            char name[16];const auto n=std::snprintf(name,sizeof(name),"%d",field.children[child].id);
            if(n<1 || static_cast<size_t>(n)>=sizeof(name))return false;
            if(!key_name(*node,name))continue;
            if(fields[child])return false;
            fields[child]=node->value;found=true;break;
        }
        if(!found)return false;node=node->next;
    }
    return node==nullptr;
}
bool map_arrays(const J* value,const J** keys,const J** values) noexcept {
    assert(keys && values);assert(value);
    if(value->kind!=j::ValueKind::Object || value->count!=2)return false;
    const auto* node=value->head;
    for(uint32_t i=0;i<2;++i) {
        if(!node)return false;
        if(key_name(*node,"keys")) {if(*keys)return false;*keys=node->value;}
        else if(key_name(*node,"values")) {if(*values)return false;*values=node->value;}
        else return false;node=node->next;
    }
    return !node && *keys && *values && (*keys)->kind==j::ValueKind::Array && (*values)->kind==j::ValueKind::Array &&
        (*keys)->count>=0 && (*keys)->count==(*values)->count;
}
bool write_value(State&,const SchemaField&,const nr::TypeTree&,uint32_t,const J*,RowWriter&,uint32_t) noexcept;
bool write_struct(State& state,const SchemaField& field,const nr::TypeTree& tree,uint32_t at,
    const J* value,RowWriter& writer,uint32_t depth) noexcept {
    assert(at<tree.n);assert(field.n_children<=nr::kMaxNodes);
    const J* fields[nr::kMaxNodes]{};if(!object_fields(field,value,fields))return false;
    for(uint32_t i=0;i<field.n_children;++i)
        if(!write_value(state,field.children[i],tree,tree.nodes[at].first_child+i,fields[i],writer,depth+1))return false;
    return true;
}
struct KeySpan {uint32_t offset=0,size=0;int32_t next=-1;};
bool unique_key(RowWriter& writer,SwissTable& hashes,KeySpan* keys,uint32_t ordinal,uint32_t begin) noexcept {
    assert(writer.bytes && keys);assert(begin<=writer.n);
    const auto length=writer.n-begin;const auto hash=hash_bytes(writer.bytes+begin,length);
    const auto previous=hashes.find(static_cast<uint64_t>(hash));auto current=previous;
    for(uint32_t guard=0;current>=0 && guard<ordinal;++guard) {
        if(static_cast<uint32_t>(current)>=ordinal)return false;
        const auto& key=keys[current];
        if(key.size==length && !std::memcmp(writer.bytes+key.offset,writer.bytes+begin,length))return false;
        current=key.next;
    }
    if(current>=0)return false;
    keys[ordinal]={begin,length,previous};return hashes.insert(static_cast<uint64_t>(hash),ordinal);
}
bool write_sequence(State& state,const SchemaField& field,const nr::TypeTree& tree,uint32_t at,
    const J* value,RowWriter& writer,uint32_t depth) noexcept {
    assert(value);assert(at<tree.n);
    const bool map=tree.nodes[at].type==BoltType::Map;
    const J* keys=nullptr,*values=value;
    if(map) {values=nullptr;if(!map_arrays(value,&keys,&values))return false;}
    if(!values || values->kind!=j::ValueKind::Array || values->count<0)return false;
    const auto count=static_cast<uint32_t>(values->count);
    if(count>state.limits.values-state.values)return state.capacity();
    if(!writer.count(count))return false;
    SwissTable hashes{};KeySpan* spans=nullptr;
    if(map && writer.bytes && count) {
        spans=state.arena->allocate_array<KeySpan>(count);
        if(!spans || !SwissTable::create(&hashes,count,state.arena))return state.capacity();
    }
    const auto* key=keys?keys->head:nullptr;const auto* child=values->head;
    const auto first=tree.nodes[at].first_child;
    for(uint32_t i=0;i<count;++i) {
        if(!child)return false;
        if(map) {
            if(!key || !state.count() || !writer.byte(1))return false;
            const auto start=writer.n;
            if(!write_value(state,field.children[0],tree,tree.nodes[first].first_child,key->value,writer,depth+2))return false;
            if(writer.bytes && !unique_key(writer,hashes,spans,i,start))return false;
            key=key->next;
        }
        if(!write_value(state,field.children[map?1:0],tree,map?tree.nodes[first].first_child+1u:first,child->value,writer,depth+(map?2u:1u)))return false;
        child=child->next;
    }
    return !child && !key;
}
bool write_value(State& state,const SchemaField& field,const nr::TypeTree& tree,uint32_t at,
    const J* value,RowWriter& writer,uint32_t depth) noexcept {
    assert(at<tree.n);assert(depth<=nr::kMaxDepth+1u);
    if(depth>nr::kMaxDepth || !state.count())return state.capacity();
    const bool null=is_null(value);
    if(null && field.required)return false;
    if(!writer.byte(null?0u:1u))return false;
    if(null)return true;
    const auto& node=tree.nodes[at];
    if(node.type==BoltType::Struct)return write_struct(state,field,tree,at,value,writer,depth);
    if(node.type==BoltType::List || node.type==BoltType::Map)return write_sequence(state,field,tree,at,value,writer,depth);
    Scalar scalar{};if(!scalar_parse(state,field,value,scalar))return false;
    if(node.fixed_bytes)return writer.put(scalar.cell,node.fixed_bytes);
    return writer.count(scalar.size) && writer.put(scalar.bytes,scalar.size);
}
