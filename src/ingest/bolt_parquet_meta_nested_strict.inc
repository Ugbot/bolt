// Same raw-member validator as exact flat admission, with a retained tree.
bool strict_nested_schema_list(TcCursor* c,StrictSchema& s,uint32_t count) noexcept {
    assert(c && s.tree && s.leaf_nodes); assert(s.fields && s.max_columns);
    const auto& tree=*s.tree;
    if(count!=tree.n_nodes)return false;
    struct Frame {uint32_t node,left;uint8_t def,rep;};
    Frame parents[kPqMaxNestDepth+2]{};uint32_t depth=0;
    for(uint32_t i=0;i<count;++i) {
        SchemaElem element{};if(!strict_element(c,s,&element,i==0))return false;
        const auto& node=tree.nodes[i];
        if(std::strcmp(element.col.name,node.name) || element.col.has_field_id!=node.has_field_id ||
            (node.has_field_id && element.col.field_id!=node.field_id) ||
            element.num_children<0 || static_cast<uint32_t>(element.num_children)!=node.n_children)return false;
        uint8_t def=0,rep=0;
        if(i) {
            while(depth && !parents[depth-1].left)--depth;
            if(!depth || node.parent!=static_cast<int32_t>(parents[depth-1].node))return false;
            auto& parent=parents[depth-1];--parent.left;
            const uint8_t repetition=element.repeated?2u:element.col.optional?1u:0u;
            def=static_cast<uint8_t>(parent.def+(repetition!=0));
            rep=static_cast<uint8_t>(parent.rep+(repetition==2));
            if(node.repetition!=repetition || node.def!=def || node.rep!=rep)return false;
        } else if(node.parent!=-1 || element.has_type)return false;
        if(element.num_children) {
            if(depth==kPqMaxNestDepth+2)return strict_refuse(s,StrictStatus::Unsupported);
            parents[depth++]={i,static_cast<uint32_t>(element.num_children),def,rep};
        } else {
            if(!element.has_type || s.columns>=s.max_columns || s.columns>=tree.n_columns ||
                node.column!=static_cast<int32_t>(s.columns))return false;
            s.fields[s.columns]=element.col;s.leaf_nodes[s.columns++]=i;
        }
    }
    while(depth && !parents[depth-1].left)--depth;
    if(depth || s.columns!=tree.n_columns)return false;
    for(uint32_t i=1;i<count;++i)s.node_order[i-1u]=i;
    std::sort(s.node_order,s.node_order+count-1u,[&tree](uint32_t a,uint32_t b) noexcept {
        const auto& x=tree.nodes[a];const auto& y=tree.nodes[b];
        return x.parent!=y.parent ? x.parent<y.parent : std::strcmp(x.name,y.name)<0;
    });
    for(uint32_t i=1;i<count-1u;++i) {
        const auto& a=tree.nodes[s.node_order[i-1u]];const auto& b=tree.nodes[s.node_order[i]];
        if(a.parent==b.parent && !std::strcmp(a.name,b.name))return strict_refuse(s,StrictStatus::Unsupported);
    }
    return true;
}
bool strict_nested_chunk_path(TcCursor* c,const StrictSchema& s,uint32_t column) noexcept {
    assert(c && s.tree && s.leaf_nodes); assert(column<s.columns);
    const auto& tree=*s.tree;
    uint32_t path[kPqMaxNestDepth+2]{},depth=0;
    int32_t node=static_cast<int32_t>(s.leaf_nodes[column]);
    for(uint32_t step=0;node>0 && step<kPqMaxNestDepth+2;++step) {
        if(static_cast<uint32_t>(node)>=tree.n_nodes || depth==kPqMaxNestDepth+2)return false;
        path[depth++]=static_cast<uint32_t>(node);
        const auto parent=tree.nodes[node].parent;
        if(parent<0 || parent>=node)return false;
        node=parent;
    }
    uint8_t type=0;uint32_t count=0;
    if(node!=0 || !strict_list(c,&type,&count) || type!=kTcBinary || count!=depth)return false;
    for(uint32_t i=0;i<count;++i) {
        const uint8_t* bytes=nullptr;uint32_t length=0;
        const auto* expected=tree.nodes[path[depth-1u-i]].name;
        if(!strict_binary(c,&bytes,&length) || length!=std::strlen(expected) || std::memcmp(bytes,expected,length))return false;
    }
    return true;
}
