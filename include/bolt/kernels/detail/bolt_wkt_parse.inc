struct Parse {
    Text& in;Sink& out;uint32_t values=0;
    bool count() noexcept {if(values==out.limits.values) {out.status=Status::Capacity;return false;}++values;return true;}
    bool header(uint32_t base,uint32_t group) noexcept {
        const uint8_t little=1;return out.put(&little,1) && out.u32(group*1000u+base);
    }
    bool ordinate(double* value) noexcept {
        assert(value);assert(in.at<=in.n);
        const auto previous=in.at;in.skip();if(in.at==in.n)return false;
        if(in.bytes[in.at]=='+')++in.at;
        const auto result=std::from_chars(in.bytes+in.at,in.bytes+in.n,*value,std::chars_format::general);
        if(result.ec!=std::errc{} || result.ptr==in.bytes+in.at || !std::isfinite(*value))return false;
        // A leading '+' cannot be followed by another sign.
        if(previous<in.at && in.bytes[in.at-1u]=='+' && (in.bytes[in.at]=='+' || in.bytes[in.at]=='-'))return false;
        in.at=static_cast<uint32_t>(result.ptr-in.bytes);return count();
    }
    bool coordinate(uint32_t group,double* x=nullptr,double* y=nullptr) noexcept {
        assert(group<=3);assert(out.n<=out.limits.bytes);
        const auto dims=wkb::detail::dims_of(group);
        for(uint32_t i=0;i<dims;++i) {
            if(i>0) {if(in.at==in.n || !space(in.bytes[in.at]))return false;}
            double value=0;if(!ordinate(&value) || !out.number(value))return false;
            if(i==0 && x)*x=value;if(i==1 && y)*y=value;
        }
        return true;
    }
    bool empty_point(uint32_t group) noexcept {
        assert(group<=3);assert(out.n<=out.limits.bytes);
        for(uint32_t i=0;i<wkb::detail::dims_of(group);++i)if(!count() || !out.number(std::numeric_limits<double>::quiet_NaN()))return false;
        return true;
    }
    bool point(uint32_t group) noexcept {
        if(in.word("EMPTY"))return empty_point(group);
        return in.take('(') && coordinate(group) && in.take(')');
    }
    bool sequence(uint32_t group,bool ring) noexcept {
        assert(group<=3);assert(out.n<=out.limits.bytes);
        const auto mark=out.n;if(!out.u32(0))return false;
        if(in.word("EMPTY"))return true;
        if(!in.take('('))return false;
        uint32_t n=0;double first_x=0,first_y=0,x=0,y=0;
        do {
            if(!coordinate(group,&x,&y))return false;
            if(n==0) {first_x=x;first_y=y;}++n;
        } while(n<=out.limits.values && in.take(','));
        if(!in.take(')') || (!ring && n<2) || (ring && (n<4 || x!=first_x || y!=first_y)))return false;
        return out.patch(mark,n);
    }
    bool polygon(uint32_t group) noexcept {
        assert(group<=3);assert(out.n<=out.limits.bytes);
        const auto mark=out.n;if(!out.u32(0))return false;
        if(in.word("EMPTY"))return true;
        if(!in.take('('))return false;uint32_t n=0;
        do {if(!count() || !sequence(group,true))return false;++n;}while(n<=out.limits.values && in.take(','));
        return in.take(')') && out.patch(mark,n);
    }
    bool collection(uint32_t base,uint32_t group,uint32_t depth) noexcept {
        assert(base>=4 && base<=7);assert(depth<=wkb::k_max_depth);
        const auto mark=out.n;if(!out.u32(0))return false;
        if(in.word("EMPTY"))return true;
        if(!in.take('('))return false;uint32_t n=0;
        do {
            if(base==7) {if(!geometry(static_cast<int32_t>(group),depth+1u))return false;}
            else {
                if(depth==out.limits.depth) {out.status=Status::Capacity;return false;}
                const auto child=base-3u;if(!count() || !header(child,group))return false;
                if(base==4) {
                    in.skip();const bool wrapped=in.at<in.n && in.bytes[in.at]=='(';
                    if(wrapped || in.word("EMPTY")) {
                        if(wrapped) {if(!point(group))return false;}
                        else if(!empty_point(group))return false;
                    } else if(!coordinate(group))return false;
                } else if(base==5) {if(!sequence(group,false))return false;}
                else if(!polygon(group))return false;
            }
            ++n;
        }while(n<=out.limits.values && in.take(','));
        return in.take(')') && out.patch(mark,n);
    }
    bool geometry(int32_t inherited,uint32_t depth) noexcept {
        assert(depth<=wkb::k_max_depth+1u);assert(inherited>=-1 && inherited<=3);
        if(depth>out.limits.depth || !count()) {out.status=Status::Capacity;return false;}
        uint32_t base=0;
        for(uint32_t i=1;i<=7;++i)if(in.word(names[i])) {base=i;break;}
        if(!base)return false;
        uint32_t group=inherited<0?0u:static_cast<uint32_t>(inherited);
        if(in.word("ZM"))group=3;else if(in.word("Z"))group=1;else if(in.word("M"))group=2;
        if(inherited>=0 && group!=static_cast<uint32_t>(inherited))return false;
        if(!header(base,group))return false;
        if(base==1)return point(group);
        if(base==2)return sequence(group,false);
        if(base==3)return polygon(group);
        return collection(base,group,depth);
    }
};
