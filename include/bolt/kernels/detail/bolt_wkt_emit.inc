struct Emit {
    wkb::detail::Cursor& in;Sink& out;uint32_t values=0;
    bool count() noexcept {if(values==out.limits.values) {out.status=Status::Capacity;return false;}++values;return true;}
    bool coordinate(uint32_t group) noexcept {
        assert(group<=3);assert(in.at<=in.n);
        for(uint32_t i=0;i<wkb::detail::dims_of(group);++i) {
            if(in.remaining()<8 || !count())return false;
            const double value=wkb::detail::f64(&in);if(!std::isfinite(value))return false;
            char text[32];const auto result=std::to_chars(text,text+sizeof(text),value,std::chars_format::general,std::numeric_limits<double>::max_digits10);
            if(result.ec!=std::errc{})return false;
            if(i && !out.text(" "))return false;
            if(!out.put(text,static_cast<uint32_t>(result.ptr-text)))return false;
        }
        return true;
    }
    bool point(uint32_t group) noexcept {
        assert(group<=3);assert(in.remaining()>=wkb::detail::dims_of(group)*8u);
        auto look=in;bool empty=true;
        for(uint32_t i=0;i<wkb::detail::dims_of(group);++i)empty=std::isnan(wkb::detail::f64(&look)) && empty;
        if(empty) {
            for(uint32_t i=0;i<wkb::detail::dims_of(group);++i)if(!count())return false;
            in=look;return out.text("EMPTY");
        }
        return out.text("(") && coordinate(group) && out.text(")");
    }
    bool sequence(uint32_t group) noexcept {
        assert(group<=3);assert(in.at<=in.n);
        uint32_t n=0;if(!wkb::detail::u32(&in,&n))return false;
        if(!n)return out.text("EMPTY");
        if(n==1)return false;
        if(n>out.limits.values-values) {out.status=Status::Capacity;return false;}
        if(!out.text("("))return false;
        for(uint32_t i=0;i<n;++i)if((i && !out.text(", ")) || !coordinate(group))return false;
        return out.text(")");
    }
    bool polygon(uint32_t group) noexcept {
        assert(group<=3);assert(in.at<=in.n);
        uint32_t n=0;if(!wkb::detail::u32(&in,&n))return false;
        if(!n)return out.text("EMPTY");
        if(n>out.limits.values-values) {out.status=Status::Capacity;return false;}
        if(!out.text("("))return false;
        for(uint32_t i=0;i<n;++i)if(!count() || (i && !out.text(", ")) || !sequence(group))return false;
        return out.text(")");
    }
    bool geometry(bool tagged,uint32_t depth) noexcept {
        assert(depth<=wkb::k_max_depth+1u);assert(in.at<=in.n);
        if(depth>out.limits.depth || !count()) {out.status=Status::Capacity;return false;}
        if(in.remaining()<5)return false;in.le=in.p[in.at++]!=0;
        uint32_t type=0;if(!wkb::detail::u32(&in,&type))return false;
        const auto base=type%1000u,group=type/1000u;if(base<1 || base>7 || group>3)return false;
        if(tagged) {
            if(!out.text(names[base]))return false;
            if(group && !out.text(group==1?" Z":group==2?" M":" ZM"))return false;
            if(!out.text(" "))return false;
        }
        if(base==1)return point(group);
        if(base==2)return sequence(group);
        if(base==3)return polygon(group);
        uint32_t n=0;if(!wkb::detail::u32(&in,&n))return false;
        if(!n)return out.text("EMPTY");
        if(n>out.limits.values-values) {out.status=Status::Capacity;return false;}
        if(!out.text("("))return false;
        for(uint32_t i=0;i<n;++i)if((i && !out.text(", ")) || !geometry(base==7,depth+1u))return false;
        return out.text(")");
    }
};
