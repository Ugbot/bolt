#pragma once
// ISO simple-feature WKT <-> WKB, a wire-boundary codec. Seven base types,
// XY/XYZ/XYM/XYZM, bounded nesting/coordinates/bytes, no heap or exceptions.
// A null destination measures output. Scratch may be written on failure;
// *size publishes only on success. WKB validation shares bolt_wkb.h.
#include "bolt/kernels/bolt_wkb.h"
#include <bit>
#include <charconv>
#include <limits>

namespace bolt::kernels::wkt {
enum class Status : uint8_t { Ok, Invalid, Capacity };
struct Limits {uint32_t bytes=16u<<20;uint32_t values=1u<<20;uint32_t depth=wkb::k_max_depth;};
namespace detail {
struct Sink {
    uint8_t* bytes;uint32_t cap;Limits limits;uint32_t n=0;Status status=Status::Invalid;
    bool put(const void* data,uint32_t size) noexcept {
        assert(data || size==0);assert(n<=limits.bytes);
        if(size>limits.bytes-n || (bytes && size>cap-n)) {status=Status::Capacity;return false;}
        if(bytes && size)std::memcpy(bytes+n,data,size);n+=size;return true;
    }
    bool text(const char* value) noexcept {return put(value,static_cast<uint32_t>(std::strlen(value)));}
    bool u32(uint32_t value) noexcept {
        uint8_t data[4];for(uint32_t i=0;i<4;++i)data[i]=static_cast<uint8_t>(value>>(8u*i));return put(data,4);
    }
    bool patch(uint32_t at,uint32_t value) noexcept {
        assert(at<=n);assert(n-at>=4);
        if(bytes)for(uint32_t i=0;i<4;++i)bytes[at+i]=static_cast<uint8_t>(value>>(8u*i));return true;
    }
    bool number(double value) noexcept {
        assert(sizeof(value)==8);assert(sizeof(uint64_t)==8);
        const auto bits=std::bit_cast<uint64_t>(value);uint8_t data[8];
        for(uint32_t i=0;i<8;++i)data[i]=static_cast<uint8_t>(bits>>(8u*i));return put(data,8);
    }
};
inline constexpr const char* names[]{"", "POINT","LINESTRING","POLYGON","MULTIPOINT","MULTILINESTRING","MULTIPOLYGON","GEOMETRYCOLLECTION"};
inline bool space(char c) noexcept {return c==' ' || c=='\t' || c=='\r' || c=='\n';}
inline bool valid_limits(const Limits& v) noexcept {
    return v.bytes>0 && v.bytes<=INT32_MAX && v.values>0 && v.values<=INT32_MAX && v.depth<=wkb::k_max_depth;
}
struct Text {
    const char* bytes;uint32_t n;uint32_t at=0;
    void skip() noexcept {while(at<n && space(bytes[at]))++at;}
    bool take(char c) noexcept {skip();if(at==n || bytes[at]!=c)return false;++at;return true;}
    bool word(const char* expected) noexcept {
        skip();const auto start=at;const auto size=static_cast<uint32_t>(std::strlen(expected));
        if(size>n-at)return false;
        for(uint32_t i=0;i<size;++i) {char c=bytes[at+i];if(c>='a' && c<='z')c=static_cast<char>(c-'a'+'A');if(c!=expected[i])return false;}
        const auto end=at+size;if(end<n && ((bytes[end]>='A' && bytes[end]<='Z') || (bytes[end]>='a' && bytes[end]<='z')))return false;
        at=start+size;return true;
    }
};
#include "bolt/kernels/detail/bolt_wkt_parse.inc"
#include "bolt/kernels/detail/bolt_wkt_emit.inc"
} // namespace detail
inline Status to_wkb(const char* text,uint32_t length,uint8_t* buffer,uint32_t cap,uint32_t* size,const Limits& limits={}) noexcept {
    assert(size);assert(!buffer || cap>0);
    if(!text || !length || !detail::valid_limits(limits))return Status::Invalid;
    if(length>limits.bytes)return Status::Capacity;
    detail::Text input{text,length};detail::Sink output{buffer,cap,limits};detail::Parse parser{input,output};
    if(!parser.geometry(-1,0))return output.status;
    input.skip();if(input.at!=input.n)return Status::Invalid;
    if(buffer) {wkb::Bounds bounds{};if(!wkb::bounds_add(&bounds,buffer,output.n))return Status::Invalid;}
    *size=output.n;return Status::Ok;
}
inline Status from_wkb(const uint8_t* bytes,uint32_t length,char* buffer,uint32_t cap,uint32_t* size,const Limits& limits={}) noexcept {
    assert(size);assert(!buffer || cap>0);
    if(!bytes || !detail::valid_limits(limits))return Status::Invalid;
    if(length>limits.bytes)return Status::Capacity;
    wkb::Bounds bounds{};if(!wkb::bounds_add(&bounds,bytes,length))return Status::Invalid;
    wkb::detail::Cursor input{bytes,length,0,true};detail::Sink output{reinterpret_cast<uint8_t*>(buffer),cap,limits};
    detail::Emit emitter{input,output};if(!emitter.geometry(true,0))return output.status;
    if(input.remaining())return Status::Invalid;*size=output.n;return Status::Ok;
}
} // namespace bolt::kernels::wkt
