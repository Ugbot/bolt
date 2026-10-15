#pragma once
#include "bolt/ingest/bolt_arrow_ipc_reader.h"
#include "bolt/bolt_column.h"
#include "bolt/wire/byte_reader.h"
#include "bolt/kernels/bolt_utf8_cp.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <new>
namespace bolt::ingest::ipc_read_detail {
using Status = IpcReadStatus;
inline bool range(std::uint64_t p, std::uint64_t n, std::uint64_t cap) noexcept {
    return p <= cap && n <= cap-p;
}
inline bool overlap(const void* a, std::uint64_t an, const void* b, std::uint64_t bn) noexcept {
    if (!a || !b || !an || !bn) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y-x < an : x-y < bn;
}
struct Fb {
    const std::uint8_t* p = nullptr; std::uint32_t n = 0;
    template<class T> bool get(std::uint32_t at, T& v) const noexcept {
        wire::ByteReader r{p,n,at}; return wire::read_le(r,v);
    }
    bool root(std::uint32_t& t) const noexcept {
        return get(0,t) && t >= 4 && t%4==0 && range(t,4,n);
    }
};
struct Table {
    Fb fb{}; std::uint32_t at=0, vt=0; std::uint16_t vs=0, os=0;
    bool init(Fb f, std::uint32_t a) noexcept {
        std::int32_t d=0; if (a%4 || !f.get(a,d) || d==0) return false;
        const auto v=static_cast<std::int64_t>(a)-d;
        if (v<0 || v>UINT32_MAX || v%2) return false;
        fb=f; at=a; vt=static_cast<std::uint32_t>(v);
        return fb.get(vt,vs) && fb.get(vt+2,os) && vs>=4 && !(vs&1) && os>=4 &&
            range(vt,vs,fb.n) && range(at,os,fb.n);
    }
    bool field(std::uint32_t slot, std::uint32_t width, std::uint32_t& pos) const noexcept {
        pos=0; const auto index=4ull+2ull*slot;
        if (index>=vs) return true;
        std::uint16_t off=0; if (!fb.get(vt+static_cast<std::uint32_t>(index),off)) return false;
        if (!off) return true;
        if (off<4 || !range(off,width,os) || (at+off)%width) return false;
        pos=at+off; return true;
    }
    template<class T> bool scalar(std::uint32_t slot, T def, T& v) const noexcept {
        std::uint32_t pos=0; if (!field(slot,sizeof(T),pos)) return false;
        v=def; return pos==0 || fb.get(pos,v);
    }
    bool ref(std::uint32_t slot, std::uint32_t& target) const noexcept {
        std::uint32_t pos=0,d=0; if (!field(slot,4,pos)) return false;
        target=0; if (!pos) return true;
        if (!fb.get(pos,d) || d<4 || !range(pos,d,fb.n) || !range(pos+d,4,fb.n)) return false;
        target=pos+d; return target%4==0;
    }
    bool vector(std::uint32_t slot, std::uint32_t stride, std::uint32_t& start,
                std::uint32_t& count) const noexcept {
        std::uint32_t p=0; if (!ref(slot,p)) return false;
        start=0;count=0; if (!p) return true;
        if (!fb.get(p,count) || !range(p+4,static_cast<std::uint64_t>(count)*stride,fb.n)) return false;
        // Empty vectors have no element payload to align. Reference Arrow
        // writers may place the length word at only four-byte alignment.
        start=p+4; return count==0 || start%(stride<8?stride:8)==0;
    }
    bool string(std::uint32_t slot, IpcReadString& out) const noexcept {
        std::uint32_t p=0,n=0; if (!ref(slot,p)) return false;
        out={}; if (!p) return true;
        if (!fb.get(p,n) || !range(p+4,static_cast<std::uint64_t>(n)+1,fb.n) || fb.p[p+4+n]) return false;
        const auto* data=reinterpret_cast<const char*>(fb.p+p+4);
        for (std::uint32_t i=0;i<n;) {
            std::uint32_t cp=0;
            i+=kernels::utf8::utf8_cp_decode(data,n,i,&cp);
            if (cp==kernels::utf8::kUtf8Invalid) return false;
        }
        out={data,n}; return true;
    }
};
inline bool indirect(Fb f, std::uint32_t p, std::uint32_t& out) noexcept {
    std::uint32_t d=0; if (!f.get(p,d) || d<4 || !range(p,d,f.n) || !range(p+d,4,f.n)) return false;
    out=p+d;return out%4==0;
}
struct Message { Fb fb{}; Table header{}; std::uint64_t body=0, next=0; std::uint32_t metadata=0;
    std::int16_t version=0; std::uint8_t kind=0; bool eos=false; };
struct Layout { const std::uint8_t *validity=nullptr,*values=nullptr,*offsets=nullptr;
    std::uint64_t value_bytes=0; std::int64_t nulls=0; };
struct State {
    IpcReadLimits limits{}; const std::uint8_t* input=nullptr;
    std::uint64_t input_bytes=0,stream_end=0,cursor=0,workspace_bytes=0,workspace_capacity=0;
    std::uint32_t file_blocks=0,block_count=0;std::uint64_t batch_index=0;
    Fb footer{}; bool file=false;
    std::int16_t version=0;
    IpcReadSchema schema{};
    IpcReadField *fields=nullptr,*compare_fields=nullptr;
    IpcReadKeyValue *kv=nullptr,*compare_kv=nullptr;
    Layout* layout=nullptr; BoltField* bolt_fields=nullptr;
    BoltColumn *cols=nullptr,*other_cols=nullptr,*offset_cols=nullptr;
    std::uint8_t* output=nullptr;
    BoltBatch batch{};
};
Status message(const State&, std::uint64_t offset, Message&) noexcept;
Status parse_schema(State&, Table, bool compare) noexcept;
Status key_values(Table,std::uint32_t,IpcReadKeyValue*,std::uint32_t,std::uint32_t&,
                  const IpcReadKeyValue*&,std::uint32_t&) noexcept;
Status open_input(State&) noexcept;
Status preflight(State&,const Message&,std::int64_t&,std::uint64_t&) noexcept;
void publish(State&,std::int64_t rows) noexcept;
} // namespace bolt::ingest::ipc_read_detail
