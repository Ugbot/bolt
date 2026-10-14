#include "bolt/ingest/bolt_zstd_stream.h"
#include "bolt/ingest/bolt_zstd_dec.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <new>
namespace bolt::ingest {
namespace {
#include "bolt_zstd_dec_internal.h"
using Status = ZstdStreamStatus;
enum class Phase : std::uint8_t { Magic, Header, SkipHeader, Skip, BlockHeader, Body, Decode, Drain, Trailer, End, Done, Failed };
constexpr std::uint64_t kStreamMagic = 0x31535444535a5442ull;
struct Hash {
    std::uint64_t lanes[4]{kP1+kP2,kP2,0,0ull-kP1}, total=0;
    std::uint8_t tail[32]{}; std::uint32_t used=0;
};
void stripe(Hash& h,const std::uint8_t* p) noexcept {
    assert(p!=nullptr); assert(h.used<=32);
    for (std::uint32_t i=0;i<4;++i) h.lanes[i]=xxh_round(h.lanes[i],rd64(p+8*i));
}
void hash_update(Hash& h,const std::uint8_t* p,std::uint32_t bytes) noexcept {
    assert(p!=nullptr); assert(h.used<32);
    h.total+=bytes;
    if (h.used!=0) {
        const auto n=bytes<32-h.used?bytes:32-h.used;
        std::memcpy(h.tail+h.used,p,n); h.used+=n; p+=n; bytes-=n;
        if (h.used<32) return;
        stripe(h,h.tail); h.used=0;
    }
    for (std::uint32_t n=0;n<kZstdStreamBlockBytes/32 && bytes>=32;++n) {
        stripe(h,p); p+=32; bytes-=32;
    }
    if (bytes!=0) std::memcpy(h.tail,p,bytes);
    h.used=bytes; assert(h.used<32);
}
std::uint64_t hash_digest(const Hash& h) noexcept {
    assert(h.used<32); assert(h.total>=h.used);
    auto value=kP5;
    if (h.total>=32) {
        value=rotl64(h.lanes[0],1)+rotl64(h.lanes[1],7)+rotl64(h.lanes[2],12)+rotl64(h.lanes[3],18);
        for (const auto lane:h.lanes) value=xxh_merge(value,lane);
    }
    value+=h.total; std::uint32_t offset=0;
    for (;offset+8<=h.used;offset+=8) { value^=xxh_round(0,rd64(h.tail+offset)); value=rotl64(value,27)*kP1+kP4; }
    if (offset+4<=h.used) { value^=rd32(h.tail+offset)*kP1; value=rotl64(value,23)*kP2+kP3; offset+=4; }
    for (;offset<h.used;++offset) { value^=std::uint64_t(h.tail[offset])*kP5; value=rotl64(value,11)*kP1; }
    value^=value>>33; value*=kP2; value^=value>>29; value*=kP3; return value^(value>>32);
}
bool range(const void* p,std::uint64_t n) noexcept {
    assert(sizeof(std::uintptr_t)==sizeof(void*)); assert(SIZE_MAX<=UINT64_MAX);
    return n==0 || (p!=nullptr && n<=SIZE_MAX && n-1<=UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(p));
}
bool overlap(const void* a,std::uint64_t an,const void* b,std::uint64_t bn) noexcept {
    assert(sizeof(std::uintptr_t)==sizeof(void*)); assert(sizeof(std::uint64_t)>=sizeof(void*));
    if (an==0 || bn==0) return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<an:x-y<bn;
}
std::uint64_t little(const std::uint8_t* p,std::uint32_t n) noexcept {
    assert(p!=nullptr); assert(n<=8);
    std::uint64_t v=0; for (std::uint32_t i=0;i<n;++i) v|=std::uint64_t(p[i])<<(8*i); return v;
}
} // namespace
struct ZstdStream {
    std::uint64_t magic=kStreamMagic, workspace_bytes=0, history_bytes=0;
    std::uint8_t* history=nullptr; ZstdStreamLimits limits{};
    Scratch entropy{}; Hash hash{};
    std::uint8_t encoded[kZstdStreamBlockBytes]{},decoded[kZstdStreamBlockBytes]{},head[18]{};
    std::uint64_t frame_index=0,frame_out=0,total_out=0,total_input=0,window=0,fcs=0;
    std::uint64_t history_write=0,history_available=0;
    std::uint32_t fill=0,header_need=5,body_bytes=0,block_bytes=0,block_type=0;
    std::uint32_t decoded_bytes=0,drained=0,skip_left=0,rep[3]{1,4,8};
    Phase phase=Phase::Magic; Status error=Status::Progress;
    ZstdFrameIntegrity integrity=ZstdFrameIntegrity::Pending;
    bool checksum=false,have_fcs=false,last=false,seen=false,input_ended=false;
};
namespace {
struct RingOutput {
    static constexpr bool strict=true;
    static constexpr int capacity_error=kZstdBadInput;
    ZstdStream& s; std::uint32_t written=0;
    bool literals(const std::uint8_t* p,std::uint64_t n) noexcept {
        assert(p!=nullptr || n==0); assert(written<=s.block_bytes);
        if (n>s.block_bytes-written) return false;
        if (n!=0) std::memcpy(s.decoded+written,p,static_cast<std::size_t>(n));
        written+=static_cast<std::uint32_t>(n); return true;
    }
    bool sequence(const std::uint8_t** lit,std::uint64_t* left,std::uint32_t ll,
        std::uint32_t ml,std::uint32_t distance) noexcept {
        assert(lit!=nullptr && left!=nullptr); assert(written<=s.block_bytes);
        if (ll>*left || !literals(*lit,ll)) return false;
        *lit+=ll; *left-=ll;
        if (distance==0 || distance>s.window || written>UINT64_MAX-s.frame_out ||
            distance>s.frame_out+written || ml>s.block_bytes-written) return false;
        std::uint32_t remaining=ml;
        for (std::uint32_t part=0;part<2 && remaining!=0 && distance>written;++part) {
            const auto back=std::uint64_t(distance-written);
            if (back>s.history_available || s.window==0) return false;
            const auto index=s.history_write>=back?s.history_write-back:s.window-(back-s.history_write);
            auto n=std::uint64_t(remaining); if (n>back) n=back; if (n>s.window-index) n=s.window-index;
            std::memcpy(s.decoded+written,s.history+index,static_cast<std::size_t>(n));
            written+=static_cast<std::uint32_t>(n); remaining-=static_cast<std::uint32_t>(n);
        }
        if (remaining!=0 && distance>written) return false;
        for (std::uint32_t i=0;i<remaining;++i) { s.decoded[written]=s.decoded[written-distance]; ++written; }
        return true;
    }
    int sequence_error(std::uint32_t,std::uint32_t) const noexcept {
        assert(written<=s.block_bytes); assert(s.block_bytes<=kZstdStreamBlockBytes); return kZstdBadInput;
    }
};
ZstdStep result(const ZstdStream& s,Status status,std::uint32_t input=0,std::uint32_t output=0) noexcept {
    assert(s.magic==kStreamMagic); assert(input<=kZstdStreamBlockBytes && output<=kZstdStreamBlockBytes);
    return {status,input,output,s.frame_index,s.frame_out,s.total_out,s.integrity};
}
Status fail(ZstdStream& s,Status code) noexcept {
    assert(code>=Status::BadInput); assert(s.magic==kStreamMagic);
    s.error=code; s.phase=Phase::Failed; return code;
}
Status gather(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,
    std::uint8_t* target,std::uint32_t need,std::uint32_t& used) noexcept {
    assert(s.fill<=need); assert(need<=kZstdStreamBlockBytes);
    used=n<need-s.fill?n:need-s.fill;
    if (used>UINT64_MAX-s.total_input) { used=0; return fail(s,Status::CounterOverflow); }
    if (used!=0) std::memcpy(target+s.fill,input,used);
    s.fill+=used; s.total_input+=used;
    if (s.fill==need) return Status::Progress;
    return final?fail(s,Status::Truncated):Status::NeedInput;
}
Status magic(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.fill<=4); assert(s.phase==Phase::Magic);
    if (n==0 && s.fill==0 && final) {
        if (!s.seen) return fail(s,Status::Truncated);
        s.phase=Phase::Done; return Status::Finished;
    }
    const auto status=gather(s,input,n,final,s.head,4,used); if (status!=Status::Progress) return status;
    const auto value=little(s.head,4);
    if ((value&0xfffffff0u)==0x184d2a50u) s.phase=Phase::SkipHeader;
    else if (value==0xfd2fb528u) { s.phase=Phase::Header; s.header_need=5; }
    else return fail(s,Status::BadInput);
    return Status::Progress;
}
Status admit_header(ZstdStream& s) noexcept {
    assert(s.fill==s.header_need); assert(s.fill<=sizeof(s.head));
    const auto descriptor=s.head[4]; const bool single=(descriptor&0x20u)!=0;
    const std::uint32_t did[]{0,1,2,4};
    if ((descriptor&8u)!=0) return fail(s,Status::BadInput);
    if (did[descriptor&3u]!=0) return fail(s,Status::UnsupportedDictionaryEncoding);
    auto offset=5u; std::uint64_t window=0;
    if (!single) {
        const auto wd=s.head[offset++]; const auto base=std::uint64_t(1)<<(10+(wd>>3));
        window=base+(base/8)*(wd&7u);
    }
    const std::uint32_t sizes[]{single?1u:0u,2,4,8}; const auto bytes=sizes[descriptor>>6];
    s.fcs=bytes==0?0:little(s.head+offset,bytes); if (bytes==2) s.fcs+=256;
    if (single) window=s.fcs;
    if (window>UINT32_MAX) return fail(s,Status::UnsupportedWindow);
    if (window>s.limits.window_bytes) return fail(s,Status::WindowLimit);
    if (bytes!=0 && s.fcs>s.limits.max_decoded_bytes-s.total_out) return fail(s,Status::OutputLimit);
    if (s.frame_index==UINT64_MAX) return fail(s,Status::CounterOverflow);
    ++s.frame_index; s.frame_out=0; s.window=window; s.have_fcs=bytes!=0;
    s.checksum=(descriptor&4u)!=0; s.integrity=ZstdFrameIntegrity::Pending;
    s.history_write=0; s.history_available=0; s.hash=Hash{};
    s.entropy.huf.valid=false; s.entropy.seq.have_ll=s.entropy.seq.have_of=s.entropy.seq.have_ml=false;
    s.rep[0]=1; s.rep[1]=4; s.rep[2]=8; s.fill=0; s.phase=Phase::BlockHeader; return Status::Progress;
}
Status header(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::Header); assert(s.fill>=4 && s.fill<=18);
    auto status=gather(s,input,n,final,s.head,s.header_need,used); if (status!=Status::Progress) return status;
    if (s.header_need==5) {
        const auto descriptor=s.head[4]; const bool single=(descriptor&0x20u)!=0;
        const std::uint32_t did[]{0,1,2,4}, fcs[]{single?1u:0u,2,4,8};
        if ((descriptor&8u)!=0) return fail(s,Status::BadInput);
        s.header_need=5u+std::uint32_t(!single)+did[descriptor&3u]+fcs[descriptor>>6];
        if (s.fill<s.header_need) return Status::Progress;
    }
    return admit_header(s);
}
Status skip_header(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::SkipHeader); assert(s.fill>=4 && s.fill<=8);
    const auto status=gather(s,input,n,final,s.head,8,used); if (status!=Status::Progress) return status;
    s.skip_left=static_cast<std::uint32_t>(little(s.head+4,4)); s.fill=0; s.phase=Phase::Skip; return Status::Progress;
}
Status skip(ZstdStream& s,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::Skip); assert(s.fill==0);
    used=n<s.skip_left?n:s.skip_left; if (used>kZstdStreamBlockBytes) used=kZstdStreamBlockBytes;
    if (used>UINT64_MAX-s.total_input) { used=0; return fail(s,Status::CounterOverflow); }
    s.skip_left-=used; s.total_input+=used;
    if (s.skip_left==0) { s.seen=true; s.phase=Phase::Magic; return Status::Progress; }
    return final && used==n?fail(s,Status::Truncated):(used==0?Status::NeedInput:Status::Progress);
}
Status block_header(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::BlockHeader); assert(s.fill<=3);
    const auto status=gather(s,input,n,final,s.head,3,used); if (status!=Status::Progress) return status;
    const auto bits=static_cast<std::uint32_t>(little(s.head,3)); s.last=(bits&1u)!=0;
    s.block_type=(bits>>1)&3u; s.body_bytes=bits>>3;
    s.block_bytes=static_cast<std::uint32_t>(s.window<kZstdStreamBlockBytes?s.window:kZstdStreamBlockBytes);
    if (s.block_type==3 || s.body_bytes>s.block_bytes) return fail(s,Status::BadInput);
    if (s.block_type==1) { s.decoded_bytes=s.body_bytes; s.body_bytes=1; }
    s.fill=0; s.phase=Phase::Body; return Status::Progress;
}
Status body(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::Body); assert(s.body_bytes<=kZstdStreamBlockBytes);
    const auto status=gather(s,input,n,final,s.encoded,s.body_bytes,used);
    if (status==Status::Progress) s.phase=Phase::Decode;
    return status;
}
void history_append(ZstdStream& s) noexcept {
    assert(s.decoded_bytes<=s.window); assert(s.decoded_bytes<=kZstdStreamBlockBytes);
    if (s.decoded_bytes==0) return;
    const auto first=static_cast<std::uint32_t>(s.window-s.history_write<s.decoded_bytes?s.window-s.history_write:s.decoded_bytes);
    std::memcpy(s.history+s.history_write,s.decoded,first);
    const auto rest=s.decoded_bytes-first;
    if (rest!=0) std::memcpy(s.history,s.decoded+first,rest);
    s.history_write+=s.decoded_bytes; if (s.history_write>=s.window) s.history_write-=s.window;
    s.history_available=s.history_available>s.window-s.decoded_bytes?s.window:s.history_available+s.decoded_bytes;
}
Status decode(ZstdStream& s) noexcept {
    assert(s.phase==Phase::Decode); assert(s.fill==s.body_bytes);
    if (s.block_type==0) { s.decoded_bytes=s.body_bytes; if (s.decoded_bytes!=0) std::memcpy(s.decoded,s.encoded,s.decoded_bytes); }
    else if (s.block_type==1) { if (s.decoded_bytes!=0) std::memset(s.decoded,s.encoded[0],s.decoded_bytes); }
    else {
        RingOutput output{s};
        if (decode_compressed_block_to(&s.entropy,s.encoded,s.body_bytes,output,s.rep)!=kZstdOk) return fail(s,Status::BadInput);
        s.decoded_bytes=output.written;
    }
    if (s.decoded_bytes>UINT64_MAX-s.frame_out || s.decoded_bytes>UINT64_MAX-s.total_out) return fail(s,Status::CounterOverflow);
    if (s.decoded_bytes>s.limits.max_decoded_bytes-s.total_out) return fail(s,Status::OutputLimit);
    if (s.have_fcs && s.decoded_bytes>s.fcs-s.frame_out) return fail(s,Status::BadInput);
    history_append(s); hash_update(s.hash,s.decoded,s.decoded_bytes);
    s.frame_out+=s.decoded_bytes; s.total_out+=s.decoded_bytes; s.drained=0; s.fill=0; s.phase=Phase::Drain;
    assert(s.hash.total==s.frame_out); assert(s.history_available<=s.window);
    return Status::Progress;
}
Status drain(ZstdStream& s,std::uint8_t* output,std::uint32_t capacity,std::uint32_t& produced) noexcept {
    assert(s.phase==Phase::Drain); assert(s.drained<=s.decoded_bytes);
    produced=capacity<s.decoded_bytes-s.drained?capacity:s.decoded_bytes-s.drained;
    if (produced!=0) std::memcpy(output,s.decoded+s.drained,produced);
    s.drained+=produced;
    if (s.drained!=s.decoded_bytes) return Status::NeedOutput;
    s.phase=s.last?(s.checksum?Phase::Trailer:Phase::End):Phase::BlockHeader; return Status::Progress;
}
Status frame_end(ZstdStream& s) noexcept {
    assert(s.phase==Phase::End); assert(s.drained==s.decoded_bytes);
    if (s.have_fcs && s.frame_out!=s.fcs) return fail(s,Status::BadInput);
    if (!s.checksum) s.integrity=ZstdFrameIntegrity::NoChecksum;
    s.seen=true; s.fill=0; s.phase=Phase::Magic; return Status::FrameEnd;
}
Status trailer(ZstdStream& s,const std::uint8_t* input,std::uint32_t n,bool final,std::uint32_t& used) noexcept {
    assert(s.phase==Phase::Trailer); assert(s.checksum);
    if (s.have_fcs && s.frame_out!=s.fcs) return fail(s,Status::BadInput);
    const auto status=gather(s,input,n,final,s.head,4,used); if (status!=Status::Progress) return status;
    if (static_cast<std::uint32_t>(hash_digest(s.hash))!=little(s.head,4)) return fail(s,Status::ChecksumMismatch);
    s.integrity=ZstdFrameIntegrity::ChecksumVerified; s.phase=Phase::End; return Status::Progress;
}
bool alias(const ZstdStream& s,const void* input,std::uint32_t n,void* output,std::uint32_t cap) noexcept {
    assert(s.workspace_bytes>=sizeof(s)); assert(s.history_bytes>=s.limits.window_bytes);
    return !range(input,n) || !range(output,cap) || overlap(input,n,output,cap) ||
        overlap(input,n,&s,s.workspace_bytes) || overlap(output,cap,&s,s.workspace_bytes) ||
        overlap(input,n,s.history,s.history_bytes) || overlap(output,cap,s.history,s.history_bytes);
}
} // namespace
std::uint64_t zstd_stream_state_size() noexcept {
    assert(alignof(ZstdStream)<=kZstdStreamAlignment); assert(sizeof(ZstdStream)>2*kZstdStreamBlockBytes);
    return sizeof(ZstdStream);
}
ZstdStreamStatus zstd_stream_init(void* memory,std::uint64_t bytes,std::uint8_t* history,
    std::uint64_t history_bytes,ZstdStreamLimits limits,ZstdStream** out) noexcept {
    assert(sizeof(std::uintptr_t)==sizeof(void*)); assert(alignof(ZstdStream)<=kZstdStreamAlignment);
    if (!range(memory,bytes) || !range(history,history_bytes) || !range(out,sizeof(*out)) ||
        memory==nullptr || out==nullptr || history==nullptr || limits.window_bytes==0 ||
        reinterpret_cast<std::uintptr_t>(memory)%kZstdStreamAlignment!=0 ||
        overlap(memory,bytes,history,history_bytes) || overlap(out,sizeof(*out),memory,bytes) ||
        overlap(out,sizeof(*out),history,history_bytes)) return Status::InvalidArgument;
    if (bytes<sizeof(ZstdStream) || history_bytes<limits.window_bytes) return Status::WorkspaceSmall;
    auto* s=new(memory) ZstdStream{}; s->workspace_bytes=bytes; s->history=history;
    s->history_bytes=history_bytes; s->limits=limits; *out=s; return Status::Progress;
}
ZstdStep zstd_stream_step(ZstdStream* s,const std::uint8_t* input,std::uint32_t bytes,bool final,
    std::uint8_t* output,std::uint32_t capacity) noexcept {
    assert(kZstdStreamBlockBytes==kBlockMax); assert(kZstdStreamAlignment>=alignof(ZstdStream));
    if (s==nullptr || s->magic!=kStreamMagic) return {};
    if (alias(*s,input,bytes,output,capacity)) return result(*s,Status::InvalidArgument);
    if (s->phase==Phase::Failed) return result(*s,s->error);
    if (s->input_ended && bytes!=0) return result(*s,Status::InvalidArgument);
    if (s->phase==Phase::Done) return result(*s,bytes==0?Status::Finished:Status::InvalidArgument);
    final=final || s->input_ended;
    std::uint32_t consumed=0,produced=0; Status status=Status::Progress;
    switch (s->phase) {
    case Phase::Magic: status=magic(*s,input,bytes,final,consumed); break;
    case Phase::Header: status=header(*s,input,bytes,final,consumed); break;
    case Phase::SkipHeader: status=skip_header(*s,input,bytes,final,consumed); break;
    case Phase::Skip: status=skip(*s,bytes,final,consumed); break;
    case Phase::BlockHeader: status=block_header(*s,input,bytes,final,consumed); break;
    case Phase::Body: status=body(*s,input,bytes,final,consumed); break;
    case Phase::Decode: status=decode(*s); break;
    case Phase::Drain: status=drain(*s,output,capacity,produced); break;
    case Phase::Trailer: status=trailer(*s,input,bytes,final,consumed); break;
    case Phase::End: status=frame_end(*s); break;
    case Phase::Done: case Phase::Failed: return result(*s,Status::InvalidArgument);
    }
    if (final && consumed==bytes) s->input_ended=true;
    return result(*s,status,consumed,produced);
}
} // namespace bolt::ingest
