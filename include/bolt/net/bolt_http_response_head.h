#pragma once
// Caller-buffer HTTP/1 response head parser. Does not read a socket or allocate.
// Slices borrow input bytes. Transport/body-framing policy belongs to the caller.
#include <cassert>
#include <cstdint>
#include <cstring>
namespace bolt::net {
inline constexpr std::uint32_t kHttpResponseHeadCap = 8192;
inline constexpr std::uint32_t kHttpResponseHeaderCap = 32;
enum class HttpHeadParse : std::uint8_t { kIncomplete, kInvalid, kComplete };
struct HttpHeadSlice { const char* bytes; std::uint32_t size; };
struct HttpResponseHeaderView { HttpHeadSlice name, value; };
struct HttpResponseHead {
    std::uint32_t status, bytes, header_count;
    HttpResponseHeaderView headers[kHttpResponseHeaderCap];
    std::uint64_t content_length;
    bool has_content_length, has_transfer_encoding;
};
namespace http_head_detail {
inline bool iequal(HttpHeadSlice v, const char* name) noexcept {
    assert(v.bytes != nullptr); assert(name != nullptr);
    if (std::strlen(name) != v.size) return false;
    for (auto i=0u;i<v.size;++i) {
        auto c=v.bytes[i]; if (c>='A' && c<='Z') c=static_cast<char>(c-'A'+'a');
        if (c != name[i]) return false;
    }
    return true;
}
inline bool token(char c) noexcept {
    return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') ||
        std::strchr("!#$%&'*+-.^_`|~",c)!=nullptr;
}
inline bool line(const char* b,std::uint32_t start,std::uint32_t end, HttpResponseHead& out) noexcept {
    assert(b != nullptr); assert(start <= end);
    auto colon=start;
    while (colon<end && b[colon]!=':') { if (!token(b[colon]) || b[colon]==0) return false; ++colon; }
    if (colon==start || colon==end || out.header_count==kHttpResponseHeaderCap) return false;
    auto first=colon+1, last=end;
    while (first<last && (b[first]==' ' || b[first]=='\t')) ++first;
    while (last>first && (b[last-1]==' ' || b[last-1]=='\t')) --last;
    HttpResponseHeaderView h{{b+start,colon-start},{b+first,last-first}};
    for (auto i=first;i<last;++i) { const auto c=static_cast<unsigned char>(b[i]); if ((c<32 && c!=9) || c==127) return false; }
    if (iequal(h.name,"content-length")) {
        if (out.has_content_length || out.has_transfer_encoding || first==last) return false;
        std::uint64_t n=0;
        for (auto i=first;i<last;++i) {
            const auto c=b[i]; if (c<'0' || c>'9' || n>(UINT64_MAX-static_cast<std::uint64_t>(c-'0'))/10) return false;
            n=n*10+static_cast<std::uint64_t>(c-'0');
        }
        out.has_content_length=true; out.content_length=n;
    }
    if (iequal(h.name,"transfer-encoding")) {
        if (out.has_transfer_encoding || out.has_content_length || first==last) return false;
        out.has_transfer_encoding=true;
    }
    out.headers[out.header_count++]=h; return true;
}
}
inline HttpHeadParse http_response_head_parse(const char* b,std::uint32_t n,HttpResponseHead* out) noexcept {
    assert(b != nullptr && out != nullptr); assert(kHttpResponseHeadCap>=16);
    *out={};
    std::uint32_t end=0;
    const auto scan=n<kHttpResponseHeadCap?n:kHttpResponseHeadCap;
    for (auto i=0u;i<scan;++i) {
        if (b[i]=='\n' && (i==0 || b[i-1]!='\r')) return HttpHeadParse::kInvalid;
        if (b[i]=='\r' && i+1<scan && b[i+1]!='\n') return HttpHeadParse::kInvalid;
        if (i>=3 && std::memcmp(b+i-3,"\r\n\r\n",4)==0) { end=i+1; break; }
    }
    if (end==0) return n>=kHttpResponseHeadCap?HttpHeadParse::kInvalid:HttpHeadParse::kIncomplete;
    if (end<16 || (std::memcmp(b,"HTTP/1.1 ",9)!=0 && std::memcmp(b,"HTTP/1.0 ",9)!=0) ||
        b[9]<'1' || b[9]>'5' || b[10]<'0' || b[10]>'9' || b[11]<'0' || b[11]>'9' || b[12]!=' ') return HttpHeadParse::kInvalid;
    out->status=static_cast<std::uint32_t>((b[9]-'0')*100+(b[10]-'0')*10+b[11]-'0');
    auto i=13u;
    for (;i+1<end && b[i]!='\r';++i) { const auto c=static_cast<unsigned char>(b[i]); if ((c<32 && c!=9) || c==127) return HttpHeadParse::kInvalid; }
    if (i+1>=end || b[i+1]!='\n') return HttpHeadParse::kInvalid;
    i+=2;
    while (i+1<end) {
        if (b[i]=='\r' && b[i+1]=='\n') { out->bytes=end; return HttpHeadParse::kComplete; }
        const auto start=i;
        while (i+1<end && b[i]!='\r') ++i;
        if (i+1>=end || b[i+1]!='\n' || !http_head_detail::line(b,start,i,*out)) return HttpHeadParse::kInvalid;
        i+=2;
    }
    return HttpHeadParse::kInvalid;
}
}
