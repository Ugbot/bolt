#pragma once
// Allocation-free fixed-scalar normalization into caller-owned storage.
#include "bolt/bolt_column.h"
#include "bolt/bolt_branchless.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt::kernels {
namespace fixed_gather_detail {
inline std::uint32_t width(const BoltColumn& c) noexcept {
    if(c.logical!=BoltLogical::None)return 0;
    switch(c.type) {
        case BoltType::Bool:case BoltType::Int8:case BoltType::UInt8:return 1;
        case BoltType::Int16:case BoltType::UInt16:return 2;
        case BoltType::Int32:case BoltType::UInt32:return 4;
        case BoltType::Int64:case BoltType::UInt64:case BoltType::Float64:return 8;
        default:return 0;
    }
}
struct Span {const void* pointer;std::uint64_t bytes;};
inline bool valid(Span s) noexcept {
    return (!s.bytes || s.pointer) && s.bytes<=SIZE_MAX &&
        s.bytes<=UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(s.pointer);
}
inline bool disjoint(Span a,Span b) noexcept {
    if(!valid(a) || !valid(b))return false;
    const auto x=reinterpret_cast<std::uintptr_t>(a.pointer),y=reinterpret_cast<std::uintptr_t>(b.pointer);
    return !a.bytes || !b.bytes || x+a.bytes<=y || y+b.bytes<=x;
}
inline bool buffers(const BoltColumn& src,const std::int32_t* selection,std::int64_t logical,
        std::int64_t input,std::uint32_t bytes,void* dst,std::uint64_t dst_bytes,
        std::uint8_t* validity,std::uint64_t validity_bytes,std::uint32_t* count) noexcept {
    const bool constant=src.format==ColumnFormat::Constant;
    const Span outputs[]={{dst,dst_bytes},{validity,validity_bytes},{count,sizeof(*count)}};
    const std::uint64_t bitmap=src.validity && input?
        (std::uint64_t(src.validity_offset)+std::uint64_t(input)+7)/8:0;
    const Span inputs[]={{&src,sizeof(src)},
        {constant?src.inline_value:src.data,constant?sizeof(src.inline_value):std::uint64_t(input)*bytes},
        {src.validity,bitmap},{selection,selection?std::uint64_t(logical)*sizeof(*selection):0}};
    for(unsigned i=0;i<3;++i) {
        if(!valid(outputs[i]))return false;
        for(unsigned j=0;j<i;++j)if(!disjoint(outputs[i],outputs[j]))return false;
        for(const auto in:inputs)if(!disjoint(outputs[i],in))return false;
    }
    return true;
}
template<unsigned Bytes> inline void copy(const BoltColumn& src,const std::int32_t* selection,
        std::int64_t rows,void* dst) noexcept {
    assert(rows>0);assert(dst!=nullptr);
    if(src.format==ColumnFormat::Constant) {
        struct Pod {std::uint8_t bytes[Bytes];};
        Pod value{};std::memcpy(&value,src.inline_value,Bytes);
        std::fill_n(static_cast<Pod*>(dst),static_cast<std::size_t>(rows),value);
    } else if(selection) bolt::branchless::gather_pod<Bytes>(src.data,selection,rows,dst);
    else std::memcpy(dst,src.data,static_cast<std::size_t>(rows)*Bytes);
}
} // namespace fixed_gather_detail

// Flat/View require contiguous exact-width scalar data; Constant reads inline
// value bytes and the SAME per-row validity bitmap used by BoltColumn::is_null.
// Cached stats.all_valid/null_count never override an actual bitmap. No Float32,
// logical annotations, strides, sequence, strings or nested encodings.
// Dense input requires logical_rows==input_rows; selection indices are physical
// rows in [0,input_rows), with input_rows<=src.length. logical_rows<=UINT32_MAX.
// Source/selection are immutable for the call. All advertised output extents and
// *null_count must be disjoint from each other and every source span/header.
// Byte alignment is sufficient (POD gather preserves UInt64/Float64/NaN bits).
// Every refusal preserves ALL outputs. Success writes exact logical payload,
// ceil(rows/8) canonical validity bytes (unused high bits zero), and null count;
// trailing output capacity is untouched. Zero rows permits null zero-cap buffers
// and publishes null_count=0. No allocation, growth or retained pointers.
inline bool fixed_gather_to(const BoltColumn& src,const std::int32_t* selection,
        std::int64_t logical_rows,std::int64_t input_rows,void* dst,std::uint64_t dst_bytes,
        std::uint8_t* validity,std::uint64_t validity_bytes,std::uint32_t* null_count) noexcept {
    using namespace fixed_gather_detail;
    const auto bytes=width(src);
    if(!null_count || !bytes || src.type_size_bytes!=bytes || logical_rows<0 || input_rows<0 ||
        logical_rows>UINT32_MAX || input_rows>src.length || (!selection && logical_rows!=input_rows) ||
        src.validity_offset<0 || src.validity_offset>INT64_MAX-input_rows ||
        (src.format!=ColumnFormat::Flat && src.format!=ColumnFormat::View && src.format!=ColumnFormat::Constant) ||
        std::uint64_t(input_rows)>UINT64_MAX/bytes || std::uint64_t(logical_rows)*bytes>dst_bytes ||
        (std::uint64_t(logical_rows)+7)/8>validity_bytes)return false;
    assert(bytes==1 || bytes==2 || bytes==4 || bytes==8);assert(logical_rows<=UINT32_MAX);
    if(!buffers(src,selection,logical_rows,input_rows,bytes,dst,dst_bytes,validity,validity_bytes,null_count))return false;
    if(logical_rows && ((!dst || !validity) || (src.format!=ColumnFormat::Constant && !src.data)))return false;
    for(std::int64_t i=0;selection && i<logical_rows;++i)
        if(selection[i]<0 || std::int64_t(selection[i])>=input_rows)return false;
    if(!logical_rows){*null_count=0;return true;}
    switch(bytes) {
        case 1:copy<1>(src,selection,logical_rows,dst);break;
        case 2:copy<2>(src,selection,logical_rows,dst);break;
        case 4:copy<4>(src,selection,logical_rows,dst);break;
        case 8:copy<8>(src,selection,logical_rows,dst);break;
        default:return false;
    }
    std::memset(validity,0,static_cast<std::size_t>((logical_rows+7)/8));
    std::uint32_t nulls=0;
    for(std::int64_t i=0;i<logical_rows;++i) {
        const auto physical=selection?std::int64_t(selection[i]):i;
        const auto bit=src.validity_offset+physical;
        const bool present=!src.validity || (src.validity[bit>>3]&(1u<<(bit&7)))!=0;
        if(present)validity[i>>3]|=static_cast<std::uint8_t>(1u<<(i&7));else ++nulls;
    }
    *null_count=nulls;assert(nulls<=std::uint64_t(logical_rows));return true;
}
} // namespace bolt::kernels
