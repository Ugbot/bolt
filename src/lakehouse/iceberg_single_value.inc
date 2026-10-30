// Appendix D scalar binary codec helpers. PartitionValue is a bounded carrier;
// wider statistics retain their separate budgeted buffers, never truncation here.
uint32_t single_width(IcebergType type) noexcept {
    using T=IcebergType;
    switch(type) {
        case T::kBoolean:return 1;
        case T::kInt:case T::kDate:case T::kFloat:return 4;
        case T::kLong:case T::kTime:case T::kTimestamp:case T::kTimestampTz:
        case T::kTimestampNs:case T::kTimestampTzNs:case T::kDouble:return 8;
        default:return 0;
    }
}
bool single_bytes_type(IcebergType type) noexcept {
    return is_bytes(type) || type==IcebergType::kGeometry || type==IcebergType::kGeography;
}
bool single_decimal_valid(const IcebergTypeInfo& type,const uint8_t* bytes,uint32_t length) noexcept {
    assert(bytes || length==0);assert(type.type==IcebergType::kDecimal);
    if(!length || length>16 || type.precision<1 || type.precision>38 || type.scale>type.precision ||
        type.fixed_len!=decimal_required_bytes(type.precision))return false;
    int64_t high=0;uint64_t low=0;decimal_from_be(bytes,length,&high,&low);
    const kernels::decimal::Decimal128 value{std::bit_cast<int64_t>(low),high};
    return kernels::decimal::d128_column_fits_precision(&value,1,nullptr,0,type.precision);
}
bool single_value_valid(const IcebergTypeInfo& type,const PartitionValue& value) noexcept {
    assert(sizeof(value.str)==kLakeMaxValBytes);assert(kMaxValue<UINT8_MAX);
    if(value.is_null || value.is_partial)return false;
    if(is_intlike(type.type)) {
        if(!value.is_int || value.is_float || value.is_str)return false;
        if(type.type==IcebergType::kInt || type.type==IcebergType::kDate)
            return value.i64>=INT32_MIN && value.i64<=INT32_MAX;
        if(type.type==IcebergType::kBoolean)return value.i64==0 || value.i64==1;
        if(type.type==IcebergType::kTime)return value.i64>=0 && value.i64<INT64_C(86400000000);
        return true;
    }
    if(type.type==IcebergType::kFloat || type.type==IcebergType::kDouble) {
        if(!value.is_float || value.is_int || value.is_str)return false;
        return type.type!=IcebergType::kFloat || !std::isfinite(value.f64) ||
            std::fabs(value.f64)<=std::numeric_limits<float>::max();
    }
    if(!single_bytes_type(type.type) || !value.is_str || value.is_int || value.is_float)return false;
    const auto length=partition_value_bytes_len(value);
    if(length>kMaxValue || (!value.str_len && value.str[length]!=0))return false;
    if(type.type==IcebergType::kUuid)return type.fixed_len==16 && length==16;
    if(type.type==IcebergType::kFixed)return type.fixed_len>0 && length==type.fixed_len;
    if(type.type==IcebergType::kDecimal)
        return single_decimal_valid(type,reinterpret_cast<const uint8_t*>(value.str),length);
    return true;
}
uint64_t single_read_le(const uint8_t* bytes,uint32_t width) noexcept {
    assert(bytes);assert(width>0 && width<=8);
    uint64_t bits=0;
    for(uint32_t i=0;i<width;++i)bits|=uint64_t{bytes[i]}<<(8u*i);
    return bits;
}
void single_write_le(uint64_t bits,uint32_t width,uint8_t* out) noexcept {
    assert(out);assert(width>0 && width<=8);
    for(uint32_t i=0;i<width;++i)out[i]=static_cast<uint8_t>(bits>>(8u*i));
}
uint32_t encode_value(const IcebergTypeInfo& type,const PartitionValue& value,uint8_t* out,uint32_t cap) noexcept {
    assert(out);assert(single_value_valid(type,value));
    const auto width=single_width(type.type);
    if(width) {
        if(cap<width)return 0;
        uint64_t bits=static_cast<uint64_t>(value.i64);
        if(type.type==IcebergType::kFloat)bits=std::bit_cast<uint32_t>(static_cast<float>(value.f64));
        if(type.type==IcebergType::kDouble)bits=std::bit_cast<uint64_t>(value.f64);
        single_write_le(bits,width,out);return width;
    }
    const auto length=partition_value_bytes_len(value);
    if(type.type==IcebergType::kDecimal) {
        int64_t high=0;uint64_t low=0;
        decimal_from_be(reinterpret_cast<const uint8_t*>(value.str),length,&high,&low);
        uint8_t minimal[16];const auto bytes=decimal_min_be_bytes(high,low,minimal);
        if(bytes>cap)return 0;
        std::memcpy(out,minimal,bytes);return bytes;
    }
    if(length>cap)return 0;
    if(length)std::memmove(out,value.str,length);
    return length;
}
bool decode_single_value(const IcebergTypeInfo& type,const uint8_t* bytes,uint32_t length,PartitionValue* out) noexcept {
    assert(bytes || !length);assert(out);
    const auto width=single_width(type.type);
    if(width) {
        if(length!=width)return false;
        const auto bits=single_read_le(bytes,width);
        if(type.type==IcebergType::kFloat || type.type==IcebergType::kDouble) {
            out->is_float=true;
            out->f64=width==4 ? std::bit_cast<float>(static_cast<uint32_t>(bits)) : std::bit_cast<double>(bits);
        } else {
            out->is_int=true;
            out->i64=width==1 ? (bits!=0) : width==4 ? std::bit_cast<int32_t>(static_cast<uint32_t>(bits)) : std::bit_cast<int64_t>(bits);
        }
    } else {
        if(!single_bytes_type(type.type) || length>kMaxValue)return false;
        if(type.type==IcebergType::kDecimal) {
            if(!single_decimal_valid(type,bytes,length))return false;
            int64_t high=0;uint64_t low=0;decimal_from_be(bytes,length,&high,&low);
            if(!put_decimal(out,high,low,type.fixed_len))return false;
        } else put_bytes(out,bytes,length);
    }
    return single_value_valid(type,*out);
}
