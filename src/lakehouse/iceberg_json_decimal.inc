bool decimal_parse(const uint8_t* text,uint32_t size,const IcebergTypeInfo& type,dec::Decimal128* out) noexcept {
    assert(text || !size);assert(out && type.precision<=38);
    if(!size || !type.precision || type.scale>type.precision)return false;
    uint32_t at=0,significant=0,fraction=0,digits_seen=0;bool negative=false,point=false;
    if(text[at]=='+' || text[at]=='-') {negative=text[at]=='-';if(++at==size)return false;}
    dec::Decimal128 value{0,0};
    for(;at<size;++at) {
        const auto c=text[at];if(c=='.' && !point) {point=true;continue;}
        if(c=='e' || c=='E')break;
        if(c<'0' || c>'9')return false;
        ++digits_seen;if(point)++fraction;
        if(significant || c!='0') {if(++significant>type.precision)return false;}
        value=dec::d128_add(dec::d128_mul(value,dec::d128_from_i64(10)),dec::d128_from_i64(c-'0'));
    }
    if(!digits_seen)return false;
    int64_t exponent=0;
    if(at<size) {
        ++at;bool minus=false;if(at<size && (text[at]=='+' || text[at]=='-'))minus=text[at++]=='-';
        const auto first=at;
        for(;at<size;++at) {if(text[at]<'0' || text[at]>'9' || exponent>1000000)return false;exponent=exponent*10+(text[at]-'0');}
        if(at==first)return false;if(minus)exponent=-exponent;
    }
    if(int64_t{fraction}-exponent!=type.scale)return false;
    if(negative)value=dec::d128_neg(value);
    if(!dec::d128_column_fits_precision(&value,1,nullptr,0,type.precision))return false;
    *out=value;return true;
}
uint32_t decimal_format(dec::Decimal128 value,uint8_t scale,char* out,uint32_t cap) noexcept {
    assert(out);assert(scale<=38);
    if(cap<44)return 0;
    const bool negative=value.hi<0;uint64_t high=static_cast<uint64_t>(value.hi),low=static_cast<uint64_t>(value.lo);
    if(negative) {low=~low+1u;high=~high+(low==0?1u:0u);}
    char digits[40];uint32_t count=0;
    do {uint64_t qh=0,ql=0,rem=0;dec::div_u128_u64(high,low,10,&qh,&ql,&rem);digits[count++]=static_cast<char>('0'+rem);high=qh;low=ql;}
    while((high || low) && count<40);
    if(high || low)return 0;
    while(count<=scale)digits[count++]='0';
    uint32_t at=0;if(negative)out[at++]='-';
    for(uint32_t i=count;i>0;--i) {if(i==scale && scale)out[at++]='.';out[at++]=digits[i-1];}
    out[at]=0;return at;
}
