bool digits(const uint8_t* text,uint32_t size,uint32_t* at,uint32_t count,uint32_t* out) noexcept {
    assert(text && at && out);assert(*at<=size);
    if(count>size-*at)return false;
    uint32_t value=0;
    for(uint32_t i=0;i<count;++i) {const auto c=text[*at+i];if(c<'0' || c>'9')return false;value=value*10u+(c-'0');}
    *at+=count;*out=value;return true;
}
bool iso_date(const uint8_t* text,uint32_t size,uint32_t* at,int64_t* days) noexcept {
    assert(text && at && days);assert(*at<=size);
    bool negative=false,sign=false;
    if(*at<size && (text[*at]=='-' || text[*at]=='+')) {sign=true;negative=text[(*at)++]=='-';}
    const auto first=*at;uint32_t year=0,month=0,day=0;
    while(*at<size && text[*at]>='0' && text[*at]<='9' && *at-first<9u)year=year*10u+(text[(*at)++]-'0');
    if(*at-first<4 || (!sign && *at-first!=4) || *at>=size || text[(*at)++]!='-' ||
        !digits(text,size,at,2,&month) || *at>=size || text[(*at)++]!='-' || !digits(text,size,at,2,&day) ||
        month<1 || month>12 || day<1 || day>31)return false;
    const auto y=negative?-static_cast<int32_t>(year):static_cast<int32_t>(year);
    const auto value=kernels::date::days_from_civil(y,month,day);
    if(value<INT32_MIN || value>INT32_MAX)return false;
    const auto checked=kernels::civil_from_days(value);
    if(checked.y!=y || checked.m!=month || checked.d!=day)return false;
    *days=value;return true;
}
bool iso_clock(const uint8_t* text,uint32_t size,uint32_t* at,int64_t unit,int64_t* seconds,int64_t* fraction) noexcept {
    assert(text && at && seconds && fraction);assert(unit==1000000 || unit==1000000000);
    uint32_t hour=0,minute=0,second=0;
    if(!digits(text,size,at,2,&hour) || *at>=size || text[(*at)++]!=':' || !digits(text,size,at,2,&minute))return false;
    const bool has_seconds=*at<size && text[*at]==':';
    if(has_seconds) {++*at;if(!digits(text,size,at,2,&second))return false;}
    if(hour>23 || minute>59 || second>59)return false;
    int64_t value=0,scale=unit;
    if(*at<size && text[*at]=='.') {
        if(!has_seconds)return false;
        ++*at;const auto first=*at;
        while(*at<size && text[*at]>='0' && text[*at]<='9') {
            if(scale==1)return false;
            scale/=10;value+=int64_t{text[(*at)++]-'0'}*scale;
        }
        if(*at==first)return false;
    }
    *seconds=int64_t{hour}*3600+minute*60+second;*fraction=value;return true;
}
bool compose_ticks(int64_t seconds,int64_t fraction,int64_t unit,int64_t* out) noexcept {
    assert(out && fraction>=0 && fraction<unit);assert(unit==1000000 || unit==1000000000);
    const auto low=kernels::fdiv_i64(INT64_MIN,unit),high=INT64_MAX/unit;
    const auto low_part=unit+INT64_MIN%unit,high_part=INT64_MAX%unit;
    if(seconds<low || seconds>high || (seconds==low && fraction<low_part) || (seconds==high && fraction>high_part))return false;
    *out=seconds==low ? INT64_MIN+(fraction-low_part) : seconds*unit+fraction;return true;
}
bool temporal_parse(IcebergType kind,const uint8_t* text,uint32_t size,int64_t* out) noexcept {
    assert(text || !size);assert(out);
    using T=IcebergType;uint32_t at=0;int64_t days=0,seconds=0,fraction=0;
    const bool time=kind==T::kTime,date=kind==T::kDate;
    const bool utc=kind==T::kTimestampTz || kind==T::kTimestampTzNs;
    const int64_t unit=kind==T::kTimestampNs || kind==T::kTimestampTzNs?1000000000:1000000;
    if(!size)return false;
    if(!time && !iso_date(text,size,&at,&days))return false;
    if(date) {if(at!=size)return false;*out=days;return true;}
    if(!time && (at>=size || text[at++]!='T'))return false;
    if(!iso_clock(text,size,&at,unit,&seconds,&fraction))return false;
    if(utc) {if(size-at!=6 || std::memcmp(text+at,"+00:00",6))return false;at+=6;}
    if(at!=size)return false;
    return compose_ticks(days*86400+seconds,fraction,unit,out);
}
uint32_t temporal_format(IcebergType kind,int64_t ticks,char* out,uint32_t cap) noexcept {
    assert(out);assert(cap>=80);
    using T=IcebergType;const bool date=kind==T::kDate,time=kind==T::kTime;
    const bool utc=kind==T::kTimestampTz || kind==T::kTimestampTzNs;
    const int64_t unit=kind==T::kTimestampNs || kind==T::kTimestampTzNs?1000000000:1000000;
    int64_t seconds=date?0:kernels::fdiv_i64(ticks,unit),fraction=date?0:kernels::fmod_i64(ticks,unit);
    int64_t day=date?ticks:kernels::fdiv_i64(seconds,86400);
    if((date && (ticks<INT32_MIN || ticks>INT32_MAX)) || (time && (ticks<0 || ticks>=INT64_C(86400000000))))return 0;
    uint32_t at=0;
    if(!time) {
        const auto ymd=kernels::civil_from_days(day);const int64_t y=ymd.y;
        const int n=std::snprintf(out,cap,y<0?"-%04lld-%02u-%02u":y>9999?"+%04lld-%02u-%02u":"%04lld-%02u-%02u",
            static_cast<long long>(y<0?-y:y),ymd.m,ymd.d);
        if(n<=0 || static_cast<uint32_t>(n)>=cap)return 0;
        at=static_cast<uint32_t>(n);if(date)return at;out[at++]='T';
    }
    const auto clock=seconds-day*86400;
    const int n=std::snprintf(out+at,cap-at,"%02lld:%02lld:%02lld",static_cast<long long>(clock/3600),
        static_cast<long long>((clock/60)%60),static_cast<long long>(clock%60));
    if(n<=0 || static_cast<uint32_t>(n)>=cap-at)return 0;
    at+=static_cast<uint32_t>(n);
    if(fraction) {
        const int nfrac=std::snprintf(out+at,cap-at,unit==1000000?".%06lld":".%09lld",static_cast<long long>(fraction));
        if(nfrac<=0 || static_cast<uint32_t>(nfrac)>=cap-at)return 0;
        at+=static_cast<uint32_t>(nfrac);
    }
    if(utc) {if(cap-at<7)return 0;std::memcpy(out+at,"+00:00",6);at+=6;out[at]=0;}
    return at;
}
