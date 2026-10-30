kernels::wkt::Limits geo_limits(const State& state) noexcept {
    assert(state.limits.encoded_bytes>0);assert(state.limits.values>0);
    return {state.limits.encoded_bytes,state.limits.values,kernels::wkb::k_max_depth};
}
bool geo_status(State& state,kernels::wkt::Status status) noexcept {
    assert(state.arena);assert(state.limits.values>0);
    if(status==kernels::wkt::Status::Capacity)return state.capacity();
    return status==kernels::wkt::Status::Ok;
}
bool geo_parse(State& state,const J* value,const uint8_t** out,uint32_t* size) noexcept {
    assert(out && size);assert(value);
    const uint8_t* text=nullptr;uint32_t length=0;if(!raw_string(state,value,&text,&length))return false;
    const auto limits=geo_limits(state);uint32_t needed=0;
    auto status=kernels::wkt::to_wkb(reinterpret_cast<const char*>(text),length,nullptr,0,&needed,limits);
    if(!geo_status(state,status))return false;
    auto* bytes=state.arena->allocate_array<uint8_t>(needed);if(!bytes)return state.capacity();
    status=kernels::wkt::to_wkb(reinterpret_cast<const char*>(text),length,bytes,needed,&needed,limits);
    if(!geo_status(state,status))return false;
    *out=bytes;*size=needed;return true;
}
