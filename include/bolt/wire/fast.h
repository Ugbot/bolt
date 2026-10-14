#pragma once
// FAST 1.1 (final 1.x.1, 2006-12-20), §§6.3,10. Partial scalar codec,
// not an XML/template interpreter. Caller owns/reset-scopes dictionaries.
// Input, scalar outputs, dictionary cells, pmap and ASCII output ranges must
// be disjoint. Refusals preserve cursors, destination and dictionary state.
#include "bolt/wire/byte_reader.h"
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace bolt::wire::fast {
enum class Result : std::uint8_t { Ok, Truncated, Overflow, Overlong, Capacity, Undefined, Invalid };
inline constexpr std::uint32_t kIntegerBytes = 10, kPresenceBytes = 16;
// FAST 1.1 §5: decimal exponent format range, not a capacity clamp.
inline constexpr std::int32_t kMinExponent = -63, kMaxExponent = 63;
template<class T> struct Value { T value{}; bool present = false; };
enum class State : std::uint8_t { Undefined, Empty, Assigned };
template<class T> struct Cell { T value{}; State state = State::Undefined; };
struct Decimal { std::int64_t mantissa{}; std::int32_t exponent{}; };
enum class Operator : std::uint8_t { None, Constant, Default, Copy, Increment, Delta };
struct Entity { std::uint64_t bits{}; std::uint8_t first{}, second{}, count{}; };
inline Result entity(ByteReader& reader, Entity& out) noexcept {
    assert(reader.offset <= reader.size); assert(kIntegerBytes == 10);
    if (!byte_reader_valid(reader)) return Result::Invalid;
    auto r = reader; Entity v{};
    for (std::uint32_t i = 0; i < kIntegerBytes; ++i) {
        if (r.offset == r.size) return Result::Truncated;
        const auto b = r.data[r.offset++];
        if (i == 0) v.first = b & 127u;
        if (i == 1) v.second = b & 127u;
        v.bits = (v.bits << 7u) | (b & 127u); ++v.count;
        if ((b & 128u) != 0) { reader = r; out = v; return Result::Ok; }
    }
    return Result::Overflow;
}
inline Result unsigned_integer(ByteReader& reader, bool optional, Value<std::uint64_t>& out) noexcept {
    assert(reader.offset <= reader.size); assert(sizeof(std::uint64_t) == 8);
    auto r = reader; Entity e{}; auto status = entity(r,e); if (status != Result::Ok) return status;
    if (e.count > 1 && e.first == 0) return Result::Overlong;
    const bool wide = e.count == 10 && e.first == 2 && e.bits == 0;
    if (e.count == 10 && e.first > 1 && !(optional && wide)) return Result::Overflow;
    Value<std::uint64_t> value{};
    value.present = !optional || e.bits != 0 || wide;
    value.value = optional && value.present ? e.bits-1u : e.bits;
    reader = r; out = value; return Result::Ok;
}
inline Result signed_integer(ByteReader& reader, bool optional, Value<std::int64_t>& out) noexcept {
    assert(reader.offset <= reader.size); assert(sizeof(std::int64_t) == 8);
    auto r = reader; Entity e{}; auto status = entity(r,e); if (status != Result::Ok) return status;
    const bool negative = (e.first & 64u) != 0;
    if (e.count > 1 && ((e.first == 0 && (e.second & 64u) == 0) ||
        (e.first == 127 && (e.second & 64u) != 0))) return Result::Overlong;
    const bool nullable_max = optional && e.count == 10 && e.first == 1 && e.bits == (1ull << 63);
    if (e.count == 10 && ((!negative && (e.first != 0 || (e.bits >> 63) != 0) && !nullable_max) ||
        (negative && (e.first != 127 || (e.bits >> 63) == 0)))) return Result::Overflow;
    if (negative && e.count < 10) e.bits |= UINT64_MAX << (7u*e.count);
    Value<std::int64_t> value{}; value.present = !optional || e.bits != 0;
    if (optional && value.present && !negative) --e.bits;
    value.value = std::bit_cast<std::int64_t>(e.bits);
    reader = r; out = value; return Result::Ok;
}
template<class T> inline Result integer(ByteReader& reader, bool optional, Value<T>& out) noexcept {
    static_assert(std::is_integral_v<T> && sizeof(T) <= 8);
    assert(reader.offset <= reader.size); assert(sizeof(T) == 4 || sizeof(T) == 8);
    auto r = reader;
    using Wide = std::conditional_t<std::is_signed_v<T>,std::int64_t,std::uint64_t>;
    Value<Wide> v{};
    Result status{};
    if constexpr (std::is_signed_v<T>) status = signed_integer(r,optional,v);
    else status = unsigned_integer(r,optional,v);
    if (status != Result::Ok) return status;
    if (v.present && (v.value < std::numeric_limits<T>::min() || v.value > std::numeric_limits<T>::max())) return Result::Overflow;
    reader = r; out = {static_cast<T>(v.value),v.present}; return Result::Ok;
}
struct Presence {
    const std::uint8_t* bytes{}; std::uint32_t size{}, used{};
    bool next() noexcept {
        assert(size <= kPresenceBytes); assert(used < UINT32_MAX);
        const auto bit = used++;
        return bit/7u < size && (bytes[bit/7u] & (64u >> (bit%7u))) != 0;
    }
    bool exhausted() const noexcept {
        assert(size <= kPresenceBytes); assert(bytes != nullptr);
        for (auto bit = used; bit < size*7u; ++bit) if ((bytes[bit/7u] & (64u >> (bit%7u))) != 0) return false;
        return true;
    }
};
inline Result presence(ByteReader& reader, Presence& out) noexcept {
    assert(reader.offset <= reader.size); assert(kPresenceBytes > 0);
    if (!byte_reader_valid(reader)) return Result::Invalid;
    auto r = reader; const auto start = r.offset;
    for (std::uint32_t i = 0; i < kPresenceBytes; ++i) {
        if (r.offset == r.size) return Result::Truncated;
        const auto b = r.data[r.offset++];
        if ((b & 128u) != 0) {
            if (i != 0 && (b & 127u) == 0) return Result::Overlong;
            out = {r.data+start,i+1,0}; reader = r; return Result::Ok;
        }
    }
    return Result::Capacity;
}
// Decodes <=capacity ASCII characters; no string dictionary or tail/delta codec.
// Caller storage is modified only after complete bounded entity validation.
inline Result ascii(ByteReader& reader, bool optional, char* output, std::uint32_t capacity,
    Value<std::uint32_t>& out) noexcept {
    assert(reader.offset <= reader.size); assert(output != nullptr || capacity == 0);
    if (!byte_reader_valid(reader) || (output == nullptr && capacity != 0)) return Result::Invalid;
    auto r = reader; const auto start = r.offset;
    for (std::uint64_t i = 0; i <= std::uint64_t(capacity)+2; ++i) {
        if (r.offset == r.size) return Result::Truncated;
        const auto b = r.data[r.offset++]; if ((b & 128u) == 0) continue;
        auto begin = start; auto count = r.offset-start;
        const bool present = !(optional && count == 1 && (b & 127u) == 0);
        if (present && optional && (r.data[begin] & 127u) == 0) {
            ++begin; --count;
            if (count != 0 && (r.data[begin] & 127u) != 0) return Result::Overlong;
        }
        if (present && count != 0 && (r.data[begin] & 127u) == 0) {
            ++begin; --count;
            if (count != 0 && (r.data[begin] & 127u) != 0) return Result::Overlong;
        }
        if (!present) count = 0;
        if (count > capacity) return Result::Capacity;
        for (std::uint32_t j = 0; j < count; ++j) output[j] = static_cast<char>(r.data[begin+j] & 127u);
        out = {count,present}; reader = r; return Result::Ok;
    }
    return Result::Capacity;
}
inline Result decimal(ByteReader& reader, bool optional, Value<Decimal>& out) noexcept {
    assert(reader.offset <= reader.size); assert(kMinExponent == -63);
    auto r = reader; Value<std::int32_t> exponent{}; Value<std::int64_t> mantissa{};
    auto status = integer(r,optional,exponent); if (status != Result::Ok) return status;
    if (exponent.present) {
        if (exponent.value < kMinExponent || exponent.value > kMaxExponent) return Result::Overflow;
        status = integer(r,false,mantissa); if (status != Result::Ok) return status;
    }
    out = {{mantissa.value,exponent.value},exponent.present}; reader = r; return Result::Ok;
}
template<class T> inline void reset(Cell<T>* cells, std::uint32_t count) noexcept {
    assert(cells != nullptr || count == 0); assert(count <= UINT32_MAX/sizeof(Cell<T>));
    for (std::uint32_t i = 0; i < count; ++i) cells[i] = {};
}
template<class T> inline Result copied(Cell<T>& cell, bool optional, Value<T> initial, bool increment,
    Value<T>& out) noexcept {
    assert(cell.state <= State::Assigned); assert(!increment || std::is_integral_v<T>);
    auto next = cell;
    if (next.state == State::Undefined) {
        if (!initial.present && !optional) return Result::Undefined;
        next = {initial.value,initial.present ? State::Assigned : State::Empty};
    } else if (next.state == State::Empty && !optional) return Result::Undefined;
    else if (next.state == State::Assigned && increment) {
        if constexpr (std::is_integral_v<T>) {
            // FAST increment wraps at the declared field width (§6.3.6).
            using U = std::make_unsigned_t<T>;
            next.value = std::bit_cast<T>(static_cast<U>(static_cast<U>(next.value)+U{1}));
        }
    }
    out = {next.value,next.state == State::Assigned}; cell = next; return Result::Ok;
}
template<class T> inline bool add_delta(T base, std::int64_t delta, T& out) noexcept {
    static_assert(std::is_integral_v<T>);
    assert(sizeof(T) <= 8); assert(sizeof(delta) == 8);
    if constexpr (std::is_signed_v<T>) {
        if ((delta > 0 && std::int64_t(base) > INT64_MAX-delta) ||
            (delta < 0 && std::int64_t(base) < INT64_MIN-delta)) return false;
        const auto value = std::int64_t(base)+delta;
        if (value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max()) return false;
        out = static_cast<T>(value);
    } else {
        const auto magnitude = delta < 0 ? std::uint64_t(-(delta+1))+1u : std::uint64_t(delta);
        if (delta < 0 ? std::uint64_t(base) < magnitude : magnitude > std::numeric_limits<T>::max()-std::uint64_t(base)) return false;
        out = static_cast<T>(delta < 0 ? std::uint64_t(base)-magnitude : std::uint64_t(base)+magnitude);
    }
    return true;
}
template<class T> inline Result field(ByteReader& reader, Presence& pmap, Operator op, bool optional,
    Value<T> initial, Cell<T>& cell, Value<T>& out) noexcept {
    assert(reader.offset <= reader.size); assert(op <= Operator::Delta);
    if (op == Operator::Constant && !initial.present) return Result::Undefined;
    auto r = reader; auto p = pmap; auto next = cell; Value<T> value{}; Result status = Result::Ok;
    if (op == Operator::Delta) {
        Value<std::int64_t> delta{}; status = integer(r,optional,delta);
        if (status == Result::Ok && delta.present) {
            if (cell.state == State::Empty) return Result::Undefined;
            const auto base = cell.state == State::Assigned ? cell.value : initial.present ? initial.value : T{};
            if (!add_delta(base,delta.value,value.value)) return Result::Overflow;
            value.present = true; next = {value.value,State::Assigned};
        }
    } else if (op == Operator::None || (op != Operator::Constant && p.next())) {
        status = integer(r,optional,value);
        if (status == Result::Ok && (op == Operator::Copy || op == Operator::Increment))
            next = {value.value,value.present ? State::Assigned : State::Empty};
    } else if (op == Operator::Constant) value = (!optional || p.next()) ? initial : Value<T>{};
    else if (op == Operator::Default) { value = initial; if (!value.present && !optional) return Result::Undefined; }
    else status = copied(next,optional,initial,op == Operator::Increment,value);
    if (status != Result::Ok) return status;
    reader = r; pmap = p; cell = next; out = value; return Result::Ok;
}
// Decimal delta subset: no XML initial value; undefined base is exactly0/0.
inline Result decimal_delta(ByteReader& reader, bool optional, Cell<Decimal>& cell, Value<Decimal>& out) noexcept {
    assert(reader.offset <= reader.size); assert(cell.state <= State::Assigned);
    auto r = reader; Value<std::int32_t> exponent{}; Value<std::int64_t> mantissa{}; Value<Decimal> v{};
    auto status = integer(r,optional,exponent); if (status != Result::Ok) return status;
    if (exponent.present) {
        if (cell.state == State::Empty) return Result::Undefined;
        status = integer(r,false,mantissa); if (status != Result::Ok) return status;
        const auto base = cell.state == State::Assigned ? cell.value : Decimal{};
        if (!add_delta(base.exponent,exponent.value,v.value.exponent) || !add_delta(base.mantissa,mantissa.value,v.value.mantissa) ||
            v.value.exponent < kMinExponent || v.value.exponent > kMaxExponent) return Result::Overflow;
        v.present = true;
    }
    if (v.present) cell = {v.value,State::Assigned};
    reader = r; out = v; return Result::Ok;
}
inline Result decimal_field(ByteReader& reader, Presence& pmap, Operator op, bool optional,
    Value<Decimal> initial, Cell<Decimal>& cell, Value<Decimal>& out) noexcept {
    assert(reader.offset <= reader.size); assert(op <= Operator::Delta);
    if (op == Operator::Constant && !initial.present) return Result::Undefined;
    if (op == Operator::Delta)
        return initial.present ? Result::Invalid : decimal_delta(reader,optional,cell,out);
    if (op == Operator::Increment) return Result::Invalid;
    auto r = reader; auto p = pmap; auto next = cell; Value<Decimal> value{}; Result status = Result::Ok;
    if (op == Operator::None || (op != Operator::Constant && p.next())) {
        status = decimal(r,optional,value);
        if (status == Result::Ok && op == Operator::Copy) next = {value.value,value.present ? State::Assigned : State::Empty};
    } else if (op == Operator::Constant) value = (!optional || p.next()) ? initial : Value<Decimal>{};
    else if (op == Operator::Default) { value = initial; if (!value.present && !optional) return Result::Undefined; }
    else status = copied(next,optional,initial,false,value);
    if (status != Result::Ok) return status;
    reader = r; pmap = p; cell = next; out = value; return Result::Ok;
}
} // namespace bolt::wire::fast
