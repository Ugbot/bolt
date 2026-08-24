// ---------------------------------------------------------------------------
// bolt::parse::json — byte-faithful JSON writer / value builder impl.
//
// The write side of bolt::parse::json (see bolt_json_writer.h). Mirrors the
// reader's style: arena-only, noexcept, bounded, compile-time-dispatched
// sinks so one recursive walk serves both the length-count pass and the
// byte-write pass with zero runtime branching between them.
// ---------------------------------------------------------------------------
#include "bolt/parse/bolt_json_writer.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bolt {
namespace parse {
namespace json {

namespace {

// ---------------------------------------------------------------------------
// Node allocation helpers.
// ---------------------------------------------------------------------------
JsonValue* alloc_value(Arena* arena, ValueKind kind) noexcept {
    assert(arena != nullptr);
    JsonValue* v = arena->allocate_array<JsonValue>(1);
    if (v == nullptr) return nullptr;
    v->kind  = kind;
    v->_pad[0] = 0; v->_pad[1] = 0; v->_pad[2] = 0; v->_pad[3] = 0;
    v->_pad[4] = 0; v->_pad[5] = 0; v->_pad[6] = 0;
    v->bytes = nullptr;
    v->len   = 0;
    v->count = 0;
    v->head  = nullptr;
    v->tail  = nullptr;
    return v;
}

// Append a pre-built node (key may be nullptr for arrays). The key bytes are
// referenced as-is — callers that need a copy do it before calling.
bool container_append(Arena* arena, JsonValue* container,
                      const char* key, int32_t key_len,
                      JsonValue* val) noexcept {
    assert(arena != nullptr);
    assert(container != nullptr);
    if (val == nullptr) return false;
    JsonNode* node = arena->allocate_array<JsonNode>(1);
    if (node == nullptr) return false;
    node->next    = nullptr;
    node->key     = key;
    node->key_len = key_len;
    node->_pad    = 0;
    node->value   = val;
    if (container->tail == nullptr) {
        container->head = node;
        container->tail = node;
    } else {
        container->tail->next = node;
        container->tail = node;
    }
    ++container->count;
    return true;
}

// Signed 64-bit -> decimal ASCII. Returns the byte count [1..20]. Handles
// INT64_MIN via unsigned magnitude. Writes into `out` (>= 20 bytes).
int32_t format_int64(char* out, int64_t v) noexcept {
    assert(out != nullptr);
    char tmp[20];
    int32_t n = 0;
    uint64_t mag = (v < 0) ? (~static_cast<uint64_t>(v) + 1u)
                           : static_cast<uint64_t>(v);
    do {
        tmp[n++] = static_cast<char>('0' + static_cast<int>(mag % 10u));
        mag /= 10u;
    } while (mag != 0u);
    int32_t w = 0;
    if (v < 0) out[w++] = '-';
    for (int32_t i = n - 1; i >= 0; --i) out[w++] = tmp[i];
    assert(w >= 1 && w <= 20);
    return w;
}

// ---------------------------------------------------------------------------
// dtoa_impl — SHORTEST-round-trip double formatting (Grisu2).
//
// A faithful port of nlohmann::json's `detail::dtoa_impl` + `detail::to_chars`
// (MIT-licensed; nlohmann/json v3.11.2), specialised to IEEE-754 `double`.
// Reproducing nlohmann's dtoa BYTE-FOR-BYTE is the whole point of STATION-87:
// the deferred consumers dump floats via this exact algorithm, so anything but
// a port (e.g. "%.17g") would silently diverge (0.1 -> "0.10000000000000001").
// Only the `double` path is kept (kPrecision=53); the algorithm and the
// cached-power table are otherwise unchanged from upstream. Style follows the
// surrounding file: no exceptions, asserts for invariants.
// ---------------------------------------------------------------------------
namespace dtoa_impl {

struct diyfp {  // f * 2^e
    uint64_t f;
    int      e;

    static diyfp sub(const diyfp& x, const diyfp& y) noexcept {
        assert(x.e == y.e && x.f >= y.f);
        return {x.f - y.f, x.e};
    }

    static diyfp mul(const diyfp& x, const diyfp& y) noexcept {
        const uint64_t u_lo = x.f & 0xFFFFFFFFu;
        const uint64_t u_hi = x.f >> 32u;
        const uint64_t v_lo = y.f & 0xFFFFFFFFu;
        const uint64_t v_hi = y.f >> 32u;
        const uint64_t p0 = u_lo * v_lo;
        const uint64_t p1 = u_lo * v_hi;
        const uint64_t p2 = u_hi * v_lo;
        const uint64_t p3 = u_hi * v_hi;
        const uint64_t p0_hi = p0 >> 32u;
        const uint64_t p1_lo = p1 & 0xFFFFFFFFu;
        const uint64_t p1_hi = p1 >> 32u;
        const uint64_t p2_lo = p2 & 0xFFFFFFFFu;
        const uint64_t p2_hi = p2 >> 32u;
        uint64_t Q = p0_hi + p1_lo + p2_lo;
        Q += uint64_t{1} << (64u - 32u - 1u);  // round, ties up
        const uint64_t h = p3 + p2_hi + p1_hi + (Q >> 32u);
        return {h, x.e + y.e + 64};
    }

    static diyfp normalize(diyfp x) noexcept {
        assert(x.f != 0);
        while ((x.f >> 63u) == 0) { x.f <<= 1u; x.e--; }
        return x;
    }

    static diyfp normalize_to(const diyfp& x, const int target_exponent) noexcept {
        const int delta = x.e - target_exponent;
        assert(delta >= 0);
        assert(((x.f << delta) >> delta) == x.f);
        return {x.f << delta, target_exponent};
    }
};

struct boundaries {
    diyfp w;
    diyfp minus;
    diyfp plus;
};

// Compute the normalized diyfp of `value` (finite, > 0) and its boundaries.
inline boundaries compute_boundaries(double value) noexcept {
    // double: p = 53 (incl hidden bit), bias = 1023 + 52 = 1075.
    constexpr int      kPrecision = 53;
    constexpr int      kBias      = 1075;
    constexpr int      kMinExp    = 1 - kBias;
    constexpr uint64_t kHiddenBit = uint64_t{1} << (kPrecision - 1);

    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint64_t E = bits >> (kPrecision - 1);
    const uint64_t F = bits & (kHiddenBit - 1);

    const bool  is_denormal = (E == 0);
    const diyfp v = is_denormal
                        ? diyfp{F, kMinExp}
                        : diyfp{F + kHiddenBit, static_cast<int>(E) - kBias};

    const bool  lower_boundary_is_closer = (F == 0 && E > 1);
    const diyfp m_plus  = diyfp{2 * v.f + 1, v.e - 1};
    const diyfp m_minus = lower_boundary_is_closer
                              ? diyfp{4 * v.f - 1, v.e - 2}
                              : diyfp{2 * v.f - 1, v.e - 1};

    const diyfp w_plus  = diyfp::normalize(m_plus);
    const diyfp w_minus = diyfp::normalize_to(m_minus, w_plus.e);
    return {diyfp::normalize(v), w_minus, w_plus};
}

constexpr int kAlpha = -60;
constexpr int kGamma = -32;

struct cached_power {  // c = f * 2^e ~= 10^k
    uint64_t f;
    int      e;
    int      k;
};

inline cached_power get_cached_power_for_binary_exponent(int e) noexcept {
    constexpr int kCachedPowersMinDecExp = -300;
    constexpr int kCachedPowersDecStep   = 8;
    static const cached_power kCachedPowers[79] = {
        { 0xAB70FE17C79AC6CAull, -1060, -300 }, { 0xFF77B1FCBEBCDC4Full, -1034, -292 },
        { 0xBE5691EF416BD60Cull, -1007, -284 }, { 0x8DD01FAD907FFC3Cull,  -980, -276 },
        { 0xD3515C2831559A83ull,  -954, -268 }, { 0x9D71AC8FADA6C9B5ull,  -927, -260 },
        { 0xEA9C227723EE8BCBull,  -901, -252 }, { 0xAECC49914078536Dull,  -874, -244 },
        { 0x823C12795DB6CE57ull,  -847, -236 }, { 0xC21094364DFB5637ull,  -821, -228 },
        { 0x9096EA6F3848984Full,  -794, -220 }, { 0xD77485CB25823AC7ull,  -768, -212 },
        { 0xA086CFCD97BF97F4ull,  -741, -204 }, { 0xEF340A98172AACE5ull,  -715, -196 },
        { 0xB23867FB2A35B28Eull,  -688, -188 }, { 0x84C8D4DFD2C63F3Bull,  -661, -180 },
        { 0xC5DD44271AD3CDBAull,  -635, -172 }, { 0x936B9FCEBB25C996ull,  -608, -164 },
        { 0xDBAC6C247D62A584ull,  -582, -156 }, { 0xA3AB66580D5FDAF6ull,  -555, -148 },
        { 0xF3E2F893DEC3F126ull,  -529, -140 }, { 0xB5B5ADA8AAFF80B8ull,  -502, -132 },
        { 0x87625F056C7C4A8Bull,  -475, -124 }, { 0xC9BCFF6034C13053ull,  -449, -116 },
        { 0x964E858C91BA2655ull,  -422, -108 }, { 0xDFF9772470297EBDull,  -396, -100 },
        { 0xA6DFBD9FB8E5B88Full,  -369,  -92 }, { 0xF8A95FCF88747D94ull,  -343,  -84 },
        { 0xB94470938FA89BCFull,  -316,  -76 }, { 0x8A08F0F8BF0F156Bull,  -289,  -68 },
        { 0xCDB02555653131B6ull,  -263,  -60 }, { 0x993FE2C6D07B7FACull,  -236,  -52 },
        { 0xE45C10C42A2B3B06ull,  -210,  -44 }, { 0xAA242499697392D3ull,  -183,  -36 },
        { 0xFD87B5F28300CA0Eull,  -157,  -28 }, { 0xBCE5086492111AEBull,  -130,  -20 },
        { 0x8CBCCC096F5088CCull,  -103,  -12 }, { 0xD1B71758E219652Cull,   -77,   -4 },
        { 0x9C40000000000000ull,   -50,    4 }, { 0xE8D4A51000000000ull,   -24,   12 },
        { 0xAD78EBC5AC620000ull,     3,   20 }, { 0x813F3978F8940984ull,    30,   28 },
        { 0xC097CE7BC90715B3ull,    56,   36 }, { 0x8F7E32CE7BEA5C70ull,    83,   44 },
        { 0xD5D238A4ABE98068ull,   109,   52 }, { 0x9F4F2726179A2245ull,   136,   60 },
        { 0xED63A231D4C4FB27ull,   162,   68 }, { 0xB0DE65388CC8ADA8ull,   189,   76 },
        { 0x83C7088E1AAB65DBull,   216,   84 }, { 0xC45D1DF942711D9Aull,   242,   92 },
        { 0x924D692CA61BE758ull,   269,  100 }, { 0xDA01EE641A708DEAull,   295,  108 },
        { 0xA26DA3999AEF774Aull,   322,  116 }, { 0xF209787BB47D6B85ull,   348,  124 },
        { 0xB454E4A179DD1877ull,   375,  132 }, { 0x865B86925B9BC5C2ull,   402,  140 },
        { 0xC83553C5C8965D3Dull,   428,  148 }, { 0x952AB45CFA97A0B3ull,   455,  156 },
        { 0xDE469FBD99A05FE3ull,   481,  164 }, { 0xA59BC234DB398C25ull,   508,  172 },
        { 0xF6C69A72A3989F5Cull,   534,  180 }, { 0xB7DCBF5354E9BECEull,   561,  188 },
        { 0x88FCF317F22241E2ull,   588,  196 }, { 0xCC20CE9BD35C78A5ull,   614,  204 },
        { 0x98165AF37B2153DFull,   641,  212 }, { 0xE2A0B5DC971F303Aull,   667,  220 },
        { 0xA8D9D1535CE3B396ull,   694,  228 }, { 0xFB9B7CD9A4A7443Cull,   720,  236 },
        { 0xBB764C4CA7A44410ull,   747,  244 }, { 0x8BAB8EEFB6409C1Aull,   774,  252 },
        { 0xD01FEF10A657842Cull,   800,  260 }, { 0x9B10A4E5E9913129ull,   827,  268 },
        { 0xE7109BFBA19C0C9Dull,   853,  276 }, { 0xAC2820D9623BF429ull,   880,  284 },
        { 0x80444B5E7AA7CF85ull,   907,  292 }, { 0xBF21E44003ACDD2Dull,   933,  300 },
        { 0x8E679C2F5E44FF8Full,   960,  308 }, { 0xD433179D9C8CB841ull,   986,  316 },
        { 0x9E19DB92B4E31BA9ull,  1013,  324 },
    };
    assert(e >= -1500 && e <= 1500);
    const int f = kAlpha - e - 1;
    const int k = (f * 78913) / (1 << 18) + static_cast<int>(f > 0);
    const int index = (-kCachedPowersMinDecExp + k + (kCachedPowersDecStep - 1)) / kCachedPowersDecStep;
    assert(index >= 0 && index < 79);
    const cached_power cached = kCachedPowers[index];
    assert(kAlpha <= cached.e + e + 64 && kGamma >= cached.e + e + 64);
    return cached;
}

inline int find_largest_pow10(const uint32_t n, uint32_t& pow10) noexcept {
    if (n >= 1000000000) { pow10 = 1000000000; return 10; }
    if (n >= 100000000)  { pow10 = 100000000;  return  9; }
    if (n >= 10000000)   { pow10 = 10000000;   return  8; }
    if (n >= 1000000)    { pow10 = 1000000;    return  7; }
    if (n >= 100000)     { pow10 = 100000;     return  6; }
    if (n >= 10000)      { pow10 = 10000;      return  5; }
    if (n >= 1000)       { pow10 = 1000;       return  4; }
    if (n >= 100)        { pow10 = 100;        return  3; }
    if (n >= 10)         { pow10 = 10;         return  2; }
    pow10 = 1;                                 return  1;
}

inline void grisu2_round(char* buf, int len, uint64_t dist, uint64_t delta,
                         uint64_t rest, uint64_t ten_k) noexcept {
    assert(len >= 1 && dist <= delta && rest <= delta && ten_k > 0);
    while (rest < dist
           && delta - rest >= ten_k
           && (rest + ten_k < dist || dist - rest > rest + ten_k - dist)) {
        assert(buf[len - 1] != '0');
        buf[len - 1]--;
        rest += ten_k;
    }
}

inline void grisu2_digit_gen(char* buffer, int& length, int& decimal_exponent,
                             diyfp M_minus, diyfp w, diyfp M_plus) noexcept {
    assert(M_plus.e >= kAlpha && M_plus.e <= kGamma);
    uint64_t delta = diyfp::sub(M_plus, M_minus).f;
    uint64_t dist  = diyfp::sub(M_plus, w).f;

    const diyfp one{uint64_t{1} << -M_plus.e, M_plus.e};
    auto     p1 = static_cast<uint32_t>(M_plus.f >> -one.e);
    uint64_t p2 = M_plus.f & (one.f - 1);

    assert(p1 > 0);
    uint32_t  pow10{};
    const int k = find_largest_pow10(p1, pow10);

    int n = k;
    while (n > 0) {
        const uint32_t d = p1 / pow10;
        const uint32_t r = p1 % pow10;
        assert(d <= 9);
        buffer[length++] = static_cast<char>('0' + d);
        p1 = r;
        n--;
        const uint64_t rest = (uint64_t{p1} << -one.e) + p2;
        if (rest <= delta) {
            decimal_exponent += n;
            const uint64_t ten_n = uint64_t{pow10} << -one.e;
            grisu2_round(buffer, length, dist, delta, rest, ten_n);
            return;
        }
        pow10 /= 10;
    }

    assert(p2 > delta);
    int m = 0;
    for (;;) {
        assert(p2 <= (UINT64_MAX) / 10);
        p2 *= 10;
        const uint64_t d = p2 >> -one.e;
        const uint64_t r = p2 & (one.f - 1);
        assert(d <= 9);
        buffer[length++] = static_cast<char>('0' + d);
        p2 = r;
        m++;
        delta *= 10;
        dist  *= 10;
        if (p2 <= delta) break;
    }

    decimal_exponent -= m;
    const uint64_t ten_m = one.f;
    grisu2_round(buffer, length, dist, delta, p2, ten_m);
}

inline void grisu2(char* buf, int& len, int& decimal_exponent, double value) noexcept {
    const boundaries w = compute_boundaries(value);
    const cached_power cached = get_cached_power_for_binary_exponent(w.plus.e);
    const diyfp c_minus_k{cached.f, cached.e};
    const diyfp W       = diyfp::mul(w.w,     c_minus_k);
    const diyfp W_minus = diyfp::mul(w.minus, c_minus_k);
    const diyfp W_plus  = diyfp::mul(w.plus,  c_minus_k);
    const diyfp M_minus{W_minus.f + 1, W_minus.e};
    const diyfp M_plus {W_plus.f  - 1, W_plus.e };
    decimal_exponent = -cached.k;
    grisu2_digit_gen(buf, len, decimal_exponent, M_minus, W, M_plus);
}

inline char* append_exponent(char* buf, int e) noexcept {
    assert(e > -1000 && e < 1000);
    if (e < 0) { e = -e; *buf++ = '-'; }
    else       { *buf++ = '+'; }
    auto k = static_cast<uint32_t>(e);
    if (k < 10) {
        *buf++ = '0';
        *buf++ = static_cast<char>('0' + k);
    } else if (k < 100) {
        *buf++ = static_cast<char>('0' + k / 10); k %= 10;
        *buf++ = static_cast<char>('0' + k);
    } else {
        *buf++ = static_cast<char>('0' + k / 100); k %= 100;
        *buf++ = static_cast<char>('0' + k / 10);  k %= 10;
        *buf++ = static_cast<char>('0' + k);
    }
    return buf;
}

inline char* format_buffer(char* buf, int len, int decimal_exponent,
                           int min_exp, int max_exp) noexcept {
    assert(min_exp < 0 && max_exp > 0);
    const int k = len;
    const int n = len + decimal_exponent;

    if (k <= n && n <= max_exp) {  // digits[000].0
        memset(buf + k, '0', static_cast<size_t>(n) - static_cast<size_t>(k));
        buf[n + 0] = '.';
        buf[n + 1] = '0';
        return buf + (static_cast<size_t>(n) + 2);
    }
    if (0 < n && n <= max_exp) {   // dig.its
        assert(k > n);
        memmove(buf + (static_cast<size_t>(n) + 1), buf + n,
                static_cast<size_t>(k) - static_cast<size_t>(n));
        buf[n] = '.';
        return buf + (static_cast<size_t>(k) + 1u);
    }
    if (min_exp < n && n <= 0) {   // 0.[000]digits
        memmove(buf + (2 + static_cast<size_t>(-n)), buf, static_cast<size_t>(k));
        buf[0] = '0';
        buf[1] = '.';
        memset(buf + 2, '0', static_cast<size_t>(-n));
        return buf + (2u + static_cast<size_t>(-n) + static_cast<size_t>(k));
    }
    if (k == 1) {                  // dE+123
        buf += 1;
    } else {                       // d.igitsE+123
        memmove(buf + 2, buf + 1, static_cast<size_t>(k) - 1);
        buf[1] = '.';
        buf += 1 + static_cast<size_t>(k);
    }
    *buf++ = 'e';
    return append_exponent(buf, n - 1);
}

}  // namespace dtoa_impl

// Format a FINITE double into `out` (needs >= 25 bytes) exactly as
// nlohmann::detail::to_chars would, returning the byte count. Caller guarantees
// std::isfinite(value). -0.0 -> "-0.0", 0.0 -> "0.0".
int32_t dtoa_double(char* out, double value) noexcept {
    assert(out != nullptr);
    assert(std::isfinite(value));
    char* first = out;
    if (std::signbit(value)) { value = -value; *first++ = '-'; }
    if (value == 0.0) {  // +-0
        *first++ = '0'; *first++ = '.'; *first++ = '0';
        return static_cast<int32_t>(first - out);
    }
    int len = 0;
    int decimal_exponent = 0;
    dtoa_impl::grisu2(first, len, decimal_exponent, value);
    // Match to_chars: kMinExp = -4, kMaxExp = digits10 = 15 for double.
    char* last = dtoa_impl::format_buffer(first, len, decimal_exponent, -4, 15);
    return static_cast<int32_t>(last - out);
}

// ---------------------------------------------------------------------------
// Emit sinks. Compile-time dispatch: CountSink measures the exact output
// length, WriteSink fills the buffer. Both drive the SAME emit routines, so
// pass-1 length and pass-2 write can never disagree.
// ---------------------------------------------------------------------------
struct CountSink {
    int64_t n;
    BOLT_FORCE_INLINE void put(char) noexcept { ++n; }
    BOLT_FORCE_INLINE void put_bytes(const char*, int32_t k) noexcept { n += k; }
};

struct WriteSink {
    char* p;
    char* end;
    bool  ok;
    BOLT_FORCE_INLINE void put(char c) noexcept {
        if (p < end) { *p++ = c; } else { ok = false; }
    }
    BOLT_FORCE_INLINE void put_bytes(const char* b, int32_t k) noexcept {
        assert(k >= 0);
        if (k > 0 && p + k <= end) { memcpy(p, b, static_cast<size_t>(k)); p += k; }
        else if (k > 0) { ok = false; }
    }
};

// JSON string-escape one byte run into the sink. Both passes call this, so the
// counted and written lengths are identical by construction. Bytes >= 0x80 are
// passed through verbatim (raw UTF-8 is legal in JSON string content).
template <typename Sink>
BOLT_FORCE_INLINE void emit_escaped(Sink& s, const char* b, int32_t len) noexcept {
    assert(b != nullptr || len == 0);
    static const char kHex[] = "0123456789abcdef";
    for (int32_t i = 0; i < len; ++i) {
        const uint8_t c = static_cast<uint8_t>(b[i]);
        switch (c) {
            case '"':  s.put('\\'); s.put('"');  break;
            case '\\': s.put('\\'); s.put('\\'); break;
            case '\b': s.put('\\'); s.put('b');  break;
            case '\f': s.put('\\'); s.put('f');  break;
            case '\n': s.put('\\'); s.put('n');  break;
            case '\r': s.put('\\'); s.put('r');  break;
            case '\t': s.put('\\'); s.put('t');  break;
            default:
                if (c < 0x20) {
                    s.put('\\'); s.put('u'); s.put('0'); s.put('0');
                    s.put(kHex[(c >> 4) & 0xF]);
                    s.put(kHex[c & 0xF]);
                } else {
                    s.put(static_cast<char>(c));
                }
                break;
        }
    }
}

// Recursive value emit. `depth` is bounded by kJsonMaxDepth so a pathological
// (or maliciously deep) tree cannot blow the C stack.
template <typename Sink>
bool emit_value(const JsonValue* v, Sink& s, int32_t depth) noexcept {
    assert(v != nullptr);
    if (depth > kJsonMaxDepth) return false;
    switch (v->kind) {
        case ValueKind::Literal:
            s.put_bytes(v->bytes, v->len);
            return true;
        case ValueKind::StringVerbatim:
            s.put('"');
            s.put_bytes(v->bytes, v->len);
            s.put('"');
            return true;
        case ValueKind::StringEscaped:
            s.put('"');
            emit_escaped(s, v->bytes, v->len);
            s.put('"');
            return true;
        case ValueKind::Object: {
            s.put('{');
            bool first = true;
            for (const JsonNode* m = v->head; m != nullptr; m = m->next) {
                if (!first) s.put(',');
                first = false;
                s.put('"');
                s.put_bytes(m->key, m->key_len);
                s.put('"');
                s.put(':');
                if (!emit_value(m->value, s, depth + 1)) return false;
            }
            s.put('}');
            return true;
        }
        case ValueKind::Array: {
            s.put('[');
            bool first = true;
            for (const JsonNode* e = v->head; e != nullptr; e = e->next) {
                if (!first) s.put(',');
                first = false;
                if (!emit_value(e->value, s, depth + 1)) return false;
            }
            s.put(']');
            return true;
        }
    }
    return false;  // unreachable for a well-formed node
}

// ---------------------------------------------------------------------------
// nlohmann-faithful emit (STATION-87). Separate from emit_value above: it
// escapes object KEYS (raw text), applies ensure_ascii, and pretty-prints —
// none of which the byte-faithful compact path does.
// ---------------------------------------------------------------------------

// Decode one UTF-8 codepoint from b[i..len). Returns bytes consumed (1..4) and
// sets *cp; returns 0 on an invalid or truncated sequence. Rejects overlong
// encodings, surrogates (U+D800..DFFF) and > U+10FFFF, matching a strict
// decoder (nlohmann throws on the same inputs; here the caller fails the emit).
BOLT_FORCE_INLINE int32_t utf8_decode_one(const char* b, int32_t len,
                                          int32_t i, uint32_t* cp) noexcept {
    const uint8_t c0 = static_cast<uint8_t>(b[i]);
    if (c0 < 0x80) { *cp = c0; return 1; }
    auto cont = [&](int32_t j) -> bool {
        return j < len && (static_cast<uint8_t>(b[j]) & 0xC0) == 0x80;
    };
    if ((c0 & 0xE0) == 0xC0) {                       // 2-byte
        if (!cont(i + 1)) return 0;
        const uint32_t v = (uint32_t(c0 & 0x1F) << 6) |
                           (uint32_t(uint8_t(b[i + 1])) & 0x3F);
        if (v < 0x80) return 0;                      // overlong
        *cp = v; return 2;
    }
    if ((c0 & 0xF0) == 0xE0) {                       // 3-byte
        if (!cont(i + 1) || !cont(i + 2)) return 0;
        const uint32_t v = (uint32_t(c0 & 0x0F) << 12) |
                           ((uint32_t(uint8_t(b[i + 1])) & 0x3F) << 6) |
                           (uint32_t(uint8_t(b[i + 2])) & 0x3F);
        if (v < 0x800) return 0;                     // overlong
        if (v >= 0xD800 && v <= 0xDFFF) return 0;     // surrogate
        *cp = v; return 3;
    }
    if ((c0 & 0xF8) == 0xF0) {                       // 4-byte
        if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return 0;
        const uint32_t v = (uint32_t(c0 & 0x07) << 18) |
                           ((uint32_t(uint8_t(b[i + 1])) & 0x3F) << 12) |
                           ((uint32_t(uint8_t(b[i + 2])) & 0x3F) << 6) |
                           (uint32_t(uint8_t(b[i + 3])) & 0x3F);
        if (v < 0x10000 || v > 0x10FFFF) return 0;   // overlong / out of range
        *cp = v; return 4;
    }
    return 0;                                        // stray continuation / 0xF8+
}

// Emit `\u` + 4 lowercase hex digits of a 16-bit value.
template <typename Sink>
BOLT_FORCE_INLINE void emit_u16(Sink& s, uint32_t u) noexcept {
    static const char kHex[] = "0123456789abcdef";
    s.put('\\'); s.put('u');
    s.put(kHex[(u >> 12) & 0xF]);
    s.put(kHex[(u >>  8) & 0xF]);
    s.put(kHex[(u >>  4) & 0xF]);
    s.put(kHex[u & 0xF]);
}

// nlohmann dump_escaped, reproduced for one raw (unescaped) UTF-8 string.
// Returns false only when ensure_ascii is set and the bytes are not valid
// UTF-8 (nlohmann raises type_error 316; a noexcept writer fails instead).
template <typename Sink>
bool emit_escaped_nl(Sink& s, const char* b, int32_t len,
                     bool ensure_ascii) noexcept {
    assert(b != nullptr || len == 0);
    for (int32_t i = 0; i < len; ) {
        const uint8_t c = static_cast<uint8_t>(b[i]);
        if (c < 0x80) {
            switch (c) {
                case '"':  s.put('\\'); s.put('"');  break;
                case '\\': s.put('\\'); s.put('\\'); break;
                case '\b': s.put('\\'); s.put('b');  break;
                case '\f': s.put('\\'); s.put('f');  break;
                case '\n': s.put('\\'); s.put('n');  break;
                case '\r': s.put('\\'); s.put('r');  break;
                case '\t': s.put('\\'); s.put('t');  break;
                default:
                    // Control chars (<=0x1F) always escape; 0x7F escapes only
                    // under ensure_ascii (nlohmann: codepoint >= 0x7F).
                    if (c <= 0x1F || (ensure_ascii && c == 0x7F)) emit_u16(s, c);
                    else                                          s.put(static_cast<char>(c));
                    break;
            }
            ++i;
            continue;
        }
        // c >= 0x80: multibyte UTF-8.
        if (!ensure_ascii) { s.put(static_cast<char>(c)); ++i; continue; }
        uint32_t cp = 0;
        const int32_t n = utf8_decode_one(b, len, i, &cp);
        if (n == 0) return false;  // invalid UTF-8 under ensure_ascii
        if (cp <= 0xFFFF) {
            emit_u16(s, cp);
        } else {
            // UTF-16 surrogate pair (nlohmann: 0xD7C0 + (cp>>10), 0xDC00 + low10)
            emit_u16(s, 0xD7C0u + (cp >> 10u));
            emit_u16(s, 0xDC00u + (cp & 0x3FFu));
        }
        i += n;
    }
    return true;
}

template <typename Sink>
BOLT_FORCE_INLINE void emit_indent(Sink& s, char ch, int32_t count) noexcept {
    for (int32_t i = 0; i < count; ++i) s.put(ch);
}

// Recursive nlohmann-faithful emit. `pretty` == (opts.indent >= 0);
// `cur_indent` is the count of indent chars at this level (nlohmann's
// current_indent). Returns false on depth overflow / invalid-UTF-8 escape.
template <typename Sink>
bool emit_faithful(const JsonValue* v, Sink& s, const EmitOptions& opts,
                   bool pretty, int32_t cur_indent, int32_t depth) noexcept {
    assert(v != nullptr);
    if (depth > kJsonMaxDepth) return false;
    switch (v->kind) {
        case ValueKind::Literal:
            s.put_bytes(v->bytes, v->len);
            return true;
        case ValueKind::StringVerbatim:
            s.put('"');
            s.put_bytes(v->bytes, v->len);
            s.put('"');
            return true;
        case ValueKind::StringEscaped:
            s.put('"');
            if (!emit_escaped_nl(s, v->bytes, v->len, opts.ensure_ascii)) return false;
            s.put('"');
            return true;
        case ValueKind::Object: {
            if (v->head == nullptr) { s.put('{'); s.put('}'); return true; }
            if (pretty) {
                const int32_t ni = cur_indent + opts.indent;
                s.put('{'); s.put('\n');
                for (const JsonNode* m = v->head; m != nullptr; m = m->next) {
                    emit_indent(s, opts.indent_char, ni);
                    s.put('"');
                    if (!emit_escaped_nl(s, m->key, m->key_len, opts.ensure_ascii)) return false;
                    s.put('"'); s.put(':'); s.put(' ');
                    if (!emit_faithful(m->value, s, opts, true, ni, depth + 1)) return false;
                    if (m->next != nullptr) s.put(',');
                    s.put('\n');
                }
                emit_indent(s, opts.indent_char, cur_indent);
                s.put('}');
            } else {
                s.put('{');
                for (const JsonNode* m = v->head; m != nullptr; m = m->next) {
                    s.put('"');
                    if (!emit_escaped_nl(s, m->key, m->key_len, opts.ensure_ascii)) return false;
                    s.put('"'); s.put(':');
                    if (!emit_faithful(m->value, s, opts, false, cur_indent, depth + 1)) return false;
                    if (m->next != nullptr) s.put(',');
                }
                s.put('}');
            }
            return true;
        }
        case ValueKind::Array: {
            if (v->head == nullptr) { s.put('['); s.put(']'); return true; }
            if (pretty) {
                const int32_t ni = cur_indent + opts.indent;
                s.put('['); s.put('\n');
                for (const JsonNode* e = v->head; e != nullptr; e = e->next) {
                    emit_indent(s, opts.indent_char, ni);
                    if (!emit_faithful(e->value, s, opts, true, ni, depth + 1)) return false;
                    if (e->next != nullptr) s.put(',');
                    s.put('\n');
                }
                emit_indent(s, opts.indent_char, cur_indent);
                s.put(']');
            } else {
                s.put('[');
                for (const JsonNode* e = v->head; e != nullptr; e = e->next) {
                    if (!emit_faithful(e->value, s, opts, false, cur_indent, depth + 1)) return false;
                    if (e->next != nullptr) s.put(',');
                }
                s.put(']');
            }
            return true;
        }
    }
    return false;  // unreachable for a well-formed node
}

// ---------------------------------------------------------------------------
// Tape -> value tree. Walks the pre-order token stream with a shared cursor.
// ---------------------------------------------------------------------------
bool build_from_tape(const StructuralIndex* idx, int32_t* cur,
                     Arena* arena, int32_t depth, JsonValue** out) noexcept {
    assert(idx != nullptr);
    assert(cur != nullptr);
    if (depth > kJsonMaxDepth) return false;
    if (*cur < 0 || *cur >= idx->token_count) return false;

    const char* src = reinterpret_cast<const char*>(idx->src);
    const Token& t = idx->tokens[*cur];
    switch (t.type) {
        case TokenType::BeginObject: {
            ++*cur;
            JsonValue* obj = make_object(arena);
            if (obj == nullptr) return false;
            while (true) {
                if (*cur >= idx->token_count) return false;
                const Token& kt = idx->tokens[*cur];
                if (kt.type == TokenType::EndObject) { ++*cur; break; }
                if (kt.type != TokenType::Key) return false;
                const char* key = src + kt.start;
                const int32_t key_len = kt.length;
                ++*cur;
                JsonValue* val = nullptr;
                if (!build_from_tape(idx, cur, arena, depth + 1, &val)) return false;
                // Reference the source key bytes directly (zero-copy, verbatim).
                if (!container_append(arena, obj, key, key_len, val)) return false;
            }
            *out = obj;
            return true;
        }
        case TokenType::BeginArray: {
            ++*cur;
            JsonValue* arr = make_array(arena);
            if (arr == nullptr) return false;
            while (true) {
                if (*cur >= idx->token_count) return false;
                if (idx->tokens[*cur].type == TokenType::EndArray) { ++*cur; break; }
                JsonValue* val = nullptr;
                if (!build_from_tape(idx, cur, arena, depth + 1, &val)) return false;
                if (!container_append(arena, arr, nullptr, 0, val)) return false;
            }
            *out = arr;
            return true;
        }
        case TokenType::String: {
            JsonValue* v = make_string_raw(arena, src + t.start, t.length);
            if (v == nullptr) return false;
            ++*cur;
            *out = v;
            return true;
        }
        case TokenType::Int64:
        case TokenType::Float64:
        case TokenType::BoolTrue:
        case TokenType::BoolFalse:
        case TokenType::Null: {
            // Every scalar keyword/number token slices its exact source bytes
            // ("true"/"false"/"null"/the raw number) — emit them verbatim.
            JsonValue* v = make_literal(arena, src + t.start, t.length);
            if (v == nullptr) return false;
            ++*cur;
            *out = v;
            return true;
        }
        case TokenType::Key:
        case TokenType::EndObject:
        case TokenType::EndArray:
        case TokenType::End:
            return false;  // not a value start
    }
    return false;
}

}  // namespace

// ===========================================================================
// Constructors.
// ===========================================================================
JsonValue* make_null(Arena* arena) noexcept {
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = "null";
    v->len   = 4;
    return v;
}

JsonValue* make_bool(Arena* arena, bool b) noexcept {
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = b ? "true" : "false";
    v->len   = b ? 4 : 5;
    return v;
}

JsonValue* make_int(Arena* arena, int64_t x) noexcept {
    assert(arena != nullptr);
    char tmp[20];
    const int32_t n = format_int64(tmp, x);
    char* copy = static_cast<char*>(arena->copy_into(tmp, static_cast<size_t>(n), 1));
    if (copy == nullptr) return nullptr;
    return make_literal(arena, copy, n);
}

JsonValue* make_double(Arena* arena, double x) noexcept {
    assert(arena != nullptr);
    if (!std::isfinite(x)) return make_null(arena);  // NaN/Inf -> null (valid JSON)
    // SHORTEST-round-trip (Grisu2) — reproduces nlohmann's dump() float bytes.
    // to_chars needs at most 1(sign)+17(digits)+... ; 32 is comfortably ample.
    char tmp[32];
    const int32_t n = dtoa_double(tmp, x);
    assert(n > 0 && n < static_cast<int32_t>(sizeof(tmp)));
    char* copy = static_cast<char*>(arena->copy_into(tmp, static_cast<size_t>(n), 1));
    if (copy == nullptr) return nullptr;
    return make_literal(arena, copy, n);
}

JsonValue* make_string(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::StringEscaped);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_string_raw(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::StringVerbatim);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_literal(Arena* arena, const char* bytes, int32_t len) noexcept {
    if (len < 0) return nullptr;
    JsonValue* v = alloc_value(arena, ValueKind::Literal);
    if (v == nullptr) return nullptr;
    v->bytes = bytes;
    v->len   = len;
    return v;
}

JsonValue* make_object(Arena* arena) noexcept {
    return alloc_value(arena, ValueKind::Object);
}

JsonValue* make_array(Arena* arena) noexcept {
    return alloc_value(arena, ValueKind::Array);
}

// ===========================================================================
// Container mutation.
// ===========================================================================
bool object_add(Arena* arena, JsonValue* obj,
                const char* key, int32_t key_len, JsonValue* val) noexcept {
    if (arena == nullptr || obj == nullptr) return false;
    if (obj->kind != ValueKind::Object) return false;
    if (key_len < 0 || (key == nullptr && key_len != 0)) return false;
    char* key_copy = nullptr;
    if (key_len > 0) {
        key_copy = static_cast<char*>(
            arena->copy_into(key, static_cast<size_t>(key_len), 1));
        if (key_copy == nullptr) return false;
    }
    return container_append(arena, obj, key_copy, key_len, val);
}

bool array_add(Arena* arena, JsonValue* arr, JsonValue* val) noexcept {
    if (arena == nullptr || arr == nullptr) return false;
    if (arr->kind != ValueKind::Array) return false;
    return container_append(arena, arr, nullptr, 0, val);
}

JsonValue* object_get(const JsonValue* obj,
                      const char* key, int32_t key_len) noexcept {
    if (obj == nullptr || obj->kind != ValueKind::Object) return nullptr;
    if (key_len < 0 || (key == nullptr && key_len != 0)) return nullptr;
    for (const JsonNode* m = obj->head; m != nullptr; m = m->next) {
        if (m->key_len == key_len &&
            (key_len == 0 ||
             memcmp(m->key, key, static_cast<size_t>(key_len)) == 0)) {
            return m->value;
        }
    }
    return nullptr;
}

// ===========================================================================
// In-place scalar mutation.
// ===========================================================================
void set_string(JsonValue* v, const char* bytes, int32_t len) noexcept {
    assert(v != nullptr);
    assert(len >= 0);
    v->kind  = ValueKind::StringEscaped;
    v->bytes = bytes;
    v->len   = len;
}

void set_string_raw(JsonValue* v, const char* bytes, int32_t len) noexcept {
    assert(v != nullptr);
    assert(len >= 0);
    v->kind  = ValueKind::StringVerbatim;
    v->bytes = bytes;
    v->len   = len;
}

// ===========================================================================
// Tape -> value tree.
// ===========================================================================
bool value_from_index(const StructuralIndex* idx, Arena* arena,
                      JsonValue** out_root) noexcept {
    if (idx == nullptr || arena == nullptr || out_root == nullptr) return false;
    if (idx->tokens == nullptr || idx->token_count <= 0) return false;
    if (idx->src == nullptr) return false;
    int32_t cur = 0;
    JsonValue* root = nullptr;
    if (!build_from_tape(idx, &cur, arena, 0, &root)) return false;
    // The only token allowed to remain is the terminal End sentinel.
    if (cur >= idx->token_count) return false;
    if (idx->tokens[cur].type != TokenType::End) return false;
    *out_root = root;
    return true;
}

// ===========================================================================
// Serialise.
// ===========================================================================
bool serialize(const JsonValue* root, Arena* arena,
               const uint8_t** out_bytes, int32_t* out_len) noexcept {
    if (root == nullptr || arena == nullptr ||
        out_bytes == nullptr || out_len == nullptr) {
        return false;
    }
    // Pass 1: exact length.
    CountSink cs{0};
    if (!emit_value(root, cs, 0)) return false;
    if (cs.n < 0 || cs.n > INT32_MAX) return false;
    const int32_t total = static_cast<int32_t>(cs.n);

    // Single allocation (never zero-sized so the pointer is always valid).
    char* buf = arena->allocate_array<char>(total == 0 ? 1 : static_cast<size_t>(total));
    if (buf == nullptr) return false;

    // Pass 2: fill it. The length must match pass 1 exactly.
    WriteSink ws{buf, buf + total, true};
    if (!emit_value(root, ws, 0)) return false;
    assert(ws.ok && "write pass overran the counted length");
    assert(ws.p == buf + total && "write pass length != count pass length");
    if (!ws.ok || ws.p != buf + total) return false;

    *out_bytes = reinterpret_cast<const uint8_t*>(buf);
    *out_len   = total;
    return true;
}

bool serialize(const JsonValue* root, Arena* arena, const EmitOptions& opts,
               const uint8_t** out_bytes, int32_t* out_len) noexcept {
    if (root == nullptr || arena == nullptr ||
        out_bytes == nullptr || out_len == nullptr) {
        return false;
    }
    const bool pretty = (opts.indent >= 0);

    // Pass 1: exact length (also surfaces invalid-UTF-8-under-ensure_ascii /
    // depth-overflow failures BEFORE any allocation).
    CountSink cs{0};
    if (!emit_faithful(root, cs, opts, pretty, 0, 0)) return false;
    if (cs.n < 0 || cs.n > INT32_MAX) return false;
    const int32_t total = static_cast<int32_t>(cs.n);

    char* buf = arena->allocate_array<char>(total == 0 ? 1 : static_cast<size_t>(total));
    if (buf == nullptr) return false;

    // Pass 2: fill. Length must match pass 1 exactly.
    WriteSink ws{buf, buf + total, true};
    if (!emit_faithful(root, ws, opts, pretty, 0, 0)) return false;
    assert(ws.ok && "write pass overran the counted length");
    assert(ws.p == buf + total && "write pass length != count pass length");
    if (!ws.ok || ws.p != buf + total) return false;

    *out_bytes = reinterpret_cast<const uint8_t*>(buf);
    *out_len   = total;
    return true;
}

}  // namespace json
}  // namespace parse
}  // namespace bolt
