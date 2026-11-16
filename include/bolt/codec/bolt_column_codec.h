// bolt_column_codec.h — per-column codec chains (PROTOTYPE, not an MSEG page
// format yet; the bytes here are not stable).
//
// A column declares an ordered list of stages, ClickHouse CODEC(...) style:
//     "delta,for,zstd(3)"      "template,dict,for,zstd"      "front,lz4"
// Each stage accepts one stream kind. Encoding a stream applies the first
// remaining stage that accepts it and recurses on the children with the rest
// of the chain; a stream no remaining stage accepts is written plain. The
// output is self-describing (every node carries its stage id), so decode needs
// no chain.
//
//   stage      accepts          children
//   delta      ints             ints
//   for        ints             bytes (FastLanes blocks, ref + width each)
//   rle        ints             ints values, ints run lengths
//   dict       strings          sorted strings, ints codes  (skipped if NDV > n/2)
//   front      strings          ints prefix, ints suffix length, bytes
//   template   strings          strings templates, ints variables, strings variables
//   lz4 / zstd(level) / zstdself   bytes   (kept only if smaller)
//
// template(p): a row is cut at delimiters; a canonical decimal token becomes
// an int variable, any other token holding a digit a string variable, the rest
// stays in the template. p bit 0: only all-digit tokens are variables.
// p bit 1: . - _ @ + also delimit.

#pragma once

#include <algorithm>

#include "bolt/bolt_hash.h"
#include "bolt/codec/bolt_codec_stream.h"
#include "bolt/ingest/bolt_lz4_raw.h"
#include "bolt/ingest/bolt_zstd.h"
#include "bolt/ingest/bolt_zstd_dec.h"
#include "bolt/ingest/bolt_zstd_enc.h"
#include "bolt/join/bolt_swiss.h"

namespace bolt {
namespace codec {

enum class Stage : uint8_t {
    kPlain = 0, kDelta = 1, kFor = 2, kRle = 3, kDict = 4, kFront = 5,
    kTemplate = 6, kLz4 = 7, kZstd = 8, kZstdSelf = 9, kCount = 10,
};
struct StageSpec { Stage s = Stage::kPlain; uint32_t param = 0; };
inline constexpr uint32_t kMaxStages = 8;
struct Chain { StageSpec st[kMaxStages]; uint32_t n = 0; };

inline constexpr uint8_t kTplEsc = 0x10, kTplInt = 0x11, kTplStr = 0x12;
inline constexpr uint32_t kTplDigitsOnly = 1u, kTplMoreDelims = 2u;

// "a,b(3),c" -> Chain. "" / "none" is the empty chain. False on an unknown
// name or more than kMaxStages.
inline bool chain_parse(const char* text, Chain* out) noexcept {
    assert(text != nullptr && out != nullptr);
    static constexpr struct { const char* name; Stage s; uint32_t dflt; } kNames[] = {
        {"delta", Stage::kDelta, 0}, {"for", Stage::kFor, 0}, {"rle", Stage::kRle, 0},
        {"dict", Stage::kDict, 0}, {"front", Stage::kFront, 0}, {"template", Stage::kTemplate, 0},
        {"lz4", Stage::kLz4, 0}, {"zstd", Stage::kZstd, 3}, {"zstdself", Stage::kZstdSelf, 0},
    };
    out->n = 0;
    if (std::strcmp(text, "none") == 0) return true;
    for (const char* p = text; *p;) {
        const char* e = p;
        while (*e && *e != ',' && *e != '(') ++e;
        const size_t len = static_cast<size_t>(e - p);
        const StageSpec* hit = nullptr; StageSpec spec;
        for (const auto& k : kNames)
            if (std::strlen(k.name) == len && std::memcmp(k.name, p, len) == 0) {
                spec.s = k.s; spec.param = k.dflt; hit = &spec;
            }
        if (hit == nullptr || out->n == kMaxStages) return false;
        if (*e == '(') {
            uint32_t v = 0; ++e;
            if (*e < '0' || *e > '9') return false;
            while (*e >= '0' && *e <= '9') v = v * 10 + static_cast<uint32_t>(*e++ - '0');
            if (*e++ != ')') return false;
            spec.param = v;
        }
        out->st[out->n++] = spec;
        if (*e == ',') ++e; else if (*e) return false;
        p = e;
    }
    return true;
}

inline bool stage_accepts(Stage s, Kind k) noexcept {
    switch (s) {
        case Stage::kDelta: case Stage::kFor: case Stage::kRle: return k == Kind::kInts;
        case Stage::kDict: case Stage::kTemplate: return k == Kind::kStrs;
        case Stage::kFront: return k == Kind::kStrs || k == Kind::kStrsSorted;
        case Stage::kLz4: case Stage::kZstd: case Stage::kZstdSelf: return k == Kind::kBytes;
        default: return false;
    }
}

struct Ctx {
    Arena* a = nullptr;
    ingest::ZstdEncState* zenc = nullptr;
    void* zdec = nullptr;
};

template <class T> inline T* arr(Ctx& c, uint64_t n) noexcept {
    return static_cast<T*>(c.a->allocate((n ? n : 1) * sizeof(T), alignof(T) < 8 ? 8 : alignof(T)));
}

inline bool tpl_delim(uint8_t ch, uint32_t param) noexcept {
    switch (ch) {
        case ' ': case '\t': case '/': case '?': case '&': case '=': case ':': case ',':
        case ';': case '[': case ']': case '(': case ')': case '{': case '}': case '"':
        case '\'': case '<': case '>': case '|': return true;
        case '.': case '-': case '_': case '@': case '+': return (param & kTplMoreDelims) != 0;
        default: return false;
    }
}

// Builds a strings stream out of a byte Buf and an offsets Buf (uint32 each).
struct StrOut { Buf bytes, offs; uint64_t n = 0; };
inline void strout_init(StrOut& s, Arena* a) noexcept {
    s.bytes.a = a; s.offs.a = a; const uint32_t z = 0; buf_put(s.offs, &z, 4);
}
inline void strout_end(StrOut& s) noexcept {
    const uint32_t o = static_cast<uint32_t>(s.bytes.n); buf_put(s.offs, &o, 4); ++s.n;
}
inline Stream strout_stream(const StrOut& s, Kind k = Kind::kStrs) noexcept {
    return strs_stream(reinterpret_cast<const uint32_t*>(s.offs.p), s.bytes.p, s.n, k);
}
inline bool strout_ok(const StrOut& s) noexcept {
    return s.bytes.ok && s.offs.ok && s.bytes.n < kMaxChunkBytes;
}

// Byte classes of the template split: delimiter, digit, needs an escape.
inline constexpr uint8_t kClsDelim = 1, kClsDigit = 2, kClsEsc = 4;
struct TplClass { uint8_t c[256]; };
inline void tpl_class_init(TplClass& t, uint32_t param) noexcept {
    for (uint32_t b = 0; b < 256; ++b) {
        const uint8_t ch = static_cast<uint8_t>(b);
        t.c[b] = static_cast<uint8_t>((tpl_delim(ch, param) ? kClsDelim : 0) | (ch >= '0' && ch <= '9' ? kClsDigit : 0) |
                                      (ch >= kTplEsc && ch <= kTplStr ? kClsEsc : 0));
    }
}

inline void tpl_split(const Stream& in, uint32_t param, StrOut& tpl, Buf& ivars, StrOut& svars) noexcept {
    assert(in.kind == Kind::kStrs);
    TplClass cls; tpl_class_init(cls, param);
    if (!buf_reserve(tpl.bytes, in.nbytes * 2 + 8)) return;   // every byte at most escaped
    uint8_t* w = tpl.bytes.p + tpl.bytes.n;
    for (uint64_t r = 0; r < in.n; ++r) {
        const uint8_t* p = in.bytes + in.offs[r];
        const uint8_t* end = in.bytes + in.offs[r + 1];
        while (p < end) {
            while (p < end && (cls.c[*p] & kClsDelim)) *w++ = *p++;
            if (p == end) break;
            const uint8_t* t = p; uint32_t all = kClsDigit, any = 0;
            do { const uint8_t k = cls.c[*p++]; all &= k; any |= k; } while (p < end && !(cls.c[*p] & kClsDelim));
            const uint64_t len = static_cast<uint64_t>(p - t);
            if (all && len <= 18 && (len == 1 || t[0] != '0')) {
                int64_t v = 0;
                for (uint64_t i = 0; i < len; ++i) v = v * 10 + (t[i] - '0');
                *w++ = kTplInt; buf_put(ivars, &v, 8);
            } else if ((any & kClsDigit) && (all || !(param & kTplDigitsOnly))) {
                *w++ = kTplStr; buf_put(svars.bytes, t, len); strout_end(svars);
            } else if (!(any & kClsEsc)) {
                std::memcpy(w, t, len); w += len;
            } else {
                for (uint64_t i = 0; i < len; ++i) {
                    *w = kTplEsc; w += (cls.c[t[i]] & kClsEsc) != 0;
                    *w++ = t[i];
                }
            }
        }
        tpl.bytes.n = static_cast<uint64_t>(w - tpl.bytes.p);
        strout_end(tpl);
    }
}

inline bool tpl_join(const Stream& tpl, const Stream& iv, const Stream& sv, StrOut& out) noexcept {
    uint64_t ni = 0, ns = 0;
    for (uint64_t r = 0; r < tpl.n; ++r) {
        const uint8_t* p = tpl.bytes + tpl.offs[r];
        const uint8_t* end = tpl.bytes + tpl.offs[r + 1];
        while (p < end) {
            const uint64_t run = find_below(p, end, kTplStr + 1);
            buf_put(out.bytes, p, run); p += run;
            if (p == end) break;
            if (*p == kTplEsc) { if (++p == end) return false; buf_u8(out.bytes, *p); }
            else if (*p == kTplInt) {
                if (ni == iv.n || iv.ints[ni] < 0) return false;
                char tmp[20]; uint32_t k = 20; uint64_t v = static_cast<uint64_t>(iv.ints[ni++]);
                do { tmp[--k] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
                buf_put(out.bytes, tmp + k, 20 - k);
            } else if (*p == kTplStr) {
                if (ns == sv.n) return false;
                buf_put(out.bytes, sv.bytes + sv.offs[ns], sv.offs[ns + 1] - sv.offs[ns]); ++ns;
            } else buf_u8(out.bytes, *p);
            ++p;
        }
        strout_end(out);
    }
    return ni == iv.n && ns == sv.n && strout_ok(out);
}

inline int str_cmp(const Stream& s, uint32_t a, uint32_t b) noexcept {
    const uint32_t la = s.offs[a + 1] - s.offs[a], lb = s.offs[b + 1] - s.offs[b];
    const int c = std::memcmp(s.bytes + s.offs[a], s.bytes + s.offs[b], la < lb ? la : lb);
    return c ? c : (la < lb ? -1 : la > lb);
}

// Sorted dictionary + codes. Returns the distinct count, or n + 1 as soon as
// more than half the rows are distinct (the caller skips the stage). Rows are
// grouped through a hash table; only the distinct values are sorted.
inline uint64_t dict_build(Ctx& c, const Stream& in, StrOut& dict, int64_t* codes) noexcept {
    assert(in.kind == Kind::kStrs && codes != nullptr);
    const uint64_t limit = in.n / 2;
    SwissTable t;
    auto* first = arr<uint32_t>(c, limit + 1);   // first row of each distinct value
    auto* seen = arr<uint32_t>(c, in.n);         // first-seen id of each row
    if (first == nullptr || seen == nullptr || !SwissTable::create(&t, limit + 1, c.a)) { dict.bytes.ok = false; return 0; }
    uint32_t ndv = 0;
    for (uint64_t i = 0; i < in.n; ++i) {
        const uint8_t* p = in.bytes + in.offs[i];
        const uint32_t len = in.offs[i + 1] - in.offs[i];
        uint64_t free_h = 0;
        const int32_t g = gb_table_find(&t, hash_bytes(p, len), [&](uint32_t id) {
            const uint32_t r = first[id];
            return in.offs[r + 1] - in.offs[r] == len && std::memcmp(in.bytes + in.offs[r], p, len) == 0;
        }, &free_h);
        if (g >= 0) { seen[i] = static_cast<uint32_t>(g); continue; }
        if (ndv == limit) return in.n + 1;
        if (g == -2 || !t.insert(free_h, ndv)) { dict.bytes.ok = false; return 0; }
        first[ndv] = static_cast<uint32_t>(i); seen[i] = ndv++;
    }
    auto* order = arr<uint32_t>(c, ndv); auto* rank = arr<uint32_t>(c, ndv);
    if (order == nullptr || rank == nullptr) { dict.bytes.ok = false; return 0; }
    for (uint32_t k = 0; k < ndv; ++k) order[k] = k;
    std::sort(order, order + ndv, [&](uint32_t a, uint32_t b) { return str_cmp(in, first[a], first[b]) < 0; });
    for (uint32_t k = 0; k < ndv; ++k) {
        const uint32_t r = first[order[k]];
        rank[order[k]] = k;
        buf_put(dict.bytes, in.bytes + in.offs[r], in.offs[r + 1] - in.offs[r]); strout_end(dict);
    }
    for (uint64_t i = 0; i < in.n; ++i) codes[i] = rank[seen[i]];
    return ndv;
}

void encode_stream(Ctx& c, const Stream& s, const Chain& ch, uint32_t pos, Buf& out) noexcept;
bool decode_stream(Ctx& c, Rd& r, Stream* out) noexcept;

inline void put_head(Buf& out, Stage s, Kind k, uint64_t n) noexcept {
    buf_u8(out, static_cast<uint8_t>(s)); buf_u8(out, static_cast<uint8_t>(k)); buf_var(out, n);
}

inline void encode_plain(Ctx& c, const Stream& s, const Chain& ch, uint32_t pos, Buf& out) noexcept {
    put_head(out, Stage::kPlain, s.kind, s.n);
    if (s.kind == Kind::kBytes) { buf_put(out, s.bytes, s.n); return; }
    if (s.kind == Kind::kInts) {
        encode_stream(c, bytes_stream(reinterpret_cast<const uint8_t*>(s.ints), s.n * 8), ch, pos, out);
        return;
    }
    auto* lens = arr<int64_t>(c, s.n);
    if (lens == nullptr) { out.ok = false; return; }
    for (uint64_t i = 0; i < s.n; ++i) lens[i] = s.offs[i + 1] - s.offs[i];
    encode_stream(c, ints_stream(lens, s.n), ch, pos, out);
    encode_stream(c, bytes_stream(s.bytes, s.nbytes), ch, pos, out);
}

// False: the block did not shrink; the caller tries the next stage.
inline bool encode_block(Ctx& c, const Stream& s, StageSpec sp, Buf& out) noexcept {
    const uint64_t cap = sp.s == Stage::kLz4 ? ingest::lz4_raw_bound(s.n)
                       : sp.s == Stage::kZstd && ingest::zstd_available()
                             ? ingest::zstd_max_compressed_len(s.n) : ingest::zstd_enc_bound(s.n);
    auto* tmp = arr<uint8_t>(c, cap);
    if (tmp == nullptr || s.n == 0) return false;
    uint64_t len = cap; bool ok = false;
    if (sp.s == Stage::kLz4) ok = ingest::lz4_raw_compress(s.bytes, s.n, tmp, cap, &len);
    else if (sp.s == Stage::kZstd && ingest::zstd_available())
        ok = ingest::zstd_compress(s.bytes, s.n, tmp, &len, static_cast<int>(sp.param)) == 0;
    else {
        if (c.zenc == nullptr) {
            const uint64_t sz = ingest::zstd_enc_state_size();
            void* mem = c.a->allocate(sz, 64);
            c.zenc = mem ? ingest::zstd_enc_state_init(mem, sz) : nullptr;
        }
        ok = c.zenc && ingest::zstd_compress_self(s.bytes, s.n, tmp, cap, &len, c.zenc);
    }
    if (!ok || len >= s.n) return false;
    put_head(out, sp.s, Kind::kBytes, s.n); buf_var(out, len); buf_put(out, tmp, len);
    return true;
}

inline void encode_stream(Ctx& c, const Stream& s, const Chain& ch, uint32_t pos, Buf& out) noexcept {
    assert(c.a != nullptr && s.n <= kMaxChunkBytes);
    const uint32_t start = pos;   // a stage this kind skips may still take a plain child
    while (pos < ch.n && !stage_accepts(ch.st[pos].s, s.kind)) ++pos;
    if (pos == ch.n) { encode_plain(c, s, ch, start, out); return; }
    const StageSpec sp = ch.st[pos];
    const uint32_t next = pos + 1;
    switch (sp.s) {
        case Stage::kDelta: {
            auto* d = arr<int64_t>(c, s.n);
            if (d == nullptr) break;
            delta_encode(s.ints, s.n, d);
            put_head(out, sp.s, s.kind, s.n);
            const int64_t base = s.n ? d[0] : 0;   // kept out of the deltas: it would widen block 0
            if (s.n) d[0] = 0;
            buf_put(out, &base, 8);
            encode_stream(c, ints_stream(d, s.n), ch, next, out);
            return;
        }
        case Stage::kFor: {
            Buf t; t.a = c.a; for_encode(s.ints, s.n, t);
            if (!t.ok) break;
            put_head(out, sp.s, s.kind, s.n);
            encode_stream(c, bytes_stream(t.p, t.n), ch, next, out);
            return;
        }
        case Stage::kRle: {
            auto* v = arr<int64_t>(c, s.n); auto* l = arr<int64_t>(c, s.n + 1);
            if (v == nullptr || l == nullptr) break;
            const uint64_t runs = rle_encode(s.ints, s.n, v, l);
            if (runs * 2 > s.n) { encode_stream(c, s, ch, next, out); return; }
            put_head(out, sp.s, s.kind, s.n);
            encode_stream(c, ints_stream(v, runs), ch, next, out);
            encode_stream(c, ints_stream(l, runs), ch, next, out);
            return;
        }
        case Stage::kDict: {
            StrOut d; strout_init(d, c.a);
            auto* codes = arr<int64_t>(c, s.n);
            if (codes == nullptr) break;
            const uint64_t ndv = dict_build(c, s, d, codes);
            if (!strout_ok(d)) break;
            if (ndv * 2 > s.n) { encode_stream(c, s, ch, next, out); return; }
            put_head(out, sp.s, s.kind, s.n);
            encode_stream(c, strout_stream(d, Kind::kStrsSorted), ch, next, out);
            encode_stream(c, ints_stream(codes, s.n), ch, next, out);
            return;
        }
        case Stage::kFront: {
            auto* pre = arr<int64_t>(c, s.n); auto* suf = arr<int64_t>(c, s.n);
            Buf t; t.a = c.a;
            if (pre == nullptr || suf == nullptr || !buf_reserve(t, s.nbytes)) break;
            for (uint64_t i = 0; i < s.n; ++i) {
                const uint8_t* cur = s.bytes + s.offs[i];
                const uint32_t len = s.offs[i + 1] - s.offs[i];
                const uint32_t pl = i ? s.offs[i] - s.offs[i - 1] : 0;
                const uint32_t k = common_prefix(cur, cur - pl, len < pl ? len : pl);
                pre[i] = k; suf[i] = len - k;
                std::memcpy(t.p + t.n, cur + k, len - k); t.n += len - k;
            }
            if (!t.ok) break;
            put_head(out, sp.s, s.kind, s.n);
            encode_stream(c, ints_stream(pre, s.n), ch, next, out);
            encode_stream(c, ints_stream(suf, s.n), ch, next, out);
            encode_stream(c, bytes_stream(t.p, t.n), ch, next, out);
            return;
        }
        case Stage::kTemplate: {
            StrOut tpl, sv; Buf iv; iv.a = c.a;
            strout_init(tpl, c.a); strout_init(sv, c.a);
            tpl_split(s, sp.param, tpl, iv, sv);
            if (!strout_ok(tpl) || !strout_ok(sv) || !iv.ok) break;
            put_head(out, sp.s, s.kind, s.n);
            encode_stream(c, strout_stream(tpl), ch, next, out);
            encode_stream(c, ints_stream(reinterpret_cast<const int64_t*>(iv.p), iv.n / 8), ch, next, out);
            encode_stream(c, strout_stream(sv), ch, next, out);
            return;
        }
        default:
            if (!encode_block(c, s, sp, out)) encode_stream(c, s, ch, next, out);
            return;
    }
    out.ok = false;
}

inline bool decode_child(Ctx& c, Rd& r, Kind want, Stream* out) noexcept {
    if (!decode_stream(c, r, out)) return false;
    if (want == Kind::kStrs) return out->kind == Kind::kStrs || out->kind == Kind::kStrsSorted;
    return out->kind == want;
}

inline bool finish_strs(Ctx& c, const int64_t* lens, uint64_t n, const uint8_t* bytes,
                        uint64_t nbytes, Kind k, Stream* out) noexcept {
    auto* offs = arr<uint32_t>(c, n + 1);
    if (offs == nullptr) return false;
    uint64_t at = 0; offs[0] = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (lens[i] < 0 || static_cast<uint64_t>(lens[i]) > nbytes - at) return false;
        at += static_cast<uint64_t>(lens[i]); offs[i + 1] = static_cast<uint32_t>(at);
    }
    *out = strs_stream(offs, bytes, n, k);
    return at == nbytes;
}

inline bool decode_stream(Ctx& c, Rd& r, Stream* out) noexcept {
    assert(c.a != nullptr && out != nullptr);
    const uint8_t tag = rd_u8(r);
    const Kind kind = static_cast<Kind>(rd_u8(r));
    const uint64_t n = rd_var(r);
    if (!r.ok || tag >= static_cast<uint8_t>(Stage::kCount) || n > kMaxChunkBytes) return false;
    if (kind != Kind::kBytes && n > kMaxRows) return false;
    Stream a, b, d;
    switch (static_cast<Stage>(tag)) {
        case Stage::kPlain: {
            if (kind == Kind::kBytes) {
                const uint8_t* p = rd_bytes(r, n);
                *out = bytes_stream(p, n); return r.ok;
            }
            if (kind == Kind::kInts) {
                if (!decode_child(c, r, Kind::kBytes, &a) || a.n != n * 8) return false;
                auto* v = arr<int64_t>(c, n);
                if (v == nullptr) return false;
                if (n) std::memcpy(v, a.bytes, n * 8);
                *out = ints_stream(v, n); return true;
            }
            if (!decode_child(c, r, Kind::kInts, &a) || a.n != n) return false;
            if (!decode_child(c, r, Kind::kBytes, &b) || b.n >= kMaxChunkBytes) return false;
            return finish_strs(c, a.ints, n, b.bytes, b.n, kind, out);
        }
        case Stage::kDelta: {
            const uint8_t* bp = rd_bytes(r, 8);
            if (bp == nullptr || !decode_child(c, r, Kind::kInts, &a) || a.n != n) return false;
            auto* v = arr<int64_t>(c, n);
            if (v == nullptr) return false;
            uint64_t base; std::memcpy(&base, bp, 8);
            delta_decode(a.ints, n, v);
            for (uint64_t i = 0; i < n; ++i) v[i] = static_cast<int64_t>(static_cast<uint64_t>(v[i]) + base);
            *out = ints_stream(v, n); return true;
        }
        case Stage::kFor: {
            if (!decode_child(c, r, Kind::kBytes, &a)) return false;
            auto* v = arr<int64_t>(c, n);
            Rd rr; rr.p = a.bytes; rr.end = a.bytes + a.n;
            if (v == nullptr || !for_decode(rr, n, v) || rr.p != rr.end) return false;
            *out = ints_stream(v, n); return true;
        }
        case Stage::kRle: {
            if (!decode_child(c, r, Kind::kInts, &a) || !decode_child(c, r, Kind::kInts, &b) || a.n != b.n) return false;
            auto* v = arr<int64_t>(c, n);
            if (v == nullptr) return false;
            uint64_t at = 0;
            for (uint64_t i = 0; i < a.n; ++i) {
                if (b.ints[i] <= 0 || static_cast<uint64_t>(b.ints[i]) > n - at) return false;
                for (int64_t k = 0; k < b.ints[i]; ++k) v[at++] = a.ints[i];
            }
            *out = ints_stream(v, n); return at == n;
        }
        case Stage::kDict: {
            if (!decode_child(c, r, Kind::kStrs, &a) || !decode_child(c, r, Kind::kInts, &b) || b.n != n) return false;
            auto* offs = arr<uint32_t>(c, n + 1);
            if (offs == nullptr) return false;
            uint64_t total = 0, bad = 0;
            for (uint64_t i = 0; i < n; ++i) {   // sizes first, no branch on the data
                const uint64_t code = static_cast<uint64_t>(b.ints[i]);
                bad |= code >= a.n;
                const uint64_t k = code < a.n ? code : 0;
                offs[i] = static_cast<uint32_t>(total);
                total += a.n ? a.offs[k + 1] - a.offs[k] : 0;
            }
            if (bad || total >= kMaxChunkBytes) return false;
            offs[n] = static_cast<uint32_t>(total);
            auto* bytes = arr<uint8_t>(c, total);
            if (bytes == nullptr) return false;
            for (uint64_t i = 0; i < n; ++i)
                std::memcpy(bytes + offs[i], a.bytes + a.offs[b.ints[i]], offs[i + 1] - offs[i]);
            *out = strs_stream(offs, bytes, n, kind); return true;
        }
        case Stage::kFront: {
            if (!decode_child(c, r, Kind::kInts, &a) || !decode_child(c, r, Kind::kInts, &b) ||
                !decode_child(c, r, Kind::kBytes, &d) || a.n != n || b.n != n) return false;
            StrOut o; strout_init(o, c.a);
            uint64_t at = 0, prev_off = 0, prev_len = 0;
            for (uint64_t i = 0; i < n; ++i) {
                const uint64_t pre = static_cast<uint64_t>(a.ints[i]), suf = static_cast<uint64_t>(b.ints[i]);
                if (pre > prev_len || suf > d.n - at || !buf_reserve(o.bytes, pre + suf)) return false;
                const uint64_t off = o.bytes.n;
                std::memmove(o.bytes.p + off, o.bytes.p + prev_off, pre);
                std::memcpy(o.bytes.p + off + pre, d.bytes + at, suf);
                o.bytes.n += pre + suf; at += suf; prev_off = off; prev_len = pre + suf;
                strout_end(o);
            }
            *out = strout_stream(o, kind); return at == d.n && strout_ok(o);
        }
        case Stage::kTemplate: {
            if (!decode_child(c, r, Kind::kStrs, &a) || !decode_child(c, r, Kind::kInts, &b) ||
                !decode_child(c, r, Kind::kStrs, &d) || a.n != n) return false;
            StrOut o; strout_init(o, c.a);
            if (!tpl_join(a, b, d, o)) return false;
            *out = strout_stream(o, kind); return true;
        }
        default: {
            const uint64_t len = rd_var(r);
            const uint8_t* src = rd_bytes(r, len);
            auto* dst = arr<uint8_t>(c, n);
            if (!r.ok || dst == nullptr || kind != Kind::kBytes) return false;
            bool ok = false;
            if (tag == static_cast<uint8_t>(Stage::kLz4)) ok = ingest::lz4_raw_decompress(src, len, dst, n);
            else if (tag == static_cast<uint8_t>(Stage::kZstd) && ingest::zstd_available())
                ok = ingest::zstd_decompress(src, len, dst, n) == 0;
            else {
                const uint64_t sz = ingest::zstd_scratch_size();
                if (c.zdec == nullptr) c.zdec = c.a->allocate(sz, 64);
                uint64_t got = 0;
                ok = c.zdec && ingest::zstd_decode_raw(src, len, dst, n, &got, c.zdec, sz) == 0 && got == n;
            }
            *out = bytes_stream(dst, n); return ok;
        }
    }
}

// One column chunk through a chain. out->a must be set.
inline bool encode_column(const Stream& in, const Chain& chain, Arena& a, Buf* out) noexcept {
    assert(out != nullptr && out->a != nullptr);
    Ctx c; c.a = &a;
    encode_stream(c, in, chain, 0, *out);
    return out->ok;
}

// The result points into `a` (or into [p, p + len) for a plain byte stream).
inline bool decode_column(const uint8_t* p, uint64_t len, Arena& a, Stream* out) noexcept {
    assert(p != nullptr && out != nullptr);
    Ctx c; c.a = &a;
    Rd r; r.p = p; r.end = p + len;
    return decode_stream(c, r, out) && r.ok && r.p == r.end;
}

}  // namespace codec
}  // namespace bolt
