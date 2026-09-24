// bolt/lakehouse/object_store_http.h — shared HTTP plumbing for the cloud
// ObjectStores (S3 / Azure Blob / GCS). Internal to src/lakehouse.
//
// Tiger Style: bounded buffers, ≥2 asserts/fn, no exceptions.

#pragma once

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "bolt/lakehouse/object_store.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace oshttp {

// Bounded string builder over a caller buffer; `ok` latches false on overflow.
struct Buf {
    char*    p;
    uint32_t cap;
    uint32_t n;
    bool     ok;
};

inline void buf_init(Buf* b, char* p, uint32_t cap) noexcept {
    assert(b != nullptr && p != nullptr);
    assert(cap > 0);
    b->p = p; b->cap = cap; b->n = 0; b->ok = true; p[0] = '\0';
}

inline void buf_raw(Buf* b, const char* s, uint32_t len) noexcept {
    assert(b != nullptr);
    assert(s != nullptr || len == 0);
    if (!b->ok || b->n + len + 1u > b->cap) { b->ok = false; return; }
    std::memcpy(b->p + b->n, s, len);
    b->n += len;
    b->p[b->n] = '\0';
}

inline void buf_str(Buf* b, const char* s) noexcept {
    assert(b != nullptr);
    buf_raw(b, s != nullptr ? s : "", s != nullptr
                ? static_cast<uint32_t>(std::strlen(s)) : 0u);
}

inline bool unreserved(unsigned char c) noexcept {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
           c == '~';
}

// RFC 3986 percent-encode `s`; '/' survives only when keep_slash.
inline void buf_pct(Buf* b, const char* s, bool keep_slash) noexcept {
    assert(b != nullptr);
    assert(s != nullptr);
    static const char kHex[] = "0123456789ABCDEF";
    for (uint32_t i = 0; s[i] != '\0' && i < kOsMaxKey; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (unreserved(c) || (keep_slash && c == '/')) {
            const char ch = static_cast<char>(c);
            buf_raw(b, &ch, 1u);
        } else {
            const char esc[3] = {'%', kHex[c >> 4], kHex[c & 0xF]};
            buf_raw(b, esc, 3u);
        }
    }
}

// Split "scheme://host[:port][/path]" into the "scheme://host[:port]" origin
// and the path (no trailing '/'). A bare "host[:port]" gets https://.
inline bool split_endpoint(const char* ep, Buf* origin, Buf* path) noexcept {
    assert(ep != nullptr);
    assert(origin != nullptr && path != nullptr);
    const char* rest = ep;
    if (std::strncmp(ep, "http://", 7) == 0) {
        buf_str(origin, "http://"); rest = ep + 7;
    } else if (std::strncmp(ep, "https://", 8) == 0) {
        buf_str(origin, "https://"); rest = ep + 8;
    } else {
        buf_str(origin, "https://");
    }
    const char* slash = std::strchr(rest, '/');
    const uint32_t hl = slash != nullptr
        ? static_cast<uint32_t>(slash - rest)
        : static_cast<uint32_t>(std::strlen(rest));
    if (hl == 0) return false;
    buf_raw(origin, rest, hl);
    if (slash != nullptr) {
        uint32_t pl = static_cast<uint32_t>(std::strlen(slash));
        while (pl > 0 && slash[pl - 1] == '/') --pl;
        buf_raw(path, slash, pl);
    }
    return origin->ok && path->ok;
}

// Host exactly as bolt::net sends it in the Host header (port stripped).
inline bool url_host(const char* url, char* host, uint32_t cap) noexcept {
    assert(url != nullptr && host != nullptr);
    assert(cap > 0);
    bool https = false;
    uint16_t port = 0;
    char path[net::kHttpMaxUrl];
    return net::http_url_parse(url, &https, host, cap, &port, path,
                               sizeof(path));
}

inline void utc_now(std::tm* out) noexcept {
    assert(out != nullptr);
    const std::time_t t = std::time(nullptr);
    assert(t > 0);
#ifdef _WIN32
    gmtime_s(out, &t);
#else
    gmtime_r(&t, out);
#endif
}

// Status → kOs* for the common verbs. 2xx ok, 404 not-found.
inline int map_status(int status) noexcept {
    assert(status >= 0);
    assert(status < 1000);
    if (status / 100 == 2) return kOsOk;
    if (status == 404) return kOsNotFound;
    return kOsIoError;
}

// Content between the next <tag>...</tag> at or after *pos, copied into out.
// Returns false when no further tag exists or the value overflows.
inline bool xml_next(const char* s, uint32_t len, uint32_t* pos,
                     const char* tag, char* out, uint32_t cap) noexcept {
    assert(s != nullptr && pos != nullptr);
    assert(tag != nullptr && out != nullptr && cap > 0);
    char open[64];
    char close[64];
    const int ol = std::snprintf(open, sizeof(open), "<%s>", tag);
    const int cl = std::snprintf(close, sizeof(close), "</%s>", tag);
    if (ol <= 0 || cl <= 0) return false;
    for (uint32_t i = *pos; i + static_cast<uint32_t>(ol) <= len; ++i) {
        if (std::memcmp(s + i, open, static_cast<size_t>(ol)) != 0) continue;
        const uint32_t v = i + static_cast<uint32_t>(ol);
        for (uint32_t j = v; j + static_cast<uint32_t>(cl) <= len; ++j) {
            if (std::memcmp(s + j, close, static_cast<size_t>(cl)) != 0) continue;
            const uint32_t n = j - v;
            if (n + 1u > cap) return false;
            std::memcpy(out, s + v, n);
            out[n] = '\0';
            *pos = j + static_cast<uint32_t>(cl);
            return true;
        }
        return false;
    }
    return false;
}

// Decode the five predefined XML entities in place.
inline void xml_unescape(char* s) noexcept {
    assert(s != nullptr);
    struct Ent { const char* e; char c; };
    static const Ent kEnts[] = {{"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'},
                                {"&quot;", '"'}, {"&apos;", '\''}};
    uint32_t w = 0;
    for (uint32_t r = 0; s[r] != '\0' && r < kOsMaxKey * 8u;) {
        bool hit = false;
        if (s[r] == '&') {
            for (const Ent& e : kEnts) {
                const size_t el = std::strlen(e.e);
                if (std::strncmp(s + r, e.e, el) == 0) {
                    s[w++] = e.c; r += static_cast<uint32_t>(el); hit = true;
                    break;
                }
            }
        }
        if (!hit) s[w++] = s[r++];
    }
    s[w] = '\0';
}

}  // namespace oshttp
}  // namespace lakehouse
}  // namespace bolt
